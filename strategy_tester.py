"""
strategy_tester.py
Manual batch tester that ranks strategies, outputs CSV and a PDF chart-pack.
Rewritten to remove vectorbt dependency and support leverage multipliers.

Usage:
    python strategy_tester.py --ticker SPY --start 2020-01-01 --end 2025-06-25 --leverage 1.0,3.0,10.0 --thresholds 0.005,0.01,0.02 --stop-loss 0.05
    python strategy_tester.py --ticker SPY --start 2020-01-01 --leverage  1,3,5 --thresholds 0.005,0.01,0.02 --stop-loss 0.05
"""

import argparse
import inspect
import warnings
from pathlib import Path
import datetime as dt

import yfinance as yf
import numpy as np
import pandas as pd
from fredapi import Fred
from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import shorten

from Strategies import STRATEGY_REGISTRY
from dotenv import load_dotenv

load_dotenv()

# -------------------- CONFIG -------------------------------------------------
COMM_PER_TRADE     = 0.0005   # 0.05% per trade, on notional change
FIXED_COMM         = 0     # $10 per side, per trade
CAPITAL_GAINS_TAX  = 0.25     # 25% on realized gains, applied annually
INIT_CASH_DEFAULT  = 1.0

fred = Fred()  # assumes FRED_API_KEY in env

# -------------------- HELPERS -------------------------------------------------
def load_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      • Open, High, Low, Close, Volume    for `ticker`
      • ^VIX, ^TNX closes
    indexed by Date.
    Works whether yf.download gives (ticker,field) or (field,ticker) in its MultiIndex.
    """
    symbols = [ticker, "^VIX", "^VIX3M","^TNX"]
    raw = yf.download(symbols, start=start, end=end,
                      auto_adjust=True, progress=False)

    # === 1) Extract the main ticker’s OHLCV ===
    if raw.columns.nlevels == 2:
        # determine which level holds the field names
        level0 = set(raw.columns.get_level_values(0))
        known_fields = {"Open","High","Low","Close","Adj Close","Volume"}

        if level0 <= known_fields:
            # Level-0 = field, Level-1 = ticker
            df_main = raw.xs(ticker, axis=1, level=1)
        else:
            # Level-0 = ticker, Level-1 = field
            df_main = raw[ticker]
    else:
        # single-level (only one symbol) — assume that’s your ticker
        df_main = raw.copy()

    # rename “Adj Close” → “Close” if needed
    if "Adj Close" in df_main.columns:
        df_main = df_main.rename(columns={"Adj Close": "Close"})

    # keep only what we need
    df_main = df_main[["Open","High","Low","Close","Volume"]].copy()

    # === 2) Extract all “Close” series for the universe ===
    if raw.columns.nlevels == 2:
        # pick out the Close columns regardless of level ordering
        # if fields are level-0, use .xs on level 0; otherwise level-1
        lvl0 = set(raw.columns.get_level_values(0))
        if lvl0 <= known_fields:
            closes = raw.xs("Close", axis=1, level=0)
        else:
            closes = raw.xs("Close", axis=1, level=1)
        closes.columns = [c.upper() for c in closes.columns]
    else:
        # no multi-index means no VIX/TNX — download them separately or error
        raise ValueError("Expected multi‐ticker download to include VIX & TNX")

    # === 3) Join them, dropping the main‐ticker duplicate Close ===
    # closes looks like { 'SPY':…, '^VIX':…, '^TNX':… }
    data = df_main.join(closes.drop(columns=ticker.upper()))
    data.index.name = "Date"
    return data

def load_macro_series() -> dict[str, pd.Series]:
    cpi    = fred.get_series("CPIAUCSL")
    unrate = fred.get_series("UNRATE")
    return {"CPIAUCSL": cpi, "UNRATE": unrate}

def simulate_portfolio(
    price:      pd.Series,
    signals:    dict[str, pd.Series],
    leverage:   float = 1.0,
    init_cash:  float = 1.0,
    comm_pct:   float = 0.0005,   # 0.05 % of notional on exit
    fixed_comm: float = 0.0,      # e.g. $10/exit
    cooldown_days: int = 3        # days to wait after an exit
) -> tuple[pd.Series, list[dict], pd.Series]:
    """
    Marks positions to market daily.                    │
    • Commission is only charged on exit.               │
    • Entry → Exit order lets same-bar signals close.   │
    Returns: equity curve, list-of-trades dicts, position % series.
    """
    dates   = price.index
    rets    = price.pct_change().fillna(0.0).to_numpy()

    equity   = pd.Series(np.nan, index=dates, dtype=float)
    position = pd.Series(0.0,  index=dates, dtype=float)
    trades   = []

    cash         = init_cash
    cur_pos      = 0.0            # fraction of equity long (+) / short (–)
    open_trade   = None           # None or dict
    next_allowed = dates[0]

    def _commission(notional: float) -> float:
        return comm_pct * notional + fixed_comm

    for i, date in enumerate(dates):

        # ------------------------------------------------------------------
        # 0) MTM on yesterday’s position (if any)
        # ------------------------------------------------------------------
        if i > 0 and cur_pos != 0.0:
            cash += cur_pos * cash * rets[i]

        # ------------------------------------------------------------------
        # 1) ENTRY  (runs *before* exit)
        # ------------------------------------------------------------------
        if (open_trade is None
                and signals["entries"].iloc[i]
                and date >= next_allowed):

            entry_px  = signals.get("entry_price", price).iloc[i]
            size_pct  = signals["size_pct"].iloc[i]          # 0-1 from strategy
            cur_pos   = leverage * size_pct
            notional  = cash * cur_pos

            open_trade = {
                "entry_date":  date,
                "entry_idx":   i,
                "entry_price": entry_px,
                "notional":    notional,
                "equity_at_entry": cash,
            }

        # ------------------------------------------------------------------
        # 2) EXIT
        # ------------------------------------------------------------------
        if open_trade and signals["exits"].iloc[i]:

            notional   = open_trade["notional"]
            price_exit = price.iloc[i]

            # realise intraday gain only if we opened *today*
            realised_gain = 0.0
            if i == open_trade["entry_idx"]:
                realised_gain = (price_exit / open_trade["entry_price"] - 1) * notional
                cash += realised_gain

            # exit commission
            cash -= _commission(notional)

            trades.append({
                "entry_date":  open_trade["entry_date"],
                "entry_price": open_trade["entry_price"],
                "exit_date":   date,
                "exit_price":  price_exit,
                "pnl":         cash - open_trade["equity_at_entry"],
                "win":         (cash - open_trade["equity_at_entry"]) > 0,
            })

            open_trade   = None
            cur_pos      = 0.0
            next_allowed = date + pd.Timedelta(days=cooldown_days)

        # ------------------------------------------------------------------
        # 3) Ledger
        # ------------------------------------------------------------------
        equity.iloc[i]   = cash
        position.iloc[i] = cur_pos

        # optional: stop-out on bankruptcy
        if cash <= 0:
            equity.iloc[i:]   = 0.0
            position.iloc[i:] = 0.0
            break

    return equity, trades, position

def add_metrics_table(pdf, metrics_df, title):
    # remove unwanted columns
    metrics_df = metrics_df.drop(columns=["Leverage", "TotalReturn"], errors="ignore")

    # Prepare the subset & orientation logic as before
    rows, cols = metrics_df.shape
    max_rows_per_page = 42
    pages = (rows - 1) // max_rows_per_page + 1

    for p in range(pages):
        sub = metrics_df.iloc[p*max_rows_per_page:(p+1)*max_rows_per_page]

        # Figure out portrait vs. landscape
        landscape = sub.shape[0] < sub.shape[1]
        fig_w, fig_h = (11, 8.5) if landscape else (8.5, 11)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # Build the text for each cell (shorten strategy names)
        col_labels = ["Strategy"] + sub.columns.tolist()
        rows_text  = []
        for idx in sub.index:
            row = [idx]
            for col in sub.columns:
                val = sub.loc[idx, col]
                # your existing fmt()
                if col in ("TotalReturn","MaxDD%","WinRate","CAGR"):
                    cell = f"{val*100:0.1f}%"
                elif col == "Position":
                    cell = f"{int(val)}"
                elif col == "LastTrade":
                    cell = val.strftime("%Y-%m-%d") if not pd.isna(val) else ""
                else:
                    cell = f"{val:0.2f}"
                row.append(cell)
            rows_text.append(row)

        # Compute max text length per column (including header)
        all_text = [col_labels] + rows_text
        max_lens = [max(len(r[i]) for r in all_text) for i in range(len(col_labels))]
        total_len = sum(max_lens)
        # Normalize to get colWidths summing to 1
        col_widths = [l/total_len for l in max_lens]

        # Auto font sizing
        fs = max(6, min(8, 300 / max(sub.shape)))

        # Draw the title at the top with a little padding
        ax.set_title(title, fontweight="bold", fontsize=fs+2, pad=20)

        # Place the table in the remaining area (below the title)
        tbl = ax.table(
            cellText   = rows_text,
            colLabels  = col_labels,
            colWidths  = col_widths,
            loc        = "upper center",
            cellLoc    = "center",
            colLoc     = "center",
            bbox       = [0.0, 0.0, 1.0, 0.95]   # [x, y, width, height]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fs)
        tbl.scale(1.0, 1.0)

        # tighten only inside the lower 80% so title isn't squashed
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

# -------------------- MAIN ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",    required=True)
    parser.add_argument("--start",     default="2000-01-01")
    parser.add_argument("--end",       default=None)
    parser.add_argument(
        "--leverage",
        default="1.0",
        help="Comma-separated leverage multipliers, e.g. 1.0,3.0,-1.0"
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated dip thresholds for BuyDipPct, e.g. 0.005,0.01,0.02"
    )
    parser.add_argument(
        "--stop-loss",
        dest="stop_loss",
        type=float,
        default=0.05,
        help="Fractional stop-loss (e.g. 0.05 = 5%)"
    )
    parser.add_argument(
        "--init-cash",
        dest="init_cash",
        type=float,
        default=INIT_CASH_DEFAULT
    )
    args = parser.parse_args()
    model_params = vars(args)

    # parse leverage
    levs = [float(x) for x in args.leverage.split(",")]

    # parse thresholds (empty → single None)
    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",")]
    else:
        thresholds = [None]

    # Load OHLCV + VIX/TNX closes
    data  = load_market_data(args.ticker, args.start, args.end)
    print("DEBUG — data.columns:", list(data.columns))
    # => ['Open','High','Low','Close','Volume','^VIX','^TNX']

    # Build price & aux
    price = data["Close"]
    aux = {
        "open":   data["Open"],
        "high":   data["High"],
        "low":    data["Low"],
        "volume": data["Volume"],
        "^VIX":   data["^VIX"],
        "^VIX3M":   data["^VIX3M"],
        "^TNX":   data["^TNX"],
        **load_macro_series(),
    }

    equity_curves = {}
    trade_books   = {}
    pos_dict      = {}

    for name, Strat in STRATEGY_REGISTRY.items():
        sig = inspect.signature(Strat.__init__)

        # if this strategy takes `threshold`, sweep them; else do it once with None
        if "threshold" in sig.parameters:
            threshold_values = thresholds
        else:
            threshold_values = [None]

        for th in threshold_values:
            for lev in levs:  # Run each strategy × each leverage
                # build kwargs based on what the Strat constructor wants
                init_kwargs = {"price": price, "aux": aux}
                if "stop_loss" in sig.parameters:
                    init_kwargs["stop_loss"] = args.stop_loss
                if th is not None:
                    init_kwargs["threshold"] = th
                if "max_leverage" in inspect.signature(Strat.__init__).parameters:
                    init_kwargs["max_leverage"] = lev

                strat = Strat(**init_kwargs)
                signals = strat.generate_signals()
                signals["low"] = aux["low"]
                # — for BuyDipPct, set the fill to prev_close*(1-threshold)
                if hasattr(strat, "threshold"):
                    prev_close = price.shift(1)
                    entry_price = price.copy()
                    mask = signals["entries"].astype(bool)
                    entry_price[mask] = prev_close[mask] * (1 - strat.threshold)
                    signals["entry_price"] = entry_price

                if "entry_price" not in signals:               # non-intraday models
                    signals["entry_price"] = aux["open"].shift(-1)   # next day’s open

                eq_curve, trades, pos_series = simulate_portfolio(
                price, signals, lev, init_cash=args.init_cash
                )

                # construct a key that only includes threshold if we used one
                parts = [name]
                if th is not None:
                    parts.append(f"th{th:.3f}")
                parts.append(f"{lev}x")
                key = "_".join(parts)
                equity_curves[key] = eq_curve
                trade_books[key]   = trades
                pos_dict[key]      = pos_series
                print(f">>> {key}: final equity = {eq_curve.iloc[-1]:.4f}, "f"{len(trades)} trades")
                
    # Build metrics table
    metrics = []
    for key, eq in equity_curves.items():
        daily = eq.pct_change().dropna()
        sharpe = (daily.mean() / daily.std() * np.sqrt(252)
                if daily.std() != 0 else np.nan)

        dd       = eq / eq.cummax() - 1
        max_dd   = dd.min()
        total_rt = eq.iloc[-1] / args.init_cash - 1.0
        days  = (eq.index[-1] - eq.index[0]).days
        years = days / 365.25
        cagr  = (eq.iloc[-1] / args.init_cash) ** (1/years) - 1

        calmar   = total_rt / abs(max_dd) if max_dd != 0 else np.nan

        trades_k = trade_books[key]            # ← correct list
        pos_k    = pos_dict[key]               # ← correct series

        wins     = sum(t["win"] for t in trades_k)
        win_rate = wins / len(trades_k) if trades_k else np.nan
        last_td  = trades_k[-1]["exit_date"] if trades_k else pd.NaT

        lev_str = key.split("_")[-1]           # e.g. "3.0x"
        lev_val = float(lev_str.rstrip("x"))   # → 3.0

        metrics.append({
            "Strategy":    key,
            "Leverage":    lev_val,
            "TotalReturn": total_rt,
            "CAGR":        cagr,
            "Sharpe":      sharpe,
            "Calmar":      calmar,
            "MaxDD%":      max_dd,
            "WinRate":     win_rate,
            "Position":    int(pos_k.iloc[-1] != 0),
            "LastTrade":   last_td,
        })

    metrics_df = pd.DataFrame(metrics).set_index("Strategy")

    # Save CSV
    outdir = Path("results") / args.ticker
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(outdir / "strategy_metrics.csv", float_format="%.4f")
    print("\n=== Strategy Metrics ===")
    print(metrics_df.to_string(float_format="%.4f"))

    # Generate PDF report for top 3 by TotalReturn
    pdf_path= outdir / f"strategy_report-{dt.datetime.today().date()}.pdf"
    with PdfPages(pdf_path) as pdf:
        # ───────────────────────────────────────────────────────────
        # PAGE 0: Parameters Summary
        # ───────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        # Title
        title_str = f"{args.ticker} - Back Test {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ax.text(0.5, 0.97, title_str, va="top", ha="center", fontsize=16, fontweight="bold")

        # “Parameters:” subtitle
        ax.text(0.01, 0.92, "Parameters:", va="top", ha="left", fontsize=14, fontweight="bold")

        # Parameters list, keep a handle for bbox measurement
        param_lines = [f"{name}: {value}" for name, value in model_params.items()]
        param_txt = "\n".join(param_lines)
        param_text_obj = ax.text(
            0.01, 0.90,
            param_txt,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace"
        )

        # Force a draw so we can measure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = param_text_obj.get_window_extent(renderer=renderer)

        # Convert bbox from display coords to axes coords
        # We grab the bottom‐left corner (x0,y0)
        x0_ax, y0_ax = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))

        # Position “Strategies:” 5% below that bottom
        strategies_y = y0_ax - 0.02
        strategies_y = max(strategies_y, 0.01)  # don’t go below the page

        # “Strategies:” subtitle
        ax.text(
            0.01, strategies_y,
            "Strategies:",
            va="top",
            ha="left",
            fontsize=14,
            fontweight="bold"
        )

        # List out your registry just below the subtitle
        strategy_lines = [f"- {s}" for s in STRATEGY_REGISTRY]
        max_per_col = 30

        # build 3 columns
        cols = [
            strategy_lines[0 : max_per_col],
            strategy_lines[max_per_col : 2 * max_per_col],
            strategy_lines[2 * max_per_col :     ]
        ]

        # x‐positions for each column (in axes fraction)
        xs = [0.01, 0.34, 0.67]
        y0 = strategies_y - 0.02

        for x, col in zip(xs, cols):
            if not col:
                continue
            ax.text(
                x, y0,
                "\n".join(col),
                va="top", ha="left",
                fontsize=10, family="monospace",
                transform=ax.transAxes
            )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ───────────────────────────────────────────────────────────
        # MAP PAGE: two plots, color‐coded by leverage + best‐fit curve (with capped outliers)
        # ───────────────────────────────────────────────────────────

        # prep data as percentages
        x1 = metrics_df["MaxDD%"] * 100
        y1 = metrics_df["CAGR"]  * 100

        # leverage mapping
        lev_raw     = metrics_df.get("Leverage", pd.Series(1, index=metrics_df.index))
        lev         = pd.to_numeric(lev_raw, errors="coerce")
        unique_levs = sorted(lev.dropna().unique())
        cmap        = plt.get_cmap("tab10")
        color_map   = {v: cmap(i) for i, v in enumerate(unique_levs)}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))

        # — Top: CAGR vs Max Drawdown —
        # mask NaNs
        mask = np.isfinite(x1) & np.isfinite(y1)
        x = x1[mask]
        y = y1[mask]
        lev = metrics_df["Leverage"][mask]

        # compute 95th‐pct cap and best‐fit curve
        cap   = np.percentile(y, 99)
        coeff = np.polyfit(x, y, 2)
        xs    = np.linspace(x.min(), x.max(), 200)
        ys    = np.polyval(coeff, xs)

        # decide the y‐axis top (bigger of cap or parabola peak)
        peak    = ys.max()
        y_top   = max(cap, peak) * 1.05

        # split inliers vs outliers
        is_out = y > cap
        in_x   = x[~is_out]
        in_y   = y[~is_out]
        out_x  = x[is_out]

        # — plot inliers at their true y —
        for v in unique_levs:
            idx = (lev == v) & (~is_out)
            ax1.scatter(
                in_x[idx], in_y[idx],
                label=f"Lev {int(v)}×",
                color=color_map[v],
                alpha=0.7
            )

        # — plot outlier circles at y_top —
        for v in unique_levs:
            idx = (lev == v) & (is_out)
            ax1.scatter(
                out_x[idx], 
                [y_top]*idx.sum(),
                label=None,  # already in legend via the inliers
                color=color_map[v],
                alpha=0.7
            )

        # — draw best‐fit curve —
        ax1.plot(xs, ys, label="Best-fit (deg=2)", linewidth=2)

        # — draw X markers at y_top for outliers and label them —
        if is_out.any():
            ax1.scatter(
                out_x, [y_top]*is_out.sum(),
                marker="X", s=80,
                facecolors="none", edgecolors="black",
                label="Outliers",
                zorder=10
            )
            for xi, yi in zip(out_x, y[is_out]):
                ax1.text(
                    xi, y_top * 0.97, f"{yi:.1f}%",
                    ha="center", va="top", fontsize=8
                )

        ax1.set_ylim(bottom=y.min(), top=y_top)
        ax1.set_xlabel("Max Drawdown (%)")
        ax1.set_ylabel("CAGR (%)")
        ax1.set_title("CAGR vs Max Drawdown")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(title="Leverage")

        # — Bottom: CAGR vs Hit Rate — (apply the same pattern)
        x2 = metrics_df["WinRate"] * 100
        y2 = metrics_df["CAGR"]  * 100   # <— use the new CAGR column

        # mask NaNs
        mask = np.isfinite(x2) & np.isfinite(y2)
        x = x2[mask]
        y = y2[mask]
        lev = metrics_df["Leverage"][mask]

        # compute 95th‐pct cap and best‐fit curve
        cap   = np.percentile(y, 99)
        coeff = np.polyfit(x, y, 2)
        xs    = np.linspace(x.min(), x.max(), 200)
        ys    = np.polyval(coeff, xs)

        # decide the y‐axis top (bigger of cap or parabola peak)
        peak    = ys.max()
        y_top   = max(cap, peak) * 1.1

        # split inliers vs outliers
        is_out = y > cap
        in_x   = x[~is_out]
        in_y   = y[~is_out]
        out_x  = x[is_out]

        # — plot inliers at their true y —
        for v in unique_levs:
            idx = (lev == v) & (~is_out)
            ax2.scatter(
                in_x[idx], in_y[idx],
                label=f"Lev {int(v)}×",
                color=color_map[v],
                alpha=0.7
            )

        # — plot outlier circles at y_top —
        for v in unique_levs:
            idx = (lev == v) & (is_out)
            ax2.scatter(
                out_x[idx], 
                [y_top]*idx.sum(),
                label=None,  # already in legend via the inliers
                color=color_map[v],
                alpha=0.7
            )

        # — draw best‐fit curve —
        ax2.plot(xs, ys, label="Best-fit (deg=2)", linewidth=2)

        # — draw X markers at y_top for outliers and label them —
        if is_out.any():
            ax2.scatter(
                out_x, [y_top]*is_out.sum(),
                marker="X", s=80,
                facecolors="none", edgecolors="black",
                label="Outliers",
                zorder=10
            )
            for xi, yi in zip(out_x, y[is_out]):
                ax2.text(
                    xi, y_top * 0.97, f"{yi:.1f}%",
                    ha="center", va="top", fontsize=8
                )

        ax2.set_ylim(bottom=y.min(), top=y_top)
        ax2.set_xlabel("Win Rate (%)")
        ax2.set_ylabel("CAGR (%)")
        ax2.set_title("CAGR vs Win Rate")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(title="Leverage")

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 1: top 10 by TotalReturn
        top10_ret = metrics_df.nlargest(30, "TotalReturn")
        add_metrics_table(pdf, top10_ret, "Top 30 Strategies by Total Return")

        # Page 2: top 10 by WinRate
        top10_win = metrics_df.nlargest(30, "WinRate")
        add_metrics_table(pdf, top10_win, "Top 30 Strategies by Win Rate")

        # Pages for each top strategy
        top3 = metrics_df["TotalReturn"].nlargest(3).index
        for key in top3:
            # 0) build a win‐flag series and rolling rates ----
            trades = trade_books[key]
            if trades:
                df_tr = pd.DataFrame(trades)
                # ensure exit_date is datetime
                df_tr["exit_date"] = pd.to_datetime(df_tr["exit_date"])
                # make exit_date the index and sort
                df_tr.set_index("exit_date", inplace=True)
                df_tr.sort_index(inplace=True)

                df_tr["win_int"] = df_tr["win"].astype(int)

                win_rates = pd.DataFrame({
                    "3-trade"    : df_tr["win_int"].rolling(3).mean(),
                    "5-trade"    : df_tr["win_int"].rolling(5).mean(),
                    "cumulative" : df_tr["win_int"].expanding().mean(),
                })

                # now reindex onto the full datetime index
                win_rates = (
                    win_rates
                    .reindex(price.index)   # align to your price dates
                    .ffill()                # carry last value forward
                    .fillna(0.0)            # initial periods → zero
                )
            else:
                win_rates = pd.DataFrame(
                    0.0,
                    index=price.index,
                    columns=["3-trade","5-trade","cumulative"]
                )

            # --- 1) make a 4×1 shared-x figure ---
            fig, axes = plt.subplots(4, 1, figsize=(8.5, 11),gridspec_kw={"height_ratios":[2,1,1,1]}, sharex=True)
            
            # 2) Price + trades
            price.plot(ax=axes[0],title=f"{key} – Price & Trades",alpha=0.75,zorder=2)
            sma200 = price.rolling(window=200, min_periods=1).mean()
            axes[0].plot(price.index,sma200,label="200d SMA",alpha=0.5,linewidth=1,zorder=1)
            pos = pos_dict[key]              # 0 → flat, ±… → in trade
            entries = pos[(pos.shift(1)==0) & (pos!=0)].index
            exits   = pos[(pos.shift(1)!=0) & (pos==0)].index
            axes[0].scatter(entries,price.loc[entries],marker="^",color="green",label="Entry",s=50,zorder=5)
            axes[0].scatter(exits,price.loc[exits],marker="v",color="red",label="Exit",s=50,zorder=5)
            axes[0].legend(loc="best")

            # 3) Rolling win-rates
            axes[1].plot(win_rates.index, win_rates["3-trade"],    label="3-trade")
            axes[1].plot(win_rates.index, win_rates["5-trade"],    label="5-trade")
            axes[1].plot(win_rates.index, win_rates["cumulative"], label="cumulative")
            axes[1].set_ylabel("Win Rate")
            axes[1].set_ylim(0,1)
            axes[1].legend(loc="best")
            axes[1].set_title(f"{key} – Rolling Win Rates")

            # 4) Equity + leverage shading
            axes[2].fill_between(pos.index, 0, pos, step='post', alpha=0.3, label='Leverage')
            ax2 = axes[2].twinx()
            equity_curves[key].plot(ax=ax2, label='Equity', title=f"{key} – Equity & Leverage")
            axes[2].set_yticks([0, 1, pos.max()], labels=['0×','1×',f'{int(pos.max())}×']);  axes[2].set_ylabel('Leverage')
            ax2.set_ylabel('Equity Value')
            axes[2].legend(loc='upper left');  ax2.legend(loc='upper right')

            # 5) Drawdown
            dd = equity_curves[key] / equity_curves[key].cummax() - 1
            dd.plot(ax=axes[3], title=f"{key} – Drawdown Curve")
            axes[3].set_ylabel("Drawdown")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Summary saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
