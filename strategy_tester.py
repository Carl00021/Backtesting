"""
strategy_tester.py
Manual batch tester that ranks strategies, outputs CSV and a PDF chart-pack.
Rewritten to remove vectorbt dependency and support leverage multipliers.

Usage:
    python strategy_tester.py --ticker SPY --start 2020-01-01 --end 2025-06-25 --leverage 1.0,3.0,10.0 --thresholds 0.005,0.01,0.02 --stop-loss 0.05 --init-cash 1.0
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
COMM_PER_TRADE     = 0.0025   # 0.25% per trade, on notional change
CAPITAL_GAINS_TAX  = 0.20     # (not used here, but you can layer it in)
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
    symbols = [ticker, "^VIX", "^TNX"]
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
    price: pd.Series,
    signals: dict[str, pd.Series],
    leverage: float,
    init_cash: float = INIT_CASH_DEFAULT
) -> tuple[pd.Series, list[dict], pd.Series]:
    """
    Simple daily P&L simulator:
      - entries/exits at close of bar i
      - commission = COMM_PER_TRADE * |Δposition| * equity_before_trade
      - daily PnL = position_prev * equity_prev * daily_return
      - position expressed in 'x' leverage units
    Returns:
      - equity curve (pd.Series)
      - trade list with entry/exit dates + PnL + win flag
    """
    rets   = price.pct_change().fillna(0)
    dates  = price.index
    n      = len(price)

    equity   = pd.Series(index=dates, dtype=float)
    position = np.zeros(n, dtype=float)

    equity.iloc[0]   = init_cash
    position[0]      = 0.0

    trades = []
    open_trade = None

    # ---------- NEW: handle an entry on bar-0 -----------------
    if signals["entries"].iloc[0]:
        position[0] = leverage * signals["size_pct"].iloc[0]
        open_trade  = {
            "entry_date":  dates[0],
            "entry_price": price.iloc[0],
            "leverage":    leverage,
        }

    for i in range(1, n):
        prev_eq   = equity.iloc[i-1]
        prev_pos  = position[i-1]
        date      = dates[i]

        # 1) Determine desired position
        if signals["entries"].iloc[i]:
            desired_pos = leverage * signals["size_pct"].iloc[i]
        elif signals["exits"].iloc[i]:
            desired_pos = 0.0
        else:
            desired_pos = prev_pos

        # 2) Commission on position change
        change     = desired_pos - prev_pos
        commission = COMM_PER_TRADE * abs(change) * prev_eq

        # 3) Apply daily P&L on previous position
        pnl        = prev_pos * prev_eq * rets.iloc[i]
        eq_today   = prev_eq + pnl - commission

        equity.iloc[i] = eq_today
        position[i]    = desired_pos

        # 4) Record trades based on your entry/exit signals
        # — record trades using the entry_price series (threshold fill) or fallback to close
        entry_price_series = signals.get("entry_price", price)

        # Entry: fill at the threshold price (if provided) or at the close
        if signals["entries"].iloc[i] and open_trade is None:
            entry_price = entry_price_series.iloc[i]
            open_trade = {
                "entry_date":  date,
                "entry_price": entry_price,
                "leverage":    leverage
            }

        elif signals["exits"].iloc[i] and open_trade is not None:
            # Normal exit
            exit_px = price.iloc[i]
            pnl_px  = (exit_px - open_trade["entry_price"]) / open_trade["entry_price"]
            open_trade.update({
                "exit_date": date,
                "exit_price": exit_px,
                "pnl": pnl_px * leverage,
                "win": pnl_px > 0,
            })
            trades.append(open_trade)
            open_trade = None

    pos_series = pd.Series(position, index=price.index)
    return equity, trades, pos_series
    # wrap the raw numpy positions into a pd.Series
    pos_series = pd.Series(position, index=price.index)
    return equity, trades, pos_series

def add_metrics_table(pdf, metrics_df, title):
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
            row = [shorten(idx, width=28, placeholder="…")]
            for col in sub.columns:
                val = sub.loc[idx, col]
                # your existing fmt()
                if col in ("TotalReturn","MaxDD%","WinRate"):
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
        fs = max(6, min(10, 500 / max(sub.shape)))

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

                strat = Strat(**init_kwargs)
                signals = strat.generate_signals()
                signals["low"] = aux["low"]
                # — for BuyDipPct, set the fill to prev_close*(1-threshold)
                if name == "buy_dip_pct" and th is not None:
                    # th is your threshold (e.g. 0.01)
                    signals["entry_price"] = price.shift(1) * (1 - th)
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
        total_rt = eq.iloc[-1] / args.init_cash - 1
        calmar   = total_rt / abs(max_dd) if max_dd != 0 else np.nan

        trades_k = trade_books[key]            # ← correct list
        pos_k    = pos_dict[key]               # ← correct series

        wins     = sum(t["win"] for t in trades_k)
        win_rate = wins / len(trades_k) if trades_k else np.nan
        last_td  = trades_k[-1]["exit_date"] if trades_k else pd.NaT

        lev_str = key.split("_")[-1]           # e.g. "3.0x"
        lev_val = float(lev_str.rstrip("x"))   # → 3.0

        metrics.append({
            "Strategy":   key,
            "Leverage":   lev_val,
            "TotalReturn": total_rt,
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
            fontsize=12,
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
        strategies_y = y0_ax - 0.05
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
        ax.text(
            0.01, strategies_y - 0.02,
            "\n".join(strategy_lines),
            va="top",
            ha="left",
            fontsize=12,
            family="monospace"
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ───────────────────────────────────────────────────────────
        # MAP PAGE: two plots, color-coded by leverage + best-fit curve
        # ───────────────────────────────────────────────────────────
        # prep data as percentages
        x1 = metrics_df["MaxDD%"] * 100
        y1 = metrics_df["TotalReturn"] * 100

        x2 = metrics_df["WinRate"] * 100
        y2 = metrics_df["TotalReturn"] * 100

        # extract leverage, coerce non-numeric (dynamic) to NaN so we ignore them
        lev_raw = metrics_df.get("Leverage", pd.Series(1, index=metrics_df.index))
        lev = pd.to_numeric(lev_raw, errors="coerce")

        # build a palette for each static leverage level
        unique_levs = sorted(lev.dropna().unique())      # e.g. [1, 3, 10]
        cmap = plt.get_cmap("tab10")
        color_map = {v: cmap(i) for i, v in enumerate(unique_levs)}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11), sharex=False)

        # — Top: Total Return vs Max Drawdown —
        for v in unique_levs:
            mask = lev == v
            ax1.scatter(
                x1[mask], y1[mask],
                label=f"Lev {int(v)}×",
                alpha=0.7,
                color=color_map[v]
            )
        ax1.set_xlabel("Max Drawdown (%)")
        ax1.set_ylabel("Total Return (%)")
        ax1.set_title("Total Return vs Max Drawdown")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # overall 2nd-degree best-fit
        mask1 = np.isfinite(x1) & np.isfinite(y1)
        x1_clean = x1[mask1]
        y1_clean = y1[mask1]
        if x1_clean.size > 2:
            coeffs1 = np.polyfit(x1_clean, y1_clean, deg=2)
            xs1 = np.linspace(x1_clean.min(), x1_clean.max(), 200)
            ys1 = np.polyval(coeffs1, xs1)
            ax1.plot(xs1, ys1, label="Best-fit (deg=2)", linewidth=2)
            ax1.legend(title="Leverage")

        # — Bottom: Total Return vs Hit Rate —
        for v in unique_levs:
            mask = lev == v
            ax2.scatter(
                x2[mask], y2[mask],
                label=f"Lev {int(v)}×",
                alpha=0.7,
                color=color_map[v]
            )
        ax2.set_xlabel("Hit Rate (%)")
        ax2.set_ylabel("Total Return (%)")
        ax2.set_title("Total Return vs Hit Rate")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # overall 2nd-degree best-fit
        mask2 = np.isfinite(x2) & np.isfinite(y2)
        x2_clean = x2[mask2]
        y2_clean = y2[mask2]
        if x2_clean.size > 2:
            coeffs2 = np.polyfit(x2_clean, y2_clean, deg=2)
            xs2 = np.linspace(x2_clean.min(), x2_clean.max(), 200)
            ys2 = np.polyval(coeffs2, xs2)
            ax2.plot(xs2, ys2, label="Best-fit (deg=2)", linewidth=2)
            ax2.legend(title="Leverage")

        # save this as one page
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 1: top 10 by TotalReturn
        top10_ret = metrics_df.nlargest(20, "TotalReturn")
        add_metrics_table(pdf, top10_ret, "Top 10 Strategies by Total Return")

        # Page 2: top 10 by WinRate
        top10_win = metrics_df.nlargest(20, "WinRate")
        add_metrics_table(pdf, top10_win, "Top 10 Strategies by Win Rate")

        # Pages for each top strategy
        top3 = metrics_df["TotalReturn"].nlargest(3).index
        for key in top3:
            fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)

            # 1) Price + trade markers
            price.plot(ax=axes[0], title=f"{key} – Price & Trades", alpha=0.5)
            entries = [t["entry_date"] for t in trade_books[key]]
            exits   = [t["exit_date"]  for t in trade_books[key]]
            axes[0].scatter(entries, price.loc[entries], marker="^", label="Entry", color="green")
            axes[0].scatter(exits,   price.loc[exits],   marker="v", label="Exit",  color="red")
            axes[0].legend(loc="best")

            # 2) Equity curve
            equity_curves[key].plot(ax=axes[1], title=f"{key} – Equity Curve")
            axes[1].set_ylabel("Equity Value")

            # 3) Drawdown
            dd = equity_curves[key] / equity_curves[key].cummax() - 1
            dd.plot(ax=axes[2], title=f"{key} – Drawdown Curve")
            axes[2].set_ylabel("Drawdown")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Summary saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
