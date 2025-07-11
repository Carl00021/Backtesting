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

# ---------------------------------------------------------------------------
# Portfolio simulator — handles both “always-on” (buy-and-hold) strategies
# and same-bar round-trip Buy-the-Dip (BTD_*) variants.
#
# Key fixes vs. earlier versions
#   • realised P/L from every exit (same-day or multi-day) is booked into equity
#   • commission charged on each executed side and netted out of win/loss
#   • win flag = True only when a trade beats its own commission
#   • 1-day cooldown for dip strategies to avoid back-to-back entries
# ---------------------------------------------------------------------------
def simulate_portfolio(
    price: pd.Series,
    signals: dict[str, pd.Series],
    leverage: float,
    init_cash: float = INIT_CASH_DEFAULT
) -> tuple[pd.Series, list[dict], pd.Series]:

    dates     = price.index
    rets      = price.pct_change().fillna(0.0)
    n         = len(price)

    # Identify dip strategies by the presence of an entry_price series
    is_dip_strategy = "entry_price" in signals
    last_trade_date = None

    equity   = pd.Series(np.nan, index=dates, dtype=float)
    position = np.zeros(n,          dtype=float)
    trades   = []

    # ----------------------------------------------------------------------
    # Day-0: open a position if the strategy fires immediately (buy-and-hold)
    # ----------------------------------------------------------------------
    equity.iloc[0]  = init_cash
    open_trade      = None

    if signals["entries"].iloc[0]:
        entry_px   = signals.get("entry_price", price).iloc[0]
        size_pct   = signals["size_pct"].iloc[0]
        notional   = init_cash * leverage * size_pct
        commission = COMM_PER_TRADE * notional            # one side

        open_trade = {
            "entry_date":    dates[0],
            "entry_price":   entry_px,
            "leverage":      leverage,
            "size_pct":      size_pct,
            "entry_notional": notional,                   # store for exit
        }

        equity.iloc[0] -= commission                      # pay entry fee
        position[0]     = leverage * size_pct
        last_trade_date = dates[0]

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------
    for i in range(1, n):
        prev_eq   = equity.iloc[i - 1]
        prev_pos  = position[i - 1]
        date      = dates[i]

        # -----------------------------
        # Optional 1-day dip cooldown
        # -----------------------------
        if is_dip_strategy and signals["entries"].iloc[i]:
            if last_trade_date == dates[i - 1]:
                signals["entries"].iloc[i] = False   # veto today’s entry

        entry_exec = False
        exit_exec  = False
        desired_pos = prev_pos

        # -------------------------------------------------------------
        # A) SAME-BAR round-trip (dip strategies only)
        # -------------------------------------------------------------
        if (is_dip_strategy and
            signals["entries"].iloc[i] and
            signals["exits"].iloc[i] and
            open_trade is None):

            # --- entry side ---
            entry_px   = signals.get("entry_price", price).iloc[i]
            size_pct   = signals["size_pct"].iloc[i]
            notional   = prev_eq * leverage * size_pct
            comm_entry = COMM_PER_TRADE * notional

            # --- exit side ---
            exit_px    = price.iloc[i]
            pnl_raw    = (exit_px / entry_px) - 1.0
            realised   = notional * pnl_raw
            comm_exit  = COMM_PER_TRADE * notional
            net_pnl    = realised - comm_entry - comm_exit

            # book trade
            trades.append({
                "entry_date":  date,
                "entry_price": entry_px,
                "exit_date":   date,
                "exit_price":  exit_px,
                "pnl":         net_pnl,
                "win":         net_pnl > 0,
            })

            # update ledger
            equity.iloc[i] = prev_eq + net_pnl
            position[i]    = 0.0
            last_trade_date = date
            continue  # nothing left to do for this bar

        # -------------------------------------------------------------
        # B) NORMAL multi-bar exit
        # -------------------------------------------------------------
        if signals["exits"].iloc[i] and open_trade is not None:
            exit_exec  = True
            desired_pos = 0.0

            exit_px    = price.iloc[i]
            notional   = open_trade["entry_notional"]
            pnl_raw    = (exit_px / open_trade["entry_price"]) - 1.0
            realised   = notional * pnl_raw
            comm_exit  = COMM_PER_TRADE * notional
            net_pnl    = realised - comm_exit

            open_trade.update({
                "exit_date":  date,
                "exit_price": exit_px,
                "pnl":        net_pnl,
                "win":        net_pnl > 0,
            })
            trades.append(open_trade)
            open_trade = None
            last_trade_date = date

        # -------------------------------------------------------------
        # C) NORMAL entry (only if flat)
        # -------------------------------------------------------------
        if signals["entries"].iloc[i] and open_trade is None:
            entry_exec  = True
            size_pct    = signals["size_pct"].iloc[i]
            entry_px    = signals.get("entry_price", price).iloc[i]
            notional    = prev_eq * leverage * size_pct
            comm_entry  = COMM_PER_TRADE * notional

            open_trade = {
                "entry_date":     date,
                "entry_price":    entry_px,
                "leverage":       leverage,
                "size_pct":       size_pct,
                "entry_notional": notional,
            }
            desired_pos     = leverage * size_pct
            prev_eq        -= comm_entry             # pay entry fee
            last_trade_date = date

        # -------------------------------------------------------------
        # D) Mark-to-market overnight P/L on carried position
        # -------------------------------------------------------------
        pnl_overnight = prev_pos * prev_eq * rets.iloc[i]

        equity.iloc[i] = prev_eq + pnl_overnight
        position[i]    = desired_pos

    # ------------------------------------------------------------------
    return equity, trades, pd.Series(position, index=dates)


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
            row = [idx]
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
                if hasattr(strat, "threshold"):
                    prev_close = price.shift(1)
                    entry_price = price.copy()
                    mask = signals["entries"].astype(bool)
                    entry_price[mask] = prev_close[mask] * (1 - strat.threshold)
                    signals["entry_price"] = entry_price

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
        # MAP PAGE: two plots, color‐coded by leverage + best‐fit curve (with capped outliers)
        # ───────────────────────────────────────────────────────────

        # prep data as percentages
        x1 = metrics_df["MaxDD%"] * 100
        y1 = metrics_df["TotalReturn"] * 100
        x2 = metrics_df["WinRate"]   * 100
        y2 = metrics_df["TotalReturn"] * 100

        # leverage mapping
        lev_raw     = metrics_df.get("Leverage", pd.Series(1, index=metrics_df.index))
        lev         = pd.to_numeric(lev_raw, errors="coerce")
        unique_levs = sorted(lev.dropna().unique())
        cmap        = plt.get_cmap("tab10")
        color_map   = {v: cmap(i) for i, v in enumerate(unique_levs)}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))


        # — Top: Total Return vs Max Drawdown —
        mask1      = np.isfinite(x1) & np.isfinite(y1)
        x1_clean   = x1[mask1]
        y1_clean   = y1[mask1]
        cap1       = np.percentile(y1_clean, 95)
        y1_clamped = np.minimum(y1, cap1)

        # scatter (clamped)
        for v in unique_levs:
            idx = lev == v
            ax1.scatter(
                x1[idx], y1_clamped[idx],
                label=f"Lev {int(v)}×",
                alpha=0.7,
                color=color_map[v]
            )

        # best‐fit (degree=2) on uncapped data
        if x1_clean.size > 2:
            coeffs1 = np.polyfit(x1_clean, y1_clean, deg=2)
            xs1     = np.linspace(x1_clean.min(), x1_clean.max(), 200)
            ys1     = np.polyval(coeffs1, xs1)
            ax1.plot(xs1, ys1, label="Best-fit (deg=2)", linewidth=2)

        # ensure the y‐limit covers both the cap and the peak of the fit
        if x1_clean.size > 2:
            top1 = max(cap1 * 1.05, ys1.max() * 1.05)
        else:
            top1 = cap1 * 1.05

        # outliers at cap
        out1 = y1 > cap1
        if out1.any():
            # 1) scatter the X’s on top of everything
            ax1.scatter(
                x1[out1],
                np.full(out1.sum(), cap1),
                marker="X", s=80,
                facecolors="none",
                edgecolors="black",
                label="Outliers",
                zorder=10
            )
            # 2) label each with its true TotalReturn (y1) just below the X
            for xi, yi in zip(x1[out1], y1[out1]):
                ax1.text(
                    xi,
                    cap1 - (top1 * 0.03),      # a little below the cap line
                    f"{yi:.1f}%",
                    ha="center", va="top",
                    fontsize=8,
                    zorder=11
                )

        ax1.set_ylim(bottom=y1_clean.min(), top=top1)
        ax1.set_xlabel("Max Drawdown (%)")
        ax1.set_ylabel("Total Return (%)")
        ax1.set_title("Total Return vs Max Drawdown")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(title="Leverage")


        # — Bottom: Total Return vs Hit Rate —
        mask2      = np.isfinite(x2) & np.isfinite(y2)
        x2_clean   = x2[mask2]
        y2_clean   = y2[mask2]
        cap2       = np.percentile(y2_clean, 95)
        y2_clamped = np.minimum(y2, cap2)

        # scatter (clamped)
        for v in unique_levs:
            idx = lev == v
            ax2.scatter(
                x2[idx], y2_clamped[idx],
                label=f"Lev {int(v)}×",
                alpha=0.7,
                color=color_map[v]
            )

        # best‐fit (degree=2) on uncapped data
        if x2_clean.size > 2:
            coeffs2 = np.polyfit(x2_clean, y2_clean, deg=2)
            xs2     = np.linspace(x2_clean.min(), x2_clean.max(), 200)
            ys2     = np.polyval(coeffs2, xs2)
            ax2.plot(xs2, ys2, label="Best-fit (deg=2)", linewidth=2)

        # y‐limit to cover cap and fit peak
        if x2_clean.size > 2:
            top2 = max(cap2 * 1.05, ys2.max() * 1.05)
        else:
            top2 = cap2 * 1.05

        # outliers at cap
        out2 = y2 > cap2
        if out2.any():
            ax2.scatter(
                x2[out2],
                np.full(out2.sum(), cap2),
                marker="X", s=80,
                facecolors="none",
                edgecolors="black",
                label="Outliers",
                zorder=10
            )
            for xi, yi in zip(x2[out2], y2[out2]):
                ax2.text(
                    xi,
                    cap2 - (top2 * 0.03),
                    f"{yi:.1f}%",
                    ha="center", va="top",
                    fontsize=8,
                    zorder=11
                )


        ax2.set_ylim(bottom=y2_clean.min(), top=top2)
        ax2.set_xlabel("Hit Rate (%)")
        ax2.set_ylabel("Total Return (%)")
        ax2.set_title("Total Return vs Hit Rate")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(title="Leverage")


        # save page
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
