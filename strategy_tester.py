"""
strategy_tester.py
Manual batch tester that ranks strategies, outputs CSV and a PDF chart-pack.
Rewritten to remove vectorbt dependency and support leverage multipliers.

Usage:
    python strategy_tester.py --ticker SPY --start 2020-01-01 --end 2025-06-25 --leverage 1.0,3.0,10.0 --stop-loss 0.05
"""

import argparse
from pathlib import Path

import yfinance as yf
import numpy as np
import pandas as pd
from fredapi import Fred
from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    raw = yf.download(
        [ticker, "^VIX", "^TNX"],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    # MultiIndex => select Close; else single-ticker edge-case
    if isinstance(raw.columns, pd.MultiIndex):
        price = raw.xs("Close", level=0, axis=1)
    else:
        price = raw[["Close"]].rename(columns={"Close": ticker})
    price.columns = [c.upper() for c in price.columns]
    return price

def load_macro_series() -> dict[str, pd.Series]:
    cpi    = fred.get_series("CPIAUCSL")
    unrate = fred.get_series("UNRATE")
    return {"CPIAUCSL": cpi, "UNRATE": unrate}

def simulate_portfolio(
    price: pd.Series,
    signals: dict[str, pd.Series],
    leverage: float,
    init_cash: float = INIT_CASH_DEFAULT
) -> tuple[pd.Series, list[dict]]:
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

        # 4) Record trades
        if change > 0 and open_trade is None:
            open_trade = {
                "entry_date": date,
                "entry_equity": eq_today,
                "leverage": leverage
            }
        elif change < 0 and open_trade is not None:
            open_trade.update({
                "exit_date": date,
                "exit_equity": eq_today,
            })
            open_trade["pnl"] = open_trade["exit_equity"] - open_trade["entry_equity"]
            open_trade["win"] = open_trade["pnl"] > 0
            trades.append(open_trade)
            open_trade = None

    return equity, trades

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

    levs = [float(x) for x in args.leverage.split(",")]

    # Load data
    data  = load_market_data(args.ticker, args.start, args.end)
    price = data[args.ticker]
    aux   = {
        "^VIX": data["^VIX"],
        "^TNX": data["^TNX"],
        **load_macro_series(),
    }

    equity_curves = {}
    trade_books   = {}

    # Run each strategy × each leverage
    for lev in levs:
        for name, Strat in STRATEGY_REGISTRY.items():
            strat    = Strat(price, aux=aux, stop_loss=args.stop_loss)
            signals  = strat.generate_signals()
            eq_curve, trades = simulate_portfolio(
                price, signals, lev, init_cash=args.init_cash
            )
            key = f"{name}_{lev}x"
            equity_curves[key] = eq_curve
            trade_books[key]   = trades
            print(f">>> {key}: final equity = {eq_curve.iloc[-1]:.4f}, "
                  f"{len(trades)} trades")

    # Build metrics table
    metrics = []
    for key, eq in equity_curves.items():
        total_ret  = eq.iloc[-1] / eq.iloc[0] - 1
        daily_rets = eq.pct_change().dropna()
        sharpe     = (
            daily_rets.mean() / daily_rets.std() * np.sqrt(252)
            if daily_rets.std() != 0 else np.nan
        )
        dd     = eq / eq.cummax() - 1
        max_dd = dd.min()
        calmar = total_ret / abs(max_dd) if max_dd != 0 else np.nan
        wins   = sum(1 for t in trade_books[key] if t["win"])
        winrate= wins / len(trade_books[key]) if trade_books[key] else np.nan

        metrics.append({
            "Strategy":   key,
            "TotalReturn": total_ret,
            "Sharpe":      sharpe,
            "Calmar":      calmar,
            "MaxDD%":      max_dd,
            "WinRate":     winrate
        })

    metrics_df = pd.DataFrame(metrics).set_index("Strategy")

    # Save CSV
    outdir = Path("results") / args.ticker
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(outdir / "strategy_metrics.csv", float_format="%.4f")
    print("\n=== Strategy Metrics ===")
    print(metrics_df.to_string(float_format="%.4f"))

    # Generate PDF report for top 3 by Sharpe
    top3    = metrics_df["Sharpe"].nlargest(3).index
    pdf_path= outdir / "strategy_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: metrics table
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        tbl = ax.table(
            cellText=[
                [idx] + [f"{metrics_df.loc[idx, col]:.4f}" for col in metrics_df.columns]
                for idx in metrics_df.index
            ],
            colLabels=["Strategy"] + list(metrics_df.columns),
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        ax.set_title("Strategy Metrics", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # Pages for each top strategy
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
