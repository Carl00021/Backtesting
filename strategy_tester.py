"""
strategy_tester.py
Vectorised batch tester that ranks strategies, outputs CSV and a PDF chart-pack.
Patched for npNaN: python -m pip install "git+https://github.com/twopirllc/pandas-ta.git@master" 
Or Use: python -m pip install "numpy<2.0" --upgrade --force-reinstall

Usage:
    python strategy_tester.py --ticker AAPL --start 2020-01-01 --end 2025-06-25
"""

import argparse
from pathlib import Path
import yfinance as yf
import numpy as np
import pandas as pd
from fredapi import Fred
from fpdf import FPDF  # lightweight PDF builder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import vectorbt as vbt
vbt.settings.array_wrapper['freq']      = '1D'
vbt.settings.returns['year_freq']       = '252d'
from Strategies import STRATEGY_REGISTRY
from dotenv import load_dotenv
load_dotenv()   
print("Loaded strategies:", list(STRATEGY_REGISTRY.keys()))
# -------------------- CONFIG -------------------------------------------------
COMM_PER_TRADE = 0.0025   # 0.25 % per trade
CAPITAL_GAINS_TAX = 0.20  # 20 % of realised PnL
DAILY_MARK_TO_MARKET = True

fred = Fred()  # assumes FRED_API_KEY in env

# -------------------- HELPERS ------------------------------------------------
# strategy_tester.py  - replace load_market_data
def load_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # ----- OPTION A: adjusted prices (simplest) --------------------------
    raw = yf.download(
        [ticker, "^VIX", "^TNX"],
        start=start,
        end=end,
        auto_adjust=True,      # keep
        progress=False,
    )

    # When multiple tickers are passed, yfinance returns a MultiIndex frame
    # Level-0 = field ("Open"/"High"/"Close"/"Volume"), Level-1 = ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        price = raw.xs("Close", level=0, axis=1)   # grab all “Close” columns
    else:                                          # single-ticker edge-case
        price = raw[["Close"]].rename(columns={"Close": ticker})

    price.columns = [c.upper() for c in price.columns]  # e.g. "SPY", "^VIX"
    return price

def load_macro_series() -> dict[str, pd.Series]:
    cpi = fred.get_series("CPIAUCSL")
    unrate = fred.get_series("UNRATE")
    return {"CPIAUCSL": cpi, "UNRATE": unrate}

def build_portfolio(price, strat_obj):
    sig = strat_obj.generate_signals()

    # Reindex here instead of in generate_signals:
    entries = sig["entries"].reindex(price.index, fill_value=False).to_numpy()
    exits   = sig["exits"].reindex(price.index, fill_value=False).to_numpy()
    size    = sig["size_pct"].reindex(price.index, fill_value=0.0).to_numpy()

    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        size=size,
        size_type="percent",
        fees=COMM_PER_TRADE,
        freq="1D",
        init_cash=1.0,
        call_seq="auto",
    )

    # 1) extract trade PnL and exit indices as numpy arrays
    pnl_arr    = pf.trades.pnl.values              # shape (n_trades,)
    exit_idx   = pf.trades.exit_idx.values         # same shape

    # 2) only tax positive PnL
    realized_pnl = np.maximum(pnl_arr, 0)
    tax_arr      = realized_pnl * CAPITAL_GAINS_TAX  # per‐trade tax amount

    # 3) map exit indices back to dates
    exit_dates = price.index[exit_idx]
    tax_series = pd.Series(tax_arr, index=exit_dates)

    # 4) sum taxes by bar and build after‐tax equity curve
    tax_by_bar      = tax_series.groupby(level=0).sum()
    equity          = pf.value()
    cum_tax         = tax_by_bar.reindex(equity.index, fill_value=0).cumsum()
    equity_after_tax = equity - cum_tax

    print(
            f">>> {strat_obj.__class__.__name__}: "
            f"{len(pf.trades)} trades, "
            f"final equity = {equity_after_tax.iloc[-1]:.4f}"
        )

    return pf, equity_after_tax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    data = load_market_data(args.ticker, args.start, args.end)
    price = data[args.ticker]

    aux = {
        "^VIX": data["^VIX"],
        "^TNX": data["^TNX"],
        **load_macro_series(),
    }

    portfolios = {}
    equity_curves = {}
    for name, klass in STRATEGY_REGISTRY.items():
        pf, eq_after_tax = build_portfolio(price, klass(price, aux=aux, stop_loss=0.05))
        portfolios[name]   = pf
        equity_curves[name]= eq_after_tax

    # ------------------ METRICS & RANKING -----------------------------------
    metrics = pd.DataFrame({
        name: {
            "TotalReturn": eq_curve.iloc[-1] / eq_curve.iloc[0] - 1,

            "Sharpe": eq_curve
                .pct_change()
                .fillna(0)
                .vbt
                .returns(freq='1D')        # ← freq supplied here
                .sharpe_ratio(),

            "Calmar": (eq_curve.iloc[-1] / eq_curve.iloc[0] - 1)
                    / abs((eq_curve / eq_curve.cummax() - 1).min()),

            "MaxDD%": (eq_curve / eq_curve.cummax() - 1).min(),

            "WinRate": portfolios[name].trades.win_rate(),
        }
        for name, eq_curve in equity_curves.items()
    }).T

    # Rank by each metric
    leaders = {m: metrics[m].sort_values(ascending=False) for m in metrics.columns}

    # Export CSV summary
    outdir = Path("results") / args.ticker
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / "strategy_metrics.csv", float_format="%.4f")
    print("\n=== Strategy Metrics ===")
    print(metrics.to_string(float_format="%.4f"))

    # ------------------ PDF Report on Top 3 Strategies ----------------------
    top3 = metrics['Sharpe'].sort_values(ascending=False).head(3).index
    pdf_path = outdir / "strategy_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: Metrics table
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        tbl = ax.table(
            cellText=[ [idx] + [f"{metrics.loc[idx, col]:.4f}" for col in metrics.columns]
                       for idx in metrics.index ],
            colLabels=["Strategy"] + list(metrics.columns),
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        ax.set_title("Strategy Metrics", fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        # Pages for each top strategy
        for name in top3:
            fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)

            # 1) Price + trade markers
            price.plot(ax=axes[0], title=f"{name} - Price & Trades", alpha=0.5)
            entries = portfolios[name].trades.entry_idx.values
            exits = portfolios[name].trades.exit_idx.values
            entry_dates = price.index[entries]
            exit_dates = price.index[exits]
            axes[0].scatter(entry_dates, price.loc[entry_dates], marker='^', color='green', label='Entry')
            axes[0].scatter(exit_dates, price.loc[exit_dates], marker='v', color='red', label='Exit')
            axes[0].legend(loc='best')

            # 2) Equity curve
            equity_curves[name].plot(ax=axes[1], title=f"{name} - Equity Curve (After Tax)")
            axes[1].set_ylabel("Equity Value")

            # 3) Drawdown
            dd = equity_curves[name] / equity_curves[name].cummax() - 1
            dd.plot(ax=axes[2], title=f"{name} - Drawdown Curve")
            axes[2].set_ylabel("Drawdown")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Summary saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
