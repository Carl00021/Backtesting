#!/usr/bin/env python3
"""
multi_strategy_backtest_pdf.py

Backtest multiple flavors of dip-buying strategies (including Intraday Drop)
with a stop loss parameter and generate a multi-page PDF report.
"""

import os
import warnings
import textwrap
from itertools import product

import numpy as np
import pandas as pd
from numba import njit
import yfinance as yf
from curl_cffi import requests

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------------------------------------------------
# User-configurable parameters
# --------------------------------------------------------------------
TICKER        = "QQQ"
START_DATE    = "2015-01-01"
END_DATE      = "2025-06-13"
INITIAL_CAP   = 100_000.0
FEE_PER_TRADE = 10.0
DIV_TAX       = 0.38
CAP_GAIN_TAX  = 0.25
PDF_NAME      = f"{TICKER}-BTD_Report-{END_DATE}.pdf"

# Parameter grids
LEVERAGES       = [-1.0, 1.0, 2.0, 3.0, 5.0, 10.0]
N_CONSEC_DS     = [2,3,4,5]
PCT_DROPS       = np.linspace(-0.03, 0.0, 8).tolist()
INTRA_PCT_DROP  = np.linspace(-0.03, 0.0, 8).tolist()
MULTI_PCT_DROPS = np.linspace(-0.03, -0.1, 8).tolist() + np.linspace(-0.15, -0.50, 8).tolist()
MULTI_DROP_DAYS = [2, 3, 5, 10, 15, 20, 50]
HOLD_DAYS       = [1, 2, 3, 4, 5] + np.linspace(10, 50, 9).tolist()
STOP_LOSS_PCTS  = [0.02,0.05,0.1,0.3,0.5,1]  # Stop loss percentages to optimize (1%, 2%, 5%, 10%)

warnings.filterwarnings("ignore", category=SyntaxWarning)

# --------------------------------------------------------------------
# Numba-accelerated signal generation and backtest
# --------------------------------------------------------------------
@njit(fastmath=True)
def _consecutive_downs(diff):
    n = diff.size
    res = np.zeros(n, np.int32)
    run = 0
    for i in range(n):
        run = run + 1 if diff[i] < 0.0 else 0
        res[i] = run
    return res

@njit(fastmath=True)
def _fill_hold(trigger, hold):
    n = trigger.size
    pos = np.zeros(n, np.int8)
    i = 1
    while i < n:
        if trigger[i-1]:
            end = min(i + hold, n)
            for j in range(i, end):
                pos[j] = 1
            i = end
        else:
            i += 1
    return pos

@njit(fastmath=True)
def generate_btd_signal(price, n_consec, pct_drop, multi_drop, multi_days, hold_days):
    n = price.size
    pct_ret = np.empty(n, np.float64)
    pct_ret[0] = 0.0
    for i in range(1, n):
        pct_ret[i] = (price[i] - price[i-1]) / price[i-1]
    diff = np.empty(n, np.float64)
    diff[0] = 0.0; diff[1:] = price[1:] - price[:-1]
    consec = _consecutive_downs(diff)
    drop_nd = price / np.roll(price, multi_days) - 1.0
    drop_nd[:multi_days] = 0.0
    trigger = (consec >= n_consec) | (pct_ret <= pct_drop) | (drop_nd <= multi_drop)
    return _fill_hold(trigger, hold_days)

@njit(fastmath=True)
def backtest(price, divs, pos, lev, fee, div_tax, cap_tax, initial_cap, stop_loss_pct):
    """
    Backtest a trading strategy with daily mark-to-market, bankruptcy check, and stop loss.
    
    Parameters:
    - price: Array of daily prices
    - divs: Array of daily dividends
    - pos: Array indicating position (1 to hold, 0 to be out)
    - lev: Leverage factor (positive for long, negative for short)
    - fee: Transaction fee per trade
    - div_tax: Dividend tax rate
    - cap_tax: Capital gains tax rate
    - initial_cap: Initial capital
    - stop_loss_pct: Percentage loss to trigger stop loss (e.g., 0.05 for 5%)
    
    Returns:
    - equity: Array of daily equity values
    - total_return: Total return over the period
    - ann_return: Annualized return
    - entry_days: Boolean array indicating entry days
    - normal_exit_days: Boolean array indicating normal exit days
    - stop_loss_exit_days: Boolean array indicating stop loss exit days
    """
    n = len(price)
    equity = np.empty(n, dtype=np.float64)
    equity[0] = initial_cap
    entry_days = np.zeros(n, dtype=np.bool_)
    normal_exit_days = np.zeros(n, dtype=np.bool_)
    stop_loss_exit_days = np.zeros(n, dtype=np.bool_)
    
    # Initialize portfolio variables
    cash = initial_cap
    debt = 0.0
    shares = 0.0
    entry_price = 0.0
    in_pos = False
    
    for i in range(1, n):
        if in_pos:
            position_value = shares * price[i]
            equity[i] = cash + position_value - debt
            equity[i] += divs[i] * shares * (1.0 - div_tax)
            
            # Check for bankruptcy
            if equity[i] <= 0.0:
                equity[i:] = 0.0
                break
            
            # Check stop loss
            if (lev > 0 and price[i] < entry_price * (1 - stop_loss_pct)) or \
               (lev < 0 and price[i] > entry_price * (1 + stop_loss_pct)):
                # Exit at price[i]
                cash_from_sale = shares * price[i]
                if debt > 0:
                    cash_after_debt = cash_from_sale - debt
                else:
                    cash_after_debt = cash + cash_from_sale
                cash_after_debt -= fee
                pnl = (price[i] - entry_price) * shares
                if pnl > 0:
                    tax = pnl * cap_tax
                    cash_after_debt -= tax
                equity[i] = cash_after_debt
                cash = cash_after_debt
                shares = 0.0
                debt = 0.0
                in_pos = False
                stop_loss_exit_days[i] = True
        else:
            equity[i] = cash
        
        # Handle entry and exit
        if (pos[i-1] == 1) and (not in_pos):
            # Enter position at price[i-1]
            entry_price = price[i-1]
            investment = equity[i] * lev
            shares = investment / entry_price
            if lev > 0:
                if lev > 1:
                    debt = (lev - 1) * equity[i]
                    cash = 0.0
                else:
                    debt = 0.0
                    cash = equity[i] - (shares * entry_price)
            else:  # lev < 0, short position
                cash = equity[i] - (shares * entry_price)  # shares < 0, so cash increases
                debt = 0.0
            # Pay transaction fee
            if cash >= fee:
                cash -= fee
            else:
                debt += (fee - cash)
                cash = 0.0
            entry_days[i-1] = True
            in_pos = True
        elif (pos[i-1] == 0) and in_pos:
            # Exit position at price[i-1]
            cash_from_sale = shares * price[i-1]
            if debt > 0:
                cash_after_debt = cash_from_sale - debt
            else:
                cash_after_debt = cash + cash_from_sale
            cash_after_debt -= fee
            pnl = (price[i-1] - entry_price) * shares
            if pnl > 0:
                tax = pnl * cap_tax
                cash_after_debt -= tax
            equity[i] = cash_after_debt
            cash = cash_after_debt
            shares = 0.0
            debt = 0.0
            normal_exit_days[i-1] = True  # Changed from i to i-1 to mark the correct exit day
            in_pos = False
    
    # Calculate returns
    total_return = equity[-1] / initial_cap - 1.0
    if n > 1:
        ann_return = (equity[-1] / initial_cap) ** (252.0 / (n - 1)) - 1.0
    else:
        ann_return = 0.0
    
    return equity, total_return, ann_return, entry_days, normal_exit_days, stop_loss_exit_days

# --------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------
def fetch_ticker(ticker, start, end=None):
    sess = requests.Session(impersonate="chrome")
    df = yf.Ticker(ticker, session=sess).history(start=start, end=end).tz_localize(None)
    if df.empty:
        raise ValueError(f"No data for ticker {ticker}")
    df = df[["Close", "Dividends"]].rename(columns={"Close": "Price", "Dividends": "Div"})
    return df

# --------------------------------------------------------------------
# Strategy flavor wrappers
# --------------------------------------------------------------------
def strat_consec(price, divs, lev, n_consec, hold):
    return generate_btd_signal(price, int(n_consec), -1.0, 0.0, 1, int(hold))

def strat_one(price, divs, lev, pct_drop, hold):
    return generate_btd_signal(price, 9999, pct_drop, 0.0, 1, int(hold))

def strat_multi(price, divs, lev, multi_drop, multi_days, hold):
    return generate_btd_signal(price, 9999, 0.0, multi_drop, int(multi_days), int(hold))

def strat_comb(price, divs, lev, n_consec, pct_drop, multi_drop, multi_days, hold):
    return generate_btd_signal(price, int(n_consec), pct_drop, multi_drop, int(multi_days), int(hold))

def strat_intraday(price, divs, lev, pct_drop, hold):
    return generate_btd_signal(price, 9999, 0.0, pct_drop, 1, int(hold))

# Parameter grids and registry
CONSEC_PARAMS    = list(product(LEVERAGES, N_CONSEC_DS, HOLD_DAYS, STOP_LOSS_PCTS))
ONE_PARAMS       = list(product(LEVERAGES, PCT_DROPS, HOLD_DAYS, STOP_LOSS_PCTS))
MULTI_PARAMS     = list(product(LEVERAGES, MULTI_PCT_DROPS, MULTI_DROP_DAYS, HOLD_DAYS, STOP_LOSS_PCTS))
COMB_PARAMS      = list(product(LEVERAGES, N_CONSEC_DS, PCT_DROPS, MULTI_PCT_DROPS, MULTI_DROP_DAYS, HOLD_DAYS, STOP_LOSS_PCTS))
INTRADAY_PARAMS  = list(product(LEVERAGES, PCT_DROPS, HOLD_DAYS, STOP_LOSS_PCTS))

STRATS = {
    "Consecutive": (strat_consec,    CONSEC_PARAMS,   ["n_consec", "hold"]),
    "One-day":     (strat_one,       ONE_PARAMS,      ["pct_drop", "hold"]),
    "Multi-day":   (strat_multi,     MULTI_PARAMS,    ["multi_drop", "multi_days", "hold"]),
    "Combined":    (strat_comb,      COMB_PARAMS,     ["n_consec", "pct_drop", "multi_drop", "multi_days", "hold"]),
    "Intraday":    (strat_intraday,  INTRADAY_PARAMS, ["pct_drop", "hold"]),
}

# --------------------------------------------------------------------
# Evaluate one strategy flavor over its parameter grid
# --------------------------------------------------------------------
def eval_flavor(price_s, div_s, name, func, params, pnames, fee=FEE_PER_TRADE, div_tax=DIV_TAX, cap_tax=CAP_GAIN_TAX, initial_cap=INITIAL_CAP):
    """Run all param tuples for one strategy flavor with stop loss, capped losses, and bankruptcy enforcement."""
    rows = []
    price_np = price_s.values.astype(np.float64)
    div_np = div_s.values.astype(np.float64)
    n = price_np.size

    for p in params:
        lev = float(p[0])
        stop_loss_pct = p[-1]
        args = p[1:-1]
        hold = int(args[-1])  # hold period as integer

        pos = func(price_np, div_np, lev, *args)
        eq, tot, ann, _, _, _ = backtest(price_np, div_np, pos, lev, fee, div_tax, cap_tax, initial_cap, stop_loss_pct)

        # Identify entries and exits as integer arrays
        entries = np.where((pos[:-1] == 1) & (np.roll(pos, 1)[:-1] == 0))[0]
        exits = entries + hold
        # Keep only trades that complete before end
        mask = exits < n
        entries = entries[mask]
        exits = exits[mask]

        # Compute per-trade returns, clipped at -100%
        raw_rets = (price_np[exits] / price_np[entries] - 1.0) * lev
        returns = np.maximum(raw_rets, -1.0)
        # Enforce total wipe-out if any single loss hits -100%
        if returns.size and np.any(returns <= -1.0):
            tot = -1.0
            ann = -1.0

        # Trade statistics
        hit = np.nan if returns.size == 0 else (returns > 0).mean()
        pos_rets = returns[returns > 0]
        neg_rets = returns[returns <= 0]
        avg_pos = np.nan if pos_rets.size == 0 else pos_rets.mean()
        avg_neg = np.nan if neg_rets.size == 0 else neg_rets.mean()
        last_ret = np.nan if returns.size == 0 else returns[-1]

        action = "hold" if pos[-1] == 1 else "buy"

        # Assemble result row
        row = {
            "flavor": name,
            "lev": lev,
            "stop_loss_pct": stop_loss_pct,
            "tot_ret": tot,
            "ann_ret": ann,
            "hit_rate": hit,
            "max_drawdown": (eq / np.maximum.accumulate(eq) - 1.0).min(),
            "average_positive_return": avg_pos,
            "average_negative_return": avg_neg,
            "last_trade_return": last_ret,
            "buy_or_hold": action,
        }
        # Add parameter values
        for nm, val in zip(pnames, args):
            row[nm] = val
        rows.append(row)

    return pd.DataFrame(rows)

# --------------------------------------------------------------------
# Table drawing helper for a given Axes with wrapped headers
# --------------------------------------------------------------------
def draw_table_ax(ax, df, title):
    dfc = df.copy()
    # format percentages
    for col in ["pct_drop", "multi_drop", "stop_loss_pct"]:
        if col in dfc.columns:
            dfc[col] = dfc[col].map(lambda x: f"{x:.2%}")
    for col in ["tot_ret", "ann_ret", "hit_rate", "max_drawdown",
                "average_positive_return", "average_negative_return", "last_trade_return"]:
        if col in dfc.columns:
            dfc[col] = dfc[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    # prepare column order
    base_cols = ["flavor", "lev", "stop_loss_pct", "tot_ret", "ann_ret", "hit_rate", "max_drawdown",
                 "average_positive_return", "average_negative_return", "last_trade_return", "buy_or_hold"]
    cols = [c for c in dfc.columns if c not in base_cols]
    display_cols = base_cols + cols
    # wrap and humanize headers
    header_labels = []
    for c in display_cols:
        human = c.replace("_", " ").title()
        wrapped = "\n".join(textwrap.wrap(human, width=10))
        header_labels.append(wrapped)
    ax.axis("off")
    ax.set_title(title, pad=8)
    tbl = ax.table(
        cellText=dfc[display_cols].values,
        colLabels=header_labels,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    # Double header row height
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_height(cell.get_height() * 2)

# --------------------------------------------------------------------
# Generate PDF report
# --------------------------------------------------------------------
def save_pdf(price_s, div_s, best_df, master_df, top1x_df, topm1x_df, strategies):
    """Generate PDF report with equity curve, markers, drawdown, and hit-rate subplots."""
    # Compute Buy & Hold metrics
    price_np = price_s.values
    n = len(price_np)
    bh_tot = price_np[-1] / price_np[0] - 1.0
    bh_ann = (1.0 + bh_tot)**(252.0/(n-1)) - 1.0
    bh_eq = price_np / price_np[0] * INITIAL_CAP
    bh_dd = (bh_eq - np.maximum.accumulate(bh_eq)) / np.maximum.accumulate(bh_eq)
    bh_max_dd = bh_dd.min()
    bh_hit = (price_np[1:] > price_np[:-1]).mean()

    # Build a single "baseline" row dict with blanks for non‐applicable cols
    def make_bh_row(df):
        row = {
            "flavor":             "Buy & Hold",
            "lev":                1.0,
            "stop_loss_pct":      "",
            "tot_ret":            bh_tot,
            "ann_ret":            bh_ann,
            "hit_rate":           bh_hit,
            "max_drawdown":       bh_max_dd,
            "average_positive_return": "",
            "average_negative_return": "",
            "last_trade_return":  "",
            "buy_or_hold":        "hold",
        }
        # ensure we have every column the table expects
        for c in df.columns:
            if c not in row:
                row[c] = ""
        # prepend it
        return pd.concat([pd.DataFrame([row])[df.columns], df], ignore_index=True)
    
    best_df   = make_bh_row(best_df)
    top1x_df  = make_bh_row(top1x_df)
    topm1x_df = make_bh_row(topm1x_df)
    
    with PdfPages(PDF_NAME) as pdf:
        # Page 1
        fig, axes = plt.subplots(3, 1, figsize=(11, 8),
                                 gridspec_kw={"height_ratios":[1,1,1], "hspace":0.3})
        draw_table_ax(axes[0], best_df,    "Best Strategy & Params by Flavor")
        draw_table_ax(axes[1], top1x_df,   "+1× Top Performer by Flavor")
        draw_table_ax(axes[2], topm1x_df,  "-1× Top Performer by Flavor")
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Top-10 across all flavors
        top10p = master_df.nlargest(10, "ann_ret")
        fig2, ax2 = plt.subplots(figsize=(11, 6))
        draw_table_ax(ax2, top10p, "Top-10 — Overall")
        fig2.tight_layout(); pdf.savefig(fig2); plt.close(fig2)

        # Page 3: Top-10 @ +1× leverage
        top10p = master_df[master_df.lev == 1.0].nlargest(10, "ann_ret")
        fig3, ax3 = plt.subplots(figsize=(11, 6))
        draw_table_ax(ax3, top10p, "Top-10 — +1× Leverage")
        fig3.tight_layout(); pdf.savefig(fig3); plt.close(fig3)

        # Page 4: Top-10 @ −1× leverage
        top10n = master_df[master_df.lev == -1.0].nlargest(10, "ann_ret")
        fig4, ax4 = plt.subplots(figsize=(11, 6))
        draw_table_ax(ax4, top10n, "Top-10 — -1× Leverage")
        fig4.tight_layout(); pdf.savefig(fig4); plt.close(fig4)

        # Pages 5–8: Top-10 per flavor
        for name in strategies:
            df_fl = master_df[master_df.flavor == name]
            for lab, df_sel in [("Overall", df_fl.nlargest(10, "ann_ret")),
                                ("+1×", df_fl[df_fl.lev == 1.0].nlargest(10, "ann_ret")),
                                ("-1×", df_fl[df_fl.lev == -1.0].nlargest(10, "ann_ret"))]:
                fig_f, ax_f = plt.subplots(figsize=(11, 6))
                draw_table_ax(ax_f, df_sel, f"Top-10 — {name} ({lab})")
                fig_f.tight_layout(); pdf.savefig(fig_f); plt.close(fig_f)

        # Pages 9+: Equity, markers, drawdown, hit-rate
        for _, row in best_df.iterrows():
            func, pnames = strategies[row.flavor][0], strategies[row.flavor][2]
            args = [row[p] for p in pnames]
            hold = int(args[-1])

            # Prepare data
            price_np = price_s.values.astype(np.float64)
            div_np   = div_s.values.astype(np.float64)
            pos      = func(price_np, div_np, row.lev, *args)

            # Backtest results
            eq, tot_ret, ann_ret, entry_days, normal_exit_days, stop_loss_exit_days = backtest(
                price_np, div_np, pos, row.lev,
                FEE_PER_TRADE, DIV_TAX, CAP_GAIN_TAX, INITIAL_CAP, row['stop_loss_pct']
            )

            # Buy & Hold benchmark
            bh_eq = INITIAL_CAP * (1.0 + row.lev * (price_np / price_np[0] - 1.0))

            # Drawdown
            peak = np.maximum.accumulate(eq)
            drawdown = (eq - peak) / peak

            # Hit-rate metrics
            wins = (eq[1:] > eq[:-1]).astype(float)
            total_hit = wins.mean()
            roll5  = pd.Series(wins).rolling(5).mean().values
            roll10 = pd.Series(wins).rolling(10).mean().values

            # Title
            param_text = ", ".join(f"{n}={v}" for n, v in zip(pnames, args))
            title = f"{row.flavor} ({param_text}, stop_loss_pct={row['stop_loss_pct']:.2%}) — Equity Curve (ann_ret={ann_ret:.2%})"

            # Plot subplots
            fig, (ax1, ax_markers, ax2, ax3) = plt.subplots(
                4, 1, figsize=(11, 10), sharex=True,
                gridspec_kw={"height_ratios": [3, 2, 1, 1], "hspace": 0.3}
            )
            
            # Equity curve (Subplot 1) - remains unchanged
            ax1.plot(price_s.index, eq, label='Strategy Equity')
            ax1.plot(price_s.index, bh_eq, label=f'Buy & Hold ({row.lev}×)')
            ax1.set_yscale('log')
            ax1.text(0.02, 0.95, f"Strat Ret: {tot_ret:.2%}\nB&H Ret: {bh_eq[-1]/INITIAL_CAP - 1.0:.2%}",
                     transform=ax1.transAxes, va='top', fontsize=8,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax1.set_title(title, pad=12)
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, lw=0.3)
            ax1.legend(loc='best', fontsize=8)
            
            # Markers subplot (Subplot 2) - modified to show only entries, normal exits, and stop loss exits
            if entry_days.any():
                ax_markers.scatter(price_s.index[entry_days], price_s[entry_days], marker='^', s=50,
                                   label='Entry', color='green')
            if normal_exit_days.any():
                ax_markers.scatter(price_s.index[normal_exit_days], price_s[normal_exit_days], marker='v', s=50,
                                   label='Normal Exit', color='blue')
            if stop_loss_exit_days.any():
                ax_markers.scatter(price_s.index[stop_loss_exit_days], price_s[stop_loss_exit_days], marker='x', s=50,
                                   label='Stop Loss Exit', color='red')
            ax_markers.set_ylabel('Price')
            ax_markers.grid(True, lw=0.3)
            ax_markers.legend(loc='best', fontsize=8)
            
            # Drawdown subplot (Subplot 3) - remains unchanged
            ax2.plot(price_s.index, drawdown)
            ax2.set_ylabel('Drawdown')
            ax2.grid(True, lw=0.3)
            
            # Hit-rate subplot (Subplot 4) - remains unchanged
            ax3.plot(price_s.index[1:], [total_hit] * (len(price_s.index)-1), label='Total Hit')
            ax3.plot(price_s.index[1:], roll5, label='Rolling 5')
            ax3.plot(price_s.index[1:], roll10, label='Rolling 10')
            ax3.set_ylabel('Hit Rate')
            ax3.set_ylim(0, 1)
            ax3.grid(True, lw=0.3)
            ax3.legend(loc='best', fontsize=8)
            ax3.set_xlabel('Date')

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

# --------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------
def main():
    df = fetch_ticker(TICKER, START_DATE, END_DATE)
    price_s = df["Price"]
    div_s = df["Div"]

    dfs = []
    for name, (func, params, pnames) in STRATS.items():
        print(f"Running {name}...")
        dfs.append(eval_flavor(price_s, div_s, name, func, params, pnames))
    master_df = pd.concat(dfs, ignore_index=True)

    # Debug: Inspect master_df
    print("master_df columns and dtypes:")
    print(master_df.dtypes)
    print("Sample of master_dfansson:")
    print(master_df.head())

    # Filter duplicates from master_df based on tot_ret, hit_rate, and max_drawdown
    filtered_dfs = []
    for flavor in STRATS.keys():
        df_flavor = master_df[master_df['flavor'] == flavor].copy()
        print(f"Processing flavor: {flavor}, rows before filtering: {len(df_flavor)}")
        
        # Handle NaN values in hit_rate and max_drawdown for comparison
        df_flavor['hit_rate'] = df_flavor['hit_rate'].fillna(-999)
        df_flavor['max_drawdown'] = df_flavor['max_drawdown'].fillna(-999)
        
        # Round values to avoid floating-point precision issues
        df_flavor['tot_ret'] = df_flavor['tot_ret'].round(8)
        df_flavor['hit_rate'] = df_flavor['hit_rate'].round(8)
        df_flavor['max_drawdown'] = df_flavor['max_drawdown'].round(8)
        
        # Drop duplicates based on tot_ret, hit_rate, and max_drawdown, keeping the first
        df_filtered = df_flavor.drop_duplicates(
            subset=['tot_ret', 'hit_rate', 'max_drawdown'], 
            keep='first'
        )
        
        print(f"Flavor: {flavor}, rows after filtering: {len(df_filtered)}")
        filtered_dfs.append(df_filtered)
    
    master_df_filtered = pd.concat(filtered_dfs, ignore_index=True)

    # Use master_df_filtered instead of master_df in subsequent code
    best_idx = master_df_filtered.groupby("flavor").ann_ret.idxmax()
    best_df = master_df_filtered.loc[best_idx].reset_index(drop=True)

    top1x_df = (master_df_filtered[master_df_filtered.lev == 1.0]
                .groupby("flavor").apply(lambda df: df.loc[df.ann_ret.idxmax()]).reset_index(drop=True))
    topm1x_df = (master_df_filtered[master_df_filtered.lev == -1.0]
                 .groupby("flavor").apply(lambda df: df.loc[df.ann_ret.idxmax()]).reset_index(drop=True))

    save_pdf(price_s, div_s, best_df, master_df_filtered, top1x_df, topm1x_df, STRATS)
    print(f"PDF generated at {os.path.abspath(PDF_NAME)}")
    
if __name__ == "__main__":
    main()