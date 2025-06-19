# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 23:57:17 2025

@author: 14165
"""

import re
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --------------------------- CONFIGURATION --------------------------- #
# Helps get around Rate Limit Requests
from curl_cffi import requests
session = requests.Session(impersonate="chrome")

def example_multi_ticker_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Example strategy that:
      • Uses the primary ticker’s price & volume (columns "Close", "Volume").
      • Also references a secondary ticker’s price, e.g. VIX_Close.

    For demonstration: “Go long primary (position=1) only when VIX_Close > 40.
    Otherwise flat. Forward-fill so we hold until VIX_Close drops below 40.”
    """
    # Assume df already has columns:
    #   "Close"  → primary ticker’s close
    #   "Volume" → primary ticker’s volume
    #   "<SEC>_Close" → e.g. "VIX_Close"
    sec_prefix = sanitize_ticker("^VIX")
    if f"{sec_prefix}_Close" not in df.columns:
        raise ValueError(f"Need '{sec_prefix}_Close' in DataFrame for this strategy.")
    
    df["SMA200"] = ta.sma(df["Close"], length=200)
    valid_sma200 = df["SMA200"].notna()
    cond_price_above_200 = pd.Series(False, index=df.index)
    cond_price_above_200.loc[ valid_sma200 ] = (     df.loc[valid_sma200, "Close"] > df.loc[valid_sma200, "SMA200"])
    
    
    #VIX
    vix_low = 21
    vix_high = 45
    cond_vix_below_vix_low = df[f"{sec_prefix}_Close"] < vix_low
    cond_vix_crossing_vix_high_below = df[f"{sec_prefix}_Close"] >vix_high & (df[f"{sec_prefix}_Close"].shift(1) <= vix_high)
    cond_vix_crossing_vix_high_above = df[f"{sec_prefix}_Close"] <vix_high & (df[f"{sec_prefix}_Close"].shift(1) >= vix_high)
    cond_vix_crossing_vix_low_above = df[f"{sec_prefix}_Close"] <vix_low & (df[f"{sec_prefix}_Close"].shift(1) >= vix_low)

    
    sig = pd.Series(0, index=df.index)
    #If Over 200D and Vix Under 20 then Buy
    sig.loc[cond_price_above_200 & cond_vix_below_vix_low] = 1
    mask_below_200 = ~cond_price_above_200
    #If Under 200D but Vix crosses vix_high then buy
    sig.loc[mask_below_200 & cond_vix_crossing_vix_high_above] = 1
    #if Under200D but VIX is under 20 then sell
    sig.loc[mask_below_200 & cond_vix_crossing_vix_low_above] = 0
    
    return sig.ffill().fillna(0).astype(int)


CONFIG = {
    "primary_ticker": "ZWB.TO/SPY",
    "secondary_tickers": ["^VIX", "TLT"],
    "start_date":    dt.datetime(2019, 1, 1),
    "end_date":      dt.datetime.now(),
    "initial_cap":   1_000_000.0,
    "leverage":      3.0,
    "trade_fee":     0.0,
    "strategy_func": example_multi_ticker_strategy,  # Set your strategy function here
}


def sanitize_ticker(ticker: str) -> str:
    return re.sub(r"\W+", "", ticker)

# --------------------------- FETCH MULTIPLE TICKERS --------------------------- #
def fetch_multiple_tickers(tickers: list, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    merged = None
    for ticker in tickers:
        prefix = sanitize_ticker(ticker)
        df = yf.Ticker(ticker, session=session).history(start=start, end=end).tz_localize(None)
        df_t = pd.DataFrame(index=df.index)
        # Include OHLC and Volume for indicator calculations
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df_t[f"{prefix}_{col}"] = df[col]
        merged = df_t if merged is None else merged.join(df_t, how="inner")
    return merged

# --------------------------- BACKTEST FUNCTIONS --------------------------- #
def backtest(df: pd.DataFrame, positions: pd.Series, initial_cap: float, fee: float, leverage: float) -> pd.DataFrame:
    df = df.copy()
    df["Position"] = positions
    df["PctRet"]   = df["Close"].pct_change().fillna(0.0)
    df["StratRet"] = df["Position"].shift(1) * df["PctRet"] * leverage
    df["Equity"]   = initial_cap * (1 + df["StratRet"]).cumprod()
    trades    = df["Position"].diff().abs()
    df["Equity"] = df["Equity"] - trades * fee
    return df


def plot_backtest(results: pd.DataFrame, config: dict) -> plt.Figure:
    # Year-over-year change (252 trading days)
    results["Stock_YoY"] = results["Close"].pct_change(252)
    results["Port_YoY"] = results["Equity"].pct_change(252)

    # Calculate benchmark and drawdown
    if "TotalRet" not in results:
        results["TotalRet"] = results["Equity"] / config["initial_cap"] - 1
    results["PeakEquity"] = results["Equity"].cummax()
    results["Drawdown"]   = results["Equity"] / results["PeakEquity"] - 1
    results["PctRet"]          = results["Close"].pct_change().fillna(0.0)
    results["BenchmarkEquity"] = config["initial_cap"] * (1 + results["PctRet"]).cumprod()
    results["StockTotalRet"]   = results["BenchmarkEquity"] / config["initial_cap"] - 1
    results["ExcessRet"] = results["Equity"] / results["BenchmarkEquity"] - 1

    # Plot layout: price, volume, YoY, total return, excess, drawdown
    fig, (ax_price, ax_vol, ax_yoy, ax_tr, ax_excess, ax_dd) = plt.subplots(
        6, 1,
        figsize=(12, 18),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1]}
    )

    # Subplot 1: Price + Buy/Sell
    entry_ix = results.index[(results["Position"] == 1) & (results["Position"].shift(1).fillna(0) == 0)]
    exit_ix  = results.index[(results["Position"] == 0) & (results["Position"].shift(1).fillna(1) == 1)]
    if results["Position"].iloc[-1] == 1:
        exit_ix = exit_ix.append(pd.Index([results.index[-1]]))

    ax_price.plot(results.index, results["Close"], label="Close", alpha=0.9)
    ax_price.plot(results.index, results["SMA50"], label="SMA50", alpha=0.5)
    ax_price.plot(results.index, results["SMA200"], label="SMA200", alpha=0.5)
    ax_price.scatter(entry_ix, results.loc[entry_ix, "Close"], marker="^", s=50, color="green", label="Buy")
    ax_price.scatter(exit_ix[:-1], results.loc[exit_ix[:-1], "Close"], marker="v", s=50, color="red", label="Sell")
    last_price = results["Close"].iloc[-1]
    ax_price.text(results.index[-1], last_price,f"{last_price:.2f}", va="bottom", ha="right")
    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(f"{config['primary_ticker']} – Price & Signals")
    ax_price.grid(True)
    ax_price.legend(loc="upper left")

    # Subplot 2: Volume
    ax_vol.bar(results.index, results["Volume"], label="Volume")
    last_vol = results["Volume"].iloc[-1]
    ax_vol.text(results.index[-1], last_vol,f"{int(last_vol):,}", va="bottom", ha="right")
    ax_vol.set_ylabel("Volume")
    ax_vol.set_title("Volume")
    ax_vol.grid(True)

    # Subplot 3: Year-over-Year Change
    ax_yoy.plot(results.index, results["Stock_YoY"] * 100, label="Stock YoY %")
    ax_yoy.plot(results.index, results["Port_YoY"] * 100, label="Port YoY %") 
    last_stock_YoY = results["Stock_YoY"].iloc[-1] * 100
    ax_yoy.text(results.index[-1], last_stock_YoY,f"{last_stock_YoY:.2f}%", va="bottom", ha="right")
    last_port_YoY = results["Port_YoY"].iloc[-1] * 100
    ax_yoy.text(results.index[-1], last_port_YoY,f"{last_port_YoY:.2f}%", va="bottom", ha="right")
    ax_yoy.set_ylabel("YoY (%)")
    ax_yoy.set_title("Year-over-Year Change")
    ax_yoy.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_yoy.grid(True)
    ax_price.legend(loc="upper left")

    # Subplot 4: Total Return Comparison
    strat_leveraged = results["TotalRet"] * config["leverage"] * 100
    stock_leveraged = results["StockTotalRet"] * config["leverage"] * 100
    ax_tr.plot(results.index, strat_leveraged, label=f"Strategy Ret ×{config['leverage']} (%)", color="green")
    ax_tr.plot(results.index, stock_leveraged, label=f"Buy-&-Hold Ret ×{config['leverage']} (%)", color="blue", linestyle="--")
    strat_leveraged = results["TotalRet"] * config["leverage"] * 100
    stock_leveraged = results["StockTotalRet"] * config["leverage"] * 100
    ax_tr.text(results.index[-1], strat_leveraged.iloc[-1],               f"{strat_leveraged.iloc[-1]:.2f}%", va="bottom", ha="right")
    ax_tr.text(results.index[-1], stock_leveraged.iloc[-1],               f"{stock_leveraged.iloc[-1]:.2f}%", va="top", ha="right")
    ax_tr.set_ylabel("Total Return (%)")
    ax_tr.set_title("Total Return: Strategy vs. Buy-and-Hold")
    ax_tr.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_tr.grid(True)
    ax_tr.legend(loc="upper left")

    # Subplot 5: Excess Return
    ax_excess.plot(results.index, results["ExcessRet"] * 100, label="Excess Ret (%)", color="teal")
    ax_excess.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    last_excess = results["ExcessRet"].iloc[-1] * 100
    ax_excess.text(results.index[-1], last_excess,               f"{last_excess:.2f}%", va="bottom", ha="right")
    ax_excess.set_ylabel("Excess Return (%)")
    ax_excess.set_title("Excess Return")
    ax_excess.grid(True)
    ax_excess.legend(loc="upper left")

    # Subplot 6: Drawdown
    ax_dd.plot(results.index, results["Drawdown"] * 100, label="Drawdown (%)", color="orange")
    last_dd = results["Drawdown"].iloc[-1] * 100
    ax_dd.text(results.index[-1], last_dd,           f"{last_dd:.2f}%", va="bottom", ha="right")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_title("Equity Drawdown")
    ax_dd.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_dd.grid(True)
    ax_dd.legend(loc="upper left")

    plt.xlabel("Date")
    plt.tight_layout()
    return fig

# -------------------- MAIN SCRIPT -------------------- #
if __name__ == "__main__":
    raw_primary = CONFIG["primary_ticker"]
    secondaries = CONFIG["secondary_tickers"]
    
        # 1) If primary is a spread like "QQQ/SPY", split it and fetch both legs:
    if "/" in raw_primary:
        base_tkr, quote_tkr = raw_primary.split("/")
        # make sure we still fetch your other secondaries too
        all_tickers = [base_tkr, quote_tkr] + secondaries
        # fetch raw OHLCV for both legs + your usual secondaries
        prices_raw = fetch_multiple_tickers(all_tickers,
                                            CONFIG["start_date"] - timedelta(days=300),
                                            CONFIG["end_date"]).dropna()

        # compute a sanitized prefix for the spread
        base_pre   = sanitize_ticker(base_tkr)
        quote_pre  = sanitize_ticker(quote_tkr)
        spread_pre = sanitize_ticker(raw_primary)  # e.g. "QQQSPY"

        # 2) build the spread series in-place on prices_raw
        for col in ["Open","High","Low","Close"]:
            prices_raw[f"{spread_pre}_{col}"] = (
                prices_raw[f"{base_pre}_{col}"] /
                prices_raw[f"{quote_pre}_{col}"]
            )
        # for Volume you can pick one leg (e.g. the base ticker):
        prices_raw[f"{spread_pre}_Volume"] = prices_raw[f"{base_pre}_Volume"]

        primary_prefix = spread_pre

    else:
        # no slash → the old behavior
        all_tickers    = [raw_primary] + secondaries
        prices_raw     = fetch_multiple_tickers(all_tickers,
                                                CONFIG["start_date"] - timedelta(days=300),
                                                CONFIG["end_date"]).dropna()
        primary_prefix = sanitize_ticker(raw_primary)

    # Prepare multi_df for strategy
    multi_df = prices_raw.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        multi_df[col] = prices_raw[f"{primary_prefix}_{col}"]
    # Generate signals
    positions = CONFIG["strategy_func"](multi_df)
    # Build primary-only prices and run backtest
    prices = pd.DataFrame(
        {col: prices_raw[f"{primary_prefix}_{col}"] for col in ["Open","High","Low","Close","Volume"]},
        index=prices_raw.index
    )
    results = backtest(prices, positions,
                       CONFIG["initial_cap"], CONFIG["trade_fee"], CONFIG["leverage"]).dropna()
    # Ensure SMAs for backtest chart
    results["SMA50"] = ta.sma(results["Close"], length=50)
    results["SMA200"] = ta.sma(results["Close"], length=200)

    # Compute page 2 indicators
    sma20 = ta.sma(results["Close"], length=20)
    sma50 = ta.sma(results["Close"], length=50)
    sma500 = ta.sma(results["Close"], length=500)
    macd = ta.macd(results["Close"])
    bias = ta.bias(results["Close"], length=14)
    bop = ta.bop(results["Open"], results["High"], results["Low"], results["Close"])
    roc = ta.roc(results["Close"], length=10)
    mom = ta.mom(results["Close"], length=10)
    stoch_rsi = ta.stochrsi(results["Close"])
    sqz = ta.squeeze(results["High"], results["Low"], results["Close"], lazybear=True)

    # Compute page 3 indicators
    vol = results["Volume"]
    vol_z = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    vwma20 = ta.vwma(results["Close"], results["Volume"], length=20)
    vwma50 = ta.vwma(results["Close"], results["Volume"], length=50)
    vwma200 = ta.vwma(results["Close"], results["Volume"], length=200)
    adosc = ta.adosc(results["High"], results["Low"], results["Close"], results["Volume"])
    obv = ta.obv(results["Close"], results["Volume"])
    cmf = ta.cmf(results["High"], results["Low"], results["Close"], results["Volume"])
    fidx = results["Close"].diff() * results["Volume"]
    nvi = ta.nvi(results["Close"], results["Volume"])
    vpt = (results["Close"].diff() / results["Close"].shift(1) * results["Volume"]).cumsum()

    # Create report
    with PdfPages("multi_ticker_strategy_report.pdf") as pdf:
        # Page 1
        fig1 = plot_backtest(results, CONFIG)
        pdf.savefig(fig1); plt.close(fig1)

                # Page 2: Momentum Indicators (8 subplots)
        fig2, axs2 = plt.subplots(8, 1, figsize=(12, 32), sharex=True)
        # Subplot 1: Price + SMAs
        ax0 = axs2[0]
        for series, label in [(results["Close"], "Price"), (sma20, "SMA20"), (sma50, "SMA50"), (sma500, "SMA500")]:
            line, = ax0.plot(results.index, series, label=label)
            y = series.iloc[-1]
            ax0.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax0.legend(); ax0.set_title("Price & SMAs"); ax0.grid(True)
        # Subplot 2: MACD (MACD line & Signal)
        ax1 = axs2[1]
        line_macd, = ax1.plot(results.index, macd[macd.columns[0]], label="MACD")
        line_sig, = ax1.plot(results.index, macd[macd.columns[1]], label="Signal")
        for line, series in [(line_macd, macd[macd.columns[0]]), (line_sig, macd[macd.columns[1]])]:
            y = series.iloc[-1]
            ax1.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax1.legend(); ax1.set_title("MACD"); ax1.grid(True)
        # Subplot 3: BIAS
        ax2 = axs2[2]
        line, = ax2.plot(results.index, bias, label="BIAS")
        y = bias.iloc[-1]; ax2.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax2.legend(); ax2.set_title("Bias"); ax2.grid(True)
        # Subplot 4: Balance of Power
        ax3 = axs2[3]
        line, = ax3.plot(results.index, bop, label="BOP")
        y = bop.iloc[-1]; ax3.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax3.legend(); ax3.set_title("Balance of Power"); ax3.grid(True)
        # Subplot 5: Rate of Change Momentum
        ax4 = axs2[4]
        line, = ax4.plot(results.index, roc, label="ROC")
        y = roc.iloc[-1]; ax4.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax4.legend(); ax4.set_title("Rate of Change Momentum"); ax4.grid(True)
        # Subplot 6: Momentum (MOM)
        ax5 = axs2[5]
        line, = ax5.plot(results.index, mom, label="MOM")
        y = mom.iloc[-1]; ax5.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax5.legend(); ax5.set_title("Momentum"); ax5.grid(True)
        # Subplot 7: Stochastic RSI
        ax6 = axs2[6]
        line, = ax6.plot(results.index, stoch_rsi[stoch_rsi.columns[0]], label="StochRSI")
        y = stoch_rsi.iloc[-1, 0]; ax6.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax6.legend(); ax6.set_title("Stochastic RSI"); ax6.grid(True)
        # Subplot 8: Squeeze (Lazybear)
        ax7 = axs2[7]
        line, = ax7.plot(results.index, sqz[sqz.columns[0]], label="Squeeze")
        y = sqz.iloc[-1, 0]; ax7.text(results.index[-1], y, f"{y:.2f}", color=line.get_color(), va="bottom", ha="right")
        ax7.legend(); ax7.set_title("Squeeze (Lazybear)"); ax7.grid(True)
        plt.xlabel("Date"); plt.tight_layout()
        pdf.savefig(fig2); plt.close(fig2)

        # Page 3: Volume
        fig3, axs3 = plt.subplots(8,1,figsize=(12,32),sharex=True)
        # Volume & Z-score
        ax0=axs3[0]
        ax0.bar(results.index,vol,label="Volume")
        zline,=ax0.twinx().plot(results.index,vol_z,label="Z-Score",color="red")
        y=vol_z.iloc[-1]; ax0.twinx().text(results.index[-1],y,f"{y:.2f}",color=zline.get_color(),va="bottom",ha="right")
        ax0.legend(loc="upper left"); ax0.set_title("Volume & Z-Score"); ax0.grid(True)
        # VWMA
        ax1=axs3[1]
        for s,label in [(vwma20,"VWMA20"),(vwma50,"VWMA50"),(vwma200,"VWMA200")]:
            line,=ax1.plot(results.index,s,label=label)
            y=s.iloc[-1]; ax1.text(results.index[-1],y,f"{y:.2f}",color=line.get_color(),va="bottom",ha="right")
        ax1.legend(); ax1.set_title("VWMA"); ax1.grid(True)
        # Other volume indicators
        vols = [(adosc,"ADOSC"),(obv,"OBV"),(cmf,"CMF"),(fidx,"Force Index"),(nvi,"NVI"),(vpt,"VPT")]
        for i,(ser,label) in enumerate(vols, start=2):
            ax=axs3[i]
            line,=ax.plot(results.index,ser,label=label)
            y=ser.iloc[-1]; ax.text(results.index[-1],y,f"{y:.2f}",color=line.get_color(),va="bottom",ha="right")
            ax.legend(); ax.set_title(label); ax.grid(True)
        plt.xlabel("Date"); plt.tight_layout()
        pdf.savefig(fig3); plt.close(fig3)

                # Pages for each secondary ticker
        for ticker in CONFIG["secondary_tickers"]:
            prefix = sanitize_ticker(ticker)
            # Build secondary DataFrame for plotting
            df_sec = pd.DataFrame(index=prices_raw.index)
            df_sec["Close"] = prices_raw[f"{prefix}_Close"]
            df_sec["Volume"] = prices_raw[f"{prefix}_Volume"]
            # Calculate moving averages
            ma_lengths = [50, 200]
            ma_data = {f"SMA{l}": ta.sma(df_sec["Close"], length=l) for l in ma_lengths}
            fig_sec, axs_sec = plt.subplots(3, 1, figsize=(12, 16), sharex=True,
                                           gridspec_kw={"height_ratios": [2, 1, 1]})
            ax_c, ax_m, ax_v = axs_sec
            # Price subplot
            line_c, = ax_c.plot(df_sec.index, df_sec["Close"], label="Close")
            y_c = df_sec["Close"].iloc[-1]
            ax_c.text(df_sec.index[-1], y_c, f"{y_c:.2f}", color=line_c.get_color(), va="bottom", ha="right")
            ax_c.set_title(f"{ticker} Price"); ax_c.legend(); ax_c.grid(True)
            # Moving Averages subplot
            for name, series in ma_data.items():
                line_ma, = ax_m.plot(df_sec.index, series, label=name)
                y_ma = series.iloc[-1]
                ax_m.text(df_sec.index[-1], y_ma, f"{y_ma:.2f}", color=line_ma.get_color(), va="bottom", ha="right")
            ax_m.set_title("Moving Averages"); ax_m.legend(); ax_m.grid(True)
            # Volume subplot
            line_v = ax_v.bar(df_sec.index, df_sec["Volume"], label="Volume")
            y_v = df_sec["Volume"].iloc[-1]
            ax_v.text(df_sec.index[-1], y_v, f"{y_v:,}", va="bottom", ha="right")
            ax_v.set_title(f"{ticker} Volume"); ax_v.legend(); ax_v.grid(True)
            plt.xlabel("Date"); plt.tight_layout()
            pdf.savefig(fig_sec); plt.close(fig_sec)

        # Final Page: Trade Log Table
        df_trades = results[["Close", "Position"]].copy()
        pos = df_trades["Position"]
        pos_shift = pos.shift(1).fillna(0)
        entries = df_trades.index[(pos == 1) & (pos_shift == 0)]
        exits = df_trades.index[(pos == 0) & (pos_shift == 1)]
        if len(entries) > 0 and results["Position"].iloc[-1] == 1:
            exits = exits.append(pd.Index([results.index[-1]]))
        trades_list = []
        for date in entries:
            trades_list.append({"Date": date, "Direction": "Buy"})
        for date in exits:
            trades_list.append({"Date": date, "Direction": "Sell"})
        df_log = pd.DataFrame(trades_list)
        # Sort descending by Date and take last 30
        df_log = df_log.sort_values("Date", ascending=False).head(30)
        # Format Date for display
        df_log["Date"] = df_log["Date"].dt.strftime("%Y-%m-%d")
        fig_log, ax_log = plt.subplots(figsize=(8.5, 11))
        ax_log.axis("off")
        table = ax_log.table(cellText=df_log.values, colLabels=df_log.columns,
                              loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
        fig_log.subplots_adjust(top=0.85)
        fig_log.text(0.5, 0.93, "Trade Log (Last 30 Trades)", ha="center", va="center", fontsize=14)
        pdf.savefig(fig_log); plt.close(fig_log)

    print("PDF saved as 'multi_ticker_strategy_report.pdf'")