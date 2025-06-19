import re
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product


# --------------------------- CONFIGURATION --------------------------- #
# Helps get around Rate Limit Requests
from curl_cffi import requests
session = requests.Session(impersonate="chrome")


# --------------------------- CONFIGURATION --------------------------- #

def sanitize_ticker(ticker: str) -> str:
    """
    Remove non-alphanumeric characters to use as a column prefix.
    E.g. '^VIX' → 'VIX', 'BRK-B' → 'BRKB'
    """
    return re.sub(r"\W+", "", ticker)

# --------------------------- FETCH MULTI-TICKER DATA --------------------------- #

def fetch_multiple_tickers(tickers: list, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """
    Download daily history (Close, Volume if available) for each ticker in `tickers`,
    renaming columns to "<PREFIX>_Close" and "<PREFIX>_Volume" (when available),
    and inner-join all into one DataFrame indexed by date.
    """
    merged = None
    for ticker in tickers:
        prefix = sanitize_ticker(ticker)
        df_raw = yf.Ticker(ticker, session = session).history(start=start, end=end).tz_localize(None)
        df_t = pd.DataFrame(index=df_raw.index)
        df_t[f"{prefix}_Close"] = df_raw["Close"]
        if "Volume" in df_raw.columns:
            df_t[f"{prefix}_Volume"] = df_raw["Volume"]
        # ── NEW: include dividends if available
        if "Dividends" in df_raw.columns:
            df_t[f"{prefix}_Dividends"] = df_raw["Dividends"]
        if merged is None:
            merged = df_t
        else:
            merged = merged.join(df_t, how="inner")
    return merged

# --------------------------- BACKTEST LOGIC --------------------------- #

def backtest(df: pd.DataFrame,
             positions: pd.Series,
             initial_cap: float,
             fee: float,
             leverage: float,
             div_tax: float = 0.0,
             cap_tax: float = 0.0) -> pd.DataFrame:

    df = df.copy()
    df["Position"] = positions
    df["PctRet"]   = df["Close"].pct_change().fillna(0.0)

    # ── DIVIDEND LOGIC ──
    # 1) If the raw Dividends series exists, compute yield from it
    if "Dividends" in df.columns:
        df["Dividend"] = df["Dividends"].fillna(0.0)
        df["DivYield"] = df["Dividend"] / df["Close"].shift(1)
    # 2) Else if a precomputed DivYield exists (your spread), use it
    elif "DivYield" in df.columns:
        df["DivYield"] = df["DivYield"].fillna(0.0)
        # also rebuild a notional "Dividend" series so plotting still works
        df["Dividend"] = df["DivYield"] * df["Close"].shift(1).fillna(0.0)
    # 3) Otherwise no dividends
    else:
        df["Dividend"] = 0.0
        df["DivYield"] = 0.0

    # after-tax dividend return
    df["DivRetAT"] = df["DivYield"] * (1.0 - div_tax)

    # ── REST OF YOUR BACKTEST (unchanged) ──
    df["Turnover"] = df["Position"].diff().abs().fillna(0.0)
    df["StratRet"] = (
        df["Position"].shift(1)
        * (df["PctRet"] * leverage + df["DivRetAT"] * leverage)
        - df["Turnover"] * fee
    )
    df["Equity"]   = initial_cap * (1 + df["StratRet"]).cumprod()
    
    
    # ── flat capital-gains tax on each round-trip exit
    eq = df["Equity"].copy()
    entry_eq = initial_cap
    in_trade = False

    for i in range(1, len(df)):
        prev_pos = df["Position"].iat[i-1]
        pos      = df["Position"].iat[i]
        # detect entry
        if not in_trade and prev_pos == 0 and pos == 1:
            entry_eq = eq.iat[i]
            in_trade = True
            # detect exit
        elif in_trade and prev_pos == 1 and pos == 0:
            gain = eq.iat[i] - entry_eq
            tax  = gain * cap_tax
            eq.iloc[i:] = eq.iloc[i:] - tax
            in_trade = False
    df["Equity"] = eq
    
    trades    = df["Position"].diff().abs()
    fees_paid = trades * fee
    df["Equity"] = df["Equity"] - fees_paid.cumsum()
    return df

# --------------------------- STRATEGY & METRICS --------------------------- #

def generate_vix_sma_signal(
    df: pd.DataFrame,
    sma_short: int,
    sma_med: int,
    vix_low: float,
    vix_high: float
) -> pd.Series:
    """
    Combined SMA & VIX-based entry/exit rules:
      • Compute SMA_short, SMA_med, SMA200 on primary ticker’s Close.
      • Let cond_price_above_200 = Close > SMA200.
      • Let cond_sma_short_above_med = SMA_short > SMA_med.
      • VIX thresholds:
          - cond_vix_below_low = VIX_Close < vix_low
          - cond_vix_cross_up_high = (VIX_Close > vix_high) & (VIX_Close.shift(1) <= vix_high)
          - cond_vix_cross_down_low = (VIX_Close < vix_low) & (VIX_Close.shift(1) >= vix_low)
      • Entry:
          - If cond_price_above_200 & cond_vix_below_low → signal = 1
          - Else if (not cond_price_above_200) & cond_vix_cross_up_high → signal = 1
          - Else if cond_price_above_200 & cond_sma_short_above_med → signal = 1
      • Exit:
          - If (not cond_price_above_200) & cond_vix_cross_down_low → signal = 0
          - Else if cond_price_above_200 & (not cond_sma_short_above_med) → signal = 0
      • Forward-fill and fill initial NaN with 0.
    """
    # Compute SMAs
    df[f"SMA{sma_short}"] = ta.sma(df["Close"], length=sma_short)
    df[f"SMA{sma_med}"]   = ta.sma(df["Close"], length=sma_med)
    df["SMA200"]          = ta.sma(df["Close"], length=200)

    # Boolean masks for SMA rules
    cond_price_above_200     = df["Close"] > df["SMA200"]
    cond_sma_short_above_med = df[f"SMA{sma_short}"] > df[f"SMA{sma_med}"]

    # VIX calculations
    sec_prefix = sanitize_ticker("^VIX")
    vix         = df[f"{sec_prefix}_Close"]
    cond_vix_below_low      = vix < vix_low
    cond_vix_cross_up_high  = (vix > vix_high) & (vix.shift(1) <= vix_high)
    cond_vix_cross_down_high  = (vix < vix_high) & (vix.shift(1) >= vix_high)
    cond_vix_cross_up_low  = (vix > vix_low) & (vix.shift(1) <= vix_low)
    cond_vix_cross_down_low = (vix < vix_low) & (vix.shift(1) >= vix_low)

    sig = pd.Series(0, index=df.index)

    # Entry conditions
    sig.loc[cond_price_above_200 & cond_vix_below_low] = 1                      #Normal - Above 200D and VIX Below Threshold
    sig.loc[cond_vix_cross_up_high] = 1                                         #Buying Fear - Buy 40 Vix
    sig.loc[(~cond_price_above_200) & cond_sma_short_above_med] = 1             #When Below 200D Buy when SMA 5D > SMA 20D 

    # Exit conditions
    sig.loc[(~cond_price_above_200) & cond_vix_below_low] = 0
    sig.loc[cond_vix_cross_up_low] = 0
    #sig.loc[cond_price_above_200 & (~cond_sma_short_above_med)] = 0

    return sig.ffill().fillna(0).astype(int)

def evaluate_vix_sma_strategy(prices: pd.DataFrame,initial_cap: float,fee: float,leverage: float,div_tax: float,cap_tax: float,sma_short: int,sma_med: int,vix_low: float,vix_high: float) -> dict:
    """
    Given a set of parameters, compute:
      - total_return, annual_vol, max_drawdown, sharpe_ratio, sortino_ratio
      of the backtest over `prices`, plus the final position.
    Uses the backtest’s 'Equity' series as the portfolio value. 
    """
    # 1) Signals & last position
    df        = prices.copy()
    positions = generate_vix_sma_signal(df, sma_short, sma_med, vix_low, vix_high)
    last_sig  = int(positions.iloc[-1])
    last_pos  = "Buy" if last_sig == 1 else "Sell"

    # 2) Run backtest (produces an 'Equity' column) :contentReference[oaicite:0]{index=0}
    results = backtest(df, positions,initial_cap, fee, leverage,div_tax, cap_tax).dropna()

    # 3) Handle too–short results
    if results.empty or len(results) < 2:
        return {
            "leverage":      leverage,
            "sma_short":     sma_short,
            "sma_med":       sma_med,
            "vix_low":       vix_low,
            "vix_high":      vix_high,
            "total_return":  0.0,
            "annual_vol":    0.0,
            "max_drawdown":  0.0,
            "sharpe_ratio":  0.0,
            "sortino_ratio": 0.0,
            "last_position": last_pos
        }

    # 4) Compute metrics off 'Equity' & 'StratRet' :contentReference[oaicite:1]{index=1}
    total_ret = results["Equity"].iloc[-1] / initial_cap - 1
    daily_rets = results["StratRet"].dropna()
    ann_vol   = daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0.0
    peak_eq   = results["Equity"].cummax()
    max_dd    = (results["Equity"] / peak_eq - 1).min()
    sharpe    = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                 if daily_rets.std() > 0 else 0.0)
    neg       = daily_rets[daily_rets < 0]
    sortino   = (daily_rets.mean() / neg.std() * np.sqrt(252)
                 if not neg.empty and neg.std() > 0 else 0.0)

    # 5) Return all metrics + last position
    return {
        "leverage":      leverage,
        "sma_short":     sma_short,
        "sma_med":       sma_med,
        "vix_low":       vix_low,
        "vix_high":      vix_high,
        "total_return":  total_ret,
        "annual_vol":    ann_vol,
        "max_drawdown":  max_dd,
        "sharpe_ratio":  sharpe,
        "sortino_ratio": sortino,
        "last_position": last_pos
    }
# ----------------------- PLOTTING FUNCTION ----------------------- #

def plot_backtest(results: pd.DataFrame, config: dict, setup_desc: str) -> plt.Figure:
    """
    Create a 7-panel figure:
      1) Primary price + buy/sell + Stock Total Return (%)
      2) Volume (stock) shaded when invested
      3) SMAs shaded when invested
      4) VIX Close shaded when invested
      5) Total Return Comparison
      6) Excess Return
      7) Drawdown
    """
    # Unpack positions and compute entry/exit points
    pos = results["Position"]
    pos_shift = pos.shift(1).fillna(0)
    entry_ix = results.index[(pos == 1) & (pos_shift == 0)]
    exit_ix = results.index[(pos == 0) & (pos_shift == 1)]
    if pos.iloc[-1] == 1:
        exit_ix = exit_ix.append(pd.Index([results.index[-1]]))

    # Compute derived series if missing
    if "TotalRet" not in results:
        results["TotalRet"] = results["Equity"] / config["initial_cap"] - 1
    results["PeakEquity"] = results["Equity"].cummax()
    results["Drawdown"] = results["Equity"] / results["PeakEquity"] - 1
    results["PctRet"] = results["Close"].pct_change().fillna(0.0)
    results["BenchmarkEquity"] = config["initial_cap"] * (1 + results["PctRet"]).cumprod()
    results["StockTotalRet"] = results["BenchmarkEquity"] / config["initial_cap"] - 1
    results["ExcessRet"] = results["Equity"] / results["BenchmarkEquity"] - 1

    # Prepare subplots: 7 rows, shared x-axis
    fig, axes = plt.subplots(
        7, 1,
        figsize=(12, 20),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1, 1, 1, 1, 1]}
    )
    ax_price, ax_vol, ax_sma, ax_vix, ax_tr, ax_excess, ax_dd = axes

    last_date = results.index[-1]

    # ── Subplot 1: Price + buy/sell ──
    ax_price.plot(results.index, results["Close"], label="Close", alpha=0.7)
    ax_price.set_ylabel("Price ($)")
    ax_price.scatter(entry_ix, results.loc[entry_ix, "Close"], marker="^", s=80, color="green", label="Buy")
    ax_price.scatter(exit_ix[:-1], results.loc[exit_ix[:-1], "Close"], marker="v", s=80, color="red", label="Sell")
    ax_price.set_title(f"{config['primary_ticker']} – {setup_desc}")
    ax_price.grid(True)
    # annotate last price
    last_price = results["Close"].iloc[-1]
    ax_price.annotate(f"{last_price:.2f}", xy=(last_date, last_price), xytext=(5, 0),textcoords="offset points", va="center", ha="left")
    
    # ── NEW: pure hold total-return (price + dividends) ──
    tr_idx = (results["Close"] * (1 + results["DivYield"] + results["PctRet"]))
    # or rebuild from 1.0:
    tr_idx = (1 + results["PctRet"] + results["DivYield"]).cumprod() * results["Close"].iloc[0]

    # ── NEW: after-tax hold total-return (div tax only; no cap-gains) ──
    tr_at_idx = (1 + results["PctRet"] + results["DivYield"] * (1 - div_tax)).cumprod()* results["Close"].iloc[0]
    ax_price.plot(results.index, tr_at_idx, label="Hold After-Tax Total Return", linestyle=":", alpha=0.8)

    # ── Subplot 2: Volume (stock) ──
    ax_vol.plot(results.index, results["Volume"], label="Stock Volume", alpha=0.6)
    for s, e in zip(entry_ix, exit_ix):
        ax_vol.axvspan(s, e, facecolor="green", alpha=0.1)
    ax_vol.set_ylabel("Volume")
    ax_vol.set_title("Volume: Stock")
    ax_vol.legend(loc="upper left", fontsize=8)
    ax_vol.grid(True)
    # annotate last volume
    last_vol = results["Volume"].iloc[-1]
    ax_vol.annotate(f"{int(last_vol)}", xy=(last_date, last_vol), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left")

    # ── Subplot 3: SMAs ──
    ss = config["setup_params"]["sma_short"]
    sm = config["setup_params"]["sma_med"]
    ax_sma.plot(results.index, results[f"SMA{ss}"], label=f"SMA{ss}", alpha=0.8)
    ax_sma.plot(results.index, results[f"SMA{sm}"], label=f"SMA{sm}", alpha=0.8)
    ax_sma.plot(results.index, results["SMA200"], label="SMA200", alpha=0.8)
    for s, e in zip(entry_ix, exit_ix):
        ax_sma.axvspan(s, e, facecolor="green", alpha=0.1)
    ax_sma.set_ylabel("SMA")
    ax_sma.set_title("SMAs")
    ax_sma.legend(loc="upper left", fontsize=8)
    ax_sma.grid(True)
    # annotate last SMAs
    last_sma_short = results[f"SMA{ss}"].iloc[-1]
    ax_sma.annotate(f"{last_sma_short:.2f}", xy=(last_date, last_sma_short), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left")
    last_sma_med = results[f"SMA{sm}"].iloc[-1]
    ax_sma.annotate(f"{last_sma_med:.2f}", xy=(last_date, last_sma_med), xytext=(5, -10),
                    textcoords="offset points", va="center", ha="left")
    last_sma_200 = results["SMA200"].iloc[-1]
    ax_sma.annotate(f"{last_sma_200:.2f}", xy=(last_date, last_sma_200), xytext=(5, -20),
                    textcoords="offset points", va="center", ha="left")

    # ── Subplot 4: VIX Close ──
    sec_prefix = sanitize_ticker("^VIX")
    vix = results[f"{sec_prefix}_Close"]
    ax_vix.plot(results.index, vix, label="VIX Close", alpha=0.8)
    for s, e in zip(entry_ix, exit_ix):
        ax_vix.axvspan(s, e, facecolor="green", alpha=0.1)
    ax_vix.set_ylabel("VIX")
    ax_vix.set_title("VIX Index")
    ax_vix.legend(loc="upper left", fontsize=8)
    ax_vix.grid(True)
    # annotate last VIX
    last_vix = vix.iloc[-1]
    ax_vix.annotate(f"{last_vix:.2f}", xy=(last_date, last_vix), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left")

    # ── Subplot 5: Total Return Comparison ──
    strat_ret = results["TotalRet"] * 100
    bh_ret = results["StockTotalRet"] * config["leverage"] * 100
    ax_tr.plot(results.index, strat_ret, label="Strategy Total Ret (%)", color="green", alpha=0.8)
    ax_tr.plot(results.index, bh_ret, label=f"Stock Total Ret ×{config['leverage']} (%)", color="blue",
               linestyle="--", alpha=0.8)
    ax_tr.set_ylabel("Total Return (%)")
    ax_tr.set_title("Total Return: Strategy vs. Buy-and-Hold (leveraged)")
    ax_tr.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_tr.grid(True)
    ax_tr.legend(loc="upper left", fontsize=8)
    # annotate last returns
    ax_tr.annotate(f"{strat_ret.iloc[-1]:.2f}%", xy=(last_date, strat_ret.iloc[-1]), xytext=(5, 0),
                   textcoords="offset points", va="center", ha="left")
    ax_tr.annotate(f"{bh_ret.iloc[-1]:.2f}%", xy=(last_date, bh_ret.iloc[-1]), xytext=(5, -10),
                   textcoords="offset points", va="center", ha="left")

    # ── Subplot 6: Excess Return ──
    excess_ret = results["ExcessRet"] * 100
    ax_excess.plot(results.index, excess_ret, label="Excess Return (%)", alpha=0.8)
    ax_excess.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_excess.set_ylabel("Excess Return (%)")
    ax_excess.set_title("Excess Return: Strategy vs. Buy-and-Hold")
    ax_excess.grid(True)
    ax_excess.legend(loc="upper left", fontsize=8)
    # annotate last excess
    ax_excess.annotate(f"{excess_ret.iloc[-1]:.2f}%", xy=(last_date, excess_ret.iloc[-1]), xytext=(5,
                    0), textcoords="offset points", va="center", ha="left")

    # ── Subplot 7: Drawdown ──
    dd = results["Drawdown"] * 100
    ax_dd.plot(results.index, dd, label="Drawdown (%)", alpha=0.8)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_title("Equity Drawdown")
    ax_dd.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_dd.grid(True)
    ax_dd.legend(loc="upper left", fontsize=8)
    # annotate last drawdown
    ax_dd.annotate(f"{dd.iloc[-1]:.2f}%", xy=(last_date, dd.iloc[-1]), xytext=(5, 0),
                   textcoords="offset points", va="center", ha="left")
    plt.xlabel("Date")
    plt.tight_layout()
    return fig

def plot_setup_and_trades(row, setup_name):
    ss  = int(row["sma_short"])
    sm  = int(row["sma_med"])
    vl  = float(row["vix_low"])
    vh  = float(row["vix_high"])
    lev = float(row["leverage"])

    # Recompute signal & backtest
    df_tmp    = prices.copy()
    positions = generate_vix_sma_signal(df_tmp, ss, sm, vl, vh)
    results_bt= backtest(df_tmp,positions,initial_cap,fee,lev,div_tax,cap_tax).dropna()

    # Add the derived columns (SMA200 already exists)
    results_bt["TotalRet"]        = results_bt["Equity"] / initial_cap - 1
    results_bt["PeakEquity"]      = results_bt["Equity"].cummax()
    results_bt["Drawdown"]        = results_bt["Equity"] / results_bt["PeakEquity"] - 1
    results_bt["PctRet"]          = results_bt["Close"].pct_change().fillna(0.0)
    results_bt["BenchmarkEquity"] = initial_cap * (1 + results_bt["PctRet"]).cumprod()
    results_bt["StockTotalRet"]   = results_bt["BenchmarkEquity"] / initial_cap - 1
    results_bt["ExcessRet"]       = results_bt["Equity"] / results_bt["BenchmarkEquity"] - 1

    # Package config for plotting
    cfg = {
        "primary_ticker": primary,
        "initial_cap":    initial_cap,
        "leverage":       lev,
        "setup_params": {
            "leverage":  lev,
            "sma_short": ss,
            "sma_med":   sm,
            "vix_low":   vl,
            "vix_high":  vh
        }
    }
    fig = plot_backtest(results_bt, cfg, setup_name)

    # Generate last 30 trades based on changes in Position
    pos = results_bt["Position"]
    diff = pos.diff().fillna(0)

    trades_list = []
    sec_prefix = sanitize_ticker("^VIX")

    for date, change in diff.items():  # use .items() instead of .iteritems()
        if change == 1:
            direction = "Buy"
            # Determine reason at this entry date
            close_price = results_bt.loc[date, "Close"]
            sma200_val   = results_bt.loc[date, "SMA200"]
            sma_short_val= results_bt.loc[date, f"SMA{ss}"]
            sma_med_val  = results_bt.loc[date, f"SMA{sm}"]
            vix_curr     = prices.loc[date, f"{sec_prefix}_Close"]

            prev_idx = prices.index.get_loc(date) - 1
            if prev_idx >= 0:
                vix_prev = prices.iloc[prev_idx][f"{sec_prefix}_Close"]
            else:
                vix_prev = np.nan

            if close_price > sma200_val and vix_curr < vl:
                reason = f"Price>SMA200 & VIX<{vl}"
            elif (close_price <= sma200_val) and (vix_curr > vh and not np.isnan(vix_prev) and vix_prev <= vh):
                reason = f"VIX crossed above {vh}"
            elif close_price > sma200_val and sma_short_val > sma_med_val:
                reason = f"SMA{ss}>{ss} & SMA{sm}={sm}"
            else:
                reason = "Entry"

            trades_list.append({
                "Date":      date.strftime("%Y-%m-%d"),
                "Direction": direction,
                "Reason":    reason
            })

        elif change == -1:
            direction = "Sell"
            trades_list.append({
                "Date":      date.strftime("%Y-%m-%d"),
                "Direction": direction,
                "Reason":    "Exit"
            })

    df_trades_log = pd.DataFrame(trades_list).tail(30)
    return fig, df_trades_log

# ----------------------- GRID SEARCH & MAIN SCRIPT ----------------------- #

if __name__ == "__main__":
    # 1) Configuration parameters for grid search
    primary     = "ZWB.TO/ZEB.TO"      # ← now supports spreads
    secondaries = ["^VIX"]
    start_date  = dt.datetime(2020, 1, 1)
    end_date    = dt.datetime.now()
    initial_cap = 100_000.0
    fee         = 0.0
    div_tax     = 0.15      # ── dividend tax rate
    cap_tax     = 0.20      # ── flat capital-gains tax rate

    # Grid of parameters
    leverages     = [-1,1,2]#[-1,1,2,3,5,7.5,10]
    sma_shorts    = [0,5, 10, 20,1000]
    sma_meds      = [0,20, 50, 100,1000]
    vix_lows      = [0,17,18,19,20,25,30,100]
    vix_highs     = [0,25,30,35,40, 45,50,55,60,100]

    # Fetching & prepping
    if "/" in primary:
        base_tkr, quote_tkr = primary.split("/")
        # include VIX (and any other secondaries) in one fetch
        all_tickers = [base_tkr, quote_tkr] + secondaries
        prices_raw  = fetch_multiple_tickers(
            all_tickers,
            start_date - timedelta(days=300),
            end_date
        ).dropna()

        # sanitize prefixes
        bp = sanitize_ticker(base_tkr)
        qp = sanitize_ticker(quote_tkr)
        sp = sanitize_ticker(primary)  # e.g. "QQQSPY"

        # build the spread series
        for ohlc in ("Close","Volume"):
            left  = f"{bp}_{ohlc}"
            right = f"{qp}_{ohlc}"
            if ohlc == "Volume":
                # choose one leg’s volume (usually the base ticker)
                prices_raw[f"{sp}_{ohlc}"] = prices_raw[left]
            else:
                prices_raw[f"{sp}_{ohlc}"] = prices_raw[left] / prices_raw[right]
                
        primary_prefix = sp

    else:
        # single-ticker path (unchanged)
        all_tickers    = [primary] + secondaries
        prices_raw     = fetch_multiple_tickers(
            all_tickers,
            start_date - timedelta(days=300),
            end_date
        ).dropna()
        primary_prefix = sanitize_ticker(primary)

    # Rename for backtest: set `Close` and `Volume` columns
    prices = prices_raw.copy()
    prices["Close"]  = prices_raw[f"{primary_prefix}_Close"]
    prices["Volume"] = prices_raw[f"{primary_prefix}_Volume"]
    # now add a single DivYield column for the synthetic spread:
    prices["DivYield"] = (
        # ZWB dividend yield
        prices_raw[f"{bp}_Dividends"].fillna(0)   / prices_raw[f"{bp}_Close"].shift(1)
        # minus ZEB dividend yield (because you’re short that leg)
      - prices_raw[f"{qp}_Dividends"].fillna(0)   / prices_raw[f"{qp}_Close"].shift(1)
    )


    # Grid search evaluation
    results = []
    for lev, ss, sm, vl, vh in product(leverages, sma_shorts, sma_meds, vix_lows, vix_highs):
        if ss >= sm or vl >= vh:
            continue
        metrics = evaluate_vix_sma_strategy(prices, initial_cap, fee,
                                            lev, div_tax, cap_tax, ss, sm, vl, vh)
        results.append(metrics)

    df_results = pd.DataFrame(results).sort_values(["total_return", "max_drawdown"], ascending=[False, True]).reset_index(drop=True)
    # Compute last switch dates
    def last_switch(row):
        ss, sm = int(row["sma_short"]), int(row["sma_med"])
        vl, vh = float(row["vix_low"]), float(row["vix_high"])
        flips = generate_vix_sma_signal(prices.copy(), ss, sm, vl, vh).diff().abs()
        idxs = flips[flips == 1].index
        return idxs[-1].strftime("%Y-%m-%d") if len(idxs) else "N/A"
    df_results["last_switch"] = df_results.apply(last_switch, axis=1)

    # 4) Sort results by total_return desc, then by drawdown asc
    df_sorted = df_results.sort_values(
        ["total_return", "max_drawdown"],
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # ── Compute a "last_switch" date for each parameter combination ──
    def last_switch_date_for(params_row):
        ss, sm = int(params_row["sma_short"]), int(params_row["sma_med"])
        vl, vh = float(params_row["vix_low"]),    float(params_row["vix_high"])
        # regenerate the signal
        sig = generate_vix_sma_signal(prices.copy(), ss, sm, vl, vh)
        # a flip in position is where diff.abs()==1
        flips = sig[sig.diff().abs() == 1].index
        return flips[-1].strftime("%Y-%m-%d") if len(flips) else "N/A"

    # apply over every row
    df_sorted["last_switch"] = df_sorted.apply(last_switch_date_for, axis=1)

    # 5a) Compute SPY’s buy-and-hold return over the same period
    stock_return = prices["Close"].iloc[-1] / prices["Close"].iloc[0] - 1

    # 5b) Filter for strategies that (a) are long-only (lev >= 1) and
    #     (b) beat the buy-and-hold return
    df_drawdown_mask = df_results[
        (df_results["leverage"] >= 1.0) &
        (df_results["total_return"] > stock_return)
    ]

    # 5c) From those, pick the “best” drawdown (i.e. max_drawdown closest to 0)
    best_drawdown = df_drawdown_mask.sort_values(
        "max_drawdown", ascending=False
    ).iloc[0]

    # 5d) Your other selections remain the same
    best_return      = df_sorted.iloc[0]
    best_return_lev1 = df_sorted[df_sorted["leverage"] == 1.0].iloc[0]
    best_return_lev_neg1 = df_sorted[df_sorted["leverage"] == -1.0].iloc[0]

    # 6) Build PDF report
    report_filename = f"{primary_prefix}_vix_sma_optimization_report_{end_date.date()}.pdf"
    with PdfPages(report_filename) as pdf:
        def table_page(df_in, title):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            # Buy % line
            texts = [("Top %d" % N,
                      (df_in.head(N)["last_position"] == "Buy").mean() * 100)
                     for N in [5, 10, 20, 30]]
            line = "Buy % – " + ", ".join([f"Top {N}: {p:.1f}%" for N,p in texts])
            ax.text(0.5, 0.9, line, transform=fig.transFigure,
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.6, pad=1))
            df_disp = df_in.head(30).round(2)
            n = len(df_disp)
            fs = max(6, min(10, int(400/(n+1))))
            cs, rs = 1.1, min(1.0, 30/(n+1))
            hdrs = ["\n".join(c.split("_")) for c in df_disp.columns]
            fig.text(0.5, 0.95, title, ha='center', va='center', fontsize=12)
            tbl = ax.table(cellText=df_disp.values,
                           colLabels=hdrs,
                           loc='center', cellLoc='center')
            tbl.auto_set_font_size(False);
            tbl.set_fontsize(fs);
            tbl.scale(cs, rs)
            for j in range(len(hdrs)): tbl[(0,j)].set_height(0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        table_page(df_results, f"{primary_prefix} VIX+SMA Strategy Optimization\nTop Parameter Combinations")
        table_page(df_results[df_results['leverage']==1.0], "Top Parameter Combinations (Leverage = 1×)")
        table_page(df_results[df_results['leverage']==-1.0], "Top Parameter Combinations (Short -1×)")


        # ---- Page 2 & 3: Highest-return setup & its last 30 trades ----
        fig2, trades2 = plot_setup_and_trades(best_return, "Highest Return")
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig2b, ax2b = plt.subplots(figsize=(8.5, 11))
        ax2b.axis("off")
        ax2b.set_title("Last 30 Trades: Highest Return", fontsize=14)
        table2b = ax2b.table(
            cellText=trades2.values,
            colLabels=trades2.columns,
            loc="upper center",
            cellLoc="center"
        )
        table2b.auto_set_font_size(False)
        table2b.set_fontsize(10)
        table2b.scale(1, 1.5)
        pdf.savefig(fig2b, bbox_inches="tight")
        plt.close(fig2b)

        # ---- Page 4 & 5: Best return with leverage=1 & its trades ----
        fig3, trades3 = plot_setup_and_trades(best_return_lev1, "Best Return (Lev=1)")
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        fig3b, ax3b = plt.subplots(figsize=(8.5, 11))
        ax3b.axis("off")
        ax3b.set_title("Last 30 Trades: Best Return (Lev=1)", fontsize=14)
        table3b = ax3b.table(
            cellText=trades3.values,
            colLabels=trades3.columns,
            loc="upper center",
            cellLoc="center"
        )
        table3b.auto_set_font_size(False)
        table3b.set_fontsize(10)
        table3b.scale(1, 1.5)
        pdf.savefig(fig3b, bbox_inches="tight")
        plt.close(fig3b)

        # ---- Page 6 & 7: Lowest drawdown setup & its trades ----
        fig4, trades4 = plot_setup_and_trades(best_drawdown, "Lowest Drawdown")
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

        fig4b, ax4b = plt.subplots(figsize=(8.5, 11))
        ax4b.axis("off")
        ax4b.set_title("Last 30 Trades: Lowest Drawdown", fontsize=14)
        table4b = ax4b.table(
            cellText=trades4.values,
            colLabels=trades4.columns,
            loc="upper center",
            cellLoc="center"
        )
        table4b.auto_set_font_size(False)
        table4b.set_fontsize(10)
        table4b.scale(1, 1.5)
        pdf.savefig(fig4b, bbox_inches="tight")
        plt.close(fig4b)
        
        # ---- Page 8 & 9: Short ----
        fig_short, trades_short = plot_setup_and_trades(best_return_lev_neg1,"Best Return (Lev = -1)")
        pdf.savefig(fig_short, bbox_inches="tight")
        plt.close(fig_short)

        fig5b, ax5b = plt.subplots(figsize=(8.5, 11))
        ax5b.axis("off")
        ax5b.set_title("Last 30 Trades: Best Return (Lev = -1)", fontsize=14)
        table5b = ax5b.table(
            cellText=trades_short.values,
            colLabels=trades_short.columns,
            loc="upper center",
            cellLoc="center"
        )
        table5b.auto_set_font_size(False)
        table5b.set_fontsize(10)
        table5b.scale(1, 1.5)
        pdf.savefig(fig5b, bbox_inches="tight")
        plt.close(fig5b)

    print("PDF saved as 'vix_sma_optimization_report.pdf'")
