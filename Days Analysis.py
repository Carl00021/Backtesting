import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ─── CONFIG ─────────────────────────────────────────────────────
TICKER       = "SPY"            # e.g. "^VIX", "AAPL", "SPY"
START        = "2000-01-01"      # or None
END          = None              # or "2025-07-08"
MAX_X_DAYS   = 200               # clip any duration above this to the edge
MIN_MOVE_PCT = 5                 # only show segments with ≥5% move
MIN_RETURN   = 10
# ────────────────────────────────────────────────────────────────

def get_close(ticker, start=START, end=END):
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df["Close"].dropna()

def compute_drawdown_segments(close):
    close    = close.squeeze()
    roll_max = close.cummax()
    in_dd    = close < roll_max

    segments = []
    start_idx = None

    for date, flag in in_dd.items():
        if flag:
            if start_idx is None:
                start_idx = date
        else:
            if start_idx is not None:
                segment      = close.loc[start_idx:date]
                peak_price   = roll_max.loc[start_idx]
                trough_price = segment.min()
                duration     = len(segment)
                dd_pct       = 100 * (peak_price - trough_price) / peak_price

                segments.append({
                    "start_date":   start_idx,
                    "end_date":     date,
                    "duration":     duration,
                    "peak_price":   peak_price,
                    "trough_price": trough_price,
                    "drawdown_pct": dd_pct,
                    "year":         date.year
                })
            start_idx = None

    df = pd.DataFrame(segments)
    if df.empty:
        return df

    # clip durations to MAX_X_DAYS for plotting
    df["duration_clipped"] = df["duration"].clip(upper=MAX_X_DAYS)
    return df

def compute_peaks_by_drawdown(close, drop_pct, min_return=0.0):
    """
    Identify every run from trough → peak, where the peak is
    the highest point reached *before* a drop of drop_pct%.

    Returns a DataFrame with:
      - trough_date : date of the low
      - peak_date   : date of the local high (just before drop)
      - duration    : (peak_date - trough_date).days
      - unwind_pct  : 100*(peak_price - trough_price)/trough_price
      - year        : peak_date.year
      - duration_clipped : duration clipped at MAX_X_DAYS
    """
    import pandas as pd

    # unpack DataFrame if needed
    if isinstance(close, pd.DataFrame):
        if "Close" in close.columns:
            close = close["Close"]
        elif close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError("Need a Series or 1-col DataFrame")

    prices = close.values
    dates  = close.index.to_list()
    n      = len(prices)
    thresh = drop_pct / 100.0

    segments = []
    # start from the very first bar as an initial trough
    trough_price = prices[0]
    trough_date  = dates[0]
    i = 0

    while i < n - 1:
        # --- find the peak after i ---
        peak_price = prices[i]
        peak_date  = dates[i]
        j = i + 1

        # scan forward until we hit a drawdown of thresh off that peak
        while j < n:
            p = prices[j]
            # update peak if we hit a new high
            if p > peak_price:
                peak_price = p
                peak_date  = dates[j]
            # if we drop ≥ thresh off our current peak, stop
            if p <= peak_price * (1 - thresh):
                break
            j += 1

        # if we never hit the drop, we're done
        if j == n:
            break

        # record the segment trough → peak
        duration = (peak_date - trough_date).days
        pct      = 100 * (peak_price - trough_price) / trough_price
        segments.append({
            "trough_date":       trough_date,
            "peak_date":         peak_date,
            "duration":          duration,
            "unwind_pct":        pct,
            "year":              peak_date.year
        })

        # reset: the bar that triggered the drop becomes the next trough
        trough_price = prices[j]
        trough_date  = dates[j]
        i = j

    df = pd.DataFrame(segments)
    if not df.empty:
        df["duration_clipped"] = df["duration"].clip(upper=MAX_X_DAYS)
        df = df[df["unwind_pct"] >= min_return]
    return df

def plot_segment_scatter(df, title, pct_col, pdf, close=None, cmap="viridis"):
    # 1) create fig/ax and shrink the right margin so the plot itself is wider
    fig, ax = plt.subplots(figsize=(10,6))
    fig.subplots_adjust(right=0.93)   # <-- reduce white space on the right

    sc = ax.scatter(
        df["duration_clipped"],
        df["year"],
        c=df[pct_col],
        cmap=cmap,
        vmin=0,      
        vmax=30, 
        s=50,
        edgecolor="k",
        alpha=0.8,
    )

    ax.set_xlabel(f"Duration (days, clipped at {MAX_X_DAYS})")
    ax.set_ylabel("Year of Recovery")
    ax.set_title(title)
    ax.set_xlim(0, MAX_X_DAYS)

    # 2) explicitly label every single year on the y-axis
    today     = pd.to_datetime("today").normalize()
    min_year  = int(df["year"].min())
    max_year  = max(int(df["year"].max()), today.year)     # include current year
    years     = list(range(min_year, max_year+1))
    ax.set_yticks(years)
    ax.set_yticklabels(years)

    # 3) add a faint horizontal grid line at each year
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    # colorbar with reduced width and tighter padding
    cbar = fig.colorbar(
        sc, ax=ax,
        fraction=0.03,   # <-- makes the bar thinner
        pad=0.02         # <-- moves it closer to the plot
    )
    cbar.set_label(f"% move ({pct_col})")

    # ── NEW: only if they passed in the full price series ────────────
    if close is not None:
        # 1) coerce to Series if it's a 1-col DataFrame
        if isinstance(close, pd.DataFrame):
            if "Close" in close.columns:
                close = close["Close"]
            elif close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                raise ValueError("plot_segment_scatter: close must be a Series or 1-col DataFrame")

        # 2) recompute in-drawdown flag
        roll_max = close.cummax()
        in_dd    = close < roll_max

        today      = pd.to_datetime("today").normalize()
        current_y  = today.year

        # ─ mark “Today” on the drawdown chart only if we’re currently in a drawdown
        if "Drawdown Duration" in title and in_dd.iloc[-1].any():
            # 1) build an integer mask (1 = in drawdown, 0 = not)
            dd_mask = in_dd.astype(int)
            # 2) find where we *entered* drawdown (diff goes from 0→1)
            entries = dd_mask.diff() == 1
            # 3) the last entry-date is the start of the *ongoing* drawdown
            if entries.any():
                dd_start = entries[entries].index[-1]
            else:
                # we were already in drawdown at the very first bar
                dd_start = close.index[0]

            days_in_dd = (today - dd_start).days
            x_today = min(days_in_dd, MAX_X_DAYS)
            y_today = current_y

            ax.scatter(
                x_today, y_today,
                s=200, marker="*", c="gold", edgecolor="k",
                label="Today"
            )

        # ─ mark “Today” on the unwind chart ────────────────────────────
        if "Trough to Peak" in title:
            # 1) build a 0/1 mask for being in drawdown
            dd_mask = in_dd.astype(int)

            # 2) diff==1 marks the *entry* into drawdown (i.e. the trough start)
            trough_entries = dd_mask.diff() == 1

            # 3) pick the last one (or fallback to the very first bar)
            if trough_entries.any():
                dd_trough = trough_entries[trough_entries].index[-1]
            else:
                dd_trough = close.index[0]

            # now days since trough_date
            days_unwind = (today - dd_trough).days
            x_today = min(days_unwind, MAX_X_DAYS)
            y_today = current_y

            ax.scatter(
                x_today, y_today,
                s=200,
                marker="*",
                c="gold",
                edgecolor="k",
                label="Today"
            )


        ax.legend(loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

if __name__ == "__main__":
    close = get_close(TICKER)

    with PdfPages(f"{TICKER}_drawdown_and_rallies_{MIN_MOVE_PCT}%_{dt.datetime.today().date()}.pdf") as pdf:
        # 1) Drawdowns
        df_dd = compute_drawdown_segments(close)
        df_dd = df_dd[df_dd["drawdown_pct"] >= MIN_MOVE_PCT]
        if not df_dd.empty:
            plot_segment_scatter(
                df_dd,
                title=f"{TICKER} Drawdown Duration (≥{MIN_MOVE_PCT}% drop)",
                pct_col="drawdown_pct",
                pdf=pdf,
                close=close,           # ← pass here
                cmap="Reds",
            )

        # 2) Momentum-unwind
        df_un = compute_peaks_by_drawdown(close, MIN_MOVE_PCT,MIN_RETURN)
        if not df_un.empty:
            plot_segment_scatter(
                df_un,
                title=f"{TICKER} Days from Trough to Peak (without ≥{MIN_MOVE_PCT}% drawdown)",
                pct_col="unwind_pct",
                pdf=pdf,
                close=close,           # ← and pass here
                cmap="Greens",
            )

    print(f"✅ Saved charts to {TICKER}_drawdown_and_unwind.pdf")
