# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:59:03 2023

@author: HB

V1  - Monthly Seasonality
V2  - Added Price Interval Feature to toggle 1M, 3M, 1W etc.
V3  - Fixed data error and created output PDF
V4  - Added 2 ticker spread support
V5  - Added Heatmaps
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import calendar
import datetime as dt
from itertools import islice

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
stocks = ["^SKEW"]          # One symbol → single name analysis | Two symbols → spread
start = dt.datetime(2000, 1, 1)  # Earliest date to pull (back fill as you like)
end   = dt.datetime.now()

plot_dayofmonth = False
plot_weekly     = False
plot_monthly    = True  # histogram view – now complemented by heat maps
plot_value_heatmaps = True  # price level (e.g. for VIX)
plot_volume_heatmaps  = True   # monthly volume + z  score
plot_volatility_heatmaps = True   # NEW – daily std & intraday spread

sns.set_theme(rc={"figure.figsize": (20, 8)})

# ──────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD / PREPARATION
# ──────────────────────────────────────────────────────────────────────────────
name = stocks[0]
current_year = end.year
df = yf.Ticker(stocks[0]).history(start=start, end=end)  # max daily granularity

# Spread support – simple price ratio of Close columns
if len(stocks) == 2:
    df1 = df.iloc[:, :4]  # OHL C from first ticker
    df2 = (
        yf.Ticker(stocks[1])
        .history(start=start, end=end)
        .iloc[:, :4]
    )
    df  = (df1 / df2).dropna(how="all")
    name = f"{stocks[0]}∕{stocks[1]}"

# Resample helpers
_df_weekly  = df.resample("W").nearest()
_df_monthly = df.resample("M").nearest()
_df_monthly_mean = df.Close.resample("M").mean()
_df_monthly_vol   = df.Volume.resample("M").sum()  # **total shares traded in month**

# Return series – DAY / WEEK / MONTH
ret_1d = (df.Close / df.Close.shift(1) - 1).dropna()
ret_1w = (_df_weekly.Close / _df_weekly.Close.shift(1) - 1).dropna()
ret_1m = (_df_monthly.Close / _df_monthly.Close.shift(1) - 1).dropna()

#Volatility
_df_intraday_spread = (df.High - df.Low) / df.Close  # relative spread
_df_monthly_std   = ret_1d.resample("M").std()          # std dev of daily returns
_df_monthly_spread= _df_intraday_spread.resample("M").mean()   # mean intraday spread

# YTD return measured at *each* observation (daily) then sampled monthly
_df_with_year = df.copy()
_df_with_year["Year"] = _df_with_year.index.year
first_prices   = _df_with_year.groupby("Year")["Close"].first()
ret_ytd = _df_with_year.apply(
    lambda row: row["Close"] / first_prices[row["Year"]] - 1, axis=1
)
ret_ytd = ret_ytd.resample("M").nearest()

# ──────────────────────────────────────────────────────────────────────────────
# PDF OUTPUT SET UP
# ──────────────────────────────────────────────────────────────────────────────
pdf_fname = (
    f"{name} - Seasonality Data from {df.index[0].year} to {df.index[-1].year}.pdf"
)
pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_fname)

# ──────────────────────────────────────────────────────────────────────────────
# ─ EXISTING HISTOGRAM CHARTS --------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────────
df_returns_1d = (df.Close/df.Close.shift(1)-1).dropna(how='any')
df_returns_open_1d = (df.Open/df.Close.shift(1)-1).dropna(how='any')
df_returns_low_1d = (df.Low/df.Close.shift(1)-1).dropna(how='any')
df_returns_high_1d = (df.High/df.Close.shift(1)-1).dropna(how='any')
df_returns_prevcloseopen_1d = (df.Open/df.Close.shift(1)-1).dropna(how='any')
df_returns_openclose_1d = (df.Close/df.Open-1).dropna(how='any')
df_returns_lowhigh_1d = (df.High/df.Low-1).dropna(how='any')
df_returns_lowclose_1d = (df.Close/df.Low-1).dropna(how='any')
df_returns_highclose_1d = (df.Close/df.High-1).dropna(how='any')

#1D
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(15, 10))
for i, (day, day_df) in enumerate(df_returns_open_1d.groupby(df_returns_open_1d.index.dayofweek)):
    ax = axs[0, i]
    interval_df = df_returns_open_1d[df_returns_open_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Open")
    
for i, (day, day_df) in enumerate(df_returns_1d.groupby(df_returns_1d.index.dayofweek)):
    ax = axs[1, i]
    interval_df = df_returns_1d[df_returns_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Close")
    
for i, (day, day_df) in enumerate(df_returns_low_1d.groupby(df_returns_low_1d.index.dayofweek)):
    ax = axs[2, i]
    interval_df = df_returns_low_1d[df_returns_low_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Low")
    
for i, (day, day_df) in enumerate(df_returns_high_1d.groupby(df_returns_high_1d.index.dayofweek)):
    ax = axs[3, i]
    interval_df = df_returns_high_1d[df_returns_high_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" High")

fig.suptitle(name+' Day of Week')
fig.tight_layout()
pdf.savefig(fig)


#1D Stats
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))
    
for i, (day, day_df) in enumerate(df_returns_prevcloseopen_1d.groupby(df_returns_prevcloseopen_1d.index.dayofweek)):
    ax = axs[0, i]
    interval_df = df_returns_prevcloseopen_1d[df_returns_prevcloseopen_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Previous to Open")

for i, (day, day_df) in enumerate(df_returns_openclose_1d.groupby(df_returns_openclose_1d.index.dayofweek)):
    ax = axs[1, i]
    interval_df = df_returns_openclose_1d[df_returns_openclose_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Open to Close")

for i, (day, day_df) in enumerate(df_returns_1d.groupby(df_returns_1d.index.dayofweek)):
    ax = axs[2, i]
    interval_df = df_returns_1d[df_returns_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Close to Close")

fig.suptitle(name+' Open & Close Day of Week')
fig.tight_layout()
pdf.savefig(fig)


fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 10))

for i, (day, day_df) in enumerate(df_returns_lowhigh_1d.groupby(df_returns_lowhigh_1d.index.dayofweek)):
    ax = axs[0, i]
    interval_df = df_returns_lowhigh_1d[df_returns_lowhigh_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Low to High")

for i, (day, day_df) in enumerate(df_returns_low_1d.groupby(df_returns_low_1d.index.dayofweek)):
    ax = axs[1, i]
    interval_df = df_returns_low_1d[df_returns_low_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Prev to Low")

for i, (day, day_df) in enumerate(df_returns_lowclose_1d.groupby(df_returns_lowclose_1d.index.dayofweek)):
    ax = axs[2, i]
    interval_df = df_returns_lowclose_1d[df_returns_lowclose_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Low to Close")

for i, (day, day_df) in enumerate(df_returns_high_1d.groupby(df_returns_high_1d.index.dayofweek)):
    ax = axs[3, i]
    interval_df = df_returns_high_1d[df_returns_high_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" Prev to High")

for i, (day, day_df) in enumerate(df_returns_highclose_1d.groupby(df_returns_highclose_1d.index.dayofweek)):
    ax = axs[4, i]
    interval_df = df_returns_highclose_1d[df_returns_highclose_1d.index.dayofweek == day]
    ax.hist(interval_df.values, bins=20)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
    ax.legend()
    day_name = calendar.day_name[day]
    ax.set_title(day_name+" High to Close")

fig.suptitle(name+' High & Lows Day of Week')
fig.tight_layout()
pdf.savefig(fig)


if plot_dayofmonth == True:
    ## Day of Month
    # Initialize a list to hold figures and their axes
    figs = []

    # Iterate over the days with an index
    for i, (day, day_df) in enumerate(df_returns_1d.groupby(df_returns_1d.index.day)):
        # Determine which figure this day belongs to
        fig_num = i // 20
        if fig_num >= len(figs):
            # Create a new figure with 5x4 subplots
            fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(15, 10))
            figs.append((fig, axs))
        fig, axs = figs[fig_num]
        # Determine the subplot position in the current figure
        subplot_idx = i % 20
        row = subplot_idx // 4
        col = subplot_idx % 4
        ax = axs[row, col]
        # Plot the histogram
        ax.hist(day_df.values, bins=20)
        avg = day_df.values.mean()
        median = np.median(day_df.values)
        ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
        ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
        ax.legend()
        ax.set_title(f'Day {day}')
        ax.set_ylabel('Frequency')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))

    # Save each figure
    for fig_num, (fig, axs) in enumerate(figs):
        fig.suptitle(name+' Day of Month')
        fig.tight_layout()
        pdf.savefig(fig)

if plot_weekly == True:
#1W
    fig1, axs1 = plt.subplots(nrows=5, ncols=4, figsize=(15, 10))
    fig2, axs2 = plt.subplots(nrows=5, ncols=4, figsize=(15, 10))
    fig3, axs3 = plt.subplots(nrows=5, ncols=4, figsize=(15, 10))
    # Iterate over each day and plot a histogram
    for i, (week, week_df) in enumerate(df_returns_1w.groupby(df_returns_1w.index.isocalendar().week)):
        if i < 20:
            ax = axs1[i // 4, i % 4]
            fig = fig1
        elif i < 40:
            ax = axs2[(i-20) // 4, (i-20) % 4]
            fig = fig2
        else:
            ax = axs3[(i-40) // 4, (i-40) % 4]
            fig = fig3
        ax.hist(week_df.values, bins=20)
        avg = week_df.values.mean()
        median = np.median(week_df.values)
        ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
        ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')
        ax.legend()
        ax.set_title(f'Week {week}')
        ax.set_ylabel('Frequency')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))

    fig1.suptitle(name+' Week of Year')
    fig2.suptitle(name+' Week of Year')
    fig3.suptitle(name+' Week of Year')
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)

if plot_monthly == True:
    #Month over Month
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
    for i, month in enumerate(range(1, 13)):
        ax = axs[i // 3, i % 3]
        interval_df =  ret_1m[ ret_1m.index.month == month]
        ax.hist(interval_df.values, bins=50)
        avg = interval_df.values.mean()
        median = np.median(interval_df.values)
        skewness = interval_df.skew()
        kurt = interval_df.kurt()
        avg_line = ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
        median_line = ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')

        # Custom legend entries for skewness and kurtosis
        skew_line = Line2D([0], [0], color='black', linestyle='-', label=f'Skew: {skewness:.2f}')
        kurt_line = Line2D([0], [0], color='black', linestyle='--', label=f'Kurt: {kurt:.2f}')
        # Add legend with all entries
        ax.legend(handles=[avg_line, median_line, skew_line, kurt_line], 
                labels=[f'Average: {avg:.2%}', f'Median: {median:.2%}', f'Skew: {skewness:.2f}', f'Kurt: {kurt:.2f}'])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
        ax.set_title(pd.to_datetime(f'2022-{month}-01').strftime('%b'))
    fig.suptitle(name+' Monthly Returns')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    #YTD Monthly
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
    for i, month in enumerate(range(1, 13)):
        ax = axs[i // 3, i % 3]
        interval_df = ret_ytd[ret_ytd.index.month == month]
        ax.hist(interval_df.values, bins=50)
        avg = interval_df.values.mean()
        median = np.median(interval_df.values)
        skewness = interval_df.skew()
        kurt = interval_df.kurt()
        avg_line = ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.2%}')
        median_line = ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2%}')

        # Custom legend entries for skewness and kurtosis
        skew_line = Line2D([0], [0], color='black', linestyle='-', label=f'Skew: {skewness:.2f}')
        kurt_line = Line2D([0], [0], color='black', linestyle='--', label=f'Kurt: {kurt:.2f}')
        # Add legend with all entries
        ax.legend(handles=[avg_line, median_line, skew_line, kurt_line], 
                labels=[f'Average: {avg:.2%}', f'Median: {median:.2%}', f'Skew: {skewness:.2f}', f'Kurt: {kurt:.2f}'])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}%'.format(x*100)))
        ax.set_title(pd.to_datetime(f'2022-{month}-01').strftime('%b'))
    fig.suptitle(name+' YTD Returns')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# HEAT  MAP HELPERS (used by all flavours)
# ──────────────────────────────────────────────────────────────────────────────
month_labels = [calendar.month_abbr[m] for m in range(1, 13)]

def add_heatmap(data: pd.DataFrame,
                title: str,
                fmt: str = ".2f",
                cmap: str = "RdYlGn",
                center: float | None = 0.0,
                percent: bool = True,
                cbar: bool = True):
    
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_data = data * 100 if percent else data
    sns.heatmap(
        plot_data,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=fmt,
        linewidths=0.4,
        cbar_kws={"format": PercentFormatter()} if (percent and cbar) else None,
        ax=ax,
    )
    ax.set_xticklabels(month_labels, rotation=0)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# %  RETURN HEAT  MAPS (as in v5)
# ──────────────────────────────────────────────────────────────────────────────

def make_return_heatmaps():
    mo_pivot = (
        ret_1m.to_frame("Return")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Return")
        .reindex(columns=range(1, 13))
    )

    ytd_pivot = (
        ret_ytd.to_frame("Return")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Return")
        .reindex(columns=range(1, 13))
    )

    add_heatmap(mo_pivot, f"{name} Month  over  Month % Returns")
    add_heatmap(ytd_pivot, f"{name} Year  to  Date % Returns (month  end)")

make_return_heatmaps()

# ──────────────────────────────────────────────────────────────────────────────
# NEW — ABSOLUTE VALUE HEAT  MAPS (optional)
# ──────────────────────────────────────────────────────────────────────────────
if plot_value_heatmaps:

    # 1) Month  end raw Close values
    price_pivot = (
        _df_monthly.Close.to_frame("Px")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Px")
        .reindex(columns=range(1, 13))
    )

    add_heatmap(
        price_pivot,
        f"{name} Month  end LEVEL",
        fmt=".1f",
        cmap="YlGnBu",
        center=None,
        percent=False,
    )

    # Monthly  mean price for each Year×Month cell
    price_pivot_mean = (
        _df_monthly_mean.to_frame("Px")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Px")
        .reindex(columns=range(1, 13))
    )

    add_heatmap(
        price_pivot_mean,
        f"{name} Average Daily LEVEL (mean of closes in each month)",
        fmt=".1f",
        cmap="YlGnBu",
        center=None,
        percent=False,
    )

# ──────────────────────────────────────────────────────────────────────────────
# VOLATILITY GRIDS (toggle)
# ──────────────────────────────────────────────────────────────────────────────
if plot_volatility_heatmaps and len(stocks) == 1:
    # 1) Monthly std dev of daily returns
    std_pivot = (
        _df_monthly_std.to_frame("Std")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Std")
        .reindex(columns=range(1, 13))
    )
    add_heatmap(std_pivot, f"{name} Monthly Daily Std Dev (Close to Close)",
                fmt=".2f", cmap="YlGnBu", center=None, percent=True)

    # 2) Monthly mean intraday spread
    spread_pivot = (
        _df_monthly_spread.to_frame("Spread")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Spread")
        .reindex(columns=range(1, 13))
    )
    add_heatmap(spread_pivot, f"{name} Monthly Avg Intraday Spread (High Low)/Close",
                fmt=".2f", cmap="YlGnBu", center=None, percent=True)

# ──────────────────────────────────────────────────────────────────────────────
# VOLUME HEAT  MAPS (optional)
# ──────────────────────────────────────────────────────────────────────────────
if plot_volume_heatmaps and "Volume" in df.columns:
    # 1) Raw volume heat  map
    vol_pivot = (
        _df_monthly_vol.to_frame("Vol")
        .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
        .pivot(index="Year", columns="Month", values="Vol")
        .reindex(columns=range(1, 13))
    )

    add_heatmap(vol_pivot, f"{name} Monthly Volume (shares)", fmt=".1e",
                cmap="YlOrBr", center=None, percent=False, cbar=False)

    # 2) Z  Score calculation with adaptive baseline for current year
    vol_df = _df_monthly_vol.to_frame("Vol")
    vol_df["Year"]  = vol_df.index.year
    vol_df["Month"] = vol_df.index.month

    # Pre  calculate within  year stats for historical years
    year_stats = (
        vol_df[vol_df["Year"] < current_year]
        .groupby("Year")["Vol"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu", "std": "sigma"})
    )

    def z_score(row):
        yr, vol = row["Year"], row["Vol"]
        if yr < current_year:
            mu, sigma = year_stats.loc[yr, ["mu", "sigma"]]
        else:  # current year → use YTD months so far
            subset = vol_df[(vol_df["Year"] == current_year) & (vol_df["Month"] <= row["Month"])]
            mu, sigma = subset["Vol"].mean(), subset["Vol"].std(ddof=0)
        return np.nan if sigma == 0 or np.isnan(sigma) else (vol - mu) / sigma

    vol_df["Z"] = vol_df.apply(z_score, axis=1)

    z_pivot = (
        vol_df.pivot(index="Year", columns="Month", values="Z")
        .reindex(columns=range(1, 13))
    )

    add_heatmap(z_pivot, f"{name} Monthly Volume Z  Score", fmt=".2f",
                cmap="RdYlGn", center=0, percent=False)

# ──────────────────────────────────────────────────────────────────────────────
# FINALISE PDF
# ──────────────────────────────────────────────────────────────────────────────

pdf.close()
print(f"✅ Seasonality PDF saved → {pdf_fname}")
