# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:56:48 2024

V1: Boxplot, Lineplot, Histogram for Price Change Filter.
V2: Added backtest variable seperate from stock returns. You can now see stock return vs backtest   filter.
V3: Added Line Chart
V4: Added fixed level function drawdown level
V5: Added Fred Data for Backtest. Line105 Issue with index dates on FRED vs Stock Dates. Will need to merge and backfill probably or try to go forward from the fred date on stock df.
V6: Title changes with date function.
V7: Added Vix Date for Vix Level and Rate of Change
V8: Added option to do spread i.e. SPX/NDX
V9: Added option to input your own dates.
V10: Fixed the lookforward overlap. The idea is to remove double counting, if the overlap removes the signal lower the lookforward period.
V11: Drawdown Chart
V12:Max Rate of Change After Period vs Return after x period (includes sharp reversals)
V12: Added 2nd Backtest Filter
V13: Added 2nd Step Change Percentage
V14: Added YF Earnings Date and Custom Date Functions. Added Seasonality Charts.
V14.4 Added Rate Limit Request Work around

"""

import yfinance as yf
import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from statsmodels.regression.linear_model import OLS
import pandas as pd
from pandas.tseries.offsets import BDay
import datetime as dt
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from Fred_API import fred_get_series

#Helps get around Rate Limit Requests
from curl_cffi import requests
session = requests.Session(impersonate="chrome")

#Input Variables
end = dt.datetime.now()
start = dt.datetime(2018,1,1)
stocks = ['NFLX'] #can put one or two, if you put two tickers its the ticker1/ticker2
stockrelationship = None #"1-x/(y*2.52)" #"1.97/x-1"#"(0*x + 0.03552*y+2.75)/x-1" #use x and y (case sensitive) for example if there is a merger spread with stock and cash
backtest_stock = None #use None for Spreads, Use backtest=stocks[0] if you just want to test vs own historical price #^VIX, ^VXN, ^TNX, ^MOVE, ^SKEW
backtest_fred = None #'A053RC1Q027SBEA' #'UNRATE' #CCSM #FEDFUNDS
threshold_high =1
threshold_low = 0.1
rate_of_change = 0
dollar_rate_of_change = 1
period_chg= 1 #Based on trading days
look_forward= 30 #Based on business days
look_back= 30

log_scale = True
highlight_backtest = True
sns.set_theme(rc={'figure.figsize':(11.5,8)})
x_axis_multiple = max(round((look_back+look_forward)/20),1)
chart_2_dd = True

#Get the data for the stock data
if len(stocks) == 1:
    name_stock = stocks[0]
    backtest = name_stock
    df_stock = yf.Ticker(stocks[0],session=session).history(start=start,end=end).iloc[:,:4].tz_localize(None)
    df_returns = df_stock/df_stock.shift(1)-1
    if stockrelationship != None:
        df_stock = eval(stockrelationship,{"x":df_stock})
if len(stocks) == 2:
    name_stock = stocks[0] +'∕'+ stocks[1]
    backtest = name_stock
    df_stock_1 = yf.Ticker(stocks[0],session=session).history(start=start,end=end).iloc[:,:4]
    df_stock_2 = yf.Ticker(stocks[1],session=session).history(start=start,end=end).iloc[:,:4]
    if stockrelationship != None:
        df_stock = eval(stockrelationship,{"x":df_stock_1,"y":df_stock_2})
    else:
        df_stock = df_stock_1/df_stock_2
    df_stock = df_stock.dropna(axis=0, how='any').tz_localize(None)
    df_returns = df_stock/df_stock.shift(1)-1
else:
    print("Input only one or two tickers")
    
if backtest_stock != None:
    #Get the data for the backtest variable from Yfinance
    backtest = backtest_stock
    df_backtest = yf.Ticker(backtest,session=session).history(start=start,end=end).iloc[:,:4]
    #df_backtest.index = pd.to_datetime(df_backtest.index)  
    df_backtest.index = df_backtest.index.tz_localize(None)
    start = df_backtest.index[0].tz_localize(None)
    df_backtest = df_backtest.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
elif backtest_fred != None:
    backtest = backtest_fred
    #Get the data for the backtest variable from FRED
    df_backtest = fred_get_series(backtest)
    df_backtest.index = df_backtest['date'].tz_localize(None)
    df_backtest.index = pd.to_datetime(df_backtest.index)
    df_backtest['Close'] = pd.to_numeric(df_backtest['value'],errors='coerce')
    df_backtest = df_backtest.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
elif backtest_stock == None: # for spreads.
    df_backtest = df_stock


# Finds all the dates where the stock breached the threshold
def date_drawdown_pct(df,threshold_high,threshold_low,period_chg=1):
    #df=df_backtest
    #df['daily_pct_change'] = df.Close.pct_change(periods=1)
    #df['period_pct_chg'] = df.Close.pct_change(periods=period_chg)
    if threshold_high < 0:
        df['roll_min_price'] = df.Close.rolling(period_chg).min()
        df['roll_min_price_chg'] = df['roll_min_price']/df['Close'].shift(period_chg)-1
        threshold_list = df[(df['roll_min_price_chg']<threshold_high) & (df['roll_min_price_chg']>threshold_low)]
    else:
        df['roll_max_price'] = df.Close.rolling(period_chg).max()
        df['roll_max_price_chg'] = df['roll_max_price']/df['Close'].shift(period_chg)-1
        threshold_list = df[(df['roll_max_price_chg']<threshold_high) & (df['roll_max_price_chg']>threshold_low)]
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + BDay(look_forward):
            clean_dates.append(list_dates[i])
    if threshold_high > 0:
        threshold = threshold_low
    else:
        threshold = threshold_high
    title = ' - After '+f'{threshold:+.1%} Change in ' +backtest+' Over '+str(period_chg)+ 'D'
    return clean_dates,list_dates,title    
def date_drawdown_dollar(df,threshold_high,threshold_low,period_chg=1):
    df['daily_dollar_chg'] = df.Close - df.Close - df.Close.shift(periods=1)
    df['period_dollar_chg'] = df.Close - df.Close.shift(periods=period_chg)
    threshold_list = df[(df['period_dollar_chg']<threshold_high) & (df['period_dollar_chg']>threshold_low)]
        
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + BDay(look_forward):
            clean_dates.append(list_dates[i])
    if threshold_high > 0:
        threshold = threshold_low
    else:
        threshold = threshold_high
    title = ' - After '+f'{threshold:+.1f} Dollar Change in ' +backtest+' Over '+str(period_chg)+ 'D'
    return clean_dates,list_dates,title     
def date_drawdown_level(df,threshold_high,threshold_low,period_chg=1):
    threshold_list = df[(df['Close']<threshold_high) & (df['Close']>threshold_low)]
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])

    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + BDay(look_forward):
            clean_dates.append(list_dates[i])
            
    title = backtest +' - From '+f'{threshold_low:+.1f}'+' to ' +f'{threshold_high:+.1f}' 
    return clean_dates,list_dates,title   
def date_level_roc(df,threshold_low,rate_of_change,period_chg=1):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    threshold_list = df[(df['Close']>threshold_low) & (df['daily_pct_change']>rate_of_change)]
    
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])
    
    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + BDay(look_forward):
            clean_dates.append(list_dates[i])

    title = ' - Above '+f'{threshold_low:.1f}'+' '+ backtest +' After '+f'{rate_of_change:+.1%} Change in ' +backtest+' Over '+str(period_chg)+ 'D'
    return clean_dates,list_dates,title   
def date_first_rate_cut():
    fedfunds = fred_get_series('FEDFUNDS')
    fedfunds.index = fedfunds['date']
    fedfunds.index = pd.to_datetime(fedfunds.index)
    fedfunds = fedfunds['value']
    fedfunds = pd.to_numeric(fedfunds)
    
    df_fedfunds = pd.DataFrame(fedfunds)
    df_rates = df_fedfunds.join(df_stock.Close).dropna(how='any')
    df_rates = df_rates.rename(columns={"value": "FedFunds", "Close": "US10Y"})
    
    
    threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)&(df_rates.FedFunds.rolling(12).max() - df_rates.FedFunds.shift(1)<0.25)] #First Rate Hike, Filters out rolling 12M noise 
    #threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)&(df_rates.FedFunds.rolling(12).max() - df_rates.FedFunds.shift(1)<0.25)&(df_rates.FedFunds > df_rates.US10Y)]  #First Cut with no changes in prior 12M and Fedfunds > US10Y
    #threshold_list = df_rates[(df_rates.FedFunds.shift(1)-df_rates.FedFunds > 0.25)] #any rate cut not just fisrt rate hike   
    list_dates = threshold_list.index.tolist()
    title = ' -  After First Rate Cut'
    return list_dates,list_dates,title   
def date_below_sma(df,period_chg=period_chg):
    df=df_backtest
    df['SMA'] = df.Close.rolling(window=period_chg).mean()
    threshold_list = df[(df['Close']<df['SMA'])]
    list_dates = threshold_list.index.tolist()
    clean_dates = []
    clean_dates.append(list_dates[0])
    for i in range(0, len(list_dates)):
        if list_dates[i] > clean_dates[-1] + BDay(look_forward):
            clean_dates.append(list_dates[i])
   
    title = ' - Below '+str(period_chg)+'D SMA'
    return clean_dates,list_dates,title  
def date_earnings():
    earnings =  yf.Ticker(stocks[0],session=session).get_earnings_dates(limit=100)
    dates = earnings.index.tz_localize(None).normalize()
    dates = dates[dates<= end]
    dates = dates[dates>= df_stock.index[0]]
    dates = dates.sort_values()
    all_dates = dates
    title = ' Earnings Date'
    return dates, all_dates, title
def date_custom(title,dates,):    
    #dates = ['2024-08-06','2024-05-09','2024-04-26','2024-02-29','2023-11-14','2023-11-14','2023-08-08','2023-05-03','2023-04-21','2023-02-28','2022-11-02','2022-08-02','2022-05-10','2022-04-28','2022-03-01']
    for x in range(0,len(dates)):
        dates[x] = dt.datetime.strptime(dates[x], "%Y-%m-%d")#.timestamp()
    dates.sort()
    all_dates = dates
    title = title
    return dates, all_dates, title

dates,all_dates,title = date_earnings()
#dates,all_dates,title = date_custom('Assassins Creed Launches',['2024-08-06','2024-05-09'])
#dates,all_dates,title = date_drawdown_pct(df_backtest,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#dates,all_dates,title = date_drawdown_dollar(df_backtest,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#dates,all_dates,title = date_drawdown_level(df_backtest,threshold_high=threshold_high,threshold_low=threshold_low,period_chg=period_chg)
#dates,all_dates,title = date_first_rate_cut()
#dates,all_dates,title = date_level_roc(df_backtest,threshold_low,rate_of_change)
#dates,all_dates,title = date_below_sma(df_backtest, period_chg=50)

base_rate = len(all_dates)/df_stock.Close.count()

if backtest_fred != None:
    for x in range(0,len(dates)):
        dates[x] = df_stock.index[df_stock.index.searchsorted(dates[x])] #converts Fred Dates to Nearest Stock market dates

#Append Timeseries Since Date
vintage_table_list = []
date = df_stock.index.get_loc(dates[0])
series = df_stock.iloc[date-period_chg: date + look_forward, 3].rename(dates[0]).reset_index(drop=True)

for i in dates:
    date = min(range(len(df_stock.index)), key=lambda d: abs(df_stock.index[d] - i)) # This gets the closest date # Original date = df_stock.index.get_loc(i)
    series = df_stock.iloc[date-look_back-1: date + look_forward+1, 3].rename(i).reset_index(drop=True)
    vintage_table_list.append(series)

#Adds Line for Today (for look back). It won't mess with the averages because the forward will be NAN Values.
series = df_stock.iloc[len(df_stock)-look_back-1-1: len(df_stock) + look_forward+1, 3].rename(end.replace(hour=0, minute=0, second=0, microsecond=0)).reset_index(drop=True)
vintage_table_list.append(series)   

df_price = pd.concat(vintage_table_list, axis = 1)
df_price.columns = map(str, df_price.columns)
df_price = df_price.set_axis(range(-look_back-1, len(df_price)-look_back-1))
df_columns = len(df_price.columns)
                 
df_return = df_price/df_price.shift(1)-1
df_return = df_return.loc[-look_back:look_forward]
df_return['average'] = df_return.iloc[:].mean(axis=1)
df_return['max'] = df_return.iloc[:,:df_columns].max(axis=1)
df_return['min'] = df_return.iloc[:,:df_columns].min(axis=1)
df_return['median'] = df_return.iloc[:,:df_columns].median(axis=1)
df_return['sd'] = df_return.iloc[:,:df_columns].std(axis=1)

df_index = df_price/df_price.loc[0]-1
df_index = df_index.loc[-look_back:look_forward]
df_index['average'] = df_index.iloc[:].mean(axis=1)
df_index['max'] = df_index.iloc[:,:df_columns].max(axis=1)
df_index['min'] = df_index.iloc[:,:df_columns].min(axis=1)
df_index['median'] = df_index.iloc[:,:df_columns].median(axis=1)
df_index['sd'] = df_index.iloc[:,:df_columns].std(axis=1)

df_stock['rolling_max'] = df_stock['Close'].cummax()
df_stock['Drawdown'] = (df_stock['Close']-df_stock['rolling_max']) / df_stock['rolling_max']

df_backtest['rolling_max'] = df_backtest['Close'].cummax()
df_backtest['Drawdown'] = (df_backtest['Close']-df_backtest['rolling_max']) / df_backtest['rolling_max']

df_return_boxplot = df_return.transpose()
return_period_avg = df_return_boxplot.iloc[-5,:].reset_index(drop=True)
return_period_std = df_return_boxplot.iloc[-1,:].reset_index(drop=True)
return_period_median = df_return_boxplot.iloc[-2,:].reset_index(drop=True)

df_index_boxplot = df_index.iloc[::x_axis_multiple].transpose()
index_period_avg = df_index_boxplot.iloc[-5,:].reset_index(drop=True)
index_period_std = df_index_boxplot.iloc[-1,:].reset_index(drop=True)
index_period_median = df_index_boxplot.iloc[-2,:].reset_index(drop=True)

#Create PDF
pdf = PdfPages(name_stock + title+' - '+str(look_forward)+'D Forward Backtest ' +str(start.year)+ " to "+ str(end.year)+" as of "+dt.datetime.now().strftime("%Y-%m-%d")+'.pdf')

##Line Backtest Trigger Charts ##
transparency= max(0.81 - len(all_dates)/100,0.10)

#To show both stocks in a spread
if len(stocks) == 2: 
    df_comparison_index = pd.concat((df_stock_1['Close'],df_stock_2['Close']),axis=1).dropna()
    df_comparison_index.columns = ['Stock1', 'Stock2']
    df_comparison_index = df_comparison_index/df_comparison_index.iloc[0,:]
    df_comparison_index['Stock1'].plot(label=stocks[0])
    df_comparison_index['Stock2'].plot(label=stocks[1])
    plt.text(df_stock.index[-1],df_comparison_index['Stock1'][-1],f'{df_comparison_index["Stock1"][-1]:,.1f}x',weight='bold')
    plt.text(df_stock.index[-1],df_comparison_index['Stock2'][-1],f'{df_comparison_index["Stock2"][-1]:,.1f}x',weight='bold')
    
    if highlight_backtest == True:
        for x in range(0,len(all_dates)):
            plt.axvline(all_dates[x],color='red', alpha=transparency)
    plt.text (0.01,0.95,f'Occurrances: {len(all_dates):.0f}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
    plt.text (0.01,0.99,f'Base Rate: {base_rate:.2%}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
    plt.yscale('log')
    plt.legend()
    plt.title(name_stock + title + " - Index Log Scale")
    plt.xlim(df_stock.index[0],end)
    plt.tight_layout()
    pdf.savefig() 
    plt.show()

if ((backtest_stock == None) & (backtest_fred == None)) or (backtest_stock == stocks[0]):
    df_stock['Close'].plot(color='black',title=name_stock)
    plt.text(df_stock.index[-1],df_stock['Close'][-1],f'{df_stock["Close"][-1]:,.2f}',weight='bold')
    if highlight_backtest == True:
        for x in range(0,len(all_dates)):
            plt.axvline(all_dates[x],color='red', alpha=transparency)
    plt.text (0.01,0.95,f'Occurrances: {len(all_dates):.0f}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
    plt.text (0.01,0.99,f'Base Rate: {base_rate:.2%}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
    plt.title(name_stock + title)
else:
    fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True)
    axes[0].plot(df_stock['Close'],color='black')
    axes[0].text(df_stock.index[-1],df_stock['Close'][-1],f'{df_stock["Close"][-1]:,.2f}',weight='bold')
    axes[0].set_title(name_stock)
    axes[1].plot(df_backtest['Close'],color='black')
    axes[1].text(df_backtest.index[-1],df_backtest['Close'][-1],f'{df_backtest["Close"][-1]:.2f}',weight='bold')
    axes[1].set_title(backtest)
    if log_scale == True:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
    if highlight_backtest == True:
        for x in range(0,len(all_dates)):
            axes[0].axvline(all_dates[x],color='red', alpha=transparency)
            plt.text (0.01,0.90,f'Occurrances: {len(all_dates):.0f}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
            plt.text (0.01,0.99,f'Base Rate: {base_rate:.2%}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
            axes[1].axvline(all_dates[x],color='red', alpha=transparency)
    plt.subplots_adjust(hspace=0.15)  # Decrease the vertical spacing
    plt.suptitle(name_stock +' '+ title)

#plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=2, numticks=20))
#plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: '{:.1%}'.format(x)))
plt.xlim(df_stock.index[0],end)
plt.tight_layout()
pdf.savefig() 
plt.show()

#Drawdown Chart
if ((backtest_stock == None) & (backtest_fred == None)) or (backtest_stock == stocks[0]):
    df_stock['Drawdown'].plot(color='black',title=name_stock)
    plt.axhline(df_stock['Drawdown'][-1],color='black', linestyle='--')
    plt.text(df_stock.index[-1],df_stock['Drawdown'][-1],f'{df_stock["Drawdown"][-1]:+.2%}',weight='bold')
    if highlight_backtest == True:
        for x in range(0,len(all_dates)):
            plt.axvline(all_dates[x],color='red', alpha=transparency)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
    plt.title(name_stock +' Drawdown'+ title)
else:
    fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True)
    axes[0].plot(df_stock['Drawdown'],color='black')
    axes[0].axhline(df_stock['Drawdown'][-1],color='black', linestyle='--')
    axes[0].text(df_stock.index[-1],df_stock['Drawdown'][-1],f'{df_stock["Drawdown"][-1]:+.2%}',weight='bold')
    axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
    axes[0].set_title(name_stock+" Drawdown")
    if chart_2_dd == True:
        axes[1].plot(df_backtest['Drawdown'],color='black')
        axes[1].text(df_backtest.index[-1],df_backtest['Drawdown'][-1],f'{df_backtest["Drawdown"][-1]:+.2%}',weight='bold')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
    else:
        axes[1].plot(df_backtest['Close'],color='black')
        axes[1].text(df_backtest.index[-1],df_backtest['Close'][-1],f'{df_backtest["Close"][-1]:.2f}',weight='bold')
        if log_scale == True:
            axes[1].set_yscale('log')
    axes[1].set_title(backtest)
    if highlight_backtest == True:
        for x in range(0,len(all_dates)):
            axes[0].axvline(all_dates[x],color='red', alpha=transparency)
            axes[1].axvline(all_dates[x],color='red', alpha=transparency)
    plt.subplots_adjust(hspace=0.15)  # Decrease the vertical spacing
    plt.suptitle(name_stock +' '+ title)
plt.xlim(df_stock.index[0],end)
plt.tight_layout()
pdf.savefig() 
plt.show()

#Show clean dates (remove overlap)
if ((backtest_stock == None) & (backtest_fred == None)) or (backtest_stock == stocks[0]):
    df_stock['Close'].plot(color='black',title=name_stock)
    if highlight_backtest == True:
        for x in range(0,len(dates)):
            plt.axvline(dates[x],color='red', alpha=1)
    plt.title(name_stock +" - Removed Look Forward Overlap")
else:
    fig, axes = plt.subplots(nrows=2, ncols=1,sharex=True)
    axes[0].plot(df_stock['Close'],color='black')
    axes[1].plot(df_backtest['Close'],color='black')
    axes[0].set_title(name_stock)
    axes[1].set_title(backtest)
    if log_scale == True:
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
    if highlight_backtest == True:
        for x in range(0,len(dates)):
            axes[0].axvline(dates[x],color='red', alpha=1)
            axes[1].axvline(dates[x],color='red', alpha=1)
    plt.subplots_adjust(hspace=0.15)  # Decrease the vertical spacing
    plt.suptitle(name_stock +' '+ title+" - Removed Look Forward Overlap")
plt.xlim(df_stock.index[0],end)
plt.tight_layout()
pdf.savefig() 
plt.show()

#Lines Historical Incidences
last_5=df_index.iloc[-1:,-11:-6].squeeze(axis=0)
plt.plot(df_index.iloc[:,-11:-6],linewidth=2, alpha=0.5)
plt.plot(df_index.iloc[:,-6:-5],linewidth=3, color='black')
plt.axvline(0, color='black')

try:
    for x in range(0,len(last_5)):
        if pd.isna(last_5[x]):
           last_instance =  df_index[last_5.index[x]].dropna() #-6 is extra rows. 3 is -3+2  -4+3
           last_5[x] = last_instance.iloc[-1]
           plt.text(last_instance.index[-1],last_5[x],f'{last_5[x]:+.1%}',weight='bold') 
        else:  
           plt.text(df_index.index[-1],last_5[x],f'{last_5[x]:+.1%}',weight='bold')
except:
    for x in range(0,len(last_5)):    
        plt.text(df_index.index[-1],last_5[x],f'{last_5[x]:+.1%}',weight='bold')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_axis_multiple))
plt.legend(df_index.columns[-11:-5])
plt.suptitle(name_stock +' - Cumulative Return Last 5 Instances '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

num_bt_instances = min(5,len(df_index.columns)-5)#Min is if there are less than 5 instances
largest_values = df_index.iloc[-1,:-5].nlargest(num_bt_instances) 
df_best_outcomes = df_index.loc[:, df_index.iloc[-1].isin(largest_values)]
df_best_outcomes = df_best_outcomes.iloc[:,:num_bt_instances]
df_best_outcomes[df_index.columns[-6]] = df_index.iloc[:,-6]
plt.plot(df_best_outcomes,linewidth=2, alpha=0.5)
plt.plot(df_index.iloc[:,-6:-5],linewidth=3, color='black')
for x in range(0,len(largest_values)):    
    plt.text(df_index.index[-1],largest_values[x],f'{largest_values[x]:+.1%}',weight='bold')
plt.axvline(0, color='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_axis_multiple))
plt.legend(df_best_outcomes)
plt.suptitle(name_stock +' - Cumulative Return Last 5 Best Instances '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

smallest_values = df_index.iloc[-1,:-5].nsmallest(num_bt_instances)
df_worst_outcomes = df_index.loc[:, df_index.iloc[-1].isin(smallest_values)]
df_worst_outcomes = df_worst_outcomes.iloc[:,:num_bt_instances]
df_worst_outcomes[df_index.columns[-6]] = df_index.iloc[:,-6]
plt.plot(df_worst_outcomes,linewidth=2, alpha=0.5)
plt.plot(df_index.iloc[:,-6:-5],linewidth=3, color='black')
for x in range(0,len(smallest_values)):    
    plt.text(df_index.index[-1],smallest_values[x],f'{smallest_values[x]:+.1%}',weight='bold')
plt.axvline(0, color='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(x_axis_multiple))
plt.legend(df_worst_outcomes)
plt.suptitle(name_stock +' - Cumulative Return Last 5 Worst Instances '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

#BoxPlot
if (look_back+look_forward) < 31:
    boxplot = sns.boxplot(df_return_boxplot)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    for x in boxplot.get_xticks():
        if return_period_avg[x] > 0:
            boxplot.text(x,return_period_avg[x],f'{return_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='green',weight='bold',backgroundcolor='white')
        else:
            boxplot.text(x,return_period_avg[x],f'{return_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='red',weight='bold',backgroundcolor='white')
            plt.suptitle(name_stock +' - Average Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
    plt.show()

boxplot = sns.boxplot(df_index_boxplot.iloc[:-5,:])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
for x in boxplot.get_xticks():
    if index_period_avg[x] > 0:
        boxplot.text(x,index_period_avg[x],f'{index_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='green',weight='bold',backgroundcolor='white')
    else:
        boxplot.text(x,index_period_avg[x],f'{index_period_avg[x]:.1%}',horizontalalignment='center',size='medium',color='red',weight='bold',backgroundcolor='white')
plt.suptitle(name_stock +' - Average Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Swarm Line Plot
if len(df_index_boxplot.index) < 50:
    swarmplot = sns.swarmplot(data=df_index_boxplot.iloc[:-5,:])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.suptitle(name_stock +' - Average Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
    plt.tight_layout()
    pdf.savefig() 
    plt.show()
    df_index['average']+df_index['sd']*2

#Line Plot
avg_line = sns.lineplot(x=df_index.index, y= df_index['average'],data=df_index,label="Average", linewidth=2)
median_line = sns.lineplot(x=df_index.index, y= df_index['median'],data=df_index,label="Median", linewidth=2)
max_line = sns.lineplot(x=df_index.index, y= df_index['max'],data=df_index,label="Max", linewidth=1, color='green')
min_line = sns.lineplot(x=df_index.index, y= df_index['min'],data=df_index,label="Min", linewidth=1, color = 'red')
plt.fill_between(df_index.index,df_index['average']-df_index['sd'], df_index['average']+df_index['sd'], alpha=0.3) #2 Standard Deviation
plt.fill_between(df_index.index,df_index['average']-df_index['sd']*2, df_index['average']+df_index['sd']*2, alpha=0.15) #2 Standard Deviation
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(x_axis_multiple))
plt.suptitle(name_stock +' - Average Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.legend()
plt.tight_layout()
pdf.savefig() 
plt.show()

#Histogram Returns after Look Forward Period
df_index_last = df_index.iloc[-1]
df_index_last[:-5].plot.hist(bins=50, alpha=0.75)
plt.text (0.01,0.99,f'Base Rate: {base_rate:.2%}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
plt.text (0.01,0.95,f'Occurrances: {len(all_dates):.0f}', transform=plt.gca().transAxes,ha='left', va='top', bbox=dict(facecolor='white', edgecolor='none', pad=5))
plt.axvline(df_index_last['average'], color='black', linestyle='dashed', linewidth=3)
plt.text(df_index_last['average'],1,f'Average {df_index_last["average"]:.1%}',horizontalalignment='center',size='large',color='black',weight='bold',backgroundcolor='white')
plt.axvline(df_index_last['median'], color='black', linestyle='dashed', linewidth=3)
plt.text(df_index_last['median'],0.5,f'Median {df_index_last["median"]:.1%}',horizontalalignment='center',size='large',color='black',weight='bold',backgroundcolor='white')
plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.suptitle(name_stock +' - Average '+str(look_forward)+' Day Cumulative Return '+ title +' - '+ str(start.year)+ " to "+ str(end.year))
plt.tight_layout()
pdf.savefig() 
plt.show()

#Price Seasonality Monthly
df_1m = df_stock.resample('M').nearest()
df_returns_1m = (df_1m.Close/df_1m.Close.shift(1)-1).dropna(how='any')
df_stock['year'] = df_stock.index.year
first_prices = df_stock.groupby('year')['Close'].first()
df_returns_ytd = df_stock.apply(lambda row: (row['Close'] / first_prices[row['year']] - 1), axis=1)
df_returns_ytd = df_returns_ytd.resample('M').nearest()

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
for i, month in enumerate(range(1, 13)):
    ax = axs[i // 3, i % 3]
    interval_df = df_returns_1m[df_returns_1m.index.month == month]
    ax.hist(interval_df.values, bins=50)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.1%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.1%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}%'.format(x*100)))
    ax.legend()
    ax.set_title(pd.to_datetime(f'2022-{month}-01').strftime('%b'))
fig.suptitle(name_stock+' Monthly Returns')
fig.tight_layout()
pdf.savefig(fig)
plt.show()

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
for i, month in enumerate(range(1, 13)):
    ax = axs[i // 3, i % 3]
    interval_df = df_returns_ytd[df_returns_ytd.index.month == month]
    ax.hist(interval_df.values, bins=50)
    avg = interval_df.values.mean()
    median = np.median(interval_df.values)
    ax.axvline(avg, color='red', linestyle='--', label=f'Average: {avg:.1%}')
    ax.axvline(median, color='green', linestyle='-', label=f'Median: {median:.1%}')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}%'.format(x*100)))
    ax.legend()
    ax.set_title(pd.to_datetime(f'2022-{month}-01').strftime('%b'))
fig.suptitle(name_stock+' YTD Returns')
fig.tight_layout()
pdf.savefig(fig)
plt.show()

pdf.close()


