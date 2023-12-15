# -*- coding: utf-8 -*-
"""
Futures Screener

@author: jquintero
"""

# Imports
import pandas_datareader as pdr #Fix at: https://github.com/pydata/pandas-datareader/issues/867
from yahoo_fin import stock_info as si
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
import datetime
import time
yf.pdr_override()

# Variables
tickers = si.get_futures().Symbol
str_tickers = [item.replace(".", "-") for item in tickers]
start_date = datetime.datetime.now() - datetime.timedelta(days=365)
end_date = datetime.date.today()
exportList = pd.DataFrame(columns=['Future', "RS_Rating", "50 Day MA", 
                                   "150 Day Ma", "200 Day MA", 
                                   "52 Week Low", "52 week High"])
returns_multiples = []

# Find top 30% performing futures
for ticker in tickers:
    # Download historical data as CSV for each future (faster)
    df = pdr.get_data_yahoo(ticker, start_date, end_date)
    df.to_csv(f'{ticker}.csv')

    # Calculating returns relative (returns multiple)
    df['Percent Change'] = df['Adj Close'].pct_change()
    fut_return = (df['Percent Change'] + 1).cumprod()[-1]
    
    returns_multiple = round(fut_return, 2)
    returns_multiples.extend([returns_multiple])

# Creating dataframe of only top 30%
rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), 
                     columns=['Ticker','Returns_multiple'])
rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
rs_df = rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(.70)]

# Checking Minervini conditions of top 30% of futures in given list
rs_fut = rs_df['Ticker']
for fut in rs_fut:    
    try:
        df = pd.read_csv(f'{fut}.csv', index_col=0)
        sma = [50, 150, 200]
        for x in sma:
            df["SMA_"+str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)
        
        # Storing required values 
        currentClose = df["Adj Close"][-1]
        moving_average_50 = df["SMA_50"][-1]
        moving_average_150 = df["SMA_150"][-1]
        moving_average_200 = df["SMA_200"][-1]
        low_of_52week = round(min(df["Low"][-260:]), 2)
        high_of_52week = round(max(df["High"][-260:]), 2)
        RS_Rating = round(rs_df[rs_df['Ticker']==fut].RS_Rating.tolist()[0])
        
        try:
            moving_average_200_20 = df["SMA_200"][-20]
        except Exception:
            moving_average_200_20 = 0

        # Condition 1: Current Price > 150 SMA and > 200 SMA
        condition_1 = currentClose > moving_average_150 > moving_average_200
        
        # Condition 2: 150 SMA and > 200 SMA
        condition_2 = moving_average_150 > moving_average_200

        # Condition 3: 200 SMA trending up for at least 1 month
        condition_3 = moving_average_200 > moving_average_200_20
        
        # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
        condition_4 = moving_average_50 > moving_average_150 > moving_average_200
           
        # Condition 5: Current Price > 50 SMA
        condition_5 = currentClose > moving_average_50
           
        # Condition 6: Current Price is at least 30% above 52 week low
        condition_6 = currentClose >= (1.3*low_of_52week)
           
        # Condition 7: Current Price is within 25% of 52 week high
        condition_7 = currentClose >= (.75*high_of_52week)
        
        # If all conditions above are true, add stock to exportList
        if(condition_1 and condition_2 and condition_3 and condition_4 and 
           condition_5 and condition_6 and condition_7):
            exportList = exportList.append({'Future': fut, 
                                            "RS_Rating": RS_Rating ,
                                            "50 Day MA": moving_average_50,
                                            "150 Day Ma": moving_average_150,
                                            "200 Day MA": moving_average_200,
                                            "52 Week Low": low_of_52week,
                                            "52 week High": high_of_52week},
                                           ignore_index=True)
            print (fut + " made the Minervini requirements")
    except Exception as e:
        print (e)
        print(f"Could not gather data on {fut}")

exportList = exportList.sort_values(by='RS_Rating', ascending=False)
print('\n', exportList)

# Export Screen
writer = ExcelWriter("ScreenOutput.xlsx")
exportList.to_excel(writer, "Sheet1")
writer.save()
