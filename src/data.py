import pandas as pd
import numpy  as np
import yfinance as yf
import pandas_datareader as web
import os

def clean_data (input_path, output_path):
    #loading and cleaning data
    eco= pd.read_csv(input_path)
    eco['Date']= pd.to_datetime(eco['Date'], format='%d-%m-%Y')
    eco= eco.set_index('Date').sort_index()
    eco= eco[['Close']].rename(columns={'Close':'ECO'})

    #date range
    start_date= eco.index.min().strftime('%Y-%m-%d')
    end_date= eco.index.max().strftime('%Y-%m-%d')

    #PSE and WTI from yfinance, GSPC added for robustness check later
    yftickers={'^PSE':'PSE', 'CL=F': 'OIL', '^GSPC':'SP500'}
    yf_raw= yf.download(list(yftickers.keys()), start=start_date, end=end_date, auto_adjust=False)
    yf_data=yf_raw['Close'].rename(columns=yftickers)

    #3 month t-bill
    fred_data= web.DataReader('DTB3', 'fred', start_date, end_date).rename(columns={'DTB3':'RATE'})

    #Combining the four variables
    combined_df=eco.join([yf_data, fred_data], how='outer')
    combined_df= combined_df.sort_index()

    #filling missing values
    combined_df=combined_df.ffill()

    #wednesday close
    wed= combined_df[combined_df.index.dayofweek==2].copy()

    #saving
    wed.to_csv(output_path)
    return wed