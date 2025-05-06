"""Trying this again...
Switching now to the Alpaca API. Data on SimFin is missing. Not going to divide by EPS anymore.
"""

import os
import pandas as pd
from dotenv import load_dotenv
import pdb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if __name__ == "__main__":
    utility_tickers = ['NEE', 'SO', 'DUK', 'AEP', 'EXC', 'D', 'PCG', 'XEL', 
                        'ED', 'WEC', 'DTE', 'PPL', 'AEE', 'CNP', 'FE', 'CMS', 
                        'EIX', 'ETR', 'EVRG', 'LNT', 'PNW', 'OGE', 'IDA']
    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
    request_params = StockBarsRequest(symbol_or_symbols=utility_tickers,
                                      timeframe=TimeFrame.Hour,
                                      start=datetime(2023, 5, 1))

    #bars = client.get_stock_bars(request_params).df
    #print(bars)
    #bars.to_csv('data/utility_bars.csv')
    bars = pd.read_csv('data/utility_bars.csv')
    bars.set_index(['symbol', 'timestamp'], inplace=True)
    print(bars.info())