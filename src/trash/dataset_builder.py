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
import json
import requests
import time
from tqdm import tqdm
import numpy as np
from util import load_sec_tickers, save_data
import matplotlib.pyplot as plt
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY")
SIMFIN_RATE_LIMIT = 2 # PER SECOND

def make_request(url:str, key:str) -> pd.DataFrame:
    headers = {
        "accept": "application/json",
        "Authorization": key
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.text}")
    return response.json()

def normalize_price_data_by_derived_data(price_data:pd.DataFrame, eps_small:pd.DataFrame) -> pd.DataFrame:
    """Normalize the price data by the derived data
    price_data is a dataframe with a MultiIndex of (ticker, timestamp)
    eps_small is a dataframe with a MultiIndex of (ticker, Report Date)
    We want to normalize the price data by the derived data, grouped by ticker and report date. 
    We want to use the last reported EPS TTM for normalization. 
    """
    tickers = price_data.index.get_level_values('symbol').unique()
    # calculate intersection of tickers
    tickers = list(set(tickers) & set(eps_small.index.get_level_values('ticker').unique()))
    print(f"Normalizing prices for {tickers}")
    normalized_prices = price_data.copy()

    # For each ticker
    for ticker in tqdm(tickers, desc="Normalizing Prices"):
        try:
            # Get share prices for this ticker
            ticker_prices = normalized_prices.loc[ticker]
            
            # Get EPS values for this ticker
            ticker_eps = eps_small.loc[ticker]
            ticker_eps = ticker_eps.sort_index()
        
            #print(ticker_eps)

            # For each date in ticker_prices
            for date, row in ticker_prices.iterrows():
                # Find the most recent EPS value before this date
                valid_eps = ticker_eps[ticker_eps.index > date]
                
                #pdb.set_trace()
                if not valid_eps.empty:
                    # Get the most recent EPS value
                    latest_eps = valid_eps.iloc[0]["EPS TTM"]
                    inferred = False
                else:
                    inferred = True
            
                # add new column for eps
                normalized_prices.loc[(ticker, date), "EPS TTM"] = latest_eps
                normalized_prices.loc[(ticker, date), "inferred"] = inferred

                if latest_eps != 0:
                    # normalize the price data for OHLC
                    normalized_prices.loc[(ticker, date), "close (e)"] = row["close"] / latest_eps
                    normalized_prices.loc[(ticker, date), "open (e)"] = row["open"] / latest_eps
                    normalized_prices.loc[(ticker, date), "high (e)"] = row["high"] / latest_eps
                    normalized_prices.loc[(ticker, date), "low (e)"] = row["low"] / latest_eps
                else:
                    # set to NaN
                    normalized_prices.loc[(ticker, date), "close (e)"] = np.nan
                    normalized_prices.loc[(ticker, date), "open (e)"] = np.nan
                    normalized_prices.loc[(ticker, date), "high (e)"] = np.nan
                    normalized_prices.loc[(ticker, date), "low (e)"] = np.nan
        except Exception as e:
            print(f"Error: {e}")
            raise
            continue
    return normalized_prices

class SimFinClient:
    def __init__(self) -> None:
        pass

    def get_derived_data(self, ticker:list | str, start_date:datetime, end_date:datetime, condensed:bool=False) -> pd.DataFrame:
        """Get the derived data for a list of tickers
        """
        def get_derived_data_for_ticker(ticker:str, start_date:datetime=None, end_date:datetime=None) -> pd.DataFrame:
            """Get the derived data for a single ticker
            This uses TTM data.
            """
            # format the dates
            if start_date is not None:
                start_date = start_date.strftime("%Y-%m-%d")
            if end_date is not None:
                end_date = end_date.strftime("%Y-%m-%d")

            # make the request
            url = ( f"https://backend.simfin.com/api/v3/companies/statements/compact?ticker={ticker}"
                    f"&ttm=true&statements=DERIVED" )
            if start_date is not None:
                url += f"&start={start_date}"
            if end_date is not None:
                url += f"&end={end_date}"
            
            print(f"Requesting {url}")

            response = make_request(url, SIMFIN_API_KEY)

            # scanning the response
            item = response[0] 
            assert "statements" in item, "No statements found"
            assert item["ticker"] == ticker, "Ticker mismatch"
            
            # create a dataframe
            # statements is a list of dicts. find the one with DERIVED
            derived_statement = next((s for s in item["statements"] if s["statement"] == "DERIVED"), None)
            if derived_statement is None:
                raise ValueError(f"No derived statement found for ticker {ticker}")
            
            # get the columns and data
            columns = derived_statement["columns"]
            data = derived_statement["data"]
            df = pd.DataFrame(data, columns=columns)

            # convert Report Date to datetime
            df["Report Date"] = pd.to_datetime(df["Report Date"])

            # sort by Report Date
            df = df.sort_values(by="Report Date")

            # add ticker to the dataframe and make it the first column
            df.insert(0, "ticker", ticker)
            
            return df
        
        if isinstance(ticker, str):
            return get_derived_data_for_ticker(ticker)
        elif isinstance(ticker, list):
            dfs = []
            # rate limit
            for t in tqdm(ticker, desc="Derived Statements"):
                try:
                    df = get_derived_data_for_ticker(t, start_date, end_date)
                    dfs.append(df)
                    time.sleep(1/SIMFIN_RATE_LIMIT)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            ct = pd.concat(dfs)

            # add utc timezone to Report Date
            ct["Report Date"] = pd.to_datetime(ct["Report Date"]).dt.tz_localize("UTC")

            # set index to ticker and Report Date
            ct.set_index(["ticker", "Report Date"], inplace=True)


            if condensed:
                # only keep "Earnings Per Share (Basic)"
                ct = ct[["Fiscal Period", "Fiscal Year", "Earnings Per Share, Basic"]]
                # rename the column to "EPS"
                ct.rename(columns={"Earnings Per Share, Basic": "EPS TTM"}, inplace=True)
                # drop rows where EPS is None
                ct.dropna(subset=["EPS TTM"], inplace=True)
            return ct
        else:
            raise ValueError(f"Invalid ticker type: {type(ticker)}")

def plot_price_data(df:pd.DataFrame, title:str=None):
    """Plot the price data.
    """
    #import pdb; pdb.set_trace()
    series = df.loc[:, 'close (e)']
    
    plt.figure(figsize=(12, 8))
    plt.xlabel("Date")
    plt.ylabel("Price Adjusted for EPS")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.plot(series)
    # also plot the eps data on its own y axis
    plt.xticks(series.index[::len(series)//20], series.index[::len(series)//20].strftime('%Y-%m-%d'), rotation=45)
    ax2 = plt.twinx()
    ax2.plot(df.loc[:, 'EPS TTM'], color='red')
    ax2.set_ylabel("EPS TTM")
    plt.legend(["Price Adjusted for EPS", "EPS TTM"])
    plt.show()

if __name__ == "__main__":
    utility_tickers = ['NEE', 'EXC', 'D', 'PCG', 'XEL', 
                        'ED', 'WEC', 'DTE', 'PPL', 'AEE', 'CNP', 'FE', 'CMS', 
                        'EIX', 'ETR', 'EVRG', 'LNT', 'PNW', 'IDA']
    client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

    # Get the utility bars
    start_date = datetime(2023, 5, 1)
    end_date = datetime.today()
    request_params = StockBarsRequest(symbol_or_symbols=utility_tickers,
                                      timeframe=TimeFrame.Hour,
                                      start=start_date,
                                      end=end_date) # adjustment=all?
    # bars = client.get_stock_bars(request_params).df
    #save_data(bars, 'utility_bars.csv')
    bars = pd.read_pickle('data/utility_bars.pkl')

    simfin_client = SimFinClient()
    # derived_data = simfin_client.get_derived_data(utility_tickers, start_date=None, end_date=end_date, condensed=False)
    # save_data(derived_data, 'derived_data.csv')
    derived_data = pd.read_pickle('data/derived_data.pkl')

    derived_data = derived_data[["Fiscal Period", "Fiscal Year", "Earnings Per Share, Basic"]]
    derived_data.rename(columns={"Earnings Per Share, Basic": "EPS TTM"}, inplace=True)
    derived_data.dropna(subset=["EPS TTM"], inplace=True)
    derived_data.info()

    # normalize the price data by the derived data
    #normalized_data = normalize_price_data_by_derived_data(bars, derived_data)
    #save_data(normalized_data, 'normalized_data.csv')
    normalized_data = pd.read_pickle('data/normalized_data.pkl')
    normalized_data.info()

    # plot the price data
    plot_price_data(normalized_data.loc['NEE'], title="NEE Hourly Price Data")