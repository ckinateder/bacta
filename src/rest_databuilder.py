# Use the REST API to get the data: https://simfin.readme.io/reference/
# Tried to use the python wrapper, but it was not working (as per usual. ALWAYS USE REST API)

import requests
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import pdb
from tqdm import tqdm
import time
import warnings
import numpy as np
load_dotenv()

SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY")
SIMFIN_RATE_LIMIT = 2 # PER SECOND
"""Steps:
1. Get company list (simfin)
2. Get utility company sector list (~80 tickers)
3. Download derived data for these companies (simfin?)
4. Download daily price data for all companies (polygon)
5. Calculate normalized price data (P/E) 
6. Save to csv
"""


def make_request(url:str, key:str) -> pd.DataFrame:
    headers = {
        "accept": "application/json",
        "Authorization": key
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} {response.text}")
    return response.json()

class SimFinWrangler:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_company_list(remove_delisted:bool=True) -> pd.DataFrame:
        """Get the list of companies in the US market
        """
        url = "https://backend.simfin.com/api/v3/companies/list"

        df = pd.DataFrame(make_request(url, SIMFIN_API_KEY))
        df.set_index("ticker", inplace=True)
        df.rename(columns={"sectorName": "sector", "industryName": "industry"}, inplace=True)

        if remove_delisted:
            todrop = []
            for ticker in df.index:
                if ticker == None or "delisted" in ticker.lower() or "old" in ticker.lower():
                    todrop.append(ticker)
            print(f"Dropping {len(todrop)} delisted or old companies")
            # drop the companies where index is in todrop
            df = df[~df.index.isin(todrop)]

        return df

    @staticmethod
    def get_derived_data(ticker:list | str) -> pd.DataFrame:
        """Get the derived data for a list of tickers
        """
        def get_derived_data_for_ticker(ticker:str) -> pd.DataFrame:
            """Get the derived data for a single ticker
            """
            url = f"https://backend.simfin.com/api/v3/companies/statements/compact?ticker={ticker}&statements=DERIVED"
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
            
            columns = derived_statement["columns"]
            data = derived_statement["data"]
            df = pd.DataFrame(data, columns=columns)

            # convert Report Date to datetime
            df["Report Date"] = pd.to_datetime(df["Report Date"])
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
                    df = get_derived_data_for_ticker(t)
                    dfs.append(df)
                    time.sleep(1/SIMFIN_RATE_LIMIT)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            ct = pd.concat(dfs)
            # set index to ticker and Report Date
            ct.set_index(["ticker", "Report Date"], inplace=True)
            return ct
        else:
            raise ValueError(f"Invalid ticker type: {type(ticker)}")


    def get_price_data(ticker:list | str) -> pd.DataFrame:
        """Get the price data for a list of tickers
        """
        def get_price_data_for_ticker(ticker:str) -> pd.DataFrame:
            """Get the price data for a single ticker
            """
            url = f"https://backend.simfin.com/api/v3/companies/prices/compact?ticker={ticker}"
            response = make_request(url, SIMFIN_API_KEY)
            if len(response) == 0:
                raise ValueError(f"No price data found for ticker {ticker}")
            
            item = response[0] 
            assert item["ticker"] == ticker, "Ticker mismatch"

            columns = item["columns"]
            data = item["data"]
            df = pd.DataFrame(data, columns=columns)

            df["Date"] = pd.to_datetime(df["Date"])
            df.insert(0, "ticker", ticker)
            return df

        if isinstance(ticker, str):
            return get_price_data_for_ticker(ticker)
        elif isinstance(ticker, list):
            dfs = []
            for t in tqdm(ticker, desc="Price Data"):
                try:
                    df = get_price_data_for_ticker(t)
                    dfs.append(df)
                    time.sleep(1/SIMFIN_RATE_LIMIT)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            ct = pd.concat(dfs)
            # set index to ticker and Date
            ct.set_index(["ticker", "Date"], inplace=True)
            return ct
        else:
            raise ValueError(f"Invalid ticker type: {type(ticker)}")
        
    @staticmethod
    def calculate_normalized_prices(share_prices:pd.DataFrame, derived_data:pd.DataFrame, filter:list=None) -> pd.DataFrame:
        """Calculate normalized prices using the daily share prices and quarterly earnings per share values (basic).
        For each date interval, divide the daily share price by the EPS value for that ticker.

        Args:
            share_prices (pd.DataFrame): _description_
            eps (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        eps_small = derived_data[["Earnings Per Share, Basic"]]
        # Create copy of share prices dataframe
        normalized_prices = share_prices.copy()
       
        tickers = normalized_prices.index.get_level_values('ticker').unique()

        # filter
        if filter != None:
            tickers = list(set(filter) & set(tickers))
            normalized_prices = normalized_prices.loc[tickers]

        print(f"Normalizing prices for {tickers}")
        
        # For each ticker
        for ticker in tqdm(tickers, desc="Normalizing Prices"):
            try:
                # Get share prices for this ticker
                ticker_prices = normalized_prices.loc[ticker]
                
                # Get EPS values for this ticker
                ticker_eps = eps_small.loc[ticker]

                # For each date in ticker_prices
                for date, row in ticker_prices.iterrows():
                    # Find the most recent EPS value before this date
                    valid_eps = ticker_eps[ticker_eps.index <= date]
                    
                    if not valid_eps.empty:
                        # Get the most recent EPS value
                        latest_eps = valid_eps.iloc[-1]["Earnings Per Share, Basic"]
                        
                        # Avoid division by zero or negative EPS
                        if latest_eps > 0:
                            # Calculate normalized price (P/E ratio), in a new column
                            normalized_prices.loc[(ticker, date), "Opening Price (E)"] = row["Opening Price"] / latest_eps
                            normalized_prices.loc[(ticker, date), "Highest Price (E)"] = row["Highest Price"] / latest_eps
                            normalized_prices.loc[(ticker, date), "Lowest Price (E)"] = row["Lowest Price"] / latest_eps
                            normalized_prices.loc[(ticker, date), "Last Closing Price (E)"] = row["Last Closing Price"] / latest_eps
                            normalized_prices.loc[(ticker, date), "Adjusted Closing Price (E)"] = row["Adjusted Closing Price"] / latest_eps
                        else:
                            # Set to NaN if EPS is zero or negative
                            normalized_prices.loc[(ticker, date), "Opening Price (E)"] = np.nan
                            normalized_prices.loc[(ticker, date), "Highest Price (E)"] = np.nan
                            normalized_prices.loc[(ticker, date), "Lowest Price (E)"] = np.nan
                            normalized_prices.loc[(ticker, date), "Last Closing Price (E)"] = np.nan
                            normalized_prices.loc[(ticker, date), "Adjusted Closing Price (E)"] = np.nan
                    else:
                        # No valid EPS data for this date
                        normalized_prices.loc[(ticker, date), "Opening Price (E)"] = np.nan
                        normalized_prices.loc[(ticker, date), "Highest Price (E)"] = np.nan
                        normalized_prices.loc[(ticker, date), "Lowest Price (E)"] = np.nan
                        normalized_prices.loc[(ticker, date), "Last Closing Price (E)"] = np.nan
                        normalized_prices.loc[(ticker, date), "Adjusted Closing Price (E)"] = np.nan
            except KeyError:
                # Skip if ticker not found in EPS data
                warnings.warn(f"Ticker {ticker} missing from EPS data")

        # if any rows for a ticker are NaN, drop the ticker
        normalized_prices = normalized_prices[normalized_prices.notna().any(axis=1)]
        return normalized_prices


if __name__ == "__main__":
    # get list of companies
    companies = SimFinWrangler.get_company_list()
    print(companies)

    # filter companies for sector Utilities
    #utility_companies = companies[companies["industry"] == "Utilities"]
    #utility_tickers = utility_companies.index.to_list()
    utility_tickers = ['AEE', 'PPL', 'D', 'CPK', 'CNP', 'EVRG', 'GNE', 'NWN', 
                       'SJW', 'PCYO', 'EXC', 'AWR', 'IDA', 'PEG', 'BKH', 'SBS', 
                       'POR', 'PCG', 'NEP', 'PNM', 'OGE', 'LNT', 'HE', 'NRG', 
                       'MNTK', 'MSEX', 'KEP', 'NI', 'MGEE', 'ED', 'OTTR', 
                       'FE', 'CWCO', 'GWRS', 'ELP', 'WTRG', 'OGS', 'ETR', 'FCEL', 
                       'CMS', 'ALE', 'EDN', 'AEP', 'SPH', 'NEE', 'SWX', 
                       'MDU', 'WEC', 'ATO', 'VST', 'AGR', 'VWTR', 'RGCO', 'NJR',
                       'AES', 'DTE', 'NFE', 'SJI', 'CWEN', 'ES', 'ORA', 'NWE', 
                       'CNLPL', 'DUK', 'AWK', 'BIP', 'SO', 'PNW', 'EIX', 'SR', 
                       'ARIS', 'AVA', 'UTL', 'XEL', 'ARTNA', 'SRE', 'CWT']
    
    utility_tickers = ['NEE', 'SO', 'DUK', 'AEP', 'EXC', 'D', 'PCG', 'XEL', 
                       'ED', 'WEC', 'DTE', 'PPL', 'AEE', 'CNP', 'FE', 'CMS', 
                       'EIX', 'ETR', 'EVRG', 'LNT', 'PNW', 'OGE', 'IDA']

    # get derived data
    if not os.path.exists("data/derived_data.csv"):
        derived_data = SimFinWrangler.get_derived_data(utility_tickers)
        derived_data.to_csv("data/derived_data.csv")
    else:
        derived_data = pd.read_csv("data/derived_data.csv")
        derived_data.set_index(["ticker", "Report Date"], inplace=True)
    # get price data
    if not os.path.exists("data/price_data.csv"):
        price_data = SimFinWrangler.get_price_data(utility_tickers)
        price_data.to_csv("data/price_data.csv")
    else:
        price_data = pd.read_csv("data/price_data.csv")
        price_data.set_index(["ticker", "Date"], inplace=True)

    # normalize prices
    normalized_prices = SimFinWrangler.calculate_normalized_prices(price_data, derived_data, filter=utility_tickers)
    normalized_prices.to_csv("data/normalized_prices.csv")