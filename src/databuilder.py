## Collection of functions to load data from the web

import pandas as pd
import yfinance as yf
import simfin as sf
import requests
from simfin.names import *
import os
import numpy as np
from dotenv import load_dotenv
import pdb
from tqdm import tqdm, trange
load_dotenv()  # take environment variables
SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY")
sf.set_api_key(SIMFIN_API_KEY)
sf.set_data_dir(os.getenv("SIMFIN_DATA_DIR"))

def save_to_csv(df:pd.DataFrame, filename:str):
    """Save a dataframe to a csv file
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    df.to_csv(filename)

class SimFinWrangler:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calculate_eps_from_income(variant:str="quarterly") -> pd.DataFrame:
        """Calculate earnings per share from income and create 
        a new dataframe indexed, by SimFinId with new columns
        'Earnings Per Share (Basic)' and 'Earnings Per Share (Diluted)'

        Returns:
            pd.DataFrame: 27 columns
        """
        # load income
        df_income = sf.load_income(variant=variant, market='us')
        df_income["Earnings Per Share (Basic)"] = df_income["Net Income (Common)"] / df_income["Shares (Basic)"]
        df_income["Earnings Per Share (Diluted)"] = df_income["Net Income (Common)"] / df_income["Shares (Diluted)"]

        return df_income

    @staticmethod
    def add_industry_to_companies(companies:pd.DataFrame, industries:pd.DataFrame=None) -> pd.DataFrame:
        """Add industry to companies dataframe
        """
        if industries is None:
            industries = sf.load_industries()
        # Preserve the original index
        original_index = companies.index
        # Merge industry info into companies dataframe using IndustryId
        companies = companies.merge(industries, on='IndustryId', how='left')
        # Restore the original index
        companies.index = original_index
        companies.dropna(inplace=True)
        return companies

    @staticmethod
    def calculate_normalized_prices(share_prices:pd.DataFrame, eps:pd.DataFrame, filter:list=None) -> pd.DataFrame:
        """Calculate normalized prices using the daily share prices and quarterly earnings per share values (basic).
        For each date interval, divide the daily share price by the EPS value for that ticker.

        Args:
            share_prices (pd.DataFrame): _description_
            eps (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        eps_small = eps[["SimFinId", "Earnings Per Share (Basic)"]]
        print(eps_small.index)
        # Create copy of share prices dataframe
        normalized_prices = share_prices.copy()
        
        # Get unique SimFinIds from share prices
        tickers = normalized_prices.index.get_level_values('Ticker').unique()

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
                        latest_eps = valid_eps.iloc[-1]["Earnings Per Share (Basic)"]
                        
                        # Avoid division by zero or negative EPS
                        if latest_eps > 0:
                            # Calculate normalized price (P/E ratio)
                            normalized_prices.loc[(ticker, date), "Close"] = row["Close"] / latest_eps
                        else:
                            # Set to NaN if EPS is zero or negative
                            normalized_prices.loc[(ticker, date), "Close"] = np.nan
                    else:
                        # No valid EPS data for this date
                        normalized_prices.loc[(ticker, date), "Close"] = np.nan
            except KeyError:
                # Skip if ticker not found in EPS data
                continue
                
        return normalized_prices


if __name__ == "__main__":
    # get list of companies
    companies = sf.load_companies(market="us")
    df_prices_latest = sf.load_shareprices(variant='latest', market='us')
    df_industries = sf.load_industries()
    df_prices = sf.load_shareprices(variant='daily', market='us')
    
    companies = SimFinWrangler.add_industry_to_companies(companies)
    print(companies)

    # filter companies for sector Utilities
    utility_companies = companies[companies["Sector"] == "Utilities"]
    utility_tickers = utility_companies.index.to_list()
    
    eps = SimFinWrangler.calculate_eps_from_income()
    normalized_prices = SimFinWrangler.calculate_normalized_prices(df_prices, eps, filter=utility_tickers)
    save_to_csv(normalized_prices, os.path.join(os.getenv("DATA_DIR"), "normalized_prices.csv"))
