## Collection of functions to load data from the web

import pandas as pd
import yfinance as yf
import simfin as sf
import requests
from simfin.names import *
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables
SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY")
sf.set_api_key(SIMFIN_API_KEY)
sf.set_data_dir(os.getenv("SIMFIN_DATA_DIR"))

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

if __name__ == "__main__":
    # get list of companies
    companies = sf.load_companies(market="us")
    df_prices_latest = sf.load_shareprices(variant='latest', market='us')
    df_industries = sf.load_industries()
    
    eps = SimFinWrangler.calculate_eps_from_income()
    print(eps)

    print(companies)