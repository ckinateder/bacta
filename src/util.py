import os
import pandas as pd
import json
from names import SEC_TICKERS_FILENAME
from dotenv import load_dotenv
import matplotlib.pyplot as plt
load_dotenv()

def load_sec_tickers(data_dir:str=os.getenv("DATA_DIR")) -> pd.DataFrame:
    """Load the CIK table downloaded from https://www.kaggle.com/datasets/svendaj/sec-edgar-cik-ticker-exchange.

    Args:
        data_dir (str, optional): The directory to save the CIK table. Defaults to os.getenv("DATA_DIR").

    Returns:
        pd.DataFrame: The CIK table. Index is cik, columns are name, ticker, and exchange.
    """
    path = os.path.join(data_dir, SEC_TICKERS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    
    # Load the JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(data=data["data"], columns=data["fields"])
    df.set_index("cik", inplace=True)
    return df

def save_data(df:pd.DataFrame, filename:str, data_dir:str=os.getenv("DATA_DIR")):
    """Save the data to a CSV and pickle file.
    """
    path = os.path.join(data_dir, filename)
    df.to_csv(path+".csv")
    df.to_pickle(path+".pkl")

if __name__ == "__main__":
    sec_tickers = load_sec_tickers()
