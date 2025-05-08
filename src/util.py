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

def plot_price_data(series:pd.Series, title:str=None, xlabel:str=None, ylabel:str=None):
    """Plot the price data.
    """
    #import pdb; pdb.set_trace()
    
    plt.figure(figsize=(10, 5))
    plt.plot(series)
    # dotted grid
    plt.grid(True, linestyle='--', alpha=0.7)
    # set the x tick labels to be just the date, use strftime. only show a total of 20 ticks
    plt.xticks(series.index[::len(series)//20], series.index[::len(series)//20].strftime('%Y-%m-%d'), rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    sec_tickers = load_sec_tickers()
