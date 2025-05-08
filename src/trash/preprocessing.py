import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DEFAULT_FIG_SIZE = (10, 5)

def load_sp500_data(file_path: str) -> pd.DataFrame:
    """
    Load the SP500 data from the given file path.
    We want to return a multi-index dataframe. We have 500 stocks and an fixed number of features for each stock for each day.
    """
    df = pd.read_csv(file_path)
    df.set_index(['Date', 'Symbol'], inplace=True)
    # sort alphabetically by symbol
    df = df.sort_index(level='Symbol')
    # remove all rows with NaN values
    df = df.dropna()
    
    return df

def plot_stock(df: pd.DataFrame, symbol: str):
    plt.figure(figsize=DEFAULT_FIG_SIZE)

    stock_data = df.xs(symbol, level="Symbol")
    close_prices = stock_data["Close"]

    plt.plot(stock_data.index, close_prices)
    
    # put 10 points on the x-axis
    plt.xticks(stock_data.index[::len(stock_data)//10], rotation=45)

    # dot grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.title(symbol)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()

if __name__ == "__main__":
    df = load_sp500_data(os.path.join("data", "sp500_dataset", "sp500_stocks.csv"))

    print(df.index.get_level_values(1).unique())
    print(df.xs("ABBV", level="Symbol"))

    plot_stock(df, "ABBV")