import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from util import (
    save_dataframe,
    plt_show,
    plot_price_data,
    get_earnings_date,
    load_dataframe,
    save_json,
    load_json,
)
from sklearn.decomposition import PCA
# from arch.unitroot.cointegration import engle_granger
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from __init__ import Position
import pytz
from bar_ops import BarUtils, resample_multi_ticker_bars, resample_bars
from sklearn.linear_model import LinearRegression

ept = pytz.timezone('US/Eastern')
utc = pytz.utc

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["axes.grid"] = True


class PairSelector:
    def __init__(self):
        pass

    def select_pairs(self, close_prices: pd.DataFrame, method: str = "spread_adf"):
        if method == "spread_adf":
            return self.select_pairs_by_spread_adf(close_prices)
        else:
            raise ValueError(f"Invalid method: {method}")

    def select_pairs_by_spread_adf(self, close_prices: pd.DataFrame) -> pd.DataFrame:
        """Given a set of close prices, select the pairs with the lowest spread.

        Args:
            close_prices (pd.DataFrame): A dataframe of close prices for the tickers.

        Returns:
            pd.DataFrame: A dataframe of the pairs with the lowest spread.
        """
        # check for equal spacing between the close prices
        if not close_prices.index.is_unique:
            raise ValueError("The close prices must have unique timestamps.")
        if not close_prices.index.is_monotonic_increasing:
            raise ValueError(
                "The close prices must be monotonically increasing.")

        # stationarity testing
        tickers = close_prices.columns
        stationarity_results = pd.DataFrame(index=tickers, columns=tickers)
        for primary, secondary in combinations(tickers, 2):
            if stationarity_results.loc[secondary, primary] is not np.nan:
                stationarity_results.loc[primary, secondary] = stationarity_results.loc[
                    secondary, primary
                ]
                continue

            # normalize the prices and calculate the spread
            normalized_prices = close_prices[[primary, secondary]].apply(
                lambda x: (x / x.mean()))
            spread = normalized_prices[primary] - normalized_prices[secondary]

            # calculate how stationary the spread is
            spread_pvalue = adfuller(spread, maxlag=0)[1]

            stationarity_results.loc[primary, secondary] = spread_pvalue
            stationarity_results.loc[secondary, primary] = spread_pvalue

        return stationarity_results

    def select_pairs_by_spread_regression_adf(self, close_prices: pd.DataFrame) -> pd.DataFrame:
        """Given a set of close prices, select the pairs with the lowest spread.
        Coeffecients are selected by linear regression.

        Args:
            close_prices (pd.DataFrame): A dataframe of close prices for the tickers.

        Returns:
            pd.DataFrame: A dataframe of the pairs with the lowest spread.
        """
        if not close_prices.index.is_unique:
            raise ValueError("The close prices must have unique timestamps.")
        if not close_prices.index.is_monotonic_increasing:
            raise ValueError(
                "The close prices must be monotonically increasing.")

        tickers = close_prices.columns
        stationarity_results = pd.DataFrame(index=tickers, columns=tickers)
        for primary, secondary in combinations(tickers, 2):
            if stationarity_results.loc[secondary, primary] is not np.nan:
                stationarity_results.loc[primary, secondary] = stationarity_results.loc[
                    secondary, primary
                ]
                continue
            #


if __name__ == "__main__":
    utility_tickers = [
        "NEE",
        "EXC",
        "D",
        "PCG",
        "XEL",
        "ED",
        "WEC",
        "DTE",
        "PPL",
        "AEE",
        "CNP",
        "FE",
        "CMS",
        "EIX",
        "ETR",
        "EVRG",
        "LNT",
        "PNW",
        "IDA",
        "AEP",
        "DUK",
        "SRE",
        "ATO",
        "NRG",
    ]

    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET
    )

    # Get the utility bars
    start_date = datetime(2023, 1, 1)
    end_date = datetime.today() - timedelta(minutes=15)
    timeframe = TimeFrame.Hour
    request_params = StockBarsRequest(
        symbol_or_symbols=utility_tickers,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="all",
    )

    refresh_bars = False
    if refresh_bars:
        bars = client.get_stock_bars(request_params).df  # get the bars
        save_dataframe(bars, "utility_bars")  # save the bars
        bars = load_dataframe("utility_bars")  # load the bars

        # convert all the dates to est. this is a multi-index dataframe, so we need to convert the index
        bars.index = bars.index.map(lambda x: (x[0], x[1].astimezone(ept)))

        # resample the bars. apply to each ticker
        bars = resample_multi_ticker_bars(bars)
        save_dataframe(bars, "utility_bars_resampled")

        # get the close prices
        close_prices = bars["close"].unstack(level=0)
        # save the close prices
        save_dataframe(close_prices, "utility_close_prices")

        earnings_dates = {
            ticker: get_earnings_date(ticker) for ticker in utility_tickers
        }
        save_json(earnings_dates, "utility_earnings_dates")

    close_prices = load_dataframe("utility_close_prices")
    earnings_dates = load_json("utility_earnings_dates")
    plot_price_data(close_prices, "Utility Price Data")
    plt_show(prefix="utility_price_data")

    # train test split by rate
    split = 0.8
    split_index = int(split * len(close_prices))
    train_prices = close_prices.iloc[:split_index]
    test_prices = close_prices.iloc[split_index:]

    #########################################################
    # PCA
    returns = close_prices.apply(lambda x: np.log(x / x.shift(1))).iloc[1:]
    pca = PCA()
    pca.fit(returns)
    components = [str(x + 1) for x in range(pca.n_components_)]
    explained_variance_pct = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(15, 10))
    plt.bar(components, explained_variance_pct)
    plt.title("Ratio of Explained Variance")
    plt.xlabel("Principle Component #")
    plt.ylabel("%")
    plt_show(prefix="pca_explained_variance")

    # most important component
    first_component = pca.components_[0, :]
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(4):
        component = pca.components_[i, :]
        axes[i].bar(utility_tickers, component)
        axes[i].set_title(f"Weightings of each asset in component {i+1}")
        axes[i].set_xlabel("Assets")
        axes[i].set_ylabel("Weighting")
        axes[i].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt_show(prefix="pca_components")

    highest_index = abs(first_component).argmax()
    lowest_index = abs(first_component).argmin()
    highest = utility_tickers[highest_index]
    lowest = utility_tickers[lowest_index]
    print("Using PCA to select the highest and lowest weighting assets:")
    print(
        f"The highest weighting is {highest} with a weighting of {first_component[highest_index]}"
    )
    print(
        f"The lowest weighting is {lowest} with a weighting of {first_component[lowest_index]}"
    )
    print("--------------------------------")
