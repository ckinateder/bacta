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
    is_market_open,
    BarUtils,
)
from sklearn.decomposition import PCA
# from arch.unitroot.cointegration import engle_granger
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from __init__ import Position
import pytz
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
        bars = BarUtils.resample_multi_ticker_bars(bars)
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
    print("Using Cointegration to select the pairs:")

    pair_selector = PairSelector()
    cointegration_results = pair_selector.select_pairs(
        close_prices, method="spread_adf"
    )

    # print the spreads
    print(cointegration_results)

    # show the cointegration results as a heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        cointegration_results.values.astype(np.float32),
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
    )
    plt.grid(False)
    plt.title("Cointegration Results")
    plt.xlabel("Ticker")
    plt.ylabel("Ticker")
    # add ticker labels
    plt.xticks(np.arange(len(utility_tickers)) +
               0.5, utility_tickers, rotation=30)
    plt.yticks(np.arange(len(utility_tickers)) +
               0.5, utility_tickers, rotation=0)
    plt_show(prefix="cointegration_results")

    # find the pair with the lowest p-value
    lowest_pvalue = cointegration_results.min().min()
    print(f"The lowest p-value from cointegration is {lowest_pvalue}")

    # remove duplicaets
    lowest_pairs = list(
        cointegration_results.stack().sort_values().head(10).index)

    # remove duplicate pairs; order of each pair is not important
    lowest_pairs = list(set([tuple(sorted(pair)) for pair in lowest_pairs]))

    # put bollinger bands
    for lowest_pair in lowest_pairs:
        print(f"Plotting {lowest_pair}")
        primary, secondary = lowest_pair
        prices = close_prices[[primary, secondary]]  # .iloc[-1000:]
        normalized_spread = BarUtils.put_normalized_spread(prices)
        bands = BarUtils.put_bollinger_bands(normalized_spread)
        bands["position"] = np.where(normalized_spread > bands["upper_band"], Position.LONG.value, np.where(
            normalized_spread < bands["lower_band"], Position.SHORT.value, Position.NEUTRAL.value))

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(15, 12), height_ratios=[2, 0.5, 1]
        )

        # Plot spread in the largest subplot
        rolling_mean = bands["rolling_mean"]
        upper_band = bands["upper_band"]
        lower_band = bands["lower_band"]
        normalized_spread = bands["normalized_spread"]
        position = bands["position"]
        normalized_spread.plot(
            ax=ax1, title=f"Normalized Spread of {primary} and {secondary}"
        )
        rolling_mean.plot(
            ax=ax1, label="Rolling Mean", color="red", linewidth=0.5, alpha=0.8
        )
        upper_band.plot(
            ax=ax1, label="Upper Band", color="green", linewidth=0.5, alpha=0.8
        )
        lower_band.plot(
            ax=ax1, label="Lower Band", color="green", linewidth=0.5, alpha=0.8
        )
        ax1.set_ylabel("Quantity")
        ax1.set_xlabel("")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend()
        ax1.tick_params(axis="x", which="both",
                        bottom=False, labelbottom=False)

        # Plot spread direction in the middle subplot
        position.plot(ax=ax2, title=f"Signal of {primary} and {secondary}")
        ax2.set_ylabel("Signal")
        ax2.set_xlabel("")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.tick_params(axis="x", which="both",
                        bottom=False, labelbottom=False)

        # Plot individual prices in the bottom subplot
        prices.plot(ax=ax3, title=f"Price of {primary} and {secondary} ($)")
        ax3.set_ylabel("Price ($)")
        ax3.set_xlabel("Date")
        ax3.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt_show(prefix=f"spread_and_prices_{primary}_{secondary}")
