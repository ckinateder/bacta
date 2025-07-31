from datetime import datetime, timedelta
from itertools import combinations

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

from __init__ import *
from utilities import get_logger
from backtester import Position
from utilities import getenv
from utilities.bars import BarUtils, download_close_prices
from utilities.market import get_earnings_date, eastern
from utilities.plotting import plot_price_data, plt_show, DEFAULT_FIGSIZE
load_dotenv()

ALPACA_API_KEY = getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = getenv("ALPACA_API_SECRET")

plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["axes.grid"] = True

logger = get_logger("pairs_selection")


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
            close_prices (pd.DataFrame): A dataframe of close prices for the symbols.

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
        symbols = close_prices.columns
        stationarity_results = pd.DataFrame(index=symbols, columns=symbols)
        for primary, secondary in combinations(symbols, 2):
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
            close_prices (pd.DataFrame): A dataframe of close prices for the symbols.

        Returns:
            pd.DataFrame: A dataframe of the pairs with the lowest spread.
        """
        if not close_prices.index.is_unique:
            raise ValueError("The close prices must have unique timestamps.")
        if not close_prices.index.is_monotonic_increasing:
            raise ValueError(
                "The close prices must be monotonically increasing.")

        symbols = close_prices.columns
        stationarity_results = pd.DataFrame(index=symbols, columns=symbols)
        for primary, secondary in combinations(symbols, 2):
            if stationarity_results.loc[secondary, primary] is not np.nan:
                stationarity_results.loc[primary, secondary] = stationarity_results.loc[
                    secondary, primary
                ]
                continue
            #


if __name__ == "__main__":
    utility_symbols = [
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
        symbol_or_symbols=utility_symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="all",
    )

    refresh_bars = False
    # if refresh_bars:
    # earnings_dates = {
    #    symbol: get_earnings_date(symbol) for symbol in utility_symbols
    # }
    # save_json(earnings_dates, "utility_earnings_dates")

    close_prices = download_close_prices(
        utility_symbols, start_date, end_date, timeframe)
    # earnings_dates = load_json("utility_earnings_dates")
    plot_price_data(close_prices, "Utility Price Data")
    plt_show(prefix="utility_price_data")

    # train test split by rate
    split = 0.8
    split_index = int(split * len(close_prices))
    train_prices = close_prices.iloc[:split_index]
    test_prices = close_prices.iloc[split_index:]

    #########################################################
    logger.info("Using Cointegration to select the pairs:")

    pair_selector = PairSelector()
    cointegration_results = pair_selector.select_pairs(
        train_prices, method="spread_adf"
    )

    # logger.info the spreads
    logger.info(cointegration_results)

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
    plt.xlabel("symbol")
    plt.ylabel("symbol")
    # add symbol labels
    plt.xticks(np.arange(len(utility_symbols)) +
               0.5, utility_symbols, rotation=30)
    plt.yticks(np.arange(len(utility_symbols)) +
               0.5, utility_symbols, rotation=0)
    plt_show(prefix="cointegration_results")

    # find the pair with the lowest p-value
    lowest_pvalue = cointegration_results.min().min()
    logger.info(f"The lowest p-value from cointegration is {lowest_pvalue}")

    # remove duplicaets
    lowest_pairs = list(
        cointegration_results.stack().sort_values().head(10).index)

    # remove duplicate pairs; order of each pair is not important
    lowest_pairs = list(set([tuple(sorted(pair)) for pair in lowest_pairs]))

    # put bollinger bands
    for lowest_pair in lowest_pairs:
        logger.info(f"Plotting {lowest_pair}")
        primary, secondary = lowest_pair
        prices = close_prices[[primary, secondary]]  # .iloc[-1000:]
        normalized_prices = prices.apply(lambda x: (x / x.mean()))
        normalized_spread = normalized_prices[primary] - \
            normalized_prices[secondary]
        bands = BarUtils.create_bollinger_bands(normalized_spread)
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
        normalized_spread = bands["series"]
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
