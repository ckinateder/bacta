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
ept = pytz.timezone('US/Eastern')
utc = pytz.utc

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["axes.grid"] = True


def resample_multi_ticker_bars(
    bars: pd.DataFrame, time_frame: timedelta = timedelta(hours=1)
) -> pd.DataFrame:
    """Resample the bars to the given time frame and fill the missing values.
    For OHLC, use the last value in the time frame to fill forward.
    For volume and trade count, fill with 0.
    THIS IS FOR MULTIPLE TICKERS. Use groupby to apply to multiple tickers.
    """
    # Create a list to store resampled data for each ticker
    resampled_data = []

    # Get unique tickers
    tickers = bars.index.get_level_values("symbol").unique()

    latest_start = None
    earliest_end = None

    # Process each ticker
    for ticker in tickers:
        # Get data for this ticker
        ticker_data = bars.loc[ticker]
        # start and end of the ticker data
        start = ticker_data.index.get_level_values("timestamp").min()
        end = ticker_data.index.get_level_values("timestamp").max()
        if latest_start is None or start > latest_start:
            latest_start = start
        if earliest_end is None or end < earliest_end:
            earliest_end = end

        # Resample the data
        resampled_ticker = resample_bars(ticker_data, time_frame)
        # Add the ticker back to the index
        resampled_ticker["symbol"] = ticker
        resampled_data.append(resampled_ticker)

    # trim the data to the latest start and earliest end
    for i in range(len(resampled_data)):
        data = resampled_data[i]
        resampled_data[i] = data.loc[latest_start:earliest_end]

    # Combine all resampled data
    result = pd.concat(resampled_data)
    # Set the multi-index
    result.set_index(["symbol", result.index], inplace=True)

    return result


def resample_bars(
    bars: pd.DataFrame, time_frame: timedelta = timedelta(hours=1)
) -> pd.DataFrame:
    """Resample the bars to the given time frame and fill the missing values.
    For OHLC, use the last value in the time frame to fill forward.
    For volume and trade count, fill with 0.
    THIS IS FOR ONE TICKER. Use groupby to apply to multiple tickers.
    """
    # resample the bars
    part1 = bars[["open", "high", "low", "close", "vwap"]]
    part2 = bars[["volume", "trade_count"]]
    part1 = part1.resample(time_frame).last().ffill()
    part2 = part2.resample(time_frame).asfreq().fillna(0)
    bars = pd.concat([part1, part2], axis=1)
    return bars


class PairSelector:
    def __init__(self):
        pass

    def select_pairs(self, close_prices: pd.DataFrame, method: str = "spread_adf"):
        if method == "spread_adf":
            return self.select_pairs_by_spread_adf(close_prices)
        else:
            raise ValueError(f"Invalid method: {method}")

    def select_pairs_by_spread_adf(self, close_prices: pd.DataFrame) -> dict:
        """Given a set of close prices, select the pairs with the lowest spread.

        Args:
            close_prices (pd.DataFrame): A dataframe of close prices for the tickers.

        Returns:
            dict:
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

            # normalize the prices amd calculate the spread
            normalized_prices = close_prices[[primary, secondary]].apply(
                lambda x: (x / x.mean())
            )
            spread = normalized_prices[primary] - normalized_prices[secondary]

            # calculate how stationary the spread is
            spread_pvalue = adfuller(spread, maxlag=0)[1]

            stationarity_results.loc[primary, secondary] = spread_pvalue
            stationarity_results.loc[secondary, primary] = spread_pvalue

        return stationarity_results


class BarUtils:
    def __init__(self):
        pass

    @staticmethod
    def put_normalized_spread(prices: pd.DataFrame) -> pd.DataFrame:
        """Put the normalized spread on the prices. 
        Args:
            prices (pd.DataFrame): A dataframe of TWO tickers.

        Returns:
            pd.DataFrame: A dataframe with the normalized spread.
        """
        if len(prices.columns) != 2:
            raise ValueError("The prices must have two tickers.")
        primary, secondary = prices.columns
        normalized_prices = prices.apply(lambda x: (x / x.mean()))
        normalized_spread = normalized_prices[primary] - \
            normalized_prices[secondary]
        return normalized_spread

    @staticmethod
    def put_bollinger_bands(normalized_spread: pd.DataFrame, rolling_window: int = 8, std_multiplier: float = 1) -> pd.DataFrame:
        """Put bollinger bands on the normalized spread.

        Args:
            normalized_spread (pd.DataFrame): _description_
            rolling_window (int, optional): _description_. Defaults to 8.
            std_multiplier (float, optional): _description_. Defaults to 1.

        Returns:
            pd.DataFrame: A dataframe with the bollinger bands and position. Columns are:
                - normalized_spread: The normalized spread
                - rolling_mean: The rolling mean of the normalized spread
                - upper_band: The upper band of the normalized spread
                - lower_band: The lower band of the normalized spread
        """
        # make bands
        # This is used for determining how many days ahead to use to calculate the rolling mean
        rolling_window = 8
        std_multiplier = 1
        rolling_mean = (
            normalized_spread.rolling(window=rolling_window).mean()
        ).dropna()
        rolling_std = (normalized_spread.rolling(
            window=rolling_window).std()).dropna()
        upper_band = rolling_mean + (rolling_std * std_multiplier)
        lower_band = rolling_mean - (rolling_std * std_multiplier)

        # fix the bands and rollings to be the same length as the spread
        rolling_mean = rolling_mean.reindex(normalized_spread.index)
        upper_band = upper_band.reindex(normalized_spread.index)
        lower_band = lower_band.reindex(normalized_spread.index)

        output = pd.DataFrame(index=normalized_spread.index, columns=[
                              "normalized_spread", "rolling_mean", "upper_band", "lower_band"])
        output["normalized_spread"] = normalized_spread
        output["rolling_mean"] = rolling_mean
        output["upper_band"] = upper_band
        output["lower_band"] = lower_band
        return output


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
    lowest_pairs += [
        (lowest, highest)
    ]  # add the highest and lowest weighting assets from PCA test

    # put bollinger bands
    print("Putting bollinger bands on the lowest pairs")
    for lowest_pair in lowest_pairs:
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
