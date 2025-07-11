import pandas as pd
import numpy as np
from datetime import timedelta


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
