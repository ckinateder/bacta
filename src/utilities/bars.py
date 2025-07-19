import os
from datetime import datetime, timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pytz
import warnings
from dotenv import load_dotenv

# path wrangling
try:
    from src.utilities import *
except ImportError:
    from __init__ import *

from src.utilities import load_dataframe, save_dataframe, getenv
from src.utilities.market import eastern


load_dotenv()


def download_close_prices(tickers: list[str], start_date: datetime, end_date: datetime,
                          timeframe: TimeFrame, refresh_bars: bool = False,
                          data_dir: str = getenv("DATA_DIR"), filename: str | None = None) -> pd.DataFrame:
    """Download the bars for the given tickers. Will first check if the bars are already downloaded and if not, will download them.
    This requires an Alpaca API key.
    Args:
        tickers (list[str]): The tickers to download bars for.
        start_date (datetime): The start date to download bars for.
        end_date (datetime): The end date to download bars for.
        timeframe (TimeFrame): The timeframe to download bars for.
        refresh_bars (bool): Whether to force refresh the bars. If False, will check if the bars are already downloaded and if not, will download them.
        data_dir (str): The directory to save the bars.
        filename (str): The filename to save the bars to. If None, will be a combination of the tickers, start date, end date, and timeframe.

    Returns:
        pd.DataFrame: The close prices for the given tickers, resampled to the given timeframe.
    """
    ALPACA_API_KEY = getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET = getenv("ALPACA_API_SECRET")
    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET
    )
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="all",
    )
    if filename is None:
        filename = f"{'_'.join(tickers)}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{timeframe.value}"

    if refresh_bars or not (os.path.exists(os.path.join(data_dir, filename + "_close_prices.csv")) or os.path.exists(os.path.join(data_dir, filename + "_close_prices.pkl"))):
        bars = client.get_stock_bars(request_params).df  # get the bars
        # save_dataframe(bars, filename, data_dir)  # save the bars
        # bars = load_dataframe(filename, data_dir)  # load the bars
        # convert all the dates to est. this is a multi-index dataframe, so we need to convert the index
        bars.index = bars.index.map(lambda x: (x[0], x[1].astimezone(eastern)))
        # resample the bars. apply to each ticker
        bars = BarUtils.resample_multi_ticker_bars(bars)
        # save_dataframe(bars, filename + "_resampled", data_dir)
        # get the close prices
        close_prices = bars["close"].unstack(level=0)
        save_dataframe(close_prices, filename + "_close_prices", data_dir)

    return load_dataframe(filename + "_close_prices", data_dir)


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
    def create_bollinger_bands(series: pd.DataFrame, rolling_window: int = 8, std_multiplier: float = 1) -> pd.DataFrame:
        """Put bollinger bands on the normalized spread.

        Args:
            series (pd.DataFrame): The series to put the bollinger bands on.
            rolling_window (int, optional): The rolling window to use. Defaults to 8.
            std_multiplier (float, optional): The standard deviation multiplier to use. Defaults to 1.

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
            series.rolling(window=rolling_window).mean()
        ).dropna()
        rolling_std = (series.rolling(
            window=rolling_window).std()).dropna()
        upper_band = rolling_mean + (rolling_std * std_multiplier)
        lower_band = rolling_mean - (rolling_std * std_multiplier)

        # fix the bands and rollings to be the same length as the spread
        rolling_mean = rolling_mean.reindex(series.index)
        upper_band = upper_band.reindex(series.index)
        lower_band = lower_band.reindex(series.index)

        output = pd.DataFrame(index=series.index, columns=[
                              "series", "rolling_mean", "upper_band", "lower_band"])
        output["series"] = series
        output["rolling_mean"] = rolling_mean
        output["upper_band"] = upper_band
        output["lower_band"] = lower_band
        return output

    @staticmethod
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
            resampled_ticker = BarUtils.resample_bars(ticker_data, time_frame)
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

    @staticmethod
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
