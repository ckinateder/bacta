import os
from datetime import datetime, timedelta
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient


from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# path wrangling
from bacta.utilities import load_dataframe, save_dataframe, getenv, get_logger, eastern

# Create a logger for the bars module
logger = get_logger("examples.bars")


def timeframe_to_timedelta(timeframe: TimeFrame) -> timedelta:
    """Convert a TimeFrame to a timedelta.
    """
    # value will be in form "15Min", "1Hour", "1Day", "1Week", "1Month", "15Min", "1Hour", "1Day", "1Week", "1Month"
    # we need to convert this to a timedelta
    if timeframe.unit == TimeFrameUnit.Minute:
        return timedelta(minutes=timeframe.amount_value)
    elif timeframe.unit == TimeFrameUnit.Hour:
        return timedelta(hours=timeframe.amount_value)
    elif timeframe.unit == TimeFrameUnit.Day:
        return timedelta(days=timeframe.amount_value)
    elif timeframe.unit == TimeFrameUnit.Week:
        return timedelta(weeks=timeframe.amount_value)
    elif timeframe.unit == TimeFrameUnit.Month:
        raise ValueError("Monthly timeframe not supported")
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")


def download_bars(symbols: list[str], start_date: datetime, end_date: datetime = datetime.now() - timedelta(minutes=15),
                  timeframe: TimeFrame = TimeFrame.Hour, refresh_bars: bool = False,
                  data_dir: str = getenv("DATA_DIR"), resample: bool = True) -> pd.DataFrame:
    """Download the bars for the given symbols. Will first check if the bars are already downloaded and if not, will download them.
    This requires an Alpaca API key.
    """
    if all(symbol.endswith("/USD") for symbol in symbols):
        return download_crypto_bars(symbols, start_date, end_date,
                                    timeframe, refresh_bars, data_dir, resample)
    else:
        return download_stock_bars(symbols, start_date, end_date,
                                   timeframe, refresh_bars, data_dir, resample)


def download_stock_bars(symbols: list[str], start_date: datetime, end_date: datetime,
                        timeframe: TimeFrame, refresh_bars: bool = False,
                        data_dir: str = getenv("DATA_DIR"), resample: bool = True) -> pd.DataFrame:
    """Download the bars for the given symbols. Will first check if the bars are already downloaded and if not, will download them.
    This requires an Alpaca API key.

    Args:
        symbols (list[str]): The symbols to download bars for.
        start_date (datetime): The start date to download bars for.
        end_date (datetime): The end date to download bars for.
        timeframe (TimeFrame): The timeframe to download bars for.
        refresh_bars (bool, optional): Whether to force refresh the bars. If False, will check if the bars are already downloaded and if not, will download them.
        data_dir (str, optional): The directory to save the bars.
        resample (bool, optional): Whether to resample the bars. Defaults to True.

    Returns:
        pd.DataFrame: The bars for the given symbols, resampled to the given timeframe. 
        This is a multi-index dataframe with the symbols as the first level and the timestamp as the second level.
        Example:
                                            open    high  ...    volume  trade_count
        symbol timestamp                                   ...                       
        CMS    2023-01-03 09:00:00-05:00   58.560   58.93  ...  171942.0       1941.0
               2023-01-03 10:00:00-05:00   58.330   58.36  ...  169171.0       3700.0
               2023-01-03 11:00:00-05:00   58.340   58.38  ...   98527.0       2302.0
               2023-01-03 12:00:00-05:00   58.230   58.50  ...  151657.0       2355.0
               2023-01-03 13:00:00-05:00   58.200   58.67  ...   92756.0       1905.0
        ...                                   ...     ...  ...       ...          ...
        DTE    2025-07-18 12:00:00-04:00  136.535  136.99  ...   68742.0       2193.0
               2025-07-18 13:00:00-04:00  136.530  136.80  ...   99874.0       2391.0
               2025-07-18 14:00:00-04:00  136.680  136.72  ...  103748.0       3041.0
               2025-07-18 15:00:00-04:00  136.700  137.39  ...  560404.0       9704.0
               2025-07-18 16:00:00-04:00  137.260  137.98  ...  415033.0         55.0
    """
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    logger.debug(
        f"Loading close prices for {len(symbols)} symbols from {start_date} to {end_date} with timeframe {timeframe.value}")
    ALPACA_API_KEY = getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET = getenv("ALPACA_API_SECRET")
    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET
    )
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="all",
    )
    symbol_str = ""
    for symbol in symbols:
        symbol_str += symbol.replace("/", "_")
    filename = f"{symbol_str}_{start_date}_{end_date}_{timeframe.value}"
    if refresh_bars or not (os.path.exists(os.path.join(data_dir, filename + ".csv")) or os.path.exists(os.path.join(data_dir, filename + ".pkl"))):
        logger.debug(
            f"Refreshing... downloading bars for {symbols} from {start_date} to {end_date} with timeframe {timeframe.value}")
        bars = client.get_stock_bars(request_params).df  # get the bars
        # convert all the dates to est. this is a multi-index dataframe, so we need to convert the index
        bars.index = bars.index.map(lambda x: (x[0], x[1].astimezone(eastern)))
        if resample:
            bars = BarUtils.resample_multi_symbol_bars(
                bars, timeframe_to_timedelta(timeframe))
        save_dataframe(bars, filename, data_dir)
        logger.debug(f"Saved bars to {filename}")

    return load_dataframe(filename, data_dir)


def download_crypto_bars(symbols: list[str], start_date: datetime, end_date: datetime,
                         timeframe: TimeFrame, refresh_bars: bool = False,
                         data_dir: str = getenv("DATA_DIR"), resample: bool = True) -> pd.DataFrame:
    """Download the bars for the given symbols. Will first check if the bars are already downloaded and if not, will download them.
    This requires an Alpaca API key.
    """
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    logger.debug(
        f"Loading close prices for {len(symbols)} symbols from {start_date} to {end_date} with timeframe {timeframe.value}")
    ALPACA_API_KEY = getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET = getenv("ALPACA_API_SECRET")
    client = CryptoHistoricalDataClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET
    )
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="all",
    )
    symbol_str = ""
    for symbol in symbols:
        symbol_str += symbol.replace("/", "_")
    filename = f"{symbol_str}_{start_date}_{end_date}_{timeframe.value}"
    if refresh_bars or not (os.path.exists(os.path.join(data_dir, filename + ".csv")) or os.path.exists(os.path.join(data_dir, filename + ".pkl"))):
        logger.debug(
            f"Refreshing... downloading bars for {symbols} from {start_date} to {end_date} with timeframe {timeframe.value}")
        bars = client.get_crypto_bars(request_params).df
        if resample:
            bars = BarUtils.resample_multi_symbol_bars(
                bars, timeframe_to_timedelta(timeframe))
        save_dataframe(bars, filename, data_dir)
        logger.debug(f"Saved bars to {filename}")

    return load_dataframe(filename, data_dir)


def download_close_prices(symbols: list[str], start_date: datetime, end_date: datetime,
                          timeframe: TimeFrame, refresh_bars: bool = False,
                          data_dir: str = getenv("DATA_DIR"),) -> pd.DataFrame:
    """Download the bars for the given symbols. Will first check if the bars are already downloaded and if not, will download them.
    This requires an Alpaca API key.
    Args:
        symbols (list[str]): The symbols to download bars for.
        start_date (datetime): The start date to download bars for.
        end_date (datetime): The end date to download bars for.
        timeframe (TimeFrame): The timeframe to download bars for.
        refresh_bars (bool): Whether to force refresh the bars. If False, will check if the bars are already downloaded and if not, will download them.
        data_dir (str): The directory to save the bars.
        filename (str): The filename to save the bars to. If None, will be a combination of the symbols, start date, end date, and timeframe.

    Returns:
        pd.DataFrame: The close prices for the given symbols, resampled to the given timeframe.
    """
    filename = f"{'_'.join(symbols)}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{timeframe.value}_close_prices"

    if refresh_bars or not (os.path.exists(os.path.join(data_dir, filename + ".csv")) or os.path.exists(os.path.join(data_dir, filename + ".pkl"))):
        logger.debug(
            f"Refreshing... downloading bars for {symbols} from {start_date} to {end_date} with timeframe {timeframe.value}")
        bars = download_stock_bars(symbols, start_date, end_date,
                                   timeframe, refresh_bars, data_dir)
        close_prices = bars["close"].unstack(level=0)
        save_dataframe(close_prices, filename, data_dir)
        logger.debug(f"Saved close prices to {filename}")

    return load_dataframe(filename, data_dir)


def separate_bars_by_symbol(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separate the bars into individual symbol dataframes.
    Args:
        bars (pd.DataFrame): The bars to split. Multi-index with (symbol, timestamp) index and OHLCV columns.

    Returns:
        dict[str, pd.DataFrame]: A dictionary of symbol dataframes.
    """
    symbols = bars.index.get_level_values(0).unique()
    return {symbol: bars.xs(symbol, level=0) for symbol in symbols}


def split_multi_index_bars_train_test(bars: pd.DataFrame, split_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the bars into train and test sets.

    Args:
        bars (pd.DataFrame): The bars to split. Multi-index with (symbol, timestamp) index and OHLCV columns.
        split_ratio (float, optional): The ratio of the bars to split into train and test. Defaults to 0.8.

    Raises:
        ValueError: If the bars are not a multi-index dataframe.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The train and test bars.
    """
    assert bars.index.nlevels == 2, "Bars must have a multi-index with (symbol, timestamp) index"
    symbols = bars.index.get_level_values(0).unique()
    # split the bars into individual symbol dataframes
    symbol_bars = separate_bars_by_symbol(bars)
    # make sure each symbol has the same number of bars
    num_bars = symbol_bars[symbols[0]].shape[0]
    for symbol in symbols:
        assert symbol_bars[symbol].shape[
            0] == num_bars, f"Bars for {symbol} must have the same number of bars, try resampling and/or changing the timeframe"

    # split the bars into train and test
    train_bars = {symbol: symbol_bars[symbol].iloc[:int(
        len(symbol_bars[symbol]) * split_ratio)] for symbol in symbols}
    test_bars = {symbol: symbol_bars[symbol].iloc[int(
        len(symbol_bars[symbol]) * split_ratio):] for symbol in symbols}
    # recombine the bars into a multi-index dataframe
    train_bars = pd.concat(train_bars, axis=0)
    test_bars = pd.concat(test_bars, axis=0)
    train_bars.index.names = ["symbol", "timestamp"]
    test_bars.index.names = ["symbol", "timestamp"]
    return train_bars, test_bars


class BarUtils:
    def __init__(self):
        pass

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
    def resample_multi_symbol_bars(
        bars: pd.DataFrame, time_frame: timedelta = timedelta(hours=1)
    ) -> pd.DataFrame:
        """Resample the bars to the given time frame and fill the missing values.
        For OHLC, use the last value in the time frame to fill forward.
        For volume and trade count, fill with 0.
        THIS IS FOR MULTIPLE symbolS. Use groupby to apply to multiple symbols.

        Args:
            bars (pd.DataFrame): The bars to resample. Multi-index with (symbol, timestamp) index and OHLCV columns.
            time_frame (timedelta, optional): The time frame to resample to. Defaults to timedelta(hours=1).

        Returns:
            pd.DataFrame: The resampled bars. Multi-index with (symbol, timestamp) index and OHLCV columns.
        """
        # Create a list to store resampled data for each symbol
        resampled_data = []

        # Get unique symbols
        symbols = bars.index.get_level_values("symbol").unique()

        latest_start = None
        earliest_end = None

        # Process each symbol
        for symbol in symbols:
            # Get data for this symbol
            symbol_data = bars.loc[symbol]
            # start and end of the symbol data
            start = symbol_data.index.get_level_values("timestamp").min()
            end = symbol_data.index.get_level_values("timestamp").max()
            if latest_start is None or start > latest_start:
                latest_start = start
            if earliest_end is None or end < earliest_end:
                earliest_end = end

            # Resample the data
            resampled_symbol = BarUtils.resample_bars(symbol_data, time_frame)
            # Add the symbol back to the index
            resampled_symbol["symbol"] = symbol
            resampled_data.append(resampled_symbol)

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
        THIS IS FOR ONE symbol. Use groupby to apply to multiple symbols.

        Args:
            bars (pd.DataFrame): The bars to resample. Single index with timestamp index and OHLCV columns.
            time_frame (timedelta, optional): The time frame to resample to. Defaults to timedelta(hours=1).

        Returns:
            pd.DataFrame: The resampled bars. Single index with timestamp index and OHLCV columns.
        """
        # resample the bars
        part1 = bars[["open", "high", "low", "close", "vwap"]]
        part2 = bars[["volume", "trade_count"]]
        part1 = part1.resample(time_frame).last().ffill()
        part2 = part2.resample(time_frame).asfreq().fillna(0)
        bars = pd.concat([part1, part2], axis=1)
        return bars


if __name__ == "__main__":
    bars = download_crypto_bars(["ETH/USD"], datetime(2025, 7, 1),
                                datetime.now(), TimeFrame.Hour)
    print(bars)

    bars = download_stock_bars(["AAPL"], datetime(2025, 7, 1),
                               datetime.now() - timedelta(minutes=15), TimeFrame.Hour)
    print(bars)
