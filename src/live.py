from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live.stock import StockDataStream
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import pytz

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

# Define timezones
UTC = pytz.UTC
EST = pytz.timezone("US/Eastern")


def convert_to_est(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to EST."""
    if utc_dt.tzinfo is None:
        utc_dt = UTC.localize(utc_dt)
    return utc_dt.astimezone(EST)


class LiveStreamer:
    def __init__(
        self,
        tickers: list[str],
        window_hours: int = 1,
        api_key: str = ALPACA_API_KEY,
        secret_key: str = ALPACA_API_SECRET,
    ) -> None:
        self.tickers = tickers
        self.window_hours = window_hours
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_stream = StockDataStream(api_key, secret_key)

        # Initialize dataframes with proper dtypes
        self.bars_df = pd.DataFrame(
            {
                "symbol": pd.Series(dtype="str"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="int64"),
                "trade_count": pd.Series(dtype="int64"),
                "vwap": pd.Series(dtype="float64"),
            }
        )
        self.bars_df.index = pd.DatetimeIndex([], tz="UTC")

        self.quotes_df = pd.DataFrame(
            {
                "symbol": pd.Series(dtype="str"),
                "bid_price": pd.Series(dtype="float64"),
                "bid_size": pd.Series(dtype="int64"),
                "bid_exchange": pd.Series(dtype="str"),
                "ask_price": pd.Series(dtype="float64"),
                "ask_size": pd.Series(dtype="int64"),
                "ask_exchange": pd.Series(dtype="str"),
                "conditions": pd.Series(dtype="str"),
                "tape": pd.Series(dtype="str"),
            }
        )
        self.quotes_df.index = pd.DatetimeIndex([], tz="UTC")

        # subscribe
        self.data_stream.subscribe_quotes(
            self.on_quote, *self.tickers)
        self.data_stream.subscribe_bars(
            self.on_bar, *self.tickers)

    def _maintain_rolling_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maintain a rolling window of data using UTC timezone.

        Args:
            df (pd.DataFrame): The dataframe to maintain the window for

        Returns:
            pd.DataFrame: The dataframe with only data within the window
        """
        utc_now = datetime.now(UTC)
        cutoff_time = utc_now - timedelta(hours=self.window_hours)
        return df[df.index > cutoff_time]

    async def on_quote(self, quote):
        # Create a new row for the quote
        new_row = pd.DataFrame(
            {
                "symbol": [quote.symbol],
                "bid_price": [float(quote.bid_price)],
                "bid_size": [int(quote.bid_size)],
                "bid_exchange": [quote.bid_exchange],
                "ask_price": [float(quote.ask_price)],
                "ask_size": [int(quote.ask_size)],
                "ask_exchange": [quote.ask_exchange],
                "conditions": [",".join(quote.conditions)],
                "tape": [quote.tape],
            },
            index=pd.DatetimeIndex([quote.timestamp], tz="UTC"),
        )

        # Append to quotes dataframe
        self.quotes_df = pd.concat([self.quotes_df, new_row], axis=0)
        self.quotes_df = self._maintain_rolling_window(self.quotes_df)

        # Convert timestamp to EST for display
        est_time = convert_to_est(quote.timestamp)

        # Print formatted quote
        formatted = (
            f"[{est_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {quote.symbol} "
            f"BID: ${quote.bid_price} x {int(quote.bid_size)} ({quote.bid_exchange}) | "
            f"ASK: ${quote.ask_price} x {int(quote.ask_size)} ({quote.ask_exchange}) "
            f"COND: {','.join(quote.conditions)} TAPE: {quote.tape}"
        )
        print(formatted)

    async def on_bar(self, bar):
        # Create a new row for the bar
        new_row = pd.DataFrame(
            {
                "symbol": [bar.symbol],
                "open": [float(bar.open)],
                "high": [float(bar.high)],
                "low": [float(bar.low)],
                "close": [float(bar.close)],
                "volume": [int(bar.volume)],
                "trade_count": [int(bar.trade_count)],
                "vwap": [float(bar.vwap)],
            },
            index=pd.DatetimeIndex([bar.timestamp], tz="UTC"),
        )

        # Append to bars dataframe
        self.bars_df = pd.concat([self.bars_df, new_row], axis=0)
        self.bars_df = self._maintain_rolling_window(self.bars_df)

        # Convert timestamp to EST for display
        est_time = convert_to_est(bar.timestamp)

        # Print formatted bar
        formatted = (
            f"[{est_time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {bar.symbol} "
            f"OHLC: {bar.open}/{bar.high}/{bar.low}/{bar.close} "
            f"VOL: {int(bar.volume)} TRADES: {int(bar.trade_count)} VWAP: {bar.vwap}"
        )
        print(formatted)

    def save_dataframes(self, prefix: str = "live_data"):
        """Save the current state of both dataframes to disk."""
        # Convert UTC to EST before saving
        bars_est = self.bars_df.copy()
        bars_est.index = bars_est.index.map(convert_to_est)
        bars_est.to_csv(f"{prefix}_bars.csv")

        quotes_est = self.quotes_df.copy()
        quotes_est.index = quotes_est.index.map(convert_to_est)
        quotes_est.to_csv(f"{prefix}_quotes.csv")

    def clear_dataframes(self):
        """Clear both dataframes."""
        self.bars_df = pd.DataFrame(
            {
                "symbol": pd.Series(dtype="str"),
                "open": pd.Series(dtype="float64"),
                "high": pd.Series(dtype="float64"),
                "low": pd.Series(dtype="float64"),
                "close": pd.Series(dtype="float64"),
                "volume": pd.Series(dtype="int64"),
                "trade_count": pd.Series(dtype="int64"),
                "vwap": pd.Series(dtype="float64"),
            }
        )
        self.bars_df.index = pd.DatetimeIndex([], tz="UTC")

        self.quotes_df = pd.DataFrame(
            {
                "symbol": pd.Series(dtype="str"),
                "bid_price": pd.Series(dtype="float64"),
                "bid_size": pd.Series(dtype="int64"),
                "bid_exchange": pd.Series(dtype="str"),
                "ask_price": pd.Series(dtype="float64"),
                "ask_size": pd.Series(dtype="int64"),
                "ask_exchange": pd.Series(dtype="str"),
                "conditions": pd.Series(dtype="str"),
                "tape": pd.Series(dtype="str"),
            }
        )
        self.quotes_df.index = pd.DatetimeIndex([], tz="UTC")

    def start_stream(self):
        self.data_stream.run()

    def stop_stream(self):
        self.data_stream.stop()


if __name__ == "__main__":
    tickers = ["CMS", "NEE"]
    # Example with custom 12-hour window
    streamer = LiveStreamer(tickers, window_hours=12)
    print("Created streamer tracking", tickers)

    try:
        streamer.start_stream()
    except KeyboardInterrupt:
        print("\nSaving data before exit...")
        streamer.save_dataframes()
        streamer.stop_stream()
