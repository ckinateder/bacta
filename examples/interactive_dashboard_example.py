from datetime import datetime, timedelta
import logging

from alpaca.data.timeframe import TimeFrame
import pandas as pd
from talib import ATR, EMA

from bars import (
    download_bars,
    separate_bars_by_symbol,
    split_multi_index_bars_train_test,
)

from __init__ import *
from src import *
from src.backtester import EventBacktester, Order, Position
from src.utilities import dash, get_logger, set_log_level

set_log_level(logging.DEBUG)


class KeltnerChannelBacktester(EventBacktester):
    """
    Backtester that uses the Keltner Channel to make decisions.
    """

    def __init__(self, active_symbols, cash, **kwargs):
        super().__init__(active_symbols, cash, **kwargs)
        self.keltner_channel_period = 21

    def precompute_step(self, bars: pd.DataFrame):
        """
        Preload the indicators for the backtest.
        """

        split_bars = separate_bars_by_symbol(bars)
        self.middle_bands = {symbol: EMA(
            split_bars[symbol].loc[:, "close"], timeperiod=self.keltner_channel_period) for symbol in self.active_symbols}
        self.upper_bands = {symbol: self.middle_bands[symbol] + 2 * ATR(split_bars[symbol].loc[:, "high"], split_bars[symbol].loc[:, "low"],
                                                                        split_bars[symbol].loc[:, "close"], timeperiod=self.keltner_channel_period) for symbol in self.active_symbols}
        self.lower_bands = {symbol: self.middle_bands[symbol] - 2 * ATR(split_bars[symbol].loc[:, "high"], split_bars[symbol].loc[:, "low"],
                                                                        split_bars[symbol].loc[:, "close"], timeperiod=self.keltner_channel_period) for symbol in self.active_symbols}

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """
        Update the state of the backtester.
        """
        split_bars = separate_bars_by_symbol(bars)
        self.middle_bands = {symbol: EMA(
            split_bars[symbol].loc[:, "close"], timeperiod=self.keltner_channel_period) for symbol in self.active_symbols}
        self.upper_bands = {symbol: self.middle_bands[symbol] + 2 * ATR(split_bars[symbol].loc[:, "high"], split_bars[symbol].loc[:, "low"],
                                                                        split_bars[symbol].loc[:, "close"], timeperiod=self.keltner_channel_period) for symbol in self.active_symbols}
        self.lower_bands = {symbol: self.middle_bands[symbol] - 2 * ATR(split_bars[symbol].loc[:, "high"], split_bars[symbol].loc[:, "low"],
                                                                        split_bars[symbol].loc[:, "close"], timeperiod=self.keltner_channel_period) for symbol in self.active_symbols}

    def generate_order(self, bar: pd.DataFrame, index: pd.Timestamp) -> Order:
        """
        Make a decision based on the prices.
        """
        close_prices = bar.loc[:, "close"]

        for symbol in self.active_symbols:
            if close_prices[symbol] > self.upper_bands[symbol][index]:
                return Order(symbol, Position.SHORT, close_prices[symbol], 1)
            elif close_prices[symbol] < self.lower_bands[symbol][index]:
                return Order(symbol, Position.LONG, close_prices[symbol], 1)


def run_backtest_and_show_dashboard():
    """
    Run a backtest and then launch the interactive Streamlit dashboard.
    """
    symbols = ["DUK", "NRG"]

    # download the bars
    bars = download_bars(symbols, start_date=datetime(
        2024, 1, 1), end_date=datetime.now() - timedelta(minutes=15), timeframe=TimeFrame.Hour)

    # split the bars into train and test
    train_bars, test_bars = split_multi_index_bars_train_test(
        bars, split_ratio=0.9)

    # create the backtester
    backtester = KeltnerChannelBacktester(
        symbols, cash=2000, allow_short=True, allow_overdraft=False, min_trade_value=1, market_hours_only=True)

    # preload the train bars
    backtester.load_train_bars(train_bars)

    # run the backtest
    backtester.run_backtest(test_bars)

    # Print performance summary
    print(dash("Performance Summary"))
    print(backtester.analyze_performance())

    # Launch the interactive dashboard
    print(dash("Launching Interactive Dashboard"))
    print("The interactive dashboard will open in your browser.")
    print("You can interact with the plots, zoom, pan, and hover for details.")

    # Create a simple Streamlit app
    import streamlit as st

    st.set_page_config(
        page_title="Backtester Dashboard",
        page_icon="📈",
        layout="wide"
    )

    st.title("Interactive Backtester Dashboard")
    st.write(
        "This dashboard shows the results of the Keltner Channel strategy backtest.")

    # Display the interactive dashboard
    backtester.plot_interactive_dashboard(
        "Keltner Channel Strategy Performance")


if __name__ == "__main__":
    run_backtest_and_show_dashboard()
