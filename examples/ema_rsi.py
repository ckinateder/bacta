# %% [markdown]
# # EMA RSI Strategy
#
# Winning ideas:
# - Hour timeframe
# - mean reverting stocks

# %%
from datetime import datetime, timedelta
import logging
import pytz
import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from talib import ATR, EMA, RSI

from data import (
    download_bars,
    separate_bars_by_symbol,
    split_multi_index_bars_train_test,
)

from bacta.backtester import EventBacktester, Order, Side
from bacta.utilities import dash
from bacta.utilities.logger import get_logger, set_log_level
logger = get_logger()
set_log_level(logging.DEBUG)

# %%


class EmaStrategy(EventBacktester):
    """
    Backtester that uses the Keltner Channel to make decisions.
    """

    def __init__(self, active_symbols, cash, **kwargs):
        super().__init__(active_symbols, cash, **kwargs)
        stretch = 1
        self.short_ema_period = 21 * stretch
        self.long_ema_period = 200 * stretch
        self.rsi_period = 14 * stretch
        self.rsi_high = 65
        self.rsi_low = 35
        self.max_position_value = 700

    def precompute_step(self, bars: pd.DataFrame):
        """
        Preload the indicators for the backtest.
        """

        split_bars = separate_bars_by_symbol(bars)

        self.short_emas = {
            symbol: EMA(
                split_bars[symbol].loc[:,
                                       "close"], timeperiod=self.short_ema_period
            )
            for symbol in self.active_symbols
        }
        self.long_emas = {
            symbol: EMA(
                split_bars[symbol].loc[:,
                                       "close"], timeperiod=self.long_ema_period
            )
            for symbol in self.active_symbols
        }
        self.rsis = {
            symbol: RSI(split_bars[symbol].loc[:, "close"],
                        timeperiod=self.rsi_period)
            for symbol in self.active_symbols
        }

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """
        Update the state of the backtester.
        """
        split_bars = separate_bars_by_symbol(bars)

        self.short_emas = {
            symbol: EMA(
                split_bars[symbol].loc[:,
                                       "close"], timeperiod=self.short_ema_period
            )
            for symbol in self.active_symbols
        }
        self.long_emas = {
            symbol: EMA(
                split_bars[symbol].loc[:,
                                       "close"], timeperiod=self.long_ema_period
            )
            for symbol in self.active_symbols
        }
        self.rsis = {
            symbol: RSI(split_bars[symbol].loc[:, "close"],
                        timeperiod=self.rsi_period)
            for symbol in self.active_symbols
        }

    def generate_orders(self, bar: pd.DataFrame, index: pd.Timestamp) -> list[Order]:
        """
        Make a decision based on the prices.
        """
        close_prices = bar.loc[:, "close"]
        # short ema is 21, long ema is 200
        # rsi is 14
        # if rsi is > 75 and short ema is > long ema, then short
        # if rsi is < 25 and short ema is < long ema, then long
        orders = []

        for symbol in self.active_symbols:
            position_value = abs(self.get_position(
                symbol)*close_prices[symbol]) if self.get_position(symbol) != 0 else 0

            quantity = round(200 / close_prices[symbol], 4)
            if (
                self.rsis[symbol][index] > self.rsi_high
                and self.short_emas[symbol][index] > self.long_emas[symbol][index]
            ) and position_value < self.max_position_value:
                orders.append(
                    Order(symbol, Side.SELL,
                          close_prices[symbol], quantity)
                )
            elif (
                self.rsis[symbol][index] < self.rsi_low
                and self.short_emas[symbol][index] < self.long_emas[symbol][index]
            ) and position_value < self.max_position_value:
                orders.append(
                    Order(symbol, Side.BUY,
                          close_prices[symbol], quantity)
                )

        return orders


# %%
symbols = ["DUK", "LLY", "AEP"]

shift = timedelta(days=0)
bars = download_bars(
    symbols,
    start_date=datetime(2025, 4, 1) - shift,
    end_date=datetime(2025, 7, 1) - shift,
    timeframe=TimeFrame(1, TimeFrameUnit.Hour)
)
# split the bars into train and test
train_bars, test_bars = split_multi_index_bars_train_test(
    bars, split_ratio=0.2)

# create the backtester
backtester = EmaStrategy(
    symbols,
    cash=2000,
    allow_short=True,
    allow_overdraft=False,
    min_cash_balance=200,
    min_trade_value=1,
    market_hours_only=True,
    transaction_cost=0.000,
    transaction_cost_type="percentage",
)

# preload the train bars
backtester.load_train_bars(train_bars)

# run_backtest the backtest
backtester.run_backtest(test_bars)

# plot the order and state history
print(dash("order history"))
print(backtester.get_history())
print(dash("state history"))
print(backtester.get_state_history())

# %%
# plot the performance
print(dash("performance"))
print(backtester.pretty_format_performance())

# %%
# Plot the results
print("plotting...")
backtester.plot_performance_analysis(
    title="_".join(symbols) + " EMA RSI Strategy Performance", show_plot=False)

# %%
backtester.plot_trade_history(title="_".join(
    symbols) + " EMA RSI Strategy Trades", show_plot=False)

# %%
backtester.plot_equity_curve(
    title="_".join(symbols) + " EMA RSI Strategy Equity Curve", show_plot=False
)
