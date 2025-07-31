from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
import logging
import random
import pdb

from alpaca.data.timeframe import TimeFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from talib import ATR, EMA, RSI
from tqdm import tqdm, trange

from __init__ import *
from utilities import get_logger, dash
from utilities.bars import download_bars, separate_bars_by_symbol, split_multi_index_bars_train_test
from utilities.market import is_market_open
from utilities.plotting import DEFAULT_FIGSIZE, plt_show


# Get logger for this module
logger = get_logger("backtester")


class Position(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


QUANTITY_PRECISION = 4


class Order:
    def __init__(self, symbol: str, position: Position, price: float, quantity: float):
        """
        Order object. This is used to represent an order to be placed.

        Args:
            symbol (str): The symbol of the asset.
            position (Position): The position to take.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
        """
        self.symbol = symbol
        self.position = position
        self.price = price
        self.quantity = quantity

    def get_value(self) -> float:
        """Get the value of the order.
        """
        return round(self.price * self.quantity, QUANTITY_PRECISION)

    def __str__(self) -> str:
        """String representation of the order.
        """
        return f"{self.position.name} {self.quantity} of {self.symbol} at {self.price}"


class EventBacktester(ABC):
    """
    Event-based backtester superclass.
    There are two central dataframes:
    - state_history: the state of the backtester at each time step.
    - order_history: the history of orders placed.

    Functions to be overridden:
    - precompute_step: precompute_step the indicators for the backtest. Optional.
    - update_step: update the indicators for the backtest at each time step. Optional.
    - generate_order: make a decision based on the indicators and the latest bar.

    The backtester is designed to run purely on the test dataset.
    If the user wants to precompute_step the train bars, they must call the load_train_bars method before running the backtest.
    This calls the precompute_step method with the train bars and sets the train_bars attribute. This is useful to ensure a seamless
    transition from train to test data, like in production. Indicators and models relying on lags need historical data at the start
    of the testing period.

    These structures are updated based on the events that occur, and are used to calculate the performance of the backtester.
    The user may want to override the run method to handle the events.

    To use this class, the user must override the generate_order method. If necessary, the user can override the precompute_step and update_step methods.
    The user can call the run method to run the backtest.

    The backtester is designed to be used with a multi-index dataframe of bars. The index is (symbol, timestamp) and the columns are OHLCV.
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

    def __init__(self, active_symbols: list[str], cash: float = 100, allow_short: bool = True, allow_overdraft: bool = False, min_trade_value: float = 1.0, market_hours_only: bool = True):
        """
        Initialize the backtester.

        Args:
            active_symbols (list[str]): The symbols to trade.
            cash (float, optional): The initial cash balance. Defaults to 100.
            allow_short (bool, optional): Whether to allow short positions. If False, will not allow the backtester to shortsell. Defaults to True.
            allow_overdraft (bool, optional): Whether to allow overdraft in the bank. If False, will not allow the backtester to go into negative cash. Defaults to False.
            min_trade_value (float, optional): The minimum dollar value of a trade. If the order value is less than this, the order will be skipped. Defaults to 1.0.
            market_hours_only (bool, optional): Whether to only place orders during market hours. Defaults to True. 
        """
        logger.debug(
            f"Initializing backtester with active symbols: {active_symbols}, cash: {cash}, allow_short: {allow_short}, allow_overdraft: {allow_overdraft}, min_trade_value: {min_trade_value}, market_hours_only: {market_hours_only}")
        self.active_symbols = active_symbols
        self.initialize_bank(cash)
        self.allow_short = allow_short
        self.allow_overdraft = allow_overdraft
        self.min_trade_value = min_trade_value
        self.market_hours_only = market_hours_only

        # may or may not be needed
        self.train_bars = None
        self.test_bars = None
        # self.full_bars = None

    def initialize_bank(self, cash: float = 100):
        """Initialize the bank and the state history.

        Args:
            cash (float, optional): The initial cash balance. Defaults to 100.
        """
        self.state_history = pd.DataFrame(
            columns=["cash", "portfolio_value", *self.active_symbols])
        self.state_history.loc[0] = [
            cash, cash, *[0] * len(self.active_symbols)]

        # history of orders
        self.order_history = pd.DataFrame(
            columns=["symbol", "position", "price", "quantity"])

    def _update_state(self, symbol: str, price: float, quantity: float, order: Position, index: pd.Timestamp, ffill: bool = True):
        """Update the state of the backtester.
        Initial state is 0.

        Args:
            symbol (str): The symbol of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
            order (Position): The order to place.
            index (pd.Timestamp): The index of the state.
            ffill (bool, optional): Whether to ffill the state for uninitialized values. Defaults to True.
        """
        # get current cash
        current_cash = self.state_history.iloc[-1]["cash"]
        current_symbol_quantity = self.state_history.iloc[-1][symbol]
        if current_symbol_quantity == np.nan:
            current_symbol_quantity = 0

        # add index if not present
        if index not in self.state_history.index:
            self.state_history = pd.concat(
                [self.state_history, pd.DataFrame(index=[index])])

        # update state
        if order == Position.LONG:
            self.state_history.loc[index,
                                   "cash"] = current_cash - price * quantity
            self.state_history.loc[index,
                                   symbol] = current_symbol_quantity + quantity
        elif order == Position.SHORT:
            self.state_history.loc[index,
                                   "cash"] = current_cash + price * quantity
            self.state_history.loc[index,
                                   symbol] = current_symbol_quantity - quantity

        # update portfolio valuation
        self._update_portfolio_value(pd.Series({symbol: price}), index)

        # ffill to get rid of Nans
        if ffill:
            self.state_history = self.state_history.ffill()

    def _update_portfolio_value(self, prices: pd.Series, index: pd.Timestamp):
        """Update the portfolio value at the given index using current prices.

        Args:
            index (pd.Timestamp): The timestamp for the update
            prices (dict[str, float]): Dictionary mapping symbol to current price
        """
        if index not in self.state_history.index:
            return

        # Get current positions
        current_state = self.state_history.loc[index]
        cash = current_state["cash"]

        # Calculate portfolio value: cash + sum of (position * price) for each symbol
        portfolio_value = cash
        for symbol in self.active_symbols:
            if symbol in prices:
                position = current_state[symbol]
                portfolio_value += position * prices[symbol]

        # Update portfolio value in state history
        self.state_history.loc[index, "portfolio_value"] = portfolio_value

    def _update_history(self, symbol: str, price: float, quantity: float, order: Position, index: pd.Timestamp):
        """Update the history of the backtester.

        Args:
            symbol (str): The symbol of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
            order (Position): The order to place.
            index (pd.Timestamp): The index of the state.
        """
        new_history = pd.DataFrame([{
            "symbol": symbol,
            "position": order.value,
            "price": price,
            "quantity": quantity,
        }], index=[index])

        # ignore warnings about concat
        if self.order_history.empty:
            self.order_history = new_history
        else:
            self.order_history = pd.concat([self.order_history, new_history])

    def _place_buy_order(self, symbol: str, price: float, quantity: float, index: pd.Timestamp):
        """
        Place a buy order for a given symbol.
        """
        # update state
        self._update_state(symbol, price, quantity, Position.LONG, index)
        self._update_history(symbol, price, quantity, Position.LONG, index)

    def _place_sell_order(self, symbol: str, price: float, quantity: float, index: pd.Timestamp):
        """
        Place a sell order for a given symbol.
        """
        # update state
        self._update_state(symbol, price, quantity, Position.SHORT, index)
        self._update_history(symbol, price, quantity, Position.SHORT, index)

    def _place_order(self, order: Order, index: pd.Timestamp):
        """Place an order for a given symbol.

        Args:
            order (Order): The order to place.
            index (pd.Timestamp): The index of the state.
        """
        # check if overdraft is allowed
        adjusted = False
        if not self.allow_overdraft and order.position == Position.LONG and self.get_state()["cash"] < order.get_value():
            order.quantity = self.get_state()["cash"] // order.price
            adjusted = True
        # check if shorting is allowed
        if not self.allow_short and order.position == Position.SHORT and self.get_state()[order.symbol] < order.quantity:
            order.quantity = self.get_state()[order.symbol]
            adjusted = True

        # don't place an order if the value is less than the minimum trade value
        if order.get_value() < self.min_trade_value:
            logger.debug(
                f"Skipping {order.position.name} order for {order.symbol} because trade value is less than ${self.min_trade_value:.2f}.")
            return
        else:
            logger.debug(
                f"Placing {f'adjusted ' if adjusted else ''}{order.position.name} order for {order.symbol} with quantity {order.quantity} at {order.price:.2f} for ${order.get_value():.2f}.")

        if order.position == Position.LONG:
            self._place_buy_order(
                order.symbol, order.price, order.quantity, index)
        elif order.position == Position.SHORT:
            self._place_sell_order(
                order.symbol, order.price, order.quantity, index)

    def _close_positions(self, prices: pd.Series, index: pd.Timestamp):
        """
        Close all positions at the given prices.

        Args:
            prices (pd.Series): The prices to close the positions at. Name is the date. Index is the symbols.
        """
        for symbol in self.active_symbols:
            if self.get_state()[symbol] != 0:
                position = self.get_state()[symbol]
                if position > 0:
                    self._place_sell_order(
                        symbol, prices[symbol], abs(position), index)
                else:
                    self._place_buy_order(
                        symbol, prices[symbol], abs(position), index)

    def run(self, test_bars: pd.DataFrame, close_positions: bool = True):
        """
        Run a single period of the backtest over the given dataframe.
        Assume that prices have their indicators already calculated and are in the prices dataframe.

        Args:
            test_bars: DataFrame with bars of the assets. Multi-index with (symbol, timestamp) index and OHLCV columns.
                See the class docstring for more details.
            close_positions (bool, optional): Whether to close positions at the end of the backtest. Defaults to True.
        """
        # check if the bars are in the correct format
        assert test_bars.index.nlevels == 2, "Bars must have a multi-index with (symbol, timestamp) index"
        assert test_bars.index.get_level_values(0).unique().isin(
            self.active_symbols).all(), "All symbols must be in the bars"
        for symbol in self.active_symbols:
            symbol_bars = test_bars.xs(symbol, level=0)
            assert symbol_bars.index.is_monotonic_increasing, f"Bars for {symbol} must have a monotonic increasing timestamp"
            assert symbol_bars.index.is_unique, f"Bars for {symbol} must have a unique timestamp"
        assert isinstance(test_bars.index.get_level_values(
            1)[0], pd.Timestamp), "Bars must have a timestamp index"

        # store
        self.test_bars = test_bars

        # if train bars is not None, we want to make the full price history
        if self.train_bars is not None:
            logger.info(
                "Train bars have been previously loaded. Concatenating with test bars...")
            full_bars = pd.concat([self.train_bars, test_bars])
            start_loc = len(self.train_bars.index.get_level_values(1).unique())
        else:
            full_bars = test_bars
            start_loc = 0

        # get the timestamps
        timestamps = full_bars.index.get_level_values(1).unique()

        # log the backtest range
        logger.info(
            f"Running backtest over {len(full_bars.index[start_loc:])} bars from {timestamps[start_loc]} to {timestamps[-1]}...")

        # iterate through the index of the bars
        for index in tqdm(timestamps[start_loc:], desc="Backtesting", leave=False, dynamic_ncols=True, total=len(timestamps[start_loc:]), position=0):
            # perform update step
            self.update_step(full_bars, index)
            if not self.market_hours_only or is_market_open(index):
                # make a decision
                current_bar = full_bars.xs(index, level=1)
                order = self.generate_order(current_bar, index)

                # place the order if not None
                if order is not None:
                    self._place_order(order, index)

                # Update portfolio value with current close prices
                current_close_prices = current_bar.loc[:, "close"]
                self._update_portfolio_value(current_close_prices, index)

            if close_positions and index == timestamps[-2]:
                logger.info(f"Closing positions at {index}...")
                self._close_positions(full_bars.xs(
                    index, level=1).loc[:, "close"], index)
                break
        # return the state history
        return self.get_state_history()

    def analyze_performance(self) -> pd.Series:
        """
        Analyze the performance of the backtest.
        """
        # get the state history
        state_history = self.get_state_history()
        start_portfolio_value = state_history.iloc[0]["portfolio_value"]
        end_portfolio_value = state_history.iloc[-1]["portfolio_value"]
        return_on_investment = 1 + \
            ((end_portfolio_value - start_portfolio_value) / start_portfolio_value)

        # calculate max drawdown using portfolio value
        cumulative_return = (
            1 + state_history["portfolio_value"].pct_change()).cumprod()
        max_drawdown = cumulative_return.cummax() - cumulative_return
        max_drawdown_percentage = max_drawdown.min()

        # calculate win rate
        win_rate = self.get_win_rate()

        return pd.Series({
            "trading_period": f"{state_history.index[1]} to {state_history.index[-1]}",
            "return_on_investment": return_on_investment,
            "max_drawdown_percentage": max_drawdown_percentage,
            "start_portfolio_value": start_portfolio_value,
            "end_portfolio_value": end_portfolio_value,
            "win_rate": win_rate
        })

    def get_win_rate(self, percentage_threshold: float = 0.0, return_net_profits: bool = False) -> tuple[float, pd.DataFrame]:
        """Get the win rate of the backtest. This is done by calculating the net profit for each open position.

        Example:

        ```
        self.order_history = pd.DataFrame([
            {"symbol": "AAPL", "position": Position.LONG.value, "price": 24.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.SHORT.value, "price": 22.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.LONG.value, "price": 25.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.SHORT.value, "price": 24.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.SHORT.value, "price": 22.0, "quantity": 3},
        ])

        self.get_win_rate() -> (0.6666666666666666,
          symbol  entry_price  exit_price  quantity  net_profit_dollars  net_profit_percentage    win
        0   AAPL         20.0        24.0       1.0                 4.0               0.200000   True
        1   AAPL         21.0        22.0       2.0                 2.0               0.047619   True
        2   AAPL         25.0        22.0       1.0                -3.0              -0.120000   False
        ```

        Args:
            percentage_threshold (float, optional): The threshold for the net profit percentage. Defaults to 0.0. 
                If the net profit percentage is greater than this threshold, the position is considered a win.
                If the net profit percentage is less than this threshold, the position is considered a loss.
            return_net_profits (bool, optional): Whether to return the net profits. Defaults to False.

        Returns:
            tuple[float, pd.DataFrame]: win rate and net profits if return_net_profits is True, otherwise just the win rate
        """
        net_profits = pd.DataFrame(
            columns=["symbol", "entry_price",
                     "exit_price", "quantity", "net_profit_dollars", "net_profit_percentage", "win"],
            index=[])
        net_profits = net_profits.astype({
            "symbol": "string",
            "entry_price": "float64",
            "exit_price": "float64",
            "quantity": "float64",
            "net_profit_dollars": "float64",
            "net_profit_percentage": "float64",
            "win": "boolean"
        })

        order_history = self.order_history.copy()
        net_pointer = 0

        for symbol in self.active_symbols:
            # pointers
            entry_pointer = 0
            exit_pointer = 0

            # get all long and short positions
            all_long_positions = order_history[(order_history["symbol"] == symbol) & (
                order_history["position"] == Position.LONG.value)]
            all_short_positions = order_history[(order_history["symbol"] == symbol) & (
                order_history["position"] == Position.SHORT.value)]

            while exit_pointer < len(all_short_positions) and entry_pointer < len(all_long_positions):
                # get index and row of pointers
                entry_index = all_long_positions.index[entry_pointer]
                entry_row = all_long_positions.iloc[entry_pointer]
                exit_index = all_short_positions.index[exit_pointer]
                exit_row = all_short_positions.iloc[exit_pointer]

                net_profits.loc[net_pointer, "symbol"] = entry_row["symbol"]
                net_profits.loc[net_pointer,
                                "entry_price"] = entry_row["price"]
                net_profits.loc[net_pointer, "exit_price"] = exit_row["price"]

                if exit_row["quantity"] > entry_row["quantity"]:
                    # exit row has more quantity than entry row
                    # add the net profit to the net_profits dataframe
                    # zero out the entry row
                    # DON'T increment the exit pointer
                    # increment the net pointer
                    # increment the entry pointer
                    swing_quantity = entry_row["quantity"]
                    net_profits.loc[net_pointer, "net_profit_dollars"] = (
                        exit_row["price"] - entry_row["price"]) * swing_quantity
                    net_profits.loc[net_pointer, "net_profit_percentage"] = (
                        exit_row["price"] - entry_row["price"]) / entry_row["price"]
                    net_profits.loc[net_pointer, "quantity"] = swing_quantity

                    net_profits.loc[net_pointer, "win"] = net_profits.loc[net_pointer,
                                                                          "net_profit_percentage"] > percentage_threshold

                    all_short_positions.loc[exit_index,
                                            "quantity"] -= swing_quantity
                    all_long_positions.loc[entry_index,
                                           "quantity"] -= swing_quantity
                    entry_pointer += 1
                elif exit_row["quantity"] < entry_row["quantity"]:
                    # entry row has more quantity than exit row
                    # add the net profit to the net_profits dataframe
                    # zero out the exit row
                    # DON'T increment the entry pointer
                    # increment the net pointer
                    # increment the exit pointer
                    swing_quantity = exit_row["quantity"]
                    net_profits.loc[net_pointer, "net_profit_dollars"] = (
                        exit_row["price"] - entry_row["price"]) * swing_quantity
                    net_profits.loc[net_pointer, "net_profit_percentage"] = (
                        exit_row["price"] - entry_row["price"]) / entry_row["price"]
                    net_profits.loc[net_pointer, "quantity"] = swing_quantity
                    net_profits.loc[net_pointer, "win"] = net_profits.loc[net_pointer,
                                                                          "net_profit_percentage"] > percentage_threshold

                    all_long_positions.loc[entry_index,
                                           "quantity"] -= swing_quantity
                    all_short_positions.loc[exit_index,
                                            "quantity"] -= swing_quantity
                    exit_pointer += 1
                else:
                    # exit row has the same quantity as entry row
                    # add the net profit to the net_profits dataframe
                    # zero out the entry and exit rows
                    # increment the net pointer
                    # increment the entry and exit pointers
                    swing_quantity = exit_row["quantity"]
                    net_profits.loc[net_pointer, "net_profit_dollars"] = (
                        exit_row["price"] - entry_row["price"]) * swing_quantity
                    net_profits.loc[net_pointer, "net_profit_percentage"] = (
                        exit_row["price"] - entry_row["price"]) / entry_row["price"]
                    net_profits.loc[net_pointer, "quantity"] = swing_quantity
                    net_profits.loc[net_pointer, "win"] = net_profits.loc[net_pointer,
                                                                          "net_profit_percentage"] > percentage_threshold

                    all_long_positions.loc[entry_index,
                                           "quantity"] = 0
                    all_short_positions.loc[exit_index,
                                            "quantity"] = 0
                    entry_pointer += 1
                    exit_pointer += 1

                net_pointer += 1
            net_pointer += 1

        # reset index
        net_profits = net_profits.reset_index(drop=True)

        # calculate win rate as the number of positive net profits divided by the total number of net profits
        win_rate = len([profit for profit in net_profits["win"]
                       if profit]) / len(net_profits)

        # convert d

        if return_net_profits:
            return win_rate, net_profits
        else:
            return win_rate

    # getters

    def get_state(self) -> pd.Series:
        """
        Get the current state of the checkbook.
        """
        return self.state_history.iloc[-1]

    def get_history(self) -> pd.DataFrame:
        """
        Get the history of the checkbook.
        """
        return self.order_history

    def get_state_history(self) -> pd.DataFrame:
        """
        Build the state history from the current state and the trade history.
        Cols will be the same as the state, index will be the same as the order history.
        """
        history = self.state_history.copy()
        # history["cash"] = history["cash"].round(2)
        return history

    def load_train_bars(self, train_bars: pd.DataFrame):
        """
        Preload the indicators for the backtest. This is meant to be overridden by the child class.
        The user must explicitly call this method before running the backtest.

        Args:
            train_bars (pd.DataFrame): The bars of the assets over the TRAINING period.
                Multi-index with (symbol, timestamp) index and OHLCV columns.
        """
        assert all(
            symbol in train_bars.index.get_level_values(0) for symbol in self.active_symbols), "symbol not found in bars"
        for symbol in self.active_symbols:
            symbol_bars = train_bars.xs(symbol, level=0)
            assert symbol_bars.index.is_monotonic_increasing, f"Bars for {symbol} must have a monotonic increasing timestamp"
            assert symbol_bars.index.is_unique, f"Bars for {symbol} must have a unique timestamp"
        assert isinstance(
            train_bars.index.get_level_values(1)[0], pd.Timestamp), "Bars must have a timestamp index"
        self.train_bars = train_bars
        self.precompute_step(train_bars)

    def plot_equity_curve(self, figsize: tuple = DEFAULT_FIGSIZE, title: str = "Equity Curve", save_plot: bool = True, show_plot: bool = False) -> plt.Figure:
        """
        Plot the equity curve of the backtester.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
            title (str): Title for the plot
        """
        state_history = self.get_state_history()

        if state_history.empty:
            logger.warning(
                "No state history available for plotting equity curve")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot portfolio value over time
        portfolio_values = state_history["portfolio_value"]
        # Filter out the initial state (index 0) and only plot timestamp-indexed data
        portfolio_values_filtered = portfolio_values[portfolio_values.index != 0]
        ax.step(portfolio_values_filtered.index, portfolio_values_filtered,
                label="Portfolio Value", linewidth=2, color='blue', where='post')

        # Add horizontal line for initial portfolio value
        initial_value = portfolio_values_filtered.iloc[0]
        ax.axhline(y=initial_value, color='red', linestyle='--', alpha=0.7,
                   label=f'Initial Value: ${initial_value:.2f}')

        # Calculate and display key metrics
        final_value = portfolio_values_filtered.iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100

        # Calculate drawdown
        cumulative_max = portfolio_values_filtered.cummax()
        drawdown = ((portfolio_values_filtered -
                    cumulative_max) / cumulative_max) * 100
        max_drawdown = drawdown.min()

        # Add text box with performance metrics
        textstr = (
            f'Total Return: {total_return:.2f}%\n'
            f'Max Drawdown: {max_drawdown:.2f}%\n'
            f'Final Value: ${final_value:.2f}'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Format the plot
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_xlabel("Date")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        if save_plot:
            plt_show(prefix=title.replace(" ", "_"))

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_performance_analysis(self, figsize: tuple = (20, 10), save_plot: bool = True, show_plot: bool = False, title: str = "Performance Analysis") -> plt.Figure:
        """
        Create a comprehensive performance analysis plot with multiple subplots.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
            title (str): Title for the plot
        """
        state_history = self.get_state_history()

        if state_history.empty:
            logger.warning(
                "No state history available for plotting performance analysis")
            return

        # Create figure with subplots (2 rows, 3 columns)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
              ) = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(title)

        portfolio_values = state_history["portfolio_value"]
        # Filter out the initial state (index 0) and only plot timestamp-indexed data
        portfolio_values_filtered = portfolio_values[portfolio_values.index != 0]

        # Subplot 1: Cumulative Returns
        returns = portfolio_values_filtered.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        ax1.step(cumulative_returns.index, cumulative_returns,
                 label="Cumulative Returns", linewidth=2, color='green', where='post')
        ax1.axhline(y=1, color='red', linestyle='--',
                    alpha=0.7, label='Break-even')
        ax1.set_title("Cumulative Returns")
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Subplot 2: Drawdown
        cumulative_max = portfolio_values_filtered.cummax()
        drawdown = ((portfolio_values_filtered -
                    cumulative_max) / cumulative_max) * 100
        # For step plots, we need to use fill_between differently
        ax2.fill_between(drawdown.index, drawdown, 0,
                         color='red', alpha=0.3, label='Drawdown', step='post')
        ax2.step(drawdown.index, drawdown, color='red',
                 linewidth=1, where='post')
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Subplot 3: Returns Distribution
        ax3.hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(returns.mean(), color='red', linestyle='--',
                    label=f'Mean: {returns.mean():.4f}')
        ax3.set_title("Returns Distribution")
        ax3.set_xlabel("Return")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Subplot 4: Equity Curve
        ax4.step(portfolio_values_filtered.index, portfolio_values_filtered,
                 label="Portfolio Value", linewidth=2, color='blue', where='post')
        initial_value = portfolio_values_filtered.iloc[0]
        ax4.axhline(y=initial_value, color='red', linestyle='--', alpha=0.7,
                    label=f'Initial Value: ${initial_value:.2f}')
        ax4.set_title("Equity Curve")
        ax4.set_ylabel("Portfolio Value ($)")
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7)

        # Subplot 5: symbol Prices
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            for symbol in self.active_symbols:
                if symbol in self.test_bars.index.get_level_values(0):
                    ax5.step(self.test_bars.xs(symbol, level=0).index, self.test_bars.xs(symbol, level=0).loc[:, "close"],
                             label=symbol, linewidth=1.5, where='post')
            ax5.set_title("Symbol Prices")
            ax5.set_ylabel("Price ($)")
            ax5.legend()
            ax5.grid(True, linestyle='--', alpha=0.7)
        else:
            ax5.text(0.5, 0.5, 'No test prices available',
                     transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title("symbol Prices")

        # Subplot 6: Buy and Hold Returns
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            all_cum_returns = []
            for symbol in self.active_symbols:
                if symbol in self.test_bars.index.get_level_values(0):
                    symbol_bars = self.test_bars.xs(
                        symbol, level=0)
                    symbol_returns = symbol_bars.loc[:,
                                                     "close"].pct_change().dropna()
                    symbol_cum_returns = (1 + symbol_returns).cumprod()
                    ax6.step(symbol_cum_returns.index, symbol_cum_returns,
                             label=f'{symbol} B&H', linewidth=1.5, alpha=0.7, where='post')
                    all_cum_returns.append(symbol_cum_returns)

                    # Calculate combined returns (equal-weighted portfolio)
            if all_cum_returns:
                combined_returns = pd.concat(
                    all_cum_returns, axis=1).mean(axis=1)
                ax6.step(combined_returns.index, combined_returns,
                         label='Combined B&H', linewidth=2, color='black', linestyle='-', where='post')

            ax6.axhline(y=1, color='red', linestyle='--',
                        alpha=0.7, label='Break-even')
            ax6.set_title("Buy and Hold Returns")
            ax6.set_ylabel("Cumulative Return")
            ax6.legend()
            ax6.grid(True, linestyle='--', alpha=0.7)
        else:
            ax6.text(0.5, 0.5, 'No test prices available',
                     transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title("Buy and Hold Returns")

        # Rotate x-axis labels for all subplots
        for ax in [ax1, ax2, ax4, ax5, ax6]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_plot:
            plt_show(prefix=title.replace(" ", "_"))

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_trade_history(self, figsize: tuple = (20, 12), save_plot: bool = True, show_plot: bool = False, title: str = "Trade History", summary_stats: bool = False) -> plt.Figure:
        """
        Plot the price history with trade markers showing buy and sell orders.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
            title (str): Title for the plot
        """
        order_history = self.get_history()

        if order_history.empty:
            logger.warning(
                "No order history available for plotting trade history")
            return None

        if not hasattr(self, 'test_bars') or self.test_bars is None:
            logger.warning("No test bars available for plotting trade history")
            return None

        # Create figure with subplots - one for each symbol
        num_symbols = len(self.active_symbols)
        fig, axes = plt.subplots(num_symbols, 1, figsize=figsize, sharex=True)
        fig.suptitle(title)

        # Handle single symbol case
        if num_symbols == 1:
            axes = [axes]

        # Only plot the test bars, not the train bars. no trades are made on the train bars.
        full_bars = self.test_bars

        for i, symbol in enumerate(self.active_symbols):
            ax = axes[i]

            # Get price data for this symbol
            if symbol in full_bars.index.get_level_values(0):
                symbol_bars = full_bars.xs(symbol, level=0)

                # Plot price history
                ax.plot(symbol_bars.index, symbol_bars['close'],
                        label=f'{symbol} Close Price', linewidth=1.5, color='blue', alpha=0.7)

                # Get trades for this symbol
                symbol_trades = order_history[order_history['symbol'] == symbol]

                if not symbol_trades.empty:
                    # Separate buy and sell orders
                    buy_orders = symbol_trades[symbol_trades['position']
                                               == Position.LONG.value]
                    sell_orders = symbol_trades[symbol_trades['position']
                                                == Position.SHORT.value]

                    # Plot buy orders (green triangles pointing up)
                    if not buy_orders.empty:
                        ax.scatter(buy_orders.index, buy_orders['price'],
                                   marker='^', s=100, color='green', alpha=0.8,
                                   label=f'Buy Orders ({len(buy_orders)})', zorder=5)

                        # Add quantity annotations for buy orders
                        for idx, row in buy_orders.iterrows():
                            ax.annotate(f"{row['quantity']:.0f}",
                                        (idx, row['price']),
                                        xytext=(5, 10), textcoords='offset points',
                                        fontsize=8, color='green', weight='bold')

                    # Plot sell orders (red triangles pointing down)
                    if not sell_orders.empty:
                        ax.scatter(sell_orders.index, sell_orders['price'],
                                   marker='v', s=100, color='red', alpha=0.8,
                                   label=f'Sell Orders ({len(sell_orders)})', zorder=5)

                        # Add quantity annotations for sell orders
                        for idx, row in sell_orders.iterrows():
                            ax.annotate(f"{row['quantity']:.0f}",
                                        (idx, row['price']),
                                        xytext=(5, -15), textcoords='offset points',
                                        fontsize=8, color='red', weight='bold')

                # Format the subplot
                ax.set_title(
                    f'{symbol} Price History with Trade Markers', fontsize=12, fontweight='bold')
                ax.set_ylabel('Price ($)', fontsize=10)
                ax.legend(loc='upper left')
                ax.grid(True, linestyle='--', alpha=0.3)

                # Add summary statistics
                if not symbol_trades.empty and summary_stats:
                    total_trades = len(symbol_trades)
                    total_volume = symbol_trades['quantity'].sum()
                    avg_price = (
                        symbol_trades['price'] * symbol_trades['quantity']).sum() / symbol_trades['quantity'].sum()

                    stats_text = f'Trades: {total_trades} | Volume: {total_volume:.0f} | Avg Price: ${avg_price:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'No data available for {symbol}',
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{symbol} - No Data Available')

        # Set x-axis label for the bottom subplot only
        axes[-1].set_xlabel('Date', fontsize=10)

        # Rotate x-axis labels for better readability
        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_plot:
            plt_show(prefix=title.replace(" ", "_"))

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    ################################################################################################
    ################################################################################################

    def precompute_step(self, train_bars: pd.DataFrame):
        """
        Preload the indicators for the backtest. This is meant to be overridden by the child class.

        Args:
            train_bars (pd.DataFrame): The bars of the assets over the TRAINING period.
                Multi-index with (symbol, timestamp) index and OHLCV columns.

        Raises:
            NotImplementedError: This method must be overridden by the child class.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """
        Args:
            bars: DataFrame with bars of the assets. Multi-index with (symbol, timestamp) index and OHLCV columns.
                Bars is the full price history. Might remove this in the future and go with the class state.
            index: The index point of the current step.
        This is optional and can be overridden by the child class.
        """
        pass

    @abstractmethod
    def generate_order(self, bars: pd.DataFrame, index: pd.Timestamp) -> Order:
        """
        Make a decision based on the current prices. This is meant to be overridden by the child class.
        This will place an order if needed.

        Args:
            bars: DataFrame with bars of the assets. Multi-index with (symbol, timestamp) index and OHLCV columns.
            index: The index point of the bars.
        Returns:
            Position: The order to place.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")


class WalkForwardBacktester(EventBacktester, ABC):
    """
    Walk forward backtester that runs the backtest for a given number of periods.
    """

    def __init__(self, active_symbols: list[str], cash: float = 100):
        super().__init__(active_symbols, cash)

    def run_walk_forward(self, prices: pd.DataFrame, walk_forward_periods: int = 8, split_ratio: float = 0.8):
        """
        Run the backtest using walk forward optimization.


        Args:
            prices: DataFrame with prices of the assets. Columns are the symbols, index is the date.
                Close prices are used.
            walk_forward_periods: Number of periods to walk forward.
        Returns:
            DataFrame with the backtest results.
        """
        # split the prices into walk forward periods
        # all periods are the same length. last period will be shorter if the length is not a multiple of the walk forward periods
        period_length = len(prices) // walk_forward_periods
        remainder = len(prices) % walk_forward_periods

        # we want the strategy to train on
        raise NotImplementedError()


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
                return Order(symbol, Position.SHORT, close_prices[symbol], random.randint(1, 3))
            elif close_prices[symbol] < self.lower_bands[symbol][index]:
                return Order(symbol, Position.LONG, close_prices[symbol], random.randint(1, 3))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    symbols = ["DUK"]  # , "NRG"]
    utility_symbols = [
        "NEE", "EXC", "D", "PCG", "XEL",
        "ED", "WEC", "DTE", "PPL", "AEE",
        "CNP", "FE", "CMS", "EIX", "ETR",
        "EVRG", "LNT", "PNW", "IDA", "AEP",
        "DUK", "SRE", "ATO", "NRG",
    ]
    bars = download_bars(symbols, start_date=datetime(
        2024, 1, 1), end_date=datetime.now() - timedelta(minutes=15), timeframe=TimeFrame.Hour)

    train_bars, test_bars = split_multi_index_bars_train_test(
        bars, split_ratio=0.9)
    backtester = KeltnerChannelBacktester(
        symbols, cash=2000, allow_short=False, allow_overdraft=False, min_trade_value=1, market_hours_only=True)
    backtester.load_train_bars(train_bars)
    backtester.run(test_bars)

    print(dash("order history"))
    print(backtester.get_history())
    print(dash("state history"))
    print(backtester.get_state_history())

    print(dash("performance"))
    print(backtester.analyze_performance())

    # Plot the equity curve
    print(dash("plotting..."))
    backtester.plot_equity_curve(
        title="KC Strategy Equity Curve "+"_".join(symbols))
    backtester.plot_performance_analysis(
        title="KC Strategy Performance Analysis "+"_".join(symbols))
    backtester.plot_trade_history(
        title="KC Strategy Trade History "+"_".join(symbols))
