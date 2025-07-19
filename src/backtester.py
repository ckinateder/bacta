from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pandas as pd

# path wrangling
try:
    from src import *
except ImportError:
    from __init__ import *

from src.utilities import dash, load_dataframe
from src.utilities.market import is_market_open


class Position(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class EventBacktester(ABC):
    """
    Event-based backtester superclass.
    There are two central dataframes:
    - state_history: the state of the backtester at each time step.
    - order_history: the history of orders placed.

    Functions to be overridden:
    - preload: preload the indicators for the backtest. Optional.
    - update_step: update the indicators for the backtest at each time step. Optional.
    - take_action: make a decision based on the indicators and the prices.

    If the user wants to load the train prices, they must call the load_train_prices method before running the backtest.
    This calls the preload method with the train prices and sets the train_prices attribute.

    These structures are updated based on the events that occur, and are used to calculate the performance of the backtester.
    The user may want to override the run method to handle the events.

    To use this class, the user must override the take_action method. If necessary, the user can override the preload and update_step methods.
    The user can call the run method to run the backtest.
    """

    def __init__(self, active_tickers: list[str], cash: float = 100):
        """
        Initialize the backtester.

        Args:
            active_tickers (list[str]): The tickers to trade.
            cash (float, optional): The initial cash balance. Defaults to 100.
        """
        self.active_tickers = active_tickers
        self.initialize_bank(cash)

        # may or may not be needed
        self.train_prices = None
        self.test_prices = None
        self.full_price_history = None

    def initialize_bank(self, cash: float = 100):
        """Initialize the bank and the state history.

        Args:
            cash (float, optional): The initial cash balance. Defaults to 100.
        """
        self.state_history = pd.DataFrame(
            columns=["cash", *self.active_tickers])
        self.state_history.loc[0] = [cash, *[0] * len(self.active_tickers)]

        # history of orders
        self.order_history = pd.DataFrame(
            columns=["ticker", "position", "price", "quantity"])

    def _update_state(self, ticker: str, price: float, quantity: float, order: Position, index: pd.Timestamp, ffill: bool = True):
        """Update the state of the backtester.
        Initial state is 0.

        Args:
            ticker (str): The ticker of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
            order (Position): The order to place.
            index (pd.Timestamp): The index of the state.
            ffill (bool, optional): Whether to ffill the state for uninitialized values. Defaults to True.
        """
        # get current cash
        current_cash = self.state_history.iloc[-1]["cash"]
        current_ticker_quantity = self.state_history.iloc[-1][ticker]
        if current_ticker_quantity == np.nan:
            current_ticker_quantity = 0

        # add index if not present
        if index not in self.state_history.index:
            self.state_history = pd.concat(
                [self.state_history, pd.DataFrame(index=[index])])

        # update state
        if order == Position.LONG:
            self.state_history.loc[index,
                                   "cash"] = current_cash - price * quantity
            self.state_history.loc[index,
                                   ticker] = current_ticker_quantity + quantity
        elif order == Position.SHORT:
            self.state_history.loc[index,
                                   "cash"] = current_cash + price * quantity
            self.state_history.loc[index,
                                   ticker] = current_ticker_quantity - quantity

        # ffill to get rid of Nans
        if ffill:
            self.state_history = self.state_history.ffill()

    def _update_history(self, ticker: str, price: float, quantity: float, order: Position, index: pd.Timestamp):
        """Update the history of the backtester.

        Args:
            ticker (str): The ticker of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
            order (Position): The order to place.
            index (pd.Timestamp): The index of the state.
        """
        new_history = pd.DataFrame([{
            "ticker": ticker,
            "position": order.value,
            "price": price,
            "quantity": quantity,
        }], index=[index])

        # ignore warnings about concat
        if self.order_history.empty:
            self.order_history = new_history
        else:
            self.order_history = pd.concat([self.order_history, new_history])

    def place_buy_order(self, ticker: str, price: float, quantity: float, index: pd.Timestamp):
        """
        Place a buy order for a given ticker.
        """
        # update state
        self._update_state(ticker, price, quantity, Position.LONG, index)
        self._update_history(ticker, price, quantity, Position.LONG, index)

    def place_sell_order(self, ticker: str, price: float, quantity: float, index: pd.Timestamp):
        """
        Place a sell order for a given ticker.
        """
        # update state
        self._update_state(ticker, price, quantity, Position.SHORT, index)
        self._update_history(ticker, price, quantity, Position.SHORT, index)

    def place_order(self, order: Position, index: pd.Timestamp, ticker: str, price: float, quantity: float):
        """
        Place an order for a given ticker.
        """
        if order == Position.LONG:
            self.place_buy_order(ticker, price, quantity, index)
        elif order == Position.SHORT:
            self.place_sell_order(ticker, price, quantity, index)

    def close_positions(self, prices: pd.Series):
        """
        Close all positions at the given prices.

        Args:
            prices (pd.Series): The prices to close the positions at. Name is the date. Index is the tickers.
        """
        index = prices.name
        for ticker in self.active_tickers:
            if self.get_state()[ticker] != 0:
                position = self.get_state()[ticker]
                if position > 0:
                    self.place_sell_order(
                        ticker, prices[ticker], abs(position), index)
                else:
                    self.place_buy_order(
                        ticker, prices[ticker], abs(position), index)

    def run(self, test_prices: pd.DataFrame, ignore_market_open: bool = False, close_positions: bool = True):
        """
        Run a single period of the backtest over the given dataframe.
        Assume that prices have their indicators already calculated and are in the prices dataframe.

        Args:|
            test_prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
                Close prices are used.
            ignore_market_open (bool, optional): Whether to ignore the market operating hours. Defaults to False.
            close_positions (bool, optional): Whether to close positions at the end of the backtest. Defaults to True.
        """
        # check if active tickers are in the prices dataframe
        assert all(
            ticker in test_prices.columns for ticker in self.active_tickers), "Ticker not found in prices dataframe"
        assert test_prices.index.is_unique, "Prices dataframe must have a unique index"
        assert test_prices.index.is_monotonic_increasing, "Prices dataframe must have a monotonic increasing index"
        assert isinstance(
            test_prices.index[0], pd.Timestamp), "Prices dataframe index must be a timestamp"

        # if train prices is not None, we want to make the full price history
        if self.train_prices is not None:
            print(
                f"Train prices have been previously loaded. Concatenating with test prices...")
            full_price_history = pd.concat([self.train_prices, test_prices])
            start_loc = len(self.train_prices)
        else:
            full_price_history = test_prices
            start_loc = 0

        # print the backtest range
        print(
            f"Running backtest over {len(full_price_history.index[start_loc:])} bars from {full_price_history.index[start_loc]} to {full_price_history.index[-1]}...")
        # iterate through the index of the prices
        for index in full_price_history.index[start_loc:]:
            # perform update step
            self.update_step(full_price_history, index)
            # check if the market is open
            if ignore_market_open or is_market_open(index):
                # make a decision
                current_bar_prices = full_price_history.loc[index]
                self.take_action(current_bar_prices)

            if close_positions:
                if index == full_price_history.index[-2]:
                    print(f"Closing positions at {index}...")
                    self.close_positions(full_price_history.iloc[-1])
                    break

        print("Backtest complete.")
        # return the state history
        return self.get_state_history()

    # getters

    def get_state(self) -> pd.DataFrame:
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
        history["cash"] = history["cash"].round(2)
        return history

    def load_train_prices(self, train_prices: pd.DataFrame):
        """
        Preload the indicators for the backtest. This is meant to be overridden by the child class.
        The user must explicitly call this method before running the backtest.

        Args:
            train_prices (pd.DataFrame): The prices of the assets over the TRAINING period.
                Columns are the tickers, index is the date. Close prices are used.
        """
        assert all(
            ticker in train_prices.columns for ticker in self.active_tickers), "Ticker not found in prices dataframe"
        assert train_prices.index.is_unique, "Prices dataframe must have a unique index"
        assert train_prices.index.is_monotonic_increasing, "Prices dataframe must have a monotonic increasing index"
        assert isinstance(
            train_prices.index[0], pd.Timestamp), "Prices dataframe index must be a timestamp"
        self.train_prices = train_prices
        self.preload(train_prices)

    def preload(self, train_prices: pd.DataFrame):
        """
        Preload the indicators for the backtest. This is meant to be overridden by the child class.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")

    def update_step(self, prices: pd.DataFrame, index: pd.Timestamp):
        """
        Args:
            prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
                Close prices are used.
            index: The index point of the prices.
        This is optional and can be overridden by the child class.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")

    @abstractmethod
    def take_action(self, prices: pd.Series):
        """
        Make a decision based on the current prices. This is meant to be overridden by the child class.
        This will place an order if needed.

        Args:
            prices: Series with prices of the assets. Index is the date.
        Returns:
            Position: The order to place.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")


class WalkForwardBacktester(EventBacktester):
    """
    Walk forward backtester that runs the backtest for a given number of periods.
    """

    def __init__(self, active_tickers: list[str], cash: float = 100):
        super().__init__(active_tickers, cash)

    def run_walk_forward(self, prices: pd.DataFrame, walk_forward_periods: int = 8, split_ratio: float = 0.8):
        """
        Run the backtest using walk forward optimization.


        Args:
            prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
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


class SMABacktester(EventBacktester):
    """
    Backtester that uses a simple moving average strategy.
    """

    def __init__(self, active_tickers: list[str], cash: float = 100):
        super().__init__(active_tickers, cash)

    def preload(self, prices: pd.DataFrame):
        """
        Preload the indicators for the backtest.
        """
        self.sma_shorts = {ticker: prices[ticker].rolling(
            window=15).mean() for ticker in self.active_tickers}
        self.sma_longs = {ticker: prices[ticker].rolling(
            window=50).mean() for ticker in self.active_tickers}

    def update_step(self, prices: pd.DataFrame, index: pd.Timestamp):
        """
        Update the state of the backtester.
        """
        # update the indicators
        self.sma_shorts = {ticker: prices[ticker].rolling(
            window=15).mean() for ticker in self.active_tickers}
        self.sma_longs = {ticker: prices[ticker].rolling(
            window=50).mean() for ticker in self.active_tickers}

    def take_action(self, prices: pd.Series):
        """
        Make a decision based on the prices.
        """
        for ticker in self.active_tickers:
            if self.sma_shorts[ticker][prices.name] > self.sma_longs[ticker][prices.name]:
                self.place_order(Position.LONG, prices.name,
                                 ticker, prices[ticker], 1)
            else:
                self.place_order(Position.SHORT, prices.name,
                                 ticker, prices[ticker], 1)


if __name__ == "__main__":
    backtester = SMABacktester(["NEE", "EXC"], cash=1000)

    filename = "NEE_EXC_D_PCG_XEL_ED_WEC_DTE_PPL_AEE_CNP_FE_CMS_EIX_ETR_EVRG_LNT_PNW_IDA_AEP_DUK_SRE_ATO_NRG_2023-01-01_2025-07-19_1Hour_close_prices"
    prices = load_dataframe(filename)
    prices = prices[["NEE", "EXC"]].tail(1000)
    midpoint = int(prices.shape[0] * 0.8)
    train_prices = prices.iloc[:midpoint]
    test_prices = prices.iloc[midpoint:]

    backtester.load_train_prices(train_prices)
    backtester.run(test_prices, ignore_market_open=True)

    print(dash("order history"))
    print(backtester.get_history())
    print(dash("state history"))
    print(backtester.get_state_history())

    print(dash("ending state"))
    print(backtester.get_state())
