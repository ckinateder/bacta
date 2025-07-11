import pandas as pd
import numpy as np
from abc import abstractmethod
from enum import Enum
from util import dash
from __init__ import Order
from __init__ import Position
from util import is_market_open


class BasicBacktester:
    """
    Basic backtester that buys and sells pairs of stocks.

    This is a general backtester that can handle any number of assets.

    Args:
        active_tickers: List of tickers to trade.
        cash: Initial cash balance.

    """

    def __init__(self, active_tickers: list[str], cash: float = 100):
        self.active_tickers = active_tickers
        self.initialize_bank(cash)

    def initialize_bank(self, cash: float = 100):
        # history of states
        self.state_history = pd.DataFrame(
            columns=["cash", *self.active_tickers])
        self.state_history.loc[0] = [cash, *[0] * len(self.active_tickers)]

        # history of orders
        self.order_history = pd.DataFrame(
            columns=["ticker", "position", "price", "quantity"])

    def _update_state(self, ticker: str, price: float, quantity: float, order: Order, index: pd.Timestamp, ffill: bool = True):
        """Update the state of the backtester.
        Initial state is 0.

        Args:
            ticker (str): The ticker of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
            order (Order): The order to place.
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
        if order == Order.BUY:
            self.state_history.loc[index,
                                   "cash"] = current_cash - price * quantity
            self.state_history.loc[index,
                                   ticker] = current_ticker_quantity + quantity
        elif order == Order.SELL:
            self.state_history.loc[index,
                                   "cash"] = current_cash + price * quantity
            self.state_history.loc[index,
                                   ticker] = current_ticker_quantity - quantity

        # ffill to get rid of Nans
        if ffill:
            self.state_history = self.state_history.ffill()

    def _update_history(self, ticker: str, price: float, quantity: float, order: Order, index: pd.Timestamp):
        """Update the history of the backtester.

        Args:
            ticker (str): The ticker of the asset.
            price (float): The price of the asset.
            quantity (float): The quantity of the asset.
            order (Order): The order to place. 
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
        self._update_state(ticker, price, quantity, Order.BUY, index)
        self._update_history(ticker, price, quantity, Order.BUY, index)

    def place_sell_order(self, ticker: str, price: float, quantity: float, index: pd.Timestamp):
        """
        Place a sell order for a given ticker.
        """
        # update state
        self._update_state(ticker, price, quantity, Order.SELL, index)
        self._update_history(ticker, price, quantity, Order.SELL, index)

    def place_order(self, order: Order, index: pd.Timestamp, ticker: str, price: float, quantity: float):
        """
        Place an order for a given ticker.
        """
        if order == Order.BUY:
            self.place_buy_order(ticker, price, quantity, index)
        elif order == Order.SELL:
            self.place_sell_order(ticker, price, quantity, index)

    def run_period(self, prices: pd.DataFrame):
        """
        Run a single period of the backtest. Prices are only from the test period. 
        Assume that prices have their indicators already calculated and are in the prices dataframe.

        Args:
            prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
                Close prices are used.
        """
        # check if active tickers are in the prices dataframe
        assert all(
            ticker in prices.columns for ticker in self.active_tickers), "Ticker not found in prices dataframe"

        # bank has already been initialized
        # iterate through the index of the prices
        for index in prices.index:
            # check if the market is open
            if is_market_open(index):
                # make a decision
                current_bar_prices = prices.loc[index]
                order = self.make_decision(current_bar_prices)
                if order is not None:
                    self.place_order(
                        order, index, current_bar_prices.name, current_bar_prices.values[0], 1)

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
        return self.state_history

    @abstractmethod
    def make_decision(self, prices: pd.DataFrame) -> Order:
        """
        Make a decision based on the prices. This is meant to be overridden by the child class.

        Args:
            prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
                Close prices are used.

        Returns:
            Order: The order to place.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")


class WalkForwardBacktester(BasicBacktester):
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


if __name__ == "__main__":
    backtester = BasicBacktester(["AAPL", "MSFT"], cash=1000)
    backtester.place_buy_order("AAPL", 100, 1, pd.Timestamp("2025-01-01"))
    backtester.place_sell_order("AAPL", 100, 1, pd.Timestamp("2025-01-02"))

    print(dash("state"))
    print(backtester.get_state())
    print(dash("history"))
    print(backtester.get_history())
    print(dash("state history"))
    print(backtester.get_state_history())
