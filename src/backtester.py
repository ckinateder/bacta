import pandas as pd
import numpy as np
from abc import abstractmethod
from enum import Enum
from util import dash

class Order(Enum):
    BUY = 1
    SELL = -1

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
        self.starting_cash = cash
        
        # history of states
        self.state_history = pd.DataFrame(columns=["cash", *active_tickers])
        self.state_history.loc[0] = [cash, *[0] * len(active_tickers)]

        # history of orders
        self.order_history = pd.DataFrame(columns=["ticker", "position", "price", "quantity"])

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
            self.state_history = pd.concat([self.state_history, pd.DataFrame(index=[index])])

        # update state
        if order == Order.BUY:
            self.state_history.loc[index, "cash"] = current_cash - price * quantity
            self.state_history.loc[index, ticker] = current_ticker_quantity + quantity
        elif order == Order.SELL:
            self.state_history.loc[index, "cash"] = current_cash + price * quantity
            self.state_history.loc[index, ticker] = current_ticker_quantity - quantity

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
        
    def run(self, prices: pd.DataFrame):
        """
        Run the backtest.

        Args:
            prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
                Close prices are used.

        Returns:
            DataFrame with the backtest results.
        """
        pass

    @abstractmethod
    def buy_decision(self, state: pd.DataFrame) -> bool:
        """
        Decide whether to buy a pair of stocks.
        """
        pass

    @abstractmethod
    def sell_decision(self, state: pd.DataFrame) -> bool:
        """
        Decide whether to sell a pair of stocks.
        """
        pass


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