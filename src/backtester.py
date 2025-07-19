import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from __init__ import Position, Position
from utilities import dash, load_dataframe
from utilities.market import is_market_open

# enums


class EventBacktester(ABC):
    """
    Event-based backtester superclass.
    There are two central dataframes:
    - state_history: the state of the backtester at each time step.
    - order_history: the history of orders placed.

    User should override the update_step and make_decision methods.
    update_step is called at each time step and is optional. This is where the user can perform
    any calculations that are needed at each time step, update indicators, etc.
    make_decision is called at each time step and is required. This is where the user should make a decision
    based on the indicators and the prices.

    These structures are updated based on the events that occur, and are used to calculate the performance of the backtester.
    The user may want to override the run method to handle the events.
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

    def run(self, prices: pd.DataFrame, start_index: pd.Timestamp = None, ignore_market_open: bool = False, close_positions: bool = True):
        """
        Run a single period of the backtest over the given dataframe.
        Assume that prices have their indicators already calculated and are in the prices dataframe.

        Args:|
            prices: DataFrame with prices of the assets. Columns are the tickers, index is the date.
                Close prices are used.
            start_index (pd.Timestamp, optional): The index to start the backtest at. Defaults to None.
            ignore_market_open (bool, optional): Whether to ignore the market operating hours. Defaults to False.
            close_positions (bool, optional): Whether to close positions at the end of the backtest. Defaults to True.
        """
        # check if active tickers are in the prices dataframe
        assert all(
            ticker in prices.columns for ticker in self.active_tickers), "Ticker not found in prices dataframe"
        assert prices.index.is_unique, "Prices dataframe must have a unique index"
        assert prices.index.is_monotonic_increasing, "Prices dataframe must have a monotonic increasing index"
        assert isinstance(
            prices.index[0], pd.Timestamp), "Prices dataframe index must be a timestamp"

        # get the start and end locations
        start_loc = 0
        if start_index is not None:
            assert start_index in prices.index, f"Start index {start_index} not found in prices dataframe"
            start_loc = prices.index.get_loc(start_index)

        # if close_positions, we want to close the positions at the second to last index

        # print the backtest range
        print(
            f"Running backtest from {prices.index[start_loc]}...")

        # iterate through the index of the prices
        for index in prices.index[start_loc:]:
            # perform update step
            self.update_step(prices, index)
            # check if the market is open
            if ignore_market_open or is_market_open(index):
                # make a decision
                current_bar_prices = prices.loc[index]
                self.make_decision(current_bar_prices)

            if close_positions:
                if index == prices.index[-2]:
                    print(f"Closing positions at {index}...")
                    self.close_positions(prices.iloc[-1])
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

    @abstractmethod
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
    def make_decision(self, prices: pd.Series) -> Position:
        """
        Make a decision based on the current prices. This is meant to be overridden by the child class.

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

    def update_step(self, prices: pd.DataFrame, index: pd.Timestamp):
        """
        Update the state of the backtester.
        """
        pass

    def make_decision(self, prices: pd.Series) -> Position:
        """
        Make a decision based on the prices.
        """
        self.place_order(Position.LONG, prices.name,
                         prices.index[1], prices.values[1], 1)
        self.place_order(Position.SHORT, prices.name,
                         prices.index[0], prices.values[0], 1)


if __name__ == "__main__":
    backtester = SMABacktester(["NEE", "EXC"], cash=1000)

    filename = "../data/NEE_EXC_D_PCG_XEL_ED_WEC_DTE_PPL_AEE_CNP_FE_CMS_EIX_ETR_EVRG_LNT_PNW_IDA_AEP_DUK_SRE_ATO_NRG_2023-01-01_2025-07-19_1Hour_close_prices"
    prices = load_dataframe(filename)
    prices = prices[["NEE", "EXC"]].tail(100)
    backtester.run(prices, ignore_market_open=True)

    print(dash("order history"))
    print(backtester.get_history())
    print(dash("state history"))
    print(backtester.get_state_history())

    print(dash("ending state"))
    print(backtester.get_state())
