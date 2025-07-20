from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum

from alpaca.data.timeframe import TimeFrame
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yfinance.base import PriceHistory

# path wrangling
try:
    from src import Position, get_logger
except ImportError:
    from __init__ import Position, get_logger

from src.utilities import dash, load_dataframe
from src.utilities.bars import download_bars, download_close_prices, split_bars
from src.utilities.market import is_market_open
from src.utilities.plotting import DEFAULT_FIGSIZE, plt_show


# Get logger for this module
logger = get_logger("backtester")


class EventBacktester(ABC):
    """
    Event-based backtester superclass.
    There are two central dataframes:
    - state_history: the state of the backtester at each time step.
    - order_history: the history of orders placed.

    Functions to be overridden:
    - preload: preload the indicators for the backtest. Optional.
    - update_step: update the indicators for the backtest at each time step. Optional.
    - take_action: make a decision based on the indicators and the latest bar.

    The backtester is designed to run purely on the test dataset.
    If the user wants to preload the train bars, they must call the load_train_bars method before running the backtest.
    This calls the preload method with the train bars and sets the train_bars attribute. This is useful to ensure a seamless
    transition from train to test data, like in production. Indicators and models relying on lags need historical data at the start
    of the testing period.

    These structures are updated based on the events that occur, and are used to calculate the performance of the backtester.
    The user may want to override the run method to handle the events.

    To use this class, the user must override the take_action method. If necessary, the user can override the preload and update_step methods.
    The user can call the run method to run the backtest.

    The backtester is designed to be used with a multi-index dataframe of bars. The index is (ticker, timestamp) and the columns are OHLCV.
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
        self.train_bars = None
        self.test_bars = None
        # self.full_bars = None

    def initialize_bank(self, cash: float = 100):
        """Initialize the bank and the state history.

        Args:
            cash (float, optional): The initial cash balance. Defaults to 100.
        """
        self.state_history = pd.DataFrame(
            columns=["cash", "portfolio_value", *self.active_tickers])
        self.state_history.loc[0] = [
            cash, cash, *[0] * len(self.active_tickers)]

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

        # update portfolio valuation
        self._update_portfolio_value(pd.Series({ticker: price}), index)

        # ffill to get rid of Nans
        if ffill:
            self.state_history = self.state_history.ffill()

    def _update_portfolio_value(self, prices: pd.Series, index: pd.Timestamp):
        """Update the portfolio value at the given index using current prices.

        Args:
            index (pd.Timestamp): The timestamp for the update
            prices (dict[str, float]): Dictionary mapping ticker to current price
        """
        if index not in self.state_history.index:
            return

        # Get current positions
        current_state = self.state_history.loc[index]
        cash = current_state["cash"]

        # Calculate portfolio value: cash + sum of (position * price) for each ticker
        portfolio_value = cash
        for ticker in self.active_tickers:
            if ticker in prices:
                position = current_state[ticker]
                portfolio_value += position * prices[ticker]

        # Update portfolio value in state history
        self.state_history.loc[index, "portfolio_value"] = portfolio_value

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

    def close_positions(self, prices: pd.Series, index: pd.Timestamp):
        """
        Close all positions at the given prices.

        Args:
            prices (pd.Series): The prices to close the positions at. Name is the date. Index is the tickers.
        """
        for ticker in self.active_tickers:
            if self.get_state()[ticker] != 0:
                position = self.get_state()[ticker]
                if position > 0:
                    self.place_sell_order(
                        ticker, prices[ticker], abs(position), index)
                else:
                    self.place_buy_order(
                        ticker, prices[ticker], abs(position), index)

    def run(self, test_bars: pd.DataFrame, ignore_market_open: bool = False, close_positions: bool = True):
        """
        Run a single period of the backtest over the given dataframe.
        Assume that prices have their indicators already calculated and are in the prices dataframe.

        Args:
            test_bars: DataFrame with bars of the assets. Multi-index with (ticker, timestamp) index and OHLCV columns.
                See the class docstring for more details.
            ignore_market_open (bool, optional): Whether to ignore the market operating hours. Defaults to False.
            close_positions (bool, optional): Whether to close positions at the end of the backtest. Defaults to True.
        """
        # check if the bars are in the correct format
        assert test_bars.index.nlevels == 2, "Bars must have a multi-index with (ticker, timestamp) index"
        assert test_bars.index.get_level_values(0).unique().isin(
            self.active_tickers).all(), "All tickers must be in the bars"
        for ticker in self.active_tickers:
            ticker_bars = test_bars.xs(ticker, level=0)
            assert ticker_bars.index.is_monotonic_increasing, f"Bars for {ticker} must have a monotonic increasing timestamp"
            assert ticker_bars.index.is_unique, f"Bars for {ticker} must have a unique timestamp"
        assert isinstance(test_bars.index.get_level_values(
            1)[0], pd.Timestamp), "Bars must have a timestamp index"

        # store
        self.test_bars = test_bars

        # if train bars is not None, we want to make the full price history
        if self.train_bars is not None:
            logger.info(
                "Train bars have been previously loaded. Concatenating with test bars...")
            full_bars = pd.concat([self.train_bars, test_bars])
            start_loc = len(self.train_bars)
        else:
            full_bars = test_bars
            start_loc = 0

        # get the timestamps
        timestamps = full_bars.index.get_level_values(1)

        # log the backtest range
        logger.info(
            f"Running backtest over {len(full_bars.index[start_loc:])} bars from {timestamps[start_loc]} to {timestamps[-1]}...")

        # iterate through the index of the bars
        for index in timestamps[start_loc:]:
            # perform update step
            self.update_step(full_bars, index)
            if ignore_market_open or is_market_open(index):
                # make a decision
                current_bar = full_bars.xs(index, level=1)
                self.take_action(current_bar, index)

                # Update portfolio value with current close prices
                current_close_prices = current_bar.loc[:, "close"]
                self._update_portfolio_value(current_close_prices, index)

            if close_positions and index == timestamps[-2]:
                logger.info(f"Closing positions at {index}...")
                self.close_positions(full_bars.xs(
                    index, level=1).loc[:, "close"], index)
                break

        logger.info("Backtest complete.")
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

        return pd.Series({
            "return_on_investment": return_on_investment,
            "max_drawdown_percentage": max_drawdown_percentage,
            "start_portfolio_value": start_portfolio_value,
            "end_portfolio_value": end_portfolio_value
        })

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
                Multi-index with (ticker, timestamp) index and OHLCV columns.
        """
        assert all(
            ticker in train_bars.index.get_level_values(0) for ticker in self.active_tickers), "Ticker not found in bars"
        for ticker in self.active_tickers:
            ticker_bars = train_bars.xs(ticker, level=0)
            assert ticker_bars.index.is_monotonic_increasing, f"Bars for {ticker} must have a monotonic increasing timestamp"
            assert ticker_bars.index.is_unique, f"Bars for {ticker} must have a unique timestamp"
        assert isinstance(
            train_bars.index.get_level_values(1)[0], pd.Timestamp), "Bars must have a timestamp index"
        self.train_bars = train_bars
        self.preload(train_bars)

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
        textstr = f'Total Return: {total_return:.2f}%\nMax Drawdown: {max_drawdown:.2f}%\nFinal Value: ${final_value:.2f}'
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
            plt_show(prefix="equity_curve")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_performance_analysis(self, figsize: tuple = (20, 10), save_plot: bool = True, show_plot: bool = True):
        """
        Create a comprehensive performance analysis plot with multiple subplots.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
        """
        state_history = self.get_state_history()

        if state_history.empty:
            logger.warning(
                "No state history available for plotting performance analysis")
            return

        # Create figure with subplots (2 rows, 3 columns)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
              ) = plt.subplots(2, 3, figsize=figsize)

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

        # Subplot 5: Ticker Prices
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            for ticker in self.active_tickers:
                if ticker in self.test_bars.index.get_level_values(0):
                    ax5.step(self.test_bars.xs(ticker, level=0).index, self.test_bars.xs(ticker, level=0).loc[:, "close"],
                             label=ticker, linewidth=1.5, where='post')
            ax5.set_title("Ticker Prices")
            ax5.set_ylabel("Price ($)")
            ax5.legend()
            ax5.grid(True, linestyle='--', alpha=0.7)
        else:
            ax5.text(0.5, 0.5, 'No test prices available',
                     transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title("Ticker Prices")

        # Subplot 6: Buy and Hold Returns
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            all_cum_returns = []
            for ticker in self.active_tickers:
                if ticker in self.test_bars.index.get_level_values(0):
                    ticker_bars = self.test_bars.xs(
                        ticker, level=0)
                    ticker_returns = ticker_bars.loc[:,
                                                     "close"].pct_change().dropna()
                    ticker_cum_returns = (1 + ticker_returns).cumprod()
                    ax6.step(ticker_cum_returns.index, ticker_cum_returns,
                             label=f'{ticker} B&H', linewidth=1.5, alpha=0.7, where='post')
                    all_cum_returns.append(ticker_cum_returns)

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
            plt_show(prefix="performance_analysis")

        if show_plot:
            plt.show()
        else:
            plt.close()

    ################################################################################################
    ################################################################################################

    def preload(self, train_bars: pd.DataFrame):
        """
        Preload the indicators for the backtest. This is meant to be overridden by the child class.

        Args:
            train_bars (pd.DataFrame): The bars of the assets over the TRAINING period.
                Multi-index with (ticker, timestamp) index and OHLCV columns.

        Raises:
            NotImplementedError: This method must be overridden by the child class.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """
        Args:
            bars: DataFrame with bars of the assets. Multi-index with (ticker, timestamp) index and OHLCV columns.
                Bars is the full price history. Might remove this in the future and go with the class state.
            index: The index point of the current step.
        This is optional and can be overridden by the child class.
        """
        pass

    @abstractmethod
    def take_action(self, bars: pd.DataFrame, index: pd.Timestamp):
        """
        Make a decision based on the current prices. This is meant to be overridden by the child class.
        This will place an order if needed.

        Args:
            bars: DataFrame with bars of the assets. Multi-index with (ticker, timestamp) index and OHLCV columns.
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
        self.short_window = 5
        self.long_window = 21

    def preload(self, bars: pd.DataFrame):
        """
        Preload the indicators for the backtest.
        """
        self.sma_shorts = {ticker: bars.xs(ticker, level=0).loc[:, "close"].rolling(
            window=self.short_window).mean() for ticker in self.active_tickers}
        self.sma_longs = {ticker: bars.xs(ticker, level=0).loc[:, "close"].rolling(
            window=self.long_window).mean() for ticker in self.active_tickers}

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """
        Update the state of the backtester.
        """
        # update the indicators
        self.sma_shorts = {ticker: bars.xs(ticker, level=0).loc[:, "close"].rolling(
            window=self.short_window).mean() for ticker in self.active_tickers}
        self.sma_longs = {ticker: bars.xs(ticker, level=0).loc[:, "close"].rolling(
            window=self.long_window).mean() for ticker in self.active_tickers}

    def take_action(self, bar: pd.DataFrame, index: pd.Timestamp):
        """
        Make a decision based on the prices.
        """
        close_prices = bar.loc[:, "close"]
        for ticker in self.active_tickers:
            if self.sma_shorts[ticker][index] > self.sma_longs[ticker][index]:
                self.place_order(Position.LONG, index,
                                 ticker, close_prices[ticker], 1)
            else:
                self.place_order(Position.SHORT, index,
                                 ticker, close_prices[ticker], 1)


if __name__ == "__main__":
    backtester = SMABacktester(["NEE", "EXC"], cash=1000)
    utility_tickers = [
        "NEE", "EXC", "D", "PCG", "XEL",
        "ED", "WEC", "DTE", "PPL", "AEE",
        "CNP", "FE", "CMS", "EIX", "ETR",
        "EVRG", "LNT", "PNW", "IDA", "AEP",
        "DUK", "SRE", "ATO", "NRG",
    ]

    bars = download_bars(["NEE", "EXC"], start_date=datetime(
        2024, 1, 1), end_date=datetime.now() - timedelta(minutes=15), timeframe=TimeFrame.Hour)

    train_bars, test_bars = split_bars(bars, split_ratio=0.9)
    backtester.load_train_bars(train_bars)
    backtester.run(test_bars, ignore_market_open=False)

    print(dash("order history"))
    print(backtester.get_history())
    print(dash("state history"))
    print(backtester.get_state_history())

    print(dash("ending state"))
    print(backtester.get_state())

    print(dash("performance"))
    print(backtester.analyze_performance())

    # Plot the equity curve
    print(dash("plotting equity curve"))
    backtester.plot_equity_curve(title="SMA Strategy Equity Curve")

    # Plot comprehensive performance analysis
    print(dash("plotting performance analysis"))
    backtester.plot_performance_analysis()
