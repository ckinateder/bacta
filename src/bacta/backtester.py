from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from bacta.utilities import get_logger, is_market_open, floor_decimal
from bacta.utilities.plotting import DEFAULT_FIGSIZE, plt_show

# Get logger for this module
logger = get_logger()

# get version from pyproject.toml
VERSION = "0.4.5"


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
        self.filled = False
        self.fill_price = None

    def get_value(self) -> float:
        """Get the value of the order.
        """
        return self.price * self.quantity

    def __str__(self) -> str:
        """String representation of the order.
        """
        return f"{self.position.name} {round(self.quantity, QUANTITY_PRECISION)} {self.symbol} @ ${self.price:<.3f}"


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
    The user may want to override the run_backtest method to handle the events.

    To use this class, the user must override the generate_order method. If necessary, the user can override the precompute_step and update_step methods.
    The user can call the run_backtest method to run_backtest the backtest.

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

    def __init__(self, active_symbols: list[str], cash: float = 2000, allow_short: bool = True, allow_overdraft: bool = False, min_cash_balance: float = 10.0, min_trade_value: float = 1.0, market_hours_only: bool = True, transaction_cost: float = 0.0, transaction_cost_type: str = "percentage"):
        """
        Initialize the backtester.

        Args:
            active_symbols (list[str]): The symbols to trade.
            cash (float, optional): The initial cash balance. Defaults to 100.
            allow_short (bool, optional): Whether to allow short positions. If False, will not allow the backtester to shortsell. Defaults to True.
            allow_overdraft (bool, optional): Whether to allow overdraft. If False, will not allow the backtester to borrow money. Defaults to False.
            min_cash_balance (float, optional): The minimum cash balance to maintain. If the cash balance is less than this, the backtester will not place any orders. Defaults to 10.0.
            min_trade_value (float, optional): The minimum dollar value of a trade. If the order value is less than this, the order will be skipped. Defaults to 1.0.
            market_hours_only (bool, optional): Whether to only place orders during market hours. Defaults to True.
            transaction_cost (float, optional): The transaction cost as a percentage of the trade value. Defaults to 0.0.
            transaction_cost_type (str, optional): The type of transaction cost to use (dollar or percentage). Defaults to "percentage".
        """
        logger.debug(
            f"Initializing backtester with active symbols: {active_symbols}, cash: {cash}, allow_short: {allow_short}, allow_overdraft: {allow_overdraft}, min_cash_balance: {min_cash_balance}, min_trade_value: {min_trade_value}, market_hours_only: {market_hours_only}")
        self.active_symbols = active_symbols
        self.initial_cash = cash
        self.initialize_bank(cash)
        self.allow_short = allow_short
        self.allow_overdraft = allow_overdraft
        self.min_cash_balance = min_cash_balance
        self.min_trade_value = min_trade_value
        self.market_hours_only = market_hours_only
        self.transaction_cost = transaction_cost
        self.transaction_cost_type = transaction_cost_type
        assert transaction_cost_type in [
            "percentage", "dollar"], "Transaction cost type must be either 'percentage' or 'dollar'"

        # may or may not be needed
        self.train_bars = None
        self.test_bars = None
        self.__already_ran = False

    def reset(self) -> None:
        """Reset the backtester.
        """
        self.__already_ran = False
        self.test_bars = None
        self.initialize_bank(self.initial_cash)

    def initialize_bank(self, cash: float) -> None:
        """Initialize the bank and the state history.

        Args:
            cash (float, optional): The initial cash balance. Defaults to 100.
        """
        self.state_history = pd.DataFrame(
            columns=["cash", "portfolio_value", *self.active_symbols])
        self.state_history.loc[0] = [
            cash, cash, *[0] * len(self.active_symbols)]
        self.order_history = pd.DataFrame(
            columns=["symbol", "position", "price", "quantity"])
        assert len(self.order_history) == 0, "Order history must be empty"
        assert len(
            self.state_history) == 1, "State history must be empty except for the initial state"
        assert self.state_history.index.is_unique, "State history must have a unique index"

    def _update_state(self, symbol: str, price: float, quantity: float, order: Position, index: pd.Timestamp):
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

    def _update_portfolio_value(self, prices: pd.Series, index: pd.Timestamp, ffill: bool = True):
        """Update the portfolio value at the given index using current prices.

        Args:
            index (pd.Timestamp): The timestamp for the update
            prices (dict[str, float]): Dictionary mapping symbol to current price
        """
        if index not in self.state_history.index:
            self.state_history = pd.concat(
                [self.state_history, pd.DataFrame(index=[index])])

        # Calculate portfolio value: cash + sum of (position * price) for each symbol
        portfolio_value = self.state_history.iloc[-1]["cash"]
        for symbol in self.active_symbols:
            if symbol in prices:
                position = self.state_history.iloc[-1][symbol]
                portfolio_value += position * prices[symbol]

        # Update portfolio value in state history
        self.state_history.loc[index, "portfolio_value"] = portfolio_value

        # ffill to get rid of Nans
        if ffill:
            self.state_history = self.state_history.ffill()

    def _update_order_history(self, symbol: str, price: float, quantity: float, order: Position, index: pd.Timestamp):
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

    def _place_buy_order(self, order: Order, index: pd.Timestamp):
        """
        Place a buy order for a given symbol.
        """
        # update state
        self._update_state(order.symbol, order.price,
                           order.quantity, Position.LONG, index)
        self._update_order_history(
            order.symbol, order.price, order.quantity, Position.LONG, index)
        self.state_history.loc[index,
                               "cash"] -= self._calculate_transaction_cost(order)

    def _place_sell_order(self, order: Order, index: pd.Timestamp):
        """
        Place a sell order for a given symbol.
        """

        # update statez
        self._update_state(order.symbol, order.price,
                           order.quantity, Position.SHORT, index)
        self._update_order_history(
            order.symbol, order.price, order.quantity, Position.SHORT, index)
        self.state_history.loc[index,
                               "cash"] -= self._calculate_transaction_cost(order)

    def _calculate_transaction_cost(self, order: Order) -> float:
        """Calculate the transaction cost for a given order.
        """
        if self.transaction_cost_type == "percentage":
            return order.get_value() * self.transaction_cost
        elif self.transaction_cost_type == "dollar":
            return self.transaction_cost

    def _place_order(self, order: Order, index: pd.Timestamp):
        """Place an order for a given symbol.

        Args:
            order (Order): The order to place.
            index (pd.Timestamp): The index of the state.
        """
        # transaction cost
        transaction_cost = self._calculate_transaction_cost(order)

        # check if overdraft is allowed
        adjusted = False
        reason = ""
        if not self.allow_overdraft and order.position == Position.LONG and self.get_current_cash() < (order.get_value() + transaction_cost + self.min_cash_balance):
            order.quantity = floor_decimal(
                (self.get_current_cash() - self.min_cash_balance - transaction_cost) / order.price, QUANTITY_PRECISION)
            adjusted = True
            reason = "(not enough cash)"

        # check if shorting is allowed
        if not self.allow_short and order.position == Position.SHORT and self.get_state()[order.symbol] < order.quantity:
            # can only short what we have
            order.quantity = self.get_state()[order.symbol]
            adjusted = True
            reason = "(no shorting)"

        # transaction cost recalculated with adjusted quantity
        transaction_cost = self._calculate_transaction_cost(order)

        # don't place an order if the value is less than the minimum trade value
        if order.get_value() < self.min_trade_value:
            logger.debug(
                f"Skipping {order} {reason} ({index})")
            return
        else:
            logger.debug(
                f"Placing {f'adjusted ' if adjusted else ''}{order}{f' + ${transaction_cost:.3f} TC' if transaction_cost > 0 else ''} ({index})")

        if order.position == Position.LONG:
            self._place_buy_order(order, index)
        elif order.position == Position.SHORT:
            self._place_sell_order(order, index)

    def _close_positions(self, prices: pd.Series, index: pd.Timestamp):
        """
        Close all positions at the given prices. This bypasses the order placement logic and just places the order.
        The bank is forced to close the position at the given prices.

        Args:
            prices (pd.Series): The prices to close the positions at. Name is the date. Index is the symbols.
        """
        for symbol in self.active_symbols:
            if self.get_state()[symbol] != 0:
                position = self.get_state()[symbol]
                if position > 0:
                    self._place_sell_order(
                        Order(symbol, Position.SHORT, prices[symbol], abs(position)), index)
                else:
                    self._place_buy_order(
                        Order(symbol, Position.LONG, prices[symbol], abs(position)), index)

    def _validate_bars(self, bars: pd.DataFrame):
        """
        Validate the bars.
        """
        assert bars.index.nlevels == 2, "Bars must have a multi-index with (symbol, timestamp) index"
        assert bars.index.get_level_values(0).unique().isin(
            self.active_symbols).all(), "All symbols must be in the bars"
        for symbol in self.active_symbols:
            symbol_bars = bars.xs(symbol, level=0)
            assert symbol_bars.index.is_monotonic_increasing, f"Bars for {symbol} must have a monotonic increasing timestamp"
            assert symbol_bars.index.is_unique, f"Bars for {symbol} must have a unique timestamp"
        assert isinstance(bars.index.get_level_values(
            1)[0], pd.Timestamp), "Bars must have a timestamp index"

        return True

    def run_backtest(self, test_bars: pd.DataFrame, close_positions: bool = True, disable_tqdm: bool = False):
        """
        Run a single period of the backtest over the given dataframe.
        Assume that prices have their indicators already calculated and are in the prices dataframe.

        Args:
            test_bars: DataFrame with bars of the assets. Multi-index with (symbol, timestamp) index and OHLCV columns.
                See the class docstring for more details.
            close_positions (bool, optional): Whether to close positions at the end of the backtest. Defaults to True.
            disable_tqdm (bool, optional): Whether to disable the tqdm progress bar. Defaults to False.
        """
        if self.__already_ran:
            logger.warning(
                "Backtester has already been run. Run self.reset() to reset the backtester.")
            return
        self.__already_ran = True

        # check if the bars are in the correct format
        self._validate_bars(test_bars)
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

        # find the last bar that is in the market hours
        # iterate backwards through the bars
        last_market_bar = None
        for index in timestamps[start_loc:][::-1]:
            if is_market_open(index):
                last_market_bar = index
                break
        if last_market_bar is None:
            logger.warning(
                "No market hours found in the test bars.")

        # iterate through the index of the bars
        with logging_redirect_tqdm(loggers=[get_logger()]):
            pbar = tqdm(enumerate(timestamps[start_loc:]), desc="Backtesting", leave=False, dynamic_ncols=True, total=len(
                timestamps[start_loc:]), position=0, disable=disable_tqdm)
            for i, index in pbar:
                # ensure no leakage
                mask = full_bars.index.get_level_values(1) <= index
                bars_up_to_index = full_bars.loc[mask].copy()

                # Update portfolio value with current close prices before any orders are placed
                current_bar = bars_up_to_index.xs(index, level=1)
                current_close_prices = current_bar.loc[:, "close"]
                self._update_portfolio_value(current_close_prices, index)

                # perform update step
                self.update_step(bars_up_to_index, index)
                if close_positions and (self.market_hours_only and last_market_bar is not None and index == last_market_bar) or (not self.market_hours_only and index == timestamps[-2]):
                    logger.info(f"Closing positions at {index}...")
                    self._close_positions(bars_up_to_index.xs(
                        index, level=1).loc[:, "close"], index)
                    break

                if not self.market_hours_only or is_market_open(index):
                    # make a decision
                    orders = self.generate_orders(current_bar, index)

                    # place the order if not None
                    for order in orders:
                        self._place_order(order, index)

                # Update portfolio value with current close prices after all orders are placed
                current_close_prices = current_bar.loc[:, "close"]
                self._update_portfolio_value(current_close_prices, index)

                # make sure that the state history is the same length as the test bars up to the current index. raise an error if not
                if len(self.state_history) - 1 != i + 1:
                    raise ValueError(
                        "State history is not the same length as the test bars")

                # add current portfolio value to pbar description
                pbar.set_description(
                    f"Backtesting (${self.get_state()['portfolio_value']:.2f})")

            pbar.close()
        # return the state history
        return self.get_state_history()

    def get_buy_and_hold_returns(self, test_bars: pd.DataFrame) -> pd.Series:
        """Assuming an equal weight allocation to all symbols, calculate the buy and hold returns.
        This is done by calculating the returns of each symbol and then taking the mean of the returns.

        Returns:
            pd.Series: The buy and hold returns.
        """
        all_cum_returns = {}
        for symbol in self.active_symbols:
            if symbol in test_bars.index.get_level_values(0):
                symbol_bars = test_bars.xs(
                    symbol, level=0)
                symbol_returns = symbol_bars.loc[:,
                                                 "close"].pct_change().dropna()
                all_cum_returns[symbol] = (1 + symbol_returns).cumprod()

        return pd.DataFrame(all_cum_returns)

    def analyze_performance(self) -> pd.Series:
        """
        Analyze the performance of the backtest.
        """
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            return pd.Series()

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
        max_drawdown_pct = max_drawdown.max()

        # calculate win rate
        win_rate, net_profits = self.get_win_rate(return_net_profits=True)
        avg_trade_return = net_profits["pnl_pct"].mean()
        largest_win = net_profits["pnl_pct"].max()
        largest_loss = net_profits["pnl_pct"].min(
        ) if net_profits["pnl_pct"].min() < 0 else 0
        largest_win_dollars = net_profits["pnl_dollars"].max()
        largest_loss_dollars = net_profits["pnl_dollars"].min()
        max_consecutive_wins = net_profits["win"].astype(
            int).diff().ne(0).cumsum().max()
        max_consecutive_losses = net_profits["win"].astype(
            int).diff().ne(0).cumsum().min()

        # avg trades per day
        if len(state_history) > 1:
            time_diff = state_history.index[-1] - state_history.index[1]
            if time_diff.days > 0:
                avg_orders_per_day = self.order_history.shape[0] / \
                    time_diff.days
            else:
                avg_orders_per_day = 0.0
        else:
            avg_orders_per_day = 0.0

        # calculate percentage of time in market. meaning, the percentage of time that the portfolio was not empty
        # count the number of non-zero symbols
        time_in_market = (
            state_history[self.active_symbols].sum(axis=1) != 0).sum() / len(state_history)

        # compare to buy and hold (prices)
        all_cum_returns = self.get_buy_and_hold_returns(self.test_bars)
        combined_returns = all_cum_returns.mean(axis=1)

        # calculate Sharpe ratio
        try:
            sharpe_ratio = self.calculate_sharpe_ratio()
        except ValueError:
            sharpe_ratio = float('nan')

        return pd.Series({
            "version": VERSION,
            "trading_period_start": state_history.index[1] if len(state_history) > 1 else state_history.index[0],
            "trading_period_end": state_history.index[-1],
            "trading_period_length": state_history.index[-1] - state_history.index[1] if len(state_history) > 1 else pd.Timedelta(0),
            "return_on_investment": return_on_investment,
            "max_drawdown_pct": max_drawdown_pct,
            "start_portfolio_value": start_portfolio_value,
            "end_portfolio_value": end_portfolio_value,
            "min_portfolio_value": state_history["portfolio_value"].min().round(2),
            "max_portfolio_value": state_history["portfolio_value"].max().round(2),
            "min_cash_balance": state_history["cash"].min().round(2),
            "max_cash_balance": state_history["cash"].max().round(2),
            "buy_and_hold_return": combined_returns.iloc[-1],
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "number_of_orders": self.order_history.shape[0],
            "avg_orders_per_day": avg_orders_per_day,
            "number_of_winning_trades": len(net_profits[net_profits["win"]]),
            "number_of_losing_trades": len(net_profits[~net_profits["win"]]),
            "avg_trade_return": avg_trade_return,
            "largest_win":  largest_win,
            "largest_loss": largest_loss,
            "largest_win_dollars": round(largest_win_dollars, 2),
            "largest_loss_dollars": round(largest_loss_dollars, 2),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "time_in_market": time_in_market
        })

    def pretty_format_performance(self) -> str:
        """
        Pretty print the performance of the backtest.
        """
        performance = self.analyze_performance()
        output_lines = [
            f"Backtest Performance:",
            f"- Return on Investment: {(performance['return_on_investment']-1)*100:.2f}%",
            f"- vs. Buy and Hold Return: {(performance['buy_and_hold_return']-1)*100:.2f}%",
            f"- Sharpe Ratio: {performance['sharpe_ratio']:.2f}",
            f"- Max Drawdown Percentage: {(performance['max_drawdown_pct'])*100:.2f}%",
            "",
            f"- Start Portfolio Value: ${performance['start_portfolio_value']:.2f}",
            f"- End Portfolio Value: ${performance['end_portfolio_value']:.2f}",
            f"- Min Portfolio Value: ${performance['min_portfolio_value']:.2f}",
            f"- Max Portfolio Value: ${performance['max_portfolio_value']:.2f}",
            f"- Min Cash Balance: ${performance['min_cash_balance']:.2f}",
            f"- Max Cash Balance: ${performance['max_cash_balance']:.2f}",
            f"- Win Rate: {performance['win_rate']*100:.2f}%",
            "",
            f"- Number of Orders: {performance['number_of_orders']}",
            f"- Avg Orders per Day: {performance['avg_orders_per_day']:.2f}",
            f"- Number of Winning Trades: {performance['number_of_winning_trades']}",
            f"- Number of Losing Trades: {performance['number_of_losing_trades']}",
            f"- Avg Trade Return: {performance['avg_trade_return']*100:.2f}%",
            f"- Largest Win: {performance['largest_win']*100:.2f}% (${performance['largest_win_dollars']:.2f})",
            f"- Largest Loss: {performance['largest_loss']*100:.2f}% (${performance['largest_loss_dollars']:.2f})",
            f"- Max Consecutive Wins: {performance['max_consecutive_wins']}",
            f"- Max Consecutive Losses: {performance['max_consecutive_losses']}",
            "",
            f"- Trading Period Start: {performance['trading_period_start']}",
            f"- Trading Period End: {performance['trading_period_end']}",
            f"- Trading Period Length: {performance['trading_period_length']}",
            f"- Time in Market: {performance['time_in_market']*100:.2f}%"
        ]
        output = "\n".join(output_lines)
        return output

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = None, return_cumulative: bool = False) -> Union[float, pd.Series, Tuple[float, pd.Series]]:
        """
        Calculate the Sharpe ratio for the backtest.

        The Sharpe ratio is a measure of risk-adjusted return, calculated as:
        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation

        Args:
            risk_free_rate (float, optional): The risk-free rate of return (annualized). Defaults to 0.0.
            periods_per_year (int, optional): Number of periods per year for annualization.
                If None, automatically calculated from timestamp differences. Defaults to None.
            return_cumulative (bool, optional): If True, also return the cumulative Sharpe ratio series.
                Defaults to False.

        Returns:
            Union[float, pd.Series, Tuple[float, pd.Series]]: 
                - If return_cumulative=False: The annualized Sharpe ratio (float)
                - If return_cumulative=True: Tuple of (final_sharpe_ratio, cumulative_series)

        Raises:
            ValueError: If there's insufficient data to calculate the Sharpe ratio
        """
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            if return_cumulative:
                return 0.0, pd.Series(dtype=float)
            return 0.0

        state_history = self.get_state_history()

        if state_history.empty or len(state_history) < 2:
            if return_cumulative:
                raise ValueError("Insufficient data to calculate Sharpe ratio")
            else:
                raise ValueError("Insufficient data to calculate Sharpe ratio")

        # Get portfolio values, excluding the initial state (index 0)
        portfolio_values = state_history["portfolio_value"]
        portfolio_values_filtered = portfolio_values[portfolio_values.index != 0]

        if len(portfolio_values_filtered) < 2:
            if return_cumulative:
                raise ValueError(
                    "Insufficient trading data to calculate Sharpe ratio")
            else:
                raise ValueError(
                    "Insufficient trading data to calculate Sharpe ratio")

        # Calculate returns
        returns = portfolio_values_filtered.pct_change().dropna()

        if len(returns) == 0:
            if return_cumulative:
                raise ValueError(
                    "No valid returns data to calculate Sharpe ratio")
            else:
                raise ValueError(
                    "No valid returns data to calculate Sharpe ratio")

        # Auto-calculate periods per year if not provided
        if periods_per_year is None:
            periods_per_year = auto_calculate_periods_per_year(
                portfolio_values_filtered)

        # Calculate final Sharpe ratio
        final_sharpe = calculate_sharpe_ratio_from_returns(
            returns, risk_free_rate, periods_per_year)

        # If only final Sharpe ratio is requested, return it
        if not return_cumulative:
            return final_sharpe

        # Calculate cumulative average Sharpe ratio
        cumulative_sharpe = pd.Series(index=returns.index, dtype=float)

        for i in range(len(returns)):
            # Get all returns from the beginning up to the current point
            cumulative_returns = returns.iloc[:i+1]

            # Use the shared calculation method for consistency
            sharpe_ratio = calculate_sharpe_ratio_from_returns(
                cumulative_returns, risk_free_rate, periods_per_year)

            # Handle infinite values by capping them
            if np.isinf(sharpe_ratio):
                if sharpe_ratio > 0:
                    cumulative_sharpe.iloc[i] = 10.0  # Cap at reasonable value
                else:
                    cumulative_sharpe.iloc[i] = - \
                        10.0  # Cap at reasonable value
            else:
                cumulative_sharpe.iloc[i] = sharpe_ratio

        return final_sharpe, cumulative_sharpe

    def get_win_rate(self, percentage_threshold: float = 0.0, return_net_profits: bool = False) -> tuple[float, pd.DataFrame]:
        """Get the win rate of the backtest. This is done by calculating the net profit for each open position.

        This function handles both:
        1. Long trades: Buy low, sell high (LONG -> SHORT)
        2. Short trades: Sell high, buy low (SHORT -> LONG)

        Example:

        ```
        self.order_history = pd.DataFrame([
            {"symbol": "AAPL", "position": Position.LONG.value,
                "price": 24.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.SHORT.value,
                "price": 22.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.LONG.value,
                "price": 25.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.SHORT.value,
                "price": 24.0, "quantity": 1},
            {"symbol": "AAPL", "position": Position.SHORT.value,
                "price": 22.0, "quantity": 3},
        ])

        self.get_win_rate() -> (0.6666666666666666,
          symbol  entry_price  exit_price  quantity  pnl_dollars  pnl_pct    win
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
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            return 0.0, pd.DataFrame()

        net_profits = pd.DataFrame(
            columns=["symbol", "entry_price",
                     "exit_price", "quantity", "pnl_dollars", "pnl_pct", "win"],
            index=[])
        net_profits = net_profits.astype({
            "symbol": "string",
            "entry_price": "float64",
            "exit_price": "float64",
            "quantity": "float64",
            "pnl_dollars": "float64",
            "pnl_pct": "float64",
            "win": "boolean"
        })

        order_history = self.order_history.copy()
        net_pointer = 0

        for symbol in self.active_symbols:
            # Get all orders for this symbol and sort by timestamp
            symbol_orders = order_history[order_history["symbol"] == symbol].copy(
            )
            symbol_orders = symbol_orders.sort_index()  # Sort by timestamp

            # Separate long and short orders
            long_orders = symbol_orders[symbol_orders["position"]
                                        == Position.LONG.value].copy()
            short_orders = symbol_orders[symbol_orders["position"]
                                         == Position.SHORT.value].copy()

            # Process trades by matching orders chronologically
            long_pointer = 0
            short_pointer = 0

            while long_pointer < len(long_orders) and short_pointer < len(short_orders):
                long_order = long_orders.iloc[long_pointer]
                short_order = short_orders.iloc[short_pointer]
                long_index = long_orders.index[long_pointer]
                short_index = short_orders.index[short_pointer]

                # Determine trade direction based on chronological order
                if long_index < short_index:
                    # LONG -> SHORT trade (buy low, sell high)
                    entry_price = long_order["price"]
                    exit_price = short_order["price"]
                    entry_quantity = long_order["quantity"]
                    exit_quantity = short_order["quantity"]

                    # Calculate trade quantity (minimum of entry and exit)
                    trade_quantity = min(entry_quantity, exit_quantity)

                    # Calculate profit
                    pnl_dollars = (
                        exit_price - entry_price) * trade_quantity
                    pnl_pct = (exit_price - entry_price) / entry_price

                    # Record the trade
                    net_profits.loc[net_pointer, "symbol"] = symbol
                    net_profits.loc[net_pointer, "entry_price"] = entry_price
                    net_profits.loc[net_pointer, "exit_price"] = exit_price
                    net_profits.loc[net_pointer, "quantity"] = trade_quantity
                    net_profits.loc[net_pointer,
                                    "pnl_dollars"] = pnl_dollars
                    net_profits.loc[net_pointer,
                                    "pnl_pct"] = pnl_pct
                    net_profits.loc[net_pointer,
                                    "win"] = pnl_pct > percentage_threshold

                    # Update remaining quantities
                    long_orders.loc[long_index, "quantity"] -= trade_quantity
                    short_orders.loc[short_index, "quantity"] -= trade_quantity

                    # Move pointers
                    if long_orders.loc[long_index, "quantity"] <= 0:
                        long_pointer += 1
                    if short_orders.loc[short_index, "quantity"] <= 0:
                        short_pointer += 1

                else:
                    # SHORT -> LONG trade (sell high, buy low)
                    entry_price = short_order["price"]
                    exit_price = long_order["price"]
                    entry_quantity = short_order["quantity"]
                    exit_quantity = long_order["quantity"]

                    # Calculate trade quantity (minimum of entry and exit)
                    trade_quantity = min(entry_quantity, exit_quantity)

                    # Calculate profit
                    pnl_dollars = (
                        entry_price - exit_price) * trade_quantity
                    pnl_pct = (
                        entry_price - exit_price) / entry_price

                    # Record the trade
                    net_profits.loc[net_pointer, "symbol"] = symbol
                    net_profits.loc[net_pointer, "entry_price"] = entry_price
                    net_profits.loc[net_pointer, "exit_price"] = exit_price
                    net_profits.loc[net_pointer, "quantity"] = trade_quantity
                    net_profits.loc[net_pointer,
                                    "pnl_dollars"] = pnl_dollars
                    net_profits.loc[net_pointer,
                                    "pnl_pct"] = pnl_pct
                    net_profits.loc[net_pointer,
                                    "win"] = pnl_pct > percentage_threshold

                    # Update remaining quantities
                    short_orders.loc[short_index, "quantity"] -= trade_quantity
                    long_orders.loc[long_index, "quantity"] -= trade_quantity

                    # Move pointers
                    if short_orders.loc[short_index, "quantity"] <= 0:
                        short_pointer += 1
                    if long_orders.loc[long_index, "quantity"] <= 0:
                        long_pointer += 1

                net_pointer += 1

        # reset index
        net_profits = net_profits.reset_index(drop=True)

        # calculate win rate as the number of positive net profits divided by the total number of net profits
        if len(net_profits) == 0:
            win_rate = 0.0
        else:
            win_rate = len([profit for profit in net_profits["win"]
                           if profit]) / len(net_profits)

        if return_net_profits:
            return win_rate, net_profits
        else:
            return win_rate

    # getters

    def get_state(self, symbol: str = None, index: pd.Timestamp = None) -> pd.Series:
        """
        Get the current state of the checkbook.
        """
        if symbol is None:
            return self.state_history.iloc[-1]
        else:
            if index is None:
                return self.state_history[symbol].iloc[-1]
            else:
                return self.state_history[symbol].loc[index]

    def get_position(self, symbol: str, index: pd.Timestamp = None) -> float:
        """Get the current position of the symbol.

        Args:
            symbol (str): The symbol of the asset.
            index (pd.Timestamp, optional): The index of the state. Defaults to None.

        Returns:
            pd.Series: _description_
        """
        if index is None:
            return self.state_history[symbol].iloc[-1]
        else:
            return self.state_history[symbol].loc[index]

    def get_current_cash(self) -> float:
        """
        Get the current cash balance.
        """
        return self.state_history.iloc[-1]["cash"]

    def get_current_short_value(self) -> float:
        """
        Get the current total dollar value of all short positions using the
        last known prices from the state history.

        Returns:
            float: Total dollar value of short positions
        """
        current_state = self.get_state()
        total_short_value = 0.0

        for symbol in self.active_symbols:
            position = current_state[symbol]
            if position < 0:  # Short position
                price = current_state[symbol]
                total_short_value += abs(position) * price

        return total_short_value

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
        self._validate_bars(train_bars)
        self.train_bars = train_bars
        self.precompute_step(train_bars)

    def plot_equity_curve(self, figsize: tuple = (20, 12), title: str = "Equity Curve Analysis", save_plot: bool = True, show_plot: bool = False) -> plt.Figure:
        """
        Plot a clean equity curve analysis with strategy vs buy & hold comparison and performance metrics table.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
            title (str): Title for the plot
        """
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            return

        state_history = self.get_state_history()

        if state_history.empty:
            logger.warning(
                "No state history available for plotting equity curve")
            return

        # Create figure with subplots (1 row, 2 columns)
        fig, (ax_plot, ax_table) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Get portfolio values
        portfolio_values = state_history["portfolio_value"]
        portfolio_values_filtered = portfolio_values[portfolio_values.index != 0]

        if len(portfolio_values_filtered) == 0:
            logger.warning(
                "No trading data available for plotting equity curve")
            return None

        # Calculate returns and cumulative returns
        returns = portfolio_values.pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()

        # Get performance metrics
        performance = self.analyze_performance()

        # Main plot: Strategy vs Buy & Hold (like subplot 6 from performance analysis)
        ax_plot.step(cumulative_returns.index, cumulative_returns,
                     label="Strategy", linewidth=2, color='blue', where='post')

        # Add buy and hold comparison
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            all_cum_returns = self.get_buy_and_hold_returns(self.test_bars)
            if not all_cum_returns.empty:
                combined_returns = all_cum_returns.mean(axis=1)
                ax_plot.step(combined_returns.index, combined_returns,
                             label='Buy & Hold (Equal-Weighted)', linewidth=2, color='orange', linestyle='-', where='post')

        ax_plot.axhline(y=1, color='red', linestyle='--',
                        alpha=0.7, label='Break-even')
        ax_plot.set_title("Strategy vs Buy & Hold")
        ax_plot.set_ylabel("Cumulative Return")
        ax_plot.set_xlabel("Date")
        ax_plot.legend()
        ax_plot.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax_plot.get_xticklabels(), rotation=45, ha='right')

        # Performance metrics table
        ax_table.axis('off')

        # Create table data
        table_data = [
            ['Metric', 'Value'],
            ['Return on Investment',
                f"{(performance['return_on_investment']-1)*100:.2f}%"],
            ['Buy & Hold Return',
                f"{(performance['buy_and_hold_return']-1)*100:.2f}%"],
            ['Sharpe Ratio', f"{performance['sharpe_ratio']:.2f}"],
            ['Max Drawdown',
                f"{performance['max_drawdown_pct']*100:.2f}%"],
            ['Win Rate', f"{performance['win_rate']*100:.1f}%"],
            ['Number of Orders', f"{performance['number_of_orders']}"],
            ['Number of Winning Trades',
                f"{performance['number_of_winning_trades']}"],
            ['Number of Losing Trades',
                f"{performance['number_of_losing_trades']}"],
            ['Avg Trade Return',
                f"{performance['avg_trade_return']*100:.2f}%"],
            ['Largest Win', f"{performance['largest_win']*100:.2f}%"],
            ['Largest Loss', f"{performance['largest_loss']*100:.2f}%"],
            ['Start Value', f"${performance['start_portfolio_value']:.2f}"],
            ['End Value', f"${performance['end_portfolio_value']:.2f}"],
            ['Min Portfolio Value',
                f"${performance['min_portfolio_value']:.2f}"],
            ['Max Portfolio Value',
                f"${performance['max_portfolio_value']:.2f}"],
            ['Max Consecutive Wins', f"{performance['max_consecutive_wins']}"],
            ['Max Consecutive Losses',
                f"{performance['max_consecutive_losses']}"],
            ['Trading Period Length',
                f"{performance['trading_period_length']}"],
            ['Time in Market', f"{performance['time_in_market']*100:.1f}%"]
        ]

        # Create table
        table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                               cellLoc='left', loc='upper center',
                               colWidths=[0.6, 0.4])

        # Style the table
        cell_height = 1.7
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, cell_height)

        # Color header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color alternating rows
        for i in range(1, len(table_data)-1):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        # Add second table for configuration flags
        config_table_data = [
            ['Setting', 'Value'],
            ['Allow Short', str(self.allow_short)],
            ['Market Hours Only', str(self.market_hours_only)],
            ['Min Cash Balance', f"${self.min_cash_balance:.2f}"],
            ['Allow Overdraft', str(self.allow_overdraft)],
            ['Min Trade Value', f"${self.min_trade_value:.2f}"],
            ['Initial Cash', f"${self.state_history.iloc[0]['cash']:.2f}"],
        ]

        # Create second table (positioned below the first table)
        config_table = ax_table.table(cellText=config_table_data[1:], colLabels=config_table_data[0],
                                      cellLoc='left', loc='lower center',
                                      colWidths=[0.6, 0.4])

        # Style the second table
        config_table.auto_set_font_size(False)
        config_table.set_fontsize(9)
        config_table.scale(1, cell_height)

        # Color header row
        for i in range(len(config_table_data[0])):
            # Different color for config table
            config_table[(0, i)].set_facecolor('#2196F3')
            config_table[(0, i)].set_text_props(weight='bold', color='white')

        # Color alternating rows
        for i in range(1, len(config_table_data)-1):
            for j in range(len(config_table_data[0])):
                if i % 2 == 0:
                    # Light blue for alternating rows
                    config_table[(i, j)].set_facecolor('#e3f2fd')

        # Adjust layout
        plt.tight_layout()

        if save_plot:
            plt_show(prefix=title.replace(" ", "_").replace(
                "/", ""), show_plot=show_plot)
        elif show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_performance_analysis(self, figsize: tuple = (20, 12), save_plot: bool = True, show_plot: bool = False, title: str = "Performance Analysis") -> plt.Figure:
        """
        Create a comprehensive performance analysis plot with multiple subplots.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
            title (str): Title for the plot
        """
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            return

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

        if len(portfolio_values_filtered) == 0:
            logger.warning(
                "No trading data available for plotting performance analysis")
            return None

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
        _, net_profits = self.get_win_rate(return_net_profits=True)
        returns = net_profits["pnl_pct"]
        ax3.hist(returns, bins=10, alpha=0.7, color='green', edgecolor='black')

        # Mean line
        mean_val = returns.mean()
        ax3.axvline(mean_val, color='red', linestyle='--',
                    label=f'Mean: {mean_val:.4f}')

        # Median line
        # median_val = returns.median()
        # ax3.axvline(median_val, color='orange', linestyle='-.',
        #            label=f'Median: {median_val:.4f}')

        # Std deviation lines
        std_val = returns.std()
        ax3.axvline(mean_val + std_val, color='purple', linestyle=':',
                    label=f'+1 Std: {(mean_val + std_val):.4f}')
        ax3.axvline(mean_val - std_val, color='purple', linestyle=':',
                    label=f'-1 Std: {(mean_val - std_val):.4f}')

        ax3.set_title("Trade Returns Distribution")
        ax3.set_xlabel("Return")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Subplot 4: Cumulative Sharpe Ratio
        final_sharpe, cumulative_sharpe = self.calculate_sharpe_ratio(
            periods_per_year=None, return_cumulative=True)
        if not cumulative_sharpe.empty:
            ax4.plot(cumulative_sharpe.index, cumulative_sharpe.values,
                     label="Cumulative Sharpe Ratio", linewidth=2, color='purple')

            # Add final Sharpe ratio line (should match calculate_sharpe_ratio)
            ax4.axhline(y=final_sharpe, color='orange', linestyle='-',
                        alpha=0.8, linewidth=2, label=f'Final: {final_sharpe:.3f}')

            ax4.axhline(y=0, color='red', linestyle='--',
                        alpha=0.7, label='Zero Sharpe')
            ax4.axhline(y=1, color='green', linestyle='--',
                        alpha=0.7, label='Sharpe = 1')
            ax4.set_title("Cumulative Sharpe Ratio")
            ax4.set_ylabel("Sharpe Ratio")
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.7)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for cumulative Sharpe ratio',
                     transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title("Cumulative Sharpe Ratio")

        # Subplot 5: Buy and Hold Returns
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            all_cum_returns = self.get_buy_and_hold_returns(self.test_bars)
            for symbol in all_cum_returns.columns:
                ax5.step(all_cum_returns.index, all_cum_returns[symbol],
                         label=f'{symbol} B&H', linewidth=1.5, alpha=0.5, where='post')

            # Calculate combined returns (equal-weighted portfolio)
            if not all_cum_returns.empty:
                combined_returns = all_cum_returns.mean(axis=1)
                ax5.step(combined_returns.index, combined_returns,
                         label='Combined B&H', linewidth=2, color='black', linestyle='-', where='post')

            ax5.axhline(y=1, color='red', linestyle='--',
                        alpha=0.7, label='Break-even')
            ax5.set_title("Buy and Hold Returns")
            ax5.set_ylabel("Cumulative Return")
            ax5.legend()
            ax5.grid(True, linestyle='--', alpha=0.7)
        else:
            ax5.text(0.5, 0.5, 'No test prices available',
                     transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title("Buy and Hold Returns")

        # Subplot 6: Strategy vs Buy and Hold
        ax6.step(cumulative_returns.index, cumulative_returns,
                 label="Strategy Returns", linewidth=2, color='blue', where='post')

        # Add buy and hold comparison
        if hasattr(self, 'test_bars') and self.test_bars is not None:
            all_cum_returns = self.get_buy_and_hold_returns(self.test_bars)
            if not all_cum_returns.empty:
                combined_returns = all_cum_returns.mean(axis=1)
                ax6.step(combined_returns.index, combined_returns,
                         label='Combined Buy & Hold', linewidth=2, color='orange', linestyle='-', where='post')

        ax6.axhline(y=1, color='red', linestyle='--',
                    alpha=0.7, label='Break-even')
        ax6.set_title("Strategy vs Buy & Hold")
        ax6.set_ylabel("Cumulative Return")
        ax6.legend()
        ax6.grid(True, linestyle='--', alpha=0.7)

        # Rotate x-axis labels for all subplots
        for ax in [ax1, ax2, ax4, ax5, ax6]:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_plot:
            plt_show(prefix=title.replace(" ", "_").replace(
                "/", ""), show_plot=show_plot)
        elif show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_trade_history(self, figsize: tuple = (20, 12), save_plot: bool = True, show_plot: bool = False, title: str = "Trade History", summary_stats: bool = False, show_quantity: bool = True) -> plt.Figure:
        """
        Plot the price history with trade markers showing buy and sell orders.

        Args:
            figsize (tuple): Figure size for the plot
            save_plot (bool): Whether to save the plot to file
            show_plot (bool): Whether to display the plot
            title (str): Title for the plot
        """
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            return

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
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Handle single symbol case
        if num_symbols == 1:
            axes = [axes]

        for i, symbol in enumerate(self.active_symbols):
            ax = axes[i]

            # Get price data for this symbol
            if symbol in self.test_bars.index.get_level_values(0):
                symbol_bars = self.test_bars.xs(symbol, level=0)

                # Plot price history
                ax.plot(symbol_bars.index, symbol_bars['close'],
                        label=f'{symbol} Close Price', linewidth=1.5, color='blue', alpha=0.7)

                # Add market closed shading if market_hours_only is True
                if self.market_hours_only:
                    # Find periods where market is closed
                    market_closed_periods = []
                    current_period_start = None

                    for timestamp in symbol_bars.index:
                        if not is_market_open(timestamp):
                            if current_period_start is None:
                                current_period_start = timestamp
                        else:
                            if current_period_start is not None:
                                market_closed_periods.append(
                                    (current_period_start, timestamp))
                                current_period_start = None

                    # Handle case where market is closed at the end
                    if current_period_start is not None:
                        market_closed_periods.append(
                            (current_period_start, symbol_bars.index[-1]))

                    # Shade the market closed periods
                    for start_time, end_time in market_closed_periods:
                        ax.axvspan(start_time, end_time, alpha=0.15, color='gray',
                                   label='Market Closed' if start_time == market_closed_periods[0][0] else "")

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
                            if show_quantity:
                                if row['quantity'] < 1:
                                    fmtstr = f"{row['quantity']:.2e}"
                                else:
                                    fmtstr = f"{row['quantity']:.0f}"
                                ax.annotate(fmtstr,
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
                            if show_quantity:
                                if row['quantity'] < 1:
                                    fmtstr = f"{row['quantity']:.2e}"
                                else:
                                    fmtstr = f"{row['quantity']:.0f}"
                                ax.annotate(fmtstr,
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
            plt_show(prefix=title.replace(" ", "_").replace(
                "/", ""), show_plot=show_plot)
        elif show_plot:
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
    def generate_orders(self, bars: pd.DataFrame, index: pd.Timestamp) -> list[Order]:
        """
        Make a decision based on the current prices. This is meant to be overridden by the child class.
        This will place an order if needed.

        Args:
            bars: DataFrame with bars of the assets. Multi-index with (symbol, timestamp) index and OHLCV columns.
            index: The index point of the bars.
        Returns:
            list[Order]: The orders to place.
        """
        raise NotImplementedError(
            "This method must be overridden by the child class.")


class WalkForwardBacktester(EventBacktester):
    """
    Walk forward backtester that runs the backtest for a given number of periods.
    """

    def __init__(self, single_tester: EventBacktester, walk_forward_periods: int = 8, split_ratio: float = 0.8):
        """
        Walk forward backtester that runs the backtest for a given number of periods.
        """
        self.single_tester = single_tester
        self.walk_forward_periods = walk_forward_periods
        self.split_ratio = split_ratio


def calculate_sharpe_ratio_from_returns(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> float:
    """
    Internal method to calculate Sharpe ratio from a series of returns.
    This ensures consistent calculation logic between single and cumulative Sharpe ratios.

    Calculates the annualized Sharpe ratio using the standard formula:
    Sharpe = (R_p - R_f) / σ_p * sqrt(periods_per_year)

    Where:
    - R_p is the mean return per period
    - R_f is the risk-free rate per period
    - σ_p is the standard deviation of returns per period
    - periods_per_year is the number of periods per year

    Args:
        returns (pd.Series): Series of simple returns (pct_change)
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%)
        periods_per_year (int): Number of periods per year

    Returns:
        float: Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate mean return and standard deviation
    mean_return = returns.mean()
    std_return = returns.std()

    # If std_return is nan (e.g. only two identical returns), treat as zero
    if np.isnan(std_return):
        std_return = 0.0

    # Handle edge case where standard deviation is zero
    if std_return == 0:
        rf_per_period = risk_free_rate / periods_per_year
        if np.isclose(mean_return, rf_per_period, atol=1e-10):
            return 0.0  # Zero Sharpe ratio when return equals risk-free rate
        elif mean_return > rf_per_period:
            # Infinite Sharpe ratio for risk-free positive returns
            return float('inf')
        elif mean_return < rf_per_period:
            # Negative infinite Sharpe ratio for risk-free negative returns
            return float('-inf')

    # Calculate Sharpe ratio using the standard annualization formula
    # Sharpe = (R_p - R_f) / σ_p * sqrt(periods_per_year)
    rf_per_period = risk_free_rate / periods_per_year
    sharpe_ratio = (mean_return - rf_per_period) / \
        std_return * np.sqrt(periods_per_year)

    return sharpe_ratio


def auto_calculate_periods_per_year(data: pd.Series) -> int:
    """
    Automatically calculate periods per year from timestamp differences.

    Args:
        data (pd.Series): Series with timestamp index

    Returns:
        int: Calculated periods per year
    """
    # Get timestamps (excluding the initial state index 0)
    timestamps = data.index

    if len(timestamps) >= 2:
        # Calculate average time difference between consecutive timestamps
        time_diffs = []
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i-1]
            time_diffs.append(diff.total_seconds())

        if time_diffs:
            avg_time_diff_seconds = np.mean(time_diffs)
            # Convert to periods per year
            periods_per_year = int(
                365.25 * 24 * 3600 / avg_time_diff_seconds)
            logger.info(
                f"Auto-calculated periods per year: {periods_per_year} (avg time diff: {avg_time_diff_seconds:.1f} seconds)")
            return periods_per_year
        else:
            logger.warning(
                "Could not calculate periods per year, using default: 252")
            return 252  # Fallback to daily
    else:
        logger.warning(
            "Insufficient timestamps to calculate periods per year, using default: 252")
        return 252  # Fallback to daily


"""

    def plot_pairs_spread(self, figsize: tuple = (20, 12), save_plot: bool = True, show_plot: bool = False, title: str | None = None, mark_trades: bool = True, show_quantity: bool = True) -> plt.Figure | None:
Plot the close-price spread between the two active symbols on a single axis.

This function only works when exactly two symbols are active. It is intended for
 pairs trading visualization. The spread is computed as Close(symbol_1) - Close(symbol_2)
  using the intersection of timestamps between both symbols over the test period.

   Args:
        figsize(tuple): Figure size for the plot.
        save_plot(bool): Whether to save the plot to file.
        show_plot(bool): Whether to display the plot.
        title(str | None): Optional title
        defaults to "Pairs Spread: {s1} - {s2}".
        mark_trades(bool): Whether to overlay trade markers for both symbols.
        show_quantity(bool): Whether to annotate trade quantities next to markers.

    Returns:
        plt.Figure | None: The matplotlib Figure, or None if plotting is not possible.
        if not self.__already_ran:
            logger.warning(
                "Backtester has not been run. Run self.run_backtest() to run the backtest.")
            return None

        if not hasattr(self, 'test_bars') or self.test_bars is None:
            logger.warning("No test bars available for plotting pairs spread")
            return None

        if len(self.active_symbols) != 2:
            logger.warning(
                "Pairs spread plot requires exactly two active symbols")
            return None

        symbol_1, symbol_2 = self.active_symbols[0], self.active_symbols[1]

        # Extract close price series for both symbols from test bars
        full_bars = self.test_bars
        if symbol_1 not in full_bars.index.get_level_values(0) or symbol_2 not in full_bars.index.get_level_values(0):
            logger.warning(
                "One or both symbols have no test bars for plotting pairs spread")
            return None

        s1_bars = full_bars.xs(symbol_1, level=0)
        s2_bars = full_bars.xs(symbol_2, level=0)

        if 'close' not in s1_bars.columns or 'close' not in s2_bars.columns:
            logger.warning("Close prices not found for one or both symbols")
            return None

        # Align on common timestamps and compute spread
        aligned = pd.concat(
            [s1_bars['close'].rename(symbol_1), s2_bars['close'].rename(symbol_2)], axis=1
        ).dropna()

        if aligned.empty:
            logger.warning(
                "No overlapping timestamps between the two symbols to compute spread")
            return None

        spread = aligned[symbol_1] - aligned[symbol_2]

        # Build figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        _title = title if title is not None else f"Pairs Spread: {symbol_1} - {symbol_2}"
        fig.suptitle(_title, fontsize=14, fontweight='bold')

        # Plot spread
        ax.plot(spread.index, spread.values,
                label=f"{symbol_1} - {symbol_2}", linewidth=1.8, color='blue', alpha=0.85)

        # Shade market closed periods if applicable
        if self.market_hours_only:
            market_closed_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            current_period_start: pd.Timestamp | None = None
            for timestamp in spread.index:
                if not is_market_open(timestamp):
                    if current_period_start is None:
                        current_period_start = timestamp
                else:
                    if current_period_start is not None:
                        market_closed_periods.append(
                            (current_period_start, timestamp))
                        current_period_start = None
            if current_period_start is not None:
                market_closed_periods.append(
                    (current_period_start, spread.index[-1]))
            for start_time, end_time in market_closed_periods:
                ax.axvspan(start_time, end_time, alpha=0.12, color='gray',
                           label='Market Closed' if start_time == market_closed_periods[0][0] else "")

        # Optionally plot trade markers for both symbols on spread axis
        if mark_trades:
            order_history = self.get_history()
            if not order_history.empty:
                # Ensure we only consider timestamps present in the spread index
                # This prevents scatter plotting on times not in the plotted range
                oh = order_history.loc[order_history.index.isin(spread.index)]

                for sym, color in [(symbol_1, 'green'), (symbol_2, 'orange')]:
                    sym_trades = oh[oh['symbol'] == sym]
                    if sym_trades.empty:
                        continue

                    long_trades = sym_trades[sym_trades['position']
                                             == Position.LONG.value]
                    short_trades = sym_trades[sym_trades['position']
                                              == Position.SHORT.value]

                    # Use spread value at trade timestamps for y coordinates
                    if not long_trades.empty:
                        y_vals = spread.loc[long_trades.index]
                        ax.scatter(long_trades.index, y_vals, marker='^', s=90, color=color, alpha=0.9,
                                   label=f"Buy {sym} ({len(long_trades)})", zorder=5)
                        if show_quantity:
                            for idx, row in long_trades.iterrows():
                                qty = row['quantity']
                                fmtstr = f"{qty:.2e}" if qty < 1 else f"{qty:.0f}"
                                ax.annotate(fmtstr, (idx, spread.loc[idx]), xytext=(6, 10), textcoords='offset points',
                                            fontsize=8, color=color, weight='bold')

                    if not short_trades.empty:
                        y_vals = spread.loc[short_trades.index]
                        ax.scatter(short_trades.index, y_vals, marker='v', s=90, color='red' if sym == symbol_1 else 'purple', alpha=0.9,
                                   label=f"Sell {sym} ({len(short_trades)})", zorder=5)
                        if show_quantity:
                            for idx, row in short_trades.iterrows():
                                qty = row['quantity']
                                fmtstr = f"{qty:.2e}" if qty < 1 else f"{qty:.0f}"
                                ax.annotate(fmtstr, (idx, spread.loc[idx]), xytext=(6, -14), textcoords='offset points',
                                            fontsize=8, color='red' if sym == symbol_1 else 'purple', weight='bold')

        ax.set_ylabel('Spread ($)', fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.35)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        if save_plot:
            plt_show(prefix=(_title if _title else "Pairs_Spread").replace(
                " ", "_").replace("/", ""), show_plot=show_plot)
        elif show_plot:
            plt.show()
        else:
            plt.close()

        return fig

"""
