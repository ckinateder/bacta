import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtester import EventBacktester, Position

# set logger level to warning
from src import set_log_level
set_log_level("WARNING")


class TestBacktester(EventBacktester):
    """
    Concrete implementation of EventBacktester for testing purposes.
    """

    def __init__(self, active_tickers: list[str], cash: float = 100):
        super().__init__(active_tickers, cash)
        self.preload_called = False
        self.update_step_called = False
        self.take_action_called = False

    def preload(self, train_prices: pd.DataFrame):
        """Test implementation of preload method."""
        self.preload_called = True
        # Store some basic indicators for testing
        self.sma_short = {ticker: train_prices[ticker].rolling(window=5).mean()
                          for ticker in self.active_tickers}
        self.sma_long = {ticker: train_prices[ticker].rolling(window=10).mean()
                         for ticker in self.active_tickers}

    def update_step(self, prices: pd.DataFrame, index: pd.Timestamp):
        """Test implementation of update_step method."""
        self.update_step_called = True
        # Update indicators
        self.sma_short = {ticker: prices[ticker].rolling(window=5).mean()
                          for ticker in self.active_tickers}
        self.sma_long = {ticker: prices[ticker].rolling(window=10).mean()
                         for ticker in self.active_tickers}

    def take_action(self, prices: pd.Series):
        """Test implementation of take_action method."""
        self.take_action_called = True
        # Simple strategy: buy if short SMA > long SMA, sell otherwise
        for ticker in self.active_tickers:
            if (self.sma_short[ticker].iloc[-1] > self.sma_long[ticker].iloc[-1] and
                    self.get_state()[ticker] <= 0):
                self.place_order(Position.LONG, prices.name,
                                 ticker, prices[ticker], 1)
            elif (self.sma_short[ticker].iloc[-1] <= self.sma_long[ticker].iloc[-1] and
                  self.get_state()[ticker] >= 0):
                self.place_order(Position.SHORT, prices.name,
                                 ticker, prices[ticker], 1)


def generate_dummy_price_data(tickers: list[str], start_date: datetime = None,
                              periods: int = 100) -> pd.DataFrame:
    """
    Generate dummy price data for testing.

    Args:
        tickers: List of ticker symbols
        start_date: Start date for the data (defaults to 2023-01-01)
        periods: Number of periods to generate

    Returns:
        DataFrame with dummy price data
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1, 9, 30)

    # Generate timestamps (hourly data)
    timestamps = [start_date + timedelta(hours=i) for i in range(periods)]

    # Generate random walk prices for each ticker
    data = {}
    for ticker in tickers:
        # Start with a base price between 50 and 200
        base_price = np.random.uniform(50, 200)
        prices = [base_price]

        # Generate random walk
        for _ in range(periods - 1):
            # Random price change between -2% and +2%
            change = np.random.uniform(-0.02, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure price doesn't go below 1

        data[ticker] = prices

    df = pd.DataFrame(data, index=timestamps)
    return df


class TestEventBacktester(unittest.TestCase):
    """Test cases for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tickers = ["AAPL", "GOOGL", "MSFT"]
        self.initial_cash = 1000.0
        self.backtester = TestBacktester(self.tickers, self.initial_cash)

        # Generate dummy price data
        self.train_prices = generate_dummy_price_data(self.tickers, periods=50)
        self.test_prices = generate_dummy_price_data(
            self.tickers,
            start_date=self.train_prices.index[-1] + timedelta(hours=1),
            periods=30
        )

    def test_initialization(self):
        """Test backtester initialization."""
        self.assertEqual(self.backtester.active_tickers, self.tickers)
        self.assertEqual(self.backtester.get_state()
                         ["cash"], self.initial_cash)

        # Check initial positions are zero
        for ticker in self.tickers:
            self.assertEqual(self.backtester.get_state()[ticker], 0)

    def test_initialize_bank(self):
        """Test bank initialization."""
        # Test with different cash amount
        self.backtester.initialize_bank(500.0)
        self.assertEqual(self.backtester.get_state()["cash"], 500.0)
        self.assertEqual(self.backtester.get_state()["portfolio_value"], 500.0)

        # Check state history structure
        expected_columns = ["cash", "portfolio_value"] + self.tickers
        self.assertListEqual(
            list(self.backtester.state_history.columns), expected_columns)

        # Check order history structure
        expected_order_columns = ["ticker", "position", "price", "quantity"]
        self.assertListEqual(
            list(self.backtester.order_history.columns), expected_order_columns)

    def test_place_buy_order(self):
        """Test placing buy orders."""
        ticker = "AAPL"
        price = 150.0
        quantity = 2.0
        index = pd.Timestamp("2023-01-01 10:00:00")

        self.backtester.place_buy_order(ticker, price, quantity, index)

        # Check state update
        state = self.backtester.get_state()
        expected_cash = self.initial_cash - (price * quantity)
        self.assertEqual(state["cash"], expected_cash)
        self.assertEqual(state[ticker], quantity)

        # Check order history
        history = self.backtester.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["ticker"], ticker)
        self.assertEqual(history.iloc[0]["position"], Position.LONG.value)
        self.assertEqual(history.iloc[0]["price"], price)
        self.assertEqual(history.iloc[0]["quantity"], quantity)

    def test_place_sell_order(self):
        """Test placing sell orders."""
        ticker = "AAPL"
        price = 150.0
        quantity = 2.0
        index = pd.Timestamp("2023-01-01 10:00:00")

        self.backtester.place_sell_order(ticker, price, quantity, index)

        # Check state update
        state = self.backtester.get_state()
        expected_cash = self.initial_cash + (price * quantity)
        self.assertEqual(state["cash"], expected_cash)
        self.assertEqual(state[ticker], -quantity)

        # Check order history
        history = self.backtester.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["ticker"], ticker)
        self.assertEqual(history.iloc[0]["position"], Position.SHORT.value)
        self.assertEqual(history.iloc[0]["price"], price)
        self.assertEqual(history.iloc[0]["quantity"], quantity)

    def test_place_order(self):
        """Test generic place_order method."""
        ticker = "AAPL"
        price = 150.0
        quantity = 2.0
        index = pd.Timestamp("2023-01-01 10:00:00")

        # Test LONG order
        self.backtester.place_order(
            Position.LONG, index, ticker, price, quantity)
        state = self.backtester.get_state()
        self.assertEqual(state[ticker], quantity)

        # Reset and test SHORT order
        self.backtester.initialize_bank(self.initial_cash)
        self.backtester.place_order(
            Position.SHORT, index, ticker, price, quantity)
        state = self.backtester.get_state()
        self.assertEqual(state[ticker], -quantity)

    def test_close_positions(self):
        """Test closing positions."""
        ticker = "AAPL"
        price = 150.0
        quantity = 2.0
        index = pd.Timestamp("2023-01-01 10:00:00")

        # First place a buy order
        self.backtester.place_buy_order(ticker, price, quantity, index)

        # Create closing prices
        closing_prices = pd.Series(
            {ticker: 160.0}, name=pd.Timestamp("2023-01-01 11:00:00"))

        # Close positions
        self.backtester.close_positions(closing_prices)

        # Check that position is closed (should be 0)
        state = self.backtester.get_state()
        self.assertEqual(state[ticker], 0)

        # Check that we have two orders (buy and sell)
        history = self.backtester.get_history()
        self.assertEqual(len(history), 2)

    def test_load_train_prices(self):
        """Test loading training prices."""
        self.backtester.load_train_prices(self.train_prices)

        # Check that train_prices is set
        self.assertIsNotNone(self.backtester.train_prices)
        self.assertTrue(self.backtester.preload_called)

        # Check that indicators are calculated
        for ticker in self.tickers:
            self.assertIn(ticker, self.backtester.sma_short)
            self.assertIn(ticker, self.backtester.sma_long)

    def test_run_backtest(self):
        """Test running a complete backtest."""
        # Load training prices first
        self.backtester.load_train_prices(self.train_prices)

        # Run backtest
        result = self.backtester.run(self.test_prices, ignore_market_open=True)

        # Check that methods were called
        self.assertTrue(self.backtester.update_step_called)
        self.assertTrue(self.backtester.take_action_called)

        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check that state history has expected structure
        expected_columns = ["cash", "portfolio_value"] + self.tickers
        self.assertListEqual(list(result.columns), expected_columns)

    def test_get_state_history(self):
        """Test getting state history."""
        # Place some orders to create history
        ticker = "AAPL"
        price = 150.0
        quantity = 2.0
        index = pd.Timestamp("2023-01-01 10:00:00")

        self.backtester.place_buy_order(ticker, price, quantity, index)

        # Get state history
        history = self.backtester.get_state_history()

        # Check structure
        expected_columns = ["cash", "portfolio_value"] + self.tickers
        self.assertListEqual(list(history.columns), expected_columns)

        # Check that cash is rounded to 2 decimal places
        self.assertTrue(all(history["cash"].apply(lambda x: x == round(x, 2))))

        # Check that portfolio_value column exists and has reasonable values
        self.assertIn("portfolio_value", history.columns)
        # Portfolio value should never be negative
        self.assertTrue(all(history["portfolio_value"] >= 0))

    def test_portfolio_value_calculation(self):
        """Test that portfolio value is calculated correctly."""
        ticker = "AAPL"
        price = 150.0
        quantity = 2.0
        index = pd.Timestamp("2023-01-01 10:00:00")

        # Place a buy order
        self.backtester.place_buy_order(ticker, price, quantity, index)

        # Update portfolio value with current price
        self.backtester._update_portfolio_value(index, {ticker: price})

        # Get state
        state = self.backtester.get_state()

        # Calculate expected portfolio value: cash + (position * price)
        expected_cash = self.initial_cash - (price * quantity)
        expected_portfolio_value = expected_cash + (quantity * price)

        self.assertEqual(state["cash"], expected_cash)
        self.assertEqual(state["portfolio_value"], expected_portfolio_value)

        # Test with different price
        new_price = 160.0
        self.backtester._update_portfolio_value(index, {ticker: new_price})
        state = self.backtester.get_state()

        # Portfolio value should update with new price
        expected_portfolio_value_new = expected_cash + (quantity * new_price)
        self.assertEqual(state["portfolio_value"],
                         expected_portfolio_value_new)

    def test_multiple_orders_same_ticker(self):
        """Test multiple orders for the same ticker."""
        ticker = "AAPL"
        index1 = pd.Timestamp("2023-01-01 10:00:00")
        index2 = pd.Timestamp("2023-01-01 11:00:00")

        # Place buy order
        self.backtester.place_buy_order(ticker, 150.0, 2.0, index1)

        # Place another buy order
        self.backtester.place_buy_order(ticker, 160.0, 1.0, index2)

        # Check final state
        state = self.backtester.get_state()
        expected_cash = self.initial_cash - (150.0 * 2.0) - (160.0 * 1.0)
        self.assertEqual(state["cash"], expected_cash)
        self.assertEqual(state[ticker], 3.0)  # 2 + 1

        # Check order history
        history = self.backtester.get_history()
        self.assertEqual(len(history), 2)

    def test_invalid_ticker_assertion(self):
        """Test that invalid tickers raise assertion error."""
        invalid_prices = pd.DataFrame({
            "INVALID": [100, 101, 102],
            "ALSO_INVALID": [200, 201, 202]
        }, index=pd.date_range("2023-01-01", periods=3, freq="h"))

        with self.assertRaises(AssertionError):
            self.backtester.load_train_prices(invalid_prices)

    def test_non_unique_index_assertion(self):
        """Test that non-unique index raises assertion error."""
        non_unique_prices = pd.DataFrame({
            "AAPL": [100, 101, 102],
            "GOOGL": [200, 201, 202]
        }, index=pd.DatetimeIndex([
            pd.Timestamp("2023-01-01 10:00:00"),
            pd.Timestamp("2023-01-01 10:00:00"),  # Duplicate
            pd.Timestamp("2023-01-01 12:00:00")
        ]))

        with self.assertRaises(AssertionError):
            self.backtester.load_train_prices(non_unique_prices)

    def test_non_monotonic_index_assertion(self):
        """Test that non-monotonic index raises assertion error."""
        non_monotonic_prices = pd.DataFrame({
            "AAPL": [100, 101, 102],
            "GOOGL": [200, 201, 202]
        }, index=pd.DatetimeIndex([
            pd.Timestamp("2023-01-01 12:00:00"),  # Later time first
            pd.Timestamp("2023-01-01 10:00:00"),
            pd.Timestamp("2023-01-01 11:00:00")
        ]))

        with self.assertRaises(AssertionError):
            self.backtester.load_train_prices(non_monotonic_prices)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
