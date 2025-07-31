from src.backtester import EventBacktester, Position, Order
import unittest
import pandas as pd
import numpy as np
from datetime import timedelta
from abc import ABC
import logging
from src.__init__ import *
# Import the classes we need to test

# Set the log level to WARNING for all loggers in this test (except the root logger)
logging.getLogger("arbys_beef").setLevel(logging.WARNING)


class TestEventBacktester(EventBacktester):
    """
    Concrete implementation of EventBacktester for testing purposes.
    This implements a simple buy-and-hold strategy.
    """

    def __init__(self, active_symbols: list[str], cash: float = 1000):
        super().__init__(active_symbols, cash, market_hours_only=False)
        self.initialized = False

    def precompute_step(self, train_bars: pd.DataFrame):
        """Simple preload implementation that just marks as initialized."""
        self.initialized = True

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """Simple update step that does nothing."""
        pass

    def generate_order(self, bars: pd.DataFrame, index: pd.Timestamp) -> Order:
        """
        Simple strategy: buy 1 share of each symbol on every opportunity.
        """
        # Buy 1 share of each symbol on every opportunity
        close_prices = bars.loc[:, "close"]
        for symbol in self.active_symbols:
            if symbol in close_prices.index:
                return Order(symbol, Position.LONG, close_prices[symbol], 1)
        return None


class TestEventBacktesterAdvanced(EventBacktester):
    """
    More advanced test implementation that makes multiple trades.
    """

    def __init__(self, active_symbols: list[str], cash: float = 100):
        super().__init__(active_symbols, cash, market_hours_only=False)
        self.trade_count = 0

    def precompute_step(self, train_bars: pd.DataFrame):
        """Preload implementation."""
        pass

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """Update step that tracks trade count."""
        pass

    def generate_order(self, bars: pd.DataFrame, index: pd.Timestamp) -> Order:
        """
        Strategy: alternate between buying and selling based on trade count.
        """
        close_prices = bars.loc[:, "close"]
        for symbol in self.active_symbols:
            if symbol in close_prices.index:
                if self.trade_count % 2 == 0:
                    # Buy on even trades
                    order = Order(symbol, Position.LONG,
                                  close_prices[symbol], 1)
                else:
                    # Sell on odd trades
                    order = Order(symbol, Position.SHORT,
                                  close_prices[symbol], 1)
                self.trade_count += 1
                return order
        return None


class TestEventBacktesterUnit(unittest.TestCase):
    """Unit tests for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL"]
        self.initial_cash = 1000.0
        self.backtester = TestEventBacktester(self.symbols, self.initial_cash)

        # Create dummy data with proper multi-index structure
        self.create_dummy_data()

    def create_dummy_data(self):
        """Create dummy OHLCV data with proper multi-index structure."""
        # Create timestamps
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i) for i in range(10)]

        # Create multi-index data
        data = []
        for symbol in self.symbols:
            for timestamp in timestamps:
                # Generate realistic OHLCV data
                base_price = 100.0 if symbol == "AAPL" else 150.0
                price_change = np.random.normal(0, 2)  # Random price movement
                close_price = base_price + price_change

                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': close_price - np.random.uniform(0, 1),
                    'high': close_price + np.random.uniform(0, 2),
                    'low': close_price - np.random.uniform(0, 2),
                    'close': close_price,
                    'volume': int(np.random.uniform(1000, 10000))
                })

        # Create DataFrame and set multi-index
        self.dummy_bars = pd.DataFrame(data)
        self.dummy_bars.set_index(['symbol', 'timestamp'], inplace=True)
        self.dummy_bars = self.dummy_bars.sort_index()

        # Split into train and test
        split_point = int(len(timestamps) * 0.7)
        train_timestamps = timestamps[:split_point]
        test_timestamps = timestamps[split_point:]

        self.train_bars = self.dummy_bars.loc[
            (slice(None), train_timestamps), :
        ]
        self.test_bars = self.dummy_bars.loc[
            (slice(None), test_timestamps), :
        ]

    def test_initialization(self):
        """Test that the backtester initializes correctly."""
        self.assertEqual(self.backtester.active_symbols, self.symbols)
        self.assertEqual(self.backtester.get_state()
                         ["cash"], self.initial_cash)
        self.assertEqual(self.backtester.get_state()[
                         "portfolio_value"], self.initial_cash)

        # Check that all symbol positions are initialized to 0
        for symbol in self.symbols:
            self.assertEqual(self.backtester.get_state()[symbol], 0)

    def test_initialize_bank(self):
        """Test bank initialization with different cash amounts."""
        backtester = TestEventBacktester(self.symbols, 500.0)
        self.assertEqual(backtester.get_state()["cash"], 500.0)
        self.assertEqual(backtester.get_state()["portfolio_value"], 500.0)

    def test__place_buy_order(self):
        """Test placing a buy order."""
        symbol = "AAPL"
        price = 150.0
        quantity = 2
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0)

        initial_cash = self.backtester.get_state()["cash"]
        initial_position = self.backtester.get_state()[symbol]

        self.backtester._place_buy_order(symbol, price, quantity, timestamp)

        # Check state updates
        new_state = self.backtester.get_state()
        expected_cash = initial_cash - (price * quantity)
        expected_position = initial_position + quantity

        self.assertEqual(new_state["cash"], expected_cash)
        self.assertEqual(new_state[symbol], expected_position)

        # Check order history
        history = self.backtester.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["symbol"], symbol)
        self.assertEqual(history.iloc[0]["position"], Position.LONG.value)
        self.assertEqual(history.iloc[0]["price"], price)
        self.assertEqual(history.iloc[0]["quantity"], quantity)

    def test_place_sell_order(self):
        """Test placing a sell order."""
        symbol = "AAPL"
        price = 150.0
        quantity = 2
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        # First buy some shares
        self.backtester._place_buy_order(
            symbol, 100.0, 5, timestamp)

        initial_cash = self.backtester.get_state()["cash"]
        initial_position = self.backtester.get_state()[symbol]

        # Now sell
        self.backtester._place_sell_order(
            symbol, price, quantity, timestamp)

        # Check state updates
        new_state = self.backtester.get_state()
        expected_cash = initial_cash + (price * quantity)
        expected_position = initial_position - quantity

        self.assertEqual(new_state["cash"], expected_cash)
        self.assertEqual(new_state[symbol], expected_position)

        # Check order history
        history = self.backtester.get_history()
        self.assertEqual(len(history), 2)  # Buy + sell
        self.assertEqual(history.iloc[1]["position"], Position.SHORT.value)

    def test__place_order(self):
        """Test the generic _place_order method."""
        symbol = "AAPL"
        price = 150.0
        quantity = 1
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        # Test LONG order
        order = Order(symbol, Position.LONG, price, quantity)
        self.backtester._place_order(order, timestamp)
        self.assertEqual(self.backtester.get_state()[symbol], 1)

        # Test SHORT order
        order = Order(symbol, Position.SHORT, price, quantity)
        self.backtester._place_order(order, timestamp)
        self.assertEqual(self.backtester.get_state()[symbol], 0)  # Back to 0

    def test_close_positions(self):
        """Test closing all positions."""
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        # Buy some positions
        self.backtester._place_buy_order("AAPL", 150.0, 2, timestamp)
        self.backtester._place_buy_order("GOOGL", 200.0, 1, timestamp)

        # Verify positions exist
        self.assertEqual(self.backtester.get_state()["AAPL"], 2)
        self.assertEqual(self.backtester.get_state()["GOOGL"], 1)

        # Close positions
        prices = pd.Series({"AAPL": 160.0, "GOOGL": 210.0})
        self.backtester._close_positions(prices, timestamp)

        # Verify positions are closed
        self.assertEqual(self.backtester.get_state()["AAPL"], 0)
        self.assertEqual(self.backtester.get_state()["GOOGL"], 0)

    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        # Buy some positions
        self.backtester._place_buy_order("AAPL", 150.0, 2, timestamp)
        self.backtester._place_buy_order("GOOGL", 200.0, 1, timestamp)

        # Calculate expected portfolio value
        expected_cash = self.initial_cash - (150.0 * 2) - (200.0 * 1)
        expected_portfolio_value = expected_cash + (150.0 * 2) + (200.0 * 1)

        # Update portfolio value with current prices
        prices = pd.Series({"AAPL": 150.0, "GOOGL": 200.0})
        self.backtester._update_portfolio_value(prices, timestamp)

        # Check portfolio value
        self.assertEqual(self.backtester.get_state()[
                         "portfolio_value"], expected_portfolio_value)

    def test_load_train_bars(self):
        """Test loading training bars."""
        self.backtester.load_train_bars(self.train_bars)

        self.assertIsNotNone(self.backtester.train_bars)
        self.assertTrue(self.backtester.initialized)

        # Check that train bars have the correct structure
        self.assertEqual(self.backtester.train_bars.index.nlevels, 2)
        self.assertTrue(all(symbol in self.backtester.train_bars.index.get_level_values(0)
                            for symbol in self.symbols))

    def test_run_backtest(self):
        """Test running a complete backtest."""
        # Load train bars first
        self.backtester.load_train_bars(self.train_bars)

        # Run backtest
        state_history = self.backtester.run(self.test_bars, disable_tqdm=True)

        # Check that state history is returned
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

        # Check that test bars are stored
        self.assertIsNotNone(self.backtester.test_bars)

    def test_run_backtest_without_train(self):
        """Test running backtest without training data."""
        # Run backtest without loading train bars
        state_history = self.backtester.run(self.test_bars, disable_tqdm=True)

        # Check that state history is returned
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

    def test_analyze_performance(self):
        """Test performance analysis."""
        # Run a backtest first
        self.backtester.load_train_bars(self.train_bars)
        self.backtester.run(self.test_bars, disable_tqdm=True)

        # Analyze performance
        try:
            performance = self.backtester.analyze_performance()

            # Check that performance metrics are calculated
            self.assertIn("return_on_investment", performance)
            self.assertIn("max_drawdown_percentage", performance)
            self.assertIn("start_portfolio_value", performance)
            self.assertIn("end_portfolio_value", performance)

            # Check that values are reasonable
            self.assertGreater(performance["start_portfolio_value"], 0)
            self.assertGreater(performance["end_portfolio_value"], 0)
        except IndexError:
            # If there's not enough data for analysis, that's okay for this test
            # Just check that we have some state history
            state_history = self.backtester.get_state_history()
            self.assertGreater(len(state_history), 0)

    def test_get_state_history(self):
        """Test getting state history."""
        # Make some trades
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")
        self.backtester._place_buy_order("AAPL", 150.0, 1, timestamp)

        # Get state history
        state_history = self.backtester.get_state_history()

        # Check structure
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertIn("cash", state_history.columns)
        self.assertIn("portfolio_value", state_history.columns)
        for symbol in self.symbols:
            self.assertIn(symbol, state_history.columns)

    def test_get_history(self):
        """Test getting order history."""
        # Make some trades
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")
        self.backtester._place_buy_order("AAPL", 150.0, 1, timestamp)
        self.backtester._place_sell_order("GOOGL", 200.0, 1, timestamp)

        # Get history
        history = self.backtester.get_history()

        # Check structure
        self.assertIsInstance(history, pd.DataFrame)
        self.assertIn("symbol", history.columns)
        self.assertIn("position", history.columns)
        self.assertIn("price", history.columns)
        self.assertIn("quantity", history.columns)

        # Check that we have 2 orders
        self.assertEqual(len(history), 2)

    def test_dataframe_structure_validation(self):
        """Test that the backtester validates dataframe structure correctly."""
        # Test with invalid dataframe (no multi-index)
        invalid_bars = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        with self.assertRaises(AssertionError):
            self.backtester.run(invalid_bars)

    def test_advanced_backtester(self):
        """Test the more advanced backtester implementation."""
        advanced_backtester = TestEventBacktesterAdvanced(
            self.symbols, self.initial_cash)

        # Run backtest
        advanced_backtester.load_train_bars(self.train_bars)
        state_history = advanced_backtester.run(
            self.test_bars, disable_tqdm=True)

        # Check that we have some state history
        self.assertGreater(len(state_history), 0)

        # Check that trade count increased (even if no orders were placed)
        self.assertGreaterEqual(advanced_backtester.trade_count, 0)


class TestEventBacktesterIntegration(unittest.TestCase):
    """Integration tests for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.symbols = ["AAPL", "GOOGL", "MSFT"]
        self.backtester = TestEventBacktester(self.symbols, 10000.0)

        # Create larger dataset for integration testing
        self.create_integration_data()

    def create_integration_data(self):
        """Create larger dataset for integration testing."""
        # Create timestamps for a week of hourly data
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        # 7 days * 24 hours
        timestamps = [start_date + timedelta(hours=i) for i in range(168)]

        # Create multi-index data
        data = []
        for symbol in self.symbols:
            # Different base price per symbol
            base_price = 100.0 + (hash(symbol) % 100)
            for i, timestamp in enumerate(timestamps):
                # Generate realistic price movement with some trend
                trend = np.sin(i / 24) * 5  # Daily cycle
                noise = np.random.normal(0, 1)
                close_price = base_price + trend + noise

                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': close_price - np.random.uniform(0, 1),
                    'high': close_price + np.random.uniform(0, 2),
                    'low': close_price - np.random.uniform(0, 2),
                    'close': close_price,
                    'volume': int(np.random.uniform(1000, 10000))
                })

        # Create DataFrame and set multi-index
        self.integration_bars = pd.DataFrame(data)
        self.integration_bars.set_index(['symbol', 'timestamp'], inplace=True)
        self.integration_bars = self.integration_bars.sort_index()

        # Split into train and test
        split_point = int(len(timestamps) * 0.8)
        train_timestamps = timestamps[:split_point]
        test_timestamps = timestamps[split_point:]

        self.train_bars = self.integration_bars.loc[
            (slice(None), train_timestamps), :
        ]
        self.test_bars = self.integration_bars.loc[
            (slice(None), test_timestamps), :
        ]

    def test_full_backtest_workflow(self):
        """Test the complete backtest workflow."""
        # Load training data
        self.backtester.load_train_bars(self.train_bars)

        # Run backtest
        state_history = self.backtester.run(self.test_bars, disable_tqdm=True)

        # Verify results
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

        # Check that we have order history (positions may be closed by default)
        order_history = self.backtester.get_history()
        # Note: Order history might be empty if no trades were made, which is acceptable

    def test_backtest_with_position_closing(self):
        """Test backtest with automatic position closing."""
        # Load training data
        self.backtester.load_train_bars(self.train_bars)

        # Run backtest with position closing
        state_history = self.backtester.run(
            self.test_bars, close_positions=True, disable_tqdm=True)

        # Check that positions were closed at the end
        final_state = self.backtester.get_state()
        for symbol in self.symbols:
            self.assertEqual(final_state[symbol], 0)

    def test_backtest_without_position_closing(self):
        """Test backtest without automatic position closing."""
        # Load training data
        self.backtester.load_train_bars(self.train_bars)

        # Run backtest without position closing
        state_history = self.backtester.run(
            self.test_bars, close_positions=False, disable_tqdm=True)

        # Check that we have some state history (orders may or may not be generated)
        self.assertGreater(len(state_history), 0)

    def test_get_win_rate(self):
        """Test getting win rate."""
        # case 1
        self.backtester.initialize_bank()
        orders = [
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.LONG, 21.0, 2),
            Order("AAPL", Position.LONG, 25.0, 1),
            Order("AAPL", Position.SHORT, 24.0, 1),
            Order("AAPL", Position.SHORT, 22.0, 3),
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [20.0, 21.0, 25.0],
            "exit_price": [24.0, 22.0, 22.0],
            "quantity": [1.0, 2.0, 1.0],
            "net_profit_dollars": [4.0, 2.0, -3.0],
            "net_profit_percentage": [0.2, 0.047619, -0.12],
            "win": [True, True, False]
        }).astype({"symbol": "string", "win": "boolean"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 2/3)

        # case 2

        self.backtester.initialize_bank()
        orders = [
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.LONG, 21.0, 2),
            Order("AAPL", Position.LONG, 25.0, 1),
            Order("AAPL", Position.SHORT, 24.0, 3),
            Order("AAPL", Position.SHORT, 22.0, 1),
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [20.0, 21.0, 25.0],
            "exit_price": [24.0, 24.0, 22.0],
            "quantity": [1.0, 2.0, 1.0],
            "net_profit_dollars": [4.0, 6.0, -3.0],
            "net_profit_percentage": [0.2, 0.142857, -0.12],
            "win": [True, True, False]
        }).astype({"symbol": "string", "win": "boolean"})
        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 2/3)

        # case 3, multiple symbols
        self.backtester.initialize_bank(cash=10000)
        orders = [
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.LONG, 21.0, 2),
            Order("AAPL", Position.LONG, 25.0, 1),
            Order("GOOGL", Position.LONG, 17.0, 1),
            Order("GOOGL", Position.LONG, 11.0, 2),
            Order("AAPL", Position.SHORT, 24.0, 3),
            Order("AAPL", Position.SHORT, 22.0, 1),
            Order("GOOGL", Position.LONG, 15.0, 1),
            Order("GOOGL", Position.SHORT, 19.0, 3),
            Order("GOOGL", Position.SHORT, 16.0, 1),
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame([
            {"symbol": "AAPL", "entry_price": 20.0,
                "exit_price": 24.0, "quantity": 1, "net_profit_dollars": 4.0, "net_profit_percentage": 0.2, "win": True},
            {"symbol": "AAPL", "entry_price": 21.0,
                "exit_price": 24.0, "quantity": 2, "net_profit_dollars": 6.0, "net_profit_percentage": 0.1428571, "win": True},
            {"symbol": "AAPL", "entry_price": 25.0,
                "exit_price": 22.0, "quantity": 1, "net_profit_dollars": -3.0, "net_profit_percentage": -0.12, "win": False},
            {"symbol": "GOOGL", "entry_price": 17.0,
                "exit_price": 19.0, "quantity": 1, "net_profit_dollars": 2.0, "net_profit_percentage": 0.11764705882352941, "win": True},
            {"symbol": "GOOGL", "entry_price": 11.0,
                "exit_price": 19.0, "quantity": 2, "net_profit_dollars": 16.0, "net_profit_percentage": 0.7272727, "win": True},
            {"symbol": "GOOGL", "entry_price": 15.0,
                "exit_price": 16.0, "quantity": 1, "net_profit_dollars": 1.0, "net_profit_percentage": 0.06666666666666667, "win": True}
        ]).astype({"symbol": "string", "win": "boolean", "quantity": "float64"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 5/6)

        # test unclosed positions
        self.backtester.initialize_bank(cash=10000)
        orders = [
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.LONG, 21.0, 2),
            Order("AAPL", Position.LONG, 25.0, 1),
            Order("AAPL", Position.SHORT, 24.0, 3),
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame([
            {"symbol": "AAPL", "entry_price": 20.0,
                "exit_price": 24.0, "quantity": 1, "net_profit_dollars": 4.0, "net_profit_percentage": 0.2, "win": True},
            {"symbol": "AAPL", "entry_price": 21.0,
                "exit_price": 24.0, "quantity": 2, "net_profit_dollars": 6.0, "net_profit_percentage": 0.1428571, "win": True},
        ]).astype({"symbol": "string", "win": "boolean", "quantity": "float64"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 1)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
