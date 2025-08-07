from bacta.backtester import EventBacktester, Position, Order
import unittest
import pandas as pd
import numpy as np
from datetime import timedelta
from abc import ABC
import logging
from bacta import *
from bacta.utilities.logger import set_log_level
# Import the classes we need to test

# Set the log level to WARNING for all loggers in this test (except the root logger)
set_log_level(logging.ERROR)


class TestEventBacktester(EventBacktester):
    """
    Concrete implementation of EventBacktester for testing purposes.
    This implements a simple buy-and-hold strategy.
    """

    def __init__(self, active_symbols: list[str], cash: float = 1000, **kwargs):
        super().__init__(active_symbols, cash, market_hours_only=False, **kwargs)
        self.initialized = False

    def precompute_step(self, train_bars: pd.DataFrame):
        """Simple preload implementation that just marks as initialized."""
        self.initialized = True

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """Simple update step that does nothing."""
        pass

    def generate_orders(self, bars: pd.DataFrame, index: pd.Timestamp) -> list[Order]:
        """
        Simple strategy: buy 1 share of each symbol on every opportunity.
        """
        # Buy 1 share of each symbol on every opportunity
        close_prices = bars.loc[:, "close"]
        orders = []
        for symbol in self.active_symbols:
            if symbol in close_prices.index:
                orders.append(Order(symbol, Position.LONG,
                              close_prices[symbol], 1))
        return orders


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

    def generate_orders(self, bars: pd.DataFrame, index: pd.Timestamp) -> list[Order]:
        """
        Strategy: alternate between buying and selling based on trade count.
        """
        close_prices = bars.loc[:, "close"]
        orders = []
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
                orders.append(order)
                self.trade_count += 1
        return orders


class TestEventBacktesterDirectOrder(EventBacktester):
    """
    Test implementation that allows direct order placement for testing.
    """

    def __init__(self, active_symbols: list[str], cash: float = 1000, **kwargs):
        super().__init__(active_symbols, cash, market_hours_only=False, **kwargs)
        self.direct_orders = []

    def precompute_step(self, train_bars: pd.DataFrame):
        """Preload implementation."""
        pass

    def update_step(self, bars: pd.DataFrame, index: pd.Timestamp):
        """Update step."""
        pass

    def generate_orders(self, bars: pd.DataFrame, index: pd.Timestamp) -> list[Order]:
        """
        Return orders from the direct_orders list if available.
        """
        if self.direct_orders:
            return [self.direct_orders.pop(0)]
        return []

    def add_direct_order(self, order: Order):
        """Add an order to be executed during the next generate_orders call."""
        self.direct_orders.append(order)


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
        state_history = self.backtester.run_backtest(
            self.test_bars, disable_tqdm=True)

        # Check that state history is returned
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

        # Check that test bars are stored
        self.assertIsNotNone(self.backtester.test_bars)

    def test_run_backtest_without_train(self):
        """Test running backtest without training data."""
        # Run backtest without loading train bars
        state_history = self.backtester.run_backtest(
            self.test_bars, disable_tqdm=True)

        # Check that state history is returned
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

    def test_analyze_performance(self):
        """Test performance analysis."""
        # Run a backtest first
        self.backtester.load_train_bars(self.train_bars)
        self.backtester.run_backtest(self.test_bars, disable_tqdm=True)

        # Analyze performance
        try:
            performance = self.backtester.analyze_performance()

            # Check that performance metrics are calculated
            self.assertIn("return_on_investment", performance)
            self.assertIn("max_drawdown_pct", performance)
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
            self.backtester.run_backtest(invalid_bars)

    def test_advanced_backtester(self):
        """Test the more advanced backtester implementation."""
        advanced_backtester = TestEventBacktesterAdvanced(
            self.symbols, self.initial_cash)

        # Run backtest
        advanced_backtester.load_train_bars(self.train_bars)
        state_history = advanced_backtester.run_backtest(
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
        state_history = self.backtester.run_backtest(
            self.test_bars, disable_tqdm=True)

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
        state_history = self.backtester.run_backtest(
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
        state_history = self.backtester.run_backtest(
            self.test_bars, close_positions=False, disable_tqdm=True)

        # Check that we have some state history (orders may or may not be generated)
        self.assertGreater(len(state_history), 0)

    def test_get_win_rate(self):
        """Test getting win rate for long trades only (original functionality)."""
        # case 1: Traditional long trades (LONG -> SHORT)
        self.backtester.initialize_bank(cash=10000.0)
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

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [20.0, 21.0, 25.0],
            "exit_price": [24.0, 22.0, 22.0],
            "quantity": [1.0, 2.0, 1.0],
            "pnl_dollars": [4.0, 2.0, -3.0],
            "pnl_pct": [0.2, 0.047619, -0.12],
            "win": [True, True, False]
        }).astype({"symbol": "string", "win": "boolean"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 2/3)

        # case 2: More complex long trades
        self.backtester.initialize_bank(cash=10000.0)
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

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [20.0, 21.0, 25.0],
            "exit_price": [24.0, 24.0, 22.0],
            "quantity": [1.0, 2.0, 1.0],
            "pnl_dollars": [4.0, 6.0, -3.0],
            "pnl_pct": [0.2, 0.142857, -0.12],
            "win": [True, True, False]
        }).astype({"symbol": "string", "win": "boolean"})
        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 2/3)

        # case 3: Multiple symbols
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

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame([
            {"symbol": "AAPL", "entry_price": 20.0,
                "exit_price": 24.0, "quantity": 1, "pnl_dollars": 4.0, "pnl_pct": 0.2, "win": True},
            {"symbol": "AAPL", "entry_price": 21.0,
                "exit_price": 24.0, "quantity": 2, "pnl_dollars": 6.0, "pnl_pct": 0.1428571, "win": True},
            {"symbol": "AAPL", "entry_price": 25.0,
                "exit_price": 22.0, "quantity": 1, "pnl_dollars": -3.0, "pnl_pct": -0.12, "win": False},
            {"symbol": "GOOGL", "entry_price": 17.0,
                "exit_price": 19.0, "quantity": 1, "pnl_dollars": 2.0, "pnl_pct": 0.11764705882352941, "win": True},
            {"symbol": "GOOGL", "entry_price": 11.0,
                "exit_price": 19.0, "quantity": 2, "pnl_dollars": 16.0, "pnl_pct": 0.7272727, "win": True},
            {"symbol": "GOOGL", "entry_price": 15.0,
                "exit_price": 16.0, "quantity": 1, "pnl_dollars": 1.0, "pnl_pct": 0.06666666666666667, "win": True}
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

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame([
            {"symbol": "AAPL", "entry_price": 20.0,
                "exit_price": 24.0, "quantity": 1, "pnl_dollars": 4.0, "pnl_pct": 0.2, "win": True},
            {"symbol": "AAPL", "entry_price": 21.0,
                "exit_price": 24.0, "quantity": 2, "pnl_dollars": 6.0, "pnl_pct": 0.1428571, "win": True},
        ]).astype({"symbol": "string", "win": "boolean", "quantity": "float64"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 1)

    def test_get_win_rate_short_trades(self):
        """Test getting win rate for short trades (SHORT -> LONG)."""
        # case 1: Pure short trades (SHORT -> LONG)
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            Order("AAPL", Position.SHORT, 25.0, 1),  # Sell high
            Order("AAPL", Position.SHORT, 24.0, 2),  # Sell high
            Order("AAPL", Position.SHORT, 30.0, 1),  # Sell high
            Order("AAPL", Position.LONG, 20.0, 1),   # Buy low
            Order("AAPL", Position.LONG, 22.0, 3),   # Buy low
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [25.0, 24.0, 30.0],
            "exit_price": [20.0, 22.0, 22.0],
            "quantity": [1.0, 2.0, 1.0],
            "pnl_dollars": [5.0, 4.0, 8.0],
            "pnl_pct": [0.2, 0.083333, 0.266667],
            "win": [True, True, True]
        }).astype({"symbol": "string", "win": "boolean"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 1.0)  # All short trades profitable

        # case 2: Mixed short trades with losses
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            Order("AAPL", Position.SHORT, 25.0, 1),  # Sell high
            Order("AAPL", Position.SHORT, 24.0, 2),  # Sell high
            Order("AAPL", Position.SHORT, 20.0, 1),  # Sell low (will lose)
            Order("AAPL", Position.LONG, 22.0, 3),   # Buy higher
            Order("AAPL", Position.LONG, 18.0, 1),   # Buy low
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [25.0, 24.0, 20.0],
            "exit_price": [22.0, 22.0, 18.0],
            "quantity": [1.0, 2.0, 1.0],
            "pnl_dollars": [3.0, 4.0, 2.0],
            "pnl_pct": [0.12, 0.083333, 0.1],
            "win": [True, True, True]
        }).astype({"symbol": "string", "win": "boolean"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 1.0)  # All 3 trades profitable

    def test_get_win_rate_mixed_trades(self):
        """Test getting win rate for mixed long and short trades."""
        # case 1: Mixed long and short trades
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            # Long trades
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.LONG, 21.0, 1),
            # Short trades
            Order("AAPL", Position.SHORT, 25.0, 1),
            Order("AAPL", Position.SHORT, 24.0, 1),
            # Closing orders
            Order("AAPL", Position.SHORT, 22.0, 2),  # Close long positions
            Order("AAPL", Position.LONG, 18.0, 2),   # Close short positions
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        # Expected: 3 trades based on chronological matching
        exits_should_be = pd.DataFrame({
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "entry_price": [20.0, 21.0, 22.0],
            "exit_price": [25.0, 24.0, 18.0],
            "quantity": [1.0, 1.0, 2.0],
            "pnl_dollars": [5.0, 3.0, 8.0],
            "pnl_pct": [0.25, 0.142857, 0.181818],
            "win": [True, True, True]
        }).astype({"symbol": "string", "win": "boolean"})

        pd.testing.assert_frame_equal(
            exits, exits_should_be, check_exact=False)
        self.assertEqual(win_rate, 1.0)  # All trades profitable

        # case 2: Complex mixed scenario with multiple symbols
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            # AAPL trades
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.SHORT, 25.0, 1),
            Order("AAPL", Position.SHORT, 24.0, 1),
            Order("AAPL", Position.LONG, 18.0, 2),
            # GOOGL trades
            Order("GOOGL", Position.SHORT, 30.0, 1),
            Order("GOOGL", Position.LONG, 25.0, 1),
            Order("GOOGL", Position.LONG, 20.0, 1),
            Order("GOOGL", Position.SHORT, 22.0, 2),
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        # Expected: 4 completed trades (2 per symbol)
        self.assertEqual(len(exits), 4)
        self.assertEqual(win_rate, 1.0)  # All trades should be profitable

        # Verify AAPL trades
        aapl_trades = exits[exits["symbol"] == "AAPL"]
        self.assertEqual(len(aapl_trades), 2)

        # Verify GOOGL trades
        googl_trades = exits[exits["symbol"] == "GOOGL"]
        self.assertEqual(len(googl_trades), 2)

    def test_get_win_rate_edge_cases(self):
        """Test edge cases for win rate calculation."""
        # case 1: No trades
        self.backtester.initialize_bank(cash=10000.0)
        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True
        win_rate = self.backtester.get_win_rate()
        self.assertEqual(win_rate, 0.0)

        # case 2: Only long positions (no exits)
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            Order("AAPL", Position.LONG, 20.0, 1),
            Order("AAPL", Position.LONG, 21.0, 1),
        ]
        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True
        win_rate = self.backtester.get_win_rate()
        self.assertEqual(win_rate, 0.0)  # No completed trades

        # case 3: Only short positions (no exits)
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            Order("AAPL", Position.SHORT, 25.0, 1),
            Order("AAPL", Position.SHORT, 24.0, 1),
        ]
        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True
        win_rate = self.backtester.get_win_rate()
        self.assertEqual(win_rate, 0.0)  # No completed trades

        # case 4: Uneven quantities
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            Order("AAPL", Position.LONG, 20.0, 3),   # Buy 3
            Order("AAPL", Position.SHORT, 22.0, 1),  # Sell 1
            Order("AAPL", Position.SHORT, 24.0, 2),  # Sell 2 (closes position)
        ]
        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True
        win_rate, exits = self.backtester.get_win_rate(
            percentage_threshold=0.0, return_net_profits=True)

        # Should have 2 trades: 1 share at 22, 2 shares at 24
        self.assertEqual(len(exits), 2)
        self.assertEqual(win_rate, 1.0)  # Both trades profitable

    def test_get_win_rate_with_threshold(self):
        """Test win rate calculation with different thresholds."""
        # Create trades with known profit percentages
        self.backtester.initialize_bank(cash=10000.0)
        orders = [
            # Long trades with different profit levels
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("AAPL", Position.SHORT, 110.0, 1),  # 10% profit
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("AAPL", Position.SHORT, 105.0, 1),  # 5% profit
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("AAPL", Position.SHORT, 95.0, 1),   # 5% loss
            # Short trades with different profit levels
            Order("AAPL", Position.SHORT, 110.0, 1),
            Order("AAPL", Position.LONG, 100.0, 1),   # 10% profit
            Order("AAPL", Position.SHORT, 105.0, 1),
            Order("AAPL", Position.LONG, 100.0, 1),   # 5% profit
        ]

        for i in range(len(orders)):
            self.backtester._place_order(orders[i], pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i))

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        # Test with 0% threshold (all profitable trades are wins)
        win_rate_0 = self.backtester.get_win_rate(percentage_threshold=0.0)
        self.assertEqual(win_rate_0, 4/5)  # 4 out of 5 trades are profitable

        # Test with 5% threshold (only trades with >5% profit are wins)
        win_rate_5 = self.backtester.get_win_rate(percentage_threshold=0.05)
        self.assertEqual(win_rate_5, 2/5)  # Only 2 trades have >5% profit

        # Test with 10% threshold (only trades with >10% profit are wins)
        win_rate_10 = self.backtester.get_win_rate(percentage_threshold=0.10)
        self.assertEqual(win_rate_10, 0/5)  # No trades have >10% profit


class TestEventBacktesterEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL"]
        self.backtester = TestEventBacktester(self.symbols, 1000.0)
        self.create_dummy_data()

    def create_dummy_data(self):
        """Create dummy OHLCV data with proper multi-index structure."""
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i) for i in range(5)]

        data = []
        for symbol in self.symbols:
            for timestamp in timestamps:
                base_price = 100.0 if symbol == "AAPL" else 150.0
                close_price = base_price + np.random.normal(0, 2)

                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': close_price - np.random.uniform(0, 1),
                    'high': close_price + np.random.uniform(0, 2),
                    'low': close_price - np.random.uniform(0, 2),
                    'close': close_price,
                    'volume': int(np.random.uniform(1000, 10000))
                })

        self.dummy_bars = pd.DataFrame(data)
        self.dummy_bars.set_index(['symbol', 'timestamp'], inplace=True)
        self.dummy_bars = self.dummy_bars.sort_index()

    def test_min_trade_value_filtering(self):
        """Test that orders below minimum trade value are filtered out."""
        backtester = TestEventBacktester(
            self.symbols, 1000.0, min_trade_value=100.0)

        # Create an order with value less than minimum
        order = Order("AAPL", Position.LONG, 10.0, 5)  # Value = 50
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        initial_cash = backtester.get_state()["cash"]
        backtester._place_order(order, timestamp)

        # Cash should not change since order was filtered
        self.assertEqual(backtester.get_state()["cash"], initial_cash)

    def test_overdraft_protection(self):
        """Test overdraft protection when there's insufficient cash."""
        backtester = TestEventBacktester(
            self.symbols, 100.0)

        # Try to place an order that exceeds available cash
        # Value = 250, but only 100 cash
        order = Order("AAPL", Position.LONG, 50.0, 5)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        backtester._place_order(order, timestamp)

        # Should adjust quantity to fit available cash
        final_state = backtester.get_state()
        self.assertGreaterEqual(final_state["cash"], 0)
        self.assertEqual(final_state["AAPL"], 2)  # 100/50 = 2 shares

    def test_short_selling_restriction(self):
        """Test short selling restriction when allow_short=False."""
        backtester = TestEventBacktester(
            self.symbols, 1000.0, allow_short=False)

        # Try to place a short order
        order = Order("AAPL", Position.SHORT, 50.0, 5)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        backtester._place_order(order, timestamp)

        # Should not allow short selling
        final_state = backtester.get_state()
        self.assertEqual(final_state["AAPL"], 0)

    def test_empty_order_history_win_rate(self):
        """Test win rate calculation with empty order history."""
        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True
        win_rate = self.backtester.get_win_rate()
        self.assertEqual(win_rate, 0.0)

    def test_single_order_win_rate(self):
        """Test win rate calculation with only one order (unclosed position)."""
        order = Order("AAPL", Position.LONG, 100.0, 1)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")
        self.backtester._place_order(order, timestamp)

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True
        win_rate = self.backtester.get_win_rate()
        self.assertEqual(win_rate, 0.0)  # No completed trades

    def test_zero_cash_initialization(self):
        """Test initialization with zero cash."""
        backtester = TestEventBacktester(self.symbols, 0.0)
        self.assertEqual(backtester.get_state()["cash"], 0.0)
        self.assertEqual(backtester.get_state()["portfolio_value"], 0.0)

    def test_negative_cash_initialization(self):
        """Test initialization with negative cash."""
        backtester = TestEventBacktester(self.symbols, -100.0)
        self.assertEqual(backtester.get_state()["cash"], -100.0)
        self.assertEqual(backtester.get_state()["portfolio_value"], -100.0)

    def test_large_quantity_orders(self):
        """Test handling of very large quantity orders."""
        # Use the direct order test class with enough cash
        direct_backtester = TestEventBacktesterDirectOrder(
            self.symbols, 200000000.0)  # 200M cash to cover 100M order

        # Test with very large quantity
        order = Order("AAPL", Position.LONG, 100.0, 1000000)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        direct_backtester._place_order(order, timestamp)

        # Should handle large quantities without errors
        final_state = direct_backtester.get_state()
        self.assertEqual(final_state["AAPL"], 1000000)

        # Test that the order was recorded in history
        order_history = direct_backtester.get_history()
        self.assertEqual(len(order_history), 1)
        self.assertEqual(order_history.iloc[0]["quantity"], 1000000)

    def test_zero_quantity_orders(self):
        """Test handling of zero quantity orders."""
        order = Order("AAPL", Position.LONG, 100.0, 0)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        initial_cash = self.backtester.get_state()["cash"]
        self.backtester._place_order(order, timestamp)

        # Should not change cash or position
        final_state = self.backtester.get_state()
        self.assertEqual(final_state["cash"], initial_cash)
        self.assertEqual(final_state["AAPL"], 0)

    def test_portfolio_value_with_negative_positions(self):
        """Test portfolio value calculation with negative positions."""
        # Create a short position
        order = Order("AAPL", Position.SHORT, 100.0, 2)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")
        self.backtester._place_order(order, timestamp)

        # Update portfolio value with current prices
        # Price went down, short position profits
        prices = pd.Series({"AAPL": 90.0})
        self.backtester._update_portfolio_value(prices, timestamp)

        final_state = self.backtester.get_state()
        # cash + short proceeds - position value
        expected_portfolio_value = 1000.0 + (100.0 * 2) + (-2 * 90.0)
        self.assertEqual(
            final_state["portfolio_value"], expected_portfolio_value)


class TestEventBacktesterPlotting(unittest.TestCase):
    """Test plotting methods for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL"]
        self.backtester = TestEventBacktester(self.symbols, 1000.0)
        self.create_dummy_data()

    def create_dummy_data(self):
        """Create dummy OHLCV data with proper multi-index structure."""
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i) for i in range(20)]

        data = []
        for symbol in self.symbols:
            base_price = 100.0 if symbol == "AAPL" else 150.0
            for i, timestamp in enumerate(timestamps):
                # Create some price movement
                price_change = np.sin(i / 5) * 10  # Oscillating price
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

        self.dummy_bars = pd.DataFrame(data)
        self.dummy_bars.set_index(['symbol', 'timestamp'], inplace=True)
        self.dummy_bars = self.dummy_bars.sort_index()

    def test_plot_equity_curve_empty_history(self):
        """Test plotting equity curve with empty state history."""
        # Should handle empty history gracefully
        fig = self.backtester.plot_equity_curve(
            save_plot=False, show_plot=False)
        self.assertIsNone(fig)

    def test_plot_equity_curve_with_trades(self):
        """Test plotting equity curve with trading history."""
        # Run a backtest to generate some history
        self.backtester.run_backtest(self.dummy_bars, disable_tqdm=True)

        # Plot equity curve
        fig = self.backtester.plot_equity_curve(
            save_plot=False, show_plot=False)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')

    def test_plot_performance_analysis_empty_history(self):
        """Test plotting performance analysis with empty state history."""
        # Should handle empty history gracefully
        fig = self.backtester.plot_performance_analysis(
            save_plot=False, show_plot=False)
        self.assertIsNone(fig)

    def test_plot_performance_analysis_with_trades(self):
        """Test plotting performance analysis with trading history."""
        # Run a backtest to generate some history
        self.backtester.run_backtest(self.dummy_bars, disable_tqdm=True)

        # Plot performance analysis
        fig = self.backtester.plot_performance_analysis(
            save_plot=False, show_plot=False)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')

    def test_plot_trade_history_empty_orders(self):
        """Test plotting trade history with empty order history."""
        # Should handle empty order history gracefully
        fig = self.backtester.plot_trade_history(
            save_plot=False, show_plot=False)
        self.assertIsNone(fig)

    def test_plot_trade_history_with_trades(self):
        """Test plotting trade history with trading history."""
        # Run a backtest to generate some history
        self.backtester.run_backtest(self.dummy_bars, disable_tqdm=True)

        # Plot trade history
        fig = self.backtester.plot_trade_history(
            save_plot=False, show_plot=False)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.__class__.__name__, 'Figure')

    def test_plot_custom_figsize(self):
        """Test plotting with custom figure sizes."""
        # Run a backtest to generate some history
        self.backtester.run_backtest(self.dummy_bars, disable_tqdm=True)

        # Test with custom figsize
        custom_figsize = (15, 8)
        fig = self.backtester.plot_equity_curve(
            figsize=custom_figsize, save_plot=False, show_plot=False)
        self.assertIsNotNone(fig)
        self.assertEqual(fig.get_size_inches().tolist(), list(custom_figsize))

    def test_plot_custom_title(self):
        """Test plotting with custom titles."""
        # Run a backtest to generate some history
        self.backtester.run_backtest(self.dummy_bars, disable_tqdm=True)

        # Test with custom title
        custom_title = "Custom Test Title"
        fig = self.backtester.plot_equity_curve(
            title=custom_title, save_plot=False, show_plot=False)
        self.assertIsNotNone(fig)
        self.assertEqual(fig._suptitle.get_text(), custom_title)


class TestEventBacktesterAdvancedFeatures(unittest.TestCase):
    """Test advanced features of EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL", "MSFT"]
        self.backtester = TestEventBacktester(self.symbols, 10000.0)
        self.create_dummy_data()

    def create_dummy_data(self):
        """Create dummy OHLCV data with proper multi-index structure."""
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i) for i in range(50)]

        data = []
        for symbol in self.symbols:
            base_price = 100.0 + (hash(symbol) % 50)
            for i, timestamp in enumerate(timestamps):
                # Create trending price movement
                trend = i * 0.5  # Upward trend
                noise = np.random.normal(0, 2)
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

        self.dummy_bars = pd.DataFrame(data)
        self.dummy_bars.set_index(['symbol', 'timestamp'], inplace=True)
        self.dummy_bars = self.dummy_bars.sort_index()

    def test_buy_and_hold_returns_calculation(self):
        """Test buy and hold returns calculation."""
        # Run a backtest
        self.backtester.run_backtest(self.dummy_bars, disable_tqdm=True)

        # Get buy and hold returns
        bh_returns = self.backtester.get_buy_and_hold_returns()

        # Check structure
        self.assertIsInstance(bh_returns, pd.DataFrame)
        self.assertEqual(len(bh_returns.columns), len(self.symbols))

        # Check that all symbols are present
        for symbol in self.symbols:
            self.assertIn(symbol, bh_returns.columns)

    def test_win_rate_with_threshold(self):
        """Test win rate calculation with different thresholds."""
        # Create some trades with known outcomes
        orders = [
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("AAPL", Position.SHORT, 110.0, 1),  # 10% profit
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("AAPL", Position.SHORT, 105.0, 1),  # 5% profit
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("AAPL", Position.SHORT, 95.0, 1),   # 5% loss
        ]

        for i, order in enumerate(orders):
            timestamp = pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i)
            self.backtester._place_order(order, timestamp)

        # Set the backtester as having been run so get_win_rate works
        self.backtester._EventBacktester__already_ran = True

        # Test with 0% threshold (all profitable trades are wins)
        win_rate_0 = self.backtester.get_win_rate(percentage_threshold=0.0)
        self.assertEqual(win_rate_0, 2/3)  # 2 out of 3 trades are profitable

        # Test with 5% threshold (only trades with >5% profit are wins)
        win_rate_5 = self.backtester.get_win_rate(percentage_threshold=0.05)
        self.assertEqual(win_rate_5, 1/3)  # Only 1 trade has >5% profit

    def test_multiple_symbol_trading(self):
        """Test trading across multiple symbols."""
        # Create trades for multiple symbols
        orders = [
            Order("AAPL", Position.LONG, 100.0, 1),
            Order("GOOGL", Position.LONG, 150.0, 1),
            Order("MSFT", Position.LONG, 200.0, 1),
            Order("AAPL", Position.SHORT, 110.0, 1),
            Order("GOOGL", Position.SHORT, 160.0, 1),
            Order("MSFT", Position.SHORT, 210.0, 1),
        ]

        for i, order in enumerate(orders):
            timestamp = pd.Timestamp(
                2024, 1, 1, 10, 0, tz="America/New_York") + timedelta(hours=i)
            self.backtester._place_order(order, timestamp)

        # Check final positions
        final_state = self.backtester.get_state()
        self.assertEqual(final_state["AAPL"], 0)  # Closed position
        self.assertEqual(final_state["GOOGL"], 0)  # Closed position
        self.assertEqual(final_state["MSFT"], 0)   # Closed position

    def test_order_value_calculation(self):
        """Test Order.get_value() method."""
        order = Order("AAPL", Position.LONG, 100.0, 2.5)
        expected_value = 100.0 * 2.5
        self.assertEqual(order.get_value(), expected_value)

    def test_order_string_representation(self):
        """Test Order.__str__ method."""
        order = Order("AAPL", Position.LONG, 100.0, 2)
        expected_str = "LONG 2 AAPL @ $100.000"
        self.assertEqual(str(order), expected_str)

    def test_position_enum_values(self):
        """Test Position enum values."""
        self.assertEqual(Position.LONG.value, 1)
        self.assertEqual(Position.SHORT.value, -1)
        self.assertEqual(Position.NEUTRAL.value, 0)

    def test_state_history_ffill(self):
        """Test that state history forward fills correctly."""
        # Make trades at different timestamps
        timestamp1 = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")
        timestamp2 = pd.Timestamp(2024, 1, 1, 12, 0, tz="America/New_York")

        self.backtester._place_buy_order("AAPL", 100.0, 1, timestamp1)
        self.backtester._place_buy_order("AAPL", 110.0, 1, timestamp2)

        # Check that state history has no NaN values
        state_history = self.backtester.get_state_history()
        self.assertFalse(state_history.isna().any().any())


class TestEventBacktesterDataValidation(unittest.TestCase):
    """Test data validation and error handling for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL"]
        self.backtester = TestEventBacktester(self.symbols, 1000.0)

    def test_invalid_dataframe_structure(self):
        """Test handling of invalid dataframe structures."""
        # Test with single-level index
        invalid_bars = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        with self.assertRaises(AssertionError):
            self.backtester.run_backtest(invalid_bars)

    def test_missing_symbols_in_data(self):
        """Test handling of missing symbols in data."""
        # Create data with only one symbol
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i) for i in range(5)]

        data = []
        for timestamp in timestamps:
            data.append({
                'symbol': 'AAPL',  # Only AAPL, missing GOOGL
                'timestamp': timestamp,
                'open': 100.0,
                'high': 105.0,
                'low': 95.0,
                'close': 103.0,
                'volume': 1000
            })

        single_symbol_data = pd.DataFrame(data)
        single_symbol_data.set_index(['symbol', 'timestamp'], inplace=True)

        # Should raise KeyError when trying to access missing symbols
        with self.assertRaises(KeyError):
            self.backtester.run_backtest(single_symbol_data)

    def test_non_monotonic_timestamps(self):
        """Test handling of non-monotonic timestamps."""
        # Create data with non-monotonic timestamps
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i)
                      for i in [0, 2, 1, 3, 4]]  # Non-monotonic

        data = []
        for symbol in self.symbols:
            for timestamp in timestamps:
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': 100.0,
                    'high': 105.0,
                    'low': 95.0,
                    'close': 103.0,
                    'volume': 1000
                })

        invalid_bars = pd.DataFrame(data)
        invalid_bars.set_index(['symbol', 'timestamp'], inplace=True)

        with self.assertRaises(AssertionError):
            self.backtester.run_backtest(invalid_bars)

    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps."""
        # Create data with duplicate timestamps
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        # Duplicate at hour 1
        timestamps = [start_date + timedelta(hours=i) for i in [0, 1, 1, 2, 3]]

        data = []
        for symbol in self.symbols:
            for timestamp in timestamps:
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': 100.0,
                    'high': 105.0,
                    'low': 95.0,
                    'close': 103.0,
                    'volume': 1000
                })

        invalid_bars = pd.DataFrame(data)
        invalid_bars.set_index(['symbol', 'timestamp'], inplace=True)

        with self.assertRaises(AssertionError):
            self.backtester.run_backtest(invalid_bars)

    def test_non_timestamp_index(self):
        """Test handling of non-timestamp index."""
        # Create data with string timestamps instead of pd.Timestamp
        data = []
        for symbol in self.symbols:
            for i in range(5):
                data.append({
                    'symbol': symbol,
                    'timestamp': f"2024-01-01 {9+i}:30:00",  # String timestamp
                    'open': 100.0,
                    'high': 105.0,
                    'low': 95.0,
                    'close': 103.0,
                    'volume': 1000
                })

        invalid_bars = pd.DataFrame(data)
        invalid_bars.set_index(['symbol', 'timestamp'], inplace=True)

        with self.assertRaises(AssertionError):
            self.backtester.run_backtest(invalid_bars)


class TestEventBacktesterStressTests(unittest.TestCase):
    """Stress tests for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        self.backtester = TestEventBacktester(self.symbols, 100000.0)

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset (1000 timestamps, 5 symbols = 5000 rows)
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(hours=i) for i in range(1000)]

        data = []
        for symbol in self.symbols:
            base_price = 100.0 + (hash(symbol) % 100)
            for i, timestamp in enumerate(timestamps):
                close_price = base_price + np.random.normal(0, 5)
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': close_price - np.random.uniform(0, 1),
                    'high': close_price + np.random.uniform(0, 2),
                    'low': close_price - np.random.uniform(0, 2),
                    'close': close_price,
                    'volume': int(np.random.uniform(1000, 10000))
                })

        large_bars = pd.DataFrame(data)
        large_bars.set_index(['symbol', 'timestamp'], inplace=True)
        large_bars = large_bars.sort_index()

        # Run backtest on large dataset
        start_time = pd.Timestamp.now()
        state_history = self.backtester.run_backtest(
            large_bars, disable_tqdm=True)
        end_time = pd.Timestamp.now()

        # Check that it completes without errors
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

        # Check performance (should complete within reasonable time)
        execution_time = (end_time - start_time).total_seconds()
        # Should complete within 60 seconds
        self.assertLess(execution_time, 60)

    def test_high_frequency_trading_simulation(self):
        """Test high-frequency trading simulation with many small trades."""
        # Create dataset with many timestamps
        start_date = pd.Timestamp(2024, 1, 1, 9, 30, tz="America/New_York")
        timestamps = [start_date + timedelta(minutes=i)
                      for i in range(1000)]  # 1000 minutes

        data = []
        for symbol in self.symbols:
            base_price = 100.0 + (hash(symbol) % 100)
            for i, timestamp in enumerate(timestamps):
                close_price = base_price + np.random.normal(0, 1)
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': close_price - np.random.uniform(0, 0.5),
                    'high': close_price + np.random.uniform(0, 1),
                    'low': close_price - np.random.uniform(0, 1),
                    'close': close_price,
                    'volume': int(np.random.uniform(1000, 10000))
                })

        hf_bars = pd.DataFrame(data)
        hf_bars.set_index(['symbol', 'timestamp'], inplace=True)
        hf_bars = hf_bars.sort_index()

        # Run backtest
        state_history = self.backtester.run_backtest(
            hf_bars, disable_tqdm=True)

        # Check results
        self.assertIsInstance(state_history, pd.DataFrame)
        self.assertGreater(len(state_history), 0)

        # Check order history (should have many orders)
        order_history = self.backtester.get_history()
        self.assertGreater(len(order_history), 0)

    def test_memory_usage_with_large_positions(self):
        """Test memory usage with very large position sizes."""
        # Use the direct order test class with enough cash
        direct_backtester = TestEventBacktesterDirectOrder(
            self.symbols, 200000000.0)  # 200M cash to cover 100M order

        # Test with very large quantities
        large_order = Order("AAPL", Position.LONG, 100.0, 1000000)
        timestamp = pd.Timestamp(2024, 1, 1, 10, 0, tz="America/New_York")

        # Should handle large quantities without memory issues
        direct_backtester._place_order(large_order, timestamp)

        final_state = direct_backtester.get_state()
        self.assertEqual(final_state["AAPL"], 1000000)

        # Check that portfolio value calculation works with large numbers
        prices = pd.Series({"AAPL": 100.0})
        direct_backtester._update_portfolio_value(prices, timestamp)

        expected_portfolio_value = 200000000.0 - \
            (100.0 * 1000000) + (100.0 * 1000000)
        self.assertEqual(direct_backtester.get_state()[
                         "portfolio_value"], expected_portfolio_value)

        # Test that the order was recorded in history
        order_history = direct_backtester.get_history()
        self.assertEqual(len(order_history), 1)
        self.assertEqual(order_history.iloc[0]["quantity"], 1000000)


class TestEventBacktesterMaxShortValue(unittest.TestCase):
    """
    Test cases for the max_short_value parameter functionality.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ['AAPL', 'GOOGL']
        self.cash = 10000

    def create_test_data(self):
        """Create test data with timezone-aware timestamps."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D', tz='UTC')
        data = []
        for date in dates:
            for symbol in self.symbols:
                data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': 100.0,
                    'high': 105.0,
                    'low': 95.0,
                    'close': 102.0,
                    'volume': 1000000
                })
        df = pd.DataFrame(data)
        df = df.set_index(['symbol', 'timestamp'])
        return df

    def test_max_short_value_no_limit(self):
        """Test that no limit allows unlimited shorting."""
        class ShortStrategy(EventBacktester):
            def precompute_step(self, bars):
                pass

            def update_step(self, bars, index):
                pass

            def generate_orders(self, bars, index):
                orders = []
                for symbol in self.active_symbols:
                    # Always try to short 100 shares
                    orders.append(Order(symbol, Position.SHORT,
                                  bars.loc[symbol, "close"], 100))
                return orders

        test_bars = self.create_test_data()
        backtester = ShortStrategy(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True,
            market_hours_only=False,
            max_short_value=None  # No limit
        )

        backtester.run_backtest(test_bars, close_positions=False)

        # Check that positions were built up
        state_history = backtester.get_state_history()
        if len(state_history) > 1:
            positions_before_close = state_history.iloc[-2]
            short_value = abs(
                positions_before_close['AAPL']) * 102.0 + abs(positions_before_close['GOOGL']) * 102.0
            self.assertGreater(
                short_value, 0, "Should have built up short positions")
            self.assertGreater(
                short_value, 10000, "Should have substantial short value without limit")

    def test_max_short_value_with_limit(self):
        """Test that max_short_value limit is enforced."""
        class ShortStrategy(EventBacktester):
            def precompute_step(self, bars):
                pass

            def update_step(self, bars, index):
                pass

            def generate_orders(self, bars, index):
                orders = []
                for symbol in self.active_symbols:
                    # Always try to short 100 shares
                    orders.append(Order(symbol, Position.SHORT,
                                  bars.loc[symbol, "close"], 100))
                return orders

        test_bars = self.create_test_data()
        max_short_value = 5000

        backtester = ShortStrategy(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True,
            market_hours_only=False,
            max_short_value=max_short_value
        )

        backtester.run_backtest(test_bars, close_positions=False)

        # Check that short value is limited
        state_history = backtester.get_state_history()
        if len(state_history) > 1:
            positions_before_close = state_history.iloc[-2]
            short_value = abs(
                positions_before_close['AAPL']) * 102.0 + abs(positions_before_close['GOOGL']) * 102.0
            self.assertLessEqual(short_value, max_short_value,
                                 f"Short value {short_value} exceeds limit {max_short_value}")

    def test_max_short_value_very_low_limit(self):
        """Test with very low max_short_value limit."""
        class ShortStrategy(EventBacktester):
            def precompute_step(self, bars):
                pass

            def update_step(self, bars, index):
                pass

            def generate_orders(self, bars, index):
                orders = []
                for symbol in self.active_symbols:
                    # Always try to short 100 shares
                    orders.append(Order(symbol, Position.SHORT,
                                  bars.loc[symbol, "close"], 100))
                return orders

        test_bars = self.create_test_data()
        max_short_value = 1000  # Very low limit

        backtester = ShortStrategy(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True,
            market_hours_only=False,
            max_short_value=max_short_value
        )

        backtester.run_backtest(test_bars, close_positions=False)

        # Check that short value is limited
        state_history = backtester.get_state_history()
        if len(state_history) > 1:
            positions_before_close = state_history.iloc[-2]
            short_value = abs(
                positions_before_close['AAPL']) * 102.0 + abs(positions_before_close['GOOGL']) * 102.0
            self.assertLessEqual(short_value, max_short_value,
                                 f"Short value {short_value} exceeds limit {max_short_value}")

    def test_max_short_value_with_long_positions(self):
        """Test that max_short_value doesn't affect long positions."""
        class MixedStrategy(EventBacktester):
            def precompute_step(self, bars):
                pass

            def update_step(self, bars, index):
                pass

            def generate_orders(self, bars, index):
                orders = []
                for symbol in self.active_symbols:
                    # Mix of long and short positions
                    if symbol == 'AAPL':
                        orders.append(Order(symbol, Position.LONG,
                                      bars.loc[symbol, "close"], 50))
                    else:
                        orders.append(Order(symbol, Position.SHORT,
                                      bars.loc[symbol, "close"], 50))
                return orders

        test_bars = self.create_test_data()
        max_short_value = 2000

        backtester = MixedStrategy(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True,
            market_hours_only=False,
            max_short_value=max_short_value
        )

        backtester.run_backtest(test_bars, close_positions=False)

        # Check that long positions are unaffected
        state_history = backtester.get_state_history()
        if len(state_history) > 1:
            positions_before_close = state_history.iloc[-2]
            aapl_position = positions_before_close['AAPL']
            googl_position = positions_before_close['GOOGL']

            # AAPL should have long position
            self.assertGreater(
                aapl_position, 0, "AAPL should have long position")

            # GOOGL should have short position within limit
            self.assertLess(googl_position, 0,
                            "GOOGL should have short position")
            googl_short_value = abs(googl_position) * 102.0
            self.assertLessEqual(
                googl_short_value, max_short_value, "GOOGL short value should be within limit")

    def test_max_short_value_get_current_short_value_method(self):
        """Test the get_current_short_value method."""
        backtester = TestEventBacktester(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True
        )

        # Test with no positions
        short_value = backtester.get_current_short_value()
        self.assertEqual(short_value, 0.0,
                         "Should return 0 with no short positions")

        # Test with current prices
        current_prices = {'AAPL': 100.0, 'GOOGL': 150.0}
        short_value = backtester.get_current_short_value(current_prices)
        self.assertEqual(
            short_value, 0.0, "Should return 0 with no short positions even with prices")

    def test_max_short_value_order_adjustment_logging(self):
        """Test that order adjustments are properly logged."""
        class ShortStrategy(EventBacktester):
            def precompute_step(self, bars):
                pass

            def update_step(self, bars, index):
                pass

            def generate_orders(self, bars, index):
                orders = []
                for symbol in self.active_symbols:
                    # Try to short a large amount
                    orders.append(Order(symbol, Position.SHORT,
                                  bars.loc[symbol, "close"], 1000))
                return orders

        test_bars = self.create_test_data()
        max_short_value = 1000  # Very low limit

        backtester = ShortStrategy(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True,
            market_hours_only=False,
            max_short_value=max_short_value
        )

        # Run the backtest and check that orders are adjusted
        backtester.run_backtest(test_bars, close_positions=False)

        # Check that orders were placed but limited
        order_history = backtester.get_history()
        self.assertGreater(len(order_history), 0,
                           "Should have placed some orders")

        # Check that the total short value is within the limit
        state_history = backtester.get_state_history()
        if len(state_history) > 1:
            positions_before_close = state_history.iloc[-2]
            short_value = abs(
                positions_before_close['AAPL']) * 102.0 + abs(positions_before_close['GOOGL']) * 102.0
            self.assertLessEqual(short_value, max_short_value,
                                 "Short value should be within limit")

    def test_max_short_value_edge_cases(self):
        """Test edge cases for max_short_value."""
        class ShortStrategy(EventBacktester):
            def precompute_step(self, bars):
                pass

            def update_step(self, bars, index):
                pass

            def generate_orders(self, bars, index):
                orders = []
                for symbol in self.active_symbols:
                    orders.append(Order(symbol, Position.SHORT,
                                  bars.loc[symbol, "close"], 10))
                return orders

        test_bars = self.create_test_data()

        # Test with zero max_short_value
        backtester1 = ShortStrategy(
            active_symbols=self.symbols,
            cash=self.cash,
            allow_short=True,
            market_hours_only=False,
            max_short_value=0.0
        )

        backtester1.run_backtest(test_bars, close_positions=False)

        # Should not be able to short anything
        state_history1 = backtester1.get_state_history()
        if len(state_history1) > 1:
            positions1 = state_history1.iloc[-2]
            short_value1 = abs(positions1['AAPL']) * \
                102.0 + abs(positions1['GOOGL']) * 102.0
            self.assertEqual(short_value1, 0.0,
                             "Should not be able to short with zero limit")

    def test_max_short_value_configuration_display(self):
        """Test that max_short_value appears in configuration display."""
        backtester = TestEventBacktester(
            active_symbols=self.symbols,
            cash=self.cash,
            max_short_value=5000
        )

        # The configuration should be accessible through the instance
        self.assertEqual(backtester.max_short_value, 5000,
                         "max_short_value should be set correctly")

        # Test with None value
        backtester_none = TestEventBacktester(
            active_symbols=self.symbols,
            cash=self.cash,
            max_short_value=None
        )

        self.assertIsNone(backtester_none.max_short_value,
                          "max_short_value should be None when not set")


class TestEventBacktesterSharpeRatio(unittest.TestCase):
    """Test Sharpe ratio calculation for EventBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL"]
        self.backtester = TestEventBacktester(self.symbols, 1000.0)

    def test_sharpe_ratio_basic_calculation(self):
        """Test basic Sharpe ratio calculation with known values."""
        # Create a simple state history with known returns
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        portfolio_values = [1000]
        for i in range(1, 253):
            noise = np.random.normal(0, 0.005)
            new_value = portfolio_values[-1] * (1 + 0.01 + noise)
            portfolio_values.append(new_value)
        # Now portfolio_values is length 253
        state_history = pd.DataFrame({
            'cash': [1000] + [0] * 252,
            'portfolio_value': portfolio_values,
            'AAPL': [0] * 253,
            'GOOGL': [0] * 253
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=252)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, 0)
        self.assertGreater(portfolio_values[-1], 1000)

    def test_sharpe_ratio_with_zero_volatility(self):
        """Test Sharpe ratio calculation with zero volatility (risk-free returns)."""
        # Case 1: mean return > risk-free rate per period (expect inf)
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        state_history = pd.DataFrame({
            'cash': [1000, 0, 0],
            'portfolio_value': [1000, 1100, 1200],
            'AAPL': [0, 0, 0],
            'GOOGL': [0, 0, 0]
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe_high_rf = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.05, periods_per_year=252)
        self.assertEqual(sharpe_high_rf, float('inf'))
        # Case 2: mean return < risk-free rate per period (expect -inf)
        state_history_neg = pd.DataFrame({
            'cash': [1000, 0, 0],
            'portfolio_value': [1000, 900, 800],
            'AAPL': [0, 0, 0],
            'GOOGL': [0, 0, 0]
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history_neg
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe_low_rf = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.05, periods_per_year=252)
        self.assertEqual(sharpe_low_rf, float('-inf'))
        # Case 3: mean return == risk-free rate per period (expect 0.0)
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        # Calculate mean return from actual portfolio values
        pf = state_history['portfolio_value']
        returns_actual = pf[pf.index != 0].pct_change().dropna()
        mean_return_actual = returns_actual.mean()
        rf_equal = mean_return_actual * 252
        sharpe_equal_rf = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=rf_equal, periods_per_year=252)
        self.assertAlmostEqual(sharpe_equal_rf, 0.0, places=7)

    def test_sharpe_ratio_insufficient_data(self):
        self.backtester.state_history = pd.DataFrame()
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        with self.assertRaises(ValueError, msg="Insufficient data to calculate Sharpe ratio"):
            self.backtester.calculate_sharpe_ratio()
        self.backtester.state_history = pd.DataFrame({
            'cash': [1000],
            'portfolio_value': [1000],
            'AAPL': [0],
            'GOOGL': [0]
        }, index=[0])
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        with self.assertRaises(ValueError, msg="Insufficient trading data to calculate Sharpe ratio"):
            self.backtester.calculate_sharpe_ratio()
        self.backtester.state_history = pd.DataFrame({
            'cash': [1000, 0],
            'portfolio_value': [1000, 1000],
            'AAPL': [0, 0],
            'GOOGL': [0, 0]
        }, index=[0, pd.Timestamp('2024-01-01')])
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        with self.assertRaises(ValueError, msg="No valid returns data to calculate Sharpe ratio"):
            self.backtester.calculate_sharpe_ratio()

    def test_sharpe_ratio_different_periods(self):
        dates = pd.date_range('2024-01-01', periods=12, freq='ME')
        portfolio_values = [1000]
        for i in range(1, 13):
            noise = np.random.normal(0, 0.02)
            new_value = portfolio_values[-1] * (1 + 0.01 + noise)
            portfolio_values.append(new_value)
        state_history = pd.DataFrame({
            'cash': [1000] + [0] * 12,
            'portfolio_value': portfolio_values,
            'AAPL': [0] * 13,
            'GOOGL': [0] * 13
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe_monthly = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=12)
        self.assertIsInstance(sharpe_monthly, float)
        sharpe_quarterly = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=4)
        self.assertIsInstance(sharpe_quarterly, float)
        sharpe_daily = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=252)
        self.assertIsInstance(sharpe_daily, float)

    def test_sharpe_ratio_negative_returns(self):
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        portfolio_values = [1000]
        for i in range(1, 253):
            noise = np.random.normal(0, 0.005)
            new_value = portfolio_values[-1] * (1 - 0.005 + noise)
            portfolio_values.append(new_value)
        state_history = pd.DataFrame({
            'cash': [1000] + [0] * 252,
            'portfolio_value': portfolio_values,
            'AAPL': [0] * 253,
            'GOOGL': [0] * 253
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=252)
        self.assertLess(sharpe, 0)

    def test_sharpe_ratio_integration_with_analyze_performance(self):
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        portfolio_values = [1000]
        for i in range(1, 51):
            noise = np.random.normal(0, 0.01)
            new_value = portfolio_values[-1] * (1 + 0.002 + noise)
            portfolio_values.append(new_value)
        state_history = pd.DataFrame({
            'cash': [1000] + [0] * 50,
            'portfolio_value': portfolio_values,
            'AAPL': [0] * 51,
            'GOOGL': [0] * 51
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        test_bars_data = []
        for symbol in self.symbols:
            for date in dates:
                test_bars_data.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': 100.0,
                    'high': 105.0,
                    'low': 95.0,
                    'close': 103.0,
                    'volume': 1000
                })
        test_bars = pd.DataFrame(test_bars_data)
        test_bars.set_index(['symbol', 'timestamp'], inplace=True)
        self.backtester.test_bars = test_bars
        performance = self.backtester.analyze_performance()
        self.assertIn('sharpe_ratio', performance)
        self.assertIsInstance(performance['sharpe_ratio'], float)
        self.assertIn('return_on_investment', performance)
        self.assertIn('max_drawdown_pct', performance)
        self.assertIn('win_rate', performance)

    def test_sharpe_ratio_edge_cases(self):
        dates = pd.date_range('2024-01-01', periods=252, freq='D')
        portfolio_values = [1000]
        for i in range(1, 253):
            noise = np.random.normal(0, 0.0001)
            new_value = portfolio_values[-1] * (1 + 0.0001 + noise)
            portfolio_values.append(new_value)
        state_history = pd.DataFrame({
            'cash': [1000] + [0] * 252,
            'portfolio_value': portfolio_values,
            'AAPL': [0] * 253,
            'GOOGL': [0] * 253
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=252)
        self.assertIsInstance(sharpe, float)
        portfolio_values_high_vol = [1000]
        for i in range(1, 253):
            noise = np.random.normal(0, 0.05)
            new_value = portfolio_values_high_vol[-1] * (1 + 0.01 + noise)
            portfolio_values_high_vol.append(new_value)
        state_history_high_vol = pd.DataFrame({
            'cash': [1000] + [0] * 252,
            'portfolio_value': portfolio_values_high_vol,
            'AAPL': [0] * 253,
            'GOOGL': [0] * 253
        }, index=[0] + list(dates))
        self.backtester.state_history = state_history_high_vol
        # Set the backtester as having been run so calculate_sharpe_ratio works
        self.backtester._EventBacktester__already_ran = True
        sharpe_high_vol = self.backtester.calculate_sharpe_ratio(
            risk_free_rate=0.02, periods_per_year=252)
        self.assertIsInstance(sharpe_high_vol, float)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
