"""
Interactive Backtester Dashboard - Streamlit App

Run this with: streamlit run streamlit_app.py

This app demonstrates the interactive dashboard functionality of the backtester.
"""

import streamlit as st
from datetime import datetime, timedelta
import logging

from alpaca.data.timeframe import TimeFrame
import pandas as pd
from talib import ATR, EMA

from bars import (
    download_bars,
    separate_bars_by_symbol,
    split_multi_index_bars_train_test,
)

from __init__ import *
from src import *
from src.backtester import EventBacktester, Order, Position
from src.utilities import dash, get_logger, set_log_level

set_log_level(logging.INFO)


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
                return Order(symbol, Position.SHORT, close_prices[symbol], 1)
            elif close_prices[symbol] < self.lower_bands[symbol][index]:
                return Order(symbol, Position.LONG, close_prices[symbol], 1)


# Streamlit app configuration
st.set_page_config(
    page_title="Interactive Backtester Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Interactive Backtester Dashboard")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("Strategy Configuration")

# Symbol selection
symbol_options = ["DUK", "NRG", "NEE", "EXC", "D",
                  "PCG", "XEL", "ED", "WEC", "DTE", "PPL", "AEE"]
selected_symbols = st.sidebar.multiselect(
    "Select Symbols",
    symbol_options,
    default=["DUK", "NRG"],
    max_selections=4
)

# Strategy parameters
cash = st.sidebar.number_input(
    "Initial Cash", min_value=100, max_value=10000, value=2000, step=100)
allow_short = st.sidebar.checkbox("Allow Short Positions", value=True)
market_hours_only = st.sidebar.checkbox("Market Hours Only", value=True)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2024, 1, 1).date(),
        max_value=datetime.now().date()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=(datetime.now() - timedelta(days=1)).date(),
        max_value=datetime.now().date()
    )

# Keltner Channel parameters
st.sidebar.subheader("Keltner Channel Parameters")
keltner_period = st.sidebar.slider(
    "Period", min_value=5, max_value=50, value=21, step=1)
atr_multiplier = st.sidebar.slider(
    "ATR Multiplier", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

# Run backtest button
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    if not selected_symbols:
        st.error("Please select at least one symbol.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        with st.spinner("Running backtest..."):
            try:
                # Download data
                bars = download_bars(
                    selected_symbols,
                    start_date=datetime.combine(
                        start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    timeframe=TimeFrame.Hour
                )

                # Split data
                train_bars, test_bars = split_multi_index_bars_train_test(
                    bars, split_ratio=0.9)

                # Create and configure backtester
                backtester = KeltnerChannelBacktester(
                    selected_symbols,
                    cash=cash,
                    allow_short=allow_short,
                    allow_overdraft=False,
                    min_trade_value=1,
                    market_hours_only=market_hours_only
                )

                # Update Keltner Channel parameters
                backtester.keltner_channel_period = keltner_period

                # Run backtest
                backtester.load_train_bars(train_bars)
                backtester.run_backtest(test_bars)

                # Store backtester in session state
                st.session_state.backtester = backtester
                st.session_state.performance = backtester.analyze_performance()

                st.success("Backtest completed successfully!")

            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

# Display results if backtest has been run
if 'backtester' in st.session_state and 'performance' in st.session_state:
    backtester = st.session_state.backtester
    performance = st.session_state.performance

    # Performance summary
    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{performance['return_on_investment']:.2%}")
    with col2:
        st.metric("Max Drawdown",
                  f"{performance['max_drawdown_percentage']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.3f}")
    with col4:
        st.metric("Win Rate", f"{performance['win_rate']:.1%}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Number of Trades", performance['number_of_orders'])
    with col2:
        st.metric("Winning Trades", performance['number_of_winning_trades'])
    with col3:
        st.metric("Losing Trades", performance['number_of_losing_trades'])
    with col4:
        st.metric("Buy & Hold Return",
                  f"{performance['buy_and_hold_return']:.2%}")

    st.markdown("---")

    # Interactive dashboard
    st.subheader("📈 Interactive Dashboard")
    st.write(
        "Use the tabs below to explore different aspects of the backtest results.")

    # Display the interactive dashboard
    backtester.plot_interactive_dashboard(
        "Keltner Channel Strategy Performance")

    # Additional details
    with st.expander("📋 Detailed Performance Metrics"):
        st.dataframe(performance.to_frame().T)

    with st.expander("📝 Trade History"):
        trade_history = backtester.get_history()
        if not trade_history.empty:
            st.dataframe(trade_history)
        else:
            st.write("No trades were executed during this period.")

else:
    st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to get started.")

    # Show example configuration
    st.subheader("💡 Example Configuration")
    st.markdown("""
    **Quick Start:**
    1. Select 2-3 symbols (e.g., DUK, NRG)
    2. Set initial cash to $2000
    3. Keep default Keltner Channel parameters (Period: 21, ATR Multiplier: 2.0)
    4. Click "Run Backtest"
    
    **Strategy Description:**
    The Keltner Channel strategy uses exponential moving averages and Average True Range (ATR) 
    to identify potential entry and exit points:
    - **Long Position**: When price closes below the lower Keltner Channel
    - **Short Position**: When price closes above the upper Keltner Channel
    
    This is a mean-reversion strategy that assumes prices will return to the middle band.
    """)

# Footer
st.markdown("---")
st.markdown(
    "*Built with Streamlit and Plotly for interactive financial analysis*")
