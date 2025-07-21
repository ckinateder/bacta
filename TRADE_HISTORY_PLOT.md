# Trade History Plot Function

## Overview

The `plot_trade_history()` function is a new addition to the `EventBacktester` class that visualizes the trading activity of your backtester. It plots the price history of each symbol with markers showing where buy and sell orders were executed.

## Features

- **Price History**: Shows the closing price over time as a blue line
- **Trade Markers**: 
  - Green triangles pointing up (^) for buy orders
  - Red triangles pointing down (v) for sell orders
- **Quantity Annotations**: Shows the quantity traded at each marker
- **Summary Statistics**: Displays total trades, volume, and average price
- **Multi-symbol Support**: Creates separate subplots for each symbol

## Usage

```python
from src.backtester import SMABacktester  # or any other backtester

# Create and run your backtester
backtester = SMABacktester(active_symbols=['AAPL'], cash=10000)
backtester.load_train_bars(train_bars)
backtester.run(test_bars, ignore_market_open=True)

# Generate the trade history plot
fig = backtester.plot_trade_history(
    figsize=(20, 12),      # Figure size
    save_plot=True,        # Save to file
    show_plot=True         # Display the plot
)
```

## Parameters

- `figsize` (tuple): Figure size for the plot (default: (20, 12))
- `save_plot` (bool): Whether to save the plot to file (default: True)
- `show_plot` (bool): Whether to display the plot (default: True)

## Output

The function returns a matplotlib Figure object and optionally:
- Saves the plot as `trade_history.png` in the `plots/` directory
- Displays the plot if `show_plot=True`

## Example Output

The plot will show:
- **Blue line**: Price history over time
- **Green triangles (^)**: Buy orders with quantity annotations
- **Red triangles (v)**: Sell orders with quantity annotations
- **Summary box**: Total trades, volume, and average price
- **Grid**: For easier price reading

## Requirements

- The backtester must have been run with `test_bars` data
- Order history must be available (trades must have been executed)
- The `test_bars` attribute must contain price data

## Error Handling

The function includes error handling for:
- Empty order history
- Missing test bars data
- Missing symbol data

If any of these conditions are met, the function will log a warning and return `None`.

## Example Script

See `example_trade_history_plot.py` for a complete working example that demonstrates how to use this function with sample data. 