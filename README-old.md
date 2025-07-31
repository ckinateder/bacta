# Arby's

Prototyping statistical arbitrage.

![symbols](img/performance_analysis.png)

## Stack
- Python
## Usage

## Development

### Docker Setup

Run the following command to build the docker image:

```bash
docker build -t arbys . 
```

Run the following command to run the docker image:

```bash
docker run -it --rm  -v $(pwd):/app -w /app arbys
```
**Note that you may have issues showing `matplotlib` images when inside the docker container; a workaround is to [just save the generated plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html) instead.**

### Tests

Run all tests with:
```bash
python -m unittest discover test
```

### Backtesting

Central to this project is the backtester module. The `EventBacktester` is designed for the user to make their own strategies, by inheriting from `EventBacktester` and implementing just one or more methods. 

#### Methods

The `EventBacktester` class has the following initialization parameters:

- `active_symbols`: The symbols to trade.
- `cash`: The initial cash balance.
- `allow_short`: Whether to allow short positions.
- `allow_overdraft`: Whether to allow overdraft in the bank.
- `min_trade_value`: The minimum dollar value of a trade. If the order value is less than this, the order will be skipped.
- `market_hours_only`: Whether to only place orders during market hours.

The `EventBacktester` class has the following predefined user-facing methods:

- `load_train_bars(bars: pd.DataFrame)`: Load the training bars for the backtest. This method then calls the `precompute_step` method.
- `run(test_bars: pd.DataFrame, ignore_market_open: bool = False, close_positions: bool = True)`: Run the backtest.
- `plot_equity_curve(title: str = "Equity Curve", save_plot: bool = True, show_plot: bool = False)`: Plot the equity curve.
- `plot_performance_analysis(title: str = "Performance Analysis", save_plot: bool = True, show_plot: bool = False)`: Plot the performance analysis.
- `plot_trade_history(title: str = "Trade History", save_plot: bool = True, show_plot: bool = False)`: Plot the trade history.
- `analyze_performance()`: Analyze the performance of the backtest.

The user must implement the following methods:

- `generate_order(bar: pd.DataFrame, index: pd.Timestamp) -> Order`: Make a decision based on the prices. This method is called by `run` and returns an `Order` object.
- `update_step(bars: pd.DataFrame, index: pd.Timestamp)`: Update the state of the backtester. This method is called by `run`.
- (optional) `precompute_step(bars: pd.DataFrame)`: Preload the indicators for the backtest. This method is called by `load_train_bars`.

In a typical workflow, the user will implement the `generate_order` and `update_step` methods, and optionally the `precompute_step` method. The user will first call the `load_train_bars` method to load the training bars into the backtester. This method then calls the `precompute_step` method, which the user must implement. The user can then call the `run` method to run the backtest. This method will call the `update_step` method for each bar in the test bars, and then call the `generate_order` method for each bar in the test bars.
