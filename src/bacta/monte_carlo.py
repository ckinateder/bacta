"""
Monte Carlo analysis of the backtest. This is not working yet.
"""

import numpy as np
import pandas as pd
from typing import List


def bootstrap_ohlcv_series_multi(
    data: pd.DataFrame, n: int = 10, noise_std: float = 0.0, seed: int = None
) -> List[pd.DataFrame]:
    """
    Generate n synthetic OHLCV time series for each symbol using bootstrapped log returns and optional noise.
    Returns a list of synthetic OHLCV dataframes.

    Args:
        data (pd.DataFrame): Multi-index OHLCV data (index: symbol, timestamp), with columns: open, high, low, close, volume
        n (int): Number of synthetic series per symbol
        noise_std (float): Std deviation of Gaussian noise to add to log returns
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Multi-index OHLCV data (index: symbol, timestamp), with columns: open, high, low, close, volume
    """
    if seed is not None:
        np.random.seed(seed)

    symbols = data.index.get_level_values(0).unique()
    timestamps = data.index.get_level_values(1).unique()
    output = []

    for _ in range(n):
        simulated = []

        for symbol in symbols:
            df_symbol = data.loc[symbol]
            close_prices = df_symbol["close"]
            log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

            # Bootstrap with replacement
            bootstrapped_returns = np.random.choice(
                log_returns.values, size=len(log_returns), replace=True
            )

            # Perturb with Gaussian noise (if any)
            if noise_std > 0:
                bootstrapped_returns += np.random.normal(
                    0, noise_std, size=len(bootstrapped_returns)
                )

            # Reconstruct prices from bootstrapped log returns
            first_price = close_prices.iloc[0]
            bootstrapped_prices = (
                np.exp(np.insert(bootstrapped_returns, 0, 0)).cumprod() * first_price
            )

            # Rebuild OHLCV dataframe
            df_boot = df_symbol.copy()
            df_boot["close"] = bootstrapped_prices
            df_boot["open"] = (
                df_boot["close"].shift(1).fillna(df_boot["close"])
            )  # naive open
            df_boot["high"] = np.maximum(df_boot["open"], df_boot["close"]) * (
                1 + 0.01
            )  # 1% range up
            df_boot["low"] = np.minimum(
                # 1% range down
                df_boot["open"],
                df_boot["close"],
            ) * (1 - 0.01)

            simulated.append(df_boot)

        df_simulated = pd.concat(simulated, keys=symbols, names=["symbol", "timestamp"])
        output.append(df_simulated)

    return output
