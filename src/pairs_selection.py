import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from util import save_data, plt_show, plot_price_data
from sklearn.decomposition import PCA
from arch.unitroot.cointegration import engle_granger
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")


def resample_multi_ticker_bars(
    bars: pd.DataFrame, time_frame: timedelta = timedelta(hours=1)
) -> pd.DataFrame:
    """Resample the bars to the given time frame and fill the missing values.
    For OHLC, use the last value in the time frame to fill forward.
    For volume and trade count, fill with 0.
    THIS IS FOR MULTIPLE TICKERS. Use groupby to apply to multiple tickers.
    """
    # Create a list to store resampled data for each ticker
    resampled_data = []

    # Get unique tickers
    tickers = bars.index.get_level_values("symbol").unique()

    latest_start = None
    earliest_end = None

    # Process each ticker
    for ticker in tickers:
        # Get data for this ticker
        ticker_data = bars.loc[ticker]
        # start and end of the ticker data
        start = ticker_data.index.get_level_values("timestamp").min()
        end = ticker_data.index.get_level_values("timestamp").max()
        if latest_start is None or start > latest_start:
            latest_start = start
        if earliest_end is None or end < earliest_end:
            earliest_end = end

        # Resample the data
        resampled_ticker = resample_bars(ticker_data, time_frame)
        # Add the ticker back to the index
        resampled_ticker["symbol"] = ticker
        resampled_data.append(resampled_ticker)

    # trim the data to the latest start and earliest end
    for i in range(len(resampled_data)):
        data = resampled_data[i]
        resampled_data[i] = data.loc[latest_start:earliest_end]

    # Combine all resampled data
    result = pd.concat(resampled_data)
    # Set the multi-index
    result.set_index(["symbol", result.index], inplace=True)

    return result


def resample_bars(
    bars: pd.DataFrame, time_frame: timedelta = timedelta(hours=1)
) -> pd.DataFrame:
    """Resample the bars to the given time frame and fill the missing values.
    For OHLC, use the last value in the time frame to fill forward.
    For volume and trade count, fill with 0.
    THIS IS FOR ONE TICKER. Use groupby to apply to multiple tickers.
    """
    # resample the bars
    part1 = bars[["open", "high", "low", "close", "vwap"]]
    part2 = bars[["volume", "trade_count"]]
    part1 = part1.resample(time_frame).last().ffill()
    part2 = part2.resample(time_frame).asfreq().fillna(0)
    bars = pd.concat([part1, part2], axis=1)
    return bars


if __name__ == "__main__":
    utility_tickers = [
        "NEE",
        "EXC",
        "D",
        "PCG",
        "XEL",
        "ED",
        "WEC",
        "DTE",
        "PPL",
        "AEE",
        "CNP",
        "FE",
        "CMS",
        "EIX",
        "ETR",
        "EVRG",
        "LNT",
        "PNW",
        "IDA",
    ]
    client = StockHistoricalDataClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET
    )

    # Get the utility bars
    start_date = datetime(2023, 5, 1)
    end_date = datetime.today()
    request_params = StockBarsRequest(
        symbol_or_symbols=utility_tickers,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date,
        adjustment="all",
    )
    """
    bars = client.get_stock_bars(request_params).df # get the bars
    save_data(bars, 'utility_bars.csv') # save the bars
    bars = pd.read_pickle("data/utility_bars.pkl")  # load the bars
    # resample the bars. apply to each ticker
    bars = resample_multi_ticker_bars(bars)
    save_data(bars, "utility_bars_resampled")
    close_prices = bars["close"].unstack(level=0)  # get the close prices
    save_data(close_prices, "utility_close_prices")  # save the close prices

    # calculate the log returns
    returns = close_prices.apply(lambda x: np.log(x / x.shift(1))).iloc[1:]
    save_data(returns, "utility_returns")
    """
    returns = pd.read_pickle("data/utility_returns.pkl")
    close_prices = pd.read_pickle("data/utility_close_prices.pkl")
    plot_price_data(
        close_prices, "Utility Price Data"
    )  # [["NEE", "EXC", "D", "PCG", "XEL"]])
    plt_show()

    # PCA
    pca = PCA()
    pca.fit(returns)
    components = [str(x + 1) for x in range(pca.n_components_)]
    explained_variance_pct = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(15, 10))
    plt.bar(components, explained_variance_pct)
    plt.title("Ratio of Explained Variance")
    plt.xlabel("Principle Component #")
    plt.ylabel("%")
    plt_show()

    first_component = pca.components_[0, :]
    highest = utility_tickers[abs(first_component).argmax()]
    lowest = utility_tickers[abs(first_component).argmin()]
    print(
        f"The highest-absolute-weighing asset: {highest}\nThe lowest-absolute-weighing asset: {lowest}"
    )

    plt.figure(figsize=(15, 10))
    plt.bar(utility_tickers, first_component)
    plt.title("Weightings of each asset in the first component")
    plt.xlabel("Assets")
    plt.ylabel("Weighting")
    plt.xticks(rotation=30)
    plt_show()

    log_prices = np.log(close_prices[[highest, lowest]])
    coint_result = engle_granger(
        log_prices.iloc[:, 0], log_prices.iloc[:, 1], trend="c", lags=0
    )
    print(coint_result)
    coint_vector = coint_result.cointegrating_vector[:2]
    spread = log_prices @ coint_vector

    pvalue = adfuller(spread, maxlag=0)[1]
    print(
        f"The ADF test p-value is {pvalue}, so it is {'' if pvalue < 0.05 else 'not '}stationary."
    )

    spread.plot(figsize=(15, 10), title=f"Spread of {highest} and {lowest}")
    plt.ylabel("Spread")

    plt_show()
