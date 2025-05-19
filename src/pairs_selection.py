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
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7
plt.rcParams["axes.grid"] = True


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
    start_date = datetime(2023, 1, 1)
    end_date = datetime.today() - timedelta(minutes=15)
    request_params = StockBarsRequest(
        symbol_or_symbols=utility_tickers,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date,
        adjustment="all",
    )
    refresh_bars = False
    if refresh_bars:
        bars = client.get_stock_bars(request_params).df  # get the bars
        save_data(bars, "utility_bars")  # save the bars
        bars = pd.read_pickle("data/utility_bars.pkl")  # load the bars
        # resample the bars. apply to each ticker
        bars = resample_multi_ticker_bars(bars)
        save_data(bars, "utility_bars_resampled")
        close_prices = bars["close"].unstack(level=0)  # get the close prices
        save_data(close_prices, "utility_close_prices")  # save the close prices

        # calculate the log returns
        returns = close_prices.apply(lambda x: np.log(x / x.shift(1))).iloc[1:]
        save_data(returns, "utility_returns")

    returns = pd.read_pickle("data/utility_returns.pkl")
    close_prices = pd.read_pickle("data/utility_close_prices.pkl")
    plot_price_data(close_prices, "Utility Price Data")
    plt_show(prefix="utility_price_data")

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
    plt_show(prefix="pca_explained_variance")

    # most important component
    first_component = pca.components_[0, :]
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(4):
        component = pca.components_[i, :]
        axes[i].bar(utility_tickers, component)
        axes[i].set_title(f"Weightings of each asset in component {i+1}")
        axes[i].set_xlabel("Assets")
        axes[i].set_ylabel("Weighting")
        axes[i].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt_show(prefix="pca_components")

    # calculate the cointegration results
    cointegration_results = pd.DataFrame(index=utility_tickers, columns=utility_tickers)
    for ticker1, ticker2 in combinations(utility_tickers, 2):
        if cointegration_results.loc[ticker2, ticker1] is not np.nan:
            cointegration_results.loc[ticker1, ticker2] = cointegration_results.loc[
                ticker2, ticker1
            ]
            continue
        normalized_prices = close_prices[[ticker1, ticker2]].apply(
            lambda x: (x / x.mean())
        )

        coint_result = engle_granger(
            normalized_prices.iloc[:, 0],
            normalized_prices.iloc[:, 1],
            trend="c",
            lags=0,
        )
        coint_pvalue = coint_result.pvalue
        coint_vector = coint_result.cointegrating_vector[:2]

        spread = normalized_prices.iloc[:, 0] - normalized_prices.iloc[:, 1]

        spread_pvalue = adfuller(spread, maxlag=0)[1]
        if spread_pvalue < 0.05:
            print(coint_result)
            print(
                f"The ADF test for {ticker1}-{ticker2} p-value is {spread_pvalue}, so it is {'' if spread_pvalue < 0.05 else 'not '}stationary."
            )

        cointegration_results.loc[ticker1, ticker2] = spread_pvalue
        cointegration_results.loc[ticker2, ticker1] = spread_pvalue

    # print the spreads
    print(cointegration_results)

    # show the cointegration results as a heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        cointegration_results.values.astype(np.float32),
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
    )
    plt.grid(False)
    plt.title("Cointegration Results")
    plt.xlabel("Ticker")
    plt.ylabel("Ticker")
    # add ticker labels
    plt.xticks(np.arange(len(utility_tickers)) + 0.5, utility_tickers, rotation=30)
    plt.yticks(np.arange(len(utility_tickers)) + 0.5, utility_tickers, rotation=0)
    plt_show(prefix="cointegration_results")

    # find the pair with the lowest p-value
    lowest_pvalue = cointegration_results.min().min()
    print(f"The lowest p-value is {lowest_pvalue}")

    lowest_pairs = list(cointegration_results.stack().sort_values().head(20).index)
    # remove duplicate pairs; order of each pair is not important
    lowest_pairs = list(set([tuple(sorted(pair)) for pair in lowest_pairs]))
    for lowest_pair in lowest_pairs:
        print(
            f"The pair with the lowest p-value is {lowest_pair} with p-value {lowest_pvalue}"
        )

    # plot the spread of the lowest pair
    for lowest_pair in lowest_pairs:
        ticker1, ticker2 = lowest_pair
        prices = close_prices[[ticker1, ticker2]]  # .iloc[-1000:]
        spread = prices.iloc[:, 0] - prices.iloc[:, 1]

        # normalized_prices = close_prices[[ticker1, ticker2]].apply(
        #    lambda x: (x / x.mean())
        # )
        # normalized_spread = normalized_prices.iloc[:, 0] - normalized_prices.iloc[:, 1]

        # make bands
        rolling_window = 8  # This is used for determining how many days ahead to use to calculate the rolling mean
        std_multiplier = 1
        rolling_mean = (spread.rolling(window=rolling_window).mean()).dropna()
        rolling_std = (spread.rolling(window=rolling_window).std()).dropna()
        upper_band = rolling_mean + (rolling_std * std_multiplier)
        lower_band = rolling_mean - (rolling_std * std_multiplier)

        # fix the bands and rollings to be the same length as the spread
        rolling_mean = rolling_mean.reindex(spread.index)
        upper_band = upper_band.reindex(spread.index)
        lower_band = lower_band.reindex(spread.index)

        # make a series where the value is 1 if the spread is above the upper band and -1 if it is below the lower band
        position = pd.Series(
            np.where(spread > upper_band, 1, np.where(spread < lower_band, -1, 0)),
            index=spread.index,
        )

        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(15, 12), height_ratios=[2, 0.5, 1]
        )

        # Plot spread in the largest subplot
        spread.plot(ax=ax1, title=f"Spread of {ticker1} and {ticker2}")
        rolling_mean.plot(
            ax=ax1, label="Rolling Mean", color="red", linewidth=0.5, alpha=0.8
        )
        upper_band.plot(
            ax=ax1, label="Upper Band", color="green", linewidth=0.5, alpha=0.8
        )
        lower_band.plot(
            ax=ax1, label="Lower Band", color="green", linewidth=0.5, alpha=0.8
        )
        ax1.set_ylabel("Spread ($)")
        ax1.set_xlabel("")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend()
        ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Plot spread direction in the middle subplot
        position.plot(ax=ax2, title=f"Signal of {ticker1} and {ticker2}")
        ax2.set_ylabel("Signal")
        ax2.set_xlabel("")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # Plot individual prices in the bottom subplot
        prices.plot(ax=ax3, title=f"Price of {ticker1} and {ticker2} ($)")
        ax3.set_ylabel("Price ($)")
        ax3.set_xlabel("Date")
        ax3.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt_show(prefix=f"spread_and_prices_{ticker1}_{ticker2}")
