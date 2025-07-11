import os
import pandas as pd
import json
from __init__ import SEC_TICKERS_FILENAME
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, time
import yfinance as yf
import pytz
import holidays

load_dotenv()

PLT_CNTR = 0
nyse_holidays = holidays.NYSE()  # this is a dict-like object
eastern = pytz.timezone("US/Eastern")


def dash(text: str = None):
    """Print a dash line with text centered in the middle."""
    terminal_width = os.get_terminal_size().columns
    if text is None:
        return "-" * terminal_width
    else:
        return "- " + text + " -" + ("-" * (terminal_width - len(text) - 4))


def plt_show(prefix: str = "plt", folder: str = "plots", plt_cntr: bool = False):
    global PLT_CNTR
    plt.show()
    PLT_CNTR += 1
    plt.savefig(
        os.path.join(folder, f"{prefix}{'_'+PLT_CNTR if plt_cntr else ''}.png"), dpi=300
    )
    plt.close()


def plot_price_data(df: pd.DataFrame, title: str = None, figsize: tuple = (15, 10)):
    """Plot the price data."""
    plt.figure(figsize=figsize)
    NUM_COLORS = len(df.columns)
    COLORS = sns.color_palette("husl", NUM_COLORS)

    # also plot the eps data on its own y axis
    plt.xticks(
        df.index[:: len(df) // 20],
        df.index[:: len(df) // 20].strftime("%Y-%m-%d"),
        rotation=45,
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    for i, column in enumerate(df.columns):
        plt.plot(df[column], color=COLORS[i], label=column)
    # legend, columns
    plt.legend(df.columns, loc="upper left", ncol=len(df.columns) // 4 + 1)


def load_sec_tickers(data_dir: str = os.getenv("DATA_DIR")) -> pd.DataFrame:
    """Load the CIK table downloaded from https://www.kaggle.com/datasets/svendaj/sec-edgar-cik-ticker-exchange.

    Args:
        data_dir (str, optional): The directory to save the CIK table. Defaults to os.getenv("DATA_DIR").

    Returns:
        pd.DataFrame: The CIK table. Index is cik, columns are name, ticker, and exchange.
    """
    path = os.path.join(data_dir, SEC_TICKERS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")

    # Load the JSON file
    with open(path, "r") as f:
        data = json.load(f)

    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(data=data["data"], columns=data["fields"])
    df.set_index("cik", inplace=True)
    return df


def save_dataframe(
    df: pd.DataFrame, filename: str, data_dir: str = os.getenv("DATA_DIR")
):
    """Save the data to a CSV and pickle file."""
    path = os.path.join(data_dir, filename)
    df.to_csv(path + ".csv")
    df.to_pickle(path + ".pkl")


def load_dataframe(
    filename: str, data_dir: str = os.getenv("DATA_DIR")
) -> pd.DataFrame:
    """Load the data from a CSV and pickle file."""
    path = os.path.join(data_dir, filename)
    if os.path.exists(path + ".pkl"):
        print(f"Loading {path + '.pkl'}")
        return pd.read_pickle(path + ".pkl")
    elif os.path.exists(path + ".csv"):
        print(f"Loading {path + '.csv'}")
        return pd.read_csv(path + ".csv")
    raise FileNotFoundError(f"File {path} not found.")


def save_json(data: dict, filename: str, data_dir: str = os.getenv("DATA_DIR")):
    """Save the data to a JSON file."""
    path = os.path.join(data_dir, filename)
    with open(path + ".json", "w") as f:
        json.dump(
            data,
            f,
            indent=4,
            default=lambda o: (
                o.__str__() if isinstance(o, datetime) or isinstance(o, date) else None
            ),
        )


def load_json(filename: str, data_dir: str = os.getenv("DATA_DIR")) -> dict:
    """Load the data from a JSON file."""
    path = os.path.join(data_dir, filename)
    with open(path + ".json", "r") as f:
        return json.load(f)


def get_earnings_date(ticker: str) -> datetime.date:
    """Get the next earnings date for a given ticker."""
    stock = yf.Ticker(ticker)

    # Fetch the calendar data, which includes upcoming events like the earnings date
    try:
        calendar = stock.calendar
        earnings_date = calendar["Earnings Date"][0]
        return earnings_date
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def is_market_open(dt: datetime) -> bool:
    """
    Check if a timezone-aware datetime is during US stock market open hours (NYSE),
    including holiday and early close checks.

    Parameters:
    - dt: datetime.datetime (must be timezone-aware)

    Returns:
    - bool: True if the market is open at that time, False otherwise
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Datetime must be timezone-aware")

    # Convert to Eastern Time
    dt_eastern = dt.astimezone(eastern)

    # if the day is a weekend or holiday, return False
    if dt_eastern.weekday() >= 5 or dt_eastern.date() in nyse_holidays:
        return False

    # if the time is before 9:30am or after 4:00pm, return False
    if dt_eastern.time() < time(9, 30) or dt_eastern.time() > time(16, 0):
        return False

    return True
