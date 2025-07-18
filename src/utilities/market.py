import os
import json
from datetime import datetime, time
import pandas as pd
import yfinance as yf
import holidays
import pytz
import warnings

try:
    from utilities import getenv
except ImportError:
    warnings.warn(
        "Could not import utilities from current directory. Using utilities from src.utilities (this is likely a jupyter notebook)")
    from src.utilities import getenv


# global variables
nyse_holidays = holidays.NYSE()  # this is a dict-like object
eastern = pytz.timezone("US/Eastern")  # this is a timezone object

DATA_DIR = getenv("DATA_DIR")

# names
SEC_TICKERS_FILENAME = "sec_tickers.json"


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
