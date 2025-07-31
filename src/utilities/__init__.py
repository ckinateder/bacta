import os
import pandas as pd
from dotenv import load_dotenv
import sys
from .logger import get_logger
import pytz
import holidays
from datetime import datetime, time

# global variables
nyse_holidays = holidays.NYSE()  # this is a dict-like object
eastern = pytz.timezone("US/Eastern")  # this is a timezone object

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Create a logger for the utilities module
logger = get_logger("utilities")

load_dotenv()


def dash(text: str | None = None) -> str:
    """Return a dash line with text centered in the middle."""
    terminal_width = os.get_terminal_size().columns
    if text is None:
        return "-" * terminal_width
    else:
        return "- " + text + " -" + ("-" * (terminal_width - len(text) - 4))


def getenv(key: str) -> str | None:
    """Get the environment variable.

    Args:
        key (str): The key to get the environment variable for.

    Returns:
        str: The environment variable.
    """
    if key not in os.environ:
        raise ValueError(
            f"Environment variable {key} not found. Please set it in the .env file.")
    return os.getenv(key)


def save_dataframe(
    df: pd.DataFrame, filename: str, data_dir: str = getenv("DATA_DIR")
) -> None:
    """Save the data to a CSV and pickle file.

    Args:
        df (pd.DataFrame): The dataframe to save.
        filename (str): The filename to save the dataframe to.
        data_dir (str, optional): The directory to save the dataframe to. Defaults to os.getenv("DATA_DIR").

    Returns:
        None
    """
    path = os.path.join(data_dir, filename)
    logger.debug(f"Saving {path + '.csv'}")
    df.to_csv(path + ".csv")
    logger.debug(f"Saving {path + '.pkl'}")
    df.to_pickle(path + ".pkl")


def load_dataframe(
    filename: str, data_dir: str = getenv("DATA_DIR")
) -> pd.DataFrame:
    """Load the data from a CSV and pickle file."""
    path = os.path.join(data_dir, filename)
    if os.path.exists(path + ".pkl"):
        logger.debug(f"Loading {path + '.pkl'}")
        return pd.read_pickle(path + ".pkl")
    elif os.path.exists(path + ".csv"):
        logger.debug(f"Loading {path + '.csv'}")
        return pd.read_csv(path + ".csv")
    logger.error(f"File {path} not found.")
    raise FileNotFoundError(f"File {path} not found.")


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
