import os
import pandas as pd
import json
from dotenv import load_dotenv
from datetime import datetime, date
import sys
from .logger import get_logger

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Create a logger for the utilities module
logger = get_logger("utilities")

load_dotenv()


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


def dash(text: str | None = None) -> str:
    """Return a dash line with text centered in the middle."""
    terminal_width = os.get_terminal_size().columns
    if text is None:
        return "-" * terminal_width
    else:
        return "- " + text + " -" + ("-" * (terminal_width - len(text) - 4))


def save_json(data: dict, filename: str, data_dir: str = getenv("DATA_DIR")) -> None:
    """Save the data to a JSON file.

    Args:
        data (dict): The data to save.
        filename (str): The filename to save the data to.
        data_dir (str, optional): The directory to save the data to. Defaults to os.getenv("DATA_DIR").
    """
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


def load_json(filename: str, data_dir: str = getenv("DATA_DIR")) -> dict | None:
    """Load the data from a JSON file.

    Args:
        filename (str): The filename to load the data from.
        data_dir (str, optional): The directory to load the data from. Defaults to os.getenv("DATA_DIR").

    Returns:
        dict | None: The data from the JSON file.
    """
    path = os.path.join(data_dir, filename)
    with open(path + ".json", "r") as f:
        return json.load(f)


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
