import os
import pandas as pd
import json
from names import SEC_TICKERS_FILENAME
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

PLT_CNTR = 0


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


def save_data(df: pd.DataFrame, filename: str, data_dir: str = os.getenv("DATA_DIR")):
    """Save the data to a CSV and pickle file."""
    path = os.path.join(data_dir, filename)
    df.to_csv(path + ".csv")
    df.to_pickle(path + ".pkl")


if __name__ == "__main__":
    sec_tickers = load_sec_tickers()
