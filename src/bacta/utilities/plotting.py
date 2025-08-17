import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PLT_CNTR = 0
DEFAULT_FIGSIZE = (15, 10)


def plot_price_data(
    df: pd.DataFrame, title: str = "", figsize: tuple = DEFAULT_FIGSIZE
):
    """
    Plot the price data.
    Args:
        df (pd.DataFrame): DataFrame to plot. Index should be datetime or sortable.
        title (str, optional): Title for the plot.
        figsize (tuple, optional): Figure size.
    """
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


def plot_split_dataframe(
    df: pd.DataFrame,
    split_index: int,
    title: str = "",
    figsize: tuple = DEFAULT_FIGSIZE,
    split_label_1: str = "Train",
    split_label_2: str = "Test",
):
    """
    Plot a DataFrame split into two parts with a vertical dashed line at the split point.

    Args:
        df (pd.DataFrame): DataFrame to plot. Index should be datetime or sortable.
        split_index (int): The index (row number) at which to split the DataFrame.
        title (str, optional): Title for the plot.
        figsize (tuple, optional): Figure size.
        split_label_1 (str, optional): Label for the first part.
        split_label_2 (str, optional): Label for the second part.
    """
    plt.figure(figsize=figsize)

    # Plot each column
    for i, column in enumerate(df.columns):
        plt.plot(
            df.index[:split_index],
            df[column].iloc[:split_index],
            color="blue",
            label=f"{column} ({split_label_1})",
        )
        plt.plot(
            df.index[split_index:],
            df[column].iloc[split_index:],
            color="red",
            label=f"{column} ({split_label_2})",
        )

    # Draw vertical dashed line at split
    if 0 < split_index < len(df):
        split_x = df.index[split_index]
        plt.axvline(
            x=split_x, color="black", linestyle="dashed", linewidth=1.5, alpha=0.7
        )
        plt.text(
            split_x,
            plt.ylim()[1],
            " Split",
            color="black",
            va="top",
            ha="left",
            fontsize=10,
            alpha=0.7,
        )

    plt.xticks(
        df.index[:: max(1, len(df) // 20)],
        [
            x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
            for x in df.index[:: max(1, len(df) // 20)]
        ],
        rotation=45,
    )
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title if title else "DataFrame Split Plot")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper left", ncol=max(1, len(df.columns) // 4 + 1))
    plt.tight_layout()


def plt_show(
    prefix: str = "plt",
    folder: str = "plots",
    plt_cntr: bool = False,
    show_plot: bool = True,
):
    """
    Show the plot and save it to the given folder.

    Args:
        prefix (str, optional): Prefix for the plot file name. Defaults to "plt".
        folder (str, optional): Folder to save the plot. Defaults to "plots".
        plt_cntr (bool, optional): Whether to add a counter to the plot file name. Defaults to False.
    """
    global PLT_CNTR
    if show_plot:
        plt.show()
    PLT_CNTR += 1
    prefix = prefix.replace(" ", "_").replace("&", "and")
    plt.savefig(
        os.path.join(folder, f"{prefix}{'_'+PLT_CNTR if plt_cntr else ''}.png"), dpi=300
    )
    plt.close()
