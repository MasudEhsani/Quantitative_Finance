"""
Load raw price data and turn it into returns suitable for PCA.

The expected input is a CSV with one row per trading day and one column per
ticker, where the final column is the level of the benchmark index itself
(by default ``SPX``). This matches the layout of
``data/spx_holdings_and_spx_closeprice.csv``.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "spx_holdings_and_spx_closeprice.csv"


def load_asset_prices(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load daily close prices for the stock universe plus the index level.

    Rows with any missing values are dropped, which mainly removes the first
    few days of trading before some late-adding constituents have quotes.

    Arguments:
        path -- path to the prices CSV (dates in the first column)

    Return:
        prices -- pandas.DataFrame indexed by date, one column per ticker
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
    prices = prices.sort_index()
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute daily returns from a price panel.

    Arguments:
        prices -- pandas.DataFrame of prices, one column per ticker
        method -- 'simple' for percentage returns, 'log' for log returns

    Return:
        returns -- pandas.DataFrame of returns, first (NaN) row dropped
    """
    if method == "log":
        returns = np.log(prices) - np.log(prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change(periods=1)
    else:
        raise ValueError("method must be 'simple' or 'log', got %r" % method)
    return returns.iloc[1:, :]


def center_returns(r_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize returns: subtract the mean and divide by the std of each column.

    Standardizing puts every asset on the same scale before PCA, so that
    the principal components reflect co-movement (correlation) rather than
    being dominated by the most volatile names.

    Arguments:
        r_df -- a pandas.DataFrame of asset returns

    Return:
        normed_df -- normalized (z-scored) returns
    """
    mean_r = r_df.mean(axis=0)
    sd_r = r_df.std(axis=0)
    normed_df = (r_df - mean_r) / sd_r
    return normed_df


def split_index_column(df: pd.DataFrame, index_col: str = "SPX") -> Tuple[pd.DataFrame, pd.Series]:
    """Split a price/return panel into (constituent stocks, benchmark index).

    Arguments:
        df -- panel that includes the benchmark as one of its columns
        index_col -- name of the benchmark column, default 'SPX'

    Return:
        (stocks_df, index_series) -- constituents only, and the index alone
    """
    if index_col not in df.columns:
        raise KeyError(f"{index_col!r} column not found in {list(df.columns)[:5]}...")
    stocks_df = df.drop(columns=[index_col])
    index_series = df[index_col]
    return stocks_df, index_series
