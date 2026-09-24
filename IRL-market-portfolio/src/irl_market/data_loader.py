"""
Load market-cap/price panels and put them in the form the GMR estimation
routines expect: a business-day-indexed DataFrame of positive levels, one
column per asset.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# A representative (illustrative, not necessarily the exact historical
# constituent list on every date) set of 30 large-cap Dow-style names, with a
# rough GICS-like sector grouping. Used both to label the synthetic DJI
# dataset and to demonstrate the "per-sector" parametrization option the
# assignment suggests.
# ---------------------------------------------------------------------------
DJI_SECTORS: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "IBM": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "XOM": "Energy", "CVX": "Energy",
    "JPM": "Financials", "GS": "Financials", "AXP": "Financials", "V": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "UNH": "Healthcare",
    "KO": "Consumer Staples", "PG": "Consumer Staples", "WMT": "Consumer Staples", "WBA": "Consumer Staples",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "DIS": "Consumer Discretionary",
    "BA": "Industrials", "CAT": "Industrials", "MMM": "Industrials", "HON": "Industrials",
    "VZ": "Telecom",
}

# Rough, illustrative 2010-era market caps ($B), used only to rescale the
# synthetic DJI dataset to a realistic-looking dollar range. Not sourced from
# real market data -- see scripts/generate_dja_data.py.
DJI_BASE_CAP_BILLIONS: Dict[str, float] = {
    "AAPL": 300, "MSFT": 240, "IBM": 180, "INTC": 115, "CSCO": 120, "CRM": 15, "ORCL": 65,
    "XOM": 320, "CVX": 200,
    "JPM": 170, "GS": 85, "AXP": 55, "V": 65,
    "JNJ": 175, "PFE": 140, "MRK": 110, "UNH": 35,
    "KO": 150, "PG": 180, "WMT": 190, "WBA": 30,
    "HD": 55, "MCD": 75, "NKE": 30, "DIS": 70,
    "BA": 40, "CAT": 35, "MMM": 55, "HON": 30,
    "VZ": 100,
}


def load_dja_caps(path: Path = DATA_DIR / "dja_cap.csv", start_date: str = "2010-01-04") -> pd.DataFrame:
    """Load the DJI 30 market-cap panel.

    The source file (matching the original course dataset's format) has no
    date column -- rows are simply consecutive business days starting
    ``start_date``, one column per ticker.

    Arguments:
        path -- path to dja_cap.csv
        start_date -- first business day in the file

    Return:
        df_cap -- pandas.DataFrame, DatetimeIndex, one column per ticker
    """
    df_cap = pd.read_csv(path)
    dates = pd.bdate_range(start=start_date, periods=df_cap.shape[0], freq="B")
    df_cap["date"] = dates
    df_cap.set_index("date", inplace=True)
    return df_cap


def load_spx_caps(path: Path = DATA_DIR / "spx_holdings_and_spx_closeprice.csv",
                   drop_index_col: str = "SPX") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load the S&P 500 constituent price panel used as a (much larger)
    second universe for Part 3.

    As noted in the project background, GMR dynamics for market caps carry
    over directly to prices so long as shares outstanding are held fixed, so
    -- exactly as in the original Part 3 code -- constituent *prices* are
    used directly as the model's state variable ``X_t``.

    Arguments:
        path -- path to spx_holdings_and_spx_closeprice.csv
        drop_index_col -- name of the benchmark index column to split off
            (kept separately, not included in the estimation universe)

    Return:
        (stock_prices, index_series)
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
    prices = prices.sort_index()
    index_series = None
    if drop_index_col in prices.columns:
        index_series = prices[drop_index_col]
        prices = prices.drop(columns=[drop_index_col])
    return prices, index_series


def normalize_levels(X: pd.DataFrame, window: int = 30, method: str = "first_window"
                      ) -> Tuple[pd.DataFrame, pd.Series]:
    """Rescale each asset's level series to be O(1) and comparable across
    assets.

    Why normalize at all: the GMR state equation's mean-reversion target
    ``W @ z'_t`` is shared (or shared-within-group) across assets in the
    simplest parametrizations the assignment suggests. That only makes
    sense if the state itself is on a common, asset-independent scale --
    raw dollar market caps for a $15B company and a $700B company are not
    comparable. (This is the normalization the assignment alludes to when
    it suggests changing units, e.g. to log-levels; an indexed level is the
    multiplicative-model analogue of that suggestion.)

    Two methods are offered:

    - ``"first_window"`` -- divide by each asset's own average level over
      the first ``window`` observations, giving a growth-of-1 total-return
      index. Simple and literal, and fine for a sample short enough (a few
      years) that levels don't drift arbitrarily far from 1. On a long,
      highly divergent panel (a decade-plus of 400+ stocks, some of which
      10x and others of which nearly go to zero) this normalization lets
      ``x_t`` range over two orders of magnitude, which in turn makes the
      pooled regression numerically unstable (``x_t`` becomes an extreme,
      high-leverage outlier for a handful of names and dominates the fit).
    - ``"rolling"`` -- divide by each asset's own trailing rolling mean
      (window in days). This detrends slow, long-run growth and keeps the
      normalized state oscillating around ~1.0 for the *entire* sample
      regardless of how long it is or how much dispersion there is in
      long-run performance across names -- much better suited to a large,
      long-horizon universe such as the S&P 500 constituents used in Part 3.

    Arguments:
        X -- pandas.DataFrame of raw levels (price or market cap), one
            column per asset
        window -- number of observations used for the baseline (leading
            window average for "first_window"; rolling window for "rolling")
        method -- "first_window" or "rolling"

    Return:
        (X_norm, baseline) -- normalized levels, and the baseline used to
            produce them (a per-asset Series for "first_window", or the
            full rolling-mean DataFrame for "rolling" -- either way,
            ``X_norm * baseline`` recovers the original raw levels)
    """
    if method == "first_window":
        baseline = X.iloc[:window].mean(axis=0)
        if (baseline <= 0).any():
            raise ValueError("Non-positive baseline level encountered; check input data")
        return X.divide(baseline, axis=1), baseline
    elif method == "rolling":
        baseline = X.rolling(window).mean()
        X_norm = (X / baseline).dropna(how="any")
        return X_norm, baseline.loc[X_norm.index]
    else:
        raise ValueError("method must be 'first_window' or 'rolling', got %r" % method)


def sector_map_for(columns, sectors: Dict[str, str] = DJI_SECTORS, default: str = "Other") -> pd.Series:
    """Map a list of tickers to sector labels, defaulting unknown tickers to
    a single 'Other' bucket (relevant for the S&P universe, which is far
    larger than the DJI sector table above)."""
    return pd.Series({c: sectors.get(c, default) for c in columns}, name="sector")
