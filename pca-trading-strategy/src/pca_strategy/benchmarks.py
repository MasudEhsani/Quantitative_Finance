"""
Equity / Fixed-Income return series used to backtest the AR-Delta strategy.

The original course project benchmarked the strategy against real VTI
(equity) and AGG (bond) ETF returns, supplied as separate CSV files that are
not part of this project's single data source. To keep this project fully
self-contained and runnable from just
``data/spx_holdings_and_spx_closeprice.csv``, two options are supported:

1. Drop a CSV named ``data/eq_fi_returns.csv`` with a date index and two
   columns ``EQ`` and ``FI`` (e.g. real VTI/AGG daily returns) -- if present,
   it is used automatically and takes priority.
2. Otherwise, EQ is taken directly from the SPX index column already in the
   provided dataset, and FI falls back to a clearly-labeled *synthetic*
   bond-like proxy (low, roughly AGG-like volatility and drift, seeded for
   reproducibility) -- this is a placeholder for a real fixed-income series,
   not a market data feed, and should be treated accordingly.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def synthesize_fi_returns(
    index: pd.DatetimeIndex,
    annual_return: float = 0.035,
    annual_vol: float = 0.035,
    seed: int = 7,
) -> pd.Series:
    """Generate a synthetic, low-volatility fixed-income-like daily return
    series as a stand-in for a real bond ETF (e.g. AGG) when no such data is
    available.

    This is intentionally simple (i.i.d. Gaussian daily returns calibrated
    to plausible aggregate-bond annualized return/volatility) -- it is
    *not* fit to any real bond market data, and exists purely so the
    strategy backtest has a second asset to allocate to out of the box.

    Arguments:
        index -- DatetimeIndex to generate returns over
        annual_return -- target annualized mean return
        annual_vol -- target annualized volatility
        seed -- RNG seed, for reproducibility

    Return:
        fi_returns -- pandas.Series of daily returns, indexed by ``index``
    """
    periods_per_year = 252
    daily_mean = annual_return / periods_per_year
    daily_vol = annual_vol / np.sqrt(periods_per_year)
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=daily_mean, scale=daily_vol, size=len(index))
    return pd.Series(draws, index=index, name="FI")


def load_or_build_eq_fi_returns(
    spx_returns: pd.Series,
    data_dir: Path,
) -> pd.DataFrame:
    """Return a (EQ, FI) daily returns DataFrame for backtesting.

    Arguments:
        spx_returns -- pandas.Series of SPX index daily returns (used as the
            EQ leg, and as the date index, when no override file is found)
        data_dir -- directory to look for an optional ``eq_fi_returns.csv``
            override file

    Return:
        eq_fi -- pandas.DataFrame with columns ['EQ', 'FI']
        used_real_data -- bool, True if a real override file was found
    """
    override_path = Path(data_dir) / "eq_fi_returns.csv"
    if override_path.exists():
        eq_fi = pd.read_csv(override_path, index_col=0, parse_dates=True)
        eq_fi = eq_fi[["EQ", "FI"]].dropna()
        return eq_fi, True

    fi_returns = synthesize_fi_returns(spx_returns.index)
    eq_fi = pd.DataFrame({"EQ": spx_returns.values, "FI": fi_returns.values}, index=spx_returns.index)
    return eq_fi, False
