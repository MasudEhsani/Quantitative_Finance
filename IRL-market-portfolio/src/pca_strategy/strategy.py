"""
The "AR Delta" regime-switching strategy, following M. Kritzman: use shifts
in the Absorption Ratio to tilt a portfolio between Equity (EQ) and Fixed
Income (FI).

    AR_delta = (AR_15d - AR_1yr) / AR_sigma_1yr

AR_delta standardizes the recent (15-day) move in the Absorption Ratio
against its own trailing one-year mean and volatility. A sharp *rise*
in systemic risk (AR_delta high) triggers a shift out of equities; a
sharp *fall* (AR_delta low) is read as a risk-on signal.
"""
from typing import Tuple

import numpy as np
import pandas as pd


def compute_ar_delta(
    ts_absorption_ratio: pd.Series,
    short_window: int = 15,
    long_window: int = 252,
) -> pd.DataFrame:
    """Compute AR_delta and its component moving statistics.

    Arguments:
        ts_absorption_ratio -- pandas.Series of the Absorption Ratio over time
        short_window -- short moving-average window, in days (e.g. 15)
        long_window -- long moving-average / volatility window, in days (e.g. 252)

    Return:
        df -- pandas.DataFrame with columns 'AR_1yr', 'AR_15d', 'AR_sigma_1yr',
            'AR_delta', indexed like ``ts_absorption_ratio``
    """
    ar_mean_long = ts_absorption_ratio.rolling(long_window).mean()
    ar_mean_short = ts_absorption_ratio.rolling(short_window).mean()
    ar_sd_long = ts_absorption_ratio.rolling(long_window).std()
    ar_delta = (ar_mean_short - ar_mean_long) / ar_sd_long

    return pd.DataFrame(
        {
            "AR_1yr": ar_mean_long,
            "AR_15d": ar_mean_short,
            "AR_sigma_1yr": ar_sd_long,
            "AR_delta": ar_delta,
        },
        index=ts_absorption_ratio.index,
    )


def get_weight(ar_delta: float, threshold: float = 1.0) -> list:
    """Map a single AR_delta reading to EQ / FI portfolio weights.

    Rule (Kritzman):
        AR_delta > +threshold  -> risk-off:  0% EQ / 100% FI
        AR_delta < -threshold  -> risk-on: 100% EQ /   0% FI
        otherwise               -> neutral:  50% EQ /  50% FI

    Arguments:
        ar_delta -- a single AR_delta reading (float)
        threshold -- the +/- standard-deviation trigger level

    Return:
        wgts -- [EQ weight, FI weight]
    """
    if ar_delta > threshold:
        return [0.0, 1.0]
    if ar_delta < -threshold:
        return [1.0, 0.0]
    return [0.5, 0.5]


def get_weights_series(ar_delta: pd.Series, threshold: float = 1.0) -> pd.DataFrame:
    """Vectorized version of ``get_weight`` applied to a whole Series.

    Arguments:
        ar_delta -- pandas.Series of AR_delta values
        threshold -- the +/- standard-deviation trigger level

    Return:
        wgts -- pandas.DataFrame, columns ['EQ', 'FI'], indexed like ar_delta
    """
    eq = np.select(
        [ar_delta.values > threshold, ar_delta.values < -threshold],
        [0.0, 1.0],
        default=0.5,
    )
    fi = 1.0 - eq
    return pd.DataFrame({"EQ": eq, "FI": fi}, index=ar_delta.index)


def count_rebalances(wgts: pd.DataFrame) -> pd.Series:
    """Flag rows where the EQ weight changed from the prior row (a trade),
    and roll them up into a trades-per-calendar-year count.

    Arguments:
        wgts -- pandas.DataFrame with an 'EQ' column, DatetimeIndex

    Return:
        trades_per_year -- pandas.Series indexed by year
    """
    changed = wgts["EQ"].diff().fillna(0).ne(0)
    return changed.groupby(wgts.index.year).sum()


def backtest_strategy(
    strat_wgts: pd.DataFrame,
    asset_returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> Tuple[float, float, float]:
    """Backtest a (possibly time-varying) set of portfolio weights against
    realized asset returns, and summarize performance.

    On each date, the portfolio return is the dot product of that day's
    weights and that day's asset returns (i.e. weights are assumed to be
    set using only information available up to, and held through, that
    day -- callers are responsible for not leaking future information into
    ``strat_wgts``).

    Arguments:
        strat_wgts -- pandas.DataFrame of weights, one column per asset,
            aligned (same columns) with ``asset_returns``
        asset_returns -- pandas.DataFrame of asset returns
        periods_per_year -- number of return observations per year, used to
            annualize (252 for daily data)

    Return:
        (ann_ret, ann_vol, sharpe) -- annualized return, annualized
            volatility, and (zero-rate) Sharpe ratio of the strategy
    """
    common_cols = [c for c in strat_wgts.columns if c in asset_returns.columns]
    if not common_cols:
        raise ValueError("strat_wgts and asset_returns share no columns")

    common_idx = strat_wgts.index.intersection(asset_returns.index)
    w = strat_wgts.loc[common_idx, common_cols]
    r = asset_returns.loc[common_idx, common_cols]

    port_returns = (w.values * r.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=common_idx)

    ann_ret = float(port_returns.mean() * periods_per_year)
    ann_vol = float(port_returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan

    return ann_ret, ann_vol, sharpe


def cumulative_growth(port_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Turn a return series into a cumulative growth-of-$1 curve."""
    return start_value * (1.0 + port_returns).cumprod()
