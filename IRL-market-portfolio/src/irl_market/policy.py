"""
Part 4: turn the estimated GMR/IRL model into a simple trading strategy.

The model's mean equation says each asset drifts toward a signal-implied
"fair value" level ``target_t,i = W . z'_t,i``. The gap ``target_t,i -
x_t,i`` is therefore a natural, model-implied mispricing signal: assets
trading below their model-implied target are, under the model, expected to
rise; assets trading above it are expected to fall. This module turns that
gap into a simple dollar-neutral, cross-sectionally ranked long/short
policy and backtests it -- directly analogous to (and comparable against)
the PCA/Absorption-Ratio-based strategy built in Course 2
(``pca_strategy.strategy``, vendored into this project -- see ``main.py``
Part 4 and ``src/pca_strategy/``).
"""
from typing import Dict, Union

import numpy as np
import pandas as pd

from .estimation import GMRFitResult


def compute_target(signals_t: Dict[str, pd.DataFrame], W: Union[pd.Series, pd.DataFrame],
                    include_const: bool = True) -> pd.DataFrame:
    """Reconstruct the model-implied target level target_t,i = W . z'_t,i.

    Arguments:
        signals_t -- dict of {signal_name: DataFrame (T x N)}, as produced by
            ``estimation.prepare_regression_arrays``
        W -- fitted loadings: a pooled ``pd.Series`` (shared across assets)
            or a per-asset/per-sector ``pd.DataFrame`` (rows = asset)
        include_const -- whether W includes a 'const' entry

    Return:
        target -- pandas.DataFrame (T x N), same shape as the signal panels
    """
    any_panel = next(iter(signals_t.values()))
    if isinstance(W, pd.Series):
        target = pd.DataFrame(0.0, index=any_panel.index, columns=any_panel.columns)
        if include_const and "const" in W.index:
            target = target + W["const"]
        for name, panel in signals_t.items():
            if name in W.index:
                target = target + W[name] * panel
        return target

    # per-asset / per-sector: W is a DataFrame, rows = asset
    target = pd.DataFrame(0.0, index=any_panel.index, columns=W.index)
    if include_const and "const" in W.columns:
        target = target.add(W["const"], axis=1)
    for name, panel in signals_t.items():
        if name in W.columns:
            target = target + panel[W.index].multiply(W[name], axis=1)
    return target


def mispricing_zscore(target: pd.DataFrame, X_t: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, each day, of (target - x_t): how far below
    (positive) or above (negative sign flipped) its model-implied fair
    value each asset is, relative to the rest of the universe that day.
    """
    mispricing = target[X_t.columns] - X_t
    mu = mispricing.mean(axis=1)
    sd = mispricing.std(axis=1)
    return mispricing.sub(mu, axis=0).div(sd, axis=0)


def dollar_neutral_weights(signal_z: pd.DataFrame, clip: float = 2.5) -> pd.DataFrame:
    """Turn a cross-sectional z-scored signal into dollar-neutral portfolio
    weights: long the assets with the most positive signal, short the most
    negative, scaled so gross exposure is 2.0 (1.0 long / 1.0 short) each day.
    """
    z = signal_z.clip(-clip, clip)
    z = z.sub(z.mean(axis=1), axis=0)  # re-center after clipping
    gross = z.abs().sum(axis=1)
    gross = gross.replace(0, np.nan)
    return z.div(gross, axis=0) * 2.0


def backtest_policy(weights: pd.DataFrame, X_t: pd.DataFrame, X_next: pd.DataFrame,
                     periods_per_year: int = 252, cost_bps: float = 0.0):
    """Backtest a set of daily dollar-neutral weights against realized
    one-step returns implied by the same (X_t, X_next) pair used for
    estimation.

    A short-horizon, daily-rebalanced cross-sectional signal on a wide,
    liquid universe like this one can show an unrealistically high gross
    (frictionless) Sharpe ratio -- with N=400+ names rebalanced ~1000
    times, Grinold's "fundamental law of active management" says even a
    weak per-bet edge compounds into a very large *paper* Sharpe purely
    from breadth. ``cost_bps`` lets that be stress-tested: a simple
    proportional cost is charged on each day's turnover (sum of absolute
    weight changes), which is normally what separates an academically
    "significant" short-horizon signal from a strategy that would actually
    be worth trading.

    Arguments:
        weights -- pandas.DataFrame (T x N) of portfolio weights, indexed
            like ``X_t`` (the "as-of" date of each day's positions)
        X_t, X_next -- level panels one step apart *and paired positionally*
            (as from ``estimation.prepare_regression_arrays`` -- X_next's
            DatetimeIndex labels are one business day ahead of X_t's, by
            design, so they must be aligned by position, not by label)
        periods_per_year -- for annualization
        cost_bps -- proportional transaction cost, in basis points of
            traded notional, charged on each day's turnover (0 = frictionless)

    Return:
        (ann_ret, ann_vol, sharpe, port_returns) -- performance summary and
            the daily net-of-cost portfolio return series
    """
    cols = weights.columns.intersection(X_t.columns)
    common_idx = weights.index.intersection(X_t.index)
    positions = X_t.index.get_indexer(common_idx)

    w = weights.loc[common_idx, cols].fillna(0.0)
    X_t_sub = X_t.loc[common_idx, cols]
    X_next_sub = X_next.iloc[positions][cols]  # positional pairing, not label-based
    r = (X_next_sub.values - X_t_sub.values) / X_t_sub.values

    gross_returns = (w.values * r).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1).fillna(0.0).values
    cost = turnover * (cost_bps / 1e4)

    port_returns = pd.Series(gross_returns - cost, index=common_idx)

    ann_ret = float(port_returns.mean() * periods_per_year)
    ann_vol = float(port_returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe, port_returns


def implied_policy_from_fit(fit: GMRFitResult, X: pd.DataFrame, signal_panels: Dict[str, pd.DataFrame],
                             include_const: bool = True, clip: float = 2.5):
    """Convenience wrapper: go straight from a ``GMRFitResult`` (+ the
    signal panels used to produce it) to a backtested long/short policy.

    Return:
        (ann_ret, ann_vol, sharpe, port_returns, weights)
    """
    from .estimation import prepare_regression_arrays

    X_t, X_next, signals_t = prepare_regression_arrays(X, signal_panels)
    target = compute_target(signals_t, fit.W, include_const=include_const)
    z = mispricing_zscore(target, X_t)
    weights = dollar_neutral_weights(z, clip=clip)
    ann_ret, ann_vol, sharpe, port_returns = backtest_policy(weights, X_t, X_next, )
    return ann_ret, ann_vol, sharpe, port_returns, weights
