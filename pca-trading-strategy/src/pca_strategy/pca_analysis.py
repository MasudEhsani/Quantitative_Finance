"""
Rolling PCA over a universe of asset returns, and the Absorption Ratio (AR)
measure of market systemic risk (Kritzman, Li, Page & Rigobon, 2011).

The Absorption Ratio is the fraction of the total variance of a set of
assets explained by a fixed, small number of principal components. When a
market is fragile, a large share of return variance is explained by very
few common factors (AR is high); when risk is more diversified across many
independent sources, AR is lower.
"""
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .weighting import exponent_weighting


def absorption_ratio(explained_variance: np.ndarray, n_components: int) -> float:
    """Absorption ratio: share of total variance explained by the top components.

    Arguments:
        explained_variance -- 1D np.array of explained variance per principal
            component, in descending order (e.g. PCA(...).explained_variance_)
        n_components -- number of leading principal components to use in the
            numerator

    Return:
        ar -- absorption ratio, a float in [0, 1]
    """
    explained_variance = np.asarray(explained_variance)
    return float(np.sum(explained_variance[:n_components]) / np.sum(explained_variance))


def weighted_covariance(returns_window: pd.DataFrame, weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Sample covariance matrix, optionally with exponentially-decaying weights.

    Arguments:
        returns_window -- pandas.DataFrame, rows = time, columns = assets,
            ordered oldest-to-newest
        weights -- optional 1D array of per-row weights aligned with
            ``returns_window`` (e.g. from ``exponent_weighting``, reversed so
            the most recent row gets the largest weight). If None, an
            ordinary equally-weighted sample covariance is returned.

    Return:
        cov -- pandas.DataFrame, the (weighted) covariance matrix
    """
    if weights is None:
        return returns_window.cov()

    weights = np.asarray(weights)
    assert len(weights) == len(returns_window), "weights must match number of rows"
    x = returns_window.values
    mean = np.average(x, axis=0, weights=weights)
    xc = x - mean
    # weighted, bias-corrected covariance: sum_j w_j * xc_j xc_j^T / (1 - sum w_j^2)
    cov = (xc * weights[:, None]).T @ xc / (1.0 - np.sum(weights ** 2))
    return pd.DataFrame(cov, index=returns_window.columns, columns=returns_window.columns)


def rolling_pca_absorption_ratio(
    normed_returns: pd.DataFrame,
    lookback_window: int = 252 * 2,
    step_size: int = 1,
    var_threshold: float = 0.8,
    absorb_fraction: float = 0.2,
    start_offset: int = 0,
    end_index: Optional[int] = None,
    use_ewm: bool = False,
    half_life: int = 252,
    recompute_every: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Roll a PCA over a moving window of returns and track two things at
    each step: the Absorption Ratio, and how many components it takes to
    explain ``var_threshold`` of total variance.

    A fresh PCA fit is normally an O(window x n_assets^2) operation, which
    adds up over thousands of trading days, so ``recompute_every`` lets the
    result be refreshed only every N steps and forward-filled in between --
    this mirrors how the metric is used in practice (it does not need to be
    literally daily to be a useful signal) while keeping the walk-forward
    loop itself fast. Set ``recompute_every=1`` for a fully daily series.

    Arguments:
        normed_returns -- standardized (z-scored) returns, one column per
            asset (the benchmark index column should already be excluded)
        lookback_window -- size of the rolling estimation window, in days
        step_size -- how many days to advance the window on each iteration
        var_threshold -- fraction of variance that must be explained to
            determine the "number of components" series
        absorb_fraction -- fraction of all assets' worth of principal
            components used in the Absorption Ratio numerator (e.g. 0.2 ->
            top 20% of components)
        start_offset -- number of extra days of history to require before
            the first window (e.g. to line up with a benchmark start date)
        end_index -- stop after this many rows of ``normed_returns`` (None
            = use the full history)
        use_ewm -- if True, weight each window with exponentially decaying
            weights (``half_life``) before computing the covariance matrix
        half_life -- half-life (days) for the exponential weights
        recompute_every -- refit PCA every N steps and forward-fill between
            refits (1 = refit on every step)
        verbose -- print a short progress line when done

    Return:
        result -- pandas.DataFrame indexed by date with columns
            'n_components' (int, components needed for var_threshold) and
            'absorption_ratio' (float)
    """
    n_assets = normed_returns.shape[1]
    absorb_comp = max(1, int(absorb_fraction * n_assets))
    end_index = len(normed_returns) if end_index is None else min(end_index, len(normed_returns))

    idx_range = range(lookback_window + start_offset, end_index, step_size)
    ts_index = normed_returns.index[list(idx_range)]

    n_comp_arr = np.full(len(ts_index), np.nan)
    ar_arr = np.full(len(ts_index), np.nan)

    exp_probs = exponent_weighting(lookback_window, half_life)[::-1] if use_ewm else None

    t0 = time.time()
    for i, ix in enumerate(idx_range):
        if i == 0 or i % recompute_every == 0:
            window = normed_returns.iloc[ix - lookback_window:ix, :]
            cov_mat = weighted_covariance(window, exp_probs)

            eigvals = np.linalg.eigvalsh(cov_mat.values)
            eigvals = np.sort(eigvals)[::-1]  # descending, like PCA's convention

            cum_var = np.cumsum(eigvals / eigvals.sum())
            n_comp_arr[i] = int(np.searchsorted(cum_var, var_threshold) + 1)
            ar_arr[i] = absorption_ratio(eigvals, absorb_comp)
        else:
            n_comp_arr[i] = n_comp_arr[i - 1]
            ar_arr[i] = ar_arr[i - 1]

    if verbose:
        print(f"Rolling PCA / Absorption Ratio: {len(ts_index)} steps in "
              f"{time.time() - t0:.1f}s (recompute every {recompute_every} step(s))")

    return pd.DataFrame(
        {"n_components": n_comp_arr, "absorption_ratio": ar_arr},
        index=ts_index,
    )


def fit_pca(returns_window: pd.DataFrame) -> PCA:
    """Fit scikit-learn PCA on a covariance matrix of a returns window.

    Kept as a thin wrapper (rather than inlining ``PCA().fit(...)``
    everywhere) so the rest of the code base has one place to swap in, say,
    a shrinkage estimator for the covariance matrix.
    """
    cov_mat = returns_window.cov()
    return PCA().fit(cov_mat.values)
