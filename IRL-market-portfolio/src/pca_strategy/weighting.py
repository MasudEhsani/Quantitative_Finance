"""Exponentially-decaying sample weights, used to optionally down-weight
older observations when estimating a covariance matrix."""
import numpy as np


def exponent_weighting(n_periods: int, half_life: int = 252) -> np.ndarray:
    """Exponentially smoothed, normalized (sums to 1) sample weights.

    Defines a decaying sequence

        x_j = exp(-log(2) / half_life * j),   j = 0 .. n_periods - 1

    and normalizes it so the weights form a probability distribution:

        w_j = x_j / sum(x)

    j = 0 gets the largest weight, so callers should order their window with
    the most recent observation first (or reverse the weights) depending on
    convention -- see ``pca_analysis.rolling_pca_absorption_ratio`` for how
    this is applied to a returns window.

    Arguments:
        n_periods -- number of periods, an integer, N in the formula above
        half_life -- half-life which determines the speed of decay

    Return:
        exp_probs -- exponentially smoothed weights, np.array, length n_periods
    """
    j = np.arange(n_periods)
    x = np.exp(-np.log(2) * j / half_life)
    return x / x.sum()
