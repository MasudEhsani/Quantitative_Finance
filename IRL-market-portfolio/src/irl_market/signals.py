"""
Predictive signals ``z_t`` used as regressors for the GMR mean-reversion
target. Every function here takes a level panel (price or market cap, one
column per asset) and returns a same-shaped panel of a dimensionless signal,
so signals can be freely mixed and matched for the model comparisons in
Part 1/2/3.
"""
import pandas as pd


def sma_deviation(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Percentage deviation of the level from its own trailing simple moving
    average: (X - SMA) / SMA. Positive => currently above its recent average.
    """
    sma = X.rolling(window).mean()
    return (X - sma) / sma


def momentum(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing total return over ``window`` days: X_t / X_{t-window} - 1."""
    return X.pct_change(window)


def volatility(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing realized volatility: rolling std of daily returns."""
    return X.pct_change().rolling(window).std()


def drawdown_from_high(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Distance below the trailing ``window``-day high: (X - rolling_max) /
    rolling_max, always <= 0. A proxy for loss-aversion / anchoring-driven
    mean reversion: stocks far below a recent high may be "cheap" relative
    to where investors recently anchored their expectations.
    """
    rolling_max = X.rolling(window).max()
    return (X - rolling_max) / rolling_max


def zscore(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling z-score of the level relative to its own trailing mean/std:
    a smoother, scale-free alternative to the raw SMA deviation.
    """
    roll_mean = X.rolling(window).mean()
    roll_std = X.rolling(window).std()
    return (X - roll_mean) / roll_std


# Convenient named bundles used by main.py / the notebook.
BASELINE_SIGNALS = {
    "sma10": lambda X: sma_deviation(X, 10),
    "sma30": lambda X: sma_deviation(X, 30),
}

EXTENDED_SIGNALS = {
    **BASELINE_SIGNALS,
    "mom20": lambda X: momentum(X, 20),
    "vol20": lambda X: volatility(X, 20),
    "dd60": lambda X: drawdown_from_high(X, 60),
    "z60": lambda X: zscore(X, 60),
}


def build_signal_panels(X: pd.DataFrame, signal_specs: dict) -> dict:
    """Apply each ``name -> fn(X)`` spec in ``signal_specs`` to ``X``.

    Arguments:
        X -- level panel (price or market cap)
        signal_specs -- dict of {signal_name: callable(X) -> DataFrame}

    Return:
        dict of {signal_name: DataFrame}, same shape as X, aligned index/columns
    """
    return {name: fn(X) for name, fn in signal_specs.items()}
