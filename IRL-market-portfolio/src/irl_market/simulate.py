"""
Synthetic data generator for the GMR/IRL market model.

Used for two purposes:

1. Producing a documented, reproducible placeholder ``dja_cap.csv`` when
   the real course dataset isn't available (see ``scripts/generate_dja_data.py``).
2. Parameter-recovery testing: since the data-generating process here uses
   known ``kappa`` / ``W`` / ``sigma``, fitting ``estimation.fit_gmr_pooled``
   to the simulated panel should recover values close to the ground truth --
   a strong end-to-end check that the estimator is implemented correctly
   (see ``tests/test_irl_market.py``).

The simulation is intentionally self-consistent with how the model is
*estimated*: at each step, the same SMA-deviation signals used later for
fitting are computed from the path generated so far, then fed through the
known ``kappa``/``W`` to get the next step's drift. This can't be
vectorized away (each step depends on the signal computed from the
just-generated history), so it is a genuine sequential simulation loop.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def simulate_gmr_with_sma_signals(
    tickers,
    n_days: int,
    kappa: float,
    w0: float,
    w_sma_short: float,
    w_sma_long: float,
    sma_short: int = 10,
    sma_long: int = 30,
    sigma: float = 0.015,
    x0: float = 1.0,
    x0_jitter: float = 0.05,
    idio_vol_jitter: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a panel of normalized (~1.0-scale) asset levels from

        Delta x_t / x_t = kappa * (target_t - x_t) + eps_t
        target_t = w0 + w_sma_short * sma_dev(x, sma_short)_t + w_sma_long * sma_dev(x, sma_long)_t
        eps_t ~ N(0, sigma_i^2)  (independent across assets, heterogeneous idiosyncratic vol)

    Arguments:
        tickers -- list of asset names (columns of the output panel)
        n_days -- number of business days to simulate
        kappa -- ground-truth (shared, "pooled") mean-reversion speed
        w0 -- ground-truth equilibrium level (constant term)
        w_sma_short, w_sma_long -- ground-truth loadings on the two SMA-deviation signals
        sma_short, sma_long -- lookback windows for those signals
        sigma -- base idiosyncratic daily volatility
        x0 -- starting normalized level (same order of magnitude for every asset)
        x0_jitter -- relative spread of starting levels across assets
        idio_vol_jitter -- relative spread of per-asset volatility around ``sigma``
        seed -- RNG seed

    Return:
        x_path -- pandas.DataFrame, business-day index, one column per ticker,
            normalized levels (start near 1.0)
    """
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    warmup = sma_long  # need this many days of history before signals are defined

    x0_vec = x0 * (1.0 + rng.uniform(-x0_jitter, x0_jitter, size=n_assets))
    sigma_vec = sigma * (1.0 + rng.uniform(-idio_vol_jitter, idio_vol_jitter, size=n_assets))

    path = np.zeros((n_days, n_assets))
    path[0, :] = x0_vec

    # Warm-up phase: plain reversion to w0 (SMA signals undefined without history).
    for t in range(1, min(warmup, n_days)):
        drift = kappa * (w0 - path[t - 1, :])
        noise = sigma_vec * rng.normal(size=n_assets)
        path[t, :] = path[t - 1, :] * (1.0 + drift + noise)

    # Main phase: recompute SMA-deviation signals from the path generated so far.
    for t in range(warmup, n_days):
        window_short = path[t - sma_short:t, :]
        window_long = path[t - sma_long:t, :]
        sma_s = window_short.mean(axis=0)
        sma_l = window_long.mean(axis=0)
        current = path[t - 1, :]
        sig_short = (current - sma_s) / sma_s
        sig_long = (current - sma_l) / sma_l

        target = w0 + w_sma_short * sig_short + w_sma_long * sig_long
        drift = kappa * (target - current)
        noise = sigma_vec * rng.normal(size=n_assets)
        path[t, :] = current * (1.0 + drift + noise)

    dates = pd.bdate_range(start="2010-01-04", periods=n_days, freq="B")
    return pd.DataFrame(path, index=dates, columns=tickers)


def to_dollar_caps(x_norm: pd.DataFrame, base_caps_billions: Dict[str, float]) -> pd.DataFrame:
    """Rescale a normalized (~1.0) simulated panel to plausible dollar market
    caps, given a per-ticker baseline (in $ billions)."""
    base = pd.Series(base_caps_billions).reindex(x_norm.columns)
    return x_norm.multiply(base, axis=1) * 1e9
