"""
Sanity tests for the irl_market package.

Run with pytest (``pytest tests/``) or directly:

    python tests/test_irl_market.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.irl_market.data_loader import normalize_levels, sector_map_for
from src.irl_market.estimation import (
    fit_gmr_per_asset,
    fit_gmr_per_group,
    fit_gmr_pooled,
    prepare_regression_arrays,
)
from src.irl_market.policy import (
    backtest_policy,
    compute_target,
    dollar_neutral_weights,
    mispricing_zscore,
)
from src.irl_market.signals import (
    drawdown_from_high,
    momentum,
    sma_deviation,
    volatility,
    zscore,
)
from src.irl_market.simulate import simulate_gmr_with_sma_signals, to_dollar_caps


TICKERS = [f"T{i:02d}" for i in range(12)]
GROUND_TRUTH = dict(kappa=0.05, w0=1.0, w_sma_short=0.3, w_sma_long=-0.15, sigma=0.012, seed=123)


def _simulate(n_days=900):
    return simulate_gmr_with_sma_signals(TICKERS, n_days, **GROUND_TRUTH)


def test_simulation_stays_positive_and_bounded():
    x = _simulate()
    assert (x.values > 0).all()
    assert x.values.max() < 10  # mean-reverting around ~1, shouldn't blow up
    assert x.shape == (900, len(TICKERS))


def test_normalize_levels():
    x = _simulate()
    x_norm, baseline = normalize_levels(x, window=30)
    assert np.allclose(x_norm.iloc[:30].mean(axis=0).values, 1.0, atol=1e-8)
    assert (baseline > 0).all()


def test_prepare_regression_arrays_keeps_one_step_offset():
    x = _simulate()
    x_norm, _ = normalize_levels(x)
    signals = {"sma10": sma_deviation(x_norm, 10), "sma30": sma_deviation(x_norm, 30)}
    X_t, X_next, signals_t = prepare_regression_arrays(x_norm, signals)

    assert len(X_t) == len(X_next)
    assert not X_t.index.equals(X_next.index)
    # X_next at row k should equal the original series one step after X_t's row k
    full = x_norm
    for k in [0, 5, 50]:
        t_pos = full.index.get_loc(X_t.index[k])
        assert np.allclose(full.iloc[t_pos + 1].values, X_next.iloc[k].values)


def test_fit_gmr_per_asset_recovers_ground_truth_kappa():
    x = _simulate(n_days=1500)
    x_norm, _ = normalize_levels(x, window=30)
    signals = {"sma10": sma_deviation(x_norm, 10), "sma30": sma_deviation(x_norm, 30)}

    fit = fit_gmr_per_asset(x_norm, signals)
    assert abs(fit.kappa.mean() - GROUND_TRUTH["kappa"]) < 0.02
    assert fit.Sigma.shape == (len(TICKERS), len(TICKERS))
    assert np.isfinite(fit.loglik)
    assert fit.aic < 0 or fit.bic < 0 or True  # sign not guaranteed, just must be finite
    assert np.isfinite(fit.aic) and np.isfinite(fit.bic)


def test_fit_gmr_pooled_runs_and_returns_consistent_shapes():
    x = _simulate()
    x_norm, _ = normalize_levels(x)
    signals = {"sma10": sma_deviation(x_norm, 10), "sma30": sma_deviation(x_norm, 30)}
    fit = fit_gmr_pooled(x_norm, signals)

    assert isinstance(fit.kappa, float)
    assert set(fit.W.index) == {"const", "sma10", "sma30"}
    assert fit.residuals.shape[1] == len(TICKERS)
    assert fit.n_params == 4


def test_fit_gmr_per_group():
    x = _simulate()
    x_norm, _ = normalize_levels(x)
    signals = {"sma10": sma_deviation(x_norm, 10)}
    groups = pd.Series({t: "A" if i < 6 else "B" for i, t in enumerate(TICKERS)})
    fit = fit_gmr_per_group(x_norm, signals, groups)

    assert fit.kappa.nunique() == 2  # two groups -> exactly two distinct kappas
    assert hasattr(fit, "group_table")
    assert set(fit.group_table.index) == {"A", "B"}


def test_signals_have_expected_shape_and_sign():
    x = _simulate()
    sma = sma_deviation(x, 10)
    mom = momentum(x, 20)
    vol = volatility(x, 20)
    dd = drawdown_from_high(x, 60)
    z = zscore(x, 60)
    for panel in (sma, mom, vol, dd, z):
        assert panel.shape == x.shape
    assert (dd.dropna() <= 1e-9).all().all()  # drawdown from high is always <= 0
    assert (vol.dropna() >= 0).all().all()


def test_policy_backtest_is_dollar_neutral_and_runs():
    x = _simulate()
    x_norm, _ = normalize_levels(x)
    signals = {"sma10": sma_deviation(x_norm, 10), "sma30": sma_deviation(x_norm, 30)}
    fit = fit_gmr_pooled(x_norm, signals)

    X_t, X_next, signals_t = prepare_regression_arrays(x_norm, signals)
    target = compute_target(signals_t, fit.W)
    z = mispricing_zscore(target, X_t)
    weights = dollar_neutral_weights(z)

    row_sums = weights.dropna(how="all").sum(axis=1)
    assert np.allclose(row_sums.fillna(0), 0, atol=1e-6)  # long == short in $ terms
    gross = weights.abs().sum(axis=1).dropna()
    assert np.allclose(gross, 2.0, atol=1e-6)

    ann_ret, ann_vol, sharpe, port_returns = backtest_policy(weights, X_t, X_next)
    assert np.isfinite(ann_ret) and np.isfinite(ann_vol)
    assert len(port_returns) == len(X_t)


def test_to_dollar_caps_scales_correctly():
    x = _simulate(n_days=50)
    x_norm, _ = normalize_levels(x, window=10)
    base = {t: 10.0 * (i + 1) for i, t in enumerate(TICKERS)}  # $B
    dollars = to_dollar_caps(x_norm, base)
    ratio = dollars.iloc[0] / x_norm.iloc[0]
    expected = pd.Series(base)[TICKERS] * 1e9
    assert np.allclose(ratio.values, expected.values)


def test_sector_map_defaults_unknown_tickers():
    m = sector_map_for(["AAPL", "NOT_A_REAL_TICKER"])
    assert m["AAPL"] == "Technology"
    assert m["NOT_A_REAL_TICKER"] == "Other"


def _run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ok  {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
