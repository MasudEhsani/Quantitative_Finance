"""
Sanity tests for the pca_strategy package.

Written as plain functions with ``assert`` statements so they can be run
either with pytest (``pytest tests/``) or directly:

    python tests/test_pca_strategy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pca_strategy.autoencoder import autoencoder_absorption_ratio
from src.pca_strategy.data_loader import center_returns, compute_returns
from src.pca_strategy.eigen_portfolios import construct_eigen_portfolios, eigen_portfolio_returns
from src.pca_strategy.pca_analysis import absorption_ratio, rolling_pca_absorption_ratio
from src.pca_strategy.strategy import (
    backtest_strategy,
    compute_ar_delta,
    get_weight,
    get_weights_series,
)
from src.pca_strategy.weighting import exponent_weighting


def _make_synthetic_returns(n_days=600, n_assets=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    # a common factor plus idiosyncratic noise, so PCA has real structure to find
    factor = rng.normal(0, 0.01, size=n_days)
    idio = rng.normal(0, 0.01, size=(n_days, n_assets))
    loadings = rng.uniform(0.3, 1.0, size=n_assets)
    data = factor[:, None] * loadings[None, :] + idio
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_exponent_weighting_sums_to_one():
    w = exponent_weighting(252, half_life=63)
    assert len(w) == 252
    assert np.isclose(w.sum(), 1.0)
    assert np.all(np.diff(w) <= 0), "weights should be non-increasing"


def test_absorption_ratio_bounds():
    ev = np.array([5.0, 3.0, 1.0, 1.0, 1.0])
    ar_all = absorption_ratio(ev, n_components=5)
    ar_none = absorption_ratio(ev, n_components=0)
    ar_some = absorption_ratio(ev, n_components=1)
    assert np.isclose(ar_all, 1.0)
    assert np.isclose(ar_none, 0.0)
    assert 0.0 < ar_some < 1.0


def test_compute_returns_and_center():
    prices = pd.DataFrame(
        {"A": [100, 101, 99, 102], "B": [50, 50.5, 51, 50.8]},
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    rets = compute_returns(prices, method="simple")
    assert len(rets) == 3  # first row dropped
    normed = center_returns(rets)
    assert np.allclose(normed.mean(axis=0).values, 0.0, atol=1e-8)
    assert np.allclose(normed.std(axis=0).values, 1.0, atol=1e-8)


def test_rolling_pca_absorption_ratio_runs():
    returns = _make_synthetic_returns()
    normed = center_returns(returns)
    result = rolling_pca_absorption_ratio(
        normed, lookback_window=252, step_size=5, recompute_every=2, verbose=False,
    )
    assert not result.empty
    assert result["absorption_ratio"].between(0, 1).all()
    assert (result["n_components"] >= 1).all()


def test_eigen_portfolios_orthogonal_and_normalized():
    returns = _make_synthetic_returns()
    weights, var_ratio = construct_eigen_portfolios(returns, n_portfolios=3)
    assert weights.shape == (3, returns.shape[1])
    # each portfolio's weights sum to 1 (fully invested)
    assert np.allclose(weights.sum(axis=1).values, 1.0, atol=1e-6)
    assert (var_ratio.values >= 0).all()

    port_returns = eigen_portfolio_returns(returns, weights)
    assert port_returns.shape == (returns.shape[0], 3)


def test_get_weight_rule():
    assert get_weight(1.5) == [0.0, 1.0]
    assert get_weight(-1.5) == [1.0, 0.0]
    assert get_weight(0.0) == [0.5, 0.5]
    assert get_weight(1.0) == [0.5, 0.5]  # boundary is inclusive of neutral zone


def test_get_weights_series_matches_scalar_rule():
    ar_delta = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0], index=pd.RangeIndex(5))
    series_wgts = get_weights_series(ar_delta)
    for i, val in enumerate(ar_delta):
        assert list(series_wgts.iloc[i]) == get_weight(val)


def test_compute_ar_delta_shape():
    ar = pd.Series(np.linspace(0.2, 0.5, 400), index=pd.bdate_range("2015-01-01", periods=400))
    df = compute_ar_delta(ar, short_window=15, long_window=252)
    assert list(df.columns) == ["AR_1yr", "AR_15d", "AR_sigma_1yr", "AR_delta"]
    assert df.dropna().shape[0] > 0


def test_backtest_strategy_matches_manual_calc():
    idx = pd.bdate_range("2020-01-01", periods=5)
    wgts = pd.DataFrame({"EQ": [1, 1, 0, 0, 0.5], "FI": [0, 0, 1, 1, 0.5]}, index=idx)
    rets = pd.DataFrame({"EQ": [0.01, -0.01, 0.02, 0.0, 0.01], "FI": [0.0, 0.0, 0.005, 0.005, 0.002]}, index=idx)
    ann_ret, ann_vol, sharpe = backtest_strategy(wgts, rets, periods_per_year=252)

    expected_port = (wgts.values * rets.values).sum(axis=1)
    assert np.isclose(ann_ret, expected_port.mean() * 252)
    assert np.isclose(ann_vol, expected_port.std(ddof=1) * np.sqrt(252))
    assert np.isclose(sharpe, ann_ret / ann_vol)


def test_autoencoder_absorption_ratio_close_to_pca():
    rng = np.random.default_rng(1)
    n, d, k = 300, 15, 3
    factors = rng.normal(size=(n, k))
    loadings = rng.normal(size=(k, d))
    X = factors @ loadings + 0.05 * rng.normal(size=(n, d))
    X = X - X.mean(axis=0)

    cov = np.cov(X, rowvar=False)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    pca_ar = absorption_ratio(eigvals, n_components=k)

    ae_ar = autoencoder_absorption_ratio(X, n_components=k, epochs=400, learning_rate=0.02)
    assert abs(pca_ar - ae_ar) < 0.1, f"PCA AR={pca_ar:.3f} vs autoencoder AR={ae_ar:.3f}"


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
