#!/usr/bin/env python3
"""
End-to-end pipeline: PCA-based eigen-portfolios, Absorption Ratio, and the
AR-Delta EQ/FI trading strategy on S&P 500 constituent data.

Run with:

    python main.py

Reads ``data/spx_holdings_and_spx_closeprice.csv`` and writes plots + CSV
results into ``results/``.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.pca_strategy import benchmarks, plotting
from src.pca_strategy.autoencoder import autoencoder_absorption_ratio
from src.pca_strategy.data_loader import (
    center_returns,
    compute_returns,
    load_asset_prices,
    split_index_column,
)
from src.pca_strategy.eigen_portfolios import (
    construct_eigen_portfolios,
    eigen_portfolio_returns,
)
from src.pca_strategy.pca_analysis import absorption_ratio, rolling_pca_absorption_ratio
from src.pca_strategy.strategy import (
    backtest_strategy,
    compute_ar_delta,
    count_rebalances,
    cumulative_growth,
    get_weights_series,
)
from src.pca_strategy.weighting import exponent_weighting

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Hyper-parameters (mirroring the original study's choices) -----------
LOOKBACK_WINDOW = 252 * 2   # 2 years of daily data
HALF_LIFE = 252             # for the optional exponentially-weighted covariance
VAR_THRESHOLD = 0.80        # fraction of variance defining "n_components"
ABSORB_FRACTION = 0.20      # top 20% of components used in the Absorption Ratio
AR_SHORT_WINDOW = 15        # days
AR_LONG_WINDOW = 252        # days
AR_DELTA_THRESHOLD = 1.0    # +/- 1 std dev trigger for the trading rule
PCA_RECOMPUTE_EVERY = 21    # refit PCA ~monthly, forward-fill between refits
N_EIGEN_PORTFOLIOS = 5


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    section("1. Load data & compute returns")
    # ------------------------------------------------------------------
    prices = load_asset_prices(DATA_DIR / "spx_holdings_and_spx_closeprice.csv")
    print(f"Loaded prices: {prices.shape[0]} days x {prices.shape[1]} tickers "
          f"({prices.index[0].date()} -> {prices.index[-1].date()})")

    returns = compute_returns(prices, method="simple")
    stock_returns, spx_returns = split_index_column(returns, index_col="SPX")
    assert "SPX" not in stock_returns.columns

    normed_returns = center_returns(stock_returns)
    print(f"Stock universe: {stock_returns.shape[1]} tickers, "
          f"{stock_returns.shape[0]} daily return observations")

    # ------------------------------------------------------------------
    section("2. Exponentially-decaying sample weights")
    # ------------------------------------------------------------------
    exp_probs = exponent_weighting(LOOKBACK_WINDOW, HALF_LIFE)
    plotting.plot_series(
        pd.Series(exp_probs, name="weight"),
        title=f"Exponentially-Decaying Sample Weights (half-life={HALF_LIFE}d)",
        out_path=RESULTS_DIR / "exponential_weights.png",
        ylabel="weight",
    )
    print(f"exp_probs sums to {exp_probs.sum():.6f} (should be 1.0)")

    # ------------------------------------------------------------------
    section("3. Rolling PCA & Absorption Ratio (market systemic risk)")
    # ------------------------------------------------------------------
    ar_result = rolling_pca_absorption_ratio(
        normed_returns,
        lookback_window=LOOKBACK_WINDOW,
        step_size=1,
        var_threshold=VAR_THRESHOLD,
        absorb_fraction=ABSORB_FRACTION,
        recompute_every=PCA_RECOMPUTE_EVERY,
    )
    ts_absorb_ratio = ar_result["absorption_ratio"]
    ts_pca_components = ar_result["n_components"]
    ar_result.to_csv(RESULTS_DIR / "absorption_ratio_timeseries.csv")

    plotting.plot_series(
        ts_absorb_ratio,
        title="Absorption Ratio via PCA (top 20% of components)",
        out_path=RESULTS_DIR / "absorption_ratio.png",
        ylabel="Absorption Ratio",
    )
    plotting.plot_series(
        ts_pca_components,
        title=f"Number of Principal Components Explaining {int(VAR_THRESHOLD * 100)}% of Variance",
        out_path=RESULTS_DIR / "pca_components.png",
        ylabel="# components",
    )
    print(f"Absorption Ratio: mean={ts_absorb_ratio.mean():.3f}, "
          f"min={ts_absorb_ratio.min():.3f}, max={ts_absorb_ratio.max():.3f}")

    # ------------------------------------------------------------------
    section("3b. Cross-check: Absorption Ratio via a linear autoencoder")
    # ------------------------------------------------------------------
    n_assets = normed_returns.shape[1]
    absorb_comp = max(1, int(ABSORB_FRACTION * n_assets))
    last_window = normed_returns.iloc[-LOOKBACK_WINDOW:, :].values
    last_window = last_window - last_window.mean(axis=0)  # center for the AE's variance calc

    cov_last = np.cov(last_window, rowvar=False)
    eigvals_last = np.sort(np.linalg.eigvalsh(cov_last))[::-1]
    pca_ar_last = absorption_ratio(eigvals_last, absorb_comp)

    ae_ar_last = autoencoder_absorption_ratio(last_window, n_components=absorb_comp, epochs=300)
    print(f"Most recent {LOOKBACK_WINDOW}-day window -- "
          f"Absorption Ratio via PCA: {pca_ar_last:.4f} | via linear autoencoder: {ae_ar_last:.4f}")
    print("(A linear, bias-free autoencoder should recover ~the same value as PCA -- Baldi & Hornik, 1989.)")

    # ------------------------------------------------------------------
    section("4. AR Delta: standardized shift in the Absorption Ratio")
    # ------------------------------------------------------------------
    ar_delta_df = compute_ar_delta(ts_absorb_ratio, AR_SHORT_WINDOW, AR_LONG_WINDOW).dropna()
    ar_delta_df.to_csv(RESULTS_DIR / "ar_delta.csv")
    plotting.plot_ar_delta(ar_delta_df, RESULTS_DIR / "ar_delta.png")
    print(f"AR Delta series: {len(ar_delta_df)} observations, "
          f"std={ar_delta_df['AR_delta'].std():.3f}")

    # ------------------------------------------------------------------
    section("5. AR-Delta EQ/FI trading strategy")
    # ------------------------------------------------------------------
    wgts = get_weights_series(ar_delta_df["AR_delta"], threshold=AR_DELTA_THRESHOLD)
    wgts.to_csv(RESULTS_DIR / "strategy_weights.csv")
    plotting.plot_weights(wgts, RESULTS_DIR / "strategy_weights.png")

    trades_per_year = count_rebalances(wgts)
    trades_per_year.to_csv(RESULTS_DIR / "trades_per_year.csv")
    print(f"Average number of trades per year: {trades_per_year.mean():.2f}")
    print(trades_per_year.to_string())

    # ------------------------------------------------------------------
    section("6. Backtest: AR-Delta strategy vs. static 50/50 EQ/FI")
    # ------------------------------------------------------------------
    eq_fi_returns, used_real_data = benchmarks.load_or_build_eq_fi_returns(spx_returns, DATA_DIR)
    if used_real_data:
        print("Using data/eq_fi_returns.csv for EQ/FI returns.")
    else:
        print("No data/eq_fi_returns.csv found -- using SPX as EQ and a SYNTHETIC "
              "bond-like proxy as FI (see src/pca_strategy/benchmarks.py). Drop a real "
              "EQ/FI return series at that path to backtest against actual ETF data.")

    ann_ret, ann_vol, sharpe = backtest_strategy(wgts, eq_fi_returns)
    print(f"AR-Delta strategy   : ann. return={ann_ret:+.4f}  ann. vol={ann_vol:.4f}  Sharpe={sharpe:.3f}")

    eq_wgts = wgts.copy()
    eq_wgts.iloc[:, :] = 0.5
    ann_ret_bh, ann_vol_bh, sharpe_bh = backtest_strategy(eq_wgts, eq_fi_returns)
    print(f"Static 50/50 EQ/FI  : ann. return={ann_ret_bh:+.4f}  ann. vol={ann_vol_bh:.4f}  Sharpe={sharpe_bh:.3f}")

    summary = pd.DataFrame(
        {
            "ann_return": [ann_ret, ann_ret_bh],
            "ann_vol": [ann_vol, ann_vol_bh],
            "sharpe": [sharpe, sharpe_bh],
        },
        index=["AR_delta_strategy", "static_50_50"],
    )
    summary.to_csv(RESULTS_DIR / "performance_summary.csv")

    common_idx = wgts.index.intersection(eq_fi_returns.index)
    strat_port_ret = (wgts.loc[common_idx].values * eq_fi_returns.loc[common_idx, ["EQ", "FI"]].values).sum(axis=1)
    bh_port_ret = (eq_wgts.loc[common_idx].values * eq_fi_returns.loc[common_idx, ["EQ", "FI"]].values).sum(axis=1)
    curves = {
        "AR-Delta strategy": cumulative_growth(pd.Series(strat_port_ret, index=common_idx)),
        "Static 50/50": cumulative_growth(pd.Series(bh_port_ret, index=common_idx)),
    }
    plotting.plot_cumulative_growth(curves, RESULTS_DIR / "cumulative_growth.png")

    # ------------------------------------------------------------------
    section("7. Eigen-portfolios")
    # ------------------------------------------------------------------
    fit_window = stock_returns.iloc[-LOOKBACK_WINDOW:, :]
    ep_weights, ep_var_ratio = construct_eigen_portfolios(fit_window, n_portfolios=N_EIGEN_PORTFOLIOS)
    ep_weights.to_csv(RESULTS_DIR / "eigen_portfolio_weights.csv")
    ep_var_ratio.to_csv(RESULTS_DIR / "eigen_portfolio_explained_variance.csv")
    plotting.plot_eigen_portfolio_variance(ep_var_ratio, RESULTS_DIR / "eigen_portfolio_variance.png")
    print(ep_var_ratio.to_string())

    ep_full_history_returns = eigen_portfolio_returns(stock_returns, ep_weights)
    ep_curves = {col: cumulative_growth(ep_full_history_returns[col]) for col in ep_full_history_returns.columns}
    ep_curves["SPX"] = cumulative_growth(spx_returns.loc[ep_full_history_returns.index])
    plotting.plot_cumulative_growth(
        ep_curves, RESULTS_DIR / "eigen_portfolio_growth.png",
        title="Eigen-Portfolios vs. SPX (weights fit on most recent 2yr window)",
    )
    ep1_corr = ep_full_history_returns["EP1"].corr(spx_returns.loc[ep_full_history_returns.index])
    print(f"Correlation of EP1 (first eigen-portfolio) with SPX: {ep1_corr:.3f} "
          f"(expected to be high -- EP1 typically resembles the broad market)")

    section("Done")
    print(f"All results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
