#!/usr/bin/env python3
"""
End-to-end pipeline for the IRL-based market portfolio project.

Part 1 -- calibrate the GMR/IRL model on DJI-30 data with baseline SMA
          signals, at increasing levels of cross-sectional pooling.
Part 2 -- propose and evaluate alternative signals.
Part 3 -- repeat the analysis on the (much larger) real S&P 500 universe.
Part 4 -- turn the fitted model into a trading strategy and compare it
          against the PCA / Absorption-Ratio strategy from Course 2.

Run with:

    python main.py

Reads data/dja_cap.csv and data/spx_holdings_and_spx_closeprice.csv, writes
plots + CSV results into results/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.irl_market import plotting
from src.irl_market.data_loader import (
    DJI_BASE_CAP_BILLIONS,
    load_dja_caps,
    load_spx_caps,
    normalize_levels,
    sector_map_for,
)
from src.irl_market.estimation import (
    compare_fits,
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
from src.irl_market.signals import BASELINE_SIGNALS, EXTENDED_SIGNALS, build_signal_panels, sma_deviation

# Course-2 PCA / Absorption-Ratio package, vendored for the Part 4 comparison.
from src.pca_strategy import pca_analysis as pca
from src.pca_strategy import strategy as pca_strategy_mod
from src.pca_strategy.data_loader import center_returns as pca_center_returns
from src.pca_strategy.data_loader import compute_returns as pca_compute_returns

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

NORMALIZE_WINDOW = 30
SMA_SHORT, SMA_LONG = 10, 30


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ==================================================================
    section("Part 0. Load DJI-30 data & build baseline signals")
    # ==================================================================
    dja_path = DATA_DIR / "dja_cap.csv"
    if not dja_path.exists():
        raise FileNotFoundError(
            f"{dja_path} not found. Run `python scripts/generate_dja_data.py` first "
            f"(or drop a real dja_cap.csv there)."
        )
    df_cap = load_dja_caps(dja_path)
    print(f"DJI panel: {df_cap.shape[0]} days x {df_cap.shape[1]} tickers "
          f"({df_cap.index[0].date()} -> {df_cap.index[-1].date()})")

    short_rolling = df_cap.rolling(SMA_SHORT).mean()
    long_rolling = df_cap.rolling(SMA_LONG).mean()
    plotting.plot_level_with_smas(
        df_cap, short_rolling, long_rolling, ticker="AAPL",
        start_date="2015-01-01", end_date="2017-09-01",
        out_path=RESULTS_DIR / "dji_aapl_smas.png",
        window_short=SMA_SHORT, window_long=SMA_LONG, ylabel="Cap in $",
    )

    x_norm, baseline = normalize_levels(df_cap, window=NORMALIZE_WINDOW)
    baseline_signals = build_signal_panels(x_norm, BASELINE_SIGNALS)

    # ==================================================================
    section("Part 1. Model calibration with SMA signals (DJI-30)")
    # ==================================================================
    fit_pooled = fit_gmr_pooled(x_norm, baseline_signals)
    fit_per_asset = fit_gmr_per_asset(x_norm, baseline_signals)
    sectors = sector_map_for(x_norm.columns)
    fit_per_sector = fit_gmr_per_group(x_norm, baseline_signals, sectors)

    for fit in (fit_pooled, fit_per_sector, fit_per_asset):
        print(fit.summary())

    part1_table = compare_fits({"pooled": fit_pooled, "per_sector": fit_per_sector, "per_asset": fit_per_asset})
    part1_table.to_csv(RESULTS_DIR / "part1_pooling_comparison.csv")
    print("\nPooling-level comparison (lower BIC/AIC = better penalized fit):")
    print(part1_table.round(2).to_string())

    plotting.plot_residual_hist(fit_pooled.residuals, "Part 1: Pooled-fit Residuals (DJI-30)",
                                 RESULTS_DIR / "part1_residuals_pooled.png")
    plotting.plot_kappa_by_asset(fit_per_asset.kappa, "Part 1: Per-Asset Kappa (DJI-30)",
                                  RESULTS_DIR / "part1_kappa_per_asset.png")
    plotting.plot_sigma_heatmap(fit_pooled.Sigma, "Part 1: Residual Covariance (DJI-30, pooled fit)",
                                 RESULTS_DIR / "part1_sigma_heatmap.png")
    plotting.plot_model_comparison(part1_table, RESULTS_DIR / "part1_model_comparison.png")

    fit_per_asset.kappa.to_csv(RESULTS_DIR / "part1_kappa_per_asset.csv")
    fit_per_asset.W.to_csv(RESULTS_DIR / "part1_W_per_asset.csv")
    fit_per_sector.group_table.to_csv(RESULTS_DIR / "part1_group_table_per_sector.csv")

    # ==================================================================
    section("Part 2. Alternative signals")
    # ==================================================================
    extended_signals = build_signal_panels(x_norm, EXTENDED_SIGNALS)

    part2_fits = {"baseline (sma10+sma30)": fit_gmr_pooled(x_norm, baseline_signals)}
    for name in ["mom20", "vol20", "dd60", "z60"]:
        combo = {**baseline_signals, name: extended_signals[name]}
        part2_fits[f"baseline + {name}"] = fit_gmr_pooled(x_norm, combo)
    part2_fits["all signals"] = fit_gmr_pooled(x_norm, extended_signals)

    part2_table = compare_fits(part2_fits)
    part2_table.to_csv(RESULTS_DIR / "part2_signal_comparison.csv")
    print("Signal-set comparison, pooled fits (lower BIC = better penalized fit):")
    print(part2_table.round(2).to_string())

    best_signal_set = part2_table["BIC"].idxmin()
    print(f"\nBest signal set by BIC: {best_signal_set!r}")
    plotting.plot_model_comparison(part2_table, RESULTS_DIR / "part2_signal_comparison.png")

    # ==================================================================
    section("Part 3. Repeat the analysis on the S&P 500 universe")
    # ==================================================================
    spx_prices, spx_index = load_spx_caps(DATA_DIR / "spx_holdings_and_spx_closeprice.csv")
    print(f"S&P universe: {spx_prices.shape[0]} days x {spx_prices.shape[1]} tickers "
          f"({spx_prices.index[0].date()} -> {spx_prices.index[-1].date()})")

    # The S&P universe spans 14 years with huge cross-sectional dispersion in
    # long-run performance (some names 10x, others nearly delist) -- a single
    # early-window baseline lets x_t range over two orders of magnitude and
    # destabilizes the pooled regression (see data_loader.normalize_levels).
    # A rolling baseline keeps the normalized state O(1) throughout.
    SPX_NORMALIZE_WINDOW = 252
    spx_norm, spx_baseline = normalize_levels(spx_prices, window=SPX_NORMALIZE_WINDOW, method="rolling")
    spx_signals = build_signal_panels(spx_norm, BASELINE_SIGNALS)

    spx_fit_pooled = fit_gmr_pooled(spx_norm, spx_signals)
    print(spx_fit_pooled.summary())
    print("Recovered W (pooled, S&P 500):")
    print(spx_fit_pooled.W)

    # per-asset is a simple per-column OLS loop -- fast even for 400+ names
    spx_fit_per_asset = fit_gmr_per_asset(spx_norm, spx_signals)
    print(spx_fit_per_asset.summary())

    part3_table = compare_fits({"pooled": spx_fit_pooled, "per_asset": spx_fit_per_asset})
    part3_table.to_csv(RESULTS_DIR / "part3_spx_comparison.csv")
    print(part3_table.round(2).to_string())

    plotting.plot_residual_hist(spx_fit_pooled.residuals, "Part 3: Pooled-fit Residuals (S&P 500)",
                                 RESULTS_DIR / "part3_residuals_pooled.png")
    spx_fit_per_asset.kappa.sort_values().to_csv(RESULTS_DIR / "part3_kappa_per_asset.csv")

    print(f"\nDJI pooled kappa={fit_pooled.kappa:.4f}  vs.  S&P pooled kappa={spx_fit_pooled.kappa:.4f}")
    print("(A larger, more diversified universe typically shows a smaller, more "
          "stable pooled mean-reversion estimate -- individual-name idiosyncrasies "
          "average out across 400+ names in a way they don't across 30.)")

    # ==================================================================
    section("Part 4. IRL-implied trading strategy vs. PCA / Absorption Ratio (Course 2)")
    # ==================================================================
    # --- 4a. In-sample IRL-implied long/short policy on the S&P universe ---
    X_t, X_next, signals_t = prepare_regression_arrays(spx_norm, spx_signals)
    target = compute_target(signals_t, spx_fit_pooled.W)
    z = mispricing_zscore(target, X_t)
    weights = dollar_neutral_weights(z)
    ann_ret_is, ann_vol_is, sharpe_is, port_ret_is = backtest_policy(weights, X_t, X_next)
    print(f"IRL long/short policy (in-sample)  : ann. return={ann_ret_is:+.4f}  "
          f"ann. vol={ann_vol_is:.4f}  Sharpe={sharpe_is:.3f}")

    # --- 4b. Walk-forward (out-of-sample) version: fit on the first 70%, trade the rest ---
    split = int(0.7 * len(spx_norm))
    train_dates, test_dates = spx_norm.index[:split], spx_norm.index[split:]
    spx_norm_train = spx_norm.loc[:train_dates[-1]]
    train_signals = build_signal_panels(spx_norm_train, BASELINE_SIGNALS)
    oos_fit = fit_gmr_pooled(spx_norm_train, train_signals)

    X_t_oos, X_next_oos, signals_t_oos = prepare_regression_arrays(spx_norm, spx_signals)
    oos_mask = X_t_oos.index >= test_dates[0]
    X_t_test = X_t_oos.loc[oos_mask]
    X_next_test = X_next_oos.iloc[np.where(oos_mask)[0]]
    signals_t_test = {k: v.loc[oos_mask] for k, v in signals_t_oos.items()}

    target_oos = compute_target(signals_t_test, oos_fit.W)
    z_oos = mispricing_zscore(target_oos, X_t_test)
    weights_oos = dollar_neutral_weights(z_oos)
    ann_ret_oos, ann_vol_oos, sharpe_oos, port_ret_oos = backtest_policy(weights_oos, X_t_test, X_next_test)
    print(f"IRL long/short policy (walk-forward OOS): ann. return={ann_ret_oos:+.4f}  "
          f"ann. vol={ann_vol_oos:.4f}  Sharpe={sharpe_oos:.3f}")
    print(f"  (trained on {train_dates[0].date()}..{train_dates[-1].date()}, "
          f"traded on {test_dates[0].date()}..{test_dates[-1].date()})")

    turnover = weights_oos.diff().abs().sum(axis=1).dropna()
    print(f"\nGross Sharpe above is frictionless and daily-rebalanced across "
          f"{weights_oos.shape[1]} names (avg. daily turnover = {turnover.mean():.2f}); "
          f"a wide, high-breadth short-horizon reversal signal like this is well known to "
          f"show an unrealistically high *paper* Sharpe -- Grinold's fundamental law of "
          f"active management says even a weak per-name edge compounds into a large "
          f"aggregate Sharpe purely from breadth (~400 names x ~1000 rebalances). "
          f"Charging a simple proportional cost on turnover shows how fast that erodes:")
    cost_rows = []
    for bps in [0, 2, 5, 10, 20, 50]:
        ar_, av_, sh_, _ = backtest_policy(weights_oos, X_t_test, X_next_test, cost_bps=bps)
        cost_rows.append({"cost_bps": bps, "ann_return": ar_, "ann_vol": av_, "sharpe": sh_})
    cost_table = pd.DataFrame(cost_rows).set_index("cost_bps")
    cost_table.to_csv(RESULTS_DIR / "part4_cost_sensitivity.csv")
    print(cost_table.round(4).to_string())

    # --- 4c. Course-2 PCA / Absorption-Ratio AR-Delta EQ/FI strategy, same OOS window ---
    from src.pca_strategy import benchmarks as pca_benchmarks

    spx_stock_returns = pca_compute_returns(spx_prices, method="simple")
    spx_index_returns = spx_index.pct_change().iloc[1:]
    normed_spx_returns = pca_center_returns(spx_stock_returns)

    ar_result = pca.rolling_pca_absorption_ratio(
        normed_spx_returns, lookback_window=252 * 2, step_size=1,
        var_threshold=0.80, absorb_fraction=0.20, recompute_every=21, verbose=False,
    )
    ar_delta_df = pca_strategy_mod.compute_ar_delta(ar_result["absorption_ratio"], 15, 252).dropna()
    ar_wgts = pca_strategy_mod.get_weights_series(ar_delta_df["AR_delta"], threshold=1.0)

    eq_fi_returns, used_real = pca_benchmarks.load_or_build_eq_fi_returns(spx_index_returns, DATA_DIR)
    ar_wgts_test = ar_wgts.loc[ar_wgts.index.intersection(test_dates)]
    ann_ret_ar, ann_vol_ar, sharpe_ar = pca_strategy_mod.backtest_strategy(ar_wgts_test, eq_fi_returns)
    print(f"Course-2 AR-Delta EQ/FI (same OOS window): ann. return={ann_ret_ar:+.4f}  "
          f"ann. vol={ann_vol_ar:.4f}  Sharpe={sharpe_ar:.3f}"
          + ("" if used_real else "  [FI leg is the synthetic proxy from Course 2, see its README]"))

    comparison = pd.DataFrame(
        {
            "ann_return": [ann_ret_is, ann_ret_oos, ann_ret_ar],
            "ann_vol": [ann_vol_is, ann_vol_oos, ann_vol_ar],
            "sharpe": [sharpe_is, sharpe_oos, sharpe_ar],
        },
        index=["IRL_long_short_in_sample", "IRL_long_short_OOS", "Course2_AR_Delta_EQ_FI_OOS_window"],
    )
    comparison.to_csv(RESULTS_DIR / "part4_strategy_comparison.csv")
    print("\nStrategy comparison:")
    print(comparison.round(4).to_string())

    curves = {
        "IRL long/short (OOS)": (1.0 + port_ret_oos).cumprod(),
        "IRL long/short (in-sample)": (1.0 + port_ret_is).cumprod(),
    }
    plotting.plot_cumulative_growth(curves, RESULTS_DIR / "part4_irl_policy_growth.png",
                                     title="IRL-Implied Long/Short Policy: Cumulative Growth of $1")

    section("Done")
    print(f"All results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
