#!/usr/bin/env python3
"""Generate notebooks/pca_trading_strategy.ipynb.

Writes the notebook JSON directly (no nbformat dependency needed) so the
project has zero install requirements beyond numpy/pandas/scikit-learn/
matplotlib.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "pca_trading_strategy.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md(
        "# Trading Strategy Based on PCA\n"
        "\n"
        "This notebook uses Principal Component Analysis on S&P 500 constituent\n"
        "returns to:\n"
        "\n"
        "- construct **eigen-portfolios**,\n"
        "- measure market systemic risk with the **Absorption Ratio** (Kritzman,\n"
        "  Li, Page & Rigobon, 2011), and\n"
        "- drive a simple **EQ / FI regime-switching strategy** from shifts in the\n"
        "  Absorption Ratio (\"AR Delta\").\n"
        "\n"
        "All of the underlying logic lives in the `src/pca_strategy` package so it\n"
        "can be unit tested and reused outside the notebook (see `tests/` and\n"
        "`main.py`, which runs the same pipeline end-to-end as a script)."
    ),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "PROJECT_ROOT = Path.cwd()\n"
        "if not (PROJECT_ROOT / \"src\").exists():\n"
        "    PROJECT_ROOT = PROJECT_ROOT.parent  # running from notebooks/\n"
        "sys.path.insert(0, str(PROJECT_ROOT))\n"
        "\n"
        "from src.pca_strategy import benchmarks, plotting\n"
        "from src.pca_strategy.autoencoder import autoencoder_absorption_ratio\n"
        "from src.pca_strategy.data_loader import center_returns, compute_returns, load_asset_prices, split_index_column\n"
        "from src.pca_strategy.eigen_portfolios import construct_eigen_portfolios, eigen_portfolio_returns\n"
        "from src.pca_strategy.pca_analysis import absorption_ratio, rolling_pca_absorption_ratio\n"
        "from src.pca_strategy.strategy import backtest_strategy, compute_ar_delta, count_rebalances, cumulative_growth, get_weights_series\n"
        "from src.pca_strategy.weighting import exponent_weighting\n"
        "\n"
        "%matplotlib inline\n"
        "plt.rcParams[\"figure.figsize\"] = (12, 6)"
    ),
    md(
        "## 1. Data: daily prices of S&P 500 constituents\n"
        "\n"
        "`data/spx_holdings_and_spx_closeprice.csv` holds daily close prices for a\n"
        "subset of S&P 500 constituents from 2000-2013, plus the SPX index level\n"
        "itself in the last column."
    ),
    code(
        "DATA_DIR = PROJECT_ROOT / \"data\"\n"
        "\n"
        "prices = load_asset_prices(DATA_DIR / \"spx_holdings_and_spx_closeprice.csv\")\n"
        "print(\"Asset prices shape:\", prices.shape)\n"
        "prices.iloc[:, :8].head()"
    ),
    md(
        "### Daily returns and standardization\n"
        "\n"
        "PCA is sensitive to scale, so before fitting it we standardize\n"
        "(z-score) each stock's return series -- this way the principal\n"
        "components reflect co-movement between stocks rather than being\n"
        "dominated by whichever names happen to be most volatile."
    ),
    code(
        "returns = compute_returns(prices, method=\"simple\")\n"
        "stock_returns, spx_returns = split_index_column(returns, index_col=\"SPX\")\n"
        "assert \"SPX\" not in stock_returns.columns\n"
        "\n"
        "normed_returns = center_returns(stock_returns)\n"
        "print(f\"Universe: {stock_returns.shape[1]} tickers, {stock_returns.shape[0]} daily returns\")\n"
        "normed_returns.iloc[:, :8].head()"
    ),
    md(
        "## 2. Exponentially-weighted sample weights\n"
        "\n"
        "$$ x_j = e^{-\\frac{\\log 2}{H} j}, \\qquad w_j = \\frac{x_j}{\\sum_k x_k} $$\n"
        "\n"
        "where $H$ is the half-life. This lets a rolling covariance estimate\n"
        "optionally down-weight older observations rather than treating the\n"
        "whole lookback window equally."
    ),
    code(
        "exp_probs = exponent_weighting(252, half_life=252)\n"
        "pd.Series(exp_probs).plot(linewidth=3, title=\"Exponentially-decaying weights (half-life=252d)\")\n"
        "plt.ylabel(\"weight\")\n"
        "plt.show()\n"
        "print(\"sums to\", exp_probs.sum())"
    ),
    md(
        "## 3. Absorption Ratio\n"
        "\n"
        "$$ AR = \\frac{\\sum_{i=1}^{n} \\sigma^2_{E_i}}{\\sum_{j=1}^{N} \\sigma^2_{A_j}} $$\n"
        "\n"
        "the fraction of total variance explained by a fixed number `n` of\n"
        "leading principal components out of `N` total. A high AR means market\n"
        "risk is concentrated in very few common factors -- a fragile, tightly\n"
        "coupled market; a low AR means risk is more diversified across many\n"
        "independent sources."
    ),
    code(
        "print(absorption_ratio.__doc__)"
    ),
    md(
        "### Rolling PCA over the full history\n"
        "\n"
        "We roll a 2-year (504-day) lookback window forward one day at a time,\n"
        "refit PCA on the covariance matrix roughly monthly (`recompute_every=21`)\n"
        "and forward-fill between refits -- a good approximation of daily PCA at a\n"
        "fraction of the compute, and consistent with the original methodology\n"
        "(which also recomputed on a similar cadence)."
    ),
    code(
        "LOOKBACK_WINDOW = 252 * 2\n"
        "VAR_THRESHOLD = 0.80\n"
        "ABSORB_FRACTION = 0.20\n"
        "\n"
        "ar_result = rolling_pca_absorption_ratio(\n"
        "    normed_returns,\n"
        "    lookback_window=LOOKBACK_WINDOW,\n"
        "    step_size=1,\n"
        "    var_threshold=VAR_THRESHOLD,\n"
        "    absorb_fraction=ABSORB_FRACTION,\n"
        "    recompute_every=21,\n"
        ")\n"
        "ts_absorb_ratio = ar_result[\"absorption_ratio\"]\n"
        "ts_pca_components = ar_result[\"n_components\"]\n"
        "\n"
        "ts_absorb_ratio.plot(title=\"Absorption Ratio via PCA\", linewidth=2)\n"
        "plt.ylabel(\"Absorption Ratio\")\n"
        "plt.show()"
    ),
    md(
        "Notice the sharp rise heading into 2008 and the sustained elevated level\n"
        "through the 2011 European debt crisis -- exactly the systemic-risk\n"
        "build-up the Absorption Ratio is designed to flag."
    ),
    code(
        "ts_pca_components.plot(title=f\"# components to explain {int(VAR_THRESHOLD*100)}% of variance\", linewidth=2)\n"
        "plt.ylabel(\"# components\")\n"
        "plt.show()"
    ),
    md(
        "### Cross-check: a linear autoencoder recovers the same Absorption Ratio\n"
        "\n"
        "A single-hidden-layer autoencoder with **no bias terms and no\n"
        "activation function**, trained to minimize mean-squared reconstruction\n"
        "error, learns (up to rotation) the same subspace as PCA restricted to\n"
        "its leading components -- the classical Baldi & Hornik (1989) result.\n"
        "`src/pca_strategy/autoencoder.py` implements this in plain NumPy (full-batch\n"
        "Adam) as a light-weight, dependency-free stand-in for the original\n"
        "TensorFlow `LinearAutoEncoder`, whose `tf.contrib` API no longer exists\n"
        "in modern TensorFlow."
    ),
    code(
        "n_assets = normed_returns.shape[1]\n"
        "absorb_comp = max(1, int(ABSORB_FRACTION * n_assets))\n"
        "\n"
        "last_window = normed_returns.iloc[-LOOKBACK_WINDOW:, :].values\n"
        "last_window = last_window - last_window.mean(axis=0)\n"
        "\n"
        "cov_last = np.cov(last_window, rowvar=False)\n"
        "eigvals_last = np.sort(np.linalg.eigvalsh(cov_last))[::-1]\n"
        "pca_ar_last = absorption_ratio(eigvals_last, absorb_comp)\n"
        "ae_ar_last = autoencoder_absorption_ratio(last_window, n_components=absorb_comp, epochs=300)\n"
        "\n"
        "print(f\"Absorption Ratio via PCA:               {pca_ar_last:.4f}\")\n"
        "print(f\"Absorption Ratio via linear autoencoder: {ae_ar_last:.4f}\")"
    ),
    md(
        "## 4. AR Delta: a standardized shift in the Absorption Ratio\n"
        "\n"
        "Following Kritzman:\n"
        "\n"
        "$$ AR\\delta = \\frac{AR_{15d} - AR_{1yr}}{\\sigma(AR_{1yr})} $$\n"
        "\n"
        "This standardizes the *recent* move in the Absorption Ratio against its\n"
        "own trailing one-year mean and volatility."
    ),
    code(
        "ar_delta_df = compute_ar_delta(ts_absorb_ratio, short_window=15, long_window=252).dropna()\n"
        "\n"
        "fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)\n"
        "ar_delta_df[[\"AR_1yr\", \"AR_15d\"]].plot(ax=axes[0], linewidth=2)\n"
        "axes[0].set_title(\"Absorption Ratio: 1yr vs 15d moving average\")\n"
        "ar_delta_df[\"AR_delta\"].plot(ax=axes[1], linewidth=2, color=\"darkred\")\n"
        "axes[1].axhline(1.0, linestyle=\"--\", color=\"gray\")\n"
        "axes[1].axhline(-1.0, linestyle=\"--\", color=\"gray\")\n"
        "axes[1].set_title(\"AR Delta\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 5. AR-Delta trading strategy\n"
        "\n"
        "Simple regime rule for an EQ / FI portfolio:\n"
        "\n"
        "| Condition | EQ | FI |\n"
        "|---|---|---|\n"
        "| $-1\\sigma < AR\\delta < +1\\sigma$ | 50% | 50% |\n"
        "| $AR\\delta > +1\\sigma$ | 0% | 100% |\n"
        "| $AR\\delta < -1\\sigma$ | 100% | 0% |\n"
        "\n"
        "i.e. shift defensively into fixed income when systemic risk is\n"
        "*rising sharply*, and lean into equities when it is *falling sharply*."
    ),
    code(
        "wgts = get_weights_series(ar_delta_df[\"AR_delta\"], threshold=1.0)\n"
        "wgts.plot.area(title=\"AR-Delta Strategy Allocation\", stacked=True, linewidth=0)\n"
        "plt.ylim(0, 1)\n"
        "plt.show()\n"
        "\n"
        "trades_per_year = count_rebalances(wgts)\n"
        "print(\"Average number of trades per year: %.2f\" % trades_per_year.mean())\n"
        "trades_per_year"
    ),
    md(
        "## 6. Backtest vs. a static 50/50 EQ/FI portfolio\n"
        "\n"
        "The original course project benchmarked against real VTI (equity) and\n"
        "AGG (bond) ETF returns supplied as separate files. Those files aren't\n"
        "part of this project's single data source, so by default `EQ` here is\n"
        "the SPX index (already in our dataset) and `FI` falls back to a\n"
        "clearly-labeled **synthetic**, low-volatility bond-like proxy --\n"
        "see `src/pca_strategy/benchmarks.py`. Drop a real `data/eq_fi_returns.csv`\n"
        "(columns `EQ`, `FI`) to backtest against actual market data instead."
    ),
    code(
        "eq_fi_returns, used_real_data = benchmarks.load_or_build_eq_fi_returns(spx_returns, DATA_DIR)\n"
        "print(\"Using real eq_fi_returns.csv override\" if used_real_data else \"Using SPX + synthetic FI proxy\")\n"
        "\n"
        "ann_ret, ann_vol, sharpe = backtest_strategy(wgts, eq_fi_returns)\n"
        "print(f\"AR-Delta strategy : return={ann_ret:+.4f}  vol={ann_vol:.4f}  Sharpe={sharpe:.3f}\")\n"
        "\n"
        "eq_wgts = wgts.copy()\n"
        "eq_wgts.iloc[:, :] = 0.5\n"
        "ann_ret_bh, ann_vol_bh, sharpe_bh = backtest_strategy(eq_wgts, eq_fi_returns)\n"
        "print(f\"Static 50/50      : return={ann_ret_bh:+.4f}  vol={ann_vol_bh:.4f}  Sharpe={sharpe_bh:.3f}\")"
    ),
    code(
        "common_idx = wgts.index.intersection(eq_fi_returns.index)\n"
        "strat_ret = (wgts.loc[common_idx].values * eq_fi_returns.loc[common_idx, [\"EQ\", \"FI\"]].values).sum(axis=1)\n"
        "bh_ret = (eq_wgts.loc[common_idx].values * eq_fi_returns.loc[common_idx, [\"EQ\", \"FI\"]].values).sum(axis=1)\n"
        "\n"
        "growth = pd.DataFrame({\n"
        "    \"AR-Delta strategy\": cumulative_growth(pd.Series(strat_ret, index=common_idx)),\n"
        "    \"Static 50/50\": cumulative_growth(pd.Series(bh_ret, index=common_idx)),\n"
        "})\n"
        "growth.plot(title=\"Cumulative Growth of $1\", linewidth=2)\n"
        "plt.show()"
    ),
    md(
        "## 7. Eigen-portfolios\n"
        "\n"
        "Each principal component of the (standardized) return correlation\n"
        "matrix defines an *eigen-portfolio*: convert the eigenvector to raw-asset\n"
        "weights by dividing by each stock's volatility, then rescale to sum to 1\n"
        "(Avellaneda & Lee, 2010). By construction, eigen-portfolios are mutually\n"
        "uncorrelated over the estimation window.\n"
        "\n"
        "The first eigen-portfolio typically resembles a broad market-cap-like\n"
        "basket and tracks the benchmark closely; later ones pick up\n"
        "progressively more specific, sector- or style-like sources of\n"
        "co-movement."
    ),
    code(
        "fit_window = stock_returns.iloc[-LOOKBACK_WINDOW:, :]\n"
        "ep_weights, ep_var_ratio = construct_eigen_portfolios(fit_window, n_portfolios=5)\n"
        "print(ep_var_ratio)\n"
        "\n"
        "ep_var_ratio.plot(kind=\"bar\", title=\"Variance Explained by Each Eigen-Portfolio\")\n"
        "plt.show()"
    ),
    code(
        "ep_returns = eigen_portfolio_returns(stock_returns, ep_weights)\n"
        "ep_growth = pd.DataFrame({col: cumulative_growth(ep_returns[col]) for col in ep_returns.columns})\n"
        "ep_growth[\"SPX\"] = cumulative_growth(spx_returns.loc[ep_returns.index])\n"
        "ep_growth.plot(title=\"Eigen-Portfolios vs. SPX (weights fit on most recent 2yr window)\", linewidth=1.5)\n"
        "plt.show()\n"
        "\n"
        "corr_ep1_spx = ep_returns[\"EP1\"].corr(spx_returns.loc[ep_returns.index])\n"
        "print(f\"Correlation of EP1 with SPX: {corr_ep1_spx:.3f}\")"
    ),
    md(
        "## Next steps\n"
        "\n"
        "Ideas for extending this further:\n"
        "\n"
        "- Swap the synthetic FI proxy for real ETF (e.g. AGG) or Treasury\n"
        "  returns in `data/eq_fi_returns.csv`.\n"
        "- Use `rolling_out_of_sample_eigen_portfolios` in\n"
        "  `src/pca_strategy/eigen_portfolios.py` for a walk-forward\n"
        "  (no-look-ahead) eigen-portfolio backtest.\n"
        "- Try `use_ewm=True` in `rolling_pca_absorption_ratio` to compare\n"
        "  equally-weighted vs. exponentially-weighted covariance estimates.\n"
        "- Sweep `AR_DELTA_THRESHOLD` / `AR_SHORT_WINDOW` / `AR_LONG_WINDOW` and\n"
        "  compare Sharpe ratios out of sample."
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, indent=1))
print("Wrote", NB_PATH)
