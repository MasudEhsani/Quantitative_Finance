# PCA Trading Strategy: Eigen-Portfolios & Absorption Ratio

A self-contained project that uses Principal Component Analysis on a
universe of S&P 500 stock returns to:

1. construct **eigen-portfolios** -- orthogonal baskets of stocks derived
   directly from the eigenvectors of the return correlation matrix,
2. measure **market systemic risk** over time with the **Absorption
   Ratio** (AR), following Kritzman, Li, Page & Rigobon (2011), and
3. drive a simple **regime-switching Equity / Fixed-Income strategy**
   ("AR Delta") from sharp shifts in the Absorption Ratio.

This started from a Coursera course-project notebook and has been rebuilt
as a proper, tested Python project: the grading/submission scaffolding is
gone, the code is organized into a reusable package, the previously-blank
exercise stubs (`backtest_strategy`, the AR-Delta rule) are fully
implemented, and it's extended with explicit eigen-portfolio construction,
a dependency-free NumPy autoencoder (replacing the original's TensorFlow
`tf.contrib` code, which no longer exists in modern TensorFlow), unit
tests, and a runnable CLI pipeline.

## Project layout

```
pca-trading-strategy/
├── data/
│   └── spx_holdings_and_spx_closeprice.csv   # daily prices, ~418 tickers + SPX, 2000-2013
├── src/pca_strategy/
│   ├── data_loader.py        # load prices, compute & standardize returns
│   ├── weighting.py          # exponentially-decaying sample weights
│   ├── pca_analysis.py       # rolling PCA, Absorption Ratio
│   ├── eigen_portfolios.py   # construct & backtest eigen-portfolios
│   ├── autoencoder.py        # linear autoencoder (PCA via NumPy, no TF dependency)
│   ├── strategy.py           # AR-Delta signal, position sizing, backtest
│   ├── benchmarks.py         # EQ/FI return series for the strategy backtest
│   └── plotting.py           # matplotlib helpers
├── notebooks/
│   └── pca_trading_strategy.ipynb   # walkthrough notebook (same pipeline as main.py)
├── tests/
│   └── test_pca_strategy.py  # unit tests (plain asserts; also pytest-compatible)
├── main.py                   # end-to-end CLI pipeline -> results/
├── results/                  # populated by main.py (plots + CSVs)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

Only `numpy`, `pandas`, `scikit-learn`, and `matplotlib` are required to
run the pipeline and tests; `pytest` and `jupyter` are optional
conveniences.

## Running it

```bash
python main.py            # runs the full pipeline, writes plots/CSVs to results/
python tests/test_pca_strategy.py   # or: pytest tests/
jupyter notebook notebooks/pca_trading_strategy.ipynb
```

`main.py` prints a short narrative of each stage and writes its outputs to
`results/`: the Absorption Ratio time series and plot, the AR-Delta signal
and plot, strategy weights and rebalance counts, a strategy-vs-benchmark
performance summary and growth-of-$1 chart, and eigen-portfolio weights /
explained-variance / growth charts.

## Method summary

**Absorption Ratio.** Standardize returns, take a rolling lookback window
(default 2 years), fit PCA on the window's covariance matrix, and compute

```
AR = sum(explained_variance[:n]) / sum(explained_variance)
```

for a fixed `n` (default: top 20% of components). A high AR means a small
number of common factors explain most of the market's variance -- a
tightly-coupled, fragile state; a low AR means risk is more diversified.
Running this over 2000-2013 on the included data reproduces the expected
pattern: AR rises sharply heading into the 2008 financial crisis and stays
elevated through the 2011 European debt crisis.

**AR Delta.**

```
AR_delta = (AR_15d - AR_1yr) / std(AR_1yr)
```

a standardized measure of how much the Absorption Ratio has moved
recently relative to its own trailing year. The strategy in
`strategy.get_weight` / `get_weights_series` allocates:

| AR Delta | Equity | Fixed Income |
|---|---|---|
| between -1σ and +1σ | 50% | 50% |
| above +1σ (risk rising fast) | 0% | 100% |
| below -1σ (risk falling fast) | 100% | 0% |

On the included data this trades only ~2-3 times per year and, in
particular, rotates out of equities heading into the 2008 drawdown (see
`results/cumulative_growth.png` after running `main.py`).

**Eigen-portfolios.** Each eigenvector of the (standardized) return
correlation matrix, rescaled by asset volatility and renormalized to sum
to 1, defines a fully-invested "eigen-portfolio" (Avellaneda & Lee, 2010).
The first one is typically highly correlated with the broad market (>0.97
correlation with SPX on this data); later ones capture progressively more
specific sources of common variation. `eigen_portfolios.py` also includes
a walk-forward, look-ahead-free variant
(`rolling_out_of_sample_eigen_portfolios`) that refits weights
periodically and applies them only to *subsequent* returns.

**Linear autoencoder cross-check.** A single-hidden-layer autoencoder with
no bias terms and no activation function, trained by minimizing
mean-squared reconstruction error, recovers (up to rotation) the same
subspace as PCA restricted to its leading components -- the classical
Baldi & Hornik (1989) result. `autoencoder.py` implements this training
loop in plain NumPy (full-batch Adam) purely to cross-check the PCA-based
Absorption Ratio, since the original notebook's TensorFlow 1.x
`tf.contrib.layers` API is no longer available in current TensorFlow.

## A note on the EQ/FI backtest data

The original course project benchmarked the strategy against real VTI
(equity) and AGG (bond) ETF daily returns, distributed as separate CSV
files that aren't part of this project's data source. To keep this project
runnable from just `data/spx_holdings_and_spx_closeprice.csv`:

- `EQ` returns are the SPX index column already present in that file.
- `FI` returns fall back to a **synthetic**, low-volatility bond-like
  proxy (`benchmarks.synthesize_fi_returns`) -- seeded Gaussian daily
  returns calibrated to plausible aggregate-bond return/volatility, and
  explicitly *not* fit to real market data.

`main.py` prints a warning whenever it falls back to the synthetic proxy.
To backtest against real data instead, drop a CSV at
`data/eq_fi_returns.csv` with a date index and columns `EQ`, `FI` (e.g.
actual VTI/AGG daily returns) -- `benchmarks.load_or_build_eq_fi_returns`
picks it up automatically and the synthetic fallback is skipped.

## Extending this

- Swap in real EQ/FI (or other asset pair) returns as described above.
- Try `use_ewm=True` in `rolling_pca_absorption_ratio` to compare an
  exponentially-weighted covariance estimate against the equally-weighted
  default.
- Use `rolling_out_of_sample_eigen_portfolios` for a proper walk-forward
  eigen-portfolio backtest (e.g. long the first eigen-portfolio, short
  SPX, as a statistical-arbitrage-style spread).
- Sweep `AR_DELTA_THRESHOLD`, `AR_SHORT_WINDOW`, and `AR_LONG_WINDOW` and
  compare out-of-sample Sharpe ratios.
- Extend `get_weight` beyond a 3-bucket step function to a continuous
  position size (e.g. proportional to `AR_delta`, capped).

## References

- Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2011). "Principal
  Components as a Measure of Systemic Risk." *Journal of Portfolio
  Management*.
- Avellaneda, M., & Lee, J.-H. (2010). "Statistical Arbitrage in the U.S.
  Equities Market." *Quantitative Finance*.
- Baldi, P., & Hornik, K. (1989). "Neural Networks and Principal Component
  Analysis: Learning from Examples Without Local Minima." *Neural
  Networks*.
