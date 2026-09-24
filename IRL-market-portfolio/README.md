# Econometric Estimation of an IRL-Based Market Portfolio Model

A self-contained project estimating the Inverse-Reinforcement-Learning
(IRL) based market model from Halperin & Feldshteyn (2018), "Market
Self-Learning of Signals, Impact and Optimal Trading: Invisible Hand
Inference with Free Energy." IRL of a market-optimal portfolio policy,
combined with the underlying trading model's state and return equations,
reduces (in the continuous-time, small-mean-reversion limit) to a
multivariate **Geometric Mean Reversion (GMR)** process for asset market
caps / prices:

```
dX_t = kappa ⊙ X_t ⊙ (theta/kappa − X_t) dt + X_t ⊙ [w z_t dt + sigma dW_t]
```

This started from a Coursera final-project notebook (peer-review
instructions, point values, and grading scaffolding removed) and has been
rebuilt as a tested Python project, organized around the assignment's four
parts:

- **Part 1** — calibrate the model on DJI-30 data with SMA signals, at
  increasing levels of cross-sectional pooling (pooled / per-sector /
  per-asset).
- **Part 2** — propose and evaluate alternative signals.
- **Part 3** — repeat the analysis on the real S&P 500 universe.
- **Part 4** — turn the fitted model into a trading strategy and compare
  it with the PCA / Absorption-Ratio strategy from Course 2.

## A note on data

**DJI-30 market caps (Parts 1–2).** The real course dataset (`dja_cap.csv`)
wasn't available in this environment. `data/dja_cap.csv` is instead a
**synthetic, reproducible placeholder**, simulated directly from this
project's own GMR model with known ground-truth parameters
(`scripts/generate_dja_data.py`; kappa=0.06, w0=1.0, w_sma10=0.35,
w_sma30=−0.20). This has a real upside: fitting the model back on this file
is a genuine parameter-recovery test (see Part 1 below), which a real,
unlabeled dataset couldn't give you. To use real data instead, drop a
same-format file (no date column, one column per DJI ticker, consecutive
business days starting 2010-01-04) at `data/dja_cap.csv` — nothing else
needs to change.

**S&P 500 prices (Part 3).** `data/spx_holdings_and_spx_closeprice.csv` is
real data — the same file used in the Course-2 PCA/Absorption-Ratio
project, reused here.

## Project layout

```
irl-market-portfolio/
├── data/
│   ├── dja_cap.csv                             # synthetic, see above
│   └── spx_holdings_and_spx_closeprice.csv      # real, ~418 tickers, 2000-2013
├── src/irl_market/
│   ├── data_loader.py     # load DJI/SPX panels, normalize levels, sector map
│   ├── signals.py         # SMA, momentum, volatility, drawdown, z-score signals
│   ├── simulate.py        # synthetic GMR data generator (for dja_cap.csv & tests)
│   ├── estimation.py      # OLS/MLE fitting: pooled / per-sector / per-asset
│   ├── policy.py          # IRL-implied trading signal, dollar-neutral backtest
│   └── plotting.py        # matplotlib helpers
├── src/pca_strategy/       # vendored from the Course-2 project, for Part 4's comparison
├── notebooks/
│   └── irl_market_portfolio.ipynb   # Parts 1-4 walkthrough (same pipeline as main.py)
├── scripts/
│   ├── generate_dja_data.py   # builds data/dja_cap.csv
│   └── build_notebook.py      # regenerates the notebook from source
├── tests/
│   └── test_irl_market.py     # unit tests incl. parameter recovery
├── main.py                    # end-to-end CLI pipeline -> results/
├── results/                   # populated by main.py (plots + CSVs)
└── requirements.txt
```

## Setup & running

```bash
pip install -r requirements.txt   # numpy, pandas, scikit-learn, matplotlib (+ optional pytest/jupyter)

python scripts/generate_dja_data.py   # only needed once, already run; regenerates data/dja_cap.csv
python main.py                         # runs Parts 1-4, writes results/
python tests/test_irl_market.py        # or: pytest tests/
jupyter notebook notebooks/irl_market_portfolio.ipynb
```

## Method: linear MLE instead of hand-rolled optimization

The original notebook's approach to Part 1 hand-rolled a negative
log-likelihood with a single pooled scalar variance and optimized it with
`scipy.optimize.minimize`. That's unnecessary: for a **fixed** choice of
signals, the model

```
Δx_t / x_t = kappa (W · z'_t − x_t) + eps_t,   eps_t ~ N(0, Sigma_x)
```

is **linear** in kappa (the coefficient on −x_t) and b = kappa·W (the
coefficients on the extended signal vector z'_t = [1, signals...]). The
Gaussian maximum-likelihood estimate of the mean equation is therefore
exactly **ordinary least squares** — `src/irl_market/estimation.py`
implements this directly via `np.linalg.lstsq`, then estimates the
residuals' full N×N cross-sectional covariance Σ_x and computes a proper
multivariate-Gaussian log-likelihood (and AIC/BIC) from it, rather than
assuming one pooled scalar variance shared by every asset and every day.

Three pooling levels are implemented, matching the assignment's suggested
complexity ladder:

| Function | Parametrization |
|---|---|
| `fit_gmr_pooled` | one shared kappa, W for every asset |
| `fit_gmr_per_group` | shared kappa, W within each sector |
| `fit_gmr_per_asset` | fully heterogeneous kappa_i, W_i per asset |

## Key findings

**Part 1 (DJI-30, synthetic data with known ground truth).** The per-asset
fit recovers kappa almost exactly (mean 0.0606 vs. true 0.06). The pooled
fit is *attenuated* (0.036) — not a bug, but a genuine pooling-bias finding:
between-asset variation in normalized level here is partly driven by
idiosyncratic starting points/volatility rather than true differences in
mean-reversion speed, and pooled OLS conflates the two. BIC nonetheless
prefers the simplest pooled specification once the extra per-asset/
per-sector parameters are penalized for — the added flexibility isn't
earning its keep on just 30 similar blue-chip names.

**Part 2 (alternative signals).** Momentum and volatility add almost
nothing to the SMA baseline. Drawdown-from-high and the 60-day z-score
signal look very different by raw log-likelihood — but that's largely a
sample-size artifact (they use a 60-day window and so are fit on ~900 fewer
asset-days than the 30-day baseline), a good illustration of why AIC/BIC
comparisons need a held-fixed sample to be meaningful.

**Part 3 (S&P 500).** Reusing `normalize_levels` unmodified on the S&P
universe initially *broke* the regression: a single early-window baseline
lets 14 years of highly divergent performance push `x_t` across two orders
of magnitude, turning a handful of extreme-growth/near-delisting names into
severe leverage points (`kappa` collapsed to ~2e-5 with wildly unstable W).
`normalize_levels(..., method="rolling")` — dividing by each asset's own
trailing rolling mean instead of a fixed baseline — fixes this by detrending
long-run growth so the state stays O(1) for the whole sample. This is
documented in the function's docstring in `data_loader.py` as a real,
scale-dependent modeling decision, not swept under the rug.

**Part 4 (trading strategy vs. Course 2).** The IRL-implied long/short
policy's raw backtested Sharpe (~4.5, out-of-sample) is implausibly high
for a real strategy — and the project says so explicitly rather than
reporting it uncritically. A frictionless, daily-rebalanced long/short book
across ~400 names is a textbook setting for short-horizon reversal signals
to look spectacular on paper (Grinold's "fundamental law of active
management": even a weak per-name edge compounds into a large *aggregate*
Sharpe purely from breadth). `main.py`/the notebook charge a simple
proportional cost on daily turnover and show the edge collapsing to
negative between 20–50 bps of round-trip cost — a realistic range once
spread, commissions, and price impact are considered for a book this size.
The Course-2 AR-Delta EQ/FI strategy, evaluated over the identical
out-of-sample window, is a much lower-turnover (a few trades/year) signal
and its Sharpe (~0.84) is a far more credible estimate of real,
implementable performance.

## Extending this

- Use the Absorption Ratio itself (from `src/pca_strategy`) as an
  additional macro signal in z_t, tying the single-stock IRL model to the
  same systemic-risk regime the Course-2 strategy times off of.
- Add ridge/shrinkage regularization to the pooled fit to address the
  const/x_t collinearity noted in `estimation.py`.
- Estimate a genuinely time-varying kappa_t via a short rolling re-fit
  instead of one fixed value per pooling level.
- Swap in a real `dja_cap.csv` (see "A note on data" above) and compare
  Part 1's findings against real data.

## References

- Halperin, I., & Feldshteyn, I. (2018). "Market Self-Learning of Signals,
  Impact and Optimal Trading: Invisible Hand Inference with Free Energy."
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3174498
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization."
  *Financial Analysts Journal*.
- Bertsimas, D., Gupta, V., & Paschalidis, I. Ch. (2012). "Inverse
  Optimization: A New Perspective on the Black-Litterman Model."
  *Operations Research*, 60(6), 1389-1403.
- Dixit, A. K., & Pindyck, R. S. (1994). *Investment Under Uncertainty*.
  Princeton University Press.
- Ewald, C.-O., & Yang, Z. (2007). "Geometric Mean Reversion: Formulas for
  the Equilibrium Density and Analytic Moment Matching." University of St
  Andrews Economics Preprints.
- Grinold, R. C. (1989). "The Fundamental Law of Active Management."
  *Journal of Portfolio Management*.
