"""
irl_market
==========

Econometric estimation of an IRL-based (Inverse Reinforcement Learning)
model of market dynamics, following Halperin & Feldshteyn, "Market
Self-Learning of Signals, Impact and Optimal Trading: Invisible Hand
Inference with Free Energy" (2018).

The headline result used here: the market-optimal investment policy
implied by IRL of a market portfolio, combined with the state and return
equations of the underlying trading model, reduces (in the continuous-time,
small-mean-reversion limit) to a multivariate **Geometric Mean Reversion**
(GMR) process for asset market caps / prices:

    dX_t = kappa ⊙ X_t ⊙ (theta/kappa - X_t) dt + X_t ⊙ [w z_t dt + sigma dW_t]

which this package estimates from data via linear (Gaussian) maximum
likelihood, after folding the signal-dependent equilibrium level into a
single "target" term `W @ z'_t` (z'_t = [1, signals...]):

    Δx_t / x_t = kappa ⊙ (W z'_t - x_t) + eps_t,   eps_t ~ N(0, Sigma_x)

Modules:

- ``data_loader``  -- load DJI / S&P price-or-cap panels, normalize levels
- ``signals``      -- moving-average, momentum, volatility, and other signals
- ``simulate``      -- synthetic GMR data generator (used for the DJI
                        placeholder dataset and for parameter-recovery tests)
- ``estimation``    -- OLS/MLE fitting at pooled / per-sector / per-asset
                        granularity, plus log-likelihood & information criteria
- ``policy``        -- IRL-implied trading signal & simple backtest
- ``plotting``      -- matplotlib helpers
"""

__version__ = "1.0.0"
