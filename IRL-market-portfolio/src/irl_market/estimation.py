"""
Maximum-likelihood estimation of the GMR/IRL market model:

    Delta x_t / x_t = kappa * (W . z'_t - x_t) + eps_t,   eps_t ~ N(0, Sigma_x)

where z'_t = [1, signal_1_t, ..., signal_K_t] is the extended signal vector
(a leading constant plus K chosen predictors) for a given asset at time t.

The key simplification used throughout this module: for a *fixed* choice of
signals, the right-hand side is linear in the unknown parameters
``b = kappa * W`` and ``kappa`` itself (as the coefficient on ``-x_t``), so
the Gaussian maximum-likelihood estimate of the mean equation is exactly
ordinary least squares -- no numerical optimizer, hand-rolled likelihood
loop, or scalar-only covariance shortcut is needed. This is both simpler
and more correct than iteratively maximizing a hand-written NLL with
scipy.optimize, which is how the original notebook approached it.

Three levels of cross-sectional pooling are supported, matching the
complexity levels the assignment suggests:

- ``"pooled"``    -- one shared kappa and one shared W for every asset
- ``"per_sector"`` -- shared kappa/W within each sector group
- ``"per_asset"``  -- fully heterogeneous kappa_i, W_i for every asset

In every case, once the mean equation is fit, the residuals' NxN
cross-sectional covariance matrix is estimated directly (``Sigma_x``), and
a proper multivariate-Gaussian log-likelihood (and AIC/BIC, for comparing
different signal sets or pooling levels) is computed from it.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GMRFitResult:
    mode: str
    kappa: object              # float ("pooled"), pd.Series indexed by asset ("per_asset"/"per_sector")
    W: object                  # pd.Series ("pooled") or pd.DataFrame, rows=asset, cols=[const, signals...]
    groups: Optional[pd.Series]  # asset -> group label, only set for "per_sector"
    residuals: pd.DataFrame    # T x N
    Sigma: pd.DataFrame        # N x N residual covariance
    loglik: float
    n_params: int
    n_obs: int
    aic: float = field(init=False)
    bic: float = field(init=False)

    def __post_init__(self):
        self.aic = 2 * self.n_params - 2 * self.loglik
        self.bic = self.n_params * np.log(self.n_obs) - 2 * self.loglik

    def summary(self) -> str:
        lines = [
            f"GMR fit ({self.mode}): kappa={_fmt_kappa(self.kappa)}",
            f"  log-likelihood = {self.loglik:,.1f}   AIC = {self.aic:,.1f}   BIC = {self.bic:,.1f}",
            f"  n_params = {self.n_params}   n_obs (asset-days) = {self.n_obs:,}",
        ]
        return "\n".join(lines)


def _fmt_kappa(kappa) -> str:
    if np.isscalar(kappa):
        return f"{kappa:.5f}"
    return f"mean={np.mean(kappa):.5f}, std={np.std(kappa):.5f}"


def prepare_regression_arrays(X: pd.DataFrame, signal_panels: Dict[str, pd.DataFrame]):
    """Align X and a set of signal panels to a common, NaN-free window and
    split into (X_t, X_next, signals_t) ready for regression.

    Arguments:
        X -- normalized level panel (T x N)
        signal_panels -- dict of {name: DataFrame}, same shape as X

    Return:
        (X_t, X_next, signals_t) -- X_t/X_next are (T'-1 x N) DataFrames one
            step apart, signals_t is a dict of (T'-1 x N) DataFrames aligned
            with X_t
    """
    frames = [X] + list(signal_panels.values())
    start = max(f.dropna(how="any").index[0] for f in frames)
    X_aligned = X.loc[start:]
    signals_aligned = {name: df.loc[start:] for name, df in signal_panels.items()}

    X_t = X_aligned.iloc[:-1]
    X_next = X_aligned.iloc[1:]
    signals_t = {name: df.loc[X_t.index] for name, df in signals_aligned.items()}

    # Validity mask, indexed positionally (X_t and X_next are intentionally
    # offset by one day, so their DatetimeIndex labels differ -- combining
    # boolean Series here via label alignment would silently scramble the
    # t / t+1 pairing rather than raise, so we drop to plain numpy arrays).
    mask = X_t.notna().all(axis=1).to_numpy() & X_next.notna().all(axis=1).to_numpy()
    for s in signals_t.values():
        mask &= s.notna().all(axis=1).to_numpy()
    if not mask.all():
        X_t, X_next = X_t.loc[mask], X_next.loc[mask]
        signals_t = {name: s.loc[mask] for name, s in signals_t.items()}

    assert len(X_t) == len(X_next) and not X_t.index.equals(X_next.index), \
        "X_t / X_next lost their one-step time alignment"

    return X_t, X_next, signals_t


def _design_and_target(X_t: pd.DataFrame, X_next: pd.DataFrame, signals_t: Dict[str, pd.DataFrame],
                        include_const: bool):
    """Stack a (possibly multi-asset) panel regression into flat arrays.

    Returns (Z, y, names) where Z has shape (T*N, n_regressors), y has
    shape (T*N,), and names[-1] == '__X_t__' is always the regressor whose
    coefficient is -kappa.
    """
    T, N = X_t.shape
    r = (X_next.values - X_t.values) / X_t.values

    cols, names = [], []
    if include_const:
        cols.append(np.ones((T, N)))
        names.append("const")
    for name, df in signals_t.items():
        cols.append(df.values)
        names.append(name)
    cols.append(X_t.values)
    names.append("__X_t__")

    Z = np.stack([c.ravel(order="C") for c in cols], axis=1)
    y = r.ravel(order="C")
    return Z, y, names, (T, N)


def _ols(Z: np.ndarray, y: np.ndarray):
    coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    return coef, resid


def _gaussian_loglik(residuals: pd.DataFrame, Sigma: pd.DataFrame) -> float:
    """Multivariate-Gaussian log-likelihood of a T x N residual panel under
    a shared (across time) N x N covariance ``Sigma``."""
    T, N = residuals.shape
    sign, logdet = np.linalg.slogdet(Sigma.values)
    if sign <= 0:
        # Numerically singular / not PD (can happen with very short samples
        # or near-collinear signals) -- regularize with a small ridge.
        Sigma = Sigma + np.eye(N) * 1e-8 * np.trace(Sigma.values) / N
        sign, logdet = np.linalg.slogdet(Sigma.values)
    Sigma_inv = np.linalg.inv(Sigma.values)
    resid = residuals.values
    quad = np.einsum("ti,ij,tj->t", resid, Sigma_inv, resid).sum()
    return float(-0.5 * T * (N * np.log(2 * np.pi) + logdet) - 0.5 * quad)


def fit_gmr_pooled(X: pd.DataFrame, signal_panels: Dict[str, pd.DataFrame],
                    include_const: bool = True) -> GMRFitResult:
    """Fit one shared kappa and one shared W across the whole cross-section
    (the assignment's simplest suggested parametrization).
    """
    X_t, X_next, signals_t = prepare_regression_arrays(X, signal_panels)
    Z, y, names, (T, N) = _design_and_target(X_t, X_next, signals_t, include_const)
    coef, resid_flat = _ols(Z, y)

    kappa = -coef[-1]
    W = pd.Series(coef[:-1] / kappa, index=names[:-1])

    residuals = pd.DataFrame(resid_flat.reshape(T, N, order="C"), index=X_t.index, columns=X_t.columns)
    Sigma = residuals.cov()
    loglik = _gaussian_loglik(residuals, Sigma)
    n_params = len(coef)  # kappa + each W component
    n_obs = T * N

    return GMRFitResult("pooled", float(kappa), W, None, residuals, Sigma, loglik, n_params, n_obs)


def fit_gmr_per_asset(X: pd.DataFrame, signal_panels: Dict[str, pd.DataFrame],
                       include_const: bool = True) -> GMRFitResult:
    """Fit a fully separate kappa_i, W_i for every asset (the assignment's
    most flexible suggested parametrization)."""
    X_t, X_next, signals_t = prepare_regression_arrays(X, signal_panels)
    assets = X_t.columns

    kappas, W_rows, resid_cols = {}, {}, {}
    for asset in assets:
        X_t_i = X_t[[asset]]
        X_next_i = X_next[[asset]]
        signals_i = {name: df[[asset]] for name, df in signals_t.items()}
        Z, y, names, (T, _) = _design_and_target(X_t_i, X_next_i, signals_i, include_const)
        coef, resid_flat = _ols(Z, y)

        kappa_i = -coef[-1]
        kappas[asset] = kappa_i
        W_rows[asset] = coef[:-1] / kappa_i
        resid_cols[asset] = resid_flat

    kappa = pd.Series(kappas)
    W = pd.DataFrame(W_rows, index=names[:-1]).T
    residuals = pd.DataFrame(resid_cols, index=X_t.index)[assets]
    Sigma = residuals.cov()
    loglik = _gaussian_loglik(residuals, Sigma)
    n_params = len(assets) * len(names)  # kappa_i + W_i per asset
    n_obs = residuals.shape[0] * residuals.shape[1]

    return GMRFitResult("per_asset", kappa, W, None, residuals, Sigma, loglik, n_params, n_obs)


def fit_gmr_per_group(X: pd.DataFrame, signal_panels: Dict[str, pd.DataFrame], groups: pd.Series,
                       include_const: bool = True) -> GMRFitResult:
    """Fit one shared kappa_g, W_g per group (e.g. sector) -- the
    assignment's intermediate suggested parametrization.

    Arguments:
        groups -- pandas.Series mapping asset (column name) -> group label
    """
    X_t, X_next, signals_t = prepare_regression_arrays(X, signal_panels)
    assets = X_t.columns
    groups = groups.reindex(assets)

    kappa_per_asset, W_per_asset, resid_frames = {}, {}, []
    group_report = {}
    for group_label, members in groups.groupby(groups).groups.items():
        members = list(members)
        X_t_g, X_next_g = X_t[members], X_next[members]
        signals_g = {name: df[members] for name, df in signals_t.items()}
        Z, y, names, (T, Ng) = _design_and_target(X_t_g, X_next_g, signals_g, include_const)
        coef, resid_flat = _ols(Z, y)

        kappa_g = -coef[-1]
        W_g = coef[:-1] / kappa_g
        group_report[group_label] = {"kappa": kappa_g, **dict(zip(names[:-1], W_g))}

        resid_g = pd.DataFrame(resid_flat.reshape(T, Ng, order="C"), index=X_t_g.index, columns=members)
        resid_frames.append(resid_g)
        for m in members:
            kappa_per_asset[m] = kappa_g
            W_per_asset[m] = W_g

    residuals = pd.concat(resid_frames, axis=1)[assets]
    Sigma = residuals.cov()
    loglik = _gaussian_loglik(residuals, Sigma)
    n_groups = groups.nunique()
    n_params = n_groups * (len(signal_panels) + int(include_const) + 1)
    n_obs = residuals.shape[0] * residuals.shape[1]

    kappa = pd.Series(kappa_per_asset)[assets]
    W = pd.DataFrame(W_per_asset, index=list(signals_t.keys()) if not include_const
                      else ["const"] + list(signals_t.keys())).T.loc[assets]
    group_table = pd.DataFrame(group_report).T

    result = GMRFitResult("per_sector", kappa, W, groups, residuals, Sigma, loglik, n_params, n_obs)
    result.group_table = group_table  # extra attribute, handy for reporting
    return result


def compare_fits(results: Dict[str, GMRFitResult]) -> pd.DataFrame:
    """Tabulate log-likelihood / AIC / BIC across several fits (e.g.
    different signal sets, or different pooling levels) for model
    comparison."""
    rows = {
        name: {
            "n_params": r.n_params,
            "n_obs": r.n_obs,
            "loglik": r.loglik,
            "AIC": r.aic,
            "BIC": r.bic,
            "kappa_mean": float(np.mean(r.kappa)),
        }
        for name, r in results.items()
    }
    return pd.DataFrame(rows).T.sort_values("BIC")
