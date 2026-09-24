"""
Construct eigen-portfolios from PCA on a universe of stock returns.

An eigen-portfolio is a portfolio whose weights come directly from a single
eigenvector of the return covariance/correlation matrix: each eigenvector
gives an *orthogonal* basket of stocks, so the resulting portfolios are, by
construction, uncorrelated with each other over the estimation window. The
first eigen-portfolio typically resembles a broad market-cap-ish basket and
tracks the benchmark index closely; later ones capture progressively more
niche, sector- or style-like sources of common variation.

Reference: Avellaneda & Lee (2010), "Statistical Arbitrage in the U.S.
Equities Market".
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def construct_eigen_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Fit PCA on a window of returns and turn the leading eigenvectors into
    dollar-neutral-free, fully-invested eigen-portfolio weights.

    Each eigenvector v_k (loadings on standardized returns) is converted to
    portfolio weights on the *raw* (non-standardized) assets by dividing by
    each asset's volatility, then renormalizing so the weights sum to 1:

        Q_k,j = v_k,j / sigma_j
        w_k,j = Q_k,j / sum_j(Q_k,j)

    Arguments:
        returns -- pandas.DataFrame of raw (non-standardized) asset returns,
            one column per asset, one row per period
        n_portfolios -- number of leading eigen-portfolios to construct

    Return:
        (weights, explained_variance_ratio) --
            weights: pandas.DataFrame, shape (n_portfolios, n_assets),
                index 'EP1', 'EP2', ... , columns = asset tickers
            explained_variance_ratio: pandas.Series of the fraction of total
                variance each corresponding component explains
    """
    sigma = returns.std(axis=0)
    corr = returns.corr()

    pca = PCA(n_components=n_portfolios).fit(corr.values)
    eigvecs = pca.components_  # shape (n_portfolios, n_assets), rows = eigenvectors

    raw_weights = eigvecs / sigma.values[None, :]
    norm_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    weights = pd.DataFrame(
        norm_weights,
        index=[f"EP{k + 1}" for k in range(n_portfolios)],
        columns=returns.columns,
    )
    explained_variance_ratio = pd.Series(
        pca.explained_variance_ratio_,
        index=weights.index,
        name="explained_variance_ratio",
    )
    return weights, explained_variance_ratio


def eigen_portfolio_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Apply eigen-portfolio weights to a panel of returns.

    Arguments:
        returns -- pandas.DataFrame of asset returns, columns matching
            ``weights.columns``
        weights -- pandas.DataFrame as returned by
            ``construct_eigen_portfolios`` (rows = portfolios, columns = assets)

    Return:
        portfolio_returns -- pandas.DataFrame, one column per eigen-portfolio
    """
    aligned = returns[weights.columns]
    return pd.DataFrame(
        aligned.values @ weights.values.T,
        index=returns.index,
        columns=weights.index,
    )


def rolling_out_of_sample_eigen_portfolios(
    returns: pd.DataFrame,
    lookback_window: int = 252,
    n_portfolios: int = 3,
    rebalance_every: int = 21,
) -> pd.DataFrame:
    """Walk-forward eigen-portfolio backtest: on each rebalance date, fit the
    eigen-portfolio weights on the trailing ``lookback_window`` of returns,
    then hold those weights fixed and apply them to *subsequent, unseen*
    returns until the next rebalance. This avoids look-ahead bias -- weights
    are always fit strictly on data available at the time.

    Arguments:
        returns -- pandas.DataFrame of raw asset returns
        lookback_window -- estimation window, in days, used to fit weights
        n_portfolios -- number of eigen-portfolios to track
        rebalance_every -- how often (in days) to refit the weights

    Return:
        oos_returns -- pandas.DataFrame of out-of-sample daily returns for
            each eigen-portfolio, indexed like ``returns`` (starting after
            the first lookback window)
    """
    n = len(returns)
    chunks = []
    rebal_points = list(range(lookback_window, n, rebalance_every))
    for start in rebal_points:
        fit_window = returns.iloc[start - lookback_window:start, :]
        weights, _ = construct_eigen_portfolios(fit_window, n_portfolios=n_portfolios)

        end = min(start + rebalance_every, n)
        holding_period = returns.iloc[start:end, :]
        if holding_period.empty:
            continue
        chunks.append(eigen_portfolio_returns(holding_period, weights))

    if not chunks:
        return pd.DataFrame(columns=[f"EP{k + 1}" for k in range(n_portfolios)])
    return pd.concat(chunks, axis=0)
