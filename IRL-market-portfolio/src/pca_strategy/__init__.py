"""
pca_strategy
============

A small, self-contained toolkit for using Principal Component Analysis (PCA)
on a universe of stock returns to:

1. build eigen-portfolios (portfolios whose weights come directly from the
   principal components of the return covariance/correlation matrix),
2. measure market systemic risk over time via the *Absorption Ratio* (AR),
   following Kritzman, Li, Page & Rigobon (2011), and
3. drive a simple regime-switching Equity/Fixed-Income allocation strategy
   from shifts in the Absorption Ratio ("AR Delta").

The package is organized as:

- ``data_loader``      -- load prices, compute & normalize returns
- ``weighting``         -- exponentially-decaying sample weights
- ``pca_analysis``      -- rolling PCA, absorption ratio
- ``eigen_portfolios``  -- construct & evaluate eigen-portfolios
- ``autoencoder``       -- optional linear autoencoder (PCA via neural net)
- ``strategy``          -- AR-Delta regime signal, position sizing, backtest
- ``plotting``          -- matplotlib helpers used by ``main.py``
"""

__version__ = "1.0.0"
