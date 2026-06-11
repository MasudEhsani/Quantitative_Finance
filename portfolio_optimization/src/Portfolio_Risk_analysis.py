# ============================================================
# 1. IMPORTS
# ============================================================

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import cvxpy as cp


# ============================================================
# 2. ASSET UNIVERSE (TICKERS)
# ============================================================

tickers = [
    "IBM","GE","LLY","F","BAC","KR","ITUB","GOLD","SIRI","ACB",
    "VALE","PFE","T","LUMN","OVV","ABEV","CVS","MRVL","CSCO","MO",
    "BMY","WFC","HBAN"
]


# ============================================================
# 3. (OPTIONAL) DATA DOWNLOAD FROM YFINANCE - COMMENTED BLOCK
# ============================================================
"""
This section downloads historical adjusted close prices
from Yahoo Finance and computes daily returns.

It is currently disabled and replaced by Excel-based input.

start = "2016-01-03"
end = "2017-12-29"

data = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=True
)

prices = data["Close"]

# Compute percentage returns
daily_returns = prices.pct_change(fill_method=None).dropna()

print(daily_returns.head())
print("IBM first 50 days:\n", daily_returns["IBM"].head(50))

# Convert to aligned numpy array format
asset_arrays = {ticker: prices[ticker].dropna().values for ticker in tickers}
aligned_prices = prices.dropna()

X_prices = np.vstack([aligned_prices[t].values for t in tickers])

# Return computation
X = X_prices[:, 1:] / X_prices[:, :-1] - 1

means = X.mean(axis=1)
cov_matrix = np.cov(X)

print("\nMeans:\n", means)
print("\nCovariance matrix:\n", cov_matrix)
"""


# ============================================================
# 4. DATA LOADING (EXCEL INPUT)
# ============================================================

# Load first sheet of Excel file
df = pd.read_excel("stocks.xlsx", sheet_name=0)

# Remove first column (assumed to be dates or index)
df = df.iloc[:, 1:]

# Ensure numeric consistency
df = df.apply(pd.to_numeric, errors='coerce')

# Drop missing values for clean covariance estimation
df = df.dropna()


# ============================================================
# 5. BASIC RETURN INSPECTION
# ============================================================

# IBM is assumed to be the first column
ibm_returns = df.iloc[:, 0]

print("IBM returns:\n")
print(ibm_returns.head(50))


# Convert dataframe to matrix form (assets x observations)
X = df.T.values

# Mean return per asset
means = X.mean(axis=1)

print("\nAverage returns:")
print(means)

# Covariance matrix of asset returns
cov_matrix = np.cov(X)

print("\nCovariance matrix:")
print(cov_matrix)


# ============================================================
# 6. SHRINKAGE MEAN ESTIMATION
# ============================================================

alpha = 0.9

# Sample mean per asset
R_bar = df.mean(axis=0)

# Global mean across assets
R_bar_global = R_bar.mean()

# Shrinkage estimator of expected returns
mu = alpha * R_bar + (1 - alpha) * R_bar_global

# IBM expected return estimate
mu_IBM = mu.iloc[0]

print("Shrinkage estimate μ_IBM (alpha=0.9):", mu_IBM)


# ============================================================
# 7. MARKOWITZ PORTFOLIO OPTIMIZATION (LONG-ONLY)
# ============================================================

mu = df.mean().values
V = df.cov().values

n = len(mu)

# Portfolio weights
x = cp.Variable(n)

# Objective: minimize portfolio variance
objective = cp.Minimize(cp.quad_form(x, V))

# Constraints:
# 1. minimum return target
# 2. fully invested portfolio
# 3. long-only constraint
constraints = [
    mu @ x >= 0.0005,
    cp.sum(x) == 1,
    x >= 0
]

# Solve optimization problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract IBM weight (first asset)
ibm_weight = x.value[0]

print("IBM weight:", ibm_weight)
print("IBM weight (%):", round(100 * ibm_weight, 3))


# ============================================================
# 8. LONG-SHORT PORTFOLIO OPTIMIZATION
# ============================================================

# Long and short decomposition
x_plus = cp.Variable(n)
x_minus = cp.Variable(n)

# Net portfolio position
x = x_plus - x_minus

# Objective: minimize risk
objective = cp.Minimize(cp.quad_form(x, V))

# Constraints:
# - minimum return
# - fully invested
# - limited short exposure
# - non-negative decomposition
constraints = [
    mu @ x >= 0.0005,
    cp.sum(x) == 1,
    cp.sum(x_minus) <= 0.1,
    x_plus >= 0,
    x_minus >= 0
]

# Solve optimization
problem = cp.Problem(objective, constraints)
problem.solve()

# IBM net weight
ibm_weight = (x_plus.value - x_minus.value)[0]

print("IBM weight:", ibm_weight)
print("IBM weight (%):", round(100 * ibm_weight, 3))


# ============================================================
# 9. RISK METRICS: VAR & CVAR PER STOCK
# ============================================================

# Ensure tickers align with dataframe columns
df.columns = tickers[:df.shape[1]]

results = []

alpha = 0.90  # Confidence level (90%)

for col in df.columns:
    r = df[col].dropna()

    # Variance of returns
    var = np.var(r, ddof=1)

    # Value-at-Risk (10% worst-case threshold)
    var_90 = np.quantile(r, 0.10)

    # Conditional Value-at-Risk (expected loss beyond VaR)
    cvar_90 = r[r <= var_90].mean()

    results.append([col, var, var_90, cvar_90])

# Convert results into structured DataFrame
res = pd.DataFrame(results, columns=["Stock", "Variance", "VaR(90)", "CVaR(90)"])

print(res)


# ============================================================
# 10. EXTREME RISK IDENTIFICATION
# ============================================================

# Highest volatility stock
max_var_stock = res.loc[res["Variance"].idxmax()]

# Worst VaR (most negative return threshold)
worst_var_stock = res.loc[res["VaR(90)"].idxmin()]

# Worst CVaR (worst tail expectation)
worst_cvar_stock = res.loc[res["CVaR(90)"].idxmin()]

print("\nHighest Variance Stock:")
print(max_var_stock)

print("\nWorst VaR(90%) Stock:")
print(worst_var_stock)

print("\nWorst CVaR(90%) Stock:")
print(worst_cvar_stock)


# ============================================================
# 11. EQUAL-WEIGHT PORTFOLIO RISK ANALYSIS
# ============================================================

# Equal-weight portfolio construction
n = df.shape[1]
weights = np.ones(n) / n

# Portfolio returns
portfolio_returns = df.values @ weights

# Portfolio VaR (10% quantile)
VaR_90 = np.quantile(portfolio_returns, 0.10)

print("VaR_0.9 (portfolio):", round(VaR_90, 4))

# Portfolio CVaR (expected tail loss)
CVaR_90 = portfolio_returns[portfolio_returns <= VaR_90].mean()

print("CVaR_0.9 (portfolio):", round(CVaR_90, 4))