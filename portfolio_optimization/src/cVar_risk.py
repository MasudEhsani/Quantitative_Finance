import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import yfinance as yf
import pandas as pd
import cvxpy as cp


tickers = [
    "IBM","GE","LLY","F","BAC","KR","ITUB","GOLD","SIRI","ACB",
    "VALE","PFE","T","LUMN","OVV","ABEV","CVS","MRVL","CSCO","MO",
    "BMY","WFC","HBAN"
]


"""
start = "2016-01-03"
end = "2017-12-29"   # end is exclusive in yfinance, so use +1 day

# download adjusted close prices
data = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=True
)

# robust price selection
prices = data["Close"]

# compute returns
daily_returns = prices.pct_change(fill_method=None).dropna()

print(daily_returns.head())

print("IBM first 50 days:\n")
print(daily_returns["IBM"].head(50))

asset_arrays = {ticker: prices[ticker].dropna().values for ticker in tickers}

aligned_prices = prices.dropna()   # removes rows with any missing values

X_prices = np.vstack([aligned_prices[t].values for t in tickers])

X = X_prices[:, 1:] / X_prices[:, :-1] - 1

means = X.mean(axis=1)
cov_matrix = np.cov(X)

print("\nMeans:\n", means)
print("\nCovariance matrix:\n", cov_matrix)

"""


# read first sheet of Excel
df = pd.read_excel("stocks.xlsx", sheet_name=0)

# remove first column if it's dates or index
# (adjust depending on file structure)
df = df.iloc[:, 1:]   # starts from column B

# convert to numeric (just in case)
df = df.apply(pd.to_numeric, errors='coerce')

# drop missing values (important for covariance)
df = df.dropna()


# IBM is first column (column B in Excel)
ibm_returns = df.iloc[:, 0]

print("IBM returns:\n")
print(ibm_returns.head(50))



X = df.T.values


means = X.mean(axis=1)

print("\nAverage returns:")
print(means)


cov_matrix = np.cov(X)

print("\nCovariance matrix:")
print(cov_matrix)



alpha = 0.9

# Step 1: sample mean return for each asset (R̄_i)
R_bar = df.mean(axis=0)   # pandas Series, one mean per column

# Step 2: global mean of means
R_bar_global = R_bar.mean()

# Step 3: shrinkage estimator for all μ_i
mu = alpha * R_bar + (1 - alpha) * R_bar_global

# Step 4: IBM is first column
mu_IBM = mu.iloc[0]

print("Shrinkage estimate μ_IBM (alpha=0.9):", mu_IBM)


mu = df.mean().values
V = df.cov().values

n = len(mu)

# Decision variable
x = cp.Variable(n)

# Objective (risk minimization)
objective = cp.Minimize(cp.quad_form(x, V))

# Constraints (ONLY constraints here)
constraints = [
    mu @ x >= 0.0005,
    cp.sum(x) == 1,
    x >= 0
]

# Problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract IBM weight (first asset)
ibm_weight = x.value[0]

print("IBM weight:", ibm_weight)
print("IBM weight (%):", round(100 * ibm_weight, 3))




# Variables: long and short parts
x_plus = cp.Variable(n)
x_minus = cp.Variable(n)

# Net portfolio
x = x_plus - x_minus

# Objective
objective = cp.Minimize(cp.quad_form(x, V))

# Constraints
constraints = [
    mu @ x >= 0.0005,
    cp.sum(x) == 1,
    cp.sum(x_minus) <= 0.1,
    x_plus >= 0,
    x_minus >= 0
]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()

# IBM weight (first asset)
ibm_weight = (x_plus.value - x_minus.value)[0]

print("IBM weight:", ibm_weight)
print("IBM weight (%):", round(100 * ibm_weight, 3))



# Tickers
tickers = [
    "IBM","GE","LLY","F","BAC","KR","ITUB","GOLD","SIRI","ACB",
    "VALE","PFE","T","LUMN","OVV","ABEV","CVS","MRVL","CSCO","MO",
    "BMY","WFC","HBAN"
]

# Align just in case
df.columns = tickers[:df.shape[1]]

# Storage
results = []

alpha = 0.90  # VaR/CVaR level

for col in df.columns:
    r = df[col].dropna()

    var = np.var(r, ddof=1)

    # VaR(90%) = 10th percentile loss threshold
    var_90 = np.quantile(r, 0.10)

    # CVaR(90%) = mean of worst 10%
    cvar_90 = r[r <= var_90].mean()

    results.append([col, var, var_90, cvar_90])

# Create DataFrame
res = pd.DataFrame(results, columns=["Stock", "Variance", "VaR(90)", "CVaR(90)"])

print(res)



# Highest variance
max_var_stock = res.loc[res["Variance"].idxmax()]

# Worst VaR (most negative = worse loss, so we take min)
worst_var_stock = res.loc[res["VaR(90)"].idxmin()]

# Worst CVaR
worst_cvar_stock = res.loc[res["CVaR(90)"].idxmin()]

print("\nHighest Variance Stock:")
print(max_var_stock)

print("\nWorst VaR(90%) Stock:")
print(worst_var_stock)

print("\nWorst CVaR(90%) Stock:")
print(worst_cvar_stock)


# Equal weights
n = df.shape[1]
weights = np.ones(n) / n

# Portfolio returns
portfolio_returns = df.values @ weights

# VaR(0.9) = 10% quantile (left tail)
VaR_90 = np.quantile(portfolio_returns, 0.10)

print("VaR_0.9 (portfolio):", round(VaR_90, 4))

# CVaR(0.9) = mean of worst 10%
CVaR_90 = portfolio_returns[portfolio_returns <= VaR_90].mean()

print("CVaR_0.9 (portfolio):", round(CVaR_90, 4))