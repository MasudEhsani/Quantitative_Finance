import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


asset1 = np.array([
    2.1675, 3.3875, -2.5194, -1.3959, -2.7264, 0.7832, 3.9749,
    -3.3869, -0.1714, 0.5439, 1.0913, -1.0361, -0.871, 1.0377,
    -2.7236, 2.7485, -0.8348, -1.2728, -0.5232, 0.1174, 1.1843,
    0.6003, -0.5785, -1.8834, 1.0666, 3.7147, -0.3213, 1.3712,
    -0.7987, 0.949, -2.5322, 0.4613, -1.6205, 4.7482, 1.6551,
    -5.1563, -1.0738, -1.2798, -0.7793, -0.3613, 0.1896, -2.6047,
    -3.1645, -0.6066, 0.8595, 3.642, 0.6684, 1.5556, 1.4717,
    -2.6414, -1.6742, 1.5918, -1.4625, 0.0491, -0.2674, 1.0214,
    1.0679, -5.4239, 0.4125, 4.9665
])

asset2 = np.array([
    -1.8883, -0.5608, 1.5773, 1.4225, 1.0408, 1.2914, -0.2969,
    0.6602, 0.2207, 0.3632, 0.103, -0.5964, 2.2177, 0.2439,
    2.7658, -1.4885, -1.3056, 2.2504, 1.8409, 0.9127, 1.6094,
    -0.1249, -0.2565, 0.5511, -0.555, 0.2948, 0.8969, -0.6688,
    -0.0603, -0.0582, 3.4582, 1.0112, 0.7624, -2.1118, 0.6686,
    1.286, 0.5736, -0.3968, -0.6159, -2.5366, 0.889, 1.2735,
    1.264, 1.6454, 0.1415, -0.3455, 1.9907, -1.1351, -1.3031,
    0.9842, 2.6791, -0.6697, -2.1408, 0.656, 0.0338, 1.0945,
    0.9726, 3.2911, 0.2005, -2.4944
])

asset3 = np.array([
    3.0712, 1.2428, -3.184, 3.0923, -0.0131, -2.0597, 2.0496,
    -0.585, 0.6504, 1.0962, 2.7193, -0.8361, -4.2316, 1.624,
    -3.2315, 7.4063, 0.3338, -4.3389, 1.5357, -1.2732, -0.208,
    -0.6808, 1.451, -1.5372, -1.0502, 3.36, -0.3788, -0.1197,
    -0.1823, -0.1217, -4.8028, -0.5589, -0.5277, 3.0093, 1.0955,
    -2.0927, 0.7881, -1.4555, 2.2117, 2.9634, -3.5207, -4.4262,
    -4.5345, 0.1385, -3.6547, 2.8741, -0.0495, 0.9447, 1.3656,
    1.7807, -1.7593, 0.9009, -2.0389, 0.1216, 4.0254, 2.3588,
    0.7781, -4.9794, 0.2146, -0.2641
])

# Stack into matrix (assets x observations)
X = np.vstack([asset1, asset2, asset3])

# Mean vector
means = X.mean(axis=1)

# Covariance matrix
cov_matrix = np.cov(X)


rf = 1
mu = 12 * means
Sigma = 12 * cov_matrix * 10**-4
Sigma = np.round(Sigma, 4)
target_vol = 0.05

print("Means:\n", mu)
print("\nCovariance matrix:\n",Sigma)


#### Compute an estimated efficient portfolio with 5% volatility. What is the estimated return on this portfolio? 

n = len(mu)

# objective: negative return (we minimize)
def objective(w):
    return -w @ mu

# constraint: volatility = target
def vol_constraint(w):
    return np.sqrt(w @ Sigma @ w) - target_vol

# constraint: weights sum to 1
def sum_constraint(w):
    return np.sum(w) - 1

constraints = [
    {"type": "eq", "fun": vol_constraint},
    {"type": "eq", "fun": sum_constraint}
]

bounds = [(None, None)] * n  # allow short selling

w0 = np.ones(n) / n

result = minimize(
    objective,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

w_opt = result.x

portfolio_return = w_opt @ mu
portfolio_vol = np.sqrt(w_opt @ Sigma @ w_opt)

print("Optimal weights:\n", w_opt)
print("\nExpected return:", portfolio_return)
print("Volatility:", portfolio_vol)



# portfolio returns over time (realized series)
portfolio_returns = w_opt @ X   # shape: (T,)

# realized mean return (monthly)
realized_return = np.mean(portfolio_returns)

# realized volatility (monthly)
realized_vol = np.std(portfolio_returns)



realized_return_annual = 12 * realized_return
realized_vol_annual = np.sqrt(12) * realized_vol

print("Realized annual return:", realized_return_annual)
print("Realized annual volatility:", realized_vol_annual)
print("Realized mean return (monthly):", realized_return)
print("Realized volatility (monthly):", realized_vol)




rf = 1

mu = 12 * means 
Sigma = 12 * cov_matrix * 1e-4
Sigma = np.round(Sigma, 4)

excess_mu = mu - rf

def objective(w):
    return -w @ excess_mu

def vol_constraint(w):
    return np.sqrt(w @ Sigma @ w) - target_vol

def sum_constraint(w):
    return np.sum(w) - 1

constraints = [
    {"type": "eq", "fun": vol_constraint},
    {"type": "eq", "fun": sum_constraint}
]

w0 = np.ones(len(mu)) / len(mu)

result = minimize(objective, w0, method="SLSQP",
                  bounds=[(None,None)]*len(mu),
                  constraints=constraints)

w_opt = result.x

# correct expected return including rf
portfolio_return = rf + w_opt @ (mu - rf)
portfolio_vol = np.sqrt(w_opt @ Sigma @ w_opt)

print("Expected return (correct):", portfolio_return)
print("Volatility:", portfolio_vol)




# Inputs
# -----------------------------
rf = 1

mu = 12 * means
Sigma = 12 * cov_matrix * 1e-4
Sigma = np.round(Sigma, 4)

target_vol = 0.05

# -----------------------------
# Step 1: Tangency portfolio
# w_tan ∝ Σ^{-1}(μ - rf)
# -----------------------------
excess_mu = mu - rf

w_tan = np.linalg.inv(Sigma) @ excess_mu

# normalize (direction only matters here)
w_tan = w_tan / np.sum(w_tan)

# -----------------------------
# Step 2: Volatility of tangency portfolio
# -----------------------------
vol_tan = np.sqrt(w_tan @ Sigma @ w_tan)

# -----------------------------
# Step 3: Scale to target volatility
# -----------------------------
k = target_vol / vol_tan
w_risky = k * w_tan

# -----------------------------
# Step 4: Risk-free weight
# -----------------------------
w_rf = 1 - np.sum(w_risky)

# -----------------------------
# Step 5: Portfolio statistics
# -----------------------------
portfolio_return = rf + w_risky @ excess_mu
portfolio_vol = np.sqrt(w_risky @ Sigma @ w_risky)

print("Risky weights:\n", w_risky)
print("\nRisk-free weight:", w_rf)

print("\nExpected return:", portfolio_return)
print("Volatility:", portfolio_vol)

# portfolio risky returns over time
risky_returns = w_risky @ X   # shape: (T,)

# risk-free return (constant each period)
rf_series = np.full(X.shape[1], rf/12)

# full portfolio returns
portfolio_returns = risky_returns + (1 - np.sum(w_risky)) * rf_series


# monthly realized metrics
realized_return = np.mean(portfolio_returns)
realized_vol = np.std(portfolio_returns)

# annualized
realized_return_annual = 12 * realized_return
realized_vol_annual = np.sqrt(12) * realized_vol

print("Realized annual return:", realized_return_annual)
print("Realized annual volatility:", realized_vol_annual)

print("Realized monthly return:", realized_return)
print("Realized monthly volatility:", realized_vol)
















losses= -X.mean(axis=0)
print("loss_equally_weithed:", losses)

# sort losses (ascending)
sorted_losses = np.sort(losses)

N = len(sorted_losses)
p = 0.90

# index for VaR (ceiling rule)
k_p = int(np.ceil(p * N)) - 1   # -1 for Python indexing

VaR_90 = sorted_losses[k_p]

# K_p (ceiling rule as in your earlier definition)
K_p = int(np.ceil(p * N)) - 1   # convert to Python index

# VaR
VaR_90 = sorted_losses[K_p]

# STRICT CVaR: sum of largest N - K_p + 1 elements
tail_sum = np.sum(sorted_losses[K_p:])

# denominator (IMPORTANT: exactly as you specified)
CVaR_90 = tail_sum / ((1 - p) * N)
print("VaR (90%):", VaR_90)
print("CVaR (90%):", CVaR_90)



from scipy.stats import binom

n = 15
p = 0.5
k = 12

prob = binom.sf(k-1, n, p)  # P(X >= 12)

print("Probability of 12 or more successes:", prob)



n = 15
p = 0.5
k = 14

prob = binom.sf(k-1, n, p)  # P(X >= 12)

M =100

prob2 = 1 - ( 1 - prob) ** M

print("Probability of 12 or more successes:", prob2)
