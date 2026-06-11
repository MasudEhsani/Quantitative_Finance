import numpy as np
from scipy.optimize import minimize

# Mean returns (convert % to decimals)
mu = np.array([0.06, 0.02, 0.04])
rf= 0.01
# Covariance matrix
Sigma = np.array([
    [0.00800, -0.00200, 0.00400],
    [-0.00200, 0.00200, -0.00200],
    [0.00400, -0.00200, 0.00800]
])

n = len(mu)
ones = np.ones(n)

# =========================
# 1. Analytical solution
# =========================
Sigma_inv = np.linalg.inv(Sigma)

w_analytical = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
ret_analytical = w_analytical @ mu
var_analytical = w_analytical @ Sigma @ w_analytical

# =========================
# 2. Numerical optimization
# =========================

# Objective: portfolio variance
def portfolio_variance(w):
    return w @ Sigma @ w

# Constraint: sum of weights = 1
constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# Initial guess
w0 = np.ones(n) / n

# No bounds → short selling allowed
result = minimize(portfolio_variance, w0, constraints=constraints)

w_numerical = result.x
ret_numerical = w_numerical @ mu
var_numerical = w_numerical @ Sigma @ w_numerical

# =========================
# 3. Output comparison
# =========================
print("=== Analytical Solution ===")
for i, w in enumerate(w_analytical):
    print(f"Asset {i+1}: {w:.6f}")
print(f"Return: {ret_analytical*100:.4f}%")
print(f"Variance: {var_analytical:.8f}")

print("\n=== Numerical Solution ===")
for i, w in enumerate(w_numerical):
    print(f"Asset {i+1}: {w:.6f}")
print(f"Return: {ret_numerical*100:.4f}%")
print(f"Variance: {var_numerical:.8f}")

# =========================
# 4. Differences
# =========================
print("\n=== Differences ===")
print("Weights diff:", np.round(w_analytical - w_numerical, 8))
print("Return diff:", abs(ret_analytical - ret_numerical))
print("Variance diff:", abs(var_analytical - var_numerical))




# Excess returns
mu_excess = mu - rf * ones

# Inverse covariance
Sigma_inv = np.linalg.inv(Sigma)

# Tangency (Sharpe-optimal) portfolio weights
w_tan = Sigma_inv @ mu_excess / (ones @ Sigma_inv @ mu_excess)

# Portfolio statistics
port_return = w_tan @ mu
port_variance = w_tan @ Sigma @ w_tan
port_std = np.sqrt(port_variance)
sharpe = (port_return - rf) / port_std

# =========================
# Output
# =========================
print("=== Sharpe-Optimal (Tangency) Portfolio ===\n")

for i, w in enumerate(w_tan):
    print(f"Weight Asset {i+1}: {w:.6f}")

print("\n--- Portfolio Characteristics ---")
print(f"Expected Return: {port_return*100:.4f}%")
print(f"Variance: {port_variance:.8f}")
print(f"Volatility (Std Dev): {port_std:.6f}")
print(f"Sharpe Ratio: {sharpe:.6f}")