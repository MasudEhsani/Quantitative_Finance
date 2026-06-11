import numpy as np

# -----------------------------
# Parameters
# -----------------------------
n = 10
r0 = 0.05
u = 1.1
d = 0.9
q = 0.5

# -----------------------------
# Short-rate lattice
# r[i,j] = short rate at time i, state j
# -----------------------------
r = np.zeros((n+1, n+1))

# initial node
r[0, 0] = r0

# build tree
for i in range(1, n):
    for j in range(i + 1):
        r[i, j] = r0 * (u ** j) * (d ** (i - j))

# -----------------------------
# Display tree
# -----------------------------
np.set_printoptions(precision=4, suppress=True)
print("Binomial short-rate tree:")
print(r)



# =========================================================
# 1. BINOMIAL SHORT RATE TREE
# =========================================================
n = 10
r0 = 0.05
u = 1.1
d = 0.9

r = np.zeros((n+1, n+1))
r[0, 0] = r0

for i in range(1, n+1):
    for j in range(i+1):
        r[i, j] = r0 * (u ** j) * (d ** (i - j))

# =========================================================
# 2. CREDIT PARAMETERS
# =========================================================
F = 100
R = 0.2

a = 0.01
b = 1.01

def hazard(i, j):
    return a * (b ** (j - i / 2))

# =========================================================
# 3. DEFAULTABLE BOND PRICING
# =========================================================
P = np.zeros((n+1, n+1))

# terminal payoff
for j in range(n+1):
    P[n, j] = F

# backward induction
for i in reversed(range(n)):

    for j in range(i+1):

        h = hazard(i, j)

        cont = 0.5 * (P[i+1, j] + P[i+1, j+1])

        P[i, j] = (1 / (1 + r[i, j])) * (
            (1 - h) * cont + h * R * F
        )

# =========================================================
# 4. RESULT
# =========================================================
print("Defaultable zero-coupon bond price:", P[0, 0])