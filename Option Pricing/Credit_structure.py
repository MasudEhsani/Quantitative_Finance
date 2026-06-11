import numpy as np
from scipy.optimize import brentq

# =========================================================
# 1. MARKET DATA
# =========================================================
spot_rates = np.array([
    0.0300, 0.0310, 0.0320, 0.0330, 0.0340,
    0.0350, 0.0355, 0.0360, 0.0365, 0.0370
])

n = len(spot_rates)
face = 100

market_zcb = np.array([
    face / (1 + r) ** (t + 1)
    for t, r in enumerate(spot_rates)
])

# =========================================================
# 2. BDT PARAMETERS
# =========================================================
b = 0.1

calibrated_a = np.zeros(n)

# =========================================================
# 3. SHORT RATE TREE
# =========================================================
def build_tree(a):
    r = np.zeros((n+1, n+1))
    for i in range(n):
        for j in range(i+1):
            r[i, j] = a[i] * np.exp(b * j)
    return r

# =========================================================
# 4. ZERO COUPON PRICER
# =========================================================
def price_zcb(a, maturity):
    r = build_tree(a)
    P = np.zeros((n+1, n+1))

    # terminal payoff
    for j in range(maturity + 1):
        P[maturity, j] = face

    # backward induction
    for i in reversed(range(maturity)):
        for j in range(i+1):
            P[i, j] = (0.5 * (P[i+1, j] + P[i+1, j+1])) / (1 + r[i, j])

    return P[0, 0]

# =========================================================
# 5. BDT BOOTSTRAP CALIBRATION (FIXED)
# =========================================================
for i in range(n):

    def objective(x):
        a_temp = calibrated_a.copy()
        a_temp[i] = x
        return price_zcb(a_temp, i+1) - market_zcb[i]

    # ensure valid bracket
    a_low, a_high = 1e-6, 1.0

    if objective(a_low) * objective(a_high) > 0:
        raise ValueError(f"No root bracket at i={i}")

    calibrated_a[i] = brentq(objective, a_low, a_high, xtol=1e-12)

    print(f"a[{i}] = {calibrated_a[i]:.10f}")

# =========================================================
# 6. BUILD FINAL TREE
# =========================================================
r = build_tree(calibrated_a)

# =========================================================
# 7. SWAP PRICING (PAYER SWAP)
# =========================================================
notional = 1_000_000
K = 0.039
exercise_time = 3

def swap_value():

    V = np.zeros((n+1, n+1))

    # terminal condition: no value after maturity
    for j in range(n+1):
        V[n, j] = 0.0

    # backward induction ONLY for swap life: t = 9 → 3
    for i in reversed(range(exercise_time, n)):

        for j in range(i+1):

            df = 1 / (1 + r[i, j])

            cont = 0.5 * (V[i+1, j] + V[i+1, j+1])

            # -----------------------------
            # CASHFLOWS AT t = i+1 (IN ARREARS)
            # -----------------------------

            fixed_cf = notional * K

            # floating uses SHORT RATE FROM PREVIOUS PERIOD (as stated)
            floating_cf = notional * r[i, j]

            # payer swap = pay fixed, receive floating
            V[i, j] = df * (cont + floating_cf - fixed_cf)

    return V

swap = swap_value()

# =========================================================
# 8. SWAPTION PRICING (AMERICAN EXERCISE AT t=3)
# =========================================================
swaption = np.zeros((n+1, n+1))

for j in range(exercise_time + 1):
    swaption[exercise_time, j] = max(swap[exercise_time, j], 0)

for i in reversed(range(exercise_time)):
    for j in range(i+1):

        df = 1 / (1 + r[i, j])
        cont = 0.5 * (swaption[i+1, j] + swaption[i+1, j+1])

        swaption[i, j] = df * cont

price = swaption[0, 0]
print(price)