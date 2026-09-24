#!/usr/bin/env python3
"""
Generate data/dja_cap.csv: a synthetic, reproducible placeholder for the
course's DJI-30 market-cap dataset, which wasn't available in this
environment.

The panel is simulated directly from the GMR model this project estimates
(known kappa / W / sigma), then rescaled to plausible dollar market caps
per ticker. This makes it possible to run Parts 1-2 end-to-end out of the
box, and doubles as a parameter-recovery check: fitting the model back on
this file should recover values close to the ground truth printed below.

To use real data instead, drop a same-format ``dja_cap.csv`` (no date
column; one column per DJI ticker; consecutive business days starting
2010-01-04) at ``data/dja_cap.csv`` -- everything downstream is unaffected.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.irl_market.data_loader import DJI_BASE_CAP_BILLIONS, DJI_SECTORS
from src.irl_market.simulate import simulate_gmr_with_sma_signals, to_dollar_caps

# Ground-truth parameters for the synthetic data-generating process.
GROUND_TRUTH = dict(
    kappa=0.06,
    w0=1.0,
    w_sma_short=0.35,
    w_sma_long=-0.20,
    sigma=0.016,
    seed=42,
)
N_DAYS = 2016  # ~8 trading years, matching the original 2010-01-04 start


def main():
    tickers = list(DJI_SECTORS.keys())
    assert len(tickers) == 30, f"expected 30 DJI tickers, got {len(tickers)}"

    print("Simulating synthetic DJI-30 market caps with ground-truth parameters:")
    for k, v in GROUND_TRUTH.items():
        print(f"  {k} = {v}")

    x_norm = simulate_gmr_with_sma_signals(tickers, N_DAYS, **GROUND_TRUTH)
    dollar_caps = to_dollar_caps(x_norm, DJI_BASE_CAP_BILLIONS)

    out_path = PROJECT_ROOT / "data" / "dja_cap.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Match the original file's format: no date column, dates reconstructed
    # by the loader via a fixed business-day range starting 2010-01-04.
    dollar_caps.reset_index(drop=True).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({dollar_caps.shape[0]} days x {dollar_caps.shape[1]} tickers)")
    print(dollar_caps.iloc[:3, :6].round(1))


if __name__ == "__main__":
    main()
