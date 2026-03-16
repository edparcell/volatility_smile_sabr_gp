"""Figure: Market implied vol smile vs globally calibrated SABR."""

import sys
from pathlib import Path

# Ensure repo root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from volsurface import (
    DATA_DIR,
    calibrate_sabr,
    paper_style,
    prepare_market_data,
    sabr_smile,
    savefig,
)
import matplotlib.pyplot as plt


def main():
    # --- Data ---
    md = prepare_market_data(
        DATA_DIR / "clm6-future-2026-03-13.csv",
        DATA_DIR / "clm6-options-2026-03-13.csv",
        T=0.1726,
        r=0.037,
        use_settle=True,
    )
    F, strikes, market_vols, T = md["F"], md["strikes"], md["market_vols"], md["T"]

    # --- Calibrate SABR ---
    cal = calibrate_sabr(F, strikes, market_vols, T, beta=0.5)
    print(f"SABR calibration: σ0={cal['sigma0']:.4f}  α={cal['alpha']:.4f}  "
          f"ρ={cal['rho']:.4f}  RMSE={cal['rmse']:.6f}")

    # --- Evaluate SABR on a fine grid ---
    K_fine = np.linspace(strikes.min() - 5, strikes.max() + 5, 300)
    sabr_vols = sabr_smile(F, K_fine, T, cal["sigma0"], cal["alpha"],
                           cal["beta"], cal["rho"])

    # --- Plot ---
    with paper_style():
        fig, ax = plt.subplots()
        ax.plot(strikes, market_vols * 100, "o", color="C0", label="Market",
                zorder=5)
        ax.plot(K_fine, sabr_vols * 100, "-", color="C1",
                label=f"SABR (β={cal['beta']:.1f})")
        ax.axvline(F, color="grey", ls=":", lw=0.8, label=f"F = {F:.2f}")
        ax.set_xlabel("Strike ($/bbl)")
        ax.set_ylabel("Implied volatility (%)")
        ax.legend()
        savefig(fig, "smile_sabr_fit")


if __name__ == "__main__":
    main()
