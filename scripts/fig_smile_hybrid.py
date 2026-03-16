"""Figure: Hybrid SABR+GP smile vs market vs pure SABR, with GP uncertainty."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from volsurface import (
    DATA_DIR,
    calibrate_sabr,
    fit_gp_sigma0,
    hybrid_smile,
    invert_sabr_sigma0_curve,
    paper_style,
    prepare_market_data,
    sabr_smile,
    savefig,
)
import matplotlib.pyplot as plt


def main():
    md = prepare_market_data(
        DATA_DIR / "clm6-future-2026-03-13.csv",
        DATA_DIR / "clm6-options-2026-03-13.csv",
        T=0.1726,
        r=0.037,
        use_settle=True,
    )
    F, strikes, market_vols, T = md["F"], md["strikes"], md["market_vols"], md["T"]

    # Step 1: global SABR
    cal = calibrate_sabr(F, strikes, market_vols, T, beta=0.5)

    # Step 2: per-strike sigma0 inversion
    sigma0_curve = invert_sabr_sigma0_curve(
        F, strikes, T, market_vols, cal["alpha"], cal["beta"], cal["rho"]
    )

    # Step 3: GP on sigma0(K)
    gp = fit_gp_sigma0(strikes, sigma0_curve, prior_mean=cal["sigma0"])

    # Evaluate on fine grid
    K_fine = np.linspace(strikes.min() - 10, strikes.max() + 10, 400)
    sabr_vols = sabr_smile(F, K_fine, T, cal["sigma0"], cal["alpha"],
                           cal["beta"], cal["rho"])
    hybrid_vols, hybrid_std = hybrid_smile(
        F, K_fine, T, gp, cal["alpha"], cal["beta"], cal["rho"]
    )

    # Approximate vol uncertainty from sigma0 uncertainty (first-order)
    # delta_vol ≈ (dvol/dsigma0) * delta_sigma0 — we use a finite difference
    bump = 1e-4
    hybrid_vols_up, _ = hybrid_smile(
        F, K_fine, T, gp, cal["alpha"], cal["beta"], cal["rho"]
    )
    # Simpler: just use hybrid_std scaled roughly
    # For a proper propagation we'd need the Jacobian, but for the plot
    # we use a simple finite-difference approach at each point
    vol_std = np.zeros_like(K_fine)
    from volsurface import predict_gp_sigma0, sabr_implied_vol
    s0_mu, s0_std = predict_gp_sigma0(gp, K_fine)
    for i, (K, s0, ds0) in enumerate(zip(K_fine, s0_mu, s0_std)):
        v0 = sabr_implied_vol(F, K, T, s0, cal["alpha"], cal["beta"], cal["rho"])
        v_up = sabr_implied_vol(F, K, T, s0 + bump, cal["alpha"], cal["beta"], cal["rho"])
        dvds = (v_up - v0) / bump
        vol_std[i] = abs(dvds) * ds0

    with paper_style():
        fig, ax = plt.subplots()

        # Confidence band
        ax.fill_between(
            K_fine,
            (hybrid_vols - 2 * vol_std) * 100,
            (hybrid_vols + 2 * vol_std) * 100,
            alpha=0.2, color="C2", label="GP ±2σ",
        )

        ax.plot(strikes, market_vols * 100, "o", color="C0", zorder=5,
                label="Market")
        ax.plot(K_fine, sabr_vols * 100, "--", color="C1", label="SABR")
        ax.plot(K_fine, hybrid_vols * 100, "-", color="C2",
                label="SABR + GP")
        ax.axvline(F, color="grey", ls=":", lw=0.8)

        ax.set_xlabel("Strike ($/bbl)")
        ax.set_ylabel("Implied volatility (%)")
        ax.legend()
        savefig(fig, "smile_hybrid")


if __name__ == "__main__":
    main()
