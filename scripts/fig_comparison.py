"""Figure: Direct GP on σ_BS(K) vs hybrid SABR+GP — showing why the
decomposition matters, especially in the wings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from volsurface import (
    DATA_DIR,
    calibrate_sabr,
    fit_gp_sigma0,
    hybrid_smile,
    invert_sabr_sigma0_curve,
    paper_style,
    prepare_market_data,
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

    # --- Hybrid SABR+GP ---
    cal = calibrate_sabr(F, strikes, market_vols, T, beta=0.5)
    sigma0_curve = invert_sabr_sigma0_curve(
        F, strikes, T, market_vols, cal["alpha"], cal["beta"], cal["rho"]
    )
    gp_hybrid = fit_gp_sigma0(strikes, sigma0_curve, prior_mean=cal["sigma0"])

    K_fine = np.linspace(strikes.min() - 15, strikes.max() + 15, 400)
    hybrid_vols, _ = hybrid_smile(
        F, K_fine, T, gp_hybrid, cal["alpha"], cal["beta"], cal["rho"]
    )

    # --- Direct GP on σ_BS(K) ---
    kernel_direct = (
        ConstantKernel(0.01) * RBF(length_scale=10.0, length_scale_bounds=(1.0, 50.0))
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-3))
    )
    gp_direct = GaussianProcessRegressor(
        kernel=kernel_direct, n_restarts_optimizer=10, alpha=0.0,
    )
    gp_direct.fit(strikes.reshape(-1, 1), market_vols)
    direct_vols, direct_std = gp_direct.predict(K_fine.reshape(-1, 1), return_std=True)

    with paper_style():
        fig, ax = plt.subplots()
        ax.plot(strikes, market_vols * 100, "o", color="C0", zorder=5,
                label="Market")
        ax.plot(K_fine, direct_vols * 100, "-", color="C3",
                label=r"GP on $\sigma_{\mathrm{BS}}(K)$")
        ax.fill_between(K_fine,
                         (direct_vols - 2*direct_std) * 100,
                         (direct_vols + 2*direct_std) * 100,
                         alpha=0.15, color="C3")
        ax.plot(K_fine, hybrid_vols * 100, "-", color="C2",
                label="SABR + GP")
        ax.axvline(F, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("Strike ($/bbl)")
        ax.set_ylabel("Implied volatility (%)")
        ax.legend()
        savefig(fig, "comparison_direct_vs_hybrid")


if __name__ == "__main__":
    main()
