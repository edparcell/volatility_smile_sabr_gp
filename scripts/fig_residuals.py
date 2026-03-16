"""Figure: Fitting residuals at market strikes — SABR vs SABR+GP."""

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

    cal = calibrate_sabr(F, strikes, market_vols, T, beta=0.5)
    sabr_at_strikes = sabr_smile(F, strikes, T, cal["sigma0"], cal["alpha"],
                                 cal["beta"], cal["rho"])

    sigma0_curve = invert_sabr_sigma0_curve(
        F, strikes, T, market_vols, cal["alpha"], cal["beta"], cal["rho"]
    )
    gp = fit_gp_sigma0(strikes, sigma0_curve, prior_mean=cal["sigma0"])
    hybrid_at_strikes, _ = hybrid_smile(
        F, strikes, T, gp, cal["alpha"], cal["beta"], cal["rho"]
    )

    sabr_resid = (sabr_at_strikes - market_vols) * 100  # in vol points
    hybrid_resid = (hybrid_at_strikes - market_vols) * 100

    with paper_style():
        fig, ax = plt.subplots()
        ax.stem(strikes, sabr_resid, linefmt="C1-", markerfmt="C1o",
                basefmt="k-", label="SABR")
        ax.stem(strikes + 0.3, hybrid_resid, linefmt="C2-", markerfmt="C2s",
                basefmt="k-", label="SABR + GP")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Strike ($/bbl)")
        ax.set_ylabel("Residual (vol points)")
        ax.legend()

        print(f"  SABR RMSE:   {np.sqrt(np.mean(sabr_resid**2)):.4f} vol pts")
        print(f"  Hybrid RMSE: {np.sqrt(np.mean(hybrid_resid**2)):.4f} vol pts")

        savefig(fig, "residuals")


if __name__ == "__main__":
    main()
