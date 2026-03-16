"""Figure: Per-strike σ₀(K) curve — shows how flat it is when SABR fits well."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from volsurface import (
    DATA_DIR,
    calibrate_sabr,
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

    cal = calibrate_sabr(F, strikes, market_vols, T, beta=0.5)

    sigma0_curve = invert_sabr_sigma0_curve(
        F, strikes, T, market_vols, cal["alpha"], cal["beta"], cal["rho"]
    )

    with paper_style():
        fig, ax = plt.subplots()
        ax.plot(strikes, sigma0_curve, "s-", color="C2", ms=4,
                label=r"$\sigma_0(K)$ per-strike")
        ax.axhline(cal["sigma0"], color="C1", ls="--", lw=1.0,
                    label=rf"Global $\hat{{\sigma}}_0$ = {cal['sigma0']:.4f}")
        ax.axvline(F, color="grey", ls=":", lw=0.8)
        ax.set_xlabel("Strike ($/bbl)")
        ax.set_ylabel(r"$\sigma_0$")
        ax.legend()
        savefig(fig, "sigma0_curve")


if __name__ == "__main__":
    main()
