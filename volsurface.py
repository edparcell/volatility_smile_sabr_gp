"""
volsurface — Black-Scholes, SABR, and GP-based volatility smile fitting.

This module provides:
  - Black-Scholes pricing and implied vol inversion
  - SABR implied vol (Hagan et al. asymptotic expansion)
  - SABR calibration (global fit and per-strike sigma0 inversion)
  - GP interpolation of strike-dependent sigma0
  - Data loading helpers for CME CSV exports
  - Matplotlib styling for publication-quality figures
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "output" / "figures"


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
PAPER_RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (5.5, 3.8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}


@contextlib.contextmanager
def paper_style():
    """Context manager that applies publication-quality rcParams."""
    with plt.rc_context(PAPER_RCPARAMS):
        yield


def savefig(fig: plt.Figure, name: str, fmt: str = "pdf") -> Path:
    """Save a figure to output/figures/ and return the path."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cme_future(path: Path | str) -> dict:
    """Load a CME futures CSV and return a dict with the forward price etc."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    row = df.iloc[0]
    return {
        "label": row["future"],
        "last": float(row["last"]),
        "prior_settle": float(row["prior_settle"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "volume": int(row["volume"]),
    }


def load_cme_options(path: Path | str) -> pd.DataFrame:
    """Load a CME options CSV and return a tidy DataFrame.

    Columns: strike, call_last, call_prior_settle, call_volume,
             put_last, put_prior_settle, put_volume.
    Missing prices ('-') become NaN.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in df.columns:
        if col != "strike":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------
def bs_price(F: float, K: float, T: float, sigma: float, r: float = 0.0,
             cp: int = 1) -> float:
    """Black-76 (forward) option price. cp=+1 call, cp=-1 put."""
    if T <= 0 or sigma <= 0:
        return max(cp * (F - K), 0.0)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df = np.exp(-r * T)
    return df * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))


def bs_vega(F: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 vega (dPrice/dSigma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_t = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
    return F * np.exp(-r * T) * norm.pdf(d1) * sqrt_t


def implied_vol(price: float, F: float, K: float, T: float,
                r: float = 0.0, cp: int = 1,
                lo: float = 0.001, hi: float = 5.0) -> float | None:
    """Invert Black-76 price to find implied vol via Brent's method.

    Returns None if no solution exists in [lo, hi].
    """
    intrinsic = max(cp * (F - K), 0.0) * np.exp(-r * T)
    if price <= intrinsic + 1e-12:
        return None
    try:
        return brentq(lambda s: bs_price(F, K, T, s, r, cp) - price, lo, hi,
                       xtol=1e-10, maxiter=200)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# SABR
# ---------------------------------------------------------------------------
def sabr_implied_vol(F: float, K: float, T: float,
                     sigma0: float, alpha: float, beta: float,
                     rho: float) -> float:
    """Hagan et al. SABR implied vol approximation.

    Parameters
    ----------
    F : forward price
    K : strike
    T : time to expiry (years)
    sigma0 : initial volatility
    alpha : vol-of-vol
    beta : CEV exponent (0 <= beta <= 1)
    rho : correlation (-1 < rho < 1)
    """
    eps = 1e-8
    if abs(F - K) < eps:
        # ATM limit
        Fb = F**beta
        v = (sigma0 / Fb) * (
            1.0 + T * (
                ((1.0 - beta)**2 / 24.0) * sigma0**2 / F**(2.0 - 2.0*beta)
                + 0.25 * rho * beta * alpha * sigma0 / F**(1.0 - beta)
                + (2.0 - 3.0 * rho**2) / 24.0 * alpha**2
            )
        )
        return v

    Fmid = 0.5 * (F + K)
    lnFK = np.log(F / K)

    # CEV function and derivatives
    C_mid = Fmid**beta
    gamma1 = beta / Fmid
    gamma2 = -beta * (1.0 - beta) / Fmid**2

    # zeta
    if abs(beta - 1.0) < eps:
        zeta = (alpha / sigma0) * lnFK
    else:
        zeta = (alpha / (sigma0 * (1.0 - beta))) * (
            F**(1.0 - beta) - K**(1.0 - beta)
        )

    # D(zeta)
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * zeta + zeta**2)
    D_zeta = np.log((sqrt_term + zeta - rho) / (1.0 - rho))
    if abs(D_zeta) < eps:
        D_zeta = 1.0  # limit as zeta -> 0

    # Correction term
    term1 = (2.0 * gamma2 - gamma1**2 + 1.0 / Fmid**2) / 24.0 * (
        sigma0 * C_mid / alpha
    ) ** 2
    term2 = rho * gamma1 / 4.0 * (sigma0 * C_mid / alpha)
    term3 = (2.0 - 3.0 * rho**2) / 24.0
    correction = 1.0 + (term1 + term2 + term3) * T * alpha**2

    return alpha * lnFK / D_zeta * correction


def sabr_smile(F: float, strikes: np.ndarray, T: float,
               sigma0: float, alpha: float, beta: float,
               rho: float) -> np.ndarray:
    """Vectorized SABR implied vol across an array of strikes."""
    return np.array([
        sabr_implied_vol(F, K, T, sigma0, alpha, beta, rho)
        for K in strikes
    ])


def calibrate_sabr(
    F: float, strikes: np.ndarray, market_vols: np.ndarray, T: float,
    beta: float = 0.5,
    x0: tuple[float, float, float] | None = None,
) -> dict:
    """Calibrate SABR (sigma0, alpha, rho) to market vols with beta fixed.

    Returns dict with keys: sigma0, alpha, rho, beta, rmse.
    """
    if x0 is None:
        atm_vol = np.interp(F, strikes, market_vols)
        x0 = (atm_vol, 0.5, -0.3)

    def objective(params):
        s0, a, r = params
        if s0 <= 0 or a <= 0 or r <= -1 or r >= 1:
            return 1e10
        model = sabr_smile(F, strikes, T, s0, a, beta, r)
        return np.sum((model - market_vols) ** 2)

    res = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-10, "fatol": 1e-14},
    )
    s0, a, r = res.x
    model = sabr_smile(F, strikes, T, s0, a, beta, r)
    rmse = np.sqrt(np.mean((model - market_vols) ** 2))

    return {"sigma0": s0, "alpha": a, "rho": r, "beta": beta, "rmse": rmse}


# ---------------------------------------------------------------------------
# Per-strike sigma0 inversion
# ---------------------------------------------------------------------------
def invert_sabr_sigma0(
    F: float, K: float, T: float,
    target_vol: float,
    alpha: float, beta: float, rho: float,
    lo: float = 0.001, hi: float = 10.0,
) -> float:
    """Find sigma0 such that SABR(sigma0) = target_vol at strike K."""
    return brentq(
        lambda s0: sabr_implied_vol(F, K, T, s0, alpha, beta, rho) - target_vol,
        lo, hi, xtol=1e-12, maxiter=200,
    )


def invert_sabr_sigma0_curve(
    F: float, strikes: np.ndarray, T: float,
    market_vols: np.ndarray,
    alpha: float, beta: float, rho: float,
) -> np.ndarray:
    """Invert SABR at each strike to get sigma0(K) curve."""
    return np.array([
        invert_sabr_sigma0(F, K, T, mv, alpha, beta, rho)
        for K, mv in zip(strikes, market_vols)
    ])


# ---------------------------------------------------------------------------
# GP interpolation of sigma0(K)
# ---------------------------------------------------------------------------
def fit_gp_sigma0(
    strikes: np.ndarray,
    sigma0_values: np.ndarray,
    prior_mean: float,
    length_scale_bounds: tuple[float, float] = (1.0, 50.0),
    noise_level: float = 1e-6,
) -> GaussianProcessRegressor:
    """Fit a GP to the sigma0(K) curve.

    The GP prior mean is the globally calibrated sigma0. We subtract it,
    fit a zero-mean GP, and wrap the result so predictions add it back.

    Returns a fitted GaussianProcessRegressor (on the de-meaned targets).
    Store prior_mean as gp.prior_mean_ for later use.
    """
    X = strikes.reshape(-1, 1)
    y = sigma0_values - prior_mean

    kernel = (
        ConstantKernel(1e-4, constant_value_bounds=(1e-8, 1e-1))
        * RBF(length_scale=10.0, length_scale_bounds=length_scale_bounds)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e-3))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=False,
        alpha=0.0,
    )
    gp.fit(X, y)
    gp.prior_mean_ = prior_mean
    return gp


def predict_gp_sigma0(
    gp: GaussianProcessRegressor,
    strikes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict sigma0(K) from the fitted GP.

    Returns (mean, std) arrays with the prior mean added back.
    """
    X = strikes.reshape(-1, 1)
    mu, std = gp.predict(X, return_std=True)
    return mu + gp.prior_mean_, std


# ---------------------------------------------------------------------------
# Full hybrid model: SABR + GP
# ---------------------------------------------------------------------------
def hybrid_smile(
    F: float, strikes_eval: np.ndarray, T: float,
    gp: GaussianProcessRegressor,
    alpha: float, beta: float, rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the SABR+GP hybrid smile at arbitrary strikes.

    Returns (implied_vols, sigma0_std) arrays.
    """
    sigma0_mu, sigma0_std = predict_gp_sigma0(gp, strikes_eval)
    vols = np.array([
        sabr_implied_vol(F, K, T, s0, alpha, beta, rho)
        for K, s0 in zip(strikes_eval, sigma0_mu)
    ])
    return vols, sigma0_std


# ---------------------------------------------------------------------------
# Data preparation: market vols from CME CSVs
# ---------------------------------------------------------------------------
def implied_forward_from_options(opts: pd.DataFrame, r: float,
                                  T: float,
                                  use_settle: bool) -> float:
    """Infer the forward price from put-call parity: C - P = DF*(F - K).

    Uses the median implied forward across strikes where both call and put
    prices are available, excluding deep OTM/ITM strikes where prices may
    be stale.
    """
    c_col = "call_prior_settle" if use_settle else "call_last"
    p_col = "put_prior_settle" if use_settle else "put_last"
    df = np.exp(-r * T)

    implied_forwards = []
    for _, row in opts.iterrows():
        K = row["strike"]
        c = row[c_col]
        p = row[p_col]
        if pd.isna(c) or pd.isna(p) or c <= 0 or p <= 0:
            continue
        F_impl = K + (c - p) / df
        implied_forwards.append(F_impl)

    if not implied_forwards:
        raise ValueError("Cannot infer forward: no strikes with both C and P prices")
    return float(np.median(implied_forwards))


def prepare_market_data(
    future_path: Path | str,
    options_path: Path | str,
    T: float,
    r: float,
    min_volume: int = 0,
    use_settle: bool = True,
    forward_override: float | None = None,
) -> dict:
    """Load CME data and compute implied vols.

    Parameters
    ----------
    future_path, options_path : paths to CSVs
    T : time to expiry in years
    r : risk-free rate
    min_volume : minimum option volume to include
    use_settle : if True use prior_settle prices, else use last prices
    forward_override : if given, use this as F instead of inferring

    Returns dict with keys: F, F_last, T, r, strikes, market_vols, call_or_put.
    """
    fut = load_cme_future(future_path)
    opts = load_cme_options(options_path)

    # Infer forward from put-call parity on the option prices, since
    # the futures "last" price may not match the option settlement time.
    if forward_override is not None:
        F = forward_override
    else:
        F = implied_forward_from_options(opts, r, T, use_settle)

    price_col_call = "call_prior_settle" if use_settle else "call_last"
    price_col_put = "put_prior_settle" if use_settle else "put_last"

    strikes = []
    vols = []
    cps = []

    for _, row in opts.iterrows():
        K = row["strike"]
        # Use OTM options: calls for K >= F, puts for K < F
        if K >= F:
            cp = 1
            price = row[price_col_call]
            vol = row["call_volume"]
        else:
            cp = -1
            price = row[price_col_put]
            vol = row["put_volume"]

        if pd.isna(price) or price <= 0:
            continue
        if vol < min_volume:
            continue

        iv = implied_vol(price, F, K, T, r, cp)
        if iv is not None and 0.01 < iv < 3.0:
            strikes.append(K)
            vols.append(iv)
            cps.append(cp)

    return {
        "F": F,
        "F_last": fut["last"],
        "T": T,
        "r": r,
        "strikes": np.array(strikes),
        "market_vols": np.array(vols),
        "call_or_put": np.array(cps),
    }
