# Fitting Volatility Smiles with SABR and Gaussian Processes

A Python implementation of a hybrid method for interpolating and extrapolating the implied volatility smile, combining the SABR stochastic volatility model with Gaussian Process regression.

This is the companion code to the working paper: *Fitting Volatility Smiles with Structural Models and Machine Learning* (Parcell, 2026).

## The idea

Standard SABR calibration captures the broad shape of the volatility smile but rarely fits every market quote exactly. A Gaussian Process can interpolate perfectly, but reverts to a flat prior in the wings where data is sparse.

The hybrid approach works by:

1. Calibrating SABR globally to the observed smile.
2. At each market strike, inverting the SABR formula to find the value of the initial volatility parameter σ₀ that exactly reproduces the market quote.
3. Fitting a GP to the resulting function σ₀(K), which is nearly flat when SABR fits well.
4. Evaluating the model at any strike by plugging the GP-predicted σ₀ back into SABR.

This gives you exact calibration at observed strikes, SABR-shaped extrapolation in the wings, and GP uncertainty quantification — without asking the GP to learn the full smile shape from scratch.

## Project structure

```
├── volsurface.py            # Core module: Black IV, SABR, GP fitting, hybrid model
├── build_figures.py          # Run all figure scripts at once
├── scripts/
│   ├── fig_smile_sabr_fit.py         # Market vs SABR fit
│   ├── fig_sigma0_curve.py           # Strike-dependent σ₀ inversion
│   ├── fig_smile_hybrid.py           # Hybrid SABR+GP smile with confidence bands
│   ├── fig_residuals.py              # SABR vs hybrid residuals
│   └── fig_comparison.py             # Hybrid vs direct GP comparison
├── data/
│   ├── futures.csv                   # Futures settlement data
│   └── options.csv                   # Options settlement data
└── output/figures/                   # Generated PDF figures
```

## Requirements

- Python ≥ 3.10
- numpy, scipy, pandas, matplotlib, scikit-learn

## Installation

```bash
git clone https://github.com/edparcell/volatility_smile_sabr_gp.git
cd volatility_smile_sabr_gp
pip install -e .
```

## Usage

Generate all figures from the paper:

```bash
python build_figures.py
```

Or run individual figure scripts:

```bash
python scripts/fig_smile_hybrid.py
```

Output PDFs are written to `output/figures/`.

## Data

The included dataset is settlement prices for options on the June 2026 WTI crude oil future (CLM6) from March 13, 2026, filtered to strikes with volume above 100 contracts.

## License

MIT
