# Volatility Surface Pipeline

A robust data pipeline for reconstructing missing Implied Volatility (IV) in options data. It leverages a two-phase approach: cross-strike spatial imputation via Akima 1D interpolation, followed by a multi-scale rolling Principal Component Analysis (PCA) ensemble for temporal smoothing and gap-filling.

## Overview
This repository contains the optimal configuration (Pipeline 35) for IV reconstruction, which has proven to yield the best results across k-fold cross-validation and parameter sweeps.

### Key Features
- **Spatial Imputation (Cross-Strike)**: Utilizes `Akima1DInterpolator` to preserve the natural volatility smile across strikes, with linear extrapolation for boundary conditions.
- **Temporal Imputation (Multi-Scale Rolling PCA)**: Employs a rolling PCA with a Gaussian weighting window across an ensemble of 5 window sizes `[90, 105, 120, 135, 150]`.
- **Optimal Parameters**: Uses an `alpha` momentum of `0.7` and `8` iterations per window to allow PCA to rapidly inject structural corrections for consecutively missing data.
- **Signal Smoothing**: Applies a Savitzky-Golay filter to remove localized jitter.

## Usage
The core logic resides in `run_pipeline35_optimal.py`. It requires a `dataset.csv` with options IV data over time.

```bash
python run_pipeline35_optimal.py
```

This will output `submission_pipeline35.csv` containing the final imputed values.

## Requirements
- Python 3.8+
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
