# Volatility Surface Pipeline

A robust machine learning pipeline for reconstructing missing Implied Volatility (IV) in options data.

## Overview
**Pipeline 35** is the optimal configuration for IV reconstruction, utilizing a two-phase approach (Spatial and Temporal) to achieve minimal error.

### Pipeline Architecture
1. **Data Preprocessing**: Parses option tickers and formats data for time-series analysis.
2. **Spatial Imputation (Cross-Strike)**: Uses `Akima1DInterpolator` to preserve the natural volatility smile, with linear extrapolation for boundaries.
3. **Temporal Imputation (Multi-Scale Rolling PCA)**: Employs a rolling PCA ensemble (windows: `[90, 105, 120, 135, 150]`) with Gaussian weighting. Tuned with `alpha=0.7` and `8` iterations for optimal structural gap-filling.
4. **Signal Smoothing**: Applies a Savitzky-Golay filter to remove noise, softly blended with PCA predictions.
5. **Validation**: Enforces strict positive IV bounds (minimum `0.001`) with forward/backward fill failsafes.

## Usage
Ensure `dataset.csv` is in the same directory, then run:

```bash
python run_pipeline35_optimal.py
```
Output: `submission_pipeline35.csv`

## Requirements
- Python 3.8+
- `pandas`, `numpy`, `scipy`, `scikit-learn`
