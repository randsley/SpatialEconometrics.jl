# Geographically Weighted Regression

## GWR

Estimates locally varying coefficients using a kernel-weighted regression at each observation:

```julia
m = fit_gwr(X, y, coords; bandwidth=AdaptiveBandwidth(40))

# Local coefficients: m.coef_local (N × K+1 matrix)
# Global R²: m.R2
```

### Bandwidth Types

- `AdaptiveBandwidth(k)` — k nearest neighbors (recommended)
- `FixedBandwidth(d)` — fixed distance threshold

### Kernel Functions

- `GaussianKernel()` (default)
- `BisquareKernel()`
- `ExponentialKernel()`

## MGWR (Multiscale GWR)

Each covariate gets its own optimized bandwidth:

```julia
m = fit_mgwr(X, y, coords)
# m.bandwidths — per-variable bandwidths
# m.coef_local — N × (K+1) local coefficients
```

MGWR is preferred when covariates operate at different spatial scales (e.g., temperature varies regionally but policy varies locally).
