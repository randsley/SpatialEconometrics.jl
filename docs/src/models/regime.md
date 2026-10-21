# Spatial Regime Models

## Overview

Spatial regime models allow regression coefficients to vary across pre-defined groups (regimes), capturing structural heterogeneity in the spatial process (Anselin, 1988).

```math
y = \rho W y + X_r \beta_r + \varepsilon
```

where each regime `r` gets its own intercept and slope coefficients.

## Exogenous Regimes

Assign observations to regimes based on known groupings:

```julia
labels = [i <= 100 ? 1 : 2 for i in 1:N]
regimes = ExogenousRegimes(labels)
m = fit_spatial_regime(W, X, y, regimes; spatial_lag=true, varnames=["gdp", "pop"])
```

## SKATER Clustering

For endogenous regime identification, SKATER partitions observations into spatially contiguous groups using a minimum spanning tree on the spatial connectivity graph:

```julia
regimes = skater_regimes(W, X, 3)  # 3 contiguous regimes
m = fit_spatial_regime(W, X, y, regimes)
```

Based on Assunção et al. (2006).

## Regime Stability Test

Test whether coefficients are equal across regimes using a likelihood ratio (Chow-type) test:

```julia
m_pooled = fit_spatial(CrossSection.SAR(), W, X, y)
m_regime = fit_spatial_regime(W, X, y, regimes)
ct = regime_chow_test(m_pooled, m_regime)
```

## API

See the full [API Reference](@ref) for `fit_spatial_regime`, `skater_regimes`, and `regime_chow_test`.
