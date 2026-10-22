# Getting Started

## Installation

```julia
using Pkg
Pkg.add("SpatialEconometrics")
```

## Basic Workflow

A typical spatial econometrics workflow follows these steps:

### 1. Construct a spatial weight matrix

```julia
using SpatialEconometrics

# From a regular grid
W = make_rook_weights(20)   # 20×20 grid → 400 observations
W = make_queen_weights(20)  # includes diagonal neighbors

# From coordinates
coords = hcat(lon, lat)
W = knn_weights(coords, 5)                          # 5 nearest neighbors
W = distance_weights(coords, 100.0; decay=:inverse)  # inverse distance, 100km cutoff
W = band_weights(coords, 10.0, 50.0)                # binary, 10-50km band

# Inspect
sc = spatial_connectivity(W)
```

### 2. Test for spatial dependence

```julia
# Moran's I on raw data or OLS residuals
mr = moran_global(y, W)

# LM tests to choose between lag and error specifications
lm = lm_tests(y, hcat(ones(length(y)), X), W)
```

### 3. Estimate a spatial model

```julia
# Cross-sectional SAR
m_sar = fit_spatial(SAR(), W, X, y)

# Cross-sectional SDM (Spatial Durbin)
m_sdm = fit_spatial(SDM(), W, X, y)

# Spatial Error Model
m_sem = fit_spatial_error(W, X, y)
```

### 4. Compare and select models

```julia
# Information criteria
spatial_aic(m_sar)
spatial_bic(m_sdm)

# Likelihood ratio test (SAR nested in SDM)
lr = lr_test(m_sar, m_sdm)

# Formatted comparison table
println(model_comparison_table(m_sar, m_sdm, m_sem; names=["SAR","SDM","SEM"]))
```

### 5. Compute spatial effects

```julia
effects = spatial_effects(m_sdm)
# effects.direct, effects.indirect, effects.total
```

## Specification Tags

Model variants, fixed-effects specs and log-determinant methods are single
shared types, exported directly. You never qualify them with a module name:

```julia
fit_spatial(SAR(), W, X, y)                                  # SAR, SDM
fit_spatial_panel(SDM(), W, X, y, N, T; fe = TwoWayFE())     # NoFE, IndivFE, TimeFE, TwoWayFE
fit_spatial(SAR(), W, X, y; method = ChebyshevApprox(order = 8))  # ExactLU, ChebyshevApprox
```

## Submodules

The submodules organise the source and are **not** exported, so they cannot
collide with registered packages that share a name. Reach one explicitly if
you need an internal helper:

```julia
const CS = SpatialEconometrics.CrossSection
```

| Full name | Internal alias |
|---|---|
| `SpatialEconometricsPrototype` | `CrossSection` |
| `SpatialPanelEconometrics` | `StaticPanel` |
| `DynamicSpatialPanelEconometrics` | `DynamicPanel` |
| `SpatialGS2SLSPanel` | `GS2SLSPanel` |
| `DynamicPanelSystemGMM` | `SystemGMMMod` |
| `SpatialWeights` | `WeightsMod` |
| `SpatialModelSelection` | `SelectionMod` |
| `SpatialPanelDiagnostics` | `PanelDiagMod` |
| `SpatialDiagnostics` | `DiagMod` |
| `SpatialErrorModel` | `SEMMod` |
| `GWREstimation` | `GWRMod` |
| `SpatialGravityModels` | `GravityMod` |
