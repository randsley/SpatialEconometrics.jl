# Semiparametric Spatial Models

API reference for `SemiparametricSpatial`. See
[Semiparametric Models](../models/semiparametric.md) for the discussion,
including what the identification diagnostic can and cannot claim.

## Module

```@docs
SpatialEconometrics.SemiparametricSpatial
```

## The smoother and the identification diagnostic

```@docs
local_polynomial_smoother
SpectralOverlapResult
spectral_overlap
```

## Partially linear SAR (Su & Jin 2010)

```@docs
PartiallyLinearSARModel
fit_partially_linear_sar
```

## Closed-form ρ (Smirnov 2022; Geniaux 2026)

```@docs
ClosedFormRhoResult
closed_form_rho
```

## Time-varying coefficients (Liang, Gao & Gong 2022)

```@docs
TVCoefSARModel
fit_tv_coef_sar
```
