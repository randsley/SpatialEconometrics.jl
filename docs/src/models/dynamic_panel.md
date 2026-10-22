# Dynamic Spatial Panel Models

## Specification

```math
y_{it} = \rho W y_{it} + \gamma y_{i,t-1} + \eta W y_{i,t-1} + X_{it} \beta + \mu_i + \varepsilon_{it}
```

## Usage

```julia
m = fit_dynamic_spatial_panel(SAR(), IndivFE(),
    W, X, y, N, T)
```

## Dynamic Effects

```julia
effects = dynamic_spatial_effects(m)
# Short-run and long-run direct/indirect/total effects
```

## GS2SLS with Endogenous Regressors

For models with endogenous variables, use the GS2SLS estimator:

```julia
m = fit_gs2sls_panel(IndivFE(), W, X_exog, X_endog, Z, y, N, T)
```

## System GMM

For dynamic panels without spatial structure:

```julia
m = fit_dynamic_gmm(SystemGMM(), Y_mat, X_tensor; max_lag=3)
m = fit_dynamic_gmm(DifferenceGMM(), Y_mat, X_tensor; max_lag=3)
```
