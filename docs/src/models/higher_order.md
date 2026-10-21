# Higher-Order Spatial Models

## SLX — Spatial Lag of X

The SLX model includes spatially lagged covariates without a spatial lag of `y`:

```math
y = X\beta + WX\theta + \varepsilon
```

Estimated by OLS. Recommended as a starting point before SAR/SDM (Halleck Vega & Elhorst, 2015).

```julia
m = fit_slx(W, X, y; varnames=["gdp", "pop"])
```

## SAC — Spatial Autoregressive Combined

The SAC (also called SARAR) model combines spatial lag and spatial error:

```math
y = \rho W_1 y + X\beta + u, \quad u = \lambda W_2 u + \varepsilon
```

Allows different weight matrices for the lag and error components (Kelejian & Prucha, 1998).

```julia
m = fit_sac(W_lag, W_error, X, y; varnames=["gdp", "pop"])
```

When `W_lag == W_error`, this is the classic SARAR model.

## SDEM — Spatial Durbin Error Model

The SDEM combines the SDM covariate structure with the SEM error structure:

```math
y = X\beta + WX\theta + u, \quad u = \lambda W u + \varepsilon
```

Estimated via concentrated maximum likelihood (LeSage & Pace, 2009).

```julia
m = fit_sdem(W, X, y; varnames=["gdp", "pop"])
```

## API

See the full [API Reference](@ref) for `fit_slx`, `fit_sac`, and `fit_sdem`.
