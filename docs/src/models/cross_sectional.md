# Cross-Sectional Spatial Models

## SAR (Spatial Autoregressive)

```math
y = \rho W y + X \beta + \varepsilon
```

```julia
m = fit_spatial(SAR(), W, X, y)
```

## SDM (Spatial Durbin Model)

```math
y = \rho W y + X \beta + W X \theta + \varepsilon
```

```julia
m = fit_spatial(SDM(), W, X, y)
```

## Estimation Methods

Both models support exact (LU decomposition) and approximate (Chebyshev) log-determinant computation:

```julia
# Exact (default, good for N < 5000)
m = fit_spatial(SAR(), W, X, y)

# Chebyshev approximation (for large N)
m = fit_spatial(SAR(), W, X, y;
    method=ChebyshevApprox(order=7))
```

## Spatial Effects

For SDM, coefficients alone are not interpretable — use `spatial_effects`:

```julia
effects = spatial_effects(m)
# effects.direct   — average direct effect per variable
# effects.indirect — average indirect (spillover) effect
# effects.total    — direct + indirect
```
