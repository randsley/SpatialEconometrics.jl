# Static Spatial Panel Models

## Specification

```math
y_{it} = \rho W y_{it} + X_{it} \beta + \mu_i + \xi_t + \varepsilon_{it}
```

## Fixed Effects

| Type | Meaning | Removes |
|---|---|---|
| `NoFE()` | No fixed effects | — |
| `IndivFE()` | Individual FE | Unit-specific means |
| `TimeFE()` | Time FE | Period-specific means |
| `TwoWayFE()` | Two-way FE | Both |

## Estimation Approaches

```julia
# Lee-Yu bias-corrected transformation (default)
m = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T;
    approach=LeeYuTransformation())

# Direct QML with analytical bias correction
m = fit_spatial_panel(SAR(), IndivFE(), W, X, y, N, T;
    approach=DirectQML(bias_correct=true))
```

## Formula DSL

```julia
m = fit_spatial_panel(
    @formula(y ~ x1 + x2 + slag(x1 + x2) + fe(region) + fe(year)),
    table, W; panel=:region, time=:year
)
```
