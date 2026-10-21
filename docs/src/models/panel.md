# Static Spatial Panel Models

## Specification

```math
y_{it} = \rho W y_{it} + X_{it} \beta + \mu_i + \xi_t + \varepsilon_{it}
```

## Fixed Effects

| Type | Alias | Removes |
|---|---|---|
| `StaticPanel.NoFE()` | No fixed effects | — |
| `StaticPanel.IndivFE()` | Individual FE | Unit-specific means |
| `StaticPanel.TimeFE()` | Time FE | Period-specific means |
| `StaticPanel.TwoWayFE()` | Two-way FE | Both |

## Estimation Approaches

```julia
# Lee-Yu bias-corrected transformation (default)
m = fit_spatial_panel(StaticPanel.SAR(), StaticPanel.TwoWayFE(), W, X, y, N, T;
    approach=LeeYuTransformation())

# Direct QML with analytical bias correction
m = fit_spatial_panel(StaticPanel.SAR(), StaticPanel.IndivFE(), W, X, y, N, T;
    approach=DirectQML(bias_correct=true))
```

## Formula DSL

```julia
m = fit_spatial_panel(
    @formula(y ~ x1 + x2 + slag(x1 + x2) + fe(region) + fe(year)),
    table, W; panel=:region, time=:year
)
```
