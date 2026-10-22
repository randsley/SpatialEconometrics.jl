# Spatial Error Model (SEM)

## Specification

```math
y = X \beta + u, \quad u = \lambda W u + \varepsilon
```

## Cross-Sectional SEM

```julia
m = fit_spatial_error(W, X, y)
# m.lambda — spatial error parameter
```

## Panel SEM

```julia
m = fit_spatial_error(IndivFE(), W, X, y, N, T)
m = fit_spatial_error(TwoWayFE(), W, X, y, N, T)
```

## When to Use SEM

SEM is appropriate when spatial dependence operates through unobserved factors (omitted variables, shocks) rather than through direct interactions between observations. The LM tests help distinguish:

```julia
lm = lm_tests(y, hcat(ones(N), X), W)
# If lm.error is significant but lm.lag is not → SEM
# If lm.lag is significant but lm.error is not → SAR
# If both significant → use robust versions or SDM
```
