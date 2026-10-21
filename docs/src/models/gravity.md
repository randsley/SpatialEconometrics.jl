# Gravity / Flow Models

## Specification

```math
E[F_{ij}] = \exp(\alpha_i + \gamma_j + X_{ij} \beta)
```

## PPML (Poisson Pseudo-Maximum Likelihood)

The recommended estimator for gravity models (Santos Silva & Tenreyro, 2006):

```julia
m = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id;
    varnames=["log_dist", "log_gdp"])
```

## Negative Binomial

For overdispersed count data:

```julia
m = fit_gravity(NegBinGravity(), flows, X, origin_id, dest_id;
    varnames=["log_dist", "log_gdp"])
# m.alpha — overdispersion parameter
```

## Fixed Effects

Both estimators absorb high-dimensional fixed effects via the Frisch-Waugh-Lovell approach:

- `OriginFE` — origin fixed effects only
- `DestinationFE` — destination fixed effects only
- `BilateralFE` — pair fixed effects (for panel gravity)

## Multilateral Resistance

```julia
mr = multilateral_resistance(m)
# Computes Anderson & van Wincoop (2003) multilateral resistance terms
```
