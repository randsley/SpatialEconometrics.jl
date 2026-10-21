# Spatial Discrete Choice

Binary and ordered outcomes with spatial dependence — land use change, technology
adoption, voting, firm entry/exit. Standard probit ignores the spatial correlation,
giving inefficient estimates and invalid inference.

All three estimators target the spatial autoregressive latent model

```math
y^* = \rho W y^* + X\beta + \varepsilon, \qquad \varepsilon \sim N(0, I)
```

with `σ² = 1` fixed for identification.

## Linearized GMM

Fast, no MCMC. A standard probit supplies the `ρ = 0` starting value; the moment
conditions are linearized around it and estimated by 2SLS with instruments
`[X, WX, W²X]` (Klier & McMillen, 2008).

```julia
m = fit_spatial_probit(W, X, y; varnames=["gdp", "pop"])
```

`y` must be coded 0/1. Standard errors use a heteroskedasticity-robust 2SLS
sandwich.

!!! warning "Efficiency cost"
    Linearizing around `ρ = 0` trades precision for speed. Over 8 replications at
    `ρ = 0.4` this estimator had a standard deviation of roughly 0.25 at `N = 196`
    and 0.18 at `N = 400` — about 2.5× that of the Gibbs sampler on the same data.
    It is approximately unbiased, so a single small-sample `ρ̂` is a coarse
    reading. Prefer the Gibbs sampler when precision on `ρ` matters.

## Bayesian (Albert–Chib)

Gibbs sampling with data augmentation, following LeSage & Pace (2009, Ch. 10).
Each sweep draws the latent `y*` one site at a time from its truncated
conditional, draws `β` from its conjugate normal conditional, then updates `ρ` by
random-walk Metropolis.

```julia
m = fit_bayesian_spatial_probit(W, X, y; n_samples=2000, burnin=500)
```

Reported `coef` and `vcov` are the posterior mean and covariance. Raw draws are
available as `m.beta_draws` and `m.rho_draws`.

!!! note "Check the acceptance rate"
    `m.accept_rate` should land in roughly 0.2–0.5. The default
    `rho_proposal_sd = 0.2` targets that band; 0.05 overshoots to about 0.77 and
    mixes slowly. Tune it if your acceptance rate falls outside the band.

!!! warning "Weak identification at small N"
    The observed signs pin the latent vector only up to scale, so `‖β‖` and `ρ`
    trade off along a weakly identified ridge. The posterior is centred correctly
    — the posterior mean averaged 0.374 across 8 replications at `ρ = 0.4`,
    `N = 196` — but its spread is wide (SD ≈ 0.10). A single chain at small `N`
    can sit well away from the truth. Lengthen the chain and inspect
    `m.rho_draws` before reading much into one estimate.

## Ordered probit

For `y` in categories `1:J`, with `τ_{j-1} < y* ≤ τ_j`. Same Gibbs machinery,
plus uniform conditional draws for the cutpoints (Wang, Kockelman & Wang, 2013).

```julia
m = fit_spatial_ordered_probit(W, X, y_ordered; n_categories=3)
```

For identification there is no intercept and `τ_1 = 0`; the free cutpoints
`τ_2 … τ_{J-1}` are estimated. A binary `y` is better served by
`fit_bayesian_spatial_probit`, which carries an intercept.

## Marginal effects

Coefficients are not marginal effects in a spatial probit — a change in `xᵢ`
propagates through `(I - ρW)⁻¹`. For covariate `k` the effect matrix is
`diag(φ(index)) (I-ρW)⁻¹ βₖ`:

```julia
eff = marginal_effects(m)
```

The direct effect averages the diagonal, the total averages the row sums, and the
indirect (spillover) effect is the difference (LeSage & Pace, 2009).

!!! note "Cost"
    Fitted probabilities and marginal effects both need a dense inverse of
    `I - ρW` — O(N³). These estimators are intended for moderate `N`.

## API

See the full [API Reference](@ref) for `fit_spatial_probit`,
`fit_bayesian_spatial_probit`, `fit_spatial_ordered_probit`, and
`marginal_effects`.
