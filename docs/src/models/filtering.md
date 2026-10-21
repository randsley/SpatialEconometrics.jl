# Spatial Filtering

Eigenvector spatial filtering (ESF) removes spatial autocorrelation by adding a
small set of synthetic covariates — eigenvectors of the doubly-centred weight
matrix — to an otherwise standard regression. The spatial structure is absorbed
into those regressors, so the residuals satisfy the usual independence
assumptions and the model is fitted with ordinary OLS or GLM machinery. No
spatial parameter is estimated and no Jacobian is needed, which makes the
approach attractive for generalised linear models where full spatial ML is
awkward (Griffith, 2003).

## The eigenvector basis

Let `M = I - 11'/N` be the centring projector and `Wᵤ = (W + W')/2`. The basis is
the eigendecomposition of `Ω = M Wᵤ M`, and eigenvector `eᵢ` has Moran
coefficient

```math
MC_i = \frac{N}{S_0}\lambda_i
```

so the leading eigenvectors are exactly the map patterns with the strongest
positive autocorrelation.

```julia
se = spatial_eigenvectors(W; min_moran_ratio=0.25)
```

`min_moran_ratio` keeps only candidates with `MCᵢ/max(MC) ≥ ratio`, the usual
screening rule. The result has `N - 1` eigenvectors before screening: the
constant direction is excluded by construction.

!!! note "Why the constant is projected out first"
    `Ω` has a multi-dimensional null space, and `eigen` is free to return an
    arbitrary *rotation* within it. Decomposing `Ω` directly therefore smears the
    constant direction across several near-zero eigenvectors, each of which is
    then partly collinear with the intercept — a magnitude threshold cannot
    detect this. `spatial_eigenvectors` instead builds an orthonormal basis `Q`
    of the complement of `1` and decomposes `Q'WᵤQ`, which is valid because
    `MQ = Q`. The returned eigenvectors satisfy `E'E = I` and `Σᵢ eᵢ = 0`
    exactly.

## Stepwise ESF

```julia
m = fit_esf(X, y, W; family=:gaussian, max_eigenvectors=20, varnames=["x1", "x2"])
```

Eigenvectors are added greedily, either to minimise the residual Moran
coefficient (`criterion=:moran`, the default) or to minimise AIC
(`criterion=:aic`). Selection stops when `max_eigenvectors` is reached or the
residual Moran coefficient falls inside `moran_tol` of zero.

Four families are supported. `:gaussian` is fitted by OLS; `:poisson`, `:logit`
and `:negbin` by IRLS, with the negative binomial dispersion `α` updated between
iterations.

```julia
m = fit_esf(X, counts, W; family=:poisson)
m = fit_esf(X, binary, W; family=:logit)
m = fit_esf(X, counts, W; family=:negbin)   # m.alpha holds the dispersion
```

`selected_eigenvectors(m)` returns the chosen indices with their Moran
coefficients and the residual Moran coefficient before and after filtering.

Selecting *zero* eigenvectors is a legitimate outcome, not a failure: it means
the residuals were already within `moran_tol` of no autocorrelation and there was
nothing to filter. This happens readily with binary outcomes, where binarisation
discards much of the spatial signal present in the latent variable.

## Random-effects ESF

Stepwise selection introduces model-selection variance. RE-ESF avoids it by
retaining the whole screened pool and shrinking each coefficient instead
(Murakami & Griffith, 2015):

```math
y = X\beta + E\gamma + \varepsilon, \quad
\gamma \sim N(0, \tau^2\Lambda), \quad \varepsilon \sim N(0, \sigma^2 I)
```

A single variance ratio `θ = τ²/σ²` governs the shrinkage and is estimated by
profile REML.

```julia
m = fit_re_esf(X, y, W; varnames=["x1", "x2"])
```

Because `E` has orthonormal columns the covariance inverts in closed form,

```math
V_0^{-1} = I - E D E', \qquad D = \mathrm{diag}\!\left(\frac{\theta\lambda_l}{1+\theta\lambda_l}\right),
\qquad \log|V_0| = \sum_l \log(1 + \theta\lambda_l)
```

so each REML evaluation costs `O(NL)` rather than `O(N³)`; only the
decomposition itself is cubic. The random effects are recovered by their BLUP,
`γ̂ₗ = θλₗ/(1+θλₗ)·(E'r)ₗ`.

### Reading the residual Moran coefficient

For RE-ESF the reported `moran_final` is the Moran coefficient of the
*conditional* residual `y - Xβ̂ - Eγ̂`, and **it is expected to be substantially
negative. Do not compare it against zero.**

Subtracting `Eγ̂` removes the positively autocorrelated directions, so what
remains lives mostly in the orthogonal complement of `span(E)` — and a vector
drawn at random from that complement is itself strongly negatively
autocorrelated. Its expected Moran coefficient is available in closed form,
since the `N-1` centred eigenvalues sum to `tr(M Wᵤ M) = -S₀/N`:

```math
\frac{N}{S_0}\cdot\frac{-S_0/N - \sum_l \lambda_l}{N - 1 - L}
```

This is reported as `moran_reference`. On data generated *from the RE-ESF model*
with `N = 121` and `L = 40`, the reference is `-0.304` while the fitted residual
lands near `-0.240` — slightly *less* negative than chance, with `β` and `σ²`
both recovered accurately. A `moran_final` at or above `moran_reference` is
therefore the healthy case.

!!! warning "Diagnosing noise absorption"
    Because the whole pool is retained, a large pool lets the random effects
    absorb noise as well as signal. Two diagnostics matter:

    * `moran_final` falling *far below* `moran_reference`.
    * `edf`, the effective degrees of freedom `P + Σ θλ/(1+θλ)`, approaching `N`.

    On SAR-generated data with `N = 121`, retaining all 55 positive eigenvectors
    gave `edf = 47` and residual Moran `-0.348`; the default `min_moran_ratio`
    of 0.25 reduced this to 40 eigenvectors, `edf = 35` and `-0.146`, and a
    stricter 0.5 screen gave 23 eigenvectors, `edf = 23` and `-0.035`. Tighten
    `min_moran_ratio` or cap `n_candidates` when `edf` is an uncomfortable
    fraction of `N`.

## Choosing between them

Use `fit_esf` when a parsimonious, interpretable set of spatial regressors is
wanted, or for a non-Gaussian family. Use `fit_re_esf` for Gaussian data when the
instability of stepwise selection is the greater concern — it has no selection
step at all, and its single shrinkage parameter is estimated rather than chosen.

## References

* Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
* Murakami, D., & Griffith, D. A. (2015). Random effects specifications in
  eigenvector spatial filtering. *Journal of Geographical Systems*, 17(4), 311–331.

Function signatures are listed in the [API Reference](@ref).
