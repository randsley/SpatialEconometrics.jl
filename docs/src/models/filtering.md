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

## Spatio-temporal principal components

ESF and the eigenvector basis above describe one map. `stpca` describes a
*sequence* of maps: `n` units, each carrying `p` features observed at `T`
moments, reduced to components that are simultaneously high-variance and
spatially structured.

```julia
m = stpca(X, W; measure = :moran, n_basis = 5,
          times = 0.5:1:16.5, interval = (0.0, 17.0))
m.scores[:, 1]                  # where each unit sits on the first component
component_groups(m)             # which components are global, which local
feature_contributions(m, 1)     # which of the p features the component reads
```

The construction has two stages. First each unit-feature time series is
replaced by a continuous function, expanded in `n_basis` orthonormal Fourier
functions by least squares, which turns `n × p × T` discrete observations
into an `n × B` coefficient matrix `A` with `B = p · n_basis`. Time enters
only through that expansion. Second, components are found in the coefficient
space, where space enters through `W`.

### One eigenproblem, two factors

A functional principal component `U = γ'α` has estimated variance `γ'Sγ`
with `S = A'A/n`, and produces scores `y = Aγ` across the units. Both the
variance and the spatial autocorrelation of those scores are ratios of
quadratic forms in `γ`, and in their product the denominators cancel:

```math
\hat{\mathrm{Var}}(U)\, I(A\gamma)
  = \frac{1}{n}\gamma'A'A\gamma \cdot
    \frac{n}{\sum_{ij}w_{ij}}\frac{\gamma'A'WA\gamma}{\gamma'A'A\gamma}
  = \gamma' C \gamma,
\qquad C = \Big(\sum_{ij}w_{ij}\Big)^{-1}\!A'WA
```

so maximising the product subject to `‖γ‖ = 1` is an eigenproblem, and each
eigenvalue **is** the value of the objective for its own component — not a
proxy for it. That identity is what the tests pin: `values[k]` equals
`variance[k] * autocorr[k]` to 1e-9, with both factors recomputed from
[`moran_coefficient`](@ref) and `SpatialDiagnostics.geary_c` rather than
from anything `stpca` computed.

Substituting Geary's `C` for Moran's `I` gives the same algebra with the
contiguity matrix `B*` in place of `W`:

```math
E = \frac{n-1}{2na}A'B^*A, \qquad
B^*_{ij} = \tfrac{1}{2}(B + B')_{ij}, \quad
y'By = \sum_{ij}w_{ij}(y_i - y_j)^2
```

### Global and local components

`C` is not a covariance matrix, and nothing requires it to be non-negative
definite. Under `measure = :moran` its eigenvalues take both signs, and the
sign is the result: a positive eigenvalue is a **global** component, where
linked units carry similar scores, and a negative one is a **local**
component, where a unit's score is surrounded by unlike ones.
[`component_groups`](@ref) returns the two groups in the reference's order —
positives by decreasing eigenvalue, negatives by decreasing absolute
eigenvalue.

Under `measure = :geary` the second factor is a sum of squares, so `E` is
non-negative definite by construction and the negative group is empty. That
is a property of the method, not of the data, and the two measures therefore
answer different questions.

### The two measures do not order the spectrum the same way

Over 800 draws with a known field planted in one feature:

| Planted field | `:moran` first | `:moran` last | `:geary` first | `:geary` last |
|---|---|---|---|---|
| smooth (`I ≈ +0.92`), `N = 36` | 86.5% | 0.0% | 0.0% | 7.0% |
| smooth (`I ≈ +0.96`), `N = 64` | 73.0% | 0.0% | 0.0% | 14.0% |
| high-contrast (`I ≈ -1.0`), `N = 36` | 0.0% | 86.0% | 33.5% | 0.0% |
| high-contrast (`I ≈ -1.0`), `N = 64` | 0.0% | 71.5% | 7.0% | 0.0% |

The entry is the share of draws in which the component most correlated with
the planted field was the first or the last of the spectrum.

Read the `:moran` rows first: the split is sharp and never inverted. A
smooth field is the leading component or nothing, a high-contrast field is
the trailing one or nothing, and neither ever landed at the wrong end in 800
draws.

The `:geary` rows are the warning. Maximising `Var · C` does demote smooth
structure — a smooth field was never the leading `:geary` component — but it
does not simply promote the roughest, which led in only 7–34% of draws. The
objective is a *product*, and a component with moderate `C` and large
variance outranks one with high `C` and little. Read the `:geary` ordering
as "variance weighted by dissimilarity", not as "most dissimilar first".

Also worth knowing before quoting a correlation: the planted field is spread
across several components rather than captured whole. The median correlation
between it and its best-matching component was 0.56 at `N = 36` and 0.43 at
`N = 64`. The component ranking is the reliable output; the loading is not a
recovery of the field.

### Reading a component

A component is a direction in coefficient space, which [`component_weight`](@ref)
reads back as a function of time: `u_k(t)` is a `length(t) × p` matrix whose
column `j` is the weight the component puts on feature `j` at each moment.
[`feature_contributions`](@ref) summarises that over the whole interval as
the share of the area under `|u_kj(t)|`, so the largest share names the
series the component is mostly tracking. Both are needed: the scores say
which units a component separates, and these say what it separated them on.

!!! note "The expansion interval is padded by default"
    A Fourier basis is periodic. Expanding on exactly `[t₁, t_T]` forces the
    fitted curve to take the same value at the first and last observation,
    which the data have no reason to do, so `interval` defaults to `times`
    padded by half the mean spacing at each end. For annual data at
    `t = 0.5, 1.5, …, 16.5` that default is `[0, 17]`, which is the interval
    the reference uses.

!!! warning "The components depend on `W`, and there is no rule for choosing it"
    Changing `W` changes the components and so changes any clustering built
    on them. The reference is explicit that no guideline exists for which
    `W` to use and compares nine of them. Treat the choice as a reported
    assumption, and vary it — [`select_knn_w`](@ref) chooses `k` for a knn
    `W` by information criterion if the candidate set is that family.
