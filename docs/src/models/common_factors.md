# Common Factors and Spatio-Temporal Models

Every panel estimator elsewhere in this package assumes cross-sectional
correlation is **weak**: that `W` governs it, so conditioning on the spatial
lag is enough. The panel literature separates that from **strong**
dependence driven by unobserved common factors `λᵢ'fₜ`, which no weight
matrix absorbs.

This page covers the pair of things that case needs — tests that say which
kind of dependence is present, and estimators that survive the strong kind —
together with the space-time autoregression and the impulse responses that
go with it.

Runnable: `examples/27_common_factors.jl` for the diagnostics and the
loading-distribution table, `examples/28_spatio_temporal.jl` for STAR,
impulse responses, forecasting and the rotation indeterminacy.

## The one table to read first

Every design below has a genuine common factor. All that changes is the
*distribution of the loadings*. Measured bias in `ρ̂` at a true `ρ = 0.4`,
`N = 42`, `T = 30`, 200 replications:

| loadings | `NoFE` | `TwoWayFE` | `fit_sp_cce` | `fit_interactive_fe` |
|---|---|---|---|---|
| mean 2 (dominant) | **+0.484** | +0.002 / **0.029** | −0.001 / 0.047 | −0.032 / 0.041 |
| mean 1 | **+0.285** | +0.003 / 0.068 | −0.000 / 0.047 | −0.017 / **0.032** |
| mean 0, random | −0.004 | +0.003 / 0.068 | −0.008 / 0.059 | −0.002 / **0.025** |
| mean 0, balanced ±1 | **+0.329** | **+0.373** ⚠ | +0.005 / 0.100 | −0.023 / **0.035** |

(bias / RMSE where both are shown.)

Nothing about this is the expected "more robust is safer" story:

- **A time effect is a factor with equal loadings.** `TimeFE` subtracts the
  cross-sectional mean at each period, so a factor with `λᵢ ≡ λ` is removed
  to machine precision — measured at `5.55e-16`. What a time effect can
  reach is the *mean* of the loadings, and nothing else.
- **When the loadings have a strong common mean, `TwoWayFE` wins
  outright** (RMSE 0.029 against CCE's 0.047). It is the correctly
  specified parsimonious model, and CCE spends `N·q` projected columns for
  nothing. Reaching for the more general estimator costs you here.
- **When the loadings are balanced in sign, `TwoWayFE` is worse than doing
  nothing** — bias `+0.373` against `NoFE`'s `+0.329`. Removing a
  cross-sectional mean that is already zero destroys variation and leaves
  the factor untouched. This is the case CCE and interactive FE exist for.

So the loading structure decides the estimator, and the next section is
about finding out what it is.

## Diagnosing it: and why one statistic is not enough

[`pesaran_cd`](@ref) implements the CD statistic of Pesaran (2015). Its
implicit null is **weak dependence, not independence** — his eq. (12),
`H₀w : α < (2−ε)/4` with `T = O(Nᵋ)`. The implementation reproduces his own
headline result exactly (`N = 100`, `T = 50`, 3,000 reps, 5% level):

| true `α` | 0.00 | 0.10 | 0.25 | 0.40 | 0.50 | 0.75 |
|---|---|---|---|---|---|---|
| rejection | 0.048 | 0.054 | 0.055 | 0.144 | 0.549 | 1.000 |

Correctly sized across `[0, ¼]`, over-rejecting above — the transition sits
precisely at the boundary he states.

[`alpha_exponent`](@ref) estimates the exponent itself, following Bailey,
Kapetanios & Pesaran (2016), and enforces their two thresholds: `α` is
**not identified** at or below `1/2`, where the function throws rather than
returning a number for a parameter that does not exist, and valid inference
needs `α > 4/7`.

### The configuration where both of them fail

With loadings that cancel — the `±1` row of the table above — `ρ̂ᵢⱼ ∝ λᵢλⱼ`
does not accumulate, and:

| loadings | CD rejection | `α` identified? | mean \|ρ̂\| |
|---|---|---|---|
| independent | 0.047 | no (correctly) | 0.104 |
| homogeneous | 1.000 | yes, 300/300 | 0.496 |
| mean-zero random | **0.348** | sometimes | 0.293 |
| balanced ±1 | 0.911, **all negative** | **no, 0/300** | 0.911 |

Two distinct failures, and neither is a quiet non-rejection. With random
mean-zero loadings CD's power collapses to about a third and **does not
recover as `N` grows** (0.377 at `N = 10`, 0.343 at `N = 160`) — it is
structural. With balanced `±1` loadings CD rejects 91% of the time but the
statistic is **negative**, reporting a strong common factor as negative
average dependence. And `alpha_exponent` refuses outright, because the
cross-sectional average cancels and `Var(x̄ₜ)` collapses.

What survives is the **average absolute** correlation, reported as
`abs_rho_bar`. Against its independence benchmark `sqrt(2/(πT))` — measured
0.1036 against a predicted 0.1030 — it flags 1.000 of mean-zero-loading
draws where the signed statistic manages 0.348.

[`dependence_verdict`](@ref) reads all of this together, and is the
recommended entry point:

```julia
m = fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T)
v = dependence_verdict(residuals(m), N, T)
v.kind    # :weak, :semi_weak, :strong or :unidentified
println(v.recommendation)
```

### `CD` versus the LM statistic it replaces

`variant = :lm` gives the Breusch–Pagan statistic in Pesaran's scaled form.
It is offered to reproduce the comparison, not as a recommendation. On
*estimated* residuals — which is the case his critique is about, since
`E(Tρ̂²−1)` is mis-centred when the residuals are fitted — it fails badly,
and worsens monotonically with `N` exactly as he says:

| `N` (`T = 10`) | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| `CD` size | 0.045 | 0.048 | 0.040 | 0.064 | 0.048 |
| `CD_lm` size | 0.068 | 0.248 | 0.794 | **1.000** | **1.000** |

## Estimating it

### `fit_sp_cce` — common correlated effects

Projects each unit's series off the cross-sectional averages
`[ȳₜ, X̄ₜ]`, *unit by unit*, which is what permits heterogeneous loadings
and distinguishes CCE from a time effect. The residualised system is then
estimated by **2SLS** with instruments `[X̃, WX̃, W²X̃]`.

A likelihood is deliberately not used: projecting off the factor proxies
makes the error covariance non-scalar, so a QML written for spherical
errors would be misspecified, and the log-determinant it would buy is not
worth a wrong variance.

### `fit_interactive_fe` — Bai (2009) iterative principal components

Profiles `ρ` out on a one-dimensional search, and at each trial `ρ`
alternates between least squares with the factor space projected off, and
re-extracting the factors by eigendecomposition of the residual panel.

**Factors and loadings are identified only up to rotation.** For any
invertible `R`, `(FR, ΛR⁻ᵀ)` fits identically, so comparing `m.factors`
across fits is meaningless. What is identified is the space they span:

```julia
factor_projection(m)    # F(F'F)⁻¹F' -- compare THIS
```

Measured, two fits from different starts agree on the projection to
`5.2e-10` and on `ρ̂`/`β̂` to `~1e-9`, while the raw factors differ. A sign
flip — the simplest rotation — leaves the projection unchanged to exactly
`0.0e0`.

`n_factors = 0` removes the factor step entirely, and the estimator then
reproduces `fit_spatial_panel` with the log-likelihood agreeing to
**0.000e+00**.

## Space-time dynamics

[`fit_star`](@ref) fits Pfeifer & Deutsch's (1980) space-time
autoregression,

```
yₜ = Σₖ Σₗ φₖₗ Wᵏ yₜ₋ₗ + Xₜβ + εₜ
```

Every spatial lag is also *temporally* lagged, so all regressors are
predetermined and least squares is consistent — no Jacobian, and arbitrary
lag orders in both dimensions are affordable. The price is explicit: there
is no contemporaneous `Wyₜ`, so within-period spatial feedback is ruled
out. That specification is `fit_spatial_panel`, and neither nests the
other.

Stationarity is the spectral radius of the `L·N` **companion** matrix, not
a condition on any single `Aₗ`, and it is computed rather than assumed. For
a row-normalised `W` the leading eigenvalue is exactly 1, so for
`A₁ = aI + bW` the familiar `|a| + |b| < 1` is not merely sufficient but
exactly the boundary; drop the normalisation and the same coefficients
diverge at `|a| + |b| = 0.8`, because the spectrum then reaches 3.65 on the
same lattice. With `L > 1` there is no coefficient sum to check at all.

[`space_time_irf`](@ref) traces a shock through both dimensions,
`Ψ₀ = I`, `Ψₕ = Σₗ Aₗ Ψₕ₋ₗ` with `Aₗ = Σₖ φₖₗ Wᵏ`, splitting each horizon
into an own effect and a spillover. Summed over horizons it converges to
the long-run multiplier `(I − ΣₗAₗ)⁻¹`, which is the identity it is tested
against — agreement to `1.1e-15`.

!!! note "`cumulative` accumulates the summaries, not the matrices"
    `psi[h+1]` is always `∂yₜ₊ₕ/∂εₜ'`, the response *at* horizon `h`,
    whatever `cumulative` is set to. The keyword controls `own`,
    `spillover` and `total`. To compare against `(I − ΣₗAₗ)⁻¹` directly,
    sum `psi` yourself — or read `total[end]`, which is that sum's mean
    row sum.

The path is what the IRF adds over the multiplier, and the timing is not
incidental: on a 7×6 lattice with `φ̂ = (0.393, 0.396)` the own effect peaks
at `h = 0` while the spillover peaks at `h = 2`, and the spillover/own ratio
rises monotonically from there. A single direct/indirect/total decomposition
is the `h → ∞` column of that path, so for any question with a policy
horizon shorter than convergence it is the wrong summary.

[`spatial_forecast`](@ref) iterates the same recursion forward, dynamically:
forecasts are substituted for realisations once the data runs out. The model
does not retain `y`, so the last `L` periods must be passed as `y_history` —
omitting them throws rather than defaulting to zero.

## References

- Bai, J. (2009). Panel data models with interactive fixed effects.
  *Econometrica*, 77(4), 1229–1279.
- Bailey, N., Kapetanios, G., & Pesaran, M. H. (2016). Exponent of
  cross-sectional dependence: Estimation and inference. *Journal of
  Applied Econometrics*, 31(6), 929–960.
- Ertur, C., & Musolesi, A. (2017). Weak and strong cross-sectional
  dependence. *Journal of Applied Econometrics*, 32(3), 477–503.
- Pesaran, M. H. (2006). Estimation and inference in large heterogeneous
  panels with a multifactor error structure. *Econometrica*, 74(4),
  967–1012.
- Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large
  panels. *Econometric Reviews*, 34(6–10), 1089–1117.
- Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative
  procedure for space-time modeling. *Technometrics*, 22(1), 35–47.
- Shi, W., & Lee, L. F. (2017). Spatial dynamic panel data models with
  interactive fixed effects. *Journal of Econometrics*, 197(2), 323–347.
- Yang, C. F. (2021). Common factors and spatial dependence: An
  application to US house prices. *Econometric Reviews*, 40(1), 14–50.
