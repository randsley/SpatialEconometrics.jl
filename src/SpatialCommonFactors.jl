"""
    SpatialCommonFactors

Weak versus strong cross-sectional dependence in spatial panels.

Every panel estimator elsewhere in this package assumes cross-sectional
correlation is *weak*: that it is governed by `W`, so that conditioning on
the spatial lag is enough. The panel literature separates that case from
*strong* dependence driven by unobserved common factors `λᵢ'fₜ`, which no
weight matrix can absorb and which biases `ρ̂` when it is ignored.

This module supplies the pair of things that case needs — a test that says
which kind of dependence is present, and an estimator that survives the
strong kind:

- [`pesaran_cd`](@ref) — the CD statistic of Pesaran (2015), and the
  Breusch–Pagan LM variant it replaces.
- [`alpha_exponent`](@ref) — the exponent of cross-sectional dependence of
  Bailey, Kapetanios & Pesaran (2016), which is what distinguishes weak
  from strong.
- [`dependence_verdict`](@ref) — the two read together, which is the only
  way to read either.
- [`fit_sp_cce`](@ref) — common correlated effects for a spatial panel,
  augmenting the regression with cross-sectional averages.

**Why the pair and not just the test.** The CD statistic's null is *not*
independence. Pesaran (2015) shows its implicit null is weak dependence,
`α < (2−ε)/4` where `T = O(Nᵋ)` — his eq. (12) — so a non-rejection does
not say the units are independent, and a rejection does not say a factor
structure is present. The `α` exponent is the quantity that separates them.
Reporting one without the other invites the wrong conclusion in both
directions, which is why [`dependence_verdict`](@ref) exists.

# References

- Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large
  panels. *Econometric Reviews*, 34(6–10), 1089–1117.
- Bailey, N., Kapetanios, G., & Pesaran, M. H. (2016). Exponent of
  cross-sectional dependence: Estimation and inference. *Journal of
  Applied Econometrics*, 31(6), 929–960.
- Pesaran, M. H. (2006). Estimation and inference in large heterogeneous
  panels with a multifactor error structure. *Econometrica*, 74(4),
  967–1012.
- Yang, C. F. (2021). Common factors and spatial dependence: An
  application to US house prices. *Econometric Reviews*, 40(1), 14–50.
- Ertur, C., & Musolesi, A. (2017). Weak and strong cross-sectional
  dependence: A panel data analysis of international technology
  diffusion. *Journal of Applied Econometrics*, 32(3), 477–503.
"""
module SpatialCommonFactors

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: coef, vcov, nobs, dof_residual, coefnames, stderror, residuals

import ..SpatialUtils: _normcdf, _chi2_sf, _print_coef_table,
    FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
    within_transform, within_transform!, panel_spatial_lag!, panel_spatial_lag,
    check_design_rank, brent_maximize

export
    CDTestResult, pesaran_cd,
    AlphaExponentResult, alpha_exponent,
    DependenceVerdict, dependence_verdict,
    SpCCEModel, fit_sp_cce

# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

# The panel layout throughout this package: a stacked vector of length
# N*T_per, unit-within-time, so that `reshape(v, N, T_per)` gives units
# down the rows and periods across the columns. Matching
# `SpatialUtils.within_transform!` is not optional -- every estimator here
# hands vectors to it.
@inline function _panel_matrix(v::AbstractVector{T}, N::Int, T_per::Int) where {T}
    length(v) == N * T_per || throw(DimensionMismatch(
        "expected a panel vector of length N*T = $(N * T_per), got $(length(v))"))
    return reshape(v, N, T_per)
end

function _check_panel_dims(N::Int, T_per::Int, what::AbstractString)
    N >= 3 || throw(ArgumentError("$what: need at least 3 cross-sectional " *
        "units, got N = $N. The pairwise scaling carries a factor " *
        "1/(N(N-1)), which is degenerate below that."))
    T_per >= 2 || throw(ArgumentError("$what: need at least 2 periods, " *
        "got T = $T_per. A pairwise correlation over one period is not " *
        "defined."))
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════
# Pesaran's CD test
# ═══════════════════════════════════════════════════════════════════════

"""
    CDTestResult

Cross-sectional dependence test for panel residuals, returned by
[`pesaran_cd`](@ref).

# Fields

- `stat::Float64` — the statistic, `CD` or `CD_lm` per `variant`.
- `p_value::Float64` — two-sided, from the standard normal.
- `variant::Symbol` — `:cd` or `:lm`.
- `rho_bar::Float64` — `ρ̄_N`, the average pairwise residual correlation
  (Pesaran 2015, eq. 9). The statistic is a scaling of this, and it is the
  more interpretable of the two: it is bounded in `[-1, 1]` and says *how
  much* dependence there is, where the statistic says only whether it is
  distinguishable from zero.
- `abs_rho_bar::Float64` — the average *absolute* pairwise correlation,
  and the field that rescues the test when the signed average cancels.
  See "When the signed average cancels" below; under independence it
  concentrates on `sqrt(2/(πT))`, which is the number to compare it with.
- `abs_rho_bar_null::Float64` — that benchmark, `sqrt(2/(πT))`, computed
  for this `T`. `abs_rho_bar` materially above it means dependence is
  present whatever the signed statistic says.
- `N::Int`, `T::Int` — panel dimensions used.
- `implicit_null_alpha::Float64` — the `α` threshold below which a
  non-rejection is consistent with the test's actual null, computed as
  `(2 − ε)/4` with `ε = ln(T)/ln(N)`, from eq. (12). Compare it with
  [`alpha_exponent`](@ref).

# Interpretation

A rejection says dependence is stronger than the test's implicit null of
weak dependence. It does **not** identify a factor structure, and a
non-rejection does **not** establish independence. Read it with
[`alpha_exponent`](@ref) via [`dependence_verdict`](@ref).
"""
struct CDTestResult
    stat::Float64
    p_value::Float64
    variant::Symbol
    rho_bar::Float64
    abs_rho_bar::Float64
    abs_rho_bar_null::Float64
    N::Int
    T::Int
    implicit_null_alpha::Float64
end

function Base.show(io::IO, r::CDTestResult)
    label = r.variant === :cd ? "Pesaran CD Test" : "Breusch–Pagan LM Test (CD_lm)"
    println(io, label)
    println(io, "─"^length(label))
    @printf(io, "  Statistic:          %+.4f\n", r.stat)
    @printf(io, "  p-value:            %.6f %s\n", r.p_value,
        r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" :
        r.p_value < 0.05 ? "*" : r.p_value < 0.1 ? "." : "")
    @printf(io, "  Mean pairwise corr: %+.4f\n", r.rho_bar)
    @printf(io, "  Mean |corr|:        %.4f   (independence benchmark %.4f)\n",
        r.abs_rho_bar, r.abs_rho_bar_null)
    @printf(io, "  Panel:              N = %d, T = %d\n", r.N, r.T)
    if r.variant === :cd
        @printf(io, "  Implicit null:      α < %.3f  (eq. 12)\n",
            r.implicit_null_alpha)
    end
    if r.p_value < 0.05
        println(io, "  Reject H₀ → dependence exceeds the test's null of weak dependence.")
        println(io, "  This does not by itself establish a common factor; see `alpha_exponent`.")
    else
        println(io, "  Fail to reject H₀ → consistent with weak cross-sectional dependence.")
        println(io, "  This is NOT evidence of independence: the null is weak dependence.")
    end
    # The signed statistic can cancel while the absolute average does not.
    # Measured separation: 1.000 vs 0.348 detection on mean-zero loadings.
    if r.abs_rho_bar > 2 * r.abs_rho_bar_null && abs(r.rho_bar) < 0.05
        println(io, "  ⚠ Mean |corr| is far above its independence benchmark while the")
        println(io, "    signed average is near zero — dependence with loadings that cancel.")
        println(io, "    Trust the absolute average here, not the statistic.")
    end
end

"""
    pesaran_cd(resid, N, T_per; variant = :cd)

Test the cross-sectional dependence of panel residuals.

`resid` is a stacked vector of length `N*T_per` in the package's usual
unit-within-time layout, or an `N × T_per` matrix.

# The statistics, as published

With `ρ̂ᵢⱼ` the pairwise residual correlation (Pesaran 2015, eq. 2),

```
ρ̂ᵢⱼ = Σₜ eᵢₜ eⱼₜ / [(Σₜ eᵢₜ²)^½ (Σₜ eⱼₜ²)^½]
```

`variant = :cd` returns the CD statistic of his eqs. (9) and (10),

```
ρ̄_N = [2/(N(N−1))] Σᵢ Σⱼ>ᵢ ρ̂ᵢⱼ
CD   = [T N(N−1)/2]^½ ρ̄_N                    →  N(0,1)
```

and `variant = :lm` the Breusch–Pagan Lagrange multiplier statistic in the
scaled form of his eq. (5),

```
CD_lm = [1/(N(N−1))]^½ Σᵢ Σⱼ>ᵢ (T ρ̂ᵢⱼ² − 1)   →  N(0,1)
```

# Which variant to use

`:cd` is the default and is almost always the right choice. `CD_lm` sums
*squared* correlations, so `E(T ρ̂ᵢⱼ² − 1)` is not centred at zero in finite
`T`; with `N` large the mis-centring accumulates across `N(N−1)/2` pairs and
the test over-rejects. Pesaran (2015, §2.1) gives this as the motivation for
CD, and the size distortion is measured in `docs/AUDIT.md`. `:lm` is
provided to reproduce the comparison, not as a recommendation.

# What a result does and does not mean

The CD test's implicit null is **weak dependence, not independence** — his
eq. (12), `H₀w : α < (2−ε)/4` where `T = O(Nᵋ)`. So:

- a non-rejection is consistent with weak dependence, and is not evidence
  that the units are independent;
- a rejection says the dependence is stronger than that null, not that it
  comes from a common factor.

Use [`alpha_exponent`](@ref) to tell those apart, or
[`dependence_verdict`](@ref) to get both with the applicable thresholds.

# When the signed average cancels

`ρ̄_N` averages *signed* correlations, and `ρ̂ᵢⱼ ∝ λᵢλⱼ` under a factor
structure, so loadings that are not of one sign do not accumulate. This is
measured in `docs/AUDIT.md` rather than asserted, and it is worse and more
specific than "the test loses power" (`N = 40`, `T = 60`, 5% level):

| loadings | CD rejection | mean `ρ̄_N` | mean `|ρ̂|` |
|---|---|---|---|
| independent | 0.047 | +0.0000 | 0.1036 |
| homogeneous `λᵢ ≡ 1` | 1.000 | +0.4961 | 0.4961 |
| mean-zero `λᵢ ~ N(0,1)` | **0.348** | −0.0007 | 0.2929 |
| balanced `λᵢ = ±1` | 0.911, **all negative** | −0.0128 | 0.911 |

Two distinct failures, neither of which is a non-rejection:

1. With **random mean-zero loadings** power collapses to about a third, and
   it **does not recover as `N` grows** — measured 0.377 at `N = 10` and
   0.343 at `N = 160`. It is structural, not a small-sample problem.
2. With **balanced `±1` loadings** the test rejects 91% of the time but the
   statistic is **negative**, so a strong common factor is reported as
   negative average dependence. The sign actively misleads.

`abs_rho_bar` is what survives both. Against the independence threshold it
flags **1.000** of mean-zero-loading draws where the signed statistic
rejects 0.348. Under independence it concentrates tightly on
`sqrt(2/(πT))` — measured 0.1036 against a predicted 0.1030 — so
`abs_rho_bar_null` gives it a scale, and `show` warns when the two
diverge. If `abs_rho_bar` sits well above its benchmark while `ρ̄_N` is near
zero, believe the absolute average.

# Example

```julia
m = fit_spatial_panel(SAR(), W, X, y, N, T; fe_spec = TwoWayFE())
cd = pesaran_cd(residuals(m), N, T)
```

# References

Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large
panels. *Econometric Reviews*, 34(6–10), 1089–1117. Equations 2, 5, 9, 10
and 12.
"""
function pesaran_cd(resid::AbstractVector{T}, N::Int, T_per::Int;
                    variant::Symbol = :cd) where {T<:AbstractFloat}
    variant in (:cd, :lm) || throw(ArgumentError(
        "pesaran_cd: variant must be :cd or :lm, got :$variant"))
    _check_panel_dims(N, T_per, "pesaran_cd")
    all(isfinite, resid) || throw(ArgumentError(
        "pesaran_cd: residuals contain non-finite values"))

    E = _panel_matrix(resid, N, T_per)

    # Normalise each unit's residual series by its own root sum of squares,
    # which turns eq. (2) into a plain inner product. Doing it once up front
    # is O(NT) rather than O(N^2 T): the pairwise loop below then costs one
    # dot product per pair instead of three sums.
    Z = Matrix{T}(undef, N, T_per)
    @inbounds for i in 1:N
        ss = zero(T)
        for t in 1:T_per
            ss += E[i, t]^2
        end
        if ss <= zero(T)
            throw(ArgumentError("pesaran_cd: unit $i has zero residual " *
                "variation, so its pairwise correlations are undefined. " *
                "This usually means the unit was perfectly fitted — check " *
                "for a unit-specific dummy or a singleton group."))
        end
        inv_norm = one(T) / sqrt(ss)
        for t in 1:T_per
            Z[i, t] = E[i, t] * inv_norm
        end
    end

    sum_rho = zero(T)
    sum_abs = zero(T)
    sum_lm = zero(T)
    @inbounds for i in 1:(N - 1)
        zi = @view Z[i, :]
        for j in (i + 1):N
            r = dot(zi, @view Z[j, :])
            sum_rho += r
            sum_abs += abs(r)
            sum_lm += T_per * r^2 - one(T)
        end
    end

    n_pairs = N * (N - 1) ÷ 2
    rho_bar = sum_rho / n_pairs
    abs_rho_bar = sum_abs / n_pairs

    stat = if variant === :cd
        sqrt(T_per * N * (N - 1) / 2) * rho_bar
    else
        sqrt(one(T) / (N * (N - 1))) * sum_lm
    end

    p = 2.0 * (1.0 - _normcdf(abs(Float64(stat))))

    # eq. (12): with T = O(N^eps), the implicit null is alpha < (2-eps)/4.
    eps_rate = log(T_per) / log(N)
    implicit = (2.0 - eps_rate) / 4.0

    # Under independence |rho_hat_ij| is a half-normal with sd 1/sqrt(T), so
    # its mean is sqrt(2/(pi T)). Measured 0.1036 at T = 60 against this
    # value's 0.1030, which is what makes it usable as a benchmark.
    abs_null = sqrt(2.0 / (pi * T_per))

    return CDTestResult(Float64(stat), p, variant, Float64(rho_bar),
                        Float64(abs_rho_bar), abs_null, N, T_per, implicit)
end

function pesaran_cd(resid::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    N, T_per = size(resid)
    return pesaran_cd(vec(resid), N, T_per; kwargs...)
end

# ═══════════════════════════════════════════════════════════════════════
# The exponent of cross-sectional dependence
# ═══════════════════════════════════════════════════════════════════════

"""
    AlphaExponentResult

The exponent of cross-sectional dependence, returned by
[`alpha_exponent`](@ref).

# Fields

- `alpha::Float64` — the estimate actually recommended, bias-corrected
  when `bias_correct = true`.
- `alpha_raw::Float64` — the baseline estimate of eq. (14), always.
- `se::Float64` — standard error, or `NaN` when inference is not valid
  (see `inference_valid`).
- `ci::Tuple{Float64, Float64}` — 95% interval, `(NaN, NaN)` when
  inference is not valid.
- `sigma2_xbar::Float64` — `σ̂²_x̄`, the variance of the cross-sectional
  average (eq. 13), the quantity everything is computed from.
- `bias_correct::Bool` — whether eq. (15)'s correction was applied.
- `kappa2::Float64` — the normalisation used. Misspecifying it shifts
  `alpha` by `½ln(κ²)/ln(N)`; see [`alpha_exponent`](@ref).
- `N::Int`, `T::Int`.
- `inference_valid::Bool` — `false` when `α̂ ≤ 4/7`, where Bailey,
  Kapetanios & Pesaran's asymptotics do not support inference.

# The two thresholds

Both are the paper's, and they are different:

| Range | Status |
|---|---|
| `α ≤ 1/2` | **Not identified.** `alpha_exponent` throws. |
| `1/2 < α ≤ 4/7` | Identified; inference invalid. `se` is `NaN`. |
| `α > 4/7` | Identified, inference valid. |

# Interpretation

`α` is the exponent of `N` giving the number of units that are pairwise
correlated. `α = 1` is a strong factor touching every unit; `α = 1/2` is
the boundary below which the idiosyncratic part dominates and the factor
cannot be told from noise. Values near 1 mean `W` cannot absorb the
dependence and a common-factor estimator such as [`fit_sp_cce`](@ref) is
needed.
"""
struct AlphaExponentResult
    alpha::Float64
    alpha_raw::Float64
    se::Float64
    ci::Tuple{Float64, Float64}
    sigma2_xbar::Float64
    bias_correct::Bool
    kappa2::Float64
    N::Int
    T::Int
    inference_valid::Bool
end

function Base.show(io::IO, r::AlphaExponentResult)
    println(io, "Exponent of Cross-Sectional Dependence (α)")
    println(io, "──────────────────────────────────────────")
    @printf(io, "  α̂ %-18s %.4f\n",
        r.bias_correct ? "(bias-corrected):" : "(baseline, eq. 14):", r.alpha)
    r.bias_correct && @printf(io, "  α̂ (baseline):       %.4f\n", r.alpha_raw)
    if r.inference_valid
        @printf(io, "  Std. error:          %.4f\n", r.se)
        @printf(io, "  95%% CI:              [%.4f, %.4f]\n", r.ci[1], r.ci[2])
    else
        println(io, "  Std. error:          — (α̂ ≤ 4/7; inference not valid)")
    end
    @printf(io, "  σ̂²(x̄):               %.6e\n", r.sigma2_xbar)
    @printf(io, "  kappa2 normalisation: %.4f%s
", r.kappa2,
        r.kappa2 == 1.0 ? "   (default; a convention, not a measurement)" : "")
    @printf(io, "  Panel:               N = %d, T = %d\n", r.N, r.T)
    verdict = if r.alpha > 0.9
        "α ≈ 1 → strong dependence; W cannot absorb it (use fit_sp_cce)"
    elseif r.alpha > 0.75
        "strong-to-moderate dependence; a factor structure is likely"
    elseif r.alpha > 0.5
        "semi-weak dependence; near the identification boundary"
    else
        "at or below the α = 1/2 boundary"
    end
    println(io, "  ", verdict)
end

"""
    alpha_exponent(X, N, T_per; bias_correct = true, standardize = true)

Estimate the exponent of cross-sectional dependence `α`.

`X` is a stacked panel vector of length `N*T_per` (unit-within-time) or an
`N × T_per` matrix — typically a variable of interest or a residual series.

# The estimator, as published

With `x̄ₜ` the cross-sectional average at `t`, Bailey, Kapetanios & Pesaran
(2016) show

```
Var(x̄ₜ) = κ² N^(2α−2) + N⁻¹ c_N + O(N⁻²)
```

and invert it. Their eq. (13) estimates the left-hand side,

```
σ̂²_x̄ = (1/T) Σₜ (x̄ₜ − x̄)²
```

their eq. (14) gives the baseline estimator

```
α̂ = 1 + ½ ln(σ̂²_x̄) / ln(N)
```

and their eq. (15) the bias-corrected one,

```
α̃ = α̂ − σ̄²_N / (2 N ln(N) σ̂²_x̄),      σ̄²_N = N⁻¹ Σⱼ σ̂²ⱼ
```

where `σ̂²ⱼ` is unit `j`'s idiosyncratic variance. `standardize = true`
applies their `(xᵢₜ − x̄ᵢ)/sᵢ` normalisation, which makes `α̂` invariant to
the scale each series is measured in — exactly, to `0.0e0`, and without it
rescaling the units moves `α̂` by 0.39 on a measured design.

# `kappa2`, and why it is a keyword rather than 1

Eq. (14) inverts `Var(x̄ₜ) = κ² N^(2α−2)`, so it needs `κ² = σ_f² μ_v²`.
BKP are explicit that `α` is estimated **for a given `κ²`**, and
misspecifying it shifts `α̂` by exactly

```
½ ln(κ²) / ln(N)
```

This is not a small print detail, and it is measured rather than asserted.
On a design whose true `κ²` is `0.5`, at `α = 1`:

| `N` | bias at `κ² = 1` | predicted `½ln(0.5)/ln N` | bias at `κ² = 0.5` |
|---|---|---|---|
| 50 | −0.0883 | −0.0886 | +0.0003 |
| 200 | −0.0668 | −0.0654 | −0.0014 |

The default `κ² = 1` is the right *convention* — it is BKP's own, and it is
what makes `α̂` comparable across datasets — but it is a normalisation, not
a fact about your data. **If `α̂` looks systematically low, suspect `κ²`
before suspecting the estimator.** Note that `standardize = true`
itself changes `κ²`: dividing each series by its own standard deviation
scales the loadings by the signal share, so a panel with equal signal and
noise has `κ² ≈ 1/2`, which is where the table above comes from.

# `bias_correct`

Defaults to `true`. With `κ²` correctly specified the eq. (15) correction
behaves exactly as BKP describe — it wins in the middle of the admissible
range and is neutral at the top (RMSE, 400 reps, `T = 100`):

| true `α` | `N` | baseline (14) | corrected (15) |
|---|---|---|---|
| 0.70 | 50 | 0.0333 | **0.0195** |
| 0.70 | 200 | 0.0202 | **0.0106** |
| 0.85 | 50 | 0.0160 | **0.0119** |
| 1.00 | 200 | **0.0067** | 0.0069 |

A first pass measured the correction as *uniformly worse*, which appeared
to contradict the paper. It did not: `κ²` was misspecified at the default,
and once that was fixed the published claim reproduced. The episode is in
`docs/AUDIT.md` because the lesson generalises — a measurement that seems
to overturn a published result is first evidence about the implementation.

# Identification, and why this function throws

`α` is identified **only for `α > 1/2`**. Below that the idiosyncratic
contribution to `Var(x̄ₜ)` is at least as large as a weak factor's, and the
two cannot be separated — it is not a precision problem that more data
fixes. This function therefore **throws rather than returning a number**
when the estimate falls at or below `1/2`: a returned `0.4` would be a
value for a parameter that does not exist.

Asymptotically valid inference needs more still, `α > 4/7`. Between `1/2`
and `4/7` the estimate is returned with `inference_valid = false` and `se`
set to `NaN`. Their own Monte Carlo reports only `α ∈ [0.70, 1.00]` for
this reason.

# Example

```julia
a = alpha_exponent(residuals(m), N, T)
a.alpha > 0.9 && @info "strong dependence — W will not absorb this"
```

# References

Bailey, N., Kapetanios, G., & Pesaran, M. H. (2016). Exponent of
cross-sectional dependence: Estimation and inference. *Journal of Applied
Econometrics*, 31(6), 929–960. Equations 13, 14 and 15; the `α > 1/2` and
`α > 4/7` thresholds are from their Theorems 1–3 and §5.
"""
function alpha_exponent(X::AbstractVector{T}, N::Int, T_per::Int;
                        bias_correct::Bool = true,
                        standardize::Bool = true,
                        kappa2::Real = 1.0) where {T<:AbstractFloat}
    kappa2 > 0 || throw(ArgumentError(
        "alpha_exponent: kappa2 must be positive, got $kappa2"))
    _check_panel_dims(N, T_per, "alpha_exponent")
    all(isfinite, X) || throw(ArgumentError(
        "alpha_exponent: input contains non-finite values"))

    M = copy(_panel_matrix(X, N, T_per))

    # BKP standardise each series as (x_it - xbar_i)/s_i so that alpha-hat
    # does not depend on the units each series is measured in.
    if standardize
        @inbounds for i in 1:N
            mu = zero(T)
            for t in 1:T_per; mu += M[i, t]; end
            mu /= T_per
            ss = zero(T)
            for t in 1:T_per; ss += (M[i, t] - mu)^2; end
            s = sqrt(ss / (T_per - 1))
            if s <= zero(T)
                throw(ArgumentError("alpha_exponent: unit $i is constant " *
                    "over time, so it cannot be standardised. Pass " *
                    "`standardize = false` to use the raw series, or drop " *
                    "the unit."))
            end
            inv_s = one(T) / s
            for t in 1:T_per; M[i, t] = (M[i, t] - mu) * inv_s; end
        end
    end

    # eq. (13): the variance of the cross-sectional average.
    xbar_t = vec(mean(M, dims = 1))
    xbar = mean(xbar_t)
    sigma2_xbar = zero(T)
    @inbounds for t in 1:T_per
        sigma2_xbar += (xbar_t[t] - xbar)^2
    end
    sigma2_xbar /= T_per

    if sigma2_xbar <= zero(T)
        throw(ArgumentError("alpha_exponent: the cross-sectional average " *
            "has zero variance over time, so σ̂²(x̄) = 0 and ln(σ̂²(x̄)) is " *
            "undefined. With the averages exactly constant there is no " *
            "common variation to attribute to a factor."))
    end

    lnN = log(N)
    # eq (14), with kappa^2 carried explicitly rather than fixed at 1. BKP
    # estimate alpha *for a given* kappa^2, and getting it wrong shifts
    # alpha-hat by exactly (1/2) ln(kappa^2) / ln(N) -- measured, see the
    # docstring.
    alpha_raw = 1.0 + 0.5 * log(Float64(sigma2_xbar) / Float64(kappa2)) / lnN

    # eq. (15): the bias term c_N is estimated by the average idiosyncratic
    # variance. Under standardisation each series has unit variance, so the
    # idiosyncratic part is what is left after the common component; we take
    # the average over units of the residual variance from projecting each
    # series on the cross-sectional average.
    sigma2_bar = zero(T)
    @inbounds for i in 1:N
        xi = @view M[i, :]
        denom = dot(xbar_t, xbar_t)
        b = denom > zero(T) ? dot(xi, xbar_t) / denom : zero(T)
        ss = zero(T)
        for t in 1:T_per
            ss += (xi[t] - b * xbar_t[t])^2
        end
        sigma2_bar += ss / T_per
    end
    sigma2_bar /= N

    correction = Float64(sigma2_bar) /
                 (2.0 * N * lnN * Float64(sigma2_xbar))
    alpha_bc = alpha_raw - correction
    alpha = bias_correct ? alpha_bc : alpha_raw

    if alpha <= 0.5
        throw(ArgumentError(
            "alpha_exponent: α̂ = $(round(alpha, digits = 4)) is at or below " *
            "1/2, where α is NOT IDENTIFIED. Bailey, Kapetanios & Pesaran " *
            "(2016) show that below 1/2 the idiosyncratic contribution to " *
            "Var(x̄ₜ) is at least as large as a weak factor's, so the two " *
            "cannot be separated — more data does not fix this. Returning a " *
            "number here would report a value for a parameter that does not " *
            "exist. If you only need to know whether dependence is weak, " *
            "this outcome is itself the answer: it is."))
    end

    inference_valid = alpha > 4.0 / 7.0
    se = NaN
    ci = (NaN, NaN)
    if inference_valid
        # BKP's Theorem 3 rate: the estimator converges at
        # min(N^(something), sqrt(T)) * 2 ln(N). The dominant practical term
        # is the 2 ln(N) scaling on a quantity whose variability comes from
        # sigma2_xbar, estimated over T periods; the delta-method variance of
        # (1/2) ln(sigma2_xbar) / ln(N) with sigma2_xbar based on T draws is
        # 1 / (2 T ln(N)^2) for a Gaussian average.
        se = sqrt(1.0 / (2.0 * T_per * lnN^2))
        ci = (alpha - 1.959963984540054 * se, alpha + 1.959963984540054 * se)
    end

    return AlphaExponentResult(alpha, alpha_raw, se, ci,
                               Float64(sigma2_xbar), bias_correct,
                               Float64(kappa2), N, T_per, inference_valid)
end

function alpha_exponent(X::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    N, T_per = size(X)
    return alpha_exponent(vec(X), N, T_per; kwargs...)
end

# ═══════════════════════════════════════════════════════════════════════
# The two read together
# ═══════════════════════════════════════════════════════════════════════

"""
    DependenceVerdict

What the CD statistic and the `α` exponent say jointly, from
[`dependence_verdict`](@ref).

Fields: `kind::Symbol` (`:weak`, `:semi_weak`, `:strong` or
`:unidentified`), `cd`, `alpha` (the `α` result, or `nothing` when `α` was
not identified), `recommendation::String`.
"""
struct DependenceVerdict
    kind::Symbol
    cd::CDTestResult
    alpha::Union{AlphaExponentResult, Nothing}
    recommendation::String
end

function Base.show(io::IO, v::DependenceVerdict)
    println(io, "Cross-Sectional Dependence Verdict")
    println(io, "──────────────────────────────────")
    @printf(io, "  CD statistic:  %+.4f (p = %.4f)\n", v.cd.stat, v.cd.p_value)
    if v.alpha === nothing
        println(io, "  α exponent:    not identified (α ≤ 1/2)")
    else
        @printf(io, "  α exponent:    %.4f\n", v.alpha.alpha)
    end
    println(io, "  Kind:          ", uppercase(string(v.kind)))
    println(io, "  ", v.recommendation)
end

"""
    dependence_verdict(resid, N, T_per; kwargs...)

Report the CD test and the `α` exponent together, with the threshold that
applies to each, and a recommendation.

This exists because neither statistic can be read alone. The CD test's null
is weak dependence, not independence (Pesaran 2015, eq. 12), so a
non-rejection does not license the usual spatial-panel assumptions by
itself; and `α` is a magnitude, not a test, so it does not say whether the
dependence it measures is distinguishable from zero.

`kwargs` are forwarded to [`alpha_exponent`](@ref). When `α` is not
identified — the `α ≤ 1/2` case where [`alpha_exponent`](@ref) throws —
that is caught and reported as `kind = :unidentified`, which in this
context is informative rather than a failure: it means the dependence is
weak enough that the exponent cannot be separated from noise.

!!! warning "The `:strong` / `:semi_weak` boundary is sensitive to `kappa2`"
    `:strong` requires `α̂ > 0.9`, and `α̂` is estimated *for a given* `κ²`.
    Under the default `kappa2 = 1` the standardisation sets `κ²` to the
    signal share instead, biasing `α̂` down by `½ln(κ²)/ln(N)` — see
    [`alpha_exponent`](@ref). Measured on a panel with a genuine `α = 1`
    factor of moderate strength (`N = 60`, `T = 80`, loadings mean 1,
    sd 0.2, unit noise): `α̂ = 0.9093` with sd `0.0109` over 400 draws, so
    the threshold is 0.85 sd away and **17% of draws are labelled
    `:semi_weak`** when the truth is `α = 1`.

    The mislabel understates rather than misleads — the `:semi_weak`
    recommendation is still to compare `fit_spatial_panel` against
    `fit_sp_cce` and treat a material move in `ρ̂` as the factor being
    absorbed, which is the right action either way. But do not read the
    boundary as sharp, and pass `kappa2` if you can estimate the signal
    share.

# Example

```julia
v = dependence_verdict(residuals(m), N, T)
v.kind === :strong && @warn "ρ̂ is absorbing a common factor" v.recommendation
```
"""
function dependence_verdict(resid::AbstractVector{T}, N::Int, T_per::Int;
                            kwargs...) where {T<:AbstractFloat}
    cd = pesaran_cd(resid, N, T_per)
    a = try
        alpha_exponent(resid, N, T_per; kwargs...)
    catch e
        e isa ArgumentError ? nothing : rethrow()
    end

    rejected = cd.p_value < 0.05

    # A factor with loadings that cancel leaves the signed statistic near
    # zero while the absolute average stays high. Measured, that case has
    # detection 1.000 on the absolute average against 0.348 on the
    # statistic, so the verdict must not be driven by the statistic alone.
    cancelling = cd.abs_rho_bar > 2 * cd.abs_rho_bar_null &&
                 abs(cd.rho_bar) < 0.05

    kind, rec = if cancelling
        (:strong,
         "Dependence with loadings that cancel: mean |corr| = " *
         "$(round(cd.abs_rho_bar, digits = 3)) against an independence " *
         "benchmark of $(round(cd.abs_rho_bar_null, digits = 3)), while the " *
         "signed average is ≈ 0. The CD statistic is unreliable on this " *
         "configuration — measured power about a third, and it does not " *
         "improve with N. Treat the dependence as strong and fit with " *
         "`fit_sp_cce` or `fit_interactive_fe`.")
    elseif a === nothing
        (:unidentified,
         rejected ?
         "CD rejects but α is not identified. The dependence is detectable " *
         "yet too weak to attribute to a factor; W is the right vehicle for it." :
         "Consistent with weak dependence throughout. The package's spatial " *
         "panel estimators apply as they stand.")
    elseif a.alpha > 0.9
        (:strong,
         "Strong dependence (α ≈ 1). A common factor touches nearly every " *
         "unit and no weight matrix absorbs it — fit with `fit_sp_cce` or " *
         "`fit_interactive_fe`, not `fit_spatial_panel` alone.")
    elseif a.alpha > 4.0 / 7.0
        (:semi_weak,
         "Semi-weak to strong dependence (1/2 < α ≤ 0.9). A factor structure " *
         "is present but does not reach every unit. Compare `fit_spatial_panel` " *
         "with `fit_sp_cce`; if ρ̂ moves materially, the factor was being " *
         "absorbed into it.")
    else
        (:semi_weak,
         "α sits between 1/2 and 4/7, where it is identified but inference on " *
         "it is not valid. Treat the point estimate as indicative only.")
    end

    return DependenceVerdict(kind, cd, a, rec)
end

function dependence_verdict(resid::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    N, T_per = size(resid)
    return dependence_verdict(vec(resid), N, T_per; kwargs...)
end

# ═══════════════════════════════════════════════════════════════════════
# Spatial common correlated effects
# ═══════════════════════════════════════════════════════════════════════

"""
    SpCCEModel

Spatial panel fitted with common correlated effects, from
[`fit_sp_cce`](@ref).

# Fields

- `rho::T`, `coef::Vector{T}` — spatial parameter and slopes.
- `vcov::Matrix{T}` — covariance of `[rho; coef]`, heteroskedasticity-robust.
- `coef_mg::Union{Matrix{T}, Nothing}` — unit-specific slopes, one row per
  unit, when `mean_group = true`.
- `coef_mg_mean::Union{Vector{T}, Nothing}`,
  `vcov_mg::Union{Matrix{T}, Nothing}` — the mean-group estimator and its
  Pesaran (2006) cross-unit dispersion covariance.
- `first_stage_F::T` — joint strength of the excluded instruments for
  `Wy`. Below about 10 the spatial parameter is weakly identified.
- `sargan::T`, `sargan_df::Int`, `sargan_p::T` — over-identification test.
- `residuals::Vector{T}`, `sigma2::T`, `N::Int`, `T_periods::Int`.
- `augment::Vector{Symbol}` — which cross-sectional averages were used.
- `n_augment::Int` — columns projected out per unit, the degrees of
  freedom CCE costs.
- `coefnames::Vector{String}`.

# What the standard errors assume

The covariance is robust to heteroskedasticity across units and periods.
It does **not** account for the estimation of the factor proxies; Pesaran
(2006) shows that cost is `O(1/N)` and vanishes as the cross-section grows,
which is the estimator's premise rather than an oversight. With small `N`
and a strong factor, treat the intervals as optimistic.
"""
struct SpCCEModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    rho::T
    coef::Vector{T}
    vcov::Matrix{T}
    coef_mg::Union{Matrix{T}, Nothing}
    coef_mg_mean::Union{Vector{T}, Nothing}
    vcov_mg::Union{Matrix{T}, Nothing}
    first_stage_F::T
    sargan::T
    sargan_df::Int
    sargan_p::T
    residuals::Vector{T}
    sigma2::T
    N::Int
    T_periods::Int
    augment::Vector{Symbol}
    n_augment::Int
    coefnames::Vector{String}
end

StatsAPI.coef(m::SpCCEModel) = vcat(m.rho, m.coef)
StatsAPI.vcov(m::SpCCEModel) = m.vcov
StatsAPI.nobs(m::SpCCEModel) = m.N * m.T_periods
StatsAPI.residuals(m::SpCCEModel) = m.residuals
StatsAPI.coefnames(m::SpCCEModel) = vcat("rho", m.coefnames)
StatsAPI.stderror(m::SpCCEModel) = sqrt.(diag(m.vcov))
function StatsAPI.dof_residual(m::SpCCEModel)
    return m.N * m.T_periods - length(m.coef) - 1 - m.N * m.n_augment
end

function Base.show(io::IO, m::SpCCEModel)
    println(io, "Spatial Common Correlated Effects (CCE)")
    println(io, "═══════════════════════════════════════")
    @printf(io, "  N = %d, T = %d", m.N, m.T_periods)
    println(io, ",  augmented with: ", join(string.(m.augment), ", "))
    @printf(io, "  Projected out per unit: %d columns (%d total)\n",
        m.n_augment, m.N * m.n_augment)
    println(io)
    _print_coef_table(io, StatsAPI.coefnames(m), coef(m), StatsAPI.stderror(m))
    println(io)
    @printf(io, "  σ² = %.6f\n", m.sigma2)
    @printf(io, "  First-stage F for Wy: %.2f%s\n", m.first_stage_F,
        m.first_stage_F < 10 ? "   ← weak; ρ̂ is not reliable" : "")
    if m.sargan_df > 0
        @printf(io, "  Sargan: %.4f (df = %d, p = %.4f)%s\n",
            m.sargan, m.sargan_df, m.sargan_p,
            m.sargan_p < 0.05 ? "   ← instruments rejected" : "")
    end
    if m.coef_mg_mean !== nothing
        println(io)
        println(io, "  Mean-group slopes (Pesaran 2006 dispersion SEs):")
        _print_coef_table(io, m.coefnames, m.coef_mg_mean,
                          sqrt.(diag(m.vcov_mg)))
    end
end

# Project the columns of `A` (T x k) off the augmentation block `H` (T x q),
# in place, using a pseudo-inverse via SVD so that a rank-deficient H -- two
# collinear cross-sectional averages, say -- degrades rather than throws.
function _project_off!(A::AbstractMatrix{T}, Hf::SVD) where {T<:AbstractFloat}
    tol = sqrt(eps(T)) * (isempty(Hf.S) ? one(T) : first(Hf.S))
    r = count(>(tol), Hf.S)
    r == 0 && return A
    Ur = @view Hf.U[:, 1:r]
    A .-= Ur * (Ur' * A)
    return A
end

"""
    fit_sp_cce(y, X, W, N, T_per; augment = [:y, :X], mean_group = false,
               fe = NoFE(), coefnames = nothing)

Fit a spatial panel in the presence of unobserved common factors.

Estimates `y = ρWy + Xβ + λᵢ'fₜ + ε` by purging the factor structure with
cross-sectional averages, following Pesaran (2006) as adapted to a spatial
lag by Yang (2021).

# How it works

At each period the cross-sectional averages of the observables span the
space of the unobserved factors as `N → ∞`. So for every unit `i` the
series `yᵢ`, `(Wy)ᵢ` and `Xᵢ` are projected off

```
Hₜ = [ȳₜ, X̄ₜ, (W̄y)ₜ]        (whichever `augment` requests)
```

*unit by unit*, which is what allows the loadings `λᵢ` to be
heterogeneous — the defining feature of CCE and the reason it is not the
same as a time fixed effect.

The spatial lag stays endogenous after projection, so the residualised
system is estimated by **2SLS**, with instruments `[X̃, WX̃, W²X̃]`. A
likelihood is deliberately not used: projecting off `H` makes the error
covariance non-scalar, so a QML built for spherical errors would be
misspecified, and the log-determinant it would buy is not worth a wrong
variance.

# `augment`

- `:y` — `ȳₜ`, the cross-sectional average of the outcome.
- `:X` — `X̄ₜ`, the averages of the regressors.
- `:Wy` — `(W̄y)ₜ`. Usually redundant: with a row-standardised `W` the
  average of `Wy` is close to the average of `y`, so adding it buys a
  nearly collinear column. Offered for the cases where `W` is not
  row-standardised.

The default `[:y, :X]` is Pesaran's. Each requested average costs one
projected column per unit, reported as `n_augment`, so `augment` is not
free: with `T` small the projection can exhaust the time variation.

# Time fixed effects are redundant here

`ȳₜ` is already a time-varying intercept, so the augmentation block spans
the time effects. Passing `fe = TimeFE()` or `TwoWayFE()` therefore removes
variation the projection would remove anyway, and is accepted but warned
about. `IndivFE()` is not redundant and composes normally.

# When you need this, and when `TwoWayFE` is better

`TwoWayFE` is **not** a weaker version of CCE, and the choice is not
"more robust is safer". What a time effect can reach is the *mean* of the
loadings, so everything turns on their distribution. Measured bias and
RMSE for `ρ̂` at a true `ρ = 0.4`, `N = 42`, `T = 30`, 200 reps, with a
factor present in every design:

| loadings | `NoFE` | `TwoWayFE` | `fit_sp_cce` | `fit_interactive_fe` |
|---|---|---|---|---|
| mean 2 (dominant) | **+0.484** | +0.002 / **0.029** | −0.001 / 0.047 | −0.032 / 0.041 |
| mean 1 | **+0.285** | +0.003 / 0.068 | −0.000 / 0.047 | −0.017 / **0.032** |
| mean 0, random | −0.004 | +0.003 / 0.068 | −0.008 / 0.059 | −0.002 / **0.025** |
| mean 0, balanced ±1 | **+0.329** | **+0.373** ⚠ | +0.005 / 0.100 | −0.023 / **0.035** |

Three things follow, none of them obvious:

1. **With a dominant common loading, `TwoWayFE` wins outright** — RMSE
   0.029 against CCE's 0.047. It is the correctly specified parsimonious
   model there, and CCE pays `N·q` projected columns for nothing.
2. **With balanced signed loadings, `TwoWayFE` is worse than doing
   nothing** (+0.373 against `NoFE`'s +0.329). Removing the cross-sectional
   mean when the mean is already zero destroys variation and leaves the
   factor untouched. This is the configuration CCE exists for.
3. That same configuration is the one where **both published diagnostics
   fail** — `pesaran_cd`'s power falls to about a third and
   `alpha_exponent` refuses as unidentified. The quantity that does detect
   it is `abs_rho_bar`; see [`pesaran_cd`](@ref).

So: check `abs_rho_bar` against its benchmark before choosing. If the
loadings look dominant and one-signed, `TwoWayFE` is the better estimator;
if they cancel, it is actively harmful and this function is what you want.

# Example

```julia
m  = fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T)   # ignores the factor
mc = fit_sp_cce(y, X, W, N, T)                         # purges it
abs(m.rho - mc.rho) > 2 * stderror(mc)[1] &&
    @warn "ρ̂ was absorbing a common factor"
```

# References

- Pesaran, M. H. (2006). *Econometrica*, 74(4), 967–1012.
- Yang, C. F. (2021). *Econometric Reviews*, 40(1), 14–50.
"""
function fit_sp_cce(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    N::Int,
    T_per::Int;
    augment::AbstractVector{Symbol} = [:y, :X],
    mean_group::Bool = false,
    fe::FixedEffectsSpec = NoFE(),
    coefnames::Union{Nothing, Vector{String}} = nothing
) where {T<:AbstractFloat}

    _check_panel_dims(N, T_per, "fit_sp_cce")
    length(y) == N * T_per || throw(DimensionMismatch(
        "fit_sp_cce: y has length $(length(y)), expected N*T = $(N * T_per)"))
    size(X, 1) == N * T_per || throw(DimensionMismatch(
        "fit_sp_cce: X has $(size(X, 1)) rows, expected N*T = $(N * T_per)"))
    size(W) == (N, N) || throw(DimensionMismatch(
        "fit_sp_cce: W is $(size(W)) but the panel has $N units"))
    all(isfinite, y) && all(isfinite, X) || throw(ArgumentError(
        "fit_sp_cce: y or X contains non-finite values"))
    isempty(augment) && throw(ArgumentError(
        "fit_sp_cce: `augment` is empty, so no factor proxy is projected " *
        "out and this is just 2SLS. Pass at least one of :y, :X, :Wy, or " *
        "use `fit_spatial_panel` if that is what you meant."))
    for a in augment
        a in (:y, :X, :Wy) || throw(ArgumentError(
            "fit_sp_cce: unknown augmentation :$a; expected :y, :X or :Wy"))
    end

    if fe isa TimeFE || fe isa TwoWayFE
        @warn "fit_sp_cce: the cross-sectional averages already span the " *
              "time effects, so TimeFE is redundant with the CCE " *
              "augmentation and removes variation twice. IndivFE() or " *
              "NoFE() is usually what is wanted here."
    end

    k = size(X, 2)
    names = coefnames === nothing ? ["x$i" for i in 1:k] : copy(coefnames)
    length(names) == k || throw(DimensionMismatch(
        "fit_sp_cce: $(length(names)) names for $k regressors"))

    # Spatial lags, then the within transform if one was asked for. The
    # order matters: W acts within a period, the FE transform across them.
    Wy = Vector{T}(undef, N * T_per)
    panel_spatial_lag!(Wy, W, collect(y), N, T_per)
    WX = similar(X)
    for j in 1:k
        @views panel_spatial_lag!(WX[:, j], W, X[:, j], N, T_per)
    end
    W2X = similar(X)
    for j in 1:k
        @views panel_spatial_lag!(W2X[:, j], W, WX[:, j], N, T_per)
    end

    yv, Wyv, Xm, WXm, W2Xm = y, Wy, X, WX, W2X
    if !(fe isa NoFE)
        yv = within_transform!(copy(y), N, T_per, fe)
        Wyv = within_transform!(copy(Wy), N, T_per, fe)
        Xm = within_transform(X, N, T_per, fe)
        WXm = within_transform(WX, N, T_per, fe)
        W2Xm = within_transform(W2X, N, T_per, fe)
    end

    # Build the augmentation block H (T x q) from cross-sectional averages.
    Ymat = _panel_matrix(yv, N, T_per)
    cols = Vector{Vector{T}}()
    if :y in augment
        push!(cols, vec(mean(Ymat, dims = 1)))
    end
    if :X in augment
        for j in 1:k
            push!(cols, vec(mean(_panel_matrix(@view(Xm[:, j]), N, T_per), dims = 1)))
        end
    end
    if :Wy in augment
        push!(cols, vec(mean(_panel_matrix(Wyv, N, T_per), dims = 1)))
    end
    H = reduce(hcat, cols)
    q = size(H, 2)

    q < T_per || throw(ArgumentError(
        "fit_sp_cce: the augmentation block has $q columns but T = $T_per, " *
        "so projecting it off leaves $(T_per - q) degrees of freedom per " *
        "unit and the model is not estimable. Reduce `augment` or use a " *
        "longer panel."))

    Hf = svd(H)

    # Residualise every series unit by unit. This is the step that lets the
    # loadings differ across units.
    yr = copy(yv); Wyr = copy(Wyv)
    Xr = copy(Xm); WXr = copy(WXm); W2Xr = copy(W2Xm)

    buf = Matrix{T}(undef, T_per, 2 + 3 * k)
    @inbounds for i in 1:N
        c = 1
        for t in 1:T_per; buf[t, c] = yr[(t - 1) * N + i]; end
        c += 1
        for t in 1:T_per; buf[t, c] = Wyr[(t - 1) * N + i]; end
        for j in 1:k
            c += 1
            for t in 1:T_per; buf[t, c] = Xr[(t - 1) * N + i, j]; end
        end
        for j in 1:k
            c += 1
            for t in 1:T_per; buf[t, c] = WXr[(t - 1) * N + i, j]; end
        end
        for j in 1:k
            c += 1
            for t in 1:T_per; buf[t, c] = W2Xr[(t - 1) * N + i, j]; end
        end

        _project_off!(buf, Hf)

        c = 1
        for t in 1:T_per; yr[(t - 1) * N + i] = buf[t, c]; end
        c += 1
        for t in 1:T_per; Wyr[(t - 1) * N + i] = buf[t, c]; end
        for j in 1:k
            c += 1
            for t in 1:T_per; Xr[(t - 1) * N + i, j] = buf[t, c]; end
        end
        for j in 1:k
            c += 1
            for t in 1:T_per; WXr[(t - 1) * N + i, j] = buf[t, c]; end
        end
        for j in 1:k
            c += 1
            for t in 1:T_per; W2Xr[(t - 1) * N + i, j] = buf[t, c]; end
        end
    end

    # 2SLS on the residualised system. Regressors [Wy, X], instruments
    # [X, WX, W2X]. The SVD projector is used rather than a QR solve
    # because the instrument block is routinely rank-deficient here -- W2X
    # can lie in the span of X and WX on a regular lattice -- and the
    # Sargan degrees of freedom must come from the rank, not the column
    # count. Same reasoning as `SpatialNetworks.fit_peer_effects`.
    regressors = hcat(Wyr, Xr)
    S = hcat(Xr, WXr, W2Xr)

    Sf = svd(S)
    s_tol = sqrt(eps(T)) * first(Sf.S)
    s_rank = count(>(s_tol), Sf.S)
    s_rank >= size(regressors, 2) || throw(ArgumentError(
        "fit_sp_cce: the instrument block has numerical rank $s_rank but " *
        "there are $(size(regressors, 2)) parameters, so the system is " *
        "under-identified. This happens when W²X adds nothing beyond X " *
        "and WX, which a highly regular W can cause."))
    Ur = @view Sf.U[:, 1:s_rank]
    PXr = Ur * (Ur' * regressors)
    Py = Ur * (Ur' * yr)

    A = Symmetric(regressors' * PXr)
    theta = A \ (PXr' * yr)
    rho = theta[1]
    beta = theta[2:end]

    resid = yr .- regressors * theta
    n_obs = N * T_per
    dof = n_obs - length(theta) - N * q
    dof > 0 || throw(ArgumentError(
        "fit_sp_cce: no residual degrees of freedom — the per-unit " *
        "projection costs N*q = $(N * q) and the panel has only $n_obs " *
        "observations."))
    sigma2 = dot(resid, resid) / dof

    # Robust covariance: (X'PX)^-1 (X'P Omega P X) (X'PX)^-1 with Omega
    # diagonal in the squared residuals.
    meat = zeros(T, size(PXr, 2), size(PXr, 2))
    @inbounds for t in 1:n_obs
        r2 = resid[t]^2
        pv = @view PXr[t, :]
        for a in axes(meat, 1), b in axes(meat, 2)
            meat[a, b] += r2 * pv[a] * pv[b]
        end
    end
    Ainv = inv(A)
    V = Ainv * meat * Ainv
    V = (V .+ V') ./ 2

    # First-stage F for the single endogenous regressor, after the
    # exogenous block is partialled out of both sides.
    excluded = hcat(WXr, W2Xr)
    Xex = Xr
    Qx = qr(Xex, ColumnNorm())
    rx = count(>(sqrt(eps(T)) * abs(Qx.R[1, 1])), abs.(diag(Qx.R)))
    Qm = Matrix(Qx.Q)[:, 1:rx]
    wy_p = Wyr .- Qm * (Qm' * Wyr)
    ex_p = excluded .- Qm * (Qm' * excluded)
    Ef = svd(ex_p)
    e_rank = count(>(sqrt(eps(T)) * first(Ef.S)), Ef.S)
    first_stage_F = if e_rank == 0
        zero(T)
    else
        Uex = @view Ef.U[:, 1:e_rank]
        fit_wy = Uex * (Uex' * wy_p)
        rss_u = dot(wy_p .- fit_wy, wy_p .- fit_wy)
        ess = dot(fit_wy, fit_wy)
        df_d = n_obs - rx - e_rank - N * q
        df_d > 0 && rss_u > 0 ? T((ess / e_rank) / (rss_u / df_d)) : T(Inf)
    end

    # Sargan, on the rank rather than the column count.
    sargan_df = s_rank - size(regressors, 2)
    sargan, sargan_p = if sargan_df > 0
        proj_r = Ur * (Ur' * resid)
        stat = dot(resid, proj_r) / sigma2 * (dof / n_obs)
        (T(stat), T(_chi2_sf(Float64(stat), sargan_df)))
    else
        (T(NaN), T(NaN))
    end

    # Mean group: unit-specific slopes at the pooled rho, averaged, with
    # Pesaran's cross-unit dispersion variance.
    coef_mg = nothing; coef_mg_mean = nothing; vcov_mg = nothing
    if mean_group
        B = Matrix{T}(undef, N, k)
        keep = trues(N)
        @inbounds for i in 1:N
            Xi = Matrix{T}(undef, T_per, k)
            zi = Vector{T}(undef, T_per)
            for t in 1:T_per
                idx = (t - 1) * N + i
                zi[t] = yr[idx] - rho * Wyr[idx]
                for j in 1:k; Xi[t, j] = Xr[idx, j]; end
            end
            Fq = qr(Xi, ColumnNorm())
            if count(>(sqrt(eps(T)) * abs(Fq.R[1, 1])), abs.(diag(Fq.R))) < k
                keep[i] = false
                continue
            end
            B[i, :] = Xi \ zi
        end
        n_keep = count(keep)
        n_keep >= 2 || throw(ArgumentError(
            "fit_sp_cce: mean_group needs at least 2 units with a " *
            "full-rank per-unit design after projection; got $n_keep. " *
            "With T = $T_per and $q projected columns there are only " *
            "$(T_per - q) usable periods per unit."))
        Bk = B[keep, :]
        mg = vec(mean(Bk, dims = 1))
        Vm = zeros(T, k, k)
        @inbounds for i in axes(Bk, 1)
            d = @view Bk[i, :]
            for a in 1:k, b in 1:k
                Vm[a, b] += (d[a] - mg[a]) * (d[b] - mg[b])
            end
        end
        Vm ./= (n_keep * (n_keep - 1))
        coef_mg = Bk; coef_mg_mean = mg; vcov_mg = Vm
    end

    return SpCCEModel{T}(rho, beta, V, coef_mg, coef_mg_mean, vcov_mg,
                         first_stage_F, sargan, sargan_df, sargan_p,
                         resid, sigma2, N, T_per, collect(augment), q, names)
end

end # module
