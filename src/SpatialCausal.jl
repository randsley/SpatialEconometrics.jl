module SpatialCausal

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames, dof_residual

import ..SpatialUtils
import ..SpatialUtils: check_design_rank, _print_coef_table
import ..SpatialHAC: AbstractRobustVCOV, ConleySpatialHAC, ClusterRobustVCOV,
       DriscollKraayVCOV, vcov_conley, vcov_cluster, vcov_driscoll_kraay
import ..SpatialWeights: band_weights, row_standardize

export ExposureMap, exposure, ring_exposure,
       InterferenceReport, check_interference,
       SpatialDiDResult, fit_spatial_did

# ═══════════════════════════════════════════════════════════════════════
# Exposure Mapping
# ═══════════════════════════════════════════════════════════════════════

"""
    ExposureMap

A treatment vector together with its spatial exposure columns, returned by
[`exposure`](@ref) and [`ring_exposure`](@ref).

An exposure mapping is the object that makes a spillover estimable. SUTVA
assumes unit `i`'s outcome responds only to `D_i`; an exposure mapping
replaces that with a named, finite summary of the treatment *around* `i`,
and the spillover parameter is then defined relative to that summary rather
than left implicit.

# Fields
- `D`: the treatment indicator, one entry per unit.
- `E`: `N x order` exposure columns. Column `k` is `W^k D`.
- `order`: how many spatial orders `E` carries.
- `labels`: names for the columns of `E`, e.g. `["WD", "W2D"]`.
- `standardized`: whether `W` was row-standardized before forming `E`,
  which is what makes column 1 the *fraction* of `i`'s neighbours treated.

!!! warning "The mapping is an assumption, not a measurement"
    Every spillover estimate is relative to the mapping that produced it.
    A `δ` estimated on `W` built from 5 nearest neighbours answers a
    different question from one built on a 50 km band, and neither is
    "the" spillover. Reporting the mapping alongside the estimate is not
    optional, which is why this type is carried in
    [`SpatialDiDResult`](@ref) rather than discarded after the fit.
"""
struct ExposureMap{T<:AbstractFloat}
    D::Vector{T}
    E::Matrix{T}
    order::Int
    labels::Vector{String}
    standardized::Bool
end

"""
    exposure(D, W; order = 1, standardize = true)

Build an [`ExposureMap`](@ref) from a treatment indicator and a weight
matrix: column `k` of `E` is `W^k D`.

# Arguments
- `D`: treatment indicator, length `N`. `Bool`, integer or float; anything
  other than 0 and 1 is rejected, because a fractional "treatment" makes
  the direct/spillover split ill-defined.
- `W`: `N x N` sparse weight matrix.

# Keywords
- `order`: number of spatial orders. `1` gives `WD` alone; `2` adds `W²D`,
  which lets a second ring differ from the first instead of being forced
  to decay at whatever rate `W` implies.
- `standardize`: row-standardize `W` first. On (the default), `WD` is the
  **fraction** of `i`'s neighbours that are treated, which is bounded in
  `[0, 1]` and comparable across units with different neighbour counts.
  Off, it is a count, and a unit with twenty neighbours is mechanically
  more "exposed" than one with two.

# Examples
```julia
em = exposure(D, W)                 # fraction of neighbours treated
em = exposure(D, W; order = 2)      # first and second rings separately
```
"""
function exposure(D::AbstractVector, W::AbstractSparseMatrix;
                  order::Int=1, standardize::Bool=true)
    N = length(D)
    size(W, 1) == size(W, 2) || throw(DimensionMismatch("`W` is not square: $(size(W))"))
    size(W, 1) == N || throw(DimensionMismatch("`W` is $(size(W, 1))x$(size(W, 2)) but `D` has $N entries"))
    order >= 1 || throw(ArgumentError("`order` must be at least 1, got $order"))

    T = promote_type(Float64, eltype(W) <: AbstractFloat ? eltype(W) : Float64)
    Dv = Vector{T}(undef, N)
    @inbounds for i in 1:N
        d = T(D[i])
        (d == zero(T) || d == one(T)) ||
            throw(ArgumentError("`D[$i]` is $(D[i]); the treatment indicator must be 0 or 1, " *
                                "because the direct effect is defined as the response to a unit " *
                                "switching from untreated to treated"))
        Dv[i] = d
    end
    all(iszero, Dv) && throw(ArgumentError("no unit is treated, so there is no effect to estimate"))
    all(isone, Dv) && throw(ArgumentError("every unit is treated, so there is no comparison group"))

    Wt = standardize ? row_standardize(T.(W)) : T.(W)

    E = Matrix{T}(undef, N, order)
    col = Wt * Dv
    E[:, 1] = col
    for k in 2:order
        col = Wt * col
        E[:, k] = col
    end

    labels = [k == 1 ? "WD" : "W$(k)D" for k in 1:order]
    return ExposureMap{T}(Dv, E, order, labels, standardize)
end

"""
    ring_exposure(D, coords, d_min, d_max; order = 1, standardize = true)

Build an [`ExposureMap`](@ref) from a distance band rather than from an
existing `W`: a unit is exposed through neighbours at distance in
`(d_min, d_max]`.

A ring is what separates a spillover from a displacement effect. With
`d_min = 0` the band includes immediate neighbours, where a policy's effect
and its displacement are mixed together; starting the band beyond them
reads the effect at a distance where mechanical displacement is implausible.

Builds the band with [`band_weights`](@ref) and hands it to
[`exposure`](@ref).

# Examples
```julia
near = ring_exposure(D, coords, 0.0, 10.0)    # within 10 units
far  = ring_exposure(D, coords, 10.0, 25.0)   # the next ring out
```
"""
function ring_exposure(D::AbstractVector, coords::AbstractMatrix, d_min::Real, d_max::Real;
                       order::Int=1, standardize::Bool=true)
    size(coords, 1) == length(D) ||
        throw(DimensionMismatch("`coords` has $(size(coords, 1)) rows but `D` has $(length(D)) entries"))
    W = band_weights(coords, d_min, d_max; row_standardize=false)
    return exposure(D, W; order=order, standardize=standardize)
end

# ═══════════════════════════════════════════════════════════════════════
# Interference Diagnostics
# ═══════════════════════════════════════════════════════════════════════

"""
    InterferenceReport

What an exposure design can and cannot identify, returned by
[`check_interference`](@ref).

# Fields
- `n_treated`, `n_control`: units by treatment status.
- `n_control_unexposed`: controls with zero exposure — the units that
  carry the identification of the spillover.
- `n_treated_unexposed`: treated units with no treated neighbour.
- `exposure_var_control`: variance of exposure among controls.
- `exposure_cor`: correlation between `D` and first-order exposure.
- `mean_exposure_treated`, `mean_exposure_control`: the two means the
  reported spillovers are evaluated at.
- `identified`: whether the spillover is separable from the time effect.
- `message`: what to do about it when it is not.

# Why `n_control_unexposed` is the number to look at

The spillover coefficient is identified by controls that differ in
exposure. If every control has a treated neighbour there is no unexposed
baseline, and `δ` is soaked up by the constant: the fit still returns a
number, but it is the difference between *more* and *less* exposed, not
between exposed and not. If no control has a treated neighbour, `δ` rests
entirely on the treated units' own exposure and cannot be separated from
the direct effect at all.
"""
struct InterferenceReport{T<:AbstractFloat}
    n_treated::Int
    n_control::Int
    n_control_unexposed::Int
    n_treated_unexposed::Int
    exposure_var_control::T
    exposure_cor::T
    mean_exposure_treated::T
    mean_exposure_control::T
    identified::Bool
    message::String
end

"""
    check_interference(em::ExposureMap; var_tol = 1e-10)

Assess whether the exposure design in `em` can identify a spillover, and
return an [`InterferenceReport`](@ref).

This is the "check the no-interference-beyond-`W` assumption" step. It is a
property of the design and the weight matrix alone — it needs no outcome —
so it can and should be run before any fitting.

# Examples
```julia
em = exposure(D, W)
rep = check_interference(em)
rep.identified || @warn rep.message
```
"""
function check_interference(em::ExposureMap{T}; var_tol::Real=1e-10) where {T}
    D = em.D
    E1 = @view em.E[:, 1]
    treated = D .== one(T)
    control = .!treated

    n_t = count(treated)
    n_c = count(control)
    Ec = @view E1[control]
    Et = @view E1[treated]

    n_c_unexp = count(iszero, Ec)
    n_t_unexp = count(iszero, Et)
    var_c = n_c > 1 ? T(var(Ec)) : zero(T)
    mean_t = n_t > 0 ? T(mean(Et)) : zero(T)
    mean_c = n_c > 0 ? T(mean(Ec)) : zero(T)

    sD = std(D)
    sE = std(E1)
    cor_DE = (sD > 0 && sE > 0) ? T(cor(D, E1)) : zero(T)

    identified = true
    msg = "The spillover is identified: $(n_c_unexp) of $(n_c) controls have zero exposure, " *
          "so there is an unexposed baseline to compare against."

    if n_c == 0
        identified = false
        msg = "Every unit is treated, so there is no control group."
    elseif var_c <= var_tol
        identified = false
        if n_c_unexp == n_c
            msg = "No control unit has a treated neighbour, so exposure is identically zero " *
                  "among the controls. The spillover coefficient then rests entirely on " *
                  "variation among the treated, where it cannot be separated from the direct " *
                  "effect. Widen `W`, or use a distance band that reaches the controls."
        else
            msg = "Exposure is (near-)constant among the controls, variance $(var_c). The " *
                  "spillover coefficient is absorbed by the intercept and is not identified. " *
                  "This happens when `W` connects every control to a treated unit; a narrower " *
                  "`W`, or a ring that excludes the immediate neighbours, restores a baseline."
        end
    elseif n_c_unexp == 0
        msg = "No control unit is unexposed: all $(n_c) have a treated neighbour. The " *
              "spillover is still identified, by controls differing in *how* exposed they " *
              "are, but it is the more-versus-less-exposed contrast, not " *
              "exposed-versus-not. Read the reported spillover against the exposure means, " *
              "not as an absolute effect."
    end

    return InterferenceReport{T}(n_t, n_c, n_c_unexp, n_t_unexp, var_c, cor_DE,
                                 mean_t, mean_c, identified, msg)
end

function Base.show(io::IO, r::InterferenceReport)
    println(io, "Interference / exposure design report")
    println(io, "═"^62)
    @printf(io, "  treated %d,  control %d\n", r.n_treated, r.n_control)
    @printf(io, "  controls with zero exposure : %d\n", r.n_control_unexposed)
    @printf(io, "  treated with zero exposure  : %d\n", r.n_treated_unexposed)
    @printf(io, "  var(exposure | control)     : %.6g\n", r.exposure_var_control)
    @printf(io, "  cor(D, exposure)            : %.4f\n", r.exposure_cor)
    @printf(io, "  mean exposure, treated      : %.4f\n", r.mean_exposure_treated)
    @printf(io, "  mean exposure, control      : %.4f\n", r.mean_exposure_control)
    println(io, "─"^62)
    println(io, r.identified ? "  identified: yes" : "  identified: NO")
    for line in _wrap(r.message, 58)
        println(io, "    ", line)
    end
    print(io, "═"^62)
end

# Wrap a message to a width, so a multi-sentence diagnostic stays readable
# in a terminal rather than running off the edge.
function _wrap(s::AbstractString, width::Int)
    words = split(s)
    isempty(words) && return String[]
    lines = String[]
    cur = String(words[1])
    for w in words[2:end]
        if length(cur) + 1 + length(w) <= width
            cur = cur * " " * w
        else
            push!(lines, cur)
            cur = String(w)
        end
    end
    push!(lines, cur)
    return lines
end

# ═══════════════════════════════════════════════════════════════════════
# Spatial Difference-in-Differences
# ═══════════════════════════════════════════════════════════════════════

"""
    SpatialDiDResult

A spatial difference-in-differences fit, returned by
[`fit_spatial_did`](@ref).

# The four effects, and why they are reported apart

| Field | Meaning |
|---|---|
| `att_direct` | effect on a treated unit of its *own* treatment |
| `att_spill_treated` | effect on a treated unit of its neighbours' treatment |
| `att_spill_control` | effect on a control unit of its neighbours' treatment |
| `att_total_treated` | `att_direct + att_spill_treated` |

The two spillovers are evaluated at different exposure means and are not
interchangeable. `att_spill_control` is the policy's reach beyond its
target — the quantity a spillover design exists to measure.
`att_spill_treated` is part of what a naive DiD would report *as* the
direct effect, which is the bias this estimator removes.

Each is a fixed linear combination of the coefficient vector, so the
standard errors are exact rather than approximations: `Var(a'θ̂) = a'Σa`.

# Fields
- `att_direct`, `att_spill_treated`, `att_spill_control`,
  `att_total_treated` and their `*_se` counterparts.
- `coef`, `vcov`, `coefnames`, `residuals`: the underlying regression.
- `exposure_map`: the [`ExposureMap`](@ref) the spillovers are relative to.
- `interference`: the [`InterferenceReport`](@ref) for that design.
- `vcov_name`: which covariance estimator produced `vcov`.
- `design`: `:two_period` or `:staggered`.
- `n_units`, `n_periods`.
"""
struct SpatialDiDResult{T<:AbstractFloat} <: StatsAPI.RegressionModel
    att_direct::T
    att_direct_se::T
    att_spill_treated::T
    att_spill_treated_se::T
    att_spill_control::T
    att_spill_control_se::T
    att_total_treated::T
    att_total_treated_se::T
    coef::Vector{T}
    vcov::Matrix{T}
    coefnames::Vector{String}
    residuals::Vector{T}
    exposure_map::ExposureMap{T}
    interference::InterferenceReport{T}
    vcov_name::String
    design::Symbol
    n_units::Int
    n_periods::Int
end

StatsAPI.coef(m::SpatialDiDResult) = m.coef
StatsAPI.vcov(m::SpatialDiDResult) = m.vcov
StatsAPI.stderror(m::SpatialDiDResult) = sqrt.(max.(zero(eltype(m.vcov)), diag(m.vcov)))
StatsAPI.nobs(m::SpatialDiDResult) = length(m.residuals)
StatsAPI.residuals(m::SpatialDiDResult) = m.residuals
StatsAPI.coefnames(m::SpatialDiDResult) = m.coefnames
StatsAPI.dof_residual(m::SpatialDiDResult) = length(m.residuals) - length(m.coef)

"""
    fit_spatial_did(y, treat, W; X = nothing, vcov = :cluster, coords = nothing,
                    dist_cutoff = nothing, order = 1, standardize = true,
                    cluster_groups = nothing)

Difference-in-differences with spatial spillovers: decompose a policy
effect into what it did to the treated and what it did to their
neighbours.

# The estimator

With `D` the treatment, `E = W D` the exposure and `post` the policy
period, the design is

```
y_it = μ_i + γ_t + τ·D_i·post_t + δ·E_i·post_t + ε_it
```

For two periods, first-differencing removes `μ_i` exactly and leaves an
OLS problem in `[1, D, E, ΔX]`, whose coefficients on `D` and `E` are `τ`
and `δ`. For more than two, a two-way within transform does the same job
with `D_it` and `E_it` varying over time.

A conventional DiD omits `E`, which does not make the spillover zero — it
pushes it into the error, where it is correlated with `D` by construction,
and into the control group, which is no longer untreated. Both biases run
through `τ̂`.

# Arguments
- `y`: `N x T` outcomes, units by periods.
- `treat`: either a length-`N` indicator (two periods, treatment switching
  on in the second) or an `N x T` `Bool`/0-1 matrix giving the full
  treatment path, which is how a staggered rollout is expressed.
- `W`: `N x N` sparse weight matrix defining exposure.

# Keywords
- `X`: optional `N x T x K` array, or `N x K` matrix for time-invariant
  covariates, of controls.
- `vcov`: `:cluster` (default), `:conley`, `:ols`, or an
  `AbstractRobustVCOV` instance. See below — the default is measured, not
  conventional.
- `coords`, `dist_cutoff`: required by `:conley`.
- `cluster_groups`: required by `:cluster`; unit index is the usual choice.
- `order`, `standardize`: passed to [`exposure`](@ref).

# Which covariance, and why the default is not the spatial one

Coverage of a nominal 95% interval on `att_direct`, 2,000 replications,
`N = 144` on a rook lattice with a treated block, errors generated as
`(I - 0.6W)^{-1}ε`. `SE/MCsd` compares the mean reported standard error
with the Monte Carlo standard deviation of the estimate, which is the
quantity it is trying to estimate:

| Estimator | Coverage | SE/MCsd |
|---|---|---|
| `:cluster` (on unit) | 95.0% | 1.025 |
| `:ols` | 96.5% | 1.068 |
| `:conley`, cutoff 1.5 | 94.5% | 1.005 |
| `:conley`, cutoff 2.5 | 92.5% | 0.948 |
| `:conley`, cutoff 4.0 | 90.5% | 0.893 |
| `:conley`, cutoff 6.0 | 85.3% | 0.785 |

`:cluster` is the default on that evidence, not on convention. The ordering
held across nine configurations — treated block, scattered, half-grid and
alternating stripes, at `ρ = 0`, `0.6` and `0.9`, at `N = 144` and `400`:
`:cluster` stayed within `93.0–97.0%` and Conley fell monotonically as the
cutoff widened, reaching `82%` at the widest.

**Conley degrades identically when there is no spatial correlation at
all** (`93.0% → 83.9%` across the same cutoffs at `ρ = 0`), which is the
diagnostic that settles what is happening: the decay is the finite-sample
cost of estimating more pair covariances, not the correction working. It is
not the PSD projection either — `psd_fix = false` reproduces every figure
to three decimals, because the meat is already positive semi-definite here.

**Why the spatial correction matters less than it should.** The exposure
column is what does it. Including `WD` means `τ` is identified off the
contrast between a unit and its neighbours, and that contrast differences
out the component of the error they share — which is precisely the
spatially correlated part. In these designs `cor(D, WD)` runs from `0.93`
to `0.99`, and `OLS`'s `SE/MCsd` barely moves between `ρ = 0` and
`ρ = 0.9` (`1.010` to `1.207`). In the one design where `cor(D, WD) = 0`
exactly — alternating stripes — the spatial correlation is orthogonal to
the identifying contrast and every estimator over-covers.

So a spatial HAC is still the right instrument for a *spatially correlated
regressor*, and `:conley` is here for designs where the exposure column
does not absorb the dependence. But reaching for it reflexively, at a
generous cutoff, makes inference worse than doing nothing. If you use it,
report more than one cutoff, and prefer the narrow end.

!!! note "What clustering on unit is actually doing here"
    In the two-period design the regression is first-differenced, so there
    is one observation per unit and each cluster is a singleton: `:cluster`
    reduces to a heteroskedasticity-robust (HC1) covariance. That is why it
    performs well — not because it is absorbing serial or spatial
    correlation, neither of which survives the differencing. In a staggered
    design with `T > 2` the clusters are genuine and it absorbs
    within-unit serial correlation as usual.

# Examples
```julia
m = fit_spatial_did(y, D, W; coords = coords, dist_cutoff = 3.0)
m.att_direct, m.att_spill_control

# staggered rollout
m = fit_spatial_did(y, D_path, W; coords = coords, dist_cutoff = 3.0)

# the conventional standard errors, chosen deliberately
m = fit_spatial_did(y, D, W; vcov = :cluster, cluster_groups = 1:N)
```

!!! warning "Staggered adoption with heterogeneous effects"
    The two-way fixed effects estimator used when `T > 2` does not recover
    the sample-weighted average treatment effect once effects differ across
    adoption cohorts: it applies its own implicit weights, some of them
    negative (Goodman-Bacon). Measured on three-cohort rollouts over eight
    periods, `N = 100`, no noise, so every gap below is bias and not
    sampling error:

    | Cohort layout | Effect profile | True ATT | `att_direct` |
    |---|---|---|---|
    | contiguous bands | constant per cohort | 1.667 | 2.781 |
    | contiguous bands | growing with exposure | 4.167 | **1.805** |
    | interleaved | constant per cohort | 1.662 | 2.000 |
    | interleaved | growing with exposure | 4.155 | 3.440 |

    The error runs from 17% to 67% and **its sign is not predictable** —
    too high with constant effects, too low with growing ones.

    The spatial dimension makes this worse rather than better, and that is
    specific to this estimator. When the adoption cohorts are spatially
    contiguous — which is how rollouts actually happen — cohort identity is
    nearly collinear with the exposure column, so `WD` absorbs
    cohort-specific heterogeneity and `att_direct` moves to compensate:
    `1.805` against a true `4.167`. With the cohorts spatially interleaved
    the same profile gives `3.440`, and with no exposure column at all,
    `3.600`.

    The estimate did stay inside the convex hull of the realised effects in
    all four designs, but that hull spans `[1.0, 8.0]` in the dynamic cases,
    so it is a weak guarantee rather than a reassuring one.

    For a staggered design with suspected heterogeneity, estimate cohorts
    separately and aggregate deliberately. With homogeneous effects this
    path is exact — testset 34d pins `τ` and `δ` to `5e-15` on a noiseless
    staggered design — so the caveat is about heterogeneity alone, not
    about the implementation.

# References
Delgado, M. & Florax, R. (2015). Difference-in-differences techniques for
spatial spillovers. *Economics Letters*, 137.
Chagas, A., Azzoni, C. & Almeida, A. (2016). A spatial
difference-in-differences analysis of the impact of sugarcane production.
*Regional Science and Urban Economics*, 59.
Goodman-Bacon, A. (2021). Difference-in-differences with variation in
treatment timing. *Journal of Econometrics*, 225(2).
"""
function fit_spatial_did(y::AbstractMatrix, treat, W::AbstractSparseMatrix;
                         X=nothing,
                         vcov::Union{Symbol, AbstractRobustVCOV}=:cluster,
                         coords=nothing,
                         dist_cutoff=nothing,
                         order::Int=1,
                         standardize::Bool=true,
                         cluster_groups=nothing)
    N, T_per = size(y)
    T_el = promote_type(Float64, eltype(y) <: AbstractFloat ? eltype(y) : Float64)
    T_per >= 2 || throw(ArgumentError("difference-in-differences needs at least 2 periods, got $T_per"))

    Dpath, D, design = _treatment_path(treat, N, T_per)
    em = exposure(D, W; order=order, standardize=standardize)
    rep = check_interference(em)

    if design === :two_period
        Z, dep, names, E_used, treated_mask = _build_two_period(T_el, y, em, X)
        unit_of_row = collect(1:N)
    else
        Z, dep, names, unit_of_row, E_used, treated_mask =
            _build_staggered(T_el, y, Dpath, em, W, X, standardize)
    end

    check_design_rank(Z, names, "fit_spatial_did";
                      context="This is usually the exposure column: " * rep.message)

    F = qr(Z)
    θ = F \ dep
    resid = dep - Z * θ

    Σ, vname = _did_vcov(vcov, Z, resid, coords, dist_cutoff, cluster_groups,
                         unit_of_row, N, design, T_per)

    # The four effects are each a'θ for a fixed `a`, so their variances are
    # exact. `a` places the exposure means on the exposure coefficients and
    # 1 on the direct one.
    i_dir, i_exp = _effect_slots(design, order)
    Ē_T, Ē_C = _exposure_means(E_used, treated_mask)

    a_dir = zeros(T_el, length(θ)); a_dir[i_dir] = one(T_el)
    a_spt = zeros(T_el, length(θ)); a_spc = zeros(T_el, length(θ))
    for k in 1:order
        a_spt[i_exp[k]] = Ē_T[k]
        a_spc[i_exp[k]] = Ē_C[k]
    end
    a_tot = a_dir .+ a_spt

    q(a) = sqrt(max(zero(T_el), dot(a, Σ, a)))

    return SpatialDiDResult{T_el}(
        dot(a_dir, θ), q(a_dir),
        dot(a_spt, θ), q(a_spt),
        dot(a_spc, θ), q(a_spc),
        dot(a_tot, θ), q(a_tot),
        θ, Matrix(Σ), names, resid, em, rep, vname, design, N, T_per)
end

# ─── Treatment path ──────────────────────────────────────────────────
#
# A length-N indicator means "two periods, switching on in the second".
# An N x T matrix is the general form and is what a staggered rollout
# needs, so the vector case is normalised into it rather than handled
# separately downstream.

function _treatment_path(treat::AbstractVector, N::Int, T_per::Int)
    length(treat) == N ||
        throw(DimensionMismatch("`treat` has $(length(treat)) entries but `y` has $N rows"))
    T_per == 2 || throw(ArgumentError(
        "a length-$N treatment vector describes a two-period design, but `y` has $T_per " *
        "periods. Pass an $N x $T_per treatment matrix to describe when each unit is treated."))
    D = [Float64(t) for t in treat]
    path = hcat(zeros(Float64, N), D)
    return path, D, :two_period
end

function _treatment_path(treat::AbstractMatrix, N::Int, T_per::Int)
    size(treat) == (N, T_per) ||
        throw(DimensionMismatch("`treat` is $(size(treat)) but `y` is $(N)x$(T_per)"))
    path = [Float64(t) for t in treat]
    for v in path
        (v == 0.0 || v == 1.0) ||
            throw(ArgumentError("`treat` must hold only 0 and 1; found $v"))
    end
    # "Ever treated" is the cross-sectional indicator the exposure is built
    # from: a unit's neighbours spill over onto it whether or not it is
    # itself treated yet.
    D = [Float64(any(!iszero, @view path[i, :])) for i in 1:N]

    if T_per == 2 && all(iszero, @view path[:, 1])
        return path, D, :two_period
    end
    return path, D, :staggered
end

# ─── Two-period design: first differences ────────────────────────────
#
# Differencing removes the unit effect exactly, which is why this path is
# exact rather than approximate: with no noise it returns τ and δ to
# machine precision.

function _build_two_period(::Type{T}, y::AbstractMatrix, em::ExposureMap, X) where {T}
    N = size(y, 1)
    Δy = T.(@view y[:, 2]) .- T.(@view y[:, 1])

    cols = Any[ones(T, N), em.D]
    names = String["(Intercept)", "D"]
    for k in 1:em.order
        push!(cols, @view em.E[:, k])
        push!(names, em.labels[k])
    end

    if X !== nothing
        ΔX, xn = _difference_covariates(T, X, N)
        for j in axes(ΔX, 2)
            push!(cols, @view ΔX[:, j])
            push!(names, xn[j])
        end
    end

    Z = Matrix{T}(undef, N, length(cols))
    for (j, c) in enumerate(cols)
        Z[:, j] = c
    end
    return Z, Δy, names, em.E, em.D .== one(T)
end

# A time-invariant covariate differences to zero and cannot enter a
# first-differenced equation. Saying so beats silently dropping it or
# producing a rank-deficient design the user has to decode.
function _difference_covariates(::Type{T}, X, N::Int) where {T}
    if ndims(X) == 3
        size(X, 1) == N || throw(DimensionMismatch("`X` has $(size(X, 1)) units but `y` has $N"))
        size(X, 3) == 2 || throw(ArgumentError("a two-period design needs `X` with 2 periods, got $(size(X, 3))"))
        K = size(X, 2)
        ΔX = Matrix{T}(undef, N, K)
        for k in 1:K
            ΔX[:, k] = T.(@view X[:, k, 2]) .- T.(@view X[:, k, 1])
        end
        return ΔX, ["ΔX$k" for k in 1:K]
    elseif ndims(X) == 2
        throw(ArgumentError(
            "`X` is an $(size(X, 1))x$(size(X, 2)) matrix, which a first-differenced " *
            "two-period design reads as time-invariant — and a time-invariant covariate " *
            "differences to exactly zero, so it cannot be included. Pass an N x K x 2 array " *
            "if the covariate varies over time, or omit it."))
    else
        throw(ArgumentError("`X` must be an N x K x T array, got $(ndims(X)) dimensions"))
    end
end

# ─── Staggered design: two-way within transform ──────────────────────

function _build_staggered(::Type{T}, y::AbstractMatrix, Dpath::Matrix{Float64},
                          em::ExposureMap, W::AbstractSparseMatrix, X,
                          standardize::Bool) where {T}
    N, T_per = size(y)
    Wt = standardize ? row_standardize(T.(W)) : T.(W)

    # Exposure now varies over time: E_it is built from the treatment
    # actually in force at t, not from "ever treated".
    Epath = Array{T, 3}(undef, N, em.order, T_per)
    for t in 1:T_per
        col = Wt * T.(@view Dpath[:, t])
        Epath[:, 1, t] = col
        for k in 2:em.order
            col = Wt * col
            Epath[:, k, t] = col
        end
    end

    NT = N * T_per
    ncol = 1 + em.order + (X === nothing ? 0 : _covariate_count(X, N, T_per))
    Zr = Matrix{T}(undef, NT, ncol)
    yv = Vector{T}(undef, NT)
    unit_of_row = Vector{Int}(undef, NT)

    # Stacked units-fastest within period, which is the layout
    # `vcov_conley` and `vcov_driscoll_kraay` expect.
    for t in 1:T_per, i in 1:N
        r = (t - 1) * N + i
        yv[r] = T(y[i, t])
        Zr[r, 1] = T(Dpath[i, t])
        for k in 1:em.order
            Zr[r, 1 + k] = Epath[i, k, t]
        end
        unit_of_row[r] = i
    end
    names = String["D"]
    append!(names, em.labels)

    if X !== nothing
        K = _covariate_count(X, N, T_per)
        for k in 1:K, t in 1:T_per, i in 1:N
            Zr[(t - 1) * N + i, 1 + em.order + k] = T(X[i, k, t])
        end
        append!(names, ["X$k" for k in 1:K])
    end

    # The reported spillover is δ times a mean exposure *in levels*, so the
    # means have to come from the exposure before the within transform
    # touches it. Taken here rather than recomputed from `em.E`, which
    # holds the static ever-treated exposure and would not match the
    # time-varying regressor the coefficient belongs to.
    E_used = Zr[:, 2:(1 + em.order)]
    treated_mask = Zr[:, 1] .== one(T)

    # Two-way demeaning: unit means, period means, grand mean added back.
    # This absorbs μ_i and γ_t exactly, so no intercept column is needed
    # and none is added -- it would be collinear with the transform.
    _twoway_demean!(yv, N, T_per)
    for j in axes(Zr, 2)
        _twoway_demean!(@view(Zr[:, j]), N, T_per)
    end

    return Zr, yv, names, unit_of_row, E_used, treated_mask
end

function _covariate_count(X, N::Int, T_per::Int)
    ndims(X) == 3 || throw(ArgumentError("a staggered design needs `X` as an N x K x T array, got $(ndims(X)) dimensions"))
    size(X, 1) == N && size(X, 3) == T_per ||
        throw(DimensionMismatch("`X` is $(size(X)) but `y` is $(N)x$(T_per)"))
    return size(X, 2)
end

function _twoway_demean!(v::AbstractVector{T}, N::Int, T_per::Int) where {T}
    grand = zero(T)
    @inbounds for r in eachindex(v); grand += v[r]; end
    grand /= T(length(v))

    unit_m = zeros(T, N)
    per_m = zeros(T, T_per)
    @inbounds for t in 1:T_per
        off = (t - 1) * N
        s = zero(T)
        for i in 1:N
            x = v[off + i]
            unit_m[i] += x
            s += x
        end
        per_m[t] = s / T(N)
    end
    @inbounds for i in 1:N; unit_m[i] /= T(T_per); end

    @inbounds for t in 1:T_per
        off = (t - 1) * N
        for i in 1:N
            v[off + i] -= unit_m[i] + per_m[t] - grand
        end
    end
    return v
end

# ─── Effect slots and exposure means ─────────────────────────────────

# Where `D` and the exposure columns sit in θ. The two-period design
# carries an intercept, the staggered one does not (the within transform
# absorbed it), so the offset differs by exactly one.
function _effect_slots(design::Symbol, order::Int)
    off = design === :two_period ? 1 : 0
    return off + 1, [off + 1 + k for k in 1:order]
end

"""
    _exposure_means(E_used, treated_mask)

Mean exposure among treated and among untreated rows of the regression
that was actually run.

`E_used` is the exposure in levels, in the same row order as the design,
and `treated_mask` marks the treated rows: units in the two-period design,
unit-periods in the staggered one. Taking both from the builder is what
keeps the reported spillover on the same footing as the coefficient it
multiplies — the staggered design's exposure varies over time, and
averaging the static ever-treated exposure instead would scale `δ̂` by the
wrong number.
"""
function _exposure_means(E_used::AbstractMatrix{T}, treated_mask::AbstractVector{Bool}) where {T}
    order = size(E_used, 2)
    Ē_T = zeros(T, order)
    Ē_C = zeros(T, order)
    control_mask = .!treated_mask
    any_t = any(treated_mask)
    any_c = any(control_mask)
    for k in 1:order
        col = @view E_used[:, k]
        Ē_T[k] = any_t ? T(mean(@view col[treated_mask])) : zero(T)
        Ē_C[k] = any_c ? T(mean(@view col[control_mask])) : zero(T)
    end
    return Ē_T, Ē_C
end

# ─── Covariance ──────────────────────────────────────────────────────

function _did_vcov(spec, Z::Matrix{T}, resid::Vector{T},
                   coords, dist_cutoff, cluster_groups, unit_of_row::Vector{Int},
                   N::Int, design::Symbol, T_per::Int) where {T}
    n, K = size(Z)
    bread = inv(Symmetric(Z' * Z))
    n_eff_periods = design === :two_period ? 1 : T_per

    if spec isa AbstractRobustVCOV
        return _apply_robust(spec, Z, resid, bread, N, n_eff_periods), _vcov_label(spec)
    elseif spec === :ols
        s2 = dot(resid, resid) / T(max(1, n - K))
        return Matrix(Symmetric(s2 .* bread)), "OLS (homoskedastic)"
    elseif spec === :conley
        coords === nothing && throw(ArgumentError(
            "`vcov = :conley` needs coordinates: pass `coords` (an N x 2 matrix) and " *
            "`dist_cutoff`."))
        dist_cutoff === nothing && throw(ArgumentError(
            "`vcov = :conley` needs `dist_cutoff`, the distance beyond which units are " *
            "treated as independent. It is a judgement call and the result is sensitive " *
            "to it, so there is no default; report a couple of cutoffs."))
        size(coords, 1) == N || throw(DimensionMismatch(
            "`coords` has $(size(coords, 1)) rows but there are $N units"))
        cspec = ConleySpatialHAC(Matrix{T}(coords), T(dist_cutoff);
                                 max_time_lag=(n_eff_periods > 1 ? 1 : 0))
        return vcov_conley(Z, resid, N, n_eff_periods, cspec; bread_inv=bread),
               "Conley spatial HAC (cutoff $(dist_cutoff))"
    elseif spec === :cluster
        groups = cluster_groups === nothing ? unit_of_row : collect(cluster_groups)
        length(groups) == n || throw(DimensionMismatch(
            "`cluster_groups` has $(length(groups)) entries but the regression has $n rows" *
            (design === :two_period ?
             " (one per unit, because the two-period design is first-differenced)" : "")))
        cspec = ClusterRobustVCOV(groups)
        return vcov_cluster(Z, resid, cspec; bread_inv=bread),
               "cluster-robust ($(length(unique(groups))) groups)"
    else
        throw(ArgumentError("unknown `vcov = :$spec`; expected :conley, :cluster, :ols, " *
                            "or an AbstractRobustVCOV instance"))
    end
end

function _apply_robust(spec::ConleySpatialHAC, Z, resid, bread, N, T_per)
    return vcov_conley(Z, resid, N, T_per, spec; bread_inv=bread)
end
function _apply_robust(spec::ClusterRobustVCOV, Z, resid, bread, N, T_per)
    return vcov_cluster(Z, resid, spec; bread_inv=bread)
end
function _apply_robust(spec::DriscollKraayVCOV, Z, resid, bread, N, T_per)
    T_per > 1 || throw(ArgumentError(
        "Driscoll-Kraay averages scores across units within each period and corrects over " *
        "periods, so it needs more than one. A first-differenced two-period design has one; " *
        "use `:conley` or `:cluster`."))
    return vcov_driscoll_kraay(Z, resid, N, T_per; bread_inv=bread, spec=spec)
end

_vcov_label(::ConleySpatialHAC) = "Conley spatial HAC"
_vcov_label(s::ClusterRobustVCOV) = "cluster-robust ($(length(unique(s.groups))) groups)"
_vcov_label(::DriscollKraayVCOV) = "Driscoll-Kraay"

# ─── Display ─────────────────────────────────────────────────────────

function Base.show(io::IO, m::SpatialDiDResult{T}) where {T}
    dname = m.design === :two_period ? "two-period" : "staggered"
    println(io, "Spatial Difference-in-Differences ($dname)")
    println(io, "═"^68)
    @printf(io, "  N = %d units, T = %d periods,  %d treated\n",
            m.n_units, m.n_periods, m.interference.n_treated)
    println(io, "  exposure: ", join(m.exposure_map.labels, ", "),
            m.exposure_map.standardized ? " (row-standardized W)" : " (unstandardized W)")
    println(io, "  vcov: ", m.vcov_name)
    m.interference.identified || println(io, "  ⚠ spillover NOT identified — see `.interference`")
    println(io, "")
    println(io, "  Treatment effects")
    println(io, "  " * "─"^66)
    @printf(io, "  %-28s %12s %12s %9s\n", "", "Estimate", "Std. Error", "z")
    for (lab, v, s) in (("direct (own treatment)", m.att_direct, m.att_direct_se),
                        ("spillover, on treated", m.att_spill_treated, m.att_spill_treated_se),
                        ("spillover, on controls", m.att_spill_control, m.att_spill_control_se),
                        ("total, on treated", m.att_total_treated, m.att_total_treated_se))
        z = s > 0 ? v / s : T(NaN)
        @printf(io, "  %-28s %12.4f %12.4f %9.2f\n", lab, v, s, z)
    end
    println(io, "")
    println(io, "  Underlying regression")
    println(io, "  " * "─"^66)
    _print_coef_table(io, m.coefnames, m.coef, StatsAPI.stderror(m))
end

end # module
