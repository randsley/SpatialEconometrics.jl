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
import ..SpatialHAC: DistanceMetric, EuclideanDistance, HaversineDistance,
       pairwise_distance
import Random
import Random: AbstractRNG

export ExposureMap, exposure, ring_exposure,
       InterferenceReport, check_interference,
       SpatialDiDResult, fit_spatial_did,
       SyntheticControlResult, fit_synthetic_control, donor_weights,
       PlaceboTest, placebo_test,
       SpilloverGaps, spillover_gaps

# ═══════════════════════════════════════════════════════════════════════
# Exposure Mapping
# ═══════════════════════════════════════════════════════════════════════

"""
    ExposureMap

A treatment vector together with its spatial exposure columns, returned by
[`exposure`](@ref) and [`ring_exposure`](@ref).

An exposure mapping is the object that makes a spillover estimable. SUTVA
assumes unit `i`'s outcome responds only to `D_i`. An exposure mapping
replaces that with a named, finite summary of the treatment *around* `i`.
The spillover parameter is then defined relative to that summary, rather
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
- `D`: treatment indicator, length `N`. `Bool`, integer or float. Anything
  other than 0 and 1 is rejected, because a fractional "treatment" makes
  the direct/spillover split ill-defined.
- `W`: `N x N` sparse weight matrix.

# Keywords
- `order`: number of spatial orders. `1` gives `WD` alone. `2` adds `W²D`,
  which lets a second ring differ from the first, instead of being forced
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
and its displacement are mixed together. Starting the band beyond them
reads the effect at a distance where mechanical displacement is
implausible.

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
baseline, and the constant soaks up `δ`. The fit still returns a number,
but it is the difference between *more* and *less* exposed, not between
exposed and not. If no control has a treated neighbour, `δ` rests
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
    fit_spatial_did(y, treat, W; X = nothing, vcov = :cluster,
                    coords = nothing,
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

A conventional DiD omits `E`, which does not make the spillover zero. It
pushes the spillover into the error, where it is correlated with `D` by
construction, and into the control group, which is no longer untreated.
Both biases run through `τ̂`.

# Which spatial DiD this is

This is **Delgado and Florax (2015), their eq. (6)**, read in the paper:
`y = β₀ + β₁D + β₂T + β₃D·T + β₄(WD)·T + ε`. Their derivation carries
the restriction `β₄ = λβ₃`, which this estimator does **not** impose —
`D` and `E` enter as free columns — so their restricted case is nested
rather than assumed. Their `ATE(wd) = β₃ + β₄·wd` is what
`att_total_treated` reports.

`att_spill_treated` and `att_spill_control` come from **one** exposure
coefficient per order, scaled by the mean exposure of the treated and of
the controls. The two differ because the groups have different average
exposure, not because two coefficients were estimated.

**Chagas, Azzoni and Almeida (2016) eq. (7) is a different estimator and
is not implemented here.** They split exposure into two channels with
separate free coefficients — `W_{T,T}b₁` from treated onto treated and
`W_{NT,T}b₂` from treated onto controls — and their footnote 4 states
that "Delgado and Florax (2015) consider only the restricted case".
That generalisation is a roadmap row, not a feature of this function.
Cite that paper here for the applied design and for the two-channel
idea, not for the estimator this code fits.

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
- `cluster_groups`: required by `:cluster`. Unit index is the usual choice.
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
held across nine configurations: treated block, scattered, half-grid and
alternating stripes, at `ρ = 0`, `0.6` and `0.9`, at `N = 144` and `400`.
`:cluster` stayed within `93.0–97.0%`. Conley fell monotonically as the
cutoff widened, reaching `82%` at the widest.

**Conley degrades identically when there is no spatial correlation at
all** (`93.0% → 83.9%` across the same cutoffs at `ρ = 0`). That is the
diagnostic that settles what is happening. The decay is the finite-sample
cost of estimating more pair covariances, not the correction working. It is
not the PSD projection either — `psd_fix = false` reproduces every figure
to three decimals, because the meat is already positive semi-definite here.

**Why the spatial correction matters less than it should.** The exposure
column is what does it. Including `WD` identifies `τ` off the contrast
between a unit and its neighbours. That contrast differences out the
component of the error they share, which is precisely the spatially
correlated part. In these designs `cor(D, WD)` runs from `0.93`
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
    is one observation per unit and each cluster is a singleton. `:cluster`
    then reduces to a heteroskedasticity-robust (HC1) covariance. That is
    why it performs well. It is not absorbing serial or spatial
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
    adoption cohorts. It applies its own implicit weights, some of them
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
Delgado, M. S. & Florax, R. J. G. M. (2015). Difference-in-differences
techniques for spatial data: local autocorrelation and spatial
interaction. *Economics Letters*, 137, 123–126. **Read**, from Tinbergen
Institute Discussion Paper 2015-091/VIII. Eq. (6) is the design above.
Chagas, A. L. S., Azzoni, C. R. & Almeida, A. N. (2016). A spatial
difference-in-differences analysis of the impact of sugarcane production
on respiratory diseases. *Regional Science and Urban Economics*, 59,
24–36. **Read.** Its eq. (7), the unrestricted two-channel SDID, is
**not** what this function fits; see above.
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
multiplies. The staggered design's exposure varies over time. Averaging
the static ever-treated exposure instead would scale `δ̂` by the wrong
number.
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


# ═══════════════════════════════════════════════════════════════════════
# Synthetic Control
# ═══════════════════════════════════════════════════════════════════════

# ─── Projection onto the unit simplex ────────────────────────────────

# Euclidean projection of `w` onto {w >= 0, sum(w) == 1}, in place.
#
# The projection subtracts one common threshold θ from every coordinate
# and clips at zero, so the only unknown is θ. Sorting descending and
# accumulating finds the size of the support, and θ follows from the
# requirement that the surviving coordinates sum to one. O(J log J),
# exact up to floating point, and no inner iteration — which matters
# because the projected-gradient loop below calls it every step.
function _project_simplex!(w::AbstractVector{T}) where {T<:AbstractFloat}
    J = length(w)
    J == 0 && throw(ArgumentError("_project_simplex!: empty weight vector."))
    if J == 1
        w[1] = one(T)
        return w
    end
    u = sort(w; rev = true)
    css = zero(T)
    θ = zero(T)
    @inbounds for j in 1:J
        css += u[j]
        tj = (css - one(T)) / T(j)
        if u[j] - tj > zero(T)
            θ = tj
        else
            break
        end
    end
    @inbounds for j in 1:J
        w[j] = max(w[j] - θ, zero(T))
    end
    return w
end

# ─── The penalised synthetic control program ─────────────────────────

# minimise  (x1 - X0 w)'(x1 - X0 w) + λ Σ_j d_j w_j   s.t.  w ≥ 0, Σ w = 1
#
# Convex quadratic over the unit simplex, solved by accelerated projected
# gradient (FISTA) with the exact Lipschitz constant L = 2 σ_max(X0'X0),
# so the step size is derived rather than tuned. FISTA is not monotone, so
# a plain projected-gradient polish follows. Its own KKT gap then certifies
# the returned point, rather than this code trusting it.
#
# Returns `(w, objective, iterations, kkt_gap, kkt_violation)`. At an
# optimum the gradient is equal across the support and no smaller off it,
# so both diagnostics are ~0. They are carried into the result instead of
# discarded: weights that did not converge look exactly like weights that
# did, unless something is reported.
function _solve_pensynth(X0::Matrix{T}, x1::Vector{T}, d::Vector{T}, λ::T;
                         tol::T = T(1e-14),
                         maxiter::Int = 50_000) where {T<:AbstractFloat}
    J = size(X0, 2)
    G = X0' * X0
    b = X0' * x1
    L = T(2) * opnorm(G)
    L > zero(T) || (L = one(T))
    step = one(T) / L

    w  = fill(one(T) / T(J), J)
    z  = copy(w)
    wp = copy(w)
    g  = similar(w)
    tk = one(T)
    iters = maxiter
    fw = _pensynth_obj(G, b, d, λ, w, x1)

    # The stopping rule is the KKT gap, not the step length. When the
    # donor pool is larger than the predictor set, which is the usual case,
    # `G` is rank deficient. The solution set is then a face rather than a
    # point, and the iterate keeps drifting along the flat directions long
    # after the objective has stopped moving. A step-length test never
    # fires there. The gradient condition is the quantity that actually
    # characterises the optimum.
    gscale = max(one(T), maximum(abs, b), λ * maximum(abs, d))
    check_every = 10

    for k in 1:maxiter
        mul!(g, G, z)
        @. g = T(2) * (g - b) + λ * d
        @. wp = z - step * g
        _project_simplex!(wp)

        fp = _pensynth_obj(G, b, d, λ, wp, x1)
        tk1 = T(0.5) * (one(T) + sqrt(one(T) + T(4) * tk^2))
        @. z = wp + ((tk - one(T)) / tk1) * (wp - w)
        # Adaptive restart (O'Donoghue & Candès). FISTA is not monotone,
        # and on an ill-conditioned problem the momentum overshoots and
        # oscillates. Dropping it when the objective genuinely rises
        # recovers the fast rate. The candidate is always accepted, because
        # it is the iterate sequence. The threshold is relative, since near
        # the optimum the objective moves by rounding alone, and an exact
        # comparison restarts forever.
        if fp > fw * (one(T) + T(1e-12)) + T(1e-300)
            tk1 = one(T)
            copyto!(z, wp)
        end
        copyto!(w, wp)
        fw = fp
        tk = tk1

        if k % check_every == 0 || k == maxiter
            kg, kv = _pensynth_kkt(G, b, d, λ, w, g)
            if max(kg, kv) <= tol * gscale
                iters = k
                break
            end
        end
    end

    # Monotone polish. Plain projected gradient from the final point
    # cannot increase the objective, so this only ever helps.
    for _ in 1:2_000
        mul!(g, G, w)
        @. g = T(2) * (g - b) + λ * d
        @. wp = w - step * g
        _project_simplex!(wp)
        if _pensynth_obj(G, b, d, λ, wp, x1) > fw
            break
        end
        if maximum(abs, wp .- w) < eps(T)
            copyto!(w, wp)
            break
        end
        copyto!(w, wp)
        fw = _pensynth_obj(G, b, d, λ, w, x1)
    end

    sw = sum(w)
    sw > zero(T) && (w ./= sw)

    kkt, viol = _pensynth_kkt(G, b, d, λ, w, g)
    f = sum(abs2, x1 .- X0 * w) + λ * dot(d, w)
    return w, f, iters, kkt / gscale, viol / gscale
end

# Objective  ||x1 - X0 w||^2 + λ d'w, formed through `G` and `b` so the
# inner loop does not touch `X0` every call.
function _pensynth_obj(G::Matrix{T}, b::Vector{T}, d::Vector{T}, λ::T,
                       w::Vector{T}, x1::Vector{T}) where {T<:AbstractFloat}
    return dot(w, G, w) - T(2) * dot(b, w) + dot(x1, x1) + λ * dot(d, w)
end

# The optimality certificate. At a constrained optimum the gradient is
# equal across the donors carrying weight and no smaller on those at
# zero, so `(gap, violation)` both vanish. `g` is used as scratch.
function _pensynth_kkt(G::Matrix{T}, b::Vector{T}, d::Vector{T}, λ::T,
                       w::Vector{T}, g::Vector{T}) where {T<:AbstractFloat}
    mul!(g, G, w)
    @. g = T(2) * (g - b) + λ * d
    sup = w .> T(1e-10)
    any(sup) || return zero(T), zero(T)
    gs_max = maximum(g[sup])
    gap = gs_max - minimum(g[sup])
    viol = all(sup) ? zero(T) : max(zero(T), gs_max - minimum(g[.!sup]))
    return gap, viol
end

# ─── Treated set ─────────────────────────────────────────────────────

function _treated_indices(treated::AbstractVector{Bool}, N::Int)
    length(treated) == N || throw(DimensionMismatch(
        "fit_synthetic_control: `treated` has $(length(treated)) entries but " *
        "`Y` has $N rows."))
    return findall(treated)
end

function _treated_indices(treated::AbstractVector{<:Real}, N::Int)
    if length(treated) == N && all(v -> v == 0 || v == 1, treated)
        return findall(!iszero, treated)
    end
    all(v -> v == round(v) && 1 <= v <= N, treated) || throw(ArgumentError(
        "fit_synthetic_control: `treated` is neither a length-$N 0/1 " *
        "indicator nor a vector of unit indices in 1:$N."))
    idx = Int.(treated)
    length(unique(idx)) == length(idx) || throw(ArgumentError(
        "fit_synthetic_control: `treated` repeats a unit index."))
    return idx
end

_treated_indices(treated::Integer, N::Int) =
    (1 <= treated <= N) ? [Int(treated)] : throw(ArgumentError(
        "fit_synthetic_control: treated unit $treated is outside 1:$N."))

# ─── Donor pool ──────────────────────────────────────────────────────

# Returns `(donors, excluded_neighbours)`. Treated units are never donors
# under any rule. A unit cannot help build its own counterfactual. And with
# several treated units, one treated unit's post-period outcome is
# contaminated by its own treatment.
function _donor_pool(rule, treated::Vector{Int}, N::Int,
                     W::Union{Nothing,AbstractSparseMatrix})
    is_treated = falses(N)
    is_treated[treated] .= true

    if rule isa AbstractVector
        cand = Int.(collect(rule))
        all(j -> 1 <= j <= N, cand) || throw(ArgumentError(
            "fit_synthetic_control: `donor_pool` contains an index outside 1:$N."))
        donors = [j for j in unique(cand) if !is_treated[j]]
        isempty(donors) && throw(ArgumentError(
            "fit_synthetic_control: the explicit `donor_pool` is empty once " *
            "treated units are removed."))
        return donors, Int[]
    end

    if rule === :all
        donors = [j for j in 1:N if !is_treated[j]]
        isempty(donors) && throw(ArgumentError(
            "fit_synthetic_control: every one of the $N units is treated, so " *
            "there is no donor to build a counterfactual from. A synthetic " *
            "control needs untreated units."))
        return donors, Int[]
    end

    rule === :exclude_neighbours || throw(ArgumentError(
        "fit_synthetic_control: `donor_pool` must be `:exclude_neighbours`, " *
        "`:all`, or a vector of unit indices; got `$rule`."))

    W === nothing && throw(ArgumentError(
        "fit_synthetic_control: `donor_pool = :exclude_neighbours` needs `W` " *
        "to know which units neighbour a treated one. Pass `W`, or set " *
        "`donor_pool = :all` to keep every untreated unit — but note that a " *
        "neighbour of a treated unit is contaminated by the spillover, so " *
        "`:all` builds the counterfactual partly out of treated territory."))

    # Reuses this module's own exposure mapping: with `D` the treated
    # indicator, `E = W̃D` is positive exactly on the units that neighbour
    # a treated one. No separate adjacency walk.
    D = zeros(Float64, N)
    D[treated] .= 1.0
    em = exposure(D, W; order = 1, standardize = true)
    E = view(em.E, :, 1)

    excluded = [j for j in 1:N if !is_treated[j] && E[j] > 0.0]
    donors   = [j for j in 1:N if !is_treated[j] && E[j] == 0.0]
    isempty(donors) && throw(ArgumentError(
        "fit_synthetic_control: every untreated unit neighbours a treated " *
        "one, so `donor_pool = :exclude_neighbours` leaves no donors — " *
        "$(length(excluded)) of $(N - length(treated)) untreated units were " *
        "excluded. The design cannot separate the policy from its own " *
        "spillover: either the treatment is too widespread for a synthetic " *
        "control, or `W` is too dense."))
    return donors, excluded
end

# ─── Predictors ──────────────────────────────────────────────────────

# Builds the `K x n` predictor design, units in columns. The default
# predictor set is the pre-treatment outcome path, which is the usual
# choice and the one Abadie (2021) recommends when the outcome is
# measured over a long pre-period.
#
# `standardize` scales each predictor row by its standard deviation across
# the units used. Without it, a predictor measured in large units dominates
# the fit for no substantive reason. This is the role `V` plays in the
# nested formulation. It is taken here as `V = I` on standardised
# predictors, so the program stays one convex problem with a reproducible
# solution.
function _scm_predictors(::Type{T}, Y::AbstractMatrix, predictors, T0::Int,
                         used::Vector{Int}, standardize::Bool) where {T}
    X = if predictors === nothing
        Matrix{T}(permutedims(T.(Y[used, 1:T0])))        # T0 x n
    else
        ndims(predictors) == 2 || throw(ArgumentError(
            "fit_synthetic_control: `predictors` must be an N x K matrix."))
        size(predictors, 1) == size(Y, 1) || throw(DimensionMismatch(
            "fit_synthetic_control: `predictors` has " *
            "$(size(predictors, 1)) rows but `Y` has $(size(Y, 1))."))
        Matrix{T}(permutedims(T.(predictors[used, :])))  # K x n
    end
    scales = ones(T, size(X, 1))
    if standardize
        @inbounds for k in axes(X, 1)
            s = std(view(X, k, :))
            if isfinite(s) && s > zero(T)
                scales[k] = s
                X[k, :] ./= s
            end
        end
    end
    return X, scales
end

# ─── Penalty distances ───────────────────────────────────────────────

# `d[q]` is the distance from treated unit `i` to donor `donors[q]`, which
# the penalty multiplies. Abadie & L'Hour (2021) penalise pairwise
# discrepancy to break the non-uniqueness of the unpenalised weights.
# `:spatial` makes that discrepancy geographic, which is what this package
# is for.
function _penalty_distances(::Type{T}, metric::Symbol, i::Int,
                            donors::Vector{Int}, coords, Xcols::Matrix{T},
                            col_of::Dict{Int,Int},
                            dmetric::DistanceMetric) where {T}
    J = length(donors)
    metric === :none && return zeros(T, J)

    if metric === :spatial
        coords === nothing && throw(ArgumentError(
            "fit_synthetic_control: `penalty_metric = :spatial` needs " *
            "`coords` (an N x 2 matrix) to measure distance from the treated " *
            "unit to each donor. Pass `coords`, or use " *
            "`penalty_metric = :covariate` to penalise predictor distance."))
        size(coords, 2) == 2 || throw(ArgumentError(
            "fit_synthetic_control: `coords` must have 2 columns."))
        d = Vector{T}(undef, J)
        @inbounds for (q, j) in enumerate(donors)
            d[q] = T(pairwise_distance(dmetric, T(coords[i, 1]), T(coords[i, 2]),
                                       T(coords[j, 1]), T(coords[j, 2])))
        end
        return d
    end

    if metric === :covariate
        d = Vector{T}(undef, J)
        xi = view(Xcols, :, col_of[i])
        @inbounds for (q, j) in enumerate(donors)
            d[q] = sqrt(sum(abs2, xi .- view(Xcols, :, col_of[j])))
        end
        return d
    end

    throw(ArgumentError(
        "fit_synthetic_control: `penalty_metric` must be `:spatial`, " *
        "`:covariate` or `:none`; got `$metric`."))
end

# ─── Result ──────────────────────────────────────────────────────────

"""
    SyntheticControlResult

The fitted synthetic control for one or more treated units, returned by
[`fit_synthetic_control`](@ref).

# Fields
- `weights`: `m x J` donor weights, one row per treated unit. Each row is
  non-negative and sums to one.
- `treated`, `donors`: unit indices. `donors` indexes the columns of
  `weights`.
- `excluded_neighbours`: untreated units kept out of the donor pool for
  neighbouring a treated unit. Empty unless `donor_pool =
  :exclude_neighbours` did some work.
- `donor_pool_rule`: which rule produced `donors`.
- `synthetic`, `gaps`: `m x T`. `gaps[k, t]` is treated unit `k`'s outcome
  minus its synthetic control at `t`, so the pre-period entries measure
  fit and the post-period entries are the estimated effect.
- `att_path`: length-`T` mean gap across treated units.
- `att`: mean of `att_path` over the post-treatment periods.
- `rmspe_pre`, `rmspe_post`, `ratio`: per treated unit. `ratio` is
  `rmspe_post / rmspe_pre`, the statistic the placebo test ranks.
- `ratio_aggregate`: the same ratio formed from `att_path`.
- `kkt_gap`, `kkt_violation`, `iterations`: solver diagnostics per treated
  unit. The two KKT numbers certify the weights are at the constrained
  optimum and should be ~0. See [`fit_synthetic_control`](@ref).

# Why there is no `vcov`

`coef`, `coefnames`, `nobs` and `residuals` are defined. `vcov` and
`stderror` are **not**, and the omission is deliberate. The weights are
the solution of a constrained optimisation, not an estimator with a
sampling distribution of known form. The gap is a difference between one
observed path and one constructed path, rather than a regression
coefficient. There is no standard error to report. Inference is the
permutation test in [`placebo_test`](@ref), with the caveats measured
there.
"""
struct SyntheticControlResult{T<:AbstractFloat} <: StatsAPI.StatisticalModel
    weights::Matrix{T}
    treated::Vector{Int}
    donors::Vector{Int}
    excluded_neighbours::Vector{Int}
    donor_pool_rule::Symbol
    synthetic::Matrix{T}
    gaps::Matrix{T}
    att_path::Vector{T}
    att::T
    rmspe_pre::Vector{T}
    rmspe_post::Vector{T}
    ratio::Vector{T}
    ratio_aggregate::T
    kkt_gap::Vector{T}
    kkt_violation::Vector{T}
    iterations::Vector{Int}
    penalty::T
    penalty_metric::Symbol
    standardized::Bool
    n_predictors::Int
    T0::Int
    n_units::Int
    n_periods::Int
end

"""
    donor_weights(scm::SyntheticControlResult)

The `m x J` matrix of donor weights, one row per treated unit. Column `q`
corresponds to `scm.donors[q]`.
"""
donor_weights(scm::SyntheticControlResult) = scm.weights

StatsAPI.coef(scm::SyntheticControlResult) = scm.weights
StatsAPI.coefnames(scm::SyntheticControlResult) =
    ["unit $(j)" for j in scm.donors]
StatsAPI.nobs(scm::SyntheticControlResult) =
    length(scm.treated) * scm.n_predictors
StatsAPI.residuals(scm::SyntheticControlResult) =
    scm.gaps[:, 1:scm.T0]

# ─── The estimator ───────────────────────────────────────────────────

"""
    fit_synthetic_control(Y, treated, T0; W = nothing, coords = nothing,
                          predictors = nothing, standardize = true,
                          donor_pool = :exclude_neighbours,
                          penalty = 0.0, penalty_metric = :spatial,
                          metric = EuclideanDistance(),
                          tol = 1e-12, maxiter = 20_000)

Synthetic control for a comparative case study, with the donor pool
screened for spillover contamination and the donor weights optionally
penalised by distance.

# The estimator

For treated unit `i`, weights `w` over the donor pool solve

```
minimise   (x_i - X0 w)'(x_i - X0 w) + λ Σ_j d_ij w_j
subject to w ≥ 0,  Σ_j w_j = 1
```

where `x_i` collects `i`'s pre-treatment predictors, `X0` the donors', and
`d_ij` is the distance from `i` to donor `j`. The synthetic control is
`Σ_j w_j y_jt` and the gap `y_it - Σ_j w_j y_jt` is the estimated effect
after `T0`. With several treated units each gets its own weights and the
`att` is the mean of their post-period gaps.

The program is convex, with a unique minimum whenever `X0` has full column
rank on the active face. It is solved to its own KKT certificate rather
than to an iteration budget. See `kkt_gap` below.

# Why the donor pool is screened

This is the part that makes the method spatial, and it is not cosmetic. If
the policy spills over, the treated unit's neighbours are themselves
affected. The unpenalised fit will then build the counterfactual out of
them, because they track the treated unit best in the pre-period.
The spillover then lands inside the comparison and the gap understates the
effect. `donor_pool = :exclude_neighbours` (the default) removes any
untreated unit adjacent to a treated one under `W`, reusing
[`exposure`](@ref) to find them, and records which units went and why.

`donor_pool = :all` keeps everyone, which is the conventional choice and
is wrong whenever the spillover is real. The cost is measured in
`examples/24_synthetic_control.jl` and pinned in the test suite.

# Why the penalty

The unpenalised weights are frequently **not unique**, especially when the
donor pool is larger than the pre-period is long — the usual case. Several
weight vectors then fit the pre-period identically and imply different
counterfactuals, and which one a solver returns is an accident of the
algorithm. Abadie & L'Hour (2021) break the tie with a penalty on pairwise
discrepancy between the treated unit and each donor it draws on.

`penalty_metric = :spatial` makes that discrepancy geographic distance. It
needs `coords`, with `metric = HaversineDistance()` for lon/lat.
`:covariate` uses standardised predictor distance, and `:none` sets
`λ = 0`. Raising `λ` moves the solution toward the single nearest donor.
`λ = 0` is the unpenalised program.

# Arguments
- `Y`: `N x T` outcomes, units by periods — the same layout as
  [`fit_spatial_did`](@ref).
- `treated`: a unit index, a vector of indices, or a length-`N` 0/1 or
  `Bool` indicator.
- `T0`: the last pre-treatment period. Periods `T0+1:T` are the
  post-treatment window.

# Keywords
- `W`: `N x N` sparse weights, required by the default `donor_pool`.
- `coords`: `N x 2`, required by `penalty_metric = :spatial`.
- `predictors`: optional `N x K` pre-treatment predictors. The default is
  the pre-period outcome path, `Y[:, 1:T0]`.
- `standardize`: scale each predictor to unit variance before fitting.
  This is `V = I` on standardised predictors, taken deliberately in place
  of a nested search over `V`. The nested form is non-unique, so two
  equally good `V` give different weights, and the result stops being
  reproducible.
- `penalty`, `penalty_metric`, `metric`: as above.
- `tol`, `maxiter`: solver controls.

# Fit diagnostics, and reading them honestly

`rmspe_pre` is the pre-period root mean squared gap. A synthetic control
with a poor pre-period fit says nothing about the post period. So
`ratio = rmspe_post / rmspe_pre` is the statistic to look at, rather than
the gap alone. A large gap on a badly fitting unit is not evidence.
[`placebo_test`](@ref) ranks exactly this ratio.

`kkt_gap` and `kkt_violation` are the solution's own optimality
certificate: at the constrained optimum the objective gradient is equal
across the donors with positive weight and no smaller on those at zero.
Both should be at the `1e-10` level or below. They are reported because
unconverged weights are otherwise indistinguishable from converged ones.

# Returns

A [`SyntheticControlResult`](@ref). Note that it carries no `vcov` — see
that type's docstring for why, and [`placebo_test`](@ref) for what to use
instead.

# Example

```julia
W = knn_weights(coords, 4)
scm = fit_synthetic_control(Y, 7, 20; W = W, coords = coords,
                            penalty = 0.1)
scm.att                       # mean post-period gap
pt = placebo_test(scm, Y)     # permutation inference
spillover_gaps(scm, Y; W = W, coords = coords)
```

# References
- Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control
  methods for comparative case studies. *JASA*, 105(490), 493-505.
- Abadie, A. & L'Hour, J. (2021). A penalized synthetic control estimator
  for disaggregated data. *JASA*, 116(536), 1817-1834.
- Abadie, A. (2021). Using synthetic controls: feasibility, data
  requirements, and methodological aspects. *JEL*, 59(2), 391-425.
"""
function fit_synthetic_control(Y::AbstractMatrix, treated, T0::Int;
                               W::Union{Nothing,AbstractSparseMatrix} = nothing,
                               coords = nothing,
                               predictors = nothing,
                               standardize::Bool = true,
                               donor_pool = :exclude_neighbours,
                               penalty::Real = 0.0,
                               penalty_metric::Symbol = :spatial,
                               metric::DistanceMetric = EuclideanDistance(),
                               tol::Real = 1e-14,
                               maxiter::Int = 50_000)
    T = float(eltype(Y)) <: AbstractFloat ? float(eltype(Y)) : Float64
    N, T_per = size(Y)
    N >= 2 || throw(ArgumentError(
        "fit_synthetic_control: need at least 2 units, got $N."))
    1 <= T0 < T_per || throw(ArgumentError(
        "fit_synthetic_control: `T0` must satisfy 1 <= T0 < T = $T_per, got " *
        "$T0. There must be at least one post-treatment period."))

    tr = _treated_indices(treated, N)
    isempty(tr) && throw(ArgumentError(
        "fit_synthetic_control: no treated unit. `treated` selected none."))
    m = length(tr)

    λ = T(penalty)
    λ >= zero(T) || throw(ArgumentError(
        "fit_synthetic_control: `penalty` must be non-negative, got $penalty."))
    pmetric = (λ == zero(T)) ? :none : penalty_metric

    donors, excluded = _donor_pool(donor_pool, tr, N, W)
    J = length(donors)
    rule = donor_pool isa AbstractVector ? :explicit : donor_pool

    # One predictor design over treated units and donors together, so the
    # standardisation is common to both sides of the comparison.
    used = vcat(tr, donors)
    Xall, _ = _scm_predictors(T, Y, predictors, T0, used, standardize)
    K = size(Xall, 1)
    col_of = Dict{Int,Int}(u => q for (q, u) in enumerate(used))
    X0 = Xall[:, (m + 1):(m + J)]

    Yf = T.(Y)
    Yd = Yf[donors, :]

    weights = Matrix{T}(undef, m, J)
    synthetic = Matrix{T}(undef, m, T_per)
    gaps = Matrix{T}(undef, m, T_per)
    kkt = Vector{T}(undef, m)
    viol = Vector{T}(undef, m)
    iters = Vector{Int}(undef, m)

    for (k, i) in enumerate(tr)
        x1 = Vector{T}(Xall[:, col_of[i]])
        d = _penalty_distances(T, pmetric, i, donors, coords, Xall, col_of, metric)
        w, _, it, kg, kv = _solve_pensynth(Matrix{T}(X0), x1, d, λ;
                                           tol = T(tol), maxiter = maxiter)
        weights[k, :] .= w
        syn = Yd' * w                       # length T_per
        synthetic[k, :] .= syn
        gaps[k, :] .= view(Yf, i, :) .- syn
        kkt[k] = kg
        viol[k] = kv
        iters[k] = it
    end

    att_path = vec(mean(gaps; dims = 1))
    post = (T0 + 1):T_per
    att = mean(view(att_path, post))

    rmspe_pre = T[sqrt(mean(abs2, view(gaps, k, 1:T0))) for k in 1:m]
    rmspe_post = T[sqrt(mean(abs2, view(gaps, k, post))) for k in 1:m]
    ratio = T[rmspe_pre[k] > zero(T) ? rmspe_post[k] / rmspe_pre[k] : T(Inf)
              for k in 1:m]
    pre_agg = sqrt(mean(abs2, view(att_path, 1:T0)))
    post_agg = sqrt(mean(abs2, view(att_path, post)))
    ratio_agg = pre_agg > zero(T) ? post_agg / pre_agg : T(Inf)

    return SyntheticControlResult{T}(
        weights, tr, donors, excluded, rule, synthetic, gaps, att_path, att,
        rmspe_pre, rmspe_post, ratio, ratio_agg, kkt, viol, iters,
        λ, pmetric, standardize, K, T0, N, T_per)
end

function Base.show(io::IO, scm::SyntheticControlResult{T}) where {T}
    m = length(scm.treated)
    println(io, "Synthetic Control", m == 1 ? "" : " ($m treated units)")
    println(io, "═"^68)
    @printf(io, "  N = %d units, T = %d periods (T0 = %d, %d post)\n",
            scm.n_units, scm.n_periods, scm.T0, scm.n_periods - scm.T0)
    @printf(io, "  treated: %s\n",
            m <= 8 ? string(scm.treated) : "$m units")
    @printf(io, "  donor pool: %d units (%s)\n", length(scm.donors),
            scm.donor_pool_rule === :exclude_neighbours ?
                "$(length(scm.excluded_neighbours)) neighbours excluded" :
                String(scm.donor_pool_rule))
    if scm.penalty > zero(T)
        @printf(io, "  penalty: λ = %.4g on %s distance\n",
                scm.penalty, String(scm.penalty_metric))
    else
        println(io, "  penalty: none (λ = 0, weights may not be unique)")
    end
    println(io, "")
    @printf(io, "  ATT (mean post-period gap) = %.6f\n", scm.att)
    @printf(io, "  pre-period RMSPE  %.6g   post/pre ratio  %.4g\n",
            mean(scm.rmspe_pre), scm.ratio_aggregate)
    mk = maximum(scm.kkt_gap)
    mv = maximum(scm.kkt_violation)
    @printf(io, "  KKT gap %.2e, violation %.2e, %d-%d iterations%s\n",
            mk, mv, minimum(scm.iterations), maximum(scm.iterations),
            (mk > 1e-6 || mv > 1e-6) ? "  ⚠ NOT converged" : "")
    println(io, "  no standard errors by construction — see `placebo_test`")

    println(io, "")
    println(io, "  Donor weights" * (m == 1 ? "" : " (treated unit 1)"))
    println(io, "  " * "─"^66)
    w = view(scm.weights, 1, :)
    ord = sortperm(collect(w); rev = true)
    shown = 0
    for q in ord
        w[q] > T(1e-6) || break
        @printf(io, "  %-20s %10.6f\n", "unit $(scm.donors[q])", w[q])
        shown += 1
        shown >= 10 && break
    end
    nz = count(>(T(1e-6)), w)
    nz > shown && @printf(io, "  %-20s %10d more\n", "", nz - shown)
    @printf(io, "  %-20s %10.6f  (%d of %d donors)\n", "total", sum(w), nz,
            length(scm.donors))
end

# ─── Placebo inference ───────────────────────────────────────────────

"""
    PlaceboTest

The result of [`placebo_test`](@ref): a permutation p-value for a
synthetic control, with the information needed to read it.

# Fields
- `p_value`: `(1 + #{ratio_placebo >= ratio_observed}) / (1 + R)`.
- `ratio_observed`: the fitted units' post/pre RMSPE ratio.
- `ratio_placebo`: the `R` placebo ratios.
- `n_placebo`: `R`, the number of placebo assignments used.
- `exact`: `true` if every possible assignment was enumerated, `false` if
  they were sampled. An exact permutation p-value and a sampled one are
  not the same object, so which happened is recorded rather than implied.
- `n_possible`: the number of distinct assignments, `binomial(J, m)`, as a
  `Float64` because it overflows `Int` quickly.
- `min_attainable_p`: `1 / (1 + R)`. The smallest p-value the design can
  produce, whatever the effect size.
- `m_treated`, `n_donors`: the assignment's shape.
"""
struct PlaceboTest{T<:AbstractFloat}
    p_value::T
    ratio_observed::T
    ratio_placebo::Vector{T}
    n_placebo::Int
    exact::Bool
    n_possible::Float64
    min_attainable_p::T
    m_treated::Int
    n_donors::Int
end

# Number of m-subsets of J, as a Float64 so it cannot overflow.
function _n_subsets(J::Int, m::Int)
    (m < 0 || m > J) && return 0.0
    r = 1.0
    for q in 1:m
        r *= (J - m + q) / q
        isfinite(r) || return Inf
    end
    return r
end

# All m-subsets of `pool`, in lexicographic order.
function _enumerate_subsets(pool::Vector{Int}, m::Int)
    out = Vector{Vector{Int}}()
    J = length(pool)
    idx = collect(1:m)
    while true
        push!(out, pool[idx])
        k = m
        while k >= 1 && idx[k] == J - m + k
            k -= 1
        end
        k == 0 && break
        idx[k] += 1
        for q in (k + 1):m
            idx[q] = idx[q - 1] + 1
        end
    end
    return out
end

"""
    placebo_test(scm::SyntheticControlResult, Y; n_placebo = 1000,
                 rng = Random.default_rng(), kwargs...)

Permutation inference for a synthetic control: reassign the treatment to
units that were never treated, refit, and ask how unusual the real
assignment's post/pre RMSPE ratio is.

The statistic is `ratio = rmspe_post / rmspe_pre`, aggregated across the
treated units, and the p-value is

```
p = (1 + #{ratio_placebo >= ratio_observed}) / (1 + R)
```

over `R` placebo assignments. Dividing by the pre-period fit is what makes
placebo ratios comparable across units. A donor that the synthetic control
tracks badly before treatment will show a large post-period gap for
reasons that have nothing to do with any policy.

`Y` must be the same outcome matrix the fit used. Remaining keyword
arguments are forwarded to [`fit_synthetic_control`](@ref) and must match
the original call — `W`, `coords`, `predictors`, `penalty` and so on —
since a placebo fitted under different settings is not a comparison.

# Assignments: enumerated or sampled

With `J` clean donors and `m` treated units there are `binomial(J, m)`
possible assignments. All of them are enumerated when that number is at
most `n_placebo`, which gives an exact permutation test. Otherwise
`n_placebo` assignments are sampled without replacement using `rng`. The
result records which happened in its `exact` field.

Placebo units are drawn from the donor pool only. Units excluded for
neighbouring a treated unit are not eligible: they are contaminated by the
spillover, so a "placebo" there is not a null.

!!! warning "What this p-value cannot do"
    The smallest attainable value is `1 / (1 + R)`, and when assignments
    are enumerated `R = binomial(J, m)` may be small. A single treated
    unit with 18 donors cannot produce a p-value below 0.053 however large
    the effect, so it cannot reject at the 5% level at all. The bound is
    reported as `min_attainable_p` rather than left for a user to
    rediscover.

!!! warning "Permutation inference assumes exchangeable units"
    The test treats any unit as equally likely to have been treated.
    Spatial correlation breaks that, and the rejection rate under the null
    is measured rather than assumed — see the table below and
    `docs/AUDIT.md`.

# Measured size under the null

Null distribution measured, not assumed: no treatment effect anywhere, a
7x7 rook lattice (`N = 49`), `T = 16` with `T0 = 12`, an interior treated
unit so 4 neighbours are excluded, leaving 44 clean donors. The
permutation is therefore exact over 44 assignments and the smallest
attainable p-value is `1/45 = 0.0222`. Errors generated as
`(I - rho*W)^{-1}e`, 1,000 replications per row.

| `rho` | `p <= 0.022` | `p <= 0.05` | `p <= 0.10` | mean `p` |
|---|---|---|---|---|
| 0.0 | 0.023 | 0.046 | 0.095 | 0.503 |
| 0.6 | 0.017 | 0.045 | 0.094 | 0.492 |
| 0.9 | 0.041 | 0.068 | 0.125 | 0.438 |

The test is correctly sized with no spatial correlation and with
moderate correlation — a mean p-value of 0.50 is what exchangeability
predicts. At `rho = 0.9` it over-rejects by roughly a third of the
nominal level (6.8% against 5%, 12.5% against 10%, both about 2.6
standard errors above nominal at 1,000 replications). Treat a p-value
near the threshold as uninformative when the outcome is strongly
spatially correlated.

The ratio statistic is what keeps the distortion this small. Scaling the
post-period gap by the pre-period fit makes units comparable even when
their error variances are not, so spatial dependence largely divides
out.

# Returns

A [`PlaceboTest`](@ref).
"""
function placebo_test(scm::SyntheticControlResult{T}, Y::AbstractMatrix;
                      n_placebo::Int = 1000,
                      rng::AbstractRNG = Random.default_rng(),
                      kwargs...) where {T}
    size(Y) == (scm.n_units, scm.n_periods) || throw(DimensionMismatch(
        "placebo_test: `Y` is $(size(Y)) but the fit used " *
        "$((scm.n_units, scm.n_periods)). Pass the same outcome matrix."))
    n_placebo >= 1 || throw(ArgumentError(
        "placebo_test: `n_placebo` must be at least 1, got $n_placebo."))

    m = length(scm.treated)
    pool = scm.donors
    J = length(pool)
    J > m || throw(ArgumentError(
        "placebo_test: the donor pool has $J units and $m were treated, so " *
        "there is no placebo assignment to compare against. A permutation " *
        "test needs more clean donors than treated units."))

    npos = _n_subsets(J, m)
    exact = npos <= n_placebo
    assignments = if exact
        _enumerate_subsets(pool, m)
    else
        seen = Set{Vector{Int}}()
        out = Vector{Vector{Int}}()
        tries = 0
        maxtries = 200 * n_placebo
        while length(out) < n_placebo && tries < maxtries
            tries += 1
            s = sort(pool[Random.randperm(rng, J)[1:m]])
            if !(s in seen)
                push!(seen, s)
                push!(out, s)
            end
        end
        out
    end

    ratios = Vector{T}(undef, length(assignments))
    for (r, set) in enumerate(assignments)
        # The placebo pool is the real donor pool minus this placebo's own
        # units, so a placebo never builds its counterfactual from itself.
        sub = [j for j in pool if !(j in set)]
        pscm = fit_synthetic_control(Y, set, scm.T0;
                                     donor_pool = sub,
                                     standardize = scm.standardized,
                                     penalty = scm.penalty,
                                     penalty_metric = scm.penalty_metric,
                                     kwargs...)
        ratios[r] = pscm.ratio_aggregate
    end

    R = length(ratios)
    obs = scm.ratio_aggregate
    p = T((1 + count(>=(obs), ratios)) / (1 + R))
    return PlaceboTest{T}(p, obs, ratios, R, exact, npos,
                          T(1 / (1 + R)), m, J)
end

function Base.show(io::IO, pt::PlaceboTest{T}) where {T}
    println(io, "Placebo (permutation) test for a synthetic control")
    println(io, "═"^68)
    @printf(io, "  statistic: post/pre RMSPE ratio = %.4f\n", pt.ratio_observed)
    @printf(io, "  assignments: %d %s", pt.n_placebo,
            pt.exact ? "(all of them — exact permutation test)\n" :
                       "sampled")
    pt.exact || @printf(io, " of %.3g possible\n", pt.n_possible)
    @printf(io, "  %d treated unit%s drawn from %d clean donors\n",
            pt.m_treated, pt.m_treated == 1 ? "" : "s", pt.n_donors)
    println(io, "")
    @printf(io, "  p = %.4f   (smallest attainable: %.4f)\n",
            pt.p_value, pt.min_attainable_p)
    if pt.min_attainable_p > T(0.05)
        println(io, "  ⚠ the donor pool is too small to reject at the 5% level")
    end
    q = _quantile_sorted(sort(pt.ratio_placebo), (0.5, 0.9, 0.95))
    @printf(io, "  placebo ratios: median %.3f, 90th %.3f, 95th %.3f\n",
            q[1], q[2], q[3])
end

# Linear-interpolated quantiles of an already sorted vector. Local so the
# module does not depend on Statistics' quantile for three numbers in a
# show method.
function _quantile_sorted(v::AbstractVector{T}, ps) where {T}
    n = length(v)
    n == 0 && return map(_ -> T(NaN), ps)
    return map(ps) do p
        h = (n - 1) * T(p) + one(T)
        lo = clamp(floor(Int, h), 1, n)
        hi = clamp(lo + 1, 1, n)
        v[lo] + (h - lo) * (v[hi] - v[lo])
    end
end

# ─── Spillover gaps on the excluded neighbours ───────────────────────

"""
    SpilloverGaps

What the policy did to the treated units' neighbours, returned by
[`spillover_gaps`](@ref).

# Fields
- `units`: the neighbour units, in the order of `gaps`' rows.
- `gaps`: `k x T`, each row a neighbour's outcome minus its synthetic
  control built from the clean donor pool.
- `spillover_path`: length-`T` mean gap across neighbours.
- `spillover`: mean of `spillover_path` over the post-treatment periods —
  the estimated spillover effect.
- `rmspe_pre`: per-neighbour pre-period fit, to be read before the
  spillover is believed.
- `donors`: the donor pool used, the same clean pool as the main fit.
"""
struct SpilloverGaps{T<:AbstractFloat}
    units::Vector{Int}
    gaps::Matrix{T}
    spillover_path::Vector{T}
    spillover::T
    rmspe_pre::Vector{T}
    donors::Vector{Int}
end

"""
    spillover_gaps(scm::SyntheticControlResult, Y; kwargs...)

Estimate the spillover onto the units the donor pool excluded.

[`fit_synthetic_control`](@ref) with `donor_pool = :exclude_neighbours`
drops every untreated unit adjacent to a treated one, because the
spillover contaminates them. That contamination is itself the quantity a
spatial policy evaluation wants. This function synthesises each excluded
neighbour from the **same clean donor pool** and reads its gap as the
estimated spillover — the synthetic control counterpart of
`att_spill_control` in [`fit_spatial_did`](@ref).

The neighbour is not treated, so under no spillover its gap should be
noise around zero. A systematic post-period gap is the spillover. The same
caution applies as for the main fit: check `rmspe_pre` before believing a
gap, and use [`placebo_test`](@ref) on the main fit to calibrate what gap
sizes arise by chance.

`Y` must be the outcome matrix the fit used. Remaining keyword arguments
are forwarded to [`fit_synthetic_control`](@ref) and should match the
original call.

Errors if the fit excluded no neighbours — there is then nothing to
report, and returning an empty result would read as "no spillover" rather
than "not asked".
"""
function spillover_gaps(scm::SyntheticControlResult{T}, Y::AbstractMatrix;
                        kwargs...) where {T}
    size(Y) == (scm.n_units, scm.n_periods) || throw(DimensionMismatch(
        "spillover_gaps: `Y` is $(size(Y)) but the fit used " *
        "$((scm.n_units, scm.n_periods)). Pass the same outcome matrix."))
    isempty(scm.excluded_neighbours) && throw(ArgumentError(
        "spillover_gaps: the fit excluded no neighbours, so there is no " *
        "spillover to estimate. This needs " *
        "`donor_pool = :exclude_neighbours` (the default) and at least one " *
        "untreated unit adjacent to a treated one under `W`."))

    units = scm.excluded_neighbours
    k = length(units)
    gaps = Matrix{T}(undef, k, scm.n_periods)
    rmspe_pre = Vector{T}(undef, k)

    for (q, u) in enumerate(units)
        s = fit_synthetic_control(Y, u, scm.T0;
                                  donor_pool = scm.donors,
                                  standardize = scm.standardized,
                                  penalty = scm.penalty,
                                  penalty_metric = scm.penalty_metric,
                                  kwargs...)
        gaps[q, :] .= view(s.gaps, 1, :)
        rmspe_pre[q] = s.rmspe_pre[1]
    end

    path = vec(mean(gaps; dims = 1))
    post = (scm.T0 + 1):scm.n_periods
    return SpilloverGaps{T}(units, gaps, path, mean(view(path, post)),
                            rmspe_pre, scm.donors)
end

function Base.show(io::IO, sg::SpilloverGaps{T}) where {T}
    println(io, "Spillover onto excluded neighbours")
    println(io, "═"^68)
    @printf(io, "  %d neighbour unit%s, synthesised from %d clean donors\n",
            length(sg.units), length(sg.units) == 1 ? "" : "s",
            length(sg.donors))
    @printf(io, "  estimated spillover (mean post-period gap) = %.6f\n",
            sg.spillover)
    @printf(io, "  mean pre-period RMSPE = %.6g\n", mean(sg.rmspe_pre))
end

end # module
