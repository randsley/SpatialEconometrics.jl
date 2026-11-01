module SpatialShiftShare

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames, dof_residual

import ..SpatialUtils: check_design_rank, _print_coef_table, _normcdf, _chi2_sf
import ..SpatialHAC: ClusterRobustVCOV, vcov_cluster
import ..SpatialDiagnostics: MoranResult, moran_global
import Random
import Random: AbstractRNG

export ShiftShareDesign, shift_share_design,
       ShiftShareFit, fit_shift_share,
       ExposureConcentration, exposure_concentration,
       ShockAggregates, shock_aggregate,
       BalanceTest, balance_test,
       RotembergWeights, rotemberg_weights,
       ShiftShareMoran, shift_share_moran

# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

# Residual of `v` after projecting out the columns of `Z`. Z always carries
# an intercept here, so this is a demeaning as well as a partialling-out.
_resid_on(v::AbstractVector{T}, Z::AbstractMatrix{T}) where {T<:AbstractFloat} =
    v .- Z * (Z \ v)

_resid_on(V::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T<:AbstractFloat} =
    V .- Z * (Z \ V)

# Weighted mean, used for the shock-level regressions where the weight is
# average exposure.
function _wmean(v::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    sw = sum(w)
    sw > zero(T) || throw(ArgumentError("weights sum to $sw, cannot form a weighted mean"))
    return dot(v, w) / sw
end

# ═══════════════════════════════════════════════════════════════════════
# Design
# ═══════════════════════════════════════════════════════════════════════

"""
    ShiftShareDesign

A validated shift-share (Bartik) design: an `N × K` matrix of exposure
shares, a `K`-vector of shocks, the constructed regressor, and the controls
the shares and shocks will be read against. Built by
[`shift_share_design`](@ref).

The regressor is

```
X_i = Σ_k W[i, k] * g[k]
```

so region `i`'s treatment is a weighted average of common shocks, with
region-specific weights. That construction is the whole source of this
design's difficulty: two regions with similar industry mixes have
mechanically correlated values of `X`, and therefore correlated residuals,
**whether or not they are anywhere near each other**. Inference that
assumes independent or merely spatially-clustered regions is not valid
here, which is why [`fit_shift_share`](@ref) defaults to the Adão, Kolesár
& Morales standard error rather than a heteroskedasticity-robust one.

# Fields

- `W::Matrix{T}` — `N × K` exposure shares, region by shock.
- `g::Vector{T}` — the `K` shocks.
- `X::Vector{T}` — the constructed regressor, `W * g`.
- `Z::Matrix{T}` — controls, always including an intercept column first.
- `s_bar::Vector{T}` — average exposure to each shock, `ŝ_k = (1/N) Σ_i W[i,k]`.
- `exposure_sum::Vector{T}` — per-region `Σ_k W[i,k]`.
- `complete_shares::Bool` — whether `exposure_sum` is constant across regions.
- `shock_names::Vector{String}`

# References

Borusyak, Hull & Jaravel (2022), *Review of Economic Studies* 89(1),
181–213.
"""
struct ShiftShareDesign{T<:AbstractFloat}
    W::Matrix{T}
    g::Vector{T}
    X::Vector{T}
    Z::Matrix{T}
    s_bar::Vector{T}
    exposure_sum::Vector{T}
    complete_shares::Bool
    shock_names::Vector{String}
end

"""
    shift_share_design(W, g; controls = nothing, shock_names = nothing)

Validate a shift-share design and construct its regressor.

`W` is `N × K` exposure shares (region by shock) and `g` is the `K`-vector
of shocks. `controls` is an optional `N × L` matrix; an intercept is always
added and should not be supplied.

# Incomplete shares

Borusyak, Hull & Jaravel's framework assumes the shares sum to a constant
across shocks for every region. When they do not — the usual case when the
shares cover only part of the economy, such as manufacturing employment
only — the sum of exposure is itself a regional characteristic that
correlates with the regressor, and **it must be included as a control**.
This function reports `complete_shares = false` and warns once; it does not
silently add the control, because whether the exposure sum belongs on the
right-hand side is the user's identification decision, not a default.

# Errors

Rank-deficient controls, a shock vector of the wrong length, non-finite
entries, fewer than two shocks, and an all-zero share column are each
refused by name. An all-zero column matters because that shock has no
exposure anywhere, so its shock-level aggregate in
[`shock_aggregate`](@ref) would divide by zero.
"""
function shift_share_design(W::AbstractMatrix{<:Real},
                            g::AbstractVector{<:Real};
                            controls::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                            shock_names::Union{Nothing, AbstractVector{<:AbstractString}} = nothing)
    T = float(promote_type(eltype(W), eltype(g),
                           isnothing(controls) ? Bool : eltype(controls)))
    Wm = convert(Matrix{T}, W)
    gv = convert(Vector{T}, g)
    N, K = size(Wm)

    N >= 2 || throw(ArgumentError(
        "shift_share_design: need at least 2 regions, got N = $N."))
    K >= 2 || throw(ArgumentError(
        "shift_share_design: need at least 2 shocks to have a shift-share " *
        "design at all, got K = $K. With one shock the regressor is that " *
        "shock scaled by a single share column, and there is nothing to " *
        "average over."))
    length(gv) == K || throw(DimensionMismatch(
        "shift_share_design: `g` has $(length(gv)) entries but `W` has $K " *
        "columns; the shares and the shocks must be aligned shock by shock."))
    all(isfinite, Wm) || throw(ArgumentError(
        "shift_share_design: `W` contains a non-finite entry."))
    all(isfinite, gv) || throw(ArgumentError(
        "shift_share_design: `g` contains a non-finite entry."))

    dead = [k for k in 1:K if all(iszero, @view Wm[:, k])]
    isempty(dead) || throw(ArgumentError(
        "shift_share_design: shock column(s) $(dead) are identically zero, " *
        "so no region is exposed to them. They contribute nothing to the " *
        "regressor and their shock-level aggregates would divide by zero. " *
        "Drop those columns, along with the matching entries of `g`."))

    Z = isnothing(controls) ? ones(T, N, 1) :
        hcat(ones(T, N), convert(Matrix{T}, controls))
    size(Z, 1) == N || throw(DimensionMismatch(
        "shift_share_design: `controls` has $(size(Z, 1)) rows, `W` has $N."))
    all(isfinite, Z) || throw(ArgumentError(
        "shift_share_design: `controls` contains a non-finite entry."))
    znames = ["(intercept)";
              ["control[$j]" for j in 1:(size(Z, 2) - 1)]]
    check_design_rank(Z, znames, "shift_share_design";
                      context = "the controls, with the intercept")

    names = isnothing(shock_names) ? ["shock[$k]" for k in 1:K] :
            String.(collect(shock_names))
    length(names) == K || throw(DimensionMismatch(
        "shift_share_design: `shock_names` has $(length(names)) entries, `W` has $K columns."))

    X = Wm * gv
    s_bar = vec(sum(Wm; dims = 1)) ./ T(N)
    esum = vec(sum(Wm; dims = 2))
    complete = maximum(esum) - minimum(esum) <= sqrt(eps(T)) * max(one(T), maximum(abs, esum))

    if !complete
        @warn "shift_share_design: exposure shares do not sum to a constant " *
              "across shocks (range $(minimum(esum)) to $(maximum(esum))). " *
              "Borusyak, Hull & Jaravel (2022, §3.2) show the sum of exposure " *
              "is then a confounder and belongs in `controls`. It has not been " *
              "added for you."
    end

    return ShiftShareDesign{T}(Wm, gv, X, Z, s_bar, esum, complete, names)
end

Base.show(io::IO, d::ShiftShareDesign) = begin
    N, K = size(d.W)
    println(io, "ShiftShareDesign: $N regions, $K shocks")
    println(io, "  controls         : $(size(d.Z, 2) - 1) (plus intercept)")
    println(io, "  complete shares  : $(d.complete_shares)")
    @printf(io, "  regressor X      : mean %.4f, sd %.4f\n",
            mean(d.X), length(d.X) > 1 ? std(d.X) : zero(eltype(d.X)))
end

# ═══════════════════════════════════════════════════════════════════════
# Exposure concentration
# ═══════════════════════════════════════════════════════════════════════

"""
    ExposureConcentration

How concentrated average exposure is across shocks, returned by
[`exposure_concentration`](@ref).

Borusyak, Hull & Jaravel's identification argument needs a Herfindahl
condition on average exposure, `Σ_k ŝ_k² → 0`: no finite set of shocks may
drive the estimate asymptotically, which they put equivalently as "the
largest industry becomes vanishingly small as the sample grows". The
`effective_shocks` field is the plain reciprocal `1 / Σ_k ŝ_k²` of that
quantity, reported because a count is easier to read than a sum of squares.
It is a rescaling of their condition for legibility and not a named
statistic of theirs.

Shares are normalised to sum to one across shocks before the Herfindahl is
formed, so that `hhi` is comparable across designs; with `K` shocks and
uniform exposure, `effective_shocks == K` exactly.

# Fields

- `hhi::T` — `Σ_k (ŝ_k / Σ_j ŝ_j)²`.
- `effective_shocks::T` — `1 / hhi`.
- `n_shocks::Int` — `K`, for comparison with `effective_shocks`.
- `max_share::T` — the largest normalised average exposure.
- `order::Vector{Int}` — shock indices, most exposed first.
- `top_shares::Vector{T}` — the normalised shares in that order.
"""
struct ExposureConcentration{T<:AbstractFloat}
    hhi::T
    effective_shocks::T
    n_shocks::Int
    max_share::T
    order::Vector{Int}
    top_shares::Vector{T}
end

"""
    exposure_concentration(design) -> ExposureConcentration

Herfindahl concentration of average exposure, and its reciprocal as an
effective number of shocks.

Read it before trusting any shift-share estimate: the asymptotics are in
the number of shocks, so an estimate resting on an effective handful of
them is not supported by the theory however many regions there are. A
design with 300 industries and `effective_shocks` of 4 is a four-shock
study wearing a large-sample disguise.

# Example

```julia
d = shift_share_design(W, g)
c = exposure_concentration(d)
c.effective_shocks   # compare with c.n_shocks
```
"""
function exposure_concentration(d::ShiftShareDesign{T}) where {T<:AbstractFloat}
    tot = sum(d.s_bar)
    tot > zero(T) || throw(ArgumentError(
        "exposure_concentration: total average exposure is $tot; the share " *
        "matrix carries no exposure at all."))
    sn = d.s_bar ./ tot
    hhi = sum(abs2, sn)
    ord = sortperm(sn; rev = true)
    return ExposureConcentration{T}(hhi, inv(hhi), length(sn),
                                    sn[ord[1]], ord, sn[ord])
end

Base.show(io::IO, c::ExposureConcentration) = begin
    println(io, "ExposureConcentration")
    @printf(io, "  Herfindahl of average exposure : %.6f\n", c.hhi)
    @printf(io, "  effective number of shocks     : %.2f of %d\n",
            c.effective_shocks, c.n_shocks)
    @printf(io, "  largest single exposure share  : %.4f\n", c.max_share)
    ntop = min(5, length(c.order))
    print(io, "  most exposed shocks            : ")
    for j in 1:ntop
        @printf(io, "%d (%.3f)%s", c.order[j], c.top_shares[j],
                j == ntop ? "\n" : ", ")
    end
    if c.effective_shocks < 50
        println(io, "  WARNING: fewer than 50 effective shocks. The Borusyak-Hull-")
        println(io, "  Jaravel asymptotics are in the number of shocks, not regions,")
        println(io, "  and this is the binding constraint on every inference in a")
        println(io, "  shift-share design. Measured in docs/AUDIT.md: AKM coverage of")
        println(io, "  a nominal 95% interval fell from 0.936 at 100 effective shocks")
        println(io, "  to 0.864 at 20 and 0.827 at 38-with-concentration, and the")
        println(io, "  balance test rejected a true null 23% of the time at 9.")
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Shock-level aggregation (the Borusyak-Hull-Jaravel transform)
# ═══════════════════════════════════════════════════════════════════════

"""
    ShockAggregates

The shift-share design rewritten at the level of the shocks, returned by
[`shock_aggregate`](@ref).

Borusyak, Hull & Jaravel (2022, eq. 4–5) show that a shift-share estimate
is **numerically identical** to an exposure-weighted estimate computed from
`K` shock-level observations:

```
β̂ = Σ_k ŝ_k g_k ȳ_k / Σ_k ŝ_k g_k x̄_k
```

where `ŝ_k` is average exposure, `v̄_k = Σ_i W[i,k] ṽ_i / Σ_i W[i,k]` is the
exposure-weighted mean of a residualised regional variable, and `ȳ`, `x̄`
are those means for the outcome and the regressor or treatment. This is an
identity, not an approximation, and `beta_equiv` reproduces
[`fit_shift_share`](@ref)'s estimate to machine precision.

That matters for more than elegance. It says the design has only `K`
independent observations' worth of information, which is why inference must
be done at this level — and it makes balance and pre-trend tests
straightforward, since conventional standard errors *are* valid once the
regression is a shock-level one. See [`balance_test`](@ref).

# Fields

- `s_bar::Vector{T}`, `g::Vector{T}` — average exposure and shocks.
- `y_bar::Vector{T}` — exposure-weighted means of the residualised outcome.
- `x_bar::Vector{T}` — the same for the regressor (direct) or treatment (IV).
- `beta_equiv::T` — the ratio above.
- `is_iv::Bool`
"""
struct ShockAggregates{T<:AbstractFloat}
    s_bar::Vector{T}
    g::Vector{T}
    y_bar::Vector{T}
    x_bar::Vector{T}
    beta_equiv::T
    is_iv::Bool
end

# Exposure-weighted shock-level mean of a residualised regional vector.
function _aggregate(d::ShiftShareDesign{T}, v::AbstractVector{T}) where {T<:AbstractFloat}
    num = d.W' * v
    den = vec(sum(d.W; dims = 1))
    return num ./ den
end

"""
    shock_aggregate(design, y; treatment = nothing) -> ShockAggregates

Collapse the design to shock level, following Borusyak, Hull & Jaravel
(2022, eq. 4–5), and return the equivalent estimate.

`y` is the regional outcome. With `treatment = nothing` the regressor of
interest is the shift-share variable itself; supply `treatment` for the
instrumental-variables design, where the shift-share variable instruments
it.

Both the outcome and the regressor are residualised on the design's
controls first, exactly as in the regional regression, so `beta_equiv`
equals the regional estimate rather than merely resembling it.
"""
function shock_aggregate(d::ShiftShareDesign{T}, y::AbstractVector{<:Real};
                         treatment::Union{Nothing, AbstractVector{<:Real}} = nothing) where {T<:AbstractFloat}
    N = size(d.W, 1)
    yv = convert(Vector{T}, y)
    length(yv) == N || throw(DimensionMismatch(
        "shock_aggregate: `y` has $(length(yv)) entries, the design has $N regions."))

    ỹ = _resid_on(yv, d.Z)
    is_iv = !isnothing(treatment)
    xsrc = if is_iv
        tv = convert(Vector{T}, treatment)
        length(tv) == N || throw(DimensionMismatch(
            "shock_aggregate: `treatment` has $(length(tv)) entries, the design has $N regions."))
        _resid_on(tv, d.Z)
    else
        _resid_on(d.X, d.Z)
    end

    y_bar = _aggregate(d, ỹ)
    x_bar = _aggregate(d, xsrc)

    w = d.s_bar .* d.g
    num = dot(w, y_bar)
    den = dot(w, x_bar)
    beta = den == zero(T) ? T(NaN) : num / den

    return ShockAggregates{T}(copy(d.s_bar), copy(d.g), y_bar, x_bar, beta, is_iv)
end

Base.show(io::IO, a::ShockAggregates) = begin
    println(io, "ShockAggregates: $(length(a.g)) shocks ",
            a.is_iv ? "(IV design)" : "(direct design)")
    @printf(io, "  equivalent estimate : %.6f\n", a.beta_equiv)
    println(io, "  (identical to the regional estimate; Borusyak-Hull-Jaravel eq. 4)")
end

# ═══════════════════════════════════════════════════════════════════════
# Fit, with the Adao-Kolesar-Morales standard error
# ═══════════════════════════════════════════════════════════════════════

"""
    ShiftShareFit

A shift-share regression with its standard error, returned by
[`fit_shift_share`](@ref). Subtypes `StatsAPI.RegressionModel`, and
`coef`, `vcov`, `stderror`, `nobs`, `residuals`, `coefnames` and
`dof_residual` are all defined.

# Fields

- `beta::T`, `se::T` — the coefficient of interest and its standard error.
- `vcov::Matrix{T}` — `1 × 1`, for interface compatibility.
- `residuals::Vector{T}` — regression residuals (structural residuals for IV).
- `X_tilde::Vector{T}` — the regressor with controls partialled out.
- `X_ring::Vector{T}` — `X̊`, the `K` coefficients from regressing `X_tilde`
  on the share matrix. Not the raw shocks; see [`fit_shift_share`](@ref).
- `R_hat::Vector{T}` — `R̂_k = Σ_i W[i,k] ε̂_i`, the share-weighted residual sums.
- `first_stage::T` — `π̂` for IV, `NaN` otherwise.
- `vcov_type::Symbol` — `:akm` or `:hc1`.
- `is_iv::Bool`
"""
struct ShiftShareFit{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::T
    se::T
    vcov::Matrix{T}
    residuals::Vector{T}
    X_tilde::Vector{T}
    X_ring::Vector{T}
    R_hat::Vector{T}
    first_stage::T
    vcov_type::Symbol
    is_iv::Bool
    coefnames::Vector{String}
    design::ShiftShareDesign{T}
end

StatsAPI.coef(m::ShiftShareFit) = [m.beta]
StatsAPI.vcov(m::ShiftShareFit) = m.vcov
StatsAPI.stderror(m::ShiftShareFit) = [m.se]
StatsAPI.nobs(m::ShiftShareFit) = length(m.residuals)
StatsAPI.residuals(m::ShiftShareFit) = m.residuals
StatsAPI.coefnames(m::ShiftShareFit) = m.coefnames
StatsAPI.dof_residual(m::ShiftShareFit) =
    length(m.residuals) - size(m.design.Z, 2) - 1

"""
    fit_shift_share(y, design; treatment = nothing, vcov = :akm)

Fit a shift-share regression and attach a standard error that is valid
under the design's actual dependence structure.

With `treatment = nothing` this regresses `y` on the shift-share regressor
and the controls. Supply `treatment` for the canonical Bartik design, where
the shift-share variable instruments an endogenous regional regressor.

# Why the default standard error is not the usual one

Adão, Kolesár & Morales (2019) show by placebo that conventional standard
errors in this design reject a true null **in up to 55% of samples at the
5% level**. The reason is structural rather than incidental: residuals are
correlated across regions with similar exposure shares, no matter how far
apart those regions are, so neither heteroskedasticity-robust nor
spatially-clustered covariances address it.

Their estimator (eq. 29) is

```
se(β̂) = sqrt( Σ_k X̊_k² R̂_k² ) / ( Σ_i X̃_i² ),   R̂_k = Σ_i W[i,k] ε̂_i
```

and for the IV design (eq. 39) the same numerator over
`Σ_i X̃_i² · |π̂|`, with `ε̂` the structural residual and `π̂` the first-stage
coefficient.

`X̊` deserves emphasis because it is easy to get wrong: it is **not** the
shock vector `g`. It is the vector of coefficients from regressing the
residualised regressor `X̃` on the share matrix `W`, following step 2 of
their Remark 5. The two coincide only when there are no controls and `W` is
square and invertible.

Inference here is conditional on the shares and on the potential outcomes,
with randomness coming from the shocks (their Remark 4). That is why the
estimator tolerates arbitrary cross-regional correlation in the residuals.

# Keyword `vcov`

- `:akm` (default) — the estimator above.
- `:hc1` — heteroskedasticity-robust, provided **for comparison only**, so
  the gap can be seen rather than described. It is the standard error the
  paper shows to be badly undersized; do not report it.

# Requirements

`X̊` is identified only when the share matrix has at least as many regions
as shocks and is of full column rank. A design with more shocks than
regions is refused by name rather than returning a least-norm solution that
would look like an answer.

# Measured coverage

Coverage of a nominal 95% interval, 2,000 replications, shares and
residuals fixed and the shocks resampled (their Remark 4 framework), with
`N = 300`. "Sector frac" is the share of residual variance carried by a
sector-mix component — the dependence this estimator exists for. The
cluster column clusters on each region's largest-exposure sector.

| Effective shocks | Sector frac | AKM | HC1 | cluster |
|---:|---:|---:|---:|---:|
| 99.9 | 0.00 | 0.930 | 0.950 | 0.947 |
| 99.9 | 0.40 | 0.938 | 0.902 | 0.894 |
| 99.9 | 0.82 | 0.936 | 0.732 | 0.720 |
| 99.9 | 1.00 | 0.937 | 0.664 | 0.658 |
| 50.0 | 0.82 | 0.919 | 0.632 | 0.619 |
| 20.0 | 0.77 | 0.864 | 0.426 | 0.404 |
| 37.9 | 0.90 | 0.827 | 0.289 | 0.241 |

Three things to take from it. The failure the paper describes is real and
large: HC1 rejects a true null in up to **71%** of samples here. Clustering
on the dominant sector is **not** a substitute — it tracks HC1 almost
exactly, because the dependence runs through the whole share vector and not
through one sector. And AKM itself degrades once the effective number of
shocks is small, to 0.864 at 20 and 0.827 at 38 with concentrated exposure.

That last row is the one to respect: coverage was *worse* at 37.9 effective
shocks with concentrated exposure than at 50.0 with dispersed exposure, so
the effective-shock count is a useful summary and not a sufficient one.
Adão, Kolesár & Morales anticipate the small-shock case themselves — eq. 29
generalises a cluster-robust formula, so it inherits that formula's
downward bias from estimation noise in the residuals, which is why they
also propose a null-imposed interval (their Remark 6). That variant is
**not implemented here**; where [`exposure_concentration`](@ref) reports few
effective shocks, treat this interval as optimistic rather than exact.

With no sector component at all (first row) AKM costs a little: 0.930
against HC1's 0.950. That is the price of robustness when it is not needed.

# References

Adão, Kolesár & Morales (2019), *Quarterly Journal of Economics* 134(4),
1949–2010, eqs. 29 and 39 and Remarks 4–5.
"""
function fit_shift_share(y::AbstractVector{<:Real}, d::ShiftShareDesign{T};
                         treatment::Union{Nothing, AbstractVector{<:Real}} = nothing,
                         vcov::Symbol = :akm) where {T<:AbstractFloat}
    N, K = size(d.W)
    yv = convert(Vector{T}, y)
    length(yv) == N || throw(DimensionMismatch(
        "fit_shift_share: `y` has $(length(yv)) entries, the design has $N regions."))
    all(isfinite, yv) || throw(ArgumentError("fit_shift_share: `y` contains a non-finite entry."))
    vcov in (:akm, :hc1) || throw(ArgumentError(
        "fit_shift_share: `vcov` must be :akm or :hc1, got :$vcov."))

    X̃ = _resid_on(d.X, d.Z)
    denom = dot(X̃, X̃)
    denom > zero(T) || throw(ArgumentError(
        "fit_shift_share: the shift-share regressor has no variation left " *
        "after the controls are partialled out. Either the shares and shocks " *
        "produce a constant, or a control already spans the regressor."))

    is_iv = !isnothing(treatment)
    if is_iv
        tv = convert(Vector{T}, treatment)
        length(tv) == N || throw(DimensionMismatch(
            "fit_shift_share: `treatment` has $(length(tv)) entries, the design has $N regions."))
        all(isfinite, tv) || throw(ArgumentError(
            "fit_shift_share: `treatment` contains a non-finite entry."))
        xt = dot(X̃, tv)
        abs(xt) > zero(T) || throw(ArgumentError(
            "fit_shift_share: the instrument is orthogonal to the treatment " *
            "after partialling out the controls, so the first stage is exactly " *
            "zero and the IV estimate is undefined."))
        β = dot(X̃, yv) / xt
        ε = _resid_on(yv .- tv .* β, d.Z)
        π̂ = xt / denom
        se_denom = abs(xt)
    else
        # Regress y on [X Z] and read the coefficient on X. Equivalent to
        # the partialled-out ratio, computed that way so the residual is the
        # full-design residual rather than one reconstructed from it.
        β = dot(X̃, yv) / denom
        ε = _resid_on(yv .- d.X .* β, d.Z)
        π̂ = T(NaN)
        se_denom = denom
    end

    # Step 2 of Remark 5: the shock-level projection of the residualised
    # regressor. This is the piece that is not simply `g`.
    N >= K || throw(ArgumentError(
        "fit_shift_share: the design has $K shocks and only $N regions, so " *
        "the shock-level projection in Adao-Kolesar-Morales Remark 5 step 2 " *
        "is underdetermined and the standard error is not identified. Pool " *
        "shocks into coarser groups, or add regions."))
    rk = rank(d.W)
    rk == K || throw(ArgumentError(
        "fit_shift_share: the share matrix has rank $rk with $K columns, so " *
        "the shock-level projection is not unique and the standard error is " *
        "not identified. Two or more shocks have collinear exposure patterns; " *
        "pool them."))
    X̊ = d.W \ X̃
    R̂ = d.W' * ε

    se = if vcov === :akm
        sqrt(sum(abs2, X̊ .* R̂)) / se_denom
    else
        # HC1 on the partialled-out regression, for comparison only.
        dof = N - size(d.Z, 2) - 1
        dof > 0 || throw(ArgumentError(
            "fit_shift_share: no residual degrees of freedom for an HC1 covariance."))
        meat = sum(abs2, X̃ .* ε) * T(N) / T(dof)
        sqrt(meat) / se_denom
    end

    name = is_iv ? "treatment (shift-share IV)" : "shift-share regressor"
    return ShiftShareFit{T}(β, se, fill(se^2, 1, 1), ε, X̃, X̊, R̂, π̂,
                            vcov, is_iv, [name], d)
end

Base.show(io::IO, m::ShiftShareFit) = begin
    println(io, "ShiftShareFit: ", m.is_iv ? "shift-share IV" : "direct shift-share regression")
    println(io, "  regions $(nobs(m)), shocks $(length(m.R_hat))")
    println(io, "  standard error: ", m.vcov_type === :akm ?
            "Adao-Kolesar-Morales" : "HC1 (for comparison only; undersized here)")
    m.is_iv && @printf(io, "  first stage   : %.6f\n", m.first_stage)
    println(io)
    _print_coef_table(io, m.coefnames, [m.beta], [m.se])
    if m.vcov_type === :hc1
        println(io)
        println(io, "  WARNING: HC1 standard errors are not valid in a shift-share")
        println(io, "  design. Adao, Kolesar & Morales (2019) report rejection of a")
        println(io, "  true null in up to 55% of placebo samples. Use vcov = :akm.")
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Balance test
# ═══════════════════════════════════════════════════════════════════════

"""
    BalanceTest

Shock-level balance of regional characteristics against the shocks,
returned by [`balance_test`](@ref).

# Fields

- `coefs`, `ses`, `t_stats`, `p_values` — one per characteristic tested, for
  the coefficient on the shock in the exposure-weighted shock-level
  regression.
- `names::Vector{String}`
- `joint_chi2::T`, `joint_df::Int`, `joint_p::T` — a Wald test that **all**
  the characteristics are balanced, accounting for their correlation across
  shocks. With one characteristic, `joint_chi2 == t_stats[1]^2` exactly.
- `n_shocks::Int` — the number of observations the test actually has.
"""
struct BalanceTest{T<:AbstractFloat}
    coefs::Vector{T}
    ses::Vector{T}
    t_stats::Vector{T}
    p_values::Vector{T}
    names::Vector{String}
    joint_chi2::T
    joint_df::Int
    joint_p::T
    n_shocks::Int
end

"""
    balance_test(design, r; controls = nothing, names = nothing) -> BalanceTest

Test whether the shocks are balanced with respect to regional
characteristics — the exogeneity check a shift-share design actually
requires.

`r` is an `N`-vector or `N × M` matrix of regional characteristics that
should be unrelated to the shocks if the design is valid: pre-trends,
predetermined demographics, anything a reader would worry correlates with
the shocks. `controls` is an optional `K × P` matrix of *shock-level*
controls.

# What is being tested, and why it is at this level

The identifying assumption is quasi-random assignment of the **shocks**,
not of the shares. So the test belongs at shock level: form the
exposure-weighted average `r̄_k` of each residualised characteristic, then
regress it on the shock `g_k`, weighting by average exposure `ŝ_k`
(Borusyak, Hull & Jaravel 2022, §4.3). Conventional
heteroskedasticity-robust standard errors *are* valid for that regression,
which is the practical payoff of the shock-level equivalence in
[`shock_aggregate`](@ref).

A test at regional level would be answering a different and largely
irrelevant question, because the shares are permitted to be endogenous in
this framework.

# Reading the result

`n_shocks` is the real sample size. A balance test over 12 shocks is weak
evidence of balance whatever its p-value, and the field is reported so that
it cannot be overlooked.

# Measured size

Rejection of a true null, 2,000 replications, `N = 300`, with the
characteristic carrying a sector-mix component so that it is structured
rather than iid:

| Effective shocks | 10% | 5% | 1% | joint, 4 characteristics at 5% |
|---:|---:|---:|---:|---:|
| 99.9 | 0.108 | 0.056 | 0.010 | 0.088 |
| 50.0 | 0.115 | 0.064 | 0.018 | 0.147 |
| 20.0 | 0.170 | 0.109 | 0.043 | 0.324 |
| 13.0 | 0.278 | 0.170 | 0.072 | — |
| 9.1 | 0.335 | 0.232 | 0.114 | — |

**The test is correctly sized only when the effective number of shocks is
large.** At 100 it is almost exact; by 20 it rejects a true null twice as
often as it should, and at 9 effective shocks it rejects 23% of the time at
a nominal 5%. The joint test degrades faster still, because it estimates an
`M × M` covariance across characteristics from as many observations as
there are shocks: with four characteristics and 20 shocks it rejects 32% of
the time.

So a *passed* balance test in a concentrated design is close to
uninformative, and a *failed* one may be the test failing rather than the
design. Read [`exposure_concentration`](@ref) first and treat this test as
diagnostic rather than decisive when the effective count is low.

# References

Borusyak, Hull & Jaravel (2022), *Review of Economic Studies* 89(1),
181–213, §4.3.
"""
function balance_test(d::ShiftShareDesign{T},
                      r::Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}};
                      controls::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
                      names::Union{Nothing, AbstractVector{<:AbstractString}} = nothing) where {T<:AbstractFloat}
    N, K = size(d.W)
    R = r isa AbstractVector ? reshape(convert(Vector{T}, r), :, 1) :
                               convert(Matrix{T}, r)
    size(R, 1) == N || throw(DimensionMismatch(
        "balance_test: the characteristics have $(size(R, 1)) rows, the design has $N regions."))
    all(isfinite, R) || throw(ArgumentError("balance_test: `r` contains a non-finite entry."))
    M = size(R, 2)

    nms = isnothing(names) ? ["characteristic[$m]" for m in 1:M] : String.(collect(names))
    length(nms) == M || throw(DimensionMismatch(
        "balance_test: `names` has $(length(nms)) entries for $M characteristics."))

    # Shock-level design: intercept, the shock, and any shock-level controls.
    Q = hcat(ones(T, K), d.g)
    if !isnothing(controls)
        Qc = convert(Matrix{T}, controls)
        size(Qc, 1) == K || throw(DimensionMismatch(
            "balance_test: `controls` has $(size(Qc, 1)) rows but there are $K shocks. " *
            "These are shock-level controls, one row per shock."))
        Q = hcat(Q, Qc)
    end
    K > size(Q, 2) || throw(ArgumentError(
        "balance_test: $K shocks cannot support a shock-level regression with " *
        "$(size(Q, 2)) parameters. There is no residual variation to test with."))
    qn = vcat(["(intercept)", "shock"],
              ["shock control[$j]" for j in 1:(size(Q, 2) - 2)])
    check_design_rank(Q, qn, "balance_test"; context = "the shock-level design")

    w = copy(d.s_bar)
    sw = sqrt.(w)
    Qw = Q .* sw                       # weighted least squares by average exposure

    # Residualise each characteristic regionally, then aggregate to shocks.
    R̃ = _resid_on(R, d.Z)
    Rbar = reduce(hcat, (_aggregate(d, @view R̃[:, m]) for m in 1:M))

    # Weighted regression of each aggregate on the shock-level design. The
    # coefficient of interest is the second, on the shock itself.
    coefs = zeros(T, M); ses = zeros(T, M)
    scores = zeros(T, K, M)
    QwtQw_inv = inv(Symmetric(Qw' * Qw))
    for m in 1:M
        yb = @view Rbar[:, m]
        ybw = yb .* sw
        θ = QwtQw_inv * (Qw' * ybw)
        resid = ybw .- Qw * θ
        # HC1-style robust covariance for the weighted regression.
        S = (Qw .* resid)' * (Qw .* resid)
        V = QwtQw_inv * S * QwtQw_inv
        dof = K - size(Q, 2)
        V .*= T(K) / T(dof)
        coefs[m] = θ[2]
        ses[m] = sqrt(max(zero(T), V[2, 2]))
        # Score for the shock coefficient, used for the joint test.
        e2 = QwtQw_inv[2, :]
        scores[:, m] = (Qw * e2) .* resid .* sqrt(T(K) / T(dof))
    end

    t = similar(coefs)
    p = similar(coefs)
    for m in 1:M
        t[m] = ses[m] > zero(T) ? coefs[m] / ses[m] : T(NaN)
        p[m] = isnan(t[m]) ? one(T) : T(2 * (1 - _normcdf(abs(Float64(t[m])))))
    end

    # Joint Wald test across characteristics, using the empirical
    # cross-product of the scores so that correlation between them is
    # respected rather than assumed away.
    Vjoint = scores' * scores
    chi2, df, pj = if M == 1
        v = Vjoint[1, 1]
        c2 = v > zero(T) ? coefs[1]^2 / v : T(NaN)
        (c2, 1, isnan(c2) ? one(T) : T(_chi2_sf(Float64(c2), 1)))
    else
        Vs = Symmetric(Vjoint)
        if rank(Vs) < M
            (T(NaN), M, one(T))
        else
            c2 = dot(coefs, Vs \ coefs)
            (c2, M, T(_chi2_sf(Float64(c2), M)))
        end
    end

    return BalanceTest{T}(coefs, ses, t, p, nms, chi2, df, pj, K)
end

Base.show(io::IO, b::BalanceTest) = begin
    println(io, "BalanceTest: $(length(b.coefs)) characteristic(s) against $(b.n_shocks) shocks")
    println(io)
    _print_coef_table(io, b.names, b.coefs, b.ses)
    println(io)
    @printf(io, "  joint Wald chi2(%d) = %.4f, p = %.4f\n",
            b.joint_df, b.joint_chi2, b.joint_p)
    if b.n_shocks < 50
        println(io, "  WARNING: only $(b.n_shocks) shocks. This test is correctly sized")
        println(io, "  only when the effective number of shocks is large: measured at")
        println(io, "  nominal 5% it rejects a true null 5.6% of the time at 100")
        println(io, "  effective shocks, 10.9% at 20, and 23.2% at 9. The joint test")
        println(io, "  degrades faster still, since it estimates a covariance across")
        println(io, "  characteristics from as many observations as there are shocks.")
        println(io, "  Check exposure_concentration first; see docs/AUDIT.md.")
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Rotemberg weights
# ═══════════════════════════════════════════════════════════════════════

"""
    RotembergWeights

The decomposition of a shift-share IV estimate into one just-identified
estimate per shock, returned by [`rotemberg_weights`](@ref).

Goldsmith-Pinkham, Sorkin & Swift (2020, Proposition 3) show

```
β̂ = Σ_k α̂_k β̂_k,   β̂_k = (Z̃_k'T̃)⁻¹ Z̃_k'ỹ,
α̂_k = g_k Z̃_k'T̃ / Σ_j g_j Z̃_j'T̃,   Σ_k α̂_k = 1
```

with `Z̃_k` the `k`-th exposure share column, residualised on the controls.
Both the unit sum and the decomposition itself hold exactly.

**Negative weights are expected, not a defect.** The paper says so
explicitly: `α̂_k` can be negative, and under constant effects this poses no
conceptual problem. The fields `n_negative` and `negative_mass` report them
rather than hiding them.

The weights measure sensitivity to misspecification: if shock `k` is
invalid, `α̂_k` governs how much of that invalidity reaches the overall
estimate. The practical use is to report the few shocks carrying the most
weight, because those are the ones whose exogeneity has to be defended.

# Fields

- `alpha::Vector{T}`, `beta_k::Vector{T}` — weights and just-identified estimates.
- `beta_overall::T` — the shift-share IV estimate, `Σ_k α̂_k β̂_k`.
- `alpha_sum::T` — should be `1`; carried so the identity is visible.
- `n_negative::Int`, `negative_mass::T` — count and total of negative weights.
- `order::Vector{Int}` — shocks by `|α̂_k|`, largest first.
- `undefined::Vector{Int}` — shocks whose first stage is exactly zero, whose
  `β̂_k` is `NaN` and whose weight is zero.
"""
struct RotembergWeights{T<:AbstractFloat}
    alpha::Vector{T}
    beta_k::Vector{T}
    beta_overall::T
    alpha_sum::T
    n_negative::Int
    negative_mass::T
    order::Vector{Int}
    undefined::Vector{Int}
end

"""
    rotemberg_weights(y, treatment, design) -> RotembergWeights

Decompose a shift-share IV estimate into the per-shock just-identified
estimates it averages, with the weights from Goldsmith-Pinkham, Sorkin &
Swift (2020, Proposition 3).

Use it to find which shocks actually identify the estimate. A design with
hundreds of industries whose weight is concentrated on three of them is a
three-industry study, and those three are the ones a referee should be
asking about.

# References

Goldsmith-Pinkham, Sorkin & Swift (2020), *American Economic Review*
110(8), 2586–2624.
"""
function rotemberg_weights(y::AbstractVector{<:Real},
                           treatment::AbstractVector{<:Real},
                           d::ShiftShareDesign{T}) where {T<:AbstractFloat}
    N, K = size(d.W)
    yv = convert(Vector{T}, y); tv = convert(Vector{T}, treatment)
    length(yv) == N || throw(DimensionMismatch(
        "rotemberg_weights: `y` has $(length(yv)) entries, the design has $N regions."))
    length(tv) == N || throw(DimensionMismatch(
        "rotemberg_weights: `treatment` has $(length(tv)) entries, the design has $N regions."))

    ỹ = _resid_on(yv, d.Z)
    T̃ = _resid_on(tv, d.Z)
    Z̃ = _resid_on(d.W, d.Z)

    zt = vec(Z̃' * T̃)             # Z̃_k' T̃
    zy = vec(Z̃' * ỹ)             # Z̃_k' ỹ

    D = dot(d.g, zt)
    abs(D) > zero(T) || throw(ArgumentError(
        "rotemberg_weights: the weights share a denominator of exactly zero " *
        "(the overall first stage vanishes), so the decomposition is undefined."))

    alpha = (d.g .* zt) ./ D
    beta_k = similar(alpha)
    undef_idx = Int[]
    for k in 1:K
        if zt[k] == zero(T)
            beta_k[k] = T(NaN)
            push!(undef_idx, k)
        else
            beta_k[k] = zy[k] / zt[k]
        end
    end

    # Sum the contributions, skipping shocks with a zero first stage: their
    # weight is exactly zero, so they contribute nothing, but 0 * NaN is NaN.
    total = zero(T)
    for k in 1:K
        zt[k] == zero(T) && continue
        total += alpha[k] * beta_k[k]
    end

    neg = count(<(zero(T)), alpha)
    negmass = sum((a for a in alpha if a < zero(T)); init = zero(T))
    ord = sortperm(abs.(alpha); rev = true)

    return RotembergWeights{T}(alpha, beta_k, total, sum(alpha),
                               neg, negmass, ord, undef_idx)
end

Base.show(io::IO, r::RotembergWeights) = begin
    K = length(r.alpha)
    println(io, "RotembergWeights: $K shocks")
    @printf(io, "  overall estimate      : %.6f\n", r.beta_overall)
    @printf(io, "  weights sum to        : %.12f\n", r.alpha_sum)
    @printf(io, "  negative weights      : %d (mass %.4f)\n", r.n_negative, r.negative_mass)
    isempty(r.undefined) ||
        println(io, "  zero first stage at   : $(r.undefined) (beta undefined, weight 0)")
    ntop = min(5, K)
    println(io, "  largest |weight| shocks:")
    for j in 1:ntop
        k = r.order[j]
        @printf(io, "    shock %-4d  alpha %9.4f   beta_k %9.4f\n", k, r.alpha[k], r.beta_k[k])
    end
    r.n_negative > 0 && println(io,
        "  NOTE: negative weights are expected here (Goldsmith-Pinkham,\n" *
        "  Sorkin & Swift 2020, Prop. 3), not a sign of an error.")
end

# ═══════════════════════════════════════════════════════════════════════
# Spatial correlation
# ═══════════════════════════════════════════════════════════════════════

"""
    ShiftShareMoran

Spatial autocorrelation of a shift-share design's regressor, residuals and
exposure shares, returned by [`shift_share_moran`](@ref).

# Fields

- `regressor::MoranResult{T}` — Moran's I of the constructed regressor.
- `residual::Union{Nothing, MoranResult{T}}` — of the fitted residuals, when
  a fit was supplied.
- `share_I::Vector{T}`, `share_p::Vector{T}` — Moran's I and its p-value for
  each exposure share column.
- `worst_share::Int` — the share column with the largest `|I|`.
"""
struct ShiftShareMoran{T<:AbstractFloat}
    regressor::MoranResult{T}
    residual::Union{Nothing, MoranResult{T}}
    share_I::Vector{T}
    share_p::Vector{T}
    worst_share::Int
end

"""
    shift_share_moran(design, W_spatial; fit = nothing, kwargs...) -> ShiftShareMoran

Moran's I for the shift-share regressor, for each exposure share column,
and optionally for a fit's residuals.

`W_spatial` is a spatial weight matrix over the regions — a different object
from the design's exposure shares, and the two should not be confused.
Keyword arguments are passed to
[`moran_global`](@ref).

# What this does and does not tell you

Spatially correlated exposure shares are the normal case — industries
cluster geographically — and finding it is not by itself a problem for a
shift-share design, whose identification runs through the shocks. What it
does tell you is that any attempt to fix this design's inference with a
*spatial* correction is addressing the wrong dependence: the residual
correlation that matters here travels through shared industry mix, and two
regions on opposite coasts with the same industries are as correlated as
neighbours. Use it to document the spatial structure, and
[`fit_shift_share`](@ref) with `vcov = :akm` to handle inference.
"""
function shift_share_moran(d::ShiftShareDesign{T},
                           Wsp::AbstractSparseMatrix{T};
                           fit::Union{Nothing, ShiftShareFit{T}} = nothing,
                           kwargs...) where {T<:AbstractFloat}
    N, K = size(d.W)
    size(Wsp, 1) == N && size(Wsp, 2) == N || throw(DimensionMismatch(
        "shift_share_moran: `W_spatial` is $(size(Wsp)) but the design has $N regions. " *
        "This argument is a spatial weight matrix over regions, not the exposure shares."))

    mreg = moran_global(d.X, Wsp; kwargs...)
    mres = isnothing(fit) ? nothing : moran_global(residuals(fit), Wsp; kwargs...)

    sI = zeros(T, K); sp = zeros(T, K)
    for k in 1:K
        col = d.W[:, k]
        if maximum(col) - minimum(col) <= zero(T)
            sI[k] = T(NaN); sp[k] = one(T)
        else
            mk = moran_global(col, Wsp; kwargs...)
            sI[k] = mk.I_stat; sp[k] = mk.p_value
        end
    end
    finite = [k for k in 1:K if isfinite(sI[k])]
    worst = isempty(finite) ? 0 : finite[argmax(abs.(sI[finite]))]

    return ShiftShareMoran{T}(mreg, mres, sI, sp, worst)
end

Base.show(io::IO, m::ShiftShareMoran) = begin
    println(io, "ShiftShareMoran")
    @printf(io, "  regressor : I = %+.4f, p = %.4f\n",
            m.regressor.I_stat, m.regressor.p_value)
    isnothing(m.residual) ||
        @printf(io, "  residuals : I = %+.4f, p = %.4f\n",
                m.residual.I_stat, m.residual.p_value)
    if m.worst_share > 0
        @printf(io, "  most autocorrelated share column: %d (I = %+.4f, p = %.4f)\n",
                m.worst_share, m.share_I[m.worst_share], m.share_p[m.worst_share])
    end
    println(io, "  (spatially clustered shares are normal; they do not justify a")
    println(io, "  spatial correction here. See fit_shift_share with vcov = :akm.)")
end

end # module
