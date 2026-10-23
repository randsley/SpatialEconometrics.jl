module GWREstimation

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI
using NearestNeighbors

import ..SpatialUtils: _normcdf, _print_coef_table

export BandwidthType, FixedBandwidth, AdaptiveBandwidth,
       KernelFunction, GaussianKernel, BisquareKernel, ExponentialKernel,
       GWRModel, MGWRModel,
       fit_gwr, fit_mgwr

# ─── Bandwidth types ──────────────────────────────────────────────────

"""
    BandwidthType

How the GWR neighbourhood around each location is sized. Subtypes:
[`FixedBandwidth`](@ref) and [`AdaptiveBandwidth`](@ref).

The bandwidth is the single most consequential GWR choice: too small and
the local coefficients chase noise, too large and they flatten towards the
global regression.
"""
abstract type BandwidthType end
"""
    FixedBandwidth(h)

A constant kernel bandwidth `h`, in the units of the coordinates, used at
every location.

Appropriate when units are spread evenly. Where density varies, a fixed
radius means sparse regions contribute few observations to their local
fit — sometimes fewer than there are parameters — so
[`AdaptiveBandwidth`](@ref) is usually safer with irregular data.

# Examples
```julia
fit_gwr(X, y, coords; bandwidth = FixedBandwidth(2.5))
```
"""
struct FixedBandwidth{T<:AbstractFloat} <: BandwidthType
    h::T
end
"""
    AdaptiveBandwidth(k)

A nearest-neighbour bandwidth: each local regression uses its `k` closest
observations, so the radius expands in sparse areas and contracts in dense
ones.

The more robust default for irregularly spaced data, since every local fit
gets the same number of observations. `k` must comfortably exceed the
number of parameters.

# Examples
```julia
fit_gwr(X, y, coords; bandwidth = AdaptiveBandwidth(30))
```
"""
struct AdaptiveBandwidth <: BandwidthType
    k::Int
end

# ─── Kernel functions ─────────────────────────────────────────────────

"""
    KernelFunction

Weighting applied to neighbours within a GWR bandwidth. Subtypes:
[`GaussianKernel`](@ref), [`BisquareKernel`](@ref),
[`ExponentialKernel`](@ref).
"""
abstract type KernelFunction end
"""
    GaussianKernel()

Gaussian weighting, `exp(-0.5 (d/h)^2)`.

Weights decline smoothly and never reach zero, so every observation
contributes to every local fit. That makes the coefficient surface smooth
but means the "local" regression is never strictly local.

See also [`BisquareKernel`](@ref).
"""
struct GaussianKernel    <: KernelFunction end
"""
    BisquareKernel()

Bisquare weighting, `(1 - (d/h)^2)^2` inside the bandwidth and exactly zero
outside.

The default here and the usual GWR choice: the hard cut-off makes each
local fit genuinely local and keeps the computation sparse.

See also [`GaussianKernel`](@ref), [`ExponentialKernel`](@ref).
"""
struct BisquareKernel    <: KernelFunction end
"""
    ExponentialKernel()

Exponential weighting, `exp(-d/h)`. Decays faster near the origin than
[`GaussianKernel`](@ref) and, like it, has no hard cut-off.
"""
struct ExponentialKernel <: KernelFunction end

@inline function kernel_weight(::GaussianKernel, d::T, h::T) where {T<:AbstractFloat}
    return exp(-T(0.5) * (d / h)^2)
end

@inline function kernel_weight(::BisquareKernel, d::T, h::T) where {T<:AbstractFloat}
    u = d / h
    return u < one(T) ? (one(T) - u^2)^2 : zero(T)
end

@inline function kernel_weight(::ExponentialKernel, d::T, h::T) where {T<:AbstractFloat}
    return exp(-d / h)
end

# ─── GWR result type ─────────────────────────────────────────────────

"""
    GWRModel

A fitted geographically weighted regression, returned by
[`fit_gwr`](@ref). Implements the `StatsAPI.RegressionModel` interface.

Rather than one coefficient vector, GWR estimates a separate regression at
every location, so the headline output is the `N x K` surface
`coef_local`.

# Fields
- `coef_local`, `se_local`: `N x K` local coefficients and their standard
  errors, columns ordered as `coefnames`.
- `coef_global`: column means of `coef_local`, a summary only.
- `y`, `X`, `coords`, `residuals`, `fitted`: the data and the fit. `X`
  includes the intercept column that was added for you.
- `local_R2`: goodness of fit of each local regression.
- `bandwidth`, `bandwidth_type`, `kernel`: the neighbourhood actually used.
- `AICc`, `R2`, `sigma2`: global fit measures.
- `hat_trace`: `tr(S)`, the effective number of parameters.
- `coefnames`: `"(Intercept)"` followed by the covariate names.

!!! warning
    The local standard errors are not adjusted for the fact that `N`
    regressions were run on overlapping data, so pointwise `t`-statistics
    overstate significance. Read the surface as description, and use
    `plot_gwr_coefficients` to look at it.
"""
struct GWRModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    coef_local::Matrix{T}       # N × K local coefficients
    se_local::Matrix{T}         # N × K local standard errors
    coef_global::Vector{T}      # K global (mean) coefficients
    y::Vector{T}
    X::Matrix{T}
    coords::Matrix{T}           # N × 2
    residuals::Vector{T}
    fitted::Vector{T}
    local_R2::Vector{T}
    bandwidth::Float64
    bandwidth_type::BandwidthType
    kernel::KernelFunction
    AICc::T
    R2::T
    sigma2::T
    hat_trace::T                # tr(S) for effective DOF
    coefnames::Vector{String}
end

"""
    MGWRModel

A fitted multiscale geographically weighted regression, returned by
[`fit_mgwr`](@ref).

Differs from [`GWRModel`](@ref) in carrying `bandwidths`, one per
covariate, instead of a single bandwidth.

# Fields
- `coef_local`, `se_local`: `N x K` local coefficients and standard errors.
- `coef_global`: column means of `coef_local`.
- `y`, `X`, `coords`, `residuals`, `fitted`: the data and the fit.
- `bandwidths`: the selected bandwidth per fitted coefficient, aligned with
  `coefnames` (intercept first). This is the substantive result — a large
  value means that relationship is close to constant in space, a small one
  that it varies locally.
- `kernel`, `R2`, `coefnames`: as for [`GWRModel`](@ref).
- `enp`: `tr(R_k)` for each fitted coefficient, the effective number of
  parameters that covariate's own bandwidth buys. Aligned with
  `coefnames`. This is the quantity that makes two MGWR fits comparable,
  and the per-covariate counterpart of `hat_trace`.
- `hat_trace`: `tr(S) = sum(enp)`, the effective number of parameters of
  the whole fit.
- `enp_se`: the Monte Carlo standard error of each `enp`, zero when the
  degrees of freedom were computed exactly and `NaN` when they were not
  computed at all. Under `:stochastic` it is not decoration: `AICc`
  inherits the same noise, so a difference of a few AICc units between two
  fits may be the estimator rather than the models.
- `df_method`: `:exact`, `:stochastic` or `:none`, recording how the
  degrees of freedom were obtained.
- `AICc`, `sigma2`: penalised by `hat_trace`, so directly comparable with
  the same fields on a [`GWRModel`](@ref).
- `iterations`, `converged`: state of the back-fitting loop. **Check
  `converged`** before interpreting anything.

!!! note
    `enp`, `hat_trace` and `AICc` are `NaN` when the degrees-of-freedom
    accumulator was skipped — see the `inference` keyword of
    [`fit_mgwr`](@ref) — because an MGWR fit carries no usable information
    criterion without it.
"""
struct MGWRModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    coef_local::Matrix{T}
    se_local::Matrix{T}
    coef_global::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    coords::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    bandwidths::Vector{Float64}
    kernel::KernelFunction
    AICc::T
    R2::T
    sigma2::T
    hat_trace::T               # tr(S) = sum(enp); NaN if not accumulated
    enp::Vector{T}             # tr(R_k) per coefficient
    enp_se::Vector{T}          # Monte Carlo SE of enp; zero when exact
    df_method::Symbol          # :exact, :stochastic or :none
    coefnames::Vector{String}
    iterations::Int
    converged::Bool
end

# ─── Core GWR computation ────────────────────────────────────────────

function _build_tree(coords::AbstractMatrix{T}) where {T}
    return KDTree(permutedims(coords))  # KDTree expects D×N
end

function _get_bandwidth_distance(tree::KDTree, coords::AbstractMatrix{T}, i::Int, bw::AdaptiveBandwidth) where {T}
    point = coords[i, :]
    # +1 includes self, but never ask the tree for more neighbours than it
    # holds: knn throws on k > N, so AdaptiveBandwidth(N) -- which the
    # bandwidth search used to put in range -- was an error rather than the
    # natural 'every point' case.
    nq = min(bw.k + 1, size(coords, 1))
    idxs, dists = knn(tree, point, nq, true)
    return T(dists[end])
end

function _get_weights(tree::KDTree, coords::AbstractMatrix{T}, i::Int, bw::FixedBandwidth{T}, kernel::KernelFunction) where {T}
    N = size(coords, 1)
    point = coords[i, :]
    idxs = inrange(tree, point, bw.h)
    w = zeros(T, N)
    for j in idxs
        d = sqrt(sum((coords[i, :] .- coords[j, :]).^2))
        w[j] = kernel_weight(kernel, d, bw.h)
    end
    return w
end

function _get_weights(tree::KDTree, coords::AbstractMatrix{T}, i::Int, bw::AdaptiveBandwidth, kernel::KernelFunction) where {T}
    N = size(coords, 1)
    h_i = _get_bandwidth_distance(tree, coords, i, bw)
    point = coords[i, :]
    idxs = inrange(tree, point, h_i)
    w = zeros(T, N)
    for j in idxs
        d = sqrt(sum((coords[i, :] .- coords[j, :]).^2))
        w[j] = kernel_weight(kernel, d, h_i)
    end
    return w
end

function _local_regression(X::Matrix{T}, y::Vector{T}, w::Vector{T}) where {T}
    W_diag = Diagonal(sqrt.(w))
    Xw = W_diag * X
    yw = W_diag * y
    qr_Xw = qr(Xw)
    β = qr_Xw \ yw
    return β
end

function _gwr_fit_at_bandwidth(X::Matrix{T}, y::Vector{T}, coords::Matrix{T}, tree::KDTree,
                                bw::BandwidthType, kernel::KernelFunction) where {T}
    N, K = size(X)
    coef_local = zeros(T, N, K)
    hat_diag = zeros(T, N)
    residuals = zeros(T, N)
    fitted = zeros(T, N)

    for i in 1:N
        w = _get_weights(tree, coords, i, bw, kernel)
        W_sqrt = Diagonal(sqrt.(w))
        Xw = W_sqrt * X
        yw = W_sqrt * y

        F = qr(Xw)
        β_i = F \ yw
        coef_local[i, :] = β_i
        fitted[i] = dot(X[i, :], β_i)
        residuals[i] = y[i] - fitted[i]

        # Hat matrix diagonal: S[i,i] = w_i * x_i' (X'W_i X)^{-1} x_i.
        # With Xw = W^{1/2}X and R from its QR, X'W_i X = R'R, so
        # (X'W_i X)^{-1} = R^{-1}R^{-T} and S_ii = ||R^{-T} x_i sqrt(w_i)||².
        # That is R' \ xi_w, NOT R \ xi_w: the latter forms (RR')^{-1} and
        # inflates tr(S) past N on a badly scaled X, which then collapses
        # the N - tr(S) divisor below to its floor of 1.
        xi_w = X[i, :] .* sqrt(w[i])
        hat_diag[i] = sum(abs2, F.R' \ xi_w)
    end

    tr_S = sum(hat_diag)
    RSS = dot(residuals, residuals)
    TSS = sum((y .- mean(y)).^2)
    sigma2 = RSS / max(N - tr_S, one(T))
    R2 = one(T) - RSS / TSS
    AICc = N * log(RSS / N) + N * (N + tr_S) / max(N - 2 - tr_S, one(T))

    return coef_local, residuals, fitted, tr_S, RSS, R2, sigma2, AICc
end

# ─── Bandwidth selection via AICc golden-section search ───────────────

function _select_bandwidth_aicc(X::Matrix{T}, y::Vector{T}, coords::Matrix{T}, tree::KDTree,
                                 kernel::KernelFunction, ::Type{AdaptiveBandwidth};
                                 k_min::Int=size(X, 2) + 2,
                                 k_max::Int=min(size(X, 1) - 1, 500)) where {T}
    N, K = size(X)
    k_min = clamp(max(k_min, K + 2), 1, N - 1)
    k_max = clamp(k_max, k_min, N - 1)

    aicc_at(k) = _gwr_fit_at_bandwidth(X, y, coords, tree, AdaptiveBandwidth(k), kernel)[8]

    # The AICc curve in k is not unimodal. On the Columbus data it reverses
    # direction eleven times, and the golden-section search this replaced
    # returned k = 43 where the grid minimum is k = 23 -- a bandwidth nearly
    # twice too wide, and 2.3 AICc units worse. Golden section is only valid
    # on a unimodal objective, so scan instead: exhaustively over a short
    # range, and coarse-then-local over a long one, where the curve is
    # smoother because each step in k moves the neighbourhood far less.
    span = k_max - k_min + 1
    best_k = k_min
    best_a = aicc_at(k_min)

    if span <= 48
        for k in (k_min + 1):k_max
            a = aicc_at(k)
            if a < best_a
                best_a = a
                best_k = k
            end
        end
        return AdaptiveBandwidth(best_k)
    end

    stride = cld(span, 24)
    for k in (k_min + stride):stride:k_max
        a = aicc_at(k)
        if a < best_a
            best_a = a
            best_k = k
        end
    end
    for k in max(k_min, best_k - stride):min(k_max, best_k + stride)
        k == best_k && continue
        a = aicc_at(k)
        if a < best_a
            best_a = a
            best_k = k
        end
    end
    return AdaptiveBandwidth(best_k)
end
function _select_bandwidth_aicc(X::Matrix{T}, y::Vector{T}, coords::Matrix{T}, tree::KDTree,
                                 kernel::KernelFunction, ::Type{FixedBandwidth};
                                 h_min::T=T(0.0), h_max::T=T(0.0)) where {T}
    # Extent from the bounding box. The previous version measured distances
    # from coords[1, :] only, so a first point near the middle of the study
    # area understated the extent and capped h_max below the useful range.
    lo = vec(minimum(coords; dims = 1))
    hi = vec(maximum(coords; dims = 1))
    extent = sqrt(sum(abs2, hi .- lo))
    if h_min <= zero(T); h_min = extent * T(0.02); end
    if h_max <= zero(T); h_max = extent; end
    h_min = max(h_min, eps(T))
    h_max = max(h_max, h_min * (one(T) + eps(T)))

    aicc_at(h) = _gwr_fit_at_bandwidth(X, y, coords, tree, FixedBandwidth(h), kernel)[8]

    # As in the adaptive case the objective is not unimodal, so bracket the
    # minimum on a grid first and only then refine, where unimodality is a
    # local claim rather than a global one. The grid is geometric because h
    # acts multiplicatively on the kernel.
    M = 40
    ratio = (h_max / h_min)^(one(T) / (M - 1))
    grid = T[h_min * ratio^(m - 1) for m in 1:M]
    best_m = 1
    best_a = aicc_at(grid[1])
    for m in 2:M
        a = aicc_at(grid[m])
        if a < best_a
            best_a = a
            best_m = m
        end
    end

    a = grid[max(1, best_m - 1)]
    b = grid[min(M, best_m + 1)]
    gr = T(0.6180339887498949)
    c = b - gr * (b - a); fc = aicc_at(c)
    d = a + gr * (b - a); fd = aicc_at(d)
    for _ in 1:40
        (b - a) < T(1e-4) * extent && break
        if fc < fd
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = aicc_at(c)
        else
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = aicc_at(d)
        end
    end

    h_ref = fc < fd ? c : d
    return FixedBandwidth(min(fc, fd) < best_a ? h_ref : grid[best_m])
end

# ─── Main GWR fit function ───────────────────────────────────────────

"""
    fit_gwr(X, y, coords; bandwidth = nothing, kernel = BisquareKernel(), varnames = nothing)

Fit a geographically weighted regression: a separate weighted least squares
at every location.

Each local fit weights observations by their distance from the target
location, so the result is a coefficient *surface* rather than a single
vector. This describes how a relationship varies over space; it does not
model spatial dependence in the error, which is what
`fit_spatial` and `fit_spatial_error` are for.

# Arguments
- `X`: `N x K` covariates, **without** an intercept — one is added.
- `y`: response of length `N`.
- `coords`: `N x 2` matrix of locations.

# Keywords
- `bandwidth`: a [`FixedBandwidth`](@ref) or [`AdaptiveBandwidth`](@ref).
  `nothing` (the default) selects an adaptive bandwidth by minimizing
  corrected AIC, which costs an extra search but is the sensible default.
- `kernel`: a [`KernelFunction`](@ref).
- `varnames`: covariate names; defaults to `x1, x2, ...`.

# Returns
A [`GWRModel`](@ref).

# Examples
```julia
m = fit_gwr(X, y, coords; varnames = ["income", "density"])
m.bandwidth
m.coef_local[:, 2]                        # the surface for "income"

fit_gwr(X, y, coords; bandwidth = AdaptiveBandwidth(30), kernel = GaussianKernel())
plot_gwr_coefficients(m, "income")        # needs a Makie backend
```

!!! warning
    Local standard errors ignore the multiplicity of running `N`
    overlapping regressions, so pointwise significance is overstated.

See also [`fit_mgwr`](@ref), which lets each covariate have its own
bandwidth.

# References
Fotheringham, A. S., Brunsdon, C. & Charlton, M. (2002). *Geographically
Weighted Regression*. Wiley.
"""
function fit_gwr(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    coords::AbstractMatrix{T};
    bandwidth::Union{Nothing, BandwidthType}=nothing,
    kernel::KernelFunction=BisquareKernel(),
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    @assert size(coords) == (N, 2) "coords must be N×2"
    @assert length(y) == N "y must have N elements"

    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = isnothing(varnames) ? vcat(["(Intercept)"], ["x$i" for i in 1:K_raw]) : vcat(["(Intercept)"], varnames)

    coords_mat = Matrix{T}(coords)
    tree = _build_tree(coords_mat)

    # Bandwidth selection
    if isnothing(bandwidth)
        bw = _select_bandwidth_aicc(X_aug, Vector{T}(y), coords_mat, tree, kernel, AdaptiveBandwidth)
    else
        bw = bandwidth
    end

    coef_local, residuals, fitted, tr_S, RSS, R2, sigma2, AICc =
        _gwr_fit_at_bandwidth(X_aug, Vector{T}(y), coords_mat, tree, bw, kernel)

    # Local standard errors
    se_local = zeros(T, N, K)
    for i in 1:N
        w = _get_weights(tree, coords_mat, i, bw, kernel)
        W_sqrt = Diagonal(sqrt.(w))
        Xw = W_sqrt * X_aug
        XwXw_inv = inv(Symmetric(Xw' * Xw))
        for k in 1:K
            se_local[i, k] = sqrt(max(zero(T), sigma2 * XwXw_inv[k, k]))
        end
    end

    # Local R²
    local_R2 = zeros(T, N)
    ȳ = mean(y)
    for i in 1:N
        w = _get_weights(tree, coords_mat, i, bw, kernel)
        tss_i = dot(w, (y .- ȳ).^2)
        rss_i = dot(w, residuals.^2)
        local_R2[i] = tss_i > zero(T) ? one(T) - rss_i / tss_i : zero(T)
    end

    coef_global = vec(mean(coef_local; dims=1))
    bw_val = bw isa AdaptiveBandwidth ? Float64(bw.k) : Float64(bw.h)

    return GWRModel{T}(
        coef_local, se_local, coef_global,
        Vector{T}(y), X_aug, coords_mat,
        residuals, fitted, local_R2,
        bw_val, bw, kernel,
        AICc, R2, sigma2, tr_S, vnames
    )
end

# ─── MGWR (Multiscale GWR via backfitting) ───────────────────────────

"""
    _mgwr_weight_cache(X_aug, bandwidths, coords, tree, kernel)

The frozen kernel weights, as `K` vectors of sparse rows plus each local
regression's denominator `x_k' W_i x_k`.

Once the bandwidths stop moving these never change, so the KD-tree queries
and kernel evaluations are worth doing once and reusing. Storage is
`O(K N bw)` -- 48 MiB at `N = 2000` with a bandwidth of 500 and three
coefficients, against 128 MiB for one of the dense `N x N` operators that
[`_mgwr_backfit_operator`](@ref) needs `K + 1` of.
"""
function _mgwr_weight_cache(X_aug::Matrix{T}, bandwidths::Vector{Float64},
                            coords::Matrix{T}, tree::KDTree,
                            kernel::KernelFunction) where {T}
    N, K = size(X_aug)
    idx = [Vector{Vector{Int}}(undef, N) for _ in 1:K]
    val = [Vector{Vector{T}}(undef, N) for _ in 1:K]
    den = [zeros(T, N) for _ in 1:K]

    for k in 1:K
        bw_k = AdaptiveBandwidth(round(Int, bandwidths[k]))
        xk = @view X_aug[:, k]
        for i in 1:N
            w = _get_weights(tree, coords, i, bw_k, kernel)
            nz = findall(!iszero, w)
            idx[k][i] = nz
            # Store w_ij * x_jk, which is the only combination used below.
            val[k][i] = T[w[j] * xk[j] for j in nz]
            d = zero(T)
            @inbounds for j in nz
                d += w[j] * xk[j]^2
            end
            den[k][i] = d
        end
    end
    return idx, val, den
end

"""
    _mgwr_apply_smoother!(f, v, X_aug, idx, val, den; tol, max_iter)

Apply the converged back-fitting smoother to `v`, writing the `k`-th additive
component `R_k v` into `f[:, k]`, so that `sum(f; dims = 2)` is `S v`.

This is the same fixed point as [`_mgwr_backfit_operator`](@ref) reaches, but
evaluated at one vector instead of built as a matrix: the iteration carries
`K` vectors rather than `K` dense `N x N` operators, which is what lets it
run at an `N` where the exact accumulator will not fit. Cost is
`O(max_iter K N bw)` with no allocation beyond `f`.

Returns `(iters, converged)`.
"""
function _mgwr_apply_smoother!(f::Matrix{T}, v::Vector{T}, X_aug::Matrix{T},
                               idx, val, den;
                               tol::T=T(1e-4), max_iter::Int=250) where {T}
    N, K = size(X_aug)
    fill!(f, zero(T))
    partial = Vector{T}(undef, N)
    iters = 0
    converged = false

    for it in 1:max_iter
        iters = it
        max_change = zero(T)
        for k in 1:K
            # partial = v - sum_{j != k} f_j, the residual this component is
            # fitted to, exactly as in the coefficient loop.
            copyto!(partial, v)
            for j in 1:K
                j == k && continue
                @inbounds @views partial .-= f[:, j]
            end

            xk = @view X_aug[:, k]
            ik = idx[k]; vk = val[k]; dk = den[k]
            @inbounds for i in 1:N
                dk[i] <= T(1e-12) && (f[i, k] = zero(T); continue)
                acc = zero(T)
                nz = ik[i]; wv = vk[i]
                for t in eachindex(nz)
                    acc += wv[t] * partial[nz[t]]
                end
                new = xk[i] * acc / dk[i]
                d = abs(new - f[i, k])
                d > max_change && (max_change = d)
                f[i, k] = new
            end
        end
        if max_change < tol
            converged = true
            break
        end
    end
    return iters, converged
end

"""
    _mgwr_enp_hutchinson(X_aug, bandwidths, coords, tree, kernel; probes, rng, tol, max_iter)

Estimate `tr(R_k)` for every `k` by Hutchinson probing, with the Monte Carlo
standard error of each estimate.

For Rademacher `v`, `E[v' R_k v] = tr(R_k)`, because the off-diagonal terms
have mean zero. One pass of the back-fitting iteration returns every
component `R_k v` at once, so a single probe estimates the whole `enp` vector
rather than just its sum -- the per-coefficient effective parameters, which
are what MGWR is usually read for.

The estimate is unbiased but random, so the standard error travels with it:
`AICc` built on it inherits that noise, and seeding `rng` is what makes a
run reproducible. It gives no help with `se_local`, which needs a row norm
per observation rather than a trace.

Validated against a back-fitting fixed point solved exactly as an
`(NK) x (NK)` linear system: unbiased to within Monte Carlo error, with
`enp_se` slightly optimistic at low probe counts (see `fit_mgwr`'s
`probes` keyword).

Returns `(enp, enp_se, converged)`.
"""
function _mgwr_enp_hutchinson(X_aug::Matrix{T}, bandwidths::Vector{Float64},
                              coords::Matrix{T}, tree::KDTree,
                              kernel::KernelFunction;
                              probes::Int=200, rng::AbstractRNG=Random.default_rng(),
                              tol::T=T(1e-4), max_iter::Int=250) where {T}
    N, K = size(X_aug)
    idx, val, den = _mgwr_weight_cache(X_aug, bandwidths, coords, tree, kernel)

    f = zeros(T, N, K)
    v = Vector{T}(undef, N)
    s1 = zeros(T, K)
    s2 = zeros(T, K)
    all_converged = true

    for _ in 1:probes
        @inbounds for i in 1:N
            v[i] = rand(rng, Bool) ? one(T) : -one(T)
        end
        _, ok = _mgwr_apply_smoother!(f, v, X_aug, idx, val, den;
                                      tol = tol, max_iter = max_iter)
        all_converged &= ok
        for k in 1:K
            t = dot(v, @view f[:, k])
            s1[k] += t
            s2[k] += t * t
        end
    end

    enp = s1 ./ probes
    # Standard error of the mean over probes. Clamped at zero because the
    # two-pass form can go slightly negative on round-off.
    varp = max.(s2 ./ probes .- enp .^ 2, zero(T))
    enp_se = probes > 1 ? sqrt.(varp ./ (probes - 1)) : fill(T(NaN), K)
    return enp, enp_se, all_converged
end
"""
    _mgwr_backfit_operator(X_aug, bandwidths, coords, tree, kernel; tol, max_iter)

The converged back-fitting smoother, as `K` transposed operators `R_k'` and
the squared row norms of each coefficient operator `B_k`. Yu et al. (2020).

At the fixed point the `k`-th additive component is `f_k = R_k y`, where

    R_k = S^(k) (I - sum_{j != k} R_j),    S^(k) = diag(x_k) C_k

and row `i` of `C_k` is `(x_k' W_i x_k)^-1 (w_i .* x_k)'`. The recursion
depends only on the frozen bandwidths and on `x_k` -- not on `y`, and not on
the fitted coefficients -- so it runs after the fit and cannot perturb it.
That independence is the whole reason it is a separate phase: it also
converges more slowly, having started from zero rather than from a GWR fit.

Returns `(Rt, Brow2, iters, converged)`.
"""
function _mgwr_backfit_operator(X_aug::Matrix{T}, bandwidths::Vector{Float64},
                                coords::Matrix{T}, tree::KDTree,
                                kernel::KernelFunction;
                                tol::T=T(1e-4), max_iter::Int=250) where {T}
    N, K = size(X_aug)
    # Rt[k] holds R_k', transposed because every read below wants a *row* of
    # R_k or of M and column-major storage makes the transposed column the
    # contiguous one. The diagonals alone will not do: forming R_k needs the
    # full off-diagonal content of every other R_j, which is what makes this
    # O(K N^2) in memory and why fit_mgwr gates it on N.
    Rt = [zeros(T, N, N) for _ in 1:K]
    Mt = Matrix{T}(undef, N, N)
    brow = Vector{T}(undef, N)
    # Brow2[i, k] = ||B_k[i, :]||^2, which is what the standard errors need
    # and R_k alone cannot give: R_k is diag(x_k) B_k, so recovering B_k from
    # it divides by x_ik.
    Brow2 = zeros(T, N, K)
    iters = 0
    converged = false

    for it in 1:max_iter
        iters = it
        max_r_change = zero(T)

        for k in 1:K
            fill!(Mt, zero(T))
            for j in 1:K
                j == k && continue
                Mt .-= Rt[j]
            end
            @inbounds for i in 1:N
                Mt[i, i] += one(T)
            end

            bw_k = AdaptiveBandwidth(round(Int, bandwidths[k]))
            xk = @view X_aug[:, k]
            for i in 1:N
                w = _get_weights(tree, coords, i, bw_k, kernel)
                denom = zero(T)
                @inbounds for j in 1:N
                    denom += w[j] * xk[j]^2
                end

                # Row i of B_k is row i of C_k times M. w vanishes outside
                # the kernel's support, which keeps this O(N * bw) per row
                # rather than O(N^2).
                fill!(brow, zero(T))
                if denom > T(1e-12)
                    @inbounds for j in 1:N
                        c = w[j] * xk[j]
                        iszero(c) && continue
                        cd = c / denom
                        @views brow .+= cd .* Mt[:, j]
                    end
                end
                Brow2[i, k] = dot(brow, brow)

                xki = xk[i]
                rcol = @view Rt[k][:, i]
                @inbounds for t in 1:N
                    v = xki * brow[t]
                    dR = abs(v - rcol[t])
                    dR > max_r_change && (max_r_change = dR)
                    rcol[t] = v
                end
            end
        end

        if max_r_change < tol
            converged = true
            break
        end
    end

    return Rt, Brow2, iters, converged
end

"""
    fit_mgwr(X, y, coords; kernel = BisquareKernel(), varnames = nothing,
             max_iter = 50, tol = 1e-4)

Fit a multiscale GWR, giving every covariate its own bandwidth.

Standard [`fit_gwr`](@ref) forces one bandwidth on all relationships, which
is a real restriction: some processes genuinely operate at a regional scale
and others very locally. MGWR relaxes it by back-fitting, cycling through
the covariates and re-selecting each bandwidth with the others held fixed
until the coefficients stop moving.

The selected `bandwidths` are themselves the interesting output — a
bandwidth near `N` says that relationship is effectively global.

# Arguments
As [`fit_gwr`](@ref): `X` without an intercept, `y`, and `N x 2` `coords`.

# Keywords
- `kernel`: a [`KernelFunction`](@ref).
- `varnames`: covariate names.
- `max_iter`, `tol`: back-fitting iteration cap and convergence tolerance.
  Back-fitting runs in two phases: bandwidths are re-selected each sweep
  until they stop moving, then held fixed while the coefficients converge.
  The second phase is where the sweeps go -- Columbus settles its
  bandwidths after 5 sweeps and then needs about 100 more -- so `max_iter`
  has to be generous. Those later sweeps are cheap, since the AICc search
  is what costs.
- `inference`: how to obtain the degrees of freedom, which is what
  identifies `enp`, `hat_trace` and `AICc`.
  * `true` (default): exact, by accumulating the smoother operators. Needs
    `K + 1` dense `N x N` matrices, hence `max_inference_n`.
  * `:stochastic`: Hutchinson probing. `O(N)` in memory, so it has no size
    gate, and it estimates the whole `enp` vector rather than just its sum.
    The estimate is unbiased but random -- see `enp_se` -- and `se_local`
    stays approximate, because a standard error needs a row norm per
    observation and no trace estimator provides one.
  * `false`: skip it. `enp`, `hat_trace` and `AICc` are `NaN`.
- `max_inference_n`: fall back from exact to `NaN` above this `N` (default
  `2000`), warning rather than erroring. Pass `inference = :stochastic` to
  get degrees of freedom there instead.
- `probes`, `rng`: probe count and random stream for `:stochastic` (default
  `200`). Seed `rng` for a reproducible `AICc`; `enp_se` falls as
  `1/sqrt(probes)`. The default is 200 rather than the 30-50 usually quoted
  for Hutchinson because `enp_se` is mildly optimistic at small probe
  counts: against an exactly solved fixed point its z-scores have standard
  deviation 1.28 at 50 probes and 1.04 at 200, so nominal 95% intervals
  cover about 91%. The estimate itself is unbiased at either count -- it is
  the width that is understated, because a quadratic form in Rademacher
  variables is not Gaussian and its variance is itself estimated from the
  same few probes.

# Returns
An [`MGWRModel`](@ref).

!!! warning
    Back-fitting is not guaranteed to converge, and `converged` now requires
    both that the bandwidths settled and that the coefficients reached
    `tol`. Check `m.converged` before reading the coefficients; if it is
    `false`, raise `max_iter` or loosen `tol`. Convergence is slow when the
    covariates are spatially correlated with each other, because the
    additive components then compete for the same variation: the contraction
    factor on Columbus is about 0.94 per sweep.

# Examples
```julia
m = fit_mgwr(X, y, coords; varnames = ["income", "density"])
m.converged || @warn "back-fitting did not converge"
m.bandwidths               # aligned with m.coefnames, intercept first
```

# References
Fotheringham, A. S., Yang, W. & Kang, W. (2017). Multiscale
geographically weighted regression. *Annals of the American Association of
Geographers*, 107(6).

Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W. & Wolf, L. J.
(2020). Inference in multiscale geographically weighted regression.
*Geographical Analysis*, 52(1) - the degrees-of-freedom accumulator.
"""
function fit_mgwr(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    coords::AbstractMatrix{T};
    kernel::KernelFunction=BisquareKernel(),
    varnames::Union{Nothing, Vector{String}}=nothing,
    max_iter::Int=250,
    tol::T=T(1e-4),
    inference::Union{Bool, Symbol}=true,
    max_inference_n::Int=2000,
    probes::Int=200,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = isnothing(varnames) ? vcat(["(Intercept)"], ["x$i" for i in 1:K_raw]) : vcat(["(Intercept)"], varnames)

    coords_mat = Matrix{T}(coords)
    tree = _build_tree(coords_mat)

    # The accumulator below needs K+1 dense N x N operators, so it is gated
    # on N. Above the gate we report NaN rather than an unpenalised number:
    # AICc without tr(S) is the tr(S) = 0 case, and putting that beside a
    # GWR AICc makes MGWR look better by exactly the penalty it is missing.
    inference isa Symbol && inference !== :stochastic &&
        throw(ArgumentError("fit_mgwr: inference must be true, false or " *
                            ":stochastic, got :$inference"))
    want_exact = inference === true
    want_stoch = inference === :stochastic
    do_inference = want_exact && N <= max_inference_n
    if want_exact && !do_inference
        mib = (K + 1) * N^2 * sizeof(T) / 1024^2
        cost = mib < 1024 ? "$(round(mib; digits = 1)) MiB" :
                            "$(round(mib / 1024; digits = 2)) GiB"
        @warn "fit_mgwr: skipping the exact degrees-of-freedom accumulator " *
              "because N = $N exceeds max_inference_n = $max_inference_n. " *
              "enp, hat_trace and AICc will be NaN. Either raise " *
              "max_inference_n -- it needs about $cost -- or pass " *
              "inference = :stochastic, which estimates them in O(N) memory."
    end

    # Initialize with GWR
    m_gwr = fit_gwr(X, y, coords; kernel=kernel, varnames=isnothing(varnames) ? nothing : varnames)
    coef_local = copy(m_gwr.coef_local)
    bandwidths = fill(m_gwr.bandwidth, K)

    converged = false
    iter = 0
    # Phase 1 re-selects every bandwidth each sweep; phase 2 holds them fixed
    # and lets the coefficients converge. Splitting them matters three ways:
    # the AICc search is ~95% of a sweep's cost, so the many sweeps the
    # coefficients actually need are only affordable without it; the
    # smoother S is a fixed linear operator only once the bandwidths stop
    # moving, which is what makes tr(S) well defined; and re-selecting
    # forever lets a flat AICc curve flip the argmin on numerical noise.
    frozen = false
    froze_at = 0

    for outer_iter in 1:max_iter
        iter = outer_iter
        max_change = zero(T)
        bw_changed = false

        for k in 1:K
            # Partial residual for variable k
            partial_y = Vector{T}(y)
            for j in 1:K
                j == k && continue
                partial_y .-= X_aug[:, j] .* coef_local[:, j]
            end

            # Optimize bandwidth for this variable. k_min is left at the
            # selector's statistical floor of K + 2 = 3 for these univariate
            # sweeps; it used to be pinned at 20, which bound on every
            # Columbus covariate -- their grid optima are 9 to 15 -- and so
            # reported a floor rather than a selection.
            x_k = X_aug[:, k:k]
            if frozen
                bw_k = AdaptiveBandwidth(round(Int, bandwidths[k]))
            else
                bw_k = _select_bandwidth_aicc(x_k, partial_y, coords_mat, tree,
                                              kernel, AdaptiveBandwidth)
                bw_changed |= Float64(bw_k.k) != bandwidths[k]
            end

            # Fit local regression for this variable
            coef_k_old = coef_local[:, k]
            for i in 1:N
                w = _get_weights(tree, coords_mat, i, bw_k, kernel)
                w_sqrt = sqrt.(w)
                xw = x_k[:, 1] .* w_sqrt
                yw = partial_y .* w_sqrt
                denom = dot(xw, xw)
                coef_local[i, k] = denom > T(1e-12) ? dot(xw, yw) / denom : zero(T)
            end

            bandwidths[k] = Float64(bw_k.k)
            max_change = max(max_change, maximum(abs.(coef_local[:, k] .- coef_k_old)))
        end

        if !frozen && !bw_changed
            frozen = true
            froze_at = outer_iter
        elseif frozen && max_change < tol
            converged = true
            break
        end
    end

    fitted = vec(sum(X_aug .* coef_local; dims=2))
    residuals = Vector{T}(y) .- fitted
    RSS = dot(residuals, residuals)
    TSS = sum((y .- mean(y)).^2)
    R2 = one(T) - RSS / TSS
    coef_global = vec(mean(coef_local; dims=1))

    # One rule for all of it: report degrees of freedom only when there is a
    # converged operator to take them from. Otherwise NaN -- a trace that has
    # not converged is not the trace of anything, and handing back a finite
    # number there repeats the mistake this whole exercise was about.
    did_accumulate = false
    did_estimate = false
    Rt = Vector{Matrix{T}}()
    Brow2 = zeros(T, 0, 0)
    enp_se = fill(T(NaN), K)
    if !converged && frozen
        @warn "fit_mgwr: bandwidths settled at sweep $froze_at but the " *
              "back-fitting had not converged after $iter sweeps. " *
              "Convergence is slow when the covariates are spatially " *
              "correlated with each other; raise max_iter or loosen tol."
    end
    if do_inference && !frozen
        @warn "fit_mgwr: the bandwidths never settled in $max_iter sweeps, " *
              "so there is no fixed smoother to take a trace of. enp, " *
              "hat_trace and AICc are NaN. Raise max_iter."
    end

    if want_stoch && !frozen
        @warn "fit_mgwr: the bandwidths never settled in $max_iter sweeps, " *
              "so there is no fixed smoother to probe. enp, hat_trace and " *
              "AICc are NaN. Raise max_iter."
    elseif want_stoch
        # Hutchinson probing. The smoother is applied to vectors rather than
        # built as a matrix, so this holds K vectors where the exact path
        # holds K + 1 dense N x N operators -- the whole reason it has no
        # size gate. A probe that has not converged biases the estimate
        # rather than merely adding variance, so a failure here is fatal to
        # the number in a way more probes cannot repair.
        enp_h, se_h, probes_ok = _mgwr_enp_hutchinson(
            X_aug, bandwidths, coords_mat, tree, kernel;
            probes = probes, rng = rng, tol = tol, max_iter = max_iter)
        if probes_ok
            did_estimate = true
            enp_est = enp_h
            enp_se = se_h
        else
            @warn "fit_mgwr: the smoother did not converge on at least one " *
                  "probe, which biases the estimate rather than just " *
                  "adding noise to it, so enp, hat_trace and AICc are NaN. " *
                  "Raise max_iter or loosen tol."
        end
    end

    if do_inference && frozen
        # Phase 3: the smoother, iterated on its own now that the bandwidths
        # are fixed. Keeping it out of the fitting loop is what lets the
        # coefficients come out bit-identical whether or not inference is
        # asked for -- inference changes what is reported, never the fit.
        Rt, Brow2, acc_iters, acc_converged =
            _mgwr_backfit_operator(X_aug, bandwidths, coords_mat, tree, kernel;
                                   tol = tol, max_iter = max_iter)
        did_accumulate = acc_converged
        if !acc_converged
            @warn "fit_mgwr: the smoother recursion did not converge in " *
                  "$acc_iters iterations, so enp, hat_trace and AICc are " *
                  "NaN and se_local is approximate. The operator starts " *
                  "from zero and so converges more slowly than the " *
                  "coefficients; raise max_iter or loosen tol."
        end
    end

    if did_accumulate
        # The fit is additive in the components, so S = sum_k R_k and the
        # trace decomposes. The per-term traces are the interesting half:
        # they say how many parameters each bandwidth actually bought.
        enp = T[tr(Rt[k]) for k in 1:K]
        tr_S = sum(enp)

        # Self-check, and still worth running even though the recursion
        # converged on its own criterion: that criterion says the operator
        # stopped moving, not that it is the smoother of *this* fit. If the
        # coefficient loop hit max_iter, `fitted` is not S y and the gap is
        # where it shows. It also disposes of any worry about drift.
        f_hat = zeros(T, N)
        yv = Vector{T}(y)
        for k in 1:K
            mul!(f_hat, transpose(Rt[k]), yv, one(T), one(T))
        end
        gap = maximum(abs, f_hat .- fitted)
        scale = max(sqrt(RSS / N), eps(T))
        if gap > T(0.01) * scale
            @warn "fit_mgwr: the accumulated smoother reproduces the fitted " *
                  "values only to $(gap) against a residual scale of " *
                  "$(scale), so hat_trace, AICc and se_local are " *
                  "unreliable. Check `converged`; raise max_iter or tol."
        end

        sigma2 = RSS / max(N - tr_S, one(T))
        AICc = N * log(RSS / N) + N * (N + tr_S) / max(N - 2 - tr_S, one(T))
        se_local = sqrt.(sigma2 .* Brow2)
        enp_se = zeros(T, K)
        df_method = :exact
    elseif did_estimate
        enp = enp_est
        tr_S = sum(enp)
        sigma2 = RSS / max(N - tr_S, one(T))
        AICc = N * log(RSS / N) + N * (N + tr_S) / max(N - 2 - tr_S, one(T))
        df_method = :stochastic

        # se_local has no stochastic counterpart: it needs ||B_k[i, :]||^2
        # for every i, which is N quantities rather than a trace. So it keeps
        # the approximate form, now at least scaled by a real sigma2.
        se_local = zeros(T, N, K)
        for i in 1:N
            for k in 1:K
                bw_k = AdaptiveBandwidth(round(Int, bandwidths[k]))
                w = _get_weights(tree, coords_mat, i, bw_k, kernel)
                xw_k = X_aug[:, k] .* sqrt.(w)
                denom = dot(xw_k, xw_k)
                se_local[i, k] = denom > T(1e-12) ? sqrt(sigma2 * w[i] / denom) : zero(T)
            end
        end
    else
        enp = fill(T(NaN), K)
        tr_S = T(NaN)
        sigma2 = RSS / N
        AICc = T(NaN)
        df_method = :none

        # Approximate: this treats each coefficient as if it came from one
        # weighted regression on a known partial residual, ignoring that the
        # other components were estimated too. Reached with inference = false,
        # above max_inference_n, or when the bandwidths never settled.
        se_local = zeros(T, N, K)
        for i in 1:N
            for k in 1:K
                bw_k = AdaptiveBandwidth(round(Int, bandwidths[k]))
                w = _get_weights(tree, coords_mat, i, bw_k, kernel)
                xw_k = X_aug[:, k] .* sqrt.(w)
                denom = dot(xw_k, xw_k)
                se_local[i, k] = denom > T(1e-12) ? sqrt(sigma2 * w[i] / denom) : zero(T)
            end
        end
    end

    return MGWRModel{T}(
        coef_local, se_local, coef_global,
        Vector{T}(y), X_aug, coords_mat,
        residuals, fitted, bandwidths, kernel,
        AICc, R2, sigma2, tr_S, enp, enp_se, df_method, vnames, iter, converged
    )
end

# ─── Show methods ─────────────────────────────────────────────────────

function Base.show(io::IO, m::GWRModel)
    bw_str = m.bandwidth_type isa AdaptiveBandwidth ? "Adaptive k=$(round(Int, m.bandwidth))" : "Fixed h=$(round(m.bandwidth; digits=2))"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Geographically Weighted Regression (GWR) — $bw_str")
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "%-14s | %10s | %10s | %10s | %10s\n", "Variable", "Mean β", "SD β", "Min β", "Max β")
    println(io, "───────────────┼────────────┼────────────┼────────────┼────────────")
    for k in eachindex(m.coefnames)
        col = @view m.coef_local[:, k]
        @printf(io, "%-14s | %10.4f | %10.4f | %10.4f | %10.4f\n", m.coefnames[k], mean(col), std(col), minimum(col), maximum(col))
    end
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "R² = %.4f    σ² = %.4f    AICc = %.2f    N = %d    tr(S) = %.1f\n", m.R2, m.sigma2, m.AICc, length(m.y), m.hat_trace)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

function Base.show(io::IO, m::MGWRModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Multiscale GWR (MGWR) — $(m.converged ? "Converged" : "Not converged") in $(m.iterations) iterations")
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "%-14s | %10s | %10s | %10s | %10s | %8s | %7s\n", "Variable", "Mean β", "SD β", "Min β", "Max β", "BW (k)", "ENP")
    println(io, "───────────────┼────────────┼────────────┼────────────┼────────────┼──────────┼─────────")
    for k in eachindex(m.coefnames)
        col = @view m.coef_local[:, k]
        @printf(io, "%-14s | %10.4f | %10.4f | %10.4f | %10.4f | %8.0f | %7.2f\n", m.coefnames[k], mean(col), std(col), minimum(col), maximum(col), m.bandwidths[k], m.enp[k])
    end
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "R² = %.4f    σ² = %.4f    AICc = %.2f    N = %d    tr(S) = %.2f\n", m.R2, m.sigma2, m.AICc, length(m.y), m.hat_trace)
    if m.df_method === :none
        println(io, "σ² is RSS/N and AICc unavailable — see the `inference` keyword of fit_mgwr.")
    elseif m.df_method === :stochastic
        @printf(io, "tr(S) estimated by Hutchinson probing, se = %.2f; AICc carries the same noise.\n",
                sqrt(sum(abs2, m.enp_se)))
    end
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─── StatsAPI interface ───────────────────────────────────────────────

StatsAPI.coef(m::GWRModel) = m.coef_global
StatsAPI.vcov(m::GWRModel) = cov(m.coef_local)
StatsAPI.stderror(m::GWRModel) = vec(mean(m.se_local; dims=1))
StatsAPI.nobs(m::GWRModel) = length(m.y)
StatsAPI.residuals(m::GWRModel) = m.residuals
StatsAPI.coefnames(m::GWRModel) = m.coefnames

StatsAPI.coef(m::MGWRModel) = m.coef_global
StatsAPI.vcov(m::MGWRModel) = cov(m.coef_local)
StatsAPI.stderror(m::MGWRModel) = vec(mean(m.se_local; dims=1))
StatsAPI.nobs(m::MGWRModel) = length(m.y)
StatsAPI.residuals(m::MGWRModel) = m.residuals
StatsAPI.coefnames(m::MGWRModel) = m.coefnames

end # module
