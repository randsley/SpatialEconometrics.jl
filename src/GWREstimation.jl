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

abstract type BandwidthType end
struct FixedBandwidth{T<:AbstractFloat} <: BandwidthType
    h::T
end
struct AdaptiveBandwidth <: BandwidthType
    k::Int
end

# ─── Kernel functions ─────────────────────────────────────────────────

abstract type KernelFunction end
struct GaussianKernel    <: KernelFunction end
struct BisquareKernel    <: KernelFunction end
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
    idxs, dists = knn(tree, point, bw.k + 1, true)  # +1 includes self
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

        # Hat matrix diagonal element: S[i,i] = x_i' (Xw'Xw)^{-1} x_iw * w_i
        # Efficient: R = F.R, then S_ii = ||R^{-T} * (x_i * sqrt(w_i))||²
        xi_w = X[i, :] .* sqrt(w[i])
        hat_diag[i] = sum(abs2, F.R \ xi_w)
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
                                 k_min::Int=max(size(X, 2) + 2, 20),
                                 k_max::Int=min(size(X, 1), 500)) where {T}
    K = size(X, 2)
    k_min = max(k_min, K + 2)

    function aicc_at_k(k::Int)
        bw = AdaptiveBandwidth(k)
        _, _, _, _, _, _, _, aicc = _gwr_fit_at_bandwidth(X, y, coords, tree, bw, kernel)
        return aicc
    end

    # Golden-section search over integer k
    gr = 0.6180339887498949
    a = k_min; b = k_max
    c = round(Int, b - gr * (b - a))
    d = round(Int, a + gr * (b - a))
    fc = aicc_at_k(c)
    fd = aicc_at_k(d)

    for _ in 1:50
        (b - a) <= 3 && break
        if fc < fd
            b = d
            d = c; fd = fc
            c = round(Int, b - gr * (b - a))
            c = clamp(c, a, b)
            fc = aicc_at_k(c)
        else
            a = c
            c = d; fc = fd
            d = round(Int, a + gr * (b - a))
            d = clamp(d, a, b)
            fd = aicc_at_k(d)
        end
    end

    best_k = fc < fd ? c : d
    return AdaptiveBandwidth(best_k)
end

function _select_bandwidth_aicc(X::Matrix{T}, y::Vector{T}, coords::Matrix{T}, tree::KDTree,
                                 kernel::KernelFunction, ::Type{FixedBandwidth};
                                 h_min::T=T(0.0), h_max::T=T(0.0)) where {T}
    N = size(X, 1)
    # Determine range from data extent
    dists_all = [sqrt(sum((coords[1, :] .- coords[j, :]).^2)) for j in 1:N]
    max_dist = maximum(dists_all)
    if h_min <= zero(T); h_min = max_dist * T(0.05); end
    if h_max <= zero(T); h_max = max_dist * T(0.8); end

    function aicc_at_h(h::T)
        bw = FixedBandwidth(h)
        _, _, _, _, _, _, _, aicc = _gwr_fit_at_bandwidth(X, y, coords, tree, bw, kernel)
        return aicc
    end

    gr = T(0.6180339887498949)
    a = h_min; b = h_max
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = aicc_at_h(c)
    fd = aicc_at_h(d)

    for _ in 1:50
        (b - a) < T(1e-4) * max_dist && break
        if fc < fd
            b = d; d = c; fd = fc
            c = b - gr * (b - a)
            fc = aicc_at_h(c)
        else
            a = c; c = d; fc = fd
            d = a + gr * (b - a)
            fd = aicc_at_h(d)
        end
    end

    best_h = fc < fd ? c : d
    return FixedBandwidth(best_h)
end

# ─── Main GWR fit function ───────────────────────────────────────────

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

function fit_mgwr(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    coords::AbstractMatrix{T};
    kernel::KernelFunction=BisquareKernel(),
    varnames::Union{Nothing, Vector{String}}=nothing,
    max_iter::Int=50,
    tol::T=T(1e-4)
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = isnothing(varnames) ? vcat(["(Intercept)"], ["x$i" for i in 1:K_raw]) : vcat(["(Intercept)"], varnames)

    coords_mat = Matrix{T}(coords)
    tree = _build_tree(coords_mat)

    # Initialize with GWR
    m_gwr = fit_gwr(X, y, coords; kernel=kernel, varnames=isnothing(varnames) ? nothing : varnames)
    coef_local = copy(m_gwr.coef_local)
    bandwidths = fill(m_gwr.bandwidth, K)

    converged = false
    iter = 0

    for outer_iter in 1:max_iter
        iter = outer_iter
        max_change = zero(T)

        for k in 1:K
            # Partial residual for variable k
            partial_y = Vector{T}(y)
            for j in 1:K
                j == k && continue
                partial_y .-= X_aug[:, j] .* coef_local[:, j]
            end

            # Optimize bandwidth for this variable
            x_k = X_aug[:, k:k]
            bw_k = _select_bandwidth_aicc(x_k, partial_y, coords_mat, tree, kernel, AdaptiveBandwidth;
                                           k_min=max(3, 20), k_max=min(N, 500))

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

        if max_change < tol
            converged = true
            break
        end
    end

    fitted = vec(sum(X_aug .* coef_local; dims=2))
    residuals = Vector{T}(y) .- fitted
    RSS = dot(residuals, residuals)
    TSS = sum((y .- mean(y)).^2)
    R2 = one(T) - RSS / TSS
    sigma2 = RSS / N
    AICc = N * log(RSS / N) + N  # approximate
    coef_global = vec(mean(coef_local; dims=1))

    # SE from last iteration
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

    return MGWRModel{T}(
        coef_local, se_local, coef_global,
        Vector{T}(y), X_aug, coords_mat,
        residuals, fitted, bandwidths, kernel,
        AICc, R2, sigma2, vnames, iter, converged
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
    @printf(io, "%-14s | %10s | %10s | %10s | %10s | %8s\n", "Variable", "Mean β", "SD β", "Min β", "Max β", "BW (k)")
    println(io, "───────────────┼────────────┼────────────┼────────────┼────────────┼──────────")
    for k in eachindex(m.coefnames)
        col = @view m.coef_local[:, k]
        @printf(io, "%-14s | %10.4f | %10.4f | %10.4f | %10.4f | %8.0f\n", m.coefnames[k], mean(col), std(col), minimum(col), maximum(col), m.bandwidths[k])
    end
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "R² = %.4f    σ² = %.4f    AICc = %.2f    N = %d\n", m.R2, m.sigma2, m.AICc, length(m.y))
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
