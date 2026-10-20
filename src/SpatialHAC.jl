module SpatialHAC

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI

export KernelType, BartlettKernel, ParzenKernel, QuadraticSpectralKernel, UniformKernel,
       DistanceMetric, EuclideanDistance, HaversineDistance,
       AbstractRobustVCOV, DriscollKraayVCOV, ConleySpatialHAC,
       vcov_driscoll_kraay, vcov_conley,
       RobustInferenceTable

abstract type KernelType end
struct BartlettKernel          <: KernelType end
struct ParzenKernel            <: KernelType end
struct QuadraticSpectralKernel <: KernelType end
struct UniformKernel           <: KernelType end

@inline function kernel_weight(::BartlettKernel, x::T) where {T<:AbstractFloat}
    ax = abs(x); return ax < one(T) ? (one(T) - ax) : zero(T)
end

@inline function kernel_weight(::ParzenKernel, x::T) where {T<:AbstractFloat}
    ax = abs(x)
    if ax <= T(0.5); return one(T) - T(6) * ax^2 + T(6) * ax^3
    elseif ax < one(T); return T(2) * (one(T) - ax)^3
    else; return zero(T); end
end

@inline function kernel_weight(::QuadraticSpectralKernel, x::T) where {T<:AbstractFloat}
    iszero(x) && return one(T)
    z = T(6) * T(π) * x / T(5)
    return (T(3) / z^2) * (sin(z) / z - cos(z))
end

@inline function kernel_weight(::UniformKernel, x::T) where {T<:AbstractFloat}
    return abs(x) <= one(T) ? one(T) : zero(T)
end

abstract type DistanceMetric end
struct EuclideanDistance <: DistanceMetric end

struct HaversineDistance{T<:AbstractFloat} <: DistanceMetric
    radius_km::T
    HaversineDistance(; radius_km::T=6371.0088) where {T<:AbstractFloat} = new{T}(radius_km)
end

@inline function pairwise_distance(::EuclideanDistance, x1::T, y1::T, x2::T, y2::T) where {T<:AbstractFloat}
    return hypot(x2 - x1, y2 - y1)
end

@inline function pairwise_distance(m::HaversineDistance, lat1_deg::T, lon1_deg::T, lat2_deg::T, lon2_deg::T) where {T<:AbstractFloat}
    deg2rad_f = T(π) / T(180)
    φ1 = lat1_deg * deg2rad_f; φ2 = lat2_deg * deg2rad_f
    Δφ = (lat2_deg - lat1_deg) * deg2rad_f; Δλ = (lon2_deg - lon1_deg) * deg2rad_f
    a = sin(Δφ * T(0.5))^2 + cos(φ1) * cos(φ2) * sin(Δλ * T(0.5))^2
    return T(m.radius_km) * (T(2) * atan(sqrt(max(zero(T), a)), sqrt(max(zero(T), one(T) - a))))
end

abstract type AbstractRobustVCOV end

struct DriscollKraayVCOV{K<:KernelType} <: AbstractRobustVCOV
    max_lag::Union{Nothing, Int}
    kernel::K
    small_sample_correct::Bool
    function DriscollKraayVCOV(; max_lag::Union{Nothing, Int}=nothing, kernel::K=BartlettKernel(), small_sample_correct::Bool=true) where {K<:KernelType}
        return new{K}(max_lag, kernel, small_sample_correct)
    end
end

struct ConleySpatialHAC{T<:AbstractFloat, M<:DistanceMetric, KS<:KernelType, KT<:KernelType} <: AbstractRobustVCOV
    coords::Matrix{T}; dist_cutoff::T; max_time_lag::Int; metric::M; space_kernel::KS; time_kernel::KT; psd_fix::Bool
    function ConleySpatialHAC(coords::AbstractMatrix{T}, dist_cutoff::Real; max_time_lag::Int=0, metric::M=EuclideanDistance(), space_kernel::KS=BartlettKernel(), time_kernel::KT=BartlettKernel(), psd_fix::Bool=true) where {T<:AbstractFloat, M<:DistanceMetric, KS<:KernelType, KT<:KernelType}
        return new{T, M, KS, KT}(Matrix{T}(coords), T(dist_cutoff), max_time_lag, metric, space_kernel, time_kernel, psd_fix)
    end
end

function vcov_driscoll_kraay(Z::AbstractMatrix{T}, residuals::AbstractVector{T}, N::Int, T_per::Int; bread_inv::Union{Nothing, AbstractMatrix{T}}=nothing, spec::DriscollKraayVCOV=DriscollKraayVCOV()) where {T<:AbstractFloat}
    NT, P = size(Z)
    L = isnothing(spec.max_lag) ? max(1, floor(Int, 4.0 * (T_per / 100.0)^(2 / 9))) : spec.max_lag
    L = min(L, T_per - 1)

    H_scores = zeros(T, P, T_per)
    @inbounds for t in 1:T_per
        offset = (t - 1) * N
        for p in 1:P
            s_tp = zero(T)
            for i in 1:N; s_tp += Z[offset + i, p] * residuals[offset + i]; end
            H_scores[p, t] = s_tp
        end
    end

    S_meat = H_scores * H_scores'
    Ω_l = zeros(T, P, P)
    for l in 1:L
        w_l = kernel_weight(spec.kernel, T(l) / T(L + 1))
        w_l <= zero(T) && continue
        fill!(Ω_l, zero(T))
        @inbounds for t in (l + 1):T_per
            @views ht = H_scores[:, t]; @views ht_lag = H_scores[:, t - l]
            mul!(Ω_l, ht, ht_lag', one(T), one(T))
        end
        @. S_meat += w_l * (Ω_l + Ω_l')
    end

    if spec.small_sample_correct && NT > P; S_meat .*= T(NT) / T(NT - P); end
    B_inv = isnothing(bread_inv) ? inv(Symmetric(Z' * Z)) : bread_inv
    return Symmetric(B_inv * S_meat * B_inv') |> Matrix
end

function build_spatial_neighbor_pairs(coords::Matrix{T}, cutoff::T, metric::DistanceMetric, space_kernel::KernelType) where {T<:AbstractFloat}
    N = size(coords, 1)
    cell_x = metric isa HaversineDistance ? cutoff / 110.0 : cutoff
    cell_y = cell_x
    min_x = minimum(@view coords[:, 1]); min_y = minimum(@view coords[:, 2])

    grid = Dict{Tuple{Int, Int}, Vector{Int}}()
    cell_coords = Vector{Tuple{Int, Int}}(undef, N)
    @inbounds for i in 1:N
        gx = floor(Int, (coords[i, 1] - min_x) / cell_x)
        gy = floor(Int, (coords[i, 2] - min_y) / cell_y)
        key = (gx, gy); cell_coords[i] = key
        push!(get!(Vector{Int}, grid, key), i)
    end

    I_pair = Int[]; J_pair = Int[]; W_pair = T[]
    @inbounds for i in 1:N
        xi = coords[i, 1]; yi = coords[i, 2]; gx, gy = cell_coords[i]
        for dx in -1:1, dy in -1:1
            nb_bucket = get(grid, (gx + dx, gy + dy), nothing)
            isnothing(nb_bucket) && continue
            for j in nb_bucket
                j <= i && continue
                d_ij = pairwise_distance(metric, xi, yi, coords[j, 1], coords[j, 2])
                if d_ij < cutoff
                    w_ij = kernel_weight(space_kernel, d_ij / cutoff)
                    if w_ij > zero(T)
                        push!(I_pair, i); push!(J_pair, j); push!(W_pair, w_ij)
                    end
                end
            end
        end
    end
    return I_pair, J_pair, W_pair
end

function vcov_conley(Z::AbstractMatrix{T}, residuals::AbstractVector{T}, N::Int, T_per::Int, spec::ConleySpatialHAC{T}; bread_inv::Union{Nothing, AbstractMatrix{T}}=nothing) where {T<:AbstractFloat}
    NT, P = size(Z)
    U = Array{T, 3}(undef, P, N, T_per)
    @inbounds for t in 1:T_per
        offset = (t - 1) * N
        for i in 1:N
            e_it = residuals[offset + i]
            for p in 1:P; U[p, i, t] = Z[offset + i, p] * e_it; end
        end
    end

    I_pair, J_pair, W_pair = build_spatial_neighbor_pairs(spec.coords, spec.dist_cutoff, spec.metric, spec.space_kernel)
    S_meat = zeros(T, P, P)

    @inbounds for t in 1:T_per
        for i in 1:N
            for c in 1:P, r in 1:P; S_meat[r, c] += U[r, i, t] * U[c, i, t]; end
        end
        for k in eachindex(I_pair)
            i = I_pair[k]; j = J_pair[k]; w_ij = W_pair[k]
            for c in 1:P, r in 1:P
                S_meat[r, c] += w_ij * (U[r, i, t] * U[c, j, t] + U[r, j, t] * U[c, i, t])
            end
        end
    end

    L = min(spec.max_time_lag, T_per - 1)
    for l in 1:L
        w_t = kernel_weight(spec.time_kernel, T(l) / T(L + 1))
        w_t <= zero(T) && continue
        @inbounds for t in (l + 1):T_per
            for i in 1:N
                for c in 1:P, r in 1:P
                    S_meat[r, c] += w_t * (U[r, i, t] * U[c, i, t - l] + U[r, i, t - l] * U[c, i, t])
                end
            end
        end
    end

    if spec.psd_fix
        eig = eigen(Symmetric(S_meat))
        S_meat = eig.vectors * Diagonal(max.(eig.values, zero(T))) * eig.vectors'
    end

    B_inv = isnothing(bread_inv) ? inv(Symmetric(Z' * Z)) : bread_inv
    return Symmetric(B_inv * S_meat * B_inv') |> Matrix
end

function extract_spatial_jacobian_scores(X_aug::Matrix{T}, W::AbstractSparseMatrix{T}, coef_vec::Vector{T}, ρ::T, sigma2::T, vcov_qml::Matrix{T}, N::Int, T_per::Int) where {T<:AbstractFloat}
    P_beta = size(X_aug, 2); β = coef_vec[1:P_beta]
    S_lu = lu(sparse(one(T) * I, N, N) - ρ * W)
    Xb = X_aug * β; GXb = similar(Xb)
    mul!(reshape(GXb, N, T_per), W, S_lu \ reshape(Xb, N, T_per))
    return hcat(X_aug, GXb), sigma2 .* vcov_qml
end

struct RobustInferenceTable{T<:AbstractFloat}
    estimator_name::String; coefnames::Vector{String}; coef::Vector{T}; se_robust::Vector{T}; se_baseline::Vector{T}; z_robust::Vector{T}; vcov_robust::Matrix{T}
end

function Base.show(io::IO, tab::RobustInferenceTable)
    println(io, "====================================================================================")
    println(io, "Non-Parametric Spatio-Temporal Robust Inference: $(tab.estimator_name)")
    println(io, "------------------------------------------------------------------------------------")
    @printf(io, "%-14s | %12s | %13s | %13s | %10s\n", "Parameter", "Estimate", "Robust SE", "Baseline SE", "Robust z")
    println(io, "------------------------------------------------------------------------------------")
    for i in eachindex(tab.coef)
        @printf(io, "%-14s | %12.4f | %13.4f | %13.4f | %10.2f\n", tab.coefnames[i], tab.coef[i], tab.se_robust[i], tab.se_baseline[i], tab.z_robust[i])
    end
    println(io, "====================================================================================")
end

end # module
