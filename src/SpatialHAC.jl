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

"""
    KernelType

Kernel used to down-weight distant observations in a HAC covariance.

Subtypes: [`BartlettKernel`](@ref), [`ParzenKernel`](@ref),
[`QuadraticSpectralKernel`](@ref), [`UniformKernel`](@ref). The choice
affects the weight profile but rarely the conclusion; what matters more is
the bandwidth (`max_lag` or `dist_cutoff`).
"""
abstract type KernelType end
"""
    BartlettKernel()

Triangular kernel, `max(0, 1 - |x|)`. The Newey-West default and the
default throughout this package: weights decline linearly to zero at the
bandwidth, which guarantees a positive semi-definite covariance.

See also [`KernelType`](@ref).
"""
struct BartlettKernel          <: KernelType end
"""
    ParzenKernel()

Parzen kernel: smoother decay than [`BartlettKernel`](@ref), also
guaranteeing positive semi-definiteness. Slightly more efficient when the
true autocorrelation decays smoothly.

See also [`KernelType`](@ref).
"""
struct ParzenKernel            <: KernelType end
"""
    QuadraticSpectralKernel()

Quadratic spectral kernel. Asymptotically optimal in the rate sense
(Andrews, 1991) and has no hard truncation point: weights decay smoothly
without ever reaching exactly zero, so every lag contributes something.

See also [`KernelType`](@ref).
"""
struct QuadraticSpectralKernel <: KernelType end
"""
    UniformKernel()

Equal weight on every lag inside the bandwidth and zero outside — the
truncated, unweighted estimator.

!!! warning
    Unlike the other kernels this does **not** guarantee a positive
    semi-definite covariance matrix, so standard errors can come out
    undefined. Use it only when you have a reason to.

See also [`KernelType`](@ref).
"""
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

"""
    DistanceMetric

How distance between two units is measured when building spatial HAC
weights. Subtypes: [`EuclideanDistance`](@ref), [`HaversineDistance`](@ref).
"""
abstract type DistanceMetric end
"""
    EuclideanDistance()

Straight-line distance in the units of the coordinates, `hypot(dx, dy)`.

The default. Appropriate for projected coordinates; for raw latitude and
longitude use [`HaversineDistance`](@ref) instead, since a degree of
longitude is not a fixed distance.
"""
struct EuclideanDistance <: DistanceMetric end

"""
    HaversineDistance(; radius_km = 6371.0088)

Great-circle distance in kilometres between points given as
`(latitude, longitude)` in **degrees**.

Use this whenever coordinates are unprojected geographic degrees. The
default radius is the Earth's mean radius.

!!! note
    Column order is latitude first, then longitude, and the cutoff you pair
    it with is therefore in kilometres.

# Examples
```julia
spec = ConleySpatialHAC(coords, 150.0; metric = HaversineDistance())
```
"""
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

"""
    AbstractRobustVCOV

A robust covariance specification, passed to [`robust_vcov`](@ref),
[`robust_coeftable`](@ref) or [`spatial_effects`](@ref).

Subtypes: [`DriscollKraayVCOV`](@ref) and [`ConleySpatialHAC`](@ref).
"""
abstract type AbstractRobustVCOV end

"""
    DriscollKraayVCOV(; max_lag = nothing, kernel = BartlettKernel(), small_sample_correct = true)

Driscoll-Kraay spatially-correlated-consistent covariance for panel data.

Averages the scores across units within each period and then applies a
Newey-West correction over periods. Because the cross-sectional dependence
is handled by that averaging, it is robust to **arbitrary** spatial
correlation — no distance, coordinates or cutoff are needed — as well as to
heteroskedasticity and serial correlation.

The price is that it leans on having a reasonable number of time periods;
with very short panels the time-series correction is poorly estimated, and
[`ConleySpatialHAC`](@ref) is the better choice when you have coordinates
and few periods.

# Keywords
- `max_lag`: lag truncation. `nothing` picks a rule-of-thumb bandwidth from
  the number of periods.
- `kernel`: a [`KernelType`](@ref).
- `small_sample_correct`: apply the finite-sample degrees-of-freedom
  scaling.

# Examples
```julia
m = fit_spatial_panel(SAR(), W, X, y, N, T; fe = TwoWayFE())
robust_coeftable(m, DriscollKraayVCOV())
robust_coeftable(m, DriscollKraayVCOV(max_lag = 5, kernel = ParzenKernel()))
```

# References
Driscoll, J. & Kraay, A. (1998). *Review of Economics and Statistics*,
80(4).
"""
struct DriscollKraayVCOV{K<:KernelType} <: AbstractRobustVCOV
    max_lag::Union{Nothing, Int}
    kernel::K
    small_sample_correct::Bool
    function DriscollKraayVCOV(; max_lag::Union{Nothing, Int}=nothing, kernel::K=BartlettKernel(), small_sample_correct::Bool=true) where {K<:KernelType}
        return new{K}(max_lag, kernel, small_sample_correct)
    end
end

"""
    ConleySpatialHAC(coords, dist_cutoff; max_time_lag = 0, metric = EuclideanDistance(),
                     space_kernel = BartlettKernel(), time_kernel = BartlettKernel(),
                     psd_fix = true)

Conley spatial HAC covariance: correlation decaying with geographic
distance.

Pairs of units closer than `dist_cutoff` contribute to the covariance with
a kernel weight based on their distance; pairs beyond it contribute
nothing. Unlike [`DriscollKraayVCOV`](@ref) this needs coordinates, but in
exchange it works with few time periods and with pure cross-sections.

# Arguments
- `coords`: `N x 2` matrix of positions, **positional**. With
  [`HaversineDistance`](@ref) the columns are latitude then longitude in
  degrees.
- `dist_cutoff`: distance beyond which units are treated as independent,
  **positional**, in the units of `metric` (kilometres for Haversine).

# Keywords
- `max_time_lag`: periods of serial correlation to allow; `0` for a
  cross-section.
- `metric`: a [`DistanceMetric`](@ref).
- `space_kernel`, `time_kernel`: [`KernelType`](@ref)s for the spatial and
  temporal dimensions.
- `psd_fix`: project the result to the nearest positive semi-definite
  matrix if the kernel weighting leaves it indefinite. Keep this on unless
  you are diagnosing the covariance itself.

!!! note
    `dist_cutoff` is a judgement call and the result is sensitive to it:
    too small and spatial correlation is missed, too large and the estimate
    is noisy. Report standard errors at a couple of cutoffs.

# Examples
```julia
m = fit_spatial(SAR(), W, X, y)
robust_coeftable(m, ConleySpatialHAC(coords, 3.0))

ConleySpatialHAC(coords, 150.0; metric = HaversineDistance(), max_time_lag = 2)
```

# References
Conley, T. (1999). GMM estimation with cross sectional dependence.
*Journal of Econometrics*, 92(1).
"""
struct ConleySpatialHAC{T<:AbstractFloat, M<:DistanceMetric, KS<:KernelType, KT<:KernelType} <: AbstractRobustVCOV
    coords::Matrix{T}; dist_cutoff::T; max_time_lag::Int; metric::M; space_kernel::KS; time_kernel::KT; psd_fix::Bool
    function ConleySpatialHAC(coords::AbstractMatrix{T}, dist_cutoff::Real; max_time_lag::Int=0, metric::M=EuclideanDistance(), space_kernel::KS=BartlettKernel(), time_kernel::KT=BartlettKernel(), psd_fix::Bool=true) where {T<:AbstractFloat, M<:DistanceMetric, KS<:KernelType, KT<:KernelType}
        return new{T, M, KS, KT}(Matrix{T}(coords), T(dist_cutoff), max_time_lag, metric, space_kernel, time_kernel, psd_fix)
    end
end

"""
    vcov_driscoll_kraay(Z, residuals, N, T_per; bread_inv = nothing, spec = DriscollKraayVCOV())

Driscoll-Kraay covariance from a score matrix directly.

The low-level routine behind [`robust_vcov`](@ref). Prefer that on a fitted
model, which builds `Z` and `bread_inv` correctly for you — in particular
it maps maximum-likelihood scores through the Jacobian of the spatial
transformation, which a raw call here will not do.

# Arguments
- `Z`: `(N*T_per) x K` regressor or score matrix, stacked with units
  varying fastest within each period.
- `residuals`: matching residual vector.
- `N`, `T_per`: units and periods.

# Keywords
- `bread_inv`: the `(X'X)^{-1}`-analogue for the sandwich. `nothing` uses
  the OLS form.
- `spec`: a [`DriscollKraayVCOV`](@ref) carrying the kernel and bandwidth.
"""
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

"""
    vcov_conley(Z, residuals, N, T_per, spec; bread_inv = nothing)

Conley spatial HAC covariance from a score matrix directly.

The low-level routine behind [`robust_vcov`](@ref); see
[`vcov_driscoll_kraay`](@ref) for why the fitted-model path is usually the
one you want. `spec` is a [`ConleySpatialHAC`](@ref), which carries the
coordinates, cutoff, metric and kernels.
"""
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

"""
    RobustInferenceTable

Coefficient table pairing robust with model-based standard errors, returned
by [`robust_coeftable`](@ref). Pretty-prints via `show`.

# Fields
- `estimator_name`: which robust estimator was used.
- `coefnames`, `coef`: parameter names and estimates.
- `se_robust`, `se_baseline`: robust and model-based standard errors. The
  ratio between them is the diagnostic worth reading.
- `z_robust`: `coef ./ se_robust`.
- `vcov_robust`: the full robust covariance.
"""
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
