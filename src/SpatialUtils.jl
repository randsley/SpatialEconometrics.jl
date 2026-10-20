module SpatialUtils

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

export LogDetMethod, ExactLU, ChebyshevApprox,
       FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
       precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, brent_maximize,
       panel_spatial_lag!, panel_spatial_lag,
       within_transform!, within_transform,
       effective_dims, has_indiv_fe, has_time_fe,
       _normcdf, _print_coef_table

# ─── Log-determinant methods ──────────────────────────────────────────

abstract type LogDetMethod end
struct ExactLU <: LogDetMethod end

struct ChebyshevApprox{R<:AbstractRNG} <: LogDetMethod
    order::Int
    n_probes::Int
    rng::R
    function ChebyshevApprox(; order::Int=6, n_probes::Int=0, rng::AbstractRNG=Random.default_rng())
        @assert order >= 2 "Chebyshev order must be >= 2"
        return new{typeof(rng)}(order, n_probes, rng)
    end
end

# ─── Fixed effects ────────────────────────────────────────────────────

abstract type FixedEffectsSpec end
struct NoFE     <: FixedEffectsSpec end
struct IndivFE  <: FixedEffectsSpec end
struct TimeFE   <: FixedEffectsSpec end
struct TwoWayFE <: FixedEffectsSpec end

effective_dims(N::Int, T_per::Int, ::NoFE)     = (N, T_per)
effective_dims(N::Int, T_per::Int, ::IndivFE)  = (N, T_per - 1)
effective_dims(N::Int, T_per::Int, ::TimeFE)   = (N - 1, T_per)
effective_dims(N::Int, T_per::Int, ::TwoWayFE) = (N - 1, T_per - 1)

has_indiv_fe(::FixedEffectsSpec) = false
has_indiv_fe(::IndivFE)          = true
has_indiv_fe(::TwoWayFE)         = true

has_time_fe(::FixedEffectsSpec)  = false
has_time_fe(::TimeFE)            = true
has_time_fe(::TwoWayFE)          = true

# ─── Chebyshev log-determinant ────────────────────────────────────────

function precompute_chebyshev_traces(W::AbstractSparseMatrix{T}, method::ChebyshevApprox) where {T<:AbstractFloat}
    N = size(W, 1)
    q = method.order
    τ = zeros(T, q + 1)
    τ[1] = T(N)
    τ[2] = tr(W)

    if method.n_probes == 0
        T_prev = sparse(one(T) * I, N, N)
        T_curr = copy(W)
        for k in 2:q
            T_next = 2 .* (W * T_curr) .- T_prev
            τ[k + 1] = tr(T_next)
            T_prev = T_curr
            T_curr = T_next
        end
    else
        S = method.n_probes
        V0 = Matrix{T}(undef, N, S)
        rand!(method.rng, V0, (-one(T), one(T)))
        V_prev = copy(V0)
        V_curr = W * V0
        V_next = similar(V0)
        for k in 2:q
            mul!(V_next, W, V_curr)
            @. V_next = 2 * V_next - V_prev
            τ[k + 1] = dot(V0, V_next) / S
            copyto!(V_prev, V_curr)
            copyto!(V_curr, V_next)
        end
    end
    return τ
end

@inline function chebyshev_logdet(ρ::R, τ::Vector{T}) where {R<:Real, T<:AbstractFloat}
    q = length(τ) - 1
    N_nodes = q + 1
    logdet_val = zero(promote_type(R, T))
    for k in 0:q
        ck = zero(promote_type(R, T))
        for j in 1:N_nodes
            θ_j = T(π) * (j - T(0.5)) / N_nodes
            x_j = cos(θ_j)
            ck += log(one(R) - ρ * x_j) * cos(k * θ_j)
        end
        ck *= (T(2) / N_nodes)
        k == 0 && (ck *= T(0.5))
        logdet_val += ck * τ[k + 1]
    end
    return logdet_val
end

# ─── Neumann series traces ────────────────────────────────────────────

function precompute_neumann_traces(
    W::AbstractSparseMatrix{T};
    max_power::Int=60,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    N = size(W, 1)
    μ = zeros(T, max_power + 1)
    μ[1] = one(T)
    μ[2] = tr(W) / N

    V0 = Matrix{T}(undef, N, n_probes)
    rand!(rng, V0, (-one(T), one(T)))
    V_curr = W * V0
    V_next = similar(V0)

    for m in 2:max_power
        mul!(V_next, W, V_curr)
        μ[m + 1] = dot(V0, V_next) / (N * n_probes)
        copyto!(V_curr, V_next)
    end
    return μ
end

# ─── Brent's method maximizer ─────────────────────────────────────────

function brent_maximize(f::F, a::T, b::T; tol::T=T(1e-7), maxiter::Int=200) where {F, T<:AbstractFloat}
    cg = T(0.3819660112501051)
    x = w = v = a + cg * (b - a)
    fx = fw = fv = -f(x)
    d = e = zero(T)

    for _ in 1:maxiter
        xm = T(0.5) * (a + b)
        tol1 = tol * abs(x) + T(1e-10)
        tol2 = T(2) * tol1
        abs(x - xm) <= (tol2 - T(0.5) * (b - a)) && return x, -fx

        p = q = r = zero(T)
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = T(2) * (q - r)
            q > zero(T) ? (p = -p) : (q = -q)
            r = e; e = d
        end

        if abs(p) < abs(T(0.5) * q * r) && p > q * (a - x) && p < q * (b - x)
            d = p / q
            u = x + d
            if (u - a) < tol2 || (b - u) < tol2
                d = x < xm ? tol1 : -tol1
            end
        else
            e = (x < xm ? b : a) - x
            d = cg * e
        end

        u = abs(d) >= tol1 ? x + d : x + (d > zero(T) ? tol1 : -tol1)
        fu = -f(u)

        if fu <= fx
            u >= x ? (a = x) : (b = x)
            v, fv = w, fw; w, fw = x, fx; x, fx = u, fu
        else
            u < x ? (a = u) : (b = u)
            if fu <= fw || w == x
                v, fv = w, fw; w, fw = u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    return x, -fx
end

# ─── Panel spatial lag ────────────────────────────────────────────────

function panel_spatial_lag!(
    WY::AbstractVector{T},
    W::AbstractSparseMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_per::Int
) where {T<:AbstractFloat}
    Y_mat = reshape(y, N, T_per)
    WY_mat = reshape(WY, N, T_per)
    mul!(WY_mat, W, Y_mat)
    return WY
end

function panel_spatial_lag(W::AbstractSparseMatrix{T}, X::AbstractMatrix{T}, N::Int, T_per::Int) where {T<:AbstractFloat}
    WX = similar(X)
    for k in 1:size(X, 2)
        @views panel_spatial_lag!(WX[:, k], W, X[:, k], N, T_per)
    end
    return WX
end

# ─── Within-transformation (FE demeaning) ─────────────────────────────

within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::NoFE) where {T<:AbstractFloat} = z

function within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::IndivFE) where {T<:AbstractFloat}
    Z = reshape(z, N, T_per)
    @inbounds for i in 1:N
        row_sum = zero(T)
        for t in 1:T_per; row_sum += Z[i, t]; end
        row_mean = row_sum / T_per
        for t in 1:T_per; Z[i, t] -= row_mean; end
    end
    return z
end

function within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::TimeFE) where {T<:AbstractFloat}
    Z = reshape(z, N, T_per)
    @inbounds for t in 1:T_per
        col_sum = zero(T)
        for i in 1:N; col_sum += Z[i, t]; end
        col_mean = col_sum / N
        for i in 1:N; Z[i, t] -= col_mean; end
    end
    return z
end

function within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::TwoWayFE) where {T<:AbstractFloat}
    within_transform!(z, N, T_per, IndivFE())
    within_transform!(z, N, T_per, TimeFE())
    return z
end

function within_transform(X::AbstractMatrix{T}, N::Int, T_per::Int, fe::FixedEffectsSpec) where {T<:AbstractFloat}
    X_tilde = copy(X)
    for k in 1:size(X, 2)
        @views within_transform!(X_tilde[:, k], N, T_per, fe)
    end
    return X_tilde
end

# ─── Normal CDF approximation (Abramowitz & Stegun 26.2.17) ──────────

function _normcdf(z::Real)
    x = Float64(z)
    if x < -8.0; return 0.0; end
    if x >  8.0; return 1.0; end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * exp(-x * x / 2.0) * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return x > 0.0 ? 1.0 - p : p
end

# ─── Shared show helper ──────────────────────────────────────────────

function _print_coef_table(io::IO, names::Vector{String}, coefs::Vector{T}, ses::Vector{T}) where {T<:AbstractFloat}
    @printf(io, "%-16s | %11s | %11s | %10s | %8s\n", "Parameter", "Estimate", "Std. Err", "z-stat", "P>|z|")
    println(io, "─────────────────┼─────────────┼─────────────┼────────────┼──────────────")
    for i in eachindex(coefs)
        se_i = ses[i]
        z = se_i > zero(T) ? coefs[i] / se_i : T(NaN)
        p = isnan(z) ? 1.0 : 2.0 * (1.0 - _normcdf(abs(Float64(z))))
        stars = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : p < 0.1 ? "." : ""
        @printf(io, "%-16s | %11.4f | %11.4f | %10.3f | %7.4f %s\n", names[i], coefs[i], se_i, z, p, stars)
    end
end

end # module
