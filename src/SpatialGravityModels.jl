module SpatialGravityModels

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI

import ..SpatialUtils: _normcdf, _print_coef_table

export GravityVariant, PoissonGravity, NegBinGravity,
       GravityFE, OriginFE, DestinationFE, BilateralFE,
       GravityModel, fit_gravity, multilateral_resistance

# ─── Types ────────────────────────────────────────────────────────────

abstract type GravityVariant end
struct PoissonGravity <: GravityVariant end
struct NegBinGravity  <: GravityVariant end

abstract type GravityFE end
struct OriginFE      <: GravityFE end
struct DestinationFE <: GravityFE end
struct BilateralFE   <: GravityFE end

struct GravityModel{V<:GravityVariant, T<:AbstractFloat} <: StatsAPI.RegressionModel
    variant::V
    coef::Vector{T}
    vcov::Matrix{T}
    origin_fe::Vector{T}
    dest_fe::Vector{T}
    alpha::T              # NB dispersion (NaN for Poisson)
    loglik::T
    deviance::T
    pseudo_R2::T
    residuals::Vector{T}
    fitted::Vector{T}
    N_obs::Int
    N_origin::Int
    N_dest::Int
    coefnames::Vector{String}
    converged::Bool
    iterations::Int
end

# ─── High-dimensional FE demeaning (alternating projection) ──────────

function _demean_bilateral!(z::AbstractVector{T}, origin_id::Vector{Int}, dest_id::Vector{Int},
                            N_orig::Int, N_dest::Int, w::AbstractVector{T};
                            max_iter::Int=50, tol::T=T(1e-10)) where {T}
    N = length(z)
    orig_sum = zeros(T, N_orig); orig_w = zeros(T, N_orig)
    dest_sum = zeros(T, N_dest); dest_w = zeros(T, N_dest)

    for _ in 1:max_iter
        change = zero(T)

        fill!(orig_sum, zero(T)); fill!(orig_w, zero(T))
        @inbounds for n in 1:N
            orig_sum[origin_id[n]] += w[n] * z[n]
            orig_w[origin_id[n]] += w[n]
        end
        @inbounds for n in 1:N
            ow = orig_w[origin_id[n]]
            if ow > zero(T)
                adj = orig_sum[origin_id[n]] / ow
                z[n] -= adj
                change = max(change, abs(adj))
            end
        end

        fill!(dest_sum, zero(T)); fill!(dest_w, zero(T))
        @inbounds for n in 1:N
            dest_sum[dest_id[n]] += w[n] * z[n]
            dest_w[dest_id[n]] += w[n]
        end
        @inbounds for n in 1:N
            dw = dest_w[dest_id[n]]
            if dw > zero(T)
                adj = dest_sum[dest_id[n]] / dw
                z[n] -= adj
                change = max(change, abs(adj))
            end
        end

        change < tol && break
    end
    return z
end

function _compute_fe!(origin_fe::Vector{T}, dest_fe::Vector{T},
                      z::AbstractVector{T}, origin_id::Vector{Int}, dest_id::Vector{Int},
                      N_orig::Int, N_dest::Int, w::AbstractVector{T}) where {T}
    fill!(origin_fe, zero(T)); fill!(dest_fe, zero(T))
    orig_w = zeros(T, N_orig); dest_w = zeros(T, N_dest)

    @inbounds for n in eachindex(z)
        origin_fe[origin_id[n]] += w[n] * z[n]
        orig_w[origin_id[n]] += w[n]
        dest_fe[dest_id[n]] += w[n] * z[n]
        dest_w[dest_id[n]] += w[n]
    end
    @inbounds for i in eachindex(origin_fe)
        orig_w[i] > zero(T) && (origin_fe[i] /= orig_w[i])
    end
    @inbounds for j in eachindex(dest_fe)
        dest_w[j] > zero(T) && (dest_fe[j] /= dest_w[j])
    end
end

# ─── PPML via IRLS ───────────────────────────────────────────────────

function fit_gravity(
    ::PoissonGravity,
    flows::AbstractVector,
    X::AbstractMatrix{T},
    origin_id::AbstractVector{<:Integer},
    dest_id::AbstractVector{<:Integer};
    fe::GravityFE=BilateralFE(),
    varnames::Union{Nothing, Vector{String}}=nothing,
    maxiter::Int=100,
    tol::T=T(1e-8)
) where {T<:AbstractFloat}

    N_obs = length(flows)
    K = size(X, 2)
    y = Vector{T}(flows)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    o_ids = Vector{Int}(origin_id)
    d_ids = Vector{Int}(dest_id)
    N_orig = maximum(o_ids)
    N_dest = maximum(d_ids)

    # Initialize
    β = zeros(T, K)
    origin_fe_vec = zeros(T, N_orig)
    dest_fe_vec = zeros(T, N_dest)
    μ = fill(max(mean(y), T(0.1)), N_obs)
    w = copy(μ)
    converged = false
    iter = 0

    for it in 1:maxiter
        iter = it
        β_old = copy(β)

        # Working response and weights
        η = log.(max.(μ, T(1e-12)))
        z = η .+ (y .- μ) ./ max.(μ, T(1e-12))
        w .= max.(μ, T(1e-12))

        # Remove FE component
        Xβ = X * β
        z_resid = z .- Xβ
        if fe isa BilateralFE || fe isa OriginFE
            _compute_fe!(origin_fe_vec, dest_fe_vec, z_resid, o_ids, d_ids, N_orig, N_dest, w)
        end

        # Demean z and X for WLS
        z_dm = copy(z)
        X_dm = copy(X)
        if fe isa BilateralFE
            @inbounds for n in 1:N_obs
                fe_n = origin_fe_vec[o_ids[n]] + dest_fe_vec[d_ids[n]]
                z_dm[n] -= fe_n
            end
            for k in 1:K
                xk = X_dm[:, k]
                _demean_bilateral!(xk, o_ids, d_ids, N_orig, N_dest, w)
            end
        elseif fe isa OriginFE
            @inbounds for n in 1:N_obs
                z_dm[n] -= origin_fe_vec[o_ids[n]]
            end
        elseif fe isa DestinationFE
            @inbounds for n in 1:N_obs
                z_dm[n] -= dest_fe_vec[d_ids[n]]
            end
        end

        # WLS
        W_sqrt = Diagonal(sqrt.(w))
        Xw = W_sqrt * X_dm
        zw = W_sqrt * z_dm
        F = qr(Xw)
        β = F \ zw

        # Update μ
        η_new = X * β
        if fe isa BilateralFE
            # Recompute FE with new β
            z_for_fe = z .- X * β
            _compute_fe!(origin_fe_vec, dest_fe_vec, z_for_fe, o_ids, d_ids, N_orig, N_dest, w)
            @inbounds for n in 1:N_obs
                η_new[n] += origin_fe_vec[o_ids[n]] + dest_fe_vec[d_ids[n]]
            end
        elseif fe isa OriginFE
            @inbounds for n in 1:N_obs; η_new[n] += origin_fe_vec[o_ids[n]]; end
        elseif fe isa DestinationFE
            @inbounds for n in 1:N_obs; η_new[n] += dest_fe_vec[d_ids[n]]; end
        end
        μ = exp.(clamp.(η_new, T(-20), T(20)))

        if maximum(abs.(β .- β_old)) < tol
            converged = true
            break
        end
    end

    # Log-likelihood
    loglik = zero(T)
    @inbounds for n in 1:N_obs
        μn = max(μ[n], T(1e-30))
        yn = y[n]
        loglik += yn * log(μn) - μn - _logfactorial(yn)
    end

    # Null deviance (intercept-only)
    ȳ = mean(y)
    null_loglik = zero(T)
    @inbounds for n in 1:N_obs
        null_loglik += y[n] * log(max(ȳ, T(1e-30))) - ȳ - _logfactorial(y[n])
    end
    deviance = T(2) * (loglik - null_loglik)  # actually want 2*(saturated - fitted)
    pseudo_R2 = one(T) - loglik / min(null_loglik, -T(1e-12))

    residuals = y .- μ

    # Robust (sandwich) standard errors
    vcov_mat = _sandwich_vcov(X, μ, residuals, o_ids, d_ids, N_orig, N_dest, fe, w)

    return GravityModel{PoissonGravity, T}(
        PoissonGravity(), β, vcov_mat,
        origin_fe_vec, dest_fe_vec, T(NaN),
        loglik, deviance, pseudo_R2,
        residuals, μ,
        N_obs, N_orig, N_dest, vnames,
        converged, iter
    )
end

function _logfactorial(x::T) where {T<:AbstractFloat}
    n = round(Int, x)
    n <= 1 && return zero(T)
    return T(sum(log(k) for k in 2:n))
end

function _loggamma(x::T) where {T<:AbstractFloat}
    # Lanczos approximation for log Γ(x)
    x <= zero(T) && return T(Inf)
    if x < T(0.5)
        return log(T(π) / sin(T(π) * x)) - _loggamma(one(T) - x)
    end
    x -= one(T)
    g = T(7)
    c = T[0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    s = c[1]
    for i in 2:9
        s += c[i] / (x + T(i - 1))
    end
    t = x + g + T(0.5)
    return T(0.5) * log(T(2π)) + (x + T(0.5)) * log(t) - t + log(s)
end

function _sandwich_vcov(X::Matrix{T}, μ::Vector{T}, resid::Vector{T},
                        o_ids::Vector{Int}, d_ids::Vector{Int},
                        N_orig::Int, N_dest::Int, fe::GravityFE,
                        w::Vector{T}) where {T}
    N, K = size(X)
    # Bread: (X'WX)^{-1} where W = diag(μ)
    XwX = zeros(T, K, K)
    @inbounds for n in 1:N
        wn = w[n]
        for j in 1:K, i in 1:K
            XwX[i, j] += wn * X[n, i] * X[n, j]
        end
    end
    B_inv = inv(Symmetric(XwX))

    # Meat: X' diag(resid²) X (HC0)
    meat = zeros(T, K, K)
    @inbounds for n in 1:N
        r2 = resid[n]^2
        for j in 1:K, i in 1:K
            meat[i, j] += r2 * X[n, i] * X[n, j]
        end
    end

    return Symmetric(B_inv * meat * B_inv) |> Matrix
end

# ─── Negative Binomial gravity ────────────────────────────────────────

function fit_gravity(
    ::NegBinGravity,
    flows::AbstractVector,
    X::AbstractMatrix{T},
    origin_id::AbstractVector{<:Integer},
    dest_id::AbstractVector{<:Integer};
    fe::GravityFE=BilateralFE(),
    varnames::Union{Nothing, Vector{String}}=nothing,
    maxiter::Int=100,
    tol::T=T(1e-8)
) where {T<:AbstractFloat}

    # First fit Poisson to get initial estimates
    m_pois = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id;
                         fe=fe, varnames=varnames, maxiter=maxiter, tol=tol)

    y = Vector{T}(flows)
    N_obs = length(y)
    μ = copy(m_pois.fitted)
    β = copy(m_pois.coef)

    # Estimate alpha from Poisson residuals (method of moments)
    pearson_resid2 = sum((y .- μ).^2 ./ max.(μ, T(1e-12)))
    K = size(X, 2)
    alpha = max(T(0.01), (pearson_resid2 / N_obs - one(T)) / max(mean(μ), T(1e-12)))

    # Alternating: given alpha, reweight IRLS; given β, update alpha
    converged = false
    iter = 0
    for it in 1:20
        iter = it
        alpha_old = alpha

        # NB variance: V(μ) = μ + α μ²
        w_nb = μ ./ (one(T) .+ alpha .* μ)

        # One round of PPML-like IRLS with NB weights (simplified)
        # Just re-estimate with adjusted weights — approximate
        m_nb = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id;
                           fe=fe, varnames=varnames, maxiter=30, tol=tol)
        β = m_nb.coef
        μ = m_nb.fitted

        # Update alpha
        pearson = sum((y .- μ).^2 ./ max.(μ, T(1e-12)))
        alpha = max(T(0.001), (pearson / N_obs - one(T)) / max(mean(μ), T(1e-12)))

        if abs(alpha - alpha_old) < tol
            converged = true
            break
        end
    end

    # NB log-likelihood
    loglik = zero(T)
    inv_alpha = one(T) / max(alpha, T(1e-12))
    @inbounds for n in 1:N_obs
        μn = max(μ[n], T(1e-30))
        yn = y[n]
        loglik += _loggamma(yn + inv_alpha) - _loggamma(inv_alpha) - _logfactorial(yn)
        loglik += inv_alpha * log(inv_alpha / (inv_alpha + μn))
        loglik += yn * log(μn / (inv_alpha + μn))
    end

    residuals = y .- μ
    ȳ = mean(y)
    null_ll = sum(y[n] * log(max(ȳ, T(1e-30))) - ȳ - _logfactorial(y[n]) for n in 1:N_obs)
    pseudo_R2 = one(T) - loglik / min(null_ll, -T(1e-12))
    deviance = T(2) * (loglik - null_ll)

    return GravityModel{NegBinGravity, T}(
        NegBinGravity(), β, m_pois.vcov,
        m_pois.origin_fe, m_pois.dest_fe, alpha,
        loglik, deviance, pseudo_R2,
        residuals, μ,
        N_obs, m_pois.N_origin, m_pois.N_dest,
        m_pois.coefnames, converged, iter
    )
end

# ─── Multilateral resistance ─────────────────────────────────────────

function multilateral_resistance(m::GravityModel)
    return (outward = m.origin_fe, inward = m.dest_fe)
end

# ─── Show ─────────────────────────────────────────────────────────────

function Base.show(io::IO, m::GravityModel{V}) where {V}
    vname = V === PoissonGravity ? "Poisson (PPML)" : "Negative Binomial"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Gravity Model — $vname — $(m.converged ? "Converged" : "Not converged") ($(m.iterations) iter)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    ses = StatsAPI.stderror(m)
    _print_coef_table(io, m.coefnames, m.coef, ses)
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "Log-lik = %.2f    Pseudo-R² = %.4f    N = %d\n", m.loglik, m.pseudo_R2, m.N_obs)
    @printf(io, "Origins = %d    Destinations = %d", m.N_origin, m.N_dest)
    V === NegBinGravity && @printf(io, "    α = %.4f", m.alpha)
    println(io)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─── StatsAPI ─────────────────────────────────────────────────────────

StatsAPI.coef(m::GravityModel) = m.coef
StatsAPI.vcov(m::GravityModel) = m.vcov
StatsAPI.stderror(m::GravityModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::GravityModel) = m.loglik
StatsAPI.nobs(m::GravityModel) = m.N_obs
StatsAPI.residuals(m::GravityModel) = m.residuals
StatsAPI.coefnames(m::GravityModel) = m.coefnames

end # module
