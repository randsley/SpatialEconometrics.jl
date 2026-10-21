module HigherOrderSpatial

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames,
       loglikelihood, dof_residual, fitted, predict

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, brent_maximize, _normcdf, _print_coef_table

const LogDetMethod = SpatialUtils.LogDetMethod
const ExactLU = SpatialUtils.ExactLU
const ChebyshevApprox = SpatialUtils.ChebyshevApprox

export SLXModel, fit_slx,
       SACModel, fit_sac,
       SDEMModel, fit_sdem

# ═══════════════════════════════════════════════════════════════════════
# SLX: Spatial Lag of X
#   y = Xβ + WXθ + ε    (OLS — no spatial lag of y, no spatial error)
# ═══════════════════════════════════════════════════════════════════════

struct SLXModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    sigma2::T
    loglik::T
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

"""
    fit_slx(W, X, y; varnames=nothing)

Fit the SLX (Spatial Lag of X) model by OLS:
  y = Xβ + WXθ + ε

No spatial lag of y and no spatial error structure.
Recommended as a starting point before SAR/SDM (Halleck Vega & Elhorst, 2015).
"""
function fit_slx(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    WX = W * X
    X_aug = hcat(ones(T, N), X, WX)
    cnames = vcat(["(Intercept)"], vnames, ["W_" * v for v in vnames])

    β = X_aug \ Vector{T}(y)
    res = y .- X_aug * β
    σ2 = dot(res, res) / N
    ll = -T(0.5) * N * (log(T(2π)) + log(σ2) + one(T))

    V = σ2 .* inv(Symmetric(X_aug' * X_aug))

    return SLXModel{T, M}(β, V, σ2, ll, W, X_aug, Vector{T}(y), res, cnames)
end

StatsAPI.coef(m::SLXModel) = m.coef
StatsAPI.vcov(m::SLXModel) = m.vcov
StatsAPI.stderror(m::SLXModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::SLXModel) = length(m.y)
StatsAPI.residuals(m::SLXModel) = m.residuals
StatsAPI.coefnames(m::SLXModel) = m.coefnames
StatsAPI.loglikelihood(m::SLXModel) = m.loglik
StatsAPI.dof_residual(m::SLXModel) = length(m.y) - length(m.coef)

function Base.show(io::IO, m::SLXModel{T}) where {T}
    println(io, "SLX Model (Spatial Lag of X)")
    println(io, "════════════════════════════")
    @printf(io, "  N = %d,  σ² = %.4f,  Log-Lik = %.2f\n", nobs(m), m.sigma2, m.loglik)
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
end

# ═══════════════════════════════════════════════════════════════════════
# SAC: Spatial Autoregressive Combined
#   y = ρ₁W₁y + Xβ + u,  u = ρ₂W₂u + ε
#   Allows different W matrices for lag and error components
# ═══════════════════════════════════════════════════════════════════════

struct SACModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    lambda::T
    sigma2::T
    loglik::T
    W_lag::M
    W_error::M
    X_aug::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

"""
    fit_sac(W_lag, W_error, X, y; varnames=nothing, method=ChebyshevApprox())

Fit the SAC (Spatial Autoregressive Combined) model via concentrated ML:
  y = ρW₁y + Xβ + u,  u = λW₂u + ε

Also known as SARAR when W₁ = W₂.
Kelejian & Prucha (1998).
"""
function fit_sac(
    W_lag::M,
    W_error::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    rho_bounds::Tuple{T, T}=(T(-0.99), T(0.99)),
    lambda_bounds::Tuple{T, T}=(T(-0.99), T(0.99))
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    X_aug = hcat(ones(T, N), X)
    P = size(X_aug, 2)
    cnames = vcat(["(Intercept)"], vnames, ["ρ", "λ"])

    W1y = W_lag * y
    W2y = W_error * y
    W2X = W_error * X_aug
    W2W1y = W_error * W1y

    if method isa ChebyshevApprox
        τ_lag = precompute_chebyshev_traces(W_lag, method)
        τ_err = precompute_chebyshev_traces(W_error, method)
        logdet_lag = ρ -> chebyshev_logdet(ρ, τ_lag)
        logdet_err = λ -> chebyshev_logdet(λ, τ_err)
    else
        I_sp = sparse(one(T) * I, N, N)
        logdet_lag = ρ -> first(logabsdet(lu(I_sp - ρ * W_lag)))
        logdet_err = λ -> first(logabsdet(lu(I_sp - λ * W_error)))
    end

    const_term = -T(0.5) * N * (log(T(2π)) + one(T))

    function concentrated_ll(params)
        ρ, λ = params
        y_f = y .- ρ .* W1y .- λ .* W2y .+ (ρ * λ) .* W2W1y
        X_f = X_aug .- λ .* W2X
        β_f = X_f \ y_f
        res_f = y_f .- X_f * β_f
        σ2_f = dot(res_f, res_f) / N
        return const_term + logdet_lag(ρ) + logdet_err(λ) - T(0.5) * N * log(σ2_f)
    end

    best_ll = T(-Inf); best_ρ = zero(T); best_λ = zero(T)
    n_grid = 11
    ρ_grid = range(rho_bounds[1] * T(0.8), rho_bounds[2] * T(0.8), length=n_grid)
    λ_grid = range(lambda_bounds[1] * T(0.8), lambda_bounds[2] * T(0.8), length=n_grid)
    for ρ_try in ρ_grid, λ_try in λ_grid
        ll_try = concentrated_ll((ρ_try, λ_try))
        if ll_try > best_ll
            best_ll = ll_try; best_ρ = ρ_try; best_λ = λ_try
        end
    end

    step = T(0.01)
    for _ in 1:100
        improved = false
        for dρ in (-step, zero(T), step), dλ in (-step, zero(T), step)
            (dρ == 0 && dλ == 0) && continue
            ρ_try = clamp(best_ρ + dρ, rho_bounds[1], rho_bounds[2])
            λ_try = clamp(best_λ + dλ, lambda_bounds[1], lambda_bounds[2])
            ll_try = concentrated_ll((ρ_try, λ_try))
            if ll_try > best_ll + T(1e-10)
                best_ll = ll_try; best_ρ = ρ_try; best_λ = λ_try; improved = true
            end
        end
        improved || (step > T(1e-6) ? (step *= T(0.5)) : break)
    end

    ρ_hat = best_ρ; λ_hat = best_λ
    y_f = y .- ρ_hat .* W1y .- λ_hat .* W2y .+ (ρ_hat * λ_hat) .* W2W1y
    X_f = X_aug .- λ_hat .* W2X
    β_hat = X_f \ y_f
    res = y_f .- X_f * β_hat
    σ2_hat = dot(res, res) / N
    max_ll = concentrated_ll((ρ_hat, λ_hat))

    V_β = σ2_hat .* inv(Symmetric(X_f' * X_f))
    n_total = P + 2
    V_full = zeros(T, n_total, n_total)
    V_full[1:P, 1:P] .= V_β
    V_full[P+1, P+1] = T(1) / N
    V_full[P+2, P+2] = T(1) / N
    full_coef = vcat(β_hat, ρ_hat, λ_hat)

    return SACModel{T, M}(full_coef, V_full, ρ_hat, λ_hat, σ2_hat, max_ll,
        W_lag, W_error, X_aug, Vector{T}(y), res, cnames)
end

StatsAPI.coef(m::SACModel) = m.coef
StatsAPI.vcov(m::SACModel) = m.vcov
StatsAPI.stderror(m::SACModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::SACModel) = length(m.y)
StatsAPI.residuals(m::SACModel) = m.residuals
StatsAPI.coefnames(m::SACModel) = m.coefnames
StatsAPI.loglikelihood(m::SACModel) = m.loglik
StatsAPI.dof_residual(m::SACModel) = length(m.y) - length(m.coef)

function Base.show(io::IO, m::SACModel{T}) where {T}
    println(io, "SAC Model (Spatial Autoregressive Combined)")
    println(io, "════════════════════════════════════════════")
    @printf(io, "  N = %d,  ρ = %.4f,  λ = %.4f\n", nobs(m), m.rho, m.lambda)
    @printf(io, "  σ² = %.4f,  Log-Lik = %.2f\n", m.sigma2, m.loglik)
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
end

# ═══════════════════════════════════════════════════════════════════════
# SDEM: Spatial Durbin Error Model
#   y = Xβ + WXθ + u,  u = λWu + ε
# ═══════════════════════════════════════════════════════════════════════

struct SDEMModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    lambda::T
    lambda_se::T
    sigma2::T
    loglik::T
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

"""
    fit_sdem(W, X, y; varnames=nothing, method=ChebyshevApprox())

Fit the SDEM (Spatial Durbin Error Model) via concentrated ML:
  y = Xβ + WXθ + u,  u = λWu + ε

Combines the SDM covariate structure with SEM error structure.
LeSage & Pace (2009).
"""
function fit_sdem(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995))
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    WX = W * X
    X_aug = hcat(ones(T, N), X, WX)
    P = size(X_aug, 2)
    cnames = vcat(["(Intercept)"], vnames, ["W_" * v for v in vnames], ["λ"])

    Wy = W * y
    W_Xaug = W * X_aug

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = λ -> chebyshev_logdet(λ, τ)
    else
        I_sp = sparse(one(T) * I, N, N)
        logdet_fn = λ -> first(logabsdet(lu(I_sp - λ * W)))
    end

    const_term = -T(0.5) * N * (log(T(2π)) + one(T))
    function concentrated_ll(λ)
        y_f = y .- λ .* Wy
        X_f = X_aug .- λ .* W_Xaug
        β_f = X_f \ y_f
        res_f = y_f .- X_f * β_f
        σ2_f = dot(res_f, res_f) / N
        return const_term + logdet_fn(λ) - T(0.5) * N * log(σ2_f)
    end

    λ_hat, max_ll = brent_maximize(concentrated_ll, lambda_bounds[1], lambda_bounds[2])

    y_f = y .- λ_hat .* Wy
    X_f = X_aug .- λ_hat .* W_Xaug
    β_hat = X_f \ y_f
    res = y_f .- X_f * β_hat
    σ2_hat = dot(res, res) / N

    V_β = σ2_hat .* inv(Symmetric(X_f' * X_f))
    n_total = P + 1
    V_full = zeros(T, n_total, n_total)
    V_full[1:P, 1:P] .= V_β
    V_full[n_total, n_total] = T(1) / N
    λ_se = sqrt(V_full[n_total, n_total])

    full_coef = vcat(β_hat, λ_hat)

    return SDEMModel{T, M}(full_coef, V_full, λ_hat, λ_se, σ2_hat, max_ll,
        W, X_aug, Vector{T}(y), res, cnames)
end

StatsAPI.coef(m::SDEMModel) = m.coef
StatsAPI.vcov(m::SDEMModel) = m.vcov
StatsAPI.stderror(m::SDEMModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::SDEMModel) = length(m.y)
StatsAPI.residuals(m::SDEMModel) = m.residuals
StatsAPI.coefnames(m::SDEMModel) = m.coefnames
StatsAPI.loglikelihood(m::SDEMModel) = m.loglik
StatsAPI.dof_residual(m::SDEMModel) = length(m.y) - length(m.coef)

function Base.show(io::IO, m::SDEMModel{T}) where {T}
    println(io, "SDEM Model (Spatial Durbin Error)")
    println(io, "═════════════════════════════════")
    @printf(io, "  N = %d,  λ = %.4f (SE: %.4f)\n", nobs(m), m.lambda, m.lambda_se)
    @printf(io, "  σ² = %.4f,  Log-Lik = %.2f\n", m.sigma2, m.loglik)
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
end

end # module
