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
       precompute_neumann_traces, g_traces,
       brent_maximize, _normcdf, _print_coef_table

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

"""
    SLXModel

A fitted SLX model, `y = X*beta + W*X*theta + e`, returned by
[`fit_slx`](@ref).

The simplest way to admit spillovers: covariates of neighbours enter, but
`y` does not depend on `W*y`. That makes it OLS on `[X W*X]`, so there is
no log-determinant and no endogeneity, and `theta` is the spillover
directly — no `(I - rho*W)^{-1}` decomposition needed.

Often the right starting point before reaching for a lag model.
"""
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

    # SLX is plain OLS, so the exact finite-sample covariance is available.
    # This used the ML σ² = RSS/N that the log-likelihood needs, which
    # understates every standard error by sqrt((N-P)/N) -- 1.75% at N = 144
    # with P = 5 -- while dof_residual already reported N - P. The stored
    # σ2 stays the ML value so the likelihood is unchanged.
    s2_unbiased = dot(res, res) / max(N - size(X_aug, 2), 1)
    V = s2_unbiased .* inv(Symmetric(X_aug' * X_aug))

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

"""
    SACModel

A fitted SAC / SARAR model, returned by [`fit_sac`](@ref):
`y = rho*W*y + X*beta + u` with `u = lambda*W*u + e`.

Dependence in both `y` and the errors, so it nests `SAR`
(`lambda = 0`) and the spatial error model (`rho = 0`).

!!! warning
    With the same `W` in both roles, `rho` and `lambda` are only weakly
    separated: the likelihood is often nearly flat along a ridge trading one
    against the other, so individual estimates can be imprecise even when
    the fit is good. Check them against the simpler nested models.
"""
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
    lambda_bounds::Tuple{T, T}=(T(-0.99), T(0.99)),
    max_exact_n::Int=1000,
    vcov_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
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

    V_full = _sac_asymptotic_vcov(X_f, β_hat, W_lag, W_error, ρ_hat, λ_hat,
                                  σ2_hat, N; max_exact_n=max_exact_n,
                                  n_probes=vcov_probes, rng=rng)
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

"""
    _sac_asymptotic_vcov(X_f, β, W_lag, W_error, ρ, λ, σ2, N; kwargs...)

Asymptotic covariance of `(β, ρ, λ)` for the SAC/SARAR model, from the
expected information matrix.

Previously the two spatial parameters had no variance computed at all: the
diagonal slots were filled with `1/N`, so every reported standard error for
`ρ` and `λ` came out as exactly `1/sqrt(N)` — identical across datasets,
unchanged when the estimates themselves ranged from 0.08 to 0.36, and so
every z-statistic and p-value built on them was meaningless.

Writing `A = I - ρW₁`, `B = I - λW₂`, `X* = BX`, `G_A = W₁A⁻¹`,
`G_B = W₂B⁻¹` and `M = B G_A B⁻¹`, the residual is `ε = B(Ay - Xβ)` and
`c = M X*β`. Differentiating the log-likelihood gives

    I_ββ = X*ᵀX*/σ²
    I_βρ = X*ᵀc/σ²
    I_βλ = 0
    I_ρρ = cᵀc/σ² + tr(MᵀM) + tr(M²)
    I_ρλ = tr(MᵀG_B) + tr(M G_B)
    I_λλ = tr(G_BᵀG_B) + tr(G_B²)
    I_ρσ² = tr(M)/σ²,  I_λσ² = tr(G_B)/σ²,  I_σ²σ² = N/(2σ⁴)

`I_βλ` vanishes because `∂ε/∂λ = -G_B ε` has mean zero, which is the same
reason the pure error model is block diagonal; `I_βρ` does not, because the
lag term puts `Xβ` into `∂ε/∂ρ`. `M` is similar to `G_A`, so `tr(M)` and
`tr(M²)` could be taken from `G_A` — they are computed from `M` here so the
exact and probe paths agree term by term.

Exact below `max_exact_n`; above it the traces come from Hutchinson probes
drawn against `rng`.
"""
function _sac_asymptotic_vcov(
    X_f::AbstractMatrix{T},
    β::Vector{T},
    W_lag::AbstractSparseMatrix{T},
    W_error::AbstractSparseMatrix{T},
    ρ::T,
    λ::T,
    σ2::T,
    N::Int;
    max_exact_n::Int=1000,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    P = size(X_f, 2)
    Isp = sparse(one(T) * I, N, N)
    A_lu = lu(Isp - ρ * W_lag)
    B_lu = lu(Isp - λ * W_error)
    Bmat = Isp - λ * W_error

    Xfβ = X_f * β

    if N <= max_exact_n
        # A and W commute, so A_lu \ W is G_A.
        G_A = A_lu \ Matrix{T}(W_lag)
        G_B = B_lu \ Matrix{T}(W_error)
        M = Bmat * (G_A * (B_lu \ Matrix{T}(I, N, N)))
        c = M * Xfβ

        tr_M = zero(T); tr_M2 = zero(T); tr_MtM = zero(T)
        tr_GB = zero(T); tr_GB2 = zero(T); tr_GBtGB = zero(T)
        tr_MtGB = zero(T); tr_MGB = zero(T)
        @inbounds for j in 1:N
            tr_M += M[j, j]
            tr_GB += G_B[j, j]
            for i in 1:N
                tr_M2 += M[i, j] * M[j, i]
                tr_MtM += M[i, j]^2
                tr_GB2 += G_B[i, j] * G_B[j, i]
                tr_GBtGB += G_B[i, j]^2
                tr_MtGB += M[i, j] * G_B[i, j]
                tr_MGB += M[i, j] * G_B[j, i]
            end
        end
    else
        V = Matrix{T}(undef, N, n_probes)
        rand!(rng, V, (-one(T), one(T)))
        GAV = A_lu \ (W_lag * V)
        MV = Bmat * (A_lu \ (W_lag * (B_lu \ V)))
        GBV = B_lu \ (W_error * V)
        MMV = Bmat * (A_lu \ (W_lag * (B_lu \ MV)))
        GBGBV = B_lu \ (W_error * GBV)
        MGBV = Bmat * (A_lu \ (W_lag * (B_lu \ GBV)))

        tr_M = dot(V, MV) / n_probes
        tr_M2 = dot(V, MMV) / n_probes
        tr_MtM = dot(MV, MV) / n_probes
        tr_GB = dot(V, GBV) / n_probes
        tr_GB2 = dot(V, GBGBV) / n_probes
        tr_GBtGB = dot(GBV, GBV) / n_probes
        tr_MtGB = dot(MV, GBV) / n_probes
        tr_MGB = dot(V, MGBV) / n_probes
        c = Bmat * (A_lu \ (W_lag * (B_lu \ Xfβ)))
    end

    nt = P + 3
    Info = zeros(T, nt, nt)
    iρ = P + 1; iλ = P + 2; iσ = P + 3

    Info[1:P, 1:P] .= (X_f' * X_f) ./ σ2
    Info[1:P, iρ] .= (X_f' * c) ./ σ2
    Info[iρ, 1:P] .= Info[1:P, iρ]
    # Info[1:P, iλ] stays zero: β and λ are asymptotically independent.

    Info[iρ, iρ] = dot(c, c) / σ2 + tr_MtM + tr_M2
    Info[iρ, iλ] = tr_MtGB + tr_MGB
    Info[iλ, iρ] = Info[iρ, iλ]
    Info[iλ, iλ] = tr_GBtGB + tr_GB2

    Info[iρ, iσ] = tr_M / σ2
    Info[iσ, iρ] = Info[iρ, iσ]
    Info[iλ, iσ] = tr_GB / σ2
    Info[iσ, iλ] = Info[iλ, iσ]
    Info[iσ, iσ] = N / (2 * σ2^2)

    V_full = inv(Symmetric(Info))
    return Matrix{T}(V_full[1:(P + 2), 1:(P + 2)])
end

# ═══════════════════════════════════════════════════════════════════════
# SDEM: Spatial Durbin Error Model
#   y = Xβ + WXθ + u,  u = λWu + ε
# ═══════════════════════════════════════════════════════════════════════

"""
    SDEMModel

A fitted spatial Durbin error model, returned by [`fit_sdem`](@ref):
`y = X*beta + W*X*theta + u` with `u = lambda*W*u + e`.

Lagged covariates plus a spatial error, but no lag of `y`. Like
[`SLXModel`](@ref), the absence of feedback through `y` means `theta` reads
as the spillover directly; unlike it, the errors are modelled too, so
inference accounts for residual spatial correlation.
"""
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
    lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995)),
    max_exact_n::Int=1000,
    vcov_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
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

    # SDEM is an error model in augmented regressors, so β is asymptotically
    # independent of (λ, σ²) and σ²(X_f'X_f)⁻¹ is its correct variance. λ's
    # was not computed at all: the slot held 1/N, making every reported
    # standard error for λ exactly 1/sqrt(N) whatever the data said. Here it
    # comes from the same information matrix the error model uses,
    # I_λλ = tr(G² + G'G) with G = W(I - λW)⁻¹, inverted jointly with σ².
    tr_G, tr_G2, tr_GtG = g_traces(W, λ_hat; max_exact_n=max_exact_n,
                                   n_probes=vcov_probes, rng=rng)
    I_λλ = tr_G2 + tr_GtG
    I_λσ2 = tr_G / σ2_hat
    I_σ2σ2 = N / (2 * σ2_hat^2)
    V_nuis = inv(Symmetric([I_λλ I_λσ2; I_λσ2 I_σ2σ2]))

    V_β = σ2_hat .* inv(Symmetric(X_f' * X_f))
    n_total = P + 1
    V_full = zeros(T, n_total, n_total)
    V_full[1:P, 1:P] .= V_β
    V_full[n_total, n_total] = max(zero(T), V_nuis[1, 1])
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
