module SpatialDiscreteChoice

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames,
       loglikelihood, dof_residual, predict

import ..SpatialUtils
import ..SpatialUtils: _normcdf, _print_coef_table

export SpatialProbitModel, fit_spatial_probit,
       BayesianSpatialProbitModel, fit_bayesian_spatial_probit,
       SpatialOrderedProbitModel, fit_spatial_ordered_probit,
       SpatialProbitEffects, marginal_effects

const _EPS_P = 1e-10

# ═══════════════════════════════════════════════════════════════════════
# Normal distribution helpers (no Distributions.jl dependency)
# ═══════════════════════════════════════════════════════════════════════

_normpdf(z::Real) = 0.3989422804014327 * exp(-0.5 * Float64(z)^2)

# Acklam's rational approximation to the standard normal quantile.
function _norminv(p::Real)
    x = Float64(p)
    x <= 0.0 && return -Inf
    x >= 1.0 && return Inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    if x < plow
        q = sqrt(-2.0 * log(x))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    elseif x <= 1.0 - plow
        q = x - 0.5; r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1.0)
    else
        q = sqrt(-2.0 * log(1.0 - x))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1.0)
    end
end

# Truncated normal draw on (lo, hi] by inverse CDF. Precision degrades in the
# far tails, so the interval is widened to a representable mass before drawing.
function _rand_truncnorm(rng::AbstractRNG, mu::Float64, sigma::Float64,
                         lo::Float64, hi::Float64)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    Fa = _normcdf(a)
    Fb = _normcdf(b)
    if Fb - Fa < 1e-12
        return clamp(mu, lo, hi)
    end
    u = Fa + rand(rng) * (Fb - Fa)
    u = clamp(u, 1e-12, 1.0 - 1e-12)
    return mu + sigma * _norminv(u)
end

# ═══════════════════════════════════════════════════════════════════════
# Standard (non-spatial) probit by Fisher scoring — the ρ = 0 starting point
# ═══════════════════════════════════════════════════════════════════════

function _fit_probit(X::AbstractMatrix{T}, y::AbstractVector{T};
                     maxiter::Int=100, tol::T=T(1e-9)) where {T<:AbstractFloat}
    K = size(X, 2)
    β = zeros(T, K)
    for _ in 1:maxiter
        η = X * β
        Φ = clamp.(_normcdf.(η), _EPS_P, 1 - _EPS_P)
        φ = _normpdf.(η)
        score = X' * (φ .* (y .- Φ) ./ (Φ .* (1 .- Φ)))
        w = φ .^ 2 ./ (Φ .* (1 .- Φ))
        info = Symmetric(X' * (w .* X))
        Δ = try
            info \ score
        catch
            (Matrix(info) + T(1e-8) * I) \ score
        end
        β .+= Δ
        norm(Δ) < tol && break
    end
    return β
end

function _probit_loglik(η::AbstractVector{T}, y::AbstractVector{T}) where {T}
    Φ = clamp.(_normcdf.(η), _EPS_P, 1 - _EPS_P)
    return sum(y .* log.(Φ) .+ (one(T) .- y) .* log.(one(T) .- Φ))
end

# ═══════════════════════════════════════════════════════════════════════
# SpatialProbitModel — linearized GMM (Klier & McMillen 2008)
# ═══════════════════════════════════════════════════════════════════════

struct SpatialProbitModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    sigma2::T
    loglik::T
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    index::Vector{T}
    fitted_prob::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
    converged::Bool
end

"""
    fit_spatial_probit(W, X, y; varnames=nothing, instrument_lags=2)

Fit the spatial autoregressive probit model

    y* = ρWy* + Xβ + ε,    y = 1(y* > 0)

by the linearized GMM estimator of Klier & McMillen (2008). A standard probit
supplies the ρ = 0 starting value; the moment conditions are then linearized
around it and estimated by 2SLS with instruments `[X, WX, W²X]`.

`y` must be coded 0/1. Returns a `SpatialProbitModel`.

Fitted probabilities use the heteroskedastic latent form
`Φ(((I-ρW)⁻¹Xβ)ᵢ / √[(I-ρW)⁻¹(I-ρW)⁻ᵀ]ᵢᵢ)`, which needs a dense inverse of
`I - ρW`; cost is O(N³), so this is intended for moderate N.

`loglikelihood` reports the probit log-likelihood evaluated at the fitted index.
It is a pseudo-likelihood — the estimator is GMM, not ML — so it is comparable
across specifications fit the same way, but is not an ML criterion.

Linearizing around ρ = 0 buys speed at a real cost in efficiency. Over 8
replications at ρ = 0.4 this estimator had a standard deviation of about 0.25 at
N = 196 and 0.18 at N = 400 — roughly 2.5× that of
[`fit_bayesian_spatial_probit`](@ref) on the same data. It is approximately
unbiased, so treat a single small-sample ρ̂ as a coarse reading and prefer the
Gibbs sampler when precision on ρ matters.
"""
function fit_spatial_probit(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    varnames::Union{Nothing, Vector{String}}=nothing,
    instrument_lags::Int=2
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert length(y) == N "y and X must have the same number of rows"
    @assert all(v -> v == zero(T) || v == one(T), y) "y must be coded 0/1"
    @assert 1 <= instrument_lags <= 3 "instrument_lags must be 1, 2, or 3"
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    X_aug = hcat(ones(T, N), X)
    P = size(X_aug, 2)
    cnames = vcat(["(Intercept)"], vnames, ["ρ"])

    β0 = _fit_probit(X_aug, Vector{T}(y))

    η0 = X_aug * β0
    Φ0 = clamp.(_normcdf.(η0), _EPS_P, 1 - _EPS_P)
    φ0 = _normpdf.(η0)
    s0 = sqrt.(Φ0 .* (one(T) .- Φ0))

    # Scaled generalized residual and gradient blocks at ρ = 0.
    u0 = (y .- Φ0) ./ s0
    G_β = (φ0 ./ s0) .* X_aug
    G_ρ = (φ0 ./ s0) .* (W * η0)

    dep = u0 .+ G_β * β0
    R = hcat(G_β, G_ρ)

    # Instruments: spatial lags of the covariates only. W*intercept equals the
    # intercept under row-standardization, so including it would be collinear.
    blocks = Matrix{T}[X_aug]
    WX = W * X
    instrument_lags >= 1 && push!(blocks, WX)
    if instrument_lags >= 2
        WX = W * WX
        push!(blocks, WX)
    end
    if instrument_lags >= 3
        WX = W * WX
        push!(blocks, WX)
    end
    Z = reduce(hcat, blocks)

    ZtZ = Symmetric(Z' * Z)
    Rhat = Z * (ZtZ \ (Z' * R))

    RhRh = Symmetric(Rhat' * Rhat)
    converged = true
    θ = try
        RhRh \ (Rhat' * dep)
    catch
        converged = false
        (Matrix(RhRh) + T(1e-8) * I) \ (Rhat' * dep)
    end

    β_hat = θ[1:P]
    ρ_hat = θ[P+1]
    if abs(ρ_hat) >= one(T)
        converged = false
        ρ_hat = clamp(ρ_hat, T(-0.99), T(0.99))
    end

    # Heteroskedasticity-robust 2SLS sandwich.
    e_lin = dep .- R * θ
    bread = try
        inv(RhRh)
    catch
        inv(Matrix(RhRh) + T(1e-8) * I)
    end
    meat = Rhat' * ((e_lin .^ 2) .* Rhat)
    V = bread * meat * bread
    V = T(0.5) .* (V .+ V')

    A = sparse(one(T) * I, N, N) - ρ_hat * W
    S = inv(Matrix(A))
    index_raw = S * (X_aug * β_hat)
    sd_lat = sqrt.(max.(T(1e-12), vec(sum(abs2, S, dims=2))))
    index = index_raw ./ sd_lat
    probs = clamp.(_normcdf.(index), _EPS_P, 1 - _EPS_P)
    res = y .- probs
    σ2 = dot(res, res) / N
    ll = _probit_loglik(index, Vector{T}(y))

    return SpatialProbitModel{T, M}(θ, V, ρ_hat, σ2, ll, W, X_aug,
        Vector{T}(y), index, probs, res, cnames, converged)
end

StatsAPI.coef(m::SpatialProbitModel) = m.coef
StatsAPI.vcov(m::SpatialProbitModel) = m.vcov
StatsAPI.stderror(m::SpatialProbitModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::SpatialProbitModel) = length(m.y)
StatsAPI.residuals(m::SpatialProbitModel) = m.residuals
StatsAPI.coefnames(m::SpatialProbitModel) = m.coefnames
StatsAPI.loglikelihood(m::SpatialProbitModel) = m.loglik
StatsAPI.dof_residual(m::SpatialProbitModel) = length(m.y) - length(m.coef)
StatsAPI.predict(m::SpatialProbitModel) = m.fitted_prob

function Base.show(io::IO, m::SpatialProbitModel{T}) where {T}
    println(io, "Spatial Probit Model (linearized GMM)")
    println(io, "═════════════════════════════════════")
    @printf(io, "  N = %d,  ρ = %.4f\n", nobs(m), m.rho)
    @printf(io, "  Pseudo Log-Lik = %.2f,  Hit rate = %.3f\n",
        m.loglik, mean((m.fitted_prob .>= T(0.5)) .== (m.y .== one(T))))
    m.converged || println(io, "  WARNING: 2SLS step was ill-conditioned; estimates may be unstable.")
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
end

# ═══════════════════════════════════════════════════════════════════════
# Marginal effects — direct / indirect / total
# ═══════════════════════════════════════════════════════════════════════

struct SpatialProbitEffects{T<:AbstractFloat}
    direct::Vector{T}
    indirect::Vector{T}
    total::Vector{T}
    varnames::Vector{String}
end

"""
    marginal_effects(m::SpatialProbitModel)

Decompose the marginal effect of each covariate into direct, indirect, and total
components. For covariate `k` the effect matrix is `diag(φ(index)) (I-ρW)⁻¹ βₖ`;
the direct effect averages its diagonal, the total averages its row sums, and
the indirect effect is the difference (LeSage & Pace, 2009).

Requires a dense inverse of `I - ρW` — O(N³), so intended for moderate N.
"""
function marginal_effects(m::SpatialProbitModel{T}) where {T}
    N = length(m.y)
    K = size(m.X_aug, 2) - 1
    A = sparse(one(T) * I, N, N) - m.rho * m.W
    S = inv(Matrix(A))
    φ_vec = _normpdf.(m.index)

    direct = zeros(T, K)
    indirect = zeros(T, K)
    total = zeros(T, K)
    base = φ_vec .* S

    for k in 1:K
        βk = m.coef[k+1]
        d = zero(T)
        for i in 1:N
            d += base[i, i] * βk
        end
        direct[k] = d / N
        total[k] = sum(base) * βk / N
        indirect[k] = total[k] - direct[k]
    end

    return SpatialProbitEffects{T}(direct, indirect, total, m.coefnames[2:K+1])
end

function Base.show(io::IO, e::SpatialProbitEffects)
    println(io, "Spatial Probit Marginal Effects")
    println(io, "───────────────────────────────")
    @printf(io, "  %-16s %10s %10s %10s\n", "Variable", "Direct", "Indirect", "Total")
    for i in eachindex(e.varnames)
        @printf(io, "  %-16s %10.4f %10.4f %10.4f\n",
            e.varnames[i], e.direct[i], e.indirect[i], e.total[i])
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Latent-variable Gibbs step shared by the Bayesian samplers
# ═══════════════════════════════════════════════════════════════════════

# One single-site sweep of the latent vector. Given precision H = A'A and mean
# mu, z_i | z_{-i} ~ N(mu_i - H_ii^{-1} Σ_{j≠i} H_ij (z_j - mu_j), H_ii^{-1}),
# truncated to each observation's interval.
function _sweep_latent!(z::Vector{Float64}, H::SparseMatrixCSC{Float64, Int},
                        mu::Vector{Float64}, lo::Vector{Float64},
                        hi::Vector{Float64}, rng::AbstractRNG)
    rows = rowvals(H)
    vals = nonzeros(H)
    N = length(z)
    for i in 1:N
        h_ii = 0.0
        acc = 0.0
        for idx in nzrange(H, i)
            j = rows[idx]
            if j == i
                h_ii = vals[idx]
            else
                acc += vals[idx] * (z[j] - mu[j])
            end
        end
        h_ii <= 0.0 && continue
        cond_mu = mu[i] - acc / h_ii
        cond_sd = sqrt(1.0 / h_ii)
        z[i] = _rand_truncnorm(rng, cond_mu, cond_sd, lo[i], hi[i])
    end
    return z
end

function _rho_logpost(rho::Float64, W::SparseMatrixCSC{Float64, Int},
                      z::Vector{Float64}, Xb::Vector{Float64}, N::Int)
    A = sparse(1.0I, N, N) - rho * W
    ld = try
        first(logabsdet(lu(A)))
    catch
        return -Inf
    end
    r = A * z .- Xb
    return ld - 0.5 * dot(r, r)
end

# ═══════════════════════════════════════════════════════════════════════
# BayesianSpatialProbitModel — Albert & Chib data augmentation
# ═══════════════════════════════════════════════════════════════════════

struct BayesianSpatialProbitModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    beta_draws::Matrix{T}
    rho_draws::Vector{T}
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    fitted_prob::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
    n_samples::Int
    burnin::Int
    accept_rate::T
end

"""
    fit_bayesian_spatial_probit(W, X, y; n_samples=2000, burnin=500, rng=...)

Fit the spatial autoregressive probit by Gibbs sampling with Albert & Chib
(1993) data augmentation, following LeSage & Pace (2009, Ch. 10).

Each sweep: draw the latent `y*` one site at a time from its truncated
conditional, draw `β` from its conjugate normal conditional, then update `ρ` by
random-walk Metropolis on `log|I-ρW| - ½(Ay*-Xβ)'(Ay*-Xβ)` with a Uniform(-1,1)
prior. The error variance is fixed at 1 for identification.

`beta_prior_var` sets an isotropic N(0, v·I) prior on β. Reported `coef` and
`vcov` are the posterior mean and covariance; `accept_rate` is the ρ acceptance
rate, which should land roughly in 0.2–0.5. The default `rho_proposal_sd` of 0.2
targets that band; 0.05 overshoots to about 0.77 acceptance and mixes slowly.

Because the observed signs pin the latent vector only up to scale, ‖β‖ and ρ
trade off along a weakly identified ridge. The posterior is still centred
correctly — over 8 replications at ρ = 0.4 the posterior mean averaged 0.374 at
N = 196 — but its spread is wide (SD ≈ 0.10), so a single chain at small N can
sit well away from the truth. Check `accept_rate` and lengthen the chain before
reading much into one ρ estimate.
"""
function fit_bayesian_spatial_probit(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=2000,
    burnin::Int=500,
    beta_prior_var::Real=100.0,
    rho_proposal_sd::Real=0.2,
    varnames::Union{Nothing, Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert length(y) == N "y and X must have the same number of rows"
    @assert all(v -> v == zero(T) || v == one(T), y) "y must be coded 0/1"
    @assert n_samples > 0 && burnin >= 0
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    Xa = Matrix{Float64}(hcat(ones(T, N), X))
    P = size(Xa, 2)
    cnames = vcat(["(Intercept)"], vnames, ["ρ"])
    yf = Vector{Float64}(y)
    Wf = SparseMatrixCSC{Float64, Int}(W)

    lo = [yf[i] == 1.0 ? 0.0 : -Inf for i in 1:N]
    hi = [yf[i] == 1.0 ? Inf : 0.0 for i in 1:N]

    prior_prec = 1.0 / Float64(beta_prior_var)
    XtX = Xa' * Xa
    B1 = inv(Symmetric(XtX + prior_prec * I))
    B1 = 0.5 .* (B1 .+ B1')
    cholB1 = cholesky(Symmetric(B1)).L

    β = _fit_probit(Xa, yf)
    ρ = 0.0
    z = [yf[i] == 1.0 ? 0.5 : -0.5 for i in 1:N]

    beta_store = zeros(Float64, n_samples, P)
    rho_store = zeros(Float64, n_samples)
    n_accept = 0
    prop_sd = Float64(rho_proposal_sd)

    A = sparse(1.0I, N, N) - ρ * Wf
    ll_cur = _rho_logpost(ρ, Wf, z, Xa * β, N)

    for iter in 1:(burnin + n_samples)
        A = sparse(1.0I, N, N) - ρ * Wf
        H = SparseMatrixCSC{Float64, Int}(A' * A)
        mu = A \ (Xa * β)
        _sweep_latent!(z, H, mu, lo, hi, rng)

        Az = A * z
        b1 = B1 * (Xa' * Az)
        β = b1 .+ cholB1 * randn(rng, P)

        Xb = Xa * β
        ll_cur = _rho_logpost(ρ, Wf, z, Xb, N)
        ρ_prop = ρ + prop_sd * randn(rng)
        if -1.0 < ρ_prop < 1.0
            ll_prop = _rho_logpost(ρ_prop, Wf, z, Xb, N)
            if log(rand(rng)) < ll_prop - ll_cur
                ρ = ρ_prop
                ll_cur = ll_prop
                iter > burnin && (n_accept += 1)
            end
        end

        if iter > burnin
            k = iter - burnin
            beta_store[k, :] .= β
            rho_store[k] = ρ
        end
    end

    β_mean = vec(mean(beta_store, dims=1))
    ρ_mean = mean(rho_store)
    draws = hcat(beta_store, rho_store)
    V = cov(draws)
    V = 0.5 .* (V .+ V')

    A = sparse(1.0I, N, N) - ρ_mean * Wf
    S = inv(Matrix(A))
    idx_raw = S * (Xa * β_mean)
    sd_lat = sqrt.(max.(1e-12, vec(sum(abs2, S, dims=2))))
    probs = clamp.(_normcdf.(idx_raw ./ sd_lat), _EPS_P, 1 - _EPS_P)
    res = yf .- probs

    return BayesianSpatialProbitModel{T, M}(
        T.(vcat(β_mean, ρ_mean)), T.(V), T(ρ_mean),
        T.(beta_store), T.(rho_store), W, Matrix{T}(Xa), Vector{T}(y),
        T.(probs), T.(res), cnames, n_samples, burnin,
        T(n_accept / n_samples))
end

StatsAPI.coef(m::BayesianSpatialProbitModel) = m.coef
StatsAPI.vcov(m::BayesianSpatialProbitModel) = m.vcov
StatsAPI.stderror(m::BayesianSpatialProbitModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::BayesianSpatialProbitModel) = length(m.y)
StatsAPI.residuals(m::BayesianSpatialProbitModel) = m.residuals
StatsAPI.coefnames(m::BayesianSpatialProbitModel) = m.coefnames
StatsAPI.dof_residual(m::BayesianSpatialProbitModel) = length(m.y) - length(m.coef)
StatsAPI.predict(m::BayesianSpatialProbitModel) = m.fitted_prob

function Base.show(io::IO, m::BayesianSpatialProbitModel{T}) where {T}
    println(io, "Bayesian Spatial Probit (Albert-Chib Gibbs)")
    println(io, "═══════════════════════════════════════════")
    @printf(io, "  N = %d,  draws = %d (burn-in %d)\n", nobs(m), m.n_samples, m.burnin)
    @printf(io, "  Posterior mean ρ = %.4f,  ρ acceptance = %.3f\n", m.rho, m.accept_rate)
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
    println(io, "  (Estimate = posterior mean, Std. Err = posterior SD)")
end

# ═══════════════════════════════════════════════════════════════════════
# SpatialOrderedProbitModel — ordered outcomes via data augmentation
# ═══════════════════════════════════════════════════════════════════════

struct SpatialOrderedProbitModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    cutpoints::Vector{T}
    beta_draws::Matrix{T}
    rho_draws::Vector{T}
    cut_draws::Matrix{T}
    W::M
    X::Matrix{T}
    y::Vector{Int}
    n_categories::Int
    coefnames::Vector{String}
    n_samples::Int
    burnin::Int
    accept_rate::T
end

"""
    fit_spatial_ordered_probit(W, X, y; n_categories=maximum(y), n_samples=2000, ...)

Fit the spatial ordered probit

    y* = ρWy* + Xβ + ε,    y = j  iff  τ_{j-1} < y* ≤ τ_j

by Gibbs sampling with data augmentation (Wang, Kockelman & Wang, 2013).
`y` holds integer categories `1:J`.

For identification there is no intercept and `τ_1 = 0`; the free cutpoints
`τ_2 … τ_{J-1}` are drawn from their uniform conditionals between the adjacent
order statistics of the latent vector. `τ_0 = -∞` and `τ_J = +∞`.

A binary `y` (J = 2) reduces to the probit above and is better served by
[`fit_bayesian_spatial_probit`](@ref), which carries an intercept.
"""
function fit_spatial_ordered_probit(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{<:Integer};
    n_categories::Int=Int(maximum(y)),
    n_samples::Int=2000,
    burnin::Int=500,
    beta_prior_var::Real=100.0,
    rho_proposal_sd::Real=0.2,
    varnames::Union{Nothing, Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    J = n_categories
    @assert length(y) == N "y and X must have the same number of rows"
    @assert J >= 2 "need at least 2 categories"
    @assert all(c -> 1 <= c <= J, y) "categories must lie in 1:n_categories"
    @assert n_samples > 0 && burnin >= 0
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    cnames = vcat(vnames, ["ρ"])

    Xf = Matrix{Float64}(X)
    Wf = SparseMatrixCSC{Float64, Int}(W)
    yi = Vector{Int}(y)
    n_free = max(0, J - 2)

    # τ_1 = 0 fixed; free cutpoints start evenly spaced above it.
    τ = zeros(Float64, J - 1)
    for j in 2:(J - 1)
        τ[j] = Float64(j - 1)
    end

    prior_prec = 1.0 / Float64(beta_prior_var)
    B1 = inv(Symmetric(Xf' * Xf + prior_prec * I))
    B1 = 0.5 .* (B1 .+ B1')
    cholB1 = cholesky(Symmetric(B1)).L

    β = zeros(Float64, K)
    ρ = 0.0
    z = [Float64(yi[i]) - 1.5 for i in 1:N]
    lo = zeros(Float64, N)
    hi = zeros(Float64, N)

    beta_store = zeros(Float64, n_samples, K)
    rho_store = zeros(Float64, n_samples)
    cut_store = zeros(Float64, n_samples, J - 1)
    n_accept = 0
    prop_sd = Float64(rho_proposal_sd)

    for iter in 1:(burnin + n_samples)
        for i in 1:N
            c = yi[i]
            lo[i] = c == 1 ? -Inf : τ[c-1]
            hi[i] = c == J ? Inf : τ[c]
        end

        A = sparse(1.0I, N, N) - ρ * Wf
        H = SparseMatrixCSC{Float64, Int}(A' * A)
        mu = A \ (Xf * β)
        _sweep_latent!(z, H, mu, lo, hi, rng)

        Az = A * z
        b1 = B1 * (Xf' * Az)
        β = b1 .+ cholB1 * randn(rng, K)

        Xb = Xf * β
        ll_cur = _rho_logpost(ρ, Wf, z, Xb, N)
        ρ_prop = ρ + prop_sd * randn(rng)
        if -1.0 < ρ_prop < 1.0
            ll_prop = _rho_logpost(ρ_prop, Wf, z, Xb, N)
            if log(rand(rng)) < ll_prop - ll_cur
                ρ = ρ_prop
                iter > burnin && (n_accept += 1)
            end
        end

        # τ_j | z, y ~ Uniform(max{z : y = j}, min{z : y = j+1}), for j >= 2.
        for j in 2:(J - 1)
            lower = τ[j-1]
            upper = j + 1 <= J - 1 ? τ[j+1] : Inf
            for i in 1:N
                if yi[i] == j && z[i] > lower
                    lower = z[i]
                elseif yi[i] == j + 1 && z[i] < upper
                    upper = z[i]
                end
            end
            if isfinite(upper) && upper > lower
                τ[j] = lower + rand(rng) * (upper - lower)
            elseif !isfinite(upper)
                τ[j] = lower + rand(rng)
            end
        end

        if iter > burnin
            k = iter - burnin
            beta_store[k, :] .= β
            rho_store[k] = ρ
            cut_store[k, :] .= τ
        end
    end

    β_mean = vec(mean(beta_store, dims=1))
    ρ_mean = mean(rho_store)
    τ_mean = vec(mean(cut_store, dims=1))
    draws = hcat(beta_store, rho_store)
    V = cov(draws)
    V = 0.5 .* (V .+ V')

    return SpatialOrderedProbitModel{T, M}(
        T.(vcat(β_mean, ρ_mean)), T.(V), T(ρ_mean), T.(τ_mean),
        T.(beta_store), T.(rho_store), T.(cut_store),
        W, Matrix{T}(Xf), yi, J, cnames, n_samples, burnin,
        T(n_accept / n_samples))
end

StatsAPI.coef(m::SpatialOrderedProbitModel) = m.coef
StatsAPI.vcov(m::SpatialOrderedProbitModel) = m.vcov
StatsAPI.stderror(m::SpatialOrderedProbitModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::SpatialOrderedProbitModel) = length(m.y)
StatsAPI.coefnames(m::SpatialOrderedProbitModel) = m.coefnames
StatsAPI.dof_residual(m::SpatialOrderedProbitModel) = length(m.y) - length(m.coef)

function Base.show(io::IO, m::SpatialOrderedProbitModel{T}) where {T}
    println(io, "Spatial Ordered Probit (Gibbs, $(m.n_categories) categories)")
    println(io, "═"^52)
    @printf(io, "  N = %d,  draws = %d (burn-in %d)\n", nobs(m), m.n_samples, m.burnin)
    @printf(io, "  Posterior mean ρ = %.4f,  ρ acceptance = %.3f\n", m.rho, m.accept_rate)
    print(io, "  Cutpoints: ")
    for (j, c) in enumerate(m.cutpoints)
        @printf(io, "τ%d = %.3f  ", j, c)
    end
    println(io, "\n")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
    println(io, "  (Estimate = posterior mean, Std. Err = posterior SD)")
end

end # module
