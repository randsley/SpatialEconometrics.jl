module SpatialEconometricsTuringExt

using SpatialEconometrics
using Turing
using Distributions
using LinearAlgebra
using SparseArrays

export bayesian_sar, bayesian_sdm, bayesian_sem, BayesianSpatialResult

struct BayesianSpatialResult{T<:AbstractFloat}
    chain::Turing.Chains
    model_type::Symbol
    coefnames::Vector{String}
    posterior_mean::Vector{T}
    posterior_std::Vector{T}
    W::AbstractSparseMatrix{T}
    N::Int
end

function Base.show(io::IO, r::BayesianSpatialResult)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Bayesian Spatial $(r.model_type) — MCMC Posterior Summary")
    println(io, "───────────────────────────────────────────────────────────────────────")
    Printf.@printf(io, "%-14s | %10s | %10s\n", "Parameter", "Post. Mean", "Post. SD")
    println(io, "───────────────┼────────────┼────────────")
    for i in eachindex(r.coefnames)
        Printf.@printf(io, "%-14s | %10.4f | %10.4f\n", r.coefnames[i], r.posterior_mean[i], r.posterior_std[i])
    end
    println(io, "───────────────────────────────────────────────────────────────────────")
    println(io, "N = $(r.N)    Samples = $(size(r.chain, 1))    Chains = $(size(r.chain, 3))")
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─── Bayesian SAR ─────────────────────────────────────────────────────

@model function _sar_model(y, X_aug, W, τ, N, K)
    # Priors
    β ~ MvNormal(zeros(K), 10.0 * I)
    ρ ~ Uniform(-0.99, 0.99)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # Log-determinant via Chebyshev
    logdet_val = SpatialEconometrics.CrossSection.chebyshev_logdet(ρ, τ)
    Turing.@addlogprob! logdet_val

    # Likelihood: y = (I - ρW)^{-1}(Xβ + ε)
    # Equivalently: (I - ρW)y ~ N(Xβ, σ²I)
    Ay = y .- ρ .* (W * y)
    μ = X_aug * β
    Ay ~ MvNormal(μ, σ^2 * I)
end

function bayesian_sar(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=1000,
    n_chains::Int=2,
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = isnothing(varnames) ? vcat(["(Intercept)"], ["x$i" for i in 1:K_raw], ["ρ", "σ"]) : vcat(["(Intercept)"], varnames, ["ρ", "σ"])

    τ = SpatialEconometrics.CrossSection.precompute_chebyshev_traces(W, SpatialEconometrics.CrossSection.ChebyshevApprox(order=8))

    model = _sar_model(Vector{T}(y), Matrix{T}(X_aug), W, τ, N, K)
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

    # Extract posterior summaries
    param_names = vnames
    post_mean = T[]
    post_std = T[]
    for i in 1:K
        sym = Symbol("β[$i]")
        push!(post_mean, T(mean(chain[sym])))
        push!(post_std, T(std(chain[sym])))
    end
    push!(post_mean, T(mean(chain[:ρ])))
    push!(post_std, T(std(chain[:ρ])))
    push!(post_mean, T(mean(chain[:σ])))
    push!(post_std, T(std(chain[:σ])))

    return BayesianSpatialResult{T}(chain, :SAR, param_names, post_mean, post_std, W, N)
end

# ─── Bayesian SEM ─────────────────────────────────────────────────────

@model function _sem_model(y, X_aug, W, τ, N, K)
    β ~ MvNormal(zeros(K), 10.0 * I)
    λ ~ Uniform(-0.99, 0.99)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    logdet_val = SpatialEconometrics.CrossSection.chebyshev_logdet(λ, τ)
    Turing.@addlogprob! logdet_val

    # SEM: y = Xβ + (I-λW)^{-1}ε
    # Transform: (I-λW)y - (I-λW)Xβ ~ N(0, σ²I)  ... no
    # Actually: (I-λW)(y - Xβ) ~ N(0, σ²I)
    resid = y .- X_aug * β
    filtered = resid .- λ .* (W * resid)
    filtered ~ MvNormal(zeros(N), σ^2 * I)
end

function bayesian_sem(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=1000,
    n_chains::Int=2,
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = isnothing(varnames) ? vcat(["(Intercept)"], ["x$i" for i in 1:K_raw], ["λ", "σ"]) : vcat(["(Intercept)"], varnames, ["λ", "σ"])

    τ = SpatialEconometrics.CrossSection.precompute_chebyshev_traces(W, SpatialEconometrics.CrossSection.ChebyshevApprox(order=8))

    model = _sem_model(Vector{T}(y), Matrix{T}(X_aug), W, τ, N, K)
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

    param_names = vnames
    post_mean = T[]
    post_std = T[]
    for i in 1:K
        sym = Symbol("β[$i]")
        push!(post_mean, T(mean(chain[sym])))
        push!(post_std, T(std(chain[sym])))
    end
    push!(post_mean, T(mean(chain[:λ])))
    push!(post_std, T(std(chain[:λ])))
    push!(post_mean, T(mean(chain[:σ])))
    push!(post_std, T(std(chain[:σ])))

    return BayesianSpatialResult{T}(chain, :SEM, param_names, post_mean, post_std, W, N)
end

# ─── Bayesian SDM ─────────────────────────────────────────────────────

@model function _sdm_model(y, X_aug, W, τ, N, K)
    β ~ MvNormal(zeros(K), 10.0 * I)
    ρ ~ Uniform(-0.99, 0.99)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    logdet_val = SpatialEconometrics.CrossSection.chebyshev_logdet(ρ, τ)
    Turing.@addlogprob! logdet_val

    Ay = y .- ρ .* (W * y)
    μ = X_aug * β
    Ay ~ MvNormal(μ, σ^2 * I)
end

function bayesian_sdm(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=1000,
    n_chains::Int=2,
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    WX = W * X
    X_aug = hcat(ones(T, N), X, WX)
    K = 1 + 2 * K_raw
    vn = isnothing(varnames) ? ["x$i" for i in 1:K_raw] : varnames
    vnames = vcat(["(Intercept)"], vn, ["W_" * v for v in vn], ["ρ", "σ"])

    τ = SpatialEconometrics.CrossSection.precompute_chebyshev_traces(W, SpatialEconometrics.CrossSection.ChebyshevApprox(order=8))

    model = _sdm_model(Vector{T}(y), Matrix{T}(X_aug), W, τ, N, K)
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains)

    param_names = vnames
    post_mean = T[]
    post_std = T[]
    for i in 1:K
        sym = Symbol("β[$i]")
        push!(post_mean, T(mean(chain[sym])))
        push!(post_std, T(std(chain[sym])))
    end
    push!(post_mean, T(mean(chain[:ρ])))
    push!(post_std, T(std(chain[:ρ])))
    push!(post_mean, T(mean(chain[:σ])))
    push!(post_std, T(std(chain[:σ])))

    return BayesianSpatialResult{T}(chain, :SDM, param_names, post_mean, post_std, W, N)
end

end # module
