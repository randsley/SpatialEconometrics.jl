module SpatialEconometricsTuringExt

using SpatialEconometrics
using Turing
using Distributions
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

# Methods are added to the stubs declared in the parent package. An
# extension cannot introduce reachable functions of its own: nothing ever
# `using`s this module, so a function defined here would be invisible.
import SpatialEconometrics: bayesian_sar, bayesian_sdm, bayesian_sem

const _CS = SpatialEconometrics.CrossSection

"""
    BayesianSpatialResult

Posterior summary of a Bayesian spatial fit. `chain` holds the raw
`MCMCChains.Chains` object; `posterior_mean` and `posterior_std` are
aligned with `coefnames`.
"""
struct BayesianSpatialResult{T<:AbstractFloat, C} <: SpatialEconometrics.AbstractBayesianSpatialResult
    chain::C
    model_type::Symbol
    coefnames::Vector{String}
    posterior_mean::Vector{T}
    posterior_std::Vector{T}
    W::AbstractSparseMatrix{T}
    N::Int
end

function Base.show(io::IO, r::BayesianSpatialResult)
    println(io, "═"^71)
    println(io, "Bayesian Spatial $(r.model_type) — MCMC Posterior Summary")
    println(io, "─"^71)
    @printf(io, "%-14s | %10s | %10s\n", "Parameter", "Post. Mean", "Post. SD")
    println(io, "───────────────┼────────────┼────────────")
    for i in eachindex(r.coefnames)
        @printf(io, "%-14s | %10.4f | %10.4f\n",
                r.coefnames[i], r.posterior_mean[i], r.posterior_std[i])
    end
    println(io, "─"^71)
    println(io, "N = $(r.N)    Samples = $(size(r.chain, 1))    " *
                "Chains = $(_n_chains(r.chain))")
    println(io, "═"^71)
end

# ─── Shared plumbing ─────────────────────────────────────────────────

_traces(W) = _CS.precompute_chebyshev_traces(W, _CS.ChebyshevApprox(order=12))

# log N(resid | 0, σ²I).
_gaussian_loglik(resid, σ, N) =
    -0.5 * N * log(2π) - N * log(σ) - sum(abs2, resid) / (2 * σ^2)

# Draw the posterior. MCMCThreads only helps when Julia actually has
# threads; asking for it with one thread just produces a warning.
function _draw(rng, model, n_samples::Int, n_chains::Int)
    n_chains <= 1 && return sample(rng, model, NUTS(), n_samples)
    ensemble = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()
    return sample(rng, model, NUTS(), ensemble, n_samples, n_chains)
end

# Chain introspection, written against two incompatible back ends.
#
# Turing switched from MCMCChains to FlexiChains in 0.45. They differ in
# both respects we depend on:
#
#   MCMCChains.Chains      3-D (draws, params, chains); a vector-valued
#                          parameter is split across keys β[1], β[2], …
#   FlexiChains.FlexiChain 2-D (draws, chains); a vector-valued parameter
#                          lives under the single key β, whose entries are
#                          one Vector per draw.
#
# Scalar indexing — chain[:ρ] — happens to agree, so only the vector case
# and the chain count need bridging.

_n_chains(chain) = (s = size(chain); length(s) == 3 ? s[3] : s[2])

_scalar_draws(chain, name::Symbol) = vec(collect(chain[name]))

function _vector_draws(chain, name::Symbol, K::Int)
    try
        return [vec(collect(chain[Symbol("$(name)[$i]")])) for i in 1:K]
    catch
        packed = vec(collect(chain[name]))
        (!isempty(packed) && length(first(packed)) == K) || error(
            "cannot read $K draws of `$(name)` out of a " *
            "$(nameof(typeof(chain))); this Turing version stores chains " *
            "in a layout the extension does not recognise.")
        return [[d[i] for d in packed] for i in 1:K]
    end
end

# Posterior mean/SD for β[1..K] followed by the named scalar parameters.
function _summarize(::Type{T}, chain, K::Int, scalars) where {T<:AbstractFloat}
    post_mean = T[]; post_std = T[]
    for draws in _vector_draws(chain, :β, K)
        push!(post_mean, T(mean(draws))); push!(post_std, T(std(draws)))
    end
    for s in scalars
        draws = _scalar_draws(chain, s)
        push!(post_mean, T(mean(draws))); push!(post_std, T(std(draws)))
    end
    return post_mean, post_std
end

_slope_names(varnames, K_raw) =
    isnothing(varnames) ? ["x$i" for i in 1:K_raw] : String.(varnames)

# ─── Bayesian SAR ────────────────────────────────────────────────────

@model function _sar_model(y, X_aug, W, τ, N, K)
    β ~ MvNormal(zeros(K), 10.0 * I)
    ρ ~ Uniform(-0.99, 0.99)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # The whole likelihood goes through @addlogprob!. `~` cannot be used
    # here: its left-hand side would be a local variable rather than a
    # model argument, so DynamicPPL would treat it as one more latent
    # variable to sample and the data would never reach the posterior.
    #
    #   log L = log|I - ρW| + log N((I - ρW)y - Xβ | 0, σ²I)
    Turing.@addlogprob! begin
        resid = y .- ρ .* (W * y) .- X_aug * β
        _CS.chebyshev_logdet(ρ, τ) + _gaussian_loglik(resid, σ, N)
    end
end

"""
    bayesian_sar(W, X, y; n_samples=1000, n_chains=2, varnames=nothing, rng=...)

Bayesian SAR by NUTS. Returns a [`BayesianSpatialResult`](@ref).
"""
function bayesian_sar(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=1000,
    n_chains::Int=2,
    varnames::Union{Nothing, Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    length(y) == N || throw(DimensionMismatch("y has $(length(y)) rows, X has $N"))
    size(W, 1) == N || throw(DimensionMismatch("W is $(size(W)), X has $N rows"))

    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = vcat(["(Intercept)"], _slope_names(varnames, K_raw), ["ρ", "σ"])

    model = _sar_model(Vector{T}(y), Matrix{T}(X_aug), W, _traces(W), N, K)
    chain = _draw(rng, model, n_samples, n_chains)
    post_mean, post_std = _summarize(T, chain, K, (:ρ, :σ))

    return BayesianSpatialResult(chain, :SAR, vnames, post_mean, post_std, W, N)
end

# ─── Bayesian SEM ────────────────────────────────────────────────────

@model function _sem_model(y, X_aug, W, τ, N, K)
    β ~ MvNormal(zeros(K), 10.0 * I)
    λ ~ Uniform(-0.99, 0.99)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # y = Xβ + (I - λW)⁻¹ε, so the filtered residual is the i.i.d. one:
    #   log L = log|I - λW| + log N((I - λW)(y - Xβ) | 0, σ²I)
    Turing.@addlogprob! begin
        resid = y .- X_aug * β
        filtered = resid .- λ .* (W * resid)
        _CS.chebyshev_logdet(λ, τ) + _gaussian_loglik(filtered, σ, N)
    end
end

"""
    bayesian_sem(W, X, y; n_samples=1000, n_chains=2, varnames=nothing, rng=...)

Bayesian spatial error model by NUTS. Returns a [`BayesianSpatialResult`](@ref).
"""
function bayesian_sem(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=1000,
    n_chains::Int=2,
    varnames::Union{Nothing, Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    length(y) == N || throw(DimensionMismatch("y has $(length(y)) rows, X has $N"))
    size(W, 1) == N || throw(DimensionMismatch("W is $(size(W)), X has $N rows"))

    X_aug = hcat(ones(T, N), X)
    K = K_raw + 1
    vnames = vcat(["(Intercept)"], _slope_names(varnames, K_raw), ["λ", "σ"])

    model = _sem_model(Vector{T}(y), Matrix{T}(X_aug), W, _traces(W), N, K)
    chain = _draw(rng, model, n_samples, n_chains)
    post_mean, post_std = _summarize(T, chain, K, (:λ, :σ))

    return BayesianSpatialResult(chain, :SEM, vnames, post_mean, post_std, W, N)
end

# ─── Bayesian SDM ────────────────────────────────────────────────────

@model function _sdm_model(y, X_aug, W, τ, N, K)
    β ~ MvNormal(zeros(K), 10.0 * I)
    ρ ~ Uniform(-0.99, 0.99)
    σ ~ truncated(Normal(0, 5), 0, Inf)

    # Identical to the SAR likelihood; `X_aug` is what differs — it
    # carries the WX terms.
    Turing.@addlogprob! begin
        resid = y .- ρ .* (W * y) .- X_aug * β
        _CS.chebyshev_logdet(ρ, τ) + _gaussian_loglik(resid, σ, N)
    end
end

"""
    bayesian_sdm(W, X, y; n_samples=1000, n_chains=2, varnames=nothing, rng=...)

Bayesian spatial Durbin model by NUTS: the design is augmented with `WX`.
Returns a [`BayesianSpatialResult`](@ref).
"""
function bayesian_sdm(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    n_samples::Int=1000,
    n_chains::Int=2,
    varnames::Union{Nothing, Vector{String}}=nothing,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    N, K_raw = size(X)
    length(y) == N || throw(DimensionMismatch("y has $(length(y)) rows, X has $N"))
    size(W, 1) == N || throw(DimensionMismatch("W is $(size(W)), X has $N rows"))

    WX = W * X
    X_aug = hcat(ones(T, N), X, WX)
    K = 1 + 2 * K_raw
    vn = _slope_names(varnames, K_raw)
    vnames = vcat(["(Intercept)"], vn, ["W_" * v for v in vn], ["ρ", "σ"])

    model = _sdm_model(Vector{T}(y), Matrix{T}(X_aug), W, _traces(W), N, K)
    chain = _draw(rng, model, n_samples, n_chains)
    post_mean, post_std = _summarize(T, chain, K, (:ρ, :σ))

    return BayesianSpatialResult(chain, :SDM, vnames, post_mean, post_std, W, N)
end

end # module
