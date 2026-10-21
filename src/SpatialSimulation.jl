module SpatialSimulation

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

import ..SpatialDiagnostics: moran_global, lm_test_lag, lm_test_error, lm_test_robust_lag, lm_test_robust_error, MoranResult, LMTestResult

export SpatialDGP, SAR_DGP, SEM_DGP, SDM_DGP, SLX_DGP, SAC_DGP, SDEM_DGP,
       simulate_spatial, MonteCarloConfig, monte_carlo_spatial,
       MonteCarloResult, PowerAnalysisResult, power_analysis

# ═══════════════════════════════════════════════════════════════════════
# Data Generating Processes
# ═══════════════════════════════════════════════════════════════════════

abstract type SpatialDGP end

struct SAR_DGP{T<:AbstractFloat} <: SpatialDGP
    rho::T
    beta::Vector{T}
    sigma::T
end

struct SEM_DGP{T<:AbstractFloat} <: SpatialDGP
    lambda::T
    beta::Vector{T}
    sigma::T
end

struct SDM_DGP{T<:AbstractFloat} <: SpatialDGP
    rho::T
    beta::Vector{T}
    theta::Vector{T}
    sigma::T
end

struct SLX_DGP{T<:AbstractFloat} <: SpatialDGP
    beta::Vector{T}
    theta::Vector{T}
    sigma::T
end

struct SAC_DGP{T<:AbstractFloat} <: SpatialDGP
    rho::T
    lambda::T
    beta::Vector{T}
    sigma::T
end

struct SDEM_DGP{T<:AbstractFloat} <: SpatialDGP
    lambda::T
    beta::Vector{T}
    theta::Vector{T}
    sigma::T
end

"""
    simulate_spatial(dgp, W, X; rng=Random.default_rng())

Generate one realization from the specified spatial DGP.
Returns `(y, X)`.
"""
function simulate_spatial(
    dgp::SAR_DGP{T}, W::AbstractSparseMatrix{T}, X::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng()
) where {T}
    N = size(X, 1)
    ε = dgp.sigma .* randn(rng, T, N)
    A = sparse(one(T) * I, N, N) - dgp.rho * W
    y = A \ (X * dgp.beta .+ ε)
    return y, X
end

function simulate_spatial(
    dgp::SEM_DGP{T}, W::AbstractSparseMatrix{T}, X::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng()
) where {T}
    N = size(X, 1)
    ε = dgp.sigma .* randn(rng, T, N)
    B = sparse(one(T) * I, N, N) - dgp.lambda * W
    u = B \ ε
    y = X * dgp.beta .+ u
    return y, X
end

function simulate_spatial(
    dgp::SDM_DGP{T}, W::AbstractSparseMatrix{T}, X::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng()
) where {T}
    N = size(X, 1)
    ε = dgp.sigma .* randn(rng, T, N)
    WX = W * X
    A = sparse(one(T) * I, N, N) - dgp.rho * W
    y = A \ (X * dgp.beta .+ WX * dgp.theta .+ ε)
    return y, X
end

function simulate_spatial(
    dgp::SLX_DGP{T}, W::AbstractSparseMatrix{T}, X::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng()
) where {T}
    N = size(X, 1)
    ε = dgp.sigma .* randn(rng, T, N)
    WX = W * X
    y = X * dgp.beta .+ WX * dgp.theta .+ ε
    return y, X
end

function simulate_spatial(
    dgp::SAC_DGP{T}, W::AbstractSparseMatrix{T}, X::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng()
) where {T}
    N = size(X, 1)
    ε = dgp.sigma .* randn(rng, T, N)
    B = sparse(one(T) * I, N, N) - dgp.lambda * W
    u = B \ ε
    A = sparse(one(T) * I, N, N) - dgp.rho * W
    y = A \ (X * dgp.beta .+ u)
    return y, X
end

function simulate_spatial(
    dgp::SDEM_DGP{T}, W::AbstractSparseMatrix{T}, X::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng()
) where {T}
    N = size(X, 1)
    ε = dgp.sigma .* randn(rng, T, N)
    WX = W * X
    B = sparse(one(T) * I, N, N) - dgp.lambda * W
    u = B \ ε
    y = X * dgp.beta .+ WX * dgp.theta .+ u
    return y, X
end

# ═══════════════════════════════════════════════════════════════════════
# Monte Carlo Framework
# ═══════════════════════════════════════════════════════════════════════

struct MonteCarloConfig
    n_replications::Int
    seed::Int
end
MonteCarloConfig(; n_replications::Int=500, seed::Int=42) =
    MonteCarloConfig(n_replications, seed)

struct MonteCarloResult{T<:AbstractFloat}
    param_means::Vector{T}
    param_stds::Vector{T}
    param_bias::Vector{T}
    param_rmse::Vector{T}
    param_names::Vector{String}
    rejection_rates::Dict{String, T}
end

"""
    monte_carlo_spatial(dgp, W, N, K, estimator; config=MonteCarloConfig(), x_generator=nothing)

Run a Monte Carlo experiment. `estimator` is a function `(W, X, y) -> model` returning
a StatsAPI.RegressionModel. Returns `MonteCarloResult` with bias, RMSE, and test
rejection rates.
"""
function monte_carlo_spatial(
    dgp::SpatialDGP,
    W::AbstractSparseMatrix{T},
    N::Int,
    K::Int,
    estimator::Function;
    config::MonteCarloConfig=MonteCarloConfig(),
    x_generator::Union{Nothing, Function}=nothing
) where {T<:AbstractFloat}

    rng = Random.MersenneTwister(config.seed)
    gen_X = isnothing(x_generator) ? () -> randn(rng, T, N, K) : x_generator

    true_params = _extract_true_params(dgp)
    n_params = length(true_params)
    param_store = zeros(T, config.n_replications, n_params)
    moran_reject = 0
    lm_lag_reject = 0
    lm_err_reject = 0

    for rep in 1:config.n_replications
        X_sim = gen_X()
        y_sim, _ = simulate_spatial(dgp, W, X_sim; rng=rng)

        try
            m = estimator(W, X_sim, y_sim)
            c = _coef_vector(m)
            for j in 1:min(n_params, length(c))
                param_store[rep, j] = c[j]
            end
        catch
            continue
        end

        try
            X_ols = hcat(ones(T, N), X_sim)
            β_ols = X_ols \ y_sim
            res_ols = y_sim .- X_ols * β_ols
            mr = moran_global(res_ols, W)
            mr.p_value < T(0.05) && (moran_reject += 1)
            lt_lag = lm_test_lag(y_sim, X_ols, W)
            lt_lag.p_value < T(0.05) && (lm_lag_reject += 1)
            lt_err = lm_test_error(y_sim, X_ols, W)
            lt_err.p_value < T(0.05) && (lm_err_reject += 1)
        catch
        end
    end

    means = vec(mean(param_store, dims=1))
    stds = vec(std(param_store, dims=1))
    bias = means .- true_params
    rmse = sqrt.(bias .^ 2 .+ stds .^ 2)
    pnames = _param_names(dgp)

    R = T(config.n_replications)
    rej = Dict{String, T}(
        "moran_05" => moran_reject / R,
        "lm_lag_05" => lm_lag_reject / R,
        "lm_error_05" => lm_err_reject / R
    )

    return MonteCarloResult{T}(means, stds, bias, rmse, pnames, rej)
end

function _extract_true_params(dgp::SAR_DGP{T}) where {T}
    return vcat(dgp.beta, dgp.rho)
end
function _extract_true_params(dgp::SEM_DGP{T}) where {T}
    return vcat(dgp.beta, dgp.lambda)
end
function _extract_true_params(dgp::SDM_DGP{T}) where {T}
    return vcat(dgp.beta, dgp.theta, dgp.rho)
end
function _extract_true_params(dgp::SLX_DGP{T}) where {T}
    return vcat(dgp.beta, dgp.theta)
end
function _extract_true_params(dgp::SAC_DGP{T}) where {T}
    return vcat(dgp.beta, dgp.rho, dgp.lambda)
end
function _extract_true_params(dgp::SDEM_DGP{T}) where {T}
    return vcat(dgp.beta, dgp.theta, dgp.lambda)
end

function _param_names(dgp::SAR_DGP)
    return vcat(["β$i" for i in 1:length(dgp.beta)], ["ρ"])
end
function _param_names(dgp::SEM_DGP)
    return vcat(["β$i" for i in 1:length(dgp.beta)], ["λ"])
end
function _param_names(dgp::SDM_DGP)
    return vcat(["β$i" for i in 1:length(dgp.beta)], ["θ$i" for i in 1:length(dgp.theta)], ["ρ"])
end
function _param_names(dgp::SLX_DGP)
    return vcat(["β$i" for i in 1:length(dgp.beta)], ["θ$i" for i in 1:length(dgp.theta)])
end
function _param_names(dgp::SAC_DGP)
    return vcat(["β$i" for i in 1:length(dgp.beta)], ["ρ", "λ"])
end
function _param_names(dgp::SDEM_DGP)
    return vcat(["β$i" for i in 1:length(dgp.beta)], ["θ$i" for i in 1:length(dgp.theta)], ["λ"])
end

_coef_vector(m) = collect(Float64, StatsAPI.coef(m))

function Base.show(io::IO, r::MonteCarloResult{T}) where {T}
    println(io, "Monte Carlo Simulation Results")
    println(io, "══════════════════════════════")
    println(io, "")
    @printf(io, "  %-10s %10s %10s %10s %10s\n", "Param", "Mean", "Std", "Bias", "RMSE")
    println(io, "  ", "─"^50)
    for i in eachindex(r.param_names)
        @printf(io, "  %-10s %10.4f %10.4f %10.4f %10.4f\n",
            r.param_names[i], r.param_means[i], r.param_stds[i],
            r.param_bias[i], r.param_rmse[i])
    end
    println(io, "")
    println(io, "  Test Rejection Rates (α=0.05):")
    for (k, v) in sort(collect(r.rejection_rates))
        @printf(io, "    %-20s %.3f\n", k, v)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Power Analysis
# ═══════════════════════════════════════════════════════════════════════

struct PowerAnalysisResult{T<:AbstractFloat}
    sample_sizes::Vector{Int}
    rho_values::Vector{T}
    power_moran::Matrix{T}
    power_lm_lag::Matrix{T}
    power_lm_error::Matrix{T}
    alpha::T
end

"""
    power_analysis(dgp_constructor, grid_sizes, rho_values; n_reps=200, alpha=0.05, seed=42)

Compute power curves for Moran's I, LM-lag, and LM-error across sample sizes
and spatial parameter strengths.

`dgp_constructor(rho) -> SpatialDGP` returns a DGP parameterised by the spatial
parameter of interest.
`grid_sizes` is a vector of grid side-lengths (N = L²).
"""
function power_analysis(
    dgp_constructor::Function,
    grid_sizes::Vector{Int},
    rho_values::Vector{T};
    n_reps::Int=200,
    alpha::T=T(0.05),
    seed::Int=42
) where {T<:AbstractFloat}

    n_sizes = length(grid_sizes)
    n_rhos = length(rho_values)
    power_m = zeros(T, n_sizes, n_rhos)
    power_lag = zeros(T, n_sizes, n_rhos)
    power_err = zeros(T, n_sizes, n_rhos)
    sample_sizes = [L^2 for L in grid_sizes]

    for (si, L) in enumerate(grid_sizes)
        N = L^2
        W = _make_rook(L, T)
        rng = Random.MersenneTwister(seed)

        for (ri, ρ) in enumerate(rho_values)
            dgp = dgp_constructor(ρ)
            rej_m = 0; rej_lag = 0; rej_err = 0

            for _ in 1:n_reps
                X_sim = randn(rng, T, N, _n_covariates(dgp))
                y_sim, _ = simulate_spatial(dgp, W, X_sim; rng=rng)

                X_ols = hcat(ones(T, N), X_sim)
                β_ols = X_ols \ y_sim
                res_ols = y_sim .- X_ols * β_ols

                try
                    mr = moran_global(res_ols, W)
                    mr.p_value < alpha && (rej_m += 1)
                catch; end

                try
                    lt = lm_test_lag(y_sim, X_ols, W)
                    lt.p_value < alpha && (rej_lag += 1)
                catch; end

                try
                    lt = lm_test_error(y_sim, X_ols, W)
                    lt.p_value < alpha && (rej_err += 1)
                catch; end
            end

            power_m[si, ri] = rej_m / n_reps
            power_lag[si, ri] = rej_lag / n_reps
            power_err[si, ri] = rej_err / n_reps
        end
    end

    return PowerAnalysisResult{T}(sample_sizes, rho_values, power_m, power_lag, power_err, alpha)
end

_n_covariates(dgp::SAR_DGP) = length(dgp.beta)
_n_covariates(dgp::SEM_DGP) = length(dgp.beta)
_n_covariates(dgp::SDM_DGP) = length(dgp.beta)
_n_covariates(dgp::SLX_DGP) = length(dgp.beta)
_n_covariates(dgp::SAC_DGP) = length(dgp.beta)
_n_covariates(dgp::SDEM_DGP) = length(dgp.beta)

function _make_rook(L::Int, ::Type{T}) where {T}
    N = L * L; I_idx = Int[]; J_idx = Int[]; V_val = T[]
    for r in 1:L, c in 1:L
        u = (r - 1) * L + c; nb = Int[]
        r > 1 && push!(nb, (r - 2) * L + c); r < L && push!(nb, r * L + c)
        c > 1 && push!(nb, (r - 1) * L + (c - 1)); c < L && push!(nb, (r - 1) * L + (c + 1))
        w = one(T) / length(nb)
        for v in nb; push!(I_idx, u); push!(J_idx, v); push!(V_val, w); end
    end
    return sparse(I_idx, J_idx, V_val, N, N)
end

function Base.show(io::IO, r::PowerAnalysisResult{T}) where {T}
    println(io, "Power Analysis Results (α=$(r.alpha))")
    println(io, "═"^45)
    println(io, "Moran's I rejection rates:")
    @printf(io, "  %8s", "N \\ ρ")
    for ρ in r.rho_values; @printf(io, " %8.2f", ρ); end
    println(io)
    for (i, n) in enumerate(r.sample_sizes)
        @printf(io, "  %8d", n)
        for j in eachindex(r.rho_values)
            @printf(io, " %8.3f", r.power_moran[i, j])
        end
        println(io)
    end
end

end # module
