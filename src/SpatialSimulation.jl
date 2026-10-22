module SpatialSimulation

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI
import StatsAPI: coef

import ..SpatialDiagnostics: moran_global, lm_test_lag, lm_test_error, lm_test_robust_lag, lm_test_robust_error, MoranResult, LMTestResult

export SpatialDGP, SAR_DGP, SEM_DGP, SDM_DGP, SLX_DGP, SAC_DGP, SDEM_DGP,
       simulate_spatial, MonteCarloConfig, monte_carlo_spatial,
       MonteCarloResult, PowerAnalysisResult, power_analysis

# ═══════════════════════════════════════════════════════════════════════
# Data Generating Processes
# ═══════════════════════════════════════════════════════════════════════

"""
    SpatialDGP

A data-generating process for simulation, passed to
[`simulate_spatial`](@ref) and [`monte_carlo_spatial`](@ref).

Six processes are provided, spanning the usual specification ladder:
[`SAR_DGP`](@ref), [`SEM_DGP`](@ref), [`SDM_DGP`](@ref),
[`SLX_DGP`](@ref), [`SAC_DGP`](@ref), [`SDEM_DGP`](@ref).

!!! note
    None of them includes an intercept, while the estimators all fit one.
    [`monte_carlo_spatial`](@ref) drops the leading `(Intercept)`
    coefficient before comparing, so `beta` lines up with the estimates.
"""
abstract type SpatialDGP end

"""
    SAR_DGP(rho, beta, sigma)

Spatial autoregressive process: `y = rho*W*y + X*beta + e`, with
`e ~ N(0, sigma^2)`.

Drawn as `y = (I - rho*W)^{-1} (X*beta + e)`. Parameter order for
comparison is `beta` then `rho`.

# Examples
```julia
dgp = SAR_DGP(0.5, [1.5, -0.8], 0.5)
y, X = simulate_spatial(dgp, W, X)
```

See also [`SpatialDGP`](@ref).
"""
struct SAR_DGP{T<:AbstractFloat} <: SpatialDGP
    rho::T
    beta::Vector{T}
    sigma::T
end

"""
    SEM_DGP(lambda, beta, sigma)

Spatial error process: `y = X*beta + u` with `u = lambda*W*u + e`.

Dependence is in the disturbance only, so OLS stays unbiased but its
standard errors are wrong — which is what makes this the useful null for
`lm_test_error`. Parameter order is `beta` then `lambda`.

See also [`SpatialDGP`](@ref).
"""
struct SEM_DGP{T<:AbstractFloat} <: SpatialDGP
    lambda::T
    beta::Vector{T}
    sigma::T
end

"""
    SDM_DGP(rho, beta, theta, sigma)

Spatial Durbin process: `y = rho*W*y + X*beta + W*X*theta + e`.

Nests [`SAR_DGP`](@ref) at `theta = 0` and [`SLX_DGP`](@ref) at `rho = 0`,
so it is the natural process for checking
`common_factor_test`. Parameter order is `beta`, `theta`, `rho`.

See also [`SpatialDGP`](@ref).
"""
struct SDM_DGP{T<:AbstractFloat} <: SpatialDGP
    rho::T
    beta::Vector{T}
    theta::Vector{T}
    sigma::T
end

"""
    SLX_DGP(beta, theta, sigma)

Spatially lagged covariates only: `y = X*beta + W*X*theta + e`.

No feedback through `y`, so OLS on `[X W*X]` is consistent and the
spillovers are read straight off `theta` without inverting
`(I - rho*W)`. Parameter order is `beta` then `theta`.

See also [`SpatialDGP`](@ref).
"""
struct SLX_DGP{T<:AbstractFloat} <: SpatialDGP
    beta::Vector{T}
    theta::Vector{T}
    sigma::T
end

"""
    SAC_DGP(rho, lambda, beta, sigma)

Dependence in both `y` and the errors (SAC / SARAR):
`y = rho*W*y + X*beta + u`, `u = lambda*W*u + e`.

With one weight matrix doing both jobs, `rho` and `lambda` are only weakly
separated in finite samples — useful precisely for showing how hard the
specification search gets. Parameter order is `beta`, `rho`, `lambda`.

See also [`SpatialDGP`](@ref), `SARAR`.
"""
struct SAC_DGP{T<:AbstractFloat} <: SpatialDGP
    rho::T
    lambda::T
    beta::Vector{T}
    sigma::T
end

"""
    SDEM_DGP(lambda, beta, theta, sigma)

Spatial Durbin error process: `y = X*beta + W*X*theta + u` with
`u = lambda*W*u + e`.

Lagged covariates plus a spatial error, but no lag of `y`. Parameter order
is `beta`, `theta`, `lambda`.

See also [`SpatialDGP`](@ref).
"""
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

"""
    MonteCarloConfig(; n_replications = 500, seed = 42)

Replication count and seed for [`monte_carlo_spatial`](@ref).

The seed fixes the whole experiment, so a run is reproducible; raising
`n_replications` is what tightens the Monte Carlo standard error on the
reported bias and RMSE.
"""
struct MonteCarloConfig
    n_replications::Int
    seed::Int
end
MonteCarloConfig(; n_replications::Int=500, seed::Int=42) =
    MonteCarloConfig(n_replications, seed)

"""
    MonteCarloResult

Output of [`monte_carlo_spatial`](@ref): sampling behaviour of an estimator
over repeated draws from a known process.

# Fields
- `param_means`, `param_stds`: mean and standard deviation of the estimates
  across replications.
- `param_bias`: mean estimate minus the true value.
- `param_rmse`: root mean squared error, the summary that trades bias
  against variance.
- `param_names`: row labels, taken from the **DGP**.
- `rejection_rates`: empirical rejection rate per test — size when the null
  holds, power when it does not.
- `n_successful`, `n_replications`: replications that produced an estimate,
  and replications attempted. Failures are excluded rather than averaged
  in, so a gap between these two is itself a finding.

!!! warning
    `param_names` describes the DGP, not the estimator. Fitting a
    deliberately misspecified model still gives meaningful `beta` rows, but
    a trailing row labelled `rho` or `lambda` holds whatever parameter the
    estimator returned in that position.
"""
struct MonteCarloResult{T<:AbstractFloat}
    param_means::Vector{T}
    param_stds::Vector{T}
    param_bias::Vector{T}
    param_rmse::Vector{T}
    param_names::Vector{String}
    rejection_rates::Dict{String, T}
    n_successful::Int
    n_replications::Int
end

"""
    monte_carlo_spatial(dgp, W, N, K, estimator; config=MonteCarloConfig(), x_generator=nothing)

Run a Monte Carlo experiment. `estimator` is a function `(W, X, y) -> model` returning
a StatsAPI.RegressionModel. Returns `MonteCarloResult` with bias, RMSE, and test
rejection rates.

The DGPs generate `y` without an intercept while the estimators fit one, so a
leading `(Intercept)` coefficient is dropped before comparison; the remainder
lines up with the DGP's parameter ordering for all six DGPs.

!!! warning "Rows are labelled by the DGP, not the estimator"
    `param_names` comes from `dgp`, so the table is only fully meaningful when
    the estimator matches the DGP. Deliberately fitting a misspecified model
    still reports useful bias and RMSE for the `β` rows, but a trailing row
    labelled `ρ` or `λ` holds whichever parameter the estimator happens to
    return in that position — for an SLX fitted to SAR data, for instance, it
    is the `W_x1` coefficient.

Replications whose estimator throws are excluded and reported rather than
averaged in; `n_successful` records how many contributed.
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
    succeeded = falses(config.n_replications)
    n_diag = 0
    moran_reject = 0
    lm_lag_reject = 0
    lm_err_reject = 0

    for rep in 1:config.n_replications
        X_sim = gen_X()
        y_sim, _ = simulate_spatial(dgp, W, X_sim; rng=rng)

        try
            m = estimator(W, X_sim, y_sim)
            c = _coef_vector(m, n_params)
            for j in 1:min(n_params, length(c))
                param_store[rep, j] = c[j]
            end
            succeeded[rep] = true
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
            n_diag += 1
        catch
        end
    end

    # Summarise only the replications that actually produced an estimate.
    # Averaging in the failures would silently drag every mean toward zero and
    # report that as bias, which looks like a result rather than a malfunction.
    n_ok = count(succeeded)
    n_ok == 0 && error("monte_carlo_spatial: all $(config.n_replications) " *
                       "replications failed — the estimator threw on every draw")
    n_ok < config.n_replications &&
        @warn "monte_carlo_spatial: $(config.n_replications - n_ok) of " *
              "$(config.n_replications) replications failed and were excluded"

    kept = @view param_store[succeeded, :]
    means = vec(mean(kept, dims=1))
    stds = n_ok > 1 ? vec(std(kept, dims=1)) : zeros(T, n_params)
    bias = means .- true_params
    rmse = sqrt.(bias .^ 2 .+ stds .^ 2)
    pnames = _param_names(dgp)

    D = T(max(n_diag, 1))
    rej = Dict{String, T}(
        "moran_05" => moran_reject / D,
        "lm_lag_05" => lm_lag_reject / D,
        "lm_error_05" => lm_err_reject / D
    )

    return MonteCarloResult{T}(means, stds, bias, rmse, pnames, rej,
                               n_ok, config.n_replications)
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

# Align an estimator's coefficient vector with the DGP's true-parameter vector.
#
# The DGPs in this module generate y without an intercept (y = Xb + ...), but
# every estimator here fits one and reports it first. Taking the leading
# n_params coefficients therefore compares the intercept against beta_1, beta_1
# against beta_2, and so on — a silent off-by-one that shows up as enormous
# bias with a tiny standard deviation. Drop the leading intercept and the rest
# lines up for all six DGPs, because each estimator orders its output as
# [intercept, beta..., theta..., spatial params...].
function _coef_vector(m, n_params::Int)
    c = collect(Float64, StatsAPI.coef(m))
    names = try
        String.(StatsAPI.coefnames(m))
    catch
        String[]
    end
    drop = if !isempty(names) && occursin("Intercept", first(names))
        true
    else
        length(c) == n_params + 1      # fallback when names are unavailable
    end
    return drop && length(c) > 1 ? c[2:end] : c
end

function Base.show(io::IO, r::MonteCarloResult{T}) where {T}
    println(io, "Monte Carlo Simulation Results")
    println(io, "══════════════════════════════")
    println(io, "")
    if r.n_successful == r.n_replications
        @printf(io, "  %d replications

", r.n_replications)
    else
        @printf(io, "  %d of %d replications succeeded (%d failed and excluded)

",
            r.n_successful, r.n_replications, r.n_replications - r.n_successful)
    end
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

"""
    PowerAnalysisResult

Power surfaces from [`power_analysis`](@ref), for the three standard tests
for spatial dependence.

# Fields
- `sample_sizes`: the `N` values examined (one per grid side length).
- `rho_values`: the spatial parameter strengths examined.
- `power_moran`, `power_lm_lag`, `power_lm_error`: matrices of rejection
  rates, indexed `[sample size, rho]`.
- `alpha`: nominal size used.

Read the `rho = 0` column as a check on size: it should sit near `alpha`,
and a test that fails there has no credible power anywhere else.
"""
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
            # Count the replications each test actually evaluated. Dividing by
            # n_reps regardless would understate power whenever a test throws,
            # and report 0.0 — indistinguishable from "no power" — if every
            # replication failed.
            ok_m = 0; ok_lag = 0; ok_err = 0

            for _ in 1:n_reps
                X_sim = randn(rng, T, N, _n_covariates(dgp))
                y_sim, _ = simulate_spatial(dgp, W, X_sim; rng=rng)

                X_ols = hcat(ones(T, N), X_sim)
                β_ols = X_ols \ y_sim
                res_ols = y_sim .- X_ols * β_ols

                try
                    mr = moran_global(res_ols, W)
                    mr.p_value < alpha && (rej_m += 1)
                    ok_m += 1
                catch; end

                try
                    lt = lm_test_lag(y_sim, X_ols, W)
                    lt.p_value < alpha && (rej_lag += 1)
                    ok_lag += 1
                catch; end

                try
                    lt = lm_test_error(y_sim, X_ols, W)
                    lt.p_value < alpha && (rej_err += 1)
                    ok_err += 1
                catch; end
            end

            if min(ok_m, ok_lag, ok_err) < n_reps
                @warn "power_analysis: some diagnostics failed at N=$(N), rho=$(ρ)" *
                      " (Moran $(ok_m)/$(n_reps), LM-lag $(ok_lag)/$(n_reps)," *
                      " LM-error $(ok_err)/$(n_reps)); rates use the successful ones"
            end

            power_m[si, ri] = ok_m > 0 ? rej_m / ok_m : T(NaN)
            power_lag[si, ri] = ok_lag > 0 ? rej_lag / ok_lag : T(NaN)
            power_err[si, ri] = ok_err > 0 ? rej_err / ok_err : T(NaN)
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
