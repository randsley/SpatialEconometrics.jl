module SpatialEconometricsPrototype

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, brent_maximize, _normcdf, _print_coef_table

const LogDetMethod = SpatialUtils.LogDetMethod
const ExactLU = SpatialUtils.ExactLU
const ChebyshevApprox = SpatialUtils.ChebyshevApprox

export SpatialVariant, SAR, SDM,
       LogDetMethod, ExactLU, ChebyshevApprox,
       SpatialEffectsResult, SpatialRegressionModel,
       fit_spatial, spatial_effects

const SpatialVariant = SpatialUtils.SpatialVariant
const SAR = SpatialUtils.SAR
const SDM = SpatialUtils.SDM

"""
    SpatialRegressionModel

A fitted cross-sectional spatial model, as returned by
[`fit_spatial`](@ref). Implements the `StatsAPI.RegressionModel` interface,
so `coef`, `vcov`, `nobs`, `loglikelihood`, `residuals` and `coefnames` all
work on it.

# Fields
- `variant`: the `SpatialVariant` that was fitted.
- `coef`: estimated coefficients, ordered as `coefnames`.
- `vcov`: covariance matrix of `coef`.
- `rho`, `rho_se`: the spatial autoregressive parameter and its standard
  error.
- `sigma2`: estimated error variance.
- `loglik`: maximized log-likelihood.
- `W`: the weight matrix used.
- `X_aug`: the design actually estimated, including the intercept and, for
  `SDM`, the `WX` block.
- `y`, `residuals`: response and residuals.
- `trace_powers`: traces of powers of `W`, reused by
  `spatial_effects`.
- `coefnames`: `"(Intercept)"`, the covariate names, then `W_`-prefixed
  names for an `SDM`.

!!! note
    `coef` holds the structural `β`, not marginal effects. In a spatial-lag
    model a change in `x` feeds back through `(I - ρW)^{-1}`, so use
    `spatial_effects` for interpretable quantities.
"""
struct SpatialRegressionModel{V<:SpatialVariant, T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    variant::V
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    rho_se::T
    sigma2::T
    loglik::T
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    trace_powers::Vector{T}
    coefnames::Vector{String}
end

"""
    SpatialEffectsResult

Direct, indirect and total marginal effects with standard errors, as
returned by `spatial_effects`. All six vectors are aligned with
`varnames`.

# Fields
- `varnames`: covariate names, excluding the intercept.
- `direct`, `direct_se`: the own-unit effect — the average diagonal of
  `∂y/∂x`.
- `indirect`, `indirect_se`: the spillover onto other units, i.e. the
  average off-diagonal row sum.
- `total`, `total_se`: `direct + indirect`.
"""
struct SpatialEffectsResult{T<:AbstractFloat}
    varnames::Vector{String}
    direct::Vector{T}
    direct_se::Vector{T}
    indirect::Vector{T}
    indirect_se::Vector{T}
    total::Vector{T}
    total_se::Vector{T}
end

function build_design_matrix(::SAR, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}, varnames::Vector{String}) where {T<:AbstractFloat}
    N = size(X, 1)
    X_aug = hcat(ones(T, N), X)
    names_aug = vcat(["(Intercept)"], varnames, ["ρ"])
    return X_aug, names_aug
end

function build_design_matrix(::SDM, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T}, varnames::Vector{String}) where {T<:AbstractFloat}
    N = size(X, 1)
    WX = W * X
    X_aug = hcat(ones(T, N), X, WX)
    w_names = ["W_" * v for v in varnames]
    names_aug = vcat(["(Intercept)"], varnames, w_names, ["ρ"])
    return X_aug, names_aug
end

"""
    fit_spatial(variant, W, X, y; method, varnames, rho_bounds)

Fit a cross-sectional spatial regression by concentrated maximum
likelihood.

The likelihood is concentrated so that `β` and `σ²` are solved in closed
form at each trial `ρ`, leaving a one-dimensional maximization over `ρ`
alone, which is done by Brent search within `rho_bounds`.

# Arguments
- `variant`: `SAR` or `SDM`.
- `W`: `N × N` sparse weight matrix, conventionally row-standardized.
- `X`: `N × K` covariates, **without** an intercept — one is added.
- `y`: response of length `N`.

# Keywords
- `method`: how `log|I - ρW|` is evaluated — `ExactLU` or
  `ChebyshevApprox` (default, order 6).
- `varnames`: covariate names; defaults to `x1, x2, …`.
- `rho_bounds`: search interval for `ρ`, default `(-0.995, 0.995)`. The
  bounds stay inside `±1` because `I - ρW` is singular at the edge of the
  stationary region.

# Returns
A [`SpatialRegressionModel`](@ref).

# Examples
```julia
W = make_rook_weights(10)
m = fit_spatial(SAR(), W, X, y; varnames = ["income", "density"])
m.rho
coef(m)
spatial_effects(m)          # interpretable effects, not coef
```

A formula interface is also available: `fit_spatial(@formula(y ~ x1 +
slag(x2)), table, W)`.

!!! note
    Read `coef` as structural parameters. `spatial_effects` gives
    the marginal effects, which differ whenever `ρ ≠ 0`.

See also `fit_spatial_panel`, `fit_spatial_error`,
`lm_tests` for choosing between lag and error specifications.

# References
LeSage, J. & Pace, R. K. (2009). *Introduction to Spatial Econometrics*.
CRC Press.
"""
function fit_spatial(
    variant::V,
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    rho_bounds::Tuple{T, T}=(T(-0.995), T(0.995))
) where {V<:SpatialVariant, T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert size(W) == (N, N) "Dimension mismatch between W and X"
    @assert length(y) == N "Dimension mismatch between y and X"

    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    X_aug, coefnames = build_design_matrix(variant, X, W, vnames)

    Wy = W * y
    qr_X = qr(X_aug)
    β0 = qr_X \ y
    βd = qr_X \ Wy

    e0 = y .- X_aug * β0
    ed = Wy .- X_aug * βd

    e0_e0 = dot(e0, e0)
    e0_ed = dot(e0, ed)
    ed_ed = dot(ed, ed)

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = ρ -> chebyshev_logdet(ρ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        logdet_fn = ρ -> first(logabsdet(lu(I_sparse - ρ * W)))
    end

    const_term = -T(0.5) * N * (log(T(2π)) + one(T))
    function concentrated_ll(ρ)
        rss = e0_e0 - 2 * ρ * e0_ed + (ρ^2) * ed_ed
        σ2 = rss / N
        return const_term + logdet_fn(ρ) - T(0.5) * N * log(σ2)
    end

    ρ_hat, max_ll = brent_maximize(concentrated_ll, rho_bounds[1], rho_bounds[2])

    β_hat = β0 .- ρ_hat .* βd
    res = e0 .- ρ_hat .* ed
    σ2_hat = dot(res, res) / N

    μ_traces = precompute_neumann_traces(W; max_power=60, n_probes=250)
    vcov_mat, ρ_se = compute_asymptotic_vcov(X_aug, W, β_hat, ρ_hat, σ2_hat, μ_traces)
    full_coef = vcat(β_hat, ρ_hat)

    return SpatialRegressionModel{V, T, M}(
        variant, full_coef, vcov_mat, ρ_hat, ρ_se, σ2_hat, max_ll,
        W, X_aug, Vector{T}(y), res, μ_traces, coefnames
    )
end

function compute_asymptotic_vcov(
    X_aug::Matrix{T},
    W::AbstractSparseMatrix{T},
    β::Vector{T},
    ρ::T,
    σ2::T,
    μ_traces::Vector{T}
) where {T<:AbstractFloat}
    N, P = size(X_aug)
    I_sparse = sparse(one(T) * I, N, N)
    A_lu = lu(I_sparse - ρ * W)
    Xb = X_aug * β
    GXb = W * (A_lu \ Xb)

    tr_G2 = zero(T)
    for m in 1:(length(μ_traces) - 2)
        tr_G2 += m * (ρ^(m - 1)) * (N * μ_traces[m + 2])
    end

    n_p = 30
    V_probe = rand((-one(T), one(T)), N, n_p)
    GV = A_lu \ (W * V_probe)
    tr_GtG = dot(GV, GV) / n_p

    I_ββ = (X_aug' * X_aug) ./ σ2
    I_βρ = (X_aug' * GXb) ./ σ2
    I_ρρ = (dot(GXb, GXb) / σ2) + tr_G2 + tr_GtG

    tr_G = zero(T)
    for m in 1:(length(μ_traces) - 1)
        tr_G += (ρ^(m - 1)) * (N * μ_traces[m + 1])
    end

    I_ρσ2 = tr_G / σ2
    I_σ2σ2 = N / (2 * σ2^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_ββ
    Info[1:P, P + 1] .= I_βρ
    Info[P + 1, 1:P] .= I_βρ
    Info[P + 1, P + 1] = I_ρρ
    Info[P + 1, P + 2] = I_ρσ2
    Info[P + 2, P + 1] = I_ρσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))
    V_coef = V_full[1:(P + 1), 1:(P + 1)]
    ρ_se = sqrt(max(zero(T), V_coef[P + 1, P + 1]))

    return V_coef, ρ_se
end

function spatial_effects(
    model::SpatialRegressionModel{V, T};
    max_power::Int=50,
    vcov_mat::AbstractMatrix{T}=model.vcov
) where {V<:SpatialVariant, T<:AbstractFloat}
    μ = model.trace_powers
    q = min(max_power, length(μ) - 2)
    θ_hat = model.coef
    Σ_θ = vcov_mat

    P_total = length(θ_hat) - 1
    K = V === SAR ? (P_total - 1) : div(P_total - 1, 2)

    direct = zeros(T, K); direct_se = zeros(T, K)
    indirect = zeros(T, K); indirect_se = zeros(T, K)
    total = zeros(T, K); total_se = zeros(T, K)

    for k in 1:K
        direct_fn = θ -> begin
            ρ = θ[end]
            β_k = θ[1 + k]
            γ_k = V === SDM ? θ[1 + K + k] : zero(eltype(θ))
            acc = zero(eltype(θ))
            for m in 0:q
                acc += (ρ^m) * (β_k * μ[m + 1] + γ_k * μ[m + 2])
            end
            return acc
        end

        total_fn = θ -> begin
            ρ = θ[end]
            β_k = θ[1 + k]
            γ_k = V === SDM ? θ[1 + K + k] : zero(eltype(θ))
            return (β_k + γ_k) / (one(eltype(θ)) - ρ)
        end

        indirect_fn = θ -> total_fn(θ) - direct_fn(θ)

        direct[k] = direct_fn(θ_hat)
        total[k] = total_fn(θ_hat)
        indirect[k] = indirect_fn(θ_hat)

        ∇_dir = ForwardDiff.gradient(direct_fn, θ_hat)
        ∇_tot = ForwardDiff.gradient(total_fn, θ_hat)
        ∇_ind = ForwardDiff.gradient(indirect_fn, θ_hat)

        direct_se[k] = sqrt(max(zero(T), dot(∇_dir, Σ_θ, ∇_dir)))
        total_se[k] = sqrt(max(zero(T), dot(∇_tot, Σ_θ, ∇_tot)))
        indirect_se[k] = sqrt(max(zero(T), dot(∇_ind, Σ_θ, ∇_ind)))
    end

    base_names = model.coefnames[2:(1 + K)]
    return SpatialEffectsResult{T}(
        base_names, direct, direct_se, indirect, indirect_se, total, total_se
    )
end

function Base.show(io::IO, m::SpatialRegressionModel{V}) where {V}
    vname = V === SAR ? "SAR" : "SDM"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Spatial $vname Model — Maximum Likelihood")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "ρ = %.4f    σ² = %.4f    Log-lik = %.2f    N = %d\n", m.rho, m.sigma2, m.loglik, length(m.y))
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

function Base.show(io::IO, e::SpatialEffectsResult)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Spatial Effects (LeSage & Pace)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "%-12s | %10s | %10s | %10s | %10s | %10s | %10s\n", "Variable", "Direct", "SE", "Indirect", "SE", "Total", "SE")
    println(io, "─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────")
    for i in eachindex(e.varnames)
        @printf(io, "%-12s | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n",
            e.varnames[i], e.direct[i], e.direct_se[i], e.indirect[i], e.indirect_se[i], e.total[i], e.total_se[i])
    end
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

StatsAPI.coef(m::SpatialRegressionModel) = m.coef
StatsAPI.vcov(m::SpatialRegressionModel) = m.vcov
StatsAPI.stderror(m::SpatialRegressionModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::SpatialRegressionModel) = m.loglik
StatsAPI.nobs(m::SpatialRegressionModel) = length(m.y)
StatsAPI.dof_residual(m::SpatialRegressionModel) = length(m.y) - length(m.coef)
StatsAPI.residuals(m::SpatialRegressionModel) = m.residuals
StatsAPI.coefnames(m::SpatialRegressionModel) = m.coefnames

end # module
