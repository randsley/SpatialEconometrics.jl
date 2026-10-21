module SpatialRegimeModels

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI
import StatsAPI: loglikelihood, coef, coefnames

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       brent_maximize, _normcdf, _print_coef_table

const LogDetMethod = SpatialUtils.LogDetMethod
const ExactLU = SpatialUtils.ExactLU
const ChebyshevApprox = SpatialUtils.ChebyshevApprox

export SpatialRegimeModel, fit_spatial_regime,
       RegimeAssignment, ExogenousRegimes, skater_regimes,
       ChowRegimeTest, regime_chow_test

# ═══════════════════════════════════════════════════════════════════════
# Regime Assignment
# ═══════════════════════════════════════════════════════════════════════

abstract type RegimeAssignment end

struct ExogenousRegimes <: RegimeAssignment
    labels::Vector{Int}
end

"""
    skater_regimes(W, X, n_regimes; rng=Random.default_rng())

Spatially constrained clustering via SKATER-like minimum spanning tree partitioning.
Returns an `ExogenousRegimes` assignment with `n_regimes` contiguous groups.

Assunção et al. (2006) "Efficient regionalization techniques for socio‐economic
geographical units using minimum spanning trees."
"""
function skater_regimes(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    n_regimes::Int;
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    N = size(W, 1)
    @assert n_regimes >= 2 && n_regimes <= N

    X_std = copy(X)
    for j in axes(X_std, 2)
        μ = mean(@view X_std[:, j])
        σ = std(@view X_std[:, j])
        σ > zero(T) && (X_std[:, j] .= (X_std[:, j] .- μ) ./ σ)
    end

    rows = rowvals(W)
    edges = Tuple{Int, Int, T}[]
    for j in 1:N
        for idx in nzrange(W, j)
            i = rows[idx]
            i > j || continue
            d = norm(@view(X_std[i, :]) .- @view(X_std[j, :]))
            push!(edges, (i, j, d))
        end
    end
    sort!(edges, by=e -> e[3])

    parent = collect(1:N)
    rank = zeros(Int, N)
    function find_root(x)
        while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end
        return x
    end
    function union!(a, b)
        ra, rb = find_root(a), find_root(b)
        ra == rb && return false
        if rank[ra] < rank[rb]; parent[ra] = rb
        elseif rank[ra] > rank[rb]; parent[rb] = ra
        else parent[rb] = ra; rank[ra] += 1
        end
        return true
    end

    mst_edges = Tuple{Int, Int, T}[]
    for (i, j, d) in edges
        if union!(i, j)
            push!(mst_edges, (i, j, d))
            length(mst_edges) == N - 1 && break
        end
    end

    sort!(mst_edges, by=e -> -e[3])
    cuts = min(n_regimes - 1, length(mst_edges))

    parent .= 1:N
    rank .= 0
    kept_edges = @view mst_edges[(cuts + 1):end]
    for (i, j, _) in kept_edges
        union!(i, j)
    end

    roots = [find_root(i) for i in 1:N]
    unique_roots = unique(roots)
    label_map = Dict(r => k for (k, r) in enumerate(unique_roots))
    labels = [label_map[r] for r in roots]

    return ExogenousRegimes(labels)
end

# ═══════════════════════════════════════════════════════════════════════
# Spatial Regime Model
#   y = Xβ_r + ε  or  y = ρWy + Xβ_r + ε  (SAR with regime-specific β)
# ═══════════════════════════════════════════════════════════════════════

struct SpatialRegimeModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    sigma2::T
    loglik::T
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
    regime_labels::Vector{Int}
    n_regimes::Int
    has_spatial_lag::Bool
end

"""
    fit_spatial_regime(W, X, y, regimes; spatial_lag=true, varnames=nothing, method=ChebyshevApprox())

Fit a spatial regime model where coefficients vary across regimes:
  y = ρWy + X_r β_r + ε   (spatial_lag=true)
  y = X_r β_r + ε          (spatial_lag=false)

Each regime gets its own intercept and slope coefficients.
Anselin (1988, 1990).
"""
function fit_spatial_regime(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    regimes::ExogenousRegimes;
    spatial_lag::Bool=true,
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    labels = regimes.labels
    n_reg = maximum(labels)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    P_per_regime = K + 1
    P_total = n_reg * P_per_regime
    X_regime = zeros(T, N, P_total)
    cnames = String[]
    for r in 1:n_reg
        mask = labels .== r
        col_start = (r - 1) * P_per_regime + 1
        X_regime[mask, col_start] .= one(T)
        for k in 1:K
            X_regime[mask, col_start + k] .= X[mask, k]
        end
        push!(cnames, "R$(r)_(Intercept)")
        for v in vnames; push!(cnames, "R$(r)_$v"); end
    end

    if !spatial_lag
        β = X_regime \ Vector{T}(y)
        res = y .- X_regime * β
        σ2 = dot(res, res) / N
        ll = -T(0.5) * N * (log(T(2π)) + log(σ2) + one(T))
        V = σ2 .* inv(Symmetric(X_regime' * X_regime))
        return SpatialRegimeModel{T, M}(β, V, zero(T), σ2, ll, W, X_regime,
            Vector{T}(y), res, cnames, labels, n_reg, false)
    end

    push!(cnames, "ρ")
    Wy = W * y

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = ρ -> chebyshev_logdet(ρ, τ)
    else
        I_sp = sparse(one(T) * I, N, N)
        logdet_fn = ρ -> first(logabsdet(lu(I_sp - ρ * W)))
    end

    const_term = -T(0.5) * N * (log(T(2π)) + one(T))
    function concentrated_ll(ρ)
        y_f = y .- ρ .* Wy
        β_f = X_regime \ y_f
        res_f = y_f .- X_regime * β_f
        σ2_f = dot(res_f, res_f) / N
        return const_term + logdet_fn(ρ) - T(0.5) * N * log(σ2_f)
    end

    ρ_hat, max_ll = brent_maximize(concentrated_ll, T(-0.995), T(0.995))
    y_f = y .- ρ_hat .* Wy
    β_hat = X_regime \ y_f
    res = y_f .- X_regime * β_hat
    σ2_hat = dot(res, res) / N

    V_β = σ2_hat .* inv(Symmetric(X_regime' * X_regime))
    n_total = P_total + 1
    V_full = zeros(T, n_total, n_total)
    V_full[1:P_total, 1:P_total] .= V_β
    V_full[n_total, n_total] = T(1) / N
    full_coef = vcat(β_hat, ρ_hat)

    return SpatialRegimeModel{T, M}(full_coef, V_full, ρ_hat, σ2_hat, max_ll, W,
        X_regime, Vector{T}(y), res, cnames, labels, n_reg, true)
end

StatsAPI.coef(m::SpatialRegimeModel) = m.coef
StatsAPI.vcov(m::SpatialRegimeModel) = m.vcov
StatsAPI.stderror(m::SpatialRegimeModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::SpatialRegimeModel) = length(m.y)
StatsAPI.residuals(m::SpatialRegimeModel) = m.residuals
StatsAPI.coefnames(m::SpatialRegimeModel) = m.coefnames
StatsAPI.loglikelihood(m::SpatialRegimeModel) = m.loglik
StatsAPI.dof_residual(m::SpatialRegimeModel) = length(m.y) - length(m.coef)

function Base.show(io::IO, m::SpatialRegimeModel{T}) where {T}
    println(io, "Spatial Regime Model ($(m.n_regimes) regimes$(m.has_spatial_lag ? ", SAR" : ", OLS"))")
    println(io, "═"^50)
    @printf(io, "  N = %d,  σ² = %.4f,  Log-Lik = %.2f\n", nobs(m), m.sigma2, m.loglik)
    m.has_spatial_lag && @printf(io, "  ρ = %.4f\n", m.rho)
    for r in 1:m.n_regimes
        @printf(io, "  Regime %d: %d obs\n", r, count(==(r), m.regime_labels))
    end
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
end

# ═══════════════════════════════════════════════════════════════════════
# Chow-type Regime Stability Test
# ═══════════════════════════════════════════════════════════════════════

struct ChowRegimeTest{T<:AbstractFloat}
    stat::T
    df::Int
    p_value::T
    ll_pooled::T
    ll_regime::T
end

"""
    regime_chow_test(pooled_model, regime_model)

Likelihood ratio test for structural stability across spatial regimes.
H₀: coefficients are equal across all regimes.
"""
function regime_chow_test(
    pooled::StatsAPI.RegressionModel,
    regime::SpatialRegimeModel{T}
) where {T}
    ll_p = loglikelihood(pooled)
    ll_r = loglikelihood(regime)
    df = length(coef(regime)) - length(coef(pooled))
    stat = T(2) * (ll_r - ll_p)
    stat = max(stat, zero(T))
    p = _chi2_sf(stat, df)
    return ChowRegimeTest{T}(stat, df, p, ll_p, ll_r)
end

function _chi2_sf(x::T, k::Int) where {T<:AbstractFloat}
    k <= 0 && return one(T)
    x <= zero(T) && return one(T)
    a = T(k) / T(2)
    z = x / T(2)
    s = zero(T); term = exp(-z) * z^a / _gamma(a + one(T))
    for n in 0:200
        s += term
        term *= z / (a + T(n + 1))
        abs(term) < T(1e-15) * abs(s) && break
    end
    return max(zero(T), one(T) - s)
end

function _gamma(z::T) where {T<:AbstractFloat}
    coefs = T.([76.18009172947146, -86.50532032941677, 24.01409824083091,
                 -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5])
    x = z - one(T); t = x + T(5.5); t -= (x + T(0.5)) * log(t)
    ser = T(1.000000000190015)
    for (j, c) in enumerate(coefs); ser += c / (x + T(j)); end
    return exp(-t + log(T(2.5066282746310005) * ser / z))
end

function Base.show(io::IO, t::ChowRegimeTest)
    println(io, "Spatial Regime Chow Test (LR)")
    println(io, "─"^35)
    @printf(io, "  LR stat  = %.4f  (df = %d)\n", t.stat, t.df)
    @printf(io, "  p-value  = %.6f\n", t.p_value)
    @printf(io, "  LL(pooled) = %.2f,  LL(regime) = %.2f\n", t.ll_pooled, t.ll_regime)
end

end # module
