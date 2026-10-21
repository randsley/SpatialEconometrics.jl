module SpatialRegimeModels

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames,
       loglikelihood, dof_residual, fitted, predict

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       brent_maximize, _normcdf, _print_coef_table, _chi2_sf

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
    skater_regimes(W, X, n_regimes; min_size=1, rng=Random.default_rng())

Spatially constrained clustering via SKATER minimum spanning tree partitioning.
Returns an `ExogenousRegimes` assignment with `n_regimes` contiguous groups, each
guaranteed to be a connected region of `W`.

A minimum spanning tree is built over the contiguity graph with edge weights given
by Euclidean distance between the standardised rows of `X`. The tree is then split
`n_regimes - 1` times, each time choosing the edge whose removal most reduces the
within-group sum of squared deviations.

`min_size` is the smallest admissible region; cuts that would leave a smaller
component are skipped. Raise it when the regions feed a regime regression, which
needs enough units per region to identify its own coefficients.

Assunção et al. (2006) "Efficient regionalization techniques for socio‐economic
geographical units using minimum spanning trees."
"""
function skater_regimes(
    W::AbstractSparseMatrix{T},
    X::AbstractMatrix{T},
    n_regimes::Int;
    min_size::Int=1,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    N = size(W, 1)
    @assert n_regimes >= 2 && n_regimes <= N
    @assert min_size >= 1 "min_size must be at least 1"
    @assert n_regimes * min_size <= N "n_regimes * min_size exceeds the number of units"

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

    # Adjacency of the spanning tree, for locating the two sides of a cut.
    adj = [Int[] for _ in 1:N]
    for (i, j, _) in mst_edges
        push!(adj[i], j)
        push!(adj[j], i)
    end

    # Within-group sum of squared deviations, summed over the clustering variables.
    function ssd(nodes)
        n = length(nodes)
        n <= 1 && return zero(T)
        total = zero(T)
        for j in axes(X_std, 2)
            m = zero(T)
            for v in nodes; m += X_std[v, j]; end
            m /= T(n)
            for v in nodes; total += (X_std[v, j] - m)^2; end
        end
        return total
    end

    # The component containing `u` after edge (u, v) is removed. In a tree this
    # is exactly one side of the cut, so a traversal from `u` that refuses to
    # cross (u, v) finds it.
    function cut_side(u::Int, v::Int, inset::Set{Int})
        seen = Set{Int}((u,))
        stack = [u]
        while !isempty(stack)
            a = pop!(stack)
            for b in adj[a]
                ((a == u && b == v) || (a == v && b == u)) && continue
                (b in inset && !(b in seen)) || continue
                push!(seen, b)
                push!(stack, b)
            end
        end
        return seen
    end

    # Top-down SKATER: repeatedly split the component whose division yields the
    # largest reduction in within-group SSD, subject to `min_size`.
    #
    # Selecting the LONGEST edge instead (single-linkage style) is what produces
    # degenerate singleton regions: the longest tree edge is usually the one
    # attaching an outlying leaf, so cutting it peels off one unit and leaves the
    # downstream regime regression rank-deficient. Splitting off a single
    # observation barely reduces SSD, so this criterion avoids that by
    # construction.
    components = Set{Int}[Set(1:N)]

    for _ in 1:(n_regimes - 1)
        best_gain = typemin(T)
        best = nothing

        for (ci, comp) in enumerate(components)
            length(comp) < 2 * min_size && continue
            parent_ssd = ssd(comp)

            for (i, j, _) in mst_edges
                (i in comp && j in comp) || continue
                A = cut_side(i, j, comp)
                length(A) < min_size && continue
                length(comp) - length(A) < min_size && continue
                B = setdiff(comp, A)
                gain = parent_ssd - ssd(A) - ssd(B)
                if gain > best_gain
                    best_gain = gain
                    best = (ci, A, B)
                end
            end
        end

        best === nothing && break      # no admissible cut left
        ci, A, B = best
        components[ci] = A
        push!(components, B)
    end

    labels = zeros(Int, N)
    for (k, comp) in enumerate(components)
        for v in comp; labels[v] = k; end
    end

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

function Base.show(io::IO, t::ChowRegimeTest)
    println(io, "Spatial Regime Chow Test (LR)")
    println(io, "─"^35)
    @printf(io, "  LR stat  = %.4f  (df = %d)\n", t.stat, t.df)
    @printf(io, "  p-value  = %.6f\n", t.p_value)
    @printf(io, "  LL(pooled) = %.2f,  LL(regime) = %.2f\n", t.ll_pooled, t.ll_regime)
end

end # module
