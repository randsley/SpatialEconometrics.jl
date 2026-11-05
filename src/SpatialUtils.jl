module SpatialUtils

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf

export LogDetMethod, ExactLU, ChebyshevApprox,
       SpatialVariant, SAR, SDM, SARAR,
       FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
       precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, spatial_trace_powers, g_traces,
       brent_maximize,
       panel_spatial_lag!, panel_spatial_lag,
       within_transform!, within_transform,
       effective_dims, has_indiv_fe, has_time_fe,
       rank_deficient_columns, check_design_rank,
       _normcdf, _print_coef_table,
       _chi2_sf, _gamma_series, _gamma_cf, _loggamma

# ─── Log-determinant methods ──────────────────────────────────────────

"""
    LogDetMethod

How `log|I - ρW|` is evaluated inside a spatial likelihood.

Every concentrated-likelihood estimator in this package needs that
log-determinant at each trial value of the spatial parameter, so the choice
is the main speed/accuracy lever available. Subtypes: [`ExactLU`](@ref) and
[`ChebyshevApprox`](@ref).
"""
abstract type LogDetMethod end

"""
    ExactLU()

Evaluate `log|I - ρW|` exactly, by sparse LU factorization at each trial `ρ`.

Accurate to machine precision and the right default for moderate `N`. Cost
grows with the fill-in of the factorization, so for large or densely
connected `W` — and for samplers that need thousands of evaluations —
[`ChebyshevApprox`](@ref) is usually the better trade.

See also [`LogDetMethod`](@ref).
"""
struct ExactLU <: LogDetMethod end

"""
    ChebyshevApprox(; order = 6, n_probes = 0, rng = Random.default_rng())

Approximate `log|I - ρW|` by a Chebyshev expansion in `ρ`.

The traces the expansion needs depend only on `W`, so they are computed once
by `precompute_chebyshev_traces` and every later evaluation is a
polynomial in `ρ` — which is what makes repeated likelihood evaluation
affordable.

# Arguments
- `order`: degree of the expansion, at least 2. Accuracy improves with
  order; 6 is adequate for point estimation and 12 is accurate to several
  decimal places across `ρ ∈ (-1, 1)`, degrading mildly as `|ρ| → 1`.
- `n_probes`: number of Hutchinson random probes used to estimate the
  traces. `0` computes them deterministically from sparse matrix powers,
  which is exact and preferred unless `W` is too large for that.
- `rng`: random source, used only when `n_probes > 0`. **Seed it if the
  estimate must be reproducible**: stochastic traces move `ρ̂` by about
  `1e-4`, unbiased but not repeatable. A likelihood-ratio test must
  also reuse one set of traces across the models it compares, or the
  statistic absorbs the trace noise.

# Choosing between exact and stochastic traces
This is **not an accuracy decision**. Measured at `N = 3600`, order 12:
the exact recursion fills in on `knn` weights and at `k = 24` costs 58
LU-equivalents against an optimiser budget of 42, a net loss, while
Hutchinson costs 1 to 10. On a lattice the recursion stays cheap and is
exact. Decide on `W`'s fill, not on the error.

# Accuracy, and what it is accuracy *in*
Order 6 is adequate for point estimation because the error in `ρ̂` runs
three to four orders of magnitude below the error in the
log-determinant: at `N = 2500, ρ = 0.9`, a log-determinant error of
`7.8e-01` cost `3.4e-04` in `ρ̂`. The profile maximiser depends on the
shape of the approximation near the peak, not on its level.

# Reference
Pace, R. K. & LeSage, J. P. (2004). Chebyshev approximation of
log-determinants of spatial weight matrices. *Computational Statistics
& Data Analysis*, 45(2), 179–196.

The method assumes `W` is symmetric or **similar** to a symmetric
matrix. By Ord (1975, p. 125) a row-standardised *symmetric* adjacency
qualifies; a row-standardised `knn` matrix does not, and its eigenvalues
are genuinely complex. Measured, that costs nothing — the error stays
the same order of magnitude as on a lattice — but the paper's error
bound does not apply there. See `docs/AUDIT.md`.

See also [`ExactLU`](@ref), `chebyshev_logdet`.
"""
struct ChebyshevApprox{R<:AbstractRNG} <: LogDetMethod
    order::Int
    n_probes::Int
    rng::R
    function ChebyshevApprox(; order::Int=6, n_probes::Int=0, rng::AbstractRNG=Random.default_rng())
        @assert order >= 2 "Chebyshev order must be >= 2"
        return new{typeof(rng)}(order, n_probes, rng)
    end
end

# ─── Model variants ───────────────────────────────────────────────────
#
# One hierarchy shared by every estimation module. These tags previously
# existed as a separate, unrelated copy inside each of
# SpatialEconometricsPrototype, SpatialPanelEconometrics,
# DynamicSpatialPanelEconometrics and SpatialGS2SLSPanel, which is why
# calling code had to write `CrossSection.SAR()` to say which `SAR` it
# meant. Defining them once lets `SAR` and `SDM` be exported directly.

"""
    SpatialVariant

Which spatial structure a model places on the data.

One tag type shared by every estimation module, so the same `SAR()` can be
passed to `fit_spatial`, `fit_spatial_panel` and
`fit_dynamic_spatial_panel`. Subtypes: [`SAR`](@ref), [`SDM`](@ref),
[`SARAR`](@ref).
"""
abstract type SpatialVariant end

"""
    SAR()

Spatial autoregressive model — dependence in the dependent variable.

```math
y = ρWy + Xβ + ε
```

Because `y` appears on both sides, a change in one unit's `x` propagates
through `(I - ρW)^{-1}` to every other unit, so coefficients are not
marginal effects; use `spatial_effects` to decompose them into
direct, indirect and total.

See also [`SDM`](@ref), [`SpatialVariant`](@ref).
"""
struct SAR   <: SpatialVariant end

"""
    SDM()

Spatial Durbin model — dependence in `y` and in the covariates.

```math
y = ρWy + Xβ + WXθ + ε
```

The design is augmented with the spatial lags of `X`, so the fitted
coefficient vector carries `W_`-prefixed entries for the `θ` terms. It
nests [`SAR`](@ref) at `θ = 0`, which `common_factor_test` tests.

See also [`SpatialVariant`](@ref).
"""
struct SDM   <: SpatialVariant end

"""
    SARAR()

Spatial dependence in both the dependent variable and the errors.

```math
y = ρWy + Xβ + u,  u = λWu + ε
```

Used by `fit_gs2sls_panel`, which estimates it by generalized
spatial two-stage least squares rather than maximum likelihood, so
endogenous regressors can be instrumented at the same time.

See also [`SpatialVariant`](@ref).
"""
struct SARAR <: SpatialVariant end

# ─── Fixed effects ────────────────────────────────────────────────────

"""
    FixedEffectsSpec

Which fixed effects a panel estimator sweeps out before estimating.

The transform is applied by `within_transform`, and the degrees of
freedom it consumes are reported by `effective_dims`. Subtypes:
[`NoFE`](@ref), [`IndivFE`](@ref), [`TimeFE`](@ref), [`TwoWayFE`](@ref).
"""
abstract type FixedEffectsSpec end

"""
    NoFE()

Pooled estimation: no fixed effects removed.

Consistent only if the unobserved heterogeneity is uncorrelated with the
regressors. `spatial_hausman_test` is the usual check.

See also [`FixedEffectsSpec`](@ref).
"""
struct NoFE     <: FixedEffectsSpec end

"""
    IndivFE()

Individual (unit) fixed effects, removed by subtracting unit means.

Absorbs any time-invariant unit characteristic, at the cost of not
identifying time-invariant regressors. Costs `T - 1` effective periods.

See also [`FixedEffectsSpec`](@ref), [`TwoWayFE`](@ref).
"""
struct IndivFE  <: FixedEffectsSpec end

"""
    TimeFE()

Time fixed effects, removed by subtracting period means.

Absorbs common shocks hitting every unit in a period. Costs `N - 1`
effective units.

See also [`FixedEffectsSpec`](@ref).
"""
struct TimeFE   <: FixedEffectsSpec end

"""
    TwoWayFE()

Both individual and time fixed effects.

The usual specification for spatial panels, since it absorbs unit
heterogeneity and common time shocks together — the latter matters because a
common shock can otherwise masquerade as spatial dependence. Costs `N - 1`
units and `T - 1` periods.

See also [`FixedEffectsSpec`](@ref), [`IndivFE`](@ref).
"""
struct TwoWayFE <: FixedEffectsSpec end

effective_dims(N::Int, T_per::Int, ::NoFE)     = (N, T_per)
effective_dims(N::Int, T_per::Int, ::IndivFE)  = (N, T_per - 1)
effective_dims(N::Int, T_per::Int, ::TimeFE)   = (N - 1, T_per)
effective_dims(N::Int, T_per::Int, ::TwoWayFE) = (N - 1, T_per - 1)

has_indiv_fe(::FixedEffectsSpec) = false
has_indiv_fe(::IndivFE)          = true
has_indiv_fe(::TwoWayFE)         = true

has_time_fe(::FixedEffectsSpec)  = false
has_time_fe(::TimeFE)            = true
has_time_fe(::TwoWayFE)          = true

# ─── Chebyshev log-determinant ────────────────────────────────────────

"""
    _spmm_block!(Pout, Wt, Pin, b)

`Pout = W * Pin` where both are `b x N`, gathering from `Wt`, the same way
[`_spmm_rows_dot!`](@ref) does and for the same reason: the `b` entries of
one node are adjacent, and each output column is written by one thread.
"""
function _spmm_block!(Pout::Matrix{T}, Wt::SparseMatrixCSC{T}, Pin::Matrix{T},
                      b::Int) where {T<:AbstractFloat}
    rv = rowvals(Wt)
    nz = nonzeros(Wt)
    Threads.@threads for i in 1:size(Wt, 2)
        base = (i - 1) * b
        @inbounds for q in 1:b
            Pout[base + q] = zero(T)
        end
        @inbounds for k in nzrange(Wt, i)
            c = rv[k]
            v = nz[k]
            cb = (c - 1) * b
            @simd for q in 1:b
                Pout[base + q] += v * Pin[cb + q]
            end
        end
    end
    return Pout
end

"""
    _cheb_traces_blocked!(τ, W, T_prev, T_curr, k0, q, bsize)

Finish the exact Chebyshev trace recursion from order `k0` to `q` without
ever holding a whole `T_k(W)`.

Only the **trace** of `T_k(W)` is wanted, and a trace is a sum over
columns, so the recursion can run on a block of `bsize` columns at a time
and accumulate. Peak memory is `3 · N · bsize` doubles instead of the
fill-in of `T_k(W)`, which has no bound below `N²`.

`T_prev` and `T_curr` are orders `k0 - 2` and `k0 - 1` as the sparse
recursion left them, so nothing already computed is thrown away: each
block starts by densifying just its own columns of those two.

This is slower than the sparse recursion whenever the fill-in is still
modest — measured 4× to 25× slower on a rook lattice — and faster once it
saturates, 3.7× at `N = 900, q = 12`. It is a fallback for the memory
ceiling, not a replacement.
"""
function _cheb_traces_blocked!(τ::Vector{T}, W::SparseMatrixCSC{T},
                               T_prev::SparseMatrixCSC{T},
                               T_curr::SparseMatrixCSC{T},
                               k0::Int, q::Int, bsize::Int) where {T<:AbstractFloat}
    N = size(W, 1)
    Wt = sparse(transpose(W))
    Pprev = Matrix{T}(undef, bsize, N)
    Pcurr = Matrix{T}(undef, bsize, N)
    Pnext = Matrix{T}(undef, bsize, N)
    for start in 1:bsize:N
        cols = start:min(start + bsize - 1, N)
        _densify_columns!(Pprev, T_prev, cols)
        _densify_columns!(Pcurr, T_curr, cols)
        for k in k0:q
            _spmm_block!(Pnext, Wt, Pcurr, bsize)
            @inbounds @simd for t in eachindex(Pnext)
                Pnext[t] = T(2) * Pnext[t] - Pprev[t]
            end
            s = zero(T)
            @inbounds for (lj, j) in enumerate(cols)
                s += Pnext[lj, j]
            end
            τ[k + 1] += s
            Pprev, Pcurr, Pnext = Pcurr, Pnext, Pprev
        end
    end
    return τ
end

"""
    _densify_columns!(P, A, cols)

Write `A[:, cols]` transposed into the `b x N` buffer `P`, zeroing the rest.
Row `lj` of `P` becomes column `cols[lj]` of `A`.
"""
function _densify_columns!(P::Matrix{T}, A::SparseMatrixCSC{T},
                           cols::AbstractRange{Int}) where {T<:AbstractFloat}
    fill!(P, zero(T))
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for (lj, j) in enumerate(cols)
        for t in nzrange(A, j)
            P[lj, rv[t]] = nz[t]
        end
    end
    return P
end

"""
    precompute_chebyshev_traces(W, method; max_fill_bytes)

The Chebyshev trace moments `τ[k + 1] = tr(T_k(W))` that
`chebyshev_logdet` needs, computed once per `W`.

With `method.n_probes == 0` these are exact, from the three-term
recursion `T_{k+1} = 2 W T_k - T_{k-1}` carried on sparse matrices. That
is the fast path and it is the default, but `T_k(W)` fills in as `k`
grows, and the fill per row saturates at `N`, so peak memory is `Θ(N²)`
on a dense enough `W`. How soon that bites depends entirely on the
weights: a rook lattice at `N = 8100` and order 12 peaks at only 18.9
MiB, while knn weights with `k = 24` are already at 90 MiB by
`N = 3600`.

`max_fill_bytes` is the ceiling on that, 128 MiB by default. When the
recursion is about to cross it the remaining orders finish one column
block at a time in `_cheb_traces_blocked!`, seeded from the two sparse
matrices already computed, so nothing is discarded; a trace is a sum
over columns, which is what makes that possible without ever forming
`T_k(W)`. The traces agree to roundoff.

The ceiling is not a better algorithm, only a bounded one: on a rook
lattice the blocked path is 4× to 25× slower, because the sparse
recursion touches only the fill that exists while the blocked path pays
`N · b` per level whatever the density. It is faster where it is needed
— knn `k = 24, N = 3600`, order 12 goes from 1118 ms to 576 ms. At the
default every case benchmarked stays on the fast path and is
bit-identical, order 6, which is what `fit_spatial` uses, included.

With `method.n_probes > 0` the traces are Hutchinson estimates instead
and the question does not arise: that path has never held more than the
probe block.
"""
function precompute_chebyshev_traces(W::AbstractSparseMatrix{T}, method::ChebyshevApprox;
                                     max_fill_bytes::Int=128 * 1024 * 1024) where {T<:AbstractFloat}
    N = size(W, 1)
    q = method.order
    τ = zeros(T, q + 1)
    τ[1] = T(N)
    τ[2] = tr(W)

    if method.n_probes == 0
        Ws = SparseMatrixCSC{T,Int}(W)
        T_prev = sparse(one(T) * I, N, N)
        T_curr = copy(Ws)
        # A stored entry costs a value and a row index. The check is on
        # `T_curr` rather than on the `T_next` about to be built, because by
        # the time `T_next` exists its memory is already committed; one
        # level of the recursion multiplies the fill by roughly the mean
        # degree, so leaving a factor of four of headroom keeps the peak
        # inside the budget rather than one level past it.
        per_entry = sizeof(T) + sizeof(Int)
        for k in 2:q
            if 4 * per_entry * nnz(T_curr) > max_fill_bytes
                bsize = _cheb_block_size(N, max_fill_bytes, T)
                return _cheb_traces_blocked!(τ, Ws, T_prev, T_curr, k, q, bsize)
            end
            T_next = 2 .* (Ws * T_curr) .- T_prev
            τ[k + 1] = tr(T_next)
            T_prev = T_curr
            T_curr = T_next
        end
    else
        S = method.n_probes
        V0 = Matrix{T}(undef, N, S)
        rand!(method.rng, V0, (-one(T), one(T)))
        V_prev = copy(V0)
        V_curr = W * V0
        V_next = similar(V0)
        for k in 2:q
            mul!(V_next, W, V_curr)
            @. V_next = 2 * V_next - V_prev
            τ[k + 1] = dot(V0, V_next) / S
            copyto!(V_prev, V_curr)
            copyto!(V_curr, V_next)
        end
    end
    return τ
end

"""
    _cheb_block_size(N, max_fill_bytes, T)

Columns per block: as many as three `N`-long dense buffers fit in the
budget, at least 16 and never more than `N`.
"""
function _cheb_block_size(N::Int, max_fill_bytes::Int, ::Type{T}) where {T}
    b = max_fill_bytes ÷ (3 * N * sizeof(T))
    return clamp(b, 16, N)
end

@inline function chebyshev_logdet(ρ::R, τ::Vector{T}) where {R<:Real, T<:AbstractFloat}
    q = length(τ) - 1
    N_nodes = q + 1
    logdet_val = zero(promote_type(R, T))
    for k in 0:q
        ck = zero(promote_type(R, T))
        for j in 1:N_nodes
            θ_j = T(π) * (j - T(0.5)) / N_nodes
            x_j = cos(θ_j)
            ck += log(one(R) - ρ * x_j) * cos(k * θ_j)
        end
        ck *= (T(2) / N_nodes)
        k == 0 && (ck *= T(0.5))
        logdet_val += ck * τ[k + 1]
    end
    return logdet_val
end

# ─── Neumann series traces ────────────────────────────────────────────

"""
    _rademacher_probes(T, N, p, rng)

A `p x N` Rademacher probe block: probe `q` at node `i` is entry `(q, i)`.

Drawn into an `N x p` matrix first and then transposed, which looks
wasteful and is deliberate. `rand!` fills column-major, so drawing
straight into `p x N` would pair the same stream of random numbers with
different `(node, probe)` positions, and a caller who passed a seed to
get one probe set would silently get another. This way the probe set is
exactly the one the column-major draw gives, laid out so that the
probes of a single node are contiguous.
"""
function _rademacher_probes(::Type{T}, N::Int, p::Int, rng::AbstractRNG) where {T<:AbstractFloat}
    V = Matrix{T}(undef, N, p)
    rand!(rng, V, (-one(T), one(T)))
    return permutedims(V)      # V is unreachable from here, so collectable
end

"""
    _spmm_rows_dot!(Vout, Wt, Vin, V0, rowdot)

Write `Vout = W * Vin` in the probes-contiguous layout and return
`dot(V0, Vout)`, in one pass over the nonzeros of `W`.

`Wt` is `sparse(transpose(W))`, so column `i` of `Wt` holds row `i` of
`W`. That turns the product into a gather: row `i` of the output reads
other rows of the input and is written by nobody else. The alternative,
a scatter from `W` in CSC form, has threads racing to accumulate into
the same output row.

Two things make this faster than `mul!` on the `N x p` layout. The inner
loop runs over the `p` probes of one node, which are adjacent in memory,
so it vectorises instead of striding by `N`. And the inner product that
yields the trace estimate is taken while row `i` is still in cache,
rather than in a second pass that re-reads the whole block.

Determinism does not depend on the thread count: each thread writes only
`rowdot[i]` for rows it owns, and the sum over `rowdot` is a serial pass
in index order afterwards. Against `mul!` plus `dot` the accumulation
order within a row is the same -- both add the contributions of column
`j` in increasing `j` -- but the order of the outer sum is by node
rather than by probe, so the trace estimates differ in the last few
bits. That is reordering, not a different estimator: the probes are
identical and the shift is ~1e-14 relative against a Hutchinson
standard error of order `1 / sqrt(p)`.
"""
function _spmm_rows_dot!(
    Vout::Matrix{T},
    Wt::SparseMatrixCSC{T},
    Vin::Matrix{T},
    V0::Matrix{T},
    rowdot::Vector{T}
) where {T<:AbstractFloat}
    p = size(Vout, 1)
    rv = rowvals(Wt)
    nz = nonzeros(Wt)
    Threads.@threads for i in 1:size(Wt, 2)
        base = (i - 1) * p
        @inbounds for q in 1:p
            Vout[base + q] = zero(T)
        end
        @inbounds for k in nzrange(Wt, i)
            c = rv[k]
            v = nz[k]
            cb = (c - 1) * p
            @simd for q in 1:p
                Vout[base + q] += v * Vin[cb + q]
            end
        end
        s = zero(T)
        @inbounds @simd for q in 1:p
            s += V0[base + q] * Vout[base + q]
        end
        @inbounds rowdot[i] = s
    end
    acc = zero(T)
    @inbounds for i in eachindex(rowdot)
        acc += rowdot[i]
    end
    return acc
end

"""
    precompute_neumann_traces(W; max_power, n_probes, rng)

Hutchinson estimates of `tr(Wᵐ) / N` for `m = 0, ..., max_power`.

`tr(Wᵐ)` is estimated as the mean of `vᵀWᵐv` over `n_probes` Rademacher
vectors `v`, which costs one sparse product per power because `Wᵐv` is
built by repeated multiplication. `m = 0` and `m = 1` are not estimated:
they are `1` and `tr(W) / N` exactly.

The estimates are stochastic, so pass `rng` if the result has to be
reproducible. It does not depend on the number of threads.
"""
function precompute_neumann_traces(
    W::AbstractSparseMatrix{T};
    max_power::Int=60,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    N = size(W, 1)
    μ = zeros(T, max_power + 1)
    μ[1] = one(T)
    μ[2] = tr(W) / N

    V0 = _rademacher_probes(T, N, n_probes, rng)
    Wt = sparse(transpose(W))
    V_curr = similar(V0)
    V_next = similar(V0)
    rowdot = Vector{T}(undef, N)

    _spmm_rows_dot!(V_curr, Wt, V0, V0, rowdot)   # m = 1; μ[2] is exact
    for m in 2:max_power
        s = _spmm_rows_dot!(V_next, Wt, V_curr, V0, rowdot)
        μ[m + 1] = s / (N * n_probes)
        V_curr, V_next = V_next, V_curr           # swap, do not copy
    end
    return μ
end

"""
    spatial_trace_powers(W; max_power, max_exact_n, n_probes, rng)

Traces of powers of `W`, normalised: entry `m + 1` is `tr(Wᵐ) / N`.

Computed exactly from the eigenvalues of `W` when `N <= max_exact_n`,
since `tr(Wᵐ) = Σᵢ λᵢᵐ` and one decomposition serves every power. Above
that size it falls back to `precompute_neumann_traces`, whose
Hutchinson estimates are stochastic — so pass `rng` if the result has to
be reproducible.

These traces are not an internal detail: `spatial_effects` reads them to
split a marginal effect into its direct and indirect parts. Estimating
them by unseeded random probes made the reported impacts, and the
standard errors built from the same traces, differ from one run to the
next on identical data.

!!! note "The exact branch is the expensive one, deliberately"
    `max_exact_n = 1000` is not a performance default. The exact branch
    is a dense non-symmetric eigendecomposition: at `N = 900` it takes
    about 0.3 s against 6 ms for 250 probes, so the default costs **51×**
    at its own boundary, and it is `O(N³)` in time and `O(N²)` in memory
    — 1.7 GB at `N = 14400` — so above the threshold it is not merely
    slower but infeasible.

    It is the default anyway because what it buys is that two runs on the
    same data report the same impacts. The probe branch has a Hutchinson
    standard error of order `1 / sqrt(n_probes)`, about 6% at the default
    250, and `spatial_effects` and its standard errors inherit it. Pass
    `max_exact_n = 0` to take the speed instead; pass a larger
    `n_probes` to buy the accuracy back at `sqrt` cost.

    There is no third option here. The eigendecomposition could be made
    3–4× faster and real-arithmetic by symmetrising `W` to
    `D^(1/2) W D^(-1/2)`, which is similar to `W` and symmetric when `W`
    is a row-standardised symmetric adjacency — but that needs the
    unstandardised adjacency and its degrees, and this function only ever
    receives the standardised `W`. Recovering the degrees from `W` alone
    means solving `dᵢ W[i,j] = dⱼ W[j,i]` across the graph, which is
    possible and was judged not worth the failure modes.
"""
function spatial_trace_powers(
    W::AbstractSparseMatrix{T};
    max_power::Int=60,
    max_exact_n::Int=1000,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}
    N = size(W, 1)
    if N > max_exact_n
        return precompute_neumann_traces(W; max_power=max_power,
                                         n_probes=n_probes, rng=rng)
    end
    λ = eigvals(Matrix{T}(W))          # generally complex: W need not be symmetric
    μ = zeros(T, max_power + 1)
    pow = ones(complex(T), N)
    μ[1] = one(T)
    @inbounds for m in 1:max_power
        pow .*= λ
        μ[m + 1] = T(real(sum(pow)) / N)
    end
    return μ
end

"""
    g_traces(W, ρ; project_constant, max_exact_n, n_probes, rng, μ)

The three traces of `G = W (I - ρW)⁻¹` that a spatial information matrix
needs: `(tr(G), tr(G²), tr(GᵀG))`.

When `N <= max_exact_n` these are formed from `G` itself — one sparse
factorization and `N` solves — and are exact. Above that size `tr(G)` and
`tr(G²)` come from the power traces `μ` (supply them, or they are
recomputed) and `tr(GᵀG)` from Hutchinson probes drawn against `rng`.
Only `tr(GᵀG)` genuinely needs probing: it is not a function of the
eigenvalues of `W`, being invariant under neither similarity nor
transposition.

`project_constant` removes the constant eigenvector before taking the
traces, for the panel estimators that sweep out time effects. For a
row-standardised `W` the constant direction is an eigenvector of `G` with
eigenvalue `1 / (1 - ρ)`; projecting `G` on both sides handles it without
assuming that, which the closed-form subtraction it replaces did assume.
"""
function g_traces(
    W::AbstractSparseMatrix{T},
    ρ::T;
    project_constant::Bool=false,
    max_exact_n::Int=1000,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng(),
    μ::Union{Nothing,Vector{T}}=nothing
) where {T<:AbstractFloat}
    N = size(W, 1)
    A_lu = lu(sparse(one(T) * I, N, N) - ρ * W)

    if N <= max_exact_n
        G = A_lu \ Matrix{T}(W)        # A and W commute, so this is W(I-ρW)⁻¹
        if project_constant
            G .-= mean(G; dims=1)
            G .-= mean(G; dims=2)
        end
        tr_G = zero(T); tr_G2 = zero(T); tr_GtG = zero(T)
        @inbounds for j in 1:N
            tr_G += G[j, j]
            for i in 1:N
                tr_G2 += G[i, j] * G[j, i]
                tr_GtG += G[i, j]^2
            end
        end
        return (tr_G, tr_G2, tr_GtG)
    end

    μv = isnothing(μ) ? precompute_neumann_traces(W; max_power=60,
                                                  n_probes=n_probes, rng=rng) : μ
    tr_G = zero(T)
    @inbounds for m in 1:(length(μv) - 1)
        tr_G += (ρ^(m - 1)) * (N * μv[m + 1])
    end
    tr_G2 = zero(T)
    @inbounds for m in 1:(length(μv) - 2)
        tr_G2 += m * (ρ^(m - 1)) * (N * μv[m + 2])
    end
    if project_constant
        eig_1 = one(T) / (one(T) - ρ)
        tr_G -= eig_1
        tr_G2 -= eig_1^2
    end

    V = Matrix{T}(undef, N, n_probes)
    rand!(rng, V, (-one(T), one(T)))
    project_constant && (V .-= mean(V; dims=1))
    GV = A_lu \ (W * V)
    project_constant && (GV .-= mean(GV; dims=1))
    tr_GtG = dot(GV, GV) / n_probes

    return (tr_G, tr_G2, tr_GtG)
end

# ─── Brent's method maximizer ─────────────────────────────────────────

function brent_maximize(f::F, a::T, b::T; tol::T=T(1e-7), maxiter::Int=200) where {F, T<:AbstractFloat}
    cg = T(0.3819660112501051)
    x = w = v = a + cg * (b - a)
    fx = fw = fv = -f(x)
    d = e = zero(T)

    for _ in 1:maxiter
        xm = T(0.5) * (a + b)
        tol1 = tol * abs(x) + T(1e-10)
        tol2 = T(2) * tol1
        abs(x - xm) <= (tol2 - T(0.5) * (b - a)) && return x, -fx

        p = q = r = zero(T)
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = T(2) * (q - r)
            q > zero(T) ? (p = -p) : (q = -q)
            r = e; e = d
        end

        if abs(p) < abs(T(0.5) * q * r) && p > q * (a - x) && p < q * (b - x)
            d = p / q
            u = x + d
            if (u - a) < tol2 || (b - u) < tol2
                d = x < xm ? tol1 : -tol1
            end
        else
            e = (x < xm ? b : a) - x
            d = cg * e
        end

        u = abs(d) >= tol1 ? x + d : x + (d > zero(T) ? tol1 : -tol1)
        fu = -f(u)

        if fu <= fx
            u >= x ? (a = x) : (b = x)
            v, fv = w, fw; w, fw = x, fx; x, fx = u, fu
        else
            u < x ? (a = u) : (b = u)
            if fu <= fw || w == x
                v, fv = w, fw; w, fw = u, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u, fu
            end
        end
    end
    return x, -fx
end

# ─── Panel spatial lag ────────────────────────────────────────────────

function panel_spatial_lag!(
    WY::AbstractVector{T},
    W::AbstractSparseMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_per::Int
) where {T<:AbstractFloat}
    Y_mat = reshape(y, N, T_per)
    WY_mat = reshape(WY, N, T_per)
    mul!(WY_mat, W, Y_mat)
    return WY
end

function panel_spatial_lag(W::AbstractSparseMatrix{T}, X::AbstractMatrix{T}, N::Int, T_per::Int) where {T<:AbstractFloat}
    WX = similar(X)
    for k in 1:size(X, 2)
        @views panel_spatial_lag!(WX[:, k], W, X[:, k], N, T_per)
    end
    return WX
end

# ─── Within-transformation (FE demeaning) ─────────────────────────────

within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::NoFE) where {T<:AbstractFloat} = z

function within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::IndivFE) where {T<:AbstractFloat}
    Z = reshape(z, N, T_per)
    @inbounds for i in 1:N
        row_sum = zero(T)
        for t in 1:T_per; row_sum += Z[i, t]; end
        row_mean = row_sum / T_per
        for t in 1:T_per; Z[i, t] -= row_mean; end
    end
    return z
end

function within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::TimeFE) where {T<:AbstractFloat}
    Z = reshape(z, N, T_per)
    @inbounds for t in 1:T_per
        col_sum = zero(T)
        for i in 1:N; col_sum += Z[i, t]; end
        col_mean = col_sum / N
        for i in 1:N; Z[i, t] -= col_mean; end
    end
    return z
end

function within_transform!(z::AbstractVector{T}, N::Int, T_per::Int, ::TwoWayFE) where {T<:AbstractFloat}
    within_transform!(z, N, T_per, IndivFE())
    within_transform!(z, N, T_per, TimeFE())
    return z
end

function within_transform(X::AbstractMatrix{T}, N::Int, T_per::Int, fe::FixedEffectsSpec) where {T<:AbstractFloat}
    X_tilde = copy(X)
    for k in 1:size(X, 2)
        @views within_transform!(X_tilde[:, k], N, T_per, fe)
    end
    return X_tilde
end

# ─── Numerical rank of a design matrix ───────────────────────────────

# Which columns of `X` a column-pivoted QR puts past the numerical rank.
#
# Returns an empty vector when `X` has full column rank, and otherwise the
# indices -- in the original column order -- of the columns that are linear
# combinations of the ones the pivoting kept. An exactly singular design is
# not a numerical hiccup to be ridged away: it means a coefficient is not
# identified, and the estimator should say so rather than hand back whatever
# the solver's last pivot produced. That failure mode is findings 16 and 27a.
#
# Column pivoting is what makes this a rank test rather than a guess. An
# unpivoted factorization can leave a tiny diagonal entry anywhere, and
# whether it comes out positive, negative or exactly zero depends on the
# LAPACK build; pivoting orders |R[j,j]| non-increasing, so comparing against
# the leading entry is a conditioning statement that holds across platforms.
# `tol` of `sqrt(eps)` marks a condition number of roughly 1/sqrt(eps) --
# about half the available digits gone -- which is the same standard
# `_chol_or_nothing` applies to an information matrix.
function rank_deficient_columns(X::AbstractMatrix{T};
                                tol::Real = sqrt(eps(T))) where {T<:AbstractFloat}
    K = size(X, 2)
    K == 0 && return Int[]
    F = qr(Matrix(X), ColumnNorm())
    d = abs.(diag(F.R))
    (isempty(d) || first(d) == zero(T)) && return collect(1:K)
    r = count(>(T(tol) * first(d)), d)
    return sort(F.p[(r + 1):end])
end

# Throw if `X` cannot identify its coefficients.
#
# `colnames` names the columns of the design actually being solved, which
# after an augmentation or a within transform is not what the caller passed;
# `caller` is the entry point to name in the message, and `context` the one
# sentence that says what to do about it, since the cure differs by
# estimator -- drop a column, drop an intercept, change the fixed effects.
function check_design_rank(X::AbstractMatrix{T},
                           colnames::AbstractVector{<:AbstractString},
                           caller::AbstractString;
                           context::AbstractString = "") where {T<:AbstractFloat}
    bad = rank_deficient_columns(X)
    isempty(bad) && return nothing
    named = join(("`" * String(colnames[j]) * "`" for j in bad), ", ")
    subject = length(bad) == 1 ? named * " is a linear combination of" :
                                 named * " are linear combinations of"
    msg = caller * ": " * subject * " the other columns of the design, so " *
          "the coefficients are not identified."
    isempty(context) || (msg *= " " * context)
    throw(ArgumentError(msg))
end

# ─── Normal CDF approximation (Abramowitz & Stegun 26.2.17) ──────────

function _normcdf(z::Real)
    x = Float64(z)
    if x < -8.0; return 0.0; end
    if x >  8.0; return 1.0; end
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * exp(-x * x / 2.0) * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return x > 0.0 ? 1.0 - p : p
end

# ─── Incomplete gamma / chi-squared survival ─────────────────────────

# Upper tail of the chi-squared distribution with `df` degrees of freedom,
# i.e. Q(df/2, x/2): the p-value of a statistic `x` referred to χ²(df).
#
# The lower-incomplete-gamma SERIES only converges for z < a + 1; past that its
# terms keep growing for roughly z iterations before they decay, so a truncated
# series silently returns a tiny partial sum and the survival function comes back
# near 1. That turns a decisive rejection into "fail to reject". Switch to the
# continued fraction for the upper tail once z >= a + 1 (Numerical Recipes).
#
# A degenerate df <= 0 returns 1.0: there is no restriction to reject.
function _chi2_sf(x::Real, df::Integer)
    df <= 0 && return 1.0
    a = df / 2.0
    z = Float64(x) / 2.0
    z <= 0.0 && return 1.0
    return z < a + 1.0 ? clamp(1.0 - _gamma_series(a, z), 0.0, 1.0) :
                         clamp(_gamma_cf(a, z), 0.0, 1.0)
end

# Regularized lower incomplete gamma P(a, z) by series. Valid for z < a + 1.
function _gamma_series(a::Real, z::Real)
    z <= 0.0 && return 0.0
    ap = Float64(a)
    s = 1.0 / a
    del = s
    for _ in 1:1000
        ap += 1.0
        del *= z / ap
        s += del
        abs(del) < abs(s) * 1e-14 && break
    end
    return s * exp(-z + a * log(z) - _loggamma(a))
end

# Regularized upper incomplete gamma Q(a, z) by continued fraction. Valid for
# z >= a + 1, where the series above fails.
function _gamma_cf(a::Real, z::Real)
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in 1:1000
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        abs(d) < tiny && (d = tiny)
        c = b + an / c
        abs(c) < tiny && (c = tiny)
        d = 1.0 / d
        del = d * c
        h *= del
        abs(del - 1.0) < 1e-14 && break
    end
    return exp(-z + a * log(z) - _loggamma(a)) * h
end

# log Gamma(z) by the Lanczos approximation (Numerical Recipes `gammln`, in the
# x = z - 1 shifted form). Kept in the log domain: Gamma(z) itself overflows
# Float64 past z ~ 171, which a chi-squared test with large df would reach.
function _loggamma(z::Real)
    z <= 0.0 && return Inf
    coefs = (76.18009172947146, -86.50532032941677, 24.01409824083091,
             -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
    x = Float64(z) - 1.0
    t = x + 5.5
    t -= (x + 0.5) * log(t)
    ser = 1.000000000190015
    for (j, c) in enumerate(coefs); ser += c / (x + j); end
    return -t + log(2.5066282746310005 * ser)
end

# ─── Shared show helper ──────────────────────────────────────────────

function _print_coef_table(io::IO, names::Vector{String}, coefs::Vector{T}, ses::Vector{T}) where {T<:AbstractFloat}
    @printf(io, "%-16s | %11s | %11s | %10s | %8s\n", "Parameter", "Estimate", "Std. Err", "z-stat", "P>|z|")
    println(io, "─────────────────┼─────────────┼─────────────┼────────────┼──────────────")
    for i in eachindex(coefs)
        se_i = ses[i]
        z = se_i > zero(T) ? coefs[i] / se_i : T(NaN)
        p = isnan(z) ? 1.0 : 2.0 * (1.0 - _normcdf(abs(Float64(z))))
        stars = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : p < 0.1 ? "." : ""
        @printf(io, "%-16s | %11.4f | %11.4f | %10.3f | %7.4f %s\n", names[i], coefs[i], se_i, z, p, stars)
    end
end

end # module
