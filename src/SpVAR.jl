"""
    SpVAR

Spatial vector autoregression — Di Giacinto (2010).

`fit_star` in `SpatioTemporalModels` traces **one** variable through
space and time. This module does the same for a **system** of `K`
interdependent variables observed on `N` locations, which forces a question
a univariate model never raises: within a single period, which variable
moves first?

The answer is an identifying assumption, not something the data settles, so
`ordering` is a required argument of [`fit_spvar`](@ref) with no default.

# What makes this tractable

Di Giacinto's two structural choices are what keep the model usable:

1. **Spatially structured coefficient blocks** (his eq. 7). Each `N×N` block
   of each `NK×NK` coefficient matrix is `Σₗ W⁽ˡ⁾Φₗ`, a combination of
   order-`l` weights matrices. Free parameters per lag fall from `(NK)²` to
   `NK²(s+1)` — **linear in `N`, not quadratic.**
2. **Block-triangular `C₀`** (his eq. 9), not strictly triangular. A
   strictly triangular `C₀` would impose a recursive causal order on the
   *locations* as well as the variables — a one-way causal chain through
   space, which is wrong for spatial problems where interaction is
   bilateral. Block triangularity keeps the ordering across variables and
   restores bilateral interaction in space.

Choice 2 has a consequence that does the computational work. Block
triangularity gives `|C₀| = Πᵣ |A⁽⁰⁾ᵣᵣ|`, so the concentrated
log-likelihood **separates into `K` unrelated terms**, one per equation
(his eqs. 22–23), each maximisable on its own. A SpVAR is `K` spatial
autoregressions, not one `NK×NK` problem.

# The asymptotics are not the package's usual ones

Inference here is **`T → ∞` at fixed `N`**, as the paper states: these are
standard structural-VAR asymptotics. Every other panel estimator in this
package is consistent as `N` grows. Bringing a wide, short panel to
`fit_spvar` is a mistake, and [`fit_spvar`](@ref) says so in its docstring
with the measured rates.

# References

- Di Giacinto, V. (2010). On vector autoregressive modeling in space and
  time. *Journal of Geographical Systems*, 12(2), 125–154.
- Anselin, L., & Smirnov, O. (1996). Efficient algorithms for constructing
  proper higher order spatial lag operators. *Journal of Regional Science*,
  36(1), 67–89.
- Lütkepohl, H. (2007). *New Introduction to Multiple Time Series
  Analysis*. Springer.
"""
module SpVAR

using LinearAlgebra
using SparseArrays
using Printf
using Random

import StatsAPI
import StatsAPI: coef, vcov, stderror, residuals, loglikelihood, nobs,
                 coefnames

import ..SpatialUtils: brent_maximize, check_design_rank, _normcdf, _chi2_sf
import ..SpatialWeights: order_weights_list

export SpVARModel, fit_spvar,
       OrderConditionResult, spvar_order_condition,
       STIRResult, space_time_stir,
       spvar_irf, spvar_forecast,
       NestedPanelVARModel, fit_nested_panel_var,
       lr_spatial_interaction

# ═══════════════════════════════════════════════════════════════════════
# Identification: the order condition
# ═══════════════════════════════════════════════════════════════════════

"""
    OrderConditionResult

Verdict from [`spvar_order_condition`](@ref).

# Fields

- `satisfied::Bool` — whether the order condition holds.
- `n_restrictions::Int` — zero restrictions on the structural error
  covariance, `NK(NK−1)/2`, since every covariance is nil and every
  variance free.
- `n_free::Int` — free coefficients in `C₀`.
- `n_min::Int` — the smallest `N` that satisfies the condition at this
  `K` and `s`.
- `homogeneous::Bool` — which parameterisation was counted.
- `N`, `K`, `s`.

!!! warning "Necessary, not sufficient"
    This is the **order** condition for identification via zero
    restrictions (Hamilton 1994, p. 332): it counts equations against
    unknowns and nothing more. Satisfying it does not establish
    identification, and a model that satisfies it can still be
    near-singular. It is a cheap refusal for the cases Di Giacinto names as
    genuinely under-identified — small `N` with high spatial order — not a
    certificate.
"""
struct OrderConditionResult
    satisfied::Bool
    n_restrictions::Int
    n_free::Int
    n_min::Int
    homogeneous::Bool
    N::Int
    K::Int
    s::Int
end

function Base.show(io::IO, r::OrderConditionResult)
    println(io, "SpVAR order condition (",
        r.homogeneous ? "spatially homogeneous" : "spatially heterogeneous",
        ")")
    println(io, "─────────────────────────────────────────────")
    @printf(io, "  N = %d, K = %d, s = %d\n", r.N, r.K, r.s)
    @printf(io, "  restrictions on Λ:    %d\n", r.n_restrictions)
    @printf(io, "  free C₀ coefficients: %d\n", r.n_free)
    @printf(io, "  minimum N:            %d\n", r.n_min)
    println(io, "  Verdict: ", r.satisfied ? "satisfied" : "NOT SATISFIED")
    r.satisfied ||
        println(io, "  → raise N to $(r.n_min), or lower s.")
    println(io, "  (order condition only — necessary, not sufficient)")
end

# Free C₀ coefficients.
#
# Heterogeneous (Di Giacinto's eq. 13): off-diagonal blocks r<k contribute
# (s+1) orders x N locations each, and there are K(K-1)/2 of them; diagonal
# blocks contribute s orders x N locations each (l starts at 1, eq. 10),
# and there are K of them. That is
#
#   N(s+1)K(K-1)/2 + NsK = NK(K-1)/2 + sNK(K+1)/2
#
# which is the paper's printed count — derived independently here and
# pinned against it in the suite.
_n_free_c0(N, K, s, homogeneous) =
    homogeneous ? (s + 1) * div(K * (K - 1), 2) + s * K :
                  N * (s + 1) * div(K * (K - 1), 2) + N * s * K

"""
    spvar_order_condition(N, K, s; homogeneous = true)

Check Di Giacinto's order condition for identification of `C₀` — his eqs.
(13) and (14) — and return an [`OrderConditionResult`](@ref).

The condition compares the `NK(NK−1)/2` zero restrictions on the structural
error covariance against the free coefficients in `C₀`. In the spatially
**heterogeneous** parameterisation it reduces to

```
N ≥ s + (s + K)/K
```

so for `K = 2` it requires `N ≥ 3` at `s = 1` and `N ≥ 4` at `s = 2` —
the paper's own worked numbers, which the suite pins exactly. The bound
*decreases* in `K`, because coefficients grow linearly in `K` while the
restrictions grow quadratically.

In the **homogeneous** parameterisation (his eq. 8) the coefficient count
does not scale with `N` at all, so the condition binds only for very small
`N`.

[`fit_spvar`](@ref) calls this and refuses rather than returning estimates
for an under-identified system.
"""
function spvar_order_condition(N::Int, K::Int, s::Int;
                               homogeneous::Bool = true)
    N >= 1 || throw(ArgumentError("spvar_order_condition: N must be ≥ 1"))
    K >= 1 || throw(ArgumentError("spvar_order_condition: K must be ≥ 1"))
    s >= 0 || throw(ArgumentError("spvar_order_condition: s must be ≥ 0"))

    nres = div(N * K * (N * K - 1), 2)
    nfree = _n_free_c0(N, K, s, homogeneous)

    nmin = N
    for cand in 1:max(N, 1000)
        if div(cand * K * (cand * K - 1), 2) >=
           _n_free_c0(cand, K, s, homogeneous)
            nmin = cand
            break
        end
    end

    return OrderConditionResult(nres >= nfree, nres, nfree, nmin,
                                homogeneous, N, K, s)
end

# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

# log|A| for a sparse A, -Inf when singular or sign-flipped. Same guard as
# SpatioTemporalModels._logdet_jacobian: a negative determinant means the
# model is outside the admissible region, not that we should take |.|.
function _logdet_block(A::AbstractSparseMatrix{T}) where {T}
    F = lu(A; check = false)
    issuccess(F) || return T(-Inf)
    ld, sgn = logabsdet(F)
    sgn > 0 || return T(-Inf)
    return T(ld)
end

# A⁽⁰⁾ₖₖ = I − Σ_{l≥1} ρ_l W⁽ˡ⁾   (eq. 10, diagonal block)
function _a0_diag(Wl::Vector{<:AbstractSparseMatrix{T}}, rho::AbstractVector{T},
                  N::Int) where {T}
    A = sparse(one(T) * I, N, N)
    @inbounds for l in eachindex(rho)
        A = A - rho[l] * Wl[l + 1]        # Wl[1] is W⁽⁰⁾ = I
    end
    return A
end

# Reshape the user's N*T_per x K panel into K matrices of size N x T_per,
# each row a location's time series. Input rows follow the package's
# unit-within-time convention, (t-1)*N + i.
function _split_variables(Y::AbstractMatrix{T}, N::Int, T_per::Int) where {T}
    K = size(Y, 2)
    out = Vector{Matrix{T}}(undef, K)
    @inbounds for k in 1:K
        M = Matrix{T}(undef, N, T_per)
        col = view(Y, :, k)
        for t in 1:T_per, i in 1:N
            M[i, t] = col[(t - 1) * N + i]
        end
        out[k] = M
    end
    return out
end

# Subtract each location's own mean over the estimation window. This
# absorbs Di Giacinto's location-specific intercept vector μ exactly
# (Frisch-Waugh), so estimating it and demeaning are the same thing; the
# latter just keeps the Hessian small.
function _demean_rows!(M::AbstractMatrix{T}, t0::Int, t1::Int) where {T}
    mu = Vector{T}(undef, size(M, 1))
    @inbounds for i in axes(M, 1)
        s = zero(T)
        for t in t0:t1
            s += M[i, t]
        end
        m = s / (t1 - t0 + 1)
        mu[i] = m
        for t in axes(M, 2)
            M[i, t] -= m
        end
    end
    return mu
end

# ═══════════════════════════════════════════════════════════════════════
# The model
# ═══════════════════════════════════════════════════════════════════════

"""
    SpVARModel

A fitted spatial VAR, from [`fit_spvar`](@ref).

# Fields

- `rho::Vector{Vector{T}}` — `rho[k][l]` is the own contemporaneous spatial
  coefficient on `W⁽ˡ⁾` in equation `k`, `l = 1,…,s`. These are the
  endogenous terms, the ones the Jacobian exists for.
- `theta::Vector{Vector{T}}` — the predetermined coefficients of equation
  `k`: the `r < k` contemporaneous blocks first, then the lag blocks.
- `lambda::Matrix{T}` — `N×K` of structural error variances `λ̂ᵢₖ`
  (eq. 21). Location-specific by construction: the paper's `Λ` is diagonal
  with free diagonal.
- `means::Matrix{T}` — `N×K` location means removed to absorb the paper's
  intercept vector `μ`, in causal order. Kept because
  [`spvar_forecast`](@ref) needs them to return levels rather than
  deviations.
- `C0::Matrix{T}`, `C::Vector{Matrix{T}}` — the assembled `NK×NK`
  structural matrices, `C[h]` for lag `h`.
- `Wl::Vector{SparseMatrixCSC{T,Int}}` — `[W⁽⁰⁾ = I, W⁽¹⁾, …, W⁽ˢ⁾]`.
- `ordering::Vector{Int}` — the causal order, as column indices of the `Y`
  that was passed in. `ordering[1]` moves first.
- `varnames::Vector{String}` — in causal order.
- `vcov_eq::Vector{Matrix{T}}` — per-equation covariance of
  `[rho[k]; theta[k]]`, from the numerical Hessian of that equation's
  concentrated log-likelihood. Block diagonal across equations because the
  likelihood separates (eq. 23).
- `loglik::T` — the concentrated log-likelihood, eq. (22).
- `loglik_eq::Vector{T}` — its `K` separable components, eq. (23).
- `residuals::Matrix{T}` — `N*T_eff × K` structural residuals, in causal
  order.
- `N`, `K`, `p`, `s`, `T_periods`, `T_eff`.
- `spectral_radius::T`, `stationary::Bool` — of the `Kp·N` companion
  matrix of the reduced form, not of any single `C_h`.
- `converged::Bool`, `iterations::Int`.
- `homogeneous::Bool` — always `true` in this version; see
  [`fit_spvar`](@ref).
"""
struct SpVARModel{T<:AbstractFloat} <: StatsAPI.StatisticalModel
    rho::Vector{Vector{T}}
    theta::Vector{Vector{T}}
    lambda::Matrix{T}
    means::Matrix{T}
    C0::Matrix{T}
    C::Vector{Matrix{T}}
    Wl::Vector{SparseMatrixCSC{T, Int}}
    ordering::Vector{Int}
    varnames::Vector{String}
    vcov_eq::Vector{Matrix{T}}
    coefnames_eq::Vector{Vector{String}}
    loglik::T
    loglik_eq::Vector{T}
    residuals::Matrix{T}
    N::Int
    K::Int
    p::Int
    s::Int
    T_periods::Int
    T_eff::Int
    spectral_radius::T
    stationary::Bool
    converged::Bool
    iterations::Int
    homogeneous::Bool
end

StatsAPI.nobs(m::SpVARModel) = m.N * m.T_eff
StatsAPI.loglikelihood(m::SpVARModel) = m.loglik
StatsAPI.residuals(m::SpVARModel) = m.residuals
StatsAPI.coef(m::SpVARModel) = vcat((vcat(m.rho[k], m.theta[k])
                                     for k in 1:m.K)...)
StatsAPI.coefnames(m::SpVARModel) = vcat(m.coefnames_eq...)

function StatsAPI.vcov(m::SpVARModel{T}) where {T}
    n = sum(size(V, 1) for V in m.vcov_eq)
    V = zeros(T, n, n)
    off = 0
    for Vk in m.vcov_eq
        q = size(Vk, 1)
        V[(off + 1):(off + q), (off + 1):(off + q)] .= Vk
        off += q
    end
    return V
end

StatsAPI.stderror(m::SpVARModel) = sqrt.(max.(diag(vcov(m)), 0))

function Base.show(io::IO, m::SpVARModel{T}) where {T}
    println(io, "Spatial VAR — SpVAR(", m.p, ",", m.s, ")")
    println(io, "═════════════════════════════════════════════════")
    @printf(io, "  N = %d locations, K = %d variables, T = %d (%d effective)\n",
        m.N, m.K, m.T_periods, m.T_eff)
    println(io, "  causal ordering: ", join(m.varnames, " → "))
    @printf(io, "  log-likelihood: %.4f   (Σ of %d separable terms)\n",
        m.loglik, m.K)
    @printf(io, "  spectral radius: %.4f  %s\n", m.spectral_radius,
        m.stationary ? "(stationary)" : "⚠ NOT stationary")
    m.converged || println(io,
        "  ⚠ the profile search did not converge in $(m.iterations) iterations")
    println(io)
    se = stderror(m)
    off = 0
    for k in 1:m.K
        println(io, "  Equation ", k, ": ", m.varnames[k])
        nm = m.coefnames_eq[k]
        th = vcat(m.rho[k], m.theta[k])
        @printf(io, "    %-22s %10s %10s %8s\n", "", "coef", "s.e.", "z")
        for j in eachindex(th)
            s = se[off + j]
            @printf(io, "    %-22s %10.4f %10.4f %8.2f\n",
                nm[j], th[j], s, s > 0 ? th[j] / s : NaN)
        end
        off += length(th)
        println(io)
    end
    println(io, "  Inference is T → ∞ at fixed N (Di Giacinto 2010).")
end

# ═══════════════════════════════════════════════════════════════════════
# Estimation
# ═══════════════════════════════════════════════════════════════════════

# Build the per-equation design. Returns (yv, Zend, Zpre, names).
#
#   yv    : the dependent variable, stacked over (i, t)
#   Zend  : W⁽ˡ⁾y_kt for l = 1..s          -- endogenous, Jacobian terms
#   Zpre  : W⁽ˡ⁾y_rt for r < k, l = 0..s   -- predetermined by the ordering
#           W⁽ˡ⁾y_{r,t-h} for all r, h, l
function _equation_design(Ys::Vector{Matrix{T}},
                         Wl::Vector{<:AbstractSparseMatrix{T}},
                         k::Int, K::Int, N::Int, p::Int, s::Int,
                         T_per::Int, varnames::Vector{String}) where {T}
    T_eff = T_per - p
    nrow = N * T_eff
    yv = Vector{T}(undef, nrow)
    @inbounds for t in (p + 1):T_per, i in 1:N
        yv[(t - p - 1) * N + i] = Ys[k][i, t]
    end

    Zend = Matrix{T}(undef, nrow, s)
    @inbounds for l in 1:s
        Wk = Wl[l + 1]
        for t in (p + 1):T_per
            col = Wk * view(Ys[k], :, t)
            off = (t - p - 1) * N
            for i in 1:N
                Zend[off + i, l] = col[i]
            end
        end
    end

    ncol = (s + 1) * (k - 1) + (s + 1) * K * p
    Zpre = Matrix{T}(undef, nrow, ncol)
    names = String[]
    c = 0
    # Contemporaneous, r < k only: the block-triangular structure of C₀.
    @inbounds for r in 1:(k - 1), l in 0:s
        c += 1
        Wk = Wl[l + 1]
        for t in (p + 1):T_per
            colv = Wk * view(Ys[r], :, t)
            off = (t - p - 1) * N
            for i in 1:N
                Zpre[off + i, c] = colv[i]
            end
        end
        push!(names, l == 0 ? varnames[r] : "W^$l·$(varnames[r])")
    end
    # Lags, all variables.
    @inbounds for h in 1:p, r in 1:K, l in 0:s
        c += 1
        Wk = Wl[l + 1]
        for t in (p + 1):T_per
            colv = Wk * view(Ys[r], :, t - h)
            off = (t - p - 1) * N
            for i in 1:N
                Zpre[off + i, c] = colv[i]
            end
        end
        push!(names,
            (l == 0 ? "" : "W^$l·") * "$(varnames[r])(t-$h)")
    end

    endnames = ["W^$l·$(varnames[k])" for l in 1:s]
    return yv, Zend, Zpre, vcat(endnames, names)
end

# Iterated GLS for the predetermined coefficients with location-specific
# variances, plus the resulting λ̂. The fixed point of eq. (21) and the GLS
# normal equations: given λ, weight; given θ, update λ.
function _igls(ytil::Vector{T}, Z::Matrix{T}, N::Int, T_eff::Int;
               maxiter::Int = 50, tol::T = T(1e-11)) where {T}
    nrow = length(ytil)
    lam = ones(T, N)
    theta = zeros(T, size(Z, 2))
    u = similar(ytil)
    w = Vector{T}(undef, nrow)
    prev = T(Inf)
    local iters = 0
    for it in 1:maxiter
        iters = it
        @inbounds for t in 1:T_eff, i in 1:N
            w[(t - 1) * N + i] = one(T) / sqrt(lam[i])
        end
        if size(Z, 2) > 0
            Zw = Z .* w
            yw = ytil .* w
            theta = Zw \ yw
            mul!(u, Z, theta)
            @. u = ytil - u
        else
            copyto!(u, ytil)
        end
        fill!(lam, zero(T))
        @inbounds for t in 1:T_eff, i in 1:N
            lam[i] += u[(t - 1) * N + i]^2
        end
        @. lam = lam / T_eff
        any(x -> !(x > 0) || !isfinite(x), lam) && return theta, lam, u, false, iters
        obj = sum(log, lam)
        abs(prev - obj) <= tol * (one(T) + abs(obj)) && return theta, lam, u, true, iters
        prev = obj
    end
    return theta, lam, u, false, iters
end

# The separable concentrated log-likelihood of one equation, eq. (23),
# dropping the constant. Returns (value, theta, lambda, residuals).
function _eq_profile(rho::Vector{T}, yv::Vector{T}, Zend::Matrix{T},
                     Zpre::Matrix{T}, Wl::Vector{<:AbstractSparseMatrix{T}},
                     N::Int, T_eff::Int; igls_maxiter::Int = 50) where {T}
    A0 = _a0_diag(Wl, rho, N)
    ld = _logdet_block(A0)
    isfinite(ld) || return (T(-Inf), T[], T[], T[])
    ytil = size(Zend, 2) > 0 ? yv .- Zend * rho : copy(yv)
    theta, lam, u, ok, _ = _igls(ytil, Zpre, N, T_eff;
                                 maxiter = igls_maxiter)
    ok || return (T(-Inf), theta, lam, u)
    val = T_eff * ld - (T_eff / 2) * sum(log, lam)
    return (val, theta, lam, u)
end

"""
    fit_spvar(Y, W, N, T_per; ordering, p = 1, s = 1, Wl = nothing,
              rho_bounds = (-0.95, 0.95), maxiter = 200, tol = 1e-9,
              varnames = nothing)

Fit Di Giacinto's (2010) structural spatial VAR by full-information maximum
likelihood.

# Arguments

- `Y`: `N*T_per × K`. Columns are variables; rows follow the package's
  unit-within-time convention, `(t−1)*N + i`, as every other panel
  estimator here takes. The paper's location-within-variable stacking is
  built internally.
- `W`: first-order `N×N` weights. Higher orders are derived with
  [`order_weights_list`](@ref) unless `Wl` is supplied.
- `ordering`: **required.** A permutation of `1:K` (or a vector of names)
  giving the contemporaneous causal order, first mover first.

# Why `ordering` has no default

Identification of `C₀` rests entirely on it. There is no data-driven
default and offering one would hide the single assumption the user is
making. Reversing it changes the impulse responses — measured, reversing a
two-variable ordering moves the peak global outward STIR by a factor
reported in `docs/AUDIT.md`, not by a rounding error.

# Spatial order and the log-determinant

Each equation needs `log|I − Σₗ ρₗW⁽ˡ⁾|`. At `s = 1` this is the familiar
`log|I − ρW|`; at `s ≥ 2` it is a combination of several weights matrices,
and the package's `chebyshev_logdet` **does not apply** — it approximates a
one-parameter family by construction. Sparse LU is used throughout, which
is also what the paper assumes ("moderate" `N`).

# Spatial homogeneity

This version fits the **spatially homogeneous** specification, the paper's
eq. (8): coefficients are common across locations, while the structural
error variances `λᵢₖ` remain location-specific (its eq. 4 keeps `Λ`'s
diagonal free). The fully heterogeneous eq. (7), with a separate
coefficient per location, is not implemented; `spvar_order_condition` still
counts both so the guard is available either way.

# Asymptotics

`T → ∞` at fixed `N`. This is **not** a large-`N` estimator, unlike
everything else in the package's panel modules. Location-specific
intercepts are absorbed by demeaning each location's series over the
estimation window, which is algebraically the same as estimating them
(Frisch–Waugh) and carries the usual dynamic-panel bias of order `1/T`.

See also [`space_time_stir`](@ref), [`spvar_irf`](@ref),
[`lr_spatial_interaction`](@ref).
"""
function fit_spvar(
    Y::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    N::Int,
    T_per::Int;
    ordering::Union{Nothing, AbstractVector} = nothing,
    p::Int = 1,
    s::Int = 1,
    Wl::Union{Nothing, Vector{<:AbstractSparseMatrix{T}}} = nothing,
    rho_bounds::Tuple{Real, Real} = (-0.95, 0.95),
    maxiter::Int = 200,
    tol::Real = 1e-9,
    varnames::Union{Nothing, Vector{String}} = nothing
) where {T<:AbstractFloat}

    K = size(Y, 2)
    K >= 1 || throw(ArgumentError("fit_spvar: Y must have ≥ 1 column"))
    p >= 1 || throw(ArgumentError("fit_spvar: p must be ≥ 1, got $p"))
    s >= 0 || throw(ArgumentError("fit_spvar: s must be ≥ 0, got $s"))
    size(Y, 1) == N * T_per || throw(DimensionMismatch(
        "fit_spvar: Y has $(size(Y, 1)) rows, expected N*T_per = $(N * T_per)"))
    size(W) == (N, N) || throw(DimensionMismatch(
        "fit_spvar: W is $(size(W)) but N = $N"))
    T_per > p + 1 || throw(ArgumentError(
        "fit_spvar: need T_per > p + 1 for a non-degenerate sample, " *
        "got T_per = $T_per and p = $p"))
    all(isfinite, Y) || throw(ArgumentError(
        "fit_spvar: Y contains non-finite values"))

    names0 = varnames === nothing ? ["y$k" for k in 1:K] : copy(varnames)
    length(names0) == K || throw(DimensionMismatch(
        "fit_spvar: varnames has $(length(names0)) entries but K = $K"))

    # ── the ordering is the identifying assumption ──────────────────
    ordering === nothing && throw(ArgumentError(
        "fit_spvar: `ordering` is required and has no default. " *
        "Identification of C₀ rests on the contemporaneous causal order " *
        "of the K = $K variables, which the data cannot settle. Pass a " *
        "permutation of 1:$K (first mover first), or the variable names " *
        "in that order."))
    ord = if eltype(ordering) <: Integer
        collect(Int, ordering)
    else
        idx = Vector{Int}(undef, length(ordering))
        for (j, nm) in enumerate(ordering)
            pos = findfirst(==(String(nm)), names0)
            pos === nothing && throw(ArgumentError(
                "fit_spvar: ordering names \"$nm\", which is not in varnames"))
            idx[j] = pos
        end
        idx
    end
    sort(ord) == collect(1:K) || throw(ArgumentError(
        "fit_spvar: ordering must be a permutation of 1:$K, got $ord"))

    # ── identification guard, before any estimation ─────────────────
    oc = spvar_order_condition(N, K, s; homogeneous = true)
    oc.satisfied || throw(ArgumentError(
        "fit_spvar: the order condition for identification of C₀ fails — " *
        "$(oc.n_free) free coefficients against $(oc.n_restrictions) " *
        "restrictions at N = $N, K = $K, s = $s. Raise N to " *
        "$(oc.n_min) or lower s. (Di Giacinto 2010, eqs. 13–14.)"))

    # ── higher-order weights ────────────────────────────────────────
    Wlist = if Wl === nothing
        order_weights_list(W, s)
    else
        length(Wl) == s + 1 || throw(DimensionMismatch(
            "fit_spvar: Wl has $(length(Wl)) matrices, expected s + 1 = $(s + 1)"))
        Wl
    end
    Wls = SparseMatrixCSC{T, Int}[sparse(T.(M)) for M in Wlist]
    for (l, M) in enumerate(Wls)
        size(M) == (N, N) || throw(DimensionMismatch(
            "fit_spvar: W⁽$(l - 1)⁾ is $(size(M)) but N = $N"))
    end

    # ── reshape, permute to causal order, absorb μ ──────────────────
    Ys_all = _split_variables(T.(Y), N, T_per)
    Ys = [copy(Ys_all[ord[k]]) for k in 1:K]
    names = [names0[ord[k]] for k in 1:K]
    mu = Matrix{T}(undef, N, K)
    for k in 1:K
        mu[:, k] .= _demean_rows!(Ys[k], p + 1, T_per)
    end

    T_eff = T_per - p
    lo, hi = T(rho_bounds[1]), T(rho_bounds[2])

    rhos = Vector{Vector{T}}(undef, K)
    thetas = Vector{Vector{T}}(undef, K)
    lam = Matrix{T}(undef, N, K)
    lls = Vector{T}(undef, K)
    resid = Matrix{T}(undef, N * T_eff, K)
    cn = Vector{Vector{String}}(undef, K)
    allconv = true
    totiter = 0

    # ── eq. (23): K unrelated maximisations ─────────────────────────
    for k in 1:K
        yv, Zend, Zpre, nm = _equation_design(Ys, Wls, k, K, N, p, s,
                                              T_per, names)
        if size(Zpre, 2) > 0
            check_design_rank(Zpre, nm[(s + 1):end], "fit_spvar";
                context = "Equation $k ($(names[k])) is the one affected. " *
                    "A redundant spatial order is the usual cause: if " *
                    "W⁽ˡ⁾ repeats structure already in a lower order, " *
                    "lower s.")
        end

        obj = r -> _eq_profile(r, yv, Zend, Zpre, Wls, N, T_eff)[1]

        rho_k = zeros(T, s)
        conv = true
        iters = 0
        if s == 0
            conv = true
        elseif s == 1
            r1, _ = brent_maximize(x -> obj([x]), lo, hi; tol = T(tol))
            rho_k[1] = r1
            iters = 1
        else
            # Cyclic coordinate ascent on Brent. Deliberately simple and
            # deliberately reported: it can stall on a ridge, so the suite
            # checks it against a grid search.
            prev = obj(rho_k)
            conv = false
            for it in 1:maxiter
                iters = it
                for l in 1:s
                    f = x -> begin
                        cand = copy(rho_k); cand[l] = x; obj(cand)
                    end
                    xl, _ = brent_maximize(f, lo, hi; tol = T(tol))
                    rho_k[l] = xl
                end
                cur = obj(rho_k)
                if abs(cur - prev) <= tol * (one(T) + abs(cur))
                    conv = true
                    break
                end
                prev = cur
            end
        end
        allconv &= conv
        totiter += iters

        val, th, lk, u = _eq_profile(rho_k, yv, Zend, Zpre, Wls, N, T_eff)
        isfinite(val) || throw(ErrorException(
            "fit_spvar: equation $k ($(names[k])) has a non-finite " *
            "concentrated log-likelihood at the optimum; C₀'s diagonal " *
            "block is numerically singular."))

        rhos[k] = rho_k
        thetas[k] = th
        lam[:, k] .= lk
        lls[k] = val
        resid[:, k] .= u
        cn[k] = ["eq$k: " * x for x in nm]
    end

    # ── assemble C₀ and C_h (eqs. 6, 9, 10) ─────────────────────────
    C0 = zeros(T, N * K, N * K)
    Cs = [zeros(T, N * K, N * K) for _ in 1:p]
    blk(k) = ((k - 1) * N + 1):(k * N)
    for k in 1:K
        C0[blk(k), blk(k)] .= Matrix(_a0_diag(Wls, rhos[k], N))
        c = 0
        for r in 1:(k - 1), l in 0:s
            c += 1
            C0[blk(k), blk(r)] .-= thetas[k][c] .* Matrix(Wls[l + 1])
        end
        for h in 1:p, r in 1:K, l in 0:s
            c += 1
            Cs[h][blk(k), blk(r)] .+= thetas[k][c] .* Matrix(Wls[l + 1])
        end
    end

    # ── stationarity of the reduced form's companion matrix ─────────
    rad, stat = _spvar_radius(C0, Cs, N, K, p)

    # eq. (22): the constant that eq. (23) drops.
    const_term = -T(N * K * T_eff) / 2 * log(T(2π)) - T(N * K * T_eff) / 2
    ll = sum(lls) + const_term

    # ── per-equation covariance from the numerical Hessian ──────────
    vcs = Vector{Matrix{T}}(undef, K)
    for k in 1:K
        yv, Zend, Zpre, _ = _equation_design(Ys, Wls, k, K, N, p, s,
                                             T_per, names)
        vcs[k] = _eq_vcov(rhos[k], thetas[k], yv, Zend, Zpre, Wls, N, T_eff)
    end

    return SpVARModel{T}(rhos, thetas, lam, mu, C0, Cs, Wls, ord, names,
                         vcs, cn, ll, lls, resid, N, K, p, s, T_per, T_eff,
                         rad, stat, allconv, totiter, true)
end

# log-likelihood of one equation as a function of the FULL parameter vector
# (rho; theta), with λ concentrated out by eq. (21). Used only for the
# Hessian: the profile above is faster but gives no curvature in theta.
function _eq_ll_full(par::AbstractVector{T}, yv::Vector{T}, Zend::Matrix{T},
                     Zpre::Matrix{T}, Wl::Vector{<:AbstractSparseMatrix{T}},
                     N::Int, T_eff::Int, s::Int) where {T}
    rho = par[1:s]
    th = par[(s + 1):end]
    A0 = _a0_diag(Wl, collect(rho), N)
    ld = _logdet_block(A0)
    isfinite(ld) || return T(-Inf)
    u = copy(yv)
    s > 0 && (u .-= Zend * rho)
    size(Zpre, 2) > 0 && (u .-= Zpre * th)
    lam = zeros(T, N)
    @inbounds for t in 1:T_eff, i in 1:N
        lam[i] += u[(t - 1) * N + i]^2
    end
    @. lam = lam / T_eff
    any(x -> !(x > 0) || !isfinite(x), lam) && return T(-Inf)
    return T_eff * ld - (T_eff / 2) * sum(log, lam)
end

function _eq_vcov(rho::Vector{T}, theta::Vector{T}, yv::Vector{T},
                  Zend::Matrix{T}, Zpre::Matrix{T},
                  Wl::Vector{<:AbstractSparseMatrix{T}},
                  N::Int, T_eff::Int) where {T}
    s = length(rho)
    par = vcat(rho, theta)
    q = length(par)
    q == 0 && return zeros(T, 0, 0)
    f = x -> _eq_ll_full(x, yv, Zend, Zpre, Wl, N, T_eff, s)
    H = zeros(T, q, q)
    h = Vector{T}(undef, q)
    @inbounds for j in 1:q
        h[j] = max(sqrt(eps(T)) * max(abs(par[j]), one(T)), T(1e-6))
    end
    f0 = f(par)
    @inbounds for a in 1:q
        for b in a:q
            if a == b
                pp = copy(par); pp[a] += h[a]
                pm = copy(par); pm[a] -= h[a]
                H[a, a] = (f(pp) - 2 * f0 + f(pm)) / (h[a]^2)
            else
                ppp = copy(par); ppp[a] += h[a]; ppp[b] += h[b]
                ppm = copy(par); ppm[a] += h[a]; ppm[b] -= h[b]
                pmp = copy(par); pmp[a] -= h[a]; pmp[b] += h[b]
                pmm = copy(par); pmm[a] -= h[a]; pmm[b] -= h[b]
                v = (f(ppp) - f(ppm) - f(pmp) + f(pmm)) /
                    (4 * h[a] * h[b])
                H[a, b] = v
                H[b, a] = v
            end
        end
    end
    # -H⁻¹, symmetrised, with a pseudo-inverse fallback when the numerical
    # Hessian is not negative definite rather than a silent NaN block.
    Hs = Symmetric(-(H + H') / 2)
    V = try
        inv(cholesky(Hs))
    catch
        pinv(Matrix(Hs))
    end
    return Matrix{T}((V + V') / 2)
end

# Spectral radius of the reduced form's companion matrix. Built from
# C̃_h = C₀⁻¹C_h (eq. 11) and stacked into the usual Kp·N companion form,
# because with p > 1 no single C_h carries the answer.
function _spvar_radius(C0::Matrix{T}, Cs::Vector{Matrix{T}},
                       N::Int, K::Int, p::Int) where {T}
    n = N * K
    F = lu(C0; check = false)
    issuccess(F) || return T(Inf), false
    Ct = [F \ Ch for Ch in Cs]
    if p == 1
        return T(maximum(abs, eigvals(Ct[1]))), maximum(abs, eigvals(Ct[1])) < 1
    end
    Comp = zeros(T, n * p, n * p)
    for h in 1:p
        Comp[1:n, ((h - 1) * n + 1):(h * n)] .= Ct[h]
    end
    for h in 1:(p - 1)
        Comp[(h * n + 1):((h + 1) * n), ((h - 1) * n + 1):(h * n)] .=
            Matrix{T}(I, n, n)
    end
    rad = T(maximum(abs, eigvals(Comp)))
    return rad, rad < 1
end

# ═══════════════════════════════════════════════════════════════════════
# Impulse responses (eqs. 24–28)
# ═══════════════════════════════════════════════════════════════════════

"""
    spvar_irf(m::SpVARModel; horizon = 10)

The structural moving-average coefficients `Ψ₀,…,Ψ_H` of Di Giacinto's
eqs. (24)–(26), as a `Vector` of `NK×NK` matrices.

`Ψ_h[(k−1)N+i, (r−1)N+j] = ∂yᵢₖ,ₜ₊ₕ/∂u_{jrt}` — his eq. (28): the response
of variable `k` at location `i`, `h` periods on, to a one-unit structural
shock to variable `r` at location `j`.

Built from the reduced form `C̃_h = C₀⁻¹C_h` by `Ψ̃₀ = I`,
`Ψ̃ᵢ = Σ_{h} Ψ̃ᵢ₋ₕ C̃_h`, then `Ψ_h = Ψ̃_h C₀⁻¹`.

Summed over all horizons this converges to `(C₀ − Σ_h C_h)⁻¹`, the long-run
multiplier, which is the identity the suite pins it against.

For anything but a handful of locations the raw `Ψ_h` is unreadable —
`NK×NK` numbers per horizon. [`space_time_stir`](@ref) is the summary
designed for that.
"""
function spvar_irf(m::SpVARModel{T}; horizon::Int = 10) where {T}
    horizon >= 0 || throw(ArgumentError(
        "spvar_irf: horizon must be ≥ 0, got $horizon"))
    n = m.N * m.K
    F = lu(m.C0; check = false)
    issuccess(F) || throw(ErrorException(
        "spvar_irf: C₀ is numerically singular, so the reduced form does " *
        "not exist. The fit should not have produced this; report it."))
    C0inv = F \ Matrix{T}(I, n, n)
    Ct = [F \ Ch for Ch in m.C]

    psit = Vector{Matrix{T}}(undef, horizon + 1)
    psit[1] = Matrix{T}(I, n, n)
    for i in 1:horizon
        P = zeros(T, n, n)
        for h in 1:min(i, m.p)
            P .+= psit[i - h + 1] * Ct[h]
        end
        psit[i + 1] = P
    end
    return [P * C0inv for P in psit]
end

# ═══════════════════════════════════════════════════════════════════════
# The space-time impulse response function (eqs. 29–32)
# ═══════════════════════════════════════════════════════════════════════

"""
    STIRResult

Space-time impulse responses, from [`space_time_stir`](@ref).

# Fields

- `outward::Array{T,4}` — `[h+1, l+1, k, r]`, the **global outward** STIR
  of eq. (31): the average effect, `h` periods on, on variable `k` at the
  order-`l` neighbours of a unit shock to variable `r`.
- `inward::Array{T,4}` — the **global inward** STIR of eq. (32): the
  average effect on variable `k` at a location of shocks to variable `r` at
  *its* order-`l` neighbours.
- `local_outward`, `local_inward::Array{T,5}` — `[h+1, l+1, k, r, i]`, the
  per-location eqs. (29)–(30). Empty unless `store_local = true`.
- `horizon::Int`, `orders::UnitRange{Int}` — `0:s`.
- `varnames::Vector{String}` — in causal order.
- `N`, `K`, `cumulative::Bool`.

# Reading it

Order `l = 0` uses `W⁽⁰⁾ = I`, so it is the **own-location** response; the
spillovers are `l ≥ 1`, indexed by graph distance. Outward and inward
differ by a transpose of the response block and coincide only in special
cases — see [`space_time_stir`](@ref).
"""
struct STIRResult{T<:AbstractFloat}
    outward::Array{T, 4}
    inward::Array{T, 4}
    local_outward::Array{T, 5}
    local_inward::Array{T, 5}
    horizon::Int
    orders::UnitRange{Int}
    varnames::Vector{String}
    N::Int
    K::Int
    cumulative::Bool
end

function Base.show(io::IO, r::STIRResult)
    println(io, "Space-Time Impulse Response (STIR)",
        r.cumulative ? " — cumulative" : "")
    println(io, "══════════════════════════════════════════")
    println(io, "  global outward: effect on order-l neighbours of a shock here")
    for k in 1:r.K, rr in 1:r.K
        @printf(io, "\n  %s ← shock to %s\n", r.varnames[k], r.varnames[rr])
        @printf(io, "    %-5s", "h")
        for l in r.orders
            @printf(io, " %12s", l == 0 ? "own (l=0)" : "l = $l")
        end
        println(io)
        for h in 0:min(r.horizon, 8)
            @printf(io, "    %-5d", h)
            for l in r.orders
                @printf(io, " %12.6f", r.outward[h + 1, l + 1, k, rr])
            end
            println(io)
        end
    end
end

"""
    space_time_stir(m::SpVARModel; horizon = 10, cumulative = false,
                    store_local = false)

Di Giacinto's space-time impulse response function — his eqs. (29)–(32).

The raw `Ψ_h` of [`spvar_irf`](@ref) holds `(NK)²` numbers per horizon,
which no one can read. The STIR collapses the spatial dimension by
averaging over *spatial order* instead of over locations, so what survives
is a response indexed by **time since the shock** and **distance from it** —
which is the question a spatial VAR is usually fitted to answer.

Two directions, and they are not the same quantity:

- **outward**, eq. (29)/(31):
  `ζ⁽ʰˡ⁾ₖᵣ(i) = Σⱼ w⁽ˡ⁾ᵢⱼ Ψ⁽ʰ⁾ₖᵣ(j,i)` — shock at `i`, effect on `i`'s
  order-`l` neighbours. "Where does my shock go?"
- **inward**, eq. (30)/(32):
  `ξ⁽ʰˡ⁾ₖᵣ(i) = Σⱼ w⁽ˡ⁾ᵢⱼ Ψ⁽ʰ⁾ₖᵣ(i,j)` — shocks at the neighbours, effect
  at `i`. "Whose shocks reach me?"

They differ by transposing the response block, so globally they are
`tr(W⁽ˡ⁾Ψ)/N` against `Σᵢⱼ W⁽ˡ⁾ᵢⱼΨᵢⱼ/N`. These coincide exactly when the
response block is symmetric, and differ otherwise — a unit that sends more
than it receives is precisely what an asymmetric response captures.

!!! note "Row standardisation alone makes them differ"
    The obvious guess is that outward and inward coincide whenever `K = 1`,
    since the blocks are then polynomials in the weights and commute. That
    is true only if `W⁽ˡ⁾` is **symmetric**, and row-standardising an
    irregular graph is enough to break it: on a 14-unit path the
    row-standardised `W⁽¹⁾` has `max|W − Wᵀ| = 0.5`, and the two STIRs
    differ by `1.345e-02` (`examples/29_spvar.jl`, section 4).

    Measured on the same design with symmetry preserved, they agree to
    machine precision: `2.776e-17` with the **spectrally** standardised
    path weights, whose `max|W − Wᵀ|` is exactly zero, and `0.0e+00` on a
    **ring**, where every unit has two neighbours so even row
    standardisation keeps `W⁽¹⁾` symmetric.

    If you want the two directions to coincide on an irregular graph,
    `spectral_standardize` bounds the spectrum without touching symmetry.
    Note that leaving the weights raw is not an alternative: a binary path
    has eigenvalues near 2, so a model that is stationary under
    standardised weights is explosive under raw ones.

    So the gap is not only a multivariate effect. Under row
    standardisation, a unit with fewer neighbours sends and receives on
    different scales, and the outward/inward split reports that.

`cumulative = true` accumulates over horizons, so entry `h` is the total
effect through period `h`.

With `store_local = true` the per-location eqs. (29)–(30) are kept as well,
at `N` times the memory.
"""
function space_time_stir(m::SpVARModel{T}; horizon::Int = 10,
                         cumulative::Bool = false,
                         store_local::Bool = false) where {T}
    psi = spvar_irf(m; horizon = horizon)
    N, K, s = m.N, m.K, m.s
    orders = 0:s

    out = zeros(T, horizon + 1, s + 1, K, K)
    inw = zeros(T, horizon + 1, s + 1, K, K)
    lout = store_local ? zeros(T, horizon + 1, s + 1, K, K, N) :
                         zeros(T, 0, 0, 0, 0, 0)
    linw = store_local ? zeros(T, horizon + 1, s + 1, K, K, N) :
                         zeros(T, 0, 0, 0, 0, 0)

    blk(k) = ((k - 1) * N + 1):(k * N)
    for hh in 0:horizon
        P = psi[hh + 1]
        for k in 1:K, r in 1:K
            B = view(P, blk(k), blk(r))
            for l in orders
                Wl = m.Wl[l + 1]
                so = zero(T); si = zero(T)
                rows = rowvals(Wl); vals = nonzeros(Wl)
                for j in 1:N
                    for idx in nzrange(Wl, j)
                        i = rows[idx]
                        w = vals[idx]           # w = W⁽ˡ⁾[i, j]
                        so += w * B[j, i]       # eq. (29)
                        si += w * B[i, j]       # eq. (30)
                        if store_local
                            lout[hh + 1, l + 1, k, r, i] += w * B[j, i]
                            linw[hh + 1, l + 1, k, r, i] += w * B[i, j]
                        end
                    end
                end
                out[hh + 1, l + 1, k, r] = so / N
                inw[hh + 1, l + 1, k, r] = si / N
            end
        end
    end

    if cumulative
        for l in 1:(s + 1), k in 1:K, r in 1:K
            for hh in 2:(horizon + 1)
                out[hh, l, k, r] += out[hh - 1, l, k, r]
                inw[hh, l, k, r] += inw[hh - 1, l, k, r]
            end
            if store_local
                for i in 1:N, hh in 2:(horizon + 1)
                    lout[hh, l, k, r, i] += lout[hh - 1, l, k, r, i]
                    linw[hh, l, k, r, i] += linw[hh - 1, l, k, r, i]
                end
            end
        end
    end

    return STIRResult{T}(out, inw, lout, linw, horizon, orders,
                         copy(m.varnames), N, K, cumulative)
end

# ═══════════════════════════════════════════════════════════════════════
# Forecasting (eq. 12)
# ═══════════════════════════════════════════════════════════════════════

"""
    spvar_forecast(m::SpVARModel, h; y_history)

`h`-step dynamic forecasts from the reduced form, his eq. (12):
`E[yₜ | ℱₜ₋₁] = Σ_h C̃_h yₜ₋ₕ`, with forecasts substituted for
realisations once the data runs out.

`y_history` is required and must be `N*p × K` in the same layout as the `Y`
passed to [`fit_spvar`](@ref) — the last `p` periods, oldest first, and in
the **original** column order, not the causal one. The model does not
retain the sample, so omitting it is an error rather than a default.

Returns `N*h × K` in that same layout and in **levels**: the
location-specific means absorbed during fitting are added back, which is
what `m.means` is kept for.
"""
function spvar_forecast(m::SpVARModel{T}, h::Int;
                        y_history::Union{Nothing, AbstractMatrix{T}} = nothing
                        ) where {T}
    h >= 1 || throw(ArgumentError("spvar_forecast: h must be ≥ 1, got $h"))
    N, K, p = m.N, m.K, m.p
    y_history === nothing && throw(ArgumentError(
        "spvar_forecast: `y_history` is required — the model does not " *
        "retain y, so the last p = $p period(s) must be supplied as an " *
        "N*p = $(N * p) by K = $K matrix in the same layout as Y."))
    size(y_history) == (N * p, K) || throw(DimensionMismatch(
        "spvar_forecast: y_history is $(size(y_history)), expected " *
        "($(N * p), $K)"))

    m.stationary || @warn "spvar_forecast: the fitted SpVAR is not " *
        "stationary (spectral radius " *
        "$(round(m.spectral_radius, digits = 4))), so these forecasts " *
        "diverge with the horizon."

    n = N * K
    F = lu(m.C0; check = false)
    issuccess(F) || throw(ErrorException("spvar_forecast: C₀ is singular"))
    Ct = [F \ Ch for Ch in m.C]

    # History into causal order, as deviations from the fitted means.
    hist = Vector{Vector{T}}(undef, p)       # hist[1] = most recent
    for lag in 1:p
        t = p - lag + 1                       # row block in y_history
        v = Vector{T}(undef, n)
        for k in 1:K
            src = m.ordering[k]
            for i in 1:N
                v[(k - 1) * N + i] =
                    y_history[(t - 1) * N + i, src] - m.means[i, k]
            end
        end
        hist[lag] = v
    end

    out = Matrix{T}(undef, N * h, K)
    for step in 1:h
        pred = zeros(T, n)
        for lag in 1:p
            pred .+= Ct[lag] * hist[lag]
        end
        for k in 1:K
            dst = m.ordering[k]
            for i in 1:N
                out[(step - 1) * N + i, dst] = pred[(k - 1) * N + i] +
                                               m.means[i, k]
            end
        end
        for lag in p:-1:2
            hist[lag] = hist[lag - 1]
        end
        hist[1] = pred
    end
    return out
end

# ═══════════════════════════════════════════════════════════════════════
# The nested panel VAR (eqs. 15–18) and the test it supports
# ═══════════════════════════════════════════════════════════════════════

"""
    NestedPanelVARModel

A fitted nested panel VAR — Di Giacinto's eqs. (15)–(18), the SpVAR with
every spatial interaction coefficient set to zero, from
[`fit_nested_panel_var`](@ref).

# Fields

- `coefs::Vector{Matrix{T}}` — `coefs[k]` is `N×q` (heterogeneous) or `1×q`
  (homogeneous) of the equation-`k` coefficients: the `r < k`
  contemporaneous terms first, then the `Kp` lag terms.
- `lambda::Matrix{T}` — `N×K` structural error variances.
- `loglik::T`, `loglik_eq::Vector{T}`, `residuals::Matrix{T}`.
- `n_params::Int` — free coefficients, for the degrees of freedom of a
  nested test.
- `homogeneous::Bool`, `ordering::Vector{Int}`, `varnames`.
- `N`, `K`, `p`, `T_periods`, `T_eff`.
"""
struct NestedPanelVARModel{T<:AbstractFloat} <: StatsAPI.StatisticalModel
    coefs::Vector{Matrix{T}}
    lambda::Matrix{T}
    loglik::T
    loglik_eq::Vector{T}
    residuals::Matrix{T}
    n_params::Int
    homogeneous::Bool
    ordering::Vector{Int}
    varnames::Vector{String}
    N::Int
    K::Int
    p::Int
    T_periods::Int
    T_eff::Int
end

StatsAPI.loglikelihood(m::NestedPanelVARModel) = m.loglik
StatsAPI.residuals(m::NestedPanelVARModel) = m.residuals
StatsAPI.nobs(m::NestedPanelVARModel) = m.N * m.T_eff

function Base.show(io::IO, m::NestedPanelVARModel)
    println(io, "Nested panel VAR (no spatial interaction) — ",
        m.homogeneous ? "homogeneous" : "heterogeneous", " coefficients")
    println(io, "────────────────────────────────────────────────────────")
    @printf(io, "  N = %d, K = %d, p = %d, T = %d (%d effective)\n",
        m.N, m.K, m.p, m.T_periods, m.T_eff)
    println(io, "  causal ordering: ", join(m.varnames, " → "))
    @printf(io, "  log-likelihood: %.4f  (%d free coefficients)\n",
        m.loglik, m.n_params)
end

"""
    fit_nested_panel_var(Y, N, T_per; ordering, p = 1, homogeneous = true,
                         varnames = nothing)

Fit Di Giacinto's nested panel VAR, his eqs. (15)–(18): the SpVAR with all
spatial interaction coefficients zero, so every `N×N` block of every `C_h`
is diagonal and the system decomposes into one structural recursive VAR per
location.

Because `M₀,ᵢ` is **unit** lower triangular its determinant is 1, so there
is no Jacobian and no spatial parameter — this is ordinary least squares
(or iterated GLS, with the location-specific variances the paper's `Λ`
allows) equation by equation.

Di Giacinto's point about this model is structural, not computational:
block triangularity of `C₀` **preserves this model's identification even
when the SpVAR itself is under-identified**, which is his stated reason for
preferring it to a strictly triangular `C₀`. That is what makes
[`lr_spatial_interaction`](@ref) well posed.

!!! note "`homogeneous = true` is the comparable case"
    The paper writes eqs. (15)–(18) with location-indexed coefficients
    `φᵢₖᵣ`, matching its spatially heterogeneous SpVAR. [`fit_spvar`](@ref)
    fits the *homogeneous* specification (eq. 8), whose zero-spatial
    restriction gives a panel VAR with coefficients **common** across
    locations. The heterogeneous version has more parameters, not fewer, so
    it is **not** nested in the homogeneous SpVAR and must not be used for
    a likelihood-ratio test against it. `homogeneous = true`, the default,
    is the one that nests.

This is written independently of [`fit_spvar`](@ref) — no shared solver —
precisely so that `fit_spvar(…, s = 0)` can be checked against it.
"""
function fit_nested_panel_var(
    Y::AbstractMatrix{T}, N::Int, T_per::Int;
    ordering::Union{Nothing, AbstractVector} = nothing,
    p::Int = 1,
    homogeneous::Bool = true,
    varnames::Union{Nothing, Vector{String}} = nothing
) where {T<:AbstractFloat}

    K = size(Y, 2)
    p >= 1 || throw(ArgumentError("fit_nested_panel_var: p must be ≥ 1"))
    size(Y, 1) == N * T_per || throw(DimensionMismatch(
        "fit_nested_panel_var: Y has $(size(Y, 1)) rows, expected $(N * T_per)"))
    T_per > p + 1 || throw(ArgumentError(
        "fit_nested_panel_var: need T_per > p + 1"))
    ordering === nothing && throw(ArgumentError(
        "fit_nested_panel_var: `ordering` is required, as in fit_spvar — " *
        "the contemporaneous causal order is the identifying assumption."))

    names0 = varnames === nothing ? ["y$k" for k in 1:K] : copy(varnames)
    ord = eltype(ordering) <: Integer ? collect(Int, ordering) :
          [something(findfirst(==(String(nm)), names0)) for nm in ordering]
    sort(ord) == collect(1:K) || throw(ArgumentError(
        "fit_nested_panel_var: ordering must be a permutation of 1:$K"))

    Ys_all = _split_variables(T.(Y), N, T_per)
    Ys = [copy(Ys_all[ord[k]]) for k in 1:K]
    names = [names0[ord[k]] for k in 1:K]
    for k in 1:K
        _demean_rows!(Ys[k], p + 1, T_per)
    end

    T_eff = T_per - p
    lam = Matrix{T}(undef, N, K)
    resid = Matrix{T}(undef, N * T_eff, K)
    lls = Vector{T}(undef, K)
    coefs = Vector{Matrix{T}}(undef, K)
    nparams = 0

    for k in 1:K
        q = (k - 1) + K * p
        if homogeneous
            # Pooled across locations, variances location-specific.
            yv = Vector{T}(undef, N * T_eff)
            Z = Matrix{T}(undef, N * T_eff, q)
            for t in (p + 1):T_per, i in 1:N
                row = (t - p - 1) * N + i
                yv[row] = Ys[k][i, t]
                c = 0
                for r in 1:(k - 1)
                    c += 1; Z[row, c] = Ys[r][i, t]
                end
                for hh in 1:p, r in 1:K
                    c += 1; Z[row, c] = Ys[r][i, t - hh]
                end
            end
            th, lk, u, _, _ = _igls(yv, Z, N, T_eff)
            coefs[k] = reshape(th, 1, q)
            lam[:, k] .= lk
            resid[:, k] .= u
            nparams += q
        else
            B = Matrix{T}(undef, N, q)
            u = Vector{T}(undef, N * T_eff)
            lk = Vector{T}(undef, N)
            for i in 1:N
                yi = Vector{T}(undef, T_eff)
                Zi = Matrix{T}(undef, T_eff, q)
                for t in (p + 1):T_per
                    rr = t - p
                    yi[rr] = Ys[k][i, t]
                    c = 0
                    for r in 1:(k - 1)
                        c += 1; Zi[rr, c] = Ys[r][i, t]
                    end
                    for hh in 1:p, r in 1:K
                        c += 1; Zi[rr, c] = Ys[r][i, t - hh]
                    end
                end
                bi = q > 0 ? Zi \ yi : T[]
                ri = q > 0 ? yi .- Zi * bi : yi
                B[i, :] .= bi
                lk[i] = sum(abs2, ri) / T_eff
                for t in 1:T_eff
                    u[(t - 1) * N + i] = ri[t]
                end
            end
            coefs[k] = B
            lam[:, k] .= lk
            resid[:, k] .= u
            nparams += N * q
        end
        lls[k] = -(T(T_eff) / 2) * sum(log, view(lam, :, k))
    end

    const_term = -T(N * K * T_eff) / 2 * log(T(2π)) - T(N * K * T_eff) / 2
    ll = sum(lls) + const_term

    return NestedPanelVARModel{T}(coefs, lam, ll, lls, resid, nparams,
                                  homogeneous, ord, names, N, K, p,
                                  T_per, T_eff)
end

"""
    lr_spatial_interaction(m::SpVARModel, Y, W)

Likelihood-ratio test of **no spatial interaction**: `SpVAR(p, s)` against
`SpVAR(p, 0)`, the nested panel VAR of eqs. (15)–(18).

Returns a `NamedTuple` `(stat, df, p_value, loglik_full,
loglik_restricted, s)`.

`Y` and `W` must be the data `m` was fitted to; the model does not retain
them, and passing something else gives a meaningless answer rather than an
error, so this is checked for shape only. The restricted fit re-uses `m`'s
ordering, `p` and variable names.

Setting `s = 0` makes every `W⁽ˡ⁾` block collapse to `W⁽⁰⁾ = I`, so
`A⁽⁰⁾ₖₖ = I`, the Jacobian vanishes and the model *is* the nested panel VAR
in homogeneous form. The two hypotheses are therefore nested by
construction rather than by argument, and the degrees of freedom are
counted from the actual parameter vectors, not from a formula.

!!! warning "Asymptotics again"
    The `χ²` reference distribution is justified by `T → ∞` at fixed `N`.
    The suite measures this statistic's size, and that measurement — not
    the asymptotic claim — is what `docs/AUDIT.md` reports.
"""
function lr_spatial_interaction(m::SpVARModel{T}, Y::AbstractMatrix{T},
                                W::AbstractSparseMatrix{T}) where {T}
    m.s >= 1 || throw(ArgumentError(
        "lr_spatial_interaction: the fitted model already has s = 0, so " *
        "there is no spatial interaction to test."))
    size(Y, 1) == m.N * m.T_periods && size(Y, 2) == m.K || throw(
        DimensionMismatch(
        "lr_spatial_interaction: Y is $(size(Y)), expected " *
        "($(m.N * m.T_periods), $(m.K)) — it must be the data the model " *
        "was fitted to."))

    names_orig = Vector{String}(undef, m.K)
    for k in 1:m.K
        names_orig[m.ordering[k]] = m.varnames[k]
    end

    r0 = fit_spvar(Y, W, m.N, m.T_periods; ordering = m.ordering,
                   p = m.p, s = 0, varnames = names_orig)

    nfull = sum(length(m.rho[k]) + length(m.theta[k]) for k in 1:m.K)
    nrest = sum(length(r0.rho[k]) + length(r0.theta[k]) for k in 1:m.K)
    df = nfull - nrest
    df >= 1 || throw(ErrorException(
        "lr_spatial_interaction: degrees of freedom came out $df, which " *
        "cannot happen for s ≥ 1; report it."))

    stat = 2 * (m.loglik - r0.loglik)
    return (stat = stat, df = df, p_value = _chi2_sf(max(stat, zero(T)), df),
            loglik_full = m.loglik, loglik_restricted = r0.loglik, s = m.s)
end

end # module
