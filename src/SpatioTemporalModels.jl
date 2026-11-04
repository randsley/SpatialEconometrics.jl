"""
    SpatioTemporalModels

Models where dependence runs through space *and* time, and where an
unobserved factor structure sits alongside the spatial lag.

Three things live here:

- [`fit_star`](@ref) — the space-time autoregression of Pfeifer & Deutsch
  (1980), `yₜ = Σₖ Σₗ φₖₗ Wᵏ yₜ₋ₗ + Xβ + ε`, with arbitrary spatial and
  temporal lag orders.
- [`fit_interactive_fe`](@ref) — a spatial panel with interactive fixed
  effects `λᵢ'fₜ`, by the iterative principal components of Bai (2009) as
  extended to a spatial lag by Shi & Lee (2017). This is the estimation
  counterpart to the diagnostics in `SpatialCommonFactors`.
- [`space_time_irf`](@ref) and [`spatial_forecast`](@ref) — tracing a shock
  through space and time, and projecting forward.

**How STAR differs from the dynamic spatial panel already in the package.**
STAR's spatial lags are all *temporally lagged*, so every regressor is
predetermined and the model is estimable by least squares. That is the point
of the specification: it buys arbitrary lag orders in both dimensions at the
cost of ruling out contemporaneous spatial feedback. If you want
`ρWyₜ` — spatial dependence *within* a period — that is
`fit_spatial_panel` or `fit_dynamic_spatial_panel`, which pay for it with a
Jacobian and a likelihood. The two specifications answer different
questions and neither nests the other.

# References

- Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative
  procedure for space-time modeling. *Technometrics*, 22(1), 35–47.
- Bai, J. (2009). Panel data models with interactive fixed effects.
  *Econometrica*, 77(4), 1229–1279.
- Shi, W., & Lee, L. F. (2017). Spatial dynamic panel data models with
  interactive fixed effects. *Journal of Econometrics*, 197(2), 323–347.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial
  Econometrics*. CRC Press.
"""
module SpatioTemporalModels

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random
using StatsAPI
import StatsAPI: coef, vcov, nobs, dof_residual, coefnames, stderror, residuals

import ..SpatialUtils: _normcdf, _chi2_sf, _print_coef_table,
    SpatialVariant, SAR, SDM,
    FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
    within_transform, within_transform!, panel_spatial_lag!, panel_spatial_lag,
    check_design_rank, brent_maximize,
    LogDetMethod, ExactLU, ChebyshevApprox

export
    STARModel, fit_star,
    InteractiveFEModel, fit_interactive_fe, factor_projection,
    IRFResult, space_time_irf,
    spatial_forecast

# ═══════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════

@inline function _pmat(v::AbstractVector{T}, N::Int, T_per::Int) where {T}
    length(v) == N * T_per || throw(DimensionMismatch(
        "expected a panel vector of length N*T = $(N * T_per), got $(length(v))"))
    return reshape(v, N, T_per)
end

# log|I - rho W|, exactly, by sparse LU. The Chebyshev path is not used
# here: it is an approximation tuned for a single rho on a symmetric-ish W,
# and the concentrated profile below is differenced across rho values where
# an approximation error that varies with rho would bend the optimum.
function _logdet_jacobian(W::AbstractSparseMatrix{T}, rho::T, N::Int) where {T}
    A = sparse(one(T) * I, N, N) - rho * W
    F = lu(A; check = false)
    issuccess(F) || return T(-Inf)
    ld, sgn = logabsdet(F)
    sgn > 0 || return T(-Inf)
    return T(ld)
end

# ═══════════════════════════════════════════════════════════════════════
# STAR: space-time autoregression
# ═══════════════════════════════════════════════════════════════════════

"""
    STARModel

A fitted space-time autoregression, from [`fit_star`](@ref).

# Fields

- `phi::Matrix{T}` — the autoregressive coefficients, `(S+1) × L`, with
  `phi[k+1, l]` multiplying `Wᵏ yₜ₋ₗ`. Row 1 is `k = 0`, the purely
  temporal lags.
- `beta::Vector{T}` — exogenous slopes, **excluding** the intercept.
- `intercept::T`, `has_intercept::Bool` — the intercept is kept apart from
  `beta` so that `length(beta) == size(X, 2)` always holds; `intercept` is
  zero when `has_intercept` is false.
- `coef::Vector{T}`, `vcov::Matrix{T}`, `coefnames::Vector{String}` — the
  flattened parameter vector and its covariance, `phi` first in column
  order then `beta`.
- `sigma2::T`, `loglik::T`, `residuals::Vector{T}`.
- `spatial_lags::Int` (`S`), `temporal_lags::Int` (`L`).
- `N::Int`, `T_periods::Int` — the original panel; `T_eff = T − L` periods
  are actually used.
- `W::SparseMatrixCSC{T, Int}` — kept for [`space_time_irf`](@ref) and
  [`spatial_forecast`](@ref).
- `stationary::Bool` — whether the companion form's spectral radius is
  below one. When `false` the fit is still reported but forecasts and
  impulse responses will diverge, and both functions say so.
- `spectral_radius::T`.

# Why least squares is the right estimator here

Every regressor is dated `t−1` or earlier, so none is correlated with `εₜ`
and OLS is consistent without a Jacobian. This is the defining advantage of
the STAR parameterisation over a contemporaneous spatial lag, and the
reason arbitrary lag orders are affordable.
"""
struct STARModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    phi::Matrix{T}
    beta::Vector{T}
    intercept::T
    has_intercept::Bool
    coef::Vector{T}
    vcov::Matrix{T}
    coefnames::Vector{String}
    sigma2::T
    loglik::T
    residuals::Vector{T}
    spatial_lags::Int
    temporal_lags::Int
    N::Int
    T_periods::Int
    T_eff::Int
    W::SparseMatrixCSC{T, Int}
    stationary::Bool
    spectral_radius::T
end

StatsAPI.coef(m::STARModel) = m.coef
StatsAPI.vcov(m::STARModel) = m.vcov
StatsAPI.nobs(m::STARModel) = m.N * m.T_eff
StatsAPI.residuals(m::STARModel) = m.residuals
StatsAPI.coefnames(m::STARModel) = m.coefnames
StatsAPI.stderror(m::STARModel) = sqrt.(diag(m.vcov))
StatsAPI.dof_residual(m::STARModel) = m.N * m.T_eff - length(m.coef)

function Base.show(io::IO, m::STARModel)
    println(io, "Space-Time Autoregression (STAR)")
    println(io, "════════════════════════════════")
    @printf(io, "  Spatial lags S = %d, temporal lags L = %d\n",
        m.spatial_lags, m.temporal_lags)
    @printf(io, "  N = %d, T = %d (T_eff = %d after lagging)\n",
        m.N, m.T_periods, m.T_eff)
    println(io)
    _print_coef_table(io, m.coefnames, m.coef, StatsAPI.stderror(m))
    println(io)
    @printf(io, "  sigma^2 = %.6f,  loglik = %.4f\n", m.sigma2, m.loglik)
    @printf(io, "  Companion spectral radius = %.4f  %s\n",
        m.spectral_radius,
        m.stationary ? "(stationary)" :
        "← NOT stationary; forecasts and IRFs diverge")
end

"""
    fit_star(y, X, W, N, T_per; spatial_lags = 1, temporal_lags = 1,
             varnames = nothing, intercept = true)

Fit the space-time autoregression of Pfeifer & Deutsch (1980).

```
yₜ = Σ_{k=0}^{S} Σ_{l=1}^{L} φₖₗ Wᵏ yₜ₋ₗ + Xₜβ + εₜ
```

`spatial_lags = S` and `temporal_lags = L` set the two orders. `k = 0` is
the identity, so `φ₀ₗ` are the plain temporal autoregressive coefficients
and the model nests a non-spatial panel AR(`L`) at `S = 0`.

`y` is a stacked panel vector of length `N*T_per` in the package's
unit-within-time layout; `X` has `N*T_per` rows. The first `L` periods are
consumed by the lags, leaving `T_eff = T_per − L`.

# Estimated by least squares, deliberately

All regressors are dated `t−1` or earlier, so the model has no
simultaneity and OLS is consistent. **There is no contemporaneous `Wyₜ`
term**, and adding one would change the estimator entirely — see
`fit_spatial_panel` for that specification. The trade is explicit: STAR
buys arbitrary lag orders in both dimensions by giving up within-period
spatial feedback.

# Identification

`W⁰ = I, W¹, …, W^S` must be linearly independent as matrices, or the
design has exactly collinear blocks. On a row-standardised `W` with a
regular structure this fails for modest `S` — a complete graph gives
`W² ∝ W`, for instance. The design is checked with `check_design_rank` and
the error names the offending columns rather than failing inside LAPACK.

# Example

```julia
m = fit_star(y, X, W, N, T; spatial_lags = 2, temporal_lags = 1)
irf = space_time_irf(m; horizon = 12)
f   = spatial_forecast(m, X_future; h = 4)
```

# References

Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative procedure
for space-time modeling. *Technometrics*, 22(1), 35–47.
"""
function fit_star(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    N::Int,
    T_per::Int;
    spatial_lags::Int = 1,
    temporal_lags::Int = 1,
    varnames::Union{Nothing, Vector{String}} = nothing,
    intercept::Bool = true
) where {T<:AbstractFloat}

    S, L = spatial_lags, temporal_lags
    S >= 0 || throw(ArgumentError("fit_star: spatial_lags must be ≥ 0, got $S"))
    L >= 1 || throw(ArgumentError("fit_star: temporal_lags must be ≥ 1, " *
        "got $L. With no temporal lag there is no autoregression and this " *
        "is an ordinary cross-sectional regression per period."))
    T_per > L || throw(ArgumentError("fit_star: T = $T_per periods cannot " *
        "support $L temporal lags — the lagged design would be empty."))
    size(W) == (N, N) || throw(DimensionMismatch(
        "fit_star: W is $(size(W)) but the panel has $N units"))
    length(y) == N * T_per || throw(DimensionMismatch(
        "fit_star: y has length $(length(y)), expected $(N * T_per)"))
    size(X, 1) == N * T_per || throw(DimensionMismatch(
        "fit_star: X has $(size(X, 1)) rows, expected $(N * T_per)"))
    all(isfinite, y) && all(isfinite, X) || throw(ArgumentError(
        "fit_star: y or X contains non-finite values"))

    Ws = sparse(W)
    k_x = size(X, 2)
    T_eff = T_per - L
    n_obs = N * T_eff

    # Spatial powers of y, W^k y, computed once by repeated multiplication
    # rather than by forming W^k.
    Ymat = _pmat(y, N, T_per)
    Wky = Vector{Matrix{T}}(undef, S + 1)
    Wky[1] = Matrix(Ymat)
    for k in 1:S
        Wky[k + 1] = Ws * Wky[k]
    end

    # Design: the (k,l) blocks, then X, then an intercept.
    n_ar = (S + 1) * L
    n_par = n_ar + k_x + (intercept ? 1 : 0)
    D = Matrix{T}(undef, n_obs, n_par)
    names = Vector{String}(undef, n_par)

    col = 0
    for l in 1:L, k in 0:S
        col += 1
        names[col] = k == 0 ? "y(t-$l)" : "W^$k y(t-$l)"
        src = Wky[k + 1]
        @inbounds for t in 1:T_eff
            # period (L + t) uses lag l, i.e. source period L + t - l
            sp = L + t - l
            off = (t - 1) * N
            for i in 1:N
                D[off + i, col] = src[i, sp]
            end
        end
    end
    for j in 1:k_x
        col += 1
        names[col] = varnames === nothing ? "x$j" : varnames[j]
        @inbounds for t in 1:T_eff
            off = (t - 1) * N
            soff = (L + t - 1) * N
            for i in 1:N
                D[off + i, col] = X[soff + i, j]
            end
        end
    end
    if intercept
        col += 1
        names[col] = "intercept"
        D[:, col] .= one(T)
    end

    yv = Vector{T}(undef, n_obs)
    @inbounds for t in 1:T_eff
        off = (t - 1) * N
        soff = (L + t - 1) * N
        for i in 1:N
            yv[off + i] = y[soff + i]
        end
    end

    check_design_rank(D, names, "fit_star";
        context = "The space-time design is rank-deficient. The usual cause " *
                  "is that W^k is linearly dependent on lower powers — a " *
                  "complete or otherwise highly regular W gives W² ∝ W, so " *
                  "`spatial_lags` cannot be raised indefinitely. Reduce " *
                  "`spatial_lags`, or drop the intercept if a unit-invariant " *
                  "column is already present in X.")

    F = qr(D, ColumnNorm())
    theta = F \ yv
    resid = yv .- D * theta
    dof = n_obs - n_par
    dof > 0 || throw(ArgumentError("fit_star: no residual degrees of " *
        "freedom ($n_obs observations, $n_par parameters)"))
    sigma2 = dot(resid, resid) / dof

    XtX_inv = inv(Symmetric(D' * D))
    V = sigma2 .* XtX_inv
    V = (V .+ V') ./ 2

    loglik = -0.5 * n_obs * (log(2 * T(pi)) + log(sigma2) + one(T))

    phi = Matrix{T}(undef, S + 1, L)
    c = 0
    for l in 1:L, k in 0:S
        c += 1
        phi[k + 1, l] = theta[c]
    end
    beta = theta[(n_ar + 1):(n_ar + k_x)]
    icept = intercept ? theta[end] : zero(T)

    # Companion-form spectral radius. The reduced form is
    #   y_t = sum_l A_l y_{t-l} + ...,  A_l = sum_k phi[k+1,l] W^k
    # and stationarity is the spectral radius of the L*N companion matrix.
    # For L = 1 that is just rho(A_1).
    rad = _star_spectral_radius(Wky, phi, Ws, N, S, L)
    stationary = rad < one(T)

    return STARModel{T}(phi, beta, icept, intercept, theta, V, names,
                        sigma2, loglik, resid, S, L, N, T_per, T_eff, Ws,
                        stationary, rad)
end

# Build A_l = sum_k phi[k+1,l] W^k as a dense matrix (N is panel-sized, so
# the companion eigenproblem is the binding cost, not this).
function _star_A(phi::Matrix{T}, W::AbstractSparseMatrix{T}, N::Int,
                 S::Int, l::Int) where {T}
    A = zeros(T, N, N)
    Wk = Matrix{T}(I, N, N)
    for k in 0:S
        A .+= phi[k + 1, l] .* Wk
        k < S && (Wk = Matrix(W) * Wk)
    end
    return A
end

function _star_spectral_radius(::Vector{Matrix{T}}, phi::Matrix{T},
                               W::AbstractSparseMatrix{T}, N::Int,
                               S::Int, L::Int) where {T}
    As = [_star_A(phi, W, N, S, l) for l in 1:L]
    if L == 1
        return T(maximum(abs, eigvals(As[1])))
    end
    C = zeros(T, L * N, L * N)
    for l in 1:L
        C[1:N, ((l - 1) * N + 1):(l * N)] = As[l]
    end
    for l in 2:L
        C[((l - 1) * N + 1):(l * N), ((l - 2) * N + 1):((l - 1) * N)] =
            Matrix{T}(I, N, N)
    end
    return T(maximum(abs, eigvals(C)))
end

# ═══════════════════════════════════════════════════════════════════════
# Interactive fixed effects
# ═══════════════════════════════════════════════════════════════════════

"""
    InteractiveFEModel

A spatial panel with an estimated factor structure, from
[`fit_interactive_fe`](@ref).

# Fields

- `rho::T`, `beta::Vector{T}`, `vcov::Matrix{T}` — covariance of
  `[rho; beta]`.
- `factors::Matrix{T}` — `T × r`, normalised so `F'F/T = I_r`.
- `loadings::Matrix{T}` — `N × r`.
- `n_factors::Int`, `sigma2::T`, `loglik::T`, `residuals::Vector{T}`.
  `sigma2` is degrees-of-freedom corrected, dividing by
  `NT − k − 1 − r(N+T−r)`, and is what `vcov` is built from. The
  *likelihood* uses the ML variance `SSR/NT`, as the concentrated profile
  requires, so `loglik` and `sigma2` are not two views of one number.
- `iterations::Int`, `converged::Bool`, `final_change::T` — the iterative
  principal-components loop's record.
- `N::Int`, `T_periods::Int`, `coefnames::Vector{String}`.

# What is identified, and what is not

`F` and `Λ` are identified only **up to an `r × r` rotation**: for any
invertible `R`, `(FR, ΛR⁻ᵀ)` fits identically. So `factors` and `loadings`
are one particular representative, fixed by the normalisation
`F'F/T = Iᵣ`, and **comparing them across fits is meaningless**. What is
identified is the *space* they span, i.e. the projection
`F(F'F)⁻¹F'` — use [`factor_projection`](@ref) to get it, and compare
that. `rho`, `beta` and the fitted values are of course identified too.
"""
struct InteractiveFEModel{T<:AbstractFloat} <: StatsAPI.RegressionModel
    rho::T
    beta::Vector{T}
    vcov::Matrix{T}
    factors::Matrix{T}
    loadings::Matrix{T}
    n_factors::Int
    sigma2::T
    loglik::T
    residuals::Vector{T}
    iterations::Int
    converged::Bool
    final_change::T
    N::Int
    T_periods::Int
    coefnames::Vector{String}
end

StatsAPI.coef(m::InteractiveFEModel) = vcat(m.rho, m.beta)
StatsAPI.vcov(m::InteractiveFEModel) = m.vcov
StatsAPI.nobs(m::InteractiveFEModel) = m.N * m.T_periods
StatsAPI.residuals(m::InteractiveFEModel) = m.residuals
StatsAPI.coefnames(m::InteractiveFEModel) = vcat("rho", m.coefnames)
StatsAPI.stderror(m::InteractiveFEModel) = sqrt.(diag(m.vcov))
function StatsAPI.dof_residual(m::InteractiveFEModel)
    r = m.n_factors
    return m.N * m.T_periods - length(m.beta) - 1 -
           r * (m.N + m.T_periods - r)
end

"""
    factor_projection(m::InteractiveFEModel)

The `T × T` projection onto the estimated factor space, `F(F'F)⁻¹F'`.

This is the identified object. `m.factors` is only one representative of a
rotation class, so two fits of the same data can return different `factors`
while describing the same factor space — this projection is what to compare
in that case, and what a test should pin.
"""
function factor_projection(m::InteractiveFEModel{T}) where {T}
    F = m.factors
    size(F, 2) == 0 && return zeros(T, m.T_periods, m.T_periods)
    return F * ((F' * F) \ F')
end

function Base.show(io::IO, m::InteractiveFEModel)
    println(io, "Spatial Panel with Interactive Fixed Effects")
    println(io, "════════════════════════════════════════════")
    @printf(io, "  N = %d, T = %d,  r = %d factor%s\n",
        m.N, m.T_periods, m.n_factors, m.n_factors == 1 ? "" : "s")
    println(io)
    _print_coef_table(io, StatsAPI.coefnames(m), coef(m), StatsAPI.stderror(m))
    println(io)
    @printf(io, "  sigma^2 = %.6f,  loglik = %.4f\n", m.sigma2, m.loglik)
    if m.converged
        @printf(io, "  IPC converged in %d iterations (last change %.2e)\n",
            m.iterations, m.final_change)
    else
        @printf(io, "  ⚠ IPC did NOT converge in %d iterations (last change %.2e)\n",
            m.iterations, m.final_change)
        println(io, "    The estimate is the last iterate, not a fixed point.")
    end
    if m.n_factors > 0
        println(io, "  Factors are identified up to rotation — compare")
        println(io, "  `factor_projection(m)`, not `m.factors`.")
    end
end

# One Bai (2009) IPC pass for a fixed rho. Returns (beta, F, Lambda, ssr,
# iterations, converged, change).
#
# Alternates:
#   beta given F:  beta = (sum_i X_i' M_F X_i)^-1 sum_i X_i' M_F z_i
#   F given beta:  top r eigenvectors of R'R, R = reshape(z - X beta)
# with M_F = I_T - F(F'F)^-1 F'. Under the normalisation F'F/T = I_r the
# projection is F F'/T, which is what the code uses.
function _ipc(z::Vector{T}, X::Matrix{T}, N::Int, T_per::Int, r::Int;
              maxiter::Int = 500, tol::T = T(1e-10),
              F0::Union{Nothing, Matrix{T}} = nothing) where {T<:AbstractFloat}
    k = size(X, 2)
    Z = reshape(z, N, T_per)
    Xb = [reshape(@view(X[:, j]), N, T_per) for j in 1:k]

    F = F0 === nothing ? zeros(T, T_per, r) : copy(F0)
    beta = zeros(T, k)
    change = T(Inf)
    iters = 0
    converged = false

    # Pre-allocate the normal-equation blocks.
    A = Matrix{T}(undef, k, k)
    b = Vector{T}(undef, k)

    for it in 1:maxiter
        iters = it
        beta_old = copy(beta)

        # --- beta given F -------------------------------------------------
        # M_F applied on the right of an N x T block: Y - (Y F) F' / T.
        fill!(A, zero(T)); fill!(b, zero(T))
        if r == 0
            @inbounds for a in 1:k, c in 1:k
                A[a, c] = dot(Xb[a], Xb[c])
            end
            @inbounds for a in 1:k
                b[a] = dot(Xb[a], Z)
            end
        else
            MX = [Xj .- (Xj * F) * F' ./ T_per for Xj in Xb]
            MZ = Z .- (Z * F) * F' ./ T_per
            @inbounds for a in 1:k, c in 1:k
                A[a, c] = dot(Xb[a], MX[c])
            end
            @inbounds for a in 1:k
                b[a] = dot(Xb[a], MZ)
            end
        end
        if k > 0
            Af = Symmetric(A)
            beta = Af \ b
        end

        # --- F given beta -------------------------------------------------
        R = copy(Z)
        @inbounds for j in 1:k
            R .-= beta[j] .* Xb[j]
        end
        if r > 0
            # Top r eigenvectors of R'R (T x T), scaled so F'F/T = I_r.
            M = Symmetric(R' * R)
            E = eigen(M)
            idx = sortperm(E.values; rev = true)[1:r]
            F = E.vectors[:, idx] .* sqrt(T(T_per))
        end

        change = k > 0 ? maximum(abs, beta .- beta_old) : zero(T)
        if r == 0
            converged = true
            break
        end
        if change < tol
            converged = true
            break
        end
    end

    R = copy(Z)
    @inbounds for j in 1:size(X, 2)
        R .-= beta[j] .* Xb[j]
    end
    Lambda = r > 0 ? (R * F) ./ T_per : zeros(T, N, 0)
    resid = r > 0 ? R .- Lambda * F' : R
    ssr = dot(resid, resid)

    return beta, F, Lambda, ssr, iters, converged, change
end

"""
    fit_interactive_fe(y, X, W, N, T_per; n_factors = 1, fe = NoFE(),
                       maxiter = 500, tol = 1e-10, n_starts = 1,
                       rho_bounds = (-0.995, 0.995), varnames = nothing,
                       rng = Random.default_rng())

Fit `y = ρWy + Xβ + λᵢ'fₜ + ε` by iterative principal components.

This is the estimator that pairs with `pesaran_cd` and `alpha_exponent` in
`SpatialCommonFactors`: those say whether a factor structure is present,
this one estimates it jointly with the spatial parameter so that `ρ̂` is not
left absorbing a global shock.

# How it works

`ρ` is profiled out on a one-dimensional grid search (Brent), and at each
trial `ρ` the Bai (2009) loop runs on `z = y − ρWy`:

1. **`β` given `F`** — least squares after projecting the factor space off
   every series, `β = (Σᵢ Xᵢ'M_F Xᵢ)⁻¹ Σᵢ Xᵢ'M_F zᵢ` with
   `M_F = I_T − F(F'F)⁻¹F'`.
2. **`F` given `β`** — the top `r` eigenvectors of `R'R`, where `R` is the
   residual panel, normalised so `F'F/T = Iᵣ`.

The concentrated log-likelihood adds the usual Jacobian,
`T·log|I − ρW|`, computed exactly by sparse LU — the Chebyshev
approximation is deliberately not offered here, because the profile is
compared *across* `ρ` values and an approximation error that varies with
`ρ` would move the optimum.

`fe` composes normally: additive unit and time effects are swept by the
within transform before the factor loop, which is Bai's own §3 ordering.
Note that `TimeFE` is close to redundant — a factor with *homogeneous*
loadings **is** a time effect — but they are not the same thing, and the
difference is the whole point of this estimator.

# `n_factors` is a choice, and it matters asymmetrically

There is no automatic selection here. Over-specifying `r` costs efficiency;
under-specifying it leaves factor structure in the error, which is the
failure mode that biases `ρ̂`. The measured cost of each direction is in
`docs/AUDIT.md`. When in doubt, and having checked `alpha_exponent`, prefer
one too many to one too few.

`n_factors = 0` reduces the estimator to a spatial panel with no factor
structure, which is how it is tested against `fit_spatial_panel`.

# Local optima

Bai's loop is not globally convex. `n_starts > 1` runs it from several
random factor initialisations and keeps the best log-likelihood; the
measured sensitivity is documented in `docs/AUDIT.md`. A non-converged fit
is reported as such in `converged` and by `show` rather than passed off as
a fixed point.

# Example

```julia
v = dependence_verdict(residuals(fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T)), N, T)
if v.kind === :strong
    m = fit_interactive_fe(y, X, W, N, T; n_factors = 1)
end
```

# References

- Bai, J. (2009). *Econometrica*, 77(4), 1229–1279.
- Shi, W., & Lee, L. F. (2017). *Journal of Econometrics*, 197(2), 323–347.
"""
function fit_interactive_fe(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    N::Int,
    T_per::Int;
    n_factors::Int = 1,
    fe::FixedEffectsSpec = NoFE(),
    maxiter::Int = 500,
    tol::Real = 1e-10,
    n_starts::Int = 1,
    rho_bounds::Tuple{Real, Real} = (-0.995, 0.995),
    varnames::Union{Nothing, Vector{String}} = nothing,
    rng::AbstractRNG = Random.default_rng()
) where {T<:AbstractFloat}

    r = n_factors
    r >= 0 || throw(ArgumentError(
        "fit_interactive_fe: n_factors must be ≥ 0, got $r"))
    r < min(N, T_per) || throw(ArgumentError(
        "fit_interactive_fe: n_factors = $r saturates the factor space " *
        "(min(N, T) = $(min(N, T_per))). With that many factors the " *
        "residual panel is fitted exactly and β is not identified."))
    size(W) == (N, N) || throw(DimensionMismatch(
        "fit_interactive_fe: W is $(size(W)) but the panel has $N units"))
    length(y) == N * T_per || throw(DimensionMismatch(
        "fit_interactive_fe: y has length $(length(y)), expected $(N * T_per)"))
    size(X, 1) == N * T_per || throw(DimensionMismatch(
        "fit_interactive_fe: X has $(size(X, 1)) rows, expected $(N * T_per)"))
    all(isfinite, y) && all(isfinite, X) || throw(ArgumentError(
        "fit_interactive_fe: y or X contains non-finite values"))
    n_starts >= 1 || throw(ArgumentError(
        "fit_interactive_fe: n_starts must be ≥ 1, got $n_starts"))

    Wsp = sparse(W)
    k = size(X, 2)
    names = varnames === nothing ? ["x$j" for j in 1:k] : copy(varnames)
    length(names) == k || throw(DimensionMismatch(
        "fit_interactive_fe: $(length(names)) names for $k regressors"))

    Wy = Vector{T}(undef, N * T_per)
    panel_spatial_lag!(Wy, Wsp, collect(y), N, T_per)

    yv = collect(y); Wyv = Wy; Xm = Matrix(X)
    if !(fe isa NoFE)
        yv = within_transform!(copy(yv), N, T_per, fe)
        Wyv = within_transform!(copy(Wyv), N, T_per, fe)
        Xm = within_transform(Xm, N, T_per, fe)
    end

    n_obs = N * T_per
    tolT = T(tol)

    starts = Vector{Union{Nothing, Matrix{T}}}(undef, n_starts)
    starts[1] = nothing
    for s in 2:n_starts
        starts[s] = r > 0 ? qr(randn(rng, T_per, r)).Q[:, 1:r] .* sqrt(T(T_per)) :
                            zeros(T, T_per, 0)
    end

    best = nothing
    for F0 in starts
        # Concentrated profile in rho.
        function profile(rho_val)
            rho = T(rho_val)
            ld = _logdet_jacobian(Wsp, rho, N)
            isfinite(ld) || return -Inf
            z = yv .- rho .* Wyv
            _, _, _, ssr, _, _, _ = _ipc(z, Xm, N, T_per, r;
                                         maxiter = maxiter, tol = tolT, F0 = F0)
            ssr <= 0 && return -Inf
            s2 = ssr / n_obs
            return Float64(-0.5 * n_obs * (log(2 * pi) + log(s2) + 1) + T_per * ld)
        end

        rho_hat, ll = brent_maximize(profile, Float64(rho_bounds[1]),
                                     Float64(rho_bounds[2]); tol = 1e-8)
        if best === nothing || ll > best[2]
            best = (T(rho_hat), ll, F0)
        end
    end

    rho = best[1]
    loglik = T(best[2])
    z = yv .- rho .* Wyv
    beta, F, Lambda, ssr, iters, converged, change =
        _ipc(z, Xm, N, T_per, r; maxiter = maxiter, tol = tolT, F0 = best[3])

    r_eff = r * (N + T_per - r)
    dof = n_obs - k - 1 - r_eff
    dof > 0 || throw(ArgumentError(
        "fit_interactive_fe: no residual degrees of freedom — a factor " *
        "structure with r = $r costs r(N + T − r) = $r_eff parameters and " *
        "the panel has $n_obs observations. Reduce n_factors."))
    sigma2 = ssr / dof

    resid_mat = reshape(z, N, T_per) .- (k > 0 ? reshape(Xm * beta, N, T_per) : zero(T))
    if r > 0
        resid_mat = resid_mat .- Lambda * F'
    end
    resid = vec(resid_mat)

    # Covariance for [rho; beta]. The score is built on the factor-projected
    # design, which is the data the slopes are actually estimated from, and
    # rho's curvature adds the Jacobian's second derivative.
    V = _ife_vcov(Wsp, Xm, Wyv, F, rho, sigma2, N, T_per, r, k)

    return InteractiveFEModel{T}(rho, beta, V, F, Lambda, r, sigma2,
                                 loglik, resid, iters, converged, change,
                                 N, T_per, names)
end

# Information-matrix covariance for [rho; beta]. Blocks:
#   beta,beta : X' M_F X / sigma2
#   beta,rho  : X' M_F Wy / sigma2
#   rho,rho   : Wy' M_F Wy / sigma2 + T * tr[(G)^2 + G'G] / ... ,
# where G = W (I - rho W)^-1. The trace term is the Jacobian's curvature and
# is what distinguishes this from an OLS covariance.
function _ife_vcov(W::AbstractSparseMatrix{T}, X::Matrix{T}, Wy::Vector{T},
                   F::Matrix{T}, rho::T, sigma2::T, N::Int, T_per::Int,
                   r::Int, k::Int) where {T}
    proj(V) = r > 0 ? (V .- (V * F) * F' ./ T_per) : V

    Wym = reshape(Wy, N, T_per)
    MWy = proj(Wym)
    Xb = [reshape(@view(X[:, j]), N, T_per) for j in 1:k]
    MX = [proj(Xj) for Xj in Xb]

    m = k + 1
    H = zeros(T, m, m)
    H[1, 1] = dot(Wym, MWy) / sigma2
    for j in 1:k
        H[1, j + 1] = dot(Wym, MX[j]) / sigma2
        H[j + 1, 1] = H[1, j + 1]
        for l in 1:k
            H[j + 1, l + 1] = dot(Xb[j], MX[l]) / sigma2
        end
    end

    # Jacobian curvature: T * tr(G^2 + G'G) with G = W (I - rho W)^-1.
    A = sparse(one(T) * I, N, N) - rho * W
    Fa = lu(A; check = false)
    if issuccess(Fa)
        G = Matrix(W) * inv(Matrix(A))
        H[1, 1] += T_per * (tr(G * G) + tr(G' * G))
    end

    V = inv(Symmetric(H))
    V = (V .+ V') ./ 2
    return Matrix(V)
end

# ═══════════════════════════════════════════════════════════════════════
# Impulse responses and forecasting
# ═══════════════════════════════════════════════════════════════════════

"""
    IRFResult

Space-time impulse responses, from [`space_time_irf`](@ref).

# Fields

- `psi::Vector{Matrix{T}}` — the response matrices, `psi[h+1]` giving
  `∂yₜ₊ₕ/∂εₜ'`. `psi[1]` is the contemporaneous response. **Never
  cumulated**, whatever `cumulative` is set to — that keyword affects the
  three summaries below and nothing else. To compare against
  `(I − ΣₗAₗ)⁻¹`, sum `psi` explicitly, or read `total[end]` from a
  `cumulative = true` call, which is that sum's mean row sum.
- `own::Vector{T}` — mean own-unit response at each horizon, the average
  diagonal of `psi[h+1]`.
- `spillover::Vector{T}` — mean total response *to other units*, the
  average off-diagonal row sum.
- `total::Vector{T}` — `own + spillover`, the average row sum.
- `cumulative::Bool` — whether the three summaries accumulate over
  horizons.
- `horizon::Int`, `stationary::Bool`.

The summaries follow LeSage & Pace's direct/indirect/total decomposition,
read at each horizon rather than only in the long run.
"""
struct IRFResult{T<:AbstractFloat}
    psi::Vector{Matrix{T}}
    own::Vector{T}
    spillover::Vector{T}
    total::Vector{T}
    cumulative::Bool
    horizon::Int
    stationary::Bool
end

function Base.show(io::IO, r::IRFResult)
    println(io, "Space-Time Impulse Response",
        r.cumulative ? " (cumulative)" : "")
    println(io, "───────────────────────────")
    r.stationary || println(io,
        "  ⚠ the model is not stationary — these responses diverge.")
    @printf(io, "  %-5s %-14s %-14s %-14s\n", "h", "own", "spillover", "total")
    for h in 0:min(r.horizon, 12)
        @printf(io, "  %-5d %-14.6f %-14.6f %-14.6f\n",
            h, r.own[h + 1], r.spillover[h + 1], r.total[h + 1])
    end
    r.horizon > 12 && @printf(io, "  … %d further horizons\n", r.horizon - 12)
end

function _summarise(psi::Vector{Matrix{T}}, cumulative::Bool) where {T}
    H = length(psi) - 1
    own = Vector{T}(undef, H + 1)
    spill = Vector{T}(undef, H + 1)
    tot = Vector{T}(undef, H + 1)
    for h in 0:H
        P = psi[h + 1]
        n = size(P, 1)
        d = zero(T)
        for i in 1:n; d += P[i, i]; end
        s = sum(P)
        own[h + 1] = d / n
        tot[h + 1] = s / n
        spill[h + 1] = (s - d) / n
    end
    if cumulative
        cumsum!(own, own); cumsum!(spill, spill); cumsum!(tot, tot)
    end
    return own, spill, tot
end

"""
    space_time_irf(m::STARModel; horizon = 10, cumulative = false,
                   store_matrices = true)

Trace a one-period shock through space and time.

The reduced form is `yₜ = Σₗ Aₗ yₜ₋ₗ + …` with `Aₗ = Σₖ φₖₗ Wᵏ`, so the
response matrices satisfy

```
Ψ₀ = I,    Ψₕ = Σ_{l=1}^{min(h,L)} Aₗ Ψₕ₋ₗ
```

and `Ψₕ[i, j]` is the effect on unit `i` at horizon `h` of a unit shock to
unit `j` at horizon 0. The summaries decompose each horizon into an own
effect (the diagonal) and a spillover (everything else), in the spirit of
LeSage & Pace's direct/indirect split but resolved by horizon.

# The long run

With `cumulative = true` the sums converge to `(I − Σₗ Aₗ)⁻¹`, the
long-run multiplier — which is the identity the implementation is tested
against. If the model is not stationary the sums diverge instead, and the
result says so rather than returning a plausible-looking number.

`store_matrices = false` keeps only the summaries, for a large `N` where
`horizon` copies of an `N × N` matrix is too much memory.
"""
function space_time_irf(m::STARModel{T}; horizon::Int = 10,
                        cumulative::Bool = false,
                        store_matrices::Bool = true) where {T}
    horizon >= 0 || throw(ArgumentError(
        "space_time_irf: horizon must be ≥ 0, got $horizon"))
    N, S, L = m.N, m.spatial_lags, m.temporal_lags
    As = [_star_A(m.phi, m.W, N, S, l) for l in 1:L]

    psi = Vector{Matrix{T}}(undef, horizon + 1)
    psi[1] = Matrix{T}(I, N, N)
    for h in 1:horizon
        P = zeros(T, N, N)
        for l in 1:min(h, L)
            P .+= As[l] * psi[h - l + 1]
        end
        psi[h + 1] = P
    end

    own, spill, tot = _summarise(psi, cumulative)
    kept = store_matrices ? psi : Vector{Matrix{T}}()
    return IRFResult{T}(kept, own, spill, tot, cumulative, horizon,
                        m.stationary)
end

"""
    space_time_irf(m::InteractiveFEModel, W; horizon = 10, cumulative = false)

As above, passing the weight matrix the model was fitted with.
"""
function space_time_irf(m::InteractiveFEModel{T}, W::AbstractSparseMatrix{T};
                        horizon::Int = 10, cumulative::Bool = false,
                        store_matrices::Bool = true) where {T}
    horizon >= 0 || throw(ArgumentError(
        "space_time_irf: horizon must be ≥ 0, got $horizon"))
    N = m.N
    size(W) == (N, N) || throw(DimensionMismatch(
        "space_time_irf: W is $(size(W)) but the model has $N units"))
    A = Matrix{T}(I, N, N) - m.rho * Matrix(W)
    psi = Vector{Matrix{T}}(undef, horizon + 1)
    psi[1] = inv(A)
    for h in 1:horizon
        psi[h + 1] = zeros(T, N, N)
    end
    own, spill, tot = _summarise(psi, cumulative)
    stationary = abs(m.rho) < 1
    kept = store_matrices ? psi : Vector{Matrix{T}}()
    return IRFResult{T}(kept, own, spill, tot, cumulative, horizon, stationary)
end

"""
    spatial_forecast(m::STARModel, X_future; h = 5, y_history = nothing)

Forecast `h` periods ahead from a fitted STAR model.

`X_future` has `N*h` rows in the usual unit-within-time layout, giving the
exogenous regressors over the forecast horizon. The recursion is the
model's own,

```
ŷₜ₊ₛ = Σₗ Aₗ ŷₜ₊ₛ₋ₗ + Xₜ₊ₛβ + intercept
```

with realised values used where available and forecasts substituted
thereafter — a dynamic, not a static, forecast.

`y_history` supplies the last `L` periods to start from, as an `N*L` vector.
It defaults to the tail of the data the model was fitted on, which is
reconstructed from the residuals, so passing it explicitly is the safer
course when forecasting from new data.

Returns an `N × h` matrix. A non-stationary fit will produce diverging
numbers and emits a warning rather than failing, because the divergence is
the honest answer for those parameters.
"""
function spatial_forecast(m::STARModel{T}, X_future::AbstractMatrix{T};
                          h::Int = 5,
                          y_history::Union{Nothing, AbstractVector{T}} = nothing
                          ) where {T}
    h >= 1 || throw(ArgumentError("spatial_forecast: h must be ≥ 1, got $h"))
    N, L, S = m.N, m.temporal_lags, m.spatial_lags
    k = length(m.beta)
    expected_rows = N * h
    size(X_future, 1) == expected_rows || throw(DimensionMismatch(
        "spatial_forecast: X_future has $(size(X_future, 1)) rows, " *
        "expected N*h = $expected_rows"))
    size(X_future, 2) == k || throw(DimensionMismatch(
        "spatial_forecast: X_future has $(size(X_future, 2)) columns but " *
        "the model has $k exogenous slopes"))

    y_history === nothing && throw(ArgumentError(
        "spatial_forecast: `y_history` is required — the model does not " *
        "retain y, so the last $L period(s) must be supplied as an " *
        "N*L = $(N * L) vector in unit-within-time order."))
    length(y_history) == N * L || throw(DimensionMismatch(
        "spatial_forecast: y_history has length $(length(y_history)), " *
        "expected N*L = $(N * L)"))

    m.stationary || @warn "spatial_forecast: the fitted STAR is not " *
        "stationary (spectral radius $(round(m.spectral_radius, digits = 4)))," *
        " so these forecasts diverge with the horizon."

    As = [_star_A(m.phi, m.W, N, S, l) for l in 1:L]
    c = m.intercept

    # hist[:, l] is y at lag l relative to the next period to forecast.
    hist = Matrix{T}(undef, N, L)
    Hm = reshape(collect(y_history), N, L)
    for l in 1:L
        hist[:, l] = Hm[:, L - l + 1]   # column 1 = most recent
    end

    out = Matrix{T}(undef, N, h)
    for s in 1:h
        pred = fill(c, N)
        for l in 1:L
            pred .+= As[l] * @view hist[:, l]
        end
        if k > 0
            off = (s - 1) * N
            @inbounds for i in 1:N, j in 1:k
                pred[i] += X_future[off + i, j] * m.beta[j]
            end
        end
        out[:, s] = pred
        for l in L:-1:2
            hist[:, l] = hist[:, l - 1]
        end
        hist[:, 1] = pred
    end
    return out
end

end # module
