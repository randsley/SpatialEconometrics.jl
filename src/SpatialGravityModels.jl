module SpatialGravityModels

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI

import ..SpatialUtils: _normcdf, _print_coef_table

export GravityVariant, PoissonGravity, NegBinGravity,
       GravityFE, OriginFE, DestinationFE, BilateralFE,
       GravityModel, fit_gravity, multilateral_resistance

# ─── Types ────────────────────────────────────────────────────────────

"""
    GravityVariant

Which count model a gravity equation is estimated with. Subtypes:
[`PoissonGravity`](@ref) and [`NegBinGravity`](@ref).
"""
abstract type GravityVariant end
"""
    PoissonGravity()

Poisson pseudo-maximum likelihood (PPML) estimation of a gravity equation.

The standard choice, for two reasons. Trade and migration flow data are
full of zeros, which log-linear estimation simply drops; and under
heteroskedasticity — the norm for flows — log-linear OLS is inconsistent
because `E[log y] != log E[y]`. PPML is consistent under either, needing
only the conditional mean to be correctly specified, and handles zeros
natively.

See also [`NegBinGravity`](@ref), [`fit_gravity`](@ref).

# References
Santos Silva, J. M. C. & Tenreyro, S. (2006). The log of gravity. *Review
of Economics and Statistics*, 88(4).
"""
struct PoissonGravity <: GravityVariant end
"""
    NegBinGravity()

Negative binomial estimation of a gravity equation, adding a dispersion
parameter `alpha` to accommodate variance above the Poisson mean.

More efficient than [`PoissonGravity`](@ref) when the overdispersion is
real, but it buys that by assuming the variance function, so it is less
robust to getting that assumption wrong. Fitted starting from the Poisson
estimates. The estimated `alpha` is on the fitted model; near zero means
the Poisson restriction was adequate.

See also [`fit_gravity`](@ref).
"""
struct NegBinGravity  <: GravityVariant end

"""
    GravityFE

Which fixed effects a gravity equation absorbs. Subtypes:
[`OriginFE`](@ref), [`DestinationFE`](@ref), [`BilateralFE`](@ref).
"""
abstract type GravityFE end
"""
    OriginFE()

Origin fixed effects only — one per sending unit.

Absorbs everything about the origin, including its outward multilateral
resistance, but leaves destination characteristics in the error.

See also [`BilateralFE`](@ref).
"""
struct OriginFE      <: GravityFE end
"""
    DestinationFE()

Destination fixed effects only — one per receiving unit.

The mirror of [`OriginFE`](@ref).
"""
struct DestinationFE <: GravityFE end
"""
    BilateralFE()

Both origin and destination fixed effects. The default.

This is the theory-consistent specification: structural gravity implies
origin and destination terms that absorb both multilateral resistances, and
omitting them is the classic gold-medal error that biases the distance and
border coefficients. They are swept out by alternating projection, so the
cost stays manageable as the number of units grows.

Recover the absorbed terms with [`multilateral_resistance`](@ref).

# References
Anderson, J. & van Wincoop, E. (2003). Gravity with gravitas. *American
Economic Review*, 93(1).
"""
struct BilateralFE   <: GravityFE end

"""
    GravityModel

A fitted gravity / flow model, returned by [`fit_gravity`](@ref).
Implements the `StatsAPI.RegressionModel` interface.

# Fields
- `variant`: [`PoissonGravity`](@ref) or [`NegBinGravity`](@ref).
- `coef`, `vcov`, `coefnames`: the structural coefficients on `X`. The
  fixed effects are **not** in here.
- `origin_fe`, `dest_fe`: the absorbed fixed effects, i.e. the multilateral
  resistance terms — see [`multilateral_resistance`](@ref).
- `alpha`: negative binomial dispersion; `NaN` for Poisson.
- `loglik`, `deviance`, `pseudo_R2`: fit measures.
- `residuals`, `fitted`: on the scale of the flows, not logs.
- `N_obs`, `N_origin`, `N_dest`: observation and unit counts.
- `converged`, `iterations`: state of the IRLS loop. **Check `converged`.**

!!! note
    Coefficients are semi-elasticities on the flow scale, since the model
    is `E[y] = exp(X*beta + FE)`. A log-distance coefficient is therefore
    read as an elasticity directly.
"""
struct GravityModel{V<:GravityVariant, T<:AbstractFloat} <: StatsAPI.RegressionModel
    variant::V
    coef::Vector{T}
    vcov::Matrix{T}
    origin_fe::Vector{T}
    dest_fe::Vector{T}
    alpha::T              # NB dispersion (NaN for Poisson)
    loglik::T
    deviance::T
    pseudo_R2::T
    residuals::Vector{T}
    fitted::Vector{T}
    N_obs::Int
    N_origin::Int
    N_dest::Int
    coefnames::Vector{String}
    converged::Bool
    iterations::Int
end

# ─── High-dimensional FE sweeping (alternating projection) ───────────
#
# The within transformation has to be applied to the response and to every
# regressor with the *same* operator, or the weighted least squares below
# regresses one projection onto another. Both of the sweeps that used to live
# here got that wrong in different ways: under `BilateralFE` the response was
# swept with a single pass of independent origin and destination means -- not
# the two-way projection, and double-counting the overall mean -- while the
# regressors were swept by alternating projection; under one-way FE the
# regressors were not swept at all, which is unbiased only when they happen to
# be orthogonal to the effects.

"""
    _demean_oneway!(z, ids, N_g, w)

Subtract the weighted group mean of `z` within each level of `ids`, in place.
"""
function _demean_oneway!(z::AbstractVector{T}, ids::Vector{Int}, N_g::Int,
                         w::AbstractVector{T}) where {T}
    num = zeros(T, N_g)
    den = zeros(T, N_g)
    @inbounds for n in eachindex(z)
        num[ids[n]] += w[n] * z[n]
        den[ids[n]] += w[n]
    end
    @inbounds for g in 1:N_g
        den[g] > zero(T) && (num[g] /= den[g])
    end
    @inbounds for n in eachindex(z)
        z[n] -= num[ids[n]]
    end
    return z
end

"""
    _demean_bilateral!(z, origin_id, dest_id, N_orig, N_dest, w; max_iter, tol)

Sweep out both origin and destination effects by alternating projection.

Two one-way projections do not commute, so the two-way within transformation
is their limit rather than their sum, which is why this iterates.
"""
function _demean_bilateral!(z::AbstractVector{T}, origin_id::Vector{Int}, dest_id::Vector{Int},
                            N_orig::Int, N_dest::Int, w::AbstractVector{T};
                            max_iter::Int=1000, tol::T=T(1e-12)) where {T}
    N = length(z)
    orig_sum = zeros(T, N_orig); orig_w = zeros(T, N_orig)
    dest_sum = zeros(T, N_dest); dest_w = zeros(T, N_dest)

    for _ in 1:max_iter
        change = zero(T)

        fill!(orig_sum, zero(T)); fill!(orig_w, zero(T))
        @inbounds for n in 1:N
            orig_sum[origin_id[n]] += w[n] * z[n]
            orig_w[origin_id[n]] += w[n]
        end
        @inbounds for n in 1:N
            ow = orig_w[origin_id[n]]
            if ow > zero(T)
                adj = orig_sum[origin_id[n]] / ow
                z[n] -= adj
                change = max(change, abs(adj))
            end
        end

        fill!(dest_sum, zero(T)); fill!(dest_w, zero(T))
        @inbounds for n in 1:N
            dest_sum[dest_id[n]] += w[n] * z[n]
            dest_w[dest_id[n]] += w[n]
        end
        @inbounds for n in 1:N
            dw = dest_w[dest_id[n]]
            if dw > zero(T)
                adj = dest_sum[dest_id[n]] / dw
                z[n] -= adj
                change = max(change, abs(adj))
            end
        end

        change < tol && break
    end
    return z
end

"""
    _demean_fe!(z, fe, o_ids, d_ids, N_orig, N_dest, w)

Apply the within transformation implied by `fe`, in place. One entry point, so
the response and the regressors cannot drift onto different projections.
"""
function _demean_fe!(z::AbstractVector{T}, fe::GravityFE, o_ids::Vector{Int},
                     d_ids::Vector{Int}, N_orig::Int, N_dest::Int,
                     w::AbstractVector{T}) where {T}
    if fe isa BilateralFE
        _demean_bilateral!(z, o_ids, d_ids, N_orig, N_dest, w)
    elseif fe isa OriginFE
        _demean_oneway!(z, o_ids, N_orig, w)
    elseif fe isa DestinationFE
        _demean_oneway!(z, d_ids, N_dest, w)
    end
    return z
end

"""
    _solve_fe!(origin_fe, dest_fe, r, fe, o_ids, d_ids, N_orig, N_dest, w; max_iter, tol)

Split `r` into origin and destination effects, so that
`r[n] - origin_fe[o] - dest_fe[d]` is swept clean.

Accumulating the alternating-projection adjustments is what makes this the
inverse of [`_demean_fe!`](@ref). Taking one weighted mean of `r` per origin
and another per destination -- as the previous version did -- is not: the two
are measured from the same uncorrected `r`, so their sum counts the overall
level twice.

Two-way effects are identified only up to a constant moved from one side to
the other. Starting from zero fixes that: the origin effects carry the level.
Effects not in the model are left at zero rather than filled with a number
that plays no part in the fit.
"""
function _solve_fe!(origin_fe::Vector{T}, dest_fe::Vector{T}, r::AbstractVector{T},
                    fe::GravityFE, o_ids::Vector{Int}, d_ids::Vector{Int},
                    N_orig::Int, N_dest::Int, w::AbstractVector{T};
                    max_iter::Int=1000, tol::T=T(1e-12)) where {T}
    fill!(origin_fe, zero(T))
    fill!(dest_fe, zero(T))
    use_o = fe isa BilateralFE || fe isa OriginFE
    use_d = fe isa BilateralFE || fe isa DestinationFE
    (use_o || use_d) || return origin_fe, dest_fe

    onum = zeros(T, N_orig); oden = zeros(T, N_orig)
    dnum = zeros(T, N_dest); dden = zeros(T, N_dest)

    for _ in 1:max_iter
        change = zero(T)

        if use_o
            fill!(onum, zero(T)); fill!(oden, zero(T))
            @inbounds for n in eachindex(r)
                e = r[n] - origin_fe[o_ids[n]] - dest_fe[d_ids[n]]
                onum[o_ids[n]] += w[n] * e
                oden[o_ids[n]] += w[n]
            end
            @inbounds for i in 1:N_orig
                if oden[i] > zero(T)
                    a = onum[i] / oden[i]
                    origin_fe[i] += a
                    change = max(change, abs(a))
                end
            end
        end

        if use_d
            fill!(dnum, zero(T)); fill!(dden, zero(T))
            @inbounds for n in eachindex(r)
                e = r[n] - origin_fe[o_ids[n]] - dest_fe[d_ids[n]]
                dnum[d_ids[n]] += w[n] * e
                dden[d_ids[n]] += w[n]
            end
            @inbounds for j in 1:N_dest
                if dden[j] > zero(T)
                    b = dnum[j] / dden[j]
                    dest_fe[j] += b
                    change = max(change, abs(b))
                end
            end
        end

        change < tol && break
    end
    return origin_fe, dest_fe
end

# ─── PPML via IRLS ───────────────────────────────────────────────────

"""
    fit_gravity(variant, flows, X, origin_id, dest_id; fe = BilateralFE(),
                varnames = nothing, maxiter = 100, tol = 1e-8)

Fit a gravity / flow model by pseudo-maximum likelihood, absorbing
high-dimensional origin and destination fixed effects.

Estimates `E[flows] = exp(X*beta + origin FE + destination FE)` by IRLS,
with the fixed effects swept out by alternating projection rather than
expanded into dummies, so the number of units is not a constraint.

# Arguments
- `variant`: [`PoissonGravity`](@ref) or [`NegBinGravity`](@ref).
- `flows`: the dependent flow variable, **on its own scale, not logged**.
  Zeros are fine and should be kept.
- `X`: `N_obs x K` bilateral covariates — log distance, contiguity, common
  language and so on. No intercept, and nothing that varies only by origin
  or only by destination, since the fixed effects absorb that.
- `origin_id`, `dest_id`: integer unit indices per observation.

# Keywords
- `fe`: a [`GravityFE`](@ref); [`BilateralFE`](@ref) by default and
  generally what theory calls for.
- `varnames`: covariate names; defaults to `x1, x2, ...`.
- `maxiter`, `tol`: IRLS iteration cap and convergence tolerance.

# Returns
A [`GravityModel`](@ref).

# Examples
```julia
m = fit_gravity(PoissonGravity(), flows, X, orig, dest;
                varnames = ["log_dist", "contiguous"])
m.converged || @warn "IRLS did not converge"
coef(m)
multilateral_resistance(m)
```

!!! warning
    Do not log the flows and do not drop the zeros. Doing either is the
    practice PPML exists to replace.

# References
Santos Silva, J. M. C. & Tenreyro, S. (2006). *Review of Economics and
Statistics*, 88(4).
"""
function fit_gravity(
    ::PoissonGravity,
    flows::AbstractVector,
    X::AbstractMatrix{T},
    origin_id::AbstractVector{<:Integer},
    dest_id::AbstractVector{<:Integer};
    fe::GravityFE=BilateralFE(),
    varnames::Union{Nothing, Vector{String}}=nothing,
    maxiter::Int=100,
    tol::T=T(1e-8)
) where {T<:AbstractFloat}

    N_obs = length(flows)
    K = size(X, 2)
    y = Vector{T}(flows)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    o_ids = Vector{Int}(origin_id)
    d_ids = Vector{Int}(dest_id)
    N_orig = maximum(o_ids)
    N_dest = maximum(d_ids)

    β, μ, origin_fe_vec, dest_fe_vec, X_dm, w, converged, iter =
        _gravity_irls(y, Matrix{T}(X), o_ids, d_ids, fe, N_orig, N_dest;
                      alpha = zero(T), maxiter = maxiter, tol = tol)

    loglik = zero(T)
    @inbounds for n in 1:N_obs
        μn = max(μ[n], T(1e-30))
        loglik += y[n] * log(μn) - μn - _logfactorial(y[n])
    end

    ȳ = mean(y)
    null_loglik = zero(T)
    @inbounds for n in 1:N_obs
        null_loglik += y[n] * log(max(ȳ, T(1e-30))) - ȳ - _logfactorial(y[n])
    end

    deviance = _poisson_deviance(y, μ)
    pseudo_R2 = one(T) - loglik / min(null_loglik, -T(1e-12))
    residuals = y .- μ
    vcov_mat = _sandwich_vcov(X_dm, residuals, w)

    return GravityModel{PoissonGravity, T}(
        PoissonGravity(), β, vcov_mat,
        origin_fe_vec, dest_fe_vec, T(NaN),
        loglik, deviance, pseudo_R2,
        residuals, μ,
        N_obs, N_orig, N_dest, vnames,
        converged, iter
    )
end

function _logfactorial(x::T) where {T<:AbstractFloat}
    n = round(Int, x)
    n <= 1 && return zero(T)
    return T(sum(log(k) for k in 2:n))
end

function _loggamma(x::T) where {T<:AbstractFloat}
    # Lanczos approximation for log Γ(x)
    x <= zero(T) && return T(Inf)
    if x < T(0.5)
        return log(T(π) / sin(T(π) * x)) - _loggamma(one(T) - x)
    end
    x -= one(T)
    g = T(7)
    c = T[0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    s = c[1]
    for i in 2:9
        s += c[i] / (x + T(i - 1))
    end
    t = x + g + T(0.5)
    return T(0.5) * log(T(2π)) + (x + T(0.5)) * log(t) - t + log(s)
end

# ─── Shared IRLS core ────────────────────────────────────────────────

"""
    _gravity_irls(y, X, o_ids, d_ids, fe, N_orig, N_dest; alpha, maxiter, tol)

IRLS for a log-link count model with the fixed effects swept out.

The working response and weights are those of any GLM with
`E[y] = exp(eta)`: `z = eta + (y - mu)/mu` and `w = mu^2 / V(mu)`. For Poisson
`V = mu`, so `w = mu`; for NB2 `V = mu + alpha*mu^2`, so `w = mu/(1 + alpha*mu)`.
`alpha = 0` therefore reproduces PPML exactly, which is why both variants come
through here -- the negative binomial differs from Poisson in the weights and
nowhere else.

Returns `(beta, mu, origin_fe, dest_fe, X_dm, w, converged, iter)`. `X_dm` is
the final within-transformed design, which is the one the variance has to be
built from; using the raw `X` there ignores the fixed effects entirely.
"""
function _gravity_irls(y::Vector{T}, X::Matrix{T}, o_ids::Vector{Int},
                       d_ids::Vector{Int}, fe::GravityFE,
                       N_orig::Int, N_dest::Int;
                       alpha::T=zero(T), maxiter::Int=100,
                       tol::T=T(1e-8)) where {T}
    N_obs, K = size(X)
    β = zeros(T, K)
    origin_fe = zeros(T, N_orig)
    dest_fe = zeros(T, N_dest)
    μ = fill(max(mean(y), T(0.1)), N_obs)
    w = copy(μ)
    X_dm = Matrix{T}(undef, N_obs, K)
    converged = false
    iter = 0

    for it in 1:maxiter
        iter = it
        β_old = copy(β)

        η = log.(max.(μ, T(1e-12)))
        z = η .+ (y .- μ) ./ max.(μ, T(1e-12))
        @inbounds for n in 1:N_obs
            m = max(μ[n], T(1e-12))
            w[n] = m / (one(T) + alpha * m)
        end

        z_dm = _demean_fe!(copy(z), fe, o_ids, d_ids, N_orig, N_dest, w)
        for k in 1:K
            X_dm[:, k] = _demean_fe!(X[:, k], fe, o_ids, d_ids, N_orig, N_dest, w)
        end

        if it == 1
            # A regressor the fixed effects absorb completely leaves a zero
            # column here, and the least squares below would hand back an
            # arbitrary coefficient for it with no indication of trouble.
            @inbounds for k in 1:K
                scale = max(one(T), maximum(abs, @view X[:, k]))
                if maximum(abs, @view X_dm[:, k]) <= T(1e-10) * scale
                    throw(ArgumentError(
                        "fit_gravity: regressor $k has no variation left after " *
                        "the fixed effects are swept out, so its coefficient is " *
                        "not identified. Drop it, or use a different `fe`."))
                end
            end
        end

        W_sqrt = Diagonal(sqrt.(w))
        β = (W_sqrt * X_dm) \ (W_sqrt * z_dm)

        _solve_fe!(origin_fe, dest_fe, z .- X * β, fe, o_ids, d_ids,
                   N_orig, N_dest, w)

        η_new = X * β
        @inbounds for n in 1:N_obs
            η_new[n] += origin_fe[o_ids[n]] + dest_fe[d_ids[n]]
        end
        μ = exp.(clamp.(η_new, T(-30), T(30)))

        if maximum(abs, β .- β_old) < tol
            converged = true
            break
        end
    end

    return β, μ, origin_fe, dest_fe, X_dm, w, converged, iter
end

"""
    _sandwich_vcov(X_dm, resid, w)

HC0 sandwich variance for `beta`, built from the within-transformed design.

The score of a log-link GLM is `X_dm' (y - mu)`, so the meat is
`sum (y_n - mu_n)^2 x_n x_n'` over the demeaned rows and the bread is
`(X_dm' W X_dm)^-1`. Both have to use `X_dm`: with the raw `X` the bread is
the variance of a model without fixed effects.
"""
function _sandwich_vcov(X_dm::AbstractMatrix{T}, resid::Vector{T},
                        w::Vector{T}) where {T}
    N, K = size(X_dm)
    XwX = zeros(T, K, K)
    meat = zeros(T, K, K)
    @inbounds for n in 1:N
        wn = w[n]
        r2 = resid[n]^2
        for j in 1:K, i in 1:K
            XwX[i, j] += wn * X_dm[n, i] * X_dm[n, j]
            meat[i, j] += r2 * X_dm[n, i] * X_dm[n, j]
        end
    end
    B = inv(Symmetric(XwX))
    return Matrix(Symmetric(B * meat * B))
end

"""
    _nb_count_params(fe, K, N_orig, N_dest)

Parameters consumed by the mean: the regressors plus the fixed effects the
sweep absorbs. Two-way effects lose one to the level they share.
"""
function _nb_count_params(fe::GravityFE, K::Int, N_orig::Int, N_dest::Int)
    fe isa BilateralFE && return K + N_orig + N_dest - 1
    fe isa OriginFE && return K + N_orig
    fe isa DestinationFE && return K + N_dest
    return K
end

"""
    _nb_alpha_mom(y, mu, dof)

The NB2 dispersion solving the Pearson moment condition
`sum (y - mu)^2 / (mu + alpha mu^2) = N - dof`.

`dof` has to count the absorbed fixed effects, not just the regressors.
Using `N - K` biases `alpha` down by about a fifth at 30 origins and 25
destinations: the target is then too large by the number of effects, and
the statistic falls as `alpha` rises, so the root comes back too small.

Maximum likelihood is the obvious alternative and is worse here, which is
why this is a moment estimator. Profiling the NB2 likelihood over `alpha`
takes no account of the `N_orig + N_dest` effects the mean consumed -- the
incidental-parameters problem -- and measured against known dispersion it
came out further off than the corrected moment condition at every value
tried (0.058 against 0.096 at a true 0.1, over twelve replications).

What remains is a mild downward bias that grows with the dispersion: about
4% low at `alpha = 0.5` and 13% low at `alpha = 2` with 750 observations,
30 origins and 25 destinations. Since `alpha` widens the standard errors,
read those as slightly optimistic when the dispersion is large.

The left side falls monotonically in `alpha`, so a bisection on `log alpha`
finds the root without derivatives and without the risk of a negative or
runaway estimate. Returns the lower bound when the data show no
overdispersion at all.
"""
function _nb_alpha_mom(y::Vector{T}, μ::Vector{T}, dof::Int;
                       lo::T=T(1e-8), hi::T=T(1e4), iters::Int=200) where {T}
    target = T(max(length(y) - dof, 1))
    function pearson(a)
        s = zero(T)
        @inbounds for n in eachindex(y)
            m = max(μ[n], T(1e-12))
            s += (y[n] - m)^2 / (m + a * m^2)
        end
        return s - target
    end
    pearson(lo) <= zero(T) && return lo
    pearson(hi) >= zero(T) && return hi
    a_lo = lo; a_hi = hi
    for _ in 1:iters
        mid = sqrt(a_lo * a_hi)
        pearson(mid) > zero(T) ? (a_lo = mid) : (a_hi = mid)
    end
    return sqrt(a_lo * a_hi)
end

"""
    _nb_loglik(y, mu, alpha)

NB2 log-likelihood at a given mean and dispersion.
"""
function _nb_loglik(y::Vector{T}, μ::Vector{T}, alpha::T) where {T}
    inv_a = one(T) / max(alpha, T(1e-12))
    ll = zero(T)
    @inbounds for n in eachindex(y)
        m = max(μ[n], T(1e-30))
        yn = y[n]
        ll += _loggamma(yn + inv_a) - _loggamma(inv_a) - _logfactorial(yn)
        ll += inv_a * log(inv_a / (inv_a + m))
        ll += yn * log(m / (inv_a + m))
    end
    return ll
end

"""
    _poisson_deviance(y, mu)

`2 * sum[ y log(y/mu) - (y - mu) ]`, with the convention `0 log 0 = 0`.
"""
function _poisson_deviance(y::Vector{T}, μ::Vector{T}) where {T}
    d = zero(T)
    @inbounds for n in eachindex(y)
        m = max(μ[n], T(1e-30))
        yn = y[n]
        d += yn > zero(T) ? yn * log(yn / m) - (yn - m) : m
    end
    return T(2) * d
end

"""
    _nb_deviance(y, mu, alpha)

`2 * sum[ y log(y/mu) - (y + 1/alpha) log((y + 1/alpha)/(mu + 1/alpha)) ]`.
"""
function _nb_deviance(y::Vector{T}, μ::Vector{T}, alpha::T) where {T}
    inv_a = one(T) / max(alpha, T(1e-12))
    d = zero(T)
    @inbounds for n in eachindex(y)
        m = max(μ[n], T(1e-30))
        yn = y[n]
        first = yn > zero(T) ? yn * log(yn / m) : zero(T)
        d += first - (yn + inv_a) * log((yn + inv_a) / (m + inv_a))
    end
    return T(2) * d
end

# ─── Negative Binomial gravity ────────────────────────────────────────

"""
    fit_gravity(NegBinGravity(), flows, X, origin_id, dest_id; fe = BilateralFE(),
                varnames = nothing, maxiter = 100, tol = 1e-8, alpha_iter = 50)

Fit the same gravity model with an NB2 variance, `V(mu) = mu + alpha*mu^2`.

Use this when the flows are overdispersed beyond what Poisson allows. PPML
stays consistent under overdispersion, so the coefficients usually move
little; what changes is the weighting and, more to the point, the standard
errors, which Poisson understates in exactly this situation.

`alpha` is estimated by alternating: given `alpha`, IRLS for `beta`; given the
fit, the Pearson moment condition for `alpha`. Read it off `m.alpha`, where
`0` would mean Poisson.

!!! warning
    Check `m.converged`, which here requires both the inner IRLS and the outer
    `alpha` iteration to have settled.

# References
Cameron, A. C. & Trivedi, P. K. (2013). *Regression Analysis of Count Data*,
2nd ed. Cambridge University Press, chapter 3.
"""
function fit_gravity(
    ::NegBinGravity,
    flows::AbstractVector,
    X::AbstractMatrix{T},
    origin_id::AbstractVector{<:Integer},
    dest_id::AbstractVector{<:Integer};
    fe::GravityFE=BilateralFE(),
    varnames::Union{Nothing, Vector{String}}=nothing,
    maxiter::Int=100,
    tol::T=T(1e-8),
    alpha_iter::Int=50
) where {T<:AbstractFloat}

    N_obs = length(flows)
    K = size(X, 2)
    y = Vector{T}(flows)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    o_ids = Vector{Int}(origin_id)
    d_ids = Vector{Int}(dest_id)
    N_orig = maximum(o_ids)
    N_dest = maximum(d_ids)
    Xm = Matrix{T}(X)

    # Poisson start, which is alpha = 0 through the same core.
    β, μ, origin_fe_vec, dest_fe_vec, X_dm, w, irls_ok, _ =
        _gravity_irls(y, Xm, o_ids, d_ids, fe, N_orig, N_dest;
                      alpha = zero(T), maxiter = maxiter, tol = tol)
    dof = _nb_count_params(fe, K, N_orig, N_dest)
    alpha = _nb_alpha_mom(y, μ, dof)

    # Alternate until the dispersion settles. The weights w = mu/(1 + alpha*mu)
    # are the whole of the negative binomial here: the previous version of this
    # function computed them, discarded them, and refit the identical Poisson
    # model on every pass, so its coefficients were Poisson's and its standard
    # errors were Poisson's too.
    converged = false
    iter = 0
    for it in 1:alpha_iter
        iter = it
        alpha_old = alpha

        β, μ, origin_fe_vec, dest_fe_vec, X_dm, w, irls_ok, _ =
            _gravity_irls(y, Xm, o_ids, d_ids, fe, N_orig, N_dest;
                          alpha = alpha, maxiter = maxiter, tol = tol)
        alpha = _nb_alpha_mom(y, μ, dof)

        if abs(alpha - alpha_old) <= tol * max(one(T), alpha_old)
            converged = irls_ok
            break
        end
    end

    inv_alpha = one(T) / max(alpha, T(1e-12))
    loglik = _nb_loglik(y, μ, alpha)

    # Null model under the same variance, so the pseudo-R2 compares like with
    # like rather than an NB fit against a Poisson null.
    ȳ = max(mean(y), T(1e-30))
    null_ll = zero(T)
    @inbounds for n in 1:N_obs
        yn = y[n]
        null_ll += _loggamma(yn + inv_alpha) - _loggamma(inv_alpha) - _logfactorial(yn)
        null_ll += inv_alpha * log(inv_alpha / (inv_alpha + ȳ))
        null_ll += yn * log(ȳ / (inv_alpha + ȳ))
    end

    residuals = y .- μ
    deviance = _nb_deviance(y, μ, alpha)
    pseudo_R2 = one(T) - loglik / min(null_ll, -T(1e-12))
    vcov_mat = _sandwich_vcov(X_dm, residuals, w)

    return GravityModel{NegBinGravity, T}(
        NegBinGravity(), β, vcov_mat,
        origin_fe_vec, dest_fe_vec, alpha,
        loglik, deviance, pseudo_R2,
        residuals, μ,
        N_obs, N_orig, N_dest, vnames,
        converged, iter
    )
end

# ─── Multilateral resistance ─────────────────────────────────────────

"""
    multilateral_resistance(model) -> NamedTuple

Recover the fitted fixed effects as `(; outward, inward)`.

In structural gravity the origin and destination fixed effects *are* the
multilateral resistance terms: `outward` measures how hard it is for each
origin to reach all markets, `inward` how accessible each destination is
from all sources. They are nuisance parameters for estimating `beta` but
often the quantity of substantive interest.

Both vectors are identified only up to a normalization, so compare values
within a vector rather than reading levels.

# Examples
```julia
mr = multilateral_resistance(m)
sortperm(mr.inward, rev = true)      # most accessible destinations
```

See also [`GravityModel`](@ref), [`BilateralFE`](@ref).
"""
function multilateral_resistance(m::GravityModel)
    return (outward = m.origin_fe, inward = m.dest_fe)
end

# ─── Show ─────────────────────────────────────────────────────────────

function Base.show(io::IO, m::GravityModel{V}) where {V}
    vname = V === PoissonGravity ? "Poisson (PPML)" : "Negative Binomial"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Gravity Model — $vname — $(m.converged ? "Converged" : "Not converged") ($(m.iterations) iter)")
    println(io, "───────────────────────────────────────────────────────────────────────")
    ses = StatsAPI.stderror(m)
    _print_coef_table(io, m.coefnames, m.coef, ses)
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "Log-lik = %.2f    Pseudo-R² = %.4f    N = %d\n", m.loglik, m.pseudo_R2, m.N_obs)
    @printf(io, "Origins = %d    Destinations = %d", m.N_origin, m.N_dest)
    V === NegBinGravity && @printf(io, "    α = %.4f", m.alpha)
    println(io)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─── StatsAPI ─────────────────────────────────────────────────────────

StatsAPI.coef(m::GravityModel) = m.coef
StatsAPI.vcov(m::GravityModel) = m.vcov
StatsAPI.stderror(m::GravityModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::GravityModel) = m.loglik
StatsAPI.nobs(m::GravityModel) = m.N_obs
StatsAPI.residuals(m::GravityModel) = m.residuals
StatsAPI.coefnames(m::GravityModel) = m.coefnames

end # module
