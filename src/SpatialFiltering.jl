module SpatialFiltering

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames,
       loglikelihood, dof_residual, predict, fitted

import ..SpatialUtils
import ..SpatialUtils: brent_maximize, _normcdf, _print_coef_table

export SpatialEigen, spatial_eigenvectors, moran_coefficient,
       ESFModel, fit_esf, selected_eigenvectors,
       REESFModel, fit_re_esf

const _ESF_FAMILIES = (:gaussian, :poisson, :logit, :negbin)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

# Lanczos log-gamma (Numerical Recipes gammln), for z > 0.
function _lgamma(z::Real)
    x = Float64(z)
    x <= 0.0 && return Inf
    c = (76.18009172947146, -86.50532032941677, 24.01409824083091,
         -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5)
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * log(tmp)
    ser = 1.000000000190015
    for j in 1:6
        y += 1.0
        ser += c[j] / y
    end
    return -tmp + log(2.5066282746310005 * ser / x)
end

"""
    moran_coefficient(y, W)

Moran coefficient `(N/S₀) · (z'Wz)/(z'z)` for `z = y - ȳ`, where `S₀` is the sum
of all weights. The bare statistic with no inference — use
`SpatialDiagnostics.moran_global` when you need a p-value.
"""
function moran_coefficient(y::AbstractVector{T}, W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    return _moran_stat(y, W, T(sum(W)))
end

function _moran_stat(y::AbstractVector{T}, W::AbstractSparseMatrix{T}, S0::T) where {T<:AbstractFloat}
    N = length(y)
    z = y .- mean(y)
    den = dot(z, z)
    den <= zero(T) && return zero(T)
    return (T(N) / S0) * dot(z, W * z) / den
end

# ═══════════════════════════════════════════════════════════════════════
# Eigenvector extraction
# ═══════════════════════════════════════════════════════════════════════

"""
    SpatialEigen

A screened eigenvector basis for spatial filtering, as returned by
[`spatial_eigenvectors`](@ref).

# Fields
- `values::Vector{T}`: eigenvalues of `Ω = M Wᵤ M`, in descending order.
- `vectors::Matrix{T}`: the corresponding eigenvectors, one per column. Columns
  are orthonormal and each sums to zero.
- `moran::Vector{T}`: each eigenvector's Moran coefficient, `(N/S₀)λᵢ`.
- `S0::T`: the sum of all weights, `Σᵢⱼ wᵢⱼ`.
- `N::Int`: the number of spatial units.

Pass an instance as the `eigen_basis` keyword of [`fit_esf`](@ref) or
[`fit_re_esf`](@ref) to reuse one decomposition across several fits.
"""
struct SpatialEigen{T<:AbstractFloat}
    values::Vector{T}
    vectors::Matrix{T}
    moran::Vector{T}
    S0::T
    N::Int
end

"""
    spatial_eigenvectors(W; n_candidates=nothing, min_moran_ratio=0.0)

Eigendecomposition of the doubly-centred weight matrix
`Ω = (I - 11'/N) Wsym (I - 11'/N)`, following Griffith (2003).

The eigenvectors are mutually orthogonal and uncorrelated, and each carries
Moran coefficient `MCᵢ = (N/S₀)λᵢ`. Eigenvectors with large positive eigenvalues
describe positive spatial autocorrelation patterns; large negative eigenvalues
describe negative ones. Results are sorted by eigenvalue, descending.

An asymmetric `W` (e.g. row-standardised) is symmetrised as `(W + W')/2` so the
eigenvalues are real.

The constant vector is removed by construction rather than by discarding an
eigenvector afterwards: with `Q` an orthonormal basis of `1`'s orthogonal
complement, `Q'ΩQ = Q'WsymQ`, so eigendecomposing the reduced
`(N-1)×(N-1)` matrix and mapping back as `E = QV` yields vectors that are
orthonormal *and* exactly mean-zero. Decomposing `Ω` directly would not: its
null space is multi-dimensional, the solver may return any rotation within it,
and the constant direction then smears across several near-zero eigenvectors
that are each partly collinear with the intercept.

`n_candidates` keeps only the leading eigenvectors; `min_moran_ratio` keeps those
with `MCᵢ / max(MC) ≥ ratio`, the usual screen for positive-SA filtering (0.25 is
a common choice). Both are applied after sorting.

Dense `eigen` is O(N³) — intended for moderate N.
"""
function spatial_eigenvectors(
    W::AbstractSparseMatrix{T};
    n_candidates::Union{Nothing, Int}=nothing,
    min_moran_ratio::Real=0.0
) where {T<:AbstractFloat}

    N = size(W, 1)
    @assert size(W, 2) == N "W must be square"
    S0 = T(sum(W))
    @assert S0 > zero(T) "W must have positive total weight"

    Wd = Matrix{T}(W)
    Wsym = T(0.5) .* (Wd .+ transpose(Wd))

    # Orthonormal basis Q of the complement of 1, via the full QR of 1.
    Qfull = Matrix(qr(ones(T, N, 1)).Q * Matrix{T}(I, N, N))
    Q = Qfull[:, 2:N]

    # Q'ΩQ = Q'WsymQ, since MQ = Q for the centring projector M.
    Ωr = Symmetric(T(0.5) .* (Q' * Wsym * Q .+ transpose(Q' * Wsym * Q)))

    F = eigen(Ωr)
    ord = sortperm(F.values; rev=true)
    vals = F.values[ord]
    vecs = Q * F.vectors[:, ord]

    mc = (T(N) / S0) .* vals

    if min_moran_ratio > 0
        mc_max = maximum(mc)
        if mc_max > zero(T)
            sel = findall(v -> v / mc_max >= T(min_moran_ratio), mc)
            vals = vals[sel]; vecs = vecs[:, sel]; mc = mc[sel]
        end
    end

    if !isnothing(n_candidates) && n_candidates < length(vals)
        r = 1:n_candidates
        vals = vals[r]; vecs = vecs[:, r]; mc = mc[r]
    end

    return SpatialEigen{T}(vals, vecs, mc, S0, N)
end

function Base.show(io::IO, se::SpatialEigen)
    println(io, "Spatial Eigenvectors (Griffith double-centring)")
    println(io, "───────────────────────────────────────────────")
    @printf(io, "  N = %d,  candidates = %d,  S₀ = %.2f\n",
        se.N, length(se.values), se.S0)
    if !isempty(se.values)
        @printf(io, "  eigenvalue range: [%.4f, %.4f]\n",
            minimum(se.values), maximum(se.values))
        @printf(io, "  Moran range:      [%.4f, %.4f]\n",
            minimum(se.moran), maximum(se.moran))
        @printf(io, "  positive-SA candidates (MC > 0): %d\n", count(>(0), se.moran))
    end
end

# ═══════════════════════════════════════════════════════════════════════
# GLM machinery (IRLS) — no Distributions.jl / GLM.jl dependency
# ═══════════════════════════════════════════════════════════════════════

function _linkinv(family::Symbol, η::T) where {T}
    if family === :gaussian
        return η
    elseif family === :poisson || family === :negbin
        return exp(clamp(η, T(-30), T(30)))
    else # :logit
        return one(T) / (one(T) + exp(-clamp(η, T(-30), T(30))))
    end
end

# Working weight and working response for one IRLS step.
function _irls_parts(family::Symbol, μ::T, y::T, η::T, α::T) where {T}
    if family === :gaussian
        return one(T), y
    elseif family === :poisson
        μc = max(μ, T(1e-10))
        return μc, η + (y - μc) / μc
    elseif family === :negbin
        μc = max(μ, T(1e-10))
        return μc / (one(T) + α * μc), η + (y - μc) / μc
    else # :logit
        v = max(μ * (one(T) - μ), T(1e-10))
        return v, η + (y - μ) / v
    end
end

function _glm_loglik(family::Symbol, y::AbstractVector{T}, μ::AbstractVector{T},
                     α::T, σ2::T) where {T}
    if family === :gaussian
        N = length(y)
        return -T(0.5) * N * (log(T(2π)) + log(max(σ2, T(1e-300))) + one(T))
    elseif family === :poisson
        s = zero(T)
        for i in eachindex(y)
            μi = max(μ[i], T(1e-10))
            s += y[i] * log(μi) - μi - T(_lgamma(y[i] + 1))
        end
        return s
    elseif family === :logit
        s = zero(T)
        for i in eachindex(y)
            μi = clamp(μ[i], T(1e-10), one(T) - T(1e-10))
            s += y[i] * log(μi) + (one(T) - y[i]) * log(one(T) - μi)
        end
        return s
    else # :negbin
        s = zero(T)
        invα = one(T) / max(α, T(1e-10))
        for i in eachindex(y)
            μi = max(μ[i], T(1e-10))
            s += T(_lgamma(y[i] + invα)) - T(_lgamma(invα)) - T(_lgamma(y[i] + 1)) +
                 invα * log(invα / (invα + μi)) + y[i] * log(μi / (invα + μi))
        end
        return s
    end
end

# Fit a GLM by IRLS. Returns (β, μ, working weights, converged).
function _fit_glm(X::AbstractMatrix{T}, y::AbstractVector{T}, family::Symbol;
                  α::T=zero(T), maxiter::Int=100, tol::T=T(1e-9)) where {T<:AbstractFloat}
    N, P = size(X)
    if family === :gaussian
        β = X \ y
        μ = X * β
        return β, μ, ones(T, N), true
    end

    β = zeros(T, P)
    # Start the intercept at the link of the mean outcome.
    ybar = clamp(mean(y), T(1e-6), family === :logit ? one(T) - T(1e-6) : T(Inf))
    β[1] = family === :logit ? log(ybar / (one(T) - ybar)) : log(max(ybar, T(1e-6)))

    w = ones(T, N)
    μ = similar(y)
    converged = false
    for _ in 1:maxiter
        η = X * β
        z = similar(y)
        @inbounds for i in 1:N
            μ[i] = _linkinv(family, η[i])
            wi, zi = _irls_parts(family, μ[i], y[i], η[i], α)
            w[i] = wi
            z[i] = zi
        end
        XtWX = Symmetric(X' * (w .* X))
        rhs = X' * (w .* z)
        β_new = try
            XtWX \ rhs
        catch
            (Matrix(XtWX) + T(1e-8) * I) \ rhs
        end
        Δ = norm(β_new .- β)
        β = β_new
        if Δ < tol
            converged = true
            break
        end
    end

    η = X * β
    @inbounds for i in 1:N
        μ[i] = _linkinv(family, η[i])
        w[i], _ = _irls_parts(family, μ[i], y[i], η[i], α)
    end
    return β, μ, w, converged
end

# Method-of-moments dispersion update for the negative binomial.
function _update_negbin_alpha(y::AbstractVector{T}, μ::AbstractVector{T}) where {T}
    N = length(y)
    num = zero(T)
    for i in 1:N
        μi = max(μ[i], T(1e-10))
        num += ((y[i] - μi)^2 - μi) / (μi^2)
    end
    return max(T(1e-6), num / N)
end

function _pearson_resid(family::Symbol, y::AbstractVector{T}, μ::AbstractVector{T},
                        α::T) where {T}
    r = similar(y)
    @inbounds for i in eachindex(y)
        μi = μ[i]
        v = if family === :gaussian
            one(T)
        elseif family === :poisson
            max(μi, T(1e-10))
        elseif family === :negbin
            max(μi + α * μi^2, T(1e-10))
        else
            max(μi * (one(T) - μi), T(1e-10))
        end
        r[i] = (y[i] - μi) / sqrt(v)
    end
    return r
end

# ═══════════════════════════════════════════════════════════════════════
# ESFModel — eigenvector spatial filtering with forward selection
# ═══════════════════════════════════════════════════════════════════════

"""
    ESFModel

An eigenvector spatial filtering fit, returned by [`fit_esf`](@ref).

Spatial structure is absorbed by adding selected eigenvectors of the
doubly-centred weight matrix to the design as synthetic regressors, chosen
stepwise. Because they are orthogonal to the intercept and to each other,
the covariate coefficients stay interpretable as ordinary regression
coefficients — the appeal of the method over a lag model.

Inspect the chosen basis with [`selected_eigenvectors`](@ref).

!!! note
    Selecting zero eigenvectors is a valid outcome, not a failure: it means
    the residuals were already free of spatial autocorrelation.
"""
struct ESFModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    family::Symbol
    alpha::T
    loglik::T
    sigma2::T
    W::M
    X_aug::Matrix{T}
    y::Vector{T}
    fitted::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
    selected::Vector{Int}
    selected_moran::Vector{T}
    moran_initial::T
    moran_final::T
    n_covariates::Int
    converged::Bool
end

"""
    fit_esf(X, y, W; family=:gaussian, max_eigenvectors=20, criterion=:moran, ...)

Eigenvector spatial filtering (Griffith, 2003). Selected eigenvectors of the
doubly-centred `W` enter as extra regressors, absorbing the spatial
autocorrelation so the remaining coefficients can be read as aspatial:

```math
g(\\mu) = \\alpha + X\\beta + E_{sel}\\gamma
```

`family` is `:gaussian` (OLS), `:poisson`, `:logit`, or `:negbin`; the last three
are fit by IRLS, with the negative binomial's dispersion updated by method of
moments between IRLS passes (Chun et al., 2016). This makes ESF usable wherever a
GLM is, which parametric SAR/SEM cannot match.

Eigenvectors are added greedily. `criterion=:moran` picks, at each step, the
candidate that most reduces the absolute Moran coefficient of the residuals
(Pearson residuals for GLM families); `criterion=:aic` picks the one that most
reduces AIC. Selection stops at `max_eigenvectors`, when no candidate improves
the criterion, or — under `:moran` — once `|MC|` falls below `moran_tol`.

`min_moran_ratio` screens the candidate pool to eigenvectors with
`MCᵢ/max(MC) ≥ ratio` (default 0.25, the usual positive-SA screen). Pass a
precomputed [`SpatialEigen`](@ref) as `eigen_basis` to reuse one decomposition
across several fits.

Cost is `O(max_eigenvectors × n_candidates)` model fits, so keep the pool bounded
for GLM families.
"""
function fit_esf(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    W::M;
    family::Symbol=:gaussian,
    max_eigenvectors::Int=20,
    criterion::Symbol=:moran,
    min_moran_ratio::Real=0.25,
    moran_tol::Real=0.05,
    n_candidates::Union{Nothing, Int}=nothing,
    eigen_basis::Union{Nothing, SpatialEigen{T}}=nothing,
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert length(y) == N "y and X must have the same number of rows"
    @assert family in _ESF_FAMILIES "family must be one of $(_ESF_FAMILIES)"
    @assert criterion in (:moran, :aic) "criterion must be :moran or :aic"
    @assert max_eigenvectors >= 0
    family === :logit && @assert all(v -> v == zero(T) || v == one(T), y) "logit requires y coded 0/1"
    if family === :poisson || family === :negbin
        @assert all(v -> v >= zero(T), y) "count families require y >= 0"
    end
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    se = isnothing(eigen_basis) ?
        spatial_eigenvectors(W; n_candidates=n_candidates, min_moran_ratio=min_moran_ratio) :
        eigen_basis
    n_cand = length(se.values)

    X_base = hcat(ones(T, N), X)
    yv = Vector{T}(y)
    S0 = T(sum(W))

    function fit_with(cols::Vector{Int})
        Xf = isempty(cols) ? X_base : hcat(X_base, se.vectors[:, cols])
        α = zero(T)
        β, μ, w, conv = _fit_glm(Xf, yv, family)
        if family === :negbin
            for _ in 1:12
                α_new = _update_negbin_alpha(yv, μ)
                abs(α_new - α) < T(1e-8) && (α = α_new; break)
                α = α_new
                β, μ, w, conv = _fit_glm(Xf, yv, family; α=α)
            end
        end
        return Xf, β, μ, w, α, conv
    end

    _, _, μ0, _, α0, _ = fit_with(Int[])
    r0 = _pearson_resid(family, yv, μ0, α0)
    mc0 = _moran_stat(r0, W, S0)

    function crit_of(cols::Vector{Int})
        Xf, β, μ, _, α, _ = fit_with(cols)
        r = _pearson_resid(family, yv, μ, α)
        if criterion === :moran
            return abs(_moran_stat(r, W, S0))
        else
            σ2 = dot(r, r) / N
            ll = _glm_loglik(family, yv, μ, α, σ2)
            n_par = length(β) + (family === :gaussian ? 1 : 0) + (family === :negbin ? 1 : 0)
            return T(-2) * ll + T(2) * n_par
        end
    end

    selected = Int[]
    remaining = collect(1:n_cand)
    best_crit = crit_of(Int[])
    max_ev = min(max_eigenvectors, n_cand)

    while length(selected) < max_ev && !isempty(remaining)
        if criterion === :moran && best_crit < T(moran_tol)
            break
        end
        step_best = best_crit
        step_pick = 0
        for c in remaining
            v = crit_of(vcat(selected, c))
            if v < step_best - T(1e-12)
                step_best = v
                step_pick = c
            end
        end
        step_pick == 0 && break
        push!(selected, step_pick)
        deleteat!(remaining, findfirst(==(step_pick), remaining))
        best_crit = step_best
    end

    Xf, β, μ, w, α, conv = fit_with(selected)
    r_pearson = _pearson_resid(family, yv, μ, α)
    mc_final = _moran_stat(r_pearson, W, S0)
    res = yv .- μ
    σ2 = dot(res, res) / N
    ll = _glm_loglik(family, yv, μ, α, σ2)

    XtWX = Symmetric(Xf' * (w .* Xf))
    V = try
        family === :gaussian ? σ2 .* inv(XtWX) : inv(XtWX)
    catch
        conv = false
        Matrix{T}(I, size(Xf, 2), size(Xf, 2))
    end
    V = T(0.5) .* (V .+ transpose(V))

    cnames = vcat(["(Intercept)"], vnames, ["E$(c)" for c in selected])

    return ESFModel{T, M}(β, Matrix(V), family, α, ll, σ2, W, Xf, yv, μ, res,
        cnames, selected, se.moran[selected], mc0, mc_final, K, conv)
end

StatsAPI.coef(m::ESFModel) = m.coef
StatsAPI.vcov(m::ESFModel) = m.vcov
StatsAPI.stderror(m::ESFModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::ESFModel) = length(m.y)
StatsAPI.residuals(m::ESFModel) = m.residuals
StatsAPI.coefnames(m::ESFModel) = m.coefnames
StatsAPI.loglikelihood(m::ESFModel) = m.loglik
StatsAPI.dof_residual(m::ESFModel) = length(m.y) - length(m.coef)
StatsAPI.fitted(m::ESFModel) = m.fitted
StatsAPI.predict(m::ESFModel) = m.fitted

"""
    selected_eigenvectors(m::ESFModel)

Indices of the selected eigenvectors (into the candidate pool) with their Moran
coefficients, plus the residual Moran coefficient before and after filtering.
"""
function selected_eigenvectors(m::ESFModel)
    return (indices = m.selected,
            moran = m.selected_moran,
            moran_initial = m.moran_initial,
            moran_final = m.moran_final)
end

function Base.show(io::IO, m::ESFModel{T}) where {T}
    println(io, "ESF Model — eigenvector spatial filtering ($(m.family))")
    println(io, "═"^56)
    @printf(io, "  N = %d,  covariates = %d,  eigenvectors selected = %d\n",
        nobs(m), m.n_covariates, length(m.selected))
    @printf(io, "  Residual Moran: %.4f → %.4f\n", m.moran_initial, m.moran_final)
    @printf(io, "  Log-Lik = %.2f", m.loglik)
    m.family === :negbin && @printf(io, ",  α = %.4f", m.alpha)
    m.family === :gaussian && @printf(io, ",  σ² = %.4f", m.sigma2)
    println(io)
    m.converged || println(io, "  WARNING: IRLS did not converge; estimates may be unstable.")
    println(io, "")
    # Print covariates only; the eigenvector block is nuisance.
    n_show = 1 + m.n_covariates
    _print_coef_table(io, m.coefnames[1:n_show], m.coef[1:n_show], stderror(m)[1:n_show])
    length(m.selected) > 0 && println(io,
        "  (+ $(length(m.selected)) eigenvector terms omitted)")
end

# ═══════════════════════════════════════════════════════════════════════
# REESFModel — random-effects ESF (Murakami & Griffith 2015)
# ═══════════════════════════════════════════════════════════════════════

"""
    REESFModel

A random-effects eigenvector spatial filtering fit, returned by
[`fit_re_esf`](@ref).

Instead of selecting eigenvectors stepwise as [`fit_esf`](@ref) does, this
keeps the whole screened pool and shrinks every coefficient towards zero,
governed by one variance ratio `theta = tau2/sigma2` estimated by profile
REML. That avoids the model-selection variance stepwise search introduces.

!!! warning
    The reported residual Moran coefficient is for the *conditional*
    residual and is **supposed to be substantially negative** — do not
    compare it against zero. Subtracting the fitted spatial term leaves the
    remainder in the orthogonal complement of the retained eigenvectors,
    which is itself negatively autocorrelated. Compare it against
    `moran_reference`, the closed-form expectation stored on the model.
"""
struct REESFModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    gamma::Vector{T}
    theta::T
    tau2::T
    sigma2::T
    loglik::T
    edf::T
    W::M
    X_aug::Matrix{T}
    E::Matrix{T}
    lambda::Vector{T}
    y::Vector{T}
    fitted::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
    moran_initial::T
    moran_final::T
    moran_reference::T
end

"""
    fit_re_esf(X, y, W; min_moran_ratio=0.25, n_candidates=nothing, varnames=nothing)

Random-effects eigenvector spatial filtering (Murakami & Griffith, 2015):

```math
y = X\\beta + E\\gamma + \\varepsilon, \\quad
\\gamma \\sim N(0, \\tau^2\\Lambda), \\quad \\varepsilon \\sim N(0, \\sigma^2 I)
```

Rather than selecting eigenvectors discretely, every positive-eigenvalue
eigenvector is retained and its coefficient shrunk toward zero, with the
shrinkage governed by the single variance ratio `θ = τ²/σ²` estimated by profile
REML. This sidesteps the model-selection variance of stepwise ESF.

Because `E` has orthonormal columns, `V = σ²(I + θEΛE')` inverts in closed form —
its eigenvalues are `1 + θλₗ` inside `span(E)` and 1 outside — so each REML
evaluation is O(NL), not O(N³). The decomposition itself is still O(N³).

Only positive-eigenvalue eigenvectors are used, the standard choice for positive
spatial autocorrelation, and `min_moran_ratio` screens them further to
`MCᵢ/max(MC) ≥ ratio`.

!!! note "Reading the residual Moran coefficient"
    `moran_final` is the Moran coefficient of the *conditional* residual
    `y - Xβ̂ - Eγ̂`, and **it is expected to be substantially negative — do not
    compare it against zero.** Subtracting `Eγ̂` removes the positively
    autocorrelated directions, leaving a residual that lives mostly in the
    orthogonal complement of `span(E)`; a vector drawn at random from that
    complement has Moran coefficient

    ```math
    \\frac{N}{S_0}\\cdot\\frac{-S_0/N - \\sum_l \\lambda_l}{N - 1 - L}
    ```

    which is reported as `moran_reference`. On correctly specified data with
    `N = 121` and `L = 40` this reference is −0.304 and the fitted residual lands
    near −0.24, i.e. slightly *less* negative than chance. A `moran_final` close
    to or above `moran_reference` is the healthy case; one far below it suggests
    the random effects are absorbing noise, as does an `edf` (effective degrees
    of freedom, `P + Σ θλ/(1+θλ)`) approaching `N`.
"""
function fit_re_esf(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    W::M;
    min_moran_ratio::Real=0.25,
    n_candidates::Union{Nothing, Int}=nothing,
    eigen_basis::Union{Nothing, SpatialEigen{T}}=nothing,
    varnames::Union{Nothing, Vector{String}}=nothing
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert length(y) == N "y and X must have the same number of rows"
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    se = isnothing(eigen_basis) ?
        spatial_eigenvectors(W; n_candidates=n_candidates,
                             min_moran_ratio=min_moran_ratio) : eigen_basis
    pos = findall(>(T(1e-10)), se.values)
    @assert !isempty(pos) "no positive-eigenvalue eigenvectors available"
    E = se.vectors[:, pos]
    λ = se.values[pos]
    L = length(λ)

    Xa = hcat(ones(T, N), X)
    P = size(Xa, 2)
    yv = Vector{T}(y)
    S0 = T(sum(W))

    EtX = E' * Xa
    Ety = E' * yv

    # V₀⁻¹ = I - E D E' with D = diag(θλ/(1+θλ)); log|V₀| = Σ log(1+θλ).
    function reml(logθ)
        θ = exp(logθ)
        d = (θ .* λ) ./ (one(T) .+ θ .* λ)
        XtVX = Symmetric(Xa' * Xa .- EtX' * (d .* EtX))
        XtVy = Xa' * yv .- EtX' * (d .* Ety)
        β = try
            XtVX \ XtVy
        catch
            return T(-Inf)
        end
        ytVy = dot(yv, yv) - dot(Ety, d .* Ety)
        rss = ytVy - dot(XtVy, β)
        rss <= zero(T) && return T(-Inf)
        logdetV0 = sum(log.(one(T) .+ θ .* λ))
        ld_XtVX = try
            first(logabsdet(cholesky(XtVX)))
        catch
            return T(-Inf)
        end
        dfr = N - P
        return -T(0.5) * (logdetV0 + dfr * log(rss / dfr) + ld_XtVX)
    end

    logθ_hat, _ = brent_maximize(reml, T(-12), T(8))
    θ = exp(logθ_hat)

    d = (θ .* λ) ./ (one(T) .+ θ .* λ)
    XtVX = Symmetric(Xa' * Xa .- EtX' * (d .* EtX))
    XtVy = Xa' * yv .- EtX' * (d .* Ety)
    β = XtVX \ XtVy
    ytVy = dot(yv, yv) - dot(Ety, d .* Ety)
    rss = ytVy - dot(XtVy, β)
    dfr = N - P
    σ2 = max(rss / dfr, T(1e-12))
    τ2 = θ * σ2

    # BLUP: γ̂ₗ = θλₗ/(1+θλₗ) · (E'r)ₗ
    r = yv .- Xa * β
    γ = d .* (E' * r)
    fit = Xa * β .+ E * γ
    res = yv .- fit

    V = σ2 .* inv(XtVX)
    V = T(0.5) .* (V .+ transpose(V))

    ll = -T(0.5) * (N * log(T(2π) * σ2) + sum(log.(one(T) .+ θ .* λ)) + rss / σ2)
    mc0 = _moran_stat(yv .- Xa * (Xa \ yv), W, S0)
    mc1 = _moran_stat(res, W, S0)
    edf = T(P) + sum(d)
    # Expected Moran coefficient of a random vector in the complement of span(E).
    # All N-1 centred eigenvalues sum to tr(M*Wsym*M) = -S0/N, so the non-retained
    # ones sum to -S0/N - sum(lambda) and average over the remaining N-1-L dims.
    nrest = N - 1 - L
    mcref = nrest > 0 ?
        (T(N) / S0) * ((-S0 / T(N)) - sum(λ)) / T(nrest) : zero(T)

    cnames = vcat(["(Intercept)"], vnames)

    return REESFModel{T, M}(β, Matrix(V), γ, θ, τ2, σ2, ll, edf, W, Xa, E, λ,
        yv, fit, res, cnames, mc0, mc1, mcref)
end

StatsAPI.coef(m::REESFModel) = m.coef
StatsAPI.vcov(m::REESFModel) = m.vcov
StatsAPI.stderror(m::REESFModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::REESFModel) = length(m.y)
StatsAPI.residuals(m::REESFModel) = m.residuals
StatsAPI.coefnames(m::REESFModel) = m.coefnames
StatsAPI.loglikelihood(m::REESFModel) = m.loglik
StatsAPI.dof_residual(m::REESFModel) = length(m.y) - length(m.coef)
StatsAPI.fitted(m::REESFModel) = m.fitted
StatsAPI.predict(m::REESFModel) = m.fitted

function Base.show(io::IO, m::REESFModel{T}) where {T}
    println(io, "RE-ESF Model — random-effects spatial filtering")
    println(io, "═"^48)
    @printf(io, "  N = %d,  eigenvectors = %d,  edf = %.1f\n",
        nobs(m), length(m.lambda), m.edf)
    @printf(io, "  θ = τ²/σ² = %.4f,  τ² = %.4f,  σ² = %.4f\n", m.theta, m.tau2, m.sigma2)
    @printf(io, "  Residual Moran: %.4f → %.4f  (complement reference %.4f)\n",
        m.moran_initial, m.moran_final, m.moran_reference)
    @printf(io, "  Log-Lik = %.2f\n", m.loglik)
    if m.moran_final < m.moran_reference - T(0.15)
        println(io, "  NOTE: residual Moran is well below the complement reference;")
        println(io, "        check edf against N for noise absorption.")
    end
    println(io, "")
    _print_coef_table(io, m.coefnames, m.coef, stderror(m))
end

end # module
