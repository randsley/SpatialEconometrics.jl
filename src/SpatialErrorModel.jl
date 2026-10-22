module SpatialErrorModel

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, brent_maximize, panel_spatial_lag!, panel_spatial_lag,
       within_transform!, within_transform, effective_dims, has_indiv_fe, has_time_fe,
       _normcdf, _print_coef_table

const LogDetMethod = SpatialUtils.LogDetMethod
const ExactLU = SpatialUtils.ExactLU
const ChebyshevApprox = SpatialUtils.ChebyshevApprox
const FixedEffectsSpec = SpatialUtils.FixedEffectsSpec
const NoFE = SpatialUtils.NoFE
const IndivFE = SpatialUtils.IndivFE
const TimeFE = SpatialUtils.TimeFE
const TwoWayFE = SpatialUtils.TwoWayFE

export SpatialErrorRegressionModel, SpatialErrorPanelModel,
       fit_spatial_error

# ─── Cross-section SEM result ─────────────────────────────────────────

"""
    SpatialErrorRegressionModel

A fitted cross-sectional spatial error model, returned by
[`fit_spatial_error`](@ref). Implements the `StatsAPI.RegressionModel`
interface.

Carries `coef` and `vcov` for the regression parameters, the spatial error
parameter `lambda` with its standard error, `sigma2`, `loglik`, the weight
matrix, residuals and `coefnames`.

!!! note
    Unlike [`SpatialRegressionModel`](@ref), `coef` **is** interpretable
    directly: dependence sits in the error, so there is no feedback through
    `y` and no need for [`spatial_effects`](@ref). What the model buys you
    is correct standard errors, not different point estimates.
"""
struct SpatialErrorRegressionModel{T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    lambda::T
    lambda_se::T
    sigma2::T
    loglik::T
    W::M
    X::Matrix{T}
    y::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

# ─── Panel SEM result ─────────────────────────────────────────────────

"""
    SpatialErrorPanelModel

A fitted spatial error panel, returned by the panel method of
[`fit_spatial_error`](@ref).

As [`SpatialErrorRegressionModel`](@ref), with the fixed-effects
specification and the panel dimensions `N` and `T_per` recorded alongside.
"""
struct SpatialErrorPanelModel{FE<:FixedEffectsSpec, T<:AbstractFloat, M<:AbstractSparseMatrix{T}} <: StatsAPI.RegressionModel
    fe_spec::FE
    N::Int
    T_periods::Int
    N_eff::Int
    T_eff::Int
    coef::Vector{T}
    vcov::Matrix{T}
    lambda::T
    lambda_se::T
    sigma2::T
    loglik::T
    W::M
    X_tilde::Matrix{T}
    y_tilde::Vector{T}
    residuals::Vector{T}
    coefnames::Vector{String}
end

# ─── Cross-section SEM estimation ─────────────────────────────────────

"""
    fit_spatial_error(W, X, y; method, varnames, lambda_bounds)
    fit_spatial_error(fe_spec, W, X, y, N, T_per; method, varnames, lambda_bounds)

Fit a spatial error model by concentrated maximum likelihood.

```math
y = X*beta + u,  u = lambda*W*u + e
```

Dependence is in the disturbance rather than in `y`. OLS stays unbiased
under this process but its standard errors are wrong, so the gain is
inferential: `beta` is interpretable as it stands, with no spillover
decomposition needed.

The likelihood is concentrated by spatially filtering `y` and `X` with
`(I - lambda*W)` at each trial `lambda`, then maximizing over `lambda`
alone.

The first method fits a cross-section; the second fits a panel, taking a
[`FixedEffectsSpec`](@ref) and the panel dimensions, with `X` and `y`
stacked units-fastest.

# Arguments
- `W`: `N x N` sparse weight matrix.
- `X`: covariates without an intercept — one is added.
- `y`: response.

# Keywords
- `method`: a [`LogDetMethod`](@ref).
- `varnames`: covariate names.
- `lambda_bounds`: search interval for `lambda`, default
  `(-0.995, 0.995)`.

# Returns
A [`SpatialErrorRegressionModel`](@ref) or
[`SpatialErrorPanelModel`](@ref).

# Examples
```julia
m = fit_spatial_error(W, X, y; varnames = ["income", "density"])
m.lambda
coef(m)                              # directly interpretable

fit_spatial_error(TwoWayFE(), W, X, y, N, T_per)
```

Use [`lm_tests`](@ref) to decide between this and [`fit_spatial`](@ref)
before fitting either.

# References
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
"""
function fit_spatial_error(
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T};
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995))
) where {T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    N, K = size(X)
    @assert size(W) == (N, N) "Dimension mismatch between W and X"
    @assert length(y) == N "Dimension mismatch between y and X"

    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    X_aug = hcat(ones(T, N), X)
    coefnames_vec = vcat(["(Intercept)"], vnames, ["λ"])

    Wy = W * y
    WX = W * X_aug

    # Concentrated ML: for each λ, filter y_λ = y - λWy, X_λ = X - λWX
    # β(λ) = X_λ \ y_λ, σ²(λ) = ||y_λ - X_λ β(λ)||² / N
    qr_X = qr(X_aug)
    qr_WX = qr(WX)

    β0 = qr_X \ y
    βd = qr_X \ Wy
    Xβ0_w = qr_X \ (WX * β0)  # not needed directly

    # More efficient: precompute components for concentrated LL
    # y_λ = y - λ Wy, X_λ = X_aug - λ WX
    # β(λ) = (X_λ'X_λ)^{-1} X_λ'y_λ
    # Easier: use the Anselin (1988) trick
    # e_0 = OLS resid from y on X, e_d = OLS resid from Wy on X
    # Then resid(λ) = e_0 - λ e_d
    e0_y = y .- X_aug * (qr_X \ y)
    ed_y = Wy .- X_aug * (qr_X \ Wy)

    # But SEM filtering is different from SAR. We need:
    # y_λ = y - λWy, X_λ = X - λWX, then OLS on (X_λ, y_λ)
    # The Anselin trick: define a0 = X\y, ad = X\(Wy), b0 = X\(WX * a0), ...
    # Actually for SEM the concentrated LL approach is simpler:
    # For each λ, compute y_λ, X_λ, then β_λ = X_λ\y_λ, σ²_λ, LL_λ.
    # Since brent_maximize only evaluates ~50 times, this is fast.

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = λ -> chebyshev_logdet(λ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        logdet_fn = λ -> first(logabsdet(lu(I_sparse - λ * W)))
    end

    const_term = -T(0.5) * N * (log(T(2π)) + one(T))

    function concentrated_sem_ll(λ)
        y_filt = y .- λ .* Wy
        X_filt = X_aug .- λ .* WX
        β_filt = X_filt \ y_filt
        res_filt = y_filt .- X_filt * β_filt
        σ2_filt = dot(res_filt, res_filt) / N
        return const_term + logdet_fn(λ) - T(0.5) * N * log(σ2_filt)
    end

    λ_hat, max_ll = brent_maximize(concentrated_sem_ll, lambda_bounds[1], lambda_bounds[2])

    # Final estimates at λ_hat
    y_filt = y .- λ_hat .* Wy
    X_filt = X_aug .- λ_hat .* WX
    β_hat = X_filt \ y_filt
    res = y_filt .- X_filt * β_hat
    σ2_hat = dot(res, res) / N

    # Asymptotic variance
    vcov_mat, λ_se = _sem_asymptotic_vcov(X_filt, W, λ_hat, σ2_hat, N)
    full_coef = vcat(β_hat, λ_hat)

    return SpatialErrorRegressionModel{T, M}(
        full_coef, vcov_mat, λ_hat, λ_se, σ2_hat, max_ll,
        W, Matrix{T}(X_aug), Vector{T}(y), res, coefnames_vec
    )
end

function _sem_asymptotic_vcov(
    X_filt::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    λ::T,
    σ2::T,
    N::Int
) where {T<:AbstractFloat}
    P = size(X_filt, 2)
    I_sparse = sparse(one(T) * I, N, N)
    B_lu = lu(I_sparse - λ * W)

    # tr(G) where G = W(I - λW)^{-1}
    n_p = 30
    V_probe = rand((-one(T), one(T)), N, n_p)
    GV = W * (B_lu \ V_probe)
    tr_G = tr(GV' * V_probe) / n_p

    # tr(G² + G'G)
    GGV = W * (B_lu \ GV)
    tr_G2 = tr(GV' * GV) / n_p  # tr(G'G)
    tr_GG = tr(V_probe' * GGV) / n_p  # tr(G²)
    tr_G2_GtG = tr_GG + tr_G2

    I_ββ = (X_filt' * X_filt) ./ σ2
    I_λλ = tr_G2_GtG + tr_G^2 / σ2  # simplified: just use trace terms
    # More precisely: I_λλ = tr(G² + G'G)
    I_λλ = tr_G2_GtG
    I_λσ2 = tr_G / σ2
    I_σ2σ2 = N / (2 * σ2^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_ββ
    # β and λ are asymptotically independent in SEM
    Info[P + 1, P + 1] = I_λλ
    Info[P + 1, P + 2] = I_λσ2
    Info[P + 2, P + 1] = I_λσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))
    V_coef = V_full[1:(P + 1), 1:(P + 1)]
    λ_se = sqrt(max(zero(T), V_coef[P + 1, P + 1]))

    return V_coef, λ_se
end

# ─── Panel SEM estimation ─────────────────────────────────────────────

function fit_spatial_error(
    fe_spec::FE,
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_per::Int;
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995))
) where {FE<:FixedEffectsSpec, T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    NT, K = size(X)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    if fe_spec isa NoFE
        X_aug = hcat(ones(T, NT), X)
        coefnames_vec = vcat(["(Intercept)"], vnames, ["λ"])
    else
        X_aug = copy(X)
        coefnames_vec = vcat(vnames, ["λ"])
    end

    # Spatial lag of y and X for filtering
    Wy = similar(y); panel_spatial_lag!(Wy, W, y, N, T_per)
    WX = panel_spatial_lag(W, X_aug, N, T_per)

    # Within-transform
    y_tilde = within_transform!(copy(y), N, T_per, fe_spec)
    Wy_tilde = within_transform!(copy(Wy), N, T_per, fe_spec)
    X_tilde = within_transform(X_aug, N, T_per, fe_spec)
    WX_tilde = within_transform(WX, N, T_per, fe_spec)

    N_eff, T_eff = effective_dims(N, T_per, fe_spec)
    NT_eff = N_eff * T_eff

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = λ -> chebyshev_logdet(λ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        logdet_fn = λ -> first(logabsdet(lu(I_sparse - λ * W)))
    end

    const_term = -T(0.5) * NT_eff * (log(T(2π)) + one(T))

    function concentrated_panel_sem_ll(λ)
        y_filt = y_tilde .- λ .* Wy_tilde
        X_filt = X_tilde .- λ .* WX_tilde
        β_filt = X_filt \ y_filt
        res_filt = y_filt .- X_filt * β_filt
        σ2_filt = dot(res_filt, res_filt) / NT_eff
        ld = logdet_fn(λ)
        if has_time_fe(fe_spec)
            ld -= log(one(λ) - λ)
        end
        return const_term + T_eff * ld - T(0.5) * NT_eff * log(σ2_filt)
    end

    λ_hat, max_ll = brent_maximize(concentrated_panel_sem_ll, lambda_bounds[1], lambda_bounds[2])

    y_filt = y_tilde .- λ_hat .* Wy_tilde
    X_filt = X_tilde .- λ_hat .* WX_tilde
    β_hat = X_filt \ y_filt
    res = y_filt .- X_filt * β_hat
    σ2_hat = dot(res, res) / NT_eff

    vcov_mat, λ_se = _sem_panel_vcov(X_filt, W, λ_hat, σ2_hat, N, T_per, N_eff, T_eff, fe_spec)
    full_coef = vcat(β_hat, λ_hat)

    return SpatialErrorPanelModel{FE, T, M}(
        fe_spec, N, T_per, N_eff, T_eff,
        full_coef, vcov_mat, λ_hat, λ_se, σ2_hat, max_ll,
        W, X_tilde, y_tilde, res, coefnames_vec
    )
end

function _sem_panel_vcov(
    X_filt::AbstractMatrix{T},
    W::AbstractSparseMatrix{T},
    λ::T,
    σ2::T,
    N::Int,
    T_per::Int,
    N_eff::Int,
    T_eff::Int,
    fe_spec::FixedEffectsSpec
) where {T<:AbstractFloat}
    P = size(X_filt, 2)
    NT_eff = N_eff * T_eff
    I_sparse = sparse(one(T) * I, N, N)
    B_lu = lu(I_sparse - λ * W)

    n_p = 30
    V_probe = rand((-one(T), one(T)), N, n_p)
    has_time_fe(fe_spec) && (V_probe .-= mean(V_probe; dims=1))
    GV = W * (B_lu \ V_probe)
    has_time_fe(fe_spec) && (GV .-= mean(GV; dims=1))

    tr_GtG = dot(GV, GV) / n_p
    GGV = W * (B_lu \ GV)
    has_time_fe(fe_spec) && (GGV .-= mean(GGV; dims=1))
    tr_G2 = dot(V_probe, GGV) / n_p

    tr_G = zero(T)
    # Stochastic trace of G
    V_id = rand((-one(T), one(T)), N, n_p)
    GV_id = W * (B_lu \ V_id)
    tr_G = dot(V_id, GV_id) / n_p
    if has_time_fe(fe_spec)
        tr_G -= one(T) / (one(T) - λ)
    end

    I_ββ = (X_filt' * X_filt) ./ σ2
    I_λλ = T_eff * (tr_G2 + tr_GtG)
    I_λσ2 = (T_eff * tr_G) / σ2
    I_σ2σ2 = NT_eff / (2 * σ2^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_ββ
    Info[P + 1, P + 1] = I_λλ
    Info[P + 1, P + 2] = I_λσ2
    Info[P + 2, P + 1] = I_λσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))
    V_coef = V_full[1:(P + 1), 1:(P + 1)]
    λ_se = sqrt(max(zero(T), V_coef[P + 1, P + 1]))

    return V_coef, λ_se
end

# ─── Show methods ─────────────────────────────────────────────────────

function Base.show(io::IO, m::SpatialErrorRegressionModel)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Spatial Error Model (SEM) — Maximum Likelihood")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "λ = %.4f    σ² = %.4f    Log-lik = %.2f    N = %d\n", m.lambda, m.sigma2, m.loglik, length(m.y))
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

function Base.show(io::IO, m::SpatialErrorPanelModel{FE}) where {FE}
    fname = FE === NoFE ? "None" : FE === IndivFE ? "Individual" : FE === TimeFE ? "Time" : "Two-Way"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Spatial Error Panel Model (SEM) — Maximum Likelihood — FE: $fname")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "λ = %.4f    σ² = %.4f    Log-lik = %.2f    N = %d    T = %d\n", m.lambda, m.sigma2, m.loglik, m.N, m.T_periods)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

# ─── StatsAPI interface ───────────────────────────────────────────────

StatsAPI.coef(m::SpatialErrorRegressionModel) = m.coef
StatsAPI.vcov(m::SpatialErrorRegressionModel) = m.vcov
StatsAPI.stderror(m::SpatialErrorRegressionModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::SpatialErrorRegressionModel) = m.loglik
StatsAPI.nobs(m::SpatialErrorRegressionModel) = length(m.y)
StatsAPI.dof_residual(m::SpatialErrorRegressionModel) = length(m.y) - length(m.coef)
StatsAPI.residuals(m::SpatialErrorRegressionModel) = m.residuals
StatsAPI.coefnames(m::SpatialErrorRegressionModel) = m.coefnames

StatsAPI.coef(m::SpatialErrorPanelModel) = m.coef
StatsAPI.vcov(m::SpatialErrorPanelModel) = m.vcov
StatsAPI.stderror(m::SpatialErrorPanelModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::SpatialErrorPanelModel) = m.loglik
StatsAPI.nobs(m::SpatialErrorPanelModel) = m.N * m.T_periods
StatsAPI.dof_residual(m::SpatialErrorPanelModel) = (m.N_eff * m.T_eff) - length(m.coef)
StatsAPI.residuals(m::SpatialErrorPanelModel) = m.residuals
StatsAPI.coefnames(m::SpatialErrorPanelModel) = m.coefnames

end # module
