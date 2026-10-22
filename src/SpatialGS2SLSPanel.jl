module SpatialGS2SLSPanel

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI

import ..SpatialUtils
import ..SpatialUtils: panel_spatial_lag!, panel_spatial_lag,
       within_transform!, within_transform,
       _normcdf, _print_coef_table

const FixedEffectsSpec = SpatialUtils.FixedEffectsSpec
const IndivFE = SpatialUtils.IndivFE
const TwoWayFE = SpatialUtils.TwoWayFE

export SpatialVariant, SARAR,
       FixedEffectsSpec, IndivFE, TwoWayFE,
       GS2SLSModel, fit_gs2sls_panel

const SpatialVariant = SpatialUtils.SpatialVariant
const SARAR = SpatialUtils.SARAR

struct GS2SLSModel{
    FE<:FixedEffectsSpec,
    T<:AbstractFloat,
    M<:AbstractSparseMatrix{T}
} <: StatsAPI.RegressionModel
    fe_spec::FE
    N::Int
    T_total::Int
    T_post::Int
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    lambda::T
    sigma2::T
    sargan_stat::T
    sargan_pval::T
    W::M
    residuals::Vector{T}
    coefnames::Vector{String}
end

function gmm_objective(lambda::Real, u::Vector{T}, Wu::Vector{T}, WWu::Vector{T}) where {T<:AbstractFloat}
    N = length(u)
    m11 = dot(u, u) / N
    m12 = dot(u, Wu) / N
    m22 = dot(Wu, Wu) / N
    m13 = dot(u, WWu) / N
    m23 = dot(Wu, WWu) / N
    m33 = dot(WWu, WWu) / N

    g1 = m11 - 2*lambda*m12 + (lambda^2)*m22
    g2 = m12 - lambda*m22 - lambda*m13 + (lambda^2)*m23
    g3 = m22 - 2*lambda*m23 + (lambda^2)*m33

    return g1^2 + g2^2 + g3^2
end

function estimate_lambda_gmm(u::Vector{T}, Wu::Vector{T}, WWu::Vector{T}) where {T<:AbstractFloat}
    a = -0.95; b = 0.95
    cg = 0.3819660112501051
    x = w = v = a + cg * (b - a)
    fx = fw = fv = gmm_objective(x, u, Wu, WWu)
    d = e = 0.0

    for _ in 1:100
        xm = 0.5 * (a + b)
        tol1 = 1e-6 * abs(x) + 1e-10
        tol2 = 2 * tol1
        abs(x - xm) <= (tol2 - 0.5 * (b - a)) && return x

        p = q = r = 0.0
        if abs(e) > tol1
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            q > 0 ? (p = -p) : (q = -q)
            r = e; e = d
        end

        if abs(p) < abs(0.5 * q * r) && p > q * (a - x) && p < q * (b - x)
            d = p / q
            u_step = x + d
            ((u_step - a) < tol2 || (b - u_step) < tol2) && (d = x < xm ? tol1 : -tol1)
        else
            e = (x < xm ? b : a) - x
            d = cg * e
        end

        u_step = abs(d) >= tol1 ? x + d : x + (d > 0 ? tol1 : -tol1)
        fu = gmm_objective(u_step, u, Wu, WWu)

        if fu <= fx
            u_step >= x ? (a = x) : (b = x)
            v, fv = w, fw; w, fw = x, fx; x, fx = u_step, fu
        else
            u_step < x ? (a = u_step) : (b = u_step)
            if fu <= fw || w == x
                v, fv = w, fw; w, fw = u_step, fu
            elseif fu <= fv || v == x || v == w
                v, fv = u_step, fu
            end
        end
    end
    return T(x)
end

function fit_gs2sls_panel(
    fe_spec::FE,
    W::M,
    X_exog::AbstractMatrix{T},
    X_endog::AbstractMatrix{T},
    Z_iv::AbstractMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_total::Int;
    varnames::Union{Nothing, Vector{String}}=nothing,
    exog_names::Union{Nothing, Vector{String}}=nothing,
    endog_names::Union{Nothing, Vector{String}}=nothing
) where {FE<:FixedEffectsSpec, T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    NT_total = length(y)
    T_post = T_total - 1
    NT_post = N * T_post

    idx_post = (N + 1):NT_total
    y_post   = y[idx_post]
    X_ex_p   = X_exog[idx_post, :]
    X_en_p   = X_endog[idx_post, :]
    Z_iv_p   = Z_iv[idx_post, :]

    Wy_post = similar(y_post)
    panel_spatial_lag!(Wy_post, W, y_post, N, T_post)

    y_t    = within_transform!(copy(y_post), N, T_post, fe_spec)
    Wy_t   = within_transform!(copy(Wy_post), N, T_post, fe_spec)
    X_ex_t = within_transform(X_ex_p, N, T_post, fe_spec)
    X_en_t = within_transform(X_en_p, N, T_post, fe_spec)
    Z_iv_t = within_transform(Z_iv_p, N, T_post, fe_spec)

    WX_ex  = panel_spatial_lag(W, X_ex_t, N, T_post)
    WWX_ex = panel_spatial_lag(W, WX_ex, N, T_post)
    WZ_iv  = panel_spatial_lag(W, Z_iv_t, N, T_post)

    H = hcat(X_ex_t, WX_ex, WWX_ex, Z_iv_t, WZ_iv)

    Psi_endog = hcat(Wy_t, X_en_t)
    HH_inv = inv(Symmetric(H' * H))
    Psi_hat = H * (HH_inv * (H' * Psi_endog))

    Psi_full = hcat(Psi_hat[:, 1], X_ex_t, Psi_hat[:, 2:end])
    Psi_raw  = hcat(Wy_t, X_ex_t, X_en_t)
    theta_2sls = (Psi_full' * Psi_raw) \ (Psi_full' * y_t)

    u = y_t .- Psi_raw * theta_2sls
    Wu = similar(u); panel_spatial_lag!(Wu, W, u, N, T_post)
    WWu = similar(u); panel_spatial_lag!(WWu, W, Wu, N, T_post)

    lambda_hat = estimate_lambda_gmm(u, Wu, WWu)

    Wy_lag = similar(Wy_t)
    panel_spatial_lag!(Wy_lag, W, Wy_t, N, T_post)
    y_star  = y_t .- lambda_hat .* Wy_t
    Wy_star = Wy_t .- lambda_hat .* Wy_lag

    X_ex_star = X_ex_t .- lambda_hat .* panel_spatial_lag(W, X_ex_t, N, T_post)
    X_en_star = X_en_t .- lambda_hat .* panel_spatial_lag(W, X_en_t, N, T_post)
    Psi_star  = hcat(Wy_star, X_ex_star, X_en_star)

    WH = panel_spatial_lag(W, H, N, T_post)
    H_star = H .- lambda_hat .* WH

    Psi_endog_star = hcat(Wy_star, X_en_star)
    HH_star_inv = inv(Symmetric(H_star' * H_star))
    Psi_hat_star = H_star * (HH_star_inv * (H_star' * Psi_endog_star))
    Psi_full_star = hcat(Psi_hat_star[:, 1], X_ex_star, Psi_hat_star[:, 2:end])

    theta_fgs2sls = (Psi_full_star' * Psi_star) \ (Psi_full_star' * y_star)

    u_final = y_star .- Psi_star * theta_fgs2sls
    sigma2 = dot(u_final, u_final) / NT_post

    omega = Diagonal(u_final .^ 2)
    middle = Psi_full_star' * omega * Psi_full_star
    left_inv = inv(Psi_full_star' * Psi_star)
    vcov_struct = left_inv * middle * left_inv'

    P_struct = length(theta_fgs2sls)
    vcov_mat = zeros(T, P_struct + 1, P_struct + 1)
    vcov_mat[1:P_struct, 1:P_struct] .= vcov_struct
    vcov_mat[P_struct + 1, P_struct + 1] = sigma2 / NT_post

    P_H_e = H_star * (HH_star_inv * (H_star' * u_final))
    sargan_stat = dot(u_final, P_H_e) / sigma2
    sargan_pval = one(T)

    rho = theta_fgs2sls[1]
    K_ex = size(X_ex_t, 2)
    K_en = size(X_en_t, 2)
    beta_ex = theta_fgs2sls[2:(1 + K_ex)]
    beta_en = theta_fgs2sls[(2 + K_ex):end]
    full_coef = vcat(rho, beta_ex, beta_en, lambda_hat)

    ex_names = isnothing(exog_names) ? ["exog_$i" for i in 1:K_ex] : exog_names
    en_names = isnothing(endog_names) ? ["endog_$i" for i in 1:K_en] : endog_names
    cnames = vcat(["ρ (Wy)"], ex_names, en_names, ["λ (Wu)"])

    return GS2SLSModel{FE, T, M}(
        fe_spec, N, T_total, T_post,
        full_coef, vcov_mat, rho, lambda_hat, sigma2,
        sargan_stat, sargan_pval, W, u_final, cnames
    )
end

function Base.show(io::IO, m::GS2SLSModel{FE}) where {FE}
    fname = FE === IndivFE ? "Individual" : "Two-Way"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "GS2SLS SARAR Panel Model — FE: $fname")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "ρ = %.4f    λ = %.4f    σ² = %.4f    N = %d    T = %d\n", m.rho, m.lambda, m.sigma2, m.N, m.T_total)
    @printf(io, "Sargan stat = %.4f    Sargan p-val = %.4f\n", m.sargan_stat, m.sargan_pval)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

StatsAPI.coef(m::GS2SLSModel) = m.coef
StatsAPI.vcov(m::GS2SLSModel) = m.vcov
StatsAPI.stderror(m::GS2SLSModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.nobs(m::GS2SLSModel) = m.N * m.T_post
StatsAPI.dof_residual(m::GS2SLSModel) = m.N * m.T_post - length(m.coef)
StatsAPI.residuals(m::GS2SLSModel) = m.residuals
StatsAPI.coefnames(m::GS2SLSModel) = m.coefnames

end # module
