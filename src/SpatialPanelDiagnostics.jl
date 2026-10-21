module SpatialPanelDiagnostics

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: coef, vcov, nobs, dof_residual, coefnames

import ..SpatialUtils: _normcdf, _chi2_sf, _loggamma, FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
    within_transform, within_transform!, panel_spatial_lag!
import ..SpatialDiagnostics: MoranResult, moran_global

export
    HausmanTestResult, spatial_hausman_test,
    ChowTestResult, spatial_chow_test,
    PanelLMTestResult, panel_lm_tests,
    PanelMoranResult, panel_moran

# ─── Spatial Hausman Test ─────────────────────────────────────────────

struct HausmanTestResult
    stat::Float64
    df::Int
    p_value::Float64
    coefnames::Vector{String}
end

function Base.show(io::IO, r::HausmanTestResult)
    println(io, "Spatial Hausman Test (FE vs RE)")
    println(io, "───────────────────────────────")
    @printf(io, "  Hausman statistic: %.4f\n", r.stat)
    @printf(io, "  Degrees of freedom: %d\n", r.df)
    println(io, "  Compared:          ", join(r.coefnames, ", "))
    @printf(io, "  p-value:           %.6f %s\n", r.p_value,
        r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : "")
    verdict = r.p_value < 0.05 ? "Reject H₀ → FE preferred" : "Fail to reject H₀ → RE consistent"
    println(io, "  $verdict")
end

"""
    spatial_hausman_test(m_fe, m_re; names=nothing)

Spatial Hausman test comparing two spatial panel estimators.
Under H₀ the second model is consistent and efficient; under H₁ only the
first is consistent, so a rejection favours `m_fe`.

Coefficients are matched **by name**, using the names the two models share,
in the order they appear in `m_fe`. The names actually compared are returned
in the result. Pass `names` to restrict the comparison to a chosen subset.
The models should otherwise share the same dependent variable, covariates
and `W`.

Matching by name matters. A `NoFE` panel fit carries an intercept that an FE
fit does not, so the first position is the intercept in one model and the
first slope in the other. Comparing those positionally builds a statistic
out of unrelated parameters: it looks like a decisive rejection and means
nothing.
"""
function spatial_hausman_test(
    m_fe::StatsAPI.RegressionModel,
    m_re::StatsAPI.RegressionModel;
    names::Union{Nothing, AbstractVector{<:AbstractString}}=nothing
)
    nm_fe = String.(coefnames(m_fe))
    nm_re = String.(coefnames(m_re))

    shared = if isnothing(names)
        [n for n in nm_fe if n in nm_re]
    else
        req = String.(names)
        for n in req
            n in nm_fe ||
                error("coefficient $(n) is not in the first model (has: $(nm_fe))")
            n in nm_re ||
                error("coefficient $(n) is not in the second model (has: $(nm_re))")
        end
        req
    end
    unique!(shared)
    isempty(shared) && error(
        "the two models share no coefficient names, so there is nothing to " *
        "compare: $(nm_fe) vs $(nm_re)")

    i_fe = Int[findfirst(==(n), nm_fe) for n in shared]
    i_re = Int[findfirst(==(n), nm_re) for n in shared]

    β_fe_c = coef(m_fe)[i_fe]
    β_re_c = coef(m_re)[i_re]

    V_fe = vcov(m_fe)[i_fe, i_fe]
    V_re = vcov(m_re)[i_re, i_re]

    V_diff = V_fe .- V_re
    V_diff_sym = Symmetric(V_diff)

    eig_vals = eigvals(V_diff_sym)
    if any(eig_vals .< -1e-8)
        V_diff_sym = Symmetric(V_diff + max(0.0, -minimum(eig_vals) + 1e-6) * I)
    end

    diff = β_fe_c .- β_re_c
    V_inv = try
        inv(V_diff_sym)
    catch
        pinv(V_diff_sym)
    end

    stat = dot(diff, V_inv * diff)
    stat = max(stat, 0.0)
    df = length(shared)
    p_value = _chi2_sf(stat, df)

    return HausmanTestResult(stat, df, p_value, shared)
end

# ─── Spatial Chow Test ───────────────────────────────────────────────

struct ChowTestResult
    stat::Float64
    df::Int
    p_value::Float64
    break_period::Int
end

function Base.show(io::IO, r::ChowTestResult)
    println(io, "Spatial Chow Test (Structural Break)")
    println(io, "────────────────────────────────────")
    @printf(io, "  F-statistic:  %.4f\n", r.stat)
    @printf(io, "  Degrees of freedom: (%d, pooled_dof)\n", r.df)
    @printf(io, "  p-value:      %.6f %s\n", r.p_value,
        r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : "")
    @printf(io, "  Break period: %d\n", r.break_period)
end

"""
    spatial_chow_test(y, X, W, N, T, break_period)

Spatial Chow test for structural break at `break_period`.
Tests whether parameters are stable across two sub-periods.
Uses the LR-based approach adapted for spatial models.
"""
function spatial_chow_test(
    y::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    W::AbstractSparseMatrix,
    N::Int, T_per::Int, break_period::Int
)
    @assert 1 < break_period < T_per "break_period must be in (1, T)"
    NT = N * T_per
    @assert length(y) == NT && size(X, 1) == NT

    K = size(X, 2)

    ssr_pooled = _spatial_ols_ssr(y, X, W, N, T_per)

    idx1 = vcat([((t-1)*N+1):(t*N) for t in 1:break_period]...)
    idx2 = vcat([((t-1)*N+1):(t*N) for t in (break_period+1):T_per]...)

    T1 = break_period; T2 = T_per - break_period
    ssr1 = _spatial_ols_ssr(y[idx1], X[idx1, :], W, N, T1)
    ssr2 = _spatial_ols_ssr(y[idx2], X[idx2, :], W, N, T2)

    ssr_sum = ssr1 + ssr2
    df_num = K
    df_den = NT - 2 * K

    stat = ((ssr_pooled - ssr_sum) / df_num) / (ssr_sum / df_den)
    stat = max(stat, 0.0)

    p_value = _f_sf(stat, df_num, df_den)

    return ChowTestResult(stat, df_num, p_value, break_period)
end

function _spatial_ols_ssr(y, X, W, N, T_per)
    NT = length(y)
    Wy = zeros(NT)
    for t in 1:T_per
        idx = ((t-1)*N+1):min(t*N, NT)
        actual_n = length(idx)
        if actual_n == N
            @views panel_spatial_lag!(Wy[idx], W, y[idx], N, 1)
        end
    end
    X_aug = hcat(X, Wy)
    β = X_aug \ Vector{Float64}(y)
    resid = y .- X_aug * β
    return dot(resid, resid)
end

# ─── Joint LM Tests for Panels ──────────────────────────────────────

struct PanelLMTestResult
    joint_stat::Float64
    joint_df::Int
    joint_p::Float64
    lag_stat::Float64
    lag_p::Float64
    error_stat::Float64
    error_p::Float64
    re_stat::Float64
    re_p::Float64
end

function Base.show(io::IO, r::PanelLMTestResult)
    println(io, "Joint LM Tests for Spatial Panels")
    println(io, "─────────────────────────────────")
    @printf(io, "  Joint test:    LM = %.4f  (df=%d, p=%.6f) %s\n",
        r.joint_stat, r.joint_df, r.joint_p,
        r.joint_p < 0.05 ? "*" : "")
    @printf(io, "  Spatial lag:   LM = %.4f  (df=1, p=%.6f) %s\n",
        r.lag_stat, r.lag_p, r.lag_p < 0.05 ? "*" : "")
    @printf(io, "  Spatial error: LM = %.4f  (df=1, p=%.6f) %s\n",
        r.error_stat, r.error_p, r.error_p < 0.05 ? "*" : "")
    @printf(io, "  Random effects:LM = %.4f  (df=1, p=%.6f) %s\n",
        r.re_stat, r.re_p, r.re_p < 0.05 ? "*" : "")
end

"""
    panel_lm_tests(y, X, W, N, T)

Joint LM tests for spatial panels (Baltagi, Song & Koh 2003).
Tests simultaneously for spatial lag, spatial error, and random effects
from a pooled OLS baseline.
Returns individual and joint test statistics.
"""
function panel_lm_tests(
    y::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real},
    W::AbstractSparseMatrix,
    N::Int, T_per::Int
)
    NT = N * T_per
    @assert length(y) == NT && size(X, 1) == NT

    β_ols = X \ Vector{Float64}(y)
    e = y .- X * β_ols
    σ² = dot(e, e) / NT

    Wy = zeros(NT)
    We = zeros(NT)
    for t in 1:T_per
        idx = ((t-1)*N+1):(t*N)
        @views panel_spatial_lag!(Wy[idx], W, y[idx], N, 1)
        @views panel_spatial_lag!(We[idx], W, e[idx], N, 1)
    end

    T_W = _trace_WpW(W)

    eWe = dot(e, We)
    LM_error = (eWe / σ²)^2 / (N * T_per * T_W)
    LM_error = max(LM_error, 0.0)

    eWy = dot(e, Wy)
    J = (Wy' * (I - X * ((X' * X) \ X')) * Wy) / σ²
    J = max(J, eps())
    LM_lag = (eWy / σ²)^2 / J
    LM_lag = max(LM_lag, 0.0)

    E_mat = reshape(e, N, T_per)
    e_bar_i = vec(mean(E_mat, dims=2))
    sum_sq_bar = T_per^2 * dot(e_bar_i, e_bar_i)
    LM_re = (sum_sq_bar / (dot(e, e)) - 1.0)^2 * NT / (2.0 * (T_per - 1))
    LM_re = max(LM_re, 0.0)

    LM_joint = LM_lag + LM_error + LM_re
    joint_df = 3

    return PanelLMTestResult(
        LM_joint, joint_df, _chi2_sf(LM_joint, joint_df),
        LM_lag, _chi2_sf(LM_lag, 1),
        LM_error, _chi2_sf(LM_error, 1),
        LM_re, _chi2_sf(LM_re, 1)
    )
end

function _trace_WpW(W::AbstractSparseMatrix)
    WpW = W' * W + W * W
    return tr(WpW) / size(W, 1)
end

# ─── Panel Moran's I ─────────────────────────────────────────────────

struct PanelMoranResult
    period_results::Vector{MoranResult}
    pooled::MoranResult
    T_periods::Int
end

function Base.show(io::IO, r::PanelMoranResult)
    println(io, "Panel Moran's I")
    println(io, "───────────────")
    for (t, mr) in enumerate(r.period_results)
        sig = mr.p_value < 0.001 ? "***" : mr.p_value < 0.01 ? "**" : mr.p_value < 0.05 ? "*" : ""
        @printf(io, "  Period %2d: I = %7.4f  (z = %6.3f, p = %.4f) %s\n",
            t, mr.I_stat, mr.z_stat, mr.p_value, sig)
    end
    sig = r.pooled.p_value < 0.001 ? "***" : r.pooled.p_value < 0.01 ? "**" : r.pooled.p_value < 0.05 ? "*" : ""
    @printf(io, "  Pooled:    I = %7.4f  (z = %6.3f, p = %.4f) %s\n",
        r.pooled.I_stat, r.pooled.z_stat, r.pooled.p_value, sig)
end

"""
    panel_moran(residuals, W, N, T)

Compute Moran's I for each period and a pooled statistic from panel residuals.
"""
function panel_moran(
    resids::AbstractVector{<:Real},
    W::AbstractSparseMatrix,
    N::Int, T_per::Int
)
    NT = N * T_per
    @assert length(resids) == NT

    period_results = MoranResult[]
    for t in 1:T_per
        idx = ((t-1)*N+1):(t*N)
        push!(period_results, moran_global(resids[idx], W))
    end

    pooled = moran_global(resids[1:N], W)
    if T_per > 1
        I_vals = [mr.I_stat for mr in period_results]
        z_vals = [mr.z_stat for mr in period_results]
        avg_I = mean(I_vals)
        avg_z = mean(z_vals)
        meta_z = avg_z * sqrt(T_per)
        meta_p = 2.0 * (1.0 - _normcdf(abs(meta_z)))
        pooled = MoranResult(avg_I, pooled.E_I, pooled.Var_I, meta_z, meta_p, :panel_meta)
    end

    return PanelMoranResult(period_results, pooled, T_per)
end

# ─── F survival function (regularized incomplete beta) ───────────────
# The chi-squared survival function and log Gamma now live in SpatialUtils.

function _f_sf(x::Real, df1::Int, df2::Int)
    x <= 0.0 && return 1.0
    _regularized_beta_upper(df1 / 2.0, df2 / 2.0, df1 * x / (df1 * x + df2))
end

function _regularized_beta_upper(a::Real, b::Real, x::Real)
    x <= 0.0 && return 1.0
    x >= 1.0 && return 0.0
    1.0 - _regularized_beta(a, b, x)
end

function _regularized_beta(a::Real, b::Real, x::Real)
    x <= 0.0 && return 0.0
    x >= 1.0 && return 1.0
    bt = exp(_loggamma(a + b) - _loggamma(a) - _loggamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0)
        bt * _beta_cf(a, b, x) / a
    else
        1.0 - bt * _beta_cf(b, a, 1.0 - x) / b
    end
end

function _beta_cf(a::Real, b::Real, x::Real)
    qab = a + b; qap = a + 1.0; qam = a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap; abs(d) < 1e-30 && (d = 1e-30); d = 1.0 / d; h = d
    for m in 1:200
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; abs(d) < 1e-30 && (d = 1e-30); c = 1.0 + aa / c; abs(c) < 1e-30 && (c = 1e-30)
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; abs(d) < 1e-30 && (d = 1e-30); c = 1.0 + aa / c; abs(c) < 1e-30 && (c = 1e-30)
        d = 1.0 / d; del = d * c; h *= del
        abs(del - 1.0) < 1e-12 && break
    end
    return h
end

end # module
