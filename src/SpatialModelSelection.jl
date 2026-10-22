module SpatialModelSelection

using LinearAlgebra
using Statistics
using Printf
using StatsAPI
import StatsAPI: loglikelihood, nobs, dof_residual, coef, vcov

import ..SpatialUtils: _normcdf, _chi2_sf

export
    spatial_aic, spatial_bic,
    LRTestResult, lr_test,
    VuongTestResult, vuong_test,
    CommonFactorTestResult, common_factor_test,
    WaldTestResult, wald_test,
    model_comparison_table

# ─── Information Criteria ─────────────────────────────────────────────

"""
    spatial_aic(m)

Akaike Information Criterion: `AIC = -2 ln L + 2k`
where k is the number of estimated parameters.
"""
function spatial_aic(m::StatsAPI.RegressionModel)
    ll = loglikelihood(m)
    k = nobs(m) - dof_residual(m)
    return -2.0 * ll + 2.0 * k
end

"""
    spatial_bic(m)

Bayesian Information Criterion: `BIC = -2 ln L + k ln(n)`
"""
function spatial_bic(m::StatsAPI.RegressionModel)
    ll = loglikelihood(m)
    n = nobs(m)
    k = n - dof_residual(m)
    return -2.0 * ll + k * log(n)
end

# ─── Likelihood Ratio Test ───────────────────────────────────────────

"""
    LRTestResult

Likelihood-ratio test between nested models, returned by
[`lr_test`](@ref).

Fields: `stat`, `df`, `p_value`, and both log-likelihoods
(`ll_restricted`, `ll_unrestricted`).

!!! warning
    Valid only when one model genuinely nests the other and both were fitted
    to the same data by maximum likelihood. For non-nested comparisons use
    [`vuong_test`](@ref).
"""
struct LRTestResult
    stat::Float64
    df::Int
    p_value::Float64
    ll_restricted::Float64
    ll_unrestricted::Float64
end

function Base.show(io::IO, r::LRTestResult)
    println(io, "Likelihood Ratio Test")
    println(io, "─────────────────────")
    @printf(io, "  LR statistic:  %.4f\n", r.stat)
    @printf(io, "  Degrees of freedom: %d\n", r.df)
    @printf(io, "  p-value:       %.6f %s\n", r.p_value,
        r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : "")
    @printf(io, "  LL (restricted):    %.4f\n", r.ll_restricted)
    @printf(io, "  LL (unrestricted):  %.4f\n", r.ll_unrestricted)
end

"""
    lr_test(m_restricted, m_unrestricted)

Likelihood ratio test for nested models. The unrestricted model must have
more parameters than the restricted model.
"""
function lr_test(m_restricted::StatsAPI.RegressionModel, m_unrestricted::StatsAPI.RegressionModel)
    ll_r = loglikelihood(m_restricted)
    ll_u = loglikelihood(m_unrestricted)

    k_r = nobs(m_restricted) - dof_residual(m_restricted)
    k_u = nobs(m_unrestricted) - dof_residual(m_unrestricted)
    df = k_u - k_r
    df <= 0 && error("Unrestricted model must have more parameters than restricted model")

    stat = 2.0 * (ll_u - ll_r)
    stat = max(stat, 0.0)
    p_value = _chi2_sf(stat, df)

    return LRTestResult(stat, df, p_value, ll_r, ll_u)
end

# ─── Vuong Test (Non-Nested) ─────────────────────────────────────────

"""
    VuongTestResult

Vuong test between two non-nested models, returned by
[`vuong_test`](@ref).

Fields: `stat`, `p_value`, `aic_correction`, and `preference`, which is
`:model1`, `:model2` or `:indistinguishable`.

Use this where [`lr_test`](@ref) does not apply — comparing a spatial lag
against a spatial error specification, for instance, since neither nests
the other. `:indistinguishable` is a genuine outcome, not a failure: the
data may simply not separate the two.
"""
struct VuongTestResult
    stat::Float64
    p_value::Float64
    preference::Symbol   # :model1, :model2, :indistinguishable
    aic_correction::Float64
end

function Base.show(io::IO, r::VuongTestResult)
    println(io, "Vuong Non-Nested Model Test")
    println(io, "───────────────────────────")
    @printf(io, "  Vuong statistic:   %.4f\n", r.stat)
    @printf(io, "  p-value (2-sided): %.6f\n", r.p_value)
    @printf(io, "  AIC correction:    %.4f\n", r.aic_correction)
    pref = r.preference == :model1 ? "Model 1" :
           r.preference == :model2 ? "Model 2" : "Indistinguishable"
    @printf(io, "  Preference:        %s\n", pref)
end

"""
    vuong_test(m1, m2)

Vuong (1989) test for non-nested model comparison.
Uses AIC-corrected version. Positive statistic favors model 1.
"""
function vuong_test(m1::StatsAPI.RegressionModel, m2::StatsAPI.RegressionModel)
    n1 = nobs(m1); n2 = nobs(m2)
    n1 == n2 || error("Both models must be estimated on the same sample (nobs differ: $n1 vs $n2)")
    n = n1

    ll1 = loglikelihood(m1); ll2 = loglikelihood(m2)
    k1 = n - dof_residual(m1); k2 = n - dof_residual(m2)

    aic_correction = (k1 - k2) / n

    diff = (ll1 - ll2) / n
    adj_diff = diff - aic_correction

    omega = sqrt(abs(2.0 * (ll1 - ll2) / n - adj_diff^2) + eps())

    stat = sqrt(Float64(n)) * adj_diff / omega

    p_value = 2.0 * (1.0 - _normcdf(abs(stat)))

    preference = if abs(stat) < 1.96
        :indistinguishable
    elseif stat > 0
        :model1
    else
        :model2
    end

    return VuongTestResult(stat, p_value, preference, aic_correction)
end

# ─── Common Factor Test ──────────────────────────────────────────────

"""
    CommonFactorTestResult

Test of the common-factor restriction, returned by
[`common_factor_test`](@ref).

Fields: `stat`, `df`, `p_value`, `reject`.

A Spatial Durbin model reduces to a spatial error model exactly when
`theta = -rho * beta`. Failing to reject means the simpler error
specification is adequate; rejecting means the Durbin terms carry
information the error model cannot represent.

See also `SDM`, `fit_spatial_error`.
"""
struct CommonFactorTestResult
    stat::Float64
    df::Int
    p_value::Float64
    reject::Bool
end

function Base.show(io::IO, r::CommonFactorTestResult)
    println(io, "Common Factor Test (SDM → SEM restriction)")
    println(io, "───────────────────────────────────────────")
    @printf(io, "  Wald statistic: %.4f\n", r.stat)
    @printf(io, "  Degrees of freedom: %d\n", r.df)
    @printf(io, "  p-value:        %.6f %s\n", r.p_value,
        r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : "")
    verdict = r.reject ? "Reject H₀ → SDM preferred over SEM" : "Fail to reject H₀ → SEM may be adequate"
    println(io, "  $verdict")
end

"""
    common_factor_test(coef_beta, coef_theta, rho, vcov_full)

Tests whether SDM simplifies to SEM, i.e. H₀: θ + ρβ = 0.
- `coef_beta`: vector of K slope coefficients (excluding intercept)
- `coef_theta`: vector of K spatial lag coefficients (WX terms)
- `rho`: spatial autoregressive parameter
- `vcov_full`: full variance-covariance matrix of [β; θ; ρ]

Uses the Wald test formulation from Burridge (1981).
"""
function common_factor_test(
    coef_beta::AbstractVector{<:Real},
    coef_theta::AbstractVector{<:Real},
    rho::Real,
    vcov_full::AbstractMatrix{<:Real}
)
    K = length(coef_beta)
    @assert length(coef_theta) == K "beta and theta must have same length"

    g = coef_theta .+ rho .* coef_beta

    n_params = size(vcov_full, 1)
    G = zeros(K, n_params)
    beta_start = 2  # skip intercept
    theta_start = beta_start + K
    rho_idx = n_params

    for k in 1:K
        G[k, beta_start + k - 1] = rho
        G[k, theta_start + k - 1] = 1.0
        G[k, rho_idx] = coef_beta[k]
    end

    V_g = G * vcov_full * G'
    V_g_inv = try
        inv(Symmetric(V_g))
    catch
        pinv(V_g)
    end

    stat = dot(g, V_g_inv * g)
    stat = max(stat, 0.0)
    p_value = _chi2_sf(stat, K)

    return CommonFactorTestResult(stat, K, p_value, p_value < 0.05)
end

# ─── Wald Test ───────────────────────────────────────────────────────

"""
    WaldTestResult

Wald test of a linear restriction `R*beta = r`, returned by
[`wald_test`](@ref).

Fields: `stat`, `df`, `p_value`. Needs only the fitted model's coefficients
and covariance, so unlike [`lr_test`](@ref) it does not require refitting
under the restriction.
"""
struct WaldTestResult
    stat::Float64
    df::Int
    p_value::Float64
end

function Base.show(io::IO, r::WaldTestResult)
    println(io, "Wald Test (Rβ = r)")
    println(io, "───────────────────")
    @printf(io, "  Wald statistic: %.4f\n", r.stat)
    @printf(io, "  Degrees of freedom: %d\n", r.df)
    @printf(io, "  p-value:        %.6f %s\n", r.p_value,
        r.p_value < 0.001 ? "***" : r.p_value < 0.01 ? "**" : r.p_value < 0.05 ? "*" : "")
end

"""
    wald_test(m, R, r)

General Wald test: H₀: Rβ = r.
- `R`: Q×K restriction matrix
- `r`: Q-vector of restriction values
"""
function wald_test(m::StatsAPI.RegressionModel, R::AbstractMatrix{<:Real}, r::AbstractVector{<:Real})
    β = coef(m)
    V = vcov(m)
    Q = size(R, 1)
    @assert size(R, 2) == length(β) "R columns must match number of coefficients"
    @assert length(r) == Q "r length must match number of restrictions"

    diff = R * β .- r
    RVR = R * V * R'
    RVR_inv = try
        inv(Symmetric(RVR))
    catch
        pinv(RVR)
    end

    stat = dot(diff, RVR_inv * diff)
    stat = max(stat, 0.0)
    p_value = _chi2_sf(stat, Q)

    return WaldTestResult(stat, Q, p_value)
end

# ─── Model Comparison Table ──────────────────────────────────────────

"""
    model_comparison_table(models...; names=nothing)

Print a formatted comparison table of spatial models.
"""
function model_comparison_table(models::StatsAPI.RegressionModel...; names::Union{Nothing, Vector{String}}=nothing)
    n_models = length(models)
    if names === nothing
        names = ["Model $i" for i in 1:n_models]
    end
    @assert length(names) == n_models

    io = IOBuffer()
    println(io, "Model Comparison Table")
    println(io, "═" ^ 78)
    @printf(io, "%-15s | %8s | %8s | %10s | %10s | %10s\n",
        "Model", "n", "k", "Log-Lik", "AIC", "BIC")
    println(io, "─" ^ 78)

    for (i, m) in enumerate(models)
        n = nobs(m)
        k = n - dof_residual(m)
        ll = loglikelihood(m)
        aic = spatial_aic(m)
        bic = spatial_bic(m)
        @printf(io, "%-15s | %8d | %8d | %10.2f | %10.2f | %10.2f\n",
            names[i], n, k, ll, aic, bic)
    end
    println(io, "═" ^ 78)

    aics = [spatial_aic(m) for m in models]
    best_idx = argmin(aics)
    @printf(io, "Preferred (AIC): %s\n", names[best_idx])

    return String(take!(io))
end

end # module
