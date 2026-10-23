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

"""
    HausmanTestResult

Fixed- versus random-effects test for a spatial panel, returned by
[`spatial_hausman_test`](@ref).

Fields: `stat`, `df`, `p_value` and the `coefnames` compared. Rejection
favours fixed effects, because it says the unit effects are correlated with
the regressors and the random-effects estimator is therefore inconsistent.
"""
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

# When the statistic is not computable

The test inverts `V(m_fe) - V(m_re)`, which the Hausman algebra requires
to be positive definite — `m_re` must be the *efficient* estimator under
the null. Passing the arguments the other way round inverts the ordering,
and where there is a genuine efficiency ordering it leaves nothing to
invert.

Directions whose eigenvalue is not clearly positive are dropped and `df`
reports how many were kept, so `df` may be smaller than the number of
names compared. If no direction survives, `stat` and `p_value` are `NaN`
and `df` is `0`: the two fits are not ordered by precision in any
direction and there is nothing to compare. **Check `df` before reading
the p-value.** The one exception is two fits with identical coefficients,
where there is no difference to weigh and the result is `stat = 0`,
`p_value = 1`, `df = 0`.

!!! warning "Both common fixed-effects pairings fail the null"
    Two `fit_spatial_panel` fits that differ only in their fixed-effects
    specification generally do not satisfy the efficiency condition, and
    this test is badly sized on them. Measured over 250 replications at
    `N = 49`, `T = 5`, `ρ = 0.4`, with the null true:

    | pairing | result |
    |---|---|
    | `IndivFE` vs `TwoWayFE`, no time effects | degenerate: `NaN` in 160 of 250 draws, and 6 of the remaining 90 reject at the 5% level |
    | `NoFE` vs `IndivFE`, effects uncorrelated with `X` | computable every time but oversized: 25.6% rejection at the 5% level |

    The first pairing is a near-cancellation — the two estimates differ by
    a standard deviation of 0.005 while each standard error is 0.037 — so
    `V_fe - V_re` is noise. The second is computable because an unmodelled
    individual effect inflates `σ̂²` and so every standard error in the
    `NoFE` fit, but that inflation comes from the misspecification itself
    rather than from an efficiency ordering, so the null distribution is
    not `χ²`. It does have power — 100% rejection when the effects are
    correlated with `X` — but treat a borderline p-value from either
    pairing as uninformative.
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
    diff = β_fe_c .- β_re_c

    # The statistic needs V_fe - V_re to be positive definite, and often it
    # is not: the two variance matrices are close, so their difference is a
    # cancellation whose sign is decided by sampling noise. Shifting the
    # eigenvalues to force definiteness is not a repair -- it leaves the
    # smallest one at the shift itself, so the inverse carries an eigenvalue
    # of 1/shift and the statistic is whatever that noise direction happens
    # to be. Instead invert only the directions that are clearly positive,
    # and report how many were used as the degrees of freedom.
    E = eigen(Symmetric(V_diff))
    tol = 1e-8 * max(maximum(abs, E.values), eps(eltype(E.values)))
    keep = E.values .> tol
    k = count(keep)

    if k == 0
        if norm(diff) <= 1e-10 * max(one(eltype(diff)), norm(β_fe_c))
            # The two fits give the same coefficients, so there is no
            # difference to weigh and the statistic is zero however their
            # variances compare. This is the self-comparison case.
            return HausmanTestResult(0.0, 0, 1.0, shared)
        end
        # Otherwise the fits are not ordered by precision in any direction,
        # so the difference between them cannot be weighed at all.
        return HausmanTestResult(NaN, 0, NaN, shared)
    end

    z = transpose(E.vectors[:, keep]) * diff
    stat = sum(z .^ 2 ./ E.values[keep])
    stat = max(stat, 0.0)
    df = k
    p_value = _chi2_sf(stat, df)

    return HausmanTestResult(stat, df, p_value, shared)
end

# ─── Spatial Chow Test ───────────────────────────────────────────────

"""
    ChowTestResult

Structural-break test over time for a spatial panel, returned by
[`spatial_chow_test`](@ref).

Fields: `stat`, `df`, `p_value` and `break_period`, the period tested.
Rejection says the coefficients differ either side of that period, so a
single pooled specification is not adequate.
"""
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

Spatial Chow test for a structural break at `break_period`: an F test of
whether the parameters are stable across the two sub-periods.

Each model is fitted by OLS on `[X Wy]`, so the break restricts the `K`
slopes **and** the spatial lag coefficient — `K + 1` restrictions against
`2(K + 1)` unrestricted parameters, hence `df = (K + 1, NT - 2(K + 1))`.
Counting only the `K` slopes roughly doubles the size of the test: over
1500 replications with no break at `N = 64`, `T = 6`, `K = 3`, the
`K`-restriction version rejects 8.5-10.5% of the time at the 5% level and
1.8-2.7% at the 1% level, against 4.1-5.8% and 0.5-1.2% here (Monte-Carlo
standard error 0.56%).

!!! note
    `Wy` is endogenous, so the sub-period fits are not consistent estimates
    of a SAR and the F distribution is an approximation. The calibration
    above is the evidence that it is an adequate one at these sample sizes;
    it has not been checked for small `N` or for a weights matrix far from
    a lattice.
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
    # K slopes plus the spatial lag coefficient break together.
    n_par = K + 1
    df_num = n_par
    df_den = NT - 2 * n_par

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

"""
    PanelLMTestResult

Joint and individual LM tests for a spatial panel, returned by
[`panel_lm_tests`](@ref).

Carries a joint test (`joint_stat`, `joint_df`, `joint_p`) against the
hypothesis of no spatial dependence and no random effects, plus the
component tests: `lag_stat`/`lag_p`, `error_stat`/`error_p` and
`re_stat`/`re_p` for the random-effects variance.

Read the joint test first; if it does not reject, the components are not
worth chasing individually.
"""
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

LM tests for a spatial panel against a pooled OLS baseline: a spatial lag
test, a spatial error test, a Breusch-Pagan test for random effects, and a
joint test of all three.

The three component tests are each `χ²(1)`. With `A = I_T ⊗ W` and
`τ = tr(A²) + tr(A′A) = T · (tr(W′W) + tr(W²))`, the scores at the null are
`d_ρ = e′Ay/σ̂²` and `d_λ = e′Ae/σ̂²`, with

    Var(d_λ) = τ,   Var(d_ρ) = Cov(d_ρ, d_λ) + τ = D,
    D = (AXβ̂)′M(AXβ̂)/σ̂² + τ

so the lag test must be scaled by `D`, not by `τ` and not by the spread of
the lagged *observed* `y` (Anselin 1988). The joint test is **not** the sum
of the three components: `d_ρ` and `d_λ` are correlated, with
`Cov(d_ρ, d_λ) = τ`, so their sum has variance above `χ²(2)`. The spatial
block uses the quadratic form in the inverse information instead (Anselin,
Bera, Florax & Yoon 1996),

    LM_spatial = [τ d_ρ² - 2τ d_ρ d_λ + D d_λ²] / [τ (D - τ)]

which is `χ²(2)`. The random-effects score is a quadratic form in `e` whose
cross-trace with `A` is `tr(J_T ⊗ W) = T · tr(W) = 0`, so it is
asymptotically uncorrelated with both spatial scores and simply adds, giving
a `χ²(3)` joint test.

# Calibration

Under the null, 4000 replications at `N = 64`, `T = 5` (Monte-Carlo standard
error 0.34% on a rejection rate):

| test | mean | variance | rejects at 5% |
|---|---|---|---|
| lag | 1.022 | 2.07 | 5.17% |
| error | 0.998 | 2.33 | 5.40% |
| random effects | 1.041 | 2.04 | 5.08% |
| joint | 3.046 | 6.30 | 5.47% |

Power at `ρ = 0.15`: 97.0% for the lag test, 89.9% for the joint test.

!!! note
    Every statistic is clamped at zero, so a value of exactly `0.0` means
    the formula returned something negative and the test is uninformative,
    not that the null holds exactly.
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

    # τ = tr(A²) + tr(A′A) for A = I_T ⊗ W; _trace_WpW divides by N.
    τ = N * T_per * _trace_WpW(W)

    d_λ = dot(e, We) / σ²
    LM_error = max(d_λ^2 / τ, 0.0)

    # The lag score is scaled by the signal in the lagged *fitted* values
    # plus the same trace term. Using the lagged observed y instead, and
    # dropping the trace term, inflates the statistic by about 17%.
    ŷ = X * β_ols
    Wŷ = zeros(NT)
    for t in 1:T_per
        idx = ((t-1)*N+1):(t*N)
        @views panel_spatial_lag!(Wŷ[idx], W, ŷ[idx], N, 1)
    end
    MWŷ = Wŷ .- X * ((X' * X) \ (X' * Wŷ))
    D = dot(Wŷ, MWŷ) / σ² + τ

    d_ρ = dot(e, Wy) / σ²
    LM_lag = max(d_ρ^2 / D, 0.0)

    E_mat = reshape(e, N, T_per)
    e_bar_i = vec(mean(E_mat, dims=2))
    sum_sq_bar = T_per^2 * dot(e_bar_i, e_bar_i)
    LM_re = (sum_sq_bar / (dot(e, e)) - 1.0)^2 * NT / (2.0 * (T_per - 1))
    LM_re = max(LM_re, 0.0)

    # d_ρ and d_λ are correlated, so the joint spatial test is the quadratic
    # form in the inverse information, not LM_lag + LM_error.
    den = τ * (D - τ)
    LM_spatial = den > 0 ?
        max((τ * d_ρ^2 - 2 * τ * d_ρ * d_λ + D * d_λ^2) / den, 0.0) :
        LM_lag + LM_error
    LM_joint = LM_spatial + LM_re
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

"""
    PanelMoranResult

Moran's I computed period by period for a panel, returned by
[`panel_moran`](@ref).

Fields: `period_results`, a vector of `MoranResult` with one entry
per period; `pooled`; and `T_periods`.

`pooled` is not Moran's I of the stacked residuals. Its `I_stat` is the
average of the per-period statistics and its `z_stat` is Stouffer's
combination `z̄ √T`, computed from the per-period z values rather than
from `I_stat` and `Var_I`. Both treat the periods as independent, which
residuals from a single panel fit are not: they share `β̂` and the fixed
effects. Measured over 400 replications at `N = 64`, `T = 5` on residuals
from a correctly specified `IndivFE` SAR fit, the combined z has standard
deviation 1.07 rather than 1.00 and rejects 6.5-7.2% of the time at the 5%
level, so read it as mildly liberal. On genuinely independent draws it is
accurate (standard deviation 0.95, rejecting 3.9%).

The per-period series is the useful part — it shows whether spatial
dependence is stable over time or concentrated in particular periods, which
a single pooled number hides.
"""
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
        # I_stat here is an average over T_per periods, so the variance
        # beside it has to be the variance of that average. Carrying the
        # single-period variance over left the field too large by a factor
        # of T_per. E_I needs no change: the mean of an average is the mean.
        pooled = MoranResult(avg_I, pooled.E_I, pooled.Var_I / T_per,
                             meta_z, meta_p, :panel_meta)
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
