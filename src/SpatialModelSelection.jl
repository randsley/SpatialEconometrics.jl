module SpatialModelSelection

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: loglikelihood, nobs, dof_residual, coef, vcov, residuals

import ..SpatialUtils: _normcdf, _chi2_sf, _loggamma, check_design_rank,
    LogDetMethod, ExactLU, ChebyshevApprox, precompute_chebyshev_traces,
    chebyshev_logdet
import ..SpatialWeights: knn_weights, spatial_connectivity
import ..SpatialEconometricsPrototype

export
    spatial_aic, spatial_bic,
    LRTestResult, lr_test,
    VuongTestResult, vuong_test,
    CommonFactorTestResult, common_factor_test,
    WaldTestResult, wald_test,
    KnnSelectionResult, select_knn_w,
    model_comparison_table,
    log_marginal_likelihood, bayes_spec_probs, SpecEvidence,
    SpecRoadmapResult, sdem_roadmap

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
    _pointwise_loglik(m) -> Vector{Float64}

Per-observation log-likelihood contributions, summing to `loglikelihood(m)`.

Vuong's ω̂ is the standard deviation of the per-observation log-likelihood
ratio, so the aggregate log-likelihood is not enough to compute it.

For the Gaussian models here the density splits into an observation-wise
part and a Jacobian. The Jacobian — `log|I − ρW|`, and a second factor for
SAC — is recovered as the residual between the stored log-likelihood and
the observation-wise part rather than re-derived per model type, so this
sums to `loglikelihood(m)` by construction and needs no knowledge of which
spatial structure produced `m`.

The Jacobian is then shared out evenly across observations. That is a
convention, not an identity: `log|I − ρW|` is not separable across
observations, and under spatial dependence no split of it is. It is the
usual practical choice, and it cancels from ω̂ entirely whenever the two
models' Jacobians agree.

Requires `residuals(m)` to be the residual that enters the likelihood and
the ML variance to be `e'e/n`, which holds for every model type here.
"""
function _pointwise_loglik(m::StatsAPI.RegressionModel)
    e = residuals(m)
    n = length(e)
    σ2 = sum(abs2, e) / n
    σ2 > 0 || error("cannot split the log-likelihood: the residual variance is zero")
    base = -0.5 .* log(2π * σ2) .- (e .^ 2) ./ (2 * σ2)
    jac = loglikelihood(m) - sum(base)
    return base .+ jac / n
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

    # ω̂ is the standard deviation of the PER-OBSERVATION log-likelihood
    # ratio. It cannot be had from the aggregate log-likelihoods, and the
    # previous version tried: sqrt(abs(2(ll1-ll2)/n - adj^2)) is twice a mean
    # difference, not a variance. It came out 0.71x the right size on two
    # non-nested SLX models, and because the quantity under the root goes
    # negative whenever model 1 fits worse, the abs() left the statistic
    # non-antisymmetric in its arguments -- +0.9871 against -0.9804, where a
    # Vuong statistic must flip sign exactly.
    mᵢ = _pointwise_loglik(m1) .- _pointwise_loglik(m2)
    omega = sqrt(sum(abs2, mᵢ .- mean(mᵢ)) / n)

    # Vuong's AIC-corrected LR, over sqrt(n) ω̂.
    numerator = (ll1 - ll2) - (k1 - k2)
    stat = omega > 0 ? numerator / (sqrt(Float64(n)) * omega) : 0.0

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


# ─── Selecting k for a k-nearest-neighbour W ──────────────────────────
#
# Every estimator in this package takes W as given, and a knn W takes k as
# given. Nothing in the data fixes k, and the fit is not indifferent to it:
# the likelihood is a genuinely non-monotone function of k, so the choice can
# be made by refitting across a range and reading off the optimum rather than
# by convention.
#
# A caveat worth stating plainly. With the variant and X held fixed, the
# number of estimated parameters does not change with k, so AIC and BIC rank
# the candidates in exactly the same order as the log-likelihood does. The
# criterion is doing no penalising here; it is a monotone relabelling of the
# likelihood, reported as a criterion so the numbers are comparable with
# `spatial_aic` elsewhere. The penalty only starts to matter when the
# comparison runs across variants as well as across k.

"""
    KnnSelectionResult

The result of [`select_knn_w`](@ref): one fit per candidate `k`, and the `W`
that won.

# Fields
- `k_values`: the candidate `k`s actually fitted, in ascending order. Values
  of the requested range outside `1 <= k < N` are dropped.
- `aic`, `bic`, `loglik`, `rho`: one entry per candidate, aligned with
  `k_values`. `NaN` where the fit failed.
- `converged`: `false` where the fit threw, or where `rho` landed on the edge
  of `rho_bounds` and is therefore not an interior maximum.
- `n_islands`: isolated units in each candidate `W`. Zero throughout for knn
  by construction — every unit has exactly `k` neighbours — and reported as a
  check on that rather than because it is expected to fire.
- `best_k`, `best_W`: the minimiser over the *converged* candidates, and its
  weight matrix, ready to pass to a fit.
- `criterion`: `:aic` or `:bic`, whichever was minimised.
"""
struct KnnSelectionResult{T<:AbstractFloat, M<:AbstractSparseMatrix{T}}
    k_values::Vector{Int}
    aic::Vector{T}
    bic::Vector{T}
    loglik::Vector{T}
    rho::Vector{T}
    converged::Vector{Bool}
    n_islands::Vector{Int}
    best_k::Int
    best_W::M
    criterion::Symbol
end

function Base.show(io::IO, r::KnnSelectionResult)
    crit = r.criterion === :aic ? r.aic : r.bic
    println(io, "k-Nearest-Neighbour W Selection")
    println(io, "───────────────────────────────────────────────")
    @printf(io, "  Candidates:  k = %d to %d (%d fits, %d converged)\n",
            first(r.k_values), last(r.k_values), length(r.k_values),
            count(r.converged))
    @printf(io, "  Criterion:   %s\n", uppercase(String(r.criterion)))
    @printf(io, "  Selected:    k = %d  (%s = %.3f, ρ̂ = %.4f)\n",
            r.best_k, uppercase(String(r.criterion)),
            crit[findfirst(==(r.best_k), r.k_values)],
            r.rho[findfirst(==(r.best_k), r.k_values)])
    ok = findall(r.converged)
    if length(ok) > 1
        spread = maximum(crit[ok]) - minimum(crit[ok])
        @printf(io, "  Spread:      %.3f across the range\n", spread)
    end
    if any(!, r.converged)
        bad = r.k_values[.!r.converged]
        println(io, "  Not converged at k = ", join(bad, ", "))
    end
end

"""
    select_knn_w(coords, y, X; variant = SAR(), k_range = 2:50, criterion = :aic, ...)

Choose `k` for a k-nearest-neighbour weight matrix by refitting the model at
every candidate `k` and minimising an information criterion.

`coords` is `N × D` positions, and `y` and `X` are the same data for every
candidate — which is what makes the criteria comparable. `X` must **not**
include an intercept; `fit_spatial` adds one.

Returns a [`KnnSelectionResult`](@ref), whose `best_W` can be passed straight
to a fit.

# What the curve looks like
The criterion is not monotone in `k`, which is the whole reason to search: a
`W` that is too sparse misses real neighbours, and one that is too dense
dilutes them, so the curve falls and then rises. The minimum is usually
shallow. Read `aic` (or `bic`) as a curve and not only as an argmin — if the
spread across the range is a point or two, the data has not identified `k`
and the sensitivity of whatever you conclude to `k` should be reported.

# Convergence
A candidate is marked not converged if its fit throws, or if `rho` lands on
the edge of `rho_bounds`, where the concentrated likelihood has no interior
maximum and its value is not comparable with the others. Non-converged
candidates keep their place in the output with `NaN` criteria and are
excluded from the argmin, so a failed fit cannot win by having no number.

# Keywords
- `variant = SAR()`: any `SpatialVariant`; `SDM()` lags `X` as well, and the
  lagged regressors then change with `k` along with `W`.
- `k_range = 2:50`: candidates. Values outside `1 <= k < N` are dropped, and
  an empty intersection is an error.
- `criterion = :aic`: or `:bic`.
- `binary = false`, `symmetrize = false`: passed to [`knn_weights`](@ref).
  Row standardisation is always on, since `rho` is only interpretable on
  `(-1, 1)` with a row-standardised `W`.
- any remaining keywords go to `fit_spatial`, e.g. `method = ExactLU()`.

# Examples
```julia
sel = select_knn_w(coords, y, X; k_range = 2:25)
println(sel)                                   # the curve's shape and winner
m = fit_spatial(SAR(), sel.best_W, X, y)       # refit on the chosen W
```

# Cost
One full fit per candidate, so `length(k_range)` fits. Each is cheap while
`N` stays under `fit_spatial`'s `max_exact_n` (1000 by default), where the
trace powers are computed exactly -- about 0.016s at `N = 200` and 0.18s at
`N = 600` on one core, so a 2:50 search costs seconds. Above `max_exact_n`
the traces switch to Hutchinson probes and the per-fit cost rises sharply;
narrow `k_range` or raise `max_exact_n` deliberately rather than by accident.

# Measured behaviour
The design: coordinates uniform on the unit square, `y` from a SAR with
`rho = 0.5` on the knn `W` at `k = 6`, two covariates.

How often the search recovers the generating `k`, 300 draws, `k` searched
over `2:20` so 5.3% by chance:

| `N` | `k̂ = 6` | within ±1 | within ±2 | median `k̂` |
|---|---|---|---|---|
| 100 | 42.3% | 68.3% | 84.0% | 6 |
| 200 | 57.3% | 86.0% | 95.7% | 6 |
| 400 | 77.0% | 95.0% | 99.7% | 6 |

Well targeted, not sharp: at `N = 100` the exact `k` is missed more often
than it is found, which is the reason `KnnSelectionResult` returns every
candidate's criterion rather than only the winner.

What a wrong `k` costs `rho-hat`, 300 draws at `N = 200`:

| `k` | mean `rho-hat` | bias | rmse |
|---|---|---|---|
| 2 | 0.325 | −35.1% | 0.180 |
| 4 | 0.426 | −14.7% | 0.083 |
| 6 (true) | 0.493 | −1.4% | 0.041 |
| 10 | 0.552 | +10.4% | 0.070 |
| 15 | 0.605 | +20.9% | 0.118 |
| 20 | 0.646 | +29.1% | 0.158 |
| selected | 0.502 | **+0.5%** | 0.052 |

Too sparse understates the dependence, too dense overstates it, and the
drift is monotone in `k` on both sides. Selecting leaves `rho-hat` close to
unbiased at the cost of a little variance against an oracle that knew `k`
(rmse 0.052 against 0.041). Paired against a fixed choice, selection gives
the smaller `|rho-hat - rho|` in 98.3% of draws at `k = 2`, 69.0% at
`k = 10` and 93.0% at `k = 20`.

Note that the bias range here is wider than the "up to 20%" usually quoted
for this effect — but it is a property of the range searched, not of the
method: `k = 20` is simply further from `k = 6` than most applications would
stray.

# References
Kubara, M., & Kopczewska, K. (2023). Akaike information criterion in
choosing the optimal k-nearest neighbours of the spatial weight matrix.
*Spatial Economic Analysis*, 19(1), 73–91.
"""
function select_knn_w(
    coords::AbstractMatrix{<:Real},
    y::AbstractVector{T},
    X::AbstractMatrix{T};
    variant = SpatialEconometricsPrototype.SAR(),
    k_range = 2:50,
    criterion::Symbol = :aic,
    binary::Bool = false,
    symmetrize::Bool = false,
    rho_bounds::Tuple{T, T} = (T(-0.995), T(0.995)),
    fit_kwargs...
) where {T<:AbstractFloat}

    N = size(coords, 1)
    criterion in (:aic, :bic) || throw(ArgumentError(
        "criterion must be :aic or :bic, got :$criterion"))
    length(y) == N || throw(DimensionMismatch(
        "y has $(length(y)) entries but coords has $N rows; the sample must " *
        "be the same for every candidate k or the criteria are not comparable"))
    size(X, 1) == N || throw(DimensionMismatch(
        "X has $(size(X, 1)) rows but coords has $N"))

    ks = sort!(collect(Iterators.filter(k -> 1 <= k < N, k_range)))
    isempty(ks) && throw(ArgumentError(
        "no candidate k in $k_range satisfies 1 <= k < N = $N"))

    n = length(ks)
    aic = fill(T(NaN), n); bic = fill(T(NaN), n)
    ll = fill(T(NaN), n);  rho = fill(T(NaN), n)
    conv = falses(n);      isl = zeros(Int, n)
    Ws = Vector{Any}(undef, n)

    for (i, k) in enumerate(ks)
        Wk = knn_weights(coords, k; binary = binary,
                                        row_standardize = true,
                                        symmetrize = symmetrize)
        Wk = convert(SparseMatrixCSC{T, Int}, Wk)
        Ws[i] = Wk
        isl[i] = spatial_connectivity(Wk).n_islands

        m = try
            SpatialEconometricsPrototype.fit_spatial(variant, Wk, X, y;
                                                     rho_bounds = rho_bounds,
                                                     fit_kwargs...)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        m === nothing && continue

        rho[i] = m.rho
        ll[i]  = loglikelihood(m)
        # A rho pinned to the edge is the optimiser reporting that it found no
        # interior maximum. Its likelihood is not comparable with the rest, so
        # the candidate is recorded and excluded rather than silently ranked.
        edge = min(abs(m.rho - rho_bounds[1]), abs(m.rho - rho_bounds[2]))
        if edge < 1e-6
            continue
        end
        aic[i] = spatial_aic(m)
        bic[i] = spatial_bic(m)
        conv[i] = true
    end

    any(conv) || throw(ArgumentError(
        "no candidate k in $(first(ks)):$(last(ks)) produced a converged " *
        "fit, so there is nothing to select between"))

    crit = criterion === :aic ? aic : bic
    ok = findall(conv)
    best = ok[argmin(crit[ok])]

    return KnnSelectionResult{T, SparseMatrixCSC{T, Int}}(
        ks, aic, bic, ll, rho, collect(conv), isl,
        ks[best], Ws[best], criterion)
end

# ─── Bayesian specification comparison ────────────────────────────────

# The six specifications this module can put a marginal likelihood on.
# `:slx` and `:ols` carry no spatial parameter; `:sar`/`:sdm` put it on the
# response (global spillovers), `:sem`/`:sdem` on the disturbance (local).
const _SPEC_KINDS = (:ols, :slx, :sar, :sdm, :sem, :sdem)

_spec_family(spec::Symbol) =
    spec in (:ols, :slx)  ? :none  :
    spec in (:sar, :sdm)  ? :lag   :
    spec in (:sem, :sdem) ? :error :
    throw(ArgumentError("unknown specification :$spec; expected one of $(_SPEC_KINDS)"))

_spec_has_durbin(spec::Symbol) = spec in (:slx, :sdm, :sdem)

# Design for a specification: intercept, covariates, and the Durbin block
# W*X where the specification calls for one.
function _spec_design(spec::Symbol, X::AbstractMatrix{T},
                      W::AbstractSparseMatrix{T}) where {T<:AbstractFloat}
    N = size(X, 1)
    return _spec_has_durbin(spec) ? hcat(ones(T, N), X, W * X) :
                                    hcat(ones(T, N), X)
end

function _spec_colnames(spec::Symbol, vnames::Vector{String})
    base = vcat(["(Intercept)"], vnames)
    return _spec_has_durbin(spec) ? vcat(base, ["W_" * v for v in vnames]) : base
end

_lgamma(x::T) where {T<:AbstractFloat} = T(_loggamma(Float64(x)))

# log p(y | spatial parameter) with beta and sigma integrated out under the
# improper prior p(beta, sigma^2) proportional to 1/sigma^2:
#
#   p(y|ρ) = |A| · π^(-(n-k)/2) · |Z'Z|^(-1/2) · Γ((n-k)/2) · S(ρ)^(-(n-k)/2)
#
# with A the Jacobian of whichever transform the family applies. The
# normalising constants are kept rather than dropped: the point is to compare
# specifications whose designs have different k, and the Γ and |Z'Z| terms are
# exactly what penalise the wider design.
function _log_cond_marginal(nk::Int, log_det_ZZ::T, S::T, log_jac::T) where {T<:AbstractFloat}
    return log_jac - T(0.5) * nk * log(T(pi)) - T(0.5) * log_det_ZZ +
           _lgamma(T(0.5) * nk) - T(0.5) * nk * log(S)
end

# Residual sum of squares and log|Z'Z| from one QR, which is what keeps this
# stable when the Durbin block leaves Z ill-conditioned.
function _rss_and_logdet(Z::AbstractMatrix{T}, z::AbstractVector{T}) where {T<:AbstractFloat}
    F = qr(Z)
    g = F \ z
    r = z .- Z * g
    return dot(r, r), T(2) * sum(log, abs.(diag(F.R)))
end

# Trapezoid rule carried out in logs: the integrand spans hundreds of orders
# of magnitude, so exponentiating it directly overflows at any realistic N.
function _log_trapezoid(l::Vector{T}, h::T) where {T<:AbstractFloat}
    m = maximum(l)
    isfinite(m) || return m
    acc = zero(T)
    @inbounds for i in eachindex(l)
        w = (i == 1 || i == length(l)) ? T(0.5) : one(T)
        acc += w * exp(l[i] - m)
    end
    return m + log(acc * h)
end

function _log_prior_density(grid, a::T, b::T, prior::Symbol, d::T) where {T<:AbstractFloat}
    if prior === :uniform
        return fill(-log(b - a), length(grid))
    end
    # Beta(d, d) on the unit interval, stretched over [a, b].
    out = Vector{T}(undef, length(grid))
    logB = _lgamma(d) + _lgamma(d) - _lgamma(2d)
    @inbounds for (i, rho) in enumerate(grid)
        u = (T(rho) - a) / (b - a)
        out[i] = (u <= 0 || u >= 1) ? T(-Inf) :
            (d - 1) * log(u) + (d - 1) * log(one(T) - u) - logB - log(b - a)
    end
    return out
end

function _logdet_closure(W::AbstractSparseMatrix{T}, N::Int,
                         method::Union{Nothing, LogDetMethod},
                         max_exact_n::Int) where {T<:AbstractFloat}
    m = isnothing(method) ? (N <= max_exact_n ? ExactLU() :
                             ChebyshevApprox(order = 8, n_probes = 0)) : method
    if m isa ChebyshevApprox
        tau = precompute_chebyshev_traces(W, m)
        return rho -> chebyshev_logdet(rho, tau)
    end
    I_sp = sparse(one(T) * I, N, N)
    return rho -> T(first(logabsdet(lu(I_sp - rho * W))))
end

"""
    log_marginal_likelihood(spec, y, X, W; kwargs...)

Log marginal likelihood of a spatial specification, with `β` and `σ²`
integrated out analytically and the spatial parameter integrated numerically.

`spec` is one of `:ols`, `:slx`, `:sar`, `:sdm`, `:sem`, `:sdem`. The design
is `[1 X]`, or `[1 X WX]` for the three Durbin specifications; `:sar` and
`:sdm` put the spatial parameter on the response — global spillovers —
`:sem` and `:sdem` put it on the disturbance — local spillovers — and `:ols`
and `:slx` have none, so their marginal is closed form and no quadrature
runs.

Under the improper prior `p(β, σ²) ∝ 1/σ²` the inner integral is exact:

```
p(y | ρ) = |A(ρ)| · π^(-(n-k)/2) · |Z'Z|^(-1/2) · Γ((n-k)/2) · S(ρ)^(-(n-k)/2)
```

with `A(ρ) = I - ρW` for a lag model and `I - λW` for an error model, and `S`
the residual sum of squares at that value. For an error model both `S` and
`|Z'Z|` are evaluated on the filtered data `(I - λW)[y Z]`, so the
determinant term moves with `λ`. Dropping it is an easy mistake — it *is*
constant for a lag model — and it biases the comparison toward the error
specifications.

The outer integral is a trapezoid rule on `n_grid` points of `rho_bounds`,
carried out by log-sum-exp so the integrand's scale cannot overflow.

# Keywords
- `rho_bounds`: the support of the prior, `(-0.995, 0.995)` by default,
  which is the stationary interval for a row-standardised `W`.
- `prior`: `:uniform` (default) or `:beta`, a Beta(`beta_shape`,
  `beta_shape`) density stretched over `rho_bounds`.
- `n_grid`: quadrature nodes, 200 by default.
- `method`: `ExactLU()` or `ChebyshevApprox()` for `log|A|`. Exact at or
  below `max_exact_n` observations, Chebyshev above it, by default.
- `varnames`: covariate names, used in the message if the design turns out
  to be rank deficient.

!!! note "The prior matters"
    A marginal likelihood is not a likelihood ratio: the prior on the
    spatial parameter does not wash out with sample size, because it is a
    prior over a parameter the `:ols` and `:slx` specifications do not have.
    Widening `rho_bounds` spreads that prior thinner and penalises the
    spatial specifications. Report the setting alongside the probabilities,
    and if a comparison is close, check it under both priors.

# References
LeSage, J. P. (2014). Spatial econometric panel data model specification: A
Bayesian approach. *Spatial Statistics*, 9, 122–145.

LeSage, J. P. & Pace, R. K. (2009). *Introduction to Spatial Econometrics*.
CRC Press, chapter 5.
"""
function log_marginal_likelihood(
    spec::Symbol,
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    W::AbstractSparseMatrix{T};
    rho_bounds::Tuple{T, T} = (T(-0.995), T(0.995)),
    prior::Symbol = :uniform,
    beta_shape::Real = 1.01,
    n_grid::Int = 200,
    method::Union{Nothing, LogDetMethod} = nothing,
    max_exact_n::Int = 1000,
    varnames::Union{Nothing, Vector{String}} = nothing
) where {T<:AbstractFloat}

    family = _spec_family(spec)
    N, K = size(X)
    @assert length(y) == N "y and X must have the same number of rows"
    @assert size(W) == (N, N) "W must be N x N"
    @assert n_grid >= 4 "n_grid must be at least 4"
    prior in (:uniform, :beta) ||
        throw(ArgumentError("prior must be :uniform or :beta, got :$prior"))

    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames
    Z = _spec_design(spec, X, W)
    check_design_rank(Z, _spec_colnames(spec, vnames), "log_marginal_likelihood";
        context = "`X` must not contain its own intercept, since one is " *
                  "added. A Durbin specification also carries `W*X`, so a " *
                  "`W` for which `W*X` lies in the span of `X` is deficient.")
    k = size(Z, 2)
    nk = N - k
    nk > 0 || throw(ArgumentError("log_marginal_likelihood: the :$spec " *
        "design has $k columns for $N observations; the marginal likelihood " *
        "needs n > k."))
    yv = Vector{T}(y)

    if family === :none
        S, logdet_ZZ = _rss_and_logdet(Z, yv)
        return _log_cond_marginal(nk, logdet_ZZ, S, zero(T))
    end

    logdet_fn = _logdet_closure(W, N, method, max_exact_n)
    a, b = rho_bounds
    a < b || throw(ArgumentError("rho_bounds must be increasing"))
    grid = range(a, b; length = n_grid)
    h = (b - a) / (n_grid - 1)
    log_prior = _log_prior_density(grid, a, b, prior, T(beta_shape))

    l = Vector{T}(undef, n_grid)
    if family === :lag
        # S(ρ) is quadratic in ρ, so the design is factorised once.
        Wy = W * yv
        F = qr(Z)
        g0 = F \ yv
        gd = F \ Wy
        e0 = yv .- Z * g0
        ed = Wy .- Z * gd
        e0e0, e0ed, eded = dot(e0, e0), dot(e0, ed), dot(ed, ed)
        logdet_ZZ = T(2) * sum(log, abs.(diag(F.R)))
        @inbounds for i in 1:n_grid
            rho = T(grid[i])
            S = e0e0 - 2rho * e0ed + rho^2 * eded
            l[i] = _log_cond_marginal(nk, logdet_ZZ, S, logdet_fn(rho)) + log_prior[i]
        end
    else
        # The filtered design moves with λ, so S and |Z'Z| are both recomputed
        # at every node.
        WZ = W * Z
        Wy = W * yv
        @inbounds for i in 1:n_grid
            lam = T(grid[i])
            S, logdet_ZZ = _rss_and_logdet(Z .- lam .* WZ, yv .- lam .* Wy)
            l[i] = _log_cond_marginal(nk, logdet_ZZ, S, logdet_fn(lam)) + log_prior[i]
        end
    end

    return _log_trapezoid(l, T(h))
end

"""
    SpecEvidence

Posterior model probabilities over specifications and weight matrices, as
returned by [`bayes_spec_probs`](@ref).

# Fields
- `specs`: the specifications compared, in row order.
- `w_names`: the weight matrices, in column order.
- `log_marginal`: `length(specs) × length(w_names)` log marginal likelihoods.
- `prob`: posterior probabilities over the whole grid; sums to 1.
- `prob_spec`, `prob_w`: the row and column margins of `prob`.
- `best`: `(spec, w_name)` of the highest posterior cell.
- `prob_global`, `prob_local`: the probability mass on global-spillover
  specifications (`:sar`, `:sdm`) and on local ones (`:sem`, `:sdem`,
  `:slx`). `:ols` counts as neither.
"""
struct SpecEvidence{T<:AbstractFloat}
    specs::Vector{Symbol}
    w_names::Vector{String}
    log_marginal::Matrix{T}
    prob::Matrix{T}
    prob_spec::Vector{T}
    prob_w::Vector{T}
    best::Tuple{Symbol, String}
    prob_global::T
    prob_local::T
    prior::Symbol
    rho_bounds::Tuple{T, T}
end

function Base.show(io::IO, r::SpecEvidence)
    println(io, "Bayesian specification comparison")
    println(io, "─"^62)
    @printf(io, "%-8s", "spec")
    for w in r.w_names; @printf(io, "%14s", first(w, 13)); end
    @printf(io, "%10s\n", "P(spec)")
    for (i, s) in enumerate(r.specs)
        @printf(io, "%-8s", String(s))
        for j in eachindex(r.w_names); @printf(io, "%14.4f", r.prob[i, j]); end
        @printf(io, "%10.4f\n", r.prob_spec[i])
    end
    @printf(io, "%-8s", "P(W)")
    for j in eachindex(r.w_names); @printf(io, "%14.4f", r.prob_w[j]); end
    println(io)
    println(io, "─"^62)
    @printf(io, "best: %s on %s   P(global) = %.4f   P(local) = %.4f\n",
            String(r.best[1]), r.best[2], r.prob_global, r.prob_local)
    @printf(io, "prior on the spatial parameter: %s on (%.3f, %.3f)\n",
            String(r.prior), r.rho_bounds[1], r.rho_bounds[2])
end

"""
    bayes_spec_probs(y, X, Ws; specs, w_names, kwargs...)

Posterior probabilities over spatial specifications **and** weight matrices,
computed from the log marginal likelihoods of every combination.

This is the global-versus-local spillover question of LeSage (2014) put
jointly with the choice of `W`: with equal prior weight on every cell of the
`specs × Ws` grid, the posterior is

```
P(spec, W) ∝ exp(log p(y | spec, W))
```

normalised over the grid, with the maximum subtracted before exponentiating.
`prob_spec` and `prob_w` are its margins, so a comparison that is decisive
about the spillover structure while being indifferent between weight
matrices — or the other way round — is visible rather than hidden in a
single winner.

`Ws` is a vector of weight matrices; pass one to compare specifications
alone. `specs` defaults to `[:ols, :slx, :sar, :sdm, :sem, :sdem]`.

Keyword arguments other than `specs` and `w_names` are passed through to
[`log_marginal_likelihood`](@ref), including the prior, which is worth
reading about there before reporting a number from here.

# References
LeSage, J. P. (2014). *Spatial Statistics*, 9, 122–145.
"""
function bayes_spec_probs(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Ws::AbstractVector{<:AbstractSparseMatrix{T}};
    specs::Vector{Symbol} = collect(_SPEC_KINDS),
    w_names::Union{Nothing, Vector{String}} = nothing,
    rho_bounds::Tuple{T, T} = (T(-0.995), T(0.995)),
    prior::Symbol = :uniform,
    kwargs...
) where {T<:AbstractFloat}

    isempty(specs) && throw(ArgumentError("specs must not be empty"))
    isempty(Ws) && throw(ArgumentError("Ws must not be empty"))
    for s in specs; _spec_family(s); end
    names = isnothing(w_names) ? ["W$j" for j in eachindex(Ws)] : w_names
    length(names) == length(Ws) ||
        throw(ArgumentError("w_names must have one entry per weight matrix"))

    S, G = length(specs), length(Ws)
    lm = Matrix{T}(undef, S, G)
    for j in 1:G, i in 1:S
        lm[i, j] = log_marginal_likelihood(specs[i], y, X, Ws[j];
                                           rho_bounds = rho_bounds,
                                           prior = prior, kwargs...)
    end

    m = maximum(lm)
    p = exp.(lm .- m)
    p ./= sum(p)

    prob_spec = vec(sum(p; dims = 2))
    prob_w = vec(sum(p; dims = 1))
    idx = argmax(p)
    best = (specs[idx[1]], names[idx[2]])

    is_global = [s in (:sar, :sdm) for s in specs]
    is_local  = [s in (:sem, :sdem, :slx) for s in specs]

    return SpecEvidence{T}(copy(specs), names, lm, p, prob_spec, prob_w, best,
                           sum(prob_spec[is_global]), sum(prob_spec[is_local]),
                           prior, rho_bounds)
end

"""
    SpecRoadmapResult

The per-`W` outcome of [`sdem_roadmap`](@ref): which of the local-spillover
specifications wins on each weight matrix, and by how much.

# Fields
- `w_names`, `specs`: the grid searched.
- `criterion`: `:marginal`, `:aic` or `:bic`.
- `score`: `length(w_names) × length(specs)` criterion values. Higher is
  better for `:marginal`, lower for `:aic` and `:bic`.
- `best_spec`: the winning specification on each `W`.
- `margin`: the gap to the runner-up on that `W`, in the units of the
  criterion. A small margin is the signal that the search has not settled.
- `best_w`, `best_overall`: the winning cell of the whole grid.
"""
struct SpecRoadmapResult{T<:AbstractFloat}
    w_names::Vector{String}
    specs::Vector{Symbol}
    criterion::Symbol
    score::Matrix{T}
    best_spec::Vector{Symbol}
    margin::Vector{T}
    best_w::String
    best_overall::Symbol
end

function Base.show(io::IO, r::SpecRoadmapResult)
    println(io, "SDEM specification roadmap (criterion: ", String(r.criterion), ")")
    println(io, "─"^58)
    @printf(io, "%-14s", "W")
    for s in r.specs; @printf(io, "%12s", String(s)); end
    @printf(io, "%10s%10s\n", "best", "margin")
    for (i, w) in enumerate(r.w_names)
        @printf(io, "%-14s", first(w, 13))
        for j in eachindex(r.specs); @printf(io, "%12.3f", r.score[i, j]); end
        @printf(io, "%10s%10.3f\n", String(r.best_spec[i]), r.margin[i])
    end
    println(io, "─"^58)
    @printf(io, "overall: %s on %s\n", String(r.best_overall), r.best_w)
end

"""
    sdem_roadmap(y, X, Ws; criterion = :marginal, w_names, specs, kwargs...)

The Sutton & Sutton (2023) specification search for the local-spillover
case: for each candidate weight matrix, decide which of the local
specifications fits best, then read the winner across weight matrices.

The default `specs` are the three the roadmap is about — `:sdem` (their SDE,
Durbin terms with a spatial error), `:slx` (the same without the error
term), and `:sem` (the error term without the Durbin terms). Pass others to
widen the search; `:sar` and `:sdm` put the dependence on the response
instead, which is the global-spillover case [`bayes_spec_probs`](@ref) is
for.

`criterion` is `:marginal` (log marginal likelihood, higher is better),
`:aic` or `:bic` (lower is better). The marginal likelihood is the default
because it is the only one of the three that integrates over the spatial
parameter rather than plugging in an estimate, and the specifications being
compared differ in whether they *have* that parameter.

`margin[i]` reports the gap to the runner-up on weight matrix `i`. The point
of the roadmap is that the choice of `W` and the choice of specification are
decided together; a margin near zero says the data do not separate them, and
should be reported rather than resolved by taking the maximum.

# References
Sutton, J. & Sutton, J. (2023). A road map to capture the spatial dependence
underlying regions' economic resilience. *Spatial Economic Analysis*, 19(1),
57–72.
"""
function sdem_roadmap(
    y::AbstractVector{T},
    X::AbstractMatrix{T},
    Ws::AbstractVector{<:AbstractSparseMatrix{T}};
    criterion::Symbol = :marginal,
    specs::Vector{Symbol} = [:sdem, :slx, :sem],
    w_names::Union{Nothing, Vector{String}} = nothing,
    kwargs...
) where {T<:AbstractFloat}

    criterion in (:marginal, :aic, :bic) ||
        throw(ArgumentError("criterion must be :marginal, :aic or :bic, " *
                            "got :$criterion"))
    isempty(Ws) && throw(ArgumentError("Ws must not be empty"))
    names = isnothing(w_names) ? ["W$j" for j in eachindex(Ws)] : w_names
    length(names) == length(Ws) ||
        throw(ArgumentError("w_names must have one entry per weight matrix"))

    G, S = length(Ws), length(specs)
    score = Matrix{T}(undef, G, S)
    for i in 1:G, j in 1:S
        score[i, j] = criterion === :marginal ?
            log_marginal_likelihood(specs[j], y, X, Ws[i]; kwargs...) :
            _spec_information_criterion(specs[j], y, X, Ws[i], criterion; kwargs...)
    end

    best_spec = Vector{Symbol}(undef, G)
    margin = Vector{T}(undef, G)
    for i in 1:G
        row = score[i, :]
        o = sortperm(row; rev = criterion === :marginal)
        best_spec[i] = specs[o[1]]
        margin[i] = S > 1 ? abs(row[o[1]] - row[o[2]]) : T(Inf)
    end

    flat = criterion === :marginal ? argmax(score) : argmin(score)
    return SpecRoadmapResult{T}(names, copy(specs), criterion, score,
                                best_spec, margin, names[flat[1]],
                                specs[flat[2]])
end

# AIC/BIC for a specification, from the Gaussian profile likelihood at the
# maximising spatial parameter. Counted parameters: the design columns, σ²,
# and the spatial parameter where the specification has one.
function _spec_information_criterion(spec::Symbol, y::AbstractVector{T},
                                     X::AbstractMatrix{T},
                                     W::AbstractSparseMatrix{T},
                                     which::Symbol;
                                     rho_bounds::Tuple{T, T} = (T(-0.995), T(0.995)),
                                     n_grid::Int = 200,
                                     method::Union{Nothing, LogDetMethod} = nothing,
                                     max_exact_n::Int = 1000,
                                     kwargs...) where {T<:AbstractFloat}
    family = _spec_family(spec)
    N = length(y)
    Z = _spec_design(spec, X, W)
    k = size(Z, 2)
    yv = Vector{T}(y)

    if family === :none
        S, _ = _rss_and_logdet(Z, yv)
        ll = -T(0.5) * N * (log(T(2pi)) + log(S / N) + one(T))
        p = k + 1
        return which === :aic ? -2ll + 2p : -2ll + p * log(T(N))
    end

    logdet_fn = _logdet_closure(W, N, method, max_exact_n)
    a, b = rho_bounds
    grid = range(a, b; length = n_grid)
    WZ = W * Z
    Wy = W * yv
    best = T(-Inf)
    for g in grid
        par = T(g)
        S = if family === :lag
            r, _ = _rss_and_logdet(Z, yv .- par .* Wy); r
        else
            r, _ = _rss_and_logdet(Z .- par .* WZ, yv .- par .* Wy); r
        end
        ll = logdet_fn(par) - T(0.5) * N * (log(T(2pi)) + log(S / N) + one(T))
        ll > best && (best = ll)
    end
    p = k + 2
    return which === :aic ? -2best + 2p : -2best + p * log(T(N))
end

end # module
