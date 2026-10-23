module SpatialModelSelection

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: loglikelihood, nobs, dof_residual, coef, vcov, residuals

import ..SpatialUtils: _normcdf, _chi2_sf
import ..SpatialWeights
import ..SpatialEconometricsPrototype

export
    spatial_aic, spatial_bic,
    LRTestResult, lr_test,
    VuongTestResult, vuong_test,
    CommonFactorTestResult, common_factor_test,
    WaldTestResult, wald_test,
    KnnSelectionResult, select_knn_w,
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
        Wk = SpatialWeights.knn_weights(coords, k; binary = binary,
                                        row_standardize = true,
                                        symmetrize = symmetrize)
        Wk = convert(SparseMatrixCSC{T, Int}, Wk)
        Ws[i] = Wk
        isl[i] = SpatialWeights.spatial_connectivity(Wk).n_islands

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

end # module
