module SpatialPanelEconometrics

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI

import ..SpatialUtils
import ..SpatialUtils: precompute_chebyshev_traces, chebyshev_logdet,
       precompute_neumann_traces, spatial_trace_powers, g_traces,
       brent_maximize, panel_spatial_lag!, panel_spatial_lag,
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

export SpatialVariant, SAR, SDM,
       FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
       PanelEstimationApproach, LeeYuTransformation, DirectQML,
       LogDetMethod, ExactLU, ChebyshevApprox,
       SpatialPanelModel, SpatialEffectsResult,
       fit_spatial_panel, spatial_effects

const SpatialVariant = SpatialUtils.SpatialVariant
const SAR = SpatialUtils.SAR
const SDM = SpatialUtils.SDM

abstract type PanelEstimationApproach end
"""
    LeeYuTransformation()

Estimate a spatial panel using the Lee-Yu orthogonal transformation.

Sweeping out fixed effects by demeaning leaves the variance estimate
inconsistent as the number of periods grows slowly, because the demeaned
errors are no longer independent. The Lee-Yu transformation instead
projects onto an orthonormal basis of the demeaning operator's complement,
which removes the fixed effects while keeping the errors independent, so no
bias correction is needed afterwards.

The default, and the right choice for short panels.

See also [`DirectQML`](@ref).

# References
Lee, L.-F. & Yu, J. (2010). Estimation of spatial autoregressive panel data
models with fixed effects. *Journal of Econometrics*, 154(2).
"""
struct LeeYuTransformation <: PanelEstimationApproach end
"""
    DirectQML(; bias_correct = true)

Estimate a spatial panel by quasi-maximum likelihood on the demeaned data,
optionally applying an analytical bias correction.

The straightforward approach: demean, then maximize the likelihood. With
`bias_correct = true` the known `O(1/T)` bias in the variance and spatial
parameter is corrected analytically after the fact; setting it to `false`
leaves the raw QML estimates, which is useful mainly for showing what the
correction does.

[`LeeYuTransformation`](@ref) avoids the bias by construction and is
preferred unless you want this estimator specifically.

# References
Lee, L.-F. & Yu, J. (2010). *Journal of Econometrics*, 154(2).
"""
struct DirectQML <: PanelEstimationApproach
    bias_correct::Bool
    DirectQML(; bias_correct::Bool=true) = new(bias_correct)
end

"""
    SpatialPanelModel

A fitted static spatial panel, returned by [`fit_spatial_panel`](@ref).
Implements the `StatsAPI.RegressionModel` interface.

Carries the fitted `variant`, `fe_spec` and estimation `approach`
alongside `coef`, `vcov`, the spatial parameter `rho` with its standard
error, `sigma2`, `loglik`, the weight matrix, the transformed design
`X_tilde`, residuals and `coefnames`, plus the panel dimensions `N` and
`T_periods`.

As in the cross-section, `coef` holds structural parameters rather than
marginal effects — use `spatial_effects`.

See also [`fit_spatial_panel`](@ref), `robust_vcov`.
"""
struct SpatialPanelModel{
    V<:SpatialVariant,
    FE<:FixedEffectsSpec,
    A<:PanelEstimationApproach,
    T<:AbstractFloat,
    M<:AbstractSparseMatrix{T}
} <: StatsAPI.RegressionModel
    variant::V
    fe_spec::FE
    approach::A
    N::Int
    T_periods::Int
    N_eff::Int
    T_eff::Int
    coef::Vector{T}
    coef_uncorrected::Vector{T}
    vcov::Matrix{T}
    rho::T
    rho_se::T
    sigma2::T
    sigma2_uncorrected::T
    loglik::T
    W::M
    X_tilde::Matrix{T}
    y_tilde::Vector{T}
    residuals::Vector{T}
    trace_powers::Vector{T}
    coefnames::Vector{String}
end

struct SpatialEffectsResult{T<:AbstractFloat}
    varnames::Vector{String}
    direct::Vector{T}
    direct_se::Vector{T}
    indirect::Vector{T}
    indirect_se::Vector{T}
    total::Vector{T}
    total_se::Vector{T}
end

function build_design_matrix(
    ::SAR, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T},
    N::Int, T_per::Int, fe::FixedEffectsSpec, varnames::Vector{String}
) where {T<:AbstractFloat}
    if fe isa NoFE
        X_aug = hcat(ones(T, N * T_per), X)
        names_aug = vcat(["(Intercept)"], varnames, ["ρ"])
    else
        X_aug = copy(X)
        names_aug = vcat(varnames, ["ρ"])
    end
    return X_aug, names_aug
end

function build_design_matrix(
    ::SDM, X::AbstractMatrix{T}, W::AbstractSparseMatrix{T},
    N::Int, T_per::Int, fe::FixedEffectsSpec, varnames::Vector{String}
) where {T<:AbstractFloat}
    WX = panel_spatial_lag(W, X, N, T_per)
    w_names = ["W_" * v for v in varnames]
    if fe isa NoFE
        X_aug = hcat(ones(T, N * T_per), X, WX)
        names_aug = vcat(["(Intercept)"], varnames, w_names, ["ρ"])
    else
        X_aug = hcat(X, WX)
        names_aug = vcat(varnames, w_names, ["ρ"])
    end
    return X_aug, names_aug
end

"""
    fit_spatial_panel(variant, fe_spec, W, X, y, N, T_per; approach, method,
                      varnames, rho_bounds)

Fit a static spatial panel by quasi-maximum likelihood with fixed effects.

# Arguments
- `variant`: `SAR` or `SDM`.
- `fe_spec`: a `FixedEffectsSpec` — `NoFE`,
  `IndivFE`, `TimeFE` or `TwoWayFE`.
- `W`: `N x N` sparse weight matrix for the cross-section, **not** the
  stacked `NT x NT` matrix.
- `X`: `(N*T_per) x K` covariates, no intercept.
- `y`: stacked response of length `N*T_per`.
- `N`, `T_per`: units and periods.

# Keywords
- `approach`: [`LeeYuTransformation`](@ref) (default) or
  [`DirectQML`](@ref).
- `method`: a `LogDetMethod`.
- `varnames`, `rho_bounds`: as for `fit_spatial`.

# Returns
A [`SpatialPanelModel`](@ref).

!!! warning "Stacking order"
    `X` and `y` must be stacked with **units varying fastest** within each
    period: `[unit_1 t_1, unit_2 t_1, ..., unit_N t_1, unit_1 t_2, ...]`.
    The wrong order will not error; it will silently misalign every unit
    with its neighbours.

# Examples
```julia
m = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per;
                      varnames = ["income", "density"])
m.rho
robust_coeftable(m, DriscollKraayVCOV())
```

See also `fit_dynamic_spatial_panel`, `panel_lm_tests`,
`spatial_hausman_test`.

# References
Lee, L.-F. & Yu, J. (2010). *Journal of Econometrics*, 154(2).
"""
function fit_spatial_panel(
    variant::V,
    fe_spec::FE,
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_per::Int;
    approach::A=LeeYuTransformation(),
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    rho_bounds::Tuple{T, T}=(T(-0.995), T(0.995)),
    max_exact_n::Int=1000,
    vcov_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {
    V<:SpatialVariant,
    FE<:FixedEffectsSpec,
    A<:PanelEstimationApproach,
    T<:AbstractFloat,
    M<:AbstractSparseMatrix{T}
}
    NT, K = size(X)
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    X_aug, coefnames = build_design_matrix(variant, X, W, N, T_per, fe_spec, vnames)
    Wy = similar(y)
    panel_spatial_lag!(Wy, W, y, N, T_per)

    y_tilde = within_transform!(copy(y), N, T_per, fe_spec)
    Wy_tilde = within_transform!(Wy, N, T_per, fe_spec)
    X_tilde = within_transform(X_aug, N, T_per, fe_spec)

    qr_X = qr(X_tilde)
    β0 = qr_X \ y_tilde
    βd = qr_X \ Wy_tilde

    e0 = y_tilde .- X_tilde * β0
    ed = Wy_tilde .- X_tilde * βd

    e0_e0 = dot(e0, e0)
    e0_ed = dot(e0, ed)
    ed_ed = dot(ed, ed)

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        base_logdet = ρ -> chebyshev_logdet(ρ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        base_logdet = ρ -> first(logabsdet(lu(I_sparse - ρ * W)))
    end

    N_eff, T_eff = effective_dims(N, T_per, fe_spec)
    time_fe_active = has_time_fe(fe_spec)

    N_opt = approach isa LeeYuTransformation ? N_eff : N
    T_opt = approach isa LeeYuTransformation ? T_eff : T_per
    NT_opt = N_opt * T_opt

    const_term = -T(0.5) * NT_opt * (log(T(2π)) + one(T))

    function concentrated_panel_ll(ρ)
        rss = e0_e0 - 2 * ρ * e0_ed + (ρ^2) * ed_ed
        σ2_opt = rss / NT_opt
        ld = base_logdet(ρ)
        if (approach isa LeeYuTransformation) && time_fe_active
            ld -= log(one(ρ) - ρ)
        end
        return const_term + T_opt * ld - T(0.5) * NT_opt * log(σ2_opt)
    end

    ρ_raw, max_ll = brent_maximize(concentrated_panel_ll, rho_bounds[1], rho_bounds[2])
    β_raw = β0 .- ρ_raw .* βd
    res_raw = e0 .- ρ_raw .* ed
    rss_raw = dot(res_raw, res_raw)

    σ2_uncorrected = rss_raw / (N * T_per)
    σ2_trans = rss_raw / (N_eff * T_eff)

    μ_traces = spatial_trace_powers(W; max_power=60, max_exact_n=max_exact_n,
                                    n_probes=vcov_probes, rng=rng)

    vcov_mat, β_final, ρ_final, σ2_final, res_final = finalize_panel_inference(
        approach, fe_spec, X_tilde, y_tilde, Wy_tilde, W,
        β_raw, ρ_raw, σ2_uncorrected, σ2_trans,
        N, T_per, N_eff, T_eff, μ_traces;
        max_exact_n=max_exact_n, n_probes=vcov_probes, rng=rng
    )

    P = length(β_final)
    ρ_se = sqrt(max(zero(T), vcov_mat[P + 1, P + 1]))

    return SpatialPanelModel{V, FE, A, T, M}(
        variant, fe_spec, approach, N, T_per, N_eff, T_eff,
        vcat(β_final, ρ_final), vcat(β_raw, ρ_raw),
        vcov_mat, ρ_final, ρ_se, σ2_final, σ2_uncorrected, max_ll,
        W, X_tilde, y_tilde, res_final, μ_traces, coefnames
    )
end

function finalize_panel_inference(
    approach::PanelEstimationApproach,
    fe_spec::FixedEffectsSpec,
    X_tilde::Matrix{T},
    y_tilde::Vector{T},
    Wy_tilde::Vector{T},
    W::AbstractSparseMatrix{T},
    β_raw::Vector{T},
    ρ_raw::T,
    σ2_uncorrected::T,
    σ2_trans::T,
    N::Int,
    T_per::Int,
    N_eff::Int,
    T_eff::Int,
    μ_traces::Vector{T};
    max_exact_n::Int=1000,
    n_probes::Int=250,
    rng::AbstractRNG=Random.default_rng()
) where {T<:AbstractFloat}

    P = length(β_raw)
    I_sparse = sparse(one(T) * I, N, N)
    A_lu = lu(I_sparse - ρ_raw * W)

    Xb = X_tilde * β_raw
    GXb = similar(Xb)
    Xb_mat = reshape(Xb, N, T_per)
    GXb_mat = reshape(GXb, N, T_per)
    mul!(GXb_mat, W, A_lu \ Xb_mat)
    within_transform!(GXb, N, T_per, fe_spec)

    tr_G, tr_G2, tr_GtG = g_traces(W, ρ_raw;
                                   project_constant=has_time_fe(fe_spec),
                                   max_exact_n=max_exact_n, n_probes=n_probes,
                                   rng=rng, μ=μ_traces)

    σ2_eval = (approach isa DirectQML && !approach.bias_correct) ? σ2_uncorrected : σ2_trans
    NT_eff = N_eff * T_eff

    I_ββ = (X_tilde' * X_tilde) ./ σ2_eval
    I_βρ = (X_tilde' * GXb) ./ σ2_eval
    I_ρρ = (dot(GXb, GXb) / σ2_eval) + T_eff * (tr_G2 + tr_GtG)
    I_ρσ2 = (T_eff * tr_G) / σ2_eval
    I_σ2σ2 = NT_eff / (2 * σ2_eval^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_ββ
    Info[1:P, P + 1] .= I_βρ
    Info[P + 1, 1:P] .= I_βρ
    Info[P + 1, P + 1] = I_ρρ
    Info[P + 1, P + 2] = I_ρσ2
    Info[P + 2, P + 1] = I_ρσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))
    V_coef = V_full[1:(P + 1), 1:(P + 1)]

    if (approach isa LeeYuTransformation) || !approach.bias_correct
        res = y_tilde .- ρ_raw .* Wy_tilde .- X_tilde * β_raw
        return V_coef, β_raw, ρ_raw, σ2_eval, res
    end

    score_bias = zeros(T, P + 2)
    if has_indiv_fe(fe_spec)
        tr_G_full = has_time_fe(fe_spec) ? (tr_G + one(T) / (one(T) - ρ_raw)) : tr_G
        score_bias[P + 1] += -tr_G_full
        score_bias[P + 2] += -T(N) / (2 * σ2_trans)
    end
    if has_time_fe(fe_spec)
        score_bias[P + 1] += -T(T_per) / (one(T) - ρ_raw)
        score_bias[P + 2] += -T(T_per) / (2 * σ2_trans)
    end

    θ_raw_full = vcat(β_raw, ρ_raw, σ2_uncorrected)
    Δ_bias = V_full * score_bias
    θ_bc = θ_raw_full .- Δ_bias

    β_bc = θ_bc[1:P]
    ρ_bc = clamp(θ_bc[P + 1], T(-0.995), T(0.995))
    σ2_bc = σ2_trans
    res_bc = y_tilde .- ρ_bc .* Wy_tilde .- X_tilde * β_bc

    return V_coef, β_bc, ρ_bc, σ2_bc, res_bc
end

function spatial_effects(
    model::SpatialPanelModel{V, FE, A, T};
    max_power::Int=50,
    vcov_mat::AbstractMatrix{T}=model.vcov
) where {V, FE, A, T}
    μ = model.trace_powers
    q = min(max_power, length(μ) - 2)
    θ_hat = model.coef
    Σ_θ = vcov_mat

    has_intercept = (FE === NoFE)
    offset = has_intercept ? 1 : 0
    P_cov = length(θ_hat) - 1 - offset
    K = V === SAR ? P_cov : div(P_cov, 2)

    direct = zeros(T, K); direct_se = zeros(T, K)
    indirect = zeros(T, K); indirect_se = zeros(T, K)
    total = zeros(T, K); total_se = zeros(T, K)

    for k in 1:K
        direct_fn = θ -> begin
            ρ = θ[end]
            β_k = θ[offset + k]
            γ_k = V === SDM ? θ[offset + K + k] : zero(eltype(θ))
            acc = zero(eltype(θ))
            for m in 0:q
                acc += (ρ^m) * (β_k * μ[m + 1] + γ_k * μ[m + 2])
            end
            return acc
        end

        total_fn = θ -> begin
            ρ = θ[end]
            β_k = θ[offset + k]
            γ_k = V === SDM ? θ[offset + K + k] : zero(eltype(θ))
            return (β_k + γ_k) / (one(eltype(θ)) - ρ)
        end

        indirect_fn = θ -> total_fn(θ) - direct_fn(θ)

        direct[k] = direct_fn(θ_hat)
        total[k] = total_fn(θ_hat)
        indirect[k] = indirect_fn(θ_hat)

        ∇_dir = ForwardDiff.gradient(direct_fn, θ_hat)
        ∇_tot = ForwardDiff.gradient(total_fn, θ_hat)
        ∇_ind = ForwardDiff.gradient(indirect_fn, θ_hat)

        direct_se[k] = sqrt(max(zero(T), dot(∇_dir, Σ_θ, ∇_dir)))
        total_se[k] = sqrt(max(zero(T), dot(∇_tot, Σ_θ, ∇_tot)))
        indirect_se[k] = sqrt(max(zero(T), dot(∇_ind, Σ_θ, ∇_ind)))
    end

    base_names = model.coefnames[(offset + 1):(offset + K)]
    return SpatialEffectsResult{T}(
        base_names, direct, direct_se, indirect, indirect_se, total, total_se
    )
end

function Base.show(io::IO, m::SpatialPanelModel{V, FE, A}) where {V, FE, A}
    vname = V === SAR ? "SAR" : "SDM"
    aname = A === LeeYuTransformation ? "Lee-Yu QML" : "Direct QML"
    fname = FE === NoFE ? "None" : FE === IndivFE ? "Individual" : FE === TimeFE ? "Time" : "Two-Way"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Spatial Panel $vname Model — $aname — FE: $fname")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "ρ = %.4f    σ² = %.4f    Log-lik = %.2f    N = %d    T = %d\n", m.rho, m.sigma2, m.loglik, m.N, m.T_periods)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

StatsAPI.coef(m::SpatialPanelModel) = m.coef
StatsAPI.vcov(m::SpatialPanelModel) = m.vcov
StatsAPI.stderror(m::SpatialPanelModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::SpatialPanelModel) = m.loglik
StatsAPI.nobs(m::SpatialPanelModel) = m.N * m.T_periods
StatsAPI.dof_residual(m::SpatialPanelModel) = (m.N_eff * m.T_eff) - length(m.coef)
StatsAPI.residuals(m::SpatialPanelModel) = m.residuals
StatsAPI.coefnames(m::SpatialPanelModel) = m.coefnames

end # module
