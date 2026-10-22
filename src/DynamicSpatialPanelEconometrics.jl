module DynamicSpatialPanelEconometrics

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
       within_transform!, within_transform, has_time_fe,
       _normcdf, _print_coef_table

const LogDetMethod = SpatialUtils.LogDetMethod
const ExactLU = SpatialUtils.ExactLU
const ChebyshevApprox = SpatialUtils.ChebyshevApprox
const FixedEffectsSpec = SpatialUtils.FixedEffectsSpec
const IndivFE = SpatialUtils.IndivFE
const TwoWayFE = SpatialUtils.TwoWayFE

export SpatialVariant, SAR, SDM,
       FixedEffectsSpec, IndivFE, TwoWayFE,
       LogDetMethod, ExactLU, ChebyshevApprox,
       DynamicSpatialPanelModel, DynamicSpatialEffectsResult,
       fit_dynamic_spatial_panel, dynamic_spatial_effects

const SpatialVariant = SpatialUtils.SpatialVariant
const SAR = SpatialUtils.SAR
const SDM = SpatialUtils.SDM

"""
    DynamicSpatialPanelModel

A fitted dynamic spatial panel, returned by
[`fit_dynamic_spatial_panel`](@ref). Implements the
`StatsAPI.RegressionModel` interface.

The specification carries **three** dependence parameters, all reported
separately:

| Field | Term | Meaning |
|---|---|---|
| `gamma` | `y_{t-1}` | pure time lag |
| `eta` | `W*y_{t-1}` | space-time lag: last period's neighbours |
| `rho` | `W*y_t` | contemporaneous spatial lag |

# Other fields
- `coef`, `vcov`, `coefnames`, `sigma2`, `loglik`, `residuals`: as usual.
  `coefnames` runs `"y_lag"`, `"Wy_lag"`, the covariates, the `W_`-prefixed
  terms for an `SDM`, then `"ρ"`.
- `coef_uncorrected`, `sigma2_uncorrected`: the estimates before the bias
  correction, so you can see what it moved.
- `spectral_radius_A`: stability diagnostic for the implied system. The
  long-run effects only exist if this is below 1.
- `W`, `Z_tilde`, `y_tilde`, `trace_powers`: the weight matrix, the
  transformed design and response, and traces reused by
  [`dynamic_spatial_effects`](@ref).
- `N`, `T_total`, `T_post`: one period is consumed in forming the lag, so
  estimation uses `T_post = T_total - 1`.

Because dependence runs over both space and time, short-run and long-run
effects differ — see [`dynamic_spatial_effects`](@ref).
"""
struct DynamicSpatialPanelModel{
    V<:SpatialVariant,
    FE<:FixedEffectsSpec,
    T<:AbstractFloat,
    M<:AbstractSparseMatrix{T}
} <: StatsAPI.RegressionModel
    variant::V
    fe_spec::FE
    N::Int
    T_total::Int
    T_post::Int
    coef::Vector{T}
    coef_uncorrected::Vector{T}
    vcov::Matrix{T}
    gamma::T
    eta::T
    rho::T
    sigma2::T
    sigma2_uncorrected::T
    spectral_radius_A::T
    loglik::T
    W::M
    Z_tilde::Matrix{T}
    y_tilde::Vector{T}
    residuals::Vector{T}
    trace_powers::Vector{T}
    coefnames::Vector{String}
end

"""
    DynamicSpatialEffectsResult

Short-run and long-run direct, indirect and total effects for a dynamic
spatial panel, returned by [`dynamic_spatial_effects`](@ref).

The short-run effects are the impact within the period of the shock; the
long-run effects cumulate the time dynamics to the new steady state and are
larger whenever the autoregressive coefficient is positive.
"""
struct DynamicSpatialEffectsResult{T<:AbstractFloat}
    varnames::Vector{String}
    sr_direct::Vector{T}
    sr_direct_se::Vector{T}
    sr_indirect::Vector{T}
    sr_indirect_se::Vector{T}
    sr_total::Vector{T}
    sr_total_se::Vector{T}
    lr_direct::Vector{T}
    lr_direct_se::Vector{T}
    lr_indirect::Vector{T}
    lr_indirect_se::Vector{T}
    lr_total::Vector{T}
    lr_total_se::Vector{T}
end

"""
    fit_dynamic_spatial_panel(variant, fe_spec, W, X, y, N, T_total; method,
                              varnames, rho_bounds, n_bias_probes, rng)

Fit a dynamic spatial panel — one carrying both a time lag and a spatial
lag of the dependent variable — by bias-corrected quasi-maximum
likelihood.

With fixed effects and a lagged dependent variable the QML estimator
suffers the Nickell bias, of order `1/T`. The Yu, de Jong and Lee
analytical correction is applied, using stochastic probes to estimate the
traces the correction needs.

# Arguments
- `variant`: `SAR` or `SDM`.
- `fe_spec`: a `FixedEffectsSpec`; `IndivFE` or
  `TwoWayFE` are the meaningful choices here.
- `W`: `N x N` cross-sectional weight matrix.
- `X`, `y`: stacked over all `T_total` periods, units varying fastest.
- `N`, `T_total`: units and periods **including** the one consumed by the
  lag; estimation uses `T_total - 1`.

# Keywords
- `n_bias_probes`, `rng`: probes used by the bias correction, and the
  random source for them. Results vary slightly with `rng`; pass a seeded
  one for reproducibility.
- `method`, `varnames`, `rho_bounds`: as for `fit_spatial_panel`.

# Returns
A [`DynamicSpatialPanelModel`](@ref).

!!! warning
    Same stacking requirement as `fit_spatial_panel`: units vary
    fastest within each period. The bias correction also assumes `T` is
    small relative to `N`; with long panels the correction matters less and
    the uncorrected estimates are already close.

# Examples
```julia
m = fit_dynamic_spatial_panel(SAR(), IndivFE(), W, X, y, N, T_total;
                              rng = MersenneTwister(1))
dynamic_spatial_effects(m)
```

# References
Yu, J., de Jong, R. & Lee, L.-F. (2008). *Journal of Econometrics*, 146(1).
"""
function fit_dynamic_spatial_panel(
    variant::V,
    fe_spec::FE,
    W::M,
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    N::Int,
    T_total::Int;
    method::LogDetMethod=ChebyshevApprox(order=6, n_probes=0),
    varnames::Union{Nothing, Vector{String}}=nothing,
    rho_bounds::Tuple{T, T}=(T(-0.995), T(0.995)),
    n_bias_probes::Int=80,
    rng::AbstractRNG=Random.default_rng()
) where {V<:SpatialVariant, FE<:FixedEffectsSpec, T<:AbstractFloat, M<:AbstractSparseMatrix{T}}

    NT_total, K = size(X)
    T_post = T_total - 1
    NT_post = N * T_post
    vnames = isnothing(varnames) ? ["x$i" for i in 1:K] : varnames

    Wy_all = similar(y)
    panel_spatial_lag!(Wy_all, W, y, N, T_total)

    idx_post = (N + 1):NT_total
    idx_lag  = 1:(NT_total - N)

    y_post   = y[idx_post]
    Wy_post  = Wy_all[idx_post]
    y_lag    = y[idx_lag]
    Wy_lag   = Wy_all[idx_lag]
    X_post   = X[idx_post, :]

    if variant isa SAR
        Z_post = hcat(y_lag, Wy_lag, X_post)
        coefnames = vcat(["y_lag", "Wy_lag"], vnames, ["ρ"])
    else
        WX_post = panel_spatial_lag(W, X_post, N, T_post)
        w_names = ["W_" * v for v in vnames]
        Z_post = hcat(y_lag, Wy_lag, X_post, WX_post)
        coefnames = vcat(["y_lag", "Wy_lag"], vnames, w_names, ["ρ"])
    end

    y_tilde  = within_transform!(copy(y_post), N, T_post, fe_spec)
    Wy_tilde = within_transform!(copy(Wy_post), N, T_post, fe_spec)
    Z_tilde  = within_transform(Z_post, N, T_post, fe_spec)

    qr_Z = qr(Z_tilde)
    δ0 = qr_Z \ y_tilde
    δd = qr_Z \ Wy_tilde

    e0 = y_tilde .- Z_tilde * δ0
    ed = Wy_tilde .- Z_tilde * δd

    e0_e0 = dot(e0, e0)
    e0_ed = dot(e0, ed)
    ed_ed = dot(ed, ed)

    if method isa ChebyshevApprox
        τ = precompute_chebyshev_traces(W, method)
        logdet_fn = ρ -> chebyshev_logdet(ρ, τ)
    else
        I_sparse = sparse(one(T) * I, N, N)
        logdet_fn = ρ -> first(logabsdet(lu(I_sparse - ρ * W)))
    end

    const_term = -T(0.5) * NT_post * (log(T(2π)) + one(T))
    function concentrated_dyn_ll(ρ)
        rss = e0_e0 - 2 * ρ * e0_ed + (ρ^2) * ed_ed
        σ2 = rss / NT_post
        return const_term + logdet_fn(ρ) - T(0.5) * NT_post * log(σ2)
    end

    ρ_raw, max_ll = brent_maximize(concentrated_dyn_ll, rho_bounds[1], rho_bounds[2])
    δ_raw = δ0 .- ρ_raw .* δd
    res_raw = e0 .- ρ_raw .* ed
    σ2_raw = dot(res_raw, res_raw) / NT_post

    μ_traces = precompute_neumann_traces(W; max_power=60, n_probes=250, rng=rng)

    vcov_mat, δ_bc, ρ_bc, σ2_bc, res_bc, spec_rad = yudjlee_bias_correction(
        fe_spec, Z_tilde, y_tilde, Wy_tilde, W,
        δ_raw, ρ_raw, σ2_raw, N, T_post, μ_traces, n_bias_probes, rng
    )

    return DynamicSpatialPanelModel{V, FE, T, M}(
        variant, fe_spec, N, T_total, T_post,
        vcat(δ_bc, ρ_bc), vcat(δ_raw, ρ_raw),
        vcov_mat, δ_bc[1], δ_bc[2], ρ_bc, σ2_bc, σ2_raw, spec_rad, max_ll,
        W, Z_tilde, y_tilde, res_bc, μ_traces, coefnames
    )
end

function yudjlee_bias_correction(
    fe_spec::FixedEffectsSpec,
    Z_tilde::Matrix{T},
    y_tilde::Vector{T},
    Wy_tilde::Vector{T},
    W::AbstractSparseMatrix{T},
    δ_raw::Vector{T},
    ρ_raw::T,
    σ2_raw::T,
    N::Int,
    T_post::Int,
    μ_traces::Vector{T},
    n_probes::Int,
    rng::AbstractRNG
) where {T<:AbstractFloat}

    P = length(δ_raw)
    γ_raw = δ_raw[1]
    η_raw = δ_raw[2]

    I_sparse = sparse(one(T) * I, N, N)
    S_lu = lu(I_sparse - ρ_raw * W)

    Zδ = Z_tilde * δ_raw
    GZδ = similar(Zδ)
    Zδ_mat = reshape(Zδ, N, T_post)
    GZδ_mat = reshape(GZδ, N, T_post)
    mul!(GZδ_mat, W, S_lu \ Zδ_mat)
    within_transform!(GZδ, N, T_post, fe_spec)

    tr_G = zero(T)
    for m in 1:(length(μ_traces) - 1)
        tr_G += (ρ_raw^(m - 1)) * (N * μ_traces[m + 1])
    end

    tr_G2 = zero(T)
    for m in 1:(length(μ_traces) - 2)
        tr_G2 += m * (ρ_raw^(m - 1)) * (N * μ_traces[m + 2])
    end

    if has_time_fe(fe_spec)
        eig_1 = one(T) / (one(T) - ρ_raw)
        tr_G  -= eig_1
        tr_G2 -= eig_1^2
    end

    V0 = Matrix{T}(undef, N, n_probes)
    rand!(rng, V0, (-one(T), one(T)))

    V_proj = has_time_fe(fe_spec) ? (V0 .- mean(V0; dims=1)) : V0
    GV = S_lu \ (W * V_proj)
    has_time_fe(fe_spec) && (GV .-= mean(GV; dims=1))
    tr_GtG = dot(GV, GV) / n_probes

    U_curr = S_lu \ V0
    H_V0   = ((T_post - 1) / T(T_post)) .* U_curr
    WU_buf = similar(U_curr)

    for t in 2:(T_post - 1)
        mul!(WU_buf, W, U_curr)
        @. WU_buf = γ_raw * U_curr + η_raw * WU_buf
        U_curr = S_lu \ WU_buf
        weight_t = (T_post - t) / T(T_post)
        @. H_V0 += weight_t * U_curr
    end

    u_prev_norm = norm(@view U_curr[:, 1])
    mul!(WU_buf, W, U_curr)
    @. WU_buf = γ_raw * U_curr + η_raw * WU_buf
    U_next = S_lu \ WU_buf
    spec_rad = norm(@view U_next[:, 1]) / max(u_prev_norm, T(1e-12))

    tr_H = dot(V0, H_V0) / n_probes
    WH_V0 = W * H_V0
    tr_WH = dot(V0, WH_V0) / n_probes

    AH_V0  = S_lu \ (γ_raw .* H_V0 .+ η_raw .* WH_V0)
    GAH_V0 = W * (S_lu \ AH_V0)
    tr_GAH = dot(V0, GAH_V0) / n_probes

    NT_post = N * T_post
    I_δδ = (Z_tilde' * Z_tilde) ./ σ2_raw
    I_δρ = (Z_tilde' * GZδ) ./ σ2_raw
    I_ρρ = (dot(GZδ, GZδ) / σ2_raw) + T_post * (tr_G2 + tr_GtG)
    I_ρσ2 = (T_post * tr_G) / σ2_raw
    I_σ2σ2 = NT_post / (2 * σ2_raw^2)

    Info = zeros(T, P + 2, P + 2)
    Info[1:P, 1:P] .= I_δδ
    Info[1:P, P + 1] .= I_δρ
    Info[P + 1, 1:P] .= I_δρ
    Info[P + 1, P + 1] = I_ρρ
    Info[P + 1, P + 2] = I_ρσ2
    Info[P + 2, P + 1] = I_ρσ2
    Info[P + 2, P + 2] = I_σ2σ2

    V_full = inv(Symmetric(Info))

    score_bias = zeros(T, P + 2)
    tr_G_full = has_time_fe(fe_spec) ? (tr_G + one(T) / (one(T) - ρ_raw)) : tr_G
    score_bias[1]     += -tr_H
    score_bias[2]     += -tr_WH
    score_bias[P + 1] += -(tr_GAH + tr_G_full)
    score_bias[P + 2] += -T(N) / (2 * σ2_raw)

    if has_time_fe(fe_spec)
        s_inv_1 = one(T) / (one(T) - ρ_raw)
        a_1 = (γ_raw + η_raw) * s_inv_1
        h_1 = zero(T)
        for t in 1:(T_post - 1)
            h_1 += ((T_post - t) / T(T_post)) * (a_1^(t - 1))
        end
        H_1 = h_1 * s_inv_1
        G_1 = s_inv_1

        score_bias[1]     += -T(T_post) * H_1 + H_1
        score_bias[2]     += -T(T_post) * H_1 + H_1
        score_bias[P + 1] += -T(T_post) * (G_1 * a_1 * H_1 + G_1) + (G_1 * a_1 * H_1 + G_1)
        score_bias[P + 2] += -T(T_post - 1) / (2 * σ2_raw)
    end

    θ_raw_full = vcat(δ_raw, ρ_raw, σ2_raw)
    θ_bc_full  = θ_raw_full .- (V_full * score_bias)

    δ_bc  = θ_bc_full[1:P]
    ρ_bc  = clamp(θ_bc_full[P + 1], T(-0.995), T(0.995))

    res_bc = y_tilde .- ρ_bc .* Wy_tilde .- Z_tilde * δ_bc
    N_eff = has_time_fe(fe_spec) ? (N - 1) : N
    T_eff = T_post - 1
    σ2_bc = max(θ_bc_full[P + 2], dot(res_bc, res_bc) / (N_eff * T_eff))

    V_coef = V_full[1:(P + 1), 1:(P + 1)] .* (σ2_bc / σ2_raw)
    return V_coef, δ_bc, ρ_bc, σ2_bc, res_bc, spec_rad
end

"""
    dynamic_spatial_effects(model; max_power = 50, vcov_mat = model.vcov)
    dynamic_spatial_effects(model, spec::AbstractRobustVCOV; max_power = 50)

Short-run and long-run marginal effects for a dynamic spatial panel.

The short-run effects are the within-period response through
`(I - rho*W)^{-1}`, so the total is `(beta + theta) / (1 - rho)`.

The long-run effects cumulate the time dynamics to the steady state, which
brings in both lag parameters: the total becomes

```math
(beta + theta) / (1 - gamma - rho - eta)
```

and the direct effect is evaluated at the long-run spatial parameter
`(rho + eta) / (1 - gamma)`, scaled by `1 / (1 - gamma)`. Each of the six
quantities comes with a delta-method standard error.

# Returns
A [`DynamicSpatialEffectsResult`](@ref), whose fields are prefixed `sr_`
for short-run and `lr_` for long-run.

!!! warning
    The long-run effects exist only if `gamma + rho + eta < 1`. As that sum
    approaches 1 the denominator goes to zero, the effects diverge, and the
    delta-method standard errors become unreliable well before that point.
    Check `model.spectral_radius_A` first.
"""
function dynamic_spatial_effects(
    model::DynamicSpatialPanelModel{V, FE, T};
    max_power::Int=50,
    vcov_mat::AbstractMatrix{T}=model.vcov
) where {V<:SpatialVariant, FE<:FixedEffectsSpec, T<:AbstractFloat}

    μ = model.trace_powers
    q = min(max_power, length(μ) - 2)
    θ_hat = model.coef
    Σ_θ = vcov_mat

    P_exog = length(θ_hat) - 3
    K = V === SAR ? P_exog : div(P_exog, 2)

    sr_dir = zeros(T, K); sr_dir_se = zeros(T, K)
    sr_ind = zeros(T, K); sr_ind_se = zeros(T, K)
    sr_tot = zeros(T, K); sr_tot_se = zeros(T, K)

    lr_dir = zeros(T, K); lr_dir_se = zeros(T, K)
    lr_ind = zeros(T, K); lr_ind_se = zeros(T, K)
    lr_tot = zeros(T, K); lr_tot_se = zeros(T, K)

    for k in 1:K
        sr_dir_fn = θ -> begin
            ρ = θ[end]
            β_k = θ[2 + k]
            γ_k = V === SDM ? θ[2 + K + k] : zero(eltype(θ))
            acc = zero(eltype(θ))
            for m in 0:q
                acc += (ρ^m) * (β_k * μ[m + 1] + γ_k * μ[m + 2])
            end
            return acc
        end

        sr_tot_fn = θ -> begin
            ρ = θ[end]
            β_k = θ[2 + k]
            γ_k = V === SDM ? θ[2 + K + k] : zero(eltype(θ))
            return (β_k + γ_k) / (one(eltype(θ)) - ρ)
        end

        sr_ind_fn = θ -> sr_tot_fn(θ) - sr_dir_fn(θ)

        lr_dir_fn = θ -> begin
            γ = θ[1]
            η = θ[2]
            ρ = θ[end]
            β_k = θ[2 + k]
            γ_k = V === SDM ? θ[2 + K + k] : zero(eltype(θ))
            denom = one(eltype(θ)) - γ
            ρ_lr = (ρ + η) / denom
            acc = zero(eltype(θ))
            for m in 0:q
                acc += (ρ_lr^m) * (β_k * μ[m + 1] + γ_k * μ[m + 2])
            end
            return acc / denom
        end

        lr_tot_fn = θ -> begin
            γ = θ[1]
            η = θ[2]
            ρ = θ[end]
            β_k = θ[2 + k]
            γ_k = V === SDM ? θ[2 + K + k] : zero(eltype(θ))
            return (β_k + γ_k) / (one(eltype(θ)) - γ - ρ - η)
        end

        lr_ind_fn = θ -> lr_tot_fn(θ) - lr_dir_fn(θ)

        sr_dir[k] = sr_dir_fn(θ_hat); sr_tot[k] = sr_tot_fn(θ_hat); sr_ind[k] = sr_ind_fn(θ_hat)
        lr_dir[k] = lr_dir_fn(θ_hat); lr_tot[k] = lr_tot_fn(θ_hat); lr_ind[k] = lr_ind_fn(θ_hat)

        ∇_sr_d = ForwardDiff.gradient(sr_dir_fn, θ_hat)
        ∇_sr_t = ForwardDiff.gradient(sr_tot_fn, θ_hat)
        ∇_sr_i = ForwardDiff.gradient(sr_ind_fn, θ_hat)

        ∇_lr_d = ForwardDiff.gradient(lr_dir_fn, θ_hat)
        ∇_lr_t = ForwardDiff.gradient(lr_tot_fn, θ_hat)
        ∇_lr_i = ForwardDiff.gradient(lr_ind_fn, θ_hat)

        sr_dir_se[k] = sqrt(max(zero(T), dot(∇_sr_d, Σ_θ, ∇_sr_d)))
        sr_tot_se[k] = sqrt(max(zero(T), dot(∇_sr_t, Σ_θ, ∇_sr_t)))
        sr_ind_se[k] = sqrt(max(zero(T), dot(∇_sr_i, Σ_θ, ∇_sr_i)))

        lr_dir_se[k] = sqrt(max(zero(T), dot(∇_lr_d, Σ_θ, ∇_lr_d)))
        lr_tot_se[k] = sqrt(max(zero(T), dot(∇_lr_t, Σ_θ, ∇_lr_t)))
        lr_ind_se[k] = sqrt(max(zero(T), dot(∇_lr_i, Σ_θ, ∇_lr_i)))
    end

    base_names = model.coefnames[3:(2 + K)]
    return DynamicSpatialEffectsResult{T}(
        base_names,
        sr_dir, sr_dir_se, sr_ind, sr_ind_se, sr_tot, sr_tot_se,
        lr_dir, lr_dir_se, lr_ind, lr_ind_se, lr_tot, lr_tot_se
    )
end

function Base.show(io::IO, m::DynamicSpatialPanelModel{V, FE}) where {V, FE}
    vname = V === SAR ? "SAR" : "SDM"
    fname = FE === IndivFE ? "Individual" : "Two-Way"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Dynamic Spatial Panel $vname — Bias-Corrected QML — FE: $fname")
    println(io, "───────────────────────────────────────────────────────────────────────")
    _print_coef_table(io, m.coefnames, m.coef, sqrt.(max.(0.0, diag(m.vcov))))
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "ρ = %.4f    γ = %.4f    η = %.4f    σ² = %.4f    Log-lik = %.2f\n", m.rho, m.gamma, m.eta, m.sigma2, m.loglik)
    @printf(io, "N = %d    T = %d (post = %d)    Spectral radius = %.4f\n", m.N, m.T_total, m.T_post, m.spectral_radius_A)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

function Base.show(io::IO, e::DynamicSpatialEffectsResult)
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Dynamic Spatial Effects — Short-Run and Long-Run")
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "%-10s | %8s | %8s | %8s | %8s | %8s | %8s\n", "", "SR Dir", "SE", "SR Ind", "SE", "SR Tot", "SE")
    println(io, "───────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────")
    for i in eachindex(e.varnames)
        @printf(io, "%-10s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            e.varnames[i], e.sr_direct[i], e.sr_direct_se[i], e.sr_indirect[i], e.sr_indirect_se[i], e.sr_total[i], e.sr_total_se[i])
    end
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "%-10s | %8s | %8s | %8s | %8s | %8s | %8s\n", "", "LR Dir", "SE", "LR Ind", "SE", "LR Tot", "SE")
    println(io, "───────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────")
    for i in eachindex(e.varnames)
        @printf(io, "%-10s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n",
            e.varnames[i], e.lr_direct[i], e.lr_direct_se[i], e.lr_indirect[i], e.lr_indirect_se[i], e.lr_total[i], e.lr_total_se[i])
    end
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

StatsAPI.coef(m::DynamicSpatialPanelModel) = m.coef
StatsAPI.vcov(m::DynamicSpatialPanelModel) = m.vcov
StatsAPI.stderror(m::DynamicSpatialPanelModel) = sqrt.(max.(0.0, diag(m.vcov)))
StatsAPI.loglikelihood(m::DynamicSpatialPanelModel) = m.loglik
StatsAPI.nobs(m::DynamicSpatialPanelModel) = m.N * m.T_post
StatsAPI.dof_residual(m::DynamicSpatialPanelModel) = m.N * m.T_post - length(m.coef)
StatsAPI.residuals(m::DynamicSpatialPanelModel) = m.residuals
StatsAPI.coefnames(m::DynamicSpatialPanelModel) = m.coefnames

end # module
