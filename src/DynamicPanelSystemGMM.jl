module DynamicPanelSystemGMM

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using StatsAPI

export GMMVariant, DifferenceGMM, SystemGMM,
       SystemGMMModel, fit_dynamic_gmm

"""
    GMMVariant

Which moment conditions a dynamic panel GMM estimator uses. Subtypes:
[`DifferenceGMM`](@ref) and [`SystemGMM`](@ref).
"""
abstract type GMMVariant end
"""
    DifferenceGMM()

Arellano-Bond difference GMM: first-difference away the unit effects and
instrument with lagged levels.

Consistent for a dynamic panel with fixed effects, but the lagged-level
instruments become weak as the autoregressive parameter approaches 1 —
a highly persistent series provides little information about its own
differences. [`SystemGMM`](@ref) adds moment conditions to address exactly
that.

# References
Arellano, M. & Bond, S. (1991). *Review of Economic Studies*, 58(2).
"""
struct DifferenceGMM <: GMMVariant end
"""
    SystemGMM()

Blundell-Bond system GMM: the difference equations of
[`DifferenceGMM`](@ref) together with the levels equations, instrumented
with lagged differences.

Much better behaved when the series is persistent, which is the usual case.
The extra moment conditions require that changes in the regressors be
uncorrelated with the unit effects — a real assumption, and the thing to
examine if the Sargan test rejects.

# References
Blundell, R. & Bond, S. (1998). *Journal of Econometrics*, 87(1).
"""
struct SystemGMM     <: GMMVariant end

"""
    SystemGMMModel

A fitted dynamic panel GMM model, returned by
[`fit_dynamic_gmm`](@ref). Implements the `StatsAPI.RegressionModel`
interface.

# Fields
- `variant`: [`DifferenceGMM`](@ref) or [`SystemGMM`](@ref).
- `coef_twostep`, `coef_onestep`: the two-step and one-step estimates.
- `vcov_windmeijer`: two-step covariance with the Windmeijer finite-sample
  correction — **the one to report**.
- `vcov_naive`: uncorrected two-step covariance, kept for comparison.
  Without the correction two-step standard errors are severely biased
  downwards, so the gap between this and `vcov_windmeijer` is worth a look.
- `vcov_onestep`: covariance of the one-step estimator.
- `hansen_j`, `hansen_df`: Hansen J over-identification test.
- `ar1_z`, `ar2_z`: Arellano-Bond tests for first- and second-order
  residual autocorrelation.
- `n_instruments`, `collapsed`: how many instruments were used and whether
  the instrument matrix was collapsed.
- `N`, `T_total`, `coefnames`.

!!! note
    Read the three diagnostics together. `ar1_z` is *expected* to reject,
    since differencing induces first-order correlation by construction;
    `ar2_z` should **not** reject, or the lag structure is misspecified and
    the instruments are invalid. Then check `hansen_j`, bearing in mind the
    `n_instruments` caveat on [`fit_dynamic_gmm`](@ref).

# References
Windmeijer, F. (2005). *Journal of Econometrics*, 126(1).
"""
struct SystemGMMModel{V<:GMMVariant, T<:AbstractFloat} <: StatsAPI.RegressionModel
    variant::V
    N::Int
    T_total::Int
    collapsed::Bool
    n_instruments::Int
    coef_twostep::Vector{T}
    coef_onestep::Vector{T}
    vcov_windmeijer::Matrix{T}
    vcov_naive::Matrix{T}
    vcov_onestep::Matrix{T}
    hansen_j::T
    hansen_df::Int
    ar1_z::T
    ar2_z::T
    coefnames::Vector{String}
end

function build_gmm_system(
    variant::GMMVariant,
    Y_mat::Matrix{T},
    X_ten::Array{T, 3},
    max_lag::Int,
    collapsed::Bool
) where {T<:AbstractFloat}

    N, T_tot, K = size(X_ten)
    T_diff = T_tot - 2
    @assert T_diff >= 1 "Need at least T = 3 time periods for Dynamic Panel GMM"

    if collapsed
        L_y_diff = min(max_lag, T_tot - 2)
    else
        L_y_diff = sum(min(t - 2, max_lag) for t in 3:T_tot)
    end

    is_sys = variant isa SystemGMM
    L_y_lev = is_sys ? (collapsed ? 1 : T_diff) : 0
    L_total = L_y_diff + L_y_lev + K

    M_eq = is_sys ? 2 * T_diff : T_diff
    P_par = 1 + K

    X_list = Vector{Matrix{T}}(undef, N)
    y_list = Vector{Vector{T}}(undef, N)
    Z_list = Vector{Matrix{T}}(undef, N)

    for i in 1:N
        Xi = zeros(T, M_eq, P_par)
        yi = zeros(T, M_eq)
        Zi = zeros(T, M_eq, L_total)

        col_cursor = 1
        for eq_idx in 1:T_diff
            t = eq_idx + 2
            yi[eq_idx] = Y_mat[i, t] - Y_mat[i, t - 1]
            Xi[eq_idx, 1] = Y_mat[i, t - 1] - Y_mat[i, t - 2]
            for k in 1:K
                dx_k = X_ten[i, t, k] - X_ten[i, t - 1, k]
                Xi[eq_idx, 1 + k] = dx_k
                Zi[eq_idx, L_y_diff + L_y_lev + k] = dx_k
            end

            n_lags_t = min(t - 2, max_lag)
            if collapsed
                for s in 1:n_lags_t
                    Zi[eq_idx, s] = Y_mat[i, t - 1 - s]
                end
            else
                for s in 1:n_lags_t
                    Zi[eq_idx, col_cursor] = Y_mat[i, t - 1 - s]
                    col_cursor += 1
                end
            end
        end

        if is_sys
            for eq_idx in 1:T_diff
                row = T_diff + eq_idx
                t = eq_idx + 2
                yi[row] = Y_mat[i, t]
                Xi[row, 1] = Y_mat[i, t - 1]
                for k in 1:K
                    x_k = X_ten[i, t, k]
                    Xi[row, 1 + k] = x_k
                    Zi[row, L_y_diff + L_y_lev + k] = x_k
                end

                dy_lag = Y_mat[i, t - 1] - Y_mat[i, t - 2]
                if collapsed
                    Zi[row, L_y_diff + 1] = dy_lag
                else
                    Zi[row, L_y_diff + eq_idx] = dy_lag
                end
            end
        end

        X_list[i] = Xi
        y_list[i] = yi
        Z_list[i] = Zi
    end

    H = zeros(T, M_eq, M_eq)
    for r in 1:T_diff
        H[r, r] = T(2)
        r > 1 && (H[r, r - 1] = -one(T))
        r < T_diff && (H[r, r + 1] = -one(T))
    end
    if is_sys
        for r in (T_diff + 1):M_eq
            H[r, r] = one(T)
        end
    end

    return X_list, y_list, Z_list, H, T_diff, L_total
end

"""
    fit_dynamic_gmm(variant, Y_mat, X_ten; max_lag = 4, collapsed = false,
                    varnames = nothing)

Fit a dynamic panel by Arellano-Bond or Blundell-Bond GMM.

# Arguments
- `variant`: [`DifferenceGMM`](@ref) or [`SystemGMM`](@ref).
- `Y_mat`: `N x T` matrix of the dependent variable — units by periods, not
  stacked as the spatial panel estimators expect.
- `X_ten`: `N x T x K` array of covariates.

# Keywords
- `max_lag`: how many lags to use as instruments. More lags mean more
  moment conditions and, with many periods, an instrument count that grows
  quadratically and overfits.
- `collapsed`: collapse the instrument matrix, holding the count down. Worth
  turning on whenever `T` is more than small.
- `varnames`: covariate names.

# Returns
A [`SystemGMMModel`](@ref).

!!! warning
    Instrument proliferation is the standard failure here: too many
    instruments relative to `N` makes the over-identification test lose all
    power, so a comfortable p-value can be meaningless. Keep the instrument
    count well below `N`, using `max_lag` and `collapsed`.

!!! note
    Note the data layout: `Y_mat` is `N x T` and `X_ten` is `N x T x K`,
    unlike the stacked vectors the spatial panel estimators take.

# References
Roodman, D. (2009). How to do xtabond2. *Stata Journal*, 9(1).
"""
function fit_dynamic_gmm(
    variant::V,
    Y_mat::Matrix{T},
    X_ten::Array{T, 3};
    max_lag::Int=4,
    collapsed::Bool=false,
    varnames::Union{Nothing, Vector{String}}=nothing
) where {V<:GMMVariant, T<:AbstractFloat}

    N, T_tot, K = size(X_ten)
    X_list, y_list, Z_list, H, T_diff, L = build_gmm_system(variant, Y_mat, X_ten, max_lag, collapsed)
    P = 1 + K

    S_ZX = zeros(T, L, P)
    S_Zy = zeros(T, L)
    Ω_1  = zeros(T, L, L)

    for i in 1:N
        Zi = Z_list[i]
        S_ZX .+= Zi' * X_list[i]
        S_Zy .+= Zi' * y_list[i]
        Ω_1  .+= Zi' * H * Zi
    end

    W_1 = pinv(Symmetric(Ω_1))
    G_1 = S_ZX' * W_1 * S_ZX
    G_1_inv = inv(Symmetric(G_1))
    θ_1 = G_1_inv * (S_ZX' * (W_1 * S_Zy))

    Ω_2 = zeros(T, L, L)
    for i in 1:N
        e1_i = y_list[i] .- X_list[i] * θ_1
        z_e1 = Z_list[i]' * e1_i
        mul!(Ω_2, z_e1, z_e1', one(T), one(T))
    end

    W_2 = pinv(Symmetric(Ω_2))
    G_2 = S_ZX' * W_2 * S_ZX
    V_naive = inv(Symmetric(G_2))
    V_1 = G_1_inv * (S_ZX' * W_1 * Ω_2 * W_1 * S_ZX) * G_1_inv

    θ_2 = V_naive * (S_ZX' * (W_2 * S_Zy))
    g_2 = S_Zy .- S_ZX * θ_2

    u_vec = W_2 * g_2
    A_pre = -V_naive * (S_ZX' * W_2)

    windmeijer_psi = θ -> begin
        R_type = eltype(θ)
        ω_u = zeros(R_type, L)
        for i in 1:N
            ei = y_list[i] .- X_list[i] * θ
            ze = Z_list[i]' * ei
            scale = dot(ze, u_vec)
            @. ω_u += ze * scale
        end
        return A_pre * ω_u
    end

    D_mat = ForwardDiff.jacobian(windmeijer_psi, θ_1)
    V_wind = V_naive .+ (D_mat * V_1 * V_naive) .+ (V_naive * V_1 * D_mat') .+ (D_mat * V_1 * D_mat')

    hansen_j = dot(g_2, W_2, g_2)
    hansen_df = max(0, L - P)

    ar1_z, ar2_z = compute_ab_serial_correlation(X_list, y_list, Z_list, θ_2, W_2, S_ZX, V_wind, T_diff)

    vnames = isnothing(varnames) ? ["x$k" for k in 1:K] : varnames
    cnames = vcat(["y_lag"], vnames)

    return SystemGMMModel{V, T}(
        variant, N, T_tot, collapsed, L,
        θ_2, θ_1, V_wind, V_naive, V_1,
        hansen_j, hansen_df, ar1_z, ar2_z, cnames
    )
end

function compute_ab_serial_correlation(
    X_list::Vector{Matrix{T}},
    y_list::Vector{Vector{T}},
    Z_list::Vector{Matrix{T}},
    θ::Vector{T},
    W_2::Matrix{T},
    S_ZX::Matrix{T},
    V_θ::Matrix{T},
    T_diff::Int
) where {T<:AbstractFloat}

    N = length(X_list)
    P = length(θ)

    function ar_stat(m::Int)
        T_diff <= m && return T(NaN)
        num = zero(T)
        denom_var = zero(T)
        w_X = zeros(T, P)
        M_2 = W_2 * S_ZX * V_θ
        L = size(Z_list[1], 2)
        ze_sum = zeros(T, L)

        for i in 1:N
            @views e_diff = y_list[i][1:T_diff] .- X_list[i][1:T_diff, :] * θ
            @views X_diff = X_list[i][1:T_diff, :]

            c_i = zero(T)
            for t in (m + 1):T_diff
                c_i += e_diff[t - m] * e_diff[t]
                @. w_X += e_diff[t - m] * X_diff[t, :]
            end
            num += c_i
            denom_var += c_i^2

            ei_full = y_list[i] .- X_list[i] * θ
            ze_i = Z_list[i]' * ei_full
            @. ze_sum += ze_i * c_i
        end

        avar = denom_var - 2 * dot(w_X, M_2' * ze_sum) + dot(w_X, V_θ, w_X)
        return num / sqrt(max(avar, T(1e-12)))
    end

    return ar_stat(1), ar_stat(2)
end

function Base.show(io::IO, m::SystemGMMModel{V}) where {V}
    vname = V === DifferenceGMM ? "Difference GMM" : "System GMM"
    println(io, "═══════════════════════════════════════════════════════════════════════")
    println(io, "Dynamic Panel $vname $(m.collapsed ? "(Collapsed)" : "")")
    println(io, "───────────────────────────────────────────────────────────────────────")
    ses = sqrt.(max.(0.0, diag(m.vcov_windmeijer)))
    @printf(io, "%-16s | %11s | %11s | %10s\n", "Parameter", "Estimate", "Std. Err", "z-stat")
    println(io, "─────────────────┼─────────────┼─────────────┼────────────")
    for i in eachindex(m.coefnames)
        z = ses[i] > 0.0 ? m.coef_twostep[i] / ses[i] : NaN
        @printf(io, "%-16s | %11.4f | %11.4f | %10.3f\n", m.coefnames[i], m.coef_twostep[i], ses[i], z)
    end
    println(io, "───────────────────────────────────────────────────────────────────────")
    @printf(io, "N = %d    T = %d    Instruments = %d\n", m.N, m.T_total, m.n_instruments)
    @printf(io, "Hansen J = %.4f (df = %d)    AR(1) z = %.3f    AR(2) z = %.3f\n", m.hansen_j, m.hansen_df, m.ar1_z, m.ar2_z)
    println(io, "═══════════════════════════════════════════════════════════════════════")
end

StatsAPI.coef(m::SystemGMMModel) = m.coef_twostep
StatsAPI.vcov(m::SystemGMMModel) = m.vcov_windmeijer
StatsAPI.stderror(m::SystemGMMModel) = sqrt.(max.(0.0, diag(m.vcov_windmeijer)))
StatsAPI.nobs(m::SystemGMMModel) = m.N * (m.T_total - 2)
StatsAPI.dof_residual(m::SystemGMMModel) = m.N * (m.T_total - 2) - length(m.coef_twostep)
StatsAPI.coefnames(m::SystemGMMModel) = m.coefnames

end # module
