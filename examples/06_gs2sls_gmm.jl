# ═══════════════════════════════════════════════════════════════════════
# Example 6: GS2SLS (SARAR) and System GMM
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - SARAR panel estimation with instrumental variables
#   - System GMM for dynamic panels (Blundell-Bond)
#   - Difference GMM (Arellano-Bond)
#   - Hansen J-test and AR serial correlation tests
#   - Formula interface for IV estimation

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays

# ═══════════════════════════════════════════════════════════════════════
# GS2SLS / SARAR Panel
# ═══════════════════════════════════════════════════════════════════════

println("═══ GS2SLS SARAR Panel with Endogenous Regressor ═══")

rng = MersenneTwister(404)
L = 10; N = L * L; T_tot = 6; NT = N * T_tot
W = make_rook_weights(L)

# Exogenous regressors
X_exog = randn(rng, NT, 2)

# Endogenous regressor with instruments
z1 = randn(rng, NT); z2 = randn(rng, NT)
tax = 0.5 .* z1 .- 0.3 .* z2 .+ 0.2 .* randn(rng, NT)

# True parameters
β_exog = [1.2, -0.6]; δ_endog = 0.8; ρ_true = 0.3; λ_true = 0.25

S_lu = lu(sparse(1.0I, N, N) - ρ_true * W)
B_lu = lu(sparse(1.0I, N, N) - λ_true * W)
y = zeros(NT)
for t in 1:T_tot
    idx = ((t - 1) * N + 1):(t * N)
    u_t = B_lu \ (0.4 .* randn(rng, N))  # spatially correlated errors
    y[idx] = S_lu \ (X_exog[idx, :] * β_exog .+ tax[idx] .* δ_endog .+ u_t)
end

# Matrix interface
m_gs = fit_gs2sls_panel(TwoWayFE(), W,
    X_exog, reshape(tax, NT, 1), hcat(z1, z2), y, N, T_tot;
    exog_names=["investment", "openness"], endog_names=["tax_rate"])
println(m_gs)
println()

# Formula interface
region_id = zeros(Int, NT); year_id = zeros(Int, NT)
for t in 1:T_tot
    idx = ((t - 1) * N + 1):(t * N)
    region_id[idx] .= 1:N; year_id[idx] .= 2020 + t
end
shuf = sortperm(1:NT; by = i -> (region_id[i], year_id[i]))
tbl = (region=region_id[shuf], year=year_id[shuf], y=y[shuf],
       investment=X_exog[shuf,1], openness=X_exog[shuf,2],
       tax_rate=tax[shuf], z1=z1[shuf], z2=z2[shuf])

println("═══ Formula Interface ═══")
m_dsl = fit_gs2sls_panel(
    @formula(y ~ investment + openness + endo(tax_rate, z1 + z2) + fe(region) + fe(year)),
    tbl, W; panel=:region, time=:year)
println(m_dsl)
println()

# ═══════════════════════════════════════════════════════════════════════
# System GMM (Blundell-Bond)
# ═══════════════════════════════════════════════════════════════════════

println("═══ System GMM (Blundell-Bond) ═══")

rng2 = MersenneTwister(505)
N_gmm = 100; T_gmm = 8; K_gmm = 1
γ_true = 0.4; β_gmm = [1.0]

μ_i = randn(rng2, N_gmm)
Y_mat = zeros(N_gmm, T_gmm)
X_ten = randn(rng2, N_gmm, T_gmm, K_gmm)

# Generate dynamic panel
for i in 1:N_gmm
    Y_mat[i, 1] = μ_i[i] + X_ten[i, 1, 1] * β_gmm[1] + randn(rng2)
end
for t in 2:T_gmm, i in 1:N_gmm
    Y_mat[i, t] = μ_i[i] + γ_true * Y_mat[i, t-1] + X_ten[i, t, 1] * β_gmm[1] + 0.5 * randn(rng2)
end

m_sys = fit_dynamic_gmm(SystemGMM(), Y_mat, X_ten;
    max_lag=3, varnames=["productivity"])
println(m_sys)
println()

println("True γ = $γ_true,  Estimated γ̂ = $(round(m_sys.coef_twostep[1]; digits=3))")
println("AR(1) z = $(round(m_sys.ar1_z; digits=3)) — should be significant")
println("AR(2) z = $(round(m_sys.ar2_z; digits=3)) — should NOT be significant")
println()

# ─── Difference GMM comparison ────────────────────────────────────────

println("═══ Difference GMM (Arellano-Bond) ═══")
m_diff = fit_dynamic_gmm(DifferenceGMM(), Y_mat, X_ten;
    max_lag=3, varnames=["productivity"])
println(m_diff)
