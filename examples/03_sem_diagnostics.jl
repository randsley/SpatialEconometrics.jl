# ═══════════════════════════════════════════════════════════════════════
# Example 3: Spatial Error Model & Diagnostics Workflow
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Complete spatial econometrics workflow:
#     1. Explore data with diagnostics (Moran, LM tests)
#     2. Choose model based on LM test results
#     3. Estimate SAR vs SEM
#     4. Panel SEM with fixed effects
#   - LISA local spatial analysis

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays

# ─── Generate SEM data (spatial error, NOT spatial lag) ───────────────

rng = MersenneTwister(99)
W = make_rook_weights(15); N = size(W, 1)
X = randn(rng, N, 2)
β_true = [1.0, 2.0, -1.0]      # intercept, β₁, β₂
λ_true = 0.6                     # spatial error parameter

# SEM DGP: y = Xβ + u,  u = λWu + ε
u = (I - λ_true * W) \ (0.5 .* randn(rng, N))
y = β_true[1] .+ X * β_true[2:3] .+ u

# ─── Step 1: Diagnostics — is there spatial autocorrelation? ─────────

println("═══ Step 1: Global Diagnostics ═══")
println(moran_global(y, W))
println(geary_c(y, W))
println()

# ─── Step 2: LM tests — SAR or SEM? ─────────────────────────────────

println("═══ Step 2: LM Specification Tests ═══")
X_ols = hcat(ones(N), X)
lms = lm_tests(y, X_ols, W)
println(lms.lag)
println(lms.error)
println(lms.robust_lag)
println(lms.robust_error)
println()

# Interpretation:
# - If Robust LM-Error is significant but Robust LM-Lag is not → SEM
# - If Robust LM-Lag is significant but Robust LM-Error is not → SAR
# - If both significant → consider SARAR (GS2SLS)

# ─── Step 3: Estimate SEM ────────────────────────────────────────────

println("═══ Step 3: Spatial Error Model ═══")
m_sem = fit_spatial_error(W, X, y; varnames=["income", "education"])
println(m_sem)
println("True λ = $λ_true,  Estimated λ = $(m_sem.lambda)")
println()

# ─── Compare with SAR (misspecified) ─────────────────────────────────

println("═══ Comparison: SAR Model (misspecified for this DGP) ═══")
const CS = SpatialEconometrics.CrossSection
m_sar = fit_spatial(CS.SAR(), W, X, y; varnames=["income", "education"])
println(m_sar)
println()

# ─── Step 4: LISA analysis ───────────────────────────────────────────

println("═══ Step 4: LISA (Local Moran's I) ═══")
lisa = moran_local(y, W; n_perm=499, rng=MersenneTwister(999))
println(lisa)
n_sig = count(lisa.pseudo_p .< 0.05)
println("Significant local clusters: $n_sig / $N")
println()

# ─── Panel SEM ────────────────────────────────────────────────────────

println("═══ Panel SEM with Individual FE ═══")
T_per = 6; NT = N * T_per
X_pan = randn(rng, NT, 2)
μ_i = randn(rng, N)
B_lu = lu(sparse(1.0I, N, N) - λ_true * W)
y_pan = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    u_t = B_lu \ (0.4 .* randn(rng, N))
    y_pan[idx] = μ_i .+ X_pan[idx, :] * [1.5, -0.7] .+ u_t
end

const SEM = SpatialEconometrics.SEMMod
m_sem_pan = fit_spatial_error(SEM.IndivFE(), W, X_pan, y_pan, N, T_per;
    varnames=["income", "education"])
println(m_sem_pan)
println("True λ = $λ_true,  Panel λ̂ = $(m_sem_pan.lambda)")
