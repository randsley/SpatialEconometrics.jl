# ═══════════════════════════════════════════════════════════════════════
# Example 1: Cross-Sectional SAR & SDM
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - SAR and SDM estimation via concentrated maximum likelihood
#   - Chebyshev log-determinant approximation
#   - Direct, indirect, and total spatial effects with delta-method SEs
#   - Spatial diagnostics (Moran's I, LM tests)

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays

const CS = SpatialEconometrics.CrossSection

# ─── Generate synthetic spatial data ──────────────────────────────────

rng = MersenneTwister(42)
W = make_rook_weights(20)        # 400 units on a 20×20 grid, row-standardized
N = size(W, 1)

X = randn(rng, N, 2)
β_true = [2.0, 1.5, -1.0]       # intercept, β₁, β₂
θ_true = [0.8, -0.5]            # spatial lag coefficients (WX)
ρ_true = 0.55                    # spatial autoregressive parameter
σ_true = 0.50                    # error standard deviation

# DGP: y = (I - ρW)⁻¹ (β₀ + Xβ + WXθ + ε)
y = (I - ρ_true * W) \ (β_true[1] .+ X * β_true[2:3] .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))

# ─── Pre-estimation diagnostics ──────────────────────────────────────

println("═══ Pre-estimation Diagnostics ═══")
println(moran_global(y, W))
gc = geary_c(y, W)
println(gc)

lms = lm_tests(y, hcat(ones(N), X), W)
println(lms.lag)
println(lms.error)
println(lms.robust_lag)
println(lms.robust_error)
println()

# ─── Fit SAR model ───────────────────────────────────────────────────

println("═══ SAR Model ═══")
m_sar = fit_spatial(CS.SAR(), W, X, y;
    method = CS.ChebyshevApprox(order=12),
    varnames = ["income", "density"])
println(m_sar)
println()

# ─── Fit SDM model ───────────────────────────────────────────────────

println("═══ SDM Model ═══")
m_sdm = fit_spatial(CS.SDM(), W, X, y;
    method = CS.ChebyshevApprox(order=12),
    varnames = ["income", "density"])
println(m_sdm)
println()

# ─── Spatial effects (LeSage & Pace) ─────────────────────────────────

println("═══ Spatial Effects ═══")
effects = spatial_effects(m_sdm)
println(effects)
println()

# ─── Exact LU for comparison ─────────────────────────────────────────

println("═══ Exact LU (for comparison) ═══")
m_exact = fit_spatial(CS.SDM(), W, X, y;
    method = CS.ExactLU(),
    varnames = ["income", "density"])
println("Chebyshev ρ̂ = $(m_sdm.rho),  Exact LU ρ̂ = $(m_exact.rho)")
println("Difference: $(abs(m_sdm.rho - m_exact.rho))")

# ─── Using the @formula interface ─────────────────────────────────────

println("\n═══ Formula Interface ═══")
tbl = (y = y, income = X[:, 1], density = X[:, 2])
m_formula = fit_spatial(
    @formula(y ~ income + density + slag(income + density)),
    tbl, W)
println(m_formula)
