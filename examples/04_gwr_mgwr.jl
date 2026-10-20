# ═══════════════════════════════════════════════════════════════════════
# Example 4: Geographically Weighted Regression (GWR & MGWR)
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - GWR with adaptive bandwidth (AICc selection)
#   - GWR with fixed bandwidth
#   - Multiscale GWR (MGWR) with variable-specific bandwidths
#   - Comparing local vs global coefficient estimates
#   - Different kernel functions

using SpatialEconometrics
using Random, Statistics

# ─── Generate spatially varying coefficients ─────────────────────────

rng = MersenneTwister(77)
N = 300
coords = hcat(rand(rng, N) .* 10, rand(rng, N) .* 10)
X = randn(rng, N, 2)

# β₁ varies with x-coordinate (spatial non-stationarity)
β1_true = 1.0 .+ 0.4 .* coords[:, 1]
# β₂ is constant (globally stationary)
β2_true = fill(-0.5, N)
# Intercept varies with y-coordinate
β0_true = 2.0 .+ 0.2 .* coords[:, 2]

y = β0_true .+ β1_true .* X[:, 1] .+ β2_true .* X[:, 2] .+ 0.5 .* randn(rng, N)

# ─── GWR with adaptive bandwidth (auto-selected) ─────────────────────

println("═══ GWR — Adaptive Bandwidth (AICc-optimized) ═══")
m_adapt = fit_gwr(X, y, coords;
    kernel = BisquareKernel(),
    varnames = ["temperature", "rainfall"])
println(m_adapt)
println()

# ─── GWR with fixed bandwidth ────────────────────────────────────────

println("═══ GWR — Fixed Bandwidth (h=3.0) ═══")
m_fixed = fit_gwr(X, y, coords;
    bandwidth = FixedBandwidth(3.0),
    kernel = GaussianKernel(),
    varnames = ["temperature", "rainfall"])
println(m_fixed)
println()

# ─── Verify spatial variation ─────────────────────────────────────────

println("═══ Coefficient Variation Analysis ═══")
println("Variable 'temperature' — true β varies 1.0 to 5.0 across space:")
println("  Estimated range: [$(round(minimum(m_adapt.coef_local[:, 2]); digits=2)), $(round(maximum(m_adapt.coef_local[:, 2]); digits=2))]")
println("  SD of local coefficients: $(round(std(m_adapt.coef_local[:, 2]); digits=3))")
println()
println("Variable 'rainfall' — true β is constant at -0.5:")
println("  Estimated range: [$(round(minimum(m_adapt.coef_local[:, 3]); digits=2)), $(round(maximum(m_adapt.coef_local[:, 3]); digits=2))]")
println("  SD of local coefficients: $(round(std(m_adapt.coef_local[:, 3]); digits=3))")
println()

# ─── Correlation between true and estimated local β₁ ─────────────────

using Statistics
r = cor(β1_true, m_adapt.coef_local[:, 2])
println("Correlation between true and estimated local β₁: $(round(r; digits=3))")
println()

# ─── MGWR — variable-specific bandwidths ─────────────────────────────

println("═══ Multiscale GWR (MGWR) ═══")
m_mgwr = fit_mgwr(X, y, coords;
    kernel = BisquareKernel(),
    varnames = ["temperature", "rainfall"])
println(m_mgwr)
println()

println("Bandwidth comparison:")
println("  (Intercept): k = $(round(Int, m_mgwr.bandwidths[1]))")
println("  temperature: k = $(round(Int, m_mgwr.bandwidths[2]))  (should be smaller — varies locally)")
println("  rainfall:    k = $(round(Int, m_mgwr.bandwidths[3]))  (should be larger — constant globally)")
