# ═══════════════════════════════════════════════════════════════════════
# Example 7: GRAS & Cross-Entropy SAM/MRIO Calibration
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - GRAS balancing of a Social Accounting Matrix (SAM)
#   - Handling negative entries (subsidies, transfers)
#   - ForwardDiff-compatible differentiable balancing
#   - Leontief SAM multipliers
#   - Generalized cross-entropy with side constraints

using SpatialEconometrics
using Random, LinearAlgebra
using ForwardDiff

# ─── Create a synthetic SAM ──────────────────────────────────────────

rng = MersenneTwister(42)
n_sectors = 6

# Prior SAM (with some negative entries representing subsidies)
X0 = [
    0.0   15.0   8.0   0.0   -2.0   5.0;
    10.0   0.0  12.0   6.0    0.0   3.0;
    5.0    8.0   0.0  10.0    4.0   0.0;
    3.0    4.0   6.0   0.0    8.0  -1.0;
    0.0    2.0   3.0   5.0    0.0   7.0;
    8.0    0.0   4.0   2.0    6.0   0.0;
]

println("═══ Original SAM ═══")
println("Row sums: ", round.(vec(sum(X0; dims=2)); digits=1))
println("Col sums: ", round.(vec(sum(X0; dims=1)); digits=1))
println("Has negative entries: ", any(X0 .< 0))
println()

# New target margins (e.g., updated national accounts)
u_target = vec(sum(X0; dims=2)) .* [1.05, 0.98, 1.10, 1.02, 0.95, 1.08]
# Adjust column targets so totals match
v_raw = vec(sum(X0; dims=1)) .* [1.03, 1.01, 1.07, 0.99, 1.04, 1.02]
v_target = v_raw .+ (sum(u_target) - sum(v_raw)) / n_sectors

# ─── GRAS balancing ──────────────────────────────────────────────────

println("═══ GRAS Balancing ═══")
res = balance_gras_full(X0, u_target, v_target)
println(res)

println("Balanced row sums match target: ", isapprox(vec(sum(res.X_balanced; dims=2)), u_target; atol=1e-8))
println("Balanced col sums match target: ", isapprox(vec(sum(res.X_balanced; dims=1)), v_target; atol=1e-8))
println()

println("Balanced SAM (rounded):")
for i in 1:n_sectors
    println("  ", [round(res.X_balanced[i, j]; digits=2) for j in 1:n_sectors])
end
println()

# ─── Simple differentiable version ────────────────────────────────────

println("═══ Differentiable GRAS (for use with ForwardDiff) ═══")
X_bal = balance_gras(X0, u_target, v_target)
println("Same result: ", isapprox(X_bal, res.X_balanced; atol=1e-10))

# Differentiate through the balancing w.r.t. a target margin change
function total_output(u_perturbed)
    X_b = balance_gras(X0, u_perturbed, v_target)
    return sum(X_b)
end
grad = ForwardDiff.gradient(total_output, u_target)
println("∂(total_output)/∂u = ", round.(grad; digits=4))
println()

# ─── Leontief SAM multipliers ────────────────────────────────────────

println("═══ Leontief SAM Multipliers ═══")
endogenous_idx = [1, 2, 3, 4]  # first 4 sectors are endogenous
M_a, A_n = sam_leontief_multipliers(res.X_balanced, endogenous_idx)

println("Technical coefficients (A):")
for i in 1:4
    println("  ", [round(A_n[i, j]; digits=3) for j in 1:4])
end
println()

println("Multiplier matrix (M = (I-A)⁻¹):")
for i in 1:4
    println("  ", [round(M_a[i, j]; digits=3) for j in 1:4])
end
println()
println("Column sums (output multipliers): ", round.(vec(sum(M_a; dims=1)); digits=3))
