# ═══════════════════════════════════════════════════════════════════════
# Example 10: Spatial Regime Models
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Regime-specific coefficients with a shared spatial process
#   - SKATER spatially constrained clustering to discover regimes
#   - Chow test for parameter stability across regimes

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics
import StatsAPI: coef, loglikelihood, nobs, stderror


rng = MersenneTwister(77)
L = 16
W = make_rook_weights(L)
N = size(W, 1)
X = randn(rng, N, 2)

# ─── A north/south structural break ──────────────────────────────────
# Units are laid out row-major on an L×L grid, so the first half of the
# indices is the "south" half of the map.

labels_true = [i <= N ÷ 2 ? 1 : 2 for i in 1:N]

β_south = [1.0, 2.0, -0.5]
β_north = [1.0, 0.3, -1.6]      # same intercept, very different slopes
ρ_true = 0.45

η = similar(X[:, 1])
for i in 1:N
    b = labels_true[i] == 1 ? β_south : β_north
    η[i] = b[1] + X[i, 1] * b[2] + X[i, 2] * b[3]
end
y = (I - ρ_true * W) \ (η .+ 0.4 .* randn(rng, N))

# ─── Pooled model ignores the break ──────────────────────────────────

println("═══ Pooled SAR (ignores the regime break) ═══")
m_pooled = fit_spatial(SAR(), W, X, y; varnames = ["x1", "x2"])
println(m_pooled)
println("The pooled slopes land between the two regimes and fit neither.")
println()

# ─── Regime model with the TRUE regimes ──────────────────────────────

println("═══ Regime SAR (true regimes supplied) ═══")
regimes = ExogenousRegimes(labels_true)
m_regime = fit_spatial_regime(W, X, y, regimes;
    spatial_lag = true, varnames = ["x1", "x2"])
println(m_regime)
println("true south β = $(β_south)")
println("true north β = $(β_north)")
println()

# ─── Chow test: are the regimes really different? ────────────────────

println("═══ Chow Test for Parameter Stability ═══")
chow = regime_chow_test(m_pooled, m_regime)
println(chow)
println("A small p-value rejects coefficient stability, i.e. the regimes matter.")
println()

# ─── Discovering regimes with SKATER ─────────────────────────────────
# When the break is unknown, cluster on the covariates subject to spatial
# contiguity. SKATER partitions a minimum spanning tree of W, so every
# regime is guaranteed to be a connected region of the map.

println("═══ SKATER Regionalization ═══")
# Cluster on a variable that actually differs by region, otherwise there is
# no signal to find.
region_signal = hcat(Float64.(labels_true) .+ 0.3 .* randn(rng, N), X[:, 1])
# min_size guarantees every region is large enough to identify its own
# coefficients. Without it a cut can peel off a handful of units and leave the
# regime regression rank-deficient.
discovered = skater_regimes(W, region_signal, 2; min_size = 20)

agree = mean(discovered.labels .== labels_true)
# Cluster labels are arbitrary, so 0% agreement is a perfect flipped match.
println("Agreement with the true split: $(round(100 * max(agree, 1 - agree); digits=1))%")
println("Regime sizes: ", [count(==(r), discovered.labels) for r in 1:2])
println()
println("SKATER cuts the spanning-tree edge that most reduces within-group")
println("sum of squared deviations. Cutting the LONGEST edge instead would")
println("usually peel off a single outlying unit, since the longest edge")
println("attaches a leaf — a degenerate partition that cannot be regressed on.")

m_discovered = fit_spatial_regime(W, X, y, discovered;
    spatial_lag = true, varnames = ["x1", "x2"])
println()
println("logLik with true regimes       = $(round(loglikelihood(m_regime); digits=2))")
println("logLik with discovered regimes = $(round(loglikelihood(m_discovered); digits=2))")
println("logLik pooled                  = $(round(loglikelihood(m_pooled); digits=2))")
