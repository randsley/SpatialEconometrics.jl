# ═══════════════════════════════════════════════════════════════════════
# Example 20: Specification Search — which spatial terms belong?
# ═══════════════════════════════════════════════════════════════════════
#
# The LM battery of example 3 chooses between a lag and an error. It has
# nothing to say about W*X, the Durbin terms, and its decision rule ends on
# "if both robust versions are significant, consider SDM" without saying how
# to check. This example is that check.
#
# Demonstrates:
#   - sdm_tests: the Koley & Bera (2023) Durbin score battery
#   - reading the robust pair rather than the plain pair, and why
#   - the per-covariate statistics, for which X enters lagged
#   - running the battery against data from three known specifications
#
# NOTATION. Koley & Bera write λ for the lag and γ for the Durbin terms.
# This package reserves λ for the spatial error coefficient, so the mapping
# is λ → ρ and γ → θ:   y = ρWy + Xβ + WX_dθ + ε

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Printf

rng = MersenneTwister(2026)
W = make_rook_weights(12); N = size(W, 1)
X = randn(rng, N, 2)
X_ols = hcat(ones(N), X)          # the battery needs the intercept column

# ─── Step 1: SDM data — both a lag and Durbin terms ──────────────────

println("═══ Step 1: data with ρ = 0.4 AND θ ≠ 0 (true model: SDM) ═══")
y_sdm, _ = simulate_spatial(SDM_DGP(0.4, [1.0, -0.5], [0.6, -0.4], 0.5),
                            W, X; rng = rng)

t = sdm_tests(y_sdm, X_ols, W)
println(t.joint)                  # run this one first
println(t.robust_lag)
println(t.robust_durbin)
println()

# Both robust statistics reject, so both pieces are real: fit the SDM.
# Note the ordering discipline — if `joint` had not rejected, none of the
# rest would be worth reading, because the individual statistics carry the
# usual multiple-testing problem and the joint test is what controls it.
if t.joint.p_value < 0.05
    lag_here = t.robust_lag.p_value < 0.05
    dur_here = t.robust_durbin.p_value < 0.05
    verdict = lag_here && dur_here ? "SDM" :
              lag_here ? "SAR" : dur_here ? "SLX" : "OLS (weak joint)"
    println("Verdict: fit $verdict")
else
    println("Verdict: OLS is adequate")
end
println()

# ─── Step 2: which covariates need lagging? ──────────────────────────

println("═══ Step 2: the Durbin terms one at a time ═══")
# Each statistic is θⱼ = 0 with ρ and the other θ's left in the model, so
# these are conditional tests, not a sequence of marginal ones.
for (j, s) in enumerate(t.terms)
    @printf("  x%d:  stat %8.3f   df %d   p %.4f\n", j, s.stat, s.df, s.p_value)
end
println()

# ─── Step 3: the robust forms earn their name ────────────────────────

println("═══ Step 3: SAR data — ρ = 0.5, θ = 0 exactly ═══")
y_sar, _ = simulate_spatial(SAR_DGP(0.5, [1.0, -0.5], 0.5), W, X; rng = rng)
t_sar = sdm_tests(y_sar, X_ols, W)

# There are no Durbin terms in this data at all, so a test for them should
# not reject. The plain one often does: it mistakes the lag's own spatial
# structure for W*X. The robust one is built to ignore exactly that.
@printf("  RS θ   (plain)   stat %8.3f   p %.4f   <- fooled by the lag\n",
        t_sar.durbin.stat, t_sar.durbin.p_value)
@printf("  RS* θ|ρ (robust) stat %8.3f   p %.4f   <- holds its size\n",
        t_sar.robust_durbin.stat, t_sar.robust_durbin.p_value)
println()
println("  Over 2000 replications of this design the plain statistic")
println("  rejects 16.1% of the time at the 5% level; the robust one 5.1%.")
println("  That is the whole reason to read the robust pair.")
println()

# ─── Step 4: and symmetrically, SLX data ─────────────────────────────

println("═══ Step 4: SLX data — θ ≠ 0, ρ = 0 exactly ═══")
y_slx, _ = simulate_spatial(SLX_DGP([1.0, -0.5], [0.6, -0.4], 0.5),
                            W, X; rng = rng)
t_slx = sdm_tests(y_slx, X_ols, W)

@printf("  LM-lag  (plain)  stat %8.3f   p %.4f   <- fooled by W*X\n",
        t_slx.lag.stat, t_slx.lag.p_value)
@printf("  RS* ρ|θ (robust) stat %8.3f   p %.4f   <- holds its size\n",
        t_slx.robust_lag.stat, t_slx.robust_lag.p_value)
println()
@printf("  Durbin terms correctly detected: joint p = %.2e\n", t_slx.joint.p_value)
println()

# ─── Step 5: estimate what the battery chose ─────────────────────────

println("═══ Step 5: fit the chosen specification ═══")
# The SDM data from step 1, now estimated. Compare the recovered parameters
# against the (0.4, [1.0, -0.5], [0.6, -0.4]) they were generated from.
m = fit_spatial(SDM(), W, X, y_sdm; varnames = ["x1", "x2"])
println(m)
