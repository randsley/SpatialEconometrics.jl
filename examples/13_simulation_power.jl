# ═══════════════════════════════════════════════════════════════════════
# Example 13: Simulation, Monte Carlo & Power Analysis
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Generating data from six spatial DGPs
#   - Monte Carlo evaluation of estimator bias and RMSE
#   - Power curves for the spatial diagnostic tests
#
# Note: simulate_spatial takes the FULL design matrix — no intercept is
# added for you, so length(beta) must equal size(X, 2).

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics


W = make_rook_weights(14)
N = size(W, 1)

# ─── The six DGPs ────────────────────────────────────────────────────

println("═══ Simulating from Six DGPs ═══")
println()

rng = MersenneTwister(808)
X = randn(rng, N, 2)
beta = [1.5, -0.8]
theta = [0.6, -0.3]

dgps = [
    ("SAR  (lag in y)",            SAR_DGP(0.6, beta, 0.5)),
    ("SEM  (lag in errors)",       SEM_DGP(0.6, beta, 0.5)),
    ("SDM  (lag in y and X)",      SDM_DGP(0.6, beta, theta, 0.5)),
    ("SLX  (lag in X only)",       SLX_DGP(beta, theta, 0.5)),
    ("SAC  (lag in y + errors)",   SAC_DGP(0.5, -0.35, beta, 0.5)),
    ("SDEM (lag in X + errors)",   SDEM_DGP(0.45, beta, theta, 0.5)),
]

for (name, dgp) in dgps
    y_sim, _ = simulate_spatial(dgp, W, X; rng = MersenneTwister(1))
    mi = moran_global(y_sim, W)
    println("  $(rpad(name, 26)) mean(y) = $(rpad(round(mean(y_sim); digits=3), 7)) ",
            "Moran's I = $(round(mi.I_stat; digits=4))  p = $(round(mi.p_value; digits=4))")
end
println()
println("Every DGP leaves a detectable spatial signature, but they are not")
println("distinguishable from Moran's I alone — that is what the LM tests and")
println("model comparison are for.")
println()

# ─── Monte Carlo: is the SAR ML estimator unbiased? ──────────────────
# The estimator argument is any function (W, X, y) -> RegressionModel.

println("═══ Monte Carlo: SAR ML Estimator ═══")
println("(100 replications, N = $N — this takes a moment)")
println()

sar_estimator = (Wm, Xm, ym) -> fit_spatial(SAR(), Wm, Xm, ym)

mc = monte_carlo_spatial(
    SAR_DGP(0.6, beta, 0.5), W, N, 2, sar_estimator;
    config = MonteCarloConfig(n_replications = 100, seed = 2024))

println(mc)
println()
println("Bias should be small relative to the standard deviation. A large bias")
println("with a small SD would signal a systematic problem, not sampling noise.")
println()

# ─── Comparing estimators on the same DGP ────────────────────────────
# Fitting the WRONG model to SAR data: SLX cannot represent the endogenous
# feedback, so its slope estimates are biased.

println("═══ Misspecification Cost: SLX fitted to SAR data ═══")
println()

slx_estimator = (Wm, Xm, ym) -> fit_slx(Wm, Xm, ym)

mc_wrong = monte_carlo_spatial(
    SAR_DGP(0.6, beta, 0.5), W, N, 2, slx_estimator;
    config = MonteCarloConfig(n_replications = 100, seed = 2024))

println(mc_wrong)
println()
println("Compare the beta rows against the correctly-specified run above: the")
println("slopes are now biased and the RMSE is several times larger, because")
println("SLX cannot represent the endogenous feedback in the SAR DGP.")
println()
println("CAVEAT: the row labels come from the DGP, not the estimator. SLX has")
println("no rho, so the row printed as 'rho' actually holds its W_x1")
println("coefficient. Only the beta rows are comparable across the two runs.")
println()

# ─── Power analysis ──────────────────────────────────────────────────
# How large a sample, and how strong a spatial parameter, before the
# diagnostics reliably detect spatial dependence?

println("═══ Power Analysis ═══")
println("(grid sizes 8, 12, 16 × rho in 0, 0.2, 0.4, 0.6 — this takes a moment)")
println()

pa = power_analysis(
    rho -> SAR_DGP(rho, beta, 0.5),
    [8, 12, 16],
    [0.0, 0.2, 0.4, 0.6];
    n_reps = 100, seed = 11)

println(pa)
println()
println("Read the rho = 0 column as the empirical SIZE of each test: it should")
println("sit near alpha = 0.05. The remaining columns are power. Power rises")
println("with both N and rho, as it must.")
