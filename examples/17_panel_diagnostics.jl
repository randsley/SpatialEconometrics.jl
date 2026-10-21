# ═══════════════════════════════════════════════════════════════════════
# Example 17: Panel Diagnostics
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Spatial Hausman test for the fixed-effects specification
#   - Joint LM tests for spatial lag, spatial error and random effects
#   - Spatial Chow test for a structural break across periods
#   - Period-by-period and pooled Moran's I on panel residuals
#
# Each test is run twice: once on data that violate the null and once on
# data that satisfy it, so both the power and the size of the test are
# visible rather than asserted.

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics
import StatsAPI: coef, coefnames, residuals

SP = SpatialEconometrics.StaticPanel

rng = MersenneTwister(31337)
L = 10; N = L * L; T_per = 6; NT = N * T_per
W = make_rook_weights(L)
S_lu = lu(sparse(1.0I, N, N) - 0.45 * W)

β_true = [1.5, -0.8]

# ─── Panel with individual fixed effects and a spatial lag ───────────

X = randn(rng, NT, 2)
μ_i = 2.0 .* randn(rng, N)          # individual effects, correlated with nothing
y = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    y[idx] = S_lu \ (μ_i .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N))
end

println("Panel: N = $N units × T = $T_per periods = $NT observations")
println("DGP:   spatial lag ρ = 0.45, individual FE, β = $β_true")
println()

# ─── Spatial Hausman test ────────────────────────────────────────────

println("═══ Spatial Hausman Test ═══")
println("  The test contrasts two coefficient vectors POSITIONALLY — it does")
println("  not match on names. Compare specifications that share a layout.")
println("  A NoFE fit carries an intercept that the FE fits do not, so FE")
println("  against NoFE would silently compare β₁ with the intercept.")
println()
for (lbl, fe) in [("IndivFE", SP.IndivFE()), ("NoFE", SP.NoFE())]
    mm = fit_spatial_panel(SP.SAR(), fe, W, X, y, N, T_per;
        varnames = ["income", "density"])
    println("    ", rpad(lbl, 10), " coefnames = ", coefnames(mm))
end
println()

# Individual against two-way effects: both expose [income, density, ρ], so
# the contrast is well posed. Here the DGP has no time effects, so the two
# should agree and the test should stay quiet.
m_fe  = fit_spatial_panel(SP.SAR(), SP.IndivFE(), W, X, y, N, T_per;
    varnames = ["income", "density"])
m_two = fit_spatial_panel(SP.SAR(), SP.TwoWayFE(), W, X, y, N, T_per;
    varnames = ["income", "density"])
ht = spatial_hausman_test(m_fe, m_two)
println("  No time effects in the DGP:")
println("    IndivFE  β̂ = ", round.(coef(m_fe)[1:2]; digits = 4),
        "   ρ̂ = ", round(m_fe.rho; digits = 4))
println("    TwoWayFE β̂ = ", round.(coef(m_two)[1:2]; digits = 4),
        "   ρ̂ = ", round(m_two.rho; digits = 4))
println("    stat = ", round(ht.stat; digits = 3), " on ", ht.df,
        " df,  p = ", round(ht.p_value; sigdigits = 4), "  → ",
        ht.p_value < 0.05 ? "reject" : "correctly quiet")
println()

# Now a DGP where the time effect is correlated with the regressors, so
# omitting it genuinely biases the individual-effects estimator.
X_t = copy(X)
y_t = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    shift = 1.5 * (t - (T_per + 1) / 2)          # period-specific mean in X
    X_t[idx, 1] .+= shift
    ξ_t = 3.0 * shift                             # ... and in the outcome
    y_t[idx] = S_lu \ (μ_i .+ ξ_t .+ X_t[idx, :] * β_true .+ 0.5 .* randn(rng, N))
end
m_fe_t  = fit_spatial_panel(SP.SAR(), SP.IndivFE(), W, X_t, y_t, N, T_per;
    varnames = ["income", "density"])
m_two_t = fit_spatial_panel(SP.SAR(), SP.TwoWayFE(), W, X_t, y_t, N, T_per;
    varnames = ["income", "density"])
ht_t = spatial_hausman_test(m_fe_t, m_two_t)
println("  Time effects correlated with X (IndivFE is misspecified):")
println("    IndivFE  β̂ = ", round.(coef(m_fe_t)[1:2]; digits = 4),
        "   ρ̂ = ", round(m_fe_t.rho; digits = 4))
println("    TwoWayFE β̂ = ", round.(coef(m_two_t)[1:2]; digits = 4),
        "   ρ̂ = ", round(m_two_t.rho; digits = 4), "   (true β = ", β_true, ")")
println("    stat = ", round(ht_t.stat; digits = 3), " on ", ht_t.df,
        " df,  p = ", round(ht_t.p_value; sigdigits = 4), "  → ",
        ht_t.p_value < 0.05 ? "reject: add the time effects" : "missed")
println()
println("  Note: this package has no spatial random-effects estimator, so the")
println("  classic FE-vs-RE Hausman comparison is not available here; the same")
println("  machinery applies to any two nested specifications.")
println()

# ─── Joint LM tests ──────────────────────────────────────────────────

println("═══ Joint LM Tests (Baltagi, Song & Koh 2003) ═══")
X_c = hcat(ones(NT), X)          # these tests want an explicit intercept

plm = panel_lm_tests(y, X_c, W, N, T_per)
println("  On the spatially-lagged panel above:")
println("    joint (χ², df = 3) = ", round(plm.joint_stat; digits = 2))
println("    lag                = ", round(plm.lag_stat; digits = 2))
println("    error              = ", round(plm.error_stat; digits = 2))
println("    random effects     = ", round(plm.re_stat; digits = 2))

# A panel with no spatial dependence at all: the statistics should drop
# to the order of their degrees of freedom.
y_iid = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    y_iid[idx] = 1.0 .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N)
end
plm0 = panel_lm_tests(y_iid, X_c, W, N, T_per)
println("  On an i.i.d. panel (no spatial structure, no FE):")
println("    joint = ", round(plm0.joint_stat; digits = 2),
        "   lag = ", round(plm0.lag_stat; digits = 2),
        "   error = ", round(plm0.error_stat; digits = 2),
        "   RE = ", round(plm0.re_stat; digits = 2))
println()
println("  A χ² statistic near its df is what 'no evidence' looks like; the")
println("  5% critical values are 7.81 (df 3) and 3.84 (df 1).")
println()

# ─── Spatial Chow test ───────────────────────────────────────────────

println("═══ Spatial Chow Test (structural break) ═══")
# Build a panel whose slopes change at t = 4.
y_brk = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    β_t = t < 4 ? β_true : [-1.0, 1.6]       # a decisive break
    y_brk[idx] = S_lu \ (μ_i .+ X[idx, :] * β_t .+ 0.5 .* randn(rng, N))
end

ct_brk = spatial_chow_test(y_brk, X_c, W, N, T_per, 4)
ct_non = spatial_chow_test(y,     X_c, W, N, T_per, 4)
println("  Panel WITH a break at t = 4:  F = ", round(ct_brk.stat; digits = 3),
        ",  p = ", round(ct_brk.p_value; sigdigits = 4),
        "  → ", ct_brk.p_value < 0.05 ? "break detected" : "missed")
println("  Panel with NO break:          F = ", round(ct_non.stat; digits = 3),
        ",  p = ", round(ct_non.p_value; sigdigits = 4),
        "  → ", ct_non.p_value < 0.05 ? "false positive" : "correctly quiet")
println()

# ─── Panel Moran's I ─────────────────────────────────────────────────

println("═══ Panel Moran's I on Residuals ═══")
# Residuals from a NON-spatial fit on spatially dependent data: Moran's I
# should flag the dependence in every period.
β_ols = X_c \ y
r_ols = y .- X_c * β_ols
pm = panel_moran(r_ols, W, N, T_per)

println("  Pooled OLS residuals, period by period:")
println(rpad("    period", 12), rpad("I", 12), rpad("z", 12), "p")
for (t, mr) in enumerate(pm.period_results)
    println(rpad("    $t", 12), rpad(round(mr.I_stat; digits = 4), 12),
            rpad(round(mr.z_stat; digits = 3), 12),
            round(mr.p_value; sigdigits = 3))
end
println("  Pooled (meta-analytic): I = ", round(pm.pooled.I_stat; digits = 4),
        ",  z = ", round(pm.pooled.z_stat; digits = 3),
        ",  p = ", round(pm.pooled.p_value; sigdigits = 4))
println()

# The same diagnostic on residuals from the correctly specified FE model.
pm_fe = panel_moran(residuals(m_fe), W, N, T_per)
println("  Spatial FE residuals, pooled: I = ",
        round(pm_fe.pooled.I_stat; digits = 4),
        ",  z = ", round(pm_fe.pooled.z_stat; digits = 3),
        ",  p = ", round(pm_fe.pooled.p_value; sigdigits = 4))
println()
println("  Fitting the spatial lag absorbs the autocorrelation the OLS")
println("  residuals carried — which is the whole point of the exercise.")
