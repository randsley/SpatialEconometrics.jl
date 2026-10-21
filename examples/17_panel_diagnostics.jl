# ═══════════════════════════════════════════════════════════════════════
# Example 17: Panel Diagnostics
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Spatial Hausman test: fixed effects against pooled
#   - Joint LM tests for spatial lag, spatial error and random effects
#   - Spatial Chow test for a structural break across periods
#   - Period-by-period and pooled Moran's I on panel residuals
#
# Every test is run twice: once on data that violate the null and once on
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
X = randn(rng, NT, 2)

# Build a panel from a spatial-lag DGP with the given individual effects.
function make_panel(μ)
    out = zeros(NT)
    for t in 1:T_per
        idx = ((t - 1) * N + 1):(t * N)
        out[idx] = S_lu \ (μ .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N))
    end
    return out
end

y      = make_panel(2.0 .* randn(rng, N))   # genuine individual effects
y_nofe = make_panel(zeros(N))               # same DGP, no individual effects

println("Panel: N = $N units × T = $T_per periods = $NT observations")
println("DGP:   spatial lag ρ = 0.45, β = $β_true")
println()

fit_panel(fe, yy) = fit_spatial_panel(SP.SAR(), fe, W, X, yy, N, T_per;
    varnames = ["income", "density"])

# Pick coefficients out by NAME, never by position: the two specifications
# do not carry the same terms.
function slopes(m)
    nm = String.(coefnames(m))
    return [coef(m)[findfirst(==(n), nm)] for n in ("income", "density")]
end

# ─── Spatial Hausman test ────────────────────────────────────────────

println("═══ Spatial Hausman Test ═══")

m_fe   = fit_panel(SP.IndivFE(), y)
m_pool = fit_panel(SP.NoFE(), y)

println("  The test matches coefficients BY NAME, and the two specifications")
println("  do not carry the same terms:")
println("    IndivFE: ", coefnames(m_fe))
println("    NoFE:    ", coefnames(m_pool))
println("  The pooled intercept has no fixed-effects counterpart, so it is")
println("  left out. Matching positionally would contrast β_income with the")
println("  intercept and manufacture a huge statistic from unrelated terms.")
println()

ht = spatial_hausman_test(m_fe, m_pool)
println("  Data WITH individual effects:")
println("    FE     β̂ = ", round.(slopes(m_fe); digits = 4),
        "   ρ̂ = ", round(m_fe.rho; digits = 4))
println("    Pooled β̂ = ", round.(slopes(m_pool); digits = 4),
        "   ρ̂ = ", round(m_pool.rho; digits = 4))
println("    compared: ", ht.coefnames)
println("    stat = ", round(ht.stat; digits = 2), " on ", ht.df,
        " df,  p = ", round(ht.p_value; sigdigits = 4))
println("    → ", ht.p_value < 0.05 ? "reject: keep the fixed effects" :
                                      "no divergence between the two")
println()
println("    Omitting the individual effects is not merely inefficient here:")
println("    they reach the spatial lag through (I - ρW)⁻¹, which makes Wy")
println("    endogenous and the pooled estimator inconsistent.")
println()

# Size: the same DGP with no individual effects at all. Pooled is now
# correctly specified, FE is merely wasteful, and the test should be quiet.
ht0 = spatial_hausman_test(fit_panel(SP.IndivFE(), y_nofe),
                           fit_panel(SP.NoFE(), y_nofe))
println("  Data with NO individual effects:")
println("    stat = ", round(ht0.stat; digits = 3), " on ", ht0.df,
        " df,  p = ", round(ht0.p_value; sigdigits = 4))
println("    → ", ht0.p_value < 0.05 ? "false positive" :
                                       "correctly quiet: FE are not needed")
println()

# `names` restricts the comparison — here to the slopes, excluding ρ.
ht_sl = spatial_hausman_test(m_fe, m_pool; names = ["income", "density"])
println("  Restricted to the slopes: compared ", ht_sl.coefnames,
        ", stat = ", round(ht_sl.stat; digits = 2), " on ", ht_sl.df, " df")
println()
println("  Note: this package has no spatial random-effects estimator, so the")
println("  textbook FE-vs-RE contrast is not available; the same machinery")
println("  applies to any two specifications that share coefficient names.")
println()

# ─── Joint LM tests ──────────────────────────────────────────────────

println("═══ Joint LM Tests (Baltagi, Song & Koh 2003) ═══")
X_c = hcat(ones(NT), X)          # these tests want an explicit intercept

plm = panel_lm_tests(y, X_c, W, N, T_per)
println("  On the spatially-lagged panel with individual effects:")
println("    joint (χ², df = 3) = ", round(plm.joint_stat; digits = 2))
println("    lag                = ", round(plm.lag_stat; digits = 2))
println("    error              = ", round(plm.error_stat; digits = 2))
println("    random effects     = ", round(plm.re_stat; digits = 2))

# A panel with no spatial dependence at all.
y_iid = zeros(NT)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    y_iid[idx] = 1.0 .+ X[idx, :] * β_true .+ 0.5 .* randn(rng, N)
end
plm0 = panel_lm_tests(y_iid, X_c, W, N, T_per)
println("  On an i.i.d. panel (no spatial structure, no individual effects):")
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
# A panel whose slopes change at t = 4.
y_brk = zeros(NT)
μ_brk = 2.0 .* randn(rng, N)
for t in 1:T_per
    idx = ((t - 1) * N + 1):(t * N)
    β_t = t < 4 ? β_true : [-1.0, 1.6]       # a decisive break
    y_brk[idx] = S_lu \ (μ_brk .+ X[idx, :] * β_t .+ 0.5 .* randn(rng, N))
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

# The same diagnostic on residuals from the spatial FE model.
pm_fe = panel_moran(residuals(m_fe), W, N, T_per)
println("  Spatial FE residuals, pooled: I = ",
        round(pm_fe.pooled.I_stat; digits = 4),
        ",  z = ", round(pm_fe.pooled.z_stat; digits = 3),
        ",  p = ", round(pm_fe.pooled.p_value; sigdigits = 4))
println()
println("  Fitting the spatial lag absorbs the autocorrelation the OLS")
println("  residuals carried — which is the whole point of the exercise.")
