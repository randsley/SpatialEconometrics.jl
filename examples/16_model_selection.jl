# ═══════════════════════════════════════════════════════════════════════
# Example 16: Model Selection & Comparison
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - AIC / BIC across competing spatial specifications
#   - Likelihood ratio tests for nested models (SAR inside SDM)
#   - Vuong tests for non-nested models (SAR against SEM)
#   - The common factor test: does SDM collapse to SEM?
#   - Wald tests for general linear restrictions
#   - Side-by-side comparison tables
#
# The data are generated from a KNOWN SDM, so we can check that each test
# points at the specification that actually produced the data.

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays
import StatsAPI: coef, vcov, loglikelihood, nobs, dof_residual

CS = SpatialEconometrics.CrossSection

rng = MersenneTwister(4242)
L = 14; N = L * L
W = make_rook_weights(L)
IN = sparse(1.0I, N, N)

ρ_true = 0.5
β_true = [1.5, -0.8]
θ_true = [0.6, -0.3]        # the spatial lags of X — this is what makes it SDM
X = randn(rng, N, 2)
y = (IN - ρ_true * W) \ (1.0 .+ X * β_true .+ (W * X) * θ_true .+ 0.5 * randn(rng, N))

vn = ["income", "density"]
m_slx  = fit_slx(W, X, y; varnames = vn)
m_sar  = fit_spatial(CS.SAR(), W, X, y; varnames = vn)
m_sem  = fit_spatial_error(W, X, y; varnames = vn)
m_sdm  = fit_spatial(CS.SDM(), W, X, y; varnames = vn)
m_sdem = fit_sdem(W, X, y; varnames = vn)

println("True DGP: SDM with ρ = $ρ_true, β = $β_true, θ = $θ_true")
println()

# ─── Information criteria ────────────────────────────────────────────

println("═══ Information Criteria ═══")
println(rpad("  Model", 10), rpad("log-lik", 12), rpad("k", 5),
        rpad("AIC", 12), "BIC")
models = [("SLX", m_slx), ("SAR", m_sar), ("SEM", m_sem),
          ("SDM", m_sdm), ("SDEM", m_sdem)]
for (lbl, m) in models
    k = nobs(m) - dof_residual(m)
    println(rpad("  $lbl", 10),
            rpad(round(loglikelihood(m); digits = 2), 12),
            rpad(Int(round(k)), 5),
            rpad(round(spatial_aic(m); digits = 2), 12),
            round(spatial_bic(m); digits = 2))
end
best_aic = models[argmin([spatial_aic(m) for (_, m) in models])][1]
best_bic = models[argmin([spatial_bic(m) for (_, m) in models])][1]
println()
println("  Lowest AIC: $best_aic      Lowest BIC: $best_bic")
println("  BIC penalises parameters harder, so the two can disagree when the")
println("  extra terms buy little fit. Here they should both pick the SDM.")
println()

# ─── Likelihood ratio: nested models ─────────────────────────────────

println("═══ Likelihood Ratio Test (nested) ═══")
# SAR is SDM with θ = 0, so the restriction is testable by LR.
lr = lr_test(m_sar, m_sdm)
println("  H₀: θ = 0  (SAR)   vs   H₁: SDM")
println("    LR = ", round(lr.stat; digits = 3), " on ", lr.df, " df,  p = ",
        round(lr.p_value; sigdigits = 4))
println("    → ", lr.p_value < 0.05 ? "reject SAR: the WX terms belong" :
                                      "no evidence against SAR")
println()

# The same machinery against a model that is NOT restricted enough would
# error rather than silently return nonsense:
println("  lr_test insists the unrestricted model has more parameters:")
try
    lr_test(m_sdm, m_sar)
catch e
    println("    ", sprint(showerror, e))
end
println()

# ─── Vuong: non-nested models ────────────────────────────────────────

println("═══ Vuong Test (non-nested) ═══")
# SAR and SEM are not nested in one another — neither is a restriction of
# the other — so an LR test does not apply.
vt = vuong_test(m_sar, m_sem)
println("  SAR vs SEM:  statistic = ", round(vt.stat; digits = 4))
println("    preference = ", vt.preference)
println("    (positive favours the first model; the statistic is standard")
println("     normal under the null that the two fit equally well)")
println()

# ─── Common factor test ──────────────────────────────────────────────

println("═══ Common Factor Test ═══")
# The SDM collapses to the SEM exactly when θ = -ρβ. Rejecting means the
# Durbin terms carry information the error structure cannot absorb.
β_sdm = coef(m_sdm)[2:3]       # slopes, skipping the intercept
θ_sdm = coef(m_sdm)[4:5]       # the W_x terms
cft = common_factor_test(β_sdm, θ_sdm, m_sdm.rho, vcov(m_sdm))
println("  H₀: θ + ρβ = 0  (SDM reduces to SEM)")
println("    θ̂      = ", round.(θ_sdm; digits = 4))
println("    -ρ̂β̂    = ", round.(-m_sdm.rho .* β_sdm; digits = 4))
println("    stat = ", round(cft.stat; digits = 3), " on ", cft.df,
        " df,  p = ", round(cft.p_value; sigdigits = 4))
println("    → ", cft.reject ? "reject: keep the SDM" : "no evidence against the SEM")
println()

# The converse, as a check on the test's size: generate from a model where
# the restriction genuinely holds and confirm it is not rejected.
λ_sem = 0.5
y_sem = 1.0 .+ X * β_true .+ ((IN - λ_sem * W) \ (0.5 .* randn(rng, N)))
m_sdm2 = fit_spatial(CS.SDM(), W, X, y_sem; varnames = vn)
cft2 = common_factor_test(coef(m_sdm2)[2:3], coef(m_sdm2)[4:5],
                          m_sdm2.rho, vcov(m_sdm2))
println("  On SEM-generated data (restriction true): p = ",
        round(cft2.p_value; sigdigits = 4), " → ",
        cft2.reject ? "reject (a type-I error)" : "correctly fails to reject")
println()

# ─── Wald tests ──────────────────────────────────────────────────────

println("═══ Wald Tests ═══")
k = length(coef(m_sdm))

# H₀: β_income = 0 — a single restriction, expected to be rejected hard.
R1 = zeros(1, k); R1[1, 2] = 1.0
w1 = wald_test(m_sdm, R1, [0.0])
println("  H₀: β_income = 0            stat = ", round(w1.stat; digits = 2),
        " on ", w1.df, " df,  p = ", round(w1.p_value; sigdigits = 4))

# H₀: both Durbin terms are zero — the same restriction the LR test above
# examined, now by a Wald statistic on the unrestricted fit alone.
R2 = zeros(2, k); R2[1, 4] = 1.0; R2[2, 5] = 1.0
w2 = wald_test(m_sdm, R2, [0.0, 0.0])
println("  H₀: θ_income = θ_density = 0  stat = ", round(w2.stat; digits = 2),
        " on ", w2.df, " df,  p = ", round(w2.p_value; sigdigits = 4))
println("    LR said ", round(lr.stat; digits = 2), " on ", lr.df,
        " df — Wald and LR are asymptotically equivalent and should agree")
println("    in order of magnitude; they differ in finite samples.")
println()

# ─── Comparison table ────────────────────────────────────────────────

println("═══ Model Comparison Table ═══")
println(model_comparison_table(m_slx, m_sar, m_sem, m_sdm, m_sdem;
    names = ["SLX", "SAR", "SEM", "SDM", "SDEM"]))
