# ═══════════════════════════════════════════════════════════════════════
# Example 5: Gravity / Flow Models (PPML & Negative Binomial)
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Poisson Pseudo-Maximum Likelihood (PPML) gravity estimation
#   - Bilateral fixed effects (origin + destination)
#   - Negative Binomial gravity with overdispersion
#   - Multilateral resistance terms
#   - Handling zero trade flows

using SpatialEconometrics
using Random

# ─── Generate synthetic trade data ───────────────────────────────────

rng = MersenneTwister(2024)
N_countries = 20
N_partners = 15
N_obs = N_countries * N_partners

origin_id = repeat(1:N_countries, inner=N_partners)
dest_id = repeat(1:N_partners, outer=N_countries)

# Explanatory variables (log distance, common language dummy)
log_distance = rand(rng, N_obs) .* 4 .+ 2     # log km ∈ [2, 6]
common_lang = Float64.(rand(rng, N_obs) .< 0.2) # 20% share language
X = hcat(log_distance, common_lang)

# True parameters: distance elasticity = -1.3, language effect = +0.6
β_true = [-1.3, 0.6]

# Origin and destination fixed effects (GDP proxies)
α_o = randn(rng, N_countries) .* 0.8
α_d = randn(rng, N_partners) .* 0.6

# Generate expected flows
η = X * β_true
for n in 1:N_obs
    η[n] += α_o[origin_id[n]] + α_d[dest_id[n]] + 3.0  # base level
end
μ_true = exp.(η)

# Poisson-distributed flows (with some zeros)
flows = Float64[max(0.0, round(μ_true[n] + sqrt(μ_true[n]) * randn(rng))) for n in 1:N_obs]
flows = max.(flows, 0.0)

n_zeros = count(flows .== 0)
println("Trade matrix: $N_countries origins × $N_partners destinations")
println("Total flows: $N_obs,  Zero flows: $n_zeros ($(round(100*n_zeros/N_obs; digits=1))%)")
println()

# ─── PPML Gravity ─────────────────────────────────────────────────────

println("═══ PPML Gravity Model — Bilateral FE ═══")
m_ppml = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id;
    fe = BilateralFE(),
    varnames = ["log_distance", "common_language"],
    maxiter = 200)
println(m_ppml)
println()

println("True vs estimated coefficients:")
println("  log_distance: true = $(β_true[1]),  est = $(round(m_ppml.coef[1]; digits=3))")
println("  common_lang:  true = $(β_true[2]),  est = $(round(m_ppml.coef[2]; digits=3))")
println()

# ─── Multilateral resistance ─────────────────────────────────────────

mr = multilateral_resistance(m_ppml)
println("═══ Multilateral Resistance Terms ═══")
println("Outward MR (top 5 origins): ", round.(sort(mr.outward; rev=true)[1:5]; digits=3))
println("Inward MR (top 5 destinations): ", round.(sort(mr.inward; rev=true)[1:5]; digits=3))
println()

# ─── Negative Binomial (with overdispersion) ─────────────────────────

# Add overdispersion to flows
flows_od = Float64[max(0.0, round(μ_true[n] + 2*sqrt(μ_true[n]) * randn(rng))) for n in 1:N_obs]
flows_od = max.(flows_od, 0.0)

println("═══ Negative Binomial Gravity Model ═══")
m_nb = fit_gravity(NegBinGravity(), flows_od, X, origin_id, dest_id;
    fe = BilateralFE(),
    varnames = ["log_distance", "common_language"])
println(m_nb)
println("Overdispersion parameter α = $(round(m_nb.alpha; digits=4))")

# ─── Origin-only FE ──────────────────────────────────────────────────

println("\n═══ PPML with Origin FE Only ═══")
m_origin = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id;
    fe = OriginFE(),
    varnames = ["log_distance", "common_language"])
println(m_origin)
