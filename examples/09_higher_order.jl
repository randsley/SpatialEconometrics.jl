# ═══════════════════════════════════════════════════════════════════════
# Example 9: Higher-Order Spatial Models — SLX, SAC, SDEM
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - SLX: spatially lagged covariates only, no endogenous lag (plain OLS)
#   - SAC/SARAR: spatial lag in y AND spatially correlated errors
#   - SDEM: Durbin covariates with a spatial error process
#   - Choosing between them when the DGP is unknown

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays
import StatsAPI: coef, loglikelihood, nobs, stderror


rng = MersenneTwister(2024)
W = make_rook_weights(18)        # 324 units
N = size(W, 1)
X = randn(rng, N, 2)

# ─── SLX: y = Xβ + WXθ + ε ────────────────────────────────────────────
# No endogenous spatial lag, so OLS is consistent and no Jacobian is needed.

println("═══ SLX Model ═══")
β_slx = [1.0, 1.5, -0.8]
θ_slx = [0.6, -0.4]
y_slx = β_slx[1] .+ X * β_slx[2:3] .+ (W * X) * θ_slx .+ 0.5 .* randn(rng, N)

m_slx = fit_slx(W, X, y_slx; varnames = ["income", "density"])
println(m_slx)
println("true β = $(β_slx), true θ = $(θ_slx)")
println()

# ─── SAC / SARAR: y = ρWy + Xβ + u,  u = λWu + ε ──────────────────────
# Two spatial parameters. Here the same W drives both processes, but the
# lag and error weights may differ.

println("═══ SAC (SARAR) Model ═══")
ρ_true, λ_true = 0.5, -0.35
β_sac = [1.0, 1.2, -0.7]
A = I - ρ_true * W
B = I - λ_true * W
u = B \ (0.5 .* randn(rng, N))
y_sac = A \ (β_sac[1] .+ X * β_sac[2:3] .+ u)

m_sac = fit_sac(W, W, X, y_sac; varnames = ["income", "density"])
println(m_sac)
println("true ρ = $(ρ_true), true λ = $(λ_true)")
println()

# ─── SDEM: y = Xβ + WXθ + u,  u = λWu + ε ─────────────────────────────
# Durbin covariates with a spatial error process. Unlike SDM there is no
# endogenous lag, so the coefficients are directly interpretable as partials.

println("═══ SDEM Model ═══")
λ_sdem = 0.45
β_sdem = [1.0, 1.4, -0.9]
θ_sdem = [0.5, -0.3]
u2 = (I - λ_sdem * W) \ (0.5 .* randn(rng, N))
y_sdem = β_sdem[1] .+ X * β_sdem[2:3] .+ (W * X) * θ_sdem .+ u2

m_sdem = fit_sdem(W, X, y_sdem; varnames = ["income", "density"])
println(m_sdem)
println("true λ = $(λ_sdem), true β = $(β_sdem), true θ = $(θ_sdem)")
println()

# ─── Model selection on the SAC data ─────────────────────────────────
# Fit several specifications to the SAC-generated y and compare. The point
# is that SLX is nested-ish but misspecified here: it cannot represent the
# endogenous feedback, so it pays for that in log-likelihood.

println("═══ Which specification? (data are SAC) ═══")
m_sar_on_sac  = fit_spatial(SAR(), W, X, y_sac; varnames = ["income", "density"])
m_slx_on_sac  = fit_slx(W, X, y_sac; varnames = ["income", "density"])
m_sdem_on_sac = fit_sdem(W, X, y_sac; varnames = ["income", "density"])

for (name, m) in [("SAR", m_sar_on_sac), ("SLX", m_slx_on_sac),
                  ("SDEM", m_sdem_on_sac), ("SAC", m_sac)]
    k = length(coef(m))
    ll = loglikelihood(m)
    println("  $(rpad(name, 5)) logLik = $(round(ll; digits=2)),  k = $k,  AIC = $(round(-2ll + 2k; digits=2))")
end
println()
println("The SAC model should win on AIC: it is the only specification that")
println("admits both an endogenous lag and a correlated error process.")
