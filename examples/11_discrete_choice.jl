# ═══════════════════════════════════════════════════════════════════════
# Example 11: Spatial Discrete Choice
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Linearized GMM spatial probit (Klier & McMillen) — fast, no MCMC
#   - Bayesian spatial probit via Albert-Chib Gibbs sampling
#   - Spatial ordered probit
#   - Marginal effects decomposed into direct / indirect / total
#
# All three target the latent model  y* = ρWy* + Xβ + ε,  with σ² = 1
# fixed for identification, and  y = 1(y* > 0).

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics

rng = MersenneTwister(314)
W = make_rook_weights(14)        # 196 units
N = size(W, 1)
X = randn(rng, N, 2)

ρ_true = 0.4
β_true = [0.5, -0.7]             # no intercept: the latent scale is fixed

# DGP: draw the latent vector, then threshold it.
y_star = (I - ρ_true * W) \ (X * β_true .+ randn(rng, N))
y = Float64.(y_star .> 0)

println("Binary outcome: $(Int(sum(y))) ones out of $N ($(round(100*mean(y); digits=1))%)")
println()

# ─── Linearized GMM ──────────────────────────────────────────────────
# A standard probit supplies the ρ = 0 starting value; the moment conditions
# are linearized around it and estimated by 2SLS with [X, WX, W²X]
# instruments. Fast, but linearization costs precision on ρ.

println("═══ Linearized GMM Probit (Klier & McMillen) ═══")
m_gmm = fit_spatial_probit(W, X, y; varnames = ["x1", "x2"])
println(m_gmm)
println("true ρ = $(ρ_true), true β = $(β_true)")
println()
println("NOTE: at N=$N this estimator's ρ̂ has a standard deviation of roughly")
println("0.25, so a single small-sample estimate is a coarse reading. It is")
println("approximately unbiased — do not over-interpret one draw.")
println()

# ─── Bayesian (Albert-Chib Gibbs) ────────────────────────────────────
# Each sweep draws the latent y* one site at a time from its truncated normal
# conditional, draws β from its conjugate normal conditional, then updates ρ
# by random-walk Metropolis.

println("═══ Bayesian Spatial Probit (Albert-Chib) ═══")
m_bayes = fit_bayesian_spatial_probit(W, X, y;
    n_samples = 1500, burnin = 500,
    varnames = ["x1", "x2"], rng = MersenneTwister(99))
println(m_bayes)
println("true ρ = $(ρ_true), true β = $(β_true)")
println()
println("The Gibbs sampler is roughly 2.5× more precise on ρ than the GMM")
println("estimator on the same data — the payoff for not linearizing.")
println("Watch the Metropolis acceptance rate: aim for 0.2-0.5. Tune it with")
println("the rho_proposal_sd keyword (default 0.2).")
println()

# ─── Marginal effects ────────────────────────────────────────────────
# For a spatial probit the derivative of P(y=1) with respect to x runs
# through (I - ρW)⁻¹, so a change at one site propagates to every other.

println("═══ Marginal Effects ═══")
eff = marginal_effects(m_gmm)
println(eff)
println("Direct   = own-site effect (average diagonal)")
println("Indirect = spillover onto other sites (average off-diagonal row sum)")
println("Total    = direct + indirect")
println()

# ─── Ordered outcomes ────────────────────────────────────────────────
# Same Gibbs machinery plus uniform cutpoint draws. τ₁ = 0 and no intercept
# are imposed for identification.

println("═══ Spatial Ordered Probit ═══")
τ_true = [0.0, 1.2]             # 3 categories
y_ord = map(y_star) do v
    v <= τ_true[1] ? 1 : (v <= τ_true[2] ? 2 : 3)
end
println("Category counts: ", [count(==(c), y_ord) for c in 1:3])

m_ord = fit_spatial_ordered_probit(W, X, y_ord;
    n_categories = 3, n_samples = 1200, burnin = 400,
    varnames = ["x1", "x2"], rng = MersenneTwister(7))
println(m_ord)
println("true cutpoints = $(τ_true) (τ₁ fixed at 0)")
