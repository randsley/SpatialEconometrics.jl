# ═══════════════════════════════════════════════════════════════════════
# Example 19: Bayesian Spatial Models (Turing extension)
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - bayesian_sar / bayesian_sem / bayesian_sdm by NUTS
#   - Checking a posterior against the MLE on the same data
#   - Reading the posterior for a quantity the MLE reports as a point
#   - Why the Chebyshev log-determinant is what makes this affordable
#   - Reproducibility through the `rng` keyword
#
# Turing and Distributions are WEAK dependencies: the three functions only
# acquire methods once both are in the session. To run this example:
#
#     julia --project=. -e 'using Pkg; Pkg.add(["Turing", "Distributions"])'
#     julia --project=. examples/19_bayesian.jl
#
# Sampling takes a couple of minutes.

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics
import StatsAPI: coef, coefnames, stderror

CS = SpatialEconometrics.CrossSection

# ─── Before Turing is loaded ─────────────────────────────────────────

println("═══ The Extension Is Not Loaded Yet ═══")
try
    bayesian_sar(nothing)
catch e
    println("  ", sprint(showerror, e))
end
println()

using Turing, Distributions

println("═══ Extension Loaded ═══")
println("  Turing version: ", pkgversion(Turing))
println("  extension loaded: ", !isnothing(Base.get_extension(
            SpatialEconometrics, :SpatialEconometricsTuringExt)))
println()

# ─── Data from a known SAR ───────────────────────────────────────────

rng = MersenneTwister(2024)
L = 12; N = L * L
W = make_rook_weights(L)
IN = sparse(1.0I, N, N)

β_true = [1.0, 1.5, -0.8]        # intercept, income, density
ρ_true = 0.5
σ_true = 0.5

X = randn(rng, N, 2)
Xa = hcat(ones(N), X)
y = (IN - ρ_true * W) \ (Xa * β_true .+ σ_true .* randn(rng, N))

println("DGP: SAR on a $L×$L rook grid, N = $N")
println("     β = $β_true,  ρ = $ρ_true,  σ = $σ_true")
println()

# ─── Why this is affordable ──────────────────────────────────────────

println("═══ The Log-Determinant Term ═══")
# Every evaluation of the SAR likelihood needs log|I - ρW|. NUTS asks for
# thousands of them, each at a new ρ, so a dense factorization per draw is
# out of the question. The Chebyshev approximation precomputes traces once
# and then costs a polynomial evaluation per draw.
τ = CS.precompute_chebyshev_traces(W, CS.ChebyshevApprox(order = 12))
println(rpad("  ρ", 8), rpad("exact", 14), rpad("Chebyshev", 14), "abs. error")
for ρ in (-0.8, -0.4, 0.0, 0.4, 0.6, 0.8, 0.95)
    ex = logdet(Matrix(IN - ρ * W))
    ap = CS.chebyshev_logdet(ρ, τ)
    println(rpad("  $ρ", 8), rpad(round(ex; digits = 5), 14),
            rpad(round(ap; digits = 5), 14), round(abs(ex - ap); sigdigits = 3))
end
println()
println("  The approximation is exact to several decimals over the whole")
println("  parameter space, so it is the likelihood, not a stand-in for it.")
println()

# ─── Bayesian SAR ────────────────────────────────────────────────────

println("═══ Bayesian SAR ═══")
println("  Priors: β ~ N(0, 10I),  ρ ~ U(-0.99, 0.99),  σ ~ truncated N(0, 5)")
println("  Sampling (NUTS, 2 chains × 1000 draws)...")
r = bayesian_sar(W, X, y; n_samples = 1000, n_chains = 2,
                 varnames = ["income", "density"], rng = MersenneTwister(11))
println()
println(r)

# The posterior and the MLE are answering the same question on the same
# data, with priors this diffuse. If they disagreed, one of them would be
# wrong — which makes this the single most useful check to run.
m = fit_spatial(CS.SAR(), W, X, y; varnames = ["income", "density"])
println("═══ Posterior Mean vs. MLE ═══")
println(rpad("  parameter", 15), rpad("post. mean", 13), rpad("post. SD", 11),
        rpad("MLE", 13), "MLE SE")
mle_names = String.(coefnames(m))
mle_se = stderror(m)
for (i, nm) in enumerate(r.coefnames)
    j = findfirst(==(nm), mle_names)
    mle_val = isnothing(j) ? "—" : string(round(coef(m)[j]; digits = 4))
    mle_err = isnothing(j) ? "—" : string(round(mle_se[j]; digits = 4))
    println(rpad("  $nm", 15),
            rpad(round(r.posterior_mean[i]; digits = 4), 13),
            rpad(round(r.posterior_std[i]; digits = 4), 11),
            rpad(mle_val, 13), mle_err)
end
println()
println("  σ has no MLE row: `fit_spatial` reports it separately, not as a")
println("  coefficient. Posterior σ = ", round(r.posterior_mean[5]; digits = 4),
        " against a true ", σ_true, ".")
println()

# ─── What the posterior buys you ─────────────────────────────────────

println("═══ Reading the Posterior Directly ═══")
# The chain is a full joint posterior, so interval statements need no normal
# approximation and no delta method. Pull the ρ draws out and ask directly.
# Chain indexing differs between Turing's two chain back ends; go through
# the summary where you can, and the chain only when you must.
ρ_draws = vec(collect(r.chain[:ρ]))
println("  ρ draws: ", length(ρ_draws))
qs = sort(ρ_draws)
q(p) = qs[clamp(round(Int, p * length(qs)), 1, length(qs))]
println("  posterior mean      = ", round(mean(ρ_draws); digits = 4))
println("  95% credible region = [", round(q(0.025); digits = 4), ", ",
        round(q(0.975); digits = 4), "]")
println("  P(ρ > 0.4)          = ", round(mean(ρ_draws .> 0.4); digits = 4))
println("  P(ρ > 0.6)          = ", round(mean(ρ_draws .> 0.6); digits = 4))
println()
println("  That last line is the kind of statement a point estimate and a")
println("  standard error cannot make without assuming normality.")
println()

# ─── Bayesian SEM ────────────────────────────────────────────────────

println("═══ Bayesian SEM ═══")
λ_true = 0.6
y_sem = Xa * β_true .+ ((IN - λ_true * W) \ (σ_true .* randn(rng, N)))
println("  Dependence is in the errors now: λ = $λ_true. Sampling...")
r_sem = bayesian_sem(W, X, y_sem; n_samples = 1000, n_chains = 2,
                     varnames = ["income", "density"], rng = MersenneTwister(11))
println()
println(r_sem)
println("  λ̂ = ", round(r_sem.posterior_mean[4]; digits = 4),
        " (true ", λ_true, ")")
println()

# ─── Bayesian SDM ────────────────────────────────────────────────────

println("═══ Bayesian SDM ═══")
θ_true = [0.6, -0.3]
y_sdm = (IN - ρ_true * W) \
        (Xa * β_true .+ (W * X) * θ_true .+ σ_true .* randn(rng, N))
println("  Durbin terms as well: θ = $θ_true. Sampling...")
r_sdm = bayesian_sdm(W, X, y_sdm; n_samples = 1000, n_chains = 2,
                     varnames = ["income", "density"], rng = MersenneTwister(11))
println()
println(r_sdm)
println("  The design is augmented with WX, so the W_ rows are the θ terms:")
println("    W_income  = ", round(r_sdm.posterior_mean[4]; digits = 4),
        "  (true ", θ_true[1], ")")
println("    W_density = ", round(r_sdm.posterior_mean[5]; digits = 4),
        "  (true ", θ_true[2], ")")
println()

# ─── Reproducibility ─────────────────────────────────────────────────

println("═══ Reproducibility ═══")
a = bayesian_sar(W, X, y; n_samples = 200, n_chains = 1, rng = MersenneTwister(99))
b = bayesian_sar(W, X, y; n_samples = 200, n_chains = 1, rng = MersenneTwister(99))
c = bayesian_sar(W, X, y; n_samples = 200, n_chains = 1, rng = MersenneTwister(100))
println("  same seed  -> identical posterior means: ",
        a.posterior_mean == b.posterior_mean)
println("  other seed -> identical posterior means: ",
        a.posterior_mean == c.posterior_mean)
println("  max difference across seeds: ",
        round(maximum(abs.(a.posterior_mean .- c.posterior_mean)); digits = 4),
        " — Monte Carlo error, not disagreement")
println()

# ─── Caveats ─────────────────────────────────────────────────────────

println("═══ Worth Knowing ═══")
println("  - `result.chain` is the raw chain object. Turing returns an")
println("    MCMCChains.Chains before v0.45 and a FlexiChains.FlexiChain")
println("    from v0.45 on; they index differently for vector-valued")
println("    parameters. `posterior_mean`/`posterior_std` are stable across")
println("    both, so prefer them unless you need the draws themselves.")
println("  - The priors are fixed. They are deliberately diffuse, which is")
println("    why the posterior lands on the MLE; they are not meant as a")
println("    considered prior for any particular application.")
println("  - Divergent-transition warnings on small N are worth heeding.")
println("    More data or more draws usually settles them.")
