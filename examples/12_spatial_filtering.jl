# ═══════════════════════════════════════════════════════════════════════
# Example 12: Eigenvector Spatial Filtering (ESF / RE-ESF)
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Building and inspecting the eigenvector basis
#   - Stepwise ESF for Gaussian, Poisson, logit and negative binomial
#   - Random-effects ESF with profile-REML shrinkage
#   - How to read the RE-ESF residual Moran coefficient (it is NOT zero!)

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics
import StatsAPI: coef, loglikelihood, nobs, stderror

rng = MersenneTwister(5)
W = make_rook_weights(12)        # 144 units
N = size(W, 1)
S0 = sum(W)

# ─── The eigenvector basis ───────────────────────────────────────────
# ESF adds eigenvectors of Ω = M·Wsym·M as synthetic regressors, where
# M = I - 11'/N. Eigenvector i has Moran coefficient (N/S0)·λᵢ, so the
# leading ones are exactly the strongest positive-autocorrelation map
# patterns.

println("═══ Eigenvector Basis ═══")
se = spatial_eigenvectors(W)
println(se)
println()

# Three identities hold exactly, and they are worth checking:
emp = [moran_coefficient(se.vectors[:, i], W) for i in 1:length(se.values)]
println("max |MC_empirical - (N/S0)lambda| = ", maximum(abs.(emp .- se.moran)))
println("max |E'E - I|                     = ",
        maximum(abs.(transpose(se.vectors) * se.vectors - I)))
println("max |sum of each eigenvector|     = ",
        maximum(abs.(vec(sum(se.vectors, dims=1)))))
println()
println("The last one matters: the basis is built on the orthogonal complement")
println("of the constant vector, so no eigenvector is collinear with the")
println("intercept. Decomposing Omega directly would NOT give this — Omega has")
println("a multi-dimensional null space and eigen returns an arbitrary")
println("rotation within it, smearing the constant across several eigenvectors.")
println()

# Screening keeps only strongly autocorrelated candidates.
se_scr = spatial_eigenvectors(W; min_moran_ratio = 0.25)
println("Unscreened pool: $(length(se.values))  →  screened (ratio 0.25): $(length(se_scr.values))")
println()

# ─── Gaussian ESF on SAR-generated data ──────────────────────────────

println("═══ Gaussian ESF ═══")
X = randn(rng, N, 2)
y = (I - 0.6 * W) \ (1.0 .+ X * [1.5, -0.8] .+ 0.5 * randn(rng, N))

m = fit_esf(X, y, W; family = :gaussian, max_eigenvectors = 15,
            varnames = ["x1", "x2"])
println(m)

s = selected_eigenvectors(m)
println("selected eigenvectors: ", s.indices)
println("their Moran coefficients: ", round.(s.moran; digits = 3))
println("residual Moran: $(round(s.moran_initial; digits=4)) → $(round(s.moran_final; digits=4))")
println()

# Compare with plain OLS, which leaves the autocorrelation in the residuals.
Xo = hcat(ones(N), X)
println("OLS beta = ", round.(Xo \ y; digits = 3))
println("ESF beta = ", round.(coef(m)[1:3]; digits = 3),
        "  (plus $(length(s.indices)) eigenvector coefficients)")
println()

# ─── GLM families ────────────────────────────────────────────────────
# ESF's main attraction: it drops into any GLM without needing a spatial
# Jacobian. Families are fitted by IRLS.

function rpois(rng, lam)
    Lc = exp(-lam); k = 0; p = 1.0
    while true
        p *= rand(rng)
        p <= Lc && return Float64(k)
        k += 1
        k > 10_000 && return Float64(k)
    end
end

rng2 = MersenneTwister(31)
Xc = randn(rng2, N, 2)
# A spatially structured latent signal built from true eigenvectors.
sig = 1.2 * se_scr.vectors[:, 2] * sqrt(N) * 0.4 +
      0.8 * se_scr.vectors[:, 5] * sqrt(N) * 0.4
eta = 0.6 .+ Xc * [0.5, -0.35] .+ sig

println("═══ Poisson ESF ═══")
yp = [rpois(rng2, exp(e)) for e in eta]
mp = fit_esf(Xc, yp, W; family = :poisson, max_eigenvectors = 8,
             eigen_basis = se_scr, varnames = ["x1", "x2"])
println(mp)
println("beta_hat = ", round.(coef(mp)[1:3]; digits = 3), "   true = [0.6, 0.5, -0.35]")
println()

println("═══ Logit ESF ═══")
yb = Float64[(1.0 / (1.0 + exp(-e))) > rand(rng2) ? 1.0 : 0.0 for e in eta]
ml = fit_esf(Xc, yb, W; family = :logit, max_eigenvectors = 8,
             eigen_basis = se_scr, varnames = ["x1", "x2"])
println(ml)
println("Selecting ZERO eigenvectors is a legitimate outcome, not a failure:")
println("it means the residuals were already within moran_tol of no spatial")
println("autocorrelation. Binary outcomes discard much of the latent signal,")
println("so this happens readily with logit.")
println()

println("═══ Negative Binomial ESF ═══")
yn = Float64[]
for e in eta
    g = sum(-log(rand(rng2)) for _ in 1:2) * 0.5   # mean-1 gamma, shape 2
    push!(yn, rpois(rng2, exp(e) * g))
end
mn = fit_esf(Xc, yn, W; family = :negbin, max_eigenvectors = 8,
             eigen_basis = se_scr, varnames = ["x1", "x2"])
println(mn)
println("dispersion alpha_hat = $(round(mn.alpha; digits=3))")
println("var/mean of y = $(round(var(yn) / mean(yn); digits=2))  (>1 means overdispersed)")
println()

# ─── Random-effects ESF ──────────────────────────────────────────────
# Stepwise selection introduces model-selection variance. RE-ESF keeps the
# whole screened pool and shrinks instead, with one variance ratio
# theta = tau2/sigma2 estimated by profile REML.

println("═══ RE-ESF ═══")
pos = findall(>(1e-10), se_scr.values)
E = se_scr.vectors[:, pos]
lam = se_scr.values[pos]
Lp = length(pos)

# Generate from the RE-ESF model itself so the parameters have a true value.
rng3 = MersenneTwister(3)
Xr = randn(rng3, N, 2)
gam = sqrt(0.8) .* sqrt.(lam) .* randn(rng3, Lp)
yr = 1.0 .+ Xr * [1.5, -0.8] .+ E * gam .+ 0.5 .* randn(rng3, N)

mr = fit_re_esf(Xr, yr, W; varnames = ["x1", "x2"])
println(mr)
println("true beta = [1.0, 1.5, -0.8], true tau2 = 0.8, true sigma2 = 0.25, true theta = 3.2")
println()

# ─── Reading the RE-ESF residual Moran coefficient ───────────────────

println("═══ Why is the residual Moran coefficient NEGATIVE? ═══")
println()
println("moran_final here is $(round(mr.moran_final; digits=4)) — and that is CORRECT.")
println("Do not compare it against zero.")
println()
println("Subtracting E*gamma_hat removes the positively autocorrelated")
println("directions, so the residual lives mostly in the orthogonal complement")
println("of span(E). A vector drawn at RANDOM from that complement is itself")
println("strongly negatively autocorrelated. Its expected Moran coefficient is")
println("available in closed form and reported as moran_reference:")
println()
println("  moran_reference = $(round(mr.moran_reference; digits=4))")
println("  moran_final     = $(round(mr.moran_final; digits=4))")
println()
println("The fitted residual is slightly LESS negative than chance, which is")
println("the healthy case. Verify the closed form by Monte Carlo:")

function rand_perp(s, N, E)
    v = randn(MersenneTwister(s), N)
    v .-= mean(v)
    return v - E * (transpose(E) * v)
end
mc_ref = mean(moran_coefficient(rand_perp(s, N, E), W) for s in 1:300)
println("  Monte Carlo estimate of the same quantity = $(round(mc_ref; digits=4))")
println()

# ─── Diagnosing noise absorption ─────────────────────────────────────
# Because the whole pool is retained, a large pool lets the random effects
# absorb noise as well as signal. Watch edf and the reference gap.

println("═══ Pool Size and Noise Absorption ═══")
println()
println("ratio | pool |  theta  | edf   | resid MC | reference | gap")
println("──────┼──────┼─────────┼───────┼──────────┼───────────┼───────")
for r in (0.0, 0.25, 0.5, 0.75)
    seb = spatial_eigenvectors(W; min_moran_ratio = r)
    mb = fit_re_esf(Xr, yr, W; eigen_basis = seb, varnames = ["x1", "x2"])
    gap = mb.moran_final - mb.moran_reference
    println("$(rpad(r, 5)) | $(rpad(length(mb.lambda), 4)) | ",
            "$(rpad(round(mb.theta; digits=3), 7)) | ",
            "$(rpad(round(mb.edf; digits=1), 5)) | ",
            "$(rpad(round(mb.moran_final; digits=4), 8)) | ",
            "$(rpad(round(mb.moran_reference; digits=4), 9)) | ",
            "$(round(gap; digits=3))")
end
println()
println("A healthy fit has moran_final at or ABOVE moran_reference (positive")
println("gap). Tighten min_moran_ratio when edf is an uncomfortable fraction")
println("of N = $N, or when the gap turns sharply negative.")
println()
println("But do not push the ratio arbitrarily high: at 0.75 the pool is too")
println("small to absorb the structure at all, theta collapses, and the")
println("residual Moran coefficient turns POSITIVE — genuine leftover")
println("autocorrelation, i.e. under-filtering. The target is a residual that")
println("is mildly negative and close to its reference, not one of either")
println("extreme. The 0.25 default is a reasonable starting point.")
