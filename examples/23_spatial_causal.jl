# ═══════════════════════════════════════════════════════════════════════
# Example 23: Spatial difference-in-differences
# ═══════════════════════════════════════════════════════════════════════
#
# A policy is rolled out to a contiguous block of a region. A conventional
# difference-in-differences asks what it did to the treated units, and gets
# the answer wrong twice over: the effect on their neighbours is pushed
# into the error term, where it is correlated with treatment by
# construction, and the "control" group includes units the policy reached.
#
# This example generates a design where the truth is known, shows what a
# naive DiD reports, and then recovers the direct and spillover effects
# separately.
#
# Demonstrates:
#   - exposure / ring_exposure: naming what "treated nearby" means
#   - check_interference: what the design can identify, before fitting
#   - fit_spatial_did: the four effects, and why there are four
#   - what the omitted exposure column does to a naive estimate
#   - the covariance choice, and why the default is not the spatial one

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Printf, Statistics
using StatsAPI: stderror

# ─── A region, and a policy applied to a block of it ─────────────────

side = 14
N = side^2
W = make_rook_weights(side)
coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

D = zeros(N)
for i in 1:N
    r, c = coords[i, 1], coords[i, 2]
    (4 <= r <= 8 && 4 <= c <= 8) && (D[i] = 1.0)
end

println("="^70)
println("A region of $N units; the policy reaches $(Int(sum(D))) of them")
println("="^70)

# ─── Exposure: what "treated nearby" is going to mean ────────────────

em = exposure(D, W)
E = em.E[:, 1]

@printf("\nExposure is the fraction of a unit's neighbours that are treated.\n")
@printf("  mean exposure, treated units : %.3f\n", mean(E[D .== 1.0]))
@printf("  mean exposure, control units : %.3f\n", mean(E[D .== 0.0]))
@printf("  controls with any exposure   : %d of %d\n",
        count(>(0), E[D .== 0.0]), count(iszero, D))

# ─── What the design can identify, checked before fitting ────────────

println("\n", check_interference(em))

# ─── Generate from a known truth ─────────────────────────────────────

γ, τ, δ = 0.5, 3.0, -1.5     # time effect, direct effect, spillover
rng = MersenneTwister(2026)
μ = 2.0 .* randn(rng, N)     # arbitrary, persistent unit differences

y = zeros(N, 2)
y[:, 1] .= μ .+ 0.4 .* randn(rng, N)
y[:, 2] .= μ .+ γ .+ τ .* D .+ δ .* E .+ 0.4 .* randn(rng, N)

println("\n", "="^70)
println("Truth: direct effect τ = $τ, spillover coefficient δ = $δ")
println("="^70)

# ─── What a naive DiD reports ────────────────────────────────────────
#
# The textbook two-by-two: mean change among treated minus mean change
# among controls. It has no term for the spillover, so the spillover ends
# up in both halves of the subtraction.

Δy = y[:, 2] .- y[:, 1]
naive = mean(Δy[D .== 1.0]) - mean(Δy[D .== 0.0])
@printf("\nNaive 2x2 DiD        : %7.4f   (truth for the direct effect: %.4f)\n",
        naive, τ)
@printf("  error              : %+7.4f\n", naive - τ)

# The bias has a known sign here. The spillover is negative, so the
# controls nearest the block were pushed down, which makes the treated
# look better by comparison than they are.

# ─── The spillover design ────────────────────────────────────────────

m = fit_spatial_did(y, D, W)
println("\n", m)

Ē_T = mean(E[D .== 1.0])
Ē_C = mean(E[D .== 0.0])
println("\n", "="^70)
println("Recovered against the truth")
println("="^70)
@printf("%-26s %10s %10s %9s %8s\n", "", "estimate", "truth", "error", "in SEs")
for (lab, est, want, se) in (
        ("direct effect", m.att_direct, τ, m.att_direct_se),
        ("spillover, on treated", m.att_spill_treated, δ * Ē_T, m.att_spill_treated_se),
        ("spillover, on controls", m.att_spill_control, δ * Ē_C, m.att_spill_control_se),
        ("δ (the coefficient)", m.coef[3], δ, stderror(m)[3]))
    @printf("%-26s %10.4f %10.4f %+9.4f %8.2f\n", lab, est, want, est - want,
            (est - want) / se)
end

# The last column is the one to read: every effect is within sampling
# error of the truth. δ is the least precisely estimated of them, and the
# interference report above says why -- only 20 of 171 controls have any
# exposure at all, so there is little variation to identify it from.

@printf("\nThe naive estimate missed the direct effect by %.4f.\n", abs(naive - τ))
@printf("The spillover design missed it by %.4f.\n", abs(m.att_direct - τ))

# ─── Rings: where does the effect stop? ──────────────────────────────
#
# A single W answers "are your neighbours treated". Bands answer "how far
# out does this reach", which is usually the more interesting question --
# and the one that separates a real spillover from displacement.

println("\n", "="^70)
println("Reading the spillover at increasing distance")
println("="^70)
@printf("%-14s %12s %12s %10s\n", "band", "δ̂", "std. error", "z")
for (lo, hi) in ((0.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 6.0))
    emr = ring_exposure(D, coords, lo, hi)
    rep = check_interference(emr)
    if !rep.identified
        @printf("(%.1f, %.1f]  %12s   %s\n", lo, hi, "--", "not identified")
        continue
    end
    # Build the ring's weight matrix directly so the fit uses that band.
    Wr = band_weights(coords, lo, hi; row_standardize = true)
    mr = fit_spatial_did(y, D, Wr)
    @printf("(%.1f, %.1f]  %12.4f %12.4f %10.2f\n",
            lo, hi, mr.coef[3], stderror(mr)[3], mr.coef[3] / stderror(mr)[3])
end

println("""

The estimate decays with distance, which is the right qualitative read:
the generating process put the spillover on the rook neighbours alone.
But the bands are not independent regressors, and the second one is
significant here even though nothing was generated at that range. A unit
two steps from a contiguous block is adjacent to a unit one step from it,
so a single-band fit picks up the true first-ring effect through that
correlation. Reading a ring in isolation tells you how far the signal
reaches, not where it originates."""

# ─── Standard errors ─────────────────────────────────────────────────
#
# The default is cluster-robust on unit, chosen on measured coverage
# rather than on convention. Conley's spatial HAC is available, but on
# this design it degrades as the cutoff widens -- and it does so even
# with no spatial correlation present, so the decay is the finite-sample
# cost of estimating more pair covariances rather than the correction
# working. See the fit_spatial_did docstring for the figures.

println("\n", "="^70)
println("The same fit under different covariance estimators")
println("="^70)
@printf("%-28s %12s %12s\n", "estimator", "se(direct)", "z")
for (lab, kw) in (("cluster on unit (default)", (vcov = :cluster,)),
                  ("OLS", (vcov = :ols,)),
                  ("Conley, cutoff 1.5", (vcov = :conley, coords = coords, dist_cutoff = 1.5)),
                  ("Conley, cutoff 4.0", (vcov = :conley, coords = coords, dist_cutoff = 4.0)))
    mv = fit_spatial_did(y, D, W; kw...)
    @printf("%-28s %12.4f %12.2f\n", lab, mv.att_direct_se,
            mv.att_direct / mv.att_direct_se)
end

# ─── Staggered rollout ───────────────────────────────────────────────

println("\n", "="^70)
println("A staggered rollout, with the caveat that comes with it")
println("="^70)

T_per = 6
adopt = fill(0, N)
for i in 1:N
    c = coords[i, 2]
    c <= 4 ? (adopt[i] = 3) : c <= 8 ? (adopt[i] = 5) : nothing
end
Dpath = zeros(N, T_per)
for i in 1:N, t in 1:T_per
    adopt[i] != 0 && t >= adopt[i] && (Dpath[i, t] = 1.0)
end

Wt = row_standardize(W)
λ = [0.0, 0.2, -0.1, 0.3, 0.15, -0.2]
ys = zeros(N, T_per)
for i in 1:N, t in 1:T_per
    E_it = dot(view(Wt, i, :), view(Dpath, :, t))
    ys[i, t] = μ[i] + λ[t] + τ * Dpath[i, t] + δ * E_it + 0.4 * randn(rng)
end

ms = fit_spatial_did(ys, Dpath, W)
@printf("\nHomogeneous effects, %d periods, two adoption cohorts:\n", T_per)
@printf("  direct effect  %.4f  (truth %.4f)\n", ms.att_direct, τ)
@printf("  δ              %.4f  (truth %.4f)\n", ms.coef[2], δ)

println("""
With homogeneous effects this is exact up to noise. With effects that
differ across adoption cohorts it is not, and the error is large -- 17% to
67% on the designs measured in the docstring, with a sign that is not
predictable. Spatially contiguous cohorts, which is how rollouts actually
happen, are the worst case: cohort identity is then nearly collinear with
the exposure column. Read cohorts separately when heterogeneity is
plausible.""")
