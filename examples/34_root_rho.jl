# ---------------------------------------------------------------------------
# 34. root_rho: ρ without a log-determinant, with a large basis for f
#
# y = ρWy + xβ + f(u, v) + ε,  f an unknown smooth surface over the locations
#
# f is approximated by a tensor hat basis on (u, v). A basis big enough for
# valid inference makes K/n non-negligible, and the plain closed form
# (closed_form_rho, no trace correction) is then biased by O(K/n). root_rho
# uses Jin & Lee's (2012) moment, whose mean is exactly zero for any K, and
# reports two intervals: the Wald interval and the test-inversion set.
#
# Section 4 fits a penalised smoother (tensor_pspline) in place of the projection,
# section 5 its bias-aware interval, with λ chosen by the interval's length, and
# section 6 the same on a reference basis, for an f outside the span of the basis.
#
# Run: julia --project=. examples/34_root_rho.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf

const RNG = MersenneTwister(20_261_120)

# A tensor product of piecewise-linear "hat" functions on k equally spaced
# knots in each coordinate: k² columns that sum to one, so no intercept in X.
hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

# ── 1. Simulate: 900 units, one per cell of a 30 × 30 grid, jittered ─────────
#
# ρ = 0.4, β = 1, heteroskedastic errors whose variance rises across the map.

g = 30; n = g^2
P = (hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(RNG, n, 2)) ./ g
u, v = P[:, 1], P[:, 2]
W = knn_weights(P, 6)
x = randn(RNG, n)
f = 2 .* sin.(2π .* u) .* cos.(π .* v)
σ = sqrt.(exp.(2 .* (u .+ v .- 1)))
y = lu(sparse(I - 0.4 * W)) \ (x .+ f .+ σ .* randn(RNG, n))
X = reshape(x, :, 1)

# ── 2. One fit ──────────────────────────────────────────────────────────────

r = root_rho(y, X, W; basis = hatbasis(u, v, 8), varnames = ["x"])
println(r)

# ── 3. Across basis sizes, against the plain closed form ───────────────────
#
# Report ρ̂ across a few K: the interval is valid only when the basis is large
# enough that the approximation error of f is small against the noise, which
# calls for more columns than a good fit of f needs. The plain closed form
# drifts down as K grows; root_rho does not.

println("\n  K    K/n    root_rho (SE)      test-inversion set     closed_form_rho")
for k in (6, 8, 11, 14)
    B = hatbasis(u, v, k)
    rk = root_rho(y, X, W; basis = B)
    cf = closed_form_rho(y, hcat(X, B), W)
    cs = join((@sprintf("[%.3f, %.3f]", a, b) for (a, b) in rk.cs), " ∪ ")
    @printf("%4d  %.3f   %.3f (%.3f)    %-22s %.3f\n", rk.K, rk.K / n, rk.rho, rk.se, cs, cf.rho)
end

# ── 4. A penalised smoother in place of the projection ─────────────────────
#
# tensor_pspline gives a 15 × 15 tensor hat basis (225 columns) with a
# second-difference penalty. λ sets how much the surface is smoothed; tr S
# (r.df) is its effective number of columns. A smoother shrinks f, and the
# shrinkage biases ρ̂ and both intervals, so take λ small. There is no default:
# GCV smooths too much for inference. Report ρ̂ across a few λ. In this
# sample the three test-inversion sets overlap, and each contains ρ = 0.4.

ps = tensor_pspline(u, v; k = 15)
println("\n    λ      df    df/n    root_rho (SE)      test-inversion set")
for λ in (0.01, 0.1, 1.0)
    rλ = root_rho(y, X, W; ps..., lambda = λ)
    cs = join((@sprintf("[%.3f, %.3f]", a, b) for (a, b) in rλ.cs), " ∪ ")
    @printf("%6.2f  %6.1f  %.3f   %.3f (%.3f)    %s\n", λ, rλ.df, rλ.df / n, rλ.rho, rλ.se, cs)
end

# ── 5. The bias-aware interval, with λ chosen by its length ────────────────
#
# Given a bound ℛ on the roughness of f, root_rho widens the critical value by
# a bound on the shrinkage bias, so a larger λ stays valid (THEORY.md,
# Proposition 7). ℛ is an assumption, not an estimate: state it through a
# reference surface at least as rough as f is believed to be. Here the belief
# is "no rougher than 2·sin(2πu)cos(πv)", which is the truth in this
# simulation; in practice err on the large side, and report a few multiples.
# With a vector of λ, the projection is a candidate too (λ = 0, bias bound 0),
# and the shortest interval is returned.

f_ref = 2 .* sin.(2π .* u) .* cos.(π .* v)
ℛ = basis_roughness(f_ref, ps.basis, ps.penalty; X)
rp = root_rho(y, X, W; basis = ps.basis)
@printf("\nℛ = %.3f from the reference surface; the projection's Wald half-length %.3f\n",
        ℛ, (rp.ci_wald[2] - rp.ci_wald[1]) / 2)
println("  ℛ/ref    λ chosen   bias-aware interval   half-length")
for m in (1.0, 2.0, 4.0)
    rb = root_rho(y, X, W; ps..., lambda = 10.0 .^ (-2:0.5:1), roughness = m * ℛ)
    @printf("  %4.0f    %8.3g    [%.3f, %.3f]       %.3f\n", m, rb.lambda, rb.ci_bias_aware...,
            (rb.ci_bias_aware[2] - rb.ci_bias_aware[1]) / 2)
end

# ── 6. When f may not be in the span of the basis: a reference basis ───────
#
# Section 5's class assumes f = [X, basis]ψ, so the projection gets no bias
# bound. A real f leaves a remainder, which biases every candidate. With
# `reference`, the roughness bound is stated on a finer basis (here 30 × 30,
# more columns than units), and the bound covers the remainder (THEORY.md,
# Lemma 5). ℛ̃ comes from the knot values of the reference surface. Compare a
# coarse basis with a fine one: the coarse one pays for its remainder in length.

ref = tensor_pspline(u, v; k = 30)
ℛ̃ = basis_roughness((s, t) -> 2sin(2π * s) * cos(π * t), u, v; k = 30)
@printf("\nℛ̃ = %.3f on the reference basis (K̃ = %d)\n", ℛ̃, size(ref.basis, 2) + 1)
println("   k   reference   λ chosen   bias-aware interval   half-length")
for k in (6, 15)
    psk = tensor_pspline(u, v; k)
    for (lab, kw) in (("no ", (; roughness = basis_roughness(f, psk.basis, psk.penalty; X))),
                      ("yes", (; roughness = ℛ̃, reference = ref)))
        rk = root_rho(y, X, W; psk..., lambda = 10.0 .^ (-2:0.5:1), kw...)
        @printf("  %2d      %s     %8.3g    [%.3f, %.3f]       %.3f\n", k, lab, rk.lambda,
                rk.ci_bias_aware..., (rk.ci_bias_aware[2] - rk.ci_bias_aware[1]) / 2)
    end
end
