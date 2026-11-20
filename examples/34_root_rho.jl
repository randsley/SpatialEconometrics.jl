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
