# ---------------------------------------------------------------------------
# 30. Semiparametric spatial models: a smoother that can steal ρ
#
# `y = ρWy + Xβ + m(z) + ε` with `m` unknown. Su & Jin (2010) estimate it by
# profile QML: a local-polynomial smoother on `z` takes out `m`, and the
# concentrated likelihood in `ρ` keeps the log-determinant.
#
# The catch is identification. If `z` carries geography, the smoother can
# reproduce the same low-frequency spatial patterns that `ρWy` explains, and
# the two compete for them: Geniaux's (2026) "geoadditive trap". Every fit
# reports how much of `W`'s dominant modes the smoother can reproduce. That
# number is a measure, not a test: it has no null distribution, and a small
# value means only that `ρ` is identified *given W* — never that `ρ̂` is a
# causal spillover (Gibbons & Overman 2012).
#
# The last section fits a panel whose coefficients drift smoothly in time
# (Liang, Gao & Gong 2022). There `ρ` is constant; a time-varying `ρ` has no
# published method and is not offered.
#
# Run: julia --project=. examples/30_semiparametric.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: stderror

const RNG = MersenneTwister(20_261_113)

# ── 1. A partially linear SAR where z is not spatial ───────────────────────
#
# Su & Jin's own design: x = 0.2z² − z + v, m(z) = 2·logistic(z), with z
# drawn independently of location.

n = 600
coords = rand(RNG, n, 2)
W = sparse(knn_weights(coords, 6))
ρ0 = 0.5

z = randn(RNG, n)
x = 0.2 .* z .^ 2 .- z .+ randn(RNG, n)
y = (I - ρ0 * W) \ (x .+ 2 .* exp.(z) ./ (1 .+ exp.(z)) .+ randn(RNG, n))

m = fit_partially_linear_sar(y, reshape(x, :, 1), z, W; method = ExactLU(),
                             varnames = ["x"])
println(m)

# The smoother absorbs the intercept, so X must not carry one: the fit is
# refused rather than silently giving an unidentified coefficient.
try
    fit_partially_linear_sar(y, hcat(ones(n), x), z, W)
catch err
    println("Refused, as it should be: ", first(sprint(showerror, err), 90), "…\n")
end

# ── 2. The closed-form ρ on the same first stage ───────────────────────────
#
# Smirnov's estimator needs no log-determinant. With a kernel smoother in the
# first stage no consistency proof covers it, so it is shown beside the QML,
# not instead of it.

c = closed_form_rho(m)
println(c)
@printf("QML ρ̂ = %.4f (se %.4f)    closed-form ρ̂ = %.4f (SHAC se %.4f)\n\n",
        m.rho, stderror(m)[2], c.rho, c.se_shac)

# ── 3. The geoadditive trap ────────────────────────────────────────────────
#
# Now let z be the coordinates themselves, with a smooth surface m(u, v). The
# smoother in (u, v) spans the same low-frequency patterns as the dominant
# eigenvectors of W, and the overlap says so.

msurf = 2 .* sin.(π .* coords[:, 1]) .* cos.(π .* coords[:, 2])
xg = coords[:, 1] .+ randn(RNG, n)
yg = (I - ρ0 * W) \ (xg .+ msurf .+ randn(RNG, n))
mg = fit_partially_linear_sar(yg, reshape(xg, :, 1), coords, W; method = ExactLU(),
                              varnames = ["x"])

@printf("overlap π̄ with z unrelated to location: %.3f\n", m.overlap.pibar)
@printf("overlap π̄ with z = the coordinates:     %.3f\n", mg.overlap.pibar)
@printf("ρ̂ in the trap: %.4f (se %.4f), true ρ = %.2f\n\n", mg.rho, stderror(mg)[2], ρ0)
println(mg.overlap)

# ── 4. Coefficients that drift in time ─────────────────────────────────────
#
# A panel on a ring, with β(τ) = 1 + 2τ + 2τ² on x and a time-varying
# intercept 1 + τ³; ρ constant at 0.4.

N, Tp = 60, 30
Wr = spzeros(N, N)
for i in 1:N, k in (-2, -1, 1, 2)
    Wr[i, mod1(i + k, N)] = 0.25
end
τ = (1:Tp) ./ Tp
xp = randn(RNG, N, Tp)
α = randn(RNG, N)
Y = zeros(N, Tp)
for t in 1:Tp
    Y[:, t] = (I - 0.4Wr) \ ((1 + τ[t]^3) .+ (1 + 2τ[t] + 2τ[t]^2) .* xp[:, t] .+ α .+
                             randn(RNG, N))
end
tv = fit_tv_coef_sar(vec(Y), hcat(ones(N * Tp), vec(xp)), Wr, N, Tp;
                     method = ExactLU(), varnames = ["const", "x"])
println(tv)

println("  τ     β̂_x(τ)   ±1.96 se   true β_x(τ)")
for s in (3, 10, 15, 20, 28)
    @printf("%5.2f  %8.4f  %8.4f   %8.4f\n", tv.tau[s], tv.beta[s, 2],
            1.96 * tv.beta_se[s, 2], 1 + 2τ[s] + 2τ[s]^2)
end
println("\nThe band excludes the smoothing bias (order h²β''), so near the ends,")
println("where the local-linear fit is one-sided, it can miss the true curve.")
