# Monte Carlo for root_rho's bias-aware interval (THEORY.md, Lemma 4 and Proposition 7),
# through the released function, with λ chosen by the interval's length.
#
# The research check (research/trace_cfe/tc_honest.jl) used the same design but its own code,
# with the exact-diagonal projection moment as the λ = 0 candidate and as the pilot. The
# package uses Jin & Lee's REsd as the λ = 0 candidate and their first-step root as the
# pilot, so this run tests the package's variant, not a copy of the research numbers.
#
# Design: tc_honest's (n = 900 random locations, kNN-6 W from knn_weights, a 15 × 15 tensor hat basis with a
# second-difference penalty, x unpenalised as X, f = x + a·sin(2πu)cos(πv), smooth variances,
# normal errors, ρ₀ = 0.4). λ on the half-decade grid 10⁻³ … 10² (12 values, plus the
# projection). ℛ ∈ {0, ℛ*, 4ℛ*}, with ℛ* = basis_roughness(f, …), a ∈ {1, 4}, R = 300 each:
# 6 cells, 1,800 fits at about 3.4 s each.
# Pass criterion (set before the run): with ℛ = ℛ* and 4ℛ*, the chosen interval covers
# within 0.93–0.99 in all four cells. Power: with ℛ = 0 it covers below 0.90 in both.
#
# Run:    julia --project=. validation/mc_root_rho_bias_aware.jl [R]   (R = 300 by default)
#         julia --project=. validation/mc_root_rho_bias_aware.jl time  (cost per fit)
# Output: validation/mc_root_rho_bias_aware_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function design(; n = 900, kh = 15, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knn_weights(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    B = hatbasis(u, v, kh)
    D2 = diff(diff(Matrix(1.0I, kh, kh), dims = 1), dims = 1)
    Ω = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; n, W, u, v, X = reshape(x, :, 1), B, Ω, σ2, Fac = lu(sparse(I - 0.4 * W)))
end

const λS = 10.0 .^ (-3:0.5:2)

function cell(d, amp, mult, R; seed = 11)
    f = d.X[:, 1] .+ amp .* sin.(2π .* d.u) .* cos.(π .* d.v)
    ℛs = basis_roughness(f, d.B, d.Ω; X = d.X)
    rng = MersenneTwister(seed + round(Int, 10amp))           # the same draws for every ℛ
    cov = 0; hl = Float64[]; hlp = Float64[]; e = Float64[]; picks = zeros(Int, length(λS) + 1)
    for r in 1:R
        y = d.Fac \ (f .+ sqrt.(d.σ2) .* randn(rng, d.n))
        g = root_rho(y, d.X, d.W; basis = d.B, penalty = d.Ω, lambda = λS, roughness = mult * ℛs)
        g.root_found || continue
        lo, hi = g.ci_bias_aware
        cov += lo <= 0.4 <= hi; push!(hl, (hi - lo) / 2); push!(e, g.rho - 0.4)
        push!(hlp, g.halflength_grid[1]); picks[findfirst(==(g.lambda), g.lambda_grid)] += 1
    end
    m = length(hl)
    (; ℛs, cov = cov / m, hl = mean(hl), hlp = mean(hlp), bias = mean(e), sd = std(e), m, picks)
end

function main(R)
    d = design()
    println("root_rho's bias-aware interval, λ chosen by length (Proposition 7); Julia $VERSION")
    println("n = $(d.n), kNN-6, 15 × 15 hat basis + x, λ grid 10^(-3:0.5:2) plus the projection, R = $R\n")
    pass = true; power = true
    for amp in (1.0, 4.0), mult in (0.0, 1.0, 4.0)
        t0 = time(); c = cell(d, amp, mult, R)
        ok = mult == 0 ? c.cov < 0.90 : 0.93 <= c.cov <= 0.99
        mult == 0 ? (power &= ok) : (pass &= ok)
        picks = join(["$(j == 1 ? 0 : round(λS[j-1], sigdigits = 3)):$(c.picks[j])"
                      for j in eachindex(c.picks) if c.picks[j] > 0], " ")
        @printf("a = %.0f, ℛ = %.0f ℛ* (ℛ* = %.3f): coverage %.3f (%d fits), half-length %.4f (projection %.4f), bias %+.4f, sd %.4f | %s | %.1f min\n",
                amp, mult, c.ℛs, c.cov, c.m, c.hl, c.hlp, c.bias, c.sd, ok ? "ok" : "FAIL",
                (time() - t0) / 60)
        println("    picks (λ: count) ", picks)
    end
    println("\nPass criterion (ℛ = ℛ* and 4ℛ* within 0.93–0.99): ", pass ? "met" : "NOT met")
    println("Power check (ℛ = 0 below 0.90): ", power ? "met" : "NOT met")
end

if !isempty(ARGS) && ARGS[1] == "time"
    d = design(); f = d.X[:, 1] .+ sin.(2π .* d.u) .* cos.(π .* d.v)
    y = d.Fac \ (f .+ sqrt.(d.σ2) .* randn(MersenneTwister(1), d.n))
    ℛs = basis_roughness(f, d.B, d.Ω; X = d.X)
    root_rho(y, d.X, d.W; basis = d.B, penalty = d.Ω, lambda = λS, roughness = ℛs)
    t = @elapsed root_rho(y, d.X, d.W; basis = d.B, penalty = d.Ω, lambda = λS, roughness = ℛs)
    @printf("one fit over the grid: %.2f s; 6 cells × R = 300: about %.0f min\n", t, 6 * 300 * t / 60)
else
    main(isempty(ARGS) ? 300 : parse(Int, ARGS[1]))
end
