# Monte Carlo for root_rho's bias-aware interval with the smoother as the pilot (THEORY.md,
# Lemma 13 and Proposition 7″), through the released function, with λ chosen by the interval's
# length.
#
# Why: on an irregular region a tensor_pspline has basis functions with no unit, or one unit, in
# their support, so [X, basis] is rank-deficient and the projection pilot of Proposition 7 does
# not run (found on the 2,882 Portuguese parishes, THEORY.md, "Two real-data runs"). The smoother
# at λ_p is the pilot instead, at the price of two constants (ω_p and c̄_p·ℛ²·r_p²).
#
# Design A, the irregular region. n = 900 locations uniform on the triangle u + v ≤ 1, kNN-6 W,
# tensor_pspline(u, v; k = 15) over the bounding box, so about half the basis functions have no
# unit. f = x + a·sin(2πu)cos(πv), errors normal with smooth variances exp(3(u + v − 1)) scaled
# to mean 1, ρ₀ = 0.4. λ on 10^(-2:0.5:2) (9 values; the pilot at λ_p = 0.01, the default).
# ℛ ∈ {0, ℛ*, 4ℛ*}, ℛ* = basis_roughness(f, …), a ∈ {1, 4}, R = 300 each.
# Design B, the square: mc_root_rho_bias_aware.jl's design (n = 900 uniform on [0, 1]², a 15 × 15
# hat basis of full rank), with the same λ grid and ℛ = ℛ*, run with pilot = :projection and
# pilot = :penalised on the same draws: the cost of the smoother pilot where both run.
#
# Pass criteria (set before the run):
# (1) design A, ℛ = ℛ* and 4ℛ*: the chosen interval covers within 0.93–0.99 in all four cells;
# (2) design B: the smoother pilot covers within 0.93–0.99 in both cells.
# Reported, not a criterion: ℛ = 0 (no bias allowance), expected to under-cover where the
# shrinkage bias is large; and the half-length of the smoother pilot's interval against the
# projection pilot's in design B.
#
# Run:    julia -t 14 --project=. validation/mc_root_rho_pilot.jl [R]   (R = 300 by default)
# Output: validation/mc_root_rho_pilot_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
BLAS.set_num_threads(1)

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

const λS = 10.0 .^ (-2:0.5:2)

function triangle(; n = 900, k = 15, seed = 13)
    rng = MersenneTwister(seed)
    P = zeros(0, 2)
    while size(P, 1) < n
        q = rand(rng, 2)
        sum(q) <= 1 && (P = vcat(P, q'))
    end
    u, v = P[:, 1], P[:, 2]
    W = knn_weights(P, 6); x = randn(rng, n)
    ps = tensor_pspline(u, v; k)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; name = "A, triangle", n, W, u, v, X = reshape(x, :, 1), B = ps.basis, Ω = ps.penalty, σ2,
       Fac = lu(sparse(I - 0.4 * W)))
end

function square(; n = 900, kh = 15, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knn_weights(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    D2 = diff(diff(Matrix(1.0I, kh, kh), dims = 1), dims = 1)
    Ω = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; name = "B, square", n, W, u, v, X = reshape(x, :, 1), B = hatbasis(u, v, kh), Ω, σ2,
       Fac = lu(sparse(I - 0.4 * W)))
end

surface(d, amp) = d.X[:, 1] .+ amp .* sin.(2π .* d.u) .* cos.(π .* d.v)

# R draws; each draw has its own seed, so the threads do not change the numbers.
function cell(d, amp, ℛ, R; pilot = :auto, seed = 11)
    f = surface(d, amp)
    out = Vector{Any}(undef, R)
    Threads.@threads for r in 1:R
        rng = MersenneTwister(seed + 1000 * round(Int, amp) + r)
        y = d.Fac \ (f .+ sqrt.(d.σ2) .* randn(rng, d.n))
        g = root_rho(y, d.X, d.W; basis = d.B, penalty = d.Ω, lambda = λS, roughness = ℛ, pilot)
        out[r] = g.root_found ? (g.ci_bias_aware, g.rho, g.lambda, g.bias_bound / sqrt(g.vhat),
                                 g.pilot, g.pilot_left_out) : nothing
    end
    ok = filter(!isnothing, out)
    lo = [o[1][1] for o in ok]; hi = [o[1][2] for o in ok]
    (; cov = mean(lo .<= 0.4 .<= hi), hl = mean((hi .- lo) ./ 2), bias = mean(o[2] for o in ok) - 0.4,
       sd = std([o[2] for o in ok]), m = length(ok), t = mean(o[4] for o in ok),
       picks = [count(o -> o[3] == λ, ok) for λ in vcat(0.0, λS)],
       pilots = unique(o[5] for o in ok), left = maximum(o[6] for o in ok))
end

function row(label, c)
    @printf("  %-22s cover %.3f  half-length %.4f  bias %+.4f  sd %.4f  mean b̄/√V̂ %.2f  m %d  pilot %s (left out ≤ %d)\n",
            label, c.cov, c.hl, c.bias, c.sd, c.t, c.m, join(c.pilots, "/"), c.left)
    println("      λ picks (0, 10^(-2:0.5:2)): ", c.picks)
end

function main(R)
    println("root_rho's bias-aware interval with the smoother as the pilot (Proposition 7″)")
    println("Julia $VERSION, $(Threads.nthreads()) threads, R = $R per cell, λ grid 10^(-2:0.5:2)\n")
    pass = true
    A = triangle()
    Z = hcat(A.X, A.B)
    @printf("Design A: n = %d, K = %d, rank [x, B] = %d, %d basis functions with no unit\n",
            A.n, size(Z, 2), rank(Z), count(iszero, vec(maximum(A.B; dims = 1))))
    for amp in (1.0, 4.0)
        ℛs = basis_roughness(surface(A, amp), A.B, A.Ω; X = A.X)
        @printf(" a = %.0f, ℛ* = %.3f\n", amp, ℛs); flush(stdout)
        for (lab, mult) in (("ℛ = 0", 0.0), ("ℛ = ℛ*", 1.0), ("ℛ = 4ℛ*", 4.0))
            c = cell(A, amp, mult * ℛs, R)
            row(lab, c); flush(stdout)
            mult > 0 && !(0.93 <= c.cov <= 0.99) && (pass = false)
        end
    end
    B = square()
    @printf("\nDesign B: n = %d, K = %d, rank [x, B] = %d (full)\n", B.n, size(B.B, 2) + 1,
            rank(hcat(B.X, B.B)))
    for amp in (1.0, 4.0)
        ℛs = basis_roughness(surface(B, amp), B.B, B.Ω; X = B.X)
        @printf(" a = %.0f, ℛ = ℛ* = %.3f\n", amp, ℛs); flush(stdout)
        cp = cell(B, amp, ℛs, R; pilot = :projection)
        cs = cell(B, amp, ℛs, R; pilot = :penalised)
        row("pilot = :projection", cp); row("pilot = :penalised", cs)
        @printf("      half-length, smoother pilot / projection pilot: %.3f\n", cs.hl / cp.hl)
        flush(stdout)
        0.93 <= cs.cov <= 0.99 || (pass = false)
    end
    println("\nPass criteria (1) and (2): ", pass ? "PASS" : "FAIL")
end

main(length(ARGS) > 0 ? parse(Int, ARGS[1]) : 300)
