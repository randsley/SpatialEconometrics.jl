# The sign rule assumes J = −E g′(ρ₀) > 0 (THEORY.md, A9). This script
# (1) searches 3,000 hub-and-spoke designs, built from formulas, for J < 0,
#     with the exact J of the first-step moment C = (W′ − Δ)M;
# (2) on the strongest one, measures the penalised path (one-step moment
#     A = MW + MDM, penalty 0): the sign-rule root, the other root, the Wald
#     coverage, how often root_rho's check warns, and CS coverage; and the
#     projection path (REsd) on the same draws;
# (3) counts false warnings on kNN designs with J > 0 (both paths).
#
# Run: julia --project=. research/trace_cfe/tc_sign.jl > research/trace_cfe/tc_sign_output.txt

using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Test, Logging
using Printf
const RR = SpatialEconometrics.RootRho
println("Julia ", VERSION)
nwarn(f) = (l = Test.TestLogger(min_level = Logging.Warn); r = with_logger(f, l);
            (r, count(x -> occursin("slope g′", string(x.message)), l.logs)))

function hub_design(n, nh, tg, mult, ω, coef)
    W = zeros(n, n)
    for h in 1:nh; W[h, tg[h]] = 1; end
    for i in nh+1:n; W[i, mod1(mult * i, nh)] = 1; end
    x = sin.(ω .* (1:n)); Z = hcat(ones(n), x, x .^ 2)
    (sparse(W), Z, Z * coef)
end
function J_first_step(W, Z, f, ρ0)
    n = size(W, 1); K = size(Z, 2); Q = Matrix(qr(Z).Q)[:, 1:K]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    C = RR._moment_matrix(Matrix(W'), Q, Mii); Cs = (C + C') / 2
    G = Matrix(W) / (I - ρ0 * Matrix(W))
    (2dot(f, G' * Cs * f) / n, 2tr(G' * Cs) / n)
end

# (1) The search.
rng = MersenneTwister(11); n = 100; res = []
for trial in 1:3000
    nh = rand(rng, 3:7); tg = [rand(rng, setdiff(1:n, h)) for h in 1:nh]
    mult = rand(rng, 1:7); ω = round(rand(rng) * 3 + 0.2, digits = 2)
    coef = round.(randn(rng, 3) .* 3, digits = 1); ρ0 = rand(rng, (0.6, 0.9, -0.6, -0.9))
    W, Z, f = hub_design(n, nh, tg, mult, ω, coef)
    rank(Z) == 3 || continue
    s, t = J_first_step(W, Z, f, ρ0)
    push!(res, (s + t, (; nh, tg, mult, ω, coef, ρ0, s, t)))
end
sort!(res, by = first)
println("\n(1) $(length(res)) hub designs, n = 100, unit error variance: share with J < 0 = ",
        round(mean(first.(res) .< 0), digits = 4), " ($(count(first.(res) .< 0)) designs)")
println("Most negative J/n (signal + trace):")
for r in res[1:3]; @printf("  J/n = %7.2f  %s\n", r[1], r[2]); end

# (2) The strongest: the design of testset 47g.
best = res[1][2]
W, Z, f = hub_design(n, best.nh, best.tg, best.mult, best.ω, best.coef); ρ0 = best.ρ0
Q = Matrix(qr(Z).Q)[:, 1:3]; A = RR._exact_diag_moment(Matrix(I - Q * Q'), W); As = (A + A') / 2
G = Matrix(W) / (I - ρ0 * Matrix(W))
@printf("\n(2) The strongest design. Exact J/n for the penalised moment A: %.2f\n",
        2 * (dot(f, G' * As * f) + tr(G' * As)) / n)
rng = MersenneTwister(1); R = 300
pk = Float64[]; ot = Float64[]; cov = 0; warn_q = 0; cs_q = 0
pj = Float64[]; covj = 0; warn_p = 0; cs_p = 0
for r in 1:R
    y = (I - ρ0 * Matrix(W)) \ (f .+ randn(rng, n))
    fq, w = nwarn(() -> root_rho(y, Z[:, 2:3], W; basis = Z[:, 1:1], penalty = zeros(1, 1), lambda = 1.0))
    global warn_q += w > 0; global cs_q += any(lo <= ρ0 <= hi for (lo, hi) in fq.cs)
    D = fq.b^2 - 4fq.a * fq.c
    D > 0 && push!(ot, (fq.b + sqrt(D)) / (2fq.a) - ρ0)
    if isfinite(fq.rho)
        push!(pk, fq.rho - ρ0); global cov += fq.ci_wald[1] <= ρ0 <= fq.ci_wald[2]
    end
    fp, w = nwarn(() -> root_rho(y, Z[:, 2:3], W; basis = Z[:, 1:1]))
    global warn_p += w > 0; global cs_p += any(lo <= ρ0 <= hi for (lo, hi) in fp.cs)
    if isfinite(fp.rho)
        push!(pj, fp.rho - ρ0); global covj += fp.ci_wald[1] <= ρ0 <= fp.ci_wald[2]
    end
end
println("R = $R draws.")
@printf("  penalised: sign-rule root found %d; bias %+.4f, sd %.4f; Wald covers %d/%d\n",
        length(pk), mean(pk), std(pk), cov, length(pk))
@printf("             other root: bias %+.4f, sd %.4f\n", mean(ot), std(ot))
@printf("             warned %d/%d; CS covers %d/%d\n", warn_q, R, cs_q, R)
@printf("  REsd:      root found %d; bias %+.4f, sd %.4f; Wald covers %d/%d; warned %d/%d; CS covers %d/%d\n",
        length(pj), mean(pj), std(pj), covj, length(pj), warn_p, R, cs_p, R)

# (3) False warnings where J > 0.
println("\n(3) False warnings on kNN designs (testset 47's jittered grid, ρ₀ = 0.4, heteroskedastic), R = 100:")
for (kw, g, k, λ, amp) in ((6, 20, 10, 0.01, 1.0), (20, 20, 10, 0.01, 1.0), (20, 20, 10, 0.01, 4.0),
                           (6, 30, 15, 0.1, 4.0), (20, 20, 10, 1.0, 4.0))
    rng = MersenneTwister(7); n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    W = knn_weights(P, kw); ps = tensor_pspline(P[:, 1], P[:, 2]; k); x = randn(rng, n)
    f = 0.5 .* x .+ amp .* sin.(2π .* P[:, 1]) .* cos.(π .* P[:, 2])
    σ2 = exp.(3 .* (P[:, 1] .+ P[:, 2] .- 1)); σ2 ./= mean(σ2)
    wp = 0; wq = 0
    for r in 1:100
        y = lu(sparse(I - 0.4 * W)) \ (f .+ sqrt.(σ2) .* randn(rng, n))
        _, w = nwarn(() -> root_rho(y, reshape(x, :, 1), W; basis = ps.basis)); wp += w > 0
        _, w = nwarn(() -> root_rho(y, reshape(x, :, 1), W; basis = ps.basis, penalty = ps.penalty, lambda = λ)); wq += w > 0
    end
    println("  kNN-$kw, n = $n, k = $k, λ = $λ, a = $amp: projection $wp/100, smoother $wq/100")
end
