# A placebo on the 2021 Dutch districts: does root_rho's ρ̂ drift with λ when the truth has a fixed ρ₀?
#
# On the real data (tc_netherlands_income.jl, THEORY.md "A fourth real-data run"), the income design's ρ̂
# at k = 24 falls 0.490, 0.436, 0.381, 0.339 as λ goes 100, 10, 1, 0.1, with no plateau. Two readings:
# (i) a rough trend that the stiff fits leave in the error, where it loads on Wy (smoothing bias, which
# the flexible fits remove), or (ii) the method itself pulls ρ̂ down as the surface gains df, because a
# flexible surface and the lag cannot be told apart at short range. The placebo draws data with a known
# ρ₀ and a known surface on the real geometry and fits the same λ path.
#
# Design, fixed before any run:
#   geometry, W (queen), X: the income design of tc_netherlands_income.jl (n = 3,027); k = 24.
#   truth: y* = (I − ρ₀W)⁻¹(Xβ̂ + f̂ + ε*), where (β̂, f̂) is the penalised fit of the real y at λ_true
#     and ρ₀ (Xβ̂ + f̂ = Z ψ, ψ = (Z′Z + λ_true Ω̃)⁻¹Z′(y − ρ₀Wy)), and ε* = c·eᵢ·vᵢ: a wild bootstrap
#     of that fit's residuals e with Rademacher v (keeps the spatially clustered variance), c =
#     √(n/(n − df)).
#   cells:
#     A  λ_true = 100, ρ₀ = 0.490 (the λ = 100 fit): a SMOOTH truth. Tests reading (ii).
#     B  λ_true = 0.1, ρ₀ = 0.339 (the λ = 0.1 fit): a ROUGH truth. Does it reproduce the data path?
#     C  λ_true = 1,   ρ₀ = 0:  no lag at all. Does the flexible end find ρ̂ ≈ 0, and does CS cover 0?
#   fits per draw: root_rho(y*, X, W; tensor_pspline(u, v; k = 24)..., lambda = λ), λ = 100, 10, 1, 0.1.
#   R = 100 draws per cell (sd of ρ̂ ≈ 0.03 on the data, so the Monte Carlo se of a mean is ≈ 0.003).
#
# Decision rules, fixed before any run (Δ = mean ρ̂ at λ = 100 minus mean ρ̂ at λ = 0.1):
#   A: |Δ| < 0.03 ⇒ the method does not make a drift under a smooth truth: reading (ii) is rejected.
#      Δ ≥ 0.075 (half the data's 0.151) ⇒ the method makes much of the data's drift.
#   B: if the placebo means at the four λ are each within 0.03 of the data path, the data are
#      consistent with reading (i): a fixed ρ ≈ 0.34 under a rough trend.
#   C: report mean ρ̂ and CS coverage of ρ₀ = 0 at each λ; no pass mark (descriptive).
#   CS coverage of ρ₀ at each λ in every cell: reported; the nominal level is 0.95.
#
# Data credit: Bron: Centraal Bureau voor de Statistiek (CBS), Kerncijfers wijken en buurten 2021
# (StatLine 85039NED) and Wijk- en Buurtkaart 2021 versie 3, CC BY 4.0.
#
# Run: julia --project=. research/trace_cfe/tc_netherlands_placebo.jl > research/trace_cfe/tc_netherlands_placebo_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf, Random

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")
splitq(l) = [strip(s, '"') for s in split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")]
function readcsv(f)
    ls = filter(!isempty, strip.(readlines(joinpath(DATA, f))))
    h = splitq(ls[1]); rs = splitq.(ls[2:end])
    Dict(h[j] => [r[j] for r in rs] for j in eachindex(h))
end

d = readcsv("netherlands_districts_2021.csv"); c = readcsv("netherlands_districts_2021_centroids.csv")
c["wijkcode"] == d["wijkcode"] || error("centroids are not in the CSV's order")
num(k) = parse.(Float64, d[k])
n = length(d["wijkcode"])
u = parse.(Float64, c["x_m"]) ./ 1000; v = parse.(Float64, c["y_m"]) ./ 1000
W, wid = read_gal(joinpath(DATA, "netherlands_districts_2021_queen.gal"))
wid == d["wijkcode"] || error("GAL order differs from the CSV")
y = log.(num("income"))
X = hcat(log.(num("density")), num("owner_occupied"), num("non_western"), num("aged_65_plus"), num("under_15"))
p = size(X, 2)

const LAMBDAS = (100.0, 10.0, 1.0, 0.1)
const DATA_PATH = (0.490, 0.436, 0.381, 0.339)       # tc_netherlands_income_output.txt, k = 24
const R = 100
ps = tensor_pspline(u, v; k = 24)
Z = hcat(X, ps.basis); K = size(Z, 2)
Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty

covers(cs, r) = any(a <= r <= b for (a, b) in cs)

function truth(λt, ρ0)
    G = Symmetric(Z' * Z + λt * Ωt)
    y0 = y .- ρ0 .* (W * y)
    ψ = G \ (Z' * y0)
    μ = Z * ψ
    df = tr(G \ (Z' * Z))
    e = y0 .- μ
    return μ, e .* sqrt(n / (n - df)), df
end

println("Placebo on the 2021 Dutch districts, income design, k = 24, n = $n, R = $R per cell")
let r = root_rho(y, X, W; ps..., lambda = 1.0), (_, _, df) = truth(1.0, r.rho)
    @printf("check: tr S at λ = 1 by hand %.2f, root_rho's df %.2f\n\n", df, r.df)
    flush(stdout)
end

for (name, λt, ρ0) in (("A (smooth truth)", 100.0, 0.490), ("B (rough truth)", 0.1, 0.339), ("C (no lag)", 1.0, 0.0))
    μ, e, dft = truth(λt, ρ0)
    F = lu(sparse(I - ρ0 * W))
    rng = MersenneTwister(2026_11_26 + round(Int, 1000λt))
    est = zeros(R, length(LAMBDAS)); cov = zeros(Bool, R, length(LAMBDAS))
    t = @elapsed for rep in 1:R
        ys = F \ (μ .+ e .* rand(rng, (-1.0, 1.0), n))
        for (j, λ) in enumerate(LAMBDAS)
            r = root_rho(ys, X, W; ps..., lambda = λ)
            est[rep, j] = r.rho; cov[rep, j] = covers(r.cs, ρ0)
        end
    end
    @printf("cell %s: λ_true = %g (df %.1f), ρ₀ = %.3f, residual sd %.4f  (%.0fs)\n", name, λt, dft, ρ0, std(e), t)
    for (j, λ) in enumerate(LAMBDAS)
        @printf("  λ = %-5g mean ρ̂ %.3f (MC se %.4f)  sd %.3f  bias %+.3f  CS covers ρ₀ %.2f   data path %.3f\n",
                λ, mean(est[:, j]), std(est[:, j]) / sqrt(R), std(est[:, j]), mean(est[:, j]) - ρ0,
                mean(cov[:, j]), DATA_PATH[j])
    end
    Δ = est[:, 1] .- est[:, end]
    @printf("  Δ = mean ρ̂(100) − mean ρ̂(0.1) = %.3f (MC se %.4f); data Δ = %.3f\n\n", mean(Δ),
            std(Δ) / sqrt(R), DATA_PATH[1] - DATA_PATH[end])
    flush(stdout)
end
