# A placebo on the 2,882 Portuguese parishes: what does root_rho report when there is no lag but the
# errors are spatially correlated? (For notebook 11, section 8.)
#
# On the Dutch districts, SAR errors with no lag passed for a lag (THEORY.md, "The second placebo":
# λ_e = 0.5 gave ρ̂ 0.397 and the CS never covered 0). Notebook 11, section 7, reports ρ̂ 0.43–0.51 on
# the parishes with a flexible surface. This run measures the same placebo on the parish map, with the
# notebook's design, before the notebook says anything about it.
#
# Design, fixed before any run. Notebook 11's parish design: y = log income per taxpayer 2021,
# X = log density, unemployment, illiteracy; W queen; tensor_pspline(u, v; k = 24); λ = 100, 10, 1, 0.1.
# The truth's mean μ = Zψ is the penalised fit of the real y at λ_true and ρ₀; e its residuals ×
# √(n/(n − df)); v iid Rademacher.
#   N2, N5, N7: no lag (ρ₀ = 0, λ_true = 1); errors (I − λ_eW)⁻¹(e∘v), λ_e = 0.2, 0.5, 0.7.
#   L:          a lag and independent errors: ρ₀ = 0.43 (the λ = 0.1 fit on the data), λ_true = 0.1,
#               y* = (I − ρ₀W)⁻¹(μ + e∘v).
# R = 100 draws per cell. Reported: mean ρ̂ and its sd at each λ, and the CS coverage of ρ₀.
#
# What the notebook will say, by rule fixed before the run (at λ = 0.1):
#   - for each N cell: if the mean ρ̂ is at least 0.2 and the CS covers 0 in fewer than half the draws,
#     the notebook says that correlated errors of that strength pass for a lag on this map; otherwise
#     it reports the measured values without that claim.
#   - the notebook quotes the smallest λ_e whose mean ρ̂ reaches the data's 0.43, if one does; if none
#     does, it says so.
#   - L is the check that the placebo can recover a real lag: if its mean ρ̂ is within 0.05 of 0.43, the
#     notebook uses this placebo; if not, the notebook does not use it, and this is recorded.
#
# Data credit (a condition of the licences): Source: Statistics Portugal (INE), indicators 0011866,
# 0012273, 0012295, 0012328, 0012748, reference period 2021, extracted 2026-11-25. Informação
# geográfica cedida pela Direção-Geral do Território (CAOP, CC-BY-4.0).
#
# Run: julia --project=. research/trace_cfe/tc_portugal_parishes_placebo.jl > research/trace_cfe/tc_portugal_parishes_placebo_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf, Random

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")
splitq(l) = [replace(s, '"' => "") for s in split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")]
plines = [l for l in readlines(joinpath(DATA, "portugal_parishes.csv")) if !isempty(strip(l))]
phdr = splitq(plines[1]); prows = splitq.(plines[2:end])
pcol(name) = [r[findfirst(==(name), phdr)] for r in prows]
pnum(name) = parse.(Float64, pcol(name))
dicofre = pcol("dicofre")
y = log.(pnum("income_2021"))
X = hcat(log.(pnum("population") ./ pnum("area_km2")), pnum("unemployment"), pnum("illiteracy"))
W, pid = read_gal(joinpath(DATA, "portugal_parishes_queen.gal"))
pc = [splitq(l) for l in readlines(joinpath(DATA, "portugal_parishes_centroids.csv"))[2:end] if !isempty(strip(l))]
pid == dicofre && [r[1] for r in pc] == dicofre || error("parish files are out of order")
u = parse.(Float64, [r[3] for r in pc]) ./ 1000; u .-= mean(u)
v = parse.(Float64, [r[4] for r in pc]) ./ 1000; v .-= mean(v)
n = length(y); p = size(X, 2)

const LAMBDAS = (100.0, 10.0, 1.0, 0.1)
const R = parse(Int, get(ENV, "PP_R", "100"))   # a smaller R only for a smoke test
ps = tensor_pspline(u, v; k = 24)
Z = hcat(X, ps.basis); K = size(Z, 2)
Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty

function truth(λt, ρ0)
    G = Symmetric(Z' * Z + λt * Ωt)
    y0 = y .- ρ0 .* (W * y)
    μ = Z * (G \ (Z' * y0))
    df = tr(G \ (Z' * Z))
    return μ, (y0 .- μ) .* sqrt(n / (n - df)), df
end
covers(cs, r) = any(a <= r <= b for (a, b) in cs)

println("n = $n parishes, k = 24, R = $R per cell")
let r = root_rho(y, X, W; ps..., lambda = 0.1)
    @printf("data: root_rho at λ = 0.1: ρ̂ %.3f (df %.1f)\n\n", r.rho, r.df); flush(stdout)
end

for (name, λt, ρ0, λe) in (("N2", 1.0, 0.0, 0.2), ("N5", 1.0, 0.0, 0.5), ("N7", 1.0, 0.0, 0.7),
                           ("L", 0.1, 0.43, 0.0))
    μ, e, dft = truth(λt, ρ0)
    Fρ = lu(sparse(I - ρ0 * W)); Fe = lu(sparse(I - λe * W))
    rng = MersenneTwister(2026_11_26 + hash(name) % 10_000)
    est = zeros(R, length(LAMBDAS)); cov = falses(R, length(LAMBDAS))
    t = @elapsed for rep in 1:R
        ys = Fρ \ (μ .+ Fe \ (e .* rand(rng, (-1.0, 1.0), n)))
        for (j, λ) in enumerate(LAMBDAS)
            r = root_rho(ys, X, W; ps..., lambda = λ)
            est[rep, j] = r.rho; cov[rep, j] = covers(r.cs, ρ0)
        end
    end
    @printf("cell %s: λ_true = %g (df %.1f), ρ₀ = %.2f, λ_e = %.1f  (%.0fs)\n", name, λt, dft, ρ0, λe, t)
    for (j, λ) in enumerate(LAMBDAS)
        @printf("  λ = %-5g mean ρ̂ %.3f (MC se %.4f)  sd %.3f  CS covers ρ₀ %.2f\n", λ, mean(est[:, j]),
                std(est[:, j]) / sqrt(R), std(est[:, j]), mean(cov[:, j]))
    end
    flush(stdout)
end
