# The second placebo on the 2021 Dutch districts: does the bias-aware interval cover, under a truth
# rougher than any fit, and under spatially correlated errors with no lag?
#
# The first placebo (tc_netherlands_placebo.jl, THEORY.md "The placebo on the Dutch districts") showed
# that ρ̂'s drift with λ is smoothing bias, and recommended reporting the bias-aware interval. It did not
# test that interval, a truth outside the fits' basis, or correlated errors. This run does.
#
# Design, fixed before any run. Income design of tc_netherlands_income.jl (n = 3,027, queen W, five X).
# The fits always use tensor_pspline(u, v; k = 24). The truth's mean is μ = Z_t ψ, the penalised fit
# of the real y at (λ_true, ρ₀) on a basis Z_t; e are that fit's residuals × √(n/(n − df)); v are iid
# Rademacher.
#   B  rough truth:      Z_t = k 24, λ_true = 0.1, ρ₀ = 0.339;  y* = (I − ρ₀W)⁻¹(μ + e∘v)
#   C  no lag:           Z_t = k 24, λ_true = 1,   ρ₀ = 0;      y* = μ + e∘v
#   D  rougher than any fit: Z_t = k 32, λ_true = 0.1, ρ₀ = 0.30; y* = (I − ρ₀W)⁻¹(μ + e∘v)
#   E2, E5  no lag, spatially correlated errors (SEM): Z_t = k 24, λ_true = 1, ρ₀ = 0;
#        y* = μ + (I − λ_e W)⁻¹(e∘v), λ_e = 0.2 (E2) and 0.5 (E5)
# Per draw: root_rho at λ = 100, 10, 1, 0.1 (ρ̂, CS); ℛ̂ = the roughness ‖ψ‖_Ω̃ of the λ = 1 fit at its ρ̂
# (as on the real data); the bias-aware root_rho with lambda = [0.1, 1, 10, 100] and roughness = m·ℛ̂,
# m = 1 and 4. R = 150 draws per cell (MC se of a coverage near 0.95: 0.018).
#
# Decision rules, fixed before any run:
#   1. In B, C and D, the bias-aware interval at m = 4 covers ρ₀ in at least 0.91 of draws (0.95 less
#      two MC se) ⇒ the reporting advice of the first placebo holds on this geometry. Below 0.91 in any
#      of them ⇒ it does not, for that truth. m = 1: reported, no pass mark.
#   2. D: the bias of ρ̂ at λ = 0.1 is reported, no pass mark.
#   3. E2 and E5: mean ρ̂ at λ = 0.1 below 0.05 AND bias-aware coverage of 0 at m = 4 at least 0.91 ⇒
#      correlated errors of that strength do not pass for a lag. Mean ρ̂ above 0.10 OR coverage below
#      0.91 ⇒ they can, and the real-data lag must be qualified. In between: reported as unclear.
#
# Data credit: Bron: Centraal Bureau voor de Statistiek (CBS), Kerncijfers wijken en buurten 2021
# (StatLine 85039NED) and Wijk- en Buurtkaart 2021 versie 3, CC BY 4.0.
#
# Run one cell per process (they are independent):
#   julia --project=. research/trace_cfe/tc_netherlands_placebo2.jl B > research/trace_cfe/tc_netherlands_placebo2_B.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf, Random

const CELL = only(ARGS)
BLAS.set_num_threads(3)
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
const MULTS = (1, 4)
const R = parse(Int, get(ENV, "PLACEBO_R", "150"))   # a smaller R only for a smoke test

function design(k)
    ps = tensor_pspline(u, v; k)
    Z = hcat(X, ps.basis); K = size(Z, 2)
    Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty
    return ps, Z, Ωt
end

function truth(Z, Ωt, λt, ρ0)
    G = Symmetric(Z' * Z + λt * Ωt)
    y0 = y .- ρ0 .* (W * y)
    μ = Z * (G \ (Z' * y0))
    df = tr(G \ (Z' * Z))
    return μ, (y0 .- μ) .* sqrt(n / (n - df)), df
end

covers(cs, r) = any(a <= r <= b for (a, b) in cs)

const CELLS = Dict(  # (truth k, λ_true, ρ₀, λ_e)
    "B" => (24, 0.1, 0.339, 0.0), "C" => (24, 1.0, 0.0, 0.0), "D" => (32, 0.1, 0.30, 0.0),
    "E2" => (24, 1.0, 0.0, 0.2), "E5" => (24, 1.0, 0.0, 0.5))
kt, λt, ρ0, λe = CELLS[CELL]

ps, Z, Ωt = design(24)
_, Zt, Ωtt = design(kt)
μ, e, dft = truth(Zt, Ωtt, λt, ρ0)
Fρ = lu(sparse(I - ρ0 * W)); Fe = lu(sparse(I - λe * W))
@printf("cell %s: truth k = %d, λ_true = %g (df %.1f), ρ₀ = %.3f, λ_e = %.1f, n = %d, R = %d; fits k = 24\n",
        CELL, kt, λt, dft, ρ0, λe, n, R)
flush(stdout)

rng = MersenneTwister(2026_11_27 + hash(CELL) % 10_000)
est = zeros(R, length(LAMBDAS)); cov = falses(R, length(LAMBDAS))
bcov = falses(R, length(MULTS)); blen = zeros(R, length(MULTS)); brho = zeros(R, length(MULTS))
blam = zeros(R, length(MULTS)); Rhat = zeros(R)
tstart = time()
t = @elapsed for rep in 1:R
    ε = Fe \ (e .* rand(rng, (-1.0, 1.0), n))
    ys = Fρ \ (μ .+ ε)
    local r1
    for (j, λ) in enumerate(LAMBDAS)
        r = root_rho(ys, X, W; ps..., lambda = λ)
        est[rep, j] = r.rho; cov[rep, j] = covers(r.cs, ρ0)
        λ == 1.0 && (r1 = r)
    end
    ψ = (Z' * Z + Ωt) \ (Z' * (ys .- r1.rho .* (W * ys)))
    Rhat[rep] = sqrt(dot(ψ, Ωt * ψ))
    for (j, m) in enumerate(MULTS)
        rb = root_rho(ys, X, W; ps..., lambda = [0.1, 1.0, 10.0, 100.0], roughness = m * Rhat[rep])
        a, b = rb.ci_bias_aware
        bcov[rep, j] = a <= ρ0 <= b; blen[rep, j] = b - a; brho[rep, j] = rb.rho; blam[rep, j] = rb.lambda
    end
    if rep % 25 == 0
        @printf("  ... %d draws, %.0fs\n", rep, time() - tstart); flush(stdout)
    end
end
@printf("done in %.0fs; ℛ̂ over draws: mean %.3f, sd %.3f\n", t, mean(Rhat), std(Rhat))
for (j, λ) in enumerate(LAMBDAS)
    @printf("  λ = %-5g mean ρ̂ %.3f (MC se %.4f)  sd %.3f  bias %+.3f  CS covers ρ₀ %.3f\n",
            λ, mean(est[:, j]), std(est[:, j]) / sqrt(R), std(est[:, j]), mean(est[:, j]) - ρ0, mean(cov[:, j]))
end
for (j, m) in enumerate(MULTS)
    @printf("  bias-aware, %d·ℛ̂: covers ρ₀ %.3f (MC se %.3f)  mean length %.3f  mean ρ̂ %.3f  λ chosen %s\n",
            m, mean(bcov[:, j]), sqrt(mean(bcov[:, j]) * (1 - mean(bcov[:, j])) / R), mean(blen[:, j]),
            mean(brho[:, j]), join(["$(l)×$(count(==(l), blam[:, j]))" for l in sort(unique(blam[:, j]))], ", "))
end
