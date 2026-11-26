# A spatial-block wild bootstrap for the IV check of tc_iv_check.jl: do its intervals cover?
#
# tc_iv_check.jl (THEORY.md, "An instrumental-variable check") found that ρ̂_IV with instruments
# M[WX, W²X] is not fooled by SAR errors, but that its Conley-HAC intervals undercover (0.71–0.92, also
# with independent errors), so the real-data reading was not made. This run replaces the HAC with a
# wild restricted efficient (WRE) cluster bootstrap (Davidson & MacKinnon, 2010) on spatial blocks.
#
# Setting: the residualised data of tc_iv_check.jl, ỹ = My, w̃ = MWy, H̃ = M[WX, W²X] (M = I − S, S the
# penalised smoother at k, λ), treated as the data: ỹ = ρw̃ + u, w̃ = H̃π + v. (The bootstrap does not
# re-apply M; M is fixed, so the bootstrap draws stay in its range only approximately.)
# Blocks: the squares of a grid of side L km over the centroids. The t statistic uses the cluster-robust
# variance on the same blocks: V = Σ_b (Σ_{i∈b} ŵᵢ eᵢ)² / (ŵ′w̃)².
# WRE test of ρ = r: ũ = ỹ − rw̃; regress w̃ on [H̃, ũ], keep π̃ (the H̃ coefficients) and v̈ = w̃ − H̃π̃;
# for b = 1..B, one Rademacher sign s per block: u* = ũ∘s, w* = H̃π̃ + v̈∘s, y* = rw* + u*; t* from the
# same IV and variance. p = share of |t*| ≥ |t|. The interval is {r : p > 0.05}, on a grid.
#
# Part 1, placebo: the truths of tc_iv_check.jl, cells B, C, D, E2, E5; fit k = 24, λ = 0.1; R = 500,
# B = 399; L = 40 km (primary) and 20 km. Reported per cell: WRE coverage of ρ₀ (p > 0.05 at r = ρ₀),
# and the analytic cluster-robust t interval's coverage.
# Part 2, real data: k = 24 and 16, λ = 0.1; B = 999; r on a grid of step 0.005 over [−0.7, 0.9].
#
# Decision rules, fixed before any run:
#   1. WRE at L = 40 km covers ρ₀ in at least 0.93 of draws in each of B, C, E2 and E5 ⇒ its intervals
#      are usable on this geometry. D is left out in advance: ρ̂_IV there is biased by −0.075 against an
#      sd of 0.065 (tc_iv_check.jl), and no interval centred on it can cover; it is reported.
#      L = 20 km and the analytic intervals: reported, no pass mark.
#   2. Only if rule 1 passes, the real-data reading at L = 40 km, λ = 0.1, for each k:
#      the interval excludes root_rho's ρ̂ (0.339 at k = 24, 0.380 at k = 16) ⇒ part of that ρ̂ is not
#      a lag (error correlation, or bias of the kind seen in D, at most about 0.075 there); it includes
#      0 ⇒ the data are consistent with no lag; it excludes 0 ⇒ evidence of a lag that error
#      correlation does not explain.
#
# Data credit: Bron: Centraal Bureau voor de Statistiek (CBS), Kerncijfers wijken en buurten 2021
# (StatLine 85039NED) and Wijk- en Buurtkaart 2021 versie 3, CC BY 4.0.
#
# Run: julia --project=. research/trace_cfe/tc_iv_bootstrap.jl > research/trace_cfe/tc_iv_bootstrap_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf, Random

BLAS.set_num_threads(8)
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
H = hcat(W * X, W * (W * X)); q = size(H, 2)

# Grid blocks: block ids 1..G for the occupied squares of side L.
function blocks(L)
    key = [(floor(Int, (u[i] - minimum(u)) / L), floor(Int, (v[i] - minimum(v)) / L)) for i in 1:n]
    ids = Dict(k => j for (j, k) in enumerate(unique(key)))
    return [ids[k] for k in key], length(ids)
end
const LS = (40.0, 20.0)
const BLK = Dict(L => blocks(L) for L in LS)
for L in LS
    b, G = BLK[L]
    sz = [count(==(g), b) for g in 1:G]
    @printf("L = %g km: %d blocks, units per block %d–%d (median %.0f)\n", L, G, extrema(sz)..., median(sz))
end

function design(k)
    ps = tensor_pspline(u, v; k)
    Z = hcat(X, ps.basis); K = size(Z, 2)
    Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty
    return Z, Ωt
end
smoother(Z, Ωt, λ) = Z * (Symmetric(Z' * Z + λ * Ωt) \ Z')

# The IV in residualised form, with QH an orthonormal basis of H̃; returns ρ and the cluster-robust se.
function iv(yt, wt, QH, b, G)
    what = QH * (QH' * wt)
    den = dot(what, wt)
    ρ = dot(what, yt) / den
    sums = zeros(G)
    @inbounds for i in eachindex(yt)
        sums[b[i]] += what[i] * (yt[i] - ρ * wt[i])
    end
    return ρ, sqrt(sum(abs2, sums)) / abs(den)
end

# WRE bootstrap p-value for ρ = r.
function wre_p(yt, wt, MH, QH, b, G, r, B, rng)
    ρ, se = iv(yt, wt, QH, b, G)
    t = (ρ - r) / se
    ut = yt .- r .* wt
    coefs = hcat(MH, ut) \ wt
    fit = MH * coefs[1:q]
    vdd = wt .- fit
    exceed = 0
    s = zeros(G); us = similar(ut); ws = similar(wt); ys = similar(yt)
    for _ in 1:B
        for g in 1:G
            s[g] = rand(rng, Bool) ? 1.0 : -1.0
        end
        @inbounds for i in eachindex(ut)
            si = s[b[i]]
            us[i] = ut[i] * si; ws[i] = fit[i] + vdd[i] * si; ys[i] = r * ws[i] + us[i]
        end
        ρs, ses = iv(ys, ws, QH, b, G)
        abs((ρs - r) / ses) >= abs(t) && (exceed += 1)
    end
    return exceed / B, ρ, se
end

function truth(Zt, Ωtt, λt, ρ0)
    G = Symmetric(Zt' * Zt + λt * Ωtt)
    y0 = y .- ρ0 .* (W * y)
    μ = Zt * (G \ (Zt' * y0))
    df = tr(G \ (Zt' * Zt))
    return μ, (y0 .- μ) .* sqrt(n / (n - df)), df
end

const R = parse(Int, get(ENV, "IVB_R", "500"))   # a smaller R only for a smoke test
const B1 = 399
const SEED = parse(Int, get(ENV, "IVB_SEED", "0"))   # 0 reproduces the exploratory run (f2a834d)
z = 1.959963984540054
Z24, Ω24 = design(24); Z32, Ω32 = design(32)
S = smoother(Z24, Ω24, 0.1)
MH = H .- S * H; QH = Matrix(qr(MH).Q)[:, 1:q]

println("\nPART 1: placebo, fit k = 24, λ = 0.1, R = $R, B = $B1")
cells = (("B", Z24, Ω24, 0.1, 0.339, 0.0), ("C", Z24, Ω24, 1.0, 0.0, 0.0), ("D", Z32, Ω32, 0.1, 0.30, 0.0),
         ("E2", Z24, Ω24, 1.0, 0.0, 0.2), ("E5", Z24, Ω24, 1.0, 0.0, 0.5))
for (name, Zt, Ωtt, λt, ρ0, λe) in cells
    μ, e, _ = truth(Zt, Ωtt, λt, ρ0)
    Fρ = lu(sparse(I - ρ0 * W)); Fe = lu(sparse(I - λe * W))
    rng = MersenneTwister(2026_11_26 + hash(name) % 10_000 + SEED)
    rngb = MersenneTwister(4242 + hash(name) % 10_000 + SEED)
    wre = zeros(Bool, R, length(LS)); ana = zeros(Bool, R, length(LS)); est = zeros(R)
    t = @elapsed for rep in 1:R
        ys = Fρ \ (μ .+ Fe \ (e .* rand(rng, (-1.0, 1.0), n)))
        Wys = W * ys
        yt = ys .- S * ys; wt = Wys .- S * Wys
        for (l, L) in enumerate(LS)
            b, G = BLK[L]
            pval, ρ, se = wre_p(yt, wt, MH, QH, b, G, ρ0, B1, rngb)
            wre[rep, l] = pval > 0.05; ana[rep, l] = abs(ρ - ρ0) <= z * se; est[rep] = ρ
        end
    end
    @printf("cell %s: ρ₀ = %.3f, λ_e = %.1f; mean ρ̂_IV %+.3f, sd %.3f  (%.0fs)\n", name, ρ0, λe, mean(est), std(est), t)
    for (l, L) in enumerate(LS)
        cw = mean(wre[:, l]); ca = mean(ana[:, l])
        @printf("  L = %2g km: WRE covers ρ₀ %.3f (MC se %.3f)   analytic cluster-robust %.3f\n",
                L, cw, sqrt(cw * (1 - cw) / R), ca)
    end
    flush(stdout)
end

println("\nPART 2: the real data, λ = 0.1, B = 999")
const GRID = -0.7:0.005:0.9
for (k, ρrr) in ((24, 0.339), (16, 0.380))
    Z, Ω = design(k)
    Sk = smoother(Z, Ω, 0.1)
    MHk = H .- Sk * H; QHk = Matrix(qr(MHk).Q)[:, 1:q]
    Wy = W * y; yt = y .- Sk * y; wt = Wy .- Sk * Wy
    for L in LS
        b, G = BLK[L]
        rng = MersenneTwister(999 + k + round(Int, L) + SEED)
        ps = [wre_p(yt, wt, MHk, QHk, b, G, r, 999, rng)[1] for r in GRID]
        ρ, se = iv(yt, wt, QHk, b, G)
        acc = GRID[ps .> 0.05]
        # Report the accepted set as intervals (it need not be one interval).
        parts = Tuple{Float64, Float64}[]
        for r in acc
            if !isempty(parts) && r - parts[end][2] <= 0.0051
                parts[end] = (parts[end][1], r)
            else
                push!(parts, (r, r))
            end
        end
        @printf("  k = %d, L = %2g km: ρ̂_IV %+.3f; WRE 95%% set %s; analytic [%+.3f, %+.3f]; p at 0: %.3f, at root_rho's %.3f: %.3f\n",
                k, L, ρ, join([@sprintf("[%+.3f, %+.3f]", a, z2) for (a, z2) in parts], " ∪ "),
                ρ - z * se, ρ + z * se, ps[findmin(abs.(GRID))[2]], ρrr, ps[findmin(abs.(GRID .- ρrr))[2]])
        isempty(acc) || (first(acc) == first(GRID) || last(acc) == last(GRID)) &&
            println("    (the set reaches the end of the grid)")
        flush(stdout)
    end
end
