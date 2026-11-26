# Can an instrumental-variable estimate of ρ tell a spatial lag from spatially correlated errors, with a
# flexible surface? (THEORY.md, "The second placebo": root_rho reads SAR errors as a lag.)
#
# A lag passes X to the neighbours (E[Wy | X] carries WX, W²X, ...); errors that follow a spatial
# autoregression do not. So instruments [WX, W²X] give an estimate of ρ that is consistent under SAR
# errors (the first step of Kelejian & Prucha, 1998). With a surface, the smooth part of WX is absorbed
# by the surface too, and what is left to instrument with is WX's local variation. How much is left is
# the open question.
#
# Estimator (written here, not in the package). S = Z(Z′Z + λΩ̃)⁻¹Z′, Z = [X, basis], the penalised
# smoother of root_rho (only the basis is penalised, so SX = X exactly); M = I − S. Residualise:
# ỹ = My, w̃ = MWy, H̃ = M[WX, W²X]. First stage ŵ = H̃(H̃′H̃)⁻¹H̃′w̃; ρ̂_IV = ŵ′ỹ / ŵ′w̃; e = ỹ − ρ̂_IV w̃.
# Variance: Conley (1999)-type spatial HAC, V = Σᵢⱼ k(dᵢⱼ) ŵᵢeᵢ ŵⱼeⱼ / (ŵ′w̃)², Bartlett kernel
# k(d) = max(0, 1 − d/c), centroid distances, c = 10 and 25 km; and HC0 (c = 0). First-stage strength:
# the robust (Conley c = 25 km) Wald statistic for π = 0 in w̃ = H̃π + v, divided by q = 10.
#
# Part 1, placebo: the truths of tc_netherlands_placebo2.jl (same construction: μ the penalised fit of
# the real y, wild Rademacher residuals), cells B (ρ₀ 0.339, rough), C (0, no lag), D (0.30, k = 32
# truth), E2 and E5 (no lag, errors (I − λ_eW)⁻¹η, λ_e 0.2 and 0.5). Fits at k = 24, λ = 100, 1, 0.1.
# R = 500 per cell. Part 2: the real data, k = 16 and 24, λ = 100, 10, 1, 0.1, next to root_rho.
#
# Decision rules, fixed before any run (all at k = 24, λ = 0.1):
#   1. E2, E5: |mean ρ̂_IV| < 0.05 ⇒ the IV separates error correlation from a lag on this geometry.
#   2. B, D: |mean ρ̂_IV − ρ₀| < 0.05 ⇒ it recovers a lag.
#   3. Conley c = 25 km interval covers ρ₀ in at least 0.93 of draws (0.95 less two MC se at R = 500)
#      in every cell ⇒ its intervals are usable here. c = 10 km and HC0: reported.
#   4. Real data: if the median first-stage F (robust, per instrument) in the placebo cells and the F on
#      the data are both ≥ 10, the instruments count as strong. Then a c = 25 km interval on the data
#      that excludes 0 is evidence of a lag that error correlation does not explain, and one that
#      includes 0 means the data are consistent with no lag. F < 10 ⇒ the check is uninformative.
#   If rule 1, 2 or 3 fails, rule 4's reading is not made.
#
# Data credit: Bron: Centraal Bureau voor de Statistiek (CBS), Kerncijfers wijken en buurten 2021
# (StatLine 85039NED) and Wijk- en Buurtkaart 2021 versie 3, CC BY 4.0.
#
# Run: julia --project=. research/trace_cfe/tc_iv_check.jl > research/trace_cfe/tc_iv_check_output.txt
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

# Bartlett kernel weights as sparse matrices (the diagonal is 1).
function kernel(cut)
    cut == 0 && return sparse(1.0I, n, n)
    I_, J_, V_ = Int[], Int[], Float64[]
    for j in 1:n, i in 1:n
        dij = hypot(u[i] - u[j], v[i] - v[j])
        dij < cut && (push!(I_, i); push!(J_, j); push!(V_, 1 - dij / cut))
    end
    return sparse(I_, J_, V_, n, n)
end
const CUTS = (0.0, 10.0, 25.0)
const KER = Dict(cut => kernel(cut) for cut in CUTS)
@printf("n = %d; kernel non-zeros per unit: %s\n", n,
        join([@sprintf("c = %g km: %.0f", cut, nnz(KER[cut]) / n) for cut in CUTS], ", "))

function design(k)
    ps = tensor_pspline(u, v; k)
    Z = hcat(X, ps.basis); K = size(Z, 2)
    Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty
    return ps, Z, Ωt
end
smoother(Z, Ωt, λ) = Z * (Symmetric(Z' * Z + λ * Ωt) \ Z')

struct IVFit
    rho::Float64; se::Dict{Float64, Float64}; F::Float64
end
function ivfit(S, MH, QH, ys)
    Wy = W * ys
    yt = ys .- S * ys; wt = Wy .- S * Wy
    what = QH * (QH' * wt)
    den = dot(what, wt)
    ρ = dot(what, yt) / den
    e = yt .- ρ .* wt
    ψ = what .* e
    se = Dict(cut => sqrt(max(dot(ψ, KER[cut] * ψ), 0.0)) / abs(den) for cut in CUTS)
    # First stage w̃ = H̃π + v: robust Wald for π = 0 with the c = 25 km kernel, per instrument.
    π = MH \ wt; vres = wt .- MH * π
    A = MH' * MH
    G = MH .* vres
    Bm = G' * (KER[25.0] * G)
    Vπ = A \ Bm / A
    F = dot(π, Symmetric(Vπ) \ π) / q
    return IVFit(ρ, se, F)
end
function prep(S)
    MH = H .- S * H
    QH = Matrix(qr(MH).Q)[:, 1:q]
    return MH, QH
end

ps24, Z24, Ω24 = design(24)
const LAMBDAS_P = (100.0, 1.0, 0.1)
S24 = Dict(λ => smoother(Z24, Ω24, λ) for λ in LAMBDAS_P)
P24 = Dict(λ => prep(S24[λ]) for λ in LAMBDAS_P)

function truth(Zt, Ωtt, λt, ρ0)
    G = Symmetric(Zt' * Zt + λt * Ωtt)
    y0 = y .- ρ0 .* (W * y)
    μ = Zt * (G \ (Zt' * y0))
    df = tr(G \ (Zt' * Zt))
    return μ, (y0 .- μ) .* sqrt(n / (n - df)), df
end

println("\nPART 1: placebo, fits at k = 24, R = 500 per cell")
const R = parse(Int, get(ENV, "IV_R", "500"))   # a smaller R only for a smoke test
_, Z32, Ω32 = design(32)
cells = (("B", Z24, Ω24, 0.1, 0.339, 0.0), ("C", Z24, Ω24, 1.0, 0.0, 0.0), ("D", Z32, Ω32, 0.1, 0.30, 0.0),
         ("E2", Z24, Ω24, 1.0, 0.0, 0.2), ("E5", Z24, Ω24, 1.0, 0.0, 0.5))
z = 1.959963984540054
for (name, Zt, Ωtt, λt, ρ0, λe) in cells
    μ, e, dft = truth(Zt, Ωtt, λt, ρ0)
    Fρ = lu(sparse(I - ρ0 * W)); Fe = lu(sparse(I - λe * W))
    rng = MersenneTwister(2026_11_26 + hash(name) % 10_000)
    est = zeros(R, 3); cov = zeros(R, 3, length(CUTS)); Fs = zeros(R, 3)
    t = @elapsed for rep in 1:R
        ys = Fρ \ (μ .+ Fe \ (e .* rand(rng, (-1.0, 1.0), n)))
        for (j, λ) in enumerate(LAMBDAS_P)
            f = ivfit(S24[λ], P24[λ]..., ys)
            est[rep, j] = f.rho; Fs[rep, j] = f.F
            for (l, cut) in enumerate(CUTS)
                cov[rep, j, l] = abs(f.rho - ρ0) <= z * f.se[cut]
            end
        end
    end
    @printf("cell %s: truth df %.1f, ρ₀ = %.3f, λ_e = %.1f  (%.0fs)\n", name, dft, ρ0, λe, t)
    for (j, λ) in enumerate(LAMBDAS_P)
        @printf("  λ = %-5g mean ρ̂_IV %+.3f (MC se %.4f)  sd %.3f  bias %+.3f  median F %.1f  covers ρ₀: HC0 %.3f, 10 km %.3f, 25 km %.3f\n",
                λ, mean(est[:, j]), std(est[:, j]) / sqrt(R), std(est[:, j]), mean(est[:, j]) - ρ0,
                median(Fs[:, j]), mean(cov[:, j, 1]), mean(cov[:, j, 2]), mean(cov[:, j, 3]))
    end
    flush(stdout)
end

println("\nPART 2: the real data")
for k in (16, 24)
    _, Z, Ω = design(k)
    for λ in (100.0, 10.0, 1.0, 0.1)
        S = smoother(Z, Ω, λ)
        f = ivfit(S, prep(S)..., y)
        r = root_rho(y, X, W; tensor_pspline(u, v; k)..., lambda = λ)
        @printf("  k = %d, λ = %-5g ρ̂_IV %+.3f  95%%: HC0 [%+.3f, %+.3f]  10 km [%+.3f, %+.3f]  25 km [%+.3f, %+.3f]  F %.1f   root_rho %.3f\n",
                k, λ, f.rho, f.rho - z * f.se[0.0], f.rho + z * f.se[0.0], f.rho - z * f.se[10.0],
                f.rho + z * f.se[10.0], f.rho - z * f.se[25.0], f.rho + z * f.se[25.0], f.F, r.rho)
        flush(stdout)
    end
end
