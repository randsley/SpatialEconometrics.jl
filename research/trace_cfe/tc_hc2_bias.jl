# The exact bias of the HC2 V̂ at ρ₀ for any error law, and the test-inversion set's coverage when
# K/n is large (towards Proposition 6 for K/n → θ > 0).
#
# Proposition 6's CS needs only V̂(ρ₀): Theorem 1 gives g(ρ₀)/√V → N(0, 1) for K/n → θ ∈ [0, 1), and
# the set uses no root. So CS covers asymptotically if V̂(ρ₀)/V → 1, and at least at its level if
# V̂(ρ₀) ≥ V(1 + o_p(1)). This script writes E V̂(ρ₀) − V exactly, as a sum of signed terms, checks it
# by simulation, and measures CS's coverage at ρ₀ up to K/n = 0.5.
#
# Setup at ρ₀: y₀ = f + ε, f ∈ span(B), C with diag C = 0 and CP_B = 0 (here Jin & Lee's C(ρ₀) =
# E(ρ₀)M), g = y₀′Cy₀ = b′ε + ε′Cε with b = C′f, and V = 2σ²′Ωσ² + Σᵢbᵢ²σᵢ², Ω = Cs∘Cs, exact for any
# law since diag C = 0. The package's V̂ (RootRho.jl, _vhat_poly):
#     V̂ = 2s′Ωs + Σᵢ(b̂ᵢ² − noiseᵢ)sᵢ,  sᵢ = ûᵢ²/Mᵢᵢ,  û = Mε,  b̂ = Hy₀ = b + ζ,  ζ = Hε,
#     H = (C + C′)P_B,  noiseᵢ = Σₖ Hᵢₖ²sₖ.
# With independent errors, μ₃ᵢ and κᵢ (the third moment and fourth cumulant of εᵢ), D = Diag(M),
# 𝒳 = (MΣM)∘(MΣM), 𝒦 = (M∘M)Diag(κ)(M∘M):
#     σ̄² = E s = D⁻¹(M∘M)σ²,   Cov s = D⁻¹(2𝒳 + 𝒦)D⁻¹,
# and, by the moment identities of THEORY.md's CJN paragraphs,
#   quadratic part:  E 2s′Ωs − 2σ²′Ωσ² = [2σ̄²′Ωσ̄² − 2σ²′Ωσ²]                      (Q1, the mean shift)
#                                        + 2 tr(Ω D⁻¹·2𝒳·D⁻¹)                     (Q2, Gaussian, ≥ 0)
#                                        + 2 tr(Ω D⁻¹𝒦D⁻¹)                        (Q3, the cumulant)
#   linear part:     E Σ(b̂ᵢ² − noiseᵢ)sᵢ − Σbᵢ²σᵢ² = Σbᵢ²(σ̄ᵢ² − σᵢ²)               (L1, the mean shift)
#                    + 2Σᵢbᵢ[(H∘M∘M)μ₃]ᵢ/Mᵢᵢ                                      (L2, skewness)
#                    + Σᵢ[2(HΣM)ᵢᵢ² + ((H∘H∘M∘M)κ)ᵢ]/Mᵢᵢ                          (L3, cross)
#                    + Σᵢₖ Hᵢₖ²σ̄ᵢ²(σₖ² − σ̄ₖ²)                                     (L4, noise mean)
#                    − Σᵢₖ Hᵢₖ² Cov(sₖ, sᵢ)                                        (L5, noise covariance, ≤ 0 if κ ≥ 0)
# (the proof is in THEORY.md with this script's results). Q2 ≥ 0 always; Q3 and L5 have the sign of κ
# (Ω, H∘H ≥ 0 entrywise); Q1, L1, L4 vanish under homoskedasticity (σ̄² = σ²); L2 and L3 need skewness
# or heteroskedasticity.
#
# Design: n = 900 on a jittered 30 × 30 grid (each point uniform in its own cell), so that the hat
# basis's leverage stays below 1 up to K/n = 0.5; kNN-6 W; x ~ N(0, 1); basis [tensor hats k × k, x],
# k ∈ {9, 13, 16, 19, 21} (K/n ≈ 0.09 to 0.49; a basis with leverage 1 is reported and skipped); f = x + a·P_B g, g = sin(2πu)cos(πv), a = 1 and 4 (so
# A4 holds); variances constant, smooth (exp(3(u + v − 1)), normalised) and rough (iid log-normal,
# normalised); errors normal, standardised χ²₃ (κ = 4σ⁴, μ₃ = √(8/3)σ³) and uniform (κ = −1.2σ⁴). R = 1000.
# Reported per cell: max leverage; E V̂/V predicted (1 + bias/V) and simulated, with z; the bias terms
# over V; and CS's coverage at ρ₀, P(g² ≤ 1.96²V̂), with the oracle P(g² ≤ 1.96²V) beside it.
# Checks: the formula against simulation in every cell (|z| ≲ 3); power: the formula without Q2
# (the homoskedastic-normal conjecture "V̂ consistent") must be rejected where Q2/V is large.
#
# Run: julia --project=. research/trace_cfe/tc_hc2_bias.jl > research/trace_cfe/tc_hc2_bias_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
g_surf(u, v) = sin(2π * u) * cos(π * v)
symm(Q) = (Q + Q') / 2

function setup(; side = 30, seed = 11)
    rng = MersenneTwister(seed)
    n = side^2
    P = hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ rand(rng, n, 2)
    P ./= side
    W = knnW(P, 6); Wd = Matrix(W); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σs = Dict(:constant => ones(n),
              :smooth => (s = exp.(3 .* (u .+ v .- 1)); s ./ mean(s)),
              :rough => (s = exp.(randn(rng, n)); s ./ mean(s)))
    (; n, Wd, u, v, x, σs)
end

const LAWS = Dict(:normal => (0.0, 0.0), :chisq3 => (sqrt(8 / 3), 4.0), :uniform => (0.0, -1.2))
draw(rng, law, n) = law == :normal ? randn(rng, n) :
                    law == :chisq3 ? (rand(rng, Chisq(3), n) .- 3) ./ sqrt(6) :
                    (rand(rng, n) .- 0.5) .* sqrt(12)

function basisobjs(d, k)
    Z = hcat(hatbasis(d.u, d.v, k), d.x); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]
    Pz = Q * Q'; M = I - Pz; Mii = diag(M)
    G = d.Wd / (I - 0.4 * d.Wd)
    AM = G' * M; E = G' .- Diagonal(diag(AM) ./ Mii); C = E * M      # Jin & Lee's C(ρ₀), diag C = 0
    Cs = symm(C); Ω = Cs .^ 2; H = (C + C') * Pz
    (; K = size(Q, 2), Pz, M, Mii, C, Ω, H, H2 = H .^ 2, M2 = M .^ 2, hmax = maximum(1 .- Mii),
       dC = maximum(abs, diag(C)), CP = maximum(abs, C * Pz))
end

function predict(o, σ2, f, law)
    sk, ek = LAWS[law]
    μ3 = sk .* σ2 .^ 1.5; κ = ek .* σ2 .^ 2
    M, Mii, Ω, H = o.M, o.Mii, o.Ω, o.H
    b = o.C' * f
    V = 2 * dot(σ2, Ω * σ2) + sum(b .^ 2 .* σ2)
    σ̄ = (o.M2 * σ2) ./ Mii
    MΣM = M * (σ2 .* M); 𝒳 = MΣM .^ 2; 𝒦 = o.M2 * (κ .* o.M2)
    Cov = (2 .* 𝒳 .+ 𝒦) ./ (Mii .* Mii')
    Q1 = 2 * dot(σ̄, Ω * σ̄) - 2 * dot(σ2, Ω * σ2)
    Q2 = 2 * sum(Ω .* (2 .* 𝒳 ./ (Mii .* Mii')))
    Q3 = 2 * sum(Ω .* (𝒦 ./ (Mii .* Mii')))
    L1 = sum(b .^ 2 .* (σ̄ .- σ2))
    L2 = 2 * sum(b .* ((H .* o.M2) * μ3) ./ Mii)
    L3 = sum((2 .* vec(sum(H .* (σ2' .* M); dims = 2)) .^ 2 .+ (o.H2 .* o.M2) * κ) ./ Mii)
    L4 = sum(o.H2 .* (σ̄ .* (σ2 .- σ̄)'))
    L5 = -sum(o.H2 .* Cov)
    (; V, Vq = 2 * dot(σ2, Ω * σ2), terms = (; Q1, Q2, Q3, L1, L2, L3, L4, L5), b)
end

function cell(d, o, vname, law, amp; R = 1000, seed = 3)
    σ2 = d.σs[vname]; sd = sqrt.(σ2)
    f = d.x .+ amp .* (o.Pz * g_surf.(d.u, d.v))
    p = predict(o, σ2, f, law)
    bias = sum(values(p.terms))
    rng = MersenneTwister(seed + 100o.K + round(Int, 10amp) + hash((vname, law)) % 1000)
    r = zeros(R); cs = 0; co = 0; neg = 0
    for i in 1:R
        ε = sd .* draw(rng, law, d.n); y0 = f .+ ε
        û = o.M * y0; s = û .^ 2 ./ o.Mii
        b̂ = o.H * y0; noise = o.H2 * s
        Vh = 2 * dot(s, o.Ω * s) + sum((b̂ .^ 2 .- noise) .* s)
        g = dot(y0, o.C * y0)
        r[i] = Vh / p.V; neg += Vh <= 0
        cs += Vh > 0 && g^2 <= 1.96^2 * Vh; co += g^2 <= 1.96^2 * p.V
    end
    pred = 1 + bias / p.V; z = (mean(r) - pred) / (std(r) / sqrt(R))
    zno2 = (mean(r) - (pred - p.terms.Q2 / p.V)) / (std(r) / sqrt(R))
    t = p.terms
    @printf("  %-8s %-7s a=%.0f | E V̂/V pred %.4f sim %.4f (z %+5.1f; without Q2 z %+6.1f) sd %.3f | Q1 %+.4f Q2 %+.4f Q3 %+.4f L1 %+.4f L2 %+.4f L3 %+.4f L4 %+.4f L5 %+.4f | lin %.2f | CS %.3f oracle %.3f%s\n",
            vname, law, amp, pred, mean(r), z, zno2, std(r), t.Q1 / p.V, t.Q2 / p.V, t.Q3 / p.V,
            t.L1 / p.V, t.L2 / p.V, t.L3 / p.V, t.L4 / p.V, t.L5 / p.V, 1 - p.Vq / p.V, cs / R, co / R,
            neg > 0 ? @sprintf(" (V̂ ≤ 0 in %d)", neg) : "")
    (; z, cs = cs / R, co = co / R, pred)
end

d = setup()
println("HC2 V̂ at ρ₀: exact bias and CS coverage; n = $(d.n), jittered grid, kNN-6, hat basis + x, R = 1000")
println("Terms are over V. lin = the linear part's share of V. MC SE of a coverage near 0.95: 0.007.\n")
t0 = time(); zs = Float64[]; covs = Float64[]; cells = []
for k in (9, 13, 16, 19, 21)
    o = basisobjs(d, k)
    if o.hmax > 1 - 1e-8
        @printf("k = %d: K = %d, K/n = %.3f, max leverage 1: some Mᵢᵢ = 0, so sᵢ = ûᵢ²/Mᵢᵢ is undefined (THEORY.md, \"Leverage 1 rules out any estimate free of f\"); skipped\n\n",
                k, o.K, o.K / d.n)
        continue
    end
    @printf("k = %d: K = %d, K/n = %.3f, max leverage %.3f (checks: max|diag C| %.1e, max|CP_B| %.1e)\n",
            k, o.K, o.K / d.n, o.hmax, o.dC, o.CP)
    for amp in (1.0, 4.0), vname in (:constant, :smooth, :rough), law in (:normal, :chisq3, :uniform)
        c = cell(d, o, vname, law, amp); push!(zs, c.z); push!(covs, c.cs); push!(cells, (k, vname, law, amp))
    end
    @printf("  (%.1f min so far)\n\n", (time() - t0) / 60)
end
@printf("Formula against simulation: largest |z| %.1f over %d cells. CS coverage at ρ₀: %.3f–%.3f.\n",
        maximum(abs, zs), length(zs), minimum(covs), maximum(covs))
# The cell with the largest |z|, again with R = 20000 and a new seed: the degree-4 statistic has heavy
# tails under χ²₃ errors, so R = 1000 can understate the standard error of its mean.
j = argmax(abs.(zs)); (k, vname, law, amp) = cells[j]
println("\nRecheck of the largest |z| (k = $k, $vname, $law, a = $(Int(amp))) with R = 20000:")
cell(d, basisobjs(d, k), vname, law, amp; R = 20000, seed = 99)
