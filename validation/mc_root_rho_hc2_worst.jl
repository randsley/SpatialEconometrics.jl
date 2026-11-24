# Monte Carlo for root_rho's test-inversion set when K/n does not tend to 0 (THEORY.md, Lemmas 7–10
# and Proposition 6″), through the released function, and for its `vhat_worst_bias` diagnostic.
#
# Proposition 6″: P(ρ₀ ∈ CS) − [2Φ(z√(1 + β_V)) − 1] → 0, where β_V = (E V̂(ρ₀) − V)/V. Under
# homoskedastic symmetric errors, Lemma 8 gives β_V = (Q₂ + L₅ᴳ + κΣₖcₖ)/(2Σᵢⱼ Ωᵢⱼ + ‖b‖²), b = C′f.
# The research checks (tc_hc2_bias.jl, tc_hc2_worst.jl) used a fixed C(ρ₀). root_rho uses REsd's
# C(ρ̃), which is random; the prediction here is computed at C(ρ₀), so this measures the package
# against the proposition, not a copy of the research numbers.
#
# Design: a jittered 30 × 30 grid (n = 900), kNN-6, X = x, basis tensor_pspline(u, v; k),
# ρ₀ = 0.4, σ² = 1. Cells: k ∈ {9, 13, 16} (K/n = 0.09, 0.19, 0.29) × errors Rademacher (κ = −2,
# Lemma 8's worst case) and normal (κ = 0) × f = 0 (the worst f) and f = 0.5x + P_B g,
# g = 2 sin(2πu)cos(πv). R = 1000 per cell; the MC SE of a coverage near 0.95 is 0.007.
# Per draw: whether ρ₀ ∈ r.cs; the same from g(ρ₀)² ≤ z²V̂(ρ₀) with r's quadratic and quartic;
# and, for the power check, g(ρ₀)² ≤ z²·0.7·V̂(ρ₀).
# Pass criterion (set before the run):
#   1. every cell: CS's coverage within ±0.021 (3 SE) of 2Φ(z√(1 + β_V)) − 1;
#   2. every cell: CS's coverage ≥ the mean reported bound 2Φ(z√(1 + vhat_worst_bias)) − 1 − 0.021;
#   3. power: with V̂ scaled by 0.7, coverage within ±0.021 of 2Φ(z√(0.7(1 + β_V))) − 1 and below
#      0.929 in every cell (a check that could not tell 0.95 from 0.90 would pass anything);
#   4. the mean reported vhat_worst_bias within 0.005 of β_min at C(ρ₀);
#   5. the membership in r.cs and the direct check agree in every draw.
#
# Run:    julia --project=. validation/mc_root_rho_hc2_worst.jl [R]   (R = 1000 by default)
# Output: validation/mc_root_rho_hc2_worst_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf, Distributions

const RHO0 = 0.4
const Z95 = 1.959963984540054
Φ(x) = cdf(Normal(), x)
cover(β, c = 1.0) = 2Φ(Z95 * sqrt(c * (1 + β))) - 1

function design(; side = 30, seed = 5)
    rng = MersenneTwister(seed); n = side^2
    P = (hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ rand(rng, n, 2)) ./ side
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, 6); x = randn(rng, n)
    (; n, W, Wd = Matrix(W), u, v, x, X = reshape(x, :, 1), Fac = lu(sparse(I - RHO0 * W)))
end

# Lemma 8's pieces at C(ρ₀), dense: β_V for κ and f, and β_min.
function lemma8(d, B, f)
    Z = hcat(d.X, B); Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]; Pz = Q * Q'; M = I - Pz; Mii = diag(M)
    Gt = Matrix((I - RHO0 * d.Wd)' \ d.Wd')
    C = (Gt .- Diagonal(diag(Gt * M) ./ Mii)) * M
    Ω = ((C + C') ./ 2) .^ 2; H2 = ((C + C') * Pz) .^ 2; M2 = M .^ 2
    Q2 = 4 * sum(Ω .* M2 ./ (Mii .* Mii')); L5G = -2 * sum(H2 .* M2 ./ (Mii .* Mii'))
    U = M2 ./ Mii
    Σc = sum(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .- sum(U .* (H2 * U); dims = 1))
    V0 = 2 * sum(Ω); b = C' * f
    (; βV = κ -> (Q2 + L5G + κ * Σc) / (V0 + sum(abs2, b)), βmin = min(0, Q2 + L5G - 2Σc) / V0, Σc,
       K = size(Z, 2), hmax = maximum(1 .- Mii))
end

incs(cs, ρ) = any(lo <= ρ <= hi for (lo, hi) in cs)
polyv(vp, ρ) = sum(vp[k + 1] * ρ^k for k in 0:4)

function cell(d, k, law, fname, R)
    B = tensor_pspline(d.u, d.v; k).basis
    gsurf = 2 .* sin.(2π .* d.u) .* cos.(π .* d.v)
    f = fname == :zero ? zeros(d.n) : 0.5 .* d.x .+ B * (B \ gsurf)
    L = lemma8(d, B, f)
    κ = law == :rademacher ? -2.0 : 0.0
    βV = L.βV(κ)
    rng = MersenneTwister(1000k + (law == :rademacher) + 2 * (fname == :zero))
    hit = 0; hit07 = 0; agree = 0; rep = Float64[]
    for _ in 1:R
        ε = law == :rademacher ? rand(rng, (-1.0, 1.0), d.n) : randn(rng, d.n)
        y = d.Fac \ (f .+ ε)
        r = root_rho(y, d.X, d.W; basis = B)
        g = r.c - r.b * RHO0 + r.a * RHO0^2; Vh = polyv(r.vpoly, RHO0)
        inset = incs(r.cs, RHO0); direct = Vh > 0 && g^2 <= Z95^2 * Vh
        hit += inset; agree += inset == direct
        hit07 += Vh > 0 && g^2 <= Z95^2 * 0.7 * Vh
        push!(rep, r.vhat_worst_bias)
    end
    cs = hit / R; c07 = hit07 / R
    bound = mean(cover.(rep))
    ok1 = abs(cs - cover(βV)) <= 0.021
    ok2 = cs >= bound - 0.021
    ok3 = abs(c07 - cover(βV, 0.7)) <= 0.021 && c07 < 0.929
    ok4 = abs(mean(rep) - L.βmin) <= 0.005
    ok5 = agree == R
    @printf("k = %2d (K/n %.3f, max leverage %.3f) %-10s f = %-6s | β_V %+.4f → %.4f, CS %.4f | reported β_min %+.4f (at C(ρ₀) %+.4f) → bound %.4f | ×0.7: predicted %.4f, simulated %.4f | %s%s%s%s%s\n",
            k, L.K / d.n, L.hmax, law, fname, βV, cover(βV), cs, mean(rep), L.βmin, bound,
            cover(βV, 0.7), c07, ok1 ? "1" : "-", ok2 ? "2" : "-", ok3 ? "3" : "-", ok4 ? "4" : "-", ok5 ? "5" : "-")
    return ok1 && ok2 && ok3 && ok4 && ok5
end

R = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
d = design()
println("root_rho's test-inversion set at K/n = 0.09–0.29 against Proposition 6″; n = $(d.n), R = $R")
println("Flags: the criteria passed in that cell (1–5, see the header); '-' marks a failure.\n")
t0 = time(); allok = true
for k in (9, 13, 16), law in (:rademacher, :normal), fname in (:zero, :smooth)
    global allok &= cell(d, k, law, fname, R)
end
@printf("\nAll criteria met in every cell: %s (%.1f min)\n", allok ? "yes" : "NO", (time() - t0) / 60)
