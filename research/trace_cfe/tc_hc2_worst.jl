# The worst case of the HC2 V̂'s bias at ρ₀ over error laws (THEORY.md, Lemma 8), and the coverage of
# the test-inversion set that it guarantees.
#
# Under homoskedasticity (σᵢ² = σ², and by scale σ² = 1) with symmetric errors (μ₃ = 0), Lemma 7's
# Q₁ = L₁ = L₂ = L₄ = 0 and the Gaussian part of L₃ is 0 (HM = 0), so
#     E V̂ − V = Q₂ + L₅ᴳ + Σₖ cₖκₖ,
#     Q₂ = 4Σᵢⱼ ΩᵢⱼMᵢⱼ²/(MᵢᵢMⱼⱼ),  L₅ᴳ = −2Σᵢⱼ Hᵢⱼ²Mᵢⱼ²/(MᵢᵢMⱼⱼ),
#     cₖ = 2uₖ′Ωuₖ + Σᵢ Hᵢₖ²Mᵢₖ²/Mᵢᵢ − uₖ′(H∘H)uₖ,   uₖ = (Mᵢₖ²/Mᵢᵢ)ᵢ,
# all of them functions of W and the basis alone. Every law has κₖ ≥ −2σ⁴ (E ε⁴ ≥ (E ε²)²), with
# equality only for the two-point law ±σ. For iid errors (one κ for all units) the bias is
# Q₂ + L₅ᴳ + κΣₖcₖ, smallest at κ = −2 when Σₖcₖ > 0. For unit-specific κₖ ∈ [−2, κ̄] it is smallest
# at κₖ = −2 where cₖ > 0 and at κₖ = κ̄ where cₖ < 0, so it needs the bound κ̄ unless every cₖ ≥ 0.
# V = 2Σ Ωᵢⱼ + ‖b‖² is smallest at b = 0. The iid worst case is
#     β_min = min(0, Q₂ + L₅ᴳ − 2Σₖcₖ) / (2Σᵢⱼ Ωᵢⱼ),
# and, with V̂ concentrated, liminf P(ρ₀ ∈ CS) ≥ 2Φ(1.96√(1 + β_min)) − 1.
#
# Checks: (1) min cₖ, Σcₖ, β_min and the κ̄ slope, per design; (2) Rademacher errors attain it:
# simulated E V̂/V against the prediction at f = 0 (b = 0), and CS's coverage at ρ₀ against
# 2Φ(1.96√(1 + β)) − 1 and against the oracle (the true V), R = 4000;
# (3) power: uniform errors (κ = −1.2) give a smaller bias than Rademacher in every design, and
# normal errors (κ = 0) a smaller one still.
# Designs: the jittered 30 × 30 grid of tc_hc2_bias.jl with hat bases k = 9, 13, 16, 19; uniform random
# locations (n = 900) with k = 7, 10, 12; and a 30 × 30 rook lattice with k = 9, 13, 16. kNN-6 W except
# on the rook lattice; basis [hats, x]; Jin & Lee's C(ρ₀), ρ₀ = 0.4.
#
# Run: julia --project=. research/trace_cfe/tc_hc2_worst.jl > research/trace_cfe/tc_hc2_worst_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
symm(Q) = (Q + Q') / 2

function rookW(side)
    n = side^2; I_ = Int[]; J_ = Int[]
    idx(i, j) = (i - 1) * side + j
    for i in 1:side, j in 1:side, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        a, b = i + di, j + dj
        1 <= a <= side && 1 <= b <= side && (push!(I_, idx(i, j)); push!(J_, idx(a, b)))
    end
    A = sparse(I_, J_, 1.0, n, n); A ./ sum(A; dims = 2)
end

function design(kind; side = 30, seed = 11)
    rng = MersenneTwister(seed); n = side^2
    P = if kind == :jittered
        (hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ rand(rng, n, 2)) ./ side
    elseif kind == :random
        rand(rng, n, 2)
    else
        (hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ 0.5) ./ side
    end
    W = kind == :rook ? rookW(side) : knnW(P, 6)
    (; n, Wd = Matrix(W), u = P[:, 1], v = P[:, 2], x = randn(rng, n))
end

function objs(d, k)
    Z = hcat(hatbasis(d.u, d.v, k), d.x); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]
    Pz = Q * Q'; M = I - Pz; Mii = diag(M)
    maximum(1 .- Mii) > 1 - 1e-8 && return nothing
    G = d.Wd / (I - 0.4 * d.Wd)
    E = G' .- Diagonal(diag(G' * M) ./ Mii); C = E * M
    Ω = symm(C) .^ 2; H = (C + C') * Pz; H2 = H .^ 2; M2 = M .^ 2
    off = 1 .- I(d.n)
    Q2 = 4 * sum(Ω .* off .* M2 ./ (Mii .* Mii'))
    L5G = -2 * sum(H2 .* M2 ./ (Mii .* Mii'))
    U = M2 ./ Mii                                          # U[i, k] = Mᵢₖ²/Mᵢᵢ, column k is uₖ
    c = vec(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .- sum(U .* (H2 * U); dims = 1))
    ΣΩ = sum(Ω)
    (; K = size(Q, 2), M, Mii, C, Ω, H, H2, Q2, L5G, c, ΣΩ, hmax = maximum(1 .- Mii),
       num = κ -> Q2 + L5G + κ * sum(c))
end

draw(rng, law, n) = law == :rademacher ? rand(rng, (-1.0, 1.0), n) :
                    law == :uniform ? (rand(rng, n) .- 0.5) .* sqrt(12) : randn(rng, n)
const KAPPA = Dict(:rademacher => -2.0, :uniform => -1.2, :normal => 0.0)
covbound(β) = 2 * cdf(Normal(), 1.96 * sqrt(1 + β)) - 1

function simulate(d, o, law; R = 4000, seed = 5)
    rng = MersenneTwister(seed + o.K + hash(law) % 997)
    V = 2 * o.ΣΩ                                           # f = 0, so b = 0
    r = zeros(R); cs = 0; oc = 0
    for i in 1:R
        y0 = draw(rng, law, d.n)
        û = o.M * y0; s = û .^ 2 ./ o.Mii
        Vh = 2 * dot(s, o.Ω * s) + sum(((o.H * y0) .^ 2 .- o.H2 * s) .* s)
        g = dot(y0, o.C * y0)
        r[i] = Vh / V; cs += Vh > 0 && g^2 <= 1.96^2 * Vh; oc += g^2 <= 1.96^2 * V
    end
    β = o.num(KAPPA[law]) / V
    (; β, sim = mean(r), z = (mean(r) - 1 - β) / (std(r) / sqrt(R)), cs = cs / R, oracle = oc / R)
end

println("Worst case of HC2's bias at ρ₀ over symmetric error laws, homoskedastic (Lemma 8); n = 900, R = 4000")
println("MC SE of a coverage near 0.95: 0.0034.\n")
t0 = time(); allok = true; zs = Float64[]; gaps = Float64[]; ogaps = Float64[]; βiid = Float64[]
for (kind, ks) in ((:jittered, (9, 13, 16, 19)), (:random, (7, 10, 12)), (:rook, (9, 13, 16)))
    d = design(kind)
    for k in ks
        o = objs(d, k)
        if o === nothing
            @printf("%-8s k = %2d: leverage 1, skipped\n", kind, k); continue
        end
        βmin = min(0.0, o.num(-2.0)) / (2 * o.ΣΩ); push!(βiid, βmin)
        cneg = sum(min.(o.c, 0)); cpos = sum(max.(o.c, 0))
        βκ̄(κ̄) = min(0.0, o.Q2 + o.L5G - 2cpos + κ̄ * cneg) / (2 * o.ΣΩ)
        @printf("%-8s k = %2d: K/n %.3f, max leverage %.3f | min cₖ %+.2e (all ≥ 0: %s) | Q₂/V %.4f L₅ᴳ/V %+.4f −2Σcₖ/V %+.4f | β_min %+.4f → coverage ≥ %.4f\n",
                kind, k, o.K / d.n, o.hmax, minimum(o.c), minimum(o.c) >= 0 ? "yes" : "NO",
                o.Q2 / (2o.ΣΩ), o.L5G / (2o.ΣΩ), -2 * sum(o.c) / (2o.ΣΩ), βmin, covbound(βmin))
        @printf("    unit-by-unit κₖ ∈ [−2, κ̄]: Σ_{cₖ<0} cₖ/V %+.2e; β_min(κ̄ = 4) %+.4f, β_min(κ̄ = 20) %+.4f\n",
                cneg / (2o.ΣΩ), βκ̄(4.0), βκ̄(20.0))
        res = Dict(law => simulate(d, o, law) for law in (:rademacher, :uniform, :normal))
        for law in (:rademacher, :uniform, :normal)
            s = res[law]; push!(zs, s.z); push!(gaps, s.cs - covbound(s.β)); push!(ogaps, s.cs - s.oracle)
            @printf("    %-10s κ = %+.1f: E V̂/V predicted %.4f, simulated %.4f (z %+.1f) | CS at ρ₀ %.4f, oracle %.4f, 2Φ(1.96√(1+β)) − 1 = %.4f\n",
                    law, KAPPA[law], 1 + s.β, s.sim, s.z, s.cs, s.oracle, covbound(s.β))
        end
        ordered = res[:rademacher].β <= res[:uniform].β <= res[:normal].β
        global allok &= ordered
        println("    power: Rademacher ≤ uniform ≤ normal in E V̂/V: ", ordered ? "yes" : "NO")
    end
end
@printf("\nlargest |z| %.1f; iid worst case β_min %+.4f to %+.4f (coverage ≥ %.4f); CS minus its predicted coverage %+.4f to %+.4f; CS minus the oracle %+.4f to %+.4f; ordering held in every design: %s (%.1f min)\n",
        maximum(abs, zs), minimum(βiid), maximum(βiid), covbound(minimum(βiid)), minimum(gaps), maximum(gaps),
        minimum(ogaps), maximum(ogaps), allok ? "yes" : "NO", (time() - t0) / 60)
