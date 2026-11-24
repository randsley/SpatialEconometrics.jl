# Lemma 12: the HC2 V̂'s worst-case bias at ρ₀ when the errors may be skewed (homoskedastic, σ² = 1).
#
# Lemma 7 with σᵢ² = 1: Q₁ = L₁ = L₄ = 0 and L₃ᴳ = 0, so
#     E V̂ − V = A + Σₖ cₖκₖ + 2 b′Tγ,   V = V₀ + ‖b‖²,
# with A = Q₂ + L₅ᴳ, Lemma 8's cₖ, γₖ = μ₃ₖ, T = D⁻¹(H∘M∘M) and b = C′f ∈ 𝓑 = range(C′[X, B]).
# Every law has κ ≥ γ² − 2 (Pearson), with equality only for a two-point law.
#
# (b) One law for all units, |γ| ≤ γ̄, Σc ≥ 0: the worst κ is γ² − 2. With τ = ‖Π_𝓑 T1‖ and
#     a(γ) = A − 2Σc + γ²Σc, the minimum over b ∈ 𝓑 of (a − 2|γ|τs)/(V₀ + s²) is the smaller eigenvalue
#     of [[a, −|γ|τ√V₀], [−|γ|τ√V₀, 0]]/V₀:
#         β(γ) = [a − √(a² + 4γ²τ²V₀)] / (2V₀),   at ‖b‖ = s* = [a + √(a² + 4γ²τ²V₀)]/(2|γ|τ),
#     and β_skew(γ̄) = min over 0 ≤ γ ≤ γ̄ (a one-dimensional grid).
# (c) Rate. The crude bound τ² ≤ ‖T1‖² ≤ ‖H‖_F² gives only O(√(K/n)). The sharp one: |Pᵢₖ| ≤ h̄, so
#     |Hᵢₖ| = |(C′P)ᵢₖ| ≤ ‖C‖₁h̄ (‖C‖₁ the largest absolute column sum), and
#     |(T1)ᵢ| = |HᵢᵢMᵢᵢ + Σ_{k≠i} HᵢₖPᵢₖ²/Mᵢᵢ| ≤ ‖C‖₁h̄/(1 − h̄), so τ/√V₀ ≤ ‖C‖₁h̄/((1 − h̄)√(V₀/n)) =: τ̄.
#     And β_skew(γ̄) ≥ β_min − γ̄τ/√V₀ (√(a² + 4g²V₀) ≤ |a| + 2g√V₀, and a(γ) ≥ a(0)).
#
# Checks:
# (1) the closed form against brute force: 2000 random directions in 𝓑 (and ±T1's projection), each with
#     its best s and γ on a grid; none may go below β_skew, and the closed form's b must attain it.
# (2) Monte Carlo at the worst (γ, f), with two-point errors of skewness γ (κ = γ² − 2), R = 4000:
#     E V̂/V against 1 + β_skew; and at f = 0 (power: there L₂ = 0, so the prediction without L₂ holds
#     and the one with it is rejected, or the reverse at the worst f).
# (3) τ/√V₀ against K/n and √(K/n): k = 7–16 at n = 900, and K/n ≈ 0.19 at n = 400, 900, 1600, 2500.
# (4) unit-specific γₖ ∈ [−γ̄, γ̄], κₖ = γₖ² − 2 (cₖ > 0) or κ̄ (cₖ < 0): projected gradient on the
#     closed-form value over γ from 10 starts; how much worse than one common γ.
#
# Run: julia --project=. research/trace_cfe/tc_hc2_skew.jl > research/trace_cfe/tc_hc2_skew_output.txt
include(joinpath(@__DIR__, "tc_hc2_hetero.jl"))

Cnorm1(C) = maximum(sum(abs.(C); dims = 1))
τbar(s) = Cnorm1(s.o.C) * s.o.hmax / ((1 - s.o.hmax) * sqrt(s.V0 / s.n))

function skewobjs(d, k)
    o = objs(d, k); n = d.n
    Z = hcat(hatbasis(d.u, d.v, k), d.x)
    DD = o.Mii .* o.Mii'
    A = 4 * sum(o.Ω .* o.M2 ./ DD) - 2 * sum(o.H2 .* o.M2 ./ DD)
    T = (o.H .* o.M2) ./ o.Mii
    CZ = o.C' * Z; F = svd(CZ); r = count(F.S .> 1e-10 * F.S[1])
    U = F.U[:, 1:r]                                          # orthonormal basis of 𝓑
    V0 = 2 * sum(o.Ω)
    (; o, Z, A, T, U, V0, sc = sum(o.c), n)
end

βof(a, g, V0) = (a - sqrt(a^2 + 4g^2 * V0)) / (2V0)        # g = |γ|τ (or ‖Π Tγ‖)

function worst_common(s, γ̄; ng = 2001)
    t1 = s.U' * (s.T * ones(s.n)); τ = norm(t1)
    best = (Inf, 0.0)
    for γ in range(0, γ̄, length = ng)
        a = s.A - 2s.sc + γ^2 * s.sc
        β = βof(a, γ * τ, s.V0); β < best[1] && (best = (β, γ))
    end
    β, γ = best; a = s.A - 2s.sc + γ^2 * s.sc
    sstar = γ * τ > 0 ? (a + sqrt(a^2 + 4(γ * τ)^2 * s.V0)) / (2γ * τ) : 0.0
    bdir = τ > 0 ? -(s.U * t1) ./ τ : zeros(s.n)              # b = sstar · bdir (γ ≥ 0)
    (; β, γ, τ, b = sstar .* bdir, sstar, βsym = min(0, (s.A - 2s.sc) / s.V0))
end

function brute(s, γ̄, w, rng; ndir = 2000)
    t1 = s.T * ones(s.n); lo = Inf
    γs = range(-γ̄, γ̄, length = 81); ss = 10 .^ range(-3, 3, length = 121) .* sqrt(s.V0)
    dirs = Any[s.U * (s.U' * t1), -(s.U * (s.U' * t1))]
    for _ in 1:ndir; push!(dirs, s.U * randn(rng, size(s.U, 2))); end
    for v in dirs
        v = v ./ norm(v); p = dot(v, t1)
        for γ in γs, sv in ss
            β = (s.A - 2s.sc + γ^2 * s.sc + 2γ * sv * p) / (s.V0 + sv^2); lo = min(lo, β)
        end
    end
    # the closed form's own b attains its value
    att = (s.A - 2s.sc + w.γ^2 * s.sc + 2w.γ * dot(w.b, t1)) / (s.V0 + sum(abs2, w.b))
    lo, att
end

twopoint_p(γ) = γ == 0 ? 0.5 : (1 - γ / sqrt(γ^2 + 4)) / 2  # P(X = hi) for skewness γ
function twopoint(rng, γ, n)
    p = twopoint_p(γ); hi = sqrt((1 - p) / p); lo = -sqrt(p / (1 - p))
    [rand(rng) < p ? hi : lo for _ in 1:n]
end

function mc(s, γ, b; R = 4000, seed = 12)
    o = s.o; rng = MersenneTwister(seed)
    ψ = s.Z \ zeros(s.n)
    f = sum(abs2, b) > 0 ? s.Z * ((o.C' * s.Z) \ b) : zeros(s.n)
    bb = o.C' * f; V = s.V0 + sum(abs2, bb)
    pred = 1 + (s.A + dot(o.c, fill(γ^2 - 2, s.n)) + 2γ * dot(bb, s.T * ones(s.n))) / V
    predsym = 1 + (s.A + dot(o.c, fill(γ^2 - 2, s.n))) / V
    r = zeros(R); cs = 0; oc = 0
    for i in 1:R
        y0 = f .+ twopoint(rng, γ, s.n)
        sres = (o.M * y0) .^ 2 ./ o.Mii
        Vh = 2 * dot(sres, o.Ω * sres) + sum(((o.H * y0) .^ 2 .- o.H2 * sres) .* sres)
        g = dot(y0, o.C * y0)
        r[i] = Vh / V; cs += Vh > 0 && g^2 <= 1.96^2 * Vh; oc += g^2 <= 1.96^2 * V
    end
    se = std(r) / sqrt(R)
    (; pred, predsym, sim = mean(r), z = (mean(r) - pred) / se, zsym = (mean(r) - predsym) / se,
       cs = cs / R, oc = oc / R, bshare = sum(abs2, bb) / V)
end

function worst_unit(s, γ̄, rng; starts = 10, iters = 2000)
    o = s.o; cp = max.(o.c, 0); cm = min.(o.c, 0)
    a0 = s.A - 2 * sum(cp) + KBAR * sum(cm)
    UT = s.U' * s.T
    F(γ) = (a = a0 + dot(cp, γ .^ 2); βof(a, norm(UT * γ), s.V0))
    best = (Inf, zeros(s.n))
    for st in 1:starts
        γ = st == 1 ? fill(γ̄, s.n) : γ̄ .* (2 .* rand(rng, s.n) .- 1)
        val = F(γ); step = 1.0
        for it in 1:iters
            a = a0 + dot(cp, γ .^ 2); u = UT * γ; g2 = dot(u, u); rt = sqrt(a^2 + 4g2 * s.V0)
            grad = ((1 - a / rt) / (2s.V0)) .* (2 .* cp .* γ) .- (1 / rt) .* (2 .* (UT' * u))
            ok = false
            while step > 1e-10
                γn = clamp.(γ .- step .* grad, -γ̄, γ̄); vn = F(γn)
                if vn <= val - 1e-4 * dot(grad, γ .- γn)
                    γ, val = γn, vn; step *= 2; ok = true; break
                end
                step /= 2
            end
            ok || break
        end
        val < best[1] && (best = (val, γ))
    end
    best
end

println("Lemma 12: the HC2 V̂'s worst-case bias with skewed errors (σ² = 1); κ̄ = $(KBAR) where cₖ < 0\n")
rng = MersenneTwister(2028); t0 = time()
println("(1), (2), (4) Four designs, n = 900")
for (name, kind, k) in (("jittered k = 13", :jittered, 13), ("jittered k = 16", :jittered, 16),
                        ("random   k = 10", :random, 10), ("rook     k = 13", :rook, 13))
    s = skewobjs(design(kind), k)
    @printf("%s (K/n %.3f, max leverage %.3f) | τ/√V₀ %.4f, √(K/n) %.3f | symmetric worst (Lemma 8) %+.4f\n",
            name, s.o.K / s.n, s.o.hmax, norm(s.U' * (s.T * ones(s.n))) / sqrt(s.V0), sqrt(s.o.K / s.n),
            min(0, (s.A - 2s.sc) / s.V0))
    for γ̄ in (1.0, 2.0)
        w = worst_common(s, γ̄); lo, att = brute(s, γ̄, w, rng)
        @printf("    γ̄ = %.0f: worst β %+.4f at γ = %.3f, ‖b‖²/V₀ = %.2f | brute force over 2002 directions: min %+.4f (closed form − brute %+.1e), closed form's b attains %+.4f | unit-specific γₖ: ",
                γ̄, w.β, w.γ, w.sstar^2 / s.V0, lo, w.β - lo, att)
        u = worst_unit(s, γ̄, rng)
        @printf("%+.4f (%.0f%% of units at ±γ̄)\n", u[1], 100 * mean(abs.(u[2]) .>= γ̄ - 1e-9))
        if γ̄ == 2.0
            m = mc(s, w.γ, w.b); m0 = mc(s, w.γ, zeros(s.n); seed = 13)
            @printf("    Monte Carlo, two-point errors at γ = %.3f, R = 4000: at the worst f (b share of V %.2f) E V̂/V pred %.4f, sim %.4f (z %+.1f; without L₂, z %+.1f) | CS %.4f, oracle %.4f, 2Φ(1.96√(1+β)) − 1 = %.4f\n",
                    w.γ, m.bshare, m.pred, m.sim, m.z, m.zsym, m.cs, m.oc, 2cdf(Normal(), 1.96sqrt(1 + w.β)) - 1)
            @printf("        at f = 0: pred %.4f, sim %.4f (z %+.1f) | CS %.4f, oracle %.4f   (%.1f min)\n",
                    m0.pred, m0.sim, m0.z, m0.cs, m0.oc, (time() - t0) / 60)
        end
    end
end

println("\n(3) The order of τ/√V₀ and of the worst β (γ̄ = 2), jittered grid")
for (side, k) in ((30, 7), (30, 10), (30, 13), (30, 16), (20, 9), (40, 17), (50, 21))
    d = design(:jittered; side); s = skewobjs(d, k); w = worst_common(s, 2.0); Kn = s.o.K / s.n
    tv = w.τ / sqrt(s.V0)
    @printf("    n = %4d, k = %2d: K/n %.3f, max leverage %.3f | τ/√V₀ %.4f, /(K/n) %.3f, /√(K/n) %.3f, /h̄ %.3f | τ̄ %.3f (‖C‖₁ %.2f), holds %s | worst β %+.4f ≥ β_min − 2τ/√V₀ = %+.4f: %s | symmetric %+.4f, (worst − symmetric)/(K/n) %+.3f   (%.1f min)\n",
            s.n, k, Kn, s.o.hmax, tv, tv / Kn, tv / sqrt(Kn), tv / s.o.hmax, τbar(s), Cnorm1(s.o.C), tv <= τbar(s),
            w.β, w.βsym - 2tv, w.β >= w.βsym - 2tv - 1e-12, w.βsym, (w.β - w.βsym) / Kn, (time() - t0) / 60)
end
