# Anatolyev & Sølvsten's leave-three-out variance estimator for the zero-diagonal moment at ρ₀:
# is it unbiased for V when K/n is large, and does the test-inversion set cover better than with HC2?
#
# Source: Anatolyev & Sølvsten (2023, JoE 236(1) 105473; arXiv 2003.07320v3, read from the page images,
# pp. 11–18). With M the residual maker of the regressors, Dᵢⱼ = MᵢᵢMⱼⱼ − Mᵢⱼ² and
# Dᵢⱼₖ = MᵢᵢDⱼₖ − (MⱼⱼMᵢₖ² + MₖₖMᵢⱼ² − 2MⱼₖMᵢⱼMᵢₖ) (p. 13), and the leave-out residuals (p. 18)
#     yᵢ − xᵢ′β̂₋ᵢⱼ  = (Mⱼⱼ(yᵢ − xᵢ′β̂) − Mᵢⱼ(yⱼ − xⱼ′β̂)) / Dᵢⱼ,
#     yᵢ − xᵢ′β̂₋ᵢⱼₖ = ((yᵢ − xᵢ′β̂) − Mᵢⱼ(yⱼ − xⱼ′β̂₋ⱼₖ) − Mᵢₖ(yₖ − xₖ′β̂₋ⱼₖ)) / (Dᵢⱼₖ/Dⱼₖ),
# they set σ̂²ᵢ,₋ⱼₖ = yᵢ(yᵢ − xᵢ′β̂₋ᵢⱼₖ) (σ̂²ᵢ,₋ⱼ when j = k) and, eqs. (10)–(11),
#     σ̂²ᵢ,₋ⱼ = yᵢ Σ_{k≠j} M̌ᵢₖ,₋ᵢⱼ yₖ,   M̌ᵢₖ,₋ᵢⱼ = (MⱼⱼMᵢₖ − MᵢⱼMⱼₖ)/Dᵢⱼ,
#     (σᵢ²σⱼ²)^ = yᵢ Σ_{k≠j} M̌ᵢₖ,₋ᵢⱼ yₖ · σ̂²ⱼ,₋ᵢₖ,
# unbiased for σᵢ²σⱼ² for any β, because σ̂²ⱼ,₋ᵢₖ is independent of (yᵢ, yₖ) and unbiased for σⱼ².
#
# Our moment (Lemma 7): y₀ = f + ε, f ∈ span(B), diag C = 0, CP_B = 0, b = C′f,
#     V = 2Σ_{i≠j} Ωᵢⱼσᵢ²σⱼ² + Σᵢ bᵢ²σᵢ²,   Ω = Cs∘Cs.
# The linear part, as in their (12): L̂ = Σᵢ Σⱼₖ CⱼᵢCₖᵢ yⱼyₖ σ̂²ᵢ,₋ⱼₖ. For j ≠ k its mean is fⱼfₖσᵢ²; for
# j = k it is (fⱼ² + σⱼ²)σᵢ², which adds Σᵢⱼ Cⱼᵢ²σᵢ²σⱼ². So the unbiased estimator is
#     V̂_L3O = Σ_{i≠j} wᵢⱼ (σᵢ²σⱼ²)^ + L̂,   wᵢⱼ = 2Ωᵢⱼ − (Cⱼᵢ² + Cᵢⱼ²)/2,
# exactly unbiased at every n when all Dᵢⱼₖ > 0 (their Assumption 1). Computed with one pivot p at a
# time: σ̂²_p,₋ₐ_b for all (a, b) is an n × n object, so the cost is O(n³) per draw, with no storage
# beyond n × n.
#
# Part 1 (n = 36): the fast leave-out residuals and V̂_L3O against explicit OLS with rows removed.
# Part 2 (n = 36, R = 200000): E V̂_L3O = V exactly (any n), under heteroskedastic χ²₃ errors with a
#   strong signal; HC2 at the same draws as the power check (Lemma 7 predicts its bias).
# Part 3 (n = 400, jittered 20 × 20 grid, kNN-6, hat basis + x): K/n ≈ 0.09, 0.20, 0.30; constant
#   and rough variances; normal and χ²₃ errors; a = 1, 4, 16. R = 1000. Per cell: E V̂/V and sd for
#   both estimators, HC2's Lemma 7 prediction, CS coverage at ρ₀ for both and for the oracle, the
#   number of draws with V̂_L3O ≤ 0 (counted as not covering), and min Dᵢⱼₖ.
#
# Run: julia -t 16 --project=. research/trace_cfe/tc_l3o.jl > research/trace_cfe/tc_l3o_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
g_surf(u, v) = sin(2π * u) * cos(π * v)
symm(Q) = (Q + Q') / 2

function setup(; side, seed = 11)
    rng = MersenneTwister(seed)
    n = side^2
    P = hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ rand(rng, n, 2)
    P ./= side
    W = knnW(P, 6); Wd = Matrix(W); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σs = Dict(:constant => ones(n),
              :rough => (s = exp.(randn(rng, n)); s ./ mean(s)))
    (; n, Wd, u, v, x, σs)
end

const LAWS = Dict(:normal => (0.0, 0.0), :chisq3 => (sqrt(8 / 3), 4.0))
draw(rng, law, n) = law == :normal ? randn(rng, n) : (rand(rng, Chisq(3), n) .- 3) ./ sqrt(6)

function basisobjs(d, k)
    Z = hcat(hatbasis(d.u, d.v, k), d.x); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]
    Pz = Q * Q'; M = Matrix(I - Pz); Mii = diag(M)
    G = d.Wd / (I - 0.4 * d.Wd)
    AM = G' * M; E = G' .- Diagonal(diag(AM) ./ Mii); C = E * M      # Jin & Lee's C(ρ₀), diag C = 0
    Cs = symm(C); Ω = Cs .^ 2; H = (C + C') * Pz
    w = 2 .* Ω .- (C .^ 2 .+ (C .^ 2)') ./ 2
    w[diagind(w)] .= 0
    D2 = Mii .* Mii' .- M .^ 2
    (; Z, K = size(Q, 2), Pz, M, Mii, C, Ω, H, H2 = H .^ 2, M2 = M .^ 2, w, D2,
       hmax = maximum(1 .- Mii))
end

# Lemma 7's prediction for HC2 (copied from tc_hc2_bias.jl).
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
    (; V, Vq = 2 * dot(σ2, Ω * σ2), bias = Q1 + Q2 + Q3 + L1 + L2 + L3 + L4 + L5)
end

function vhat_hc2(o, y)
    û = o.M * y; s = û .^ 2 ./ o.Mii
    b̂ = o.H * y; noise = o.H2 * s
    2 * dot(s, o.Ω * s) + sum((b̂ .^ 2 .- noise) .* s)
end

# R2[a, b] = yₐ − xₐ′β̂₋ₐ_b (a ≠ b), from û = My.
function l2o(o, û)
    n = length(û); M, Mii, D2 = o.M, o.Mii, o.D2
    R2 = zeros(n, n)
    @inbounds for b in 1:n, a in 1:n
        a == b && continue
        R2[a, b] = (Mii[b] * û[a] - M[a, b] * û[b]) / D2[a, b]
    end
    R2
end

# The leave-three-out residual of p without a and b (a ≠ b, both ≠ p), and D_pab.
@inline function l3o_res(M, Mii, D2, R2, û, p, a, b)
    Dpab = Mii[p] * D2[a, b] - (Mii[a] * M[p, b]^2 + Mii[b] * M[p, a]^2 - 2 * M[a, b] * M[p, a] * M[p, b])
    ((û[p] - M[p, a] * R2[a, b] - M[p, b] * R2[b, a]) * D2[a, b] / Dpab, Dpab)
end

# V̂_L3O, one pivot p per task.
function vhat_l3o(o, y)
    n = length(y); M, Mii, D2, C, w = o.M, o.Mii, o.D2, o.C, o.w
    û = M * y; R2 = l2o(o, û)
    parts = zeros(n)
    Threads.@threads for p in 1:n
        lin = 0.0; quad = 0.0
        @inbounds for b in 1:n
            b == p && continue
            zb = C[b, p] * y[b]
            for a in 1:n
                a == p && continue
                if a == b
                    s = y[p] * R2[p, a]                         # σ̂²_p,₋ₐ
                else
                    r, _ = l3o_res(M, Mii, D2, R2, û, p, a, b)
                    s = y[p] * r                                # σ̂²_p,₋ₐ_b
                end
                lin += C[a, p] * y[a] * zb * s
                # (σₐ²σ_p²)^ term of (11): w_ap yₐ M̌ₐ_b,₋ₐ_p y_b σ̂²_p,₋ₐ_b; M̌ₐ_p,₋ₐ_p = 0, so b = p drops
                quad += w[a, p] * y[a] * (Mii[p] * M[a, b] - M[a, p] * M[p, b]) / D2[a, p] * y[b] * s
            end
        end
        parts[p] = lin + quad
    end
    sum(parts)
end

function min_d3(o)
    n = size(o.M, 1); mins = fill(Inf, n)
    Threads.@threads for p in 1:n
        m = Inf
        @inbounds for b in 1:n, a in 1:b-1
            (a == p || b == p) && continue
            Dpab = o.Mii[p] * o.D2[a, b] - (o.Mii[a] * o.M[p, b]^2 + o.Mii[b] * o.M[p, a]^2 -
                                           2 * o.M[a, b] * o.M[p, a] * o.M[p, b])
            m = min(m, Dpab)
        end
        mins[p] = m
    end
    minimum(mins)
end

# ---------------------------------------------------------------------------------------------------
# Part 1: against explicit OLS with rows removed.
println("Leave-three-out V̂ at ρ₀ (Anatolyev & Sølvsten 2023), threads = $(Threads.nthreads())\n")
println("Part 1: the fast formulas against explicit OLS with rows removed (n = 36, basis 3 × 3 hats + x)")
d1 = setup(side = 6, seed = 5); o1 = basisobjs(d1, 3)
let rng = MersenneTwister(1), n = d1.n
    y = d1.x .+ 2 .* (o1.Pz * g_surf.(d1.u, d1.v)) .+ randn(rng, n)
    û = o1.M * y; R2 = l2o(o1, û)
    res(keep, i) = y[i] - dot(o1.Z[i, :], o1.Z[keep, :] \ y[keep])
    e2 = 0.0; e3 = 0.0
    for i in 1:n, j in 1:n
        i == j && continue
        e2 = max(e2, abs(R2[i, j] - res(setdiff(1:n, (i, j)), i)))
        for k in j+1:n
            k == i && continue
            r, _ = l3o_res(o1.M, o1.Mii, o1.D2, R2, û, i, j, k)
            e3 = max(e3, abs(r - res(setdiff(1:n, (i, j, k)), i)))
        end
    end
    # V̂_L3O by its definition, with σ̂² from explicit fits
    σ2l(i, j, k) = y[i] * res(setdiff(1:n, unique((i, j, k))), i)
    P̂(i, j) = y[i] * sum(((o1.Mii[j] * o1.M[i, k] - o1.M[i, j] * o1.M[j, k]) / o1.D2[i, j]) * y[k] * σ2l(j, i, k)
                         for k in 1:n if k != j)
    quad = sum(o1.w[i, j] * P̂(i, j) for i in 1:n, j in 1:n if i != j)
    lin = sum(o1.C[j, i] * o1.C[k, i] * y[j] * y[k] * σ2l(i, j, k) for i in 1:n, j in 1:n, k in 1:n if j != i && k != i)
    vb = quad + lin; vf = vhat_l3o(o1, y)
    sym = maximum(abs(P̂(i, j) - P̂(j, i)) for i in 1:6, j in 1:6 if i != j)
    @printf("  K = %d, max leverage %.3f, min Dᵢⱼₖ %.3e\n", o1.K, o1.hmax, min_d3(o1))
    @printf("  max |error|: leave-two-out residual %.1e, leave-three-out residual %.1e\n", e2, e3)
    @printf("  V̂_L3O: by definition %.10g, fast %.10g (relative difference %.1e)\n", vb, vf, abs(vb - vf) / abs(vb))
    @printf("  (11) is symmetric in i, j: max |P̂ᵢⱼ − P̂ⱼᵢ| over 30 pairs %.1e\n\n", sym)
end

# Part 2: exact unbiasedness at n = 36.
println("Part 2: E V̂/V at n = 36 (exact unbiasedness claimed at every n), R = 200000")
let n = d1.n, R = 200000
    for (vname, law, amp) in ((:constant, :normal, 1.0), (:rough, :chisq3, 1.0), (:rough, :chisq3, 4.0))
        σ2 = d1.σs[vname]; sd = sqrt.(σ2)
        f = d1.x .+ amp .* (o1.Pz * g_surf.(d1.u, d1.v))
        p = predict(o1, σ2, f, law)
        rng = MersenneTwister(7)
        rl = zeros(R); rh = zeros(R)
        for r in 1:R
            y = f .+ sd .* draw(rng, law, n)
            rl[r] = vhat_l3o(o1, y) / p.V; rh[r] = vhat_hc2(o1, y) / p.V
        end
        zl = (mean(rl) - 1) / (std(rl) / sqrt(R))
        zh = (mean(rh) - (1 + p.bias / p.V)) / (std(rh) / sqrt(R)); zh1 = (mean(rh) - 1) / (std(rh) / sqrt(R))
        @printf("  %-8s %-6s a=%2.0f | L3O E V̂/V %.4f (z vs 1 %+5.1f) sd %.3f | HC2 %.4f pred %.4f (z vs pred %+5.1f, vs 1 %+6.1f) sd %.3f\n",
                vname, law, amp, mean(rl), zl, std(rl), mean(rh), 1 + p.bias / p.V, zh, zh1, std(rh))
        flush(stdout)
    end
    println()
end

# Part 3: n = 400.
function cell(d, o, vname, law, amp; R = 1000)
    σ2 = d.σs[vname]; sd = sqrt.(σ2)
    f = d.x .+ amp .* (o.Pz * g_surf.(d.u, d.v))
    p = predict(o, σ2, f, law)
    rng = MersenneTwister(3 + 100o.K + round(Int, 10amp) + hash((vname, law)) % 1000)
    rl = zeros(R); rh = zeros(R); cl = 0; ch = 0; co = 0; neg = 0
    for i in 1:R
        y = f .+ sd .* draw(rng, law, d.n)
        g = dot(y, o.C * y)
        vl = vhat_l3o(o, y); vh = vhat_hc2(o, y)
        rl[i] = vl / p.V; rh[i] = vh / p.V; neg += vl <= 0
        cl += vl > 0 && g^2 <= 1.96^2 * vl; ch += vh > 0 && g^2 <= 1.96^2 * vh; co += g^2 <= 1.96^2 * p.V
    end
    zl = (mean(rl) - 1) / (std(rl) / sqrt(R)); zh = (mean(rh) - (1 + p.bias / p.V)) / (std(rh) / sqrt(R))
    @printf("  %-8s %-6s a=%2.0f | lin %.2f | L3O %.4f (z vs 1 %+5.1f) sd %.3f | HC2 %.4f pred %.4f (z %+5.1f) sd %.3f | CS L3O %.3f HC2 %.3f oracle %.3f%s\n",
            vname, law, amp, 1 - p.Vq / p.V, mean(rl), zl, std(rl), mean(rh), 1 + p.bias / p.V, zh, std(rh),
            cl / R, ch / R, co / R, neg > 0 ? @sprintf(" (V̂_L3O ≤ 0 in %d)", neg) : "")
    flush(stdout)
    (; zl, zh, cl = cl / R, ch = ch / R, co = co / R)
end

d = setup(side = 20)
println("Part 3: n = $(d.n), jittered 20 × 20 grid, kNN-6, hat basis + x, R = 1000")
println("lin = the linear part's share of V. MC SE of a coverage near 0.95: 0.007.\n")
let t0 = time(), y = d.x .+ randn(MersenneTwister(1), d.n)
    o = basisobjs(d, 6); vhat_l3o(o, y); t1 = time(); vhat_l3o(o, y)
    @printf("(one V̂_L3O at n = %d: %.2f s)\n\n", d.n, time() - t1)
end
t0 = time(); res = []
for k in (6, 9, 11)
    o = basisobjs(d, k)
    @printf("k = %d: K = %d, K/n = %.3f, max leverage %.3f, min Dᵢⱼₖ %.3e\n", k, o.K, o.K / d.n, o.hmax, min_d3(o))
    if o.hmax > 1 - 1e-8
        println("  max leverage 1: skipped\n"); continue
    end
    for amp in (1.0, 4.0, 16.0), vname in (:constant, :rough), law in (:normal, :chisq3)
        push!(res, cell(d, o, vname, law, amp))
    end
    @printf("  (%.1f min so far)\n\n", (time() - t0) / 60)
end
@printf("Summary over %d cells: L3O |z vs 1| ≤ %.1f; HC2 |z vs Lemma 7| ≤ %.1f; CS coverage L3O %.3f–%.3f, HC2 %.3f–%.3f, oracle %.3f–%.3f\n",
        length(res), maximum(abs(r.zl) for r in res), maximum(abs(r.zh) for r in res),
        extrema(r.cl for r in res)..., extrema(r.ch for r in res)..., extrema(r.co for r in res)...)
