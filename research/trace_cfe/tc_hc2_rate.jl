# The rate of the HC2 V̂'s worst-case bias (THEORY.md, Lemma 9) and its concentration (Lemma 10).
#
# Part 1, Lemma 9. With h = diag(P_B), r = h ./ diag(M) and h̄ = max h, the worst case of Lemma 8 obeys
#     |β_min| ≤ β̄ = [4‖C′P_B‖_F² + 4(1 + h̄) r′Ωr] / (2·1′Ω1),
# and β̄ ≤ (K/n)·[4‖C‖² + 4(1 + h̄)h̄‖Ω‖_∞/(1 − h̄)²] / (V/n), so β_min = O(K/n) when h̄ ≤ 1 − δ.
# Reported per design: β_min, β̄, β̄/|β_min| (the looseness) and β̄/(K/n). Check: β̄ ≥ |β_min| in every
# design (a bound that failed once would be wrong).
#
# Part 2, Lemma 10. Var V̂ = O(n), so sd(V̂/V) = O(n^(-1/2)). Jittered grids with n = 400, 900, 1600 at
# K/n ≈ 0.09 and ≈ 0.19 (hat bases k = 6, 9, 12 and 9, 13, 17), kNN-6 W, f = x + P_B g; variances
# constant, bounded (iid U(0.2, 1.8)) and log-normal (unbounded as n grows, so outside A1: the
# counter-example); normal and χ²₃ errors, R = 2000. Reported: √n·sd(V̂/V), and the slope of
# log sd(V̂/V) on log n, which should be near −1/2. Power: the same slope for g²/V, which does not
# concentrate (its sd is √2 for every n), should be near 0.
#
# Run: julia --project=. research/trace_cfe/tc_hc2_rate.jl > research/trace_cfe/tc_hc2_rate_output.txt
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
    (; n, Wd = Matrix(W), u = P[:, 1], v = P[:, 2], x = randn(rng, n),
       rough = (s = exp.(randn(rng, n)); s ./ mean(s)), bounded = 0.2 .+ 1.6 .* rand(rng, n))
end

function objs(d, k)
    Z = hcat(hatbasis(d.u, d.v, k), d.x); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]
    Pz = Q * Q'; M = I - Pz; Mii = diag(M)
    maximum(1 .- Mii) > 1 - 1e-8 && return nothing
    G = d.Wd / (I - 0.4 * d.Wd)
    E = G' .- Diagonal(diag(G' * M) ./ Mii); C = E * M
    Ω = symm(C) .^ 2; H = (C + C') * Pz; H2 = H .^ 2; M2 = M .^ 2
    Q2 = 4 * sum(Ω .* M2 ./ (Mii .* Mii'))
    L5G = -2 * sum(H2 .* M2 ./ (Mii .* Mii'))
    U = M2 ./ Mii
    c = vec(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .- sum(U .* (H2 * U); dims = 1))
    (; K = size(Q, 2), Pz, M, Mii, C, Ω, H, H2, Q2, L5G, c, hmax = maximum(1 .- Mii))
end

# ---------------------------------------------------------------- Part 1: Lemma 9
println("Part 1. Lemma 9: the worst case β_min (Lemma 8) against its bound β̄; n = 900\n")
ok1 = true; looseness = Float64[]; slopes = Float64[]
for (kind, ks) in ((:jittered, (9, 13, 16, 19)), (:random, (7, 10, 12)), (:rook, (9, 13, 16)))
    d = design(kind)
    for k in ks
        o = objs(d, k); o === nothing && continue
        V0 = 2 * sum(o.Ω)
        βmin = min(0.0, o.Q2 + o.L5G - 2 * sum(o.c)) / V0
        h = 1 .- o.Mii; r = h ./ o.Mii
        HF = sum(abs2, o.H)                     # ‖(C + C′)P‖_F² = ‖C′P‖_F²
        CP = sum(abs2, o.C' * o.Pz)
        bH = 4HF / V0; bΩ = 4 * (1 + o.hmax) * dot(r, o.Ω * r) / V0
        βbar = bH + bΩ
        global ok1 &= βbar >= abs(βmin)
        push!(looseness, βbar / max(abs(βmin), eps())); push!(slopes, βbar / (o.K / d.n))
        @printf("%-8s k = %2d: K/n %.3f, max leverage %.3f | β_min %+.4f | β̄ %.4f (H part %.4f, Ω part %.4f) | β̄/|β_min| %.1f | β̄/(K/n) %.2f | check ‖H‖_F² = ‖C′P‖_F²: %.1e\n",
                kind, k, o.K / d.n, o.hmax, βmin, βbar, bH, bΩ, βbar / abs(βmin), βbar / (o.K / d.n),
                abs(HF - CP) / CP)
    end
end
@printf("\nβ̄ ≥ |β_min| in every design: %s; β̄/|β_min| %.1f–%.1f; β̄/(K/n) %.2f–%.2f\n\n",
        ok1 ? "yes" : "NO", minimum(looseness), maximum(looseness), minimum(slopes), maximum(slopes))

# ---------------------------------------------------------------- Part 2: Lemma 10
draw(rng, law, n) = law == :normal ? randn(rng, n) : (rand(rng, Chisq(3), n) .- 3) ./ sqrt(6)
g_surf(u, v) = sin(2π * u) * cos(π * v)

function concentration(d, o, vname, law; R = 2000, seed = 3)
    rng = MersenneTwister(seed + o.K)
    σ2 = vname == :constant ? ones(d.n) : vname == :bounded ? d.bounded : d.rough
    σ = sqrt.(σ2)
    f = d.x .+ o.Pz * g_surf.(d.u, d.v)
    b = o.C' * f
    V = 2 * dot(σ2, o.Ω * σ2) + sum(b .^ 2 .* σ2)
    rv = zeros(R); rg = zeros(R)
    for i in 1:R
        y0 = f .+ σ .* draw(rng, law, d.n)
        û = o.M * y0; s = û .^ 2 ./ o.Mii
        Vh = 2 * dot(s, o.Ω * s) + sum(((o.H * y0) .^ 2 .- o.H2 * s) .* s)
        g = dot(y0, o.C * y0)
        rv[i] = Vh / V; rg[i] = g^2 / V
    end
    (; sdv = std(rv), sdg = std(rg))
end

slope(x, y) = (X = hcat(ones(length(x)), log.(x)); (X \ log.(y))[2])

println("Part 2. Lemma 10: sd(V̂/V) as n grows at fixed K/n; R = 2000 per cell")
println("bounded = iid U(0.2, 1.8) variances (A1); lognormal = iid log-normal, unbounded as n grows (outside A1)\n")
designs = Dict(side => design(:jittered; side) for side in (20, 30, 40))
sv = Float64[]; sg = Float64[]
t0 = time()
for (ks, lab) in (((6, 9, 12), "K/n ≈ 0.09"), ((9, 13, 17), "K/n ≈ 0.19"))
    os = [objs(designs[side], k) for (side, k) in zip((20, 30, 40), ks)]
    # Lemma 10's local-basis condition: ‖P_B‖_∞, the largest absolute row sum, bounded in n.
    @printf("%s: ‖P_B‖_∞ = %s for n = 400, 900, 1600\n", lab,
            join((@sprintf("%.2f", maximum(sum(abs, o.Pz; dims = 2))) for o in os), ", "))
    for vname in (:constant, :bounded, :rough), law in (:normal, :chisq3)
        res = [concentration(designs[side], o, vname, law) for (side, o) in zip((20, 30, 40), os)]
        ns = [designs[side].n for side in (20, 30, 40)]
        a = slope(ns, [r.sdv for r in res]); bpow = slope(ns, [r.sdg for r in res])
        vname == :rough || push!(sv, a); push!(sg, bpow)
        @printf("%s %-9s %-6s | K/n %s | max leverage %s | √n·sd(V̂/V) %s | slope %+.2f | power: slope for g²/V %+.2f\n",
                lab, vname == :rough ? :lognormal : vname, law,
                join((@sprintf("%.3f", o.K / dd.n) for (o, dd) in zip(os, (designs[20], designs[30], designs[40]))), " "),
                join((@sprintf("%.3f", o.hmax) for o in os), " "),
                join((@sprintf("%.2f", sqrt(n) * r.sdv) for (n, r) in zip(ns, res)), " "), a, bpow)
    end
    @printf("  (%.1f min so far)\n", (time() - t0) / 60)
end
@printf("\nslope of log sd(V̂/V) on log n, constant and bounded variances: %+.2f to %+.2f (Lemma 10: −1/2); for g²/V: %+.2f to %+.2f (0)\n",
        minimum(sv), maximum(sv), minimum(sg), maximum(sg))
