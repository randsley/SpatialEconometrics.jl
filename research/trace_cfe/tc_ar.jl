# Proposition 6 (THEORY.md): an interval that does not need identification, by inverting the moment test.
#
# Theorem 1 gives g(ρ₀)/√V → N(0, 1) without A9 (it never uses J), and Proposition 2's steps 1–2 give
# V̂(ρ₀)/V → 1 at ρ₀ without the root. So the set CS = {ρ : g(ρ)² ≤ 1.96² V̂(ρ)}, with V̂(ρ) the HC2 V̂
# evaluated at the candidate ρ, has coverage → 0.95 whatever J is: an Anderson–Rubin-type interval.
# Its coverage is the test's acceptance rate at ρ₀, exactly. g(ρ) = c − bρ + aρ² is quadratic, and V̂(ρ) is
# a quartic: s(ρ) = (e₀ − ρe₁)²/Mᵢᵢ is quadratic, â(ρ) = α₀ + ρα₁ linear, and the noise term is linear in s.
# So CS is found exactly on a grid of (−0.99, 0.99), and may be a union of intervals or unbounded (reaching
# both ends of the grid).
# Cells: the seven weakly identified cells of tc_coverage_robust.jl and six well-identified controls, with
# normal and χ²₃ errors, R = 1000. Reported: Wald coverage (HC2, at ρ̂), test-inversion coverage (HC2 and
# oracle V), the mean length of CS within (−0.99, 0.99), the share of CS that are not one interval, and the
# share reaching both ends. Power check: the test-inversion set must be shorter than the whole range where
# identification is strong, and the Wald interval must under-cover where it is weak.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function bandW(P, kbar)
    n = size(P, 1); tree = KDTree(permutedims(P))
    _, dk = knn(tree, permutedims(P), kbar + 1, true)
    idx = inrange(tree, permutedims(P), mean(last.(dk)))
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n, j in idx[i]
        j == i && continue
        push!(I, i); push!(J, j); push!(V, 1 / norm(P[i, :] .- P[j, :]))
    end
    W = sparse(I, J, V, n, n); spdiagm(0 => 1 ./ vec(sum(W, dims = 2))) * W
end
function rookW(g)
    n = g^2; I = Int[]; J = Int[]; id(a, b) = (a - 1) * g + b
    for a in 1:g, b in 1:g, (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        1 <= a + da <= g && 1 <= b + db <= g && (push!(I, id(a, b)); push!(J, id(a + da, b + db)))
    end
    W = sparse(I, J, 1.0, n, n); spdiagm(0 => 1 ./ vec(sum(W, dims = 2))) * W
end

noiseof(D, s) = vec(sum((D.G * (D.Q' * (s .* D.Q))) .* D.G, dims = 2))
Pproj(D, z) = D.Q * (D.Q' * z)

const GRID = collect(-0.99:0.002:0.99)

function cell(; n, wtype, kh, var, law = :normal, R = 1000, ρ0 = 0.4, seed = 5)
    rng = MersenneTwister(seed)
    if wtype == :rook
        g = round(Int, sqrt(n)); n = g^2
        u = repeat(((1:g) .- 0.5) ./ g, inner = g); v = repeat(((1:g) .- 0.5) ./ g, outer = g); W = rookW(g)
    else
        g = round(Int, sqrt(n)); n = g^2
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        u, v = P[:, 1], P[:, 2]
        W = wtype == :knn6 ? knnW(P, 6) : wtype == :knn20 ? knnW(P, 20) : bandW(P, 10)
    end
    x = randn(rng, n)
    D = tc_design(hcat(hatbasis(u, v, kh), x), W)
    f = D.Q * (D.Q' * (sin.(2π .* u) .* cos.(π .* v))) .+ x
    σ2 = var == :smooth ? exp.(3 .* (u .+ v .- 1)) : exp.(randn(rng, n)); σ2 ./= mean(σ2)
    Fac = lu(sparse(I - ρ0 * W))
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    qerr = 0.0
    covW = 0; covT = 0; covO = 0; len = 0.0; nsplit = 0; nunb = 0; ρs = zeros(R)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* draw())
        Wy = W * y; W2y = W * Wy
        e0, e1, e2 = Mproj(D, y), Mproj(D, Wy), Mproj(D, W2y)
        f1 = e1 .+ D.d .* e0; f2 = e2 .+ D.d .* e1
        ρ, a, b = cfe_root(e0, f1, e1, f2); ρs[r] = ρ
        c = dot(e0, f1)
        # V̂(ρ) as a quartic: s = s0 + ρs1 + ρ²s2, â = α0 + ρα1, noise linear in s
        s0 = e0 .^ 2 ./ D.Mii; s1 = -2 .* e0 .* e1 ./ D.Mii; s2 = e1 .^ 2 ./ D.Mii
        α0 = Mproj(D, W * Pproj(D, y)); α1 = -Mproj(D, W * Pproj(D, Wy))
        S = (s0, s1, s2); N = map(s -> noiseof(D, s), S); ΩS = map(s -> D.As2 * s, S)
        Acoef = (α0 .^ 2 .- N[1], 2 .* α0 .* α1 .- N[2], α1 .^ 2 .- N[3])
        p = zeros(5)
        for k in 0:2, l in 0:2
            p[k+l+1] += 2 * dot(S[k+1], ΩS[l+1]) + dot(Acoef[k+1], S[l+1])
        end
        Vh(t) = p[1] + t * (p[2] + t * (p[3] + t * (p[4] + t * p[5])))
        gq(t) = c - b * t + a * t^2
        if r == 1                                          # the quartic against the direct V̂, at ρ̂ and ρ₀
            for t in (ρ, ρ0)
                y0t = y .- t .* Wy; ut = Mproj(D, y0t); st = ut .^ 2 ./ D.Mii; at = Mproj(D, W * (y0t .- ut))
                Vd = 2 * dot(st, D.As2 * st) + sum((at .^ 2 .- noiseof(D, st)) .* st)
                qerr = max(qerr, abs(Vh(t) - Vd) / abs(Vd))
            end
        end
        # Wald with HC2 at ρ̂
        Vr = Vh(ρ)
        covW += Vr > 0 && abs(ρ - ρ0) <= 1.96 * sqrt(Vr) / abs(b - 2ρ * a)
        # test inversion: coverage is acceptance at ρ₀
        covT += gq(ρ0)^2 <= 1.96^2 * Vh(ρ0)
        # oracle V at ρ₀ (true σ²)
        y0 = y .- ρ0 .* Wy; aa = Mproj(D, W * Pproj(D, y0))
        Vo = 2 * dot(σ2, D.As2 * σ2) + sum((aa .^ 2 .- noiseof(D, σ2)) .* σ2)
        covO += gq(ρ0)^2 <= 1.96^2 * Vo
        acc = [gq(t)^2 <= 1.96^2 * Vh(t) for t in GRID]
        len += count(acc) * 0.002
        runs = count(i -> acc[i] && (i == 1 || !acc[i-1]), eachindex(acc))
        nsplit += runs > 1
        nunb += acc[1] && acc[end]
    end
    @printf("%-6s n %4d %-6s %-6s K/n %.3f ‖Cs‖F² %6.1f sd ρ̂ %.3f | Wald %.3f | inversion: HC2 %.3f oracle %.3f | mean length %.3f, not one interval %.3f, both ends %.3f | quartic check %.1e\n",
            wtype, n, var, law, D.K / n, sum(D.As2), std(ρs), covW / R, covT / R, covO / R, len / R, nsplit / R, nunb / R, qerr)
end

weak = [(400, :knn20, 7, :smooth), (400, :knn20, 9, :smooth), (400, :knn20, 11, :smooth),
        (400, :knn20, 9, :rough), (400, :knn20, 11, :rough), (1600, :knn20, 22, :rough), (400, :band, 11, :smooth)]
ctrl = [(400, :knn6, 7, :smooth), (900, :knn6, 10, :smooth), (1600, :knn6, 13, :smooth),
        (900, :rook, 13, :rough), (900, :band, 10, :smooth), (900, :knn20, 13, :smooth)]
cell(; n = 400, wtype = :knn6, kh = 7, var = :smooth, R = 5); println("(warm-up above) R = 1000, MC SE 0.007.\n")
t0 = time()
for law in (:normal, :chisq)
    println("weakly identified ($law):")
    for (n, w, kh, var) in weak; cell(; n, wtype = w, kh, var, law); end
    println("controls ($law):")
    for (n, w, kh, var) in ctrl; cell(; n, wtype = w, kh, var, law); end
    println()
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
