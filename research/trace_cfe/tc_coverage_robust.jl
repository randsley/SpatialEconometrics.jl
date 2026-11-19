# Step 2, the untested designs (THEORY.md "Still to do" item 3): HC2 coverage for other W, other n and K/n
# up to 0.30.
#
# tc_coverage_vhat.jl measured the coverage of ρ̂ ± 1.96 √V̂/|g′(ρ̂)| at n = 900 with a kNN W (k = 6) and
# K/n ≤ 0.25. Here, the same estimator (the exact-diagonal TC-CFE at its feasible root, tensor hat basis
# plus one covariate) with:
#   W:   knn6 and knn20 on stratified random locations (one per cell of a √n × √n grid, jittered uniformly
#        within it, so no basis cell is empty; uniform locations put leverage 1 in the basis at n = 1600); rook contiguity on a regular grid; a distance band
#        (inverse-distance weights within the radius that gives 10 neighbours on average, row-standardised,
#        so the neighbour count varies);
#   n:   400, 900, 1600;
#   K/n: about 0.05, 0.11, 0.19, 0.30;
#   σ²:  smooth and rough; normal errors; the weak signal of tc_vhat.jl; ρ₀ = 0.4; R = 1000.
# Reported: oracle and HC2 coverage, negative V̂ counts. Power check: the homoskedastic V̂ (pooled σ̂²) under
# rough variances. A design whose M∘M is singular, or whose max hᵢᵢ exceeds 0.99 (A3 fails), is skipped with
# a line saying so.
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function rookW(g)
    n = g^2; I = Int[]; J = Int[]; id(a, b) = (a - 1) * g + b
    for a in 1:g, b in 1:g, (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        1 <= a + da <= g && 1 <= b + db <= g && (push!(I, id(a, b)); push!(J, id(a + da, b + db)))
    end
    W = sparse(I, J, 1.0, n, n)
    spdiagm(0 => 1 ./ vec(sum(W, dims = 2))) * W
end

function bandW(P, kbar)
    n = size(P, 1); tree = KDTree(permutedims(P))
    _, dk = knn(tree, permutedims(P), kbar + 1, true)
    radius = mean(last.(dk))                                  # gives about kbar neighbours on average
    idx = inrange(tree, permutedims(P), radius)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n, j in idx[i]
        j == i && continue
        push!(I, i); push!(J, j); push!(V, 1 / norm(P[i, :] .- P[j, :]))
    end
    W = sparse(I, J, V, n, n)
    rs = vec(sum(W, dims = 2)); any(rs .== 0) && error("isolated unit in the distance band")
    spdiagm(0 => 1 ./ rs) * W
end

function vparts(D, W, y, ρ, s)
    y0 = y .- ρ .* (W * y)
    u = Mproj(D, y0)
    a = Mproj(D, W * (y0 .- u))
    noise = vec(sum((D.G * (D.Q' * (s .* D.Q))) .* D.G, dims = 2))
    2 * dot(s, D.As2 * s) + sum((a .^ 2 .- noise) .* s)
end

function cell(; n = 900, wtype = :knn6, kh = 10, R = 1000, var = :rough, ρ0 = 0.4, seed = 5)
    rng = MersenneTwister(seed)
    if wtype == :rook
        g = round(Int, sqrt(n)); n = g^2
        u = repeat(((1:g) .- 0.5) ./ g, inner = g); v = repeat(((1:g) .- 0.5) ./ g, outer = g)
        W = rookW(g)
    else
        g = round(Int, sqrt(n)); n = g^2                     # stratified: one point per grid cell, jittered
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        u, v = P[:, 1], P[:, 2]
        W = wtype == :knn6 ? knnW(P, 6) : wtype == :knn20 ? knnW(P, 20) : bandW(P, 10)
    end
    x = randn(rng, n)
    D = try
        tc_design(hcat(hatbasis(u, v, kh), x), W)
    catch e
        e isa SingularException || rethrow()
        @printf("%-6s n %4d %-6s K %3d K/n %.3f | skipped: M∘M singular\n", wtype, n, var, kh^2 + 1, (kh^2 + 1) / n)
        return
    end
    if maximum(1 .- D.Mii) > 0.99
        @printf("%-6s n %4d %-6s K %3d K/n %.3f | skipped: max hᵢᵢ = %.4f > 0.99, A3 fails
", wtype, n, var, D.K, D.K / n, maximum(1 .- D.Mii))
        return
    end
    f = D.Q * (D.Q' * (sin.(2π .* u) .* cos.(π .* v))) .+ x
    σ2 = var == :smooth ? exp.(3 .* (u .+ v .- 1)) : exp.(randn(rng, n)); σ2 ./= mean(σ2)
    Fac = lu(sparse(I - ρ0 * W))
    ks = (:oracle, :hc2, :homo)
    cov = Dict(k => 0 for k in ks); neg = Dict(k => 0 for k in ks); ρs = zeros(R)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n))
        Wy = W * y; W2y = W * Wy
        e0, e1, e2 = Mproj(D, y), Mproj(D, Wy), Mproj(D, W2y)
        ρ, a, b = cfe_root(e0, e1 .+ D.d .* e0, e1, e2 .+ D.d .* e1)
        ρs[r] = ρ
        y0 = y .- ρ .* Wy; û = Mproj(D, y0)
        hc2 = û .^ 2 ./ D.Mii
        for (k, s) in ((:oracle, σ2), (:hc2, hc2), (:homo, fill(mean(hc2), n)))
            V = vparts(D, W, y, ρ, s)
            if V <= 0
                neg[k] += 1
            elseif abs(ρ - ρ0) <= 1.96 * sqrt(V) / abs(b - 2ρ * a)
                cov[k] += 1
            end
        end
    end
    @printf("%-6s n %4d %-6s K %3d K/n %.3f maxh %.3f ‖Cs‖F²/n %.3f | bias %+.4f sd %.4f |", wtype, n, var, D.K,
            D.K / n, maximum(1 .- D.Mii), sum(D.As2) / n, mean(ρs) - ρ0, std(ρs))
    for k in ks
        @printf(" %s %.3f%s", k, cov[k] / R, neg[k] > 0 ? "(neg $(neg[k]))" : "")
    end
    println()
end

khs = Dict(400 => (4, 7, 9, 11), 900 => (7, 10, 13, 16), 1600 => (9, 13, 17, 22))
if get(ARGS, 1, "") == "time"
    for n in (400, 900, 1600)
        t = @elapsed cell(; n, kh = khs[n][end], R = 20)
        println("n $n, largest K, 20 replications: $(round(t, digits = 1)) s")
    end
else
    cell(; R = 5); println("(warm-up above) Coverage of the 95% interval, R = 1000 (MC SE 0.007).\n")
    t0 = time()
    for wtype in (:knn6, :knn20, :rook, :band), n in (400, 900, 1600), var in (:smooth, :rough)
        for kh in khs[n]
            cell(; n, wtype, kh, var)
        end
    end
    println("total $(round((time() - t0) / 60, digits = 1)) min")
end
