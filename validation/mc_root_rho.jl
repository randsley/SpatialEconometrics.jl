# Monte Carlo for root_rho (Jin & Lee's REsd with a projection on a growing
# basis): coverage of the Wald interval and of the test-inversion set CS.
#
# The theory (research/trace_cfe/THEORY.md) was checked by Monte Carlo only for
# the exact-diagonal moment. root_rho uses Jin & Lee's REsd, whose moment
# C(ρ̃) depends on a first-step estimate, and plugs the same C(ρ̃) into CS.
# This study measures that combination. Three parts, on the same draws:
#   A. coverage of the Wald interval and of CS, the bias and sd of ρ̂, and the
#      cells where no root exists;
#   B. root_rho against the research implementation (tc_resd.jl's moment_mats
#      and fitC, copied below) on the first draw of each cell: |Δρ̂| and |Δse|;
#   C. power: closed_form_rho (Smirnov's plain closed form, no trace
#      correction) on the same draws must show the O(K/n) bias, and root_rho
#      must not.
# Design (a subset of research/trace_cfe/tc_coverage_robust.jl): W in {kNN-6,
# kNN-20 on stratified random locations, rook on a regular grid}; n in {400,
# 900, 1600}; a tensor hat basis with K/n about 0.05, 0.11, 0.19, plus one
# covariate x with β = 1; f = the projection of sin(2πu)cos(πv) on the basis;
# smooth or rough variances; normal or standardised χ²₃ errors; ρ₀ = 0.4.
#
# Run:    julia --project=. validation/mc_root_rho.jl [R]      (R = 1000 by default)
#         julia --project=. validation/mc_root_rho.jl time     (cost per fit only)
# Output: validation/mc_root_rho_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function rookW(g)
    n = g^2; I_ = Int[]; J_ = Int[]; id(a, b) = (a - 1) * g + b
    for a in 1:g, b in 1:g, (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        1 <= a + da <= g && 1 <= b + db <= g && (push!(I_, id(a, b)); push!(J_, id(a + da, b + db)))
    end
    W = sparse(I_, J_, 1.0, n, n)
    spdiagm(0 => 1 ./ vec(sum(W, dims = 2))) * W
end

# ── The research implementation (research/trace_cfe/tc_resd.jl), dense ──
function cfe_root_ref(e0, f1, e1, f2)
    a = dot(e1, f2); b = dot(e0, f2) + dot(e1, f1); c = dot(e0, f1)
    (b - sqrt(b^2 - 4a * c)) / (2a), a, b
end
function moment_mats(A, M, Mii)
    AM = A * M; δ = diag(AM) ./ Mii
    (; Ct = A' .- Diagonal(δ), C = AM .- δ .* M)
end
function fitC(mm, y, Wy, Q, Mii)
    e0 = y .- Q * (Q' * y); e1 = Wy .- Q * (Q' * Wy)
    ρ, a, b = cfe_root_ref(e0, mm.Ct * y, e1, mm.Ct * Wy)
    y0 = y .- ρ .* Wy
    u = y0 .- Q * (Q' * y0); s2 = u .^ 2 ./ Mii
    S = mm.C + mm.C'; H = S * Q
    â = S * (y0 .- u)
    noise = vec(sum((H * (Q' * (s2 .* Q))) .* H, dims = 2))
    V = 2 * dot(s2, ((S ./ 2) .^ 2) * s2) + sum((â .^ 2 .- noise) .* s2)
    (ρ, sqrt(max(V, 0)) / abs(b - 2ρ * a))
end
function research_fit(y, W, Z)
    Wd = Matrix(W); Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]; M = I - Q * Q'; Mii = diag(M); Wy = W * y
    ρt, _ = fitC(moment_mats(Matrix(Wd'), M, Mii), y, Wy, Q, Mii)
    fitC(moment_mats(Matrix((Wd / (I - ρt * Wd))'), M, Mii), y, Wy, Q, Mii)
end

function design(wtype, n, kh, seed)
    rng = MersenneTwister(seed); g = round(Int, sqrt(n)); n = g^2
    if wtype == :rook
        u = repeat(((1:g) .- 0.5) ./ g, inner = g); v = repeat(((1:g) .- 0.5) ./ g, outer = g)
        W = rookW(g)
    else
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        u, v = P[:, 1], P[:, 2]
        W = knn_weights(P, wtype == :knn6 ? 6 : 20)
    end
    B = hatbasis(u, v, kh); x = randn(rng, n)
    Qb = Matrix(qr(B).Q)[:, 1:size(B, 2)]
    f = Qb * (Qb' * (sin.(2π .* u) .* cos.(π .* v))) .+ x
    (; n, u, v, W, B, x, f, rng)
end

function cell(; wtype, n, kh, var, law, R, ρ0 = 0.4, seed = 7)
    d = design(wtype, n, kh, seed); rng = d.rng; n = d.n
    σ2 = var == :smooth ? exp.(3 .* (d.u .+ d.v .- 1)) : exp.(randn(rng, n)); σ2 ./= mean(σ2)
    draw() = law == :normal ? randn(rng, n) : (z = randn(rng, n, 3); (vec(sum(z .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    Fac = lu(sparse(I - ρ0 * d.W)); X = reshape(d.x, :, 1)
    ρs = Float64[]; cfs = zeros(R)
    covW = 0; covCS = 0; lenCS = 0.0; noroot = 0; touch = 0; split = 0; Δρ = NaN; Δse = NaN; K = 0; maxh = 0.0
    for r in 1:R
        y = Fac \ (d.f .+ sqrt.(σ2) .* draw())
        res = root_rho(y, X, d.W; basis = d.B)
        K = res.K; maxh = res.max_leverage
        if res.root_found
            push!(ρs, res.rho)
            covW += abs(res.rho - ρ0) <= 1.959963984540054 * res.se
        else
            noroot += 1
        end
        covCS += any(lo <= ρ0 <= hi for (lo, hi) in res.cs)
        lenCS += sum((hi - lo for (lo, hi) in res.cs); init = 0.0)
        touch += res.cs_touches_bound; split += res.cs_split
        cfs[r] = closed_form_rho(y, hcat(X, d.B), d.W).rho
        if r == 1 && res.root_found
            ρref, seref = research_fit(y, d.W, hcat(X, d.B))
            Δρ = abs(ρref - res.rho); Δse = abs(seref - res.se)
        end
    end
    @printf("| %-5s | %4d | %-6s | %-6s | %3d | %.3f | %.3f | %+.4f | %.4f | %.3f | %.3f | %.3f | %d | %d | %d | %+.4f | %.1e | %.1e |\n",
            wtype, n, var, law, K, K / n, maxh, mean(ρs) - ρ0, std(ρs), covW / length(ρs), covCS / R,
            lenCS / R, noroot, touch, split, mean(cfs) - ρ0, Δρ, Δse)
    flush(stdout)
end

khs = Dict(400 => (4, 7, 9), 900 => (7, 10, 13), 1600 => (9, 13, 17))
if get(ARGS, 1, "") == "time"
    for n in (400, 900, 1600)
        cell(; wtype = :knn6, n, kh = khs[n][end], var = :rough, law = :normal, R = 2)  # compile
        t = @elapsed cell(; wtype = :knn6, n, kh = khs[n][end], var = :rough, law = :normal, R = 10)
        println("n $n, largest K: $(round(t / 10, digits = 3)) s per replication")
    end
else
    R = parse(Int, get(ARGS, 1, "1000"))
    cell(; wtype = :knn6, n = 400, kh = 4, var = :smooth, law = :normal, R = 3)   # warm-up, not data
    println("\nR = $R per cell. MC SE of a coverage near 0.95: $(round(sqrt(0.95 * 0.05 / R), digits = 4)).")
    println("Columns: bias and sd of ρ̂ over the replications with a root; Wald coverage over the same;")
    println("CS coverage and mean total length over all R; counts of no-root, CS touching a bound, CS split;")
    println("bias of closed_form_rho (power); |Δρ̂| and |Δse| against the research code on the first draw.\n")
    println("| W | n | σ² | errors | K | K/n | max h | bias | sd | Wald | CS | CS length | no root | touches | split | plain CF bias | Δρ̂ | Δse |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    t0 = time()
    for wtype in (:knn6, :rook, :knn20), n in (400, 900, 1600), kh in khs[n], var in (:smooth, :rough), law in (:normal, :chisq)
        cell(; wtype, n, kh, var, law, R)
    end
    println("\ntotal $(round((time() - t0) / 60, digits = 1)) min")
end
