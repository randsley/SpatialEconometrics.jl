# Monte Carlo for root_rho with a penalised smoother (THEORY.md, Proposition 4):
# the exact-diagonal moment y₀′Ay₀, A = MW + MDM, M = I − S, S a tensor P-spline.
#
# The research check (research/trace_cfe/tc_smoother.jl) used one W (kNN-6), one n (900)
# and normal errors. This study measures the package's path over more of the grid.
# With a smoother, E g(ρ₀) = f′Af ≠ 0: the shrinkage bias. Its size, c = f′Af/√V, is
# computed exactly in each cell, and the test-inversion set should then cover at
# P(|Z + c| ≤ z) to first order. So each cell reports:
#   A. coverage of the Wald interval and of CS, the bias and sd of ρ̂, the no-root count;
#   B. c and the predicted coverage P(|Z + c| ≤ 1.96);
#   C. power: the shortcut moment y₀′(M²W + MDM)y₀, which assumes M² = M, on the same
#      draws (its bias and its Wald coverage with the same general V̂);
#   D. root_rho against a dense reimplementation of tc_smoother.jl's formulas on the
#      first draw: |Δρ̂| and |Δse|.
# Design: W in {kNN-6, kNN-20 on stratified random locations, rook on a regular grid};
# n in {400, 900, 1600} with k = 10, 15, 20 knots a side (k²/n = 0.25); λ in {0.01, 0.1, 1};
# f = x + a·sin(2πu)cos(πv) with a in {1, 4}; smooth variances with normal errors, or rough
# variances with standardised χ²₃ errors; ρ₀ = 0.4; R = 1000. 108 cells.
# Pass criterion (set before the run): in every cell with c ≤ 0.25 (predicted coverage
# ≥ 0.943), both intervals cover within 0.93–0.99. Elsewhere the coverage is reported
# against the prediction, not passed or failed: there the smoother is too strong.
#
# Run:    julia --project=. validation/mc_root_rho_smoother.jl [R]   (R = 1000 by default)
#         julia --project=. validation/mc_root_rho_smoother.jl time  (cost per replication)
# Output: validation/mc_root_rho_smoother_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
const RR = SpatialEconometrics.RootRho

function rookW(g)
    n = g^2; I_ = Int[]; J_ = Int[]; id(a, b) = (a - 1) * g + b
    for a in 1:g, b in 1:g, (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        1 <= a + da <= g && 1 <= b + db <= g && (push!(I_, id(a, b)); push!(J_, id(a + da, b + db)))
    end
    W = sparse(I_, J_, 1.0, n, n)
    spdiagm(0 => 1 ./ vec(sum(W, dims = 2))) * W
end

const Φ = SpatialEconometrics.SpatialUtils._normcdf

function design(wtype, n, k, seed)
    rng = MersenneTwister(seed); g = round(Int, sqrt(n)); n = g^2
    if wtype == :rook
        u = repeat(((1:g) .- 0.5) ./ g, inner = g); v = repeat(((1:g) .- 0.5) ./ g, outer = g)
        W = rookW(g)
    else
        P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
        u, v = P[:, 1], P[:, 2]
        W = knn_weights(P, wtype == :knn6 ? 6 : 20)
    end
    (; n, u, v, W, ps = tensor_pspline(u, v; k), x = randn(rng, n), rng)
end

# The dense reference, written from tc_smoother.jl and independent of src/.
function reference(Z, Ωt, λ, Wd)
    G = Z' * Z + λ * Ωt
    S = Z * (G \ Z'); S = (S + S') / 2; M = I - S
    A = M * Wd + M * Diagonal((M .* M) \ (-diag(M * Wd))) * M
    Acut = M * M * Wd + M * Diagonal((M .* M) \ (-diag(M * Wd))) * M
    (; S, M, A, Acut, H = (A + A') * S, m = diag(M * M))
end
function fit_ref(R, C, y, Wy)
    a, b, c = Wy' * C * Wy, y' * C * Wy + Wy' * C * y, y' * C * y
    D = b^2 - 4a * c
    D > 0 || return (NaN, NaN)
    ρ = (b - sqrt(D)) / (2a)
    y0 = y - ρ * Wy; s = (R.M * y0) .^ 2 ./ R.m; Hc = (C + C') * R.S
    V = 2 * s' * (((C + C') ./ 2) .^ 2) * s + sum(((Hc * y0) .^ 2 .- vec(sum(Hc .^ 2 .* s'; dims = 2))) .* s)
    (ρ, sqrt(max(V, 0)) / abs(b - 2ρ * a))
end

function cell(; wtype, n, k, λ, amp, var, R, ρ0 = 0.4, seed = 7, z = 1.959963984540054)
    d = design(wtype, n, k, seed); rng = d.rng; n = d.n
    law = var == :smooth ? :normal : :chisq
    σ2 = var == :smooth ? exp.(3 .* (d.u .+ d.v .- 1)) : exp.(randn(rng, n)); σ2 ./= mean(σ2)
    draw() = law == :normal ? randn(rng, n) : (q = randn(rng, n, 3); (vec(sum(q .^ 2, dims = 2)) .- 3) ./ sqrt(6))
    X = reshape(d.x, :, 1); f = d.x .+ amp .* sin.(2π .* d.u) .* cos.(π .* d.v)
    Z = hcat(X, d.ps.basis); K = size(Z, 2)
    Ωt = zeros(K, K); Ωt[2:end, 2:end] = d.ps.penalty
    Wd = Matrix(d.W); Rf = reference(Z, Ωt, λ, Wd)
    # The exact size of the shrinkage bias. diag A = 0, so V has no cumulant terms
    # and holds for both error laws.
    As = (Rf.A + Rf.A') / 2; bv = (Rf.A + Rf.A') * f
    V = 2 * dot(σ2, (As .^ 2) * σ2) + sum(bv .^ 2 .* σ2)
    c = dot(f, Rf.A * f) / sqrt(V); pred = Φ(z - c) - Φ(-z - c)
    Fac = lu(sparse(I - ρ0 * d.W))
    ρs = Float64[]; ρc = Float64[]
    covW = 0; covCS = 0; lenCS = 0.0; noroot = 0; touch = 0; split = 0; covC = 0
    Δρ = NaN; Δse = NaN; df = NaN; maxS = NaN
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* draw()); Wy = d.W * y
        res = root_rho(y, X, d.W; d.ps..., lambda = λ)
        df = res.df; maxS = res.max_leverage
        if res.root_found
            push!(ρs, res.rho)
            covW += abs(res.rho - ρ0) <= z * res.se
        else
            noroot += 1
        end
        covCS += any(lo <= ρ0 <= hi for (lo, hi) in res.cs)
        lenCS += sum((hi - lo for (lo, hi) in res.cs); init = 0.0)
        touch += res.cs_touches_bound; split += res.cs_split
        ρk, sek = fit_ref(Rf, Rf.Acut, y, Wy)       # power: the shortcut moment
        if isfinite(ρk)
            push!(ρc, ρk); covC += abs(ρk - ρ0) <= z * sek
        end
        if r == 1 && res.root_found
            ρref, seref = fit_ref(Rf, Rf.A, y, Wy)
            Δρ = abs(ρref - res.rho); Δse = abs(seref - res.se)
        end
    end
    @printf("| %-5s | %4d | %2d | %5.2f | %5.1f | %.3f | %.3f | %d | %-6s | %-6s | %+.3f | %.3f | %+.4f | %.4f | %.3f | %.3f | %.3f | %d | %d | %d | %+.4f | %.3f | %.1e | %.1e |\n",
            wtype, n, k, λ, df, df / n, maxS, amp, var, law, c, pred, mean(ρs) - ρ0, std(ρs),
            covW / length(ρs), covCS / R, lenCS / R, noroot, touch, split,
            mean(ρc) - ρ0, covC / length(ρc), Δρ, Δse)
    flush(stdout)
end

ks = Dict(400 => 10, 900 => 15, 1600 => 20)
if get(ARGS, 1, "") == "time"
    for n in (400, 900, 1600)
        cell(; wtype = :knn6, n, k = ks[n], λ = 0.01, amp = 1.0, var = :rough, R = 2)   # compile
        t = @elapsed cell(; wtype = :knn6, n, k = ks[n], λ = 0.01, amp = 1.0, var = :rough, R = 10)
        println("n $n: $(round(t / 10, digits = 3)) s per replication")
    end
else
    R = parse(Int, get(ARGS, 1, "1000"))
    redirect_stdout(devnull) do   # warm-up, not data
        cell(; wtype = :knn6, n = 400, k = 10, λ = 0.1, amp = 1.0, var = :smooth, R = 3)
    end
    println("\nR = $R per cell. MC SE of a coverage near 0.95: $(round(sqrt(0.95 * 0.05 / R), digits = 4)).")
    println("Columns: k knots a side; df = tr S; c = f′Af/√V, exact; pred = P(|Z + c| ≤ 1.96);")
    println("bias and sd of ρ̂ over the replications with a root; Wald coverage over the same;")
    println("CS coverage and mean total length over all R; counts of no-root, CS touching a bound, CS split;")
    println("the shortcut moment's bias and Wald coverage (power); |Δρ̂| and |Δse| against the reference on the first draw.\n")
    println("| W | n | k | λ | df | df/n | max S | a | σ² | errors | c | pred | bias | sd | Wald | CS | CS length | no root | touches | split | shortcut bias | shortcut Wald | Δρ̂ | Δse |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    t0 = time()
    for wtype in (:knn6, :rook, :knn20), n in (400, 900, 1600), λ in (0.01, 0.1, 1.0), amp in (1.0, 4.0), var in (:smooth, :rough)
        cell(; wtype, n, k = ks[n], λ, amp, var, R)
    end
    println("\ntotal $(round((time() - t0) / 60, digits = 1)) min")
end
