# Proposition 1′ (THEORY.md): Jin & Lee's REsd when f ∉ span(B).
#
# Proposition 1 proves ρ̂ − ρ̂* = o_p(n^{−1/2}) (ρ̂* the infeasible root with C(ρ₀)) under Mf = 0. Its
# step 2 uses Mf = 0 only to make E η′C_ϱ(ρ₀)η = f′C_ϱf vanish. Without it, the plug-in adds
# (ρ̃ − ρ₀)·f′C_ϱf to g₀, and |f′C_ϱf| ≤ c‖f‖‖Mf‖, so the plug-in still costs nothing when
# ‖Mf‖ = o(√n). Then J*(ρ̂ − ρ₀)/√V* → N(c*, 1), c* = f′C(ρ₀)f/√V* (Proposition 3 for C(ρ₀)).
#
# Part 1, the estimator (tc_resd.jl's design, but f = 2·sin(2πu)cos(πv) itself, not its projection):
#   arm A: k = round(√(2√n)) knots a side, so ‖Mf‖/√n falls with n: sd(ρ̂ − ρ̂*)/sd(ρ̂*) should fall,
#          the bias of ρ̂ should be β*/J*, and the coverage P(|Z + c*| ≤ 1.96);
#   arm B (power): k = 4 at every n, so ‖Mf‖/√n stays put: the gap ratio should NOT fall, and should
#          approach its first-order prediction sd(ρ̃)·|d(β/J)/dϱ| / sd(ρ̂*), with β(ϱ) and J(ϱ) those of
#          C(ϱ). In the local regime β = O(√n) this is sd(ρ̃)·|f′C_ϱf|/J* to first order; in arm B β is
#          of order n, and the change in J(ϱ) matters as much.
#   Also (β(ρ̃) − β*)/√V*, the change in the bias the package bounds (at ϱ = ρ̃) against the one that
#   biases ρ̂ (at ρ₀); Proposition 1′(c) says it is O_p(‖Mf‖/√n).
#   β*, J*, V* (normal errors, so exact) and f′C_ϱf (central difference, h = 1e-5) are computed from
#   the design. Root rule: the code's (cfe_root), as in tc_resd.jl. R = 500.
# Part 2, the interval: the package's REsd alone with a reference basis (Lemma 5(c) at C(ρ̃)), on
#   validation/mc_root_rho_reference.jl's design (n = 900, f = x + a·sin(2πu)cos(πv), reference
#   k̃ = 30, ℛ̃*), R = 300. Criterion: coverage ≥ 0.93 in every cell. Power: the Wald interval of
#   the same fits, which ignores β, below 0.90 at k = 5, a = 4.
#
# Run: julia --project=. research/trace_cfe/tc_resd_approx.jl > research/trace_cfe/tc_resd_approx_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions, SpatialEconometrics

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function moment_mats(A, M, Mii)                      # C = (A − Δ)M, Δ = Diag(AM)Diag(M)⁻¹; Ct = (A − Δ)′
    AM = A * M; δ = diag(AM) ./ Mii
    (; Ct = A' .- Diagonal(δ), C = AM .- δ .* M)
end

function fitC(mm, y, Wy, Q, Mii)
    e0 = y .- Q * (Q' * y); e1 = Wy .- Q * (Q' * Wy)
    ρ, a, b = cfe_root(e0, mm.Ct * y, e1, mm.Ct * Wy)
    y0 = y .- ρ .* Wy
    u = y0 .- Q * (Q' * y0); s2 = u .^ 2 ./ Mii
    S = mm.C + mm.C'; H = S * Q
    â = S * (y0 .- u)
    noise = vec(sum((H * (Q' * (s2 .* Q))) .* H, dims = 2))
    V = 2 * dot(s2, ((S ./ 2) .^ 2) * s2) + sum((â .^ 2 .- noise) .* s2)
    (ρ, sqrt(max(V, 0)) / abs(b - 2ρ * a))
end

Gof(Wd, r) = Wd / (I - r * Wd)
Cof(Wd, r, M, Mii) = moment_mats(Matrix(Gof(Wd, r)'), M, Mii)

function cell(; n, arm, a = 2.0, ρ0 = 0.4, R = 500, seed = 3)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); u, v = P[:, 1], P[:, 2]
    W = knnW(P, 6); Wd = Matrix(W)
    k = arm == :A ? round(Int, sqrt(2sqrt(n))) : 4
    Q = Matrix(qr(hatbasis(u, v, k)).Q); Q = Q[:, 1:rank(Q)]
    M = I - Q * Q'; Mii = diag(M)
    f = a .* sin.(2π .* u) .* cos.(π .* v); r = M * f
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    G = Gof(Wd, ρ0)
    m1 = moment_mats(Matrix(Wd'), M, Mii); ms = Cof(Wd, ρ0, M, Mii)
    Cs = (ms.C + ms.C') / 2
    βs = dot(f, ms.C * f); bv = 2 .* (Cs * f)
    Vs = 2 * dot(σ2, (Cs .^ 2) * σ2) + sum(bv .^ 2 .* σ2)
    Js = 2 * (dot(G * f, Cs * f) + tr(G' * Cs * Diagonal(σ2)))
    h = 1e-5
    βJ(ϱ) = (c = Cof(Wd, ϱ, M, Mii).C; cs = (c + c') / 2;
             (dot(f, c * f), 2 * (dot(G * f, cs * f) + tr(G' * cs * Diagonal(σ2)))))
    (βp, Jp), (βm, Jm) = βJ(ρ0 + h), βJ(ρ0 - h)
    dfc = (βp - βm) / 2h; dbj = (βp / Jp - βm / Jm) / 2h   # f′C_ϱf and d(β/J)/dϱ at ρ₀
    cs = βs / sqrt(Vs)
    Fac = lu(sparse(I - ρ0 * W))
    out = zeros(R, 5)                                 # ρ̃, ρ̂, se(ρ̂), ρ̂*, (β(ρ̃) − β*)/√V*
    for i in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n)); Wy = W * y
        ρt, _ = fitC(m1, y, Wy, Q, Mii)
        mt = Cof(Wd, ρt, M, Mii)
        ρh, seh = fitC(mt, y, Wy, Q, Mii)
        ρs, _ = fitC(ms, y, Wy, Q, Mii)
        out[i, :] = [ρt, ρh, seh, ρs, (dot(f, mt.C * f) - βs) / sqrt(Vs)]
    end
    sds = std(out[:, 4]); gap = std(out[:, 2] .- out[:, 4]) / sds
    pred = std(out[:, 1]) * abs(dbj) / sds
    cov = mean(abs.(out[:, 2] .- ρ0) .<= 1.96 .* out[:, 3])
    pcov = cdf(Normal(), 1.96 - cs) - cdf(Normal(), -1.96 - cs)
    bz = (mean(out[:, 2]) - ρ0 - βs / Js) / (std(out[:, 2]) / sqrt(R))
    @printf("%s n %5d K %3d | ‖Mf‖/√n %.4f  f′C_ϱf/n %+.4f  c* %+.3f | gap sd(ρ̂−ρ̂*)/sd(ρ̂*) %.4f  ×√n %.2f  first-order %.4f (sd ρ̃ %.4f) | bias ρ̂ %+.4f  β*/J* %+.4f (z %+.1f) | coverage %.3f  predicted %.3f | (β(ρ̃)−β*)/√V* mean %+.4f sd %.4f\n",
            arm, n, size(Q, 2), norm(r) / sqrt(n), dfc / n, cs, gap, sqrt(n) * gap, pred, std(out[:, 1]),
            mean(out[:, 2]) - ρ0, βs / Js, bz, cov, pcov, mean(out[:, 5]), std(out[:, 5]))
end

# ── Part 2: the package's REsd with a reference basis ──
function part2(; k, amp, R = 300, n = 900, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knn_weights(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n); X = reshape(x, :, 1)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    Fac = lu(sparse(I - 0.4 * W)); ref = tensor_pspline(u, v; k = 30)
    g(s, t) = amp * sin(2π * s) * cos(π * t); f = x .+ g.(u, v)
    ps = tensor_pspline(u, v; k); ℛ = basis_roughness(g, u, v; k = 30)
    rng2 = MersenneTwister(seed + 100k + round(Int, amp))
    cb = 0; cw = 0; hb = Float64[]; hw = Float64[]; t = Float64[]; m = 0
    for _ in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng2, n))
        p = root_rho(y, X, W; basis = ps.basis, roughness = ℛ, reference = ref)
        p.root_found && isfinite(p.ci_bias_aware[1]) || continue
        m += 1
        cb += p.ci_bias_aware[1] <= 0.4 <= p.ci_bias_aware[2]
        cw += p.ci_wald[1] <= 0.4 <= p.ci_wald[2]
        push!(hb, (p.ci_bias_aware[2] - p.ci_bias_aware[1]) / 2)
        push!(hw, (p.ci_wald[2] - p.ci_wald[1]) / 2)
        push!(t, p.bias_bound / sqrt(p.vhat))
    end
    @printf("k = %2d, a = %.0f, ℛ̃* = %.3f: %d fits | bias-aware: coverage %.3f half-length %.4f, b̃/√V̂ mean %.2f | Wald: coverage %.3f half-length %.4f\n",
            k, amp, ℛ, m, cb / m, mean(hb), mean(t), cw / m, mean(hw))
    cb / m, cw / m
end

t = @elapsed cell(; n = 400, arm = :A, R = 5)
println("warm-up: 5 replications at n = 400 in $(round(t, digits = 1)) s")
t = @elapsed cell(; n = 2500, arm = :A, R = 3)
println("3 replications at n = 2500 in $(round(t, digits = 1)) s. R = 500 per cell; MC SE of a coverage near 0.95: 0.010.\n")
t0 = time()
println("Part 1. Arm A: K grows, ‖Mf‖/√n falls. Arm B (power): k = 4 fixed.")
for arm in (:A, :B), n in (400, 900, 1600, 2500); cell(; n, arm); end
@printf("(part 1: %.1f min)\n\n", (time() - t0) / 60)
t0 = time()
println("Part 2. root_rho(y, X, W; basis, roughness = ℛ̃*, reference = tensor_pspline(u, v; k = 30)), n = 900, R = 300")
res = [part2(; k, amp) for (k, amp) in ((5, 4.0), (5, 1.0), (15, 4.0), (15, 1.0))]
ok = all(r -> r[1] >= 0.93, res); pw = res[1][2] < 0.90
println("Criterion (bias-aware coverage ≥ 0.93 in every cell): ", ok ? "met" : "NOT met")
println("Power (Wald below 0.90 at k = 5, a = 4): ", pw ? "met" : "NOT met")
@printf("(part 2: %.1f min)\n", (time() - t0) / 60)
