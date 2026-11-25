# The first-step effect in root_rho's projection path: the test-inversion set uses REsd's moment matrix
# C̃ = C(ρ̃), which depends on the data through the first-step root ρ̃ (A = W′). Propositions 6 and 6″
# were proved for a FIXED C. What does a data-dependent C(ρ̃) change?
#
# The family: C(r) = E(r)M with A(r) = (I − rW′)⁻¹W′ (RootRho._moment_matrix), r ∈ (−1, 1). For every r,
# diag C(r) = 0 and C(r)P_B = 0, so g_r = y₀′C(r)y₀ = b_r′ε + ε′C(r)ε has mean 0 and variance V_r
# (Lemma 7(a)). The same holds for Ċ(r) = dC/dr, so ġ = y₀′Ċ(ρ₀)y₀ also has mean 0, and
#     g̃ − g_ρ₀ = (ρ̃ − ρ₀)ġ + O((ρ̃ − ρ₀)²)     is O_p(1) when ρ̃ − ρ₀ = O_p(n^{-1/2}),
# against √V = O(√n): the first step does not matter to first order. If ġ had a mean of order n, the term
# would be O_p(√n), first-order. Without identification ρ̃ does not converge, and g̃/√V_C̃ is the process
# Z(r) = g_r/√V_r at a data-dependent r, which Proposition 6's proof does not cover.
#
# Per draw (y = (I − ρ₀W)⁻¹(f + ε), released root_rho; ρ̃ = r.rho_first, or 0 when there is no first-step
# root, since root_rho then keeps C(0)):
#   d = g̃ − g₀ with g̃ = y₀′C(ρ̃)y₀ and g₀ = y₀′C(ρ₀)y₀; d_lin = (ρ̃ − ρ₀)·y₀′Ċ(ρ₀)y₀ (Ċ by central difference);
#   V_own = V at the draw's C(ρ̃) held fixed (true f and σ²), V₀ = V at C(ρ₀).
# Reported per cell:
#   √n·sd(ρ̃); n·Var(d)/V₀ (bounded in n if d = O_p(1)); corr(d, d_lin) and the regression slope of d on d_lin;
#   Var(g̃)/V₀ and E V_own/V₀ (Var g̃ = E V_own + Var of the conditional mean, so a gap measures the
#   dependence between ρ̃ and ε); Var(g̃/√V_own) and mean(g̃/√V_own);
#   coverage at ρ₀: root_rho's CS (released), the oracle with the draw's own V (|g̃| ≤ 1.96√V_own), the HC2
#   test at the fixed C(ρ₀) (= the coverage of a continuously updated set that tests each ρ with C(ρ)),
#   and at the fixed C(0) (the first-step matrix).
# Part A: identified, kNN-6, n = 196, 400, 900, 1600, K/n ≈ 0.1. Part B: weak identification, kNN-20,
# n = 400, K/n ≈ 0.2 and 0.3 (tc_ar.jl's weak cells, where sd ρ̂ ≥ 0.39 at 0.3). Part C: the n = 196 cell
# of validation/mc_root_rho_hc2_skew.jl (Lemma 12's worst f and two-point law), where Var g̃(ρ₀)/V was 1.41.
#
# Run: julia --project=. research/trace_cfe/tc_first_step.jl > research/trace_cfe/tc_first_step_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf, Distributions
const RR = SpatialEconometrics.RootRho
const ρ0 = 0.4

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

Cof(Wd, Q, Mii, r) = RR._moment_matrix(Matrix((I - r * Wd)' \ Wd'), Q, Mii)
Vexact(C, f, σ2) = (Cs = (C + C') ./ 2; 2 * dot(σ2, (Cs .^ 2) * σ2) + sum(abs2.(C' * f) .* σ2))
vhat_at(C, Q, Mii, y0) = RR._polyval(RR._vhat_poly(C, Q, Mii, y0, zeros(length(y0))), 0.0)

function run_cell(label, W, X, B, f, σ2, drawε; R)
    Wd = Matrix(W); Z = hcat(X, B); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]; n = size(Z, 1)
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    C0 = Cof(Wd, Q, Mii, ρ0); Cz = Cof(Wd, Q, Mii, 0.0)
    h = 1e-4; Cdot = (Cof(Wd, Q, Mii, ρ0 + h) - Cof(Wd, Q, Mii, ρ0 - h)) ./ (2h)
    V0 = Vexact(C0, f, σ2); Vz = Vexact(Cz, f, σ2)
    Lf = lu(sparse(I - ρ0 * W)); sd = sqrt.(σ2)
    ρt = zeros(R); d = zeros(R); dl = zeros(R); gt = zeros(R); Vown = zeros(R)
    cs = 0; orc = 0; fix0 = 0; fixz = 0; nofirst = 0; negV = 0
    for i in 1:R
        y = Lf \ (f .+ sd .* drawε())
        r = root_rho(y, X, W; basis = B)
        y0 = y .- ρ0 .* (W * y)
        rt = isfinite(r.rho_first) ? r.rho_first : (nofirst += 1; 0.0)
        Ct = Cof(Wd, Q, Mii, rt)
        g̃ = dot(y0, Ct * y0); g0 = dot(y0, C0 * y0); gz = dot(y0, Cz * y0)
        ρt[i] = rt; gt[i] = g̃; d[i] = g̃ - g0; dl[i] = (rt - ρ0) * dot(y0, Cdot * y0)
        Vown[i] = Vexact(Ct, f, σ2)
        cs += any(lo <= ρ0 <= hi for (lo, hi) in r.cs)
        orc += g̃^2 <= 1.96^2 * Vown[i]
        v0 = vhat_at(C0, Q, Mii, y0); vz = vhat_at(Cz, Q, Mii, y0); negV += (v0 <= 0) + (vz <= 0)
        fix0 += v0 > 0 && g0^2 <= 1.96^2 * v0
        fixz += vz > 0 && gz^2 <= 1.96^2 * vz
    end
    t = gt ./ sqrt.(Vown); slope = dot(dl, d) / dot(dl, dl)
    @printf("%-34s n %4d K/n %.3f | √n·sd ρ̃ %5.2f (no root %d) | n·Var d/V₀ %7.2f, corr(d, d_lin) %.3f, slope %.2f | Var g̃/V₀ %.3f, E V_own/V₀ %.3f | g̃/√V_own: mean %+.3f var %.3f | CS %.3f, oracle-own %.3f, fixed C(ρ₀) %.3f, fixed C(0) %.3f%s\n",
            label, n, size(Q, 2) / n, sqrt(n) * std(ρt), nofirst, n * var(d) / V0, cor(d, dl), slope,
            var(gt) / V0, mean(Vown) / V0, mean(t), var(t), cs / R, orc / R, fix0 / R, fixz / R,
            negV > 0 ? @sprintf(" (V̂ ≤ 0: %d)", negV) : "")
    flush(stdout)
end

function grid_design(side, knn; seed = 11)
    rng = MersenneTwister(seed); n = side^2
    P = hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ rand(rng, n, 2); P ./= side
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, knn); x = randn(rng, n)
    σs = Dict(:constant => ones(n), :rough => (s = exp.(randn(rng, n)); s ./ mean(s)))
    (; n, u, v, W, X = reshape(x, :, 1), x, σs)
end

function design_cell(side, knn, kh, vname, law, amp; R, seed = 11)
    d = grid_design(side, knn; seed)
    B = hatbasis(d.u, d.v, kh); Z = hcat(d.x, B); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]
    f = d.x .+ amp .* (Q * (Q' * (sin.(2π .* d.u) .* cos.(π .* d.v))))
    rng = MersenneTwister(seed + 1000kh + round(Int, 10amp))
    drawε = law == :normal ? (() -> randn(rng, d.n)) : (() -> (rand(rng, Chisq(3), d.n) .- 3) ./ sqrt(6))
    run_cell(@sprintf("kNN-%d %s %s a=%g", knn, vname, law, amp), d.W, d.X, B, f, d.σs[vname], drawε; R)
end

t0 = time()
println("First-step effect in root_rho's projection path (REsd's C(ρ̃)); ρ₀ = $ρ0. MC SE of a coverage near 0.95: 0.007 at R = 1000.\n")
println("Part A: identified (kNN-6, K/n ≈ 0.1, hat basis + x), normal errors")
for (side, kh) in ((14, 4), (20, 6), (30, 9), (40, 12)), vname in (:constant, :rough), amp in (1.0, 4.0)
    R = side <= 30 ? 1000 : 500
    design_cell(side, 6, kh, vname, :normal, amp; R)
end
@printf("  (%.1f min)\n\n", (time() - t0) / 60)

println("Part B: weak identification (kNN-20, n = 400), R = 1000")
for kh in (9, 11), vname in (:constant, :rough), law in (:normal, :chisq3)
    design_cell(20, 20, kh, vname, law, 1.0; R = 1000)
end
@printf("  (%.1f min)\n\n", (time() - t0) / 60)

# Part C: the n = 196 cell of validation/mc_root_rho_hc2_skew.jl (code copied from its diagnose()).
println("Part C: mc_root_rho_hc2_skew.jl's n = 196 cell (Lemma 12's worst f, two-point law γ* = 2), R = 2000")
let g = 14, k = 5, knn = 10, γ̄ = 2.0, seed = 7
    rng = MersenneTwister(seed); n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, knn); Wd = Matrix(W)
    x = randn(rng, n); X = reshape(x, :, 1)
    B = tensor_pspline(u, v; k).basis; Z = hcat(x, B); Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    C = RR._moment_matrix(Matrix((I - ρ0 * Wd)' \ Wd'), Q, Mii)
    M = I - Q * Q'; Ω = ((C + C') ./ 2) .^ 2; H = (C + C') * (Q * Q'); M2 = M .^ 2; H2 = H .^ 2
    DD = Mii .* Mii'; U = M2 ./ Mii
    A = 4 * sum(Ω .* M2 ./ DD) - 2 * sum(H2 .* M2 ./ DD)
    sc = sum(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .- sum(U .* (H2 * U); dims = 1))
    t1 = vec(sum(H .* U; dims = 2)); CZ = C' * Z; pt = CZ * (pinv(CZ) * t1); τ = norm(pt); V0 = 2 * sum(Ω)
    βof(γ) = (a = A - 2sc + γ^2 * sc; (a - sqrt(a^2 + 4(γ * τ)^2 * V0)) / (2V0))
    γs = range(0, γ̄, length = 20001); γ = γs[argmin(βof.(γs))]
    a = A - 2sc + γ^2 * sc; s = (a + sqrt(a^2 + 4(γ * τ)^2 * V0)) / (2γ * τ)
    f = Z * (CZ \ (-s .* pt ./ τ))
    q = (1 - γ / sqrt(γ^2 + 4)) / 2; hi = sqrt((1 - q) / q); lo = -sqrt(q / (1 - q))
    mc = MersenneTwister(seed + 2)
    run_cell("skew cell, two-point γ*=2", W, X, B, f, ones(n), () -> [rand(mc) < q ? hi : lo for _ in 1:n]; R = 2000)
end
@printf("\ntotal %.1f min\n", (time() - t0) / 60)
