# root_rho(...; skewness) through the released function: the HC2 V̂ at the worst skewed law and
# signal of Lemma 12 (THEORY.md), and the test-inversion set's coverage there.
#
# Design: a jittered g × g grid, kNN W, the basis [x, tensor P-spline k × k], ρ₀ = 0.4. In each cell
# the signal f ∈ span[x, B] is Lemma 12's worst f at C(ρ₀) for γ̄ = 2, and the errors are the
# two-point law of skewness γ* (excess kurtosis γ*² − 2, Pearson's bound). y = (I − ρ₀W)⁻¹(f + ε).
# Each draw calls root_rho(y, X, W; basis = B, skewness = 2).
#
# Criteria, set before the run:
# 1. Lemma 12: the mean of V̂(ρ₀)/V over the draws (V̂ from r.vpoly, V = Var g(ρ₀) at C(ρ₀)) is
#    within 4 SE of 1 + β_skew(C(ρ₀)).
# 2. Power: it differs by more than 4 SE from the prediction without the skewness term L₂.
# 3. Coverage: CS covers ρ₀ within 3 binomial SE of 2Φ(z√(1 + β_skew)) − 1.
# 4. The diagnostic: the mean of r.vhat_worst_bias (at REsd's C(ρ̃)) is within 0.003 of β_skew at
#    C(ρ₀), and r.skewness == 2 in every draw.
#
# Part 2 was added AFTER criterion 1 failed in all three cells (and 2 in one). The diagnosis: V̂ from
# r.vpoly uses REsd's final C(ρ̃), which varies with the data, and with a strong signal
# (‖b‖² = 1.5–3 V₀) the variance of g̃(ρ₀) = y₀′C(ρ̃)y₀ is larger than V at C(ρ₀). Criterion 1 compared
# V̂ with the wrong variance. Part 2's criteria, on the same design and R = 1000 fresh draws per cell:
# 1′a. V̂ at the fixed C(ρ₀) is within 4 SE of 1 + β_skew (Lemma 12 itself; testset 52b's check).
# 1′b. V̂ at C(ρ̃), divided by the variance at that draw's C(ρ̃) (C(ρ̃) held fixed), averages within
#      0.01 of 1 + the mean of r.vhat_worst_bias (Lemma 12 at C(ρ̃), which root_rho reports).
# Neither replaces criterion 1, which stays failed in the output.
#
# Run: julia --project=. validation/mc_root_rho_hc2_skew.jl > validation/mc_root_rho_hc2_skew_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf, Distributions
RR = SpatialEconometrics.RootRho
const ρ0 = 0.4; const γ̄ = 2.0; const R = 4000

function cell(g, k, knn; seed = 7)
    rng = MersenneTwister(seed); n = g^2
    P = hcat(repeat(0:g-1, inner = g), repeat(0:g-1, outer = g)) .+ rand(rng, n, 2); P ./= g
    u, v = P[:, 1], P[:, 2]; W = knn_weights(P, knn); Wd = Matrix(W)
    x = randn(rng, n); X = reshape(x, :, 1)
    B = tensor_pspline(u, v; k).basis; Z = hcat(x, B); Q = Matrix(qr(Z).Q)[:, 1:size(Z, 2)]
    Mii = 1 .- vec(sum(abs2, Q; dims = 2))
    C = RR._moment_matrix(Matrix((I - ρ0 * Wd)' \ Wd'), Q, Mii)
    # Lemma 12 at C(ρ₀), densely: the worst γ and the worst C′f
    M = I - Q * Q'; Ω = ((C + C') ./ 2) .^ 2; H = (C + C') * (Q * Q'); M2 = M .^ 2; H2 = H .^ 2
    DD = Mii .* Mii'; U = M2 ./ Mii
    A = 4 * sum(Ω .* M2 ./ DD) - 2 * sum(H2 .* M2 ./ DD)
    sc = sum(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .- sum(U .* (H2 * U); dims = 1))
    t1 = vec(sum(H .* U; dims = 2)); CZ = C' * Z; pt = CZ * (pinv(CZ) * t1); τ = norm(pt)
    V0 = 2 * sum(Ω)
    βof(γ) = (a = A - 2sc + γ^2 * sc; (a - sqrt(a^2 + 4(γ * τ)^2 * V0)) / (2V0))
    γs = range(0, γ̄, length = 20001); γ = γs[argmin(βof.(γs))]; βs = βof(γ)
    γ > 0 || error("skewness does not bite in this design")
    a = A - 2sc + γ^2 * sc; s = (a + sqrt(a^2 + 4(γ * τ)^2 * V0)) / (2γ * τ)
    f = Z * (CZ \ (-s .* pt ./ τ)); b = C' * f; V = V0 + sum(abs2, b)
    pnoL2 = 1 + (A + sc * (γ^2 - 2)) / V
    βsym = min(0, A - 2sc) / V0
    q = (1 - γ / sqrt(γ^2 + 4)) / 2; hi = sqrt((1 - q) / q); lo = -sqrt(q / (1 - q))
    Lf = lu(sparse(I - ρ0 * W))
    mc = MersenneTwister(seed + 1)
    ratio = zeros(R); cover = 0; diag_ = zeros(R); skewok = true
    for i in 1:R
        ε = [rand(mc) < q ? hi : lo for _ in 1:n]
        y = Lf \ (f .+ ε)
        r = root_rho(y, X, W; basis = B, skewness = γ̄)
        ratio[i] = RR._polyval(r.vpoly, ρ0) / V
        cover += any(lo_ <= ρ0 <= hi_ for (lo_, hi_) in r.cs)
        diag_[i] = r.vhat_worst_bias; skewok &= r.skewness == γ̄
    end
    se = std(ratio) / sqrt(R); cov = cover / R
    pc = 2cdf(Normal(), quantile(Normal(), 0.975) * sqrt(1 + βs)) - 1; seb = sqrt(pc * (1 - pc) / R)
    c1 = abs(mean(ratio) - (1 + βs)) < 4se
    c2 = abs(mean(ratio) - pnoL2) > 4se
    c3 = abs(cov - pc) <= 3seb
    c4 = abs(mean(diag_) - βs) <= 0.003 && skewok
    @printf("n = %3d, k = %d, kNN-%2d (K/n %.3f): γ* %.2f, ‖b‖²/V₀ %.2f | β_skew %+.4f (symmetric %+.4f) | E V̂/V sim %.4f, pred %.4f (z %+.2f) [1 %s]; without L₂ %.4f (z %+.1f) [2 %s] | CS %.4f, pred %.4f (%+.1f SE) [3 %s] | vhat_worst_bias mean %+.4f, sd %.4f [4 %s]\n",
            n, k, knn, size(Z, 2) / n, γ, s^2 / V0, βs, βsym, mean(ratio), 1 + βs, (mean(ratio) - 1 - βs) / se,
            c1 ? "ok" : "FAIL", pnoL2, (mean(ratio) - pnoL2) / se, c2 ? "ok" : "FAIL", cov, pc,
            (cov - pc) / seb, c3 ? "ok" : "FAIL", mean(diag_), std(diag_), c4 ? "ok" : "FAIL")
    c1 && c2 && c3 && c4
end

println("root_rho's vhat_worst_bias with skewness = 2, at Lemma 12's worst law and signal; R = $R per cell\n")
t0 = time()
ok = true
for (g, k, knn) in ((14, 5, 10), (20, 5, 10), (20, 8, 6))
    global ok &= cell(g, k, knn)
    @printf("    (%.1f min)\n", (time() - t0) / 60)
end
println("\nAll criteria met: ", ok)

function diagnose(g, k, knn; seed = 7, R = 1000)
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
    f = Z * (CZ \ (-s .* pt ./ τ)); V = V0 + sum(abs2, C' * f); pred = 1 + βof(γ)
    q = (1 - γ / sqrt(γ^2 + 4)) / 2; hi = sqrt((1 - q) / q); lo = -sqrt(q / (1 - q))
    Lf = lu(sparse(I - ρ0 * W)); mc = MersenneTwister(seed + 2)
    fixedC = zeros(R); own = Float64[]; Vt = Float64[]; wb = Float64[]; g0 = Float64[]; nofirst = 0
    for i in 1:R
        y = Lf \ (f .+ [rand(mc) < q ? hi : lo for _ in 1:n])
        r = root_rho(y, X, W; basis = B, skewness = γ̄)
        y0 = y .- ρ0 .* (W * y)
        fixedC[i] = RR._polyval(RR._vhat_poly(C, Q, Mii, y0, zeros(n)), 0.0) / V
        # With no first-step root, root_rho keeps the first-step C (A = W′); 1′b skips those draws.
        isfinite(r.rho_first) || (nofirst += 1; continue)
        Ct = RR._moment_matrix(Matrix((I - r.rho_first * Wd)' \ Wd'), Q, Mii)
        push!(Vt, 2 * sum(((Ct + Ct') ./ 2) .^ 2) + sum(abs2, Ct' * f))
        push!(own, RR._polyval(r.vpoly, ρ0) / Vt[end]); push!(wb, r.vhat_worst_bias)
        push!(g0, dot(y0, Ct * y0))
    end
    z = (mean(fixedC) - pred) / (std(fixedC) / sqrt(R))
    ca = abs(z) < 4; cb = abs(mean(own) - (1 + mean(wb))) <= 0.01
    @printf("n = %3d, k = %d, kNN-%2d: 1′a V̂ at C(ρ₀)/V %.4f, pred %.4f (z %+.2f) [%s] | 1′b V̂ at C(ρ̃)/V(C(ρ̃)) %.4f, 1 + mean vhat_worst_bias %.4f [%s] (%d draws without a first-step root skipped) | V(C(ρ̃))/V %.4f, Var g̃(ρ₀)/V over draws %.4f\n",
            n, k, knn, mean(fixedC), pred, z, ca ? "ok" : "FAIL", mean(own), 1 + mean(wb), cb ? "ok" : "FAIL",
            nofirst, mean(Vt) / V, var(g0) / V)
    ca && cb
end

println("\nPart 2, added after criterion 1 failed: the same cells, R = 1000 fresh draws")
ok2 = true
for (g, k, knn) in ((14, 5, 10), (20, 5, 10), (20, 8, 6))
    global ok2 &= diagnose(g, k, knn)
    @printf("    (%.1f min)\n", (time() - t0) / 60)
end
println("\nPart 2's criteria met: ", ok2, "   (criterion 1 of part 1 is still failed)")
