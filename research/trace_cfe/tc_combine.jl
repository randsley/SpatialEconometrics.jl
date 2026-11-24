# Combining the candidate moments (Armstrong & Kolesár 2021, §2.2) in place of picking one.
#
# Proposition 7 picks one moment g_j from the grid (REsd and the smoother at each λ) by the length of
# its bias-aware interval. Armstrong & Kolesár choose the best linear combination of all the moments.
# For g_w = Σ_j w_j g_j, normalised by Σ_j w_j J_j = 1 (their H = −k′Γ), the root's linearisation gives
#     bias β_w = Σ_j w_j β_j,   variance V_w = w′Σw,   Σ_jl = Cov(g_j(ρ₀), g_l(ρ₀)),
# and, since every candidate's bias is linear plus quadratic in the same z̃ = Π̃^{1/2}ψ̃₁ (Lemma 5),
#     β_w = ã_w′z̃ + z̃′H̃_w z̃,   ã_w = Σ_j w_j ã_j,   H̃_w = Σ_j w_j H̃_j,
# so Lemma 5's sum bound b̃(w) = ℛ̃‖ã_w‖ + ℛ̃²χ(H̃_w) and Lemma 6's b_TR(w) apply to the combination.
# The half-length is ℓ(w) = cv(b(w)/√V_w)·√V_w. This script is the ORACLE comparison: f̃, ã_j, β_j,
# J_j and Σ are exact (normal errors, so Σ needs no fourth moments), and there is no simulation.
#   Σ_jl = 2 tr(A_j^s Σ_ε A_l^s Σ_ε) + b_j′Σ_ε b_l,  b_j = 2A_j^s f̃,  A^s = (A + A′)/2,
#   J_j = 2(f̃′G′A_j^s f̃ + tr(G′A_j^s Σ_ε)),  G = W(I − ρ₀W)⁻¹.
# REsd enters at ρ₀ (C(ρ₀)), as in tc_trust.jl (1).
#
# Checks:
# (0) the optimiser against a closed form: three unbiased moments with known Σ, where the optimal
#     weights are GLS's; power: the best single moment is longer.
# (1) per cell, the single candidates' b̃/√V, which must agree with tc_trust.jl (1)'s b̃ column (to
#     0.2%: tc_trust.jl forms V with f, this script with f̃ throughout); then ℓ at
#     the best single candidate and at the optimised w (Nelder–Mead on the sum bound, started at the
#     best single candidate, so it cannot do worse), with the oracle coverage
#     P(|Z + β_w/√V_w| ≤ cv(b/√V_w)) of each, b_TR at the optimum, and the weights' shares.
# Criterion (set before the run): if the combination is less than 10% shorter than the best single
# candidate at k = 15 for both ℛ̃* and 4ℛ̃*, record it and close the item; otherwise study a plug-in.
# Design: tc_trust.jl's (n = 900, kNN-6, f = x + a·sin(2πu)cos(πv), heteroskedastic normal errors,
# ρ₀ = 0.4), k ∈ {5, 15}, a ∈ {1, 4}, reference k̃ = 30.
#
# Run: julia --project=. research/trace_cfe/tc_combine.jl > research/trace_cfe/tc_combine_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
function penalty(k)
    D2 = diff(diff(Matrix(1.0I, k, k), dims = 1), dims = 1)
    Ω = kron(D2' * D2, Matrix(1.0I, k, k)) + kron(Matrix(1.0I, k, k), D2' * D2)
    Π = zeros(k^2 + 1, k^2 + 1); Π[1:k^2, 1:k^2] = Ω; Π
end
basis(u, v, x, k) = hcat(hatbasis(u, v, k), x)
g_surf(u, v) = sin(2π * u) * cos(π * v)
interp_coef(k, amp) = (t = range(0, 1, length = k); vcat([amp * g_surf(t[i], t[j]) for i in 1:k for j in 1:k], 1.0))
function rangeparts(Π)
    E = eigen(Symmetric(Π)); keep = E.values .> 1e-10 * maximum(E.values)
    U = E.vectors[:, keep]; ω = E.values[keep]
    (; Rinv = Diagonal(1 ./ sqrt.(ω)) * U')
end
symm(Q) = (Q + Q') / 2
function cv(t; α = 0.05)
    t = abs(t); lo, hi = 0.0, t + 10
    for _ in 1:80
        c = (lo + hi) / 2
        cdf(Normal(), c - t) - cdf(Normal(), -c - t) < 1 - α ? (lo = c) : (hi = c)
    end
    (lo + hi) / 2
end
pcov(c, h) = cdf(Normal(), h - c) - cdf(Normal(), -h - c)        # P(|Z + c| ≤ h)
jl_E(A, MZ, mz) = A - Diagonal(diag(A * MZ) ./ mz)

# Lemma 6's trust-region maximum (tc_trust.jl's solver, checked there)
function tr_max(α::Vector{Float64}, h::Vector{Float64}, R::Float64)
    R == 0 && return 0.0
    hmax = h[end]; na = norm(α)
    ξn(μ) = sqrt(sum(abs2, α ./ (2 .* (μ .- h))))
    val(ξ) = dot(α, ξ) + dot(ξ, h .* ξ)
    if hmax < 0 && ξn(0.0) <= R
        return val(α ./ (2 .* (0.0 .- h)))
    end
    lo = max(hmax, 0.0)
    top = findall(>=(hmax - 1e-12 * max(abs(hmax), 1.0)), h)
    if lo == hmax && sum(abs2, α[top]) <= 1e-24 * max(na^2, 1e-300)
        rest = setdiff(eachindex(h), top)
        ξ = zeros(length(h)); ξ[rest] = α[rest] ./ (2 .* (hmax .- h[rest]))
        if norm(ξ) <= R
            ξ[top[1]] = sqrt(R^2 - norm(ξ)^2)
            return val(ξ)
        end
    end
    a0 = lo; b0 = lo + na / (2R)
    for _ in 1:200
        μ = (a0 + b0) / 2
        ξn(μ) > R ? (a0 = μ) : (b0 = μ)
    end
    val(α ./ (2 .* (b0 .- h)))
end
b_trust(a, H, R) = (E = eigen(Symmetric(H)); U = E.vectors; h = E.values;
                    max(tr_max(U' * a, h, R), tr_max(reverse(-(U' * a)), -reverse(h), R)))
function χof(H)                                                   # largest |eigenvalue|, LAPACK
    S = Symmetric(H); m = size(H, 1)
    max(abs(eigvals(S, 1:1)[1]), abs(eigvals(S, m:m)[1]))
end
# The same by Lanczos with full reorthogonalisation (m steps from a fixed start): the extreme Ritz
# values converge from inside, so this is at most χ. Used inside the optimiser only; the optimum is
# re-evaluated with χof, and the two are compared on every H̃_j in (1).
function χlanczos(H; m = 60)
    p = size(H, 1); m = min(m, p)
    V = zeros(p, m + 1); α = zeros(m); β = zeros(m)
    v = ones(p) ./ sqrt(p); V[:, 1] = v
    k = m
    for j in 1:m
        w = H * V[:, j]; α[j] = dot(w, V[:, j])
        w .-= α[j] .* V[:, j]; j > 1 && (w .-= β[j-1] .* V[:, j-1])
        for _ in 1:2, i in 1:j; w .-= dot(w, V[:, i]) .* V[:, i]; end
        β[j] = norm(w)
        if β[j] < 1e-12; k = j; break; end
        V[:, j+1] = w ./ β[j]
    end
    θ = eigvals(SymTridiagonal(α[1:k], β[1:k-1]))
    max(abs(θ[1]), abs(θ[end]))
end

# Nelder–Mead (standard coefficients), minimising F over R^p from x0
function nelder_mead(F, x0; step = 0.1, iters = 3000, tol = 1e-10)
    p = length(x0); X = [copy(x0) for _ in 1:p+1]
    for i in 1:p; X[i+1][i] += step * max(abs(x0[i]), 1.0); end
    Fv = F.(X)
    for _ in 1:iters
        o = sortperm(Fv); X = X[o]; Fv = Fv[o]
        (Fv[end] - Fv[1]) <= tol * max(abs(Fv[1]), 1e-300) && break
        c = sum(X[1:p]) / p
        xr = c .+ (c .- X[end]); fr = F(xr)
        if fr < Fv[1]
            xe = c .+ 2 .* (c .- X[end]); fe = F(xe)
            fe < fr ? (X[end] = xe; Fv[end] = fe) : (X[end] = xr; Fv[end] = fr)
        elseif fr < Fv[p]
            X[end] = xr; Fv[end] = fr
        else
            xc = fr < Fv[end] ? c .+ 0.5 .* (xr .- c) : c .+ 0.5 .* (X[end] .- c)
            fc = F(xc)
            if fc < min(fr, Fv[end])
                X[end] = xc; Fv[end] = fc
            else
                for i in 2:p+1; X[i] = X[1] .+ 0.5 .* (X[i] .- X[1]); Fv[i] = F(X[i]); end
            end
        end
    end
    i = argmin(Fv); X[i], Fv[i]
end

# Minimise ℓ(w) over {w : w′J = 1}, from w0, with restarts from the last optimum
function optimise(ℓ, J, w0; restarts = 3)
    N = nullspace(reshape(J, 1, :))                               # w = w0 + N u keeps w′J = 1
    u = zeros(size(N, 2)); best = ℓ(w0)
    for r in 1:restarts
        u2, v = nelder_mead(x -> ℓ(w0 .+ N * x), u; step = 0.5 / r)
        v < best && (best = v; u = u2)
    end
    w0 .+ N * u, best
end

# (0) the optimiser on a closed form: unbiased moments, ℓ = 1.96·√(w′Σw), optimum 1.96/√(J′Σ⁻¹J)
function check_optimiser()
    rng = MersenneTwister(3)
    A = randn(rng, 3, 3); Σ = A * A' + 0.5I; J = [1.0, 0.7, 1.3]
    ℓ(w) = cv(0.0) * sqrt(dot(w, Σ * w))
    j = argmin([ℓ(e ./ J[i]) for (i, e) in enumerate(eachcol(Matrix(1.0I, 3, 3)))])
    w0 = zeros(3); w0[j] = 1 / J[j]
    w, v = optimise(ℓ, J, w0)
    exact = cv(0.0) / sqrt(dot(J, Σ \ J))
    @printf("(0) optimiser: three unbiased moments, ℓ found %.8f, closed form (GLS) %.8f, rel. gap %.1e; best single %.6f (power: longer by %.0f%%)\n\n",
            v, exact, v / exact - 1, ℓ(w0), 100 * (ℓ(w0) / exact - 1))
end

function setup(; n = 900, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; n, Wd = Matrix(W), u, v, x, σ2)
end

const ΛS = 10.0 .^ (-3:0.5:2)

function cell(d, k, amp; kt = 30, mults = (1.0, 4.0))
    t0 = time()
    Wd = d.Wd; sd = sqrt.(d.σ2)
    Z = basis(d.u, d.v, d.x, k); Π = penalty(k); K = size(Z, 2)
    Zt = basis(d.u, d.v, d.x, kt); Πt = penalty(kt)
    Q = Matrix(qr(Z).Q)[:, 1:K]; Pz = Q * Q'; MZ = I - Pz; mz = diag(MZ)
    rt = rangeparts(Πt)
    ψt = interp_coef(kt, amp); ft = Zt * ψt; ℛt = sqrt(dot(ψt, Πt * ψt))
    MZZt = MZ * Zt; G = Wd / (I - 0.4 * Wd)
    labels = String[]; As = Matrix{Float64}[]; Ã = Vector{Float64}[]; H̃ = Matrix{Float64}[]; Cfull = Matrix{Float64}[]
    E0 = jl_E(G', MZ, mz); C0 = E0 * MZ                           # REsd at ρ₀
    push!(labels, "REsd@ρ₀"); push!(Cfull, C0); push!(As, symm(C0))
    push!(Ã, rt.Rinv * (MZZt' * (E0' * (Pz * ft))))
    push!(H̃, rt.Rinv * symm(MZZt' * E0 * MZZt) * rt.Rinv')
    for λ in ΛS
        S = symm(Z * ((Z' * Z + λ * Π) \ Z')); M = I - S
        dd = (M .* M) \ (-diag(M * Wd)); Dg = Diagonal(dd); A = M * Wd + M * Dg * M
        MZt = M * Zt
        push!(labels, @sprintf("λ %.4f", λ)); push!(Cfull, A); push!(As, symm(A))
        push!(Ã, rt.Rinv * ((MZt' * Wd) * (Pz * ft)))
        push!(H̃, rt.Rinv * symm(MZt' * (Wd * MZZt) + MZt' * Dg * MZt) * rt.Rinv')
    end
    L = length(As)
    β = [dot(ft, Cfull[j] * ft) for j in 1:L]
    b = [2 .* (As[j] * ft) for j in 1:L]
    Bs = [sd .* As[j] .* sd' for j in 1:L]                       # Σ_ε^{1/2} A^s Σ_ε^{1/2}
    Σ = [2 * sum(Bs[j] .* Bs[l]) + dot(b[j], d.σ2 .* b[l]) for j in 1:L, l in 1:L]
    J = [2 * (dot(G * ft, As[j] * ft) + tr(G' * As[j] * Diagonal(d.σ2))) for j in 1:L]
    χs = [χof(H̃[j]) for j in 1:L]
    lz = maximum(abs(χlanczos(H̃[j]) / χs[j] - 1) for j in 1:L)
    @printf("k = %d (K = %d), a = %.0f, k̃ = %d, ℛ̃* = %.3f; %d candidates; Lanczos χ against LAPACK, largest rel. gap %.1e; setup %.1f min\n",
            k, K, amp, kt, ℛt, L, lz, (time() - t0) / 60)
    print("  single b̃/√V at ℛ̃* (REsd, then λ = 10^-3, 10^-2, …, 10^2; compare tc_trust.jl (1)):")
    for j in (1, 2:2:L...)
        @printf(" %.3f", (ℛt * norm(Ã[j]) + ℛt^2 * χs[j]) / sqrt(Σ[j, j]))
    end
    println()
    @printf("  correlation of the moments: REsd with λ = 10^-3 %.3f; neighbouring λ %.3f–%.3f; λ = 10^-3 with 10^2 %.3f\n",
            Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2]),
            minimum(Σ[j, j+1] / sqrt(Σ[j, j] * Σ[j+1, j+1]) for j in 2:L-1),
            maximum(Σ[j, j+1] / sqrt(Σ[j, j] * Σ[j+1, j+1]) for j in 2:L-1),
            Σ[2, L] / sqrt(Σ[2, 2] * Σ[L, L]))
    for m in mults
        ℛ = m * ℛt
        aw(w) = sum(w[j] .* Ã[j] for j in 1:L)
        Hw(w) = sum(w[j] .* H̃[j] for j in 1:L)
        bsum(w) = ℛ * norm(aw(w)) + ℛ^2 * χof(Hw(w))             # exact, for reporting
        bfast(w) = ℛ * norm(aw(w)) + ℛ^2 * χlanczos(Hw(w))
        Vw(w) = dot(w, Σ * w)
        ℓ(w) = (V = Vw(w); V > 0 ? cv(bsum(w) / sqrt(V)) * sqrt(V) : Inf)
        ℓfast(w) = (V = Vw(w); V > 0 ? cv(bfast(w) / sqrt(V)) * sqrt(V) : Inf)
        singles = [ℓ((e = zeros(L); e[j] = 1 / J[j]; e)) for j in 1:L]
        js = argmin(singles); w0 = zeros(L); w0[js] = 1 / J[js]
        t1 = time()
        w, _ = optimise(ℓfast, J, w0)
        ℓw = ℓ(w); ℓw > singles[js] && (w = w0; ℓw = singles[js])    # never worse than the start
        βw = dot(w, β); bw = bsum(w); btr = b_trust(aw(w), Hw(w), ℛ); sV = sqrt(Vw(w))
        sj = sqrt(Σ[js, js]) / abs(J[js])                          # the single's sd in ρ units
        cs = pcov(β[js] / J[js] / sj, singles[js] / sj)
        cw = pcov(βw / sV, ℓw / sV)
        share = abs.(w .* J); share ./= sum(share); top = sortperm(share, rev = true)[1:3]
        @printf("  ℛ̃ × %.0f: best single %-9s ℓ %.5f (coverage %.3f) | combination ℓ %.5f (%+.1f%%, coverage %.3f), b̃/√V %.3f, b_TR/√V %.3f, |β|/√V %.3f | weight shares %s | %.1f min\n",
                m, labels[js], singles[js], cs, ℓw, 100 * (ℓw / singles[js] - 1), cw, bw / sV, btr / sV,
                abs(βw) / sV, join([@sprintf("%s %.2f", labels[i], share[i]) for i in top], ", "), (time() - t1) / 60)
    end
    println()
end

check_optimiser()
d = setup()
t0 = time()
if get(ARGS, 1, "") == "quick"
    cell(d, 5, 1.0; mults = (1.0,))
else
    for amp in (1.0, 4.0), k in (5, 15)
        cell(d, k, amp)
    end
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
