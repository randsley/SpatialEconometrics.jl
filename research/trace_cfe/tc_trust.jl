# The joint (trust-region) bound for Lemma 5, in place of the sum of the two worst cases.
#
# Lemma 5 bounds β = ψ̃₁′ṽ + ψ̃₁′Q̃ψ̃₁ over ψ̃₁′Π̃ψ̃₁ ≤ ℛ̃² by the sum of the linear term's maximum and
# the quadratic term's: b̃ = ℛ̃‖a‖ + ℛ̃²χ̃. With ξ = Π̃^{1/2}ψ̃₁, a = Π̃^{+1/2}ṽ and H = Π̃^{+1/2}sym(Q̃)Π̃^{+1/2},
# β = a′ξ + ξ′Hξ over ‖ξ‖ ≤ ℛ̃, and the exact maximum of |β| is
#     b_TR = max(TR(a, H), TR(−a, −H)),   TR(a, H) = max_{‖ξ‖≤ℛ̃} a′ξ + ξ′Hξ,
# a trust-region problem: ξ = (μI − H)⁻¹a/2 with μ ≥ max(λ_max(H), 0) and ‖ξ‖ = ℛ̃ (or interior, or
# the hard case). b_TR ≤ b̃ always. It is the tightest bound that uses ṽ and Q̃ only; what is left
# between b_TR and |β| is the class itself (which f the class contains), not the bound.
#
# The plug-in. b̃ needs only ‖a‖², which the debiased plug-in estimates consistently. b_TR needs a
# itself, and â = a + noise with noise of the same order as a (Proposition 7(iv) allows
# ℛ̃‖Υ̃‖_F/√V = O(1)). But TR is convex in a, with subgradient ξ*(a), so
# TR(â) ≥ TR(a) + ξ*′(â − a), and ξ*′(â − a) = O_p(ℛ̃‖Υ̃‖_op) = o_p(√V) by (iv): the raw b_TR(â) is
# conservative to first order, like the raw q̂ of Proposition 7. The rule tried is
#     b_min = min(b̃ with the debiased q̂, b_TR at the raw â),
# the smaller of two bounds that are each valid to first order.
#
# Checks:
# (0) the solver against brute force on small problems (random, and a built hard case), and
#     b_TR ≤ b̃; power: b_TR < b̃ by a margin where a is not aligned with H's top eigenvector.
# (1) the oracle decomposition at ℛ̃*, per candidate: |β̃|, b_TR(a) and b̃, all /√V, for f̃ in the
#     class, and the share (b̃ − b_TR)/(b̃ − |β̃|) of the looseness that is the bound.
# (2) coverage and half-length at λ̂ (REsd + 12 λ) of three rules: b̃ (the package's), b_TR(â) raw,
#     and b_min; ℛ̃ × 1 and 4. Power: none here beyond tc_refbasis's; the question is how much
#     shorter the new rules are, at what coverage.
# Design: tc_refbasis.jl's (n = 900, kNN-6, x, f = x + a·sin(2πu)cos(πv), heteroskedastic normal
# errors, ρ₀ = 0.4), estimation k ∈ {5, 15}, a ∈ {1, 4}, reference k̃ = 30, R = 300.
#
# Run: julia --project=. research/trace_cfe/tc_trust.jl > research/trace_cfe/tc_trust_output.txt
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
    (; Rinv = Diagonal(1 ./ sqrt.(ω)) * U', N0 = E.vectors[:, .!keep])
end
symm(Q) = (Q + Q') / 2
function sign_root(a, b, c)
    D = b^2 - 4a * c; D < 0 && return NaN
    for r in ((b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a))
        -0.99 < r < 0.99 && -b + 2a * r < 0 && return r
    end
    NaN
end
function cv(t; α = 0.05)
    t = abs(t); lo, hi = 0.0, t + 10
    for _ in 1:80
        c = (lo + hi) / 2
        cdf(Normal(), c - t) - cdf(Normal(), -c - t) < 1 - α ? (lo = c) : (hi = c)
    end
    (lo + hi) / 2
end
jl_E(A, MZ, mz) = A - Diagonal(diag(A * MZ) ./ mz)

# ── The trust-region maximum, from H's eigendecomposition (h ascending, U) ──
# max a′ξ + ξ′Hξ over ‖ξ‖ ≤ R.
function tr_max(α::Vector{Float64}, h::Vector{Float64}, R::Float64)
    R == 0 && return 0.0
    hmax = h[end]; na = norm(α)
    ξn(μ) = sqrt(sum(abs2, α ./ (2 .* (μ .- h))))
    val(ξ) = dot(α, ξ) + dot(ξ, h .* ξ)
    if hmax < 0 && ξn(0.0) <= R
        return val(α ./ (2 .* (0.0 .- h)))                       # interior maximum
    end
    lo = max(hmax, 0.0)
    top = findall(>=(hmax - 1e-12 * max(abs(hmax), 1.0)), h)
    if lo == hmax && sum(abs2, α[top]) <= 1e-24 * max(na^2, 1e-300)
        rest = setdiff(eachindex(h), top)
        ξ = zeros(length(h)); ξ[rest] = α[rest] ./ (2 .* (hmax .- h[rest]))
        if norm(ξ) <= R                                          # the hard case
            ξ[top[1]] = sqrt(R^2 - norm(ξ)^2)
            return val(ξ)
        end
    end
    a0 = lo; b0 = lo + na / (2R)                  # ‖ξ(b0)‖ ≤ R, and ‖ξ‖ falls in μ
    for _ in 1:200
        μ = (a0 + b0) / 2
        ξn(μ) > R ? (a0 = μ) : (b0 = μ)
    end
    val(α ./ (2 .* (b0 .- h)))
end
# max |a′ξ + ξ′Hξ|: for −H the eigenvalues are −h, ascending when reversed, with the coordinates
# reversed to match
b_trust_full(a, U, h, R) = max(tr_max(U' * a, h, R), tr_max(reverse(-(U' * a)), -reverse(h), R))

# (0) the solver against brute force
function check_solver()
    rng = MersenneTwister(1)
    worst = 0.0; gapmax = 0.0; sumgap = Inf; nbelow = 0
    for trial in 1:200
        p = 4; A = randn(rng, p, p); H = symm(A); a = randn(rng, p) .* (trial % 3 == 0 ? 0.1 : 1.0)
        R = 0.5 + rand(rng)
        E = eigen(Symmetric(H)); bt = b_trust_full(a, E.vectors, E.values, R)
        best = 0.0
        for _ in 1:20000
            z = randn(rng, p); z .*= R * rand(rng)^(1 / p) / norm(z)
            best = max(best, abs(dot(a, z) + dot(z, H * z)))
        end
        for _ in 1:20000                                         # the boundary, densely
            z = randn(rng, p); z .*= R / norm(z)
            best = max(best, abs(dot(a, z) + dot(z, H * z)))
        end
        worst = max(worst, (best - bt) / bt)                     # > 0 would mean the solver misses
        gapmax = max(gapmax, (bt - best) / bt)
        bs = R * norm(a) + R^2 * maximum(abs, E.values)
        bt <= bs * (1 + 1e-12) || (nbelow += 1)
        sumgap = min(sumgap, (bs - bt) / bs)
    end
    # a built hard case: a ⊥ the top eigenvector, and small
    H = Diagonal([3.0, 1.0, -1.0]); a = [0.0, 0.1, 0.2]; R = 1.0
    E = eigen(Symmetric(Matrix(H))); bt = b_trust_full(a, E.vectors, E.values, R)
    best = 0.0; rng2 = MersenneTwister(2)
    for _ in 1:200000
        z = randn(rng2, 3); z .*= R / norm(z); best = max(best, abs(dot(a, z) + dot(z, H * z)))
    end
    @printf("(0) solver: 200 random problems (p = 4): brute force above b_TR by at most %.1e relative, below by at most %.1e; b_TR > b̃ in %d; smallest (b̃ − b_TR)/b̃ %.3f\n",
            worst, gapmax, nbelow, sumgap)
    # exact: ξ = a/(2(3 − h)) off the top axis, the rest of the radius on it
    ξr = a[2:3] ./ (2 .* (3.0 .- [1.0, -1.0]))
    ex = 3.0 * (R^2 - sum(abs2, ξr)) + dot(a[2:3], ξr) + dot(ξr, [1.0, -1.0] .* ξr)
    @printf("    hard case (a ⊥ top eigenvector): b_TR %.6f, brute force %.6f, exact %.6f\n", bt, best, ex)
    println()
end

function setup(; n = 900, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; n, W, Wd = Matrix(W), u, v, x, σ2, Fac = lu(sparse(I - 0.4 * W)))
end

const ΛS = 10.0 .^ (-3:0.5:2)

function cellobjs(d, k, amp, kt)
    Wd = d.Wd
    Z = basis(d.u, d.v, d.x, k); Π = penalty(k); K = size(Z, 2)
    Zt = basis(d.u, d.v, d.x, kt); Πt = penalty(kt)
    Q = Matrix(qr(Z).Q)[:, 1:K]; Pz = Q * Q'; MZ = I - Pz; mz = diag(MZ)
    rt = rangeparts(Πt)
    f = d.x .+ amp .* g_surf.(d.u, d.v)
    ψt = interp_coef(kt, amp); ft = Zt * ψt; ℛt = sqrt(dot(ψt, Πt * ψt))
    MZZt = MZ * Zt
    G0 = Wd / (I - 0.4 * Wd)
    objs = map(ΛS) do λ
        S = Z * ((Z' * Z + λ * Π) \ Z'); S = symm(S); M = I - S
        dd = (M .* M) \ (-diag(M * Wd)); Dg = Diagonal(dd); A = M * Wd + M * Dg * M
        As = symm(A); H = (A + A') * S
        MZt = M * Zt; Tt = MZt' * Wd
        Qt = MZt' * (Wd * MZZt) + MZt' * Dg * MZt
        Eh = eigen(Symmetric(rt.Rinv * symm(Qt) * rt.Rinv'))
        b = 2 .* (As * f); V = 2 * dot(d.σ2, (As .^ 2) * d.σ2) + sum(b .^ 2 .* d.σ2)
        βt = dot(ft, A * ft); at = rt.Rinv * (Tt * (Pz * ft))
        (; λ, A, As2 = As .^ 2, H, H2 = H .^ 2, M, m2 = diag(M * M), Gt = rt.Rinv * Tt * Pz,
           U = Eh.vectors, h = Eh.values, χ = maximum(abs, Eh.values), V, βt,
           sumb = ℛt * norm(at) + ℛt^2 * maximum(abs, Eh.values),
           trb = b_trust_full(at, Eh.vectors, Eh.values, ℛt))
    end
    E0 = jl_E(G0', MZ, mz); C0 = E0 * MZ; Cs0 = symm(C0)
    b0 = 2 .* (Cs0 * f); V0 = 2 * dot(d.σ2, (Cs0 .^ 2) * d.σ2) + sum(b0 .^ 2 .* d.σ2)
    βRt = dot(ft, C0 * ft); aR = rt.Rinv * (MZZt' * (E0' * (Pz * ft)))
    ER = eigen(Symmetric(rt.Rinv * symm(MZZt' * E0 * MZZt) * rt.Rinv'))
    R0 = (; V = V0, βt = βRt, sumb = ℛt * norm(aR) + ℛt^2 * maximum(abs, ER.values),
          trb = b_trust_full(aR, ER.vectors, ER.values, ℛt))
    (; K, Pz, MZ, mz, Q, MZZt, rt, f, ℛt, objs, R0)
end

# rules: 1 = b̃ (debiased, the package's), 2 = b_TR at the raw â, 3 = min of the two
function draw(d, c, y, mults)
    Wd = d.Wd; Wy = d.W * y
    C1 = jl_E(Wd', c.MZ, c.mz) * c.MZ
    q1(C) = (dot(Wy, C * Wy), dot(y, C * Wy) + dot(Wy, C * y), dot(y, C * y))
    ρ̃ = sign_root(q1(C1)...); isnan(ρ̃) && return nothing
    y0p = y .- ρ̃ .* Wy; s0 = (c.MZ * y0p) .^ 2 ./ c.mz; fh = c.Pz * y0p
    L = length(c.objs) + 1; nm = length(mults)
    hl = fill(Inf, L, 3, nm); cov = falses(L, 3, nm)
    debias(G) = (q = sum(abs2, G * fh) - dot(vec(sum(abs2, G; dims = 1)), s0); sqrt(max(q, 0.0)))
    function setc!(j, ρ, se, Vh, G, U, h, χ)
        lin = debias(G); âr = G * fh
        for (k, m) in enumerate(mults)
            ℛ = m * c.ℛt
            b1 = ℛ * lin + ℛ^2 * χ; b2 = b_trust_full(âr, U, h, ℛ)
            for (rr, bb) in ((1, b1), (2, b2), (3, min(b1, b2)))
                hh = cv(bb / sqrt(Vh)) * se; hl[j, rr, k] = hh; cov[j, rr, k] = abs(ρ - 0.4) <= hh
            end
        end
    end
    A2 = Matrix(lu(sparse(I - ρ̃ * d.W')) \ Wd'); E = jl_E(A2, c.MZ, c.mz); C2 = E * c.MZ
    a, bq, cc = q1(C2); ρ = sign_root(a, bq, cc)
    if !isnan(ρ)
        y0 = y .- ρ .* Wy; s = (c.MZ * y0) .^ 2 ./ c.mz
        Hm = (C2 + C2') * c.Q * c.Q'; â = Hm * y0
        Vh = 2 * dot(s, (symm(C2) .^ 2) * s) + sum((â .^ 2 .- (Hm .^ 2) * s) .* s)
        if Vh > 0
            se = sqrt(Vh) / abs(-bq + 2a * ρ)
            GR = c.rt.Rinv * (E * c.MZZt)' * c.Pz
            ER = eigen(Symmetric(c.rt.Rinv * symm(c.MZZt' * E * c.MZZt) * c.rt.Rinv'))
            setc!(1, ρ, se, Vh, GR, ER.vectors, ER.values, maximum(abs, ER.values))
        end
    end
    for (j, o) in enumerate(c.objs)
        Ay = o.A * y; AWy = o.A * Wy
        a = dot(Wy, AWy); bq = dot(y, AWy) + dot(Wy, Ay); cc = dot(y, Ay)
        ρ = sign_root(a, bq, cc); isnan(ρ) && continue
        y0 = y .- ρ .* Wy; s = (o.M * y0) .^ 2 ./ o.m2
        â = o.H * y0; Vh = 2 * dot(s, o.As2 * s) + sum((â .^ 2 .- o.H2 * s) .* s)
        Vh > 0 || continue
        setc!(j + 1, ρ, sqrt(Vh) / abs(-bq + 2a * ρ), Vh, o.Gt, o.U, o.h, o.χ)
    end
    (; hl, cov)
end

const RULES = ("b̃ (package)", "b_TR raw", "min")

function cell(d, k, amp; kt = 30, R = 300, mults = (1.0, 4.0), seed = 7)
    t0 = time()
    c = cellobjs(d, k, amp, kt)
    @printf("k = %d (K = %d), a = %.0f, k̃ = %d, ℛ̃* = %.3f\n", k, c.K, amp, kt, c.ℛt)
    println("  (1) oracle at ℛ̃*, /√V: |β̃|, b_TR, b̃, and the share of b̃ − |β̃| that is the bound")
    for (lab, o) in (("REsd@ρ₀  ", c.R0), [(@sprintf("λ %8.4f", o.λ), o) for o in c.objs[1:2:end]]...)
        sV = sqrt(o.V); share = (o.sumb - o.trb) / (o.sumb - abs(o.βt))
        @printf("    %s |β̃| %7.3f  b_TR %7.3f  b̃ %7.3f  share %.2f %s\n", lab, abs(o.βt) / sV,
                o.trb / sV, o.sumb / sV, share, abs(o.βt) <= o.trb * (1 + 1e-10) ? "" : "EXCEEDED")
    end
    rng = MersenneTwister(seed + 10k + round(Int, amp))
    nm = length(mults); cov = zeros(3, nm); hl = zeros(3, nm); nR = zeros(Int, 3, nm); m = 0
    for r in 1:R
        y = d.Fac \ (c.f .+ sqrt.(d.σ2) .* randn(rng, d.n))
        o = draw(d, c, y, mults); o === nothing && continue
        m += 1
        for rr in 1:3, kk in 1:nm
            j = argmin(o.hl[:, rr, kk]); isfinite(o.hl[j, rr, kk]) || continue
            cov[rr, kk] += o.cov[j, rr, kk]; hl[rr, kk] += o.hl[j, rr, kk]; nR[rr, kk] += 1
        end
    end
    println("  (2) at λ̂ (REsd + 12 λ), $m draws:")
    for kk in 1:nm, rr in 1:3
        @printf("    ℛ̃ × %.0f, %-13s coverage %.3f, half-length %.4f (%+.0f%% against b̃)\n", mults[kk],
                RULES[rr], cov[rr, kk] / nR[rr, kk], hl[rr, kk] / nR[rr, kk],
                100 * (hl[rr, kk] / nR[rr, kk] / (hl[1, kk] / nR[1, kk]) - 1))
    end
    @printf("  %.1f min\n\n", (time() - t0) / 60)
end

check_solver()
d = setup()
cell(d, 5, 1.0; R = 2); println("(warm-up above)\n")
t0 = time()
for amp in (1.0, 4.0), k in (5, 15)
    cell(d, k, amp)
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
