# The bias-aware interval when f ∉ span(Z): a roughness class on a finer reference basis.
#
# Lemma 4 bounds the smoother's bias over F(ℛ) = {Zψ : ψ′Πψ ≤ ℛ²}, with Z = [B, x] the estimation
# basis. A real f is not in span(Z), and the remainder r = M_Z f adds to β a part that b̄ ignores; the
# projection candidate (REsd) is given b̄ = 0, which is wrong when r matters (tc_approx.jl: c up to
# 2.9 at K = 17). Here the class is put on a finer reference basis Z̃ = [B̃, x] with penalty Π̃:
#     F̃(ℛ̃) = {Z̃ψ̃ : ψ̃′Π̃ψ̃ ≤ ℛ̃²}.
# If Z̃·null(Π̃) ⊂ span(Z) (true for tensor hat bases on the same box: both reproduce the bilinear
# surfaces and x), then M_λf₀ = 0 and M_Zf₀ = 0, and with f_z = P_Zf, r = M_Zf = M_ZZ̃ψ̃₁:
#   smoother, A = MW + MDM:   β = ψ̃₁′ṽ + ψ̃₁′Q̃ψ̃₁,   ṽ = (MZ̃)′Wf_z,
#                             Q̃ = (MZ̃)′W(M_ZZ̃) + (MZ̃)′D(MZ̃);
#   REsd, C = EM_Z, E = A₂ − Δ₂:  β = ψ̃₁′ṽ_R + ψ̃₁′Q̃_Rψ̃₁,  ṽ_R = (M_ZZ̃)′E′f_z,  Q̃_R = (M_ZZ̃)′E(M_ZZ̃).
# Both ṽ are ⊥ null(Π̃), so Lemma 4's two steps give |β| ≤ ℛ̃√(ṽ′Π̃⁺ṽ) + ℛ̃²χ̃, χ̃ = max|eig| of
# Π̃^{+1/2}sym(Q̃)Π̃^{+1/2}. With Z̃ = Z this is Lemma 4; at λ = 0 it is Proposition 3's split.
# ṽ needs only f_z, plugged in as P_Zy₀(ρ̂₀) (ρ̂₀: Jin & Lee's first step, as in the package) and
# debiased as before.
#
# Checks:
# (0) the split is an identity: β(f̃) = ψ̃₁′ṽ + ψ̃₁′Q̃ψ̃₁ for f̃ = Z̃ψ̃, at every λ and for REsd;
#     Z̃·null(Π̃) ⊂ span(Z) numerically.
# (1) the oracle bound holds for f̃ ∈ F̃(ℛ̃*); power: the coarse bound (Lemma 4 on Z, ℛ from the
#     least-squares fit on Z, REsd at 0) is exceeded where r matters.
# (2) coverage at λ̂ (shortest interval over REsd + 12 smoother λ), old rule against new, ℛ × {1, 4};
#     power: the old rule under-covers at small K.
# (3) the cost: half-lengths, and the share of draws choosing REsd.
# (4) convergence of the oracle bound as the reference grid is refined (k̃ = 15, 20, 30, 40).
# ℛ̃* is the roughness of the interpolant of f at the reference knots (a least-squares fit is not
# defined once K̃ > n); f − f̃ is reported, not bounded.
# Design: tc_honest's (n = 900, kNN-6, x, f = x + a·sin(2πu)cos(πv), heteroskedastic normal errors,
# ρ₀ = 0.4); estimation k ∈ {4, 5, 7, 15}, a ∈ {1, 4}, reference k̃ = 30, R = 300.
#
# Run: julia --project=. research/trace_cfe/tc_refbasis.jl > research/trace_cfe/tc_refbasis_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
function penalty(k)                                  # second differences in each direction; x free
    D2 = diff(diff(Matrix(1.0I, k, k), dims = 1), dims = 1)
    Ω = kron(D2' * D2, Matrix(1.0I, k, k)) + kron(Matrix(1.0I, k, k), D2' * D2)
    Π = zeros(k^2 + 1, k^2 + 1); Π[1:k^2, 1:k^2] = Ω; Π
end
basis(u, v, x, k) = hcat(hatbasis(u, v, k), x)
g_surf(u, v) = sin(2π * u) * cos(π * v)
function interp_coef(k, amp)                         # the hat interpolant of f at the knots, x coef 1
    t = range(0, 1, length = k)
    vcat([amp * g_surf(t[i], t[j]) for i in 1:k for j in 1:k], 1.0)
end
function rangeparts(Π)
    E = eigen(Symmetric(Π)); keep = E.values .> 1e-10 * maximum(E.values)
    U = E.vectors[:, keep]; ω = E.values[keep]
    (; Rinv = Diagonal(1 ./ sqrt.(ω)) * U', Πp = U * Diagonal(1 ./ ω) * U', N0 = E.vectors[:, .!keep])
end
symm(Q) = (Q + Q') / 2
chi(Rinv, Q) = maximum(abs, eigvals(Symmetric(Rinv * symm(Q) * Rinv')))

function sign_root(a, b, c)
    D = b^2 - 4a * c
    D < 0 && return NaN
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
# Jin & Lee's C = (A − Δ)M_Z, Δ = Diag(AM_Z)/Diag(M_Z): the factor E = A − Δ.
jl_E(A, MZ, mz) = A - Diagonal(diag(A * MZ) ./ mz)

function setup(; n = 900, seed = 5)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]
    x = randn(rng, n)
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    (; n, W, Wd = Matrix(W), u, v, x, σ2, Fac = lu(sparse(I - 0.4 * W)))
end

const ΛS = 10.0 .^ (-3:0.5:2)

# Everything that depends on the design only, for one (k, a): per λ, the smoother's matrices and
# both bounds' fixed parts; REsd's parts are per draw.
function cellobjs(d, k, amp, kt)
    n = d.n; Wd = d.Wd
    Z = basis(d.u, d.v, d.x, k); Π = penalty(k); K = size(Z, 2)
    Zt = basis(d.u, d.v, d.x, kt); Πt = penalty(kt)
    Q = Matrix(qr(Z).Q)[:, 1:K]; Pz = Q * Q'; MZ = I - Pz; mz = diag(MZ)
    rc = rangeparts(Π); rt = rangeparts(Πt)
    f = d.x .+ amp .* g_surf.(d.u, d.v)
    ψt = interp_coef(kt, amp); ft = Zt * ψt; ℛt = sqrt(dot(ψt, Πt * ψt))
    ψc = Z \ f; ℛc = sqrt(dot(ψc, Π * ψc))
    nullgap = norm(MZ * (Zt * rt.N0)) / norm(Zt * rt.N0)
    MZZt = MZ * Zt
    ψt1 = ψt - rt.N0 * (rt.N0' * ψt)
    G0 = Wd / (I - 0.4 * Wd)
    objs = map(ΛS) do λ
        S = Z * ((Z' * Z + λ * Π) \ Z'); S = symm(S); M = I - S
        dd = (M .* M) \ (-diag(M * Wd)); Dg = Diagonal(dd); A = M * Wd + M * Dg * M
        As = symm(A); H = (A + A') * S
        MZc = M * Z; MZt = M * Zt
        Tc = MZc' * Wd; Tt = MZt' * Wd
        Qt = MZt' * (Wd * MZZt) + MZt' * Dg * MZt
        χc = chi(rc.Rinv, MZc' * Dg * MZc); χt = chi(rt.Rinv, Qt)
        b = 2 .* (As * f); V = 2 * dot(d.σ2, (As .^ 2) * d.σ2) + sum(b .^ 2 .* d.σ2)
        # (0) the identity, (1) the bounds with the true f_z / f̃
        βt = dot(ft, A * ft); vt = Tt * (Pz * ft)
        ident = abs(βt - (dot(ψt1, vt) + dot(ψt1, Qt * ψt1))) / max(abs(βt), 1.0)
        oracle_t = ℛt * sqrt(dot(vt, rt.Πp * vt)) + ℛt^2 * χt
        vc = Tc * f; oracle_c = ℛc * sqrt(dot(vc, rc.Πp * vc)) + ℛc^2 * χc
        (; λ, A, As2 = As .^ 2, H, H2 = H .^ 2, M, m2 = diag(M * M),
           Gc = rc.Rinv * Tc * Pz, Gt = rt.Rinv * Tt * Pz, χc, χt, V, β = dot(f, A * f), βt,
           ident, oracle_t, oracle_c)
    end
    # REsd at ρ̃ = ρ₀ (deterministic, for the table): its bias with f and f̃, and both bounds.
    E0 = jl_E(G0', MZ, mz); C0 = E0 * MZ; Cs0 = symm(C0)
    b0 = 2 .* (Cs0 * f); V0 = 2 * dot(d.σ2, (Cs0 .^ 2) * d.σ2) + sum(b0 .^ 2 .* d.σ2)
    βR = dot(f, C0 * f); βRt = dot(ft, C0 * ft)
    vR = MZZt' * (E0' * (Pz * ft)); QR = MZZt' * E0 * MZZt
    identR = abs(βRt - (dot(ψt1, vR) + dot(ψt1, QR * ψt1))) / max(abs(βRt), 1.0)
    oracleR = ℛt * sqrt(dot(vR, rt.Πp * vR)) + ℛt^2 * chi(rt.Rinv, QR)
    (; Z, K, Pz, MZ, mz, Q, Zt, MZZt, rt, f, ft, ℛt, ℛc, nullgap, objs,
       R0 = (; V = V0, β = βR, βt = βRt, ident = identR, oracle = oracleR))
end

# One draw: the pilot, REsd and every smoother λ; returns, per candidate and rule × mult, the
# half-length and coverage.
function draw(d, c, y, mults)
    n = d.n; Wd = d.Wd; Wy = d.W * y
    # Jin & Lee's first step: A₁ = W′.
    C1 = jl_E(Wd', c.MZ, c.mz) * c.MZ
    q1(C) = (dot(Wy, C * Wy), dot(y, C * Wy) + dot(Wy, C * y), dot(y, C * y))
    a1, b1, c1 = q1(C1); ρ̃ = sign_root(a1, b1, c1)
    isnan(ρ̃) && return nothing
    y0p = y .- ρ̃ .* Wy; s0 = (c.MZ * y0p) .^ 2 ./ c.mz; fh = c.Pz * y0p
    L = length(c.objs) + 1; nm = length(mults)
    hl = fill(Inf, L, 2, nm); cov = falses(L, 2, nm); ρs = fill(NaN, L)
    debias(G) = (q = sum(abs2, G * fh) - dot(vec(sum(abs2, G; dims = 1)), s0); sqrt(max(q, 0.0)))
    # REsd
    A2 = Matrix(lu(sparse(I - ρ̃ * d.W')) \ Wd'); E = jl_E(A2, c.MZ, c.mz); C2 = E * c.MZ
    a, bq, cc = q1(C2); ρ = sign_root(a, bq, cc)
    if !isnan(ρ)
        y0 = y .- ρ .* Wy; s = (c.MZ * y0) .^ 2 ./ c.mz
        Kq = (C2 + C2') * c.Q; Hm = Kq * c.Q'; â = Hm * y0
        Cs = symm(C2)
        Vh = 2 * dot(s, (Cs .^ 2) * s) + sum((â .^ 2 .- (Hm .^ 2) * s) .* s)
        if Vh > 0
            se = sqrt(Vh) / abs(-bq + 2a * ρ); ρs[1] = ρ
            TR = (E * c.MZZt)'; GR = c.rt.Rinv * TR * c.Pz
            linR = debias(GR); χR = chi(c.rt.Rinv, c.MZZt' * E * c.MZZt)
            for (k, m) in enumerate(mults)
                for (rr, bb) in ((1, 0.0), (2, m * c.ℛt * linR + (m * c.ℛt)^2 * χR))
                    h = cv(bb / sqrt(Vh)) * se; hl[1, rr, k] = h; cov[1, rr, k] = abs(ρ - 0.4) <= h
                end
            end
        end
    end
    for (j, o) in enumerate(c.objs)
        Ay = o.A * y; AWy = o.A * Wy
        a = dot(Wy, AWy); bq = dot(y, AWy) + dot(Wy, Ay); cc = dot(y, Ay)
        ρ = sign_root(a, bq, cc); isnan(ρ) && continue
        y0 = y .- ρ .* Wy; s = (o.M * y0) .^ 2 ./ o.m2
        â = o.H * y0; Vh = 2 * dot(s, o.As2 * s) + sum((â .^ 2 .- o.H2 * s) .* s)
        Vh > 0 || continue
        se = sqrt(Vh) / abs(-bq + 2a * ρ); ρs[j + 1] = ρ
        linc = debias(o.Gc); lint = debias(o.Gt)
        for (k, m) in enumerate(mults)
            bo = m * c.ℛc * linc + (m * c.ℛc)^2 * o.χc
            bn = m * c.ℛt * lint + (m * c.ℛt)^2 * o.χt
            for (rr, bb) in ((1, bo), (2, bn))
                h = cv(bb / sqrt(Vh)) * se; hl[j + 1, rr, k] = h; cov[j + 1, rr, k] = abs(ρ - 0.4) <= h
            end
        end
    end
    (; hl, cov, ρs)
end

function cell(d, k, amp; kt = 30, R = 300, mults = (1.0, 4.0), seed = 7)
    t0 = time()
    c = cellobjs(d, k, amp, kt)
    @printf("k = %d (K = %d), a = %.0f, reference k̃ = %d: ℛ̃* = %.3f (interpolant), ℛ_Z = %.3f (LS on Z); ‖f − f̃‖/‖f‖ = %.4f; Z̃·null(Π̃) outside span Z: %.1e\n",
            k, c.K, amp, kt, c.ℛt, c.ℛc, norm(c.f - c.ft) / norm(c.f), c.nullgap)
    println("  (0)/(1) per candidate: c = β/√V (true f), c̃ (f̃), identity error, oracle bounds /√V: old (Z, LS ℛ) and new (Z̃, ℛ̃*)")
    R0 = c.R0; sV = sqrt(R0.V)
    @printf("    REsd@ρ₀    c %+7.3f c̃ %+7.3f | id %.1e | old 0 %s | new %7.3f %s\n",
            R0.β / sV, R0.βt / sV, R0.ident, abs(R0.β) <= 0 ? "holds" : "EXCEEDED", R0.oracle / sV,
            abs(R0.βt) <= R0.oracle ? "holds" : "EXCEEDED")
    for o in c.objs
        sV = sqrt(o.V)
        @printf("    λ %8.4f c %+7.3f c̃ %+7.3f | id %.1e | old %7.3f %s | new %7.3f %s\n",
                o.λ, o.β / sV, o.βt / sV, o.ident, o.oracle_c / sV,
                abs(o.β) <= o.oracle_c ? "holds   " : "EXCEEDED", o.oracle_t / sV,
                abs(o.βt) <= o.oracle_t ? "holds" : "EXCEEDED")
    end
    rng = MersenneTwister(seed + 10k + round(Int, amp))
    nm = length(mults); L = length(ΛS) + 1
    cov = zeros(2, nm); hl = zeros(2, nm); nR = zeros(Int, 2, nm); m = 0
    picks = zeros(Int, 2, nm, L)
    for r in 1:R
        y = d.Fac \ (c.f .+ sqrt.(d.σ2) .* randn(rng, d.n))
        o = draw(d, c, y, mults); o === nothing && continue
        m += 1
        for rr in 1:2, kk in 1:nm
            j = argmin(o.hl[:, rr, kk]); isfinite(o.hl[j, rr, kk]) || continue
            cov[rr, kk] += o.cov[j, rr, kk]; hl[rr, kk] += o.hl[j, rr, kk]; nR[rr, kk] += 1
            picks[rr, kk, j] += 1
        end
    end
    println("  (2)/(3) at λ̂ (REsd + 12 λ), $m draws with a pilot:")
    for rr in 1:2, kk in 1:nm
        pk = join(["$(j == 1 ? "REsd" : round(ΛS[j-1], sigdigits = 3)):$(picks[rr, kk, j])"
                   for j in 1:L if picks[rr, kk, j] > 0], " ")
        @printf("    %s rule, ℛ × %.0f: coverage %.3f, half-length %.4f | %s\n",
                rr == 1 ? "old" : "new", mults[kk], cov[rr, kk] / nR[rr, kk], hl[rr, kk] / nR[rr, kk], pk)
    end
    @printf("  %.1f min\n\n", (time() - t0) / 60)
end

function convergence(d; k = 5, amp = 4.0, λ = 1.0, kts = (15, 20, 30, 40))
    println("(4) convergence of the oracle bound in k̃ (k = $k, a = $(Int(amp)), λ = $λ, and REsd at ρ₀):")
    for kt in kts
        c = cellobjs(d, k, amp, kt)
        o = c.objs[findfirst(==(λ), ΛS)]
        @printf("    k̃ = %2d (K̃ = %4d): ℛ̃*·(k̃−1) = %.3f | smoother: c̃ %+.3f, bound/√V %.3f | REsd: c̃ %+.3f, bound/√V %.3f\n",
                kt, size(c.Zt, 2), c.ℛt * (kt - 1), o.βt / sqrt(o.V), o.oracle_t / sqrt(o.V),
                c.R0.βt / sqrt(c.R0.V), c.R0.oracle / sqrt(c.R0.V))
    end
    println()
end

d = setup()
cell(d, 4, 1.0; R = 2); println("(warm-up above)\n")
t0 = time()
convergence(d)
for amp in (1.0, 4.0), k in (4, 5, 7, 15)
    cell(d, k, amp)
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
