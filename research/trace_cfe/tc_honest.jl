# A λ rule with a coverage guarantee, II: a bias-aware (honest) interval under a roughness bound.
#
# tc_rbc.jl: unbiased moments gain nothing, so a guarantee with a real gain needs a bound on f. Take
# the class F(M) = {f = Bθ : θ′Ωθ ≤ M²}. Split θ = θ₀ + θ₁ with Ωθ₀ = 0 and θ₁ ⊥ null(Ω). Since
# S_λBθ₀ = Bθ₀, M_λf₀ = 0, and the smoother's bias in the moment is
#     β(λ) = f′A_λ f = f₁′M W f + f₁′M D M f₁        (A = MW + MDM, M = M_λ, D = D_λ).
# The linear term is v′θ₁ with v = B′MWf, so its worst case over θ′Ωθ ≤ M² is exactly M√(v′Ω⁺v)
# (v ⊥ null(Ω) because MBθ₀ = 0). The quadratic term is at most M²μ̄ with μ̄ = max |μ|, the
# generalised eigenvalues of (MB)′D(MB) against Ω on range(Ω). So |β| ≤ b̄ = M√(v′Ω⁺v) + M²μ̄.
# f enters b̄ only through v = T f, T = B′MW. It is plugged in as v̂ = T P_B y₀(ρ̂₀), with ρ̂₀ the
# projection's root (λ = 0, unbiased), once per draw: using each λ's own ρ̂ fails at large λ, where
# ρ̂_λ is badly biased (the first run of this script). v̂′Ω⁺v̂ is debiased by its noise term
# tr(Ω⁺ T P_B Σ̂ P_B T′), with Σ̂ the projection's HC2 sᵢ, and floored at 0.
#
# Honest interval: ρ̂ ± cv(b̄/√V̂) √V̂/|g′(ρ̂)|, with cv(t) the 1 − α quantile of |N(t, 1)|.
# Rule: λ̂ minimises the half-length over the grid. Checks:
# (1) validity of the bound: |β(λ)| ≤ b̄(λ) with the true f and M* = √(θ*′Ωθ*), θ* = B \ f
#     (f ∉ span(B) here, so the residual part of β is reported, not bounded);
# (2) coverage and length of the honest interval at each λ and at λ̂, for M = M*/2, M*, 2M*, 4M*;
# (3) power: M = 0 (b̄ = 0, the plain smoother with λ chosen by length) must under-cover, and so
#     may M*/2; (4) the plug-in b̄ against the oracle b̄ (true f).
# Same design as tc_lambda.jl and tc_rbc.jl, R = 500, amplitudes 1 and 4. Sign-rule roots.
#
# Run: julia --project=. research/trace_cfe/tc_honest.jl > research/trace_cfe/tc_honest_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

function sign_root(a, b, c)
    D = b^2 - 4a * c
    D < 0 && return NaN
    for r in ((b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a))
        -0.99 < r < 0.99 && -b + 2a * r < 0 && return r
    end
    NaN
end

"""cv(t): the 1 − α quantile of |N(t, 1)|, by bisection on Φ(c − t) − Φ(−c − t) = 1 − α."""
function cv(t; α = 0.05)
    t = abs(t); lo, hi = 0.0, t + 10
    for _ in 1:80
        c = (lo + hi) / 2
        cdf(Normal(), c - t) - cdf(Normal(), -c - t) < 1 - α ? (lo = c) : (hi = c)
    end
    (lo + hi) / 2
end

function run(; n = 900, kh = 15, amp = 1.0, R = 500, ρ0 = 0.4, seed = 5,
               λs = vcat(0.0, 10.0 .^ (-3:0.5:2)), mults = (0.0, 0.5, 1.0, 2.0, 4.0))
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]; Wd = Matrix(W)
    x = randn(rng, n)
    B = hcat(hatbasis(u, v, kh), x)
    D2 = diff(diff(Matrix(1.0I, kh, kh), dims = 1), dims = 1)
    Ωh = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    Ωp = zeros(kh^2 + 1, kh^2 + 1); Ωp[1:kh^2, 1:kh^2] = Ωh
    f = amp .* sin.(2π .* u) .* cos.(π .* v) .+ x
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    G = Wd / (I - ρ0 * Wd)
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; Pb = Q * Q'
    E = eigen(Symmetric(Ωp)); keep = E.values .> 1e-10 * maximum(E.values)
    U = E.vectors[:, keep]; ω = E.values[keep]                      # range(Ω) and its eigenvalues
    Ωplus = U * Diagonal(1 ./ ω) * U'
    θs = B \ f; Mstar = sqrt(dot(θs, Ωp * θs))
    @printf("amp %.0f, n = %d, K = %d, R = %d; M* = √(θ*′Ωθ*) = %.3f; ‖f − Bθ*‖/‖f‖ = %.4f\n",
            amp, n, size(B, 2), R, Mstar, norm(f - B * θs) / norm(f))
    L = length(λs)
    objs = Vector{Any}(undef, L)
    for (j, λ) in enumerate(λs)
        S = λ == 0 ? Pb : B * ((B' * B + λ * Ωp) \ B'); S = (S + S') / 2
        M = I - S; d = (M .* M) \ (-diag(M * Wd)); A = M * Wd + M * Diagonal(d) * M
        As = (A + A') / 2; H = (A + A') * S
        MB = M * B; Qq = MB' * Diagonal(d) * MB; Qq = (Qq + Qq') / 2
        Qr = Diagonal(1 ./ sqrt.(ω)) * (U' * Qq * U) * Diagonal(1 ./ sqrt.(ω))
        μbar = maximum(abs, eigvals(Symmetric(Qr)))
        T = MB' * Wd                                                 # v = T f
        b = 2 .* (As * f); V = 2 * dot(σ2, (As .^ 2) * σ2) + sum(b .^ 2 .* σ2)
        J = 2 * (dot(G * f, As * f) + tr(G' * As * Diagonal(σ2)))
        β = dot(f, A * f); vf = T * f
        Rn = Diagonal(1 ./ sqrt.(ω)) * (U' * (T * Pb)); colsq = vec(sum(abs2, Rn; dims = 1))
        objs[j] = (; A, As2 = As .^ 2, H, H2 = H .^ 2, M, m2 = diag(M * M), T, μbar, colsq,
                     τ = opnorm(Rn), φ = norm(Rn), β, V, J, lin = sqrt(dot(vf, Ωplus * vf)))
    end
    println("  (1) the bound with the true f, M = M*: c = β/√V, b̄/√V = (M*·lin + M*²·μ̄)/√V, sd* = √V/|J|;")
    println("      Proposition 7's plug-in sizes, Γ = Ω^{+1/2}TP_B: P1 = M*τ/√V, P2 = M*φ/√V, P3 = M*τ√K/√V")
    for j in 1:L
        o = objs[j]; bb = Mstar * o.lin + Mstar^2 * o.μbar; sV = sqrt(o.V)
        @printf("    λ %8.4f | c %+8.3f  b̄/√V %8.3f (linear %7.3f, quadratic %7.3f) | %s | sd* %.4f | P1 %.4f P2 %.3f P3 %.3f\n",
                λs[j], o.β / sV, bb / sV, Mstar * o.lin / sV, Mstar^2 * o.μbar / sV,
                abs(o.β) <= bb ? "holds  " : "exceeds", sV / abs(o.J),
                Mstar * o.τ / sV, Mstar * o.φ / sV, Mstar * o.τ * sqrt(size(Q, 2)) / sV)
    end
    Fac = lu(sparse(I - ρ0 * W)); nm = length(mults)
    cov = zeros(R, L, nm); len = fill(Inf, R, L, nm); ρh = fill(NaN, R, L)
    ratio = fill(NaN, R, L); ratio_raw = fill(NaN, R, L)
    @assert λs[1] == 0
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n)); Wy = W * y
        o1 = objs[1]; Ay = o1.A * y; AWy = o1.A * Wy                  # the pilot: the projection
        ρp = sign_root(dot(Wy, AWy), dot(y, AWy) + dot(Wy, Ay), dot(y, Ay))
        isnan(ρp) && continue
        y0p = y .- ρp .* Wy; s0 = (o1.M * y0p) .^ 2 ./ o1.m2; fh = Pb * y0p
        for j in 1:L
            o = objs[j]
            Ay = o.A * y; AWy = o.A * Wy
            a = dot(Wy, AWy); bq = dot(y, AWy) + dot(Wy, Ay); c = dot(y, Ay)
            ρ = sign_root(a, bq, c); ρh[r, j] = ρ
            isnan(ρ) && continue
            y0 = y .- ρ .* Wy; û = o.M * y0; s = û .^ 2 ./ o.m2
            â = o.H * y0; Vh = 2 * dot(s, o.As2 * s) + sum((â .^ 2 .- o.H2 * s) .* s)
            Vh > 0 || continue
            sl = abs(-bq + 2a * ρ); se = sqrt(Vh) / sl
            vh = o.T * fh; q = dot(vh, Ωplus * vh); ratio_raw[r, j] = sqrt(q) / o.lin
            linh = sqrt(max(q - dot(o.colsq, s0), 0.0)); ratio[r, j] = linh / o.lin
            for (k, m) in enumerate(mults)
                M_ = m * Mstar; t = (M_ * linh + M_^2 * o.μbar) / sqrt(Vh)
                h = cv(t) * se; len[r, j, k] = h; cov[r, j, k] = abs(ρ - ρ0) <= h
            end
        end
    end
    println("  (2) honest coverage / mean half-length per λ, by M/M*; plug-in lin / oracle lin: mean (sd) debiased, mean raw")
    @printf("    %-9s |%s| lin ratio\n", "λ", join([@sprintf("    M/M* = %-4.1f   ", m) for m in mults]))
    for j in 1:L
        cells = join([@sprintf(" %.3f / %.4f ", mean(cov[:, j, k]), mean(filter(isfinite, len[:, j, k])))
                      for k in 1:nm], "|")
        rd = filter(isfinite, ratio[:, j]); rr = filter(isfinite, ratio_raw[:, j])
        @printf("    %8.4f |%s| %.3f (%.3f), raw %.3f\n", λs[j], cells, mean(rd), std(rd), mean(rr))
    end
    println("  (3) the rule λ̂ = argmin half-length:")
    for (k, m) in enumerate(mults)
        jh = [argmin(len[r, :, k]) for r in 1:R]
        cv_ = mean(cov[r, jh[r], k] for r in 1:R); hl = mean(len[r, jh[r], k] for r in 1:R)
        e = [ρh[r, jh[r]] - ρ0 for r in 1:R]
        picks = join(["$(round(λs[j], sigdigits = 3)):$(count(==(j), jh))" for j in 1:L if count(==(j), jh) > 0], " ")
        @printf("    M/M* = %.1f: coverage %.3f, half-length %.4f, bias %+.4f, sd %.4f | picks %s\n",
                m, cv_, hl, mean(e), std(e), picks)
    end
    @printf("    reference: the projection's Wald half-length %.4f\n\n", mean(filter(isfinite, len[:, 1, 1])))
end

run(; R = 3, λs = [0.0, 1.0]); println("(warm-up above)\n")
t0 = time()
for amp in (1.0, 4.0)
    run(; amp)
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
