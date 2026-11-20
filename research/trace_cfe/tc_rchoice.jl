# Choosing ℛ for Proposition 7's bias-aware interval.
#
# ℛ must bound the roughness ψ′Πψ of f. The data cannot give an upper bound (a rough perturbation of
# tiny amplitude is invisible but makes ψ′Πψ large), so any data-chosen ℛ loses the uniform guarantee.
# This script measures what the data can and cannot do, on tc_honest.jl's design:
# (1) a reference surface: for a hat basis with knot spacing h and a second-difference penalty,
#     ψ′Πψ ≈ h² ∫(f_uu² + f_vv²), so ℛ can be stated as a curvature bound. Checked against ℛ*.
# (2) the debiased estimate ℛ̂² = ψ̂′Πψ̂ − tr(noise), with ψ̂ the projection's coefficients at ρ̂₀,
#     and with shrunk coefficients from pilots λ_p = 0.1 and 1 (biased down); its mean and sd
#     against ℛ*², and a one-sided lower bound ℛ_L² = ℛ̂² − 1.645·se: how often ℛ_L ≤ ℛ*.
# (3) coverage and half-length of the interval at λ̂ (Proposition 7) with ℛ = k·ℛ* on a fine grid
#     of k (the sensitivity curve), and with the data rules ℛ̂, 2ℛ̂, ℛ_L and the pilots' ℛ̂.
# (4) a conclusion's breakdown: the largest k at which the interval excludes ρ = 0.2 (which the
#     projection's interval mostly does not), as a sensitivity report would show it.
# R = 500 at a = 1 and 4; sign-rule roots; Π, ψ, ℛ as in THEORY.md (the code says Ω for Π).
#
# Run: julia --project=. research/trace_cfe/tc_rchoice.jl > research/trace_cfe/tc_rchoice_output.txt
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
function cv(t; α = 0.05)
    t = abs(t); lo, hi = 0.0, t + 10
    for _ in 1:80
        c = (lo + hi) / 2
        cdf(Normal(), c - t) - cdf(Normal(), -c - t) < 1 - α ? (lo = c) : (hi = c)
    end
    (lo + hi) / 2
end

function run(; n = 900, kh = 15, amp = 1.0, R = 500, ρ0 = 0.4, seed = 5,
               λs = vcat(0.0, 10.0 .^ (-3:0.5:2)), λps = (0.1, 1.0), ρtest = 0.2)
    rng = MersenneTwister(seed)
    P = rand(rng, n, 2); W = knnW(P, 6); u, v = P[:, 1], P[:, 2]; Wd = Matrix(W)
    x = randn(rng, n)
    B = hcat(hatbasis(u, v, kh), x)
    D2 = diff(diff(Matrix(1.0I, kh, kh), dims = 1), dims = 1)
    Ωh = kron(D2' * D2, Matrix(1.0I, kh, kh)) + kron(Matrix(1.0I, kh, kh), D2' * D2)
    Ωp = zeros(kh^2 + 1, kh^2 + 1); Ωp[1:kh^2, 1:kh^2] = Ωh
    f = amp .* sin.(2π .* u) .* cos.(π .* v) .+ x
    σ2 = exp.(3 .* (u .+ v .- 1)); σ2 ./= mean(σ2)
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; Pb = Q * Q'
    E = eigen(Symmetric(Ωp)); keep = E.values .> 1e-10 * maximum(E.values)
    U = E.vectors[:, keep]; ω = E.values[keep]
    Ωplus = U * Diagonal(1 ./ ω) * U'; Ωhalf = U * Diagonal(sqrt.(ω)) * U'
    θs = B \ f; Rstar = sqrt(dot(θs, Ωp * θs))
    h = 1 / (kh - 1); Rcurv = h * sqrt(4.25 * π^4) * amp        # h·√∫(f_uu² + f_vv²) for a·sin(2πu)cos(πv)
    @printf("amp %.0f, n = %d, K = %d, R = %d\n", amp, n, size(B, 2), R)
    @printf("  (1) ℛ* = √(ψ*′Πψ*) = %.3f; the curvature formula h·√∫(f_uu² + f_vv²) = %.3f (ratio %.3f)\n",
            Rstar, Rcurv, Rstar / Rcurv)
    νΣ = maximum(eigvals(Symmetric(Ωp), Symmetric(B' * Diagonal(1 ./ σ2) * B)))
    @printf("      max ψ′Πψ over g = Bψ with g′Σ⁻¹g ≤ 0.2² (KL ≤ 0.02, total variation ≤ 0.1 under normal errors):
")
    @printf("      roughness 0.2·√ν = %.2f, that is %.2f ℛ*: such a g is nearly invisible, whatever its roughness says
",
            0.2 * sqrt(νΣ), 0.2 * sqrt(νΣ) / Rstar)
    L = length(λs); objs = Vector{Any}(undef, L)
    for (j, λ) in enumerate(λs)
        S = λ == 0 ? Pb : B * ((B' * B + λ * Ωp) \ B'); S = (S + S') / 2
        M = I - S; d = (M .* M) \ (-diag(M * Wd)); A = M * Wd + M * Diagonal(d) * M
        As = (A + A') / 2; H = (A + A') * S
        MB = M * B; Qq = MB' * Diagonal(d) * MB; Qq = (Qq + Qq') / 2
        Qr = Diagonal(1 ./ sqrt.(ω)) * (U' * Qq * U) * Diagonal(1 ./ sqrt.(ω))
        T = MB' * Wd
        Rn = Diagonal(1 ./ sqrt.(ω)) * (U' * (T * Pb))
        objs[j] = (; A, As2 = As .^ 2, H, H2 = H .^ 2, M, m2 = diag(M * M), T,
                     χ = maximum(abs, eigvals(Symmetric(Qr))), colsq = vec(sum(abs2, Rn; dims = 1)))
    end
    # roughness estimators: coefficient maps ψ̂ = Lₚ y₀, for the projection and the pilots
    pilots = vcat(0.0, collect(λps))
    Ls = [λp == 0 ? (B' * B) \ Matrix(B') : (B' * B + λp * Ωp) \ Matrix(B') for λp in pilots]
    HL = [Ωhalf * Lp for Lp in Ls]                                 # Π^{1/2}Lₚ
    nsq = [vec(sum(abs2, Hp; dims = 1)) for Hp in HL]              # ‖Π^{1/2}Lₚeᵢ‖²
    G0 = HL[1]' * HL[1]; G0sq = G0 .^ 2                             # for the projection's se
    Fac = lu(sparse(I - ρ0 * W))
    ρh = fill(NaN, R, L); se = fill(NaN, R, L); sV = fill(NaN, R, L); lin = fill(NaN, R, L)
    R2 = fill(NaN, R, length(pilots)); R2se = fill(NaN, R)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n)); Wy = W * y
        o1 = objs[1]; Ay = o1.A * y; AWy = o1.A * Wy
        ρp = sign_root(dot(Wy, AWy), dot(y, AWy) + dot(Wy, Ay), dot(y, Ay))
        isnan(ρp) && continue
        y0p = y .- ρp .* Wy; s0 = (o1.M * y0p) .^ 2 ./ o1.m2; fh = Pb * y0p
        for (k, Hp) in enumerate(HL)
            z = Hp * y0p; R2[r, k] = dot(z, z) - dot(nsq[k], s0)
        end
        g = HL[1]' * (HL[1] * y0p)
        R2se[r] = sqrt(max(4 * dot(s0, g .^ 2) + 2 * dot(s0, G0sq * s0), 0.0))
        for j in 1:L
            o = objs[j]; Ay = o.A * y; AWy = o.A * Wy
            a = dot(Wy, AWy); bq = dot(y, AWy) + dot(Wy, Ay); c = dot(y, Ay)
            ρ = sign_root(a, bq, c); isnan(ρ) && continue
            y0 = y .- ρ .* Wy; û = o.M * y0; s = û .^ 2 ./ o.m2
            â = o.H * y0; Vh = 2 * dot(s, o.As2 * s) + sum((â .^ 2 .- o.H2 * s) .* s)
            Vh > 0 || continue
            vh = o.T * fh; q = dot(vh, Ωplus * vh)
            ρh[r, j] = ρ; sV[r, j] = sqrt(Vh); se[r, j] = sqrt(Vh) / abs(-bq + 2a * ρ)
            lin[r, j] = sqrt(max(q - dot(o.colsq, s0), 0.0))
        end
    end
    χs = [o.χ for o in objs]
    halflen(r, j, ℛ) = isfinite(ρh[r, j]) ? cv((ℛ * lin[r, j] + ℛ^2 * χs[j]) / sV[r, j]) * se[r, j] : Inf
    function pick(r, ℛ)
        hs = [halflen(r, j, ℛ) for j in 1:L]; j = argmin(hs); (j, hs[j])
    end
    function rule(ℛv)                                              # ℛv: one ℛ per draw
        cov = 0; hl = 0.0; m = 0
        for r in 1:R
            isfinite(ℛv[r]) || continue
            j, hh = pick(r, ℛv[r]); isfinite(hh) || continue
            m += 1; hl += hh; cov += abs(ρh[r, j] - ρ0) <= hh
        end
        (cov = cov / m, hl = hl / m, m)
    end
    println("  (2) roughness estimates, as multiples of ℛ*² = $(round(Rstar^2, digits = 3)):")
    for (k, λp) in enumerate(pilots)
        e = filter(isfinite, R2[:, k]) ./ Rstar^2
        @printf("      pilot λ = %-4g: ℛ̂²/ℛ*² mean %8.3f sd %8.3f; share ≤ 0 %.3f\n",
                λp, mean(e), std(e), mean(e .<= 0))
    end
    RL2 = R2[:, 1] .- 1.645 .* R2se
    @printf("      projection: se/ℛ*² mean %.3f; lower bound ℛ_L² = ℛ̂² − 1.645 se: share ≤ ℛ*² %.3f, share > 0 %.3f\n",
            mean(filter(isfinite, R2se)) / Rstar^2, mean(filter(isfinite, RL2) .<= Rstar^2), mean(filter(isfinite, RL2) .> 0))
    println("  (3) coverage / mean half-length at λ̂; the projection's half-length is $(round(mean(filter(isfinite, se[:, 1])) * 1.96, digits = 4))")
    for k in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0)
        s = rule(fill(k * Rstar, R)); @printf("      ℛ = %5.2f ℛ*: %.3f / %.4f\n", k, s.cov, s.hl)
    end
    sq(x) = isfinite(x) ? sqrt(max(x, 0.0)) : NaN
    for (name, ℛv) in (("ℛ̂ (projection)", sq.(R2[:, 1])), ("2ℛ̂ (projection)", 2 .* sq.(R2[:, 1])),
                       ("ℛ_L", sq.(RL2)),
                       ["ℛ̂ (pilot λ = $(pilots[k]))" => sq.(R2[:, k]) for k in 2:length(pilots)]...,
                       ["2ℛ̂ (pilot λ = $(pilots[k]))" => 2 .* sq.(R2[:, k]) for k in 2:length(pilots)]...)
        s = rule(ℛv); m = mean(filter(isfinite, ℛv)) / Rstar
        @printf("      %-26s mean ℛ/ℛ* %6.2f: %.3f / %.4f\n", name, m, s.cov, s.hl)
    end
    # (4) breakdown for "the interval excludes ρtest"
    ks = 2.0 .^ (-3:0.25:6); bd = fill(NaN, R); proj_excl = 0
    for r in 1:R
        isfinite(ρh[r, 1]) || continue
        proj_excl += abs(ρh[r, 1] - ρtest) > 1.96 * se[r, 1]
        excl = [(jh = pick(r, k * Rstar); abs(ρh[r, jh[1]] - ρtest) > jh[2]) for k in ks]
        bd[r] = all(excl) ? Inf : (excl[1] ? ks[findfirst(!, excl) - 1] : 0.0)
    end
    b = filter(!isnan, bd)
    @printf("  (4) excluding ρ = %.1f: the projection's interval does in %.3f of draws; the largest k = ℛ/ℛ* at which\n", ρtest, proj_excl / R)
    @printf("      the bias-aware interval still does: none %.3f, below 1 %.3f, 1–4 %.3f, above 4 (finite) %.3f, always %.3f\n\n",
            mean(b .== 0), mean(0 .< b .< 1), mean(1 .<= b .<= 4), mean(4 .< b .< Inf), mean(isinf.(b)))
end

run(; R = 3); println("(warm-up above)\n")
t0 = time()
for amp in (1.0, 4.0)
    run(; amp)
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
