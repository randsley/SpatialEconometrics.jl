# A λ rule with a coverage guarantee: bias-corrected moments (THEORY.md, "Still to do" item 4).
#
# tc_lambda.jl: the smoother's bias β(λ) = f′A_λf grows with λ, and a plug-in ĉ_β is too noisy to
# choose λ. Here the bias is removed from the moment matrix instead of being estimated. With the
# projection P = P_B as pilot and M₀ = I − P,
#
#     C_λ = A_λ − P A_λ P + M₀ D₀ M₀,   (M₀∘M₀) d₀ = −diag(A_λ − P A_λ P),
#
# so diag C_λ = 0 (exact mean zero under heteroskedasticity) and f′C_λ f = f′A_λ f − f′A_λ f = 0 for
# every f in span(B), for every λ. C_0 is the projection moment A_0. Every member of the family has
# zero bias when f ∈ span(B), so λ can be chosen by the estimated variance alone:
#     λ̂ = argmin_λ V̂_λ(ρ̂_λ) / g′_λ(ρ̂_λ)².
# Questions: (1) the exact sd √V/|J| of C_λ across λ, against the projection's and the plain
# smoother's; (2) the exact bias c(λ) = f′C_λ f/√V when f ∉ span(B) (the hat basis does not contain
# sin); (3) Wald coverage at each λ and at λ̂; (4) power: the same rule on the plain (biased) smoother
# must under-cover. Same design as tc_lambda.jl (n = 900, hat basis kh = 15 plus x, kNN-6 W,
# heteroskedastic), R = 500, amplitudes 1 and 4. Roots by the sign rule (g′ < 0).
#
# Run: julia --project=. research/trace_cfe/tc_rbc.jl > research/trace_cfe/tc_rbc_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))

"""The root of c − bρ + aρ² in (−0.99, 0.99) where the slope −b + 2aρ < 0; NaN if none."""
function sign_root(a, b, c)
    D = b^2 - 4a * c
    D < 0 && return NaN
    for r in ((b - sqrt(D)) / (2a), (b + sqrt(D)) / (2a))
        -0.99 < r < 0.99 && -b + 2a * r < 0 && return r
    end
    NaN
end

"""Per-λ objects for a moment matrix C, the pilot projector Pv (for â and its noise) and the residual
maker Mr with its diagonal of M_r² (for sᵢ = ûᵢ²/(M_r²)ᵢᵢ)."""
function prep(C, Pv, Mr)
    Cs = (C + C') / 2; H = (C + C') * Pv
    (; C, Cs2 = Cs .^ 2, H, H2 = H .^ 2, Mr, m2 = diag(Mr * Mr), Cs)
end
function exact(o, f, σ2, G)
    b = 2 .* (o.Cs * f)
    V = 2 * dot(σ2, o.Cs2 * σ2) + sum(b .^ 2 .* σ2)
    J = 2 * (dot(G * f, o.Cs * f) + tr(G' * o.Cs * Diagonal(σ2)))
    (; c = dot(f, o.C * f) / sqrt(V), sd = sqrt(V) / abs(J))
end
function fit(o, y, Wy)
    Cy = o.C * y; CWy = o.C * Wy
    a = dot(Wy, CWy); b = dot(y, CWy) + dot(Wy, Cy); c = dot(y, Cy)
    ρ = sign_root(a, b, c)
    isnan(ρ) && return (ρ = NaN, se = NaN, crit = Inf)
    y0 = y .- ρ .* Wy; û = o.Mr * y0; s = û .^ 2 ./ o.m2
    â = o.H * y0; V = 2 * dot(s, o.Cs2 * s) + sum((â .^ 2 .- o.H2 * s) .* s)
    sl = abs(-b + 2a * ρ)
    V > 0 || return (; ρ, se = NaN, crit = Inf)
    (; ρ, se = sqrt(V) / sl, crit = V / sl^2)
end

function run(; n = 900, kh = 15, amp = 1.0, R = 500, ρ0 = 0.4, seed = 5,
               λs = vcat(0.0, 10.0 .^ (-3:0.5:2)))
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
    Q = Matrix(qr(B).Q)[:, 1:rank(B)]; Pb = Q * Q'; M0 = I - Pb
    M0sq = M0 .* M0
    L = length(λs); plain = Vector{Any}(undef, L); rbc = Vector{Any}(undef, L)
    @printf("amp %.0f, n = %d, K = %d (K/n = %.2f), R = %d\n", amp, n, size(Q, 2), size(Q, 2) / n, R)
    for (j, λ) in enumerate(λs)
        S = λ == 0 ? Pb : B * ((B' * B + λ * Ωp) \ B'); S = (S + S') / 2
        M = I - S; d = (M .* M) \ (-diag(M * Wd)); A = M * Wd + M * Diagonal(d) * M
        Ã = A - Pb * A * Pb; d0 = M0sq \ (-diag(Ã)); C = Ã + M0 * Diagonal(d0) * M0
        @assert maximum(abs, diag(C)) < 1e-8 && maximum(abs, Q' * (C + C') * Q) < 1e-7 * opnorm(C, 1)
        plain[j] = prep(A, S, M); rbc[j] = prep(C, Pb, M0)
    end
    ex_p = [exact(plain[j], f, σ2, G) for j in 1:L]; ex_r = [exact(rbc[j], f, σ2, G) for j in 1:L]
    Fac = lu(sparse(I - ρ0 * W))
    ρp = fill(NaN, R, L); sp = similar(ρp); cp = fill(Inf, R, L)
    ρr = fill(NaN, R, L); sr = similar(ρr); cr = fill(Inf, R, L)
    for r in 1:R
        y = Fac \ (f .+ sqrt.(σ2) .* randn(rng, n)); Wy = W * y
        for j in 1:L
            o = fit(plain[j], y, Wy); ρp[r, j] = o.ρ; sp[r, j] = o.se; cp[r, j] = o.crit
            o = fit(rbc[j], y, Wy);   ρr[r, j] = o.ρ; sr[r, j] = o.se; cr[r, j] = o.crit
        end
    end
    covers(ρ, se) = isfinite(ρ) && isfinite(se) && abs(ρ - ρ0) <= 1.96 * se
    summ(ρv, sv) = (ok = isfinite.(ρv); e = ρv[ok] .- ρ0;
                    (bias = mean(e), sd = std(e), rmse = sqrt(mean(e .^ 2)),
                     cov = mean(covers.(ρv, sv)), nf = count(.!ok)))
    println("  λ        | plain: c       sd*    bias     sd      cov   | bias-corrected: c   sd*    bias     sd      cov    no root")
    for j in 1:L
        a = summ(ρp[:, j], sp[:, j]); b = summ(ρr[:, j], sr[:, j])
        @printf("  %8.4f | %+8.3f %.4f %+.4f %.4f %.3f | %+8.3f %.4f %+.4f %.4f %.3f %3d\n", λs[j],
                ex_p[j].c, ex_p[j].sd, a.bias, a.sd, a.cov, ex_r[j].c, ex_r[j].sd, b.bias, b.sd, b.cov, b.nf)
    end
    for (name, ρm, sm, cm) in (("bias-corrected", ρr, sr, cr), ("plain (power check)", ρp, sp, cp))
        jh = [argmin(cm[r, :]) for r in 1:R]
        ρh = [ρm[r, jh[r]] for r in 1:R]; sh = [sm[r, jh[r]] for r in 1:R]
        s = summ(ρh, sh)
        picks = join(["$(round(λs[j], sigdigits = 3)):$(count(==(j), jh))" for j in 1:L if count(==(j), jh) > 0], " ")
        @printf("  λ̂ = argmin V̂/g′², %s: bias %+.4f sd %.4f RMSE %.4f coverage %.3f | picks %s\n",
                name, s.bias, s.sd, s.rmse, s.cov, picks)
    end
    println()
end

run(; R = 3, λs = [0.0, 1.0]); println("(warm-up above)\n")
t0 = time()
for amp in (1.0, 4.0)
    run(; amp)
end
println("total $(round((time() - t0) / 60, digits = 1)) min")
