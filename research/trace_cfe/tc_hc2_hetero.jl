# How far heteroskedasticity can push the HC2 V̂'s bias at ρ₀ below Lemma 8's homoskedastic worst case.
#
# With f = 0 and symmetric errors, every term of Lemma 7 is a quadratic form in the vector x = σ² of
# variances, and the cumulant terms are Σₖ cₖκₖ with Lemma 8's cₖ (𝒦 does not involve σ²). Every law
# has κₖ ≥ −2σₖ⁴, so the worst cumulants are κₖ = −2σₖ⁴ where cₖ > 0 and κₖ = κ̄σₖ⁴ where cₖ < 0
# (κ̄ = 4 here). Then
#     E V̂ − V = x′Nx,   V = x′(2Ω)x,
#     N = 2(A′ΩA − Ω) + T + 2(H∘M)′D⁻¹(H∘M) + sym(A′(H∘H)(I − A)) + Diag(−2c⁺ + κ̄c⁻),
# with A = D⁻¹(M∘M) (so σ̄² = Ax), D = Diag(M), and T the matrix of Q₂ + L₅ᴳ,
#     Tₖₗ = Σᵢ MᵢₖMᵢₗ (M′ Diag(W̃ᵢ.) M)ₖₗ,   W̃ = sym(4Ω − 2(H∘H)) ./ (MᵢᵢMⱼⱼ).
# The terms are Q₁; Q₂ + L₅ᴳ; L₃'s Gaussian part; L₄; the cumulants. The worst relative bias over
# variance patterns with max/min ratio at most r is
#     β_het(r) = min { x′Nx / x′(2Ω)x : x ∈ [1, r]ⁿ },
# a non-convex problem. It is searched by projected gradient from many starts, so each value
# found is attained (an upper bound on β_het(r)), not certified.
#
# A smooth class is searched too: x = 1 + (r − 1)·sigmoid(Bₛθ) with Bₛ the 4 × 4 tensor hats, by
# gradient descent on θ from 8 starts. Moran's I (on W) of each worst pattern says how rough it is.
#
# Checks: (1) x′Nx against Lemma 7 term by term (tc_hc2_bias.jl's formulas) at random x, to 1e-10
# relative; (2) at x = 1, N gives Lemma 8's β_min (power: the formula of (1) at a non-constant x
# differs from Lemma 8); (3) at the worst pattern found for r = 4, Monte Carlo with errors σᵢ·(±1)
# (R = 4000): E V̂/V against 1 + β, and the coverage of CS at ρ₀ with the oracle.
# Designs: tc_hc2_worst.jl's jittered grid (k = 13, 16), random locations (k = 10), rook (k = 13);
# n = 900, basis [hats, x], Jin & Lee's C(ρ₀), ρ₀ = 0.4.
#
# Run: julia --project=. research/trace_cfe/tc_hc2_hetero.jl > research/trace_cfe/tc_hc2_hetero_output.txt
include(joinpath(@__DIR__, "tc_lib.jl"))
using Printf, Distributions

hat(x, k) = (t = range(0, 1, length = k); hcat([max.(0, 1 .- abs.(x .- t[j]) .* (k - 1)) for j in 1:k]...))
hatbasis(u, v, k) = (Hu = hat(u, k); Hv = hat(v, k); hcat([Hu[:, i] .* Hv[:, j] for i in 1:k for j in 1:k]...))
symm(Q) = (Q + Q') / 2
const KBAR = 4.0

function rookW(side)
    n = side^2; I_ = Int[]; J_ = Int[]
    idx(i, j) = (i - 1) * side + j
    for i in 1:side, j in 1:side, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        a, b = i + di, j + dj
        1 <= a <= side && 1 <= b <= side && (push!(I_, idx(i, j)); push!(J_, idx(a, b)))
    end
    A = sparse(I_, J_, 1.0, n, n); A ./ sum(A; dims = 2)
end

function design(kind; side = 30, seed = 11)
    rng = MersenneTwister(seed); n = side^2
    P = if kind == :jittered
        (hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ rand(rng, n, 2)) ./ side
    elseif kind == :random
        rand(rng, n, 2)
    else
        (hcat(repeat(0:side-1, inner = side), repeat(0:side-1, outer = side)) .+ 0.5) ./ side
    end
    W = kind == :rook ? rookW(side) : knnW(P, 6)
    (; n, Wd = Matrix(W), u = P[:, 1], v = P[:, 2], x = randn(rng, n))
end

function objs(d, k)
    Z = hcat(hatbasis(d.u, d.v, k), d.x); Q = Matrix(qr(Z).Q)[:, 1:rank(Z)]
    Pz = Q * Q'; M = Matrix(I - Pz); Mii = diag(M)
    G = d.Wd / (I - 0.4 * d.Wd)
    E = G' .- Diagonal(diag(G' * M) ./ Mii); C = E * M
    Ω = symm(C) .^ 2; H = (C + C') * Pz; H2 = H .^ 2; M2 = M .^ 2
    U = M2 ./ Mii
    c = vec(2 .* sum(U .* (Ω * U); dims = 1) .+ sum(H2 .* M2 ./ Mii; dims = 1) .- sum(U .* (H2 * U); dims = 1))
    (; K = size(Q, 2), M, Mii, C, Ω, H, H2, M2, c, hmax = maximum(1 .- Mii))
end

# Lemma 7's bias at f = 0, μ₃ = 0, for variances x and cumulants κ (tc_hc2_bias.jl's predict).
function lemma7(o, x, κ)
    M, Mii, Ω = o.M, o.Mii, o.Ω
    σ̄ = (o.M2 * x) ./ Mii
    MΣM = M * (x .* M); 𝒳 = MΣM .^ 2; 𝒦 = o.M2 * (κ .* o.M2)
    Cov = (2 .* 𝒳 .+ 𝒦) ./ (Mii .* Mii')
    Q1 = 2 * dot(σ̄, Ω * σ̄) - 2 * dot(x, Ω * x)
    Q2 = 4 * sum(Ω .* 𝒳 ./ (Mii .* Mii'))
    Q3 = 2 * sum(Ω .* (𝒦 ./ (Mii .* Mii')))
    L3 = sum((2 .* vec(sum(o.H .* (x' .* M); dims = 2)) .^ 2 .+ (o.H2 .* o.M2) * κ) ./ Mii)
    L4 = sum(o.H2 .* (σ̄ .* (x .- σ̄)'))
    L5 = -sum(o.H2 .* Cov)
    Q1 + Q2 + Q3 + L3 + L4 + L5
end

function buildN(o)
    n = length(o.Mii); M, Mii, Ω, H2 = o.M, o.Mii, o.Ω, o.H2
    A = o.M2 ./ Mii
    N = 2 .* (A' * Ω * A .- Ω)
    Wt = symm(4 .* Ω .- 2 .* H2) ./ (Mii .* Mii')
    T = zeros(n, n); Mw = similar(M); B = similar(M)
    for i in 1:n
        Mw .= M .* view(Wt, :, i)                          # rows of M scaled by W̃[:, i] (= W̃[i, :])
        mul!(B, Mw', M)
        m = view(M, i, :)
        @inbounds for l in 1:n, k in 1:n
            T[k, l] += m[k] * m[l] * B[k, l]
        end
    end
    N .+= T
    GH = o.H .* M
    N .+= 2 .* (GH' * (GH ./ Mii))
    N .+= symm(A' * H2 * (I - A))
    N0 = copy(N)                                           # the Gaussian part
    N .+= Diagonal(-2 .* max.(o.c, 0) .+ KBAR .* min.(o.c, 0))
    (; N = symm(N), N0 = symm(N0))
end

ratio(N, S, x) = dot(x, N * x) / dot(x, S * x)

# Projected gradient on R(x) = x′Nx/x′Sx over [1, r]ⁿ, with Armijo backtracking.
function pgmin(N, S, x0, r; iters = 3000)
    x = clamp.(copy(x0), 1.0, r); R = ratio(N, S, x); step = 1.0
    for it in 1:iters
        Nx = N * x; Sx = S * x; den = dot(x, Sx)
        g = 2 .* (Nx .- R .* Sx) ./ den
        gn = norm(g); gn < 1e-14 && break
        accepted = false
        while step > 1e-12
            xn = clamp.(x .- step .* g, 1.0, r); Rn = ratio(N, S, xn)
            if Rn <= R - 1e-4 * dot(g, x .- xn)
                x, R = xn, Rn; step *= 2; accepted = true; break
            end
            step /= 2
        end
        accepted || break
    end
    x, R
end

moran(x, Wd) = (z = x .- mean(x); length(x) / sum(Wd) * dot(z, Wd * z) / dot(z, z))

# Gradient descent on R(x(θ)), x = 1 + (r − 1)·sigmoid(Bₛθ), from several starts.
function smoothmin(N, S, Bs, r, rng; iters = 3000)
    sig(t) = 1 / (1 + exp(-t))
    xof(θ) = 1 .+ (r - 1) .* sig.(Bs * θ)
    best = (Inf, ones(size(Bs, 1)))
    for s in 1:8
        θ = s == 1 ? zeros(size(Bs, 2)) : 3 .* randn(rng, size(Bs, 2))
        x = xof(θ); R = ratio(N, S, x); step = 1.0
        for it in 1:iters
            Nx = N * x; Sx = S * x; den = dot(x, Sx)
            gx = 2 .* (Nx .- R .* Sx) ./ den
            p = sig.(Bs * θ); gθ = Bs' * (gx .* (r - 1) .* p .* (1 .- p))
            norm(gθ) < 1e-12 && break
            ok = false
            while step > 1e-12
                θn = θ .- step .* gθ; xn = xof(θn); Rn = ratio(N, S, xn)
                if Rn <= R - 1e-4 * step * dot(gθ, gθ)
                    θ, x, R = θn, xn, Rn; step *= 2; ok = true; break
                end
                step /= 2
            end
            ok || break
        end
        R < best[1] && (best = (R, x))
    end
    best
end

function study(name, d, k, rng)
    o = objs(d, k); n = d.n
    t = @elapsed NN = buildN(o)
    N, N0 = NN.N, NN.N0; S = 2 .* o.Ω
    # (1) the quadratic form against Lemma 7, at random x, with κ = −2x² (c > 0) or κ̄x² (c < 0)
    errs = Float64[]
    for _ in 1:3
        x = 1 .+ 3 .* rand(rng, n)
        κ = ifelse.(o.c .> 0, -2.0, KBAR) .* x .^ 2
        push!(errs, abs(dot(x, N * x) - lemma7(o, x, κ)) / abs(lemma7(o, x, κ)))
    end
    # (2) homoskedastic: Lemma 8's β_min (with κ̄ where cₖ < 0, its part (c))
    x1 = ones(n); β1 = min(0.0, ratio(N, S, x1))
    homo = min(0.0, (sum(N0) - 2 * sum(max.(o.c, 0)) + KBAR * sum(min.(o.c, 0))) / sum(S))
    # power: Lemma 8's formula applied at a non-constant x is not Lemma 7
    xr = 1 .+ 3 .* rand(rng, n); κr = ifelse.(o.c .> 0, -2.0, KBAR) .* xr .^ 2
    l8 = (sum(N0) - 2 * sum(max.(o.c, 0)) + KBAR * sum(min.(o.c, 0))) / sum(S)
    pw = abs(lemma7(o, xr, κr) / dot(xr, S * xr) - l8)
    @printf("%s (K/n %.3f, max leverage %.3f; N built in %.0f s) | (1) x′Nx vs Lemma 7: max rel err %.1e | (2) β at x = 1: %+.4f, Lemma 8: %+.4f; power: |Lemma 7 − Lemma 8| at a random x %.4f\n",
            name, o.K / n, o.hmax, t, maximum(errs), β1, homo, pw)
    h = 1 .- o.Mii
    Bs = hatbasis(d.u, d.v, 4)
    out = Dict{Float64, Any}()
    for r in (2.0, 4.0, 10.0)
        starts = Any[ones(n), 1 .+ (r - 1) .* rand(rng, n), ifelse.(rand(rng, n) .< 0.5, 1.0, r),
                     1 .+ (r - 1) .* h ./ maximum(h), r .- (r - 1) .* h ./ maximum(h),
                     ifelse.(rand(rng, n) .< 0.1, r, 1.0), ifelse.(rand(rng, n) .< 0.9, r, 1.0)]
        for _ in 1:5; push!(starts, 1 .+ (r - 1) .* rand(rng, n) .^ 4); end
        best = (Inf, ones(n)); vals = Float64[]
        for x0 in starts
            x, R = pgmin(N, S, x0, r); push!(vals, R)
            R < best[1] && (best = (R, x))
        end
        xb = best[2]
        # the smooth class: x = 1 + (r − 1)·sigmoid(Bₛθ), Bₛ = 4 × 4 tensor hats
        sm = smoothmin(N, S, Bs, r, rng)
        @printf("    r = %4.0f: worst β found %+.4f (homoskedastic %+.4f) → coverage ≥ %.4f at most | spread over %d starts %+.4f to %+.4f | at the worst: %.0f%% of units at a bound, corr(x, leverage) %+.2f, Moran's I %+.2f | smooth class: worst %+.4f (Moran's I %+.2f)\n",
                r, best[1], β1, 2cdf(Normal(), 1.96sqrt(1 + best[1])) - 1, length(starts), minimum(vals),
                maximum(vals), 100 * mean((xb .<= 1 + 1e-9) .| (xb .>= r - 1e-9)), cor(xb, h),
                moran(xb, d.Wd), sm[1], moran(sm[2], d.Wd))
        out[r] = best
    end
    o, out
end

function simulate(o, x, β; R = 4000, seed = 9)
    rng = MersenneTwister(seed); n = length(x); sd = sqrt.(x)
    V = dot(x, (2 .* o.Ω) * x)
    r = zeros(R); cs = 0; oc = 0
    for i in 1:R
        y0 = sd .* rand(rng, (-1.0, 1.0), n)                # κ = −2σ⁴ everywhere
        s = (o.M * y0) .^ 2 ./ o.Mii
        Vh = 2 * dot(s, o.Ω * s) + sum(((o.H * y0) .^ 2 .- o.H2 * s) .* s)
        g = dot(y0, o.C * y0)
        r[i] = Vh / V; cs += Vh > 0 && g^2 <= 1.96^2 * Vh; oc += g^2 <= 1.96^2 * V
    end
    (; sim = mean(r), z = (mean(r) - 1 - β) / (std(r) / sqrt(R)), cs = cs / R, oc = oc / R)
end

println("The HC2 V̂'s worst-case bias at ρ₀ over heteroskedastic symmetric laws (f = 0); n = 900")
println("β_het(r): worst x′Nx/x′(2Ω)x found over variance patterns with max/min ≤ r, κₖ = −2σₖ⁴ (κ̄σₖ⁴ where cₖ < 0)\n")
rng = MersenneTwister(2026); t0 = time()
for (name, kind, k) in (("jittered k = 13", :jittered, 13), ("jittered k = 16", :jittered, 16),
                        ("random   k = 10", :random, 10), ("rook     k = 13", :rook, 13))
    o, out = study(name, design(kind), k, rng)
    # (3) Monte Carlo at the worst pattern for r = 4, with ±σᵢ errors
    β4, x4 = out[4.0]
    # the simulation draws κ = −2σ⁴ everywhere; its predicted β uses that, not κ̄ where cₖ < 0
    βsim = lemma7(o, x4, -2 .* x4 .^ 2) / dot(x4, (2 .* o.Ω) * x4)
    s = simulate(o, x4, βsim)
    @printf("    Monte Carlo at the r = 4 pattern, ±σᵢ errors (R = 4000): E V̂/V predicted %.4f, simulated %.4f (z %+.1f) | CS at ρ₀ %.4f, oracle %.4f, 2Φ(1.96√(1+β)) − 1 = %.4f   (%.1f min)\n\n",
            1 + βsim, s.sim, s.z, s.cs, s.oc, 2cdf(Normal(), 1.96sqrt(1 + βsim)) - 1, (time() - t0) / 60)
end
