# Lemma 11: a design-only bound on the HC2 V̂'s bias at ρ₀ under heteroskedastic symmetric errors,
# term by term, and its rate O(q²K/n).
#
# Class: σᵢ² = xᵢ ∈ [1, q] (by scale), μ₃ = 0, κₖ ∈ [−2xₖ², κ̄xₖ²], f ∈ span([X, B]). Write h = diag(P),
# h̄ = max h, rᵢ = hᵢ/Mᵢᵢ, ω = Ω1, κ* = max(2, κ̄), V₀ = 2·1′Ω1. Lemma 11's term bounds:
#     |Q₁|     ≤ 4q(q − 1)·h′ω
#     Q₂       ≤ 4(3q − 2)²·r′Ωr
#     |L₅ᴳ|    ≤ 2q²·‖H‖_F²
#     L₃ᴳ      ≤ 2(q − 1)²·Σᵢ hᵢ‖Hᵢ.‖²
#     |L₄|     ≤ q(q − 1)·Σₖ hₖ‖H.ₖ‖²
#     |Σcₖκₖ|  ≤ κ*q²·Σ|cₖ|,  Σ|cₖ| ≤ 2(2 + h̄)·r′Ωr + 2‖H‖_F²
#     |L₁|     ≤ (q − 1)·Σᵢ bᵢ²hᵢ
# and |β_V| ≤ (sum of the bounds)/(V₀ + ‖b‖²).
#
# Checks:
# (1) each inequality, at the worst pattern found by tc_hc2_hetero.jl's search, at 40 random
#     two-point patterns {1, q}ⁿ and 10 uniform ones, with κ = −2x², κ̄x² and the worst mix; L₁ at
#     f = x_reg and at f = P·g (in span). The smallest bound/|term| over all of them must be ≥ 1.
#     The terms add up to Lemma 7's bias (to 1e-10).
# (2) looseness: the bound against the worst |β| found.
# (3) power for the q-dependence: Lemma 9's homoskedastic β̄ (a bound with no q) against the worst
#     |β| found. Where it is exceeded, a bound free of q cannot hold.
# (4) the rate: β_het(4) and the bound against K/n on the jittered grid (n = 900, k = 7 to 16), and
#     at K/n ≈ 0.19 for n = 400, 900, 1600. If the bias is of order K/n, β_het/(K/n) stays bounded
#     and β_het does not vanish as n grows at fixed K/n.
#
# Run: julia --project=. research/trace_cfe/tc_hc2_hetero_rate.jl > research/trace_cfe/tc_hc2_hetero_rate_output.txt
include(joinpath(@__DIR__, "tc_hc2_hetero.jl"))
g_surf(u, v) = sin(2π * u) * cos(π * v)                       # as in tc_hc2_bias.jl

function terms(o, x, κ, b)
    M, Mii, Ω, H2 = o.M, o.Mii, o.Ω, o.H2
    σ̄ = (o.M2 * x) ./ Mii
    MΣM = M * (x .* M); 𝒳 = MΣM .^ 2; DD = Mii .* Mii'
    (; Q1 = 2 * dot(σ̄, Ω * σ̄) - 2 * dot(x, Ω * x),
       Q2 = 4 * sum(Ω .* 𝒳 ./ DD),
       L5G = -2 * sum(H2 .* 𝒳 ./ DD),
       L3G = 2 * sum(vec(sum(o.H .* (x' .* M); dims = 2)) .^ 2 ./ Mii),
       L4 = sum(H2 .* (σ̄ .* (x .- σ̄)')),
       cum = dot(o.c, κ),
       L1 = sum(b .^ 2 .* (σ̄ .- x)))
end

function bounds(o, q, κs, b)
    h = 1 .- o.Mii; h̄ = maximum(h); r = h ./ o.Mii; ω = vec(sum(o.Ω; dims = 2))
    HF = sum(o.H2); rΩr = dot(r, o.Ω * r)
    sumc = 2 * (2 + h̄) * rΩr + 2HF
    (; Q1 = 4q * (q - 1) * dot(h, ω),
       Q2 = 4 * (3q - 2)^2 * rΩr,
       L5G = 2q^2 * HF,
       L3G = 2 * (q - 1)^2 * dot(h, vec(sum(o.H2; dims = 2))),
       L4 = q * (q - 1) * dot(h, vec(sum(o.H2; dims = 1))),
       cum = κs * q^2 * sumc,
       L1 = (q - 1) * sum(b .^ 2 .* h),
       sumc, sumc_true = sum(abs, o.c))
end

function worstx(N, S, h, r, rng)
    n = length(h)
    starts = Any[ones(n), 1 .+ (r - 1) .* rand(rng, n), ifelse.(rand(rng, n) .< 0.5, 1.0, r),
                 1 .+ (r - 1) .* h ./ maximum(h), r .- (r - 1) .* h ./ maximum(h),
                 ifelse.(rand(rng, n) .< 0.1, r, 1.0), ifelse.(rand(rng, n) .< 0.9, r, 1.0)]
    for _ in 1:5; push!(starts, 1 .+ (r - 1) .* rand(rng, n) .^ 4); end
    best = (Inf, ones(n))
    for x0 in starts
        x, R = pgmin(N, S, x0, r); R < best[1] && (best = (R, x))
    end
    best
end

const TERMS = (:Q1, :Q2, :L5G, :L3G, :L4, :cum, :L1)

# Lemma 9's homoskedastic bound β̄ (no q).
function lemma9(o)
    h = 1 .- o.Mii; r = h ./ o.Mii
    (4 * sum(o.H2) + 4 * (1 + maximum(h)) * dot(r, o.Ω * r)) / (2 * sum(o.Ω))
end

function check(name, d, k, rng; qs = (2.0, 4.0, 10.0))
    o = objs(d, k); n = d.n; NN = buildN(o); N, S = NN.N, 2 .* o.Ω
    V0 = sum(S); h = 1 .- o.Mii; Kn = o.K / n
    fs = (d.x, (I - o.M) * g_surf.(d.u, d.v))
    β9 = lemma9(o)
    @printf("%s (K/n %.3f, max leverage %.3f) | Σ|cₖ| %.4g ≤ its bound %.4g | Lemma 9's β̄ (no q) %.4f\n",
            name, Kn, o.hmax, sum(abs, o.c), bounds(o, 1.0, 2.0, zeros(n)).sumc, β9)
    rows = Dict{Float64, Any}()
    for q in qs
        βw, xw = worstx(N, S, h, q, rng)
        xs = Any[xw]
        for _ in 1:40; push!(xs, ifelse.(rand(rng, n) .< rand(rng), q, 1.0)); end
        for _ in 1:10; push!(xs, 1 .+ (q - 1) .* rand(rng, n)); end
        minratio = Dict(t => Inf for t in TERMS); maxsumerr = 0.0
        for x in xs, κkind in (:rad, :heavy, :worst), fsel in 0:2
            κ = κkind == :rad ? -2 .* x .^ 2 : κkind == :heavy ? KBAR .* x .^ 2 :
                ifelse.(o.c .> 0, -2.0, KBAR) .* x .^ 2
            b = fsel == 0 ? zeros(n) : o.C' * fs[fsel]
            T = terms(o, x, κ, b); B = bounds(o, q, max(2.0, KBAR), b)
            for t in TERMS
                a = abs(getfield(T, t)); a > 1e-14 && (minratio[t] = min(minratio[t], getfield(B, t) / a))
            end
            if fsel == 0
                tot = T.Q1 + T.Q2 + T.L5G + T.L3G + T.L4 + T.cum
                maxsumerr = max(maxsumerr, abs(tot - lemma7(o, x, κ)) / abs(lemma7(o, x, κ)))
            end
        end
        B0 = bounds(o, q, max(2.0, KBAR), zeros(n))
        β̄ = sum(getfield(B0, t) for t in TERMS) / V0
        @printf("    q = %4.0f: worst β found %+.4f | bound β̄_het %.3f (%.0f× the worst) | min bound/|term| over %d patterns × 3 κ × 3 f: %s | terms sum to Lemma 7 (max rel err %.0e) | Lemma 9's β̄ %s\n",
                q, βw, β̄, β̄ / abs(βw), length(xs),
                join([@sprintf("%s %.2f", t, minratio[t]) for t in TERMS], ", "), maxsumerr,
                β9 < abs(βw) ? @sprintf("EXCEEDED (%.4f < %.4f)", β9, abs(βw)) : "holds")
        rows[q] = (; βw, β̄, minratio)
    end
    o, rows
end

println("Lemma 11: the HC2 V̂'s bias under heteroskedastic symmetric errors, term by term; κ̄ = $(KBAR)\n")
rng = MersenneTwister(2027); t0 = time(); allok = true
println("(1)–(3) The four designs of tc_hc2_hetero.jl (n = 900)")
for (name, kind, k) in (("jittered k = 13", :jittered, 13), ("jittered k = 16", :jittered, 16),
                        ("random   k = 10", :random, 10), ("rook     k = 13", :rook, 13))
    o, rows = check(name, design(kind), k, rng)
    global allok &= all(all(v >= 1 for v in values(r.minratio)) for r in values(rows))
    @printf("    (%.1f min)\n", (time() - t0) / 60)
end

println("\n(4) The rate, q = 4: jittered grid")
function rate(d, k, rng)
    o = objs(d, k); NN = buildN(o); S = 2 .* o.Ω; h = 1 .- o.Mii
    βw, _ = worstx(NN.N, S, h, 4.0, rng)
    B = bounds(o, 4.0, max(2.0, KBAR), zeros(d.n))
    β̄ = sum(getfield(B, t) for t in TERMS) / sum(S)
    o, βw, β̄
end
for k in (7, 10, 13, 16)
    o, βw, β̄ = rate(design(:jittered), k, rng)
    @printf("    n = 900, k = %2d: K/n %.3f, max leverage %.3f | worst β %+.4f, /(K/n) %+.3f | bound %.3f, /(K/n) %.2f   (%.1f min)\n",
            k, o.K / 900, o.hmax, βw, βw / (o.K / 900), β̄, β̄ / (o.K / 900), (time() - t0) / 60)
end
println("  at K/n ≈ 0.19, growing n")
for (side, k) in ((20, 9), (30, 13), (40, 17))
    d = design(:jittered; side)
    o, βw, β̄ = rate(d, k, rng)
    @printf("    n = %4d, k = %2d: K/n %.3f, max leverage %.3f | worst β %+.4f, /(K/n) %+.3f | bound %.3f, /(K/n) %.2f   (%.1f min)\n",
            d.n, k, o.K / d.n, o.hmax, βw, βw / (o.K / d.n), β̄, β̄ / (o.K / d.n), (time() - t0) / 60)
end
println("\nEvery term bound holds at every pattern checked: ", allok)
