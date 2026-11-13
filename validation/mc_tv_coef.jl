# Monte Carlo for fit_tv_coef_sar (Liang, Gao & Gong 2022): does the pointwise
# band for β(τ) cover, and where does it fail?
#
# The band is ±1.96·beta_se, and beta_se excludes the smoothing bias, which is
# of order h²β''(τ). This study measures what that omission costs, at the ends
# of the sample and inside it, for a gently curved β and a strongly curved one.
#
# Run: julia --project=. validation/mc_tv_coef.jl
# Output: validation/mc_tv_coef_output.txt (markdown rows)
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf

function ring(N)
    W = spzeros(N, N)
    for i in 1:N, k in (-2, -1, 1, 2)
        W[i, mod1(i + k, N)] = 0.25
    end
    return W
end

const CURVES = Dict(
    :quad => τ -> 1 + 2τ + 2τ^2,       # β'' = 4
    :sine => τ -> 1 + sin(2π * τ),     # |β''| up to 4π² ≈ 39
)
const ρ0 = 0.4

function draw(rng, N, Tp, W, βx)
    τ = (1:Tp) ./ Tp
    x = randn(rng, N, Tp); α = randn(rng, N); Y = zeros(N, Tp)
    for t in 1:Tp
        Y[:, t] = (I - ρ0 * W) \ ((1 + τ[t]^3) .+ βx(τ[t]) .* x[:, t] .+ α .+ randn(rng, N))
    end
    return vec(Y), hcat(ones(N * Tp), vec(x))
end

# shrink = 1: the CV bandwidth as chosen. shrink = 0.5: refit at half of it
# (undersmoothing), so the bias falls faster than the standard error.
function run(curve, N, Tp; shrink = 1.0, reps = 500, seed = 20261113)
    rng = MersenneTwister(seed)
    W = ring(N); βx = CURVES[curve]
    τ = (1:Tp) ./ Tp; truth = βx.(τ)
    q = 1.959963984540054
    inner = findall(t -> 0.2 <= t <= 0.8, τ)
    s10 = argmin(abs.(τ .- 0.1))
    B = Vector{Float64}[]; SE = Vector{Float64}[]; cov_ρ = 0; hs = Float64[]; fails = 0
    for r in 1:reps
        y, X = draw(rng, N, Tp, W, βx)
        m = try
            m1 = fit_tv_coef_sar(y, X, W, N, Tp; method = ExactLU())
            shrink == 1 ? m1 :
                fit_tv_coef_sar(y, X, W, N, Tp; method = ExactLU(), bandwidth = shrink * m1.bandwidth)
        catch
            fails += 1; continue
        end
        push!(B, m.beta[:, 2]); push!(SE, m.beta_se[:, 2])
        cov_ρ += abs(m.rho - ρ0) <= q * m.rho_se
        push!(hs, m.bandwidth)
    end
    k = reps - fails
    Bm = reduce(hcat, B); Sm = reduce(hcat, SE)          # Tp × k
    c = vec(mean(abs.(Bm .- truth) .<= q .* Sm; dims = 2))
    μ = vec(mean(Bm; dims = 2)); b = μ .- truth
    # Is beta_se right about the variance? Its mean against the Monte Carlo
    # standard deviation, and coverage of a band centred on the Monte Carlo
    # mean: that removes the smoothing bias and leaves only the SE to judge.
    ratio = vec(mean(Sm; dims = 2)) ./ vec(std(Bm; dims = 2))
    c0 = vec(mean(abs.(Bm .- μ) .<= q .* Sm; dims = 2))
    @printf("| %s | %.1f | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %+.4f | %.4f | %.2f–%.2f | %.3f | %.3f | %d |\n",
            curve, shrink, N, Tp, mean(hs), c[1], c[s10], mean(c[inner]), minimum(c[inner]), c[end],
            b[1], maximum(abs.(b[inner])), minimum(ratio), maximum(ratio), mean(c0[inner]),
            cov_ρ / k, fails)
    flush(stdout)
end

println("| β_x(τ) | h / h_cv | N | T | mean h | cover τ=1/T | cover τ≈0.1 | cover interior (mean) | cover interior (min) | cover τ=1 | bias τ=1/T | max abs bias interior | se/sd range | recentred cover interior | cover ρ | fails |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for shrink in (1.0, 0.5), curve in (:quad, :sine), N in (50, 100), Tp in (25, 50)
    run(curve, N, Tp; shrink = shrink)
end

# ── Where the SE shortfall comes from ──────────────────────────────────────
# beta_se is σ̂²ΨΨ' conditional on h, ρ̂ and α̂. Compare its mean with the
# Monte Carlo SD for the fitted estimator and for an oracle that knows ρ and
# α and uses σ² = 1, at a fixed h and at the CV h. The oracle at fixed h
# isolates the formula; fixed against CV h isolates bandwidth selection.
const SS = SpatialEconometrics.SemiparametricSpatial
function decompose(N, Tp, hmode; reps = 400, seed = 7)
    rng = MersenneTwister(seed); W = ring(N); τ = (1:Tp) ./ Tp; βx = CURVES[:quad]
    est = Vector{Float64}[]; ora = Vector{Float64}[]; se = Vector{Float64}[]; seo = Vector{Float64}[]
    for r in 1:reps
        x = randn(rng, N, Tp); α = randn(rng, N); Y = zeros(N, Tp)
        for t in 1:Tp
            Y[:, t] = (I - ρ0 * W) \ ((1 + τ[t]^3) .+ βx(τ[t]) .* x[:, t] .+ α .+ randn(rng, N))
        end
        y = vec(Y); X = hcat(ones(N * Tp), vec(x))
        m = fit_tv_coef_sar(y, X, W, N, Tp; method = ExactLU(), bandwidth = hmode)
        push!(est, m.beta[:, 2]); push!(se, m.beta_se[:, 2])
        Xs = [X[(t-1)*N+1:t*N, :] for t in 1:Tp]
        Sm = SS._tv_smoother(Xs, m.bandwidth)
        v = y .- ρ0 .* vec(W * Y) .- repeat(α, Tp)
        push!(ora, SS._tv_beta(Sm, reshape(v, :, 1))[2, 1, :])
        push!(seo, m.beta_se[:, 2] ./ sqrt(m.sigma2))
    end
    B = reduce(hcat, est); O = reduce(hcat, ora); Sv = reduce(hcat, se); So = reduce(hcat, seo)
    for (lbl, ix) in (("ends", [1, Tp]), ("τ≈0.1", [argmin(abs.(τ .- 0.1))]),
                      ("interior", findall(t -> 0.2 <= t <= 0.8, τ)))
        rf = mean(vec(mean(Sv[ix, :]; dims = 2)) ./ vec(std(B[ix, :]; dims = 2)))
        ro = mean(vec(mean(So[ix, :]; dims = 2)) ./ vec(std(O[ix, :]; dims = 2)))
        @printf("| %s | %d | %d | %s | %.3f | %.3f | %.3f |\n", string(hmode), N, Tp, lbl, rf, ro,
                mean(abs2, Sv ./ So))
    end
    flush(stdout)
end

println()
println("| h | N | T | where | se/sd fitted | se/sd oracle (ρ, α known, σ² = 1) | mean σ̂² |")
println("|---|---|---|---|---|---|---|")
decompose(50, 25, 0.28); decompose(50, 25, :cv); decompose(100, 50, 0.207); decompose(100, 50, :cv)
