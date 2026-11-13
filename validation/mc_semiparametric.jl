# Monte Carlo for #13 pass 1 (fit_partially_linear_sar and closed_form_rho):
# coverage of the QML and closed-form standard errors, the cost of omitting
# Su & Jin's Ω under skewed errors, and Geniaux's geoadditive trap with the
# overlap π̄ beside it. Recorded in docs/AUDIT.md.
#
# Each cell reseeds its own generator, so any cell can be rerun alone and
# prints the same row.
#
# Run: julia --project=. validation/mc_semiparametric.jl
# Output: validation/mc_semiparametric_output.txt (markdown rows)
using SpatialEconometrics, StatsAPI, SparseArrays, LinearAlgebra, Random, Statistics, Printf

function rowstd(W)
    W = sparse(W); r = vec(sum(W; dims=2)); r[r .== 0] .= 1
    return spdiagm(0 => 1 ./ r) * W
end

# design: :iid  z ~ N(0,1) unrelated to location (Su & Jin-like)
#         :geo  z = coordinates, m a smooth surface (the trap)
function draw(rng, n, ρ, design, errs)
    coords = rand(rng, n, 2)
    W = rowstd(knn_weights(coords, 6))
    if design === :iid
        z = randn(rng, n); mz = 2 .* exp.(z) ./ (1 .+ exp.(z))
        x = 0.2 .* z.^2 .- z .+ randn(rng, n)
    else
        z = coords; mz = 2 .* sin.(π .* coords[:,1]) .* cos.(π .* coords[:,2])
        x = coords[:,1] .+ randn(rng, n)       # a covariate that carries geography
    end
    ε = errs === :normal ? randn(rng, n) : (randn(rng, n).^2 .+ randn(rng, n).^2 .+ randn(rng, n).^2 .- 3) ./ sqrt(6)
    y = (I - ρ * W) \ (x .+ mz .+ ε)
    return y, reshape(x, :, 1), z, W
end

function run(n, ρ, design, errs; reps=500, seed=20261113)
    rng = MersenneTwister(seed)
    q = 1.959963984540054
    ρs = Float64[]; cov_ml = 0; cov_b = 0; cfe = Float64[]; cc = 0; ch = 0; cs = 0; pb = Float64[]; nw = Float64[]
    fails = 0
    for r in 1:reps
        y, X, z, W = draw(rng, n, ρ, design, errs)
        m = try fit_partially_linear_sar(y, X, z, W; method=ExactLU()) catch; fails += 1; continue end
        se = StatsAPI.stderror(m)
        push!(ρs, m.rho); push!(pb, m.overlap.pibar); push!(nw, m.n_widened)
        cov_ml += abs(m.rho - ρ) <= q * se[2]
        cov_b  += abs(m.beta[1] - 1.0) <= q * se[1]
        c = try closed_form_rho(m) catch; nothing end
        if c !== nothing
            push!(cfe, c.rho)
            cc += abs(c.rho - ρ) <= q * c.se_classical
            ch += abs(c.rho - ρ) <= q * c.se_hc0
            cs += abs(c.rho - ρ) <= q * c.se_shac
        end
    end
    k = length(ρs); kc = length(cfe)
    @printf("| %s | %s | %d | %.2f | %+.4f | %.4f | %.3f | %.3f | %+.4f | %.4f | %.3f | %.3f | %.3f | %.3f | %.1f | %d |\n",
        design, errs, n, ρ, mean(ρs) - ρ, sqrt(mean(abs2, ρs .- ρ)), cov_ml / k, cov_b / k,
        mean(cfe) - ρ, sqrt(mean(abs2, cfe .- ρ)), cc / kc, ch / kc, cs / kc, mean(pb), mean(nw), fails)
    flush(stdout)
end

println("| design | errors | n | ρ | QML bias | QML RMSE | cover ρ | cover β | CFE bias | CFE RMSE | CFE cl | CFE HC0 | CFE SHAC | π̄ | widened | fails |")
println("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for design in (:iid, :geo), n in (400, 1000), ρ in (0.25, 0.75)
    run(n, ρ, design, :normal)
end
for n in (400, 1000), ρ in (0.25, 0.75)
    run(n, ρ, :iid, :chi2)
end
