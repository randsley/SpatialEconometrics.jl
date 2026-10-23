# Monte Carlo validation of the cross-sectional SDM (run: julia --project=. validation/mc_sdm.jl)
using SpatialEconometrics, LinearAlgebra, SparseArrays, Random, Statistics, Printf
const CS = SpatialEconometrics.CrossSection

L = 20; N = L^2
W = make_rook_weights(L)
ρ0 = 0.6; β0 = [2.0, 1.5, -1.0]; θ0 = [0.8, -0.5]; σ0 = 0.75
Wd = Matrix(W); A0 = inv(I - ρ0 * Wd)
# True average effects (LeSage-Pace): direct = tr(S)/N, total = 1'S1/N
true_dir = [tr(A0 * (I * β0[1+k] + Wd * θ0[k])) / N for k in 1:2]
true_tot = [sum(A0 * (I * β0[1+k] + Wd * θ0[k])) / N for k in 1:2]
true_ind = true_tot .- true_dir
@printf("True direct   : %s\nTrue indirect : %s\nTrue total    : %s\n\n", round.(true_dir,digits=4), round.(true_ind,digits=4), round.(true_tot,digits=4))

# ---- 1. log-determinant accuracy ----
println("== ln|I-ρW|: Chebyshev (exact traces) vs exact LU, N=$N ==")
println("order    ρ=0.3      ρ=0.6      ρ=0.9      ρ=0.99   (max abs error)")
for q in (6, 12, 24, 48)
    τ = CS.precompute_chebyshev_traces(W, CS.ChebyshevApprox(order=q))
    errs = [abs(CS.chebyshev_logdet(ρ, τ) - first(logabsdet(Matrix(I - ρ*Wd)))) for ρ in (0.3,0.6,0.9,0.99)]
    @printf("%4d  %9.2e  %9.2e  %9.2e  %9.2e\n", q, errs...)
end

# ---- 2. Monte Carlo ----
function simulate(rng)
    X = randn(rng, N, 2)
    rhs = β0[1] .+ X * β0[2:3] .+ (W * X) * θ0 .+ σ0 .* randn(rng, N)
    return X, A0 * rhs
end

function run_mc(method, R; seed=1)
    rng = MersenneTwister(seed)
    rho = Float64[]; rse = Float64[]; b = zeros(0, 5); dir = zeros(0,2); dse = zeros(0,2); ind = zeros(0,2); ise = zeros(0,2); tot = zeros(0,2); tse = zeros(0,2)
    for _ in 1:R
        X, y = simulate(rng)
        m = fit_spatial(SDM(), W, X, y; method=method, varnames=["x1","x2"])
        e = spatial_effects(m)
        push!(rho, m.rho); push!(rse, m.rho_se)
        b = vcat(b, m.coef[1:5]')
        dir = vcat(dir, e.direct'); dse = vcat(dse, e.direct_se')
        ind = vcat(ind, e.indirect'); ise = vcat(ise, e.indirect_se')
        tot = vcat(tot, e.total'); tse = vcat(tse, e.total_se')
    end
    return (; rho, rse, b, dir, dse, ind, ise, tot, tse)
end

cover(est, se, tr) = mean(abs.(est .- tr) .<= 1.96 .* se)
function report(name, r)
    println("\n== $name (R=$(length(r.rho))) ==")
    @printf("ρ        true %.3f  mean %.4f  bias %+.4f  sd %.4f  mean SE %.4f  cover95 %.2f\n", ρ0, mean(r.rho), mean(r.rho)-ρ0, std(r.rho), mean(r.rse), cover(r.rho, r.rse, ρ0))
    for (j,nm) in enumerate(("β0","β1","β2","γ1","γ2"))
        tv = (β0[1], β0[2], β0[3], θ0[1], θ0[2])[j]
        @printf("%-3s      true %+.3f  mean %+.4f  bias %+.4f  sd %.4f\n", nm, tv, mean(r.b[:,j]), mean(r.b[:,j])-tv, std(r.b[:,j]))
    end
    for k in 1:2
        @printf("x%d  direct   true %+.4f mean %+.4f sd %.4f meanSE %.4f cover %.2f\n", k, true_dir[k], mean(r.dir[:,k]), std(r.dir[:,k]), mean(r.dse[:,k]), cover(r.dir[:,k], r.dse[:,k], true_dir[k]))
        @printf("x%d  indirect true %+.4f mean %+.4f sd %.4f meanSE %.4f cover %.2f\n", k, true_ind[k], mean(r.ind[:,k]), std(r.ind[:,k]), mean(r.ise[:,k]), cover(r.ind[:,k], r.ise[:,k], true_ind[k]))
        @printf("x%d  total    true %+.4f mean %+.4f sd %.4f meanSE %.4f cover %.2f\n", k, true_tot[k], mean(r.tot[:,k]), std(r.tot[:,k]), mean(r.tse[:,k]), cover(r.tot[:,k], r.tse[:,k], true_tot[k]))
    end
end

R = 100
report("ExactLU", run_mc(ExactLU(), R))
report("Chebyshev order 6", run_mc(ChebyshevApprox(order=6), R))
