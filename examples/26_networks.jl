# ---------------------------------------------------------------------------
# 26. Spatial econometrics on networks
#
# Every other W in this package comes from distance. Here it comes from a
# graph, and one thing changes that matters more than the construction
# does: whether the model is identified at all.
#
# Bramoulle, Djebbari and Fortin (2009) prove that the endogenous peer
# effect and the contextual effect are separately identified if and only if
# I, G and G^2 are linearly independent -- and that this fails on the
# structure applied work reaches for first. This script shows the failure,
# shows what it costs, and shows what still works.
#
# Run: julia --project=. examples/26_networks.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror, residuals

const RNG = MersenneTwister(20_261_101)

# Erdos-Renyi graph with no isolates.
function er_graph(rng, n, d)
    for _ in 1:500
        e = [(i, j) for i in 1:n for j in (i + 1):n if rand(rng) < d]
        isempty(e) && continue
        G = network_weights(e, n)
        all(!iszero, sum(G, dims = 2)) && return G
    end
    error("no connected draw")
end

# The linear-in-means model: y = a + b*Gy + x*g + Gx*d + e.
function simulate(rng, G, b, g, d; a = 1.0)
    n = size(G, 1); Gd = Matrix(G)
    X = randn(rng, n, length(g))
    y = (Matrix(1.0I, n, n) - b * Gd) \
        (a .+ X * g .+ Gd * X * d .+ randn(rng, n))
    return y, X
end

const B, GAM, DEL = 0.4, [1.0, -0.5], [0.8, 0.3]

println("="^74)
println("1. The reflection problem, as a rank")
println("="^74)
println("Students interact in classrooms. Everyone in a room is affected by")
println("everyone else in it. Three rooms of five:")
println()
show(stdout, peer_identification(group_weights([5, 5, 5])))

println()
println("Nothing about that is a small-sample problem. Now make the rooms")
println("different sizes -- 4, 5 and 6 -- and change nothing else:")
println()
show(stdout, peer_identification(group_weights([4, 5, 6])))

println()
println("That is the whole of Bramoulle, Djebbari and Fortin's answer to")
println("Manski: variation in group size breaks the collinearity between a")
println("unit's peers and its peers' peers. Same model, same data volume,")
println("different verdict.")

println()
println("="^74)
println("2. What the check costs you if you skip it")
println("="^74)

G_bad = group_weights([5, 5, 5])
y_bad, X_bad = simulate(RNG, G_bad, B, GAM, DEL)

println("Fitting on the equal-size network is refused:")
println()
try
    fit_peer_effects(y_bad, X_bad, G_bad)
catch e
    for line in split(sprint(showerror, e), ". ")
        println("  ", strip(line), ".")
    end
end

println()
println("Passing `check = false` does not get you an estimate either, and")
println("that is not the override being timid. Bramoulle, Djebbari and")
println("Fortin prove in their Appendix B that when I, G and G^2 are")
println("dependent, E(Gy|x) is perfectly collinear with (iota, x, Gx) -- so")
println("the instruments are collinear with the regressors and the design is")
println("genuinely singular. There is no estimate to report:")
println()
try
    fit_peer_effects(y_bad, X_bad, G_bad; check = false)
catch e
    println("  ", first(split(sprint(showerror, e), ". ")), ".")
end
let Gd = Matrix(G_bad), nb = size(G_bad, 1)
    sys = 1.0 .+ X_bad * GAM .+ Gd * X_bad * DEL
    EGy = Gd * ((Matrix(1.0I, nb, nb) - B * Gd) \ sys)
    Z = hcat(ones(nb), X_bad, Gd * X_bad)
    @printf("\n  residual of E(Gy|x) on (iota, x, Gx): %.2e -- collinear.\n",
            maximum(abs, EGy .- Z * (Z \ EGy)))
end

println()
println("So where is the danger? It is that the obvious thing to reach for")
println("instead runs without complaint. The peer-effects model IS a spatial")
println("Durbin model, term for term, and `fit_spatial` will fit one on this")
println("exact network and tell you nothing is wrong:")
println()
sdm = fit_spatial(SDM(), G_bad, X_bad, y_bad; method = ExactLU(),
                  max_exact_n = 10_000)
@printf("  true  rho (endogenous) = %+.3f,  contextual = [%+.3f, %+.3f]\n",
        B, DEL[1], DEL[2])
@printf("  SDM   rho              = %+.3f,  contextual = [%+.3f, %+.3f]\n",
        sdm.rho, sdm.coef[4], sdm.coef[5])
@printf("  standard errors        : %s\n",
        string(round.(sqrt.(diag(sdm.vcov)), digits = 3)))
println()
println("  The endogenous effect comes back with the wrong sign, pinned to")
println("  the edge of the search interval, and the contextual effect is out")
println("  by an order of magnitude. No warning, no error, standard errors")
println("  that look like any other table. That is what the rank check is")
println("  for, and it costs one call.")

println()
println("="^74)
println("3. A network where it works")
println("="^74)

n = 300
G = er_graph(RNG, n, 0.015)
y, X = simulate(RNG, G, B, GAM, DEL)
f = fit_peer_effects(y, X, G)
show(stdout, f)
println()
@printf("  true: beta = %+.3f, gamma = [%+.3f, %+.3f], delta = [%+.3f, %+.3f]\n",
        B, GAM[1], GAM[2], DEL[1], DEL[2])
println()
println("The instruments are the characteristics of peers' peers. They")
println("belong in the reduced form and not in the structural equation,")
println("which is exactly what the rank condition in step 1 was testing.")

println()
println("="^74)
println("4. Identified is not the same as estimable")
println("="^74)
println("The rank condition is a statement about the population. Whether")
println("this sample can exploit it is a separate question, and on a dense")
println("network the answer is no: a unit's peers' peers are nearly")
println("everybody, so G^2 x flattens towards a constant.")
println()
@printf("%-12s %-8s %-10s %-12s %-10s\n",
        "density", "degree", "identified", "first-stage F", "se(beta)")
for d in (0.015, 0.03, 0.06, 0.12, 0.20)
    Gd = er_graph(RNG, n, d)
    yd, Xd = simulate(RNG, Gd, B, GAM, DEL)
    fd = fit_peer_effects(yd, Xd, Gd; check = false)
    deg = mean(sum(Matrix(Gd) .!= 0, dims = 2))
    @printf("%-12.3f %-8.1f %-10s %-12.1f %-10.3f\n",
            d, deg, fd.ident.identified, fd.first_stage_F, stderror(fd)[2])
end
println()
println("Identified all the way down, and useless by the bottom -- the")
println("standard error grows by an order of magnitude while the verdict")
println("never changes. These are single draws, so the F wobbles; the")
println("measured version over 400 draws per row is in docs/AUDIT.md, where")
println("the spread of beta-hat rises twenty-one-fold across this range")
println("while the rejection rate actually *improves*, from 0.105 to 0.062.")
println("So the failure does not announce itself as a significant test. The")
println("first-stage F is what announces it.")

println()
println("="^74)
println("5. Network fixed effects ask for more")
println("="^74)
println("If the unobservable is correlated with x at the network level, the")
println("within-local transformation (I - G) sweeps it out -- and tightens")
println("the rank condition from 3 to 4. Two classroom sizes are no longer")
println("enough; three are.")
println()
@printf("%-22s %-14s %-14s\n", "structure", "no corr. eff.", "fixed effects")
for sizes in ([5, 5, 5], [4, 6], [4, 6, 9], [3, 5, 7, 11])
    Gg = group_weights(sizes)
    a = peer_identification(Gg).identified
    b = peer_identification(Gg; fixed_effects = true).identified
    @printf("%-22s %-14s %-14s\n", string(sizes), a, b)
end
println()
println("The middle row is the one to look at. The same network, the same")
println("data, identified or not depending only on whether you are willing")
println("to assume away a correlated unobservable.")

fe = fit_peer_effects(y, X, G; fixed_effects = true, vcov = :transform)
@printf("\n  under fixed effects on the ER network: beta = %+.3f (se %.3f)\n",
        fe.beta, stderror(fe)[1])
println("  vcov = :transform accounts for the correlation (I - G) itself")
println("  induces, since Var((I-G)e) = s2 (I-G)(I-G)' is not scalar.")

println()
println("="^74)
println("6. Several channels at once")
println("="^74)
println("Units can be linked more than one way -- trade and migration, or")
println("friendship and classroom. Each channel gets its own rho, which a")
println("single pooled W cannot report.")
println()

W_trade = er_graph(RNG, 200, 0.03)
W_migr  = er_graph(RNG, 200, 0.03)
Xm = randn(RNG, 200, 2)
Am = Matrix(1.0I, 200, 200) - 0.45 * Matrix(W_trade) - 0.10 * Matrix(W_migr)
ym = Am \ (1.0 .+ Xm * [1.0, -0.6] .+ randn(RNG, 200))

mx = fit_multiplex_sar(ym, Xm, [W_trade, W_migr];
                       channel_names = ["trade", "migration"])
show(stdout, mx)
println()
println("  true rho: trade 0.45, migration 0.10")
println()
println("Pooling the two channels into one average network reports a single")
println("dependence parameter and loses the comparison entirely:")
W_pool = network_weights((Matrix(W_trade) .+ Matrix(W_migr)) ./ 2)
mp = fit_multiplex_sar(ym, Xm, [W_pool]; channel_names = ["pooled"])
@printf("  pooled rho = %+.3f\n", mp.rho[1])
println()
println("Note it is not an average of the two, and is not even between them:")
println("row-standardising the summed adjacency halves the weight each link")
println("carries, so the single parameter has to rise to reproduce the same")
println("total dependence. The pooled number is not on the same scale as")
println("either channel and cannot be compared with them -- which is a second")
println("reason to fit the channels separately rather than a restatement of")
println("the first.")

println()
println("="^74)
println("Summary")
println("="^74)
@printf("  equal groups identified      : %s\n",
        peer_identification(group_weights([5, 5, 5])).identified)
@printf("  unequal groups identified    : %s\n",
        peer_identification(group_weights([4, 5, 6])).identified)
@printf("  beta on the sparse network   : %+.3f (true %+.3f)\n", f.beta, B)
@printf("  first-stage F there          : %.1f\n", f.first_stage_F)
@printf("  multiplex rho                : trade %+.3f, migration %+.3f\n",
        mx.rho[1], mx.rho[2])
