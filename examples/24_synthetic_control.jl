# # Spatial synthetic control
#
# A policy lands on one region. There is no control group to speak of —
# one treated unit, a long pre-period, and a few dozen other regions that
# were not treated. This is the comparative case study, and it is where
# difference-in-differences is weakest: with a single treated unit there
# is nothing to average over, and the parallel-trends assumption is being
# asked to carry the whole estimate.
#
# Synthetic control builds a comparison instead of assuming one: a
# weighted average of untreated regions that tracks the treated region
# before the policy, used as its counterfactual after.
#
# The spatial problem is that the policy does not stop at the border. If
# it spills onto neighbouring regions, those regions are no longer
# untreated — and they are exactly the ones a synthetic control will
# reach for, because they look most like the treated region. The spillover
# then sits inside the comparison and the estimate is pulled toward zero.
#
# This script shows that happening, and the screening that prevents it.

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf

# ## A lattice, a policy, and a spillover
#
# Nine-by-nine regions on a grid. Regional characteristics are spatially
# smooth — neighbouring regions resemble one another, as they do in any
# real map — which is what makes a treated region's neighbours attractive
# donors in the first place.

side = 9
N = side^2
T_per = 24
T0 = 18                                   # policy starts at period 19

W = make_rook_weights(side)
coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
Wt = row_standardize(W)

rng = MersenneTwister(2026)
F = randn(rng, 3, T_per)                  # three common time factors
L = let Lraw = rand(rng, N, 3), Wd = Matrix(Wt)   # regional loadings
    for _ in 1:3                          # smoothed over the lattice
        Lraw = 0.5 .* Lraw .+ 0.5 .* (Wd * Lraw)
    end
    Lraw
end
Y = L * F .+ 0.04 .* randn(rng, N, T_per)

treated = 41                              # the middle region, (5,5)
att_true = 3.0
spillover_true = 2.0

D = zeros(N); D[treated] = 1.0
E = Wt * D                                # exposure: 1/4 for each neighbour

for t in (T0 + 1):T_per
    Y[treated, t] += att_true             # the policy effect
    Y[:, t] .+= spillover_true .* E       # and what it does next door
end

println("Policy on region $treated from period $(T0 + 1).")
println("True effect on the treated region:      ", att_true)
println("True spillover onto each neighbour:     ", spillover_true * 0.25)
println("Neighbours of region $treated: ", findall(!iszero, Matrix(W)[treated, :]))
println()

# ## The conventional fit, and why it is wrong here
#
# `donor_pool = :all` is what a non-spatial implementation does: every
# untreated region is a candidate donor, neighbours included.

scm_all = fit_synthetic_control(Y, treated, T0;
                                donor_pool = :all, penalty_metric = :none)

# ## The screened fit
#
# `donor_pool = :exclude_neighbours` — the default — drops every untreated
# region adjacent to a treated one under `W`, and records which.

scm = fit_synthetic_control(Y, treated, T0; W = W, coords = coords)

println(scm)
println()

contaminated = sum(scm_all.weights[1, q]
                   for (q, j) in enumerate(scm_all.donors)
                   if j in scm.excluded_neighbours; init = 0.0)

@printf("%-34s %10s %10s %12s\n", "", "ATT", "error", "pre-RMSPE")
@printf("%-34s %10.4f %10.4f %12.5f\n", "all donors (conventional)",
        scm_all.att, scm_all.att - att_true, scm_all.rmspe_pre[1])
@printf("%-34s %10.4f %10.4f %12.5f\n", "neighbours excluded (default)",
        scm.att, scm.att - att_true, scm.rmspe_pre[1])
@printf("%-34s %10.4f\n", "truth", att_true)
println()
@printf("The conventional fit put %.1f%% of its weight on regions that the\n",
        100 * contaminated)
@printf("policy had reached. That is the whole of its error.\n")
println()

# Note which fit has the *better* pre-treatment fit. A synthetic control
# assembled from contaminated neighbours tracks the treated region
# beautifully before the policy — the contamination is post-treatment
# only, so it is invisible in the pre-period. Pre-period fit is a
# necessary check, not a sufficient one, and it cannot detect this
# problem at all.

# ## Was the effect real? Permutation inference
#
# There is no standard error here: the weights are a constrained optimum,
# not a coefficient with a sampling distribution. Inference reassigns the
# policy to regions that never had it and asks how unusual the real
# assignment looks.

pt = placebo_test(scm, Y; W = W, coords = coords)
println(pt)
println()
@printf("Note the floor: with %d clean donors the smallest p-value the design\n",
        pt.n_donors)
@printf("can produce is %.4f, whatever the effect size.\n", pt.min_attainable_p)
println()

# ## What the policy did to the neighbours
#
# The regions screened out of the donor pool were screened out because
# the policy reached them. That is a quantity worth having, not just a
# nuisance: synthesise each of them from the same clean pool and read its
# gap.

sg = spillover_gaps(scm, Y; W = W, coords = coords)
println(sg)
println()
@printf("%-10s %12s %14s\n", "region", "gap (post)", "pre-RMSPE")
for (q, u) in enumerate(sg.units)
    @printf("%-10d %12.4f %14.5f\n", u,
            mean(sg.gaps[q, (T0 + 1):T_per]), sg.rmspe_pre[q])
end
@printf("%-10s %12.4f\n", "truth", spillover_true * 0.25)
println()
println("Read `pre-RMSPE` before any of those gaps. A neighbour the clean")
println("pool cannot track before the policy tells you nothing about what")
println("the policy did to it, and the unweighted average over all four is")
println("only as good as its worst member.")
println()

# ## The penalty
#
# Unpenalised weights are often not unique: with more donors than
# pre-periods, several weight vectors fit the pre-period identically and
# imply different counterfactuals. A penalty on distance between the
# treated region and each donor breaks the tie, following Abadie &
# L'Hour (2021), and here makes the comparison geographically local.

function penalty_table(Yp, label)
    println(label)
    @printf("  %-8s %8s %12s %14s %12s\n",
            "lambda", "donors", "max weight", "mean distance", "pre-RMSPE")
    for λ in (0.0, 0.01, 0.1, 1.0, 10.0)
        s = fit_synthetic_control(Yp, treated, T0; W = W, coords = coords,
                                  penalty = λ, penalty_metric = :spatial)
        dn = [sqrt(sum(abs2, coords[treated, :] .- coords[j, :]))
              for j in s.donors]
        @printf("  %-8.2f %8d %12.4f %14.4f %12.5f\n", λ,
                count(>(1e-6), s.weights[1, :]), maximum(s.weights[1, :]),
                dot(dn, s.weights[1, :]), s.rmspe_pre[1])
    end
    println()
end

penalty_table(Y, "Spatially smooth characteristics (the panel above):")

# Nothing moves. The unpenalised fit already chose the two closest
# available donors, so proximity and pre-period fit agree and the penalty
# has nothing to trade. That is the common case when regional
# characteristics are spatially smooth — and it is worth knowing that the
# penalty is then costless rather than useless.
#
# Where they disagree, it bites. The same lattice with *unsmoothed*
# loadings: neighbouring regions no longer resemble one another, so the
# best-fitting donors are scattered across the map.

Y_iid = let Liid = rand(MersenneTwister(2026), N, 3)
    rng2 = MersenneTwister(99)
    Liid * F .+ 0.04 .* randn(rng2, N, T_per)
end
for t in (T0 + 1):T_per
    Y_iid[treated, t] += att_true
end

penalty_table(Y_iid, "Spatially unstructured characteristics:")

println("There the penalty pulls the comparison in: mean donor distance falls")
println("monotonically and the pre-period fit pays for it. Note that the")
println("number of donors does not fall monotonically — the penalty buys")
println("proximity, not sparsity. That trade is a modelling choice, so it is")
println("reported in the result rather than tuned away.")
println()
println("The reason to pay it is identification, not tidiness: with more")
println("donors than pre-periods the unpenalised weights are not unique,")
println("and several equally good weight vectors imply different")
println("counterfactuals. The penalty picks among them on a stated ground.")
