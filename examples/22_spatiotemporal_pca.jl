# ═══════════════════════════════════════════════════════════════════════
# Example 22: Spatio-temporal principal components
# ═══════════════════════════════════════════════════════════════════════
#
# Classical PCA on panel data has to choose what to throw away. Stack the
# years and it forgets that the columns are the same feature at different
# moments; average them and it forgets the dynamics; and either way it
# never knows where the units are.
#
# STPCA keeps all three. Each unit-feature series becomes a continuous
# function through a Fourier expansion, which is where time enters, and the
# components are then found by maximising the product of a functional
# principal component's variance and the spatial autocorrelation of the
# scores it produces, which is where space enters.
#
# Demonstrates:
#   - stpca: the decomposition, under Moran's I and Geary's C
#   - the eigenvalue identity that makes the spectrum readable
#   - component_groups: global components versus local ones
#   - feature_contributions / component_weight: what a component is made of
#   - how much the answer moves when W moves

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Printf, Statistics

# The reference's grid: annual observations at the midpoints of seventeen
# unit intervals, expanded on [0, 17].
ts = collect(0.5:1.0:16.5); nt = length(ts)
IV = (0.0, 17.0)
W = row_standardize(make_rook_weights(6)); N = size(W, 1)
p = 3

# Two known fields to plant: the smoothest pattern the lattice admits and
# the most contrasting one, taken from the eigenvector basis of example 21.
se = spatial_eigenvectors(W)
smooth = se.vectors[:, 1]
rough  = se.vectors[:, end]

rng = MersenneTwister(2026)
X = Array{Float64}(undef, N, p, nt)
for i in 1:N, j in 1:p
    X[i, j, :] = randn(rng) .* cos.(ts ./ 4) .+ 0.5 .* randn(rng, nt)
end
# Feature 1 carries a smooth field on a slow profile; feature 3 carries a
# high-contrast field on a faster one. Feature 2 carries neither.
for i in 1:N
    X[i, 1, :] .+= 5.0 * smooth[i] .* sin.(2pi .* ts ./ 17)
    X[i, 3, :] .+= 5.0 * rough[i] .* sin.(4pi .* ts ./ 17)
end

println("═"^70)
println("1. The decomposition")
println("═"^70)
println()
@printf("  %d units, %d features, %d moments -> %d basis coefficients each\n",
        N, p, nt, p * 5)
@printf("  planted: smooth field (I = %+.3f) in feature 1\n",
        moran_coefficient(smooth, W))
@printf("           contrast field (I = %+.3f) in feature 3\n",
        moran_coefficient(rough, W))
println()

m = stpca(X, W; measure = :moran, n_basis = 5, times = ts,
          interval = IV, varnames = ["smooth-carrier", "noise", "contrast-carrier"])
println(m)
println()

println("═"^70)
println("2. Each eigenvalue IS the objective it maximises")
println("═"^70)
println()
println("  comp    eigenvalue   =    Var(U)    x    I(Ay)   | recomputed I")
for k in (1, 2, length(m.values) - 1, length(m.values))
    ind = moran_coefficient(m.scores[:, k], W)      # independent of stpca
    @printf("  %4d %13.6f   = %9.6f  x %9.6f | %11.6f\n",
            k, m.values[k], m.variance[k], m.autocorr[k], ind)
end
println()
println("  The last column comes from moran_coefficient, which stpca never")
println("  calls. The product is not an approximation of the eigenvalue --")
println("  it is the eigenvalue, which is why the spectrum can be read as")
println("  'how much variance, and how spatially structured'.")
println()

println("═"^70)
println("3. Global components and local ones")
println("═"^70)
println()
g = component_groups(m)
@printf("  %d global (positive), %d local (negative), %d at rounding\n",
        length(g.positive), length(g.negative),
        length(m.values) - length(g.positive) - length(g.negative))
println()
println("  The matrix being decomposed is not a covariance matrix, so its")
println("  eigenvalues are not bound to be positive. The sign is the")
println("  result: positive means linked units carry similar scores,")
println("  negative means a unit is surrounded by unlike ones.")
println()
println("  group   comp   eigenvalue   Moran I   |cor| with planted field")
for k in g.positive[1:min(2, end)]
    @printf("  global  %4d %12.5f %9.4f   smooth %.3f  contrast %.3f\n",
            k, m.values[k], m.autocorr[k],
            abs(cor(m.scores[:, k], smooth)), abs(cor(m.scores[:, k], rough)))
end
for k in g.negative[1:min(2, end)]
    @printf("  local   %4d %12.5f %9.4f   smooth %.3f  contrast %.3f\n",
            k, m.values[k], m.autocorr[k],
            abs(cor(m.scores[:, k], smooth)), abs(cor(m.scores[:, k], rough)))
end
println()
println("  The smooth field is at the top of the spectrum and the contrast")
println("  field at the bottom, which is the split working. Over 800 draws")
println("  of this design neither ever landed at the wrong end; the rates")
println("  are in docs/AUDIT.md.")
println()

println("═"^70)
println("4. What a component is made of")
println("═"^70)
println()
println("  A component is a direction in coefficient space. Read back as a")
println("  function of time it is a weight on each feature at each moment,")
println("  and the share of the area under that weight says which series")
println("  the component is tracking.")
println()
println("  comp     ", join((@sprintf("%18s", v) for v in m.varnames)))
for k in vcat(g.positive[1:min(2, end)], g.negative[1:min(1, end)])
    fc = feature_contributions(m, k)
    @printf("  %4d     %s\n", k, join((@sprintf("%17.1f%%", 100fc[j]) for j in 1:p)))
end
println()
println("  The leading global component should be mostly feature 1 and the")
println("  trailing local one mostly feature 3 -- that is where the fields")
println("  were planted. Feature 2 is noise and should lead nothing.")
println()
U = component_weight(m, g.positive[1], ts)
println("  The weight function of the leading component, over time:")
println("     t     ", join((@sprintf("%18s", v) for v in m.varnames)))
for i in 1:4:nt
    @printf("  %5.1f     %s\n", ts[i],
            join((@sprintf("%18.4f", U[i, j]) for j in 1:p)))
end
println()

println("═"^70)
println("5. Geary's C answers a different question")
println("═"^70)
println()
mg = stpca(X, W; measure = :geary, n_basis = 5, times = ts, interval = IV)
@printf("  moran: %d positive, %d negative eigenvalues\n",
        length(g.positive), length(g.negative))
gg = component_groups(mg)
@printf("  geary: %d positive, %d negative eigenvalues\n",
        length(gg.positive), length(gg.negative))
println()
println("  Geary's C is a sum of squares, so the matrix is non-negative")
println("  definite by construction and there is no sign to read. The")
println("  ordering also changes:")
println()
println("           leading component's |cor| with   smooth   contrast")
for (lbl, mm) in (("moran", m), ("geary", mg))
    @printf("  %-6s %32.3f %10.3f\n", lbl,
            abs(cor(mm.scores[:, 1], smooth)), abs(cor(mm.scores[:, 1], rough)))
end
println()
println("  Maximising Var x C demotes smooth structure -- but it does not")
println("  simply promote the roughest, because the objective is a product")
println("  and a moderate-C component with large variance outranks a")
println("  high-C component with little. Over 400 draws the contrast field")
println("  led the geary spectrum only 7-34% of the time. Read that")
println("  ordering as 'variance weighted by dissimilarity'.")
println()

println("═"^70)
println("6. The components depend on W, and there is no rule for choosing it")
println("═"^70)
println()
coords = hcat(repeat(1:6, 6), repeat(1:6, inner = 6)) .* 1.0
cands = [W,
         row_standardize(make_queen_weights(6)),
         row_standardize(knn_weights(coords, 4)),
         row_standardize(knn_weights(coords, 8))]
names = ["rook", "queen", "knn4", "knn8"]
println("  W        lead eigenvalue   #global   #local   |cor| lead vs smooth")
for (nm, Wc) in zip(names, cands)
    mc = stpca(X, Wc; measure = :moran, n_basis = 5, times = ts, interval = IV)
    gc = component_groups(mc)
    @printf("  %-8s %15.5f %9d %8d %20.3f\n", nm, mc.values[1],
            length(gc.positive), length(gc.negative),
            abs(cor(mc.scores[:, 1], smooth)))
end
println()
println("  The planted field here is strong enough that every W finds it:")
println("  the leading correlation barely moves. What does move is the")
println("  split -- how many components come out global and how many")
println("  local -- and so would any clustering built on the whole")
println("  spectrum rather than on the first component alone. On structure")
println("  weaker than this the components move too. The reference")
println("  compares nine weight matrices for exactly this reason and")
println("  offers no rule for choosing one, so report the W alongside the")
println("  components and vary it.")
