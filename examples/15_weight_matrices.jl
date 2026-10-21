# ═══════════════════════════════════════════════════════════════════════
# Example 15: Weight Matrix Construction
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Weights from coordinates: k-nearest neighbours, distance decay, bands
#   - Weights from regular grids, and block-diagonal combination
#   - Row, spectral and MinMax standardization
#   - Connectivity diagnostics, including island detection
#   - GeoDa GAL / GWT round-trips and the unit-ordering contract
#   - How the choice of W moves the estimates

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics
import StatsAPI: coef

CS = SpatialEconometrics.CrossSection

rng = MersenneTwister(2024)

# ─── From coordinates ────────────────────────────────────────────────

N = 150
coords = hcat(rand(rng, N) .* 100, rand(rng, N) .* 100)   # N×2, (x, y)

println("═══ k-Nearest Neighbours ═══")
for k in (4, 8)
    Wk = knn_weights(coords, k)
    # Every row has exactly k neighbours before standardization, so after
    # row-standardizing each non-zero is 1/k.
    counts = [count(!=(0.0), Wk[i, :]) for i in 1:N]
    println("  k = $k:  neighbours/row = $(unique(counts)),  ",
            "row sums = $(round(minimum(sum(Wk, dims=2)); digits=12))")
end

# Binary rather than inverse-distance, and forced symmetry.
W_bin = knn_weights(coords, 6; binary = true, symmetrize = true)
println("  binary + symmetrized: issymmetric = ", issymmetric(Matrix(W_bin)))
println()

println("═══ Distance Decay (cutoff 25) ═══")
for decay in (:inverse, :negative_exponential, :bisquare, :power)
    Wd = distance_weights(coords, 25.0; decay = decay, alpha = 1.5)
    println("  $(rpad(decay, 22)) nnz = $(lpad(nnz(Wd), 5))",
            "   mean |w| = ", round(mean(nonzeros(Wd)); digits = 4))
end
println()

println("═══ Distance Band ═══")
# A band excludes near neighbours — useful when the very local effect is
# already captured by a covariate and only the ring beyond it is of interest.
W_band = band_weights(coords, 10.0, 25.0)
println("  band [10, 25]: nnz = ", nnz(W_band))
println()

# ─── From regular grids ──────────────────────────────────────────────

println("═══ Grid Weights ═══")
W_rook  = make_rook_weights(10)    # 4-connected
W_queen = make_queen_weights(10)   # 8-connected
println("  rook  10×10: nnz = ", nnz(W_rook))
println("  queen 10×10: nnz = ", nnz(W_queen), "  (more links, as expected)")
println()

# Disjoint regions that do not interact — e.g. separate countries.
W_block = block_diagonal_weights([make_rook_weights(5) for _ in 1:3])
println("  block-diagonal of 3 × (5×5 rook): size = ", size(W_block),
        ", row sums still 1 = ",
        all(isapprox.(vec(sum(W_block, dims = 2)), 1.0; atol = 1e-12)))
println()

# ─── Standardization ─────────────────────────────────────────────────

println("═══ Standardization ═══")
W_raw = sparse((make_queen_weights(8) .> 0) .* 1.0)   # unstandardized 0/1

W_row  = row_standardize(W_raw)
W_spec = spectral_standardize(W_raw)
W_mm   = minmax_standardize(W_raw)

println("  row:      max row sum  = ", round(maximum(sum(W_row, dims = 2)); digits = 8))
println("  spectral: max |λ|      = ",
        round(maximum(abs.(eigvals(Matrix(W_spec)))); digits = 8))
println("  minmax:   max |λ|      = ",
        round(maximum(abs.(eigvals(Matrix(W_mm)))); digits = 8))
println()
println("  Row standardization makes ρ interpretable as an average-neighbour")
println("  effect but breaks symmetry. Spectral standardization keeps the")
println("  parameter space at (-1, 1) without destroying a symmetric W.")
println()

# The `!` forms mutate in place; the plain forms leave the input alone.
W_copy = copy(W_raw)
row_standardize!(W_copy)
println("  row_standardize! matches row_standardize: ",
        maximum(abs.(W_copy .- W_row)) < 1e-12)
println("  ... and the original was untouched:       ",
        maximum(abs.(W_raw .- sparse((make_queen_weights(8) .> 0) .* 1.0))) == 0.0)
println()

# ─── Connectivity diagnostics ────────────────────────────────────────

println("═══ Connectivity ═══")
println(spatial_connectivity(W_rook))
println()

# An island — a unit with no neighbours — is worth catching before fitting:
# it contributes nothing to the spatial lag and can make W singular.
W_island = knn_weights(coords, 3)
W_island[20, :] .= 0.0
dropzeros!(W_island)
println("  after isolating unit 20: n_islands = ",
        spatial_connectivity(W_island).n_islands)
println()

# ─── GeoDa file formats ──────────────────────────────────────────────

println("═══ GAL / GWT Round-Trips ═══")
dir = mktempdir()
ids = collect(1:size(W_rook, 1))

write_gal(joinpath(dir, "grid.gal"), W_rook, ids)
W_gal, ids_gal = read_gal(joinpath(dir, "grid.gal"))
println("  GAL: exact round-trip = ", maximum(abs.(W_gal .- W_rook)) < 1e-12,
        ",  ids preserved = ", ids_gal == string.(ids))

write_gwt(joinpath(dir, "grid.gwt"), W_rook, ids)
W_gwt, _ = read_gwt(joinpath(dir, "grid.gwt"))
println("  GWT: exact round-trip = ", maximum(abs.(W_gwt .- W_rook)) < 1e-12)

# Row i of W corresponds to ids[i]. A GAL file carries its own ordering in
# its observation records. A GWT file is only an edge list, so read_gwt
# sorts the ids it finds — pass `ids` to impose an external table's order.
odd = [101, 7, 55, 3, 99, 42, 88, 12, 76]
W3 = make_rook_weights(3)
write_gwt(joinpath(dir, "odd.gwt"), W3, odd)
W3_sorted, ids_sorted = read_gwt(joinpath(dir, "odd.gwt"))
W3_given,  ids_given  = read_gwt(joinpath(dir, "odd.gwt"); ids = odd)
println("  GWT with non-contiguous ids ", odd, ":")
println("    no `ids` given -> reader sorts them:  ", ids_sorted)
println("      the matrix is a consistent relabelling, not the source order: ",
        maximum(abs.(W3_sorted .- W3)) < 1e-12 ? "same" : "permuted")
println("    `ids` given    -> source order kept:  ", ids_given)
println("      recovers the source matrix exactly: ",
        maximum(abs.(W3_given .- W3)) < 1e-12)
rm(dir; recursive = true, force = true)
println()

# ─── W matters ───────────────────────────────────────────────────────

println("═══ The Specification of W Moves the Estimates ═══")
Ng = 144
Wg = make_rook_weights(12)
Xg = randn(rng, Ng, 2)
yg = (sparse(1.0I, Ng, Ng) - 0.6 * Wg) \ (1.0 .+ Xg * [1.5, -0.8] .+ 0.5 * randn(rng, Ng))

# Same data, several different weight matrices built on the grid centroids.
grid_coords = hcat(repeat(1.0:12.0, outer = 12), repeat(1.0:12.0, inner = 12))

# A deliberately misspecified W: the right *shape*, but the neighbour
# relations shuffled, so it encodes no real spatial structure.
scrambled = knn_weights(grid_coords[randperm(rng, Ng), :], 4)

println(rpad("  W", 26), rpad("ρ̂", 10), "β̂₁")
for (lbl, Wc) in [("rook (true)", Wg),
                  ("queen", make_queen_weights(12)),
                  ("knn k=4", knn_weights(grid_coords, 4)),
                  ("knn k=8", knn_weights(grid_coords, 8)),
                  ("scrambled (wrong)", scrambled)]
    mc = fit_spatial(CS.SAR(), Wc, Xg, yg)
    println(rpad("  $lbl", 26), rpad(round(mc.rho; digits = 4), 10),
            round(coef(mc)[2]; digits = 4))
end
println()
println("  True ρ = 0.6, β₁ = 1.5. The three plausible neighbour definitions")
println("  all land close: a reasonable W need not be the exact one. The")
println("  scrambled W is what misspecification actually looks like — ρ̂")
println("  collapses toward zero, because there is no structure left to find.")
println()
println("  Note also that ρ is defined relative to the W you chose, so it is")
println("  not comparable across specifications. Always report W with ρ.")
