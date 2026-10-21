# Weight Matrix Construction

## From Coordinates

### k-Nearest Neighbors

```julia
W = knn_weights(coords, k;
    binary=false,          # true → binary (0/1), false → inverse distance
    row_standardize=true,  # row sums = 1
    symmetrize=false)      # force symmetry via max(W, W')
```

### Distance Decay

```julia
W = distance_weights(coords, cutoff;
    decay=:inverse,   # :inverse, :negative_exponential, :bisquare, :power
    alpha=1.0,         # decay parameter
    row_standardize=true)
```

### Distance Band

```julia
W = band_weights(coords, d_min, d_max; row_standardize=true)
```

## From Regular Grids

```julia
W = make_rook_weights(L)   # L×L grid, 4-connected
W = make_queen_weights(L)  # L×L grid, 8-connected
```

## Combining

```julia
W = block_diagonal_weights([W_region1, W_region2, W_region3])
```

## Standardization

```julia
row_standardize!(W)      # each row sums to 1
spectral_standardize!(W) # divide by spectral radius
minmax_standardize!(W)   # MinMax normalization
```

Non-mutating versions (`row_standardize`, `spectral_standardize`, `minmax_standardize`) return a copy.

## Diagnostics

```julia
sc = spatial_connectivity(W)
# sc.N, sc.n_nonzero, sc.density
# sc.avg_neighbors, sc.min_neighbors, sc.max_neighbors
# sc.n_islands, sc.is_symmetric, sc.spectral_radius
```

## I/O (GeoDa Formats)

```julia
# GAL (contiguity)
write_gal("weights.gal", W, ids)
W, ids = read_gal("weights.gal")

# GWT (general weights)
write_gwt("weights.gwt", W, ids)
W, ids = read_gwt("weights.gwt")
```

Both readers return the unit ids alongside the matrix, and row `i` of `W`
corresponds to `ids[i]`. A GAL file carries its own ordering in its observation
records, so `read_gal` uses that. A GWT file does not — it is only an edge list —
so `read_gwt` sorts the ids it finds (numerically when they all parse as
integers). Pass `ids` to either reader to impose the ordering of an external
table instead:

```julia
W, _ = read_gwt("weights.gwt"; ids = df.region_code)
```

An id appearing in a neighbour list but in no observation record is treated as a
malformed file and raises an error rather than being given an invented index.
