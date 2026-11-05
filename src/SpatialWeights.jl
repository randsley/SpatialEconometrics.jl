module SpatialWeights

using LinearAlgebra
using SparseArrays
using Statistics
using NearestNeighbors
using Printf

export
    # Construction
    knn_weights, distance_weights, band_weights, block_diagonal_weights,
    order_weights, order_weights_list,
    # Standardization
    row_standardize, row_standardize!, spectral_standardize, spectral_standardize!, minmax_standardize, minmax_standardize!,
    # Diagnostics
    SpatialConnectivity, spatial_connectivity,
    # I/O
    read_gal, write_gal, read_gwt, write_gwt

# ─── k-Nearest Neighbors ─────────────────────────────────────────────

"""
    knn_weights(coords, k; binary=false, row_standardize=true, symmetrize=false)

Build a spatial weight matrix from k-nearest neighbors.
`coords` is N×D (rows = observations). Returns a sparse N×N matrix.
"""
function knn_weights(
    coords::AbstractMatrix{<:Real}, k::Int;
    binary::Bool=false, row_standardize::Bool=true, symmetrize::Bool=false
)
    N, D = size(coords)
    @assert k >= 1 && k < N "k must be in [1, N-1]"

    pts = permutedims(Float64.(coords))     # built once, queried once
    tree = KDTree(pts)
    idxs, dists = knn(tree, pts, k + 1, true)

    # `k + 1` neighbours are requested so that self can be dropped and `k`
    # remain. The selection below used to re-derive, for every candidate,
    # how many non-self neighbours the list holds and how many have been
    # seen so far -- a `filter` that allocates and a `count` over a fresh
    # slice, both inside the inner loop, so O(k²) work and O(k) allocations
    # per observation to decide something a running counter decides in O(1).
    # The rule is unchanged: keep the first `k` non-self neighbours, and if
    # self is absent from the list (which happens when points coincide) keep
    # `k` of the `k + 1` on offer.
    I_idx = Int[]; J_idx = Int[]; V_val = Float64[]
    sizehint!(I_idx, N * k); sizehint!(J_idx, N * k); sizehint!(V_val, N * k)
    for i in 1:N
        nbr = idxs[i]
        dst = dists[i]
        nonself = 0
        @inbounds for x in nbr
            x != i && (nonself += 1)
        end
        kept = 0
        @inbounds for m in eachindex(nbr)
            j = nbr[m]
            j == i && continue
            nonself >= k && kept >= k && continue
            push!(I_idx, i); push!(J_idx, j)
            push!(V_val, binary ? 1.0 : 1.0 / max(dst[m], eps()))
            kept += 1
        end
    end

    W = sparse(I_idx, J_idx, V_val, N, N)

    if symmetrize
        W = max.(W, W')
    end

    if row_standardize
        row_standardize!(W)
    end

    return W
end

# ─── Distance-Decay Weights ──────────────────────────────────────────

"""
    distance_weights(coords, cutoff; decay=:inverse, alpha=1.0, row_standardize=true)

Build distance-decay spatial weights. Pairs beyond `cutoff` get zero weight.

Decay functions:
- `:inverse`  — `1 / d^alpha`
- `:negative_exponential` — `exp(-alpha * d)`
- `:power` — `d^(-alpha)` (same as inverse, explicit name)
- `:bisquare` — `(1 - (d/cutoff)^2)^2`
"""
function distance_weights(
    coords::AbstractMatrix{<:Real}, cutoff::Real;
    decay::Symbol=:inverse, alpha::Real=1.0, row_standardize::Bool=true
)
    N = size(coords, 1)
    @assert cutoff > 0 "cutoff must be positive"

    tree = BallTree(permutedims(Float64.(coords)))
    idxs_all = inrange(tree, permutedims(Float64.(coords)), Float64(cutoff), true)

    I_idx = Int[]; J_idx = Int[]; V_val = Float64[]
    coords_f = Float64.(coords)

    for i in 1:N
        for j in idxs_all[i]
            j == i && continue
            d = sqrt(sum((coords_f[i, :] .- coords_f[j, :]).^2))
            d > cutoff && continue
            d = max(d, eps())

            w = if decay == :inverse || decay == :power
                1.0 / d^alpha
            elseif decay == :negative_exponential
                exp(-alpha * d)
            elseif decay == :bisquare
                r = d / cutoff
                (1.0 - r^2)^2
            else
                error("Unknown decay function: $decay. Use :inverse, :negative_exponential, :power, or :bisquare")
            end

            push!(I_idx, i); push!(J_idx, j); push!(V_val, w)
        end
    end

    W = sparse(I_idx, J_idx, V_val, N, N)
    row_standardize && row_standardize!(W)
    return W
end

# ─── Distance-Band (Binary) Weights ─────────────────────────────────

"""
    band_weights(coords, d_min, d_max; row_standardize=true)

Binary weights: `w_ij = 1` if `d_min < d_ij ≤ d_max`, else 0.
"""
function band_weights(
    coords::AbstractMatrix{<:Real}, d_min::Real, d_max::Real;
    row_standardize::Bool=true
)
    N = size(coords, 1)
    @assert d_max > d_min >= 0 "need d_max > d_min >= 0"

    tree = BallTree(permutedims(Float64.(coords)))
    idxs_outer = inrange(tree, permutedims(Float64.(coords)), Float64(d_max), true)

    I_idx = Int[]; J_idx = Int[]
    coords_f = Float64.(coords)

    for i in 1:N
        for j in idxs_outer[i]
            j == i && continue
            d = sqrt(sum((coords_f[i, :] .- coords_f[j, :]).^2))
            if d > d_min && d <= d_max
                push!(I_idx, i); push!(J_idx, j)
            end
        end
    end

    W = sparse(I_idx, J_idx, ones(length(I_idx)), N, N)
    row_standardize && row_standardize!(W)
    return W
end

# ─── Block-Diagonal Weights ──────────────────────────────────────────

"""
    block_diagonal_weights(W_list)

Combine a vector of sub-region weight matrices into one block-diagonal matrix.
"""
function block_diagonal_weights(W_list::Vector{<:AbstractSparseMatrix})
    isempty(W_list) && error("W_list must not be empty")
    N_total = sum(size(W, 1) for W in W_list)
    I_idx = Int[]; J_idx = Int[]; V_val = Float64[]
    offset = 0
    for W in W_list
        rows = rowvals(W)
        vals = nonzeros(W)
        n = size(W, 1)
        for col in 1:n
            for idx in nzrange(W, col)
                push!(I_idx, rows[idx] + offset)
                push!(J_idx, col + offset)
                push!(V_val, Float64(vals[idx]))
            end
        end
        offset += n
    end
    return sparse(I_idx, J_idx, V_val, N_total, N_total)
end

# ─── Higher-Order Contiguity Weights ─────────────────────────────────

"""
    order_weights(W, l; row_standardize = true, binary = true)

The **order-`l` spatial weights matrix**: unit `j` is a neighbour of `i` iff
the shortest path from `i` to `j` in `W`'s adjacency graph has exactly `l`
steps. `l = 0` returns the identity.

This is the `W⁽ˡ⁾` of Anselin & Smirnov (1996), and it is **not** `W^l`.

!!! warning "`W^l` is a different object"
    `W²` counts *walks* of length two, so it is non-zero on the diagonal —
    every `i` reaches itself back through any neighbour — and it
    double-counts units reachable by several two-step paths.
    `order_weights(W, 2)` instead **partitions** the pairs: each `j`
    appears at exactly one order, its shortest-path distance, and the
    diagonal is empty for `l ≥ 1`.

    Measured on a 4×4 rook lattice (`N = 16`, 48 first-order links): `W²`
    has **84 non-zeros, 16 of them on the diagonal**, against **68 and 0**
    for `order_weights(W, 2)`. Summed over orders 1 to 6 — 6 being the
    lattice's diameter — `order_weights` yields exactly `N(N−1) = 240`
    ordered pairs, each once.

    `W²` also retains first-order neighbours wherever a triangle exists,
    which `order_weights` never does. On a triangle with one pendant unit,
    `W²` is non-zero on **6** pairs that are already first-order
    neighbours; `order_weights(W, 2)` is non-zero on **0**.

The construction is a breadth-first search from each unit, so the cost is
`O(N·nnz(W))` and the result depends only on `W`'s sparsity pattern, not on
its values — any row-standardisation already applied to `W` is irrelevant
to which pairs come out at which order.

Pass `binary = false` to weight each order-`l` link by `1/d` where `d` is
the number of distinct shortest paths; the default `true` gives the usual
binary-then-row-standardised contiguity matrix.

# Arguments
- `W`: the first-order weights or adjacency matrix, `N×N`. Only its pattern
  is read.
- `l`: the order. `0` gives `I`; `l` greater than the graph diameter gives
  an all-zero matrix, which `order_weights_list` refuses rather than
  returning.

See also [`order_weights_list`](@ref), which returns `[W⁽⁰⁾, …, W⁽ˢ⁾]` and
checks the diameter once for the whole list.
"""
function order_weights(
    W::AbstractSparseMatrix, l::Int;
    row_standardize::Bool = true, binary::Bool = true
)
    N = size(W, 1)
    size(W, 2) == N || throw(DimensionMismatch(
        "order_weights: W is $(size(W)) but must be square"))
    l >= 0 || throw(ArgumentError("order_weights: l must be ≥ 0, got $l"))
    l == 0 && return sparse(1.0I, N, N)

    # Adjacency pattern, ignoring values and any self-loops.
    A = [Int[] for _ in 1:N]
    rows = rowvals(W)
    vals = nonzeros(W)
    for j in 1:N
        for idx in nzrange(W, j)
            i = rows[idx]
            (i == j || iszero(vals[idx])) && continue
            push!(A[i], j)        # W[i, j] != 0  ⇒  j is a neighbour of i
        end
    end

    I_idx = Int[]; J_idx = Int[]; V_val = Float64[]
    dist = fill(-1, N)
    npath = zeros(Int, N)
    frontier = Int[]; nxt = Int[]

    for src in 1:N
        fill!(dist, -1); fill!(npath, 0)
        dist[src] = 0; npath[src] = 1
        empty!(frontier); push!(frontier, src)
        for step in 1:l
            empty!(nxt)
            for u in frontier, v in A[u]
                if dist[v] == -1
                    dist[v] = step
                    npath[v] = npath[u]
                    push!(nxt, v)
                elseif dist[v] == step
                    npath[v] += npath[u]
                end
            end
            isempty(nxt) && break
            frontier, nxt = nxt, frontier
        end
        for v in frontier
            dist[v] == l || continue
            push!(I_idx, src); push!(J_idx, v)
            push!(V_val, binary ? 1.0 : 1.0 / npath[v])
        end
    end

    Wl = sparse(I_idx, J_idx, V_val, N, N)
    row_standardize && row_standardize!(Wl)
    return Wl
end

"""
    order_weights_list(W, s; row_standardize = true, binary = true)

`[W⁽⁰⁾, W⁽¹⁾, …, W⁽ˢ⁾]` with `W⁽⁰⁾ = I`, each built by
[`order_weights`](@ref).

Refuses when `s` exceeds the graph diameter, rather than returning an
all-zero matrix for the orders that do not exist: a model fitted with an
empty `W⁽ˡ⁾` has a structurally zero column, and the coefficient on it is
not identified. The error reports the diameter so the caller can lower `s`.

A disconnected `W` has no finite diameter; the check is applied per
component, so `s` may be at most the largest component's diameter and the
orders remain well defined within components.
"""
function order_weights_list(
    W::AbstractSparseMatrix, s::Int;
    row_standardize::Bool = true, binary::Bool = true
)
    s >= 0 || throw(ArgumentError("order_weights_list: s must be ≥ 0, got $s"))
    out = Vector{SparseMatrixCSC{Float64, Int}}(undef, s + 1)
    out[1] = sparse(1.0I, size(W, 1), size(W, 1))
    for l in 1:s
        Wl = order_weights(W, l; row_standardize = row_standardize,
                           binary = binary)
        if nnz(Wl) == 0
            diam = l - 1
            throw(ArgumentError(
                "order_weights_list: no pairs at order $l, so W⁽$(l)⁾ is " *
                "empty and its coefficient would not be identified. The " *
                "largest component's diameter is $diam, so s ≤ $diam."))
        end
        out[l + 1] = Wl
    end
    return out
end

# ─── Standardization ─────────────────────────────────────────────────

"""
    row_standardize!(W)

Row-standardize a sparse weight matrix in place: each row sums to 1.
Rows with no neighbors are left as zeros.
"""
function row_standardize!(W::AbstractSparseMatrix)
    N = size(W, 1)
    rows = rowvals(W); vals = nonzeros(W)
    # Two passes over the nonzeros, not one pass per row: accumulating the
    # row sums into a vector costs O(nnz) where rescanning the whole matrix
    # for each row in turn costs O(N * nnz). Both visit a given row's entries
    # in increasing column order, so the summation order -- and therefore the
    # result, to the last bit -- is unchanged.
    rowsums = zeros(eltype(vals), N)
    @inbounds for idx in eachindex(vals)
        rowsums[rows[idx]] += abs(vals[idx])
    end
    @inbounds for idx in eachindex(vals)
        rs = rowsums[rows[idx]]
        rs > 0 && (vals[idx] /= rs)
    end
    return W
end

"""
    row_standardize(W) -> SparseMatrixCSC

Row-standardized copy of `W`: every row with at least one neighbour sums
to 1. See [`row_standardize!`](@ref) for the in-place version.

This is the conventional normalization, and most of this package assumes
it. It makes `W*y` a neighbourhood average, bounds the spatial parameter
inside `(-1, 1)`, and makes Moran's I equal the slope in a Moran scatter
plot. Rows with no neighbours stay all-zero.
"""
function row_standardize(W::AbstractSparseMatrix)
    W_copy = copy(W)
    row_standardize!(W_copy)
    return W_copy
end

"""
    spectral_standardize!(W)

Divide all entries by the spectral radius (largest eigenvalue magnitude).
"""
function spectral_standardize!(W::AbstractSparseMatrix)
    λ_max = _spectral_radius(W)
    if λ_max > 0.0
        nonzeros(W) ./= λ_max
    end
    return W
end

"""
    spectral_standardize(W) -> SparseMatrixCSC

Copy of `W` divided by its spectral radius. See
[`spectral_standardize!`](@ref) for the in-place version.

Bounds the spatial parameter in `(-1, 1)` as row-standardization does, but
without rescaling rows relative to one another, so differences in
connectedness between units are preserved. Preferable when the number of
neighbours is itself meaningful; the cost is that `W*y` is no longer a
simple neighbourhood average.
"""
function spectral_standardize(W::AbstractSparseMatrix)
    W_copy = copy(W)
    spectral_standardize!(W_copy)
    return W_copy
end

"""
    minmax_standardize!(W)

MinMax standardization: `W / (min(r_max) + max(r_max))` where `r_max` is the
vector of max absolute row sums and max absolute column sums.
"""
function minmax_standardize!(W::AbstractSparseMatrix)
    N = size(W, 1)
    row_max = zeros(N); col_max = zeros(N)
    rows = rowvals(W); vals = nonzeros(W)
    for col in 1:N
        for idx in nzrange(W, col)
            v = abs(vals[idx])
            row_max[rows[idx]] = max(row_max[rows[idx]], v)
            col_max[col] = max(col_max[col], v)
        end
    end
    denom = minimum(vcat(row_max[row_max .> 0], col_max[col_max .> 0])) +
            maximum(vcat(row_max, col_max))
    if denom > 0.0
        nonzeros(W) ./= denom
    end
    return W
end

"""
    minmax_standardize(W) -> SparseMatrixCSC

Copy of `W` divided by the smaller of its largest row sum and largest
column sum. See [`minmax_standardize!`](@ref) for the in-place version.

A middle course between [`row_standardize`](@ref) and
[`spectral_standardize`](@ref): it keeps relative connectedness while being
cheaper than an eigenvalue computation, which matters for large `W`.
"""
function minmax_standardize(W::AbstractSparseMatrix)
    W_copy = copy(W)
    minmax_standardize!(W_copy)
    return W_copy
end

# ─── Connectivity Diagnostics ────────────────────────────────────────

"""
    SpatialConnectivity

Summary of a weight matrix's structure, returned by
[`spatial_connectivity`](@ref). Worth looking at before fitting anything.

# Fields
- `N`, `n_nonzero`, `density`: size and sparsity.
- `avg_neighbors`, `min_neighbors`, `max_neighbors`: the degree
  distribution.
- `n_islands`: units with no neighbours at all.
- `is_symmetric`: whether `W == W'`. Row-standardizing a symmetric
  contiguity matrix generally destroys symmetry, so `false` here is
  normal and not a problem.
- `spectral_radius`: largest eigenvalue magnitude; 1 for a
  row-standardized matrix.

!!! warning
    `n_islands > 0` deserves attention. An isolated unit contributes
    nothing to the spatial likelihood and its row of `W*y` is zero, so it
    is effectively dropped from the spatial part of the model while still
    counting towards `N`.
"""
struct SpatialConnectivity
    N::Int
    n_nonzero::Int
    density::Float64
    avg_neighbors::Float64
    min_neighbors::Int
    max_neighbors::Int
    n_islands::Int
    is_symmetric::Bool
    spectral_radius::Float64
end

function Base.show(io::IO, sc::SpatialConnectivity)
    println(io, "Spatial Weight Matrix Connectivity")
    println(io, "──────────────────────────────────")
    @printf(io, "  Observations:      %d\n", sc.N)
    @printf(io, "  Non-zero entries:  %d\n", sc.n_nonzero)
    @printf(io, "  Density:           %.4f\n", sc.density)
    @printf(io, "  Avg neighbors:     %.2f\n", sc.avg_neighbors)
    @printf(io, "  Min neighbors:     %d\n", sc.min_neighbors)
    @printf(io, "  Max neighbors:     %d\n", sc.max_neighbors)
    @printf(io, "  Islands (isolates): %d\n", sc.n_islands)
    @printf(io, "  Symmetric:         %s\n", sc.is_symmetric ? "yes" : "no")
    @printf(io, "  Spectral radius:   %.4f\n", sc.spectral_radius)
end

"""
    spatial_connectivity(W)

Compute diagnostic statistics for a spatial weight matrix.
"""
function spatial_connectivity(W::AbstractSparseMatrix)
    N = size(W, 1)
    neighbor_counts = zeros(Int, N)
    rows = rowvals(W); vals = nonzeros(W)
    for col in 1:N
        for idx in nzrange(W, col)
            if abs(vals[idx]) > 0.0
                neighbor_counts[rows[idx]] += 1
            end
        end
    end

    n_islands = count(neighbor_counts .== 0)
    is_sym = issymmetric(W) || norm(W - W', Inf) < 1e-10

    SpatialConnectivity(
        N,
        nnz(W),
        nnz(W) / (N * (N - 1)),
        mean(neighbor_counts),
        minimum(neighbor_counts),
        maximum(neighbor_counts),
        n_islands,
        is_sym,
        _spectral_radius(W)
    )
end

function _spectral_radius(W::AbstractSparseMatrix)
    N = size(W, 1)
    if N <= 500
        return maximum(abs.(eigvals(Matrix(W))))
    end
    v = randn(N)
    v ./= norm(v)
    λ_est = 0.0
    for _ in 1:100
        w = W * v
        λ_new = dot(v, w)
        w_norm = norm(w)
        w_norm < eps() && break
        v .= w ./ w_norm
        abs(λ_new - λ_est) < 1e-10 && (λ_est = abs(λ_new); break)
        λ_est = abs(λ_new)
    end
    return λ_est
end

# ─── GAL Format I/O ──────────────────────────────────────────────────

"""
    read_gal(path; ids=nothing) → (W, ids)

Read a GeoDa GAL contiguity weights file. Returns the row-standardized sparse
weight matrix and the observation ID vector, where row `i` of `W` corresponds
to `ids[i]`.

The ordering comes from the file's observation records, in record order. Pass
`ids` to impose an external table's ordering instead. A neighbour id with no
observation record raises an error.
"""
function read_gal(path::AbstractString; ids::Union{Nothing, AbstractVector}=nothing)
    lines = readlines(path)
    isempty(lines) && error("Empty GAL file")

    header = split(strip(lines[1]))
    N = length(header) >= 2 ? parse(Int, header[2]) : parse(Int, header[1])

    # Pass 1: collect the observation records. A record is an "id count" line
    # followed by a line listing that unit's neighbours (absent when count = 0).
    rec_ids = String[]
    rec_nbrs = Vector{Vector{String}}()
    line_idx = 2
    while line_idx <= length(lines) && length(rec_ids) < N
        parts = split(strip(lines[line_idx]))
        if isempty(parts)
            line_idx += 1
            continue
        end
        length(parts) < 2 && error("Malformed GAL record on line $line_idx: $(lines[line_idx])")
        push!(rec_ids, parts[1])
        n_neighbors = parse(Int, parts[2])
        line_idx += 1
        if n_neighbors > 0
            line_idx > length(lines) &&
                error("GAL file ends before the neighbour list for unit $(rec_ids[end])")
            push!(rec_nbrs, split(strip(lines[line_idx])))
            line_idx += 1
        else
            push!(rec_nbrs, String[])
        end
    end

    length(rec_ids) == N ||
        error("GAL header declares $N units but only $(length(rec_ids)) records were found")

    # The ordering comes from the observation records (or from `ids`, to align
    # with an external table), NEVER from the order in which ids happen to be
    # mentioned. Assigning indices on first appearance put a neighbour listed
    # before its own record at the wrong index, silently permuting the units.
    order = isnothing(ids) ? rec_ids : string.(ids)
    length(order) == N ||
        error("`ids` has $(length(order)) entries but the GAL file declares $N units")
    id_map = Dict{String, Int}(id => k for (k, id) in enumerate(order))
    length(id_map) == N || error("duplicate unit ids in GAL file")

    # Pass 2: resolve the edges against that fixed map.
    I_idx = Int[]; J_idx = Int[]
    for (rid, nbrs) in zip(rec_ids, rec_nbrs)
        i = get(id_map, rid, 0)
        i == 0 && error("GAL unit id $(rid) is not present in the id ordering")
        for nb in nbrs
            j = get(id_map, nb, 0)
            j == 0 && error("GAL unit $(rid) lists unknown neighbour id $(nb)")
            push!(I_idx, i); push!(J_idx, j)
        end
    end

    W = sparse(I_idx, J_idx, ones(length(I_idx)), N, N)
    row_standardize!(W)
    return W, order
end

"""
    write_gal(path, W, ids)

Write a weight matrix in GeoDa GAL format.
"""
function write_gal(path::AbstractString, W::AbstractSparseMatrix, ids::AbstractVector)
    N = size(W, 1)
    @assert length(ids) == N "ids length must match W dimension"

    open(path, "w") do io
        println(io, "0 $N")
        rows = rowvals(W); vals = nonzeros(W)
        for i in 1:N
            neighbors = Int[]
            for col in 1:N
                for idx in nzrange(W, col)
                    if rows[idx] == i && abs(vals[idx]) > 0.0
                        push!(neighbors, col)
                    end
                end
            end
            println(io, "$(ids[i]) $(length(neighbors))")
            if !isempty(neighbors)
                println(io, join([ids[j] for j in neighbors], " "))
            end
        end
    end
    return nothing
end

# ─── GWT Format I/O ──────────────────────────────────────────────────

"""
    read_gwt(path; ids=nothing) → (W, ids)

Read a GeoDa GWT (general weights) file. Returns the sparse weight matrix and
the observation ID vector, where row `i` of `W` corresponds to `ids[i]`.

A GWT file is only an edge list and carries no ordering of its own, so the ids
found in it are sorted — numerically when they all parse as integers, else
lexicographically. Pass `ids` to impose an external table's ordering instead.
"""
function read_gwt(path::AbstractString; ids::Union{Nothing, AbstractVector}=nothing)
    lines = readlines(path)
    isempty(lines) && error("Empty GWT file")

    header = split(strip(lines[1]))
    N = parse(Int, header[2])

    # Pass 1: read every (origin, destination, weight) triple.
    src = String[]; dst = String[]; V_val = Float64[]
    for line_idx in 2:length(lines)
        parts = split(strip(lines[line_idx]))
        length(parts) < 3 && continue
        push!(src, parts[1]); push!(dst, parts[2])
        push!(V_val, parse(Float64, parts[3]))
    end

    # A GWT file carries no unit ordering of its own, so one must be imposed
    # deterministically. Indexing ids by first appearance made the matrix depend
    # on the order the edges happened to be written, silently permuting the
    # units; pass `ids` to align with an external table instead.
    order = if !isnothing(ids)
        string.(ids)
    else
        uniq = unique(vcat(src, dst))
        # Sort numerically when every id is an integer, else lexicographically.
        nums = tryparse.(Int, uniq)
        all(!isnothing, nums) ? uniq[sortperm(Int.(nums))] : sort(uniq)
    end
    length(order) <= N ||
        error("GWT file references $(length(order)) distinct ids but the header declares $N")
    id_map = Dict{String, Int}(id => k for (k, id) in enumerate(order))

    I_idx = Int[]; J_idx = Int[]
    for (a, b) in zip(src, dst)
        i = get(id_map, a, 0); j = get(id_map, b, 0)
        (i == 0 || j == 0) && error("GWT edge references unknown id: $(a) -> $(b)")
        push!(I_idx, i); push!(J_idx, j)
    end

    W = sparse(I_idx, J_idx, V_val, N, N)
    return W, order
end

"""
    write_gwt(path, W, ids; threshold=0.0)

Write a weight matrix in GeoDa GWT format.
"""
function write_gwt(path::AbstractString, W::AbstractSparseMatrix, ids::AbstractVector; threshold::Float64=0.0)
    N = size(W, 1)
    @assert length(ids) == N "ids length must match W dimension"

    open(path, "w") do io
        println(io, "0 $N 0")
        rows = rowvals(W); vals = nonzeros(W)
        for col in 1:N
            for idx in nzrange(W, col)
                v = vals[idx]
                abs(v) > threshold || continue
                println(io, "$(ids[rows[idx]]) $(ids[col]) $v")
            end
        end
    end
    return nothing
end

end # module
