module SpatialWeights

using LinearAlgebra
using SparseArrays
using Statistics
using NearestNeighbors
using Printf

export
    # Construction
    knn_weights, distance_weights, band_weights, block_diagonal_weights,
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

    tree = KDTree(permutedims(Float64.(coords)))
    idxs, dists = knn(tree, permutedims(Float64.(coords)), k + 1, true)

    I_idx = Int[]; J_idx = Int[]; V_val = Float64[]
    for i in 1:N
        for m in 1:length(idxs[i])
            j = idxs[i][m]
            j == i && continue
            length(filter(x -> x != i, idxs[i])) >= k && count(x -> x != i, idxs[i][1:m]) > k && continue
            push!(I_idx, i); push!(J_idx, j)
            push!(V_val, binary ? 1.0 : 1.0 / max(dists[i][m], eps()))
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

# ─── Standardization ─────────────────────────────────────────────────

"""
    row_standardize!(W)

Row-standardize a sparse weight matrix in place: each row sums to 1.
Rows with no neighbors are left as zeros.
"""
function row_standardize!(W::AbstractSparseMatrix)
    N = size(W, 1)
    rows = rowvals(W); vals = nonzeros(W)
    for i in 1:N
        rs = 0.0
        for col in 1:size(W, 2)
            for idx in nzrange(W, col)
                rows[idx] == i && (rs += abs(vals[idx]))
            end
        end
        if rs > 0.0
            for col in 1:size(W, 2)
                for idx in nzrange(W, col)
                    rows[idx] == i && (vals[idx] /= rs)
                end
            end
        end
    end
    return W
end

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

function minmax_standardize(W::AbstractSparseMatrix)
    W_copy = copy(W)
    minmax_standardize!(W_copy)
    return W_copy
end

# ─── Connectivity Diagnostics ────────────────────────────────────────

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
