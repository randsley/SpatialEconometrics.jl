"""
    SpatialNetworks

Spatial econometrics on networks rather than on geography.

Everything else in this package builds `W` from distance: `knn_weights`,
`distance_weights`, `band_weights`. Here `W` comes from a graph — trade
links, friendships, supply chains, citations — and the econometrics
changes with it in one specific way that matters more than the
construction does.

In a geographic model, `y = ρWy + Xβ + WXθ + ε` is identified because
distance decays: a unit's neighbours and its neighbours' neighbours are
different sets. On a network that is not automatic. Bramoullé, Djebbari and
Fortin (2009) show that the endogenous effect `β` and the contextual effect
`δ` are separately identified **if and only if** `I`, `G` and `G²` are
linearly independent, and that this fails exactly on the structure applied
work reaches for first: interaction in equal-sized groups. That is Manski's
(1993) reflection problem, and it is an identification failure, so no
quantity of data repairs it.

So the centre of this module is [`peer_identification`](@ref), which answers that
question mechanically before anything is estimated, and
[`fit_peer_effects`](@ref), which refuses to report a peer effect it cannot
identify.

# References
- Bramoullé, Y., Djebbari, H. & Fortin, B. (2009). Identification of peer
  effects through social networks. *Journal of Econometrics*, 150(1),
  41–55.
- Manski, C. F. (1993). Identification of endogenous social effects: the
  reflection problem. *Review of Economic Studies*, 60(3), 531–542.
- Leenders, R. T. A. J. (2002). Modeling social influence through network
  autocorrelation: constructing the weight matrix. *Social Networks*,
  24(1), 21–47.
- Lee, L. F. & Liu, X. (2010). Efficient GMM estimation of high order
  spatial autoregressive models with autoregressive disturbances.
  *Econometric Theory*, 26(1), 187–230.
"""
module SpatialNetworks

using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames, dof_residual

import ..SpatialUtils: check_design_rank, _print_coef_table, _normcdf,
                       _chi2_sf, brent_maximize
import ..SpatialWeights: row_standardize
import ..SpatialDiagnostics: MoranResult, moran_global
import Random
import Random: AbstractRNG

export network_weights, group_weights,
       PeerIdentification, peer_identification,
       PeerEffectsFit, fit_peer_effects,
       MultiplexSAR, fit_multiplex_sar,
       network_moran

# ═══════════════════════════════════════════════════════════════════════
# Weight construction
# ═══════════════════════════════════════════════════════════════════════

"""
    network_weights(A; normalize = :row, allow_self = false) -> SparseMatrixCSC

Interaction matrix from a network adjacency matrix.

`A` may be dense or sparse, square, symmetric (an undirected graph) or not
(a directed one — citations, trade flows, who-follows-whom). Entries may be
0/1 or arbitrary non-negative strengths; Leenders (2002) is a sustained
argument that the choice of entries *is* the model of social influence, not
a preliminary to it, so nothing is assumed about them here beyond
non-negativity.

# Keywords
- `normalize`: `:row` (default) divides each row by its sum, making `Gy` a
  neighbourhood average and bounding the interaction parameter inside
  `(-1, 1)`; `:none` leaves the entries alone.
- `allow_self`: by default a non-zero diagonal is an error, because a
  self-link makes `Gy` depend on `yᵢ` directly and the interaction
  parameter stops meaning what the model says it means. Pass `true` to keep
  it, or set the diagonal to zero yourself.

# Isolates
Rows with no links stay all-zero rather than becoming `NaN`. This is
deliberate and networks need it — a geographic `W` rarely has isolates and a
social one usually does. But note that the second half of Bramoullé,
Djebbari and Fortin's Proposition 1 (the *non*-identification half) is
stated for networks with no isolated individual, so
[`peer_identification`](@ref) reports the isolate count and withholds that half of
the verdict when any are present.

# Examples
```julia
A = sparse([0 1 1; 1 0 0; 1 0 0])
G = network_weights(A)             # row-standardised, undirected
```

See also [`group_weights`](@ref), [`peer_identification`](@ref).
"""
function network_weights(A::AbstractMatrix;
                         normalize::Symbol = :row,
                         allow_self::Bool = false)
    n, m = size(A)
    n == m || throw(DimensionMismatch(
        "network_weights: the adjacency matrix must be square, got $(n)×$(m)."))
    normalize in (:row, :none) || throw(ArgumentError(
        "network_weights: `normalize` must be :row or :none, got :$(normalize)."))

    S = sparse(Float64.(A))
    any(x -> x < 0, nonzeros(S)) && throw(ArgumentError(
        "network_weights: negative adjacency entries. A link strength below " *
        "zero has no interpretation as exposure, and row standardisation " *
        "would not preserve its sign."))

    if !allow_self
        d = diag(S)
        bad = findall(!iszero, d)
        isempty(bad) || throw(ArgumentError(
            "network_weights: the adjacency matrix has $(length(bad)) " *
            "non-zero diagonal entr$(length(bad) == 1 ? "y" : "ies") " *
            "(first at index $(bad[1])). A self-link makes `Gy` depend on " *
            "`y[i]` itself, so the endogenous effect is no longer the effect " *
            "of peers. Zero the diagonal, or pass `allow_self = true` if the " *
            "self-reference is intended."))
    end

    return normalize === :row ? row_standardize(S) : S
end

"""
    network_weights(edges, n; directed = false, normalize = :row, allow_self = false)

Interaction matrix from an edge list.

`edges` is a vector of `(i, j)` pairs or `(i, j, w)` triples; `n` is the
number of units, which must be given because isolates do not appear in an
edge list and would otherwise be silently dropped. With `directed = false`
each edge is entered both ways.

Duplicate edges are summed before normalisation, which matches the
multigraph reading (two links are twice the exposure).

# Examples
```julia
G = network_weights([(1, 2), (1, 3), (4, 5)], 6)   # unit 6 is an isolate
```
"""
function network_weights(edges::AbstractVector, n::Integer;
                         directed::Bool = false,
                         normalize::Symbol = :row,
                         allow_self::Bool = false)
    n > 0 || throw(ArgumentError("network_weights: `n` must be positive, got $n."))
    I_idx = Int[]; J_idx = Int[]; V = Float64[]
    for (k, e) in enumerate(edges)
        length(e) in (2, 3) || throw(ArgumentError(
            "network_weights: edge $k has $(length(e)) entries; expected " *
            "`(i, j)` or `(i, j, weight)`."))
        i, j = Int(e[1]), Int(e[2])
        w = length(e) == 3 ? Float64(e[3]) : 1.0
        (1 <= i <= n && 1 <= j <= n) || throw(ArgumentError(
            "network_weights: edge $k is ($i, $j) but `n` = $n."))
        w >= 0 || throw(ArgumentError(
            "network_weights: edge $k has negative weight $w."))
        push!(I_idx, i); push!(J_idx, j); push!(V, w)
        if !directed
            push!(I_idx, j); push!(J_idx, i); push!(V, w)
        end
    end
    A = sparse(I_idx, J_idx, V, n, n, +)
    return network_weights(A; normalize = normalize, allow_self = allow_self)
end

"""
    group_weights(sizes; include_self = false, normalize = true) -> SparseMatrixCSC

Block-diagonal interaction matrix for interaction in groups: everyone in a
group is affected by everyone else in it, and by nobody outside.

`sizes` gives the group sizes, so `group_weights([4, 4, 4])` is three
classrooms of four. This is the structure the peer-effects literature argues
about, and the two conventions are not interchangeable:

- `include_self = false` (default) is Moffitt's (2001) and Lee's (2007)
  convention — the group mean **excludes** `i`, so `Gᵢⱼ = 1/(s-1)` for
  `j ≠ i` in the group. Then `G² = I/(s-1) + ((s-2)/(s-1))G` within a block,
  so equal group sizes make `I, G, G²` dependent and identification fails —
  but **unequal sizes break the dependence and it succeeds**.
- `include_self = true` is Manski's (1993) convention — the mean includes
  `i`. Then `G² = G` exactly, whatever the sizes are, and social effects are
  never identified. This is the reflection problem in its original form.

That contrast is the cheapest demonstration of why
[`peer_identification`](@ref) exists, and both cases are pinned as tests.

`normalize = false` returns the unstandardised adjacency instead.

# Examples
```julia
peer_identification(group_weights([5, 5, 5])).identified        # false
peer_identification(group_weights([4, 5, 6])).identified        # true
peer_identification(group_weights([4, 5, 6]; include_self = true)).identified  # false
```
"""
function group_weights(sizes::AbstractVector{<:Integer};
                       include_self::Bool = false,
                       normalize::Bool = true)
    isempty(sizes) && throw(ArgumentError("group_weights: `sizes` is empty."))
    all(>(0), sizes) || throw(ArgumentError(
        "group_weights: group sizes must be positive, got $(collect(sizes))."))
    if !include_self && any(==(1), sizes)
        throw(ArgumentError(
            "group_weights: a group of size 1 has no peers once the unit " *
            "itself is excluded, so its row would be all-zero and the unit " *
            "would be an isolate. Use `include_self = true`, or drop the " *
            "singleton groups."))
    end
    n = sum(sizes)
    I_idx = Int[]; J_idx = Int[]; V = Float64[]
    off = 0
    for s in sizes
        for i in 1:s, j in 1:s
            (i == j && !include_self) && continue
            push!(I_idx, off + i); push!(J_idx, off + j); push!(V, 1.0)
        end
        off += s
    end
    A = sparse(I_idx, J_idx, V, n, n)
    return normalize ? row_standardize(A) : A
end

# ═══════════════════════════════════════════════════════════════════════
# The rank machinery: BDF footnote 17, and channel separability
# ═══════════════════════════════════════════════════════════════════════

# Relative tolerance for the rank decisions below.
#
# Not a guess: measured. Forming the Gram matrix squares the condition
# number, so a singular value that is truly zero comes back at roughly
# sqrt(eps) relative rather than at zero. Across the degenerate structures
# Bramoulle, Djebbari and Fortin name -- equal-sized groups, self-included
# group means, two group sizes under fixed effects, stars, complete
# bipartite graphs -- the largest such spurious value measured was 1.97e-8,
# on two equal groups of 8. Across identified structures the smallest
# genuine deciding value measured was 4.0e-3. The two populations are
# separated by a factor of 2.0e5, and this tolerance sits 50x above the
# noise and 4000x below the signal.
#
# An earlier default of 1e-8 sat *below* that measured noise floor and
# reported two equal groups of 8 as identified, contradicting their
# Proposition 2. The margin is the point of the constant, so change it only
# against a rerun of that measurement.
const DEFAULT_RANK_TOL = 1e-6

# Gram matrix of a list of matrices under the trace inner product,
# ⟨A, B⟩ = tr(A'B) = sum(A .* B) = dot(vec(A), vec(B)).
#
# This is how the rank of [vec(M₁) … vec(M_m)] is obtained without ever
# forming that n² × m matrix. For a network of 3,000 nodes the explicit
# version is 9 million rows and 288 MB in Float64; the Gram matrix is m × m.
#
# The cost is that forming the Gram squares the condition number, so the
# singular values it returns are the true ones squared and a borderline
# case is harder to call. That is measured rather than assumed: the tests
# compare this against a dense SVD of the explicit matrix and report the
# agreement.
function _vec_gram(mats::AbstractVector)
    m = length(mats)
    Gm = zeros(Float64, m, m)
    @inbounds for a in 1:m, b in a:m
        v = dot(mats[a], mats[b])
        Gm[a, b] = v
        Gm[b, a] = v
    end
    return Gm
end

# Singular values of [vec(M₁) … vec(M_m)], via the Gram matrix above.
function _vec_svals(mats::AbstractVector)
    Gm = _vec_gram(mats)
    λ = eigvals(Symmetric(Gm))              # ascending, PSD up to rounding
    return sqrt.(max.(0.0, reverse(λ)))     # descending singular values
end

# Rank of [vec(M₁) … vec(M_m)] at a relative tolerance, plus the relative
# size of the singular value that decided the call.
#
# `required` is the rank being tested for, and the gap reported is
# sv[required]/sv[1] -- the smallest singular value that had to clear the
# bar. Below the tolerance it is the spurious value that failed; above it,
# the margin by which the design passed. One number, either way, and it is
# the number the tolerance was calibrated against.
function _vec_rank(mats::AbstractVector; tol::Float64 = DEFAULT_RANK_TOL,
                   required::Int = length(mats))
    sv = _vec_svals(mats)
    s1 = isempty(sv) ? 0.0 : sv[1]
    s1 <= 0 && return (0, sv, 0.0)
    r = count(>(tol * s1), sv)
    gap = required <= length(sv) ? sv[required] / s1 : 0.0
    return (r, sv, gap)
end

# Diameter of the network's underlying undirected graph by breadth-first
# search from every node. `connected` is false when some pair is unreachable,
# in which case the value returned is the largest finite distance found.
function _network_diameter(G::AbstractSparseMatrix)
    n = size(G, 1)
    P = G .!= 0
    P = P .| permutedims(P)                 # reachability ignores direction
    rows = rowvals(P)
    diam = 0
    connected = true
    dist = Vector{Int}(undef, n)
    queue = Vector{Int}(undef, n)
    for s in 1:n
        fill!(dist, -1)
        dist[s] = 0
        queue[1] = s
        head = 1; tail = 1
        while head <= tail
            u = queue[head]; head += 1
            for idx in nzrange(P, u)
                v = rows[idx]
                if dist[v] < 0
                    dist[v] = dist[u] + 1
                    tail += 1
                    queue[tail] = v
                end
            end
        end
        any(<(0), dist) && (connected = false)
        d = maximum(dist)
        d > diam && (diam = d)
    end
    return (diam, connected)
end

# ═══════════════════════════════════════════════════════════════════════
# Identification
# ═══════════════════════════════════════════════════════════════════════

"""
    PeerIdentification

Verdict on whether the endogenous and contextual peer effects are separately
identified on a given network, as returned by [`peer_identification`](@ref).

# Fields
- `identified`: the verdict. See the caveat under `isolates` below.
- `rank`, `required`: the rank of `[vec(I) vec(G) vec(G²)]`, and the rank
  Bramoullé, Djebbari and Fortin require — 3 without fixed effects
  (their Proposition 1), 4 with them, where the matrix is extended by
  `vec(G³)` (their Proposition 4).
- `svals`: the singular values of that matrix, descending.
- `gap`: `svals[required] / svals[1]` — the relative size of the singular
  value that decided the call. A `gap` of `1e-14` is a degenerate network;
  a `gap` of `0.3` is a comfortable one; something like `1e-7` means the
  verdict is a judgement about tolerance and should not be leaned on.
- `fixed_effects`: which proposition was applied.
- `isolates`: units with no links. Proposition 1's identification half
  holds for an arbitrary `G`, but its *non*-identification half is stated
  only for networks without isolates, so when `isolates > 0` and the rank
  is short, `identified` is `false` but `conclusive` is `false` too.
- `conclusive`: whether the verdict rests on the published result or on its
  identification half alone.
- `diameter`, `connected`: diameter of the underlying undirected graph, and
  whether it is connected. Their Corollary 1 says a diameter of 3 or more
  is sufficient for identification under network fixed effects. `-1` when
  not computed.
- `instrument_rank`, `n_instruments`: rank of the realised instrument block
  `[G²X]` (and `G³X` under fixed effects) in the sample. **This is a
  different question** — it is about whether the instruments have power
  here, not about whether the model is identified in population — and it is
  reported separately for that reason. `-1` when no `X` was supplied.
"""
struct PeerIdentification
    identified::Bool
    conclusive::Bool
    rank::Int
    required::Int
    svals::Vector{Float64}
    gap::Float64
    fixed_effects::Bool
    isolates::Int
    diameter::Int
    connected::Bool
    instrument_rank::Int
    n_instruments::Int
    n::Int
end

"""
    peer_identification(G; fixed_effects = false, X = nothing,
                        tol = 1e-6, compute_diameter = size(G, 1) <= 1000)

Decide whether the endogenous peer effect `β` and the contextual effect `δ`
in

```
y = αι + βGy + γx + δGx + ε
```

are separately identified on the network `G`, by Bramoullé, Djebbari and
Fortin's (2009) rank condition.

Their Proposition 1: social effects are identified if `I`, `G` and `G²` are
linearly independent; if those are linearly dependent and no individual is
isolated, social effects are **not** identified. Their footnote 17 gives the
check used here, in their words: vectorize each matrix, stack the columns,
and verify whether the matrix formed by concatenating these stacked vectors
has rank three.

With `fixed_effects = true` the model is their eq. (10), the structural
equation after the within-local transformation `(I - G)` that sweeps out a
network-level unobservable. Proposition 4 then requires `I, G, G², G³` to be
linearly independent — strictly more demanding, because, in their words,
information has been lost to take into account the presence of correlated
effects.

# Why this is not a formality

The condition fails on the structures applied work uses most. Interaction in
equal-sized groups gives rank 2, not 3. With the group mean taken over
everyone including oneself it gives rank 2 whatever the sizes are, since
then `G² = G` exactly. Under fixed effects, two distinct group sizes are not
enough and three are needed; star and complete bipartite networks give
`G³ = G` and fail. In every one of those cases a peer-effects regression
still runs and still prints two coefficients.

# The tolerance

`tol` is a relative tolerance on the singular values, defaulting to `1e-6`.
That default is measured rather than conventional: across the degenerate
structures the paper names the largest spurious singular value observed was
`1.97e-8`, and across identified structures the smallest genuine one was
`4.0e-3`. The default sits between them with two orders of magnitude of
margin on each side. `gap` reports where a given network fell, so a verdict
that rests on the tolerance is visible rather than implied.

# The instrument rank is a different question

Pass `X` to also get the rank of the realised instrument block. The rank
condition above is about the population model and does not involve `X` at
all. Instrument rank is about whether this sample has the variation to
exploit it. Both can be informative; only the first is the theorem.

# Examples
```julia
peer_identification(group_weights([5, 5, 5]))     # rank 2, not identified
peer_identification(group_weights([4, 5, 6]))     # rank 3, identified
peer_identification(group_weights([4, 5, 6]); fixed_effects = true)  # rank 4
```

See also [`fit_peer_effects`](@ref), [`group_weights`](@ref).
"""
function peer_identification(G::AbstractMatrix;
                             fixed_effects::Bool = false,
                             X::Union{Nothing, AbstractMatrix} = nothing,
                             tol::Float64 = DEFAULT_RANK_TOL,
                             compute_diameter::Bool = size(G, 1) <= 1000)
    n, m = size(G)
    n == m || throw(DimensionMismatch(
        "peer_identification: `G` must be square, got $(n)×$(m)."))
    Gs = sparse(Float64.(G))

    G2 = Gs * Gs
    Id = sparse(1.0I, n, n)
    mats = fixed_effects ? Any[Id, Gs, G2, G2 * Gs] : Any[Id, Gs, G2]
    required = fixed_effects ? 4 : 3
    rank_vec, svals, gap = _vec_rank(mats; tol = tol, required = required)

    isolates = count(iszero, vec(sum(abs, Gs; dims = 2)))
    identified = rank_vec >= required
    # Proposition 1's non-identification half is stated for networks with no
    # isolated individual. Short rank with isolates present is therefore not
    # a published negative result, and saying so is cheaper than overclaiming.
    conclusive = identified || isolates == 0

    diam, connected = compute_diameter ? _network_diameter(Gs) : (-1, true)

    irank, ninst = -1, -1
    if X !== nothing
        size(X, 1) == n || throw(DimensionMismatch(
            "peer_identification: `X` has $(size(X, 1)) rows but `G` is $(n)×$(n)."))
        Xf = Matrix{Float64}(X)
        Z = fixed_effects ? hcat(G2 * Xf, G2 * (Gs * Xf)) : Matrix(G2 * Xf)
        if fixed_effects
            M = Id - Gs
            Z = Matrix(M * Z)
        end
        ninst = size(Z, 2)
        irank = ninst == 0 ? 0 : rank(Z)
    end

    return PeerIdentification(identified, conclusive, rank_vec, required,
                              svals, gap, fixed_effects, isolates,
                              diam, connected, irank, ninst, n)
end

function Base.show(io::IO, p::PeerIdentification)
    prop = p.fixed_effects ? "Proposition 4 (network fixed effects)" :
                             "Proposition 1 (no correlated effects)"
    verdict = p.identified ? "IDENTIFIED" : "NOT IDENTIFIED"
    println(io, "PeerIdentification: ", verdict)
    println(io)
    println(io, "  Bramoulle, Djebbari & Fortin (2009), ", prop)
    mats = p.fixed_effects ? "[vec(I) vec(G) vec(G2) vec(G3)]" :
                             "[vec(I) vec(G) vec(G2)]"
    @printf(io, "  rank %s = %d, required %d\n", mats, p.rank, p.required)
    @printf(io, "  deciding singular value, relative: %.3e\n", p.gap)
    @printf(io, "  units %d, isolates %d\n", p.n, p.isolates)
    if p.diameter >= 0
        @printf(io, "  diameter %d%s\n", p.diameter,
                p.connected ? "" : " (on the largest reachable set; graph is disconnected)")
    end
    if p.instrument_rank >= 0
        @printf(io, "  sample instrument rank %d of %d columns\n",
                p.instrument_rank, p.n_instruments)
    end
    println(io)
    if !p.identified
        println(io, "  The endogenous and contextual effects cannot be told apart")
        println(io, "  on this network. This is Manski's reflection problem, and it")
        println(io, "  is a property of the interaction structure, not of the sample")
        println(io, "  size -- more data does not fix it.")
        if !p.conclusive
            println(io)
            println(io, "  Note: the network has isolates, and the published")
            println(io, "  non-identification result assumes none. Read this as a")
            println(io, "  failure to establish identification rather than as a")
            println(io, "  proof of its absence.")
        end
    elseif p.gap < 100 * DEFAULT_RANK_TOL
        println(io, "  Identified, but only just: the deciding singular value is")
        println(io, "  within two orders of magnitude of the tolerance, where no")
        println(io, "  degenerate structure and no identified one was measured")
        println(io, "  when the tolerance was set. The verdict turns on `tol`")
        println(io, "  rather than on the network. Treat as near-degenerate.")
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════
# Peer effects
# ═══════════════════════════════════════════════════════════════════════

"""
    PeerEffectsFit

Estimated linear-in-means peer-effects model, as returned by
[`fit_peer_effects`](@ref).

# Fields
- `beta`: the **endogenous** effect — the response to peers' outcomes.
- `gamma`, `delta`: own characteristics, and the **contextual** effect of
  peers' characteristics. Aligned with `varnames`.
- `alpha`: the intercept, or `nothing` under fixed effects, where the
  within-local transformation sweeps it out.
- `coefs`, `ses`, `names`: the full coefficient vector and its standard
  errors, in the order the table prints.
- `vcov_kind`: `:robust`, `:homoskedastic` or `:transform`.
- `sargan`, `sargan_df`, `sargan_p`: over-identification test. `df` is the
  *rank* of the instrument block less the number of parameters, not its
  column count: a redundant instrument column supplies no restriction, and
  on rigid group structures there is often one. `df` is zero — and the test
  uninformative — when the model is exactly identified.
- `social_multiplier`: `γₖβ + δₖ` for each characteristic. Bramoullé,
  Djebbari and Fortin maintain `γβ + δ ≠ 0` throughout; when it is zero the
  social effects cancel out of the reduced form and nothing is identified,
  so a value near zero undercuts the estimates regardless of their standard
  errors. With several characteristics it must hold for at least one.
- `first_stage_F`: partial F of the excluded instruments on `Gy`, after the
  included regressors are partialled out. The identification condition is a
  population statement; this is whether *this* network has the variation to
  exploit it. It collapses as the network gets denser, because in a dense
  graph a unit's peers' peers are nearly everybody and `G²x` flattens
  towards a constant. Measured on Erdős–Rényi graphs at `n = 300`, the
  median falls from 36.5 at average degree 4.6 to 4.4 at average degree 60,
  while the spread of `β̂` rises twenty-one-fold.
- `ident`: the [`PeerIdentification`](@ref) computed before fitting.
- `residuals`, `fitted`, `n`.
"""
struct PeerEffectsFit{T<:AbstractFloat} <: StatsAPI.RegressionModel
    beta::T
    gamma::Vector{T}
    delta::Vector{T}
    alpha::Union{T, Nothing}
    coefs::Vector{T}
    ses::Vector{T}
    names::Vector{String}
    vcovmat::Matrix{T}
    vcov_kind::Symbol
    sargan::T
    sargan_df::Int
    sargan_p::T
    social_multiplier::Vector{T}
    first_stage_F::T
    ident::PeerIdentification
    fixed_effects::Bool
    resid::Vector{T}
    fitted::Vector{T}
    sigma2::T
    varnames::Vector{String}
    n::Int
end

coef(f::PeerEffectsFit) = f.coefs
vcov(f::PeerEffectsFit) = f.vcovmat
stderror(f::PeerEffectsFit) = f.ses
coefnames(f::PeerEffectsFit) = f.names
nobs(f::PeerEffectsFit) = f.n
residuals(f::PeerEffectsFit) = f.resid
dof_residual(f::PeerEffectsFit) = f.n - length(f.coefs)

"""
    fit_peer_effects(y, X, G; fixed_effects = false, vcov = :robust,
                     instrument_order = 2, varnames = nothing,
                     tol = 1e-6, check = true)

Estimate the linear-in-means peer-effects model

```
y = αι + βGy + γx + δGx + ε
```

by two-stage least squares, using Bramoullé, Djebbari and Fortin's (2009)
instruments, after checking that the model is identified at all.

`β` is the endogenous effect and `δ` the contextual effect. `Gy` is
endogenous by construction — `i` affects `j` and `j` affects `i` — and the
instruments are the characteristics of peers' peers, `G²x`, which enter the
reduced form but not the structural equation.

# The identification check is not optional

[`peer_identification`](@ref) runs first. If the network fails the rank condition
the fit **throws**, because what it would otherwise return is two
coefficients that are not separately estimable: the regression succeeds, the
standard errors look ordinary, and the split between `β` and `δ` is
arbitrary. Pass `check = false` to override, which is occasionally
legitimate — you may want the point estimates on a near-degenerate network
while knowing exactly what they are worth — and never silent.

It cannot rescue an *exactly* degenerate network, and that is not a
limitation of the override. Bramoullé, Djebbari and Fortin prove in their
Appendix B that `I, G, G²` are linearly dependent if and only if `E(Gy|x)`
is perfectly collinear with `(ι, x, Gx)` — so when identification fails the
instrument block is genuinely collinear with the included regressors, the
design is singular, and no estimate exists to report. The fit then throws
from the design-rank guard instead, naming the instrument columns. Measured
on equal-sized groups, the residual of `E(Gy|x)` on `(ι, x, Gx)` is
`8.9e-16`.

# Keywords
- `fixed_effects`: apply the within-local transformation `(I - G)` to sweep
  out a network-level unobservable correlated with `x`, the model of their
  Section 3. Costs the intercept and tightens the rank condition from 3 to
  4.
- `vcov`: `:robust` (default) is the heteroskedasticity-robust 2SLS
  sandwich. Bramoullé, Djebbari and Fortin leave the error variance matrix
  unrestricted and explicitly do not impose homoskedasticity, so this is the
  default rather than an option. `:homoskedastic` is `σ̂²(X̃'PX̃)⁻¹`.
  `:transform` is for the fixed-effects case only and accounts for the
  correlation the transformation itself induces, `Var((I-G)ε) =
  σ²(I-G)(I-G)'`; it is the right answer when the underlying errors really
  are homoskedastic and the wrong one when they are not.
- `instrument_order`: `2` (default) uses `G²x`, which is the instrument set
  in their empirical section. `3` adds `G³x`, buying over-identifying
  restrictions — hence a usable Sargan test — at the cost of weaker
  instruments.
- `check`: see above.

# Examples
```julia
G = network_weights(A)
fit = fit_peer_effects(y, X, G)
fit.beta                      # endogenous peer effect
fit.ident.gap                 # how comfortable the identification was
```

# References
Bramoullé, Djebbari & Fortin (2009), eqs. (1), (10) and §4.

See also [`peer_identification`](@ref), [`network_moran`](@ref).
"""
function fit_peer_effects(y::AbstractVector,
                          X::AbstractMatrix,
                          G::AbstractMatrix;
                          fixed_effects::Bool = false,
                          vcov::Symbol = :robust,
                          instrument_order::Int = 2,
                          varnames::Union{Nothing, Vector{String}} = nothing,
                          tol::Float64 = DEFAULT_RANK_TOL,
                          check::Bool = true)
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch(
        "fit_peer_effects: `X` has $(size(X, 1)) rows but `y` has $n."))
    size(G, 1) == size(G, 2) == n || throw(DimensionMismatch(
        "fit_peer_effects: `G` is $(size(G, 1))×$(size(G, 2)) but `y` has $n."))
    instrument_order in (2, 3) || throw(ArgumentError(
        "fit_peer_effects: `instrument_order` must be 2 or 3, got $instrument_order."))
    vcov in (:robust, :homoskedastic, :transform) || throw(ArgumentError(
        "fit_peer_effects: `vcov` must be :robust, :homoskedastic or " *
        ":transform, got :$(vcov)."))
    (vcov === :transform && !fixed_effects) && throw(ArgumentError(
        "fit_peer_effects: `vcov = :transform` corrects for the correlation " *
        "induced by the within-local transformation, which is only applied " *
        "when `fixed_effects = true`. Use :robust or :homoskedastic."))

    T = Float64
    yv = Vector{T}(y)
    Xm = Matrix{T}(X)
    Gs = sparse(T.(G))
    K = size(Xm, 2)
    vnames = varnames === nothing ? ["x$i" for i in 1:K] : copy(varnames)
    length(vnames) == K || throw(DimensionMismatch(
        "fit_peer_effects: $(length(vnames)) names for $K characteristics."))

    ident = peer_identification(Gs; fixed_effects = fixed_effects, X = Xm,
                                tol = tol)
    if check && !ident.identified
        throw(ArgumentError(
            "fit_peer_effects: the peer effects are not identified on this " *
            "network. " * (fixed_effects ?
            "Bramoulle, Djebbari and Fortin's Proposition 4 requires I, G, " *
            "G^2 and G^3 to be linearly independent" :
            "Bramoulle, Djebbari and Fortin's Proposition 1 requires I, G " *
            "and G^2 to be linearly independent") *
            "; the rank here is $(ident.rank), not $(ident.required). The " *
            "endogenous and contextual effects cannot be told apart on this " *
            "structure, and fitting anyway would report a split between them " *
            "that the data does not determine. This is Manski's reflection " *
            "problem and more data does not fix it. If the point estimates " *
            "are wanted anyway, pass `check = false`."))
    end

    Gy  = Gs * yv
    GX  = Matrix(Gs * Xm)
    G2X = Matrix(Gs * GX)
    G3X = instrument_order == 3 ? Matrix(Gs * G2X) : zeros(T, n, 0)

    if fixed_effects
        # The within-local transformation of their eq. (10): subtract from
        # each unit the mean of its own peers. The intercept goes with it.
        M = v -> v .- Gs * v
        yt   = M(yv)
        Gyt  = M(Gy)
        Xt   = reduce(hcat, (M(@view Xm[:, j]) for j in 1:K))
        GXt  = reduce(hcat, (M(@view GX[:, j]) for j in 1:K))
        Zt   = reduce(hcat, (M(@view G2X[:, j]) for j in 1:K))
        if instrument_order == 3
            Zt = hcat(Zt, reduce(hcat, (M(@view G3X[:, j]) for j in 1:K)))
        end
        regressors = hcat(Gyt, Xt, GXt)
        included   = hcat(Xt, GXt)
        names = vcat(["Gy"], vnames, ["G*" .* v for v in vnames])
        yfit = yt
    else
        iota = ones(T, n)
        regressors = hcat(iota, Gy, Xm, GX)
        included   = hcat(iota, Xm, GX)
        Zt = instrument_order == 3 ? hcat(G2X, G3X) : G2X
        names = vcat(["(intercept)", "Gy"], vnames, ["G*" .* v for v in vnames])
        yfit = yv
    end

    check_design_rank(regressors, names, "fit_peer_effects";
        context = "On a network the spatial lag of a characteristic can " *
                  "coincide with the characteristic itself -- in groups " *
                  "where everyone shares a trait, `G*x` equals `x`. Drop " *
                  "the offending column.")

    S = hcat(included, Zt)

    # The instrument block is allowed to be rank deficient, and on rigid
    # structures it often is: with interaction in two groups, `G^2 x` lies
    # in the span of `x`, `G x` and a group indicator, so one of its columns
    # is redundant even though the network passes the rank condition. A
    # redundant instrument adds no information but invalidates nothing --
    # the projector onto the column span of S exists whatever its rank, and
    # only the (S'S)^{-1} form of it fails. So the projection is taken from
    # an orthonormal basis of that span rather than from a normal-equation
    # inverse, and the rank is carried forward to the degrees of freedom
    # below, where counting columns instead would overstate them.
    Sf = svd(S)
    s_rank = count(>(sqrt(eps(T)) * first(Sf.S)), Sf.S)
    if s_rank < size(regressors, 2)
        throw(ArgumentError(
            "fit_peer_effects: the instruments span $(s_rank) dimensions " *
            "but there are $(size(regressors, 2)) parameters to estimate, " *
            "so the order condition fails and no 2SLS estimate exists. The " *
            "instrument block `G^2 x` has to add variation that `x` and " *
            "`G x` do not already carry; when it does not, the network is " *
            "at or near the degenerate case `peer_identification` reports " *
            "on. Bramoulle, Djebbari and Fortin's Appendix B is why the " *
            "two coincide."))
    end
    Ur = @view Sf.U[:, 1:s_rank]
    project = v -> Ur * (Ur' * v)

    # 2SLS: theta = (Xr' P Xr)^{-1} Xr' P y, with P the orthogonal projector
    # onto the instrument span.
    PXr = project(regressors)
    A = regressors' * PXr
    b = PXr' * yfit
    theta = Symmetric(A) \ b

    u = yfit .- regressors * theta
    dfree = max(1, n - length(theta))
    sigma2 = dot(u, u) / dfree

    Vmat = _tsls_vcov(regressors, PXr, u, sigma2, vcov, Gs, n)
    ses = sqrt.(max.(zero(T), diag(Vmat)))

    # Sargan over-identification. With one excluded instrument per endogenous
    # regressor there is nothing to test and df is zero.
    sdf = s_rank - size(regressors, 2)
    s_hat = dot(u, u) / n
    sstat = sdf > 0 ? dot(u, project(u)) / s_hat : zero(T)
    spval = sdf > 0 ? T(_chi2_sf(Float64(sstat), sdf)) : one(T)

    # First-stage partial F of the excluded instruments on the endogenous
    # regressor, after partialling out the included ones. Cheap, and the
    # thing that actually fails on a dense network.
    endog = regressors[:, fixed_effects ? 1 : 2]
    r0 = endog .- included * (qr(included) \ endog)
    r1 = endog .- project(endog)
    q = s_rank - size(included, 2)
    dfF = n - s_rank
    ssr0 = dot(r0, r0); ssr1 = dot(r1, r1)
    fstat = (q > 0 && dfF > 0 && ssr1 > 0) ?
            T(((ssr0 - ssr1) / q) / (ssr1 / dfF)) : T(NaN)

    ipos = fixed_effects ? 1 : 2
    beta = theta[ipos]
    alpha = fixed_effects ? nothing : theta[1]
    gamma = theta[(ipos + 1):(ipos + K)]
    delta = theta[(ipos + K + 1):(ipos + 2K)]
    socmult = gamma .* beta .+ delta

    return PeerEffectsFit{T}(beta, gamma, delta, alpha, theta, ses, names,
                             Vmat, vcov, sstat, sdf, spval, socmult, fstat, ident,
                             fixed_effects, u, regressors * theta, sigma2,
                             vnames, n)
end

# Covariance of the 2SLS estimator. `PXr` is the projection of the
# regressors onto the instrument span, which is the right-hand matrix in
# every one of the three sandwiches.
function _tsls_vcov(Xr::Matrix{T}, PXr::Matrix{T}, u::Vector{T},
                    sigma2::T, kind::Symbol, G::AbstractSparseMatrix,
                    n::Int) where {T<:AbstractFloat}
    bread = inv(Symmetric(Xr' * PXr))
    if kind === :homoskedastic
        return Symmetric(bread .* sigma2) |> Matrix
    elseif kind === :robust
        meat = (PXr .* u)' * (PXr .* u)
        return Matrix(Symmetric(bread * meat * bread))
    else  # :transform
        # Var((I-G)eps) = s2 (I-G)(I-G)'. Estimate s2 by moments: the
        # residual sum of squares divided by tr((I-G)(I-G)'), since
        # E[u'u] = s2 tr((I-G)(I-G)').
        Id = sparse(one(T) * I, n, n)
        M = Id - G
        trMM = sum(abs2, M)
        s2 = dot(u, u) / trMM
        MP = M' * PXr
        meat = s2 .* (MP' * MP)
        return Matrix(Symmetric(bread * meat * bread))
    end
end

function Base.show(io::IO, f::PeerEffectsFit)
    kind = f.fixed_effects ? "with network fixed effects" : "no correlated effects"
    println(io, "PeerEffectsFit: linear-in-means peer effects (", kind, ")")
    println(io, "  n = ", f.n, ", 2SLS, vcov = :", f.vcov_kind)
    println(io)
    _print_coef_table(io, f.names, f.coefs, f.ses)
    println(io)
    @printf(io, "  endogenous effect  beta = %+.4f  (se %.4f)\n",
            f.beta, f.ses[f.fixed_effects ? 1 : 2])
    if f.sargan_df > 0
        @printf(io, "  Sargan  chi2(%d) = %.3f,  p = %.4f\n",
                f.sargan_df, f.sargan, f.sargan_p)
    else
        println(io, "  Sargan: not available -- the model is exactly identified")
    end
    @printf(io, "  first-stage F on the excluded instruments = %.2f
", f.first_stage_F)
    if isfinite(f.first_stage_F) && f.first_stage_F < 10
        println(io)
        println(io, "  WARNING: the instruments are weak. `G^2 x` adds little")
        println(io, "  to `x` and `G x` here, which is what a dense network does:")
        println(io, "  a unit's peers' peers become nearly everybody and the")
        println(io, "  instrument flattens towards a constant. The identification")
        println(io, "  condition can hold in population while this sample has")
        println(io, "  almost nothing to exploit. Measured on random graphs, the")
        println(io, "  spread of the endogenous effect rises twenty-one-fold as")
        println(io, "  the median F falls from 36 to 4, and the estimate biases")
        println(io, "  toward zero -- while the rejection rate stays near nominal,")
        println(io, "  so this will not show up as a significant test.")
    end
    sm = maximum(abs, f.social_multiplier)
    @printf(io, "  max |gamma*beta + delta| = %.4e\n", sm)
    if sm < 1e-3
        println(io)
        println(io, "  WARNING: gamma*beta + delta is close to zero for every")
        println(io, "  characteristic. Bramoulle, Djebbari and Fortin maintain")
        println(io, "  this quantity is non-zero throughout; when it vanishes the")
        println(io, "  social effects cancel out of the reduced form and are")
        println(io, "  absent from it, so these estimates rest on nothing the")
        println(io, "  data can see, whatever their standard errors say.")
    end
    if !f.ident.identified
        println(io)
        println(io, "  WARNING: fitted with `check = false` on a network that")
        println(io, "  fails the rank condition. The split between the")
        println(io, "  endogenous and contextual effects is not determined.")
    end
    return nothing
end

"""
    network_moran(fit, W; kwargs...) -> MoranResult

Moran's I of a peer-effects fit's residuals against a weight matrix `W`,
which need not be the interaction matrix the model was fitted on.

The useful case is passing a **different** `W` from the `G` used to fit: it
asks whether dependence remains along some channel the model did not
include — geography, say, after a social network has been controlled for.
Passing the same `G` back asks whether the endogenous effect has absorbed
the dependence it was meant to.

Keyword arguments are forwarded to `moran_global`, including `inference`,
`n_perm` and `rng`.
"""
function network_moran(fit::PeerEffectsFit, W::AbstractMatrix; kwargs...)
    size(W, 1) == fit.n || throw(DimensionMismatch(
        "network_moran: `W` is $(size(W, 1))×$(size(W, 2)) but the fit has " *
        "$(fit.n) observations."))
    return moran_global(residuals(fit), sparse(Float64.(W)); kwargs...)
end

# ═══════════════════════════════════════════════════════════════════════
# Multiplex SAR
# ═══════════════════════════════════════════════════════════════════════

"""
    MultiplexSAR

Spatial autoregression with several interaction channels at once, as
returned by [`fit_multiplex_sar`](@ref).

# Fields
- `rho`, `rho_se`: one autoregressive parameter per channel, with standard
  errors from the analytic information matrix.
- `beta`, `beta_se`, `varnames`: the regression coefficients.
- `sigma2`, `loglik`, `n`.
- `channel_names`.
- `channel_rank`, `channel_svals`: rank of `[vec(W₁) … vec(W_J)]`. Channels
  that are collinear **as matrices** cannot have their `ρ`s told apart, for
  the same reason `I, G, G²` must be independent in a peer-effects model.
- `converged`, `iterations`: coordinate ascent diagnostics.
"""
struct MultiplexSAR{T<:AbstractFloat} <: StatsAPI.RegressionModel
    rho::Vector{T}
    rho_se::Vector{T}
    beta::Vector{T}
    beta_se::Vector{T}
    sigma2::T
    loglik::T
    vcovmat::Matrix{T}
    channel_names::Vector{String}
    varnames::Vector{String}
    channel_rank::Int
    channel_svals::Vector{Float64}
    converged::Bool
    iterations::Int
    resid::Vector{T}
    n::Int
end

coef(m::MultiplexSAR) = vcat(m.beta, m.rho)
vcov(m::MultiplexSAR) = m.vcovmat
stderror(m::MultiplexSAR) = vcat(m.beta_se, m.rho_se)
coefnames(m::MultiplexSAR) = vcat(m.varnames, m.channel_names)
nobs(m::MultiplexSAR) = m.n
residuals(m::MultiplexSAR) = m.resid

"""
    fit_multiplex_sar(y, X, Ws; channel_names = nothing, varnames = nothing,
                      rho_bound = 0.995, tol = 1e-9, maxiter = 200)

Fit

```
y = Σⱼ ρⱼ Wⱼ y + Xβ + ε
```

by concentrated maximum likelihood, where the `Wⱼ` are different interaction
channels over the same units — trade and migration and shared ownership, say
— rather than different lags of one network.

Each channel gets its own `ρⱼ`, so the fit answers which channel carries the
dependence, which a single pooled `W` cannot. An intercept is added; do not
include one in `X`.

# How it is fitted

`β` and `σ²` are concentrated out at each trial `ρ`, leaving a
`J`-dimensional profile maximised by coordinate ascent, each coordinate by
the same Brent search the single-channel fit uses. `log|I - Σρⱼ Wⱼ|` is
computed exactly by sparse LU at every evaluation: the Chebyshev
approximation available for one `W` rests on precomputed traces of its
powers and does not carry over to an arbitrary combination, so it is not
offered here rather than offered and quietly wrong.

Standard errors come from the analytic information matrix, which generalises
the single-channel one; with one channel this function reproduces
`fit_spatial(SAR, ...)` to machine precision, and that is pinned as a test.

# Stationarity

The parameter space is `Σⱼ |ρⱼ| < rho_bound`. With each `Wⱼ` row-standardised
this is sufficient for `I - Σρⱼ Wⱼ` to be invertible, since the spectral
radius of the combination is then at most `Σ|ρⱼ|`. It is sufficient and not
necessary — some stationary points lie outside it — so a fit that lands
against the boundary is reported with `converged = false` rather than
silently accepted.

# A caveat worth reading

Coordinate ascent is exact along each axis but can stall on a ridge, which
is what a pair of strongly overlapping channels produces. `channel_rank`
reports whether the channels are distinguishable as matrices at all, and
convergence from several starting points is cheap insurance. The tests
measure the gap against a dense grid search.

# References
Lee, L. F. & Liu, X. (2010). Efficient GMM estimation of high order spatial
autoregressive models with autoregressive disturbances. *Econometric
Theory*, 26(1), 187–230.
"""
function fit_multiplex_sar(y::AbstractVector,
                           X::AbstractMatrix,
                           Ws::AbstractVector;
                           channel_names::Union{Nothing, Vector{String}} = nothing,
                           varnames::Union{Nothing, Vector{String}} = nothing,
                           rho_bound::Float64 = 0.995,
                           tol::Float64 = 1e-9,
                           maxiter::Int = 200)
    T = Float64
    n = length(y)
    J = length(Ws)
    J >= 1 || throw(ArgumentError("fit_multiplex_sar: no channels supplied."))
    size(X, 1) == n || throw(DimensionMismatch(
        "fit_multiplex_sar: `X` has $(size(X, 1)) rows but `y` has $n."))
    for (j, W) in enumerate(Ws)
        size(W) == (n, n) || throw(DimensionMismatch(
            "fit_multiplex_sar: channel $j is $(size(W, 1))×$(size(W, 2)) " *
            "but `y` has $n observations."))
    end
    0 < rho_bound < 1 || throw(ArgumentError(
        "fit_multiplex_sar: `rho_bound` must lie in (0, 1), got $rho_bound."))

    Wl = [sparse(T.(W)) for W in Ws]
    yv = Vector{T}(y)
    K = size(X, 2)
    vnames = vcat(["(intercept)"],
                  varnames === nothing ? ["x$i" for i in 1:K] : copy(varnames))
    cnames = channel_names === nothing ? ["rho_$j" for j in 1:J] :
                                         copy(channel_names)
    length(cnames) == J || throw(DimensionMismatch(
        "fit_multiplex_sar: $(length(cnames)) channel names for $J channels."))
    Xa = hcat(ones(T, n), Matrix{T}(X))
    check_design_rank(Xa, vnames, "fit_multiplex_sar";
        context = "An intercept is added, so `X` must not carry one.")

    # Channels that are collinear as matrices cannot have their rhos told
    # apart -- the same algebra as the peer-effects rank condition.
    crank, csv, _ = _vec_rank(Any[W for W in Wl])
    if crank < J
        throw(ArgumentError(
            "fit_multiplex_sar: the $J channels span only $crank dimensions " *
            "as matrices, so their autoregressive parameters are not " *
            "separately identified -- any shift along the dependency can be " *
            "offset by the others with an identical fit. Drop the redundant " *
            "channel(s), or combine them into one."))
    end

    Id = sparse(one(T) * I, n, n)
    Wy = [W * yv for W in Wl]
    qrX = qr(Xa)

    function profile(rho)
        sum(abs, rho) >= rho_bound && return T(-Inf)
        e = copy(yv)
        @inbounds for j in 1:J
            e .-= rho[j] .* Wy[j]
        end
        A = Id - sum(rho[j] .* Wl[j] for j in 1:J)
        ld = try
            first(logabsdet(lu(A)))
        catch
            return T(-Inf)
        end
        isfinite(ld) || return T(-Inf)
        u = e .- Xa * (qrX \ e)
        s2 = dot(u, u) / n
        s2 <= 0 && return T(-Inf)
        return -T(0.5) * n * (log(T(2π)) + one(T)) + ld - T(0.5) * n * log(s2)
    end

    rho = zeros(T, J)
    ll = profile(rho)
    converged = false
    iters = 0
    for it in 1:maxiter
        iters = it
        prev = ll
        for j in 1:J
            slack = rho_bound - sum(abs, rho) + abs(rho[j])
            lo = max(-slack + 1e-10, -rho_bound + 1e-10)
            hi = min(slack - 1e-10, rho_bound - 1e-10)
            hi <= lo && continue
            f = r -> (rho[j] = r; profile(rho))
            best = rho[j]
            rj, llj = brent_maximize(f, lo, hi; tol = T(1e-9))
            if llj >= ll
                rho[j] = rj
                ll = llj
            else
                rho[j] = best
            end
        end
        if abs(ll - prev) < tol * (abs(prev) + 1.0)
            converged = true
            break
        end
    end
    # A fit pinned against the stationarity boundary has not converged to an
    # interior maximum, whatever the likelihood change says.
    sum(abs, rho) > rho_bound - 1e-6 && (converged = false)

    e = copy(yv)
    for j in 1:J
        e .-= rho[j] .* Wy[j]
    end
    beta = qrX \ e
    u = e .- Xa * beta
    s2 = dot(u, u) / n

    Vmat = _multiplex_vcov(Xa, Wl, beta, rho, s2, n)
    P = size(Xa, 2)
    allse = sqrt.(max.(zero(T), diag(Vmat)))

    return MultiplexSAR{T}(rho, allse[(P + 1):(P + J)], beta, allse[1:P],
                           s2, ll, Vmat, cnames, vnames, crank, csv,
                           converged, iters, u, n)
end

# Analytic information matrix for the multiplex SAR. This is the
# single-channel SAR information matrix of `compute_asymptotic_vcov`
# generalised to J channels: the rho block picks up a cross term
# tr(G_j G_k) + tr(G_j' G_k) for every pair rather than only on the
# diagonal. Forms A^{-1} densely, which is what makes the traces exact.
function _multiplex_vcov(Xa::Matrix{T}, Wl::Vector{<:AbstractSparseMatrix},
                         beta::Vector{T}, rho::Vector{T}, s2::T,
                         n::Int) where {T<:AbstractFloat}
    J = length(Wl)
    P = size(Xa, 2)
    Id = sparse(one(T) * I, n, n)
    A = Id - sum(rho[j] .* Wl[j] for j in 1:J)
    Ainv = Matrix(lu(A) \ Matrix{T}(I, n, n))
    Gm = [Matrix(Wl[j] * Ainv) for j in 1:J]
    Xb = Xa * beta
    GXb = [Gm[j] * Xb for j in 1:J]

    sz = P + J + 1
    Info = zeros(T, sz, sz)
    Info[1:P, 1:P] .= (Xa' * Xa) ./ s2
    for j in 1:J
        Info[1:P, P + j] .= (Xa' * GXb[j]) ./ s2
        Info[P + j, 1:P] .= Info[1:P, P + j]
        Info[P + j, sz] = tr(Gm[j]) / s2
        Info[sz, P + j] = Info[P + j, sz]
        for k in j:J
            v = dot(GXb[j], GXb[k]) / s2 + tr(Gm[j] * Gm[k]) +
                dot(Gm[j], Gm[k])
            Info[P + j, P + k] = v
            Info[P + k, P + j] = v
        end
    end
    Info[sz, sz] = n / (2 * s2^2)

    Vfull = inv(Symmetric(Info))
    return Matrix(Vfull[1:(P + J), 1:(P + J)])
end

function Base.show(io::IO, m::MultiplexSAR)
    println(io, "MultiplexSAR: ", length(m.rho), " interaction channels, n = ", m.n)
    println(io)
    _print_coef_table(io, m.varnames, m.beta, m.beta_se)
    println(io)
    println(io, "  Channel dependence:")
    _print_coef_table(io, m.channel_names, m.rho, m.rho_se)
    println(io)
    @printf(io, "  sigma2 = %.5f,  loglik = %.4f,  sum|rho| = %.4f\n",
            m.sigma2, m.loglik, sum(abs, m.rho))
    if !m.converged
        println(io)
        println(io, "  WARNING: coordinate ascent did not converge to an interior")
        println(io, "  maximum. Either the iteration limit was reached or the fit")
        println(io, "  is pinned against the stationarity boundary sum|rho| < 1.")
        println(io, "  The standard errors assume an interior maximum and should")
        println(io, "  not be read.")
    end
    return nothing
end

end # module
