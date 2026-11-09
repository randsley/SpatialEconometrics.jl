# Region and country geometry from OpenStreetMap, for build_europe.jl.
#
# Why OpenStreetMap and not GISCO: GISCO's NUTS boundaries are licensed for
# non-commercial use only, and anything derived from them inherits that. OSM is
# under the Open Database License (ODbL 1.0), which permits commercial use; a
# database derived from it (the neighbour lists and centroids written here)
# must itself be offered under the ODbL and credit "© OpenStreetMap
# contributors". Areas do not come from OSM at all: coastal regions in OSM
# extend into territorial waters, so land areas are read from Eurostat.
#
# Contiguity needs no geometry engine. OSM stores a boundary between two
# regions once, as a way that both regions' relations list as a member, so
# two regions share an edge (rook) when their relations share a land way, and
# touch (queen) when their land ways share a node.

const OVERPASS = "https://overpass-api.de/api/interpreter"
const USER_AGENT = "SpatialEconometrics.jl data build (https://github.com/randsley/SpatialEconometrics.jl)"

# Boundaries at sea that OSM does not tag maritime=yes: Åland's sea border
# with Sweden. They join regions that have no land border, so they are dropped
# by name rather than by tag. Found by comparing against an independent
# vertex-matched contiguity; every other OSM land link agrees with it.
const SEA_LINKS = Set([("FI20", "SE11"), ("FI20", "SE12")])

# Lake Constance has no agreed border: Austria, Germany and Switzerland never
# divided it, so OSM leaves the lake outside all three. Eurostat's NUTS
# boundaries split it by convention, and the links that split creates are
# added here so the weights match the usual NUTS-based ones. Rook adds only
# the shared edge; the queen links include the points where three meet.
const LAKE_ROOK = Set([("CH05", "DE14")])
const LAKE_QUEEN = Set([("AT34", "DE14"), ("CH05", "DE14"), ("CH05", "DE27")])

"""
    overpass(query, name)

Run an Overpass query, cached under `name`. The public server sheds load with
429/502/504 when busy, so failures are retried with exponential backoff.

The request goes through the `curl` executable, not Downloads.jl: Overpass
sends nothing while a query waits in its queue, and Downloads.jl aborts any
transfer that is silent for 20 seconds, so every query on a busy server
failed that way before it could start.
"""
function overpass(query, name)
    path = joinpath(CACHE, name)
    if !isfile(path)
        qfile = path * ".query"
        write(qfile, query)
        part = path * ".part"
        wait = 30
        for attempt in 1:7
            @info "overpass $name (attempt $attempt)"
            status = try
                readchomp(`curl -s --max-time 420 -A $USER_AGENT --data-urlencode data@$qfile
                           -o $part -w "%{http_code}" $OVERPASS`)
            catch err
                sprint(showerror, err)
            end
            if status == "200" && isfile(part) && startswith(lstrip(read(part, String)), "{")
                mv(part, path; force = true)
                break
            end
            @warn "overpass $name: status $status"
            attempt == 7 && error("overpass $name failed after 7 attempts")
            sleep(wait); wait = min(2wait, 600)
        end
        rm(part; force = true); rm(qfile; force = true)
    end
    push!(SOURCES, (name = name, url = OVERPASS, sha256 = bytes2hex(open(sha256, path)),
                    bytes = filesize(path)))
    return JSON3.read(read(path, String))
end

nuts_query(cc, digits) = """
    [out:json][timeout:300];
    ( relation["ref:nuts:$(digits - 2)"~"^$(cc)"];
      relation["ref:NUTS"~"^$(cc)[0-9A-Z]{$(digits - 2)}\$"];
      relation["ref:nuts"~"^$(cc)[0-9A-Z]{$(digits - 2)}\$"]; )->.r;
    .r out body;
    way(r.r);
    out body geom;
    """

struct OSMWay
    nodes::Vector{Int}
    lon::Vector{Float64}
    lat::Vector{Float64}
    sea::Bool
end

# The NUTS code a relation carries at `digits` characters, if any.
function nuts_code(tags, digits)
    for key in ("ref:nuts:$(digits - 2)", "ref:NUTS", "ref:nuts")
        v = get(tags, Symbol(key), nothing)
        v isa AbstractString && length(v) == digits && return (String(v), key == "ref:nuts:$(digits - 2)")
    end
    return nothing
end

# Several relations can carry one code (a statistical unit and an older
# administrative or historic one with the same boundary). Prefer statistical,
# then administrative, then anything but historic, then the dedicated key.
const BOUNDARY_RANK = Dict("statistical" => 0, "administrative" => 1, "census" => 2, "historic" => 9)

function parse_overpass(d, digits, wanted)
    ways = Dict{Int,OSMWay}()
    picks = Dict{String,Tuple}()
    for e in d.elements
        if e.type == "way"
            g = e.geometry
            tags = get(e, :tags, Dict())
            sea = get(tags, :maritime, "") == "yes" || get(tags, :natural, "") == "coastline" ||
                  get(tags, :boundary, "") == "maritime"
            ways[e.id] = OSMWay(collect(Int, e.nodes), [Float64(p.lon) for p in g],
                                [Float64(p.lat) for p in g], sea)
        elseif e.type == "relation"
            tags = get(e, :tags, Dict())
            hit = nuts_code(tags, digits)
            hit === nothing && continue
            code, dedicated = hit
            wanted(code) || continue
            rank = (get(BOUNDARY_RANK, String(get(tags, :boundary, "")), 5), dedicated ? 0 : 1)
            members = [(Int(m.ref), String(m.role)) for m in e.members if m.type == "way"]
            if !haskey(picks, code) || rank < picks[code][1]
                picks[code] = (rank, Int(e.id), String(get(tags, :boundary, "")), members)
            end
        end
    end
    return ways, picks
end

const MAX_GAP_M = 10_000.0

"""
    stitch(wayids, ways) -> (rings, gaps)

Join ways into closed rings by matching end nodes, reversing a way where
needed. Rings are returned as (lon, lat) vectors.

OSM boundaries are sometimes broken: a way is missing from the relation, or
two ways end on different nodes a metre apart. Chains left open are joined
end to end across the shortest gap, and each gap's width in metres is
returned in `gaps`, so the provenance can state it. A gap wider than
`MAX_GAP_M` is an error, not a repair.
"""
function stitch(wayids, ways)
    rings, chains = _chains(wayids, ways)
    gaps = Float64[]
    dist(a, b) = (p = laea(a...); q = laea(b...); hypot(p[1] - q[1], p[2] - q[2]))
    head(c) = (c[1][1], c[2][1]); tail(c) = (c[1][end], c[2][end])
    while !isempty(chains)
        c = popfirst!(chains)
        while true
            # Closing on itself competes with joining another chain.
            best, bi, brev = dist(tail(c), head(c)), 0, false
            for (i, o) in enumerate(chains)
                d1 = dist(tail(c), head(o)); d1 < best && ((best, bi, brev) = (d1, i, false))
                d2 = dist(tail(c), tail(o)); d2 < best && ((best, bi, brev) = (d2, i, true))
            end
            best > MAX_GAP_M && error("boundary gap of $(round(Int, best)) m exceeds $(MAX_GAP_M) m")
            push!(gaps, best)
            bi == 0 && (push!(c[1], c[1][1]); push!(c[2], c[2][1]); break)
            o = popat!(chains, bi)
            append!(c[1], brev ? reverse(o[1]) : o[1]); append!(c[2], brev ? reverse(o[2]) : o[2])
        end
        push!(rings, c)
    end
    return rings, gaps
end

function _chains(wayids, ways)
    pool = Set(w for w in wayids if haskey(ways, w) && length(ways[w].nodes) >= 2)
    byend = Dict{Int,Vector{Int}}()
    for w in pool, n in (first(ways[w].nodes), last(ways[w].nodes))
        push!(get!(Vector{Int}, byend, n), w)
    end
    rings = Tuple{Vector{Float64},Vector{Float64}}[]
    unclosed = Tuple{Vector{Float64},Vector{Float64}}[]
    while !isempty(pool)
        w = pop!(pool)
        nodes = copy(ways[w].nodes); lon = copy(ways[w].lon); lat = copy(ways[w].lat)
        while first(nodes) != last(nodes)
            nxt = findfirst(in(pool), get(byend, last(nodes), Int[]))
            nxt === nothing && break
            v = byend[last(nodes)][nxt]
            delete!(pool, v)
            wv = ways[v]
            fwd = first(wv.nodes) == last(nodes)
            ns, xs, ys = fwd ? (wv.nodes, wv.lon, wv.lat) :
                               (reverse(wv.nodes), reverse(wv.lon), reverse(wv.lat))
            append!(nodes, ns[2:end]); append!(lon, xs[2:end]); append!(lat, ys[2:end])
        end
        push!(first(nodes) == last(nodes) ? rings : unclosed, (lon, lat))
    end
    return rings, unclosed
end

# ETRS89 / LAEA Europe (EPSG:3035), ellipsoidal forward formulas from IOGP
# Guidance Note 7-2: metres on an equal-area projection, so areas and
# distances between centroids mean what they say across the whole continent.
const _A, _E2 = 6378137.0, 0.0066943800229
const _E = sqrt(_E2)
_q(φ) = (1 - _E2) * (sin(φ) / (1 - _E2 * sin(φ)^2) -
                     log((1 - _E * sin(φ)) / (1 + _E * sin(φ))) / (2_E))
const _QP = _q(π / 2)
const _φ0, _λ0 = deg2rad(52.0), deg2rad(10.0)
const _β0 = asin(_q(_φ0) / _QP)
const _RQ = _A * sqrt(_QP / 2)
const _D = _A * (cos(_φ0) / sqrt(1 - _E2 * sin(_φ0)^2)) / (_RQ * cos(_β0))

function laea(lon, lat)
    φ, λ = deg2rad(lat), deg2rad(lon)
    β = asin(_q(φ) / _QP)
    B = _RQ * sqrt(2 / (1 + sin(_β0) * sin(β) + cos(_β0) * cos(β) * cos(λ - _λ0)))
    return (4321000.0 + B * _D * cos(β) * sin(λ - _λ0),
            3210000.0 + (B / _D) * (cos(_β0) * sin(β) - sin(_β0) * cos(β) * cos(λ - _λ0)))
end

# Signed shoelace sums for one ring: (2A, 6A·cx, 6A·cy).
function shoelace(x, y)
    a = 0.0; sx = 0.0; sy = 0.0
    for i in 1:length(x)-1
        c = x[i] * y[i+1] - x[i+1] * y[i]
        a += c; sx += (x[i] + x[i+1]) * c; sy += (y[i] + y[i+1]) * c
    end
    return a, sx, sy
end

"""
Area-weighted centroid of outer rings less inner rings, in EPSG:3035 metres,
plus a planar longitude/latitude version for placing labels.
"""
function centroid(outer, inner)
    function acc(rings, s, f)
        A = 0.0; X = 0.0; Y = 0.0
        for (lon, lat) in rings
            x, y = f(lon, lat)
            a, sx, sy = shoelace(x, y)
            k = s * sign(a)
            A += k * a / 2; X += k * sx / 6; Y += k * sy / 6
        end
        return A, X, Y
    end
    proj(lon, lat) = (p = laea.(lon, lat); (first.(p), last.(p)))
    ident(lon, lat) = (lon, lat)
    A1, X1, Y1 = acc(outer, 1, proj);  A2, X2, Y2 = acc(inner, -1, proj)
    B1, U1, V1 = acc(outer, 1, ident); B2, U2, V2 = acc(inner, -1, ident)
    A, B = A1 + A2, B1 + B2
    return (x = (X1 + X2) / A, y = (Y1 + Y2) / A, lon = (U1 + U2) / B, lat = (V1 + V2) / B,
            footprint_km2 = A / 1e6)
end

"""
    osm_regions(countries, ids) -> NamedTuple

Geometry for the NUTS-2 regions `ids`, fetched one country at a time. A
NUTS-2 code with no relation of its own (new units that OSM tags only at
NUTS-3) is assembled from its NUTS-3 children: its boundary is the set of
ways that exactly one child uses.
"""
function osm_regions(countries, ids)
    want = Set(ids)
    ways = Dict{Int,OSMWay}()
    members = Dict{String,Vector{Tuple{Int,String}}}()
    how = Dict{String,String}()
    for cc in countries
        d = overpass(nuts_query(cc, 4), "osm_nuts2_$cc.json")
        w, picks = parse_overpass(d, 4, in(want))
        merge!(ways, w)
        for (code, p) in picks
            members[code] = p[4]
            how[code] = "relation $(p[2]) ($(isempty(p[3]) ? "untagged" : p[3]))"
        end
        missing_here = [c for c in ids if startswith(c, cc) && !haskey(members, c)]
        isempty(missing_here) && continue
        d3 = overpass(nuts_query(cc, 5), "osm_nuts3_$cc.json")
        w3, picks3 = parse_overpass(d3, 5, c -> c[1:4] in missing_here)
        merge!(ways, w3)
        for c in missing_here
            kids = [p for (k, p) in picks3 if startswith(k, c)]
            isempty(kids) && continue
            uses = Dict{Int,Int}()
            for p in kids, (wid, role) in p[4]
                role in ("outer", "") && (uses[wid] = get(uses, wid, 0) + 1)
            end
            members[c] = [(wid, "outer") for (wid, n) in uses if n == 1]
            how[c] = "union of $(length(kids)) NUTS-3 relations"
        end
    end
    absent = [c for c in ids if !haskey(members, c)]
    isempty(absent) || error("no OSM geometry for $(join(absent, ", "))")

    geo = Dict{String,Any}(); broken = Dict{String,Vector{Float64}}()
    for c in ids
        outer, g1 = stitch([w for (w, r) in members[c] if r in ("outer", "")], ways)
        inner, g2 = stitch([w for (w, r) in members[c] if r == "inner"], ways)
        isempty(outer) && error("$c: no outer ring in OSM")
        # Joins under a metre are node-matching slips, not missing boundary.
        gs = filter(>=(1.0), vcat(g1, g2))
        isempty(gs) || (broken[c] = sort!(gs; rev = true))
        geo[c] = centroid(outer, inner)
    end

    # Contiguity on land ways only.
    land = Dict(c => Set(w for (w, _) in members[c] if haskey(ways, w) && !ways[w].sea) for c in ids)
    idx = Dict(c => i for (i, c) in enumerate(ids))
    byway = Dict{Int,Vector{Int}}(); bynode = Dict{Int,Set{Int}}()
    for c in ids, w in land[c]
        push!(get!(Vector{Int}, byway, w), idx[c])
        for n in ways[w].nodes
            push!(get!(Set{Int}, bynode, n), idx[c])
        end
    end
    pairs(groups, extra) = begin
        nb = [Set{Int}() for _ in ids]
        for g in groups, a in g, b in g
            a == b && continue
            (ids[a], ids[b]) in SEA_LINKS || (ids[b], ids[a]) in SEA_LINKS || push!(nb[a], b)
        end
        for (a, b) in extra
            haskey(idx, a) && haskey(idx, b) || continue
            push!(nb[idx[a]], idx[b]); push!(nb[idx[b]], idx[a])
        end
        [sort!(collect(s)) for s in nb]
    end
    return (geo = geo, rook = pairs(values(byway), LAKE_ROOK),
            queen = pairs(values(bynode), LAKE_QUEEN),
            how = how, broken = broken)
end

"""Centroid of a single country from its own NUTS-2 relation (Iceland)."""
function osm_single(cc)
    d = overpass(nuts_query(cc, 4), "osm_nuts2_$cc.json")
    ways, picks = parse_overpass(d, 4, _ -> true)
    isempty(picks) && error("no OSM NUTS-2 relation for $cc")
    _, p = first(sort(collect(picks); by = x -> x[2][1]))
    outer, _ = stitch([w for (w, r) in p[4] if r in ("outer", "")], ways)
    inner, _ = stitch([w for (w, r) in p[4] if r == "inner"], ways)
    return centroid(outer, inner)
end
