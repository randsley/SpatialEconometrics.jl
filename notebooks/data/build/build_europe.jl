# Build the European datasets in notebooks/data from Eurostat, OpenStreetMap
# and AMECO.
#
#     julia --project=notebooks/data/build notebooks/data/build/build_europe.jl
#
# Run by hand, once per refresh. Nothing in the package, the test suite or the
# notebooks calls the network: they read the files this script writes.
#
# Writes, into notebooks/data/:
#   europe_nuts2.csv            regional panel, NUTS-2 2024, one row per region-year
#   europe_nuts2_regions.csv    one row per region: name, centroid, land area
#   europe_nuts2_queen.gal      contiguity by shared boundary node
#   europe_nuts2_rook.gal       contiguity by shared boundary way
#   europe_ameco.csv            country panel from AMECO, one row per country-year
#   europe_countries.csv        one row per country: name, centroid, land area
#   europe_countries_queen.gal  country contiguity, from the regional links
#   europe_PROVENANCE.md        sources, vintages, hashes, choices, credits
#
# Raw downloads are cached in notebooks/data/build/cache/ (gitignored). Delete
# the cache to force a fresh pull. A cold build spends most of its time on the
# Overpass queries for the boundaries (about half an hour, one country at a
# time, retried with backoff when the public server is busy).

using CSV, Dates, Downloads, JSON3, Printf, SHA

const HERE  = @__DIR__
const OUT   = normpath(joinpath(HERE, ".."))
const CACHE = joinpath(HERE, "cache")
mkpath(CACHE)

const EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
const AMECO    = "https://webgate.ec.europa.eu/ecfin/redisstat/api/dissemination/sdmx/2.1/data"

# EU27 plus Norway and Switzerland for the regions; Iceland as well for the
# countries (it has no NUTS-2 regional accounts in these tables).
const EU27 = split("AT BE BG CY CZ DE DK EE EL ES FI FR HR HU IE IT LT LU LV MT NL PL PT RO SE SI SK")
const REGION_COUNTRIES  = [EU27; "NO"; "CH"]
const COUNTRY_COUNTRIES = [EU27; "NO"; "CH"; "IS"]

# Regions with no place in a contiguity analysis of Europe: the outermost
# regions (French overseas departments, Canarias, Azores, Madeira), the North
# African enclaves Ceuta and Melilla, and Svalbard and Jan Mayen.
const EXCLUDED = Dict(
    "FRY1" => "outermost region (Guadeloupe)", "FRY2" => "outermost region (Martinique)",
    "FRY3" => "outermost region (Guyane)",     "FRY4" => "outermost region (La Réunion)",
    "FRY5" => "outermost region (Mayotte)",    "ES70" => "outermost region (Canarias)",
    "PT20" => "outermost region (Açores)",     "PT30" => "outermost region (Madeira)",
    "ES63" => "enclave in North Africa (Ceuta)", "ES64" => "enclave in North Africa (Melilla)",
    "NO0B" => "Svalbard and Jan Mayen")

# NUTS country codes to the ISO-3 codes AMECO uses (Romania is ROM there).
const ISO3 = Dict(
    "AT"=>"AUT","BE"=>"BEL","BG"=>"BGR","CY"=>"CYP","CZ"=>"CZE","DE"=>"DEU","DK"=>"DNK",
    "EE"=>"EST","EL"=>"GRC","ES"=>"ESP","FI"=>"FIN","FR"=>"FRA","HR"=>"HRV","HU"=>"HUN",
    "IE"=>"IRL","IT"=>"ITA","LT"=>"LTU","LU"=>"LUX","LV"=>"LVA","MT"=>"MLT","NL"=>"NLD",
    "PL"=>"POL","PT"=>"PRT","RO"=>"ROM","SE"=>"SWE","SI"=>"SVN","SK"=>"SVK","NO"=>"NOR",
    "CH"=>"CHE","IS"=>"ISL")

# ── downloads ─────────────────────────────────────────────────────────────────

const SOURCES = Vector{NamedTuple}()   # every raw file, for the provenance table

function download_cached(url, name)
    path = joinpath(CACHE, name)
    if !isfile(path)
        @info "downloading $name"
        Downloads.download(url, path)
    end
    push!(SOURCES, (name = name, url = url, sha256 = bytes2hex(open(sha256, path)),
                    bytes = filesize(path)))
    return path
end

# ── Eurostat JSON-stat ────────────────────────────────────────────────────────

"""
    eurostat(dataset, fixed; level = "nuts2") -> (values, updated, labels)

`values[(geo, year)]` for one series of a JSON-stat 2.0 response, every
dimension other than `geo` and `time` pinned by `fixed`. A pinned category
the dataset does not have is an error, not an empty series. `labels` maps
each geo code to Eurostat's English name for it.
"""
function eurostat(dataset, fixed::Dict{String,String}; level = "nuts2")
    path = download_cached("$EUROSTAT/$dataset?format=JSON&lang=EN&geoLevel=$level",
                           level == "nuts2" ? "$dataset.json" : "$(dataset)_$level.json")
    d = JSON3.read(read(path, String))
    ids  = String.(d.id)
    sz = Int.(d.size)
    stride = ones(Int, length(sz))
    for i in length(sz)-1:-1:1
        stride[i] = stride[i+1] * sz[i+1]
    end
    index(dim) = Dict(String(k) => Int(v) for (k, v) in d.dimension[Symbol(dim)].category.index)
    base = 0
    for dim in ids
        dim in ("geo", "time") && continue
        haskey(fixed, dim) || error("$dataset: dimension $dim is not pinned")
        idx = index(dim)
        haskey(idx, fixed[dim]) || error("$dataset: no category $(fixed[dim]) in $dim")
        base += idx[fixed[dim]] * stride[findfirst(==(dim), ids)]
    end
    gs, ts = stride[findfirst(==("geo"), ids)], stride[findfirst(==("time"), ids)]
    vals = d.value
    get_value(pos) = vals isa JSON3.Object ? get(vals, Symbol(string(pos)), nothing) :
                                            vals[pos + 1]
    out = Dict{Tuple{String,Int},Float64}()
    for (g, gi) in index("geo"), (t, ti) in index("time")
        v = get_value(base + gi * gs + ti * ts)
        v === nothing || (out[(g, parse(Int, t))] = Float64(v))
    end
    labels = Dict(String(k) => String(v) for (k, v) in d.dimension.geo.category.label)
    return out, String(d.updated), labels
end

"""Most recent value per geo code."""
latest(vals) = Dict(g => v for ((g, t), v) in sort(collect(vals); by = x -> x[1][2]))

const REGIONAL = [
    # column              dataset              pinned categories
    ("gdp_mio_eur",       "nama_10r_2gdp",     Dict("freq"=>"A", "unit"=>"MIO_EUR")),
    ("gdp_pps_per_head",  "nama_10r_2gdp",     Dict("freq"=>"A", "unit"=>"PPS_EU27_2020_HAB")),
    ("real_gdp_index",    "nama_10r_2gvagr",   Dict("freq"=>"A", "na_item"=>"B1GQ", "unit"=>"I20")),
    ("population_ths",    "nama_10r_3popgdp",  Dict("freq"=>"A", "unit"=>"THS")),
    ("employment_ths",    "lfst_r_lfe2emp",    Dict("freq"=>"A", "unit"=>"THS_PER", "sex"=>"T", "age"=>"Y15-64")),
    ("unemployment_rate", "lfst_r_lfu3rt",     Dict("freq"=>"A", "unit"=>"PC", "sex"=>"T", "age"=>"Y15-74", "isced11"=>"TOTAL")),
    ("gfcf_mio_eur",      "nama_10r_2gfcf",    Dict("freq"=>"A", "sector"=>"S1", "currency"=>"MIO_EUR", "nace_r2"=>"TOTAL")),
]

# ── geometry: OpenStreetMap ───────────────────────────────────────────────────

include(joinpath(HERE, "osm_geometry.jl"))

function write_gal(path, ids, nbrs, label)
    open(path, "w") do io
        println(io, "0 ", length(ids), " ", label, " NUTS_ID")
        for (i, id) in enumerate(ids)
            println(io, id, " ", length(nbrs[i]))
            isempty(nbrs[i]) || println(io, join(ids[nbrs[i]], " "))
        end
    end
end

csvfield(s) = occursin(r"[,\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
# Fixed-point, never scientific notation, so the files read the same in any tool.
num(v; digits = 6) = v === nothing || (v isa Float64 && isnan(v)) ? "" :
    (s = Printf.format(Printf.Format("%.$(digits)f"), v);
     occursin('.', s) ? rstrip(rstrip(s, '0'), '.') : s)

const COUNTRY_NAMES = Dict(
    "AT"=>"Austria","BE"=>"Belgium","BG"=>"Bulgaria","CY"=>"Cyprus","CZ"=>"Czechia",
    "DE"=>"Germany","DK"=>"Denmark","EE"=>"Estonia","EL"=>"Greece","ES"=>"Spain",
    "FI"=>"Finland","FR"=>"France","HR"=>"Croatia","HU"=>"Hungary","IE"=>"Ireland",
    "IT"=>"Italy","LT"=>"Lithuania","LU"=>"Luxembourg","LV"=>"Latvia","MT"=>"Malta",
    "NL"=>"Netherlands","PL"=>"Poland","PT"=>"Portugal","RO"=>"Romania","SE"=>"Sweden",
    "SI"=>"Slovenia","SK"=>"Slovakia","NO"=>"Norway","CH"=>"Switzerland","IS"=>"Iceland")

# ── AMECO SDMX ────────────────────────────────────────────────────────────────

const AMECO_SERIES = [
    # column               chapter               item    unit  (212 PPS, 99 EUR, 0 national)
    ("gdp_bn_pps",         "AMECO_CHAPTER06-01", "UVGD", "212"),
    ("gdp_bn_eur",         "AMECO_CHAPTER06-01", "UVGD", "99"),
    ("real_gdp_bn_nc",     "AMECO_CHAPTER06-01", "OVGD", "0"),
    ("population_ths",     "AMECO_CHAPTER01-01", "NPTD", "0"),
    ("employment_ths",     "AMECO_CHAPTER01-04", "NETD", "0"),
    ("unemployment_rate",  "AMECO_CHAPTER01-03", "ZUTN", "0"),
    ("gfcf_bn_eur",        "AMECO_CHAPTER03-01", "UIGT", "99"),
    ("compensation_bn_eur","AMECO_CHAPTER07-01", "UWCD", "99"),
    ("net_capital_bn_eur", "AMECO_CHAPTER08-01", "OKND", "99"),
    ("exports_bn_eur",     "AMECO_CHAPTER09-01", "UXGS", "99"),
    ("imports_bn_eur",     "AMECO_CHAPTER09-02", "UMGS", "99"),
]

# Release year from LAST UPDATE ("dd/mm/yy hh:mm:ss", or ISO); nothing if unreadable.
function release_year(s)
    m = match(r"^\s*(\d{4})-\d{2}-\d{2}", s); m === nothing || return parse(Int, m[1])
    m = match(r"^\s*\d{1,2}/\d{1,2}/(\d{2}|\d{4})\b", s); m === nothing && return nothing
    y = parse(Int, m[1]); return y < 100 ? 2000 + y : y
end

function ameco(iso3, chapter, item, unit)
    path = download_cached("$AMECO/ECFIN,$chapter,1.0/A.$iso3.1..$unit.0.$item?format=SDMX-CSV",
                 "ameco_$(iso3)_$(item)_$(unit).csv")
    out = Dict{Int,Float64}(); releases = Set{Int}()
    for row in CSV.File(path; types = String, silencewarnings = true)
        hasproperty(row, :TIME_PERIOD) || break
        (ismissing(row.OBS_VALUE) || isempty(row.OBS_VALUE)) && continue
        out[parse(Int, row.TIME_PERIOD)] = parse(Float64, row.OBS_VALUE)
        lu = getproperty(row, Symbol("LAST UPDATE"))
        ismissing(lu) || (y = release_year(lu); y === nothing || push!(releases, y))
    end
    return out, releases
end

# ── build ─────────────────────────────────────────────────────────────────────

function main()
    # Regions ------------------------------------------------------------------
    series = Dict{String,Dict{Tuple{String,Int},Float64}}()
    updated = Dict{String,String}()
    labels = Dict{String,String}()
    for (col, ds, fixed) in REGIONAL
        series[col], updated[ds], l = eurostat(ds, fixed)
        merge!(labels, l)
    end
    # The region list is NUTS 2024 as Eurostat's regional accounts publish it.
    keep_region(id) = length(id) == 4 && id[1:2] in REGION_COUNTRIES &&
                      !endswith(id, "ZZ") && !haskey(EXCLUDED, id)
    rid = sort(unique(g for ((g, _), _) in series["gdp_mio_eur"] if keep_region(g)))
    @info "regions" length(rid)
    rname = [labels[id] for id in rid]

    area_vals, updated["reg_area3"], _ = eurostat("reg_area3",
        Dict("freq" => "A", "landuse" => "L0008", "unit" => "KM2"))
    rarea = latest(area_vals)
    missing_area = [id for id in rid if !haskey(rarea, id)]
    isempty(missing_area) || error("no land area for $(join(missing_area, ", "))")

    osm = osm_regions(REGION_COUNTRIES, rid)
    rgeo = [osm.geo[id] for id in rid]
    rqueen, rrook = osm.queen, osm.rook

    years = sort(unique(t for s in values(series) for ((g, t), _) in s if g in rid))
    open(joinpath(OUT, "europe_nuts2.csv"), "w") do io
        println(io, join(["nuts_id", "country", "year", first.(REGIONAL)...], ","))
        for id in rid, t in years
            vals = [get(series[c], (id, t), nothing) for c in first.(REGIONAL)]
            all(isnothing, vals) && continue
            println(io, join([id, id[1:2], string(t), num.(vals)...], ","))
        end
    end
    open(joinpath(OUT, "europe_nuts2_regions.csv"), "w") do io
        println(io, "nuts_id,country,x_laea,y_laea,lon,lat,land_area_km2,queen_neighbours,rook_neighbours,name")
        for (i, id) in enumerate(rid)
            g = rgeo[i]
            println(io, join([id, id[1:2], num(g.x; digits = 1), num(g.y; digits = 1),
                              num(g.lon), num(g.lat), num(rarea[id]; digits = 1),
                              string(length(rqueen[i])), string(length(rrook[i])),
                              csvfield(rname[i])], ","))
        end
    end
    write_gal(joinpath(OUT, "europe_nuts2_queen.gal"), rid, rqueen, "europe_nuts2")
    write_gal(joinpath(OUT, "europe_nuts2_rook.gal"),  rid, rrook,  "europe_nuts2")

    # Countries ----------------------------------------------------------------
    # Geometry from the regions: a country's centroid is the land-area-weighted
    # mean of its regions' centroids (so the excluded outermost regions do not
    # pull it out to sea), and two countries are neighbours when any of their
    # regions are. Iceland has no regions here and is fetched on its own.
    cid = sort(COUNTRY_COUNTRIES)
    carea_vals, _, _ = eurostat("reg_area3",
        Dict("freq" => "A", "landuse" => "L0008", "unit" => "KM2"); level = "country")
    carea = latest(carea_vals)
    cgeo = map(cid) do c
        members = [i for (i, id) in enumerate(rid) if id[1:2] == c]
        if isempty(members)
            g = osm_single(c)
            return (x = g.x, y = g.y, lon = g.lon, lat = g.lat)
        end
        w = [rarea[rid[i]] for i in members]; w ./= sum(w)
        (x = sum(w .* [rgeo[i].x for i in members]), y = sum(w .* [rgeo[i].y for i in members]),
         lon = sum(w .* [rgeo[i].lon for i in members]), lat = sum(w .* [rgeo[i].lat for i in members]))
    end
    cidx = Dict(c => k for (k, c) in enumerate(cid))
    cn = [Set{Int}() for _ in cid]
    for (i, nb) in enumerate(rqueen), j in nb
        a, b = cidx[rid[i][1:2]], cidx[rid[j][1:2]]
        a == b || (push!(cn[a], b); push!(cn[b], a))
    end
    cqueen = [sort!(collect(s)) for s in cn]
    cname = [COUNTRY_NAMES[c] for c in cid]

    am = Dict{String,Dict{Tuple{String,Int},Float64}}()
    releases = Set{Int}()
    for (col, ch, item, unit) in AMECO_SERIES
        am[col] = Dict{Tuple{String,Int},Float64}()
        for c in cid
            s, r = ameco(ISO3[c], ch, item, unit)
            union!(releases, r)
            for (t, v) in s
                am[col][(c, t)] = v
            end
        end
    end
    isempty(releases) && error("no AMECO release year readable from LAST UPDATE: " *
                               "forecast years cannot be identified, refusing to write")
    release = maximum(releases)
    last_actual = release - 1          # the release year and the next are forecasts
    ayears = sort(unique(t for s in values(am) for ((_, t), _) in s if t <= last_actual))
    open(joinpath(OUT, "europe_ameco.csv"), "w") do io
        println(io, join(["nuts_id", "iso3", "year", first.(AMECO_SERIES)...], ","))
        for c in cid, t in ayears
            vals = [get(am[col], (c, t), nothing) for col in first.(AMECO_SERIES)]
            all(isnothing, vals) && continue
            println(io, join([c, ISO3[c], string(t), num.(vals)...], ","))
        end
    end
    open(joinpath(OUT, "europe_countries.csv"), "w") do io
        println(io, "nuts_id,iso3,x_laea,y_laea,lon,lat,land_area_km2,queen_neighbours,name")
        for (i, c) in enumerate(cid)
            g = cgeo[i]
            println(io, join([c, ISO3[c], num(g.x; digits = 1), num(g.y; digits = 1),
                              num(g.lon), num(g.lat), num(get(carea, c, nothing); digits = 1),
                              string(length(cqueen[i])), csvfield(cname[i])], ","))
        end
    end
    write_gal(joinpath(OUT, "europe_countries_queen.gal"), cid, cqueen, "europe_countries")

    write_provenance(rid, rname, rqueen, rrook, osm, series, updated, years,
                     cid, cname, cqueen, release, last_actual, ayears)
end

function coverage(series, ids, cols, years)
    map(cols) do c
        ts = [t for ((g, t), _) in series[c] if g in ids]
        (c, count(id -> any(t -> haskey(series[c], (id, t)), years), ids),
         minimum(ts), maximum(ts))
    end
end

function write_provenance(rid, rname, rqueen, rrook, osm, series, updated, years,
                          cid, cname, cqueen, release, last_actual, ayears)
    today = Dates.format(Dates.today(), "yyyy-mm-dd")
    isolated(ids, nb) = [ids[i] for i in eachindex(ids) if isempty(nb[i])]
    nlinks(nb) = sum(length, nb) ÷ 2
    named(ids, names, sel) = join(["`$x` " * names[findfirst(==(x), ids)] for x in sel], ", ")
    open(joinpath(OUT, "europe_PROVENANCE.md"), "w") do io
        print(io, """
        # European regions and countries: provenance and attribution

        Three sources, each with a credit that is a condition of its terms.
        Keep all three with any published result.

        > Source: Eurostat, regional statistics (datasets listed below), extracted $today.
        >
        > Source: European Commission, DG ECFIN, AMECO database, $(release) release, extracted $today.
        >
        > Boundaries: © OpenStreetMap contributors, available under the Open Database License (ODbL)

        **Eurostat** statistical data may be reused, commercially or not, provided
        the source is acknowledged (Commission Decision 2011/833/EU; Eurostat
        copyright notice). **AMECO** is Commission content under the same
        decision, licensed CC BY 4.0: credit the source and indicate changes.

        **Boundaries** come from **OpenStreetMap**, under the Open Database
        License 1.0 (<https://opendatacommons.org/licenses/odbl/1-0/>), which
        permits commercial use. No boundary geometry is redistributed here, but
        five files are derived from it: `europe_nuts2_regions.csv` (the
        centroid columns), `europe_nuts2_queen.gal`, `europe_nuts2_rook.gal`,
        `europe_countries.csv` (the centroid columns) and
        `europe_countries_queen.gal`. As a database derived from OSM they are
        offered under the **ODbL**: credit "© OpenStreetMap contributors", and
        if you publicly use an adapted version of them, offer that version
        under the ODbL too. The land areas in those files are Eurostat's, not
        OSM's. The package's MIT licence covers its code, not these files.

        GISCO's NUTS boundaries were the first choice and are not used: Eurostat
        licenses them for non-commercial use only, and anything derived from
        them would carry that restriction.

        Built by `notebooks/data/build/build_europe.jl` (with
        `osm_geometry.jl`), which writes every file listed here in one run. The
        changes made to the sources are the ones this file describes: series
        selected and pinned, units converted only where stated, forecast years
        removed, regions excluded, sea boundaries dropped from the contiguity,
        Lake Constance links added to it, and boundary gaps closed.

        ## Files

        | file | units | contents |
        |---|---|---|
        | `europe_nuts2.csv` | $(length(rid)) | regional panel, $(first(years))-$(last(years)), one row per region-year with any value |
        | `europe_nuts2_regions.csv` | $(length(rid)) | Eurostat name, centroid (EPSG:3035 metres and approximate WGS84), Eurostat land area, neighbour counts |
        | `europe_nuts2_queen.gal` | $(length(rid)) | contiguity by shared land boundary node, $(nlinks(rqueen)) links |
        | `europe_nuts2_rook.gal` | $(length(rid)) | contiguity by shared land boundary way, $(nlinks(rrook)) links |
        | `europe_ameco.csv` | $(length(cid)) | country panel, $(first(ayears))-$(last(ayears)), one row per country-year |
        | `europe_countries.csv` | $(length(cid)) | name, centroid, Eurostat land area, neighbour count |
        | `europe_countries_queen.gal` | $(length(cid)) | country contiguity, $(nlinks(cqueen)) links |

        Regions are keyed by the **NUTS 2024** code and countries by the NUTS
        country code (`EL` for Greece), with the ISO-3 code AMECO uses beside it
        (`ROM` for Romania, as in AMECO). Every file is sorted by that key, so
        the files line up by sorting alone. Missing values are empty fields.

        ## Regions

        NUTS-2 regions (2024 classification) of the EU27, Norway and Switzerland,
        as Eurostat's regional accounts list them. Excluded, because they have
        no place in a contiguity analysis of Europe:

        | code | reason |
        |---|---|
        """)
        for k in sort(collect(keys(EXCLUDED)))
            println(io, "| `", k, "` | ", EXCLUDED[k], " |")
        end
        iso = isolated(rid, rqueen)
        built = sort([c for (c, h) in osm.how if startswith(h, "union")])
        builtnote = isempty(built) ? "" :
            "OSM has no NUTS-2 relation for " * join(["`$c`" for c in built], ", ") *
            ", new in NUTS 2024, so each is assembled from its NUTS-3 relations: " *
            "its boundary is the set of ways that exactly one of them uses."
        brokennote = isempty(osm.broken) ? "" :
            "Where an OSM boundary does not close, the open ends are joined across " *
            "the shortest gap (widths in metres): " *
            join(["`$c` ($(join([string(round(g; digits = 1)) for g in gs], ", ")))"
                  for (c, gs) in sort(collect(osm.broken))], "; ") * "."
        linklist(s) = join(["`$(a)`–`$(b)`" for (a, b) in sort(collect(s))], ", ")
        sealinks = linklist(SEA_LINKS)
        print(io, """

        **$(length(iso)) regions have no land neighbour** in the queen file:
        $(named(rid, rname, iso)).
        A contiguity matrix over all $(length(rid)) therefore has $(length(iso))
        zero rows, and row-standardising divides by zero. Drop them, or use a
        k-nearest-neighbour matrix from the centroids
        (`knn_weights(hcat(x_laea, y_laea), k)`), which gives every region
        neighbours. Distances should come from the EPSG:3035 coordinates, which
        are in metres; the WGS84 centroids are a planar approximation for labels.

        ### How the geometry was built

        Each region is the OSM boundary relation tagged with its NUTS code
        (`ref:nuts:2`, or `ref:NUTS`/`ref:nuts`), preferring a statistical
        boundary over an administrative or historic one with the same code.
        $builtnote

        Contiguity needs no geometry engine. OSM stores a boundary between two
        regions once, as a way both relations list, so two regions share an
        edge (rook) when they share a way and touch (queen) when their ways
        share a node. Only land boundaries count: ways tagged `maritime=yes`,
        `boundary=maritime` or `natural=coastline` are dropped, and so are
        $(length(SEA_LINKS)) sea links OSM does not tag as such: $sealinks.
        Lake Constance is the exception the other way: no state has divided
        it, so OSM leaves it outside every region, and the links that the
        conventional NUTS split of the lake creates are added by hand: rook
        $(linklist(LAKE_ROOK)); queen $(linklist(LAKE_QUEEN)).

        Centroids are area-weighted over each region's OSM polygon in
        ETRS89-LAEA (EPSG:3035). OSM's coastal regions extend into territorial
        waters, so a coastal centroid sits slightly seaward of the land
        centroid; that is also why areas come from Eurostat (`reg_area3`, land
        area), not from the polygons. $brokennote

        ### Regional series

        The panel is unbalanced. NUTS 2024 redrew regions in several countries,
        and some tables have no back series for the new units. Select a
        balanced subset for the question at hand, and report which regions it
        drops.

        | column | dataset | categories | regions with data | years |
        |---|---|---|---|---|
        """)
        for ((col, ds, fixed), (c, n, y0, y1)) in zip(REGIONAL, coverage(series, rid, first.(REGIONAL), years))
            cats = join(["$k=$v" for (k, v) in sort(collect(fixed)) if k != "freq"], ", ")
            println(io, "| `$col` | `$ds` (updated $(updated[ds][1:10])) | $cats | $n | $y0-$y1 |")
        end
        print(io, """

        `real_gdp_index` is the chain-linked volume index of GDP, 2020 = 100;
        growth rates come from it, not from current-price GDP. Land area is
        `reg_area3` (`landuse=L0008`, updated $(updated["reg_area3"][1:10])), the
        most recent year published for each region.

        ## Countries (AMECO)

        The EU27, Norway, Switzerland and Iceland. AMECO publishes the
        Commission's forecasts for the release year and the next one in the
        same series as the observed data, and nothing in the response marks
        them. The release year is read from the `LAST UPDATE` column ($(release)),
        and every year from $(release) on is dropped: the last year kept is
        $(last_actual), which is itself provisional. The build refuses to write
        the file if no release year can be read.

        | column | item | unit |
        |---|---|---|
        """)
        unitname = Dict("212" => "billion PPS", "99" => "billion EUR", "0" => "national (see AMECO)")
        for (col, ch, item, unit) in AMECO_SERIES
            println(io, "| `$col` | `$item` (`$ch`) | $(unitname[unit]) |")
        end
        ciso = isolated(cid, cqueen)
        print(io, """

        `population_ths` and `employment_ths` are thousands of persons,
        `unemployment_rate` is a percentage of the labour force, and
        `real_gdp_bn_nc` is GDP at constant prices in national currency: use it
        for growth rates only, never for levels across countries. Ratios
        (investment share, wage share, openness) should be formed from the EUR
        columns, and level comparisons from the PPS column.

        A country's centroid is the land-area-weighted mean of its regions'
        centroids, so the excluded outermost regions do not pull it out to sea
        (France is metropolitan France, Portugal mainland Portugal); Iceland,
        which has no regions here, has its own. Two countries are neighbours
        when any of their regions are.

        **$(length(ciso)) countries have no land neighbour** inside the set:
        $(named(cid, cname, ciso)).
        Ireland's only land border is with the United Kingdom, which is outside
        it. With about 30 countries the country panel suits a static spatial
        panel, not methods whose asymptotics need many units; use the regions
        for those.

        ## Raw files

        Hashes of the downloads this build read, so a refresh can be told apart
        from a re-run.

        | file | bytes | sha256 |
        |---|---|---|
        """)
        for s in unique(x -> x.name, SOURCES)
            occursin("ameco_", s.name) && continue
            println(io, "| `$(s.name)` | $(s.bytes) | `$(s.sha256[1:16])…` |")
        end
        nam = count(x -> occursin("ameco_", x.name), unique(x -> x.name, SOURCES))
        println(io, "| AMECO | $nam files, one per country and series | not listed |")
    end
end

main()
