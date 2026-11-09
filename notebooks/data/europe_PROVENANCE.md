# European regions and countries: provenance and attribution

Three sources, each with a credit that is a condition of its terms.
Keep all three with any published result.

> Source: Eurostat, regional statistics (datasets listed below), extracted 2026-11-09.
>
> Source: European Commission, DG ECFIN, AMECO database, 2026 release, extracted 2026-11-09.
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
| `europe_nuts2.csv` | 247 | regional panel, 1999-2025, one row per region-year with any value |
| `europe_nuts2_regions.csv` | 247 | Eurostat name, centroid (EPSG:3035 metres and approximate WGS84), Eurostat land area, neighbour counts |
| `europe_nuts2_queen.gal` | 247 | contiguity by shared land boundary node, 561 links |
| `europe_nuts2_rook.gal` | 247 | contiguity by shared land boundary way, 556 links |
| `europe_ameco.csv` | 30 | country panel, 1960-2025, one row per country-year |
| `europe_countries.csv` | 30 | name, centroid, Eurostat land area, neighbour count |
| `europe_countries_queen.gal` | 30 | country contiguity, 41 links |

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
| `ES63` | enclave in North Africa (Ceuta) |
| `ES64` | enclave in North Africa (Melilla) |
| `ES70` | outermost region (Canarias) |
| `FRY1` | outermost region (Guadeloupe) |
| `FRY2` | outermost region (Martinique) |
| `FRY3` | outermost region (Guyane) |
| `FRY4` | outermost region (La Réunion) |
| `FRY5` | outermost region (Mayotte) |
| `NO0B` | Svalbard and Jan Mayen |
| `PT20` | outermost region (Açores) |
| `PT30` | outermost region (Madeira) |

**11 regions have no land neighbour** in the queen file:
`CY00` Kýpros, `EL41` Voreio Aigaio, `EL42` Notio Aigaio, `EL43` Kriti, `EL62` Ionia Nisia, `ES53` Illes Balears, `FI20` Åland, `FRM0` Corse, `ITG1` Sicilia, `ITG2` Sardegna, `MT00` Malta.
A contiguity matrix over all 247 therefore has 11
zero rows, and row-standardising divides by zero. Drop them, or use a
k-nearest-neighbour matrix from the centroids
(`knn_weights(hcat(x_laea, y_laea), k)`), which gives every region
neighbours. Distances should come from the EPSG:3035 coordinates, which
are in metres; the WGS84 centroids are a planar approximation for labels.

### How the geometry was built

Each region is the OSM boundary relation tagged with its NUTS code
(`ref:nuts:2`, or `ref:NUTS`/`ref:nuts`), preferring a statistical
boundary over an administrative or historic one with the same code.
OSM has no NUTS-2 relation for `PT19`, `PT1C`, `PT1D`, new in NUTS 2024, so each is assembled from its NUTS-3 relations: its boundary is the set of ways that exactly one of them uses.

Contiguity needs no geometry engine. OSM stores a boundary between two
regions once, as a way both relations list, so two regions share an
edge (rook) when they share a way and touch (queen) when their ways
share a node. Only land boundaries count: ways tagged `maritime=yes`,
`boundary=maritime` or `natural=coastline` are dropped, and so are
2 sea links OSM does not tag as such: `FI20`–`SE11`, `FI20`–`SE12`.
Lake Constance is the exception the other way: no state has divided
it, so OSM leaves it outside every region, and the links that the
conventional NUTS split of the lake creates are added by hand: rook
`CH05`–`DE14`; queen `AT34`–`DE14`, `CH05`–`DE14`, `CH05`–`DE27`.

Centroids are area-weighted over each region's OSM polygon in
ETRS89-LAEA (EPSG:3035). OSM's coastal regions extend into territorial
waters, so a coastal centroid sits slightly seaward of the land
centroid; that is also why areas come from Eurostat (`reg_area3`, land
area), not from the polygons. Where an OSM boundary does not close, the open ends are joined across the shortest gap (widths in metres): `HU33` (3064.9).

### Regional series

The panel is unbalanced. NUTS 2024 redrew regions in several countries,
and some tables have no back series for the new units. Select a
balanced subset for the question at hand, and report which regions it
drops.

| column | dataset | categories | regions with data | years |
|---|---|---|---|---|
| `gdp_mio_eur` | `nama_10r_2gdp` (updated 2026-02-10) | unit=MIO_EUR | 247 | 2000-2024 |
| `gdp_pps_per_head` | `nama_10r_2gdp` (updated 2026-02-10) | unit=PPS_EU27_2020_HAB | 240 | 2000-2024 |
| `real_gdp_index` | `nama_10r_2gvagr` (updated 2026-03-23) | na_item=B1GQ, unit=I20 | 234 | 2000-2024 |
| `population_ths` | `nama_10r_3popgdp` (updated 2026-02-10) | unit=THS | 240 | 2000-2024 |
| `employment_ths` | `lfst_r_lfe2emp` (updated 2026-10-30) | age=Y15-64, sex=T, unit=THS_PER | 247 | 1999-2025 |
| `unemployment_rate` | `lfst_r_lfu3rt` (updated 2026-10-30) | age=Y15-74, isced11=TOTAL, sex=T, unit=PC | 246 | 1999-2025 |
| `gfcf_mio_eur` | `nama_10r_2gfcf` (updated 2026-03-26) | currency=MIO_EUR, nace_r2=TOTAL, sector=S1 | 240 | 2000-2024 |

`real_gdp_index` is the chain-linked volume index of GDP, 2020 = 100;
growth rates come from it, not from current-price GDP. Land area is
`reg_area3` (`landuse=L0008`, updated 2026-01-22), the
most recent year published for each region.

## Countries (AMECO)

The EU27, Norway, Switzerland and Iceland. AMECO publishes the
Commission's forecasts for the release year and the next one in the
same series as the observed data, and nothing in the response marks
them. The release year is read from the `LAST UPDATE` column (2026),
and every year from 2026 on is dropped: the last year kept is
2025, which is itself provisional. The build refuses to write
the file if no release year can be read.

| column | item | unit |
|---|---|---|
| `gdp_bn_pps` | `UVGD` (`AMECO_CHAPTER06-01`) | billion PPS |
| `gdp_bn_eur` | `UVGD` (`AMECO_CHAPTER06-01`) | billion EUR |
| `real_gdp_bn_nc` | `OVGD` (`AMECO_CHAPTER06-01`) | national (see AMECO) |
| `population_ths` | `NPTD` (`AMECO_CHAPTER01-01`) | national (see AMECO) |
| `employment_ths` | `NETD` (`AMECO_CHAPTER01-04`) | national (see AMECO) |
| `unemployment_rate` | `ZUTN` (`AMECO_CHAPTER01-03`) | national (see AMECO) |
| `gfcf_bn_eur` | `UIGT` (`AMECO_CHAPTER03-01`) | billion EUR |
| `compensation_bn_eur` | `UWCD` (`AMECO_CHAPTER07-01`) | billion EUR |
| `net_capital_bn_eur` | `OKND` (`AMECO_CHAPTER08-01`) | billion EUR |
| `exports_bn_eur` | `UXGS` (`AMECO_CHAPTER09-01`) | billion EUR |
| `imports_bn_eur` | `UMGS` (`AMECO_CHAPTER09-02`) | billion EUR |

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

**4 countries have no land neighbour** inside the set:
`CY` Cyprus, `IE` Ireland, `IS` Iceland, `MT` Malta.
Ireland's only land border is with the United Kingdom, which is outside
it. With about 30 countries the country panel suits a static spatial
panel, not methods whose asymptotics need many units; use the regions
for those.

## Raw files

Hashes of the downloads this build read, so a refresh can be told apart
from a re-run.

| file | bytes | sha256 |
|---|---|---|
| `nama_10r_2gdp.json` | 770971 | `93116cdd7dc56481…` |
| `nama_10r_2gvagr.json` | 465539 | `c523f735d1fe841a…` |
| `nama_10r_3popgdp.json` | 119085 | `f0336df1d645e43b…` |
| `lfst_r_lfe2emp.json` | 4982143 | `1f3fa870ccafaf08…` |
| `lfst_r_lfu3rt.json` | 18460696 | `c8c80741c546fcef…` |
| `nama_10r_2gfcf.json` | 3261640 | `aea9c9754608bbde…` |
| `reg_area3.json` | 149544 | `b90032f9c8da9174…` |
| `osm_nuts2_AT.json` | 11727356 | `8762f7c29842e6aa…` |
| `osm_nuts2_BE.json` | 7746946 | `08ddbabd67e666c2…` |
| `osm_nuts2_BG.json` | 7387648 | `4cb637db79875cd2…` |
| `osm_nuts2_CY.json` | 134113 | `ab46de5aff3ed945…` |
| `osm_nuts2_CZ.json` | 13582101 | `d6d1333ea3c323a2…` |
| `osm_nuts2_DE.json` | 34030549 | `c786d393c45076cb…` |
| `osm_nuts2_DK.json` | 21042440 | `13585db38ad08e80…` |
| `osm_nuts2_EE.json` | 823358 | `7414cdd4e7bba22a…` |
| `osm_nuts2_EL.json` | 6784071 | `87f4f3e47e3f7d02…` |
| `osm_nuts2_ES.json` | 47185555 | `af5a2acca9c49e46…` |
| `osm_nuts2_FI.json` | 2054071 | `5862798a49326756…` |
| `osm_nuts2_FR.json` | 64333521 | `022c78c9283d6714…` |
| `osm_nuts2_HR.json` | 4993268 | `22ca0379e4f9a38e…` |
| `osm_nuts2_HU.json` | 3995055 | `48694f77d8e3bae6…` |
| `osm_nuts2_IE.json` | 27298215 | `5e4e00665122e1be…` |
| `osm_nuts2_IT.json` | 41588637 | `c0747a29cbb3e769…` |
| `osm_nuts2_LT.json` | 4270482 | `7a3eb0633426c46c…` |
| `osm_nuts2_LU.json` | 1235084 | `2982765db3b50e7e…` |
| `osm_nuts2_LV.json` | 2672969 | `244188a44eed1dcf…` |
| `osm_nuts2_MT.json` | 45199 | `5771032393f01c84…` |
| `osm_nuts2_NL.json` | 6022428 | `55864a66f8ae0ce1…` |
| `osm_nuts2_PL.json` | 13617688 | `56ca76a3bded9703…` |
| `osm_nuts2_PT.json` | 12383490 | `ee64ac04a5bcb2cf…` |
| `osm_nuts3_PT.json` | 15392363 | `01731552e6bfc37b…` |
| `osm_nuts2_RO.json` | 4420745 | `b69c0b567cb37994…` |
| `osm_nuts2_SE.json` | 2625152 | `ae65de8ea40a261a…` |
| `osm_nuts2_SI.json` | 4248166 | `cdd8716e1fe3f662…` |
| `osm_nuts2_SK.json` | 4480266 | `2af2e7fc225eb4f8…` |
| `osm_nuts2_NO.json` | 3504132 | `564965c80ac7bd06…` |
| `osm_nuts2_CH.json` | 6290382 | `66d4e3e9d46d73f3…` |
| `reg_area3_country.json` | 16350 | `e720c0b5cca35df5…` |
| `osm_nuts2_IS.json` | 62036 | `231a2cda1544878b…` |
| AMECO | 330 files, one per country and series | not listed |
