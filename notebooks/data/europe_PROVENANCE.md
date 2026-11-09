# European regions and countries: provenance and attribution

Three sources, each with a credit that is a condition of its terms.
Keep all three with any published result.

> Source: Eurostat, regional statistics (datasets listed below), extracted 2026-11-09.
>
> Source: European Commission, DG ECFIN, AMECO database, 2026 release, extracted 2026-11-09.
>
> © EuroGeographics for the administrative boundaries

**Eurostat** statistical data may be reused, commercially or not, provided
the source is acknowledged (Commission Decision 2011/833/EU; Eurostat
copyright notice). **AMECO** is Commission content under the same
decision, licensed CC BY 4.0: credit the source and indicate changes.
The **boundaries** are GISCO's NUTS 2024 layer, and their terms differ:
Eurostat grants a non-exclusive, non-transferable right to use and
process them for **non-commercial purposes only**, with the EuroGeographics
notice above visible on any publication. No boundary geometry is
redistributed here, but five files are derived from it and are offered
under those same non-commercial terms: `europe_nuts2_regions.csv`,
`europe_nuts2_queen.gal`, `europe_nuts2_rook.gal`,
`europe_countries.csv` and `europe_countries_queen.gal`. Commercial use
of those five needs a licence from EuroGeographics. The package's MIT
licence covers its code, not these files.

Built by `notebooks/data/build/build_europe.jl`, which writes every file
listed here in one run. The changes made to the sources are the ones
this file describes: series selected and pinned, units converted only
where stated, forecast years removed, regions excluded.

## Files

| file | units | contents |
|---|---|---|
| `europe_nuts2.csv` | 247 | regional panel, 1999-2025, one row per region-year with any value |
| `europe_nuts2_regions.csv` | 247 | name, centroid (EPSG:3035 metres and approximate WGS84), area, neighbour counts |
| `europe_nuts2_queen.gal` | 247 | contiguity by shared vertex, 561 links |
| `europe_nuts2_rook.gal` | 247 | contiguity by shared edge, 555 links |
| `europe_ameco.csv` | 30 | country panel, 1960-2025, one row per country-year |
| `europe_countries.csv` | 30 | name, centroid, area, neighbour count |
| `europe_countries_queen.gal` | 30 | country contiguity by shared vertex, 41 links |

Regions are keyed by the **NUTS 2024** code and countries by the NUTS
country code (`EL` for Greece), with the ISO-3 code AMECO uses beside it
(`ROM` for Romania, as in AMECO). Every file is sorted by that key, so
the files line up by sorting alone. Missing values are empty fields.

## Regions

NUTS-2 regions (2024 classification) of the EU27, Norway and Switzerland.
Excluded, because they have no place in a contiguity analysis of Europe:

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

**10 regions have no land neighbour** in the queen file:
`CY00` Kýpros, `EL41` Voreio Aigaio, `EL42` Notio Aigaio, `EL43` Kriti, `ES53` Illes Balears, `FI20` Åland, `FRM0` Corse, `ITG1` Sicilia, `ITG2` Sardegna, `MT00` Malta.
A contiguity matrix over all 247 therefore has 10
zero rows, and row-standardising divides by zero. Drop them, or use a
k-nearest-neighbour matrix from the centroids
(`knn_weights(hcat(x_laea, y_laea), k)`), which gives every region
neighbours. Distances should come from the EPSG:3035 coordinates, which
are in metres; the WGS84 centroids are a planar approximation for labels.

The contiguity is exact-vertex matching on the 1:1 million layer. The
coarser 1:20 million layer generalises some boundaries away and loses
links, which is why it is not used.

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
growth rates come from it, not from current-price GDP.

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

**4 countries have no land neighbour** inside the set:
`CY` Kýpros, `IE` Éire/Ireland, `IS` Ísland, `MT` Malta.
Ireland's only land border is with the United Kingdom, which is outside
it. With about 30 countries the country panel suits a static spatial
panel, not methods whose asymptotics need many units; use the regions
for those.

## Raw files

Hashes of the downloads this build read, so a refresh can be told apart
from a re-run.

| file | bytes | sha256 |
|---|---|---|
| `NUTS_RG_01M_2024_3035_LEVL_2.geojson` | 15242672 | `8ca30d7d341460f6…` |
| `NUTS_RG_01M_2024_4326_LEVL_2.geojson` | 16744412 | `d59fe1bfdf2c279b…` |
| `nama_10r_2gdp.json` | 770971 | `93116cdd7dc56481…` |
| `nama_10r_2gvagr.json` | 465539 | `c523f735d1fe841a…` |
| `nama_10r_3popgdp.json` | 119085 | `f0336df1d645e43b…` |
| `lfst_r_lfe2emp.json` | 4982143 | `1f3fa870ccafaf08…` |
| `lfst_r_lfu3rt.json` | 18460696 | `c8c80741c546fcef…` |
| `nama_10r_2gfcf.json` | 3261640 | `aea9c9754608bbde…` |
| `NUTS_RG_01M_2024_3035_LEVL_0.geojson` | 8812980 | `69126e1db3840d8c…` |
| `NUTS_RG_01M_2024_4326_LEVL_0.geojson` | 9773747 | `a4bd4c6f61943e8b…` |
| AMECO | 330 files, one per country and series | not listed |
