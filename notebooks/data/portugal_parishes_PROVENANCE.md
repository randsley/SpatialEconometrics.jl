# Portuguese parishes: provenance and attribution

Two sources, each with a credit that is a condition of its licence.
Keep both with any published result.

> Source: Statistics Portugal (INE), indicators 0011866, 0012273, 0012276, 0012328, 0012514, 0012748, reference periods 2021, 2024, extracted 2026-11-25.
>
> Informação geográfica cedida pela Direção-Geral do Território (CAOP, CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/)

INE data is open for reuse, commercial and non-commercial, under EU
Directive 2019/1024. INE retains copyright: reuse is conditional on
citing the source. The boundaries are **CAOP2024** (mainland) from
Direção-Geral do Território, licensed **CC-BY-4.0**, read from `caop2024_freguesias.json`, SHA-256 `dd6be6d0cf4b741a4833e94dae749b5d18c97a9c7335745796e18a15cd30120d`.

## Files

| file | contents |
|---|---|
| `portugal_parishes.csv` | 2882 mainland parishes, attributes |
| `portugal_parishes_queen.gal` | contiguity by shared vertex, 8409 links |
| `portugal_parishes_rook.gal` | contiguity by shared edge, 8283 links |
| `portugal_parishes_centroids.csv` | area-weighted centroids, EPSG:3763 (PT-TM06), metres |

Everything is keyed and ordered by **DICOFRE** (CAOP's `dtmnfr`), the
official district, municipality and parish code, so the files line up by
sorting alone. Its first four characters are the municipality's DICO.
It is text, not a number: the 8 unions of Barcelos are
coded 0302FA, 0302FB, 0302FC, 0302FD, 0302FE, 0302FF, 0302FG, 0302FH, and leading zeros matter.

## Columns

| column | INE indicator | period | filter | unit |
|---|---|---|---|---|
| `dicofre` | official parish code | | | 6 characters |
| `name` | parish name, as INE gives it | | | |
| `dico`, `district`, `district_name` | from the DICOFRE | | | |
| `income_2024` | 0012748 | 2024 | none | EUR per taxpayer (gross declared income) |
| `income_2021` | 0012748 | 2021 | none | EUR per taxpayer (gross declared income) |
| `unemployment` | 0012328 | 2021 | Dim3=T | % (Census 2021, both sexes) |
| `illiteracy` | 0012273 | 2021 | Dim3=T | % (Census 2021, both sexes) |
| `population` | 0012276 | 2021 | Dim3=T, Dim4=T | persons (Census 2021) |
| `quarters_density` | 0012514 | 2021 | none | living quarters per km2 (Census 2021) |
| `area_km2` | 0011866 | 2021 | none | km2 |

Full indicator titles, as INE states them:

- **0011866** - Area (km²) of territorial units by Geographic localization (NUTS - 2024); Annual - Directorate-General of Territorial Development
- **0012273** - Illiteracy rate (%) by Place of residence at Census date [2021] (NUTS - 2024) and Sex; Decennial - Statistics Portugal, Population and housing census - 2021
- **0012276** - Resident population (No.) by Place of residence at Census date [2021] (NUTS - 2024), Sex and Age group (By life cycles); Decennial - Statistics Portugal, Population and housing census - 2021
- **0012328** - Unemployment rate (%) by Place of residence at Census date [2021] (NUTS - 2024) and Sex; Decennial - Statistics Portugal, Population and housing census - 2021
- **0012514** - Living quarters density (No./ km²) by Geographic location at Census date [2021] (NUTS - 2024); Decennial - Statistics Portugal, Population and housing census - 2021
- **0012748** - Gross reported income per taxable person (€) by Geographic localization (NUTS - 2024); Annual - Statistics Portugal, Income Statistics at local level produced by Ministry of Finance - Tax and Customs Authority

## Things to know before modelling

**The parish map is the one of the 2013 reform.** Law 25-A/2025
restored parishes that the 2013 reform had merged; the current CAOP
(2024.1 and 2025) has 3,049 mainland parishes. INE's parish figures, up
to the 2024 reference period, are on the 2013 map, and so are these
boundaries. Do not join these files to a current CAOP layer.

**The income column is the mean, not the median.** INE's median
declared income net of tax (0012757) is suppressed for most parishes
(a value for 735 of the 2,882 in 2024). The mean gross declared income
per taxpayer (0012748) covers all of them.

**`area_km2` drops 302 empty codes.** INE lists both parish maps
in that indicator's geography; the 302 parishes restored in 2025
have no value for the period used and are not in these files.

**The GAL files are binary.** Standardise after any further subsetting,
never before.

## Weights

Queen: 8409 links, mean degree 5.836 (min 1, max 17). Rook: 8283 links.
They differ by 126 pairs that meet at a single point. No parish
is without a neighbour.

Contiguity was computed by hashing shared vertices and edges, which is
exact on a node-matched planar partition. That was checked on this layer:
rounding every vertex to the millimetre leaves both graphs unchanged.
Checked once more on 2026-11-25 against an independent geometry engine:
shapely 2.1 (boundaries that intersect, and intersect in a line) gives
the same 8,409 queen and 8,283 rook links, link for link.

## How this was produced

`scripts/caop_gpkg_to_geojson.py` (Python standard library only; the
coordinates are copied as stored) and `scripts/build_parish_dataset.jl`
in [PortugalINE.jl](https://github.com/randsley/PortugalINE.jl), on 2026-11-26.
The script fails rather than writes if an indicator is not published to
parish level, if any indicator returns other than 2882 mainland parishes,
if the indicators or the boundaries disagree about which parishes exist,
or if any parish has no neighbours.
