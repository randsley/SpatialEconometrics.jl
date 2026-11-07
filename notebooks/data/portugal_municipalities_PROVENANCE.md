# Portuguese municipalities: provenance and attribution

Two sources, each with a credit that is a condition of its licence.
Keep both with any published result.

> Source: Statistics Portugal (INE), indicators 0014580, 0014581, 0014582, 0013189, 0012909, reference period 2023, extracted 2026-11-07.
>
> Informação geográfica cedida pela Direção-Geral do Território (CAOP, CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/)

INE data is open for reuse, commercial and non-commercial, under EU
Directive 2019/1024. INE retains copyright: the data is openly licensed,
not public domain, and reuse is conditional on citing the source.
Boundaries are **CAOP2025** from Direção-Geral do Território, licensed
**CC-BY-4.0** with public access unrestricted.

## Files

| file | units | contents |
|---|---|---|
| `portugal_municipalities.csv` | 308 | attributes, one row per municipality |
| `portugal_municipalities_queen.gal` | 278 | contiguity by shared vertex, 739 links |
| `portugal_municipalities_rook.gal` | 278 | contiguity by shared edge, 737 links |
| `portugal_municipalities_centroids.csv` | 278 | area-weighted centroids, WGS84 |

Everything is keyed and ordered by **DICO**, the official
district-and-municipality code, so the files line up by sorting alone.
The key is the DICO and not INE's NUTS code, which changes between
vintages, nor the name, because `Calheta` names two municipalities.

## Columns

| column | indicator | unit |
|---|---|---|
| `dico` | official district-and-municipality code | 4 digits |
| `name` | municipality name | |
| `district`, `district_name` | from the first two DICO digits | 1-18 mainland |
| `mainland` | whether district <= 18 | |
| `purchasing_power` | 0014580 | index, Portugal = 100 |
| `pp_share_national` | 0014581 | % of national total |
| `pp_dynamism` | 0014582 | relative factor |
| `density` | 0013189 | inhabitants / km2 |
| `ageing_ratio` | 0012909 | persons 65+ per 100 aged 0-14 |

Full indicator titles, as INE states them:

- **0014580** - Per capita purchasing power by Geographic localization (NUTS - 2024); Biennial - Statistics Portugal, Study on the local purchasing power
- **0014581** - Proportion of purchasing power (% - in the total of Country) by Geographic localization (NUTS - 2024); Biennial - Statistics Portugal, Study on the local purchasing power
- **0014582** - Dynamism relative factor of purchasing power (-) by Geographic localization (NUTS - 2024); Biennial - Statistics Portugal, Study on the local purchasing power
- **0013189** - Population's density (No./ km²) by Place of residence (NUTS - 2024); Annual - Statistics Portugal, Annual estimates of resident population
- **0012909** - Ageing ratio (No.) by Place of residence (NUTS - 2024); Annual - Statistics Portugal, Annual estimates of resident population

## Three things to know before modelling

**The islands have no land neighbours.** Of the 308 municipalities, 30 are
in Madeira and the Azores. That is why the weights files cover the 278
mainland units and the CSV covers all 308: a contiguity matrix over
every row would have 30 zero rows and could not be
row-standardised. Filter on `mainland`, or use a k-nearest or
distance-based weighting that gives every unit neighbours. The column is
here so the choice is explicit rather than discovered in a singular matrix.

**`purchasing_power`, `pp_share_national` and `pp_dynamism` come from the
same INE study** and are not independent of one another. Use one as a
dependent variable; do not put two of them on opposite sides of a
regression and read the fit as a finding.

**The GAL files are row-standardised on read, not on disk.** They hold
binary contiguity. Standardise after any further subsetting, never
before: dropping a unit changes its neighbours' degrees, so a
standardised matrix is only valid for the exact set it was computed over.

## Queen against rook

The two differ by exactly **2 pairs**, meeting at a single point rather than
along a border:

- Mêda and Sernancelhe
- Lisboa and Vila Franca de Xira

Mean degree is 5.317 under queen and 5.302 under rook. The choice is
usually presented as consequential; on these boundaries it moves 2 links
out of 739.

Contiguity was computed by hashing vertices and edges, not by
intersecting polygons, because CAOP is a clean planar partition: a shared
border carries identical vertices in both rings. That was measured, not
assumed -- hashing exact `Float64` pairs and hashing coordinates rounded
to six decimals give the same graph. No geometry engine is involved and
the result is exact rather than tolerance-dependent.

## How this was produced

`scripts/build_notebook_dataset.jl` in
[PortugalINE.jl](https://github.com/randsley/PortugalINE.jl), in one run.
It fails rather than writes if an indicator is not published to municipal
level, if any indicator returns other than 308 municipalities, if the
indicators disagree about which municipalities exist, if the boundaries
are not the 278 mainland units, or if any unit ends up with no neighbours.

Values are written back in the form INE published them, so a whole
number stays a whole number.

One trap worth repeating: indicator `0008258` is also "Ageing ratio"
with identical values, but on the **NUTS 2013** geography (344 codes
against 347). Joining it to the NUTS 2024 indicators would misalign
silently, so `0012909` is used instead. Check the geography vintage named
in an indicator's title before joining it to anything.
