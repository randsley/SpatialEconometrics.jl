# Portuguese municipalities: provenance and attribution

> Source: Statistics Portugal (INE). Indicators 0014580, 0014581, 0014582, 0013189, 0012909, reference period S7A2023 (2023),
> extracted 2026-11-06.

INE data is open for reuse, commercial and non-commercial, under EU
Directive 2019/1024. INE retains copyright: the data is openly
licensed, not public domain, and reuse is conditional on citing the
source. The quoted block above is that citation; keep it with any
published result built on these figures.

## Contents

`portugal_municipalities.csv`: 308 municipalities (278 mainland, 30 island), one row each.

| column | indicator | unit |
|---|---|---|
| `dico` | official district-and-municipality code | 4 digits |
| `name` | municipality name | |
| `district`, `district_name` | district, from the first two DICO digits | 1-18 mainland |
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

## Two things to know before modelling

**The islands have no land neighbours.** Of the 308 municipalities, 30 are in Madeira and the Azores. In a
contiguity weights matrix over all of them those rows are entirely
zero, and row-standardising divides by zero. Filter on `mainland` for
the 278-municipality contiguous set, or use a k-nearest or
distance-based weighting that gives every unit neighbours. The column
is here so the choice is explicit rather than discovered in a singular
matrix.

**`purchasing_power`, `pp_share_national` and `pp_dynamism` come from
the same INE study** and are not independent of one another. Use one as
a dependent variable; do not put two of them on opposite sides of a
regression and read the fit as a finding.

## No weights matrix yet

This extract is attributes only. A contiguity `W` needs boundaries,
which come from INE's BGRI, and BGRI is not served by the indicator
API -- it is a separate download from INE's geospatial portal. Until
`PortugalINE.jl` layer 3 reads it, build `W` from coordinates or
supply your own.

## How this was produced

`scripts/build_spatial_dataset.py` in `PortugalINE.jl`, which pulls
through `pyptine`. Each indicator is two-dimensional (period x
geography), so no dimension flattening is involved, and the script
fails rather than writes if any indicator returns other than
308 municipalities, if any DICO is non-numeric or duplicated,
or if the municipality names disagree across indicators.

Indicator `0008258` is also "Ageing ratio" with identical values, but
on the **NUTS 2013** geography (344 codes against 347). Joining it to
the NUTS 2024 indicators would misalign silently, so `0012909`, the
NUTS 2024 series, is used instead. Check the geography vintage named
in an indicator's title before joining it to anything.
