# portugal_parishes_panel: provenance and attribution

> Source: Statistics Portugal (INE), indicators 0008234, 0011866, 0012748, reference periods 2018 to 2023, extracted 2026-11-26. Informação geográfica cedida pela Direção-Geral do Território (CAOP, CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/)

INE data is open for reuse, commercial and otherwise, under EU Directive
2019/1024. INE retains copyright, and the one condition is that the source
be cited. CAOP is CC-BY-4.0 and the credit above is a condition of that
licence. Keep both with any published result.

Written 2026-11-26 by `write_panel` in PortugalINE.jl.

## Units and periods

**2882 parishes x 6 periods** (2018, 2019, 2020, 2021, 2022, 2023), keyed by DICOFRE. The CSV is long: one row per unit and period, ordered by period and then by DICOFRE. The same units are present in every period, so one weights matrix serves them all.

| dropped | count |
|---|---|
| boundaries missing from some indicator or period | 0 |
| island units with no boundary | 210 |
| units of another map (no value, no boundary) | 302 |
| other units with values and no boundary | 0 |

## Columns

| column | INE indicator | filter | language | extracted | missing (of 17292) |
|---|---|---|---|---|---|
| `income` | 0012748 | none | EN | 2026-11-26 | 0 |
| `births` | 0008234 | Dim3=T | EN | 2026-11-26 | 0 |
| `area_km2` | 0011866 | none | EN | 2026-11-26 | 0 |

## Boundaries

| | |
|---|---|
| CAOP edition | CAOP2024 |
| file | `caop2024_freguesias.json` |
| SHA-256 | `dd6be6d0cf4b741a4833e94dae749b5d18c97a9c7335745796e18a15cd30120d` |

## Weights

Queen contiguity: a shared vertex is enough.
**8409 links**, mean degree 5.836, minimum 1, maximum 17. The GAL file is binary and not
row-standardised: standardise after any further subsetting, never before.

## What the join noticed

- 210 island parishes have data but no boundary, which is expected: CAOP serves Madeira and the Azores separately and they have no land neighbours
- 302 parishes listed in the data have no value in any period used and no boundary: units of another map (INE's area indicator 0011866 lists the 302 parishes restored in 2025 alongside the 2013 map)
- 18 (unit, column, period) names differ from the first spelling, which is kept. The DICOFRE join is unaffected

## The columns

| column | INE indicator | unit |
|---|---|---|
| `income` | 0012748 | EUR per taxpayer (gross declared income, mean) |
| `births` | 0008234 | live births, both sexes (count) |
| `area_km2` | 0011866 | km2 |

Titles, as INE states them in Portuguese:

- **0012748** - Rendimento bruto declarado por sujeito passivo (€)
- **0008234** - Nados-vivos (N.º)
- **0011866** - Superfície (km²) das unidades territoriais

## Things to know before modelling

**Why 2018-2023, and why these.** On 2026-11-26 only two parish-level
indicators covered all 2,882 mainland parishes in more than one year on
the 2013 map: income (2018-2024) and births (2014-2023). The years are
their overlap. INE publishes no annual population by parish; the
Census 2021 is the only parish count, so births are a count, not a rate.

**The parish map is the one of the 2013 reform.** Law 25-A/2025 restored
302 mainland parishes; the current CAOP carries 3,049. INE's area
indicator lists both maps, and its restored parishes, empty before 2025,
are counted above as units of another map. Do not join these files to a
current CAOP layer.

**Births abroad are not a parish.** INE's births carry `000000000`,
"Estrangeiro", which PortugalINE.jl leaves out.

Produced by `scripts/build_parish_panel.jl` in
[PortugalINE.jl](https://github.com/randsley/PortugalINE.jl). It fails
rather than writes if an indicator is not at parish level, lacks a year,
or leaves the panel short of 2882 parishes, or if a parish has no
neighbours.
