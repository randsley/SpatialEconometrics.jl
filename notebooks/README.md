# Notebooks

Two kinds of notebook live here. The numbered `.ipynb` files are linear
walkthroughs meant to be read top to bottom. The `interactive_*.jl` and
`teaching_*.jl` files are [Pluto](https://plutojl.org) notebooks with sliders,
where the point is to change something and watch the consequences.

## Running them

**Jupyter** (`.ipynb`) needs IJulia:

```julia
using Pkg; Pkg.add("IJulia")
using IJulia; notebook(dir = "notebooks")
```

**Pluto** (`.jl`):

```julia
using Pkg; Pkg.add("Pluto")
using Pluto; Pluto.run()
```

Each notebook activates the package environment itself, so no setup is needed
beyond the two commands above. `interactive_extensions_explorer.jl` is the
exception: it builds a temporary environment and installs Makie and Turing,
which takes a few minutes the first time.

## Worked analyses

| Notebook | What it covers |
|---|---|
| `01_getting_started.ipynb` | Weights, a first SAR and SDM, effects |
| `02_spatial_panels.ipynb` | Static and dynamic panels, fixed effects, SEM |
| `03_gwr_gravity.ipynb` | Geographically weighted regression, gravity models |
| `04_columbus_crime.ipynb` | **Real data.** The Anselin (1988) Columbus analysis end to end |
| `05_spatial_heterogeneity.ipynb` | **Real data.** Regimes, Chow tests, SKATER regionalisation |
| `06_spatial_causal.ipynb` | Spatial DiD with exposure, synthetic control with neighbour exclusion, shift-share, peer effects and the reflection problem |
| `07_space_time.ipynb` | STAR, spatio-temporal impulse responses, SpVAR with a causal ordering |
| `08_networks_and_factors.ipynb` | Common factors: the CD test, the alpha exponent, CCE, spatio-temporal principal components |
| `09_model_selection.ipynb` | Information criteria, Bayesian specification probabilities, convex combinations of weights |
| `10_diagnostics_and_robust_inference.ipynb` | The full LM battery, and Driscoll-Kraay and Conley standard errors checked against a sampling distribution |
| `11_portugal_municipalities.ipynb` | **Real data.** 278 Portuguese municipalities: the battery, SAR/SEM/SDM, a coastal/interior regime split that rejects, and a Conley bandwidth sweep |
| `12_europe_regions.ipynb` | **Real data.** 223 EU NUTS-2 regions, growth 2012-2022: β-convergence that a spatial lag removes, country dummies that flip it to divergence, an old/new member-state regime split |
| `13_europe_countries.ipynb` | **Real data.** 30 European economies, 1996-2025, from AMECO: spillovers against common shocks, with ρ̂ from −0.04 to 0.70 depending on the treatment, and a coefficient that does not move |

`06` to `10` use simulated data throughout and print the data-generating
process before each fit, so every number has a known answer to be read against.
That is deliberate: these five cover the parts of the package where a plausible
wrong answer is hardest to spot, and three of them turn on a result that
contradicts the obvious reading.

`11` is the counterpart on real data at a sample size that supports it, and it
is worth reading against `04` and `05`, which run the same tests on 49
Columbus neighbourhoods. The LM battery settles nothing on Columbus and
predicts the right model here; the Chow test fails to reject on Columbus's
north/south split and rejects on Portugal's coastal/interior one. Same tests,
same package, opposite conclusions — which is the most useful thing a pair of
notebooks can show.

`12` and `13` take the same tools across borders, and both turn on the same
point: in Europe, much of what a spatial lag finds is national or common to
everyone. In `12`, country dummies explain 60% of regional growth and turn ρ̂
from +0.58 to −0.23. In `13`, year effects and common correlated effects take
ρ̂ from 0.70 to 0.48 and then to −0.04, and the last estimator fails its own
Sargan test. `13` is the real-data counterpart of the simulation in `08`.

## Interactive

| Notebook | What you can vary |
|---|---|
| `teaching_lm_battery.jl` | ρ, λ, lattice size, noise — watch the specification test recover (or fail to recover) the true process |
| `interactive_spatial_explorer.jl` | The basic SAR fit |
| `interactive_higher_order_explorer.jl` | SAR vs SDM, higher-order structures |
| `interactive_gwr_explorer.jl` | GWR bandwidth and kernel |
| `interactive_probit_explorer.jl` | Spatial probit |
| `interactive_filtering_explorer.jl` | Eigenvector spatial filtering |
| `interactive_extensions_explorer.jl` | The Makie and Turing extensions |

## Using these in teaching

`04`, `05` and `teaching_lm_battery.jl` were written with a course in mind and
each ends with exercises. A sequence that works:

1. **`teaching_lm_battery.jl`** first, before any real data. Students generate
   data from a process they choose, so they can see what the test is supposed to
   recover before being asked to trust it on data where the answer is unknown.
   The exercise that matters most is setting ρ = λ = 0 and counting rejections
   across seeds.
2. **`04_columbus_crime.ipynb`** next, on the dataset the field has used since
   1988. The robust lag statistic there lands just short of the 5% critical
   value, which makes it a better discussion than a clean verdict would have
   been.
3. **`05_spatial_heterogeneity.ipynb`** last. Its regime table looks like a
   finding and the Chow test says it is not, which is the point.

The exercises are deliberately open at the end — several have no single right
answer, and a few (the power question in `05`, the weight-matrix question in
`04`) are worth a seminar rather than a solution.

## Data

Three real datasets, all in `data/`.

### Portuguese municipalities, 2023

`portugal_municipalities.csv` carries **308 municipalities**
(278 mainland, 30 in Madeira and the Azores) with per capita purchasing power,
its share of the national total, the purchasing-power dynamism factor,
population density and the ageing ratio, all for 2023.

It is a better teaching set than Columbus in every dimension that matters.
Columbus has 49 units; this has 278 contiguous ones, which is enough for the
asymptotics the estimators assume to mean something. Its coastal/interior split
is a regime boundary that corresponds to something, where Columbus offers a
north/south dummy on which the Chow test declines to reject. And purchasing
power across Portuguese
municipalities is spatially dependent for reasons nobody has to invent:
Lisboa at 181.4 and Porto at 162.2 against a municipal mean of 82.5.

Two of the warnings in `portugal_municipalities_PROVENANCE.md` are worth
repeating here. The 30 island municipalities share no land border with anything, so
a contiguity `W` over all 308 has 30 entirely zero rows and row-standardising
divides by zero — filter on the `mainland` column, or use a weighting that
gives every unit neighbours. And `purchasing_power`, `pp_share_national` and
`pp_dynamism` all come from the same INE study, so they are not independent of
one another.

The weights are here too. `portugal_municipalities_queen.gal` and
`..._rook.gal` are GeoDa GAL contiguity files over the 278 mainland
municipalities, and `..._centroids.csv` has their area-weighted centroids for
distance- or neighbour-based weightings. `read_gal` reads them directly and
row-standardises on the way in; the files themselves hold binary contiguity.
The DICO ordering matches the attribute CSV once both are sorted, so no lookup
table is needed.

Queen has 739 links and rook 737: they differ by exactly two pairs, Mêda with
Sernancelhe and Lisboa with Vila Franca de Xira, which meet at a single point
rather than along a border. Mean degree is 5.32, the minimum is 1 (Nazaré) and
no municipality is isolated.

The spatial dependence is real and strong, not a teaching fiction: Moran's I on
purchasing power is **0.445** with z = 12.1, and still **0.336** on the OLS
residuals, so it is not merely inherited from density and age. The LM battery
rejects both ways (lag 40.0, error 77.6) and the robust pair points at the
error model five times more firmly than the lag (robust lag 8.8, p 0.003;
robust error 46.4, p 1e-11). The likelihood then agrees: SEM beats SAR by 34
AIC points. A SAR fit gives ρ = 0.379, log density +3.57 and the ageing ratio
−0.0244 per point of the ratio, or about −2.4 index points per hundred:
denser municipalities richer, older ones poorer.

`11_portugal_municipalities.ipynb` works all of that through.

All five files come from one run of `scripts/build_notebook_dataset.jl` in
[PortugalINE.jl](https://github.com/randsley/PortugalINE.jl): indicators from
INE's API, boundaries from DGT's CAOP, joined on DICO. Full provenance,
including every indicator's title and the reasons behind each choice, is in
`portugal_municipalities_PROVENANCE.md`.

Two licences apply and each requires its credit, so keep both with any
published result:

> Source: Statistics Portugal (INE), indicators 0014580, 0014581, 0014582,
> 0013189 and 0012909, reference period 2023, extracted 2026-11-07.
>
> Informação geográfica cedida pela Direção-Geral do Território

INE data is open for reuse under EU Directive 2019/1024 with copyright
retained, and CAOP2025 is CC-BY-4.0.

### European regions and countries

Two panels, built together by `data/build/build_europe.jl` and documented in
`europe_PROVENANCE.md`.

`europe_nuts2.csv` is a **regional panel of 247 NUTS-2 regions** (2024
classification) in the EU27, Norway and Switzerland, 1999-2025: GDP in euros
and in PPS per head, a real GDP volume index, population, employment,
unemployment and investment, from Eurostat's regional tables. The outermost
regions, Ceuta, Melilla and Svalbard are left out. The panel is unbalanced,
because NUTS 2024 redrew regions in several countries and some tables have no
back series for the new units; real growth over 2010-2022 is complete for 234
regions. Eleven regions are islands with no land neighbour, so the queen and
rook files have eleven zero rows: drop them, or build k-nearest-neighbour weights from
the EPSG:3035 centroids in `europe_nuts2_regions.csv`.

`europe_ameco.csv` is a **country panel of 30 countries** (the EU27, Norway,
Switzerland and Iceland) from the Commission's AMECO database: GDP, real
GDP, population, employment, unemployment, investment, compensation, capital
stock and trade, with the Commission's forecast years removed. Thirty units
suit a static spatial panel, not methods whose asymptotics need many units.

`12_europe_regions.ipynb` uses the regional panel and `13_europe_countries.ipynb`
the country panel.

Three credits apply:

> Source: Eurostat, regional statistics, extracted 2026-11-09.
>
> Source: European Commission, DG ECFIN, AMECO database, 2026 release,
> extracted 2026-11-09.
>
> Boundaries: © OpenStreetMap contributors, available under the Open
> Database License (ODbL)

Eurostat and AMECO values may be reused, commercially or not, with the
source acknowledged. The five files derived from OpenStreetMap boundaries
(the two centroid files and the three `.gal` files) are offered under the
ODbL, which also permits commercial use: credit OpenStreetMap, and offer any
adapted version you publish under the ODbL too. The package's MIT licence
does not extend to them.

### Columbus, Ohio, 1980

`columbus.csv` and `columbus.gal` are the Columbus, Ohio 1980 neighbourhood
crime data: 49 contiguous planning neighbourhoods, with crime rate, household
income and housing value, plus the first-order contiguity structure in GeoDa
GAL format. Smaller and older than the Portuguese set, and kept because it is
the dataset the field has used since 1988 and the one with a published result
to check against.

Taken from the [libpysal](https://github.com/pysal/libpysal) example
collection, which carries the values published in:

> Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Dordrecht:
> Kluwer Academic.

OLS on these data reproduces the published result,
`CRIME = 68.619 − 1.597·INC − 0.274·HOVAL` with R² = 0.5524, which is a useful
first check that your environment is working.
