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

`06` to `10` use simulated data throughout and print the data-generating
process before each fit, so every number has a known answer to be read against.
That is deliberate: these five cover the parts of the package where a plausible
wrong answer is hardest to spot, and three of them turn on a result that
contradicts the obvious reading.

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

Two real datasets, both in `data/`.

### Portuguese municipalities, 2023

`portugal_municipalities.csv` carries **308 municipalities**
(278 mainland, 30 in Madeira and the Azores) with per capita purchasing power,
its share of the national total, the purchasing-power dynamism factor,
population density and the ageing ratio, all for 2023.

It is a better teaching set than Columbus in every dimension that matters.
Columbus has 49 units; this has 278 contiguous ones, which is enough for the
asymptotics the estimators assume to mean something. It has a real
coastal/interior split, which the regimes, Chow and SKATER examples currently
demonstrate on simulated data. And purchasing power across Portuguese
municipalities is spatially dependent for reasons nobody has to invent:
Lisboa at 181.4 and Porto at 162.2 against a municipal mean of 82.5.

Two warnings, both in `portugal_municipalities_PROVENANCE.md` and both worth
repeating. The 30 island municipalities share no land border with anything, so
a contiguity `W` over all 308 has 30 entirely zero rows and row-standardising
divides by zero — filter on the `mainland` column, or use a weighting that
gives every unit neighbours. And `purchasing_power`, `pp_share_national` and
`pp_dynamism` all come from the same INE study, so they are not independent of
one another.

**There is no `W` in this extract yet.** A contiguity matrix needs boundaries,
which come from INE's BGRI and are a separate download from the indicator API.
Until then, supply your own or build one from coordinates.

Produced by `scripts/build_spatial_dataset.py` in
[PortugalINE.jl](https://github.com/randsley/PortugalINE.jl), pulled through
[pyptine](https://github.com/randsley/pyptine). The data is open for reuse
under EU Directive 2019/1024, INE retains copyright, and reuse is conditional
on citing the source:

> Source: Statistics Portugal (INE), indicators 0014580, 0014581, 0014582,
> 0013189 and 0012909, reference period 2023, extracted 2026-11-06.

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
