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

`data/columbus.csv` and `data/columbus.gal` are the Columbus, Ohio 1980
neighbourhood crime data: 49 contiguous planning neighbourhoods, with crime
rate, household income and housing value, plus the first-order contiguity
structure in GeoDa GAL format.

Taken from the [libpysal](https://github.com/pysal/libpysal) example
collection, which carries the values published in:

> Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Dordrecht:
> Kluwer Academic.

OLS on these data reproduces the published result,
`CRIME = 68.619 − 1.597·INC − 0.274·HOVAL` with R² = 0.5524, which is a useful
first check that your environment is working.
