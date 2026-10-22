# SpatialEconometrics.jl

A spatial and structural econometrics suite in Julia, built around sparse
linear algebra, multiple dispatch and automatic differentiation. It covers
cross-sectional, panel and higher-order spatial models, spatial discrete
choice, eigenvector spatial filtering, regime models, spatial and panel
diagnostics, model selection, geographically weighted regression, gravity
models, HAC-robust inference, weight-matrix construction, input-output
calibration, Monte Carlo simulation and publication-ready output — with a
unified formula interface and optional Bayesian and visualization extensions.

> **Version 0.3.0** (registration in review) — 492 test assertions across 26
> testsets (527 with the two package extensions installed), passing on Julia
> 1.13.2. See [Validation Status](#validation-status) for Monte Carlo results.

**Documentation:** [dev](https://randsley.github.io/SpatialEconometrics.jl/dev/)
— a `stable` build appears with the first tagged release.

Every exported name carries a docstring, so `?fit_spatial` works in the
REPL. All 187 of them appear in the API reference — the 188th, `@formula`,
is re-exported from StatsModels.jl and documented there. The docs build runs
with no `warnonly` exemptions, so a missing docstring or a broken
cross-reference fails CI.

---

## Installation

```julia
using Pkg
Pkg.add("SpatialEconometrics")
```

> Registration in the General registry is still in review, so `Pkg.add` by name
> does not resolve yet. Until it merges, install from the repository:
>
> ```julia
> using Pkg
> Pkg.add(url = "https://github.com/randsley/SpatialEconometrics.jl")
> ```

To work on the package itself, develop a local checkout and run the suite:

```julia
using Pkg
Pkg.develop(path = "path/to/SpatialEconometrics")
Pkg.test("SpatialEconometrics")
```

**Dependencies:** `ChainRulesCore`, `ForwardDiff`, `NearestNeighbors`,
`StatsAPI`, `StatsModels`, `Tables`, plus standard libraries
(`LinearAlgebra`, `SparseArrays`, `Statistics`, `Random`, `Printf`). Julia
1.10 or later.

**Optional extensions:** `Turing.jl` + `Distributions.jl` (Bayesian),
`Makie.jl` (visualization) — loaded automatically when those packages are
available. See
[Package Extensions](https://randsley.github.io/SpatialEconometrics.jl/dev/extensions/).

---

## Quick Start

```julia
using SpatialEconometrics, Random, LinearAlgebra, SparseArrays

# Create a spatial weight matrix (20×20 grid → 400 units)
W = make_rook_weights(20)
# Or: W = make_queen_weights(15)

# Simulate data
rng = MersenneTwister(42); N = 400
X = randn(rng, N, 2)
y = (I - 0.5 * W) \ (2.0 .+ X * [1.5, -0.8] .+ 0.5 .* randn(rng, N))

# Step 1: Diagnostics — is there spatial autocorrelation?
println(moran_global(y, W))
lms = lm_tests(y, hcat(ones(N), X), W)
println(lms.robust_lag)    # 278.55, p < 1e-4
println(lms.robust_error)  # 4.43,  p = 0.035
# Both reject at 5%, so read them against each other: the robust lag
# statistic is two orders of magnitude larger, which points to SAR.

# Step 2: Estimate
m = fit_spatial(SAR(), W, X, y;
    method = ChebyshevApprox(order=12),
    varnames = ["income", "density"])
println(m)

# Step 3: Spatial effects
println(spatial_effects(m))
```

Nineteen worked examples live in [`examples/`](examples/), one per topic,
each runnable on its own.

---

## Documentation

The manual carries the per-model material — specifications, estimation
options and how to read the output — and the API reference documents every
exported name.

| Topic | |
|---|---|
| [Getting Started](https://randsley.github.io/SpatialEconometrics.jl/dev/getting_started/) | Specification tags, submodules, a first model |
| [Weight Matrices](https://randsley.github.io/SpatialEconometrics.jl/dev/weights/) | k-NN, distance, band, standardization, GeoDa GAL/GWT I/O |
| [Cross-Sectional](https://randsley.github.io/SpatialEconometrics.jl/dev/models/cross_sectional/) | SAR and SDM by concentrated ML, effects decomposition |
| [Panel](https://randsley.github.io/SpatialEconometrics.jl/dev/models/panel/) | Static spatial panels, Lee-Yu and direct QML |
| [Dynamic Panel](https://randsley.github.io/SpatialEconometrics.jl/dev/models/dynamic_panel/) | Time lag plus spatial lag, bias correction |
| [Spatial Error (SEM)](https://randsley.github.io/SpatialEconometrics.jl/dev/models/sem/) | Dependence in the disturbance |
| [GS2SLS / SARAR](https://randsley.github.io/SpatialEconometrics.jl/dev/models/gs2sls/) | Endogenous regressors and instruments |
| [Dynamic Panel GMM](https://randsley.github.io/SpatialEconometrics.jl/dev/models/gmm/) | Arellano-Bond, Blundell-Bond, Hansen and AR tests |
| [Higher-Order](https://randsley.github.io/SpatialEconometrics.jl/dev/models/higher_order/) | SLX, SAC/SARAR, SDEM |
| [Regime Models](https://randsley.github.io/SpatialEconometrics.jl/dev/models/regime/) | Coefficients varying by group, SKATER regionalization |
| [Discrete Choice](https://randsley.github.io/SpatialEconometrics.jl/dev/models/discrete_choice/) | Spatial probit by linearized GMM and Gibbs sampling |
| [Spatial Filtering](https://randsley.github.io/SpatialEconometrics.jl/dev/models/filtering/) | ESF and RE-ESF |
| [GWR & MGWR](https://randsley.github.io/SpatialEconometrics.jl/dev/models/gwr/) | Coefficient surfaces, multiscale backfitting |
| [Gravity Models](https://randsley.github.io/SpatialEconometrics.jl/dev/models/gravity/) | PPML and negative binomial with high-dimensional FE |
| [Spatial Diagnostics](https://randsley.github.io/SpatialEconometrics.jl/dev/diagnostics/spatial/) | Moran's I, Geary's c, LISA, Anselin LM battery |
| [Panel Diagnostics](https://randsley.github.io/SpatialEconometrics.jl/dev/diagnostics/panel/) | Spatial Hausman, Chow, joint panel LM |
| [Model Selection](https://randsley.github.io/SpatialEconometrics.jl/dev/diagnostics/model_selection/) | AIC/BIC, LR, Vuong, Wald, common factor |
| [Robust Inference](https://randsley.github.io/SpatialEconometrics.jl/dev/robust/) | Driscoll-Kraay and Conley spatial HAC |
| [Simulation](https://randsley.github.io/SpatialEconometrics.jl/dev/simulation/) | Six DGPs, Monte Carlo, power analysis |
| [Calibration & Input-Output](https://randsley.github.io/SpatialEconometrics.jl/dev/calibration/) | GRAS, cross-entropy, SAM multipliers |
| [Formula Interface](https://randsley.github.io/SpatialEconometrics.jl/dev/formula/) | `slag`, `fe`, `endo`, panel ordering |
| [Output & Tables](https://randsley.github.io/SpatialEconometrics.jl/dev/tables/) | Coefficient tables, LaTeX / CSV / Markdown |
| [API Reference](https://randsley.github.io/SpatialEconometrics.jl/dev/api/) | Every exported name, by topic |

---

## Modules

The package exports functions, result types and specification tags — **not**
submodules. Model variants (`SAR`, `SDM`, `SARAR`), fixed-effects specs
(`NoFE`, `IndivFE`, `TimeFE`, `TwoWayFE`) and log-determinant methods
(`ExactLU`, `ChebyshevApprox`) are single shared types, usable unqualified in
any call:

```julia
m = fit_spatial(SAR(), W, X, y)                              # cross-section
m = fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T_per)   # panel
```

The submodules below organise the source. They are not exported, so they
cannot collide with packages of the same name; reach one explicitly when you
want an internal helper:

```julia
τ = SpatialEconometrics.CrossSection.precompute_chebyshev_traces(W,
        ChebyshevApprox(order = 12))
```

| Submodule | Description | Key Reference |
|---|---|---|
| `SpatialUtils` | Shared numerical utilities (Chebyshev log-det, Brent optimizer, panel transforms) | — |
| `SpatialEconometricsPrototype` | Cross-sectional SAR & SDM by concentrated ML | LeSage & Pace (2009) |
| `SpatialPanelEconometrics` | Static spatial panels with FE, Lee-Yu & Direct QML | Lee & Yu (2010) |
| `DynamicSpatialPanelEconometrics` | Dynamic spatial panels with bias correction | Yu, de Jong & Lee (2008) |
| `SpatialGS2SLSPanel` | SARAR panels with endogenous regressors | Kelejian & Prucha (1998) |
| `DynamicPanelSystemGMM` | Arellano-Bond & Blundell-Bond GMM | Roodman (2009) |
| `StructuralMacroCalibration` | GRAS & cross-entropy SAM/MRIO balancing | Junius & Oosterhaven (2003) |
| `SpatialHAC` | Driscoll-Kraay & Conley HAC covariances | Conley (1999) |
| `SpatialDiagnostics` | Moran's I, LISA, LM tests, Geary's c | Anselin (1988) |
| `SpatialErrorModel` | Cross-section & panel SEM by concentrated ML | Anselin (1988) |
| `GWREstimation` | GWR & MGWR with KDTree spatial indexing | Fotheringham et al. (2002) |
| `SpatialGravityModels` | PPML & NB gravity with high-dim FE | Santos Silva & Tenreyro (2006) |
| `FormulaDSL` | `@formula` interface with `slag`, `fe`, `endo` | — |
| `SpatialWeights` | k-NN / distance / band weights, standardization, GeoDa I/O | Anselin & Rey (2014) |
| `SpatialModelSelection` | AIC/BIC, LR, Vuong, Wald, common factor tests | Vuong (1989) |
| `SpatialPanelDiagnostics` | Spatial Hausman, Chow, joint panel LM, panel Moran | Baltagi, Song & Koh (2003) |
| `HigherOrderSpatial` | SLX, SAC/SARAR, SDEM | Halleck Vega & Elhorst (2015) |
| `SpatialRegimeModels` | Regime models with SKATER regionalization | Assuncao et al. (2006) |
| `SpatialDiscreteChoice` | Spatial probit — linearized GMM and Albert-Chib Gibbs | Klier & McMillen (2008) |
| `SpatialFiltering` | Eigenvector spatial filtering (ESF) and RE-ESF | Griffith (2003) |
| `SpatialSimulation` | Six DGPs, Monte Carlo, power analysis | — |
| `SpatialTables` | Coefficient and regression tables, LaTeX / CSV / Markdown | — |

---

## Relationship to Other Packages

[SpatialDependence.jl](https://github.com/javierbarbero/SpatialDependence.jl)
is the Julia package for **exploratory** spatial data analysis: polygon
contiguity weights, Getis-Ord and local Geary statistics, significance
adjustment, and choropleth mapping. If your work is ESDA and mapping, use it
— this package does none of those things.

SpatialEconometrics.jl is an **estimation** package. The overlap is Moran's I,
Geary's c, local Moran and the k-NN / distance weight constructors. We keep
our own because:

- The LM specification tests (`lm_test_lag`, `lm_test_error` and their robust
  variants) have no counterpart there, and they are what tells you whether to
  fit SAR or SEM. They are regression diagnostics, so they would be out of
  scope upstream.
- Our estimators operate on `SparseMatrixCSC` directly — Chebyshev trace
  precomputation, `log|I - ρW|`, GS2SLS instruments. SpatialDependence.jl's
  `SpatialWeights` type exposes neighbours and weights per row rather than the
  underlying sparse matrix, so a dependency would mean a conversion layer and
  we would still hold our own matrix.
- `monte_carlo_spatial` and `power_analysis` call `moran_global` and the four
  LM tests inside replication loops using analytic normal inference.
  SpatialDependence.jl's statistics are permutation-based by default
  (`permutations = 9999`), which is both a different inferential basis and far
  too slow in that position.

---

## Repository Layout

```text
SpatialEconometrics/
├── src/            # 23 modules: the umbrella plus 22 submodules
├── ext/            # Makie and Turing package extensions
├── test/           # runtests.jl — 26 testsets
├── examples/       # 19 runnable examples, one per topic
├── docs/           # Documenter source for the manual and API reference
└── validation/     # Monte Carlo output referenced below
```

---

## Validation Status

| Module | Test Suite | Monte Carlo Validated |
|---|---|---|
| Cross-sectional SAR/SDM | ✓ | **Yes** (100 draws, N=400, ρ=0.6, 96% coverage) |
| Spatial Diagnostics | ✓ | Moran/LM correctly identifies SAR DGP |
| Spatial Error Model | ✓ | λ recovery within atol=0.12 |
| Static Spatial Panel | ✓ | ρ recovery within atol=0.12 |
| Dynamic Spatial Panel | ✓ | ρ, γ recovery within atol=0.15 |
| GS2SLS / SARAR | ✓ | Coefficient recovery, T_post correct |
| System GMM | ✓ | γ recovery, Hansen J, AR tests |
| GWR / MGWR | ✓ | R²>0.5, spatial variation detected |
| Gravity (PPML/NB) | ✓ | Distance coefficient negative, α>0 |
| GRAS / Cross-Entropy | ✓ | Convergence, margin matching |
| Driscoll-Kraay / Conley | ✓ | PSD matrices, SEs differ from baseline |
| Spatial Weights | ✓ | Row sums, connectivity, exact GAL/GWT round-trips |
| Model Selection | ✓ | χ² upper tail matches the exact closed form at even df |
| Panel Diagnostics | ✓ | F tail matches the exact integer-parameter beta reference |
| Higher-Order (SLX/SAC/SDEM) | ✓ | Coefficient recovery on matching DGPs |
| Spatial Regimes | ✓ | SKATER contiguity and balance; Chow test size and power |
| Simulation / Monte Carlo | ✓ | **Yes** (β bias 0.0004 / −0.0029, ρ bias −0.0081) |
| Spatial Discrete Choice | ✓ | **Yes** (GMM approx. unbiased, SD ≈ 0.25 at N=196; Gibbs posterior mean 0.374 at ρ=0.4) |
| Spatial Filtering | ✓ | **Yes** (E'E = I exactly; RE-ESF β and σ² recovery; closed-form Moran reference matches the full spectrum) |
| Output & Tables | ✓ | LaTeX / CSV / Markdown round-trips, Tables.jl interface |

**Cross-sectional SDM Monte Carlo results** (`validation/mc_sdm_output.txt`):
ρ̂ mean = 0.595 (true 0.600); mean SE = 0.0324, empirical SD = 0.0328; 95%
coverage = 0.96. Direct, indirect, and total effects coverage between
0.92–0.97.

---

## Known Limitations

- **Chebyshev order:** Default order 6 is adequate for moderate ρ. Use order 24+ for ρ > 0.85, or `ExactLU()`.
- **Stochastic traces:** Neumann traces use Hutchinson probes (250 by default). For small N, exact traces would be more precise.
- **MGWR convergence:** Backfitting may be slow with many variables. Monitor `m.converged` and `m.iterations`.
- **Gravity FE:** Bilateral FE demeaning uses alternating projections (iterative). Very large O-D matrices may need more iterations.
- **Type stability:** Some `0.0` literals will promote `Float32` to `Float64`. Use `Float64` consistently.
- **Panel data ordering:** Data must be stacked with N units varying fastest within each time period: `[unit₁_t₁, unit₂_t₁, ..., unitN_t₁, unit₁_t₂, ...]`.

---

## Development

This package was developed with assistance from a generative AI coding tool
(Anthropic's Claude). The AI was used to draft module code, the test
suite, the documentation, and examples. All estimators were specified against
the published econometric literature cited in [References](#references), and
the numerical results are checked by the test suite and the Monte Carlo
studies reported in [Validation Status](#validation-status).

---

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Arellano, M. & Bond, S. (1991). *Review of Economic Studies*, 58(2).
- Blundell, R. & Bond, S. (1998). *Journal of Econometrics*, 87(1).
- Conley, T. (1999). *Journal of Econometrics*, 92(1).
- Driscoll, J. & Kraay, A. (1998). *Review of Economics and Statistics*, 80(4).
- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression*. Wiley.
- Junius, T. & Oosterhaven, J. (2003). *Economic Systems Research*, 15(1).
- Kelejian, H. & Prucha, I. (1998). *Journal of Real Estate Finance and Economics*, 17(1).
- Lee, L.-F. & Yu, J. (2010). *Journal of Econometrics*, 154(2).
- LeSage, J. & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Roodman, D. (2009). *Stata Journal*, 9(1).
- Santos Silva, J. M. C. & Tenreyro, S. (2006). *Review of Economics and Statistics*, 88(4).
- Windmeijer, F. (2005). *Journal of Econometrics*, 126(1).
- Yu, J., de Jong, R. & Lee, L.-F. (2008). *Journal of Econometrics*, 146(1).
