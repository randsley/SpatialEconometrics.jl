# SpatialEconometrics.jl

A spatial and structural econometrics suite in Julia, built around sparse
linear algebra, multiple dispatch and automatic differentiation. It covers
cross-sectional, panel and higher-order spatial models, spatial discrete
choice, eigenvector spatial filtering, regime models, spatial and panel
diagnostics, model selection, geographically weighted regression, gravity
models, HAC-robust inference, weight-matrix construction, input-output
calibration, spatial causal inference and shift-share designs, network
peer effects, common factors and spatio-temporal models, spatial VARs,
semiparametric spatial models, short-panel dynamic models with correlated
random effects or common shocks, spatial log-GARCH for the variance, Monte Carlo simulation and publication-ready output — with a
unified formula interface and optional Bayesian and visualization extensions.

> **Version 0.6.0** — 4,241 test assertions across 64 top-level testsets
> with both package extensions installed, measured on Julia 1.13.4. Without
> them, 4,206 pass and two broken markers stand in for the extension testsets,
> measured on Julia 1.10.13. See [Validation Status](#validation-status) for
> Monte Carlo results.

**Documentation:** [stable](https://randsley.github.io/SpatialEconometrics.jl/stable/)
(the latest tagged release) and [dev](https://randsley.github.io/SpatialEconometrics.jl/dev/)
(`master`).

Every exported name carries a docstring, so `?fit_spatial` works in the
REPL. All 291 of them appear in the API reference — the 292nd, `@formula`,
is re-exported from StatsModels.jl and documented there. The docs build runs
with no `warnonly` exemptions, so a missing docstring or a broken
cross-reference fails CI.

---

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/randsley/SpatialEconometrics.jl")
```

> The package is installed from this repository, so `Pkg.add` by name does not
> resolve it.

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

Thirty-four worked examples live in [`examples/`](examples/), one per topic,
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
| [Short-T Dynamic Panel (CRE)](https://randsley.github.io/SpatialEconometrics.jl/dev/models/cre_dynamic_panel/) | Fixed-`T` M-estimator with correlated random effects |
| [Short-T Dynamic Panel (Common Shocks)](https://randsley.github.io/SpatialEconometrics.jl/dev/models/common_shock_dynamic_panel/) | Fixed-`T` conditional QML with common factors and spatially enriched random loadings |
| [Spatial Error (SEM)](https://randsley.github.io/SpatialEconometrics.jl/dev/models/sem/) | Dependence in the disturbance |
| [GS2SLS / SARAR](https://randsley.github.io/SpatialEconometrics.jl/dev/models/gs2sls/) | Endogenous regressors and instruments |
| [Dynamic Panel GMM](https://randsley.github.io/SpatialEconometrics.jl/dev/models/gmm/) | Arellano-Bond, Blundell-Bond, Hansen and AR tests |
| [Higher-Order](https://randsley.github.io/SpatialEconometrics.jl/dev/models/higher_order/) | SLX, SAC/SARAR, SDEM |
| [Regime Models](https://randsley.github.io/SpatialEconometrics.jl/dev/models/regime/) | Coefficients varying by group, SKATER regionalization |
| [Discrete Choice](https://randsley.github.io/SpatialEconometrics.jl/dev/models/discrete_choice/) | Spatial probit by linearized GMM and Gibbs sampling |
| [Spatial Filtering](https://randsley.github.io/SpatialEconometrics.jl/dev/models/filtering/) | ESF and RE-ESF |
| [Causal Inference](https://randsley.github.io/SpatialEconometrics.jl/dev/models/causal/) | Exposure mapping, spatial DiD, synthetic control, shift-share |
| [Networks](https://randsley.github.io/SpatialEconometrics.jl/dev/models/networks/) | Peer-effects identification, network 2SLS, multiplex SAR |
| [Common Factors & Spatio-Temporal](https://randsley.github.io/SpatialEconometrics.jl/dev/models/common_factors/) | CD and `α` tests, SP-CCE, STAR, interactive fixed effects |
| [Spatial VAR](https://randsley.github.io/SpatialEconometrics.jl/dev/models/spvar/) | Multivariate space-time systems, STIR, the nested panel VAR |
| [Semiparametric Models](https://randsley.github.io/SpatialEconometrics.jl/dev/models/semiparametric/) | Partially linear SAR, closed-form `ρ`, `β(τ)` |
| [Root Estimator of ρ](https://randsley.github.io/SpatialEconometrics.jl/dev/models/root_rho/) | `ρ` without a log-determinant, unbiased for a large basis, or with a penalised smoother; a test-inversion set valid under weak identification |
| [Spatial log-GARCH](https://randsley.github.io/SpatialEconometrics.jl/dev/models/spatial_garch/) | Spatial dependence in the conditional variance; covariate-distance `W` for unknown locations |
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

## Using it with Claude Code

`skills/spatial-econometrics/` is a skill for
[Claude Code](https://docs.claude.com/en/docs/claude-code/skills). Once it is
linked into `~/.claude/skills/`, Claude follows the package's own workflow
when asked about spatial data: it checks `W`, runs the LM battery, reports
effects rather than raw coefficients for lag models, and reads each fit's
warnings before reporting. Its function reference is generated from the
docstrings and checked by the test suite. See
[`skills/README.md`](skills/README.md) to install it.

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
| `CREDynamicSpatialPanel` | Short-`T` dynamic spatial panel with correlated random effects | Li & Yang (2021) |
| `CommonShockDynamicPanel` | Short-`T` dynamic spatial panel with common shocks | Bai & Chen (2026) |
| `SpatialGARCH` | Spatial log-GARCH: two-step QML with a derived sandwich | Sato & Matsuda (2021) |
| `RootRho` | Root estimator of `ρ` with a growing basis (REsd) or a penalised smoother (`tensor_pspline`): HC2 variance, test-inversion set | Jin & Lee (2012) |
| `SpatialGS2SLSPanel` | SARAR panels with endogenous regressors | Kelejian & Prucha (1998, 2010) |
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
| `SpatialCausal` | Exposure mapping, spatial DiD, synthetic control | Abadie, Diamond & Hainmueller (2010) |
| `SpatialShiftShare` | Shift-share designs and the AKM standard error | Adao, Kolesar & Morales (2019) |
| `SpatialNetworks` | Peer-effects identification, network 2SLS, multiplex SAR | Bramoulle, Djebbari & Fortin (2009) |
| `SpatialCommonFactors` | Weak vs strong dependence: CD test, `α` exponent, SP-CCE | Bailey, Kapetanios & Pesaran (2016) |
| `SpatioTemporalModels` | STAR, interactive fixed effects, forecasting | Pfeifer & Deutsch (1980); Shi & Lee (2017) |
| `SpVAR` | Spatial vector autoregression, STIR, nested panel VAR | Di Giacinto (2010) |
| `SemiparametricSpatial` | Partially linear SAR, closed-form `ρ`, overlap `π̄`, `β(τ)` | Su & Jin (2010) |
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
├── src/            # 34 modules: the umbrella plus 33 submodules
├── ext/            # Makie and Turing package extensions (see ext/README.md)
├── test/           # runtests.jl + testsets/: 64 top-level testsets, one file each (see test/README.md)
├── skills/         # a Claude Code skill for the package (see skills/README.md)
├── examples/       # 34 runnable examples, one per topic
├── docs/           # Documenter source for the manual and API reference
├── validation/     # Monte Carlo studies and their output (see validation/README.md)
└── benchmark/      # BenchmarkTools suite; see benchmark/README.md
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
| Spatial Causal Inference | ✓ | Two-period DiD exact; staggered rollout exact under homogeneity; synthetic control against the exact optimum |
| Shift-Share | ✓ | Shock-level equivalence, Rotemberg decomposition and AKM-to-clustering reduction, all exact |
| Networks | ✓ | Published algebraic identities; rank condition verdict by verdict; 2SLS against explicit normal equations |
| Common Factors | ✓ | CD and `α` against the printed equations; `α` scale invariance exact |
| Spatio-Temporal / SpVAR | ✓ | STAR against explicit normal equations; IRF long run equals the spatial multiplier; `s = 0` SpVAR is the nested panel VAR |
| Semiparametric | ✓ | **Yes** (12 cells, `validation/mc_semiparametric_output.txt`; flat smoother reproduces `fit_spatial` to 3e-10) |
| Short-T CRE Dynamic Panel | ✓ | **Yes** (Li & Yang's Table 1 replicated, 4,500/4,500 fits; sandwich coverage 0.92–0.97 for `β`, `ρ`, `λ`) |
| Short-T Common-Shock Panel | ✓ | **Yes** (`validation/mc_serl_sdpd_output.txt`: at `L = 1` coverage of `ρ` 0.93–0.96 to `N = 4000`, at `L = 0` it falls to 0.40; the default `Q` biases `β` at large `N` when regressors are serially correlated with the loadings, and every period of `X` as `Q` removes it) |
| Spatial log-GARCH | ✓ | **Yes** (`validation/mc_spatial_log_garch_output.txt`: Sato & Matsuda's Table 1 reproduced at `n = 400`, where the derived sandwich covers 0.93–0.96 except near `λ = 1`; the quasi-likelihood is unbounded as `λ → 1`, so 36–39% of `n = 100` samples at `λ = 0.9` have no interior maximum and the fit throws; a proxy `W` biases `ρ̂` more as `n` grows) |
| Root Estimator of ρ | ✓ | **Yes** (`validation/mc_root_rho_output.txt`: 108 cells, 3 `W`, `n` 400–1600, `K/n` up to 0.19; the test-inversion set covers 0.935–0.975 and the Wald interval 0.931–0.975, with `|bias|` at most 0.06, while the plain closed form is biased by −0.05 to −1.76 on the same draws; and with the penalised smoother, `validation/mc_root_rho_smoother_output.txt`: 108 cells, both intervals cover 0.934–0.982 wherever the exact shrinkage bias `c` is at most 0.25, and as predicted by `c` elsewhere; the growing-basis theory has not been checked by anyone else) |

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
- Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *Review of Economic Studies*, 58(2), 277–297. https://doi.org/10.2307/2297968
- Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*, 87(1), 115–143. https://doi.org/10.1016/S0304-4076(98)00009-8
- Conley, T. G. (1999). GMM estimation with cross sectional dependence. *Journal of Econometrics*, 92(1), 1–45. https://doi.org/10.1016/S0304-4076(98)00084-0
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549–560. https://doi.org/10.1162/003465398557825
- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically Weighted Regression*. Wiley.
- Junius, T., & Oosterhaven, J. (2003). The solution of updating or regionalizing a matrix with both positive and negative entries. *Economic Systems Research*, 15(1), 87–96. https://doi.org/10.1080/0953531032000056954
- Kelejian, H. H., & Prucha, I. R. (1998). A generalized spatial two-stage least squares procedure for estimating a spatial autoregressive model with autoregressive disturbances. *Journal of Real Estate Finance and Economics*, 17(1), 99–121. https://doi.org/10.1023/A:1007707430416
- Lee, L.-F., & Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*, 154(2), 165–185. https://doi.org/10.1016/j.jeconom.2009.08.001
- LeSage, J. & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. *Stata Journal*, 9(1), 86–136. https://doi.org/10.1177/1536867X0900900106
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. *Review of Economics and Statistics*, 88(4), 641–658. https://doi.org/10.1162/rest.88.4.641
- Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. *Journal of Econometrics*, 126(1), 25–51. https://doi.org/10.1016/j.jeconom.2004.02.005
- Yu, J., de Jong, R., & Lee, L.-F. (2008). Quasi-maximum likelihood estimators for spatial dynamic panel data with fixed effects when both n and T are large. *Journal of Econometrics*, 146(1), 118–134. https://doi.org/10.1016/j.jeconom.2008.08.002

## License

MIT; see [`LICENSE`](LICENSE). The licence covers the code. The datasets in
[`notebooks/data/`](notebooks/data/) keep their sources' terms, set out in
each `*_PROVENANCE.md` file there. Five of the European files are derived from
OpenStreetMap boundaries and are offered under the
[Open Database License](https://opendatacommons.org/licenses/odbl/1-0/)
(© OpenStreetMap contributors); see
[`europe_PROVENANCE.md`](notebooks/data/europe_PROVENANCE.md).
