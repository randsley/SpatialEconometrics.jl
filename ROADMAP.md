# SpatialEconometrics.jl — Development Roadmap

> **Current version:** 0.3.0 (registered) → 0.4.0 (on `master`, unreleased)
>
> 22 submodules, 453 test assertions in 23 testsets, 14 examples, 8 notebooks,
> 2 package extensions, CI/CD and 22 Documenter.jl pages. **Phases 4, 5 and 6
> are complete except item #10 (Performance)**, which is the next piece of work.
> Tiers 1 and 2 are otherwise finished; Tier 3 is untouched.
>
> This roadmap outlines planned additions organized by impact tier and development phase. Each item includes the motivation, key deliverables, estimated complexity, and relevant references.

---

## Table of Contents

- [Tier 1: High Impact](#tier-1-high-impact)
  - [1. Spatial Panel Data Toolkit](#1-spatial-panel-data-toolkit)
  - [2. Spatial Regime Models](#2-spatial-regime-models)
  - [3. Higher-Order Spatial Models](#3-higher-order-spatial-models)
  - [4. Spatial Probit / Logit](#4-spatial-probit--logit)
  - [5. Model Selection & Comparison](#5-model-selection--comparison)
- [Tier 2: Medium Impact](#tier-2-medium-impact)
  - [6. Weight Matrix Construction](#6-weight-matrix-construction)
  - [7. Spatial Filtering (Eigenvector)](#7-spatial-filtering-eigenvector)
  - [8. Simulation & Power Analysis](#8-simulation--power-analysis)
  - [9. Interoperability](#9-interoperability)
  - [10. Performance](#10-performance)
- [Tier 3: Niche but Valuable](#tier-3-niche-but-valuable)
  - [11. Spatial Econometrics for Networks](#11-spatial-econometrics-for-networks)
  - [12. Spatio-Temporal Models](#12-spatio-temporal-models)
  - [13. Semiparametric Spatial](#13-semiparametric-spatial)
  - [14. Documentation & Ecosystem](#14-documentation--ecosystem)
- [Development Phases](#development-phases)
- [References](#references)

---

## Tier 1: High Impact

### 1. Spatial Panel Data Toolkit

**Motivation:** Panel-specific diagnostics are essential for applied researchers choosing between spatial model specifications in longitudinal settings. Currently only cross-sectional LM tests are available.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Spatial Hausman test | Choose between spatial random effects vs fixed effects; compares FE and RE estimators under the null of RE consistency | Mutl & Pfaffermayr (2011) |
| Spatial Chow test | Structural break detection across spatial regimes in panels; tests parameter stability over time or across spatial clusters | Anselin (1990) |
| Joint LM tests for panels | Extend cross-section LM-lag/LM-error to panel settings; test for spatial lag, spatial error, and random effects jointly | Baltagi, Song & Koh (2003) |
| Panel Moran's I | Time-averaged and period-specific Moran statistics for panel residuals | Anselin et al. (2008) |

**Status:** ✅ Implemented — `src/SpatialPanelDiagnostics.jl` (343 lines). Docs: `docs/src/diagnostics/panel.md`. Example: **no example script yet**. Tests: 15 assertions in testset “Spatial Panel Diagnostics”.

**Key functions:**
- `spatial_hausman_test(m_fe::SpatialPanelModel, m_re)` → `HausmanTestResult`
- `spatial_chow_test(y, X, W, N, T, break_period)` → `ChowTestResult`
- `panel_lm_tests(y, X, W, N, T)` → NamedTuple with joint LM statistics
- `panel_moran(residuals, W, N, T)` → Vector of per-period `MoranResult`

**Dependencies:** None (uses existing SpatialUtils infrastructure)

---

### 2. Spatial Regime Models

**Motivation:** Many spatial processes exhibit structural heterogeneity — coefficients differ systematically across geographic regimes (urban vs rural, north vs south, developed vs developing). Ignoring regime heterogeneity biases both β and ρ estimates.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Spatial regimes SAR | Different β per regime cluster with a single global ρ; block-diagonal design matrix with regime indicators | Anselin (1990) |
| Spatial regimes SDM | Full Durbin specification with regime-varying β and θ but shared ρ | LeSage & Pace (2009, Ch. 8) |
| Regime Chow test | F-test / LR-test for coefficient homogeneity across regimes | Anselin (1990) |
| SKATER clustering | Spatially constrained minimum spanning tree clustering for automatic regime detection from covariates | Assunção et al. (2006) |

**Status:** ✅ Implemented — `src/SpatialRegimeModels.jl` (363 lines). Docs: `docs/src/models/regime.md`. Example: `examples/10_spatial_regimes.jl`. Tests: 12 assertions in testset “Spatial Regime Models”.

**Key types:**
```julia
struct SpatialRegimeModel{V, T, M} <: StatsAPI.RegressionModel
    variant::V
    regime_ids::Vector{Int}
    n_regimes::Int
    coef_by_regime::Matrix{T}  # n_regimes × K
    rho::T                      # global spatial parameter
    # ... standard fields
end
```

**Key functions:**
- `fit_spatial_regime(variant, W, X, y, regime_ids; ...)` → `SpatialRegimeModel`
- `regime_chow_test(m_pooled, m_regime)` → `ChowTestResult`
- `skater_cluster(coords, X, W; n_clusters)` → `Vector{Int}` regime assignments

**Dependencies:** None for core; SKATER uses the existing NearestNeighbors.jl

---

### 3. Higher-Order Spatial Models

**Motivation:** The current package covers SAR, SDM, SEM, and SARAR. Several commonly used specifications are missing, particularly the SAC model with heterogeneous weight matrices and the simple SLX model which is increasingly recommended as a starting point.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| SAC model | General spatial autoregressive combined: `y = ρ₁W₁y + Xβ + u, u = ρ₂W₂u + ε` with potentially different W matrices for lag and error | Kelejian & Prucha (1998) |
| SLX model | Spatial lag of X only: `y = Xβ + WXθ + ε` (no spatial lag of y, no spatial error) — OLS consistent, often sufficient | Halleck Vega & Elhorst (2015) |
| SDEM model | Spatial Durbin Error: `y = Xβ + WXθ + u, u = λWu + ε` — SDM structure in errors | LeSage & Pace (2009) |
| Time-varying W panels | `W_t` changes across periods; requires period-specific log-determinant computation | Lee & Yu (2012) |

**Status:** ✅ Implemented — `src/HigherOrderSpatial.jl` (336 lines). Docs: `docs/src/models/higher_order.md`. Example: `examples/09_higher_order.jl`. Tests: 22 assertions in testset “Higher-Order Spatial Models”.

**Key functions:**
- `fit_slx(W, X, y; ...)` → `SLXModel` (OLS, no ML needed)
- `fit_sac(W_lag, W_error, X, y; ...)` → `SACModel`
- `fit_sdem(W, X, y; ...)` → `SDEMModel`
- `fit_spatial_panel_tvw(variant, fe, W_list, X, y, N, T; ...)` → time-varying W panel

**Dependencies:** None

---

### 4. Spatial Probit / Logit

**Motivation:** Binary and ordered outcomes with spatial dependence are common in applied work (land use change, technology adoption, voting patterns, firm entry/exit). Standard probit/logit ignores spatial correlation, leading to inefficient estimates and invalid inference.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Spatial probit (linearized GMM) | Klier & McMillen (2008) linearized GMM approach — fast, avoids MCMC | Klier & McMillen (2008) |
| Spatial probit (Bayesian) | Full Bayesian estimation via data augmentation (Albert & Chib) with spatial structure | LeSage & Pace (2009, Ch. 10) |
| Spatial ordered probit | Extension to ordered categorical outcomes | Wang, Kockelman & Wang (2013) |
| Marginal effects | Direct/indirect/total effects for spatial probit (not just coefficients) | LeSage & Pace (2009) |

**Status:** ✅ Implemented — `src/SpatialDiscreteChoice.jl` (700 lines). Docs: `docs/src/models/discrete_choice.md`. Example: `examples/11_discrete_choice.jl`. Tests: 51 assertions in testset “Spatial Discrete Choice”.

**Key types:**
```julia
struct SpatialProbitModel{T, M} <: StatsAPI.RegressionModel
    coef::Vector{T}
    vcov::Matrix{T}
    rho::T
    W::M
    fitted_prob::Vector{T}
    marginal_effects::Matrix{T}  # direct, indirect, total
    # ...
end
```

**Key functions:**
- `fit_spatial_probit(W, X, y; method=:linearized_gmm)` → `SpatialProbitModel`
- `fit_spatial_probit(W, X, y; method=:bayesian, n_samples=2000)` — requires Turing extension
- `fit_spatial_ordered_probit(W, X, y_ordered; ...)` → `SpatialOrderedProbitModel`
- `marginal_effects(m::SpatialProbitModel)` → direct/indirect/total

**Dependencies:** None for GMM; Bayesian version in the Turing extension

---

### 5. Model Selection & Comparison

**Motivation:** Applied researchers need formal tools to choose between competing spatial specifications. Currently, the only diagnostic is comparing log-likelihoods manually. Proper information criteria, nesting tests, and the common factor test are essential for rigorous model selection.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Spatial AIC / BIC | Information criteria correcting for effective degrees of freedom (trace of the hat matrix for spatial models) | Burnham & Anderson (2002) |
| Likelihood ratio test | Nested model comparison: SAR vs SDM, SEM vs SDEM, SAR vs SAC | Anselin (2001) |
| Vuong test | Non-nested model comparison with spatial correction (SAR vs SEM) | Vuong (1989) |
| Common factor test | Tests whether the SDM simplifies to SEM (i.e., θ = -ρβ); rejection → SDM preferred over SEM | Burridge (1981); Elhorst (2010) |
| Wald test | General linear hypothesis testing on spatial model parameters | — |

**Status:** ✅ Implemented — `src/SpatialModelSelection.jl` (297 lines). Docs: `docs/src/diagnostics/model_selection.md`. Example: **no example script yet**. Tests: 16 assertions in testset “Model Selection & Comparison”.

**Key functions:**
- `spatial_aic(m)`, `spatial_bic(m)` → corrected information criteria
- `lr_test(m_restricted, m_unrestricted)` → `LRTestResult`
- `vuong_test(m1, m2)` → `VuongTestResult`
- `common_factor_test(m_sdm)` → `CommonFactorTestResult` (Wald-type)
- `wald_test(m, R, r)` → `WaldTestResult` (Rβ = r)
- `model_comparison_table(models...)` → formatted comparison table

**Dependencies:** None

---

## Tier 2: Medium Impact

### 6. Weight Matrix Construction

**Motivation:** The current package only provides `make_rook_weights` and `make_queen_weights` for regular grids. Real-world applications require weight matrices from coordinates, shapefiles, or custom distance functions. This is the most common user request for any spatial econometrics library.

**Deliverables:**

| Feature | Description |
|---|---|
| k-nearest neighbors W | `knn_weights(coords, k)` — asymmetric by default, with symmetrize option |
| Distance-decay W | `distance_weights(coords, cutoff; decay=:inverse)` — inverse distance, negative exponential, power functions |
| Distance-band W | `band_weights(coords, d_min, d_max)` — binary weights within distance band |
| Contiguity from geometry | `contiguity_weights(geometries; type=:queen)` — read GeoJSON/Shapefile polygons and compute shared boundaries |
| Block-diagonal W | `block_diagonal_weights(W_list)` — combine sub-region weight matrices |
| Row standardization | `row_standardize!(W)`, `spectral_standardize!(W)`, `minmax_standardize!(W)` |
| W diagnostics | `spatial_connectivity(W)` — number of islands, average neighbors, histogram, eigenvalue bounds |
| Import/export | `read_gal(path)`, `write_gal(W, ids, path)`, `read_gwt(path)`, `write_gwt(W, ids, path)` — GeoDa format compatibility |

**Status:** ✅ Implemented — `src/SpatialWeights.jl` (526 lines). Docs: `docs/src/weights.md`. Example: **no example script yet**. Tests: 22 assertions in testset “Spatial Weights Construction”.

**Key functions:**
```julia
# From coordinates
W = knn_weights(coords, 5)
W = distance_weights(coords, 100.0; decay=:inverse_squared)
W = band_weights(coords, 0.0, 50.0)

# From geometries (requires GeoInterface.jl)
W = contiguity_weights(polygons; type=:rook)

# Utilities
row_standardize!(W)
info = spatial_connectivity(W)
# info.n_islands, info.avg_neighbors, info.min_neighbors, info.max_neighbors
```

**Dependencies:** NearestNeighbors.jl (already available); optional GeoInterface.jl for geometry-based weights

---

### 7. Spatial Filtering (Eigenvector)

**Motivation:** Moran eigenvector spatial filtering (ESF) provides a nonparametric alternative to parametric spatial models. It works by including selected eigenvectors of the spatial weight matrix as additional regressors, effectively filtering out spatial autocorrelation. ESF can be combined with any GLM, making it more flexible than SAR/SEM.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Eigenvector extraction | Compute candidate eigenvectors from `(I - 11'/N) W (I - 11'/N)` | Griffith (2003) |
| Forward selection | Stepwise selection of eigenvectors minimizing Moran's I of residuals | Griffith (2003) |
| ESF-OLS | Spatial filtering for linear models | Griffith (2003) |
| ESF-GLM | Spatial filtering for Poisson, logit, negative binomial | Griffith (2003); Chun et al. (2016) |
| Random effects ESF | Eigenvector spatial filtering with random effects for panels | Murakami & Griffith (2015) |

**Status:** ✅ Implemented — `src/SpatialFiltering.jl` (718 lines). Docs: `docs/src/models/filtering.md`. Example: `examples/12_spatial_filtering.jl`. Tests: 47 assertions in testset “Spatial Filtering”.

**Key functions (delivered):**
- `spatial_eigenvectors(W; n_candidates=nothing, min_moran_ratio=0.0)` → `SpatialEigen`
- `moran_coefficient(x, W)` → Moran coefficient of a vector
- `fit_esf(X, y, W; family=:gaussian, max_eigenvectors=20, criterion=:moran)` → `ESFModel`
  (families `:gaussian`, `:poisson`, `:logit`, `:negbin`; `criterion` `:moran` or `:aic`)
- `selected_eigenvectors(m::ESFModel)` → indices and Moran values of selected eigenvectors
- `fit_re_esf(X, y, W; min_moran_ratio=0.25)` → `REESFModel` (profile-REML shrinkage)

**Implementation notes:**
- The basis is built on the orthogonal complement of `1` via QR rather than by
  decomposing `Ω` directly. `Ω` has a multi-dimensional null space, so `eigen`
  returns an arbitrary rotation within it, which smears the constant direction
  across several near-zero eigenvectors and leaves each partly collinear with the
  intercept. The QR construction gives `E'E = I` and `Σᵢ eᵢ = 0` exactly.
- RE-ESF reports `moran_reference`, the closed-form expected Moran coefficient of
  the complement of `span(E)`. The conditional residual Moran coefficient is
  expected to be substantially negative and must be read against that reference
  rather than against zero.

**Dependencies:** None (uses LinearAlgebra `eigen`)

---

### 8. Simulation & Power Analysis

**Motivation:** Researchers need tools to (a) generate realistic spatial data under different DGPs for Monte Carlo studies, (b) assess the statistical power of spatial tests before collecting data, and (c) validate custom estimators against known ground truth.

**Deliverables:**

| Feature | Description |
|---|---|
| DGP generators | `simulate_sar(W, X, β, ρ, σ; rng)`, `simulate_sdm(...)`, `simulate_sem(...)`, `simulate_dynamic_panel(...)` — return y with known parameters |
| Monte Carlo engine | `monte_carlo(dgp_fn, estimator_fn, n_reps; rng)` → bias, RMSE, coverage, size |
| Power curves | `power_curve(test_fn, dgp_fn, param_range; n_reps, α)` → power at each parameter value |
| Bootstrap inference | `spatial_bootstrap(m, n_boot; type=:residual)` — residual and wild bootstrap for spatial models |
| Size-power comparison | `size_power_table(tests, dgps, n_reps)` — tabulate size and power across multiple tests and DGPs |

**Status:** ✅ Implemented — `src/SpatialSimulation.jl` (484 lines). Docs: `docs/src/simulation.md`. Example: `examples/13_simulation_power.jl`. Tests: 11 assertions in testset “Spatial Simulation & Monte Carlo”.

**Key functions:**
```julia
# Generate data from known DGP
y, params = simulate_sar(W, X, β=[1.0, -0.5], ρ=0.4, σ=0.5; rng=rng)

# Monte Carlo
results = monte_carlo(
    rng -> simulate_sar(W, randn(rng, N, 2), β=[1.0, -0.5], ρ=0.4, σ=0.5; rng=rng),
    (y, X) -> fit_spatial(CS.SAR(), W, X, y),
    1000; rng=MersenneTwister(1))
# results.bias, results.rmse, results.coverage_95

# Power curve for Moran's I at different ρ
pc = power_curve(
    (y, X) -> moran_global(y, W).p_value < 0.05,
    ρ -> simulate_sar(W, X, β=[1.0], ρ=ρ, σ=1.0),
    0.0:0.05:0.8; n_reps=500)
```

**Dependencies:** None

---

### 9. Interoperability

**Motivation:** Integration with the broader Julia data ecosystem reduces friction for users coming from R (`spdep`, `spatialreg`) or Stata (`spreg`, `sppack`). Export to standard formats enables publication-ready output.

**Deliverables:**

| Feature | Status | Description |
|---|---|---|
| GeoDa file I/O | ✅ | `read_gal`/`write_gal`, `read_gwt`/`write_gwt`. Both readers return `(W, ids)` and take an `ids` keyword to align with an external table |
| DataFrame integration | ✅ | `coeftable(m)` returns a Tables.jl-compatible table, consumable by DataFrames, CSV and friends |
| Stargazer-style tables | ✅ | `regression_table(m1, m2, …)` side by side, plus `to_latex`, `to_csv`, `to_markdown` |
| CSV export | 🟡 | Covered by `to_csv(coeftable(m))`; there is no single-call `export_results(m, path)` |
| GeoDataFrames bridge | ⬜ | Not started |
| R interop | ⬜ | Not started — `read_rds_weights` does not exist |

**Status:** 🟡 Mostly implemented — `src/SpatialTables.jl` (308 lines) plus the I/O
half of `src/SpatialWeights.jl`. Note that **`src/SpatialIO.jl` was never created**:
the GeoDa readers and writers live in `SpatialWeights.jl`, since they return a
weight matrix and share its sparse plumbing. The GeoDataFrames bridge and the R
interop remain outstanding, and the planned `comparison_table(…; format=:latex)`
signature became the pair `regression_table` / `to_latex`.

**Dependencies:** Optional DataFrames.jl (weakdep), optional GeoDataFrames.jl (weakdep)

---

### 10. Performance

**Motivation:** For large spatial datasets (N > 10,000 units, long panels), current implementations become slow due to dense matrix operations in the information matrix and GWR local regressions. Targeted optimizations can extend the practical range by 10–100×.

**Deliverables:**

| Feature | Description |
|---|---|
| Multithreaded GWR | `Threads.@threads` over local regressions in `fit_gwr` — embarrassingly parallel, linear speedup |
| Sparse Chebyshev | Maintain sparsity in Chebyshev polynomial recursion (currently densifies at high orders) |
| GPU spatial lag | CUDA.jl kernel for `W * y` when N > 50,000 — weight matrix stays on GPU |
| Out-of-core panels | Arrow.jl / DuckDB.jl streaming for `fit_spatial_panel` — process partitioned Parquet files without loading full NT matrix |
| Precompiled sysimage | PackageCompiler.jl sysimage recipe for zero-latency interactive use |
| Benchmark suite | BenchmarkTools.jl suite comparing across N sizes, W densities, and Chebyshev orders |

**Status:** ⬜ Not started — the only outstanding item in Tiers 1–2.

**Files to add:** Existing modules + new `src/SpatialGPU.jl` (extension), `benchmark/` directory

**Dependencies:** Optional CUDA.jl (weakdep), optional Arrow.jl (weakdep), optional DuckDB.jl (weakdep)

**Estimated complexity:** ~600 lines across multiple files

**Verification note:** all six deliverables are testable on the current
development machine — 16 logical cores, an RTX 3080 (compute capability 8.6,
16 GB VRAM, `CUDA.functional()` confirmed) and ample free disk for a panel that
genuinely exceeds the 32 GB of RAM. CI runners have no GPU, so the CUDA paths
must be gated on `CUDA.functional()` behind a package extension, following the
pattern the Turing and Makie extensions already use.

---

## Tier 3: Niche but Valuable

### 11. Spatial Econometrics for Networks

**Motivation:** Traditional spatial econometrics assumes geographic proximity drives interactions. Modern applications increasingly involve non-geographic networks: trade links, social connections, supply chains, citation networks. Network autocorrelation models generalize spatial models to arbitrary graph structures.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Network autocorrelation model | `y = ρWy + Xβ + ε` where W is derived from network adjacency (not geography) | Leenders (2002) |
| Network weight construction | `network_weights(adjacency; normalize=:row)` — from adjacency matrix, edge list, or graph |
| Peer effects model | `y_i = ρ * mean(y_peers) + Xβ + ε` — endogenous vs exogenous peer effects (Manski reflection problem) | Manski (1993); Bramoullé et al. (2009) |
| Network centrality instruments | Construct instruments from higher-order network structure (friends-of-friends) to address endogeneity | Bramoullé et al. (2009) |
| Multiplex spatial models | Multiple W matrices representing different interaction channels (trade + migration + FDI) | LeSage & Pace (2009, Ch. 9) |

**New file:** `src/SpatialNetworks.jl` (~400 lines)

**Dependencies:** None for core (W is just a sparse matrix); optional Graphs.jl for graph construction

**Estimated complexity:** ~400 lines

---

### 12. Spatio-Temporal Models

**Motivation:** While the dynamic spatial panel captures temporal dependence through lagged y, it assumes a specific AR(1) structure. More flexible spatio-temporal models allow richer dynamics, time-varying spatial effects, and impulse response analysis.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| STAR model | Space-Time Autoregressive: `y_t = Σ_k ρ_k W^k y_{t-1} + Xβ + ε` with higher-order spatial and temporal lags | Pfeifer & Deutsch (1980) |
| SpVAR | Spatial Vector Autoregression: multivariate spatial panel VAR with impulse response functions | Di Giacinto (2010) |
| Space-time impulse response | Trace the propagation of a shock through space and time; compute cumulative spatial multipliers | LeSage & Pace (2009) |
| Time-varying spatial effects | Allow ρ_t to change over time via state-space or rolling-window estimation | — |
| Forecasting | `spatial_forecast(m, X_future, W; h=5)` — h-step ahead spatial panel forecasts | — |

**New file:** `src/SpatioTemporalModels.jl` (~500 lines)

**Dependencies:** None

**Estimated complexity:** ~500 lines

---

### 13. Semiparametric Spatial

**Motivation:** Parametric spatial models assume linear functional forms for the relationship between X and y. When relationships are nonlinear (e.g., pollution effects with thresholds, diminishing returns to agglomeration), semiparametric methods avoid specification bias while retaining spatial structure.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Partially linear spatial model | `y = ρWy + Xβ + f(z) + ε` where f(z) is estimated nonparametrically (kernel or P-spline) | Su & Jin (2010) |
| Spatial quantile regression | Effects vary across the conditional distribution: `Q_τ(y|X,W) = ρ_τ Wy + Xβ_τ` | Kostov (2009) |
| Spatial varying coefficient model | Generalization of GWR with penalized spline smoothing — avoids GWR's boundary bias and bandwidth sensitivity | Hastie & Tibshirani (1993); Geniaux & Martinetti (2017) |
| Penalized spatial regression | LASSO / elastic net with spatial lag: variable selection in high-dimensional spatial models | Liu & Yang (2020) |
| Additive spatial model | `y = ρWy + Σ_k f_k(x_k) + ε` — spatial GAM with backfitting | — |

**New file:** `src/SemiparametricSpatial.jl` (~500 lines)

**Dependencies:** None for P-splines (implement basis construction); optional MLJ.jl for penalized regression

**Estimated complexity:** ~500 lines

---

### 14. Documentation & Ecosystem

**Motivation:** A package is only as good as its documentation. Moving from research prototype to community-adopted tool requires tutorials, benchmarks, and academic publication.

**Deliverables:**

| Feature | Description |
|---|---|
| Documenter.jl site | Full API documentation hosted on GitHub Pages with docstrings for every public function |
| Jupyter tutorials | Step-by-step notebooks: (a) "Spatial Econometrics in 30 Minutes", (b) "Choosing Between SAR, SEM, and SDM", (c) "GWR for Non-Stationary Processes", (d) "Gravity Models for Trade Analysis" |
| Benchmarks vs R/Stata | Timing and accuracy comparison tables against `spdep`/`spatialreg` (R) and `spreg`/`sppack` (Stata) for SAR, SDM, GWR, and gravity models |
| Pluto.jl reactive notebooks | Interactive versions of examples where users can adjust ρ, N, kernel and see results update live |
| JuliaCon talk | Conference submission demonstrating the package |
| JSS paper | Journal of Statistical Software submission: "SpatialEconometrics.jl: A Comprehensive Spatial Econometrics Suite in Julia" |
| Contributing guide | `CONTRIBUTING.md` with coding conventions, testing requirements, and PR checklist |
| CI/CD pipeline | GitHub Actions for automated testing on Julia 1.10, 1.13, nightly; code coverage via Codecov |

**New files/directories:** `docs/`, `notebooks/`, `benchmark/`, `.github/workflows/`, `CONTRIBUTING.md`

**Dependencies:** Documenter.jl, Literate.jl (for notebook generation)

**Estimated complexity:** ~2,000 lines (docs + notebooks + CI config)

---

## Development Phases

> **Known gap:** `SpatialWeights`, `SpatialModelSelection` and
> `SpatialPanelDiagnostics` have documentation and tests but no dedicated
> example script. Examples 01–14 cover the other eleven modules.

### Phase 4 — ✅ Complete

| Priority | Item | Est. | Actual | Status |
|---|---|---|---|---|
| 1 | Weight Matrix Construction (#6) | ~500 | 526 | ✅ |
| 2 | Model Selection & Comparison (#5) | ~300 | 297 | ✅ |
| 3 | Spatial Panel Diagnostics (#1) | ~350 | 343 | ✅ |
| 4 | CI/CD + Documenter.jl (#14 partial) | ~500 | — | ✅ GitHub Actions, 22 docs pages |
| | **Phase 4 total** | **~1,650** | | |

### Phase 5 — ✅ Complete

| Priority | Item | Est. | Actual | Status |
|---|---|---|---|---|
| 5 | Higher-Order Spatial Models (#3) | ~450 | 336 | ✅ |
| 6 | Spatial Regime Models (#2) | ~400 | 363 | ✅ |
| 7 | Simulation & Power Analysis (#8) | ~400 | 484 | ✅ |
| 8 | Interoperability (#9) | ~500 | 308 | 🟡 Tables and GeoDa I/O done; GeoDataFrames and R interop not started |
| | **Phase 5 total** | **~1,750** | | |

### Phase 6 — 🟡 Two of three

| Priority | Item | Est. | Actual | Status |
|---|---|---|---|---|
| 9 | Spatial Probit / Logit (#4) | ~500 | 700 | ✅ |
| 10 | Spatial Filtering (#7) | ~350 | 718 | ✅ |
| 11 | Performance (#10) | ~600 | — | ⬜ **Next up** |
| | **Phase 6 total** | **~1,450** | | |

### Phase 7 — ⬜ Not started

| Priority | Item | Est. Lines | Rationale |
|---|---|---|---|
| 12 | Spatial Networks (#11) | ~400 | Extends beyond geography |
| 13 | Spatio-Temporal Models (#12) | ~500 | STAR, SpVAR, forecasting |
| 14 | Semiparametric Spatial (#13) | ~500 | Quantile regression, penalized models |
| 15 | Full documentation & paper (#14) | ~2,000 | JSS paper, tutorials, benchmarks |
| | **Phase 7 total** | **~3,400** | |

### Size: Actuals and Projections

Rows above the rule are measured; rows below are estimates.

| Milestone | Modules | Test assertions | Lines of `src/` |
|---|---|---|---|
| **v0.3.0** (registered) | 13 | 54 | ~6,500 |
| **master** (Phases 4–6 bar #10) | 22 | 453 | 8,817 |
| — | | | |
| **v0.4.0** (next release) | 22–23 | ~470 | ~9,400 |
| **v1.0.0** (Phase 7) | 26–27 | ~600 | ~13,000 |

Measured on master: 22 submodules plus the umbrella across 8,817 lines of
`src/`, 1,149 lines of tests (453 assertions in 23 testsets), 1,426 lines of
examples across 14 scripts, 318 lines of package extensions, 22 documentation
pages and 8 notebooks.

The earlier per-release projections (v0.4.0 at 16 modules, v0.5.0 at 20,
v0.6.0 at 23) were overtaken: Phases 4, 5 and 6 all landed on the 0.4.0
development line rather than across three releases.

---

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Anselin, L. (1990). Spatial dependence and spatial structural instability in applied regression analysis. *Journal of Regional Science*, 30(2), 185–207.
- Anselin, L. (2001). Rao's score test in spatial econometrics. *Journal of Statistical Planning and Inference*, 97(1), 113–139.
- Anselin, L., Le Gallo, J., & Jayet, H. (2008). Spatial panel econometrics. In *The Econometrics of Panel Data* (pp. 625–660). Springer.
- Assunção, R. M., Neves, M. C., Câmara, G., & Da Costa Freitas, C. (2006). Efficient regionalization techniques for socio-economic geographical units using minimum spanning trees. *International Journal of Geographical Information Science*, 20(7), 797–811.
- Baltagi, B. H., Song, S. H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123–150.
- Bramoullé, Y., Djebbari, H., & Fortin, B. (2009). Identification of peer effects through social networks. *Journal of Econometrics*, 150(1), 41–55.
- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
- Burridge, P. (1981). Testing for a common factor in a spatial autoregression model. *Environment and Planning A*, 13(7), 795–800.
- Chun, Y., Griffith, D. A., Lee, M., & Sinha, P. (2016). Eigenvector selection with stepwise regression techniques to construct eigenvector spatial filters. *Journal of Geographical Systems*, 18(1), 67–85.
- Di Giacinto, V. (2010). On vector autoregressive modeling in space and time. *Journal of Geographical Systems*, 12(2), 125–154.
- Elhorst, J. P. (2010). Applied spatial econometrics: Raising the bar. *Spatial Economic Analysis*, 5(1), 9–28.
- Geniaux, G., & Martinetti, D. (2017). A new method for dealing with spatially varying coefficients. *Geographical Analysis*, 50(1), 28–50.
- Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
- Halleck Vega, S., & Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55(3), 339–363.
- Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models. *Journal of the Royal Statistical Society B*, 55(4), 757–796.
- Klier, T., & McMillen, D. P. (2008). Clustering of auto supplier plants in the United States. *Journal of Business & Economic Statistics*, 26(4), 460–471.
- Kostov, P. (2009). A spatial quantile regression hedonic model of agricultural land prices. *Spatial Economic Analysis*, 4(1), 53–72.
- Leenders, R. T. A. J. (2002). Modeling social influence through network autocorrelation. *Social Networks*, 24(1), 21–47.
- Lee, L.-F., & Yu, J. (2012). QML estimation of spatial dynamic panel data models with time varying spatial weights matrices. *Spatial Economic Analysis*, 7(1), 31–74.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Liu, X., & Yang, Z. (2020). Variable selection in spatial autoregressive models. *Journal of Econometrics*, 218(1), 85–109.
- Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531–542.
- Murakami, D., & Griffith, D. A. (2015). Random effects specifications in eigenvector spatial filtering. *Journal of Geographical Systems*, 17(4), 311–331.
- Mutl, J., & Pfaffermayr, M. (2011). The Hausman test in a Cliff and Ord panel model. *Econometrics Journal*, 14(1), 48–76.
- Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative procedure for space-time modeling. *Technometrics*, 22(1), 35–47.
- Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of partially linear spatial autoregressive models. *Journal of Econometrics*, 157(1), 18–33.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57(2), 307–333.
- Wang, C., Kockelman, K. M., & Wang, X. (2013). The spatial dimensions of land use change dynamics in the Austin, Texas Metropolitan Area. *Transportation Research Record*, 2397(1), 131–142.
