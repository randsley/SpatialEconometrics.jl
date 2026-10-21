# SpatialEconometrics.jl

A comprehensive spatial and structural econometrics suite in Julia, built around sparse linear algebra, multiple dispatch and automatic differentiation. The package provides 13 submodules covering cross-sectional and panel spatial models, diagnostics, non-parametric spatial regression, gravity models, HAC-robust inference, and input-output calibration — all with a unified formula interface and optional Bayesian and visualization extensions.

> **Version 0.3.0** — 54 tests passing on Julia 1.13.1. See [Validation Status](#validation-status) for Monte Carlo results.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Spatial Diagnostics](#spatial-diagnostics)
- [Cross-Sectional Models (SAR, SDM, SEM)](#cross-sectional-models)
- [Static Spatial Panels](#static-spatial-panels)
- [Dynamic Spatial Panels](#dynamic-spatial-panels)
- [GS2SLS / SARAR Panels](#gs2sls--sarar-panels)
- [System GMM](#system-gmm)
- [GWR & MGWR](#gwr--mgwr)
- [Gravity / Flow Models](#gravity--flow-models)
- [GRAS & Cross-Entropy Calibration](#gras--cross-entropy-calibration)
- [HAC-Robust Inference](#hac-robust-inference)
- [Formula Interface](#formula-interface)
- [Package Extensions](#package-extensions)
- [Repository Layout](#repository-layout)
- [Validation Status](#validation-status)
- [References](#references)

---

## Installation

```julia
using Pkg
Pkg.activate("path/to/SpatialEconometrics")
Pkg.instantiate()
Pkg.test()
```

**Dependencies:** `ChainRulesCore`, `ForwardDiff`, `NearestNeighbors`, `StatsAPI`, `StatsModels`, `Tables`, plus standard libraries (`LinearAlgebra`, `SparseArrays`, `Statistics`, `Random`, `Printf`). Julia 1.10 or later.

**Optional extensions:** `Turing.jl` + `Distributions.jl` (Bayesian), `Makie.jl` (visualization) — loaded automatically when these packages are available.

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
println(lms.robust_lag)    # significant → SAR
println(lms.robust_error)  # not significant → not SEM

# Step 2: Estimate
m = fit_spatial(CrossSection.SAR(), W, X, y;
    method = CrossSection.ChebyshevApprox(order=12),
    varnames = ["income", "density"])
println(m)

# Step 3: Spatial effects
println(spatial_effects(m))
```

---

## Modules

| Submodule | Alias | Description | Key Reference |
|---|---|---|---|
| `SpatialUtils` | `UtilsMod` | Shared numerical utilities (Chebyshev log-det, Brent optimizer, panel transforms) | — |
| `SpatialEconometricsPrototype` | `CrossSection` | Cross-sectional SAR & SDM by concentrated ML | LeSage & Pace (2009) |
| `SpatialPanelEconometrics` | `StaticPanel` | Static spatial panels with FE, Lee-Yu & Direct QML | Lee & Yu (2010) |
| `DynamicSpatialPanelEconometrics` | `DynamicPanel` | Dynamic spatial panels with bias correction | Yu, de Jong & Lee (2008) |
| `SpatialGS2SLSPanel` | `GS2SLSPanel` | SARAR panels with endogenous regressors | Kelejian & Prucha (1998) |
| `DynamicPanelSystemGMM` | `SystemGMMMod` | Arellano-Bond & Blundell-Bond GMM | Roodman (2009) |
| `StructuralMacroCalibration` | `MacroCalib` | GRAS & cross-entropy SAM/MRIO balancing | Junius & Oosterhaven (2003) |
| `SpatialHAC` | `HACMod` | Driscoll-Kraay & Conley HAC covariances | Conley (1999) |
| `SpatialDiagnostics` | `DiagMod` | Moran's I, LISA, LM tests, Geary's C | Anselin (1988) |
| `SpatialErrorModel` | `SEMMod` | Cross-section & panel SEM by concentrated ML | Anselin (1988) |
| `GWREstimation` | `GWRMod` | GWR & MGWR with KDTree spatial indexing | Fotheringham et al. (2002) |
| `SpatialGravityModels` | `GravityMod` | PPML & NB gravity with high-dim FE | Santos Silva & Tenreyro (2006) |
| `FormulaDSL` | `DSLMod` | `@formula` interface with `slag`, `fe`, `endo` | — |

---

## Spatial Diagnostics

```julia
# Global Moran's I (asymptotic or permutation-based)
mr = moran_global(y, W)                          # asymptotic
mr = moran_global(y, W; inference=:permutation, n_perm=999)

# Local Moran's I (LISA) with cluster classification
lisa = moran_local(y, W; n_perm=999)
# lisa.cluster_type: Vector{Symbol} with :HH, :LL, :HL, :LH, :ns

# Geary's C
gc = geary_c(y, W)

# LM specification tests (Anselin 1988)
lm_test_lag(y, X, W)           # LM-Lag
lm_test_error(y, X, W)         # LM-Error
lm_test_robust_lag(y, X, W)    # Robust LM-Lag
lm_test_robust_error(y, X, W)  # Robust LM-Error
lm_tests(y, X, W)              # all 4 as NamedTuple
```

**Decision rule:** If Robust LM-Lag significant and Robust LM-Error not → SAR. If vice versa → SEM. If both → SARAR (GS2SLS).

---

## Cross-Sectional Models

### SAR (Spatial Autoregressive)

```julia
const CS = SpatialEconometrics.CrossSection
m = fit_spatial(CS.SAR(), W, X, y;
    method = CS.ChebyshevApprox(order=12),  # or CS.ExactLU()
    varnames = ["x1", "x2"])
```

### SDM (Spatial Durbin)

```julia
m = fit_spatial(CS.SDM(), W, X, y; varnames = ["x1", "x2"])
effects = spatial_effects(m)  # direct, indirect, total with SEs
```

### SEM (Spatial Error)

```julia
m = fit_spatial_error(W, X, y; varnames = ["x1", "x2"])
# m.lambda: estimated spatial error parameter
```

---

## Static Spatial Panels

```julia
const SP = SpatialEconometrics.StaticPanel

# Data: X is NT×K, y is NT×1, stacked as [unit₁_t₁, unit₂_t₁, ..., unitN_t₁, unit₁_t₂, ...]
m = fit_spatial_panel(SP.SDM(), SP.TwoWayFE(), W, X, y, N, T;
    approach = LeeYuTransformation(),          # or DirectQML(bias_correct=true)
    method = SP.ChebyshevApprox(order=12))

# Fixed effects options: SP.NoFE(), SP.IndivFE(), SP.TimeFE(), SP.TwoWayFE()
```

### Panel SEM

```julia
const SEM = SpatialEconometrics.SEMMod
m = fit_spatial_error(SEM.TwoWayFE(), W, X, y, N, T)
```

---

## Dynamic Spatial Panels

```julia
const DP = SpatialEconometrics.DynamicPanel

m = fit_dynamic_spatial_panel(DP.SAR(), DP.IndivFE(), W, X, y, N, T_total;
    n_bias_probes = 80)

# Short-run and long-run multipliers
de = dynamic_spatial_effects(m)
# de.sr_direct, de.sr_indirect, de.sr_total (+ SEs)
# de.lr_direct, de.lr_indirect, de.lr_total (+ SEs)
```

---

## GS2SLS / SARAR Panels

For models with both spatial lag in y AND spatial error, plus endogenous regressors:

```julia
m = fit_gs2sls_panel(GS2SLSPanel.TwoWayFE(), W, X_exog, X_endog, Z_iv, y, N, T;
    exog_names = ["x1", "x2"],
    endog_names = ["tax_rate"])
# m.rho: spatial lag parameter
# m.lambda: spatial error parameter
```

---

## System GMM

```julia
# Y_mat: N×T matrix, X_ten: N×T×K array
m = fit_dynamic_gmm(SystemGMM(), Y_mat, X_ten;
    max_lag = 4, collapsed = false, varnames = ["x1"])
# m.hansen_j: Hansen J-test statistic
# m.ar1_z, m.ar2_z: AR serial correlation test z-statistics

m_diff = fit_dynamic_gmm(DifferenceGMM(), Y_mat, X_ten; max_lag = 3)
```

---

## GWR & MGWR

Geographically Weighted Regression with local coefficients that vary across space.

```julia
# coords: N×2 matrix of (x, y) or (lon, lat) coordinates
m = fit_gwr(X, y, coords;
    bandwidth = AdaptiveBandwidth(50),    # k nearest neighbors
    kernel = BisquareKernel(),
    varnames = ["temperature", "rainfall"])

# Auto-select bandwidth via AICc
m_auto = fit_gwr(X, y, coords; kernel = BisquareKernel())

# Fixed bandwidth
m_fixed = fit_gwr(X, y, coords; bandwidth = FixedBandwidth(3.0))

# m.coef_local: N×K matrix of local coefficients
# m.R2, m.AICc, m.hat_trace: model fit statistics
```

### Multiscale GWR (MGWR)

Variable-specific bandwidths via backfitting:

```julia
m = fit_mgwr(X, y, coords; kernel = BisquareKernel())
# m.bandwidths: Vector of per-variable bandwidths
```

**Kernels:** `GaussianKernel()`, `BisquareKernel()`, `ExponentialKernel()`

---

## Gravity / Flow Models

Poisson Pseudo-Maximum Likelihood (PPML) and Negative Binomial gravity models for origin-destination flow data.

```julia
m = fit_gravity(PoissonGravity(), flows, X, origin_id, dest_id;
    fe = BilateralFE(),     # or OriginFE(), DestinationFE()
    varnames = ["log_distance", "common_language"])

# Negative Binomial (handles overdispersion)
m_nb = fit_gravity(NegBinGravity(), flows, X, origin_id, dest_id)
# m_nb.alpha: overdispersion parameter

# Multilateral resistance terms
mr = multilateral_resistance(m)
# mr.outward: origin fixed effects
# mr.inward: destination fixed effects
```

---

## GRAS & Cross-Entropy Calibration

Balance SAM/MRIO matrices to new margins, preserving structure and handling negative entries.

```julia
# Full result with convergence info
res = balance_gras_full(X0, u_target, v_target; tol=1e-11)
# res.X_balanced, res.converged, res.kl_divergence, res.iterations

# Simple interface (differentiable via ChainRulesCore + ForwardDiff)
X_bal = balance_gras(X0, u_target, v_target)

# Generalized cross-entropy with side constraints
X_bal, λ_dual, μ_dual = balance_generalized_ce(X0, G, b, sigma_b)

# Leontief SAM multipliers
M, A = sam_leontief_multipliers(X_sam, endogenous_idx)
```

---

## HAC-Robust Inference

Non-parametric covariance estimation robust to spatial and temporal dependence.

```julia
# Driscoll-Kraay (panel SCC)
dk = DriscollKraayVCOV(max_lag=3, kernel=BartlettKernel())
V_robust = robust_vcov(m_panel, dk)
tab = robust_coeftable(m_panel, dk)

# Conley spatial HAC
conley = ConleySpatialHAC(coords, 35.0;   # distance cutoff
    max_time_lag = 2,
    metric = EuclideanDistance(),           # or HaversineDistance()
    space_kernel = BartlettKernel())
V_robust = robust_vcov(m_panel, conley)

# HAC-robust spatial effects
effects = spatial_effects(m_panel, dk)

# Kernels: BartlettKernel(), ParzenKernel(), QuadraticSpectralKernel(), UniformKernel()
# Metrics: EuclideanDistance(), HaversineDistance(radius_km=6371.0)
```

---

## Formula Interface

The `@formula` macro with spatial extensions `slag()`, `fe()`, and `endo()`:

```julia
# SDM panel with two-way FE
m = fit_spatial_panel(
    @formula(y ~ x1 + x2 + slag(x1 + x2) + fe(region) + fe(year)),
    table, W; panel=:region, time=:year)

# GS2SLS with endogenous regressor and IV
m = fit_gs2sls_panel(
    @formula(y ~ x1 + endo(tax, z1 + z2) + fe(region) + fe(year)),
    table, W; panel=:region, time=:year)

# Cross-section SDM
m = fit_spatial(
    @formula(y ~ x1 + x2 + slag(x1 + x2)),
    table, W)

# SEM via formula (cross-section)
m = fit_spatial_error(@formula(y ~ x1 + x2 + fe(region)), table, W)

# SEM panel via formula
m = fit_spatial_error(@formula(y ~ x1 + x2 + fe(region) + fe(year)),
    table, W; panel=:region, time=:year)

# Gravity via formula
m = fit_gravity(PoissonGravity(),
    @formula(flows ~ log_dist + common_lang), table;
    origin=:exporter, dest=:importer)
```

---

## Package Extensions

### Bayesian Spatial Econometrics (Turing.jl)

```julia
using Turing, Distributions
using SpatialEconometrics

result = bayesian_sar(W, X, y; n_samples=1000, n_chains=2)
result = bayesian_sdm(W, X, y; n_samples=1000)
result = bayesian_sem(W, X, y; n_samples=1000)
# result.chain: MCMCChains.Chains object
# result.posterior_mean, result.posterior_std
```

### Visualization (Makie.jl)

```julia
using CairoMakie  # or GLMakie
using SpatialEconometrics

plot_coefficients(m)                          # coefficient CI plot
plot_moran_scatter(y, W)                      # Moran scatterplot
plot_local_moran(lisa, coords)                # LISA cluster map
plot_gwr_coefficients(m_gwr, "temperature")   # GWR coefficient map
plot_spatial_effects(effects)                 # effects bar chart
```

---

## Repository Layout

```text
SpatialEconometrics/
├── Project.toml
├── Manifest.toml
├── README.md
├── src/
│   ├── SpatialEconometrics.jl           # umbrella module
│   ├── SpatialUtils.jl                  # shared utilities
│   ├── SpatialEconometricsPrototype.jl  # cross-sectional SAR/SDM
│   ├── SpatialPanelEconometrics.jl      # static spatial panels
│   ├── DynamicSpatialPanelEconometrics.jl
│   ├── SpatialGS2SLSPanel.jl           # SARAR / GS2SLS
│   ├── DynamicPanelSystemGMM.jl        # Arellano-Bond / Blundell-Bond
│   ├── StructuralMacroCalibration.jl   # GRAS / cross-entropy
│   ├── SpatialHAC.jl                   # Driscoll-Kraay / Conley
│   ├── SpatialDiagnostics.jl           # Moran, LISA, LM tests
│   ├── SpatialErrorModel.jl            # SEM (cross-section + panel)
│   ├── GWREstimation.jl                # GWR + MGWR
│   ├── SpatialGravityModels.jl         # PPML + NB gravity
│   ├── SpatialWeights.jl               # KNN / distance / band weights, I/O
│   ├── SpatialModelSelection.jl        # LR, Vuong, Wald, common factor
│   ├── SpatialPanelDiagnostics.jl      # Hausman, panel LM, panel Moran
│   ├── HigherOrderSpatial.jl           # SLX, SAC/SARAR, SDEM
│   ├── SpatialRegimeModels.jl          # regime models + SKATER
│   ├── SpatialSimulation.jl            # DGPs, Monte Carlo, power analysis
│   ├── SpatialTables.jl                # coefficient / regression tables
│   ├── SpatialDiscreteChoice.jl        # spatial probit (GMM + Bayesian)
│   ├── SpatialFiltering.jl             # ESF / RE-ESF
│   └── FormulaDSL.jl                   # @formula interface
├── ext/
│   ├── SpatialEconometricsTuringExt.jl # Bayesian extension
│   └── SpatialEconometricsMakieExt.jl  # visualization extension
├── test/
│   └── runtests.jl                     # 453 assertions, 23 testsets
├── validation/
│   ├── mc_sdm.jl                       # Monte Carlo validation
│   └── mc_sdm_output.txt
├── examples/
│   ├── 01_cross_section.jl             # SAR, SDM, diagnostics
│   ├── 02_spatial_panels.jl            # static + dynamic panels
│   ├── 03_sem_diagnostics.jl           # SEM workflow with diagnostics
│   ├── 04_gwr_mgwr.jl                  # GWR + MGWR
│   ├── 05_gravity_models.jl            # PPML + NB gravity
│   ├── 06_gs2sls_gmm.jl                # SARAR + System GMM
│   ├── 07_gras_calibration.jl          # SAM balancing
│   ├── 08_hac_robust.jl                # HAC-robust inference
│   ├── 09_higher_order.jl              # SLX, SAC, SDEM + specification choice
│   ├── 10_spatial_regimes.jl           # regimes, SKATER, Chow test
│   ├── 11_discrete_choice.jl           # spatial probit, ordered, marg. effects
│   ├── 12_spatial_filtering.jl         # ESF families + RE-ESF diagnostics
│   ├── 13_simulation_power.jl          # six DGPs, Monte Carlo, power curves
│   └── 14_output_tables.jl             # LaTeX / CSV / Markdown export
└── notebooks/
    ├── 01_getting_started.ipynb        # Jupyter
    ├── 02_spatial_panels.ipynb         # Jupyter
    ├── 03_gwr_gravity.ipynb            # Jupyter
    ├── interactive_spatial_explorer.jl        # Pluto
    ├── interactive_gwr_explorer.jl            # Pluto
    ├── interactive_filtering_explorer.jl      # Pluto — ESF / RE-ESF
    ├── interactive_probit_explorer.jl         # Pluto — GMM vs Gibbs probit
    └── interactive_higher_order_explorer.jl   # Pluto — SLX/SAC/SDEM + regimes
```

Run an example with `julia --project=. examples/12_spatial_filtering.jl`. Open a
Pluto notebook with `using Pluto; Pluto.run()` and navigate to the file; each one
activates this project itself.

---

## Validation Status

| Module | Test Suite | Monte Carlo Validated |
|---|---|---|
| Cross-sectional SAR/SDM | 54/54 ✓ | **Yes** (100 draws, N=400, ρ=0.6, 96% coverage) |
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

**Cross-sectional SDM Monte Carlo results** (`validation/mc_sdm_output.txt`): ρ̂ mean = 0.595 (true 0.600); mean SE = 0.0324, empirical SD = 0.0328; 95% coverage = 0.96. Direct, indirect, and total effects coverage between 0.92–0.97.

---

## Known Limitations

- **Chebyshev order:** Default order 6 is adequate for moderate ρ. Use order 24+ for ρ > 0.85, or `ExactLU()`.
- **Stochastic traces:** Neumann traces use Hutchinson probes (250 by default). For small N, exact traces would be more precise.
- **MGWR convergence:** Backfitting may be slow with many variables. Monitor `m.converged` and `m.iterations`.
- **Gravity FE:** Bilateral FE demeaning uses alternating projections (iterative). Very large O-D matrices may need more iterations.
- **Type stability:** Some `0.0` literals will promote `Float32` to `Float64`. Use `Float64` consistently.
- **Panel data ordering:** Data must be stacked with N units varying fastest within each time period: `[unit₁_t₁, unit₂_t₁, ..., unitN_t₁, unit₁_t₂, ...]`.

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
