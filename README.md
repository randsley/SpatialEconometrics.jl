# SpatialEconometrics.jl

A comprehensive spatial and structural econometrics suite in Julia, built around sparse linear algebra, multiple dispatch and automatic differentiation. The package provides 22 submodules covering cross-sectional, panel and higher-order spatial models, spatial discrete choice, eigenvector spatial filtering, regime models, spatial and panel diagnostics, model selection, non-parametric spatial regression, gravity models, HAC-robust inference, weight-matrix construction, input-output calibration, Monte Carlo simulation and publication-ready output — all with a unified formula interface and optional Bayesian and visualization extensions.

> **Version 0.3.0** (registration in review) — 492 test assertions across 26 testsets (527 with the two package extensions installed), passing on Julia 1.13.2. See [Validation Status](#validation-status) for Monte Carlo results.

**Documentation:** [stable](https://randsley.github.io/SpatialEconometrics.jl/stable/) · [dev](https://randsley.github.io/SpatialEconometrics.jl/dev/)

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
- [Weight Matrix Construction](#weight-matrix-construction)
- [Spatial Diagnostics](#spatial-diagnostics)
- [Cross-Sectional Models (SAR, SDM, SEM)](#cross-sectional-models)
- [Higher-Order Models (SLX, SAC, SDEM)](#higher-order-models)
- [Spatial Regime Models](#spatial-regime-models)
- [Spatial Discrete Choice](#spatial-discrete-choice)
- [Spatial Filtering (ESF / RE-ESF)](#spatial-filtering)
- [Static Spatial Panels](#static-spatial-panels)
- [Dynamic Spatial Panels](#dynamic-spatial-panels)
- [GS2SLS / SARAR Panels](#gs2sls--sarar-panels)
- [System GMM](#system-gmm)
- [Panel Diagnostics](#panel-diagnostics)
- [GWR & MGWR](#gwr--mgwr)
- [Gravity / Flow Models](#gravity--flow-models)
- [GRAS & Cross-Entropy Calibration](#gras--cross-entropy-calibration)
- [HAC-Robust Inference](#hac-robust-inference)
- [Model Selection & Comparison](#model-selection--comparison)
- [Simulation & Power Analysis](#simulation--power-analysis)
- [Output & Tables](#output--tables)
- [Formula Interface](#formula-interface)
- [Package Extensions](#package-extensions)
- [Repository Layout](#repository-layout)
- [Validation Status](#validation-status)
- [Known Limitations](#known-limitations)
- [References](#references)

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

**Dependencies:** `ChainRulesCore`, `ForwardDiff`, `NearestNeighbors`, `StatsAPI`, `StatsModels`, `Tables`, plus standard libraries (`LinearAlgebra`, `SparseArrays`, `Statistics`, `Random`, `Printf`). Julia 1.10 or later.

**Optional extensions:** `Turing.jl` + `Distributions.jl` (Bayesian), `Makie.jl` (visualization) — loaded automatically when these packages are available. See [Package Extensions](#package-extensions).

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
m = fit_spatial(SAR(), W, X, y;
    method = ChebyshevApprox(order=12),
    varnames = ["income", "density"])
println(m)

# Step 3: Spatial effects
println(spatial_effects(m))
```

---

## Modules

The package exports functions, result types and specification tags — not
submodules. Model variants (`SAR`, `SDM`, `SARAR`), fixed-effects specs
(`NoFE`, `IndivFE`, `TimeFE`, `TwoWayFE`) and log-determinant methods
(`ExactLU`, `ChebyshevApprox`) are single shared types usable unqualified:

```julia
m = fit_spatial(SAR(), W, X, y)                       # cross-section
m = fit_spatial_panel(SAR(), W, X, y, N, T; fe = TwoWayFE())   # panel
```

The submodules below organise the source. They are **not** exported, so they
cannot collide with packages of the same name; reach one explicitly when you
want an internal helper:

```julia
const CS = SpatialEconometrics.CrossSection
```

| Submodule | Internal alias | Description | Key Reference |
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
| `SpatialWeights` | `WeightsMod` | k-NN / distance / band weights, standardization, GeoDa I/O | Anselin & Rey (2014) |
| `SpatialModelSelection` | `SelectionMod` | AIC/BIC, LR, Vuong, Wald, common factor tests | Vuong (1989) |
| `SpatialPanelDiagnostics` | `PanelDiagMod` | Spatial Hausman, Chow, joint panel LM, panel Moran | Baltagi, Song & Koh (2003) |
| `HigherOrderSpatial` | `HigherMod` | SLX, SAC/SARAR, SDEM | Halleck Vega & Elhorst (2015) |
| `SpatialRegimeModels` | `RegimeMod` | Regime models with SKATER regionalization | Assuncao et al. (2006) |
| `SpatialDiscreteChoice` | `DiscreteMod` | Spatial probit — linearized GMM and Albert-Chib Gibbs | Klier & McMillen (2008) |
| `SpatialFiltering` | `FilterMod` | Eigenvector spatial filtering (ESF) and RE-ESF | Griffith (2003) |
| `SpatialSimulation` | `SimMod` | Six DGPs, Monte Carlo, power analysis | — |
| `SpatialTables` | `TablesMod` | Coefficient and regression tables, LaTeX / CSV / Markdown | — |

---

## Weight Matrix Construction

Spatial weight matrices from coordinates, regular grids, or GeoDa files.

```julia
# From coordinates (N×2 matrix of (x, y) or (lon, lat))
W = knn_weights(coords, 5;
    binary = false,          # false → inverse distance, true → 0/1
    row_standardize = true,
    symmetrize = false)      # force symmetry via max(W, W')

W = distance_weights(coords, 50.0;
    decay = :inverse,        # :inverse, :negative_exponential, :bisquare, :power
    alpha = 1.0)

W = band_weights(coords, 0.0, 50.0)   # distance band [d_min, d_max]

# From a regular L×L grid
W = make_rook_weights(20)    # 4-connected, N = 400
W = make_queen_weights(20)   # 8-connected

# Combine disjoint regions
W = block_diagonal_weights([W_region1, W_region2, W_region3])

# Standardization — the `!` forms mutate, the others return a copy
row_standardize!(W)       # each row sums to 1
spectral_standardize!(W)  # divide by the spectral radius
minmax_standardize!(W)

# Connectivity report
sc = spatial_connectivity(W)
# sc.N, sc.density, sc.avg_neighbors, sc.min_neighbors, sc.max_neighbors,
# sc.n_islands, sc.is_symmetric, sc.spectral_radius
```

### GeoDa Interchange Formats

```julia
write_gal("weights.gal", W, ids);  W, ids = read_gal("weights.gal")
write_gwt("weights.gwt", W, ids);  W, ids = read_gwt("weights.gwt")

# Row i of W corresponds to ids[i]. A GAL file carries its own ordering in its
# observation records, so read_gal uses that. A GWT file is only an edge list,
# so read_gwt sorts the ids it finds. Pass `ids` to impose the ordering of an
# external table instead:
W, _ = read_gwt("weights.gwt"; ids = df.region_code)
```

A neighbour id that appears in no observation record raises an error rather than
being given an invented index.

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

## Higher-Order Models

SLX, SAC/SARAR and SDEM — the specifications between OLS and full SDM.

```julia
# SLX: y = Xβ + WXθ + ε   — OLS; a sound starting point before SAR/SDM
m = fit_slx(W, X, y; varnames = ["gdp", "pop"])

# SAC / SARAR: y = ρW₁y + Xβ + u,  u = λW₂u + ε
# Separate weight matrices for the lag and error components
m = fit_sac(W_lag, W_error, X, y; varnames = ["gdp", "pop"])

# SDEM: y = Xβ + WXθ + u,  u = λWu + ε   — concentrated ML
m = fit_sdem(W, X, y; varnames = ["gdp", "pop"])
```

---

## Spatial Regime Models

Regression coefficients that vary across groups, with the regimes either given or
discovered.

```julia
# Known groupings — each regime gets its own intercept and slopes
labels = [i <= 100 ? 1 : 2 for i in 1:N]
m = fit_spatial_regime(W, X, y, ExogenousRegimes(labels);
    spatial_lag = true, varnames = ["gdp", "pop"])

# Endogenous, spatially contiguous regimes via SKATER: a minimum spanning tree
# on the connectivity graph, cut to maximise the reduction in within-group SSD
regimes = skater_regimes(W, X, 3; min_size = 10)
m_regime = fit_spatial_regime(W, X, y, regimes)

# Are the coefficients actually different across regimes?
m_pooled = fit_spatial(SAR(), W, X, y)
ct = regime_chow_test(m_pooled, m_regime)   # ct.stat, ct.df, ct.p_value
```

`min_size` guards against the degenerate cuts a pure longest-edge criterion
produces, where a single outlying leaf becomes its own regime.

---

## Spatial Discrete Choice

Binary and ordered outcomes with spatial dependence, all targeting the latent
model `y* = ρWy* + Xβ + ε` with `σ² = 1` fixed for identification.

```julia
# Linearized GMM (Klier & McMillen) — fast, no MCMC; y coded 0/1
m = fit_spatial_probit(W, X, y; varnames = ["gdp", "pop"])

# Bayesian Albert–Chib Gibbs — slower, markedly more precise on ρ
m = fit_bayesian_spatial_probit(W, X, y; n_samples = 2000, burnin = 500)
# m.accept_rate should sit in 0.2–0.5; m.beta_draws and m.rho_draws hold the chain

# Ordered outcomes in 1:J (no intercept, τ₁ = 0 for identification)
m = fit_spatial_ordered_probit(W, X, y_ordered; n_categories = 3)

# Coefficients are NOT marginal effects: a change in xᵢ propagates through (I − ρW)⁻¹
eff = marginal_effects(m)   # eff.direct, eff.indirect, eff.total
```

Linearizing around `ρ = 0` trades precision for speed: at `N = 196` the GMM `ρ̂`
has a standard deviation of roughly 0.25, about 2.5× the Gibbs sampler's. Both
are approximately unbiased, so a single small-sample estimate is a coarse
reading. Fitted probabilities and marginal effects need a dense inverse of
`I − ρW`, so these estimators suit moderate `N`.

---

## Spatial Filtering

Eigenvector spatial filtering absorbs the spatial structure into synthetic
covariates, leaving a standard OLS or GLM problem — no spatial parameter, no
Jacobian.

```julia
# The basis: eigenvectors of M Wᵤ M, ranked by Moran coefficient MCᵢ = (N/S₀)λᵢ
se = spatial_eigenvectors(W; min_moran_ratio = 0.25)

# Stepwise ESF — greedy selection on residual Moran (default) or AIC
m = fit_esf(X, y, W; family = :gaussian, max_eigenvectors = 20,
            criterion = :moran, varnames = ["x1", "x2"])
m = fit_esf(X, counts, W; family = :poisson)   # also :logit, :negbin (m.alpha)
selected_eigenvectors(m)                        # chosen indices and their MCs

# Random-effects ESF — keeps the whole screened pool and shrinks instead,
# with the variance ratio θ = τ²/σ² estimated by profile REML
m = fit_re_esf(X, y, W; varnames = ["x1", "x2"])
```

For RE-ESF, `m.moran_final` is the Moran coefficient of the *conditional*
residual and **is expected to be substantially negative — do not compare it
against zero.** Subtracting `Eγ̂` strips the positively autocorrelated
directions, so compare against `m.moran_reference`, the closed-form expectation
for a random vector in the complement. A value at or above the reference is the
healthy case. Watch `m.edf` too: effective degrees of freedom approaching `N`
means the random effects are absorbing noise.

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
m = fit_gs2sls_panel(TwoWayFE(), W, X_exog, X_endog, Z_iv, y, N, T;
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

## Panel Diagnostics

Specification tests for the panel models above.

```julia
# FE vs RE — reject H₀ → fixed effects preferred
ht = spatial_hausman_test(m_fe, m_re)        # ht.stat, ht.p_value

# Structural break across periods
ct = spatial_chow_test(y, X, W, N, T, break_period)

# Joint LM for spatial lag, spatial error and random effects
# (Baltagi, Song & Koh 2003)
plm = panel_lm_tests(y, X_with_intercept, W, N, T)
# plm.joint_stat (χ², df = 3), plm.lag_stat, plm.error_stat, plm.re_stat

# Period-by-period and pooled Moran on panel residuals
pm = panel_moran(residuals(m), W, N, T)
# pm.period_results — per-period MoranResult; pm.pooled — meta-analytic statistic
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

## Model Selection & Comparison

```julia
spatial_aic(m); spatial_bic(m)

# Nested models (SAR nested in SDM)
lr = lr_test(m_sar, m_sdm)          # lr.stat, lr.df, lr.p_value

# Non-nested models (SAR vs SEM)
vt = vuong_test(m_sar, m_sem)
# vt.stat (positive favours the first), vt.preference ∈ (:model1, :model2, :indistinguishable)

# Does SDM collapse to SEM?  H₀: θ + ρβ = 0
cft = common_factor_test(beta_slope, theta, rho, vcov_full)   # cft.reject

# General linear hypothesis Rβ = r
R = zeros(1, length(coef(m))); R[1, 2] = 1.0   # H₀: β₁ = 0
wt = wald_test(m, R, [0.0])

println(model_comparison_table(m_sar, m_sdm, m_sem; names = ["SAR", "SDM", "SEM"]))
```

---

## Simulation & Power Analysis

Six data-generating processes, Monte Carlo experiments and power curves.

```julia
# SAR, SEM, SDM, SLX, SAC and SDEM DGPs
dgp = SAR_DGP(0.5, [1.5, -0.8], 0.5)        # ρ, β, σ
y, X = simulate_spatial(dgp, W, X)

# Monte Carlo: bias, RMSE and coverage
mc = monte_carlo_spatial(dgp, W, N, 2,
    (W, X, y) -> fit_spatial(SAR(), W, X, y);
    config = MonteCarloConfig(n_replications = 500, seed = 42))
# mc.n_successful vs mc.n_replications — failed replications are reported, not hidden

# Power curves across grid sizes and spatial strengths
pa = power_analysis(ρ -> SAR_DGP(ρ, [1.5, -0.8], 0.5),
    [5, 10, 15, 20],            # grid sizes, N = L²
    [0.0, 0.1, 0.3, 0.5, 0.7];  # ρ values
    n_reps = 200, alpha = 0.05)
# pa.power_moran, pa.power_lm_lag, pa.power_lm_error — the ρ = 0 column is the test's size
```

---

## Output & Tables

```julia
ct = coeftable(m; level = 0.95)   # Tables.jl-compatible: feed to DataFrames, CSV, …

println(to_latex(ct))
println(to_csv(ct))
println(to_markdown(ct))

ss = summary_stats(m; model_type = "SAR")   # N, K, log-lik, AIC, BIC, σ², R²

# Stargazer-style side-by-side comparison
println(regression_table(m_sar, m_sem; model_names = ["SAR", "SEM"]))
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

Two optional feature sets ship as Julia package extensions. Neither is
installed with the package; each activates automatically once its trigger
packages are in the session.

| Extension | Trigger | Provides | Verified against |
|---|---|---|---|
| `SpatialEconometricsMakieExt` | `Makie` | 5 plotting functions | Makie 0.22.10, 0.24.16 |
| `SpatialEconometricsTuringExt` | `Turing` + `Distributions` | 3 Bayesian estimators | Turing 0.44.5, 0.49.0 |

All eight function names are declared and exported by `SpatialEconometrics`
itself; the extensions only add methods to them. Nothing ever writes
`using SpatialEconometricsMakieExt`, so a function an extension declared on
its own would be unreachable — you therefore import nothing but the parent.
Calling one before its trigger packages are loaded names the package to
load rather than raising a bare `MethodError`:

```julia
julia> using SpatialEconometrics

julia> plot_coefficients(m)
ERROR: plot_coefficients requires Makie: run `using Makie` to load the
SpatialEconometricsMakieExt extension, then call it again.
```

### Visualization (Makie.jl)

Install a backend rather than `Makie` itself — `CairoMakie` writes files
headlessly, `GLMakie` opens a window:

```julia
using Pkg; Pkg.add("CairoMakie")
```

```julia
using SpatialEconometrics, CairoMakie

save("coefs.png",   plot_coefficients(m))                 # 95% CI by default
save("coefs99.png", plot_coefficients(m; level = 0.99))   # level is honoured
save("moran.png",   plot_moran_scatter(y, W))             # slope = Moran's I
save("lisa.png",    plot_local_moran(moran_local(y, W), coords))
save("gwr.png",     plot_gwr_coefficients(m_gwr, "income"))
save("effects.png", plot_spatial_effects(spatial_effects(m)))
```

Every function returns a `Makie.Figure`, so the results compose with the
rest of Makie. `level` outside `(0, 1)`, a `varname` the GWR model does not
carry, or `coords` that do not match the LISA result each raise an error
rather than producing a misleading picture.

### Bayesian Spatial Econometrics (Turing.jl)

```julia
using Pkg; Pkg.add(["Turing", "Distributions"])
```

```julia
using SpatialEconometrics, Turing, Distributions

r = bayesian_sar(W, X, y; n_samples = 1000, n_chains = 2,
                 varnames = ["income", "density"], rng = MersenneTwister(11))
r = bayesian_sem(W, X, y)    # dependence in the errors
r = bayesian_sdm(W, X, y)    # design augmented with WX

r.posterior_mean    # aligned with r.coefnames
r.posterior_std
r.chain             # the raw chain object
```

An intercept is added for you, so `X` holds the covariates only. Sampling is
by NUTS, with diffuse priors — `β ~ N(0, 10I)`, `ρ, λ ~ U(-0.99, 0.99)`,
`σ ~ truncated(Normal(0, 5), 0, Inf)` — which is why the posterior mean
lands on the maximum-likelihood estimate for the same data:

```text
  parameter    post. mean   post. SD   MLE          MLE SE
  (Intercept)  1.0377       0.086      1.0364       0.0842
  income       1.4918       0.0425     1.4902       0.0407
  density      -0.7794      0.0324     -0.7798      0.0332
  ρ            0.4707       0.0368     0.4713       0.0359
  σ            0.4327       0.0262     —            —
```

The priors are not configurable, and they are not offered as a considered
prior for any particular application. What the posterior buys you is the
joint distribution: statements such as `P(ρ > 0.6) = 0.0015` need no normal
approximation and no delta method.

Each likelihood evaluation needs `log|I - ρW|`, which NUTS asks for
thousands of times at a new `ρ` every time; the extension uses the Chebyshev
approximation, exact to several decimals across the parameter space, so this
is the likelihood rather than a stand-in for it.

**`result.chain` is not version-stable.** Turing returns an
`MCMCChains.Chains` before v0.45 and a `FlexiChains.FlexiChain` from v0.45
on, and the two index differently for vector-valued parameters.
`posterior_mean` and `posterior_std` are stable across both — prefer them
unless you need the draws themselves.

### Running the extension tests

Testsets 25 and 26 cover the extensions. They skip themselves when the
trigger packages are absent, so a plain `Pkg.test()` does not have to build
Makie and Turing; the `extensions` CI job installs them and runs the suite
again with both live.

See `docs/src/extensions.md` for the full reference,
`examples/18_visualization.jl` and `examples/19_bayesian.jl` for runnable
walk-throughs, and `notebooks/interactive_extensions_explorer.jl` for an
interactive version.

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
│   └── runtests.jl                     # 466 assertions, 23 testsets
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
│   ├── 14_output_tables.jl             # LaTeX / CSV / Markdown export
│   ├── 15_weight_matrices.jl           # knn/distance/band, standardization, GeoDa I/O
│   ├── 16_model_selection.jl           # AIC/BIC, LR, Vuong, common factor, Wald
│   ├── 17_panel_diagnostics.jl         # Hausman, panel LM, Chow, panel Moran
│   ├── 18_visualization.jl             # Makie extension — all five plots
│   └── 19_bayesian.jl                  # Turing extension — NUTS vs the MLE
└── notebooks/
    ├── 01_getting_started.ipynb        # Jupyter
    ├── 02_spatial_panels.ipynb         # Jupyter
    ├── 03_gwr_gravity.ipynb            # Jupyter
    ├── interactive_spatial_explorer.jl        # Pluto
    ├── interactive_gwr_explorer.jl            # Pluto
    ├── interactive_filtering_explorer.jl      # Pluto — ESF / RE-ESF
    ├── interactive_probit_explorer.jl         # Pluto — GMM vs Gibbs probit
    ├── interactive_higher_order_explorer.jl   # Pluto — SLX/SAC/SDEM + regimes
    └── interactive_extensions_explorer.jl     # Pluto — Makie + Turing extensions
```

Run an example with `julia --project=. examples/12_spatial_filtering.jl`. Open a
Pluto notebook with `using Pluto; Pluto.run()` and navigate to the file; each one
activates this project itself.

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
