# SpatialEconometrics.jl — Development Roadmap

> **Current version:** 0.3.0 → 0.4.0 (on `master`, unreleased)
>
> 23 submodules, 2,430 test assertions in 51 top-level testsets with both
> package extensions installed and 2,395 without them — the same counts on
> Julia 1.13.2 and 1.10.12, since the difference is the extensions and not
> the version. 24 examples, 12 notebooks, 2 package extensions, CI/CD and
> 50 Documenter.jl pages. **Phases 4, 5, 6 and 6B are complete.** Item #10 (Performance) closed on 2026-10-30 with its scope
> reduced on measured grounds and may be revisited. Phase 6C, the GWR
> compact local design, is delivered. **Phase 7A is underway:** item #15's
> exposure mapping and spatial difference-in-differences landed on
> 2026-10-30 and its spatial synthetic control on 2026-10-31, leaving its
> shift-share row plus item
> #11 (Networks). Tier 3 is otherwise untouched, and the correctness audit
> in `docs/AUDIT.md` is closed — findings 1–25, with 29–32 added afterwards
> from the benchmark suite and Phase 6C.
>
> Items #15 and #16, and every block marked **Planned additions**, were added
> after that audit closed. They are the extend half of clean → fix → extend.
> #15 is now half delivered — exposure mapping and spatial difference-in-
> differences — and everything else in that group is still unstarted.
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
  - [15. Spatial Causal Inference & Policy Evaluation](#15-spatial-causal-inference--policy-evaluation)
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
  - [16. Spatial GARCH & Non-Geographic W](#16-spatial-garch--non-geographic-w)
  - [14. Documentation & Ecosystem](#14-documentation--ecosystem)
- [Development Phases](#development-phases)
- [References](#references)

Items are numbered in the order they were added to this roadmap, not in
reading order: #15 sits at the end of Tier 1 and #16 at the end of Tier 3.
Numbers #1–#14 are referenced from `docs/AUDIT.md`, the module docstrings
and the phase tables, so they do not get renumbered.

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

**Status:** ✅ Implemented — `src/SpatialPanelDiagnostics.jl` (343 lines). Docs: `docs/src/diagnostics/panel.md`. Example: `examples/17_panel_diagnostics.jl`. Tests: 15 assertions in testset “Spatial Panel Diagnostics”.

**Key functions:**
- `spatial_hausman_test(m_fe::SpatialPanelModel, m_re)` → `HausmanTestResult`
- `spatial_chow_test(y, X, W, N, T, break_period)` → `ChowTestResult`
- `panel_lm_tests(y, X, W, N, T)` → NamedTuple with joint LM statistics
- `panel_moran(residuals, W, N, T)` → Vector of per-period `MoranResult`

**Planned additions — weak vs strong cross-sectional dependence:** the
delivered diagnostics all assume cross-sectional correlation is *weak*, that
is, governed by `W`. The panel literature (Pesaran, Bai, Elhorst) separates
that from *strong* dependence driven by unobserved global factors, which `W`
cannot absorb and which biases `ρ̂` when ignored.

| Feature | Description | Reference |
|---|---|---|
| Spatial panel CCE | Augment the spatial dynamic panel with cross-sectional averages `ȳ_t`, `X̄_t`, `Wȳ_t` to purge common factors | Pesaran (2006); Ertur & Musolesi (2017) |
| Cross-sectional dependence tests | Pesaran CD, and the Bailey–Kapetanios–Pesaran exponent `α` that says whether the dependence is weak or strong | Pesaran (2015); Bailey et al. (2016) |

Interactive fixed effects with spatial lags are the estimation counterpart
and sit under [#12](#12-spatio-temporal-models).

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
- `fit_spatial_probit(W, X, y; instrument_lags=2)` → `SpatialProbitModel` (linearized GMM)
- `fit_bayesian_spatial_probit(W, X, y; n_samples=2000, burnin=500)` — Albert–Chib Gibbs, no weak dependency
- `fit_spatial_ordered_probit(W, X, y_ordered; ...)` → `SpatialOrderedProbitModel`
- `marginal_effects(m::SpatialProbitModel)` → direct/indirect/total

**Dependencies:** None. The Albert–Chib sampler is implemented in
`SpatialDiscreteChoice.jl` itself; the Turing extension covers SAR, SEM and
SDM only, and is not involved here.

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

**Status:** ✅ Implemented — `src/SpatialModelSelection.jl` (1,161 lines). Docs: `docs/src/diagnostics/model_selection.md`. Examples: `examples/16_model_selection.jl`, `examples/21_bayesian_specification.jl`. Tests: 16 assertions in testset “Model Selection & Comparison”, 25 in testset 28 (`select_knn_w`) and 88 in testset 30 (the Bayesian comparison and the roadmap).

**Key functions:**
- `spatial_aic(m)`, `spatial_bic(m)` → corrected information criteria
- `lr_test(m_restricted, m_unrestricted)` → `LRTestResult`
- `vuong_test(m1, m2)` → `VuongTestResult`
- `common_factor_test(m_sdm)` → `CommonFactorTestResult` (Wald-type)
- `wald_test(m, R, r)` → `WaldTestResult` (Rβ = r)
- `model_comparison_table(models...)` → formatted comparison table
- `log_marginal_likelihood(spec, y, X, W)` → one specification's integrated likelihood
- `bayes_spec_probs(y, X, Ws)` → `SpecEvidence`, the posterior over specifications and `W`
- `sdem_roadmap(y, X, Ws)` → `SpecRoadmapResult`, the local-spillover search

**Planned additions — the specification search itself:** the delivered tools
compare two models the user has already fitted. The 2023 *Spatial Economic
Analysis* special issue is largely about the search that comes first: which
spatial lags belong in the model, and which `W` to run it on.

| Feature | Description | Reference |
|---|---|---|
| ~~Durbin score tests~~ — **delivered**, see `sdm_tests` | Five Rao score statistics for the spatial Durbin model: joint `ρ = 0, θ = 0`; `ρ = 0` with and without `θ ≠ 0`; `θ = 0` with and without `ρ ≠ 0`; plus significance of each individual Durbin term | Koley & Bera (2023); Bera & Yoon (1993) |
| ~~SDEM specification roadmap~~ — **delivered**, see `sdem_roadmap` | Iterative search over {SDE, SLX, SE} × `W`, for the local-spillover case where the Durbin terms sit in the error | Sutton & Sutton (2023) |
| ~~Bayesian SD vs SDE comparison~~ — **delivered**, see `bayes_spec_probs` | Posterior model probabilities over global versus local spillovers, jointly with the `W` specification | LeSage (2014, 2015) |

The score tests were the closest fit to what was already here, and are now
delivered: `SpatialDiagnostics.jl` ships `lm_test_sdm`, `lm_test_durbin`,
`lm_test_robust_durbin`, `lm_test_robust_lag_durbin` and
`lm_test_durbin_terms`, collected by `sdm_tests`, extending the existing
`lm_test_lag`/`lm_test_error` battery from SAR/SE to the Durbin model under
the same Rao score framework. Their measured null distribution and the
robust-size property are in `docs/AUDIT.md`; `examples/20_specification_search.jl`
runs the battery end to end. `W` selection by AIC is in
[#6](#6-weight-matrix-construction).

The remaining two are now delivered as well, and answer a question the score
battery cannot: `:sdm` and `:sdem` have the same design and the same number
of parameters, so no likelihood-ratio test runs between them.
`log_marginal_likelihood` integrates a specification's likelihood over its
parameters — `β` and `σ²` analytically, the spatial parameter by quadrature —
which puts designs of different width on one scale;
`bayes_spec_probs` normalises that over a grid of specifications **and**
weight matrices and reports both margins plus the global/local split;
`sdem_roadmap` is the Sutton & Sutton search within the local case, with the
margin to the runner-up on each `W`. Selection rates over 200 draws from
each of six specifications are in `docs/AUDIT.md`;
`examples/21_bayesian_specification.jl` runs the comparison end to end.

**Dependencies:** None

---

### 15. Spatial Causal Inference & Policy Evaluation

**Motivation:** SUTVA is the assumption that one unit's treatment does not
affect another's outcome, and geographic spillovers violate it by
construction — which is the whole premise of this package. Applied work has
moved from reading SAR/SEM coefficients as associations toward decomposing a
policy effect into a direct effect on the treated and an indirect effect on
their neighbours. Nothing in Tiers 1–3 currently addresses a treatment
design.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| ~~Spatial difference-in-differences~~ — **delivered**, see `fit_spatial_did` | Direct ATT and spillover ATT through an exposure mapping on `D` and `WD`, two-period and staggered. Four effects rather than three: the spillover onto the treated and onto the controls are evaluated at different exposure means and are not interchangeable | Delgado & Florax (2015); Chagas et al. (2016) |
| ~~Spatial synthetic control~~ — **delivered**, see `fit_synthetic_control` | Donor-weight optimisation on the unit simplex, penalised by geographic or economic distance, with the donor pool screened so the synthetic unit is not assembled from the treated unit's own neighbours. Permutation inference, and the neighbours' own gaps read back as an estimated spillover | Abadie, Diamond & Hainmueller (2010); Abadie & L'Hour (2021) |
| Spatial shift-share diagnostics | Exogeneity and spatial-correlation tests for local exposure shares and regional industry weights | Borusyak et al. (2022); Adao et al. (2019) |
| ~~Exposure mapping utilities~~ — **delivered**, see `exposure` / `ring_exposure` / `check_interference` | `D`, `W^k D`, ring and distance-band exposures. The assumption check is the substantive part: `δ` is identified only by controls that differ in exposure, which is a property of the design and `W` alone and so is checkable before any fit | — |

**Status:** 🟡 **Three of four delivered** — exposure mapping and
spatial DiD on 2026-10-30, spatial synthetic control on 2026-10-31, all in
`src/SpatialCausal.jl` (1,920 lines). Docs:
`docs/src/models/causal.md`. Examples: `examples/23_spatial_causal.jl`,
`examples/24_synthetic_control.jl`. Tests: 162 assertions in testset 34 and
197 in testset 35. Ledger: `docs/AUDIT.md`, under the standing rule.
Only the shift-share diagnostics remain.

**Delivered API:**

```julia
exposure(D, W; order = 1, standardize = true)   -> ExposureMap
ring_exposure(D, coords, d_min, d_max)          -> ExposureMap
check_interference(em)                          -> InterferenceReport
fit_spatial_did(y, treat, W; X, vcov, coords, dist_cutoff, order)
    -> SpatialDiDResult

fit_synthetic_control(Y, treated, T0; W, coords, predictors, donor_pool,
                      penalty, penalty_metric, metric)
    -> SyntheticControlResult
placebo_test(scm, Y; n_placebo, rng, kwargs...)  -> PlaceboTest
spillover_gaps(scm, Y; kwargs...)                -> SpilloverGaps
```

`treat` is a length-`N` indicator for two periods or an `N × T` path for a
staggered rollout, which is what selects the estimator. Four effects are
reported — `att_direct`, `att_spill_treated`, `att_spill_control`,
`att_total_treated` — each a fixed linear combination of `θ̂`, so their
standard errors are exact.

**The `clusters` decision, resolved by measurement rather than by
reasoning.** The open question here was whether to add a cluster-robust
covariance or map the keyword onto the two existing estimators.
`ClusterRobustVCOV` and `vcov_cluster` were added to `src/SpatialHAC.jl`
(+120 lines) — and then the default went to clustering rather than to
Conley, which is the opposite of what was planned. Conley's coverage
*degrades* monotonically as the cutoff widens, and it does so identically
when there is no spatial correlation at all, so the decay is the
finite-sample cost of estimating more pair covariances rather than the
correction working. The reason the spatial correction matters less than
expected is the exposure column itself: including `WD` identifies `τ` off
the contrast between a unit and its neighbours, which differences out the
error component they share. Nine configurations, the full table and the
`ρ = 0` control are in `docs/AUDIT.md`.

**Verification:** the two-period design first-differences to an exact OLS
problem, so the standard is algebra and not a tolerance — `5.3e-15` on
`att_direct` from a noiseless design, and
`att_total_treated == att_direct + att_spill_treated` exactly. The
staggered path is `4.7e-15` under homogeneous effects and **badly biased
under heterogeneous ones**: 17% to 67% across four designs, with a sign
that is not predictable, worst when adoption cohorts are spatially
contiguous because cohort identity is then nearly collinear with the
exposure column. All eight figures are pinned in testset 34e so the
docstring's warning cannot drift from the code.

**Synthetic control, delivered 2026-10-31.** `treated` takes one unit or
many; each gets its own weights and the ATT is the mean of their
post-period gaps. `V = I` on standardised predictors rather than a nested
search over `V`, deliberately: the outer problem is non-unique, so two
equally good `V` imply different counterfactuals and the estimate stops
being reproducible.

**Verification is the exact optimum, not a tolerance.** The weights solve a
convex QP over the unit simplex, and for a small donor pool the optimum is
computable independently by enumerating all `2^J − 1` supports. Over 500
random problems with a unique optimum the objective agrees to `3.55e-15`
and the weights to `1.31e-14`, with a relative KKT gap of `9.68e-15` and no
violation. Where there are more donors than pre-periods the solution set is
a face, so only the objective is defined — that non-uniqueness is the
argument for the penalty rather than a defect.

**Donor-pool contamination turned out to be an identity.** The spillover
falls in the post-treatment periods only, so the weights — fitted
pre-period — are unchanged by it to `0.00e+00`. Hence
`att(contaminated) = att(clean) − Σ w_j · E_j · spillover` exactly, pinned to
`1e-12` at four spillover sizes. **And the contaminated fit has the
*better* pre-period fit** — 0.045 against 0.097 in the example, while
carrying 90.1% of its weight on treated territory and being wrong by
−0.433 against the screened fit's +0.005. A pre-period fit diagnostic
cannot detect this failure, because the contamination is post-treatment by
construction.

**The placebo test is better behaved than the plan assumed.** Permutation
inference needs exchangeable units and spatial correlation breaks that, so
over-rejection was expected. Measured over 1,000 replications with no
effect anywhere, it is correctly sized at `ρ = 0` and `ρ = 0.6` (0.046 and
0.045 at nominal 5%, mean p-value 0.50) and over-rejects by about a third
of nominal only at `ρ = 0.9` (0.068). The ratio statistic is why: scaling
the gap by the pre-period fit makes spatial dependence largely divide out.
The binding limit is granularity instead — the smallest attainable
p-value is `1/(1+R)`, so one treated unit with 18 donors cannot reject at
5% whatever the effect, and `min_attainable_p` reports it.

**A citation was withdrawn.** This row cited Cerqua et al. (2022), already
flagged below as unverified. It does not exist as described; the reference
set is now Abadie, Diamond & Hainmueller (2010), Abadie & L'Hour (2021),
Abadie (2021) and Di Stefano & Mellace (2024), each checked against the
published record.

**Not delivered, deliberately:** the shift-share exogeneity diagnostics.
They are test statistics, so each owes a null-distribution Monte Carlo
under the standing rule, and bundling estimator families into one commit is
the shape that produced audit findings 13–16.

**Actual complexity:** 910 lines of `src/SpatialCausal.jl` plus 120 in
`SpatialHAC.jl`, against an estimate of ~600 for a scope that included the
two deliverables above.

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

**Status:** ✅ Implemented — `src/SpatialWeights.jl` (526 lines). Docs: `docs/src/weights.md`. Example: `examples/15_weight_matrices.jl`. Tests: 22 assertions in testset “Spatial Weights Construction”.

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

**Planned additions — choosing and estimating `W`:** the delivered
constructors all take `W` as given. The persistent criticism of applied
spatial work is that this choice is arbitrary, and the two responses in the
literature are to select `W` by a criterion or to estimate it.

| Feature | Description | Reference |
|---|---|---|
| ~~knn `W` selection by AIC~~ — **delivered**, see `select_knn_w` | Refit the same model for `k = 2…50` and read AIC against `k`; the curve is non-monotone with an interior minimum, and a wrong `k` biases `ρ̂` badly — measured at −35.1% at `k = 2` and +29.1% at `k = 20` against a true `k = 6`, wider than the source's "up to 20%" because the range searched is wider | Kubara & Kopczewska (2023) |
| Endogenous / gravity-formed `W` | Parameterised `w_ij = f(d_ij, Z_ij; γ)` estimated jointly with the model by concentrated ML or GMM | Kelejian & Piras (2014); Qu & Lee (2015) |
| Time-varying `W` | `W_t` from bilateral flows, with the panel estimators taking a vector of weight matrices | Lee & Yu (2012) |

The AIC-over-`k` search was nearly free, and is now delivered as
`select_knn_w`: `knn_weights` and `spatial_aic` both existed, and the loop
between them plus its guards was the deliverable. The caveat from the source
— comparing AICs across different `W` is valid only when the dependent
variable and the sample do not change between fits — is checked rather than
documented, and a candidate whose fit fails or whose `ρ̂` pins to the bound
is excluded from the argmin instead of winning it by having no number.

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

**Status:** ✅ Implemented — `src/SpatialFiltering.jl` (1,426 lines). Docs: `docs/src/models/filtering.md`. Examples: `examples/12_spatial_filtering.jl`, `examples/22_spatiotemporal_pca.jl`. Tests: 47 assertions in testset “Spatial Filtering” and 192 in testset 31 (spatio-temporal components).

**Key functions (delivered):**
- `spatial_eigenvectors(W; n_candidates=nothing, min_moran_ratio=0.0)` → `SpatialEigen`
- `moran_coefficient(x, W)` → Moran coefficient of a vector
- `fit_esf(X, y, W; family=:gaussian, max_eigenvectors=20, criterion=:moran)` → `ESFModel`
  (families `:gaussian`, `:poisson`, `:logit`, `:negbin`; `criterion` `:moran` or `:aic`)
- `selected_eigenvectors(m::ESFModel)` → indices and Moran values of selected eigenvectors
- `fit_re_esf(X, y, W; min_moran_ratio=0.25)` → `REESFModel` (profile-REML shrinkage)
- `stpca(X, W; measure=:geary|:moran, n_basis=5, times, interval)` → `STPCAResult`
  (spatio-temporal principal components: `n × p × T` data, Fourier expansion in
  time, components maximising variance × spatial autocorrelation)
- `component_groups(m)` → the global (positive) and local (negative) components
- `feature_contributions(m, k)`, `component_weight(m, k, t)` → what a component reads
- `fourier_basis(times, n_basis; interval)` → the orthonormal basis itself

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

**Spatio-temporal principal components — delivered, see `stpca`:** the same
eigenvector machinery supports exploratory analysis, not just filtering.
Each unit-feature series is expanded in an orthonormal Fourier basis, and
the components maximise the product of a functional principal component's
variance and the spatial autocorrelation of the scores it produces — which
collapses to an eigenproblem whose eigenvalues *are* that product.

| Feature | Description | Reference |
|---|---|---|
| ~~Spatio-temporal principal components~~ — **delivered** | Moran-weighted eigendecomposition in the functional coefficient space, with signed eigenvalues splitting global from local components | Krzyśko et al. (2024), *Spatial Economic Analysis* 19(1), eq. (21) |
| ~~Geary's contiguity variant~~ — **delivered** | The same construction with Geary's `C` in place of Moran's `I`, which is non-negative definite and so has no sign to read | Krzyśko et al. (2023), *CEUS* 103, eq. (4) |

The eigenvalue signs are the substantive output under `:moran`: positive is a
global component, negative a local one, and `component_groups` returns the
two groups in the reference's order. The basis depends on `W` throughout, so
the choice of `W` remains the substantive decision — the reference compares
nine and offers no rule.

**Not delivered:** the permutation test of the Moran paper's section 5, which
tests the joint significance of the positive and negative groups. It is a
test statistic, so it falls under the standing rule in `docs/AUDIT.md` and
needs its own null distribution before it can be believed.

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
| Multithreaded GWR — ✅ **delivered** | `Threads.@threads` over local regressions in `fit_gwr`. Not a linear speedup: 4.0–6.3× on 16 cores, held back by an `O(N²)` allocation and a dense local design, both recorded in `benchmark/README.md` |
| Sparse Chebyshev — ✅ **bounded, not replaced** | The recursion still densifies, because on a sparse `W` that is the fast path; `max_fill_bytes` now caps the peak and finishes the remaining orders a column block at a time. Keeping sparsity throughout would be 4–25× slower on a lattice |
| GPU spatial lag — ⏸️ **parked** | CUDA.jl kernel for `W * y` when N > 50,000. Targets the log-determinant path, measured at under 2% of a fit, and adds a weakdep to do it |
| Out-of-core panels — ⏸️ **parked** | Arrow.jl / DuckDB.jl streaming for `fit_spatial_panel`. A memory axis rather than a speed one, and two more weakdeps with no measured demand |
| Precompiled sysimage — ⏸️ **deferred** | PackageCompiler.jl recipe for zero-latency interactive use. Cheap and dependency-free; deferred on scheduling, not merit |
| Benchmark suite — ✅ **delivered** | `benchmark/`: BenchmarkTools.jl suite across N sizes, W densities and Chebyshev orders, with `--compare` against a saved baseline |
| Matrix-free log-determinant — ⏸️ **parked** | Stochastic Lanczos quadrature for `log det(I − ρW)` — Aune et al. (2014); Barry & Pace (1999). Still the right tool at the `N` where Chebyshev fill-in binds, but that is beyond anything benchmarked, and the whole path is under 2% of a fit |

The GWR and MGWR degrees of freedom are **not** part of this item: the
Hutchinson trace estimator and the recursive back-fitting accumulator are
already in `src/GWREstimation.jl` (`_mgwr_enp_hutchinson`, `df_method` of
`:exact`, `:stochastic` or `:none`), with a Monte Carlo standard error on
each `enp`. What remains here is the same idea applied to the spatial
log-determinant, where only the Chebyshev approximation exists today.

**Status:** ✅ **Closed 2026-10-30, scope reduced on measured grounds** —
may be revisited. Six passes delivered: the benchmark suite, the
trace-powers rewrite, threaded GWR, the `knn_weights` selection loop, the
reused log-determinant factorisation and the Chebyshev memory ceiling.
`benchmark/README.md` has every before-and-after number and
`docs/AUDIT.md` findings 29–31 the ledger.

| Delivered | Result |
|---|---|
| Benchmark suite (`benchmark/`) | `julia --project=benchmark -t auto benchmark/run.jl`, `--compare` against a saved baseline |
| `row_standardize!` was O(N · nnz) | 1,677× at `N = 8100`, output bit-identical (finding 29) |
| `precompute_neumann_traces` relaid out | 3.4–12.2× on the probe branch, **1.8–2.7× on `fit_spatial`** at `N ≥ 3600` (finding 30) |
| `fit_gwr` threaded, two allocation faults fixed | **4.0–6.3×** on 16 cores, 1.6–2.0× on one, bit-identical (finding 31) |
| `log_marginal_likelihood` reuses its symbolic factorisation | 1.4–1.5×, all 200 log-determinants bit-identical (finding 31) |
| Chebyshev recursion given a memory ceiling | `max_fill_bytes`, 128 MiB default, blocked fallback (finding 31) |

**Why it closes short of its own deliverables list.** Three of the seven —
GPU spatial lag, matrix-free log-determinant, out-of-core panels — were
written before anything was measured, and measurement does not support
them. The first two target `log det(I − ρW)`, which is **under 2% of
`fit_spatial`**: `brent_maximize` calls the concentrated likelihood six
times, so no implementation of either can return more than about 2%, and
the GPU one adds a CUDA.jl weakdep to do it. Out-of-core panels addresses
a different axis — memory for panels larger than RAM, not speed — and
carries two more weakdeps with no measured demand behind it. They are
parked here rather than deleted; the profile that would justify them is a
much larger `N` on a much denser `W` than anything benchmarked.

The precompiled sysimage recipe is deferred too, on scheduling rather than
on merit: it costs little and has no dependencies, and it is worth doing
when interactive latency is the complaint.

**What measurement actually found, which the item never mentioned.** 71–92%
of a fit was `spatial_trace_powers`; the dominant cost of GWR was
allocation rather than parallelism; two of the five findings from the first
pass were wrong and had to be withdrawn — `knn_weights` is O(N log N), not
O(N²), and the Chebyshev "461 MiB" was cumulative allocation rather than a
peak. Both corrections are in `docs/AUDIT.md` finding 31. The general
lesson is in finding 29: a benchmark suite is not only for tuning, and for
a defect in asymptotic cost it is the only instrument that sees it at all.

**One optimisation carried forward rather than parked** — the GWR compact
local design, listed as priority 12 under Development Phases. `fit_gwr`
builds a dense `N × K` weighted design per observation when only `bw` rows
are nonzero, about 4.5× wasted arithmetic at `FixedBandwidth(8.0)` on a
30×30 grid, and it compounds with the threading now in place. It is carried
forward and not closed with the rest of this item because the measurement
supports it, unlike the three above. It is not inside this item because a
compact design reorders the QR's rows and so moves reported coefficients
in their last bits — the same trade as findings 30 and 31, but it deserves
its own pass rather than being folded into a closure.

**What the first measurements changed.** Three of the seven deliverables
above — sparse Chebyshev, GPU spatial lag, matrix-free log-determinant —
target the log-determinant, which is **under 2% of `fit_spatial`**. The
Brent search calls the concentrated likelihood six times, so neither the
precompute nor the per-evaluation cost matters. 71–92% of a fit is
`spatial_trace_powers`, which this item never mentioned.

| Finding | Status |
|---|---|
| `row_standardize!` was O(N · nnz), not O(nnz) — 1,677× at `N = 8100`, output bit-identical | ✅ Fixed, testset 32, finding 29 |
| `spatial_trace_powers` is 71–92% of `fit_spatial` | ✅ Fixed, testset 32b — 3.4–12.2× on the probe branch, 1.8–2.7× on `fit_spatial` at `N ≥ 3600` |
| `max_exact_n = 1000` picks the dense `eigvals` branch, now **51×** slower than probes at `N = 900` (0.3 s against 6.0 ms) and O(N³)/O(N²)-memory above it | ✅ Decided — default kept, exactness preferred; the 51× is documented in the docstring |
| ~~The default `ChebyshevApprox(n_probes = 0)` densifies: 461 MiB at `N = 8100, q = 12`~~ — **461 MiB was cumulative allocation; peak fill is 18.9 MiB.** The `Θ(N²)` ceiling is real but arrives later (90 MiB at knn `k = 24, N = 3600`) | ✅ Bounded by `max_fill_bytes`, 128 MiB default, blocked fallback, finding 31 |
| No `Threads` call anywhere in `src/` — GWR's local regressions are embarrassingly parallel; the probe loop is now threaded but is bandwidth-bound, so it gained 4% at `N = 14400` | ✅ `fit_gwr` threaded: 4.0–6.3× at 16 threads, 1.6–2.0× at one, bit-identical, testset 32c |
| ~~`log_marginal_likelihood` makes the same `max_exact_n` mistake~~ — the 200 LUs are 97–99% of the call, but the fix is not to drop exactness: the pattern of `I − ρW` is fixed, so the symbolic factorisation is reused | ✅ 1.4–1.5×, all 200 log-determinants bit-identical, testset 32e |
| ~~`knn_weights` is O(N²) — 820 ms at `N = 8100, k = 12`~~ — **wrong on both counts.** It is O(N log N) via a KD-tree and takes 26 ms; the 820 ms was mostly the `row_standardize!` quadratic it calls by default | ✅ Corrected; its selection loop fixed anyway, 1.2–1.5× and 3× less allocation, testset 32d |

The deprioritised three are not wrong, and the matrix-free log-determinant
is still the right tool at the `N` where the Chebyshev fill-in binds. They
simply cannot pay off on the measured profile, and two of them carry
optional dependencies, so they go after the work above.

**The `max_exact_n` decision is not the implementer's to make.** Below
`max_exact_n` the trace powers are exact, from the eigenvalues of `W`;
above it they are Hutchinson estimates with a roughly 6% standard error at
250 probes. `spatial_effects` reads these traces to split a marginal
effect into direct and indirect parts, and the standard errors come from
the same place, so lowering the threshold would make reported impacts
stochastic for every `N <= 1000` — which is most applied work with this
package, and is the exact surprise the current default was chosen to
avoid. The price of that choice has risen from about 4.6x to 51x, and
there is no third option: the exact branch needs the original adjacency
and its degrees to be symmetrised into a fast real eigenproblem, and
`spatial_trace_powers` only ever receives the standardised `W`.


**Files touched:** `benchmark/` (new, delivered), `src/SpatialUtils.jl`,
`src/SpatialWeights.jl`, `src/GWREstimation.jl`,
`src/SpatialModelSelection.jl`, testset 32. `src/SpatialGPU.jl` was never
created, the GPU deliverable being parked.

**Dependencies:** none added. The three weakdeps this item once planned —
CUDA.jl, Arrow.jl, DuckDB.jl — went unused, which is part of why it closed
where it did.

**Actual complexity:** ~390 lines of `src/` and ~270 of tests, against an
estimate of ~600 for a scope that included three deliverables now parked.

**Verification note, kept for if this is revisited:** the parked
deliverables are all testable on the current development machine — 16
logical cores, an RTX 3080 (compute capability 8.6, 16 GB VRAM,
`CUDA.functional()` confirmed) and ample free disk for a panel that
genuinely exceeds the 32 GB of RAM. CI runners have no GPU, so any CUDA
path must be gated on `CUDA.functional()` behind a package extension,
following the pattern the Turing and Makie extensions already use.

Everything delivered was verified on Julia 1.13.2 **and** 1.10.12, since
`randn(::MersenneTwister)` streams differ between minor releases. Each
optimisation was held to a stated standard: bit-identical where exactness
was the point (`row_standardize!`, the log-determinant sweep, `fit_gwr`,
`knn_weights` across 144 configurations), and roundoff-with-the-reordering
-named where it was not (the trace kernel at 7e-14, the Chebyshev fallback
at 1e-14, both against estimator errors of order 6%).

---

## Tier 3: Niche but Valuable

### 11. Spatial Econometrics for Networks

**Motivation:** Traditional spatial econometrics assumes geographic proximity drives interactions. Modern applications increasingly involve non-geographic networks: trade links, social connections, supply chains, citation networks. Network autocorrelation models generalize spatial models to arbitrary graph structures.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Network autocorrelation model | `y = ρWy + Xβ + ε` where W is derived from network adjacency (not geography) | Leenders (2002) |
| Network weight construction | `network_weights(adjacency; normalize=:row)` — from adjacency matrix, edge list, or graph | — |
| Peer effects model | `y_i = ρ * mean(y_peers) + Xβ + ε` — endogenous vs exogenous peer effects (Manski reflection problem) | Manski (1993); Bramoullé et al. (2009) |
| Network centrality instruments | Construct instruments from higher-order network structure (friends-of-friends) to address endogeneity | Bramoullé et al. (2009) |
| Multiplex spatial models | Multiple W matrices representing different interaction channels (trade + migration + FDI) | LeSage & Pace (2009, Ch. 9) |

**Refinement:** the Manski reflection problem and Bramoullé instruments are
already two of the rows above; what the recent literature adds is making the
identification check mechanical rather than assumed — automated rank
validation of the `W²X`, `W³X` instrument set, reported alongside the fit so
a user sees when the exclusion restriction has nothing to stand on.

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
| Interactive fixed effects with spatial lags | Joint estimation of a factor structure `λᵢ'f_t` alongside `Wy` and `WX`, separating genuine spatial interaction from global shocks | Bai (2009); Shi & Lee (2017) |

The factor-structure row is the estimator that pairs with the cross-sectional
dependence diagnostics under [#1](#1-spatial-panel-data-toolkit): the tests
say whether a factor structure is there, this estimates it.

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
| Spatial quantile regression | Effects vary across the conditional distribution: `Q_τ(y \| X, W) = ρ_τ Wy + Xβ_τ` | Kostov (2009) |
| Spatial varying coefficient model | Generalization of GWR with penalized spline smoothing — avoids GWR's boundary bias and bandwidth sensitivity | Hastie & Tibshirani (1993); Geniaux & Martinetti (2017) |
| Penalized spatial regression | LASSO / elastic net with spatial lag: variable selection in high-dimensional spatial models | Liu & Yang (2020) |
| Additive spatial model | `y = ρWy + Σ_k f_k(x_k) + ε` — spatial GAM with backfitting | — |

**New file:** `src/SemiparametricSpatial.jl` (~500 lines)

**Dependencies:** None for P-splines (implement basis construction); optional MLJ.jl for penalized regression

**Estimated complexity:** ~500 lines

---

### 16. Spatial GARCH & Non-Geographic W

**Motivation:** Every model in this package puts the spatial structure in
the conditional mean. Financial and risk applications need it in the
conditional *variance*: volatility clusters across firms and assets that are
close in some economic sense, not in kilometres. Fülle & Otto (2023) fit a
spatial log-GARCH to the 50 largest Dow Jones Global Titans constituents
with `W` built from covariate distance, and find a seven-covariate
non-geographic `W` beats every geographic candidate. This is also the
cleanest demonstration that the package's machinery is not about geography.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Spatial GARCH | Conditional variance with a spatial lag, `log h = α + ρ W log h + ...`, estimated by QML | Fülle & Otto (2023) |
| Covariate-distance W | `W` from Euclidean distance over a covariate set rather than coordinates, with forward selection over covariates | Santi et al. (2021); Fülle & Otto (2023) |
| Endogeneity caveat | Diagnostics for feedback from `y` into the covariates that define `W` — the failure mode the editorial flags | Qu & Lee (2015) |

**New file:** `src/SpatialGARCH.jl` (~450 lines)

**Dependencies:** None

**Estimated complexity:** ~450 lines

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

### Phase 6 — ✅ Complete

| Priority | Item | Est. | Actual | Status |
|---|---|---|---|---|
| 9 | Spatial Probit / Logit (#4) | ~500 | 700 | ✅ |
| 10 | Spatial Filtering (#7) | ~350 | 718 | ✅ |
| 11 | Performance (#10) | ~600 | 390 | ✅ Closed with scope reduced; three deliverables parked, see #10 |
| | **Phase 6 total** | **~1,450** | | |

### Phase 6B — ✅ Specification search & `W` selection

Small, cohesive, and all of it extends modules that are already finished and
audited. Cheap to build and cheap to verify, which is why it sits before the
Tier 3 builds rather than inside them.

| Priority | Item | Est. Lines | Rationale |
|---|---|---|---|
| ~~12~~ | ✅ Durbin score tests (#5) | 350 | Delivered: `sdm_tests`, testset 27, null distribution in `docs/AUDIT.md` |
| ~~13~~ | ✅ knn `W` selection by AIC (#6) | 210 | Delivered: `select_knn_w`, testset 28, recovery rates in `docs/AUDIT.md` |
| ~~14~~ | ✅ SDEM roadmap + Bayesian SD/SDE (#5) | 515 | Delivered: `log_marginal_likelihood`, `bayes_spec_probs`, `sdem_roadmap`, testset 30, selection rates in `docs/AUDIT.md` |
| ~~15~~ | ✅ Spatio-temporal principal components (#7) | 400 | Delivered: `stpca` under Moran's I and Geary's C, testset 31, recovery rates in `docs/AUDIT.md` |
| | **Phase 6B total** | **~1,200** | |

### Phase 6C — ✅ Complete

One optimisation carried forward out of #10 rather than parked with the
rest, because it is the one the measurements supported. It sat on its own
because it changes reported numbers, which a closure note should not.

| Priority | Item | Est. | Actual | Status |
|---|---|---|---|---|
| 12 | GWR compact local design (#10) | ~250 | 251 | ✅ **1.6–6.5× single-threaded, 2.3–5.0× on 16**, allocation down 8–24×, finding 32, testset 33 |
| | **Phase 6C total** | **~250** | | |

`fit_gwr` and `fit_mgwr` built a dense `N × K` weighted design per
observation when only the `b` rows inside the kernel's support carried
anything, making the cost `Θ(N² K)` where `Θ(N · b · K)` does. The gain
tracks `1 / (b/N)`, so it grows with `N` at a fixed bandwidth rather than
being a constant factor — cumulatively, `FixedBandwidth(8.0)` at `N = 900`
went from 105.4 ms to 6.68 ms across #10 and 6C together, **15.8×**.

Coefficients move by 3e-14 and `tr_S` by 1.7e-16, the stated standard being
roundoff rather than equality; the AICc margin that decides a bandwidth is
`1e11` eps, so no selected bandwidth changes. Both are checked against the
dense form, which is kept in `_get_weights` for that purpose.

Not fixed, and recorded rather than quietly left: MGWR gains only 1.22×,
because its cost is `_select_bandwidth_aicc` re-scanning ~26 bandwidths per
column per sweep rather than the local designs. Cheapening that means not
re-scanning a bandwidth that has stopped moving, which changes which fixed
point the back-fitting reaches — a decision, not a tidy-up.

### Phase 7A — 🟡 Networks & causal policy evaluation

| Priority | Item | Est. Lines | Status |
|---|---|---|---|
| 13 | Spatial Causal Inference (#15) | ~600 | 🟡 **Three of four delivered.** Exposure mapping and spatial DiD 2026-10-30; spatial synthetic control, its permutation inference and the neighbour spillover read-out 2026-10-31. 2,040 lines across `SpatialCausal.jl` and `SpatialHAC.jl` against an estimate of ~600 for a scope that excluded two of these. Shift-share diagnostics outstanding |
| 14 | Spatial Networks (#11) | ~400 | ⬜ Extends beyond geography. Cheaper than #15 but mostly reframes what `W` may represent rather than adding inference, so it goes second |
| | **Phase 7A total** | **~1,000** | |

The #15 row's original rationale said it "builds on `spatial_effects` —
already written and audited". That turned out to be half true and is worth
recording: what is reusable is the *pattern* — a delta method over a closed
form with a swappable covariance — not the function, which hard-codes the
panel coefficient layout (intercept offset, `ρ` last, `K = P_cov` or
`P_cov/2`) that a DiD design does not have. In the event the two-period
decomposition needed no delta method at all, every reported effect being
linear in `θ̂`.

The synthetic control pass reused more than expected, and from a different
place: `exposure` (written for the DiD) finds the neighbours to screen out
of the donor pool, and `SpatialHAC`'s `pairwise_distance` with its
`EuclideanDistance` / `HaversineDistance` types supplies the penalty metric,
so lon/lat data works without new code. What it could **not** reuse was an
optimiser — the package has no Optim or JuMP dependency and none was added;
the simplex-constrained program is solved by accelerated projected gradient
with an exact Lipschitz step, about 110 lines, and verified against an
independent support enumeration.

### Phase 7B — ⬜ Spatio-temporal, common factors & semiparametrics

| Priority | Item | Est. Lines | Rationale |
|---|---|---|---|
| 18 | Spatio-Temporal Models (#12) | ~500 | STAR, SpVAR, forecasting, interactive FE |
| 19 | Common factors in panels (#1) | ~400 | SP-CCE and the CD / `α` exponent tests |
| 20 | Semiparametric Spatial (#13) | ~500 | Quantile regression, penalized models |
| 21 | Spatial GARCH (#16) | ~450 | Spatial structure in the variance |
| | **Phase 7B total** | **~1,850** | |

### Phase 8 — ⬜ Ecosystem, benchmarks & JSS submission

| Priority | Item | Est. Lines | Rationale |
|---|---|---|---|
| 22 | Full documentation & paper (#14) | ~2,000 | JSS paper, tutorials, benchmarks |
| 23 | Cross-language benchmarks | ~400 | Against R `spatialreg` and Python `PySAL/spreg` |
| | **Phase 8 total** | **~2,400** | |

### Size: Actuals and Projections

Rows above the rule are measured; rows below are estimates. v0.4.0 has
crossed the rule: it is not a projection any more, it is what `master`
holds today, and the figures it once projected — ~1,200 assertions and
~13,800 lines — were passed some time ago.

| Milestone | Modules | Test assertions | Lines of `src/` |
|---|---|---|---|
| **v0.3.0** | 13 | 54 | ~6,500 |
| **master** (Phases 4–6 before #10, audit closed) | 22 | 1,156 | 13,331 |
| **master** (+ Phase 6B items 1–2) | 22 | 1,270 | 13,995 |
| **master** (+ Phase 6B item 14, identification guards) | 22 | 1,374 | 14,843 |
| **master** (+ Phase 6B item 15, so 6B complete) | 22 | 1,566 | 15,271 |
| **master** (+ #10 benchmarks, finding 29) | 22 | 1,576 | 15,269 |
| **master** (+ #10 trace kernel, finding 30) | 22 | 1,600 | 15,367 |
| **master** (+ #10 GWR, knn, logdet, Chebyshev; finding 31) | 22 | 1,691 | 15,653 |
| **master** (+ Phase 6C, finding 32) | 22 | 2,036 | 15,904 |
| **master** (+ #15 exposure mapping and spatial DiD) | 23 | 2,233 † | 16,949 |
| **master** (+ #15 spatial synthetic control) | 23 | 2,430 † | 17,962 |
| **v0.4.0** (= master today, unreleased) | 23 | 2,430 † | 17,962 |
| — | | | |
| **v1.0.0** (Phases 6C–8) | 27–29 | ~2,400 | ~21,000 |

Measured on master: 23 submodules plus the umbrella across 17,962 lines of
`src/`, 6,773 lines of tests, 3,297 lines of examples across 24 scripts,
398 lines of package extensions, 50 documentation pages and 12 notebooks
(5 Jupyter, 7 Pluto).

**State the environment with the assertion count.** The figures above are
2,430 with Makie and Turing **installed**, where testsets 25 and 26
contribute 11 and 24 assertions, and 2,395 with both **absent**, where each
reports a single broken marker instead of running. Both were measured in the
repository working tree, which matters — see the correction below. That
difference is 35 passing assertions — 2,395 + 11 + 24 = 2,430, the two
broken markers being replaced rather than added, which is why the arithmetic
is 35 and not 33. A count quoted without its environment can therefore look
like a regression when nothing has regressed.

**Corrected on 2026-10-31: there is no cross-version assertion difference.**
Earlier revisions of this section described a genuine six-assertion gap
between Julia 1.13.2 and 1.10.12. Measurement says otherwise. Running the
suite verbose on each version and diffing the per-testset counts — the
diagnostic this section has always prescribed — every one of the 51
testsets reports an **identical** count on 1.13.2 and 1.10.12; the diff is
timings only. The extension testsets are the sole source of count
variation.

Where the old six-assertion figure came from is now settled, and it was
never the Julia version. `Pkg.test()` in the repository working tree
reports 2,395; the same command on a copy holding only `Project.toml`,
`Manifest.toml`, `src/`, `test/` and `ext/` reports 2,389, on either Julia
version. Two candidate explanations were tested and **both refuted** — not
the version (per-testset counts are identical across 1.13.2 and 1.10.12),
and not dependency resolution (copying the local `Manifest.toml` across
does not reproduce the 2,395). The cause is the **working tree**, and the
six assertions are named: testset 11b gates a block on

```julia
csv = joinpath(@__DIR__, "..", "notebooks", "data", "columbus.csv")
if isfile(csv)
```

and that block holds exactly six assertions — one `@test_logs` guarding
`fit_mgwr` against convergence warnings, and five `@test`s on the fit. So
testset 11b reports **12 in the repository and 6 in a `src`-and-`test`-only
copy**. `notebooks/data/columbus.csv` is tracked, so CI always has it: the
extensions-absent figure CI sees is **2,395**, not 2,389.

Two consequences for anyone measuring this suite. A scratch copy must
include `notebooks/data/` to be comparable with the repository at all. And
a verbose variant has to be run from `test/` **inside** the repository: run
the same script from a scratchpad and `@__DIR__` moves with it, the
`isfile` check fails, and the block skips silently — which is exactly how
the earlier 2,424 figure in this section came to be six too low.

Note also that `--project=<dir>` does not hide the extensions: `@v1.13`
stays on `LOAD_PATH`, so Makie and Turing load from the default environment
anyway. `Pkg.test()` is the only entry point that genuinely excludes them.

† The last three rows count the extension testsets; the Phase 6C row above
them does not. So the step from 2,036 to 2,233 is **not** testset 34's
size: that testset is 162 assertions, and the other 35 is testsets 25 and 26
running rather than reporting a marker. The step from 2,233 to 2,430 **is**
testset 35's size exactly, 197. That it comes out exact is itself the check:
an earlier revision of this table recorded 2,430 as 2,424 and so read the
step as 191, and the missing six were the Columbus block described above,
skipped because the measuring script ran from outside `test/`. Comparing two
rows of this table means checking they were measured the same way first.

The jump from 527 assertions to 1,156 is the audit, not new features: every
finding in `docs/AUDIT.md` closed with a test pinning the result to a closed
form, a published number or an independently written reference. The 114 on
top of that are Phase 6B's first two items, which came in under the same
rule.

The earlier per-release projections (v0.4.0 at 16 modules, v0.5.0 at 20,
v0.6.0 at 23) were overtaken: Phases 4, 5 and 6 all landed on the 0.4.0
development line rather than across three releases.

---

## References

- Adao, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: Theory and inference. *Quarterly Journal of Economics*, 134(4), 1949–2010.
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Anselin, L. (1990). Spatial dependence and spatial structural instability in applied regression analysis. *Journal of Regional Science*, 30(2), 185–207.
- Anselin, L. (2001). Rao's score test in spatial econometrics. *Journal of Statistical Planning and Inference*, 97(1), 113–139.
- Anselin, L., Le Gallo, J., & Jayet, H. (2008). Spatial panel econometrics. In *The Econometrics of Panel Data* (pp. 625–660). Springer.
- Assunção, R. M., Neves, M. C., Câmara, G., & Da Costa Freitas, C. (2006). Efficient regionalization techniques for socio-economic geographical units using minimum spanning trees. *International Journal of Geographical Information Science*, 20(7), 797–811.
- Aune, E., Simpson, D. P., & Eidsvik, J. (2014). Parameter estimation in high dimensional Gaussian distributions. *Statistics and Computing*, 24(2), 247–263.
- Bai, J. (2009). Panel data models with interactive fixed effects. *Econometrica*, 77(4), 1229–1279.
- Bailey, N., Kapetanios, G., & Pesaran, M. H. (2016). Exponent of cross-sectional dependence: Estimation and inference. *Journal of Applied Econometrics*, 31(6), 929–960.
- Baltagi, B. H., Song, S. H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123–150.
- Barry, R. P., & Pace, R. K. (1999). Monte Carlo estimates of the log determinant of large sparse matrices. *Linear Algebra and its Applications*, 289(1–3), 41–54.
- Bera, A. K., & Yoon, M. J. (1993). Specification testing with locally misspecified alternatives. *Econometric Theory*, 9(4), 649–658.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share research designs. *Review of Economic Studies*, 89(1), 181–213.
- Bramoullé, Y., Djebbari, H., & Fortin, B. (2009). Identification of peer effects through social networks. *Journal of Econometrics*, 150(1), 41–55.
- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
- Burridge, P. (1981). Testing for a common factor in a spatial autoregression model. *Environment and Planning A*, 13(7), 795–800.
- Chagas, A. L. S., Azzoni, C. R., & Almeida, A. N. (2016). A spatial difference-in-differences analysis of the impact of sugarcane production on respiratory diseases. *Regional Science and Urban Economics*, 59, 24–36.
- Chun, Y., Griffith, D. A., Lee, M., & Sinha, P. (2016). Eigenvector selection with stepwise regression techniques to construct eigenvector spatial filters. *Journal of Geographical Systems*, 18(1), 67–85.
- Delgado, M. S., & Florax, R. J. G. M. (2015). Difference-in-differences techniques for spatial data: Local autocorrelation and spatial interaction. *Economics Letters*, 137, 123–126.
- Di Giacinto, V. (2010). On vector autoregressive modeling in space and time. *Journal of Geographical Systems*, 12(2), 125–154.
- Elhorst, J. P. (2010). Applied spatial econometrics: Raising the bar. *Spatial Economic Analysis*, 5(1), 9–28.
- Ertur, C., & Musolesi, A. (2017). Weak and strong cross-sectional dependence: A panel data analysis of international technology diffusion. *Journal of Applied Econometrics*, 32(3), 477–503.
- Fülle, M. J., & Otto, P. (2023). Spatial GARCH models for unknown spatial locations — an application to financial stock returns. *Spatial Economic Analysis*, 19(1), 92–105.
- Geniaux, G., & Martinetti, D. (2017). A new method for dealing with spatially varying coefficients. *Geographical Analysis*, 50(1), 28–50.
- Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
- Halleck Vega, S., & Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55(3), 339–363.
- Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models. *Journal of the Royal Statistical Society B*, 55(4), 757–796.
- Kelejian, H. H., & Piras, G. (2014). Estimation of spatial models with endogenous weighting matrices, and an application to a demand model for cigarettes. *Regional Science and Urban Economics*, 46, 140–149.
- Klier, T., & McMillen, D. P. (2008). Clustering of auto supplier plants in the United States. *Journal of Business & Economic Statistics*, 26(4), 460–471.
- Koley, M., & Bera, A. (2023). To use, or not to use the spatial Durbin model? — that is the question. *Spatial Economic Analysis*, 19(1), 30–56.
- Kopczewska, K., & Elhorst, J. P. (2023). New developments in spatial econometric modelling. *Spatial Economic Analysis*, 19(1), 1–7.
- Kostov, P. (2009). A spatial quantile regression hedonic model of agricultural land prices. *Spatial Economic Analysis*, 4(1), 53–72.
- Krzyśko, M., Nijkamp, P., Ratajczak, W., Wołyński, W., & Wenerska, B. (2023). Spatio-temporal principal component analysis. *Spatial Economic Analysis*, 19(1), 8–29.
- Kubara, M., & Kopczewska, K. (2023). Akaike information criterion in choosing the optimal k-nearest neighbours of the spatial weight matrix. *Spatial Economic Analysis*, 19(1), 73–91.
- Lee, L.-F., & Yu, J. (2012). QML estimation of spatial dynamic panel data models with time varying spatial weights matrices. *Spatial Economic Analysis*, 7(1), 31–74.
- Leenders, R. T. A. J. (2002). Modeling social influence through network autocorrelation. *Social Networks*, 24(1), 21–47.
- LeSage, J. P. (2014). Spatial econometric panel data model specification: A Bayesian approach. *Spatial Statistics*, 9, 122–145.
- LeSage, J. P. (2015). Software for Bayesian cross section and panel spatial model comparison. *Journal of Geographical Systems*, 17(4), 297–310.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Liu, X., & Yang, Z. (2020). Variable selection in spatial autoregressive models. *Journal of Econometrics*, 218(1), 85–109.
- Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531–542.
- Murakami, D., & Griffith, D. A. (2015). Random effects specifications in eigenvector spatial filtering. *Journal of Geographical Systems*, 17(4), 311–331.
- Mutl, J., & Pfaffermayr, M. (2011). The Hausman test in a Cliff and Ord panel model. *Econometrics Journal*, 14(1), 48–76.
- Pesaran, M. H. (2006). Estimation and inference in large heterogeneous panels with a multifactor error structure. *Econometrica*, 74(4), 967–1012.
- Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large panels. *Econometric Reviews*, 34(6–10), 1089–1117.
- Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative procedure for space-time modeling. *Technometrics*, 22(1), 35–47.
- Qu, X., & Lee, L. F. (2015). Estimating a spatial autoregressive model with an endogenous spatial weight matrix. *Journal of Econometrics*, 184(2), 209–232.
- Santi, F., Dickson, M. M., Espa, G., Taufer, E., & Mazzitelli, A. (2021). Handling spatial dependence under unknown unit locations. *Spatial Economic Analysis*, 16(2), 194–216.
- Shi, W., & Lee, L. F. (2017). Spatial dynamic panel data models with interactive fixed effects. *Journal of Econometrics*, 197(2), 323–347.
- Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of partially linear spatial autoregressive models. *Journal of Econometrics*, 157(1), 18–33.
- Sutton, J., & Sutton, J. (2023). A road map to capture the spatial dependence underlying regions' economic resilience. *Spatial Economic Analysis*, 19(1), 57–72.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57(2), 307–333.
- Wang, C., Kockelman, K. M., & Wang, X. (2013). The spatial dimensions of land use change dynamics in the Austin, Texas Metropolitan Area. *Transportation Research Record*, 2397(1), 131–142.
- Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W., & Wolf, L. J. (2020). Inference in multiscale geographically weighted regression. *Geographical Analysis*, 52(1), 87–106.

**Citations to verify before use — one of them has now failed.** Cerqua et
al. (2022) was listed here for spatial synthetic control and has been
**withdrawn**: checked against the published record on 2026-10-31, it does
not exist as described, and item #15 now cites Abadie, Diamond &
Hainmueller (2010), Abadie & L'Hour (2021), Abadie (2021) and Di Stefano &
Mellace (2024) instead. That leaves the 2025–26 *Spatial
Economic Analysis* volume 21(2) papers — Anselin, Serenini & Amaral on
specification search; Bijwaard & Schluter on interdependent hazards;
Kotowska & Toroj on spatial difference-in-differences; Moreno Jimenez on
fiscal spillovers; Ling, Bai & Yang on Laplace approximation for
random-effects panels — which are a plausible set of leads for Phases 7A and
7B but are not listed above because they have not been checked. Confirm each
before it becomes a deliverable's reference.
