# SpatialEconometrics.jl — Development Roadmap

> **Current version:** 0.6.0, tagged `v0.6.0` on 2026-11-19 (on `edbca6d`). It adds to
> `v0.5.0` (tagged 2026-11-17): the check of the `ρ` search interval
> against `W`'s spectrum (AUDIT 33); #15's two-channel spatial DiD; the
> heteroskedasticity-robust spatial-error GMM and GS2SLS, with the fixes to
> the default GS2SLS's panel inference (AUDIT 34–36); the Claude Code skill
> (#14); and the notebooks on Google Colab. No tag is registered.
>
> 33 submodules, 4,287 test assertions in 65 top-level testsets with both
> package extensions installed and 4,252 (plus 2 broken markers) without
> them, measured on 2026-11-20 on Julia 1.13.4, and without them on 1.10.13.
> 34 examples, 20 notebooks, 2 package extensions, CI/CD and
> 68 Documenter.jl pages. **Phases 4, 5, 6, 6B, 6C and 7A are complete.** Item #10 (Performance) closed on 2026-10-30 with its scope
> reduced on measured grounds and may be revisited. Phase 6C, the GWR
> compact local design, is delivered. **Phase 7A closed on 2026-11-01:**
> item #15's exposure mapping and spatial difference-in-differences landed
> on 2026-10-30, its spatial synthetic control and shift-share diagnostics
> on 2026-10-31, and item #11 (Networks) on 2026-11-01. **Phase 7B is open
> and mostly done:** items #1 (common factors) and #12 (spatio-temporal)
> closed together on 2026-11-04, and row 22's **SpVAR** closed the same day
> in a second pass, with time-varying `ρ` re-filed under #13 where its
> published method belongs; #13 remains (pass 2 on hold since 2026-11-18; a research item on a projection-corrected closed-form `ρ` was opened the same day and built as `root_rho` on 2026-11-20), and #16 (spatial log-GARCH) was
> built on 2026-11-16. Tier 3 is otherwise untouched,
> and the correctness audit in `docs/AUDIT.md` is closed — findings 1–25,
> with 29–32 added afterwards from the benchmark suite and Phase 6C, and 33–36 since.
>
> Items #15 and #16, and every block marked **Planned additions**, were added
> after that audit closed. They are the extend half of clean → fix → extend.
> #15 is delivered, including Chagas et al.'s two-channel spatial DiD and
> a heteroskedasticity-robust spatial error for it (2026-11-18; AUDIT 34
> and 35). The same estimator for the SARAR GS2SLS followed the same day
> (AUDIT 36), which also found and fixed three faults in the default
> path's panel inference.
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

**Weak vs strong cross-sectional dependence — ✅ Complete (2026-11-04):**
the diagnostics above all assume cross-sectional correlation is *weak*, that
is, governed by `W`. The panel literature (Pesaran, Bai, Elhorst) separates
that from *strong* dependence driven by unobserved global factors, which `W`
cannot absorb and which biases `ρ̂` when ignored.

| Feature | Description | Reference |
|---|---|---|
| Spatial panel CCE | Augment the spatial dynamic panel with cross-sectional averages `ȳ_t`, `X̄_t`, `Wȳ_t` to purge common factors | Pesaran (2006); **Yang (2021)**; Ertur & Musolesi (2017) |
| Cross-sectional dependence tests | Pesaran CD, and the Bailey–Kapetanios–Pesaran exponent `α` that says whether the dependence is weak or strong | Pesaran (2015); Bailey et al. (2016) |

`src/SpatialCommonFactors.jl`, submodule 26. Docs:
`docs/src/models/common_factors.md`, `docs/src/api/common_factors.md`.
Example: `examples/27_common_factors.jl`. Tests: testset 38.

`pesaran_cd`, `alpha_exponent`, `dependence_verdict`, `fit_sp_cce`. Two
measured facts drove the design and are in `docs/AUDIT.md` rather than
inferred from the papers: CD's power against **random mean-zero loadings is
0.348 and does not improve with `N`**, and against **balanced ±1 loadings
the statistic is large but negative** — so `abs_rho_bar` and its
independence benchmark are returned alongside, and `dependence_verdict`
checks them before trusting the sign. `alpha_exponent` **throws rather than
returning a number** when `α ≤ 1/2`, where Bailey, Kapetanios & Pesaran
prove it is not identified.

Yang (2021) was added this pass: the row previously cited only Pesaran
(2006) and Ertur & Musolesi (2017), and the latter is an *application* paper
correctly cited for the weak/strong distinction but not the methodological
source for a spatial CCE estimator. Recorded in `docs/AUDIT.md` as an
attribution gap.

Interactive fixed effects with spatial lags are the estimation counterpart
and sit under [#12](#12-spatio-temporal-models).

**Planned addition — common shocks in short panels (✅ both rows built
2026-11-14; the heading's original status was "⬜ candidate, not
scheduled"):** everything above, and the interactive fixed effects under #12,
needs large `T`. With fixed `T`, estimating `N` loading vectors one unit at a
time is an incidental-parameters problem of the Nickell kind. The real
panels this package now ships (Portuguese municipalities, European regions)
are short, so this is a gap the data exposes.

| Feature | Description | Reference |
|---|---|---|
| Fixed-`T` dynamic spatial panel, individual effects only | The same SDPD form `fit_dynamic_spatial_panel` already fits (`y_{t−1}`, `W₂y_{t−1}`, `W₁y_t`, plus a spatial error `W₃`), with time dummies and **correlated random effects** in Mundlak form, `μ = X̄π + ε`, so time-invariant regressors `Z` are estimable. The estimator is an M-estimator: the conditional quasi-scores given `y₀` are biased when `y₀` is endogenous, so each biased component is corrected by its expectation (Lemma 2.1, the adjusted quasi-score of Yang 2018). The correction depends on the process start time `m` but on nothing else about `y₀`, and the simulations report robustness to a wrong `m`. The VC matrix is a hybrid of analytic terms and sample analogues, because neither error component can be estimated with fixed `T`. Simulations: `T ∈ {3, 6}`, `n ≤ 400`, rook, queen and group `W`, non-normal errors, 2000 replications. The conditional QMLE of the dynamic parameter is biased and does not converge with `n`; the M-estimator is nearly unbiased and its robust SEs track the empirical sds. The spatial-error parameter `λ₃` stays biased at `T = 3`. **This is the short-`T` counterpart of the existing `fit_dynamic_spatial_panel`, whose Yu–de Jong–Lee bias correction is a large-`T` device**, so it is the natural first build here, before Bai & Chen's factor version | Li & Yang (2021), *J. Econometrics* 221(2) — **read in part** (§§1–6; from the Supplementary Appendix, §II and Tables 1–2; the Appendices B–C proofs not). ✅ **Built 2026-11-14** as `fit_cre_dynamic_spatial_panel` (`src/CREDynamicSpatialPanel.jl`, submodule 30). The VC is not this hybrid: it is the corrected plug-in of Li, Meng & Yang (2026), adapted to the CRE model, and it is checked against the exact `Var(S*)`. The Table 1 replication is in `docs/AUDIT.md` |
| Fixed-`T` dynamic spatial panel with common shocks | `y_t = ρWy_t + γy_{t−1} + X_tβ + Λf_t + ε_t` for large `N` and fixed `T`. The loadings are projected on observed controls, `y₀`, and their spatial lags `W y₀ … W^L y₀` ("spatially enriched random loadings", SERL), so no unit-specific parameter is estimated. Conditional Gaussian QML keeps the Jacobian `\|I − ρW\|^T`. Block-coordinate algorithm: GLS for `(γ, β)`, ECME updates for the factor covariance, scalar search over `ρ`. **What §3 proves, read 2026-11-14:** consistency and asymptotic normality by standard QMLE reasoning (White 1982) plus a martingale CLT on score contributions rearranged as in Li & Yang (2021), under Assumptions 1–4. Gaussianity is not needed. The errors may be heteroskedastic over time (`D_ε = diag(σ²_t)`) but must be identically distributed across units, the same cross-sectional restriction as Li & Yang. `F` is normalised to `[I_r; F₂]`, so its top `r × r` block must be full rank, and `r` is fixed and known: the theory gives no rule for choosing it (Appendix G compares two and three factors on the application). Identification is assumed, not derived: Assumption 3.1 asserts a uniquely maximised limiting criterion. **The initial condition is handled entirely by Assumption 2**: given `C_N = σ(y₀, X, W)`, the loading residuals `η_i` have mean zero, a common covariance, and are independent across units. This is exact only when `y₀` is generated so that `E(λ_i | C_N) = Az_i` for the chosen `L`. If `y₀` comes from a long-running spatial process, `y₀` carries neighbours' loadings through `B(ρ)⁻¹`, and a finite `L` is, in the paper's own words (§2.5), an approximation. Their Table 4 measures the cost: at `L = 0`, coverage for `ρ` **falls** as `N` doubles (0.642 to 0.312 at `T = 5`, `ρ = 0.5`; 0.884 to 0.820 at `T = 10`), the signature of an inconsistent estimator. At `L = 1` it does not fall (0.918 to 0.935, 0.964 to 0.960), but only `N ∈ {500, 1000}` is measured, so whether `L = 1` is enough is an open question that a build must answer by Monte Carlo at larger `N`, not assume. **Variance:** the sandwich `Ĥ⁻¹Ω̂Ĥ⁻¹/N`, where `Ω̂` is the outer product of the martingale contributions with known-zero cross-terms removed (Proposition 1). It is robust to non-normal `η_i` and `ε_i`. The preferred version multiplies the pair component by `N/(N − p_α)`, a finite-sample device the authors do not claim is unbiased. `θ = (ρ, γ, β)` has a profiled covariance (§3.4). **Appendix D, read 2026-11-14:** with `C(ρ, φ) = I_N⊗R_T(φ) − ρW⊗I_T` and `G = C⁻¹`, each unit's residual is a linear map of all units' errors, so `g_ρ,i` and `g_φ,i` hold a pair sum `Σ_{j<i} e_i'[MW_ij + W_ji'M]e_j`, where `W_ij = Σ_k w_ik G_kj` is a `T × T` block. The `β` and nuisance scores (`δ`, `A`, `F₂`, `vech Σ_η`, `σ²_t`) are unitwise. The covariance estimator (eq. 54) is `Ω̂_M = N⁻¹Σ g†_i g†_i' + N⁻¹Σ_i Σ_{j<i} v_ij v_ij'`: the pair sums are replaced by the sum of their squares, so cross-products with zero conditional mean are dropped. Each pair term is symmetric in `i` and `j`, so `Ω̂_M` does not depend on the order of the units, although each `g_i` does. **Cost:** `G` is dense, so the pair component touches all `N(N−1)/2` pairs, each a `T × T` bilinear form: about 8·10⁶ pairs at `N = 4000`. It must be streamed, never stored. Proposition 1 proves `Ω̂_M(α̂) → Ω(α₀)` under Assumptions 1–4, Assumption 4 being the convergence of the predictable variation, which is itself assumed. **Appendix H** gives an alternative that models `y₀` jointly through an auxiliary initial equation with its own `ρ_I` and projects the loadings on regressors only; its consistency is asserted to carry over, not separately proved | Bai & Chen (2026), arXiv 2608.22499 — ⚠️ **preprint, read in part** (§§1–3, §4 to Step 4, §6.2, §7 opening, Tables 3–4, 9–10, Appendix H; Appendix D through Lemma 1 and the first part of Proposition 1's proof; the remaining proofs not). ✅ **Built 2026-11-14** as `fit_common_shock_dynamic_panel` (`src/CommonShockDynamicPanel.jl`, submodule 31; testset 43; example 32): main-text SERL QML and the Appendix D sandwich, the pair part from `N × N` matrix products rather than streamed pairs. Appendix H, §5 and Remark 3 are not built. **The open question is answered** (`docs/AUDIT.md`): in the long-running design, coverage of `ρ` at `L = 1` holds to `N = 4000` (.926–.964), while at `L = 0` it falls to .398. The Monte Carlo found what the paper's sizes hide: the default `Q_X` (unit means) biases `β̂` by about .009 when the regressors' deviations from the loadings are serially correlated, so coverage of `β` falls to .85–.89 at `N = 4000` at any `L`. Passing every period of `X` as `Q` removes the bias |

Two of its measured results need to be checked before any build. At `T = 5`,
first-order coverage for `ρ` is below nominal, and adding one spatial lag
to the loading projection (`L = 1`) raised average coverage of `ρ` from
0.786 to 0.944 in their long-run-initial-condition design. And in their
sparse design a 30-term Hutchinson approximation of `log|I − ρW|` lowered
coverage for `ρ`, so the paper uses exact sparse LU for every result it
relies on. That concerns Hutchinson traces, not this package's Chebyshev
route with exact traces, but it is evidence of the same kind as the
Chapter 6 measurements. Bai & Chen's §3 was read on 2026-11-14, and the row
above now says what it proves. It does not settle the first of those two
results: Theorem 1 holds under Assumption 2, which a long-running process
satisfies only approximately at finite `L`, and the paper measures `L = 1` at
two values of `N` only. So the row can be scheduled, on two conditions. Its
first Monte Carlo is the long-run design at `N` well beyond 1000, to see
whether coverage at `L = 1` holds or falls with `N`. Appendix D, the second
condition, was read the same day; it adds a cost, not a gap: the variance's
pair component visits every pair of units. Li &
Yang (2021) has now been read: Bai & Chen's two descriptions of it check
out. Its scalar, time-invariant CRE is a special case of their loading
projection, and its score is re-centred using the process start time.
Li & Yang's own row above is now built (2026-11-14); its proofs are still unread.
Bai & Chen's row was built the same day, with its first Monte Carlo the
long-run design to `N = 4000`, as the condition above required.

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
| Threshold SAR | `Y_i = (ρ + ϱ) Σ_j w_ij Y_j + X_i'(β + Δ) + e_i` if `q_i ≤ γ`, else `ρ Σ_j w_ij Y_j + X_i'β + e_i`, i.e. `Y = ρWY + ϱW_γY + Xβ + X_γΔ + e` with `W_γ` the rows of `W` for which `q_i ≤ γ` and `D = I − ρW − ϱW_γ`. Unlike `fit_spatial_regime`, `ρ` also switches across regimes and the regime boundary `γ` is estimated, not given. QMLE profiled over `γ ∈ {q_1,…,q_N}`; sup-Wald, sup-LM and sup-LR tests of no threshold with a bootstrap, because `γ` is unidentified under the null (the Davies problem). The source estimates a shrinking-threshold-effect `γ̂` at rate `O_p(N^{2ν−1})`, `0 < ν < ½`. Needs a null-distribution Monte Carlo for each test before it ships. **Not scheduled** | Li (2022, MPRA 113568 working paper): **read**, model, estimator, tests and simulations; the published Li & Lin (2024) is not held |

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
| ~~Durbin score tests~~ — **delivered**, see `sdm_tests` | Five Rao score statistics for the spatial Durbin model: joint `ρ = 0, θ = 0`; `ρ = 0` with and without `θ ≠ 0`; `θ = 0` with and without `ρ ≠ 0`; plus significance of each individual Durbin term | Koley & Bera (2024); Bera & Yoon (1993) |
| ~~SDEM specification roadmap~~ — **delivered**, see `sdem_roadmap` | Iterative search over {SDE, SLX, SE} × `W`, for the local-spillover case where the Durbin terms sit in the error | Sutton & Sutton (2024) |
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
| ~~Spatial difference-in-differences~~ — **delivered**, see `fit_spatial_did` | Direct ATT and spillover ATT through an exposure mapping on `D` and `WD`, two-period and staggered. Four effects rather than three: the spillover onto the treated and onto the controls are evaluated at different exposure means and are not interchangeable | Delgado & Florax (2015), eq. (6) — **read**, and the estimator actually fitted |
| ~~Unrestricted two-channel SDID~~ — ✅ **built 2026-11-18**, see `fit_spatial_did(...; channels = :split)` | Chagas et al. eq. (7): free coefficients on `D∘WD` (their `W_{T,T}D`) and `(1−D)∘WD` (their `W_{NT,T}D`), in the two-period and staggered designs, with a Wald test of the pooled restriction (`spillover_equality`). The staggered split uses the treatment in force each period, and a design where every unit is treated at some time is now accepted. **Its Monte Carlo found AUDIT finding 34:** with spatially correlated errors the spillover and total-effect intervals under-cover for every `vcov`, in the pooled model too (`att_total_treated` 0.69 in a contiguous block), so `errors = :spatial` was added: the same design with a spatial error by ML (their SDEM), which covers at 0.94–0.95 when the error variance is constant. `validation/mc_spatial_did_channels.jl` and its diagnostics; testsets 34k and 34l | Chagas et al. (2016), eq. (7) — **read** |
| ~~Heteroskedasticity-robust spatial error for the DiD~~ — ✅ **built 2026-11-18**, see `fit_spatial_did(...; errors = :spatial_gmm)` and `fit_spatial_error(...; estimator = :gmm)` | `errors = :spatial` takes its covariance from the information matrix, and with the error sd doubled on the treated in a contiguous design it covered `att_total_treated` at 0.76 (AUDIT 34). Kelejian & Prucha's GM estimator for exogenous regressors: `λ` from the zero-diagonal moments, feasible GLS, a CR3 sandwich. The two-period design is their cross-section; the staggered panel is the package's extension, measured but without a published proof. Also offered for the cross-section SEM. AUDIT finding 35; `validation/mc_spatial_did_gmm.jl`; testsets 10a and 34m | Kelejian & Prucha (2010), *J. Econometrics* 157(1) — **read** (PMC copy); Arraiz, Drukker, Kelejian & Prucha (2010), *J. Regional Science* 50(2) — **read** apart from the appendix |
| ~~Heteroskedasticity-robust GS2SLS (SARAR)~~ — ✅ **built 2026-11-18**, see `fit_gs2sls_panel(...; errors = :heteroskedastic)` | The default estimated `λ` from the 1998 moments, which need equal variances: with the error sd proportional to the neighbour count its `λ̂` averaged 0.24 for 0.4 and covered at 0.61–0.72. Kelejian & Prucha's 2010 estimator with endogenous regressors: GS2SLS, then `λ` from the zero-diagonal moments with the first-stage term `aᵣ` in `Ψ̂`, the joint covariance of `(θ̂, λ̂)` by unit, Hansen's `J`. It covers at 0.95. The panel version is the package's extension. Linearly dependent instrument columns are now dropped, which lets an intercept be fitted under `NoFE()`. AUDIT finding 36; `validation/mc_gs2sls_het.jl`; testset 4d | Kelejian & Prucha (2010) — **read** (PMC copy) |
| ~~Default GS2SLS panel inference~~ — ✅ **found and fixed 2026-11-18** | With constant variance, the default path's own assumptions, its intervals covered at 0.89–0.92 in six-period panels and its Sargan test rejected 13% of true nulls. There were three causes: HC0 on serially correlated within-transformed residuals, a `σ̂²` that estimates `σ²(T−1)/T`, and a `λ` variance that treated the demeaned periods as independent. After the fix it covers at 0.93–0.95 and Sargan rejects 5–6%. AUDIT finding 36 | — |
| ~~Spatial synthetic control~~ — **delivered**, see `fit_synthetic_control` | Donor-weight optimisation on the unit simplex, penalised by geographic or economic distance, with the donor pool screened so the synthetic unit is not assembled from the treated unit's own neighbours. Permutation inference, and the neighbours' own gaps read back as an estimated spillover | Abadie, Diamond & Hainmueller (2010); Abadie & L'Hour (2021) |
| ~~Spatial shift-share diagnostics~~ — **delivered**, see `fit_shift_share` | Exposure concentration against the Herfindahl condition, the shock-level equivalence transform, a shock-level balance test for exogeneity, Rotemberg weights, Moran's I on the exposure shares, and the Adão-Kolesár-Morales standard error — which is the substantive part, since conventional standard errors reject a true null in up to 71% of samples here | Borusyak, Hull & Jaravel (2022); Adão, Kolesár & Morales (2019); Goldsmith-Pinkham, Sorkin & Swift (2020) |
| ~~Exposure mapping utilities~~ — **delivered**, see `exposure` / `ring_exposure` / `check_interference` | `D`, `W^k D`, ring and distance-band exposures. The assumption check is the substantive part: `δ` is identified only by controls that differ in exposure, which is a property of the design and `W` alone and so is checkable before any fit | — |

**Status:** ✅ **Complete** — exposure mapping and spatial DiD on
2026-10-30, spatial synthetic control and shift-share diagnostics on
2026-10-31. `src/SpatialCausal.jl` (1,926 lines) plus the new
`src/SpatialShiftShare.jl`, which is submodule 24: shift-share dependence
runs through shared industry exposure rather than through a weight matrix,
so it is a separate family and not an addition to the spatial-causal file.
Docs: `docs/src/models/causal.md`. Examples:
`examples/23_spatial_causal.jl`, `examples/24_synthetic_control.jl`,
`examples/25_shift_share.jl`. Tests: testsets 34, 35 and 36. Ledger:
`docs/AUDIT.md`, under the standing rule.

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

shift_share_design(W, g; controls, shock_names)  -> ShiftShareDesign
fit_shift_share(y, design; treatment, vcov)      -> ShiftShareFit
exposure_concentration(design)                   -> ExposureConcentration
shock_aggregate(design, y; treatment)            -> ShockAggregates
balance_test(design, r; controls, names)         -> BalanceTest
rotemberg_weights(y, treatment, design)          -> RotembergWeights
shift_share_moran(design, W_spatial; fit)        -> ShiftShareMoran
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

**The shift-share row, and what measurement changed.** Three results are
worth not relitigating.

*Clustering on the dominant industry does not fix shift-share inference.*
This was worth measuring precisely because `vcov_cluster` is already in the
package and is the obvious thing to reach for. Over 2,000 replications it
tracks HC1 almost exactly — 0.720 against 0.732 coverage of a nominal 95%
interval — and is *worse* than HC1 on a concentrated design (0.241 against
0.289). The dependence runs through the whole share vector, so collapsing
each region onto one industry discards most of it.

*The conventional failure is larger than the paper advertises.* Adão,
Kolesár & Morales report over-rejection in up to 55% of placebo samples;
here HC1 rejects a true null in up to **71%**, with its median standard
error 0.211 of the true sampling standard deviation. The design was built
to be realistic, not adversarial.

*The AKM standard error degrades when shocks are few, and that is
documented rather than shipped quietly.* Coverage falls from 0.936 at 100
effective shocks to 0.864 at 20 and 0.827 at 38 with concentrated exposure.
The authors anticipate it — eq. 29 generalises a cluster-robust formula and
inherits its downward bias — and propose a null-imposed interval (their
Remark 6) which is **not implemented here**. The shortfall is named in the
docstring and warned about by `exposure_concentration`. The same
effective-shock count governs the balance test, which rejects a true null
23% of the time at a nominal 5% when it falls to 9. In a concentrated
shift-share design a passed balance test is close to uninformative, and the
docs say so.

A fourth, smaller: the effective-shock count is a useful summary but not a
sufficient one. Coverage was worse at 37.9 effective shocks with
concentrated exposure than at 50.0 with dispersed exposure, so the scalar
does not order designs by reliability on its own.

**A citation was added, and verified first.** Rotemberg weights required
Goldsmith-Pinkham, Sorkin & Swift (2020), which was not in the verified
list below. It was checked against the published record *and* against the
article's own header line (*American Economic Review* 2020, 110(8),
2586–2624, doi:10.1257/aer.20181047) before use. It holds. Every formula in
this row was read out of the primary PDFs rather than recalled — which
caught one that recall would have got wrong: the shock-level projection in
AKM's Remark 5 is the coefficient vector from regressing the *residualised*
regressor on the share matrix, not the shock vector `g`.

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

**Not delivered, deliberately:** the null-imposed AKM interval (their
Remark 6, "AKM0"), the panel variant of the AKM standard error (their
§5.2), and an overidentification test across shocks. The first is the
documented remedy for the few-shock under-coverage measured above and is
the one that matters; it is named in the docstring at the point where the
shortfall appears, rather than left for a reader to find out.

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
| ~~knn `W` selection by AIC~~ — **delivered**, see `select_knn_w` | Refit the same model for `k = 2…50` and read AIC against `k`; the curve is non-monotone with an interior minimum, and a wrong `k` biases `ρ̂` badly — measured at −35.1% at `k = 2` and +29.1% at `k = 20` against a true `k = 6`, wider than the source's "up to 20%" because the range searched is wider | Kubara & Kopczewska (2024) |
| Endogenous `W`, control function | `w_ij = h_ij(Z, ρ_ij)` with a known bounded `h`, where the variables that form `W` follow `Z = X₂Γ + ε` and are correlated with the outcome error through `E(v\|ε) = ε'δ`. Fit `Y = λWY + X₁β + (Z − X₂Γ̂)δ + ξ` by generalised 2SIV, by QMLE (its quasi-likelihood carries `ln\|I − λW\|`, which the package's log-determinant routines already supply) or by GMM. **Not scheduled.** The earlier wording of this row, `w_ij = f(d_ij, Z_ij; γ)` estimated by concentrated ML, did not match either source once read | Qu & Lee (2015): **read**, §§2–3; asymptotics and Monte Carlo unread |
| Endogenous `W`, observed-variable weights | Panel `y(t) = μ + X(t)β + λW(t)y(t) + Y(t)γ + u(t)` with `E[w*_ij(t)] = f[p_ij(t), q_ij(t)]`, approximated as linear in the observed `p_ij`. Instruments are built from `X`, the exogenous weight variables and their fitted values; 2SLS with a HAC covariance; `N → ∞` with `T` fixed; order condition `k + 2g ≥ 1 + q`. Reuses the GS2SLS machinery. **Not scheduled** | Kelejian & Piras (2014): **read** in the accepted manuscript, which has no journal pagination; application unread |
| Convex combination of weight matrices | `W_c(Γ) = Σ_ℓ γ_ℓ W_ℓ` with `γ_ℓ ∈ [0,1]`, `Σγ_ℓ = 1`, `ρ ∈ (−1, 1)`. Write `y = Xβ + Ωω` with `[y, W₁y, …, W_Ly]` precomputed, and take `ln\|I − ρW_c\| ≈ −Σ_{i=1}^{4} ρ^i tr(W_c^i)/i` with the traces precomputed as polynomials in Γ (`tr(W_iW_j)` by Hadamard sums). Flat priors, `β` and `σ²` integrated out; Metropolis–Hastings-guided Monte Carlo integration gives log-marginal likelihoods, then Bayesian model averaging over the `2^L − L − 1` subsets of two or more matrices. Extends `bayes_spec_probs` and can reuse `precompute_neumann_traces`. **Not scheduled.** The order-4 truncation needs its own accuracy test against the exact log-determinant | Debarsy & LeSage (2020): **read**, §2 and §2.6 of the accepted author version; effects (§2.5), §2.6.3 and the illustration unread |
| Bayesian estimation of a binary `W` | `W` is the row-standardised `Ω` with unknown `ω_ij ∈ {0,1}`. Prior: independent Bernoulli with fixed or spatially informed inclusion probability, or a hierarchical beta-binomial prior on the neighbour count for sparsity. Gibbs sampler with a griddy-Gibbs step for `ρ`. Without a symmetry restriction an `ω_ij` update changes only row `i` of `I − ρW`, so `\|A\|` and `A⁻¹` update by the matrix determinant lemma and Sherman–Morrison, and only the branch that differs from the current matrix needs evaluating. **Not scheduled, and the evidence is mixed:** in the source's own simulation `ρ` is recovered poorly at weak dependence (RMSE of `ρ` 0.28 at `N = 100`, `T = 40`, `ρ = 0.3`, against 0.027 and 0.033 in the two exogenous-`W` columns), and the source motivates the method by `T` small relative to `N` | Krisztin & Piribauer (2021, arXiv v2 working paper): **read**, framework, priors, sampler and simulation design; application unread. The journal version is not held |
| Sparse adjustment to candidate `W`s | `W = Σ_i ξ_i W₀ᵢ + A`, where `A` is sparse with a zero diagonal; with no candidates (`M = 0`), `A` is itself a sparse `W` estimator. Panel SAR with fixed effects. The fit is IV-filtered least squares, not ML, so it needs no log-determinant: instruments are `U_t` and its lags `W₀ᵢU_t`, averaged with equal weights or by 2SLS. Stage 1 is a LASSO on `vec(A)` with `β` profiled out and `ξ` unpenalised. Stage 2 runs adaptive LASSO separately on `A` and on `ξ`. The solver is block-coordinate descent (the joint problem is non-convex, each block is convex), with LARS row by row for `A` and BIC for both tuning parameters. Errors may be non-Gaussian, serially dependent and cross-correlated. `N` and `T` must both diverge (`N` may grow faster), and there are `N²` unknowns, so it does not apply to a cross-section. **Not scheduled**: the package has no LASSO solver. In the source's simulation (`N ≤ 75`, `T = 50…200`), the zeros of `A` are found well (specificity mostly above 0.93), but selection of the `ξ` is weaker (in the extracted table, specificity is 0.71–0.88 with partial knowledge). It is the frequentist counterpart of the two Bayesian rows above, and unlike Debarsy & LeSage it allows a deviation from the combination | Lam & Souza (2020): **read in part** in the accepted manuscript — §§1–2, §4, §5 and the M assumptions; theorems and proofs unread |
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
| Matrix-free log-determinant — ❌ **closed, measured** | Stochastic Lanczos quadrature for `log det(I − ρW)`. Implemented and benchmarked 2026-11-05, and it **does not help**: the Gauss quadrature saturates at `m ≈ 10`, so what remains is the same Hutchinson variance the Chebyshev path already faces, at 24× the wall time per matrix-vector product; and it needs a symmetric operator, so by Ord (1975, p. 125) it cannot be used on row-standardised `knn` weights at all. Hutch++ was answered without implementing it — `log(I − ρW)` has no decaying spectrum to sketch. The fill-in case it was parked for is served by the existing Hutchinson trace path instead. See `docs/AUDIT.md` |

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
| ~~Network weight construction~~ | ~~`network_weights(adjacency; normalize=:row)` — from adjacency matrix, edge list, or graph~~ | — |
| ~~Network autocorrelation model~~ | ~~`y = ρWy + Xβ + ε` where W is derived from network adjacency (not geography)~~ | Leenders (2002) |
| ~~Peer effects model~~ | ~~`y_i = ρ * mean(y_peers) + Xβ + ε` — endogenous vs exogenous peer effects (Manski reflection problem)~~ | Manski (1993); Bramoullé et al. (2009) |
| ~~Network centrality instruments~~ | ~~Construct instruments from higher-order network structure (friends-of-friends) to address endogeneity~~ | Bramoullé et al. (2009) |
| ~~Multiplex spatial models~~ | ~~Multiple W matrices representing different interaction channels (trade + migration + FDI)~~ | Lee & Liu (2010) |

**Status:** ✅ **Complete** (2026-11-01). `src/SpatialNetworks.jl`, submodule 25. Docs: `docs/src/models/networks.md`, `docs/src/api/networks.md`. Example: `examples/26_networks.jl`. Tests: testset 37.

**Key functions:**
- `network_weights(A)` / `network_weights(edges, n)` → `SparseMatrixCSC`
- `group_weights(sizes; include_self)` → the Manski / Moffitt / Lee group structures
- `peer_identification(G; fixed_effects)` → `PeerIdentification`
- `fit_peer_effects(y, X, G; fixed_effects, vcov, instrument_order)` → `PeerEffectsFit`
- `fit_multiplex_sar(y, X, Ws)` → `MultiplexSAR`
- `network_moran(fit, W)` → `MoranResult`

**Two corrections to this item, both found by reading the primary sources
rather than working from the plan.**

*The refinement was imprecise.* It previously read: "automated rank
validation of the `W²X`, `W³X` instrument set, reported alongside the fit so
a user sees when the exclusion restriction has nothing to stand on." That is
a real and useful diagnostic — it is now reported as `first_stage_F` — but
it is **not** Bramoullé, Djebbari and Fortin's condition. Theirs is a rank
condition on the *matrices* `I, G, G²` and does not involve `X` at all;
their footnote 17 gives the check verbatim, as a rank-3 test on
`[vec(I) vec(G) vec(G²)]`. The two answer different questions —
identification in population versus instrument power in this sample — and
both are now reported, separately labelled, because conflating them is how a
degenerate network gets fitted.

*The row's rationale was wrong.* The Phase 7A table said this item "mostly
reframes what `W` may represent rather than adding inference, so it goes
second." The reframing is indeed cheap: the network autocorrelation model is
the existing SAR with a different `W`, and the package's log-determinant
machinery was already safe for the non-symmetric `W` a directed network
produces. But the peer-effects row is an *identification* result, not a
reframing, and it fails on the structures applied work uses most. Scheduling
it second was still right; the stated reason was not.

**Dependencies:** None. An adjacency is a sparse matrix and an edge list is a
vector of pairs; Graphs.jl was considered and not added.

**Estimated complexity:** ~400 lines. **Actual: 1,187**, the overrun being
the multiplex MLE (taken at full maximum likelihood rather than deferred)
and the identification machinery the estimate did not anticipate.

---

### 12. Spatio-Temporal Models

**Motivation:** While the dynamic spatial panel captures temporal dependence through lagged y, it assumes a specific AR(1) structure. More flexible spatio-temporal models allow richer dynamics, time-varying spatial effects, and impulse response analysis.

**Deliverables:**

| Feature | Description | Reference | Status |
|---|---|---|---|
| STAR model | Space-Time Autoregressive: `y_t = Σ_k φ_k W⁽ᵏ⁾ y_{t-l} + Xβ + ε` with higher-order spatial and temporal lags. `W⁽ᵏ⁾` is `Wᵏ` by default, or the order-`k` neighbour matrix — the paper's own definition — under `spatial_weights = :orders` | Pfeifer & Deutsch (1980) — **verified, read in full from the authors' reprint (NCJRS 60692)** | ✅ |
| SpVAR | Spatial Vector Autoregression: multivariate spatial panel VAR with impulse response functions | Di Giacinto (2010) — **verified, read in full from Banca d'Italia Temi di discussione 746** | ✅ |
| Space-time impulse response | Trace the propagation of a shock through space and time; compute cumulative spatial multipliers | LeSage & Pace (2009) | ✅ |
| Time-varying spatial effects | Allow `ρ_t` to change over time. **Moved to [#13](#13-semiparametric-spatial)**, and **still unsourced there.** Liang, Gao & Gong (2022) was cited here for one commit as "the published method"; reading it showed its `ρ` is a **scalar constant** and only the regressor coefficients vary with time. That paper names a time-varying spatial lag as future work it did not attempt, because the lag is endogenous. See `docs/AUDIT.md` | — (the earlier citation was withdrawn) | ➡️ **moved to #13, unsourced** |
| Forecasting | `spatial_forecast(m, X_future, W; h=5)` — h-step ahead spatial panel forecasts | — | ✅ |
| Interactive fixed effects with spatial lags | Joint estimation of a factor structure `λᵢ'f_t` alongside `Wy` and `WX`, separating genuine spatial interaction from global shocks | Bai (2009); Shi & Lee (2017) | ✅ |

The factor-structure row is the estimator that pairs with the cross-sectional
dependence diagnostics under [#1](#1-spatial-panel-data-toolkit): the tests
say whether a factor structure is there, this estimates it.

**Status:** ✅ **Complete** (2026-11-04), in two passes.
`src/SpatioTemporalModels.jl`, submodule 27, and `src/SpVAR.jl`, submodule
28. Docs: `docs/src/models/common_factors.md`,
`docs/src/api/common_factors.md`, `docs/src/models/spvar.md`,
`docs/src/api/spvar.md`. Examples: `examples/28_spatio_temporal.jl`,
`examples/29_spvar.jl`. Tests: testsets 39 and 40.

`fit_star`, `fit_interactive_fe`, `factor_projection`, `space_time_irf`,
`spatial_forecast`. `fit_star` places every spatial lag one or more periods
back, so all regressors are predetermined, least squares is consistent and
there is no Jacobian — the price, stated rather than hidden, is that there is
no contemporaneous `Wyₜ`. That specification is `fit_spatial_panel`, and
**neither nests the other**. `fit_interactive_fe` exposes
`factor_projection(m)` because factors and loadings are identified only up to
rotation; comparing `m.factors` across fits compares starting values.

**SpVAR shipped in a second pass the same day**, as `src/SpVAR.jl`:
`fit_spvar`, `spvar_order_condition`, `space_time_stir`, `spvar_irf`,
`spvar_forecast`, `fit_nested_panel_var`, `lr_spatial_interaction`. Di
Giacinto makes `C₀` **block** triangular in the variable index rather than
strictly triangular, because strict triangularity would impose a recursive
causal ordering on the *locations* — a unilateral causal chain in space,
when spatial interaction is bilateral. Two consequences: the parameter
count is linear in `N`, and `|C₀| = Π_r |A⁽⁰⁾_rr|`, so concentrating `Λ` out
splits the likelihood into `K` unrelated terms. A spatial VAR is `K`
spatial autoregressions, not one `NK×NK` problem, and both identities are
pinned at `0.000e+00`.

`ordering` has **no default**, because reversing it on a 14-region ring
moved the log-likelihood by 9.949 and the peak contemporaneous spillover by
a factor of 5.25 — and moved its date from `h = 0` to `h = 2`. Its
asymptotics are `T → ∞` at **fixed `N`**, the opposite of every other panel
estimator here, which was measured rather than asserted: growing `T`
removes the dynamic bias by a factor of 24, growing `N` eightfold does not
move it at all.

This pass also added `order_weights` / `order_weights_list` to
`SpatialWeights.jl`, since Di Giacinto's `W⁽ˡ⁾` is the order-`l` weights
matrix of Anselin & Smirnov (1996) and **`Wˡ` is not that** — powers count
walks, sit on the diagonal and double-count paths. Nothing in the package
built order-`l` weights and no row had planned one.

**Time-varying `ρ` moved to [#13](#13-semiparametric-spatial)** rather than
staying here. Its roadmap entry was unsourced — reference column `—`, method
given as "state-space or rolling-window" — and there is no published
treatment of `ρ_t` as a latent state for spatial panels; the obstacle is
structural, since `ρ` enters the likelihood nonlinearly through
`log|I − ρₜ W|`. Building it here would have meant inventing a method.

**The replacement citation was also wrong, and lasted one commit.** This
paragraph previously named Liang, Gao & Gong (2022) as "the published
approach" for `ρ_t`. The paper was then obtained and read. Its model is
`Y_it = ρ₀ Σ_j w_ij Y_jt + X_it' β₀(τ) + α₀ᵢ + e_it`: **`ρ₀` is a scalar
constant** and only the regressor coefficients are time-varying. The paper
states the limitation itself — a nonparametric spatial lag term would be
less restrictive, since spatial dependence probably does change over time,
but the lag is endogenous and they leave it to future work. So it is the
sourced method for **time-varying `β`**, which is a worthwhile row in its
own right, and **not** for time-varying `ρ`, which remains unsourced.

The move to #13 still stands, because local-linear concentrated QML does
share a kernel-weighted likelihood with #13's Su & Jin (2010) partially
linear model, and built apart that machinery gets written twice. What
changed is the row's claim, not its address.

**Dependencies:** None

---

### 13. Semiparametric Spatial

**Motivation:** Parametric spatial models assume linear functional forms for the relationship between X and y. When relationships are nonlinear (e.g., pollution effects with thresholds, diminishing returns to agglomeration), semiparametric methods avoid specification bias while retaining spatial structure.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| ✅ Partially linear spatial model | **Delivered 2026-11-13 as `fit_partially_linear_sar`** (see the status note below the table). `y = ρWy + Xβ + f(z) + ε` with `f` unknown. **Su & Jin's route:** for fixed `ρ`, a local-polynomial kernel smoother (order `r ≥ 1`, so local linear for `q ≤ 3`) of `(I − ρW)y − Xβ` on `z`; then the profile Gaussian quasi-likelihood in `ρ`, which **keeps** `log\|I − ρW\|`. The smoother matrix `S` depends on `z` and the bandwidth only, not on `ρ`, so `σ̂²(ρ)` is a quadratic in `ρ` built from `(I − S)`-residualised `y` and `Wy`: the existing `chebyshev_logdet` serves the Jacobian, and Geniaux's closed-form route below can reuse the same residuals. Rates: `√n` for `ρ` when each unit has a bounded number of neighbours (`l_n` bounded); `√(n/l_n)` with undersmoothing (`nh^{2(r+1)} → 0`) when `l_n → ∞`. Homoskedastic errors only (Assumption 2), compact parameter space for `ρ`, and fixed bounded regressors. Identification (their Assumption 7) needs `GXβ`, after `(I − S)` and the projection off `X`, to stay non-degenerate. A smoother in the coordinates would weaken exactly this, which is our reading of how Geniaux's geoadditive trap shows up in their theory. Their simulations use the Case (1991) block `W = I_R ⊗ B_m` with `z` i.i.d. or SAR-generated, never geographic, so they never test the trap. Bandwidth: rule of thumb `h = s_z n^{−1/(q+4)}` as a first pass, then least-squares cross-validation of `y − Xβ̃ − ρ̃Wy` on `z`; they report the two give similar results. The kernel must have compact support (Assumption 5), with `nh^{2q} → ∞` and `nh^{4(r+1)} → 0`. **Variance:** Lee's (2004) QML sandwich `Σ⁻¹(Σ + Ω)Σ⁻¹` with `P = (I − S)'(I − S)` in place of the identity in the `X` and `R = G(Xβ + m)` blocks; `Ω` carries the third and fourth error moments and is zero under normality (Theorem 4.3, Remark 2). Read 2026-11-13 for the build: model, estimator, Assumptions 1–8, Theorems 4.1–4.4 and 5.1–5.4 as stated, Remarks 1–3 | Su & Jin (2010) — **read in part** (from the 2007 submitted manuscript; see References; proofs not read); Geniaux (2026) (closed-form, below) |
| ✅ Closed-form `ρ` for a smooth first stage | **SAR version delivered 2026-11-13 as `closed_form_rho`**, with classical, HC0 and SHAC standard errors; the SEM version is not built. A route that needs no log-determinant for the partially linear and additive rows. Regress `y`, `Wy` and `W²y` on the same fixed spline basis, keep the residuals `e₀, e₁, e₂`, and solve the quadratic quasi-score `aρ² − bρ + c = 0` with `a = e₁'e₂`, `b = e₀'e₂ + e₁'e₁`, `c = e₀'e₁`, taking the root `(b − √(b² − 4ac))/2a` (Smirnov 2022, eqs 11–16). The quasi-score keeps only the first term of the power-series expansion of the ML score and replaces the rest by their expectation, zero; this uses `tr W = 0`. SDM is the same procedure with `[X : WX]`. **Smirnov's SEM version projects on `[X : WX]` (the unrestricted Durbin model) when forming `e₀, e₁`**, then refits `β, σ²` on the filtered model; Geniaux's description does not say whether his SEM first stage does the same, so check before building it. **Smirnov gives no consistency proof**: consistency rests on the quasi-score having expectation zero, plus "mild regularity conditions" that are not stated, plus Monte Carlo evidence of bias and MSE shrinking about tenfold per tenfold `n`. His variance `σ²/(b − 2ρa)` replaces an expectation by its sample value, is not heteroskedasticity-robust, and his tables report bias, MSE and estimated variance but **no coverage rates**. Efficiency against ML depends only on `ρ` in his designs: the MSE ratio is under 1.10 for `ρ ≤ 0.5`, about 2 at 0.9–0.95, and 4–5 at 0.99. Geniaux proves it stays consistent and asymptotically normal for a fixed-basis penalised spline, and that the score is Neyman-orthogonal to the smooths. The closed-form variance `σ²/(b − 2ρa)` is conservative and grows more so as `ρ` rises. The SEM version is unreliable above `ρ ≈ 0.85`, where the paper recommends full ML. **Robust variances** (Geniaux §2.5): with `mᵢ = (e₀ − ρ̂e₁)ᵢ(e₁ − ρ̂e₂)ᵢ` and `d = b − 2ρ̂a`, HC0 is `Σmᵢ²/d²` and SHAC adds `Σ_{i≠j} ω_ij mᵢmⱼ` over pairs within graph distance two in `W`. In his heteroskedastic check the closed form covers at 0.96–0.99, HC0 under-covers at 0.90–0.92, and SHAC is close to nominal. **Scope of the proof:** Geniaux's consistency and normality results cover a *fixed* spline basis (his Lemma 1, Remark 1); a kernel smoother with a shrinking bandwidth is the sieve regime of his Remark 6, where rate conditions return. So the closed-form `ρ` built on Su & Jin's smoother, with `eₖ = M(I − S)Wᵏy`, is **unproven**: its coverage must be measured by Monte Carlo before it is offered. The same residuals give Su & Jin's `σ̂²(ρ) = ‖e₀ − ρe₁‖²/n`, so the two estimators share their first stage. Pass 1 builds the SAR version only; the SEM first-stage question stays open | Geniaux (2026), arXiv 2611.16592 — ⚠️ **preprint, read in part** (§§1–5; proofs in Appendix C not); Smirnov (2022), *Geogr. Anal.* 54(1) — **read** (main text; simulation tables read via the prose) |
| 🔶 Identification check for every smoother row | **The measure `π̄` is delivered (2026-11-13, `spectral_overlap`) and is computed by every smoother fit; the remedy — penalty, plateau rule, cross-fitting — is not, and is on hold by decision (2026-11-18; see "Not built, and why" below).** **A deliverable, not an option.** A smooth of the coordinates `s(u,v)`, or a covariate that is a function of them (a distance to a fixed point), competes with `ρWy` for the same low-frequency variation: the "geoadditive trap". In Geniaux's factorial design, a free `s(u,v)` biases `ρ̂` downward even with i.i.d. covariates, and the 5% test of `ρ = 0` rejects in 43–72% of replications. Spatially autocorrelated covariates alone are almost harmless. REML and random-fold CV cannot select the smoothing, because both allocations fit equally well. Over-restricting the smooths fails the other way (100% rejection of a true `ρ = 0`). The remedy that held size at 5–9% in every cell combines: the spectral diagnostic `π_k = ‖S φ_k‖` on the 25 dominant eigenvectors `φ_k` of the **symmetrised** `W` (Appendix B.6, Figure B.10), summarised by their mean `π̄`; a penalty on the surface's projection onto those eigenvectors, with its weight chosen at the plateau of the `ρ̂` path; spatial cross-fitting of `e₀, e₁, e₂`; and SHAC standard errors (55–65% above the closed-form ones on his house-price data). The eigenvectors already exist in `SpatialFiltering.jl`. **`π̄` is a flag, not a test:** the paper gives no threshold and no null distribution (it reports 0.06 under 4-NN and 0.26 under 10-NN on the Lucas County data), so the package must report it as a descriptive overlap measure and must not attach a p-value. In Su & Jin's model the smoother is the local-polynomial `S` on `z`, so `π_k` applies directly; their Assumption 7 is the identification condition that a large overlap threatens. **What the check may claim** (Gibbons & Overman 2012; Debarsy & Le Gallo 2024 WP): a small `π̄` says only that the smoother does not compete with `ρWy` for `W`'s dominant modes, *given* that `W` is the true interaction structure. That is statistical identification of `ρ`, not causal identification of a spillover: the endogenous and correlated effects stay confounded (the reflection problem), and in a cross-section spatial heterogeneity and spillovers are observationally equivalent, so a large `π̄` cannot say which allocation is true either. The docstrings and the printed output must say this, and must not call `ρ̂` or the impacts causal | Geniaux (2026) — ⚠️ **preprint**; the plateau rule is supported by simulation only, as the paper says |
| ✅ Time-varying coefficients `β(τ)` | **Delivered 2026-11-13 as `fit_tv_coef_sar`.** `y_t = ρWy_t + X_tβ(τ) + α_i + ε_t` with `β(·)` an unknown smooth function of rescaled time, by **local-linear concentrated QML** (LLQML). `ρ` is a constant here. Parametric components converge at `√(NT)`, the nonparametric ones at `√(NTh)`. Shares its kernel-weighted concentrated likelihood with the partially linear row above, so the two should be built together | Liang, Gao & Gong (2022) — **verified, read from MPRA 108497 / Monash WP 05/21** |
| Time-varying spatial dependence `ρ_t` | **Unsourced, and deliberately not scheduled.** Moved here from [#12](#12-spatio-temporal-models) with a citation that did not survive reading — see the note under row 22. `ρ` enters the likelihood through `log\|I − ρₜ W\|` and the lag is endogenous, which is why the literature has not settled it; Liang, Gao & Gong name it as their own future work. Nothing is built until there is a method to follow | — |
| Spatial quantile regression | Effects vary across the conditional distribution: `Q_τ(y \| X, W) = ρ_τ Wy + Xβ_τ` | Kostov (2009); Ando & Lu (2019, SSRN 3516306 working paper): **read in part** — for panels with common factors the conditional quantile is defined jointly, `Q_{y_it}(τ) = ρ_τ Σ_j w_ij Q_{y_jt}(τ) + x_it'b_{i,τ} + f_t'λ_i` with `W` pre-specified, so the reduced form is `(I − ρ_τ W)⁻¹{x_t'b_τ + f_t'λ_τ}`; estimated by alternating quantile-loss minimisation with `F'F/T = I`, selecting the number of factors by an information criterion. Consistency needs `log N / T^{1/2} → 0` and `log T / N^{1/2} → 0`, so it is a large-`N`, large-`T` method. **Not scheduled**; Theorems 2–3 and the proofs are unread |
| Spatial varying coefficient model | Generalization of GWR with penalized spline smoothing — avoids GWR's boundary bias and bandwidth sensitivity | Hastie & Tibshirani (1993); Geniaux & Martinetti (2017) |
| Penalized spatial regression | LASSO / elastic net with spatial lag: variable selection in high-dimensional spatial models | Liu & Yang (2020) |
| Additive spatial model | `y = ρWy + Σ_k f_k(x_k) + ε` — spatial GAM with backfitting | — |

**Status after pass 1 (2026-11-13).** Built: the Su & Jin profile QML with a
local-polynomial smoother (rule-of-thumb or two-step LSCV bandwidth), the
closed-form `ρ` on the same first stage, the spectral overlap `π̄` on every
fit, and LGG's `β(τ)` with a leave-one-unit-out CV bandwidth. Each reduces
exactly to an existing estimator at a large bandwidth (testsets 41b, 41c,
41f). The Monte Carlo studies are in `docs/AUDIT.md`, with their scripts in
`validation/`: coverage and the trap for the cross-section, and the `β(τ)`
band. At the CV bandwidth the band under-covers in the interior (0.81–0.87)
and is close to nominal at the ends. A refit at `h/2` brings the interior to
0.92–0.94; the docs give that recipe and its failure rate at `T = 25`.

Choices made in the build that the sources do not dictate:
- **Su & Jin's `Ω` is omitted.** The standard errors are valid under normal
  errors only; the Monte Carlo measures the cost under skewed errors.
- **The modes for `π̄` are the doubly-centred (Moran) eigenvectors**, not the
  plain symmetrised `W`'s, because every smoother here reproduces constants.
- **A window with too few points is widened, not refused**, and the count is
  reported (`n_widened`): without it 221 of 500 fits failed at `n = 400`
  with `z` normal.
- **LGG's information scales the trace terms by `T`**; the package's panel
  code uses `T − 1` under individual fixed effects. The fit follows the paper,
  and testset 41f checks both conventions exactly.

Not built, and why:
- Geniaux's spectral penalty, plateau rule and spatial cross-fitting:
  #13 pass 2, **on hold by decision (2026-11-18).** The remedy penalises a
  *spline* surface's projection onto `W`'s modes, and a kernel smoother has
  no penalty to add it to. So pass 2 would be a second, penalised-spline
  first stage, not an option on the first one. Geniaux was read in full on
  2026-11-18, and the reading is why it is on hold:
  - **The theory covers a fixed basis only.** Appendix C's consistency and
    normality need `K` fixed as `n` grows (Lemma 1: the smoothers collapse
    onto the projection `P_B`, and Smirnov's linear results carry over).
    The sieve regime is left open (Remark 6), and the package's kernel
    smoother is not covered either.
  - **The penalty is `λ_s‖Γ_Ks′Bθ‖²`** on the surface `s(u,v)` only, with
    `Γ` the `K_s = 100` leading eigenvectors of the symmetrised `W` (§2.5).
    `λ_s` runs over a grid, 10⁻² to 10⁴ in the simulations and to 10⁶ in
    the application. The paper does not define its scale against the
    roughness penalty, a tolerance for the plateau, or the fold count and
    cell size of the cross-fitting.
  - **None of the three safeguards has a theory.** The plateau rule rests
    on simulation (§5, the paper's words); cross-fitting under spatial
    dependence is left to future work (Remark 7). REML-selected `λ_s` is
    shown to do nothing.
  - **What is settled:** the quasi-score is Neyman-orthogonal in the three
    nuisance fits (Proposition 7); the closed-form variance is conservative,
    by a factor of up to about 4 at `ρ = 0.9` (Remark 4); HC0 under-covers
    (0.90–0.92) and the graph-distance-2 SHAC is near nominal (Remark 9).

  Building it would mean choosing the scale, the tolerance and the folds
  ourselves and measuring them by Monte Carlo.
- The SEM closed form: its first stage projects on `Z = [X, WX]`
  (Geniaux A.1, as in Smirnov), and the root kept is the admissible one
  that minimises `‖e₀ − ρe₁‖²` (A.2). Not built.
- `ρ_t`: unsourced.
- Quantile, varying-coefficient, penalised and additive rows: not in pass 1.

**Research: a projection-corrected closed-form `ρ` for a growing basis**
(opened 2026-11-18; **built 2026-11-20 as `root_rho`**, the projection
case only: see "Built" at the end of this entry). The scripts, their outputs
and the method are in `research/trace_cfe/`.
- **The problem.** With a projection first stage `P_B`, the closed form's
  moment has mean `−σ² tr(P_B W)/n` at the true `ρ`. That is O(K/n), and a
  free surface makes K large. In the Monte Carlo it biases `ρ̂` by −0.07 at
  K = 29 and by −0.5 at K = 191, and the plain closed form covers at
  0.01–0.05. In those designs this is the bias Geniaux removes by
  cross-fitting.
- **The correction is not new** (read 2026-11-18). Jin & Lee (2012) give
  the closed-form root of `y′S(ρ)′PMS(ρ)y = 0`:
  - with `tr(PM) = 0` (§3.1). Their `P = W′ − tr(W′M)/(n − d)·I` is our
    pooled `τ` exactly;
  - with `PM` zero-diagonal under heteroskedasticity (§3.2). Their initial
    estimator uses `P = W′ − Diag(W′M)Diag(M)⁻¹`, and REsd, the two-step,
    uses `G′(ρ̂)` in place of `W′`.

  Liu & Yang (2015) apply the per-unit version to the QML score. Both
  assume a regressor matrix of fixed dimension, with `PM` bounded in row and
  column sums.
- **Our exact-diagonal matrix `MW + MDM` is not better** (measured
  2026-11-19, `tc_jinlee.jl`, 20 cells of 500 replications):
  - against Jin & Lee's first step its variance ratio is 0.99–1.01 at
    K = 100, and it is worse at K = 199 (1.11–1.19);
  - their two-step REsd has the smallest variance of the three: 0.87–0.99 of
    their first step's;
  - all three are unbiased and cover at 0.936–0.952.

  The first version of the comparison reported a 25–40% variance gain for
  ours. That gain was an error: Jin & Lee's matrices were implemented with
  `W` in place of `W′`, because `pdftotext` dropped the primes, which are
  visible on the page images. With an asymmetric `W` the unprimed version
  uses the instrument `M(W′ − Δ)f` instead of `MWf` (`tc_misalign.jl`,
  part D). Read formulas from the page image, not from extracted text. Also
  measured (`tc_commute.jl`, `tc_misalign.jl`): the two corrections are
  equal when `MW = WM`, and the delta-method variance matches the simulated
  sd in every design checked.
- **What may still be new:**
  - the regime where the basis grows with `n`: there `M` is dense and the
    trace term is first-order, and the published results assume fixed `d`
    or rate conditions such as `k²/n → 0`;
  - the noise term in the robust variance for the estimated `P_B y₀`.
    Without it the variance ran 3–10% high, and it is negligible when the
    dimension is fixed;
  - the link to Geniaux: in our designs his own-observation bias is the
    Jin–Lee trace term.
- **The theory to work out,** for Jin & Lee's REsd with a growing basis, in
  order:
  1. A CLT for the linear-quadratic moment when `M` is dense. The Kelejian &
     Prucha (2010) theorem needs bounded row sums, which a growing `P_B`
     breaks. This step sets the rate at which K may grow with `n`.
     **Sketch written 2026-11-19** (`research/trace_cfe/THEORY.md`; not
     checked by anyone else). The martingale route needs only norm ratios:
     `‖C‖_op` stays bounded because `‖M‖_op = 1`, so bounded row sums are
     needed for `W` alone. In the sketch, `K/n → c < 1` is allowed, and the
     binding rate is the undersmoothing lower bound `√n‖r‖_∞² → 0`. In
     `tc_clt.jl` normality improves with `n` in every kNN design, including
     K/n ≈ 0.15 and a Chebyshev basis with leverage 0.75. The power check
     (two complete groups) stays far from normal.
  2. The consistency of the robust variance, including the noise term.
     **Theorem 1 proved and step 2 analysed, 2026-11-19**
     (`research/trace_cfe/THEORY.md`; not checked by anyone else). The CLT
     for the moment now has a full proof: martingale differences and Hall &
     Heyde's Corollary 3.1. It needs only `‖Cs‖_op/‖Cs‖_F → 0`: the
     triangular part is bounded in the Schatten-4 norm with no log factor
     (Matsaev 1961, *Dokl.* 139(4), Theorem 1, read 2026-11-20; Bhansali,
     Giraitis & Kokoszka 2007, Lemma 2.1, cite his other 1961 note),
     checked in `tc_schatten.jl` against a matrix whose triangular part grows
     like log n in the operator norm. It needs only `2 + δ` moments: the
     zero diagonal leaves the `εᵢ²` only in the conditional variance, where
     von Bahr & Esseen's inequality gives the law of large numbers, and the
     Lyapunov step is of order `2 + δ`, by Rosenthal's inequality. Against
     Kelejian & Prucha (2001), it drops bounded row sums and the rate `n`,
     with the same moments. In `tc_clt_moments.jl` (`t₃` and Pareto errors,
     no fourth moment) normality improves with n but slowly: KS 0.038 and
     0.067 at n = 4900, against a critical value of 0.030. In
     `tc_coverage_tails.jl` this makes the 95% intervals for `ρ̂`
     conservative, not anti-conservative: robust coverage 0.953–0.977 under
     those laws, n = 400 to 2500, against 0.948–0.952 with normal errors.
     The Corollary (the limit of `ρ̂`) is proved too, assuming
     identification (`liminf J/n > 0`) and a bounded `(I − ρ₀W)⁻¹`, for the
     root at which `g′ < 0`. The code's rule (the root nearest the OLS ratio)
     is not covered by the proof, but in `tc_roots.jl` it picked the same
     root in 7999 of 8000 replications. Jin & Lee's two-step REsd has the
     same limit as the infeasible estimator with `G(ρ₀)` (Proposition 1):
     `C(r)f = 0` and `diag C(r) = 0` for every `r` make the plug-in's
     first-order effect mean-zero. In `tc_resd.jl` its gap to the infeasible
     estimator falls from 0.116 to 0.027 of the latter's sd as n goes from
     400 to 2500, while a fixed wrong plug-in stays at 0.20–0.24. For step 2,
     the noise term removes the `O(K/n)` bias of the linear part, but the
     plug-in `σ̂ᵢ² = ûᵢ²/Mᵢᵢ` is biased by `Σⱼ hᵢⱼ²(σⱼ² − σᵢ²)/(1 − hᵢᵢ)`.
     In `tc_vhat.jl` (n = 900, R = 1000, the oracle at 0.997–1.002) the
     robust variance is unbiased under constant and spatially smooth
     variances, but conservative under rough heteroskedasticity: 1.01 to
     1.13 (normal) and 1.03 to 1.17 (`χ²₃`) as K/n goes from 0.05 to 0.39.
     Cattaneo, Jansson & Newey's `(M∘M)⁻¹û²` fails here (1.1–2.7 × V at
     K/n ≥ 0.28), because the basis has leverage up to 1. Kline, Saggio &
     Sølvsten's leave-one-out `y₀ᵢûᵢ/Mᵢᵢ` is exactly unbiased per unit at any
     leverage below 1; in `tc_vhat_lo.jl` it brings the rough cases to
     0.974–1.030 (all 30 cells 0.968–1.030). But with a strong signal
     (`tc_vhat_lo_lin.jl`, linear part 90% of `V`) its quadratic part
     collapses, through a product term in `f` divided by `MᵢᵢMⱼⱼ`: the
     total is 10–17% low at K/n = 0.28 and negative at 0.39, where HC2 is
     within 4%. With a hat basis whose leverage stays ≤ 0.94
     (`tc_vhat_lo_hat.jl`) leave-one-out is still up to 7.5% low with a strong
     signal: its bias scales with the signal-to-noise ratio times K/n, not
     with leverage. Leverage 1 rules out any estimator free of `f`. CJN's
     `(M∘M)⁻¹û²`, which is free of `f`, is within 2% at K/n ≤ 0.11 at both
     signal strengths and falls to 0.84–0.99 at K/n = 0.25. Proposition 2
     proves the coded HC2 `V̂` consistent for K/n → 0, with four moments (not
     the eight the conjecture asked for) and a new condition A10, that the
     linear-part signal does not sit on high-leverage units. The key fact
     is that `Cs∘Cs` has bounded row sums although `Cs` is dense. In
     `tc_coverage_vhat.jl` HC2 gives the best coverage of `ρ̂`, 0.94–0.97
     in every cell, while leave-one-out (strong signal) and CJN (weak
     signal) fall to 0.91 at K/n = 0.25. CJN's quadratic-part bias is
     exactly `4tr(ΩΛ[(MΣM)∘(MΣM)]Λ)` for any error law, because the
     cumulant term meets the zero diagonal of `Cs∘Cs`; `tc_cjn_bias.jl`
     confirms it in 24 of 24 cells, and the plug-in correction works to
     K/n = 0.19 but not at 0.25. The linear part's bias has an exact
     formula too (`tc_cjn_lin.jl`, 32 of 32 cells); with both parts
     corrected CJN is within 1.6% of `V` at K/n ≤ 0.19, and worse than
     uncorrected at 0.25. Its coverage (`tc_coverage_cjnc.jl`) gains at
     most 0.5 points at K/n ≤ 0.19 and loses up to 3.6 at 0.25. So the
     coded HC2 `V̂` stays: consistent for K/n → 0 and covering 0.94–0.97 in
     every cell up to K/n = 0.25. Step 2 is closed for these designs;
     untested are K/n above 0.25, other `W`, and other `n`. Proposition 3
     (approximation error, `f ∉ span B`) replaces the sketched Remark 1:
     the moment's bias is `β = r′(W + D)r + r′r_W` with `r_W = MWP_Bf`, the
     sketch missed the second term, and that term dominates at large K in
     `tc_approx.jl` because a rough covariate's lag is outside `span B`. The
     interval covers `P(|Z + β/√V| ≤ 1.96)`, which the simulation confirms
     while `β/√V` ≤ 1.1; validity needs `β = o(√n)`, a lower bound on K.
     Proposition 4 carries the theory to a symmetric penalised smoother
     (`tc_smoother.jl`, a tensor P-spline): Theorem 1 holds (KS ≤ 0.034 in
     12 of 12 cells), shrinkage bias follows Proposition 3, and over-
     smoothing is what breaks coverage. The moment must be formed as
     `y₀′(MW + MDM)y₀`: the residual-vector shortcut assumes `M² = M`, and
     for a smoother it covers 0.86–0.90 where the correct moment covers
     0.95. A smoothed TC-CFE in `src/` must not use that shortcut.
     Proposition 4(c) proves its HC2 `V̂` (with `ûᵢ²/(M²)ᵢᵢ`) consistent
     when `tr S/n → 0` and `‖Mf‖² = o(n)`. `tc_smoother_vhat.jl` matches the
     exact shrinkage terms (|z| ≤ 2.2 in 10 cells); over-smoothing biases
     `V̂` down, to 0.55 of `V` at the largest λ. Choosing λ
     (`tc_lambda.jl`): GCV over-smooths for inference (coverage 0.888 and
     0.320), the RMSE-best λ already under-covers (0.908 at a = 4), and a
     plug-in `ĉ` has sd 0.8–2.9, too noisy to choose λ. A smoothed TC-CFE
     in `src/` must not default to GCV; undersmooth and report the
     sensitivity across λ. Step 2 for other `W` and `n`
     (`tc_coverage_robust.jl`: kNN-6, kNN-20, rook and a distance band;
     n = 400–1600; K/n up to 0.30; 90 cells): HC2 is within −2.6 to +1.8
     points of the oracle everywhere and covers 0.930–0.966 wherever
     `sd(ρ̂)` < 0.39. The seven failing cells are weakly identified (kNN-20
     at n = 400, `‖Cs‖_F²` 1.6–14), and the oracle fails there too.
     Proposition 6: the test-inversion (Anderson–Rubin) set
     `{ρ : g(ρ)² ≤ 1.96² V̂(ρ)}` is valid without A9, because Theorem 1
     never uses `J`. `tc_ar.jl`: it covers 0.938–0.981 in the weak cells
     (Wald 0.841–0.963) with normal and `χ²₃` errors, and in the controls it
     is as short as the Wald interval. `V̂(ρ)` is an exact quartic, so the
     set costs no more than `V̂`. It is the interval a smoothed or
     projected TC-CFE in `src/` should report. Over the whole grid
     with normal and `χ²₃` errors (180 cells) `CS` covers 0.932–0.992; it
     over-covers only where the leverage is near 0.95.
     Proposition 5 derives A5 from `W`: `‖Cs‖_F²/n ≥ κ_W/2 −
     2(K/n)(c_W + d̄)²` with `κ_W n ≤ ‖W‖_F²`, so A5 holds for K/n → 0. In
     `tc_a5.jl` the bound is close at small K/n (0.137 against 0.148) and
     vanishes by K/n ≈ 0.05. A spectral form, `‖Cs‖_F² ≥ ‖X‖_F²(1 − 2π̄_w)`
     with `π̄_w` the overlap of `B` with the eigenvectors of `X = Ws + D`,
     weighted by the squared eigenvalues `ωⱼ²`, needs no rate on K/n and is
     sharper (0.143 against
     0.148); it vanishes where `π̄_w` passes 1/2, at K/n ≈ 0.10. Beyond that
     A5 stays an assumption. `tc_a5.jl` also shows `d` exploding (to 10⁹)
     where the leverage reaches 1.
  3. Whether REsd's optimality under homoskedasticity (their
     Proposition 3) survives when K grows.
- **Checks still owed** by the null-distribution rule:
  - ~~a null-distribution check of the robust variance with REsd's matrix~~:
    done by `validation/mc_root_rho.jl` (2026-11-20; below);
  - ~~a penalised smoother, where `M = I − S_λ` is not idempotent~~:
    done by `validation/mc_root_rho_smoother.jl` (2026-11-20; below);
  - Geniaux's own E3 design.
- **Outcome if the theory holds:** a note on Geniaux: one closed-form
  correction, Jin & Lee's, removes the bias his three safeguards target.
- **Built (2026-11-20):** `root_rho(y, X, W; basis)` in `src/RootRho.jl`
  (submodule 33, testset 46, `examples/34_root_rho.jl`). It is a new
  function, not a `correction = :jin_lee` option on `closed_form_rho`,
  because it returns a set as well as a point. It is Jin & Lee's two-step
  REsd with a projection on `[X, basis]`, the HC2 `V̂` as an exact quartic
  in `ρ`, the Wald interval and the test-inversion set `CS`
  (Proposition 6), with the endpoints found as exact roots. `β` is a point
  estimate with no SE. The guards refuse a unit with leverage above 0.99 and
  `n > 4000`, and warn above `K/n` = 0.30.
  `validation/mc_root_rho.jl` measured REsd's plug-in in both intervals,
  which the research Monte Carlo had not: 108 cells (kNN-6, rook, kNN-20;
  n = 400–1600; K/n ≈ 0.05–0.19; two variance patterns; normal and `χ²₃`
  errors; R = 1000). `CS` covers 0.935–0.975 and the Wald interval
  0.931–0.975 in every cell. `|bias|` is at most 0.060, against −0.05 to
  −1.76 for `closed_form_rho` on the same draws, and the package equals the
  research code to 2.5e-14. AUDIT, "`root_rho`".
- **Built (2026-11-20, later): the penalised smoother.** `root_rho(y, X, W;
  basis, penalty, lambda)`, with `tensor_pspline(u, v; k)` for the basis and
  a difference penalty (testset 47, section 4 of `examples/34_root_rho.jl`).
  The moment is the one-step exact-diagonal `y₀′Ay₀`, `A = MW + MDM`, formed
  as a matrix, with HC2 `ûᵢ²/(M²)ᵢᵢ` (Proposition 4). `lambda` has no
  default, and `r.df = tr S` is reported. The shrinkage bias `f′Af` is the
  price: `validation/mc_root_rho_smoother.jl` (108 cells): in the 75 cells with `c ≤ 0.25` both intervals cover
  0.934–0.982; elsewhere `CS` is within 0.027 of `P(|Z + c| ≤ 1.96)`.
  AUDIT, "`root_rho` with a penalised smoother". Not built: a rule for
  choosing K.
- **Built (2026-11-20, later): the bias-aware interval.** `root_rho(...;
  roughness = ℛ)` adds `ci_bias_aware`, valid over `ψ′Πψ ≤ ℛ²` at any `λ`
  (THEORY.md, Lemma 4 and Proposition 7), and a vector `lambda` chooses `λ`
  by the interval's length, with the projection as `λ = 0`.
  `basis_roughness` states `ℛ` from a reference surface (testset 48,
  section 5 of `examples/34_root_rho.jl`). `validation/mc_root_rho_bias_aware.jl`:
  coverage 0.957–0.987 at `ℛ*` and `4ℛ*`, 26–44% shorter than the
  projection at `ℛ*`; 0.490 and 0.000 at `ℛ = 0`. AUDIT, "the bias-aware
  interval". `ℛ` is not estimated: every data rule under-covered.


**New file:** `src/SemiparametricSpatial.jl` (1,136 lines against the ~500
estimated; the four estimators with their variances, two bandwidth selectors
and the overlap measure)

**Dependencies:** none beyond what the package already loads (`NearestNeighbors`
for the smoother windows)

---

### 16. Spatial GARCH & Non-Geographic W

**Motivation:** Every model in this package puts the spatial structure in
the conditional mean. Financial and risk applications need it in the
conditional *variance*: volatility clusters across firms and assets that are
close in some economic sense, not in kilometres. Fülle & Otto (2024) fit
Sato & Matsuda's (2021) spatial log-GARCH to one year of log-returns of the
50 Dow Jones Global Titans firms. Their `W` is the `k`-nearest-neighbour graph
in a space of standardised balance-sheet positions, and the positions are
chosen by forward selection on the log-likelihood from 66 candidates. The
best `W` uses seven. This is also a clean demonstration that the package's
machinery is not about geography.

**Correction, 2026-11-16, when the paper was read:** this paragraph said the
seven-covariate `W` "beats every geographic candidate". The paper fits no
geographic `W`: Table 2 compares covariate spaces only. The deliverables
table also had the model wrong: `ρ` multiplies `W log y²`, and `λ`
multiplies `W log h`.

**Deliverables:**

| Feature | Description | Reference |
|---|---|---|
| Spatial log-GARCH | `y_i = √h_i ε_i`, `log h = α1 + ρW₁ log y² + λW₂ log h (+ Zδ)`. With `W₁ = W₂ = W` it is a SARMA(1,1) in `log y²`: `log y² = (λ + ρ)W log y² + α1 + Zδ + (I − λW) log ε²`. A solution exists if `\|λ + ρ\| < 1` for row-standardised `W`. Estimated by Sato & Matsuda's two-step procedure: Gaussian QML of the SARMA for `(λ, ρ, δ)`, with the intercept absorbing `E log ε²`, then `α` in closed form (their eq. 11) from the `log χ²₁` quasi-likelihood | Sato & Matsuda (2021); Fülle & Otto (2024) |
| Covariate-distance W | `k`-nearest-neighbour `W` from Euclidean distance between standardised covariate vectors, which `knn_weights` already computes in any dimension. Fülle & Otto choose the covariates by forward selection on the log-likelihood | Santi et al. (2021); Fülle & Otto (2024) |
| Endogeneity caveat | Diagnostics for feedback from `y` into the covariates that define `W`, the failure mode the editorial flags | Qu & Lee (2015) |

**What the two papers establish, read 2026-11-16** (Fülle & Otto in full;
Sato & Matsuda's main text and its four-page supplement of proofs):

- **Only Sato & Matsuda have theory.** Theorems 3.1–3.2 give consistency
  and asymptotic normality of the first-step `(λ, ρ, δ, σ²)` under
  Assumptions 3.1–3.8 (`log ε²` i.i.d. with `4 + η` moments, bounded `W`,
  a compact parameter space), with the sandwich `Σ⁻¹ + Σ⁻¹ΩΣ⁻¹` for
  non-normal `log ε²`. Theorem 3.3 gives only consistency for `α̂`, with no
  distribution, so there is no standard error for `α`, and their Table 3
  reports SEs for `λ` and `ρ` only. The proof of Theorem 3 needs only
  `E ε² = 1`, not Gaussian `ε`. **The explicit `Ω_ψ` is not given.** The
  main text says it is "expressed explicitly in Appendix A", but the
  supplement gives only the Hessian deviations `D_ψψ`, so the middle of the
  sandwich must be derived here, in Lee's (2004) form for linear-quadratic
  scores with the third and fourth moments of `log ε²`. Theorem 2's proof
  also has a truncated sentence. Their simulations (rook `W`,
  `n ∈ {100, 400}`, normal, `χ²₃` and log-normal `ε`) include a regressor.
  The first step is nearly unbiased there; `α̂` is not at `n = 100` with a
  small `λ` (bias −0.43 to −1.01).
- **The mean part cannot separate `λ` from `ρ`.** Assumption 3.8 is the
  mean-part identification condition of Lee (2004). The mean of `log y²`
  depends on `(λ, ρ)` only through `λ + ρ`, so the condition's quantity is
  zero all along the ridge `λ + ρ = λ₀ + ρ₀`, whatever the regressors. With
  an intercept only and a row-standardised `W` (Fülle & Otto's case), it is
  zero everywhere. Checked numerically on 2026-11-16: about `1e-30` in both
  cases, against 0.10 off the ridge with one regressor. This is this
  roadmap's derivation, not the paper's. So `λ` and `ρ` are told apart only
  by the variance part, the quadratic identification condition of Lee
  (2004), which neither the main text nor the supplement states. The
  supplement's uniqueness step cites an information inequality, which gives
  only `Q(θ) ≤ Q(θ₀)`, and then says uniqueness "holds by contradiction in
  the same way as Lee (2004)". Fülle & Otto's simulation shows
  what that costs: with true locations at `n = 200`, sd(`λ̂`) is .26–.31
  against .08–.12 for `ρ̂`.
- **Fülle & Otto's estimator is not Sato & Matsuda's.** They maximise the
  full likelihood with `ε ~ N(0, 1)` and the Jacobian of their eqs (5)–(8),
  and give no asymptotic theory: identifiability is referred to Sato &
  Matsuda (note 1). Their Table 3 reports SEs, including one for `α`, with
  no stated method.
- **Their simulation table is internally inconsistent.** For `α`, the RMSE
  is below the sd in every cell (e.g. `n = 50`, `ω = 0`: sd .280, RMSE
  .090), which cannot happen, since RMSE² ≈ sd² + bias². §3.1 also says an
  inverse-distance `W^(X)` is computed, where §2.1 and the design use `k`NN
  with `k = 10`.
- **A proxy `W` makes the estimator inconsistent, not just noisier.** Their
  conclusion says the estimates are "asymptotically unbiased … when a large
  linear correlation is imposed". In their Table 1 that holds at `ω = 0`,
  which is the true locations. At `ρ₀ = .3`, `ω = .5`, `ρ̂` is .05–.06 at
  every `n`, and its RMSE stays at .26.
- **The application has no inference on the selection.** It is one cross
  section, one return per firm (`n = 50`). `W` is chosen by the largest
  log-likelihood over forward steps through 66 covariates, with no penalty
  for the search. AIC and BIC are set aside because each candidate has the
  same three parameters. The chosen model has `ρ̂ = .034` (.016) and
  `λ̂ = .869` (.019).

**What a build would therefore be:**
- Sato & Matsuda's two-step estimator with its sandwich SEs for `(λ, ρ, δ)`,
  `α̂` without an SE;
- the covariate-distance `W` as a constructor;
- forward selection, if offered at all, as a search with no inference
  attached.

Before any of it is offered, the open questions get a null-distribution
Monte Carlo: the weak identification of `λ`, the intercept-only case, and
exact zeros in `y`. An exact zero makes `log y²` undefined, and neither
paper deals with it. The derived `Ω` gets the same check as every variance
here: its mean against the exact variance of the score at the true
parameters, by simulation.

**Santi et al. (2021), read 2026-11-16.** This is where the covariate-distance
`W` comes from. It is proposed for the conditional mean (SLM, `y = ρ_X W_X y
+ Xβ + u`), not for GARCH, and as an empirical device, not an estimator with
theory. The paper says it "does not guarantee that point estimators and
standard error estimators are consistent" (p. 3).
- **Approximation quality is measured against the true `W`.** It is
  `cor(W_X u, W u)`, from the Frobenius distance `‖ρ_X W_X − ρW‖²_F` (eq.
  8), so it needs the very `W` that is unknown. In the Beijing application
  they could compute it (0.4428) only because latitude and longitude were
  in the data. There is no diagnostic for the case the method is for.
- **Its guidance is derived for `k`NN constant-weight matrices:**
  - use only spatially non-stationary covariates, standardised;
  - choose `k` close to the true neighbourhood size, slightly larger if
    unsure;
  - use constant weights within a neighbourhood;
  - row standardisation is then neutral.
- **`ρ̂` is badly attenuated unless the proxy is good.** At `ρ = .7`, the
  relative bias of `ρ̂` is −80% for `cor(W_X u, Wu) ≤ .1`, −8% at .7–.8, and
  −0.4% above .9 (Table 2). Slopes of covariates are less biased than OLS
  throughout. That is the paper's claim, and the tables support it.
- **Table 1's `ρ` block and LM-test block have misplaced row labels**: four
  rows for three models, one unlabelled. Table 2 has the same layout
  correctly labelled.

So all three papers agree that a proxy `W` gives biased `ρ̂` that a larger
`n` does not cure, and none gives a way to measure the proxy's quality from
the data alone. A build should say so where it offers the constructor.

✅ **Built 2026-11-16** as `fit_spatial_log_garch` (`src/SpatialGARCH.jl`,
submodule 32; testset 44; example 33) and `covariate_weights`. The
constructor stands alone: the user chose on 2026-11-16 not to have a forward
selection search. The endogeneity-caveat row is not built. What the build
added to the reading (details in `docs/AUDIT.md`):
- **`Ω` is derived and checked.** At `ψ₀` every score component is
  linear-quadratic in `V`, so its covariance is exact in `σ²`, `μ₃` and
  `μ₄`. Simulation under three error laws matches it (largest `|z|` 1.9 over
  15 entries). The Gaussian information fails the same test (`|z|` = 30).
- **The quasi-likelihood is unbounded above as `λ → 1`, in every sample.**
  Because `W1 = 1`, the intercept absorbs what `R(λ)⁻¹` inflates, so `σ̂²`
  stays bounded while `−log|R(λ)|` grows like `−log(1 − λ)`. Neither paper
  states this. The estimate is an interior local maximum. At `n = 100` with
  `(λ, ρ) = (.9, .05)`, 36–39% of samples have none, and the fit throws.
  Sato & Matsuda's Table 1 reports every sample.
- **Table 1 is reproduced at `n = 400`.** At `n = 100`:
  - the `(.9, .05)` cells differ, for the reason above;
  - the `λ` and `ρ` RMSEs at `(.05, .9)` are about a third larger than
    theirs. A diagnostic run rules out the start and the `z` draw. A
    narrower search region, unstated in the paper, is the most likely
    cause: `λ ≥ 0` moves the figures towards theirs and past them.

  Coverage at `n = 400`, which they do not report, is .93–.96. The
  exception is `λ` and `ρ` at `(.9, .05)`, at .87–.89.
- **With an intercept only, `λ` is identified in practice.** Its sd halves
  as `n` quadruples (.125 at 400, .061 at 1600), and coverage is .94–.96
  from `n = 400`. But its sd is 2.3 times that of `λ̂ + ρ̂`, and
  `corr(λ̂, ρ̂)` ≈ −.95.
- **A proxy `W` biases `ρ̂`, and a larger `n` makes it worse at fixed
  covariate noise.** At `n = 1600`:
  - a proxy that keeps 37% of the true neighbours biases `ρ̂` by −.08, with
    coverage .53;
  - one that keeps 6% biases it by −.17, with coverage .11.

**Deferred to a possible later pass, by the user's decision of 2026-11-16:
Fülle & Otto's forward selection of the covariates.** It would leave the
fit and the constructor unchanged, but before it ships it needs three
things:
- **Its own null Monte Carlo.** Generate data with a true `W` unrelated to
  the candidate covariates, run the selection, and count how often the
  selected model's `ρ` is significant. The sandwich treats `W` as fixed, and
  neither paper measures what the search costs.
- **A rule for candidates with no interior maximum**, which throw.
- **A check that the search does not favour candidates whose `λ̂` sits near
  1**, where the `−log(1 − λ)` term inflates the likelihood.

If it is built, it should return the ranked candidates with a warning that
the SEs ignore the search.

**New file:** `src/SpatialGARCH.jl` (~450 lines estimated; built: 454, plus
59 in `SpatialUtils` and 62 in `SpatialWeights`)

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
| Paper: JOSS, then JSS | **Decided 2026-11-16: both, JOSS first, on the back burner while the companion book is written.** JOSS: a short `paper.md` / `paper.bib` in the repository, reviewed as software. JSS: the full article with a replication script. **The contribution to argue** is not the feature list, which `spatialreg` and PySAL largely share, but: (a) the breadth in one framework (short-`T` panels, spatial log-GARCH, semiparametric, causal, networks); (b) the validation discipline (null-distribution Monte Carlo for every variance, value-against-value tests, the `docs/AUDIT.md` ledger); (c) what replication found in the sources (Sato & Matsuda's unbounded likelihood and missing `Ω`, the bias Bai & Chen's design hides, Fülle & Otto's inconsistent table). **Prerequisites before submission:** installation by name, which reopens registration; the cross-language benchmarks of Phase 8; a real-data application (PortugalINE.jl data, or the Columbus data already in `notebooks/data/`); a statement of how the software was developed, including generative-AI assistance, per each journal's policy at the time; authorship settled by the author |
| Contributing guide | `CONTRIBUTING.md` with coding conventions, testing requirements, and PR checklist |
| CI/CD pipeline | GitHub Actions for automated testing on Julia 1.10, 1.13, nightly; code coverage via Codecov |
| Claude skill — ✅ **built 2026-11-17** (`skills/`; see `skills/README.md`) | A skill for Claude Code: `SKILL.md` plus reference files. **Contents:** a decision guide from data type to estimator (cross-section, panel, dynamic, variance, unknown locations); the `W` constructors and their pitfalls (row standardisation, connectivity, proxy `W`); how to read each fit's warnings and throws (e.g. no interior maximum near `λ = 1` means no `λ̂` to report); per-function reference generated from the docstrings, with a check that keeps it in sync. **Limits:** it needs Julia installed, and a persistent REPL or a system image to avoid paying compile latency on every call. Estimated at a day |
| MCP server (after registration) | For Claude Desktop, which has no shell. **Design:** a persistent Julia process (which also removes compile latency); a few coarse tools (`load_data`, `build_weights`, `fit`, `diagnostics`, `summary`), with fits held server-side by handle; results, warnings and throws returned as structured JSON; packaged for one-click install. **Prerequisites:** the skill's decision guide, which supplies the tool descriptions, and registration, so the install does not depend on a GitHub URL. Check the maturity of the Julia MCP libraries before choosing between native Julia and a thin Python wrapper. Estimated at about a week |
| Notebooks on Google Colab — ✅ **built 2026-11-17, checked on Colab 2026-11-18** (notebook 02 on a fresh session: only the package's dependencies precompiled, and the run was clean) | Decided 2026-11-17. Colab's native Julia runtime was assumed to be Julia 1.10 LTS; on 2026-11-18 it was 1.12, which the package also supports. **Changes:** a shared setup cell that activates the repository when it runs inside a clone and otherwise runs `Pkg.add(url = …, rev = REV)`, `REV` being the latest tag (bumped at each release); a data helper that reads `notebooks/data/` or downloads the file from `raw.githubusercontent.com` (notebooks 04, 11–13); neutral Julia kernel metadata in place of `julia-1.11`; "Open in Colab" badges in `notebooks/README.md`. **Check:** run every notebook from an empty folder on Julia 1.10 locally; the author opens one from its badge on Colab. The Pluto notebooks are out of scope. **Done 2026-11-17:** the setup cell finds the clone by walking up from `pwd()` to a `Project.toml` named `SpatialEconometrics`; outside one it installs `REV` into a temporary environment, with `StatsAPI`, which notebooks import. Notebook 05 also reads data, so the helper covers 04, 05 and 11–13. **Checked:** all 13 run from an empty folder on Julia 1.10.13 against `v0.5.0` from GitHub, with the data downloaded, and the folder stays empty; all 13 run inside the clone on 1.13.3 against `master`, with no finding-33 warning. The 1.10 run found three notebook bugs, now fixed: 02 used `coefnames` without importing it; 06 assigned to `eps` after calling `Base.eps`, which 1.10 refuses; 12 built `W` with `Float64.(W .> 0)`, whose eltype is `Any` on 1.10. **Fixed 2026-11-18:** installing into Colab's kernel environment made `Pkg.add` precompile every package Colab ships (over 20 minutes on the author's run: Makie, Lux, GPUCompiler, DataFrames…), although it changed no versions. The install now runs with `JULIA_PKG_PRECOMPILE_AUTO=0`, and `Pkg.precompile(["SpatialEconometrics", "StatsAPI"])` then compiles only their dependency tree, in parallel (leaving it to `using` compiled the tree one package at a time, and IJulia on 1.12 printed `caught exception of type :MethodError while trying to print a failed Task notice` meanwhile); checked locally on 1.10 against an environment holding an unprecompiled package, which stays unprecompiled (the old cell compiled it). **When a release is tagged, set `REV` in all 13 notebooks to it**, or a notebook on `master` runs against the older release |

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

### Phase 7A — ✅ Complete

| Priority | Item | Est. Lines | Status |
|---|---|---|---|
| 13 | Spatial Causal Inference (#15) | ~600 | ✅ **Complete.** Exposure mapping and spatial DiD 2026-10-30; spatial synthetic control and shift-share diagnostics 2026-10-31. About 2,900 lines across `SpatialCausal.jl`, the new `SpatialShiftShare.jl` and `SpatialHAC.jl`, against an estimate of ~600 for a scope that excluded three of these |
| 14 | Spatial Networks (#11) | ~400 | ✅ **Complete** 2026-11-01. 1,187 lines in the new `SpatialNetworks.jl`. The "mostly reframes what `W` may represent rather than adding inference" rationale was wrong — see #11 — though the ordering it produced was right |
| | **Phase 7A total** | **~1,000** | **Actual: about 4,100 lines across four submodules** |

The #15 row's original rationale said it "builds on `spatial_effects` —
already written and audited". That turned out to be half true and is worth
recording: what is reusable is the *pattern* — a delta method over a closed
form with a swappable covariance — not the function, which hard-codes the
panel coefficient layout (intercept offset, `ρ` last, `K = P_cov` or
`P_cov/2`) that a DiD design does not have. In the event the two-period
decomposition needed no delta method at all, every reported effect being
linear in `θ̂`.

The networks pass is the one where reading the source changed the work. The
plan called the identification check a formality implied by the roadmap's
own refinement; the refinement turned out to describe a different quantity
from the published condition, and the published one fails on equal-sized
groups, on self-included group means whatever the sizes, on two group sizes
under fixed effects, and on stars. All four are now pinned as integer rank
facts. Two things were also caught by measurement rather than by reasoning:
the rank tolerance, whose first value sat below the numerical noise floor
and reported a degenerate network as identified; and the claim that a robust
covariance is needed here, which is true only under heteroskedasticity
around an order of magnitude in spread and is invisible at 1.5x.

Reuse was high and from the expected places for once: `fit_spatial` is the
network autocorrelation model with a different `W`, `brent_maximize` is the
inner solver of the multiplex coordinate ascent, `row_standardize` already
left isolate rows all-zero, and `spatial_trace_powers` was already
documented as safe for the non-symmetric `W` a directed graph produces. The
one thing the pass added rather than reused was a fourth copy of a
chi-squared survival function — in `SpatialShiftShare`, the previous pass,
spelled `_chisq_sf` so the existing identity test could not see it. Removed,
and that test now pins module identity rather than name.

The synthetic control pass reused more than expected, and from a different
place: `exposure` (written for the DiD) finds the neighbours to screen out
of the donor pool, and `SpatialHAC`'s `pairwise_distance` with its
`EuclideanDistance` / `HaversineDistance` types supplies the penalty metric,
so lon/lat data works without new code. What it could **not** reuse was an
optimiser — the package has no Optim or JuMP dependency and none was added;
the simplex-constrained program is solved by accelerated projected gradient
with an exact Lipschitz step, about 110 lines, and verified against an
independent support enumeration.

### Phase 7B — 🔶 Spatio-temporal, common factors & semiparametrics

| Priority | Item | Est. Lines | Rationale |
|---|---|---|---|
| 18 | ✅ Spatio-Temporal Models (#12) | ~500 | STAR, forecasting, interactive FE; SpVAR in a second pass |
| 19 | ✅ Common factors in panels (#1) | ~400 | SP-CCE and the CD / `α` exponent tests |
| 20 | 🔶 Semiparametric Spatial (#13) | ~500 (pass 1: 1,136) | Pass 1 done 2026-11-13: Su & Jin PLSAR, closed-form `ρ`, the overlap `π̄`, LGG `β(τ)`. Geniaux's remedy, quantile and penalised rows remain |
| 21 | ✅ Spatial GARCH (#16) | ~450 (actual: 575) | Sato & Matsuda (2021) two-step log-GARCH with a derived sandwich, plus `covariate_weights`; built 2026-11-16, Monte Carlo in `docs/AUDIT.md` |
| 22 | ✅ **SpVAR** (deferred from #12) | ~450 | Di Giacinto (2010): block-triangular `C₀`, separable FIML, outward/inward STIR, the nested panel VAR and the LR test against it; plus `order_weights` |
| 23 | ✅ Short-`T` CRE dynamic panel (#1 planned addition) | ~1,500 (actual: 1,029) | Li & Yang (2021) M-estimator with the corrected plug-in variance of Li, Meng & Yang (2026); built 2026-11-14, Table 1 replicated (`docs/AUDIT.md`) |
| 24 | ✅ Short-`T` dynamic panel with common shocks (#1 planned addition) | — (actual: 691) | Bai & Chen (2026) SERL QML with the Appendix D sandwich; built 2026-11-14, Monte Carlo to `N = 4000` in `docs/AUDIT.md` |
| | **Phase 7B total** | **~2,300** | Items 18, 19 and 22 closed 2026-11-04; items 23 and 24 on 2026-11-14; item 21 on 2026-11-16 |

**Items 18 and 19 were taken as one pass**, on 2026-11-04, because the
roadmap cross-references them as two halves of one phenomenon and that is
right: the CD and `α` tests say whether a factor structure is *there*, and
interactive fixed effects *estimate* it. Shipping either alone would have
delivered a test whose remedy does not exist, or a remedy with no test to
motivate it.

**Row 22 was the deferral, and it was split rather than carried.** SpVAR
and time-varying `ρ` were both cut from item #12's first pass by explicit
decision, recorded here so the scope reduction was visible in the plan
rather than only in the ledger. On the same day the row was reopened, and
reading the sources changed its shape: **SpVAR was built** (Di Giacinto
2010, verified and read in full), and **time-varying `ρ` moved to #13**
because its roadmap entry was unsourced.

**And then the replacement citation failed too.** The move named Liang,
Gao & Gong (2022) as the published method for `ρ_t`. Its bibliographic
record had been verified — title, journal, volume, issue, pages, DOI, all
correct — but the **paper itself had not been read**, because tandfonline
is paywalled and the open version was left for "when #13 is built". When it
was finally obtained, its `ρ` turned out to be a scalar constant. Verifying
that a paper *exists as cited* is not verifying that it *does what you say
it does*, and the first check had been mistaken for the second.

The lesson is worth keeping: **a deferred row is not a frozen row.** Row
22's original description proposed "state-space or rolling-window" for
`ρ_t`, with an empty reference column. Had it been implemented as written,
the package would have shipped an invented method. The reference column
being empty was the signal, and it was only read because the row was
reopened and re-sourced rather than executed from its own summary.

### Phase 8 — ⬜ Ecosystem, benchmarks & JSS submission

| Priority | Item | Est. Lines | Rationale |
|---|---|---|---|
| 22 | Full documentation & paper (#14) | ~2,000 | JSS paper, tutorials, benchmarks |
| 23 | Cross-language benchmarks | ~400 | Against R `spatialreg` and Python `PySAL/spreg` |
| | **Phase 8 total** | **~2,400** | |

### Size: Actuals and Projections

Rows above the rule are measured; rows below are estimates. v0.4.0 has
crossed the rule: it is not a projection any more but a tagged release
(2026-11-06), and the figures it once projected — ~1,200 assertions and
~13,800 lines — were passed some time ago. v0.5.0 (2026-11-16) and v0.6.0
(2026-11-18) are likewise measured rows.

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
| **master** (+ #15 shift-share, so #15 complete) | 24 | 3,051 † | 19,085 |
| **master** (+ #11 networks, so Phase 7A complete) | 25 | 3,217 † | 20,289 |
| **master** (+ #1 and #12, so Phase 7B half done) | 27 | 3,415 † | 22,577 |
| **master** (+ row 22 SpVAR, so Phase 7B mostly done) | 28 | 3,718 † | 24,151 |
| **master** (+ Pfeifer & Deutsch read: STAR `spatial_weights`) | 28 | 3,750 † | 24,253 |
| **v0.4.0** (tagged 2026-11-06) | 28 | 3,750 † | 24,321 |
| **master** (+ the robust-sandwich fix and seven exact assertions) | 28 | 3,757 †† | 24,321 |
| **master** (+ #13 pass 1, `SemiparametricSpatial`) | 29 | 3,775 ‡ | 25,558 |
| **master** (+ #1 short-`T` row, `CREDynamicSpatialPanel`) | 30 | 3,864 § | 26,597 |
| **master** (+ the 41b/41f optimiser-tolerance fix) | 30 | 3,866 §§ | 26,597 |
| **master** (+ #1 common-shocks row, `CommonShockDynamicPanel`) | 31 | 3,897 ¶ | 27,293 |
| **master** (+ #16 spatial log-GARCH, `SpatialGARCH`) | 32 | 3,949 ¶¶ | 27,865 |
| **v0.5.0** (tagged 2026-11-17, = the row above) | 32 | 3,949 ¶¶ | 27,865 |
| **master** (+ AUDIT finding 33, the `ρ` interval and the polish) | 32 | 3,985 ¶¶¶ | 28,142 |
| **master** (+ the Claude Code skill, #14) | 32 | 3,988 ¶⁴ | 28,143 |
| **master** (+ #15's two-channel spatial DiD and `errors = :spatial`, AUDIT 34) | 32 | 4,055 ¶⁵ | 28,391 |
| **master** (+ the heteroskedasticity-robust spatial error, `:spatial_gmm` / `estimator = :gmm`, AUDIT 35) | 32 | 4,099 ¶⁶ | 28,694 |
| **master** (+ the heteroskedasticity-robust GS2SLS and the default path's panel fixes, AUDIT 36) | 32 | 4,139 ¶⁷ | 29,005 |
| **v0.6.0** (tagged 2026-11-19, = the row above) | 32 | 4,139 ¶⁷ | 29,005 |
| **master** (+ `root_rho`, the research item on a growing basis built) | 33 | 4,182 ¶⁸ | 29,391 |
| **master** (+ `root_rho` with a penalised smoother, and `tensor_pspline`) | 33 | 4,241 ¶⁹ | 29,643 |
| **master** (+ the sign rule's check in `root_rho`, after a review of THEORY.md) | 33 | 4,248 ¶¹⁰ | 29,694 |
| **master** (+ `root_rho`'s bias-aware interval and `basis_roughness`) | 33 | 4,287 ¶¹¹ | 29,929 |
| — | | | |
| **v1.0.0** (Phases 6C–8) | 29–31 | ~4,100 | ~26,500 |

†† Measured by CI on 2026-11-06: 3,757 with both extensions live and no
broken entries, against 3,722 and two broken locally without them. The figure
was first entered as arithmetic from the local run and CI agreed with it
exactly. The `src/` count is unchanged because the fix altered one existing
line.

‡ Measured locally on 2026-11-13, on Julia 1.13 and 1.10 alike: 3,775 passed
and 2 broken, without the extensions. That is the previous local figure,
3,722, plus the 53 assertions of testset 41. CI, with both extensions live,
should read 3,810 by the same difference as the †† row; that is arithmetic
until CI runs. The smallest margin in testset 41 is 208× inside its
tolerance (the 41f log-likelihood on 1.13). The `src/` count rose by 1,237:
the new module's 1,131 lines, the umbrella's 8 lines that load it, and 98
net lines of docstring prose in four commits since the †† row (`f7bea6a`,
`a4a09e1`, `6ff3bbd`, `e1a681e`).

§ Measured locally on 2026-11-14. On Julia 1.13.3 with both extensions loaded:
3,864 passed, none broken. On 1.10.13 without them: 3,829 passed and 2
broken. Each is the ‡ figure plus the 54 assertions of testset 42, and the
1.13 figure confirms the arithmetic 3,810 of the ‡ note. Margins were measured
on both versions: the smallest is 1.47×, for the `z` test of the exact
`Var(S*)` reference in 42c. The `src/` count rose by 1,034: the new module's
1,029 lines and the umbrella's 5.

§§ Measured locally on 2026-11-14: 3,866 on Julia 1.13.3 with the extensions,
and 3,831 plus 2 broken on 1.10.13 without them. That is the § figure plus the
two exact first-stage assertions added to 41b. CI confirmed the § figures
exactly before this change: 3,864 with the extensions, and 3,829 plus 2 broken
on every other job. The 41b fix is in `docs/AUDIT.md`.

¶ Measured locally on 2026-11-16. On Julia 1.13.3 with both extensions
loaded: 3,897 passed, none broken. Without them, on 1.13.3 and on 1.10.13
alike: 3,862 passed and 2 broken. Each is the §§ figure plus the 31
assertions of testset 43. Margins were measured on both versions. The
smallest in testset 43 is 1.65×, for the largest `|z|` of the null test 43e
on 1.13 (2.73 against 4.5; 2.67 on 1.10), so the suite's smallest margin is
still 42c's 1.47×. The `src/` count rose by 696: the new module's 691 lines
and the umbrella's 5. A plain `Pkg.test()` does not see the extensions even
when they are installed globally, because its sandbox holds only the test
dependencies; the extension figure is from `julia --project=. test/runtests.jl`.
CI confirmed both figures exactly on `a336b0f`: 3,897 on the extensions job,
and 3,862 plus 2 broken on 1.10 and 1.13 (Linux), 1.13 on Windows and
macOS, and nightly.

¶¶ Measured locally on 2026-11-16. On Julia 1.13.3 with both extensions
loaded: 3,949 passed, none broken. On 1.10.13 without them: 3,914 passed and
2 broken. Each is the ¶ figure plus the 52 assertions of testset 44. Margins
were measured on both versions. The smallest in testset 44 is 1.95×, for the
power check of 44d on 1.10: the Gaussian information's `|z|` of 29.3 against
a required 15. So the suite's smallest is still 42c's 1.47×. The `src/`
count rose by 572:
- the new module's 454 lines;
- 59 in `SpatialUtils` for the two shared helpers;
- 62 in `SpatialWeights` for `covariate_weights`;
- 5 in the umbrella;
- less 8 in `CommonShockDynamicPanel`, whose log-determinant moved to
  `SpatialUtils`.

The v1.0.0 projection in the table is now passed on two of its three
counts: 32 submodules against 29–31, and 27,865 lines against `~26,500`.
The assertion count, 3,949 against `~4,100`, is not yet passed. The
projection is left as written, like the earlier ones.

¶¶¶ Measured locally on 2026-11-17. On Julia 1.13.3 with both extensions
loaded: 3,985 passed, none broken. On 1.10.13 without them: 3,950 passed and
2 broken. Each is the ¶¶ figure plus the 36 assertions of testset 29e
(finding 33). Testset 29e first errored on 1.10: `Float64.(W .> 0)` on a
sparse matrix has eltype `Any` there, so 21 of its assertions never ran. The
test now types the matrix explicitly. Margins were measured on both
versions:
- the `ρ̂` and log-likelihood comparisons are 29× or more inside their
  bounds;
- the check that `ρ̂ < −1` on a widened interval holds by 3.3 standard
  errors on 1.10 and 6.3 on 1.13.

The suite's smallest margin is still 42c's 1.47×. The `src/` count rose by
277: 161 lines in `SpatialUtils` for the spectral bounds, the edge warning
and the polish, and 116 across the 11 entry points that call them.

¶⁴ Measured locally on 2026-11-17: 3,988 passed on Julia 1.13.3 with both
extensions, and 3,953 plus 2 broken on 1.10.13 without them. Each is the ¶¶¶
figure plus the 3 assertions of testset 45, which checks the skill's
generated function reference against the docstrings: one that it is current,
and two that show the check has power (a changed, a missing and an extra
file are each caught). The reference is the same on both versions because it
is built from the raw docstring text. The `src/` count rose by 1, from the
docstring fix in `1d0f206`.

¶⁵ Measured locally on 2026-11-18: 4,055 passed on Julia 1.13.4 with both
extensions, and 4,020 plus 2 broken on 1.10.13 without them, both 67 above
¶⁴: testsets 34k (the split channels) and 34l (the spatial error), and two
assertions added to 34b. The `src/` count rose by 248, all in
`SpatialCausal.jl`, about half of it docstring prose carrying the measured
coverage.

¶⁶ Measured locally on 2026-11-18: 4,099 passed on Julia 1.13.4 with both
extensions, and 4,064 plus 2 broken on 1.10.13 without them, both 44 above
¶⁵: testsets 10a (the GM estimator's moments, their null distribution and
its steps) and 34m (the DiD route). The `src/` count rose by 303, in
`SpatialErrorModel.jl` (the estimator, its option and docstring) and
`SpatialCausal.jl` (the DiD option and its measured tables).

¶⁷ Measured locally on 2026-11-18: 4,139 passed on Julia 1.13.4 with both
extensions, and 4,104 plus 2 broken on 1.10.13 without them, both 40 above
¶⁶: testset 4d (the robust GS2SLS's derivative, its null distribution, its
steps by hand, the default path's corrected covariance and Sargan `σ̂²`, and
a coverage power check; 38) and two lines in testset 8. Testset 4c's `λ` check was
updated in place. Top-level
testsets rose from 61 to 62. The `src/` count rose by 311, in
`SpatialGS2SLSPanel.jl` (the estimator, the default path's fixes and the
docstring's measured table); the moment helpers moved from
`SpatialErrorModel.jl` to `SpatialUtils.jl`.

¶⁸ Measured locally on 2026-11-20: 4,182 passed on Julia 1.13.4 with both
extensions, and 4,147 plus 2 broken on 1.10.13 without them, both 43 above
¶⁷: testset 46 (`root_rho`). Its smallest margin is 2.24×, for the mean of
the standardised moment in 46d on 1.10 (0.103 against 0.231; 0.102 on
1.13), so the smallest recorded for the suite is still 42c's 1.47×. Top-level testsets
rose from 62 to 63. The `src/` count rose by 386: the new module's 381
lines and the umbrella's 5.

¶⁹ Measured locally on 2026-11-20: 4,241 passed on Julia 1.13.4 with both
extensions, and 4,206 plus 2 broken on 1.10.13 without them, both 59 above
¶⁸: testset 47 (`root_rho` with a penalised smoother, 59 assertions). Its
smallest margin is 2.80×, for the variance of `g(ρ₀)/√V` under `f₀` in 47d on
1.10 (0.0319 against 0.0894); on 1.13 it is 2.95×, for the mean of the
centred moment under a shrunk `f`. Testsets 46c and 47c now make a fixed
number of assertions: they had guarded `@test`s over the intervals of `CS`,
so 47c ran 2 fewer on 1.13's draw than on 1.10's.
Top-level testsets rose from 63 to 64. The `src/` count rose by 252: 210 in
`RootRho.jl` (the smoother path, its variance, `tensor_pspline` and the
docstring) and 42 in 13 other files, where titleless references got full
entries.

¶¹⁰ Measured locally on 2026-11-20: 4,248 passed on Julia 1.13.4 with both
extensions, and 4,213 plus 2 broken on 1.10.13 without them, both 7 above
¶⁹: testset 47g (the sign rule's check, on a hub design with `J < 0`; see
AUDIT, "the sign rule's assumption"). Testset 47 now has 66 assertions. The
`src/` count rose by 51, all in `RootRho.jl` (the 2SLS check and the
docstring's "The root rule").

¶¹¹ Measured locally on 2026-11-20: 4,287 passed on Julia 1.13.4 with both
extensions, and 4,252 plus 2 broken on 1.10.13 without them, both 39 above
¶¹⁰: testset 48 (the bias-aware interval). The `src/` count rose by 235,
all in `RootRho.jl`.

Measured on master: 33 submodules plus the umbrella across 29,929 lines of
`src/`, 11,837 lines of tests, 5,356 lines of examples across 34 scripts,
398 lines of package extensions, 68 documentation pages and 20 notebooks
(13 Jupyter, 7 Pluto).

The `v0.4.0` row sits 68 lines above the measured row before it, and no
feature accounts for the difference: it is docstring prose, from the two
citation corrections of 2026-11-05 and 2026-11-06 in `SpatialCausal.jl` and
`SpatialUtils.jl`. Test assertions and module count are unchanged at 3,750
and 28, which is the check that says so.

**Count `src/` the same way each time, and say which way.** The figures in
the last column are `cat src/*.jl | wc -l`. That method gives **22,598** at
the previous row's commit, which this table recorded as **22,577** — so the
series carries a method difference of about twenty lines at that join. The
gap is not worth retrofitting through eleven rows, but it is worth naming,
because a column whose definition nobody writes down is a column that
drifts. From this row on the method is the one stated here.

**The v1.0.0 projection has now been passed twice in two days, and that is
worth saying rather than quietly editing.** It read `~22,000` lines until
2026-11-04, when #1 and #12 took `master` to 22,577 with two Phase 7B rows
still open, and it was raised to `28–30` modules, `~3,700` assertions and
`~24,500` lines. Row 22's SpVAR, **the next day**, took `master` to 28
modules, 3,718 assertions and 24,151 lines — so the module count was met,
the assertion figure **exceeded** (3,718 against `~3,700`), and the line
figure 98.6% met, with #13 and #16 both still open. One row consumed a
projection that was meant to cover three.

Raised again, to `29–31` / `~4,100` / `~26,500`, on the same basis: #13
(semiparametric, and now carrying time-varying `ρ` too) and #16 (spatial
GARCH) are estimated at ~500 and ~450 lines of `src/`, and every submodule
in Phases 7A and 7B has come in well above its estimate — `src/SpVAR.jl` is
**1,401 lines against the ~700 the plan projected**, almost exactly double.

The honest reading is that the per-item line estimates in these tables are
low by roughly a factor of two, consistently, and have been for several
passes. They are left as written rather than retrofitted, because the
pattern is more useful than a set of corrected numbers: an estimate made
before reading the primary source does not know how much of the paper will
turn out to be load-bearing.

**State the environment with the assertion count.** The figures above are
3,750 with Makie and Turing **loadable**, where testsets 25 and 26
contribute 11 and 24 assertions, and 3,715 with both **absent**, where each
reports a single broken marker instead of running. Both were measured in the
repository working tree, which matters — see the correction below. That
difference is 35 passing assertions — 3,715 + 11 + 24 = 3,750, the two
broken markers being replaced rather than added, which is why the arithmetic
is 35 and not 33. A count quoted without its environment can therefore look
like a regression when nothing has regressed.

**And state the entry point, which is easy to invert.** `Pkg.test()` is the
command that genuinely **excludes** the extensions, so it is the one
reporting the *lower* figure, 3,715 plus two broken markers — measured
separately on 1.13.3 and 1.10.13 and identical on both. Running
`test/runtests.jl` directly from `test/` leaves `@v1.13` on `LOAD_PATH`,
the extensions load, and the total is 3,750 with no broken entries. The
intuition that the more isolated-sounding command gives the higher number
is backwards.

**State the Julia version too, and re-read it rather than carrying it
forward.** The figures above were measured on **1.13.3** and **1.10.13**.
Earlier revisions of this file said 1.13.2 and 1.10.12, which was true when
written; `juliaup` had since moved both channels, so the labels were stale
while the counts they qualified were not. Cheap to check with
`julia --version` and `julia +1.10 --version`, and a version label nobody
re-reads is exactly the kind of figure that drifts into being wrong.

**Corrected on 2026-10-31: there is no cross-version assertion difference.**
Earlier revisions of this section described a genuine six-assertion gap
between Julia 1.13.2 and 1.10.12. Measurement says otherwise. Running the
suite verbose on each version and diffing the per-testset counts — the
diagnostic this section has always prescribed — every one of the 51
testsets reports an **identical** count on 1.13.2 and 1.10.12; the diff is
timings only. The extension testsets are the sole source of count
variation.

Where the old six-assertion figure came from is now settled, and it was
never the Julia version. The diagnosis was done on 2026-10-31, when the
suite stood at 2,395 assertions without the extensions; those are the
figures quoted here, since changing them would destroy the arithmetic that
identified the cause. `Pkg.test()` in the repository working tree reported
**2,395**; the same command on a copy holding only `Project.toml`,
`Manifest.toml`, `src/`, `test/` and `ext/` reported **2,389**, on either
Julia version. Two candidate explanations were tested and **both refuted**
— not the version (per-testset counts are identical across 1.13.2 and
1.10.12), and not dependency resolution (copying the local `Manifest.toml`
across does not reproduce the 2,395). The cause is the **working tree**,
and the six assertions are named: testset 11b gates a block on

```julia
csv = joinpath(@__DIR__, "..", "notebooks", "data", "columbus.csv")
if isfile(csv)
```

and that block holds exactly six assertions — one `@test_logs` guarding
`fit_mgwr` against convergence warnings, and five `@test`s on the fit. So
testset 11b reports **12 in the repository and 6 in a `src`-and-`test`-only
copy**. `notebooks/data/columbus.csv` is tracked, so CI always has it, and
the extensions-absent figure CI sees is the in-repository one — 2,395 then,
3,715 now — never the copy's.

Two consequences for anyone measuring this suite. A scratch copy must
include `notebooks/data/` to be comparable with the repository at all. And
a verbose variant has to be run from `test/` **inside** the repository: run
the same script from a scratchpad and `@__DIR__` moves with it, the
`isfile` check fails, and the block skips silently — which is exactly how
the earlier 2,424 figure in this section came to be six too low.

Note also that `--project=<dir>` does not hide the extensions: `@v1.13`
stays on `LOAD_PATH`, so Makie and Turing load from the default environment
anyway. `Pkg.test()` is the only entry point that genuinely excludes them.

† The five rows marked † count the extension testsets, as does the †† row
that ends the table; the Phase 6C row above them does not. So the step from 2,036 to 2,233 is **not** testset 34's
size: that testset is 162 assertions, and the other 35 is testsets 25 and 26
running rather than reporting a marker. The next two steps **are** exactly
the new testset's size: 2,233 to 2,430 is testset 35's 197, and 2,430 to
3,051 is testset 36's 621.

The last step is **not**, and the discrepancy is the point of keeping the
check. 3,051 to 3,217 is 166, while testset 37 is 163. The other **3** were
added to testset 21, which pins the chi-squared survival function to a
single shared definition: two more module-identity assertions and one that
the removed duplicate is really gone. So the rule is that a step equals the
new testset's size *unless an existing testset also changed*, and when it
does not balance the right response is to find the three rather than to
adjust the total.

**The Phase 7B step balances, which is the useful outcome.** 3,217 to 3,415
is 198, and testsets 38 and 39 are 117 and 81 — 198 exactly.
**The SpVAR step balances too**, and it was checked at each of the three
times testset 40 grew. 3,380 to 3,683 without the extensions is 303, and
testset 40 is 303 — measured in isolation before each suite run, which is
how the `const`-in-`@testset` syntax error was caught before it could take
the whole suite down. The intermediate figures were 255 and 284, as subsets
`40a2` (`binary = false`) and `40n` (`p > 1`, `rho_bounds`) were added on
finding those options untested. So nothing elsewhere was quietly adjusted,
even though this pass touched `src/SpatialWeights.jl` to add
`order_weights`.

**The Pfeifer & Deutsch step balances as well**, and this one is the
tightest of the three because the pass changed shipped code rather than
only adding to it. 3,683 to 3,715 without the extensions is 32, and testset
`39j` holds 32 assertions — 30 written lines, one of which sits in a
two-iteration loop over both weight conventions. That matters here: the
pass moved `fit_star`'s weight matrices onto the model and rewrote
`_star_A`'s signature, so a step that balances is the evidence that
testsets 39a–i, which exercise the old path, still run and still count the
same. So the two new
rows are purely additive and nothing elsewhere in the suite was quietly
adjusted. That is worth one line of arithmetic: this pass touched two new
submodules, the umbrella, `docs/make.jl` and two new doc pages, and a step
that balances is the cheapest available evidence it left the other 53
testsets alone.

Testset 38 was **116 at first measurement and is 117** because the
cross-version run below forced an assertion to be repaired rather than
re-counted. Recording that is the point: the first figure was not wrong
when taken, and resisting the urge to quietly carry it forward is what
keeps the step arithmetic worth doing at all.

**The same check caught a defect across versions, not just across rows.**
The networks row first measured 3,215 and 3,180 on 1.13.2 against 3,179 on
1.10.12 — one assertion fewer, with no failure anywhere. Testset `37e`
asserted Corollary 1 behind an `if diameter >= 3 && connected` guard over
an Erdos-Renyi draw, and `MersenneTwister(77)` does not give the same edge
set on both versions (326 edges against 288); at 288 the draw was
disconnected, the guard failed, and the assertion did not run. A guarded
assertion that fails its guard does not fail, it disappears. It is now an
unconditional implication over deterministic structures, `37e` reports 12
on both versions, and the totals above are the post-fix ones. See
`docs/AUDIT.md`. **An assertion count that differs by environment is a
finding until its mechanism is known** — the extensions difference above is
documented as two figures because there the mechanism *is* known; this one
was coverage going missing.

That the steps come out exact is itself the check, and it has already caught
one error. An earlier revision of this table recorded 2,430 as 2,424 and so
read the first step as 191; the missing six were the Columbus block
described above, skipped because the measuring script ran from outside
`test/`. Comparing two rows of this table means checking they were measured
the same way first.

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
- Ando, T., & Lu, L. (2019). A spatial panel quantile model with unobserved heterogeneity. SSRN 3516306 (working paper, 19 Dec 2019). **Read in part** (model, assumptions, estimation, statement of Theorem 1). An earlier two-author version of the Ando, Li & Lu paper, which is not held.
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
- Anselin, L. (1990). Spatial dependence and spatial structural instability in applied regression analysis. *Journal of Regional Science*, 30(2), 185–207.
- Anselin, L. (2001). Rao's score test in spatial econometrics. *Journal of Statistical Planning and Inference*, 97(1), 113–139.
- Anselin, L., Le Gallo, J., & Jayet, H. (2008). Spatial panel econometrics. In *The Econometrics of Panel Data* (pp. 625–660). Springer.
- Anselin, L., & Smirnov, O. (1996). Efficient algorithms for constructing proper higher order spatial lag operators. *Journal of Regional Science*, 36(1), 67–89.
- Assunção, R. M., Neves, M. C., Câmara, G., & Da Costa Freitas, C. (2006). Efficient regionalization techniques for socio-economic geographical units using minimum spanning trees. *International Journal of Geographical Information Science*, 20(7), 797–811.
- Aune, E., Simpson, D. P., & Eidsvik, J. (2014). Parameter estimation in high dimensional Gaussian distributions. *Statistics and Computing*, 24(2), 247–263.
- Bai, J. (2009). Panel data models with interactive fixed effects. *Econometrica*, 77(4), 1229–1279.
- Bai, J., & Chen, J. C. (2026). Fixed-T dynamic spatial panel model with common shocks. arXiv:2608.22499v1 [econ.EM]. **Preprint, read in part**: model, §3 theory as stated, algorithm, simulations, application and Appendix H; Appendix D (the score rearrangement and the covariance estimator) read 2026-11-14; the remaining proofs not. Held in `ExtRef/`.
- Bailey, N., Kapetanios, G., & Pesaran, M. H. (2016). Exponent of cross-sectional dependence: Estimation and inference. *Journal of Applied Econometrics*, 31(6), 929–960.
- Baltagi, B. H., Song, S. H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123–150.
- Barry, R. P., & Pace, R. K. (1999). Monte Carlo estimates of the log determinant of large sparse matrices. *Linear Algebra and its Applications*, 289(1–3), 41–54. **Verified, not read**: the local scan carries a degraded OCR layer, so its equations must be read from the page images before it is cited for a specification.
- Bera, A. K., & Yoon, M. J. (1993). Specification testing with locally misspecified alternatives. *Econometric Theory*, 9(4), 649–658.
- Bivand, R. (2010). Computing the Jacobian in spatial models: An applied survey. *NHH Department of Economics Discussion Paper* SAM 20/2010. SSRN 1680467. **Verified and read** for the Chebyshev form, the Walde et al. correction and the structure of its six test graphs. The author maintains R's `spdep`, so this is the same question answered in the other ecosystem.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental shift-share research designs. *Review of Economic Studies*, 89(1), 181–213.
- Bramoullé, Y., Djebbari, H., & Fortin, B. (2009). Identification of peer effects through social networks. *Journal of Econometrics*, 150(1), 41–55.
- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
- Burridge, P. (1981). Testing for a common factor in a spatial autoregression model. *Environment and Planning A*, 13(7), 795–800.
- Chagas, A. L. S., Azzoni, C. R., & Almeida, A. N. (2016). A spatial difference-in-differences analysis of the impact of sugarcane production on respiratory diseases. *Regional Science and Urban Economics*, 59, 24–36. **Verified and read** 2026-11-05. Its eq. (7) splits exposure into two channels with free coefficients, `W_{T,T}b₁` and `W_{NT,T}b₂`, and its footnote 4 records that Delgado & Florax consider only the restricted case. **`fit_spatial_did` implements Delgado & Florax, not this**; see `docs/AUDIT.md`.
- Chun, Y., Griffith, D. A., Lee, M., & Sinha, P. (2016). Eigenvector selection with stepwise regression techniques to construct eigenvector spatial filters. *Journal of Geographical Systems*, 18(1), 67–85.
- Debarsy, N., & Le Gallo, J. (2024). Identification of spatial spillovers: Do's and dont's. LEM working paper DT2024-03 (text dated 10 February 2025), Université de Lille. **Read in part** (§§1–3, 4.3.1, 5–6). The later journal version is not held.
- Debarsy, N., & LeSage, J. P. (2020). Bayesian model averaging for spatial autoregressive models based on convex combinations of different types of connectivity matrices. *Journal of Business & Economic Statistics*, advance online publication, doi:10.1080/07350015.2020.1840993. **Read in part** in the accepted author version.
- Delgado, M. S., & Florax, R. J. G. M. (2015). Difference-in-differences techniques for spatial data: Local autocorrelation and spatial interaction. *Economics Letters*, 137, 123–126. **Verified and read** 2026-11-05, from Tinbergen Institute Discussion Paper 2015-091/VIII. Eq. (6) is the design `fit_spatial_did` fits; the restriction `β₄ = λβ₃` that their derivation carries is not imposed, so their case is nested.
- Di Giacinto, V. (2010). On vector autoregressive modeling in space and time. *Journal of Geographical Systems*, 12(2), 125–154. doi:10.1007/s10109-010-0116-6. **Verified and read in full** from Banca d'Italia *Temi di discussione* 746, which is freely available and carries the same text.
- Elhorst, J. P. (2010). Applied spatial econometrics: Raising the bar. *Spatial Economic Analysis*, 5(1), 9–28.
- Ertur, C., & Musolesi, A. (2017). Weak and strong cross-sectional dependence: A panel data analysis of international technology diffusion. *Journal of Applied Econometrics*, 32(3), 477–503.
- Fülle, M. J., & Otto, P. (2024). Spatial GARCH models for unknown spatial locations — an application to financial stock returns. *Spatial Economic Analysis*, 19(1), 92–105. Published online 6 Sep 2023. https://doi.org/10.1080/17421772.2023.2237067. **Read** 2026-11-16.
- Geniaux, G. (2026). Flexible nonlinear spatial autoregressive models under spatial confounding: a gradient boosting approach with closed-form estimation. arXiv:2611.16592v1 [stat.ME]. **Preprint, not peer reviewed, read in full** (§§1–5 on 2026-11-13; Appendices A–D on 2026-11-18). Held in `ExtRef/`.
- Geniaux, G., & Martinetti, D. (2017). A new method for dealing with spatially varying coefficients. *Geographical Analysis*, 50(1), 28–50.
- Gibbons, S., & Overman, H. G. (2012). Mostly pointless spatial econometrics? *Journal of Regional Science*, 52(2), 172–191. **Read** in the LSE working-paper version, whose pagination differs.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik instruments: What, when, why, and how. *American Economic Review*, 110(8), 2586–2624.
- Griffith, D. A. (2003). *Spatial Autocorrelation and Spatial Filtering*. Springer.
- Halleck Vega, S., & Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55(3), 339–363.
- Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models. *Journal of the Royal Statistical Society B*, 55(4), 757–796.
- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators for spatial interaction in spatial autoregressive models. *Regional Science and Urban Economics*, 42(3), 446–458. doi:10.1016/j.regsciurbeco.2011.12.004. **Read** (§§2–3, Propositions 1, 2 and 5) 2026-11-18. Held in `ExtRef/`.
- Kelejian, H. H., & Piras, G. (2014). Estimation of spatial models with endogenous weighting matrices, and an application to a demand model for cigarettes. *Regional Science and Urban Economics*, 46, 140–149. **Read in part** in the accepted manuscript.
- Kelejian, H. H., & Prucha, I. R. (2010). Specification and estimation of spatial autoregressive models with autoregressive and heteroskedastic disturbances. *Journal of Econometrics*, 157(1), 53–67. doi:10.1016/j.jeconom.2009.10.025
- Klier, T., & McMillen, D. P. (2008). Clustering of auto supplier plants in the United States. *Journal of Business & Economic Statistics*, 26(4), 460–471.
- Koley, M., & Bera, A. (2024). To use, or not to use the spatial Durbin model? — that is the question. *Spatial Economic Analysis*, 19(1), 30–56.
- Kopczewska, K., & Elhorst, J. P. (2024). New developments in spatial econometric modelling. *Spatial Economic Analysis*, 19(1), 1–7.
- Kostov, P. (2009). A spatial quantile regression hedonic model of agricultural land prices. *Spatial Economic Analysis*, 4(1), 53–72.
- Krisztin, T., & Piribauer, P. (2021). A Bayesian approach for estimation of weight matrices in spatial autoregressive models. arXiv:2101.11938v2 (working paper; an earlier draft of the *Spatial Economic Analysis* article). **Read in part** from the TeX source.
- Krzyśko, M., Nijkamp, P., Ratajczak, W., Wołyński, W., & Wenerska, B. (2024). Spatio-temporal principal component analysis. *Spatial Economic Analysis*, 19(1), 8–29.
- Kubara, M., & Kopczewska, K. (2024). Akaike information criterion in choosing the optimal k-nearest neighbours of the spatial weight matrix. *Spatial Economic Analysis*, 19(1), 73–91.
- Lam, C., & Souza, P. C. L. (2020). Estimation and selection of spatial weight matrix in a spatial lag model. *Journal of Business & Economic Statistics*, 38(3), 693–710. https://doi.org/10.1080/07350015.2019.1569526. **Read in part** in the accepted manuscript (Warwick WRAP 112458).
- Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood estimators for spatial autoregressive models. *Econometrica*, 72(6), 1899–1925. https://doi.org/10.1111/j.1468-0262.2004.00558.x
- Lee, L. F., & Liu, X. (2010). Efficient GMM estimation of high order spatial autoregressive models with autoregressive disturbances. *Econometric Theory*, 26(1), 187–230.
- Lee, L.-F., & Yu, J. (2012). QML estimation of spatial dynamic panel data models with time varying spatial weights matrices. *Spatial Economic Analysis*, 7(1), 31–74.
- Leenders, R. T. A. J. (2002). Modeling social influence through network autocorrelation: constructing the weight matrix. *Social Networks*, 24(1), 21–47.
- LeSage, J. P. (2014). Spatial econometric panel data model specification: A Bayesian approach. *Spatial Statistics*, 9, 122–145.
- LeSage, J. P. (2015). Software for Bayesian cross section and panel spatial model comparison. *Journal of Geographical Systems*, 17(4), 297–310.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Li, K. (2022). Threshold spatial autoregressive model. MPRA Paper 113568 (working paper, 26 Jun 2022). **Read.** Not Li & Lin (2024).
- Li, L., & Yang, Z. (2021). Spatial dynamic panel data models with correlated random effects. *Journal of Econometrics*, 221(2), 424–454. doi:10.1016/j.jeconom.2020.05.016. **Verified, read in part** (2026-11-13), from an SMU-repository author version held in `ExtRef/`: §§1–6 read; the proofs (Appendices B–C) and the online Supplementary Appendix not. Journal pagination is not in the file.
- Liang, X., Gao, J., & Gong, X. (2022). Semiparametric spatial autoregressive panel data model with fixed effects and time-varying coefficients. *Journal of Business & Economic Statistics*, 40(4), 1784–1802. doi:10.1080/07350015.2021.1979564. **Verified and read**, from MPRA Paper 108497 — which is the same document as Monash Econometrics and Business Statistics Working Paper 05/21, both cover pages being in the one deposit, so the "two open versions" listed here previously were one. The model is eq (2.1): the **time-varying coefficients are the regressor coefficients `β₀(τ)`, while `ρ₀` is a scalar constant.** This reference therefore supports a time-varying-`β` row and **not** the time-varying-`ρ` row it was briefly attached to; see `docs/AUDIT.md`.
- Liu, S. F., & Yang, Z. (2015). Modified QML estimation of spatial autoregressive models with unknown heteroskedasticity and nonnormality. *Regional Science and Urban Economics*. doi:10.1016/j.regsciurbeco.2015.02.003. **Read in part** (§3.1 and the assumptions) 2026-11-18. Held in `ExtRef/` (accepted manuscript).
- Liu, X., & Yang, Z. (2020). Variable selection in spatial autoregressive models. *Journal of Econometrics*, 218(1), 85–109.
- Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531–542.
- Murakami, D., & Griffith, D. A. (2015). Random effects specifications in eigenvector spatial filtering. *Journal of Geographical Systems*, 17(4), 311–331.
- Mutl, J., & Pfaffermayr, M. (2011). The Hausman test in a Cliff and Ord panel model. *Econometrics Journal*, 14(1), 48–76.
- Ord, J. K. (1975). Estimation methods for models of spatial interaction. *Journal of the American Statistical Association*, 70(349), 120–126. **Verified, and p. 125 read**: a row-standardised *symmetric* adjacency is similar to a symmetric matrix and so has real eigenvalues. The premise is that the unstandardised adjacency is already symmetric, which `knn` weights do not satisfy. Confirmed numerically at 1.9e-14; Appendix C also gives closed-form lattice eigenvalues, used as an exact benchmark baseline.
- Pace, R. K., & LeSage, J. P. (2004). Chebyshev approximation of log-determinants of spatial weight matrices. *Computational Statistics & Data Analysis*, 45(2), 179–196. doi:10.1016/S0167-9473(02)00321-3. **The source for `chebyshev_logdet`, and absent from this list until 2026-11-05.** Verified, and **read in part**: the introduction and §§2.1–2.2 were read in the paper's own text; §2.3, the Chebyshev form, and §2.4, the bounds, were not. Received 2001, revised 2002, published in the 2004 volume, which is why the article carries a 2002 copyright line.
- Pesaran, M. H. (2006). Estimation and inference in large heterogeneous panels with a multifactor error structure. *Econometrica*, 74(4), 967–1012.
- Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large panels. *Econometric Reviews*, 34(6–10), 1089–1117.
- Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative procedure for space-time modeling. *Technometrics*, 22(1), 35–47. **Verified**; read in full from the authors' reprint, NCJRS 60692, after tandfonline and JSTOR both refused automated access. Their spatial lag operator is defined on order-`k` neighbourhoods, not powers of `W` — see `docs/AUDIT.md`.
- Qu, X., & Lee, L. F. (2015). Estimating a spatial autoregressive model with an endogenous spatial weight matrix. *Journal of Econometrics*, 184(2), 209–232. **Read in part** (§§2–3).
- Santi, F., Dickson, M. M., Espa, G., Taufer, E., & Mazzitelli, A. (2021). Handling spatial dependence under unknown unit locations. *Spatial Economic Analysis*, 16(2), 194–216. Published online 3 Jul 2020. https://doi.org/10.1080/17421772.2020.1769171. **Read** 2026-11-16 (main text; the online supplement with the proofs of eqs 7–8, 10, 11 and 13 is not held).
- Sato, T., & Matsuda, Y. (2021). Spatial extension of generalized autoregressive conditional heteroskedasticity models. *Spatial Economic Analysis*, 16(2), 148–160. Published online 30 Mar 2020. https://doi.org/10.1080/17421772.2020.1742929. **Read** (main text, and the online supplement of proofs, which does not contain the explicit `Ω` the main text promises).
- Shi, W., & Lee, L. F. (2017). Spatial dynamic panel data models with interactive fixed effects. *Journal of Econometrics*, 197(2), 323–347.
- Smirnov, O. A. (2022). A closed-form consistent estimator for linear models with spatial dependence. *Geographical Analysis*, 54(1), 3–31. doi:10.1111/gean.12268 (online 28 Dec 2020, hence "2020" in Geniaux). **Verified and read** (2026-11-13) from the Early View version (paginated 1–29) held in `ExtRef/`; the DOI and title are confirmed by Scite and the issue pages come from library catalogues. Main text read in full; the cells of Tables 2–7 are scrambled by text extraction and were read through the prose that reports them. Geniaux's two uses of it check out: the estimator and the conservative variance (its Tables 3 and 6: the estimated variance exceeds the MSE by 10–20% at moderate `ρ` and by a factor of 2–3 at strong `ρ`).
- Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of partially linear spatial autoregressive models. *Journal of Econometrics*, 157(1), 18–33. doi:10.1016/j.jeconom.2009.10.033. **Verified, read in part** (2026-11-13), from the submitted manuscript of 23 Aug 2007 held in `ExtRef/`. Read: §1, §2 (model and the two-step profile QMLE), §3 (Assumptions 1–8), the statements of Theorems 4.1–4.4 and 5.1–5.4 as far as text extraction preserves them, §6 (simulations) and §7. Not read: the Appendix proofs. Its abstract carries the same content as the published one, in more detail. Because this is the pre-review manuscript, equation and theorem numbers may not match the article. In the simulations the asymptotic standard error of `m̂(0)` (about 0.08) is roughly half the empirical one (0.13–0.22) in the Table 3 rows that extraction keeps legible, so check the page images before relying on pointwise bands.
- Sutton, J., & Sutton, J. (2024). A road map to capture the spatial dependence underlying regions' economic resilience. *Spatial Economic Analysis*, 19(1), 57–72.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57(2), 307–333.
- Walde, J., Larch, M., & Tappeiner, G. (2008). Performance contest between MLE and GMM for huge spatial autoregressive models. *Journal of Statistical Computation and Simulation*, 78(2), 151–166. **Cited only as a corrected result, and not used.** It advises against the Chebyshev approximation on a test at `q = 2`; Bivand (2010) shows `q = 5` performs well and reports that a correction was to be submitted.
- Wang, C., Kockelman, K. M., & Wang, X. (2013). The spatial dimensions of land use change dynamics in the Austin, Texas Metropolitan Area. *Transportation Research Record*, 2397(1), 131–142.
- Yang, C. F. (2021). Common factors and spatial dependence: An application to US house prices. *Econometric Reviews*, 40(1), 14–50.
- Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W., & Wolf, L. J. (2020). Inference in multiscale geographically weighted regression. *Geographical Analysis*, 52(1), 87–106.

**A chapter attribution was corrected.** Item #11's multiplex row cited
LeSage & Pace (2009, Ch. 9). The book is real and correctly listed above,
but its Chapter 9 is *Matrix Exponential Spatial Models* — checked on
2026-11-01 against the Library of Congress CIP record for the 2009 CRC
edition, which gives the full contents — and the book has no multiplex
chapter. The row now cites Lee & Liu (2010), *Econometric Theory* 26(1),
187–230, verified the same day, which is the actual econometrics of
`y = Σⱼ ρⱼ Wⱼ y + Xβ + ε`. Worth distinguishing from the Cerqua case below: a
real reference pointed at the wrong chapter, not a reference that does not
exist. The three citations item #11 already carried — Leenders (2002),
Manski (1993) and Bramoullé, Djebbari & Fortin (2009) — were each checked
against the primary source and all three hold; BDF's full text was read and
every formula in `SpatialNetworks.jl` is quoted from it rather than
recalled.

**A methodological citation was missing.** Item #1's spatial CCE row cited
Pesaran (2006) and Ertur & Musolesi (2017). Both are real and both are
listed above, but Ertur & Musolesi is an *application* paper on
international technology diffusion: correct for the weak/strong
distinction, which is what it was doing in the row, and not the
methodological source for a spatial CCE *estimator*. That is Yang, C. F.
(2021), *Econometric Reviews* 40(1), 14–50, verified against the published
record on 2026-11-04 and now cited. Ertur & Musolesi was not removed — it
was never a phantom. This is weaker still than the chapter error above,
and worth distinguishing for the same reason: a *missing* citation is
harder to notice than a wrong one, because nothing in the row looks
broken.

**Citations to verify before use — one has failed, one has passed.**
Goldsmith-Pinkham, Sorkin & Swift (2020) was added to the list above on
2026-10-31 for Rotemberg weights, after being checked against the published
record and against the article's own header line (*American Economic
Review* 2020, 110(8), 2586–2624, doi:10.1257/aer.20181047). It holds, which
is worth recording alongside the failure: the practice catches bad
citations without condemning good ones. Cerqua et
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
