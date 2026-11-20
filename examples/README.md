# Examples

Thirty-three worked examples, one per topic. Each is a self-contained script
that simulates its own data, so none needs a download, and each prints its
results to the terminal.

## Running them

From the repository root:

```bash
julia --project=. examples/01_cross_section.jl
```

Every example except 18 and 19 needs only the package and its dependencies.
Those two use the package extensions, whose trigger packages are not
installed with the package:

| Example | Needs | Extension |
|---|---|---|
| 18 | `CairoMakie` | `SpatialEconometricsMakieExt` |
| 19 | `Turing`, `Distributions` | `SpatialEconometricsTuringExt` |

Install them in your default environment (`@v1.x`), which the project falls
back to, so that the repository's `Project.toml` stays unchanged:

```bash
julia -e 'using Pkg; Pkg.add(["CairoMakie", "Turing", "Distributions"])'
```

Example 15 writes GeoDa weight files to a temporary directory and reads them
back, and example 18 saves its plots as PNGs in one. No example writes to the
repository.

## The examples

### Cross-section

| # | Script | Topic |
|---|---|---|
| 01 | [`01_cross_section.jl`](01_cross_section.jl) | SAR and SDM by concentrated ML, direct and indirect effects |
| 03 | [`03_sem_diagnostics.jl`](03_sem_diagnostics.jl) | The spatial error model and the Moran / LM diagnostic workflow |
| 09 | [`09_higher_order.jl`](09_higher_order.jl) | SLX, SAC and SDEM |
| 10 | [`10_spatial_regimes.jl`](10_spatial_regimes.jl) | Coefficients that vary by regime, SKATER regionalisation |
| 11 | [`11_discrete_choice.jl`](11_discrete_choice.jl) | Spatial probit by linearised GMM and Gibbs sampling |
| 12 | [`12_spatial_filtering.jl`](12_spatial_filtering.jl) | Eigenvector spatial filtering, ESF and RE-ESF |
| 04 | [`04_gwr_mgwr.jl`](04_gwr_mgwr.jl) | Geographically weighted regression, GWR and MGWR |
| 05 | [`05_gravity_models.jl`](05_gravity_models.jl) | Gravity and flow models, PPML and negative binomial |
| 30 | [`30_semiparametric.jl`](30_semiparametric.jl) | Partially linear SAR: a smoother that can absorb `ρ` |

### Panels

| # | Script | Topic |
|---|---|---|
| 02 | [`02_spatial_panels.jl`](02_spatial_panels.jl) | Static and dynamic spatial panels |
| 06 | [`06_gs2sls_gmm.jl`](06_gs2sls_gmm.jl) | GS2SLS (SARAR) with endogenous regressors, and system GMM |
| 17 | [`17_panel_diagnostics.jl`](17_panel_diagnostics.jl) | Spatial Hausman, Chow and joint panel LM tests |
| 27 | [`27_common_factors.jl`](27_common_factors.jl) | Weak versus strong cross-sectional dependence: CD, `α`, SP-CCE |
| 31 | [`31_cre_dynamic_panel.jl`](31_cre_dynamic_panel.jl) | Short-`T` dynamic panel with correlated random effects (Li & Yang) |
| 32 | [`32_common_shocks_short_panel.jl`](32_common_shocks_short_panel.jl) | Short-`T` dynamic panel with common shocks: which SERL order `L` (Bai & Chen) |
| 33 | [`33_spatial_log_garch.jl`](33_spatial_log_garch.jl) | Spatial log-GARCH: sandwich SEs, `λ` against `λ + ρ`, a covariate-distance `W` (Sato & Matsuda) |
| 34 | [`34_root_rho.jl`](34_root_rho.jl) | `root_rho`: `ρ` with a large basis for an unknown surface, the test-inversion set, and the plain closed form's drift as `K` grows (Jin & Lee) |

### Space and time

| # | Script | Topic |
|---|---|---|
| 22 | [`22_spatiotemporal_pca.jl`](22_spatiotemporal_pca.jl) | Spatio-temporal principal components |
| 28 | [`28_spatio_temporal.jl`](28_spatio_temporal.jl) | STAR, impulse responses and forecasting |
| 29 | [`29_spvar.jl`](29_spvar.jl) | Spatial VAR, and why the causal ordering is the answer |

### Causal inference and networks

| # | Script | Topic |
|---|---|---|
| 23 | [`23_spatial_causal.jl`](23_spatial_causal.jl) | Spatial difference-in-differences with spillovers |
| 24 | [`24_synthetic_control.jl`](24_synthetic_control.jl) | Spatial synthetic control for one treated region |
| 25 | [`25_shift_share.jl`](25_shift_share.jl) | Shift-share (Bartik) diagnostics |
| 26 | [`26_networks.jl`](26_networks.jl) | Peer effects on a network: identification, 2SLS, multiplex SAR |

### Specification and inference

| # | Script | Topic |
|---|---|---|
| 16 | [`16_model_selection.jl`](16_model_selection.jl) | AIC/BIC, LR, Vuong, Wald and common-factor tests |
| 20 | [`20_specification_search.jl`](20_specification_search.jl) | Which spatial terms belong in the model |
| 21 | [`21_bayesian_specification.jl`](21_bayesian_specification.jl) | Global or local spillovers: SDM against SDEM by marginal likelihood |
| 08 | [`08_hac_robust.jl`](08_hac_robust.jl) | Driscoll-Kraay and Conley HAC standard errors |
| 13 | [`13_simulation_power.jl`](13_simulation_power.jl) | Simulation, Monte Carlo and power analysis |

### Tools

| # | Script | Topic |
|---|---|---|
| 15 | [`15_weight_matrices.jl`](15_weight_matrices.jl) | Building weight matrices, standardisation, GeoDa GAL/GWT I/O |
| 07 | [`07_gras_calibration.jl`](07_gras_calibration.jl) | GRAS and cross-entropy SAM / MRIO calibration |
| 14 | [`14_output_tables.jl`](14_output_tables.jl) | Coefficient tables as LaTeX, CSV and Markdown |
| 18 | [`18_visualization.jl`](18_visualization.jl) | Plots through the Makie extension |
| 19 | [`19_bayesian.jl`](19_bayesian.jl) | Bayesian spatial models through the Turing extension |

## Where to read more

Each example matches a page of the
[manual](https://randsley.github.io/SpatialEconometrics.jl/dev/). The
page explains the model and the options; the example shows a run. The Monte
Carlo studies behind the documented accuracy claims are in
[`../validation/`](../validation/), not here.
