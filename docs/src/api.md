# API Reference

## Estimation Functions

| Function | Model | Module |
|---|---|---|
| `fit_spatial(variant, W, X, y)` | Cross-section SAR/SDM | `CrossSection` |
| `fit_spatial_panel(variant, fe, W, X, y, N, T)` | Static spatial panel | `StaticPanel` |
| `fit_dynamic_spatial_panel(variant, fe, W, X, y, N, T)` | Dynamic spatial panel | `DynamicPanel` |
| `fit_gs2sls_panel(fe, W, X_exog, X_endog, Z, y, N, T)` | GS2SLS with IV | `GS2SLSPanel` |
| `fit_dynamic_gmm(variant, Y, X)` | System/Difference GMM | `SystemGMMMod` |
| `fit_spatial_error(W, X, y)` | Spatial Error Model | `SEMMod` |
| `fit_gwr(X, y, coords)` | Geographically Weighted Regression | `GWRMod` |
| `fit_mgwr(X, y, coords)` | Multiscale GWR | `GWRMod` |
| `fit_gravity(variant, flows, X, orig, dest)` | Gravity / flow models | `GravityMod` |

## Weight Matrix Construction

```@docs
knn_weights
distance_weights
band_weights
block_diagonal_weights
row_standardize!
spectral_standardize!
minmax_standardize!
spatial_connectivity
```

## Diagnostics

| Function | Description |
|---|---|
| `moran_global(y, W)` | Global Moran's I with permutation inference |
| `moran_local(y, W)` | LISA — local Moran's I with cluster classification |
| `geary_c(y, W)` | Geary's C statistic |
| `lm_tests(y, X, W)` | All four LM tests (lag, error, robust lag, robust error) |
| `lm_test_lag(y, X, W)` | LM test for spatial lag |
| `lm_test_error(y, X, W)` | LM test for spatial error |
| `lm_test_robust_lag(y, X, W)` | Robust LM test for spatial lag |
| `lm_test_robust_error(y, X, W)` | Robust LM test for spatial error |

## Panel Diagnostics

```@docs
spatial_hausman_test
spatial_chow_test
panel_lm_tests
panel_moran
```

## Model Selection

```@docs
spatial_aic
spatial_bic
lr_test
vuong_test
common_factor_test
wald_test
model_comparison_table
```

## I/O

```@docs
read_gal
write_gal
read_gwt
write_gwt
```

## Higher-Order Models

```@docs
fit_slx
fit_sac
fit_sdem
```

## Regime Models

```@docs
fit_spatial_regime
skater_regimes
regime_chow_test
```

## Simulation

```@docs
simulate_spatial
monte_carlo_spatial
power_analysis
```

## Output & Tables

```@docs
coeftable
summary_stats
to_latex
to_csv
to_markdown
regression_table
```

## Grid Weights

| Function | Description |
|---|---|
| `make_rook_weights(L)` | L×L grid, 4-connected rook contiguity |
| `make_queen_weights(L)` | L×L grid, 8-connected queen contiguity |

## Robust Inference

| Function | Description |
|---|---|
| `robust_vcov(m, spec)` | Robust variance-covariance matrix |
| `robust_coeftable(m, spec)` | Coefficient table with robust SEs |
| `spatial_effects(m)` | Direct/indirect/total spatial effects |
| `dynamic_spatial_effects(m)` | Dynamic spatial effects (short/long run) |

## Calibration

| Function | Description |
|---|---|
| `balance_gras(X0, u, v)` | GRAS matrix balancing |
| `balance_gras_full(X0, u, v)` | GRAS with full diagnostics |
| `balance_generalized_ce(X0, u, v)` | Generalized cross-entropy balancing |
| `sam_leontief_multipliers(SAM)` | Leontief multipliers from a SAM |
