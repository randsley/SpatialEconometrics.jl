# API Reference

## Estimation Functions

```@docs
fit_spatial
fit_spatial_panel
fit_dynamic_spatial_panel
fit_gs2sls_panel
fit_dynamic_gmm
fit_spatial_error
fit_gwr
fit_mgwr
fit_gravity
```

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

```@docs
moran_global
moran_local
geary_c
lm_tests
lm_test_lag
lm_test_error
lm_test_robust_lag
lm_test_robust_error
```

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
