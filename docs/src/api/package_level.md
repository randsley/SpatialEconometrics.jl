# Package-Level & Extensions

Functions defined on the package itself, including the eight
extension entry points whose methods live in `ext/`.

```@meta
CurrentModule = SpatialEconometrics
```

```@docs
bayesian_sar
bayesian_sdm
bayesian_sem
make_queen_weights
make_rook_weights
plot_coefficients
plot_gwr_coefficients
plot_local_moran
plot_moran_scatter
plot_spatial_effects
robust_coeftable
robust_vcov
spatial_effects
```

## Internal numerics

These two are exported by the `SpatialUtils` submodule but not re-exported
by the package, so they are reached as
`SpatialEconometrics.SpatialUtils.spatial_trace_powers`. They are the trace
machinery the asymptotic covariances and the log-determinant approximations
are built on, and they are listed here because they carry docstrings: the
docs build checks that every documented name a module exports reaches the
manual, so an undocumented-but-exported function cannot hide.

```@meta
CurrentModule = SpatialEconometrics.SpatialUtils
```

```@docs
g_traces
spatial_trace_powers
```
