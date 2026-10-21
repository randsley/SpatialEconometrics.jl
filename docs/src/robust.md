# Robust Inference

## Driscoll-Kraay Standard Errors

For spatial panels with cross-sectional and temporal dependence:

```julia
V_dk = robust_vcov(m_panel, DriscollKraayVCOV())
tbl = robust_coeftable(m_panel, DriscollKraayVCOV())
```

## Conley Spatial HAC

For cross-sectional or panel models with spatial correlation:

```julia
coords = hcat(lon, lat)
spec = ConleySpatialHAC(coords, bandwidth)

V_con = robust_vcov(m, spec)
tbl = robust_coeftable(m, spec)
```

## Supported Models

Both robust VCOV methods work with:
- `SpatialPanelModel` (from `fit_spatial_panel`)
- `DynamicSpatialPanelModel` (from `fit_dynamic_spatial_panel`)
- `SpatialRegressionModel` (cross-section, Conley only)

## Spatial Effects with Robust VCOV

```julia
effects = spatial_effects(m, DriscollKraayVCOV())
effects = spatial_effects(m, ConleySpatialHAC(coords, 3.0))
```
