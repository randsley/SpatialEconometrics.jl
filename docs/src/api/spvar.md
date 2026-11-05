# Spatial Vector Autoregression

API reference for `SpVAR`. See [Spatial VAR](../models/spvar.md) for the
discussion, and [Weight Matrices](../weights.md) for `order_weights`, which
builds the `W⁽ˡ⁾` this module needs.

## Module

```@docs
SpatialEconometrics.SpVAR
```

## The order condition

```@docs
OrderConditionResult
spvar_order_condition
```

## Estimation

```@docs
SpVARModel
fit_spvar
```

## Space-time impulse responses

```@docs
spvar_irf
STIRResult
space_time_stir
```

## Forecasting

```@docs
spvar_forecast
```

## The nested panel VAR and the test for spatial interaction

```@docs
NestedPanelVARModel
fit_nested_panel_var
lr_spatial_interaction
```
