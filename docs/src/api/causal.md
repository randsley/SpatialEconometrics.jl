# Spatial Causal Inference

Treatment effects when one unit's treatment reaches its neighbours:
exposure mapping, difference-in-differences decomposed into a direct and
a spillover effect, synthetic control with a spillover-screened donor pool,
and shift-share designs, where the dependence runs through shared industry
exposure rather than through geography.

```@meta
CurrentModule = SpatialEconometrics
```

```@docs
ExposureMap
exposure
ring_exposure
InterferenceReport
check_interference
SpatialDiDResult
fit_spatial_did
SyntheticControlResult
fit_synthetic_control
donor_weights
PlaceboTest
placebo_test
SpilloverGaps
spillover_gaps
```

## Shift-share designs

```@docs
ShiftShareDesign
shift_share_design
ShiftShareFit
fit_shift_share
ExposureConcentration
exposure_concentration
ShockAggregates
shock_aggregate
BalanceTest
balance_test
RotembergWeights
rotemberg_weights
ShiftShareMoran
shift_share_moran
```
