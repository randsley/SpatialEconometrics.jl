# Dynamic Panel GMM

Arellano–Bond and Blundell–Bond estimation for dynamic panels: a lagged
dependent variable with short `T`, where the within estimator is biased
downward by order `1/T` (Nickell bias) and lagged levels or differences serve
as instruments instead.

## Specification

```math
y_{it} = \gamma y_{i,t-1} + X_{it} \beta + \mu_i + \varepsilon_{it}
```

## Variants

| Variant | Instruments | Notes |
|---|---|---|
| `DifferenceGMM()` | Lagged levels for the differenced equation | Weak when `gamma` approaches 1 |
| `SystemGMM()` | Difference equation stacked with the levels equation | More efficient; needs the extra initial-conditions assumption |

## Estimation

```julia
# Y_mat: N×T matrix, X_ten: N×T×K array
m = fit_dynamic_gmm(SystemGMM(), Y_mat, X_ten;
    max_lag = 4, collapsed = false, varnames = ["x1"])

m_diff = fit_dynamic_gmm(DifferenceGMM(), Y_mat, X_ten; max_lag = 3)
```

`max_lag` caps how deep the instrument matrix reaches. Left uncapped, the
instrument count grows with `T²` and the Hansen test loses power against an
over-fitted model; `collapsed = true` collapses the instrument set, trading a
little efficiency for a far smaller instrument count.

## Reading the diagnostics

```julia
m.hansen_j, m.hansen_df   # over-identification: a *small* p-value is bad
m.ar1_z, m.ar2_z          # serial correlation in the differenced residuals
```

AR(1) in differences is expected and not a problem. **AR(2) is the one to
watch**: it indicates the lag structure is misspecified and that the
instruments are not valid. `m.vcov_windmeijer` is the covariance matrix to
report for the two-step estimator; the uncorrected two-step standard errors
are biased downward in short panels.

See [Dynamic Spatial Panels](dynamic_panel.md) for the case that also carries
a spatial lag.
