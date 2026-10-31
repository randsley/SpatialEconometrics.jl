# Spatial Causal Inference

## Overview

Difference-in-differences assumes that a unit's outcome responds to its own
treatment and nothing else — SUTVA. Geographic spillovers violate that by
construction, which is the premise of this whole package. When a policy
reaches a unit's neighbours, two things go wrong at once in a conventional
DiD: the spillover is pushed into the error term, where it is correlated
with treatment by construction, and the control group is no longer
untreated. Both biases land on the estimated treatment effect.

The estimator here puts the spillover in the regression instead, where it
can be measured.

```math
y_{it} = \mu_i + \gamma_t + \tau \, D_i \text{post}_t + \delta \, E_i \text{post}_t + \varepsilon_{it}
```

with ``E = WD`` the **exposure**: the fraction of ``i``'s neighbours that
are treated. For two periods, first-differencing removes ``\mu_i`` exactly
and leaves an ordinary least squares problem, so ``\tau`` and ``\delta`` are
recovered without approximation.

## Exposure mapping

An exposure mapping is what makes a spillover estimable at all: it replaces
"treatment might spread somehow" with a named, finite summary of the
treatment around each unit.

```julia
em = exposure(D, W)                  # fraction of neighbours treated
em = exposure(D, W; order = 2)       # first and second rings, separately
em = exposure(D, W; standardize = false)   # a count rather than a fraction
```

Rings read a spillover at a distance where mechanical displacement is
implausible, which is often the point of the exercise:

```julia
near = ring_exposure(D, coords, 0.0, 10.0)
far  = ring_exposure(D, coords, 10.0, 25.0)
```

Every spillover estimate is relative to the mapping that produced it. A
``\delta`` estimated on five nearest neighbours answers a different question
from one estimated on a 50 km band, and neither is "the" spillover — which
is why [`ExposureMap`](@ref) is carried in the result rather than discarded
after the fit.

## Check what the design can identify, before fitting

``\delta`` is identified by controls that **differ** in exposure. If every
control has a treated neighbour there is no unexposed baseline and
``\delta`` is absorbed by the constant; if no control has one, ``\delta``
cannot be separated from the direct effect at all. This is a property of the
design and the weight matrix alone, so it needs no outcome and should be
checked first:

```julia
rep = check_interference(em)
rep.identified || @warn rep.message
rep.n_control_unexposed      # the units carrying the identification
```

A design that is outright rank deficient is refused by name when you try to
fit it, rather than returning a number that cannot be interpreted.

## Fitting

```julia
m = fit_spatial_did(y, D, W)                 # y is N x 2
m.att_direct          # effect on a treated unit of its own treatment
m.att_spill_control   # the policy's reach beyond its target
```

Four effects are reported, not three, because the spillover onto treated
units and the spillover onto controls are evaluated at different exposure
means and are not interchangeable:

| Field | Meaning |
|---|---|
| `att_direct` | effect on a treated unit of its own treatment |
| `att_spill_treated` | effect on a treated unit of its neighbours' treatment |
| `att_spill_control` | effect on a control unit of its neighbours' treatment |
| `att_total_treated` | `att_direct + att_spill_treated` |

`att_spill_treated` is part of what a naive DiD would report *as* the direct
effect. Each quantity is a fixed linear combination of the coefficient
vector, so its standard error is exact rather than approximated.

## Staggered rollout

Pass the full treatment path as an ``N \times T`` matrix and the estimator
switches to a two-way within transform:

```julia
m = fit_spatial_did(y, D_path, W)     # D_path[i, t] = 1 once i is treated
```

With homogeneous effects this is as exact as the two-period path. With
**heterogeneous** effects it is not, and the error is large: measured on
noiseless three-cohort rollouts it runs from 17% to 67%, and its sign is
not predictable. Spatially contiguous adoption cohorts — which is how
rollouts usually happen — are the worst case, because cohort identity is
then nearly collinear with the exposure column. The
[`fit_spatial_did`](@ref) docstring has the measured table. For a staggered
design with suspected heterogeneity, estimate cohorts separately and
aggregate deliberately.

## Standard errors

The default is cluster-robust on unit. That is a measured choice rather than
a conventional one, and it came out opposite to what was expected:

| Estimator | Coverage of a nominal 95% interval |
|---|---|
| `:cluster` (default) | 95.0% |
| `:ols` | 96.5% |
| `:conley`, cutoff 1.5 | 94.5% |
| `:conley`, cutoff 6.0 | 85.3% |

Conley's spatial HAC *degrades* as the cutoff widens, and it degrades
identically when there is no spatial correlation at all — so the decay is
the finite-sample cost of estimating more pair covariances, not the
correction doing its job.

The reason the spatial correction matters less here than one would expect is
the exposure column itself. Including ``WD`` means ``\tau`` is identified
off the contrast between a unit and its neighbours, and that contrast
differences out the component of the error they share — which is exactly the
spatially correlated part.

A spatial HAC remains the right instrument for a spatially correlated
regressor, and `:conley` is available for designs where the exposure column
does not absorb the dependence:

```julia
m = fit_spatial_did(y, D, W; vcov = :conley, coords = coords, dist_cutoff = 2.5)
```

If you use it, report more than one cutoff and prefer the narrow end. Full
figures, including the nine configurations the ordering was checked across,
are in the [`fit_spatial_did`](@ref) docstring.

## What is not here yet

Roadmap item #15 also lists spatial synthetic control and shift-share
exogeneity diagnostics. Neither is implemented. The diagnostics in
particular owe a null-distribution Monte Carlo before they can be believed,
under the standing rule in `docs/AUDIT.md`, and they are a different
estimator family from this one.

## References

- Delgado, M. & Florax, R. (2015). Difference-in-differences techniques for
  spatial spillovers. *Economics Letters*, 137.
- Chagas, A., Azzoni, C. & Almeida, A. (2016). A spatial
  difference-in-differences analysis of the impact of sugarcane production.
  *Regional Science and Urban Economics*, 59.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in
  treatment timing. *Journal of Econometrics*, 225(2).
- Cameron, A. C. & Miller, D. L. (2015). A practitioner's guide to
  cluster-robust inference. *Journal of Human Resources*, 50(2).
