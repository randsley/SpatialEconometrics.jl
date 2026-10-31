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

## Synthetic control

Difference-in-differences needs a control group and a common pre-trend.
The comparative case study has neither: one treated region, a long
pre-period, and a few dozen untreated regions that are not
interchangeable with it. Synthetic control builds the comparison instead
of assuming it — a weighted average of untreated units chosen to track
the treated unit before the intervention, used as its counterfactual
after.

```julia
scm = fit_synthetic_control(Y, 41, 18; W = W, coords = coords)
scm.att                                    # mean post-period gap
pt  = placebo_test(scm, Y; W = W, coords = coords)
sg  = spillover_gaps(scm, Y; W = W, coords = coords)
```

For treated unit `i`, the weights solve

```
minimise   (x_i - X0 w)' (x_i - X0 w) + λ Σ_j d_ij w_j
subject to w ≥ 0,  Σ_j w_j = 1
```

a convex program over the unit simplex. It is solved by accelerated
projected gradient with the exact Lipschitz step, and the returned point
carries its own KKT certificate (`kkt_gap`, `kkt_violation`) rather than
an assurance — unconverged weights are otherwise indistinguishable from
converged ones.

### The spatial problem: a contaminated donor pool

If the intervention spills over, the treated unit's neighbours are not
untreated. They are also the units a synthetic control is most likely to
select, because they resemble the treated unit most. The spillover then
sits inside the comparison and the estimate is pulled toward zero.

This is not a statistical tendency to be hoped against — it is an
identity. The spillover falls in the post-treatment periods only, so the
weights, fitted on pre-treatment data, are untouched by it. The bias is
therefore exactly the weight placed on contaminated donors times their
exposure times the spillover:

```
att(contaminated pool) = att(clean pool) − Σ_{j ∈ neighbours} w_j · E_j · spillover
```

Testset 35f pins that identity to `1e-12` at four spillover sizes.

`donor_pool = :exclude_neighbours`, the default, drops every untreated
unit adjacent to a treated one under `W` — reusing [`exposure`](@ref) to
find them — and records which units went. `donor_pool = :all` is the
conventional choice and is wrong whenever the spillover is real.

**The trap is that the contaminated fit looks better.** In
`examples/24_synthetic_control.jl` the all-donor fit puts 90.1% of its
weight on regions the policy had reached, achieves a pre-period RMSPE of
0.045 against the screened fit's 0.097, and is wrong by −0.433 against a
true effect of 3.0 — where the screened fit is wrong by 0.005. A good
pre-period fit is a necessary check, never a sufficient one, and it
cannot detect this problem at all, because the contamination is invisible
before treatment.

### The penalty

Unpenalised weights are frequently not unique. With more donors than
pre-periods — the usual case — several weight vectors fit the pre-period
identically and imply different counterfactuals, and which one is
returned is an accident of the algorithm. Abadie & L'Hour (2021) break
the tie with a penalty on pairwise discrepancy between the treated unit
and each donor it draws on; `penalty_metric = :spatial` makes that
discrepancy geographic.

What the penalty buys is proximity, not sparsity, and it is paid for in
pre-period fit. Measured on a 9×9 lattice with spatially unstructured
characteristics:

| `λ` | donors | mean donor distance | pre-RMSPE |
|---|---|---|---|
| 0.00 | 3 | 3.167 | 0.076 |
| 0.10 | 4 | 3.139 | 0.076 |
| 1.00 | 5 | 2.810 | 0.094 |
| 10.0 | 3 | 1.510 | 0.275 |

Mean distance falls monotonically; the number of donors does not. On the
*same lattice with spatially smooth characteristics* the penalty changes
nothing at all — the unpenalised fit had already chosen the two nearest
donors, because there proximity and fit agree. Worth knowing in both
directions: the penalty is redundant when characteristics are smooth, and
it is doing real work when they are not.

As `λ → ∞` the support collapses into the set of minimum-distance donors
(a single donor when distances are distinct, the tied set otherwise).

### Predictor weighting

Classic synthetic control nests two optimisations: inner weights `w`
given a predictor-weighting matrix `V`, and an outer search over `V`.
This implementation takes **`V = I` on standardised predictors** and does
not search, deliberately. The outer problem is documented as non-unique,
so two equally good `V` give different weights and different
counterfactuals — the estimate stops being reproducible, which is not a
trade this package makes. `standardize = true` scales each predictor to
unit variance, which is the part of `V`'s job that is well posed.

### Inference

There are no standard errors, and `SyntheticControlResult` deliberately
does not define `vcov` or `stderror`. The weights are a constrained
optimum rather than an estimator with a sampling distribution of known
form, and the gap is a difference between an observed path and a
constructed one.

[`placebo_test`](@ref) reassigns the treatment to units that were never
treated, refits, and ranks the observed post/pre RMSPE ratio against the
placebo distribution. Dividing by the pre-period fit is what makes
placebo ratios comparable: a donor the pool tracks badly before treatment
shows a large post-period gap for reasons unrelated to any policy. All
`binomial(J, m)` assignments are enumerated when that number is small
enough, giving an exact permutation test; otherwise they are sampled from
a seeded RNG, and the result records which happened.

A **p-value is a test statistic, so its null distribution is measured**
under the standing rule in `docs/AUDIT.md`. No treatment effect anywhere,
7×7 lattice, 44 clean donors, 1,000 replications:

| `ρ` | `p ≤ 0.022` | `p ≤ 0.05` | `p ≤ 0.10` | mean `p` |
|---|---|---|---|---|
| 0.0 | 0.023 | 0.046 | 0.095 | 0.503 |
| 0.6 | 0.017 | 0.045 | 0.094 | 0.492 |
| 0.9 | 0.041 | 0.068 | 0.125 | 0.438 |

Correctly sized with no spatial correlation and with moderate
correlation — a mean p-value of 0.50 is what exchangeability predicts. At
`ρ = 0.9` it over-rejects by roughly a third of the nominal level, about
2.6 standard errors at this replication count. The planning expectation
was that spatial dependence would break the test badly; it does not,
because the ratio statistic scales the gap by the pre-period fit and
spatial dependence largely divides out.

The hard limit is granularity, not size. The smallest attainable p-value
is `1/(1+R)`, so a single treated unit with 18 donors cannot return
anything below 0.053 — it cannot reject at the 5% level however large the
effect. `min_attainable_p` reports the bound.

### Spillovers as an estimand

The units screened out of the donor pool were screened out because the
intervention reached them, which is itself worth measuring.
[`spillover_gaps`](@ref) synthesises each excluded neighbour from the same
clean pool and reads its gap — the synthetic control counterpart of
`att_spill_control` in [`fit_spatial_did`](@ref).

Read `rmspe_pre` per unit before any of those gaps. In testset 35f two of
four neighbours are exactly reproducible from the clean pool and return
the spillover to `1e-10`; the other two cannot be fitted at all
(pre-period RMSPE larger by more than ten orders of magnitude) and their
gaps are not spillovers. The unweighted average over all four is
consequently off, and the fix is reading the diagnostic rather than the
function silently discarding units.

## Shift-share designs

A shift-share, or Bartik, regressor is built from common shocks and
region-specific exposure:

```math
X_i = \sum_{k} w_{ik} \, g_k
```

with ``w_{ik}`` region ``i``'s share of industry ``k`` and ``g_k`` the
industry's national shock. It is used directly, or as an instrument for an
endogenous regional variable — the canonical Bartik design.

This belongs in a spatial package for a reason that is almost the opposite
of the rest of it. Everywhere else here, dependence travels through
geography and the fix is a spatial weight matrix. Here it travels through
**shared industry mix**: two regions with the same industries have
mechanically correlated values of ``X``, and therefore correlated
residuals, no matter how far apart they are. Two counties on opposite
coasts with identical industrial structure are as dependent as neighbours,
and a spatial HAC correction cannot see it.

### Build the design, then ask how big it really is

```julia
design = shift_share_design(W, g)          # W is N x K, g is length K
conc   = exposure_concentration(design)
conc.effective_shocks                       # compare with conc.n_shocks
```

Borusyak, Hull & Jaravel's identification argument requires a Herfindahl
condition on average exposure, ``\sum_k \hat{s}_k^2 \to 0``: no finite set
of shocks may drive the estimate. `effective_shocks` is the reciprocal of
that sum, reported because a count is easier to read.

**Read it first.** The asymptotics in this design are in the number of
shocks, not the number of regions, and every inference below degrades
together as the effective count falls. A study with 300 regions and 400
industries whose effective count is 12 is a twelve-observation study.

### The design has ``K`` observations, not ``N``

```julia
agg = shock_aggregate(design, y)
agg.beta_equiv == fit_shift_share(y, design).beta   # to machine precision
```

Borusyak, Hull & Jaravel (2022, eq. 4) show the shift-share estimate is
*numerically identical* to an exposure-weighted estimate computed from `K`
shock-level observations:

```math
\hat{\beta} = \frac{\sum_k \hat{s}_k g_k \bar{y}_k}{\sum_k \hat{s}_k g_k \bar{x}_k}
```

This is an identity, not an approximation, and it is the clearest statement
of where the information in the design actually is.

### Standard errors

```julia
fit = fit_shift_share(y, design)                    # vcov = :akm by default
fit = fit_shift_share(y, design; treatment = t)     # the IV design
```

Adão, Kolesár & Morales (2019) show by placebo that conventional standard
errors in this design reject a true null in **up to 55% of samples** at the
5% level. Their estimator (eq. 29) is

```math
\mathrm{se}(\hat{\beta}) = \frac{\sqrt{\sum_k \mathring{X}_k^2 \hat{R}_k^2}}{\sum_i \tilde{X}_i^2},
\qquad \hat{R}_k = \sum_i w_{ik}\,\hat{\varepsilon}_i
```

where ``\tilde{X}`` is the regressor with controls partialled out and
``\mathring{X}`` is the vector of coefficients from regressing
``\tilde{X}`` on the share matrix. ``\mathring{X}`` is **not** the shock
vector ``g``; the two coincide only with no controls and a square
invertible share matrix.

Coverage of a nominal 95% interval, measured over 2,000 replications with
`N = 300`, shares and residuals fixed and the shocks resampled — the
framework the estimator is derived in. "Sector frac" is the fraction of
residual variance carried by an industry-level component:

| Effective shocks | Sector frac | AKM | HC1 | cluster on top industry |
|---:|---:|---:|---:|---:|
| 99.9 | 0.00 | 0.930 | 0.950 | 0.947 |
| 99.9 | 0.40 | 0.938 | 0.902 | 0.894 |
| 99.9 | 0.82 | **0.936** | **0.732** | 0.720 |
| 99.9 | 1.00 | 0.937 | 0.664 | 0.658 |
| 50.0 | 0.82 | 0.919 | 0.632 | 0.619 |
| 20.0 | 0.77 | 0.864 | 0.426 | 0.404 |
| 37.9 | 0.90 | 0.827 | **0.289** | 0.241 |

Three things worth taking from it.

The failure is real and large. HC1 rejects a true null in up to **71%** of
samples on these designs — worse than the paper's headline figure, on a
design built to be realistic rather than adversarial.

**Clustering on each region's dominant industry does not fix it.** That is
the intuitive repair, and the cluster column tracks HC1 almost exactly,
because the dependence runs through the entire share vector rather than
through any one industry. This is worth stating plainly because the
cluster-robust machinery is available elsewhere in this package and would
be the natural thing to reach for.

**The AKM standard error itself degrades when shocks are few**, to 0.864 at
20 effective shocks and 0.827 at 38 with concentrated exposure. Adão,
Kolesár & Morales anticipate this: eq. 29 generalises a cluster-robust
formula and inherits its downward bias from estimation noise in the
residuals, which is why they also propose a null-imposed interval (their
Remark 6). **That variant is not implemented here.** Where
`exposure_concentration` reports few effective shocks, treat the interval
as optimistic.

Note also the first row: with no industry component at all, AKM costs a
little — 0.930 against HC1's 0.950. That is the price of robustness when it
is not needed, and it is small.

### Exogeneity

```julia
bal = balance_test(design, hcat(pretrend, demographics))
```

The identifying assumption is quasi-random assignment of the **shocks**,
not of the shares — shares are allowed to be endogenous. So the balance
test belongs at shock level: form the exposure-weighted average of each
residualised characteristic and regress it on the shock, weighting by
average exposure (Borusyak, Hull & Jaravel 2022, §4.3). Conventional robust
standard errors are valid there, which is the practical payoff of the
equivalence above.

Measured size, 2,000 replications, with the characteristic itself carrying
an industry component so that it is structured rather than iid:

| Effective shocks | 10% | 5% | 1% | joint, 4 characteristics at 5% |
|---:|---:|---:|---:|---:|
| 99.9 | 0.108 | 0.056 | 0.010 | 0.088 |
| 50.0 | 0.115 | 0.064 | 0.018 | 0.147 |
| 20.0 | 0.170 | 0.109 | 0.043 | 0.324 |
| 13.0 | 0.278 | 0.170 | 0.072 | — |
| 9.1 | 0.335 | **0.232** | 0.114 | — |

The test is correctly sized only when the effective number of shocks is
large. At 9 effective shocks it rejects a true null 23% of the time at a
nominal 5%, and the joint test degrades faster still since it estimates an
``M \times M`` covariance from as many observations as there are shocks.

The consequence is uncomfortable and worth saying: **in a concentrated
design, a passed balance test is close to uninformative.** It is reported
with a warning rather than left to be read as reassurance.

### Which industries identify the estimate

```julia
rot = rotemberg_weights(y, treatment, design)
rot.order[1:5]        # the industries worth defending
```

Goldsmith-Pinkham, Sorkin & Swift (2020, Prop. 3) decompose the estimate
into one just-identified estimate per industry:

```math
\hat{\beta} = \sum_k \hat{\alpha}_k \hat{\beta}_k, \qquad \sum_k \hat{\alpha}_k = 1
```

Both the unit sum and the decomposition hold exactly. **Negative weights
are expected** — the paper says so explicitly, and under constant effects
they pose no conceptual problem — so they are reported rather than
suppressed.

The use is blunt: if five industries carry most of the weight mass, the
study rests on those five, and they are the ones whose exogeneity a reader
should interrogate.

### Spatial autocorrelation, and why it is not the issue

```julia
shift_share_moran(design, W_spatial; fit = fit)
```

Industry shares cluster geographically, so Moran's I on them is almost
always significant. That is normal and not a defect. What the diagnostic is
for is documenting the spatial structure while making clear that a spatial
correction is the wrong tool: use `vcov = :akm` for inference and this for
description.

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
- Borusyak, K., Hull, P. & Jaravel, X. (2022). Quasi-experimental
  shift-share research designs. *Review of Economic Studies*, 89(1).
- Adão, R., Kolesár, M. & Morales, E. (2019). Shift-share designs: theory
  and inference. *Quarterly Journal of Economics*, 134(4).
- Goldsmith-Pinkham, P., Sorkin, I. & Swift, H. (2020). Bartik instruments:
  what, when, why, and how. *American Economic Review*, 110(8).
- Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control
  methods for comparative case studies: estimating the effect of
  California's tobacco control program. *Journal of the American
  Statistical Association*, 105(490), 493-505.
- Abadie, A. & L'Hour, J. (2021). A penalized synthetic control estimator
  for disaggregated data. *Journal of the American Statistical
  Association*, 116(536), 1817-1834.
- Abadie, A. (2021). Using synthetic controls: feasibility, data
  requirements, and methodological aspects. *Journal of Economic
  Literature*, 59(2), 391-425.
- Di Stefano, R. & Mellace, G. (2024). The inclusive synthetic control
  method.
