# Semiparametric Models

Every other spatial model in this package fixes the functional form of the
covariates. This page covers three estimators that leave part of it unknown,
and the diagnostic that has to come with them. They implement roadmap #13,
first pass.

| Function | Model | Source |
|---|---|---|
| [`fit_partially_linear_sar`](@ref) | `y = ρWy + Xβ + m(z) + ε`, `m` unknown | Su & Jin (2010) |
| [`closed_form_rho`](@ref) | the same, `ρ` without a log-determinant | Smirnov (2022); Geniaux (2026) |
| [`spectral_overlap`](@ref) | how much of `W`'s dominant modes a smoother reproduces | Geniaux (2026) |
| [`fit_tv_coef_sar`](@ref) | `Y_t = ρWY_t + X_tβ(t/T) + α + e_t`, `β(·)` unknown | Liang, Gao & Gong (2022) |

## The partially linear SAR

For a fixed `ρ`, Su & Jin estimate `m` by a local-polynomial regression of
`(I − ρW)y − Xβ` on `z`. The smoother matrix `S` depends on `z` and the
bandwidth only, never on `ρ`. That is the property everything here rests on:
after residualising on `I − S`, the concentrated likelihood in `ρ` has exactly
the shape `fit_spatial` maximises, with the log-determinant kept.

```julia
m = fit_partially_linear_sar(y, X, z, W)                    # rule-of-thumb h
m = fit_partially_linear_sar(y, X, z, W; bandwidth = :lscv) # two-step LSCV
m.rho, m.beta, m.m_hat
```

Two practical rules:

- **No intercept in `X`.** A local-polynomial smoother reproduces constants,
  so `m` absorbs the intercept; so does any column that is a function of `z`
  alone. The fit refuses such a design as rank deficient rather than reporting
  an unidentified coefficient.
- **Homoskedastic errors.** Su & Jin's Assumption 2 rules out
  heteroskedasticity, under which the QMLE is generally inconsistent. The
  standard errors also assume normal errors: the extra term their sandwich
  carries for skewness and excess kurtosis is not included. The Monte Carlo
  below measures what that costs.

A large bandwidth turns the local-linear smoother into the projection on
`[1, z]`, and the fit then reproduces `fit_spatial` with `z` as a regressor:
`ρ̂`, `β̂`, both standard errors and the log-likelihood agree to about
`1e-11` (testset 41b).

## The closed-form ρ

Smirnov's estimator replaces the likelihood's log-determinant by a quadratic
quasi-score in the residuals of `y`, `Wy` and `W²y`. Here those residuals come
from the partially linear model's own first stage, so the two estimators share
it:

```julia
c = closed_form_rho(m)
c.rho, c.se_classical, c.se_hc0, c.se_shac
```

With a smoother in the first stage **no consistency proof covers this
estimator**. Smirnov gives none at all; Geniaux proves consistency for a fixed
spline basis, and a kernel smoother with a shrinking bandwidth is outside that
proof. Its coverage is measured below rather than assumed.

## The identification diagnostic

If `z` carries geography — coordinates, or a distance to a fixed point — the
smoother can reproduce the same low-frequency spatial patterns that `ρWy`
explains. The two then compete for them, and `ρ` loses its identification.
Geniaux (2026) calls this the geoadditive trap. His diagnostic is the share
of each of `W`'s dominant spatial modes that the smoother can reproduce,
`π_k = ‖Sφ_k‖`, summarised by its mean `π̄`. Every fit computes it:

```julia
m.overlap            # SpectralOverlapResult
m.overlap.pibar
```

The modes are the eigenvectors of the doubly-centred symmetrised `W`. Geniaux
states the uncentred one; the centring is deliberate, because every smoother
here reproduces constants and the top mode of an uncentred row-standardised
`W` is nearly constant, so it would read close to 1 for any fit.

!!! warning "A measure, not a test — and not a causal statement"
    `π̄` has no threshold and no null distribution in its source, so the
    package attaches no p-value and no verdict to it. A small `π̄` says only
    that the smoother does not compete with `ρWy` for `W`'s dominant modes,
    **given that `W` is the true interaction structure**. That is statistical
    identification of `ρ`, not causal identification of a spillover. The
    endogenous and correlated effects stay confounded (Manski's reflection
    problem), and in a cross-section spatial heterogeneity and spillovers are
    observationally equivalent (Gibbons & Overman 2012; Debarsy & Le Gallo
    2024). Nothing here makes `ρ̂`, or the impacts computed from it, causal.

## Time-varying coefficients

Liang, Gao & Gong's model lets the regressor coefficients drift as smooth
functions of rescaled time `τ = t/T`, with individual fixed effects and a
**constant** `ρ`. The estimator has the same structure as Su & Jin's: a
local-linear smoother in time that does not depend on `ρ`, the fixed effects
partialled out, and a concentrated likelihood that keeps `T log|I − ρW|`.

```julia
tv = fit_tv_coef_sar(y, hcat(ones(N * T), x), W, N, T)   # bandwidth by CV
tv.rho, tv.beta, tv.beta_se, tv.tau
```

The data are stacked `N` units within each of `T` periods, as in the
package's other panels. The default bandwidth is the paper's
leave-one-unit-out cross-validation. `beta_se` is pointwise and **excludes
the smoothing bias**, of order `h²β''(τ)`. A time-varying `ρ` is a different
model, with no published method, and is not offered.

**The band under-covers in the interior, not at the ends.** That is the
measured result (`validation/mc_tv_coef.jl`, 500 replications per cell,
`N ∈ {50, 100}`, `T ∈ {25, 50}`, a gently and a strongly curved `β`):

| bandwidth | interior coverage (τ ∈ [0.2, 0.8]) | at the ends | fits refused |
|---|---|---|---|
| cross-validated `h` | 0.81–0.87 (minimum 0.77) | 0.92–0.96 | none |
| `h/2`, refitted | 0.92–0.94 (minimum 0.90) | 0.93–0.96 | up to 44 of 500 at `T = 25` |

Cross-validation picks the bandwidth that minimises mean squared error, and
that makes the squared bias comparable to the variance. That under-covers
wherever `β` curves. At the ends the variance is larger, so the same bias
costs less coverage. The varying bandwidth adds to it: `beta_se` is
conditional on the `h` that was chosen, and `h` itself varies from sample
to sample. With `h` held fixed and `ρ`, `α` known, `beta_se` matched the
Monte Carlo spread to within 5%. With `h` chosen by CV it fell to 0.87–0.92
of it.

For a band to read as a confidence band, refit at a smaller bandwidth:

```julia
tv  = fit_tv_coef_sar(y, X, W, N, T)
tvu = fit_tv_coef_sar(y, X, W, N, T; bandwidth = tv.bandwidth / 2)
```

That is undersmoothing. It trades a wider band for one that is centred
closer to `β(τ)`. With short panels it can fail: at `T = 25` the halved
window was too thin on up to 9% of samples, and coverage of `ρ` fell to
0.91–0.93 for the strongly curved `β`.

A very large bandwidth turns the fit into a within regression on `x`, `τ` and
`τx`, and `ρ̂` then matches `fit_spatial_panel(…; approach = DirectQML(bias_correct = false))`
to about `1e-11` (testset 41f). The standard errors differ by a single
convention: the paper's information scales the trace terms by `T`, the
package's panel code by `T − 1` under individual fixed effects.

## What the Monte Carlo shows

Each cell uses 500 replications. `W` is a row-standardised 6-nearest-neighbour
graph on uniform points, with true `β = 1` and the rule-of-thumb bandwidth. In the
**safe** design, `z` is drawn independently of location (Su & Jin's §6
design). In the **trap**, `z` is the coordinates, `m` is a smooth surface in
them, and `x` carries the first coordinate. Skewed errors are a centred
`χ²₃/√6`: skewness 1.6, excess kurtosis 4. Coverage is that of the nominal
95% interval. The closed-form columns are its three standard errors. The full
table, with RMSE and the fits that needed a widened window, is in
`docs/AUDIT.md`.

| design | errors | n | ρ | QML bias | cover ρ | cover β | CFE classical | CFE HC0 | CFE SHAC | mean π̄ |
|---|---|---|---|---|---|---|---|---|---|---|
| safe | normal | 400 | 0.25 | −0.005 | 0.956 | 0.956 | 0.968 | 0.916 | 0.924 | 0.118 |
| safe | normal | 400 | 0.75 | −0.004 | 0.954 | 0.954 | 0.996 | 0.920 | 0.928 | 0.118 |
| safe | normal | 1000 | 0.25 | −0.002 | 0.966 | 0.950 | 0.972 | 0.930 | 0.952 | 0.084 |
| safe | normal | 1000 | 0.75 | −0.001 | 0.968 | 0.948 | 0.998 | 0.930 | 0.954 | 0.084 |
| safe | skewed | 400 | 0.25 | −0.005 | 0.968 | 0.952 | 0.980 | 0.914 | 0.926 | 0.119 |
| safe | skewed | 400 | 0.75 | −0.004 | 0.946 | 0.958 | 1.000 | 0.914 | 0.930 | 0.119 |
| safe | skewed | 1000 | 0.25 | −0.004 | 0.944 | 0.968 | 0.970 | 0.896 | 0.936 | 0.085 |
| safe | skewed | 1000 | 0.75 | −0.003 | 0.954 | 0.964 | 0.998 | 0.896 | 0.936 | 0.085 |
| trap | normal | 400 | 0.25 | −0.099 | 0.626 | 0.948 | 0.706 | 0.578 | 0.674 | 0.597 |
| trap | normal | 400 | 0.75 | −0.069 | 0.430 | 0.942 | 0.880 | 0.578 | 0.678 | 0.597 |
| trap | normal | 1000 | 0.25 | −0.052 | 0.690 | 0.944 | 0.772 | 0.618 | 0.726 | 0.583 |
| trap | normal | 1000 | 0.75 | −0.035 | 0.526 | 0.940 | 0.906 | 0.616 | 0.740 | 0.583 |

What it shows:

- **In the safe design the profile QML is sound.** Coverage of `ρ` and `β`
  is between 0.944 and 0.968 in every cell, **including the skewed ones**.
  At these sample sizes, omitting Su & Jin's `Ω` costs nothing measurable.
- **The closed-form `ρ` has no reliable standard error.** The classical one
  is conservative, and at `ρ = 0.75` it covers at 0.996–1.000. HC0
  under-covers everywhere, at 0.896–0.930. SHAC reaches nominal only for
  normal errors at `n = 1000`, and stays at 0.926–0.936 under skewed
  errors. This matches what Geniaux reports for his spline version. With a
  kernel first stage it is a measurement, not a theorem. Use the QML
  interval for inference, and the closed form as a check.
- **The trap is real, and more data does not get you out of it.** `ρ̂` is
  biased down by 0.035–0.099. The bias halves from `n = 400` to `n = 1000`,
  but the standard error shrinks too, so the QML coverage of `ρ` stays
  between 0.43 and 0.69. Coverage of `β` does not suffer. No standard error
  repairs this. The bias is the problem, not the variance.
- **`π̄` separated the two designs.** Its mean was 0.08–0.12 in the safe
  design and falling with `n`, against 0.58–0.60 in the trap and flat in
  `n`. That is a comparison of two designs, not a calibrated threshold:
  `π̄` depends on `n`, `W` and the bandwidth, and only its mean was
  recorded. It does not license a cut-off for other data.

## References

- Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of
  partially linear spatial autoregressive models. *Journal of Econometrics*,
  157(1), 18–33.
- Smirnov, O. A. (2022). Closed-form estimator for spatial autoregressive
  models. *Geographical Analysis*, 54(1), 3–31.
- Geniaux, G. (2026). Flexible nonlinear spatial autoregressive models under
  spatial confounding. arXiv:2611.16592 (preprint).
- Liang, X., Gao, J., & Gong, X. (2022). Semiparametric spatial
  autoregressive panel data model with fixed effects and time-varying
  coefficients. *Journal of Business & Economic Statistics*, 40(4),
  1784–1802.
- Gibbons, S., & Overman, H. G. (2012). Mostly pointless spatial
  econometrics? *Journal of Regional Science*, 52(2), 172–191.
- Debarsy, N., & Le Gallo, J. (2024). Identification of spatial spillovers:
  Do's and dont's. LEM working paper DT2024-03.
