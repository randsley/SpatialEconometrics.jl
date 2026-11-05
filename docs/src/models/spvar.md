# Spatial VAR

`fit_star` models **one** variable through space and time. A regional
economist usually has several, and wants to ask whether a shock to one of
them in region `i` moves another one in region `j`, and after how long. That
is a system question, and answering it honestly means taking a position on
contemporaneous causal ordering — which is the one thing a spatial panel
cannot settle from the data.

This page covers `SpVAR`, an implementation of **Di Giacinto (2010)**, *On
vector autoregressive modeling in space and time*, Journal of Geographical
Systems 12(2), 125–154.

## The specification

Stack the data location-within-variable,
`yₜ = [y₁₁ₜ,…,y_N₁ₜ,…,y₁Kₜ,…,y_NKₜ]′`, length `NK`. The structural form is

```
C₀ yₜ = μ + C₁ yₜ₋₁ + … + C_p yₜ₋ₚ + uₜ,   E(uₜuₜ′) = Λ = diag(λ₁₁,…,λ_NK)
```

Each `C_h` is partitioned into `K×K` blocks `A⁽ʰ⁾_kr` of size `N×N`, and each
block is a polynomial in the **order-`l` spatial weights**,

```
A⁽ʰ⁾_kr = Σ_{l=0}^{s} W⁽ʰˡ⁾_kr Φ⁽ʰˡ⁾_kr
```

with `Φ` diagonal in the locations. `SpVAR` ships the nested homogeneous
case, `Φ⁽ʰˡ⁾_kr = φ⁽ʰˡ⁾_kr I_N`.

### Block triangularity, and why it is not strict triangularity

A conventional SVAR identifies `C₀` by making it triangular. Doing that here
would order not only the variables but **the locations**: a recursive causal
chain through space, so that region 3 may affect region 7 contemporaneously
but never the reverse. Spatial interaction is bilateral, so Di Giacinto
rejects that and makes `C₀` triangular in the **variable** index only, with
full `N×N` blocks on the diagonal:

```
A⁽⁰⁾_kk = I_N − Σ_{l=1}^{s} W⁽⁰ˡ⁾_kk Φ⁽ˡ⁾_kk      (the SAR-form diagonal block)
A⁽⁰⁾_kr = −Σ_{l=0}^{s} W⁽⁰ˡ⁾_kr Φ_kr              for r < k
```

Two consequences, both load-bearing.

**The parameter count becomes linear in `N`.** Each `C_h` holds `NK²(s+1)`
free coefficients instead of `(NK)²`.

**The likelihood separates.** Block triangularity gives
`|C₀| = Π_r |A⁽⁰⁾_rr|`, so concentrating `Λ` out of the FIML log-likelihood
leaves a **sum of `K` unrelated terms**, one per endogenous variable:

```
L̃ = Σ_{k=1}^{K} L̃_k,   L̃_k = c̃_k + T log|A⁽⁰⁾_kk| − (T/2) Σ_i log λ̂_ik
```

A spatial VAR is therefore `K` spatial autoregressions, not one `NK×NK`
problem. `fit_spvar` maximises each `L̃_k` on its own. Both identities are
tested to machine precision (`0.000e+00` for the determinant factorisation
and `0.000e+00` for the separability, testsets `40c` and `40d`).

## The asymptotics run the other way

!!! warning "This is a `T → ∞` estimator at fixed `N`"
    Every other panel estimator in this package is consistent as `N` grows.
    Di Giacinto's FIML is not: its asymptotics are standard SVAR
    asymptotics, `T → ∞` with `N` held fixed. A wide short panel is the
    wrong shape of data for this model.

That is a claim, so it was measured. The bias in a dynamic coefficient
(`g(t−1)`, truth `0.30`), from a homogeneous SpVAR(1,1) with `K = 2` on a
ring:

| `N` fixed at 12, `T` growing | bias | | `T` fixed at 50, `N` growing | bias |
|---|---|---|---|---|
| `T = 50`  | −0.0337 | | `N = 12` | −0.0267 |
| `T = 100` | −0.0180 | | `N = 24` | −0.0318 |
| `T = 200` | −0.0084 | | `N = 48` | −0.0296 |
| `T = 400` | −0.0040 | | `N = 96` | −0.0294 |
| `T = 800` | −0.0014 | | | |

(150 replications.) Growing `T` removes the bias; growing `N` does not
touch it. `N` buys variance — RMSE does fall — but not freedom from the
dynamic bias. The
spatial parameter `ρ₁` is far less affected in both directions, which is
why the warning is about the *dynamics*, not the spatial coefficients.

Coverage of the nominal 95% interval, 400 replications (so a Monte Carlo
standard error of about `0.011`):

| | `ρ₁` | `g(t−1)` | eq-2 `g` |
|---|---|---|---|
| `N = 12, T = 100` | 0.935 | 0.935 | 0.948 |
| `N = 12, T = 200` | 0.945 | 0.925 | 0.953 |
| `N = 12, T = 800` | 0.955 | 0.950 | 0.948 |
| `N = 24, T = 200` | 0.960 | 0.940 | 0.927 |

Coverage is close to nominal throughout and slightly short of it for the
dynamic coefficient at small `T`, which is the same finite-sample bias the
table above measures, showing up in the interval rather than the point
estimate. It does not improve with `N`.

## The ordering is required, and it changes the answer

`ordering` has **no default**. It names the contemporaneous causal order of
the `K` variables, first mover first, and it *is* the identifying
assumption. Defaulting it would hide the only choice the user has to make.

On a 14-region ring with `T = 400` (`examples/29_spvar.jl`, sections 2 and
3), reversing the ordering of a two-variable system moved the
log-likelihood by **9.949** and the peak outward STIR of `y` to a shock in
`g` by a factor of **5.25** at spatial order 0 and **1.33** at order 1.

It also moved the **timing**. Under the true ordering the order-0 response
peaks on impact, `h = 0`, because the shock reaches `y` within the period.
Under the reversed ordering it peaks at `h = 2`, because the effect now has
to travel through a lag to exist at all. The reversed fit does not re-label
the shocks; it answers a different question, and both the size and the date
of its answer are wrong.

## Impulse responses in space and time

`spvar_irf` returns the raw structural response matrices
`Ψ_h = Ψ̃_h C₀⁻¹`, where `Ψ̃_i = Σ_{h=1}^{i} Ψ̃_{i−h} C̃_h` and `Ψ̃₀ = I`, so
`Ψ⁽ʰ⁾_kr(i,j) = ∂y_{ik,t+h}/∂u_{jrt}`. Summing them reproduces the long-run
multiplier `(C₀ − Σ_h C_h)⁻¹` to `2.665e-15`.

`space_time_stir` aggregates those `NK×NK` matrices into the paper's
**space-time impulse responses**, which come in two directions:

- **outward**, `ζ⁽ʰˡ⁾_kr(i) = Σ_j w⁽ˡ⁾_ij Ψ⁽ʰ⁾_kr(j,i)` — my shock reaching
  my order-`l` neighbours;
- **inward**, `ξ⁽ʰˡ⁾_kr(i) = Σ_j w⁽ˡ⁾_ij Ψ⁽ʰ⁾_kr(i,j)` — their shocks
  reaching me.

The two differ by a transpose, which is the whole content of the
distinction. They coincide when `K = 1` and the `W⁽ˡ⁾` are symmetric, and
that is a free exactness check: measured at `5.551e-17` on an
unstandardised path and `1.388e-17` on a ring.

!!! note "Row standardisation alone makes outward and inward differ"
    On an irregular graph, row standardising `W` destroys its symmetry, and
    `order_weights_list` row standardises by default. Measured on a 14-unit
    path with `K = 1`: a `W` asymmetry of `0.5` produced an outward/inward
    STIR gap of `1.345e-02`, against `2.776e-17` for the same path
    spectrally standardised, where the asymmetry is exactly zero.

    So the gap is not only a multivariate effect: under row
    standardisation a unit with fewer neighbours sends and receives on
    different scales, and the split reports that. If you want the two to
    coincide, use `spectral_standardize`, which bounds the spectrum without
    breaking symmetry. Leaving the weights raw is **not** the alternative
    — a binary path has eigenvalues near 2, so a model that is stationary
    under standardised weights is explosive under raw ones.

## Testing for spatial interaction at all

Setting every spatial coefficient to zero collapses the SpVAR into a
**nested panel VAR**, one small VAR per location,

```
M₀,ᵢ y_it = μᵢ + Σ_h M_h,ᵢ y_i,t−h + u_it
```

with `M₀,ᵢ` unit lower triangular. Di Giacinto gives this as a reason to
prefer block triangularity: the nested model stays identified even where
the SpVAR itself does not. It also gives a likelihood-ratio test.

`fit_nested_panel_var` is written independently of `fit_spvar`, which is
what makes the reduction worth testing: `fit_spvar(…; s = 0)` reproduces
it to `0.000e+00` in both the log-likelihood and every coefficient
(testset `40e`).

`lr_spatial_interaction` is that LR test, on `s(K(K+1)/2 + K²p)` degrees of
freedom. Its null distribution was simulated rather than assumed, on data
generated with **no spatial interaction at all**:

| rejection rate under the null | 10% | 5% | 1% |
|---|---|---|---|
| `N = 12, T = 100` | 0.1100 | 0.0527 | 0.0113 |
| `N = 12, T = 400` | 0.1147 | 0.0513 | 0.0080 |
| `N = 24, T = 200` | 0.1193 | 0.0620 | 0.0100 |

1,500 replications, so the Monte Carlo standard error is about `0.008` at
the 10% level and `0.006` at 5%. The test is **mildly liberal** at 10%, by
one to two standard errors, and close to nominal at 5% and 1%. Power against
the spatial DGP is `1.000` at the 5% level already at `T = 50`.

!!! warning "150 replications said the opposite"
    An earlier run at 150 replications reported the test *conservative*
    (10%: 0.073–0.087, 5%: 0.033–0.047, 1%: 0.000–0.007). That was noise:
    the Monte Carlo standard error at 150 draws is `0.024` at the 10% level,
    wide enough to swallow the whole finding and reverse its sign. A size
    table is not worth printing until the replication count is large enough
    for the error bar to be smaller than the deviation being claimed.

!!! note "`homogeneous = true` is the comparable case"
    The heterogeneous nested panel VAR has *more* parameters than the
    homogeneous SpVAR, not fewer, so it is not nested in it and must not be
    LR-tested against it. `lr_spatial_interaction` fits the homogeneous
    restriction.

## At `K = 1`, and what it is not comparable to

With one variable the system collapses to a single spatial autoregression
with its own lags, `(I − ρW)yₜ = μ + (φ₀I + φ₁W)yₜ₋₁ + uₜ`, and `|C₀|`
reduces to `|I − ρW|`. That is a specification `fit_star` cannot express,
because `fit_star` has no contemporaneous `Wyₜ` by construction.

It is **not** the same model as `fit_dynamic_spatial_panel`, and the two
should not be compared coefficient by coefficient.

- Handing `fit_dynamic_spatial_panel` a column of `y(t−1)` **refuses** with
  an identification error, because it builds its own time lag and the
  column is exactly collinear with it.
- Fitted as intended on data from the specification above, it returns
  `ρ̂ = 0.1875` where `fit_spvar` returns `0.3749` against a true
  `ρ = 0.40`. Neither is in error: the dynamic spatial panel has no
  `W yₜ₋₁` term, and omitting one that is present does not leave `ρ`
  alone — it roughly halves it, since that spatial content has nowhere to
  go but the contemporaneous spatial parameter.
- `fit_spvar` also estimates a per-unit variance `λᵢ`; the dynamic spatial
  panel assumes a common one.

!!! note "How `μ` is absorbed, and why `s = 0` is not exact to machine precision"
    `fit_spvar` demeans each unit's **level** series once, over
    `t = p+1 … T`. This is required rather than convenient: the spatial lag
    of a demeaned series is not the demeaned spatial lag, so demeaning the
    constructed columns separately would break `(I − ρW)ỹ` and the
    Jacobian with it.

    At `s = 0` there is no spatial term and the exact profiled likelihood
    demeans the lagged and current sub-series by their own means instead,
    so the two differ in finite samples. The difference is **second
    order** in `T`, measured against an independently written profiled
    FIML: `1.45e-05` at `T = 300` falling to `1.11e-06` at `T = 1200` and
    `5.06e-08` at `T = 4800` — a ratio of about 4 per doubling, so
    `T²·gap` is roughly constant where `T·gap` is not.

## The order condition

Counting equations against free coefficients in `C₀` gives

```
NK(NK − 1) ≥ NK(K − 1) + sNK(K + 1)   ⟺   N ≥ s + (s + K)/K
```

so `K = 2, s = 1` needs `N ≥ 3` and `K = 2, s = 2` needs `N ≥ 4`.
`spvar_order_condition` returns that verdict, and `fit_spvar` **refuses
before estimating anything** when it fails, reporting the bound and the
counts that produced it. Returning numbers for an under-identified system
is not a service.

It is an **order** condition: necessary, not sufficient. A model that
satisfies it can still be unidentified in a particular sample, and
`check_design_rank` is the second line of defence.

The bound above counts the **heterogeneous** coefficients, which is the
condition the paper prints. `fit_spvar` estimates the homogeneous model, so
it applies the homogeneous count — refusing on the heterogeneous bound would
refuse models it can in fact estimate. Measured at `K = 2, s = 2` on a ring,
60 fits each: `N = 4, 5, 8` all estimate without incident (60/60, mean
companion spectral radius 0.711 to 0.727), and `N = 3` satisfies the
homogeneous condition but is refused one step earlier — by
`order_weights_list`, because a 3-ring has diameter 1 and `W⁽²⁾` is empty.
The guards bind in the right order and for the right reason.

## The weights are order-`l`, not powers of `W`

`W⁽ˡ⁾` means the matrix of **order-`l` neighbours** in the sense of Anselin
and Smirnov (1996): pairs at graph distance exactly `l`. **`W^l` is not
that.** Powers count walks, so they are non-zero on the diagonal and
double-count paths, and where the graph has triangles they retain
first-order neighbours at order 2.

`order_weights` and `order_weights_list` (in `SpatialWeights`) build them by
breadth-first shortest-path hop count. On a 4×4 rook lattice with 48
first-order links, `W²` has **84** non-zeros, **16 of them on the
diagonal**, against **68 and 0** for `order_weights(W, 2)`; summed over
orders 1 to 6 the order matrices cover exactly `N(N−1) = 240` ordered
pairs, each once.

`order_weights_list` refuses when an order is empty rather than fitting a
structurally zero column, and reports the graph diameter so you can lower
`s`.

## Forecasting, and what to compare it against

`spvar_forecast(m, h; y_history)` iterates the reduced form `C̃_h = C₀⁻¹C_h`
forward and **adds the absorbed per-unit intercepts back**, so it returns
forecasts in levels. `y_history` is required: the model stores no data.

Two things to get right when judging the result.

**Benchmark against each unit's in-sample mean, not against zero.** Because
the intercepts are absorbed during estimation and restored here, a zero
benchmark says nothing unless the series happen to be centred — and it
cannot detect a dropped intercept, which is the defect most worth ruling
out. On a system with intercepts of order 50 and −30, `spvar_forecast`
beats the sample-mean forecast by 7.5× and 4.4× in RMSE; against zero it
would appear to beat it by 200×, which measures the intercepts rather than
the model.

**Read the error by horizon.** The forecastable part decays at the
companion spectral radius. On a stationary system with radius 0.74
(`examples/29_spvar.jl`, section 8) the gain over the sample mean is clear
at `h = 1` — 1.1978 against 1.4670 — essentially gone by `h = 2`, and at
`h = 3` the model is marginally *worse*. Averaged over twelve horizons the
same forecast reads as a 1.03× improvement, which looks like a failure and
is not one: it is the model doing what the spectral radius permits. Check
`m.spectral_radius` before deciding what horizon to expect anything at.

## Computational notes

- `s = 1` makes each diagonal block the familiar `I − ρW`, so the
  Chebyshev log-determinant approximation applies.
- `s ≥ 2` makes it `I − Σ_l φ_l W⁽ˡ⁾`, a linear combination of several
  weight matrices. `chebyshev_logdet` is single-parameter by construction
  and **does not apply**; `SpVAR` uses an exact LU there. This is a
  documented limitation, not an oversight.
- The companion matrix is `Kp·N` square. `SpVARModel` reports its spectral
  radius and `stationary` flag; a non-stationary fit warns rather than
  returning silently divergent impulse responses.

## What is not here

- **Bootstrap confidence bands for the STIR** (the paper's §5 tail). The
  point estimates and asymptotic intervals stand without them.
- **Spatially heterogeneous coefficients** — the full eq (7), with `Φ`
  genuinely diagonal rather than scalar. The global STIR summaries assume
  homogeneity, and moderate `N` cannot support the heterogeneous count.
  Note that the order condition's counts are the *heterogeneous* ones, so
  the guard is implemented for both regardless.
- **Time-varying `ρ`**, which was once filed beside this row. See the
  roadmap: it belongs with the partially linear spatial model, since
  local-linear concentrated QML and profile QML share one kernel-weighted
  likelihood.

## Worked example

`examples/29_spvar.jl` fits a two-variable regional system in which the
causal ordering changes the conclusion.

## References

- Di Giacinto, V. (2010). On vector autoregressive modeling in space and
  time. *Journal of Geographical Systems*, 12(2), 125–154.
  doi:10.1007/s10109-010-0116-6
- Anselin, L. and Smirnov, O. (1996). Efficient algorithms for constructing
  proper higher order spatial lag operators. *Journal of Regional Science*,
  36(1), 67–89.
