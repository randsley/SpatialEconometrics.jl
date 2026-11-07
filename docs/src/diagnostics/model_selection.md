# Model Selection & Comparison

## Information Criteria

```julia
spatial_aic(m)  # Akaike Information Criterion
spatial_bic(m)  # Bayesian Information Criterion
```

## Likelihood Ratio Test

For nested models (e.g., SAR nested in SDM):

```julia
lr = lr_test(m_restricted, m_unrestricted)
# lr.stat — LR statistic (χ²)
# lr.df — degrees of freedom
# lr.p_value — reject H₀ → unrestricted model preferred
```

## Vuong Test

For non-nested models (e.g., SAR vs SEM):

```julia
vt = vuong_test(m1, m2)
# vt.stat — Vuong statistic (positive favors m1)
# vt.preference — :model1, :model2, or :indistinguishable
```

## Common Factor Test

Tests whether SDM simplifies to SEM (H₀: θ + ρβ = 0):

```julia
cft = common_factor_test(beta_slope, theta, rho, vcov_full)
# cft.reject — true → SDM preferred over SEM
```

## Wald Test

General linear hypothesis Rβ = r:

```julia
R = zeros(1, length(coef(m)))
R[1, 2] = 1.0  # test H₀: β₁ = 0
wt = wald_test(m, R, [0.0])
```

## Comparison Table

```julia
tbl = model_comparison_table(m_sar, m_sdm, m_sem; names=["SAR","SDM","SEM"])
println(tbl)
```

## Bayesian specification comparison

Everything above compares models you have already fitted, on a `W` you have
already chosen. `bayes_spec_probs` compares the *specifications* — including
ones of different width, and including the choice of `W` — by integrating
each one's likelihood over its parameters instead of maximising it.

```julia
ev = bayes_spec_probs(y, X, [W_rook, W_queen]; w_names = ["rook", "queen"])
println(ev)
# ev.prob         — the posterior over the (specification, W) grid
# ev.prob_spec    — its specification margin
# ev.prob_w       — its weight-matrix margin
# ev.prob_global  — mass on :sar and :sdm, the specifications with ρ on y
# ev.prob_local   — mass on :slx, :sem and :sdem, the rest
```

The six specifications, their designs, and where each puts the dependence:

| `spec` | Design | Spatial parameter | Spillovers |
|---|---|---|---|
| `:ols` | `[1 X]` | — | none |
| `:slx` | `[1 X WX]` | — | local, through `W*X` |
| `:sar` | `[1 X]` | `ρ` on `y` | global |
| `:sdm` | `[1 X WX]` | `ρ` on `y` | global |
| `:sem` | `[1 X]` | `λ` on `u` | local |
| `:sdem` | `[1 X WX]` | `λ` on `u` | local |

Global against local is not a nesting question: `:sdm` and `:sdem` have the
same design and the same number of parameters, and no likelihood-ratio test
runs between them. That is why a posterior over the grid is useful where a
test is unavailable, and `prob_global` and `prob_local` are its two sums.

Be careful what they are taken to answer. They rank **reduced forms** by
marginal likelihood over the specifications supplied, and the `Spillovers`
column above names the reach each specification *assumes*, not one the
comparison discovers. A model with dependence in the disturbance and one
with dependence in the outcome can imply the same covariance, so a decisive
`prob_global` is evidence that `ρ` on `y` describes these data better than
the offered alternatives — not that outcomes propagate between neighbours.
Separating mechanisms needs neighbourhood variation that is exogenous to the
outcome, which no information criterion or posterior in this package can
manufacture.

### What is integrated, and what is not

Under the improper prior `p(β, σ²) ∝ 1/σ²` the inner integral is exact,
leaving one dimension to quadrature:

```math
p(y \mid \rho) = |A(\rho)| \; \pi^{-(n-k)/2} \; |Z'Z|^{-1/2} \;
                 \Gamma\!\left(\tfrac{n-k}{2}\right) \; S(\rho)^{-(n-k)/2}
```

`A(ρ)` is `I - ρW` for a lag specification and `I - λW` for an error one,
`Z` is the design, and `S` is the residual sum of squares at that value of
the parameter. The `|Z'Z|^{-1/2}` and `Γ((n-k)/2)` factors are what make a
wider design pay for its width, so they are kept rather than dropped as
constants: dropping them still ranks `:sar` against `:sem` correctly and
mis-ranks every comparison that crosses a `k`.

For an error specification both `S` and `|Z'Z|` are evaluated on the filtered
data `(I - λW)[y Z]`, so the determinant term moves with `λ`. It is constant
for a lag specification, which is why leaving it out is an easy mistake and a
silent one — it biases the comparison toward the error specifications, by an
amount that grows with `|λ̂|`.

The outer integral is a trapezoid rule over `rho_bounds`, evaluated by
log-sum-exp because the integrand spans hundreds of orders of magnitude.
Where the posterior rises and falls away inside the support the rule is
geometrically accurate and 100 nodes is already at machine precision; where
it is still appreciable at the edge — a misspecified error model wanting
`λ → 1`, or any specification at very small `N` — the rate drops to the
generic `O(h²)` and the error at the default 200 nodes is around `1e-4`,
three orders below the gaps that separate specifications.

!!! warning "The prior does not wash out"
    A marginal likelihood is not a likelihood ratio. The prior sits on a
    parameter that `:ols` and `:slx` do not have, so widening `rho_bounds`
    spreads it thinner and penalises every spatial specification by exactly
    the width ratio. Report the setting next to the probabilities, and if a
    comparison is close, run it under `prior = :beta` as well. Nothing about
    increasing `N` removes this.

### How often it finds the generating specification

A selection procedure's gate is its error rate, so: 200 draws from each of
the six specifications on a rook lattice, `ρ = λ = 0.5`, `θ = (0.7, 0)`,
`σ = 0.5`, with all six compared on the generating `W` under the default
uniform prior. The entry is the share of draws whose highest posterior
probability lands on the generating specification.

| `N` | `:ols` | `:slx` | `:sar` | `:sdm` | `:sem` | `:sdem` | reach right |
|---|---|---|---|---|---|---|---|
| 100 | 93.0% | 90.0% | 81.5% | 95.5% | 88.5% | 88.5% | 90.5–100% |
| 256 | 92.5% | 96.0% | 94.0% | 100% | 96.5% | 96.0% | 96.0–100% |
| 400 | 94.5% | 94.0% | 97.5% | 100% | 96.5% | 98.5% | 98.5–100% |

Chance is 16.7%. The mean posterior probability assigned to the generating
specification runs 0.589–0.916 at `N = 100` and 0.711–0.999 at `N = 256`, so
the procedure is not merely picking the right winner — it is reasonably
confident about it, and more so as `N` grows.

The last column is the share of draws that get the *reach* right — mass on
global exceeding mass on local when the data are `:sar` or `:sdm`, and the
other way round for `:slx`, `:sem` and `:sdem`. That coarser question is
answered more reliably than the full six-way choice at every `N`, which is
what you would expect, and it is usually the question the application is
actually asking.

The confusions are the ones the structure predicts, and they are not all
inside one reach. `:ols` loses draws to `:sem`, which nests it at `λ = 0`,
and to `:slx`, which nests it at `θ = 0`. `:sar` loses them to `:sdm` — the
same reach, one Durbin block apart — and, at `N = 100`, to `:sdem` in 10.0%
of draws, which is a crossing. `:sdem` loses 9.0% to `:sdm` at `N = 100`, the
mirror image: the two have identical designs and differ only in where the
dependence sits, so they are the hardest pair in the grid and the one the
reach columns are measuring. Every crossing in the table is between
specifications carrying the same Durbin block; at `N = 400` the worst of them
is down to 1.5%.

## The SDEM roadmap

`sdem_roadmap` is the local-spillover search of Sutton & Sutton (2023): for
each candidate `W`, decide which of `:sdem`, `:slx` and `:sem` fits best,
then read the winner across the candidates.

```julia
rm = sdem_roadmap(y, X, [W_rook, W_queen, W_knn4];
                  w_names = ["rook", "queen", "knn4"])
println(rm)
# rm.best_spec — the winner on each W
# rm.margin    — the gap to the runner-up on that W
# rm.best_w, rm.best_overall — the winning cell of the grid
```

The default criterion is the log marginal likelihood, because it is the only
one of the three on offer that integrates over the spatial parameter rather
than plugging in an estimate, and the specifications being compared differ in
whether they *have* that parameter. `:aic` and `:bic` are available and are
minimised rather than maximised; `sdem_roadmap` reads them in the right
direction, and the three need not agree.

`margin` is the point of returning a table rather than a winner. The choice
of `W` and the choice of specification are decided together, and a margin
near zero says the data do not separate them — which is a result to report,
not a tie to break by taking the maximum.

!!! note "The roadmap is the local case"
    `:sar` and `:sdm` put the dependence on `y`. Pass them through `specs` if
    you want them in the same table, but comparing across the two reaches is
    what `bayes_spec_probs` is for; the roadmap holds the reach fixed and
    searches within it.

## Choosing `k` for a knn `W`

Everything above compares specifications on a `W` you have already chosen.
`select_knn_w` chooses the `W` — specifically `k`, when `W` comes from
[`knn_weights`](@ref) — by refitting across a range of candidates and
minimising an information criterion.

```julia
sel = select_knn_w(coords, y, X; k_range = 2:25)   # X without an intercept
println(sel)                                        # the curve and its winner
m = fit_spatial(SAR(), sel.best_W, X, y)            # refit on the chosen W
```

The criterion is not monotone in `k`, which is why there is something to
search: a `W` that is too sparse misses real neighbours, and one that is too
dense dilutes them. On a 200-unit draw generated from a knn `W` at `k = 6`,
AIC falls from 312.7 at `k = 2` to 294.3 at `k = 6` and then climbs to 329.1
at `k = 20`.

`k` matters because `ρ̂` moves with it, monotonically and over a wide range —
0.305 at `k = 2`, 0.482 at the true `k = 6`, 0.630 at `k = 20`, against a
true 0.5 on that same draw. Too sparse understates the dependence; too dense
overstates it.

How often the search finds the generating `k`, over 300 draws with `k`
searched over `2:20` (so 5.3% by chance):

| `N` | `k̂ = 6` | within ±1 | within ±2 |
|---|---|---|---|
| 100 | 42.3% | 68.3% | 84.0% |
| 200 | 57.3% | 86.0% | 95.7% |
| 400 | 77.0% | 95.0% | 99.7% |

The median `k̂` is 6 at every `N`. Read that as the search being well
targeted but not sharp: at `N = 100` it misses the exact `k` more often than
it finds it, and the honest use of the result is the whole curve plus a
sensitivity check, not the argmin alone. `KnnSelectionResult` returns every
candidate's criterion for that reason.

!!! note "AIC is not penalising anything here"
    With the variant and `X` held fixed, the number of estimated parameters
    does not change with `k`, so AIC, BIC and the raw log-likelihood rank the
    candidates in exactly the same order. The criterion is reported so the
    numbers line up with [`spatial_aic`](@ref) elsewhere; its penalty only
    starts to matter once the comparison runs across variants as well.

!!! warning
    A candidate whose fit throws, or whose `ρ̂` lands on the edge of
    `rho_bounds`, is marked `converged = false`, given a `NaN` criterion and
    excluded from the argmin — a failed fit must not win by having no number
    to beat. The criteria are only comparable because `y` and the sample are
    identical across candidates, which `select_knn_w` checks rather than
    assumes.
