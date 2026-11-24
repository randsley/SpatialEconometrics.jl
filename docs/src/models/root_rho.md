# Root Estimator of ρ

[`root_rho`](@ref) estimates `ρ` in

```math
y = ρWy + Xβ + f + ε
```

where `f` is an unknown smooth nuisance, approximated in the span of a basis
`B` that you supply: for example, a tensor spline or hat basis on the
locations. Like [`closed_form_rho`](@ref), it needs no log-determinant: `ρ̂`
is a root of a quadratic. Unlike it, it stays unbiased when the basis is
large, and it reports an interval that stays valid when `ρ` is weakly
identified.

```julia
r = root_rho(y, X, W; basis = B)
r.rho, r.se, r.ci_wald, r.cs
```

The basis can also be penalised, as a P-spline: see
[With a penalised smoother](#With-a-penalised-smoother).

## The moment

Write `y₀(ρ) = y − ρWy` and `M = I − P_[X,B]`. Smirnov's closed form solves
`(My₀)′W(My₀) = 0`, a quadratic in `ρ`. At the true `ρ` its mean is
`tr(MWMΣ)`, which is not zero. It is of order `K/n`, where `K` is the
number of columns of `[X, B]`, so the estimator is consistent only when the
basis is small against `n`. A basis that approximates a free surface well
enough for valid inference is not small: at `K = 191` and `n = 900` the plain
closed form is biased by about −0.5 (`research/trace_cfe/`).

Jin & Lee (2012) replace `W` by a matrix with a zero diagonal:

```math
g(ρ) = y_0(ρ)'\,C\,y_0(ρ), \qquad C = (A − Δ)M,\quad
Δ = \mathrm{Diag}(AM)\,\mathrm{Diag}(M)^{-1}.
```

In Jin & Lee's own notation (2012, p. 451), the first step solves

```math
y_n'S_n'(ρ)\{W_n' − \mathrm{Diag}(W_n'M_{Z_n})[\mathrm{Diag}(M_{Z_n})]^{-1}\}M_{Z_n}S_n(ρ)y_n = 0,
```

and REsd uses ``G_n'(\hat ρ_n) − \mathrm{Diag}[G_n'(\hat ρ_n)M_{Z_n}][\mathrm{Diag}(M_{Z_n})]^{-1}``
in place of the braces. Their `S_n(ρ)y_n` is `y₀(ρ)` here, `M_{Z_n}` is `M`,
and the braces times `M_{Z_n}` are `C`.

Then `diag C = 0` and `C[X, B] = 0`, so `E g(ρ₀) = tr(CΣ) = 0` for any
diagonal `Σ`: the moment is exact under heteroskedasticity of unknown form,
whatever `K` is. `root_rho` takes two steps, as in Jin & Lee's REsd:

1. `A = W′` gives the first-step root `ρ̃` (`r.rho_first`);
2. `A = G(ρ̃)′`, with `G(r) = W(I − rW)⁻¹`, gives `ρ̂` (`r.rho`).

The second step has a smaller variance. Each root is the one at which
`g′(ρ) < 0`; the two roots of a quadratic have slopes of opposite sign, so
that rule picks exactly one.

**The rule assumes `J = −E g′(ρ₀) > 0`** (THEORY.md, A9). If `J < 0`, the
other root is the consistent one, and the rule's root stays about `|J/g₂|`
from `ρ₀` however large `n` is. `root_rho` checks the sign at a 2SLS
estimate (instruments `[X, B, W[X, B]]`) and warns if `g′` is positive
there, for the moment that gave `ρ̂`. `J < 0` is rare: 0.7% of 3,000
hub-and-spoke designs had it, and in 1,000 fits on kNN designs the check
never fired. In the hub design of testset 47g, the smoother's root is
biased by 0.06 (about 2 SD) and its Wald interval covers 56 of 257 draws;
`CS`, which does not use the rule, still contains `ρ₀`. REsd's second
step, with `A = G(ρ̃)′`, had `J > 0` even where the first step's was
negative, so on the projection path the warning is rarer still.

## Two intervals

**The Wald interval** is `ρ̂ ± z·se`, with `se = √V̂(ρ̂)/|g′(ρ̂)|`. `V̂` is
an HC2 variance (`ûᵢ²/Mᵢᵢ`) with a correction for the estimated
`P_[X,B]y₀` in the linear part of the moment.

**The test-inversion set** is

```math
CS = \{ρ : g(ρ)^2 ≤ z^2\,\hat V(ρ)\},
```

an Anderson–Rubin-type set. It needs the moment's CLT at the true `ρ` only,
not a non-vanishing slope, so it stays valid when the data say little about
`ρ`. Then it is wide, and it may reach a bound of `rho_bounds`
(`r.cs_touches_bound`) or split into more than one interval (`r.cs_split`).
`V̂(ρ)` is an exact quartic in `ρ`, so the endpoints are exact roots, not
grid points. **Report `CS`.** Where the two intervals agree, nothing is lost;
where they differ, the Wald interval is the one that under-covers.

If the moment has no admissible root at either step, `root_found` is false,
`ρ̂` and the Wald interval are `NaN`, and `CS` is still returned.

## Choosing the basis

The intervals are valid only when the approximation error of `f` is small
against the sampling noise. That needs a basis **larger** than the one that
fits `f` best:

- Do not choose `K` by GCV or by cross-validated fit of `f`. In the research
  Monte Carlo those choices over-smoothed for inference.
- Report `ρ̂` and `CS` across a few basis sizes, as `examples/34_root_rho.jl`
  does. A `ρ̂` that drifts as `K` grows is a sign that the smaller bases
  leave part of `f` in the residual.
- The coverage was measured for `K/n` up to 0.30; beyond that `root_rho`
  warns.
- A unit whose leverage in `[X, B]` exceeds 0.99 makes `Δ` ill-posed, and the
  fit is refused. Use fewer basis functions where the data are sparse.
- A hat or B-spline basis sums to one, so do not also put an intercept in `X`.

`β̂` is reported as a point estimate from least squares of `y − ρ̂Wy` on
`[X, B]`, **without standard errors**: the theory covers `ρ` only.

## With a penalised smoother

A penalty lets the basis be large without spending one degree of freedom
per column:

```julia
ps = tensor_pspline(u, v; k = 15)         # 225 hat functions and a difference penalty
r = root_rho(y, X, W; ps..., lambda = 0.1)
r.df                                      # tr S, the effective number of columns
```

The first stage is then `S = Z(Z′Z + λΩ̃)⁻¹Z′`, with `X` unpenalised, and
`M = I − S` is not idempotent. Two things change.

- **The moment.** It is the one-step exact-diagonal moment
  `y₀′Ay₀`, with `A = MW + MDM` and `(M∘M)d = −diag(MW)`, so `diag A = 0`.
  Jin & Lee's two-step is not used. Its plug-in result needs `Mf = 0`, and a
  smoother shrinks `f`, so `Mf ≠ 0`. `A` is formed as a matrix: the shortcut
  through residual vectors, `(My₀)′(MWy₀ + DMy₀)`, is a different moment when
  `M² ≠ M`, and it is biased.
- **The variance.** The HC2 plug-in is `ûᵢ²/(M²)ᵢᵢ`, and the noise term uses
  `H = (A + A′)S` (THEORY.md, Proposition 4(c)).

**The shrinkage is a bias.** At the true `ρ` the moment's mean is `f′Af`,
not zero. It grows with `λ`, and both intervals under-cover once
`c = f′Af/√V` is not small: the test-inversion set covers at about
`P(|Z + c| ≤ 1.96)`. So choose `λ` **small**, smaller than a good fit of `f`
needs. GCV and the RMSE-optimal `λ` were too large for valid inference in
`research/trace_cfe/tc_lambda.jl`, which is why `lambda` has no default.
Report `ρ̂` across a few `λ`, as section 4 of `examples/34_root_rho.jl`
does. Or use the bias-aware interval below, which stays valid at a
larger `λ`. `df/n` above 0.30 gives a warning, and a unit with `Sᵢᵢ` above 0.99
is refused.

## The bias-aware interval

Choosing `λ` small trades precision for validity. The bias-aware interval
avoids that trade under one assumption: a bound `ℛ` on the roughness of `f`.
Over the class `{f = [X, B]ψ : ψ′Πψ ≤ ℛ²}`, with `Π` the penalty, the
shrinkage bias satisfies `|f′Af| ≤ b̄ = ℛ√(v′Π⁺v) + ℛ²χ̄`, where
`v = (MZ)′Wf` is estimated from the projection and `χ̄` bounds the quadratic
part (THEORY.md, Lemma 4). The interval widens the critical value by that
bound:

```julia
ℛ = basis_roughness(f_ref, ps.basis, ps.penalty; X)   # a surface at least as rough as f
r = root_rho(y, X, W; ps..., lambda = 1.0, roughness = ℛ)
r.ci_bias_aware                                       # ρ̂ ± cv(b̄/√V̂)·se
r = root_rho(y, X, W; ps..., lambda = 10.0 .^ (-3:0.5:2), roughness = ℛ)
r.lambda, r.lambda_grid, r.halflength_grid            # λ chosen by the interval's length
```

`cv(t)` is the `level` quantile of `|N(t, 1)|`. Its coverage holds uniformly
over the class, at any `λ` (THEORY.md, Proposition 7). With a vector of
`lambda`s, the projection is added as `λ = 0` (bias bound 0), and the
shortest candidate is returned. Proposition 7 covers that choice when one
candidate is clearly the shortest. `bonferroni = true` covers any choice, at
the price of a larger critical value.

**`ℛ` is the user's assumption.** The data cannot certify an upper bound on
the roughness: a perturbation the data can barely see can be very rough.
Every rule that estimated `ℛ` under-covered (0.45–0.77) in
`research/trace_cfe/tc_rchoice.jl`. The loss is asymmetric, so err on the
large side. A large `ℛ` moves the choice towards the projection and costs
at most the gain; a small one costs coverage. In that design
(`n = 900`, two amplitudes `a` of the surface; projection half-lengths 0.250
and 0.227):

| `ℛ / ℛ*` | 0.5 | 1 | 4 | 16 |
|---|---|---|---|---|
| coverage, a = 1 / a = 4 | 0.908 / 0.902 | 0.954 / 0.964 | 0.982 / 0.970 | 0.970 / 0.968 |
| half-length, a = 1 / a = 4 | 0.107 / 0.125 | 0.117 / 0.152 | 0.162 / 0.210 | 0.230 / 0.226 |

Here `ℛ*` is the true roughness. Report the result across a few `ℛ`, with,
for each conclusion, the largest `ℛ` at which it holds.

### When `f` is not in the span of the basis

The class above assumes `f = [X, B]ψ`, so the projection gets `b̄ = 0`.
A real `f` leaves a remainder outside `span(B)`, and that biases every
candidate, the projection included. `reference` states the roughness class
on a finer basis, while the estimator keeps `B` (THEORY.md, Lemma 5 and
Proposition 7′):

```julia
ref = tensor_pspline(u, v; k = 30)                    # the class; K̃ may exceed n
ℛ̃ = basis_roughness((s, t) -> 2sin(2πs)cos(πt), u, v; k = 30)   # knot values, not least squares
r = root_rho(y, X, W; ps..., lambda = 10.0 .^ (-3:0.5:2), roughness = ℛ̃, reference = ref)
r = root_rho(y, X, W; basis = ps.basis, roughness = ℛ̃, reference = ref)   # the projection alone
```

The reference's unpenalised functions must be unpenalised in `[X, B]`:
the same coordinates and the same penalty order. `root_rho` checks this.
As the reference grid is refined, `(k̃ − 1)·ℛ̃` settles, so the class is in
effect a bound on the curvature of `f`.

In `research/trace_cfe/tc_refbasis.jl` (`n = 900`, `f = x + a·sin(2πu)cos(πv)`):

| knots in `B` | without `reference` (a = 4) | with `reference`, `ℛ̃*` (a = 1 / a = 4) | length against without (a = 1 / a = 4) |
|---|---|---|---|
| 4 | 0.050 | 1.000 / 1.000 | 1.8× / 10× |
| 5 | 0.717 | 0.990 / 1.000 | 1.4× / 5.7× |
| 7 | 0.923 | 0.983 / 1.000 | 1.1× / 2.6× |
| 15 | 0.973 | 0.960 / 0.980 | 1.00× / 1.08× |

With `ℛ̃` four times too large and 15 knots, the cost is 1.07× and 2.1×.
So: use a fine `B`, and the reference to check that it is fine enough.
Results across bases are comparable once the class is on the reference, and
the shortest interval was at the finest `B` tried. The projection candidate
(REsd) is covered by THEORY.md's Proposition 1′ when the largest remainder
`‖Mf‖` in the class is small against `√n`.

### Choosing the basis

With the class on the reference, the intervals of different bases bound
the same thing, so the data can choose the basis too. Give `basis` (and
`penalty`) as vectors, and the shortest interval over every (basis, λ) pair
is returned (THEORY.md, Corollary 7′):

```julia
pss = [tensor_pspline(u, v; k) for k in (5, 10, 15)]
r = root_rho(y, X, W; basis = [p.basis for p in pss], penalty = [p.penalty for p in pss],
             lambda = 10.0 .^ (-3:0.5:2), roughness = ℛ̃, reference = ref)
r.basis_index, r.basis_grid      # the chosen basis, and the basis of each candidate
```

`bonferroni = true` counts every pair. Without `reference` this is refused:
each basis would then have its own class, and a coarse basis would look
short only because its class is small.

## Cost

Everything is dense: about four `n × n` matrices. `n` is capped at
`max_dense_n = 4000` by default (about 0.5 GB); raise it if the memory is
there. With the largest basis of the Monte Carlo (`K/n` ≈ 0.19), one
replication (a `root_rho` fit plus a `closed_form_rho` fit) took 0.12 s at
`n = 900` and 0.35 s at `n = 1600`. The penalised smoother holds about six
`n × n` matrices and solves one dense `n × n` system. One fit took 0.10 s at
`n = 900` (`k = 15`) and 0.29 s at `n = 1600` (`k = 20`).

## What is proved and what is measured

Jin & Lee prove consistency and asymptotic normality for a design of fixed
dimension. The results for a basis that grows with `n` are in
`research/trace_cfe/THEORY.md`: a CLT for the moment with a dense `M`
(Theorem 1), the first step's plug-in having no first-order effect
(Proposition 1), the consistency of HC2 `V̂` (Proposition 2), and the
validity of `CS` without identification (Proposition 6). The penalised
smoother, its moment and its `V̂` are Proposition 4. **No one else has
checked these proofs.**

## What the Monte Carlo shows: the projection

`validation/mc_root_rho.jl` uses 108 cells of 1000 replications each. The
cells cross:

- `W`: kNN-6, rook, kNN-20;
- `n` = 400, 900, 1600;
- three basis sizes with `K/n` ≈ 0.05, 0.11, 0.19;
- smooth or rough variances;
- normal or standardised `χ²₃` errors.

Throughout, `ρ₀ = 0.4`. The Monte Carlo SE of a coverage near 0.95 is 0.007.

| | smallest | largest |
|---|---|---|
| coverage of `CS` | 0.935 | 0.975 |
| coverage of the Wald interval | 0.931 | 0.975 |
| `|bias|` of `ρ̂` | 0.000 | 0.060 |
| bias of `closed_form_rho` on the same draws | −1.76 | −0.05 |
| `|ρ̂ − ρ̂_research|` (first draw of each cell) | 0 | 2.5e-14 |

- **The plain closed form fails.** Its bias grows with `K/n` and with the
  density of `W`. At kNN-20 with `K/n` = 0.19 it is about −1.7: the estimate
  sits at the lower bound.
- **`root_rho` stays close to unbiased.** Its largest bias, −0.06, is in the
  weakest cell: kNN-20 at `n = 400` with `K = 82`, where the sd of `ρ̂` is
  0.30.
- **What the weak cells look like.** In the kNN-20 cells at `n = 400`:
  - up to 71 of 1000 samples had no admissible root;
  - `CS` reached a bound of `rho_bounds` in up to 740 of 1000 samples, and
    split into two intervals in up to 17;
  - `CS` still covered at 0.935–0.967.

  In these cells `CS` gives a wide set where the data say little about `ρ`.
  In every other cell, `CS` and the Wald interval agree closely.

## What the Monte Carlo shows: the smoother

`validation/mc_root_rho_smoother.jl` uses 108 cells of 1000 replications
each. The cells cross:

- `W`: kNN-6, rook, kNN-20;
- `n` = 400, 900, 1600, with `k` = 10, 15, 20 knots a side (`k²/n` = 0.25);
- `λ` = 0.01, 0.1, 1;
- `f = x + a·sin(2πu)cos(πv)` with `a` = 1 or 4;
- smooth variances with normal errors, or rough variances with
  standardised `χ²₃` errors.

Throughout, `ρ₀ = 0.4`. In each cell the size of the shrinkage bias,
`c = f′Af/√V`, is computed exactly, and with it the predicted coverage
`P(|Z + c| ≤ 1.96)`. The pass criterion, set before the run: in every cell
with `c ≤ 0.25`, both intervals cover within 0.93–0.99. In the other cells
the coverage is compared with the prediction.

**Results** (`validation/mc_root_rho_smoother_output.txt`, 310 minutes).
The pass criterion holds:
- **`c ≤ 0.25` (75 of 108 cells):** the Wald interval covers 0.935–0.982 and
  `CS` 0.934–0.982, and `|bias|` of `ρ̂` is at most 0.045. This includes the
  weakly identified kNN-20 cells at `n = 400`, where up to 62 of 1000 draws
  have no root and `CS` still covers 0.929–0.964.
- **`c > 0.25` (33 cells):** the coverage falls as predicted. `CS` is within
  0.027 of `P(|Z + c| ≤ 1.96)` in every cell, and the Wald interval within
  −0.007 to +0.077 of it. At `a = 4`, `λ = 1`, `c` reaches 8.2 and both
  intervals cover almost never: the bias is `f′Af`, not noise.
- **What sets `c`:** with `a = 1`, every cell at `λ ≤ 0.1` has `c ≤ 0.13`;
  with `a = 4`, `λ = 0.01` keeps `c ≤ 0.33`, but `λ = 0.1` already gives up
  to 1.8. So choose `λ` small enough that `r.df` stays well above the
  smoothness of `f`, and report results across a few `λ`. There is no rule
  for choosing `λ` yet.
- **Power:** the residual-vector shortcut, on the same draws, is biased by
  −0.015 to −0.244 in the `c ≤ 0.25` cells, and its Wald interval covers
  below 0.93 in 48 of them (down to 0.512).
- **The package against the dense reference:** `ρ̂` equal to 1.6e-14 and the
  SE to 6.4e-16, on the first draw of every cell.

## References

- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations. *Annals of
  Mathematical Statistics*, 20(1), 46–63.
- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators
  for spatial interaction in spatial autoregressive models. *Regional
  Science and Urban Economics*, 42(3), 446–458.
- Smirnov, O. A. (2022). A closed-form consistent estimator for linear
  models with spatial dependence. *Geographical Analysis*, 54(1), 3–31.
  https://doi.org/10.1111/gean.12268
- Stock, J. H., & Wright, J. H. (2000). GMM with weak identification.
  *Econometrica*, 68(5), 1055–1096.
