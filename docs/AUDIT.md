# Correctness audit

Status ledger for the systematic audit of every exported estimator and test.
Started 2026-10-23.

## Why this exists

Four separate correctness bugs surfaced in one week — two in the LM test
battery, one in the GWR hat-matrix trace, one set in the gravity estimator.
None was exotic, and the suite was green throughout. The common cause was the
same every time: **the tests checked signs, ranges and thresholds instead of
values against a known answer.**

Measured over `test/runtests.jl` as it stood at the start of the audit:

| | count |
|---|---|
| `@test` assertions | 510 |
| that pin a value (`isapprox` / `≈`) | 75 (15%) |
| that touch a standard error or `vcov` | 21 |
| …of those, that check anything but `size`/`length` | 1 (and it compares two internal code paths) |
| that touch a log-likelihood | 7 (all `isfinite` or `< 0`) |
| that check a p-value | by `0 ≤ p ≤ 1` or direction only |

Most of the 75 value-pinning assertions are Monte-Carlo recovery checks with
`atol` between 0.10 and 0.35. A tolerance that wide passes a materially wrong
formula: the GWR trace bug inverted the wrong factor and still recovered
every coefficient.

Testsets with **no** value-pinning assertion at all: 4, 5, 7, 8, 13, 14, 16,
17, 18, 20, 24, 25.

## Method

For each exported estimator and test, check the headline numbers — not just
the point estimates but the **variance, the log-likelihood and the test
statistic** — against one of:

1. a closed form derived independently,
2. a published number for a standard dataset,
3. a reference implementation written from the textbook formula without
   looking at the package source,
4. an exact but slow computation (dense linear algebra, a solved fixed point,
   a numerical Hessian) that the fast path is supposed to approximate.

A parameter-recovery check against the data-generating truth is **not**
sufficient evidence and does not close an item.

## Verified correct

Each row was checked against an independent reference, not merely reviewed.

| Item | Reference used | Result |
|---|---|---|
| `chebyshev_logdet` | dense `logabsdet(I - ρW)`, 81-cell rook lattice | error ≤ 0.013 nats at \|ρ\|=0.9 at the default order 6; ≤ 3e-3 at order 12. Adequate; default is defensible |
| `loglik`: SAR, SDM, SEM, SLX, SAC, SDEM | Gaussian likelihood written out separately, with the Jacobian terms supplied explicitly | all agree to ≤ 1.4e-14 |
| SAR `vcov` | exact expected information matrix, dense, no stochastic probes; plus a numerical Hessian | matches to ≤ 0.4% at N = 81…900. The residual gap to the numerical Hessian is the expected-vs-observed information distinction and shrinks with N as it should |
| `wald_test` | (Rβ−r)′(RVR′)⁻¹(Rβ−r) | correct |
| `common_factor_test` | delta method on θ + ρβ | Jacobian correct for the SDM coefficient layout |
| `make_rook_weights`, `make_queen_weights` | neighbour counts 2/3/4 and 3/5/8 at corner/edge/interior, row sums, pattern symmetry | correct. `make_queen_weights` previously had **no** test coverage of any kind |
| `knn_weights` | brute-force k nearest by Euclidean distance | neighbour sets match exactly for k = 3, 5 |
| `band_weights` | brute-force `d_min < d ≤ d_max` | pattern matches exactly |
| `distance_weights` | `w_ij = d_ij^-α` | max deviation 8.9e-16 |
| `vuong_test` | Vuong's ω̂ written out from per-observation log-densities on two non-nested OLS fits | **was wrong, now fixed** — see finding 1 below |
| `spatial_trace_powers` | dense `tr(Wᵐ)/N` for m = 1…12 | exact to 1.3e-15 |
| `g_traces` | dense `G = W(I−ρW)⁻¹`, at ρ = −0.6, 0.3, 0.8, with and without the constant projection | exact to 1e-13 in all six cases |
| `_pointwise_loglik` | sums to `loglikelihood(m)` for SAR, SDM, SEM, SLX, SAC, SDEM | exact to 1e-12 |
| SLX `vcov` | OLS covariance with the unbiased `s²` | **was wrong, now fixed** — finding 4 |
| SEM `vcov` | expected information matrix, written out densely | **structure was right, traces were not** — finding 5 |
| SDEM `vcov` | the error model's expected information in augmented regressors | **λ was fabricated, now fixed** — finding 6 |
| SAC `vcov` | numerical Hessian of the full log-likelihood, plus a 400-replication Monte Carlo | **ρ and λ were fabricated, now derived** — finding 7 |
| `estimate_lambda_gmm` (GS2SLS) | the Kelejian–Prucha moment conditions, and a 400-draw Monte Carlo | **two of three moments were misspecified** — finding 8 |
| GS2SLS `sargan_pval` | χ² survival function | **was the constant 1.0** — finding 9 |
| GS2SLS λ `vcov` | closed-form covariance of Gaussian quadratic forms, Monte-Carlo validated | **was a placeholder; now derived, with a documented gap** — finding 10 |
| `fit_spatial_regime` `vcov` | the SAR expected information | **ρ was fabricated, now derived** — finding 11 |
| `fit_dynamic_spatial_panel` | Monte Carlo at T = 8, 16, 32, 64 | **ρ̂ was inconsistent** — finding 12 |
| spatial panel `vcov` | numerical Hessian of the within-transformed likelihood, `IndivFE` and `TwoWayFE` | correct: ratios 0.993–1.024 |
| `fit_spatial_probit` | 250-replication Monte Carlo | correct: bias ≤ 0.013, coverage 90–97%. No change needed |
| `panel_lm_tests` | the closed-form score variances, and a 4000-replication Monte Carlo under the null | **the lag test was mis-scaled and the joint test added correlated scores** — findings 13, 14 |
| `spatial_chow_test` | a 1500-replication Monte Carlo under the null | **the degrees of freedom were off by one** — finding 15 |
| `spatial_hausman_test` | chosen covariance matrices, plus 250 replications on both fixed-effects pairings | **fabricated a statistic out of a noise direction** — finding 16 |
| `panel_moran` | 2000 replications on independent draws and 400 on fitted residuals | statistic correct; **the pooled variance field was inconsistent with the statistic beside it** — finding 17 |
| `spatial_eigenvectors` | the spectrum of `M W_sym M`, and the trace identity `Σλ = −S₀/N` | correct to 1e-14. No change needed |
| `fit_esf` | the unbiased OLS covariance on the same design, plus 400-replication coverage | **every standard error was short by `√((N−P)/N)`** — finding 18 |
| `fit_re_esf` | dense GLS, Woodbury, a REML grid, and 120-replication recovery of both variance components | correct to 1e-12 throughout. No change needed |

## Open findings

### 1. `vuong_test` computed the wrong statistic — FIXED

`omega` is meant to be Vuong's ω̂, the standard deviation of the
**per-observation** log-likelihood ratios m_i = ln f₁(yᵢ) − ln f₂(yᵢ):

    ω̂² = (1/n) Σ (mᵢ − m̄)²

The implementation never forms the per-observation contributions. It computes

    omega = sqrt(abs(2*(ll1 - ll2)/n - adj_diff^2) + eps())

from the aggregate log-likelihoods only. `2(ll1−ll2)/n` is twice the mean
log-likelihood difference and bears no relation to the variance of mᵢ. Three
symptoms, all reproduced on two non-nested SLX models (N = 144, where the
per-observation log-likelihood is unambiguous and sums exactly to the stored
`loglik`):

- ω̂ is wrong by a factor of 0.71 (correct 0.2292, reported 0.1634).
- the statistic is 0.9871 against a correct 0.7037, so p = 0.324 against a
  correct 0.482 — a factor of 1.5, enough to flip a conclusion.
- the quantity under the square root goes **negative** whenever model 1 fits
  worse; `abs()` hides that. As a result the statistic is not antisymmetric
  under swapping the arguments: +0.9871 and −0.9804, which should sum to zero.
  A correct Vuong statistic flips sign exactly.

The numerator and the √n scaling were right; only ω̂ was wrong.

The only existing assertion was `0 ≤ vt.p_value ≤ 1`, which the bug satisfies.

**Fix.** `SpatialModelSelection._pointwise_loglik(m)` returns the
per-observation contributions, and `vuong_test` takes ω̂ as their sample
standard deviation. The decomposition needs no per-model-type code: for
these Gaussian likelihoods the observation-wise part is written down from
`residuals(m)` and `σ² = e′e/n`, and the Jacobian is recovered as the
residual between that and the stored `loglikelihood(m)`, so the
contributions sum to it by construction. Verified: both properties hold
exactly for all six model types.

The Jacobian is then shared out evenly across observations. That is a
convention and is documented as one — `log|I − ρW|` is not separable across
observations, and under spatial dependence no split of it is. It cancels
from ω̂ entirely when the two models' Jacobians agree.

Covered by testset 13c, which pins the statistic to the independent
reference, pins its value, asserts exact antisymmetry, and rebuilds the old
ω̂ to assert it is still wrong by a margin that moves the p-value — so the
test guards the defect under any draw, not just this one.

### 2. Reported standard errors and impacts were not reproducible — FIXED

`compute_asymptotic_vcov` (SAR) uses two stochastic trace estimators:
`precompute_neumann_traces` with 250 Hutchinson probes for tr(G) and tr(G²),
and a 30-probe estimate of tr(G′G). Neither takes a seed from the caller and
both draw from the global RNG, so

- fitting the same model twice gives different standard errors (run-to-run
  spread 0.5% on the ρ standard error at N = 81, 0.4% on the intercept), and
- the fit silently consumes global RNG state, perturbing anything the user
  draws afterwards.

The point estimates were unaffected — ρ̂ was identical across eight fits.

**This was wider than the standard errors.** The same power traces are stored
on the model as `trace_powers` and read by `spatial_effects`, so the reported
**direct, indirect and total marginal effects** moved between runs too. For a
spatial model those impacts are the headline output, not a diagnostic. The
same defect was present in `fit_spatial_panel`, including its time-FE branch;
`fit_dynamic_spatial_panel` already threaded a seed.

**Fix.** Two helpers in `SpatialUtils`:

- `spatial_trace_powers(W; max_power, max_exact_n, n_probes, rng)` — exact
  from the eigenvalues of `W`, since `tr(Wᵐ) = Σᵢ λᵢᵐ` and one decomposition
  serves every power.
- `g_traces(W, ρ; project_constant, max_exact_n, n_probes, rng, μ)` — the
  three traces of `G = W(I−ρW)⁻¹` formed from `G` itself. `project_constant`
  removes the constant eigenvector for the panel estimators by projecting `G`
  on both sides, which no longer assumes `W` is row-standardised as the
  closed-form subtraction it replaces did.

Both are exact below `max_exact_n` (default 1000) and fall back to probes
drawn against a caller-supplied `rng` above it. `fit_spatial`,
`fit_spatial_panel` and `fit_dynamic_spatial_panel` gained `max_exact_n`,
`vcov_probes` and `rng` keywords.

Only `tr(GᵀG)` genuinely needed probing in the first place: it is not a
function of the eigenvalues of `W`, being invariant under neither similarity
nor transposition. The other two never did.

Cost of the exact path, measured: 0.14 s at N = 400 and 0.73 s at N = 1024,
against 0.04 s and 0.15 s for the probe path. Under a second at the default
gate, in exchange for exact and reproducible output.

Covered by testset 13d (70 assertions), which checks both helpers against
dense linear algebra, checks the SAR standard errors against the expected
information matrix written out with no probes anywhere — **the first
assertion in the suite to check a standard error against a value rather than
a size** — and asserts bit-identical variance, traces and impacts across
repeated fits, for `fit_spatial` and for `fit_spatial_panel` under both
`IndivFE` and `TwoWayFE`.

### 3. Robustness concerns, not bugs

- `common_factor_test` hardcodes coefficient positions (`beta_start = 2`,
  `rho_idx = n_params`). Correct for SDM, silently wrong for any other layout.
- `lr_test`, `wald_test` and `common_factor_test` all clamp the statistic at
  zero. That hides a convergence failure, which is exactly when the
  unrestricted log-likelihood comes out below the restricted one.
- `spatial_aic` / `spatial_bic` take k = `nobs − dof_residual`, which excludes
  σ². Consistent across models, so model *comparisons* are unaffected, but the
  absolute values differ from the usual convention by 2 and by log n.

### 4. SLX understated every standard error — FIXED

SLX is plain OLS, so its covariance is exactly knowable. It was formed as
`σ²(X′X)⁻¹` with the ML `σ² = RSS/N`, the value the log-likelihood needs,
rather than the unbiased `s² = RSS/(N−P)` — while `dof_residual` already
reported `N−P`, so the reported degrees of freedom and the variance
disagreed with each other. Every SLX standard error was low by
`sqrt((N−P)/N)`: 1.75% at N = 144 with P = 5.

Fixed by using `s²` in the covariance. The stored `sigma2` stays the ML
value, so the likelihood is untouched.

### 5. SEM's λ standard error was not reproducible — FIXED

The structure was right — in the error model β is asymptotically
independent of `(λ, σ²)`, so the information matrix is block diagonal, and
`I_λλ = tr(G² + GᵀG)` with `G = W(I−λW)⁻¹` is the standard result. The
three traces, though, came from 30 unseeded Hutchinson probes off the
global RNG, so **λ's standard error moved 8.5% between runs on identical
data** while β's, built from `X_f′X_f`, stayed put. The panel variant had
the same defect, and in the time-FE case mixed a projected estimate for two
traces with a closed-form subtraction for the third.

Both now use `g_traces`, which is exact below the size gate and handles the
constant projection uniformly. Verified against the expected information
matrix written out densely: agreement to 1e-8, and bit-identical across
repeated fits. There was also a dead line — `I_λλ` was assigned twice, the
first value discarded — now gone.

### 6. SDEM never computed a variance for λ — FIXED

### 7. SAC never computed a variance for ρ or λ — FIXED

These two share one defect, and it is the most serious the audit has found.
The covariance was assembled as a block-diagonal matrix in which the
spatial parameters' diagonal entries were filled with the literal constant
`1/N`:

```julia
V_full[P+1, P+1] = T(1) / N      # "variance" of rho
V_full[P+2, P+2] = T(1) / N      # "variance" of lambda
```

So every reported standard error for ρ in SAC and λ in SAC and SDEM was
exactly `1/sqrt(N)` — not derived from the data at all. Confirmed: at
N = 144 the reported value was 0.0833333 on three datasets whose ρ̂ ranged
from 0.076 to 0.363, and identical for ρ and λ in every case. **Every
z-statistic and p-value for the spatial parameters of SAC and SDEM was
meaningless.** The existing assertions were `length(stderror(m)) ==
length(coef(m))` and `size(vcov(m)) == (k, k)`.

SDEM is an error model in augmented regressors, so it now reuses the same
expected information as SEM, with `X` replaced by `[1 X WX]`.

SAC needed the joint information for `(ρ, λ)`, which is new code. Writing
`A = I − ρW₁`, `B = I − λW₂`, `X* = BX`, `G_A = W₁A⁻¹`, `G_B = W₂B⁻¹`,
`M = B G_A B⁻¹` and `c = M X*β`:

```
I_ββ = X*ᵀX*/σ²                 I_βρ = X*ᵀc/σ²          I_βλ = 0
I_ρρ = cᵀc/σ² + tr(MᵀM) + tr(M²)
I_ρλ = tr(MᵀG_B) + tr(M G_B)    I_λλ = tr(G_BᵀG_B) + tr(G_B²)
I_ρσ² = tr(M)/σ²                I_λσ² = tr(G_B)/σ²      I_σ²σ² = N/(2σ⁴)
```

`I_βλ` vanishes because `∂ε/∂λ = −G_B ε` has mean zero, the same reason the
pure error model is block diagonal. `I_βρ` does not, because the lag term
puts `Xβ` into `∂ε/∂ρ`.

Since this derivation is mine and not taken from a reference, it was
validated two independent ways:

- against a **numerical Hessian** of the full log-likelihood at N = 100,
  196, 324 and 576 — within 4–10% on the spatial parameters and 0.1–2.5% on
  β, bouncing either way, which is the expected-versus-observed information
  gap and the same band SEM's verified expected information sits in;
- against a **400-replication Monte Carlo** at N = 196, which shares none of
  the algebra. Reported standard error over sampling standard deviation:
  0.936 for ρ, 0.955 for λ, 0.976 for β₁, against a Monte-Carlo precision of
  about 3.5%. Coverage of the nominal 95% interval: 92.8%, 92.8%, 94.2%.

Mild optimism of that order is the known finite-sample behaviour of
asymptotic ML standard errors, not a formula error. For contrast, the old
`1/sqrt(196)` = 0.0714 against λ's true sampling standard deviation of 0.113
understated it by 37%, which would have given a nominal 95% interval an
actual coverage near 78%.

`fit_sac`, `fit_sdem`, `fit_spatial_error` (both variants) gained
`max_exact_n`, `vcov_probes` and `rng`. Covered by testset 15b.

### 8. The GS2SLS spatial error parameter was badly biased — FIXED

**The first finding that is not second-order.** `gmm_objective` minimised

    (ε′ε/N)² + (ε′Wε/N)² + (ε′W′Wε/N)²,   ε(λ) = u - λWu

but only the middle term is a valid moment condition. The Kelejian–Prucha
conditions are `E[ε′ε/N] = σ²`, `E[ε′Wε/N] = 0` and
`E[ε′W′Wε/N] = σ² tr(W′W)/N`; the first and third had their right-hand
sides dropped, so the objective drove the residual variance itself toward
zero. Measured at the true λ on a 196-cell lattice with σ² = 1, the two
mis-stated moments equal 1.000 and 0.276, not 0.

Consequences, over 400 draws:

| true λ | λ̂ mean | bias | correctly specified |
|---|---|---|---|
| 0.0 | −0.002 | −0.002 | −0.002 |
| 0.3 | 0.476 | **+0.176** | −0.009 |
| 0.5 | 0.771 | **+0.271** | +0.001 |
| 0.7 | 0.933 | **+0.233** | −0.007 |

Unbiased only at λ = 0, where the three conditions happen to agree. And
because λ̂ then filters `y` and `X` for the second stage, the bias carried
into ρ and β as well.

Fixed by concentrating `σ²` out of the first moment and scaling the other
two by it, leaving two scale-free equations in λ. After the fix, a
120-replication Monte Carlo of the full `fit_gs2sls_panel` at N = 100,
T = 6 gives biases of +0.0007 (ρ), +0.0008 and −0.0051 (β), −0.0010 (δ)
and −0.0073 (λ).

### 9. The Sargan p-value was the constant 1.0 — FIXED

`sargan_pval = one(T)`. The overidentification test could never reject, and
the docstring's instruction to "check `sargan_pval` before reading anything
else" was an instruction to read a constant. Now the χ² survival function
with `instruments − structural parameters` degrees of freedom; on a
correctly specified Monte Carlo it rejects 8.3% of the time at the 5% level
(120 replications, so about 1.6 standard errors from nominal).

### 10. GS2SLS λ had no variance — FIXED, with a known gap

The slot held `σ²/NT_post`, which involves neither `W` nor `λ̂`.

`λ̂` solves a two-moment problem in quadratic forms with fixed matrices
`A₁ = W` and `A₂ = W′W − (tr(W′W)/N)I`, so `Var(λ̂) = (G′ΩG)/(G′G)²` with
`G` the moment Jacobian. For Gaussian `ε` the moment covariance is exact,
`Cov(ε′Aε, ε′Bε) = 2σ⁴tr(AₛBₛ)`, and is computed in closed form from sparse
traces. That choice mattered: the obvious alternative, an outer product of
per-observation contributions, understates the variance by 30% because `Wε`
correlates those contributions — 83–86% coverage against 93.5–96.2% for the
closed form.

**Remaining gap.** The formula treats the first-stage residuals as the true
disturbances. Given true residuals the reported/sampling ratio is 1.02, but
inside a full fit it falls to 0.81 and coverage of a nominal 95% interval to
83%. That is documented at the call site as a lower bound. The first-stage
correction is still open.

### 11. The regime model's ρ had no variance — FIXED

`V_full[n_total, n_total] = T(1) / N`, the same placeholder as SAC and SDEM,
so `se(ρ)` was again exactly `1/sqrt(N)`. With regime dummies this is still
an ordinary spatial lag model, so it now uses the SAR expected information.
Its non-spatial branch also used the ML `σ²` in an OLS covariance, the same
defect as finding 4, and now uses the unbiased `s²`.

### 12. The dynamic spatial panel's ρ̂ was inconsistent — FIXED

The concentrated likelihood added `log|I − ρW|` **once**:

```julia
return const_term + logdet_fn(ρ) - T(0.5) * NT_post * log(σ2)
```

Each of the `T_post` periods contributes its own Jacobian, as the static
panel correctly has it (`const_term + T_opt * ld - …`). Counting it once
under-penalised large `|ρ|` by a factor of `T_post`.

This was not a finite-sample bias but an inconsistency, and the Monte Carlo
shows the difference plainly — the error does not shrink with `T`, so the
confidence interval tightens around the wrong value:

| T | ρ̂ (true 0.35) | bias | bias × T | coverage of nominal 95% |
|---|---|---|---|---|
| 8 | 0.397 | +0.047 | 0.38 | 73.8% |
| 16 | 0.409 | +0.059 | 0.95 | 43.8% |
| 32 | 0.408 | +0.058 | 1.85 | 21.2% |
| 64 | 0.409 | +0.059 | 3.76 | **1.2%** |

A residual bias of order 1/T would have shown `bias × T` flat; it grows
linearly instead. The Yu–de Jong–Lee correction was not the cause — the
uncorrected estimate showed the same thing. After the fix, biases of
−0.0038, +0.0020, +0.0011 and +0.0017 at T = 8, 16, 32, 64 with coverage
92.5–97.5%, and in the four-parameter Monte Carlo every estimate is within
0.008 of the truth with coverage 94.2–98.3%.

### 13. The panel lag LM test was mis-scaled — FIXED

`LM_lag` divided the squared lag score by

```julia
J = (Wy' * (I - X * ((X' * X) \ X')) * Wy) / σ²
```

which is wrong twice over. With `A = I_T ⊗ W` and
`τ = tr(A²) + tr(A′A)`, the score `d_ρ = e′Ay/σ̂²` has variance

    D = (AXβ̂)′M(AXβ̂)/σ̂² + τ

so the denominator should use the lag of the **fitted** values, and must
add `τ`. The code used the lag of the observed `y` — which carries the
disturbance as well as the signal, inflating the first term — and dropped
`τ` entirely. The two errors do not cancel: over 4000 replications under
the null at `N = 64`, `T = 5` (Monte-Carlo standard error 0.34% on a
rejection rate),

| | mean | variance | rejects at 5% |
|---|---|---|---|
| as shipped | 1.193 | 2.82 | **7.25%** |
| corrected | 1.022 | 2.07 | 5.17% |
| `χ²(1)` | 1.0 | 2.0 | 5% |

The over-rejection is 6.6 Monte-Carlo standard errors from nominal, and it
does not shrink: across `N` from 64 to 576 and `T` from 5 to 20 the shipped
statistic rejected at 5.8–9.5% and the corrected one at 3.5–7.5%. Power is
unaffected in any way that matters — 97.0% against `ρ = 0.15`, against
98.0% for the oversized version.

`LM_error` and `LM_re` were already right (5.40% and 5.08%), so this is the
one component of the battery that was wrong.

### 14. The joint panel LM test added correlated scores — FIXED

`LM_joint = LM_lag + LM_error + LM_re` with `df = 3`. Summing marginal
statistics is only a `χ²` if their scores are orthogonal, and the two
spatial scores are not: `Cov(d_ρ, d_λ) = τ`, the whole of `Var(d_λ)`. The
sum is therefore overdispersed — variance 7.84 against 6.0 for a `χ²(3)`,
and 6.38% rejection — even once finding 13 is fixed.

The spatial block now uses the quadratic form in the inverse information
(Anselin, Bera, Florax & Yoon 1996),

    LM_spatial = [τ d_ρ² − 2τ d_ρ d_λ + D d_λ²] / [τ (D − τ)]

which is `χ²(2)`. The random-effects score survives as a separate addend
because its cross-trace with the spatial scores vanishes:
`tr(J_T ⊗ W) = T · tr(W) = 0`, since `W` has a zero diagonal. The result is
mean 3.046, variance 6.30 and 5.47% rejection against `χ²(3)`.

### 15. The spatial Chow test's degrees of freedom were off by one — FIXED

`df_num = K`, `df_den = NT - 2K` for `K = size(X, 2)`. But each sub-period
fit is an OLS regression of `y` on `[X Wy]`, so it estimates `K` slopes
**and** a spatial lag coefficient: the break imposes `K + 1` restrictions
against `2(K + 1)` unrestricted parameters. One missing restriction is
enough to roughly double the size of the test. Over 1500 replications with
no break at `N = 64`, `T = 6` (Monte-Carlo standard error 0.56%):

| | rejects at 5% | rejects at 1% |
|---|---|---|
| as shipped, `K` restrictions | 8.5–10.5% | 1.8–2.7% |
| corrected, `K + 1` | 4.1–5.8% | 0.5–1.2% |

measured at `ρ = 0`, `0.4` and `0.7` and at `K = 2` and `3`. Power against
a break of 0.5 in one slope is 97.2% after the fix.

`Wy` is endogenous, so these sub-period fits are not consistent estimates
of a SAR and the `F` distribution is an approximation. The Monte Carlo is
the evidence that it is an adequate one at these sizes; that caveat is now
in the docstring, where it was not before.

### 16. The Hausman test fabricated a statistic out of a noise direction — FIXED

The worst-behaved thing found so far, and the only finding that was
*designed* rather than mistyped. The test inverts `V_fe − V_re`, which the
Hausman algebra needs to be positive definite. When it was not, the code
shifted every eigenvalue up to make it so:

```julia
if any(eig_vals .< -1e-8)
    V_diff_sym = Symmetric(V_diff + max(0.0, -minimum(eig_vals) + 1e-6) * I)
end
```

That is not a repair. It leaves the smallest eigenvalue sitting at the
shift, `1e-6`, so the inverse carries an eigenvalue of `1e6` and the
statistic becomes whatever the data happen to project onto a direction that
was pure noise.

It fires constantly, because two panel fits differing only in their fixed
effects have nearly equal variance matrices. At `N = 49`, `T = 5`,
`ρ = 0.4`, `IndivFE` against `TwoWayFE` — the pairing the package's own
test suite used — the two estimates differ by a standard deviation of 0.005
while each standard error is 0.037, so `V_fe − V_re` has entries of order
`−7e-5` whose sign is set by sampling noise. Over 250 draws with the null
true it was indefinite in **249**, and the test rejected at the 5% level in
**86.0%** of them, median statistic 71.6 against a critical value of 7.81.

Fixed by inverting only the directions whose eigenvalue is clearly
positive and reporting the count as `df` — the Moore–Penrose form of the
statistic — and returning `NaN` with `df = 0` when no direction survives.
The same 250 draws now give `NaN` in 160 cases and 6 rejections among the
remaining 90, with 68% power when time effects are genuinely present.

**What is not fixed, because it is not a coding error.** Two
`fit_spatial_panel` fits that differ only in their fixed-effects
specification do not generally satisfy the Hausman efficiency condition, so
the statistic has no `χ²` null distribution whichever way the arguments go.
`NoFE` against `IndivFE` is computable every time — an unmodelled
individual effect inflates `σ̂²` and so every standard error in the `NoFE`
fit, which makes the difference positive — but that inflation comes from
the misspecification rather than from an efficiency ordering, and it
rejects 25.6% of the time under the null. It does have power (100% when the
effects are correlated with `X`). Both pairings and both measured sizes are
now in the docstring under a warning, so a borderline p-value cannot be
read as if it were calibrated.

### 17. Panel Moran's pooled variance did not match its own statistic — FIXED

The statistic itself is sound. On independent draws the per-period `z` has
mean −0.015 and standard deviation 0.981, the combined `z` 0.945, and the
pooled test rejects 3.9% of the time at the 5% level; `E_I` matches
`−1/(N−1)` exactly. On residuals from a correctly specified `IndivFE` SAR
fit it stays centred at `ρ = 0` and at `ρ = 0.5` — so the spatial filter is
working — with the combined `z` running a little wide, standard deviation
1.07 and 6.5–7.2% rejection, because residuals from one panel fit share
`β̂` and the fixed effects and so are not the independent periods that
Stouffer's combination assumes. That is now in the docstring.

The defect was in the pooled row's fields. `I_stat` is the **average** of
the per-period statistics, but `Var_I` was copied from period one alone:

```julia
pooled = MoranResult(avg_I, pooled.E_I, pooled.Var_I, meta_z, meta_p, :panel_meta)
```

so the variance sat a factor of `T_per` above the spread of the quantity it
was reported beside — 8.55e−3 against a sampling variance of 1.81e−3 for
the average at `T = 5`, a ratio of 4.7. Anyone standardising `I_stat` by
`Var_I` by hand got a `z` too small by `√T`. Now `Var_I / T_per`. `E_I`
needed no change, since the mean of an average is the mean.

This one is a different species from findings 13–16: the test was right and
a reported field was wrong. It only matters because the struct invites the
arithmetic — and the pooled `z_stat` does not use `Var_I` at all, which is
now stated rather than left to be discovered.

### 18. ESF understated every standard error — FIXED

Finding 4 again, and this time it is not a rounding matter. The Gaussian
branch of `fit_esf` built its covariance as

```julia
σ2 = dot(res, res) / N
V = family === :gaussian ? σ2 .* inv(XtWX) : inv(XtWX)
```

while `dof_residual` is `N − length(coef)`. A Gaussian ESF fit is just OLS
on `[1 X E_selected]`, so the covariance wants `s²(X′X)⁻¹` with the
unbiased `s²`; dividing by `N` makes every standard error short by exactly
`√((N − P)/N)`. What makes it serious here, rather than the 1.75% it cost
SLX, is that `P` grows with the number of eigenvectors selected and
`max_eigenvectors` defaults to 20:

| N | eigenvectors selected | P | standard errors short by |
|---|---|---|---|
| 36 | 20 | 23 | **39.9%** |
| 64 | 20 | 23 | 20.0% |
| 64 | 30 | 33 | 30.4% |
| 144 | 20 | 23 | 8.3% |

`fit_re_esf`, in the same file, already divided by `N − P`, which is what
makes this an oversight rather than a deliberate convention.

After the fix the covariance equals the unbiased OLS one to 1e-10 on the
same design. Coverage of a nominal 95% interval for the covariate slopes,
over 400 replications at `N = 64`:

| | β₁ | β₂ |
|---|---|---|
| as shipped, SAR(0.6) errors | 88.2% | 87.2% |
| fixed, SAR(0.6) errors | 91.5% | 90.5% |
| fixed, no autocorrelation | 94.8% | 94.5% |

**What the fix does not buy.** With autocorrelation present, coverage is
still 91.5% and 90.5%, not 95%. That residual gap is post-selection: the
eigenvectors are chosen by a greedy search over the whole candidate pool on
the same `y`, and the standard errors then treat the basis as if it had
been fixed in advance. With no autocorrelation almost nothing is selected
and coverage comes out right, which is what identifies selection as the
cause. This is a property of the method, not a coding error, so it is now
documented with the measured numbers rather than changed.

### Verified with no change: the rest of the filtering module

`spatial_eigenvectors` is exact. The basis is the complete `N − 1`
eigenvectors of `Ω = M W_sym M` excluding the constant, orthonormal and
zero-sum to 1e-14, with `Ω e_i = λ_i e_i` to 1e-15 and a spectrum matching
a dense `eigvals(Ω)` including the zero. `MC_i = (N/S₀)λ_i` agrees with the
module's own `moran_coefficient` to 1e-15, and the reported Moran range
matches the theoretical extremes for the weights, `[−1.0044, 0.9443]` on a
row-standardised 7×7 rook lattice.

A pleasant identity falls out and is now pinned as a test: since `W` has a
zero diagonal and `1′W1 = S₀`, `tr(M W_sym M) = −S₀/N`, so the Moran
coefficients over the complete basis sum to exactly `−1` and average to
`−1/(N−1)` — which is `E[I]`. `fit_re_esf`'s reference Moran coefficient is
built on that identity and reproduces a direct average over the
non-retained directions exactly.

`fit_re_esf` is the best-built estimator the audit has examined. Checked
against dense algebra at `N = 64`: the Woodbury inverse
`(I + θEΛE′)⁻¹ = I − E diag(θλ/(1+θλ)) E′` to 3e-15, `log|V₀|` to machine
precision, `β̂` against an explicit GLS solve to 1e-15, `vcov` against
`σ²(X′V₀⁻¹X)⁻¹` to 1e-14, the BLUP `γ̂ = θΛE′V₀⁻¹r` to 3e-14, and the
profile log-likelihood to 1e-10. Its REML optimum is not merely close to a
4001-point grid over the same bracket — it is *better* than the grid's best
point, so Brent is converging properly rather than stopping early. Under
its own generating model over 120 replications it recovers both variance
components, `τ̂²` averaging 0.5017 against a true 0.5 and `σ̂²` 0.2526
against 0.25, with coverage 94.3% and 95.3%. Nothing changed.

## Queue

In priority order — unchecked variance first, since that is where the
suite is blindest and where three of the four bugs this week lived.

The variance matrices are closed, and so are the panel diagnostics — all
five functions in testset 14. What is left:

Spatial filtering is closed. What is left:

1. Dynamic system GMM — testset 5, zero pins. `fit_dynamic_gmm` is the one
   estimator in that queue item not yet reached.
2. The Bayesian and ordered probit samplers — testset 19's remaining
   estimators. `fit_spatial_probit` is verified; these two report posterior
   covariances from draws and have not been checked.
3. The GLM families of `fit_esf` — `:poisson`, `:logit` and `:negbin` go
   through `_fit_glm` and `_pearson_resid`, neither of which has been
   checked against an independent GLM. Only the Gaussian branch is
   verified.
4. Simulation (17), regimes (16) beyond the variance, tables (18),
   extension entry points (24).
5. The GS2SLS first-stage correction left open in finding 10.

Done: findings 1–18. Suite: 576 → 854 passing, none failing.

A pattern worth naming. Findings 1–7 all lived in *second-order* quantities
— a variance, a trace, a test statistic's scale factor — and the point
estimates were right throughout. Findings 8 and 12 broke that: a
misspecified moment condition biased GS2SLS's λ̂ by +0.27, and a Jacobian
counted once instead of `T_post` times left the dynamic panel's ρ̂
inconsistent. Both are the kind of error a recovery test is supposed to
catch — and testsets 3 and 4 had `atol = 0.15` on ρ and nothing at all on λ,
so neither did.

Four of the first twelve findings were the same shape: a variance slot
filled with a constant (`1/N` three times, `σ²/NT` once) so that the
reported standard error did not depend on the data. Worth grepping for that
pattern in any estimator added later.

Findings 13–16 are a second shape, and it is the one to watch for in the
remaining queue: a **test statistic whose null distribution was never
checked**. All four were off by a factor — 1.45× on the lag LM, 1.3× on the
joint, 2× on the Chow, 17× on the Hausman — and every one of them still
satisfied the entirety of testset 14, because the assertions were
`stat >= 0`, `0 <= p <= 1` and `df == 3`. A statistic that is twice its
nominal size passes all three. Any diagnostic added from here on needs a
null-distribution Monte Carlo before it is believed, and the two remaining
unchecked test statistics should be assumed mis-scaled until one is run.
`panel_moran` was the first to be put through that and came out correct
(finding 17), which is worth recording as evidence the rule is a prior and
not a certainty. The filtering module then came out almost entirely clean
too — `spatial_eigenvectors` and `fit_re_esf` needed no change at all. The
prior is holding at roughly one real defect per module, not one per
function, and the defects cluster in the *newest-looking* code rather than
the most intricate: `fit_re_esf` is by far the hardest thing in its file
and is the most correct.

Finding 16 is also the first case where the defect was deliberate. The
eigenvalue shift was written to stop an inversion from failing, and it
succeeded at that: it turned a visible error into a plausible-looking
number. Silencing a numerical failure is worse than propagating it, and the
three statistics clamped at zero noted in finding 3 are the same instinct.

Closing an item means: an independent reference exists, it is committed as a
test in `test/runtests.jl`, and the row above moves to "Verified correct".
