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
| ESF GLM families | `Distributions.logpdf`, and a ForwardDiff score and Hessian at the IRLS optimum | likelihoods and IRLS exact; **the negative binomial's dispersion was biased low** — finding 19, and now has its own MLE — finding 25 |
| `build_gmm_system` | the instrument structure in Roodman (2009), checked entry by entry | correct: lag layout, collapsing, level block and `H` all as specified. No change needed |
| `fit_dynamic_gmm` `vcov` | Windmeijer (2005) eq. 4.4, and coverage over `N` from 50 to 800 | **the correction's two dominant terms were mis-assembled** — finding 20 |
| `compute_ab_serial_correlation` | 300-replication Monte Carlo under a correctly specified model | correct: AR(2) standard deviation 0.97–1.06, rejecting 5.3–6.0%. No change needed |
| `hansen_j` | the quadratic form `g′W₂g`, and its rejection rate | correct: rejects 3.3–5.3% at the 5% level. No change needed |
| `_rho_logpost` | the dense `logabsdet(A) - 0.5*dot(r,r)` form, at six values of ρ | exact to 7e−15. No change needed |
| `_sweep_latent!` | its untruncated stationary distribution against `N(μ, H⁻¹)` | correct: mean and covariance recovered to 1.4% and 1.5% over 40000 sweeps. No change needed |
| `_rand_truncnorm` | `Distributions.truncated(Normal(…))` in four configurations | exact, including μ two SD outside the kept region. No change needed |
| Turing extension SAR and SEM likelihoods | the model densities, read against `log\|A\| + log N(filtered residual)` | both correctly specified, and `@addlogprob!` is the right vehicle. No change needed |
| `chebyshev_logdet` under the Bayesian prior | the exact log-determinant across the whole `Uniform(-0.99, 0.99)` support | accurate: error below 1e-3 for `\|rho\| <= 0.9`, worst case 0.17 at `N = 64` and falling to 0.026 at `N = 400`. No change needed |
| `fit_spatial_regime` design and recovery | the block structure entry by entry, and 250-replication recovery | correct: blocks exact and full rank, biases under 0.015, coverage 92.0-94.8% across all seven parameters. No change needed |
| `regime_chow_test` | the LR statistic by hand, and its size as `N` grows | statistic and df correct; both chi-squared(3) moments converge. Finite-sample over-rejection only, documented |
| `coeftable` p-values and intervals | the exact normal CDF and quantile | correct: relative error 3.6e-3 at `z = 6` and 1e-2 at `z = 8`, monotone, never negative; `_qnorm` round-trips to 1e-10. No change needed |
| `regression_table` | alignment across models with different parameter sets | correct: matches coefficients by name, blanks a name the model does not have. No change needed |
| the three exporters' escaping | LaTeX, GitHub Markdown and RFC 4180 rules | **all three emitted their format's metacharacters unescaped, and `digits` was ignored** - finding 23 |
| the six `simulate_spatial` DGPs | closed-form mean and covariance of `y` with `X` fixed, 60000 draws | all six correct: means within 1.8 Monte Carlo se, covariances within 1.3%. No change needed |
| `monte_carlo_spatial` summaries | the bias and RMSE identities, and seed reproducibility | correct, and the OLS diagnostics it reports are sized (0.053/0.048/0.057 at ρ = 0) with power 0.84 for `lm_lag` at ρ = 0.4. **But an estimator reporting too few parameters was zero-filled** — finding 22 |
| Gibbs β and cutpoint conditionals | the conjugate posterior, and Albert–Chib's order-statistic bounds | both as stated; `τ_1 = 0` and no intercept is the right identification. **But the accuracy claim in the docstring was wrong** — finding 21 |

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

**Remaining gap — now closed by finding 24.** The formula treated the
first-stage residuals as the true disturbances. Given true residuals the
reported/sampling ratio is 1.02, but inside a full fit it fell to 0.81 and
coverage of a nominal 95% interval to 83%. The first-stage correction is
implemented in finding 24 below.

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

### 19. The ESF negative binomial's dispersion was biased low — FIXED, and finished in finding 25

First, what was right. All four `_glm_loglik` branches match the
corresponding `Distributions.logpdf` to 1e-10, and `_fit_glm` reaches the
actual maximum rather than merely stopping: differentiating an
independently written log-likelihood at the returned `β̂` gives a score of
1e-13 and a Newton step of 1e-16, for all three families. `_pearson_resid`
uses the right variance function in each case, `μ` for the Poisson,
`μ(1−μ)` for the logit and `μ + αμ²` for NB2. The IRLS working weights
reproduce the observed Hessian to 1e-15 for the Poisson and the logit,
whose links are canonical.

For NB2 the log link is **not** canonical, so `X′WX` is the expected
information and differs from the Hessian by 62% in places. That is the
right matrix to use — the NB2 information is block diagonal in `β` and `α`
— and the discrepancy is not a defect. It is recorded here because it
looks alarming and is now pinned as a test in both directions: the
canonical families must agree with the Hessian, and NB2 must not.

The defect was in `_update_negbin_alpha`. For NB2,
`Var(y) = μ + αμ²`, so `((y − μ)² − μ)/μ²` has mean `α`; the code averaged
those terms over `N`. But `μ̂` is fitted, so the squared deviations are too
small and the average comes out low. Dividing by the residual degrees of
freedom instead — the usual Pearson dispersion convention — is strictly
better. At a true `α = 0.5` over 300 replications:

| N | 60 | 81 | 150 | 300 | 600 | 1500 |
|---|---|---|---|---|---|---|
| `/N` | −14.1% | −12.3% | −4.5% | −3.2% | −1.9% | +0.6% |
| `/df` | −11.2% | −10.1% | −3.2% | −2.5% | −1.6% | +0.7% |

Inside `fit_esf` at `N = 81` the estimate moved from 0.407 to 0.433.

**Why this was only a partial fix.** The remaining bias is intrinsic to
method of moments, not a degrees-of-freedom error — it shrinks with `N` and
is gone by `N = 1500`, so the estimator is consistent. A dispersion that is
too small makes the working weights `μ/(1+αμ)` too large and so the
standard errors too small: coverage of a nominal 95% slope interval at
`N = 81` is 91.6% after the fix, against 90.8% before and 95.2% for the
Poisson on the same design. Replacing method of moments with the
one-dimensional maximum likelihood estimate of `α` would remove most of
what is left, but the docstring cites Chun et al. (2016) for the
method-of-moments scheme, so that was a design change rather than a
correction.

**Finding 25 made that change.** `dispersion=:mle` is now the default and
the moment scheme is kept as `dispersion=:moments`. Two numbers in the
paragraph above do not survive the larger Monte Carlo run there and should
be read from finding 25 instead: the single-slope figures 91.6% and 95.2%
are 250-replication estimates, and at 6000 replications the Poisson's own
coverage on this design is 93.1-93.2%, not 95.2%. The table of mean biases
above is likewise seed-sensitive — see finding 25 — though the ordering
`/df` over `/N` it was reported to establish holds.

Coverage for the two canonical families needs no caveat: 93.2% and 95.2%
for the Poisson, 93.6% and 95.2% for the logit, over 250 replications at
`N = 81`.

### 20. The Windmeijer correction was mis-assembled — FIXED

Two-step GMM standard errors are badly biased downwards in small samples,
and the Windmeijer (2005) correction exists to repair that. The code
computed every ingredient correctly and then assembled them wrongly.

The formula, as Windmeijer states it in eq. 4.4 and Roodman (2009) in eq.
23, is

    V_c = V₂ + D V₂ + (D V₂)′ + D V₁ D′

with `V₂` the uncorrected two-step variance, `V₁` the robust one-step
variance, and `D` the derivative of the two-step estimate with respect to
the one-step estimate that enters the weight matrix. The code had

```julia
V_wind = V_naive .+ (D_mat * V_1 * V_naive) .+ (V_naive * V_1 * D_mat') .+ (D_mat * V_1 * D_mat')
```

— a spurious `V_1` inside both middle terms. All four matrices are `P × P`,
so it runs silently. But `V_1` is a variance, with entries of order `1e-3`
here, so multiplying by it shrinks the two terms that carry the whole
correction by about two orders of magnitude: `‖D V₂‖ = 1.1e-3` against
`‖D V₁ V₂‖ = 5.7e-6`. What is left is `V₂ + D V₁ D′`, barely above the
uncorrected variance the correction was supposed to repair.

`D` itself is right. It is obtained by differentiating
`−V₂ S_ZX′ W₂ Ω(θ) W₂ g₂` with ForwardDiff at `θ̂₁`, which is exactly
Windmeijer's `D` — a neat way to get it without hand-coding `∂Ω/∂θ`.

Standard errors came out 15–19% short. Coverage of a nominal 95% interval
for `ρ`, over 250 replications at `T = 6`:

| N | uncorrected | as shipped | correct |
|---|---|---|---|
| 50 | 77.2% | — | **92.0%** |
| 100 | 83.6% | 87.7% | 90.4% |
| 200 | 89.2% | — | 94.0% |
| 800 | 94.0% | — | 94.0% |

At `N = 100` the shipped version recovered 4 points of the 11 available;
the correct formula recovers 7. The gain is largest at small `N`, which is
the whole point — at `N = 800` corrected and uncorrected agree, as they
should.

Note what this does **not** fix: `ρ̂` is itself biased in short panels, by
−0.041 for `DifferenceGMM` and +0.023 for `SystemGMM` at `N = 50`, in
opposite directions, both falling to about ±0.002 by `N = 800`. No variance
matrix rescues that, and it is now in the docstring beside the coverage
table.

### Verified with no change: the rest of the dynamic GMM module

`build_gmm_system` matches Roodman entry by entry, checked on an explicit
`N = 8`, `T = 6` panel rather than by reading. The differenced equations
carry `Δy_it` on `Δy_{i,t−1}` and `Δx_it`; equation `t` is instrumented by
the levels `y_{t−2}` down to `y_1`; `H` is the tridiagonal `2`/`−1`
covariance of `Δε`; collapsing reduces the count from one column per
(period, lag) pair to one per lag, with column `s` holding lag `s` in every
equation; and the System GMM level block stacks `y_it` on `y_{i,t−1}`,
instrumented by `Δy_{i,t−1}`, with an identity `H` block and zero
cross-blocks. All of that is now pinned.

The three diagnostics are calibrated, which matters after findings 13–16.
Over 300 replications at `N = 100` with a correctly specified model:
Hansen J rejects 5.3% (`DifferenceGMM`) and 3.3% (`SystemGMM`) at the 5%
level; AR(2) has mean `−0.11`/`−0.03` and standard deviation 1.023/1.024,
rejecting 5.7%/5.3%; and AR(1) sits at `−5.4`/`−5.8`, rejecting hard, which
is correct rather than alarming — `Δε` is MA(1) by construction, so a
first-order test on differenced residuals *must* reject. The AR(2) standard
deviation stayed at 1.03 after finding 20 changed the variance matrix that
feeds it, so that coupling is sound too.

### 21. The Bayesian probit's accuracy claim was wrong — DOCSTRING FIXED

This one is not a coding defect. Every internal of both Gibbs samplers
verifies exactly, and they are in the table above. What was wrong is what
the docstring told the user about how far to trust the output.

It said the posterior "is still centred correctly — over 8 replications at
ρ = 0.4 the posterior mean averaged 0.374 at N = 196". Eight replications
cannot establish centring when the posterior SD is 0.10: the Monte Carlo
standard error on that average is about 0.035, so it was consistent with a
bias anywhere in ±0.07. Forty replications per size, at the default chain
length, give:

| N | ρ bias (mc se) | β₁ bias | ρ coverage | β₁ coverage |
|---|---|---|---|---|
| 36 | −0.161 (0.045) | +37.9% | 87.5% | 97.5% |
| 64 | −0.160 (0.038) | +23.6% | 80.0% | 92.5% |
| 144 | −0.059 (0.025) | +5.5% | 90.0% | 92.5% |
| 256 | −0.025 (0.018) | +3.1% | 92.5% | 97.5% |

At `N = 36` and `N = 64` the ρ bias is 3.6 and 4.2 standard errors from
zero. It is real, it is toward zero, and ‖β‖ is inflated in step with it —
which is the ridge the docstring correctly identified and then
under-stated. ρ coverage bottoms out at 80%.

The second error was the remedy: "Check `accept_rate` and lengthen the
chain before reading much into one ρ estimate." Lengthening the chain does
not touch the bias. At `N = 64`, 30 replications, identical data across all
three settings:

| draws / burn-in | β₀ bias | β₁ bias | β₂ bias | ρ bias | accept | ESS(ρ) |
|---|---|---|---|---|---|---|
| 1500 / 400 | +0.062 | +0.197 | −0.191 | −0.107 | 0.467 | 74.5 |
| 6000 / 1500 | +0.060 | +0.194 | −0.190 | −0.096 | 0.464 | 269.6 |
| 20000 / 5000 | +0.064 | +0.194 | −0.191 | −0.101 | 0.463 | 928.7 |

A 13× longer chain multiplies the effective sample size by 12.5 — so the
sampler mixes exactly as it should — and leaves every bias where it was.
The bias is in the posterior at that `N`, not in the Monte Carlo
approximation to it. The advice was aimed at the wrong quantity: a longer
chain does cut the noise in one chain's `coef` and `vcov`, which matters
because ESS(ρ) at the defaults is only about 100, but no chain length
recovers ρ.

Both docstrings now carry the measured tables, say that the bias is a
small-`N` property rather than a mixing failure, and point to
`fit_spatial_probit` when `N` is small and ρ is the quantity of interest —
its ρ bias runs the other way (+0.105 at `N = 36`, +0.044 at 64, +0.008 at
144, on the same data) and is the smaller of the two from `N = 64` up.

The ordered sampler behaves the same way, as it must: at `N = 64`, `J = 3`,
80 replications, ρ bias −0.031, β biases +0.096 and −0.068, and `τ_2` high by
+0.103 — the same proportional inflation as ‖β‖, since cutpoints and
coefficients share the latent scale.

Why it matters beyond the wording: a user reading "centred correctly" would
take ρ̂ = 0.26 at `N = 64` as noise around 0.4 and lengthen the chain. The
chain was never the problem.

### 22. A short coefficient vector was reported as zero — FIXED

`monte_carlo_spatial` compares an estimator's coefficients against the
DGP's true parameters row by row. It copied what the estimator gave it:

```julia
for j in 1:min(n_params, length(c))
    param_store[rep, j] = c[j]
end
succeeded[rep] = true
```

`param_store` starts as `zeros`, so when the estimator reports fewer
parameters than the DGP has, the trailing slots keep their zeros and the
replication is still marked successful. Nothing warns. Fit any
non-spatial estimator to spatial data — OLS against `SAR_DGP(0.4, …)`,
which is exactly the comparison a user runs to show what ignoring
dependence costs — and the table came back:

```
  β1   mean   1.05854  std   0.14312  bias   0.05854  rmse   0.15463
  β2   mean  -0.52242  std   0.14583  bias  -0.02242  rmse   0.14754
  ρ    mean   0.00000  std   0.00000  bias  -0.40000  rmse   0.40000
```

That last row is not a measurement of ρ. It is the absence of one, printed
in the same format as the two rows above it, with a bias of exactly
`-ρ_true` and a standard deviation of exactly zero. A reader could take it
for a finding about OLS — and it is even a *plausible-looking* one, since
“OLS estimates ρ as 0” sounds right until you notice no estimate was ever
made and the zero would be identical for any true ρ.

Unestimated rows are now `NaN`, which propagates into mean, std, bias and
RMSE alike, and the call warns once naming how many rows are affected.
The β rows are unchanged. Same class of error as finding 16's ridge: a
quantity that does not exist was given a numeric value instead of being
marked missing.

### Verified with no change: the rest of the simulation module

All six DGPs match their closed-form first and second moments with `X`
held fixed — `SAR` and `SDM` filtered by `A⁻¹` with covariance
`σ²(A′A)⁻¹`, `SEM` and `SDEM` by `B⁻¹` on the errors only, `SAC` by both
(`σ² A⁻¹B⁻¹B⁻′A⁻′`), `SLX` unfiltered — over 60000 draws, means within
1.8 Monte Carlo standard errors and covariances within 1.3% of the
largest entry. `σ` scales the standard deviation, not the variance.

`monte_carlo_spatial`'s own arithmetic is right: `bias = mean - truth`
exactly, `rmse = √(bias² + std²)`, a fixed seed reproduces the experiment,
and all-replications-failed raises rather than returning an empty table.
The three OLS diagnostics it reports are correctly sized — 0.053, 0.048
and 0.057 against a nominal 0.05 over 600 replications at ρ = 0 — with
power 0.695, 0.840 and 0.580 at ρ = 0.4, `lm_lag` strongest against a lag
alternative as it should be. That calibration matters more than usual
after findings 13–16, which were all mis-sized tests.

### 23. The table exporters did not escape, and ignored `digits` - FIXED

`to_latex`, `to_markdown` and `to_csv` each write into a format with its
own metacharacters, and each passed coefficient names straight through.
The names this package produces are not safe in any of the three:
`W_x1` from SLX, `R1_(Intercept)` from regimes, and the Greek `ρ` and `λ`
everywhere.

**`to_markdown` emitted a table that will not render.** Its header was

```
| Variable | Estimate | Std.Err | z | P>|z| | Sig |
|:---------|--------:|--------:|---:|------:|:---:|
```

The `P>|z|` contributes two further cell separators, so the header row
declares eight cells against the separator row's six. No renderer lines
that up. A coefficient name containing a pipe would shift its own row the
same way.

**`to_latex` emitted a document that will not compile.** A two-regime fit
produced six bare underscores, `R1_(Intercept)` and the rest, and `_` in
LaTeX text mode is an error, not a typo that renders oddly. The raw `ρ`
also needs math mode under pdfLaTeX. Producing compilable LaTeX is the
function's entire purpose.

**`to_csv` emitted broken CSV.** A name holding a comma split into two
fields, giving eight for a seven-column schema, with no quoting anywhere.

**And both formatted exporters took a `digits` keyword and ignored it.**
`to_latex` even built the format string and then never used it:

```julia
fmt = "%." * string(digits) * "f"    # dead
est = @sprintf("%.4f", ct.estimate[i])
```

so `to_latex(ct; digits=2)` and `to_markdown(ct; digits=2)` returned four
decimals, identical to the default. The signature promised control the
body did not provide.

All four are fixed: `_escape_latex` handles `_ & % # { } $ ~ ^` and the
backslash and maps the Greek letters to math mode, `_escape_md` escapes
pipes in cells and in the header, `_csv_field` quotes per RFC 4180, and a
runtime `_fmt` built from `Printf.Format` carries `digits` into every
numeric column of both formatted exporters.

How this survived: testset 18 asserted only that the output *contained*
`\begin{tabular}`, `name,estimate,std_error` and `| Variable |`. Each
exporter was checked for a marker and never for the structure it was
supposed to produce - so a table with mismatched column counts and a
LaTeX file that will not compile both passed. Testset 18b now counts
unescaped separators per row, parses the CSV back with an RFC 4180
splitter and checks the names round-trip, requires every underscore to be
escaped and no raw Greek to remain, and counts the decimals actually
emitted at two `digits` settings.

### Verified with no change: regimes, and the statistics behind the tables

`fit_spatial_regime` builds its block-diagonal design exactly - each
regime's columns zero outside that regime, its own intercept, its own
slopes - at full rank, and recovers its parameters: over 250 replications
at `N = 100` with two regimes, biases below 0.015 and interval coverage
92.0-94.8% across all seven parameters, which also confirms the
information-matrix variance fixed earlier is sound.

`regime_chow_test` computes `2(ll_r - ll_p)` with
`df = P_regime - P_pooled`, and both modules use the identical
`-½N(log 2π + 1) + log|A| - ½N log σ²` convention, so the two likelihoods
are comparable. Its size is the usual finite-sample LR over-rejection and
nothing more:

| N | size | mean statistic (df 3) | variance (6) |
|---|---|---|---|
| 100 | 0.068 (+3.2 se) | 3.241 | 7.03 |
| 196 | 0.048 | 3.001 | 6.05 |
| 400 | 0.042 | 3.002 | 5.81 |

Both chi-squared(3) moments converge, which is what rules out a wrong
statistic or wrong df. Power against genuinely different regimes is 1.000.

The normal CDF behind every `coeftable` p-value is the Abramowitz & Stegun
26.2.17 form, which factors out the density and so keeps its accuracy
*relative* far into the tail: 3.6e-3 at `z = 6`, 1.0e-2 at `z = 8`,
monotone throughout and never negative. `_qnorm` round-trips to 1e-10.
`SpatialUtils`'s copy clamps to exactly zero past `|z| = 8` where
`SpatialTables`'s keeps going, which is harmless either way.

`regression_table` matches coefficients **by name** across models, so a
parameter one model lacks leaves a blank rather than borrowing the figure
beside it - the positional error that finding 16's sibling fixed in the
Hausman test.

### Verified with no change: the Turing extension's likelihood

Both Bayesian models are correctly specified. `_sar_model` scores
`log|I - rho W| + log N((I - rho W)y - X beta | 0, sigma^2 I)` and
`_sem_model` filters the regression residual instead,
`log|I - lambda W| + log N((I - lambda W)(y - X beta) | 0, sigma^2 I)`,
which is the right distinction between a lag and an error process. The
comment explaining why the likelihood goes through `@addlogprob!` rather
than `~` is correct and worth keeping: a `~` on a local would make the
data one more latent variable.

The log-determinant is the one place an approximation enters a posterior,
so it deserved checking at the order the extension actually uses (12, not
the frequentist default of 6). Against the exact value across the full
`Uniform(-0.99, 0.99)` prior:

| N | max error | where | worst relative posterior tilt |
|---|---|---|---|
| 64 | 0.169 | `rho = -0.99` | 1.18x |
| 196 | 0.045 | `rho = -0.99` | 1.05x |
| 400 | 0.026 | `rho = +0.98` | 1.03x |

For `|rho| <= 0.9` the error is below 1e-3 at every size, and the error
*falls* with `N`. What is left is confined to the last one percent of the
prior's support, where a posterior with any data in it has negligible
mass. No change needed.

### 24. The GS2SLS first-stage correction — FIXED

Finding 10 left `λ`'s standard error short and documented it as a lower
bound, on the reading that the missing first-stage term was a
small-sample matter. That reading was wrong, and measuring it settled
the point. Over 250 replications at `λ = 0.25`, `ρ = 0.3`, `T = 6`,
reported standard error over sampling standard deviation:

| N | ratio | coverage |
|---|---|---|
| 64 | 0.852 | 92.0% |
| 144 | 0.820 | 88.8% |
| 256 | 0.825 | 89.2% |
| 484 | 0.864 | 91.6% |

A 7.5-fold increase in `N` leaves the ratio where it started, and the
same holds raising `T` from 6 to 24 at fixed `N`. Meanwhile `λ̂` itself
converges, its bias falling from −0.012 to −0.002. So this was not a
finite-sample artefact that washes out — it was a term missing from the
asymptotic variance.

Why the usual argument fails. `λ` is estimated from the first-stage 2SLS
residuals `u = ε − Zd` with `d = θ̂ − θ`, so the filtered residual is
`r = ε − Md` with `M = (I − λW)Z`, and each moment expands as

    m_j = ε′A_jε/NT + g_j′d + O(1/NT),   g_j = −2 M′A_jˢ r / NT

One would normally dismiss `g_j′d`: with `Z` exogenous, `Z′A_jε` is
`O(√NT)` and `d` is `O(1/√NT)`, so the product is `O(1)` against a
leading term of `O(√NT)`. But `Z` holds `Wy`, which is exactly what the
instrumenting is for, so `E[Z′A_jε]` is `O(NT)` and the term lands at the
same order as the leading one. That is the whole of the discrepancy.

The correction is one additive term, and the Gaussian structure is what
makes it tractable. `d` is to first order linear in `ε` with weights built
from the exogenous instruments, so `Cov(ε′A_jε, d)` is an odd third moment
and vanishes — no cross term survives:

    Var(m) = Ω + G_d V_d G_d′

`V_d` is the first-stage sampling variance, and it has to carry the
*spatial* covariance of the first-stage disturbance, `σ²(I_T ⊗ (B′B)⁻¹)`
with `B = I − λW`. A heteroskedasticity-robust diagonal is wrong there:
that disturbance is spatially correlated by construction, which is the
reason `λ` is being estimated at all. Written as `L(B′B)⁻¹L′ = C′C` with
`C = B⁻ᵗL′`, it costs `T_post × K` sparse solves against one
factorisation — no dense inverse, so it scales with the rest of the
module.

Result, same design and replication count:

| N | ratio before | coverage before | ratio after | coverage after |
|---|---|---|---|---|
| 64 | 0.852 | 92.0% | **0.936** | 92.8% |
| 144 | 0.820 | 88.8% | **0.897** | 92.8% |
| 256 | 0.825 | 89.2% | **0.901** | 90.8% |
| 484 | 0.864 | 91.6% | **0.944** | 94.8% |

and at fixed `N = 64`, raising `T`: 0.844 to 0.923 at `T = 12`, 0.866 to
0.945 at `T = 24`.

What remains is the `O(1/NT)` quadratic term `d′M′A_jMd/NT` that the
expansion drops, plus the cost of evaluating `G_d` and `V_d` at the
estimates rather than at the truth. The correction does not overshoot at
any size tested, so `λ`'s standard error is now mildly conservative
instead of optimistic — the right direction for an interval to err in.

Passing the first-stage matrices is optional, so calling
`_lambda_gmm_se` without them still returns the uncorrected value. That
path is what checks the moment algebra against known disturbances, where
there is no first stage to correct for, and testset 4b pins both.

### 25. The ESF negative binomial's dispersion now has its own MLE — DONE

Finding 19 left the dispersion on method of moments and recorded the
consequence rather than fixing it. `fit_esf` now takes
`dispersion=:mle` (the default) or `dispersion=:moments`, which is the
Chun et al. (2016) scheme the method was published with, kept because the
docstring cites it.

The MLE is the root of the profile score for `α` at the current `μ`,
written in `θ = 1/α`:

    s(θ) = Σᵢ [ ψ(yᵢ+θ) − ψ(θ) + log(θ/(θ+μᵢ)) + 1 − (θ+yᵢ)/(θ+μᵢ) ]

with `∂ℓ/∂α = −θ²s(θ)`, solved by Newton on `log θ` inside a maintained
bracket so a bad step falls back to bisection. As `α → 0⁺` the score is
`½Σ[(yᵢ−μᵢ)² − yᵢ]`, the Poisson overdispersion score; when that is
non-positive the likelihood is maximised at the Poisson limit and `α̂`
sits on the positivity floor, which is the honest answer there rather
than a negative variance.

Checked against independent references, not merely reviewed:

- `ψ` and `ψ₁` against `ψ(1) = −γ`, `ψ(½) = −γ−2log2`,
  `ψ(m) = −γ + Σ_{k<m} 1/k`, `ψ₁(1) = π²/6`, `ψ₁(½) = π²/2`,
  `ψ₁(m) = π²/6 − Σ_{k<m} 1/k²`, and both recurrences across the switch
  from recursion to the asymptotic series: 1e-13 and 1e-12. The first
  version truncated the trigamma series at `x ≥ 10`, which left 7e-13 —
  visible against the closed form, so the series was pushed to `x ≥ 20`.
- the score against `ForwardDiff` on the `Distributions` NB2
  log-likelihood differentiated in `α`: 1e-10 at `α = 0.2, 0.5, 0.6, 1.3`.
  The reported derivative in `θ` against a central difference of the
  score: 1e-6.
- `α̂` is not merely close to the best point of a 2001-point grid over
  `[0.05, 2]` — it is better, and the score vanishes there to 1e-8.

**What it buys.** Mostly precision, not bias. At a true `α = 0.5`, over
3000 replications per cell — three independent 1000-replication runs
pooled, on the two-column design of finding 19, both estimators run on
the same samples:

| N | 60 | 81 | 150 | 300 | 600 | 1500 |
|---|---|---|---|---|---|---|
| `:moments` mean bias | −11.5% | −8.5% | −5.0% | −2.3% | −1.1% | −0.4% |
| `:mle` mean bias | −7.3% | −5.7% | −3.6% | −1.1% | −1.0% | −0.2% |
| `:moments` median bias | −18.2% | −16.2% | −9.3% | −5.0% | −2.9% | −0.8% |
| `:mle` median bias | −11.5% | −8.7% | −4.6% | −1.8% | −1.2% | −0.5% |
| `:moments` rmse | 0.309 | 0.285 | 0.204 | 0.146 | 0.109 | 0.067 |
| `:mle` rmse | **0.229** | **0.199** | **0.140** | **0.101** | **0.071** | **0.046** |
| rmse ratio | 0.74 | 0.70 | 0.69 | 0.69 | 0.65 | 0.68 |
| `:mle` the closer of the two | 62.6% | 64.9% | 64.5% | 65.1% | 68.2% | 66.5% |
| `:moments` on the floor | 4.7% | 2.4% | 0.2% | 0.0% | 0.0% | 0.0% |
| `:mle` on the floor | 0.7% | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% |

The rmse improvement is the stable part: a third off at every size
tested, and the MLE is the closer of the two on about two samples in
three. Both estimators are consistent and the bias rows mostly restate
that, but read the *median* rows rather than the mean ones — `α̂` is
right-skewed, and the mean moves several points between seeds. At
`N = 81`, across five independent 1000-replication runs, the moment
estimate's mean bias ranged over −3.7% to −9.8% while its rmse stayed
within 0.269–0.288. That is why finding 19's −10.1% at this `N` does not
reproduce at a different seed, and why it is not evidence of much.

Coverage of a nominal 95% slope interval at `N = 81`, on finding 19's
design, over 6000 replications:

| | β₁ | β₂ |
|---|---|---|
| Poisson | 93.1% | 93.2% |
| logit | 94.5% | 94.7% |
| `:mle` | **93.3%** | **92.6%** |
| `:moments` | 92.2% | 91.6% |

So the dispersion is no longer what holds the negative binomial back: on
`:mle` it covers as well as the Poisson does on the same design. Two
further notes on that. First, the 95.2% quoted for the Poisson in
finding 19 was one slope at 250 replications; at 6000 the two slopes come
out at 93.1% and 93.2%, so the target the negative binomial was short of
was never 95.2% — it was 93%. The headline of finding 19, "91.6% against
95.2% for the Poisson", overstated the gap for that reason, and the
"91.6% toward 95.2%" the queue was holding this item for was not an
attainable 3.6 points. The honest version is 1 point of coverage and a
third off the dispersion's rmse. Second, what is still missing at 95% is
the post-selection effect of finding 18's docstring note, which every
family pays: the mean standard error is 0.92 of `β̂`'s actual sampling
standard deviation for the Poisson, the logit, `:moments` and `:mle`
alike.

**Where the coverage gain comes from.** Running both schemes on each of
the 6000 replications and splitting on whether the moment estimate landed
on the positivity floor — where the working weights revert to the
Poisson's `μ` and the standard errors are far too small:

| | fits | `:moments` β₁ | `:mle` β₁ | `:moments` β₂ | `:mle` β₂ |
|---|---|---|---|---|---|
| all | 6000 | 92.2% | 93.3% | 91.6% | 92.6% |
| `:moments` on the floor | 334 | 74.0% | 84.7% | 79.3% | 88.0% |
| `:moments` not on the floor | 5666 | 93.2% | 93.8% | 92.3% | 92.9% |

So it is about half and half, not mostly the floor. The floor subset
carries 0.60 of the 1.15 points gained on β₁ (334 fits improving by 10.7
points) and 0.48 of the 0.97 on β₂; the other half comes from the 5666
ordinary fits improving by half a point each. The floor is still where
the individual fits are worst — reporting
`α̂ = 1e-6` against a true 0.5 costs 13 points of coverage on β₂ and 19
on β₁, measured against the same scheme's own off-floor fits — and the
MLE recovers 9 to 11 of those points without eliminating the problem,
since it lands on the floor for 0.7% of these fits itself.

Note also that the floor is much commoner inside `fit_esf` than the
two-column table above suggests: 5.6% of fits here against 2.4% at the
same `N`, because this design carries ten-odd columns and a weaker
signal per column.

Cost: 17% more time per negative-binomial `fit_esf` than the moment
update — 144s against 123s over the same 6000 fits, draws included.
Newton converges in a handful of iterations from the previous `α` as a
start, so it does not change the cost class of the greedy selection loop.

## Queue

Closed: the variance matrices, the panel diagnostics (all five functions
in testset 14), spatial filtering (all five functions, all four
families), dynamic GMM, the discrete-choice samplers, simulation,
regimes, the tables and their exporters, the extension entry points and
the Turing extension's likelihood, and — with finding 24 — the GS2SLS
first-stage correction that finding 10 left open.

Every module the audit set out to cover is now closed, and the one
optional improvement the queue was holding — `fit_esf`'s
negative-binomial dispersion — is done as finding 25. Nothing is
outstanding.

Done: findings 1–25. Suite: 576 → 1156 passing, none failing.

Post-audit, under the standing rule below: 1270 passing on Julia 1.13 with
both extensions, 1229 on Julia 1.10 where the extensions are absent.

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

## New diagnostics under the standing rule

The audit left one standing rule: *any diagnostic added from here on needs a
null-distribution Monte Carlo before it is believed.* This section records
diagnostics added after the audit closed and what was measured for each,
so the rule has somewhere to be satisfied.

| Added | Diagnostic | Null distribution measured | Verdict |
|---|---|---|---|
| 2026-10-23 | `sdm_tests` — the five Koley & Bera Durbin score statistics | 20000 replications, `N = 81`: mean = df to within 2.6%, variance = 2df to within 1.9%, size 0.047–0.054 at the 5% level and 0.009–0.010 at the 1% | Verified correct |
| 2026-10-24 | `select_knn_w` — choosing `k` for a knn `W` by AIC | not a test statistic, so the gate is the selection error rate: 300 draws generated at `k = 6`, `k` searched over `2:20`, exact recovery 42.3% / 57.3% / 77.0% at `N = 100 / 200 / 400` against 5.3% by chance, median `k̂ = 6` throughout | Verified correct |

### `sdm_tests` — four checks, in testset 27

1. **Reduction.** `RS_rho` is algebraically the existing `lm_test_lag`, and
   agrees with it to 0.00e+00. The per-term formula applied at the `rho`
   position reproduces `lm_test_robust_lag_durbin` to 2.22e-16, confirming
   the block-inverse identity the robust forms rest on.
2. **Scores against ForwardDiff.** The SDM Gaussian log-likelihood written
   independently in the test file, differentiated at the OLS null: `d_rho`
   agrees to 1.33e-15, the two `d_theta` to 1.33e-15 and 4.44e-16, and the
   nuisance scores vanish (1.34e-14 for `beta`, exactly 0 for `sigma2`),
   which is what makes these the right scores to be testing.
3. **Null distribution.** The table above.
4. **The robust property.** Under SAR data with `theta = 0` exactly, the
   plain `durbin` statistic rejects 16.1% of the time at the 5% level while
   `robust_durbin` rejects 5.1%. Under SLX data with `rho = 0` exactly,
   `lm_test_lag` rejects 7.0% while `robust_lag` rejects 5.2%. This is the
   sharpest of the four: the correction uses `J_rho_theta`, so a robust
   statistic cannot land on its nominal size if the cross-information block
   is wrong.

One methodological note, because it cost an hour. Check 3 was first written
as *the information matrix equals the Monte Carlo covariance of the score*,
which is the textbook identity. It failed by 37%, with the `theta` diagonal
running about a third low. The identity holds at the true `sigma2`; the
implemented score divides by `sigma2`-hat, whose own variability at `N = 81`
inflates the covariance of the score, while leaving the *statistic*
correctly scaled because the same `sigma2`-hat appears in `d` and in `J`.
Comparing the matrices is therefore the wrong check at finite `N`. The
property that matters, and the one the committed test makes, is that the
statistic is chi-squared: mean = df, variance = 2df, correct size in the
tail. Findings 13–16 were all scale errors, and the mean of the statistic
is what catches a scale error.

### `select_knn_w` — four checks, in testset 28

`select_knn_w` is a selection procedure rather than a test, so the rule was
read across to the quantity that plays the same role: its error rate. The
numbers above are the headline, and the one that matters for interpretation
is what a wrong `k` does to `rho-hat` — over 300 draws at `N = 200`, mean
`rho-hat` runs 0.325 at `k = 2` (−35.1% bias) through 0.493 at the true
`k = 6` (−1.4%) to 0.646 at `k = 20` (+29.1%), while selecting leaves it at
0.502 (+0.5%). Paired against a fixed choice, selection gives the smaller
`|rho-hat − rho|` in 98.3% of draws at `k = 2`, 69.0% at `k = 10` and 93.0%
at `k = 20`. The figure usually quoted for this effect is "up to 20%"; the
wider range here is a property of searching out to `k = 20`, not a
disagreement with the source, and the docstring says so rather than quoting
the tidier number.

A second note. The non-monotonicity of the criterion in `k` is the premise
of the whole exercise, so it is asserted rather than assumed: the committed
test checks that the AIC curve is neither increasing nor decreasing
throughout and that its minimiser is interior to the searched range. With
the variant and `X` held fixed the parameter count does not vary with `k`,
so AIC, BIC and the raw log-likelihood rank the candidates identically here
— the test pins that too, so the `criterion` keyword is not mistaken for
doing work it does not do in this search.

## A whole class of fragile assertion: pinning a random draw

CI went red on Julia 1.10 and on macOS/aarch64 while passing on Julia 1
under Linux and Windows and on nightly. Eight assertions failed, and all but
one had a single cause.

**`randn(::MersenneTwister, n)` is not reproducible across Julia minor
versions.** Julia guarantees a stream only within a minor release. The same
`MersenneTwister(1_300_014)` and the same calls give `sum(X) = −19.63` on
1.13 and `−32.82` on 1.10. Every assertion that pins a number which a
particular stream produced is therefore version-fragile by construction,
however carefully the number was measured.

Four assertions were of that kind, and two of them were already unsound on
the version they were written on:

| Assertion | What was wrong |
|---|---|
| `13c` `vt.stat ≈ 1.87685263` | The literal is one stream's value; the statistic came out 0.114 on 1.10. The test also recomputes `z_ref` independently and checks against it at `rtol = 1e-10`, which passed on **both** versions. That is the assertion carrying the finding; the literal added fragility and no coverage. Removed. |
| `19b` `0.15 < accept < 0.6` | Not only version-fragile but seed-fragile on 1.13 itself, which reaches 0.630 at seed 4202 and 0.623 at 4203. Measured range over ten seeds on both versions is 0.200–0.664, so the band is now 0.15–0.75 and is applied to every chain. |
| `19b` `e_large < e_small` | Holds in 8 of 10 seeds on 1.13 and 6 of 10 on 1.10 — it passed on seed 4200 by luck. "Bias shrinks with `N`" is a claim about the sampling distribution, so it is now the median over ten seeds: 0.911 → 0.153 on 1.13, 0.198 → 0.127 on 1.10. |
| `21c` NB2 Hessian gap `> 0.1` | The claim is the contrast with the `< 1e-10` asserted for the canonical links. Measured over five seeds on two versions the gap runs 0.026–0.616, so `0.1` sat inside its own spread and failed at 0.076. Now `> 1e-3`, twelve orders clear of the canonical case. |

Two more were mine, from the Durbin and knn work in this same session, and
show the rule is easy to break while writing a careful test:

- `28b` asserted `best_k == 6` on one draw, when `28c` three screens away
  measures that at 57% of draws. The assertion contradicted its own
  neighbouring measurement. Removed: a recovery claim belongs to a rate.
- `28d` asserted `rho_hat(k=20) > 0.55`, which is a statement about where one
  draw's sequence happens to sit. The ordering `rho(2) < rho(6) < rho(20)` is
  the actual property and is scale-free, so that is what is asserted now.

**The rule this yields.** An assertion may pin a closed form, an algebraic
identity, a structural reference recomputed in the test, or a *rate* with a
margin wide enough for its own sampling error. It may not pin a value that
came out of a random stream, and a margin must be set from a measured spread
across seeds, not from one run. The null-distribution gate already follows
this; these four did not.

A corollary for the CI matrix: the suite must pass on the oldest supported
Julia as well as the newest, and the two have different random streams, so
the matrix is what enforces the rule. Running the suite only on the
development version cannot catch this class.

## One real defect, found the same way

`27a` was not a fragile assertion. `lm_test_durbin_terms` gated singularity
on `issuccess(cholesky(Symmetric(J); check = false))`, which is **not a rank
test**: for an exactly rank-deficient Gram matrix — a repeated column in `X`,
or a `W` for which `W*X` lands in `span(X)` — whether the final pivot comes
out slightly positive or slightly negative is decided by the rounding of the
particular LAPACK build. It came out positive on macOS/aarch64 and on the
1.10 dependency set, so the factorisation reported success and the inverse
returned large meaningless numbers. Finding 16's failure mode, reached by a
new route, in code written to prevent it.

Replaced by a gate on the ratio of the Cholesky pivots: a ratio at or below
`sqrt(eps)` means a condition number near `1/eps`, which is singular by any
standard a score test cares about, while a sound block sits near 1. All three
entry points now throw on the duplicated-column design on both versions.

