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
| 2026-10-30 | `bayes_spec_probs` / `sdem_roadmap` — posterior model probabilities over specifications and `W` | not a test statistic, so the gate is the selection error rate: 200 draws from each of six specifications, all six compared on the generating `W`, recovery 81.5–95.5% at `N = 100` and 92.5–100% at `N = 256` against 16.7% by chance; the global-versus-local reach right in 90.5–100% of draws | Verified correct |
| 2026-10-30 | `fit_spatial_did` / `vcov_cluster` — spatial difference-in-differences and its covariance | two gates, since neither is a test statistic. Coverage of a nominal 95% interval, 2,000 reps, errors `(I − 0.6W)^{-1}ε`: clustering 95.0%, OLS 96.5%, Conley 94.5% at cutoff 1.5 falling to 85.3% at 6.0 — and falling identically at `ρ = 0`, so the decay is finite-sample bias, not the correction working. The estimator itself is exact where it can be: `5.3e-15` on a noiseless two-period design. Staggered adoption under heterogeneous effects is biased 17–67% with an unpredictable sign, measured across four designs and pinned | Verified correct; staggered caveat documented |
| 2026-10-30 | `stpca` — spatio-temporal principal components | a decomposition, not a statistic, so the gate is again the error rate: 200 draws per cell with a known field planted in one feature, `:moran` puts a smooth field first in 73–87% and a high-contrast field last in 72–86%, with 0 of 800 draws at the wrong end; the eigenvalue identity `λ = Var(U)·I(Aγ)` is checked against independently implemented statistics to 5.0e-14 | Verified correct |
| 2026-10-31 | `fit_synthetic_control` / `placebo_test` — spatial synthetic control and its permutation inference | two gates. The p-value is a test statistic: no effect anywhere, 7×7 lattice, 44 clean donors, exact permutation, 1,000 reps — size 0.046 / 0.095 at nominal 5% / 10% at `ρ = 0`, 0.045 / 0.094 at `ρ = 0.6`, and 0.068 / 0.125 at `ρ = 0.9`, so it is correctly sized up to moderate spatial correlation and over-rejects by about a third of nominal at strong correlation. The estimator is checked against the exact optimum by support enumeration rather than a tolerance: objective to 3.55e-15, weights to 1.31e-14. Donor-pool contamination is an algebraic identity, pinned to 1e-12 | Verified correct; `ρ = 0.9` over-rejection and the p-value granularity floor documented |
| 2026-10-31 | `fit_shift_share` / `balance_test` — shift-share designs, the AKM standard error and the shock-level balance test | two gates. Coverage of a nominal 95% interval, 2,000 reps, shares and residuals fixed with the shocks resampled: AKM 0.936 against HC1 0.732 at 100 effective shocks with a strong industry component, and HC1 rejecting a true null in up to 71% of samples — the failure the estimator exists for, reproduced. Clustering on each region's largest industry tracks HC1 (0.720) rather than fixing it. AKM itself falls to 0.864 at 20 effective shocks and 0.827 at 38 with concentrated exposure. The balance test is a test statistic: size 0.056 / 0.108 at nominal 5% / 10% at 100 effective shocks, 0.109 at 20 and 0.232 at 9, and the joint test over four characteristics reaches 0.324 at 20. Three published identities are pinned at machine precision rather than by tolerance: the Borusyak-Hull-Jaravel shock-level equivalence to 2.3e-13, the Goldsmith-Pinkham-Sorkin-Swift decomposition to 2.6e-13, and the reduction of the AKM variance to a cluster-robust one under full specialisation to 3.0e-15 | Verified correct; the few-shock degradation of both the interval and the balance test documented under warnings |
| 2026-11-01 | `peer_identification` / `fit_peer_effects` — the network identification condition and the peer-effects 2SLS | two gates, and the first is not a Monte Carlo because the quantity is not random. `peer_identification` returns a verdict, so the gate is exact agreement with the verdicts Bramoullé, Djebbari and Fortin state in words, on the structures they name: equal-sized groups, self-included group means, two group sizes under fixed effects, three sizes, stars, complete bipartite graphs, lines and cycles — all 40 agree, as integer ranks rather than tolerances. The tolerance those ranks turn on is itself measured: worst spurious singular value across degenerate structures 1.97e-8, smallest genuine one across identified structures 4.0e-3, a separation of 2.0e5, and the default sits between them. The estimator's interval is the second gate: coverage of a nominal 95% interval, 2,000 replications, `n = 300` — 0.950–0.970 with a robust covariance across densities and error structures, against 0.921–0.942 for a homoskedastic one under strong heteroskedasticity. The Sargan test is a test statistic and is well sized throughout: 0.094–0.104 at 10%, 0.044–0.049 at 5%, 0.009–0.013 at 1%, across four densities. Five published identities are pinned at machine precision: `G² = G` under Manski's convention to 2.8e-17, the Moffitt/Lee within-block expansion exactly, their page-47 `G³` expansion for two groups to 2.8e-17, `G³ = G` on complete bipartite graphs exactly, and their Appendix B collinearity equivalence to 8.9e-16 | Verified correct; the density degradation of the estimator, and the fact that it does not show up in the rejection rate, documented under a warning |
| 2026-11-04 | `pesaran_cd` / `alpha_exponent` / `dependence_verdict` — the CD test for weak cross-sectional dependence, the exponent of cross-sectional dependence, and the verdict that reads them together | three statistics, so three gates, and the strongest one is a replication. **Size under independence** at nominal 10/5/1%, across six `(N, T)` combinations including `N ≫ T`: 0.091–0.102, 0.043–0.053, 0.007–0.014. **Pesaran's own headline result reproduced**, `N = 100`, `T = 50`, 3,000 reps, 5% level: rejection 0.048 / 0.054 / 0.055 at true `α = 0.00 / 0.10 / 0.25`, then 0.144 / 0.549 / 1.000 at `α = 0.40 / 0.50 / 0.75` — correctly sized across `[0, ¼]` and over-rejecting above, with the transition at precisely the `α = 1/4` boundary his eq. (12) states. His stated motivation reproduced too, on *estimated* residuals at `T = 10`: `CD` size 0.045 / 0.048 / 0.040 / 0.064 / 0.048 against `CD_lm`'s 0.068 / 0.248 / 0.794 / 1.000 / 1.000 for `N = 10 / 25 / 50 / 100 / 200`. `α̂` is not a test statistic, so its gate is bias and RMSE over Bailey, Kapetanios & Pesaran's own admissible range, plus **refusal where the parameter does not exist**: an independent panel gives `α̂ = 0.3684` and throws, and the cancelling design is refused 300/300 against 0/300 for the homogeneous one. Scale invariance is exact at 0.0e0 under standardisation and 0.394 without it. The CD and LM statistics are recomputed from explicit double loops over eqs (2), (5), (9), (10) and agree to 3.6e-15 or better | Verified correct; two failure configurations measured and documented rather than left to be discovered — see below |
| 2026-11-04 | `lr_spatial_interaction` — the likelihood-ratio test of a SpVAR against the nested panel VAR, i.e. whether there is any spatial interaction to model | one statistic, one gate, run on data generated with **no spatial interaction at all**. Size at nominal 10/5/1%, 1,500 replications: 0.1100 / 0.0527 / 0.0113 at `N = 12, T = 100`, 0.1147 / 0.0513 / 0.0080 at `N = 12, T = 400`, and 0.1193 / 0.0620 / 0.0100 at `N = 24, T = 200`. The Monte Carlo standard error is 0.008 at the 10% level, so the test is mildly liberal there — one to two standard errors — and close to nominal at 5% and 1%. Power against the spatial DGP is 1.000 at the 5% level already at `T = 50`. The restriction itself is pinned rather than trusted: `fit_spvar(…; s = 0)` reproduces the independently written `fit_nested_panel_var` to 0.000e+00 in both the log-likelihood and every coefficient | Verified correct; the 10% liberality documented, and the 150-replication run that said the opposite recorded below |

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

### `bayes_spec_probs` / `sdem_roadmap` — five checks, in testset 30

A posterior model probability is not a test statistic, so the rule was read
across to the error rate again, as it was for `select_knn_w`. The rates in
the table above are the gate; the four checks under them are what makes the
quantity being rated the right one.

1. **The conditional marginal against 2-D quadrature.** With `β` and `σ²`
   integrated analytically the marginal at a fixed `ρ` is

   ```
   p(y|ρ) = |A| · π^(-(n-k)/2) · |Z'Z|^(-1/2) · Γ((n-k)/2) · S(ρ)^(-(n-k)/2)
   ```

   which was checked against numerical integration of the full
   `(β, σ²)` integrand at `n = 6`, `k = 1`: agreement to **5.3e-15**.

   That check failed at 7.08e-5 first, and the failure is the useful part.
   The discrepancy did not shrink as the grid was refined from 501 to 4001
   nodes per axis — it converged to 7.081e-5 and stopped — which rules out
   quadrature error and points at the domain. The marginal in `β` is a
   `t` with `n - k = 5` degrees of freedom, and the brute-force grid ran
   `±12` scale units, which leaves about 7e-5 of that distribution's mass
   outside. Re-running with `β = β̂ + s·tan(u)` over `u ∈ (-π/2, π/2)`, which
   covers the whole line, gives the 5.3e-15 above and is stable under
   refinement. A heavy-tailed reference integrand needs a substitution, not
   a wider box: `±12` sigma is ample for a Gaussian and not remotely enough
   for a `t₅`.

2. **An exact closed form, committed.** `30a` pins the same formula with the
   gamma term written as a factorial: `N = 25` with `k = 3` makes it
   `Γ(11) = 10!`, and with the Durbin block `k = 5` makes it `Γ(10) = 9!`.
   `|Z'Z|` is taken by `det` rather than the implementation's QR. Both agree
   to 1e-10. This is the check that would catch the normalising constants
   being dropped as "constants" — they are constant in `ρ` and not in `k`,
   so dropping them leaves a quantity that ranks two specifications of
   equal width correctly and mis-ranks every comparison that crosses a `k`.

3. **The error family against an independent integral.** `30b` rewrites the
   `:sem` marginal from scratch in the test file — dense `logabsdet` for the
   Jacobian, `det` of the filtered cross-product, the trapezoid written out
   longhand — and gets the same number to 1e-9. The target is the `|Z'Z|`
   term: for an error specification it is evaluated on `(I - λW)[y Z]` and
   therefore moves with `λ`, while for a lag specification it is genuinely
   constant. Dropping it is invisible in any lag-only comparison and biases
   every global-against-local comparison toward the error specifications.

4. **The flat specifications as limits.** `:ols` and `:slx` are the `λ = 0`
   members of the families they belong to, and `30c` checks all four pairs
   reproduce the closed form to 1e-8 when integrated over `(-1e-8, 1e-8)`.
   This pins the prior's normalisation against the analytic branch: a prior
   that did not integrate to one over `rho_bounds` would move one and not
   the other.

5. **What the quadrature actually does.** The trapezoid rule is
   geometrically accurate on an integrand whose Euler–Maclaurin boundary
   terms vanish, and that is the common case here: at `N = 256` with a
   correctly specified lag model the integrand is 39 and 66 logs below its
   peak at `ρ = ∓0.995`, and 100 nodes already agrees with 800 to better
   than 1e-9. It is not the universal case, and the exception is worth
   knowing: an **error** specification fitted to lag data wants its
   parameter at the edge of the stationary interval, so the integrand is
   only 8 logs down at `λ = 0.995`, the boundary term survives, and the rate
   falls back to `O(h²)` — successive-error ratios of 3.08 and 3.32 against
   the geometric case's 33.5. The same happens to every specification at
   `N = 25`, where the posterior is wide enough to be truncated by the
   prior's support. Both are harmless at the default 200 nodes, where the
   absolute error stays under 1e-4 against gaps between specifications of
   order 1, and `30d` pins the rate in both regimes rather than asserting
   machine precision across the board.

**Selection rates.** 200 draws from each of the six specifications on a rook
lattice, `ρ = λ = 0.5`, `θ = (0.7, 0)`, `σ = 0.5`, all six compared on the
generating `W` under the default uniform prior. Share of draws whose highest
posterior probability lands on the generating specification:

| `N` | `:ols` | `:slx` | `:sar` | `:sdm` | `:sem` | `:sdem` | reach right |
|---|---|---|---|---|---|---|---|
| 100 | 93.0% | 90.0% | 81.5% | 95.5% | 88.5% | 88.5% | 90.5–100% |
| 256 | 92.5% | 96.0% | 94.0% | 100% | 96.5% | 96.0% | 96.0–100% |
| 400 | 94.5% | 94.0% | 97.5% | 100% | 96.5% | 98.5% | 98.5–100% |

Chance is 16.7%; mean posterior probability on the generating specification
runs 0.589–0.916 at `N = 100` and 0.711–0.999 at `N = 256`. The last column
is the share getting the *reach* right — global mass exceeding local mass
when it should and the reverse otherwise — and it is higher than the six-way
rate at every `N`, as a coarser question should be. Every confusion in the
table is between specifications carrying the same Durbin block, including
the crossings: `:sar` goes to `:sdem` in 10.0% of draws at `N = 100` and
`:sdem` to `:sdm` in 9.0%, that pair being identical in design and differing
only in where the dependence sits. At `N = 400` the worst crossing is 1.5%.

**One methodological note, because it cost a test rewrite.** `30f` asserts
the winner on a fixed sample, and the first version of it built the
disturbances as `0.4·sin(5i) + 0.3·cos(11i)` — a deterministic stand-in for
a draw, chosen because `randn(::MersenneTwister)` is not reproducible across
Julia versions. On a lattice that is not noise. Consecutive indices are
neighbours, so a smooth function of the index is strongly spatially
autocorrelated, and the SLX sample built that way was not the
independent-error case it was written to be: the comparison picked `:sdm`
over `:slx`, correctly, because the errors really were spatially dependent.
The same mistake in the covariates made `W*X` nearly collinear with `X` and
turned an ordinary Durbin design into a hard one, which is what drove
`P(:sdm)` down to 0.496 on a sample where the Monte Carlo average is 0.999.
Both are now drawn from a small LCG with Box–Muller written into the test
file: deterministic, identical on every version, and independent of the
lattice. The general form of the trap is that on spatial data a
"deterministic surrogate for randomness" has to be checked for spatial
structure before it can stand in for an independent draw.

### `stpca` — five checks, in testset 31

Spatio-temporal principal components are a decomposition rather than a test,
so the rule is read across to the error rate once more, as for
`select_knn_w` and `bayes_spec_probs`. The rates in the table above are the
gate. What makes them a gate on the right quantity is that the eigenvalue
being ranked is checkable in closed form.

1. **Every eigenvalue is its own objective.** The method maximises the
   product of a functional principal component's variance and the spatial
   autocorrelation of the scores it produces, and the algebra collapses that
   product into a quadratic form, so for each component

   ```
   lambda_k = Var(U_k) x A(A gamma_k)
   ```

   exactly, not approximately. The committed test recomputes **both**
   factors from other modules — `SpatialDiagnostics.geary_c` and
   `SpatialFiltering.moran_coefficient`, neither of which `stpca` calls —
   and compares, for every one of the 15 components under both measures: 60
   checks, agreeing to 5.0e-14. This is the strongest form the closing
   standard can take, because the quantity being ranked has an independent
   definition and the test uses it. It fails if the scaling of `C` or `E` is
   wrong, and it fails if the coefficient matrix is not centred.

2. **The basis.** The construction rests on the Fourier basis being
   orthonormal with respect to the integral over the expansion interval —
   that is what makes the functional inner product equal the dot product of
   coefficient vectors, and so what makes `gamma'S gamma` the variance of
   the functional principal component. Checked by numerical integration on
   100,001 points: `max |G - I| = 5.6e-16`. Least squares in that basis
   recovers the coefficients of a function already in its span to 4.4e-16.

3. **The contiguity matrix.** Turning Geary's ratio into an eigenproblem
   needs `y'By = sum_ij w_ij (y_i - y_j)^2`, which is checked against the
   raw double sum rather than assumed, and the ratio it builds is checked
   against the package's own `geary_c`. Both exact to rounding.

4. **Scale and shift.** Variance scales with `c^2` and both ratios are
   scale-free, so multiplying the data by 3 multiplies the whole spectrum by
   9 and leaves the directions fixed — asserted on the eigenvalues and on
   `|V3' V1| = I`. Adding a constant to every unit's series changes nothing,
   because it moves only the coefficient of the constant basis function and
   the coefficients are centred.

5. **Sign structure.** Under `:geary` the objective is a variance times a
   sum of squares, so `E` is non-negative definite and the negative group is
   empty — a property of the construction, asserted as one. Under `:moran`
   both signs appear on data carrying both structures, and the sign of each
   eigenvalue matches the sign of its component's Moran coefficient.

**Recovery rates.** 200 draws per cell, `p = 3` features on 17 annual
moments, five Fourier functions each. A known field is planted in feature 1
with a sinusoidal temporal profile: either the leading eigenvector of the
rook `W` (smooth, `I = +0.92` at `N = 36`) or its trailing eigenvector
(high-contrast, `I = -1.00`). The entry is the share of draws in which the
component most correlated with the planted field was the first or last of
the spectrum.

| Planted field | `N` | `:moran` first | `:moran` last | `:geary` first | `:geary` last |
|---|---|---|---|---|---|
| smooth | 36 | 86.5% | 0.0% | 0.0% | 7.0% |
| smooth | 64 | 73.0% | 0.0% | 0.0% | 14.0% |
| high-contrast | 36 | 0.0% | 86.0% | 33.5% | 0.0% |
| high-contrast | 64 | 0.0% | 71.5% | 7.0% | 0.0% |

The `:moran` split is sharp and never inverted: 0 of 800 draws put a field
at the wrong end. The `:geary` ordering is the finding that changed the
documentation. It demotes smooth structure reliably — a smooth field was
never the leading `:geary` component in 400 draws — but it does **not**
simply promote the roughest, which led in 7–34% of draws. The objective is a
product, so a component with moderate `C` and large variance outranks one
with high `C` and little. An earlier draft of the `stpca` docstring claimed
the two orderings were "close to reversals of each other"; the measurement
refuted the second half of that and the docstring now says what the numbers
support.

A second number worth recording, because it bounds what the method claims:
the median correlation between the planted field and its best-matching
component was 0.56 at `N = 36` and 0.43 at `N = 64`. The field is spread
across several components rather than captured whole. The component
*ranking* is the reliable output; the loading is not a recovery of the
field, and a first draft of the committed test asserted `cor > 0.9` on one
sample before the Monte Carlo showed what the distribution actually is.

**On sources.** `measure = :moran` is equation (21) of Krzyśko, Nijkamp,
Ratajczak, Wołyński & Wenerska (2024), *Spatial Economic Analysis* 19(1),
8–29, with the symmetrisation of its equation (23); `measure = :geary` is
equation (4) of the sibling paper, Krzyśko, Nijkamp, Ratajczak, Wołyński,
Wojtyła & Wenerska (2023), *Computers, Environment and Urban Systems* 103,
101980. Both were read in full before the implementation was finalised. The
Geary paper arrived first and the Moran paper second, and the second
confirmed the branch that had been written from a secondary description —
`C = (sum w_ij)^-1 A'WA` — unchanged. The permutation test of the Moran
paper's section 5, which tests the significance of the positive and negative
groups jointly, is **not** implemented: it is a test statistic, so it falls
under the standing rule and needs its own null distribution before it can be
believed.


### `fit_spatial_did` and `vcov_cluster` — in testset 34

Phase 7A's first module, `src/SpatialCausal.jl`. Two things in it fall under
the standing rule — a covariance estimator and an estimator whose
staggered-design behaviour had to be characterised rather than assumed — and
both came out differently from the plan.

**The exact part is exact.** The two-period design first-differences to an
ordinary least squares problem, so the gate is algebra and not a tolerance:
generated noiselessly, `att_direct` returns `τ` to 5.3e-15, the two
spillovers to 1.8e-15 and 7.6e-17, and `att_total_treated == att_direct +
att_spill_treated` holds **exactly**, both being the same linear form in
`θ̂`. The staggered path under homogeneous effects is the same story at
4.7e-15. There was no reason to accept less: every reported quantity is
`a'θ̂` for a fixed `a`, so its variance is `a'Σa` and needs no delta method.

**The covariance default was reversed by measurement.** The plan chose
Conley on the reasoning that treatment in a spillover design is spatially
correlated by construction, so a spatial HAC is the matching instrument.
Coverage of a nominal 95% interval on `att_direct`, 2,000 replications,
`N = 144`, errors generated as `(I − 0.6W)^{-1}ε`:

| Estimator | Coverage | mean SE / MC sd |
|---|---|---|
| cluster on unit | 95.0% | 1.025 |
| OLS | 96.5% | 1.068 |
| Conley, cutoff 1.5 | 94.5% | 1.005 |
| Conley, cutoff 2.5 | 92.5% | 0.948 |
| Conley, cutoff 4.0 | 90.5% | 0.893 |
| Conley, cutoff 6.0 | 85.3% | 0.785 |

Checked across nine configurations — treated block, scattered, half-grid and
alternating stripes, at `ρ = 0`, `0.6`, `0.9`, at `N = 144` and `400` — the
ordering never changed: clustering stayed in `93.0–97.0%`, and Conley fell
monotonically with the cutoff to `82%` at the widest.

Two measurements settle *why*, and neither was guessed.

1. **Conley degrades identically with no spatial correlation present**
   (`93.0% → 83.9%` across the same cutoffs at `ρ = 0`). So the decay is the
   finite-sample cost of estimating more pair covariances, not the
   correction working. `psd_fix = false` reproduces every figure to three
   decimals, so it is not the PSD projection either — the meat is already
   positive semi-definite here.
2. **The exposure column absorbs the dependence before it reaches the
   standard error.** Including `WD` identifies `τ` off the contrast between
   a unit and its neighbours, and that contrast differences out the error
   component they share — which is exactly the spatially correlated part.
   `cor(D, WD)` runs 0.93–0.99 in these designs, and OLS's `SE/MC sd` barely
   moves between `ρ = 0` and `ρ = 0.9` (1.010 to 1.207). In the one design
   where `cor(D, WD) = 0` exactly, alternating stripes, every estimator
   over-covers.

The default is therefore `:cluster`, and the docstring carries the table
rather than the reasoning that lost. A spatial HAC is still right for a
spatially correlated regressor and `:conley` remains available, with the
instruction to report more than one cutoff and prefer the narrow end.

**What clustering is actually doing, stated because it is a trap.** In the
first-differenced two-period design there is one observation per unit, so
clustering on unit is a singleton-cluster design and reduces to the
heteroskedasticity-robust (HC1) form. Testset 34f asserts that coincidence
directly against the HC1 formula rather than leaving the docstring's claim
unchecked. It performs well for that reason, not because it is absorbing
serial or spatial correlation, neither of which survives the differencing.

**Staggered adoption: characterised, and worse than the textbook warning.**
Two-way fixed effects with heterogeneous effects is known not to recover a
weighted average effect (Goodman-Bacon). The usual statement is that the
estimate can leave the convex hull of the cohort effects. On these designs
**it never did** — the hull spans `[1.0, 8.0]` in the dynamic cases, which
makes that a weak guarantee rather than a reassuring one. The bias is the
thing to report, and with no noise every gap below is bias:

| Cohort layout | Effect profile | True ATT | `att_direct` | Error |
|---|---|---|---|---|
| contiguous bands | constant per cohort | 1.667 | 2.781 | +67% |
| contiguous bands | growing with exposure | 4.167 | **1.805** | −57% |
| interleaved | constant per cohort | 1.662 | 2.000 | +20% |
| interleaved | growing with exposure | 4.155 | 3.440 | −17% |

The sign is not predictable, which is the part a user most needs to know.
And the spatial dimension makes it worse rather than better, specifically
for this estimator: when adoption cohorts are spatially contiguous — which
is how rollouts actually happen — cohort identity is nearly collinear with
the exposure column, so `WD` absorbs cohort heterogeneity and `att_direct`
moves to compensate. `1.805` against a true `4.167` in that cell, where the
same profile with interleaved cohorts gives `3.440` and dropping the
exposure column entirely gives `3.600`. All eight figures are pinned in
testset 34e, so the docstring's warning cannot drift away from the code.

**`vcov_cluster` against its formula.** A covariance is a second-order
quantity, which is where audit findings 1–7 lived, so it is checked
against the sandwich written out independently in the test rather than only
through the estimator that calls it: agreement to 1e-12, the
`G/(G−1)·(n−1)/(n−K)` correction isolated by comparing with
`dof_correction = false`, invariance to row permutation to 1e-10,
non-integer group labels, and the singleton-cluster reduction to HC1. One
group and a saturated design both divide by zero in the correction and are
rejected by name instead of returning `Inf`.

**Identification is checked, not assumed.** `δ` is identified only by
controls that differ in exposure. `check_interference` reports the count of
unexposed controls, `var(E | D = 0)`, `cor(D, E)` and the two exposure
means, and it needs no outcome, so it can be run before fitting. Where the
design is outright rank deficient — a checkerboard treatment on a rook
lattice makes exposure exactly 1 for every control and 0 for every treated
unit — the existing `check_design_rank` refuses it by column name. That
reuse is the whole guard; nothing new was written for it.

**Scope, and what is deliberately absent.** Item #15 lists four
deliverables. Two are delivered. Spatial synthetic control is a constrained
optimisation rather than a regression, and the shift-share exogeneity tests
are test statistics that each owe a null-distribution Monte Carlo under this
very rule. Bundling three estimator families into one commit is the shape
that produced findings 13–16, so they are a separate pass.

Suite 2,233 on Julia 1.13.2 with Makie and Turing installed and 2,194 on
1.10.12 with both absent, zero failures either way. 910 lines in
`src/SpatialCausal.jl` and 120 in `SpatialHAC.jl`.

**A bookkeeping correction found on the way, worth recording because it
briefly looked like a regression.** Testset 34 contributes **162
assertions on both versions** — it is version-neutral — yet the two suite
totals first came back 39 apart against a documented gap of 6. Running the
suite verbose on each version accounted for all of it: testset 11b is 12
against 6, which is the genuine version-dependence and unchanged, and the
remaining 33 is Makie and Turing being installed in one environment and
not the other — testsets 25 and 26 contributing 11 and 24 assertions
against one broken marker each. Nothing had regressed; the earlier note
said the counts were taken with the extensions absent but not that the
extension difference is five times the cross-version one, so a figure
quoted from the wrong environment was unreadable. `ROADMAP.md` now names
the environment beside each number.

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


## Post-audit: what the benchmark suite found

These came from the benchmark suite built for roadmap item #10 on
2026-10-30, and none of them could have come from the test suite, which
had asserted the right answer about every one of these functions for its
whole life. Some are defects in asymptotic cost, some are optimisations
that move reported numbers in the last few bits -- which is why they are
written down here rather than left in a commit message -- and some are
corrections to earlier entries in this very section, because the first
pass of measurement got three of its five findings wrong.

### 29. `row_standardize!` was O(N · nnz) — FIXED

For each of the `N` rows in turn it scanned **every nonzero in the whole
matrix**, twice — once to accumulate that row's sum and once to divide by
it — testing `rows[idx] == i` and skipping the rest. The work is therefore
`2 · N · nnz` where `2 · nnz` suffices, and the penalty grows with `N`:

| `N` | `nnz` | before | after | |
|---|---|---|---|---|
| 900 | 3,480 | 4.02 ms | 0.0094 ms | 428× |
| 3,600 | 14,160 | 62.3 ms | 0.055 ms | 1,124× |
| 8,100 | 32,040 | 310.5 ms | 0.185 ms | 1,677× |
| 14,400 | 57,120 | — | 0.333 ms | |

Accumulating the row sums into a vector in one pass and dividing in a second
is the whole fix. Both versions visit a given row's entries in increasing
column order, so the summation order is unchanged and the output is
**bit-identical** — verified on six matrices including a rook lattice, a
queen lattice, knn and distance weights, a matrix with an empty row, and one
with negative entries, comparing `nonzeros` and `rowvals` for exact
equality, and confirmed again by the full suite returning the same 1,566
passes with every pinned value intact.

**Why no test caught it.** Every assertion about `row_standardize` was about
its *output* — rows sum to one, empty rows stay zero, `nnz` is preserved —
and all of them passed before and after. Asymptotic cost is not a property
of the output, so no output assertion can see it. Testset 32 now carries a
wall-clock bound at `N = 8100`: 50 ms, which is 260× above the linear
version's cost and 6× below the quadratic one's. That margin is what makes a
timing assertion safe on a shared runner — a tighter bound would be
measuring the runner rather than the code.

**The general lesson, which is the reason this is in the ledger.** The audit
closed on *values*: every finding ended in a test pinning a number against a
closed form or an independent implementation. Complexity is invisible to
that standard, and `row_standardize` sits under almost every example,
benchmark and test in the package, so the cost was being paid everywhere and
measured nowhere. A benchmark suite is not only for tuning; for this class
of defect it is the only instrument that detects it at all.

### 30. `precompute_neumann_traces` strode the wrong way — OPTIMISED

Not a defect: the answers were right. It is in the ledger because the fix
changes reported numbers in the last few bits, and the ledger is where that
gets stated rather than discovered.

The function estimates `tr(Wᵐ)/N` by carrying a block of `n_probes`
Rademacher vectors through repeated multiplication by `W`. The block was
`N x p`, so the inner loop of the sparse product stepped by `N` between
one probe and the next, and the trace estimate was then read off in a
second pass that re-read the whole block. Three changes:

- the block is `p x N`, so the `p` probes of one node are adjacent and the
  inner loop vectorises;
- the product gathers from `sparse(transpose(W))` — column `i` holds row
  `i` of `W` — instead of scattering from `W` in CSC form, which both
  removes the `copyto!` of the whole block per power and makes the loop
  safe to thread, since each output row is written by one thread and read
  by none;
- the inner product that yields the estimate is taken while the row is
  still in cache.

| | before | after | |
|---|---|---|---|
| `N = 900`, 250 probes | 73.2 ms | 6.0 ms | 12.2× |
| `N = 3600` | 248 ms | 41.7 ms | 5.9× |
| `N = 8100` | 684 ms | 204 ms | 3.4× |
| `fit_spatial` SAR, `N = 8100` | 806 ms | 444 ms | 1.8× |
| `fit_spatial` SEM, `N = 3600` | 329 ms | 121 ms | 2.7× |

**What changed in the output, stated precisely.** The probe set is
identical: the draw is still made into an `N x p` matrix and then
transposed, because `rand!` fills column-major and drawing straight into
`p x N` would silently hand a caller who passed a seed a different probe
set. Within a row the contributions are still accumulated in increasing
column order, as `mul!` on a CSC matrix does. What differs is the outer
sum, which now runs over nodes rather than probes, so the trace estimates
move by up to **7e-14** relative. The Hutchinson standard error at 250
probes is of order `1/sqrt(250)`, or 6%. The full suite passes with every
pinned value intact on both Julia 1.13.2 and 1.10.12.

**Determinism does not depend on the thread count.** Each thread writes
only `rowdot[i]` for the rows it owns, and the reduction over `rowdot` is
a serial pass in index order afterwards. Verified by hashing the 61
returned doubles at 1, 2, 7 and 16 threads: one distinct value. A
`Threads.@atomic` accumulator or a `@threads` reduction would have been
simpler and would have made the result vary with the machine, which for a
function feeding `spatial_effects` is the same class of surprise as an
unseeded probe.

**The threading is not where the speedup came from**, and saying so
matters because the roadmap had listed threading as a deliverable. At
`-t 1` the same rewrite is 2.4–5.7×; sixteen threads add 4% over one at
`N = 14400`. Once the probe block exceeds cache the loop is bound by
memory bandwidth, not arithmetic, so more cores have nothing to do. The
threading is kept because it is free at small `N` and provably
deterministic, not because it carried the result.

**What it did not fix.** `N = 900` is unchanged, because at or below
`max_exact_n = 1000` the traces come from a dense eigendecomposition
instead. The gap between the branches at their own boundary is now 51×
rather than 4.6×. Narrowing it means trading exactness for probes, which
changes what `spatial_effects` reports for most applied `N`, so it is
recorded in `ROADMAP.md` as a decision rather than taken here. The
symmetrisation that would make the exact branch fast and keep it exact is
not available: it needs the unstandardised adjacency and its degrees, and
`spatial_trace_powers` receives only the standardised `W`.

### 31. Four more from the suite, two of them corrections — FIXED

Taken together because they share a lesson: three of the five findings the
first benchmark pass recorded were stated wrongly, and the benchmark suite
is also what caught that.

**`fit_gwr` threaded, 4.0–6.3×.** `N` local regressions, each writing only
its own row of `coef_local`, `se_local`, `local_R2`, `residuals`, `fitted`
and `hat_diag`; `tr(S)` and `RSS` are serial passes afterwards. Unlike the
probe loop of finding 30 this one is compute-bound — a QR per observation
— so it scales with cores. Bit-identical at 1, 4 and 16 threads, including
the auto-bandwidth path, which calls the threaded fit repeatedly inside
the AICc search.

Threading alone gave only 3.1× on sixteen cores, which for an
embarrassingly parallel loop was itself the finding. Two allocation faults
accounted for the rest: `_get_weights` computed every candidate distance
as `sqrt(sum((coords[i, :] .- coords[j, :]) .^ 2))`, three vectors per
neighbour and `O(N · bw)` allocations per fit; and the local-R² loop
rebuilt two loop-invariant length-`N` vectors on each of its `N`
iterations. Allocation fell from 224 MiB to 115 MiB per fit and scaling
rose to 6.3×. It is still short of sixteen, and the reason is recorded in
`benchmark/README.md` rather than fixed: the per-call `zeros(T, N)` is an
`O(N²)` allocation, and the local design is a dense `N × K` where only
`bw` rows are nonzero. A compact local design would reorder the QR's rows
and so would not be a free change.

**`log_marginal_likelihood`, 1.4–1.5×, bit-identical.** The 200 exact
sparse LUs are 97–99% of the call. The first pass called this "the same
`max_exact_n` mistake", which framed the remedy as giving up exactness;
that was wrong. The sparsity pattern of `I − ρW` does not depend on `ρ`,
so the symbolic analysis is done once and every node reuses it through
`lu!`, with values written into the existing pattern instead of a fresh
sparse matrix per node. All 200 log-determinants are bit-identical on
Julia 1.13.2 and 1.10.12, and every marginal likelihood and specification
probability is unchanged to the last digit across six specifications,
three sizes, the Chebyshev branch, and a non-default grid, prior and `ρ`
interval. Testset 32e also pins `_affine_pattern` on a `W` with an entry
equal to 1, where a pattern built from `I - W` would cancel structurally
and lose that position at every other `ρ`.

**Correction: `knn_weights` is not `O(N²)`.** The first pass recorded
"O(N²) — 820 ms at `N = 8100, k = 12`". It has used a KD-tree all along
and takes **26 ms**; multiplying `N` by 2.25 multiplies the time by
2.36–2.41, against 5.06 for quadratic. The 820 ms came from
`row_standardize!`, which `knn_weights` calls by default and which was the
quadratic of finding 29 — a measurement of another function's defect,
attributed to this one. **A profile of a composite is not a profile of its
parts**, and the first pass did not separate them.

Something real remained, worth 1.2–1.5×: the neighbour selection
re-derived per candidate how many non-self neighbours the list held, an
allocating `filter` plus a `count` over a fresh slice inside the inner
loop, to decide what a running counter decides in `O(1)`. Allocation fell
from 40.5 MiB to 13.2 MiB at `N = 8100, k = 12`, and the output is
identical across 144 configurations of size, `k`, `binary`,
`row_standardize`, `symmetrize`, dimension and coincident points — the
last of these being the case the old guard existed to handle, where a
point is absent from its own neighbour list.

**Correction: the Chebyshev "461 MiB" was allocation, not a ceiling.** The
first pass read BenchmarkTools' `alloc` column as a memory requirement. It
is cumulative allocation over the call — twelve intermediate matrices,
nearly all collectable. Peak fill, which is what a ceiling means, is
**18.9 MiB** for a rook lattice at `N = 8100, q = 12`. The companion
figure, 97.8% of `N²` nonzeros for knn `k = 24`, was correct but was
quoted at `N = 900`, where it amounts to 12 MiB, so it was not evidence of
a ceiling either.

The ceiling is nonetheless real, and arrives later: the fill per row grows
until it saturates at `N`, making peak memory `Θ(N²)` on a sufficiently
dense `W`. knn `k = 24` is already at 90 MiB by `N = 3600`. So the fix is
a ceiling rather than a replacement. `max_fill_bytes` defaults to 128 MiB;
when the recursion is about to cross it, the remaining orders finish one
column block at a time in `3 · N · b` doubles, seeded from the two sparse
matrices already computed, so nothing is discarded. A trace is a sum over
columns, which is what makes this possible without ever forming `T_k(W)`.

Replacing the default outright would have been a regression, and
measurement is the only reason that is known: on a rook lattice the
blocked path is 4× to 25× **slower**, because the sparse recursion touches
only the fill that exists while the blocked path pays `N · b` per level
whatever the density. It is faster exactly where it is needed — knn
`k = 24, N = 3600, q = 12` goes from 1118 ms to 576 ms. At the default
every case in the suite stays on the fast path and is bit-identical, order
6, which is what `fit_spatial` uses, included.

**`max_exact_n` left alone, deliberately.** The 51× at the boundary
(finding 30) is now stated in the `spatial_trace_powers` docstring,
together with what buying it back would cost: `spatial_effects` and its
standard errors would become seed-dependent for every `N ≤ 1000`. Callers
who want the speed pass `max_exact_n = 0`. Recording the price of a
default is not the same as changing it.

Suite 1,600 → 1,691 on Julia 1.13.2 and 1,594 → 1,685 on 1.10.12, every
pinned value intact.

### 32. GWR built a dense design to use a fraction of it — FIXED

Phase 6C, and the one optimisation carried forward when item #10 closed
rather than parked with the rest, because the measurements supported it.

Every local regression in `fit_gwr` built `Xw = Diagonal(sqrt.(w)) * X`, an
`N × K` matrix, and factorised it. But `w` is zero outside the kernel's
support, so only `b` of those rows carried anything, and `b` is set by the
bandwidth rather than by `N`. The cost was `Θ(N² K)` where `Θ(N · b · K)`
does, with a dense `zeros(T, N)` per call on top. The support ratio,
measured on a square grid:

| | `N = 400` | `N = 900` | `N = 1600` |
|---|---|---|---|
| `FixedBandwidth(4.0)` | b/N 0.103 | 0.049 | 0.028 |
| `FixedBandwidth(8.0)` | 0.340 | 0.172 | 0.103 |
| `AdaptiveBandwidth(20)` | 0.054 | 0.024 | 0.013 |
| `AdaptiveBandwidth(500)` | 1.000 | 0.558 | 0.315 |

Single-threaded, against the threaded code of finding 31, so this isolates
the compact design from the parallelism already in place:

| `N = 900` | b/N | before | after | | allocation |
|---|---|---|---|---|---|
| `FixedBandwidth(4.0)` | 0.05 | 47.5 ms | **7.3 ms** | 6.46× | 107 MiB → 4.4 MiB |
| `AdaptiveBandwidth(20)` | 0.024 | 45.6 ms | **8.2 ms** | 5.57× | 106 MiB → 3.4 MiB |
| `FixedBandwidth(8.0)` | 0.17 | 53.7 ms | **14.6 ms** | 3.69× | 115 MiB → 7.2 MiB |
| `AdaptiveBandwidth(500)` | 0.56 | 157 ms | 96.4 ms | 1.63× | 140 MiB → 14.4 MiB |
| MGWR | — | 18.7 s | 15.3 s | 1.22× | 21.6 GiB → 2.66 GiB |

The speedup tracks `1 / (b/N)` as it should, which is the evidence that the
change did what it was meant to and not something else. On sixteen threads
the same cases are 2.3–5.0× further on, and cumulatively — against the
unthreaded code this session started from — `FixedBandwidth(8.0)` at
`N = 900` went from **105.4 ms to 6.68 ms, 15.8×**.

**What it changes in the output, stated precisely.** Not bit-identical, and
that was known before a line was written. A Householder QR on a `b × K`
matrix is not the same arithmetic as on the same matrix padded with `N − b`
zero rows, so coefficients move. Measured against the dense form across
three kernels, four bandwidths and three sizes: **3e-14** worst relative
difference on `coef_local`, and `tr_S` agrees to **1.7e-16** — effectively
exact, because the rows outside the support contribute exactly zero to
`X'W_i X` and so to `R'R`.

Bit-identical where it could be: `w_self` is the kernel evaluated at
distance zero rather than looked up in the support, `_euclid` is unchanged,
and scattering a support back into a dense vector reproduces the old
`_get_weights` output exactly, which testset 33a asserts with `==`.

**The one discrete risk was measured, not assumed.**
`_select_bandwidth_aicc` compares AICc values with `<`, so a near-tie could
flip the selected bandwidth — a visible, integer change, not a last-bit
one. The margin to the runner-up is **0.04–0.07 AICc units**, around `1e11`
eps, which no roundoff can cross. Testset 33c rebuilds the AICc curve from
the dense arithmetic and asserts the argmin agrees exactly.

**Row order is a contract.** `inrange` returns indices in KD-tree order, so
the compact design's rows would otherwise be ordered by tree internals.
Passing `sortres = true` gives ascending index order, which is both
reproducible across versions of `NearestNeighbors` and the order the dense
path accumulated in — the reason the difference above is 3e-14 rather than
merely small.

**Allocation, and why `Project.toml` tightened.** The dense weight vector
was the largest allocation but not the only one: `inrange` and `knn`
allocate their results, and `knn` at `k = 500` costs 8 KiB per observation
to return one number. `inrange!` and `knn!` write into buffers the task
already owns. They are present in `NearestNeighbors` 0.4.29, and which
earlier `0.4.x` first offered them could not be established, so the compat
bound moved from `"0.4"` to `"0.4.29"`. That is a tightening in name only:
nothing had ever been run against `0.4.0`, so the old bound described a
range no test had visited.

**Threading reworked to suit it.** `Threads.@threads` cannot hold per-task
scratch, and indexing scratch by `Threads.threadid()` is unsound once tasks
migrate. `_tforeach_chunks` spawns one task per contiguous chunk so each
allocates its buffers once and reuses them. Determinism is unchanged and
verified the same way: `coef_local`, `se_local`, `hat_trace` and the
auto-selected bandwidth hash identically at 1, 4 and 16 threads.

**Two pieces of duplication removed on the way.** `_local_regression` was
defined and never called, and embodied the dense pattern being replaced.
The two MGWR `se_local` blocks were byte-identical, one on each side of the
`inference` branch, and both rebuilt `AdaptiveBandwidth(bandwidths[k])` on
each of their `N · K` iterations although it depends only on `k`; they are
now one function, `_mgwr_se_approx`. In `fit_gwr` the standard-error and
local-R² loops each computed the same support, so there were two KD-tree
queries per observation for one neighbourhood; they are now one pass.

**What it did not fix.** MGWR gains only 1.22×, and the reason is not the
local designs — it is `_select_bandwidth_aicc`, called per column per
back-fitting sweep and re-scanning about 26 bandwidths each time, many of
them wide enough that `b/N` is near 1. Allocation still fell 8.1×, from
21.6 GiB to 2.66 GiB. Making the scan cheaper means not re-scanning a
bandwidth that has stopped moving, which would change which fixed point the
back-fitting reaches, so it is a separate decision and not a tidy-up.

Suite 1,691 → 2,036 on Julia 1.13.2 and 1,685 → 2,030 on 1.10.12, every
pinned GWR value intact — which is the real evidence that 3e-14 is the true
magnitude and not an artefact of the cases chosen to measure it.

## Post-audit defects: identification guards

The audit closed on the numbers an estimator reports. These two are about
what it does when the design makes those numbers meaningless, which the
ledger above never put a row against. Both were found on 2026-10-30 while
writing a documentation example, not by the suite.

### 26. `fit_spatial` failed inside LAPACK on a rank-deficient design — FIXED

`X` goes in without an intercept, because every `build_design_matrix`
prepends one. Pass a constant column anyway and the augmented design has two
of them; the concentrated likelihood's first least squares then died with
`SingularException: matrix is singular; factorization failed. Zero pivot
found at index 2`, from a `\` three frames below the user's call. Nothing in
that message names the intercept, the column, or the design.

The same applies to a duplicated covariate, to any covariate that is a linear
combination of the others, and — for `SDM`, whose design is `[1 X WX]` — to a
`W` for which `W*X` lies in the span of `X`, which is exactly the
unidentified common-factor case.

Fixed by checking the numerical rank of the augmented design before anything
is solved, and throwing an `ArgumentError` that names the dependent columns
and states the intercept rule.

### 27. `fit_gravity` returned 1e12-scale coefficients for an unidentified pair — FIXED

Finding 24's fix added a per-column check: a regressor the fixed effects
absorb completely leaves a zero column after demeaning, and that now throws.
It cannot see a **combination**. With origin fixed effects and two columns
whose sum is a pure origin-level term, each column survives demeaning on its
own, the demeaned design is rank deficient, and the weighted least squares
returned

```
coef = [-7.803811995558151e12, -7.803811995557692e12]   pseudo_R2 = 0.0293
```

— two enormous values that cancel against each other, next to a pseudo-R²
that looks like an ordinary bad fit. This is the shape of the `-1.0e53`
log-distance coefficient recorded against the gravity module before finding
24; that particular case was cured by the FE-sweeping fix, the class was not.
A duplicated regressor took the other route and threw `SingularException`
from inside the solve.

Fixed by checking the numerical rank of the weighted, demeaned design once
the first IRLS iteration has built it, and throwing an `ArgumentError` naming
the dependent regressors. The negative binomial goes through the same core,
so it is covered by the same check.

### The shared gate

Both call `rank_deficient_columns` in `src/SpatialUtils.jl`: a
column-pivoted QR, with the columns past `tol * |R[1,1]|` reported as
dependent, `tol = sqrt(eps)`. Pivoting is what makes it a rank test —
finding 27a is the record of what happens when an unpivoted factorization is
asked to decide rank, and the ratio-to-leading-pivot form is the same
conditioning statement `_chol_or_nothing` makes about an information matrix.

Testset 29 pins all of it: the helper against matrices of known rank
(including a design correlated at 0.9999 and one scaled by `1e6`, neither of
which may be flagged), the three `fit_spatial` cases plus the `SDM`
augmented-design case, and for `fit_gravity` the duplicated regressor, the
absorbed combination, and a control that still fits. Every design in the
testset is deterministic — no RNG, so nothing depends on the `randn` stream.
Testset 29 adds 30 assertions, the same 30 on both ends of the support
range. Suite through `Pkg.test`, extensions absent: 1,235 → 1,265 on Julia
1.13 and 1,229 → 1,259 on Julia 1.10. The six-assertion difference between
the two runs is older than this change and sits elsewhere in the suite.

> **Corrected 2026-10-31.** That difference was not between the two
> *versions*. The 1.10 figures here were measured in a `src`-and-`test`-only
> scratch copy, which omits `notebooks/data/columbus.csv` and so skips the
> six assertions testset 11b gates on it. Per-testset counts are identical
> on 1.13.2 and 1.10.12. See the `fit_synthetic_control` entry below.

### 28. The same trap at every other entry point — FIXED

Findings 26 and 27 were two instances of one hole, so the rest were swept in
the same pass. Every entry point that takes an `X` was probed with a
duplicated column. Not one was guarded, and they failed in two ways:

| Failure | Entry points |
|---|---|
| `SingularException` from inside the linear algebra | `fit_slx`, `fit_sdem`, `fit_spatial_regime`, `fit_gwr`, `fit_mgwr`, `fit_spatial_probit`, `fit_spatial_panel`, `fit_spatial_error` (panel), `fit_dynamic_spatial_panel`, `fit_gs2sls_panel`, `fit_dynamic_gmm` |
| returned a number | `fit_spatial_error` (cross-section), `fit_sac`, `fit_bayesian_spatial_probit`, `fit_esf`, `fit_re_esf` |

The second row is the dangerous one. A minimum-norm least squares splits the
unidentified coefficient across the dependent columns, so `fit_spatial_error`
reported

```
coef = [1.5376, 0.8833, 0.8833, 0.7398]
se   = [NaN, NaN, NaN, NaN]
```

for a design whose single true coefficient was 1.77: two plausible halves,
and a standard error that had already gone to `NaN` without anything saying
so. `fit_re_esf` split the same coefficient as 0.0956 and 1.6982 — the split
is arbitrary, not even even.

All of them now call `check_design_rank`, which wraps
`rank_deficient_columns` with the message. Where the estimator transforms the
design before solving it, the **transformed** matrix is what gets checked,
because that is what carries the rank:

- the panel estimators check after `within_transform`, which catches a
  regressor with no within variation — time-invariant under `IndivFE`,
  unit-invariant under `TimeFE`. `fit_spatial_panel` returned a coefficient
  of 1.23 for such a column before.
- `fit_gs2sls_panel` checks the transformed covariates **and** the excluded
  instruments, before `H` is built: `inv(Symmetric(H'H))` is what actually
  failed, and it runs before the structural solve.
- `fit_dynamic_gmm` checks the stacked differenced regressors, since the
  moment matrix `S_ZX' W S_ZX` inherits their rank.
- `fit_spatial_ordered_probit` has no intercept of its own — the cutpoints
  carry the level — so a constant column in `X` is itself the deficient
  case, and the message says so.

`fit_gwr` and `fit_mgwr` check the global design only. A local fit can still
be rank deficient at a narrow bandwidth with the global design full rank;
that is a different problem and is not addressed here.

Testset 29d covers all of it: every entry point on a duplicated column, the
instrument case, the time-invariant-under-`IndivFE` case **and** its control
— the same design fits under `NoFE`, which is what shows the guard is reading
the transform rather than the column it was handed. Suite 1,265 → 1,286 on
Julia 1.13 and 1,259 → 1,280 on Julia 1.10, both through `Pkg.test`.

**No false positives.** The 1,264 assertions that predate this pass all still
hold: not one legitimate design anywhere in the suite trips a guard.

### `fit_synthetic_control` and `placebo_test` — in testset 35

Phase 7A's second causal module, appended to `src/SpatialCausal.jl`. Two
things fall under the standing rule — a permutation p-value, and an
estimator whose donor-pool behaviour had to be characterised rather than
assumed — and one of them came out against the plan.

**A citation was withdrawn before anything was built.** The roadmap cited
Cerqua et al. (2022) for this row, and already flagged it as unverified —
reached through a secondary summary rather than the source. It does not
exist as described: there is no Cerqua 2022 spatial synthetic control
paper (the nearest real work, Cerqua & Pellegrini 2017, is on a different
problem). The reference set is now Abadie, Diamond & Hainmueller (2010),
Abadie & L'Hour (2021) for the penalised estimator, Abadie (2021), and Di
Stefano & Mellace (2024) for donor-pool contamination — each checked
against the published record. This is the second roadmap citation to fail
verification on contact, and the standing instruction to check before use
earned its place.

**The weights are checked against the exact optimum, not a tolerance.**
The donor weights solve a convex quadratic program over the unit simplex.
For a small donor pool the exact optimum is computable independently by
enumerating all `2^J − 1` supports and solving the equality-constrained
least squares on each — written out in the test rather than borrowed from
`src/`, so the reference is genuinely independent. Over 500 random
problems with `J ≤ T0`, where the optimum is unique:

| Quantity | Worst case |
|---|---|
| objective excess over the exact optimum | 3.55e-15 |
| deviation in the weights themselves | 1.31e-14 |
| relative KKT gap | 9.68e-15 |
| KKT violation off the support | 0.00e+00 |

With `J > T0` — more donors than pre-periods, which is the realistic
case — the solution set is a face rather than a point, so only the
objective is well defined; it agrees to 1.78e-15 while the weights
legitimately differ. That non-uniqueness is not a defect to be papered
over, it is the argument for the penalty, and the docstring says so.

An exact convex combination is recovered to 1.58e-13 and the gap
identity `gap == y − Σ w y` holds at 0.00e+00, being a definition.
`λ → ∞` drove the support into the minimum-distance donor set in 200 of
200 problems. The simplex projection satisfies its own characterisation to
1.11e-16 and was never beaten by random search over the simplex.

**The solver's stopping rule had to change, and the plan was wrong about
the failure mode.** The plan specified a step-length tolerance and a throw
on non-convergence. Both were wrong. When the donor pool exceeds the
predictor set the iterate drifts along the flat directions of a
rank-deficient `G` long after the objective has settled, so a step-length
test never fires — the realistic default path ran to `maxiter` at a KKT
gap of 3.6e-06. Throwing there would have made the ordinary unpenalised
fit fail on a legitimately non-unique problem. The rule is now the KKT
gap, relative to the gradient scale, with O'Donoghue–Candès adaptive
restart; the same case now converges to 1e-14 in 29,610 iterations and 48
ms. The certificate is reported in the result (`kkt_gap`,
`kkt_violation`) rather than converted into an exception, because
unconverged weights are otherwise indistinguishable from converged ones.

A false start worth recording: the first restart implementation rejected
the candidate point instead of merely dropping the momentum, so once near
the optimum floating-point noise restarted it on every iteration and
accuracy fell from 2.5e-14 to 4.4e-08. Adaptive restart must always accept
the iterate; the threshold on the objective increase must be relative.

**Donor-pool contamination is an identity, not a tendency.** The plan
intended to show that screening the donor pool reduces bias on average.
The first attempt at that failed its own test — the all-donor fit was
*closer* to the truth (0.0139 against 0.0161 on a true effect of 3.0) —
and the reason was a defect in the simulation, not the estimator: iid
factor loadings make a treated unit's neighbours no more similar to it
than any other unit, so the fit has no reason to select them, and
contamination it never ingests cannot bias it. How much screening matters
depends entirely on how much weight the contaminated units would
otherwise carry, which depends on whether characteristics are spatially
smooth, as real ones are:

| Loadings | screened bias | all-donor bias | weight on contaminated |
|---|---|---|---|
| iid | −0.0022 | −0.0157 | 0.049 |
| smoothed ×1 | −0.0009 | −0.0355 | 0.117 |
| smoothed ×3 | −0.0014 | −0.0603 | 0.196 |
| smoothed ×6 | −0.0022 | −0.0606 | 0.194 |

200 replications per row, true effect 3.0, spillover 1.2, `N = 81`. The
screened fit is unbiased throughout; the all-donor bias tracks the weight
it puts on contaminated donors. Holding that weight fixed, the bias is
exactly linear in the spillover: 0.0, −0.0308, −0.0603, −0.1192 at
spillovers of 0, 0.6, 1.2 and 2.4.

That linearity is not an empirical regularity, which is why the test
asserts the identity instead. The spillover falls in the post-treatment
periods only, so the weights — fitted pre-period — are untouched by it:
adding a spillover changes them by **0.00e+00**. Hence

```
att(contaminated pool) = att(clean pool) − Σ_{j} w_j · E_j · spillover
```

holds exactly, and testset 35f pins it to 1e-12 at four spillover sizes
on a design where the trap is unavoidable — the treated unit's path is an
exact half-and-half combination of two of its own neighbours, which are
themselves exact copies of two far corners. The unscreened fit takes 0.5
from each neighbour, reaches a pre-period RMSPE of 2.45e-15, and returns
2.500000000000002 against a true 3.0.

**The part a user most needs: the contaminated fit looks better.** Its
pre-period fit is *superior* (2.45e-15 against the screened fit's
1.77e-03 in the pinned design; 0.045 against 0.097 in
`examples/24_synthetic_control.jl`, where it carries 90.1% of its weight
on treated territory and is wrong by −0.433 against the screened fit's
+0.005). The contamination is post-treatment, so it is invisible in the
pre-period by construction. A pre-period fit diagnostic cannot detect this
failure at all, and the docstring no longer implies that checking it is
enough.

**The placebo p-value's size, measured.** No treatment effect anywhere,
7×7 rook lattice, `T = 16` with `T0 = 12`, an interior treated unit so 4
neighbours are excluded and 44 clean donors remain. The permutation is
exact over 44 assignments, so the smallest attainable p-value is
`1/45 = 0.0222`. Errors `(I − ρW)^{-1}ε`, 1,000 replications per row:

| `ρ` | `p ≤ 0.022` | `p ≤ 0.05` | `p ≤ 0.10` | mean `p` |
|---|---|---|---|---|
| 0.0 | 0.023 | 0.046 | 0.095 | 0.503 |
| 0.6 | 0.017 | 0.045 | 0.094 | 0.492 |
| 0.9 | 0.041 | 0.068 | 0.125 | 0.438 |

**The plan expected this to fail and it largely does not.** Permutation
inference assumes exchangeable units and spatial correlation breaks that,
so over-rejection was the predicted outcome. At `ρ = 0` and `ρ = 0.6` the
test is correctly sized and the mean p-value is 0.50, which is what
exchangeability predicts. Only at `ρ = 0.9` does it over-reject, by about
a third of the nominal level (6.8% against 5%, 12.5% against 10%, each
roughly 2.6 standard errors above nominal at this replication count). The
mechanism is the statistic: the post/pre RMSPE *ratio* scales the gap by
the pre-period fit, so a unit with a larger error variance has a larger
numerator and denominator both, and spatial dependence largely divides
out. The docstring carries the table and the `ρ = 0.9` caveat.

The binding limitation is granularity rather than size, and it is
reported rather than left to be discovered: the smallest attainable
p-value is `1/(1+R)`, so one treated unit with 18 donors cannot return
below 0.053 and cannot reject at 5% whatever the effect.
`min_attainable_p` is a field.

**Spillovers on the excluded neighbours, and when not to believe them.**
Screening removes neighbours because the policy reached them, which makes
their gaps an estimand. In the pinned design two of the four neighbours
are exactly reproducible from the clean pool and return the spillover to
1e-10; the other two cannot be fitted at all, with a pre-period RMSPE
larger by more than ten orders of magnitude, and their gaps are not
spillovers. The unweighted mean over all four is consequently off: 0.545
against a true 0.500 on Julia 1.13.2, and 0.589 on 1.10.12 — the two
unfittable gaps follow the `randn` stream, which differs between minor
versions, while the two fittable ones return `0.500000000000` on both.
The test pins the exact pair and bounds the aggregate, rather than
pinning a figure that is an artefact of the version. `SpilloverGaps` therefore reports `rmspe_pre` per
unit and the documentation says to read it first; the function does not
silently drop units, since which ones are credible is the user's
judgement and not a threshold this package should pick.

**No standard errors, deliberately.** `SyntheticControlResult` defines
`coef`, `coefnames`, `nobs` and `residuals` but not `vcov` or
`stderror`, and the test asserts that calling them raises `MethodError`.
The weights are a constrained optimum rather than an estimator with a
sampling distribution of known form. Four of the first twelve audit
findings were a variance slot filled with a constant; declining to
furnish a variance at all is the same lesson applied earlier.

**`V = I` on standardised predictors, not a nested search.** Classic
synthetic control searches over a predictor-weighting matrix `V` in an
outer loop. That problem is documented as non-unique, so two equally good
`V` give different weights and different counterfactuals — which would
make the estimate irreproducible and the exactness gates above
unstateable. The simplification is recorded in the docstring as a
simplification, with the reason.

**Suite.** 2,430 assertions with Makie and Turing installed and 2,395
with both absent, zero failures either way, testset 35 contributing 197 —
the figures as measured on 2026-10-31, when this entry was written and the
suite had 51 testsets. They are left as measured; the current totals are in
`ROADMAP.md`. A by-product of checking them: running the suite verbose on
each version and diffing the per-testset counts shows **every one of the 51
testsets reporting an identical count on Julia 1.13.2 and 1.10.12**. The
six-assertion cross-version gap this repository had documented since the
audit does not exist. It was a **working-tree** gap: testset 11b gates six
assertions on `isfile(joinpath(@__DIR__, "..", "notebooks",
"data", "columbus.csv"))`, so it reports 12 in the repository and 6 in a
`src`-and-`test`-only copy. Dependency resolution was tested as a candidate
and refuted — copying the local `Manifest.toml` across does not reproduce
the 2,395. The CSV is tracked, so **CI has it and sees the in-repository
figure** with the extensions absent, never the copy's. The same `@__DIR__` dependence is why a verbose variant
must be run from inside `test/`: run from a scratchpad it skips the block
and undercounts by exactly six. `ROADMAP.md` records the correction.

**Scope.** At the time of this entry item #15 had three of four
deliverables. Spatial shift-share exogeneity diagnostics were the fourth,
and are delivered in the entry below, which closes the item.


### `fit_shift_share` and `balance_test` — in testset 36

Shift-share is the one design in this package where the dependence does not
travel through space. The regressor is built as `X_i = Σ_k w_ik g_k`, so two
regions with the same industry mix have mechanically correlated residuals
however far apart they are, and every spatial correction in this package is
answering the wrong question about it. That is the premise, and it is worth
stating because the natural instinct — reach for `vcov_cluster`, or for
Conley — is measured below and does not work.

**Every formula was read from the primary sources rather than recalled**,
with `pdftotext` on the published papers. This was not ceremony. The
shock-level projection in Adão, Kolesár & Morales's Remark 5 step 2 is the
vector of coefficients from regressing the *residualised* regressor on the
share matrix, and recall would have substituted the shock vector `g`. The
two differ whenever there are controls, which is always, since an intercept
is a control. Testset 36e pins the actual relationship.

#### Three published identities, pinned at machine precision

Each is an identity in its source, so a tolerance would be the wrong gate
and the agreement is checked at machine precision over randomised designs
with controls present.

| Identity | Source | Max relative error |
|---|---|---|
| Shift-share estimate = exposure-weighted shock-level estimate, direct | BHJ eq. 4 | 2.29e-13 |
| The same, IV design | BHJ eq. 4 | 2.26e-13 |
| Exposure-weighted aggregates of a residual have weighted mean zero | BHJ fn. 6 | 2.04e-16 |
| Rotemberg weights sum to one | GPSS Prop. 3 | 1.42e-14 |
| `Σ_k α̂_k β̂_k` equals the shift-share IV estimate | GPSS Prop. 3 | 2.64e-13 |
| AKM variance = cluster-robust variance under full specialisation | AKM §4.2 | 2.97e-15 |
| `X̊` solves the normal equations on the share matrix | AKM Remark 5 | 2.99e-15 |
| `X̊ = g − mean(X)/c` when shares sum to `c` | derived | 1.85e-15 |

The sixth is the strongest of them. Under full specialisation — every region
in exactly one industry — AKM note their variance becomes a cluster-robust
variance clustering on that industry. The check is against this package's
own `vcov_cluster` with the degrees-of-freedom correction off, so the
reference is independently written code rather than a rearrangement of the
same expression. Agreement at 3e-15 is what licenses the rest.

The eighth was derived in the course of diagnosing a failed gate, and is
the reason the `X̊` step is believed: with shares summing to `c`, the share
matrix satisfies `W·(1_K/c) = 1_N`, so the intercept's contribution to the
projection is exactly `mean(X)/c` in every coordinate.

#### The AKM standard error: coverage

The estimator is derived conditional on the shares and on the potential
outcomes, with the sampling variation in the shocks (their Remark 4), so
the study holds the shares and residuals fixed and resamples the shocks.
2,000 replications, `N = 300`, nominal 95%. "Sector frac" is the fraction
of residual variance carried by an industry-level component — the
dependence at issue. The cluster column clusters on each region's
largest-exposure industry.

| Effective shocks | Sector frac | AKM | HC1 | cluster | med se AKM / sd(β̂) | med se HC1 / sd(β̂) |
|---:|---:|---:|---:|---:|---:|---:|
| 99.9 | 0.00 | 0.930 | 0.950 | 0.947 | 0.952 | 0.999 |
| 99.9 | 0.40 | 0.938 | 0.902 | 0.894 | 0.984 | 0.858 |
| 99.9 | 0.82 | 0.936 | 0.732 | 0.720 | 0.983 | 0.558 |
| 99.9 | 1.00 | 0.937 | 0.664 | 0.658 | 0.979 | 0.505 |
| 199.9 | 0.81 | 0.940 | 0.768 | 0.766 | 0.995 | 0.612 |
| 50.0 | 0.82 | 0.919 | 0.632 | 0.619 | 0.930 | 0.456 |
| 20.0 | 0.77 | 0.864 | 0.426 | 0.404 | 0.837 | 0.294 |
| 37.9 | 0.90 | 0.827 | 0.289 | 0.241 | 0.770 | 0.211 |

Four findings, two of them not what the plan expected.

**The failure is real and larger than advertised.** Adão, Kolesár & Morales
report conventional standard errors rejecting a true null in up to 55% of
placebo samples. Here HC1 rejects in **71%** of samples on the concentrated
design, with the median HC1 standard error only 0.211 of the true sampling
standard deviation. This is not an adversarial construction; it is a design
with realistic industry-size dispersion and an industry-level disturbance.

**Clustering on the dominant industry is not a substitute.** This was worth
measuring because the cluster-robust machinery is already in the package
and is the obvious thing to try. It tracks HC1 almost exactly across every
row, and is *worse* than HC1 on the concentrated design (0.241 against
0.289). The reason is structural: the dependence runs through the whole
share vector, so assigning each region to one industry discards most of it.

**AKM works where its asymptotics hold, and degrades where they do not.**
With 100 or 200 effective shocks it holds 0.936–0.940 regardless of how
much of the residual is sectoral, and its median standard error is within
2% of the truth. By 20 effective shocks coverage is 0.864, and on the
concentrated design 0.827. Adão, Kolesár & Morales anticipate exactly this:
eq. 29 generalises a cluster-robust formula and inherits its downward bias
from estimation noise in the residuals, which is why they also propose a
null-imposed interval (their Remark 6). **That variant is not implemented**,
and the shortfall is documented in the docstring and warned about by
`exposure_concentration` rather than left for a user to discover.

**The effective-shock count is a useful summary and not a sufficient one.**
Coverage was *worse* at 37.9 effective shocks with concentrated exposure
(0.827) than at 50.0 with dispersed exposure (0.919). So the scalar does
not order designs by reliability on its own, and the audit records that
rather than implying the diagnostic is a decision rule.

A fifth, minor: with no sectoral component at all, AKM covers 0.930 against
HC1's 0.950. Robustness is not free, but 2 points is a small premium.

#### The balance test: size under the null

A test statistic, so the standing rule applies. The characteristic is
independent of the shocks but carries an industry component, so it is
structured rather than iid — the easy case would prove nothing. 2,000
replications, `N = 300`.

| Effective shocks | 10% | 5% | 1% | mean p | joint, 4 characteristics at 5% |
|---:|---:|---:|---:|---:|---:|
| 99.9 | 0.108 | 0.056 | 0.010 | 0.497 | 0.088 |
| 50.0 | 0.115 | 0.064 | 0.018 | 0.496 | 0.147 |
| 20.0 | 0.170 | 0.109 | 0.043 | 0.454 | 0.324 |
| 13.0 | 0.278 | 0.170 | 0.072 | 0.304 | — |
| 9.1 | 0.335 | 0.232 | 0.114 | 0.305 | — |

At 100 effective shocks the test is almost exactly sized and the mean
p-value is 0.497. It degrades monotonically with the effective count: by 20
it rejects twice as often as nominal, and at 9 it rejects 23% of the time
at a nominal 5%. The joint test degrades faster still, because it estimates
an `M × M` covariance across characteristics from as many observations as
there are shocks; with four characteristics and 20 shocks it rejects 32% of
the time.

The uncomfortable consequence is recorded rather than smoothed over: **in a
concentrated design a passed balance test is close to uninformative**, and a
failed one may be the test failing rather than the design. Both the
docstring and the `show` method carry the measured rates and point at
`exposure_concentration`.

This is the same story as the coverage table, which is the point worth
carrying away from this entry: in a shift-share design the effective number
of shocks governs every inference simultaneously, and it is the first thing
to compute, not a footnote.

#### What is deliberately not here

No null-imposed interval (AKM Remark 6, their AKM0), which is the
documented remedy for the few-shock case measured above. No panel variant
(their §5.2). No overidentification test across shocks. Each is additive
and none is implied by the row's scope; the gap that matters is AKM0, and
it is named in the docstring where the shortfall appears.

**Suite.** 3,051 assertions with the extensions installed and 3,016 with
both absent, zero failures either way, on both Julia 1.13.2 and 1.10.12.
Testset 36 contributes 621, and the step from the previous total is exactly
that, which is the arithmetic check described in `ROADMAP.md`. Both Monte
Carlo studies above run outside the suite; what is asserted inside it is
the loose bound that would catch a regression, so the suite does not pin a
simulated rate.

**Scope.** Item #15 is complete: exposure mapping, spatial DiD, spatial
synthetic control and shift-share diagnostics.


### `peer_identification` and `fit_peer_effects` — in testset 37

Phase 7A's last row, and the one where reading the primary source changed
what got built. The roadmap described the deliverable as "automated rank
validation of the `W²X`, `W³X` instrument set". Bramoullé, Djebbari and
Fortin's condition is not that. Theirs is a rank condition on the matrices
`I, G, G²` and does not involve `X` at all; their footnote 17 states the
check verbatim, as a rank-3 test on `[vec(I) vec(G) vec(G²)]`. The two
answer different questions — identification in population, instrument power
in this sample — and both are now reported, separately labelled. Conflating
them is how a degenerate network gets fitted.

**1. The verdicts, as integers.** The paper states its results about named
structures, so the gate is exact agreement rather than a tolerance, and the
algebra behind each is pinned too.

| Structure | Prop. 1 | Prop. 4 | Identity pinned |
|---|---|---|---|
| Equal-sized groups | ✗ rank 2 | ✗ | — |
| Groups, mean includes self | ✗ rank 2 | ✗ | `G² = G` to 2.8e-17, any sizes |
| Groups, mean excludes self, equal | ✗ rank 2 | ✗ | `G² = I/(s-1) + ((s-2)/(s-1))G` exactly |
| Two distinct group sizes | ✓ rank 3 | ✗ rank 3 | their p.47 `G³` expansion to 2.8e-17 |
| Three distinct group sizes | ✓ | ✓ rank 4 | — |
| Star / complete bipartite | ✓ | ✗ | `G³ = G` exactly |
| Line, cycle, random graph | ✓ | ✓ | diameter ≥ 3 (their Corollary 1) |

Forty verdicts, all agreeing. The second row is the one that matters in
practice: taking a classroom average *including* the student makes
`G² = G` identically, and no variation in group size rescues it.

**2. The tolerance was wrong, and measurement caught it.** The rank is
computed from the Gram matrix of the trace inner product rather than from
an explicit `n² × m` matrix — at `n = 3,000` the explicit form is 288 MB
and the Gram is `m × m`. The cost is that forming the Gram squares the
condition number, so a singular value that is truly zero comes back near
`sqrt(eps)` rather than at zero.

Measured: the largest spurious singular value across the degenerate
structures above is **1.97e-8**, on two equal groups of 8. The smallest
genuine deciding value across identified structures is **4.0e-3**. The two
populations are separated by a factor of 2.0e5.

The first default was `1e-8`, which sits *below* that noise floor, and it
reported two equal groups of 8 as **identified** — contradicting their
Proposition 2 outright. The default is now `1e-6`, 50× above the measured
noise and 4,000× below the measured signal, and both sides of that
separation are pinned as tests, including the failing case at `1e-8` so the
regression cannot return silently. Against a dense SVD of the explicit
matrix the Gram singular values agree to better than 1e-7 throughout.

**3. Identified is not estimable, and the failure is silent.** The rank
condition is a population statement. Whether a sample can exploit it is
separate, and it degrades with network density, because a unit's peers'
peers become nearly everybody and `G²x` flattens towards a constant. 400
draws per row, Erdős–Rényi at `n = 300`, true `β = 0.4`:

| Density | Mean degree | Median `β̂` | sd(`β̂`) | Median first-stage F | Rejection at 10% |
|---|---|---|---|---|---|
| 0.015 | 4.6 | 0.402 | 0.226 | 36.5 | 0.105 |
| 0.020 | 6.0 | 0.396 | 0.304 | 30.4 | 0.130 |
| 0.030 | 9.0 | 0.421 | 0.504 | 21.8 | 0.117 |
| 0.050 | 15.0 | 0.378 | 1.160 | 13.2 | 0.087 |
| 0.100 | 29.9 | 0.309 | 1.865 | 6.8 | 0.062 |
| 0.200 | 59.8 | 0.147 | 4.695 | 4.4 | 0.062 |

This reproduces the paper's own simulation finding that precision is a
decreasing function of density. The spread of `β̂` rises
**twenty-one-fold**, the median biases toward zero, and the first-stage F
collapses from 36.5 to 4.4.

The last column is the finding worth recording. **The rejection rate does
not deteriorate — it improves**, from 0.105 to 0.062. The standard errors
grow faster than the estimates do, so the test turns conservative exactly
as the estimate becomes useless, and nothing about the significance of the
result reveals it. That is why `first_stage_F` is computed and warned on at
a threshold of 10, rather than left for the user to think of.

**4. The robust covariance earns its default, conditionally.** Coverage of
a nominal 95% interval for `β`, 2,000 replications, `n = 300`:

| `β` | Errors | `:robust` | `:homoskedastic` |
|---|---|---|---|
| 0.0 | homoskedastic | 0.966 | 0.970 |
| 0.0 | heteroskedastic | 0.970 | 0.942 |
| 0.3 | heteroskedastic | 0.964 | 0.926 |
| 0.6 | heteroskedastic | 0.961 | 0.921 |

Two things, one of which contradicted the guess that prompted the run. The
gap **widens with `β`**: the spatial multiplier `(I - βG)⁻¹` compounds the
heteroskedasticity across the network rather than averaging it away, which
is the opposite of what was expected. And a first pass using mild
heteroskedasticity — a 1.5× spread in the error standard deviation — showed
no separation at all; the table above uses an 11× spread. So the default is
right and will often make no visible difference, and saying only the first
half of that would oversell it.

**5. Sargan size.** A test statistic, so it owes a null distribution. 2,000
replications under correct specification:

| Density | 10% | 5% | 1% |
|---|---|---|---|
| 0.015 | 0.098 | 0.044 | 0.013 |
| 0.030 | 0.094 | 0.045 | 0.009 |
| 0.060 | 0.104 | 0.049 | 0.010 |
| 0.120 | 0.099 | 0.049 | 0.009 |

Correctly sized throughout, including at densities where the estimator
itself has fallen apart — which is worth knowing, since it means a passed
Sargan test says nothing about whether the instruments are strong.

**6. Exactness against independently written code.** The 2SLS estimator
reproduces the formula as printed in their §4, written out separately in
the test file, to **3.8e-13**, in both the plain and the fixed-effects
form. The multiplex fit with a single channel reproduces
`fit_spatial(SAR, …)` — already written and audited — with the maximised
log-likelihood agreeing to **5.7e-14** and the standard errors to 1.6e-10
and 6.0e-10, which is what validates the generalised information matrix
rather than merely the optimiser. The multiplex coordinate ascent is
checked against a dense grid search over the stationary region, since
coordinate ascent can stall on a ridge and that had to be measured rather
than assumed.

**7. A correctness bug the tests found.** The first implementation required
the instrument block to have full column rank. On interaction in two groups
it does not: `G²x` lies in the span of `x`, `G x` and a group indicator, so
one column is redundant — on a network that **passes** the rank condition.
A redundant instrument supplies no information but invalidates nothing; the
orthogonal projector onto the instrument span exists whatever its rank, and
only the `(S'S)⁻¹` form of it fails. The projection is now taken from an
orthonormal basis of that span, and the instrument **rank** rather than its
column count feeds the Sargan degrees of freedom, where counting columns
would have overstated the restrictions.

**8. Why `check = false` cannot rescue a degenerate network.** It was
written as an escape hatch and it does not work on an exactly degenerate
structure — which is not a limitation of the override. Their Appendix B
proves that `I, G, G²` are dependent **if and only if** `E(Gy|x)` is
perfectly collinear with `(ι, x, Gx)`, so when identification fails the
instruments are collinear with the included regressors and no estimate
exists to report. Measured on equal-sized groups, the residual of `E(Gy|x)`
on `(ι, x, Gx)` is 8.9e-16; on a two-size structure, which is identified,
it is not small. Both directions are pinned.

**A chapter attribution was corrected.** Item #11's multiplex row cited
LeSage & Pace (2009, Ch. 9). The book is real and correctly listed, but its
Chapter 9 is *Matrix Exponential Spatial Models*, confirmed against the
Library of Congress CIP record for the 2009 CRC edition, and the book has
no multiplex chapter. The row now cites Lee & Liu (2010), *Econometric
Theory* 26(1), 187–230, verified the same day. This is a weaker failure
than the Cerqua one above — a real reference pointed at the wrong chapter,
not a reference that does not exist — and is worth distinguishing as such.
The three citations the item already carried, Leenders (2002), Manski
(1993) and Bramoullé, Djebbari & Fortin (2009), were each checked against
the primary source and all three hold.

**A fourth copy of the chi-squared survival function was removed.** The
previous pass added `_chisq_sf` to `SpatialShiftShare`, duplicating
`SpatialUtils._chi2_sf`. The suite already carried a test pinning that
function to a single definition — written after three copies once produced
a wrong p-value — and it did not catch this one, because the copy was
spelled differently. The duplicate is gone and that test now pins module
identity rather than name. The shared version is also the more accurate:
the removed copy leaned on `_normcdf` and was good to 1.5e-7 on odd degrees
of freedom, against 6e-9 for the shared one.

**A guarded assertion is worse than a failing one.** Testset `37e` asserted
Corollary 1 as a `@test` inside an `if id.diameter >= 3 && id.connected`
guard, over four structures, one of which was an Erdos-Renyi draw. The
guard is the right *logic* and the wrong *construction*: when it fails the
assertion does not fail, it does not run. The suite stays green and the
only trace is that the assertion count is one lower.

That is exactly what happened, and the cross-version run is what found it.
The same `MersenneTwister(77)` and the same calls give the ER graph **326
edges on 1.13.2 and 288 on 1.10.12** — the standing rule above about pinning
a random draw, in its quietest form, since nothing here pinned a *number*.
At 288 edges the draw came out disconnected, so on 1.10 the guard failed,
the assertion vanished, and the full suite reported **3,179 + 2 broken
against 3,180 + 2** on 1.13.2. A one-assertion gap is the entire signal
available: no failure, no warning, no red CI.

Two changes, because the guard and the draw are separate faults:

| Fault | Fix |
|---|---|
| The assertion was guarded, so a failed guard silently removed coverage | Asserted as the implication it is: `@test !(diameter >= 3 && connected) \|\| identified`. Fires once per structure on every version, whatever the structure turns out to be. |
| The structure was an RNG draw, so what was covered varied by version | All four structures are now deterministic. The ER draw is replaced by a 4x4 grid — diameter 6, neither a line nor a ring — with its connectivity and diameter asserted outright, so the implication is entered rather than satisfied vacuously. |

`37e` goes from 10 assertions on 1.13.2 and 9 on 1.10.12 to **12 on both**.
The generalisable part, and the reason this is in the ledger rather than
just fixed: **an assertion count that differs across environments is a
finding, not a reconciliation exercise.** The temptation on seeing 3,179
against 3,180 is to record both figures with their versions, which the
roadmap already does for the extensions and which is correct there, because
there the mechanism is known. Here the mechanism was unknown, and the
difference was coverage going missing.

**What is not here.** No network formation model: links are taken as given
and strictly exogenous, which is the paper's own maintained assumption and
the one its conclusion names as the central remaining problem. No global
differencing (their §3.3 and Proposition 5). No GMM alternative to 2SLS.
The multiplex fit has no spatially autoregressive disturbance, so it is the
first half of Lee and Liu's model rather than all of it. Each is additive
and none is implied by the row's scope.

## Post-audit: weak versus strong cross-sectional dependence

Two roadmap rows shipped together on 2026-11-04 — the diagnostics that say
whether a common factor is present (`SpatialCommonFactors`) and the
estimators that survive one (`SpatioTemporalModels`). They were folded into
one pass deliberately: separately, one is a test whose remedy does not
exist and the other a remedy with no test to motivate it.

Suite 3,217 → 3,415 with Makie and Turing installed, and 3,182 → 3,380 with
both absent, zero failures either way, on **Julia 1.13.3 and 1.10.13** —
both channels having moved on from the 1.13.2/1.10.12 that earlier sections
of this file name, so the versions were re-read rather than carried
forward. The extensions-absent count is identical on the two versions.
Testsets 38 and 39 contribute 117 and 81 assertions, and the step of 198
balances exactly. Docs build clean under `checkdocs = :exports`, and
`examples/27_common_factors.jl` and `examples/28_spatio_temporal.jl` both
run green.

### The number that justifies the pair

Every design below has a genuine common factor `λᵢfₜ`. Nothing changes but
the *distribution of the loadings*. Bias / RMSE in `ρ̂`, true `ρ = 0.4`,
`N = 42`, `T = 30`, 200 replications:

| loadings | `NoFE` | `TwoWayFE` | `fit_sp_cce` | `fit_interactive_fe` |
|---|---|---|---|---|
| mean 2 (dominant) | **+0.484** | +0.002 / **0.029** | −0.001 / 0.047 | −0.032 / 0.041 |
| mean 1 | **+0.285** | +0.003 / 0.068 | −0.000 / 0.047 | −0.017 / **0.032** |
| mean 0, random | −0.004 | +0.003 / 0.068 | −0.008 / 0.059 | −0.002 / **0.025** |
| mean 0, balanced ±1 | **+0.329** | **+0.373** ⚠ | +0.005 / 0.100 | −0.023 / **0.035** |

Neither diagonal of that table is the expected story.

**`TwoWayFE` wins outright when the loadings have a strong common mean**
(RMSE 0.029 against CCE's 0.047). This is worth stating because the
reflexive advice is the opposite: reach for the more general estimator.
Here the general estimator loses, and for an understandable reason — a time
effect is *exactly* a factor with equal loadings, so on that design
`TwoWayFE` is correctly specified and parsimonious, while CCE spends `N·q`
projected columns buying nothing. The exactness is measurable, not
rhetorical: a factor with `λᵢ ≡ λ` is removed by the within-transform to
`5.55e-16`, and a heterogeneous one keeps 99.9% of its variance.

**`TwoWayFE` is worse than doing nothing when the loadings are balanced in
sign** — `+0.373` against `NoFE`'s `+0.329`. Removing a cross-sectional
mean that is already zero destroys variation and leaves the factor
untouched. That is the only cell in the table where a correction is
actively harmful, and it is the cell the diagnostics below are least able
to warn about, which is the whole point of the next section.

### Two failure configurations, measured rather than inferred

A docstring in the first draft said the CD test's blind spot is that
cancelling loadings give `ρ̄_N ≈ 0` and no rejection. That was a plausible
inference from eq. (9) and it is wrong. Measured, `N = 40`, `T = 60`, 2,000
reps, 5% level:

| loadings | CD rejection | `α` identified? | mean \|ρ̂\| |
|---|---|---|---|
| independent | 0.047 | no (correctly) | 0.104 |
| homogeneous | 1.000 | yes, 300/300 | 0.496 |
| mean-zero random | **0.348** | sometimes | 0.293 |
| balanced ±1 | 0.911, **all negative** | **no, 0/300** | 0.911 |

There are two distinct failures and neither is a quiet non-rejection.

With **random mean-zero loadings** CD's power collapses to about a third
and **does not recover as `N` grows**: 0.377 at `N = 10` against 0.343 at
`N = 160`. It is structural, not a small-sample matter, and that is the
sharper statement — a power of 0.348 that improved with `N` would be a
sample-size problem.

With **balanced ±1 loadings** CD rejects 91% of the time, but the statistic
is **negative**: a strong common factor reported as negative average
dependence, because `ρ̂ᵢⱼ ∝ λᵢλⱼ` and the signs cancel in the mean while
accumulating in magnitude. A user reading only the p-value concludes
correctly by accident, and a user reading the sign concludes the opposite
of the truth.

What survives both is the **average absolute** correlation, so it is
returned as `abs_rho_bar` alongside its independence benchmark
`sqrt(2/(πT))` — measured 0.1036 against a predicted 0.1030. It flags 1.000
of mean-zero-loading draws where the signed statistic manages 0.348. This
is why `dependence_verdict` exists and why it checks the absolute average
against its benchmark *before* trusting the signed statistic.

### A measurement that seemed to overturn a published result

`alpha_exponent`'s bias correction, BKP eq. (15), came out **worse than the
uncorrected eq. (14) in every cell** of the first run, with the estimator
biased −0.05 to −0.09 across the admissible range. The paper says the
correction should help in the middle of that range. The first instinct was
to treat eq. (15) as not worth its default.

That instinct was wrong, and the general lesson is worth the space: **a
measurement that appears to overturn a published result is first evidence
about the implementation.** The resolution was arithmetic.
`Var(x̄ₜ) = κ²N^(2α−2) + …` with `κ² = σ_f²μ_v²`, so `α̂` is estimated *for
a given* `κ²` — and the implementation standardises each series, which is
what makes `α̂` scale invariant, thereby setting `κ²` to the signal share
rather than 1. At `κ² = 0.5` the predicted shift is
`½ln(0.5)/ln(50) = −0.0886`; the measured bias at `N = 50` was `−0.0883`.
The entire discrepancy, to three decimals.

With `κ²` passed correctly the bias is `+0.0003`, and eq. (15) then beats
the baseline in the middle of the range exactly as BKP state: RMSE 0.0106
against 0.0202 at `α = 0.70`. So `bias_correct = true` stayed the default,
`kappa2` became a documented keyword, and the docstring records that
`kappa2 = 1` is **a convention, not a measurement** — which is the part a
user actually needs, and the part a default that merely looked validated
would have hidden.

### Exactness gates against already-audited code

The most useful checks here are not Monte Carlo. Where a new estimator has
a special case an existing audited one already covers, the agreement should
be exact, and five such cases were available:

| Reduction | Agreement |
|---|---|
| `fit_interactive_fe(n_factors = 0)` against `fit_spatial_panel` | log-likelihood **0.000e+00**, `ρ̂` 9.4e-9 |
| `fit_star` against the normal equations written out in the test file | coefficients 3.9e-16 / 1.3e-15, `σ̂²` **0.000e+00** |
| cumulative space-time IRF against `(I − ΣₗAₗ)⁻¹` | 1.1e-15 |
| `pesaran_cd` on a matrix against the same data as a vector | **0.000e+00** |
| interactive-FE factor *projection* across starting values | 5.2e-10, and 0.000e+00 under a sign flip |

The `σ²` comparison in the first of these differed at 0.9274 against
0.9336, which is exactly the 600/596 degrees-of-freedom convention — the
reference reports the ML value and this implementation the dof-corrected
one. Documented rather than changed, since neither is wrong.

The last row is the one that matters for interactive fixed effects.
**Factors and loadings are identified only up to rotation**, so for any
invertible `R` the pair `(FR, ΛR⁻ᵀ)` fits identically, and a test comparing
`m.factors` across fits would be testing the starting value. The identified
object is the projection `F(F'F)⁻¹F'`, which is what `factor_projection`
returns and what the suite pins. Getting this wrong is the standard error
in implementations of Bai (2009), and the sign flip — the simplest rotation
there is — leaves the projection unchanged to exactly `0.0e0`.

### The cross-version run earned its keep again, in a different way

The two-version rule caught something on this pass too, and the shape is
worth distinguishing from the `37e` case above. There the totals differed
by one and nothing failed. Here the **totals matched exactly** — 3,381 on
both — and one assertion *failed* on Julia 1.10.13 while passing on 1.13.3:

```
38i. the verdict ties the two together: Test Failed
  Expression: v.kind === :strong
   Evaluated: semi_weak === strong
```

So the diagnostic that finds a guarded assertion (count mismatch, no
failure) would not have found this one, and a one-version run would not
have found it at all.

The cause is not a defect in `dependence_verdict`, and that was worth
establishing before changing anything. `:strong` requires `α̂ > 0.9`. On the
test's design — `N = 60`, `T = 80`, a genuine `α = 1` factor with loadings
of mean 1 and sd 0.2 against unit noise — `α̂` is **0.9093 with sd 0.0109**
over 400 draws. The threshold sits 0.85 sd from the mean, so **17% of draws
classify as `:semi_weak`**. The test did not fail because 1.10 is different;
it failed because it had roughly a one-in-six chance of failing on any
stream, and 1.13.3 drew one of the five.

The downward bias is itself the `κ²` effect documented two sections above:
`kappa2 = 1` is a convention, the standardisation sets the true `κ²` to the
signal share, and `α̂` is biased down by `½ln(κ²)/ln(N)`. At a signal share
near 0.5 and `N = 60` that is about `−0.08`, which is the whole gap between
0.9093 and a true 1.0. Nothing is broken; the estimator is doing what its
own docstring says.

Two changes, because the fragile test and the blunt boundary are separate
issues:

| Issue | Response |
|---|---|
| The assertion sat 0.85 sd from a hard threshold, so the draw decided the outcome | Design strengthened to loadings of mean 2, where the minimum `α̂` over 400 draws is 0.9516, and the **margin is now asserted** (`α̂ > 0.93`) rather than only the verdict. The claim is now a property of the design. |
| A genuine `α = 1` factor of moderate strength is labelled `:semi_weak` about one time in six, and nothing said so | Recorded as a warning admonition on `dependence_verdict`, with the measured mean, sd and rate. The threshold was **not** moved: 0.9 is BKP's own characterisation, and moving a principled threshold to make a test pass would be the wrong repair. |

The mitigating fact, stated because it bounds how much the mislabel costs:
the `:semi_weak` recommendation is to compare `fit_spatial_panel` against
`fit_sp_cce` and read a material move in `ρ̂` as the factor being absorbed.
That is the right action on an `α = 1` panel too. The verdict understates
the kind; it does not misdirect the user.

The generalisable point, and the reason this is in the ledger rather than
just fixed: **an assertion whose subject is a continuous statistic needs its
margin measured, not assumed.** `v.kind === :strong` reads like a
categorical claim and is really a claim about a distribution's position
relative to a cutoff. The standing rule above says a new diagnostic needs a
null-distribution Monte Carlo; the corollary this pass adds is that a new
*assertion about* a diagnostic needs to know how far it is from flipping.

### An attribution gap, recorded as the lesser failure it is

The roadmap cited Pesaran (2006) and Ertur & Musolesi (2017) for spatial
CCE. Ertur & Musolesi is an *application* paper on international technology
diffusion. It is correctly cited for the weak/strong distinction, but it is
not the methodological source for a spatial CCE *estimator*. That is
**Yang, C. F. (2021), *Econometric Reviews* 40(1), 14–50**, verified
against the published record and now added.

This belongs beside the LeSage & Pace chapter error above, and is the same
weaker class: a real reference doing real work, pointed at a claim it does
not make, rather than a reference that does not exist. Ertur & Musolesi was
not removed — it was never a phantom. The failure is that the
methodological citation was *missing*, and a missing citation is harder to
notice than a wrong one precisely because nothing in the row looks broken.

### What is not here

No **SpVAR** and no **time-varying `ρ`**. Both belonged to the roadmap rows
folded into this pass, and both were deferred by explicit decision rather
than silently dropped; each is a substantial estimator in its own right,
and they are recorded as a named follow-up row rather than left to be
rediscovered. (**SpVAR was built immediately afterwards, the same day** —
see the next section. The deferral held for one pass, not permanently, and
this paragraph is left standing rather than rewritten because the record of
what a pass did not contain is worth more than a tidy one.) No GMM alternative to the CCE 2SLS. No Shi & Lee (2017)
dynamic interactive-effects estimator — `fit_interactive_fe` is static in
`y`. `fit_star` has no contemporaneous `Wyₜ` by construction, which is the
specification's defining assumption rather than a gap: that model is
`fit_spatial_panel`, and neither nests the other.

One risk is worth stating rather than hiding. Bai's iterative principal
components converges to a *local* optimum, and agreement across starting
values is evidence but not proof of a global one. The starts are therefore
exposed as `n_starts` rather than fixed internally, and the measured
agreement above is reported as exactly that.


## Post-audit: a spatial VAR, and three things it caught

`SpVAR` implements Di Giacinto (2010), *On vector autoregressive modeling in
space and time*, Journal of Geographical Systems 12(2), 125–154,
doi:10.1007/s10109-010-0116-6. **Verified against the published record**,
and read in full — not from the journal, which is paywalled, but from Bank
of Italy *Temi di discussione* **746**, which is freely available and
carries the same text. Every equation quoted in the module and its docs
page was extracted from that PDF with its own equation number. Greek glyphs
drop out of the symbol font under `pdftotext`, so symbol identities were
fixed from the surrounding prose — the same trap as BDF in #11 and Pesaran
in #1, and the third time it has had to be worked around.

Suite **3,683 → 3,718** with Makie and Turing loadable and **3,380 →
3,683** with both absent, zero failures either way, on **Julia 1.13.3 and
1.10.13**. The extensions-absent figure is **identical on the two
versions**, measured separately, so there is again no cross-version
assertion difference. Testset 40 contributes **303**, and the step from
3,380 is 303 exactly — balanced at every one of the three points at which
testset 40 grew, even though this pass also touched
`src/SpatialWeights.jl`.

Note which entry point gives which figure, because it is easy to invert:
`Pkg.test()` is the one that genuinely **excludes** the extensions, so it
reports 3,683 plus two broken markers; running `test/runtests.jl` directly
leaves `@v1.13` on `LOAD_PATH`, the extensions load, testsets 25 and 26 run
rather than reporting a marker, and the total is 3,718 with no broken
entries. 3,683 + 11 + 24 = 3,718, the markers being replaced rather than
added.

### What is exact, and pinned as such

Six quantities here are algebraic, so none of them is checked against a
tolerance chosen to pass.

| Identity | Why it matters | Measured |
|---|---|---|
| `|C₀| = Π_r |A⁽⁰⁾_rr|` | the estimator's whole speed rests on it | `0.000e+00` |
| `L̃ = Σ_k L̃_k`, eq (23) | concentrating `Λ` out separates the likelihood into `K` unrelated terms, so a spatial VAR is `K` spatial autoregressions and not one `NK×NK` problem | `0.000e+00` |
| `s = 0` ≡ nested panel VAR, eqs (15)–(18) | `fit_nested_panel_var` was written independently, which is what makes the reduction worth testing | `0.000e+00` in loglik and in every coefficient |
| `K = 1, s = 1` Jacobian | at `K = 1` the structural form is a single SAR, so `|C₀|` must be `|I − ρW|` | `< 1e-10` |
| `Σ_h Ψ_h = (C₀ − Σ_h C_h)⁻¹` | the long-run multiplier | `2.665e-15` |
| outward STIR ≡ inward STIR at `K = 1`, symmetric `W` | the two differ by a transpose, so this is a free check on the whole IRF chain | `0.0e+00` (ring), `2.776e-17` (spectrally standardised path) |
| order condition eq (14) | reproduces the paper's own worked `K=2,s=1 ⇒ N≥3` and `K=2,s=2 ⇒ N≥4` as integer bounds | exact, and the independently derived free-coefficient count matches the paper's printed `NK(K−1)/2 + sNK(K+1)/2` at every `N, K, s` tested |

### How `μ` is concentrated out, and why the reduction is not exact

The `s = 0`, `K = 1` reduction should be an ordinary heteroskedastic AR(1)
panel, and it is — but not to machine precision, which is worth explaining
because the other five reductions above are exact.

`fit_spvar` absorbs the unit intercept by demeaning each unit's **level**
series once, over `t = p+1 … T`. That is not a convenience: the spatial lag
of a demeaned series is not the demeaned spatial lag, since
`mean_t(Wyₜ)ᵢ = (Wȳ)ᵢ ≠ ȳᵢ`, so demeaning the constructed regressor columns
separately would break `(I − ρW)ỹ` and the Jacobian with it. Demeaning the
series preserves the structural form exactly, with `μ` absorbed as
`μ̃ = (I − ρW)μ`.

At `s = 0` there is no spatial term, and the exact profiled likelihood
instead demeans the lagged and current sub-series by their **own** means,
which differ by `(y_T − y₁)/(T−1)`. So the two disagree in finite samples.
Measured against an independently written profiled FIML, `N = 10`,
`φ = 0.55`, heteroskedastic `λ ∈ [0.5, 1.5]`:

| `T` | 150 | 300 | 600 | 1200 | 2400 | 4800 |
|---|---|---|---|---|---|---|
| gap | 6.23e-5 | 1.45e-5 | 4.42e-6 | 1.11e-6 | 3.90e-7 | 5.06e-8 |

The first guess was `O(1/T)`, since that is the order of the difference
between the two means. **It is wrong.** The ratio per doubling averages
about **4**, not 2; `T·gap` falls steadily from 0.0093 to 0.0002 rather
than stabilising, while `T²·gap` is roughly constant in 1.2–2.3. The two
ways of concentrating `μ` therefore agree to **second** order — a stronger
result than the one assumed, and the reason this is recorded as a
documented property of the estimator rather than fixed as a defect.

Testset `40m` asserts the implication rather than the rate: a fourfold
increase in `T` must shrink the gap by far more than fourfold. The measured
ratio is 13.1 against the 5 asserted, a margin of 2.6× — checked before
being trusted, which is the `38i` rule from the same day.

### Two estimators that look comparable and are not

`fit_dynamic_spatial_panel` is the obvious thing to pin `fit_spvar` against
at `K = 1, s = 1`. It cannot be, and the way it fails is informative.

Handing it `y(t−1)` as a regressor **refuses**, with an identification
error: it builds its own time lag, so the column is exactly collinear with
it. That refusal is pinned in testset `40m`, because it is the reason the
two cannot be compared coefficient by coefficient.

Fitted as intended instead — an exogenous regressor, the lag left to the
estimator — it returns `ρ̂ = 0.1875` on data where `fit_spvar` returns
`0.3749` against a true `ρ = 0.40`. The gap is not an error in either: the
dynamic spatial panel has **no `W yₜ₋₁` term**, and the DGP has one with
coefficient 0.10. Omitting it does not leave `ρ` alone; it roughly halves
it, because the spatial content of the lagged spatial lag has nowhere to go
but the contemporaneous spatial parameter. `fit_spvar` also carries a
per-unit `λᵢ` where the dynamic panel assumes a common variance.

The two models do not nest, and the 2× discrepancy in `ρ̂` is what a user
should expect from leaving out a spatial lag of the lagged dependent
variable. Reported rather than averaged away, because "the estimates
disagree" is the kind of result that gets quietly attributed to one of the
two implementations being wrong.

### The asymptotics run the other way, and that was measured not asserted

Di Giacinto's asymptotics are `T → ∞` at **fixed `N`**. Every other panel
estimator in this package is consistent as `N` grows, so the temptation was
to reach for the large-`N` habits in `SpatialPanelEconometrics.jl`. The
gate was designed to make the regime visible rather than assumed. Bias in
the dynamic coefficient `g(t−1)` (truth 0.30), homogeneous SpVAR(1,1),
`K = 2`, ring, 150 replications:

| `N` = 12, `T` growing | bias | | `T` = 50, `N` growing | bias |
|---|---|---|---|---|
| `T = 50` | −0.0337 | | `N = 12` | −0.0267 |
| `T = 100` | −0.0180 | | `N = 24` | −0.0318 |
| `T = 200` | −0.0084 | | `N = 48` | −0.0296 |
| `T = 400` | −0.0040 | | `N = 96` | −0.0294 |
| `T = 800` | −0.0014 | | | |

Growing `T` removes the bias by a factor of 24. Growing `N` by a factor of
eight does not move it at all.

`N` is not useless — it buys variance, and that separation is the point.
Measured independently in `examples/29_spvar.jl` section 7 at 60
replications, which reports both: at `T = 50`, going from `N = 12` to
`N = 96` moves the bias from −0.0344 to −0.0289 while RMSE falls from
0.0559 to 0.0337; at `N = 12`, going from `T = 50` to `T = 800` moves the
bias from −0.0205 to −0.0025 and RMSE from 0.0469 to 0.0108. So `N`
tightens the sampling distribution around the wrong value and `T` moves it
onto the right one. The spatial coefficient `ρ₁` is far less affected in
either direction, which is why the docstring warning is about the
*dynamics* rather than the spatial parameters.

Coverage of the nominal 95% interval, 400 replications (Monte Carlo
standard error about 0.011):

| | `ρ₁` | `g(t−1)` | eq-2 `g` |
|---|---|---|---|
| `N = 12, T = 100` | 0.935 | 0.935 | 0.948 |
| `N = 12, T = 200` | 0.945 | 0.925 | 0.953 |
| `N = 12, T = 800` | 0.955 | 0.950 | 0.948 |
| `N = 24, T = 200` | 0.960 | 0.940 | 0.927 |

Close to nominal throughout, slightly short for the dynamic coefficient at
small `T`, and not improving with `N` — the same finite-sample bias showing
up in the interval rather than the point estimate.

### 150 replications said the opposite about the LR test

The first size table for `lr_spatial_interaction` ran at 150 replications
and reported the test **conservative**: 0.073–0.087 at the 10% level,
0.033–0.047 at 5%, 0.000–0.007 at 1%. At 1,500 replications it is mildly
**liberal** at 10% — 0.110–0.119 — and close to nominal elsewhere.

Nothing changed in the code between the two runs. The Monte Carlo standard
error at 150 draws is 0.024 at the 10% level, wide enough to swallow the
whole finding and reverse its sign. This is the `38i` lesson from the same
day in a second form: there, an assertion passed on one Julia version and
failed on another because its margin was 0.85 standard deviations; here, a
size table was briefly believed because nobody had compared the deviation
being claimed against the error bar on it. **A size table is not worth
printing until the replication count makes its error bar smaller than the
deviation it claims.** Both runs are recorded, not just the one that
survived.

### Three defects the gates caught

**A refusal path that could not run.** `order_weights_list` refuses when an
order is empty, and the message interpolates the order into
`W⁽$l⁾`. Julia treats `⁾` as an identifier character, so `$l⁾` parses as a
variable named `l⁾` and the refusal threw `UndefVarError` instead of the
`ArgumentError` it was written to throw. The error path was therefore
**completely broken while the happy path was fine**, and it was caught only
because testset `40a` asserts the refusal rather than assuming it. Fixed by
writing `$(l)`. A reminder that `@test_throws` on a *specific* exception
type is doing real work: `@test_throws Exception` would have passed.

**`const` inside a `@testset`.** Three parameter tuples at the head of
testset 40 were declared `const`, which is a syntax error on a local
variable — *"syntax: unsupported `const` declaration on local variable"*.
This would have taken the whole suite down, and was caught by running the
new testset in isolation before running the suite, which is now the habit.

**A refusal test masked by an earlier refusal.** Testset `40j` checked that
`T_per > p + 1` is enforced, by passing `T_per = 2`. But `Y` still had
`N×T` rows, so the row-count check fired first and the test was really
checking the dimension guard twice. The assertion passed for the wrong
reason until the exception *type* was pinned. Fixed by reshaping `Y` to
match. Guard ordering matters, and a test that names the wrong guard is
worse than no test because it reads as coverage.

### `W⁽ˡ⁾` is not `Wˡ`, and nothing in the package built it

Di Giacinto's `W⁽ˡ⁾` is the order-`l` weights matrix in the sense of Anselin
and Smirnov (1996): pairs at graph distance exactly `l`. `Wˡ` is not that.
Powers count **walks**, so they are non-zero on the diagonal, double-count
paths, and where the graph has triangles they retain first-order neighbours
at order 2. Nothing in `src/` built order-`l` weights —
`HigherOrderSpatial.jl` is SLX/SAC/SDEM, which is two weight matrices, not
a hierarchy — and no roadmap row planned one.

`order_weights` and `order_weights_list` were therefore added to
`SpatialWeights.jl`, by breadth-first shortest-path hop count. Measured on a
4×4 rook lattice, `N = 16`, 48 first-order links:

| | non-zeros | on the diagonal |
|---|---|---|
| `W²` | 84 | 16 |
| `order_weights(W, 2)` | 68 | 0 |

and summed over orders 1 to 6 the order matrices cover exactly
`N(N−1) = 240` ordered pairs, each once. On a triangle with one pendant
unit, `W²` is non-zero on 6 pairs that are already first-order neighbours;
`order_weights(W, 2)` is non-zero on 0 of them. The trap is made into a
test rather than described in a comment.

### `p > 1` was shipping unexercised, and the ledger said otherwise

The sentence above, in its first form, read: *"No `p > 1` worked example,
though `p` is implemented and tested."* The second clause was **false**. Of
twelve subsets in testset 40, every one ran at the `p = 1` default; `p = 2`
appeared nowhere in the suite. The claim was written in the same pass that
should have checked it, which is the most instructive part: a ledger entry
asserting coverage is itself an assertion, and it got less scrutiny than
any `@test` in the file precisely because it was prose.

It mattered more than the `binary = false` gap. At `p = 1` the companion
matrix **is** the reduced-form matrix, so its block structure — the
`Kp·N`-square form with the shifted identity below — is first exercised
only at `p = 2`, as are the loop over `C_h`, the `(t−2)` coefficient names
and the longer history a forecast requires.

Measured on a SpVAR(2,1), `K = 2`, `N = 12`, `T = 600`, ring:

| check | result |
|---|---|
| `Σ_h Ψ_h` vs `(C₀ − C₁ − C₂)⁻¹` | **8.882e-16** |
| `ρ₁` recovery, both equations | 0.3049 [0.30], −0.1856 [−0.18] |
| companion radius, fitted vs DGP | 0.5251 vs 0.4924 |
| `length(m.C)`, `m.T_eff` | 2, `T − 2` |
| coefficient names | both `(t−1)` and `(t−2)` present, in equal counts |
| forecast with one period of history | refused, `DimensionMismatch` |

The long-run multiplier is the gate that would catch a dropped or
double-counted lag matrix, and at `8.882e-16` it does not. **Nothing was
wrong with the `p > 1` code.** The defect was in the verification, and in a
ledger line that described verification which did not exist. Now testset
`40n`.

`rho_bounds` was in the same condition — documented, never exercised — and
is now pinned too: bounded to `(0.5, 0.9)` where the unconstrained optimum
is near 0.30, the estimate sits **on** the boundary at `0.500000` rather
than outside it, and the bounded fit's log-likelihood is strictly worse.

Three options in this module were found undertested in one sweep
(`binary = false`, `p > 1`, `rho_bounds`), which suggests the sweep was
worth doing and should be part of closing a row rather than an afterthought:
**list the keywords, grep the suite for each, and test what comes back
zero.** `maxiter` and `tol` remain exercised only at their defaults, which
is recorded here rather than claimed otherwise.

### A documented keyword that no run had exercised

`order_weights` takes `binary = false`, which weights each order-`l` link
by `1/d` where `d` is the number of distinct **shortest** paths. The
docstring described it. Nothing ran it — every gate, the example and the
whole of testset 40 used the default — so a documented option was shipping
on the strength of having been written.

That is precisely what this project's standing rule forbids, and the rule
had been applied all pass to *statistics* while a *keyword* slipped past.
It was checked against an independent closed form rather than a tolerance:
on a rook lattice every shortest path is monotone, so the number of
shortest paths from `(1,1)` to `(a+1,b+1)` is exactly the binomial
`C(a+b, a)`.

| from `(1,1)` to | `l` | measured | `1/C(a+b,a)` |
|---|---|---|---|
| (2,2) | 2 | 0.50000000 | 1/2 |
| (3,2) | 3 | 0.33333333 | 1/3 |
| (3,3) | 4 | 0.16666667 | 1/6 |
| (4,2) | 4 | 0.25000000 | 1/4 |
| (4,3) | 5 | 0.10000000 | 1/10 |
| (4,4) | 6 | 0.05000000 | 1/20 |

Exact at `1e-12` on all six, plus `1/2` on a 4-ring's opposite unit (two
shortest paths), `1.0` at order 2 on a 6-ring (one path) and `1/2` at its
order-3 diameter (two paths). The two settings are also pinned to produce
the **same sparsity pattern**, differing only in values, with
`binary = true` all ones before standardisation. Now testset `40a2`.

The lesson is narrow and worth having: **the standing rule covers options,
not just outputs.** A keyword is a claim about behaviour in exactly the way
a diagnostic is, and a keyword nobody has run is weaker evidence than a
diagnostic nobody has simulated, because at least the diagnostic gets
exercised by its own happy path.

### Row standardisation alone splits outward from inward

The obvious guess is that the outward and inward STIRs coincide whenever
`K = 1`. They coincide when `W⁽ˡ⁾` is **symmetric**, which is not the same
thing, and `order_weights_list` row-standardises by default. On a 14-unit
path, row standardisation gives `max|W − Wᵀ| = 0.5` and an outward/inward
gap of `1.345e-02`; spectrally standardised, the same path has zero
asymmetry and a gap of `2.776e-17`. A ring is regular, so row
standardisation preserves its symmetry and the gap is `0.0e+00` there —
which is why the first version of this gate looked like a bug in
`space_time_stir` and was not one. The premise was wrong, not the code.

Leaving the weights raw is **not** the alternative: a binary path has
eigenvalues near 2, so the DGP used throughout these gates is explosive on
raw path weights. `simulate_spvar` in `examples/29_spvar.jl` refuses, with
the spectral radius in the message, which is the guard working rather than
an obstacle.

### A forecast benchmark that flattered the model by accident

`examples/29_spvar.jl` first compared `spvar_forecast` against a **zero**
forecast and reported RMSE 1.3671 against 1.4035 — a three percent gain.
Two things were wrong with that, and only one of them was visible.

The benchmark was wrong. `fit_spvar` absorbs a per-unit intercept, so the
honest comparison is each unit's in-sample mean, not zero. A zero forecast
only looks competitive on a DGP whose levels happen to sit near the origin,
which that one did (`μ ∼ N(0,1)`). And **a zero benchmark cannot detect
whether the absorbed intercepts were restored at all** — the most likely
defect in a forecast built from a demeaned fit.

To be clear about what was and was not at risk: the suite already pins
that. Testset `40h` reconstructs the one-step forecast from the reduced
form by hand and compares it to `spvar_forecast` including the
`+ m.means[i, k]` term, at `rtol = 1e-10`, so a dropped intercept fails the
suite. The weak benchmark was in the **example**, not the tests, and the
example is documentation rather than verification. The near-miss is
therefore smaller than it first looked, and saying so is part of recording
it accurately.

It was still checked directly, on a DGP with intercepts of order 50 and
−30 where forgetting them would be unmissable. `spvar_forecast` restores
them: forecast means `255.012` and `122.295` against actual `255.061` and
`122.331`, with RMSE `1.2510` and `1.2128` against `9.4165` and `5.3837`
for the sample-mean forecast — gains of 7.5× and 4.4×. The level forecast
is correct, and the original example was understating it.

The second problem was the averaging. Over twelve horizons the gain reads
as 1.03×. Per horizon:

| `h` | g: model | g: mean | y: model | y: mean |
|---|---|---|---|---|
| 1 | 1.1978 | 1.4670 | 1.0990 | 1.2871 |
| 2 | 1.3701 | 1.4629 | 1.9033 | 1.8993 |
| 3 | 1.3310 | 1.3107 | 1.7909 | 1.7815 |
| 4 | 1.3254 | 1.4050 | 1.2706 | 1.3169 |
| 8 | 1.2381 | 1.2838 | 1.1547 | 1.1549 |
| 12 | 1.7636 | 1.7864 | 1.1726 | 1.1952 |

The gain is at `h = 1`, essentially gone by `h = 2`, and at `h = 3` the
model is **marginally worse** than the sample mean. That is what a
companion spectral radius of 0.74 implies, and past a few periods these are
one draw over 14 units and mostly sampling noise. The example now prints
both cuts.

Two lessons, both general. **A number printed next to a result is read as
a check, so it had better be capable of failing** — the zero forecast was
not, and an example's printed comparison carries the same rhetorical weight
as a test without any of the force. And **an average over horizons can turn
a correct short-horizon forecast into an apparent failure**, which is the
same error in the opposite direction from the one the benchmark made.

### The order condition binds in the right order, and for the right reason

Eq (13)'s counts are the **heterogeneous** ones, which is the condition the
paper prints. `fit_spvar` ships the homogeneous case (eq 8), so it applies
the homogeneous count: refusing on the heterogeneous bound would refuse
models it can in fact estimate. Measured at `K = 2, s = 2` on a ring, 60
fits per cell:

| `N` | homogeneous condition | heterogeneous condition | fits | mean companion radius |
|---|---|---|---|---|
| 3 | satisfied | **fails** | — | refused by `order_weights_list`: a 3-ring has diameter 1, so `W⁽²⁾` is empty |
| 4 | satisfied | satisfied | 60/60 | 0.711 |
| 5 | satisfied | satisfied | 60/60 | 0.727 |
| 8 | satisfied | satisfied | 60/60 | 0.726 |

So at the one `N` where the two conditions disagree, a third guard — the
empty-order check — refuses first, and refuses for a reason that is true
regardless of which condition one believes: a structurally zero column has
no identified coefficient. The guards are not redundant and they fire in a
defensible order. The order condition remains **necessary, not
sufficient**, and it is labelled that way in the docstring; `check_design_rank`
is the second line of defence within a given sample.

### The ordering is the result

`ordering` has no default in `fit_spvar`, and the reason is a number. On a
14-region ring with `T = 400`, reversing the contemporaneous causal order of
a two-variable system moved the log-likelihood by **9.949** and the peak
outward STIR of `y` to a shock in `g` by a factor of **5.25** at spatial
order 0 and **1.33** at order 1. It also moved the **date**: the order-0
response peaks on impact under the true ordering and at `h = 2` under the
reversed one, because the effect then has to travel through a lag to exist
at all.

A default would have hidden the only choice the user is obliged to make,
and would have made the package complicit in the most common failure in
applied SVAR work. This is the same decision as refusing rather than
returning a number for an under-identified system: the estimator declines
to guess on the user's behalf.

### What is not here

No **bootstrap confidence bands for the STIR** (the paper's §5 tail); the
point estimates and asymptotic intervals stand without them, and the
coverage table above is what justifies saying so. No **spatially
heterogeneous coefficients** — the full eq (7), with `Φ` genuinely diagonal
rather than scalar. The global STIR summaries assume homogeneity and
moderate `N` cannot support the heterogeneous count, but the order
condition is implemented for **both** parameterisations regardless, since
its printed form is the heterogeneous one. No **`p > 1` worked example**, though `p` is
implemented and now tested — see below; the sentence that first stood here
claimed it was tested when it was not.

**Pfeifer and Deutsch (1980)**, *Technometrics* 22(1), 35–47, could not be
read. tandfonline returns 403 to automated access and JSTOR returns an
interstitial, the same wall as the `stpca` sources. The open question it
would settle is whether STARMA's `W⁽ˡ⁾` means order-`l` neighbourhoods
rather than powers of `W`, which bears on `fit_star`'s docstring
attribution and not on its behaviour: the two conventions **agree at
`l = 0` and `l = 1` and diverge only at `S ≥ 2`**, and `spatial_lags = 1`
is the default. The question is recorded as open rather than answered from
recollection, which is the rule that caught the LeSage & Pace chapter error
and the Ertur & Musolesi attribution gap. `order_weights_list` now exists,
so if the answer turns out to be order-`l`, exposing it in `fit_star` is an
option rather than a rewrite — and it would be filed as its own row, since
shipped tested behaviour does not change silently to match a citation.
