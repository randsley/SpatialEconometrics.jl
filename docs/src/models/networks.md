# Spatial Econometrics for Networks

Everywhere else in this package `W` comes from distance. Here it comes from
a graph — trade links, friendships, supply chains, citations — and one thing
changes that matters more than the construction does.

On a map, a unit's neighbours and its neighbours' neighbours are different
sets, because distance decays. On a network that is not automatic, and when
it fails the peer-effects model is **not identified**: the response to
peers' outcomes and the response to peers' characteristics cannot be told
apart at all. Bramoullé, Djebbari and Fortin (2009) make this precise, and
the failure lands on the structure applied work reaches for first.

So the first function here is not an estimator.

## The reflection problem, mechanically

The linear-in-means model, their eq. (1):

```
y = αι + βGy + γx + δGx + ε
```

`β` is the **endogenous** effect — the response to what peers do — and `δ`
the **contextual** effect — the response to who peers are. Separating them
is the whole point, and it is the thing that fails.

Their **Proposition 1**: social effects are identified if `I`, `G` and `G²`
are linearly independent; if those are linearly dependent and no individual
is isolated, social effects are not identified. Their footnote 17 gives the
check, which is what [`peer_identification`](@ref) runs: vectorize each
matrix, stack the columns, and take the rank of the result.

With network fixed effects — an unobservable correlated with `x` at the
network level, swept out by the within-local transformation `(I - G)` —
their **Proposition 4** requires `I, G, G², G³` to be independent instead.
Strictly more demanding, because, in their words, information has been lost
to take into account the presence of correlated effects.

### What fails, and what does not

Every verdict here is pinned as a test, and the algebraic identities behind
them hold to machine precision because they are identities in the paper.

| Interaction structure | Prop. 1 | Prop. 4 | Why |
|---|---|---|---|
| Equal-sized groups | ✗ | ✗ | `rank = 2`. Manski's reflection problem |
| Groups, mean **includes** self | ✗ | ✗ | `G² = G` exactly, for *any* sizes |
| Two distinct group sizes | ✓ | ✗ | `G³` is an exact combination of `I, G, G²` |
| Three distinct group sizes | ✓ | ✓ | the paper's "three classroom sizes are now necessary" |
| Star / complete bipartite | ✓ | ✗ | `G³ = G` exactly |
| Line, cycle, random graph | ✓ | ✓ | intransitive triads; diameter ≥ 3 suffices (their Corollary 1) |

The second row is worth dwelling on. Taking the group mean over everyone
*including* the unit itself — the obvious way to compute a classroom
average — makes `G² = G` identically, and no amount of variation in group
size rescues it. Switching to the leave-one-out mean is the difference
between a model that is identified and one that never can be.

```julia
peer_identification(group_weights([5, 5, 5]))                      # not identified
peer_identification(group_weights([4, 5, 6]))                      # identified
peer_identification(group_weights([4, 5, 6]; include_self = true)) # not identified
peer_identification(group_weights([4, 6]); fixed_effects = true)   # not identified
```

### The tolerance is measured

Rank decisions need a tolerance, and this one was set by measurement rather
than convention. Across the degenerate structures above the largest spurious
singular value observed was **1.97e-8**; across identified structures the
smallest genuine one was **4.0e-3**. The default `tol = 1e-6` sits between
them with two orders of magnitude of margin each way.

An earlier default of `1e-8` sat *below* that noise floor and reported two
equal groups of 8 as identified, contradicting Proposition 2 outright. The
`gap` field reports where a given network fell, so a verdict resting on the
tolerance is visible rather than implied.

## Estimation

[`fit_peer_effects`](@ref) runs the check first and **throws** when it
fails, because what it would otherwise return is two coefficients whose
split the data does not determine — printed with ordinary-looking standard
errors and stars. `check = false` overrides it, explicitly.

Estimation is 2SLS with the paper's own instruments: the characteristics of
peers' peers, `G²x`, which enter the reduced form but not the structural
equation. Under fixed effects everything is transformed by `(I - G)` first
and the instrument block is `(I-G)G²x`, exactly as their §4 writes it.

```julia
G   = network_weights(adjacency)
fit = fit_peer_effects(y, X, G)
fit.beta            # endogenous peer effect
fit.first_stage_F   # whether this sample can exploit the identification
```

### Identified is not estimable

The rank condition is a statement about the population. Whether a given
sample can exploit it is separate, and on a dense network the answer is no:
a unit's peers' peers are nearly everybody, so `G²x` flattens towards a
constant.

Measured over 400 draws per row, Erdős–Rényi graphs at `n = 300`, true
`β = 0.4`:

| Density | Mean degree | Median `β̂` | sd(`β̂`) | Median first-stage F | Rejection at 10% |
|---|---|---|---|---|---|
| 0.015 | 4.6 | 0.402 | 0.226 | 36.5 | 0.105 |
| 0.020 | 6.0 | 0.396 | 0.304 | 30.4 | 0.130 |
| 0.030 | 9.0 | 0.421 | 0.504 | 21.8 | 0.117 |
| 0.050 | 15.0 | 0.378 | 1.160 | 13.2 | 0.087 |
| 0.100 | 29.9 | 0.309 | 1.865 | 6.8 | 0.062 |
| 0.200 | 59.8 | 0.147 | 4.695 | 4.4 | 0.062 |

This reproduces Bramoullé, Djebbari and Fortin's own simulation finding
that precision is a decreasing function of density. The spread of `β̂`
rises **twenty-one-fold** across the range and the median biases toward
zero, while the first-stage F collapses from 36.5 to 4.4.

Note the last column. **The rejection rate does not deteriorate — it
improves.** The standard errors grow faster than the estimates do, so the
test becomes conservative exactly as the estimate becomes useless. Nothing
in the significance of the result will tell you this is happening, which is
why `first_stage_F` is reported and warned on.

### Covariance

The default is `:robust`. Bramoullé, Djebbari and Fortin leave the error
variance matrix unrestricted and explicitly do not impose homoskedasticity,
and the measurement supports the default — but only under heteroskedasticity
strong enough to matter. Coverage of nominal 95% intervals for `β`, 2,000
replications, `n = 300`:

| `β` | Errors | `:robust` | `:homoskedastic` |
|---|---|---|---|
| 0.0 | homoskedastic | 0.966 | 0.970 |
| 0.0 | heteroskedastic | 0.970 | **0.942** |
| 0.3 | heteroskedastic | 0.964 | **0.926** |
| 0.6 | heteroskedastic | 0.961 | **0.921** |

Two things to take from this. The gap **widens with `β`**: the spatial
multiplier `(I - βG)⁻¹` compounds the heteroskedasticity across the network
rather than averaging it away. And mild heteroskedasticity — a 1.5× spread
in the error standard deviation — does not separate the two estimators at
all; the table above uses an 11× spread. So the default is the right one
and will often make no difference.

Under fixed effects, `vcov = :transform` accounts for the correlation the
within-local transformation itself induces, since `Var((I-G)ε) =
σ²(I-G)(I-G)'` is not scalar even when `ε` is.

### Over-identification

With `K` characteristics and one endogenous regressor there are `K - 1`
over-identifying restrictions, so the Sargan test exists once `K ≥ 2`.
It is well sized throughout, 2,000 replications:

| Density | 10% | 5% | 1% |
|---|---|---|---|
| 0.015 | 0.098 | 0.044 | 0.013 |
| 0.030 | 0.094 | 0.045 | 0.009 |
| 0.060 | 0.104 | 0.049 | 0.010 |
| 0.120 | 0.099 | 0.049 | 0.009 |

With a single characteristic the model is exactly identified and
`sargan_df` is `0` — reported as such rather than as a p-value of 1.

## Several channels at once

Units are often linked more than one way. [`fit_multiplex_sar`](@ref) fits

```
y = Σⱼ ρⱼ Wⱼ y + Xβ + ε
```

giving each channel its own `ρⱼ`, which a single pooled `W` cannot report.
Estimated by concentrated maximum likelihood, with `β` and `σ²` concentrated
out and the `J`-dimensional profile maximised by coordinate ascent using the
same Brent search the single-channel fit uses.

`log|I - Σρⱼ Wⱼ|` is computed exactly by sparse LU at every evaluation. The
Chebyshev approximation available for a single `W` rests on precomputed
traces of its powers and does not carry over to an arbitrary combination, so
it is not offered here rather than offered and quietly wrong.

With one channel this reproduces `fit_spatial(SAR, …)` — log-likelihood to
1e-13, standard errors to 1e-9 — against independently written, already
audited code, which is what validates the generalised information matrix.

Channels that are collinear **as matrices** cannot have their `ρ`s told
apart, for the same reason `I, G, G²` must be independent above, and the fit
refuses rather than returning an arbitrary split. The parameter space is
`Σⱼ|ρⱼ| < 1`, sufficient for stationarity when each `Wⱼ` is row-standardised;
a fit pinned against that boundary is reported with `converged = false`.

## Isolates

Social networks have units with no links; geographic weight matrices rarely
do. Row standardisation leaves those rows all-zero rather than producing
`NaN`, and `peer_identification` reports the count — because Proposition 1's
*non*-identification half is stated only for networks without isolates. When
the rank is short and isolates are present the verdict is reported as a
failure to establish identification rather than as a proof of its absence,
and `conclusive` is `false`.

## Example

`examples/26_networks.jl` walks through the reflection problem, what
skipping the check costs, the density degradation, and the multiplex fit.

## References

- Bramoullé, Y., Djebbari, H. & Fortin, B. (2009). Identification of peer
  effects through social networks. *Journal of Econometrics*, 150(1),
  41–55.
- Manski, C. F. (1993). Identification of endogenous social effects: the
  reflection problem. *Review of Economic Studies*, 60(3), 531–542.
- Leenders, R. T. A. J. (2002). Modeling social influence through network
  autocorrelation: constructing the weight matrix. *Social Networks*,
  24(1), 21–47.
- Lee, L. F. & Liu, X. (2010). Efficient GMM estimation of high order
  spatial autoregressive models with autoregressive disturbances.
  *Econometric Theory*, 26(1), 187–230.
