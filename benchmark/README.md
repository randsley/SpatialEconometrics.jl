# Benchmarks

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark -t auto benchmark/run.jl                 # every group
julia --project=benchmark -t auto benchmark/run.jl fit gwr         # some groups
SE_BENCH_SIZE=large julia --project=benchmark -t auto benchmark/run.jl
```

Results land in `benchmark/results/<name>.json` (`--name` to choose, `latest`
by default). To measure a change:

```bash
julia --project=benchmark -t auto benchmark/run.jl fit --name before
# … edit …
julia --project=benchmark -t auto benchmark/run.jl fit --compare before
```

which adds a speedup column. A speedup claim without a before and an after
measured the same way is not a claim, which is what `--compare` is for.

Groups: `logdet`, `traces`, `fit`, `gwr`, `weights`, `selection`, `stpca`.

## What the first run found

Measured 2026-10-30 on Julia 1.13.2, 16 threads, 8 BLAS threads, rook
lattice with a row-standardised `W`. These numbers are the reason roadmap
item #10 was re-planned; `docs/AUDIT.md` has the full note.

**`fit_spatial` is dominated by the trace powers, not by the
log-determinant.** Stage by stage, in seconds:

| `N` | total | Chebyshev traces | Brent | `spatial_trace_powers` | vcov |
|---|---|---|---|---|---|
| 900 | 0.269 | 0.001 | 0.000 | **0.246 (92%)** | 0.020 |
| 3600 | 0.302 | 0.006 | 0.000 | **0.225 (74%)** | 0.070 |
| 8100 | 0.746 | 0.015 | 0.000 | **0.540 (72%)** | 0.170 |
| 14400 | 1.357 | 0.026 | 0.001 | **0.958 (71%)** | 0.344 |

The whole log-determinant path — the precompute plus every evaluation — is
under 2% of a fit. `brent_maximize` calls the concentrated likelihood **six
times**, so the per-evaluation cost of `chebyshev_logdet` (1.1 µs at order 6)
is irrelevant. Optimising the log-determinant cannot make `fit_spatial`
more than about 2% faster, whatever is done to it.

**The `max_exact_n = 1000` switch runs the wrong way round at its own
boundary.** At `N = 900` the exact branch — a dense non-symmetric
eigendecomposition — takes 256 ms against 59 ms for 250 Hutchinson probes:

| | `N = 900` | `N = 3600` | `N = 8100` |
|---|---|---|---|
| forced exact (dense `eigvals`) | 256 ms | not feasible | not feasible |
| forced probes | 59 ms | 235 ms | 562 ms |

The exact branch is `O(N³)` in time and `O(N²)` in memory — 1.7 GB at
`N = 14400` — so it is not merely slower, it has a ceiling. It is chosen by
default for every `N ≤ 1000`. Exactness is the justification and it is a
real one (unseeded probes made reported impacts vary run to run), but the
cost of it is 4.3× at the boundary and unbounded above it.

**The deterministic Chebyshev recursion densifies, as the roadmap said.**
Time and allocation of `precompute_chebyshev_traces`:

| | order 6 | order 12 |
|---|---|---|
| `n_probes = 0`, `N = 8100` | 23 ms / 67 MiB | 137 ms / **461 MiB** |
| `n_probes = 16`, `N = 8100` | 4.5 ms / 4.0 MiB | 8.1 ms / 4.0 MiB |

On a lattice the fill-in is survivable. On a denser `W` it is not: for knn
weights the `k`-th Chebyshev matrix reaches **97.8% of `N²` nonzeros** at
`N = 900, k = 24, order 12`, and 69.3% at `k = 12`. `fit_spatial` defaults
to `ChebyshevApprox(order = 6, n_probes = 0)`, the densifying branch.

## What the first change did

`precompute_neumann_traces` was rewritten to hold the probe block with the
probes contiguous, gather instead of scatter, and take the trace estimate
in the same pass. Measured with `--compare`, 16 threads:

| | before | after | speedup |
|---|---|---|---|
| `spatial_trace_powers`, probes, `N = 900` | 73.2 ms | 6.0 ms | **12.2x** |
| probes, `N = 3600` | 248 ms | 41.7 ms | **5.9x** |
| probes, `N = 8100` | 684 ms | 204 ms | **3.4x** |
| `fit_spatial` SAR, `N = 3600` | 321 ms | 174 ms | 1.84x |
| `fit_spatial` SAR, `N = 8100` | 806 ms | 444 ms | 1.82x |
| `fit_spatial` SEM, `N = 3600` | 329 ms | 121 ms | 2.72x |
| `fit_spatial` SDM, `N = 8100` | 799 ms | 421 ms | 1.90x |

Three details worth keeping straight about these numbers.

**The win is the memory layout, not the threads.** At `-t 1` the same
rewrite is 2.4x to 5.7x, and at `N = 14400` sixteen threads buy 4% over
one (0.404 s against 0.389 s). The loop is bandwidth-bound once the probe
block stops fitting in cache, so the threading helps at small `N` and
barely at large. It is kept because it is free and deterministic, not
because it is where the speedup came from.

**`N = 900` did not move** (1.01x) and that is the point: at or below
`max_exact_n = 1000` the traces come from a dense eigendecomposition
instead, which the rewrite does not touch. The gap between the two
branches at their own boundary has therefore gone from about 4.6x to
**51x** — 0.3 s exact against 6.0 ms for 250 probes. The policy question
is unchanged but the price of it is now an order of magnitude larger.

**Allocation went up, by one transient block**: 46.4 to 62.8 MiB at
`N = 8100`. The probes are drawn into an `N x p` matrix and transposed,
because `rand!` fills column-major and drawing straight into `p x N`
would hand a caller who passed a seed a different probe set. Drawing `p`
length-`N` vectors instead reproduces the column-major stream exactly on
both Julia 1.10 and 1.13 and would save the block, but it relies on an
undocumented detail of how `rand!` consumes the stream, so it was not
taken.

The trace estimates themselves shift by 7e-14 at worst. The probe set is
identical; what changed is that the outer sum runs over nodes rather than
probes. Against a Hutchinson standard error of order `1/sqrt(250)`, or 6%,
that is not a change in the answer. It is bit-identical across 1, 2, 7 and
16 threads.


### What this implies for the item

Three of the seven listed deliverables — sparse Chebyshev, GPU spatial lag,
matrix-free log-determinant — target the part of `fit_spatial` that is under
2% of its runtime. They are not wrong, but they cannot pay off on this
profile, and the GPU and out-of-core ones carry optional dependencies. The
work the measurements justify is, in order:

1. ~~the `spatial_trace_powers` path, which is 71–92% of every fit~~ —
   done, see above;
2. threading, since the probe loop and the GWR local regressions are both
   embarrassingly parallel and the package currently contains no `Threads`
   call at all;
3. the `max_exact_n` policy, which picks the slower branch at its boundary.
