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


## What the second pass did, and two figures it corrected

Four more findings taken, measured the same way. Two of the numbers in
the first pass above were wrong, and the corrections are the more useful
part of this section.

### `fit_gwr`: 4.0–6.3x

`N` local regressions, each reading the KD-tree and writing only its own
row, so threading them is exact rather than approximate.

| | before | after, `-t 16` | after, `-t 1` |
|---|---|---|---|
| `N = 400` | 27.8 ms | **4.40 ms** (6.3x) | 17.0 ms (1.6x) |
| `N = 900` | 105.4 ms | **26.5 ms** (4.0x) | 52.7 ms (2.0x) |

The single-threaded half came from two things that had nothing to do with
threading, and found only because 3x on sixteen cores was obviously wrong
for an embarrassingly parallel loop. `_get_weights` computed each
candidate distance as `sqrt(sum((coords[i, :] .- coords[j, :]) .^ 2))`,
allocating three vectors per neighbour, `O(N · bw)` allocations per fit;
and the local-R² loop rebuilt two loop-invariant length-`N` vectors on
every one of its `N` iterations. Allocation per fit went from 224 MiB to
115 MiB and scaling from 3.1x to 6.3x. Still short of sixteen: the
remaining `zeros(T, N)` per call is an `O(N²)` allocation, and the local
designs are dense `N x K` where only `bw` rows are nonzero. A compact
local design would be the next step and would change the QR's row order,
so it is not a free one.

Bit-identical at 1, 4 and 16 threads, including the auto-bandwidth path,
which calls the threaded fit repeatedly inside the AICc search.

### `log_marginal_likelihood`: 1.4–1.5x, bit-identical

The 200 exact sparse LUs are 97–99% of the call, which is what the first
pass said. What it got wrong was the remedy: it listed this as "the same
`max_exact_n` mistake", implying the fix was to stop being exact. It is
not. The sparsity pattern of `I - ρW` is the same at every node, so the
symbolic analysis is done once and each node reuses it through `lu!`,
with the values written into the existing pattern instead of a fresh
sparse matrix built per node. Same factorisation, fewer of them:

| | before | after |
|---|---|---|
| 200 nodes, `N = 400` | 160 ms | 116 ms (1.38x) |
| `N = 900` | 342 ms | 232 ms (1.47x) |
| `N = 1600` | 664 ms | 463 ms (1.43x) |

All 200 log-determinants bit-identical, on Julia 1.13.2 and 1.10.12, and
every marginal likelihood and specification probability unchanged to the
last digit across six specifications, three sizes, the Chebyshev branch
and a non-default grid, prior and `ρ` interval.

### Correction: `knn_weights` is not `O(N²)`

The first pass recorded "`knn_weights` is O(N²) — 820 ms at `N = 8100,
k = 12`". It is `O(N log N)`: it has used a KD-tree all along, and the
measured time is **26 ms**, not 820. Scaling confirms it — multiplying
`N` by 2.25 multiplies the time by 2.36–2.41, where quadratic would give
5.06:

| `N` | 900 | 3,600 | 8,100 | 14,400 | 32,400 |
|---|---|---|---|---|---|
| `k = 12` | 2.6 ms | 10.9 ms | 26.2 ms | 63.4 ms | 149 ms |

Where 820 ms came from is the useful part: `knn_weights` calls
`row_standardize!` by default, and that was the accidental quadratic of
finding 29. The figure was mostly a measurement of a different function's
defect, attributed to this one. A profile of a composite is not a profile
of its parts.

There was still something to fix, worth 1.2–1.5x: the neighbour selection
re-derived, for every candidate, how many non-self neighbours the list
held — an allocating `filter` plus a `count` over a fresh slice, inside
the inner loop — to decide what a running counter decides in `O(1)`.
Allocation at `N = 8100, k = 12` fell from 40.5 MiB to 13.2 MiB, and the
output is identical across 144 configurations of size, `k`, `binary`,
`row_standardize`, `symmetrize`, dimension and coincident points.

### Correction: the Chebyshev 461 MiB was allocation, not a ceiling

The first pass reported `precompute_chebyshev_traces` reaching "461 MiB
at `N = 8100, q = 12`" and read it as a memory ceiling. That number is
BenchmarkTools' **cumulative** allocation over the whole call — the sum
of twelve intermediate matrices, nearly all collectable. The peak, which
is what a ceiling is about, is `18.9 MiB`:

| | peak fill | % of `N²` | cumulative |
|---|---|---|---|
| rook, `N = 8100`, `q = 12` | **18.9 MiB** | 1.9% | 90.8 MiB |
| knn `k = 24`, `N = 900` | 12.1 MiB | 97.8% | 77.1 MiB |
| knn `k = 24`, `N = 3600` | 89.9 MiB | 45.5% | 457 MiB |

The 97.8%-of-`N²` density figure was right, but at `N = 900` that is
12 MiB, so it was not evidence of a ceiling either. The ceiling is real
and arrives later: the fill per row grows until it saturates at `N`, so
peak memory is `Θ(N²)` on a `W` dense enough, and knn `k = 24` is already
at 90 MiB by `N = 3600`.

So the fix is a ceiling rather than a replacement. `max_fill_bytes`
defaults to 128 MiB; when the recursion is about to cross it, the
remaining orders are finished one column block at a time, in `3 · N · b`
doubles, seeded from the two sparse matrices already computed so nothing
is discarded. Measured on knn `k = 24, N = 3600, q = 12`: 1118 ms and
2.5 GiB allocated becomes 576 ms and 1.1 GiB — **faster**, because that
is the saturated regime where the sparse products have become dense ones
with index overhead.

Blocking is emphatically not a general win, which is why it is a fallback
and not the default. On a rook lattice it is 4x to 25x **slower**, since
the sparse recursion only touches the fill that exists while the blocked
path pays `N · b` per level regardless. At the 128 MiB default every case
in this suite stays on the fast path and is bit-identical, order 6 — what
`fit_spatial` uses — included.


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
