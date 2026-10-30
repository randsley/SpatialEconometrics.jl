# Benchmark suite for SpatialEconometrics.jl.
#
# Defines `SUITE`, a `BenchmarkGroup`, following the convention PkgBenchmark
# expects, so it can be driven either by `include`ing this file or by
# `PkgBenchmark.benchmarkpkg`. `benchmark/run.jl` is the convenience runner.
#
# The sizes are chosen to bracket the point where each method's cost changes
# character rather than to be impressive: `max_exact_n = 1000` switches the
# trace powers from a dense eigendecomposition to Hutchinson probes, so the
# grids straddle it. Set SE_BENCH_SIZE=large for the bigger grid.

using BenchmarkTools
using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics

const LARGE = get(ENV, "SE_BENCH_SIZE", "default") == "large"
const SIDES = LARGE ? (30, 60, 90, 120, 160) : (30, 60, 90)
const KNN_K = LARGE ? (4, 12, 24) : (4, 12)

# Every case is built from a fixed seed: a benchmark that changes its input
# between runs is comparing two different problems.
function lattice_case(side::Int)
    W = row_standardize(make_rook_weights(side))
    N = size(W, 1)
    rng = MersenneTwister(1234)
    X = randn(rng, N, 2)
    y, _ = simulate_spatial(SAR_DGP(0.5, [1.0, -0.5], 0.5), W, X; rng = rng)
    return (; W, X, y, N)
end

function knn_case(side::Int, k::Int)
    coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0
    W = row_standardize(knn_weights(coords, k))
    return (; W, coords, N = size(W, 1))
end

const SUITE = BenchmarkGroup()

# These benchmarks run in milliseconds to seconds, so the default of hunting
# for an evals/sample count and collecting 10,000 samples is both pointless
# and extremely slow. One eval per sample, a few samples, a hard time budget.
BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.samples = 7
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10.0
BenchmarkTools.DEFAULT_PARAMETERS.gctrial = true

# ─── log-determinant ──────────────────────────────────────────────────
# Two costs that must not be conflated: the one-off precompute and the
# per-evaluation cost. A method with an expensive precompute and a cheap
# evaluation wins whenever the caller sweeps rho, and loses on a single
# evaluation. The estimators sweep.
SUITE["logdet"] = BenchmarkGroup()
for side in SIDES
    c = lattice_case(side)
    Isp = sparse(1.0I, c.N, c.N)
    SUITE["logdet"]["exact_lu"][c.N] =
        @benchmarkable first(logabsdet(lu($Isp - 0.5 * $(c.W))))
    for ord in (6, 12)
        SUITE["logdet"]["precompute_exact_traces"]["N=$(c.N),q=$ord"] =
            @benchmarkable SpatialEconometrics.SpatialUtils.precompute_chebyshev_traces(
                $(c.W), ChebyshevApprox(order = $ord, n_probes = 0))
        SUITE["logdet"]["precompute_probes"]["N=$(c.N),q=$ord"] =
            @benchmarkable SpatialEconometrics.SpatialUtils.precompute_chebyshev_traces(
                $(c.W), ChebyshevApprox(order = $ord, n_probes = 16,
                                        rng = MersenneTwister(7)))
        tau = SpatialEconometrics.SpatialUtils.precompute_chebyshev_traces(
            c.W, ChebyshevApprox(order = ord, n_probes = 0))
        SUITE["logdet"]["evaluate"]["N=$(c.N),q=$ord"] =
            @benchmarkable SpatialEconometrics.SpatialUtils.chebyshev_logdet(0.37, $tau)
    end
end

# ─── trace powers ─────────────────────────────────────────────────────
# The dominant cost of `fit_spatial`, which is not obvious from reading it:
# `spatial_trace_powers` is a dense eigendecomposition at or below
# `max_exact_n` and 250 x max_power sparse matvecs above it.
SUITE["traces"] = BenchmarkGroup()
for side in SIDES
    c = lattice_case(side)
    SUITE["traces"]["spatial_trace_powers"][c.N] =
        @benchmarkable SpatialEconometrics.SpatialUtils.spatial_trace_powers(
            $(c.W); max_power = 60, max_exact_n = 1000, n_probes = 250,
            rng = MersenneTwister(2))
    # Both branches at the sizes where both are feasible, so the switch at
    # `max_exact_n` can be judged rather than assumed. The exact branch is a
    # DENSE non-symmetric eigendecomposition -- O(N^3) time and N^2 memory,
    # 1.7 GB at N = 14400 -- so it is only benchmarked where that is sane.
    if c.N <= 2500
        SUITE["traces"]["forced_exact"][c.N] =
            @benchmarkable SpatialEconometrics.SpatialUtils.spatial_trace_powers(
                $(c.W); max_power = 60, max_exact_n = typemax(Int))
    end
    SUITE["traces"]["forced_probes"][c.N] =
        @benchmarkable SpatialEconometrics.SpatialUtils.spatial_trace_powers(
            $(c.W); max_power = 60, max_exact_n = 0, n_probes = 250,
            rng = MersenneTwister(2))
end

# ─── estimators ───────────────────────────────────────────────────────
SUITE["fit"] = BenchmarkGroup()
for side in SIDES
    c = lattice_case(side)
    SUITE["fit"]["sar"][c.N] = @benchmarkable fit_spatial(SAR(), $(c.W), $(c.X), $(c.y))
    SUITE["fit"]["sdm"][c.N] = @benchmarkable fit_spatial(SDM(), $(c.W), $(c.X), $(c.y))
    SUITE["fit"]["sem"][c.N] = @benchmarkable fit_spatial_error($(c.W), $(c.X), $(c.y))
    SUITE["fit"]["slx"][c.N] = @benchmarkable fit_slx($(c.W), $(c.X), $(c.y))
end

# ─── GWR ──────────────────────────────────────────────────────────────
# N local regressions, independent of each other and currently sequential.
SUITE["gwr"] = BenchmarkGroup()
for side in (LARGE ? (20, 30, 40) : (20, 30))
    c = knn_case(side, 4)
    rng = MersenneTwister(9)
    X = randn(rng, c.N, 2)
    y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* randn(rng, c.N)
    SUITE["gwr"]["fixed_bw"][c.N] =
        @benchmarkable fit_gwr($X, $y, $(c.coords);
                               bandwidth = FixedBandwidth(8.0), kernel = GaussianKernel())
end

# ─── weights ──────────────────────────────────────────────────────────
SUITE["weights"] = BenchmarkGroup()
for side in SIDES, k in KNN_K
    c = knn_case(side, k)
    SUITE["weights"]["knn"]["N=$(c.N),k=$k"] =
        @benchmarkable knn_weights($(c.coords), $k)
end
for side in SIDES
    c = lattice_case(side)
    SUITE["weights"]["row_standardize"][c.N] = @benchmarkable row_standardize($(c.W))
end

# ─── specification comparison ─────────────────────────────────────────
# Quadrature over rho: the case where the log-determinant precompute is
# amortised over 200 evaluations rather than paid once.
SUITE["selection"] = BenchmarkGroup()
for side in (LARGE ? (20, 30, 40) : (20, 30))
    c = lattice_case(side)
    for spec in (:sar, :sem, :sdm)
        SUITE["selection"]["log_marginal"]["N=$(c.N),$spec"] =
            @benchmarkable log_marginal_likelihood($spec, $(c.y), $(c.X), $(c.W))
    end
    SUITE["selection"]["bayes_spec_probs"][c.N] =
        @benchmarkable bayes_spec_probs($(c.y), $(c.X), [$(c.W)])
end

# ─── spatio-temporal PCA ──────────────────────────────────────────────
SUITE["stpca"] = BenchmarkGroup()
for side in (LARGE ? (10, 16, 24) : (10, 16)), p in (3, 8)
    W = row_standardize(make_rook_weights(side))
    N = size(W, 1)
    rng = MersenneTwister(5)
    Xa = randn(rng, N, p, 17)
    ts = collect(0.5:1.0:16.5)
    SUITE["stpca"]["moran"]["N=$N,p=$p"] =
        @benchmarkable stpca($Xa, $W; measure = :moran, times = $ts,
                             interval = (0.0, 17.0))
end

SUITE
