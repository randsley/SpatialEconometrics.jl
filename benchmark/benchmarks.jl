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

# Every leaf key is a String. BenchmarkTools serialises a group to JSON, and
# JSON has no integer keys, so an Int key comes back as a String and a saved
# baseline stops matching the run it is compared against -- silently, as a
# missing speedup column rather than an error.
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
    SUITE["logdet"]["exact_lu"]["N=$(c.N)"] =
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
    SUITE["traces"]["spatial_trace_powers"]["N=$(c.N)"] =
        @benchmarkable SpatialEconometrics.SpatialUtils.spatial_trace_powers(
            $(c.W); max_power = 60, max_exact_n = 1000, n_probes = 250,
            rng = MersenneTwister(2))
    # Both branches at the sizes where both are feasible, so the switch at
    # `max_exact_n` can be judged rather than assumed. The exact branch is a
    # DENSE non-symmetric eigendecomposition -- O(N^3) time and N^2 memory,
    # 1.7 GB at N = 14400 -- so it is only benchmarked where that is sane.
    if c.N <= 2500
        SUITE["traces"]["forced_exact"]["N=$(c.N)"] =
            @benchmarkable SpatialEconometrics.SpatialUtils.spatial_trace_powers(
                $(c.W); max_power = 60, max_exact_n = typemax(Int))
    end
    SUITE["traces"]["forced_probes"]["N=$(c.N)"] =
        @benchmarkable SpatialEconometrics.SpatialUtils.spatial_trace_powers(
            $(c.W); max_power = 60, max_exact_n = 0, n_probes = 250,
            rng = MersenneTwister(2))
end

# ─── estimators ───────────────────────────────────────────────────────
SUITE["fit"] = BenchmarkGroup()
for side in SIDES
    c = lattice_case(side)
    SUITE["fit"]["sar"]["N=$(c.N)"] = @benchmarkable fit_spatial(SAR(), $(c.W), $(c.X), $(c.y))
    SUITE["fit"]["sdm"]["N=$(c.N)"] = @benchmarkable fit_spatial(SDM(), $(c.W), $(c.X), $(c.y))
    SUITE["fit"]["sem"]["N=$(c.N)"] = @benchmarkable fit_spatial_error($(c.W), $(c.X), $(c.y))
    SUITE["fit"]["slx"]["N=$(c.N)"] = @benchmarkable fit_slx($(c.W), $(c.X), $(c.y))
end

# ─── GWR ──────────────────────────────────────────────────────────────
# N local regressions, independent of each other. Several bandwidths, not
# one: the cost of a local fit is set by the size of the kernel's support
# `b`, and `b / N` is what any compact-design work has to beat. On a square
# grid that ratio spans 0.02 to 1.0 across the cases below, so a single
# bandwidth would tell you almost nothing about the shape of the curve.
SUITE["gwr"] = BenchmarkGroup()
for side in (LARGE ? (20, 30, 40) : (20, 30))
    c = knn_case(side, 4)
    rng = MersenneTwister(9)
    X = randn(rng, c.N, 2)
    y = 1.0 .+ X * [1.0, -0.5] .+ 0.3 .* randn(rng, c.N)
    SUITE["gwr"]["fixed_bw"]["N=$(c.N)"] =
        @benchmarkable fit_gwr($X, $y, $(c.coords);
                               bandwidth = FixedBandwidth(8.0), kernel = GaussianKernel())
    # b/N about 0.05 at N = 900: the regime where a compact design should
    # pay, and the one the suite previously had no case for.
    SUITE["gwr"]["fixed_narrow"]["N=$(c.N)"] =
        @benchmarkable fit_gwr($X, $y, $(c.coords);
                               bandwidth = FixedBandwidth(4.0), kernel = GaussianKernel())
    SUITE["gwr"]["adaptive_k20"]["N=$(c.N)"] =
        @benchmarkable fit_gwr($X, $y, $(c.coords);
                               bandwidth = AdaptiveBandwidth(20), kernel = BisquareKernel())
    # b/N about 0.56 at N = 900: near the other end, and near where the
    # AICc search tends to land, so it bounds the gain from below.
    SUITE["gwr"]["adaptive_k500"]["N=$(c.N)"] =
        @benchmarkable fit_gwr($X, $y, $(c.coords);
                               bandwidth = AdaptiveBandwidth(500), kernel = BisquareKernel())
    # MGWR back-fits K bandwidths and has its own N x K `se_local` loop.
    SUITE["gwr"]["mgwr"]["N=$(c.N)"] =
        @benchmarkable fit_mgwr($X, $y, $(c.coords); inference = false, max_iter = 12)
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
    SUITE["weights"]["row_standardize"]["N=$(c.N)"] = @benchmarkable row_standardize($(c.W))
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
    SUITE["selection"]["bayes_spec_probs"]["N=$(c.N)"] =
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
