# Run the benchmark suite and print it as a table.
#
#   julia --project=benchmark -t auto benchmark/run.jl           # all groups
#   julia --project=benchmark -t auto benchmark/run.jl fit gwr   # some groups
#   SE_BENCH_SIZE=large julia --project=benchmark benchmark/run.jl
#
# Writes benchmark/results/<name>.json. To compare against an earlier run:
#
#   julia --project=benchmark benchmark/run.jl --compare baseline
#
# which reads benchmark/results/baseline.json and prints the ratio column.
# A speedup claim without a before and an after measured the same way is not
# a speedup claim, which is what the compare mode is for.

using BenchmarkTools, Printf, Statistics

const HERE = @__DIR__
const RESULTS = joinpath(HERE, "results")
mkpath(RESULTS)

args = copy(ARGS)

# Pull `--flag value` out of `args` and return the value, or nothing. A plain
# `let` block here would be a trap: at top level in a script, assigning to an
# outer name inside `let` creates a local, so the flag would be consumed from
# `args` and then thrown away -- the groups would narrow and the comparison
# would silently not happen.
function take_flag!(args, flag)
    i = findfirst(==(flag), args)
    isnothing(i) && return nothing
    i == lastindex(args) && error("$flag needs a value")
    val = args[i + 1]
    deleteat!(args, [i, i + 1])
    return val
end

compare_to = take_flag!(args, "--compare")
outname = something(take_flag!(args, "--name"), "latest")

include(joinpath(HERE, "benchmarks.jl"))

groups = isempty(args) ? collect(keys(SUITE)) : args
for g in groups
    haskey(SUITE, g) || error("no benchmark group $g; have $(collect(keys(SUITE)))")
end
suite = BenchmarkGroup()
for g in groups
    suite[g] = SUITE[g]
end

@printf("Julia %s, %d threads, %d BLAS threads\n",
        VERSION, Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads())
println("groups: ", join(groups, ", "))
# No tune!: evals is pinned to 1 in benchmarks.jl, so there is nothing to
# tune and tuning these would cost more than running them.
println("running …")
results = run(suite; verbose = false)

old = nothing
if !isnothing(compare_to)
    path = joinpath(RESULTS, compare_to * ".json")
    isfile(path) || error("no baseline at $path")
    old = BenchmarkTools.load(path)[1]
end

fmt_time(t) = t < 1e3 ? @sprintf("%.0f ns", t) :
              t < 1e6 ? @sprintf("%.1f us", t / 1e3) :
              t < 1e9 ? @sprintf("%.2f ms", t / 1e6) :
                        @sprintf("%.3f s", t / 1e9)

function walk(io, node, old_node, path)
    if node isa BenchmarkTools.BenchmarkGroup
        # Match by the string form of the key: a baseline round-tripped
        # through JSON has String keys whatever it was saved with.
        old_keys = isnothing(old_node) ? Dict{String,Any}() :
                   Dict(string(k) => old_node[k] for k in keys(old_node))
        for k in sort(collect(keys(node)); by = string)
            walk(io, node[k], get(old_keys, string(k), nothing),
                 vcat(path, string(k)))
        end
    else
        t = time(median(node))
        mem = node.memory
        label = join(path, " / ")
        if isnothing(old_node)
            @printf(io, "  %-52s %12s %12s\n", label, fmt_time(t), Base.format_bytes(mem))
        else
            t0 = time(median(old_node))
            @printf(io, "  %-52s %12s %12s %9.2fx\n", label, fmt_time(t),
                    Base.format_bytes(mem), t0 / t)
        end
    end
end

println()
if isnothing(old)
    @printf("  %-52s %12s %12s\n", "benchmark", "median", "alloc")
else
    @printf("  %-52s %12s %12s %9s\n", "benchmark", "median", "alloc", "speedup")
end
println("  ", "─"^(isnothing(old) ? 78 : 88))
walk(stdout, results, old, String[])

path = joinpath(RESULTS, outname * ".json")
BenchmarkTools.save(path, results)
println("\nwrote ", path)
