# root_rho on real data at n = 2,882: the mainland Portuguese parishes (notebooks/data, built by
# PortugalINE.jl's scripts/build_parish_dataset.jl on the 2013 parish map, CAOP2024).
#
# The question carried over from tc_portugal.jl (the 278 municipalities): there, the spatial lag of a
# SAR without a surface (ρ̂ ≈ 0.4) fell to about 0–0.2 once a smooth surface f(u, v) was added, and the
# test-inversion set then included 0 in most fits. At n = 278 the surface could have only a few degrees
# of freedom. With ten times the units, a much richer surface is affordable (K/n stays small), so this
# asks whether the lag survives a flexible trend, and whether root_rho's guards and costs behave at
# n ≈ 2,900 (its dense path holds four n × n matrices, about 270 MB here).
#
# Model: y = ρWy + Xβ + f(u, v) + ε, with y = log gross declared income per taxpayer (2021, the Census
# year), and X = log population density, unemployment rate, illiteracy rate (Census 2021).
# W: queen contiguity (GAL) and kNN-6 on the centroids. Baselines: ML SAR without a surface (with an
# intercept), and root_rho without a surface. Surfaces: the projection on tensor_pspline(u, v; k) with
# the empty columns dropped, k = 6, 8, 10, 12; the penalised smoother on the untrimmed basis (allowed
# since AUDIT 37), k = 16 and 24, λ = 0.1, 1, 10, 100.
#
# Data credit (a condition of the licences): Source: Statistics Portugal (INE), indicators 0011866,
# 0012273, 0012295, 0012328, 0012748, reference period 2021, extracted 2026-11-25. Informação
# geográfica cedida pela Direção-Geral do Território (CAOP, CC-BY-4.0).
#
# Run: julia --project=. research/trace_cfe/tc_portugal_parishes.jl > research/trace_cfe/tc_portugal_parishes_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf, Logging, Test

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")

# A CSV line, with quoted fields (parish names such as "União das freguesias de Milhazes, Vilar de
# Figos e Faria" carry commas).
function splitcsv(l)
    out = String[]; buf = IOBuffer(); q = false
    for c in l
        if c == '"'
            q = !q
        elseif c == ',' && !q
            push!(out, String(take!(buf)))
        else
            write(buf, c)
        end
    end
    push!(out, String(take!(buf)))
    out
end
function readcsv(f)
    ls = [l for l in readlines(joinpath(DATA, f)) if !isempty(strip(l))]
    h = splitcsv(ls[1]); rs = splitcsv.(ls[2:end])
    Dict(h[j] => [r[j] for r in rs] for j in eachindex(h))
end

d = readcsv("portugal_parishes.csv")
c = readcsv("portugal_parishes_centroids.csv")
c["dicofre"] == d["dicofre"] || error("centroids are not in the CSV's order")
num(k) = parse.(Float64, d[k])
place = d["name"]; n = length(place)
y = log.(num("income_2021"))
X = hcat(log.(num("population") ./ num("area_km2")), num("unemployment"), num("illiteracy"))
vn = ["log_density", "unemployment", "illiteracy"]
u = parse.(Float64, c["x_m"]) ./ 1000; v = parse.(Float64, c["y_m"]) ./ 1000      # km, EPSG:3763
u .-= mean(u); v .-= mean(v)

Wq, qid = read_gal(joinpath(DATA, "portugal_parishes_queen.gal"))
qid == d["dicofre"] || error("GAL order differs from the CSV")
Wk = knn_weights(hcat(u, v), 6)
WS = (("queen", Wq), ("kNN-6", Wk))

@printf("n = %d mainland parishes; extent %.0f km (E–W) × %.0f km (N–S)\n", n,
        maximum(u) - minimum(u), maximum(v) - minimum(v))
@printf("y = log income 2021: median %.3f, range %.3f–%.3f (max: %s); skewness %.2f\n", median(y),
        extrema(y)..., place[argmax(y)], mean(((y .- mean(y)) ./ std(y)) .^ 3))
for (nm, W) in WS
    nn = vec(sum(W .> 0; dims = 2))
    @printf("  W %-6s: links %d, neighbours per unit %d–%d (mean %.2f)\n", nm, nnz(W), extrema(nn)..., mean(nn))
end
println(); flush(stdout)

function trimmed_basis(k)
    B = tensor_pspline(u, v; k).basis
    used = vec(maximum(B; dims = 1)) .> 0
    (B[:, used], count(!, used))
end

function fit(yv, Xm, W; kw...)
    tl = TestLogger(min_level = Logging.Warn)
    r = nothing; err = nothing; t = 0.0
    with_logger(tl) do
        t = @elapsed try
            r = root_rho(yv, Xm, W; kw...)
        catch e
            err = sprint(showerror, e)
        end
    end
    (r, err, [string(l.message) for l in tl.logs], t)
end

short(s; m = 160) = (s = replace(s, '\n' => ' '); length(s) > m ? first(s, m) * "…" : s)
fmtcs(cs) = isempty(cs) ? "empty" : join([@sprintf("[%.3f, %.3f]", a, b) for (a, b) in cs], " ∪ ")

function report(label, yv, Xm, W; kw...)
    r, err, warns, t = fit(yv, Xm, W; kw...)
    if r === nothing
        @printf("  %-24s ERROR (%.0fs): %s\n", label, t, short(err))
        flush(stdout)
        return nothing
    end
    @printf("  %-24s K %4d df %6.1f hmax %.3f | ρ̂ %+.3f se %.3f Wald [%+.3f, %+.3f] | CS %s%s | wb %+.4f | β̂ %s | %.0fs\n",
            label, r.K, r.smoother === :penalised ? r.df : r.K, r.max_leverage, r.rho, r.se,
            r.ci_wald..., fmtcs(r.cs), r.cs_touches_bound ? " (reaches a bound)" : "",
            r.vhat_worst_bias, join([@sprintf("%+.4g", b) for b in r.beta], ", "), t)
    for w in unique(warns)
        println("      warning: ", short(w))
    end
    flush(stdout)
    r
end

for (wname, W) in WS
    println("W = $wname")
    m = fit_spatial(SAR(), W, X, y; method = ExactLU())   # adds its own intercept
    @printf("  %-24s ρ̂ %+.3f (se %.3f), β̂ %s\n", "ML SAR, no surface", m.rho, m.rho_se,
            join([@sprintf("%+.4g", b) for b in m.coef[2:end]], ", "))
    report("root_rho, no surface", y, hcat(ones(n), X), W)
    for k in (6, 8, 10, 12)
        B, dropped = trimmed_basis(k)
        report(@sprintf("projection k=%d (-%d)", k, dropped), y, X, W; basis = B)
    end
    for k in (16, 24)
        ps = tensor_pspline(u, v; k)
        for λ in (0.1, 1.0, 10.0, 100.0)
            report(@sprintf("penalised k=%d λ=%g", k, λ), y, X, W; ps..., lambda = λ)
        end
    end
    println(); flush(stdout)
end

# --- residuals at one flexible fit, to judge which worst case applies ------------------------------
println("Residual diagnostics: queen, penalised k = 24, λ = 1")
let ps = tensor_pspline(u, v; k = 24), r = root_rho(y, X, Wq; ps..., lambda = 1.0)
    Z = hcat(X, ps.basis); K = size(Z, 2)
    Ωt = zeros(K, K); Ωt[4:end, 4:end] = ps.penalty
    y0 = y .- r.rho .* (Wq * y)
    e = y0 .- Z * ((Z' * Z + Ωt) \ (Z' * y0))
    z = (e .- mean(e)) ./ std(e)
    @printf("  residual skewness %.2f, excess kurtosis %.2f\n", mean(z .^ 3), mean(z .^ 4) - 3)
    bd = hcat(ones(n), X[:, 1]) \ (e .^ 2)
    @printf("  e² on log density: slope %.5f (mean e² %.5f)\n", bd[2], mean(e .^ 2))
    println("  Moran's I of e: ", moran_global(e, Wq))
    println("  Moran's I of e²: ", moran_global(e .^ 2, Wq))
    ord = sortperm(abs.(z); rev = true)
    println("  largest |standardised residuals|: ",
            join([@sprintf("%s %+.1f", place[i], z[i]) for i in ord[1:6]], ", "))
end
