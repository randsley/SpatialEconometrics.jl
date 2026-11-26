# root_rho on real data at n = 3,027: the Dutch districts (wijken) of 2021, with income
# (notebooks/data, built by notebooks/data/build/build_netherlands_2021.py).
#
# The question (THEORY.md, "A third real-data run"): on the 2023 districts, with y = log house value, ρ̂
# fell from 0.74 to 0.27 as the surface grew more flexible and never levelled off, as it also drifted
# on the Portuguese parishes. Is that the method or house prices? Two outcomes on the same 3,027
# districts and the same W, both fixed before any fit:
#
#   income: y = log mean income per inhabitant (2021, × 1,000 euro); X = log population density,
#           % owner-occupied dwellings, % with a non-Western migration background, % aged 65 or over,
#           % under 15 (the income is averaged over all inhabitants, children included).
#   house:  y = log mean WOZ value (2021, value date 1 January 2020); X = log density, % owner-occupied,
#           % non-Western: the 2023 design, with the 2021 migration category.
#
# W: queen contiguity. Fits as in tc_netherlands_districts.jl: ML SAR and root_rho without a surface;
# the penalised smoother on tensor_pspline(u, v; k), k = 16 and 24, λ = 0.1, 1, 10, 100; the bias-aware
# interval (the smoother as the pilot) with ℛ = m·ℛ̂, ℛ̂ the roughness at λ = 1, m = 1, 2, 4.
#
# Data credit (a condition of the licence): Bron: Centraal Bureau voor de Statistiek (CBS), Kerncijfers
# wijken en buurten 2021 (StatLine 85039NED) and Wijk- en Buurtkaart 2021 versie 3, CC BY 4.0.
#
# Run: julia --project=. research/trace_cfe/tc_netherlands_income.jl > research/trace_cfe/tc_netherlands_income_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")
splitq(l) = [strip(s, '"') for s in split(l, r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")]
function readcsv(f)
    ls = filter(!isempty, strip.(readlines(joinpath(DATA, f))))
    h = splitq(ls[1]); rs = splitq.(ls[2:end])
    Dict(h[j] => [r[j] for r in rs] for j in eachindex(h))
end

d = readcsv("netherlands_districts_2021.csv"); c = readcsv("netherlands_districts_2021_centroids.csv")
c["wijkcode"] == d["wijkcode"] || error("centroids are not in the CSV's order")
num(k) = parse.(Float64, d[k])
place = d["name"] .* " (" .* d["municipality"] .* ")"; n = length(place)
u = parse.(Float64, c["x_m"]) ./ 1000; v = parse.(Float64, c["y_m"]) ./ 1000      # km, EPSG:28992
W, wid = read_gal(joinpath(DATA, "netherlands_districts_2021_queen.gal"))
wid == d["wijkcode"] || error("GAL order differs from the CSV")

@printf("n = %d districts; extent %.0f km (E–W) × %.0f km (N–S)\n", n, maximum(u) - minimum(u),
        maximum(v) - minimum(v))
nn = vec(sum(W .> 0; dims = 2))
@printf("W queen: links %d, neighbours per unit %d–%d (mean %.2f)\n\n", nnz(W) ÷ 2, extrema(nn)..., mean(nn))

showcs(cs) = isempty(cs) ? "empty" : join([@sprintf("[%.3f, %.3f]", a, b) for (a, b) in cs], " ∪ ")

function design(title, y, X)
    p = size(X, 2)
    println("="^100); println(title)
    @printf("y: median %.3f, range %.3f–%.3f (min: %s; max: %s)\n\n", median(y), extrema(y)...,
            place[argmin(y)], place[argmax(y)])
    m = fit_spatial(SAR(), W, X, y; method = ExactLU())          # adds its own intercept
    @printf("ML SAR, no surface: ρ̂ %.3f (se %.3f), β̂ %s\n", m.rho, m.rho_se,
            join([@sprintf("%+.4g", b) for b in m.coef[2:p+1]], ", "))
    r0 = root_rho(y, hcat(ones(n), X), W)
    @printf("root_rho, no surface: ρ̂ %.3f, CS %s\n\n", r0.rho, showcs(r0.cs))

    for k in (16, 24)
        ps = tensor_pspline(u, v; k)
        Z = hcat(X, ps.basis); K = size(Z, 2)
        @printf("k = %d: K = %d, rank [X, basis] = %d, %d basis functions with no unit\n", k, K, rank(Z),
                count(iszero, vec(maximum(ps.basis; dims = 1))))
        for λ in (100.0, 10.0, 1.0, 0.1)
            t = @elapsed r = root_rho(y, X, W; ps..., lambda = λ)
            @printf("  λ = %-5g df %6.1f  max Sᵢᵢ %.3f  ρ̂ %.3f  CS %s  (%.0fs)\n", λ, r.df, r.max_leverage,
                    r.rho, showcs(r.cs), t)
        end
        Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty
        r1 = root_rho(y, X, W; ps..., lambda = 1.0)
        ψ = (Z' * Z + Ωt) \ (Z' * (y .- r1.rho .* (W * y)))
        ℛ̂ = sqrt(dot(ψ, Ωt * ψ))
        @printf("  bias-aware, ℛ̂ = %.3f (the surface at λ = 1):\n", ℛ̂)
        for mult in (1, 2, 4)
            t = @elapsed rb = root_rho(y, X, W; ps..., lambda = [0.1, 1.0, 10.0, 100.0], roughness = mult * ℛ̂)
            @printf("    %d·ℛ̂: λ %g, ρ̂ %.3f, interval [%.3f, %.3f], b̄/√V̂ %.2f, pilot %s (left out %d); half-lengths %s (%.0fs)\n",
                    mult, rb.lambda, rb.rho, rb.ci_bias_aware..., rb.bias_bound / sqrt(rb.vhat), rb.pilot,
                    rb.pilot_left_out, join([@sprintf("%.3f", h) for h in rb.halflength_grid], " "), t)
        end
        flush(stdout)
    end

    println("\nResidual diagnostics: penalised k = 24, λ = 1")
    let ps = tensor_pspline(u, v; k = 24), r = root_rho(y, X, W; ps..., lambda = 1.0)
        Z = hcat(X, ps.basis); K = size(Z, 2)
        Ωt = zeros(K, K); Ωt[p+1:end, p+1:end] = ps.penalty
        y0 = y .- r.rho .* (W * y)
        e = y0 .- Z * ((Z' * Z + Ωt) \ (Z' * y0))
        z = (e .- mean(e)) ./ std(e)
        @printf("  residual skewness %.2f, excess kurtosis %.2f\n", mean(z .^ 3), mean(z .^ 4) - 3)
        println("  Moran's I of e²: ", moran_global(e .^ 2, W))
        ord = sortperm(abs.(z); rev = true)
        println("  largest |standardised residuals|: ",
                join([@sprintf("%s %+.1f", place[i], z[i]) for i in ord[1:6]], ", "))
    end
end

design("INCOME: y = log mean income per inhabitant 2021; X = log density, % owner-occupied, % non-Western, % 65+, % under 15",
       log.(num("income")),
       hcat(log.(num("density")), num("owner_occupied"), num("non_western"), num("aged_65_plus"), num("under_15")))
design("HOUSE: y = log mean WOZ value 2021; X = log density, % owner-occupied, % non-Western",
       log.(num("house_value")), hcat(log.(num("density")), num("owner_occupied"), num("non_western")))
