# root_rho on real data: the 278 mainland Portuguese municipalities (notebooks/data, INE 2023 extract).
#
# The model is notebook 11's SAR with a smooth surface added: purchasing power (and its log) on log
# density and the ageing ratio, y = ρWy + Xβ + f(u, v) + ε, with f a smooth function of the centroid.
# Nothing here is simulated: the point is to see what real W, real leverage and real errors do to
# root_rho, and where its guards and messages fire.
#
# Reported per fit: K, K/n, max leverage, ρ̂ (and the first-step ρ̃), the HC2 SE and Wald interval, the
# test-inversion set (pieces, whether it reaches a bound), vhat_worst_bias (symmetric, and with
# skewness = 2), β̂, and every warning or error root_rho raised. Baselines: the ML SAR without a
# surface (notebook 11) and root_rho without a surface (an intercept in X, no basis).
# Bases: the projection on tensor_pspline(u, v; k) for k = 4…10, with the columns that no municipality
# touches dropped (the bounding box of mainland Portugal includes sea and Spain); and the penalised
# smoother on k = 12 with λ = 0.1, 1, 10. W: queen and rook contiguity (the GAL files), kNN-6 on the
# centroids. Coordinates: lon/lat projected to km (equirectangular at the mean latitude).
#
# Data credit (a condition of the licences): Source: Statistics Portugal (INE), indicators 0014580,
# 0013189, 0012909, reference period 2023, extracted 2026-11-07. Informação geográfica cedida pela
# Direção-Geral do Território (CAOP, CC-BY-4.0).
#
# Run: julia --project=. research/trace_cfe/tc_portugal.jl > research/trace_cfe/tc_portugal_output.txt
using SpatialEconometrics, LinearAlgebra, SparseArrays, Statistics, Printf, Logging, Test

const DATA = joinpath(@__DIR__, "..", "..", "notebooks", "data")

# --- data, as notebook 11 reads it -----------------------------------------------------------------
lines = readlines(joinpath(DATA, "portugal_municipalities.csv"))
hdr = split(strip(lines[1]), ','); rows = [split(strip(l), ',') for l in lines[2:end] if !isempty(strip(l))]
col(name) = [r[findfirst(==(name), hdr)] for r in rows]
num(name) = [parse(Float64, s) for s in col(name)]
keep = findall(col("mainland") .== "true")
dico = col("dico")[keep]; place = col("name")[keep]
pp = num("purchasing_power")[keep]
X = hcat(log.(num("density")[keep]), num("ageing_ratio")[keep]); vn = ["log_density", "ageing_ratio"]
n = length(keep)

cl = readlines(joinpath(DATA, "portugal_municipalities_centroids.csv"))
ch = split(cl[1], ','); cr = [split(l, ',') for l in cl[2:end] if !isempty(strip(l))]
cdico = [r[findfirst(==("dico"), ch)] for r in cr]
lon = [parse(Float64, r[findfirst(==("lon"), ch)]) for r in cr]
lat = [parse(Float64, r[findfirst(==("lat"), ch)]) for r in cr]
cdico == dico || error("centroids are not in the CSV's order")
φ0 = mean(lat)
u = (lon .- mean(lon)) .* 111.32 .* cosd(φ0); v = (lat .- mean(lat)) .* 110.57     # km

Wq, qid = read_gal(joinpath(DATA, "portugal_municipalities_queen.gal"))
Wr, rid = read_gal(joinpath(DATA, "portugal_municipalities_rook.gal"))
(qid == dico && rid == dico) || error("GAL order differs from the CSV")
Wk = knn_weights(hcat(u, v), 6)
WS = (("queen", Wq), ("rook", Wr), ("kNN-6", Wk))

@printf("n = %d mainland municipalities; extent %.0f km (E–W) × %.0f km (N–S)\n", n,
        maximum(u) - minimum(u), maximum(v) - minimum(v))
@printf("purchasing power: median %.1f, range %.1f–%.1f (max: %s); skewness %.2f\n", median(pp),
        extrema(pp)..., place[argmax(pp)], mean(((pp .- mean(pp)) ./ std(pp)) .^ 3))
for (nm, W) in WS
    nn = vec(sum(W .> 0; dims = 2))
    @printf("  W %-6s: links %d, neighbours per unit %d–%d (mean %.2f)\n", nm, nnz(W), extrema(nn)..., mean(nn))
end
println()

# --- helpers ---------------------------------------------------------------------------------------
function trimmed_basis(k)
    B = tensor_pspline(u, v; k).basis
    used = vec(maximum(B; dims = 1)) .> 0
    (B[:, used], count(!, used))
end

# run root_rho, capturing warnings and errors
function fit(y, Xm, W; kw...)
    tl = TestLogger(min_level = Logging.Warn)
    r = nothing; err = nothing
    with_logger(tl) do
        try
            r = root_rho(y, Xm, W; kw...)
        catch e
            err = sprint(showerror, e)
        end
    end
    warns = [string(l.message) for l in tl.logs]
    (r, err, warns)
end

short(s; m = 150) = (s = replace(s, '\n' => ' '); length(s) > m ? first(s, m) * "…" : s)
fmtcs(cs) = isempty(cs) ? "empty" : join([@sprintf("[%.3f, %.3f]", a, b) for (a, b) in cs], " ∪ ")

function report(label, y, Xm, W; skew = true, kw...)
    r, err, warns = fit(y, Xm, W; kw...)
    if r === nothing
        @printf("  %-22s ERROR: %s\n", label, short(err))
        return nothing
    end
    ws = skew ? (fit(y, Xm, W; skewness = 2.0, kw...)[1]) : nothing
    wb2 = ws === nothing ? NaN : ws.vhat_worst_bias
    @printf("  %-22s K %3d K/n %.3f hmax %.3f | ρ̂ %+.3f (ρ̃ %+.3f) se %.3f Wald [%+.3f, %+.3f] | CS %s%s | wb %+.4f (γ̄=2 %+.4f) | β̂ %s\n",
            label, r.K, r.K / n, r.max_leverage, r.rho, r.rho_first, r.se, r.ci_wald..., fmtcs(r.cs),
            r.cs_touches_bound ? " (reaches a bound)" : "", r.vhat_worst_bias, wb2,
            join([@sprintf("%+.4g", b) for b in r.beta], ", "))
    for w in unique(warns)
        println("      warning: ", short(w))
    end
    r
end

# --- the fits --------------------------------------------------------------------------------------
for (yname, y) in (("purchasing power", pp), ("log purchasing power", log.(pp)))
    println("=== y = $yname ===")
    for (wname, W) in WS
        println("W = $wname")
        m = fit_spatial(SAR(), W, X, y; method = ExactLU())
        @printf("  %-22s ρ̂ %+.3f (se %.3f), β̂ %s\n", "ML SAR, no surface", m.rho, m.rho_se,
                join([@sprintf("%+.4g", b) for b in m.coef[2:end]], ", "))
        report("root_rho, no surface", y, hcat(ones(n), X), W; skew = false)
        for k in 4:10
            B, dropped = trimmed_basis(k)
            report(@sprintf("projection k=%d (-%d)", k, dropped), y, X, W; basis = B)
        end
        # the untrimmed basis once, to record what a user who does not drop empty columns sees
        wname == "queen" && report("k=6 untrimmed", y, X, W; basis = tensor_pspline(u, v; k = 6).basis, skew = false)
        # the released path refuses these bases (rank check); see the bypass below
        Bp = tensor_pspline(u, v; k = 12)
        report("penalised k=12 λ=1", y, X, W; skew = false, basis = Bp.basis, penalty = Bp.penalty, lambda = 1.0)
        flush(stdout)
    end
    println()
end

# --- residual diagnostics at one fit, to judge which worst case applies ----------------------------
println("Residual diagnostics: y = log purchasing power, queen, projection k = 4 (the only k that runs)")
let y = log.(pp), (B, _) = trimmed_basis(4)
    r, _, _ = fit(y, X, Wq; basis = B)
    Z = hcat(X, B); e = (I - Z * pinv(Z)) * (y .- r.rho .* (Wq * y))
    z = (e .- mean(e)) ./ std(e)
    @printf("  residual skewness %.2f, excess kurtosis %.2f\n", mean(z .^ 3), mean(z .^ 4) - 3)
    # heteroskedasticity: squared residuals against log density, and Moran's I on squared residuals
    Xd = hcat(ones(n), X[:, 1]); bd = Xd \ (e .^ 2)
    @printf("  e² on log density: slope %.4f (mean e² %.4f); ", bd[2], mean(e .^ 2))
    println("Moran's I of e²: ", moran_global(e .^ 2, Wq))
    ord = sortperm(abs.(z); rev = true)
    println("  largest |standardised residuals|: ",
            join([@sprintf("%s %+.1f", place[i], z[i]) for i in ord[1:6]], ", "))
end

# --- the penalised smoother with the rank check bypassed -------------------------------------------
# root_rho checks that [X, basis] has full rank before it branches, but the penalised path needs only
# Z′Z + λΩ̃ positive definite (it takes a Cholesky factor), and λ > 0 keeps every Sᵢᵢ below 1. So
# the untrimmed basis is fitted here through the internal RootRho._root_rho_penalised, which is
# what root_rho would run if the check were limited to what the smoother needs.
println("
Penalised smoother, rank check bypassed (RootRho._root_rho_penalised, untrimmed tensor_pspline)")
const RR = SpatialEconometrics.RootRho
for (yname, y) in (("pp", pp), ("log pp", log.(pp))), (wname, W) in WS, k in (8, 12, 16), λ in (0.1, 1.0, 10.0, 100.0)
    Bp = tensor_pspline(u, v; k)
    Z = hcat(X, Bp.basis)
    try
        r = RR._root_rho_penalised(y, Z, sparse(W), 2, Bp.penalty, λ, vn, 0.95, (-0.99, 0.99), nothing)
        @printf("  %-6s %-6s k=%2d (rank %3d of %3d) λ=%5g | df %5.1f max Sᵢᵢ %.3f | ρ̂ %+.3f se %.3f | CS %s%s
",
                yname, wname, k, rank(Z), size(Z, 2), λ, r.df, r.max_leverage, r.rho, r.se, fmtcs(r.cs),
                r.cs_touches_bound ? " (reaches a bound)" : "")
    catch e
        @printf("  %-6s %-6s k=%2d λ=%5g | ERROR %s
", yname, wname, k, λ, short(sprint(showerror, e)))
    end
end
