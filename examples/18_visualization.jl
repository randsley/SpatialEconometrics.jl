# ═══════════════════════════════════════════════════════════════════════
# Example 18: Visualization (Makie extension)
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Coefficient plots with a configurable confidence level
#   - Moran scatterplots, and what the slope is actually measuring
#   - LISA cluster maps
#   - GWR local-coefficient maps
#   - Direct / indirect / total effect bar charts
#   - What happens when Makie is NOT loaded
#
# Makie is a WEAK dependency: it is not installed with the package, and the
# five plotting functions only acquire methods once Makie is in the session.
# To run this example you need a Makie backend:
#
#     julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'
#     julia --project=. examples/18_visualization.jl
#
# CairoMakie writes PNG/SVG/PDF without a display, so this works headless.

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays, Statistics
import StatsAPI: coef, residuals


# ─── Before Makie is loaded ──────────────────────────────────────────

println("═══ The Extension Is Not Loaded Yet ═══")
# The names are exported by SpatialEconometrics itself, not by the
# extension — an extension module is never `using`d by anyone, so anything
# it declared on its own would be unreachable. What the parent provides is
# a stub that explains itself instead of raising a bare MethodError.
try
    plot_coefficients(nothing)
catch e
    println("  ", sprint(showerror, e))
end
println()

using CairoMakie

println("═══ Extension Loaded ═══")
# The extension triggers on Makie, which the backend pulls in; the two carry
# different version numbers, and it is Makie's that the compat bound covers.
println("  Makie version:      ", pkgversion(Makie))
println("  CairoMakie version: ", pkgversion(CairoMakie))
println("  extension loaded: ", !isnothing(Base.get_extension(
            SpatialEconometrics, :SpatialEconometricsMakieExt)))
println()

outdir = mktempdir()

# ─── Data ────────────────────────────────────────────────────────────

rng = MersenneTwister(808)
L = 14; N = L * L
W = make_rook_weights(L)
coords = hcat(repeat(1.0:L, outer = L), repeat(1.0:L, inner = L))

X = randn(rng, N, 2)
y = (sparse(1.0I, N, N) - 0.55 * W) \
    (1.0 .+ X * [1.5, -0.8] .+ 0.5 .* randn(rng, N))

m = fit_spatial(SAR(), W, X, y; varnames = ["income", "density"])
println("SAR fit: ρ̂ = ", round(m.rho; digits = 4),
        ",  β̂ = ", round.(coef(m)[2:3]; digits = 4))
println()

# ─── Coefficient plot ────────────────────────────────────────────────

println("═══ Coefficient Plots ═══")
save(joinpath(outdir, "coefficients_95.png"), plot_coefficients(m))
save(joinpath(outdir, "coefficients_99.png"), plot_coefficients(m; level = 0.99))
println("  wrote coefficients_95.png and coefficients_99.png")

# `level` is a real parameter, not decoration: the critical value comes from
# the normal quantile, so the 99% interval is genuinely wider.
function ci_span(fig)
    seg = fig.content[1].scene.plots[1][1][]
    return abs(seg[2][1] - seg[1][1])
end
println("  first coefficient's drawn interval: ",
        round(ci_span(plot_coefficients(m; level = 0.95)); digits = 4),
        " at 95%, ",
        round(ci_span(plot_coefficients(m; level = 0.99)); digits = 4),
        " at 99%")

# A level outside (0, 1) is rejected rather than quietly producing nonsense.
for bad in (0.0, 1.0, 1.5, -0.1)
    try
        plot_coefficients(m; level = bad)
        println("  level = $bad -> accepted (it should not have been)")
    catch e
        println("  level = ", rpad(bad, 5), " -> ", sprint(showerror, e))
    end
end
println()

# ─── Moran scatterplot ───────────────────────────────────────────────

println("═══ Moran Scatterplot ═══")
save(joinpath(outdir, "moran_y.png"), plot_moran_scatter(y, W))
save(joinpath(outdir, "moran_resid.png"), plot_moran_scatter(residuals(m), W))
println("  wrote moran_y.png (raw y) and moran_resid.png (SAR residuals)")

# The slope the plot draws is Moran's I when W is row-standardized, so the
# two should agree. Worth checking rather than asserting.
z = (y .- mean(y)) ./ std(y)
slope = dot(z, W * z) / dot(z, z)
println("  drawn slope = ", round(slope; digits = 4),
        "   moran_global(y, W).I_stat = ",
        round(moran_global(y, W).I_stat; digits = 4))
println("  on SAR residuals, Moran's I = ",
        round(moran_global(residuals(m), W).I_stat; digits = 4),
        " — the fit has absorbed the dependence")
println()

# ─── LISA cluster map ────────────────────────────────────────────────

println("═══ LISA Cluster Map ═══")
lisa = moran_local(y, W; n_perm = 999, rng = MersenneTwister(4))
save(joinpath(outdir, "lisa.png"), plot_local_moran(lisa, coords))
println("  wrote lisa.png")
# `cluster_type` holds Symbols, with :ns for the units the permutation test
# does not call significant.
for ct in (:HH, :LL, :HL, :LH, :ns)
    println("    ", rpad(ct, 4), " units: ", count(==(ct), lisa.cluster_type))
end

# coords and the LISA result have to describe the same units; a mismatch is
# caught rather than silently plotting the shorter of the two.
try
    plot_local_moran(lisa, coords[1:end-1, :])
catch e
    println("  truncated coords -> ", sprint(showerror, e))
end
println()

# ─── GWR coefficient map ─────────────────────────────────────────────

println("═══ GWR Local Coefficient Map ═══")
# Give the DGP a coefficient that genuinely varies over space, so the map
# has something to show.
β1_local = 0.5 .+ 0.15 .* coords[:, 1]
y_gwr = 1.0 .+ X[:, 1] .* β1_local .+ X[:, 2] .* (-0.8) .+ 0.4 .* randn(rng, N)
g = fit_gwr(X, y_gwr, coords; bandwidth = AdaptiveBandwidth(30),
            varnames = ["income", "density"])

save(joinpath(outdir, "gwr_income.png"), plot_gwr_coefficients(g, "income"))
save(joinpath(outdir, "gwr_density.png"), plot_gwr_coefficients(g, "density"))
println("  wrote gwr_income.png and gwr_density.png")

i_inc = findfirst(==("income"), String.(g.coefnames))
loc = g.coef_local[:, i_inc]
println("  local β_income ranges ", round(minimum(loc); digits = 3), " … ",
        round(maximum(loc); digits = 3),
        "  (true: ", round(minimum(β1_local); digits = 3), " … ",
        round(maximum(β1_local); digits = 3), ")")
println("  correlation with the true surface: ",
        round(cor(loc, β1_local); digits = 4))
println("  β_density, which does NOT vary, ranges ",
        round(minimum(g.coef_local[:, findfirst(==("density"), String.(g.coefnames))]); digits = 3),
        " … ",
        round(maximum(g.coef_local[:, findfirst(==("density"), String.(g.coefnames))]); digits = 3))

try
    plot_gwr_coefficients(g, "not_a_variable")
catch e
    println("  unknown variable -> ", sprint(showerror, e))
end
println()

# ─── Spatial effects ─────────────────────────────────────────────────

println("═══ Direct / Indirect / Total Effects ═══")
eff = spatial_effects(m)
save(joinpath(outdir, "effects.png"), plot_spatial_effects(eff))
println("  wrote effects.png")
println(rpad("  variable", 14), rpad("direct", 11), rpad("indirect", 11), "total")
for (i, v) in enumerate(eff.varnames)
    println(rpad("  $v", 14),
            rpad(round(eff.direct[i]; digits = 4), 11),
            rpad(round(eff.indirect[i]; digits = 4), 11),
            round(eff.total[i]; digits = 4))
end
println()
println("  In a SAR the indirect effect is what the bar chart exists to show:")
println("  a change in x spills through (I - ρW)⁻¹ to every other unit, so the")
println("  total effect is larger than the coefficient you estimated.")
println()

# ─── Summary ─────────────────────────────────────────────────────────

println("═══ Files Written ═══")
for f in sort(readdir(outdir))
    println("  ", rpad(f, 24), lpad(filesize(joinpath(outdir, f)), 8), " bytes")
end
println()
println("  Directory: ", outdir)
println("  All five plotting functions return a Makie `Figure`, so they")
println("  compose with the rest of Makie: `save`, layouts, further axes.")
