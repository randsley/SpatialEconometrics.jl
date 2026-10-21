module SpatialEconometricsMakieExt

using SpatialEconometrics
using Makie
using Statistics
using LinearAlgebra
using SparseArrays
using StatsAPI

# Methods are added to the stubs declared in the parent package. An
# extension cannot introduce reachable functions of its own: nothing ever
# `using`s this module, so a function defined here would be invisible.
import SpatialEconometrics: plot_coefficients, plot_moran_scatter,
    plot_local_moran, plot_gwr_coefficients, plot_spatial_effects

const _qnorm = SpatialEconometrics.SpatialTables._qnorm

"""
    plot_coefficients(m; level=0.95, kwargs...)

Horizontal coefficient plot with confidence intervals for any
`StatsAPI.RegressionModel`. `kwargs` are forwarded to `Figure`.
"""
function plot_coefficients(m::StatsAPI.RegressionModel; level::Real=0.95, kwargs...)
    0 < level < 1 || throw(ArgumentError("level must be in (0, 1), got $level"))
    β = StatsAPI.coef(m)
    se = StatsAPI.stderror(m)
    names = StatsAPI.coefnames(m)
    z_crit = _qnorm(1 - (1 - level) / 2)

    K = length(β)
    fig = Figure(; size=(600, max(300, 40 * K)), kwargs...)
    ax = Axis(fig[1, 1], xlabel="Estimate",
              title="Coefficient Plot ($(round(100 * level; digits=1))% CI)")
    ax.yticks = (1:K, String.(names))

    for i in 1:K
        lo = β[i] - z_crit * se[i]
        hi = β[i] + z_crit * se[i]
        lines!(ax, [lo, hi], [i, i]; color=:steelblue, linewidth=2)
        scatter!(ax, [β[i]], [i]; color=:steelblue, markersize=8)
    end
    vlines!(ax, [0.0]; color=:red, linestyle=:dash, linewidth=1)
    return fig
end

"""
    plot_moran_scatter(y, W; kwargs...)

Moran scatterplot: standardized `y` against its spatial lag `Wy`, with the
slope — Moran's I when `W` is row-standardized — drawn through it.
"""
function plot_moran_scatter(y::AbstractVector, W::AbstractSparseMatrix; kwargs...)
    z = (y .- mean(y)) ./ std(y)
    Wz = W * z
    slope = dot(z, Wz) / dot(z, z)

    fig = Figure(; size=(500, 500), kwargs...)
    ax = Axis(fig[1, 1], xlabel="z (standardized)", ylabel="Wz (spatial lag)",
              title="Moran Scatterplot (slope = $(round(slope; digits=3)))")

    scatter!(ax, z, Wz; color=:steelblue, markersize=5, alpha=0.6)
    hlines!(ax, [0.0]; color=:gray, linewidth=1)
    vlines!(ax, [0.0]; color=:gray, linewidth=1)

    x_range = collect(range(minimum(z), maximum(z); length=100))
    lines!(ax, x_range, slope .* x_range; color=:red, linewidth=2)
    return fig
end

"""
    plot_local_moran(lisa, coords; kwargs...)

LISA cluster map showing HH, LL, HL, LH and non-significant units.
"""
function plot_local_moran(lisa::SpatialEconometrics.LISAResult,
                          coords::AbstractMatrix; kwargs...)
    size(coords, 1) == length(lisa.local_I) || throw(DimensionMismatch(
        "coords has $(size(coords, 1)) rows but lisa covers $(length(lisa.local_I)) units"))

    palette = (HH=:red, LL=:blue, HL=:pink, LH=:lightblue, ns=:gray80)
    colors = map(ct -> get(palette, ct, palette.ns), lisa.cluster_type)

    fig = Figure(; size=(600, 500), kwargs...)
    ax = Axis(fig[1, 1], xlabel="X", ylabel="Y", title="LISA Cluster Map")
    scatter!(ax, coords[:, 1], coords[:, 2]; color=colors, markersize=8)

    elems = [MarkerElement(; color=c, marker=:circle, markersize=12)
             for c in (palette.HH, palette.LL, palette.HL, palette.LH, palette.ns)]
    Legend(fig[1, 2], elems, ["HH", "LL", "HL", "LH", "n.s."])
    return fig
end

"""
    plot_gwr_coefficients(m, varname; kwargs...)

Map of a single GWR local coefficient over the estimation coordinates.
"""
function plot_gwr_coefficients(m::SpatialEconometrics.GWRModel,
                               varname::AbstractString; kwargs...)
    idx = findfirst(==(String(varname)), String.(m.coefnames))
    isnothing(idx) && error(
        "variable \"$(varname)\" is not in the model (has: $(m.coefnames))")

    vals = m.coef_local[:, idx]
    fig = Figure(; size=(600, 500), kwargs...)
    ax = Axis(fig[1, 1], xlabel="X", ylabel="Y",
              title="GWR: local β for $(varname)")
    sc = scatter!(ax, m.coords[:, 1], m.coords[:, 2];
                  color=vals, markersize=8, colormap=:RdBu)
    Colorbar(fig[1, 2], sc; label="β")
    return fig
end

"""
    plot_spatial_effects(effects; kwargs...)

Grouped bar chart of direct, indirect and total spatial effects.
"""
function plot_spatial_effects(
    e::SpatialEconometrics.SpatialEconometricsPrototype.SpatialEffectsResult;
    kwargs...
)
    K = length(e.varnames)
    positions = repeat(1:K, 3)
    dodges = vcat(fill(1, K), fill(2, K), fill(3, K))
    values = vcat(e.direct, e.indirect, e.total)

    fig = Figure(; size=(max(400, 120 * K), 400), kwargs...)
    ax = Axis(fig[1, 1], xticks=(1:K, String.(e.varnames)), ylabel="Effect",
              title="Spatial Effects")
    barplot!(ax, positions, values; dodge=dodges, color=dodges,
             colormap=[:steelblue, :orange, :green])

    elems = [PolyElement(; color=c) for c in (:steelblue, :orange, :green)]
    Legend(fig[1, 2], elems, ["Direct", "Indirect", "Total"])
    return fig
end

end # module
