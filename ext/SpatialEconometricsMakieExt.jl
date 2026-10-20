module SpatialEconometricsMakieExt

using SpatialEconometrics
using Makie
using Statistics

export plot_coefficients, plot_moran_scatter, plot_local_moran, plot_gwr_coefficients, plot_spatial_effects

"""
    plot_coefficients(m; kwargs...)

Horizontal coefficient plot with confidence intervals for any `RegressionModel`.
"""
function plot_coefficients(m::StatsAPI.RegressionModel; level::Float64=0.95, kwargs...)
    β = StatsAPI.coef(m)
    se = StatsAPI.stderror(m)
    names = StatsAPI.coefnames(m)
    z_crit = 1.96  # ~95% CI

    K = length(β)
    fig = Figure(; size=(600, max(300, 40 * K)), kwargs...)
    ax = Axis(fig[1, 1], xlabel="Estimate", title="Coefficient Plot")
    ax.yticks = (1:K, names)

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

Moran scatterplot: y vs. Wy (standardized), with quadrant shading.
"""
function plot_moran_scatter(y::AbstractVector, W::AbstractSparseMatrix; kwargs...)
    ȳ = mean(y)
    z = (y .- ȳ) ./ std(y)
    Wz = W * z

    fig = Figure(; size=(500, 500), kwargs...)
    ax = Axis(fig[1, 1], xlabel="z (standardized)", ylabel="Wz (spatial lag)", title="Moran Scatterplot")

    scatter!(ax, z, Wz; color=:steelblue, markersize=5, alpha=0.6)
    hlines!(ax, [0.0]; color=:gray, linewidth=1)
    vlines!(ax, [0.0]; color=:gray, linewidth=1)

    # Regression line
    slope = dot(z, Wz) / dot(z, z)
    x_range = range(minimum(z), maximum(z); length=100)
    lines!(ax, collect(x_range), slope .* collect(x_range); color=:red, linewidth=2)

    return fig
end

"""
    plot_local_moran(lisa, coords; kwargs...)

LISA cluster map showing HH, LL, HL, LH, and ns classifications.
"""
function plot_local_moran(lisa::SpatialEconometrics.LISAResult, coords::AbstractMatrix; kwargs...)
    N = length(lisa.local_I)
    colors = map(lisa.cluster_type) do ct
        ct === :HH ? :red :
        ct === :LL ? :blue :
        ct === :HL ? :pink :
        ct === :LH ? :lightblue :
        :gray80
    end

    fig = Figure(; size=(600, 500), kwargs...)
    ax = Axis(fig[1, 1], xlabel="X", ylabel="Y", title="LISA Cluster Map")
    scatter!(ax, coords[:, 1], coords[:, 2]; color=colors, markersize=8)

    # Legend
    elems = [MarkerElement(; color=c, marker=:circle, markersize=12) for c in [:red, :blue, :pink, :lightblue, :gray80]]
    Legend(fig[1, 2], elems, ["HH", "LL", "HL", "LH", "n.s."])
    return fig
end

"""
    plot_gwr_coefficients(m::GWRModel, varname::String; kwargs...)

Spatial map of a single GWR local coefficient.
"""
function plot_gwr_coefficients(m::SpatialEconometrics.GWRModel, varname::String; kwargs...)
    idx = findfirst(==(varname), m.coefnames)
    isnothing(idx) && error("Variable '$varname' not found in model")

    vals = m.coef_local[:, idx]
    fig = Figure(; size=(600, 500), kwargs...)
    ax = Axis(fig[1, 1], xlabel="X", ylabel="Y", title="GWR: Local β for $varname")
    sc = scatter!(ax, m.coords[:, 1], m.coords[:, 2]; color=vals, markersize=8, colormap=:RdBu)
    Colorbar(fig[1, 2], sc; label="β")
    return fig
end

"""
    plot_spatial_effects(effects; kwargs...)

Grouped bar chart of direct, indirect, and total spatial effects.
"""
function plot_spatial_effects(e::SpatialEconometrics.SpatialEconometricsPrototype.SpatialEffectsResult; kwargs...)
    K = length(e.varnames)
    positions = repeat(1:K, 3)
    dodges = vcat(fill(1, K), fill(2, K), fill(3, K))
    values = vcat(e.direct, e.indirect, e.total)
    errors = vcat(e.direct_se, e.indirect_se, e.total_se)

    fig = Figure(; size=(max(400, 120 * K), 400), kwargs...)
    ax = Axis(fig[1, 1], xticks=(1:K, e.varnames), ylabel="Effect", title="Spatial Effects")
    barplot!(ax, positions, values; dodge=dodges, color=dodges, colormap=[:steelblue, :orange, :green])

    elems = [PolyElement(; color=c) for c in [:steelblue, :orange, :green]]
    Legend(fig[1, 2], elems, ["Direct", "Indirect", "Total"])
    return fig
end

end # module
