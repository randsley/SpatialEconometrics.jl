using Documenter
using SpatialEconometrics

makedocs(
    sitename = "SpatialEconometrics.jl",
    modules = [SpatialEconometrics],
    authors = "Nigel Randsley",
    warnonly = [:missing_docs],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://randsley.github.io/SpatialEconometrics.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Models" => [
            "Cross-Sectional" => "models/cross_sectional.md",
            "Panel" => "models/panel.md",
            "Dynamic Panel" => "models/dynamic_panel.md",
            "Spatial Error (SEM)" => "models/sem.md",
            "GWR & MGWR" => "models/gwr.md",
            "Gravity Models" => "models/gravity.md",
        ],
        "Higher-Order Models" => [
            "SLX, SAC, SDEM" => "models/higher_order.md",
            "Regime Models" => "models/regime.md",
            "Discrete Choice" => "models/discrete_choice.md",
            "Spatial Filtering" => "models/filtering.md",
        ],
        "Diagnostics" => [
            "Spatial Diagnostics" => "diagnostics/spatial.md",
            "Panel Diagnostics" => "diagnostics/panel.md",
            "Model Selection" => "diagnostics/model_selection.md",
        ],
        "Weight Matrices" => "weights.md",
        "Robust Inference" => "robust.md",
        "Simulation" => "simulation.md",
        "Output & Tables" => "tables.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/randsley/SpatialEconometrics.jl.git",
    devbranch = "master",
)
