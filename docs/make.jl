using Documenter
using SpatialEconometrics

makedocs(
    sitename = "SpatialEconometrics.jl",
    modules = [SpatialEconometrics],
    authors = "Nigel Randsley",
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
        "Package Extensions" => "extensions.md",
        "API Reference" => vcat(
            ["Overview" => "api.md"],
            [
            "Variants & Specification Tags" => "api/tags.md",
            "Cross-Sectional Models" => "api/cross_section.md",
            "Static Spatial Panels" => "api/panel.md",
            "Dynamic Spatial Panels" => "api/dynamic_panel.md",
            "GS2SLS / SARAR Panels" => "api/gs2sls.md",
            "Dynamic Panel GMM" => "api/gmm.md",
            "Spatial Error Models" => "api/sem.md",
            "Higher-Order Models" => "api/higher_order.md",
            "Spatial Regime Models" => "api/regime.md",
            "Spatial Discrete Choice" => "api/discrete_choice.md",
            "Spatial Filtering" => "api/filtering.md",
            "Geographically Weighted Regression" => "api/gwr.md",
            "Gravity & Flow Models" => "api/gravity.md",
            "Spatial Diagnostics" => "api/diagnostics.md",
            "Panel Diagnostics" => "api/panel_diagnostics.md",
            "Model Selection" => "api/model_selection.md",
            "Weight Matrices" => "api/weights.md",
            "Robust Inference" => "api/robust.md",
            "Simulation & Power Analysis" => "api/simulation.md",
            "Calibration & Input-Output" => "api/calibration.md",
            "Formula Interface" => "api/formula.md",
            "Output & Tables" => "api/tables.md",
            "Package-Level & Extensions" => "api/package_level.md",
            ],
        ),
    ],
)

deploydocs(
    repo = "github.com/randsley/SpatialEconometrics.jl.git",
    devbranch = "master",
)
