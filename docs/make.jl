using Documenter
using SpatialEconometrics

makedocs(
    sitename = "SpatialEconometrics.jl",
    modules = [SpatialEconometrics],
    # Internal helpers carry docstrings for the next reader of the
    # source, not for the manual. Checking every docstring would force
    # private algebra into the public API reference; checking the
    # exported surface keeps the build strict where it matters.
    checkdocs = :exports,
    authors = "Nigel Randsley",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://randsley.github.io/SpatialEconometrics.jl",
        # The logo pair is picked up by name; the stylesheet and the
        # favicon have to be listed, the latter so the link is emitted
        # under the versioned docs path rather than looked for at /.
        assets = [
            "assets/custom.css",
            Documenter.asset("assets/favicon.ico", class = :ico, islocal = true),
        ],
        # The sidebar carries 24 API pages. Collapsing to the top level
        # keeps it readable; a section opens when you are inside it.
        collapselevel = 1,
        edit_link = "master",
        repolink = "https://github.com/randsley/SpatialEconometrics.jl",
    ),
    repo = "https://github.com/randsley/SpatialEconometrics.jl/blob/{commit}{path}#{line}",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Models" => [
            "Cross-Sectional" => "models/cross_sectional.md",
            "Panel" => "models/panel.md",
            "Dynamic Panel" => "models/dynamic_panel.md",
            "Spatial Error (SEM)" => "models/sem.md",
            "GS2SLS / SARAR Panels" => "models/gs2sls.md",
            "Dynamic Panel GMM" => "models/gmm.md",
            "GWR & MGWR" => "models/gwr.md",
            "Gravity Models" => "models/gravity.md",
        ],
        "Higher-Order Models" => [
            "SLX, SAC, SDEM" => "models/higher_order.md",
            "Regime Models" => "models/regime.md",
            "Discrete Choice" => "models/discrete_choice.md",
            "Spatial Filtering" => "models/filtering.md",
            "Causal Inference" => "models/causal.md",
            "Networks" => "models/networks.md",
            "Common Factors & Spatio-Temporal" => "models/common_factors.md",
            "Spatial VAR" => "models/spvar.md",
        ],
        "Diagnostics" => [
            "Spatial Diagnostics" => "diagnostics/spatial.md",
            "Panel Diagnostics" => "diagnostics/panel.md",
            "Model Selection" => "diagnostics/model_selection.md",
        ],
        "Weight Matrices" => "weights.md",
        "Robust Inference" => "robust.md",
        "Simulation" => "simulation.md",
        "Calibration & Input-Output" => "calibration.md",
        "Formula Interface" => "formula.md",
        "Output & Tables" => "tables.md",
        "Package Extensions" => "extensions.md",
        # Grouped rather than listed flat: 24 entries in one column is a
        # wall, and the groups match how the manual is organised.
        "API Reference" => [
            "Overview" => "api.md",
            "Cross-Section & Heterogeneity" => [
                "Variants & Specification Tags" => "api/tags.md",
                "Cross-Sectional Models" => "api/cross_section.md",
                "Spatial Error Models" => "api/sem.md",
                "Higher-Order Models" => "api/higher_order.md",
                "Spatial Regime Models" => "api/regime.md",
                "Spatial Discrete Choice" => "api/discrete_choice.md",
                "Spatial Filtering" => "api/filtering.md",
                "Spatial Causal Inference" => "api/causal.md",
                "Networks" => "api/networks.md",
                "Common Factors & Spatio-Temporal" => "api/common_factors.md",
                "Spatial VAR" => "api/spvar.md",
                "Geographically Weighted Regression" => "api/gwr.md",
            ],
            "Panels" => [
                "Static Spatial Panels" => "api/panel.md",
                "Dynamic Spatial Panels" => "api/dynamic_panel.md",
                "GS2SLS / SARAR Panels" => "api/gs2sls.md",
                "Dynamic Panel GMM" => "api/gmm.md",
            ],
            "Flows & Calibration" => [
                "Gravity & Flow Models" => "api/gravity.md",
                "Calibration & Input-Output" => "api/calibration.md",
            ],
            "Diagnostics & Selection" => [
                "Spatial Diagnostics" => "api/diagnostics.md",
                "Panel Diagnostics" => "api/panel_diagnostics.md",
                "Model Selection" => "api/model_selection.md",
            ],
            "Infrastructure" => [
                "Weight Matrices" => "api/weights.md",
                "Robust Inference" => "api/robust.md",
                "Simulation & Power Analysis" => "api/simulation.md",
                "Formula Interface" => "api/formula.md",
                "Output & Tables" => "api/tables.md",
                "Package-Level & Extensions" => "api/package_level.md",
            ],
        ],
    ],
)

deploydocs(
    repo = "github.com/randsley/SpatialEconometrics.jl.git",
    devbranch = "master",
)
