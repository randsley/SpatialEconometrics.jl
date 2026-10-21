# ═══════════════════════════════════════════════════════════════════════
# Example 14: Output & Publication Tables
# ═══════════════════════════════════════════════════════════════════════
#
# Demonstrates:
#   - Tables.jl-compatible coefficient tables
#   - Model fit summaries
#   - Export to LaTeX, CSV and Markdown
#   - Side-by-side regression tables across specifications

using SpatialEconometrics
using Random, LinearAlgebra, SparseArrays
using Tables

const CS = SpatialEconometrics.CrossSection

rng = MersenneTwister(1001)
W = make_rook_weights(16)
N = size(W, 1)
X = randn(rng, N, 2)
y = (I - 0.55 * W) \ (1.0 .+ X * [1.5, -0.9] .+ (W * X) * [0.7, -0.4] .+ 0.5 * randn(rng, N))

m_sar  = fit_spatial(CS.SAR(), W, X, y; varnames = ["income", "density"])
m_sdm  = fit_spatial(CS.SDM(), W, X, y; varnames = ["income", "density"])
m_slx  = fit_slx(W, X, y; varnames = ["income", "density"])
m_sdem = fit_sdem(W, X, y; varnames = ["income", "density"])

# ─── Coefficient table ───────────────────────────────────────────────

println("═══ Coefficient Table ═══")
ct = coeftable(m_sdm)
println(ct)
println()

# It is a Tables.jl source, so it flows into any Tables.jl consumer
# (DataFrames, CSV.jl, Arrow, ...) without conversion.
println("Tables.istable: ", Tables.istable(typeof(ct)))
println("Column names:   ", Tables.columnnames(ct))
println()

# A different confidence level:
println("═══ 99% Confidence Intervals ═══")
println(coeftable(m_sdm; level = 0.99))
println()

# ─── Fit summary ─────────────────────────────────────────────────────

println("═══ Model Fit Summary ═══")
println(summary_stats(m_sdm; model_type = "Spatial Durbin Model"))
println()

# ─── Export formats ──────────────────────────────────────────────────

println("═══ LaTeX ═══")
println(to_latex(ct))

println("═══ Markdown ═══")
println(to_markdown(ct))

println("═══ CSV ═══")
println(to_csv(ct))

# ─── Side-by-side comparison ─────────────────────────────────────────
# The usual publication layout: one column per specification, with
# coefficients aligned on variable names.

println("═══ Regression Table (four specifications) ═══")
println(regression_table(m_slx, m_sar, m_sdm, m_sdem;
    model_names = ["SLX", "SAR", "SDM", "SDEM"]))
println()
println("The data were generated as an SDM, so column (3) should show the")
println("best fit. SLX omits the endogenous lag; SAR omits the Durbin terms.")
println()

# ─── Writing to disk ─────────────────────────────────────────────────
# Uncomment to write the files. They are left commented so running this
# example does not litter your working directory.
#
# write("sdm_table.tex", to_latex(ct))
# write("sdm_table.csv", to_csv(ct))
# write("sdm_table.md",  to_markdown(ct))
# write("comparison.md", regression_table(m_slx, m_sar, m_sdm, m_sdem;
#     model_names = ["SLX", "SAR", "SDM", "SDEM"]))

println("To write these to disk, see the commented block at the end of this file.")
