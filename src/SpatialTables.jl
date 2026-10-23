module SpatialTables

using LinearAlgebra
using Statistics
using Printf
using Tables
using StatsAPI

export CoefTable, coeftable, summary_stats,
       to_latex, to_csv, to_markdown,
       regression_table

# ═══════════════════════════════════════════════════════════════════════
# CoefTable — a Tables.jl-compatible coefficient table
# ═══════════════════════════════════════════════════════════════════════

"""
    CoefTable

A coefficient table ready for display or export, returned by
`coeftable`.

# Fields
- `names`, `estimate`, `std_error`, `z_stat`, `p_value`: the usual columns.
- `ci_lower`, `ci_upper`, `ci_level`: confidence interval bounds and the
  level they correspond to.

Export with [`to_latex`](@ref), [`to_csv`](@ref) or
[`to_markdown`](@ref).

!!! note
    Inference is normal-theory: `z_stat` and `p_value` use the standard
    normal, not a t distribution, which suits the large-sample
    maximum-likelihood estimators here.
"""
struct CoefTable{T<:AbstractFloat}
    names::Vector{String}
    estimate::Vector{T}
    std_error::Vector{T}
    z_stat::Vector{T}
    p_value::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    ci_level::T
end

Tables.istable(::Type{<:CoefTable}) = true
Tables.columnaccess(::Type{<:CoefTable}) = true
Tables.columns(ct::CoefTable) = ct
Tables.columnnames(::CoefTable) = (:name, :estimate, :std_error, :z_stat, :p_value, :ci_lower, :ci_upper)
function Tables.getcolumn(ct::CoefTable, nm::Symbol)
    nm === :name && return ct.names
    nm === :estimate && return ct.estimate
    nm === :std_error && return ct.std_error
    nm === :z_stat && return ct.z_stat
    nm === :p_value && return ct.p_value
    nm === :ci_lower && return ct.ci_lower
    nm === :ci_upper && return ct.ci_upper
    error("Unknown column: $nm")
end
Tables.schema(ct::CoefTable{T}) where {T} = Tables.Schema(
    (:name, :estimate, :std_error, :z_stat, :p_value, :ci_lower, :ci_upper),
    (String, T, T, T, T, T, T)
)

"""
    coeftable(m; level=0.95)

Construct a `CoefTable` from any `StatsAPI.RegressionModel`.
"""
function coeftable(m::StatsAPI.RegressionModel; level::Float64=0.95)
    T = Float64
    β = collect(T, StatsAPI.coef(m))
    se = sqrt.(max.(zero(T), diag(StatsAPI.vcov(m))))
    z = β ./ max.(se, eps(T))
    p = T(2) .* _normcdf.(-abs.(z))
    q = _qnorm((one(T) + T(level)) / T(2))
    lo = β .- q .* se
    hi = β .+ q .* se
    names = try String.(StatsAPI.coefnames(m)) catch; ["β$i" for i in eachindex(β)] end
    return CoefTable{T}(names, β, se, z, p, lo, hi, T(level))
end

function _normcdf(x::T) where {T<:AbstractFloat}
    t = one(T) / (one(T) + T(0.2316419) * abs(x))
    d = T(0.3989422804014327) * exp(-x * x / T(2))
    p = d * t * (T(0.3193815) + t * (T(-0.3565638) + t * (T(1.781478) + t * (T(-1.8212560) + t * T(1.3302744)))))
    return x >= zero(T) ? one(T) - p : p
end

function _qnorm(p::T) where {T<:AbstractFloat}
    a = [T(-3.969683028665376e+01), T(2.209460984245205e+02), T(-2.759285104469687e+02),
         T(1.383577518672690e+02), T(-3.066479806614716e+01), T(2.506628277459239e+00)]
    b = [T(-5.447609879822406e+01), T(1.615858368580409e+02), T(-1.556989798598866e+02),
         T(6.680131188771972e+01), T(-1.328068155288572e+01)]
    c = [T(-7.784894002430293e-03), T(-3.223964580411365e-01), T(-2.400758277161838e+00),
         T(-2.549732539343734e+00), T(4.374664141464968e+00), T(2.938163982698783e+00)]
    d = [T(7.784695709041462e-03), T(3.224671290700398e-01), T(2.445134137142996e+00),
         T(3.754408661907416e+00)]
    p_low = T(0.02425); p_high = one(T) - p_low
    if p < p_low
        q = sqrt(T(-2) * log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+one(T))
    elseif p <= p_high
        q = p - T(0.5); r = q * q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+one(T))
    else
        q = sqrt(T(-2) * log(one(T) - p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+one(T))
    end
end

function Base.show(io::IO, ct::CoefTable{T}) where {T}
    level_pct = round(Int, ct.ci_level * 100)
    @printf(io, "%-20s %10s %10s %10s %10s  [%d%% CI]\n",
        "Variable", "Estimate", "Std.Err", "z", "P>|z|", level_pct)
    println(io, "─"^75)
    for i in eachindex(ct.names)
        stars = ct.p_value[i] < 0.001 ? "***" : ct.p_value[i] < 0.01 ? "**" : ct.p_value[i] < 0.05 ? "*" : ""
        @printf(io, "%-20s %10.4f %10.4f %10.3f %10.4f  [%8.4f, %8.4f] %s\n",
            ct.names[i], ct.estimate[i], ct.std_error[i], ct.z_stat[i],
            ct.p_value[i], ct.ci_lower[i], ct.ci_upper[i], stars)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Model Summary Statistics
# ═══════════════════════════════════════════════════════════════════════

struct ModelSummary{T<:AbstractFloat}
    n::Int
    k::Int
    loglik::T
    aic::T
    bic::T
    sigma2::T
    r2::T
    model_type::String
end

"""
    summary_stats(m; model_type="Spatial Model")

Compute summary statistics (N, K, LL, AIC, BIC, σ², R²) for a fitted model.
"""
function summary_stats(m::StatsAPI.RegressionModel; model_type::String="Spatial Model")
    T = Float64
    n = Int(StatsAPI.nobs(m))
    β = StatsAPI.coef(m)
    k = length(β)
    ll = T(StatsAPI.loglikelihood(m))
    aic = T(-2) * ll + T(2) * k
    bic = T(-2) * ll + log(T(n)) * k
    res = StatsAPI.residuals(m)
    σ2 = dot(res, res) / n
    y = try m.y catch; res end
    ss_res = dot(res, res)
    y_bar = mean(y)
    ss_tot = sum((yi - y_bar)^2 for yi in y)
    r2 = ss_tot > zero(T) ? one(T) - ss_res / ss_tot : zero(T)
    return ModelSummary{T}(n, k, ll, aic, bic, σ2, r2, model_type)
end

function Base.show(io::IO, s::ModelSummary)
    println(io, "Model: $(s.model_type)")
    println(io, "─"^35)
    @printf(io, "  N = %d,  K = %d\n", s.n, s.k)
    @printf(io, "  Log-Lik = %.2f\n", s.loglik)
    @printf(io, "  AIC = %.2f,  BIC = %.2f\n", s.aic, s.bic)
    @printf(io, "  σ² = %.4f,  R² = %.4f\n", s.sigma2, s.r2)
end

# ═══════════════════════════════════════════════════════════════════════
# Export Formats
# ═══════════════════════════════════════════════════════════════════════

"""
    to_latex(ct::CoefTable; digits=4)

Format coefficient table as a LaTeX tabular environment string.
"""
# Each exporter writes into a format with its own metacharacters, and a
# coefficient name is safe in none of them. The names this package produces
# routinely carry `_` (`W_x1` from SLX, `R1_(Intercept)` from regimes) and
# Greek letters (`ρ`, `λ`), so the escaping below is not decoration: without
# it `to_latex` emits a document that will not compile and `to_markdown`
# emits a table that will not render.

const _LATEX_GREEK = Dict(
    'ρ' => "\$\\rho\$",
    'λ' => "\$\\lambda\$",
    'θ' => "\$\\theta\$",
    'σ' => "\$\\sigma\$",
    'β' => "\$\\beta\$",
    'τ' => "\$\\tau\$",
    'α' => "\$\\alpha\$",
    'μ' => "\$\\mu\$",
    'γ' => "\$\\gamma\$",
    'δ' => "\$\\delta\$",
    'φ' => "\$\\phi\$",
    'ω' => "\$\\omega\$",
    'ε' => "\$\\varepsilon\$",
    '²' => "\$^2\$"
)

# Escape a coefficient name for LaTeX text mode.
function _escape_latex(str::AbstractString)
    io = IOBuffer()
    for c in str
        if haskey(_LATEX_GREEK, c)
            print(io, _LATEX_GREEK[c])
        elseif c in ('_', '&', '%', '#', '{', '}', '$')
            print(io, '\\', c)
        elseif c == '~'
            print(io, "\\textasciitilde{}")
        elseif c == '^'
            print(io, "\\textasciicircum{}")
        elseif c == '\\'
            print(io, "\\textbackslash{}")
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

# Escape a cell for a GitHub-flavoured Markdown table: an unescaped `|`
# closes the cell early and shifts every column after it.
_escape_md(str::AbstractString) = replace(str, "\\" => "\\\\", "|" => "\\|")

# Quote a CSV field per RFC 4180 when it holds a comma, quote or newline.
function _csv_field(str::AbstractString)
    needs = any(c -> c in (',', '"', '\n', '\r'), str)
    return needs ? '"' * replace(str, "\"" => "\"\"") * '"' : String(str)
end

# `digits` has to reach Printf at run time, so build the format rather than
# relying on a literal @sprintf.
_fmt(x::Real, digits::Int) =
    Printf.format(Printf.Format("%." * string(max(digits, 0)) * "f"), x)

function to_latex(ct::CoefTable; digits::Int=4)
    io = IOBuffer()
    println(io, "\\begin{tabular}{lcccc}")
    println(io, "\\hline")
    println(io, "Variable & Estimate & Std.Err & z & P\\textgreater|z| \\\\")
    println(io, "\\hline")
    for i in eachindex(ct.names)
        stars = ct.p_value[i] < 0.001 ? "\$^{***}\$" : ct.p_value[i] < 0.01 ? "\$^{**}\$" : ct.p_value[i] < 0.05 ? "\$^{*}\$" : ""
        est = _fmt(ct.estimate[i], digits)
        se = _fmt(ct.std_error[i], digits)
        zs = _fmt(ct.z_stat[i], digits)
        pv = _fmt(ct.p_value[i], digits)
        nm = _escape_latex(ct.names[i])
        println(io, "$nm & $est & $se & $zs & $pv$stars \\\\")
    end
    println(io, "\\hline")
    println(io, "\\end{tabular}")
    return String(take!(io))
end

"""
    to_csv(ct::CoefTable)

Format coefficient table as CSV string.
"""
function to_csv(ct::CoefTable)
    io = IOBuffer()
    println(io, "name,estimate,std_error,z_stat,p_value,ci_lower,ci_upper")
    for i in eachindex(ct.names)
        @printf(io, "%s,%.8f,%.8f,%.6f,%.8f,%.8f,%.8f\n",
            _csv_field(ct.names[i]), ct.estimate[i], ct.std_error[i],
            ct.z_stat[i], ct.p_value[i], ct.ci_lower[i], ct.ci_upper[i])
    end
    return String(take!(io))
end

"""
    to_markdown(ct::CoefTable; digits=4)

Format coefficient table as a Markdown table string.
"""
function to_markdown(ct::CoefTable; digits::Int=4)
    io = IOBuffer()
    # `P>|z|` needs escaped pipes: written bare they read as two further
    # cell separators, so the header row declared eight cells against the
    # separator row's six and no renderer lined the table up.
    println(io, "| Variable | Estimate | Std.Err | z | P>\\|z\\| | Sig |")
    println(io, "|:---------|--------:|--------:|---:|------:|:---:|")
    for i in eachindex(ct.names)
        stars = ct.p_value[i] < 0.001 ? "***" : ct.p_value[i] < 0.01 ? "**" : ct.p_value[i] < 0.05 ? "*" : ""
        @printf(io, "| %s | %s | %s | %s | %s | %s |\n",
            _escape_md(ct.names[i]), _fmt(ct.estimate[i], digits),
            _fmt(ct.std_error[i], digits), _fmt(ct.z_stat[i], digits),
            _fmt(ct.p_value[i], digits), stars)
    end
    return String(take!(io))
end

# ═══════════════════════════════════════════════════════════════════════
# Multi-Model Regression Table
# ═══════════════════════════════════════════════════════════════════════

"""
    regression_table(models...; model_names=nothing)

Side-by-side regression table for multiple models (Stargazer-style).
Returns a formatted string.
"""
function regression_table(models::StatsAPI.RegressionModel...; model_names::Union{Nothing, Vector{String}}=nothing)
    n_models = length(models)
    mnames = isnothing(model_names) ? ["($i)" for i in 1:n_models] : model_names

    all_vars = String[]
    for m in models
        cns = try String.(StatsAPI.coefnames(m)) catch; ["β$i" for i in 1:length(StatsAPI.coef(m))] end
        for v in cns
            v in all_vars || push!(all_vars, v)
        end
    end

    io = IOBuffer()
    col_w = 16
    var_w = 20

    @printf(io, "%-*s", var_w, "")
    for nm in mnames; @printf(io, " %*s", col_w, nm); end
    println(io)
    println(io, "─"^(var_w + n_models * (col_w + 1)))

    for v in all_vars
        @printf(io, "%-*s", var_w, v)
        for m in models
            cns = try String.(StatsAPI.coefnames(m)) catch; ["β$i" for i in 1:length(StatsAPI.coef(m))] end
            idx = findfirst(==(v), cns)
            if isnothing(idx)
                @printf(io, " %*s", col_w, "")
            else
                β_val = StatsAPI.coef(m)[idx]
                se_val = sqrt(max(0.0, StatsAPI.vcov(m)[idx, idx]))
                p_val = 2.0 * _normcdf(-abs(β_val / max(se_val, eps())))
                stars = p_val < 0.001 ? "***" : p_val < 0.01 ? "**" : p_val < 0.05 ? "*" : ""
                est_str = @sprintf("%.4f%s", β_val, stars)
                @printf(io, " %*s", col_w, est_str)
            end
        end
        println(io)

        @printf(io, "%-*s", var_w, "")
        for m in models
            cns = try String.(StatsAPI.coefnames(m)) catch; ["β$i" for i in 1:length(StatsAPI.coef(m))] end
            idx = findfirst(==(v), cns)
            if isnothing(idx)
                @printf(io, " %*s", col_w, "")
            else
                se_val = sqrt(max(0.0, StatsAPI.vcov(m)[idx, idx]))
                se_str = @sprintf("(%.4f)", se_val)
                @printf(io, " %*s", col_w, se_str)
            end
        end
        println(io)
    end

    println(io, "─"^(var_w + n_models * (col_w + 1)))

    @printf(io, "%-*s", var_w, "N")
    for m in models; @printf(io, " %*d", col_w, StatsAPI.nobs(m)); end
    println(io)

    @printf(io, "%-*s", var_w, "Log-Lik")
    for m in models; @printf(io, " %*.2f", col_w, StatsAPI.loglikelihood(m)); end
    println(io)

    @printf(io, "%-*s", var_w, "AIC")
    for m in models
        k = length(StatsAPI.coef(m))
        aic = -2.0 * StatsAPI.loglikelihood(m) + 2.0 * k
        @printf(io, " %*.2f", col_w, aic)
    end
    println(io)

    println(io, "─"^(var_w + n_models * (col_w + 1)))
    println(io, "Significance: *** p<0.001, ** p<0.01, * p<0.05")

    return String(take!(io))
end

end # module
