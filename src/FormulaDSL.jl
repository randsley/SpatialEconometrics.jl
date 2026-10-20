module FormulaDSL

using LinearAlgebra
using SparseArrays
using Random
using StatsAPI
using StatsModels
using Tables

import ..SpatialEconometricsPrototype as CrossSection
import ..SpatialPanelEconometrics as StaticPanel
import ..DynamicSpatialPanelEconometrics as DynamicPanel
import ..SpatialGS2SLSPanel as GS2SLSPanel
import ..DynamicPanelSystemGMM as SystemGMMMod
import ..SpatialErrorModel as SEMMod
import ..SpatialGravityModels as GravityMod

export @formula, slag, fe, endo,
       ParsedSpatialFormula,
       extract_panel_permutation

slag(x...) = error("`slag` should only be used inside `@formula`.")
fe(x)      = error("`fe` should only be used inside `@formula`.")
endo(x...) = error("`endo` should only be used inside `@formula`.")

struct SpatialLagTerm{T} <: StatsModels.AbstractTerm
    inner::T
end

struct FixedEffectTerm <: StatsModels.AbstractTerm
    sym::Symbol
end

struct EndogenousTerm{E, Z} <: StatsModels.AbstractTerm
    endog::E
    instruments::Z
end

StatsModels.termvars(t::SpatialLagTerm) = t.inner isa Tuple ? reduce(vcat, StatsModels.termvars.(t.inner)) : StatsModels.termvars(t.inner)
StatsModels.termvars(t::FixedEffectTerm) = Symbol[t.sym]
StatsModels.termvars(t::EndogenousTerm) = vcat(t.endog isa Tuple ? reduce(vcat, StatsModels.termvars.(t.endog)) : StatsModels.termvars(t.endog), t.instruments isa Tuple ? reduce(vcat, StatsModels.termvars.(t.instruments)) : StatsModels.termvars(t.instruments))

function _collect_raw_terms(t::StatsModels.AbstractTerm)
    if t isa StatsModels.Term
        return StatsModels.AbstractTerm[t]
    elseif t isa StatsModels.FunctionTerm && (t.f === (+) || nameof(t.f) == :+)
        out = StatsModels.AbstractTerm[]
        for a in t.args; append!(out, _collect_raw_terms(a)); end
        return out
    elseif t isa StatsModels.TupleTerm
        out = StatsModels.AbstractTerm[]
        for a in t.terms; append!(out, _collect_raw_terms(a)); end
        return out
    else
        return StatsModels.AbstractTerm[t]
    end
end

function _extract_symbol(t::StatsModels.AbstractTerm)
    if hasproperty(t, :sym); return getproperty(t, :sym)::Symbol
    else
        vars = StatsModels.termvars(t)
        !isempty(vars) && return first(vars)
        error("Cannot extract column Symbol from fixed effect term: $t")
    end
end

function StatsModels.apply_schema(t::StatsModels.FunctionTerm{typeof(slag)}, sch::StatsModels.Schema, Mod::Type)
    raw_list = StatsModels.AbstractTerm[]
    for a in t.args; append!(raw_list, _collect_raw_terms(a)); end
    applied_list = [StatsModels.apply_schema(x, sch, Mod) for x in raw_list]
    inner_tup = length(applied_list) == 1 ? applied_list[1] : Tuple(applied_list)
    return SpatialLagTerm(inner_tup)
end

function StatsModels.apply_schema(t::StatsModels.FunctionTerm{typeof(fe)}, sch::StatsModels.Schema, Mod::Type)
    return FixedEffectTerm(_extract_symbol(t.args[1]))
end

function StatsModels.apply_schema(t::StatsModels.FunctionTerm{typeof(endo)}, sch::StatsModels.Schema, Mod::Type)
    @assert length(t.args) >= 2 "`endo(...)` requires `endo(endog_var, iv1 + iv2)`"
    e_raw = _collect_raw_terms(t.args[1])
    z_raw = StatsModels.AbstractTerm[]
    for a in t.args[2:end]; append!(z_raw, _collect_raw_terms(a)); end

    e_app = [StatsModels.apply_schema(x, sch, Mod) for x in e_raw]
    z_app = [StatsModels.apply_schema(x, sch, Mod) for x in z_raw]

    e_term = length(e_app) == 1 ? e_app[1] : Tuple(e_app)
    z_term = length(z_app) == 1 ? z_app[1] : Tuple(z_app)
    return EndogenousTerm(e_term, z_term)
end

struct ParsedSpatialFormula{T<:AbstractFloat}
    y::Vector{T}; lhs_name::String; X_exog::Matrix{T}; exog_names::Vector{String}; X_slag_source::Matrix{T}; slag_names::Vector{String}; X_endog::Matrix{T}; endog_names::Vector{String}; Z_iv::Matrix{T}; iv_names::Vector{String}; fe_symbols::Vector{Symbol}; has_slag::Bool; is_full_sdm::Bool
end

_flatten_terms(t::StatsModels.MatrixTerm) = collect(StatsModels.AbstractTerm, t.terms)
_flatten_terms(t::Tuple) = collect(StatsModels.AbstractTerm, t)
_flatten_terms(t::AbstractVector) = collect(StatsModels.AbstractTerm, t)
_flatten_terms(t::StatsModels.AbstractTerm) = StatsModels.AbstractTerm[t]

function _term_matrix_and_names(terms::Vector{StatsModels.AbstractTerm}, cols, ::Type{T}) where {T<:AbstractFloat}
    n_rows = length(first(cols))
    isempty(terms) && return zeros(T, n_rows, 0), String[]
    cols_list = Matrix{T}[]; names_list = String[]
    for term in terms
        mc = StatsModels.modelcols(term, cols)
        push!(cols_list, mc isa AbstractVector ? reshape(Vector{T}(mc), n_rows, 1) : Matrix{T}(mc))
        cn = StatsModels.coefnames(term)
        append!(names_list, cn isa AbstractString ? [String(cn)] : String.(cn))
    end
    return hcat(cols_list...), names_list
end

function parse_spatial_formula(f::StatsModels.FormulaTerm, table; perm::Union{Nothing, Vector{Int}}=nothing, T::Type{<:AbstractFloat}=Float64)
    cols_raw = Tables.columntable(table)
    cols = isnothing(perm) ? cols_raw : NamedTuple{keys(cols_raw)}(Tuple(col[perm] for col in values(cols_raw)))
    sch = StatsModels.schema(f, cols)
    f_applied = StatsModels.apply_schema(f, sch, StatsAPI.RegressionModel)

    y_vec = Vector{T}(vec(StatsModels.modelcols(f_applied.lhs, cols)))
    cn_lhs = StatsModels.coefnames(f_applied.lhs)
    lhs_name = cn_lhs isa AbstractString ? String(cn_lhs) : String(first(cn_lhs))

    rhs_terms = _flatten_terms(f_applied.rhs)
    exog_terms = StatsModels.AbstractTerm[]; slag_terms = StatsModels.AbstractTerm[]; endog_terms = StatsModels.AbstractTerm[]; iv_terms = StatsModels.AbstractTerm[]; fe_syms = Symbol[]

    for term in rhs_terms
        if term isa StatsModels.InterceptTerm || term isa StatsModels.ConstantTerm; continue
        elseif term isa SpatialLagTerm; append!(slag_terms, _flatten_terms(term.inner))
        elseif term isa FixedEffectTerm; push!(fe_syms, term.sym)
        elseif term isa EndogenousTerm; append!(endog_terms, _flatten_terms(term.endog)); append!(iv_terms, _flatten_terms(term.instruments))
        else; push!(exog_terms, term); end
    end

    X_exog, exog_names = _term_matrix_and_names(exog_terms, cols, T)
    X_slag, slag_names = _term_matrix_and_names(slag_terms, cols, T)
    X_endog, endog_names = _term_matrix_and_names(endog_terms, cols, T)
    Z_iv, iv_names = _term_matrix_and_names(iv_terms, cols, T)

    has_slag = !isempty(slag_names)
    is_full_sdm = has_slag && (slag_names == exog_names)
    return ParsedSpatialFormula{T}(y_vec, lhs_name, X_exog, exog_names, X_slag, slag_names, X_endog, endog_names, Z_iv, iv_names, fe_syms, has_slag, is_full_sdm)
end

function extract_panel_permutation(table, panel_col::Symbol, time_col::Symbol)
    cols = Tables.columntable(table)
    id_vec = getproperty(cols, panel_col); time_vec = getproperty(cols, time_col); NT = length(id_vec)
    perm = sortperm(1:NT; by = idx -> (time_vec[idx], id_vec[idx]))
    id_sorted = id_vec[perm]; time_sorted = time_vec[perm]
    N = length(unique(id_sorted)); T_per = length(unique(time_sorted))
    N * T_per != NT && throw(ArgumentError("Unbalanced spatial panel detected."))
    return perm, N, T_per
end

function _resolve_static_fe(fe_syms::Vector{Symbol}, panel_col::Symbol, time_col::Symbol)
    has_i = panel_col in fe_syms; has_t = time_col in fe_syms
    return has_i && has_t ? StaticPanel.TwoWayFE() : (has_i ? StaticPanel.IndivFE() : (has_t ? StaticPanel.TimeFE() : StaticPanel.NoFE()))
end

function _resolve_dynamic_fe(fe_syms::Vector{Symbol}, panel_col::Symbol, time_col::Symbol)
    return time_col in fe_syms ? DynamicPanel.TwoWayFE() : DynamicPanel.IndivFE()
end

function _resolve_gs2sls_fe(fe_syms::Vector{Symbol}, panel_col::Symbol, time_col::Symbol)
    return time_col in fe_syms ? GS2SLSPanel.TwoWayFE() : GS2SLSPanel.IndivFE()
end

function CrossSection.fit_spatial(f::StatsModels.FormulaTerm, table, W::AbstractSparseMatrix{T}; method::CrossSection.LogDetMethod=CrossSection.ChebyshevApprox(order=6, n_probes=0), rho_bounds::Tuple{T, T}=(T(-0.995), T(0.995))) where {T<:AbstractFloat}
    parsed = parse_spatial_formula(f, table; T=T)
    if parsed.is_full_sdm
        return CrossSection.fit_spatial(CrossSection.SDM(), W, parsed.X_exog, parsed.y; method=method, varnames=parsed.exog_names, rho_bounds=rho_bounds)
    elseif parsed.has_slag
        X_comb = hcat(parsed.X_exog, W * parsed.X_slag_source)
        return CrossSection.fit_spatial(CrossSection.SAR(), W, X_comb, parsed.y; method=method, varnames=vcat(parsed.exog_names, ["W_" * n for n in parsed.slag_names]), rho_bounds=rho_bounds)
    else
        return CrossSection.fit_spatial(CrossSection.SAR(), W, parsed.X_exog, parsed.y; method=method, varnames=parsed.exog_names, rho_bounds=rho_bounds)
    end
end

function StaticPanel.fit_spatial_panel(f::StatsModels.FormulaTerm, table, W::AbstractSparseMatrix{T}; panel::Symbol, time::Symbol, fe::Union{Nothing, StaticPanel.FixedEffectsSpec}=nothing, approach::StaticPanel.PanelEstimationApproach=StaticPanel.LeeYuTransformation(), method::StaticPanel.LogDetMethod=StaticPanel.ChebyshevApprox(order=6, n_probes=0), rho_bounds::Tuple{T, T}=(T(-0.995), T(0.995))) where {T<:AbstractFloat}
    perm, N, T_per = extract_panel_permutation(table, panel, time)
    parsed = parse_spatial_formula(f, table; perm=perm, T=T)
    fe_spec = isnothing(fe) ? _resolve_static_fe(parsed.fe_symbols, panel, time) : fe
    if parsed.is_full_sdm
        return StaticPanel.fit_spatial_panel(StaticPanel.SDM(), fe_spec, W, parsed.X_exog, parsed.y, N, T_per; approach=approach, method=method, varnames=parsed.exog_names, rho_bounds=rho_bounds)
    elseif parsed.has_slag
        X_comb = hcat(parsed.X_exog, StaticPanel.panel_spatial_lag(W, parsed.X_slag_source, N, T_per))
        return StaticPanel.fit_spatial_panel(StaticPanel.SAR(), fe_spec, W, X_comb, parsed.y, N, T_per; approach=approach, method=method, varnames=vcat(parsed.exog_names, ["W_" * n for n in parsed.slag_names]), rho_bounds=rho_bounds)
    else
        return StaticPanel.fit_spatial_panel(StaticPanel.SAR(), fe_spec, W, parsed.X_exog, parsed.y, N, T_per; approach=approach, method=method, varnames=parsed.exog_names, rho_bounds=rho_bounds)
    end
end

function DynamicPanel.fit_dynamic_spatial_panel(f::StatsModels.FormulaTerm, table, W::AbstractSparseMatrix{T}; panel::Symbol, time::Symbol, fe::Union{Nothing, DynamicPanel.FixedEffectsSpec}=nothing, method::DynamicPanel.LogDetMethod=DynamicPanel.ChebyshevApprox(order=6, n_probes=0), rho_bounds::Tuple{T, T}=(T(-0.995), T(0.995)), n_bias_probes::Int=80, rng=Random.default_rng()) where {T<:AbstractFloat}
    perm, N, T_total = extract_panel_permutation(table, panel, time)
    parsed = parse_spatial_formula(f, table; perm=perm, T=T)
    fe_spec = isnothing(fe) ? _resolve_dynamic_fe(parsed.fe_symbols, panel, time) : fe
    if parsed.is_full_sdm
        return DynamicPanel.fit_dynamic_spatial_panel(DynamicPanel.SDM(), fe_spec, W, parsed.X_exog, parsed.y, N, T_total; method=method, varnames=parsed.exog_names, rho_bounds=rho_bounds, n_bias_probes=n_bias_probes, rng=rng)
    elseif parsed.has_slag
        X_comb = hcat(parsed.X_exog, DynamicPanel.panel_spatial_lag(W, parsed.X_slag_source, N, T_total))
        return DynamicPanel.fit_dynamic_spatial_panel(DynamicPanel.SAR(), fe_spec, W, X_comb, parsed.y, N, T_total; method=method, varnames=vcat(parsed.exog_names, ["W_" * n for n in parsed.slag_names]), rho_bounds=rho_bounds, n_bias_probes=n_bias_probes, rng=rng)
    else
        return DynamicPanel.fit_dynamic_spatial_panel(DynamicPanel.SAR(), fe_spec, W, parsed.X_exog, parsed.y, N, T_total; method=method, varnames=parsed.exog_names, rho_bounds=rho_bounds, n_bias_probes=n_bias_probes, rng=rng)
    end
end

function GS2SLSPanel.fit_gs2sls_panel(f::StatsModels.FormulaTerm, table, W::AbstractSparseMatrix{T}; panel::Symbol, time::Symbol, fe::Union{Nothing, GS2SLSPanel.FixedEffectsSpec}=nothing) where {T<:AbstractFloat}
    perm, N, T_total = extract_panel_permutation(table, panel, time)
    parsed = parse_spatial_formula(f, table; perm=perm, T=T)
    @assert !isempty(parsed.endog_names) "`fit_gs2sls_panel` formula requires `endo(tax, z1 + z2)`."
    fe_spec = isnothing(fe) ? _resolve_gs2sls_fe(parsed.fe_symbols, panel, time) : fe
    X_exog = parsed.has_slag ? hcat(parsed.X_exog, GS2SLSPanel.panel_spatial_lag(W, parsed.X_slag_source, N, T_total)) : parsed.X_exog
    exog_names = parsed.has_slag ? vcat(parsed.exog_names, ["W_" * n for n in parsed.slag_names]) : parsed.exog_names
    return GS2SLSPanel.fit_gs2sls_panel(fe_spec, W, X_exog, parsed.X_endog, parsed.Z_iv, parsed.y, N, T_total; exog_names=exog_names, endog_names=parsed.endog_names)
end

function SystemGMMMod.fit_dynamic_gmm(variant::SystemGMMMod.GMMVariant, f::StatsModels.FormulaTerm, table; panel::Symbol, time::Symbol, max_lag::Int=4, collapsed::Bool=false)
    perm, N, T_tot = extract_panel_permutation(table, panel, time)
    parsed = parse_spatial_formula(f, table; perm=perm, T=Float64)
    K = size(parsed.X_exog, 2)
    return SystemGMMMod.fit_dynamic_gmm(variant, copy(reshape(parsed.y, N, T_tot)), copy(reshape(parsed.X_exog, N, T_tot, K)); max_lag=max_lag, collapsed=collapsed, varnames=parsed.exog_names)
end

# ─── SEM formula dispatch ─────────────────────────────────────────────

function SEMMod.fit_spatial_error(f::StatsModels.FormulaTerm, table, W::AbstractSparseMatrix{T}; panel::Union{Nothing,Symbol}=nothing, time::Union{Nothing,Symbol}=nothing, fe::Union{Nothing, SEMMod.FixedEffectsSpec}=nothing, method::SEMMod.LogDetMethod=SEMMod.ChebyshevApprox(order=6, n_probes=0), lambda_bounds::Tuple{T, T}=(T(-0.995), T(0.995))) where {T<:AbstractFloat}
    if isnothing(panel) || isnothing(time)
        parsed = parse_spatial_formula(f, table; T=T)
        return SEMMod.fit_spatial_error(W, parsed.X_exog, parsed.y; method=method, varnames=parsed.exog_names, lambda_bounds=lambda_bounds)
    else
        perm, N, T_per = extract_panel_permutation(table, panel, time)
        parsed = parse_spatial_formula(f, table; perm=perm, T=T)
        fe_spec = isnothing(fe) ? _resolve_static_fe(parsed.fe_symbols, panel, time) : fe
        return SEMMod.fit_spatial_error(fe_spec, W, parsed.X_exog, parsed.y, N, T_per; method=method, varnames=parsed.exog_names, lambda_bounds=lambda_bounds)
    end
end

# ─── Gravity formula dispatch ─────────────────────────────────────────

function GravityMod.fit_gravity(variant::GravityMod.GravityVariant, f::StatsModels.FormulaTerm, table; origin::Symbol, dest::Symbol, fe::GravityMod.GravityFE=GravityMod.BilateralFE(), maxiter::Int=100, tol::Float64=1e-8)
    cols = Tables.columntable(table)
    sch = StatsModels.schema(f, cols)
    f_applied = StatsModels.apply_schema(f, sch, StatsAPI.RegressionModel)
    y_vec = vec(StatsModels.modelcols(f_applied.lhs, cols))

    rhs_terms = _flatten_terms(f_applied.rhs)
    exog_terms = StatsModels.AbstractTerm[]
    for term in rhs_terms
        term isa StatsModels.InterceptTerm && continue
        term isa StatsModels.ConstantTerm && continue
        push!(exog_terms, term)
    end
    X_exog, exog_names = _term_matrix_and_names(exog_terms, cols, Float64)
    o_ids = Vector{Int}(getproperty(cols, origin))
    d_ids = Vector{Int}(getproperty(cols, dest))
    return GravityMod.fit_gravity(variant, y_vec, X_exog, o_ids, d_ids; fe=fe, varnames=exog_names, maxiter=maxiter, tol=Float64(tol))
end

end # module
