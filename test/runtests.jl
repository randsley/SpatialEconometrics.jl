using Test
using SpatialEconometrics
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using ForwardDiff
using ChainRulesCore
using StatsAPI
import StatsAPI: coef, vcov, stderror, nobs, residuals, coefnames, loglikelihood, dof_residual, predict, fitted
using StatsModels
using Tables
using NearestNeighbors
using Distributions

# A stand-in fit, so spatial_hausman_test can be handed chosen coefficient
# vectors and covariance matrices rather than whatever two panel fits happen
# to produce. Structs must be defined at top level, hence not in-testset.
# Estimators for testset 17b. SimShortMock reports no spatial parameter, so
# monte_carlo_spatial has fewer coefficients than the DGP has parameters;
# SimFullMock reports one, standing in for a real fit where only the
# DGP-side diagnostics are under test. Structs must be top level.
struct SimShortMock <: StatsAPI.RegressionModel
    b::Vector{Float64}
end
coef(m::SimShortMock) = m.b
coefnames(m::SimShortMock) = ["(Intercept)", "x1", "x2"]

struct SimFullMock <: StatsAPI.RegressionModel
    b::Vector{Float64}
end
coef(m::SimFullMock) = m.b
coefnames(m::SimFullMock) = ["(Intercept)", "x1", "x2", "ρ"]

struct HausMock <: StatsAPI.RegressionModel
    names::Vector{String}
    b::Vector{Float64}
    V::Matrix{Float64}
end
coefnames(m::HausMock) = m.names
coef(m::HausMock) = m.b
vcov(m::HausMock) = m.V

# Progress reporting for CI, off unless SE_TEST_PROGRESS=1. The Julia 1 job on
# ubuntu-latest has three times lost its runner at 63-66 minutes, which leaves
# no log, against a healthy 14-35 minutes. These lines say which testset was
# running and how memory stood when it happened: one line as each top-level
# testset starts, and a heartbeat every 60 s from the interactive thread
# (Julia 1.13 starts one by default; on 1.10, with none, only the per-testset
# lines appear). The output goes to stdout and is flushed at once.
const _PROGRESS = get(ENV, "SE_TEST_PROGRESS", "0") == "1"
const _PROGRESS_T0 = time()
const _PROGRESS_NOW = Ref("(before the first testset)")
function _progress_line(tag)
    gib(x) = round(x / 2^30, digits = 2)
    mins = round((time() - _PROGRESS_T0) / 60, digits = 1)
    println("[progress] +$(mins) min  $tag  maxrss $(gib(Sys.maxrss())) GiB  free $(gib(Sys.free_memory())) of $(gib(Sys.total_memory())) GiB")
    flush(stdout)
end
function _progress(name)
    _PROGRESS || return
    _PROGRESS_NOW[] = name
    _progress_line("start: " * name)
end
if _PROGRESS && Threads.nthreads(:interactive) > 0
    Threads.@spawn :interactive while true
        sleep(60)
        _progress_line("heartbeat, in: " * _PROGRESS_NOW[])
    end
end

# The remaining two testsets need the weak dependencies. They are skipped
# when those are not installed, so a plain `Pkg.test()` still runs, and
# they are exercised by the `extensions` CI job, which installs them.
_HAVE_MAKIE  = !isnothing(Base.identify_package("Makie"))
_HAVE_TURING = !isnothing(Base.identify_package("Turing")) &&
               !isnothing(Base.identify_package("Distributions"))

# Row-standardised rook-contiguity grid, used by testsets 38 and 39.
# Defined once here rather than inside each: a grid W is the standard
# panel test fixture and both rows need the same one.
function cf_grid_W(nx, ny)
    n = nx * ny
    I_ = Int[]; J_ = Int[]
    idx(i, j) = (j - 1) * nx + i
    for i in 1:nx, j in 1:ny, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        a, b = i + di, j + dj
        if 1 <= a <= nx && 1 <= b <= ny
            push!(I_, idx(i, j)); push!(J_, idx(a, b))
        end
    end
    A = sparse(I_, J_, ones(length(I_)), n, n)
    return sparse(A ./ sum(A, dims = 2))
end

# Each top-level testset is its own file under testsets/, run in this order.
# They used to sit inline in this one block, about 11,000 lines, and Julia
# lowers and compiles a top-level expression as a whole before running any
# of it: on 2026-11-19 that took 32 minutes and a 16 GiB peak before the
# first test ran (Julia 1.13.4), against 8 minutes for the tests themselves.
# That is past the 16 GB of an ubuntu-latest runner. `include` evaluates
# each file's testset separately, so only one is compiled at a time.
const _TESTSET_FILES = [
    "01_cross_sectional_sar_sdm.jl",
    "02_static_spatial_panels.jl",
    "03_dynamic_spatial_panels.jl",
    "04_gs2sls_sarar_panel.jl",
    "04b_gs2sls_moments_dynamic_jacobian_panel.jl",
    "04c_gs2sls_first_stage_correction_finding.jl",
    "04d_heteroskedasticity_robust_gs2sls_errors_heteroskedastic.jl",
    "05_system_gmm_arellano_bond_blundell.jl",
    "05b_dynamic_gmm_windmeijer_assembly_and.jl",
    "06_gras_cross_entropy_calibration.jl",
    "07_driscoll_kraay_conley_hac.jl",
    "08_formuladsl_jl_formula_slag_fe.jl",
    "09_spatial_diagnostics_moran_lm_geary.jl",
    "10_spatial_error_model_sem.jl",
    "11_gwr_mgwr.jl",
    "11b_bandwidth_selection_back_fitting_convergence.jl",
    "11c_mgwr_degrees_of_freedom_against.jl",
    "11d_gravity_fe_sweeping_and_nb.jl",
    "12_spatial_weights_construction.jl",
    "13_model_selection_comparison.jl",
    "13b_log_likelihoods_against_an_independent.jl",
    "13c_vuong_omega_is_a_variance.jl",
    "13d_trace_based_variances_are_exact.jl",
    "14_spatial_panel_diagnostics.jl",
    "14b_panel_diagnostics_scale_degrees_of.jl",
    "15_higher_order_spatial_models_slx.jl",
    "15b_variances_of_slx_sem_sdem.jl",
    "16_spatial_regime_models.jl",
    "17_spatial_simulation_monte_carlo.jl",
    "17b_simulation_dgps_and_monte_carlo.jl",
    "18_spatial_tables_output.jl",
    "18b_table_exporters_escape_and_honour.jl",
    "19_spatial_discrete_choice_probit.jl",
    "19b_probit_gibbs_internals_audit_finding.jl",
    "20_gravity_flow_models_ppml_nb.jl",
    "21_spatial_filtering_esf_re_esf.jl",
    "21b_spatial_filtering_basis_identities_and.jl",
    "21c_esf_glm_families_against_independent.jl",
    "22_regressions_found_by_the_example.jl",
    "23_regressions_found_by_the_full.jl",
    "24_extension_entry_points.jl",
    "25_makie_extension.jl",
    "26_turing_extension.jl",
    "27_durbin_score_tests_koley_bera.jl",
    "28_knn_w_selection_by_aic.jl",
    "29_identification_guards_on_the_design.jl",
    "30_bayesian_specification_comparison.jl",
    "31_spatio_temporal_principal_components.jl",
    "32_performance_invariants.jl",
    "33_gwr_compact_local_design.jl",
    "34_spatial_causal_inference.jl",
    "35_spatial_synthetic_control.jl",
    "36_spatial_shift_share_diagnostics.jl",
    "37_spatial_econometrics_for_networks.jl",
    "38_weak_vs_strong_cross_sectional.jl",
    "39_spatio_temporal_models.jl",
    "40_spatial_var.jl",
    "41_semiparametric_spatial.jl",
    "42_short_t_cre_spatial_dynamic.jl",
    "43_short_t_dynamic_spatial_panel.jl",
    "44_spatial_log_garch_sato_matsuda.jl",
    "45_the_claude_skill_s_function.jl",
    "46_root_rho_jin_lee_resd.jl",
    "47_root_rho_penalised_smoother.jl",
]

@testset "SpatialEconometrics.jl Full Suite Verification" begin
    for file in _TESTSET_FILES
        include(joinpath(@__DIR__, "testsets", file))
    end
end
