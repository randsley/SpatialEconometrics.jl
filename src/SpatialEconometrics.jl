module SpatialEconometrics

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using ChainRulesCore
using StatsAPI
using StatsModels
using Tables

include("SpatialUtils.jl")
include("SpatialEconometricsPrototype.jl")
include("SpatialPanelEconometrics.jl")
include("DynamicSpatialPanelEconometrics.jl")
include("SpatialGS2SLSPanel.jl")
include("DynamicPanelSystemGMM.jl")
include("StructuralMacroCalibration.jl")
include("SpatialHAC.jl")
include("SpatialDiagnostics.jl")
include("SpatialErrorModel.jl")
include("GWREstimation.jl")
include("SpatialGravityModels.jl")
include("FormulaDSL.jl")
include("SpatialWeights.jl")
include("SpatialModelSelection.jl")
include("SpatialPanelDiagnostics.jl")
include("HigherOrderSpatial.jl")
include("SpatialRegimeModels.jl")
include("SpatialSimulation.jl")
include("SpatialTables.jl")
include("SpatialDiscreteChoice.jl")

const UtilsMod      = SpatialUtils
const CrossSection  = SpatialEconometricsPrototype
const StaticPanel   = SpatialPanelEconometrics
const DynamicPanel  = DynamicSpatialPanelEconometrics
const GS2SLSPanel   = SpatialGS2SLSPanel
const SystemGMMMod  = DynamicPanelSystemGMM
const MacroCalib    = StructuralMacroCalibration
const HACMod        = SpatialHAC
const DiagMod       = SpatialDiagnostics
const SEMMod        = SpatialErrorModel
const GWRMod        = GWREstimation
const GravityMod    = SpatialGravityModels
const DSLMod        = FormulaDSL
const WeightsMod    = SpatialWeights
const SelectionMod  = SpatialModelSelection
const PanelDiagMod  = SpatialPanelDiagnostics
const HigherMod     = HigherOrderSpatial
const RegimeMod     = SpatialRegimeModels
const SimMod        = SpatialSimulation
const TablesMod     = SpatialTables
const DiscreteMod   = SpatialDiscreteChoice

import .SpatialEconometricsPrototype: SpatialRegressionModel, fit_spatial
import .SpatialPanelEconometrics: SpatialPanelModel, LeeYuTransformation, DirectQML, fit_spatial_panel
import .DynamicSpatialPanelEconometrics: DynamicSpatialPanelModel, DynamicSpatialEffectsResult, fit_dynamic_spatial_panel, dynamic_spatial_effects
import .SpatialGS2SLSPanel: GS2SLSModel, SARAR, fit_gs2sls_panel
import .DynamicPanelSystemGMM: GMMVariant, DifferenceGMM, SystemGMM, SystemGMMModel, fit_dynamic_gmm
import .StructuralMacroCalibration: CalibrationResult, balance_gras, balance_gras_full, balance_generalized_ce, sam_leontief_multipliers
import .SpatialHAC: KernelType, BartlettKernel, ParzenKernel, QuadraticSpectralKernel, UniformKernel, DistanceMetric, EuclideanDistance, HaversineDistance, AbstractRobustVCOV, DriscollKraayVCOV, ConleySpatialHAC, vcov_driscoll_kraay, vcov_conley, RobustInferenceTable
import .SpatialDiagnostics: MoranResult, LISAResult, LMTestResult, GearyCResult, moran_global, moran_local, geary_c, lm_test_lag, lm_test_error, lm_test_robust_lag, lm_test_robust_error, lm_tests
import .SpatialErrorModel: SpatialErrorRegressionModel, SpatialErrorPanelModel, fit_spatial_error
import .GWREstimation: GWRModel, MGWRModel, fit_gwr, fit_mgwr, BandwidthType, FixedBandwidth, AdaptiveBandwidth, KernelFunction, GaussianKernel, BisquareKernel, ExponentialKernel
import .SpatialGravityModels: GravityModel, GravityVariant, PoissonGravity, NegBinGravity, GravityFE, OriginFE, DestinationFE, BilateralFE, fit_gravity, multilateral_resistance
import .FormulaDSL: @formula, slag, fe, endo, ParsedSpatialFormula, extract_panel_permutation
import .SpatialWeights: knn_weights, distance_weights, band_weights, block_diagonal_weights, row_standardize, row_standardize!, spectral_standardize, spectral_standardize!, minmax_standardize, minmax_standardize!, SpatialConnectivity, spatial_connectivity, read_gal, write_gal, read_gwt, write_gwt
import .SpatialModelSelection: spatial_aic, spatial_bic, LRTestResult, lr_test, VuongTestResult, vuong_test, CommonFactorTestResult, common_factor_test, WaldTestResult, wald_test, model_comparison_table
import .SpatialPanelDiagnostics: HausmanTestResult, spatial_hausman_test, ChowTestResult, spatial_chow_test, PanelLMTestResult, panel_lm_tests, PanelMoranResult, panel_moran
import .HigherOrderSpatial: SLXModel, fit_slx, SACModel, fit_sac, SDEMModel, fit_sdem
import .SpatialRegimeModels: SpatialRegimeModel, fit_spatial_regime, ExogenousRegimes, skater_regimes, ChowRegimeTest, regime_chow_test
import .SpatialSimulation: SpatialDGP, SAR_DGP, SEM_DGP, SDM_DGP, SLX_DGP, SAC_DGP, SDEM_DGP, simulate_spatial, MonteCarloConfig, monte_carlo_spatial, MonteCarloResult, PowerAnalysisResult, power_analysis
import .SpatialTables: CoefTable, coeftable, summary_stats, to_latex, to_csv, to_markdown, regression_table
import .SpatialDiscreteChoice: SpatialProbitModel, fit_spatial_probit, BayesianSpatialProbitModel, fit_bayesian_spatial_probit, SpatialOrderedProbitModel, fit_spatial_ordered_probit, SpatialProbitEffects, marginal_effects

function robust_vcov(m::SpatialRegressionModel, spec::ConleySpatialHAC)
    Z_eff, bread_inv = SpatialHAC.extract_spatial_jacobian_scores(m.X_aug, m.W, m.coef, m.rho, m.sigma2, m.vcov, length(m.y), 1)
    return vcov_conley(Z_eff, m.residuals, length(m.y), 1, spec; bread_inv=bread_inv)
end

function robust_vcov(m::SpatialPanelModel, spec::DriscollKraayVCOV)
    Z_eff, bread_inv = SpatialHAC.extract_spatial_jacobian_scores(m.X_tilde, m.W, m.coef, m.rho, m.sigma2, m.vcov, m.N, m.T_periods)
    return vcov_driscoll_kraay(Z_eff, m.residuals, m.N, m.T_periods; bread_inv=bread_inv, spec=spec)
end

function robust_vcov(m::SpatialPanelModel, spec::ConleySpatialHAC)
    Z_eff, bread_inv = SpatialHAC.extract_spatial_jacobian_scores(m.X_tilde, m.W, m.coef, m.rho, m.sigma2, m.vcov, m.N, m.T_periods)
    return vcov_conley(Z_eff, m.residuals, m.N, m.T_periods, spec; bread_inv=bread_inv)
end

function robust_vcov(m::DynamicSpatialPanelModel, spec::DriscollKraayVCOV)
    Z_eff, bread_inv = SpatialHAC.extract_spatial_jacobian_scores(m.Z_tilde, m.W, m.coef, m.rho, m.sigma2, m.vcov, m.N, m.T_post)
    return vcov_driscoll_kraay(Z_eff, m.residuals, m.N, m.T_post; bread_inv=bread_inv, spec=spec)
end

function robust_vcov(m::DynamicSpatialPanelModel, spec::ConleySpatialHAC)
    Z_eff, bread_inv = SpatialHAC.extract_spatial_jacobian_scores(m.Z_tilde, m.W, m.coef, m.rho, m.sigma2, m.vcov, m.N, m.T_post)
    return vcov_conley(Z_eff, m.residuals, m.N, m.T_post, spec; bread_inv=bread_inv)
end

function robust_coeftable(m::StatsAPI.RegressionModel, spec::AbstractRobustVCOV)
    V_rob = robust_vcov(m, spec)
    se_base = sqrt.(max.(0.0, diag(vcov(m))))
    return RobustInferenceTable(spec isa DriscollKraayVCOV ? "Driscoll-Kraay SCC" : "Conley HAC", coefnames(m), coef(m), sqrt.(max.(0.0, diag(V_rob))), se_base, coef(m) ./ sqrt.(max.(0.0, diag(V_rob))), V_rob)
end

spatial_effects(m::SpatialRegressionModel; max_power::Int=50, vcov_mat::AbstractMatrix=m.vcov) = SpatialEconometricsPrototype.spatial_effects(m; max_power=max_power, vcov_mat=vcov_mat)
spatial_effects(m::SpatialPanelModel; max_power::Int=50, vcov_mat::AbstractMatrix=m.vcov) = SpatialPanelEconometrics.spatial_effects(m; max_power=max_power, vcov_mat=vcov_mat)
spatial_effects(m::DynamicSpatialPanelModel; max_power::Int=50, vcov_mat::AbstractMatrix=m.vcov) = DynamicSpatialPanelEconometrics.dynamic_spatial_effects(m; max_power=max_power, vcov_mat=vcov_mat)

spatial_effects(m::Union{SpatialRegressionModel, SpatialPanelModel, DynamicSpatialPanelModel}, spec::AbstractRobustVCOV; max_power::Int=50) = spatial_effects(m; max_power=max_power, vcov_mat=robust_vcov(m, spec))
dynamic_spatial_effects(m::DynamicSpatialPanelModel, spec::AbstractRobustVCOV; max_power::Int=50) = DynamicSpatialPanelEconometrics.dynamic_spatial_effects(m; max_power=max_power, vcov_mat=robust_vcov(m, spec))

function make_rook_weights(L::Int; T::Type{<:AbstractFloat}=Float64)
    N = L * L; I_idx = Int[]; J_idx = Int[]; V_val = T[]
    for r in 1:L, c in 1:L
        u = (r - 1) * L + c; nb = Int[]
        r > 1 && push!(nb, (r - 2) * L + c); r < L && push!(nb, r * L + c)
        c > 1 && push!(nb, (r - 1) * L + (c - 1)); c < L && push!(nb, (r - 1) * L + (c + 1))
        w = one(T) / length(nb)
        for v in nb; push!(I_idx, u); push!(J_idx, v); push!(V_val, w); end
    end
    return sparse(I_idx, J_idx, V_val, N, N)
end

function make_queen_weights(L::Int; T::Type{<:AbstractFloat}=Float64)
    N = L * L; I_idx = Int[]; J_idx = Int[]; V_val = T[]
    for r in 1:L, c in 1:L
        u = (r - 1) * L + c; nb = Int[]
        for dr in -1:1, dc in -1:1
            (dr == 0 && dc == 0) && continue
            nr, nc = r + dr, c + dc
            1 <= nr <= L && 1 <= nc <= L && push!(nb, (nr - 1) * L + nc)
        end
        w = one(T) / length(nb)
        for v in nb; push!(I_idx, u); push!(J_idx, v); push!(V_val, w); end
    end
    return sparse(I_idx, J_idx, V_val, N, N)
end

export
    SpatialEconometricsPrototype, CrossSection,
    SpatialPanelEconometrics, StaticPanel,
    DynamicSpatialPanelEconometrics, DynamicPanel,
    SpatialGS2SLSPanel, GS2SLSPanel,
    DynamicPanelSystemGMM, SystemGMMMod,
    StructuralMacroCalibration, MacroCalib,
    SpatialHAC, HACMod,
    SpatialDiagnostics, DiagMod,
    SpatialErrorModel, SEMMod,
    GWREstimation, GWRMod,
    SpatialGravityModels, GravityMod,
    FormulaDSL, DSLMod,
    SpatialWeights, WeightsMod,
    SpatialModelSelection, SelectionMod,
    SpatialPanelDiagnostics, PanelDiagMod,

    SpatialRegressionModel, SpatialPanelModel, DynamicSpatialPanelModel, DynamicSpatialEffectsResult, GS2SLSModel, SystemGMMModel, CalibrationResult, SpatialErrorRegressionModel, SpatialErrorPanelModel,
    LeeYuTransformation, DirectQML, SARAR, GMMVariant, DifferenceGMM, SystemGMM,
    KernelType, BartlettKernel, ParzenKernel, QuadraticSpectralKernel, UniformKernel,
    DistanceMetric, EuclideanDistance, HaversineDistance,
    AbstractRobustVCOV, DriscollKraayVCOV, ConleySpatialHAC,
    vcov_driscoll_kraay, vcov_conley, robust_vcov, RobustInferenceTable, robust_coeftable,
    @formula, slag, fe, endo, ParsedSpatialFormula, extract_panel_permutation,
    fit_spatial, fit_spatial_panel, fit_dynamic_spatial_panel, fit_gs2sls_panel, fit_dynamic_gmm, fit_spatial_error, fit_gwr, fit_mgwr,
    GWRModel, MGWRModel, BandwidthType, FixedBandwidth, AdaptiveBandwidth, KernelFunction, GaussianKernel, BisquareKernel, ExponentialKernel,
    GravityModel, GravityVariant, PoissonGravity, NegBinGravity, GravityFE, OriginFE, DestinationFE, BilateralFE, fit_gravity, multilateral_resistance,
    spatial_effects, dynamic_spatial_effects,
    balance_gras, balance_gras_full, balance_generalized_ce, sam_leontief_multipliers,
    MoranResult, LISAResult, LMTestResult, GearyCResult,
    moran_global, moran_local, geary_c,
    lm_test_lag, lm_test_error, lm_test_robust_lag, lm_test_robust_error, lm_tests,
    make_rook_weights, make_queen_weights,
    # SpatialWeights
    knn_weights, distance_weights, band_weights, block_diagonal_weights,
    row_standardize, row_standardize!, spectral_standardize, spectral_standardize!, minmax_standardize, minmax_standardize!,
    SpatialConnectivity, spatial_connectivity,
    read_gal, write_gal, read_gwt, write_gwt,
    # SpatialModelSelection
    spatial_aic, spatial_bic, LRTestResult, lr_test, VuongTestResult, vuong_test,
    CommonFactorTestResult, common_factor_test, WaldTestResult, wald_test, model_comparison_table,
    # SpatialPanelDiagnostics
    HausmanTestResult, spatial_hausman_test, ChowTestResult, spatial_chow_test,
    PanelLMTestResult, panel_lm_tests, PanelMoranResult, panel_moran,
    # HigherOrderSpatial
    HigherOrderSpatial, HigherMod,
    SLXModel, fit_slx, SACModel, fit_sac, SDEMModel, fit_sdem,
    # SpatialRegimeModels
    SpatialRegimeModels, RegimeMod,
    SpatialRegimeModel, fit_spatial_regime, ExogenousRegimes, skater_regimes,
    ChowRegimeTest, regime_chow_test,
    # SpatialSimulation
    SpatialSimulation, SimMod,
    SpatialDGP, SAR_DGP, SEM_DGP, SDM_DGP, SLX_DGP, SAC_DGP, SDEM_DGP,
    simulate_spatial, MonteCarloConfig, monte_carlo_spatial,
    MonteCarloResult, PowerAnalysisResult, power_analysis,
    # SpatialTables
    SpatialTables, TablesMod,
    CoefTable, coeftable, summary_stats,
    to_latex, to_csv, to_markdown, regression_table,
    # SpatialDiscreteChoice
    SpatialDiscreteChoice, DiscreteMod,
    SpatialProbitModel, fit_spatial_probit,
    BayesianSpatialProbitModel, fit_bayesian_spatial_probit,
    SpatialOrderedProbitModel, fit_spatial_ordered_probit,
    SpatialProbitEffects, marginal_effects

end # module
