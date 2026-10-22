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
include("SpatialFiltering.jl")

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
const FilterMod     = SpatialFiltering

import .SpatialUtils: SpatialVariant, SAR, SDM, SARAR,
       FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
       LogDetMethod, ExactLU, ChebyshevApprox
import .SpatialEconometricsPrototype: SpatialRegressionModel, fit_spatial
import .SpatialPanelEconometrics: SpatialPanelModel, LeeYuTransformation, DirectQML, fit_spatial_panel
import .DynamicSpatialPanelEconometrics: DynamicSpatialPanelModel, DynamicSpatialEffectsResult, fit_dynamic_spatial_panel, dynamic_spatial_effects
import .SpatialGS2SLSPanel: GS2SLSModel, fit_gs2sls_panel
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
import .SpatialFiltering: SpatialEigen, spatial_eigenvectors, moran_coefficient, ESFModel, fit_esf, selected_eigenvectors, REESFModel, fit_re_esf

"""
    robust_vcov(model, spec) -> Matrix

Recompute a fitted model's coefficient covariance under a robust
specification.

`spec` selects the estimator: [`DriscollKraayVCOV`](@ref) for spatial
correlation of unrestricted form within each period plus serial
correlation, or [`ConleySpatialHAC`](@ref) for distance-decayed spatial
correlation using a kernel and cutoff.

Both are sandwich estimators. Because a spatial-lag model is estimated by
maximum likelihood rather than least squares, the scores are first mapped
through the Jacobian of the spatial transformation, so the result is a
covariance for the same `coef` vector and can be substituted wherever the
model's own `vcov` is used.

Accepts [`SpatialRegressionModel`](@ref), [`SpatialPanelModel`](@ref) and
[`DynamicSpatialPanelModel`](@ref).

# Examples
```julia
m = fit_spatial_panel(SAR(), W, X, y, N, T; fe = TwoWayFE())
V = robust_vcov(m, DriscollKraayVCOV(lags = 2))

robust_coeftable(m, DriscollKraayVCOV(lags = 2))
spatial_effects(m, DriscollKraayVCOV(lags = 2))
```

See also [`robust_coeftable`](@ref), [`vcov_driscoll_kraay`](@ref),
[`vcov_conley`](@ref).

# References
Driscoll, J. & Kraay, A. (1998). *Review of Economics and Statistics*,
80(4). Conley, T. (1999). *Journal of Econometrics*, 92(1).
"""
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

"""
    robust_coeftable(model, spec) -> RobustInferenceTable

Coefficient table showing robust and model-based standard errors together.

Printing both is the point: the gap between them is the diagnostic. If the
robust errors are much larger, the model's own covariance was understating
uncertainty, usually because of spatial correlation the specification did
not capture.

`spec` is a [`DriscollKraayVCOV`](@ref) or [`ConleySpatialHAC`](@ref); the
covariance itself comes from [`robust_vcov`](@ref).

# Examples
```julia
m = fit_spatial(SAR(), W, X, y; varnames = ["income", "density"])
robust_coeftable(m, ConleySpatialHAC(cutoff = 2.0, coords = coords))
```
"""
function robust_coeftable(m::StatsAPI.RegressionModel, spec::AbstractRobustVCOV)
    V_rob = robust_vcov(m, spec)
    # Qualify these: this module has both `using StatsAPI` and `using StatsModels`,
    # which makes the bare name `vcov` ambiguous, so it resolves to nothing and
    # calling this function raised UndefVarError.
    se_base = sqrt.(max.(0.0, diag(StatsAPI.vcov(m))))
    se_rob = sqrt.(max.(0.0, diag(V_rob)))
    return RobustInferenceTable(
        spec isa DriscollKraayVCOV ? "Driscoll-Kraay SCC" : "Conley HAC",
        StatsAPI.coefnames(m), StatsAPI.coef(m), se_rob, se_base,
        StatsAPI.coef(m) ./ se_rob, V_rob)
end

"""
    spatial_effects(model; max_power = 50, vcov_mat = model.vcov)
    spatial_effects(model, spec::AbstractRobustVCOV; max_power = 50)

Decompose a fitted spatial model into direct, indirect and total marginal
effects, with standard errors.

A coefficient in a spatial-lag model is not a marginal effect: since
`dy/dx' = (I - rho*W)^{-1} (I*beta + W*theta)`, a change in one unit's
covariate feeds back through the whole system. The average diagonal of that
matrix is the **direct** effect, the average off-diagonal row sum the
**indirect** (spillover) effect, and their sum the **total**. Standard
errors are obtained by the delta method.

Works on [`SpatialRegressionModel`](@ref), [`SpatialPanelModel`](@ref) and
[`DynamicSpatialPanelModel`](@ref); for the dynamic case see
[`dynamic_spatial_effects`](@ref), which separates short-run from long-run.

# Keywords
- `max_power`: truncation of the series used for `(I - rho*W)^{-1}`. The
  default is ample for moderate `rho`; values near 1 converge more slowly.
- `vcov_mat`: covariance to propagate. Pass a robust specification as the
  second positional argument instead to use [`robust_vcov`](@ref).

# Returns
A [`SpatialEffectsResult`](@ref) with `direct`, `indirect`, `total` and the
matching `_se` vectors, aligned with `varnames`.

# Examples
```julia
m  = fit_spatial(SDM(), W, X, y; varnames = ["income", "density"])
ef = spatial_effects(m)
ef.indirect ./ ef.indirect_se

spatial_effects(m, ConleySpatialHAC(cutoff = 2.0, coords = coords))
```

# References
LeSage, J. & Pace, R. K. (2009). *Introduction to Spatial Econometrics*,
ch. 2.
"""
spatial_effects(m::SpatialRegressionModel; max_power::Int=50, vcov_mat::AbstractMatrix=m.vcov) = SpatialEconometricsPrototype.spatial_effects(m; max_power=max_power, vcov_mat=vcov_mat)
spatial_effects(m::SpatialPanelModel; max_power::Int=50, vcov_mat::AbstractMatrix=m.vcov) = SpatialPanelEconometrics.spatial_effects(m; max_power=max_power, vcov_mat=vcov_mat)
spatial_effects(m::DynamicSpatialPanelModel; max_power::Int=50, vcov_mat::AbstractMatrix=m.vcov) = DynamicSpatialPanelEconometrics.dynamic_spatial_effects(m; max_power=max_power, vcov_mat=vcov_mat)

spatial_effects(m::Union{SpatialRegressionModel, SpatialPanelModel, DynamicSpatialPanelModel}, spec::AbstractRobustVCOV; max_power::Int=50) = spatial_effects(m; max_power=max_power, vcov_mat=robust_vcov(m, spec))
dynamic_spatial_effects(m::DynamicSpatialPanelModel, spec::AbstractRobustVCOV; max_power::Int=50) = DynamicSpatialPanelEconometrics.dynamic_spatial_effects(m; max_power=max_power, vcov_mat=robust_vcov(m, spec))

"""
    make_rook_weights(L; T = Float64) -> SparseMatrixCSC

Row-standardized rook-contiguity weights for an `L x L` regular lattice.

Two cells are neighbours when they share an edge, so interior cells have
four neighbours, edge cells three and corners two. Units are numbered
row-major, `u = (r - 1) * L + c`, and each row is divided by its own
neighbour count, so every row sums to 1 and `rho` is comparable across
lattice sizes.

Returns an `N x N` sparse matrix with `N = L^2`. Intended for examples,
tests and Monte Carlo work; build weights for real data with
[`knn_weights`](@ref), [`distance_weights`](@ref) or [`read_gal`](@ref).

# Examples
```julia
W = make_rook_weights(10)        # 100 units
sum(W, dims = 2)                 # all ones
```

See also [`make_queen_weights`](@ref).
"""
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

"""
    make_queen_weights(L; T = Float64) -> SparseMatrixCSC

Row-standardized queen-contiguity weights for an `L x L` regular lattice.

Like [`make_rook_weights`](@ref) but counting diagonal neighbours too, so
interior cells have eight neighbours rather than four. Rows are divided by
their neighbour count and sum to 1.

Denser connectivity spreads any given `rho` over more neighbours, so the
two schemes are not interchangeable in a simulation: the implied strength
of dependence per neighbour differs.

# Examples
```julia
W = make_queen_weights(10)
count(!iszero, W) / 100          # average neighbours per unit
```
"""
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


# ─── Extension entry points ──────────────────────────────────────────
#
# A package extension can only add METHODS to functions that already
# exist in the parent; a function it defines itself is unreachable,
# because nothing ever `using`s the extension module. These declarations
# are those functions. The extensions in `ext/` add methods to them.

"""
    AbstractBayesianSpatialResult

Supertype for the result of a Bayesian spatial fit. The concrete type is
defined by the Turing extension, so user code can dispatch on this
without depending on Turing.
"""
abstract type AbstractBayesianSpatialResult end

# Raised when a stub is called and no extension method matched.
function _extension_required(fname::Symbol, pkg::String, ext::Symbol)
    if isnothing(Base.get_extension(@__MODULE__, ext))
        error("$(fname) requires $(pkg): run `using $(pkg)` to load the " *
              "$(ext) extension, then call it again.")
    else
        error("$(fname): the $(ext) extension is loaded, but no method " *
              "matches those argument types.")
    end
end

for (f, pkg, ext) in ((:plot_coefficients,      "Makie",  :SpatialEconometricsMakieExt),
                      (:plot_moran_scatter,     "Makie",  :SpatialEconometricsMakieExt),
                      (:plot_local_moran,       "Makie",  :SpatialEconometricsMakieExt),
                      (:plot_gwr_coefficients,  "Makie",  :SpatialEconometricsMakieExt),
                      (:plot_spatial_effects,   "Makie",  :SpatialEconometricsMakieExt),
                      (:bayesian_sar,           "Turing", :SpatialEconometricsTuringExt),
                      (:bayesian_sdm,           "Turing", :SpatialEconometricsTuringExt),
                      (:bayesian_sem,           "Turing", :SpatialEconometricsTuringExt))
    # The catch-all is deliberately the least specific method, so any
    # typed method the extension adds takes precedence over it.
    @eval begin
        function $f(args...; kwargs...)
            _extension_required($(QuoteNode(f)), $pkg, $(QuoteNode(ext)))
        end
    end
end

@doc """
    plot_coefficients(model; level = 0.95)

Horizontal coefficient plot with confidence intervals. Returns a
`Makie.Figure`.

Requires a Makie backend: run `using CairoMakie` (or `GLMakie`) to activate
the `SpatialEconometricsMakieExt` extension. The critical value is the
normal quantile for `level`; a `level` outside `(0, 1)` raises an
`ArgumentError`.
""" plot_coefficients

@doc """
    plot_moran_scatter(y, W)

Moran scatter plot: standardized `y` against its spatial lag `Wy`, with the
fitted slope drawn through it. Returns a `Makie.Figure`.

When `W` is row-standardized that slope is Moran's I, so it should agree
with `moran_global(y, W).I_stat`.

Requires a Makie backend - see [`plot_coefficients`](@ref).
""" plot_moran_scatter

@doc """
    plot_local_moran(lisa, coords)

LISA cluster map, colouring units by `cluster_type` (`:HH`, `:LL`, `:HL`,
`:LH`, `:ns`). Returns a `Makie.Figure`.

`lisa` is a [`LISAResult`](@ref) from [`moran_local`](@ref) and `coords` an
`N x 2` matrix for the same units; a mismatch raises `DimensionMismatch`
rather than plotting the shorter of the two.

Requires a Makie backend - see [`plot_coefficients`](@ref).
""" plot_local_moran

@doc """
    plot_gwr_coefficients(model, varname)

Map one GWR local coefficient over the estimation coordinates. Returns a
`Makie.Figure`.

`varname` must be one of `model.coefnames`; an unknown name raises an error
listing the names available.

Requires a Makie backend - see [`plot_coefficients`](@ref).
""" plot_gwr_coefficients

@doc """
    plot_spatial_effects(effects)

Grouped bar chart of the direct, indirect and total effects in a
[`SpatialEffectsResult`](@ref). Returns a `Makie.Figure`.

Requires a Makie backend - see [`plot_coefficients`](@ref).
""" plot_spatial_effects

@doc """
    bayesian_sar(W, X, y; n_samples = 1000, n_chains = 2, varnames = nothing, rng = Random.default_rng())

Sample the posterior of a SAR model by NUTS.

Requires Turing: run `using Turing, Distributions` to activate the
`SpatialEconometricsTuringExt` extension. An intercept is added, so `X`
holds covariates only.

Priors are diffuse - `beta ~ MvNormal(0, 10I)`, `rho ~ Uniform(-0.99,
0.99)`, `sigma ~ truncated(Normal(0, 5), 0, Inf)` - and are not
configurable. Because they are diffuse, the posterior mean should land on
the maximum-likelihood estimate for the same data, which is the most useful
check on a fit. Each likelihood evaluation uses the Chebyshev approximation
to the log-determinant, which is what makes thousands of draws affordable.

Returns a `BayesianSpatialResult` (a subtype of
[`AbstractBayesianSpatialResult`](@ref)) carrying `posterior_mean`,
`posterior_std`, `coefnames` and the raw `chain`.

!!! note
    `chain` is not version-stable: Turing returns `MCMCChains.Chains`
    before v0.45 and `FlexiChains.FlexiChain` from v0.45 on, and the two
    index vector-valued parameters differently. Prefer `posterior_mean`
    and `posterior_std`, which are stable across both.

See also [`bayesian_sdm`](@ref), [`bayesian_sem`](@ref),
[`fit_spatial`](@ref).
""" bayesian_sar

@doc """
    bayesian_sdm(W, X, y; n_samples = 1000, n_chains = 2, varnames = nothing, rng = Random.default_rng())

Sample the posterior of a Spatial Durbin model by NUTS.

As [`bayesian_sar`](@ref), but the design is augmented with `WX`, so
`coefnames` carries `W_`-prefixed entries between the slopes and `rho`.

Requires `using Turing, Distributions`.
""" bayesian_sdm

@doc """
    bayesian_sem(W, X, y; n_samples = 1000, n_chains = 2, varnames = nothing, rng = Random.default_rng())

Sample the posterior of a spatial error model by NUTS.

Dependence is placed in the errors rather than in `y`: the likelihood is
built from the filtered residual `(I - lambda*W)(y - X*beta)`, and the
spatial parameter is reported as `lambda`.

Requires `using Turing, Distributions`. See [`bayesian_sar`](@ref) for the
priors and the note about `chain`.
""" bayesian_sem

export
    # Model variants and specification tags, shared by every estimator
    SpatialVariant, SAR, SDM, SARAR,
    FixedEffectsSpec, NoFE, IndivFE, TimeFE, TwoWayFE,
    LogDetMethod, ExactLU, ChebyshevApprox,

    SpatialRegressionModel, SpatialPanelModel, DynamicSpatialPanelModel, DynamicSpatialEffectsResult, GS2SLSModel, SystemGMMModel, CalibrationResult, SpatialErrorRegressionModel, SpatialErrorPanelModel,
    LeeYuTransformation, DirectQML, GMMVariant, DifferenceGMM, SystemGMM,
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
    SLXModel, fit_slx, SACModel, fit_sac, SDEMModel, fit_sdem,
    # SpatialRegimeModels
    SpatialRegimeModel, fit_spatial_regime, ExogenousRegimes, skater_regimes,
    ChowRegimeTest, regime_chow_test,
    # SpatialSimulation
    SpatialDGP, SAR_DGP, SEM_DGP, SDM_DGP, SLX_DGP, SAC_DGP, SDEM_DGP,
    simulate_spatial, MonteCarloConfig, monte_carlo_spatial,
    MonteCarloResult, PowerAnalysisResult, power_analysis,
    # SpatialTables
    CoefTable, coeftable, summary_stats,
    to_latex, to_csv, to_markdown, regression_table,
    # SpatialDiscreteChoice
    SpatialProbitModel, fit_spatial_probit,
    BayesianSpatialProbitModel, fit_bayesian_spatial_probit,
    SpatialOrderedProbitModel, fit_spatial_ordered_probit,
    SpatialProbitEffects, marginal_effects,
    # SpatialFiltering
    SpatialEigen, spatial_eigenvectors, moran_coefficient,
    ESFModel, fit_esf, selected_eigenvectors,
    REESFModel, fit_re_esf,
    # Extension entry points (methods live in ext/)
    AbstractBayesianSpatialResult,
    bayesian_sar, bayesian_sdm, bayesian_sem,
    plot_coefficients, plot_moran_scatter, plot_local_moran,
    plot_gwr_coefficients, plot_spatial_effects

end # module
