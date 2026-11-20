# Bibliography

This page lists every work the package cites: in its code, its manual, its
research notes and its roadmap. Each entry says where it is used.

- **Sections.** The works the code or the manual use come first. Then come
  the works used only in the research notes (`research/trace_cfe/`), the
  works consulted for the roadmap, and the citations that do not yet have a
  checked entry.
- **Checks.** Each entry was compared with one of three things: the
  publisher's record (through Scite or the DOI), the title page of the copy
  held in `ExtRef/`, or the reference in `ROADMAP.md` that records how it was
  checked. `ROADMAP.md` also records which parts of each work have been read.
- **Style.** The entries are in APA style. A DOI is given where it was
  checked.
- **Quotes.** A formula quoted from one of these works keeps the work's own
  symbols (see [Notation](@ref)).

## Used by the package

- Abadie, A. (2021). Using synthetic controls: Feasibility, data
  requirements, and methodological aspects. *Journal of Economic
  Literature*, 59(2), 391–425.
  *Used in:* `SpatialCausal.jl`; `models/causal.md`.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control
  methods for comparative case studies: Estimating the effect of
  California's tobacco control program. *Journal of the American
  Statistical Association*, 105(490), 493–505.
  *Used in:* `SpatialCausal.jl`; `models/causal.md`.
- Abadie, A., & L'Hour, J. (2021). A penalized synthetic control estimator
  for disaggregated data. *Journal of the American Statistical
  Association*, 116(536), 1817–1834. https://doi.org/10.1080/01621459.2021.1971535
  *Used in:* `SpatialCausal.jl`; `models/causal.md`.
- Adão, R., Kolesár, M., & Morales, E. (2019). Shift-share designs: Theory
  and inference. *Quarterly Journal of Economics*, 134(4), 1949–2010.
  *Used in:* `SpatialShiftShare.jl`; `models/causal.md`.
- Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and
  polychotomous response data. *Journal of the American Statistical
  Association*, 88(422), 669–679. https://doi.org/10.1080/01621459.1993.10476321
  *Used in:* `SpatialDiscreteChoice.jl`.
- Anderson, J. E., & van Wincoop, E. (2003). Gravity with gravitas: A
  solution to the border puzzle. *American Economic Review*, 93(1),
  170–192. https://doi.org/10.1257/000282803321455214
  *Used in:* `SpatialGravityModels.jl`.
- Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a
  single equation in a complete system of stochastic equations. *Annals of
  Mathematical Statistics*, 20(1), 46–63. https://doi.org/10.1214/aoms/1177730090
  *Used in:* `RootRho.jl`; `models/root_rho.md`; THEORY.md.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation
  consistent covariance matrix estimation. *Econometrica*, 59(3), 817–858.
  https://doi.org/10.2307/2938229
  *Used in:* `SpatialHAC.jl`.
- Anselin, L. (1988). *Spatial econometrics: Methods and models*. Kluwer.
  *Used in:* `SpatialDiagnostics.jl`; `SpatialErrorModel.jl`.
- Anselin, L. (1995). Local indicators of spatial association—LISA.
  *Geographical Analysis*, 27(2), 93–115.
  https://doi.org/10.1111/j.1538-4632.1995.tb00338.x
  *Used in:* `SpatialDiagnostics.jl`.
- Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996). Simple
  diagnostic tests for spatial dependence. *Regional Science and Urban
  Economics*, 26(1), 77–104. https://doi.org/10.1016/0166-0462(95)02111-6
  *Used in:* `SpatialDiagnostics.jl`.
- Anselin, L., & Smirnov, O. (1996). Efficient algorithms for constructing
  proper higher order spatial lag operators. *Journal of Regional Science*,
  36(1), 67–89.
  *Used in:* `SpVAR.jl`; `SpatioTemporalModels.jl`; `models/spvar.md`;
  `models/common_factors.md`.
- Arellano, M., & Bond, S. (1991). Some tests of specification for panel
  data: Monte Carlo evidence and an application to employment equations.
  *Review of Economic Studies*, 58(2), 277–297. https://doi.org/10.2307/2297968
  *Used in:* `DynamicPanelSystemGMM.jl`.
- Arraiz, I., Drukker, D. M., Kelejian, H. H., & Prucha, I. R. (2010). A
  spatial Cliff-Ord-type model with heteroskedastic innovations: Small and
  large sample results. *Journal of Regional Science*, 50(2), 592–614.
  https://doi.org/10.1111/j.1467-9787.2009.00618.x
  *Used in:* `SpatialErrorModel.jl`; `SpatialCausal.jl`.
- Assunção, R. M., Neves, M. C., Câmara, G., & Da Costa Freitas, C. (2006).
  Efficient regionalization techniques for socio-economic geographical units
  using minimum spanning trees. *International Journal of Geographical
  Information Science*, 20(7), 797–811.
  *Used in:* `SpatialRegimeModels.jl`; `models/regime.md`.
- Bai, J. (2009). Panel data models with interactive fixed effects.
  *Econometrica*, 77(4), 1229–1279.
  *Used in:* `SpatioTemporalModels.jl`; `models/common_factors.md`.
- Bai, J., & Chen, J. C. (2026). *Fixed-T dynamic spatial panel model with
  common shocks* (arXiv:2608.22499v1) [Preprint].
  *Used in:* `CommonShockDynamicPanel.jl`;
  `models/common_shock_dynamic_panel.md`.
- Bailey, N., Kapetanios, G., & Pesaran, M. H. (2016). Exponent of
  cross-sectional dependence: Estimation and inference. *Journal of Applied
  Econometrics*, 31(6), 929–960.
  *Used in:* `SpatialCommonFactors.jl`; `models/common_factors.md`.
- Baltagi, B. H., Song, S. H., & Koh, W. (2003). Testing panel data
  regression models with spatial error correlation. *Journal of
  Econometrics*, 117(1), 123–150.
  *Used in:* `diagnostics/panel.md`.
- Bera, A. K., & Yoon, M. J. (1993). Specification testing with locally
  misspecified alternatives. *Econometric Theory*, 9(4), 649–658.
  *Used in:* `SpatialDiagnostics.jl`.
- Blundell, R., & Bond, S. (1998). Initial conditions and moment
  restrictions in dynamic panel data models. *Journal of Econometrics*,
  87(1), 115–143. https://doi.org/10.1016/S0304-4076(98)00009-8
  *Used in:* `DynamicPanelSystemGMM.jl`.
- Borusyak, K., Hull, P., & Jaravel, X. (2022). Quasi-experimental
  shift-share research designs. *Review of Economic Studies*, 89(1),
  181–213.
  *Used in:* `SpatialShiftShare.jl`; `models/causal.md`.
- Bramoullé, Y., Djebbari, H., & Fortin, B. (2009). Identification of peer
  effects through social networks. *Journal of Econometrics*, 150(1),
  41–55.
  *Used in:* `SpatialNetworks.jl`; `models/networks.md`.
- Burridge, P. (1981). Testing for a common factor in a spatial
  autoregression model. *Environment and Planning A*, 13(7), 795–800.
  *Used in:* `SpatialModelSelection.jl`.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to
  cluster-robust inference. *Journal of Human Resources*, 50(2), 317–372.
  https://doi.org/10.3368/jhr.50.2.317
  *Used in:* `SpatialHAC.jl`; `models/causal.md`.
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count
  data* (2nd ed.). Cambridge University Press.
  *Used in:* `SpatialGravityModels.jl`.
- Chagas, A. L. S., Azzoni, C. R., & Almeida, A. N. (2016). A spatial
  difference-in-differences analysis of the impact of sugarcane production
  on respiratory diseases. *Regional Science and Urban Economics*, 59,
  24–36.
  *Used in:* `SpatialCausal.jl`; `models/causal.md`.
- Chun, Y., Griffith, D. A., Lee, M., & Sinha, P. (2016). Eigenvector
  selection with stepwise regression techniques to construct eigenvector
  spatial filters. *Journal of Geographical Systems*, 18(1), 67–85.
  *Used in:* `SpatialFiltering.jl`.
- Conley, T. G. (1999). GMM estimation with cross sectional dependence.
  *Journal of Econometrics*, 92(1), 1–45.
  https://doi.org/10.1016/S0304-4076(98)00084-0
  *Used in:* `SpatialHAC.jl`.
- Debarsy, N., & Le Gallo, J. (2024). *Identification of spatial
  spillovers: Do's and dont's* (LEM Working Paper DT2024-03). Université de
  Lille.
  *Used in:* `SemiparametricSpatial.jl`; `models/semiparametric.md`.
- Delgado, M. S., & Florax, R. J. G. M. (2015). Difference-in-differences
  techniques for spatial data: Local autocorrelation and spatial
  interaction. *Economics Letters*, 137, 123–126.
  https://doi.org/10.1016/j.econlet.2015.10.035
  *Used in:* `SpatialCausal.jl`; `models/causal.md`.
- Di Giacinto, V. (2010). On vector autoregressive modeling in space and
  time. *Journal of Geographical Systems*, 12(2), 125–154.
  https://doi.org/10.1007/s10109-010-0116-6
  *Used in:* `SpVAR.jl`; `models/spvar.md`.
- Di Stefano, R., & Mellace, G. (2024). *The inclusive synthetic control
  method* (arXiv:2403.17624) [Preprint].
  https://doi.org/10.48550/arXiv.2403.17624
  *Used in:* `models/causal.md`.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix
  estimation with spatially dependent panel data. *Review of Economics and
  Statistics*, 80(4), 549–560. https://doi.org/10.1162/003465398557825
  *Used in:* `SpatialHAC.jl`.
- Ertur, C., & Musolesi, A. (2017). Weak and strong cross-sectional
  dependence: A panel data analysis of international technology diffusion.
  *Journal of Applied Econometrics*, 32(3), 477–503.
  *Used in:* `SpatialCommonFactors.jl`; `models/common_factors.md`.
- Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002). *Geographically
  weighted regression*. Wiley.
  *Used in:* `GWREstimation.jl`.
- Fotheringham, A. S., Yang, W., & Kang, W. (2017). Multiscale
  geographically weighted regression (MGWR). *Annals of the American
  Association of Geographers*, 107(6), 1247–1265.
  https://doi.org/10.1080/24694452.2017.1352480
  *Used in:* `GWREstimation.jl`.
- Fülle, M. J., & Otto, P. (2024). Spatial GARCH models for unknown spatial
  locations – an application to financial stock returns. *Spatial Economic
  Analysis*, 19(1), 92–105. https://doi.org/10.1080/17421772.2023.2237067
  *Used in:* `SpatialGARCH.jl`; `SpatialWeights.jl`; `models/spatial_garch.md`.
- Geniaux, G. (2026). *Flexible nonlinear spatial autoregressive models
  under spatial confounding: A gradient boosting approach with closed-form
  estimation* (arXiv:2611.16592v1) [Preprint].
  *Used in:* `SemiparametricSpatial.jl`; `models/semiparametric.md`;
  `research/trace_cfe/`.
- Gibbons, S., & Overman, H. G. (2012). Mostly pointless spatial
  econometrics? *Journal of Regional Science*, 52(2), 172–191.
  *Used in:* `SemiparametricSpatial.jl`; `models/semiparametric.md`.
- Golan, A., Judge, G., & Robinson, S. (1994). Recovering information from
  incomplete or partial multisectoral economic data. *Review of Economics
  and Statistics*, 76(3), 541–549. https://doi.org/10.2307/2109978
  *Used in:* `StructuralMacroCalibration.jl`.
- Goldsmith-Pinkham, P., Sorkin, I., & Swift, H. (2020). Bartik
  instruments: What, when, why, and how. *American Economic Review*,
  110(8), 2586–2624. https://doi.org/10.1257/aer.20181047
  *Used in:* `SpatialShiftShare.jl`; `models/causal.md`.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in
  treatment timing. *Journal of Econometrics*, 225(2), 254–277.
  https://doi.org/10.1016/j.jeconom.2021.03.014
  *Used in:* `models/causal.md`.
- Griffith, D. A. (2003). *Spatial autocorrelation and spatial filtering*.
  Springer.
  *Used in:* `models/filtering.md`.
- Halleck Vega, S., & Elhorst, J. P. (2015). The SLX model. *Journal of
  Regional Science*, 55(3), 339–363.
  *Used in:* `HigherOrderSpatial.jl`; `models/higher_order.md`.
- Jin, F., & Lee, L.-F. (2012). Approximated likelihood and root estimators
  for spatial interaction in spatial autoregressive models. *Regional
  Science and Urban Economics*, 42(3), 446–458.
  https://doi.org/10.1016/j.regsciurbeco.2011.12.004
  *Used in:* `RootRho.jl`; `models/root_rho.md`; `research/trace_cfe/`.
- Jombart, T., Devillard, S., Dufour, A.-B., & Pontier, D. (2008).
  Revealing cryptic spatial patterns in genetic variability by a new
  multivariate method. *Heredity*, 101, 92–103.
  https://doi.org/10.1038/hdy.2008.34
  *Used in:* `SpatialFiltering.jl`.
- Junius, T., & Oosterhaven, J. (2003). The solution of updating or
  regionalizing a matrix with both positive and negative entries.
  *Economic Systems Research*, 15(1), 87–96.
  https://doi.org/10.1080/0953531032000056954
  *Used in:* `StructuralMacroCalibration.jl`.
- Kelejian, H. H., & Prucha, I. R. (1998). A generalized spatial two-stage
  least squares procedure for estimating a spatial autoregressive model
  with autoregressive disturbances. *Journal of Real Estate Finance and
  Economics*, 17(1), 99–121. https://doi.org/10.1023/A:1007707430416
  *Used in:* `SpatialGS2SLSPanel.jl`; `HigherOrderSpatial.jl`;
  `models/higher_order.md`.
- Kelejian, H. H., & Prucha, I. R. (2010). Specification and estimation of
  spatial autoregressive models with autoregressive and heteroskedastic
  disturbances. *Journal of Econometrics*, 157(1), 53–67.
  https://doi.org/10.1016/j.jeconom.2009.10.025
  *Used in:* `SpatialErrorModel.jl`; `SpatialGS2SLSPanel.jl`;
  `SpatialCausal.jl`.
- Klier, T., & McMillen, D. P. (2008). Clustering of auto supplier plants
  in the United States. *Journal of Business & Economic Statistics*, 26(4),
  460–471.
  *Used in:* `SpatialDiscreteChoice.jl`; `models/discrete_choice.md`.
- Koley, M., & Bera, A. K. (2024). To use, or not to use the spatial Durbin
  model? – that is the question. *Spatial Economic Analysis*, 19(1), 30–56.
  https://doi.org/10.1080/17421772.2023.2256810
  *Used in:* `SpatialDiagnostics.jl`.
- Krzyśko, M., Nijkamp, P., Ratajczak, W., Wołyński, W., Wojtyła, A., &
  Wenerska, B. (2023). A novel spatio-temporal principal component analysis
  based on Geary's contiguity ratio. *Computers, Environment and Urban
  Systems*, 103, 101980. https://doi.org/10.1016/j.compenvurbsys.2023.101980
  *Used in:* `SpatialFiltering.jl`.
- Krzyśko, M., Nijkamp, P., Ratajczak, W., Wołyński, W., & Wenerska, B.
  (2024). Spatio-temporal principal component analysis. *Spatial Economic
  Analysis*, 19(1), 8–29. https://doi.org/10.1080/17421772.2023.2237532
  *Used in:* `SpatialFiltering.jl`.
- Kubara, M., & Kopczewska, K. (2024). Akaike information criterion in
  choosing the optimal k-nearest neighbours of the spatial weight matrix.
  *Spatial Economic Analysis*, 19(1), 73–91.
  https://doi.org/10.1080/17421772.2023.2176539
  *Used in:* `SpatialModelSelection.jl`.
- Lee, L.-F. (2004). Asymptotic distributions of quasi-maximum likelihood
  estimators for spatial autoregressive models. *Econometrica*, 72(6),
  1899–1925. https://doi.org/10.1111/j.1468-0262.2004.00558.x
  *Used in:* `SemiparametricSpatial.jl`; THEORY.md.
- Lee, L.-F. (2007). Identification and estimation of econometric models
  with group interactions, contextual factors and fixed effects. *Journal
  of Econometrics*, 140(2), 333–374.
  https://doi.org/10.1016/j.jeconom.2006.07.001
  *Used in:* `SpatialNetworks.jl`.
- Lee, L.-F., & Liu, X. (2010). Efficient GMM estimation of high order
  spatial autoregressive models with autoregressive disturbances.
  *Econometric Theory*, 26(1), 187–230.
  https://doi.org/10.1017/S0266466609090653
  *Used in:* `SpatialNetworks.jl`; `models/networks.md`.
- Lee, L.-F., & Yu, J. (2010). Estimation of spatial autoregressive panel
  data models with fixed effects. *Journal of Econometrics*, 154(2),
  165–185. https://doi.org/10.1016/j.jeconom.2009.08.001
  *Used in:* `SpatialPanelEconometrics.jl`.
- Leenders, R. T. A. J. (2002). Modeling social influence through network
  autocorrelation: Constructing the weight matrix. *Social Networks*,
  24(1), 21–47.
  *Used in:* `SpatialNetworks.jl`; `models/networks.md`.
- LeSage, J. P. (2014). Spatial econometric panel data model specification:
  A Bayesian approach. *Spatial Statistics*, 9, 122–145.
  *Used in:* `SpatialModelSelection.jl`.
- LeSage, J. P., & Pace, R. K. (2009). *Introduction to spatial
  econometrics*. CRC Press.
  *Used in:* `SpatialEconometricsPrototype.jl`; `SpatialModelSelection.jl`;
  `SpatioTemporalModels.jl`; `models/higher_order.md`.
- Li, L., & Yang, Z. (2021). Spatial dynamic panel data models with
  correlated random effects. *Journal of Econometrics*, 221(2), 424–454.
  https://doi.org/10.1016/j.jeconom.2020.05.016
  *Used in:* `CREDynamicSpatialPanel.jl`; `models/cre_dynamic_panel.md`;
  `models/common_shock_dynamic_panel.md`.
- Liang, X., Gao, J., & Gong, X. (2022). Semiparametric spatial
  autoregressive panel data model with fixed effects and time-varying
  coefficients. *Journal of Business & Economic Statistics*, 40(4),
  1784–1802. https://doi.org/10.1080/07350015.2021.1979564
  *Used in:* `SemiparametricSpatial.jl`; `models/semiparametric.md`.
- Lütkepohl, H. (2007). *New introduction to multiple time series
  analysis*. Springer.
  *Used in:* `SpVAR.jl`.
- Manski, C. F. (1993). Identification of endogenous social effects: The
  reflection problem. *Review of Economic Studies*, 60(3), 531–542.
  *Used in:* `SpatialNetworks.jl`; `models/networks.md`.
- Moffitt, R. A. (2001). Policy interventions, low-level equilibria, and
  social interactions. In S. N. Durlauf & H. P. Young (Eds.), *Social
  dynamics* (pp. 45–82). MIT Press.
  https://doi.org/10.7551/mitpress/6294.003.0005
  *Used in:* `SpatialNetworks.jl`.
- Mundlak, Y. (1978). On the pooling of time series and cross section data.
  *Econometrica*, 46(1), 69–85. https://doi.org/10.2307/1913646
  *Used in:* `models/cre_dynamic_panel.md`.
- Murakami, D., & Griffith, D. A. (2015). Random effects specifications in
  eigenvector spatial filtering. *Journal of Geographical Systems*, 17(4),
  311–331.
  *Used in:* `models/filtering.md`.
- Ord, J. K. (1975). Estimation methods for models of spatial interaction.
  *Journal of the American Statistical Association*, 70(349), 120–126.
  *Used in:* `SpatialEconometricsPrototype.jl`.
- Pace, R. K., & LeSage, J. P. (2004). Chebyshev approximation of
  log-determinants of spatial weight matrices. *Computational Statistics &
  Data Analysis*, 45(2), 179–196.
  https://doi.org/10.1016/S0167-9473(02)00321-3
  *Used in:* `SpatialUtils.jl`.
- Pesaran, M. H. (2006). Estimation and inference in large heterogeneous
  panels with a multifactor error structure. *Econometrica*, 74(4),
  967–1012.
  *Used in:* `SpatialCommonFactors.jl`; `models/common_factors.md`.
- Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large
  panels. *Econometric Reviews*, 34(6–10), 1089–1117.
  *Used in:* `SpatialCommonFactors.jl`; `models/common_factors.md`.
- Pfeifer, P. E., & Deutsch, S. J. (1980). A three-stage iterative
  procedure for space-time modeling. *Technometrics*, 22(1), 35–47.
  *Used in:* `SpatioTemporalModels.jl`; `models/common_factors.md`.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and
  system GMM in Stata. *Stata Journal*, 9(1), 86–136.
  https://doi.org/10.1177/1536867X0900900106
  *Used in:* `DynamicPanelSystemGMM.jl`.
- Santi, F., Dickson, M. M., Espa, G., Taufer, E., & Mazzitelli, A. (2021).
  Handling spatial dependence under unknown unit locations. *Spatial
  Economic Analysis*, 16(2), 194–216.
  https://doi.org/10.1080/17421772.2020.1769171
  *Used in:* `models/spatial_garch.md`.
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity.
  *Review of Economics and Statistics*, 88(4), 641–658.
  https://doi.org/10.1162/rest.88.4.641
  *Used in:* `SpatialGravityModels.jl`.
- Sato, T., & Matsuda, Y. (2021). Spatial extension of generalized
  autoregressive conditional heteroskedasticity models. *Spatial Economic
  Analysis*, 16(2), 148–160. https://doi.org/10.1080/17421772.2020.1742929
  *Used in:* `SpatialGARCH.jl`; `models/spatial_garch.md`.
- Shi, W., & Lee, L.-F. (2017). Spatial dynamic panel data models with
  interactive fixed effects. *Journal of Econometrics*, 197(2), 323–347.
  *Used in:* `SpatioTemporalModels.jl`; `models/common_factors.md`.
- Smirnov, O. A. (2022). A closed-form consistent estimator for linear
  models with spatial dependence. *Geographical Analysis*, 54(1), 3–31.
  https://doi.org/10.1111/gean.12268
  *Used in:* `SemiparametricSpatial.jl`; `RootRho.jl`;
  `models/semiparametric.md`; `models/root_rho.md`.
- Stock, J. H., & Wright, J. H. (2000). GMM with weak identification.
  *Econometrica*, 68(5), 1055–1096. https://doi.org/10.1111/1468-0262.00151
  *Used in:* `models/root_rho.md`; THEORY.md.
- Su, L., & Jin, S. (2010). Profile quasi-maximum likelihood estimation of
  partially linear spatial autoregressive models. *Journal of
  Econometrics*, 157(1), 18–33. https://doi.org/10.1016/j.jeconom.2009.10.033
  *Used in:* `SemiparametricSpatial.jl`; `models/semiparametric.md`.
- Sutton, J., & Sutton, J. (2024). A road map to capture the spatial
  dependence underlying regions' economic resilience. *Spatial Economic
  Analysis*, 19(1), 57–72. https://doi.org/10.1080/17421772.2023.2240405
  *Used in:* `SpatialModelSelection.jl`.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
  non-nested hypotheses. *Econometrica*, 57(2), 307–333.
  *Used in:* `SpatialModelSelection.jl`; `diagnostics/model_selection.md`.
- Wang, C., Kockelman, K. M., & Wang, X. (2013). The spatial dimensions of
  land use change dynamics in the Austin, Texas Metropolitan Area.
  *Transportation Research Record*, 2397(1), 131–142.
  *Used in:* `SpatialDiscreteChoice.jl`; `models/discrete_choice.md`.
- Windmeijer, F. (2005). A finite sample correction for the variance of
  linear efficient two-step GMM estimators. *Journal of Econometrics*,
  126(1), 25–51. https://doi.org/10.1016/j.jeconom.2004.02.005
  *Used in:* `DynamicPanelSystemGMM.jl`.
- Yang, C. F. (2021). Common factors and spatial dependence: An application
  to US house prices. *Econometric Reviews*, 40(1), 14–50.
  *Used in:* `SpatialCommonFactors.jl`; `models/common_factors.md`.
- Yu, H., Fotheringham, A. S., Li, Z., Oshan, T., Kang, W., & Wolf, L. J.
  (2020). Inference in multiscale geographically weighted regression.
  *Geographical Analysis*, 52(1), 87–106. https://doi.org/10.1111/gean.12189
  *Used in:* `GWREstimation.jl`.
- Yu, J., de Jong, R., & Lee, L.-F. (2008). Quasi-maximum likelihood
  estimators for spatial dynamic panel data with fixed effects when both
  n and T are large. *Journal of Econometrics*, 146(1), 118–134.
  https://doi.org/10.1016/j.jeconom.2008.08.002
  *Used in:* `DynamicSpatialPanelEconometrics.jl`.

## Used in the research notes

These works support the theory in `research/trace_cfe/THEORY.md` and the
notes beside it. `root_rho` rests on that theory.

- Bhansali, R. J., Giraitis, L., & Kokoszka, P. S. (2007). Convergence of
  quadratic forms with nonvanishing diagonal. *Statistics & Probability
  Letters*, 77(7), 726–734. https://doi.org/10.1016/j.spl.2006.11.007
- Cattaneo, M. D., Jansson, M., & Newey, W. K. (2018). Inference in linear
  regression models with many covariates and heteroscedasticity. *Journal
  of the American Statistical Association*, 113(523), 1350–1361.
  https://doi.org/10.1080/01621459.2017.1328360
- Chen, S., Song, X., & Yu, J. (2025). Oracally efficient estimation and
  specification testing of partially linear additive spatial autoregressive
  models. *Econometric Reviews*, 44(8), 1120–1143.
  https://doi.org/10.1080/07474938.2025.2486989
- de Jong, P. (1987). A central limit theorem for generalized quadratic
  forms. *Probability Theory and Related Fields*, 75, 261–277.
  https://doi.org/10.1007/BF00354037
- Gohberg, I. C., & Krein, M. G. (1961). On the theory of triangular
  representations of nonselfadjoint operators. *Doklady Akademii Nauk
  SSSR*, 137(5), 1034–1037. (In Russian; Math-Net.Ru dan24853)
- Hall, P., & Heyde, C. C. (1980). *Martingale limit theory and its
  application*. Academic Press. https://doi.org/10.1016/C2013-0-10818-5
- Kelejian, H. H., & Prucha, I. R. (2001). On the asymptotic distribution
  of the Moran I test statistic with applications. *Journal of
  Econometrics*, 104(2), 219–257. https://doi.org/10.1016/S0304-4076(01)00064-1
- Kline, P., Saggio, R., & Sølvsten, M. (2020). Leave-out estimation of
  variance components. *Econometrica*, 88(5), 1859–1898.
  https://doi.org/10.3982/ECTA16410
- Kwapień, S., & Pełczyński, A. (1970). The main triangle projection in
  matrix spaces and its applications. *Studia Mathematica*, 34, 43–67.
  https://doi.org/10.4064/sm-34-1-43-67
- Lee, J. (2026). [Title not recorded] (arXiv:2608.22706v1) [Preprint].
  Cited for buffered spatial cross-fitting (§2.8 and §3).
- Lee, L.-F. (2007). GMM and 2SLS estimation of mixed regressive, spatial
  autoregressive models. *Journal of Econometrics*, 137(2), 489–514.
  https://doi.org/10.1016/j.jeconom.2005.10.004
- Lee, L.-F. (2023). Spatial econometric models: Model specifications and
  basic statistics. In *Spatial econometrics: Spatial autoregressive
  models* (pp. 1–56). World Scientific.
  https://doi.org/10.1142/9789811270499_0001
- Liu, S. F., & Yang, Z. (2015). Modified QML estimation of spatial
  autoregressive models with unknown heteroskedasticity and nonnormality.
  *Regional Science and Urban Economics*.
  https://doi.org/10.1016/j.regsciurbeco.2015.02.003
- Lu, F., Tian, G., & Yang, J. (2024). GMM estimation and variable
  selection of partially linear additive spatial autoregressive model.
  *Statistical Papers*, 65, 2253–2288.
  https://doi.org/10.1007/s00362-023-01481-6
- Matsaev, V. I. (1961a). A class of completely continuous operators.
  *Doklady Akademii Nauk SSSR*, 139(3), 548–551. (In Russian; Math-Net.Ru
  dan25286)
- Matsaev, V. I. (1961b). Volterra operators obtained from self-adjoint
  operators by perturbation. *Doklady Akademii Nauk SSSR*, 139(4),
  810–813. (In Russian; Math-Net.Ru dan25330)
- Rosenthal, H. P. (1970). On the subspaces of Lᵖ (p > 2) spanned by
  sequences of independent random variables. *Israel Journal of
  Mathematics*, 8(3), 273–303. https://doi.org/10.1007/BF02771562
- Su, L. (2012). Semiparametric GMM estimation of spatial autoregressive
  models. *Journal of Econometrics*, 167(2), 543–560.
  https://doi.org/10.1016/j.jeconom.2011.09.034
- von Bahr, B., & Esseen, C.-G. (1965). Inequalities for the rth absolute
  moment of a sum of random variables, 1 ≤ r ≤ 2. *Annals of Mathematical
  Statistics*, 36(1), 299–303. https://doi.org/10.1214/aoms/1177700291

## Consulted for the roadmap

These works are cited in `ROADMAP.md` for features that are planned,
assessed or rejected. No code uses them yet.

- Ando, T., & Lu, L. (2019). *A spatial panel quantile model with
  unobserved heterogeneity* (SSRN 3516306) [Working paper].
- Anselin, L. (1990). Spatial dependence and spatial structural instability
  in applied regression analysis. *Journal of Regional Science*, 30(2),
  185–207.
- Anselin, L. (2001). Rao's score test in spatial econometrics. *Journal of
  Statistical Planning and Inference*, 97(1), 113–139.
- Anselin, L., Le Gallo, J., & Jayet, H. (2008). Spatial panel
  econometrics. In *The econometrics of panel data* (pp. 625–660).
  Springer.
- Aune, E., Simpson, D. P., & Eidsvik, J. (2014). Parameter estimation in
  high dimensional Gaussian distributions. *Statistics and Computing*,
  24(2), 247–263.
- Barry, R. P., & Pace, R. K. (1999). Monte Carlo estimates of the log
  determinant of large sparse matrices. *Linear Algebra and its
  Applications*, 289(1–3), 41–54.
- Bivand, R. (2010). *Computing the Jacobian in spatial models: An applied
  survey* (NHH Discussion Paper SAM 20/2010). SSRN 1680467.
- Burnham, K. P., & Anderson, D. R. (2002). *Model selection and multimodel
  inference* (2nd ed.). Springer.
- Debarsy, N., & LeSage, J. P. (2020). Bayesian model averaging for spatial
  autoregressive models based on convex combinations of different types of
  connectivity matrices. *Journal of Business & Economic Statistics*.
  https://doi.org/10.1080/07350015.2020.1840993
- Elhorst, J. P. (2010). Applied spatial econometrics: Raising the bar.
  *Spatial Economic Analysis*, 5(1), 9–28.
- Geniaux, G., & Martinetti, D. (2017). A new method for dealing with
  spatially varying coefficients. *Geographical Analysis*, 50(1), 28–50.
- Hastie, T., & Tibshirani, R. (1993). Varying-coefficient models. *Journal
  of the Royal Statistical Society B*, 55(4), 757–796.
- Kelejian, H. H., & Piras, G. (2014). Estimation of spatial models with
  endogenous weighting matrices, and an application to a demand model for
  cigarettes. *Regional Science and Urban Economics*, 46, 140–149.
- Kopczewska, K., & Elhorst, J. P. (2024). New developments in spatial
  econometric modelling. *Spatial Economic Analysis*, 19(1), 1–7.
  https://doi.org/10.1080/17421772.2023.2281173
- Kostov, P. (2009). A spatial quantile regression hedonic model of
  agricultural land prices. *Spatial Economic Analysis*, 4(1), 53–72.
- Krisztin, T., & Piribauer, P. (2021). *A Bayesian approach for estimation
  of weight matrices in spatial autoregressive models* (arXiv:2101.11938v2)
  [Working paper].
- Lam, C., & Souza, P. C. L. (2020). Estimation and selection of spatial
  weight matrix in a spatial lag model. *Journal of Business & Economic
  Statistics*, 38(3), 693–710. https://doi.org/10.1080/07350015.2019.1569526
- Lee, L.-F., & Yu, J. (2012). QML estimation of spatial dynamic panel data
  models with time varying spatial weights matrices. *Spatial Economic
  Analysis*, 7(1), 31–74.
- LeSage, J. P. (2015). Software for Bayesian cross section and panel
  spatial model comparison. *Journal of Geographical Systems*, 17(4),
  297–310.
- Liu, X., & Yang, Z. (2020). Variable selection in spatial autoregressive
  models. *Journal of Econometrics*, 218(1), 85–109.
- Mutl, J., & Pfaffermayr, M. (2011). The Hausman test in a Cliff and Ord
  panel model. *Econometrics Journal*, 14(1), 48–76.
- Qu, X., & Lee, L.-F. (2015). Estimating a spatial autoregressive model
  with an endogenous spatial weight matrix. *Journal of Econometrics*,
  184(2), 209–232.
- Walde, J., Larch, M., & Tappeiner, G. (2008). Performance contest between
  MLE and GMM for huge spatial autoregressive models. *Journal of
  Statistical Computation and Simulation*, 78(2), 151–166.

## Cited without a checked entry

The package names these works, but no entry for them has been checked.
Each needs its record checked before it gets a full entry.

| Citation | Where | What is missing |
|---|---|---|
| Anselin (2005) | `diagnostics/spatial.md` (the LM decision rule) | the work: probably the GeoDa workbook |
| Anselin & Rey (2014) | `README.md` (the `SpatialWeights` row) | the full entry: probably *Modern Spatial Econometrics in Practice* |
| Gänssler & Stute (1977) | THEORY.md, through Kelejian & Prucha (2001) | the full entry; Scite has no record |
| Górecki et al. (2018) | `SpatialFiltering.jl` | the full entry; Scite matched a different paper |
| Hamilton (1994) | `SpVAR.jl` (p. 332) | the full entry: *Time Series Analysis* |
| Li, K. (2022) | `ROADMAP.md` | the publisher record; MPRA Paper 113568 is held in `ExtRef/`, and Scite has no record |
| Li, L., Meng, X., & Yang, Z. (2026) | `CREDynamicSpatialPanel.jl`, `models/cre_dynamic_panel.md` | the volume and pages: it is accepted at *Spatial Economic Analysis*; Scite matched a different work |
| Sakhnovich | THEORY.md, through Matsaev (1961b) | the work Matsaev credits for `p = 2` |
