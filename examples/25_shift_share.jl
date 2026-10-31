# ---------------------------------------------------------------------------
# 25. Spatial shift-share (Bartik) diagnostics
#
# A shift-share regressor is built as X_i = sum_k W[i,k] * g_k: common
# shocks, region-specific exposure. That construction makes two regions
# with similar industry mixes mechanically correlated, whether or not they
# are anywhere near each other. This script shows what that does to
# inference, and what each diagnostic in the module tells you about it.
#
# Run: julia --project=. examples/25_shift_share.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, Random, Statistics, Printf
import StatsAPI: residuals, coef, stderror

const RNG = MersenneTwister(20_251_031)

# ---------------------------------------------------------------------------
# A design with concentrated exposure, which is the realistic case: a few
# large industries and a long tail of small ones.
# ---------------------------------------------------------------------------

N, K = 300, 60                      # 300 regions, 60 industries

function make_shares(rng, N, K; concentration)
    size_k = exp.(concentration .* randn(rng, K))      # industry sizes
    W = zeros(N, K)
    for i in 1:N
        row = size_k .* (0.2 .+ rand(rng, K))
        W[i, :] = row ./ sum(row)                      # complete shares
    end
    return W
end

W = make_shares(RNG, N, K; concentration = 1.3)
g = randn(RNG, K)                                      # the industry shocks

# Residuals with a sector-mix component: regional outcomes share an
# industry-level disturbance. This is the dependence that matters, and it is
# invisible to geography.
eta = randn(RNG, K) .* (3.0 * sqrt(K))
eps = W * eta .+ randn(RNG, N)

BETA = 2.0
design = shift_share_design(W, g)
y = design.X .* BETA .+ eps

println("="^74)
println("1. How many shocks is this design really using?")
println("="^74)
conc = exposure_concentration(design)
show(stdout, conc)

# ---------------------------------------------------------------------------
# The same estimate, two standard errors.
# ---------------------------------------------------------------------------

println()
println("="^74)
println("2. The same point estimate, two standard errors")
println("="^74)

fit_akm = fit_shift_share(y, design; vcov = :akm)
fit_hc1 = fit_shift_share(y, design; vcov = :hc1)

@printf("true beta              : %.4f\n", BETA)
@printf("estimate               : %.4f   (identical either way: %s)\n",
        fit_akm.beta, fit_akm.beta == fit_hc1.beta)
@printf("AKM standard error     : %.4f\n", fit_akm.se)
@printf("HC1 standard error     : %.4f\n", fit_hc1.se)
@printf("ratio AKM / HC1        : %.2fx\n", fit_akm.se / fit_hc1.se)
println()
println("An HC1 interval would be that much too narrow. Nothing about the")
println("point estimate changes; only what you are entitled to claim.")

# ---------------------------------------------------------------------------
# Which one is right? Resample the shocks and look.
# ---------------------------------------------------------------------------

println()
println("="^74)
println("3. Which standard error is right? Resample the shocks and count")
println("="^74)
println("Shares and residuals are held fixed and only the shocks are redrawn,")
println("which is the framework the AKM estimator is derived in (their")
println("Remark 4). Coverage of a nominal 95% interval over 600 draws:")
println()

reps = 600
cov_akm = cov_hc1 = cov_cl = 0
groups = [argmax(@view W[i, :]) for i in 1:N]
bhats = Float64[]
for _ in 1:reps
    gg = randn(RNG, K)
    d = shift_share_design(W, gg)
    yy = d.X .* BETA .+ eps
    fa = fit_shift_share(yy, d; vcov = :akm)
    fh = fit_shift_share(yy, d; vcov = :hc1)
    spec = SpatialEconometrics.ClusterRobustVCOV(groups)
    secl = sqrt(max(0.0, SpatialEconometrics.vcov_cluster(
        reshape(fa.X_tilde, :, 1), residuals(fa), spec)[1, 1]))
    push!(bhats, fa.beta)
    abs(fa.beta - BETA) <= 1.96 * fa.se && global cov_akm += 1
    abs(fa.beta - BETA) <= 1.96 * fh.se && global cov_hc1 += 1
    abs(fa.beta - BETA) <= 1.96 * secl  && global cov_cl  += 1
end

@printf("  AKM                            : %.3f\n", cov_akm / reps)
@printf("  HC1                            : %.3f\n", cov_hc1 / reps)
@printf("  clustered on largest industry  : %.3f\n", cov_cl / reps)
println()
println("The clustered column is the instructive one. Clustering on each")
println("region's dominant industry is the obvious fix, and it does not work:")
println("the dependence runs through the whole share vector, not through one")
println("industry, so it tracks HC1 rather than repairing it.")
println()
println("Note also that AKM does not reach 0.95 here. That is the warning from")
@printf("step 1 arriving in the numbers: with only %.1f effective shocks this is\n",
        conc.effective_shocks)
println("a small-sample design, and the AKM interval inherits the downward bias")
println("of the cluster-robust formula it generalises. Adao, Kolesar and")
println("Morales propose a null-imposed interval for exactly this case, and it")
println("is not implemented here -- so treat the interval as optimistic rather")
println("than exact when the effective count is low. It remains far better")
println("than either alternative above.")

# ---------------------------------------------------------------------------
# The shock-level view, which is where the information actually lives.
# ---------------------------------------------------------------------------

println()
println("="^74)
println("4. The design has K observations, not N")
println("="^74)

agg = shock_aggregate(design, y)
@printf("regional estimate      : %.10f\n", fit_akm.beta)
@printf("shock-level estimate   : %.10f\n", agg.beta_equiv)
@printf("difference             : %.3e\n", abs(agg.beta_equiv - fit_akm.beta))
println()
println("Not an approximation: Borusyak, Hull and Jaravel's equivalence is an")
println("identity. The regression has $N rows but $K independent pieces of")
@printf("information, and only %.1f of those effectively.\n", conc.effective_shocks)

# ---------------------------------------------------------------------------
# Exogeneity: are the shocks balanced?
# ---------------------------------------------------------------------------

println()
println("="^74)
println("5. Balance: are the shocks unrelated to what came before?")
println("="^74)

# A clean pre-trend, and one deliberately contaminated by the shocks. The
# contamination has to be scaled against the regressor's own dispersion:
# with shares summing to one, X has standard deviation of only about
# sqrt(1/effective shocks), so a unit-variance disturbance would swamp it.
pre_clean = W * (randn(RNG, K) .* sqrt(K)) .+ randn(RNG, N)
pre_dirty = design.X ./ std(design.X) .+ randn(RNG, N) .* 0.3

bal = balance_test(design, hcat(pre_clean, pre_dirty);
                   names = ["clean pre-trend", "contaminated pre-trend"])
show(stdout, bal)
println()
println("The contaminated characteristic is caught decisively. Look at the")
println("clean one too: its p-value sits near 0.05 despite being generated")
println("independently of the shocks. That is not bad luck, it is the")
@printf("over-rejection this test shows at %.1f effective shocks, measured in\n",
        conc.effective_shocks)
println("docs/AUDIT.md at 0.109 against a nominal 0.05 at 20 effective shocks.")
println("So a *pass* here is weaker evidence than it looks, and a marginal")
println("*failure* may be the test rather than the design.")

# ---------------------------------------------------------------------------
# Which industries actually identify the estimate?
# ---------------------------------------------------------------------------

println()
println("="^74)
println("6. Which industries identify the estimate?")
println("="^74)

treatment = design.X .* 0.9 .+ randn(RNG, N) .* 0.5
y_iv = treatment .* BETA .+ eps
rot = rotemberg_weights(y_iv, treatment, design)
show(stdout, rot)
println()
top5 = sum(abs, rot.alpha[rot.order[1:5]]) / sum(abs, rot.alpha)
@printf("The five largest |weights| carry %.1f%% of the total weight mass.\n",
        100 * top5)
println("Those are the industries whose exogeneity has to be defended; the")
println("other $(K - 5) are close to decoration.")

# ---------------------------------------------------------------------------
# And the spatial question, which is not the one that matters here.
# ---------------------------------------------------------------------------

println()
println("="^74)
println("7. Spatial autocorrelation: present, and beside the point")
println("="^74)

side = 10                                  # 100 regions on a lattice
Nsp = side * side
Wsp = make_queen_weights(side)
coords = hcat(repeat(1:side, side), repeat(1:side, inner = side)) .* 1.0

# Industry shares that vary smoothly over space, as real ones do.
Ksp = 8
Wspatial = zeros(Nsp, Ksp)
for k in 1:Ksp
    c = coords[rand(RNG, 1:Nsp), :]
    Wspatial[:, k] = [exp(-0.08 * sum(abs2, coords[i, :] .- c)) for i in 1:Nsp] .+ 0.05
end
Wspatial ./= sum(Wspatial; dims = 2)
dsp = shift_share_design(Wspatial, randn(RNG, Ksp))
ysp = dsp.X .* BETA .+ Wspatial * (randn(RNG, Ksp) .* sqrt(Ksp)) .+ randn(RNG, Nsp)
fsp = fit_shift_share(ysp, dsp)

msp = shift_share_moran(dsp, Wsp; fit = fsp)
show(stdout, msp)
println()
println("The shares are strongly clustered in space, as industry shares are.")
println("That is not a defect of the design and not what threatens inference:")
println("two regions on opposite coasts with the same industry mix are just as")
println("correlated as neighbours. A spatial HAC correction would be answering")
println("the wrong question. The AKM standard error answers the right one.")

println()
println("="^74)
println("Summary")
println("="^74)
@printf("  effective shocks           : %.1f of %d\n", conc.effective_shocks, K)
@printf("  AKM / HC1 standard error   : %.2fx\n", fit_akm.se / fit_hc1.se)
@printf("  coverage, AKM vs HC1       : %.3f vs %.3f\n",
        cov_akm / reps, cov_hc1 / reps)
@printf("  equivalence identity holds : %.3e\n", abs(agg.beta_equiv - fit_akm.beta))
@printf("  Rotemberg weights sum to   : %.12f\n", rot.alpha_sum)
