# ---------------------------------------------------------------------------
# 27. Weak versus strong cross-sectional dependence
#
# Every panel estimator in this package assumes W governs the
# cross-sectional correlation. When an unobserved common factor drives it
# instead, no weight matrix absorbs it and rho-hat quietly absorbs the
# global shock.
#
# The thing worth knowing is not "use the robust estimator". It is that the
# right estimator depends on the DISTRIBUTION OF THE LOADINGS, and that the
# two published diagnostics both fail on one particular configuration.
#
# Run: julia --project=. examples/27_common_factors.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: coef, stderror, residuals

const RNG = MersenneTwister(20_261_104)

function grid_W(nx, ny)
    n = nx * ny; I_ = Int[]; J_ = Int[]
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

# A spatial panel with a common factor whose loadings we control.
function simulate(rng, W, N, T, rho, beta, loadings)
    Im = Matrix(1.0I, N, N)
    Ainv = inv(Im - rho * Matrix(W))
    X = randn(rng, N * T, length(beta))
    f = randn(rng, T)
    lam = loadings(rng, N)
    y = Vector{Float64}(undef, N * T)
    for t in 1:T
        r = ((t - 1) * N + 1):(t * N)
        y[r] = Ainv * (X[r, :] * beta .+ lam .* f[t] .+ randn(rng, N))
    end
    return y, X
end

W = grid_W(7, 6); N = size(W, 1); T = 30
RHO, BETA = 0.4, [1.0, -0.5]

println("="^74)
println("1. A time effect is a factor with equal loadings")
println("="^74)
println("This is the whole distinction, and it is exact. TimeFE subtracts")
println("the cross-sectional mean at each period, so:")
println()
let SU = SpatialEconometrics.SpatialUtils, n = 30, t = 25
    f = randn(RNG, t)
    hom = vec(ones(n) * transpose(f))
    lam = randn(RNG, n)
    het = vec(lam * transpose(f))
    @printf("  homogeneous loadings, after TimeFE:   max|.| = %.2e\n",
            maximum(abs, SU.within_transform!(copy(hom), n, t, TimeFE())))
    after = SU.within_transform!(copy(het), n, t, TimeFE())
    @printf("  heterogeneous loadings, after TimeFE: max|.| = %.2e\n",
            maximum(abs, after))
    @printf("  share of its variance surviving:      %.3f\n",
            var(after) / var(het))
end
println()
println("So what a time effect can reach is the MEAN of the loadings and")
println("nothing else. Everything below follows from that one fact.")

println()
println("="^74)
println("2. The loading distribution decides the estimator")
println("="^74)
println("Every design here has a genuine common factor. Only the loadings")
println("change. A single draw is far too noisy to read, so each cell below")
println("is a mean over 40 replications; the 200-rep version with RMSEs is")
println("in docs/AUDIT.md.")
println()
const REPS = 40
@printf("%-24s %-10s %-11s %-10s %-10s\n",
        "loadings (bias)", "NoFE", "TwoWayFE", "CCE", "IFE")

designs = [
    ("mean 2 (dominant)", (r, n) -> 2.0 .+ 0.3 .* randn(r, n)),
    ("mean 1",            (r, n) -> 1.0 .+ randn(r, n)),
    ("mean 0, random",    (r, n) -> randn(r, n)),
    ("mean 0, +/-1",      (r, n) -> [i <= n ÷ 2 ? 1.0 : -1.0 for i in 1:n]),
]

for (nm, ld) in designs
    acc = zeros(4); cnt = zeros(Int, 4)
    for _ in 1:REPS
        y, X = simulate(RNG, W, N, T, RHO, BETA, ld)
        fns = (
            () -> fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T;
                      approach = DirectQML(), method = ExactLU()).rho,
            () -> fit_spatial_panel(SAR(), TwoWayFE(), W, X, y, N, T;
                      approach = DirectQML(), method = ExactLU()).rho,
            () -> fit_sp_cce(y, X, W, N, T).rho,
            () -> fit_interactive_fe(y, X, W, N, T; n_factors = 1).rho)
        for (j, f) in enumerate(fns)
            try
                acc[j] += f() - RHO; cnt[j] += 1
            catch
            end
        end
    end
    b = [cnt[j] > 0 ? acc[j] / cnt[j] : NaN for j in 1:4]
    @printf("%-24s %+-10.3f %+-11.3f %+-10.3f %+-10.3f\n",
            nm, b[1], b[2], b[3], b[4])
end
println()
@printf("  (bias in rho-hat; true rho = %.2f, N = %d, T = %d, %d reps)\n",
        RHO, N, T, REPS)
println()
println("These are biases, so zero is right. Two things that are not the")
println("usual advice:")
println()
println("  * On the DOMINANT design TwoWayFE is the best estimator, not the")
println("    weakest. It is correctly specified there, and CCE pays N*q")
println("    projected columns for nothing. Measured over 200 reps its RMSE")
println("    is 0.029 against CCE's 0.047.")
println()
println("  * On the +/-1 design TwoWayFE is WORSE THAN DOING NOTHING --")
println("    measured bias +0.373 against NoFE's +0.329. Removing a")
println("    cross-sectional mean that is already zero destroys variation")
println("    and leaves the factor untouched.")

println()
println("="^74)
println("3. The configuration where both published diagnostics fail")
println("="^74)
println("The +/-1 design is the dangerous one, so you would like a test to")
println("find it. Neither of the two standard ones does.")
println()

let n = 40, t = 60
    f = randn(RNG, t)
    for (nm, lam) in (("homogeneous", ones(n)),
                      ("balanced +/-1",
                       [i <= n ÷ 2 ? 1.0 : -1.0 for i in 1:n]))
        E = lam .* transpose(f) .+ 0.5 .* randn(RNG, n, t)
        cd = pesaran_cd(vec(E), n, t)
        a = try
                @sprintf("%.3f", alpha_exponent(vec(E), n, t).alpha)
            catch
                "REFUSED"
            end
        @printf("  %-15s CD = %+8.2f (p=%.4f)   rho_bar = %+.4f   alpha = %s\n",
                nm, cd.stat, cd.p_value, cd.rho_bar, a)
        @printf("  %-15s mean|corr| = %.4f against an independence benchmark of %.4f\n",
                "", cd.abs_rho_bar, cd.abs_rho_bar_null)
        println()
    end
end
println("On the balanced design the CD statistic is large and NEGATIVE --")
println("it reports a strong common factor as negative average dependence,")
println("because rho_ij is proportional to lam_i lam_j and the signs cancel.")
println("And alpha is refused outright: the cross-sectional average cancels")
println("too, so Var(xbar_t) collapses and the exponent is unidentified.")
println()
println("Measured over 1,500 draws with random mean-zero loadings, CD's")
println("power is 0.348 and DOES NOT IMPROVE WITH N (0.377 at N=10, 0.343")
println("at N=160). The absolute average flags 1.000 of the same draws.")

println()
println("="^74)
println("4. So use the verdict, not either statistic alone")
println("="^74)
let n = 40, t = 60
    f = randn(RNG, t)
    lam = [i <= n ÷ 2 ? 1.0 : -1.0 for i in 1:n]
    E = lam .* transpose(f) .+ 0.5 .* randn(RNG, n, t)
    show(stdout, dependence_verdict(vec(E), n, t))
end
println()
println("`dependence_verdict` reaches the right answer on the design that")
println("defeats both of its inputs, because it checks the absolute average")
println("against its benchmark before trusting the signed statistic.")

println()
println("="^74)
println("5. The exponent, where it is identified")
println("="^74)
println("alpha is the exponent of N giving the number of pairwise-correlated")
println("units. It is identified only for alpha > 1/2, and inference on it")
println("needs alpha > 4/7 -- both thresholds are Bailey, Kapetanios and")
println("Pesaran's, and both are enforced rather than documented.")
println()
let n = 120, t = 100
    for al in (1.0, 0.9, 0.75)
        m = max(1, round(Int, n^al))
        f = randn(RNG, t); lam = zeros(n); lam[1:m] .= 1.0
        Xa = lam .* transpose(f) .+ randn(RNG, n, t)
        r = alpha_exponent(vec(Xa), n, t; kappa2 = 0.5)
        @printf("  true alpha = %.2f (M = %3d units):  alpha-hat = %.3f  [%.3f, %.3f]\n",
                al, m, r.alpha, r.ci[1], r.ci[2])
    end
    println()
    print("  an independent panel: ")
    try
        r = alpha_exponent(vec(randn(RNG, n, t)), n, t)
        @printf("alpha-hat = %.3f  <- should have refused\n", r.alpha)
    catch e
        println("refused, as it must.")
        println("    ", first(split(sprint(showerror, e), ". ")), ".")
    end
end
println()
println("Note the kappa2 = 0.5 above. alpha is estimated FOR A GIVEN kappa2,")
println("and getting it wrong shifts alpha-hat by exactly (1/2)ln(kappa2)/ln(N).")
println("Standardising the series -- which is what makes alpha-hat scale")
println("invariant -- itself sets kappa2 to the signal share, so the default")
println("kappa2 = 1 is a convention, not a measurement.")

println()
println("="^74)
println("Summary")
println("="^74)
println("  * A time effect is a factor with equal loadings, exactly.")
println("  * With dominant one-signed loadings, TwoWayFE beats CCE.")
println("  * With balanced signed loadings, TwoWayFE is worse than nothing")
println("    and only CCE / interactive FE recover rho.")
println("  * That same case defeats the CD test and the alpha exponent;")
println("    the average ABSOLUTE correlation is what detects it.")
println("  * Which is why `dependence_verdict` reports the pair.")
