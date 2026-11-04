# ---------------------------------------------------------------------------
# 28. Space-time dynamics: STAR, impulse responses, forecasting
#
# `fit_spatial_panel` puts the spatial lag at time t: W y_t, simultaneous,
# requiring a Jacobian. `fit_star` puts every spatial lag one or more
# periods back: W^k y_{t-l}, predetermined, so least squares is consistent
# and there is no Jacobian at all.
#
# NEITHER NESTS THE OTHER, and the choice between them is a claim about the
# world: does a neighbour's shock reach you within the period, or next
# period? This file is about what that choice costs when it is made
# wrongly, and about what the space-time impulse response adds that a
# single long-run multiplier cannot.
#
# Run: julia --project=. examples/28_spatio_temporal.jl
# ---------------------------------------------------------------------------

using SpatialEconometrics
using LinearAlgebra, SparseArrays, Random, Statistics, Printf
import StatsAPI: stderror

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

# A STAR(1;1) panel: y_t = phi0 y_{t-1} + phi1 W y_{t-1} + X_t b + c + e_t.
# Diffusion is entirely across periods; within a period, units are
# independent given the past.
function simulate_star(rng, W, N, T, phi0, phi1, beta, icept; burn = 50)
    A = phi0 .* Matrix(1.0I, N, N) .+ phi1 .* Matrix(W)
    k = length(beta)
    X = randn(rng, N * (T + burn), k)
    y = zeros(N * (T + burn))
    prev = zeros(N)
    for t in 1:(T + burn)
        r = ((t - 1) * N + 1):(t * N)
        cur = A * prev .+ X[r, :] * beta .+ icept .+ randn(rng, N)
        y[r] = cur; prev = cur
    end
    keep = (burn * N + 1):((T + burn) * N)
    return y[keep], X[keep, :]
end

W = grid_W(7, 6); N = size(W, 1); T = 60
PHI0, PHI1, BETA, ICEPT = 0.40, 0.40, [1.0, -0.5], 0.25

println("="^74)
println("1. The two specifications are not interchangeable")
println("="^74)
println("The data below is STAR: all spatial transmission is lagged, and")
println("there is no contemporaneous W y_t in the truth at all.")
println()

y, X = simulate_star(RNG, W, N, T, PHI0, PHI1, BETA, ICEPT)
st = fit_star(y, X, W, N, T; spatial_lags = 1, temporal_lags = 1)

@printf("  fit_star   phi[y(t-1)]  = %+.4f   (true %+.2f)\n",
        st.phi[1, 1], PHI0)
@printf("             phi[Wy(t-1)] = %+.4f   (true %+.2f)\n",
        st.phi[2, 1], PHI1)
@printf("             beta         = %+.4f %+.4f   (true %+.2f %+.2f)\n",
        st.beta[1], st.beta[2], BETA[1], BETA[2])
@printf("             intercept    = %+.4f   (true %+.2f)\n",
        st.intercept, ICEPT)
println()
sar = fit_spatial_panel(SAR(), NoFE(), W, X, y, N, T;
                        approach = DirectQML(), method = ExactLU())
@printf("  fit_spatial_panel reads a CONTEMPORANEOUS rho-hat = %+.4f\n",
        sar.rho)
println()
println("The truth has no within-period spatial feedback whatsoever, yet the")
println("simultaneous model reports one, because lagged diffusion leaves a")
println("cross-sectional pattern in y_t that W y_t can fit. The sign and")
println("size above are what that misreading looks like on this design.")
println()
println("The reverse direction is the same mistake mirrored. The honest")
println("summary is that the lag timing is an ASSUMPTION, and neither fit")
println("can adjudicate it for you.")

println()
println("="^74)
println("2. The cumulative IRF converges on the long-run multiplier")
println("="^74)
println("This is an algebraic identity, so it is worth checking rather than")
println("asserting: sum_h Psi_h -> (I - sum_l A_l)^{-1}.")
println()
println("Note that `psi[h+1]` is always the response AT horizon h; the")
println("`cumulative` keyword accumulates the own/spillover/total summaries,")
println("not the matrices. So the sum is taken here explicitly.")
println()

let A1 = st.phi[1, 1] .* Matrix(1.0I, N, N) .+ st.phi[2, 1] .* Matrix(W),
    lr = inv(Matrix(1.0I, N, N) - A1)

    for h in (5, 10, 40, 120)
        r = space_time_irf(st; horizon = h)
        acc = zeros(N, N)
        for j in 0:h
            acc .+= r.psi[j + 1]
        end
        c = space_time_irf(st; horizon = h, cumulative = true)
        @printf("  horizon %3d:  max|sum Psi - (I-A1)^-1| = %.3e   (total %.6f)\n",
                h, maximum(abs, acc .- lr), c.total[end])
    end
    @printf("\n  long-run own effect     = %.6f\n", mean(diag(lr)))
    @printf("  long-run total per unit = %.6f\n", mean(sum(lr, dims = 2)))
end
println()
println("So the IRF is not an alternative to the multiplier -- it is the")
println("multiplier's path. What it adds is WHEN, which the multiplier")
println("discards.")

println()
println("="^74)
println("3. The spillover peaks later than the own effect")
println("="^74)
println("That timing is the whole reason to look at the path. A shock lands")
println("on the unit itself immediately and reaches its neighbours only")
println("after it has had a period to propagate:")
println()
let r = space_time_irf(st; horizon = 12, cumulative = false)
    @printf("  %-4s %-14s %-14s %-12s\n", "h", "own", "spillover", "spill/own")
    for h in 0:12
        @printf("  %-4d %-14.6f %-14.6f %-12.4f\n", h, r.own[h + 1],
                r.spillover[h + 1],
                r.own[h + 1] == 0 ? NaN : r.spillover[h + 1] / r.own[h + 1])
    end
    po = argmax(r.own) - 1; ps = argmax(r.spillover) - 1
    ratio = r.spillover[2:end] ./ r.own[2:end]
    println()
    @printf("  own effect peaks at h = %d, spillover at h = %d\n", po, ps)
    @printf("  the spillover/own ratio rises monotonically: %s\n",
            all(diff(ratio) .> 0))
end
println()
println("A single number -- direct, indirect, total -- is the h -> infinity")
println("column of that table. For any question with a policy horizon")
println("shorter than convergence, that column is the wrong summary.")

println()
println("="^74)
println("4. Stationarity is a property of the companion matrix")
println("="^74)
println("With L temporal lags the relevant object is the L*N companion")
println("matrix, not any single A_l. `fit_star` computes its spectral")
println("radius and reports it; it does not silently return a divergent")
println("model.")
println()
@printf("  the fit above:  spectral radius = %.4f, stationary = %s\n",
        st.spectral_radius, st.stationary)
println()
let W2 = grid_W(8, 5), N2 = size(W2, 1)
    yb, Xb = simulate_star(RNG, W2, N2, 40, 0.3, 0.3, BETA, ICEPT; burn = 20)
    m2l = fit_star(yb, Xb, W2, N2, 40; temporal_lags = 2)
    @printf("  two temporal lags (L=2, companion is %d x %d):\n",
            2 * N2, 2 * N2)
    @printf("    radius = %.4f, stationary = %s\n",
            m2l.spectral_radius, m2l.stationary)
end
println()
println("  and the boundary itself, for A1 = a*I + b*W:")
for (nm, Wu) in (("row-normalised", Matrix(W)),
                 ("binary, not normalised", Matrix(W) .> 0 |> x -> 1.0 .* x))
    @printf("\n    W %s (eig in [%.3f, %.3f]):\n", nm,
            minimum(real, eigvals(Wu)), maximum(real, eigvals(Wu)))
    for (a, b) in ((0.4, 0.4), (0.5, 0.5), (0.6, 0.5), (0.9, -0.5))
        A = a .* Matrix(1.0I, N, N) .+ b .* Wu
        rad = maximum(abs, eigvals(A))
        @printf("      a = %+.1f, b = %+.1f:  |a|+|b| = %.1f,  radius = %.4f  %s\n",
                a, b, abs(a) + abs(b), rad,
                rad < 1 ? "stationary" : "DIVERGENT")
    end
end
println()
println("For a row-normalised W the leading eigenvalue is exactly 1, so the")
println("familiar |a| + |b| < 1 rule is not merely sufficient there -- it is")
println("the boundary, exactly, and the shortcut costs nothing. Change")
println("nothing but the normalisation and the same coefficients diverge at")
println("|a| + |b| = 0.8, because the spectrum now reaches 3.65.")
println()
println("So the rule is a fact about row-normalisation, not about STAR. That")
println("is why the radius is computed from W's own spectrum; and with")
println("L > 1 there is no coefficient sum to check in the first place --")
println("only the companion matrix has the answer.")

println()
println("="^74)
println("5. Dynamic forecasting, and why y_history is required")
println("="^74)
println("`spatial_forecast` substitutes its own forecasts for realisations")
println("once the data runs out, so it needs the last L periods of y. The")
println("model does not retain them, and omitting them is an error rather")
println("than a default to zero.")
println()

let H = 6
    ytr, Xtr = simulate_star(RNG, W, N, T + H, PHI0, PHI1, BETA, ICEPT)
    ntr = N * T
    m = fit_star(ytr[1:ntr], Xtr[1:ntr, :], W, N, T)
    Xf = Xtr[(ntr + 1):end, :]
    yf = ytr[(ntr + 1):end]
    hist = ytr[(ntr - N + 1):ntr]
    f = vec(spatial_forecast(m, Xf; h = H, y_history = hist))

    # A benchmark that keeps the dynamics but drops the spatial channel.
    m0 = fit_star(ytr[1:ntr], Xtr[1:ntr, :], W, N, T; spatial_lags = 0)
    f0 = vec(spatial_forecast(m0, Xf; h = H, y_history = hist))

    @printf("  %-4s %-16s %-16s\n", "h", "RMSE with W", "RMSE without W")
    for h in 1:H
        r = ((h - 1) * N + 1):(h * N)
        @printf("  %-4d %-16.4f %-16.4f\n", h,
                sqrt(mean(abs2, f[r] .- yf[r])),
                sqrt(mean(abs2, f0[r] .- yf[r])))
    end
    println()
    @printf("  pooled over all %d horizons:  %.4f  vs  %.4f\n", H,
            sqrt(mean(abs2, f .- yf)), sqrt(mean(abs2, f0 .- yf)))
    println()
    print("  omitting y_history: ")
    try
        spatial_forecast(m, Xf; h = H)
        println("returned a forecast  <- should have refused")
    catch e
        println("refused.")
        println("    ", first(split(sprint(showerror, e), " — ")), ".")
    end
end

println()
println("="^74)
println("6. Interactive fixed effects: what is identified, and what is not")
println("="^74)
println("`fit_interactive_fe` estimates an unobserved factor structure")
println("jointly with rho, by Bai's (2009) iterative principal components.")
println("Factors and loadings are identified ONLY UP TO ROTATION, so the")
println("object to compare across fits is the projection, never F itself.")
println()

let Tf = 30, rho = 0.4
    Im = Matrix(1.0I, N, N); Ainv = inv(Im - rho * Matrix(W))
    Xv = randn(RNG, N * Tf, 2); fct = randn(RNG, Tf); lam = randn(RNG, N)
    yv = Vector{Float64}(undef, N * Tf)
    for t in 1:Tf
        r = ((t - 1) * N + 1):(t * N)
        yv[r] = Ainv * (Xv[r, :] * BETA .+ lam .* fct[t] .+ randn(RNG, N))
    end

    a = fit_interactive_fe(yv, Xv, W, N, Tf; n_factors = 1,
                           rng = MersenneTwister(1))
    b = fit_interactive_fe(yv, Xv, W, N, Tf; n_factors = 1, n_starts = 4,
                           rng = MersenneTwister(99))
    @printf("  rho-hat (true %.2f):   %.8f  vs  %.8f\n", rho, a.rho, b.rho)
    @printf("  max|F_a - F_b|:       %.3e   <- NOT identified\n",
            maximum(abs, a.factors .- b.factors))
    @printf("  max|P_a - P_b|:       %.3e   <- identified\n",
            maximum(abs, factor_projection(a) .- factor_projection(b)))
    println()
    let m2 = fit_interactive_fe(yv, Xv, W, N, Tf; n_factors = 1,
                                rng = MersenneTwister(1))
        m2.factors .*= -1
        @printf("  a deliberate sign flip moves P by %.3e\n",
                maximum(abs, factor_projection(m2) .- factor_projection(a)))
    end
    println()
    @printf("  converged = %s after %d iterations\n", a.converged,
            a.iterations)
end
println()
println("The sign flip is the simplest rotation there is, and it leaves the")
println("projection untouched. Any statement about `m.factors` that is not")
println("invariant to it is a statement about the algorithm's starting")
println("value, not about the data.")

println()
println("="^74)
println("Summary")
println("="^74)
println("  * STAR and SAR encode different timing; neither nests the other,")
println("    and fitting the wrong one reports spatial structure the")
println("    data-generating process does not contain.")
println("  * The cumulative IRF IS the long-run multiplier, reached as a")
println("    path -- and the path is the part a multiplier throws away.")
println("  * Spillovers peak after own effects, so a short-horizon policy")
println("    question is not answered by the long-run decomposition.")
println("  * Stationarity is the companion matrix's spectral radius, which")
println("    depends on W's spectrum and is therefore computed, not assumed.")
println("  * In interactive FE, compare `factor_projection(m)`, never")
println("    `m.factors`.")
