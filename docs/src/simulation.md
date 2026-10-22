# Simulation & Power Analysis

## Data Generating Processes

Simulate data from standard spatial DGPs:

```julia
using SpatialEconometrics

# SAR: y = (I - ρW)⁻¹(Xβ + ε)
dgp = SAR_DGP(0.5, [1.5, -0.8], 0.5)
y, X = simulate_spatial(dgp, W, X)

# SEM: y = Xβ + (I - λW)⁻¹ε
dgp = SEM_DGP(0.4, [1.0, -0.5], 0.6)

# SDM: y = (I - ρW)⁻¹(Xβ + WXθ + ε)
dgp = SDM_DGP(0.3, [1.0, -0.5], [0.4, -0.2], 0.5)

# SLX: y = Xβ + WXθ + ε
dgp = SLX_DGP([1.0, -0.5], [0.3, -0.1], 0.4)

# SAC: y = (I - ρW)⁻¹(Xβ + (I - λW)⁻¹ε)
dgp = SAC_DGP(0.3, 0.2, [1.0, -0.5], 0.5)

# SDEM: y = Xβ + WXθ + (I - λW)⁻¹ε
dgp = SDEM_DGP(0.4, [1.0, -0.5], [0.3, -0.2], 0.5)
```

## Monte Carlo Experiments

Run Monte Carlo experiments to assess estimator properties:

```julia
mc = monte_carlo_spatial(
    SAR_DGP(0.5, [1.5, -0.8], 0.5),
    W, N, 2,
    (W, X, y) -> fit_spatial(SAR(), W, X, y);
    config=MonteCarloConfig(n_replications=500, seed=42)
)
```

The result reports bias, RMSE, and test rejection rates.

## Power Analysis

Compute power curves across sample sizes and spatial parameter strengths:

```julia
result = power_analysis(
    ρ -> SAR_DGP(ρ, [1.5, -0.8], 0.5),
    [5, 10, 15, 20],           # grid sizes (N = L²)
    [0.0, 0.1, 0.3, 0.5, 0.7]; # ρ values
    n_reps=200, alpha=0.05
)
```

## API

See the full [API Reference](@ref) for `simulate_spatial`, `monte_carlo_spatial`, and `power_analysis`.
