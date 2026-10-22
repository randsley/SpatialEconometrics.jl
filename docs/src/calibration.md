# Calibration & Input-Output

Balancing a matrix to new row and column margins while preserving as much of
its structure as possible — the standard problem in updating a social
accounting matrix or a multi-regional input-output table to a new year's
totals.

## GRAS

The generalized RAS algorithm. Unlike plain RAS it handles **negative
entries**, which matter because a SAM legitimately contains them (inventory
changes, net taxes), and a sign-blind scaling would destroy them.

```julia
# Full result with convergence information
res = balance_gras_full(X0, u_target, v_target; tol = 1e-11)
# res.X_balanced, res.converged, res.kl_divergence, res.iterations

# Simple interface, differentiable via ChainRulesCore + ForwardDiff
X_bal = balance_gras(X0, u_target, v_target)
```

`u_target` is the vector of required row sums and `v_target` the required
column sums; they must agree in total, since no scaling can satisfy
inconsistent margins. Prefer `balance_gras_full` when the result feeds a
decision: it reports whether the iteration actually converged and how far the
balanced matrix moved from the prior, rather than silently returning the last
iterate.

`balance_gras` is differentiable, so a balancing step can sit inside a larger
calibration or an automatic-differentiation pipeline.

## Generalized cross-entropy

When there is information beyond the margins — a known subtotal, an estimated
coefficient, a bound — cross-entropy minimization imposes it as a side
constraint with an uncertainty on each:

```julia
X_bal, λ_dual, μ_dual = balance_generalized_ce(X0, G, b, sigma_b)
```

`G` and `b` state the constraints as `G * vec(X) = b`, and `sigma_b` gives
each one a standard deviation, so a constraint known loosely is not imposed as
though it were exact. The duals say which constraints are doing the work.

## SAM multipliers

```julia
M, A = sam_leontief_multipliers(X_sam, endogenous_idx)
```

`endogenous_idx` selects the accounts treated as endogenous; `A` is the
resulting coefficient matrix and `M` the Leontief inverse. Which accounts are
endogenous is a modelling decision, not a detail — it determines what the
multipliers mean.
