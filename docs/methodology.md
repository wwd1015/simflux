# SimFlux Methodology

A technical guide to the simulation models implemented in SimFlux, with emphasis on the two-factor credit portfolio loss framework.

---

## 1. Geometric Brownian Motion (GBM)

### 1.1 Single-Asset GBM

The standard model for equity prices. Under GBM the asset price S follows:

```
dS(t) = μ S(t) dt + σ S(t) dW(t)
```

where μ is the drift (annualized expected return), σ is the volatility (annualized), and W(t) is a standard Wiener process (Brownian motion).

**Exact discrete solution** (log-normal):

```
S(t + dt) = S(t) × exp[(μ − σ²/2) dt + σ √dt × Z]
```

where Z ~ N(0, 1). This is exact — no discretization error regardless of step size.

**Implementation**: SimFlux generates `n_paths` independent trajectories in parallel (Rayon in Rust, vectorized NumPy in Python). Each path consists of `n_steps + 1` prices starting from S₀.

### 1.2 Correlated Multi-Asset GBM

For n assets with individual drifts μᵢ and volatilities σᵢ, the correlation structure is introduced via Cholesky decomposition:

```
1. Compute L = cholesky(Σ)          where Σ is the n×n correlation matrix
2. Generate Z = [Z₁, ..., Zₙ]      independent standard normals
3. Z_corr = L × Z                    correlated normals
4. Sᵢ(t+dt) = Sᵢ(t) × exp[(μᵢ − σᵢ²/2) dt + σᵢ √dt × Z_corr_i]
```

The Cholesky factor L is computed once and reused across all paths and steps.

### 1.3 Time-Varying GBM

When drift and volatility change over time — for instance during a market stress scenario — the parameters μ(t) and σ(t) are provided as time-value pairs. At each simulation step, parameters are linearly interpolated:

```
For step i at time tᵢ:
    μᵢ = interp(tᵢ, mu_times, mu_values)
    σᵢ = interp(tᵢ, sigma_times, sigma_values)
    S(tᵢ + dt) = S(tᵢ) × exp[(μᵢ − σᵢ²/2) dt + σᵢ √dt × Z]
```

Outside the provided time range, values clamp to the nearest boundary (no extrapolation).

---

## 2. Two-Factor Credit Portfolio Loss Model

This is a Merton-framework Monte Carlo model for simulating credit losses across a portfolio of loans, bonds, or other defaultable instruments. It answers the question: *"Given my portfolio, what is the probability distribution of losses over a specified horizon?"*

### 2.1 Economic Intuition

Credit defaults are not independent. When the economy weakens:
- More borrowers default (systematic risk)
- Borrowers in the same industry default together (sector correlation)
- Losses on defaulted loans are higher (LGD-systematic correlation)

The two-factor model captures this by decomposing each borrower's creditworthiness into:
1. A **sector-level systematic factor** (shared economic shock)
2. An **idiosyncratic factor** (borrower-specific risk)

### 2.2 Asset Value Model

For each asset i in sector s, the latent asset value is:

```
Vᵢ = √ρₛ × Fₛ + √(1 − ρₛ) × εᵢ
```

where:
- **ρₛ** is the intra-sector correlation (how much assets within sector s move together)
- **Fₛ** is the sector systematic factor (shared across all assets in sector s)
- **εᵢ** ~ N(0, 1) is the idiosyncratic factor (specific to asset i)
- **Vᵢ** ~ N(0, 1) by construction

The sector factors themselves are correlated across sectors via a sector correlation matrix Σ_sector:

```
[F₁, F₂, ..., Fₖ] = L_sector × [Z₁, Z₂, ..., Zₖ]
```

where L_sector = cholesky(Σ_sector) and Zⱼ are independent standard normals.

### 2.3 Default Condition

Asset i defaults if its latent value falls below a threshold derived from its probability of default (PD):

```
Default if: Vᵢ ≤ Φ⁻¹(PDᵢ)
```

where Φ⁻¹ is the inverse standard normal CDF. This threshold maps the PD to the standard normal scale — for example:
- PD = 1% → threshold ≈ −2.326
- PD = 5% → threshold ≈ −1.645
- PD = 10% → threshold ≈ −1.282

When the economy is bad (Fₛ is very negative), √ρₛ × Fₛ drags Vᵢ down, making many assets in sector s cross their thresholds simultaneously. This is how the model generates correlated defaults.

### 2.4 Loss Given Default (LGD)

When an asset defaults, the loss severity is not fixed — it is drawn from a **Beta distribution** whose parameters are calibrated to the asset's LGD mean and standard deviation:

```
α = lgd_mean × (lgd_mean × (1 − lgd_mean) / lgd_std² − 1)
β = (1 − lgd_mean) × (lgd_mean × (1 − lgd_mean) / lgd_std² − 1)
LGD ~ Beta(α, β)
```

The Beta distribution is bounded on [0, 1], making it natural for loss rates.

**Systematic LGD correlation**: In reality, losses are worse when the economy is bad. The model incorporates this by correlating the LGD draw with the sector factor:

```
U_lgd = Φ(ρ_lgd × Fₛ + √(1 − ρ_lgd²) × η)     where η ~ N(0,1)
LGD_realized = Beta⁻¹(U_lgd; α, β)
```

where ρ_lgd is the systematic LGD correlation parameter. When Fₛ is very negative (bad economy), U_lgd shifts higher, producing higher realized LGD values.

### 2.5 Loss Calculation

```
Lossᵢ = Defaultᵢ × LGDᵢ × Exposureᵢ
Portfolio Loss = Σᵢ Lossᵢ
```

### 2.6 Multi-Period Extension

The single-period model asks "does this asset default by horizon T?" The multi-period extension asks "does it default, and if so, **when**?"

The horizon is divided into `n_periods` discrete intervals (e.g., 8 quarters for a 2-year horizon). At each period k:

1. **Generate independent sector factors** for period k (each period is an independent economic draw)
2. **For each surviving asset**, compute the asset value using the period's factors
3. **Check default** against the period's conditional PD threshold
4. **If defaulted**: record the time, compute LGD using this period's factors, mark asset as defaulted
5. **If survived**: carry forward to period k+1

**Deriving conditional PDs from a term structure:**

The input is a cumulative PD term structure — for example `[0.01, 0.025, 0.04, 0.06]` meaning 1% chance of default by Q1, 2.5% by Q2, 4% by Q3, 6% by Q4.

The conditional (forward) probability of defaulting in period k, given survival to period k, is:

```
PD_conditional(k=0) = PD_cumulative(0)
PD_conditional(k)   = [PD_cumulative(k) − PD_cumulative(k−1)] / [1 − PD_cumulative(k−1)]
```

When no term structure is provided, a constant hazard rate is assumed:

```
PD_conditional = 1 − (1 − PD_flat)^(1/n_periods)
```

This ensures the cumulative default probability over all periods equals the original flat PD.

### 2.7 Output Statistics

For each Monte Carlo trial, the model produces a single portfolio loss. Across `n_simulations` trials, the loss distribution yields:

| Statistic | Definition |
|-----------|-----------|
| **Mean** | Expected loss = E[L] |
| **Std Dev** | Volatility of loss |
| **VaR 95/99/99.9%** | Value-at-Risk: loss exceeded with 5%/1%/0.1% probability |
| **ES 95/99%** | Expected Shortfall: average loss in the worst 5%/1% of scenarios |
| **Max Loss** | Worst-case observed loss |

These are computed at both the portfolio and per-sector level.

### 2.8 Key Parameters and Their Effects

| Parameter | Typical Range | Effect When Increased |
|-----------|--------------|----------------------|
| **PD** | 0.1% – 20% | More defaults, higher expected loss |
| **LGD mean** | 30% – 70% | Higher loss per default |
| **LGD std** | 5% – 25% | Fatter tails on loss distribution |
| **Intra-sector correlation (ρ)** | 10% – 60% | Defaults cluster within sectors; fatter tail risk |
| **Inter-sector correlation** | 5% – 30% | Sectors crash together; extreme tail risk |
| **Systematic LGD correlation** | 0% – 50% | Losses spike during stress; VaR/ES increase |
| **n_periods** | 1 – 20 | Finer time resolution; enables default timing analysis |

### 2.9 Comparison with Industry Models

| Feature | SimFlux | Moody's RiskFrontier | CreditMetrics | Basel IRB |
|---------|---------|---------------------|---------------|-----------|
| Factor structure | Two-factor (sector + idiosyncratic) | Multi-factor | Single systematic | Single systematic |
| Default mechanism | Threshold on latent variable | Threshold on latent variable | Threshold on latent variable | Analytic Vasicek |
| LGD | Stochastic Beta with systematic correlation | Stochastic with systematic | Fixed or stochastic | Fixed (downturn) |
| Multi-period | Discrete time steps with term structure | Continuous | Single period | Single period |
| Correlation | Per-sector intra + cross-sector matrix | Industry/region factors | Single correlation | Single correlation |
| Computation | Monte Carlo (Rust parallelized) | Monte Carlo | Monte Carlo / Analytic | Analytic |

---

## 3. Monte Carlo Simulation Mechanics

### 3.1 Parallelization Strategy

- **GBM**: Paths are independent → parallelized across paths (Rayon `par_iter`)
- **Portfolio**: Trials are independent → parallelized across trials
- **Factor generation**: Systematic factors pre-generated for all trials before simulation begins (enables parallel trial execution without synchronization)

### 3.2 Random Number Generation

- **Rust**: ChaCha-based RNG (`StdRng` from the `rand` crate) with deterministic seeding per path/trial: `seed = base_seed + stream_index`
- **Python fallback**: `numpy.random.default_rng(seed)` (PCG64)
- **Reproducibility**: Setting the same seed produces identical results within a single backend. Cross-backend results differ due to different RNG algorithms but converge statistically (validated by cross-validation tests).

### 3.3 Convergence Guidance

| Metric | Recommended n_simulations | Expected MC error |
|--------|--------------------------|-------------------|
| Mean loss | 1,000 – 10,000 | ±2–5% |
| VaR 95% | 10,000 – 50,000 | ±5–10% |
| VaR 99% | 50,000 – 200,000 | ±10–15% |
| VaR 99.9% | 200,000 – 1,000,000 | ±15–25% |
| ES 99% | 100,000+ | ±10–20% |

These are approximate — actual convergence depends on portfolio concentration, correlation levels, and PD magnitudes. Higher correlation and concentration require more simulations.

---

## 4. References

1. Merton, R.C. (1974). *On the Pricing of Corporate Debt: The Risk Structure of Interest Rates*. Journal of Finance, 29(2), 449-470.
2. Vasicek, O.A. (2002). *The Distribution of Loan Portfolio Value*. Risk, 15(12), 160-162.
3. Gordy, M.B. (2003). *A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules*. Journal of Financial Intermediation, 12(3), 199-232.
4. Basel Committee on Banking Supervision (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions*.
5. Pykhtin, M. (2004). *Multi-Factor Adjustment*. Risk, 17(3), 85-90.
