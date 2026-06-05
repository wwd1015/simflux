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
U_lgd = Φ(−ρ_lgd × Fₛ + √(1 − ρ_lgd²) × η)     where η ~ N(0,1)
LGD_realized = Beta⁻¹(U_lgd; α, β)
```

where ρ_lgd is the systematic LGD correlation parameter. Note the **negative** loading on Fₛ: defaults fire in the *low* tail of the latent value (Fₛ very negative = bad economy), so loading LGD on −Fₛ makes a *positive* ρ_lgd encode wrong-way risk. When Fₛ is very negative (bad economy), −ρ_lgd × Fₛ is positive, U_lgd shifts higher, and realized LGD rises — losses are worse exactly when defaults cluster.

### 2.5 Loss Calculation

```
Lossᵢ = Defaultᵢ × LGDᵢ × Exposureᵢ
Portfolio Loss = Σᵢ Lossᵢ
```

### 2.6 Multi-Period Extension

The single-period model asks "does this asset default by horizon T?" The multi-period extension asks "does it default, and if so, **when**?"

The horizon is divided into `n_periods` discrete intervals (e.g., 8 quarters for a 2-year horizon). SimFlux offers **two default-timing models** via `default_timing`, distinguished by how the systematic factor evolves across periods. **Both reproduce the marginal cumulative PD term structure exactly** — the input PDs are honored regardless of the dependence assumption — so they differ only in cross-period dependence, and therefore in the tail.

**`"copula"` (default) — one-factor Gaussian copula of default times (Li, 2000).**
A single latent `V = √ρ·F + √(1−ρ)·ε` is drawn per obligor for the *whole* horizon and compared against the **cumulative** threshold staircase `τ_k = Φ⁻¹(cum_k)`. Default occurs at the first crossing, which gives both *whether* and *when*. Because there is one draw against a monotone staircase, `P(default by k) = Φ(τ_k) = cum_k` **exactly, for any correlation** — no independence assumption. All uncertainty resolves at t=0, so cross-period dependence is maximal and the loss distribution is **invariant to how finely the horizon is sliced** (slicing only re-labels timing).

**`"frailty"` — dynamic frailty (Duffie, Eckner, Horel & Saita, 2009).**
The systematic factor is **persistent**: an AR(1) across periods,
`F_k = φ·F_{k−1} + √(1−φ²)·η_k`, with per-period coefficient `φ = factor_persistence^period_length` (annual autocorrelation, grid-consistent). A *fresh* idiosyncratic shock `ε_k` arrives each period, so conditional on the factor path defaults are independent (doubly stochastic). Default is the first period `V_k ≤ b_k`, where the **barrier `b_k` is calibrated** so the marginal is preserved:

```
P(default by k) = E_path[ ∏_{j≤k} Φ((√ρ·F_j − b_j)/√(1−ρ)) ]  :=  cum_k
```

Because `F` is Markov, this collapses to a 1-D forward recursion on a grid of factor values, solved as a sequential 1-D root-find for `b_0, b_1, …` (see `python/simflux/portfolio/frailty.py`). A *naïve* AR(1) on the independence-derived **forward** thresholds — without recalibrating the barrier — biases the cumulative PD *downward* (EL drifted ~14% at ρ=0.4 in testing); the calibrated barrier is what makes frailty unbiased for any `φ`.

Limits and a caveat:
- `factor_persistence = 0` ⇒ independent periods (forward-PD thresholds; the thinnest tail). `= 1` ⇒ a frozen factor with fresh idiosyncratic shocks (the fattest tail — *not* the copula, which also freezes the idiosyncratic).
- **Frailty is monitoring-frequency-dependent**: because fresh idiosyncratic shocks arrive each period, `n_periods` is itself a modeling choice (how often default is checked), and the tail *rises* with finer grids at fixed `factor_persistence`. Choose `n_periods` to match the real observation/decision frequency. (The copula model, by contrast, is grid-invariant.)

**Choosing.** `copula` is the credit-derivatives-standard default-time model, grid-invariant, and the safe default. `frailty` is the more realistic dynamic model when new shocks genuinely arrive over the horizon — at the cost of a calibrated persistence parameter and monitoring-frequency dependence. At `n_periods = 1` the two coincide.

**Calibrating `factor_persistence`.** It is the annual autocorrelation of the systematic credit factor. Estimate it by probit-transforming an aggregate (or segment) default-rate series into a latent factor and fitting an AR(1); typical annual values are ~0.4–0.7. The default `0.5` is a cycle-realistic *illustrative* value for examples — production use should calibrate to data.

For each model, the per-period inputs are derived as follows.

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

### 2.10 Positioning of the LGD Model

In industry terms, SimFlux's LGD model is a **one-factor Gaussian-copula systematic-recovery
model with a Beta marginal**: LGD is drawn from a moment-matched Beta and coupled to the cycle
by loading a latent normal on the *negative* of the sector factor before mapping through Φ and
Beta⁻¹ (§2.4). This is squarely the **Frye (2000) / Pykhtin (2003)** school of recovery modeling.
The spectrum, from least to most sophisticated on LGD:

| Approach | LGD treatment | SimFlux relative to it |
|----------|---------------|------------------------|
| **Basel IRB** | Fixed *downturn* LGD; no stochasticity, no explicit PD–LGD correlation | SimFlux is **ahead** — stochastic and cycle-correlated |
| **CreditMetrics** (classic) | Beta recovery drawn *independently* of default | SimFlux is **ahead** — adds the systematic link |
| **Frye (2000), Pykhtin (2003)** | Recovery driven by the systematic factor | **SimFlux is here** — Beta marginal (bounded) + Monte Carlo vs. their normal/closed-form |
| **Frye–Jacobs (2012)** | *Parameter-free* conditional LGD as a function of conditional PD | Slightly ahead — removes the free ρ_lgd |
| **IFRS9 / CECL, CCAR/DFAST** | LGD regressed on *named macro covariates* (GDP, unemployment, HPI), point-in-time | Ahead on realism, but **out of scope by design** — see §2.11 |
| **t-copula / collateral-structural / ML LGD** | Tail-dependent coupling, structural collateral, gradient-boosted marginals | Research edge — ahead of SimFlux |

**Where SimFlux sits.** Above the regulatory floor and vanilla CreditMetrics; in the Frye/Pykhtin
band, with a bounded Beta marginal that is arguably cleaner than their unbounded normal/lognormal
recovery. Below macro-driven point-in-time models and tail-dependent copula/ML recovery models.

**Known limitations relative to the cutting edge.**

1. **Recovery loads on the same sector factor as default** — there is no *independent* systematic
   recovery risk. Frye/Pykhtin typically give recovery its own factor that is *correlated with* —
   not identical to — the default factor. SimFlux ties the two together (modulo idiosyncratic noise).
2. **Gaussian copula ⇒ zero tail dependence.** In the most extreme scenarios — exactly where 99.9%
   VaR lives — the LGD–default coupling weakens relative to reality. A t-copula link would inject the
   missing tail dependence.
3. **`systematic_lgd_correlation` is per-sector.** It accepts a scalar (broadcast to all
   sectors), a per-sector list (in sorted-sector order), a `{sector: value}` dict (missing
   sectors default to 0.3), or the sentinel `"match_intra"`. A *data*-calibration recipe for
   ρ_lgd (downturn-LGD multiple, or matching a historical PD–LGD correlation) remains future
   work; the *structural* `"match_intra"` option is built in.

   A parameterization subtlety governs that "match the default driver" calibration. ρ_lgd and
   the intra-sector correlation are the *same kind* of quantity — each is a latent driver's
   sensitivity to the systematic factor F — but they are stored on **different scales**. The default
   driver is `V = √ρ_intra · F + √(1−ρ_intra) · ε`, so its correlation with F is `√ρ_intra` and
   `ρ_intra` is a *squared* loading (the asset–asset correlation). The LGD driver is
   `L = −ρ_lgd · F + √(1−ρ_lgd²) · η`, so ρ_lgd is the loading *directly* — `Corr(L, F) = ρ_lgd`.
   Therefore, to give LGD the **same cycle-sensitivity** as the default driver, the matched value is

   ```
   ρ_lgd = √ρ_intra          (NOT ρ_lgd = ρ_intra)
   ```

   e.g. `ρ_intra = 0.36` ⇒ default-to-factor correlation 0.6 ⇒ matched `ρ_lgd = 0.6`. Setting
   `ρ_lgd = ρ_intra = 0.36` would *under*-couple LGD to the cycle by the square-root gap. The
   `"match_intra"` option implements exactly this — it sets `ρ_lgd_s = √ρ_intra_s` sector by sector.

### 2.11 Why Not a CCAR / Macro-Conditional LGD?

CCAR/DFAST (and IFRS9/CECL) require LGD **and** PD to be conditioned on *named, supervisor-provided
macroeconomic scenarios* — specific GDP, unemployment, and house-price paths — and to produce losses
*under that scenario*. SimFlux deliberately does **not** do this. Its entire framework, both the
default model and the LGD model, is driven by **abstract standard-normal systematic factors that are
not mapped to, or calibrated against, any observable macro variable or named scenario.**

This is a scoping decision, not an oversight:

- SimFlux answers an **unconditional** question — *"what is the loss distribution implied by the
  estimated correlation structure?"* — by integrating over all systematic states.
- CCAR answers a **conditional** question — *"what is the loss under the Fed's severely adverse
  scenario?"* — by fixing the macro path.

Supporting a CCAR-style LGD would require (a) mapping the latent sector factor to macro covariates and
(b) ingesting exogenous scenario paths to condition on — neither of which this framework does, because
neither PD nor LGD here is tied to a specific macro scenario. Macro-conditional stress testing is
therefore intentionally left to dedicated regulatory tooling; SimFlux is a scenario-agnostic risk
engine. (Recorded as an architectural decision — see `docs/adr/`.)

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
6. Frye, J. (2000). *Depressing Recoveries*. Risk, 13(11), 108-111. (Systematic recovery risk: recovery driven by the same factor as default.)
7. Pykhtin, M. (2003). *Unexpected Recovery Risk*. Risk, 16(8), 74-78. (Stochastic recovery correlated with the systematic factor.)
8. Frye, J. & Jacobs, M. (2012). *Credit Loss and Systematic Loss Given Default*. Journal of Credit Risk, 8(1), 109-140. (Parameter-free conditional-LGD function of conditional PD.)
