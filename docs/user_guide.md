# SimFlux User Guide

This guide consolidates the functionality that used to live in the
`examples/` Python scripts and walks through the most common simulation
workflows directly from an interactive shell or notebook.

The snippets below assume you have SimFlux installed in editable mode:

```bash
pip install -e .
```

If you are using the repository checkout instead of an installed wheel,
prepend `PYTHONPATH=$(pwd)/python` when running the snippets so that the
local package is discovered.

---

## 1. Geometric Brownian Motion (GBM)

### 1.1 Single-Asset GBM

Geometric Brownian Motion (GBM) models asset prices following the stochastic differential equation:
**dS_t = μ S_t dt + σ S_t dW_t**

```python
import numpy as np
import simflux as sf

# Create GBM with specific parameters
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} steps")
print(f"Average final value: {paths[:, -1].mean():.2f}")

# Optional: quick visual check (requires matplotlib)
# import matplotlib.pyplot as plt
# plt.plot(paths[0]); plt.show()
```

**Parameter Explanations:**

- **`mu=0.05`** *(drift rate)*: Expected annualized return of 5%. This represents the average rate at which the asset price grows over time. A stock with historical average return of 5% would use this value.

- **`sigma=0.2`** *(volatility)*: Annualized standard deviation of 20%. This measures price uncertainty - higher values create more volatile, unpredictable price movements. A typical stock might have 15-30% volatility.

- **`S0=100`** *(initial price)*: Starting asset value of $100. This could represent a stock price, index level, or any financial instrument's current value.

- **`n_paths=1000`** *(simulation paths)*: Number of independent price scenarios to generate. More paths provide better statistical accuracy but require more computation.

- **`n_steps=252`** *(time steps)*: Number of discrete time intervals. 252 represents daily steps over one trading year (252 ≈ business days per year).

- **`T=1.0`** *(time horizon)*: Simulation period of 1 year. Combined with n_steps, this gives dt = T/n_steps = 1/252 ≈ daily time increments.

**Key Observations:**
- The simulated mean tends toward `S0 * exp(mu * T) = 100 * exp(0.05 * 1) ≈ $105.13`
- Price variability is governed by `sigma`; lower values produce smoother, less volatile paths
- Final prices follow a log-normal distribution with parameters derived from mu and sigma

### 1.2 Correlated Multi-Asset GBM

When modeling portfolios, asset prices don't move independently. Correlated GBM captures how assets co-move through shared market factors, economic conditions, or industry effects.

```python
import numpy as np
import simflux as sf

# Define correlation structure between three assets
correlation_matrix = np.array([
    [1.0, 0.5, 0.3],  # Asset 1 correlations with [self, asset2, asset3]
    [0.5, 1.0, 0.4],  # Asset 2 correlations with [asset1, self, asset3]
    [0.3, 0.4, 1.0],  # Asset 3 correlations with [asset1, asset2, self]
])

# Create correlated simulation with different parameters per asset
correlated = sf.CorrelatedGBM(
    mu=[0.08, 0.06, 0.10],           # Different expected returns
    sigma=[0.20, 0.25, 0.30],        # Different volatilities
    S0=[100, 50, 200],               # Different starting prices
    correlation_matrix=correlation_matrix,
)

paths = correlated.simulate(n_paths=1000, n_steps=252, T=1.0)

print("Realized correlation matrix:")
final_returns = [(paths[:, i, -1] / paths[:, i, 0]) - 1 for i in range(3)]
print(np.corrcoef(final_returns))
```

**Parameter Explanations:**

**Correlation Matrix Structure:**
- **`correlation_matrix[i,j]`**: Correlation between asset i and asset j
- **Diagonal elements = 1.0**: Each asset perfectly correlates with itself
- **Matrix is symmetric**: correlation(A,B) = correlation(B,A)
- **Values between -1 and 1**: -1 (perfect negative), 0 (independent), +1 (perfect positive)

**Example Correlation Interpretation:**
- **`[1.0, 0.5, 0.3]`**: Asset 1 has 50% correlation with Asset 2, 30% with Asset 3
- **0.5 correlation**: When Asset 1 goes up 10%, Asset 2 tends to go up ~5% on average
- **0.3 correlation**: Weaker relationship - Asset 3 moves somewhat with Asset 1 but less predictably

**Asset-Specific Parameters:**
- **`mu=[0.08, 0.06, 0.10]`**: Expected annual returns of 8%, 6%, 10% respectively
  - Asset 1: Growth stock (8% expected return)
  - Asset 2: Utility stock (6% lower expected return)
  - Asset 3: Technology stock (10% higher expected return)

- **`sigma=[0.20, 0.25, 0.30]`**: Annual volatilities of 20%, 25%, 30%
  - Asset 1: Moderate risk (20% volatility)
  - Asset 2: Higher risk (25% volatility)
  - Asset 3: Highest risk (30% volatility)

- **`S0=[100, 50, 200]`**: Starting prices of $100, $50, $200
  - Different price levels don't affect correlations or returns
  - Useful for modeling actual market prices

**Financial Intuition:**
- **High correlation (0.5)** between Assets 1&2: Similar market sectors or geographies
- **Lower correlation (0.3, 0.4)** with Asset 3: Different industry or market segment
- **Diversification benefit**: Portfolio risk < weighted average of individual risks due to imperfect correlations

**Output Shape:** `paths.shape = (n_paths, n_assets, n_steps+1)`
- 1000 simulation scenarios × 3 assets × 253 time points (including t=0)

The realized correlation should closely match the input matrix when using sufficient simulation paths (1000+ typically adequate).

### 1.3 Time-Varying Parameters

Real markets don't have constant returns and volatilities. SimFlux supports **time-varying parameters** by accepting simple time series arrays for mu (drift) and sigma (volatility).

#### Simple Time-Varying Single Asset GBM

```python
import numpy as np
import simflux as sf

# Define time series for mu (drift/return)
mu_times = [0.0, 0.25, 0.75, 1.0]          # Time points
mu_values = [0.08, -0.15, 0.12, 0.08]      # Corresponding mu values

# Define time series for sigma (volatility)
sigma_times = [0.0, 0.25, 0.75, 1.0]       # Time points
sigma_values = [0.20, 0.45, 0.25, 0.20]    # Corresponding sigma values

# Create time-varying GBM
gbm = sf.TimeVaryingGBM(
    mu_times=mu_times,
    mu_values=mu_values,
    sigma_times=sigma_times,
    sigma_values=sigma_values,
    S0=100  # Initial price
)

# Simulate paths
paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)
```

**What this does:**
- **t=0.0 to 0.25**: Normal market (8% return, 20% volatility)
- **t=0.25 to 0.75**: Crisis period (-15% return, 45% volatility)
- **t=0.75 to 1.0**: Recovery (12% return, 25% volatility)

Linear interpolation is used between time points, so parameters change smoothly.

#### Multi-Asset Time-Varying GBM

```python
# Tech stock - high growth, volatile
tech_mu_times = [0.0, 0.3, 1.0]
tech_mu_values = [0.15, -0.10, 0.20]       # 15% -> -10% -> 20%
tech_sigma_times = [0.0, 0.3, 1.0]
tech_sigma_values = [0.25, 0.60, 0.30]     # 25% -> 60% -> 30%

# Bank stock - stable, crisis-affected
bank_mu_times = [0.0, 0.5, 1.0]
bank_mu_values = [0.06, 0.02, 0.08]        # 6% -> 2% -> 8%
bank_sigma_times = [0.0, 0.5, 1.0]
bank_sigma_values = [0.18, 0.35, 0.22]     # 18% -> 35% -> 22%

# Correlation matrix
correlation_matrix = np.array([[1.0, 0.4], [0.4, 1.0]])

# Create multi-asset time-varying GBM
multi_gbm = sf.TimeVaryingCorrelatedGBM(
    mu_times=[tech_mu_times, bank_mu_times],
    mu_values=[tech_mu_values, bank_mu_values],
    sigma_times=[tech_sigma_times, bank_sigma_times],
    sigma_values=[tech_sigma_values, bank_sigma_values],
    S0=[100, 100],
    correlation_matrix=correlation_matrix
)

# Simulate correlated paths
multi_paths = multi_gbm.simulate(n_paths=1000, n_steps=252, T=1.0)
# Returns shape: (n_paths, n_assets, n_steps + 1)
```

#### Check Parameter Evolution

```python
# See how parameters change over time
mu, sigma = gbm.get_parameters_at_time(0.4)  # Get values at t=0.4
print(f"At t=0.4: μ={mu:.1%}, σ={sigma:.1%}")

# Or get evolution over time
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mu, sigma = gbm.get_parameters_at_time(t)
    print(f"t={t:.2f}: μ={mu:+.1%}, σ={sigma:.1%}")
```

#### Common Use Cases

**Market Crisis Simulation:**
```python
# Normal -> Crisis -> Recovery
mu_times = [0.0, 0.2, 0.8, 1.0]
mu_values = [0.08, -0.25, 0.12, 0.08]
sigma_times = [0.0, 0.2, 0.8, 1.0]
sigma_values = [0.18, 0.55, 0.30, 0.20]
```

**Gradual Economic Change:**
```python
# Slow transition over 2 years
mu_times = [0.0, 2.0]
mu_values = [0.03, 0.12]  # 3% to 12% return
sigma_times = [0.0, 2.0]
sigma_values = [0.15, 0.25]  # 15% to 25% volatility
```

**Interest Rate Cycle:**
```python
# Quarterly rate changes
quarters = np.arange(0, 2.1, 0.25)  # 0, 0.25, 0.5, ..., 2.0
mu_rates = 0.05 + 0.03 * np.sin(2 * np.pi * quarters / 2.0)  # 2-year cycle
sigma_rates = 0.20 + 0.05 * np.cos(2 * np.pi * quarters / 2.0)
```

**Benefits of Time-Varying Parameters:**
- **Realistic Market Modeling**: Capture regime changes, crises, cycles
- **Stress Testing**: Model extreme scenarios with high volatility periods
- **Economic Cycles**: Incorporate business cycle effects
- **Policy Changes**: Model impact of central bank decisions
- **Sector Rotation**: Different assets can have different time-varying patterns

The time-varying approach provides much more realistic simulations compared to constant parameters, while remaining simple to use with just time series arrays.

---

## 2. Portfolio Loss Simulation

### 2.1 Building a Sample Portfolio

Portfolio loss simulation models credit risk using the **Two-Factor Merton Framework**, similar to Moody's RiskFrontier. Each asset can default, and correlations arise from shared systematic risk factors.

```python
import simflux as sf
import numpy as np

# Define sector-level correlation structure
sector_corr = np.array([
    [1.0, 0.15, 0.05],   # Technology sector correlations
    [0.15, 1.0, 0.10],   # Finance sector correlations
    [0.05, 0.10, 1.0],   # Healthcare sector correlations
])

# Create a portfolio with realistic sector distribution
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
    n_assets_per_sector=[50, 30, 20],
    sectors=["Technology", "Finance", "Healthcare"],
    sector_correlation_matrix=sector_corr,
    intra_sector_correlations={"Technology": 0.35, "Finance": 0.45, "Healthcare": 0.30},
)

summary = portfolio.get_portfolio_summary()
print(f"Assets: {summary['n_assets']}, sectors: {summary['n_sectors']}")
for sector, info in summary['sectors'].items():
    print(f"  {sector}: {info['n_assets']} assets, PD={info['avg_pd']:.3f}, LGD={info['avg_lgd']:.3f}")
```

**Parameter Explanations:**

**Sector Correlation Matrix (`sector_corr`):**
- **`[1.0, 0.15, 0.05]`**: Technology has 15% correlation with Finance, 5% with Healthcare
- **Economic interpretation:**
  - **0.15 (Tech-Finance)**: Moderate correlation - both affected by interest rates, economic cycles
  - **0.05 (Tech-Healthcare)**: Low correlation - different business cycles and risk factors
  - **0.10 (Finance-Healthcare)**: Low-moderate correlation - some shared economic sensitivity

**Portfolio Structure Parameters:**
- **`n_assets_per_sector=[50, 30, 20]`**: Portfolio composition
  - 50 Technology companies (e.g., software, hardware firms)
  - 30 Financial institutions (e.g., banks, insurance companies)
  - 20 Healthcare companies (e.g., pharmaceuticals, medical devices)
  - Total: 100 assets with sector diversification

- **`sectors=["Technology", "Finance", "Healthcare"]`**: Sector labels for categorization and reporting

**Intra-Sector Correlations:**
- **`"Technology": 0.35`**: Tech companies have 35% average correlation
  - Shared exposure to: innovation cycles, technology adoption, venture capital
- **`"Finance": 0.45`**: Financial firms have 45% average correlation
  - Shared exposure to: interest rates, credit cycles, regulatory changes
- **`"Healthcare": 0.30`**: Healthcare companies have 30% average correlation
  - Shared exposure to: drug approvals, healthcare policy, demographic trends

**Two-Factor Correlation Model:**

The framework decomposes each asset's risk into:

1. **Systematic Factor (Sector)**: Shared risk affecting all assets in the sector
   - Weight: `√(intra_sector_correlation)`
   - Correlation between sectors determined by `sector_correlation_matrix`

2. **Idiosyncratic Factor (Asset-specific)**: Company-specific risk
   - Weight: `√(1 - intra_sector_correlation)`
   - Independent across all assets

**Resulting Correlation Structure:**
- **Same sector**: Correlation = `intra_sector_correlation`
  - Tech-Tech pairs: 0.35 correlation
- **Cross-sector**: Correlation = `√(intra_s) × √(intra_t) × sector_matrix[s,t]`
  - Tech-Finance pair: √(0.35) × √(0.45) × 0.15 ≈ 0.06 correlation

**Credit Risk Parameters (Auto-Generated):**
- **PD (Probability of Default)**: Likelihood asset defaults within 1 year
  - Typically 1-5% for investment grade, higher for risky assets
- **LGD (Loss Given Default)**: Percentage of exposure lost if default occurs
  - Typically 40-60%, depending on seniority and collateral

#### Building from a DataFrame

For real-world applications, you'll often have asset data in tabular form from databases or spreadsheets. This example shows how to construct portfolios from detailed asset data.

```python
import pandas as pd
import numpy as np
import simflux as sf

# Define specific assets with their risk characteristics
asset_table = pd.DataFrame({
    "asset_id": [1, 2, 3, 4],
    "sector": ["Technology", "Technology", "Finance", "Healthcare"],
    "pd": [0.02, 0.035, 0.05, 0.03],                    # Probability of Default
    "lgd_mean": [0.6, 0.55, 0.45, 0.5],                # Loss Given Default (mean)
    "lgd_std": [0.2, 0.18, 0.15, 0.17],                # Loss Given Default (std)
    "exposure": [1.2e6, 0.8e6, 1.5e6, 1.1e6],          # Exposure amount
    "intra_sector_correlation": [0.40, 0.40, 0.45, 0.30], # Asset-specific correlations
})

# Sector correlations (same as before)
sector_corr = np.array([
    [1.0, 0.25, 0.10],     # Technology correlations
    [0.25, 1.0, 0.15],     # Finance correlations
    [0.10, 0.15, 1.0],     # Healthcare correlations
])

# Build portfolio from detailed data
portfolio = sf.TwoFactorPortfolio(
    assets=asset_table,
    sector_correlation_matrix=sector_corr,
)

print(portfolio.get_portfolio_summary()["correlation_structure"])
```

**Asset Table Column Explanations:**

**Required Columns:**
- **`asset_id`**: Unique identifier for each asset (loan, bond, counterparty)
- **`sector`**: Industry classification determining correlation structure
- **`pd`** *(Probability of Default)*: Annual default probability as decimal
  - Asset 1: 2.0% (investment grade technology firm)
  - Asset 2: 3.5% (higher-risk technology startup)
  - Asset 3: 5.0% (financial institution with credit risk)
  - Asset 4: 3.0% (stable healthcare company)

- **`lgd_mean`** *(Loss Given Default - Mean)*: Expected loss percentage if default occurs
  - Asset 1: 60% (unsecured technology loan)
  - Asset 2: 55% (some equipment collateral)
  - Asset 3: 45% (secured bank financing)
  - Asset 4: 50% (healthcare equipment backing)

- **`lgd_std`** *(Loss Given Default - Standard Deviation)*: Uncertainty in recovery
  - Higher values indicate more uncertain recovery rates
  - Reflects collateral quality, legal jurisdiction, asset liquidity

- **`exposure`**: Dollar amount at risk
  - Asset 1: $1.2M exposure to technology firm
  - Asset 2: $0.8M exposure to tech startup
  - Asset 3: $1.5M exposure to bank
  - Asset 4: $1.1M exposure to healthcare company

**Optional Column:**
- **`intra_sector_correlation`**: Asset-specific correlation within its sector
  - Allows fine-tuning beyond sector-wide defaults
  - Asset 3: 0.45 (higher correlation typical of financial sector)
  - Asset 4: 0.30 (lower correlation for diversified healthcare)

**Flexibility Features:**
- **Missing columns**: Use sector-wide defaults for intra-correlations
- **Automatic processing**: `AssetData.from_dataframe` handles data conversion
- **Mixed specifications**: Combine sector defaults with asset-specific overrides
- **Scalability**: Handles portfolios from dozens to thousands of assets

This approach enables realistic modeling of actual credit portfolios with varying risk characteristics, exposures, and correlation structures.

### 2.2 Running Simulations

Once your portfolio is configured, run Monte Carlo simulations to calculate risk metrics used in regulatory capital, economic capital, and stress testing.

```python
# Run Monte Carlo simulation
results = portfolio.simulate(n_simulations=50_000)
stats = results["portfolio_statistics"]

print("Portfolio loss statistics")
for key in ["mean", "std_dev", "var_95", "var_99", "var_999", "expected_shortfall_99", "max_loss"]:
    print(f"  {key}: {stats[key]:,.0f}")

print("\nSector highlights")
for sector, sector_stats in results["sector_statistics"].items():
    print(f"  {sector}: 95% VaR = {sector_stats['var_95']:,.0f}")
```

**Simulation Parameters:**
- **`n_simulations=50_000`**: Number of Monte Carlo scenarios
  - More simulations → more accurate tail risk estimates
  - 50K typical for production risk measurement
  - 10K adequate for development/testing

**Output Risk Metrics Explained:**

**Core Statistics:**
- **`mean`**: Expected loss under normal conditions
  - Mathematical expectation: Σ(PD × LGD × Exposure) across all assets
  - Used for loss provisioning and pricing

- **`std_dev`**: Standard deviation of loss distribution
  - Measures portfolio loss volatility
  - Indicates concentration vs diversification benefits

**Value at Risk (VaR) Metrics:**
- **`var_95`**: 95th percentile loss (95% VaR)
  - "We expect losses to exceed this amount only 5% of the time"
  - Regulatory capital requirement baseline

- **`var_99`**: 99th percentile loss (99% VaR)
  - "We expect losses to exceed this amount only 1% of the time"
  - Common economic capital standard

- **`var_999`**: 99.9th percentile loss (99.9% VaR)
  - "We expect losses to exceed this amount only 0.1% of the time"
  - Stress testing and extreme risk measurement

**Expected Shortfall (Conditional VaR):**
- **`expected_shortfall_99`**: Average loss when losses exceed 99% VaR
  - Measures tail risk beyond VaR
  - "Given that we're in the worst 1% of scenarios, what's the average loss?"
  - More comprehensive than VaR for risk management

**Extreme Statistics:**
- **`max_loss`**: Worst-case loss across all simulations
  - Helpful for stress testing and scenario analysis
  - Should approach total portfolio exposure in extreme cases

**Sector Breakdown:**
- **Sector VaRs**: Risk attribution by industry
  - Identifies concentration risks
  - Supports sector limit setting and diversification strategies

### 2.3 Sensitivity to Correlations

```python
import pandas as pd

base = sf.TwoFactorPortfolio.create_sample_portfolio(
    n_assets_per_sector=50,
    sectors=["Technology", "Finance"],
)

rows = []
for cross_corr in [0.05, 0.15, 0.30, 0.50]:
    sector_corr = np.array([[1.0, cross_corr], [cross_corr, 1.0]])
    scenario = sf.TwoFactorPortfolio(
        assets=base.assets,
        intra_sector_correlations=0.4,
        sector_correlation_matrix=sector_corr,
    )
    stats = scenario.simulate(n_simulations=20_000)["portfolio_statistics"]
    rows.append({
        "cross_correlation": cross_corr,
        "var_95": stats["var_95"],
        "var_99": stats["var_99"],
        "expected_shortfall_99": stats["expected_shortfall_99"],
    })

print(pd.DataFrame(rows))
```

---

## 3. Measuring Performance

The old `run_benchmark.py` script captured three illustrative tests
(single-asset GBM, correlated GBM, and portfolio loss). You can now run
the same measurements inline:

```python
import time
import numpy as np
import simflux as sf

def time_block(label, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    duration = time.perf_counter() - start
    print(f"{label}: {duration:.3f}s")
    return result, duration

gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths, t = time_block("Single Asset GBM", gbm.simulate, 10_000, 252, 1.0)
print(f"  Throughput ~ {(paths.shape[0]*paths.shape[1])/t:,.0f} ops/s")

correlation = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
multi = sf.CorrelatedGBM(mu=[0.05, 0.03, 0.07], sigma=[0.2, 0.15, 0.25], S0=[100, 50, 200], correlation_matrix=correlation)
multi_paths, t = time_block("Correlated GBM", multi.simulate, 5_000, 252, 1.0)
print(f"  Throughput ~ {(np.prod(multi_paths.shape))/t:,.0f} ops/s")

sector_corr = np.array([[1.0, 0.15], [0.15, 1.0]])
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
    n_assets_per_sector=25,
    sectors=["Technology", "Finance"],
    sector_correlation_matrix=sector_corr,
)
_, t = time_block("Portfolio Simulation", portfolio.simulate, 5_000)
print(f"  Sims per second ~ {5_000/t:,.0f}")
```

> **Tip**: Compare the timings above with `sf.GBM(...).engine._numpy_simulate_gbm` to estimate the
> expected Rust speedup once you install the binary wheel.

---

## 4. Working Without the Rust Backend

SimFlux automatically falls back to NumPy implementations when the Rust
extension module is unavailable. To verify behaviour in environments
without the compiled backend:

```python
import importlib
import sys
import simflux as sf

# Simulate a missing extension module
sys.modules.pop("simflux._rust", None)
importlib.invalidate_caches()

gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)  # Triggers fallback warning
paths = gbm.simulate(n_paths=1000, n_steps=252)
print(f"Fallback paths shape: {paths.shape}")

sector_corr = np.array([[1.0, 0.15], [0.15, 1.0]])
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(20, ["Technology", "Finance"], sector_correlation_matrix=sector_corr)
stats = portfolio.simulate(n_simulations=5_000)["portfolio_statistics"]
print(f"Mean loss (fallback): {stats['mean']:,.0f}")
```

You should see warnings indicating the fallback is in use. Performance
will be lower, but functionality remains intact.

---

## 5. Troubleshooting

- **`TypeError: Can't instantiate abstract class SimulationEngine`**:
  Ensure you are importing from the repository checkout or version
  0.1.0+ that contains the `validate_inputs` shim.
- **Performance seems slow**: Install the Rust-enabled wheel from PyPI or
  run `pip install maturin && maturin develop --release` to build locally.
- **Need interim results**: Storage features require the compiled Rust
  backend; the fallback will raise a `RuntimeError` if you request interim
  output without it.

---

This document replaces the ad-hoc scripts in `examples/`. Copy the
snippets into a notebook or shell to explore the library interactively.
