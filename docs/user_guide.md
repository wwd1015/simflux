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

```python
import numpy as np
import simflux as sf

gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} steps")
print(f"Average final value: {paths[:, -1].mean():.2f}")

# Optional: quick visual check (requires matplotlib)
# import matplotlib.pyplot as plt
# plt.plot(paths[0]); plt.show()
```

Key observations:
- The simulated mean tends toward `S0 * exp(mu * T)`.
- Variability is governed by `sigma`; decrease it for smoother paths.

### 1.2 Correlated Multi-Asset GBM

```python
import numpy as np
import simflux as sf

correlation_matrix = np.array([
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0],
])

correlated = sf.CorrelatedGBM(
    mu=[0.08, 0.06, 0.10],
    sigma=[0.20, 0.25, 0.30],
    S0=[100, 50, 200],
    correlation_matrix=correlation_matrix,
)

paths = correlated.simulate(n_paths=1000, n_steps=252, T=1.0)

print("Realized correlation matrix:")
final_returns = [(paths[:, i, -1] / paths[:, i, 0]) - 1 for i in range(3)]
print(np.corrcoef(final_returns))
```

The realized correlation should be close to the input matrix once the
number of simulated paths is large enough.

---

## 2. Portfolio Loss Simulation

### 2.1 Building a Sample Portfolio

```python
import simflux as sf

sector_corr = np.array([
    [1.0, 0.15, 0.05],
    [0.15, 1.0, 0.10],
    [0.05, 0.10, 1.0],
])

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

The two-factor correlation structure assigns each sector a systematic
shock whose pairwise correlation comes directly from the provided
`sector_correlation_matrix`. Assets load on that sector shock with weight
`sqrt(intra_sector)` and carry the remaining variance in an idiosyncratic
component with weight `sqrt(1 - intra_sector)`. Assets in the same sector
therefore match the configured intra correlation, while cross-sector
pairs exhibit correlation `sqrt(intra_s) * sqrt(intra_t) * sector_matrix[s, t]`.

#### Building from a DataFrame

If your input data already lives in tabular form you can skip the helper
and instantiate the portfolio directly from a `pandas.DataFrame`.

```python
import pandas as pd
import numpy as np
import simflux as sf

asset_table = pd.DataFrame({
    "asset_id": [1, 2, 3, 4],
    "sector": ["Technology", "Technology", "Finance", "Healthcare"],
    "pd": [0.02, 0.035, 0.05, 0.03],
    "lgd_mean": [0.6, 0.55, 0.45, 0.5],
    "lgd_std": [0.2, 0.18, 0.15, 0.17],
    "exposure": [1.2e6, 0.8e6, 1.5e6, 1.1e6],
    "intra_sector_correlation": [0.40, 0.40, 0.45, 0.30],
})

sector_corr = np.array([
    [1.0, 0.25, 0.10],
    [0.25, 1.0, 0.15],
    [0.10, 0.15, 1.0],
])

portfolio = sf.TwoFactorPortfolio(
    assets=asset_table,
    sector_correlation_matrix=sector_corr,
)

print(portfolio.get_portfolio_summary()["correlation_structure"])
```

`AssetData.from_dataframe` automatically creates sector identifiers and
consumes the optional `intra_sector_correlation` column so you can keep
per-sector preferences next to the raw asset inputs. Only the required
columns (`asset_id`, `sector`, `pd`, `lgd_mean`, `lgd_std`, `exposure`)
must be present; any omitted intra-sector metadata falls back to the
defaults used earlier, while `sector_correlation_matrix` remains an
explicit constructor argument.

### 2.2 Running Simulations

```python
results = portfolio.simulate(n_simulations=50_000)
stats = results["portfolio_statistics"]

print("Portfolio loss statistics")
for key in ["mean", "std_dev", "var_95", "var_99", "var_999", "expected_shortfall_99", "max_loss"]:
    print(f"  {key}: {stats[key]:,.0f}")

print("\nSector highlights")
for sector, sector_stats in results["sector_statistics"].items():
    print(f"  {sector}: 95% VaR = {sector_stats['var_95']:,.0f}")
```

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
