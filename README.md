# SimFlux

High-performance financial simulation library with Rust backend for portfolio risk modeling and stochastic process simulation.

## Features

- **Correlated GBM Simulation**: Multi-series Geometric Brownian Motion with correlation structures
- **Time-Varying Parameters**: GBM with time-dependent drift and volatility via direct time series input
- **Portfolio Loss Modeling**: Two-factor Merton framework similar to Moody's RiskFrontier
- **High Performance**: Rust backend for critical computational paths, vectorized NumPy fallback
- **Flexible Storage**: Optional Parquet-based interim results for detailed analysis
- **Scalable**: Handle thousands of assets with efficient memory management

## Requirements

- **Python**: 3.12+ (Latest Python versions for optimal performance and modern features)
- **Operating System**: Windows, macOS, Linux
- **Optional**: Rust toolchain (only for building from source)

## Installation

### From Binary Wheels (Recommended)
```bash
# From PyPI or internal artifactory
pip install simflux

# From internal artifactory
pip install -i https://your-artifactory/pypi/local simflux
```

### From Source (Requires Rust)
```bash
# Install Rust toolchain first
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install from source
pip install maturin
maturin develop --release
```

**Note**: Binary wheels include pre-compiled Rust code and don't require Rust installation. If binary wheels are not available, the package will automatically use NumPy fallback implementations (slower but functional).

## Performance Comparison

| Backend | Use Case | Performance | Memory Usage | Installation |
|---------|----------|-------------|--------------|--------------|
| **Rust** | Production | **~2.6–45x faster** | comparable to **much lower** | Binary wheel |
| **NumPy Fallback** | Development | Baseline | Baseline | No compilation needed |

### Benchmark Results Summary

Measured on Apple Silicon, release build (ratios are representative; absolute
times are hardware-dependent):

- **GBM**: ~2.6–3.8x faster, ~0.4x the memory.
- **Correlated GBM**: ~3–5x faster, memory at parity.
- **Portfolio**: ~10–45x faster, **~8–25x less** memory (Rust streams per-trial;
  the fallback reduces in bounded chunks).

*See [`docs/performance_benchmarks.md`](docs/performance_benchmarks.md) for the
full tables and methodology, and run `python benchmarks/performance_comparison.py`
to reproduce.*

## Key Concepts

### Simulation Parameters

All simulators share these common parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `mu` | **Drift** — annualized expected return (e.g., 0.05 = 5% per year) | `mu=0.05` |
| `sigma` | **Volatility** — annualized standard deviation of returns (e.g., 0.2 = 20%) | `sigma=0.2` |
| `S0` | **Initial price** of the asset | `S0=100` |
| `n_paths` | Number of independent simulation runs (more = better statistics) | `n_paths=1000` |
| `n_steps` | Number of discrete time steps (controls time grid resolution) | `n_steps=252` |
| `T` | **Time horizon** in years (e.g., 1.0 = one year, 0.25 = one quarter) | `T=1.0` |

**Why both `n_steps` and `T`?** Because `mu` and `sigma` are in annualized units (industry convention), the simulation needs to know the real time span to scale them correctly. The step size is `dt = T / n_steps`. For example:
- `T=1.0, n_steps=252` → daily steps over 1 year (`dt ≈ 0.004 years`)
- `T=1.0, n_steps=12` → monthly steps over 1 year (`dt ≈ 0.083 years`)
- `T=5.0, n_steps=60` → monthly steps over 5 years

### Correlation Matrix

For multi-asset simulations, you provide a correlation matrix describing how asset returns move together. Values range from -1 (perfectly opposite) to 1 (perfectly together):

```python
# Two assets with 30% correlation
correlation_matrix = np.array([
    [1.0, 0.3],
    [0.3, 1.0]
])
```

## Quick Start

### GBM Simulation

```python
import simflux as sf
import numpy as np

# Single asset: simulate 1000 paths of daily prices over 1 year
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)
# paths.shape = (1000, 253) — each row is one price path

# Multiple correlated assets
correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
gbm_multi = sf.CorrelatedGBM(
    mu=[0.05, 0.03],       # annualized drift per asset
    sigma=[0.2, 0.15],     # annualized volatility per asset
    S0=[100, 50],           # starting prices
    correlation_matrix=correlation_matrix
)
paths = gbm_multi.simulate(n_paths=1000, n_steps=252, T=1.0)
# paths.shape = (1000, 2, 253) — 1000 paths, 2 assets, 253 time points
```

### Time-Varying GBM

Use this when drift or volatility changes over time (e.g., a volatility spike mid-year). You provide time-value pairs, and parameters are linearly interpolated between them.

```python
# Volatility spikes from 20% to 30% at the 6-month mark, then settles to 15%
tv_gbm = sf.TimeVaryingGBM(
    mu_times=[0, 0.5, 1.0],      # time points (in years)
    mu_values=[0.05, 0.08, 0.03], # drift at each time point
    sigma_times=[0, 0.5, 1.0],
    sigma_values=[0.2, 0.3, 0.15],
    S0=100
)
paths = tv_gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

# For constant parameters, a single point is enough (value is held flat)
tv_gbm_flat = sf.TimeVaryingGBM(
    mu_times=[0], mu_values=[0.05],
    sigma_times=[0], sigma_values=[0.2],
    S0=100
)
```

### Portfolio Loss Simulation

Simulate credit portfolio losses using a two-factor Merton framework. Each asset can default based on its probability of default (PD), with losses determined by loss given default (LGD).

```python
# Quick start with sample portfolio
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
    n_assets_per_sector=100,
    sectors=['Technology', 'Finance', 'Healthcare'],
    intra_sector_correlations=0.4,   # how correlated assets are within a sector
    inter_sector_correlation=0.2,     # how correlated sectors are with each other
    systematic_lgd_correlation=0.3,   # how loss severity correlates with market stress
)

# Or build from asset data
portfolio = sf.TwoFactorPortfolio(
    assets=asset_data,  # List[AssetData] or DataFrame with pd, lgd_mean, lgd_std, exposure, sector
    intra_sector_correlations={'Technology': 0.5, 'Finance': 0.4},
    sector_correlation_matrix=sector_corr_matrix,  # np.ndarray, optional
    systematic_lgd_correlation=0.3,
)

# Run 100k Monte Carlo scenarios
results = portfolio.simulate(n_simulations=100000)
print(results['portfolio_statistics'])  # mean loss, VaR, expected shortfall, etc.

# The result keys are identical whether the Rust backend or the NumPy fallback ran:
#   portfolio_statistics, sector_statistics, n_trials, n_assets, n_sectors,
#   sector_names, n_periods, period_length, time_horizon
# ('analyzer' is added only when interim results are stored — see below)

# With interim results saved to disk for detailed analysis.
# StorageConfig exposes the fields the writer honors: store_interim, output_path, batch_size.
storage_config = sf.StorageConfig(
    store_interim=True,
    output_path="simulation_results.parquet",
)
results = portfolio.simulate(
    n_simulations=100000,
    storage_config=storage_config
)

# Analyze results
analyzer = sf.ParquetResultsAnalyzer("simulation_results.parquet")
high_loss_trials = analyzer.query_high_loss_trials(percentile=99)
sector_analysis = analyzer.analyze_by_sector()
```

### Multi-Period Portfolio Simulation

Simulate losses over multiple periods with quarterly PD term structures and default timing:

```python
# Define assets with cumulative PD term structure (quarterly over 2 years)
asset = sf.AssetData(
    asset_id=0, sector_id=0, pd=0.06,
    lgd_mean=0.5, lgd_std=0.1, exposure=1_000_000,
    sector_name="Tech",
    pd_term_structure=[0.01, 0.025, 0.04, 0.06],  # cumulative PDs at Q1–Q4
)

# Simulate with quarterly time steps
results = portfolio.simulate(
    n_simulations=100000,
    n_periods=8,          # 8 quarters
    period_length=0.25,   # each quarter = 0.25 years
)
# Results include time_to_default for each asset in each trial
```

When no `pd_term_structure` is provided, the flat `pd` is spread across periods assuming a constant hazard rate.

## Methodology

For a detailed explanation of the simulation models, mathematical foundations, and parameter guidance, see the **[Methodology Document](docs/methodology.md)**.

Topics covered:
- GBM discretization and Cholesky correlation
- Two-factor Merton credit framework (asset value model, default mechanism, stochastic LGD)
- Multi-period extension with conditional PD derivation
- Convergence guidance for Monte Carlo simulations
- Comparison with industry models (RiskFrontier, CreditMetrics, Basel IRB)

## Installation from Pre-Built Wheels

Pre-built wheels with the compiled Rust backend are attached to [GitHub Releases](../../releases). Available for:
- **Linux** (x86_64, manylinux)
- **macOS** (Apple Silicon + Intel)
- **Windows** (x86_64)
- **Python 3.12 and 3.13**

```bash
# Install directly from a GitHub release (replace v0.1.0 with the actual tag)
pip install simflux --find-links https://github.com/wwd1015/simflux/releases/download/v0.1.0/

# Or download the .whl file for your platform and install locally
pip install simflux-0.1.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

No Rust toolchain required when using pre-built wheels.

## License

MIT License