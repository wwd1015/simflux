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
| **Rust** | Production | **~2.4–49x faster** | lower to comparable | Binary wheel |
| **NumPy Fallback** | Development | Baseline | Baseline | No compilation needed |

### Benchmark Results Summary

Measured on SimFlux 0.7.0 (4-core x86_64 Linux, Python 3.12, release build;
ratios are representative — absolute times are hardware-dependent):

- **GBM**: ~3–5x faster, ~4x less memory.
- **Correlated GBM**: ~2.4–7x faster, memory at parity.
- **Portfolio**: **~30–49x faster**; memory is a small size-independent
  constant (~15 MB thread pool + per-trial summaries), while the chunked
  scipy fallback peaks under ~2 MB — neither backend allocates a dense
  trials × assets grid.

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

A **portfolio is a list of obligors** (`AssetData`). One Monte Carlo *path* draws
the shared sector factors once, then each obligor's idiosyncratic factor, so
defaults are decided *jointly* within a path — correlated through the sectors
obligors share. `n_simulations` is the number of such independent paths; each path
simulates the whole portfolio over `n_periods` steps. **Single-period is just
`n_periods=1`** (the default); the same call generalizes to multiple periods.

```python
import simflux as sf

# 1) Build the obligors explicitly...
assets = [
    sf.AssetData(
        asset_id=i,
        sector_id=0,
        pd=0.02,                 # 1-period probability of default
        lgd_mean=0.6,            # mean loss given default
        lgd_std=0.15,            # LGD dispersion (Beta)
        exposure=1_000_000,      # exposure at default
        sector_name="Technology",
    )
    for i in range(100)
]

# ...or from a DataFrame with columns asset_id, sector, pd, lgd_mean, lgd_std, exposure:
#   assets = sf.AssetData.from_dataframe(df)
# ...or skip construction and use a generated sample portfolio:
#   portfolio = sf.CreditPortfolio.create_sample_portfolio(
#       n_assets_per_sector=100, sectors=["Technology", "Finance", "Healthcare"],
#       intra_sector_correlations=0.4, inter_sector_correlation=0.2)

portfolio = sf.CreditPortfolio(
    assets,
    intra_sector_correlations=0.4,    # share of an obligor's variance from its sector factor
    systematic_lgd_correlation=0.3,   # wrong-way risk: positive => LGD rises in downturns
    # sector_correlation_matrix=...,  # np.ndarray, optional; defaults to identity
)

results = portfolio.simulate(n_simulations=100_000)   # single period (n_periods=1)
print(results["portfolio_statistics"])  # mean, var_95/99/999, expected_shortfall, ...
```

**Multiple periods.** Give each obligor a cumulative-PD term structure and step
through time — same method, just `n_periods > 1`:

```python
# Cumulative PD by the end of each quarter (>= n_periods long), ending at the horizon PD.
tech_curve = [0.008, 0.016, 0.024, 0.031, 0.038, 0.045, 0.052, 0.060]
assets = [
    sf.AssetData(
        asset_id=i, sector_id=0, pd=0.06,
        lgd_mean=0.5, lgd_std=0.1, exposure=1_000_000,
        sector_name="Tech", pd_term_structure=tech_curve,
    )
    for i in range(50)
]
portfolio = sf.CreditPortfolio(assets, intra_sector_correlations=0.2)

results = portfolio.simulate(
    n_simulations=100_000,
    n_periods=8,              # 8 quarters
    period_length=0.25,       # each quarter = 0.25 years
    default_timing="copula",  # or "frailty"; the two coincide at n_periods=1
)
print(results["portfolio_statistics"]["var_99"])
```

`default_timing` also accepts a timing object, which is where mode-specific
configuration lives — `sf.Copula()` or `sf.Frailty(persistence=0.6)` (the annual
autocorrelation of the credit-cycle factor, validated at construction). The
strings are sugar for default-configured objects; the old `factor_persistence`
keyword still works alongside the strings but is deprecated.

The result keys are identical whether the Rust backend or the NumPy fallback ran:
`portfolio_statistics`, `sector_statistics`, `n_trials`, `n_assets`, `n_sectors`,
`sector_names`, `n_periods`, `period_length`, `time_horizon`, `default_timing`,
`factor_persistence` (`analyzer` is added only when interim results are stored —
see below). A `pd_term_structure` **longer**
than `n_periods` runs a sub-horizon (the first `n_periods` points are used); a
structure **shorter** than `n_periods` raises `RuntimeError`; with no term
structure the flat `pd` is spread across periods at a constant hazard rate.

**Heterogeneous and time-varying obligors.** `AssetData` accepts two optional
per-obligor curves beyond `pd_term_structure`: `intra_sector_correlation` (this
obligor's own sector loading `rho_i`, so obligors in a sector may load
differently — it falls back to the sector value when unset), and
`lgd_term_structure` (a per-period LGD *mean* — an obligor defaulting in period
`k` draws LGD from the Beta with that period's mean, `lgd_std` constant). A
constant `lgd_term_structure` reduces exactly to the flat `lgd_mean`.

**Per-default detail.** Store interim results to Parquet to recover which obligor
defaulted, in which period, its `time_to_default`, and the factors at default:

```python
storage = sf.StorageConfig(store_interim=True, output_path="results.parquet")
results = portfolio.simulate(n_simulations=100_000, storage_config=storage)

analyzer = sf.ParquetResultsAnalyzer("results.parquet")
high_loss_trials = analyzer.query_high_loss_trials(percentile=99)
sector_analysis = analyzer.analyze_by_sector()
```

## Methodology

For a detailed explanation of the simulation models, mathematical foundations, and parameter guidance, see the **[Methodology Document](docs/methodology.md)**.

Topics covered:
- GBM discretization and Cholesky correlation
- Two-factor Merton credit framework (asset value model, default mechanism, stochastic LGD)
- Multi-period extension with conditional PD derivation
- Convergence guidance for Monte Carlo simulations
- Comparison with industry models (RiskFrontier, CreditMetrics, Basel IRB)

More documentation lives in [`docs/`](docs/):
- [User Guide](docs/user_guide.md) — task-oriented walkthroughs of every simulator
- [System Design](docs/system_design.md) — module responsibilities and the backend seams
- [Architecture Comparison](docs/architecture_comparison.md) — Rust backend vs NumPy fallback, flow by flow
- [Performance Benchmarks](docs/performance_benchmarks.md) — measured speedups and memory
- [Deployment Guide](docs/deployment_guide.md) — building and shipping wheels

## Installation from Pre-Built Wheels

Pre-built wheels with the compiled Rust backend are attached to [GitHub Releases](../../releases). Available for:
- **Linux** (x86_64, manylinux)
- **macOS** (Apple Silicon + Intel)
- **Windows** (x86_64)
- **Python 3.12 and 3.13**

```bash
# Install directly from a GitHub release (use the latest tag)
pip install simflux --find-links https://github.com/wwd1015/simflux/releases/download/v0.5.0/

# Or download the .whl file for your platform and install locally.
# Wheels are abi3 (cp312-abi3): one wheel per platform, installs on Python 3.12+.
pip install simflux-0.5.0-cp312-abi3-manylinux_2_17_x86_64.whl
```

No Rust toolchain required when using pre-built wheels.

## License

MIT License