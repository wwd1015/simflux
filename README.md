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
| **Rust** | Production | **3-14x faster** | **3-5x more efficient** | Binary wheel |
| **NumPy Fallback** | Development | Baseline | Higher usage | No compilation needed |

### Benchmark Results Summary

- **Average Speedup**: 7.2x faster with Rust backend
- **Maximum Speedup**: Up to 13.9x for correlated simulations  
- **Memory Efficiency**: 3.6x more memory efficient on average
- **Throughput**: 28M+ operations/second vs 6M ops/sec (NumPy)

*See [benchmarks/](benchmarks/) directory for detailed performance analysis.*

## Quick Start

### Correlated GBM Simulation

```python
import simflux as sf
import numpy as np

# Single series
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0)

# Multiple correlated series
correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
gbm_multi = sf.CorrelatedGBM(
    mu=[0.05, 0.03],
    sigma=[0.2, 0.15], 
    S0=[100, 50],
    correlation_matrix=correlation_matrix
)
paths = gbm_multi.simulate(n_paths=1000, n_steps=252, T=1.0)
```

### Time-Varying GBM

```python
# Single asset with time-dependent parameters
tv_gbm = sf.TimeVaryingGBM(
    mu_times=[0, 0.5, 1.0], mu_values=[0.05, 0.08, 0.03],
    sigma_times=[0, 0.5, 1.0], sigma_values=[0.2, 0.3, 0.15],
    S0=100
)
paths = tv_gbm.simulate(n_paths=1000, n_steps=252, T=1.0)
```

### Portfolio Loss Simulation

```python
# Quick start with sample portfolio
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
    n_assets_per_sector=100,
    sectors=['Technology', 'Finance', 'Healthcare'],
    intra_sector_correlations=0.4,
    inter_sector_correlation=0.2,
    systematic_lgd_correlation=0.3,
)

# Or build from asset data
portfolio = sf.TwoFactorPortfolio(
    assets=asset_data,  # List[AssetData] or DataFrame with pd, lgd_mean, lgd_std, exposure, sector
    intra_sector_correlations={'Technology': 0.5, 'Finance': 0.4},
    sector_correlation_matrix=sector_corr_matrix,  # np.ndarray, optional
    systematic_lgd_correlation=0.3,
)

# Fast simulation (summary only)
results = portfolio.simulate(n_simulations=100000)

# Detailed simulation with interim results
storage_config = sf.StorageConfig(
    store_interim=True,
    output_path="simulation_results.parquet"
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

## License

MIT License