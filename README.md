# SimFlux

High-performance financial simulation library with Rust backend for portfolio risk modeling and stochastic process simulation.

## Features

- **Correlated GBM Simulation**: Multi-series Geometric Brownian Motion with correlation structures
- **Portfolio Loss Modeling**: Two-factor Merton framework similar to Moody's RiskFrontier
- **High Performance**: Rust backend for critical computational paths
- **Flexible Storage**: Optional Parquet-based interim results for detailed analysis
- **Scalable**: Handle thousands of assets with efficient memory management

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

### Portfolio Loss Simulation

```python
# Define portfolio
portfolio = sf.TwoFactorPortfolio(
    assets=asset_data,  # DataFrame with PD, LGD parameters
    sector_mapping=sector_map,
    inter_sector_correlation=0.2,
    intra_sector_correlation=0.4
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