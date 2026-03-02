# SimFlux - Development Guide

## Project Overview
SimFlux is a hybrid Python/Rust financial simulation library for Monte Carlo simulations. It features GBM (Geometric Brownian Motion) simulation, time-varying parameters, and two-factor credit portfolio loss modeling.

## Build & Test Commands
```bash
maturin develop --release    # Build Rust extension
pytest tests/                # Run all tests
pytest tests/test_gbm.py -v # Run specific test file
make test                    # Via Makefile
make build                   # Build release
```

## Architecture

### Module Layout
```
python/simflux/
  core/       - BaseSimulator (ABC), SimulationEngine (Rust/NumPy backend), SimulationConfig
  processes/  - GBM, CorrelatedGBM, TimeVaryingGBM, TimeVaryingCorrelatedGBM
  portfolio/  - TwoFactorPortfolio, AssetData, TwoFactorCorrelationStructure
  utils/      - ParquetResultsAnalyzer, StorageConfig, random/correlation utilities
src/          - Rust backend (gbm.rs, portfolio.rs, correlation.rs, storage.rs)
```

### Key Design Patterns
- **Strategy**: SimulationEngine selects Rust or NumPy backend via `RUST_AVAILABLE` flag
- **Template Method**: BaseSimulator defines validation framework, subclasses specialize
- **Factory**: `AssetData.from_dataframe()`, `TwoFactorPortfolio.create_sample_portfolio()`
- **Lazy Loading**: ParquetResultsAnalyzer uses Polars lazy frames

### Class Hierarchy
```
BaseSimulator (ABC)
├── GBM                         # Single-asset GBM
├── CorrelatedGBM               # Multi-asset with correlation matrix
├── TimeVaryingGBM              # Time-varying mu/sigma
├── TimeVaryingCorrelatedGBM    # Multi-asset time-varying
└── TwoFactorPortfolio          # Credit portfolio (Merton framework)
```

### Data Flow
```
User API → Simulator class → validate_inputs() → SimulationEngine
  → Rust backend (fast) OR NumPy fallback → Results (ndarray or dict)
```

## Code Conventions
- Python 3.12+ required
- Dataclasses for configuration (SimulationConfig, StorageConfig, AssetData)
- Correlation matrices validated: symmetric, diagonal=1, values in [-1,1], positive definite
- All simulators inherit from BaseSimulator
- Tests use unittest.mock to mock Rust availability

## Dependencies
- Python: numpy>=2.2, pandas>=3.0, polars>=1.38, pyarrow>=18.0, scipy (optional)
- Rust: pyo3, ndarray, nalgebra, statrs, rayon, arrow, parquet
