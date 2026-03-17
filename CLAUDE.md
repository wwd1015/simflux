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
  core/       - Backend (registry), SimulationEngine (Rust/NumPy dispatcher), BaseSimulator (ABC), SimulationConfig
  processes/  - GBM, CorrelatedGBM, TimeVaryingGBM, TimeVaryingCorrelatedGBM
  portfolio/  - TwoFactorPortfolio, AssetData, TwoFactorCorrelationStructure
  utils/      - ParquetResultsAnalyzer, StorageConfig, random/correlation utilities
src/          - Rust backend (gbm.rs, portfolio.rs, correlation.rs, storage.rs)
```

### Key Design Patterns
- **Strategy**: Backend class (centralized registry) selects Rust or NumPy backend via `Backend.is_available()`
- **Template Method**: BaseSimulator defines validation framework, subclasses specialize
- **Factory**: `AssetData.from_dataframe()`, `TwoFactorPortfolio.create_sample_portfolio()`
- **Lazy Loading**: ParquetResultsAnalyzer uses Polars lazy frames

### Class Hierarchy
```
BaseSimulator (ABC)
├── GBM                         # Single-asset GBM (Rust: simulate_gbm)
├── CorrelatedGBM               # Multi-asset with correlation matrix (Rust: simulate_gbm_multi)
├── TimeVaryingGBM              # Time-varying mu/sigma (Rust: simulate_gbm_tv)
├── TimeVaryingCorrelatedGBM    # Multi-asset time-varying (Rust: simulate_gbm_tv_multi)
└── TwoFactorPortfolio          # Credit portfolio (Rust: simulate_portfolio)

SimulationEngine                # NOT a BaseSimulator — standalone backend dispatcher
```

### Data Flow
```
User API → Simulator class → validate_inputs() → SimulationEngine
  → Backend.is_available() → Rust backend (fast) OR NumPy fallback → Results (ndarray or dict)
```

## Code Conventions
- Python 3.12+ required
- Dataclasses for configuration (SimulationConfig, StorageConfig, AssetData)
- Correlation matrices validated: symmetric, diagonal=1, values in [-1,1], positive definite
- All simulators inherit from BaseSimulator; SimulationEngine does NOT
- Use `Backend.is_available()` / `Backend.get_rust()` for Rust detection — never module-level flags
- `CORRELATION_TOLERANCE = 1e-8` shared constant for all validation (in `core/backend.py`)
- Tests mock via `@patch.object(Backend, 'is_available', ...)` and `@patch.object(Backend, 'get_rust')`

## Dual-Implementation & Cross-Validation
Every simulation type has both a Rust and NumPy implementation. Cross-validation tests in `tests/test_cross_validation.py` run both backends on the same problem and compare distributional statistics. This strategy catches bugs in either backend and ensures the NumPy fallback remains a reliable failsafe.

## Dependencies
- Python: numpy>=2.2, pandas>=3.0, polars>=1.38, pyarrow>=18.0, scipy (optional)
- Rust: pyo3, ndarray, nalgebra, statrs, rayon, arrow, parquet
