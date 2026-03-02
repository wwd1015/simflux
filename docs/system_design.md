# SimFlux System Design

## 1. Architecture Overview

SimFlux uses a hybrid Python/Rust architecture:
- **Python layer**: User-facing API, configuration, validation, analysis
- **Rust layer**: Performance-critical computation (GBM simulation, portfolio loss calculation)
- **Fallback**: Pure NumPy implementation when Rust backend is unavailable

### Component Diagram
```
┌─────────────────────────────────────────────────┐
│                  User Code                       │
├─────────────────────────────────────────────────┤
│  Public API: GBM, CorrelatedGBM, TimeVaryingGBM │
│              TwoFactorPortfolio                  │
├─────────────────────────────────────────────────┤
│  Config: SimulationConfig, StorageConfig, AssetData  │
├────────────────────┬────────────────────────────┤
│  SimulationEngine  │  Validation & Utilities     │
│  (Backend Router)  │  (Correlation, Random, etc.) │
├────────────────────┼────────────────────────────┤
│  Rust Backend      │  NumPy Fallback             │
│  (_rust module)    │  (Pure Python)              │
└────────────────────┴────────────────────────────┘
```

## 2. Module Responsibilities

### core/base.py
- `BaseSimulator`: Abstract base class defining simulator interface
- `SimulationConfig`: Seed, memory limits, reserved thread/batch settings
- `SimulationResults`: Result container with lazy `load_interim_data()` via `ParquetResultsAnalyzer`

### core/engine.py
- `SimulationEngine`: Routes simulation calls to Rust or NumPy
- Handles Cholesky decomposition for correlated simulations
- Input validation for GBM parameters and correlation matrices

### processes/gbm.py
- `GBM`: Single-asset Geometric Brownian Motion (dS = μS dt + σS dW)
- `CorrelatedGBM`: Multi-asset GBM with correlation structure
- Returns numpy arrays: `(n_paths, n_steps+1)` or `(n_paths, n_assets, n_steps+1)`

### processes/time_varying.py
- `TimeVaryingGBM`: Time-dependent μ(t) and σ(t) via linear interpolation
- `TimeVaryingCorrelatedGBM`: Per-asset time series with shared correlation
- Direct time series input: `mu_times/mu_values`, `sigma_times/sigma_values`

### portfolio/two_factor_model.py
- `AssetData`: Credit asset parameters (PD, LGD, exposure, sector)
- `TwoFactorPortfolio`: Merton two-factor credit model
  - Systematic factor = sector loading × sector factor + idio loading × noise
  - Default if asset value ≤ Φ⁻¹(PD)
  - LGD via correlated Beta distribution

### portfolio/correlation.py
- `TwoFactorCorrelationStructure`: Builds full asset correlation matrices
- Intra-sector and inter-sector correlation handling
- Factor loading computation

### utils/storage.py
- `StorageConfig`: Parquet storage configuration
- `ParquetResultsAnalyzer`: Lazy Polars-based analysis of simulation results

### utils/random_utils.py
- `set_seed()`: Set global random seed for reproducibility (legacy API, also exported at top level)
- Correlation matrix generation, validation, and correction
- Block correlation matrices, factor-based correlation

## 3. Backend Selection Strategy

```python
try:
    from simflux import _rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
```

SimulationEngine checks `RUST_AVAILABLE` at runtime:
- **Rust available**: Calls `_rust.simulate_gbm()`, `_rust.simulate_gbm_multi()`, etc.
- **Rust unavailable**: Falls back to NumPy with `warnings.warn()`

## 4. Simulation Models

### GBM (Geometric Brownian Motion)
```
dS = μS dt + σS dW
S(t+dt) = S(t) × exp((μ - σ²/2)dt + σ√dt × Z)
```

### Correlated GBM
- Cholesky decomposition of correlation matrix: L = chol(Σ)
- Correlated normals: Z_corr = L × Z_independent

### Two-Factor Portfolio Loss
```
For each asset i in sector s:
  asset_value = √ρ_s × sector_factor_s + √(1-ρ_s) × ε_i
  default_i = (asset_value ≤ Φ⁻¹(PD_i))
  loss_i = default_i × LGD_i × exposure_i

Portfolio loss = Σ loss_i
```

## 5. Known Issues & Technical Debt

### Incomplete Features
- `n_threads` and `batch_size` config parameters not utilized (reserved for future use)
- HDF5 storage format accepted in config but not implemented

### Edge Cases
- Portfolio fallback: Beta distribution parameters can go negative (patched with `max(0.1, ...)`)
- Zero-exposure assets not explicitly handled

## 6. Performance Optimizations

The following optimizations reduce Python overhead in simulation hot paths:

### Vectorized operations
- **`norm_ppf` fallback**: Accepts and returns arrays using `np.where` and vectorized polynomial math (no per-element list comprehension)
- **`_linear_interpolate`**: Delegates to `np.interp` (C-level) instead of manual binary search
- **`TimeVaryingCorrelatedGBM.simulate`**: Inner asset loop replaced with single broadcasting expression over `(n_paths, n_assets)`
- **`erf` dispatch**: Uses `scipy.special.erf` (C-level) when available; falls back to `np.vectorize(math.erf)`

### Modern NumPy random API
All NumPy fallback paths use `np.random.default_rng(seed)` instead of the legacy `np.random.seed()` / `np.random.normal()` global state API. Affected modules: `engine.py`, `time_varying.py`, `two_factor_model.py`, `random_utils.py`. The `set_seed()` utility retains the legacy API for backward compatibility.
