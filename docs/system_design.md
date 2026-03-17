# SimFlux System Design

## 1. Architecture Overview

SimFlux uses a hybrid Python/Rust architecture:
- **Python layer**: User-facing API, configuration, validation, analysis
- **Rust layer**: Performance-critical computation (GBM simulation, time-varying GBM, portfolio loss calculation, Parquet storage)
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
├────────────────────┤                             │
│  Backend Registry  │  CORRELATION_TOLERANCE      │
│  (core/backend.py) │  (shared constant)          │
├────────────────────┼────────────────────────────┤
│  Rust Backend      │  NumPy Fallback             │
│  (_rust module)    │  (Pure Python)              │
└────────────────────┴────────────────────────────┘
```

## 2. Module Responsibilities

### core/backend.py
- `Backend`: Centralized registry for Rust extension availability — replaces per-module `RUST_AVAILABLE` flags
- `CORRELATION_TOLERANCE = 1e-8`: Single tolerance constant used by all correlation validation across Python and Rust
- Methods: `is_available()`, `get_rust()`, `force(bool)` (testing override)

### core/base.py
- `BaseSimulator`: Abstract base class defining simulator interface (`simulate`, `validate_inputs`)
- `SimulationConfig`: Seed, batch_size, memory_limit_gb, progress_callback
- `SimulationResults`: Result container with lazy `load_interim_data()` via `ParquetResultsAnalyzer`

### core/engine.py
- `SimulationEngine`: Routes simulation calls to Rust or NumPy — **not** a `BaseSimulator` subclass
- Entry-points: `simulate_gbm()`, `simulate_gbm_correlated()`, `simulate_gbm_time_varying()`, `simulate_gbm_time_varying_correlated()`
- Handles Cholesky decomposition for correlated simulations (NumPy fallback path)
- Centralized input validation — simulator classes delegate to the engine

### processes/gbm.py
- `GBM`: Single-asset Geometric Brownian Motion (dS = μS dt + σS dW)
- `CorrelatedGBM`: Multi-asset GBM with correlation structure
- Returns numpy arrays: `(n_paths, n_steps+1)` or `(n_paths, n_assets, n_steps+1)`

### processes/time_varying.py
- `TimeVaryingGBM`: Time-dependent μ(t) and σ(t) via linear interpolation — **delegates to Rust backend** (`simulate_gbm_tv`)
- `TimeVaryingCorrelatedGBM`: Per-asset time series with shared correlation — **delegates to Rust backend** (`simulate_gbm_tv_multi`)
- Direct time series input: `mu_times/mu_values`, `sigma_times/sigma_values`
- NumPy fallback preserved transparently when Rust unavailable

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
- `StorageConfig`: Parquet storage configuration (HDF5 format accepted in config but not implemented)
- `ParquetResultsAnalyzer`: Lazy Polars-based analysis of simulation results
- **Rust storage is fully functional**: Portfolio simulation writes interim results to Parquet via the Rust `ArrowWriter` backend

### utils/random_utils.py
- `set_seed()`: Set global random seed for reproducibility (legacy API, also exported at top level)
- Correlation matrix generation, validation, and correction — all using `CORRELATION_TOLERANCE`
- Block correlation matrices, factor-based correlation

## 3. Backend Selection Strategy

```python
from simflux.core.backend import Backend

# Centralized check — used by SimulationEngine, TwoFactorPortfolio, etc.
if Backend.is_available():
    _rust = Backend.get_rust()
    # Calls _rust.simulate_gbm(), _rust.simulate_gbm_tv(), etc.
else:
    # Falls back to NumPy with warnings.warn()
```

### Rust Functions Exposed to Python

| Python name | Rust function | Purpose |
|-------------|---------------|---------|
| `simulate_gbm` | `simulate_gbm_single` | Single-asset GBM |
| `simulate_gbm_multi` | `simulate_gbm_correlated` | Multi-asset correlated GBM |
| `simulate_gbm_tv` | `simulate_gbm_time_varying_single` | Time-varying single-asset GBM |
| `simulate_gbm_tv_multi` | `simulate_gbm_time_varying_correlated` | Time-varying correlated GBM |
| `simulate_portfolio` | `simulate_portfolio_losses` | Two-factor portfolio loss simulation |

### Testing Backend Availability

```python
# In tests, force a specific backend:
from simflux.core.backend import Backend
Backend.force(False)  # force NumPy fallback
Backend.force(True)   # force Rust (will fail if not compiled)
```

## 4. Simulation Models

### GBM (Geometric Brownian Motion)
```
dS = μS dt + σS dW
S(t+dt) = S(t) × exp((μ - σ²/2)dt + σ√dt × Z)
```

### Correlated GBM
- Cholesky decomposition of correlation matrix: L = chol(Σ)
- Correlated normals: Z_corr = L × Z_independent

### Time-Varying GBM
- Parameters μ(t) and σ(t) provided as (time, value) pairs
- Linear interpolation with boundary clamping (via `np.interp` in Python, binary search in Rust)
- Per-step drift and volatility: `drift_i = (μ(t_i) - σ(t_i)²/2) × dt`

### Two-Factor Portfolio Loss
```
For each asset i in sector s:
  asset_value = √ρ_s × sector_factor_s + √(1-ρ_s) × ε_i
  default_i = (asset_value ≤ Φ⁻¹(PD_i))
  loss_i = default_i × LGD_i × exposure_i

Portfolio loss = Σ loss_i
```

## 5. Dual-Implementation Strategy & Cross-Validation

Every simulation path (GBM, correlated GBM, time-varying GBM, portfolio loss) has **two independent implementations**: a Rust backend for production performance and a vectorised NumPy fallback that serves as both a failsafe and a correctness reference.

### Why both?

1. **Failsafe**: If the Rust extension isn't compiled or installed, users get identical functionality via NumPy — just slower.
2. **Accuracy verification**: The NumPy code is pure Python and much easier to audit than Rust/PyO3. Cross-validation tests run both backends on the same problem and verify that distributional statistics (mean, VaR, correlation) converge.
3. **Development velocity**: New features can be prototyped in Python first, then ported to Rust once the math is validated.

### Cross-validation tests (`tests/test_cross_validation.py`)

| Simulation type | What's compared |
|----------------|-----------------|
| Single-asset GBM | Log-return mean & std vs theoretical values |
| Correlated GBM | Realized correlation matrix |
| Time-varying GBM (single) | Log-return mean with constant params |
| Time-varying GBM (correlated) | Realized cross-asset correlation |
| Portfolio loss (single-period) | Mean portfolio loss within 30% tolerance |
| Portfolio loss (multi-period + term structure) | Mean portfolio loss within 30% tolerance |

The 30% tolerance accounts for Monte Carlo noise with 10k simulations and different RNG streams. These tests caught a real bug during development: the `norm_ppf` approximation only implemented the tail branch of Acklam's algorithm, producing 10x errors for central-region PDs like 0.06.

## 6. Known Issues & Technical Debt

### Incomplete Features
- HDF5 storage format accepted in config but not implemented
- `progress_callback` on `SimulationConfig` is wired but not invoked during simulation loops

### Edge Cases
- Portfolio fallback: Beta distribution parameters can go negative (patched with `max(0.1, ...)`)
- Zero-exposure assets not explicitly handled

## 6. Performance Optimizations

The following optimizations reduce Python overhead in simulation hot paths:

### Rust backend for all simulation types
- **GBM**: Rayon-parallelized across paths
- **Correlated GBM**: Pre-generated correlated normals + parallel path generation
- **Time-varying GBM**: Pre-computed parameter interpolation + parallel paths (new)
- **Portfolio**: Parallel across trials via Rayon

### Vectorized NumPy fallback
- **`norm_ppf` fallback**: Accepts and returns arrays using `np.where` and vectorized polynomial math
- **`_linear_interpolate`**: Delegates to `np.interp` (C-level) instead of manual binary search
- **`TimeVaryingCorrelatedGBM.simulate`**: Broadcasting over `(n_paths, n_assets)` in fallback mode
- **`erf` dispatch**: Uses `scipy.special.erf` (C-level) when available; falls back to `np.vectorize(math.erf)`

### Modern NumPy random API
All NumPy fallback paths use `np.random.default_rng(seed)` instead of the legacy `np.random.seed()` / `np.random.normal()` global state API. Affected modules: `engine.py`, `time_varying.py`, `two_factor_model.py`, `random_utils.py`. The `set_seed()` utility retains the legacy API for backward compatibility.
