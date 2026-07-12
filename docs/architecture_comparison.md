# SimFlux Architecture Comparison: Rust vs NumPy Fallback

## Overview

SimFlux provides two execution modes:
1. **Rust Backend** (Recommended) - High-performance compiled code
2. **NumPy Fallback** - Pure Python implementation for compatibility

## Function Flow Comparison

### 1. Geometric Brownian Motion (GBM) Simulation

#### Rust Backend Flow
```mermaid
graph TD
    A[Python GBM.simulate()] --> B[SimulationEngine._rust_simulate_gbm()]
    B --> C[PyO3 Bindings]
    C --> D[Rust: simulate_gbm_single()]
    D --> E[Rust: ThreadSafeRng parallel generation]
    E --> F[Rust: Rayon parallel paths]
    F --> G[Native memory allocation]
    G --> H[PyO3 conversion to NumPy]
    H --> I[Return to Python]
```

**Key Features:**
- ✅ **Multi-threaded**: Uses Rayon for parallel path generation
- ✅ **Memory efficient**: Direct memory allocation in Rust
- ✅ **SIMD optimized**: Compiler vectorization
- ✅ **Zero-copy**: Minimal data conversion overhead

#### NumPy Fallback Flow
```mermaid
graph TD
    A[Python GBM.simulate()] --> B[SimulationEngine._numpy_simulate_gbm()]
    B --> C[NumPy random.normal()]
    C --> D[NumPy vectorized operations]
    D --> E[Python memory allocation]
    E --> F[NumPy cumsum & exp]
    F --> G[Return NumPy array]
```

**Key Features:**
- ⚠️ **Single-threaded**: Limited to NumPy's threading
- ⚠️ **Higher memory usage**: Python object overhead
- ✅ **Vectorized**: NumPy optimized operations
- ⚠️ **GIL limitations**: Python Global Interpreter Lock

### 2. Correlated GBM Simulation

#### Rust Backend Flow
```mermaid
graph TD
    A[Python CorrelatedGBM.simulate()] --> B[Rust correlation validation]
    B --> C[Nalgebra Cholesky decomposition]
    C --> D[Parallel correlated random generation]
    D --> E[Multi-asset parallel simulation]
    E --> F[Memory-efficient 3D array]
    F --> G[Return to Python]
```

**Rust Implementation Details:**
```rust
// Efficient correlation matrix operations
let cholesky = Cholesky::new(correlation_matrix)?;

// Parallel random number generation
(0..n_paths).into_par_iter().map(|path_idx| {
    let correlated_randoms = generate_correlated_normals_batch();
    simulate_asset_paths(correlated_randoms)
}).collect()
```

#### NumPy Fallback Flow
```mermaid
graph TD
    A[Python CorrelatedGBM.simulate()] --> B[NumPy Cholesky decomposition]
    B --> C[Sequential random generation]
    C --> D[Matrix multiplication for correlation]
    D --> E[Nested loops for time steps]
    E --> F[Python memory allocation]
    F --> G[Return 3D NumPy array]
```

**NumPy Implementation Details:**
```python
# Cholesky decomposition with fallback
try:
    L = np.linalg.cholesky(correlation_matrix)
except np.linalg.LinAlgError:
    # Eigenvalue decomposition fallback
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 1e-8)
    L = eigenvecs @ np.diag(np.sqrt(eigenvals))

# Sequential simulation (slower)
for step in range(n_steps):
    independent_randoms = np.random.normal(0, 1, (n_paths, n_assets))
    correlated_randoms = independent_randoms @ L.T
```

### 3. Portfolio Loss Simulation

Both backends consume the same inputs: `simulate()` assembles one
`PortfolioInputs` (per-obligor arrays, the validated sector correlation matrix,
run parameters) and one **timing plan** — the per-period, per-obligor threshold
matrix derived once in Python by the chosen default-timing model (`Copula` /
`Frailty`). Neither backend derives default thresholds itself, so threshold
parity across backends holds by construction.

#### Rust Backend Flow
```mermaid
graph TD
    A[Python CreditPortfolio.simulate()] --> P[PortfolioInputs + TimingPlan]
    P --> B[Rust PortfolioConfig + thresholds]
    B --> C[Two-factor correlation structure]
    C --> D[Parallel systematic factor generation]
    D --> E[Parallel trial simulation vs plan thresholds]
    E --> F[Beta distribution sampling]
    F --> G[Portfolio aggregation]
    G --> H[Parquet storage (optional)]
    H --> I[Statistical calculations]
    I --> J[Return results]
```

**Rust Implementation Highlights:**
```rust
// Per-asset constants (sector index, loadings, LGD Beta parameters) are
// precomputed once per run; the trial loop only samples, compares against
// the plan's thresholds, and accumulates (excerpt)
let sector_factor = factors[pc.sector_index];
let idio = sample_standard_normal(&mut rng);
let value = pc.sector_loading * sector_factor + pc.idio_loading * idio;

// Parallel trial simulation
let trial_results: Vec<TrialResult> = (0..n_simulations)
    .into_par_iter()
    .map(|trial_id| simulate_single_trial(...))
    .collect();
```

#### NumPy Fallback Flow
```mermaid
graph TD
    A[Python CreditPortfolio.simulate()] --> P[PortfolioInputs + TimingPlan]
    P --> B[Vectorized chunked simulation]
    B --> C[Cholesky-correlated sector factors per chunk]
    C --> D[Latents vs plan thresholds, whole chunk at once]
    D --> E[Exact Beta LGD via scipy, tabulated fallback without it]
    E --> F[On-the-fly reduction to total + per-sector losses]
    F --> G[Same statistics, same result keys as Rust]
```

**What the fallback actually is** (`portfolio/numpy_simulation.py`): a
vectorized implementation that walks trials in bounded chunks (dense scratch is
`O(chunk × n_assets)`, retained memory `O(n_simulations × n_sectors)`), draws
correlated sector factors through the shared `CorrelationMatrix` Cholesky
factor, and prices LGD from the exact Beta inverse-CDF (scipy when installed; an
accurate tabulated inversion otherwise — only the normal quantile is
approximated without scipy). The per-simulation math mirrors the Rust path, and
cross-validation tests assert the two agree in distribution.

**Real differences from the Rust backend:** speed (Rust parallelizes trials via
Rayon with the GIL released), interim Parquet storage (Rust-only — the fallback
warns and proceeds without persistence), and RNG streams (seeds reproduce
within a backend, not across backends). The result contract is identical: a
test pins both backends' key sets against `PortfolioResult.__annotations__`.

## Performance Characteristics

### Speed and memory (measured)

See [`performance_benchmarks.md`](performance_benchmarks.md) for the full
measured tables and methodology. In short: Rust is ~3–5x faster on single-asset
GBM (~4x less memory — the fallback materializes randoms/log-returns/cumsum
intermediates), ~2.4–7x on correlated GBM (memory at parity: both are dominated
by the identical result array), and ~30–49x on portfolio simulation (where the
fallback is the memory-lean side: it chunks to <2 MB, while Rust carries a
size-independent ~15 MB of thread pool + per-trial summaries).

### Threading Model

| Backend | Threading | Parallelism | GIL Impact |
|---------|-----------|-------------|------------|
| **Rust** | Rayon work-stealing | True parallelism | None — the GIL is released for the duration of the simulation |
| **NumPy** | NumPy's own C loops only | Single-threaded orchestration | Holds the GIL between array operations |

### Algorithmic Differences

#### Random Number Generation
- **Rust**: `StdRng` (ChaCha12), one deterministic stream per path/trial
  (seed + index), so seeded runs are reproducible at any thread count
- **NumPy**: `np.random.default_rng` (PCG64) — seeds reproduce within a backend, not across backends

#### Correlation Handling
- **Rust**: nalgebra Cholesky decomposition (pure Rust), applied inline as a
  triangular dot product per step
- **NumPy**: shared `safe_cholesky` (decompose-or-repair), applied as a matrix
  multiply per step

#### Special Functions
- **Rust**: Beta LGD quantile via a bracketed Newton solver on the regularized
  incomplete beta (unit-tested against `scipy.stats.beta.ppf` at 1e-10)
- **NumPy**: scipy's `betaincinv` when installed; an accurate tabulated
  inversion otherwise

## Feature Completeness Matrix

| Feature | Rust Backend | NumPy Fallback | Notes |
|---------|-------------|----------------|--------|
| **GBM Simulation** | ✅ Full | ✅ Full | Equivalent functionality |
| **Correlated GBM** | ✅ Full | ✅ Full | Both consume the shared validated `CorrelationMatrix`; NumPy applies `safe_cholesky` |
| **Portfolio Simulation** | ✅ Full | ✅ Full | Identical result contract — a test pins both backends' keys against `PortfolioResult.__annotations__` |
| **Interim Results** | ✅ Parquet | ❌ None | Fallback warns and proceeds without persistence |
| **Multi-threading** | ✅ Full | ❌ Limited | GIL constraints |
| **Memory Efficiency** | ✅ GBM ~4x leaner | ✅ Portfolios leaner (chunked) | See measured tables in `performance_benchmarks.md` |
| **Validation** | ✅ Shared | ✅ Shared | Parameter invariants live once in `core/validation.py`; both seams call them |
| **Statistical Functions** | ✅ Machine precision | ✅ Machine precision with scipy | Without scipy, only the normal quantile is approximated (Beta inverse is an accurate tabulated inversion) |

## Dependency Requirements

### Rust Backend
```toml
# Core dependencies only
numpy>=2.2
pandas>=3.0
polars>=1.38
pyarrow>=18.0
```

### NumPy Fallback
```toml
# Same core dependencies
# Optional: scipy (with pure NumPy fallbacks available)
scipy>=1.14.0  # Optional — exact normal quantile; also the `full` extra
```

## Decision Matrix: When to Use Which Backend

### Use Rust Backend When:
- ✅ **Performance critical** applications
- ✅ **Large-scale simulations** (>10K paths)
- ✅ **Production environments** with high throughput requirements
- ✅ **Memory constrained** GBM workloads (portfolio memory is small either way)
- ✅ **Interim results analysis** needed
- ✅ **Binary wheels available** from artifactory

### Use NumPy Fallback When:
- ✅ **Development/prototyping** environments
- ✅ **Binary wheels unavailable**
- ✅ **Small-scale simulations** (<1K paths)
- ✅ **Compatibility testing**
- ✅ **Educational purposes**
- ✅ **Environments without** Rust compilation

## Migration Path

### Phase 1: Development (NumPy Fallback)
```python
# Works immediately without any compilation
import simflux as sf
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=100, n_steps=252)  # Development scale
```

### Phase 2: Production (Rust Backend)
```python
# Same API, but with pre-compiled wheels
import simflux as sf
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)  
paths = gbm.simulate(n_paths=100000, n_steps=252)  # Production scale
```

### Phase 3: High-Performance (Rust + Interim Storage)
```python
import simflux as sf

portfolio = sf.CreditPortfolio.create_sample_portfolio(1000)
results = portfolio.simulate(
    n_simulations=1000000,  # Large scale
    storage_config=sf.StorageConfig(
        store_interim=True,
        output_path="results.parquet"
    )
)
```

## Conclusion

The **Rust backend** provides significant performance advantages and full feature support, while the **NumPy fallback** ensures compatibility and ease of deployment. The dual-backend architecture allows users to:

1. **Start quickly** with NumPy fallback for development
2. **Scale efficiently** with Rust backend for production  
3. **Deploy easily** to environments without build requirements
4. **Maintain compatibility** across different deployment scenarios

This design provides the best of both worlds: performance when needed, compatibility when required.
