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
  portfolio/  - CreditPortfolio (credit loss model), AssetData, TwoFactorCorrelationStructure,
                default_timing (Copula/Frailty -> TimingPlan), inputs (PortfolioInputs), frailty (calibration impl)
  utils/      - ParquetResultsAnalyzer, StorageConfig, random/correlation utilities (incl. CorrelationMatrix)
src/          - Rust backend (gbm.rs, portfolio.rs, correlation.rs, storage.rs)
```

### Key Design Patterns
- **Strategy**: Backend class (centralized registry) selects Rust or NumPy backend via `Backend.is_available()`
- **Template Method**: BaseSimulator defines validation framework, subclasses specialize
- **Factory**: `AssetData.from_dataframe()`, `CreditPortfolio.create_sample_portfolio()`
- **Lazy Loading**: ParquetResultsAnalyzer uses Polars lazy frames

### Class Hierarchy
```
BaseSimulator (ABC)
├── GBM                         # Single-asset GBM (Rust: simulate_gbm)
├── CorrelatedGBM               # Multi-asset with correlation matrix (Rust: simulate_gbm_multi)
├── TimeVaryingGBM              # Time-varying mu/sigma (Rust: simulate_gbm_tv)
├── TimeVaryingCorrelatedGBM    # Multi-asset time-varying (Rust: simulate_gbm_tv_multi)
└── CreditPortfolio             # Credit portfolio (Rust: simulate_portfolio)

SimulationEngine                # NOT a BaseSimulator — standalone backend dispatcher
```

### Data Flow
```
User API → Simulator class → validate_inputs() → SimulationEngine
  → Backend.is_available() → Rust backend (fast) OR NumPy fallback → Results (ndarray or PortfolioResult)
```

## Code Conventions
- Python 3.12+ required
- Dataclasses for configuration (SimulationConfig, StorageConfig, AssetData)
- Correlation matrices validated: symmetric, diagonal=1, values in [-1,1], positive definite
- `safe_cholesky()` in `utils/random_utils.py` is the **single** decompose-or-repair primitive every correlated sampler crosses — it repairs near-singular matrices and raises `ValueError` (never a raw `LinAlgError`). Do not call `np.linalg.cholesky` directly in samplers.
- `CorrelationMatrix` (`utils/random_utils.py`) is the validated correlation-matrix **value type**: construction runs the strict validator once (shared `CORRELATION_TOLERANCE`); instances carry a lazily cached `cholesky()` (via `safe_cholesky`) and `tolist()` for the FFI. It is **internal currency only** — public seams accept raw arrays and construct it at their validation seam. Never re-implement symmetry/eigenvalue/bounds checks with literal tolerances; use the value type or `correlation_matrix_diagnostics()`.
- Validation is split by *kind*, not by location. **Parameter invariants** (`sigma>0`, `s0>0`, matching lengths, positive-definite correlation) are written **once** as shared validators in `core/validation.py` and called from **both** seams that need them: the simulator constructor (`validate_inputs`, for early/clear construction-time errors) and `SimulationEngine` (which is independently callable, so it self-validates). One validator, two call sites — no copy-paste, all four GBM-family simulators consistent. The `s0_label` argument lets each seam name the parameter as its caller passed it (`S0` on the simulator, `s0` on the engine). **Per-run dimension invariants** (`n_paths`/`n_steps`/`T`) are not parameter invariants and remain solely on the engine (`_validate_sim_dims`).
- **Default timing** is its own module (`portfolio/default_timing.py`): `simulate(default_timing=...)` takes a timing object (`Copula()`, `Frailty(persistence=...)`) or its string sugar; the `factor_persistence` kwarg is deprecated. The model's `plan()` derives a frozen `TimingPlan` — the `(n_periods, n_assets)` threshold matrix + per-period `factor_phi` — which is the **sole timing input both backends consume**; neither backend derives thresholds (the old staircase derivations were deleted). The **kernel** set (`"copula"`/`"frailty"`) is closed: a new derivation over an existing kernel is a pure-Python timing model; a new kernel needs inner loops in both backends + cross-validation. `portfolio/frailty.py` is the untouched implementation layer (`survival_curve` is the deterministic inverse the tests pin).
- The portfolio backend seam takes **one value each way**: `PortfolioInputs` (`portfolio/inputs.py`, assembled once per `simulate()` by `_build_inputs`, self-validating) in, statistics keys out, completed by `_complete_result()` — the single place `PortfolioResult` keys are written, living next to the TypedDict and pinned by a contract test against `PortfolioResult.__annotations__`. The full sector correlation matrix is the sole correlation input at the Rust seam (no scalar `inter_sector_correlation`).
- `StorageConfig` exposes only honored fields: `store_interim`, `output_path`, `batch_size`. `ParquetResultsAnalyzer` column names live in `utils/storage.py::Columns` (mirror of `src/storage.rs`).
- All simulators inherit from BaseSimulator; SimulationEngine does NOT
- Use `Backend.is_available()` / `Backend.get_rust()` for Rust detection — never module-level flags
- `CORRELATION_TOLERANCE = 1e-8` shared constant for all validation (in `core/backend.py`)
- Tests mock via `@patch.object(Backend, 'is_available', ...)` and `@patch.object(Backend, 'get_rust')`; cross-validation forces the NumPy backend through the **public** seam under these patches (never private `_numpy_*` methods)

## Dual-Implementation & Cross-Validation
Every simulation type has both a Rust and NumPy implementation. Cross-validation tests in `tests/test_cross_validation.py` run both backends on the same problem and compare distributional statistics — portfolio runs compare `var_95`/per-sector means (tightened to `var_99` when scipy is present), and the shared conditional-PD derivation is pinned with exact, deterministic checks.

**What cross-validation does and does not prove.** It guards *parity* — that the two backends agree — so it catches one backend *diverging* from the other (e.g. a Cholesky mismatch). It is **blind to errors in logic the two backends share**: a wrong formula present in both passes every cross-check because they agree with each other. Model *correctness* is therefore pinned separately, against external truth, by:
- `tests/test_analytic_vasicek.py` — the homogeneous single-sector limit must match the Vasicek/Basel-ASRF closed form.
- `tests/test_behavioral_invariants.py` — marginal loss = PD·LGD, tail risk monotone in correlation, coherent risk-measure ordering.
- `tests/test_lgd_wrong_way.py` — a positive systematic LGD correlation must *raise* loss (wrong-way risk).
- `TestConditionalPDDeterministic` — exact closed-form PD wiring, no MC.

When adding a model behaviour, add a behavioural/analytic test for it — do **not** rely on cross-validation to catch a model bug, only a backend-divergence bug. (Both v0.2.0-era model bugs — a phantom "global factor" and an inverted LGD sign — were shared by both backends and so passed cross-validation; behavioural tests catch that class.)

## Dependencies
- Python: numpy>=2.2, pandas>=3.0, polars>=1.38, pyarrow>=18.0, scipy (optional)
- Rust: pyo3, ndarray, nalgebra, statrs, rayon, arrow, parquet
