# Changelog

All notable changes to SimFlux are documented here. This project adheres to
[Semantic Versioning](https://semver.org/); while pre-1.0, minor versions may
contain breaking changes.

## [0.2.0] — 2026-05-28

This release sharpens the seam **between** the Rust and NumPy backends rather
than collapsing the (intentional) dual implementation. It removes dead surface
area and makes both backends honor one contract.

### Breaking changes

- **`StorageConfig` shrunk to the fields the writer actually honors.** Removed
  `store_defaults`, `store_losses`, `store_systematic_factors`, `format`,
  `partition_by`, and `compression` — none were read by the Parquet writer
  (e.g. `compression="gzip"` silently produced a Snappy file). The supported
  fields are now `store_interim`, `output_path`, and `batch_size`. `batch_size`
  is now genuinely threaded into the Rust writer instead of being hard-coded.
- **`SimulationResults` removed** (`simflux.core.base` / `simflux.core`). It had
  zero instantiation sites — no simulator returned it. Use the `dict`/
  `PortfolioResult` returned by `simulate()` and `ParquetResultsAnalyzer` for
  stored interim data.
- **`ParquetResultsAnalyzer.export_to_pandas` signature changed.** It now takes
  domain filters `export_to_pandas(trial_range=None, sectors=None, asset_ids=None, columns=None)`
  instead of a raw Polars expression (`query_filter: pl.Expr`), so callers no
  longer author Polars expressions against the physical schema.
- **`inter_sector_correlation` removed from the Rust seam.** The Rust
  `PortfolioConfig` constructor and `TwoFactorCorrelationStructure` no longer
  accept it — the full `sector_correlation_matrix` is the single source of truth
  for cross-sector coupling. (The standalone Python
  `TwoFactorCorrelationStructure` and `create_sample_portfolio` keep their
  `inter_sector_correlation` convenience argument.)
- **Dead `create_parquet_schema()` removed from `utils/storage.py`** (the Rust
  schema is authoritative; the Python copy was unused and had drifted).

### Added

- **`safe_cholesky()` in `utils/random_utils.py`** — the single decompose-or-
  repair primitive every correlated sampler now crosses. It repairs near-
  singular matrices and raises `ValueError` (never a raw `numpy.linalg.LinAlgError`).
- **`PortfolioResult` (`TypedDict`)** documenting the portfolio result contract.
  Both backends now produce an identical key set, with `n_trials` stamped
  uniformly in Python — `results['n_trials']` no longer `KeyError`s on the NumPy
  fallback.
- **`Columns` constants in `utils/storage.py`** — one Python-side source of truth
  for the interim Parquet column names (kept in sync with `src/storage.rs`).
- **Stronger cross-validation.** Portfolio cross-validation now compares
  `var_95` and per-sector means (tightened to `var_99` when scipy is present),
  the conditional-PD derivation is pinned with exact deterministic checks, and a
  non-constant time-varying schedule exercises interpolation agreement between
  the backends.
- **Rust-result shape guards.** Each Rust simulation call is marshalled through a
  helper that asserts the returned array shape, so a Python/Rust axis mismatch
  fails loudly instead of producing silently wrong numbers.

### Changed

- **`TimeVaryingGBM` / `TimeVaryingCorrelatedGBM` validation is now honest.**
  `validate_inputs` performs the real construction-time checks (and raises
  `ValueError` for an invalid correlation matrix instead of leaking a raw
  `LinAlgError`), matching the other simulators.
- **`TwoFactorCorrelationStructure` (Python) reframed as a diagnostics/inspection
  helper.** Its docstrings no longer claim to replicate the Rust backend — each
  backend computes loadings/factors inline; the class is for inspection.
- **`SimulationConfig.seed` documents the reproducibility contract** —
  within-backend reproducible, cross-backend statistical only.
- **`validate_correlation_matrix` (bool) is now a thin wrapper** over
  `validate_correlation_matrix_strict`, removing a duplicated check body.
- Cross-validation tests force the NumPy backend through the **public** engine
  seam under `Backend` patches, rather than reaching into private `_numpy_*`
  methods.
- **`TwoFactorPortfolio` now requires a positive-definite sector correlation
  matrix.** A singular / rank-deficient (PSD-but-not-PD) matrix — e.g. two
  perfectly correlated sectors — is rejected at construction with a clear
  `ValueError`, identically on both backends. Previously such a matrix was
  admitted at construction and then failed (or, with the new repairing
  `safe_cholesky`, would have silently diverged) at simulate time. This matches
  the existing requirement on `CorrelatedGBM` / `TimeVaryingCorrelatedGBM`.

### Fixed

- **`ParquetResultsAnalyzer.analyze_by_sector()`** now uses the polars ≥ 1.38
  naming API (`.name.suffix(...)`); it previously raised
  `AttributeError: 'Expr' object has no attribute 'suffix'` on the supported
  polars version.
