# Changelog

All notable changes to SimFlux are documented here. This project adheres to
[Semantic Versioning](https://semver.org/); while pre-1.0, minor versions may
contain breaking changes.

## [Unreleased]

## [0.3.0] — 2026-06-03

### Fixed

- **Wrong-way LGD sign corrected.** The systematic LGD coupling loaded on the
  sector factor with the wrong sign, so a *positive* `systematic_lgd_correlation`
  produced *lower* LGD in downturns (benign "right-way" risk) — the opposite of
  the documented intent and of credit-risk reality. LGD now loads on the negative
  of the sector factor, so a positive correlation raises LGD exactly when defaults
  cluster. Fixed identically in the Rust backend and the NumPy fallback; covered
  by a behavioral guard (`tests/test_lgd_wrong_way.py`) that pins the economics,
  which cross-validation structurally could not (both backends shared the sign).

### Added

- **Per-sector `systematic_lgd_correlation`.** In addition to a scalar, the
  `TwoFactorPortfolio` constructor now accepts a per-sector list, a
  `{sector: value}` dict (missing sectors default to 0.3), or the sentinel
  `"match_intra"`, which sets `ρ_lgd_s = √ρ_intra_s` so LGD's cycle-sensitivity
  matches the default driver's, sector by sector. See methodology §2.10.
- **`default_timing` multi-period model selector** on `simulate()`, with two
  models that both reproduce the marginal cumulative PD exactly and coincide at
  `n_periods=1`:
  - `"copula"` (default) — one-factor Gaussian copula of default times (Li 2000):
    a single latent per obligor vs the cumulative-PD staircase; grid-invariant
    loss distribution.
  - `"frailty"` — dynamic frailty (Duffie et al. 2009): a persistent AR(1)
    systematic factor (`factor_persistence`, annual autocorrelation, default 0.5)
    with fresh idiosyncratic shocks each period and a **calibrated per-period
    barrier** that preserves the marginal PD for any persistence. New module
    `simflux.portfolio.frailty` (`calibrate_barriers`, `survival_curve`,
    `per_period_phi`). See methodology §2.6 and `docs/adr/0002`.
- **`docs/adr/0001`** records the scenario-agnostic (no macro/CCAR conditioning)
  design decision; methodology §2.10–2.11 position the LGD model against industry.

### Changed

- **Wheels are now `abi3` (CPython 3.12+).** A single forward-compatible wheel
  per platform installs on Python 3.12 and newer, so newer Python releases no
  longer require a fresh build (the wheel CI previously broke when a runner
  shipped a Python newer than pyo3's supported maximum). The unused
  `pyo3-polars` dependency was dropped, and the Rust/Python lint gate (rustfmt,
  clippy, black, ruff, mypy) now passes.

### Breaking changes

- **Multi-period default timing now defaults to `"copula"`** (was the implicit
  independent per-period model). Multi-period (`n_periods > 1`) tail numbers
  change: the copula loss distribution is grid-invariant. The prior
  independent-period behavior is the `factor_persistence=0` limit of
  `default_timing="frailty"`. `n_periods=1` is unaffected (the models coincide).
- **Rust seam: `simulate_portfolio` takes `default_timing: str`, `factor_phi: f64`,
  and `barriers: Vec<Vec<f64>>`** (calibrated barriers are computed in Python and
  passed to the backend) in place of the implicit per-period independence.
- **Rust seam: `PortfolioConfig` takes `systematic_lgd_correlations: Vec<f64>`**
  (one value per sector) instead of the scalar `systematic_lgd_correlation: f64`.
  The Python `TwoFactorPortfolio` API is backward compatible — a scalar still
  works, and `.systematic_lgd_correlation` still reads back as a scalar when
  uniform (the canonical attribute is now `.systematic_lgd_correlations`).
- **`pd_term_structure` shorter than `n_periods` now raises `RuntimeError`**
  instead of silently repeating the last cumulative-PD point for the unspecified
  periods. A structure *longer* than `n_periods` is still valid — it runs a
  sub-horizon (the first `n_periods` cumulative-PD points are used), now flagged
  with a clearer `UserWarning`. Sweeping `n_periods` from 1..len over one fixed
  term structure is the intended pattern.

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
