# Changelog

All notable changes to SimFlux are documented here. This project adheres to
[Semantic Versioning](https://semver.org/); while pre-1.0, minor versions may
contain breaking changes.

## [Unreleased]

## [0.7.0] — 2026-07-07

Performance-and-quality release: measured on a 4-core Linux box against v0.6.0,
single-asset and time-varying GBM are ~2x faster, time-varying correlated GBM
~1.9x, frailty portfolio runs ~1.9x, and copula portfolio runs ~1.25x — with
seeded GBM output bit-identical to v0.6.0.

### Changed

- **LGD quantiles are now accurate to machine precision.** The Rust backend's
  Beta inverse-CDF was statrs's generic 16-step bisection — ~18 CDF evaluations
  per default event and only ~1e-4 absolute accuracy. It is replaced by a
  bracketed Newton solver on the regularized incomplete beta (`beta_inverse_cdf`
  in `math_utils.rs`, unit-tested against `scipy.stats.beta.ppf` at 1e-10),
  which is both faster and matches the quantile the NumPy backend gets from
  scipy. Seeded portfolio statistics move at the ~1e-6 relative level as a
  result; seeded GBM paths are unchanged.
- **Frailty barrier calibration is vectorized across the book.**
  `barrier_matrix` deduplicates obligors by (rho, cumulative-PD curve) and
  calibrates all distinct profiles in one batched bisection
  (`calibrate_barriers_batch`); the scalar `calibrate_barriers` is now a
  single-column wrapper over the same implementation. The erf implementation is
  resolved once at import instead of per call. Calibration of a 200-obligor,
  4-period book drops from ~0.9s to well under 0.1s; results agree with the
  scalar path to solver tolerance.
- **`SimulationConfig` is exported at the top level** (`simflux.SimulationConfig`);
  previously the only supported spelling was `simflux.core.base.SimulationConfig`
  even though every simulator constructor takes it.

### Performance

- **`import simflux` is ~4–5x faster** (~0.52s → ~0.10–0.15s): pandas and
  polars — needed only by `AssetData.from_dataframe` and
  `ParquetResultsAnalyzer` — are no longer imported eagerly through the
  package `__init__` chain. polars loads lazily on first analyzer use
  (`utils/_lazy.py`); the DataFrame `isinstance` check consults `sys.modules`
  so pandas loads only for callers who actually pass a DataFrame.
  `tests/test_import_hygiene.py` pins the property in a fresh interpreter.
- **The first portfolio `simulate()` of a session is ~5x faster to start**
  (~0.96s → ~0.2s): the timing plan's normal quantile now imports
  `scipy.special.ndtri` (the exact kernel `scipy.stats.norm.ppf` dispatches to
  for a standard normal — identical values) instead of the much heavier
  `scipy.stats`, and the NumPy portfolio adapter likewise uses
  `scipy.special.betaincinv` (the kernel behind `scipy.stats.beta.ppf`).
- **Timing-plan thresholds cross the trial loop as one contiguous asset-major
  buffer** instead of a `Vec` per asset row.
- **Release wheels build with fat LTO and a single codegen unit**
  (`[profile.release]` in `Cargo.toml`), letting LLVM inline the RNG and
  distribution crates into the Monte Carlo inner loops.
- **All GBM kernels write one flat result buffer in place** (`Array2`/`Array3`)
  instead of nested `Vec`s flattened at the FFI boundary — single-asset and
  time-varying GBM no longer allocate per path or copy the full result an
  extra time; time-varying correlated GBM no longer builds a
  `Vec<Vec<Vec<f64>>>`. Per-path RNG streams are unchanged, so seeded output
  is bit-identical.
- **The GIL is released during all Rust simulation compute**
  (`py.allow_threads`), so long simulations no longer block other Python
  threads.
- **The portfolio trial loop hoists per-asset constants** (sector/idiosyncratic
  loadings, LGD coupling, per-period LGD Beta parameters with precomputed
  `ln B(a, b)`) out of the hot path, systematic factors live in one flat
  trial-major buffer per period instead of a boxed `Vec` per trial, and the
  copula kernel accumulates losses inline instead of through eight per-trial
  scratch arrays (the frailty kernel keeps its index-order reduce, and event
  detail arrays are now allocated only when interim storage is requested).

### Added

- **Typed exception hierarchy** (`simflux.exceptions`, exported at top level):
  `SimfluxError` → `ValidationError` (also a `ValueError`), `BackendError`
  (also a `RuntimeError`), `MemoryLimitError` (also a `MemoryError`). One
  `except SimfluxError` now catches anything SimFlux raises deliberately,
  while existing `except ValueError`/`RuntimeError`/`MemoryError` code keeps
  working. One deliberate type correction: a `pd_term_structure` shorter than
  `n_periods` now raises `ValidationError` (it is a parameter problem;
  previously `RuntimeError`).
- **Parallel-scaling mode** (`regression_benchmark.py --scaling`): sweeps
  `RAYON_NUM_THREADS` over 1/2/N in fresh subprocesses and reports parallel
  efficiency per kernel. First measurement: portfolio 95% and correlated GBM
  86% efficient at 4 threads; single-asset GBM ~68% (memory-bandwidth-bound
  writing its result buffer).
- **Property-based tests** (`tests/test_properties.py`, hypothesis,
  derandomized): invariants over arbitrary valid inputs — random
  positive-definite correlation matrices factor exactly; the scipy-free
  special-function fallbacks stay within measured error bounds; copula plans
  are exact staircase quantiles, monotone in PD; and the frailty calibration
  reproduces the marginal cumulative PD for arbitrary curves, correlations,
  persistences, and grids.
- **Rust kernel micro-benchmarks** (`benches/kernels.rs`, criterion;
  `make bench-rust`): time the GBM/portfolio kernels and the Beta quantile
  with no Python in the loop, separating kernel changes from plan-derivation/
  FFI/marshalling noise in the Python harnesses.
- **`benchmarks/regression_benchmark.py`** (`make benchmark-regression`): times
  a fixed set of seeded canonical workloads on the active backend and writes
  JSON, with a `--compare` mode — for comparing two SimFlux versions on the
  same machine (complementing `performance_comparison.py`, which compares the
  two backends of one build).

### Removed

- **`benchmarks/benchmark_report_only.py` and
  `benchmarks/sample_benchmark_results.py`** — both printed hardcoded,
  fabricated timings formatted as measurement reports. The real harness,
  `benchmarks/performance_comparison.py` (subprocess-isolated timing and peak
  RSS), remains and is what `make benchmark` runs.

### Fixed

- **A latent flaky memory test.** `test_storage`'s memory check divided two
  allocator-noise-level RSS deltas and passed only by test-ordering luck
  (surfaced by the lazy-polars change). It now warms the full storage path
  once — polars' first operation initializes its thread pool and Parquet
  reader — and asserts an absolute per-run bound, which is what "no dense
  per-row grid" actually means.
- **`benchmarks/performance_comparison.py` timed cold first calls.** Each
  isolated worker now warms up once and times the steady-state second call, so
  one-time costs (lazy scipy imports, thread-pool spinup) no longer masquerade
  as backend cost — previously the small-portfolio "Rust" row was ~200x
  overstated because it was mostly the scipy import. Memory is still measured
  on the first call (peak RSS is a lifetime high-water mark).
- **Both benchmark harnesses report dispersion, not a point estimate.**
  Timing is now the median of N steady-state repeats; each row carries the
  repeats' coefficient of variation and is flagged **UNSTABLE** above 5% CV
  (console, report, and JSON — `regression_benchmark.py --compare` carries the
  flag through), so environment noise is visible instead of masquerading as a
  performance delta. The regression tool also records cpu_count, load average,
  and the NumPy version alongside its existing metadata.
- **`docs/performance_benchmarks.md` and the README carried stale numbers**
  (pre-chunking fallback memory claims and cold-call timings). Both now carry
  fresh measurements from this release with the hardware stated, including the
  honest portfolio-memory picture: the chunked scipy fallback peaks under
  ~2 MB while the Rust backend holds a small size-independent ~15 MB.

### Documentation

- **Full docs sweep to match this release.** `architecture_comparison.md` drops
  its invented memory table and pre-refactor code excerpt in favor of measured
  numbers and the current kernel shape; the feature matrix no longer calls the
  fallback's portfolio statistics "simplified" (the result contract is
  identical and contract-tested) or claims portfolio memory favors Rust (it
  favors the chunked fallback). The deployment guide's speedup banner, the
  README's stale v0.5.0 wheel URLs, and `system_design.md`'s frailty-layer
  description are updated.
- **New user-guide section: Reproducibility and Seeding** — per-simulator
  `SimulationConfig(seed=...)`, what seeding does and does not promise
  (bit-identical within a backend at any thread count; distribution-level
  agreement across backends), previously undocumented.
- **New `docs/roadmap.md`** — prioritized improvement candidates with evidence
  and trade-offs, including a metrology section (dispersion reporting,
  environment capture, kernel-vs-end-to-end separation, CI regression gates,
  parallel-scaling curves, MC-convergence reporting).

### For contributors

- The Rust `TwoFactorCorrelationStructure::generate_factors` now returns a flat
  trial-major `Vec<f64>` and the `SystematicFactors` struct is gone; Rust GBM
  kernels return `ndarray` arrays rather than nested `Vec`s. The Python-facing
  API is unchanged.

## [0.6.0] — 2026-06-11

### Added

- **Timing objects.** You can now select the default-timing model with an object
  that carries its own configuration: `simulate(default_timing=Frailty(persistence=0.6))`
  or `simulate(default_timing=Copula())`. Bad parameters fail at the line you
  typed them (`Frailty(persistence=1.5)` raises immediately), and frailty-only
  knobs no longer ride the `simulate()` signature for copula runs. The
  `"copula"`/`"frailty"` strings still work as sugar for default-configured
  objects. `Copula` and `Frailty` are exported at the top level.

### Changed

- **`CreditPortfolio.sector_correlation_matrix` is now a read-only property.**
  Reassigning it never affected an already-constructed portfolio (the simulation
  reads the validated internal value), so assignment now raises
  `AttributeError` instead of being silently ignored. Pass the matrix at
  construction.
- **`TwoFactorCorrelationStructure` validates more strictly.** The sector
  correlation matrix must now also have a unit diagonal and entries in
  `[-1, 1]` (previously only symmetry and positive semi-definiteness were
  checked) — the same invariants every other correlation seam enforces.
- **Books with PD of exactly 0 or 1 run warning-free.** Multi-period copula
  simulation no longer emits a spurious `RuntimeWarning` for obligors at the
  PD endpoints, and NaN cumulative PDs are rejected loudly by both timing
  models instead of silently zeroing risk in frailty calibration.

### Deprecated

- **`factor_persistence` keyword.** Pass `default_timing=Frailty(persistence=...)`
  instead. The keyword still works alongside the string sugar (with a
  `DeprecationWarning`) and is rejected alongside a timing object so persistence
  can never be specified twice. Note: an out-of-range `factor_persistence`
  passed with `"copula"` now warns and is ignored rather than raising (it was
  always ignored semantically).

### For contributors

- **Unified threshold plan.** The chosen timing model derives a frozen
  `TimingPlan` (per-period, per-obligor threshold matrix + AR(1) coefficient)
  once in Python; both backends consume it and neither derives thresholds
  anymore — the duplicated cumulative-PD staircase in `src/portfolio.rs` and
  `numpy_simulation.py` is deleted, so threshold parity holds by construction.
  The private Rust FFI renamed `default_timing`→`kernel` and
  `barriers`→`thresholds` (now required for both kernels, crossing as a NumPy
  array). Seeded Rust copula results shift: distributional statistics move
  within numerical tolerance (one quantile implementation now feeds both
  backends), but an individual trial whose latent sat within ~1e-9 of a
  threshold can flip default state, which changes that trial's loss and its
  subsequent RNG draws — anyone pinning seeded trial-level outputs (interim
  Parquet rows) will see changed rows. The Rust seam also gained content
  guards (NaN thresholds and out-of-range `factor_phi` now error instead of
  silently zeroing risk).
- **One value each way at the portfolio backend seam.** `PortfolioInputs`
  (self-validating) carries the whole adapter contract in;
  `_complete_result()`, next to the `PortfolioResult` TypedDict, is the single
  place result keys are written, pinned by a contract test.
- **`CorrelationMatrix` value type.** Correlation-matrix invariants are checked
  once, at construction, with the shared `CORRELATION_TOLERANCE`; instances
  carry a cached `cholesky()`. The hand-rolled checks in
  `TwoFactorCorrelationStructure` (literal `1e-8`s) are gone.

## [0.5.0] — 2026-06-08

### Added

- **Per-obligor (heterogeneous) intra-sector correlation.** `AssetData.intra_sector_correlation`
  is now honored per obligor: an obligor loads on its sector factor with its own
  `rho_i` (`V_i = sqrt(rho_i)*F_sector + sqrt(1-rho_i)*eps_i`), falling back to the
  sector-level value when unset. Obligors within a sector may differ; the previous
  homogeneity requirement (which *raised* on mixed values) is gone. Both backends
  honor it and `CreditPortfolio.asset_intra_correlations` exposes the resolved
  per-obligor loadings.
- **Time-varying LGD.** `AssetData.lgd_term_structure` gives a per-period LGD Beta
  *mean*; an obligor that defaults in period `k` draws LGD from the Beta with mean
  `lgd_term_structure[k]` (clamped to the last entry) and the constant `lgd_std`.
  A constant term structure reduces exactly to the flat `lgd_mean`. Works in both
  the `copula` and `frailty` default-timing models.

### Changed

- **Docs.** The README's single-period and multi-period portfolio sections are
  merged into one (single-period is just `n_periods=1`), and now show how
  `AssetData` is actually constructed (explicit list and `from_dataframe`).

### Breaking changes

- **`TwoFactorPortfolio` renamed to `CreditPortfolio`.** The model is a
  multi-sector, single-systematic-factor Gaussian copula (the analytic test pins
  the Vasicek single-factor limit); "two-factor" read as a factor *count* and
  caused confusion. This is a clean rename with no alias — update imports from
  `TwoFactorPortfolio` to `CreditPortfolio`.

## [0.4.3] — 2026-06-07

### Changed

- **Internal architecture (no public-API or numerical change).** Three deepening
  refactors from an architecture review:
  - **NumPy portfolio simulation extracted to its own module.**
    `TwoFactorPortfolio` no longer carries the ~280-line NumPy fallback inline;
    it lives in `portfolio/numpy_simulation.py` as a pure
    `simulate_portfolio_numpy(...)`, testable directly. `two_factor_model.py`
    drops from 1083 to 881 lines.
  - **One backend-dispatch seam.** Added `Backend.choose(rust, numpy)`, the single
    Rust-vs-NumPy decision; the four GBM engine entry points and the portfolio's
    `simulate()` now resolve their adapter through it instead of open-coding
    `if Backend.is_available()` at six sites. Resolution stays per call, so tests
    that patch `Backend.is_available` are unaffected.
  - **Shared parameter validators.** Each GBM-family parameter invariant
    (`sigma>0`, `s0>0`, matching lengths, positive-definite correlation) is now
    written once in `core/validation.py` and called from both the simulator
    constructor and `SimulationEngine`, instead of being copy-pasted across the
    two seams (GBM, CorrelatedGBM) or missing from the engine entirely
    (the time-varying simulators). All four simulators now validate consistently
    at both seams; the engine self-validates time-varying parameters for direct
    calls, which it previously did not.

## [0.4.2] — 2026-06-05

### Changed

- **NumPy portfolio fallback now reduces per-chunk instead of holding a dense
  `n_simulations × n_assets` loss grid**, making the memory benchmark a fair
  backend comparison. The Rust backend already streamed each trial to a per-trial
  summary (total loss + per-sector loss, O(`n_simulations × n_sectors`) retained);
  the fallback previously materialized ~8 simultaneous `n_simulations × n_assets`
  arrays, so the "portfolios use ~100x less memory in Rust" figure was measuring a
  data-structure choice in Python, not the backend. The fallback now walks
  simulations in bounded chunks (`_FALLBACK_CHUNK_ELEMENTS`) and reduces each chunk
  to the same two summaries, dropping its peak from a flat ~147 MB (any shape) to
  memory that scales with `n_assets` (e.g. 200a×5K: 147 MB → 34 MB; 100a×10K:
  147 MB → 26 MB). The honest portfolio memory ratio is now **~8–25x** less in
  Rust (was reported as 10–100x). The scipy-free Beta inverse-CDF is also tabulated
  **once per distinct obligor** up front rather than rebuilt on every chunk, so
  chunking adds no measurable time. The per-simulation math is unchanged (only the
  random-draw *order* changes), so the loss distribution and every cross-validated
  statistic are identical; `docs/performance_benchmarks.md`, the README, and
  `docs/architecture_comparison.md` are updated to the fair numbers.

## [0.4.1] — 2026-06-05

### Performance

- **The Rust backend is now ~2.6–45x faster than the NumPy fallback** (it had
  been up to ~3x *slower* for GBM paths). Three fixes:
  - GBM entry points return contiguous NumPy arrays instead of a Python
    list-of-lists, eliminating per-element `PyFloat` boxing (GBM Large Rust path
    0.61s → 0.079s).
  - Correlated normals are generated with one RNG per path/chunk instead of one
    per sample (which re-seeded a ChaCha RNG hundreds of thousands of times).
  - Correlated GBM writes paths inline into a single contiguous `Array3` — no
    pre-generated randoms buffer, no `Vec<Vec<Vec<f64>>>` tensor, no flatten
    copy — so its peak memory drops from ~3x the NumPy fallback to parity.
  - Net memory: single-asset GBM ~0.4x NumPy, correlated GBM ~parity, portfolios
    10–100x leaner (the sparse interim store).

### Fixed

- **`generate_correlation_matrix` now returns a strictly positive-definite
  matrix.** The repair floored eigenvalues exactly at the tolerance, leaving the
  smallest one at the strict-PD boundary, so the function violated its own
  "positive definite" docstring and raised `correlation_matrix must be positive
  definite` when its output was fed to `CorrelatedGBM`. Already-PD inputs still
  pass through with exact entries; near-singular inputs are lifted clear of the
  boundary (unit diagonal preserved).

### Changed

- **Accurate memory benchmark.** `benchmarks/performance_comparison.py` measured
  memory as an in-process RSS before/after delta — meaningless, since GC and
  allocator caching produced negative and `inf` readings. Each workload now runs
  in an isolated subprocess and reports the kernel peak RSS (`ru_maxrss`).
  `docs/performance_benchmarks.md` (and the README/deployment perf claims) are
  rewritten to the real measured numbers; the previous "3–14x speedup / 3–5x
  memory" figures were not reproducible.
- **Correlated-GBM RNG scheme changed** (one RNG per path). Exact seeded paths
  therefore differ from 0.4.0; the distribution is unchanged and within-version
  reproducibility is preserved.

## [0.4.0] — 2026-06-04

### Changed

- **Interim storage is now sparse (defaults-only), cutting peak memory.** With
  `store_interim=false` the Rust backend no longer materializes a per-asset
  result for every asset-trial — each trial reduces to its total loss and
  per-sector losses, so memory drops from O(n_simulations × n_assets) to
  O(n_simulations × n_sectors). With `store_interim=true` only **defaulted**
  obligor-trials are written (defaults are rare relative to the full grid),
  supporting much larger books and run counts.

### Breaking changes

- **Interim Parquet schema is defaults-only with new debug columns.** Every row
  is now a default event; the `defaulted` flag is gone. Added `default_period`
  and `idiosyncratic_factor`; `systematic_factor_global`/`_sector` collapse to a
  single `systematic_factor`. Portfolio totals that the rows no longer carry
  (trial/asset counts, per-sector default rates) are written to the Parquet
  file-level key-value metadata, and `ParquetResultsAnalyzer` reads them:
  `count_simulations`/`count_assets` are exact, `get_trial_losses` reindexes to
  every trial (zero-default trials contribute zero), and `analyze_by_sector`
  recovers `default_rate` from the metadata. `get_systematic_factors` now returns
  the per-default systematic and idiosyncratic factors.

### Fixed

- **NumPy fallback LGD is now correct without scipy.** When neither the Rust
  backend nor scipy was available, the fallback returned a uniform (mean ≈ 0.5)
  for realized LGD, ignoring `lgd_mean` and overstating loss behind only a
  `UserWarning`. It now maps the LGD driver through a dependency-free, tabulated
  Beta inverse-CDF (`simflux.utils.special.beta_ppf`) that matches
  `scipy.stats.beta.ppf` to ~1e-4 (≤1e-5 for typical LGD Beta parameters). New
  `tests/test_special.py`.

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
