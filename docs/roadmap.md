# Improvement Roadmap

Prioritized, concrete candidates for the next rounds of quality and speed
work. Each item states the evidence behind it (measured on this repo's
benchmark tooling where applicable) and its trade-off. Items are independent
unless noted.

## Performance

### P1. Lazy-import pandas/polars at the package boundary — **done (0.7.0)**
`import simflux` cost ~0.52s, of which ~0.35s was pandas (pulled in eagerly by
`utils/storage.py` through the package `__init__` chain) plus polars — both
needed only by `AssetData.from_dataframe` and `ParquetResultsAnalyzer`. They
are now deferred (`utils/_lazy.py` proxy for polars; a `sys.modules` check for
the DataFrame `isinstance`), cutting `import simflux` to ~0.10–0.15s.
`tests/test_import_hygiene.py` pins the property.

### P2. Correlated-GBM sampling throughput
The correlated kernels are now bound by normal sampling (`StdRng`/ChaCha12 +
ziggurat) rather than memory or marshalling — the 10-asset workload barely
moved (~1.03x) while flat-buffer wins took the other kernels ~2x. A
counter-based RNG (Philox) or a faster ziggurat could plausibly give ~1.5–2x,
**but changes seeded streams**. Requires an explicit RNG-versioning policy
first (e.g. `SimulationConfig(rng="v2")` with the default preserved for one
release, and a documented stream-compatibility promise). Do not change streams
silently.

### P3. Frailty kernel iteration
The frailty trial loop still walks every (asset, period) pair and skips
defaulted obligors with a branch. An active-list (swap-remove of defaulted
indices) plus struct-of-arrays layout for `AssetPrecomp` would improve branch
predictability and cache density; estimated ~1.2–1.5x on the kernel portion
(the calibration portion is already vectorized). Preserves RNG order only if
iteration order stays index-sorted per period — verify with the reference-hash
harness before landing.

### P4. Tuned wheel builds
Wheels currently build for the baseline x86-64 ISA. Options: `cargo`'s
`target-cpu` feature multiversioning for the hot kernels (AVX2 dispatch at
runtime), or a PGO build in CI. Historically LTO alone gave a measurable win
here; PGO typically adds a further 5–15% on branchy Monte Carlo code. Cost:
CI complexity, longer release builds.

### P5. Overlap Parquet writing with simulation
`store_interim` currently simulates all trials, then writes. Streaming record
batches to the Arrow writer from a bounded channel while trials run would hide
most of the I/O time on storage-heavy runs.

## Quality and robustness

### Q1. Property-based tests (hypothesis) — **done (0.7.0)**
`tests/test_properties.py` (derandomized for CI) asserts invariants over
arbitrary valid inputs: random positive-definite correlation matrices factor
exactly and pass diagnostics; `approx_norm_ppf` and the scipy-free `beta_ppf`
stay within measured error bounds against references; copula plans are exact
staircase quantiles, monotone in PD; and the frailty calibration reproduces
the marginal cumulative PD for arbitrary curves, correlations, persistences,
and grids. Remaining extension: loss-statistics coherence (ES ≥ VaR) needs the
stats helper lifted out of `simulate_portfolio_numpy` to be directly testable.

### Q2. Typed exception taxonomy
Errors surface as `ValueError`/`RuntimeError` with good messages but no types;
callers can't catch "invalid correlation matrix" separately from "backend
failure". A small hierarchy (`SimfluxError` → `ValidationError`,
`BackendError`, `StorageError`) is API-additive if the new types subclass the
current ones.

### Q3. mypy strict tightening
Tracked in `pyproject.toml` already: the codebase passes non-strict mypy;
ratcheting per-module (`disallow_untyped_defs` on `core/`, then `portfolio/`)
avoids a big-bang annotation PR.

### Q4. Publish to PyPI with cibuildwheel + trusted publishing
Wheels currently ship via GitHub Releases. cibuildwheel adds musllinux and
linux-aarch64 targets cheaply, and PyPI trusted publishing removes token
handling. Also enables `pip install simflux` plain.

### Q5. API reference generation
Docstrings are thorough but only reachable in-source. mkdocs-material +
mkdocstrings (or Sphinx) turning them into a published reference, with a CI
check that public symbols have docstrings, closes the docs loop.

## Metrology (measurement methodology)

The benchmark tooling became honest this release (steady-state timing,
subprocess isolation, fabricated reports deleted). What would make it
rigorous:

### M1. Report dispersion, not just a point estimate — **done (0.7.0)**
Both harnesses now time N repeats and report the **median** with IQR
(regression) or coefficient of variation (comparison); rows whose CV exceeds
5% are flagged **UNSTABLE** in the console, the report, and the JSON, and the
regression tool's `--compare` carries the flag through. The first run after
landing this immediately flagged 17–23% CV on a busy container — exactly the
condition that previously masqueraded as a code regression.

### M2. Capture the environment in every report
`regression_benchmark.py` already stamps version/backend/Python/machine. Both
harnesses should also record: CPU model and core count, cgroup CPU quota
(containers often cap cores invisibly), load average at run time, governor /
frequency state where readable, NumPy/scipy/BLAS versions, and the simflux git
SHA. A result without its environment is not comparable to anything.

### M3. Separate kernel time from end-to-end time — **done (0.7.0)**
`benches/kernels.rs` (`make bench-rust`) times the GBM, correlated-GBM, and
both portfolio kernels plus the Beta quantile with criterion — no plan
derivation, FFI, or marshalling in the loop — so kernel changes are measured
in isolation while the Python harnesses keep the end-to-end numbers. When the
two disagree, the difference is the Python side (exactly the confusion that
motivated this item).

### M4. Continuous benchmarking with regression gates
Run `regression_benchmark.py` in CI on a fixed runner class, store the JSON
per commit (e.g. github-action-benchmark or bencher.dev), and alert on >10%
regressions of the median. Cloud runners are noisy — gate on the median across
repeats and require two consecutive failing runs before flagging, or use a
dedicated self-hosted runner for stable numbers.

### M5. Parallel-scaling curve
Report throughput at 1, 2, and N threads (`RAYON_NUM_THREADS`) so parallel
efficiency is visible — a kernel that stops scaling is a finding the current
single-configuration numbers can't show. Normalize throughput as samples/sec
(paths × steps / time) so sizes are comparable.

### M6. Statistical-accuracy metrology
Speed is half the measurement; the other half is MC error. Add a convergence
report that runs the Vasicek-limit portfolio at increasing trial counts and
plots |MC − closed form| against the theoretical 1/√N Monte Carlo standard
error, verifying both the estimator's bias (should be none) and its
variance rate. The analytic-limit tests assert this pointwise; the report
makes the convergence *rate* visible and documents which tolerances are
derived from it.

### M7. Memory metrology beyond peak RSS
Peak RSS is the right headline, but it can't be re-measured after warmup
(lifetime high-water mark) and conflates one-time imports with working set.
Complement it with `/proc/self/smaps_rollup` deltas around the steady-state
call and an allocator-level count on the Rust side (a counting global
allocator behind a feature flag) when investigating, keeping peak RSS for the
published tables.
