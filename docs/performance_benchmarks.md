# SimFlux Performance Benchmarks: Rust vs NumPy Fallback

## Executive Summary

SimFlux ships a **dual backend**: a Rust extension (the default when the compiled
wheel is available) and a pure-NumPy fallback. Both honor the same API; the Rust
backend exists for speed.

**Measured results** (see methodology below): the Rust backend is **~2.4–49x
faster** than the NumPy fallback — ~3–5x on single-asset GBM, ~2.4–7x on
correlated GBM, and **30–49x on portfolio simulation**. On memory, Rust uses
~4x less for single-asset GBM, is at parity on correlated GBM, and carries a
small constant overhead (~15 MB: thread pool + per-trial summaries) on
portfolios, where the chunked scipy fallback runs in under 2 MB.

> Measured on SimFlux 0.7.0: 4-core x86_64 Linux, Python 3.12, NumPy 2.5,
> scipy installed, release build (fat LTO). Absolute times are
> hardware-dependent — the **ratios** are what carry across machines.
> Reproduce locally with `python benchmarks/performance_comparison.py`.

## Methodology

- **Timing**: wall-clock `time.perf_counter()` around **steady-state** calls —
  each worker runs the workload once untimed (absorbing one-time costs such as
  lazy scipy imports and thread-pool spinup), then times 5 repeats and reports
  their **median**. Each row carries the repeats' coefficient of variation;
  rows exceeding 5% CV are flagged **UNSTABLE** and should be treated as
  environment noise (CPU contention, frequency scaling), not backend signal —
  re-run on a quiet machine rather than publishing them.
- **Memory**: each (workload, backend) runs in its own **subprocess**, and memory
  is the kernel's **peak RSS** (`ru_maxrss`) reached during the first call above
  the pre-call baseline. Subprocess isolation prevents GC and allocator caching
  from one workload contaminating another, and peak RSS captures the Rust
  backend's allocations too (they live outside Python's allocator, so
  `tracemalloc` would miss them).
- The fallback path is exercised directly via the engine's `_numpy_*` methods
  (and the portfolio's NumPy adapter), so dispatch overhead is out of the
  picture.
- **Portfolio memory.** Both backends reduce each trial to the same summaries
  (total loss + per-sector loss), and the fallback processes simulations in
  bounded chunks, never allocating a dense `n_simulations × n_assets` grid.
  These figures are the **scipy-installed** fallback; the scipy-free fallback
  additionally tabulates per-asset Beta lookup tables.

## Detailed Results

### 1. Single-Asset GBM

| Problem size | Rust time | NumPy time | Speedup | Rust mem / NumPy mem |
|---|---|---|---|---|
| 1K paths × 252 steps | 0.002s | 0.011s | **5.4x** | ~0 / 5.3 MB |
| 10K paths × 252 steps | 0.024s | 0.112s | **4.8x** | 17 / 76 MB |
| 50K paths × 252 steps | 0.161s | 0.629s | **3.9x** | 96 / 392 MB |
| 10K paths × 1260 steps | 0.182s | 0.576s | **3.2x** | 95 / 391 MB |

Rust is ~3–5x faster and uses ~25% of the fallback's memory. The kernels write
each path directly into one flat, contiguous buffer that moves into NumPy with
no copy, and paths generate in parallel with Rayon; the fallback materializes
separate randoms/log-returns/cumsum intermediates.

### 2. Multi-Asset Correlated GBM

| Problem size | Rust time | NumPy time | Speedup | Rust mem / NumPy mem |
|---|---|---|---|---|
| 2 assets × 1K paths × 252 | 0.011s | 0.027s | **2.4x** | 0.7 / 0.9 MB |
| 5 assets × 1K paths × 252 | 0.017s | 0.072s | **4.4x** | 6.8 / 6.7 MB |
| 10 assets × 5K paths × 252 | 0.096s | 0.655s | **6.8x** | 96 / 98 MB |
| 20 assets × 1K paths × 252 | 0.048s | 0.291s | **6.0x** | 36 / 37 MB |

Rust is ~2.4–7x faster, and the advantage grows with asset count (the per-step
Cholesky application is a tight triangular loop in Rust vs a matrix multiply
plus temporaries in NumPy). Memory is **at parity**: both backends' resident
memory is dominated by the identical `(paths, assets, steps+1)` result array.

### 3. Portfolio Loss Simulation

| Problem size | Rust time | NumPy time | Speedup | Rust mem / NumPy mem |
|---|---|---|---|---|
| 50 assets × 1K simulations | 0.003s | 0.099s | **30x** | 15 / 0.9 MB |
| 200 assets × 5K simulations | 0.042s | 2.029s | **49x** | 16 / 0.9 MB |
| 500 assets × 2K simulations | 0.046s | 2.001s | **43x** | 15 / 0.9 MB |
| 100 assets × 10K simulations | 0.047s | 2.009s | **43x** | 16 / 1.3 MB |

The portfolio path is where Rust dominates on speed: **30–49x faster**. Only a
small statistics dictionary crosses the language boundary, per-asset constants
(loadings, LGD Beta parameters) are hoisted out of the trial loop, and the
Monte Carlo trials parallelize cleanly across cores.

On memory the *fallback* is leaner here: it streams bounded chunks and (with
scipy) peaks under ~2 MB, while the Rust backend's ~15 MB is essentially a
constant — Rayon thread stacks plus the per-trial summary vector — that does
not grow with portfolio size. Neither backend allocates a dense
`n_simulations × n_assets` grid.

## Why the differences exist

**Rust backend** — `Python API → PyO3 → Rust engine → Rayon → contiguous NumPy / stats dict`:
- True parallelism (no GIL — and the GIL is released for the duration of the
  simulation) via Rayon work-stealing.
- Returns contiguous NumPy arrays written in place (zero per-element boxing,
  no flattening copies) and small result dicts for portfolios.
- Streams each trial to a per-trial summary, so portfolio memory stays
  O(trials × sectors); sparse, defaults-only interim storage keeps the on-disk
  path tiny too.

**NumPy fallback** — `Python API → vectorized NumPy → result`:
- Vectorized and correct; reduces portfolios in bounded chunks to the same
  per-trial summaries (no dense `trials × assets` grid), with very low peak
  memory on the portfolio path.
- Single-threaded outside NumPy's own C loops, and each step materializes
  full-size intermediate arrays on the GBM paths.

## When each backend matters

| Use the Rust backend for | The NumPy fallback is fine for |
|---|---|
| Production and large-scale runs (>10K paths, >100 assets) | Development, prototyping, small tests |
| Time-critical or batch workloads | Environments without a binary wheel |
| Interim Parquet storage of default events | Education / quick compatibility checks |

Install the compiled wheel to get the Rust backend automatically; without it,
SimFlux falls back to NumPy with a warning and identical results (statistically).

## Reproducing these numbers

```bash
pip install psutil           # the benchmark's only extra dependency
maturin develop --release    # build the Rust extension
python benchmarks/performance_comparison.py
```

Full CSV/text reports are written alongside. To compare two SimFlux versions
on the *same* backend (e.g. before/after an optimization), use
`python benchmarks/regression_benchmark.py` — it times a fixed set of seeded
workloads and writes JSON that its `--compare` mode diffs. Its `--scaling`
mode sweeps `RAYON_NUM_THREADS` over 1/2/N and reports per-kernel parallel
efficiency (on the machine measured here: portfolio 95%, correlated GBM 86%,
single-asset GBM ~68% at 4 threads — the latter is memory-bandwidth-bound).

To measure the Rust kernels in isolation (no plan derivation, FFI, or
marshalling in the loop), run the criterion micro-benchmarks:

```bash
make bench-rust   # cargo bench --no-default-features --bench kernels
```
