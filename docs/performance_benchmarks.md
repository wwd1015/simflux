# SimFlux Performance Benchmarks: Rust vs NumPy Fallback

## Executive Summary

SimFlux ships a **dual backend**: a Rust extension (the default when the compiled
wheel is available) and a pure-NumPy fallback. Both honor the same API; the Rust
backend exists for speed and memory.

**Measured results** (see methodology below): the Rust backend is **~2.6–45x
faster** than the NumPy fallback across every workload. Memory is **comparable to
lower**: portfolios use **~8–25x less** memory in Rust (it streams per-trial,
while the fallback batches), single-asset GBM about **half**, and correlated GBM
is **at parity**.

> These are indicative numbers from one machine (Apple Silicon, release build).
> Absolute times are hardware-dependent — the **ratios** are what carry across
> machines. Reproduce locally with `python benchmarks/performance_comparison.py`.

## Methodology

- **Timing**: wall-clock `time.perf_counter()` around a single simulation call.
- **Memory**: each (workload, backend) runs in its own **subprocess**, and memory
  is the kernel's **peak RSS** (`ru_maxrss`) reached during the call above the
  pre-call baseline. Subprocess isolation prevents GC and allocator caching from
  one workload contaminating another, and peak RSS captures the Rust backend's
  allocations too (they live outside Python's allocator, so `tracemalloc` would
  miss them). The reported memory figure is **Rust ÷ NumPy** — values below 1.0
  mean Rust uses less.
- The fallback path is exercised directly via the engine's `_numpy_*` methods
  (and `_fallback_simulate_portfolio` for portfolios).
- **Portfolio memory.** Both backends reduce each trial to the same summaries
  (total loss + per-sector loss), and the fallback processes simulations in
  bounded chunks, never allocating a dense `n_simulations × n_assets` grid, so the
  comparison reflects the backends, not a data-structure difference. These figures
  are the **scipy-free** fallback; with scipy installed, the fallback's per-asset
  Beta lookup tables are replaced by scipy's `ppf` and its memory drops further.

## Detailed Results

### 1. Single-Asset GBM

| Problem size | Rust time | NumPy time | Speedup | Rust mem (×NumPy) |
|---|---|---|---|---|
| 1K paths × 252 steps | 0.002s | 0.004s | **2.6x** | 0.49x |
| 10K paths × 252 steps | 0.012s | 0.039s | **3.3x** | 0.41x |
| 50K paths × 252 steps | 0.064s | 0.205s | **3.2x** | 0.41x |
| 10K paths × 1260 steps | 0.051s | 0.195s | **3.8x** | 0.40x |

Rust is ~3x faster and uses ~40–50% of the fallback's memory. The win comes from
returning a contiguous NumPy array (not a Python list-of-lists) and parallel
path generation with Rayon.

### 2. Multi-Asset Correlated GBM

| Problem size | Rust time | NumPy time | Speedup | Rust mem (×NumPy) |
|---|---|---|---|---|
| 2 assets × 1K paths × 252 | 0.003s | 0.011s | **4.0x** | 1.21x |
| 5 assets × 1K paths × 252 | 0.008s | 0.024s | **3.2x** | 1.04x |
| 10 assets × 5K paths × 252 | 0.044s | 0.206s | **4.7x** | 0.94x |
| 20 assets × 1K paths × 252 | 0.021s | 0.079s | **3.8x** | 0.97x |

Rust is ~3–5x faster. Memory is **at parity** with NumPy (slightly leaner at
10+ assets): correlated paths are written inline into one contiguous 3-D buffer,
so there is no separate randoms buffer and no flattening copy. At trivial sizes
(2 assets) per-path setup makes Rust marginally heavier (5 MB vs 4 MB).

### 3. Portfolio Loss Simulation

| Problem size | Rust time | NumPy time | Speedup | Rust mem (×NumPy) |
|---|---|---|---|---|
| 50 assets × 1K simulations | 0.003s | 0.079s | **27x** | 0.13x |
| 200 assets × 5K simulations | 0.026s | 0.499s | **19x** | 0.05x |
| 500 assets × 2K simulations | 0.024s | 0.967s | **41x** | 0.04x |
| 100 assets × 10K simulations | 0.032s | 0.330s | **10x** | 0.11x |

The portfolio path is where Rust dominates on speed: **10–41x faster**, because
it returns a small statistics dictionary (no large array crosses the boundary)
and the Monte Carlo loop parallelizes cleanly across cores.

On memory, Rust uses **~8–25x less**. Both backends reduce each trial on the fly
to its total and per-sector loss, so neither holds a dense
`n_simulations × n_assets` grid. The remaining gap is that Rust streams per-trial
with negligible scratch, while the vectorized fallback keeps a bounded per-chunk
batch (and, on the scipy-free path, O(`n_assets`) Beta lookup tables — the
500-asset row above).

## Why the differences exist

**Rust backend** — `Python API → PyO3 → Rust engine → Rayon → contiguous NumPy / stats dict`:
- True parallelism (no GIL) via Rayon work-stealing.
- Returns contiguous NumPy arrays (zero per-element boxing) and small result
  dicts for portfolios.
- Streams each trial to a per-trial summary, so portfolio memory stays O(trials ×
  sectors); sparse, defaults-only interim storage keeps the on-disk path tiny too.

**NumPy fallback** — `Python API → vectorized NumPy → result`:
- Vectorized and correct; reduces portfolios in bounded chunks to the same
  per-trial summaries (no dense `trials × assets` grid), but still keeps a small
  vectorization batch and pays Python-object overhead at the boundary.
- Single-threaded outside NumPy's own C loops.

## When each backend matters

| Use the Rust backend for | The NumPy fallback is fine for |
|---|---|
| Production and large-scale runs (>10K paths, >100 assets) | Development, prototyping, small tests |
| Memory-constrained portfolio simulations | Environments without a binary wheel |
| Time-critical or batch workloads | Education / quick compatibility checks |
| Interim Parquet storage of default events | — |

Install the compiled wheel to get the Rust backend automatically; without it,
SimFlux falls back to NumPy with a warning and identical results (statistically).

## Reproducing these numbers

```bash
pip install psutil           # the benchmark's only extra dependency
maturin develop --release    # build the Rust extension
python benchmarks/performance_comparison.py
```

Each row is printed as `Speedup: Nx faster, Rust uses Mx the memory`, and full
CSV/text reports are written alongside.
