"""Version-to-version performance regression benchmark.

Times a fixed set of seeded, canonical workloads on the active backend and
writes the results as JSON, so two checkouts (or two installed versions) can be
compared number-for-number:

    python benchmarks/regression_benchmark.py before.json   # on the old build
    python benchmarks/regression_benchmark.py after.json    # on the new build
    python benchmarks/regression_benchmark.py --compare before.json after.json

This complements ``performance_comparison.py`` (which compares the *Rust vs
NumPy backends* of one build): here both runs use the same backend and the
variable is the build itself. Workloads are seeded so every run does identical
work. The headline number is the median of N repeats with its interquartile
range; runs whose coefficient of variation exceeds 5% are flagged UNSTABLE —
treat those as a noisy environment, not a code delta.
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from simflux import (
    GBM,
    CorrelatedGBM,
    CreditPortfolio,
    SimulationConfig,
    TimeVaryingCorrelatedGBM,
    TimeVaryingGBM,
)
from simflux.core.backend import Backend

# A run whose coefficient of variation exceeds this is flagged: the number
# reflects environment noise (CPU contention, frequency scaling) more than the
# code, and should not be compared against other runs.
UNSTABLE_CV = 0.05


def _timeit(fn, repeat=5, warmup=1):
    """Time ``fn`` and report the distribution, not just a point estimate.

    Returns a dict with best, median, IQR, and the coefficient of variation
    (stdev/mean); ``unstable`` is set when the CV exceeds ``UNSTABLE_CV``.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = statistics.median(times)
    q1, q3 = statistics.quantiles(times, n=4)[0], statistics.quantiles(times, n=4)[2]
    mean = statistics.fmean(times)
    cv = (statistics.stdev(times) / mean) if (len(times) > 1 and mean > 0) else 0.0
    return {
        "best_s": round(times[0], 4),
        "median_s": round(median, 4),
        "iqr_s": round(q3 - q1, 4),
        "cv": round(cv, 4),
        "repeats": repeat,
        "unstable": cv > UNSTABLE_CV,
    }


def _seeded():
    return SimulationConfig(seed=42)


def bench_gbm():
    sim = GBM(mu=0.05, sigma=0.2, S0=100.0, config=_seeded())
    return _timeit(lambda: sim.simulate(n_paths=50_000, n_steps=252, T=1.0))


def bench_correlated_gbm():
    n = 10
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    sim = CorrelatedGBM(
        mu=[0.05] * n,
        sigma=[0.2] * n,
        S0=[100.0] * n,
        correlation_matrix=corr,
        config=_seeded(),
    )
    return _timeit(lambda: sim.simulate(n_paths=10_000, n_steps=252, T=1.0))


def bench_time_varying_gbm():
    sim = TimeVaryingGBM(
        mu_times=[0.0, 0.5, 1.0],
        mu_values=[0.05, 0.03, 0.04],
        sigma_times=[0.0, 0.5, 1.0],
        sigma_values=[0.2, 0.25, 0.22],
        S0=100.0,
        config=_seeded(),
    )
    return _timeit(lambda: sim.simulate(n_paths=50_000, n_steps=252, T=1.0))


def bench_time_varying_correlated_gbm():
    n = 5
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    sim = TimeVaryingCorrelatedGBM(
        mu_times=[[0.0, 1.0]] * n,
        mu_values=[[0.05, 0.04]] * n,
        sigma_times=[[0.0, 1.0]] * n,
        sigma_values=[[0.2, 0.25]] * n,
        S0=[100.0] * n,
        correlation_matrix=corr,
        config=_seeded(),
    )
    return _timeit(lambda: sim.simulate(n_paths=10_000, n_steps=252, T=1.0))


def _sample_portfolio(n_per_sector=40):
    return CreditPortfolio.create_sample_portfolio(
        n_assets_per_sector=n_per_sector,
        sectors=["Tech", "Fin", "Health", "Energy", "Retail"],
        config=_seeded(),
    )


def bench_portfolio_copula():
    portfolio = _sample_portfolio()
    return _timeit(
        lambda: portfolio.simulate(
            n_simulations=20_000, n_periods=4, default_timing="copula"
        ),
        repeat=3,
    )


def bench_portfolio_frailty():
    portfolio = _sample_portfolio()
    return _timeit(
        lambda: portfolio.simulate(
            n_simulations=20_000, n_periods=4, default_timing="frailty"
        ),
        repeat=3,
    )


def bench_portfolio_large():
    portfolio = _sample_portfolio(n_per_sector=200)
    return _timeit(
        lambda: portfolio.simulate(n_simulations=10_000, n_periods=1), repeat=3
    )


BENCHES = {
    "gbm_50k_paths_252_steps": bench_gbm,
    "correlated_gbm_10_assets_10k_paths": bench_correlated_gbm,
    "time_varying_gbm_50k_paths": bench_time_varying_gbm,
    "time_varying_correlated_gbm_5_assets_10k_paths": bench_time_varying_correlated_gbm,
    "portfolio_copula_200_assets_20k_trials_4_periods": bench_portfolio_copula,
    "portfolio_frailty_200_assets_20k_trials_4_periods": bench_portfolio_frailty,
    "portfolio_1000_assets_10k_trials_1_period": bench_portfolio_large,
}


def run(output_path: str) -> None:
    import simflux

    results = {
        "_meta": {
            "simflux_version": simflux.__version__,
            "rust_backend": Backend.is_available(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "loadavg_1m": (
                round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None
            ),
        }
    }
    any_unstable = False
    for name, fn in BENCHES.items():
        stats = fn()
        results[name] = stats
        flag = "  UNSTABLE" if stats["unstable"] else ""
        print(
            f"{name:50s} median={stats['median_s']:.4f}s "
            f"±{stats['iqr_s']:.4f} (IQR, n={stats['repeats']}) "
            f"cv={stats['cv']:.1%}{flag}",
            flush=True,
        )
        any_unstable = any_unstable or stats["unstable"]
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    if any_unstable:
        print(
            f"WARNING: rows flagged UNSTABLE exceeded {UNSTABLE_CV:.0%} coefficient "
            "of variation — the environment is noisy (CPU contention, frequency "
            "scaling); do not compare those rows against other runs."
        )


def compare(before_path: str, after_path: str) -> None:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)
    # Compare medians (robust to a single outlier repeat); carry the
    # instability flags through so a noisy comparison is visibly noisy.
    print(f"{'benchmark':50s} {'before':>9s} {'after':>9s} {'speedup':>8s}")
    for name in BENCHES:
        if name in before and name in after:
            b, a = before[name]["median_s"], after[name]["median_s"]
            flag = (
                "  (unstable)"
                if before[name].get("unstable") or after[name].get("unstable")
                else ""
            )
            print(f"{name:50s} {b:8.4f}s {a:8.4f}s {b / a:7.2f}x{flag}")


# Workloads used for the --scaling sweep: the parallel Rust kernels (the GBM
# and portfolio trial loops parallelize over paths/trials via Rayon; the
# calibration-heavy frailty workload is excluded because its Python share
# would dilute the parallel-efficiency signal).
SCALING_BENCHES = [
    "gbm_50k_paths_252_steps",
    "correlated_gbm_10_assets_10k_paths",
    "portfolio_1000_assets_10k_trials_1_period",
]


def run_subset(output_path: str, names: list) -> None:
    """Run only the named benches (used by the --scaling child processes)."""
    results = {}
    for name in names:
        results[name] = BENCHES[name]()
    with open(output_path, "w") as f:
        json.dump(results, f)


def scaling(max_threads: int) -> None:
    """Measure parallel scaling: throughput at 1, 2, ..., N Rayon threads.

    RAYON_NUM_THREADS must be set before the thread pool initializes, so each
    thread count runs in a fresh subprocess. Parallel efficiency is
    ``T(1) / (t * T(t))`` — 100% means perfect scaling; a kernel whose
    efficiency collapses has stopped scaling, which single-configuration
    numbers cannot show.
    """
    import subprocess
    import sys
    import tempfile

    thread_counts = sorted({1, 2, max_threads} | {max_threads})
    thread_counts = [t for t in thread_counts if 1 <= t <= max_threads]

    measured: dict[int, dict] = {}
    for t in thread_counts:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            out = tmp.name
        env = dict(os.environ, RAYON_NUM_THREADS=str(t))
        print(f"measuring with RAYON_NUM_THREADS={t} ...", flush=True)
        subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                out,
                "--subset",
                ",".join(SCALING_BENCHES),
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        with open(out) as f:
            measured[t] = json.load(f)
        os.unlink(out)

    print(f"\n{'benchmark':45s} " + "".join(f"{t:>7d}T" for t in thread_counts))
    for name in SCALING_BENCHES:
        row = [measured[t][name]["median_s"] for t in thread_counts]
        print(f"{name:45s} " + "".join(f"{v:7.3f}s" for v in row))
        t1 = row[0]
        effs = [t1 / (t * v) if v > 0 else 0.0 for t, v in zip(thread_counts, row)]
        print(f"{'  parallel efficiency':45s} " + "".join(f"{e:7.0%} " for e in effs))
        unstable = [
            t for t in thread_counts if measured[t][name].get("unstable", False)
        ]
        if unstable:
            print(f"{'  UNSTABLE at':45s} {unstable} — treat this row as noise")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="regression_results.json")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="compare two saved result files instead of running",
    )
    parser.add_argument(
        "--subset",
        help="comma-separated bench names to run (used by --scaling children)",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="sweep RAYON_NUM_THREADS (1, 2, N) and report parallel efficiency",
    )
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.scaling:
        scaling(os.cpu_count() or 1)
    elif args.subset:
        run_subset(args.output, args.subset.split(","))
    else:
        run(args.output)


if __name__ == "__main__":
    main()
