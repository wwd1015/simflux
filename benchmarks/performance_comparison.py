"""
Performance comparison between Rust backend and NumPy fallback implementations.

This benchmark script compares:
1. Single GBM simulation
2. Correlated GBM simulation
3. Portfolio simulation
4. Memory usage
5. Scalability across different problem sizes
"""

import time
import sys
import json
import resource
import subprocess
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
import gc
import os

# Import simflux with both backends
import simflux as sf


# ---------------------------------------------------------------------------
# Isolated worker: each measurement runs in its own subprocess so that GC and
# allocator caching from one workload cannot contaminate another's memory
# reading. Memory is the kernel's peak RSS (ru_maxrss) reached during the call,
# above the pre-call baseline — the only reliable cross-backend memory metric
# (it captures Rust allocations too, which live outside Python's allocator).
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Current resident set size, in MB."""
    return psutil.Process().memory_info().rss / 1e6


def _peak_rss_mb() -> float:
    """Process peak resident set size, in MB. ru_maxrss is bytes on macOS,
    kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def _build_call(spec: Dict):
    """Reconstruct the zero-arg callable for one (test, backend, params) workload."""
    test, backend = spec["test"], spec["backend"]

    if test == "GBM":
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        if backend == "rust":
            return lambda: gbm.simulate(
                n_paths=spec["n_paths"], n_steps=spec["n_steps"], T=1.0
            )
        engine = gbm.engine
        return lambda: engine._numpy_simulate_gbm(
            mu=0.05,
            sigma=0.2,
            s0=100,
            n_paths=spec["n_paths"],
            n_steps=spec["n_steps"],
            T=1.0,
        )

    if test == "Correlated_GBM":
        n = spec["n_assets"]
        cm = sf.utils.generate_correlation_matrix(
            n=n, correlation_strength=0.3, random_state=42
        )
        mu = [0.05 + 0.01 * i for i in range(n)]
        sigma = [0.15 + 0.02 * i for i in range(n)]
        s0 = [100 + 10 * i for i in range(n)]
        cgbm = sf.CorrelatedGBM(mu=mu, sigma=sigma, S0=s0, correlation_matrix=cm)
        if backend == "rust":
            return lambda: cgbm.simulate(
                n_paths=spec["n_paths"], n_steps=spec["n_steps"], T=1.0
            )
        engine = cgbm.engine
        return lambda: engine._numpy_simulate_gbm_correlated(
            mu=mu,
            sigma=sigma,
            s0=s0,
            correlation_matrix=cm,
            n_paths=spec["n_paths"],
            n_steps=spec["n_steps"],
            T=1.0,
        )

    if test == "Portfolio":
        portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
            n_assets_per_sector=spec["n_assets"] // 2,
            sectors=["Technology", "Finance"],
            inter_sector_correlation=0.15,
        )
        if backend == "rust":
            return lambda: portfolio.simulate(n_simulations=spec["n_simulations"])
        # Exercise the NumPy adapter directly (single-period copula default), so the
        # benchmark measures the fallback compute, not the dispatch.
        return lambda: portfolio._simulate_portfolio_numpy(
            n_simulations=spec["n_simulations"],
            n_periods=1,
            period_length=1.0,
            default_timing="copula",
            factor_phi=0.0,
            barriers=None,
            store_interim=False,
            output_path=None,
            batch_size=None,
        )

    raise ValueError(f"unknown test {test!r}")


def run_worker(spec: Dict) -> Dict:
    """Run one workload in this (isolated) process; report time + peak memory."""
    try:
        call = _build_call(spec)
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"build: {exc}"}

    gc.collect()
    rss_before = _rss_mb()
    start = time.perf_counter()
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}
    elapsed = time.perf_counter() - start
    peak = _peak_rss_mb()  # result is still alive, so its allocation is counted
    del result
    return {
        "success": True,
        "time": elapsed,
        "memory_mb": max(0.0, peak - rss_before),
    }


class BenchmarkRunner:
    """Run performance benchmarks comparing Rust vs NumPy backends."""

    def __init__(self):
        self.results = []
        self.process = psutil.Process()

    def measure(self, spec: Dict) -> Dict:
        """Measure one workload in an isolated subprocess.

        Each call forks a fresh interpreter that runs exactly one simulation and
        reports its wall time and peak-RSS growth, so memory readings are immune
        to GC timing and allocator caching from other workloads in this run.
        """
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(spec)],
            capture_output=True,
            text=True,
        )
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            err = (proc.stderr or proc.stdout or "no output").strip()[-400:]
            return {
                "execution_time": 0.0,
                "memory_used_mb": 0.0,
                "success": False,
                "error": err,
            }

        if not data.get("success"):
            return {
                "execution_time": 0.0,
                "memory_used_mb": 0.0,
                "success": False,
                "error": data.get("error", "worker failed"),
            }
        return {
            "execution_time": data["time"],
            "memory_used_mb": data["memory_mb"],
            "success": True,
            "error": None,
        }

    def run_gbm_benchmark(self) -> List[Dict]:
        """Benchmark single asset GBM simulation."""
        print("\n" + "=" * 60)
        print("BENCHMARK 1: Single Asset GBM Simulation")
        print("=" * 60)

        test_cases = [
            (1000, 252, "Small"),  # 1K paths, 1 year daily
            (10000, 252, "Medium"),  # 10K paths, 1 year daily
            (50000, 252, "Large"),  # 50K paths, 1 year daily
            (10000, 1260, "Long Term"),  # 10K paths, 5 years daily
        ]

        results = []

        for n_paths, n_steps, size_label in test_cases:
            print(f"\nTesting {size_label}: {n_paths:,} paths × {n_steps} steps")

            base = {"test": "GBM", "n_paths": n_paths, "n_steps": n_steps}
            current_result = self.measure({**base, "backend": "rust"})
            numpy_result = self.measure({**base, "backend": "numpy"})

            result_entry = {
                "test": "GBM",
                "size": size_label,
                "n_paths": n_paths,
                "n_steps": n_steps,
                "current_backend_time": current_result["execution_time"],
                "current_backend_memory": current_result["memory_used_mb"],
                "numpy_time": numpy_result["execution_time"],
                "numpy_memory": numpy_result["memory_used_mb"],
                "current_success": current_result["success"],
                "numpy_success": numpy_result["success"],
            }

            if current_result["success"] and numpy_result["success"]:
                speedup = (
                    numpy_result["execution_time"] / current_result["execution_time"]
                )
                memory_ratio = numpy_result["memory_used_mb"] / max(
                    current_result["memory_used_mb"], 1
                )
                result_entry["speedup"] = speedup
                result_entry["memory_ratio"] = memory_ratio

                print(
                    f"  Current Backend: {current_result['execution_time']:.3f}s, "
                    f"{current_result['memory_used_mb']:.1f}MB"
                )
                print(
                    f"  NumPy Fallback:  {numpy_result['execution_time']:.3f}s, "
                    f"{numpy_result['memory_used_mb']:.1f}MB"
                )
                mem_x = current_result["memory_used_mb"] / max(
                    numpy_result["memory_used_mb"], 1e-6
                )
                print(
                    f"  Speedup: {speedup:.1f}x faster, "
                    f"Rust uses {mem_x:.2f}x the memory"
                )

            results.append(result_entry)

        return results

    def run_correlated_gbm_benchmark(self) -> List[Dict]:
        """Benchmark multi-asset correlated GBM simulation."""
        print("\n" + "=" * 60)
        print("BENCHMARK 2: Correlated Multi-Asset GBM Simulation")
        print("=" * 60)

        test_cases = [
            (2, 1000, 252, "2 Assets"),
            (5, 1000, 252, "5 Assets"),
            (10, 5000, 252, "10 Assets"),
            (20, 1000, 252, "20 Assets"),
        ]

        results = []

        for n_assets, n_paths, n_steps, size_label in test_cases:
            print(
                f"\nTesting {size_label}: {n_assets} assets × {n_paths:,} paths × {n_steps} steps"
            )

            base = {
                "test": "Correlated_GBM",
                "n_assets": n_assets,
                "n_paths": n_paths,
                "n_steps": n_steps,
            }
            current_result = self.measure({**base, "backend": "rust"})
            numpy_result = self.measure({**base, "backend": "numpy"})

            result_entry = {
                "test": "Correlated_GBM",
                "size": size_label,
                "n_assets": n_assets,
                "n_paths": n_paths,
                "n_steps": n_steps,
                "current_backend_time": current_result["execution_time"],
                "current_backend_memory": current_result["memory_used_mb"],
                "numpy_time": numpy_result["execution_time"],
                "numpy_memory": numpy_result["memory_used_mb"],
                "current_success": current_result["success"],
                "numpy_success": numpy_result["success"],
            }

            if current_result["success"] and numpy_result["success"]:
                speedup = (
                    numpy_result["execution_time"] / current_result["execution_time"]
                )
                memory_ratio = numpy_result["memory_used_mb"] / max(
                    current_result["memory_used_mb"], 1
                )
                result_entry["speedup"] = speedup
                result_entry["memory_ratio"] = memory_ratio

                print(
                    f"  Current Backend: {current_result['execution_time']:.3f}s, "
                    f"{current_result['memory_used_mb']:.1f}MB"
                )
                print(
                    f"  NumPy Fallback:  {numpy_result['execution_time']:.3f}s, "
                    f"{numpy_result['memory_used_mb']:.1f}MB"
                )
                mem_x = current_result["memory_used_mb"] / max(
                    numpy_result["memory_used_mb"], 1e-6
                )
                print(
                    f"  Speedup: {speedup:.1f}x faster, "
                    f"Rust uses {mem_x:.2f}x the memory"
                )

            results.append(result_entry)

        return results

    def run_portfolio_benchmark(self) -> List[Dict]:
        """Benchmark portfolio loss simulation."""
        print("\n" + "=" * 60)
        print("BENCHMARK 3: Portfolio Loss Simulation")
        print("=" * 60)

        test_cases = [
            (50, 1000, "Small Portfolio"),
            (200, 5000, "Medium Portfolio"),
            (500, 2000, "Large Portfolio"),
            (100, 10000, "Many Simulations"),
        ]

        results = []

        for n_assets, n_simulations, size_label in test_cases:
            print(
                f"\nTesting {size_label}: {n_assets} assets × {n_simulations:,} simulations"
            )

            base = {
                "test": "Portfolio",
                "n_assets": n_assets,
                "n_simulations": n_simulations,
            }
            current_result = self.measure({**base, "backend": "rust"})
            numpy_result = self.measure({**base, "backend": "numpy"})

            result_entry = {
                "test": "Portfolio",
                "size": size_label,
                "n_assets": n_assets,
                "n_simulations": n_simulations,
                "current_backend_time": current_result["execution_time"],
                "current_backend_memory": current_result["memory_used_mb"],
                "numpy_time": numpy_result["execution_time"],
                "numpy_memory": numpy_result["memory_used_mb"],
                "current_success": current_result["success"],
                "numpy_success": numpy_result["success"],
            }

            if current_result["success"] and numpy_result["success"]:
                speedup = (
                    numpy_result["execution_time"] / current_result["execution_time"]
                )
                memory_ratio = numpy_result["memory_used_mb"] / max(
                    current_result["memory_used_mb"], 1
                )
                result_entry["speedup"] = speedup
                result_entry["memory_ratio"] = memory_ratio

                print(
                    f"  Current Backend: {current_result['execution_time']:.3f}s, "
                    f"{current_result['memory_used_mb']:.1f}MB"
                )
                print(
                    f"  NumPy Fallback:  {numpy_result['execution_time']:.3f}s, "
                    f"{numpy_result['memory_used_mb']:.1f}MB"
                )
                mem_x = current_result["memory_used_mb"] / max(
                    numpy_result["memory_used_mb"], 1e-6
                )
                print(
                    f"  Speedup: {speedup:.1f}x faster, "
                    f"Rust uses {mem_x:.2f}x the memory"
                )
            else:
                if not current_result["success"]:
                    print(f"  Current Backend Error: {current_result['error']}")
                if not numpy_result["success"]:
                    print(f"  NumPy Fallback Error: {numpy_result['error']}")

            results.append(result_entry)

        return results

    def detect_backend(self) -> str:
        """Detect which backend is currently being used."""
        try:
            # Try to access Rust backend
            from simflux.core.engine import RUST_AVAILABLE

            return "Rust" if RUST_AVAILABLE else "NumPy Fallback"
        except Exception:
            return "Unknown"

    def run_all_benchmarks(self) -> pd.DataFrame:
        """Run all benchmarks and return results as DataFrame."""

        backend = self.detect_backend()
        print(f"SimFlux Backend Detected: {backend}")
        print(f"Python Version: {sys.version}")
        print(f"NumPy Version: {np.__version__}")
        print(f"System: {sys.platform}")
        print(f"CPU Count: {os.cpu_count()}")
        print(f"Available Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")

        all_results = []

        # Run individual benchmarks
        gbm_results = self.run_gbm_benchmark()
        all_results.extend(gbm_results)

        corr_gbm_results = self.run_correlated_gbm_benchmark()
        all_results.extend(corr_gbm_results)

        portfolio_results = self.run_portfolio_benchmark()
        all_results.extend(portfolio_results)

        return pd.DataFrame(all_results)

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comprehensive benchmark report."""

        report = [
            "\n" + "=" * 80,
            "SIMFLUX PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Backend: {self.detect_backend()}",
            f"Timestamp: {pd.Timestamp.now()}",
            f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}",
            "",
        ]

        # Summary statistics
        successful_results = results_df[
            results_df["current_success"] & results_df["numpy_success"]
        ]

        if not successful_results.empty:
            avg_speedup = successful_results["speedup"].mean()
            max_speedup = successful_results["speedup"].max()
            avg_memory_ratio = successful_results["memory_ratio"].mean()

            report.extend(
                [
                    "SUMMARY STATISTICS:",
                    f"  Average Speedup (Current/NumPy): {avg_speedup:.1f}x",
                    f"  Maximum Speedup: {max_speedup:.1f}x",
                    f"  Average Memory Efficiency: {1/avg_memory_ratio:.1f}x",
                    "",
                ]
            )

        # Detailed results by test type
        for test_type in results_df["test"].unique():
            test_results = results_df[results_df["test"] == test_type]
            successful_test = test_results[
                test_results["current_success"] & test_results["numpy_success"]
            ]

            report.extend([f"{test_type.upper()} RESULTS:", "-" * 40])

            if not successful_test.empty:
                for _, row in successful_test.iterrows():
                    report.append(
                        f"  {row['size']:15} | "
                        f"Current: {row['current_backend_time']:6.3f}s | "
                        f"NumPy: {row['numpy_time']:6.3f}s | "
                        f"Speedup: {row['speedup']:5.1f}x"
                    )

            report.append("")

        # Performance scaling analysis
        gbm_results = successful_results[successful_results["test"] == "GBM"]
        if not gbm_results.empty:
            report.extend(["SCALING ANALYSIS:", "-" * 40])

            for _, row in gbm_results.iterrows():
                ops_per_sec = (row["n_paths"] * row["n_steps"]) / row[
                    "current_backend_time"
                ]
                report.append(
                    f"  {row['size']:15} | " f"Throughput: {ops_per_sec:10,.0f} ops/sec"
                )
            report.append("")

        # Memory analysis
        if not successful_results.empty:
            report.extend(["MEMORY USAGE ANALYSIS:", "-" * 40])

            for _, row in successful_results.iterrows():
                report.append(
                    f"  {row['test']:12} {row['size']:15} | "
                    f"Current: {row['current_backend_memory']:6.1f}MB | "
                    f"NumPy: {row['numpy_memory']:6.1f}MB | "
                    f"Ratio: {row['memory_ratio']:5.1f}x"
                )
            report.append("")

        # Recommendations
        report.extend(
            [
                "RECOMMENDATIONS:",
                "-" * 40,
                "• Use Rust backend for production workloads (significant performance gains)",
                "• NumPy fallback suitable for development and small-scale testing",
                "• Consider memory constraints for large simulations",
                "• Binary wheel deployment recommended for best user experience",
                "",
            ]
        )

        return "\n".join(report)


def main():
    """Main benchmark execution."""
    print("Starting SimFlux Performance Benchmarks...")
    print("This may take several minutes to complete.")

    # Suppress warnings during benchmarking
    warnings.filterwarnings("ignore")

    runner = BenchmarkRunner()
    results_df = runner.run_all_benchmarks()

    # Generate and print report
    report = runner.generate_report(results_df)
    print(report)

    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"simflux_benchmark_{timestamp}.csv"
    report_filename = f"simflux_benchmark_report_{timestamp}.txt"

    results_df.to_csv(csv_filename, index=False)
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"\nResults saved to: {csv_filename}")
    print(f"Report saved to: {report_filename}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        # Isolated single-workload worker (spawned by BenchmarkRunner.measure).
        warnings.filterwarnings("ignore")
        print(json.dumps(run_worker(json.loads(sys.argv[2]))))
    else:
        main()
