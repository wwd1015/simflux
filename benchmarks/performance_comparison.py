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
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
import gc
import os

# Import simflux with both backends
import simflux as sf

class BenchmarkRunner:
    """Run performance benchmarks comparing Rust vs NumPy backends."""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
        
    def measure_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_function(self, func, *args, **kwargs) -> Dict:
        """Benchmark a function call with timing and memory measurement."""
        # Clear memory before test
        gc.collect()
        memory_before = self.measure_memory()
        
        # Time the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            success = True
            error = None
        except Exception as e:
            end_time = time.time()
            result = None
            success = False
            error = str(e)
        
        # Measure memory after
        memory_after = self.measure_memory()
        
        return {
            'execution_time': end_time - start_time,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_after - memory_before,
            'success': success,
            'error': error,
            'result_shape': getattr(result, 'shape', None) if hasattr(result, 'shape') else None
        }
    
    def run_gbm_benchmark(self) -> List[Dict]:
        """Benchmark single asset GBM simulation."""
        print("\n" + "="*60)
        print("BENCHMARK 1: Single Asset GBM Simulation")
        print("="*60)
        
        test_cases = [
            (1000, 252, "Small"),      # 1K paths, 1 year daily
            (10000, 252, "Medium"),    # 10K paths, 1 year daily  
            (50000, 252, "Large"),     # 50K paths, 1 year daily
            (10000, 1260, "Long Term") # 10K paths, 5 years daily
        ]
        
        results = []
        
        for n_paths, n_steps, size_label in test_cases:
            print(f"\nTesting {size_label}: {n_paths:,} paths × {n_steps} steps")
            
            # Test with current backend (could be Rust or NumPy)
            gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
            current_result = self.benchmark_function(
                gbm.simulate, n_paths=n_paths, n_steps=n_steps, T=1.0
            )
            
            # Force NumPy fallback test
            engine = gbm.engine
            numpy_result = self.benchmark_function(
                engine._numpy_simulate_gbm, 
                mu=0.05, sigma=0.2, s0=100, 
                n_paths=n_paths, n_steps=n_steps, T=1.0
            )
            
            result_entry = {
                'test': 'GBM',
                'size': size_label,
                'n_paths': n_paths,
                'n_steps': n_steps,
                'current_backend_time': current_result['execution_time'],
                'current_backend_memory': current_result['memory_used_mb'],
                'numpy_time': numpy_result['execution_time'],
                'numpy_memory': numpy_result['memory_used_mb'],
                'current_success': current_result['success'],
                'numpy_success': numpy_result['success']
            }
            
            if current_result['success'] and numpy_result['success']:
                speedup = numpy_result['execution_time'] / current_result['execution_time']
                memory_ratio = numpy_result['memory_used_mb'] / max(current_result['memory_used_mb'], 1)
                result_entry['speedup'] = speedup
                result_entry['memory_ratio'] = memory_ratio
                
                print(f"  Current Backend: {current_result['execution_time']:.3f}s, "
                      f"{current_result['memory_used_mb']:.1f}MB")
                print(f"  NumPy Fallback:  {numpy_result['execution_time']:.3f}s, "
                      f"{numpy_result['memory_used_mb']:.1f}MB")
                print(f"  Speedup: {speedup:.1f}x, Memory Efficiency: {1/memory_ratio:.1f}x")
            
            results.append(result_entry)
        
        return results
    
    def run_correlated_gbm_benchmark(self) -> List[Dict]:
        """Benchmark multi-asset correlated GBM simulation."""
        print("\n" + "="*60)
        print("BENCHMARK 2: Correlated Multi-Asset GBM Simulation")
        print("="*60)
        
        test_cases = [
            (2, 1000, 252, "2 Assets"),
            (5, 1000, 252, "5 Assets"),
            (10, 5000, 252, "10 Assets"),
            (20, 1000, 252, "20 Assets")
        ]
        
        results = []
        
        for n_assets, n_paths, n_steps, size_label in test_cases:
            print(f"\nTesting {size_label}: {n_assets} assets × {n_paths:,} paths × {n_steps} steps")
            
            # Generate correlation matrix
            correlation_matrix = sf.generate_correlation_matrix(
                n=n_assets, correlation_strength=0.3, random_state=42
            )
            
            # Create correlated GBM
            mu = [0.05 + 0.01 * i for i in range(n_assets)]
            sigma = [0.15 + 0.02 * i for i in range(n_assets)]  
            s0 = [100 + 10 * i for i in range(n_assets)]
            
            corr_gbm = sf.CorrelatedGBM(
                mu=mu, sigma=sigma, S0=s0,
                correlation_matrix=correlation_matrix
            )
            
            # Test current backend
            current_result = self.benchmark_function(
                corr_gbm.simulate, n_paths=n_paths, n_steps=n_steps, T=1.0
            )
            
            # Test NumPy fallback
            engine = corr_gbm.engine
            numpy_result = self.benchmark_function(
                engine._numpy_simulate_gbm_correlated,
                mu=mu, sigma=sigma, s0=s0, 
                correlation_matrix=correlation_matrix,
                n_paths=n_paths, n_steps=n_steps, T=1.0
            )
            
            result_entry = {
                'test': 'Correlated_GBM',
                'size': size_label,
                'n_assets': n_assets,
                'n_paths': n_paths, 
                'n_steps': n_steps,
                'current_backend_time': current_result['execution_time'],
                'current_backend_memory': current_result['memory_used_mb'],
                'numpy_time': numpy_result['execution_time'],
                'numpy_memory': numpy_result['memory_used_mb'],
                'current_success': current_result['success'],
                'numpy_success': numpy_result['success']
            }
            
            if current_result['success'] and numpy_result['success']:
                speedup = numpy_result['execution_time'] / current_result['execution_time']
                memory_ratio = numpy_result['memory_used_mb'] / max(current_result['memory_used_mb'], 1)
                result_entry['speedup'] = speedup
                result_entry['memory_ratio'] = memory_ratio
                
                print(f"  Current Backend: {current_result['execution_time']:.3f}s, "
                      f"{current_result['memory_used_mb']:.1f}MB")
                print(f"  NumPy Fallback:  {numpy_result['execution_time']:.3f}s, "
                      f"{numpy_result['memory_used_mb']:.1f}MB") 
                print(f"  Speedup: {speedup:.1f}x, Memory Efficiency: {1/memory_ratio:.1f}x")
            
            results.append(result_entry)
        
        return results
    
    def run_portfolio_benchmark(self) -> List[Dict]:
        """Benchmark portfolio loss simulation."""
        print("\n" + "="*60)
        print("BENCHMARK 3: Portfolio Loss Simulation")
        print("="*60)
        
        test_cases = [
            (50, 1000, "Small Portfolio"),
            (200, 5000, "Medium Portfolio"), 
            (500, 2000, "Large Portfolio"),
            (100, 10000, "Many Simulations")
        ]
        
        results = []
        
        for n_assets, n_simulations, size_label in test_cases:
            print(f"\nTesting {size_label}: {n_assets} assets × {n_simulations:,} simulations")
            
            # Create portfolio
            portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(
                n_assets_per_sector=n_assets//2,
                sectors=['Technology', 'Finance'],
                inter_sector_correlation=0.15
            )
            
            # Test current backend (might use fallback if Rust unavailable)
            current_result = self.benchmark_function(
                portfolio.simulate, n_simulations=n_simulations
            )
            
            # Test NumPy fallback directly  
            numpy_result = self.benchmark_function(
                portfolio._fallback_simulate_portfolio,
                n_simulations=n_simulations
            )
            
            result_entry = {
                'test': 'Portfolio',
                'size': size_label,
                'n_assets': n_assets,
                'n_simulations': n_simulations,
                'current_backend_time': current_result['execution_time'],
                'current_backend_memory': current_result['memory_used_mb'],
                'numpy_time': numpy_result['execution_time'],
                'numpy_memory': numpy_result['memory_used_mb'],
                'current_success': current_result['success'],
                'numpy_success': numpy_result['success']
            }
            
            if current_result['success'] and numpy_result['success']:
                speedup = numpy_result['execution_time'] / current_result['execution_time']
                memory_ratio = numpy_result['memory_used_mb'] / max(current_result['memory_used_mb'], 1)
                result_entry['speedup'] = speedup
                result_entry['memory_ratio'] = memory_ratio
                
                print(f"  Current Backend: {current_result['execution_time']:.3f}s, "
                      f"{current_result['memory_used_mb']:.1f}MB")
                print(f"  NumPy Fallback:  {numpy_result['execution_time']:.3f}s, "
                      f"{numpy_result['memory_used_mb']:.1f}MB")
                print(f"  Speedup: {speedup:.1f}x, Memory Efficiency: {1/memory_ratio:.1f}x")
            else:
                if not current_result['success']:
                    print(f"  Current Backend Error: {current_result['error']}")
                if not numpy_result['success']:
                    print(f"  NumPy Fallback Error: {numpy_result['error']}")
            
            results.append(result_entry)
        
        return results
    
    def detect_backend(self) -> str:
        """Detect which backend is currently being used."""
        try:
            # Try to access Rust backend
            from simflux.core.engine import RUST_AVAILABLE
            return "Rust" if RUST_AVAILABLE else "NumPy Fallback"
        except:
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
            "\n" + "="*80,
            "SIMFLUX PERFORMANCE BENCHMARK REPORT", 
            "="*80,
            f"Backend: {self.detect_backend()}",
            f"Timestamp: {pd.Timestamp.now()}",
            f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}",
            ""
        ]
        
        # Summary statistics
        successful_results = results_df[results_df['current_success'] & results_df['numpy_success']]
        
        if not successful_results.empty:
            avg_speedup = successful_results['speedup'].mean()
            max_speedup = successful_results['speedup'].max()
            avg_memory_ratio = successful_results['memory_ratio'].mean()
            
            report.extend([
                "SUMMARY STATISTICS:",
                f"  Average Speedup (Current/NumPy): {avg_speedup:.1f}x",
                f"  Maximum Speedup: {max_speedup:.1f}x", 
                f"  Average Memory Efficiency: {1/avg_memory_ratio:.1f}x",
                ""
            ])
        
        # Detailed results by test type
        for test_type in results_df['test'].unique():
            test_results = results_df[results_df['test'] == test_type]
            successful_test = test_results[test_results['current_success'] & test_results['numpy_success']]
            
            report.extend([
                f"{test_type.upper()} RESULTS:",
                "-" * 40
            ])
            
            if not successful_test.empty:
                for _, row in successful_test.iterrows():
                    report.append(f"  {row['size']:15} | "
                                f"Current: {row['current_backend_time']:6.3f}s | "
                                f"NumPy: {row['numpy_time']:6.3f}s | "
                                f"Speedup: {row['speedup']:5.1f}x")
            
            report.append("")
        
        # Performance scaling analysis
        gbm_results = successful_results[successful_results['test'] == 'GBM']
        if not gbm_results.empty:
            report.extend([
                "SCALING ANALYSIS:",
                "-" * 40
            ])
            
            for _, row in gbm_results.iterrows():
                ops_per_sec = (row['n_paths'] * row['n_steps']) / row['current_backend_time']
                report.append(f"  {row['size']:15} | "
                            f"Throughput: {ops_per_sec:10,.0f} ops/sec")
            report.append("")
        
        # Memory analysis
        if not successful_results.empty:
            report.extend([
                "MEMORY USAGE ANALYSIS:",
                "-" * 40
            ])
            
            for _, row in successful_results.iterrows():
                report.append(f"  {row['test']:12} {row['size']:15} | "
                            f"Current: {row['current_backend_memory']:6.1f}MB | "
                            f"NumPy: {row['numpy_memory']:6.1f}MB | "
                            f"Ratio: {row['memory_ratio']:5.1f}x")
            report.append("")
        
        # Recommendations
        report.extend([
            "RECOMMENDATIONS:",
            "-" * 40,
            "• Use Rust backend for production workloads (significant performance gains)",
            "• NumPy fallback suitable for development and small-scale testing",
            "• Consider memory constraints for large simulations",
            "• Binary wheel deployment recommended for best user experience",
            ""
        ])
        
        return "\n".join(report)


def main():
    """Main benchmark execution."""
    print("Starting SimFlux Performance Benchmarks...")
    print("This may take several minutes to complete.")
    
    # Suppress warnings during benchmarking
    warnings.filterwarnings('ignore')
    
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
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: {csv_filename}")
    print(f"Report saved to: {report_filename}")


if __name__ == "__main__":
    main()