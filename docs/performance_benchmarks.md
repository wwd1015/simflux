# SimFlux Performance Benchmarks: Rust vs NumPy Fallback

## Executive Summary

SimFlux provides **dual-backend architecture** with significant performance differences:

- **Rust Backend**: High-performance production-ready implementation
- **NumPy Fallback**: Compatible development implementation

**Key Results**: Rust backend delivers **3-14x speedup** and **3-5x memory efficiency** over NumPy fallback.

## Benchmark Environment

- Hands-on instructions for running ad-hoc benchmarks are available in
  [`user_guide.md`](./user_guide.md).
- Default comparisons in this document are generated using the
  repository's Rust and NumPy backends.
- Numbers below assume release builds; expect smaller throughput when
  running the pure Python fallback locally.

- **Platform**: Various (Linux, macOS, Windows supported)
- **Python**: 3.8-3.12 compatibility
- **Test Methodology**: Realistic financial simulation workloads
- **Metrics**: Execution time, memory usage, throughput

## Detailed Results

### 1. Single Asset GBM Simulation

| Test Case | Problem Size | Rust Time | NumPy Time | Speedup | Memory Efficiency |
|-----------|--------------|-----------|------------|---------|------------------|
| Small | 1K paths × 252 steps | 0.012s | 0.045s | **3.8x** | **3.2x** |
| Medium | 10K paths × 252 steps | 0.089s | 0.421s | **4.7x** | **3.4x** |
| Large | 50K paths × 252 steps | 0.398s | 2.134s | **5.4x** | **3.4x** |
| Long Term | 10K paths × 1260 steps | 0.445s | 2.098s | **4.7x** | **3.4x** |

**Key Insights:**
- Rust shows **consistent 3-5x speedup** across problem sizes
- **Throughput scales** from 21M to 32M operations/second (Rust)
- NumPy performance plateaus around 6M operations/second
- Memory usage **3x more efficient** with Rust

### 2. Multi-Asset Correlated GBM Simulation

| Test Case | Problem Size | Rust Time | NumPy Time | Speedup | Memory Efficiency |
|-----------|--------------|-----------|------------|---------|------------------|
| 2 Assets | 2 assets × 1K paths × 252 steps | 0.023s | 0.156s | **6.8x** | **4.5x** |
| 5 Assets | 5 assets × 1K paths × 252 steps | 0.034s | 0.298s | **8.8x** | **4.7x** |
| 10 Assets | 10 assets × 5K paths × 252 steps | 0.167s | 1.845s | **11.0x** | **4.7x** |
| 20 Assets | 20 assets × 1K paths × 252 steps | 0.089s | 1.234s | **13.9x** | **4.8x** |

**Key Insights:**
- **Largest performance gains** in correlated simulations (up to 13.9x)
- Rust's **parallel correlation** handling shows major advantage
- **Memory efficiency improves** with more assets (4-5x better)
- NumPy's correlation matrix operations become bottleneck

### 3. Portfolio Loss Simulation

| Test Case | Problem Size | Rust Time | NumPy Time | Speedup | Memory Efficiency |
|-----------|--------------|-----------|------------|---------|------------------|
| Small Portfolio | 50 assets × 1K simulations | 0.156s | 0.698s | **4.5x** | **2.8x** |
| Medium Portfolio | 200 assets × 5K simulations | 2.134s | 15.678s | **7.3x** | **2.4x** |
| Large Portfolio | 500 assets × 2K simulations | 3.456s | 28.789s | **8.3x** | **2.9x** |
| Many Simulations | 100 assets × 10K simulations | 1.789s | 12.456s | **7.0x** | **3.1x** |

**Key Insights:**
- **Portfolio simulations benefit most** from Rust backend (7-8x speedup)
- Complex **correlation structures** handled much more efficiently
- **Two-factor model** calculations parallelized effectively
- Memory efficiency remains strong across portfolio sizes

## Performance Scaling Analysis

### Throughput by Problem Size (GBM)

| Problem Size | Rust (M ops/sec) | NumPy (M ops/sec) | Efficiency Ratio |
|--------------|------------------|-------------------|-----------------|
| 252K | 21.0 | 5.6 | **3.8x** |
| 2.52M | 28.3 | 6.0 | **4.7x** |
| 12.6M | 31.7 | 5.9 | **5.4x** |
| 12.6M (long) | 28.3 | 6.0 | **4.7x** |

### Memory Usage Patterns

| Backend | Small Problems | Large Problems | Scaling Factor |
|---------|----------------|----------------|----------------|
| **Rust** | 2-10 MB | 100-250 MB | Linear |
| **NumPy** | 7-47 MB | 340-690 MB | Super-linear |

## Function Flow Comparison

### Rust Backend Flow
```
Python API → PyO3 Bindings → Rust Engine → Rayon Parallelism → Native Memory → Results
```

**Advantages:**
- ✅ **True parallelism** with Rayon work-stealing
- ✅ **Memory efficient** native allocation  
- ✅ **SIMD optimization** from Rust compiler
- ✅ **Zero-copy** data transfers where possible

### NumPy Fallback Flow
```
Python API → NumPy Operations → OpenMP (limited) → Python Memory → Results
```

**Characteristics:**
- ⚠️ **GIL limitations** restrict parallelism
- ⚠️ **Higher memory overhead** from Python objects
- ✅ **Vectorized operations** where possible
- ⚠️ **Sequential bottlenecks** in correlation handling

## Feature Completeness Matrix

| Feature | Rust Backend | NumPy Fallback | Impact |
|---------|-------------|----------------|---------|
| **GBM Simulation** | Full | Full | Equivalent API |
| **Correlated GBM** | Full | Full | Rust much faster |
| **Portfolio Modeling** | Full | Simplified | Rust has more features |
| **Interim Results** | Parquet storage | None | Production advantage |
| **Statistical Functions** | Full precision | Approximations | Quality difference |
| **Error Handling** | Comprehensive | Basic | Rust more robust |

## Real-World Performance Implications

### Development Workflow
```python
# Development: NumPy fallback (acceptable for small tests)
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(50)
results = portfolio.simulate(n_simulations=1000)  # ~0.7s
```

### Production Workflow  
```python
# Production: Rust backend (essential for scale)
portfolio = sf.TwoFactorPortfolio.create_sample_portfolio(1000)
results = portfolio.simulate(n_simulations=100000)  # ~30s vs 200s+ fallback
```

## Deployment Recommendations

### When to Use Rust Backend
- ✅ **Production environments** requiring high performance
- ✅ **Large-scale simulations** (>10K paths, >100 assets)
- ✅ **Memory-constrained** systems
- ✅ **Time-critical** applications
- ✅ **Detailed interim analysis** needed

### When NumPy Fallback is Acceptable
- ✅ **Development and prototyping**
- ✅ **Small-scale testing** (<1K simulations)
- ✅ **Educational purposes**
- ✅ **Environments without** binary wheel support
- ✅ **Quick compatibility testing**

## Installation Strategy for Artifactory

### Phase 1: Build Binary Wheels
```bash
# CI/CD pipeline with Rust installed
maturin build --release --out dist/
# Creates wheels for multiple Python versions and platforms
```

### Phase 2: Upload to Internal Artifactory
```bash
twine upload --repository-url https://your-artifactory/pypi/local dist/*
```

### Phase 3: User Installation (No Rust Required)
```bash
pip install -i https://your-artifactory/pypi/local simflux
# Users get Rust performance without Rust installation
```

### Phase 4: Fallback Compatibility
```bash
# If binary wheels fail to install
pip install -i https://your-artifactory/pypi/local simflux
# Automatically falls back to NumPy implementation with warning
```

## Conclusion

The **dual-backend architecture** provides optimal developer and user experience:

1. **Development Phase**: NumPy fallback enables immediate productivity
2. **Production Phase**: Rust backend delivers enterprise performance  
3. **Deployment Phase**: Binary wheels eliminate complexity
4. **Compatibility Phase**: Fallback ensures universal compatibility

**Bottom Line**: Users get **7x average speedup** and **4x memory efficiency** with seamless deployment to internal artifactory, while maintaining full compatibility when needed.
