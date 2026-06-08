# SimFlux Deployment Guide for Internal Artifactory

## Overview

This guide shows how to build and deploy SimFlux binary wheels to your internal Artifactory, ensuring users get high-performance Rust backend without requiring Rust installation.

## Architecture Benefits

✅ **No Rust dependency** for end users  
✅ **Automatic fallback** to NumPy if binary wheels unavailable  
✅ **Cross-platform support** (Linux, macOS, Windows)  
✅ **Multiple Python versions** (3.8-3.12)  
✅ **~2.6–45x faster** with the Rust backend (10–45x for portfolios)  

## Build Process

### Option 1: GitHub Actions (Recommended)

Create `.github/workflows/build-wheels.yml`:

```yaml
name: Build and Publish Wheels

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build-wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build wheels
      uses: PyO3/maturin-action@v1
      with:
        target: ${{ matrix.target }}
        args: --release --out dist --interpreter 3.8 3.9 3.10 3.11 3.12
        sccache: 'true'
        manylinux: auto
        
    - name: Upload wheels
      uses: actions/upload-artifact@v3
      with:
        name: wheels-${{ matrix.os }}
        path: dist

  publish:
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
    - name: Download all wheels
      uses: actions/download-artifact@v3
      
    - name: Publish to Artifactory
      run: |
        pip install twine
        twine upload --repository-url ${{ secrets.ARTIFACTORY_URL }}/pypi/local dist/*
      env:
        TWINE_USERNAME: ${{ secrets.ARTIFACTORY_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.ARTIFACTORY_TOKEN }}
```

### Option 2: Manual Build Script

```bash
#!/bin/bash
# build-wheels.sh

set -e

echo "Building SimFlux wheels for multiple platforms..."

# Clean previous builds
rm -rf dist/ build/ target/

# Install maturin
pip install maturin

# Build for current platform, multiple Python versions
for python_ver in 3.8 3.9 3.10 3.11 3.12; do
    if command -v python${python_ver} &> /dev/null; then
        echo "Building for Python ${python_ver}..."
        maturin build --release --interpreter python${python_ver}
    fi
done

echo "Build complete. Wheels in dist/:"
ls -la dist/

# Upload to artifactory
echo "Uploading to internal artifactory..."
pip install twine
twine upload --repository-url https://your-artifactory.com/pypi/local dist/*
```

### Option 3: Docker-based Cross-compilation

```dockerfile
# Dockerfile.build-wheels
FROM quay.io/pypa/manylinux2014_x86_64

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python versions and maturin
RUN for PYVER in cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312; do \
    /opt/python/$PYVER/bin/pip install maturin; \
done

WORKDIR /workspace
COPY . .

# Build wheels for all Python versions
RUN for PYVER in cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311 cp312-cp312; do \
    /opt/python/$PYVER/bin/maturin build --release --interpreter /opt/python/$PYVER/bin/python; \
done

# Repair wheels for broad compatibility
RUN for wheel in dist/*.whl; do \
    auditwheel repair "$wheel" -w dist/; \
done

CMD ["ls", "-la", "dist/"]
```

Build with Docker:
```bash
docker build -f Dockerfile.build-wheels -t simflux-builder .
docker run -v $(pwd)/dist:/workspace/dist simflux-builder
```

## Artifactory Configuration

### Repository Setup

1. **Create PyPI Repository**:
   - Repository Type: PyPI
   - Repository Key: `pypi-local`
   - Layout: `pypi-default`

2. **Set Up Virtual Repository** (optional):
   - Combines your local repo with PyPI remote
   - Allows fallback to public PyPI for other packages

### Upload Configuration

Configure `~/.pypirc`:
```ini
[distutils]
index-servers = artifactory

[artifactory]
repository = https://your-artifactory.com/pypi/local
username = your-username  
password = your-token
```

## User Installation

### Method 1: Direct Installation
```bash
pip install -i https://your-artifactory.com/pypi/local simflux
```

### Method 2: pip.conf Configuration
Create `~/.pip/pip.conf`:
```ini
[global]
extra-index-url = https://your-artifactory.com/pypi/local
trusted-host = your-artifactory.com
```

Then users can simply:
```bash
pip install simflux
```

### Method 3: Requirements File
```txt
# requirements.txt
--extra-index-url https://your-artifactory.com/pypi/local
simflux>=0.1.0
numpy>=1.21.0
pandas>=1.3.0
```

## Deployment Validation

### Test Script for Users

```python
# test_installation.py
import sys
import warnings

def test_simflux_installation():
    """Test SimFlux installation and detect backend."""
    
    print("Testing SimFlux installation...")
    print(f"Python version: {sys.version}")
    
    try:
        import simflux as sf
        print("✅ SimFlux imported successfully")
        
        # Detect backend
        try:
            from simflux.core.backend import Backend
            backend = "Rust" if Backend.is_available() else "NumPy Fallback"
            print(f"Backend detected: {backend}")

            if Backend.is_available():
                print("🚀 High-performance Rust backend available")
            else:
                print("⚠️  Using NumPy fallback (slower but functional)")
                
        except Exception as e:
            print(f"Backend detection failed: {e}")
        
        # Quick functionality test
        print("\nTesting basic functionality...")
        gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
        paths = gbm.simulate(n_paths=100, n_steps=10)
        print(f"✅ GBM simulation working: {paths.shape}")
        
        # Test portfolio
        portfolio = sf.CreditPortfolio.create_sample_portfolio(10)
        results = portfolio.simulate(n_simulations=100)
        print(f"✅ Portfolio simulation working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import SimFlux: {e}")
        return False
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_simflux_installation()
    sys.exit(0 if success else 1)
```

## Performance Verification

### Benchmark Script for Validation

```python
# validate_performance.py
import time
import simflux as sf

def validate_performance():
    """Quick performance validation."""
    
    print("Running performance validation...")
    
    # Small benchmark
    gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
    
    start_time = time.time()
    paths = gbm.simulate(n_paths=1000, n_steps=252)
    execution_time = time.time() - start_time
    
    ops_per_sec = (1000 * 252) / execution_time
    
    print(f"Execution time: {execution_time:.3f}s")
    print(f"Throughput: {ops_per_sec:,.0f} ops/sec")
    
    # Performance expectations
    try:
        from simflux.core.engine import RUST_AVAILABLE
        if RUST_AVAILABLE:
            expected_min = 1_000_000  # 1M ops/sec minimum for Rust
            if ops_per_sec >= expected_min:
                print("✅ Rust performance validated")
            else:
                print(f"⚠️  Rust performance below expected ({ops_per_sec:,.0f} < {expected_min:,})")
        else:
            expected_min = 100_000  # 100K ops/sec minimum for NumPy
            if ops_per_sec >= expected_min:
                print("✅ NumPy fallback performance acceptable")
            else:
                print(f"⚠️  NumPy performance below expected ({ops_per_sec:,.0f} < {expected_min:,})")
                
    except:
        print("Could not validate performance expectations")

if __name__ == "__main__":
    validate_performance()
```

## Troubleshooting

### Common Issues

#### Issue 1: Binary Wheel Not Found
```
ERROR: Could not find a version that satisfies the requirement simflux
```

**Solution**: Check wheel availability for your platform:
```bash
pip index versions -i https://your-artifactory.com/pypi/local simflux
```

#### Issue 2: Falls Back to Source Build
```
Building wheel for simflux (pyproject.toml) ... 
```

**Solution**: Ensure binary wheels are built for your platform/Python version.

#### Issue 3: NumPy Fallback Performance
```python
UserWarning: Rust backend not available. Using slower NumPy fallback.
```

**Solution**: This is expected when binary wheels aren't available. Performance will be slower but functional.

### Debugging Commands

```bash
# Check available wheels
pip index versions -i https://your-artifactory.com/pypi/local simflux

# Force reinstall 
pip install --force-reinstall -i https://your-artifactory.com/pypi/local simflux

# Install with verbose output
pip install -v -i https://your-artifactory.com/pypi/local simflux

# Check installed version and files
pip show -f simflux
```

## Maintenance

### Regular Tasks

1. **Update wheels** when new SimFlux versions released
2. **Monitor storage usage** in Artifactory  
3. **Test installation** across different environments
4. **Update documentation** with new features

### Version Management

- Use semantic versioning (e.g., 0.1.0, 0.1.1, 0.2.0)
- Tag releases to trigger automated builds
- Maintain multiple versions for compatibility

## Security Considerations

1. **Secure tokens** for Artifactory authentication
2. **Network security** for internal repository access
3. **Code signing** for wheel integrity (optional)
4. **Dependency scanning** for vulnerabilities

## Summary

This deployment strategy provides:

✅ **Seamless user experience** - no compilation required  
✅ **High performance** - ~2.6–45x speedup with the Rust backend  
✅ **Compatibility** - automatic NumPy fallback  
✅ **Enterprise ready** - internal artifactory deployment  
✅ **Cross-platform** - Windows, macOS, Linux support  

Users get production-grade performance with development-grade ease of installation.