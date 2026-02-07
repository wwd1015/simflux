#!/bin/bash
# Unified build and test script for SimFlux
# This replaces the separate build-wheels.sh and build-wheels-cross-platform.sh

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_TYPE="${BUILD_TYPE:-local}"  # local, ci, cross-platform
PYTHON_VERSIONS="${PYTHON_VERSIONS:-3.8,3.9,3.10,3.11,3.12}"
SKIP_TESTS="${SKIP_TESTS:-false}"

cd "$PROJECT_DIR"

echo "=================================================================================="
echo "SIMFLUX UNIFIED BUILD SCRIPT"
echo "=================================================================================="
echo "Build type: $BUILD_TYPE"
echo "Python versions: $PYTHON_VERSIONS"
echo "Skip tests: $SKIP_TESTS"
echo "Working directory: $(pwd)"
echo ""

# Function to log with timestamp
log() {
    echo "$(date '+%H:%M:%S'): $1"
}

# Function to check command availability
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log "❌ Required command '$1' not found"
        return 1
    fi
    log "✓ Found $1: $($1 --version 2>/dev/null | head -1 || echo 'version unknown')"
}

# Function to setup environment
setup_environment() {
    log "Setting up build environment..."

    # Check required tools
    check_command rustc || exit 1
    check_command python3 || exit 1

    # Install/check maturin
    if ! command -v maturin &> /dev/null; then
        log "Installing maturin..."
        pip3 install maturin
    else
        log "✓ Found maturin: $(maturin --version)"
    fi

    # Install development dependencies if not in CI
    if [[ "$BUILD_TYPE" != "ci" ]]; then
        log "Installing development dependencies..."
        pip3 install pytest pytest-benchmark black ruff mypy pre-commit
    fi
}

# Function to clean build artifacts
clean_build() {
    log "Cleaning previous build artifacts..."
    rm -rf dist/ build/ target/wheels/ target/debug target/release *.egg-info/
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    log "✓ Build artifacts cleaned"
}

# Function to run linting
run_linting() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "Skipping linting (SKIP_TESTS=true)"
        return 0
    fi

    log "Running code linting..."

    # Rust linting
    log "  Running cargo fmt check..."
    cargo fmt --all -- --check

    log "  Running cargo clippy..."
    cargo clippy --all-targets --all-features -- -D warnings

    # Python linting
    log "  Running black check..."
    black --check python/ || {
        log "❌ Black formatting check failed. Run 'black python/' to fix."
        return 1
    }

    log "  Running ruff..."
    ruff check python/

    log "  Running mypy..."
    mypy python/ --ignore-missing-imports || log "⚠️ Type check warnings found"

    log "✓ Linting completed"
}

# Function to get Python interpreters
get_python_interpreters() {
    local interpreters=()

    IFS=',' read -ra versions <<< "$PYTHON_VERSIONS"
    for version in "${versions[@]}"; do
        version=$(echo "$version" | tr -d ' ')  # Remove spaces

        # Try python3.x format first
        if command -v "python${version}" &> /dev/null; then
            interpreters+=("python${version}")
            log "✓ Found Python ${version}: $(python${version} --version)"
        # Try pythonX.Y format
        elif command -v "python${version}" &> /dev/null; then
            interpreters+=("python${version}")
            log "✓ Found Python ${version}: $(python${version} --version)"
        else
            log "⚠️ Python ${version} not found"
        fi
    done

    # Fallback to python3 if no specific versions found
    if [[ ${#interpreters[@]} -eq 0 ]] && command -v python3 &> /dev/null; then
        interpreters+=("python3")
        log "✓ Fallback to python3: $(python3 --version)"
    fi

    if [[ ${#interpreters[@]} -eq 0 ]]; then
        log "❌ No Python interpreters found!"
        exit 1
    fi

    printf '%s\n' "${interpreters[@]}"
}

# Function to build wheels
build_wheels() {
    log "Building wheels..."

    local interpreters
    mapfile -t interpreters < <(get_python_interpreters)

    local build_args="--release --out dist"

    # Add interpreter arguments
    for interpreter in "${interpreters[@]}"; do
        build_args="$build_args --interpreter $interpreter"
    done

    case "$BUILD_TYPE" in
        "cross-platform")
            log "Cross-platform build mode"
            # Use cibuildwheel for cross-platform builds
            if command -v cibuildwheel &> /dev/null; then
                log "Using cibuildwheel for cross-platform build..."
                cibuildwheel --output-dir dist
            else
                log "⚠️ cibuildwheel not found, falling back to maturin"
                maturin build $build_args
            fi
            ;;
        "ci")
            log "CI build mode"
            maturin build $build_args
            ;;
        *)
            log "Local build mode"
            maturin develop --release  # Also install in development mode
            maturin build $build_args
            ;;
    esac

    # Verify wheels were created
    if [[ -d "dist" ]] && [[ -n "$(ls -A dist/ 2>/dev/null)" ]]; then
        log "✓ Wheels built successfully:"
        ls -la dist/

        # Show wheel information
        for wheel in dist/*.whl; do
            if [[ -f "$wheel" ]]; then
                local size=$(du -h "$wheel" | cut -f1)
                log "  $(basename "$wheel") (${size})"
            fi
        done
    else
        log "❌ No wheels were built!"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "Skipping tests (SKIP_TESTS=true)"
        return 0
    fi

    log "Running tests..."

    # Install the package in development mode if not already done
    if [[ "$BUILD_TYPE" == "local" ]] && ! python3 -c "import simflux" 2>/dev/null; then
        log "Installing package in development mode..."
        maturin develop --release
    fi

    # Run Python tests
    log "  Running pytest..."
    python3 -m pytest tests/ -v --tb=short

    # Run benchmarks (quick version)
    log "  Running benchmark smoke test..."
    python3 -c "
import simflux as sf
import numpy as np
import time

# Quick benchmark to ensure everything works
start = time.time()
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=100, n_steps=50, T=1.0)
duration = time.time() - start

print(f'Basic simulation test: {paths.shape} in {duration:.3f}s')
assert paths.shape == (100, 51)
assert np.all(paths > 0)
print('✓ Basic functionality test passed')
"

    log "✓ Tests completed"
}

# Function to install wheels and test
test_wheels() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "Skipping wheel testing (SKIP_TESTS=true)"
        return 0
    fi

    log "Testing built wheels..."

    # Create a temporary virtual environment
    local temp_venv=$(mktemp -d)
    python3 -m venv "$temp_venv"
    source "$temp_venv/bin/activate"

    # Install dependencies
    pip install pytest numpy pandas

    # Test each wheel
    for wheel in dist/*.whl; do
        if [[ -f "$wheel" ]]; then
            log "  Testing $(basename "$wheel")..."

            # Install the wheel
            pip install "$wheel" --force-reinstall

            # Run a quick test
            python -c "
import simflux as sf
import numpy as np

# Test basic functionality
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100)
paths = gbm.simulate(n_paths=10, n_steps=5, T=1.0)
assert paths.shape == (10, 6)
print('✓ Wheel test passed')
"

            pip uninstall simflux -y
        fi
    done

    # Cleanup
    deactivate
    rm -rf "$temp_venv"

    log "✓ Wheel testing completed"
}

# Function to generate build summary
generate_summary() {
    log ""
    log "=================================================================================="
    log "BUILD SUMMARY"
    log "=================================================================================="
    log "Build type: $BUILD_TYPE"
    log "Build directory: $(pwd)/dist"

    if [[ -d "dist" ]] && [[ -n "$(ls -A dist/ 2>/dev/null)" ]]; then
        log ""
        log "Built artifacts:"
        ls -la dist/

        local total_size=$(du -sh dist/ 2>/dev/null | cut -f1 || echo "unknown")
        log ""
        log "Total build size: $total_size"

        log ""
        log "Next steps:"
        log "  • Test locally: pip install dist/*.whl"
        log "  • Upload to registry: twine upload dist/*"
        log "  • Install in development: maturin develop --release"
    else
        log ""
        log "❌ BUILD FAILED - No artifacts created"
        exit 1
    fi

    log ""
    log "🎉 BUILD COMPLETED SUCCESSFULLY"
    log ""
}

# Main execution flow
main() {
    setup_environment
    clean_build
    run_linting
    build_wheels
    run_tests
    test_wheels
    generate_summary
}

# Run main function
main "$@"