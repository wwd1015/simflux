# SimFlux Makefile - Unified development workflow
# This provides convenient shortcuts for common development tasks

.PHONY: help install clean build test test-rust lint format check dev wheels ci-build docs all

# Default Python version for development (requires 3.12+)
PYTHON ?= python3.12
VENV_DIR ?= venv

# Build configuration
BUILD_TYPE ?= local
SKIP_TESTS ?= false

help: ## Show this help message
	@echo "SimFlux Development Commands"
	@echo "============================"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment Variables:"
	@echo "  PYTHON      Python executable (default: python3)"
	@echo "  BUILD_TYPE  Build type: local, ci, cross-platform (default: local)"
	@echo "  SKIP_TESTS  Skip tests during build (default: false)"

install: ## Install development dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install maturin
	$(PYTHON) -m pip install -e .[dev]
	@if [ ! -f .pre-commit-config.yaml ]; then echo "Warning: No pre-commit config found"; else pre-commit install; fi

install-venv: ## Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install maturin
	$(VENV_DIR)/bin/pip install -e .[dev]
	@echo "Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"

clean: ## Clean all build artifacts
	rm -rf dist/ build/ target/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	cargo clean 2>/dev/null || true

format: ## Format code (Rust and Python)
	cargo fmt --all
	black python/
	ruff check --fix python/

lint: ## Run all linters
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	black --check python/
	ruff check python/
	mypy python/ --ignore-missing-imports

dev: ## Install in development mode
	maturin develop --release

build: ## Build wheels for current platform
	@BUILD_TYPE=$(BUILD_TYPE) SKIP_TESTS=$(SKIP_TESTS) ./scripts/build-and-test.sh

wheels: ## Build distribution wheels only (no tests)
	@BUILD_TYPE=local SKIP_TESTS=true ./scripts/build-and-test.sh

ci-build: ## Build for CI/CD (all platforms)
	@BUILD_TYPE=ci ./scripts/build-and-test.sh

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

test-rust: ## Run the Rust unit tests (links libpython; needs python on PATH)
	cargo test --no-default-features

test-fast: ## Run tests without benchmarks
	$(PYTHON) -m pytest tests/ -v --benchmark-skip

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/test_integration.py -v

benchmark: ## Run Rust-vs-NumPy performance benchmarks
	$(PYTHON) benchmarks/performance_comparison.py

benchmark-regression: ## Time canonical seeded workloads (JSON out; --compare to diff versions)
	$(PYTHON) benchmarks/regression_benchmark.py

benchmark-quick: ## Run quick benchmark smoke test
	$(PYTHON) -c "\
import simflux as sf; \
import time; \
start = time.time(); \
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100); \
paths = gbm.simulate(n_paths=1000, n_steps=252, T=1.0); \
duration = time.time() - start; \
print(f'Benchmark: {paths.shape} in {duration:.3f}s ({(paths.shape[0]*paths.shape[1])/duration:,.0f} ops/sec)')"

check: format lint test ## Run all checks (format, lint, test)

docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"
	@echo "TODO: Add Sphinx documentation build"

security: ## Run security checks
	bandit -r python/ --skip B101,B601

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

release-check: ## Check if ready for release
	@echo "Checking release readiness..."
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory build
	@echo "✓ Release checks passed"

# Development workflow targets
all: clean format lint build test ## Full development build (clean -> format -> lint -> build -> test)

quick: dev test-fast ## Quick development cycle (dev install + fast tests)

# CI targets
ci: lint test build ## CI pipeline (lint -> test -> build)

# Utility targets
rust-version: ## Show Rust version info
	@echo "Rust toolchain:"
	@rustc --version
	@cargo --version

python-version: ## Show Python version info
	@echo "Python version:"
	@$(PYTHON) --version
	@$(PYTHON) -c "import sys; print(f'Location: {sys.executable}')"

deps-update: ## Update dependencies
	cargo update
	$(PYTHON) -m pip list --outdated

env-info: rust-version python-version ## Show environment information
	@echo ""
	@echo "Build environment:"
	@echo "  OS: $$(uname -s)"
	@echo "  Architecture: $$(uname -m)"
	@echo "  Working directory: $$(pwd)"
	@echo "  Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"

# Example usage targets
example-gbm: ## Run GBM example
	$(PYTHON) -c "\
import simflux as sf; \
import numpy as np; \
print('Running GBM example...'); \
gbm = sf.GBM(mu=0.05, sigma=0.2, S0=100); \
paths = gbm.simulate(n_paths=5, n_steps=10, T=1.0); \
print(f'Generated paths shape: {paths.shape}'); \
print(f'Final prices: {paths[:, -1]}');"

example-portfolio: ## Run portfolio example
	$(PYTHON) -c "\
import simflux as sf; \
import numpy as np; \
print('Running portfolio example...'); \
portfolio = sf.CreditPortfolio.create_sample_portfolio([10, 10], ['Tech', 'Finance']); \
results = portfolio.simulate(n_simulations=100); \
stats = results['portfolio_statistics']; \
print(f'Portfolio VaR 95%: {stats[\"var_95\"]:,.0f}'); \
print(f'Portfolio VaR 99%: {stats[\"var_99\"]:,.0f}');"