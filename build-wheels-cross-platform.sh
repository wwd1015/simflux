#!/bin/bash
# Cross-platform wheel build script for SimFlux using Docker and CI
# This provides instructions and local Docker-based building

set -e

echo "=" * 60
echo "SIMFLUX CROSS-PLATFORM WHEEL BUILD"
echo "=" * 60

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo ""

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local-only     Build only for current platform (default)"
    echo "  --docker         Use Docker for Linux builds"
    echo "  --ci             Show CI/CD setup instructions"
    echo "  --install-python Install missing Python versions"
    echo "  --help           Show this help message"
    echo ""
    echo "Cross-platform building requires:"
    echo "  1. GitHub Actions (recommended)"
    echo "  2. Docker for Linux wheels"  
    echo "  3. Multiple OS VMs/machines"
    echo ""
}

install_python_versions() {
    echo "Installing missing Python versions via pyenv..."
    
    # Check if pyenv is installed
    if ! command -v pyenv &> /dev/null; then
        echo "Installing pyenv..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install pyenv
        else
            curl https://pyenv.run | bash
        fi
        
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
        source ~/.bashrc
    fi
    
    # Install Python versions
    PYTHON_VERSIONS="3.8.18 3.9.18 3.10.13 3.11.7 3.12.1"
    for version in $PYTHON_VERSIONS; do
        echo "Installing Python $version..."
        pyenv install -s $version
    done
    
    pyenv global 3.12.1
    echo "✓ Python versions installed"
}

build_with_docker() {
    echo "Building Linux wheels with Docker..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Create Dockerfile for Linux builds
    cat > Dockerfile.wheel-builder << 'EOF'
FROM quay.io/pypa/manylinux2014_x86_64

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin
RUN /opt/python/cp312-cp312/bin/pip install maturin

WORKDIR /io
COPY . .

# Build wheels for all Python versions
RUN for PYBIN in /opt/python/cp3{8,9,10,11,12}*/bin; do \
        echo "Building for $PYBIN" && \
        "$PYBIN/pip" install maturin && \
        maturin build --release --interpreter "$PYBIN/python" --out dist; \
    done

# Audit wheels
RUN for whl in dist/*.whl; do \
        echo "Auditing $whl" && \
        auditwheel repair "$whl" --plat manylinux2014_x86_64 -w dist/; \
    done
EOF

    echo "Building Linux wheels..."
    docker build -f Dockerfile.wheel-builder -t simflux-wheel-builder .
    docker run --rm -v "$(pwd)/dist:/io/dist" simflux-wheel-builder
    
    echo "✓ Linux wheels built with Docker"
    rm Dockerfile.wheel-builder
}

build_local_all_pythons() {
    echo "Building wheels for all available Python versions..."
    
    # Clean previous builds
    rm -rf dist/ target/wheels/
    
    # Use pyenv to build with multiple Python versions
    if command -v pyenv &> /dev/null; then
        PYTHON_VERSIONS=$(pyenv versions --bare | grep -E '^3\.(8|9|10|11|12)')
        echo "Found pyenv Python versions: $PYTHON_VERSIONS"
        
        for version in $PYTHON_VERSIONS; do
            echo "Building wheel for Python $version..."
            pyenv shell $version
            python -m pip install --upgrade pip maturin
            maturin build --release --interpreter python
            echo "✓ Built wheel for Python $version"
        done
        
        pyenv shell --unset
    else
        echo "⚠️  pyenv not found. Using system Python versions only."
        ./build-wheels.sh
    fi
}

show_ci_instructions() {
    echo "CROSS-PLATFORM CI/CD SETUP INSTRUCTIONS"
    echo "========================================"
    echo ""
    echo "1. GitHub Actions (Recommended):"
    echo "   - The .github/workflows/build-wheels.yml file has been created"
    echo "   - Push to GitHub and it will automatically build for:"
    echo "     * Linux (x86_64, aarch64)"  
    echo "     * Windows (x86_64)"
    echo "     * macOS (Intel + Apple Silicon)"
    echo "     * All Python versions 3.8-3.12"
    echo ""
    echo "2. Azure DevOps/Jenkins/GitLab CI:"
    echo "   - Use similar matrix strategy with different runners"
    echo "   - Linux: Use manylinux Docker images"
    echo "   - Windows: Use Windows runners with Rust installed"
    echo "   - macOS: Use macOS runners"
    echo ""
    echo "3. Manual Building on Different Machines:"
    echo "   - Set up VMs or access to different OS machines"
    echo "   - Run this script on each platform"
    echo "   - Collect wheels from each platform's dist/ directory"
    echo ""
    echo "4. Docker-based Local Cross-compilation:"
    echo "   - Use: $0 --docker (for Linux wheels)"
    echo "   - Requires Docker Desktop"
    echo ""
    echo "5. Upload to Artifactory:"
    echo "   - Collect all wheels from different platforms"
    echo "   - Upload: twine upload --repository-url <artifactory-url> dist/*.whl"
}

# Parse arguments
case "$1" in
    --docker)
        build_with_docker
        ;;
    --ci)
        show_ci_instructions
        ;;
    --install-python)
        install_python_versions
        ;;
    --local-all)
        build_local_all_pythons
        ;;
    --help)
        show_help
        ;;
    *)
        echo "Current platform build (same as build-wheels.sh):"
        ./build-wheels.sh
        echo ""
        echo "For cross-platform options, run: $0 --help"
        ;;
esac