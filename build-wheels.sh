#!/bin/bash
# Manual build script for SimFlux wheels

set -e

echo "="*60
echo "SIMFLUX MANUAL WHEEL BUILD"
echo "="*60

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo "Rust version: $(rustc --version)"
echo "Python version: $(python3 --version)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf dist/ build/ target/
echo "✓ Cleaned build directories"

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip3 install maturin
    echo "✓ Maturin installed"
else
    echo "✓ Maturin already installed: $(maturin --version)"
fi

echo ""
echo "Building wheels..."

# Find available Python versions
PYTHON_VERSIONS=()
for version in 3.12 3.13; do
    if command -v python${version} &> /dev/null; then
        PYTHON_VERSIONS+=($version)
        echo "✓ Found Python ${version}: $(python${version} --version)"
    elif command -v python3 &> /dev/null && python3 --version 2>&1 | grep -q "Python ${version}"; then
        PYTHON_VERSIONS+=(3)
        echo "✓ Found Python 3 (${version}): $(python3 --version)"
        break
    fi
done

# If no specific versions found, use python3
if [ ${#PYTHON_VERSIONS[@]} -eq 0 ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_VERSIONS+=(3)
        echo "✓ Using python3: $(python3 --version)"
    else
        echo "❌ No suitable Python installation found"
        exit 1
    fi
fi

echo ""
echo "Building wheels for Python versions: ${PYTHON_VERSIONS[@]}"
echo ""

# Build wheels for each Python version
for py_ver in "${PYTHON_VERSIONS[@]}"; do
    echo "Building wheel for Python ${py_ver}..."
    
    if [ "$py_ver" = "3" ]; then
        python_cmd="python3"
    else
        python_cmd="python${py_ver}"
    fi
    
    if command -v $python_cmd &> /dev/null; then
        echo "  Using interpreter: $($python_cmd --version)"
        
        # Build the wheel
        maturin build --release --interpreter $python_cmd 2>&1 | while read line; do
            echo "    $line"
        done
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully built wheel for Python ${py_ver}"
        else
            echo "  ❌ Failed to build wheel for Python ${py_ver}"
        fi
    else
        echo "  ⚠️ Python ${py_ver} not found, skipping..."
    fi
    echo ""
done

# Check what was built
echo "Build results:"
if [ -d "dist" ] && [ "$(ls -A dist)" ]; then
    echo "✓ Wheels successfully built:"
    ls -la dist/
    
    echo ""
    echo "Wheel details:"
    for wheel in dist/*.whl; do
        if [ -f "$wheel" ]; then
            echo "  $(basename $wheel)"
            echo "    Size: $(du -h $wheel | cut -f1)"
            
            # Try to get wheel info
            if command -v wheel &> /dev/null; then
                wheel unpack "$wheel" --dest /tmp/wheel_check &> /dev/null && \
                echo "    Structure: OK" || echo "    Structure: Could not verify"
            fi
        fi
    done
    
    echo ""
    echo "🎉 BUILD SUCCESSFUL!"
    echo ""
    echo "Next steps:"
    echo "1. Test the wheels: pip install dist/*.whl"
    echo "2. Upload to artifactory: twine upload --repository-url <your-repo> dist/*"
    echo ""
    
else
    echo "❌ No wheels were built successfully"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check that Rust and Python are properly installed"
    echo "2. Verify pyproject.toml configuration"
    echo "3. Check build logs above for specific errors"
    exit 1
fi