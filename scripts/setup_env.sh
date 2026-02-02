#!/bin/bash
# Setup script for bctpy_mrtrix project
# This script creates a virtual environment and installs all dependencies using UV

set -e  # Exit on error

echo "=========================================="
echo "Setting up bctpy_mrtrix environment"
echo "=========================================="

# Get the project root directory (parent of scripts folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"

# Check if UV is installed
echo ""
echo "Checking for UV package manager..."
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "UV installed successfully"
else
    echo "UV is installed: $(uv --version)"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment already exists at .venv"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install numpy pandas bctpy scipy statsmodels openpyxl flask waitress pyarrow h5py

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use UV to run scripts directly:"
echo "  uv run python bct_test.py"
echo ""
