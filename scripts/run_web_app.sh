#!/bin/bash
# Launch the BCT Analysis Web Interface

set -e

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "🧠 BCT Analysis Web Interface"
echo "=========================================="

# Check if venv exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "📝 Please run: bash scripts/setup_env.sh"
    exit 1
fi

PYTHON="$PROJECT_ROOT/.venv/bin/python"
PIP="$PROJECT_ROOT/.venv/bin/pip"

echo "Checking dependencies..."

# Ensure pip exists inside the venv (uv venv may omit pip)
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "• pip not found in venv; installing via ensurepip..."
    "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi

# Install Flask/Waitress using pip if available, otherwise fallback to uv
if "$PYTHON" -m pip --version >/dev/null 2>&1; then
    "$PYTHON" -m pip install --quiet flask waitress
else
    echo "• pip still unavailable; using uv to install dependencies..."
    cd "$PROJECT_ROOT"
    uv pip install flask waitress
fi

echo "✓ Starting web interface..."
echo ""

# Run the Flask app with explicit interpreter
cd "$PROJECT_ROOT/web_app"
exec "$PYTHON" app.py
