# Scripts

Setup and utility scripts for the bctpy_mrtrix project.

## Setup Environment

### Option 1: Using Python (Recommended - Cross-platform)

```bash
python scripts/setup_env.py
```

This works on Windows, macOS, and Linux.

### Option 2: Using Bash (macOS/Linux)

```bash
bash scripts/setup_env.sh
```

## What the setup script does:

1. ✅ Checks for UV package manager (installs if missing)
2. ✅ Creates a Python virtual environment (.venv)
3. ✅ Installs all required dependencies:
   - numpy
   - pandas
   - bctpy
   - scipy
   - statsmodels
   - openpyxl

## After Setup

Activate the environment:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```cmd
.venv\Scripts\activate
```

Or use UV to run scripts without activation:
```bash
uv run python bct_test.py
```

## Running the scripts

From the project root directory:

```bash
# Activate venv first
source .venv/bin/activate

# Run test script
python bct_test.py

# Or use UV directly
uv run python bct_test.py
```
