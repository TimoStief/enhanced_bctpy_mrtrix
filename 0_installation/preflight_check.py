#!/usr/bin/env python3
"""
Preflight checks for pipeline runs.

- Validates run_spec.json exists and is readable
- Checks required input/output paths
- Verifies required Python packages are installed (static list)

Usage:
  python 0_installation/preflight_check.py /path/to/run_spec.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import os


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "bct",
    "scipy",
    "matplotlib",
    "seaborn",
    "umap",
    "sklearn",
]


def load_spec(spec_path: Path) -> dict:
    if not spec_path.exists():
        raise FileNotFoundError(f"Run spec not found: {spec_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    spec["_spec_dir"] = str(spec_path.parent)
    return spec


def resolve_path(spec_dir: Path, raw_path: str) -> Path:
    raw_path = os.path.expandvars(raw_path)
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (spec_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def check_imports(requirements: list[str]) -> list[str]:
    missing = []
    for pkg in requirements:
        try:
            # Use find_spec instead of import_module to avoid slow JIT compilation
            spec = importlib.util.find_spec(pkg)
            if spec is None:
                missing.append(pkg)
        except Exception:
            missing.append(pkg)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for pipeline execution")
    parser.add_argument("run_spec", help="Path to run_spec.json")
    args = parser.parse_args()

    spec_path = Path(args.run_spec).expanduser().resolve()
    spec = load_spec(spec_path)
    spec_dir = Path(spec["_spec_dir"]).resolve()

    script_path = resolve_path(spec_dir, spec.get("script", ""))
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    inputs = spec.get("inputs", {})
    outputs = spec.get("outputs", {})

    data_dir = resolve_path(spec_dir, inputs.get("data_dir", ""))
    metadata_file = resolve_path(spec_dir, inputs.get("metadata_file", ""))
    output_dir = resolve_path(spec_dir, outputs.get("output_dir", ""))

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    missing = check_imports(REQUIRED_PACKAGES)

    print("✓ Script found:", script_path)
    print("✓ Data directory:", data_dir)
    print("✓ Metadata file:", metadata_file)
    print("✓ Output directory (will be created if needed):", output_dir)

    if missing:
        print("✗ Missing packages:", ", ".join(missing))
        sys.exit(2)

    print("✓ All required packages are installed")


if __name__ == "__main__":
    main()
