#!/usr/bin/env python3
"""
TEMPLATE: Analysis Script with Logging

This template demonstrates how to structure analysis scripts with
automatic logging for reproducibility.

Copy this template and modify for your specific analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import runpy
import json

# Add logging utility
from log_analysis import AnalysisLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script metadata
SCRIPT_NAME = "TEMPLATE_analysis.py"
DESCRIPTION = "Wrapper to run a specific analysis script with logging"

# Analysis parameters
PARAMETERS = {
    "script_to_run": "01_global_basic_metrics.py",
    "run_spec": "analysis_scripts_restructured/1_utilities/run_spec.json",
}

# File paths
BASE_DIR = Path("/data/local/129_PK01/derivatives/bct")

DEFAULT_RUN_SPEC = Path(__file__).resolve().parents[0] / "run_spec.json"

INPUT_FILES = []
OUTPUT_DIR = BASE_DIR / "global_metrics"
OUTPUT_FILES = []

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def _resolve_spec_path() -> Path:
    """Resolve run spec path from argv or defaults."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    return DEFAULT_RUN_SPEC.resolve()


def load_run_spec() -> dict:
    """Load and validate the run specification JSON."""
    spec_path = _resolve_spec_path()
    if not spec_path.exists():
        raise FileNotFoundError(f"Run spec not found: {spec_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Basic validation
    for key in ["script", "inputs", "outputs"]:
        if key not in spec:
            raise ValueError(f"Missing required key in run spec: {key}")

    return spec


def validate_run_spec(spec: dict) -> dict:
    """Validate inputs/outputs and return resolved paths."""
    resolved = {
        "script": Path(spec["script"]).expanduser().resolve(),
        "inputs": spec.get("inputs", {}),
        "outputs": spec.get("outputs", {}),
    }

    if not resolved["script"].exists():
        raise FileNotFoundError(f"Script not found: {resolved['script']}")

    inputs = resolved["inputs"]
    required_inputs = ["data_dir", "metadata_file", "atlas_name", "n_nodes", "file_pattern"]
    for key in required_inputs:
        if key not in inputs:
            raise ValueError(f"Missing required input: {key}")

    data_dir = Path(inputs["data_dir"]).expanduser().resolve()
    metadata_file = Path(inputs["metadata_file"]).expanduser().resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    outputs = resolved["outputs"]
    if "output_dir" not in outputs:
        raise ValueError("Missing required outputs.output_dir")
    output_dir = Path(outputs["output_dir"]).expanduser().resolve()

    resolved["inputs"]["data_dir"] = str(data_dir)
    resolved["inputs"]["metadata_file"] = str(metadata_file)
    resolved["outputs"]["output_dir"] = str(output_dir)

    return resolved


def load_data():
    """Load and validate run spec"""
    print("Loading run spec...")
    spec = load_run_spec()
    resolved = validate_run_spec(spec)
    print("  ✓ Run spec validated")
    return resolved


def _apply_config_overrides(spec: dict) -> dict:
    """Prepare globals for the target script with CONFIG overrides."""
    inputs = spec.get("inputs", {})
    outputs = spec.get("outputs", {})

    config_overrides = {
        "data_dir": Path(inputs["data_dir"]),
        "metadata_file": Path(inputs["metadata_file"]),
        "output_dir": Path(outputs["output_dir"]),
        "n_nodes": inputs.get("n_nodes"),
        "atlas_name": inputs.get("atlas_name"),
        "file_pattern": inputs.get("file_pattern"),
        "subject_col": inputs.get("subject_col", "participant_id"),
        "session_col": inputs.get("session_col", "session"),
        "group_col": inputs.get("group_col", "group"),
        "sex_col": inputs.get("sex_col", "sex"),
        "binarize": inputs.get("binarize", False),
        "umap_n_neighbors": inputs.get("umap_n_neighbors", 15),
        "umap_min_dist": inputs.get("umap_min_dist", 0.1),
        "umap_metric": inputs.get("umap_metric", "euclidean"),
        "include_metrics": inputs.get("include_metrics", "all"),
        "exclude_metrics": inputs.get("exclude_metrics", []),
    }

    return config_overrides


def run_analysis(data):
    """Run the target analysis script with validated inputs"""
    print("Running target analysis script...")

    script_path = Path(data["script"]).resolve()
    config_overrides = _apply_config_overrides(data)

    # Execute the script in its own global namespace with CONFIG injected
    init_globals = {
        "CONFIG": config_overrides,
    }
    runpy.run_path(str(script_path), run_name="__main__", init_globals=init_globals)
    print("  Target script execution completed")
    return data


def save_results(results):
    """Validate expected outputs if provided"""
    print("Validating expected outputs...")

    outputs = results.get("outputs", {})
    output_dir = Path(outputs.get("output_dir", OUTPUT_DIR)).expanduser().resolve()
    expected = outputs.get("expected_files", [])

    if not expected:
        print("  No expected outputs specified.")
        return

    missing = []
    for rel_path in expected:
        candidate = output_dir / rel_path
        if not candidate.exists():
            missing.append(str(candidate))

    if missing:
        raise FileNotFoundError("Missing expected outputs:\n" + "\n".join(missing))

    print("  ✓ All expected outputs present")


def compute_summary_stats(results):
    """Compute summary statistics for logging"""
    summary = {
        "script_executed": str(results.get("script")),
        "output_dir": str(results.get("outputs", {}).get("output_dir", OUTPUT_DIR)),
    }
    return summary


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline with logging"""
    
    # Initialize logger
    logger = AnalysisLogger()
    
    try:
        # Start logging
        analysis_id = logger.start_analysis(
            script_name=SCRIPT_NAME,
            description=DESCRIPTION,
            parameters=PARAMETERS,
            inputs=INPUT_FILES,
            outputs=OUTPUT_FILES,
            notes="Initial run of template analysis"
        )
        
        # Run analysis pipeline
        data = load_data()
        results = run_analysis(data)
        save_results(results)
        
        # Compute summary statistics
        summary = compute_summary_stats(results)
        
        # Finish logging (success)
        logger.finish_analysis(
            success=True,
            results_summary=summary
        )
        
        print("\n✓ Analysis completed successfully!")
        print(f"Analysis ID: {analysis_id}")
        
    except Exception as e:
        # Log error
        logger.finish_analysis(
            success=False,
            error_message=str(e)
        )
        
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
