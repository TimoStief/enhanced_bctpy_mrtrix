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

# Add logging utility
from log_analysis import AnalysisLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script metadata
SCRIPT_NAME = "TEMPLATE_analysis.py"
DESCRIPTION = "Brief description of what this analysis does"

# Analysis parameters
PARAMETERS = {
    "param1": 42,
    "param2": "value",
    "alpha": 0.05,
    "method": "pearson",
}

# File paths
BASE_DIR = Path("/data/local/129_PK01/derivatives/bct")
INPUT_FILES = [
    str(BASE_DIR / "node_level_analysis/node_level_metrics.parquet"),
    # Add more input files
]
OUTPUT_DIR = BASE_DIR / "my_analysis_output"
OUTPUT_FILES = [
    str(OUTPUT_DIR / "results.parquet"),
    str(OUTPUT_DIR / "visualization.png"),
]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_data():
    """Load input data"""
    print("Loading data...")
    
    # Example: Load node metrics
    node_metrics = pd.read_parquet(INPUT_FILES[0])
    
    print(f"  Loaded {len(node_metrics)} records")
    return node_metrics


def run_analysis(data):
    """Main analysis logic"""
    print("Running analysis...")
    
    # Example analysis
    result = data.groupby('node').agg({
        'degree': ['mean', 'std'],
        'strength': ['mean', 'std']
    }).reset_index()
    
    print(f"  Computed statistics for {len(result)} nodes")
    return result


def save_results(results):
    """Save analysis outputs"""
    print("Saving results...")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save parquet
    results.to_parquet(OUTPUT_FILES[0], index=False)
    print(f"  Saved: {OUTPUT_FILES[0]}")
    
    # Example: Create visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(results['degree']['mean'], bins=30)
    plt.xlabel('Mean Degree')
    plt.ylabel('Count')
    plt.title('Node Degree Distribution')
    plt.tight_layout()
    plt.savefig(OUTPUT_FILES[1], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_FILES[1]}")


def compute_summary_stats(results):
    """Compute summary statistics for logging"""
    summary = {
        "n_nodes": len(results),
        "mean_degree": float(results['degree']['mean'].mean()),
        "std_degree": float(results['degree']['mean'].std()),
        # Add more summary stats
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
