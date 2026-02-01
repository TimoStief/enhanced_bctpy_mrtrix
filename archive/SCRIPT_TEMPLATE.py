#!/usr/bin/env python3
"""
GENERIC ANALYSIS SCRIPT TEMPLATE
=================================

This template shows the standard structure for all analysis scripts.

KEY PRINCIPLES:
1. CLEAR PURPOSE: What does this script do?
2. INPUT SPECIFICATION: What data does it need?
3. OUTPUT SPECIFICATION: What does it create?
4. CONFIGURATION: What can be customized?
5. GENERIC: Minimal hardcoded paths (all in CONFIG)

HOW TO USE THIS TEMPLATE:
1. Copy this file
2. Edit the docstring with your analysis description
3. Edit the CONFIG section to point to your data
4. Edit the compute_metric() function with your analysis
5. Run it: python your_script.py

"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DOCUMENTATION
# ============================================================================

SCRIPT_NAME = "analysis_template.py"

DESCRIPTION = """
My Analysis Script
==================

WHAT IT DOES:
    Brief description of analysis (2-3 sentences)
    
    Specific outputs:
    - Output 1 description
    - Output 2 description

INPUT REQUIREMENTS:
    DATA:
    - Connectivity matrices (NxN arrays)
    - Format: .mat, .npy, .nii, or other
    - Path template: {data_dir}/{subject}/ses-{session}/file
    
    METADATA:
    - CSV/TSV with columns: [subject, session, group, sex, age, ...]
    - Required columns: subject, session
    - Optional columns: group, sex, age, condition, etc.

OUTPUT FILES:
    - output_metrics.parquet    → subject × session → metrics
    - output_summary.csv        → group-level statistics
    - plots/                    → visualizations
    
    Column definitions:
    - metric1: Description of what this represents
    - metric2: Description of what this represents

USAGE:
    python {script_name}
    
    For background:
    nohup python {script_name} > analysis.log 2>&1 &

VERSION: 2.0 (Generic)
AUTHOR: Your Name / Lab
DATE: YYYY-MM-DD
"""

# ============================================================================
# CONFIGURATION - CUSTOMIZE FOR YOUR DATA
# ============================================================================

CONFIG = {
    # ---- REQUIRED: Data locations ----
    "data_dir": Path("/path/to/connectivity/matrices"),
    "metadata_file": Path("/path/to/participants.tsv"),
    "output_dir": Path("/path/to/outputs"),
    
    # ---- REQUIRED: Data structure ----
    "n_nodes": 246,  # Number of brain regions
    "atlas_name": "Brainnectome",  # Atlas name
    
    # ---- REQUIRED: How to find files ----
    # Variables: {subject}, {session}, {atlas}
    "file_pattern": "{subject}_ses-{session}*/tracks_1000k_streamline/by_atlas/{atlas}/*.connectivity.mat",
    
    # ---- REQUIRED: Metadata column names ----
    "subject_col": "participant_id",
    "session_col": "session",
    "group_col": "group",
    "sex_col": "sex",
    
    # ---- OPTIONAL: Processing options ----
    "binarize": False,
    "threshold": 0.0,
    "compute_parcellations": True,
}

# ============================================================================
# INITIALIZE
# ============================================================================

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

print("="*75)
print(SCRIPT_NAME)
print("="*75)
print(f"Data directory:      {CONFIG['data_dir']}")
print(f"Metadata file:       {CONFIG['metadata_file']}")
print(f"Output directory:    {CONFIG['output_dir']}")
print("="*75)
print()

# ============================================================================
# HELPER FUNCTIONS (Generic)
# ============================================================================

def load_connectivity_matrix(subject: str, session: str) -> np.ndarray:
    """
    Load connectivity matrix from file.
    
    INPUT:
        subject (str): Subject identifier
        session (str): Session identifier
    
    OUTPUT:
        np.ndarray: Connectivity matrix (NxN), or None if not found
    
    HANDLES:
        - Automatic 'sub-' prefix if needed
        - Both .mat and .npy formats
        - Glob pattern matching
        - Shape validation
    """
    if not subject.startswith('sub-'):
        subject = f'sub-{subject}'
    
    pattern = CONFIG["file_pattern"].format(
        subject=subject,
        session=session,
        atlas=CONFIG["atlas_name"]
    )
    
    matches = list(CONFIG["data_dir"].glob(pattern))
    if not matches:
        return None
    
    try:
        filepath = matches[0]
        
        if filepath.suffix == '.mat':
            from scipy.io import loadmat
            mat_data = loadmat(filepath)
            if 'connectivity' in mat_data:
                A = mat_data['connectivity']
            else:
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                A = mat_data[keys[0]] if keys else None
        elif filepath.suffix == '.npy':
            A = np.load(filepath)
        else:
            return None
        
        if A is None:
            return None
        
        A = np.array(A, dtype=float)
        
        if A.shape[0] != CONFIG["n_nodes"] or A.shape[1] != CONFIG["n_nodes"]:
            return None
        
        return A
    
    except Exception as e:
        return None


# ============================================================================
# YOUR ANALYSIS FUNCTION
# ============================================================================

def compute_metrics(A: np.ndarray) -> dict:
    """
    Compute your metrics from connectivity matrix.
    
    INPUT:
        A (np.ndarray): Connectivity matrix (NxN)
    
    OUTPUT:
        dict: Keys should match column names you'll save
        
    NOTES:
        - Handle NaN/Inf gracefully
        - Return np.nan for invalid networks
        - Document what each metric represents
    """
    try:
        # Your analysis here
        metric1 = np.mean(A)
        metric2 = np.std(A)
        
        return {
            'metric1': metric1,
            'metric2': metric2,
        }
    
    except Exception as e:
        return {
            'metric1': np.nan,
            'metric2': np.nan,
        }


# ============================================================================
# MAIN ANALYSIS LOOP
# ============================================================================

print("Loading metadata...")
metadata = pd.read_csv(CONFIG["metadata_file"], sep='\t')
print(f"✓ Loaded {len(metadata)} records\n")

print("Computing metrics...")
results = []

for idx, row in metadata.iterrows():
    subject = row[CONFIG["subject_col"]]
    session = row[CONFIG["session_col"]]
    
    # Load data
    A = load_connectivity_matrix(str(subject), str(session))
    if A is None:
        continue
    
    # Compute metrics
    metrics = compute_metrics(A)
    
    # Build record with metadata
    record = {
        'subject': subject,
        'session': session,
    }
    
    # Add optional columns if they exist
    for col in [CONFIG["group_col"], CONFIG["sex_col"]]:
        if col in row:
            record[col] = row[col]
    
    # Add metrics
    record.update(metrics)
    results.append(record)
    
    if (idx + 1) % 10 == 0:
        print(f"  ✓ Processed {idx + 1} records")

print(f"✓ Computed metrics for {len(results)} subjects\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame(results)

# Parquet (efficient)
parquet_file = CONFIG["output_dir"] / "results.parquet"
results_df.to_parquet(parquet_file, index=False)
print(f"✓ Saved to: {parquet_file}")

# CSV (human-readable)
csv_file = CONFIG["output_dir"] / "results.csv"
results_df.to_csv(csv_file, index=False)
print(f"✓ Saved to: {csv_file}")

# Summary statistics by group
if CONFIG["group_col"] in results_df.columns:
    summary = results_df.groupby(CONFIG["group_col"]).agg(['mean', 'std', 'count'])
    summary_file = CONFIG["output_dir"] / "summary_statistics.csv"
    summary.to_csv(summary_file)
    print(f"✓ Saved to: {summary_file}")

print("\n" + "="*75)
print("ANALYSIS COMPLETE")
print("="*75)
print(f"Records processed: {len(results)}")
print(f"Output directory:  {CONFIG['output_dir']}")
print("="*75)
