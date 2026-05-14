#!/usr/bin/env python3
"""
SCRIPT: Node-Level Brain Network Analysis
==========================================

PURPOSE:
    Compute node-level network metrics from connectivity matrices.
    Input/output paths are provided via CLI flags or CLI arguments.
    All other parameters (n_nodes, atlas, file_format, columns) are
    automatically detected from the data.

USAGE:
    python 02_nodal_metrics.py CLI flags
    python 02_nodal_metrics.py --data-dir /path/to/data --metadata /path/to/meta.tsv --output-dir /path/to/out

AUTHOR: Analysis Pipeline
VERSION: 2.0 (CLI-driven, auto-detection)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import bct
from scipy.io import loadmat

# Hub classification thresholds (Guimerà & Amaral 2005)
PROVINCIAL_HUB_THRESHOLD = 2.5
CONNECTOR_HUB_THRESHOLD  = 0.30
KINLESS_HUB_THRESHOLD    = 0.05


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node-level hub identification — all inputs via CLI flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir",    required=True, help="Directory with connectivity matrices")
    parser.add_argument("--metadata",    required=True, help="Participant metadata file (CSV or TSV)")
    parser.add_argument("--output-dir",  required=True, help="Directory where results are saved")
    parser.add_argument("--binarize",    action="store_true", default=False,
                        help="Binarize connectivity matrices before analysis")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Validate paths and return config dict."""
    data_dir      = Path(args.data_dir).expanduser().resolve()
    metadata_file = Path(args.metadata).expanduser().resolve()
    output_dir    = Path(args.output_dir).expanduser().resolve()
    if not data_dir.exists():
        sys.exit(f"x Data directory not found: {data_dir}")
    if not metadata_file.exists():
        sys.exit(f"x Metadata file not found: {metadata_file}")
    return {
        "data_dir":      data_dir,
        "metadata_file": metadata_file,
        "output_dir":    output_dir,
        "binarize":      args.binarize,
    }


# ============================================================================
# AUTO-DETECTION  (same helpers as global script)
# ============================================================================

def detect_metadata_columns(metadata: pd.DataFrame) -> dict:
    """Fuzzy-match column names for subject, session, group, sex."""
    cols = {c.lower(): c for c in metadata.columns}

    def find(candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        for c in candidates:
            for col_lower, col_orig in cols.items():
                if c in col_lower:
                    return col_orig
        return None

    detected = {
        "subject_col": find(["participant_id", "subject_id", "subject", "participant", "id"]),
        "session_col": find(["session", "ses", "timepoint", "visit"]),
        "group_col":   find(["group", "condition", "arm", "intervention"]),
        "sex_col":     find(["sex", "gender"]),
    }

    print("  Auto-detected metadata columns:")
    for k, v in detected.items():
        print(f"    {k}: {'✓ ' + v if v else '⚠ not found'}")

    missing = [k for k, v in detected.items() if v is None and k in ("subject_col", "session_col")]
    if missing:
        sys.exit(f"✗ Could not detect required columns: {missing}\n"
                 f"  Available columns: {list(metadata.columns)}")
    return detected


def detect_file_format(data_dir: Path) -> str:
    """Scan data_dir for matrix files. Returns 'npy' or 'mat'."""
    npy_files = list(data_dir.rglob("*.npy"))
    mat_files = list(data_dir.rglob("*.connectivity.mat")) or list(data_dir.rglob("*.mat"))

    if not npy_files and not mat_files:
        sys.exit(f"✗ No .npy or .mat files found in: {data_dir}")

    if len(npy_files) >= len(mat_files):
        print(f"  ✓ File format detected: .npy ({len(npy_files)} files)")
        return "npy"
    print(f"  ✓ File format detected: .mat ({len(mat_files)} files)")
    return "mat"


def detect_n_nodes(data_dir: Path, fmt: str) -> int:
    """Load first matrix found and return its node count."""
    files = list(data_dir.rglob("*.npy" if fmt == "npy" else "*.mat"))
    if not files:
        sys.exit(f"✗ No {fmt} files found in: {data_dir}")
    try:
        A = _load_matrix_raw(files[0], fmt)
        if A is not None:
            print(f"  ✓ n_nodes detected: {A.shape[0]}")
            return A.shape[0]
    except Exception as e:
        sys.exit(f"✗ Could not load matrix for node detection: {e}")
    sys.exit("✗ Could not detect n_nodes from data")


def detect_atlas_name(data_dir: Path) -> str:
    """Try to extract atlas name from folder structure or filenames."""
    known      = ["brainnectome", "brodmann", "aal", "schaefer", "destrieux", "desikan", "hcp"]
    search_str = str(data_dir).lower()
    for name in known:
        if name in search_str:
            print(f"  ✓ Atlas detected: {name.capitalize()} (from path)")
            return name.capitalize()
    for f in data_dir.rglob("*.*"):
        fname = f.name.lower()
        for name in known:
            if name in fname:
                print(f"  ✓ Atlas detected: {name.capitalize()} (from filename)")
                return name.capitalize()
    print("  ⚠ Atlas name not detected, using 'Unknown'")
    return "Unknown"


def detect_sessions(metadata: pd.DataFrame, session_col: str) -> list:
    sessions = sorted(metadata[session_col].dropna().unique().tolist())
    print(f"  ✓ Sessions detected: {sessions}")
    return sessions


# ============================================================================
# MATRIX LOADING
# ============================================================================

def _load_matrix_raw(filepath: Path, fmt: str) -> np.ndarray | None:
    try:
        if fmt == "npy":
            A = np.load(filepath)
        else:
            mat  = loadmat(filepath)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            A    = mat.get("connectivity", mat[keys[0]] if keys else None)
            if A is None:
                return None
        return np.asarray(A, dtype=float)
    except Exception:
        return None


def find_matrix_file(data_dir: Path, subject: str, session: str, fmt: str) -> Path | None:
    subj_clean    = subject.replace("sub-", "")
    subj_prefixed = f"sub-{subj_clean}"
    ses_clean     = str(session).replace("ses-", "")

    patterns = [
        f"*{subj_clean}*ses*{ses_clean}*.{fmt}",
        f"*{subj_prefixed}*ses*{ses_clean}*.{fmt}",
        f"ses-{ses_clean}/*{subj_clean}*.{fmt}",
        f"ses-{ses_clean}/*{subj_prefixed}*.{fmt}",
        f"*{subj_clean}*{ses_clean}*.{fmt}",
        f"**/*{subj_clean}*{ses_clean}*.{fmt}",
    ]
    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            return matches[0]

    if fmt == "mat":
        for pattern in [
            f"*{subj_clean}*ses*{ses_clean}*.connectivity.mat",
            f"**/*{subj_clean}*{ses_clean}*.connectivity.mat",
        ]:
            matches = list(data_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def load_connectivity_matrix(data_dir: Path, subject: str, session: str,
                              fmt: str, n_nodes: int) -> np.ndarray | None:
    filepath = find_matrix_file(data_dir, subject, session, fmt)
    if filepath is None:
        return None
    A = _load_matrix_raw(filepath, fmt)
    if A is None:
        return None
    if A.shape != (n_nodes, n_nodes):
        print(f"  ⚠ Shape mismatch for {subject} ses-{session}: {A.shape} (expected {n_nodes}×{n_nodes})")
        return None
    return A


# ============================================================================
# NODE-LEVEL METRICS
# ============================================================================

def classify_hubs(participation_coef: np.ndarray,
                  within_module_zscore: np.ndarray) -> np.ndarray:
    """
    Classify nodes into hub types (Guimerà & Amaral 2005).
    Returns array of strings: provincial_hub | connector_hub | kinless_node | peripheral | unknown
    """
    hub_types = []
    for pc, wz in zip(participation_coef, within_module_zscore):
        if np.isnan(pc) or np.isnan(wz):
            hub_types.append("unknown")
        elif wz > PROVINCIAL_HUB_THRESHOLD and pc < CONNECTOR_HUB_THRESHOLD:
            hub_types.append("provincial_hub")
        elif pc > CONNECTOR_HUB_THRESHOLD and wz > 1.0:
            hub_types.append("connector_hub")
        elif pc < KINLESS_HUB_THRESHOLD and wz < 1.0:
            hub_types.append("peripheral")
        else:
            hub_types.append("kinless_node")
    return np.array(hub_types)


def compute_node_metrics(A: np.ndarray, n_nodes: int, binarize: bool = False) -> dict:
    """
    Compute comprehensive node-level metrics for a connectivity matrix.

    Returns dict with arrays of length n_nodes plus scalar modularity.
    """
    nan_vec = lambda: np.full(n_nodes, np.nan)

    if binarize:
        A = (A > 0).astype(float)
    A_bin = (A > 0).astype(int)

    metrics: dict = {}

    metrics["degree"]   = bct.degrees_und(A_bin).astype(float)
    metrics["strength"] = bct.strengths_und(A)

    try:
        metrics["clustering"] = bct.clustering_coef_wu(A)
    except Exception:
        metrics["clustering"] = nan_vec()

    try:
        metrics["local_efficiency"] = bct.efficiency_wei(A, local=True)
    except Exception:
        metrics["local_efficiency"] = nan_vec()

    try:
        metrics["betweenness"] = bct.betweenness_wei(A)
    except Exception:
        metrics["betweenness"] = nan_vec()

    try:
        Ci, Q = bct.community_louvain(A)
        metrics["community"]           = Ci.astype(float)
        metrics["modularity"]          = float(Q)
        metrics["participation_coef"]  = bct.participation_coef(A, Ci)
        metrics["within_module_zscore"] = bct.module_degree_zscore(A, Ci)
        metrics["hub_type"]            = classify_hubs(
            metrics["participation_coef"],
            metrics["within_module_zscore"],
        )
    except Exception as e:
        print(f"  ⚠ Community detection failed: {e}")
        metrics["community"]            = nan_vec()
        metrics["modularity"]           = np.nan
        metrics["participation_coef"]   = nan_vec()
        metrics["within_module_zscore"] = nan_vec()
        metrics["hub_type"]             = np.full(n_nodes, "unknown")

    return metrics


def compute_rich_club(A: np.ndarray) -> np.ndarray | None:
    """Compute rich-club coefficient across degree thresholds."""
    try:
        A_bin = (A > 0).astype(int)
        return bct.rich_club_bu(A_bin)
    except Exception:
        return None


def build_node_records(subject: str, session, atlas: str, subj_meta: pd.Series,
                       metrics: dict, rich_club_mean: float,
                       group_col: str | None, sex_col: str | None,
                       n_nodes: int) -> list[dict]:
    """Turn per-node metric arrays into a list of flat dicts (one per node)."""
    records = []
    for i in range(n_nodes):
        rec = {
            "subject": subject,
            "session": session,
            "atlas":   atlas,
            "node":    i + 1,
        }
        if group_col: rec[group_col] = subj_meta.get(group_col, np.nan)
        if sex_col:   rec[sex_col]   = subj_meta.get(sex_col, np.nan)

        for key in ["degree", "strength", "betweenness", "clustering",
                    "local_efficiency", "participation_coef",
                    "within_module_zscore", "community", "hub_type"]:
            arr = metrics.get(key)
            rec[key] = arr[i] if arr is not None and not np.isscalar(arr) else np.nan

        records.append(rec)
    return records


def build_summary_record(subject: str, session, atlas: str, subj_meta: pd.Series,
                          metrics: dict, rich_club_mean: float,
                          group_col: str | None, sex_col: str | None) -> dict:
    """Build a single subject-session summary dict."""
    hub_type = metrics.get("hub_type", np.array([]))
    summary = {
        "subject": subject,
        "session": session,
        "atlas":   atlas,
        "modularity":            metrics.get("modularity", np.nan),
        "n_provincial_hubs":     int(np.sum(hub_type == "provincial_hub")),
        "n_connector_hubs":      int(np.sum(hub_type == "connector_hub")),
        "n_kinless_nodes":       int(np.sum(hub_type == "kinless_node")),
        "n_peripheral":          int(np.sum(hub_type == "peripheral")),
        "mean_participation":    float(np.nanmean(metrics.get("participation_coef",   [np.nan]))),
        "mean_within_module_z":  float(np.nanmean(metrics.get("within_module_zscore", [np.nan]))),
        "mean_betweenness":      float(np.nanmean(metrics.get("betweenness",          [np.nan]))),
        "mean_strength":         float(np.nanmean(metrics.get("strength",             [np.nan]))),
        "rich_club_coef":        rich_club_mean,
    }
    if group_col: summary[group_col] = subj_meta.get(group_col, np.nan)
    if sex_col:   summary[sex_col]   = subj_meta.get(sex_col,   np.nan)
    return summary


# ============================================================================
# PLOTS
# ============================================================================

def make_plots(summary_df: pd.DataFrame, plot_dir: Path,
               group_col: str | None) -> None:
    plot_dir.mkdir(exist_ok=True)

    # 1. Hub counts by group
    if group_col and group_col in summary_df.columns:
        hub_cols = ["n_provincial_hubs", "n_connector_hubs", "n_kinless_nodes", "n_peripheral"]
        available = [c for c in hub_cols if c in summary_df.columns]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
            if len(available) == 1:
                axes = [axes]
            for ax, col in zip(axes, available):
                sns.boxplot(data=summary_df, x=group_col, y=col, ax=ax)
                ax.set_title(col.replace("_", " ").title())
                ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            plt.savefig(plot_dir / "hub_counts_by_group.png", dpi=150)
            plt.close()
            print("  ✓ hub_counts_by_group.png")

    # 2. Mean participation coefficient by group
    if group_col and "mean_participation" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=summary_df, x=group_col, y="mean_participation", ax=ax)
        ax.set_title("Mean Participation Coefficient by Group")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(plot_dir / "participation_by_group.png", dpi=150)
        plt.close()
        print("  ✓ participation_by_group.png")

    # 3. Hub type distribution (stacked bar)
    hub_cols = ["n_provincial_hubs", "n_connector_hubs", "n_kinless_nodes", "n_peripheral"]
    available = [c for c in hub_cols if c in summary_df.columns]
    if available:
        means = summary_df[available].mean()
        fig, ax = plt.subplots(figsize=(7, 5))
        means.plot(kind="bar", ax=ax, color=["steelblue", "tomato", "goldenrod", "gray"])
        ax.set_title("Mean Hub Type Counts per Subject-Session")
        ax.set_ylabel("Mean Count")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(plot_dir / "hub_type_distribution.png", dpi=150)
        plt.close()
        print("  ✓ hub_type_distribution.png")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = build_config(args)

    data_dir:      Path = config["data_dir"]
    metadata_file: Path = config["metadata_file"]
    output_dir:    Path = config["output_dir"]
    binarize:      bool = config["binarize"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NODE-LEVEL NETWORK METRICS ANALYSIS")
    print("=" * 70)
    print(f"Data directory:   {data_dir}")
    print(f"Metadata file:    {metadata_file}")
    print(f"Output directory: {output_dir}")
    print(f"Binarize:         {binarize}")
    print()

    # ── Load metadata ──────────────────────────────────────────────────────
    print("Loading metadata...")
    sep      = "\t" if metadata_file.suffix == ".tsv" else ","
    metadata = pd.read_csv(metadata_file, sep=sep)
    print(f"  ✓ {len(metadata)} records, columns: {list(metadata.columns)}")

    # ── Auto-detection ─────────────────────────────────────────────────────
    print("\nAuto-detecting data structure...")
    cols     = detect_metadata_columns(metadata)
    fmt      = detect_file_format(data_dir)
    n_nodes  = detect_n_nodes(data_dir, fmt)
    atlas    = detect_atlas_name(data_dir)
    sessions = detect_sessions(metadata, cols["session_col"])

    subject_col = cols["subject_col"]
    session_col = cols["session_col"]
    group_col   = cols.get("group_col")
    sex_col     = cols.get("sex_col")

    print(f"\n  Summary: atlas={atlas}, n_nodes={n_nodes}, format={fmt}, sessions={sessions}")
    print()

    # ── Compute metrics ────────────────────────────────────────────────────
    print("Computing node-level metrics...")
    all_node_records = []
    all_summaries    = []

    for _, row in metadata.iterrows():
        subject = str(row[subject_col])
        session = str(row[session_col])

        A = load_connectivity_matrix(data_dir, subject, session, fmt, n_nodes)
        if A is None:
            continue

        metrics        = compute_node_metrics(A, n_nodes, binarize=binarize)
        rc             = compute_rich_club(A)
        rc_mean        = float(np.nanmean(rc)) if rc is not None and len(rc) > 0 else np.nan

        all_node_records.extend(
            build_node_records(subject, session, atlas, row, metrics, rc_mean,
                               group_col, sex_col, n_nodes)
        )
        all_summaries.append(
            build_summary_record(subject, session, atlas, row, metrics, rc_mean,
                                 group_col, sex_col)
        )

    print(f"\n✓ Processed {len(all_summaries)} subject-sessions, {len(all_node_records)} node records")

    if not all_node_records:
        sys.exit("✗ No matrices could be loaded. Check data_dir and file naming.")

    node_df    = pd.DataFrame(all_node_records)
    summary_df = pd.DataFrame(all_summaries)

    # ── Save ───────────────────────────────────────────────────────────────
    node_df.to_parquet(output_dir / "node_level_metrics.parquet", index=False)
    node_df.to_csv(output_dir / "node_level_metrics.csv", index=False)
    summary_df.to_parquet(output_dir / "subject_hub_summaries.parquet", index=False)
    summary_df.to_csv(output_dir / "subject_hub_summaries.csv", index=False)
    print("✓ Saved: node_level_metrics.parquet/.csv + subject_hub_summaries.parquet/.csv")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nCreating plots...")
    make_plots(summary_df, output_dir / "plots", group_col)

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()