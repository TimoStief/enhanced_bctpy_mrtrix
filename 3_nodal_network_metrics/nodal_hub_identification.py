#!/usr/bin/env python3
"""
SCRIPT: Node-Level Brain Network Analysis
==========================================

PURPOSE:
    Compute node-level network metrics from connectivity matrices.
    Input/output paths are provided via run_spec.json or CLI arguments.
    All other parameters (n_nodes, atlas, file_format, columns) are
    automatically detected from the data.

USAGE:
    python 02_nodal_metrics.py run_spec.json
    python 02_nodal_metrics.py --data-dir /path/to/data --metadata /path/to/meta.tsv --output-dir /path/to/out

AUTHOR: Analysis Pipeline
VERSION: 1.0 (Auto-detection, run_spec driven)
"""

from __future__ import annotations
from datetime import datetime

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
# CLI / run_spec LOADING
# ============================================================================


def _progress(current, total, desc):
    """Simple progress display without external dependencies."""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  {desc}: |{bar}| {current}/{total} ({pct:.0f}%)", end="\r", flush=True)
    if current == total:
        print()

# ============================================================================
# MATRIX TYPE DETECTION & NORMALIZATION
# ============================================================================

def detect_matrix_type(A: np.ndarray) -> dict:
    """Auto-detect matrix type from values."""
    max_val  = A.max()
    min_val  = A[A > 0].min() if (A > 0).any() else 0
    n_unique = len(np.unique(A))
    is_binary = n_unique <= 2

    if is_binary:
        mat_type = "binary"
        confidence = "high"
        recommendation = "binary"
    elif max_val > 1000:
        mat_type = "fiber_counts"
        confidence = "high"
        recommendation = "log"
    elif max_val > 1:
        mat_type = "weighted_unnormalized"
        confidence = "medium"
        recommendation = "max"
    else:
        mat_type = "weighted_normalized"
        confidence = "high"
        recommendation = "none"

    return {
        "type": mat_type,
        "max": float(max_val),
        "min_nonzero": float(min_val),
        "recommendation": recommendation,
        "confidence": confidence,
    }


def print_matrix_type_help():
    """Print help for matrix types."""
    print("  ── Matrix type guide ──────────────────────────────────────────")
    print("  [1] fiber_counts   Raw tractography streamline counts (max >> 1)")
    print("                     e.g. MRtrix, DSI Studio count matrices")
    print("                     → log-normalization recommended")
    print("  [2] weighted       FA-weighted or NOS-normalized (max ~0-1)")
    print("                     → no normalization needed")
    print("  [3] binary         Binary connectivity (0 or 1 only)")
    print("                     → binarize recommended")
    print("  [4] log            Already log-normalized")
    print("                     → no normalization needed")
    print("  ───────────────────────────────────────────────────────────────")


def ask_normalize(data_dir: Path, fmt: str, n_nodes: int,
                  normalize_arg: str | None) -> str:
    """
    Detect matrix type and ask user for normalization preference.
    Returns: 'log', 'max', 'binary', or 'none'
    """
    # Find first matrix file
    patterns = [f"*.{fmt}", f"**/*.{fmt}"]
    first_file = None
    for pat in patterns:
        files = list(data_dir.glob(pat))
        if files:
            first_file = files[0]
            break

    if first_file is None:
        print("  ⚠ Could not find matrix for type detection")
        return normalize_arg or "none"

    try:
        if fmt == "npy":
            A = np.load(first_file)
        else:
            import scipy.io as sio
            mat = sio.loadmat(str(first_file))
            A = list(mat.values())[-1]
        A = np.array(A, dtype=float)
        np.fill_diagonal(A, 0)
    except Exception as e:
        print(f"  ⚠ Could not load matrix for detection: {e}")
        return normalize_arg or "none"

    info = detect_matrix_type(A)

    print("\n" + "=" * 70)
    print("MATRIX TYPE DETECTION")
    print("=" * 70)
    print(f"  File:          {first_file.name}")
    print(f"  Shape:         {A.shape[0]} × {A.shape[1]}")
    print(f"  Max value:     {info['max']:.1f}")
    print(f"  Min (nonzero): {info['min_nonzero']:.4f}")
    print(f"  Sparsity:      {(A == 0).sum() / A.size * 100:.1f}% zeros")
    print(f"  Symmetric:     {'yes' if np.allclose(A, A.T) else 'no'}")
    print()
    print(f"  Detected type: {info['type'].upper()} (confidence: {info['confidence']})")
    print(f"  Recommendation: {info['recommendation'].upper()} normalization")
    print()

    # If normalize_arg already given via CLI → use it
    if normalize_arg and normalize_arg != "auto":
        print(f"  Using CLI argument: --normalize {normalize_arg}")
        print("=" * 70)
        return normalize_arg

    # Ask user
    print_matrix_type_help()
    rec_map = {
        "fiber_counts": "1", "weighted_normalized": "2",
        "binary": "3", "log": "4", "weighted_unnormalized": "2"
    }
    rec_num = rec_map.get(info["type"], "1")
    rec_label = info["recommendation"]

    print(f"\n  Enter choice [1-4] or press Enter to use recommendation [{rec_num} = {rec_label}]: ", end="", flush=True)
    choice = input().strip()

    if not choice:
        choice = rec_num

    mapping = {"1": "log", "2": "none", "3": "binary", "4": "none"}
    result = mapping.get(choice, rec_label)
    print(f"  ✓ Using normalization: {result.upper()}")
    print("=" * 70)
    return result


def normalize_matrix(A: np.ndarray, normalize: str) -> np.ndarray:
    """Normalize connectivity matrix."""
    if normalize == "log":
        A_norm = np.log1p(A)
        max_val = A_norm.max()
        if max_val > 0:
            A_norm = A_norm / max_val
        return A_norm
    elif normalize == "max":
        max_val = A.max()
        return A / max_val if max_val > 0 else A
    elif normalize == "binary":
        return (A > 0).astype(float)
    else:
        return A.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node-level network metrics analysis. Pass run_spec.json or explicit paths."
    )
    parser.add_argument("run_spec", nargs="?", help="Path to run_spec.json")
    parser.add_argument("--data-dir",   help="Directory with connectivity matrices")
    parser.add_argument("--metadata",   help="Participant metadata file (CSV or TSV)")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--binarize", action="store_true", default=None,
                        help="Binarize connectivity matrices before analysis")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Build config from run_spec.json and/or CLI args. CLI args override run_spec."""
    config: dict = {}

    if args.run_spec:
        spec_path = Path(args.run_spec).expanduser().resolve()
        if not spec_path.exists():
            sys.exit(f"✗ run_spec not found: {spec_path}")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        inputs  = spec.get("inputs", {})
        outputs = spec.get("outputs", {})
        config["data_dir"]      = inputs.get("data_dir")
        config["metadata_file"] = inputs.get("metadata_file")
        config["output_dir"]    = outputs.get("output_dir")
        config["binarize"]      = spec.get("binarize", False)

    if args.data_dir:    config["data_dir"]      = args.data_dir
    if args.metadata:    config["metadata_file"] = args.metadata
    if args.output_dir:  config["output_dir"]    = args.output_dir
    if args.binarize is not None:
        config["binarize"] = args.binarize

    missing = [k for k in ("data_dir", "metadata_file", "output_dir") if not config.get(k)]
    if missing:
        sys.exit(
            f"✗ Missing required config: {', '.join(missing)}\n"
            "  Provide via run_spec.json or CLI flags (--data-dir, --metadata, --output-dir)"
        )

    config["data_dir"]      = Path(config["data_dir"]).expanduser().resolve()
    config["metadata_file"] = Path(config["metadata_file"]).expanduser().resolve()
    config["output_dir"]    = Path(config["output_dir"]).expanduser().resolve()
    config.setdefault("binarize", False)
    if args.normalize: config["normalize"] = args.normalize
    config.setdefault("normalize", None)
    return config


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
        print(f"    computing clustering coefficient...    ", end="\n", flush=True)
        metrics["clustering"] = bct.clustering_coef_wu(A)
    except Exception:
        metrics["clustering"] = nan_vec()

    try:
        print(f"    computing local efficiency (slow)...   ", end="\n", flush=True)
        metrics["local_efficiency"] = bct.efficiency_wei(A, local=True)
    except Exception:
        metrics["local_efficiency"] = nan_vec()

    try:
        print(f"    computing betweenness centrality (slow)...", end="\n", flush=True)
        metrics["betweenness"] = bct.betweenness_wei(A)
    except Exception:
        metrics["betweenness"] = nan_vec()

    try:
        print(f"    computing community detection...       ", end="\n", flush=True)
        Ci, Q = bct.community_louvain(A)
        metrics["community"]           = Ci.astype(float)
        metrics["modularity"]          = float(Q)
        print(f"    computing participation coefficient..  ", end="\n", flush=True)
        metrics["participation_coef"]  = bct.participation_coef(A, Ci)
        print(f"    computing within-module z-score...     ", end="\n", flush=True)
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
    config = load_config(args)

    data_dir:      Path = config["data_dir"]
    metadata_file: Path = config["metadata_file"]
    output_dir:    Path = config["output_dir"]
    binarize:      bool = config["binarize"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
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

    # ── Matrix normalization ───────────────────────────────────────────────
    normalize = ask_normalize(data_dir, fmt, n_nodes, config.get("normalize"))

    # ── Compute metrics ────────────────────────────────────────────────────
    print("Computing node-level metrics...")
    all_node_records = []
    all_summaries    = []

    _total_s = len(metadata)
    for _i_s, (_, row) in enumerate(metadata.iterrows()):
        _progress(_i_s + 1, _total_s, f"Computing {row[subject_col]} ses-{row[session_col]}")
        subject = str(row[subject_col])
        session = str(row[session_col])

        print(f"  → {subject} ses-{session} loading matrix...                              ", end="\n", flush=True)
        A = load_connectivity_matrix(data_dir, subject, session, fmt, n_nodes)
        if A is None:
            continue
        A = normalize_matrix(A, normalize)
        print(f"  → {subject} ses-{session} computing degree, betweenness, hub classification...  ", end="\n", flush=True)
        metrics        = compute_node_metrics(A, n_nodes, binarize=binarize)
        print(f"  ✓ {subject} ses-{session} done                                                   ", end="\n", flush=True)
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
    _end_time = datetime.now()
    _duration = _end_time - _start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Finished: " + _end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Duration: " + str(_duration).split(".")[0])
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()