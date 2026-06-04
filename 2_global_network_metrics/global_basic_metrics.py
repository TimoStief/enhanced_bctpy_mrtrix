#!/usr/bin/env python3
"""
SCRIPT: Global Network Metrics Analysis
========================================

PURPOSE:
    Compute global (whole-brain) network metrics from connectivity matrices.
    Input/output paths are provided via run_spec.json or CLI arguments.
    All other parameters (n_nodes, atlas, file_pattern, columns) are
    automatically detected from the data.

USAGE:
    python global_basic_metrics.py run_spec.json
    python global_basic_metrics.py --data-dir /path/to/data --metadata /path/to/meta.tsv --output-dir /path/to/out

AUTHOR: Analysis Pipeline
VERSION: 3.0 (Auto-detection, run_spec driven)
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
from mpl_toolkits.mplot3d import Axes3D

import bct
import seaborn as sns
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP


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
        description="Global network metrics analysis. Pass run_spec.json or explicit paths."
    )
    parser.add_argument("run_spec", nargs="?", help="Path to run_spec.json")
    parser.add_argument("--data-dir", help="Directory with connectivity matrices")
    parser.add_argument("--metadata", help="Participant metadata file (CSV or TSV)")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--binarize", action="store_true", default=None,
                        help="Binarize connectivity matrices before analysis")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """
    Build config from run_spec.json and/or CLI args.
    CLI args override run_spec values.
    """
    config: dict = {}

    if args.run_spec:
        spec_path = Path(args.run_spec).expanduser().resolve()
        if not spec_path.exists():
            sys.exit(f"✗ run_spec not found: {spec_path}")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        inputs = spec.get("inputs", {})
        outputs = spec.get("outputs", {})
        config["data_dir"] = inputs.get("data_dir")
        config["metadata_file"] = inputs.get("metadata_file")
        config["output_dir"] = outputs.get("output_dir")
        config["binarize"] = spec.get("binarize", False)

    # CLI overrides
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.metadata:
        config["metadata_file"] = args.metadata
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.binarize is not None:
        config["binarize"] = args.binarize

    # Validate required fields
    missing = [k for k in ("data_dir", "metadata_file", "output_dir") if not config.get(k)]
    if missing:
        sys.exit(
            f"✗ Missing required config: {', '.join(missing)}\n"
            "  Provide via run_spec.json or CLI flags (--data-dir, --metadata, --output-dir)"
        )

    config["data_dir"] = Path(config["data_dir"]).expanduser().resolve()
    config["metadata_file"] = Path(config["metadata_file"]).expanduser().resolve()
    config["output_dir"] = Path(config["output_dir"]).expanduser().resolve()
    config.setdefault("binarize", False)
    if args.normalize: config["normalize"] = args.normalize
    config.setdefault("normalize", None)

    return config


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_metadata_columns(metadata: pd.DataFrame) -> dict:
    """
    Fuzzy-match column names for subject, session, group, sex.
    Returns dict with detected column names.
    """
    cols = {c.lower(): c for c in metadata.columns}

    def find(candidates):
        for c in candidates:
            if c in cols:
                return cols[c]
        # partial match fallback
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
        status = f"✓ {v}" if v else "⚠ not found"
        print(f"    {k}: {status}")

    missing = [k for k, v in detected.items() if v is None and k in ("subject_col", "session_col")]
    if missing:
        sys.exit(f"✗ Could not detect required columns: {missing}\n"
                 f"  Available columns: {list(metadata.columns)}")

    return detected


def detect_file_format(data_dir: Path) -> tuple[str, str]:
    """
    Scan data_dir for connectivity matrix files.
    Returns (glob_pattern, file_format) where file_format is 'npy' or 'mat'.
    """
    npy_files = list(data_dir.rglob("*.npy"))
    mat_files = list(data_dir.rglob("*.connectivity.mat"))
    if not mat_files:
        mat_files = list(data_dir.rglob("*.mat"))

    if not npy_files and not mat_files:
        sys.exit(f"✗ No .npy or .mat files found in: {data_dir}")

    if len(npy_files) >= len(mat_files):
        fmt = "npy"
        print(f"  ✓ File format detected: .npy ({len(npy_files)} files)")
    else:
        fmt = "mat"
        print(f"  ✓ File format detected: .mat ({len(mat_files)} files)")

    return fmt


def detect_n_nodes(data_dir: Path, fmt: str) -> int:
    """Load first matrix found and return its node count."""
    pattern = "*.npy" if fmt == "npy" else "*.mat"
    files = list(data_dir.rglob(pattern))
    if not files:
        sys.exit(f"✗ No {pattern} files found in: {data_dir}")

    try:
        A = _load_matrix_raw(files[0], fmt)
        if A is not None:
            print(f"  ✓ n_nodes detected: {A.shape[0]} (from {files[0].name})")
            return A.shape[0]
    except Exception as e:
        sys.exit(f"✗ Could not load matrix for node detection: {e}")

    sys.exit("✗ Could not detect n_nodes from data")


def detect_atlas_name(data_dir: Path) -> str:
    """
    Try to extract atlas name from folder structure or filenames.
    Falls back to 'unknown'.
    """
    known = ["brainnectome", "brodmann", "aal", "schaefer", "destrieux", "desikan", "hcp"]
    search_str = str(data_dir).lower()

    for name in known:
        if name in search_str:
            print(f"  ✓ Atlas detected: {name.capitalize()} (from path)")
            return name.capitalize()

    # Try first file name
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
    """Load a single matrix file regardless of subject/session."""
    try:
        if fmt == "npy":
            A = np.load(filepath)
        else:
            mat = loadmat(filepath)
            keys = [k for k in mat.keys() if not k.startswith("__")]
            A = mat.get("connectivity", mat[keys[0]] if keys else None)
            if A is None:
                return None
        return np.asarray(A, dtype=float)
    except Exception:
        return None


def find_matrix_file(data_dir: Path, subject: str, session: str, fmt: str) -> Path | None:
    """
    Search data_dir recursively for a file matching subject and session.
    Tries multiple naming conventions.
    """
    subj_clean = subject.replace("sub-", "")
    subj_prefixed = f"sub-{subj_clean}"
    ses_clean = str(session).replace("ses-", "")

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

    # .connectivity.mat fallback
    if fmt == "mat":
        for pattern in [
            f"*{subj_clean}*ses*{ses_clean}*.connectivity.mat",
            f"**/*{subj_clean}*{ses_clean}*.connectivity.mat",
        ]:
            matches = list(data_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


def load_connectivity_matrix(
    data_dir: Path, subject: str, session: str, fmt: str, n_nodes: int
) -> np.ndarray | None:
    filepath = find_matrix_file(data_dir, subject, session, fmt)

    if filepath is None:
        print(f"  ⚠ No file found: {subject} ses-{session}")
        return None

    A = _load_matrix_raw(filepath, fmt)
    if A is None:
        print(f"  ⚠ Load failed: {filepath.name}")
        return None

    if A.shape != (n_nodes, n_nodes):
        print(f"  ⚠ Shape mismatch {A.shape} (expected {n_nodes}×{n_nodes}): {filepath.name}")
        return None

    return A


# ============================================================================
# METRICS
# ============================================================================

def compute_global_metrics(A: np.ndarray, binarize: bool = False) -> dict:
    """Compute global network metrics from connectivity matrix."""
    nan_result = {
        "density": np.nan, "path_length": np.nan, "global_efficiency": np.nan,
        "clustering_coef": np.nan, "transitivity": np.nan, "modularity": np.nan,
        "n_communities": np.nan, "participation_coef_mean": np.nan,
        "local_efficiency_mean": np.nan, "betweenness_mean": np.nan,
        "small_worldness": np.nan,
    }

    try:
        A_proc = (A > 0).astype(float) if binarize else A.copy()

        if not np.any(A_proc):
            return nan_result

        A_bin = (A_proc > 0).astype(float)
        print(f"    computing density...                    ", end="\n", flush=True)
        density = bct.density_und(A_bin)[0] if isinstance(bct.density_und(A_bin), tuple) else bct.density_und(A_bin)

        print(f"    computing path length + efficiency...   ", end="\n", flush=True)
        L = bct.distance_wei(1.0 / (A_proc + 1e-10))[0]
        path_length = np.mean(L[L > 0])
        global_efficiency = 1.0 / path_length if path_length > 0 else np.nan

        print(f"    computing clustering + transitivity...  ", end="\n", flush=True)
        C = bct.clustering_coef_wu(A_proc)
        clustering_coef = np.mean(C)
        transitivity = bct.transitivity_wu(A_proc)

        print(f"    computing community detection...        ", end="\n", flush=True)
        Ci, Q = bct.community_louvain(A_bin)
        n_communities = len(np.unique(Ci[~np.isnan(Ci)]))

        print(f"    computing participation coefficient...  ", end="\n", flush=True)
        participation = bct.participation_coef(A_proc, Ci)
        participation_mean = np.nanmean(participation)

        print(f"    computing local efficiency (slow)...    ", end="\n", flush=True)
        local_eff = bct.efficiency_wei(A_proc, local=True)
        local_eff_mean = np.nanmean(local_eff)

        print(f"    computing betweenness centrality (slow)...", end="\n", flush=True)
        betweenness = bct.betweenness_wei(A_proc)
        betweenness_mean = np.nanmean(betweenness)

        print(f"    computing small-worldness...            ", end="\n", flush=True)
        small_worldness = clustering_coef / path_length if path_length > 0 else np.nan

        return {
            "density": density,
            "path_length": path_length,
            "global_efficiency": global_efficiency,
            "clustering_coef": clustering_coef,
            "transitivity": transitivity,
            "modularity": Q,
            "n_communities": n_communities,
            "participation_coef_mean": participation_mean,
            "local_efficiency_mean": local_eff_mean,
            "betweenness_mean": betweenness_mean,
            "small_worldness": small_worldness,
        }

    except Exception as e:
        print(f"  ⚠ Metric error: {e}")
        return nan_result


# ============================================================================
# UMAP + TRAJECTORIES
# ============================================================================

def run_umap(results_df: pd.DataFrame, meta_cols: set, umap_params: dict) -> pd.DataFrame:
    feature_cols = [c for c in results_df.columns if c not in meta_cols]
    X = results_df[feature_cols].values
    X = np.nan_to_num(X, nan=0)
    X_scaled = StandardScaler().fit_transform(X)

    reducer = UMAP(
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        n_components=3,
        metric=umap_params.get("metric", "euclidean"),
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)

    out = results_df.copy()
    out["umap_1"] = embedding[:, 0]
    out["umap_2"] = embedding[:, 1]
    out["umap_3"] = embedding[:, 2]
    return out


def compute_trajectories(umap_df: pd.DataFrame, subject_col: str, session_col: str, group_col: str | None, sex_col: str | None) -> pd.DataFrame:
    records = []
    for subject in umap_df[subject_col].unique():
        subj = umap_df[umap_df[subject_col] == subject].sort_values(session_col)
        if len(subj) < 2:
            continue

        positions = subj[["umap_1", "umap_2", "umap_3"]].values
        dists = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(len(positions)-1)]
        total = sum(dists)
        acceleration = dists[1] / dists[0] if len(dists) >= 2 and dists[0] > 0.01 else np.nan

        rec = {
            subject_col: subject,
            "early_change": dists[0] if dists else np.nan,
            "late_change": dists[-1] if len(dists) > 1 else np.nan,
            "total_distance": total,
            "acceleration": acceleration,
        }
        if group_col and group_col in subj.columns:
            rec[group_col] = subj.iloc[0][group_col]
        if sex_col and sex_col in subj.columns:
            rec[sex_col] = subj.iloc[0][sex_col]

        records.append(rec)

    return pd.DataFrame(records)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def make_plots(results_df: pd.DataFrame, umap_df: pd.DataFrame, trajectory_df: pd.DataFrame,
               plot_dir: Path, group_col: str | None, subject_col: str) -> None:
    plot_dir.mkdir(exist_ok=True)

    # 1. Metrics by group
    if group_col and group_col in results_df.columns:
        metric_cols = ["path_length", "global_efficiency", "clustering_coef", "small_worldness"]
        available = [c for c in metric_cols if c in results_df.columns]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5))
            if len(available) == 1:
                axes = [axes]
            for ax, metric in zip(axes, available):
                sns.boxplot(data=results_df, x=group_col, y=metric, ax=ax)
                ax.set_title(metric.replace("_", " ").title())
                ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            plt.savefig(plot_dir / "metrics_by_group.png", dpi=150)
            plt.close()
            print("  ✓ metrics_by_group.png")

    # 2. 3D UMAP
    if "umap_1" in umap_df.columns:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        if group_col and group_col in umap_df.columns:
            groups = umap_df[group_col].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
            for grp, col in zip(groups, colors):
                mask = umap_df[group_col] == grp
                ax.scatter(umap_df[mask]["umap_1"], umap_df[mask]["umap_2"],
                           umap_df[mask]["umap_3"], label=str(grp), color=col, s=60, alpha=0.6)
            ax.legend()
        else:
            ax.scatter(umap_df["umap_1"], umap_df["umap_2"], umap_df["umap_3"], s=60, alpha=0.6)

        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
        ax.set_title("UMAP 3D — Global Metrics")
        plt.tight_layout()
        plt.savefig(plot_dir / "umap_3d.png", dpi=150)
        plt.close()
        print("  ✓ umap_3d.png")

    # 3. Trajectory scatter
    if not trajectory_df.empty and "early_change" in trajectory_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        if group_col and group_col in trajectory_df.columns:
            groups = trajectory_df[group_col].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
            for grp, col in zip(groups, colors):
                mask = trajectory_df[group_col] == grp
                ax.scatter(trajectory_df[mask]["early_change"],
                           trajectory_df[mask]["late_change"],
                           label=str(grp), color=col, s=80, alpha=0.6)
            ax.legend()
        else:
            ax.scatter(trajectory_df["early_change"], trajectory_df["late_change"], s=80, alpha=0.6)

        ax.axline((0, 0), slope=1, color="red", linestyle="--", alpha=0.4, label="Equal")
        ax.set_xlabel("Early Change (Sess 1→2)")
        ax.set_ylabel("Late Change (Sess 2→3)")
        ax.set_title("Trajectory Pattern")
        plt.tight_layout()
        plt.savefig(plot_dir / "trajectory_scatter.png", dpi=150)
        plt.close()
        print("  ✓ trajectory_scatter.png")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()
    config = load_config(args)

    data_dir: Path = config["data_dir"]
    metadata_file: Path = config["metadata_file"]
    output_dir: Path = config["output_dir"]
    binarize: bool = config["binarize"]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("GLOBAL NETWORK METRICS ANALYSIS")
    print("=" * 70)
    print(f"Data directory:   {data_dir}")
    print(f"Metadata file:    {metadata_file}")
    print(f"Output directory: {output_dir}")
    print(f"Binarize:         {binarize}")
    print()

    # ── Load metadata ──────────────────────────────────────────────────────
    print("Loading metadata...")
    sep = "\t" if metadata_file.suffix == ".tsv" else ","
    metadata = pd.read_csv(metadata_file, sep=sep)
    print(f"  ✓ {len(metadata)} records, columns: {list(metadata.columns)}")

    # ── Auto-detection ─────────────────────────────────────────────────────
    print("\nAuto-detecting data structure...")
    cols = detect_metadata_columns(metadata)
    fmt = detect_file_format(data_dir)
    n_nodes = detect_n_nodes(data_dir, fmt)
    atlas = detect_atlas_name(data_dir)
    sessions = detect_sessions(metadata, cols["session_col"])

    subject_col = cols["subject_col"]
    session_col = cols["session_col"]
    group_col = cols.get("group_col")
    sex_col = cols.get("sex_col")

    print(f"\n  Summary: atlas={atlas}, n_nodes={n_nodes}, format={fmt}, sessions={sessions}")
    print()

    # ── Matrix normalization ───────────────────────────────────────────────
    normalize = ask_normalize(data_dir, fmt, n_nodes, config.get("normalize"))

    # ── Compute metrics ────────────────────────────────────────────────────
    print("Computing global metrics...")
    results = []

    _total_s = len(metadata)
    for _i_s, (_, row) in enumerate(metadata.iterrows()):
        _progress(_i_s + 1, _total_s, f"Processing {row[subject_col]} ses-{row[session_col]}")
        subject = str(row[subject_col])
        session = str(row[session_col])

        print(f"  → {subject} ses-{session} loading matrix...                              ", end="\n", flush=True)
        A = load_connectivity_matrix(data_dir, subject, session, fmt, n_nodes)
        if A is None:
            continue
        A = normalize_matrix(A, normalize)
        print(f"  → {subject} ses-{session} computing density, clustering, betweenness...  ", end="\n", flush=True)
        metrics = compute_global_metrics(A, binarize=binarize)
        print(f"  ✓ {subject} ses-{session} done                                             ", end="\n", flush=True)

        record = {"subject": subject, "session": session, **metrics}
        if group_col and group_col in row:
            record[group_col] = row[group_col]
        if sex_col and sex_col in row:
            record[sex_col] = row[sex_col]

        results.append(record)

    print(f"\n✓ Computed metrics for {len(results)} records")

    if not results:
        sys.exit("✗ No matrices could be loaded. Check data_dir and file naming.")

    results_df = pd.DataFrame(results)

    # ── Save metrics ───────────────────────────────────────────────────────
    results_df.to_parquet(output_dir / "global_metrics.parquet", index=False)
    results_df.to_csv(output_dir / "global_metrics.csv", index=False)
    print(f"✓ Saved: global_metrics.parquet + .csv")

    # ── Summary stats ──────────────────────────────────────────────────────
    if group_col and group_col in results_df.columns:
        print("\nSummary by group:")
        for grp in results_df[group_col].unique():
            d = results_df[results_df[group_col] == grp]
            print(f"  {grp} (N={len(d)}):")
            for metric in ["global_efficiency", "clustering_coef", "path_length"]:
                if metric in d:
                    print(f"    {metric}: {d[metric].mean():.3f} ± {d[metric].std():.3f}")

    # ── UMAP ───────────────────────────────────────────────────────────────
    print("\nRunning UMAP...")
    meta_cols = {"subject", "session", group_col, sex_col} - {None}
    umap_df = run_umap(results_df, meta_cols, umap_params={})
    umap_df.to_parquet(output_dir / "umap_coordinates.parquet", index=False)
    print("✓ Saved: umap_coordinates.parquet")

    # ── Trajectories ───────────────────────────────────────────────────────
    print("\nComputing trajectories...")
    trajectory_df = compute_trajectories(umap_df, "subject", "session", group_col, sex_col)
    if not trajectory_df.empty:
        trajectory_df.to_parquet(output_dir / "trajectory_analysis.parquet", index=False)
        print(f"✓ Saved: trajectory_analysis.parquet ({len(trajectory_df)} subjects)")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nCreating plots...")
    make_plots(results_df, umap_df, trajectory_df, output_dir / "plots", group_col, "subject")

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