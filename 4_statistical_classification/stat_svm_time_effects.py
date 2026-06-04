#!/usr/bin/env python3
"""
SCRIPT: SVM Analysis – Time Effects (Slopes across Sessions)
=============================================================

PURPOSE:
    Calculates per-subject temporal slopes across sessions for all
    connectivity metrics, then trains an SVM to classify groups based
    on those slopes.  Age, sex, and n_sessions are included as
    covariates automatically when detected.

USAGE:
    python stat_svm_time_effects.py \
        --metrics-file /path/to/metrics.parquet \
        --output-dir   /path/to/output

    python stat_svm_time_effects.py \
        --metrics-file /path/to/metrics.parquet \
        --metadata     /path/to/participants.tsv \
        --output-dir   /path/to/output

REQUIRED:
    --metrics-file    Path to metrics .parquet file
    --output-dir      Directory where results are saved

OPTIONAL (all auto-detected when omitted):
    --metadata        Participant metadata file (CSV or TSV)
    --subject-col     Column name for subject IDs
    --session-col     Column name for session IDs
    --group-col       Column name for group labels
    --age-col         Column name for age
    --sex-col         Column name for sex / gender

AUTHOR: Analysis Pipeline
VERSION: 2.0 (CLI-driven, auto-detection)
"""

from __future__ import annotations
from datetime import datetime

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ============================================================================
# CLI
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVM time-effect analysis — all inputs via CLI flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metrics-file", required=True,
                        help="Path to metrics .parquet file")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory where results are saved")
    parser.add_argument("--metadata",     default=None,
                        help="Participant metadata file (CSV or TSV)")
    parser.add_argument("--subject-col",  default=None,
                        help="Subject ID column (auto-detected if omitted)")
    parser.add_argument("--session-col",  default=None,
                        help="Session column (auto-detected if omitted)")
    parser.add_argument("--group-col",    default=None,
                        help="Group column (auto-detected if omitted)")
    parser.add_argument("--age-col",      default=None,
                        help="Age column (auto-detected if omitted)")
    parser.add_argument("--sex-col",      default=None,
                        help="Sex / gender column (auto-detected if omitted)")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Validate paths and return config dict."""
    metrics_file = Path(args.metrics_file).expanduser().resolve()
    output_dir   = Path(args.output_dir).expanduser().resolve()

    if not metrics_file.exists():
        sys.exit(f"x Metrics file not found: {metrics_file}")

    return {
        "metrics_file":  metrics_file,
        "output_dir":    output_dir,
        "metadata_file": Path(args.metadata).expanduser().resolve()
                         if args.metadata else None,
        "subject_col":   args.subject_col,
        "session_col":   args.session_col,
        "group_col":     args.group_col,
        "age_col":       args.age_col,
        "sex_col":       args.sex_col,
    }


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_columns(df: pd.DataFrame, config: dict) -> dict:
    """Auto-detect subject, session, group, age, sex and metric columns."""
    cols_lower = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        for c in candidates:
            for cl, co in cols_lower.items():
                if c in cl:
                    return co
        return None

    exclude = {
        "subject", "participant_id", "session", "ses", "session_id", "session_num",
        "node", "group", "condition", "arm", "intervention",
        "sex", "gender", "age", "atlas", "hub_type", "community",
        "nr_sessions", "n_sessions", "dsi_metric", "matrix_type",
        "qc_passed", "qc_warnings",
    }

    detected = {
        "subject_col": config.get("subject_col") or find(
            ["participant_id", "subject_id", "subject", "participant"]),
        "session_col": config.get("session_col") or find(
            ["session_num", "session_id", "session", "ses", "timepoint", "visit"]),
        "group_col":   config.get("group_col") or find(
            ["group", "condition", "arm", "intervention"]),
        "age_col":     config.get("age_col") or find(["age"]),
        "sex_col":     config.get("sex_col") or find(["sex", "gender"]),
        "metric_cols": [
            c for c in df.columns
            if c.lower() not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and c.lower() not in ("node", "session_num")
        ],
    }

    print("  Auto-detected columns:")
    for k, v in detected.items():
        if k != "metric_cols":
            print(f"    {k}: {v if v else 'not found'}")
        else:
            print(f"    metric_cols: {len(v)} columns")
    return detected


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(metrics_file: Path, metadata_file, cols: dict) -> pd.DataFrame:
    df = pd.read_parquet(metrics_file)
    print(f"  + Loaded metrics: {len(df)} rows, {len(df.columns)} columns")

    if metadata_file and metadata_file.exists():
        sep  = "\t" if metadata_file.suffix == ".tsv" else ","
        meta = pd.read_csv(metadata_file, sep=sep)
        print(f"  + Loaded metadata: {len(meta)} rows")
        subj_col = cols["subject_col"]
        ses_col  = cols["session_col"]
        merge_keys = [k for k in [subj_col, ses_col]
                      if k and k in meta.columns and k in df.columns]
        if merge_keys:
            df = df.merge(meta, on=merge_keys, how="left", suffixes=("", "_meta"))
            print(f"  + Merged on: {merge_keys}")
        else:
            print("  ! Could not merge metadata — no common columns found")
    elif metadata_file:
        print(f"  ! Metadata file not found: {metadata_file}")
    return df


def aggregate_nodal(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    if "node" not in df.columns:
        return df
    print("  + Nodal data detected — aggregating to subject-session level")
    group_keys = [c for c in [cols["subject_col"], cols["session_col"],
                               cols["group_col"], cols["sex_col"], cols["age_col"]]
                  if c and c in df.columns]
    available = [c for c in cols["metric_cols"] if c in df.columns]
    agg_df    = df.groupby(group_keys)[available].mean().reset_index()
    print(f"  + Aggregated to {len(agg_df)} subject-session rows")
    return agg_df


# ============================================================================
# SESSION NUMBER EXTRACTION
# ============================================================================

def extract_session_number(session_values: pd.Series) -> np.ndarray:
    try:
        numeric = pd.to_numeric(
            session_values.astype(str).str.extract(r"(\d+)")[0]
        )
        if numeric.notna().all():
            return numeric.values.astype(float)
    except Exception:
        pass
    return np.arange(len(session_values), dtype=float)


# ============================================================================
# SLOPE CALCULATION
# ============================================================================

def calculate_slopes(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    subject_col = cols["subject_col"]
    session_col = cols["session_col"]
    group_col   = cols["group_col"]
    age_col     = cols["age_col"]
    sex_col     = cols["sex_col"]
    metric_cols = [c for c in cols["metric_cols"] if c in df.columns]

    if not subject_col or subject_col not in df.columns:
        sys.exit(f"x Subject column '{subject_col}' not found in data.")

    records = []
    _subjects_list = list(df.groupby(subject_col))
    _total_s = len(_subjects_list)
    for _i_s, (subject, subj_data) in enumerate(_subjects_list):
        _g = subj_data[group_col].iloc[0] if group_col and group_col in subj_data.columns else ""
        _progress(_i_s + 1, _total_s, "Calculating slopes")
        print(f"    {subject} [{_g}] fitting slopes across {len(subj_data)} sessions...  ", end="\n", flush=True)
        subj_data = subj_data.sort_values(session_col) if session_col else subj_data
        if len(subj_data) < 2:
            continue

        sessions = (extract_session_number(subj_data[session_col])
                    if session_col and session_col in subj_data.columns
                    else np.arange(len(subj_data), dtype=float))

        rec = {subject_col: subject, "n_sessions": len(subj_data)}
        if group_col and group_col in subj_data.columns:
            rec[group_col] = subj_data[group_col].iloc[0]
        if age_col and age_col in subj_data.columns:
            rec[age_col] = subj_data[age_col].iloc[0]
        if sex_col and sex_col in subj_data.columns:
            rec[sex_col] = subj_data[sex_col].iloc[0]

        for metric in metric_cols:
            values = pd.to_numeric(subj_data[metric], errors="coerce").values
            valid  = ~np.isnan(values)
            if valid.sum() >= 2:
                try:
                    slope = np.polyfit(sessions[valid], values[valid], 1)[0]
                    rec[f"{metric}_slope"] = slope
                except Exception:
                    pass
        records.append(rec)

    slopes_df = pd.DataFrame(records)
    print(f"  + Slopes calculated for {len(slopes_df)} subjects")
    return slopes_df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def encode_sex(slopes_df: pd.DataFrame, sex_col) -> pd.DataFrame:
    slopes_df = slopes_df.copy()
    if sex_col and sex_col in slopes_df.columns:
        known_map = {"M": 1, "F": 0, "m": 1, "f": 0, "male": 1, "female": 0, 1: 1, 2: 0}
        mapped = slopes_df[sex_col].map(known_map)
        if mapped.isna().any():
            le = LabelEncoder()
            mapped = le.fit_transform(slopes_df[sex_col].astype(str))
            print(f"  + Sex encoding (LabelEncoder): "
                  f"{dict(zip(le.classes_, le.transform(le.classes_)))}")
        else:
            print("  + Sex encoding: M→1, F→0")
        slopes_df["sex_encoded"] = mapped
    else:
        slopes_df["sex_encoded"] = 0
    return slopes_df


def prepare_features(slopes_df: pd.DataFrame, cols: dict) -> tuple:
    group_col = cols["group_col"]
    age_col   = cols["age_col"]

    slope_cols    = [c for c in slopes_df.columns if c.endswith("_slope")]
    covariate_cols = ["sex_encoded", "n_sessions"]
    if age_col and age_col in slopes_df.columns:
        covariate_cols.append(age_col)
    feature_cols = slope_cols + [c for c in covariate_cols if c in slopes_df.columns]

    if not group_col or group_col not in slopes_df.columns:
        sys.exit(f"x Group column '{group_col}' not found.")

    slopes_df = slopes_df[slopes_df[group_col].notna()].copy()
    X     = slopes_df[feature_cols].fillna(0).values.astype(float)
    y_raw = slopes_df[group_col].values
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw)

    print(f"  Feature matrix : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Group encoding : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"  Class counts   : {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, feature_cols, le, slopes_df


# ============================================================================
# SVM TRAINING
# ============================================================================

def train_svm(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder) -> dict:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_folds = min(5, int(np.min(np.bincount(y))))
    if cv_folds < 2:
        print("  ! Not enough samples per class for CV — skipping")
        cv_scores = np.array([np.nan])
    else:
        svm_cv    = SVC(kernel="rbf", C=1.0, gamma="scale",
                        class_weight="balanced", random_state=42)
        cv_scores = cross_val_score(svm_cv, X_scaled, y,
                                    cv=StratifiedKFold(cv_folds), scoring="accuracy")

    svm = SVC(kernel="rbf", C=1.0, gamma="scale",
              class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_scaled, y)
    y_pred       = svm.predict(X_scaled)
    target_names = [str(c) for c in label_encoder.classes_]

    print(f"\n  CV Accuracy : {np.nanmean(cv_scores):.3f} ± {np.nanstd(cv_scores):.3f}")
    print(f"  Train Acc   : {(y_pred == y).mean():.3f}")
    print("\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=target_names, digits=3))
    print("  Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return {
        "svm": svm, "scaler": scaler, "X_scaled": X_scaled, "y_pred": y_pred,
        "cv_scores": cv_scores,
        "cv_mean":   float(np.nanmean(cv_scores)),
        "cv_std":    float(np.nanstd(cv_scores)),
        "train_acc": float((y_pred == y).mean()),
    }


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(X: np.ndarray, y: np.ndarray,
                                feature_cols: list) -> pd.DataFrame:
    records = []
    for i, feat in enumerate(feature_cols):
        groups = [X[y == g, i] for g in np.unique(y)]
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_val = np.nan, np.nan
        records.append({"feature": feat, "f_score": f_stat, "p_value": p_val})

    importance_df = (pd.DataFrame(records)
                     .sort_values("f_score", ascending=False)
                     .reset_index(drop=True))

    print("\n  Top 15 Discriminative Features:")
    print(importance_df.head(15).to_string(index=False))
    slope_imp = importance_df[importance_df["feature"].str.endswith("_slope")]
    print("\n  Top 10 Time-Dependent Metrics:")
    print(slope_imp.head(10).to_string(index=False))
    return importance_df


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(output_dir: Path, slopes_df: pd.DataFrame, svm_result: dict,
                 y: np.ndarray, feature_cols: list, importance_df: pd.DataFrame,
                 label_encoder: LabelEncoder) -> None:
    svm      = svm_result["svm"]
    X_scaled = svm_result["X_scaled"]
    y_pred   = svm_result["y_pred"]

    summary = {
        "analysis_type":     "SVM Time Effects (Slopes across Sessions)",
        "model":             "SVM RBF kernel",
        "cv_accuracy_mean":  svm_result["cv_mean"],
        "cv_accuracy_std":   svm_result["cv_std"],
        "training_accuracy": svm_result["train_acc"],
        "n_support_vectors": int(len(svm.support_vectors_)),
        "n_samples":         int(len(y)),
        "n_features":        int(X_scaled.shape[1]),
        "groups":            label_encoder.classes_.tolist(),
        "features":          feature_cols,
        "top_features":      importance_df.head(10)["feature"].tolist(),
        "top_f_scores":      importance_df.head(10)["f_score"].tolist(),
    }

    with open(output_dir / "svm_time_effects_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    slopes_df.to_csv(output_dir / "time_effect_slopes.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance_time_effects.csv", index=False)

    print(f"  + svm_time_effects_summary.json")
    print(f"  + time_effect_slopes.csv")
    print(f"  + feature_importance_time_effects.csv")
    print(f"  → {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = build_config(args)

    metrics_file  = config["metrics_file"]
    metadata_file = config["metadata_file"]
    output_dir    = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("SVM TIME EFFECTS ANALYSIS")
    print("=" * 70)
    print(f"  Metrics file : {metrics_file}")
    if metadata_file:
        print(f"  Metadata     : {metadata_file}")
    print(f"  Output dir   : {output_dir}")
    print()

    # Load
    print("Loading data...")
    raw_df = pd.read_parquet(metrics_file)
    print("\nAuto-detecting columns...")
    cols = detect_columns(raw_df, config)

    if metadata_file:
        df   = load_data(metrics_file, metadata_file, cols)
        cols = detect_columns(df, config)
    else:
        df = raw_df

    df   = aggregate_nodal(df, cols)
    cols = detect_columns(df, config)

    if not cols["group_col"]:
        sys.exit(
            "x Could not detect group column.\n"
            f"  Available columns: {list(df.columns)}\n"
            "  Use --group-col to specify it explicitly."
        )

    # Slopes
    print("\nCalculating temporal slopes...")
    slopes_df = calculate_slopes(df, cols)
    if len(slopes_df) == 0:
        sys.exit("x No slopes calculated — ensure subjects have >= 2 sessions")

    # Covariates
    print("\nEncoding covariates...")
    slopes_df = encode_sex(slopes_df, cols["sex_col"])

    # Features
    print("\nPreparing feature matrix...")
    X, y, feature_cols, label_encoder, slopes_df = prepare_features(slopes_df, cols)
    if len(np.unique(y)) < 2:
        sys.exit("x Need at least 2 groups for classification")

    # SVM
    print("\n" + "=" * 70)
    print("TRAINING SVM")
    print("=" * 70)
    svm_result = train_svm(X, y, label_encoder)

    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    importance_df = analyze_feature_importance(X, y, feature_cols)

    # Save
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(output_dir, slopes_df, svm_result, y,
                 feature_cols, importance_df, label_encoder)

    print("\n" + "=" * 70)
    print("DONE — SVM Time Effects Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()