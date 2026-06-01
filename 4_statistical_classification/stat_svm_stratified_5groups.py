#!/usr/bin/env python3
"""
SCRIPT: Stratified Time-Effect SVM Analysis
============================================

PURPOSE:
    Separate SVM models for each sex and age group to examine
    if group classification differs by demographic factors.
    Groups, columns and atlas are auto-detected from the data.

USAGE:
    python stat_svm_stratified.py run_spec.json
    python stat_svm_stratified.py --metrics-file /path/to/metrics.parquet --output-dir /path/to/out

AUTHOR: Analysis Pipeline
VERSION: 1.0 (Auto-detection, run_spec driven)
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
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ============================================================================
# CLI / run_spec
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified SVM analysis.")
    parser.add_argument("run_spec",        nargs="?", help="Path to run_spec.json")
    parser.add_argument("--metrics-file",  help="Metrics .parquet file")
    parser.add_argument("--metadata",      help="Participant metadata TSV/CSV")
    parser.add_argument("--output-dir",    help="Output directory")
    parser.add_argument("--age-threshold", type=float, default=40.0,
                        help="Age split threshold (default: 40)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    config: dict = {}
    if args.run_spec:
        spec_path = Path(args.run_spec).expanduser().resolve()
        if not spec_path.exists():
            sys.exit(f"x run_spec not found: {spec_path}")
        with open(spec_path) as f:
            spec = json.load(f)
        inputs  = spec.get("inputs", {})
        outputs = spec.get("outputs", {})
        config["metrics_file"]   = inputs.get("metrics_file")
        config["metadata_file"]  = inputs.get("metadata_file")
        config["output_dir"]     = outputs.get("output_dir")
        config["age_threshold"]  = spec.get("age_threshold", 40.0)

    if args.metrics_file: config["metrics_file"]  = args.metrics_file
    if args.metadata:     config["metadata_file"] = args.metadata
    if args.output_dir:   config["output_dir"]    = args.output_dir
    config.setdefault("age_threshold", args.age_threshold)

    missing = [k for k in ("metrics_file", "output_dir") if not config.get(k)]
    if missing:
        sys.exit(f"x Missing: {', '.join(missing)}")

    config["metrics_file"] = Path(config["metrics_file"]).expanduser().resolve()
    config["output_dir"]   = Path(config["output_dir"]).expanduser().resolve()
    if config.get("metadata_file"):
        config["metadata_file"] = Path(config["metadata_file"]).expanduser().resolve()
    return config


# ============================================================================
# AUTO-DETECTION
# ============================================================================

def detect_columns(df: pd.DataFrame) -> dict:
    cols_lower = {c.lower(): c for c in df.columns}

    def find(candidates):
        for c in candidates:
            if c in cols_lower: return cols_lower[c]
        for c in candidates:
            for cl, co in cols_lower.items():
                if c in cl: return co
        return None

    exclude = {"subject", "participant_id", "session", "ses", "session_id",
               "node", "group", "condition", "sex", "gender", "age", "atlas",
               "hub_type", "community", "nr_sessions", "n_sessions",
               "dsi_metric", "matrix_type", "qc_passed", "qc_warnings"}

    detected = {
        "subject_col": find(["participant_id", "subject_id", "subject"]),
        "session_col": find(["session_id", "session", "ses", "timepoint"]),
        "group_col":   find(["group", "condition", "arm", "intervention"]),
        "sex_col":     find(["sex", "gender"]),
        "age_col":     find(["age"]),
        "metric_cols": [c for c in df.columns
                        if c.lower() not in exclude
                        and pd.api.types.is_numeric_dtype(df[c])
                        and c.lower() != "node"],
    }
    print("  Auto-detected columns:")
    for k, v in detected.items():
        if k != "metric_cols":
            print(f"    {k}: {v or '⚠ not found'}")
        else:
            print(f"    metric_cols: {len(v)} columns")
    return detected


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_merge(metrics_file: Path, metadata_file: Path | None,
                   cols: dict) -> pd.DataFrame:
    df = pd.read_parquet(metrics_file)
    print(f"  + Metrics: {len(df)} rows")

    if metadata_file and metadata_file.exists():
        sep  = "\t" if metadata_file.suffix == ".tsv" else ","
        meta = pd.read_csv(metadata_file, sep=sep)
        subj = cols["subject_col"]
        ses  = cols["session_col"]
        keys = [k for k in [subj, ses] if k and k in meta.columns and k in df.columns]
        if keys:
            df = df.merge(meta, on=keys, how="left", suffixes=("", "_meta"))
            print(f"  + Merged metadata on: {keys}")
    return df


def aggregate_nodal(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    if "node" not in df.columns:
        return df
    print("  + Nodal data → aggregating to subject-session level")
    keep = [c for c in [cols["subject_col"], cols["session_col"],
                         cols["group_col"], cols["sex_col"], cols["age_col"]]
            if c and c in df.columns]
    avail = [c for c in cols["metric_cols"] if c in df.columns]
    return df.groupby(keep)[avail].mean().reset_index()


def calculate_slopes(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    subj_col    = cols["subject_col"]
    session_col = cols["session_col"]
    group_col   = cols["group_col"]
    sex_col     = cols["sex_col"]
    age_col     = cols["age_col"]
    metric_cols = [c for c in cols["metric_cols"] if c in df.columns]

    records = []
    _subjects = df[subj_col].unique()
    _total = len(_subjects)
    for _i, (subj, sdata) in enumerate(df.groupby(subj_col)):
        if _i % 5 == 0 or _i == _total - 1:
            print(f"  Calculating slopes: {_i+1}/{_total} ({(_i+1)/_total*100:.0f}%)", end="
")
        sdata = sdata.sort_values(session_col)
        if len(sdata) < 2:
            continue
        sessions = sdata[session_col].values
        try:
            sessions = pd.to_numeric(
                pd.Series(sessions).str.extract(r"(\d+)")[0].values)
        except Exception:
            sessions = np.arange(len(sessions), dtype=float)

        rec = {subj_col: subj}
        if group_col and group_col in sdata.columns:
            rec[group_col] = sdata[group_col].iloc[0]
        if sex_col and sex_col in sdata.columns:
            rec[sex_col] = sdata[sex_col].iloc[0]
        if age_col and age_col in sdata.columns:
            rec[age_col] = float(sdata[age_col].iloc[0])

        for metric in metric_cols:
            vals = pd.to_numeric(sdata[metric], errors="coerce").values
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                try:
                    rec[f"{metric}_slope"] = np.polyfit(
                        sessions[valid].astype(float), vals[valid], 1)[0]
                except Exception:
                    pass
        records.append(rec)

    return pd.DataFrame(records)


def encode_sex(df: pd.DataFrame, sex_col: str) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df["sex_encoded"] = le.fit_transform(df[sex_col].fillna("M").astype(str))
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  Sex encoding: {mapping}")
    return df


def build_features(df: pd.DataFrame, group_col: str,
                   age_col: str | None, extra_cols: list) -> tuple:
    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    feat_cols  = slope_cols + [c for c in extra_cols if c in df.columns]
    X = df[feat_cols].fillna(0)
    y = df[group_col].fillna(np.nan)
    valid = ~y.isna()
    return X[valid], y[valid], feat_cols


# ============================================================================
# SVM TRAINING
# ============================================================================

def train_svm(X: pd.DataFrame, y: pd.Series, label: str) -> dict | None:
    if len(y) < 5 or len(y.unique()) < 2:
        print(f"  ⚠ {label}: not enough data — skipping")
        return None

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm      = SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", probability=True, random_state=42)
    n_folds  = min(5, len(y.unique()))
    cv       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(svm, X_scaled, y, cv=cv)
    svm.fit(X_scaled, y)
    y_pred   = svm.predict(X_scaled)

    print(f"  CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}  Train: {(y_pred==y).mean():.3f}")
    print(f"  {classification_report(y, y_pred, zero_division=0)}")

    return {
        "cv_mean": float(cv_scores.mean()),
        "cv_std":  float(cv_scores.std()),
        "train_acc": float((y_pred == y).mean()),
        "cv_scores": cv_scores.tolist(),
        "n_samples": int(len(y)),
    }


def feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        groups = [X[col][y == g].values for g in y.unique()]
        try:
            f, p = stats.f_oneway(*groups)
        except Exception:
            f, p = np.nan, np.nan
        rows.append({"feature": col, "f_score": f, "p_value": p})
    return pd.DataFrame(rows).sort_values("f_score", ascending=False)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args   = parse_args()
    config = load_config(args)

    metrics_file  = config["metrics_file"]
    metadata_file = config.get("metadata_file")
    output_dir    = config["output_dir"]
    age_threshold = config["age_threshold"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print(f"  Started:  {_start_time.strftime(\'%Y-%m-%d %H:%M:%S\')}")
    print("STRATIFIED SVM ANALYSIS")
    print("=" * 70)

    print("\nLoading data...")
    raw_df = pd.read_parquet(metrics_file)
    cols   = detect_columns(raw_df)
    df     = load_and_merge(metrics_file, metadata_file, cols)
    cols   = detect_columns(df)
    df     = aggregate_nodal(df, cols)
    cols   = detect_columns(df)

    group_col = cols["group_col"]
    sex_col   = cols["sex_col"]
    age_col   = cols["age_col"]
    subj_col  = cols["subject_col"]

    if not group_col:
        sys.exit("x Could not detect group column.")

    print("\nCalculating slopes...")
    slopes_df = calculate_slopes(df, cols)
    if len(slopes_df) == 0:
        sys.exit("x No slopes calculated — subjects need ≥2 sessions.")

    if sex_col and sex_col in slopes_df.columns:
        slopes_df = encode_sex(slopes_df, sex_col)

    extra = ["sex_encoded"] + ([age_col] if age_col and age_col in slopes_df.columns else [])

    summary = {}

    # ── By sex ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STRATIFIED BY SEX")
    print("=" * 70)
    sex_results = {}
    if sex_col and sex_col in slopes_df.columns:
        for sex_val in slopes_df[sex_col].dropna().unique():
            label = str(sex_val)
            sub   = slopes_df[slopes_df[sex_col] == sex_val]
            print(f"\n{label} (n={len(sub)}):")
            X, y, feat_cols = build_features(sub, group_col, age_col, extra)
            res = train_svm(X, y, label)
            if res:
                sex_results[label] = res
                fi = feature_importance(X, y)
                fi.to_csv(output_dir / f"feature_importance_sex_{label}.csv", index=False)
    else:
        print("  ⚠ No sex column detected — skipping sex stratification")
    summary["by_sex"] = sex_results

    # ── By age ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STRATIFIED BY AGE (threshold: {age_threshold})")
    print("=" * 70)
    age_results = {}
    if age_col and age_col in slopes_df.columns:
        slopes_df["_age_group"] = slopes_df[age_col].apply(
            lambda a: f"Young (<{age_threshold:.0f})"
            if pd.notna(a) and float(a) < age_threshold
            else f"Older (≥{age_threshold:.0f})")
        for ag in slopes_df["_age_group"].unique():
            sub = slopes_df[slopes_df["_age_group"] == ag]
            print(f"\n{ag} (n={len(sub)}):")
            X, y, feat_cols = build_features(sub, group_col, age_col, extra)
            res = train_svm(X, y, ag)
            if res:
                age_results[ag] = res
                fi = feature_importance(X, y)
                fi.to_csv(output_dir / f"feature_importance_age_{ag.replace(' ','_')}.csv", index=False)
    else:
        print("  ⚠ No age column detected — skipping age stratification")
    summary["by_age"] = age_results

    # ── Save ──────────────────────────────────────────────────────────────
    slopes_df.to_csv(output_dir / "slopes_stratified.csv", index=False)
    with open(output_dir / "stratified_svm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _end_time = datetime.now()
    _duration = _end_time - _start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Started:  {_start_time.strftime(\'%Y-%m-%d %H:%M:%S\')}")
    print(f"  Finished: {_end_time.strftime(\'%Y-%m-%d %H:%M:%S\')}")
    print(f"  Duration: {str(_duration).split(\'.\')[0]}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()