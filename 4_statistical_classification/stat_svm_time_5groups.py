#!/usr/bin/env python3
"""
SCRIPT: SVM + Random Forest Time-Effect Analysis (5 Groups)
============================================================

PURPOSE:
    Calculates temporal slopes per subject, then classifies groups using
    both Random Forest and SVM. Groups and columns are auto-detected.

USAGE:
    python stat_svm_time_5groups.py run_spec.json
    python stat_svm_time_5groups.py --metrics-file /path/to/metrics.parquet --output-dir /path/to/out

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ============================================================================
# CLI / run_spec
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
    parser = argparse.ArgumentParser(description="SVM+RF time-effect analysis.")
    parser.add_argument("run_spec",       nargs="?", help="Path to run_spec.json")
    parser.add_argument("--metrics-file", help="Metrics .parquet file")
    parser.add_argument("--metadata",     help="Participant metadata TSV/CSV")
    parser.add_argument("--output-dir",   help="Output directory")
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
        config["metrics_file"]  = inputs.get("metrics_file")
        config["metadata_file"] = inputs.get("metadata_file")
        config["output_dir"]    = outputs.get("output_dir")

    if args.metrics_file: config["metrics_file"]  = args.metrics_file
    if args.metadata:     config["metadata_file"] = args.metadata
    if args.output_dir:   config["output_dir"]    = args.output_dir

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
    if metadata_file and metadata_file.exists():
        sep  = "\t" if metadata_file.suffix == ".tsv" else ","
        meta = pd.read_csv(metadata_file, sep=sep)
        keys = [k for k in [cols["subject_col"], cols["session_col"]]
                if k and k in meta.columns and k in df.columns]
        if keys:
            df = df.merge(meta, on=keys, how="left", suffixes=("", "_meta"))
    return df


def aggregate_nodal(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    if "node" not in df.columns:
        return df
    print("  + Nodal → subject-session aggregation")
    keep  = [c for c in [cols["subject_col"], cols["session_col"],
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
    _subjects_list = list(df.groupby(subj_col))
    _total_s = len(_subjects_list)
    for _i_s, (subj, sdata) in enumerate(_subjects_list):
        _progress(_i_s + 1, _total_s, f"Slopes: {subj} ({sdata[group_col].iloc[0] if group_col and group_col in sdata.columns else ''})")
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
        for c in [group_col, sex_col, age_col]:
            if c and c in sdata.columns:
                rec[c] = sdata[c].iloc[0]
        rec["n_sessions"] = len(sdata)

        for metric in metric_cols:
            vals  = pd.to_numeric(sdata[metric], errors="coerce").values
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                try:
                    rec[f"{metric}_slope"] = np.polyfit(
                        sessions[valid].astype(float), vals[valid], 1)[0]
                except Exception:
                    pass
        records.append(rec)

    df_out = pd.DataFrame(records)
    print(f"  + Slopes for {len(df_out)} subjects")
    return df_out


def build_features(slopes_df: pd.DataFrame, group_col: str,
                   sex_col: str | None, age_col: str | None) -> tuple:
    df = slopes_df.copy()

    # Encode sex
    if sex_col and sex_col in df.columns:
        le = LabelEncoder()
        df["sex_encoded"] = le.fit_transform(df[sex_col].fillna("M").astype(str))

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    extra      = [c for c in ["sex_encoded", "n_sessions"] +
                  ([age_col] if age_col and age_col in df.columns else [])
                  if c in df.columns]
    feat_cols  = slope_cols + extra

    X     = df[feat_cols].fillna(0)
    y_raw = df[group_col]
    valid = ~y_raw.isna()
    return X[valid], y_raw[valid], feat_cols


# ============================================================================
# CLASSIFIERS
# ============================================================================

def train_random_forest(X: pd.DataFrame, y: pd.Series) -> dict:
    rf = RandomForestClassifier(n_estimators=500, max_depth=10,
                                min_samples_split=5, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, X, y, cv=min(5, len(y.unique())))
    rf.fit(X, y)
    y_pred = rf.predict(X)

    print(f"\nRandom Forest:")
    print(f"  CV: {cv.mean():.3f}±{cv.std():.3f}  Train: {(y_pred==y).mean():.3f}")
    print(classification_report(y, y_pred, zero_division=0))

    fi = pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)

    return {
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        "train_acc": float((y_pred == y).mean()),
        "feature_importance": fi,
    }


def train_svm(X: pd.DataFrame, y: pd.Series) -> dict:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm      = SVC(kernel="rbf", C=1.0, gamma="scale",
                   class_weight="balanced", probability=True, random_state=42)
    cv = cross_val_score(svm, X_scaled, y, cv=min(5, len(y.unique())))
    svm.fit(X_scaled, y)
    y_pred = svm.predict(X_scaled)

    print(f"\nSVM:")
    print(f"  CV: {cv.mean():.3f}±{cv.std():.3f}  Train: {(y_pred==y).mean():.3f}")
    print(classification_report(y, y_pred, zero_division=0))

    return {
        "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
        "train_acc": float((y_pred == y).mean()),
    }


def feature_importance_anova(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
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
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    _start_time = datetime.now()
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("SVM + RANDOM FOREST TIME-EFFECT ANALYSIS")
    print("=" * 70)

    print("\nLoading data...")
    raw_df = pd.read_parquet(metrics_file)
    cols   = detect_columns(raw_df)
    df     = load_and_merge(metrics_file, metadata_file, cols)
    cols   = detect_columns(df)
    df     = aggregate_nodal(df, cols)
    cols   = detect_columns(df)

    group_col = cols["group_col"]
    if not group_col:
        sys.exit("x Could not detect group column.")

    print("\nCalculating slopes...")
    slopes_df = calculate_slopes(df, cols)
    if len(slopes_df) == 0:
        sys.exit("x No slopes — subjects need ≥2 sessions.")

    X, y, feat_cols = build_features(slopes_df, group_col,
                                      cols["sex_col"], cols["age_col"])

    if len(X) == 0 or len(y.unique()) < 2:
        sys.exit("x Not enough data for classification.")

    print(f"\nSamples: {len(X)}, Features: {len(feat_cols)}, Groups: {list(y.unique())}")

    # Train
    print("\n" + "=" * 70)
    rf_results  = train_random_forest(X, y)
    svm_results = train_svm(X, y)

    # Feature importance
    anova_df = feature_importance_anova(X, y)
    print("\nTop 10 ANOVA Features:")
    print(anova_df.head(10).to_string(index=False))

    # Compare
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"  Random Forest: CV={rf_results['cv_mean']:.3f}±{rf_results['cv_std']:.3f}")
    print(f"  SVM:           CV={svm_results['cv_mean']:.3f}±{svm_results['cv_std']:.3f}")
    best = "Random Forest" if rf_results["cv_mean"] >= svm_results["cv_mean"] else "SVM"
    print(f"  Best model: {best}")

    # Save
    slopes_df.to_csv(output_dir / "time_effect_slopes.csv", index=False)
    anova_df.to_csv(output_dir / "feature_importance_anova.csv", index=False)
    rf_results["feature_importance"].to_csv(
        output_dir / "feature_importance_rf.csv", index=False)

    summary = {
        "best_model": best,
        "random_forest": {k: v for k, v in rf_results.items() if k != "feature_importance"},
        "svm": svm_results,
        "n_samples": int(len(X)),
        "n_features": int(len(feat_cols)),
    }
    with open(output_dir / "svm_rf_time_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n+ Saved outputs to: {output_dir}")
    _end_time = datetime.now()
    _duration = _end_time - _start_time
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("  Started:  " + _start_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Finished: " + _end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("  Duration: " + str(_duration).split(".")[0])


if __name__ == "__main__":
    main()