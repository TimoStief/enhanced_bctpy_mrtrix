"""
Tests für stat_svm_stratified.py
"""
import json
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
mod = importlib.import_module("stat_svm_stratified_5groups")

load_config         = mod.load_config
detect_columns      = mod.detect_columns
load_and_merge      = mod.load_and_merge
aggregate_nodal     = mod.aggregate_nodal
calculate_slopes    = mod.calculate_slopes
encode_sex          = mod.encode_sex
build_features      = mod.build_features
train_svm           = mod.train_svm
feature_importance  = mod.feature_importance
main                = mod.main


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _make_metrics_df(n_subjects=6, n_sessions=3):
    records = []
    groups = ["alone_2w", "alone_4w", "group_2w", "group_4w", "control", "alone_2w"]
    sexes  = ["M", "F", "M", "F", "M", "F"]
    for i, (g, s) in enumerate(zip(groups[:n_subjects], sexes[:n_subjects])):
        for ses in range(1, n_sessions + 1):
            records.append({
                "subject": f"S{i+1}",
                "session": ses,
                "group":   g,
                "sex":     s,
                "age":     float(25 + i * 3),
                "degree":  float(i + ses * 0.1),
                "strength": float(i * 0.5 + ses * 0.05),
            })
    return pd.DataFrame(records)


def _make_run_spec(tmp_path):
    df      = _make_metrics_df()
    node_dir = tmp_path / "metrics"
    node_dir.mkdir()
    df.to_parquet(node_dir / "metrics.parquet", index=False)
    out_dir = tmp_path / "output"
    spec = {
        "inputs":  {"metrics_file": str(node_dir / "metrics.parquet")},
        "outputs": {"output_dir": str(out_dir)},
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    return spec_file, node_dir / "metrics.parquet", out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# load_config
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_config_from_run_spec(tmp_path):
    spec_file, metrics_file, out_dir = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(spec_file)]):
        args   = mod.parse_args()
        config = load_config(args)
    assert config["metrics_file"] == metrics_file
    assert config["output_dir"]   == out_dir


def test_load_config_missing_exits(tmp_path):
    with patch("sys.argv", ["script"]):
        args = mod.parse_args()
        with pytest.raises(SystemExit):
            load_config(args)


def test_load_config_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        args = mod.parse_args()
        with pytest.raises(SystemExit):
            load_config(args)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_columns
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_columns_standard():
    df = _make_metrics_df()
    r  = detect_columns(df)
    assert r["subject_col"] == "subject"
    assert r["session_col"] == "session"
    assert r["group_col"]   == "group"
    assert r["sex_col"]     == "sex"
    assert r["age_col"]     == "age"


def test_detect_columns_metric_cols():
    df = _make_metrics_df()
    r  = detect_columns(df)
    assert "degree"   in r["metric_cols"]
    assert "strength" in r["metric_cols"]
    assert "group"    not in r["metric_cols"]


def test_detect_columns_no_age():
    df = _make_metrics_df().drop(columns=["age"])
    r  = detect_columns(df)
    assert r["age_col"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# aggregate_nodal
# ═══════════════════════════════════════════════════════════════════════════════

def test_aggregate_nodal_with_node_col():
    df = _make_metrics_df()
    df["node"] = [1, 2] * (len(df) // 2) + [1] * (len(df) % 2)
    cols = detect_columns(df)
    result = aggregate_nodal(df, cols)
    assert "node" not in result.columns or len(result) < len(df)


def test_aggregate_nodal_without_node_col():
    df   = _make_metrics_df()
    cols = detect_columns(df)
    result = aggregate_nodal(df, cols)
    assert len(result) == len(df)


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_slopes
# ═══════════════════════════════════════════════════════════════════════════════

def test_calculate_slopes_returns_df():
    df   = _make_metrics_df()
    cols = detect_columns(df)
    result = calculate_slopes(df, cols)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_calculate_slopes_has_slope_cols():
    df   = _make_metrics_df()
    cols = detect_columns(df)
    result = calculate_slopes(df, cols)
    slope_cols = [c for c in result.columns if c.endswith("_slope")]
    assert len(slope_cols) > 0


def test_calculate_slopes_skips_single_session():
    df   = _make_metrics_df(n_sessions=1)
    cols = detect_columns(df)
    result = calculate_slopes(df, cols)
    assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# encode_sex
# ═══════════════════════════════════════════════════════════════════════════════

def test_encode_sex_creates_column():
    df = _make_metrics_df()
    result = encode_sex(df, "sex")
    assert "sex_encoded" in result.columns


def test_encode_sex_numeric():
    df = _make_metrics_df()
    result = encode_sex(df, "sex")
    assert pd.api.types.is_numeric_dtype(result["sex_encoded"])


# ═══════════════════════════════════════════════════════════════════════════════
# build_features
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_features_returns_tuple():
    df   = _make_metrics_df()
    cols = detect_columns(df)
    slopes = calculate_slopes(df, cols)
    if len(slopes) == 0:
        pytest.skip("No slopes")
    slopes = encode_sex(slopes, "sex")
    X, y, feat_cols = build_features(slopes, "group", "age", ["sex_encoded"])
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(y)


def test_build_features_filters_nan():
    df   = _make_metrics_df()
    cols = detect_columns(df)
    slopes = calculate_slopes(df, cols)
    if len(slopes) == 0:
        pytest.skip("No slopes")
    slopes = encode_sex(slopes, "sex")
    X, y, _ = build_features(slopes, "group", "age", ["sex_encoded"])
    assert not y.isna().any()


# ═══════════════════════════════════════════════════════════════════════════════
# train_svm
# ═══════════════════════════════════════════════════════════════════════════════

def test_train_svm_returns_dict():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                      "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    y = pd.Series(["A", "A", "A", "B", "B", "B"])
    result = train_svm(X, y, "test")
    assert result is not None
    assert "cv_mean" in result
    assert "train_acc" in result


def test_train_svm_insufficient_data():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series(["A", "B"])
    result = train_svm(X, y, "test")
    assert result is None


def test_train_svm_single_class():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series(["A", "A", "A", "A", "A"])
    result = train_svm(X, y, "test")
    assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# feature_importance
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_importance_returns_df():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                      "b": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series(["A", "A", "B", "B"])
    result = feature_importance(X, y)
    assert isinstance(result, pd.DataFrame)
    assert "f_score" in result.columns
    assert "feature" in result.columns


def test_feature_importance_sorted():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                      "b": [0.1, 0.1, 0.1, 0.1]})
    y = pd.Series(["A", "A", "B", "B"])
    result = feature_importance(X, y)
    assert result.iloc[0]["f_score"] >= result.iloc[-1]["f_score"]


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs_successfully(tmp_path):
    spec_file, _, out_dir = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(spec_file)]):
        main()
    assert (out_dir / "slopes_stratified.csv").exists()
    assert (out_dir / "stratified_svm_summary.json").exists()


def test_main_missing_metrics_exits(tmp_path):
    spec = {
        "inputs":  {"metrics_file": str(tmp_path / "nope.parquet")},
        "outputs": {"output_dir": str(tmp_path / "out")},
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    with patch("sys.argv", ["script", str(spec_file)]):
        with pytest.raises((SystemExit, Exception)):
            main()


def test_main_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        with pytest.raises(SystemExit):
            main()