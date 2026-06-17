"""
Tests für stat_random_forest_comparison.py
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
mod = importlib.import_module("stat_random_forest_comparison")

load_config              = mod.load_config
detect_columns           = mod.detect_columns
detect_groups            = mod.detect_groups
aggregate_nodal          = mod.aggregate_nodal
calculate_slopes         = mod.calculate_slopes
remap_groups             = mod.remap_groups
build_features           = mod.build_features
train_rf                 = mod.train_rf
train_svm_model          = mod.train_svm_model
feature_importance_anova = mod.feature_importance_anova
run_variant              = mod.run_variant
main                     = mod.main


def _make_df(n_subjects=6, n_sessions=3):
    records = []
    groups = ["alone_2w", "alone_4w", "group_2w", "group_4w", "control", "alone_2w"]
    for i in range(min(n_subjects, 6)):
        for s in range(1, n_sessions + 1):
            records.append({
                "subject": f"S{i+1}", "session": s,
                "group": groups[i], "sex": "M" if i % 2 == 0 else "F",
                "age": float(25 + i), "degree": float(i + s * 0.1),
                "strength": float(i * 0.5),
            })
    return pd.DataFrame(records)


def _make_slopes(n_subjects=6):
    df   = _make_df(n_subjects=n_subjects)
    cols = detect_columns(df)
    return calculate_slopes(df, cols)


def _make_run_spec(tmp_path):
    df = _make_df()
    d  = tmp_path / "data"; d.mkdir()
    df.to_parquet(d / "metrics.parquet", index=False)
    out = tmp_path / "out"
    spec = {
        "inputs":  {"metrics_file": str(d / "metrics.parquet")},
        "outputs": {"output_dir": str(out)},
        "control_group": "control",
        "alone_groups":  ["alone_2w", "alone_4w"],
        "group_groups":  ["group_2w", "group_4w"],
        "short_groups":  ["alone_2w", "group_2w"],
        "long_groups":   ["alone_4w", "group_4w"],
    }
    sf = tmp_path / "run_spec.json"
    sf.write_text(json.dumps(spec))
    return sf, d / "metrics.parquet", out


# ── load_config ──────────────────────────────────────────────────────────────

def test_load_config_valid(tmp_path):
    sf, mf, _ = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(sf)]):
        config = load_config(mod.parse_args())
    assert config["metrics_file"] == mf
    assert config["control_group"] == "control"


def test_load_config_missing_exits(tmp_path):
    with patch("sys.argv", ["script"]):
        with pytest.raises(SystemExit):
            load_config(mod.parse_args())


# ── detect_columns ────────────────────────────────────────────────────────────

def test_detect_columns_standard():
    df = _make_df()
    r  = detect_columns(df)
    assert r["subject_col"] == "subject"
    assert r["group_col"]   == "group"
    assert "degree" in r["metric_cols"]


# ── detect_groups ─────────────────────────────────────────────────────────────

def test_detect_groups_from_config():
    df     = _make_df()
    config = {"control_group": "control",
              "alone_groups": ["alone_2w", "alone_4w"],
              "group_groups": ["group_2w", "group_4w"]}
    groups = detect_groups(df, "group", config)
    assert groups["control"] == "control"
    assert "alone_2w" in groups["alone"]
    assert "group_2w" in groups["social"]


def test_detect_groups_auto_label_detection():
    df     = _make_df()
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    assert "alone_2w" in groups["alone"]
    assert "group_2w" in groups["social"]


def test_detect_groups_fallback_split():
    df = pd.DataFrame({
        "subject": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 2,
        "group":   ["X"] * 4 + ["Y"] * 4 + ["Z"] * 2,
    })
    config = {"control_group": "Z"}
    groups = detect_groups(df, "group", config)
    assert len(groups["alone"]) > 0 or len(groups["short"]) > 0


# ── calculate_slopes ─────────────────────────────────────────────────────────

def test_calculate_slopes_basic():
    df   = _make_df()
    cols = detect_columns(df)
    result = calculate_slopes(df, cols)
    assert len(result) > 0
    assert any(c.endswith("_slope") for c in result.columns)


def test_calculate_slopes_single_session_skipped():
    df   = _make_df(n_sessions=1)
    cols = detect_columns(df)
    assert len(calculate_slopes(df, cols)) == 0


# ── remap_groups ─────────────────────────────────────────────────────────────

def test_remap_groups_alone():
    series = pd.Series(["alone_2w", "group_2w", "control"])
    result = remap_groups(series, ["alone_2w"], ["group_2w"], "alone", "group", "control")
    assert result.iloc[0] == "alone"
    assert result.iloc[1] == "group"
    assert result.iloc[2] == "control"


def test_remap_groups_unknown_is_nan():
    series = pd.Series(["unknown_group"])
    result = remap_groups(series, ["alone_2w"], ["group_2w"], "alone", "group", "control")
    assert pd.isna(result.iloc[0])


# ── build_features ────────────────────────────────────────────────────────────

def test_build_features_shape():
    slopes = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    group_new  = remap_groups(slopes["group"], ["alone_2w", "alone_4w"],
                               ["group_2w", "group_4w"], "alone", "group", "control")
    label_map  = {"alone": 0, "group": 1, "control": 2}
    X, y, feat = build_features(slopes, group_new, label_map, "sex", "age")
    assert len(X) == len(y)
    assert len(feat) > 0


def test_build_features_filters_nan():
    slopes = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    group_new = remap_groups(slopes["group"], ["alone_2w"], ["group_2w"],
                              "alone", "group", "control")
    X, y, _ = build_features(slopes, group_new, {"alone": 0, "group": 1, "control": 2},
                              "sex", "age")
    assert not np.any(np.isnan(y))


# ── train_rf ─────────────────────────────────────────────────────────────────

def test_train_rf_returns_dict():
    X = np.random.rand(20, 3)
    y = np.array([0] * 10 + [1] * 10)
    result = train_rf(X, y, ["A", "B"])
    assert "cv_mean" in result
    assert "feature_importance" in result


# ── train_svm_model ───────────────────────────────────────────────────────────

def test_train_svm_model_returns_dict():
    X = np.random.rand(20, 3)
    y = np.array([0] * 10 + [1] * 10)
    result = train_svm_model(X, y, ["A", "B"])
    assert "cv_mean" in result
    assert "train_acc" in result


# ── feature_importance_anova ─────────────────────────────────────────────────

def test_feature_importance_anova_sorted():
    X = np.array([[1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1]])
    y = np.array([0, 0, 1, 1])
    result = feature_importance_anova(X, y)
    assert result.iloc[0]["f_score"] >= result.iloc[-1]["f_score"]


# ── run_variant ───────────────────────────────────────────────────────────────

def test_run_variant_social(tmp_path):
    slopes = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    df   = _make_df()
    cols = detect_columns(df)
    config = {"control_group": "control",
              "alone_groups": ["alone_2w", "alone_4w"],
              "group_groups": ["group_2w", "group_4w"]}
    groups = detect_groups(df, "group", config)
    result = run_variant(slopes, "group", "sex", "age", groups, "social", tmp_path)
    assert isinstance(result, dict)


def test_run_variant_skips_missing_groups(tmp_path):
    slopes = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    df     = _make_df()
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    groups["alone"]  = []
    groups["social"] = []
    result = run_variant(slopes, "group", "sex", "age", groups, "social", tmp_path)
    assert result == {}


# ── main ──────────────────────────────────────────────────────────────────────

def test_main_runs(tmp_path):
    sf, _, out = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(sf)]):
        main()
    assert (out / "slopes_rf_comparison.csv").exists()
    assert (out / "rf_svm_comparison_summary.json").exists()


def test_main_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_metrics_exits(tmp_path):
    spec = {"inputs": {"metrics_file": str(tmp_path / "nope.parquet")},
            "outputs": {"output_dir": str(tmp_path / "out")}}
    sf = tmp_path / "run_spec.json"
    sf.write_text(json.dumps(spec))
    with patch("sys.argv", ["script", str(sf)]):
        with pytest.raises((SystemExit, Exception)):
            main()