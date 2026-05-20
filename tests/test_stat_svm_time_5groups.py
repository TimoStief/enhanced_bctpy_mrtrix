"""
Tests für stat_svm_time_5groups.py
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
mod = importlib.import_module("stat_svm_time_5groups")

load_config              = mod.load_config
detect_columns           = mod.detect_columns
load_and_merge           = mod.load_and_merge
aggregate_nodal          = mod.aggregate_nodal
calculate_slopes         = mod.calculate_slopes
build_features           = mod.build_features
train_random_forest      = mod.train_random_forest
train_svm                = mod.train_svm
feature_importance_anova = mod.feature_importance_anova
main                     = mod.main


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

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


def _make_run_spec(tmp_path, with_metadata=False):
    df = _make_df()
    d  = tmp_path / "data"
    d.mkdir()
    df.to_parquet(d / "metrics.parquet", index=False)
    out = tmp_path / "out"

    spec: dict = {
        "inputs":  {"metrics_file": str(d / "metrics.parquet")},
        "outputs": {"output_dir": str(out)},
    }

    if with_metadata:
        meta = df[["subject", "session", "group", "sex", "age"]].drop_duplicates()
        meta.to_csv(d / "meta.tsv", sep="\t", index=False)
        spec["inputs"]["metadata_file"] = str(d / "meta.tsv")

    sf = tmp_path / "run_spec.json"
    sf.write_text(json.dumps(spec))
    return sf, d / "metrics.parquet", out


def _make_slopes():
    df   = _make_df()
    cols = detect_columns(df)
    return calculate_slopes(df, cols), cols


# ═══════════════════════════════════════════════════════════════════════════════
# load_config
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_config_valid(tmp_path):
    sf, mf, _ = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(sf)]):
        config = load_config(mod.parse_args())
    assert config["metrics_file"] == mf


def test_load_config_missing_exits(tmp_path):
    with patch("sys.argv", ["script"]):
        with pytest.raises(SystemExit):
            load_config(mod.parse_args())


def test_load_config_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        with pytest.raises(SystemExit):
            load_config(mod.parse_args())


def test_load_config_cli_args(tmp_path):
    sf, mf, out = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script",
                             "--metrics-file", str(mf),
                             "--output-dir",   str(out)]):
        config = load_config(mod.parse_args())
    assert config["metrics_file"] == mf


# ═══════════════════════════════════════════════════════════════════════════════
# detect_columns
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_columns_finds_all():
    df = _make_df()
    r  = detect_columns(df)
    assert r["subject_col"] == "subject"
    assert r["group_col"]   == "group"
    assert r["sex_col"]     == "sex"
    assert r["age_col"]     == "age"
    assert "degree" in r["metric_cols"]


def test_detect_columns_no_age():
    df = _make_df().drop(columns=["age"])
    r  = detect_columns(df)
    assert r["age_col"] is None


def test_detect_columns_excludes_metadata():
    df = _make_df()
    r  = detect_columns(df)
    assert "group"   not in r["metric_cols"]
    assert "subject" not in r["metric_cols"]


# ═══════════════════════════════════════════════════════════════════════════════
# load_and_merge
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_and_merge_without_metadata(tmp_path):
    sf, mf, _ = _make_run_spec(tmp_path)
    df   = pd.read_parquet(mf)
    cols = detect_columns(df)
    result = load_and_merge(mf, None, cols)
    assert len(result) == len(df)


def test_load_and_merge_with_metadata(tmp_path):
    sf, mf, _ = _make_run_spec(tmp_path, with_metadata=True)
    df   = pd.read_parquet(mf)
    cols = detect_columns(df)
    meta_path = tmp_path / "data" / "meta.tsv"
    result = load_and_merge(mf, meta_path, cols)
    assert len(result) >= len(df)


def test_load_and_merge_nonexistent_metadata(tmp_path):
    sf, mf, _ = _make_run_spec(tmp_path)
    df   = pd.read_parquet(mf)
    cols = detect_columns(df)
    result = load_and_merge(mf, tmp_path / "nope.tsv", cols)
    assert len(result) == len(df)


# ═══════════════════════════════════════════════════════════════════════════════
# aggregate_nodal
# ═══════════════════════════════════════════════════════════════════════════════

def test_aggregate_nodal_with_node_col():
    df = _make_df()
    df["node"] = [1, 2] * (len(df) // 2) + [1] * (len(df) % 2)
    cols = detect_columns(df)
    result = aggregate_nodal(df, cols)
    assert len(result) < len(df)


def test_aggregate_nodal_without_node_col():
    df   = _make_df()
    cols = detect_columns(df)
    result = aggregate_nodal(df, cols)
    assert len(result) == len(df)


def test_aggregate_nodal_means_correctly():
    df = pd.DataFrame({
        "subject": ["S1", "S1"], "session": [1, 1],
        "group": ["A", "A"], "node": [1, 2],
        "degree": [2.0, 4.0],
    })
    cols = detect_columns(df)
    result = aggregate_nodal(df, cols)
    assert float(result["degree"].iloc[0]) == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_slopes
# ═══════════════════════════════════════════════════════════════════════════════

def test_calculate_slopes_basic():
    df   = _make_df()
    cols = detect_columns(df)
    result = calculate_slopes(df, cols)
    assert len(result) > 0
    assert any(c.endswith("_slope") for c in result.columns)


def test_calculate_slopes_needs_two_sessions():
    df   = _make_df(n_sessions=1)
    cols = detect_columns(df)
    assert len(calculate_slopes(df, cols)) == 0


def test_calculate_slopes_correct_value():
    """Perfekt lineare Daten → Slope = 1.0."""
    df = pd.DataFrame({
        "subject": ["S1", "S1", "S1"],
        "session": [1, 2, 3],
        "group":   ["A", "A", "A"],
        "degree":  [1.0, 2.0, 3.0],
    })
    cols   = detect_columns(df)
    result = calculate_slopes(df, cols)
    assert result.iloc[0]["degree_slope"] == pytest.approx(1.0)


def test_calculate_slopes_has_group_col():
    df   = _make_df()
    cols = detect_columns(df)
    result = calculate_slopes(df, cols)
    assert "group" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# build_features
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_features_returns_arrays():
    slopes, cols = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    X, y, feat_cols = build_features(slopes, "group", "sex", "age")
    assert len(X) == len(y)
    assert len(feat_cols) > 0


def test_build_features_encodes_sex():
    slopes, cols = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    X, y, feat_cols = build_features(slopes, "group", "sex", "age")
    assert "sex_encoded" in feat_cols


def test_build_features_includes_age():
    slopes, cols = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    X, y, feat_cols = build_features(slopes, "group", "sex", "age")
    assert "age" in feat_cols


def test_build_features_no_sex_col():
    slopes, cols = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    X, y, feat_cols = build_features(slopes, "group", None, "age")
    assert "sex_encoded" not in feat_cols


def test_build_features_filters_nan_groups():
    slopes, cols = _make_slopes()
    if len(slopes) == 0: pytest.skip("No slopes")
    slopes_nan = slopes.copy()
    slopes_nan.loc[slopes_nan.index[0], "group"] = np.nan
    X, y, _ = build_features(slopes_nan, "group", "sex", "age")
    assert not y.isna().any()


# ═══════════════════════════════════════════════════════════════════════════════
# train_random_forest
# ═══════════════════════════════════════════════════════════════════════════════

def test_train_rf_returns_dict():
    X = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    result = train_random_forest(X, y)
    assert "cv_mean"           in result
    assert "cv_std"            in result
    assert "train_acc"         in result
    assert "feature_importance" in result


def test_train_rf_cv_between_0_and_1():
    X = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    result = train_random_forest(X, y)
    assert 0.0 <= result["cv_mean"] <= 1.0


def test_train_rf_feature_importance_df():
    X = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    result = train_random_forest(X, y)
    fi = result["feature_importance"]
    assert isinstance(fi, pd.DataFrame)
    assert "feature"    in fi.columns
    assert "importance" in fi.columns


# ═══════════════════════════════════════════════════════════════════════════════
# train_svm
# ═══════════════════════════════════════════════════════════════════════════════

def test_train_svm_returns_dict():
    X = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    result = train_svm(X, y)
    assert "cv_mean"   in result
    assert "cv_std"    in result
    assert "train_acc" in result


def test_train_svm_cv_between_0_and_1():
    X = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
    y = pd.Series(["A"] * 10 + ["B"] * 10)
    result = train_svm(X, y)
    assert 0.0 <= result["cv_mean"] <= 1.0


def test_train_svm_multiclass():
    X = pd.DataFrame({"a": np.random.rand(30), "b": np.random.rand(30)})
    y = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    result = train_svm(X, y)
    assert "cv_mean" in result


# ═══════════════════════════════════════════════════════════════════════════════
# feature_importance_anova
# ═══════════════════════════════════════════════════════════════════════════════

def test_feature_importance_anova_returns_df():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series(["A", "A", "B", "B"])
    result = feature_importance_anova(X, y, ["a", "b"])
    assert isinstance(result, pd.DataFrame)
    assert "f_score"  in result.columns
    assert "p_value"  in result.columns
    assert "feature"  in result.columns


def test_feature_importance_anova_sorted():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                      "b": [0.1, 0.1, 0.1, 0.1]})
    y = pd.Series(["A", "A", "B", "B"])
    result = feature_importance_anova(X, y, ["a", "b"])
    assert result.iloc[0]["f_score"] >= result.iloc[-1]["f_score"]


def test_feature_importance_anova_handles_exception():
    """Einheitliche Werte → ANOVA schlägt fehl, kein Crash."""
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0]})
    y = pd.Series(["A", "A", "B", "B"])
    result = feature_importance_anova(X, y, ["a"])
    assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs(tmp_path):
    sf, _, out = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(sf)]):
        main()
    assert (out / "time_effect_slopes.csv").exists()
    assert (out / "svm_rf_time_summary.json").exists()
    assert (out / "feature_importance_anova.csv").exists()
    assert (out / "feature_importance_rf.csv").exists()


def test_main_with_metadata(tmp_path):
    sf, _, out = _make_run_spec(tmp_path, with_metadata=True)
    with patch("sys.argv", ["script", str(sf)]):
        main()
    assert (out / "time_effect_slopes.csv").exists()


def test_main_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        with pytest.raises(SystemExit):
            main()


def test_main_no_group_col_exits(tmp_path):
    df = _make_df().drop(columns=["group"])
    d  = tmp_path / "data"; d.mkdir()
    df.to_parquet(d / "metrics.parquet", index=False)
    spec = {"inputs":  {"metrics_file": str(d / "metrics.parquet")},
            "outputs": {"output_dir":   str(tmp_path / "out")}}
    sf = tmp_path / "run_spec.json"
    sf.write_text(json.dumps(spec))
    with patch("sys.argv", ["script", str(sf)]):
        with pytest.raises(SystemExit):
            main()


def test_main_no_slopes_exits(tmp_path):
    """Nur 1 Session pro Subject → keine Slopes → sys.exit."""
    df = _make_df(n_sessions=1)
    d  = tmp_path / "data"; d.mkdir()
    df.to_parquet(d / "metrics.parquet", index=False)
    spec = {"inputs":  {"metrics_file": str(d / "metrics.parquet")},
            "outputs": {"output_dir":   str(tmp_path / "out")}}
    sf = tmp_path / "run_spec.json"
    sf.write_text(json.dumps(spec))
    with patch("sys.argv", ["script", str(sf)]):
        with pytest.raises(SystemExit):
            main()


def test_main_summary_json_content(tmp_path):
    """Summary JSON enthält die erwarteten Keys."""
    sf, _, out = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(sf)]):
        main()
    import json as _json
    with open(out / "svm_rf_time_summary.json") as f:
        summary = _json.load(f)
    assert "best_model"    in summary
    assert "random_forest" in summary
    assert "svm"           in summary
    assert "n_samples"     in summary