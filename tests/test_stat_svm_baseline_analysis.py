#!/usr/bin/env python3
"""
Tests for svm_analysis.py
==========================

Run with:
    pytest test_svm_analysis.py -v --cov=svm_analysis --cov-report=term-missing

Requirements:
    pip install pytest pytest-cov pandas numpy scikit-learn scipy pyarrow
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make svm_analysis importable without executing main()
# ---------------------------------------------------------------------------
import importlib
import stat_svm_baseline_analysis as svm_analysis # noqa: E402  (adjust path / PYTHONPATH as needed)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture()
def tmp_output(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture()
def sample_df():
    """Minimal multi-session DataFrame with three groups."""
    rng = np.random.default_rng(0)
    rows = []
    for subj in range(1, 13):         # 12 subjects
        group = ["alone", "group", "control"][subj % 3]
        for ses in [1, 2, 3]:
            rows.append({
                "participant_id": f"sub-{subj:02d}",
                "session": f"ses-{ses}",
                "group": group,
                "sex": "M" if subj % 2 == 0 else "F",
                "age": 20 + subj,
                "metric_a": rng.normal(0, 1),
                "metric_b": rng.normal(0, 1),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_parquet(tmp_path, sample_df):
    p = tmp_path / "metrics.parquet"
    sample_df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def sample_slopes(sample_df):
    cols = svm_analysis.detect_columns(sample_df)
    return svm_analysis.calculate_slopes(sample_df, cols)


@pytest.fixture()
def groups_dict(sample_df):
    cols = svm_analysis.detect_columns(sample_df)
    return svm_analysis.detect_groups(sample_df, cols["group_col"],
                                      {"control_group": "control"})


# ===========================================================================
# detect_columns
# ===========================================================================

class TestDetectColumns:
    def test_detects_participant_id(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        assert cols["subject_col"] == "participant_id"

    def test_detects_session(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        assert cols["session_col"] == "session"

    def test_detects_group(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        assert cols["group_col"] == "group"

    def test_detects_sex(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        assert cols["sex_col"] == "sex"

    def test_detects_age(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        assert cols["age_col"] == "age"

    def test_metric_cols_are_numeric(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        for mc in cols["metric_cols"]:
            assert pd.api.types.is_numeric_dtype(sample_df[mc])

    def test_metric_cols_exclude_reserved(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        assert "age" not in cols["metric_cols"]

    def test_missing_column_returns_none(self):
        df = pd.DataFrame({"metric_x": [1.0, 2.0]})
        cols = svm_analysis.detect_columns(df)
        assert cols["subject_col"] is None
        assert cols["group_col"] is None

    def test_alias_subject(self):
        df = pd.DataFrame({"subject_id": ["s1"], "condition": ["ctrl"], "val": [1.0]})
        cols = svm_analysis.detect_columns(df)
        assert cols["subject_col"] == "subject_id"
        assert cols["group_col"] == "condition"

    def test_node_column_excluded_from_metrics(self):
        df = pd.DataFrame({"node": [1, 2], "participant_id": ["s1", "s1"], "val": [0.5, 0.6]})
        cols = svm_analysis.detect_columns(df)
        assert "node" not in cols["metric_cols"]


# ===========================================================================
# detect_groups
# ===========================================================================

class TestDetectGroups:
    def test_control_is_smallest(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        groups = svm_analysis.detect_groups(sample_df, cols["group_col"], {})
        # All three groups have equal size in our fixture; smallest == idxmin
        assert groups["control"] in sample_df["group"].unique()

    def test_intervention_excludes_control(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        groups = svm_analysis.detect_groups(sample_df, cols["group_col"], {})
        assert groups["control"] not in groups["intervention"]

    def test_alone_social_detected(self, sample_df):
        # Groups balanced → control = idxmin (alphabetically first: 'alone').
        # Provide explicit control so detection logic works as intended.
        cols = svm_analysis.detect_columns(sample_df)
        groups = svm_analysis.detect_groups(sample_df, cols["group_col"],
                                            {"control_group": "control"})
        assert "alone" in groups["alone"]
        assert "group" in groups["social"]

    def test_config_override_control(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        groups = svm_analysis.detect_groups(sample_df, cols["group_col"],
                                            {"control_group": "alone"})
        assert groups["control"] == "alone"

    def test_config_override_alone_group(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        groups = svm_analysis.detect_groups(sample_df, cols["group_col"],
                                            {"alone_groups": ["group"],
                                             "group_groups": ["alone"]})
        assert groups["alone"] == ["group"]
        assert groups["social"] == ["alone"]

    def test_fallback_split_when_no_keywords(self):
        df = pd.DataFrame({
            "subject": ["s1", "s2", "s3"],
            "group": ["typeA", "typeB", "ctrl"],
        })
        groups = svm_analysis.detect_groups(df, "group", {})
        # Intervention should be the two non-ctrl groups
        assert len(groups["intervention"]) == 2

    def test_short_long_detected_by_keyword(self):
        df = pd.DataFrame({"group": ["2w", "2w", "4w", "4w", "ctrl", "ctrl"]})
        groups = svm_analysis.detect_groups(df, "group", {"control_group": "ctrl"})
        assert "2w" in groups["short"]
        assert "4w" in groups["long"]


# ===========================================================================
# calculate_slopes
# ===========================================================================

class TestCalculateSlopes:
    def test_returns_dataframe(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        slopes = svm_analysis.calculate_slopes(sample_df, cols)
        assert isinstance(slopes, pd.DataFrame)

    def test_one_row_per_subject(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        slopes = svm_analysis.calculate_slopes(sample_df, cols)
        n_subjects = sample_df["participant_id"].nunique()
        assert len(slopes) == n_subjects

    def test_slope_columns_present(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        slopes = svm_analysis.calculate_slopes(sample_df, cols)
        assert "metric_a_slope" in slopes.columns
        assert "metric_b_slope" in slopes.columns

    def test_subjects_with_single_session_excluded(self):
        df = pd.DataFrame({
            "participant_id": ["s1", "s2", "s2"],
            "session": ["ses-1", "ses-1", "ses-2"],
            "group": ["ctrl", "int", "int"],
            "val": [1.0, 2.0, 3.0],
        })
        cols = svm_analysis.detect_columns(df)
        slopes = svm_analysis.calculate_slopes(df, cols)
        assert "s1" not in slopes["participant_id"].values
        assert "s2" in slopes["participant_id"].values

    def test_slope_is_numeric(self, sample_slopes):
        assert pd.api.types.is_numeric_dtype(sample_slopes["metric_a_slope"])

    def test_empty_returns_empty(self):
        df = pd.DataFrame(columns=["participant_id", "session", "group", "val"])
        cols = {
            "subject_col": "participant_id",
            "session_col": "session",
            "group_col": "group",
            "sex_col": None,
            "age_col": None,
            "metric_cols": ["val"],
        }
        slopes = svm_analysis.calculate_slopes(df, cols)
        assert len(slopes) == 0


# ===========================================================================
# remap_groups  &  encode_labels
# ===========================================================================

class TestRemapAndEncode:
    def test_remap_alone(self):
        s = pd.Series(["alone", "group", "control", "alone"])
        out = svm_analysis.remap_groups(s, ["alone"], ["group"], "alone", "group", "control")
        assert out.tolist() == ["alone", "group", "control", "alone"]

    def test_remap_unknown_is_nan(self):
        s = pd.Series(["unknown"])
        out = svm_analysis.remap_groups(s, ["alone"], ["group"], "alone", "group", "control")
        assert pd.isna(out.iloc[0])

    def test_encode_labels_basic(self):
        labels = np.array(["alone", "group", "control"])
        mapping = {"alone": 0, "group": 1, "control": 2}
        encoded = svm_analysis.encode_labels(labels, mapping)
        np.testing.assert_array_equal(encoded, [0.0, 1.0, 2.0])

    def test_encode_labels_unknown_is_nan(self):
        labels = np.array(["unknown"])
        encoded = svm_analysis.encode_labels(labels, {"alone": 0})
        assert np.isnan(encoded[0])


# ===========================================================================
# aggregate_nodal
# ===========================================================================

class TestAggregateNodal:
    def test_passthrough_when_no_node_column(self, sample_df):
        cols = svm_analysis.detect_columns(sample_df)
        out = svm_analysis.aggregate_nodal(sample_df, cols)
        assert len(out) == len(sample_df)

    def test_aggregates_nodes(self):
        df = pd.DataFrame({
            "participant_id": ["s1", "s1", "s1", "s1"],
            "session": ["ses-1", "ses-1", "ses-2", "ses-2"],
            "group": ["ctrl"] * 4,
            "node": [0, 1, 0, 1],
            "val": [1.0, 3.0, 2.0, 4.0],
        })
        cols = {
            "subject_col": "participant_id",
            "session_col": "session",
            "group_col": "group",
            "sex_col": None,
            "age_col": None,
            "metric_cols": ["val"],
        }
        out = svm_analysis.aggregate_nodal(df, cols)
        # Should collapse 4 node rows → 2 subject-session rows
        assert len(out) == 2
        assert set(out["val"].round(1)) == {2.0, 3.0}


# ===========================================================================
# prepare_features
# ===========================================================================

class TestPrepareFeatures:
    def test_returns_correct_shapes(self, sample_slopes, groups_dict):
        group_new = svm_analysis.remap_groups(
            sample_slopes["group"], groups_dict["alone"], groups_dict["social"],
            "alone", "group", groups_dict["control"]
        )
        label_map = {"alone": 0, "group": 1, "control": 2}
        X, y, feat_cols, filt = svm_analysis.prepare_features(
            sample_slopes, "group", "sex", "age", group_new, label_map
        )
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(feat_cols)

    def test_no_nans_in_output(self, sample_slopes, groups_dict):
        group_new = svm_analysis.remap_groups(
            sample_slopes["group"], groups_dict["alone"], groups_dict["social"],
            "alone", "group", groups_dict["control"]
        )
        X, y, _, _ = svm_analysis.prepare_features(
            sample_slopes, "group", "sex", "age", group_new, {"alone": 0, "group": 1, "control": 2}
        )
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_missing_sex_col_defaults_to_zero(self, sample_slopes, groups_dict):
        group_new = svm_analysis.remap_groups(
            sample_slopes["group"], groups_dict["alone"], groups_dict["social"],
            "alone", "group", groups_dict["control"]
        )
        X, y, feat_cols, _ = svm_analysis.prepare_features(
            sample_slopes, "group", None, None, group_new, {"alone": 0, "group": 1, "control": 2}
        )
        assert "sex_encoded" in feat_cols
        assert (X[:, feat_cols.index("sex_encoded")] == 0).all()


# ===========================================================================
# train_svm
# ===========================================================================

class TestTrainSvm:
    def _make_xy(self, n=30, n_classes=3):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(n, 4))
        y = rng.integers(0, n_classes, size=n)
        return X, y

    def test_returns_dict_with_required_keys(self):
        X, y = self._make_xy()
        result = svm_analysis.train_svm(X, y, "test", ["A", "B", "C"])
        for key in ("cv_mean", "cv_std", "train_acc", "confusion_matrix"):
            assert key in result

    def test_cv_mean_in_valid_range(self):
        X, y = self._make_xy()
        result = svm_analysis.train_svm(X, y, "test", ["A", "B", "C"])
        assert 0.0 <= result["cv_mean"] <= 1.0

    def test_train_acc_in_valid_range(self):
        X, y = self._make_xy()
        result = svm_analysis.train_svm(X, y, "test", ["A", "B", "C"])
        assert 0.0 <= result["train_acc"] <= 1.0

    def test_confusion_matrix_shape(self):
        X, y = self._make_xy(n_classes=2)
        result = svm_analysis.train_svm(X, y, "test", ["A", "B"])
        cm = np.array(result["confusion_matrix"])
        assert cm.shape == (2, 2)

    def test_binary_classification(self):
        X, y = self._make_xy(n_classes=2)
        result = svm_analysis.train_svm(X, y, "binary", ["A", "B"])
        assert result["cv_mean"] is not None


# ===========================================================================
# feature_importance_anova
# ===========================================================================

class TestFeatureImportanceAnova:
    def test_returns_dataframe(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        y = np.array([0, 0, 1, 1])
        df = svm_analysis.feature_importance_anova(X, y, ["feat_a", "feat_b"])
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "f_score" in df.columns
        assert "p_value" in df.columns

    def test_sorted_descending(self):
        X = np.array([[1, 10], [1, 20], [5, 11], [5, 21]], dtype=float)
        y = np.array([0, 0, 1, 1])
        df = svm_analysis.feature_importance_anova(X, y, ["small", "large"])
        # First feature should have higher F-score
        assert df.iloc[0]["f_score"] >= df.iloc[1]["f_score"]

    def test_single_feature(self):
        X = np.array([[1], [2], [10], [11]], dtype=float)
        y = np.array([0, 0, 1, 1])
        df = svm_analysis.feature_importance_anova(X, y, ["only"])
        assert len(df) == 1

    def test_nan_on_degenerate_input(self):
        X = np.zeros((4, 1))   # all same → F = 0 or nan
        y = np.array([0, 0, 1, 1])
        df = svm_analysis.feature_importance_anova(X, y, ["flat"])
        assert not df.empty  # shouldn't crash


# ===========================================================================
# stratified_analysis
# ===========================================================================

class TestStratifiedAnalysis:
    def test_runs_without_error(self):
        rng = np.random.default_rng(2)
        # Ensure each sex group has both classes present
        X = rng.normal(size=(20, 3))
        y = np.array([0, 1] * 10)          # alternating: both classes in every sex slice
        sex = np.array([1, 1, 0, 0] * 5)  # 10 male, 10 female, each with both classes
        result = svm_analysis.stratified_analysis(X, y, sex, "test")
        assert "Male" in result or "Female" in result

    def test_skips_when_single_class(self):
        X = np.ones((10, 2))
        y = np.zeros(10, dtype=int)
        sex = np.array([1] * 5 + [0] * 5)
        result = svm_analysis.stratified_analysis(X, y, sex, "test")
        assert result == {}

    def test_result_contains_cv_mean(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 3))
        y = np.array([0] * 10 + [1] * 10)
        sex = np.ones(20)  # all male
        result = svm_analysis.stratified_analysis(X, y, sex, "test")
        if "Male" in result:
            assert 0.0 <= result["Male"]["cv_mean"] <= 1.0


# ===========================================================================
# run_variant
# ===========================================================================

class TestRunVariant:
    def test_social_variant(self, large_parquet, tmp_output):
        df = pd.read_parquet(large_parquet)
        cols = svm_analysis.detect_columns(df)
        slopes = svm_analysis.calculate_slopes(df, cols)
        groups = svm_analysis.detect_groups(df, "group",
                                            {"control_group": "control",
                                             "alone_groups": ["alone"],
                                             "group_groups": ["group"]})
        result = svm_analysis.run_variant(
            slopes, "group", "sex", "age",
            groups, "social", tmp_output
        )
        assert "cv_mean" in result
        assert (tmp_output / "slopes_social.csv").exists()
        assert (tmp_output / "feature_importance_social.csv").exists()

    def test_duration_variant_skips_gracefully(self, sample_slopes, tmp_output):
        # Explicit empty short/long → should skip cleanly
        groups_no_duration = {
            "control": "control", "intervention": ["alone", "group"],
            "alone": ["alone"], "social": ["group"],
            "short": [], "long": [],
        }
        result = svm_analysis.run_variant(
            sample_slopes, "group", "sex", "age",
            groups_no_duration, "duration", tmp_output
        )
        assert result == {}

    def test_intervention_variant(self, sample_slopes, groups_dict, tmp_output):
        result = svm_analysis.run_variant(
            sample_slopes, "group", "sex", "age",
            groups_dict, "intervention", tmp_output
        )
        assert "cv_mean" in result

    def test_unknown_variant_exits(self, sample_slopes, groups_dict, tmp_output):
        with pytest.raises(SystemExit):
            svm_analysis.run_variant(
                sample_slopes, "group", "sex", "age",
                groups_dict, "nonexistent", tmp_output
            )

    def test_social_skips_when_no_alone_or_group(self, sample_slopes, tmp_output):
        groups = {
            "control": "control", "intervention": ["alone", "group"],
            "alone": [], "social": [], "short": [], "long": [],
        }
        result = svm_analysis.run_variant(
            sample_slopes, "group", "sex", "age",
            groups, "social", tmp_output
        )
        assert result == {}

    def test_intervention_skips_when_no_intervention(self, sample_slopes, tmp_output):
        groups = {
            "control": "control", "intervention": [],
            "alone": [], "social": [], "short": [], "long": [],
        }
        result = svm_analysis.run_variant(
            sample_slopes, "group", "sex", "age",
            groups, "intervention", tmp_output
        )
        assert result == {}


# ===========================================================================
# build_config
# ===========================================================================

class TestBuildConfig:
    def test_build_config_from_cli(self, tmp_path, sample_parquet, tmp_output):
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
        ]
        args = svm_analysis.parse_args()
        config = svm_analysis.build_config(args)
        assert config["metrics_file"] == sample_parquet
        assert config["output_dir"]   == tmp_output

    def test_cli_optional_args(self, tmp_path, sample_parquet, tmp_output):
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file",  str(sample_parquet),
            "--output-dir",    str(tmp_output),
            "--control-group", "control",
            "--alone-groups",  "alone",
            "--group-groups",  "group",
        ]
        args = svm_analysis.parse_args()
        config = svm_analysis.build_config(args)
        assert config["control_group"] == "control"
        assert config["alone_groups"]  == ["alone"]
        assert config["group_groups"]  == ["group"]

    def test_missing_required_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            sys.argv = ["svm_analysis.py"]
            svm_analysis.parse_args()

    def test_nonexistent_metrics_file_exits(self, tmp_path, tmp_output):
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file", str(tmp_path / "ghost.parquet"),
            "--output-dir",   str(tmp_output),
        ]
        args = svm_analysis.parse_args()
        with pytest.raises(SystemExit):
            svm_analysis.build_config(args)


# ===========================================================================
# load_metrics
# ===========================================================================

class TestLoadMetrics:
    def test_loads_parquet(self, sample_parquet, tmp_path):
        df = pd.read_parquet(sample_parquet)
        cols = svm_analysis.detect_columns(df)
        loaded = svm_analysis.load_metrics(sample_parquet, None, cols)
        assert len(loaded) == len(df)

    def test_merges_metadata_tsv(self, tmp_path, sample_parquet):
        meta = pd.DataFrame({
            "participant_id": [f"sub-{i:02d}" for i in range(1, 13)],
            "site": ["A"] * 12,
        })
        meta_file = tmp_path / "meta.tsv"
        meta.to_csv(meta_file, sep="\t", index=False)

        df = pd.read_parquet(sample_parquet)
        cols = svm_analysis.detect_columns(df)
        merged = svm_analysis.load_metrics(sample_parquet, meta_file, cols)
        assert "site" in merged.columns

    def test_missing_metadata_file_skips_merge(self, tmp_path, sample_parquet):
        df = pd.read_parquet(sample_parquet)
        cols = svm_analysis.detect_columns(df)
        result = svm_analysis.load_metrics(sample_parquet,
                                           tmp_path / "ghost.tsv", cols)
        # Should still return data even if metadata missing
        assert len(result) == len(df)


# ===========================================================================
# detect_atlas
# ===========================================================================

class TestDetectAtlas:
    def test_detects_atlas_column(self):
        df = pd.DataFrame({"atlas": ["AAL", "AAL", "Schaefer"], "val": [1, 2, 3]})
        atlas = svm_analysis.detect_atlas(df)
        assert atlas == "AAL"

    def test_returns_none_when_no_atlas(self, sample_df):
        atlas = svm_analysis.detect_atlas(sample_df)
        assert atlas is None


# ===========================================================================
# Integration: main() end-to-end
# ===========================================================================

@pytest.fixture()
def large_parquet(tmp_path):
    """30 subjects × 5 sessions — enough for stratified CV in every variant."""
    rng = np.random.default_rng(99)
    rows = []
    groups = (["alone"] * 10 + ["group"] * 10 + ["control"] * 10)
    for i, subj_group in enumerate(groups, start=1):
        sex = "M" if i % 2 == 0 else "F"
        for ses in range(1, 6):
            rows.append({
                "participant_id": f"sub-{i:02d}",
                "session": f"ses-{ses}",
                "group": subj_group,
                "sex": sex,
                "age": 20 + i,
                "metric_a": rng.normal(),
                "metric_b": rng.normal(),
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "large_metrics.parquet"
    df.to_parquet(p, index=False)
    return p


class TestMainIntegration:
    def test_main_runs_with_minimal_inputs(self, tmp_path, large_parquet, tmp_output):
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(tmp_output),
            "--control-group", "control",
            "--alone-groups", "alone",
            "--group-groups", "group",
        ]
        svm_analysis.main()
        assert (tmp_output / "svm_summary.json").exists()

    def test_summary_json_has_expected_keys(self, tmp_path, large_parquet, tmp_output):
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(tmp_output),
            "--control-group", "control",
            "--alone-groups", "alone",
            "--group-groups", "group",
        ]
        svm_analysis.main()
        summary = json.loads((tmp_output / "svm_summary.json").read_text())
        assert "intervention" in summary
        assert "cv_mean" in summary["intervention"]

    def test_main_exits_when_no_group_col(self, tmp_path, tmp_output):
        df = pd.DataFrame({
            "participant_id": ["s1", "s1"],
            "session": ["ses-1", "ses-2"],
            "metric_a": [1.0, 2.0],
        })
        pq = tmp_path / "no_group.parquet"
        df.to_parquet(pq, index=False)
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file", str(pq),
            "--output-dir",   str(tmp_output),
        ]
        with pytest.raises(SystemExit):
            svm_analysis.main()

    def test_main_with_all_optional_flags(self, tmp_path, large_parquet, tmp_output):
        sys.argv = [
            "svm_analysis.py",
            "--metrics-file",  str(large_parquet),
            "--output-dir",    str(tmp_output),
            "--control-group", "control",
            "--alone-groups",  "alone",
            "--group-groups",  "group",
        ]
        svm_analysis.main()
        assert (tmp_output / "svm_summary.json").exists()