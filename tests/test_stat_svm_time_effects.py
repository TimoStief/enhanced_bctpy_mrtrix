#!/usr/bin/env python3
"""
Tests für stat_svm_time_effects.py
====================================

Run with:
    pytest tests/test_stat_svm_time_effects.py -v \
           --cov=stat_svm_time_effects --cov-report=term-missing

Requirements:
    pip install pytest pytest-cov pandas numpy scikit-learn scipy pyarrow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "4_statistical_classification"))
import stat_svm_time_effects as te


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
    """12 subjects × 3 sessions, three groups, with age/sex."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(1, 13):
        group = ["ctrl", "alone", "group"][i % 3]
        sex   = "M" if i % 2 == 0 else "F"
        for ses in [1, 2, 3]:
            rows.append({
                "participant_id": f"sub-{i:02d}",
                "session":        f"ses-{ses}",
                "group":          group,
                "sex":            sex,
                "age":            20 + i,
                "metric_a":       rng.normal(),
                "metric_b":       rng.normal(),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_parquet(tmp_path, sample_df):
    p = tmp_path / "metrics.parquet"
    sample_df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def large_df():
    """30 subjects × 5 sessions — enough for stratified CV."""
    rng = np.random.default_rng(99)
    rows = []
    groups = ["ctrl"] * 10 + ["alone"] * 10 + ["group"] * 10
    for i, grp in enumerate(groups, start=1):
        sex = "M" if i % 2 == 0 else "F"
        for ses in range(1, 6):
            rows.append({
                "participant_id": f"sub-{i:02d}",
                "session":        f"ses-{ses}",
                "group":          grp,
                "sex":            sex,
                "age":            20 + i,
                "metric_a":       rng.normal(),
                "metric_b":       rng.normal(),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def large_parquet(tmp_path, large_df):
    p = tmp_path / "large_metrics.parquet"
    large_df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def base_config(sample_parquet, tmp_output):
    return {
        "metrics_file":  sample_parquet,
        "output_dir":    tmp_output,
        "metadata_file": None,
        "subject_col":   None,
        "session_col":   None,
        "group_col":     None,
        "age_col":       None,
        "sex_col":       None,
    }


@pytest.fixture()
def detected_cols(sample_df, base_config):
    return te.detect_columns(sample_df, base_config)


@pytest.fixture()
def sample_slopes(sample_df, detected_cols):
    return te.calculate_slopes(sample_df, detected_cols)


# ===========================================================================
# parse_args / build_config
# ===========================================================================

class TestParseArgs:
    def test_required_args_present(self, sample_parquet, tmp_output):
        sys.argv = [
            "stat_svm_time_effects.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
        ]
        args = te.parse_args()
        assert args.metrics_file == str(sample_parquet)
        assert args.output_dir   == str(tmp_output)

    def test_missing_metrics_file_exits(self, tmp_output):
        with pytest.raises(SystemExit):
            sys.argv = ["te.py", "--output-dir", str(tmp_output)]
            te.parse_args()

    def test_missing_output_dir_exits(self, sample_parquet):
        with pytest.raises(SystemExit):
            sys.argv = ["te.py", "--metrics-file", str(sample_parquet)]
            te.parse_args()

    def test_optional_args_default_none(self, sample_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
        ]
        args = te.parse_args()
        assert args.metadata    is None
        assert args.subject_col is None
        assert args.group_col   is None
        assert args.sex_col     is None
        assert args.age_col     is None

    def test_optional_args_accepted(self, sample_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
            "--group-col",    "condition",
            "--sex-col",      "gender",
            "--age-col",      "age",
        ]
        args = te.parse_args()
        assert args.group_col == "condition"
        assert args.sex_col   == "gender"
        assert args.age_col   == "age"


class TestBuildConfig:
    def test_returns_dict_with_required_keys(self, sample_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
        ]
        args   = te.parse_args()
        config = te.build_config(args)
        assert "metrics_file"  in config
        assert "output_dir"    in config
        assert "metadata_file" in config

    def test_paths_are_resolved(self, sample_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
        ]
        args   = te.parse_args()
        config = te.build_config(args)
        assert config["metrics_file"].is_absolute()
        assert config["output_dir"].is_absolute()

    def test_nonexistent_metrics_file_exits(self, tmp_path, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(tmp_path / "ghost.parquet"),
            "--output-dir",   str(tmp_output),
        ]
        args = te.parse_args()
        with pytest.raises(SystemExit):
            te.build_config(args)

    def test_metadata_none_when_not_provided(self, sample_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
        ]
        args   = te.parse_args()
        config = te.build_config(args)
        assert config["metadata_file"] is None

    def test_metadata_resolved_when_provided(self, tmp_path, sample_parquet, tmp_output):
        meta = tmp_path / "meta.tsv"
        meta.write_text("participant_id\tgroup\nsub-01\tctrl\n")
        sys.argv = [
            "te.py",
            "--metrics-file", str(sample_parquet),
            "--output-dir",   str(tmp_output),
            "--metadata",     str(meta),
        ]
        args   = te.parse_args()
        config = te.build_config(args)
        assert config["metadata_file"] == meta.resolve()


# ===========================================================================
# detect_columns
# ===========================================================================

class TestDetectColumns:
    def test_detects_subject(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        assert cols["subject_col"] == "participant_id"

    def test_detects_session(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        assert cols["session_col"] == "session"

    def test_detects_group(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        assert cols["group_col"] == "group"

    def test_detects_sex(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        assert cols["sex_col"] == "sex"

    def test_detects_age(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        assert cols["age_col"] == "age"

    def test_metric_cols_are_numeric(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        for mc in cols["metric_cols"]:
            assert pd.api.types.is_numeric_dtype(sample_df[mc])

    def test_reserved_cols_excluded_from_metrics(self, sample_df, base_config):
        cols = te.detect_columns(sample_df, base_config)
        assert "age" not in cols["metric_cols"]
        assert "session" not in cols["metric_cols"]

    def test_config_override_respected(self, sample_df, base_config):
        config = {**base_config, "group_col": "sex"}
        cols = te.detect_columns(sample_df, config)
        assert cols["group_col"] == "sex"

    def test_missing_col_returns_none(self):
        df = pd.DataFrame({"val": [1.0, 2.0]})
        cols = te.detect_columns(df, {})
        assert cols["subject_col"] is None
        assert cols["group_col"]   is None

    def test_node_excluded_from_metrics(self):
        df = pd.DataFrame({
            "participant_id": ["s1"],
            "node": [0],
            "val": [1.0],
        })
        cols = te.detect_columns(df, {})
        assert "node" not in cols["metric_cols"]


# ===========================================================================
# load_data
# ===========================================================================

class TestLoadData:
    def test_loads_parquet(self, sample_parquet, detected_cols):
        df = te.load_data(sample_parquet, None, detected_cols)
        assert len(df) > 0

    def test_merges_tsv_metadata(self, tmp_path, sample_parquet, detected_cols):
        meta = pd.DataFrame({
            "participant_id": [f"sub-{i:02d}" for i in range(1, 13)],
            "site": ["A"] * 12,
        })
        meta_file = tmp_path / "meta.tsv"
        meta.to_csv(meta_file, sep="\t", index=False)
        df = te.load_data(sample_parquet, meta_file, detected_cols)
        assert "site" in df.columns

    def test_missing_metadata_returns_data(self, tmp_path, sample_parquet, detected_cols):
        df = te.load_data(sample_parquet, tmp_path / "ghost.tsv", detected_cols)
        assert len(df) > 0

    def test_merges_csv_metadata(self, tmp_path, sample_parquet, detected_cols):
        meta = pd.DataFrame({
            "participant_id": [f"sub-{i:02d}" for i in range(1, 13)],
            "cohort": ["X"] * 12,
        })
        meta_file = tmp_path / "meta.csv"
        meta.to_csv(meta_file, index=False)
        df = te.load_data(sample_parquet, meta_file, detected_cols)
        assert "cohort" in df.columns


# ===========================================================================
# aggregate_nodal
# ===========================================================================

class TestAggregateNodal:
    def test_passthrough_without_node_col(self, sample_df, detected_cols):
        out = te.aggregate_nodal(sample_df, detected_cols)
        assert len(out) == len(sample_df)

    def test_aggregates_nodes_to_subject_session(self, detected_cols):
        df = pd.DataFrame({
            "participant_id": ["s1", "s1", "s1", "s1"],
            "session":        ["ses-1", "ses-1", "ses-2", "ses-2"],
            "group":          ["ctrl"] * 4,
            "node":           [0, 1, 0, 1],
            "val":            [1.0, 3.0, 2.0, 4.0],
        })
        cols = {**detected_cols, "metric_cols": ["val"]}
        out = te.aggregate_nodal(df, cols)
        assert len(out) == 2
        assert set(out["val"].round(1)) == {2.0, 3.0}


# ===========================================================================
# extract_session_number
# ===========================================================================

class TestExtractSessionNumber:
    def test_extracts_from_ses_string(self):
        s = pd.Series(["ses-1", "ses-2", "ses-3"])
        out = te.extract_session_number(s)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_extracts_plain_integers(self):
        s = pd.Series([1, 2, 3])
        out = te.extract_session_number(s)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_fallback_to_arange_for_strings(self):
        s = pd.Series(["baseline", "followup", "longterm"])
        out = te.extract_session_number(s)
        assert len(out) == 3
        assert out[0] < out[1] < out[2]


# ===========================================================================
# calculate_slopes
# ===========================================================================

class TestCalculateSlopes:
    def test_returns_dataframe(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        assert isinstance(slopes, pd.DataFrame)

    def test_one_row_per_subject(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        assert len(slopes) == sample_df["participant_id"].nunique()

    def test_slope_cols_created(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        assert "metric_a_slope" in slopes.columns
        assert "metric_b_slope" in slopes.columns

    def test_single_session_subjects_excluded(self, detected_cols):
        df = pd.DataFrame({
            "participant_id": ["s1", "s2", "s2"],
            "session":        ["ses-1", "ses-1", "ses-2"],
            "group":          ["ctrl", "int", "int"],
            "val":            [1.0, 2.0, 3.0],
        })
        cols = {**detected_cols, "metric_cols": ["val"]}
        slopes = te.calculate_slopes(df, cols)
        assert "s1" not in slopes["participant_id"].values
        assert "s2" in slopes["participant_id"].values

    def test_n_sessions_column_present(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        assert "n_sessions" in slopes.columns
        assert (slopes["n_sessions"] == 3).all()

    def test_group_col_preserved(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        assert "group" in slopes.columns

    def test_empty_df_returns_empty(self, detected_cols):
        df = pd.DataFrame(columns=["participant_id", "session", "group", "val"])
        cols = {**detected_cols, "metric_cols": ["val"]}
        slopes = te.calculate_slopes(df, cols)
        assert len(slopes) == 0

    def test_exits_when_no_subject_col(self, sample_df):
        cols = {
            "subject_col": None,
            "session_col": "session",
            "group_col":   "group",
            "age_col":     None,
            "sex_col":     None,
            "metric_cols": ["metric_a"],
        }
        with pytest.raises(SystemExit):
            te.calculate_slopes(sample_df, cols)


# ===========================================================================
# encode_sex
# ===========================================================================

class TestEncodeSex:
    def test_encodes_M_F(self, sample_slopes):
        out = te.encode_sex(sample_slopes, "sex")
        assert "sex_encoded" in out.columns
        assert set(out["sex_encoded"].unique()).issubset({0, 1})

    def test_defaults_to_zero_without_sex_col(self, sample_slopes):
        out = te.encode_sex(sample_slopes, None)
        assert "sex_encoded" in out.columns
        assert (out["sex_encoded"] == 0).all()

    def test_uses_label_encoder_for_unknown_values(self, sample_slopes):
        df = sample_slopes.copy()
        df["sex"] = df["sex"].map({"M": "male", "F": "female"})
        out = te.encode_sex(df, "sex")
        assert "sex_encoded" in out.columns
        assert out["sex_encoded"].notna().all()

    def test_does_not_modify_original(self, sample_slopes):
        original_cols = list(sample_slopes.columns)
        te.encode_sex(sample_slopes, "sex")
        assert list(sample_slopes.columns) == original_cols


# ===========================================================================
# prepare_features
# ===========================================================================

class TestPrepareFeatures:
    def _slopes_with_sex(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        return te.encode_sex(slopes, detected_cols["sex_col"])

    def test_returns_correct_shapes(self, sample_df, detected_cols):
        slopes = self._slopes_with_sex(sample_df, detected_cols)
        X, y, feat_cols, le, _ = te.prepare_features(slopes, detected_cols)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(feat_cols)

    def test_no_nans_in_output(self, sample_df, detected_cols):
        slopes = self._slopes_with_sex(sample_df, detected_cols)
        X, y, _, _, _ = te.prepare_features(slopes, detected_cols)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_label_encoder_returned(self, sample_df, detected_cols):
        slopes = self._slopes_with_sex(sample_df, detected_cols)
        _, _, _, le, _ = te.prepare_features(slopes, detected_cols)
        assert isinstance(le, LabelEncoder)
        assert len(le.classes_) >= 2

    def test_age_included_as_covariate(self, sample_df, detected_cols):
        slopes = self._slopes_with_sex(sample_df, detected_cols)
        _, _, feat_cols, _, _ = te.prepare_features(slopes, detected_cols)
        assert "age" in feat_cols

    def test_sex_encoded_included(self, sample_df, detected_cols):
        slopes = self._slopes_with_sex(sample_df, detected_cols)
        _, _, feat_cols, _, _ = te.prepare_features(slopes, detected_cols)
        assert "sex_encoded" in feat_cols

    def test_n_sessions_included(self, sample_df, detected_cols):
        slopes = self._slopes_with_sex(sample_df, detected_cols)
        _, _, feat_cols, _, _ = te.prepare_features(slopes, detected_cols)
        assert "n_sessions" in feat_cols

    def test_exits_when_no_group_col(self, sample_df, detected_cols):
        slopes = te.calculate_slopes(sample_df, detected_cols)
        slopes = te.encode_sex(slopes, detected_cols["sex_col"])
        cols_no_group = {**detected_cols, "group_col": None}
        with pytest.raises(SystemExit):
            te.prepare_features(slopes, cols_no_group)


# ===========================================================================
# train_svm
# ===========================================================================

class TestTrainSvm:
    def _make_xy(self, n=30, n_classes=2):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(n, 4))
        y = np.array(list(range(n_classes)) * (n // n_classes) + list(range(n % n_classes)))
        le = LabelEncoder()
        le.fit([str(i) for i in range(n_classes)])
        return X, y, le

    def test_returns_required_keys(self):
        X, y, le = self._make_xy()
        result = te.train_svm(X, y, le)
        for key in ("cv_mean", "cv_std", "train_acc", "svm", "scaler", "X_scaled", "y_pred"):
            assert key in result

    def test_cv_mean_in_valid_range(self):
        X, y, le = self._make_xy(n=40)
        result = te.train_svm(X, y, le)
        assert 0.0 <= result["cv_mean"] <= 1.0

    def test_train_acc_in_valid_range(self):
        X, y, le = self._make_xy()
        result = te.train_svm(X, y, le)
        assert 0.0 <= result["train_acc"] <= 1.0

    def test_skips_cv_when_too_few_samples(self):
        X = np.ones((3, 2))
        y = np.array([0, 1, 0])
        le = LabelEncoder().fit(["a", "b"])
        result = te.train_svm(X, y, le)
        assert np.isnan(result["cv_mean"])

    def test_multiclass_works(self):
        X, y, le = self._make_xy(n=30, n_classes=3)
        result = te.train_svm(X, y, le)
        assert result["cv_mean"] is not None

    def test_x_scaled_shape_matches_input(self):
        X, y, le = self._make_xy()
        result = te.train_svm(X, y, le)
        assert result["X_scaled"].shape == X.shape


# ===========================================================================
# analyze_feature_importance
# ===========================================================================

class TestAnalyzeFeatureImportance:
    def test_returns_dataframe(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        y = np.array([0, 0, 1, 1])
        df = te.analyze_feature_importance(X, y, ["feat_a", "feat_b"])
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "f_score" in df.columns
        assert "p_value" in df.columns

    def test_sorted_descending(self):
        X = np.array([[1, 10], [1, 20], [5, 11], [5, 21]], dtype=float)
        y = np.array([0, 0, 1, 1])
        df = te.analyze_feature_importance(X, y, ["small", "large"])
        assert df.iloc[0]["f_score"] >= df.iloc[1]["f_score"]

    def test_handles_degenerate_input(self):
        X = np.zeros((4, 1))
        y = np.array([0, 0, 1, 1])
        df = te.analyze_feature_importance(X, y, ["flat"])
        assert len(df) == 1

    def test_length_matches_features(self):
        X = np.random.default_rng(0).normal(size=(20, 5))
        y = np.array([0] * 10 + [1] * 10)
        df = te.analyze_feature_importance(X, y, [f"f{i}" for i in range(5)])
        assert len(df) == 5


# ===========================================================================
# save_results
# ===========================================================================

class TestSaveResults:
    def _run_and_save(self, large_df, detected_cols, tmp_output):
        slopes = te.calculate_slopes(large_df, detected_cols)
        slopes = te.encode_sex(slopes, detected_cols["sex_col"])
        X, y, feat_cols, le, slopes = te.prepare_features(slopes, detected_cols)
        svm_result    = te.train_svm(X, y, le)
        importance_df = te.analyze_feature_importance(X, y, feat_cols)
        te.save_results(tmp_output, slopes, svm_result, y, feat_cols, importance_df, le)
        return svm_result, importance_df, le, y, feat_cols

    def test_summary_json_created(self, large_df, detected_cols, tmp_output):
        cols = te.detect_columns(large_df, {})
        self._run_and_save(large_df, cols, tmp_output)
        assert (tmp_output / "svm_time_effects_summary.json").exists()

    def test_slopes_csv_created(self, large_df, tmp_output):
        cols = te.detect_columns(large_df, {})
        self._run_and_save(large_df, cols, tmp_output)
        assert (tmp_output / "time_effect_slopes.csv").exists()

    def test_importance_csv_created(self, large_df, tmp_output):
        cols = te.detect_columns(large_df, {})
        self._run_and_save(large_df, cols, tmp_output)
        assert (tmp_output / "feature_importance_time_effects.csv").exists()

    def test_summary_json_has_expected_keys(self, large_df, tmp_output):
        cols = te.detect_columns(large_df, {})
        self._run_and_save(large_df, cols, tmp_output)
        summary = json.loads((tmp_output / "svm_time_effects_summary.json").read_text())
        for key in ("cv_accuracy_mean", "training_accuracy", "n_samples",
                    "n_features", "groups", "top_features"):
            assert key in summary


# ===========================================================================
# Integration: main()
# ===========================================================================

class TestMainIntegration:
    def test_main_minimal(self, large_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(tmp_output),
        ]
        te.main()
        assert (tmp_output / "svm_time_effects_summary.json").exists()
        assert (tmp_output / "time_effect_slopes.csv").exists()
        assert (tmp_output / "feature_importance_time_effects.csv").exists()

    def test_main_with_explicit_group_col(self, large_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(tmp_output),
            "--group-col",    "group",
        ]
        te.main()
        assert (tmp_output / "svm_time_effects_summary.json").exists()

    def test_main_with_metadata(self, tmp_path, large_parquet, large_df, tmp_output):
        meta = pd.DataFrame({
            "participant_id": large_df["participant_id"].unique(),
            "site": ["A"] * large_df["participant_id"].nunique(),
        })
        meta_file = tmp_path / "meta.tsv"
        meta.to_csv(meta_file, sep="\t", index=False)
        sys.argv = [
            "te.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(tmp_output),
            "--metadata",     str(meta_file),
        ]
        te.main()
        assert (tmp_output / "svm_time_effects_summary.json").exists()

    def test_main_exits_on_missing_metrics(self, tmp_path, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(tmp_path / "ghost.parquet"),
            "--output-dir",   str(tmp_output),
        ]
        with pytest.raises(SystemExit):
            te.main()

    def test_main_exits_when_no_group_col(self, tmp_path, tmp_output):
        df = pd.DataFrame({
            "participant_id": ["s1", "s1"],
            "session":        ["ses-1", "ses-2"],
            "metric_a":       [1.0, 2.0],
        })
        pq = tmp_path / "no_group.parquet"
        df.to_parquet(pq, index=False)
        sys.argv = [
            "te.py",
            "--metrics-file", str(pq),
            "--output-dir",   str(tmp_output),
        ]
        with pytest.raises(SystemExit):
            te.main()

    def test_main_exits_when_only_one_session(self, tmp_path, tmp_output):
        df = pd.DataFrame({
            "participant_id": ["s1", "s2"],
            "session":        ["ses-1", "ses-1"],
            "group":          ["ctrl", "int"],
            "metric_a":       [1.0, 2.0],
        })
        pq = tmp_path / "one_session.parquet"
        df.to_parquet(pq, index=False)
        sys.argv = [
            "te.py",
            "--metrics-file", str(pq),
            "--output-dir",   str(tmp_output),
        ]
        with pytest.raises(SystemExit):
            te.main()

    def test_summary_json_content(self, large_parquet, tmp_output):
        sys.argv = [
            "te.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(tmp_output),
        ]
        te.main()
        summary = json.loads((tmp_output / "svm_time_effects_summary.json").read_text())
        assert 0.0 <= summary["cv_accuracy_mean"] <= 1.0
        assert 0.0 <= summary["training_accuracy"] <= 1.0
        assert summary["n_samples"] > 0

    def test_main_creates_output_dir(self, large_parquet, tmp_path):
        new_out = tmp_path / "nested" / "output"
        assert not new_out.exists()
        sys.argv = [
            "te.py",
            "--metrics-file", str(large_parquet),
            "--output-dir",   str(new_out),
        ]
        te.main()
        assert new_out.exists()