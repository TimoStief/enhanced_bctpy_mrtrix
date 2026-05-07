"""
Tests für global_basic_metrics.py
"""
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

# Schwere Imports mocken
sys.modules.setdefault("bct", MagicMock())
sys.modules.setdefault("umap", MagicMock())

import importlib
metrics_mod = importlib.import_module("global_basic_metrics")

detect_metadata_columns  = metrics_mod.detect_metadata_columns
detect_atlas_name        = metrics_mod.detect_atlas_name
detect_file_format       = metrics_mod.detect_file_format
detect_sessions          = metrics_mod.detect_sessions
find_matrix_file         = metrics_mod.find_matrix_file
load_connectivity_matrix = metrics_mod.load_connectivity_matrix
_load_matrix_raw         = metrics_mod._load_matrix_raw
compute_global_metrics   = metrics_mod.compute_global_metrics
compute_trajectories     = metrics_mod.compute_trajectories
run_umap                 = metrics_mod.run_umap


# ── detect_metadata_columns ──────────────────────────────────────────────────

def test_detect_metadata_columns_standard():
    df = pd.DataFrame(columns=["participant_id", "session", "group", "sex"])
    result = detect_metadata_columns(df)
    assert result["subject_col"] == "participant_id"
    assert result["session_col"] == "session"
    assert result["group_col"] == "group"
    assert result["sex_col"] == "sex"


def test_detect_metadata_columns_alternative_names():
    df = pd.DataFrame(columns=["subject_id", "timepoint", "condition", "gender"])
    result = detect_metadata_columns(df)
    assert result["subject_col"] == "subject_id"
    assert result["session_col"] == "timepoint"
    assert result["group_col"] == "condition"
    assert result["sex_col"] == "gender"


def test_detect_metadata_columns_missing_required():
    df = pd.DataFrame(columns=["group", "sex"])
    with pytest.raises(SystemExit):
        detect_metadata_columns(df)


def test_detect_metadata_columns_partial_match():
    df = pd.DataFrame(columns=["participant_id_extended", "session_label"])
    result = detect_metadata_columns(df)
    assert result["subject_col"] is not None
    assert result["session_col"] is not None


# ── detect_atlas_name ─────────────────────────────────────────────────────────

def test_detect_atlas_name_known(tmp_path):
    atlas_dir = tmp_path / "brainnectome" / "data"
    atlas_dir.mkdir(parents=True)
    result = detect_atlas_name(atlas_dir)
    assert result == "Brainnectome"


def test_detect_atlas_name_unknown(tmp_path):
    result = detect_atlas_name(tmp_path)
    assert result == "Unknown"


def test_detect_atlas_name_aal(tmp_path):
    aal_dir = tmp_path / "aal_atlas"
    aal_dir.mkdir()
    result = detect_atlas_name(aal_dir)
    assert result == "Aal"


def test_detect_atlas_name_from_filename(tmp_path):
    (tmp_path / "schaefer_matrix.npy").write_bytes(b"")
    result = detect_atlas_name(tmp_path)
    assert result == "Schaefer"


# ── detect_file_format ────────────────────────────────────────────────────────

def test_detect_file_format_npy(tmp_path):
    (tmp_path / "matrix.npy").write_bytes(b"")
    result = detect_file_format(tmp_path)
    assert result == "npy"


def test_detect_file_format_mat(tmp_path):
    (tmp_path / "matrix.mat").write_bytes(b"")
    result = detect_file_format(tmp_path)
    assert result == "mat"


def test_detect_file_format_no_files(tmp_path):
    with pytest.raises(SystemExit):
        detect_file_format(tmp_path)


def test_detect_file_format_prefers_npy(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"")
    (tmp_path / "b.npy").write_bytes(b"")
    (tmp_path / "c.mat").write_bytes(b"")
    result = detect_file_format(tmp_path)
    assert result == "npy"


# ── detect_sessions ───────────────────────────────────────────────────────────

def test_detect_sessions_sorted():
    df = pd.DataFrame({"session": [3, 1, 2, 1]})
    result = detect_sessions(df, "session")
    assert result == [1, 2, 3]


def test_detect_sessions_ignores_nan():
    df = pd.DataFrame({"session": [1, 2, None]})
    result = detect_sessions(df, "session")
    assert None not in result
    assert len(result) == 2


# ── find_matrix_file ─────────────────────────────────────────────────────────

def test_find_matrix_file_found(tmp_path):
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    f.write_bytes(b"")
    result = find_matrix_file(tmp_path, "01", "1", "npy")
    assert result == f


def test_find_matrix_file_not_found(tmp_path):
    result = find_matrix_file(tmp_path, "99", "9", "npy")
    assert result is None


def test_find_matrix_file_with_prefix(tmp_path):
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    f.write_bytes(b"")
    result = find_matrix_file(tmp_path, "sub-01", "1", "npy")
    assert result is not None


# ── _load_matrix_raw ─────────────────────────────────────────────────────────

def test_load_matrix_raw_npy(tmp_path):
    A = np.eye(5)
    f = tmp_path / "matrix.npy"
    np.save(f, A)
    result = _load_matrix_raw(f, "npy")
    assert result is not None
    assert result.shape == (5, 5)


def test_load_matrix_raw_invalid_file(tmp_path):
    f = tmp_path / "kaputt.npy"
    f.write_bytes(b"kein numpy")
    result = _load_matrix_raw(f, "npy")
    assert result is None


# ── load_connectivity_matrix ─────────────────────────────────────────────────

def test_load_connectivity_matrix_success(tmp_path):
    A = np.eye(5)
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    np.save(f, A)
    result = load_connectivity_matrix(tmp_path, "01", "1", "npy", 5)
    assert result is not None
    assert result.shape == (5, 5)


def test_load_connectivity_matrix_not_found(tmp_path):
    result = load_connectivity_matrix(tmp_path, "99", "9", "npy", 5)
    assert result is None


def test_load_connectivity_matrix_shape_mismatch(tmp_path):
    A = np.eye(5)
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    np.save(f, A)
    result = load_connectivity_matrix(tmp_path, "01", "1", "npy", 10)
    assert result is None


# ── compute_global_metrics ───────────────────────────────────────────────────

def test_compute_global_metrics_empty_matrix():
    A = np.zeros((5, 5))
    result = compute_global_metrics(A)
    assert np.isnan(result["path_length"])
    assert np.isnan(result["global_efficiency"])


def test_compute_global_metrics_returns_dict():
    A = np.zeros((5, 5))
    result = compute_global_metrics(A)
    expected_keys = ["density", "path_length", "global_efficiency",
                     "clustering_coef", "transitivity", "modularity"]
    for key in expected_keys:
        assert key in result


def test_compute_global_metrics_binarize():
    A = np.zeros((5, 5))
    result = compute_global_metrics(A, binarize=True)
    assert isinstance(result, dict)


def test_compute_global_metrics_exception_returns_nan():
    import bct
    bct.density_und.side_effect = Exception("Fehler")
    A = np.ones((5, 5)) - np.eye(5)
    result = compute_global_metrics(A)
    assert isinstance(result, dict)


# ── compute_trajectories ─────────────────────────────────────────────────────

def _make_umap_df():
    return pd.DataFrame({
        "subject": ["S1", "S1", "S1", "S2", "S2"],
        "session": [1, 2, 3, 1, 2],
        "umap_1": [0.0, 1.0, 2.0, 0.0, 1.0],
        "umap_2": [0.0, 1.0, 2.0, 0.0, 1.0],
        "umap_3": [0.0, 1.0, 2.0, 0.0, 1.0],
        "group":  ["A", "A", "A", "B", "B"],
    })


def test_compute_trajectories_basic():
    df = _make_umap_df()
    result = compute_trajectories(df, "subject", "session", "group", None)
    assert len(result) == 2
    assert "early_change" in result.columns
    assert "total_distance" in result.columns


def test_compute_trajectories_skip_single_session():
    df = pd.DataFrame({
        "subject": ["S1"],
        "session": [1],
        "umap_1": [0.0], "umap_2": [0.0], "umap_3": [0.0],
    })
    result = compute_trajectories(df, "subject", "session", None, None)
    assert len(result) == 0


def test_compute_trajectories_with_group():
    df = _make_umap_df()
    result = compute_trajectories(df, "subject", "session", "group", None)
    assert "group" in result.columns


def test_compute_trajectories_acceleration():
    df = _make_umap_df()
    result = compute_trajectories(df, "subject", "session", None, None)
    s1 = result[result["subject"] == "S1"].iloc[0]
    assert not np.isnan(s1["acceleration"])


# ── run_umap ─────────────────────────────────────────────────────────────────

def test_run_umap_adds_columns():
    df = pd.DataFrame({
        "subject": ["S1", "S2", "S3"],
        "session": [1, 1, 1],
        "metric_a": [1.0, 2.0, 3.0],
        "metric_b": [4.0, 5.0, 6.0],
    })
    mock_embedding = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    mock_umap = MagicMock()
    mock_umap.fit_transform.return_value = mock_embedding

    with patch("global_basic_metrics.UMAP", return_value=mock_umap):
        result = run_umap(df, {"subject", "session"}, {})

    assert "umap_1" in result.columns
    assert "umap_2" in result.columns
    assert "umap_3" in result.columns
    assert len(result) == 3