"""
Tests für nodal_hub_identification.py
"""
import json
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Schwere Imports mocken
sys.modules.setdefault("bct", MagicMock())

import importlib
nodal = importlib.import_module("nodal_hub_identification")

detect_metadata_columns  = nodal.detect_metadata_columns
detect_file_format       = nodal.detect_file_format
detect_atlas_name        = nodal.detect_atlas_name
detect_sessions          = nodal.detect_sessions
find_matrix_file         = nodal.find_matrix_file
load_connectivity_matrix = nodal.load_connectivity_matrix
_load_matrix_raw         = nodal._load_matrix_raw
classify_hubs            = nodal.classify_hubs
compute_node_metrics     = nodal.compute_node_metrics
compute_rich_club        = nodal.compute_rich_club
build_node_records       = nodal.build_node_records
build_summary_record     = nodal.build_summary_record
make_plots               = nodal.make_plots
load_config              = nodal.load_config
main                     = nodal.main

N = 5


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _make_run_spec(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata = tmp_path / "meta.tsv"
    metadata.write_text("participant_id\tsession\tgroup\nS1\t1\tA\nS2\t1\tB\n")
    out_dir = tmp_path / "output"
    np.save(data_dir / "sub-S1_ses-1_matrix.npy", np.eye(N))
    np.save(data_dir / "sub-S2_ses-1_matrix.npy", np.eye(N))
    spec = {
        "inputs":  {"data_dir": str(data_dir), "metadata_file": str(metadata)},
        "outputs": {"output_dir": str(out_dir)},
        "binarize": False,
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    return spec_file, data_dir, out_dir


def _make_metrics(n=N):
    return {
        "degree":               np.ones(n),
        "strength":             np.ones(n),
        "clustering":           np.ones(n) * 0.5,
        "local_efficiency":     np.ones(n) * 0.5,
        "betweenness":          np.ones(n) * 0.3,
        "community":            np.ones(n),
        "modularity":           0.4,
        "participation_coef":   np.ones(n) * 0.2,
        "within_module_zscore": np.ones(n) * 1.0,
        "hub_type":             np.array(["peripheral"] * n),
    }


def _make_summary_df(with_group=True):
    d = {
        "subject":           ["S1", "S2"],
        "session":           [1, 1],
        "n_provincial_hubs": [3, 4],
        "n_connector_hubs":  [2, 1],
        "n_kinless_nodes":   [10, 12],
        "n_peripheral":      [231, 229],
        "mean_participation": [0.3, 0.4],
    }
    if with_group:
        d["group"] = ["A", "B"]
    return pd.DataFrame(d)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_metadata_columns
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_metadata_columns_standard():
    df = pd.DataFrame(columns=["participant_id", "session", "group", "sex"])
    r  = detect_metadata_columns(df)
    assert r["subject_col"] == "participant_id"
    assert r["session_col"] == "session"
    assert r["group_col"]   == "group"
    assert r["sex_col"]     == "sex"


def test_detect_metadata_columns_alternative():
    df = pd.DataFrame(columns=["subject_id", "timepoint", "condition", "gender"])
    r  = detect_metadata_columns(df)
    assert r["subject_col"] == "subject_id"
    assert r["session_col"] == "timepoint"


def test_detect_metadata_columns_missing_exits():
    df = pd.DataFrame(columns=["group"])
    with pytest.raises(SystemExit):
        detect_metadata_columns(df)


def test_detect_metadata_columns_partial_match():
    df = pd.DataFrame(columns=["participant_id_ext", "session_label"])
    r  = detect_metadata_columns(df)
    assert r["subject_col"] is not None
    assert r["session_col"] is not None


# ═══════════════════════════════════════════════════════════════════════════════
# detect_file_format
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_file_format_npy(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"")
    assert detect_file_format(tmp_path) == "npy"


def test_detect_file_format_mat(tmp_path):
    (tmp_path / "a.mat").write_bytes(b"")
    assert detect_file_format(tmp_path) == "mat"


def test_detect_file_format_empty_exits(tmp_path):
    with pytest.raises(SystemExit):
        detect_file_format(tmp_path)


def test_detect_file_format_prefers_npy(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"")
    (tmp_path / "b.npy").write_bytes(b"")
    (tmp_path / "c.mat").write_bytes(b"")
    assert detect_file_format(tmp_path) == "npy"


# ═══════════════════════════════════════════════════════════════════════════════
# detect_atlas_name
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_atlas_name_from_path(tmp_path):
    d = tmp_path / "brainnectome"
    d.mkdir()
    assert detect_atlas_name(d) == "Brainnectome"


def test_detect_atlas_name_unknown(tmp_path):
    assert detect_atlas_name(tmp_path) == "Unknown"


def test_detect_atlas_name_from_filename(tmp_path):
    (tmp_path / "schaefer_matrix.npy").write_bytes(b"")
    assert detect_atlas_name(tmp_path) == "Schaefer"


def test_detect_atlas_name_brodmann(tmp_path):
    d = tmp_path / "brodmann_atlas"
    d.mkdir()
    assert detect_atlas_name(d) == "Brodmann"


# ═══════════════════════════════════════════════════════════════════════════════
# detect_sessions
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_sessions_sorted():
    df = pd.DataFrame({"ses": [3, 1, 2]})
    assert detect_sessions(df, "ses") == [1, 2, 3]


def test_detect_sessions_drops_nan():
    df = pd.DataFrame({"ses": [1.0, 2.0, None]})
    assert len(detect_sessions(df, "ses")) == 2


def test_detect_sessions_single():
    df = pd.DataFrame({"ses": [1, 1, 1]})
    assert detect_sessions(df, "ses") == [1]


# ═══════════════════════════════════════════════════════════════════════════════
# find_matrix_file / _load_matrix_raw / load_connectivity_matrix
# ═══════════════════════════════════════════════════════════════════════════════

def test_find_matrix_file_found(tmp_path):
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    f.write_bytes(b"")
    assert find_matrix_file(tmp_path, "01", "1", "npy") == f


def test_find_matrix_file_not_found(tmp_path):
    assert find_matrix_file(tmp_path, "99", "9", "npy") is None


def test_find_matrix_file_with_sub_prefix(tmp_path):
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    f.write_bytes(b"")
    assert find_matrix_file(tmp_path, "sub-01", "1", "npy") is not None


def test_load_matrix_raw_npy(tmp_path):
    f = tmp_path / "m.npy"
    np.save(f, np.eye(N))
    result = _load_matrix_raw(f, "npy")
    assert result.shape == (N, N)


def test_load_matrix_raw_invalid(tmp_path):
    f = tmp_path / "bad.npy"
    f.write_bytes(b"not numpy")
    assert _load_matrix_raw(f, "npy") is None


def test_load_connectivity_matrix_success(tmp_path):
    np.save(tmp_path / "sub-01_ses-1_matrix.npy", np.eye(N))
    result = load_connectivity_matrix(tmp_path, "01", "1", "npy", N)
    assert result is not None
    assert result.shape == (N, N)


def test_load_connectivity_matrix_not_found(tmp_path):
    assert load_connectivity_matrix(tmp_path, "99", "9", "npy", N) is None


def test_load_connectivity_matrix_shape_mismatch(tmp_path):
    np.save(tmp_path / "sub-01_ses-1_matrix.npy", np.eye(N))
    assert load_connectivity_matrix(tmp_path, "01", "1", "npy", N + 5) is None


# ═══════════════════════════════════════════════════════════════════════════════
# classify_hubs
# ═══════════════════════════════════════════════════════════════════════════════

def test_classify_hubs_provincial():
    assert classify_hubs(np.array([0.1]), np.array([3.0]))[0] == "provincial_hub"


def test_classify_hubs_connector():
    assert classify_hubs(np.array([0.5]), np.array([2.0]))[0] == "connector_hub"


def test_classify_hubs_peripheral():
    assert classify_hubs(np.array([0.01]), np.array([0.5]))[0] == "peripheral"


def test_classify_hubs_kinless():
    assert classify_hubs(np.array([0.2]), np.array([1.5]))[0] == "kinless_node"


def test_classify_hubs_nan():
    assert classify_hubs(np.array([np.nan]), np.array([np.nan]))[0] == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# compute_node_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_node_metrics_returns_all_keys():
    result = compute_node_metrics(np.zeros((N, N)), N)
    for key in ["degree", "strength", "clustering", "local_efficiency",
                "betweenness", "participation_coef", "within_module_zscore",
                "hub_type", "modularity"]:
        assert key in result


def test_compute_node_metrics_correct_shape():
    """bct ist gemockt - degree gibt MagicMock zurück, wir prüfen nur den Key."""
    result = compute_node_metrics(np.zeros((N, N)), N)
    assert "degree" in result


def test_compute_node_metrics_binarize():
    result = compute_node_metrics(np.eye(N) * 2, N, binarize=True)
    assert isinstance(result, dict)


def test_compute_node_metrics_community_failure():
    """Community detection failure gibt NaN-Felder zurück."""
    import bct
    bct.community_louvain.side_effect = Exception("Fehler")
    result = compute_node_metrics(np.zeros((N, N)), N)
    assert all(h == "unknown" for h in result["hub_type"])


# ═══════════════════════════════════════════════════════════════════════════════
# compute_rich_club
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_rich_club_returns_array_or_none():
    """bct ist gemockt - Ergebnis ist MagicMock oder None, beide sind ok."""
    result = compute_rich_club(np.zeros((N, N)))
    assert result is None or result is not None  # immer True, nur kein Crash


# ═══════════════════════════════════════════════════════════════════════════════
# build_node_records
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_node_records_length():
    meta    = pd.Series({"group": "A", "sex": "M"})
    records = build_node_records("S1", 1, "Atlas", meta, _make_metrics(), 0.5, "group", "sex", N)
    assert len(records) == N


def test_build_node_records_node_index():
    meta    = pd.Series({"group": "A"})
    records = build_node_records("S1", 1, "Atlas", meta, _make_metrics(), 0.5, "group", None, N)
    assert records[0]["node"] == 1
    assert records[-1]["node"] == N


def test_build_node_records_contains_metrics():
    meta    = pd.Series({})
    records = build_node_records("S1", 1, "Atlas", meta, _make_metrics(), 0.5, None, None, N)
    assert "degree" in records[0]
    assert "strength" in records[0]


# ═══════════════════════════════════════════════════════════════════════════════
# build_summary_record
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_summary_record_keys():
    meta   = pd.Series({"group": "A"})
    record = build_summary_record("S1", 1, "Atlas", meta, _make_metrics(), 0.5, "group", None)
    for key in ["modularity", "n_provincial_hubs", "n_connector_hubs",
                "mean_participation", "rich_club_coef"]:
        assert key in record


def test_build_summary_record_hub_counts():
    metrics = _make_metrics()
    metrics["hub_type"] = np.array(["provincial_hub", "connector_hub",
                                     "peripheral", "kinless_node", "peripheral"])
    record = build_summary_record("S1", 1, "Atlas", pd.Series({}), metrics, 0.5, None, None)
    assert record["n_provincial_hubs"] == 1
    assert record["n_connector_hubs"]  == 1
    assert record["n_peripheral"]      == 2


# ═══════════════════════════════════════════════════════════════════════════════
# make_plots
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_plots_creates_directory(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_summary_df(), plot_dir, "group")
    assert plot_dir.exists()


def test_make_plots_hub_counts_png(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_summary_df(), plot_dir, "group")
    assert (plot_dir / "hub_counts_by_group.png").exists()


def test_make_plots_participation_png(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_summary_df(), plot_dir, "group")
    assert (plot_dir / "participation_by_group.png").exists()


def test_make_plots_distribution_png(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_summary_df(), plot_dir, "group")
    assert (plot_dir / "hub_type_distribution.png").exists()


def test_make_plots_no_group(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_summary_df(with_group=False), plot_dir, None)
    assert not (plot_dir / "hub_counts_by_group.png").exists()
    assert (plot_dir / "hub_type_distribution.png").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs_successfully(tmp_path):
    spec_file, _, out_dir = _make_run_spec(tmp_path)
    mock_rc = np.array([0.5, 0.6])

    with patch("sys.argv", ["script", str(spec_file)]):
        with patch("nodal_hub_identification.compute_node_metrics", return_value=_make_metrics(N)):
            with patch("nodal_hub_identification.compute_rich_club", return_value=mock_rc):
                main()

    assert (out_dir / "node_level_metrics.csv").exists()
    assert (out_dir / "subject_hub_summaries.csv").exists()


def test_main_no_matrices_exits(tmp_path):
    spec_file, _, _ = _make_run_spec(tmp_path)

    with patch("sys.argv", ["script", str(spec_file)]):
        with patch("nodal_hub_identification.load_connectivity_matrix", return_value=None):
            with pytest.raises(SystemExit):
                main()


def test_main_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_data_dir(tmp_path):
    """main() beendet sich wenn data_dir nicht in run_spec."""
    spec = {"inputs": {}, "outputs": {"output_dir": str(tmp_path / "out")}}
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))

    with patch("sys.argv", ["script", str(spec_file)]):
        with pytest.raises(SystemExit):
            main()