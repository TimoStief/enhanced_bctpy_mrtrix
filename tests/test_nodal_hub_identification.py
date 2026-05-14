#!/usr/bin/env python3
"""
Zusätzliche Tests für nodal_hub_identification.py
==================================================
Diese Datei ergänzt test_nodal_hub_identification.py um die fehlende
Coverage auf 80%+ zu bringen.

Abgedeckte Bereiche:
- build_config (Pfadvalidierung, binarize)
- parse_args
- detect_metadata_columns (fehlende Spalten, Fallbacks)
- detect_file_format (mat, kein File)
- detect_n_nodes (Erfolg, Fehler)
- _load_matrix_raw (.npy, .mat, korrupt)
- find_matrix_file (alle Patterns, mat-Fallback)
- load_connectivity_matrix (None, Shape Mismatch)
- classify_hubs (alle Hub-Typen, NaN)
- compute_node_metrics (alle Metriken, binarize, Community-Fehler)
- compute_rich_club (Erfolg, Fehler)
- build_node_records / build_summary_record
- make_plots (alle Plots, ohne Gruppe)
- main() End-to-End
"""

from __future__ import annotations

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.modules.setdefault("bct", MagicMock())

import importlib
nodal = importlib.import_module("nodal_hub_identification")

detect_metadata_columns  = nodal.detect_metadata_columns
detect_file_format       = nodal.detect_file_format
detect_n_nodes           = nodal.detect_n_nodes
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
build_config             = nodal.build_config
main                     = nodal.main


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture()
def data_dir_npy(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
    np.save(d / "sub-01_ses-1_connectivity.npy", A)
    np.save(d / "sub-02_ses-1_connectivity.npy", A)
    return d


@pytest.fixture()
def metadata_tsv(tmp_path):
    meta = pd.DataFrame({
        "participant_id": ["sub-01", "sub-02"],
        "session":        ["ses-1",  "ses-1"],
        "group":          ["ctrl",   "int"],
        "sex":            ["M",      "F"],
    })
    p = tmp_path / "participants.tsv"
    meta.to_csv(p, sep="\t", index=False)
    return p


@pytest.fixture()
def mock_bct():
    import bct as _bct
    _bct.degrees_und.return_value          = np.array([2.0, 1.0, 1.0])
    _bct.strengths_und.return_value        = np.array([2.0, 1.0, 1.0])
    _bct.clustering_coef_wu.return_value   = np.array([0.5, 0.3, 0.4])
    _bct.efficiency_wei.return_value       = np.array([0.6, 0.5, 0.4])
    _bct.betweenness_wei.return_value      = np.array([1.0, 0.5, 0.5])
    _bct.community_louvain.return_value    = (np.array([0, 0, 1]), 0.3)
    _bct.participation_coef.return_value   = np.array([0.1, 0.2, 0.3])
    _bct.module_degree_zscore.return_value = np.array([3.0, 0.5, 0.1])
    _bct.rich_club_bu.return_value         = np.array([0.5, 0.6, 0.7])
    return _bct


# ===========================================================================
# parse_args / build_config
# ===========================================================================

class TestParseAndBuildConfig:
    def test_parse_required_args(self, data_dir_npy, metadata_tsv, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir_npy),
            "--metadata",   str(metadata_tsv),
            "--output-dir", str(tmp_path / "out"),
        ]
        args = nodal.parse_args()
        assert args.data_dir == str(data_dir_npy)
        assert args.binarize is False

    def test_parse_binarize_flag(self, data_dir_npy, metadata_tsv, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir_npy),
            "--metadata",   str(metadata_tsv),
            "--output-dir", str(tmp_path / "out"),
            "--binarize",
        ]
        args = nodal.parse_args()
        assert args.binarize is True

    def test_build_config_returns_dict(self, data_dir_npy, metadata_tsv, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir_npy),
            "--metadata",   str(metadata_tsv),
            "--output-dir", str(tmp_path / "out"),
        ]
        args   = nodal.parse_args()
        config = build_config(args)
        assert config["data_dir"]      == data_dir_npy.resolve()
        assert config["metadata_file"] == metadata_tsv.resolve()
        assert config["binarize"] is False

    def test_build_config_missing_data_dir_exits(self, metadata_tsv, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(tmp_path / "nope"),
            "--metadata",   str(metadata_tsv),
            "--output-dir", str(tmp_path / "out"),
        ]
        args = nodal.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)

    def test_build_config_missing_metadata_exits(self, data_dir_npy, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir_npy),
            "--metadata",   str(tmp_path / "ghost.tsv"),
            "--output-dir", str(tmp_path / "out"),
        ]
        args = nodal.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)


# ===========================================================================
# detect_metadata_columns
# ===========================================================================

class TestDetectMetadataColumns:
    def test_standard_columns(self):
        df = pd.DataFrame({
            "participant_id": ["s1"],
            "session": ["1"],
            "group": ["ctrl"],
            "sex": ["M"],
        })
        cols = detect_metadata_columns(df)
        assert cols["subject_col"] == "participant_id"
        assert cols["session_col"] == "session"

    def test_missing_session_exits(self):
        df = pd.DataFrame({"participant_id": ["s1"]})
        with pytest.raises(SystemExit):
            detect_metadata_columns(df)

    def test_missing_subject_exits(self):
        df = pd.DataFrame({"session": ["1"]})
        with pytest.raises(SystemExit):
            detect_metadata_columns(df)

    def test_partial_match_fallback(self):
        df = pd.DataFrame({
            "my_participant_id": ["s1"],
            "timepoint": ["1"],
        })
        cols = detect_metadata_columns(df)
        assert cols["subject_col"] is not None
        assert cols["session_col"] is not None

    def test_alternative_names(self):
        df = pd.DataFrame({
            "subject":   ["s1"],
            "visit":     ["1"],
            "condition": ["ctrl"],
            "gender":    ["M"],
        })
        cols = detect_metadata_columns(df)
        assert cols["group_col"] == "condition"
        assert cols["sex_col"]   == "gender"


# ===========================================================================
# detect_file_format
# ===========================================================================

class TestDetectFileFormat:
    def test_npy_detected(self, tmp_path):
        (tmp_path / "a.npy").write_bytes(b"")
        assert detect_file_format(tmp_path) == "npy"

    def test_mat_detected(self, tmp_path):
        (tmp_path / "a.mat").write_bytes(b"")
        (tmp_path / "b.mat").write_bytes(b"")
        assert detect_file_format(tmp_path) == "mat"

    def test_exits_when_no_files(self, tmp_path):
        with pytest.raises(SystemExit):
            detect_file_format(tmp_path)

    def test_npy_preferred_when_equal(self, tmp_path):
        (tmp_path / "a.npy").write_bytes(b"")
        (tmp_path / "b.mat").write_bytes(b"")
        assert detect_file_format(tmp_path) == "npy"

    def test_connectivity_mat_suffix(self, tmp_path):
        (tmp_path / "sub01_ses1.connectivity.mat").write_bytes(b"")
        assert detect_file_format(tmp_path) == "mat"


# ===========================================================================
# detect_n_nodes
# ===========================================================================

class TestDetectNNodes:
    def test_detects_from_npy(self, data_dir_npy):
        n = detect_n_nodes(data_dir_npy, "npy")
        assert n == 3

    def test_exits_when_no_files(self, tmp_path):
        with pytest.raises(SystemExit):
            detect_n_nodes(tmp_path, "npy")

    def test_exits_on_corrupt_file(self, tmp_path):
        (tmp_path / "bad.npy").write_bytes(b"garbage")
        with pytest.raises(SystemExit):
            detect_n_nodes(tmp_path, "npy")


# ===========================================================================
# _load_matrix_raw
# ===========================================================================

class TestLoadMatrixRaw:
    def test_loads_npy(self, tmp_path):
        A = np.eye(4)
        p = tmp_path / "m.npy"
        np.save(p, A)
        result = _load_matrix_raw(p, "npy")
        assert result is not None
        assert result.shape == (4, 4)

    def test_returns_none_on_corrupt_npy(self, tmp_path):
        p = tmp_path / "bad.npy"
        p.write_bytes(b"garbage")
        assert _load_matrix_raw(p, "npy") is None

    def test_returns_none_on_corrupt_mat(self, tmp_path):
        p = tmp_path / "bad.mat"
        p.write_bytes(b"garbage")
        assert _load_matrix_raw(p, "mat") is None


# ===========================================================================
# find_matrix_file
# ===========================================================================

class TestFindMatrixFile:
    def test_standard_naming(self, tmp_path):
        f = tmp_path / "01_ses-1_connectivity.npy"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "1", "npy")
        assert result == f

    def test_with_sub_prefix(self, tmp_path):
        f = tmp_path / "sub-01_ses-2_connectivity.npy"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "2", "npy")
        assert result == f

    def test_session_subdir(self, tmp_path):
        ses = tmp_path / "ses-3"
        ses.mkdir()
        f = ses / "sub-01_connectivity.npy"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "3", "npy")
        assert result == f

    def test_returns_none_when_missing(self, tmp_path):
        assert find_matrix_file(tmp_path, "sub-99", "9", "npy") is None

    def test_mat_connectivity_fallback(self, tmp_path):
        f = tmp_path / "sub-01_ses-1.connectivity.mat"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "1", "mat")
        assert result == f


# ===========================================================================
# load_connectivity_matrix
# ===========================================================================

class TestLoadConnectivityMatrix:
    def test_returns_none_when_file_missing(self, tmp_path):
        assert load_connectivity_matrix(tmp_path, "sub-99", "1", "npy", 3) is None

    def test_shape_mismatch_returns_none(self, tmp_path):
        A = np.eye(5)
        p = tmp_path / "sub-01_ses-1_connectivity.npy"
        np.save(p, A)
        assert load_connectivity_matrix(tmp_path, "sub-01", "1", "npy", 3) is None

    def test_successful_load(self, tmp_path):
        A = np.eye(3)
        p = tmp_path / "sub-01_ses-1_connectivity.npy"
        np.save(p, A)
        result = load_connectivity_matrix(tmp_path, "sub-01", "1", "npy", 3)
        assert result is not None
        assert result.shape == (3, 3)

    def test_corrupt_file_returns_none(self, tmp_path):
        p = tmp_path / "sub-01_ses-1_connectivity.npy"
        p.write_bytes(b"garbage")
        assert load_connectivity_matrix(tmp_path, "sub-01", "1", "npy", 3) is None


# ===========================================================================
# classify_hubs
# ===========================================================================

class TestClassifyHubs:
    def test_provincial_hub(self):
        pc = np.array([0.1])   # < 0.30
        wz = np.array([3.0])   # > 2.5
        result = classify_hubs(pc, wz)
        assert result[0] == "provincial_hub"

    def test_connector_hub(self):
        pc = np.array([0.5])   # > 0.30
        wz = np.array([2.0])   # > 1.0
        result = classify_hubs(pc, wz)
        assert result[0] == "connector_hub"

    def test_peripheral(self):
        pc = np.array([0.02])  # < 0.05
        wz = np.array([0.5])   # < 1.0
        result = classify_hubs(pc, wz)
        assert result[0] == "peripheral"

    def test_kinless_node(self):
        pc = np.array([0.15])
        wz = np.array([1.5])
        result = classify_hubs(pc, wz)
        assert result[0] == "kinless_node"

    def test_nan_returns_unknown(self):
        pc = np.array([np.nan])
        wz = np.array([np.nan])
        result = classify_hubs(pc, wz)
        assert result[0] == "unknown"

    def test_multiple_nodes(self):
        pc = np.array([0.1, 0.5, 0.02])
        wz = np.array([3.0, 2.0, 0.5])
        result = classify_hubs(pc, wz)
        assert len(result) == 3
        assert result[0] == "provincial_hub"
        assert result[1] == "connector_hub"
        assert result[2] == "peripheral"


# ===========================================================================
# compute_node_metrics
# ===========================================================================

class TestComputeNodeMetrics:
    def test_returns_expected_keys(self, mock_bct):
        A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
        result = compute_node_metrics(A, 3)
        for key in ["degree", "strength", "clustering", "betweenness",
                    "local_efficiency", "participation_coef",
                    "within_module_zscore", "community", "hub_type"]:
            assert key in result

    def test_shapes_match_n_nodes(self, mock_bct):
        A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
        result = compute_node_metrics(A, 3)
        assert len(result["degree"]) == 3
        assert len(result["hub_type"]) == 3

    def test_binarize_flag(self, mock_bct):
        A = np.array([[0, 0.5, 2.0],[0.5, 0, 0],[2.0, 0, 0]], dtype=float)
        result = compute_node_metrics(A, 3, binarize=True)
        assert "degree" in result

    def test_community_failure_fills_nan(self, mock_bct):
        import bct as _bct
        _bct.community_louvain.side_effect = RuntimeError("fail")
        A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
        result = compute_node_metrics(A, 3)
        assert np.all(np.isnan(result["community"]))
        assert result["modularity"] is np.nan or np.isnan(result["modularity"])
        _bct.community_louvain.side_effect = None

    def test_clustering_failure_fills_nan(self, mock_bct):
        import bct as _bct
        _bct.clustering_coef_wu.side_effect = RuntimeError("fail")
        A = np.ones((3, 3)) - np.eye(3)
        result = compute_node_metrics(A, 3)
        assert np.all(np.isnan(result["clustering"]))
        _bct.clustering_coef_wu.side_effect = None


# ===========================================================================
# compute_rich_club
# ===========================================================================

class TestComputeRichClub:
    def test_returns_array(self, mock_bct):
        A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
        result = compute_rich_club(A)
        assert result is not None

    def test_returns_none_on_exception(self):
        import bct as _bct
        _bct.rich_club_bu.side_effect = RuntimeError("fail")
        A = np.zeros((3, 3))
        result = compute_rich_club(A)
        assert result is None
        _bct.rich_club_bu.side_effect = None


# ===========================================================================
# build_node_records / build_summary_record
# ===========================================================================

class TestBuildRecords:
    def _metrics(self, n=3):
        return {
            "degree":              np.array([2.0, 1.0, 1.0]),
            "strength":            np.array([2.0, 1.0, 1.0]),
            "clustering":          np.array([0.5, 0.3, 0.4]),
            "local_efficiency":    np.array([0.6, 0.5, 0.4]),
            "betweenness":         np.array([1.0, 0.5, 0.5]),
            "participation_coef":  np.array([0.1, 0.2, 0.3]),
            "within_module_zscore": np.array([3.0, 0.5, 0.1]),
            "community":           np.array([0.0, 0.0, 1.0]),
            "hub_type":            np.array(["provincial_hub", "kinless_node", "peripheral"]),
            "modularity":          0.3,
        }

    def test_node_records_length(self):
        m = self._metrics()
        meta = pd.Series({"group": "ctrl", "sex": "M"})
        records = build_node_records("s1", "1", "AAL", meta, m, 0.5,
                                     "group", "sex", 3)
        assert len(records) == 3

    def test_node_index_starts_at_one(self):
        m = self._metrics()
        meta = pd.Series({"group": "ctrl", "sex": "M"})
        records = build_node_records("s1", "1", "AAL", meta, m, 0.5,
                                     "group", "sex", 3)
        assert records[0]["node"] == 1
        assert records[2]["node"] == 3

    def test_node_records_contain_metrics(self):
        m = self._metrics()
        meta = pd.Series({"group": "ctrl", "sex": "M"})
        records = build_node_records("s1", "1", "AAL", meta, m, 0.5,
                                     "group", "sex", 3)
        assert "degree" in records[0]
        assert "hub_type" in records[0]
        assert records[0]["group"] == "ctrl"

    def test_summary_record_keys(self):
        m = self._metrics()
        meta = pd.Series({"group": "ctrl", "sex": "M"})
        rec = build_summary_record("s1", "1", "AAL", meta, m, 0.5,
                                    "group", "sex")
        for k in ["modularity", "n_provincial_hubs", "n_connector_hubs",
                  "mean_participation", "rich_club_coef"]:
            assert k in rec

    def test_summary_hub_counts(self):
        m = self._metrics()
        meta = pd.Series({"group": "ctrl", "sex": "M"})
        rec = build_summary_record("s1", "1", "AAL", meta, m, 0.5,
                                    "group", "sex")
        assert rec["n_provincial_hubs"] == 1
        assert rec["n_peripheral"]      == 1

    def test_node_records_without_group(self):
        m = self._metrics()
        meta = pd.Series({})
        records = build_node_records("s1", "1", "AAL", meta, m, 0.5,
                                     None, None, 3)
        assert "group" not in records[0]


# ===========================================================================
# make_plots
# ===========================================================================

class TestMakePlots:
    def _summary_df(self, with_group=True):
        return pd.DataFrame({
            "subject":            ["s1", "s2", "s3", "s4"],
            "session":            ["1",  "1",  "2",  "2"],
            "n_provincial_hubs":  [1, 2, 1, 3],
            "n_connector_hubs":   [0, 1, 2, 0],
            "n_kinless_nodes":    [1, 0, 1, 2],
            "n_peripheral":       [2, 1, 0, 1],
            "mean_participation": [0.2, 0.3, 0.25, 0.15],
            "group": ["ctrl", "int", "ctrl", "int"] if with_group else ["x"]*4,
        })

    def test_creates_plot_dir(self, tmp_path):
        plot_dir = tmp_path / "plots"
        make_plots(self._summary_df(), plot_dir, "group")
        assert plot_dir.exists()

    def test_hub_counts_png_created(self, tmp_path):
        plot_dir = tmp_path / "plots"
        make_plots(self._summary_df(), plot_dir, "group")
        assert (plot_dir / "hub_counts_by_group.png").exists()

    def test_participation_png_created(self, tmp_path):
        plot_dir = tmp_path / "plots"
        make_plots(self._summary_df(), plot_dir, "group")
        assert (plot_dir / "participation_by_group.png").exists()

    def test_hub_distribution_png_created(self, tmp_path):
        plot_dir = tmp_path / "plots"
        make_plots(self._summary_df(), plot_dir, "group")
        assert (plot_dir / "hub_type_distribution.png").exists()

    def test_no_group_skips_group_plots(self, tmp_path):
        plot_dir = tmp_path / "plots"
        make_plots(self._summary_df(with_group=False), plot_dir, None)
        assert not (plot_dir / "hub_counts_by_group.png").exists()
        assert not (plot_dir / "participation_by_group.png").exists()

    def test_distribution_plot_always_created(self, tmp_path):
        plot_dir = tmp_path / "plots"
        make_plots(self._summary_df(with_group=False), plot_dir, None)
        assert (plot_dir / "hub_type_distribution.png").exists()


# ===========================================================================
# main() integration
# ===========================================================================

class TestMainIntegration:
    def _setup(self, tmp_path, mock_bct):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
        np.save(data_dir / "sub-01_ses-1_connectivity.npy", A)
        np.save(data_dir / "sub-02_ses-1_connectivity.npy", A)

        meta = pd.DataFrame({
            "participant_id": ["sub-01", "sub-02"],
            "session":        ["ses-1",  "ses-1"],
            "group":          ["ctrl",   "int"],
            "sex":            ["M",      "F"],
        })
        meta_file = tmp_path / "meta.tsv"
        meta.to_csv(meta_file, sep="\t", index=False)
        out_dir = tmp_path / "output"
        return data_dir, meta_file, out_dir

    def test_main_creates_output_files(self, tmp_path, mock_bct):
        data_dir, meta_file, out_dir = self._setup(tmp_path, mock_bct)
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
        ]
        main()
        assert (out_dir / "node_level_metrics.parquet").exists()
        assert (out_dir / "subject_hub_summaries.parquet").exists()

    def test_main_with_binarize(self, tmp_path, mock_bct):
        data_dir, meta_file, out_dir = self._setup(tmp_path, mock_bct)
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
            "--binarize",
        ]
        main()
        assert (out_dir / "node_level_metrics.parquet").exists()

    def test_main_exits_when_no_matrices(self, tmp_path, mock_bct):
        _, meta_file, out_dir = self._setup(tmp_path, mock_bct)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        sys.argv = [
            "script",
            "--data-dir",   str(empty_dir),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
        ]
        with pytest.raises(SystemExit):
            main()

    def test_main_exits_missing_data_dir(self, tmp_path, mock_bct):
        _, meta_file, out_dir = self._setup(tmp_path, mock_bct)
        sys.argv = [
            "script",
            "--data-dir",   str(tmp_path / "nope"),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
        ]
        with pytest.raises(SystemExit):
            main()

    def test_main_creates_output_dir(self, tmp_path, mock_bct):
        data_dir, meta_file, _ = self._setup(tmp_path, mock_bct)
        new_out = tmp_path / "nested" / "output"
        assert not new_out.exists()
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir),
            "--metadata",   str(meta_file),
            "--output-dir", str(new_out),
        ]
        main()
        assert new_out.exists()