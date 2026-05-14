#!/usr/bin/env python3
"""
Zusätzliche Tests für global_basic_metrics.py
===============================================
Diese Datei ergänzt test_global_basic_metrics.py um die fehlende Coverage
auf 80%+ zu bringen.

Abgedeckte Bereiche:
- build_config (Pfadvalidierung, binarize-Flag)
- parse_args
- detect_metadata_columns (missing session_col exits)
- detect_file_format (mat bevorzugt, kein File exits)
- detect_n_nodes (erfolgreich, Fehler)
- _load_matrix_raw (.mat Pfad, None-Key)
- find_matrix_file (alle Naming Conventions, .mat Fallback)
- load_connectivity_matrix (None File, Shape Mismatch, Load Error)
- compute_global_metrics (alle Metriken, binarize, Exception)
- run_umap (vollständiger Durchlauf)
- compute_trajectories (Acceleration, Group/Sex, Single Session)
- make_plots (alle Plot-Typen, ohne Gruppe)
- main() (vollständiger End-to-End Durchlauf)
"""

from __future__ import annotations

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.modules.setdefault("bct", MagicMock())
sys.modules.setdefault("umap", MagicMock())
sys.modules["umap"].UMAP = MagicMock()

import importlib
metrics_mod = importlib.import_module("global_basic_metrics")

detect_metadata_columns  = metrics_mod.detect_metadata_columns
detect_atlas_name        = metrics_mod.detect_atlas_name
detect_file_format       = metrics_mod.detect_file_format
detect_n_nodes           = metrics_mod.detect_n_nodes
detect_sessions          = metrics_mod.detect_sessions
find_matrix_file         = metrics_mod.find_matrix_file
load_connectivity_matrix = metrics_mod.load_connectivity_matrix
_load_matrix_raw         = metrics_mod._load_matrix_raw
compute_global_metrics   = metrics_mod.compute_global_metrics
compute_trajectories     = metrics_mod.compute_trajectories
run_umap                 = metrics_mod.run_umap
make_plots               = metrics_mod.make_plots
build_config             = metrics_mod.build_config
main                     = metrics_mod.main


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture()
def meta_df():
    return pd.DataFrame({
        "participant_id": ["sub-01", "sub-02", "sub-03"],
        "session":        ["ses-1",  "ses-1",  "ses-2"],
        "group":          ["ctrl",   "int",    "ctrl"],
        "sex":            ["M",      "F",      "M"],
    })


@pytest.fixture()
def data_dir_npy(tmp_path):
    """Data directory with 3×3 .npy matrices."""
    d = tmp_path / "data"
    d.mkdir()
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
    np.save(d / "sub-01_ses-1_connectivity.npy", A)
    np.save(d / "sub-02_ses-1_connectivity.npy", A)
    np.save(d / "sub-03_ses-2_connectivity.npy", A)
    return d


@pytest.fixture()
def metadata_tsv(tmp_path, meta_df):
    p = tmp_path / "participants.tsv"
    meta_df.to_csv(p, sep="\t", index=False)
    return p


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
        args = metrics_mod.parse_args()
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
        args = metrics_mod.parse_args()
        assert args.binarize is True

    def test_build_config_returns_dict(self, data_dir_npy, metadata_tsv, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir_npy),
            "--metadata",   str(metadata_tsv),
            "--output-dir", str(tmp_path / "out"),
        ]
        args = metrics_mod.parse_args()
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
        args = metrics_mod.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)

    def test_build_config_missing_metadata_exits(self, data_dir_npy, tmp_path):
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir_npy),
            "--metadata",   str(tmp_path / "ghost.tsv"),
            "--output-dir", str(tmp_path / "out"),
        ]
        args = metrics_mod.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)


# ===========================================================================
# detect_metadata_columns
# ===========================================================================

class TestDetectMetadataColumns:
    def test_missing_session_col_exits(self):
        df = pd.DataFrame({"participant_id": ["s1"], "group": ["ctrl"]})
        with pytest.raises(SystemExit):
            detect_metadata_columns(df)

    def test_missing_subject_col_exits(self):
        df = pd.DataFrame({"session": ["ses-1"], "group": ["ctrl"]})
        with pytest.raises(SystemExit):
            detect_metadata_columns(df)

    def test_partial_match_fallback(self):
        df = pd.DataFrame({
            "my_participant_id": ["s1"],
            "my_session_num":    ["1"],
        })
        cols = detect_metadata_columns(df)
        assert cols["subject_col"] is not None
        assert cols["session_col"] is not None

    def test_alternative_names(self):
        df = pd.DataFrame({
            "subject":   ["s1"],
            "timepoint": ["1"],
            "condition": ["ctrl"],
            "gender":    ["M"],
        })
        cols = detect_metadata_columns(df)
        assert cols["subject_col"] == "subject"
        assert cols["session_col"] == "timepoint"
        assert cols["group_col"]   == "condition"
        assert cols["sex_col"]     == "gender"


# ===========================================================================
# detect_file_format
# ===========================================================================

class TestDetectFileFormat:
    def test_prefers_mat_when_more(self, tmp_path):
        (tmp_path / "a.mat").write_bytes(b"")
        (tmp_path / "b.mat").write_bytes(b"")
        (tmp_path / "c.npy").write_bytes(b"")
        fmt = detect_file_format(tmp_path)
        assert fmt == "mat"

    def test_connectivity_mat_suffix(self, tmp_path):
        (tmp_path / "sub-01_ses-1.connectivity.mat").write_bytes(b"")
        fmt = detect_file_format(tmp_path)
        assert fmt == "mat"

    def test_exits_when_no_files(self, tmp_path):
        with pytest.raises(SystemExit):
            detect_file_format(tmp_path)

    def test_npy_preferred_when_equal(self, tmp_path):
        (tmp_path / "a.npy").write_bytes(b"")
        (tmp_path / "b.mat").write_bytes(b"")
        fmt = detect_file_format(tmp_path)
        assert fmt == "npy"


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

    def test_exits_when_matrix_load_fails(self, tmp_path):
        (tmp_path / "bad.npy").write_bytes(b"not a numpy file")
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
        result = _load_matrix_raw(p, "npy")
        assert result is None

    def test_returns_none_on_corrupt_mat(self, tmp_path):
        p = tmp_path / "bad.mat"
        p.write_bytes(b"garbage")
        result = _load_matrix_raw(p, "mat")
        assert result is None


# ===========================================================================
# find_matrix_file
# ===========================================================================

class TestFindMatrixFile:
    def test_finds_standard_naming(self, tmp_path):
        f = tmp_path / "01_ses-1_connectivity.npy"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "1", "npy")
        assert result == f

    def test_finds_with_sub_prefix(self, tmp_path):
        f = tmp_path / "sub-01_ses-2_connectivity.npy"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "2", "npy")
        assert result == f

    def test_finds_in_session_subdir(self, tmp_path):
        ses_dir = tmp_path / "ses-3"
        ses_dir.mkdir()
        f = ses_dir / "sub-01_connectivity.npy"
        f.write_bytes(b"")
        result = find_matrix_file(tmp_path, "sub-01", "3", "npy")
        assert result == f

    def test_returns_none_when_missing(self, tmp_path):
        result = find_matrix_file(tmp_path, "sub-99", "9", "npy")
        assert result is None

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
        result = load_connectivity_matrix(tmp_path, "sub-99", "1", "npy", 3)
        assert result is None

    def test_shape_mismatch_returns_none(self, tmp_path):
        A = np.eye(5)
        p = tmp_path / "sub-01_ses-1_connectivity.npy"
        np.save(p, A)
        result = load_connectivity_matrix(tmp_path, "sub-01", "1", "npy", 3)
        assert result is None

    def test_corrupt_file_returns_none(self, tmp_path):
        p = tmp_path / "sub-01_ses-1_connectivity.npy"
        p.write_bytes(b"garbage")
        result = load_connectivity_matrix(tmp_path, "sub-01", "1", "npy", 3)
        assert result is None

    def test_successful_load(self, tmp_path):
        A = np.eye(3)
        p = tmp_path / "sub-01_ses-1_connectivity.npy"
        np.save(p, A)
        result = load_connectivity_matrix(tmp_path, "sub-01", "1", "npy", 3)
        assert result is not None
        assert result.shape == (3, 3)


# ===========================================================================
# compute_global_metrics
# ===========================================================================

class TestComputeGlobalMetrics:
    def _mock_bct(self):
        import bct as _bct
        _bct.density_und.return_value     = (0.5,)
        _bct.distance_wei.return_value    = (np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),)
        _bct.clustering_coef_wu.return_value = np.array([0.5, 0.3, 0.4])
        _bct.transitivity_wu.return_value = 0.4
        _bct.community_louvain.return_value = (np.array([0, 0, 1]), 0.3)
        _bct.participation_coef.return_value = np.array([0.1, 0.2, 0.3])
        _bct.efficiency_wei.return_value  = np.array([0.5, 0.4, 0.6])
        _bct.betweenness_wei.return_value = np.array([1.0, 2.0, 1.5])

    def test_returns_nan_for_zero_matrix(self):
        A = np.zeros((3, 3))
        result = compute_global_metrics(A)
        assert np.isnan(result["density"])

    def test_returns_all_expected_keys(self):
        self._mock_bct()
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        result = compute_global_metrics(A)
        expected = ["density", "path_length", "global_efficiency", "clustering_coef",
                    "transitivity", "modularity", "n_communities",
                    "participation_coef_mean", "local_efficiency_mean",
                    "betweenness_mean", "small_worldness"]
        for k in expected:
            assert k in result

    def test_binarize_flag_applied(self):
        self._mock_bct()
        A = np.array([[0, 0.5, 2.0], [0.5, 0, 0], [2.0, 0, 0]], dtype=float)
        result = compute_global_metrics(A, binarize=True)
        assert isinstance(result, dict)

    def test_exception_returns_nan_dict(self):
        import bct as _bct
        _bct.density_und.side_effect = RuntimeError("bct error")
        A = np.ones((3, 3))
        result = compute_global_metrics(A)
        assert np.isnan(result["density"])
        _bct.density_und.side_effect = None


# ===========================================================================
# run_umap
# ===========================================================================

class TestRunUmap:
    def test_adds_umap_columns(self):
        import umap as _umap
        mock_embedding = np.random.default_rng(0).normal(size=(6, 3))
        _umap.UMAP.return_value.fit_transform.return_value = mock_embedding

        df = pd.DataFrame({
            "subject":            ["s1"] * 3 + ["s2"] * 3,
            "session":            ["1", "2", "3"] * 2,
            "global_efficiency":  np.random.default_rng(0).normal(size=6),
            "clustering_coef":    np.random.default_rng(1).normal(size=6),
        })
        meta_cols = {"subject", "session"}
        result = run_umap(df, meta_cols, {})
        assert "umap_1" in result.columns
        assert "umap_2" in result.columns
        assert "umap_3" in result.columns

    def test_handles_nan_features(self):
        import umap as _umap
        mock_embedding = np.zeros((4, 3))
        _umap.UMAP.return_value.fit_transform.return_value = mock_embedding

        df = pd.DataFrame({
            "subject": ["s1", "s2", "s3", "s4"],
            "session": ["1", "1", "2", "2"],
            "metric":  [1.0, np.nan, 2.0, 3.0],
        })
        result = run_umap(df, {"subject", "session"}, {})
        assert len(result) == 4


# ===========================================================================
# compute_trajectories
# ===========================================================================

class TestComputeTrajectories:
    def _umap_df(self, with_group=True, n_sessions=3):
        rows = []
        for subj in ["s1", "s2"]:
            grp = "ctrl" if subj == "s1" else "int"
            for i in range(n_sessions):
                r = {
                    "subject": subj,
                    "session": str(i + 1),
                    "umap_1": float(i),
                    "umap_2": float(i),
                    "umap_3": float(i),
                }
                if with_group:
                    r["group"] = grp
                    r["sex"]   = "M"
                rows.append(r)
        return pd.DataFrame(rows)

    def test_basic_output(self):
        df = self._umap_df()
        result = compute_trajectories(df, "subject", "session", "group", "sex")
        assert len(result) == 2
        assert "total_distance" in result.columns

    def test_skips_single_session(self):
        df = self._umap_df(n_sessions=1)
        result = compute_trajectories(df, "subject", "session", "group", "sex")
        assert len(result) == 0

    def test_acceleration_computed_for_3_sessions(self):
        df = self._umap_df(n_sessions=3)
        result = compute_trajectories(df, "subject", "session", "group", "sex")
        assert "acceleration" in result.columns

    def test_group_col_preserved(self):
        df = self._umap_df()
        result = compute_trajectories(df, "subject", "session", "group", "sex")
        assert "group" in result.columns

    def test_without_group_col(self):
        df = self._umap_df(with_group=False)
        result = compute_trajectories(df, "subject", "session", None, None)
        assert "total_distance" in result.columns
        assert "group" not in result.columns

    def test_two_session_no_acceleration(self):
        df = self._umap_df(n_sessions=2)
        result = compute_trajectories(df, "subject", "session", "group", "sex")
        assert result["acceleration"].isna().all()


# ===========================================================================
# make_plots
# ===========================================================================

class TestMakePlots:
    def _make_dfs(self, with_group=True):
        results_df = pd.DataFrame({
            "subject":           ["s1", "s2", "s3", "s4"],
            "session":           ["1",  "1",  "2",  "2"],
            "path_length":       [1.2, 1.3, 1.1, 1.4],
            "global_efficiency": [0.8, 0.7, 0.9, 0.6],
            "clustering_coef":   [0.5, 0.4, 0.6, 0.3],
            "small_worldness":   [1.5, 1.3, 1.7, 1.2],
            "group":             ["ctrl", "int", "ctrl", "int"] if with_group else ["x"]*4,
        })
        umap_df = pd.DataFrame({
            "subject": ["s1", "s2", "s3", "s4"],
            "session": ["1",  "1",  "2",  "2"],
            "umap_1":  [0.1, 0.5, 0.2, 0.6],
            "umap_2":  [0.3, 0.7, 0.4, 0.8],
            "umap_3":  [0.0, 0.1, 0.0, 0.1],
            "group":   ["ctrl", "int", "ctrl", "int"] if with_group else ["x"]*4,
        })
        traj_df = pd.DataFrame({
            "subject":       ["s1", "s2"],
            "early_change":  [0.5, 0.3],
            "late_change":   [0.6, 0.2],
            "total_distance":[1.1, 0.5],
            "group":         ["ctrl", "int"] if with_group else ["x", "x"],
        })
        return results_df, umap_df, traj_df

    def test_creates_plot_dir(self, tmp_path):
        results_df, umap_df, traj_df = self._make_dfs()
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, traj_df, plot_dir, "group", "subject")
        assert plot_dir.exists()

    def test_metrics_by_group_created(self, tmp_path):
        results_df, umap_df, traj_df = self._make_dfs()
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, traj_df, plot_dir, "group", "subject")
        assert (plot_dir / "metrics_by_group.png").exists()

    def test_umap_3d_created(self, tmp_path):
        results_df, umap_df, traj_df = self._make_dfs()
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, traj_df, plot_dir, "group", "subject")
        assert (plot_dir / "umap_3d.png").exists()

    def test_trajectory_scatter_created(self, tmp_path):
        results_df, umap_df, traj_df = self._make_dfs()
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, traj_df, plot_dir, "group", "subject")
        assert (plot_dir / "trajectory_scatter.png").exists()

    def test_no_group_skips_metrics_plot(self, tmp_path):
        results_df, umap_df, traj_df = self._make_dfs(with_group=False)
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, traj_df, plot_dir, None, "subject")
        assert not (plot_dir / "metrics_by_group.png").exists()

    def test_empty_trajectory_skips_scatter(self, tmp_path):
        results_df, umap_df, _ = self._make_dfs()
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, pd.DataFrame(), plot_dir, "group", "subject")
        assert not (plot_dir / "trajectory_scatter.png").exists()

    def test_umap_without_group(self, tmp_path):
        results_df, umap_df, traj_df = self._make_dfs(with_group=False)
        umap_df = umap_df.drop(columns=["group"])
        plot_dir = tmp_path / "plots"
        make_plots(results_df, umap_df, traj_df, plot_dir, None, "subject")
        assert (plot_dir / "umap_3d.png").exists()


# ===========================================================================
# main() integration
# ===========================================================================

class TestMainIntegration:
    def _setup(self, tmp_path):
        import bct as _bct
        _bct.density_und.return_value         = (0.5,)
        _bct.distance_wei.return_value        = (np.array([[0,1,2],[1,0,1],[2,1,0]]),)
        _bct.clustering_coef_wu.return_value  = np.array([0.5, 0.3, 0.4])
        _bct.transitivity_wu.return_value     = 0.4
        _bct.community_louvain.return_value   = (np.array([0, 0, 1]), 0.3)
        _bct.participation_coef.return_value  = np.array([0.1, 0.2, 0.3])
        _bct.efficiency_wei.return_value      = np.array([0.5, 0.4, 0.6])
        _bct.betweenness_wei.return_value     = np.array([1.0, 2.0, 1.5])

        import umap as _umap
        _umap.UMAP.return_value.fit_transform.return_value = np.zeros((3, 3))

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        A = np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float)
        np.save(data_dir / "sub-01_ses-1_connectivity.npy", A)
        np.save(data_dir / "sub-02_ses-1_connectivity.npy", A)
        np.save(data_dir / "sub-03_ses-1_connectivity.npy", A)

        meta = pd.DataFrame({
            "participant_id": ["sub-01", "sub-02", "sub-03"],
            "session":        ["ses-1",  "ses-1",  "ses-1"],
            "group":          ["ctrl",   "int",    "ctrl"],
            "sex":            ["M",      "F",      "M"],
        })
        meta_file = tmp_path / "meta.tsv"
        meta.to_csv(meta_file, sep="\t", index=False)
        out_dir = tmp_path / "output"

        return data_dir, meta_file, out_dir

    def test_main_creates_output_files(self, tmp_path):
        data_dir, meta_file, out_dir = self._setup(tmp_path)
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
        ]
        main()
        assert (out_dir / "global_metrics.parquet").exists()
        assert (out_dir / "global_metrics.csv").exists()

    def test_main_with_binarize(self, tmp_path):
        data_dir, meta_file, out_dir = self._setup(tmp_path)
        sys.argv = [
            "script",
            "--data-dir",   str(data_dir),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
            "--binarize",
        ]
        main()
        assert (out_dir / "global_metrics.parquet").exists()

    def test_main_exits_when_no_matrices(self, tmp_path):
        _, meta_file, out_dir = self._setup(tmp_path)
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

    def test_main_exits_missing_data_dir(self, tmp_path):
        _, meta_file, out_dir = self._setup(tmp_path)
        sys.argv = [
            "script",
            "--data-dir",   str(tmp_path / "nope"),
            "--metadata",   str(meta_file),
            "--output-dir", str(out_dir),
        ]
        with pytest.raises(SystemExit):
            main()