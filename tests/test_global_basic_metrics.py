"""
Tests für global_basic_metrics.py
"""
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, str(Path(__file__).parent.parent))

# Schwere Imports mocken bevor das Skript geladen wird
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
make_plots               = metrics_mod.make_plots
load_config              = metrics_mod.load_config
main                     = metrics_mod.main


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _make_umap_df(with_group=True):
    d = {
        "subject": ["S1", "S1", "S1", "S2", "S2"],
        "session": [1, 2, 3, 1, 2],
        "umap_1":  [0.0, 1.0, 2.0, 0.0, 1.0],
        "umap_2":  [0.0, 1.0, 2.0, 0.0, 1.0],
        "umap_3":  [0.0, 1.0, 2.0, 0.0, 1.0],
    }
    if with_group:
        d["group"] = ["A", "A", "A", "B", "B"]
    return pd.DataFrame(d)


def _make_results_df(with_group=True):
    d = {
        "subject": ["S1", "S2"],
        "session": [1, 1],
        "path_length": [1.5, 2.0],
        "global_efficiency": [0.6, 0.5],
        "clustering_coef": [0.4, 0.3],
        "small_worldness": [0.8, 0.7],
    }
    if with_group:
        d["group"] = ["A", "B"]
    return pd.DataFrame(d)


def _make_trajectory_df(with_group=True):
    d = {
        "subject": ["S1", "S2"],
        "early_change": [0.5, 0.8],
        "late_change": [0.3, 0.6],
        "total_distance": [1.0, 1.5],
        "acceleration": [0.6, 0.75],
    }
    if with_group:
        d["group"] = ["A", "B"]
    return pd.DataFrame(d)


def _make_run_spec(tmp_path, **overrides):
    """Erstellt eine vollständige run_spec.json mit echten Pfaden."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata = tmp_path / "meta.tsv"
    metadata.write_text("participant_id\tsession\tgroup\nS1\t1\tA\nS2\t1\tB\n")
    out_dir = tmp_path / "output"

    # Eine echte .npy Matrix erstellen
    A = np.eye(5)
    np.save(data_dir / "sub-S1_ses-1_matrix.npy", A)
    np.save(data_dir / "sub-S2_ses-1_matrix.npy", A)

    spec = {
        "inputs": {
            "data_dir": str(data_dir),
            "metadata_file": str(metadata),
        },
        "outputs": {"output_dir": str(out_dir)},
        "binarize": False,
    }
    spec.update(overrides)
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    return spec_file, data_dir, metadata, out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# detect_metadata_columns
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# detect_atlas_name
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_atlas_name_known_in_path(tmp_path):
    atlas_dir = tmp_path / "brainnectome" / "data"
    atlas_dir.mkdir(parents=True)
    assert detect_atlas_name(atlas_dir) .lower() == "brainnectome"


def test_detect_atlas_name_unknown(tmp_path):
    assert detect_atlas_name(tmp_path) == "Unknown"


def test_detect_atlas_name_aal(tmp_path):
    (tmp_path / "aal_atlas").mkdir()
    assert detect_atlas_name(tmp_path / "aal_atlas") .lower() == "aal"


def test_detect_atlas_name_from_filename(tmp_path):
    (tmp_path / "schaefer_matrix.npy").write_bytes(b"")
    assert detect_atlas_name(tmp_path) .lower() == "schaefer"


def test_detect_atlas_name_hcp(tmp_path):
    hcp_dir = tmp_path / "hcp_data"
    hcp_dir.mkdir()
    assert detect_atlas_name(hcp_dir) .lower() == "hcp"


# ═══════════════════════════════════════════════════════════════════════════════
# detect_file_format
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_file_format_npy(tmp_path):
    (tmp_path / "matrix.npy").write_bytes(b"")
    assert detect_file_format(tmp_path) == "npy"


def test_detect_file_format_mat(tmp_path):
    (tmp_path / "matrix.mat").write_bytes(b"")
    assert detect_file_format(tmp_path) == "mat"


def test_detect_file_format_no_files(tmp_path):
    with pytest.raises(SystemExit):
        detect_file_format(tmp_path)


def test_detect_file_format_prefers_npy_when_more(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"")
    (tmp_path / "b.npy").write_bytes(b"")
    (tmp_path / "c.mat").write_bytes(b"")
    assert detect_file_format(tmp_path) == "npy"


# ═══════════════════════════════════════════════════════════════════════════════
# detect_sessions
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_sessions_sorted():
    df = pd.DataFrame({"session": [3, 1, 2, 1]})
    assert detect_sessions(df, "session") == [1, 2, 3]


def test_detect_sessions_ignores_nan():
    df = pd.DataFrame({"session": [1.0, 2.0, None]})
    result = detect_sessions(df, "session")
    assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# find_matrix_file
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


def test_find_matrix_file_mat_fallback(tmp_path):
    """Findet .connectivity.mat als Fallback."""
    f = tmp_path / "sub-01_ses-1.connectivity.mat"
    f.write_bytes(b"")
    result = find_matrix_file(tmp_path, "01", "1", "mat")
    assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# _load_matrix_raw
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_matrix_raw_npy(tmp_path):
    A = np.eye(5)
    f = tmp_path / "matrix.npy"
    np.save(f, A)
    result = _load_matrix_raw(f, "npy")
    assert result is not None
    assert result.shape == (5, 5)


def test_load_matrix_raw_invalid(tmp_path):
    f = tmp_path / "kaputt.npy"
    f.write_bytes(b"kein numpy")
    assert _load_matrix_raw(f, "npy") is None


# ═══════════════════════════════════════════════════════════════════════════════
# load_connectivity_matrix
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_connectivity_matrix_success(tmp_path):
    A = np.eye(5)
    np.save(tmp_path / "sub-01_ses-1_matrix.npy", A)
    result = load_connectivity_matrix(tmp_path, "01", "1", "npy", 5)
    assert result is not None
    assert result.shape == (5, 5)


def test_load_connectivity_matrix_not_found(tmp_path):
    assert load_connectivity_matrix(tmp_path, "99", "9", "npy", 5) is None


def test_load_connectivity_matrix_shape_mismatch(tmp_path):
    np.save(tmp_path / "sub-01_ses-1_matrix.npy", np.eye(5))
    assert load_connectivity_matrix(tmp_path, "01", "1", "npy", 10) is None


# ═══════════════════════════════════════════════════════════════════════════════
# compute_global_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_global_metrics_zero_matrix_returns_nan():
    result = compute_global_metrics(np.zeros((5, 5)))
    assert np.isnan(result["path_length"])
    assert np.isnan(result["global_efficiency"])


def test_compute_global_metrics_returns_all_keys():
    result = compute_global_metrics(np.zeros((5, 5)))
    for key in ["density", "path_length", "global_efficiency",
                "clustering_coef", "transitivity", "modularity",
                "n_communities", "small_worldness"]:
        assert key in result


def test_compute_global_metrics_binarize_no_error():
    result = compute_global_metrics(np.zeros((5, 5)), binarize=True)
    assert isinstance(result, dict)


def test_compute_global_metrics_exception_returns_nan_dict():
    import bct
    bct.density_und.side_effect = Exception("Fehler")
    result = compute_global_metrics(np.ones((5, 5)) - np.eye(5))
    assert isinstance(result, dict)
    assert "path_length" in result


# ═══════════════════════════════════════════════════════════════════════════════
# compute_trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_trajectories_basic():
    result = compute_trajectories(_make_umap_df(), "subject", "session", "group", None)
    assert len(result) == 2
    assert "early_change" in result.columns
    assert "total_distance" in result.columns


def test_compute_trajectories_skips_single_session():
    df = pd.DataFrame({
        "subject": ["S1"], "session": [1],
        "umap_1": [0.0], "umap_2": [0.0], "umap_3": [0.0],
    })
    assert len(compute_trajectories(df, "subject", "session", None, None)) == 0


def test_compute_trajectories_includes_group():
    result = compute_trajectories(_make_umap_df(), "subject", "session", "group", None)
    assert "group" in result.columns


def test_compute_trajectories_acceleration_three_sessions():
    result = compute_trajectories(_make_umap_df(), "subject", "session", None, None)
    s1 = result[result["subject"] == "S1"].iloc[0]
    assert not np.isnan(s1["acceleration"])


def test_compute_trajectories_no_group():
    df = _make_umap_df(with_group=False)
    result = compute_trajectories(df, "subject", "session", None, None)
    assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# run_umap
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_umap_adds_umap_columns():
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

    assert all(c in result.columns for c in ["umap_1", "umap_2", "umap_3"])
    assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# make_plots
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_plots_creates_directory(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_results_df(), _make_umap_df(), _make_trajectory_df(),
               plot_dir, "group", "subject")
    assert plot_dir.exists()


def test_make_plots_with_group_creates_metrics_plot(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_results_df(), _make_umap_df(), _make_trajectory_df(),
               plot_dir, "group", "subject")
    assert (plot_dir / "metrics_by_group.png").exists()


def test_make_plots_creates_umap_plot(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_results_df(), _make_umap_df(), _make_trajectory_df(),
               plot_dir, "group", "subject")
    assert (plot_dir / "umap_3d.png").exists()


def test_make_plots_creates_trajectory_plot(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_results_df(), _make_umap_df(), _make_trajectory_df(),
               plot_dir, "group", "subject")
    assert (plot_dir / "trajectory_scatter.png").exists()


def test_make_plots_no_group_skips_metrics_plot(tmp_path):
    """Ohne group_col wird kein metrics_by_group.png erstellt."""
    plot_dir = tmp_path / "plots"
    make_plots(_make_results_df(with_group=False), _make_umap_df(with_group=False),
               _make_trajectory_df(with_group=False), plot_dir, None, "subject")
    assert not (plot_dir / "metrics_by_group.png").exists()


def test_make_plots_empty_trajectory(tmp_path):
    """Leerer Trajectory DataFrame wirft keinen Fehler."""
    plot_dir = tmp_path / "plots"
    make_plots(_make_results_df(), _make_umap_df(), pd.DataFrame(),
               plot_dir, "group", "subject")
    assert plot_dir.exists()


def test_make_plots_umap_without_group(tmp_path):
    """UMAP Plot ohne Gruppeninfo funktioniert."""
    plot_dir = tmp_path / "plots"
    umap_df = _make_umap_df(with_group=False)
    make_plots(_make_results_df(with_group=False), umap_df,
               pd.DataFrame(), plot_dir, None, "subject")
    assert (plot_dir / "umap_3d.png").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs_successfully(tmp_path):
    """main() läuft komplett durch mit gemockten bct/UMAP Funktionen."""
    spec_file, data_dir, metadata, out_dir = _make_run_spec(tmp_path)

    mock_metrics = {
        "density": 0.5, "path_length": 1.5, "global_efficiency": 0.6,
        "clustering_coef": 0.4, "transitivity": 0.3, "modularity": 0.2,
        "n_communities": 3, "participation_coef_mean": 0.5,
        "local_efficiency_mean": 0.4, "betweenness_mean": 0.3,
        "small_worldness": 0.8,
    }
    mock_embedding = np.random.rand(2, 3)
    mock_umap = MagicMock()
    mock_umap.fit_transform.return_value = mock_embedding

    with patch("sys.argv", ["script", str(spec_file)]):
        with patch("global_basic_metrics.compute_global_metrics", return_value=mock_metrics):
            with patch("global_basic_metrics.UMAP", return_value=mock_umap):
                main()

    assert (out_dir / "global_metrics.csv").exists()
    assert (out_dir / "global_metrics.parquet").exists()


def test_main_no_matrices_exits(tmp_path):
    """main() beendet sich wenn keine Matrizen geladen werden können."""
    spec_file, _, _, _ = _make_run_spec(tmp_path)

    with patch("sys.argv", ["script", str(spec_file)]):
        with patch("global_basic_metrics.load_connectivity_matrix", return_value=None):
            with pytest.raises(SystemExit):
                main()


def test_main_missing_run_spec(tmp_path):
    """main() beendet sich wenn run_spec.json nicht existiert."""
    with patch("sys.argv", ["script", str(tmp_path / "nicht_vorhanden.json")]):
        with pytest.raises(SystemExit):
            main()

# ═══════════════════════════════════════════════════════════════════════════
# Additional coverage tests
# ═══════════════════════════════════════════════════════════════════════════

def test_progress_function(capsys):
    metrics_mod._progress(5, 10, "Test")
    out = capsys.readouterr().out
    assert "5/10" in out or "50%" in out


def test_normalize_matrix_auto_returns_string(tmp_path):
    A = np.random.rand(5, 5) * 5000
    np.save(tmp_path / "mat.npy", A)
    result = metrics_mod.ask_normalize(tmp_path, "npy", 5, "log")
    assert result == "log"


def test_detect_matrix_type_all_branches():
    # fiber counts - need >2 unique values and max > 1000
    A = np.random.rand(5, 5) * 5000 + 100
    r = metrics_mod.detect_matrix_type(A)
    assert r["type"] == "fiber_counts"

    # weighted unnormalized (max > 1 and max < 1000, not binary)
    A2 = np.random.rand(5, 5) * 50 + 2
    r2 = metrics_mod.detect_matrix_type(A2)
    assert r2["type"] == "weighted_unnormalized"


def test_normalize_all_branches():
    A = np.array([[0, 2.0], [2.0, 0]])
    assert metrics_mod.normalize_matrix(A, "log").max() <= 1.0
    assert metrics_mod.normalize_matrix(A, "max").max() == pytest.approx(1.0)
    assert set(metrics_mod.normalize_matrix(A, "binary").flatten()) <= {0.0, 1.0}
    np.testing.assert_array_equal(metrics_mod.normalize_matrix(A, "none"), A)


def test_load_config_cli_paths(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    meta = tmp_path / "meta.tsv"
    meta.write_text("participant_id\tsession\nS1\t1\n")
    out_dir = tmp_path / "out"
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["prog", "--data-dir", str(data_dir),
                 "--metadata", str(meta),
                 "--output-dir", str(out_dir)]
    args = metrics_mod.parse_args()
    config = metrics_mod.load_config(args)
    assert config["data_dir"] == data_dir
    _sys.argv = old


def test_load_config_missing_exits():
    import sys as _sys
    old = _sys.argv
    _sys.argv = ["prog"]
    args = metrics_mod.parse_args()
    with pytest.raises(SystemExit):
        metrics_mod.load_config(args)
    _sys.argv = old


def test_find_matrix_file_not_found(tmp_path):
    result = metrics_mod.find_matrix_file(tmp_path, "sub-01", "1", "npy")
    assert result is None


def test_find_matrix_file_found(tmp_path):
    f = tmp_path / "sub-01_ses-1_matrix.npy"
    f.write_bytes(b"")
    result = metrics_mod.find_matrix_file(tmp_path, "sub-01", "1", "npy")
    assert result is not None


def test_compute_global_metrics_binary():
    from unittest.mock import patch, MagicMock
    A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    with patch("global_basic_metrics.bct") as mock_bct:
        mock_bct.distance_wei.return_value = (np.ones((3,3)), None)
        mock_bct.clustering_coef_wu.return_value = np.array([0.5,0.5,0.5])
        mock_bct.transitivity_wu.return_value = 0.5
        mock_bct.community_louvain.return_value = (np.array([1,1,2]), 0.3)
        mock_bct.participation_coef.return_value = np.array([0.3,0.3,0.3])
        mock_bct.efficiency_wei.return_value = np.array([0.4,0.4,0.4])
        mock_bct.betweenness_wei.return_value = np.array([0.2,0.2,0.2])
        result = metrics_mod.compute_global_metrics(A, binarize=False)
    assert "density" in result
    assert "clustering_coef" in result


def test_compute_global_metrics_binarize():
    from unittest.mock import patch
    A = np.array([[0,2,1],[2,0,3],[1,3,0]], dtype=float)
    with patch("global_basic_metrics.bct") as mock_bct:
        mock_bct.distance_wei.return_value = (np.ones((3,3)), None)
        mock_bct.clustering_coef_wu.return_value = np.ones(3) * 0.5
        mock_bct.transitivity_wu.return_value = 0.4
        mock_bct.community_louvain.return_value = (np.ones(3), 0.2)
        mock_bct.participation_coef.return_value = np.ones(3) * 0.3
        mock_bct.efficiency_wei.return_value = np.ones(3) * 0.4
        mock_bct.betweenness_wei.return_value = np.ones(3) * 0.1
        result = metrics_mod.compute_global_metrics(A, binarize=True)
    assert "density" in result


def test_main_runs_with_normalize_mocked(tmp_path):
    from unittest.mock import patch, MagicMock
    spec_file, data_dir, metadata, out_dir = _make_run_spec(tmp_path)
    mock_metrics = {
        "density": 0.5, "path_length": 1.5, "global_efficiency": 0.6,
        "clustering_coef": 0.4, "transitivity": 0.3, "modularity": 0.2,
        "n_communities": 3, "participation_coef_mean": 0.5,
        "local_efficiency_mean": 0.4, "betweenness_mean": 0.3,
        "small_worldness": 0.8,
    }
    mock_umap = MagicMock()
    mock_umap.fit_transform.return_value = np.random.rand(2, 3)
    with patch("sys.argv", ["script", str(spec_file)]):
        with patch("global_basic_metrics.compute_global_metrics", return_value=mock_metrics):
            with patch("global_basic_metrics.UMAP", return_value=mock_umap):
                with patch("global_basic_metrics.ask_normalize", return_value="none"):
                    main()


def test_main_no_matrices_with_normalize_mocked(tmp_path):
    from unittest.mock import patch
    spec_file, _, _, _ = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(spec_file)]):
        with patch("global_basic_metrics.load_connectivity_matrix", return_value=None):
            with patch("global_basic_metrics.ask_normalize", return_value="none"):
                with pytest.raises(SystemExit):
                    main()
