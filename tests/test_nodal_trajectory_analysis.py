"""
Tests für nodal_trajectory_analysis.py
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
traj = importlib.import_module("nodal_temporal_trajectories")

detect_group_col             = traj.detect_group_col
detect_session_col           = traj.detect_session_col
detect_metric_cols           = traj.detect_metric_cols
detect_n_nodes               = traj.detect_n_nodes
detect_groups                = traj.detect_groups
compute_nodal_trajectories   = traj.compute_nodal_trajectories
compute_intervention_effects = traj.compute_intervention_effects
compute_hub_responses        = traj.compute_hub_responses
test_hub_type_effects        = traj.test_hub_type_effects
make_plots                   = traj.make_plots
main                         = traj.main


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node_df(n_nodes=3, groups=["A", "A", "B"], sessions=[1, 2, 1]):
    records = []
    for i, (g, s) in enumerate(zip(groups, sessions)):
        for node in range(1, n_nodes + 1):
            records.append({
                "subject":   f"S{i+1}",
                "session":   s,
                "node":      node,
                "group":     g,
                "hub_type":  "peripheral",
                "degree":    float(node + i),
                "strength":  float(node * 0.5),
            })
    return pd.DataFrame(records)


def _make_trajectory_df(n_nodes=3):
    records = []
    for node in range(1, n_nodes + 1):
        for group in ["A", "B"]:
            records.append({
                "node":             node,
                "metric":           "degree",
                "group":            group,
                "slope":            0.5 if group == "A" else 0.1,
                "intercept":        1.0,
                "r_squared":        0.9,
                "change_magnitude": 1.0,
                "change_direction": "increasing",
                "n_sessions":       2,
                "mean_value":       2.0,
            })
    return pd.DataFrame(records)


def _make_cli_args(tmp_path):
    """Erstellt CLI-Argumente mit echten Pfaden und Testdaten."""
    import numpy as np, pandas as pd
    node_dir = tmp_path / "node_metrics"
    node_dir.mkdir(exist_ok=True)
    out_dir  = tmp_path / "output"
    rng = np.random.default_rng(0)
    rows = []
    for grp in ["ctrl", "int"]:
        for s in range(5):
            for ses in ["ses-1", "ses-2", "ses-3"]:
                rows.append({"participant_id": f"{grp}_sub{s:02d}",
                             "session": ses, "group": grp, "node": 0,
                             "metric_a": rng.normal(), "metric_b": rng.normal()})
    pd.DataFrame(rows).to_parquet(node_dir / "node_level_metrics.parquet", index=False)
    cli_args = ["--node-metrics-dir", str(node_dir), "--output-dir", str(out_dir)]
    return cli_args, node_dir, out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# build_config
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_config_from_cli(tmp_path):
    cli_args, node_dir, out_dir = _make_cli_args(tmp_path)
    spec = {
        "inputs":  {"node_metrics_dir": str(node_dir)},
        "outputs": {"output_dir": str(out_dir)},
    }

    with patch("sys.argv", ["script"] + cli_args):
        args   = traj.parse_args()
        config = build_config(args)

    assert config["node_metrics_dir"] == node_dir
    assert config["output_dir"]       == out_dir


def test_build_config_missing_exits(tmp_path):
    with patch("sys.argv", ["script"]):
        args = traj.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)


def test_build_config_missing_node_dir(tmp_path):
    with patch("sys.argv", ["script", "--node-metrics-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")]):
        args = traj.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_group_col
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_group_col_standard():
    df = pd.DataFrame(columns=["subject", "group", "session"])
    assert detect_group_col(df) == "group"


def test_detect_group_col_condition():
    df = pd.DataFrame(columns=["subject", "condition", "session"])
    assert detect_group_col(df) == "condition"


def test_detect_group_col_missing_exits():
    df = pd.DataFrame(columns=["subject", "session"])
    with pytest.raises(SystemExit):
        detect_group_col(df)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_session_col
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_session_col_standard():
    df = pd.DataFrame(columns=["subject", "group", "session"])
    assert detect_session_col(df) == "session"


def test_detect_session_col_timepoint():
    df = pd.DataFrame(columns=["subject", "group", "timepoint"])
    assert detect_session_col(df) == "timepoint"


def test_detect_session_col_missing_exits():
    df = pd.DataFrame(columns=["subject", "group"])
    with pytest.raises(SystemExit):
        detect_session_col(df)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_metric_cols
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_metric_cols_excludes_metadata():
    df = pd.DataFrame(columns=["subject", "session", "group", "node",
                                "hub_type", "degree", "strength"])
    df["degree"]   = 1.0
    df["strength"] = 1.0
    result = detect_metric_cols(df)
    assert "degree"   in result
    assert "strength" in result
    assert "subject"  not in result
    assert "hub_type" not in result


def test_detect_metric_cols_only_numeric():
    df = pd.DataFrame({
        "subject": ["S1"], "group": ["A"],
        "degree": [1.0], "label": ["text"]
    })
    result = detect_metric_cols(df)
    assert "degree" in result
    assert "label"  not in result


# ═══════════════════════════════════════════════════════════════════════════════
# detect_n_nodes
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_n_nodes():
    df = pd.DataFrame({"node": [1, 2, 3, 3, 2, 1]})
    assert detect_n_nodes(df) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# detect_groups
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_groups_smallest_is_control():
    df = pd.DataFrame({
        "subject": ["S1", "S2", "S3", "S4", "S5"],
        "group":   ["A",  "A",  "B",  "B",  "C"],
    })
    interv, control = detect_groups(df, "group")
    assert control == "C"
    assert "A" in interv
    assert "B" in interv


def test_detect_groups_intervention_excludes_control():
    df = pd.DataFrame({
        "subject": ["S1", "S2", "S3", "S4", "S5"],
        "group":   ["A",  "A",  "A",  "B",  "B"],
    })
    interv, control = detect_groups(df, "group")
    assert control not in interv


# ═══════════════════════════════════════════════════════════════════════════════
# compute_nodal_trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_nodal_trajectories_basic():
    df = _make_node_df(n_nodes=2,
                        groups=["A", "A", "B", "B"],
                        sessions=[1, 2, 1, 2])
    result = compute_nodal_trajectories(df, ["degree"], "group", "session", 2)
    assert len(result) > 0
    assert "slope" in result.columns
    assert "intercept" in result.columns


def test_compute_nodal_trajectories_needs_two_sessions():
    """Nur eine Session → kein Eintrag."""
    df = _make_node_df(n_nodes=2, groups=["A"], sessions=[1])
    result = compute_nodal_trajectories(df, ["degree"], "group", "session", 2)
    assert len(result) == 0


def test_compute_nodal_trajectories_direction():
    df = pd.DataFrame({
        "subject": ["S1", "S1"],
        "session": [1, 2],
        "node":    [1, 1],
        "group":   ["A", "A"],
        "degree":  [1.0, 3.0],
    })
    result = compute_nodal_trajectories(df, ["degree"], "group", "session", 1)
    assert result.iloc[0]["change_direction"] == "increasing"


def test_compute_nodal_trajectories_r_squared():
    df = pd.DataFrame({
        "subject": ["S1", "S1", "S1"],
        "session": [1, 2, 3],
        "node":    [1, 1, 1],
        "group":   ["A", "A", "A"],
        "degree":  [1.0, 2.0, 3.0],
    })
    result = compute_nodal_trajectories(df, ["degree"], "group", "session", 1)
    assert result.iloc[0]["r_squared"] == pytest.approx(1.0, abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_intervention_effects
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_intervention_effects_basic():
    traj_df = _make_trajectory_df(n_nodes=3)
    result  = compute_intervention_effects(traj_df, ["A"], "B", 3)
    assert len(result) > 0
    assert "effect_size_cohens_d" in result.columns
    assert "abs_effect_size"      in result.columns


def test_compute_intervention_effects_no_control():
    """Wenn keine Kontrollgruppe vorhanden → leeres DataFrame."""
    traj_df = _make_trajectory_df(n_nodes=2)
    result  = compute_intervention_effects(traj_df, ["A"], "X", 2)
    assert len(result) == 0


def test_compute_intervention_effects_cohens_d_direction():
    """A slope > B slope → positiver Cohen's d."""
    traj_df = pd.DataFrame([
        {"node": 1, "metric": "degree", "group": "A", "slope": 0.5},
        {"node": 1, "metric": "degree", "group": "A", "slope": 0.6},
        {"node": 1, "metric": "degree", "group": "B", "slope": 0.1},
        {"node": 1, "metric": "degree", "group": "B", "slope": 0.2},
    ])
    result = compute_intervention_effects(traj_df, ["A"], "B", 1)
    assert result.iloc[0]["effect_size_cohens_d"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# compute_hub_responses
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_hub_responses_basic():
    node_df  = _make_node_df(n_nodes=2,
                              groups=["A", "A", "B", "B"],
                              sessions=[1, 2, 1, 2])
    traj_df  = _make_trajectory_df(n_nodes=2)
    result   = compute_hub_responses(node_df, traj_df, ["A"], "B", "group", 2)
    assert "hub_type"         in result.columns
    assert "slope_difference" in result.columns


def test_compute_hub_responses_empty_if_no_hub():
    node_df = pd.DataFrame({"node": [], "hub_type": []})
    traj_df = _make_trajectory_df(n_nodes=2)
    result  = compute_hub_responses(node_df, traj_df, ["A"], "B", "group", 2)
    assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# test_hub_type_effects
# ═══════════════════════════════════════════════════════════════════════════════

def test_check_hub_type_effects_no_crash():
    hub_df = pd.DataFrame({
        "metric":           ["degree"] * 6,
        "hub_type":         ["provincial_hub"] * 3 + ["peripheral"] * 3,
        "slope_difference": [0.5, 0.6, 0.4, 0.1, 0.2, 0.1],
    })
    test_hub_type_effects(hub_df)  # darf nicht crashen


def test_check_hub_type_effects_single_group_no_crash():
    hub_df = pd.DataFrame({
        "metric":           ["degree"] * 3,
        "hub_type":         ["peripheral"] * 3,
        "slope_difference": [0.1, 0.2, 0.3],
    })
    test_hub_type_effects(hub_df)


# ═══════════════════════════════════════════════════════════════════════════════
# make_plots
# ═══════════════════════════════════════════════════════════════════════════════

def _make_effects_df(n=3):
    return pd.DataFrame({
        "node":                    list(range(1, n + 1)),
        "metric":                  ["degree"] * n,
        "effect_size_cohens_d":    [0.5, -0.3, 0.8][:n],
        "abs_effect_size":         [0.5,  0.3, 0.8][:n],
    })


def _make_hub_df():
    return pd.DataFrame({
        "metric":           ["degree"] * 4,
        "hub_type":         ["provincial_hub", "peripheral", "provincial_hub", "peripheral"],
        "slope_difference": [0.5, 0.1, 0.4, 0.2],
    })


def test_make_plots_creates_directory(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_effects_df(), _make_hub_df(), _make_trajectory_df(), plot_dir)
    assert plot_dir.exists()


def test_make_plots_heatmap(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_effects_df(), _make_hub_df(), _make_trajectory_df(), plot_dir)
    assert (plot_dir / "regional_effect_heatmap.png").exists()


def test_make_plots_per_metric_bars(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_effects_df(), _make_hub_df(), _make_trajectory_df(), plot_dir)
    assert (plot_dir / "node_effects_degree.png").exists()


def test_make_plots_hub_distributions(tmp_path):
    plot_dir = tmp_path / "plots"
    make_plots(_make_effects_df(), _make_hub_df(), _make_trajectory_df(), plot_dir)
    assert (plot_dir / "hub_type_response_distributions.png").exists()


def test_make_plots_empty_effects_no_crash(tmp_path):
    """Leere DataFrames werfen keinen Fehler."""
    plot_dir = tmp_path / "plots"
    make_plots(pd.DataFrame(columns=["node", "metric", "effect_size_cohens_d", "abs_effect_size"]),
               pd.DataFrame(columns=["metric", "hub_type", "slope_difference"]),
               pd.DataFrame(columns=["node", "metric", "group", "slope", "intercept", "n_sessions"]),
               plot_dir)
    assert plot_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs_successfully(tmp_path):
    cli_args, _, out_dir = _make_cli_args(tmp_path)

    with patch("sys.argv", ["script"] + cli_args):
        main()

    assert (out_dir / "node_trajectories.parquet").exists()
    assert (out_dir / "intervention_effect_sizes.parquet").exists()


def test_main_missing_parquet_exits(tmp_path):
    node_dir = tmp_path / "node_metrics"
    node_dir.mkdir()
    out_dir  = tmp_path / "output"
    spec = {
        "inputs":  {"node_metrics_dir": str(node_dir)},
        "outputs": {"output_dir": str(out_dir)},
    }

    with patch("sys.argv", ["script"] + cli_args):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_node_dir(tmp_path):
    with patch("sys.argv", ["script", "--node-metrics-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")]):
        with pytest.raises(SystemExit):
            main()


def test_main_with_control_group_config(tmp_path):
    """Control group aus Config wird korrekt verwendet."""
    cli_args, _, out_dir = _make_cli_args(tmp_path)
    spec = {
        "inputs":        {"node_metrics_dir": str(node_dir)},
        "outputs":       {"output_dir": str(out_dir)},
        "control_group": "B",
    }

    with patch("sys.argv", ["script"] + cli_args):
        main()

    assert (out_dir / "node_trajectories.parquet").exists()