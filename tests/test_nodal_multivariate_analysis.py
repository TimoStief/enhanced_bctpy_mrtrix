"""
Tests für nodal_multivariate_analysis.py
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
comp = importlib.import_module("nodal_multivariate_analysis")

detect_columns             = comp.detect_columns
detect_groups              = comp.detect_groups
detect_n_nodes             = comp.detect_n_nodes
normalize_sex              = comp.normalize_sex
run_five_group_anova       = comp.run_five_group_anova
run_ttest_analysis         = comp.run_ttest_analysis
run_age_correlations       = comp.run_age_correlations
make_significance_heatmaps = comp.make_significance_heatmaps
make_effect_distributions  = comp.make_effect_distributions
make_top_nodes_summary     = comp.make_top_nodes_summary
print_summary              = comp.print_summary
main                       = comp.main


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node_df(n_nodes=3, n_per_group=5):
    """Erstellt einen minimalen node_df mit mehreren Gruppen und Nodes."""
    records = []
    groups  = ["alone_2w", "alone_4w", "group_2w", "group_4w", "control"]
    for g in groups:
        for i in range(n_per_group):
            for node in range(1, n_nodes + 1):
                records.append({
                    "subject": f"S_{g}_{i}",
                    "session": 1,
                    "node":    node,
                    "group":   g,
                    "sex":     "M" if i % 2 == 0 else "F",
                    "age":     float(20 + i),
                    "hub_type": "peripheral",
                    "degree":   float(node + i),
                    "strength": float(node * 0.5 + i),
                })
    return pd.DataFrame(records)


def _make_cli_args(tmp_path, n_nodes=3, n_per_group=5):
    """Erstellt CLI-Argumente mit echten Pfaden und Testdaten."""
    import numpy as np, pandas as pd
    node_dir = tmp_path / "node_metrics"
    node_dir.mkdir(exist_ok=True)
    out_dir  = tmp_path / "output"
    rng = np.random.default_rng(0)
    rows = []
    for grp in ["ctrl", "int"]:
        for s in range(n_per_group):
            for nd in range(n_nodes):
                rows.append({"participant_id": f"{grp}_sub{s:02d}",
                             "session": "ses-1", "group": grp, "node": nd,
                             "metric_a": rng.normal(), "metric_b": rng.normal()})
    pd.DataFrame(rows).to_parquet(node_dir / "node_level_metrics.parquet", index=False)
    cli_args = ["--node-metrics-dir", str(node_dir), "--output-dir", str(out_dir)]
    return cli_args, node_dir, out_dir


def _make_results(empty=False):
    if empty:
        return {k: pd.DataFrame() for k in
                ["five_group", "social", "duration", "binary", "gender", "age"]}
    df = pd.DataFrame({
        "node":         [1, 2, 3],
        "metric":       ["degree", "degree", "strength"],
        "significant":  [True, False, True],
        "eta_squared":  [0.2, 0.05, 0.15],
        "abs_cohens_d": [0.9, 0.3, 0.6],
        "abs_r_pearson":[0.4, 0.2, 0.5],
        "significant_pearson": [True, False, True],
    })
    return {k: df.copy() for k in
            ["five_group", "social", "duration", "binary", "gender", "age"]}


# ═══════════════════════════════════════════════════════════════════════════════
# build_config
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_config_from_cli(tmp_path):
    cli_args, _, out_dir = _make_cli_args(tmp_path)
    with patch("sys.argv", ["script"] + cli_args):
        args   = comp.parse_args()
        config = build_config(args)
    assert config["node_metrics_dir"] == node_dir
    assert config["control_group"]    == "control"


def test_build_config_missing_exits(tmp_path):
    with patch("sys.argv", ["script"]):
        args = comp.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)


def test_build_config_missing_node_dir(tmp_path):
    with patch("sys.argv", ["script", "--node-metrics-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")]):
        args = comp.parse_args()
        with pytest.raises(SystemExit):
            build_config(args)


# ═══════════════════════════════════════════════════════════════════════════════
# detect_columns
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_columns_standard():
    df = _make_node_df()
    r  = detect_columns(df)
    assert r["subject_col"] == "subject"
    assert r["session_col"] == "session"
    assert r["group_col"]   == "group"
    assert r["sex_col"]     == "sex"
    assert r["age_col"]     == "age"


def test_detect_columns_metric_cols_numeric_only():
    df = _make_node_df()
    r  = detect_columns(df)
    assert "degree"   in r["metric_cols"]
    assert "strength" in r["metric_cols"]
    assert "group"    not in r["metric_cols"]
    assert "hub_type" not in r["metric_cols"]


def test_detect_columns_no_age():
    df = _make_node_df().drop(columns=["age"])
    r  = detect_columns(df)
    assert r["age_col"] is None


def test_detect_columns_no_sex():
    df = _make_node_df().drop(columns=["sex"])
    r  = detect_columns(df)
    assert r["sex_col"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# detect_groups
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_groups_control_from_config():
    df     = _make_node_df()
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    assert groups["control"] == "control"
    assert "control" not in groups["intervention"]


def test_detect_groups_alone_social_from_labels():
    df     = _make_node_df()
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    assert "alone_2w" in groups["alone"]
    assert "group_2w" in groups["social"]


def test_detect_groups_short_long_from_labels():
    df     = _make_node_df()
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    assert "alone_2w" in groups["short"]
    assert "alone_4w" in groups["long"]


def test_detect_groups_fallback_split(tmp_path):
    """Wenn keine Labels erkennbar → Fallback Split."""
    df = pd.DataFrame({
        "subject": ["S1"] * 10 + ["S2"] * 10 + ["S3"] * 2,
        "node":    [1] * 22,
        "group":   ["A"] * 10 + ["B"] * 10 + ["C"] * 2,
    })
    config = {"control_group": "C"}
    groups = detect_groups(df, "group", config)
    assert len(groups["alone"]) > 0 or len(groups["short"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# detect_n_nodes
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_n_nodes():
    df = pd.DataFrame({"node": [1, 2, 3, 2, 1]})
    assert detect_n_nodes(df) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# normalize_sex
# ═══════════════════════════════════════════════════════════════════════════════

def test_normalize_sex_numeric_to_string():
    df = pd.DataFrame({"sex": [1, 2, 1]})
    result = normalize_sex(df, "sex")
    assert set(result["sex"].unique()) == {"M", "F"}


def test_normalize_sex_already_string():
    df = pd.DataFrame({"sex": ["M", "F", "M"]})
    result = normalize_sex(df, "sex")
    assert set(result["sex"].unique()) == {"M", "F"}


def test_normalize_sex_no_col():
    df = pd.DataFrame({"group": ["A", "B"]})
    result = normalize_sex(df, None)
    assert "sex" not in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# run_five_group_anova
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_five_group_anova_returns_df():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    result = run_five_group_anova(df, ["degree"], "group", groups, 2)
    assert isinstance(result, pd.DataFrame)


def test_run_five_group_anova_has_keys():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    result = run_five_group_anova(df, ["degree"], "group", groups, 2)
    if len(result) > 0:
        assert "f_statistic" in result.columns
        assert "eta_squared" in result.columns
        assert "significant" in result.columns


def test_run_five_group_anova_significant_bool():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    config = {"control_group": "control"}
    groups = detect_groups(df, "group", config)
    result = run_five_group_anova(df, ["degree"], "group", groups, 2)
    if len(result) > 0:
        assert result["significant"].dtype == bool


# ═══════════════════════════════════════════════════════════════════════════════
# run_ttest_analysis
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_ttest_analysis_returns_df():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    result = run_ttest_analysis(df, ["degree"], "group",
                                 ["alone_2w", "alone_4w"],
                                 ["group_2w", "group_4w"],
                                 "alone", "group", 2)
    assert isinstance(result, pd.DataFrame)


def test_run_ttest_analysis_has_cohens_d():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    result = run_ttest_analysis(df, ["degree"], "group",
                                 ["alone_2w", "alone_4w"],
                                 ["group_2w", "group_4w"],
                                 "alone", "group", 2)
    if len(result) > 0:
        assert "cohens_d"     in result.columns
        assert "abs_cohens_d" in result.columns
        assert "significant"  in result.columns


def test_run_ttest_analysis_insufficient_data():
    """Weniger als 3 Datenpunkte → leeres DataFrame."""
    df = pd.DataFrame({
        "subject": ["S1"], "node": [1], "group": ["A"], "degree": [1.0]
    })
    result = run_ttest_analysis(df, ["degree"], "group", ["A"], ["B"], "a", "b", 1)
    assert len(result) == 0


def test_run_ttest_analysis_direction():
    records = []
    for i in range(5):
        records.append({"subject": f"S{i}", "node": 1, "group": "A", "degree": 10.0 + i})
        records.append({"subject": f"S{i}", "node": 1, "group": "B", "degree":  1.0 + i})
    df     = pd.DataFrame(records)
    result = run_ttest_analysis(df, ["degree"], "group", ["A"], ["B"], "a", "b", 1)
    if len(result) > 0:
        assert result.iloc[0]["cohens_d"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# run_age_correlations
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_age_correlations_returns_df():
    df     = _make_node_df(n_nodes=2, n_per_group=10)
    result = run_age_correlations(df, ["degree"], "age", 2)
    assert isinstance(result, pd.DataFrame)


def test_run_age_correlations_has_keys():
    df     = _make_node_df(n_nodes=2, n_per_group=10)
    result = run_age_correlations(df, ["degree"], "age", 2)
    if len(result) > 0:
        assert "r_pearson"         in result.columns
        assert "r_spearman"        in result.columns
        assert "significant_pearson" in result.columns


def test_run_age_correlations_min_10_samples():
    """Weniger als 10 Samples → kein Eintrag."""
    df = pd.DataFrame({
        "subject": list(range(5)),
        "node":    [1] * 5,
        "age":     [20.0, 21.0, 22.0, 23.0, 24.0],
        "degree":  [1.0,  2.0,  3.0,  4.0,  5.0],
    })
    result = run_age_correlations(df, ["degree"], "age", 1)
    assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# make_significance_heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_significance_heatmaps_creates_file(tmp_path):
    make_significance_heatmaps(_make_results(), tmp_path)
    assert (tmp_path / "significance_heatmaps.png").exists()


def test_make_significance_heatmaps_empty_no_crash(tmp_path):
    make_significance_heatmaps(_make_results(empty=True), tmp_path)
    assert (tmp_path / "significance_heatmaps.png").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# make_effect_distributions
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_effect_distributions_creates_file(tmp_path):
    make_effect_distributions(_make_results(), ["degree", "strength"], tmp_path)
    assert (tmp_path / "effect_size_distributions.png").exists()


def test_make_effect_distributions_empty_no_crash(tmp_path):
    make_effect_distributions(_make_results(empty=True), ["degree"], tmp_path)
    assert (tmp_path / "effect_size_distributions.png").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# make_top_nodes_summary
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_top_nodes_summary_creates_file(tmp_path):
    make_top_nodes_summary(_make_results(), tmp_path)
    assert (tmp_path / "top_nodes_summary.png").exists()


def test_make_top_nodes_summary_empty_no_crash(tmp_path):
    make_top_nodes_summary(_make_results(empty=True), tmp_path)
    assert (tmp_path / "top_nodes_summary.png").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# print_summary
# ═══════════════════════════════════════════════════════════════════════════════

def test_print_summary_no_crash():
    print_summary(_make_results())


def test_print_summary_empty_no_crash():
    print_summary(_make_results(empty=True))


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs_successfully(tmp_path):
    cli_args, _, out_dir = _make_cli_args(tmp_path, n_nodes=2, n_per_group=5)
    with patch("sys.argv", ["script"] + cli_args):
        main()
    assert (out_dir / "five_group_anova.parquet").exists()
    assert (out_dir / "social_effects.parquet").exists()
    assert (out_dir / "intervention_vs_control.parquet").exists()


def test_main_missing_parquet_exits(tmp_path):
    node_dir = tmp_path / "node_metrics"
    node_dir.mkdir()
    spec = {
        "inputs":  {"node_metrics_dir": str(node_dir)},
        "outputs": {"output_dir": str(tmp_path / "out")},
    }
    with patch("sys.argv", ["script"] + cli_args):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_node_dir(tmp_path):
    with patch("sys.argv", ["script", "--node-metrics-dir", str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")]):
        with pytest.raises(SystemExit):
            main()


def test_main_saves_all_outputs(tmp_path):
    cli_args, _, out_dir = _make_cli_args(tmp_path, n_nodes=2, n_per_group=5)
    with patch("sys.argv", ["script"] + cli_args):
        main()
    expected = [
        "five_group_anova.parquet",
        "social_effects.parquet",
        "duration_effects.parquet",
        "intervention_vs_control.parquet",
    ]
    for f in expected:
        assert (out_dir / f).exists(), f"Missing: {f}"