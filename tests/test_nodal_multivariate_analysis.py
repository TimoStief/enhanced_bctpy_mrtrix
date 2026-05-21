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

load_config                = comp.load_config
detect_columns             = comp.detect_columns
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
    records = []
    groups  = ["alone_2w", "alone_4w", "group_2w", "group_4w", "control"]
    for g in groups:
        for i in range(n_per_group):
            for node in range(1, n_nodes + 1):
                records.append({
                    "subject":  f"S_{g}_{i}",
                    "session":  1,
                    "node":     node,
                    "group":    g,
                    "sex":      "M" if i % 2 == 0 else "F",
                    "age":      float(20 + i),
                    "hub_type": "peripheral",
                    "degree":   float(node + i),
                    "strength": float(node * 0.5 + i),
                })
    return pd.DataFrame(records)


def _make_run_spec(tmp_path, n_nodes=3, n_per_group=5):
    node_dir = tmp_path / "node_metrics"
    node_dir.mkdir()
    out_dir  = tmp_path / "output"
    node_df  = _make_node_df(n_nodes=n_nodes, n_per_group=n_per_group)
    node_df.to_parquet(node_dir / "node_level_metrics.parquet", index=False)
    spec = {
        "inputs":  {"node_metrics_dir": str(node_dir)},
        "outputs": {"output_dir": str(out_dir)},
        "control_group": "control",
        "alone_groups":  ["alone_2w", "alone_4w"],
        "group_groups":  ["group_2w", "group_4w"],
        "short_groups":  ["alone_2w", "group_2w"],
        "long_groups":   ["alone_4w", "group_4w"],
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    return spec_file, node_dir, out_dir


def _make_results(empty=False):
    if empty:
        return {k: pd.DataFrame() for k in
                ["five_group", "social", "duration", "binary", "gender", "age"]}
    df = pd.DataFrame({
        "node":                [1, 2, 3],
        "metric":              ["degree", "degree", "strength"],
        "significant":         [True, False, True],
        "eta_squared":         [0.2, 0.05, 0.15],
        "abs_cohens_d":        [0.9, 0.3, 0.6],
        "abs_r_pearson":       [0.4, 0.2, 0.5],
        "significant_pearson": [True, False, True],
    })
    return {k: df.copy() for k in
            ["five_group", "social", "duration", "binary", "gender", "age"]}


# ═══════════════════════════════════════════════════════════════════════════════
# load_config
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_config_from_run_spec(tmp_path):
    spec_file, node_dir, out_dir = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script", str(spec_file)]):
        args   = comp.parse_args()
        config = load_config(args)
    assert config["node_metrics_dir"] == node_dir
    assert config["control_group"]    == "control"


def test_load_config_missing_exits(tmp_path):
    with patch("sys.argv", ["script"]):
        args = comp.parse_args()
        with pytest.raises(SystemExit):
            load_config(args)


def test_load_config_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        args = comp.parse_args()
        with pytest.raises(SystemExit):
            load_config(args)


def test_load_config_cli_args(tmp_path):
    spec_file, node_dir, out_dir = _make_run_spec(tmp_path)
    with patch("sys.argv", ["script",
                             "--node-metrics-dir", str(node_dir),
                             "--output-dir",       str(out_dir)]):
        args   = comp.parse_args()
        config = load_config(args)
    assert config["node_metrics_dir"] == node_dir


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


def test_detect_columns_alternative_names():
    df = pd.DataFrame({
        "participant_id": ["S1"],
        "timepoint":      [1],
        "condition":      ["ctrl"],
        "gender":         ["M"],
        "degree":         [1.0],
    })
    r = detect_columns(df)
    assert r["subject_col"] == "participant_id"
    assert r["session_col"] == "timepoint"
    assert r["group_col"]   == "condition"
    assert r["sex_col"]     == "gender"


# ═══════════════════════════════════════════════════════════════════════════════
# detect_n_nodes
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_n_nodes():
    df = pd.DataFrame({"node": [1, 2, 3, 2, 1]})
    assert detect_n_nodes(df) == 3


def test_detect_n_nodes_single():
    df = pd.DataFrame({"node": [1, 1, 1]})
    assert detect_n_nodes(df) == 1


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


def test_normalize_sex_mixed():
    df = pd.DataFrame({"sex": [1, "F", 2]})
    result = normalize_sex(df, "sex")
    assert "M" in result["sex"].values or "F" in result["sex"].values


# ═══════════════════════════════════════════════════════════════════════════════
# run_five_group_anova
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_five_group_anova_returns_df():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    cols   = detect_columns(df)
    config = {"control_group": "control"}
    from group_detection import detect_or_ask_groups
    groups = detect_or_ask_groups(df, "group", config)
    result = run_five_group_anova(df, ["degree"], "group", groups, 2)
    assert isinstance(result, pd.DataFrame)


def test_run_five_group_anova_has_keys():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    config = {"control_group": "control"}
    from group_detection import detect_or_ask_groups
    groups = detect_or_ask_groups(df, "group", config)
    result = run_five_group_anova(df, ["degree"], "group", groups, 2)
    if len(result) > 0:
        assert "f_statistic" in result.columns
        assert "eta_squared" in result.columns
        assert "significant" in result.columns


def test_run_five_group_anova_significant_bool():
    df     = _make_node_df(n_nodes=2, n_per_group=5)
    config = {"control_group": "control"}
    from group_detection import detect_or_ask_groups
    groups = detect_or_ask_groups(df, "group", config)
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


def test_run_ttest_analysis_empty_groups():
    df = _make_node_df(n_nodes=2, n_per_group=5)
    result = run_ttest_analysis(df, ["degree"], "group", [], [], "a", "b", 2)
    assert len(result) == 0


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
        assert "r_pearson"           in result.columns
        assert "r_spearman"          in result.columns
        assert "significant_pearson" in result.columns


def test_run_age_correlations_min_10_samples():
    df = pd.DataFrame({
        "subject": list(range(5)),
        "node":    [1] * 5,
        "age":     [20.0, 21.0, 22.0, 23.0, 24.0],
        "degree":  [1.0,  2.0,  3.0,  4.0,  5.0],
    })
    result = run_age_correlations(df, ["degree"], "age", 1)
    assert len(result) == 0


def test_run_age_correlations_direction():
    records = [{"subject": i, "node": 1, "age": float(i), "degree": float(i)}
               for i in range(20)]
    df = pd.DataFrame(records)
    result = run_age_correlations(df, ["degree"], "age", 1)
    if len(result) > 0:
        assert result.iloc[0]["r_pearson"] > 0


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


def test_make_effect_distributions_single_metric(tmp_path):
    make_effect_distributions(_make_results(), ["degree"], tmp_path)
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


def test_print_summary_partial_results():
    results = _make_results()
    results["age"] = pd.DataFrame()
    print_summary(results)


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def test_main_runs_successfully(tmp_path):
    spec_file, _, out_dir = _make_run_spec(tmp_path, n_nodes=2, n_per_group=5)
    with patch("sys.argv", ["script", str(spec_file)]):
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
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    with patch("sys.argv", ["script", str(spec_file)]):
        with pytest.raises(SystemExit):
            main()


def test_main_missing_run_spec(tmp_path):
    with patch("sys.argv", ["script", str(tmp_path / "nope.json")]):
        with pytest.raises(SystemExit):
            main()


def test_main_saves_all_outputs(tmp_path):
    spec_file, _, out_dir = _make_run_spec(tmp_path, n_nodes=2, n_per_group=5)
    with patch("sys.argv", ["script", str(spec_file)]):
        main()
    for f in ["five_group_anova.parquet", "social_effects.parquet",
              "duration_effects.parquet", "intervention_vs_control.parquet"]:
        assert (out_dir / f).exists(), f"Missing: {f}"


def test_main_creates_plots(tmp_path):
    spec_file, _, out_dir = _make_run_spec(tmp_path, n_nodes=2, n_per_group=5)
    with patch("sys.argv", ["script", str(spec_file)]):
        main()
    plot_dir = out_dir / "plots"
    assert plot_dir.exists()
    assert (plot_dir / "significance_heatmaps.png").exists()


def test_main_skips_age_without_col(tmp_path):
    node_dir = tmp_path / "node_metrics"
    node_dir.mkdir()
    df = _make_node_df().drop(columns=["age"])
    df.to_parquet(node_dir / "node_level_metrics.parquet", index=False)
    spec = {
        "inputs":  {"node_metrics_dir": str(node_dir)},
        "outputs": {"output_dir": str(tmp_path / "out")},
        "control_group": "control",
        "alone_groups":  ["alone_2w", "alone_4w"],
        "group_groups":  ["group_2w", "group_4w"],
        "short_groups":  ["alone_2w", "group_2w"],
        "long_groups":   ["alone_4w", "group_4w"],
    }
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))
    with patch("sys.argv", ["script", str(spec_file)]):
        main()
    assert (tmp_path / "out" / "five_group_anova.parquet").exists()