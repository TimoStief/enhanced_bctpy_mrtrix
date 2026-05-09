"""
Guardrail tests for nodal analysis auto-detection and run_spec validation.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent


def _install_dependency_stubs() -> dict:
    """Install lightweight stubs for optional scientific plotting/stats deps."""
    managed = [
        "matplotlib",
        "matplotlib.pyplot",
        "seaborn",
        "scipy",
        "scipy.stats",
    ]
    previous = {name: sys.modules.get(name) for name in managed}

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *args, **kwargs: None
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
    sys.modules["seaborn"] = types.ModuleType("seaborn")

    scipy_mod = types.ModuleType("scipy")
    stats_mod = types.ModuleType("scipy.stats")

    def _pair(*args, **kwargs):
        return 0.0, 1.0

    stats_mod.f_oneway = _pair
    stats_mod.ttest_ind = _pair
    stats_mod.pearsonr = _pair
    stats_mod.spearmanr = _pair
    scipy_mod.stats = stats_mod

    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.stats"] = stats_mod
    return previous


def _restore_dependency_modules(previous: dict) -> None:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_module(file_path: Path, module_name: str):
    previous = _install_dependency_stubs()
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_dependency_modules(previous)


temporal_mod = _load_module(
    ROOT / "3_nodal_network_metrics" / "02_nodal_temporal_trajectories.py",
    "nodal_temporal_guardrails",
)

multivar_mod = _load_module(
    ROOT / "3_nodal_network_metrics" / "03_nodal_multivariate_analysis.py",
    "nodal_multivar_guardrails",
)


def _make_group_df(groups: list[str]) -> pd.DataFrame:
    rows = []
    for i, grp in enumerate(groups, start=1):
        rows.append(
            {
                "subject": f"SUB{i:02d}",
                "group": grp,
                "node": 1,
                "session": 1,
                "dummy_metric": float(i),
            }
        )
    return pd.DataFrame(rows)


def test_temporal_schema_validation_success(tmp_path):
    spec = {
        "inputs": {"node_metrics_dir": str(tmp_path / "in")},
        "outputs": {"output_dir": str(tmp_path / "out")},
    }
    temporal_mod.validate_run_spec_schema(spec, tmp_path / "run_spec.json")


def test_temporal_schema_validation_missing_outputs(tmp_path):
    spec = {"inputs": {"node_metrics_dir": str(tmp_path / "in")}}
    with pytest.raises(SystemExit, match="missing object 'outputs'"):
        temporal_mod.validate_run_spec_schema(spec, tmp_path / "run_spec.json")


def test_temporal_detect_groups_control_keyword_success():
    df = _make_group_df(["control", "intervention_a", "intervention_b"])
    intervention, control = temporal_mod.detect_groups(df, "group")
    assert control == "control"
    assert sorted(intervention) == ["intervention_a", "intervention_b"]


def test_temporal_detect_groups_ambiguous_control_fails():
    df = _make_group_df(["control", "ctrl_group", "intervention_a"])
    with pytest.raises(SystemExit, match="Ambiguous control group detection"):
        temporal_mod.detect_groups(df, "group")


def test_temporal_detect_groups_missing_control_fails():
    df = _make_group_df(["group_a", "group_b", "group_c"])
    with pytest.raises(SystemExit, match="Could not auto-detect control group"):
        temporal_mod.detect_groups(df, "group")


def test_normalize_session_values_parses_mixed_labels():
    values = np.array(["ses-1", "2", "visit3", "bad_label"], dtype=object)
    parsed = temporal_mod.normalize_session_values(values)
    assert np.allclose(parsed[:3], np.array([1.0, 2.0, 3.0]))
    assert np.isnan(parsed[3])


def test_multivar_schema_validation_missing_outputs(tmp_path):
    spec = {"inputs": {"node_metrics_dir": str(tmp_path / "in")}}
    with pytest.raises(SystemExit, match="missing object 'outputs'"):
        multivar_mod.validate_run_spec_schema(spec, tmp_path / "run_spec.json")


def test_multivar_detect_groups_explicit_partitions_success():
    df = _make_group_df(["control", "alone_2w", "alone_4w", "social_2w", "social_4w"])
    config = {
        "control_group": "control",
        "alone_groups": ["alone_2w", "alone_4w"],
        "group_groups": ["social_2w", "social_4w"],
        "short_groups": ["alone_2w", "social_2w"],
        "long_groups": ["alone_4w", "social_4w"],
    }

    groups = multivar_mod.detect_groups(df, "group", config)
    assert groups["control"] == "control"
    assert sorted(groups["intervention"]) == ["alone_2w", "alone_4w", "social_2w", "social_4w"]


def test_multivar_detect_groups_missing_partitions_fail_fast():
    df = _make_group_df(["control", "arm_a", "arm_b", "arm_c", "arm_d"])
    config = {"control_group": "control"}
    with pytest.raises(SystemExit, match="Could not determine alone/group"):
        multivar_mod.detect_groups(df, "group", config)


def test_multivar_detect_groups_overlap_partitions_fail_fast():
    df = _make_group_df(["control", "alone_2w", "alone_4w", "social_2w", "social_4w"])
    config = {
        "control_group": "control",
        "alone_groups": ["alone_2w", "social_2w"],
        "group_groups": ["social_2w", "social_4w"],
        "short_groups": ["alone_2w", "social_2w"],
        "long_groups": ["alone_4w", "social_4w"],
    }
    with pytest.raises(SystemExit, match="appear in both alone and group"):
        multivar_mod.detect_groups(df, "group", config)
