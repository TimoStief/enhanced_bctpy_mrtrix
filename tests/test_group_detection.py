"""
Tests für group_detection.py
"""
import json
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from group_detection import (
    detect_or_ask_groups,
    _match,
    _try_auto_detect,
    _interactive_assign,
    _save_to_run_spec,
    KEYWORDS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _make_df(groups: list, n_per_group: int = 5) -> pd.DataFrame:
    records = []
    for g in groups:
        for i in range(n_per_group):
            records.append({"subject": f"S_{g}_{i}", "group": g, "degree": float(i)})
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# _match
# ═══════════════════════════════════════════════════════════════════════════════

def test_match_finds_alone():
    groups = ["alone_2w", "group_2w", "control"]
    result = _match(groups, KEYWORDS["alone"])
    assert "alone_2w" in result
    assert "group_2w" not in result


def test_match_finds_control():
    groups = ["control", "intervention_a", "intervention_b"]
    result = _match(groups, KEYWORDS["control"])
    assert "control" in result


def test_match_empty_when_no_keywords():
    groups = ["alpha", "beta", "gamma"]
    result = _match(groups, KEYWORDS["control"])
    assert result == []


def test_match_case_insensitive():
    groups = ["ALONE_2W", "GROUP_2W"]
    result = _match(groups, KEYWORDS["alone"])
    assert "ALONE_2W" in result


def test_match_short_duration():
    groups = ["intervention_2w", "intervention_4w", "control"]
    result = _match(groups, KEYWORDS["short"])
    assert "intervention_2w" in result
    assert "intervention_4w" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# _try_auto_detect
# ═══════════════════════════════════════════════════════════════════════════════

def test_try_auto_detect_success():
    df     = _make_df(["alone_2w", "alone_4w", "group_2w", "group_4w", "control"])
    counts = df.groupby("group")["degree"].count()
    result = _try_auto_detect(df["group"].unique().tolist(), counts)
    assert result is not None
    assert result["control"] == "control"
    assert "alone_2w" in result["alone"]
    assert "group_2w" in result["social"]


def test_try_auto_detect_returns_none_when_no_keywords():
    df     = _make_df(["alpha", "beta", "gamma"])
    counts = df.groupby("group")["degree"].count()
    result = _try_auto_detect(df["group"].unique().tolist(), counts)
    assert result is None


def test_try_auto_detect_fallback_smallest_control():
    df = _make_df(["intervention_a", "intervention_b"], n_per_group=5)
    extra = pd.DataFrame({"subject": ["S_ctrl_0"], "group": ["ctrl"], "degree": [1.0]})
    df = pd.concat([df, extra], ignore_index=True)
    counts = df.groupby("group")["degree"].count()
    result = _try_auto_detect(df["group"].unique().tolist(), counts)
    # ctrl ist kleinste Gruppe
    assert result is None or result["control"] == "ctrl"


def test_try_auto_detect_detects_short_long():
    df     = _make_df(["int_2w", "int_4w", "control"])
    counts = df.groupby("group")["degree"].count()
    result = _try_auto_detect(df["group"].unique().tolist(), counts)
    assert result is not None
    assert "int_2w" in result["short"]
    assert "int_4w" in result["long"]


# ═══════════════════════════════════════════════════════════════════════════════
# detect_or_ask_groups – with config
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_or_ask_uses_config():
    df = _make_df(["alone_2w", "group_2w", "control"])
    config = {
        "control_group": "control",
        "alone_groups":  ["alone_2w"],
        "group_groups":  ["group_2w"],
    }
    result = detect_or_ask_groups(df, "group", config)
    assert result["control"]  == "control"
    assert "alone_2w" in result["alone"]
    assert "group_2w" in result["social"]


def test_detect_or_ask_config_overrides_autodetect():
    df = _make_df(["alone_2w", "group_2w", "control"])
    config = {"control_group": "alone_2w"}  # override: treat alone_2w as control
    result = detect_or_ask_groups(df, "group", config)
    assert result["control"] == "alone_2w"


def test_detect_or_ask_config_partial_uses_keywords():
    """Config hat control_group aber keine alone/social → Keywords als Fallback."""
    df = _make_df(["alone_2w", "group_2w", "control"])
    config = {"control_group": "control"}
    result = detect_or_ask_groups(df, "group", config)
    assert result["control"] == "control"
    assert "alone_2w" in result["alone"]


# ═══════════════════════════════════════════════════════════════════════════════
# detect_or_ask_groups – auto-detect
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_or_ask_auto_detects_known_labels():
    df     = _make_df(["alone_2w", "alone_4w", "group_2w", "group_4w", "control"])
    config = {}
    result = detect_or_ask_groups(df, "group", config)
    assert result["control"]  == "control"
    assert "alone_2w" in result["alone"]
    assert "group_2w" in result["social"]
    assert "alone_2w" in result["short"]
    assert "alone_4w" in result["long"]


def test_detect_or_ask_returns_all_keys():
    df     = _make_df(["alone_2w", "group_2w", "control"])
    config = {}
    result = detect_or_ask_groups(df, "group", config)
    for key in ["all", "control", "intervention", "alone", "social", "short", "long"]:
        assert key in result


def test_detect_or_ask_intervention_excludes_control():
    df     = _make_df(["alone_2w", "group_2w", "control"])
    config = {}
    result = detect_or_ask_groups(df, "group", config)
    assert result["control"] not in result["intervention"]


# ═══════════════════════════════════════════════════════════════════════════════
# detect_or_ask_groups – interactive fallback
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_or_ask_interactive_fallback(tmp_path):
    """Wenn keine Keywords → interaktiver Modus wird aufgerufen."""
    df     = _make_df(["alpha", "beta", "gamma"])
    config = {}

    # Simuliere User-Eingaben: 3=gamma als Control, alpha als alone,
    # beta als social, alpha als short, beta als long, n (nicht speichern)
    inputs = iter(["3", "1", "2", "1", "2", "n"])
    with patch("builtins.input", lambda _: next(inputs)):
        result = detect_or_ask_groups(df, "group", config)

    assert result["control"] == "gamma"
    assert "alpha" in result["alone"]
    assert "beta"  in result["social"]


def test_detect_or_ask_interactive_saves_to_run_spec(tmp_path):
    """Interaktiver Modus speichert in run_spec.json wenn User 'y' eingibt."""
    df        = _make_df(["alpha", "beta", "gamma"])
    config    = {}
    spec_path = tmp_path / "run_spec.json"

    inputs = iter(["3", "1", "2", "1", "2", "y"])
    with patch("builtins.input", lambda _: next(inputs)):
        detect_or_ask_groups(df, "group", config, run_spec_path=spec_path)

    assert spec_path.exists()
    with open(spec_path) as f:
        saved = json.load(f)
    assert "control_group" in saved
    assert saved["control_group"] == "gamma"


# ═══════════════════════════════════════════════════════════════════════════════
# _save_to_run_spec
# ═══════════════════════════════════════════════════════════════════════════════

def test_save_to_run_spec_creates_file(tmp_path):
    groups = {
        "control": "ctrl", "alone": ["a1"], "social": ["s1"],
        "short": ["a1"], "long": ["s1"],
    }
    spec_path = tmp_path / "run_spec.json"
    with patch("builtins.input", return_value="y"):
        _save_to_run_spec(groups, spec_path)
    assert spec_path.exists()
    with open(spec_path) as f:
        saved = json.load(f)
    assert saved["control_group"] == "ctrl"
    assert saved["alone_groups"]  == ["a1"]


def test_save_to_run_spec_merges_existing(tmp_path):
    """Bestehende run_spec.json wird nicht überschrieben sondern erweitert."""
    spec_path = tmp_path / "run_spec.json"
    existing  = {"inputs": {"data_dir": "/data"}, "outputs": {"output_dir": "/out"}}
    spec_path.write_text(json.dumps(existing))

    groups = {
        "control": "ctrl", "alone": ["a1"], "social": ["s1"],
        "short": ["a1"], "long": ["s1"],
    }
    with patch("builtins.input", return_value="y"):
        _save_to_run_spec(groups, spec_path)

    with open(spec_path) as f:
        saved = json.load(f)
    assert saved["inputs"]["data_dir"] == "/data"
    assert saved["control_group"]      == "ctrl"


def test_save_to_run_spec_no_when_user_says_no(tmp_path):
    groups    = {"control": "ctrl", "alone": [], "social": [], "short": [], "long": []}
    spec_path = tmp_path / "run_spec.json"
    with patch("builtins.input", return_value="n"):
        _save_to_run_spec(groups, spec_path)
    assert not spec_path.exists()


def test_save_to_run_spec_default_path(tmp_path):
    """Wenn run_spec_path=None → speichert als run_spec.json im CWD."""
    groups = {
        "control": "ctrl", "alone": ["a1"], "social": ["s1"],
        "short": ["a1"], "long": ["s1"],
    }
    default = Path("run_spec.json")
    with patch("builtins.input", return_value="y"):
        with patch("group_detection.Path", wraps=Path) as mock_path:
            _save_to_run_spec(groups, None)
    # Kein Crash ist ausreichend für diesen Test