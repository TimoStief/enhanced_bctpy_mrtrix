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

# Schwere Imports mocken, die beim Import des Skripts geladen werden
sys.modules.setdefault("bct", MagicMock())
sys.modules.setdefault("umap", MagicMock())
sys.modules.setdefault("umap.UMAP", MagicMock())

import importlib
metrics_mod = importlib.import_module("global_basic_metrics")

detect_metadata_columns = metrics_mod.detect_metadata_columns
detect_atlas_name       = metrics_mod.detect_atlas_name
detect_file_format      = metrics_mod.detect_file_format


# ── detect_metadata_columns ──────────────────────────────────────────────────

def test_detect_metadata_columns_standard():
    """Standard-Spaltennamen werden korrekt erkannt."""
    df = pd.DataFrame(columns=["participant_id", "session", "group", "sex"])
    result = detect_metadata_columns(df)
    assert result["subject_col"] == "participant_id"
    assert result["session_col"] == "session"
    assert result["group_col"] == "group"
    assert result["sex_col"] == "sex"


def test_detect_metadata_columns_alternative_names():
    """Alternative Spaltennamen (z. B. 'subject_id') werden erkannt."""
    df = pd.DataFrame(columns=["subject_id", "timepoint", "condition", "gender"])
    result = detect_metadata_columns(df)
    assert result["subject_col"] == "subject_id"
    assert result["session_col"] == "timepoint"
    assert result["group_col"] == "condition"
    assert result["sex_col"] == "gender"


def test_detect_metadata_columns_missing_required():
    """Fehlen Pflichtfelder (subject, session), wird sys.exit aufgerufen."""
    df = pd.DataFrame(columns=["group", "sex"])
    with pytest.raises(SystemExit):
        detect_metadata_columns(df)


def test_detect_metadata_columns_partial_match():
    """Partielle Übereinstimmung im Spaltennamen wird als Fallback erkannt."""
    df = pd.DataFrame(columns=["participant_id_extended", "session_label"])
    result = detect_metadata_columns(df)
    assert result["subject_col"] is not None
    assert result["session_col"] is not None


# ── detect_atlas_name ─────────────────────────────────────────────────────────

def test_detect_atlas_name_known(tmp_path):
    """Bekannter Atlas-Name im Pfad wird erkannt."""
    atlas_dir = tmp_path / "brainnectome" / "data"
    atlas_dir.mkdir(parents=True)
    result = detect_atlas_name(atlas_dir)
    assert result == "Brainnectome"


def test_detect_atlas_name_unknown(tmp_path):
    """Unbekannter Pfad gibt 'unknown' zurück."""
    result = detect_atlas_name(tmp_path)
    assert result == "Unknown"


def test_detect_atlas_name_aal(tmp_path):
    """AAL-Atlas wird erkannt."""
    aal_dir = tmp_path / "aal_atlas"
    aal_dir.mkdir()
    result = detect_atlas_name(aal_dir)
    assert result == "Aal"


# ── detect_file_format ────────────────────────────────────────────────────────

def test_detect_file_format_npy(tmp_path):
    """Ordner mit .npy-Dateien gibt 'npy' zurück."""
    (tmp_path / "matrix.npy").write_bytes(b"")
    result = detect_file_format(tmp_path)
    assert result == "npy"


def test_detect_file_format_mat(tmp_path):
    """Ordner mit .mat-Dateien gibt 'mat' zurück."""
    (tmp_path / "matrix.mat").write_bytes(b"")
    result = detect_file_format(tmp_path)
    assert result == "mat"


def test_detect_file_format_no_files(tmp_path):
    """Leerer Ordner führt zu sys.exit."""
    with pytest.raises(SystemExit):
        detect_file_format(tmp_path)


def test_detect_file_format_prefers_npy(tmp_path):
    """Bei Gleichstand oder mehr .npy-Dateien wird 'npy' bevorzugt."""
    (tmp_path / "a.npy").write_bytes(b"")
    (tmp_path / "b.npy").write_bytes(b"")
    (tmp_path / "c.mat").write_bytes(b"")
    result = detect_file_format(tmp_path)
    assert result == "npy"
