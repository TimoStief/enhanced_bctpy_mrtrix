"""
Tests für preflight_check.py
"""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

# Skript importieren (liegt eine Ebene höher)
sys.path.insert(0, str(Path(__file__).parent.parent))
from preflight_check import load_spec, resolve_path, check_imports


# ── load_spec ────────────────────────────────────────────────────────────────

def test_load_spec_valid(tmp_path):
    """Gültige run_spec.json wird korrekt geladen."""
    spec = {"inputs": {"data_dir": "/data"}, "outputs": {"output_dir": "/out"}}
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text(json.dumps(spec))

    result = load_spec(spec_file)

    assert result["inputs"]["data_dir"] == "/data"
    assert result["_spec_dir"] == str(tmp_path)


def test_load_spec_missing_file(tmp_path):
    """Fehlende run_spec.json wirft FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Run spec not found"):
        load_spec(tmp_path / "nicht_vorhanden.json")


def test_load_spec_invalid_json(tmp_path):
    """Ungültiges JSON wirft einen Fehler."""
    spec_file = tmp_path / "run_spec.json"
    spec_file.write_text("{ kein gültiges json }")

    with pytest.raises(Exception):
        load_spec(spec_file)


# ── resolve_path ─────────────────────────────────────────────────────────────

def test_resolve_path_absolute(tmp_path):
    """Absoluter Pfad bleibt absolut."""
    result = resolve_path(tmp_path, str(tmp_path))
    assert result.is_absolute()


def test_resolve_path_relative(tmp_path):
    """Relativer Pfad wird relativ zu spec_dir aufgelöst."""
    result = resolve_path(tmp_path, "unterordner/datei.txt")
    assert result == (tmp_path / "unterordner" / "datei.txt").resolve()


def test_resolve_path_with_env_var(tmp_path, monkeypatch):
    """Umgebungsvariablen im Pfad werden aufgelöst."""
    monkeypatch.setenv("MEIN_ORDNER", str(tmp_path))
    result = resolve_path(tmp_path, "$MEIN_ORDNER/datei.txt")
    assert str(tmp_path) in str(result)


# ── check_imports ─────────────────────────────────────────────────────────────

def test_check_imports_all_present():
    """Bekannte installierte Pakete werden nicht als fehlend gemeldet."""
    missing = check_imports(["json", "pathlib", "os"])
    assert missing == []


def test_check_imports_missing_package():
    """Nicht installiertes Paket wird als fehlend erkannt."""
    missing = check_imports(["dieses_paket_existiert_sicher_nicht_xyz"])
    assert "dieses_paket_existiert_sicher_nicht_xyz" in missing


def test_check_imports_mixed():
    """Mischung aus vorhandenen und fehlenden Paketen."""
    missing = check_imports(["os", "paket_fehlt_abc123", "sys"])
    assert "paket_fehlt_abc123" in missing
    assert "os" not in missing
    assert "sys" not in missing


def test_check_imports_empty_list():
    """Leere Liste ergibt keine fehlenden Pakete."""
    missing = check_imports([])
    assert missing == []
