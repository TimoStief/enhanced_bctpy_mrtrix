#!/usr/bin/env python3
"""Deprecated wrapper. Use 0_installation/preflight_check.py instead."""

from __future__ import annotations

import runpy
from pathlib import Path
import sys


def main() -> None:
    target = Path(__file__).resolve().parents[2] / "0_installation" / "preflight_check.py"
    if not target.exists():
        raise FileNotFoundError(f"Preflight script not found: {target}")

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    sys.argv[0] = "preflight_check.py"
    main()
