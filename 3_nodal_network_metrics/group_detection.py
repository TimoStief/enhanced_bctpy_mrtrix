#!/usr/bin/env python3
"""
group_detection.py
==================

Shared utility for auto-detecting or interactively assigning group labels
across all pipeline scripts.

USAGE (in other scripts):
    from group_detection import detect_or_ask_groups
    groups = detect_or_ask_groups(df, group_col, config, run_spec_path)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Keyword-based auto-detection ─────────────────────────────────────────────

KEYWORDS = {
    "control":  ["control", "ctrl", "waitlist", "wait", "passive"],
    "alone":    ["alone", "individual", "solo", "single"],
    "social":   ["group", "social", "team", "collective"],
    "short":    ["2w", "short", "2week", "week2", "brief"],
    "long":     ["4w", "long",  "4week", "week4", "extended"],
}


def _match(groups: list, keywords: list) -> list:
    return [g for g in groups if any(k in str(g).lower() for k in keywords)]


def _try_auto_detect(all_groups: list, group_counts: pd.Series) -> dict | None:
    """
    Try to auto-detect groups from labels.
    Returns dict if successful, None if detection is ambiguous.
    """
    control_matches = _match(all_groups, KEYWORDS["control"])
    if not control_matches:
        # Fallback: smallest group
        control = group_counts.idxmin()
    elif len(control_matches) == 1:
        control = control_matches[0]
    else:
        return None  # ambiguous

    intervention = [g for g in all_groups if g != control]

    alone  = _match(intervention, KEYWORDS["alone"])
    social = _match(intervention, KEYWORDS["social"])
    short  = _match(intervention, KEYWORDS["short"])
    long_  = _match(intervention, KEYWORDS["long"])

    # If we found meaningful keyword matches → success
    if alone or social or short or long_:
        return {
            "all":          all_groups,
            "control":      control,
            "intervention": intervention,
            "alone":        alone,
            "social":       social,
            "short":        short,
            "long":         long_,
        }
    return None  # couldn't detect


# ── Interactive mode ─────────────────────────────────────────────────────────

def _ask(prompt: str, all_groups: list, allow_empty: bool = False) -> list:
    """Ask user to select groups interactively."""
    print(f"\n  {prompt}")
    for i, g in enumerate(all_groups):
        print(f"    [{i+1}] {g}")
    print("  Enter numbers (space-separated) or group names, or press Enter to skip:")

    while True:
        raw = input("  > ").strip()

        if not raw and allow_empty:
            return []

        # Try numeric selection
        parts = raw.split()
        result = []
        valid  = True

        for p in parts:
            if p.isdigit():
                idx = int(p) - 1
                if 0 <= idx < len(all_groups):
                    result.append(all_groups[idx])
                else:
                    print(f"  ⚠ Invalid number: {p}")
                    valid = False
                    break
            elif p in all_groups:
                result.append(p)
            else:
                print(f"  ⚠ Unknown group: {p}")
                valid = False
                break

        if valid and (result or allow_empty):
            return result
        print("  Please enter valid numbers or group names.")


def _interactive_assign(all_groups: list) -> dict:
    """Interactively assign groups when auto-detection fails."""
    print("\n" + "=" * 60)
    print("GROUP ASSIGNMENT (interactive)")
    print("=" * 60)
    print(f"  Found groups: {all_groups}")

    # Control
    print("\n  Which is the CONTROL group?")
    for i, g in enumerate(all_groups):
        print(f"    [{i+1}] {g}")
    while True:
        raw = input("  > ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(all_groups):
            control = all_groups[int(raw) - 1]
            break
        elif raw in all_groups:
            control = raw
            break
        print("  Please enter a valid number or group name.")

    intervention = [g for g in all_groups if g != control]
    print(f"\n  ✓ Control: {control}")
    print(f"  Intervention groups: {intervention}")

    # Alone vs Social
    alone  = _ask("Which are ALONE/INDIVIDUAL intervention groups?",
                  intervention, allow_empty=True)
    social = _ask("Which are GROUP/SOCIAL intervention groups?",
                  intervention, allow_empty=True)

    # Short vs Long
    short = _ask("Which are SHORT duration groups?",
                 intervention, allow_empty=True)
    long_ = _ask("Which are LONG duration groups?",
                 intervention, allow_empty=True)

    return {
        "all":          all_groups,
        "control":      control,
        "intervention": intervention,
        "alone":        alone,
        "social":       social,
        "short":        short,
        "long":         long_,
    }


def _save_to_run_spec(groups: dict, run_spec_path: Path | None) -> None:
    """Offer to save group assignments to run_spec.json."""
    print("\n  Save these settings to run_spec.json for future runs? [y/n]")
    ans = input("  > ").strip().lower()
    if ans != "y":
        return

    if run_spec_path is None:
        run_spec_path = Path("run_spec.json")

    # Load existing or create new
    if run_spec_path.exists():
        with open(run_spec_path) as f:
            spec = json.load(f)
    else:
        spec = {"inputs": {}, "outputs": {}}

    spec["control_group"] = groups["control"]
    spec["alone_groups"]  = groups["alone"]
    spec["group_groups"]  = groups["social"]
    spec["short_groups"]  = groups["short"]
    spec["long_groups"]   = groups["long"]

    with open(run_spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"  ✓ Saved to {run_spec_path} — next run will skip this step!")


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_or_ask_groups(df: pd.DataFrame, group_col: str,
                          config: dict,
                          run_spec_path: Path | None = None) -> dict:
    """
    Detect group assignments automatically or interactively.

    Priority:
    1. Explicit config (from run_spec.json or CLI)
    2. Keyword-based auto-detection from group labels
    3. Interactive prompt (if auto-detection fails)

    Parameters
    ----------
    df           : DataFrame with group column
    group_col    : Name of the group column
    config       : Config dict (may contain control_group, alone_groups, etc.)
    run_spec_path: Path to run_spec.json (for saving interactive results)

    Returns
    -------
    dict with keys: all, control, intervention, alone, social, short, long
    """
    all_groups   = df[group_col].dropna().unique().tolist()
    group_counts = df.groupby(group_col)[df.columns[0]].count()

    # ── 1. Use explicit config if provided ───────────────────────────────
    if config.get("control_group"):
        control      = config["control_group"]
        intervention = [g for g in all_groups if g != control]

        groups = {
            "all":          all_groups,
            "control":      control,
            "intervention": intervention,
            "alone":        config.get("alone_groups")  or _match(intervention, KEYWORDS["alone"]),
            "social":       config.get("group_groups")  or _match(intervention, KEYWORDS["social"]),
            "short":        config.get("short_groups")  or _match(intervention, KEYWORDS["short"]),
            "long":         config.get("long_groups")   or _match(intervention, KEYWORDS["long"]),
        }

        print(f"  ✓ Control group (from config): {control}")
        print(f"  ✓ Intervention groups: {intervention}")
        print(f"  ✓ Alone:  {groups['alone']}")
        print(f"  ✓ Social: {groups['social']}")
        print(f"  ✓ Short:  {groups['short']}")
        print(f"  ✓ Long:   {groups['long']}")
        return groups

    # ── 2. Try keyword-based auto-detection ──────────────────────────────
    groups = _try_auto_detect(all_groups, group_counts)
    if groups:
        print(f"  ✓ Control group (auto-detected): {groups['control']}")
        print(f"  ✓ Intervention groups: {groups['intervention']}")
        print(f"  ✓ Alone:  {groups['alone']}")
        print(f"  ✓ Social: {groups['social']}")
        print(f"  ✓ Short:  {groups['short']}")
        print(f"  ✓ Long:   {groups['long']}")
        return groups

    # ── 3. Interactive fallback ───────────────────────────────────────────
    print("\n  ⚠ Could not auto-detect group assignments.")
    print(f"  Found groups: {all_groups}")
    groups = _interactive_assign(all_groups)
    _save_to_run_spec(groups, run_spec_path)
    return groups