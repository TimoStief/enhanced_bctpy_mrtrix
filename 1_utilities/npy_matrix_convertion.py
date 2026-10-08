#!/usr/bin/env python3
"""
Matrix Format Converter
========================
Converts connectivity matrices from .xlsx / .txt / .csv / .mat → .npy
Output is organized in BIDS-conform folder structure: sub-XX/ses-X/

USAGE:
    python convert_matrices.py --input-dir G:/Masterarbeit/matrizen/connectogram/xlsx/Kleist/count --output-dir C:/Users/timo-/Desktop/Forschung/laufstudie_masterarbeit_npy/kleist
    python convert_matrices.py --input path/to/single_matrix.xlsx --output-dir C:/output
    python convert_matrices.py --input-dir ... --ext xlsx   # only xlsx
    python convert_matrices.py --input-dir ... --ext mat    # DSI Studio .mat files
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED = {".xlsx", ".xls", ".txt", ".csv", ".mat"}


# ── Parsers ───────────────────────────────────────────────────────────────────

def _load_xlsx(path: Path) -> np.ndarray:
    df = pd.read_excel(path, header=None, index_col=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.values.astype(np.float64)


def _load_txt(path: Path) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue  # skip header/label lines
    if not rows:
        raise ValueError(f"No numeric data found in {path.name}")
    arr = np.array(rows, dtype=np.float64)
    # Fix non-square from label row/col
    if arr.shape[0] != arr.shape[1]:
        if arr.shape[1] == arr.shape[0] + 1:
            arr = arr[:, 1:]
        elif arr.shape[0] == arr.shape[1] + 1:
            arr = arr[1:, :]
    return arr


def _load_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None, index_col=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.values.astype(np.float64)


def _load_mat(path: Path) -> np.ndarray:
    """Load DSI Studio .mat connectivity file."""
    import scipy.io
    mat = scipy.io.loadmat(str(path))
    skip = {"__header__", "__version__", "__globals__"}
    candidates = [k for k in mat if k not in skip]
    # Prefer keys with connectivity-related names
    for pref in ("connectivity", "matrix", "count", "data"):
        for c in candidates:
            if pref in c.lower():
                arr = np.array(mat[c], dtype=np.float64)
                if arr.ndim == 2:
                    return arr
    # Fall back to first 2-D array
    for c in candidates:
        arr = np.array(mat[c], dtype=np.float64)
        if arr.ndim == 2:
            return arr
    raise ValueError(f"No 2-D matrix found in {path.name}. Keys: {candidates}")


def load_matrix(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return _load_xlsx(path)
    elif ext == ".txt":
        return _load_txt(path)
    elif ext == ".csv":
        return _load_csv(path)
    elif ext == ".mat":
        return _load_mat(path)
    raise ValueError(f"Unsupported format: {ext}")


# ── BIDS path extraction ──────────────────────────────────────────────────────

def extract_bids(filename: str) -> tuple[str | None, str | None]:
    """Extract sub-XX and ses-X from filename.

    For DSI Studio files like '100206.qsdr.fz.tt.gz.HCP-MMP.connectivity.mat'
    the subject ID is the leading number → converted to sub-100206.
    """
    # Standard BIDS pattern
    sub = re.search(r"(sub-[A-Za-z0-9]+)", filename)
    ses = re.search(r"(ses-\d+)", filename)
    if sub:
        return (sub.group(1), ses.group(1) if ses else None)

    # DSI Studio pattern: leading number before first dot
    dsi_match = re.match(r"^(\d+)\.", filename)
    if dsi_match:
        return (f"sub-{dsi_match.group(1)}", ses.group(1) if ses else None)

    return (None, None)


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_file(src: Path, out_dir: Path, bids: bool = True) -> Path | None:
    print(f"  → {src.name}")

    try:
        A = load_matrix(src)
    except Exception as e:
        print(f"    ✗ Load failed: {e}")
        return None

    # Validate
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print(f"    ✗ Not square ({A.shape}) — skipped")
        return None
    if A.max() == 0:
        print(f"    ⚠ All zeros — empty matrix?")
    if not np.allclose(A, A.T, atol=1.0):
        print(f"    ⚠ Not symmetric (normal for DSI Studio fiber counts)")

    # Determine output path
    if bids:
        sub, ses = extract_bids(src.name)
        if sub and ses:
            dst_dir = out_dir / sub / ses
        elif sub:
            dst_dir = out_dir / sub / "ses-1"
        else:
            # Fallback: use source subfolder name (e.g. ses-1)
            dst_dir = out_dir / src.parent.name
    else:
        dst_dir = out_dir

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (src.stem + ".npy")
    np.save(dst, A)

    print(f"    ✓ {A.shape[0]}×{A.shape[1]}  max={A.max():.0f}  → {dst.relative_to(out_dir)}")
    return dst


def convert_directory(src_dir: Path, out_dir: Path,
                      extensions: set[str], bids: bool = True,
                      filter_pattern: str | None = None) -> tuple[int, int]:
    files = sorted(f for f in src_dir.rglob("*") if f.suffix.lower() in extensions)
    if filter_pattern:
        files = [f for f in files if filter_pattern in f.name]

    if not files:
        print(f"  ⚠ No {extensions} files found in {src_dir}")
        return 0, 0

    print(f"  Found {len(files)} file(s)\n")
    ok = failed = 0
    for f in files:
        result = convert_file(f, out_dir, bids=bids)
        if result:
            ok += 1
        else:
            failed += 1
    return ok, failed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert connectivity matrices (xlsx/txt/csv/mat) → npy with BIDS output"
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input",     help="Single file to convert")
    grp.add_argument("--input-dir", help="Folder to scan recursively")

    p.add_argument("--output-dir", required=True, help="Output folder")
    p.add_argument("--filter", default=None,
                   metavar="PATTERN",
                   help="Only convert files whose name contains this pattern. "
                        "E.g. --filter roi_normalized")
    p.add_argument("--ext", nargs="+", default=None,
                   help="Extensions to convert, e.g. --ext xlsx txt mat")
    p.add_argument("--no-bids", action="store_true",
                   help="Don't create sub-XX/ses-X folder structure")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exts  = {f".{e.lstrip('.')}" for e in args.ext} if args.ext else SUPPORTED
    out   = Path(args.output_dir).expanduser().resolve()
    bids  = not args.no_bids

    print("=" * 60)
    print("MATRIX FORMAT CONVERTER")
    print("=" * 60)
    print(f"  Formats :  {sorted(exts)}")
    print(f"  Output  :  {out}")
    print(f"  BIDS    :  {'yes (sub-XX/ses-X/)' if bids else 'no'}")
    if args.filter:
        print(f"  Filter  :  *{args.filter}*")
    print()

    if args.input:
        src = Path(args.input).expanduser().resolve()
        if not src.exists():
            sys.exit(f"✗ File not found: {src}")
        out.mkdir(parents=True, exist_ok=True)
        result = convert_file(src, out, bids=bids)
        if not result:
            sys.exit("✗ Conversion failed")

    else:
        src_dir = Path(args.input_dir).expanduser().resolve()
        if not src_dir.exists():
            sys.exit(f"✗ Directory not found: {src_dir}")
        ok, failed = convert_directory(src_dir, out, exts, bids=bids, filter_pattern=args.filter)
        print(f"\n{'='*60}")
        print(f"  ✓ Converted : {ok}")
        if failed:
            print(f"  ✗ Failed    : {failed}")
        print("=" * 60)


if __name__ == "__main__":
    main()