#!/usr/bin/env python3
"""
Phase 1c — Build manifest.csv and verify each sample.

Reads data/raw/<sample_id>/, parses description.txt for the mineral name,
finds the h5 files (A and/or B measurements), and produces:

  manifest.csv   one row per measurement (A and B are separate rows)
  rejected.csv   samples that failed verification, with reasons

Manifest columns
----------------
sample_id      e.g. 0000
mineral        e.g. Celestite
measurement    A or B
h5_path        relative path to the .hdr.h5 file
desc_path      relative path to the description txt
h5_shape       e.g. "320,435,256"
n_bands        256
in_library     True/False (mineral exists in sam_library.npz)
notes          free-form

Usage
-----
python phase1_manifest.py --raw data/raw --library sam_library.npz \
    --out manifest.csv --rejected rejected.csv
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


NAME_RE = re.compile(r"^\s*name\s*:\s*(.+?)\s*$", re.IGNORECASE)
NAME_FR_RE = re.compile(r"^\s*name\[fr\]\s*:\s*(.+?)\s*$", re.IGNORECASE)


def parse_description(path):
    """Return {'name': ..., 'raw_lines': [...]} from description.txt."""
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    name = None
    for line in text.splitlines():
        m = NAME_RE.match(line)
        if m and not NAME_FR_RE.match(line):
            name = m.group(1).strip().strip('"').strip("'")
            break
    return {"name": name, "raw_lines": text.splitlines()}


def find_h5_files(sample_dir):
    """Find all .hdr.h5 files (or any .h5) in the sample dir.
    Returns list of (path, measurement_letter) where letter is 'A' or 'B'."""
    found = []
    for p in sample_dir.iterdir():
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith(".hdr.h5"):
            # Look for measurement letter in the name
            m = re.search(r"_([AB])\.hdr\.h5$", p.name, re.IGNORECASE)
            if m:
                found.append((p, m.group(1).upper()))
            else:
                found.append((p, "?"))
        elif n.endswith(".h5") and "im" not in n:
            m = re.search(r"_([AB])\.h5$", p.name, re.IGNORECASE)
            letter = m.group(1).upper() if m else "?"
            found.append((p, letter))
    return found


def find_description(sample_dir):
    """Find the description txt (any name matching *description*)."""
    for p in sample_dir.iterdir():
        if p.is_file() and "description" in p.name.lower() and p.suffix.lower() == ".txt":
            return p
    return None


def inspect_h5(path):
    """Return (shape, ok, reason). Loads only /hdr metadata."""
    try:
        with h5py.File(path, "r") as f:
            if "/hdr" not in f:
                return None, False, "no /hdr dataset"
            dset = f["/hdr"]
            shape = dset.shape
            return tuple(shape), True, ""
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Build manifest and verify samples.")
    ap.add_argument("--raw", default="data/raw",
                    help="folder with extracted sample folders")
    ap.add_argument("--library", default="sam_library.npz",
                    help="SAM library, used to check mineral existence")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--rejected", default="rejected.csv")
    ap.add_argument("--expect-bands", type=int, default=256,
                    help="expected band count (default 256)")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.is_dir():
        sys.exit(f"{raw} is not a folder")

    # Load library mineral names (for cross-reference)
    lib_minerals = set()
    if Path(args.library).exists():
        lib = np.load(args.library, allow_pickle=True)
        lib_minerals = set(str(m) for m in lib["names"])
        # Normalize case for comparison
        lib_minerals_lower = {m.lower() for m in lib_minerals}
    else:
        print(f"WARNING: {args.library} not found; skipping library cross-ref")
        lib_minerals_lower = set()

    samples = sorted([d for d in raw.iterdir() if d.is_dir()])

    rows = []
    rejected = []

    for sdir in samples:
        sid = sdir.name

        # --- description ---
        desc_path = find_description(sdir)
        if desc_path is None:
            rejected.append({"sample_id": sid, "reason": "no description.txt"})
            continue
        parsed = parse_description(desc_path)
        if parsed is None or parsed["name"] is None:
            rejected.append({"sample_id": sid, "reason": "description unparseable"})
            continue
        mineral = parsed["name"]

        # --- library cross-ref ---
        in_lib = mineral.lower() in lib_minerals_lower
        lib_note = "" if in_lib else "mineral not in sam_library"

        # --- h5 files ---
        h5s = find_h5_files(sdir)
        if not h5s:
            rejected.append({"sample_id": sid, "reason": "no .hdr.h5 found"})
            continue

        # One row per measurement
        for h5_path, meas in h5s:
            shape, ok, reason = inspect_h5(h5_path)
            if not ok:
                rejected.append({"sample_id": sid, "measurement": meas,
                                 "reason": f"h5 bad: {reason}"})
                continue
            n_bands = shape[-1] if shape else 0
            band_note = "" if n_bands == args.expect_bands else \
                f"bands={n_bands} (expected {args.expect_bands})"
            rows.append({
                "sample_id": sid,
                "mineral": mineral,
                "measurement": meas,
                "h5_path": str(h5_path.relative_to(raw.parent)),
                "desc_path": str(desc_path.relative_to(raw.parent)),
                "h5_shape": "x".join(str(s) for s in shape),
                "n_bands": n_bands,
                "in_library": in_lib,
                "notes": "; ".join(x for x in [lib_note, band_note] if x),
            })

    # --- write manifest ---
    fieldnames = ["sample_id", "mineral", "measurement", "h5_path", "desc_path",
                  "h5_shape", "n_bands", "in_library", "notes"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- write rejected ---
    with open(args.rejected, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "measurement", "reason"])
        w.writeheader()
        for r in rejected:
            w.writerow({"sample_id": r.get("sample_id", ""),
                        "measurement": r.get("measurement", ""),
                        "reason": r.get("reason", "")})

    # --- summary ---
    print(f"\n=== Manifest summary ===")
    print(f"  samples scanned : {len(samples)}")
    print(f"  measurements ok : {len(rows)}")
    print(f"  rejected        : {len(rejected)}")
    print(f"  manifest        : {args.out}")
    print(f"  rejected        : {args.rejected}")

    # mineral histogram
    mineral_counts = Counter(r["mineral"] for r in rows)
    print(f"\n  distinct minerals : {len(mineral_counts)}")
    print(f"\n  Top 20 by measurement count:")
    for m, c in mineral_counts.most_common(20):
        print(f"    {m:32s} {c}")

    # flag minerals not in library
    not_in_lib = sorted({r["mineral"] for r in rows if not r["in_library"]})
    if not_in_lib:
        print(f"\n  WARNING: {len(not_in_lib)} minerals in dataset "
              f"not present in sam_library.npz:")
        for m in not_in_lib:
            print(f"    {m}")

    # class balance hint
    counts_only = list(mineral_counts.values())
    if counts_only:
        print(f"\n  class balance: min={min(counts_only)} "
              f"median={sorted(counts_only)[len(counts_only)//2]} "
              f"max={max(counts_only)}")

    # machine-readable summary
    summary = {
        "samples_scanned": len(samples),
        "measurements_ok": len(rows),
        "rejected": len(rejected),
        "distinct_minerals": len(mineral_counts),
        "class_min": min(counts_only) if counts_only else 0,
        "class_max": max(counts_only) if counts_only else 0,
        "mineral_counts": dict(mineral_counts),
        "not_in_library": not_in_lib,
    }
    with open("manifest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary JSON: manifest_summary.json")


if __name__ == "__main__":
    main()