#!/usr/bin/env python3
"""
Phase 1b — Extract each sample zip into its own folder and rename files
with a sample-id prefix so they don't collide.

Layout produced
---------------
data/raw/
  0000/
    0000_description.txt
    0000_A.hdr.h5
    0000_A.ply
    0000_A.png
    0000_A_im-000.jpg
    ...
    0000_B.hdr.h5          (if present)
    ...
  0001/
  ...

Usage
-----
python phase1_extract.py --zips data/zips --out data/raw
python phase1_extract.py --zips data/zips --out data/raw --force
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def is_zip_intact(path):
    """Quick integrity check without extracting everything."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            bad = z.testzip()
            return bad is None
    except zipfile.BadZipFile:
        return False


def sample_id_from_zip(zip_path):
    """0000.zip -> '0000'; Bg.zip -> 'Bg'."""
    return zip_path.stem


def rename_inside(sample_dir, sample_id, force=False):
    """
    Rename files in sample_dir so each starts with '<sample_id>_'.
    Idempotent: if already renamed, skip.
    """
    for f in list(sample_dir.iterdir()):
        if not f.is_file():
            continue
        name = f.name
        # already prefixed?
        if name.startswith(sample_id + "_"):
            continue
        # description.txt -> 0000_description.txt
        new_name = f"{sample_id}_{name}"
        target = sample_dir / new_name
        if target.exists() and not force:
            continue
        f.rename(target)


def extract_one(zip_path, out_root, force=False, log=None):
    sample_id = sample_id_from_zip(zip_path)
    sample_dir = out_root / sample_id

    # already extracted and renamed?
    marker = sample_dir / f".extracted_{sample_id}"
    if marker.exists() and not force:
        if log: log.write(f"SKIP  {zip_path.name}  (already extracted)\n")
        return "skip"

    # check integrity before extracting
    if not is_zip_intact(zip_path):
        if log: log.write(f"BAD   {zip_path.name}  (corrupt zip)\n")
        return "bad"

    # fresh extract
    if sample_dir.exists() and force:
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Some zips may contain a top-level folder; flatten it.
            for member in z.namelist():
                # Skip directories
                if member.endswith("/"):
                    continue
                # Strip any leading folder from member path
                parts = Path(member).parts
                # If there is a single top folder, drop it
                if len(parts) > 1 and not any(p.startswith(".") for p in parts):
                    # Common case: all members under one folder
                    flat_name = parts[-1]
                else:
                    flat_name = parts[-1]
                # Extract this file into sample_dir with flat name
                with z.open(member) as src, open(sample_dir / flat_name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception as e:
        if log: log.write(f"FAIL  {zip_path.name}  ({type(e).__name__}: {e})\n")
        return "fail"

    rename_inside(sample_dir, sample_id, force=force)

    # marker so re-runs skip
    marker.write_text("ok\n")

    if log: log.write(f"OK    {zip_path.name}  ->  {sample_dir}\n")
    return "ok"


def main():
    ap = argparse.ArgumentParser(description="Extract and rename Zenodo sample zips.")
    ap.add_argument("--zips", default="data/zips",
                    help="folder containing .zip files")
    ap.add_argument("--out", default="data/raw",
                    help="output folder for extracted samples")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if already done")
    ap.add_argument("--log", default="extract_log.txt")
    args = ap.parse_args()

    zips_dir = Path(args.zips)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted(zips_dir.glob("*.zip"))
    if not zips:
        sys.exit(f"No .zip files found in {zips_dir}")

    print(f"Found {len(zips)} zips in {zips_dir}")
    print(f"Extracting to {out_dir}\n")

    counts = {"ok": 0, "skip": 0, "bad": 0, "fail": 0}
    with open(args.log, "a") as log:
        log.write(f"\n=== Extract run ===\n")
        for i, zp in enumerate(zips, 1):
            print(f"[{i}/{len(zips)}] {zp.name}")
            status = extract_one(zp, out_dir, force=args.force, log=log)
            counts[status] = counts.get(status, 0) + 1
            log.flush()

    print(f"\n=== Extract summary ===")
    for k, v in counts.items():
        print(f"  {k:5s} : {v}")
    print(f"  log   : {args.log}")


if __name__ == "__main__":
    main()