#!/usr/bin/env python3
"""
Phase 1a — Download all sample zips from Zenodo.

Features
--------
- Auto-generates URLs from the Zenodo pattern (0000..0129 + Bg).
  Or reads a URL list if one is provided.
- Resumable: partial downloads save as .part files and pick up where they left off.
- Skips already-complete files (verified by Content-Length).
- Retries on transient network errors.
- Single-threaded by default (safer on slow wifi); --workers N for parallel.
- Logs everything to download_log.txt.

Usage
-----
# Quick run with auto-generated URL list
python phase1_download.py --out data/zips

# From a URL file, one per line
python phase1_download.py --url-file Download_List.txt --out data/zips

# Parallel with 3 workers (careful on slow wifi)
python phase1_download.py --out data/zips --workers 3

# Verify checksums after download (needs a checksum file)
python phase1_download.py --out data/zips --verify-sha256 checksums.txt
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CHUNK_SIZE = 1024 * 1024          # 1 MB
DEFAULT_TIMEOUT = (30, 300)       # (connect, read) seconds
CHUNK_READ_TIMEOUT = 300          # seconds per chunk read
FILE_RETRIES = 10
FILE_RETRY_WAIT = 5


def gen_zenodo_urls(start=0, end=129, extra=("Bg",)):
    """Yield URLs for the standard Zenodo 0000..NNNN pattern plus extras."""
    base = "https://zenodo.org/records/1476495/files"
    for i in range(start, end + 1):
        yield f"{base}/{i:04d}.zip?download=1", f"{i:04d}.zip"
    for name in extra:
        yield f"{base}/{name}.zip?download=1", f"{name}.zip"


def load_url_file(path):
    """Read a URL file, one per line. Returns list of (url, filename)."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Filename from URL path
            parsed = urlparse(line)
            filename = os.path.basename(parsed.path)
            if not filename:
                # fallback: use last path segment without query
                filename = line.split("/")[-1].split("?")[0]
            pairs.append((line, filename))
    return pairs


def make_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,                     # 2, 4, 8, 16, 32 s
        status_forcelist=(500, 502, 503, 504, 429),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def file_size_from_server(session, url):
    """HEAD request to get Content-Length. Returns None if unavailable."""
    try:
        r = session.head(url, allow_redirects=True, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else None
    except Exception as e:
        print(f"  [head] {url} -> {e}")
        return None


def download_one(session, url, dest, log):
    """
    Download one file with automatic resume/retry.

    Returns:
        (status, bytes_downloaded)

    status:
        "ok"   - downloaded successfully
        "skip" - already complete
        "fail" - failed after all retry attempts
    """

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # Ask server for complete file size
    server_size = file_size_from_server(session, url)

    # ---------------------------------------------------------
    # Already downloaded?
    # ---------------------------------------------------------
    if dest.exists():
        local_size = dest.stat().st_size

        if server_size is not None and local_size == server_size:
            msg = (
                f"SKIP  {dest.name}  "
                f"({local_size} bytes, already complete)"
            )
            print(msg)
            log.write(msg + "\n")
            log.flush()
            return "skip", local_size

        else:
            msg = (
                f"WARN  {dest.name} exists but size mismatch "
                f"(local={local_size}, server={server_size}). "
                f"Redownloading."
            )

            print(msg)
            log.write(msg + "\n")
            log.flush()

            dest.unlink()

    # ---------------------------------------------------------
    # Automatically retry THIS SAME FILE
    # ---------------------------------------------------------
    for attempt in range(1, FILE_RETRIES + 1):

        # Recalculate every time because .part grew during
        # the previous failed attempt.
        resume_from = part.stat().st_size if part.exists() else 0

        headers = (
            {"Range": f"bytes={resume_from}-"}
            if resume_from > 0
            else {}
        )

        mode = "ab" if resume_from > 0 else "wb"

        try:
            if resume_from > 0:
                print(
                    f"  Resume attempt {attempt}/{FILE_RETRIES} "
                    f"from {resume_from / 1e6:.1f} MB"
                )
            elif attempt > 1:
                print(
                    f"  Retry attempt {attempt}/{FILE_RETRIES}"
                )

            t0 = time.time()

            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=True,
            ) as r:

                # -------------------------------------------------
                # Resume behaviour
                # -------------------------------------------------

                if resume_from > 0 and r.status_code == 200:
                    # Server ignored Range header.
                    # Restart this file cleanly.
                    print(
                        "  Server ignored Range request. "
                        "Restarting this file from 0."
                    )

                    resume_from = 0
                    mode = "wb"
                    part.unlink(missing_ok=True)

                elif resume_from > 0 and r.status_code != 206:
                    raise RuntimeError(
                        f"unexpected status "
                        f"{r.status_code} on resume"
                    )

                r.raise_for_status()

                total = server_size if server_size else None
                downloaded = resume_from

                # -------------------------------------------------
                # Download chunks
                # -------------------------------------------------

                with open(part, mode) as f:

                    for chunk in r.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):
                        if not chunk:
                            continue

                        f.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            pct = 100.0 * downloaded / total

                            print(
                                f"\r  {dest.name}: "
                                f"{pct:5.1f}%  "
                                f"({downloaded / 1e6:.1f}/"
                                f"{total / 1e6:.1f} MB)",
                                end="",
                                flush=True,
                            )

                        else:
                            print(
                                f"\r  {dest.name}: "
                                f"{downloaded / 1e6:.1f} MB",
                                end="",
                                flush=True,
                            )

            print()

            # -------------------------------------------------
            # Verify finished file
            # -------------------------------------------------

            final_size = part.stat().st_size

            if (
                server_size is not None
                and final_size != server_size
            ):
                raise RuntimeError(
                    f"size mismatch: got {final_size}, "
                    f"expected {server_size}"
                )

            # Only rename once verified complete
            part.replace(dest)

            dt = time.time() - t0
            rate = final_size / 1e6 / max(dt, 0.01)

            msg = (
                f"OK    {dest.name}  "
                f"({final_size / 1e6:.1f} MB in "
                f"{dt:.0f}s, {rate:.2f} MB/s)"
            )

            print(msg)
            log.write(msg + "\n")
            log.flush()

            return "ok", final_size

        except KeyboardInterrupt:

            print(
                "\nInterrupted. Partial download saved "
                "as .part — rerun to resume."
            )

            log.write("INTERRUPT\n")
            log.flush()

            raise

        except Exception as e:

            current_size = (
                part.stat().st_size
                if part.exists()
                else 0
            )

            msg = (
                f"RETRY {dest.name} "
                f"attempt {attempt}/{FILE_RETRIES} "
                f"at {current_size / 1e6:.1f} MB "
                f"({type(e).__name__}: {e})"
            )

            print()
            print(msg)

            log.write(msg + "\n")
            log.flush()

            # Out of retries
            if attempt == FILE_RETRIES:
                break

            wait = min(
                FILE_RETRY_WAIT * attempt,
                30
            )

            print(
                f"  Waiting {wait}s, then resuming "
                f"from the partial file..."
            )

            time.sleep(wait)

    # ---------------------------------------------------------
    # All retries exhausted
    # ---------------------------------------------------------

    final_partial = (
        part.stat().st_size
        if part.exists()
        else 0
    )

    msg = (
        f"FAIL  {dest.name} after "
        f"{FILE_RETRIES} attempts "
        f"({final_partial / 1e6:.1f} MB retained)"
    )

    print(msg)
    log.write(msg + "\n")
    log.flush()

    return "fail", 0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksums(zip_dir, checksum_file, log):
    """checksum_file: lines of 'sha256  filename' or 'filename  sha256'."""
    expected = {}
    with open(checksum_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            a, b = parts
            if len(a) == 64:
                expected[b] = a.lower()
            elif len(b) == 64:
                expected[a] = b.lower()
    if not expected:
        print("No valid checksum lines found."); return
    n_ok = n_bad = n_missing = 0
    for name, want in expected.items():
        p = Path(zip_dir) / name
        if not p.exists():
            msg = f"MISS  {name}"; print(msg); log.write(msg + "\n"); n_missing += 1
            continue
        got = sha256_of(p)
        if got == want:
            msg = f"HASH  {name} OK"; print(msg); log.write(msg + "\n"); n_ok += 1
        else:
            msg = f"HASH  {name} MISMATCH (got {got[:12]}…, want {want[:12]}…)"
            print(msg); log.write(msg + "\n"); n_bad += 1
    print(f"\nChecksum summary: {n_ok} ok, {n_bad} mismatch, {n_missing} missing")


def main():
    ap = argparse.ArgumentParser(description="Download Zenodo sample zips.")
    ap.add_argument("--url-file", help="text file with one URL per line")
    ap.add_argument("--out", default="data/zips",
                    help="output directory for zips (default data/zips)")
    ap.add_argument("--start", type=int, default=0,
                    help="start of auto URL range (default 0)")
    ap.add_argument("--end", type=int, default=139,
                    help="end of auto URL range (default 139)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel downloads (default 1; use 3 for faster wifi)")
    ap.add_argument("--verify-sha256",
                    help="optional checksum file to verify after download")
    ap.add_argument("--log", default="download_log.txt")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build URL list
    if args.url_file:
        pairs = load_url_file(args.url_file)
        print(f"Loaded {len(pairs)} URLs from {args.url_file}")
    else:
        pairs = list(gen_zenodo_urls(args.start, args.end))
        print(f"Generated {len(pairs)} URLs for {args.start:04d}..{args.end:04d} + Bg")

    session = make_session()

    with open(args.log, "a") as log:
        log.write(f"\n=== Run {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log.write(f"Output dir: {out_dir}\n")
        log.write(f"Files     : {len(pairs)}\n")

        total_bytes = 0
        counts = {"ok": 0, "skip": 0, "fail": 0}

        if args.workers <= 1:
            # Sequential
            for i, (url, filename) in enumerate(pairs, 1):
                print(f"\n[{i}/{len(pairs)}] {filename}")
                status, n = download_one(session, url, out_dir / filename, log)
                counts[status] = counts.get(status, 0) + 1
                total_bytes += n
        else:
            # Parallel (needs concurrent.futures)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def job(pair):
                url, filename = pair
                print(f"[start] {filename}")
                return filename, download_one(session, url, out_dir / filename, log)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(job, p) for p in pairs]
                for fut in as_completed(futures):
                    try:
                        filename, (status, n) = fut.result()
                        counts[status] = counts.get(status, 0) + 1
                        total_bytes += n
                    except Exception as e:
                        print(f"[parallel] job failed: {e}")
                        counts["fail"] += 1

        log.write(f"\nSummary: {counts}, total {total_bytes/1e9:.2f} GB\n")

    print(f"\n=== Download summary ===")
    print(f"  ok   : {counts['ok']}")
    print(f"  skip : {counts['skip']}")
    print(f"  fail : {counts['fail']}")
    print(f"  total: {total_bytes/1e9:.2f} GB")
    print(f"  log  : {args.log}")

    if args.verify_sha256:
        verify_checksums(out_dir, args.verify_sha256, open(args.log, "a"))


if __name__ == "__main__":
    main()