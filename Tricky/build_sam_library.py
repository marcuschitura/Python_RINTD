#!/usr/bin/env python3
"""
Build a SAM-ready USGS splib07 Chapter M (Minerals) library.

Sensor wavelengths are read DIRECTLY from your HSI .h5 cube.
USGS splib07 v7 wavelength files live INSIDE ASCIIdata_splib07b/,
alongside the chapter folders.

IMPORTANT ASSUMPTIONS - READ BEFORE TRUSTING RESULTS
-----------------------------------------------------
1. This script assumes your HSI cube is already CALIBRATED REFLECTANCE
   (0-1 or 0-100 scale), not radiance or raw digital numbers. Spectral
   Angle Mapper against a reflectance library is meaningless if your
   cube isn't in reflectance units. Nothing below can verify this for
   you - check your sensor's processing level yourself.
2. Library spectra are resampled onto your sensor's band centers using
   a Gaussian spectral-response-function convolution (approximating
   each band as a Gaussian with the given/estimated FWHM), not naive
   nearest-neighbour interpolation. This matters most near narrow
   absorption features, which are exactly the features mineral
   classification depends on. If you don't supply per-band FWHM values,
   an FWHM is estimated from local band spacing - see --fwhm-key /
   --fwhm-file / --fwhm-factor below.

Typical use (bash / POSIX)
---------------------------
# 1. see what datasets are in your h5
python build_sam_library.py --h5 my_cube.h5 --list-h5

# 2. build (auto-detect wavelength dataset in h5)
python build_sam_library.py --h5 my_cube.h5 \
    --usgs-root "/path/to/usgs_splib07/ASCIIdata/ASCIIdata_splib07b" \
    --out sam_library.npz

# 3. build with plot
python build_sam_library.py --h5 my_cube.h5 \
    --usgs-root "/path/to/usgs_splib07/ASCIIdata/ASCIIdata_splib07b" \
    --plot --plot-minerals Kaolinite Calcite Hematite Montmorillonite \
    --out sam_library.npz

Typical use (Windows cmd.exe)
------------------------------
python build_sam_library.py --h5 my_cube.h5 --list-h5

python build_sam_library.py --h5 my_cube.h5 ^
    --usgs-root "C:/Users/Marcus/Desktop/TuckerDecomp/usgs_splib07/ASCIIdata/ASCIIdata_splib07b" ^
    --out sam_library.npz

python build_sam_library.py --h5 my_cube.h5 ^
    --usgs-root "C:/Users/Marcus/Desktop/TuckerDecomp/usgs_splib07/ASCIIdata/ASCIIdata_splib07b" ^
    --plot --plot-minerals Kaolinite Calcite Hematite Montmorillonite ^
    --out sam_library.npz
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

INVALID_THRESHOLD = -1.0e30

WL_CANDIDATES = [
    "wavelengths", "wavelength", "wvl", "wl", "lambda", "lambdas",
    "bands", "band_centers", "bandcenters", "band_centres", "bandcentres",
    "wavelength_um", "wavelengths_um", "wavelength_nm", "wavelengths_nm",
]

# Wavenumber candidates are tracked SEPARATELY from wavelength candidates
# because they need a different unit conversion (cm^-1 -> um is
# 10000/cm^-1, not a simple /1000 like nm -> um). Mixing them into one
# list and applying the nm heuristic silently corrupts the axis.
WAVENUMBER_CANDIDATES = [
    "wavenumbers", "wavenumber", "wavenumber_cm", "wavenumbers_cm",
]

FWHM_CANDIDATES = [
    "fwhm", "fwhms", "bandwidth", "bandwidths", "band_width", "band_widths",
    "spectral_resolution", "fwhm_um", "fwhm_nm",
]


# --------------------------------------------------------------------------- #
# USGS file reading
# --------------------------------------------------------------------------- #
def read_usgs_txt(path):
    vals = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line.split()[0]))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float64)


def parse_name(path):
    """splib07b_Actinolite_HS22.3B_ASDFRa_AREF.txt -> ('Actinolite','HS22.3B','ASDFRa')"""
    parts = path.stem.split("_")
    first = parts[0].lower()
    if first.startswith("splib07") and len(parts) >= 4:
        return parts[1], "_".join(parts[2:-2]), parts[-2]
    if first.startswith("splib07") and len(parts) > 1:
        return parts[1], "", ""
    if first == "s07" and len(parts) > 2:
        return parts[2], "", ""
    return parts[0], "", ""


def is_convolved(rel_path):
    s = str(rel_path).lower()
    return "_cv" in s or "rebin" in s or "_rs" in s


def find_chapter_m(root, prefer, override=None):
    if override:
        d = Path(override)
        if not d.is_dir():
            sys.exit(f"--chapter-dir is not a folder: {d}")
        return d

    cands = [d for d in root.rglob("*")
             if d.is_dir() and d.name.lower().startswith("chapterm")]
    if root.name.lower().startswith("chapterm"):
        cands.append(root)
    if not cands:
        sys.exit(f"Could not find a 'ChapterM...' folder under {root}")

    def rel(d):
        try:
            return d.relative_to(root)
        except ValueError:
            return d

    cands.sort(key=lambda d: (1 if is_convolved(rel(d)) else 0,
                              0 if f"splib07{prefer}" in str(d).lower() else 1,
                              str(d)))
    if len(cands) > 1:
        print("Chapter M candidates (using first; override with --chapter-dir):")
        for d in cands:
            print(f"   {d}")
    return cands[0]


def detect_version(chapter, prefer):
    s = str(chapter).lower()
    other = "a" if prefer == "b" else "b"
    for v in (prefer, other):
        if f"splib07{v}" in s:
            return v
    return None


def find_wavelength_files(root, spectrometers, version):
    """
    USGS v7 puts wavelength files inside ASCIIdata_splib07X/ alongside
    the chapter folders. Names look like:
      splib07b_Wavelengths_ASDFR_0.35-2.5microns_2151ch.txt
      splib07b_Wavelengths_AVIRIS_1996_interp_to_2203ch.txt
      splib07b_Wavelengths_BECK_Beckman_interp._3961_ch.txt
      splib07b_Wavelengths_NIC4_Nicolet_1.12-216microns.txt

    Accepts a single spectrometer name or a list of names.
    """
    # normalise to a lowercase list of substrings
    if isinstance(spectrometers, str):
        spectrometers = [spectrometers]
    specs = [s.lower() for s in spectrometers]

    def rel(p):
        try:
            return p.relative_to(root)
        except ValueError:
            return p

    files = [p for p in root.rglob("*.txt")
             if "wavelength" in p.name.lower()
             and "bandpass" not in p.name.lower()
             and "fwhm" not in p.name.lower()
             and not is_convolved(rel(p))]

    # keep a file if ANY of the requested spectrometers appears in its name
    files = [p for p in files if any(s in p.name.lower() for s in specs)]

    if version:
        scoped = [p for p in files if f"splib07{version}" in str(p).lower()]
        if scoped:
            files = scoped
        elif files:
            print(f"WARNING: no wavelength file tagged splib07{version}; "
                  f"using untagged/other-version files")

    files.sort(key=str)
    by_len = {}
    for p in files:
        wl = read_usgs_txt(p)
        if len(wl) in by_len:
            other = by_len[len(wl)][1]
            if not np.allclose(wl, other):
                print(f"WARNING: two different wavelength files with "
                      f"{len(wl)} channels; using the first")
            continue
        by_len[len(wl)] = (p, wl)
    return by_len


def find_fwhm_file(root, spectrometers, version):
    """Look for a matching USGS bandpass/FWHM file alongside the
    wavelength files. Returns an array or None if not found - callers
    must fall back to an estimated FWHM in that case."""
    if isinstance(spectrometers, str):
        spectrometers = [spectrometers]
    specs = [s.lower() for s in spectrometers]

    files = [p for p in root.rglob("*.txt")
             if ("bandpass" in p.name.lower() or "fwhm" in p.name.lower())]
    files = [p for p in files if any(s in p.name.lower() for s in specs)]
    if version:
        scoped = [p for p in files if f"splib07{version}" in str(p).lower()]
        if scoped:
            files = scoped
    if not files:
        return None
    files.sort(key=str)
    return read_usgs_txt(files[0])


def spectrum_matches(p, spectrometers, ref_type):
    """Spectrometer is looked for in the sample/instrument part of the
    name only, so a mineral name can never trigger a false match."""
    if isinstance(spectrometers, str):
        spectrometers = [spectrometers]
    specs = [s.lower() for s in spectrometers]
    parts = p.stem.split("_")
    tail = "_".join(parts[2:]) if len(parts) > 2 else p.stem
    return (any(s in tail.lower() for s in specs)
            and ref_type.lower() in p.stem.lower())


# --------------------------------------------------------------------------- #
# H5 wavelength / FWHM discovery
# --------------------------------------------------------------------------- #
def list_h5_datasets(path):
    import h5py
    print(f"Datasets in {path}:")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:50s} shape={obj.shape} dtype={obj.dtype}")
        f.visititems(show)


def find_cube_dataset(f, verbose=True):
    import h5py
    candidates = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3 \
                and np.issubdtype(obj.dtype, np.number):
            candidates.append((name, obj.shape, int(np.prod(obj.shape))))
    f.visititems(visit)

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[2], reverse=True)
    if verbose and len(candidates) > 1:
        print("3D numeric dataset candidates (using largest; pass "
              "--cube-key to override):")
        for name, shape, _ in candidates:
            print(f"   {name:40s} shape={shape}")
    return candidates[0][0]


def find_wl_dataset(f, cube_shape):
    import h5py
    one_d = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1 \
                and np.issubdtype(obj.dtype, np.number):
            one_d.append((name, obj.shape[0]))
    f.visititems(visit)

    lower_map = {name.lower(): name for name, _ in one_d}
    for cand in WL_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand], "wavelength"
    for cand in WAVENUMBER_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand], "wavenumber"
    for name, _ in one_d:
        ln = name.lower()
        if "wavenumb" in ln:
            return name, "wavenumber"
        if any(c in ln for c in ("wavelength", "wvl", "lambda")):
            return name, "wavelength"
    for name, n in one_d:
        if n in cube_shape:
            return name, "wavelength"
    return None, None


def find_fwhm_dataset(f, n_bands):
    import h5py
    one_d = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 1 \
                and np.issubdtype(obj.dtype, np.number):
            one_d.append((name, obj.shape[0]))
    f.visititems(visit)

    lower_map = {name.lower(): name for name, _ in one_d}
    for cand in FWHM_CANDIDATES:
        if cand in lower_map and lower_map[cand] is not None:
            name = lower_map[cand]
            if dict(one_d)[name] == n_bands:
                return name
    return None


def get_cube_and_wavelengths(h5_path, wl_key=None, cube_key=None, fwhm_key=None):
    import h5py
    with h5py.File(h5_path, "r") as f:
        cube_name = cube_key if cube_key else find_cube_dataset(f)
        if cube_name is None:
            sys.exit(f"No 3D numeric dataset found in {h5_path}")
        if cube_name not in f:
            sys.exit(f"--cube-key '{cube_name}' not found in {h5_path}")
        cube_shape = f[cube_name].shape
        print(f"Cube dataset     : {cube_name}  shape={cube_shape}")

        if wl_key:
            if wl_key not in f:
                sys.exit(f"--wl-key '{wl_key}' not found. "
                         f"Run with --list-h5 to see datasets.")
            wl_raw = np.asarray(f[wl_key][()], dtype=np.float64).ravel()
            wl_name = wl_key
            # can't infer axis type from name when user forces the key;
            # assume wavelength unless it's clearly wavenumber-scaled
            axis_type = "wavenumber" if "wavenumb" in wl_key.lower() else "wavelength"
        else:
            wl_name, axis_type = find_wl_dataset(f, cube_shape)
            if wl_name is None:
                sys.exit("Could not auto-detect a wavelength dataset. "
                         "Run --list-h5, then pass --wl-key NAME.")
            wl_raw = np.asarray(f[wl_name][()], dtype=np.float64).ravel()

        B = len(wl_raw)
        fwhm = None
        fwhm_name = fwhm_key if fwhm_key else find_fwhm_dataset(f, B)
        if fwhm_name:
            if fwhm_name not in f:
                sys.exit(f"--fwhm-key '{fwhm_name}' not found in {h5_path}")
            fwhm = np.asarray(f[fwhm_name][()], dtype=np.float64).ravel()
            if len(fwhm) != B:
                print(f"WARNING: FWHM dataset '{fwhm_name}' has {len(fwhm)} "
                      f"entries but wavelength has {B}; ignoring")
                fwhm = None
                fwhm_name = None

    if B not in cube_shape:
        sys.exit(f"Wavelength length ({B}) does not match any cube axis "
                 f"({cube_shape}). Wrong --wl-key?")

    if axis_type == "wavenumber":
        print(f"Spectral axis    : wavenumber detected (cm^-1); "
              f"converting to um via 10000/cm^-1")
        wl = 10000.0 / wl_raw
        # FWHM in cm^-1 does not convert with a simple scalar factor
        # (dlambda = 10000/wn^2 * dwn varies per band) - convert per band.
        if fwhm is not None:
            fwhm = (10000.0 / (wl_raw ** 2)) * fwhm
    else:
        if np.nanmax(wl_raw) > 100:
            print(f"Wavelength units : nm detected (max={np.nanmax(wl_raw):.0f}); "
                  f"converting to um")
            wl = wl_raw / 1000.0
            if fwhm is not None and np.nanmax(fwhm) > 1:
                fwhm = fwhm / 1000.0
        else:
            print(f"Wavelength units : um (max={np.nanmax(wl_raw):.4f})")
            wl = wl_raw

    print(f"Wavelength dataset: {wl_name}  ({B} bands, "
          f"{wl.min():.4f}-{wl.max():.4f} um)")
    if fwhm_name:
        print(f"FWHM dataset     : {fwhm_name}  "
              f"(mean={np.nanmean(fwhm):.5f} um)")
    else:
        print("FWHM dataset     : not found - will estimate from band "
              "spacing (see --fwhm-factor)")
    return cube_shape, wl, fwhm, cube_name, wl_name


def estimate_fwhm(wl, factor=1.0):
    """Estimate per-band FWHM from local sample spacing when the sensor
    doesn't supply one. FWHM ~= factor * local band spacing is a common
    rule of thumb for roughly-contiguous imaging spectrometers."""
    d = np.gradient(wl)
    return np.abs(d) * factor


# --------------------------------------------------------------------------- #
# Gaussian spectral-response-function resampling
# --------------------------------------------------------------------------- #
def gaussian_srf_resample(wl_ok, y_ok, sensor_wl, sensor_fwhm, max_gap):
    """
    Resample a library spectrum (wl_ok, y_ok) onto sensor_wl using a
    Gaussian spectral response function per output band, rather than
    plain linear interpolation. This approximates the physical
    integration a sensor performs over its bandpass and is noticeably
    more accurate near narrow absorption features.

    Returns (resampled_values, valid_mask) - valid_mask is False for
    an output band if the library doesn't have coverage within
    max_gap of that band (same semantics as the old nearest-distance
    check), so downstream masking behaviour is unchanged.
    """
    n_out = len(sensor_wl)
    out = np.zeros(n_out, dtype=np.float64)
    valid = np.zeros(n_out, dtype=bool)

    # sigma from FWHM: FWHM = 2*sqrt(2*ln2)*sigma
    sigma = np.maximum(sensor_fwhm, 1e-6) / 2.3548200450309493

    for i, (center, s) in enumerate(zip(sensor_wl, sigma)):
        # only integrate over library points within +-3 sigma (or at
        # least max_gap, whichever is larger) of the band center - far
        # outside that a Gaussian's weight is negligible anyway
        half_window = max(3.0 * s, max_gap)
        lo, hi = center - half_window, center + half_window
        in_window = (wl_ok >= lo) & (wl_ok <= hi)

        if not in_window.any():
            continue

        # coverage check: nearest library sample must be within max_gap
        # of the band center, same rule as before, so sparse/missing
        # regions of the library still get excluded rather than
        # extrapolated across
        nearest_dist = np.min(np.abs(wl_ok[in_window] - center))
        if nearest_dist > max_gap:
            continue

        w = np.exp(-0.5 * ((wl_ok[in_window] - center) / s) ** 2)
        wsum = w.sum()
        if wsum <= 0:
            continue

        out[i] = np.dot(w, y_ok[in_window]) / wsum
        valid[i] = True

    return out, valid


# --------------------------------------------------------------------------- #
# Library -> sensor bands
# --------------------------------------------------------------------------- #
def build_library(root, sensor_wl, sensor_fwhm, spectrometer, ref_type, prefer,
                  min_cov, max_gap, chapter_override=None):
    chapter = find_chapter_m(root, prefer, chapter_override)
    version = detect_version(chapter, prefer)
    if version and version != prefer:
        print(f"NOTE: requested splib07{prefer} but found splib07{version}; "
              f"using splib07{version}")

    wl_by_len = find_wavelength_files(root, spectrometer, version)
    if not wl_by_len:
        sys.exit(f"No wavelength file with '{spectrometer}' in its name found "
                 f"under {root}")

    all_files = sorted(chapter.rglob("*.txt"))
    files = [p for p in all_files if spectrum_matches(p, spectrometer, ref_type)]

    print(f"Chapter M folder : {chapter}")
    print(f"splib version    : {version or 'unknown'}")
    print(f"Candidate spectra: {len(files)} of {len(all_files)} files "
          f"(spectrometer='{spectrometer}', type='{ref_type}')")
    print("Wavelength files available:")
    for n, (wp, wl) in sorted(wl_by_len.items()):
        print(f"   {n:5d} ch  {wl.min():.4f}-{wl.max():.4f} um  {wp.name}")

    rows, masks, names, samples, instruments, fnames = [], [], [], [], [], []
    skipped = Counter()
    wl_used = Counter()

    skipped["excluded: spectrometer/ref_type filter"] = len(all_files) - len(files)

    for p in files:
        y = read_usgs_txt(p)
        entry = wl_by_len.get(len(y))
        if entry is None:
            skipped[f"no wavelength file with matching length ({len(y)})"] += 1
            continue
        wl_path, wl = entry

        ok = np.isfinite(y) & (y > INVALID_THRESHOLD)
        if ok.sum() < 5:
            skipped["too few valid channels"] += 1
            continue

        order = np.argsort(wl[ok])
        wl_ok, y_ok = wl[ok][order], y[ok][order]
        if np.any(np.diff(wl_ok) <= 0):
            skipped["duplicate/non-increasing wavelengths after sort"] += 1
            continue

        r, inside = gaussian_srf_resample(wl_ok, y_ok, sensor_wl, sensor_fwhm,
                                          max_gap)

        if inside.mean() < min_cov:
            skipped[f"covers < {min_cov:.0%} of your bands"] += 1
            continue

        if np.linalg.norm(r[inside]) == 0:
            skipped["all-zero after resampling"] += 1
            continue

        mineral, sample, instr = parse_name(p)
        rows.append(r * inside)
        masks.append(inside.astype(np.float64))
        names.append(mineral)
        samples.append(sample)
        instruments.append(instr)
        fnames.append(p.name)
        wl_used[wl_path.name] += 1

    if not rows:
        sys.exit(f"No usable spectra. Skipped: {dict(skipped)}")

    L = np.asarray(rows, dtype=np.float32)
    M = np.asarray(masks, dtype=np.float32)

    names = np.asarray(names)
    samples = np.asarray(samples)
    instruments = np.asarray(instruments)
    fnames = np.asarray(fnames)

    print(f"Usable spectra   : {len(L)}  ({len(set(names))} distinct minerals)")
    for wn, c in wl_used.items():
        print(f"   {c:5d} spectra used wavelength file {wn}")
    if skipped:
        print("Skipped          :")
        for reason, c in skipped.most_common():
            if c:
                print(f"   {c:5d}  {reason}")
    print(f"Example names    : {', '.join(sorted(set(names))[:8])} ...")

    meta = dict(chapter_dir=str(chapter), splib_version=version or "unknown")
    return L, M, names, samples, instruments, fnames, meta


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_library(L, M, names, fnames, sensor_wl, imbalance_ratio=10.0):
    print("\n" + "=" * 70)
    print("LIBRARY VERIFICATION")
    print("=" * 70)
    ok = True
    n, B = L.shape

    print("\n[1] Shapes")
    print(f"    L      : {L.shape}")
    print(f"    M      : {M.shape}")
    print(f"    names  : {names.shape}")
    print(f"    fnames : {fnames.shape}")
    print(f"    wl     : {sensor_wl.shape}")
    if M.shape != L.shape:
        print("    FAIL: M and L must have same shape"); ok = False
    if len(names) != n or len(fnames) != n:
        print("    FAIL: names/fnames length must match L rows"); ok = False
    if len(sensor_wl) != B:
        print("    FAIL: sensor wavelength count must match L columns"); ok = False

    print("\n[2] NaN / Inf")
    nan_L, inf_L, nan_M = np.isnan(L).sum(), np.isinf(L).sum(), np.isnan(M).sum()
    print(f"    NaN in L : {nan_L}   Inf in L : {inf_L}   NaN in M : {nan_M}")
    if nan_L or inf_L or nan_M:
        print("    FAIL: non-finite values present"); ok = False

    print("\n[3] Value range (valid entries only)")
    vals = L[M > 0]
    print(f"    min : {vals.min():.4f}   max : {vals.max():.4f}")
    n_neg = int((vals < 0).sum()); n_big = int((vals > 1.5).sum())
    if n_neg:
        print(f"    WARN: {n_neg} valid values are negative")
    if n_big:
        print(f"    WARN: {n_big} valid values exceed 1.5")

    print("\n[4] Validity coverage per spectrum")
    vb = M.sum(axis=1)
    print(f"    valid bands: min {int(vb.min())}  max {int(vb.max())}  "
          f"mean {vb.mean():.1f} / {B}")
    if vb.min() == 0:
        print("    FAIL: at least one spectrum has zero valid bands"); ok = False

    print("\n[5] Row norms")
    norms = np.linalg.norm(L, axis=1)
    print(f"    min : {norms.min():.4f}   max : {norms.max():.4f}")
    if norms.min() == 0:
        print("    FAIL: at least one spectrum has zero norm"); ok = False

    print("\n[6] Mask consistency")
    masked_but_nonzero = int(((M == 0) & (L != 0)).sum())
    print(f"    masked bands with non-zero value : {masked_but_nonzero}")
    if masked_but_nonzero:
        print("    FAIL: masked bands must be exactly zero"); ok = False

    print("\n[7] Mineral diversity and class balance")
    uniq = np.unique(names)
    counts = Counter(names.tolist())
    print(f"    distinct minerals : {len(uniq)}   total spectra : {n}")
    for name, c in counts.most_common(10):
        print(f"      {name:32s} {c:4d}")
    if len(uniq) < 5:
        print("    WARN: very few distinct minerals")
    if counts:
        max_c, min_c = max(counts.values()), min(counts.values())
        ratio = max_c / max(min_c, 1)
        print(f"    class counts range {min_c}-{max_c} spectra "
              f"(ratio {ratio:.1f}x)")
        if ratio >= imbalance_ratio:
            print(f"    WARN: class imbalance ratio >= {imbalance_ratio:.0f}x; "
                  f"if this library is used to train/reference a classifier, "
                  f"under-represented minerals may be poorly learned")

    print("\n[8] Name parsing sanity")
    weird = [nm for nm in uniq if any(ch.isdigit() for ch in nm) or len(nm) < 3]
    if weird:
        print(f"    WARN: malformed names: {weird[:10]}")
    else:
        print("    all names look plausible")

    print("\n[9] Wavelength range")
    print(f"    sensor wl : {sensor_wl.min():.4f} - {sensor_wl.max():.4f} um")
    if np.any(np.diff(sensor_wl) <= 0):
        print("    WARN: sensor wavelengths are not strictly increasing")

    print("\n[10] Per-band coverage")
    cov = M.mean(axis=0)
    print(f"    min {cov.min():.1%}   median {np.median(cov):.1%}   "
          f"max {cov.max():.1%}")
    poor = np.where(cov < 0.5)[0]
    if poor.size:
        print(f"    WARN: {poor.size} bands valid for < 50% of spectra")

    print("\n" + "=" * 70)
    print(f"VERIFICATION {'PASSED' if ok else 'FAILED'}")
    print("=" * 70)
    return ok


def plot_minerals(L, M, names, sensor_wl, minerals, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] matplotlib not installed - skipping")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    found_any = False
    for mineral in minerals:
        idx = np.where(names == mineral)[0]
        if len(idx) == 0:
            idx = np.where(np.array([mineral.lower() in str(nm).lower()
                                     for nm in names]))[0]
        if len(idx) == 0:
            print(f"[plot] '{mineral}' not found - skipped"); continue
        cnt = M[idx].sum(axis=0)
        tot = L[idx].sum(axis=0)
        mean_spec = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        ax.plot(sensor_wl, mean_spec, label=f"{mineral} (n={len(idx)})")
        found_any = True
    if not found_any:
        print("[plot] no requested minerals found"); plt.close(fig); return
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Reflectance (mask-aware mean)")
    ax.set_title("Library spectra for requested minerals")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[plot] saved to {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Build a SAM-ready USGS splib07 Chapter M library "
                    "(wavelengths read from your HSI .h5)")
    ap.add_argument("--h5",
                help="your HSI cube .h5 file (optional if --wavelengths given)")
    ap.add_argument("--wavelengths",
                help="text file with band centres (um or nm)")
    ap.add_argument("--wl-key",
                    help="dataset name of band wavelengths in --h5 "
                         "(auto-detected if omitted)")
    ap.add_argument("--cube-key",
                    help="dataset name of the HSI cube in --h5 "
                         "(auto-detected as the largest 3D numeric dataset "
                         "if omitted - use --list-h5 to check the guess)")
    ap.add_argument("--fwhm-key",
                    help="dataset name of per-band FWHM (bandwidth) in --h5, "
                         "same units as wavelength (auto-detected if present)")
    ap.add_argument("--fwhm-factor", type=float, default=1.0,
                    help="when no FWHM dataset is found/given, estimate "
                         "FWHM as this factor times local band spacing "
                         "(default: 1.0)")
    ap.add_argument("--list-h5", action="store_true",
                    help="print datasets in --h5 and exit")
    ap.add_argument("--usgs-root",
                    help="folder containing the USGS splib07 data "
                         "(e.g. .../ASCIIdata/ASCIIdata_splib07b)")
    ap.add_argument("--chapter-dir",
                    help="explicit path to ChapterM_Minerals")
    ap.add_argument("--spectrometer", nargs="+", default=["ASD"],
                help="one or more of: ASD, BECK, NIC4, AVIRIS "
                     "(default: ASD only). Example: --spectrometer ASD BECK")
    ap.add_argument("--splib", default="b", choices=["a", "b"])
    ap.add_argument("--ref", default="AREF",
                    help="AREF (absolute) or RREF (relative)")
    ap.add_argument("--min-wl", type=float, help="crop bands: lower (um)")
    ap.add_argument("--max-wl", type=float, help="crop bands: upper (um)")
    ap.add_argument("--min-coverage", type=float, default=0.95,
                    help="library must cover this fraction of your bands")
    ap.add_argument("--max-gap-um", type=float, default=0.02,
                    help="ignore band if nearest valid channel > this (um)")
    ap.add_argument("--imbalance-ratio", type=float, default=10.0,
                    help="warn if max/min per-mineral spectrum count "
                         "exceeds this ratio (default: 10.0)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-minerals", nargs="+",
                    default=["Kaolinite", "Calcite", "Hematite"])
    ap.add_argument("--plot-out", default="library_check.png")
    ap.add_argument("--out", default="sam_library.npz")
    ap.add_argument("--mat", help="also write a MATLAB .mat (needs scipy)")
    args = ap.parse_args()

    if args.list_h5:
        if not args.h5:
            sys.exit("--list-h5 needs --h5 FILE")
        list_h5_datasets(args.h5)
        return

    if not args.usgs_root and not args.chapter_dir:
        sys.exit("Provide --usgs-root (or --chapter-dir)")

    # read wavelengths (+ optional FWHM) from --wavelengths file if given,
    # else from --h5
    fwhm_all = None
    if args.wavelengths:
        with open(args.wavelengths) as f:
            wl_all = np.array([float(line.split()[0]) for line in f if line.strip()])
        if wl_all.size == 0:
            sys.exit(f"No numbers found in {args.wavelengths}")
        if wl_all.max() > 100:                 # nm -> um
            wl_all = wl_all / 1000.0
        cube_name = "(none)"
        wl_name = args.wavelengths
    else:
        if not args.h5:
            sys.exit("Need either --wavelengths FILE or --h5 FILE [--wl-key NAME]")
        cube_shape, wl_all, fwhm_all, cube_name, wl_name = get_cube_and_wavelengths(
            args.h5, args.wl_key, args.cube_key, args.fwhm_key)

    # optional crop
    keep = np.ones(len(wl_all), dtype=bool)
    if args.min_wl is not None:
        keep &= wl_all >= args.min_wl
    if args.max_wl is not None:
        keep &= wl_all <= args.max_wl
    wl = wl_all[keep]
    fwhm = fwhm_all[keep] if fwhm_all is not None else None
    if wl.size == 0:
        sys.exit("Cropping removed all bands.")
    print(f"Sensor bands     : {len(wl_all)} -> using {len(wl)}  "
          f"({wl.min():.3f}-{wl.max():.3f} um)")
    if np.any(np.diff(wl) <= 0):
        print("WARNING: sensor wavelengths are not strictly increasing")

    if fwhm is None:
        fwhm = estimate_fwhm(wl, args.fwhm_factor)
        print(f"FWHM             : estimated from band spacing "
              f"(factor={args.fwhm_factor}, mean={fwhm.mean():.5f} um)")

    # build
    if not args.usgs_root:
        sys.exit("--usgs-root is required (used to find wavelength files)")
    root = Path(args.usgs_root)
    L, M, names, samples, instruments, fnames, meta = build_library(
        root, wl, fwhm, args.spectrometer, args.ref, args.splib,
        args.min_coverage, args.max_gap_um, args.chapter_dir)

    # save
    np.savez_compressed(
        args.out,
        L=L, M=M, names=names, samples=samples, instruments=instruments,
        fnames=fnames, wavelengths_um=wl, fwhm_um=fwhm,
        spectrometer=args.spectrometer, reflectance_type=args.ref,
        splib_version=meta["splib_version"], chapter_dir=meta["chapter_dir"],
        source_h5=str(args.h5), source_wl_key=wl_name, source_cube_key=cube_name,
    )
    print(f"\nSaved library to: {args.out}")
    print(f"  L  shape : {L.shape}")
    print(f"  M  shape : {M.shape}")
    print(f"  names    : {len(np.unique(names))} distinct minerals "
          f"across {len(names)} spectra")

    if args.mat:
        try:
            from scipy.io import savemat
            savemat(args.mat, {
                "L": L, "M": M,
                "names": names.astype(object),
                "samples": samples.astype(object),
                "instruments": instruments.astype(object),
                "fnames": fnames.astype(object),
                "wavelengths_um": wl,
                "fwhm_um": fwhm,
            })
            print(f"Saved MATLAB file to: {args.mat}")
        except ImportError:
            print("scipy not installed - skipping --mat")

    ok = True
    data = np.load(args.out)
    if not args.no_verify:
        ok = verify_library(data["L"], data["M"], data["names"],
                            data["fnames"], data["wavelengths_um"],
                            args.imbalance_ratio)
    if args.plot:
        plot_minerals(data["L"], data["M"], data["names"],
                      data["wavelengths_um"], args.plot_minerals,
                      args.plot_out)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()