#!/usr/bin/env python3
"""
NTD Tucker + SAM classification for one HSI data cube.

Pipeline
--------
1. Load cube with load_and_clean (reads /hdr, removes NaNs).
2. Nonnegative Tucker decomposition on the cube.
3. Take the spectral factor U2 as endmembers.
4. SAM each endmember against the SAM-ready library (sam_library.npz).
5. Assign each endmember a mineral label (or 'unknown' if angle > threshold).
6. Per-pixel classification: argmax over the scaled abundance tensor
   A = (G x0 U0 x1 U1) * ||U2[:,k]||  ->  dominant endmember per pixel.
7. Save spatial class map + report.

Fixes included
--------------
- defensive L masking (zero where M==0)
- library entries with low valid-band coverage dropped before SAM
- wavelength alignment check (cube vs library), with nm->um normalisation
- reflectance sanity check on the cube
- library-vs-library angle distribution report (with correct self-match masking)
- argmax on U2 column-normalised abundance tensor
- class-map colormap ordering fixed (gray at index 0)
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
import tensorly as tl
from tensorly.decomposition import non_negative_tucker_hals

tl.set_backend("numpy")
EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def load_and_clean(folder):
    with h5py.File(folder, "r") as f:
        data = f["/hdr"][()]
    data = np.transpose(data, (1, 2, 0))
    width, height, bands = data.shape
    flat = data.reshape(-1, bands)
    for i in range(flat.shape[0]):
        row = flat[i, :]
        nan = np.isnan(row)
        if np.any(nan):
            good = np.where(~nan)[0]
            if len(good) == 0:
                row[:] = 0
            else:
                row[nan] = np.interp(np.where(nan)[0], good, row[good])
            if np.any(np.isnan(row)):
                row[nan] = row[good[0]]
            flat[i, :] = row
    return data.reshape(width, height, bands)


def try_read_cube_wavelengths(folder):
    """Best-effort read of band centres from the h5. Returns None if not found."""
    candidates = ["wavelengths", "wavelength", "wvl", "wl", "lambda",
                  "bands", "band_centers", "bandcentres", "wavelength_um",
                  "wavelengths_um", "wavelength_nm", "wavelengths_nm"]
    try:
        with h5py.File(folder, "r") as f:
            for c in candidates:
                if c in f:
                    return np.asarray(f[c][()], dtype=np.float64).ravel()
            found = {}
            def find(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
                    ln = name.lower()
                    if any(k in ln for k in ("wavelength", "wvl", "lambda")):
                        found[name] = np.asarray(obj[()], dtype=np.float64).ravel()
            f.visititems(find)
            if found:
                return next(iter(found.values()))
    except Exception:
        pass
    return None



# --------------------------------------------------------------------------- #
# 2. NTD
# --------------------------------------------------------------------------- #
def run_ntd(cube, ranks, n_iter=100, seed=0):
    """Nonnegative Tucker decomposition. cube: (W, H, B).

    Uses init='random' with a seeded random_state for reproducibility.
    """
    tucker = non_negative_tucker_hals(
        tl.tensor(cube),
        rank=ranks,
        algorithm="fista",
        n_iter_max=n_iter,
        tol=1e-5,
        init="random",
        random_state=seed,
    )
    U = [np.asarray(u, dtype=np.float64) for u in tucker.factors]
    G = np.asarray(tucker.core, dtype=np.float64)
    return U, G


# --------------------------------------------------------------------------- #
# 3. SAM
# --------------------------------------------------------------------------- #
def sam_angles(X, L, M):
    """
    X: (n, B) endmembers
    L: (m, B) library spectra (zeros where invalid)
    M: (m, B) mask (0/1)
    Returns (n, m) angles in radians.
    """
    dot = X @ L.T
    nx = np.sqrt(np.maximum((X ** 2) @ M.T, 0.0))
    nl = np.sqrt(np.maximum((L ** 2).sum(axis=1), 0.0))
    cosang = dot / (nx * nl[None, :] + EPS)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def library_angle_stats(L, M, names, n_sample=500, seed=0):
    """
    Off-diagonal angles between library spectra.
    Reports percentiles so the user can pick a meaningful threshold.
    """
    m = L.shape[0]
    if m < 2:
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=min(n_sample, m), replace=False)
    sub = L[idx]
    angles = sam_angles(sub, L, M)         # (len(idx), m)

    # correct self-match masking: element-wise assignment, not fancy-index copy
    angles[np.arange(len(idx)), idx] = np.nan

    deg = np.degrees(angles)

    same_mineral = np.zeros_like(deg, dtype=bool)
    for a, i in enumerate(idx):
        same_mineral[a] = (names == names[i])
        same_mineral[a, i] = False

    diff = deg[~same_mineral & np.isfinite(deg)]
    same = deg[same_mineral & np.isfinite(deg)]

    print(f"\nLibrary self-similarity (angles in degrees):")
    if same.size:
        print(f"  same-mineral pairs   : n={same.size:6d}  "
              f"median={np.median(same):6.2f}  p90={np.percentile(same, 90):6.2f}")
    if diff.size:
        print(f"  different-mineral    : n={diff.size:6d}  "
              f"median={np.median(diff):6.2f}  p10={np.percentile(diff, 10):6.2f}")
        print(f"  -> a threshold below the p10 of different-mineral angles "
              f"(~{np.percentile(diff, 10):.1f} deg) is meaningful; "
              f"above it, matches are ambiguous.")


def classify_endmembers(U_spectral, L, M, names, max_angle_deg):
    X = U_spectral.T  # (r, B)
    angles = sam_angles(X, L, M)
    best_idx = angles.argmin(axis=1)
    best_ang = angles[np.arange(len(best_idx)), best_idx]
    labels = np.array(
        [names[i] if np.degrees(a) <= max_angle_deg else "unknown"
         for i, a in zip(best_idx, best_ang)],
        dtype=object,
    )
    return labels, np.degrees(best_ang), best_idx


# --------------------------------------------------------------------------- #
# 4. Pixel-wise classification
# --------------------------------------------------------------------------- #
def pixel_classification(G, U0, U1, U2, endmember_labels):
    """
    Build abundance tensor A = G x0 U0 x1 U1 -> (W, H, r3).
    Scale each endmember slice by ||U2[:,k]|| so argmax compares actual
    contributions to the reconstruction, not raw core coefficients.
    """
    A = tl.tenalg.mode_dot(G, U0, mode=0)
    A = tl.tenalg.mode_dot(A, U1, mode=1)
    A = np.asarray(A, dtype=np.float64)

    u2_norms = np.linalg.norm(U2, axis=0)
    A = A * u2_norms[None, None, :]

    winner = np.argmax(A, axis=2)
    unique_labels = list(dict.fromkeys(endmember_labels))
    label_to_id = {lab: i for i, lab in enumerate(unique_labels) if lab != "unknown"}
    class_map = np.full(winner.shape, -1, dtype=np.int32)
    for r, lab in enumerate(endmember_labels):
        if lab == "unknown":
            continue
        class_map[winner == r] = label_to_id[lab]
    return class_map, unique_labels


# --------------------------------------------------------------------------- #
# 5. Report + plot
# --------------------------------------------------------------------------- #
def write_report(path, endmember_labels, endmember_angles, endmember_best_idx,
                 lib_fnames, class_map, unique_labels, ranks):
    with open(path, "w") as f:
        f.write("NTD + SAM classification report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Tucker ranks     : {ranks}\n")
        f.write(f"Cube shape       : {class_map.shape} pixels\n")
        f.write(f"Num endmembers   : {len(endmember_labels)}\n\n")
        f.write("Endmember classification\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'idx':>4}  {'angle(deg)':>10}  {'mineral':<32}  {'source file'}\n")
        for i, (lab, ang, bi) in enumerate(zip(endmember_labels,
                                                endmember_angles,
                                                endmember_best_idx)):
            f.write(f"{i:>4}  {ang:>10.2f}  {lab:<32}  {lib_fnames[bi]}\n")
        f.write("\nPixel-class fractions\n")
        f.write("-" * 60 + "\n")
        total = class_map.size
        for cid, lab in enumerate(unique_labels):
            n = int((class_map == cid).sum())
            f.write(f"  {lab:<32} {n:>8}  ({n / total:6.2%})\n")
        n_unclass = int((class_map == -1).sum())
        f.write(f"  {'unclassified':<32} {n_unclass:>8}  ({n_unclass / total:6.2%})\n")


def render_class_map(class_map, unique_labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        print("[plot] matplotlib not installed - skipping class map PNG")
        return

    n_classes = len(unique_labels)
    base = plt.get_cmap("tab20").colors
    # gray first, so value -1 (unclassified) maps to index 0
    colors = [(0.5, 0.5, 0.5)] + [base[i % len(base)] for i in range(n_classes)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(class_map.T, cmap=cmap, interpolation="nearest",
              vmin=-1, vmax=n_classes - 1, origin="lower")
    ax.set_title("Pixel-wise mineral classification (argmax of scaled abundance)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    handles = [Patch(color=colors[i + 1], label=unique_labels[i])
               for i in range(n_classes)]
    handles.append(Patch(color=colors[0], label="unclassified"))
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left",
              fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved class map to {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="NTD + SAM classification for one HSI cube.")
    ap.add_argument("--cube", required=True, help="path to the cube .h5")
    ap.add_argument("--library", required=True, help="path to sam_library.npz")
    ap.add_argument("--ranks", nargs=3, type=int, default=[20, 20, 15],
                    metavar=("R0", "R1", "R2"),
                    help="Tucker ranks (spatial, spatial, spectral). "
                         "R2 must be >= expected number of minerals per cube.")
    ap.add_argument("--n-iter", type=int, default=100,
                    help="max iterations (default 100)")
    ap.add_argument("--max-angle-deg", type=float, default=10.0,
                    help="endmember -> 'unknown' if best angle exceeds this")
    ap.add_argument("--min-lib-coverage", type=float, default=0.85,
                    help="drop library spectra with < this fraction of valid bands")
    ap.add_argument("--out-cube", default="classification")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-reflectance-check", action="store_true")
    args = ap.parse_args()

    # ---- cube ----
    print(f"Loading cube  : {args.cube}")
    cube = load_and_clean(args.cube)
    W, H, B = cube.shape
    print(f"Cube shape    : {cube.shape}  ({W}x{H} pixels, {B} bands)")

    if not args.skip_reflectance_check:
        cmin, cmax = float(np.nanmin(cube)), float(np.nanmax(cube))
        print(f"Cube value range: [{cmin:.4g}, {cmax:.4g}]")
        if cmax > 10.0 or cmin < -0.1:
            print("WARNING: cube values are outside the plausible reflectance "
                  "range [0, ~1.5]. SAM assumes reflectance; if this is radiance "
                  "or DN, angles will be meaningless. Pass "
                  "--skip-reflectance-check to suppress this warning.")

    cube_wl = try_read_cube_wavelengths(args.cube)

    # ---- library ----
    print(f"Loading library: {args.library}")
    lib = np.load(args.library, allow_pickle=True)
    L = lib["L"].astype(np.float64)
    M = lib["M"].astype(np.float64)
    names = lib["names"]
    fnames = lib["fnames"]
    wl_lib = lib["wavelengths_um"]

    if L.shape[1] != B:
        raise SystemExit(
            f"Library has {L.shape[1]} bands but cube has {B}. "
            f"Rebuild the library for this cube.")

    # enforce zeros where M == 0
    L = np.where(M > 0, L, 0.0)

    # wavelength alignment check (with nm->um normalisation)
    if cube_wl is not None:
        if np.nanmax(cube_wl) > 100:
            print(f"Wavelength units: cube looks like nm "
                  f"(max={np.nanmax(cube_wl):.0f}); converting to um for check")
            cube_wl = cube_wl / 1000.0

        if cube_wl.size == wl_lib.size and np.allclose(
                cube_wl, wl_lib, atol=1e-3):
            print("Wavelength alignment: OK (cube == library)")
        elif cube_wl.size == wl_lib.size:
            max_dev = float(np.max(np.abs(cube_wl - wl_lib)))
            print(f"WARNING: cube and library wavelengths differ "
                  f"(max deviation {max_dev:.4f} um). Check that the library "
                  f"was built from this cube.")
        else:
            print(f"WARNING: cube has {cube_wl.size} wavelengths, "
                  f"library has {wl_lib.size}. Library may be for another cube.")
    else:
        print("(Cube wavelengths not found in h5; skipping wavelength check.)")

    # filter library entries with low valid-band coverage
    coverage = M.mean(axis=1)
    keep = coverage >= args.min_lib_coverage
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} library spectra with coverage "
              f"< {args.min_lib_coverage:.0%}")
    L = L[keep]
    M = M[keep]
    names = names[keep]
    fnames = fnames[keep]
    print(f"Library       : {L.shape[0]} spectra, "
          f"{len(np.unique(names))} distinct minerals")

    library_angle_stats(L, M, names)

    # ---- NTD ----
    print(f"\nRunning NTD   : ranks = {args.ranks}")
    U, G = run_ntd(cube, args.ranks, n_iter=args.n_iter, seed=args.seed)
    U0, U1, U2 = U
    print(f"  U0 : {U0.shape}  U1 : {U1.shape}  U2 : {U2.shape}  G : {G.shape}")
    print(f"  spectral endmembers : {U2.shape[1]}")

    # ---- SAM on endmembers ----
    print(f"\nSAM on endmembers (max angle {args.max_angle_deg} deg)")
    endmember_labels, endmember_angles, endmember_best_idx = classify_endmembers(
        U2, L, M, names, args.max_angle_deg)
    for i, (lab, ang) in enumerate(zip(endmember_labels, endmember_angles)):
        print(f"  endmember {i:2d} : {ang:6.2f} deg  ->  {lab}")
    n_unknown = int(sum(1 for l in endmember_labels if l == "unknown"))
    if n_unknown > len(endmember_labels) // 2:
        print(f"  NOTE: {n_unknown}/{len(endmember_labels)} endmembers are "
              f"unknown. This usually means U2 columns are mixtures rather "
              f"than pure endmembers.")

    # ---- pixel classification ----
    print("\nPixel-wise classification (argmax of scaled abundance tensor)")
    class_map, unique_labels = pixel_classification(
        G, U0, U1, U2, endmember_labels)
    total = class_map.size
    for cid, lab in enumerate(unique_labels):
        n = int((class_map == cid).sum())
        if n:
            print(f"  {lab:<32} {n:>8}  ({n / total:6.2%})")
    n_unclass = int((class_map == -1).sum())
    print(f"  {'unclassified':<32} {n_unclass:>8}  ({n_unclass / total:6.2%})")

    # ---- save ----
    out_base = Path(args.out_cube)
    np.save(str(out_base) + "_class_map.npy", class_map)
    np.savez_compressed(
        str(out_base) + "_endmembers.npz",
        U0=U0, U1=U1, U2=U2, G=G,
        labels=np.array(endmember_labels, dtype=object),
        angles_deg=endmember_angles,
        best_lib_idx=endmember_best_idx,
        unique_labels=np.array(unique_labels, dtype=object),
        wavelengths_um=wl_lib,
    )
    write_report(str(out_base) + "_report.txt",
                 endmember_labels, endmember_angles, endmember_best_idx,
                 fnames, class_map, unique_labels, args.ranks)
    render_class_map(class_map, unique_labels, str(out_base) + "_class_map.png")

    print(f"\nWrote:")
    print(f"  {out_base}_class_map.npy")
    print(f"  {out_base}_class_map.png")
    print(f"  {out_base}_endmembers.npz")
    print(f"  {out_base}_report.txt")


if __name__ == "__main__":
    main()