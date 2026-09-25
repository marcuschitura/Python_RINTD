# The fundamentals of this algorithm are adapted from the following paper:
# Zdunek, R.; Fonał, K.
# Incremental Nonnegative Tucker
# Decomposition with BlockCoordinate Descent and Recursive
# Approaches. Symmetry 2022, 14, 113.
# https://doi.org/10.3390/
# An adaptive factor is used to reduce the influence of previous information
# when the core and factor matrices are updated at each time step
#
# The hyper spectral data used in this script is from the following database:
# Fasnacht, L., Vogt, ML., Renard, P. et al.
# A 2D hyperspectral library of mineral reflectance, from 900 to 2500 nm.
# Sci Data 6, 268 (2019). https://doi.org/10.1038/s41597-019-0261-9

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.signal import savgol_filter
import tensorly as tl
from tensorly.decomposition import non_negative_tucker_hals
import h5py

tl.set_backend('numpy')
EPS = 1e-12

# Per-preprocess angle thresholds. Raw reflectance produces tiny angles
# (1-5 deg for a good match) so 10 is generous. SNV and derivatives inflate
# the angles globally, so the cutoff has to scale with them.
PREPROCESS_ANGLE_THRESHOLD = {
    "raw":    10.0,
    "snv":    30.0,
    "cr":     15.0,
    "deriv":  40.0,
    "deriv2": 50.0,
}


# --------------------------------------------------------------------------- #
# Pre-processing (NaN-safe)
# --------------------------------------------------------------------------- #
def continuum_remove(wl, spec):
    out = np.full_like(spec, np.nan, dtype=np.float64)
    valid = np.isfinite(spec)
    if valid.sum() < 3:
        return out
    x, y = wl[valid], spec[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    hx, hy = [x[0]], [y[0]]
    for i in range(1, len(x)):
        while len(hx) >= 2:
            x1, y1 = hx[-2], hy[-2]
            x2, y2 = hx[-1], hy[-1]
            if (x2 - x1) * (y[i] - y1) - (y2 - y1) * (x[i] - x1) >= 0:
                hx.pop(); hy.pop()
            else:
                break
        hx.append(x[i]); hy.append(y[i])
    out[valid] = spec[valid] / np.maximum(np.interp(x, hx, hy), EPS)
    return out


def snv(spec):
    out = np.full_like(spec, np.nan, dtype=np.float64)
    valid = np.isfinite(spec)
    if valid.sum() < 3:
        return out
    m = float(spec[valid].mean())
    s = float(spec[valid].std())
    if s < EPS:
        return out
    out[valid] = (spec[valid] - m) / s
    return out


def deriv(wl, spec, order=1, window=11, poly=2):
    out = np.full_like(spec, np.nan, dtype=np.float64)
    valid = np.isfinite(spec)
    if valid.sum() < window + 1:
        return out
    idx = np.where(valid)[0]
    y = np.interp(np.arange(len(spec)), idx, spec[idx])
    if window % 2 == 0:
        window += 1
    if window <= poly:
        window = poly + 2
    d = savgol_filter(y, window, poly, deriv=order, delta=1.0)
    out[valid] = d[valid]
    return out


def preprocess(spec, wl, method, deriv_window=11, deriv_poly=2):
    if method == "raw":
        return spec.copy()
    if method == "snv":
        return snv(spec)
    if method == "cr":
        return continuum_remove(wl, spec)
    if method == "deriv":
        return deriv(wl, spec, order=1, window=deriv_window, poly=deriv_poly)
    if method == "deriv2":
        return deriv(wl, spec, order=2, window=deriv_window, poly=deriv_poly)
    raise ValueError(f"unknown preprocess method: {method}")


def preprocess_library(L, wl, method, deriv_window=11, deriv_poly=2):
    out = np.full_like(L, np.nan, dtype=np.float64)
    for i in range(L.shape[0]):
        out[i] = preprocess(L[i], wl, method, deriv_window, deriv_poly)
    return out


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_and_clean(folder):
    with h5py.File(folder, "r") as f:
        data = f["/hdr"][()]
    data = np.transpose(data, (1, 2, 0))    # (W, H, B)
    is_masked = ".mhdr." in str(folder).lower()
    W, H, B = data.shape
    flat = data.reshape(-1, B)

    if is_masked:
        nan_per_pixel = np.isnan(flat).sum(axis=1)
        all_nan = nan_per_pixel == B
        partial = (~all_nan) & (nan_per_pixel > 0)
        for i in np.where(partial)[0]:
            row = flat[i]
            m = np.isnan(row)
            g = np.where(~m)[0]
            if len(g) > 0:
                row[m] = np.interp(np.where(m)[0], g, row[g])
                flat[i] = row
        flat[all_nan] = 0.0
    else:
        for i in range(flat.shape[0]):
            row = flat[i]
            m = np.isnan(row)
            if m.any():
                g = np.where(~m)[0]
                if len(g) == 0:
                    row[:] = 0
                else:
                    row[m] = np.interp(np.where(m)[0], g, row[g])
                flat[i] = row
    return flat.reshape(W, H, B)


def cube_generator(folder):
    for path in sorted(Path(folder).rglob('*.h5')):
        yield str(path), load_and_clean(str(path))


def parse_description(txt_path):
    p = Path(txt_path)
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s*name\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def find_description_for(cube_path):
    cube = Path(cube_path)
    base = cube.name.split("_")[0]
    candidate = cube.parent / f"{base}_description.txt"
    if candidate.is_file():
        return candidate
    others = list(cube.parent.glob("*_description.txt"))
    return others[0] if others else None


# --------------------------------------------------------------------------- #
# Library + SAM
# --------------------------------------------------------------------------- #
def load_library(path, min_coverage=0.85):
    lib = np.load(path, allow_pickle=True)
    L = lib["L"].astype(np.float64)
    M = lib["M"].astype(np.float64)
    names = np.array([str(n) for n in lib["names"]])
    fnames = np.array([str(f) for f in lib["fnames"]])
    wl = np.asarray(lib["wavelengths_um"], dtype=np.float64)

    L = np.where(M > 0, L, np.nan)
    keep = M.mean(axis=1) >= min_coverage
    n_dropped = int((~keep).sum())
    return L[keep], M[keep], names[keep], fnames[keep], wl, n_dropped


def sam_angles(X, L):
    n, B = X.shape
    m = L.shape[0]
    angles = np.full((n, m), np.nan, dtype=np.float64)
    lib_valid = ~np.isnan(L)
    for i in range(n):
        xi = X[i]
        xi_finite = np.isfinite(xi)
        for j in range(m):
            valid = xi_finite & lib_valid[j]
            if int(valid.sum()) < 3:
                continue
            a = xi[valid]
            b = L[j, valid]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < EPS or nb < EPS:
                continue
            c = float(np.dot(a, b) / (na * nb))
            angles[i, j] = float(np.arccos(np.clip(c, -1.0, 1.0)))
    return angles


def classify_endmembers(U2, L, names, max_angle_deg, top_k=5):
    X = U2.T
    angles = sam_angles(X, L)
    out = []
    for i in range(X.shape[0]):
        row = angles[i]
        finite = np.isfinite(row)
        if not finite.any():
            out.append({"endmember_idx": i, "label": "unknown",
                        "top_k": [], "all_angles_nan": True})
            continue
        order = np.argsort(np.where(finite, row, np.inf))
        top = []
        for j in order:
            if len(top) >= top_k:
                break
            if not finite[j]:
                break
            top.append({
                "name": names[j],
                "angle_deg": float(np.degrees(row[j])),
                "lib_idx": int(j),
            })
        best = top[0]
        out.append({
            "endmember_idx": i,
            "label": best["name"] if best["angle_deg"] <= max_angle_deg
                     else "unknown",
            "top_k": top,
            "all_angles_nan": False,
        })
    return out


# --------------------------------------------------------------------------- #
# Online Tucker (streamed mode is now configurable)
# --------------------------------------------------------------------------- #
def recursive_update(X_n, U, G, P, Q, start, stream_axis=None,
                     nonneg=True, k_inner=10, gamma=1e-3, lam=1):
    N = len(U)
    if stream_axis is None:
        stream_axis = N - 1
    last = stream_axis
    block_size = X_n.shape[last]

    # --- initialize the newly-arrived rows of the streamed factor via NNLS ---
    W = G
    for m in range(N):
        if m != last:
            W = tl.tenalg.mode_dot(W, U[m], mode=m)
    Wn = tl.unfold(W, mode=last)
    Xn = tl.unfold(X_n, mode=last)

    A_last = Wn @ Wn.T
    U_new = np.zeros((block_size, Wn.shape[0]))
    for row in range(block_size):
        sol, _ = nnls(A_last, Wn @ Xn[row, :])
        U_new[row, :] = sol
    U[last][start:start + block_size, :] = U_new
    U_block_last = U_new

    # --- refine the other factors via recursive least squares ---
    for n in range(N):
        if n == last:
            continue
        W = G
        for m in range(N):
            if m != n and m != last:
                W = tl.tenalg.mode_dot(W, U[m], mode=m)
        W = tl.tenalg.mode_dot(W, U_block_last, mode=last)
        Xn = tl.unfold(X_n, mode=n)
        Wn = tl.unfold(W, mode=n)
        P[n] = lam * P[n] + Xn @ Wn.T
        Q[n] = lam * Q[n] + Wn @ Wn.T
        for _ in range(k_inner):
            Q_reg = Q[n] + gamma * np.eye(Q[n].shape[0])
            for j in range(U[n].shape[1]):
                num = P[n][:, j] - U[n] @ Q_reg[:, j]
                U[n][:, j] = U[n][:, j] + num / (Q_reg[j, j] + EPS)
                if nonneg:
                    U[n][:, j] = np.maximum(U[n][:, j], EPS)
        U[n] = U[n] / (U[n].sum(axis=0, keepdims=True) + EPS)

    # --- update the core from the incoming block ---
    Z_core = tl.tensor(X_n)
    grams = []
    for n in range(N):
        U_n = U_block_last if n == last else U[n]
        Z_core = tl.tenalg.mode_dot(Z_core, U_n.T, mode=n)
        grams.append(U_n.T @ U_n)
    A_core = grams[0]
    for g in grams[1:]:
        A_core = np.kron(A_core, g)
    z_vec = Z_core.reshape(-1)
    if nonneg:
        g_vec, _ = nnls(A_core, z_vec)
    else:
        g_vec = np.linalg.solve(A_core, z_vec)
    return U, g_vec.reshape(G.shape), P, Q


def decompose_cube(data, ranks, Lt=100, L=100, theta=10, n_iter_max=30,
                   seed=0, stream_axis=0, verbose=True):
    """
    Stream the online Tucker decomposition along `stream_axis`.
    `stream_axis` is an axis index into `data` (e.g. 0 for W/height,
    1 for H/width, 2 for B/bands).
    """
    overlap = int(L * theta / 100)
    N = data.ndim
    if stream_axis is None:
        stream_axis = N - 1
    if not (0 <= stream_axis < N):
        raise ValueError(f"stream_axis={stream_axis} out of range for "
                         f"{N}-D data")

    def _slice(lo, hi):
        s = [slice(None)] * N
        s[stream_axis] = slice(lo, hi)
        return tuple(s)

    # ---- burn-in on the first Lt slices of the streamed axis ----
    X_burn = data[_slice(0, Lt)]
    tensor_hals = non_negative_tucker_hals(
        X_burn, rank=ranks, algorithm='fista',
        n_iter_max=n_iter_max, tol=1e-5, init='random',
        random_state=seed,
    )
    U = list(tensor_hals.factors)
    G = np.asarray(tensor_hals.core)
    last = stream_axis

    # ---- seed the P/Q accumulators for the non-streamed factors ----
    P = [None] * N
    Q = [None] * N
    for n in range(N):
        if n == last:
            continue
        W = G
        for m in range(N):
            if m != n and m != last:
                W = tl.tenalg.mode_dot(W, U[m], mode=m)
        W = tl.tenalg.mode_dot(W, U[last], mode=last)
        Xn = tl.unfold(X_burn, mode=n)
        Wn = tl.unfold(W, mode=n)
        P[n] = Xn @ Wn.T
        Q[n] = Wn @ Wn.T

    # ---- pre-allocate the full streamed factor ----
    total = data.shape[stream_axis]
    U_full = list(U)
    U_full[last] = np.vstack([
        U[last],
        np.zeros((total - Lt, U[last].shape[1])),
    ])

    start = Lt
    block_idx = 0
    while start < total:
        end = min(start + L, total)
        X_n = data[_slice(start, end)]
        if verbose:
            print(f"    block {block_idx+1}: axis{last} idx {start+1}-{end} "
                  f"(size={end-start})")
        U_full, G, P, Q = recursive_update(
            X_n, U_full, G, P, Q, start=start,
            stream_axis=last,
            nonneg=True, k_inner=10, gamma=1e-3, lam=1)
        start += (L - overlap)
        block_idx += 1
    return U_full, G, {
        "n_blocks": block_idx, "Lt": Lt, "L": L,
        "overlap": overlap, "seed": seed, "stream_axis": last,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_cube_report(out_dir, cube_path, ground_truth, endmember_results,
                      stream_info, ranks, preprocess_method, max_angle):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cube_id = Path(cube_path).parent.name + "_" + Path(cube_path).stem
    report_path = out_dir / f"{cube_id}_sam.txt"
    lines = [
        "Online Tucker + SAM report (NaN-aware SAM)",
        "=" * 60,
        f"Cube          : {cube_path}",
        f"Ground truth  : {ground_truth}",
        f"Tucker ranks  : {ranks}",
        f"Preprocess    : {preprocess_method}",
        f"Angle cutoff  : {max_angle:.1f} deg",
        f"Streaming     : {stream_info}",
        "",
        "Endmember classification",
        "-" * 60,
    ]
    for r in endmember_results:
        lines.append(f"endmember {r['endmember_idx']:>3d}  ->  {r['label']}")
        if r.get("all_angles_nan"):
            lines.append("    (all library comparisons were NaN)")
            continue
        for k, m in enumerate(r["top_k"]):
            lines.append(f"    [{k}] {m['name']:<28s} "
                         f"{m['angle_deg']:6.2f} deg")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def name_in_library(query, names):
    if not query:
        return False
    q = query.strip().lower()
    return any(q in n.lower() for n in names)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--ranks", nargs=3, type=int, default=[1, 1, 1],
                    metavar=("R0", "R1", "R2"))
    ap.add_argument("--max-angle-deg", type=float, default=None,
                    help="override the preprocess-specific default "
                         "(raw=10, snv=30, cr=15, deriv=40, deriv2=50)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-lib-coverage", type=float, default=0.85)
    ap.add_argument("--out-dir", default="conveyor_out")
    ap.add_argument("--Lt", type=int, default=100)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--theta", type=int, default=10)
    ap.add_argument("--n-iter-max", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)

    ap.add_argument("--stream-axis", type=int, default=0, choices=[0, 1, 2],
                    help="axis along which the online Tucker streams. "
                         "0 = first spatial (height), 1 = second spatial "
                         "(width), 2 = spectral bands. Default: 0")

    ap.add_argument("--preprocess", default="raw",
                    choices=["raw", "snv", "cr", "deriv", "deriv2"])
    ap.add_argument("--deriv-window", type=int, default=11)
    ap.add_argument("--deriv-poly", type=int, default=2)

    ap.add_argument("--wl-window", nargs=2, type=float, default=None,
                    metavar=("WL_MIN", "WL_MAX"),
                    help="restrict SAM to this wavelength range (microns). "
                         "Bands outside are treated as invalid.")

    args = ap.parse_args()

    if args.max_angle_deg is None:
        args.max_angle_deg = PREPROCESS_ANGLE_THRESHOLD[args.preprocess]

    print("=" * 68)
    print("Online Tucker + NaN-aware SAM conveyor")
    print(f"Pre-processing   : {args.preprocess}")
    if args.preprocess in ("deriv", "deriv2"):
        print(f"  SG window={args.deriv_window}  poly={args.deriv_poly}")
    print(f"Angle cutoff     : {args.max_angle_deg:.1f} deg")
    print(f"Stream axis      : {args.stream_axis} "
          f"({'bands' if args.stream_axis == 2 else 'spatial'})")
    if args.wl_window is not None:
        print(f"Wavelength window: "
              f"{args.wl_window[0]:.3f}-{args.wl_window[1]:.3f} um")
    print("=" * 68)

    # ---- library ----
    print(f"\nLoading library : {args.library}")
    L, M, names, fnames, wl, n_dropped = load_library(
        args.library, min_coverage=args.min_lib_coverage)
    print(f"  {L.shape[0]} spectra ({n_dropped} dropped for coverage), "
          f"{len(set(names.tolist()))} unique minerals, {L.shape[1]} bands")

    if args.wl_window is not None:
        wmin, wmax = args.wl_window
        in_win = (wl >= wmin) & (wl <= wmax)
        print(f"  Restricting to {wmin:.3f}-{wmax:.3f} um "
              f"({in_win.sum()}/{len(wl)} bands)")
        L = np.where(in_win[None, :], L, np.nan)

    if args.preprocess != "raw":
        print(f"  Applying '{args.preprocess}' to library ...")
        L = preprocess_library(L, wl, args.preprocess,
                                args.deriv_window, args.deriv_poly)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "conveyor_results.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "cube_path", "cube_id", "ground_truth", "gt_in_library",
        "n_endmembers", "preprocess", "stream_axis",
        "top1_label", "top1_angle_deg", "top1_match",
        "top2_label", "top2_angle_deg", "top2_match",
        "top3_label", "top3_angle_deg", "top3_match",
        "any_match_top1", "any_match_top3",
    ])

    n_done = 0
    n_gt_defined = 0
    n_gt_in_lib = 0
    n_unmatchable = 0
    n_top1_all = 0
    n_top3_all = 0
    n_top1_inlib = 0
    n_top3_inlib = 0

    print(f"\nStreaming cubes from {args.data} ...")
    for cube_path, data in cube_generator(args.data):
        if args.limit is not None and n_done >= args.limit:
            break

        W, H, B = data.shape
        print(f"\n[{n_done+1}] {cube_path}")
        print(f"    shape W={W} H={H} B={B}")

        if B != L.shape[1]:
            print(f"    SKIP: cube has {B} bands, library has {L.shape[1]}")
            continue

        # ---- ground truth ----
        desc_path = find_description_for(cube_path)
        gt = parse_description(desc_path) if desc_path else None
        gt_in_lib = name_in_library(gt, names) if gt else False
        print(f"    ground truth: {gt}   (in library: {gt_in_lib})")

        # ---- decompose ----
        try:
            U_full, G, stream_info = decompose_cube(
                data, args.ranks,
                Lt=args.Lt, L=args.L, theta=args.theta,
                n_iter_max=args.n_iter_max, seed=args.seed,
                stream_axis=args.stream_axis)
        except Exception as e:
            print(f"    DECOMPOSE FAILED: {e}")
            continue

        # The band factor is always the last Tucker mode, regardless of
        # which axis we streamed along.
        U2 = U_full[-1]
        print(f"    endmembers: {U2.shape[1]}")

        # Apply window + preprocess to endmembers
        if args.wl_window is not None:
            U2 = np.where(in_win[:, None], U2, np.nan)
        if args.preprocess != "raw":
            U2_proc = np.stack([
                preprocess(U2[:, r], wl, args.preprocess,
                           args.deriv_window, args.deriv_poly)
                for r in range(U2.shape[1])
            ], axis=1)
        else:
            U2_proc = U2

        # ---- classify ----
        endmember_results = classify_endmembers(
            U2_proc, L, names,
            max_angle_deg=args.max_angle_deg, top_k=args.top_k)
        for r in endmember_results:
            if r.get("all_angles_nan") or not r["top_k"]:
                print(f"      em {r['endmember_idx']}: (no valid comparison)")
                continue
            print(f"      em {r['endmember_idx']}: {r['label']:<24s} "
                  f"({r['top_k'][0]['angle_deg']:.2f} deg)")

        report_path = write_cube_report(
            out_dir, cube_path, gt, endmember_results, stream_info,
            args.ranks, args.preprocess, args.max_angle_deg)
        print(f"    -> {report_path.name}")

        # ---- CSV row ----
        r0 = endmember_results[0] if endmember_results else None

        def _row_vals(r, k):
            if r is None or not r.get("top_k") or k >= len(r["top_k"]):
                return "", "", ""
            m = r["top_k"][k]
            match = ""
            if gt:
                match = "1" if gt.lower() == m["name"].lower() else "0"
            return m["name"], f"{m['angle_deg']:.2f}", match

        t1n, t1a, t1m = _row_vals(r0, 0)
        t2n, t2a, t2m = _row_vals(r0, 1)
        t3n, t3a, t3m = _row_vals(r0, 2)

        any_top1 = "0"
        any_top3 = "0"
        if gt:
            gt_norm = gt.strip().lower()
            for r in endmember_results:
                for k, m in enumerate(r.get("top_k", [])):
                    if m["name"].strip().lower() == gt_norm:
                        if k == 0:
                            any_top1 = "1"
                        if k <= 2:
                            any_top3 = "1"
                        break

        cube_id = Path(cube_path).parent.name + "_" + Path(cube_path).stem
        csv_w.writerow([
            cube_path, cube_id, gt or "", "1" if gt_in_lib else "0",
            len(endmember_results), args.preprocess, args.stream_axis,
            t1n, t1a, t1m, t2n, t2a, t2m, t3n, t3a, t3m,
            any_top1, any_top3,
        ])
        csv_f.flush()

        # ---- counters ----
        if gt:
            n_gt_defined += 1
            if gt_in_lib:
                n_gt_in_lib += 1
                if any_top1 == "1":
                    n_top1_inlib += 1
                if any_top3 == "1":
                    n_top3_inlib += 1
            else:
                n_unmatchable += 1
            if any_top1 == "1":
                n_top1_all += 1
            if any_top3 == "1":
                n_top3_all += 1

        n_done += 1

    csv_f.close()

    print("\n" + "=" * 68)
    print(f"Done. {n_done} cubes processed.  preprocess={args.preprocess}  "
          f"cutoff={args.max_angle_deg:.1f} deg  "
          f"stream_axis={args.stream_axis}")
    if n_gt_defined:
        print(f"  GT cubes with a label             : {n_gt_defined}")
        print(f"  ... of which mineral in library   : {n_gt_in_lib}")
        print(f"  ... of which NOT in library       : {n_unmatchable}")
        print(f"  top-1 match (all GT cubes)        : "
              f"{n_top1_all}/{n_gt_defined} "
              f"({n_top1_all/n_gt_defined:.3f})")
        print(f"  top-3 match (all GT cubes)        : "
              f"{n_top3_all}/{n_gt_defined} "
              f"({n_top3_all/n_gt_defined:.3f})")
        if n_gt_in_lib:
            print(f"  top-1 match (GT in library only)  : "
                  f"{n_top1_inlib}/{n_gt_in_lib} "
                  f"({n_top1_inlib/n_gt_in_lib:.3f})")
            print(f"  top-3 match (GT in library only)  : "
                  f"{n_top3_inlib}/{n_gt_in_lib} "
                  f"({n_top3_inlib/n_gt_in_lib:.3f})")
    print(f"  results CSV : {csv_path}")
    print(f"  reports dir : {out_dir}/")


if __name__ == "__main__":
    main()