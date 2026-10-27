"""Build the preprocessed CT cache ("v2 corpus") used for fast training.

For every CT-Encoders-Dataset volume:
  1. recover the voxel spacing lost in the CT-Encoders-Dataset conversion
       - DICOM-derived sources: SeriesInstanceUID -> (xy, z) from ct_encoders_recovered_geometry.json
       - NIfTI-native sources (AMOS, AbdomenCT-1K, kits23, RAOS, BTCV): header of the original file
  2. put the slice axis first -> (z, y, x)  (~9% of the volumes are stored permuted)
  3. clip to [-1024, 3071] HU
  4. downsample (never upsample) to the target spacing, Gaussian anti-aliasing + trilinear
  5. crop to the body bounding box (largest component of HU > -500, plus a margin)
  6. save int16 .npy  +  one manifest row (manifest.jsonl)

NOT done yet (needed for the production build, not for speed tests): canonical RAS orientation (flips),
from ImageOrientationPatient / the original affines.

Example:
  python tools/build_ct_cache.py --files david-ct/metadata/sample_1019.txt --src /home/shared_data/CT-Encoders-Dataset \
      --out /home/local_data/ct-v2/v2.0_1p25mm/ct-encoders --spacing 1.25 --workers 24
"""
import argparse, json, math, os, sys, time
import multiprocessing as mp

import numpy as np

NATIVE_SOURCES = ("AMOS", "AbdomenCT-1K", "kits23", "RAOS", "BTCV")
HU_MIN, HU_MAX = -1024, 3071


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", required=True, help="text file with CT-Encoders-Dataset file names, one per line")
    p.add_argument("--src", required=True, help="directory holding those .nii.gz files")
    p.add_argument("--out", required=True)
    p.add_argument("--spacing", type=float, default=1.25, help="target spacing in mm; 0 = keep native")
    p.add_argument("--body-crop", type=int, default=1)
    p.add_argument("--margin-mm", type=float, default=10.0)
    p.add_argument("--geometry", default=os.path.join(os.path.dirname(__file__), "..", "david-ct", "metadata",
                                                       "ct_encoders_recovered_geometry_filled.json"))
    p.add_argument("--series-dims", default=os.path.join(os.path.dirname(__file__), "..", "david-ct", "metadata",
                                                          "ct_encoders_series_dims.json"),
                   help="{uid: [n_slices, rows, cols]} from "
                   "tools/fill_slice_spacing.py; resolves the slice axis of permuted DICOM-derived volumes")
    p.add_argument("--native-root", default="/home/shared_data")
    p.add_argument("--native-index", default=None, help="text file of original NIfTI paths relative to "
                   "--native-root (built with `find` if missing)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--as-stored", action="store_true", help="only convert to int16 .npy (same array as the "
                   ".nii.gz, no reorder/clip/resample/crop); used to measure decode cost in isolation")
    return p.parse_args()


def split_name(name):
    base = name[:-7] if name.endswith(".nii.gz") else os.path.splitext(name)[0]
    src, _, rest = base.partition("_")
    return src, rest


def native_lookup(index_lines):
    by_name = {}
    for rel in index_lines:
        if "/kits23/" in "/" + rel:
            if rel.endswith("/imaging.nii.gz"):
                by_name["kits23:" + rel.split("/")[-2]] = rel
        else:
            by_name.setdefault(os.path.basename(rel), rel)
    return by_name


def find_spacing(name, arr_shape, geometry, native_map, native_root, series_dims=None):
    """Returns (spacing in the order of arr's axes, slice axis of arr, method string)."""
    src, rest = split_name(name)
    if src in NATIVE_SOURCES:
        import nibabel as nib
        key = f"kits23:{rest}" if src == "kits23" else rest + ".nii.gz"
        rel = native_map.get(key)
        if rel is None:
            raise KeyError(f"original not found for {name}")
        hdr = nib.load(os.path.join(native_root, rel)).header
        zooms, shape = [float(z) for z in hdr.get_zooms()[:3]], tuple(int(s) for s in hdr.get_data_shape()[:3])
        # sitk arrays are (z, y, x) = reversed nibabel order, unless the conversion permuted the axes
        perm = _match_axes(arr_shape, shape)
        sp = [zooms[perm[i]] for i in range(3)]
        return sp, _odd_axis(sp, arr_shape), "native-header"
    uid = rest.removesuffix("_0000").strip()
    if uid not in geometry:
        raise KeyError(f"no recovered geometry for {name}")
    xy, z = geometry[uid]
    s = arr_shape
    dims = (series_dims or {}).get(uid)
    if dims and dims[1] and dims[2] and sorted(s[1:]) != sorted(dims[1:]) and dims[0] in s[1:]:
        # stored permuted: axis 0 is a (cropped) in-plane axis, the slice axis has n_slices voxels
        slice_axis = 1 if s[1] == dims[0] else 2
        sp = [xy, xy, xy]
        sp[slice_axis] = z
        return sp, slice_axis, "geometry-json+series-dims"
    # in-plane = the two axes of equal length (512x512 etc.); the remaining one is the slice axis
    if s[1] == s[2] or (s[0] != s[1] and s[0] != s[2]):
        slice_axis = 0
    elif s[0] == s[2]:
        slice_axis = 1
    else:
        slice_axis = 2
    sp = [xy, xy, xy]
    sp[slice_axis] = z
    return sp, slice_axis, "geometry-json"


def _odd_axis(values, shape):
    """The axis whose spacing (else size) differs from the other two; 0 if none does."""
    for vals in (values, shape):
        for a in range(3):
            b, c = [i for i in range(3) if i != a]
            if abs(vals[b] - vals[c]) < 1e-3 and abs(vals[a] - vals[b]) >= 1e-3:
                return a
    return 0


def _match_axes(arr_shape, nib_shape):
    """perm such that arr axis i corresponds to nibabel axis perm[i] (prefer the reversed order).
    The CT-Encoders-Dataset conversion kept only a contiguous range of array axis 0 (and flipped in-plane
    axes) without resampling, so axis 0 may be shorter than in the original."""
    perms = [(2, 1, 0), (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)]
    for exact in (True, False):
        for perm in perms:
            if tuple(arr_shape[1:]) == (nib_shape[perm[1]], nib_shape[perm[2]]) and \
                    (arr_shape[0] == nib_shape[perm[0]] if exact else arr_shape[0] <= nib_shape[perm[0]]):
                return perm
    raise ValueError(f"shape mismatch {arr_shape} vs original {nib_shape}")


def resample(vol, spacing, target):
    """Downsample-only anti-aliased trilinear resampling. vol (z,y,x) float32 numpy."""
    import torch
    import torch.nn.functional as F
    if not target:
        return vol, list(spacing)
    factors = [max(1.0, target / s) for s in spacing]
    if all(f < 1.01 for f in factors):
        return vol, list(spacing)
    x = torch.from_numpy(vol)[None, None]
    for d, f in enumerate(factors):
        if f < 1.01:
            continue
        sigma = (f - 1) / 2  # skimage-style anti-aliasing sigma, in input voxels
        r = max(1, int(math.ceil(3 * sigma)))
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (t / sigma) ** 2)
        k = (k / k.sum()).view([1, 1] + [k.numel() if i == d else 1 for i in range(3)])
        pad = [0] * 6
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k)
    new_shape = [max(1, round(n * s / (s * f))) for n, s, f in zip(vol.shape, spacing, factors)]
    x = F.interpolate(x, size=new_shape, mode="trilinear", align_corners=False)
    new_spacing = [n * s / m for n, s, m in zip(vol.shape, spacing, new_shape)]
    return x[0, 0].numpy(), new_spacing


def body_bbox(vol, spacing, margin_mm):
    from scipy import ndimage
    step = 2
    m = vol[::step, ::step, ::step] > -500
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum_labels(m, lab, index=np.arange(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    idx = np.nonzero(keep)
    bbox = []
    for d in range(3):
        lo, hi = idx[d].min() * step, min(vol.shape[d], (idx[d].max() + 1) * step)
        mg = int(round(margin_mm / spacing[d]))
        bbox.append((int(max(0, lo - mg)), int(min(vol.shape[d], hi + mg))))
    return bbox


def process(job):
    name, cfg = job
    import torch
    import SimpleITK as sitk
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    out_path = os.path.join(cfg["out"], name[:-7] + ".npy")
    if os.path.exists(out_path):
        return None
    t0 = time.time()
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(cfg["src"], name)))
        orig_shape, orig_dtype = arr.shape, str(arr.dtype)
        if cfg["as_stored"]:
            out = np.clip(np.rint(arr), -32768, 32767).astype(np.int16)
            np.save(out_path + ".tmp.npy", out)
            os.replace(out_path + ".tmp.npy", out_path)
            return dict(name=name, source=split_name(name)[0], path=os.path.basename(out_path),
                        orig_shape=list(orig_shape), orig_dtype=orig_dtype, shape_zyx=list(out.shape),
                        mb=round(out.nbytes / 1e6, 2), seconds=round(time.time() - t0, 2))
        spacing, slice_axis, method = find_spacing(name, arr.shape, cfg["geometry"], cfg["native_map"],
                                                   cfg["native_root"], cfg["series_dims"])
        order = [slice_axis] + [i for i in range(3) if i != slice_axis]
        vol = np.clip(np.transpose(arr, order), HU_MIN, HU_MAX).astype(np.float32)
        spacing = [spacing[i] for i in order]
        vol, spacing = resample(vol, spacing, cfg["spacing"])
        bbox = body_bbox(vol, spacing, cfg["margin_mm"]) if cfg["body_crop"] else None
        if bbox:
            vol = vol[tuple(slice(lo, hi) for lo, hi in bbox)]
        out = np.rint(vol).astype(np.int16)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, out)
        os.replace(tmp, out_path)
        return dict(name=name, source=split_name(name)[0], path=os.path.basename(out_path), orig_shape=list(orig_shape),
                    orig_dtype=orig_dtype, spacing_method=method, slice_axis_in_orig=int(slice_axis),
                    spacing_zyx=[round(s, 4) for s in spacing], shape_zyx=list(out.shape), body_bbox=bbox,
                    mb=round(out.nbytes / 1e6, 2), seconds=round(time.time() - t0, 2))
    except Exception as e:  # keep going; failures are listed in the manifest
        return dict(name=name, error=repr(e))


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = [os.path.basename(l.strip()) for l in open(args.files) if l.strip()]
    geometry = json.load(open(args.geometry))
    index = args.native_index
    if index is None or not os.path.exists(index):
        raise SystemExit("--native-index is required (find <root>/{AMOS,AbdomenCT-1K,kits23,RAOS,BTCV} -name '*.nii.gz')")
    native_map = native_lookup([l.strip() for l in open(index) if l.strip()])
    cfg = dict(out=args.out, src=args.src, spacing=args.spacing, body_crop=args.body_crop, margin_mm=args.margin_mm,
               geometry=geometry, native_map=native_map, native_root=args.native_root, as_stored=args.as_stored,
               series_dims=json.load(open(args.series_dims)) if args.series_dims else None)
    t0, done, failed = time.time(), 0, 0
    with mp.get_context("fork").Pool(args.workers) as pool, open(os.path.join(args.out, "manifest.jsonl"), "a") as mf:
        for row in pool.imap_unordered(process, [(n, cfg) for n in names], chunksize=1):
            if row is None:
                continue
            mf.write(json.dumps(row) + "\n")
            done += 1
            failed += "error" in row
            if done % 50 == 0:
                print(f"{done}/{len(names)}  {time.time() - t0:.0f}s  failed={failed}", flush=True)
    print(f"done {done} volumes in {time.time() - t0:.0f}s, failed={failed}")


if __name__ == "__main__":
    main()
