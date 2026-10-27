"""Fill the slice spacing missing (None) in ct_encoders_recovered_geometry.json from ImagePositionPatient.

~2% of the DICOM series (e.g. CT-Lymph-Nodes, Pancreas-CT) have no SliceThickness in the header index.
For those, the slice spacing is the median distance between consecutive ImagePositionPatient values of
the series' files, read from the lab's header index (CT-Encoder/headers, header ids 122 = SeriesInstanceUID,
127 = ImagePositionPatient). Writes a new JSON; the input is not modified.

  python tools/fill_slice_spacing.py david-ct/metadata/ct_encoders_recovered_geometry.json \
      david-ct/metadata/ct_encoders_recovered_geometry_filled.json [david-ct/metadata/ct_encoders_series_dims.json]
"""
import collections, json, sys

import numpy as np
import pyarrow.parquet as pq

HEADERS = "/home/shared_data/CT-Encoder/headers/medical-dataset_metadata.parquet"
UID, IPP, ROWS, COLS = 122, 127, 135, 136


def series_dims(dst):
    """{uid: [n_files (= slices), rows, columns]}: tells which array axis is the slice axis when the
    CT-Encoders-Dataset conversion stored a volume permuted (and cropped one of its in-plane axes)."""
    t = pq.read_table(HEADERS, filters=[("header_id", "in", [UID, ROWS, COLS])], columns=["file_id", "header_id", "value"])
    d = t.to_pydict()
    per = collections.defaultdict(dict)
    for fid, hid, val in zip(d["file_id"], d["header_id"], d["value"]):
        per[fid][hid] = val
    dims = {}
    for fid, h in per.items():
        if UID not in h:
            continue
        uid = str(h[UID]).strip()
        n, r, c = dims.get(uid, [0, None, None])
        dims[uid] = [n + 1, r or _int(h.get(ROWS)), c or _int(h.get(COLS))]
    json.dump(dims, open(dst, "w"))
    print(f"series dims for {len(dims)} series -> {dst}")


def _int(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def main(src, dst):
    geom = json.load(open(src))
    missing = {u for u, v in geom.items() if v is None or v[1] is None}
    print(f"{len(missing)} series without slice spacing")
    t = pq.read_table(HEADERS, filters=[("header_id", "in", [UID, IPP])], columns=["file_id", "header_id", "value"])
    d = t.to_pydict()
    uid_of, ipp_of = {}, {}
    for fid, hid, val in zip(d["file_id"], d["header_id"], d["value"]):
        if hid == UID:
            uid_of[fid] = str(val).strip()
        else:
            ipp_of[fid] = val
    positions = collections.defaultdict(list)
    for fid, uid in uid_of.items():
        if uid in missing and fid in ipp_of:
            try:
                positions[uid].append([float(x) for x in str(ipp_of[fid]).replace("[", "").replace("]", "")
                                       .replace(",", "\\").split("\\") if x.strip()])
            except ValueError:
                pass
    filled = 0
    for uid in missing:
        pts = np.array([p for p in positions.get(uid, []) if len(p) == 3])
        if len(pts) < 2:
            continue
        # project on the axis with the largest spread (the slice normal for axial series)
        ax = int(np.argmax(pts.max(0) - pts.min(0)))
        z = np.sort(np.unique(np.round(pts[:, ax], 3)))
        if len(z) < 2:
            continue
        geom[uid] = [geom[uid][0] if geom[uid] else None, float(np.median(np.diff(z)))]
        filled += 1
    print(f"filled {filled}/{len(missing)}")
    json.dump(geom, open(dst, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:3])
    if len(sys.argv) > 3:
        series_dims(sys.argv[3])
