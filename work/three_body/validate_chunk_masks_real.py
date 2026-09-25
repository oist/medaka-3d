"""CPU integration check: split actual packed masks, join, and compare every array byte."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from chunk_masks import concatenate_packed_masks  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slice_frames(data, start, stop):
    lo, hi = map(int, data["frame_offsets"][[start, stop]])
    byte_lo, byte_hi = map(int, data["mask_byte_offsets"][[lo, hi]])
    frame_fields = ("frame_idx", "view_available", "frame_mask_sha256")
    out = {}
    for key, value in data.items():
        if key == "image_shape_hw":
            out[key] = value.copy()
        elif key == "frame_offsets":
            out[key] = value[start:stop + 1] - lo
        elif key == "mask_byte_offsets":
            out[key] = value[lo:hi + 1] - byte_lo
        elif key == "mask_bits":
            out[key] = value[byte_lo:byte_hi]
        elif key in frame_fields:
            out[key] = value[start:stop]
        else:
            out[key] = value[lo:hi]
    return out


def main():
    root = REPO / "work/three_body/inputs/transient_333519-333719"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for view, entry in manifest["masks"].items():
        path = root / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("actual mask asset hash changed")
        with np.load(path, allow_pickle=False) as arrays:
            original = {key: arrays[key] for key in arrays.files}
        pieces = [slice_frames(original, start, stop) for start, stop in
                  ((0, 67), (67, 134), (134, 200))]
        start = time.perf_counter()
        result = concatenate_packed_masks(pieces)
        elapsed = time.perf_counter() - start
        assert set(result) == set(original)
        for key in result:
            assert result[key].shape == original[key].shape, key
            assert result[key].dtype == original[key].dtype, key
            assert result[key].tobytes() == original[key].tobytes(), key
        rows.append(dict(view=view, frames=len(result["frame_idx"]),
                         instances=len(result["scores"]), chunks=3,
                         validate_and_join_seconds=elapsed, all_array_bytes_exact=True,
                         source_path=str(path), source_sha256=entry["sha256"]))
    report = dict(status="ok", cuda_operations=False, rows=rows,
                  source_hashes={str(p.relative_to(REPO)): digest(p) for p in (
                      Path(__file__), LAB / "chunk_masks.py", LAB / "prepare_sequence.py")})
    output = REPO / "work/three_body/chunk_masks_real_validation.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
