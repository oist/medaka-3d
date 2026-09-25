"""Stage ten predetermined frames from existing unfiltered Detectron2 mask caches.

Read-only inputs; no image inference, SLEAP, ID, or annotation input. A separate
offset-indexed, non-pickle NPZ is written per view. Missing HDF5 frame groups
cannot distinguish no detections from unprocessed frames and are not treated as
known-empty views. Selection uses only the intersection of stored frame indexes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from medaka3d.provenance import provenance

RELATIVE_MASKS = {
    view: f"{view}_f132000_137000_segpilot/detectron2_masks.h5"
    for view in ("top_cam", "side1_cam", "side2_cam")
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_cache(path):
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != "sam2_id_masks_v1":
            raise ValueError(f"unexpected HDF5 schema: {path}")
        shape = (int(handle.attrs["image_height"]), int(handle.attrs["image_width"]))
        if min(shape) < 1:
            raise ValueError(f"invalid image dimensions: {path}")
        frames = np.array(sorted(int(name[6:]) for name in handle if name.startswith("frame_")))
        if not len(frames) or np.any(frames < 0) or len(np.unique(frames)) != len(frames):
            raise ValueError(f"missing or invalid frame indexes: {path}")
        return frames, shape


def select_frames(available, count=10):
    shared = np.asarray(sorted(set.intersection(*(set(values) for values in available))), dtype=int)
    if count < 1 or len(shared) < count:
        raise ValueError("not enough shared stored frames for the requested fixed selection")
    targets = np.linspace(shared[0], shared[-1], count)
    # argmin on sorted frames chooses the earlier frame for an equal-distance tie.
    chosen = np.array([shared[np.argmin(np.abs(shared - target))] for target in targets])
    if len(np.unique(chosen)) != count:
        raise ValueError("equally spaced target snapping produced duplicate frames")
    return chosen, targets, shared


def extract_view(path, frames, shape):
    masks, scores, boxes, ids, source_track_ids, offsets = [], [], [], [], [], [0]
    with h5py.File(path, "r") as handle:
        for frame in frames:
            name = f"frame_{int(frame):06d}"
            if name not in handle:
                raise ValueError(f"selected frame group is missing: {path}/{name}")
            group = handle[name]
            for key in sorted(group):
                if not key.startswith("instance_"):
                    raise ValueError(f"unexpected frame child: {path}/{name}/{key}")
                instance = group[key]
                mask = instance["mask"][...]
                box = np.asarray(instance["bbox"][...], dtype=np.int32)
                score = float(instance.attrs["score"])
                if mask.shape != shape or not np.all((mask == 0) | (mask == 1)):
                    raise ValueError(f"invalid binary mask: {path}/{name}/{key}")
                if box.shape != (4,) or not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"invalid bbox or confidence: {path}/{name}/{key}")
                if int(instance.attrs["frame"]) != frame:
                    raise ValueError(
                        f"instance frame disagrees with its group: {path}/{name}/{key}"
                    )
                masks.append(mask.astype(bool))
                scores.append(score)
                boxes.append(box)
                ids.append(int(instance.attrs["instance_id"]))
                source_track_ids.append(int(instance.attrs.get("sam2_track_id", -1)))
            offsets.append(len(masks))
    return {
        "frame_idx": np.asarray(frames, dtype=np.int64),
        "frame_offsets": np.asarray(offsets, dtype=np.int64),
        "masks": np.stack(masks) if masks else np.empty((0, *shape), dtype=bool),
        "scores": np.asarray(scores, dtype=np.float64),
        "boxes": np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
        "instance_id": np.asarray(ids, dtype=np.int64),
        "source_track_id": np.asarray(source_track_ids, dtype=np.int64),
        "view_available": np.ones(len(frames), dtype=bool),
    }


def stage_inputs(source_root, out, count=10):
    source_root, out = Path(source_root).resolve(strict=True), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite mask input staging: {out}")
    sources, inspected = {}, {}
    for view, relative in RELATIVE_MASKS.items():
        path = (source_root / relative).resolve(strict=True)
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"source escaped its root or is not a file: {path}")
        sources[view] = path
        inspected[view] = inspect_cache(path)
    frames, targets, shared = select_frames([value[0] for value in inspected.values()], count)
    # Validate all requested mask data before creating the destination.
    arrays = {
        view: extract_view(path, frames, inspected[view][1]) for view, path in sources.items()
    }
    source_hashes = {view: sha256(path) for view, path in sources.items()}
    out.mkdir(parents=True, exist_ok=False)
    with provenance("stage_independent_detectron2_masks", out, name="staging") as record:
        record.param("source_root", str(source_root))
        record.param("frames", frames.tolist())
        record.param("selection_targets", targets.tolist())
        record.param(
            "selection_rule",
            "Ten equal-spaced targets over min/max shared stored "
            "frame indexes, snapped to nearest shared frame; earlier frame breaks ties",
        )
        record.input("stager", Path(__file__), sha256=sha256(__file__))
        shutil.copyfile(__file__, out / "stage_mask_inputs.py")
        views = {}
        for view, data in arrays.items():
            source = sources[view]
            record.input(view + "_h5", source, sha256=source_hashes[view])
            output = out / f"{view}.npz"
            np.savez_compressed(output, **data)
            digest = sha256(output)
            record.output(view, output, sha256=digest)
            source_frames, shape = inspected[view]
            views[view] = {
                "source_h5": str(source),
                "source_relative": RELATIVE_MASKS[view],
                "source_sha256": source_hashes[view],
                "source_bytes": source.stat().st_size,
                "output_npz": output.name,
                "output_sha256": digest,
                "image_shape_hw": list(shape),
                "source_stored_frame_count": len(source_frames),
                "source_stored_extent_inclusive": [int(source_frames[0]), int(source_frames[-1])],
                "instances_per_selected_frame": np.diff(data["frame_offsets"]).tolist(),
            }
        manifest = {
            "experiment": "20251225_K_1_2_FCMC",
            "schema": "offset_indexed_instance_masks_v1",
            "frames": frames.tolist(),
            "selection_targets": targets.tolist(),
            "source_common_frame_count": len(shared),
            "views": views,
            "npz_schema": {
                "frame_idx": "T source frame indexes",
                "frame_offsets": "T+1 offsets into K instances",
                "masks": "K,H,W boolean, full source image coordinates",
                "scores": "K float64",
                "boxes": "K,4 int32 [x0,y0,x1,y1], original stored bbox",
                "instance_id": "K original within-frame proposal indexes",
                "source_track_id": "K original sam2_track_id field; per-frame index, not ID",
                "view_available": "T bool; stored groups are available, even if explicitly empty",
            },
            "known_generation_properties": [
                "Filename is detectron2_masks.h5, not a sleap_filtered or linked derivative.",
                "HDF5 schema sam2_id_masks_v1 retains original mask/bbox/score metadata.",
                "Preserved D2 runner uses framewise proposals; indexes are not identities.",
            ],
            "unknown_generation_properties": [
                "HDF5 lacks model/config hashes, software versions, threshold and command.",
                "Old runner default threshold is 0.5; this cache's setting is unverified.",
                "Missing groups may mean zero detections, unprocessed frames or decode failure.",
            ],
            "limitations": [
                "Shared stored frame selection can favor frames with detections.",
                "Staging uses no count constraint, extra filter, SLEAP or identity annotation.",
                "Cache-selected development frames provide feasibility, not ID validation.",
            ],
        }
        path = out / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        record.output("manifest", path, sha256=sha256(path))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the three view_f132000_137000_segpilot folders",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = stage_inputs(args.source_root, args.out)
    print(json.dumps({"frames": manifest["frames"], "views": manifest["views"]}, indent=2))


if __name__ == "__main__":
    main()
