"""Stage native-rate raw frames and independent framewise Detectron2 masks.

Use the existing dedicated Detectron2 environment with ``-B``. Source videos
are read only. FFV1 AVI clips are verified against every original decoded BGR
frame; each instance mask is stored as packed bits inside its exact mask bbox.
No SLEAP, 3D trajectory, animal ID, or detection-count filter enters this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

EXPERIMENT = "20251225_K_1_2_FCMC"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEO_NAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}
WINDOWS = {
    "transient": (333519, 333719),
    "interval": (76620, 77286),
    "continuous_one_minute": (76620, 86580),
}
SHAPE = (1200, 1200)
FPS = 166.0
BACKGROUND_CANDIDATES = (20000, 40000, 60000, 80000, 100000, 160000, 220000, 280000, 340000)
BACKGROUND_GUARD = 332
MODEL_DIRS = {
    "top": "medaka_202512/detectron2_models/top_detectron2_model",
    "side": "medaka_202512/detectron2_models/side_detectron2_model_firstprompt",
}
ORIGIN = {
    "path": "labs/20260908_mask_volume/infer_masks.py",
    "sha256": "9f2dbba1873a98e6a15a8ad24d36dcad564d36f894f98c16f4d4155c59908d0f",
    "reused": "asset verification, model configuration, provenance conventions; copied, no import",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def select_background_frames(start, stop):
    """Exclude the half-open sequence plus a two-second guard on each side."""
    chosen = [
        f
        for f in BACKGROUND_CANDIDATES
        if not start - BACKGROUND_GUARD <= f < stop + BACKGROUND_GUARD
    ]
    if len(chosen) < 3:
        raise ValueError("fewer than three fixed background samples remain after exclusion")
    return chosen


def verify_assets(medaka_root):
    """Verify the four retained model/config files before loading any checkpoint."""
    medaka_root = Path(medaka_root).resolve(strict=True)
    manifest_path = medaka_root / "assets/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = {}
    for kind, relative in MODEL_DIRS.items():
        for role, filename in (
            ("weights", "model_final.pth"),
            ("config", "detectron2_config.yaml"),
        ):
            key = f"detectron2.{kind}" + (".config" if role == "config" else "")
            entry = manifest["assets"][key]
            path = (medaka_root / relative / filename).resolve(strict=True)
            if not path.is_relative_to(medaka_root) or path.stat().st_size != entry["size"]:
                raise ValueError(f"asset path or size mismatch: {path}")
            digest = sha256(path)
            if digest != entry["sha256"]:
                raise ValueError(f"asset checksum mismatch: {path}")
            verified[f"{kind}_{role}"] = {
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "asset_key": key,
                "asset_manifest_entry": entry,
            }
    return verified, manifest_path


class PackedMasks:
    """Incremental encoder; only the current frame holds full image masks."""

    def __init__(self, shape=SHAPE):
        self.shape = tuple(shape)
        self.frames, self.offsets, self.byte_offsets = [], [0], [0]
        self.bits, self.boxes, self.scores, self.pred_boxes, self.classes, self.ids = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.mask_hashes = []

    def append(self, frame, masks, scores, pred_boxes, classes):
        masks = np.asarray(masks)
        count = len(masks)
        scores = np.asarray(scores, dtype=np.float32)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        classes = np.asarray(classes, dtype=np.int64)
        if self.frames and frame != self.frames[-1] + 1:
            raise ValueError("staging requires consecutive source frames")
        if masks.dtype != bool or masks.shape != (count, *self.shape):
            raise ValueError("masks must be bool in full source image coordinates")
        if scores.shape != (count,) or pred_boxes.shape != (count, 4) or classes.shape != (count,):
            raise ValueError("prediction field counts differ")
        if not np.isfinite(scores).all() or np.any((scores < 0.5) | (scores > 1)):
            raise ValueError("score is invalid for the fixed threshold 0.5")
        if not np.isfinite(pred_boxes).all():
            raise ValueError("predicted boxes contain nonfinite coordinates")
        for index, mask in enumerate(masks):
            y, x = np.nonzero(mask)
            box = (
                [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
                if len(x)
                else [0] * 4
            )
            x0, y0, x1, y1 = box
            bits = np.packbits(mask[y0:y1, x0:x1].reshape(-1), bitorder="little")
            self.bits.append(bits)
            self.boxes.append(box)
            self.byte_offsets.append(self.byte_offsets[-1] + len(bits))
            self.ids.append(index)
        self.frames.append(int(frame))
        self.offsets.append(len(self.boxes))
        self.scores.extend(scores)
        self.pred_boxes.extend(pred_boxes)
        self.classes.extend(classes)
        self.mask_hashes.append(array_sha256(masks))

    def arrays(self):
        return {
            "image_shape_hw": np.asarray(self.shape, dtype=np.int32),
            "frame_idx": np.asarray(self.frames, dtype=np.int64),
            "frame_offsets": np.asarray(self.offsets, dtype=np.int64),
            "mask_boxes": np.asarray(self.boxes, dtype=np.int32).reshape(-1, 4),
            "mask_byte_offsets": np.asarray(self.byte_offsets, dtype=np.int64),
            "mask_bits": np.concatenate(self.bits) if self.bits else np.empty(0, dtype=np.uint8),
            "scores": np.asarray(self.scores, dtype=np.float32),
            "pred_boxes": np.asarray(self.pred_boxes, dtype=np.float32).reshape(-1, 4),
            "classes": np.asarray(self.classes, dtype=np.int64),
            "instance_id": np.asarray(self.ids, dtype=np.int64),
            "source_track_id": np.full(len(self.boxes), -1, dtype=np.int64),
            "view_available": np.ones(len(self.frames), dtype=bool),
            "frame_mask_sha256": np.asarray(self.mask_hashes, dtype="S64"),
        }


def decode_frame_masks(data, index):
    """Decode one frame's N,H,W boolean masks; no persistent ID interpretation."""
    shape = tuple(int(v) for v in data["image_shape_hw"])
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("invalid mask image shape")
    if index < 0 or index >= len(data["frame_idx"]):
        raise IndexError(index)
    offsets = data["frame_offsets"]
    if len(offsets) != len(data["frame_idx"]) + 1:
        raise ValueError("invalid frame offsets")
    lo, hi = (int(v) for v in offsets[index : index + 2])
    if not 0 <= lo <= hi <= len(data["mask_boxes"]):
        raise ValueError("mask count/offset mismatch")
    result = np.zeros((hi - lo, *shape), dtype=bool)
    for target, instance in enumerate(range(lo, hi)):
        x0, y0, x1, y1 = (int(v) for v in data["mask_boxes"][instance])
        if not 0 <= x0 <= x1 <= shape[1] or not 0 <= y0 <= y1 <= shape[0]:
            raise ValueError("mask crop is outside image")
        start, stop = (int(v) for v in data["mask_byte_offsets"][instance : instance + 2])
        count = (y1 - y0) * (x1 - x0)
        if not 0 <= start <= stop <= len(data["mask_bits"]) or stop - start != (count + 7) // 8:
            raise ValueError("mask byte count/offset mismatch")
        unpacked = np.unpackbits(data["mask_bits"][start:stop], count=count, bitorder="little")
        result[target, y0:y1, x0:x1] = unpacked.reshape(y1 - y0, x1 - x0)
    return result


def validate_capture(cap, path, expected_count=348600, shape=SHAPE):
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    details = {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "frame_count": round(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "image_shape_hw": [
            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        ],
    }
    if details["frame_count"] != expected_count or not np.isclose(details["fps"], FPS, atol=1e-6):
        raise ValueError(f"source count or fps mismatch: {details}")
    if tuple(details["image_shape_hw"]) != shape:
        raise ValueError(f"source dimensions mismatch: {details}")
    return details


def read_frame(cap, frame, *, seek=False, shape=SHAPE):
    if seek and not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
        raise RuntimeError(f"cannot seek frame {frame}")
    ok, image = cap.read()
    if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
        raise RuntimeError(f"source frame position mismatch at {frame}")
    if image.shape != (*shape, 3) or image.dtype != np.uint8:
        raise ValueError(f"unexpected image shape or dtype at {frame}")
    return image


def verify_lossless_clip(path, hashes, shape=SHAPE):
    cap = cv2.VideoCapture(str(path))
    try:
        validate_capture(cap, path, expected_count=len(hashes), shape=shape)
        for index, expected in enumerate(hashes):
            image = read_frame(cap, index, shape=shape)
            if array_sha256(image) != expected:
                raise ValueError(f"FFV1 decoded pixel mismatch: {path}, clip frame {index}")
        if cap.read()[0]:
            raise ValueError(f"FFV1 contains extra frames: {path}")
    finally:
        cap.release()
    return {"decoded_frames": len(hashes), "pixel_sha256_matches": len(hashes), "extra_frames": 0}


def process_view(source, view, start, stop, predictor, synchronize, out):
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        source_details = validate_capture(cap, source)
        samples = select_background_frames(start, stop)
        background = np.median(
            np.stack([read_frame(cap, f, seek=True) for f in samples]), axis=0
        ).astype(np.uint8)
        background_path = out / f"background_{view}.png"
        if not cv2.imwrite(str(background_path), background):
            raise RuntimeError(f"cannot write background: {background_path}")
        raw_path = out / "raw" / f"{view}.avi"
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, SHAPE[::-1])
        if not writer.isOpened():
            raise RuntimeError(f"FFV1 encoder could not open {raw_path}")
        packed, raw_hashes, times = PackedMasks(), [], []
        for frame in range(start, stop):
            image = read_frame(cap, frame, seek=frame == start)
            raw_hashes.append(array_sha256(image))
            writer.write(image)
            tic = time.perf_counter()
            pred = predictor(image)["instances"].to("cpu")
            synchronize()
            times.append(time.perf_counter() - tic)
            packed.append(
                frame,
                pred.pred_masks.numpy(),
                pred.scores.numpy(),
                pred.pred_boxes.tensor.numpy(),
                pred.pred_classes.numpy(),
            )
            if (frame - start + 1) % 50 == 0 or frame == stop - 1:
                print(
                    json.dumps(
                        {"view": view, "completed": frame - start + 1, "frames": stop - start}
                    ),
                    flush=True,
                )
        writer.release()
        writer = None
    finally:
        if writer is not None:
            writer.release()
        cap.release()
    raw_qa = verify_lossless_clip(raw_path, raw_hashes)
    mask_path = out / "masks" / f"{view}.npz"
    data = packed.arrays()
    np.savez_compressed(mask_path, **data)
    with np.load(mask_path, allow_pickle=False) as saved:
        reloaded = {key: saved[key] for key in saved.files}
    for index, expected in enumerate(packed.mask_hashes):
        if array_sha256(decode_frame_masks(reloaded, index)) != expected:
            raise ValueError(f"packed mask roundtrip mismatch: {view}, frame {start + index}")
    return {
        "source": source_details,
        "raw": {
            "path": raw_path.relative_to(out).as_posix(),
            "sha256": sha256(raw_path),
            "bytes": raw_path.stat().st_size,
            "frame_bgr_sha256": raw_hashes,
            "decode_qa": raw_qa,
        },
        "masks": {
            "path": mask_path.relative_to(out).as_posix(),
            "sha256": sha256(mask_path),
            "bytes": mask_path.stat().st_size,
            "instances_per_frame": np.diff(data["frame_offsets"]).tolist(),
            "roundtrip_verified_frames": len(packed.frames),
            "inference_s_by_frame": times,
        },
        "background": {
            "path": background_path.relative_to(out).as_posix(),
            "sha256": sha256(background_path),
            "frames": samples,
        },
    }


def run(medaka_root, out, window):
    out, medaka_root = Path(out), Path(medaka_root).resolve(strict=True)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite sequence staging: {out}")
    start, stop = WINDOWS[window]
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    conservative_bytes = (stop - start) * len(VIEWS) * np.prod(SHAPE).item() * 3 * 2
    if free < conservative_bytes:
        raise RuntimeError(f"insufficient free space: {free} < conservative {conservative_bytes}")
    sources = {v: medaka_root / "medaka_202512" / EXPERIMENT / v / VIDEO_NAMES[v] for v in VIEWS}
    for source in sources.values():
        cap = cv2.VideoCapture(str(source))
        try:
            validate_capture(cap, source)
        finally:
            cap.release()
    assets, asset_manifest = verify_assets(medaka_root)
    import detectron2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fixed sequence preparation requires the existing CUDA Detectron2 runtime"
        )
    out.mkdir(parents=True, exist_ok=False)
    with provenance("prepare_three_body_sequence", out, name="staging") as record:
        tic = time.perf_counter()
        for name in ("raw", "masks", "code_snapshot"):
            (out / name).mkdir()
        code = {}
        for path in (Path(__file__), REPO / "src/medaka3d/provenance.py"):
            shutil.copyfile(path, out / "code_snapshot" / path.name)
            code[path.relative_to(REPO).as_posix()] = sha256(path)
        record.param("window", window)
        record.param("source_interval_half_open", [start, stop])
        record.param("source_fps", FPS)
        record.param("threshold", 0.5)
        record.param("source_code_sha256", code)
        record.input("asset_manifest", asset_manifest, sha256=sha256(asset_manifest))
        for key, entry in assets.items():
            record.input(key, entry["path"], sha256=entry["sha256"])
        runtime = {
            "python": sys.version,
            "executable": sys.executable,
            "detectron2": detectron2.__version__,
            "detectron2_distribution": importlib.metadata.version("detectron2"),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
        models, views = {}, {}
        for kind, names in (("top", ("top_cam",)), ("side", ("side1_cam", "side2_cam"))):
            config_path = Path(assets[f"{kind}_config"]["path"])
            cfg = get_cfg()
            cfg.merge_from_file(str(config_path))
            cfg.MODEL.WEIGHTS = assets[f"{kind}_weights"]["path"]
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda"
            cfg.freeze()
            effective = out / f"{kind}_effective_config.yaml"
            effective.write_text(cfg.dump(), encoding="utf-8")
            shutil.copyfile(config_path, out / f"{kind}_source_config.yaml")
            predictor = DefaultPredictor(cfg)
            models[kind] = {
                "weights": assets[f"{kind}_weights"],
                "config": assets[f"{kind}_config"],
                "effective_config": effective.name,
                "effective_config_sha256": sha256(effective),
            }
            for view in names:
                views[view] = process_view(
                    sources[view], view, start, stop, predictor, torch.cuda.synchronize, out
                )
                record.input(view + "_video", sources[view])
                for part in ("raw", "masks", "background"):
                    item = views[view][part]
                    record.output(view + "_" + part, out / item["path"], sha256=item["sha256"])
            del predictor
        manifest = {
            "experiment": EXPERIMENT,
            "window": window,
            "frames": list(range(start, stop)),
            "source_interval_half_open": [start, stop],
            "fps": FPS,
            "image_shape_hw": list(SHAPE),
            "raw": {v: views[v]["raw"] for v in VIEWS},
            "masks": {v: views[v]["masks"] for v in VIEWS},
            "backgrounds": {v: views[v]["background"] for v in VIEWS},
            "source_videos": {v: views[v]["source"] for v in VIEWS},
            "raw_codec": "FFV1 AVI, lossless BGR uint8; clip index = source frame - start",
            "mask_schema": "bbox_packbits_instances_v1",
            "mask_encoding": "Crop flattened C order; packbits little endian; K+1 byte offsets",
            "mask_boxes": "K,4 int32 [x0,y0,x1,y1], exclusive upper bounds; empty mask [0,0,0,0]",
            "view_available": "true for every processed frame, including zero detections",
            "identity_fields": "instance_id: within-frame order; source_track_id: always -1",
            "background_candidates": list(BACKGROUND_CANDIDATES),
            "background_rule": "offline median excluding [start-332, stop+332), no track/ID input",
            "models": models,
            "runtime": runtime,
            "code_sha256": code,
            "code_origin": ORIGIN,
            "asset_manifest_sha256": sha256(asset_manifest),
            "disk_free_preflight_bytes": free,
            "staging_elapsed_s": time.perf_counter() - tic,
            "elapsed_excludes": "initial source-video preflight, weight hashes and imports",
            "limitations": [
                "Frame positions are verified; cross-camera timestamp sync is inherited.",
                "These development windows are not held-out identity or detector validation.",
                "No SLEAP, 3D/ID constraint, count restriction or post-hoc detection filtering.",
                "No temporal subsampling; all native-rate source frames are included.",
            ],
        }
        path = out / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        record.output("manifest", path, sha256=sha256(path))
    print(
        json.dumps(
            {"out": str(out), "frames": stop - start, "elapsed_s": manifest["staging_elapsed_s"]}
        ),
        flush=True,
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medaka-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", choices=tuple(WINDOWS), required=True)
    args = parser.parse_args(argv)
    run(args.medaka_root, args.out, args.window)


if __name__ == "__main__":
    main()
