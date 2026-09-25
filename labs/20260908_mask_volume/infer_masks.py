"""Fresh framewise Detectron2 masks using the retained, checksum-verified models.

Run with the already installed dedicated Detectron2 environment and ``-B``.
No training, SLEAP, ID input, mask-count restriction, or cached-mask input is used.
The source PNGs are full images; DefaultPredictor applies the saved standard
test resize and returns masks in original image coordinates.
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

# This dedicated environment need not install the new repository package.
# The shared provenance package imports only standard-library dependencies.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

VIEWS = ("top_cam", "side1_cam", "side2_cam")
MODEL_DIRS = {
    "top": "medaka_202512/detectron2_models/top_detectron2_model",
    "side": "medaka_202512/detectron2_models/side_detectron2_model_firstprompt",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw(raw_root):
    raw_root = Path(raw_root).resolve(strict=True)
    path = raw_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    if (
        not frames
        or any(type(f) is not int or f < 0 for f in frames)
        or frames != sorted(set(frames))
    ):
        raise ValueError("raw frame indexes must be nonnegative, sorted and unique integers")
    images, hashes = {}, {}
    for view in VIEWS:
        images[view] = []
        for frame in frames:
            image_path = raw_root / f"f{frame:06d}_{view}.png"
            expected = manifest["outputs"][image_path.name]["sha256"]
            actual = sha256(image_path)
            if actual != expected:
                raise ValueError(f"raw PNG checksum mismatch: {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None or image.shape != (1200, 1200, 3) or image.dtype != np.uint8:
                raise ValueError(f"raw PNG is not a full 1200x1200 uint8 BGR image: {image_path}")
            images[view].append(image)
            hashes[image_path.name] = actual
    return manifest, images, hashes


def verify_assets(medaka_root):
    medaka_root = Path(medaka_root).resolve(strict=True)
    manifest_path = medaka_root / "assets" / "MANIFEST.json"
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
            if not path.is_relative_to(medaka_root):
                raise ValueError(f"model path escaped source root: {path}")
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"asset size mismatch: {path}")
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


def pack_predictions(frames, predictions, shape=(1200, 1200)):
    if len(frames) != len(predictions):
        raise ValueError("prediction count differs from frame count")
    masks, scores, boxes, pred_boxes, classes, indices, offsets = [], [], [], [], [], [], [0]
    for result in predictions:
        current = np.asarray(result["masks"])
        count = len(current)
        score = np.asarray(result["scores"], dtype=np.float32)
        box = np.asarray(result["pred_boxes"], dtype=np.float32)
        category = np.asarray(result["classes"], dtype=np.int64)
        if current.dtype != bool or current.shape != (count, *shape):
            raise ValueError("predicted mask dtype or native image shape is invalid")
        if score.shape != (count,) or box.shape != (count, 4) or category.shape != (count,):
            raise ValueError("prediction fields have inconsistent counts")
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError("prediction contains an invalid score")
        if not np.isfinite(box).all():
            raise ValueError("prediction contains a nonfinite box")
        for mask in current:
            y, x = np.nonzero(mask)
            boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1] if len(x) else [0] * 4)
        masks.extend(current)
        scores.extend(score)
        pred_boxes.extend(box)
        classes.extend(category)
        indices.extend(range(count))
        offsets.append(len(masks))
    return {
        "frame_idx": np.asarray(frames, dtype=np.int64),
        "frame_offsets": np.asarray(offsets, dtype=np.int64),
        "masks": np.stack(masks) if masks else np.empty((0, *shape), dtype=bool),
        "scores": np.asarray(scores, dtype=np.float32),
        "boxes": np.asarray(boxes, dtype=np.int32).reshape(-1, 4),
        "pred_boxes": np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4),
        "classes": np.asarray(classes, dtype=np.int64),
        "instance_id": np.asarray(indices, dtype=np.int64),
        "source_track_id": np.full(len(masks), -1, dtype=np.int64),
        "view_available": np.ones(len(frames), dtype=bool),
    }


def run(raw_root, medaka_root, out):
    out, raw_root = Path(out), Path(raw_root)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite fresh inference output: {out}")
    # Hash and decode every raw PNG, and verify both complete weights before load.
    raw_manifest, images, raw_hashes = read_raw(raw_root)
    assets, asset_manifest = verify_assets(medaka_root)
    import detectron2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("this fixed experiment requires the verified CUDA Detectron2 runtime")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with provenance("fresh_independent_detectron2_masks", out, name="inference") as record:
        frames = raw_manifest["frames"]
        record.param("frames", frames)
        record.param("score_threshold", 0.5)
        record.param("device", "cuda")
        record.param("identity_input", False)
        record.param("cached_mask_input", False)
        record.input(
            "raw_manifest", raw_root / "manifest.json", sha256=sha256(raw_root / "manifest.json")
        )
        record.input("asset_manifest", asset_manifest, sha256=sha256(asset_manifest))
        for key, entry in assets.items():
            record.input(key, entry["path"], sha256=entry["sha256"])
        snapshot = out / "code_snapshot"
        snapshot.mkdir()
        code_hashes = {}
        for source in (Path(__file__), REPO / "src/medaka3d/provenance.py"):
            shutil.copyfile(source, snapshot / source.name)
            code_hashes[str(source.relative_to(REPO))] = sha256(source)
        shutil.copyfile(raw_root / "manifest.json", out / "raw_manifest.json")
        environment = {
            "python": sys.version,
            "python_executable": sys.executable,
            "detectron2": detectron2.__version__,
            "detectron2_distribution": importlib.metadata.version("detectron2"),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cuda_arches": torch.cuda.get_arch_list(),
        }
        record.param("runtime", environment)
        models, views = {}, {}
        for kind, current_views in (("top", ("top_cam",)), ("side", ("side1_cam", "side2_cam"))):
            config_path = Path(assets[f"{kind}_config"]["path"])
            cfg = get_cfg()
            cfg.merge_from_file(str(config_path))
            original_weights = cfg.MODEL.WEIGHTS
            # The saved training config points to COCO pretraining. Always replace
            # that path before predictor construction; never download its URL.
            cfg.MODEL.WEIGHTS = assets[f"{kind}_weights"]["path"]
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda"
            cfg.freeze()
            config_output = out / f"{kind}_effective_config.yaml"
            config_output.write_text(cfg.dump(), encoding="utf-8")
            shutil.copyfile(config_path, out / f"{kind}_source_config.yaml")
            tic = time.perf_counter()
            predictor = DefaultPredictor(cfg)
            torch.cuda.synchronize()
            models[kind] = {
                "weights": assets[f"{kind}_weights"],
                "source_config": assets[f"{kind}_config"],
                "effective_config": config_output.name,
                "effective_config_sha256": sha256(config_output),
                "original_training_weights_field": original_weights,
                "overrides": {
                    "MODEL.WEIGHTS": cfg.MODEL.WEIGHTS,
                    "MODEL.ROI_HEADS.SCORE_THRESH_TEST": 0.5,
                    "MODEL.DEVICE": "cuda",
                },
                "initialization_s": time.perf_counter() - tic,
            }
            for view in current_views:
                predictions, timings = [], []
                for frame, image in zip(frames, images[view], strict=True):
                    tic = time.perf_counter()
                    pred = predictor(image)["instances"].to("cpu")
                    torch.cuda.synchronize()
                    predictions.append(
                        {
                            "masks": pred.pred_masks.numpy(),
                            "scores": pred.scores.numpy(),
                            "pred_boxes": pred.pred_boxes.tensor.numpy(),
                            "classes": pred.pred_classes.numpy(),
                        }
                    )
                    timings.append(time.perf_counter() - tic)
                    print(
                        json.dumps(
                            {
                                "view": view,
                                "frame": frame,
                                "instances": len(pred),
                                "inference_s": timings[-1],
                            }
                        ),
                        flush=True,
                    )
                data = pack_predictions(frames, predictions)
                path = out / f"{view}.npz"
                np.savez_compressed(path, **data)
                views[view] = {
                    "output_npz": path.name,
                    "output_sha256": sha256(path),
                    "model": kind,
                    "image_shape_hw": [1200, 1200],
                    "instances_per_selected_frame": np.diff(data["frame_offsets"]).tolist(),
                    "inference_s_by_frame": timings,
                    "inference_s_includes": "resize, GPU inference, device-to-host copy",
                }
                record.output(view, path, sha256=views[view]["output_sha256"])
            del predictor
        manifest = {
            "experiment": raw_manifest["experiment"],
            "schema": "offset_indexed_instance_masks_v1",
            "frames": frames,
            "views": views,
            "models": models,
            "environment": environment,
            "code_sha256": code_hashes,
            "inputs": {
                "raw_root": str(raw_root.resolve()),
                "raw_manifest_sha256": sha256(raw_root / "manifest.json"),
                "raw_png_sha256": raw_hashes,
                "asset_manifest_sha256": sha256(asset_manifest),
            },
            "npz_schema": {
                "frame_idx": "T source frame indexes",
                "frame_offsets": "T+1 offsets into K instances",
                "masks": "K,H,W bool, original full image coordinates",
                "scores": "K float32 native detector scores",
                "boxes": "K,4 int32 mask bounds [x0,y0,x1,y1], exclusive upper bounds",
                "pred_boxes": "K,4 float32 original detector boxes, image coordinates",
                "classes": "K native detector class indexes",
                "instance_id": "K within-frame proposal indexes, no persistence",
                "source_track_id": "K all -1; no source IDs",
                "view_available": "T true; actual inference performed, including zero detections",
            },
            "limitations": [
                "Ten frames were selected from shared stored cache indexes before raw inference.",
                "Models originate from this experiment; this is not an independent ID benchmark.",
                "No count truncation, post-hoc SLEAP filter, identity association or training.",
                "A successfully inferred zero-mask frame is known available, not missing.",
            ],
            "inference_and_output_elapsed_s": time.perf_counter() - started,
            "elapsed_excludes": "imports, raw PNG preflight and full model checksums",
        }
        path = out / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        record.output("manifest", path, sha256=sha256(path))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--medaka-root",
        type=Path,
        required=True,
        help="Read-only root containing assets/MANIFEST.json and medaka_202512",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run(args.raw, args.medaka_root, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "frames": manifest["frames"],
                "elapsed_s": manifest["inference_and_output_elapsed_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
