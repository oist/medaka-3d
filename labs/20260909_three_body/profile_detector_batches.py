"""Bounded FP32 Detectron2 batch-size comparison on identical staged raw pixels.

Run only when the root schedules an idle GPU, using the existing dedicated D2
environment and ``-B``. DefaultPredictor batch 1 is the reference. Batches 2 and
4 reproduce its preprocessing and call the same model on a list of inputs.
No inference or staging implementation is changed by this diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from prepare_sequence import verify_assets

from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_four_frames_per_view(sequence):
    manifest_path = sequence / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest["frames"]) < 4 or manifest["fps"] != 166:
        raise ValueError("profile needs at least four native-rate staged frames")
    selected = {}
    for view in VIEWS:
        info = manifest["raw"][view]
        path = (sequence / info["path"]).resolve(strict=True)
        if not path.is_relative_to(sequence.resolve(strict=True)):
            raise ValueError("raw clip path escaped the sequence directory")
        cap = cv2.VideoCapture(str(path))
        selected[view] = []
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {path}")
            for index, frame in enumerate(manifest["frames"][:4]):
                ok, image = cap.read()
                if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                    raise RuntimeError("raw clip position mismatch")
                if image.shape != (1200, 1200, 3) or image.dtype != np.uint8:
                    raise ValueError("raw image is not native 1200x1200 uint8 BGR")
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest != info["frame_bgr_sha256"][index]:
                    raise ValueError("decoded raw pixels differ from staging")
                selected[view].append((f"f{frame:06d}_{view}", image, digest))
        finally:
            cap.release()
    return manifest, selected


def cpu_result(prediction):
    instances = prediction["instances"].to("cpu")
    return {
        "masks": instances.pred_masks.numpy(),
        "scores": instances.scores.numpy(),
        "boxes": instances.pred_boxes.tensor.numpy(),
        "classes": instances.pred_classes.numpy(),
    }


def manual_inputs(predictor, images, torch):
    inputs = []
    for image in images:
        if predictor.input_format == "RGB":
            image = image[:, :, ::-1]
        height, width = image.shape[:2]
        resized = predictor.aug.get_transform(image).apply_image(image)
        tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
        inputs.append({"image": tensor, "height": height, "width": width})
    return inputs


def predict(predictor, samples, batch_size, torch, *, manual_single=False):
    outputs = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for offset in range(0, len(samples), batch_size):
            subset = samples[offset : offset + batch_size]
            if batch_size == 1 and not manual_single:
                results = [predictor(subset[0][1])]
            else:
                results = predictor.model(
                    manual_inputs(predictor, [entry[1] for entry in subset], torch)
                )
            for (name, _, _), result in zip(subset, results, strict=True):
                outputs[name] = cpu_result(result)
    return outputs


def compare(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("comparison has different input image names")
    result = {}
    for name in reference:
        before, after = reference[name], candidate[name]
        same_shapes = all(before[key].shape == after[key].shape for key in before)
        equal = same_shapes and all(np.array_equal(before[key], after[key]) for key in before)
        entry = {
            "reference_instances": len(before["scores"]),
            "candidate_instances": len(after["scores"]),
            "same_shapes": same_shapes,
            "all_arrays_exact": equal,
        }
        if same_shapes:
            union = np.count_nonzero(before["masks"] | after["masks"], axis=(1, 2))
            changed = np.count_nonzero(before["masks"] ^ after["masks"], axis=(1, 2))
            entry.update(
                changed_mask_pixels=int(changed.sum()),
                mask_iou_by_proposal_order=(1 - changed / np.maximum(union, 1)).tolist(),
                boxes_max_abs=float(np.max(np.abs(before["boxes"] - after["boxes"]), initial=0)),
                scores_max_abs=float(np.max(np.abs(before["scores"] - after["scores"]), initial=0)),
                classes_exact=bool(np.array_equal(before["classes"], after["classes"])),
            )
        result[name] = entry
    return {
        "all_images_exact": all(value["all_arrays_exact"] for value in result.values()),
        "by_image": result,
        "matching_scope": "same-image native proposal order; no animal ID interpretation",
    }


def save_predictions(path, outputs):
    np.savez_compressed(
        path,
        **{
            f"{name}__{key}": value
            for name, fields in outputs.items()
            for key, value in fields.items()
        },
    )


def run(sequence, medaka_root, out, repeats):
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite batch profile: {out}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    manifest, images = read_four_frames_per_view(sequence)
    assets, asset_manifest = verify_assets(medaka_root)
    import detectron2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("the scheduled profile requires the existing CUDA D2 environment")
    out.mkdir(parents=True, exist_ok=False)
    with provenance("detectron2_fp32_batch_profile", out, name="profile") as record:
        code_sources = (
            Path(__file__),
            LAB / "prepare_sequence.py",
            REPO / "src/medaka3d/provenance.py",
        )
        hashes = {str(path.relative_to(REPO)): sha256(path) for path in code_sources}
        for source in code_sources:
            target = out / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        record.param("source_hashes", hashes)
        record.param("batch_sizes", [1, 2, 4])
        record.param("repeats", repeats)
        record.param("fp32_no_autocast", True)
        record.input(
            "sequence_manifest",
            sequence / "manifest.json",
            sha256=sha256(sequence / "manifest.json"),
        )
        record.input("asset_manifest", asset_manifest, sha256=sha256(asset_manifest))
        for key, entry in assets.items():
            record.input(key, entry["path"], sha256=entry["sha256"])
        environment = {
            "python": sys.version,
            "executable": sys.executable,
            "detectron2": detectron2.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "torch_cpu_threads": torch.get_num_threads(),
            "opencv_threads": cv2.getNumThreads(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "note": "backend precision defaults are recorded and unchanged for every batch",
        }
        record.param("environment", environment)
        summaries = {}
        for kind, views in (("top", ("top_cam",)), ("side", ("side1_cam", "side2_cam"))):
            cfg = get_cfg()
            cfg.merge_from_file(assets[f"{kind}_config"]["path"])
            cfg.MODEL.WEIGHTS = assets[f"{kind}_weights"]["path"]
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda"
            cfg.freeze()
            config_path = out / f"{kind}_effective_config.yaml"
            config_path.write_text(cfg.dump(), encoding="utf-8")
            predictor = DefaultPredictor(cfg)
            if any(parameter.dtype != torch.float32 for parameter in predictor.model.parameters()):
                raise ValueError("retained model parameters are not FP32")
            samples = [sample for view in views for sample in images[view]]
            reference = predict(predictor, samples, 1, torch)
            manual_one = predict(predictor, samples, 1, torch, manual_single=True)
            manual_check = compare(reference, manual_one)
            # Warm all batch shapes before timing; reserved memory therefore includes
            # allocator cache history. Peak allocated memory is the primary VRAM metric.
            for size in (1, 2, 4):
                predict(predictor, samples, size, torch)
            durations = {str(size): [] for size in (1, 2, 4)}
            allocated = {str(size): [] for size in (1, 2, 4)}
            reserved = {str(size): [] for size in (1, 2, 4)}
            representative, stability = {}, {str(size): True for size in (1, 2, 4)}
            for repeat in range(repeats):
                for size in (1, 2, 4) if repeat % 2 == 0 else (4, 2, 1):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    tic = time.perf_counter()
                    outputs = predict(predictor, samples, size, torch)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tic
                    key = str(size)
                    durations[key].append(elapsed)
                    allocated[key].append(torch.cuda.max_memory_allocated())
                    reserved[key].append(torch.cuda.max_memory_reserved())
                    if key not in representative:
                        representative[key] = outputs
                    else:
                        stability[key] &= compare(representative[key], outputs)["all_images_exact"]
                    print(
                        json.dumps(
                            {
                                "model": kind,
                                "batch": size,
                                "repeat": repeat,
                                "images": len(samples),
                                "elapsed_s": elapsed,
                                "peak_allocated_bytes": allocated[key][-1],
                            }
                        ),
                        flush=True,
                    )
            comparisons = {key: compare(reference, value) for key, value in representative.items()}
            prediction_hashes = {}
            for key, value in {
                "reference": reference,
                "manual_one": manual_one,
                **representative,
            }.items():
                path = out / f"{kind}_{key}_predictions.npz"
                save_predictions(path, value)
                prediction_hashes[path.name] = sha256(path)
                record.output(path.stem, path, sha256=prediction_hashes[path.name])
            summaries[kind] = {
                "images": len(samples),
                "frames_by_view": {view: manifest["frames"][:4] for view in views},
                "input_pixel_sha256": {name: digest for name, _, digest in samples},
                "weights": assets[f"{kind}_weights"],
                "source_config": assets[f"{kind}_config"],
                "effective_config_sha256": sha256(config_path),
                "manual_batch1_check": manual_check,
                "seconds_by_repeat": durations,
                "mean_seconds": {key: float(np.mean(value)) for key, value in durations.items()},
                "peak_allocated_bytes": {key: max(value) for key, value in allocated.items()},
                "peak_reserved_bytes": {key: max(value) for key, value in reserved.items()},
                "repeat_outputs_exact": stability,
                "vs_reference": comparisons,
                "prediction_files_sha256": prediction_hashes,
            }
            del predictor, reference, manual_one, representative, outputs
        if any(sha256(REPO / path) != digest for path, digest in hashes.items()):
            raise RuntimeError("profile source changed while running")
        result = {
            "protocol": "same_native_pixels_fp32_detectron2_batch_only",
            "environment": environment,
            "models": summaries,
            "source_sha256": hashes,
            "timing_includes": (
                "standard resize/CPU input tensor creation, GPU inference, CPU result transfer"
            ),
            "timing_excludes": (
                "raw decode/checksums, model initialization, warmup and comparison diagnostics"
            ),
            "peak_memory_scope": (
                "CUDA peak allocated includes resident model; reserved retains warmup cache"
            ),
            "limitations": [
                "This detector experiment is not end-to-end tracking or identity validation.",
                "Equal FP32 dtype does not guarantee identical outputs across batch kernels.",
                "All mask/box/score changes are reported; exact equivalence is not presumed.",
                "The sequential FFV1 writer and decode costs remain outside this speed comparison.",
                "Side views share one model, permitting cross-view side batches.",
            ],
        }
        path = out / "comparison.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("comparison", path, sha256=sha256(path))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--medaka-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    run(args.sequence, args.medaka_root, args.out, args.repeats)


if __name__ == "__main__":
    main()
