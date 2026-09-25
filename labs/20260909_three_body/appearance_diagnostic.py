"""Fixed instance-core colors, conditional on saved slots; never identity truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

EROSION_RADIUS = 2
MIN_CORE_PIXELS = 12
MIN_TEMPLATE_SAMPLES = 5
CONFUSION_KEY = "conditional_confusion_rows_reference_slot_columns_nearest_template_plus_unscored"
FEATURES = (
    "median_r_fraction",
    "median_g_fraction",
    "mean_s_cos_h_scaled",
    "mean_s_sin_h_scaled",
    "median_saturation",
    "median_value",
)


def describe_core(image, mask):
    """Six bounded color values on a 5x5 elliptical erosion, no whole-image histogram."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("native uint8 BGR image required")
    if mask.dtype != bool or mask.shape != image.shape[:2]:
        raise ValueError("full image boolean instance mask required")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core = cv2.erode(
        mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    if core.sum() < MIN_CORE_PIXELS:
        return None, core
    bgr = image[core].astype(np.float32) / 255
    normalized = bgr / np.maximum(bgr.sum(-1, keepdims=True), 1e-8)
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    angle, saturation, value = np.deg2rad(hsv[:, 0]), hsv[:, 1], hsv[:, 2]
    descriptor = np.array(
        [
            np.median(normalized[:, 2]),
            np.median(normalized[:, 1]),
            (1 + np.mean(saturation * np.cos(angle))) / 2,
            (1 + np.mean(saturation * np.sin(angle))) / 2,
            np.median(saturation),
            np.median(value),
        ],
        dtype=float,
    )
    assert np.isfinite(descriptor).all() and np.all((descriptor >= 0) & (descriptor <= 1))
    return descriptor, core


def assign_instances(masks, projected, direct):
    """Both mouth/pectoral must lie in one mask, with every latent competitor retained."""
    if projected.shape != (3, 2, 2) or direct.shape != (3,):
        raise ValueError("expected three fish mouth/pectoral UV and support flags")
    rows = np.full(3, -1, dtype=int)
    reasons = np.full(3, "no_direct_mouth_pect", dtype="U64")
    if masks is None:
        reasons[:] = "view_unavailable"
        return rows, reasons
    if masks.dtype != bool or masks.ndim != 3:
        raise ValueError("instance masks must have shape N,H,W and bool dtype")
    if not len(masks):
        reasons[:] = "no_masks"
        return rows, reasons
    height, width = masks.shape[-2:]
    finite = np.isfinite(projected).all(-1)
    uv = np.rint(np.where(np.isfinite(projected), projected, -1)).astype(int)
    valid = finite & (uv[..., 0] >= 0) & (uv[..., 0] < width)
    valid &= (uv[..., 1] >= 0) & (uv[..., 1] < height)
    inside = np.zeros((len(masks), 3, 2), bool)
    for fish in range(3):
        for node in range(2):
            if valid[fish, node]:
                inside[:, fish, node] = masks[:, uv[fish, node, 1], uv[fish, node, 0]]
    for fish in range(3):
        if not direct[fish]:
            continue
        if not valid[fish].all():
            reasons[fish] = "projection_invalid"
            continue
        candidates = np.flatnonzero(inside[:, fish].all(-1))
        if not len(candidates):
            reasons[fish] = "no_joint_instance"
        elif len(candidates) != 1:
            reasons[fish] = "multiple_joint_instances"
        elif not valid.all():
            reasons[fish] = "unknown_latent_competitor_projection"
        elif inside[candidates[0], np.arange(3) != fish].any():
            reasons[fish] = "mask_contains_other_projected_node"
        else:
            rows[fish] = candidates[0]
            reasons[fish] = "usable"
    return rows, reasons


def summarize(features, reasons, frames, train_stop):
    train = frames < train_stop
    templates = np.full((3, 3, len(FEATURES)), np.nan)
    report = {}
    distances = np.full((len(frames), 3, 3, 3), np.nan)
    predictions = np.full((len(frames), 3, 3), -1, int)
    for view in range(3):
        counts = []
        intra = []
        for fish in range(3):
            values = features[train, view, fish]
            values = values[np.isfinite(values).all(-1)]
            counts.append(len(values))
            if len(values) >= MIN_TEMPLATE_SAMPLES:
                templates[view, fish] = np.median(values, axis=0)
                within = np.linalg.norm(values - templates[view, fish], axis=-1)
                intra.append(
                    {"median": float(np.median(within)), "p95": float(np.percentile(within, 95))}
                )
            else:
                intra.append(None)
        confusion = np.zeros((3, 4), int)
        margins = [[] for _ in range(3)]
        for t in np.flatnonzero(~train):
            for fish in range(3):
                descriptor = features[t, view, fish]
                predicted = -1
                if np.isfinite(descriptor).all() and np.isfinite(templates[view]).all():
                    d = np.linalg.norm(templates[view] - descriptor, axis=-1)
                    distances[t, view, fish] = d
                    best = np.flatnonzero(np.isclose(d, d.min(), rtol=0, atol=1e-12))
                    if len(best) == 1:
                        predicted = int(best[0])
                    margins[fish].append(float(np.min(d[np.arange(3) != fish]) - d[fish]))
                predictions[t, view, fish] = predicted
                confusion[fish, predicted if predicted >= 0 else 3] += 1
        usable = int(confusion[:, :3].sum())
        report[str(view)] = {
            "train_samples_per_reference_slot": counts,
            "train_within_template_distance": intra,
            "template_pairwise_euclidean": np.where(
                np.isfinite(templates[view]).all(),
                np.linalg.norm(templates[view, :, None] - templates[view, None], axis=-1),
                np.nan,
            ).tolist(),
            CONFUSION_KEY: confusion.tolist(),
            "validation_requested_fish_probes": int((~train).sum() * 3),
            "validation_classified_fish_probes": usable,
            "same_reference_slot_fraction_among_classified_not_identity_accuracy": (
                float(np.trace(confusion[:, :3]) / usable) if usable else None
            ),
            "reference_distance_margin_other_minus_own": [
                {
                    "n": len(row),
                    "median": float(np.median(row)),
                    "p10": float(np.percentile(row, 10)),
                    "p90": float(np.percentile(row, 90)),
                }
                if row
                else None
                for row in margins
            ],
            "train_reason_counts": dict(Counter(reasons[train, view].ravel().tolist())),
            "validation_reason_counts": dict(Counter(reasons[~train, view].ravel().tolist())),
        }
    return report, templates, distances, predictions


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    lab = Path(__file__).resolve().parent
    repo = lab.parents[1]
    sys.path.insert(0, str(lab))
    from prepare_sequence import decode_frame_masks
    from run_experiment import CALIBRATION_DIR, staging_catalog, verified_input

    from medaka3d.calib import load_rig
    from medaka3d.provenance import provenance
    from medaka3d.types import VIEWS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    stage = repo / "work/three_body/inputs/interval_76620-77286"
    run = repo / "work/three_body/v1_interval_detectron2"
    input_root = repo / "work/generative_tracking_inputs"
    frames = np.r_[np.arange(76620, 76786), np.arange(76786, 77286, 10)]
    code = [
        Path(__file__),
        lab / "prepare_sequence.py",
        lab / "run_experiment.py",
        lab / "observations.py",
    ] + [repo / "src/medaka3d" / n for n in ("calib.py", "types.py", "provenance.py")]
    hashes = {str(p): sha(p) for p in code}
    for name in ("candidate.npz", "summary.json", "experiment.provenance.json"):
        hashes[str(run / name)] = sha(run / name)
    record = json.loads((run / "experiment.provenance.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert record["status"] == "ok" and summary["technical_completion"] is True
    assert summary["frames"] == [76620, 77286] and summary["returned_frames"] == 666
    for key, name in (("candidate", "candidate.npz"), ("summary", "summary.json")):
        assert record["outputs"][key]["sha256"] == hashes[str(run / name)]
    for info in record["inputs"].values():
        p = Path(info["path"])
        p = (p if p.is_absolute() else repo / p).resolve(strict=True)
        assert p.is_relative_to(repo) and sha(p) == info["sha256"]
        hashes[str(p)] = info["sha256"]
    assert summary["input_sha256_before"] == summary["input_sha256_after"]
    with np.load(run / "candidate.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["frame_idx"], np.arange(76620, 77286))
        index = frames - 76620
        pose = arrays["pose_xyz"][index]
        observed = arrays["observed"][index]
        sources = arrays["sources"][index]
        reference_review = arrays["identity_review"][index]
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes[str(manifest_path)] = sha(manifest_path)
    assert hashes[str(manifest_path)] in summary["input_sha256_before"].values()
    catalog, catalog_path = staging_catalog(input_root)
    hashes[str(catalog_path)] = sha(catalog_path)
    for relative in catalog:
        if relative.startswith(CALIBRATION_DIR):
            p = verified_input(input_root, relative, catalog)
            hashes[str(p)] = sha(p)
            assert hashes[str(p)] in summary["input_sha256_before"].values()
    rig = load_rig(input_root / CALIBRATION_DIR)
    projections = np.stack([rig.camera(v).project(pose[:, :, :2]) for v in VIEWS], axis=1)
    features = np.full((len(frames), 3, 3, len(FEATURES)), np.nan)
    reasons = np.full((len(frames), 3, 3), "pending", dtype="U64")
    selected_rows = np.full((len(frames), 3, 3), -1, int)
    core_counts = np.zeros((len(frames), 3, 3), int)
    pixel_checks = []
    examples = []
    example_metadata = []
    args.output.mkdir(parents=True, exist_ok=False)
    for vi, view in enumerate(VIEWS):
        data = {}
        for category in ("raw", "masks"):
            info = manifest[category][view]
            path = (stage / info["path"]).resolve(strict=True)
            assert path.is_relative_to(stage) and sha(path) == info["sha256"]
            assert info["sha256"] in summary["input_sha256_before"].values()
            hashes[str(path)] = info["sha256"]
            if category == "raw":
                video = path
            else:
                with np.load(path, allow_pickle=False) as arrays:
                    data = {k: arrays[k] for k in arrays.files}
                np.testing.assert_array_equal(data["frame_idx"], np.arange(76620, 77286))
        capture = cv2.VideoCapture(str(video))
        try:
            assert capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) == 666
            assert np.isclose(capture.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
            saved_example = set()
            for t, frame in enumerate(frames):
                ix = int(frame - 76620)
                if capture.get(cv2.CAP_PROP_POS_FRAMES) != ix:
                    assert capture.set(cv2.CAP_PROP_POS_FRAMES, ix)
                ok, image = capture.read()
                assert ok and capture.get(cv2.CAP_PROP_POS_FRAMES) == ix + 1
                assert image.shape == (1200, 1200, 3) and image.dtype == np.uint8
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                assert digest == manifest["raw"][view]["frame_bgr_sha256"][ix]
                pixel_checks.append({"view": view, "frame": int(frame), "bgr_sha256": digest})
                masks = decode_frame_masks(data, ix) if data["view_available"][ix] else None
                direct = observed[t, :, :2].all(-1) & (sources[t, :, vi, :2] >= 0).all(-1)
                selected_rows[t, vi], reasons[t, vi] = assign_instances(
                    masks, projections[t, vi], direct
                )
                for fish, row in enumerate(selected_rows[t, vi]):
                    if row < 0:
                        continue
                    descriptor, core = describe_core(image, masks[row])
                    core_counts[t, vi, fish] = int(core.sum())
                    if descriptor is None:
                        reasons[t, vi, fish] = "core_too_small"
                        continue
                    features[t, vi, fish] = descriptor
                    phase = "train" if frame < 76786 else "validation"
                    key = (phase, fish)
                    if key not in saved_example:
                        saved_example.add(key)
                        yy, xx = np.nonzero(masks[row])
                        x0, x1 = max(0, int(xx.min()) - 8), min(1200, int(xx.max()) + 9)
                        y0, y1 = max(0, int(yy.min()) - 8), min(1200, int(yy.max()) + 9)
                        crop = image[y0:y1, x0:x1]
                        marked = crop.copy()
                        contours, _ = cv2.findContours(
                            masks[row, y0:y1, x0:x1].astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(marked, contours, -1, (0, 200, 255), 1)
                        marked[core[y0:y1, x0:x1]] = np.rint(
                            0.5 * marked[core[y0:y1, x0:x1]] + 0.5 * np.array([255, 255, 0])
                        ).astype(np.uint8)
                        display = np.concatenate([crop, marked], axis=1)
                        display = cv2.resize(display, (400, 200), interpolation=cv2.INTER_NEAREST)
                        panel = np.full((230, 400, 3), 24, np.uint8)
                        panel[30:] = display
                        label = f"{view} S{fish} {phase} f{frame}"
                        cv2.putText(
                            panel,
                            label,
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            (240, 240, 240),
                            1,
                        )
                        examples.append(panel)
                        path = args.output / f"roi_{view}_S{fish}_{phase}_f{frame}.png"
                        assert cv2.imwrite(str(path), crop)
                        example_metadata.append(
                            {
                                "view": view,
                                "reference_slot": fish,
                                "frame": int(frame),
                                "instance_row": int(row),
                                "crop_xyxy": [x0, y0, x1, y1],
                                "core_pixels": int(core.sum()),
                                "native_crop": path.name,
                                "core_color_overlay": "cyan, display only",
                                "descriptor": descriptor.tolist(),
                            }
                        )
        finally:
            capture.release()
        print(
            json.dumps(
                {
                    "view": view,
                    "selected_probes": len(frames),
                    "usable": int(np.isfinite(features[:, vi]).all(-1).sum()),
                }
            ),
            flush=True,
        )
    statistics, templates, distances, predictions = summarize(features, reasons, frames, 76786)
    while len(examples) % 3:
        examples.append(np.full((230, 400, 3), 24, np.uint8))
    if examples:
        sheet = np.concatenate(
            [np.concatenate(examples[i : i + 3], axis=1) for i in range(0, len(examples), 3)],
            axis=0,
        )
        assert cv2.imwrite(str(args.output / "roi_examples.png"), sheet)
    np.savez_compressed(
        args.output / "descriptors.npz",
        frame_idx=frames,
        view_names=np.array(VIEWS),
        features=features,
        templates=templates,
        distances=distances,
        predictions=predictions,
        reasons=reasons,
        instance_rows=selected_rows,
        core_pixels=core_counts,
        projected_mouth_pect=projections,
        reference_identity_review=reference_review,
    )
    report = _jsonable(
        {
            "status": "ok",
            "scope": "Saved-slot conditional appearance diagnostic; NOT biological ID accuracy",
            "frames": [76620, 77286],
            "train": [76620, 76786],
            "validation": [76786, 77286],
            "validation_stride_native_frames": 10,
            "validation_probes": 50,
            "tracking_validation_or_new_inference": False,
            "gpu_used": False,
            "features": FEATURES,
            "erosion_radius_px": EROSION_RADIUS,
            "minimum_core_pixels": MIN_CORE_PIXELS,
            "minimum_template_samples": MIN_TEMPLATE_SAMPLES,
            "template": "Per-view/slot median; Euclidean distance; no threshold search",
            "views": {v: statistics[str(i)] for i, v in enumerate(VIEWS)},
            "decoded_raw_native_images": len(pixel_checks),
            "pixel_hashes": pixel_checks,
            "roi_examples": example_metadata,
            "roi_display": "Crops beside mask/core overlays, scaled to panel; metric unchanged",
            "reference_review_true_train": reference_review[frames < 76786].sum(axis=0).tolist(),
            "reference_review_true_validation": reference_review[frames >= 76786]
            .sum(axis=0)
            .tolist(),
            "runtime": {
                "seconds": time.perf_counter() - started,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "limitations": [
                "Saved slots may switch and are not human truth.",
                "Labels come from the complete 666-frame forward result, "
                "not independent prospective labels.",
                "Masks were independently inferred; assignment depends on saved geometry.",
                "Mask cores can include reflections, neighboring fish or remaining background.",
                "Review flags are recorded, not used to hide difficult samples or define truth.",
                "One recording; 50 validation probes are not 500-frame tracking.",
                "No model selection, threshold fitting, tracker change or ID correction.",
            ],
        }
    )
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    assert all(sha(p) == h for p, h in hashes.items())
    with provenance("instance_core_color_diagnostic", args.output, name="diagnostic") as rec:
        for p, h in hashes.items():
            rec.input(p, p, sha256=h)
        for p in code:
            target = args.output / "source_snapshot" / p.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        for p in [
            *sorted(args.output.glob("*.png")),
            args.output / "summary.json",
            args.output / "descriptors.npz",
        ]:
            rec.output(p.name, p, sha256=sha(p))
    return report


if __name__ == "__main__":
    main()
