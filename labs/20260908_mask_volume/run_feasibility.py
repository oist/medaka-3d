"""Visualize mask-supported volumes; no identity labels or trajectory inputs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import label
from volume import MaskVolume, VolumeConfig

from medaka3d.calib import load_rig
from medaka3d.types import VIEWS


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def top_contrast(image, background):
    """Fixed background-difference diagnostic, not calibrated fish probability."""
    delta = image.astype(np.float32) - background.astype(np.float32)
    delta -= np.median(delta, axis=(0, 1))
    strength = np.max(np.abs(delta), axis=-1)
    components, _ = label(strength >= 12)
    counts = np.bincount(components.ravel())
    keep = counts >= 6
    keep[0] = False
    return np.where(keep[components], np.clip((strength - 6) / 12, 0, 1), 0).astype(np.float32)


def read_masks(root, frames):
    data = {}
    for view in VIEWS:
        with np.load(root / f"{view}.npz", allow_pickle=False) as src:
            entry = {k: src[k] for k in src.files}
        if not np.array_equal(entry["frame_idx"], frames):
            raise ValueError(f"mask frame mismatch: {view}")
        offsets = entry["frame_offsets"]
        if len(offsets) != len(frames) + 1 or offsets[0] != 0:
            raise ValueError(f"invalid offsets: {view}")
        if offsets[-1] != len(entry["masks"]) or np.any(np.diff(offsets) < 0):
            raise ValueError(f"mask count/offset mismatch: {view}")
        if entry["masks"].shape[1:] != (1200, 1200):
            raise ValueError(f"masks are not in native image coordinates: {view}")
        data[view] = entry
    return data


def frame_masks(data, index):
    masks, instance_masks, counts = {}, {}, {}
    for view in VIEWS:
        entry = data[view]
        lo, hi = entry["frame_offsets"][index : index + 2]
        subset = entry["masks"][lo:hi]
        instance_masks[view] = subset
        counts[view] = len(subset)
        available = bool(entry.get("view_available", np.ones(len(entry["frame_idx"]), bool))[index])
        masks[view] = np.any(subset, axis=0).astype(np.float32) if available else None
    return masks, instance_masks, counts


def draw_points(canvas, uv, color):
    valid = np.isfinite(uv).all(axis=1)
    points = np.rint(uv[valid] * 0.4).astype(int)
    for x, y in np.unique(points, axis=0):
        if 0 <= x < 480 and 0 <= y < 480:
            cv2.circle(canvas, (int(x), int(y)), 1, color, -1)


def preview(path, frame, mode, raw, instance_masks, masks, volume, result):
    # Full images: no crop can hide a far-away ghost volume.
    sheet = np.full((3 * 518 + 42, 3 * 480, 3), 22, dtype=np.uint8)
    cv2.putText(
        sheet,
        f"Frame {frame} | {mode} | candidates, not identities",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    for vi, view in enumerate(VIEWS):
        image = cv2.resize(raw[view], (480, 480), interpolation=cv2.INTER_AREA)
        for row, kind in enumerate(("input", "strict3", "two_of_three")):
            panel = image.copy()
            if kind == "input" and masks[view] is None:
                caption = f"{view}: mask UNKNOWN"
            elif kind == "input":
                contours, _ = cv2.findContours(
                    (masks[view] >= 0.5).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(
                    panel,
                    [np.rint(c * 0.4).astype(np.int32) for c in contours],
                    -1,
                    (255, 255, 0),
                    1,
                )
                caption = f"{view}: input, D2 n={len(instance_masks[view])}"
                if mode != "detectron2" and view == "top_cam":
                    caption = "top_cam: background contrast"
            else:
                selected = getattr(result, kind)
                draw_points(
                    panel,
                    volume.projected_by_view[view][selected],
                    (60, 250, 70) if kind == "strict3" else (0, 160, 255),
                )
                caption = f"{kind}: {int(selected.sum())} voxels"
            y = 42 + row * 518
            x = vi * 480
            cv2.putText(
                sheet,
                caption,
                (x + 7, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            sheet[y + 38 : y + 518, x : x + 480] = panel
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"could not write preview: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    mask_manifest = args.masks / "manifest.json"
    mask_record = json.loads(mask_manifest.read_text(encoding="utf-8"))
    frames = mask_record["frames"]
    raw_manifest = args.raw / "manifest.json"
    raw_record = json.loads(raw_manifest.read_text(encoding="utf-8"))
    if raw_record["frames"] != frames:
        raise ValueError("raw and mask manifests have different frames")
    for name, details in raw_record["outputs"].items():
        if digest(args.raw / name) != details["sha256"]:
            raise ValueError(f"raw image does not match manifest: {name}")
    for view in VIEWS:
        if digest(args.masks / f"{view}.npz") != mask_record["views"][view]["output_sha256"]:
            raise ValueError(f"mask array does not match manifest: {view}")
    declared_raw = mask_record.get("inputs", {}).get("raw_manifest_sha256")
    if declared_raw is None or digest(raw_manifest) != declared_raw:
        raise ValueError("fresh inference must be tied to the exact raw-image manifest")
    data = read_masks(args.masks, frames)
    config = VolumeConfig(spacing_mm=2.0)
    rig = load_rig(args.calibration)
    started = time.perf_counter()
    volume = MaskVolume(rig, config, device=args.device)
    setup_s = time.perf_counter() - started
    cpu = MaskVolume(rig, config, device="cpu") if args.device == "cuda" else volume
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir()
    sources = [
        Path(__file__),
        Path(__file__).with_name("volume.py"),
        Path(__file__).with_name("README.md"),
        Path("src/medaka3d/calib.py"),
    ]
    hashes = {}
    for path in sources:
        target = snapshot / path.name
        shutil.copyfile(path, target)
        hashes[str(path)] = digest(path)
    calibration = {str(p): digest(p) for p in sorted(args.calibration.rglob("*.json"))}
    input_hashes = {
        str(p): digest(p)
        for p in [mask_manifest, raw_manifest, *(args.masks / f"{v}.npz" for v in VIEWS)]
    }
    background = cv2.imread(str(args.raw / "background_top_cam.png"))
    if background is None:
        raise ValueError("top background image is missing")
    summaries = []
    links = []
    for t, frame in enumerate(frames):
        raw = {v: cv2.imread(str(args.raw / f"f{frame:06d}_{v}.png")) for v in VIEWS}
        if any(image is None or image.shape != (1200, 1200, 3) for image in raw.values()):
            raise ValueError(f"missing or malformed raw image at frame {frame}")
        masks, instances, counts = frame_masks(data, t)
        for mode in ("detectron2", "top_contrast_side_detectron2"):
            current = masks.copy()
            if mode != "detectron2":
                current["top_cam"] = top_contrast(raw["top_cam"], background)
            tic = time.perf_counter()
            result = volume.evaluate(current)
            elapsed = time.perf_counter() - tic
            tic = time.perf_counter()
            reference = cpu.evaluate(current)
            cpu_s = time.perf_counter() - tic
            valid = result.known & reference.known
            parity = {
                "max_sample_abs_diff": float(
                    np.max(np.abs(result.sampled[valid] - reference.sampled[valid]), initial=0)
                ),
                "known_disagreement": int(np.count_nonzero(result.known != reference.known)),
                "support_count_disagreement": int(
                    np.count_nonzero(result.support_count != reference.support_count)
                ),
            }
            stem = f"f{frame:06d}_{mode}"
            np.savez_compressed(
                args.out / f"{stem}.npz",
                frame=frame,
                known_count=result.known_count,
                support_count=result.support_count,
                soft_score=result.soft_score,
                strict3=result.strict3,
                two_of_three=result.two_of_three,
            )
            image_path = args.out / f"{stem}.jpg"
            preview(image_path, frame, mode, raw, instances, current, volume, result)
            record = {
                "frame": frame,
                "mode": mode,
                "detector_instance_counts": counts,
                "volume_evaluation_s": elapsed,
                "cpu_reference_evaluation_s": cpu_s,
                "cpu_cuda_parity": parity,
                "components": result.components,
                "strict3_voxels": int(result.strict3.sum()),
                "two_view_voxels": int(result.two_of_three.sum()),
                "known_count_histogram": np.bincount(result.known_count, minlength=4).tolist(),
            }
            summaries.append(record)
            links.append(f'<li><a href="{image_path.name}">{html.escape(stem)}</a></li>')
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ("components", "known_count_histogram")
                    }
                ),
                flush=True,
            )
    if any(digest(path) != h for path, h in hashes.items()):
        raise RuntimeError("source changed during experiment")
    output = {
        "status": "ok",
        "purpose": "mask-volume feasibility; no ID evaluation",
        "frames": frames,
        "config": dataclasses.asdict(config),
        "grid_shape": volume.grid_shape,
        "grid_size": len(volume.xyz),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": torch.__version__,
        "setup_s": setup_s,
        "wall_s_including_cpu_audit_and_previews": time.perf_counter() - started,
        "inputs": input_hashes,
        "calibration": calibration,
        "source_hashes": hashes,
        "top_contrast": {
            "threshold": 12,
            "minimum_component_pixels": 6,
            "color_offset": "median channel residual",
            "learned": False,
        },
        "limitations": [
            "No identity or independent location labels were used or scored.",
            "Stored-frame selection can favor successful historical detections.",
            "Foreground support is not calibrated probability or biological accuracy.",
            "Two-view support tolerates a contradictory third view and creates ghosts.",
            "Two millimetre cells can miss or fragment a thin fish.",
        ],
        "results": summaries,
    }
    (args.out / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Mask volume feasibility</title><style>body{font:17px sans-serif;"
        "max-width:1000px;margin:40px auto;line-height:1.7}</style>"
        "<h1>Refraction-aware 3D support volume from independent masks</h1>"
        "<p>Development diagnostic on 10 frames, 3 views. "
        "Not an evaluation of ID accuracy or detection recall.</p>"
        "<p>Top row: raw frame and input mask (cyan). Middle row: 3-view support (green). "
        "Bottom row: support from 2 or more views (orange). Shown full-frame; "
        "the number of candidates does not mean the number of fish.</p>"
        "<p>top_contrast is the comparison in which only top was replaced with "
        "background subtraction.</p><ul>"
        + "\n".join(links)
        + '</ul><p><a href="summary.json">Numbers and run record</a></p></html>'
    )
    (args.out / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
