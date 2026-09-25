"""Read-only pixel measurements at saved final projections; no ranking or tracking."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
FRAMES = (90141, 90258, 90259)
ARMS = ("sparse_control", "nodewise", "endpoint_priority")
RENDER = WORK / "endpoint_mask_priority_ten_second_visual_comparison_v1"
PHOTOS = WORK / "endpoint_negative_native_photos_v1"
RENDER_SHA = "068d3207473deeac26bdd60fdb5ace6bf15f49ae60624f46f4c1e19fd4ab1a6e"
PHOTO_SHA = "dac90d016b249e45fd63d0030ec0f4b041a1b7f414f46cd93049ba29acf67256"
OUTPUT = WORK / "SAVED_TOP_DARK_SIGNAL_DIAGNOSIS_V1.json"


def capture(path, digest=None):
    path = Path(path)
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if digest is not None and actual != digest:
        raise ValueError(f"changed input: {path}")
    return data, dict(path=str(path), sha256=actual, size_bytes=len(data))


def member(root, seal, name):
    row = seal["outputs"][name]
    path = root / name
    if path != Path(row["path"]):
        raise ValueError("output membership path differs")
    data, ref = capture(path, row["sha256"])
    if len(data) != row["size_bytes"]:
        raise ValueError("output membership size differs")
    return data, ref


def decode(data, expected_pixel_sha):
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (1200, 1200, 3) or bgr.dtype != np.uint8:
        raise ValueError("native lossless BGR required")
    if hashlib.sha256(bgr.tobytes()).hexdigest() != expected_pixel_sha:
        raise ValueError("saved native BGR signature differs")
    return bgr


def bilinear(field, xy):
    xy = np.asarray(xy, dtype=np.float64)
    if not np.isfinite(xy).all() or not ((xy >= 0).all() and (xy <= 1199).all()):
        raise ValueError("nonfinite or out-of-image saved projection")
    return float(map_coordinates(field, xy[::-1, None], order=1, mode="constant")[0])


def run():
    if OUTPUT.exists():
        raise ValueError("fresh diagnostic output required")
    references = []

    def read_json(path, digest=None):
        data, ref = capture(path, digest)
        references.append(ref)
        return json.loads(data)

    render = read_json(RENDER / "render.provenance.json", RENDER_SHA)
    photos = read_json(PHOTOS / "preparation.provenance.json", PHOTO_SHA)
    if render["status"] != "ok" or photos["status"] != "ok":
        raise ValueError("successful saved input producers required")
    data, ref = member(RENDER, render, "projection_evidence.json")
    references.append(ref)
    projected = json.loads(data)
    data, ref = member(PHOTOS, photos, "frame_evidence.json")
    references.append(ref)
    native = json.loads(data)
    assessment = read_json(WORK / "INTRODUCTION_PERIOD_BACKGROUND_VISUAL_ASSESSMENT_V1.json")
    inventory = read_json(
        WORK / "TOP_FOREGROUND_INPUT_SOURCE_INVENTORY_V1.json",
        assessment["binding"]["inventory_sha256"],
    )
    background_ref = assessment["background_images"]["top_cam"]
    data, ref = capture(REPO / background_ref["path"], background_ref["file_sha256"])
    references.append(ref)
    background = decode(data, background_ref["BGR_sha256"])
    # Execute only the frozen, pure foreground function; do not import data readers.
    name = "labs/20260909_three_body/observations.py"
    data, ref = capture(REPO / name, inventory["source_sha256"][name]["sha256"])
    references.append(ref)
    tree = ast.parse(data)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "foreground_contrast"
    )
    namespace = {"np": np, "cv2": cv2}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(REPO / name), "exec"), namespace)
    foreground_contrast = namespace["foreground_contrast"]
    cases = projected["cases"]
    if tuple(row["frame_idx"] for row in cases) != FRAMES:
        raise ValueError("fixed three-frame coverage required")
    records, frame_records = [], []
    yy, xx = np.mgrid[-3:4, -3:4]
    disk = xx * xx + yy * yy <= 9
    for case in cases:
        frame = case["frame_idx"]
        photograph = case["photographs"]["top_cam"]
        if photograph not in native["records"]:
            raise ValueError("projection photograph differs from original native record")
        data, ref = member(PHOTOS, photos, f"native/frame_{frame}_top_cam.png")
        references.append(ref)
        bgr = decode(data, photograph["bgr_signature"]["sha256"])
        difference = bgr.astype(np.float32) - background.astype(np.float32)
        median = np.median(difference, axis=(0, 1))
        contrast = np.max(np.abs(difference - median), axis=-1).astype(np.float64)
        foreground = foreground_contrast(bgr, background)
        if not foreground.any():
            raise ValueError("no foreground: distances unavailable, not zero")
        distance = distance_transform_edt(~foreground)
        gray = bgr.astype(np.float64) @ np.array([0.114, 0.587, 0.299])
        bg_gray = background.astype(np.float64) @ np.array([0.114, 0.587, 0.299])
        frame_records.append(
            dict(
                frame_idx=frame,
                foreground_pixels=int(foreground.sum()),
                median_channel_offset_BGR=median.tolist(),
            )
        )
        if tuple(case["methods"]) != ARMS:
            raise ValueError("fixed three-arm coverage required")
        for arm in ARMS:
            method = case["methods"][arm]
            xy = np.asarray(method["projections"]["top_cam"], dtype=float)
            for body in range(3):
                for node, name in enumerate(("mouth", "chest", "tail")):
                    point = xy[body, node]
                    px, py = np.rint(point).astype(int)
                    rows, cols = py + yy[disk], px + xx[disk]
                    if (
                        np.any(rows < 0)
                        or np.any(rows > 1199)
                        or np.any(cols < 0)
                        or np.any(cols > 1199)
                    ):
                        raise ValueError("diagnostic patch outside image")
                    records.append(
                        dict(
                            frame_idx=frame,
                            arm=arm,
                            raw_slot=body,
                            node=name,
                            projected_xy=point.tolist(),
                            stored_multiview_direct=bool(method["observed"][body][node]),
                            top_source_row=int(method["sources"][body][0][node]),
                            luma_0_255=bilinear(gray, point),
                            background_luma_0_255=bilinear(bg_gray, point),
                            contrast_max_channel_0_255=bilinear(contrast, point),
                            fixed_foreground_distance_px=bilinear(distance, point),
                            nearest_pixel_foreground=bool(foreground[py, px]),
                            disk3px_foreground_fraction=float(foreground[rows, cols].mean()),
                            disk3px_luma_min=float(gray[rows, cols].min()),
                            disk3px_luma_max=float(gray[rows, cols].max()),
                            priority_D2_body_classification=(
                                case["priority_final_probe"]["classification_by_view"]["top_cam"][
                                    body
                                ]
                                if arm == "endpoint_priority"
                                else None
                            ),
                        )
                    )
    _, source = capture(__file__)
    document = dict(
        schema="saved_top_dark_signal_diagnosis_v1",
        status="ok",
        source=source,
        references=references,
        frames=list(FRAMES),
        arms=list(ARMS),
        endpoint_count=len(records),
        frame_measurements=frame_records,
        endpoints=records,
        measurement=dict(
            luma="0.114B + 0.587G + 0.299R, bilinear at saved native UV; no dark threshold",
            foreground="unchanged observations.foreground_contrast: abs median-corrected BGR diff >=12, component area >=6",
            distance="exact lattice EDT to fixed foreground, bilinear at native UV",
            patch="radius3 native-pixel disk around nearest integer UV; descriptive only",
            all_saved_MCT_all_arms_included=True,
        ),
        scope=dict(
            native_top_PNG_decodes=3,
            background_PNG_decodes=1,
            original_video_mask_SL_candidate_checkpoint_GPU_reads=0,
            classifier_ranking_or_tracking_calls=0,
            live_source_changes=0,
            scalar_pixels_are_not_anatomical_or_physical_ID_truth=True,
            fixed_foreground_not_current_detectron2_fitter_input=True,
            development_frames_not_holdout=True,
        ),
    )
    OUTPUT.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                output=str(OUTPUT),
                sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                endpoints=len(records),
            )
        )
    )
    for row in records:
        if row["node"] == "tail":
            print(
                row["frame_idx"],
                row["arm"],
                row["raw_slot"],
                "Y",
                round(row["luma_0_255"], 3),
                "contrast",
                round(row["contrast_max_channel_0_255"], 3),
                "FGdist",
                round(row["fixed_foreground_distance_px"], 3),
                "direct",
                row["stored_multiview_direct"],
            )


if __name__ == "__main__":
    run()
