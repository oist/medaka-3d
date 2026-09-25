"""Nine direct-seek stills for model diagnosis, kept separate from blind review."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from observations import FrameObservation  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, decode_frame_masks, sha256  # noqa: E402
from render_results import COLORS, panel  # noqa: E402
from review_scene import open_sources  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

RUN = REPO / "work/three_body/v1_continuous_one_minute_detectron2"
SEQUENCE = REPO / "work/three_body/inputs/continuous_one_minute_76620-86580"
OUT = REPO / "work/three_body/minute_scene0_model_diagnostic"
FRAMES = [79316, 79318, 79322, 79355, 79370, 79372, 79381, 79405, 79547]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_file(path, expected, inputs):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "input path escaped the workspace")
    require(sha256(path) == expected, f"input checksum mismatch: {path}")
    inputs[str(path)] = expected
    return path


def native_crop(image, center, size=320):
    """Keep the requested native center; explicitly pad if the crop crosses an edge."""
    require(np.isfinite(center).all(), "nonfinite projected crop center")
    left, top = np.floor(center).astype(int) - size // 2
    right, bottom = left + size, top + size
    x0, x1 = max(0, left), min(image.shape[1], right)
    y0, y1 = max(0, top), min(image.shape[0], bottom)
    require(x0 < x1 and y0 < y1, "requested diagnostic crop is outside the image")
    crop = np.full((size, size, 3), 20, np.uint8)
    crop[y0 - top : y1 - top, x0 - left : x1 - left] = image[y0:y1, x0:x1]
    metadata = dict(
        center_uv=np.asarray(center).tolist(),
        requested_box_xyxy=[int(left), int(top), int(right), int(bottom)],
        native_intersection_xyxy=[int(x0), int(y0), int(x1), int(y1)],
        padding_pixels=int(size * size - (x1 - x0) * (y1 - y0)),
        native_scale=1,
    )
    return crop, metadata


def write_png(path, image, record):
    require(cv2.imwrite(str(path), image), "PNG write failed")
    require(np.array_equal(cv2.imread(str(path)), image), "PNG round trip changed pixels")
    record.output(path.name, path, sha256=sha256(path))


def main():
    if OUT.exists():
        raise FileExistsError("diagnostic output exists; no overwrite")
    inputs = {}
    run_summary, run_seal = load(RUN / "summary.json"), load(RUN / "experiment.provenance.json")
    require(
        run_seal["status"] == "ok" and run_summary["technical_completion"],
        "candidate run incomplete",
    )
    for name in ("candidate", "summary"):
        check_file(
            Path(run_seal["outputs"][name]["path"]), run_seal["outputs"][name]["sha256"], inputs
        )
    inputs[str(RUN / "experiment.provenance.json")] = sha256(RUN / "experiment.provenance.json")
    candidate = RUN / "candidate.npz"
    require(
        Path(run_seal["outputs"]["candidate"]["path"]).resolve() == candidate,
        "wrong sealed candidate",
    )
    with np.load(candidate, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    require(len(result) == 19, "candidate schema differs from completed 19-array run")
    require(
        {key: list(value.shape) for key, value in result.items()}
        == run_summary["output_array_shapes"],
        "candidate shape summary mismatch",
    )
    require(
        np.array_equal(result["frame_idx"], np.arange(76620, 86580)),
        "candidate is not the complete native minute",
    )
    require(
        result["observed"].dtype == bool and result["identity_review"].dtype == bool,
        "candidate support/review flags changed",
    )
    manifest_path = SEQUENCE / "manifest.json"
    manifest, staging = load(manifest_path), load(SEQUENCE / "staging.provenance.json")
    require(staging["status"] == "ok", "source sequence incomplete")
    digest = sha256(manifest_path)
    require(
        digest
        == staging["outputs"]["manifest"]["sha256"]
        == run_seal["inputs"]["sequence_manifest"]["sha256"],
        "candidate/source/staging binding mismatch",
    )
    check_file(manifest_path, digest, inputs)
    check_file(RUN / "sequence_manifest.json", digest, inputs)
    inputs[str(SEQUENCE / "staging.provenance.json")] = sha256(SEQUENCE / "staging.provenance.json")
    require(
        manifest["frames"] == result["frame_idx"].tolist(),
        "raw and candidate frame association differs",
    )
    require(
        manifest["fps"] == 166 and manifest["image_shape_hw"] == [1200, 1200],
        "raw native metadata differs",
    )
    require(
        set(manifest["raw"]) == set(manifest["masks"]) == set(VIEWS),
        "source camera identities differ",
    )
    require(str(manifest["raw_codec"]).startswith("FFV1"), "direct diagnostic requires native FFV1")
    videos, masks = {}, {}
    for view in VIEWS:
        for category in ("raw", "masks"):
            item = manifest[category][view]
            require(
                item["sha256"] == run_seal["inputs"][category + "_" + view]["sha256"],
                "candidate image evidence differs",
            )
            require(
                item["sha256"] == staging["outputs"][view + "_" + category]["sha256"],
                "image evidence missing staging seal",
            )
            path = check_file(SEQUENCE / item["path"], item["sha256"], inputs)
            if category == "raw":
                videos[view] = path
            else:
                with np.load(path, allow_pickle=False) as data:
                    masks[view] = {key: data[key] for key in data.files}
                require(
                    np.array_equal(masks[view]["frame_idx"], result["frame_idx"]),
                    "mask frame association differs",
                )
    calibration = RUN / "calibration_snapshot"
    for path in calibration.iterdir():
        matches = [
            entry
            for key, entry in run_seal["inputs"].items()
            if key.startswith("calibration/") and Path(entry["path"]).name == path.name
        ]
        require(len(matches) == 1, "calibration snapshot lacks unique original binding")
        check_file(path, matches[0]["sha256"], inputs)
    rig = load_rig(calibration)
    require(
        all(rig.camera(view).view == view for view in VIEWS), "loaded camera view ordering differs"
    )
    code_paths = [
        Path(__file__),
        LAB / "render_results.py",
        LAB / "review_scene.py",
        LAB / "review_truth.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/calib.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    for path in code_paths:
        digest = sha256(path)
        if path.name in {"calib.py", "types.py"}:
            registered_code = {
                name.replace("\\", "/"): value
                for name, value in run_summary["code_sha256_before"].items()
            }
            require(
                registered_code[path.relative_to(REPO).as_posix()] == digest,
                "projection/type source differs from tracked run",
            )
        inputs[str(path)] = digest
    # Keep a before/after binding to the separate model-blind artifacts; never write there.
    blind = REPO / "work/three_body/minute_reviews/scene0_79236-79488"
    for name in (
        "summary.json",
        "index.html",
        "truth_template.json",
        "review_scene.provenance.json",
    ):
        inputs[str(blind / name)] = sha256(blind / name)
    OUT.mkdir(parents=True, exist_ok=False)
    frame_records = []
    with provenance("diagnose_minute_scene0_raw_and_candidate", OUT, name="diagnostic") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for path in code_paths:
            destination = OUT / "code_snapshot" / path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            record.output("code:" + path.name, destination, sha256=sha256(destination))
        captures = open_sources(videos, 9960, FRAMES[0] - 76620)
        try:
            for frame in FRAMES:
                t = frame - 76620
                images, frame_masks, views = {}, {}, {}
                for view in VIEWS:
                    cap = captures[view]
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) != t:
                        require(cap.set(cv2.CAP_PROP_POS_FRAMES, t), "direct native seek failed")
                    before = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    require(before == t, "native seek landed on another frame")
                    ok, image = cap.read()
                    after = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    require(
                        ok
                        and after == t + 1
                        and image.shape == (1200, 1200, 3)
                        and image.dtype == np.uint8,
                        "native frame decode alignment failed",
                    )
                    digest = array_sha256(image)
                    require(
                        digest == manifest["raw"][view]["frame_bgr_sha256"][t],
                        "native BGR pixels differ from source record",
                    )
                    images[view] = image
                    frame_masks[view] = (
                        decode_frame_masks(masks[view], t)
                        if masks[view]["view_available"][t]
                        else None
                    )
                    if frame_masks[view] is not None:
                        require(
                            array_sha256(frame_masks[view])
                            == masks[view]["frame_mask_sha256"][t].decode(),
                            "decoded D2 mask hash differs",
                        )
                    views[view] = dict(
                        source_clip_offset=t,
                        position_before=before,
                        position_after=after,
                        native_bgr_sha256=digest,
                        d2_instances=0 if frame_masks[view] is None else len(frame_masks[view]),
                    )
                observation = FrameObservation(frame, {}, images, frame_masks, {})
                context = np.concatenate(
                    [panel(observation, result, t, rig, view, width=600) for view in VIEWS], axis=1
                )
                write_png(OUT / f"f{frame}_context.png", context, record)
                for view in VIEWS:
                    pixels = rig.camera(view).project(result["pose_xyz"][t])
                    center = pixels[:2, 1].mean(axis=0)
                    raw_crop, crop_info = native_crop(images[view], center)
                    native_overlay = panel(observation, result, t, rig, view, width=1200)[54:]
                    overlay_crop, overlay_info = native_crop(native_overlay, center)
                    require(crop_info == overlay_info, "raw and overlay crop coordinates differ")
                    views[view].update(crop=crop_info, projected_nodes_uv=pixels.tolist())
                    write_png(OUT / f"f{frame}_{view}_raw_crop.png", raw_crop, record)
                    write_png(OUT / f"f{frame}_{view}_overlay_crop.png", overlay_crop, record)
                    pair = np.full((364, 640, 3), 25, np.uint8)
                    pair[44:, :320], pair[44:, 320:] = raw_crop, overlay_crop
                    cv2.putText(
                        pair,
                        f"{view} native {frame}: raw | model diagnostic",
                        (8, 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (240, 240, 240),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        pair,
                        "320 px crop; filled: observed; hollow: unsupported; ?: review",
                        (8, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.34,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )
                    write_png(OUT / f"f{frame}_{view}_pair.png", pair, record)
                frame_records.append(
                    dict(
                        frame_idx=frame,
                        views=views,
                        pose_xyz=result["pose_xyz"][t].tolist(),
                        observed=result["observed"][t].tolist(),
                        identity_review=result["identity_review"][t].tolist(),
                        sources=result["sources"][t].tolist(),
                        previous_frame_idx=frame - 1,
                        previous_pose_xyz=result["pose_xyz"][t - 1].tolist(),
                        previous_observed=result["observed"][t - 1].tolist(),
                        latent_node_step_mm=np.linalg.norm(
                            result["pose_xyz"][t] - result["pose_xyz"][t - 1], axis=-1
                        ).tolist(),
                        mask_supported=result["mask_supported"][t].tolist(),
                        proposal_kind=str(result["proposal_kind"][t]),
                    )
                )
        finally:
            for cap in captures.values():
                cap.release()
        require(
            all(sha256(path) == digest for path, digest in inputs.items()),
            "input/source changed during diagnostic generation",
        )
        summary = dict(
            scope="post-hoc model diagnostic; not human truth or an accepted identity result",
            source_frame_range=[76620, 86580],
            selected_frames=FRAMES,
            selected_native_images=27,
            native_frame_sha_and_seek_verified=True,
            direct_seeks_only=True,
            gpu_used=False,
            candidate_sha256=inputs[str(candidate)],
            source_manifest_sha256=inputs[str(manifest_path)],
            colors_bgr={f"S{index}": list(color) for index, color in enumerate(COLORS)},
            d2_contour_color_bgr=[255, 255, 0],
            latent_labels="arbitrary seeded slots S0/S1/S2",
            filled_marker_meaning=(
                "saved global observed node flag, not direct evidence in this particular camera"
            ),
            hollow_marker_meaning="node without saved direct support",
            question_mark_meaning="saved identity_review",
            crop_rule=(
                "native 320x320 centered on mean projected S0/S1 pectoral each frame/view; "
                "explicit edge padding"
            ),
            frame_diagnostics=frame_records,
            all_inputs_and_blind_review_unchanged=True,
            accepted_ids_modified=False,
            human_truth_created=False,
            temporal_subsampling_of_tracking=False,
            limitations=[
                "Selected stills are diagnostic; tracking is not recomputed or relabeled.",
                "D2 contours and node support are imperfect evidence, not identity truth.",
                "Crop centers move with model projections; full context is saved for each frame.",
            ],
        )
        path = OUT / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(
            json.dumps(
                dict(selected_frames=len(FRAMES), native_images=27, png_outputs=len(FRAMES) * 10)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
