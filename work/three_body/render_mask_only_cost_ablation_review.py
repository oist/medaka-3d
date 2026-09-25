"""Three fixed old/new candidate winners on nine bound lossy review JPEGs.

Analytical overlays only: no video, masks, SLEAP, checkpoints or fitting. The
saved candidate banks lack direct-support/review flags; none are inferred here.
S0/S1/S2 are arbitrary slots. Lower candidate cost is not physical correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
sys.path.insert(0, str(REPO / "src"))
from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

SOURCE = WORK / "mask_only_saved_transition_cost_ablation_v2"
PACK = WORK / "fast_tail_transition_raw_review_pack_v1"
STAGE = "mask_only_cost_ablation_raw_review_v1"
SOURCE_SHA = "6b8ca822e509d589958136741f70915d9378fba67e88d265e587a361ac37e5c7"
PACK_SHA = "8622bd5655ecc34058c5e3a7a2fe64e7ff357a3d9eb508cfca4e258405fb0c7f"
CASES = ((90141, "side1_cam", 7, 7), (90258, "side2_cam", 7, 4), (90259, "side2_cam", 0, 3))
CALIB_SHA = "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274"
TYPES_SHA = "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a"
PROJECTION_ATOL_PX = 1e-9
COLORS_BGR = ((20, 145, 255), (255, 190, 0), (200, 40, 230))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=digest(array.tobytes()))


def capture(path, expected, evidence):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    actual = digest(data)
    require(expected is None or actual == expected, "input SHA differs: " + str(path))
    evidence[str(path)] = dict(sha256=actual, size_bytes=len(data))
    return data


def member(root, seal, name, evidence):
    path = (root / name).resolve()
    records = [v for v in seal["outputs"].values() if Path(v["path"]).resolve() == path]
    require(len(records) == 1, "unique sealed output required: " + name)
    return capture(path, records[0]["sha256"], evidence)


def get_array(archive, descriptor):
    require(set(descriptor) == {"__array__", "signature"}, "array descriptor differs")
    array = archive[descriptor["__array__"]]
    require(signature(array) == descriptor["signature"], "saved array signature differs")
    array.flags.writeable = False
    return array


def select_winners(costs, rankings, old, new):
    for name, values in costs.items():
        require(values.ndim == 1 and np.isfinite(values).all(), "finite bank costs required")
        order = np.argsort(values, kind="stable").tolist()
        require(order == rankings[name]["stable_candidate_order"], "saved ranking differs")
        expected = old if name.startswith("old_") else new
        require(order[0] == expected, "fixed winner differs")
    return np.asarray([old, new], dtype=np.int64)


def projected_pairs(rig, poses, scored_view, saved_projection):
    require(
        poses.shape == (2, 3, 3, 3) and np.isfinite(poses).all(),
        "two finite whole-body poses required",
    )
    result = {view: rig.camera(view).project(poses) for view in VIEWS}
    require(
        all(v.shape == (2, 3, 3, 2) and np.isfinite(v).all() for v in result.values()),
        "finite all-view projections required",
    )
    error = float(np.max(np.abs(result[scored_view] - saved_projection)))
    require(error <= PROJECTION_ATOL_PX, "projection differs from saved score endpoints")
    return result, error


def overlay(image, coordinates):
    """Draw only on a copy; no support or identity confidence is encoded."""
    canvas = image.copy()
    for fish, color in enumerate(COLORS_BGR):
        points = np.rint(coordinates[fish]).astype(np.int32)
        cv2.polylines(canvas, [points], False, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        for node, point in enumerate(points):
            xy = tuple(point)
            cv2.circle(canvas, xy, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, xy, 3, color, -1, cv2.LINE_AA)
            label = f"S{fish}:{'MCT'[node]}"
            anchor = (int(point[0]) + 7, int(point[1]) - 7)
            cv2.putText(
                canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )
    return canvas


def labelled(image, title):
    result = np.zeros((image.shape[0] + 48, image.shape[1], 3), np.uint8)
    result[48:] = image
    cv2.putText(
        result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return result


def json_write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def execute(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh output inside work/three_body required",
    )
    output.mkdir()
    evidence = {}
    try:
        capture(Path(__file__), None, evidence)
        capture(REPO / "src/medaka3d/calib.py", CALIB_SHA, evidence)
        capture(REPO / "src/medaka3d/types.py", TYPES_SHA, evidence)
        seal = json.loads(capture(SOURCE / "scoring.provenance.json", SOURCE_SHA, evidence))
        require(
            seal["status"] == "ok"
            and seal["stage"] == "mask_only_saved_transition_cost_ablation_v1",
            "successful ablation required",
        )
        plan = json.loads(member(SOURCE, seal, "plan.json", evidence))
        summary = json.loads(member(SOURCE, seal, "summary.json", evidence))
        pack = json.loads(capture(PACK / "review.provenance.json", PACK_SHA, evidence))
        require(
            pack["status"] == "ok" and pack["stage"] == "fast_tail_transition_raw_review_pack_v1",
            "successful raw-only review required",
        )
        raw_summary = json.loads(member(PACK, pack, "summary.json", evidence))
        require(
            raw_summary["selection_render_sha256"] == plan["raw_source_evidence_sha256"]
            and raw_summary["model_outputs_shown"] is False,
            "raw source selection differs",
        )
        require(
            raw_summary["displayed_frames"] == list(range(90040, 90400))
            and raw_summary["source_fps"] == 166
            and raw_summary["source_frame_range"] == [89640, 91300],
            "review native order differs",
        )
        require(
            plan["actual_temporal_selection"] is False
            and plan["physical_ID_truth"] is False
            and plan["thresholds_retuned"] is False,
            "scientific scope differs",
        )
        calibration = {
            Path(k): v
            for k, v in seal["inputs"].items()
            if "calibration" in k and k.endswith(".json")
        }
        require(
            len(calibration) == 4 and len({p.parent for p in calibration}) == 1,
            "exactly four bound calibration JSONs required",
        )
        calib_root = output / "calibration_copy"
        calib_root.mkdir()
        for path, expected in calibration.items():
            with (calib_root / path.name).open("xb") as stream:
                stream.write(capture(path, expected, evidence))
        rig = load_rig(calib_root)
        records = []
        for frame, view, old, new in CASES:
            record = next(
                v for v in summary["records"] if v["frame"] == frame and v["view"] == view
            )
            stem = f"cases/f{frame}_{view}"
            document = json.loads(member(SOURCE, seal, stem + ".json", evidence))
            tree = document["tree"]
            require(
                tree["frame"] == frame
                and tree["view"] == view
                and tree["candidate_bank_or_parents_modified"] is False,
                "saved case scope differs",
            )
            require(
                tree["rankings"] == record["rankings"]
                and document["archive"] == record["result_archive"],
                "case and summary references differ",
            )
            data = member(SOURCE, seal, stem + ".npz", evidence)
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                bank, change = tree["input_candidate_bank"], tree["candidate_cost_ablation"]
                cost_names = (
                    "old_local_cost",
                    "new_local_cost",
                    "old_fixed_parent_path_cost",
                    "new_fixed_parent_path_cost",
                )
                costs = {k: get_array(archive, change[k]) for k in cost_names}
                selected = select_winners(costs, tree["rankings"], old, new)
                require(bank["frame_idx"] == frame, "bank native frame differs")
                poses = get_array(archive, bank["candidate_pose_xyz"])[selected].copy()
                saved = get_array(archive, tree["new_score"]["projected_samples"])[selected, :, :3]
                parents = get_array(archive, bank["candidate_parent"])[selected]
                plans = get_array(archive, bank["candidate_plan"])[selected]
                assignments = get_array(archive, tree["new_score"]["best_assignment"])[selected]
                distances = get_array(
                    archive, tree["new_score"]["distances_by_body_sample_instance"]
                )[selected, :, :3]
                endpoint_distance = np.stack(
                    [distances[a, np.arange(3), :, assignments[a]] for a in range(2)]
                )
            uv, projection_error = projected_pairs(rig, poses, view, saved)
            full_rows = [[], []]
            image_records = {}
            for camera in VIEWS:
                name = f"frames/f{frame}_{camera}.jpg"
                encoded = member(PACK, pack, name, evidence)
                image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                require(
                    image is not None
                    and image.shape == (1200, 1200, 3)
                    and image.dtype == np.uint8,
                    "1200-square review JPEG required",
                )
                bounds = uv[camera].reshape(-1, 2)
                lo = np.maximum(np.floor(bounds.min(axis=0)).astype(int) - 80, 0)
                hi = np.minimum(np.ceil(bounds.max(axis=0)).astype(int) + 81, [1200, 1200])
                require(np.all(hi > lo), "empty shared display crop")
                crops = []
                for row, label in enumerate(("OLD", "NEW")):
                    painted = overlay(image, uv[camera][row])
                    title = (
                        f"{label} c{selected[row]} | slots not IDs | lossy JPEG | {frame} {camera}"
                    )
                    tile = labelled(painted, title)
                    filename = f"f{frame}_{camera}_{label.lower()}.png"
                    require(
                        cv2.imwrite(str(output / filename), tile), "native display output failed"
                    )
                    full_rows[row].append(tile)
                    crops.append(labelled(painted[lo[1] : hi[1], lo[0] : hi[0]], title))
                crop_name = f"f{frame}_{camera}_old_new_crop.png"
                require(
                    cv2.imwrite(str(output / crop_name), np.concatenate(crops, axis=0)),
                    "crop output failed",
                )
                image_records[camera] = dict(
                    encoded_JPEG_sha256=digest(encoded),
                    source_path=str(PACK / name),
                    source_codec="lossy review JPEG",
                    native_pixel_identity_claimed=False,
                    crop_xyxy=[*lo.tolist(), *hi.tolist()],
                )
            panel = np.concatenate([np.concatenate(row, axis=1) for row in full_rows], axis=0)
            require(
                cv2.imwrite(str(output / f"f{frame}_all_views.png"), panel),
                "all-view panel output failed",
            )
            records.append(
                dict(
                    frame_idx=frame,
                    scored_view=view,
                    selected_candidate_index=selected.tolist(),
                    parents=parents.tolist(),
                    candidate_kind=[bank["candidate_kind"][i] for i in selected],
                    candidate_pose_xyz=poses.tolist(),
                    candidate_plan=plans.tolist(),
                    plan_is_not_verified_support=True,
                    saved_support_or_review_flags_available=False,
                    costs={k: v[selected].tolist() for k, v in costs.items()},
                    new_score_assigned_endpoint_distance_px=endpoint_distance.tolist(),
                    projections={k: v.tolist() for k, v in uv.items()},
                    projection_vs_saved_score_max_abs_px=projection_error,
                    images=image_records,
                    pose_or_review_modified=False,
                )
            )
        json_write(
            output / "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                records=records,
                scored_source_sha256=SOURCE_SHA,
                raw_pack_sha256=PACK_SHA,
                projection_tolerance_px=PROJECTION_ATOL_PX,
                saved_case_bank_decodes=3,
                bound_calibration_JSON_files=4,
                raw_review_JPEG_decodes=9,
                original_video_mask_SLEAP_39_GPU_reads=0,
                scientific_fit_or_proposal_calls=0,
                physical_ID_truth=False,
                direct_support_inferred=False,
                scope=(
                    "Old/new cheapest candidates of fixed saved pools, not actual temporal "
                    "selections. Anonymous slots; all nodes shown without direct/latent "
                    "classification. JPEG hashes bind encoded review assets, "
                    "not lossless native pixels."
                ),
            ),
        )
        outputs = {
            p.relative_to(output).as_posix(): dict(
                path=str(p), sha256=digest(p.read_bytes()), size_bytes=p.stat().st_size
            )
            for p in output.rglob("*")
            if p.is_file()
        }
        json_write(
            output / "render.provenance.json",
            dict(stage=STAGE, status="ok", inputs=evidence, outputs=outputs),
        )
    except Exception as exc:
        json_write(
            output / "failure.json",
            dict(status="failed", error_type=type(exc).__name__, error=str(exc)),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.output)
    print(json.dumps(dict(status="ok", output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
