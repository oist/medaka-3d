"""Classify all final-path bodies in one ten-second, same-horizon comparison.

Three packed mask archives are captured once and each native mask frame decoded
once, shared across all three methods. No SLEAP, raw video, state restoration,
candidate banks, fitter or temporal solver is used. Mask indices are frame-local.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import score_saved_endpoint_mask_priority as base

WORK, REPO, VIEWS = base.WORK, base.REPO, base.VIEWS
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
import chunk_masks  # noqa: E402
import prepare_sequence  # noqa: E402

require, signature = base.require, base.signature
STAGE = "endpoint_mask_final_continuation_evaluation_v1"
FIRST, START, STOP, FPS = 49800, 89640, 91300, 166.0
ARMS = ("sparse_control", "nodewise_unmasked", "endpoint_priority")
ROOTS = (
    WORK / "nodewise_negative_same_horizon_control_v1",
    WORK / "nodewise_body_row_negative_ten_second_continuation_v1",
    WORK / "endpoint_mask_priority_ten_second_continuation_v1",
)
CONTROL_SHA = "3c871a046fbd5fe6db52dfd10e8f0b907e7c34ecbc4a6cde858a9d0aeb9c3f26"
PARENT_SHA = "596b23afe5fb572a66a72b96b0b1f5ba9b3971bfef4dd43cb8ee6b32459e32ad"
PARENT_CHECKPOINT_SHA = "f76a4031f831bdd5b2e0516237242713e981a305c40f2b57966c90fa625a6e70"
CONTROL_CHECKPOINT_SHA = "d1fb9abca83edd772841ac90740454d88370b121f6efc0524bb9187925437bf4"
CHUNK_SHA = "17a9cf8721d1dc4868b7bbc9881e0222e47b6b84a8955ccb0b4b8b29e840cd92"
SOURCE_PINS = {
    "work/three_body/score_saved_endpoint_mask_priority.py": (
        "0f1175ca074661fc495312455c4f2b9f56c2ff99374143b0497dc9784a899015"
    ),
    "work/three_body/prepare_nodewise_negative_control.py": (
        "d4ba13f7a36881e5e9ee585208dc49027577a3d3fbba468a8d01ebd733418793"
    ),
    "work/three_body/run_nodewise_body_row_negative_continuation.py": (
        "4f2d31a383a8d59320ef0c09f8dcddf645dfb20e0849bd2f1dc27398d719669e"
    ),
    "work/three_body/run_endpoint_mask_priority_continuation.py": (
        "6d972c0f34c435f1036eca6cae3fc0563d2853b1a63f3273e89eb3d40b06d819"
    ),
    "labs/20260909_three_body/prepare_sequence.py": (
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8"
    ),
    "labs/20260909_three_body/chunk_masks.py": (
        "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81"
    ),
}
FIELDS = ("frame_idx", "pose_xyz", "identity_review")
CLASSES = (
    "same_instance",
    "split_instances",
    "uncovered_endpoint",
    "ambiguous_endpoint",
    "unavailable",
    "invalid_geometry",
    "invalid_projection",
)
NODE_CLASSES = (
    "unique",
    "uncovered",
    "ambiguous",
    "unavailable",
    "invalid_geometry",
    "invalid_projection",
)


def document(root, seal, name, inputs, guards):
    ref = base.member(root, seal, name)
    return base.read_json(ref["path"], ref["sha256"], inputs, guards)


def load_headers(unmasked_sha, priority_sha, inputs, guards):
    refs, seals, summaries, plans, indices = [], [], [], [], []
    for arm, root, digest in zip(
        ARMS, ROOTS, (CONTROL_SHA, unmasked_sha, priority_sha), strict=True
    ):
        name = "control.provenance.json" if arm == ARMS[0] else "tracking.provenance.json"
        seal = base.successful(root, name, digest, inputs, guards)
        summary = document(root, seal, "summary.json", inputs, guards)
        require(
            summary["schema"] == root.name
            and summary["status"] == "ok"
            and summary["retained_interval"] == [FIRST, STOP]
            and summary["parent_tracking_sha256"] == PARENT_SHA
            and len(summary["candidate_arrays"]) == 21,
            "completed same-horizon candidate required",
        )
        source_name = (
            "prepare_nodewise_negative_control.py"
            if arm == ARMS[0]
            else "run_nodewise_body_row_negative_continuation.py"
            if arm == ARMS[1]
            else "run_endpoint_mask_priority_continuation.py"
        )
        source_path = str(WORK / source_name)
        require(
            seal["inputs"].get(source_path) == SOURCE_PINS["work/three_body/" + source_name],
            "executed producer source differs",
        )
        candidate = base.member(root, seal, "candidate.npz")
        if arm == ARMS[0]:
            require(
                summary["comparison_interval"] == [START, STOP]
                and summary["terminal_horizon"] == STOP
                and summary["control_checkpoint_manifest_sha256"] == CONTROL_CHECKPOINT_SHA
                and summary["original_candidate_116200_used"] is False,
                "Sparse control must end at91300",
            )
            binding = summary["original_binding"]
        else:
            plan = document(root, seal, "plan.json", inputs, guards)
            index = document(root, seal, "input_index.json", inputs, guards)
            require(
                summary["candidate"] == candidate
                and summary["process_interval"] == [START, STOP]
                and summary["native_frames"] == STOP - START
                and summary["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA
                and [
                    summary[k]
                    for k in (
                        "initialize_count",
                        "restore_count",
                        "external_pose_resets",
                        "new_begin_calls",
                        "finish_calls",
                    )
                ]
                == [1, 7, 0, 0, 1],
                "new lifecycle or parent differs",
            )
            require(
                all(
                    summary[k] is True
                    for k in (
                        "all39_roundtrip",
                        "prior_history_prefix_unchanged",
                        "finish_matches_checkpoint_backtrace",
                    )
                ),
                "successful source preservation required",
            )
            require(
                plan["schema"] == root.name
                and plan["process_interval"] == [START, STOP]
                and plan["retained_interval"] == [FIRST, STOP]
                and plan["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA,
                "actual plan scope differs",
            )
            require(
                plan["original_binding"] == index["original_binding"] == binding,
                "shared initial/observation binding differs",
            )
            require(
                summary.get("mask_split_priority_applied", False) is (arm == ARMS[2]),
                "method policy differs",
            )
            plans.append(plan)
            indices.append(index)
        refs.append(
            dict(
                arm=arm,
                seal=dict(path=str(root / name), sha256=digest),
                candidate=candidate,
                arrays=summary["candidate_arrays"],
            )
        )
        seals.append(seal)
        summaries.append(summary)
    require(
        indices[0] == indices[1] and len(indices[0]["prefix_chunks"]) == 25,
        "same consumed25 chunk prefix required",
    )
    chunk = indices[0]["prefix_chunks"][24]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (24, START, STOP)
        and chunk["manifest_sha256"] == CHUNK_SHA,
        "exact source chunk24 required",
    )
    manifest_ref = chunk["output_hashes"]["manifest"]
    require(manifest_ref["sha256"] == CHUNK_SHA, "chunk manifest reference differs")
    require(
        all(seal["inputs"].get(manifest_ref["path"]) == CHUNK_SHA for seal in seals[1:]),
        "both methods must consume the exact chunk manifest",
    )
    manifest = base.read_json(manifest_ref["path"], CHUNK_SHA, inputs, guards)
    require(
        manifest["frames"] == list(range(START, STOP))
        and manifest["source_interval_half_open"] == [START, STOP]
        and manifest["fps"] == FPS
        and manifest["image_shape_hw"] == [1200, 1200],
        "native mask manifest scope differs",
    )
    masks = {}
    for view in VIEWS:
        ref = chunk["output_hashes"][view + "_masks"]
        row = manifest["masks"][view]
        require(
            all(seal["inputs"].get(ref["path"]) == ref["sha256"] for seal in seals[1:]),
            "both methods must consume the same mask archive",
        )
        require(
            row["sha256"] == ref["sha256"] and Path(ref["path"]).stat().st_size == row["bytes"],
            "mask source size/SHA differs",
        )
        masks[view] = dict(**ref, size_bytes=row["bytes"])
    rig_refs = {
        p: d
        for p, d in binding["inputs"]["seed"]["SL_and_calibration_sha256"].items()
        if Path(p).suffix == ".json"
    }
    require(
        len(rig_refs) == 4
        and all(seal["inputs"].get(p) == d for seal in seals[1:] for p, d in rig_refs.items()),
        "shared four-file calibration required",
    )
    return dict(
        candidate_references=refs,
        mask_references=masks,
        rig_refs=rig_refs,
        source_chunk=chunk,
        original_binding=binding,
    )


def load_candidate(reference, inputs, guards):
    ref = reference["candidate"]
    payload = base.capture(ref["path"], ref["sha256"], inputs, guards)
    require(len(payload) == ref["size_bytes"], "candidate archive size differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == 21 and set(archive.files) == set(reference["arrays"]),
            "candidate21 membership differs",
        )
        selected = {key: archive[key] for key in FIELDS}
    for key, value in selected.items():
        require(
            signature(value) == reference["arrays"][key], "candidate selected signature differs"
        )
    return candidate_rows(selected)


def candidate_rows(selected):
    frames = selected["frame_idx"]
    require(
        frames.dtype == np.int64 and np.array_equal(frames, np.arange(FIRST, STOP)),
        "native retained frame order differs",
    )
    require(
        selected["pose_xyz"].dtype == np.float64
        and selected["pose_xyz"].shape == (STOP - FIRST, 3, 3, 3)
        and selected["identity_review"].dtype == bool
        and selected["identity_review"].shape == (STOP - FIRST, 3),
        "three candidate fields differ",
    )
    result = {k: v[START - FIRST : STOP - FIRST].copy() for k, v in selected.items()}
    for value in result.values():
        value.flags.writeable = False
    return result


def capture_packed(reference, inputs, guards):
    payload = base.capture(reference["path"], reference["sha256"], inputs, guards)
    require(len(payload) == reference["size_bytes"], "mask archive size differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(
            len(archive.files) == 13 and set(archive.files) == set(chunk_masks.SCHEMA_DTYPES),
            "packed13 schema differs",
        )
        packed = {k: archive[k] for k in archive.files}
    validate_packed_structure(packed)
    for value in packed.values():
        value.flags.writeable = False
    return packed


def validate_packed_structure(data):
    require(
        set(data) == set(chunk_masks.SCHEMA_DTYPES)
        and all(data[k].dtype == dtype for k, dtype in chunk_masks.SCHEMA_DTYPES.items()),
        "packed13 dtypes differ",
    )
    require(
        np.array_equal(data["frame_idx"], np.arange(START, STOP))
        and np.array_equal(data["image_shape_hw"], [1200, 1200]),
        "packed native frame/shape differs",
    )
    count, frames = len(data["mask_boxes"]), STOP - START
    shapes = dict(
        frame_offsets=(frames + 1,),
        mask_boxes=(count, 4),
        mask_byte_offsets=(count + 1,),
        scores=(count,),
        pred_boxes=(count, 4),
        classes=(count,),
        instance_id=(count,),
        source_track_id=(count,),
        view_available=(frames,),
        frame_mask_sha256=(frames,),
    )
    require(
        all(data[k].shape == shape for k, shape in shapes.items()) and data["mask_bits"].ndim == 1,
        "packed member shapes differ",
    )
    for key, terminal in (("frame_offsets", count), ("mask_byte_offsets", len(data["mask_bits"]))):
        offsets = data[key]
        require(
            offsets[0] == 0 and offsets[-1] == terminal and np.all(np.diff(offsets) >= 0),
            "packed offset range differs",
        )
    require(np.all(data["source_track_id"] == -1), "frame-local detector source IDs required")


def decoded_frame(packed, native):
    offset = native - START
    require(
        0 <= offset < STOP - START and packed["frame_idx"][offset] == native,
        "native mask offset differs",
    )
    lo, hi = map(int, packed["frame_offsets"][offset : offset + 2])
    require(
        np.array_equal(packed["instance_id"][lo:hi], np.arange(hi - lo)),
        "within-frame mask order differs",
    )
    available = bool(packed["view_available"][offset])
    require(available or lo == hi, "unavailable frame contains instances")
    masks = prepare_sequence.decode_frame_masks(packed, offset)
    require(
        hashlib.sha256(masks.tobytes()).hexdigest()
        == bytes(packed["frame_mask_sha256"][offset]).decode("ascii"),
        "decoded native mask SHA differs",
    )
    return masks, available


def run_intervals(flags, frames):
    require(
        flags.shape == frames.shape and flags.dtype == bool and np.all(np.diff(frames) == 1),
        "continuous frame flags required",
    )
    boundaries = np.flatnonzero(np.diff(np.r_[False, flags, False])).reshape(-1, 2)
    return [
        dict(
            start_native=int(frames[a]),
            stop_native_exclusive=int(frames[b - 1]) + 1,
            native_frames=int(b - a),
            duration_seconds=(b - a) / FPS,
        )
        for a, b in boundaries
    ]


def node_classes(scored):
    counts = scored["nearby_mask_count_by_node"]
    labels = np.full(counts.shape, "unavailable", dtype="<U24")
    labels[counts == 0] = "uncovered"
    labels[counts == 1] = "unique"
    labels[counts > 1] = "ambiguous"
    for reason in ("invalid_geometry", "invalid_projection"):
        labels[scored["classification_by_body"] == reason] = reason
    return labels


def evaluate_frames(candidates, packed, rig):
    frames = candidates[0]["frame_idx"]
    require(
        len(candidates) == 3 and all(np.array_equal(c["frame_idx"], frames) for c in candidates),
        "same three-arm native frames required",
    )
    length = len(frames)
    arrays = dict(
        frame_idx=frames.copy(),
        identity_review=np.stack([c["identity_review"] for c in candidates], axis=1),
        classification=np.empty((length, 3, 3, 3), dtype="<U24"),
        node_classification=np.empty((length, 3, 3, 3, 3), dtype="<U24"),
        nearby_mask_count=np.empty((length, 3, 3, 3, 3), np.int64),
        unique_mask_index=np.empty((length, 3, 3, 3, 3), np.int64),
        split_by_body_view=np.empty((length, 3, 3, 3), bool),
        unknown_by_body_view=np.empty((length, 3, 3, 3), bool),
        view_available=np.empty((length, 3), bool),
        original_mask_count=np.empty((length, 3), np.int64),
    )
    for position, native in enumerate(frames):
        poses = np.stack([c["pose_xyz"][position] for c in candidates])
        for view_index, view in enumerate(VIEWS):
            masks, available = decoded_frame(packed[view], int(native))
            observation = base.scorer.prepare_endpoint_mask_observation(
                masks if available else None
            )
            scored = base.scorer.score_endpoint_mask_conflicts(
                poses, rig.camera(view).project, observation
            )
            arrays["view_available"][position, view_index] = available
            arrays["original_mask_count"][position, view_index] = len(masks)
            for target, key in (
                ("classification", "classification_by_body"),
                ("split_by_body_view", "split_by_body"),
                ("unknown_by_body_view", "unknown_by_body"),
                ("nearby_mask_count", "nearby_mask_count_by_node"),
                ("unique_mask_index", "unique_mask_by_node"),
            ):
                arrays[target][position, :, :, view_index] = scored[key]
            arrays["node_classification"][position, :, :, view_index] = node_classes(scored)
            del masks, observation, scored
    arrays["split_any_view_by_body"] = arrays["split_by_body_view"].any(axis=3)
    arrays["unknown_any_view_by_body"] = arrays["unknown_by_body_view"].any(axis=3)
    arrays["scene_has_split"] = arrays["split_any_view_by_body"].any(axis=2)
    arrays["scene_has_unknown"] = arrays["unknown_any_view_by_body"].any(axis=2)
    return arrays


def summarize(arrays):
    rows = []
    for arm_index, arm in enumerate(ARMS):
        split, unknown = (
            arrays["split_any_view_by_body"][:, arm_index],
            arrays["unknown_any_view_by_body"][:, arm_index],
        )
        review = arrays["identity_review"][:, arm_index]
        row = dict(
            arm=arm,
            frames=len(split),
            body_classification_counts={
                v: {
                    k: int(np.sum(arrays["classification"][:, arm_index, :, j] == k))
                    for k in CLASSES
                }
                for j, v in enumerate(VIEWS)
            },
            endpoint_classification_counts={
                v: {
                    k: int(np.sum(arrays["node_classification"][:, arm_index, :, j] == k))
                    for k in NODE_CLASSES
                }
                for j, v in enumerate(VIEWS)
            },
            split_body_frames=int(split.sum()),
            unknown_body_frames=int(unknown.sum()),
            split_scene_frames=int(split.any(axis=1).sum()),
            unknown_scene_frames=int(unknown.any(axis=1).sum()),
            split_and_unknown_scene_frames=int((split.any(axis=1) & unknown.any(axis=1)).sum()),
            all_three_bodies_all_views_unique_same_instance_frames=int(
                np.all(arrays["classification"][:, arm_index] == "same_instance", axis=(1, 2)).sum()
            ),
            split_body_reviewed=int((split & review).sum()),
            split_body_missing_review=int((split & ~review).sum()),
            scene_split_runs=run_intervals(split.any(axis=1), arrays["frame_idx"]),
            scene_unknown_runs=run_intervals(unknown.any(axis=1), arrays["frame_idx"]),
            raw_slot_split_runs={
                str(i): run_intervals(split[:, i], arrays["frame_idx"]) for i in range(3)
            },
            raw_slot_unknown_runs={
                str(i): run_intervals(unknown[:, i], arrays["frame_idx"]) for i in range(3)
            },
        )
        row["endpoint_classification_counts_by_node"] = {
            view: {
                node: {
                    category: int(
                        np.sum(
                            arrays["node_classification"][:, arm_index, :, j, node_index]
                            == category
                        )
                    )
                    for category in NODE_CLASSES
                }
                for node_index, node in enumerate(("M", "C", "T"))
            }
            for j, view in enumerate(VIEWS)
        }
        row["longest_scene_split_run_frames"] = max(
            (r["native_frames"] for r in row["scene_split_runs"]), default=0
        )
        row["longest_scene_unknown_run_frames"] = max(
            (r["native_frames"] for r in row["scene_unknown_runs"]), default=0
        )
        rows.append(row)
    require(
        rows[2]["split_body_missing_review"] == 0,
        "final priority split body lacks mandatory identity review",
    )
    return rows


def run(output, unmasked_sha, priority_sha):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and output != WORK, "fresh workspace output required")
    output.mkdir(parents=True, exist_ok=False)
    inputs, guards, outputs = {}, {}, {}
    started = datetime.now(UTC).isoformat()

    def write(name, value):
        path = output / name
        if path.suffix == ".npz":
            with path.open("xb") as stream:
                np.savez_compressed(stream, **value)
        else:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(
                    base.json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False
                )
                stream.write("\n")
        payload = path.read_bytes()
        outputs[name] = dict(
            path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
        )
        return outputs[name]

    try:
        for relative, digest in SOURCE_PINS.items():
            base.capture(REPO / relative, digest, inputs, guards)
        base.source_code(inputs, guards)
        context = load_headers(unmasked_sha, priority_sha, inputs, guards)
        rig = base.load_rig(context, inputs, guards)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                interval=[START, STOP],
                fps=FPS,
                arms=ARMS,
                views=VIEWS,
                candidate_fields=FIELDS,
                candidate_axes=["native", "arm", "raw_body_slot", "node", "xyz"],
                classification_axes=["native", "arm", "raw_body_slot", "view"],
                endpoint_axes=["native", "arm", "raw_body_slot", "view", "node"],
                body_classes=CLASSES,
                node_classes=NODE_CLASSES,
                endpoint_tolerance_px=6.0,
                no_support_filter=True,
                no_ID_assignment=True,
                candidate_banks_used=False,
                final_backtrace_only=True,
                mask_archives_captured_once=3,
                intended_frame_mask_decodes=(STOP - START) * 3,
                physical_ID_truth=False,
                **context,
            ),
        )
        candidates = [load_candidate(r, inputs, guards) for r in context["candidate_references"]]
        packed = {v: capture_packed(context["mask_references"][v], inputs, guards) for v in VIEWS}
        packed_before = {
            v: {k: signature(a) for k, a in data.items()} for v, data in packed.items()
        }
        result = evaluate_frames(candidates, packed, rig)
        rows = summarize(result)
        require(
            all(
                signature(a) == packed_before[v][k]
                for v, data in packed.items()
                for k, a in data.items()
            ),
            "packed arrays changed during streaming",
        )
        write("classifications.npz", result)
        write("array_signatures.json", {k: signature(v) for k, v in result.items()})
        with np.load(output / "classifications.npz", allow_pickle=False) as archive:
            require(
                set(archive.files) == set(result)
                and len(archive.files) == len(result)
                and all(signature(archive[k]) == signature(v) for k, v in result.items()),
                "output roundtrip differs",
            )
        write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                interval=[START, STOP],
                frames=STOP - START,
                arms=rows,
                all_priority_final_split_bodies_have_review=True,
                mask_archive_captures=3,
                native_mask_decodes=(STOP - START) * 3,
                support_or_SL_or_video_or_checkpoint_reads=0,
                finalposes_not_current_beam_diagnostics=True,
                mask_indices_are_local_observations=True,
                unknown_is_not_success=True,
                raw_slot_run_continuity_is_not_biological_identity=True,
                final_split_review_OR_checked_only_no_preOR_reconstruction=True,
                physical_ID_truth=False,
            ),
        )
        require(
            all(base.stamp(Path(p)) == before for p, before in guards.items()),
            "final input stat differs",
        )
        write(
            "evaluation.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
                inputs=inputs,
                outputs=dict(outputs),
                input_guard="SHA at capture; final stat, plus packed arrays byte signatures",
            ),
        )
    except BaseException as exc:
        write(
            "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
                inputs=inputs,
                outputs=dict(outputs),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unmasked-sha", required=True)
    parser.add_argument("--priority-sha", required=True)
    args = parser.parse_args()
    run(args.output, args.unmasked_sha, args.priority_sha)
