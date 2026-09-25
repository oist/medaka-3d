"""Fixed6px endpoint classification on all seven saved geometry frames.

All eleven depth-condition cases and all complete legacy/additional bodies are
retained. The two conditions share each decoded frame and its mask fields.
Saved top-scene membership is inspected; no mask gate or tracking change occurs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import classify_body_mask_endpoints as classifier
import numpy as np
import run_nodewise_body_mask_containment as base

proof, require, saved, frozen, prior = base.proof, base.require, base.saved, base.frozen, base.prior
WORK, VIEWS = base.WORK, base.VIEWS
STAGE = "nodewise_body_mask_classification_all_cases_v1"
FRAMES = (82720, 82727, 82728, 82750, 90141, 90258, 90259)
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore")
TOLERANCE_PX = 6.0
BASE_SHA = "07907d5051d099dfe8e5f96c4f909b693b3cfcc206994fbf94420443dcef39a3"
CLASSIFIER_SHA = "a5d87511c59d1071c6b94c205376d4ec613ff1a20d84205026e9123db54e9f8f"
BODY_FIELDS = (
    "sample_xyz",
    "projected_xy",
    "distances_by_body_sample_mask",
    "maximum_distance_by_body_mask_px",
    "best_single_mask",
    "best_single_mask_tie_count",
    "best_single_mask_distances_px",
    "single_mask_max_distance_px",
    "single_mask_mean_distance_px",
    "endpoint_single_mask_max_distance_px",
    "union_distances_by_sample_px",
    "union_max_distance_px",
    "mask_switch_advantage_px",
)
CATEGORIES = (
    "unavailable",
    "uncovered_endpoint",
    "ambiguous_endpoint",
    "same_instance",
    "split_instances",
)


def case_keys():
    return [(arm, frame) for frame in FRAMES for arm in (ARMS if frame < 90000 else ARMS[:1])]


def select_complete(case):
    pool = case["pool"]
    require([b["pool_index"] for b in pool] == list(range(len(pool))), "pool membership differs")
    bodies = []
    for body in pool:
        flags = np.asarray(body["triangulated_nodes"])
        pose = np.asarray(body["pose_xyz"], dtype=np.float64)
        require(flags.shape == (3,) and flags.dtype == bool and pose.shape == (3, 3), "body schema")
        if flags.all():
            require(np.isfinite(pose).all(), "nonfinite complete body")
            bodies.append(body)
    require(len(bodies) == case["summary"]["nodewise_scenes"]["complete_bodies"], "body coverage")
    return bodies


def load_headers(evidence):
    # Reuse the qualified two-frame source/rig/manifest checks, then extend the
    # saved records explicitly. No mutation of the frozen loader's frame scope.
    context = base.load_headers(evidence)
    seal = proof.read_json(base.GEOMETRY / "geometry.provenance.json")
    render_path = saved.member(base.GEOMETRY, seal, "render_payload.json", evidence)
    render = proof.read_json(render_path)
    require(render["schema"] == base.GEOMETRY.name + "_render_payload", "scene payload schema")
    scenes = {(r["arm"], r["frame_idx"]): r for r in render["cases"]}
    require(
        len(render["cases"]) == len(scenes) == 11 and set(scenes) == set(case_keys()), "scene scope"
    )
    groups, frames = {}, {}
    for anchor, target_frames in ((82720, FRAMES[:4]), (90258, FRAMES[4:])):
        reference = context["cases"][anchor]
        document = proof.read_json(Path(reference["visual_evidence"]["path"]))
        records = document["records"]
        records = list(records.values()) if isinstance(records, dict) else records
        chunk = reference["chunk"]
        manifest_path = (
            prior.native.checked_relative(base.MASK_ROOT, chunk["attempt_path"]) / "manifest.json"
        )
        manifest = proof.read_json(manifest_path)  # Already bound by base.load_headers.
        groups[chunk["index"]] = dict(
            chunk=chunk, captured=reference["captured"], frames=target_frames
        )
        for frame in target_frames:
            rows = [r for r in records if r.get("frame_idx", r.get("frame")) == frame]
            require(len(rows) == 1 and set(rows[0]["observations"]) == set(VIEWS), "frame evidence")
            observations = rows[0]["observations"]
            for view in VIEWS:
                ob = observations[view]
                require(
                    type(ob["mask_available"]) is bool and proof.valid_sha(ob["mask_sha256"]),
                    "mask evidence",
                )
                require(
                    manifest["masks"][view]["instances_per_frame"][frame - chunk["start"]]
                    == ob["mask_count"],
                    "frame mask count differs",
                )
            frames[frame] = dict(
                observations=observations, cases={}, visual_evidence=reference["visual_evidence"]
            )
    for arm, frame in case_keys():
        name = f"cases/{arm}_{frame}.json"
        case = proof.read_json(saved.member(base.GEOMETRY, seal, name, evidence))
        require(
            case["frame_idx"] == frame
            and case["arm"] == arm
            and case["schema"] == base.GEOMETRY.name + "_case",
            "geometry case scope",
        )
        require(
            case["lower"] == [0.0, 0.0, -6.0 if arm == ARMS[0] else -12.0]
            and case["upper"] == [192.0, 192.0, 186.0],
            "geometry bounds differ",
        )
        scene_record = scenes[arm, frame]
        require(scene_record["case_json"] == seal["outputs"][name], "scene/case binding differs")
        bodies = select_complete(case)
        by_index = {b["pool_index"]: b for b in bodies}
        top = {}
        for method, key in (("legacy", "legacy_scenes"), ("nodewise", "nodewise_scenes")):
            source = scene_record[key][0] if scene_record[key] else None
            if source is not None:
                indices = source["pool_indices"]
                require(
                    source["rank"] == 0
                    and len(set(indices)) == len(indices) == 3
                    and set(indices) <= set(by_index),
                    "saved top scene membership",
                )
                require(
                    np.array_equal(
                        np.asarray(source["pose_xyz"], dtype=np.float64),
                        np.asarray([by_index[i]["pose_xyz"] for i in indices], dtype=np.float64),
                    ),
                    "saved top scene pose differs",
                )
            top[method] = source
        frames[frame]["cases"][arm] = dict(
            geometry=case, bodies=bodies, top_scenes=top, source_case=seal["outputs"][name]
        )
    context.update(
        all_frames=frames,
        observation_groups=groups,
        scene_payload=seal["outputs"]["render_payload.json"],
    )
    return context


def unique_frame_geometry(cases):
    poses, lookup, mappings = [], {}, {}
    for arm, case in cases.items():
        positions = []
        for body in case["bodies"]:
            pose = np.asarray(body["pose_xyz"], dtype=np.float64)
            key = pose.tobytes()
            if key not in lookup:
                lookup[key] = len(poses)
                poses.append(pose)
            positions.append(lookup[key])
        mappings[arm] = positions
    return np.stack(poses), mappings


def select_measurement(result, positions):
    return {key: value[positions] if key in BODY_FIELDS else value for key, value in result.items()}


def classify_case(bodies, views):
    records = []
    for position, body in enumerate(bodies):
        for view in VIEWS:
            result = views[view]
            distances = result.get("distances_by_body_sample_mask")
            label = classifier.classify_endpoint_distances(
                distances[position, :3].tolist() if result["available"] else None,
                result["mask_indices"].tolist() if result["available"] else None,
                result["available"],
                TOLERANCE_PX,
            )
            records.append(dict(pool_index=body["pool_index"], view=view, **label))
    return records


def top_scene_review(scene, records):
    if scene is None:
        return dict(available=False, reason="no saved complete three-body scene")
    chosen = scene["pool_indices"]
    rows = [r for r in records if r["pool_index"] in chosen]
    require(len(rows) == 9, "top scene must have all nine body-view classifications")
    counts = Counter(r["classification"] for r in rows)
    return dict(
        available=True,
        pool_indices=chosen,
        source_rank=0,
        counts={category: counts[category] for category in CATEGORIES},
        any_split=counts["split_instances"] > 0,
        split_body_views=[r for r in rows if r["classification"] == "split_instances"],
        body_views=rows,
        independently_verified_physical_truth=False,
    )


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    captures, mask_inputs, masks, mask_signatures = {}, {}, {}, {}
    try:
        evidence.add(Path(base.__file__).resolve(), BASE_SHA)
        evidence.add(Path(classifier.__file__).resolve(), CLASSIFIER_SHA)
        require(
            base.bundle.DEFAULT_CONFIG.endpoint_tolerance_px == TOLERANCE_PX, "fixed6px differs"
        )
        context = load_headers(evidence)
        code = base.source_code(context, evidence)
        for path in (
            Path(base.__file__).resolve(),
            Path(classifier.__file__).resolve(),
            Path(__file__).resolve(),
        ):
            code[str(path)] = evidence.add(path)
        rig = base.load_rig(context, evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_code=code,
                frames=FRAMES,
                case_keys=case_keys(),
                geometry_sha256=base.GEOMETRY_SHA,
                scene_payload=context["scene_payload"],
                registered_calibration=context["rig_refs"],
                endpoint_tolerance_px=TOLERANCE_PX,
                sample_labels=base.containment.SAMPLE_LABELS,
                all_complete_legacy_and_additional_bodies=True,
                per_frame_view_mask_fields_generated_once=True,
                exactly_identical_poses_share_distances=True,
                classification_scope="M/C/T only; all9sample distances retained separately",
                top_scene_scope="saved rank0 legacy/nodewise triples; physical truth unknown",
                incomplete_bodies_are_excluded_without_completion=True,
                mask_archive_captures=6,
                target_mask_decodes=21,
                SLEAP_used_for_classification=False,
                no_automatic_gate=True,
                no_tracking_change=True,
                GPU_used=False,
            ),
        )
        frame_data = {}
        for frame, header in context["all_frames"].items():
            poses, mappings = unique_frame_geometry(header["cases"])
            frame_data[frame] = dict(poses=poses, mappings=mappings, views={})
        field_count = 0
        for group in context["observation_groups"].values():
            chunk = group["chunk"]
            for view in VIEWS:
                old_capture = group["captured"][view]
                packed, record = frozen.capture_masks(
                    Path(old_capture["path"]),
                    old_capture["sha256"],
                    np.arange(chunk["start"], chunk["stop"], dtype=np.int64),
                )
                require(record["arrays"] == old_capture["arrays"], "packed signatures differ")
                captures[f"chunk{chunk['index']}_{view}"] = record
                mask_inputs[record["path"]] = record["sha256"]
                for frame in group["frames"]:
                    pos = frame - chunk["start"]
                    mask = prior.prepare_sequence.decode_frame_masks(packed, pos)
                    available = bool(packed["view_available"][pos])
                    ob = context["all_frames"][frame]["observations"][view]
                    digest = prior.prepare_sequence.array_sha256(mask)
                    require(
                        digest
                        == ob["mask_sha256"]
                        == bytes(packed["frame_mask_sha256"][pos]).decode()
                        and available == ob["mask_available"]
                        and len(mask) == ob["mask_count"],
                        "target frame mask differs",
                    )
                    key = f"f{frame}_{view}"
                    mask.flags.writeable = False
                    masks[key] = mask
                    mask_signatures[key] = dict(
                        signature=proof.signature(mask),
                        frame_idx=frame,
                        view=view,
                        available=available,
                        original_mask_sha256=digest,
                    )
                    indices = np.flatnonzero(mask.any(axis=(1, 2)))
                    fields = (
                        base.sparse_instance_body_score.SparseDistanceFields.from_masks(
                            mask[indices]
                        )
                        if available and len(indices)
                        else None
                    )
                    if fields is not None:
                        field_count += 1
                    measured = base.containment.measure_body_mask_containment(
                        frame_data[frame]["poses"],
                        rig.camera(view).project,
                        fields,
                        indices if fields is not None else None,
                    )
                    frame_data[frame]["views"][view] = measured
                    del fields
                del packed
        summaries, legacy_flagged, new_flagged, total = [], [], [], Counter()
        expected_records = 0
        for arm, frame in case_keys():
            case = context["all_frames"][frame]["cases"][arm]
            data = frame_data[frame]
            views = {
                view: select_measurement(data["views"][view], data["mappings"][arm])
                for view in VIEWS
            }
            labels = classify_case(case["bodies"], views)
            expected_records += 3 * len(case["bodies"])
            counts = Counter(row["classification"] for row in labels)
            total.update(counts)
            reviews = {
                kind: top_scene_review(scene, labels) for kind, scene in case["top_scenes"].items()
            }
            for kind, target in (("legacy", legacy_flagged), ("nodewise", new_flagged)):
                if reviews[kind].get("any_split", False):
                    target.append(dict(arm=arm, frame_idx=frame, **reviews[kind]))
            record = out.write(
                f"cases/{arm}_{frame}.json",
                base.json_value(
                    dict(
                        schema=STAGE + "_case",
                        arm=arm,
                        frame_idx=frame,
                        source_case=case["source_case"],
                        bodies=case["bodies"],
                        pool_indices=[b["pool_index"] for b in case["bodies"]],
                        unique_pose_indices=data["mappings"][arm],
                        distances=views,
                        classifications=labels,
                        top_scene_review=reviews,
                        endpoint_tolerance_px=TOLERANCE_PX,
                    )
                ),
            )
            summaries.append(
                dict(
                    arm=arm,
                    frame_idx=frame,
                    complete_bodies=len(case["bodies"]),
                    counts={c: counts[c] for c in CATEGORIES},
                    top_scene_review=reviews,
                    case_json=record,
                )
            )
        require(
            len(captures) == len(mask_inputs) == 6
            and len(masks) == 21
            and sum(total.values()) == expected_records,
            "fixed coverage differs",
        )
        mask_record = out.write("frame_masks.npz", masks)
        with np.load(mask_record["path"], allow_pickle=False) as archive:
            require(
                len(archive.files) == 21 and set(archive.files) == set(masks),
                "mask output membership",
            )
            for key in archive.files:
                require(
                    proof.signature(archive[key]) == mask_signatures[key]["signature"],
                    "mask output bytes",
                )
        out.write("frame_masks_signatures.json", mask_signatures)
        out.write(
            "summary.json",
            base.json_value(
                dict(
                    schema=STAGE,
                    status="ok",
                    cases=summaries,
                    case_count=11,
                    complete_body_conditions=expected_records // 3,
                    classified_body_views=expected_records,
                    counts={c: total[c] for c in CATEGORIES},
                    flagged_legacy_top_scenes=legacy_flagged,
                    flagged_nodewise_top_scenes=new_flagged,
                    unique_pose_count_by_frame={f: len(d["poses"]) for f, d in frame_data.items()},
                    mask_archive_captures=6,
                    target_frame_mask_decodes=21,
                    nonempty_frame_view_fields_created=field_count,
                    captured_mask_records=captures,
                    frame_masks=mask_record,
                    SLEAP_used_for_classification=False,
                    original_video_or_SL_cache_or_track_NPZ_or_checkpoint_reads=0,
                    GPU_used=False,
                    limitations=[
                        "Both depth conditions are counted; duplicate bodies are not independent.",
                        "Saved top scenes were not selected by mask classification.",
                        "Top scenes have no truth labels; inspect flagged legacy images.",
                        "D2 may split or merge fish; split_instances is not physical truth.",
                        "same_instance excludes neither interior leakage nor merged detections.",
                        "No automatic rejection, cost weight, tracking policy or pose is changed.",
                    ],
                )
            ),
        )
        for record in captures.values():
            require(
                frozen.observations.file_stat(Path(record["path"])) == record["file_stat"],
                "mask changed",
            )
        evidence.final_guard()
        out.final_guard()
        receipt = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs={**evidence.paths, **mask_inputs},
            outputs=out.records,
        )
    except BaseException as error:
        receipt = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs={**evidence.paths, **mask_inputs},
            outputs=out.records,
            error=repr(error),
        )
        (output / "classification.provenance.json").write_bytes(proof.json_bytes(receipt))
        raise
    (output / "classification.provenance.json").write_bytes(proof.json_bytes(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
