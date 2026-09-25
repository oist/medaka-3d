"""Independent D2 distances for every complete saved body at two fixed frames.

No new tracking, candidate generation, mask inference, SLEAP cache, video or
checkpoint is used. SLEAP row-to-mask compatibility is a separate diagnostic;
it never removes masks from the independent whole-body measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import body_mask_containment as containment
import compare_partial_body_saved_banks as saved
import instance_body_score
import numpy as np
import partial_body_bundle_score as bundle
import sparse_instance_body_score

proof, require, frozen, prior = saved.proof, saved.require, saved.frozen, saved.prior
WORK, REPO, VIEWS = saved.WORK, saved.REPO, saved.VIEWS
STAGE = "nodewise_body_mask_containment_v1"
GEOMETRY = WORK / "nodewise_body_row_geometry_v1"
MASK_ROOT = WORK / "inputs/long_source_masks_49800-348600_v1"
GEOMETRY_SHA = "c8eab931e843ce15c7646104b6c8346b33f7430edca81411872a9e9ab5c86cff"
GEOMETRY_SOURCE_SHA = "9e843f089d188b660dbf59776c073136519dd665bef6dbf4a27b9473e24ebbcd"
ARM = "sparse_instance_rescore"
FRAMES = (82720, 90258)
FOCUS = {82720: (18, 19), 90258: (9,)}
ADDITIONAL_SOURCE_PINS = {
    "body_mask_containment.py": "752151fa4cf064337c21d513ddfa064ad60a8f354bcb2a4624944219a746a715",
    "instance_body_score.py": "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87",
    "sparse_instance_body_score.py": (
        "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
    ),
}


def json_value(value):
    """Preserve missing observations as null and infinity with an explicit sign."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else {"nonfinite": "+inf" if value > 0 else "-inf"}
    return value


def complete_bodies(case):
    pool = case["pool"]
    require([b["pool_index"] for b in pool] == list(range(len(pool))), "pool order differs")
    selected = []
    for body in pool:
        flags = np.asarray(body["triangulated_nodes"])
        pose = np.asarray(body["pose_xyz"], dtype=np.float64)
        require(flags.shape == (3,) and flags.dtype == np.dtype(bool), "node flags differ")
        require(pose.shape == (3, 3), "body geometry shape differs")
        if flags.all():
            require(np.isfinite(pose).all(), "complete body has nonfinite geometry")
            selected.append(body)
    require(selected, "at least one complete body is required")
    require(
        len(selected) == case["summary"]["nodewise_scenes"]["complete_bodies"],
        "complete-body coverage differs from saved geometry",
    )
    require(set(FOCUS[case["frame_idx"]]) <= {b["pool_index"] for b in selected}, "focus missing")
    return selected


def load_headers(evidence):
    """Only small saved metadata and source bindings; no scientific archives."""
    seal = saved.read(GEOMETRY / "geometry.provenance.json", GEOMETRY_SHA, evidence)
    require(seal["status"] == "ok" and seal["stage"] == GEOMETRY.name, "geometry must succeed")
    saved.edge(seal, WORK / "run_nodewise_body_row_geometry.py", GEOMETRY_SOURCE_SHA)
    plan = proof.read_json(saved.member(GEOMETRY, seal, "plan.json", evidence))
    cases = {}
    shared_rig = None
    specs = (
        (82720, 19, saved.VISUAL_ROOT, saved.VISUAL_SHA, saved.EVIDENCE_SHA, saved.CHUNK_SHA),
        (
            90258,
            24,
            saved.NEGATIVE_ROOT,
            saved.NEGATIVE_SHA,
            saved.NEGATIVE_EVIDENCE_SHA,
            saved.NEGATIVE_CHUNK_SHA,
        ),
    )
    for frame, chunk_index, root, digest, evidence_sha, chunk_sha in specs:
        visual = saved.read(root / "render.provenance.json", digest, evidence)
        # The old v2 directory retains its producer's v1 stage name.
        stage = "depth_identity_transition_raw_review_v1" if chunk_index == 19 else frozen.STAGE
        require(visual["status"] == "ok" and visual["stage"] == stage, "visual must succeed")
        evidence_path = saved.member(root, visual, "source_and_frame_evidence.json", evidence)
        require(evidence.paths[str(evidence_path)] == evidence_sha, "visual evidence differs")
        saved.edge(seal, evidence_path, evidence_sha)
        doc = proof.read_json(evidence_path)
        require(doc["status"] == "ok" and doc["physical_ID_truth"] is False, "evidence status")
        records = doc["records"]
        rows = list(records.values()) if isinstance(records, dict) else records
        matched = [r for r in rows if r.get("frame_idx", r.get("frame")) == frame]
        require(len(matched) == 1 and set(matched[0]["observations"]) == set(VIEWS), "native row")
        chunk = doc["source_chunks"]["19"] if chunk_index == 19 else doc["source_chunk"]
        start, stop = (81340, 83000) if chunk_index == 19 else (89640, 91300)
        require(
            (chunk["index"], chunk["start"], chunk["stop"]) == (chunk_index, start, stop)
            and chunk["manifest_sha256"] == chunk_sha
            and start <= frame < stop,
            "consumed mask chunk differs",
        )
        manifest_path = (
            prior.native.checked_relative(MASK_ROOT, chunk["attempt_path"]) / "manifest.json"
        )
        saved.edge(visual, manifest_path, chunk_sha)
        manifest = saved.read(manifest_path, chunk_sha, evidence)
        require(
            manifest["frames"] == list(range(start, stop))
            and manifest["image_shape_hw"] == [1200, 1200],
            "native mask manifest differs",
        )
        captures = doc["captured_mask_records"]
        if chunk_index == 19:
            captures = captures["19"]
        require(set(captures) == set(VIEWS), "three captured mask views required")
        rig_refs = {
            p: d
            for p, d in doc["shared_SL_calibration_sha256"].items()
            if Path(p).suffix == ".json"
        }
        require(len(rig_refs) == 4, "registered calibration requires four JSONs")
        require(shared_rig is None or rig_refs == shared_rig, "registered rig differs by frame")
        shared_rig = rig_refs
        for path, sha in rig_refs.items():
            saved.edge(seal, path, sha)
            saved.edge(visual, path, sha)
        case_path = saved.member(GEOMETRY, seal, f"cases/{ARM}_{frame}.json", evidence)
        case = proof.read_json(case_path)
        require(
            case["schema"] == GEOMETRY.name + "_case"
            and case["frame_idx"] == frame
            and case["arm"] == ARM
            and case["lower"] == [0.0, 0.0, -6.0]
            and case["upper"] == [192.0, 192.0, 186.0],
            "saved geometry condition differs",
        )
        points = {}
        for view in VIEWS:
            ob = matched[0]["observations"][view]
            points[view] = np.asarray(ob["SLEAP_points"], dtype=np.float32)
            require(
                points[view].ndim == 3
                and points[view].shape[1:] == (3, 2)
                and proof.signature(points[view])
                == ob["SL_points_signature"]
                == case["source_points_signatures"][view],
                "saved SL reconstruction differs",
            )
            record = chunk["output_hashes"][view + "_masks"]
            path = Path(record["path"]).resolve()
            require(path.is_relative_to(REPO), "mask path outside repository")
            # Both old renderers record mask captures in their sealed evidence,
            # rather than as direct seal.inputs. Bind their consumed manifest.
            mask_ref = manifest["masks"][view]
            require(
                prior.native.checked_relative(manifest_path.parent, mask_ref["path"]) == path
                and mask_ref["sha256"] == record["sha256"]
                and path.stat().st_size == mask_ref["bytes"]
                and mask_ref["instances_per_frame"][frame - start] == ob["mask_count"],
                "manifest mask descriptor differs",
            )
            require(
                str(path) == captures[view]["path"]
                and record["sha256"] == captures[view]["sha256"],
                "mask capture binding differs",
            )
            points[view].flags.writeable = False
        cases[frame] = dict(
            geometry=case,
            bodies=complete_bodies(case),
            observations=matched[0]["observations"],
            points=points,
            chunk=chunk,
            captured=captures,
            geometry_case=seal["outputs"][f"cases/{ARM}_{frame}.json"],
            visual_evidence=visual["outputs"]["source_and_frame_evidence.json"],
        )
    return dict(cases=cases, rig_refs=shared_rig, geometry_plan=plan)


def source_code(context, evidence):
    modules = (
        saved,
        frozen,
        frozen.observations,
        prior.prepare_sequence,
        frozen.observations.calib,
        proof,
        bundle,
    )
    code = {}
    old = context["geometry_plan"]["source_code"]
    for module in modules:
        path = Path(module.__file__).resolve()
        require(str(path) in old, "reused source absent from saved geometry: " + str(path))
        code[str(path)] = evidence.add(path, old[str(path)])
    for module in (containment, instance_body_score, sparse_instance_body_score):
        path = Path(module.__file__).resolve()
        code[str(path)] = evidence.add(path, ADDITIONAL_SOURCE_PINS[path.name])
    code[str(Path(__file__).resolve())] = evidence.add(Path(__file__).resolve())
    return code


def load_rig(context, evidence):
    refs = context["rig_refs"]
    for path, sha in refs.items():
        evidence.add(path, sha)
    manifest = [Path(p) for p in refs if Path(p).name == "calibration_manifest.json"]
    require(len(manifest) == 1, "one calibration manifest required")
    metadata = proof.read_json(manifest[0])
    models = {
        str(prior.native.checked_relative(manifest[0].parent, r["path"]))
        for r in metadata["model_files"]
    }
    require(models | {str(manifest[0])} == set(refs), "calibration model membership differs")
    rig = frozen.observations.calib.load_rig(manifest[0].parent)
    require(tuple(rig.views) == tuple(VIEWS) and rig.world_units == "mm", "rig units/views")
    return rig


def measure_view(poses, points, masks, available, project):
    # Preparation keeps *all* nonempty masks in fields. The compatibility matrix
    # only annotates the independent containment result, never filters fields.
    observation = bundle.prepare_partial_bundle_observation(masks if available else None, points)
    measured = containment.measure_body_mask_containment(
        poses, project, observation.fields, observation.mask_indices
    )
    row_distances = (
        None if observation.fields is None else observation.fields.sample(observation.points)
    )
    return dict(
        containment=measured,
        SL_row_mask_compatibility=dict(
            row_indices=observation.row_indices,
            present=observation.present,
            mask_indices=observation.mask_indices,
            compatible=observation.compatible_masks,
            distances_by_row_node_mask_px=row_distances,
            tolerance_px=observation.config.mask_row_tolerance_px,
            evidence=observation.evidence,
            used_for_body_mask_eligibility=False,
        ),
        SLEAP_points=points,
        joint_three_body_bundle_cost_computed=False,
    )


def compact_body_metrics(indices, views, legacy_count):
    rows = []
    keys = (
        "best_single_mask",
        "best_single_mask_tie_count",
        "single_mask_max_distance_px",
        "single_mask_mean_distance_px",
        "endpoint_single_mask_max_distance_px",
        "union_max_distance_px",
        "mask_switch_advantage_px",
    )
    for position, index in enumerate(indices):
        per_view = {}
        for view, result in views.items():
            result = result["containment"]
            per_view[view] = dict(available=result["available"])
            if result["available"]:
                per_view[view].update({key: result[key][position] for key in keys})
        rows.append(dict(pool_index=index, legacy_body=index < legacy_count, views=per_view))
    return rows


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    mask_inputs, captured, retained_masks, mask_signatures, summaries = {}, {}, {}, {}, []
    try:
        context = load_headers(evidence)
        code = source_code(context, evidence)
        rig = load_rig(context, evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                arm=ARM,
                source_code=code,
                geometry_sha256=GEOMETRY_SHA,
                registered_calibration=context["rig_refs"],
                target_policy="all legacy and additional complete bodies; no score selection",
                focus_pool_indices=FOCUS,
                sample_labels=list(containment.SAMPLE_LABELS),
                row_compatibility_config=asdict(bundle.DEFAULT_CONFIG),
                row_compatibility_used_for_mask_eligibility=False,
                original_instance_axis_preserved_in_frame_masks=True,
                archive_capture_policy="six archives once each; one frame decoded per archive",
                no_framewise_mask_union_used_as_single_instance=True,
                tracking_or_cost_changes=False,
                automatic_gate=False,
                rejection_threshold=None,
                original_SL_cache_reads=0,
                video_reads=0,
                original_candidate_or_checkpoint_reads=0,
                candidate_generation_or_fit_calls=0,
                GPU_used=False,
                physical_ID_truth=False,
            ),
        )
        for frame in FRAMES:
            case = context["cases"][frame]
            indices = [b["pool_index"] for b in case["bodies"]]
            poses = np.asarray([b["pose_xyz"] for b in case["bodies"]], dtype=np.float64)
            views = {}
            for view in VIEWS:
                chunk, old_capture = case["chunk"], case["captured"][view]
                path = Path(old_capture["path"])
                packed, record = frozen.capture_masks(
                    path,
                    old_capture["sha256"],
                    np.arange(chunk["start"], chunk["stop"], dtype=np.int64),
                )
                require(record["arrays"] == old_capture["arrays"], "packed mask signatures differ")
                pos = frame - chunk["start"]
                mask = prior.prepare_sequence.decode_frame_masks(packed, pos)
                ob = case["observations"][view]
                digest = prior.prepare_sequence.array_sha256(mask)
                available = bool(packed["view_available"][pos])
                require(
                    digest == ob["mask_sha256"] == bytes(packed["frame_mask_sha256"][pos]).decode()
                    and available == ob["mask_available"]
                    and len(mask) == ob["mask_count"],
                    "target native masks differ from saved evidence",
                )
                key = f"f{frame}_{view}"
                mask.flags.writeable = False
                retained_masks[key] = mask
                mask_signatures[key] = dict(
                    signature=proof.signature(mask),
                    original_mask_sha256=digest,
                    frame_idx=frame,
                    view=view,
                    view_available=available,
                    original_mask_count=len(mask),
                    source_archive=old_capture["path"],
                )
                mask_inputs[record["path"]] = record["sha256"]
                captured[key] = record
                # No whole-chunk mask field survives into the next capture.
                del packed
                views[view] = measure_view(
                    poses, case["points"][view], mask, available, rig.camera(view).project
                )
            legacy_count = case["geometry"]["summary"]["legacy_candidate_count"]
            metrics = compact_body_metrics(indices, views, legacy_count)
            record = out.write(
                f"cases/f{frame}.json",
                json_value(
                    dict(
                        schema=STAGE + "_case",
                        frame_idx=frame,
                        arm=ARM,
                        source_geometry_case=case["geometry_case"],
                        source_visual_evidence=case["visual_evidence"],
                        pool_indices=indices,
                        bodies=case["bodies"],
                        legacy_candidate_count=legacy_count,
                        excluded_incomplete_pool_indices=[
                            b["pool_index"]
                            for b in case["geometry"]["pool"]
                            if b["pool_index"] not in indices
                        ],
                        views=views,
                        body_metrics=metrics,
                        focus_pool_indices=FOCUS[frame],
                        no_temporal_assignment=True,
                        no_rejection_threshold=True,
                    )
                ),
            )
            summaries.append(
                dict(
                    frame_idx=frame,
                    complete_body_count=len(indices),
                    legacy_complete_count=sum(i < legacy_count for i in indices),
                    additional_complete_count=sum(i >= legacy_count for i in indices),
                    focus_pool_indices=FOCUS[frame],
                    body_metrics=metrics,
                    case_json=record,
                )
            )
            print(
                f"{frame}: measured all {len(indices)} complete bodies in three views", flush=True
            )
        require(
            len(captured) == len(mask_inputs) == len(retained_masks) == 6, "six captures required"
        )
        masks_record = out.write("frame_masks.npz", retained_masks)
        # This only rereads the new small output, not an original mask archive.
        with np.load(masks_record["path"], allow_pickle=False) as archive:
            require(
                set(archive.files) == set(retained_masks) and len(archive.files) == 6,
                "saved six-frame membership differs",
            )
            for key in archive.files:
                require(
                    proof.signature(archive[key]) == mask_signatures[key]["signature"],
                    "saved frame-mask bytes differ",
                )
        out.write("frame_masks_signatures.json", mask_signatures)
        out.write(
            "summary.json",
            json_value(
                dict(
                    schema=STAGE,
                    status="ok",
                    cases=summaries,
                    captured_mask_records=captured,
                    frame_masks=masks_record,
                    mask_archive_captures=6,
                    target_mask_decodes=6,
                    selected_geometry_counts_are_complete=True,
                    independent_masks_all_nonempty=True,
                    SL_compatibility_filters_masks=False,
                    original_candidate_reads=0,
                    original_SL_cache_reads=0,
                    video_reads=0,
                    checkpoint_reads=0,
                    tracker_or_fit_calls=0,
                    GPU_used=False,
                    physical_ID_truth=False,
                    limitations=[
                        "Image distances do not identify physical fish.",
                        "D2 may split or merge fish; a low distance is not certification.",
                        "All views, including unused node views, remain separate.",
                        "Only the single-mask statistic requires one shared mask.",
                        "SL compatibility is separate; joint bundle cost is not computed.",
                        "No acceptance threshold, tuning, cost or trajectory change is applied.",
                    ],
                )
            ),
        )
        for record in captured.values():
            require(
                frozen.observations.file_stat(Path(record["path"])) == record["file_stat"],
                "mask archive changed after capture",
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
        (output / "measurement.provenance.json").write_bytes(proof.json_bytes(receipt))
        raise
    (output / "measurement.provenance.json").write_bytes(proof.json_bytes(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
