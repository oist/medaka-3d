"""Saved-only same-78020 comparison; no rollout, images, fitting or ID truth.

The control is one audited retained-history checkpoint, not the later 86320
backtrace. A new successful producer seal is required before that payload opens.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import compare_temporal_point64_v2 as proof
import fast_tail_two_minute_metrics as metrics
import numpy as np
import tracker_checkpoint as codec

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
STAGE = "instance_rescore_ten_second_comparison_v1"
PRODUCER_STAGE = "instance_rescore_ten_second_continuation_v1"
FIRST, START, STOP = 49800, 76360, 78020
ARMS = ("fast_tail_control", "instance_rescore")
LEGACY_DIFFERENCE_FIELD_ARMS = {
    "original_sources": ARMS[0],
    "original_review": ARMS[0],
    "tail_sources": ARMS[1],
    "tail_review": ARMS[1],
}
GEOMETRY_SCOPE = (
    "Estimated geometry in raw numeric body slots, including latent nodes. "
    "Pose deltas, speed and acceleration are not biological-ID-aligned errors."
)
CONTROL = WORK / "fast_tail_five_minute_continuation_v1/checkpoints/through_078020"
CONTROL_SHA = "0e8d002761370a71c97c4a9d09dece1d65be462c007008b42e8e384309aa8354"
AUDIT = WORK / "FAST_TAIL_CHECKPOINT_078020_SAVED_AUDIT.json"
AUDIT_SHA = "744c0abd1aafb5c00cbb80e3b6380a5ad5f7b991719576db2057308ba7565de2"
PARENT_SHA = "cef133c3e6192e373df095a3ec8b3b1cd546363a3aed8a6bf8ead670ff529cf6"
PARENT_AUDIT_SHA = "77bc769867edeb3f87c84abaf6f6c8f438b4f30f9a92c747b5e2a51e02d86188"
METRICS_SHA = "a3ebd24da07308938899695723090fbde20c4bbe27f5dd8fb9d441bef9ee9ddb"
REPORTING_PINS = {
    "compare_bounded_one_minute.py": (
        "5e1a4f570afadf204f96f3c6e83d55a9089f6fa3a14cf1b4c54cc84e4fe76cd0"
    ),
    "compare_tail_one_minute.py": (
        "da5f50c1dcc1399464c9ca8edfd65287a9f037f78119eee17cef7c27be388737"
    ),
    "mask_transport_diagnostics.py": (
        "6567710637b4a3b69dd03106cc525a1f7fac1659a7ab54acadef3c98517f3e56"
    ),
}
EXECUTION_READY = True
require = proof.require
support = metrics.old.prior.support_audit


def local(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO) and path.is_file(), "input outside workspace")
    return path


def read(path, evidence, expected):
    path = local(path)
    evidence.add(path, expected)
    return json.loads(path.read_text(encoding="utf8"))


def bound_member(root, seal, name, evidence):
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe member")
    path = local(root / name)
    require(path.is_relative_to(root), "member escaped source root")
    row = seal["outputs"][name]
    require(
        local(row["path"]) == path and path.stat().st_size == row["size_bytes"],
        "sealed output descriptor differs",
    )
    evidence.add(path, row["sha256"])
    return path


def signature(arrays):
    return {key: proof.signature(value) for key, value in arrays.items()}


def exact_fields(document, required, label):
    require(
        all(type(document.get(k)) is type(v) and document[k] == v for k, v in required.items()),
        label,
    )


def validate_import(imported, previous_context, final_metadata, policy):
    exact_fields(
        imported,
        dict(
            schema="instance_rescore_fast_tail_state_import_v1",
            status="ok",
            all_arrays_byte_preserved=True,
            original_metadata_unchanged=True,
            logical_initialize_count=1,
            new_begin_calls=0,
            added_restore_count=1,
            effective_from_native=START,
            processed_frames=START - FIRST,
            previous_frame=START - 1,
            review_cleared_by_migration=False,
            companion_diagnostics_required_each_future_frame=True,
        ),
        "single-boundary state import differs",
    )
    old, new = imported["old_context"], imported["new_context"]
    require(old == previous_context and new == final_metadata["context"], "import context differs")
    expected = copy.deepcopy(new)
    require(expected.pop("instance_rescore_policy") == policy, "scoring policy differs")
    expected["implementation"] = old["implementation"]
    require(expected == old and len(imported["arrays"]) == 39, "undeclared migration changes")
    before = imported["original_metadata"]
    exact_fields(
        before,
        dict(
            previous_frame=START - 1,
            processed_frames=START - FIRST,
            initialize_count=1,
            restore_count=4,
        ),
        "wrong imported parent lifecycle",
    )
    require(before["context"] == old, "wrong original metadata context")
    after = copy.deepcopy(before)
    after["context"] = new
    after["restore_count"] += 1
    require(imported["imported_metadata"] == after, "undeclared imported metadata changes")


def load_headers(branch, expected_sha, evidence):
    """All source and completion gates precede any checkpoint/candidate decode."""
    audit = read(AUDIT, evidence, AUDIT_SHA)
    require(
        audit["status"] == "PASS"
        and audit["checkpoint_manifest_sha256"] == CONTROL_SHA
        and audit["parent_manifest_sha256"] == PARENT_SHA
        and audit["all39_codec_integrity_both_snapshots"] is True
        and audit["ordered_native_frames_all6_beams"] is True
        and audit["appended_parent_indices_valid"] is True
        and audit["appended_native_interval"] == [START, STOP]
        and len(audit["retained_all19_and_offsets"]) == 20,
        "successful same78020 control audit required",
    )
    control = read(CONTROL / "manifest.json", evidence, CONTROL_SHA)
    require(
        control["status"] == "ok"
        and control["bindings"]["parent_checkpoint_sha256"] == PARENT_SHA
        and control["bindings"]["audited_parent_sha256"] == PARENT_AUDIT_SHA,
        "control checkpoint parent differs",
    )
    seal = read(branch / "tracking.provenance.json", evidence, expected_sha)
    require(
        seal["status"] == "ok" and seal["stage"] == PRODUCER_STAGE,
        "successful bounded producer required",
    )
    docs = {
        name: json.loads(bound_member(branch, seal, name + ".json", evidence).read_text("utf8"))
        for name in ("plan", "summary", "state_import", "input_index")
    }
    summary, plan = docs["summary"], docs["plan"]
    required = dict(
        status="ok",
        process_interval=[START, STOP],
        native_frames=STOP - START,
        retained_interval=[FIRST, STOP],
        parent_audit_sha256=PARENT_AUDIT_SHA,
        parent_checkpoint_sha256=PARENT_SHA,
        external_pose_resets=0,
        new_begin_calls=0,
        initialize_count=1,
        restore_count=5,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        finish_matches_checkpoint_backtrace=True,
    )
    exact_fields(summary, required, "producer completion differs")
    original = control["bindings"]["original_binding"]
    require(plan["original_binding"] == original, "common observation/config binding differs")
    index = docs["input_index"]
    require(
        index["original_binding"] == original and len(index["prefix_chunks"]) == 17,
        "complete registered 170s source prefix required",
    )
    require(
        all(
            (r["index"], r["start"], r["stop"]) == (i, FIRST + 1660 * i, FIRST + 1660 * (i + 1))
            for i, r in enumerate(index["prefix_chunks"])
        ),
        "source chunk order differs",
    )
    final_root = branch / "checkpoints/through_078020"
    final_path = bound_member(branch, seal, "checkpoints/through_078020/manifest.json", evidence)
    final = json.loads(final_path.read_text("utf8"))
    require(
        proof.sha(final_path) == summary["checkpoint_manifest_sha256"]
        and final["status"] == "ok"
        and final["bindings"] == plan["checkpoint_binding"]
        and final["bindings"]["original_binding"] == original
        and final["bindings"]["source_sha256"] == plan["source_sha256"]
        and final["bindings"]["parent_checkpoint_sha256"] == PARENT_SHA
        and final["bindings"]["parent_audit_sha256"] == PARENT_AUDIT_SHA
        and final["bindings"]["retained_interval"] == [FIRST, STOP]
        and final["bindings"]["process_interval"] == [START, STOP],
        "new final checkpoint/source binding differs",
    )
    final_meta_path = bound_member(
        branch, seal, "checkpoints/through_078020/metadata.json", evidence
    )
    metadata = json.loads(final_meta_path.read_text("utf8"))
    require(
        final_meta_path == final_root / final["metadata"]["file"]
        and proof.sha(final_meta_path) == final["metadata"]["sha256"]
        and [
            metadata[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ]
        == [STOP - 1, STOP - FIRST, 1, 5],
        "new final metadata/lifecycle differs",
    )
    control_metadata = read(
        CONTROL / control["metadata"]["file"], evidence, control["metadata"]["sha256"]
    )
    validate_import(
        docs["state_import"], control_metadata["context"], metadata, plan["new_scoring_policy"]
    )
    # The producer already verified its new payload. Bind its descriptor without opening it.
    payload = seal["outputs"]["checkpoints/through_078020/arrays.npz"]
    require(
        Path(payload["path"]).resolve() == final_root / final["payload"]["file"]
        and payload["sha256"] == final["payload"]["sha256"]
        and payload["size_bytes"] == final["payload"]["bytes"],
        "sealed new checkpoint payload descriptor differs",
    )
    producer_path = str((WORK / "run_instance_rescore_10s.py").resolve())
    require(
        seal["inputs"].get(producer_path) == plan["source_sha256"][producer_path],
        "producer source input edge differs",
    )
    evidence.add(producer_path, plan["source_sha256"][producer_path])
    expected_code = {
        **control["bindings"]["source_sha256"],
        **{str(WORK / name): digest for name, digest in REPORTING_PINS.items()},
        str(Path(metrics.__file__).resolve()): METRICS_SHA,
    }
    expected_code[str(Path(__file__).resolve())] = proof.sha(__file__)
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if (
            path.suffix == ".py"
            and path.is_relative_to(REPO)
            and ".venv" not in path.parts
            and path.is_file()
        ):
            require(str(path) in expected_code, "unbound comparison dependency: " + str(path))
            evidence.add(path, expected_code[str(path)])
    return control, seal, docs


def load_candidates(branch, control, seal, docs, evidence):
    arrays, metadata, verified = codec.read_checkpoint(
        CONTROL, expected_bindings=control["bindings"]
    )
    require(
        verified == control
        and len(arrays) == 39
        and [
            metadata[k]
            for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
        ]
        == [STOP - 1, STOP - FIRST, 1, 5],
        "audited control payload/lifecycle differs",
    )
    for key in ("metadata", "payload"):
        descriptor = control[key]
        path = local(CONTROL / descriptor["file"])
        evidence.paths[str(path)] = descriptor["sha256"]  # Codec verified SHA and size twice.
    old = support.COMMON.traced_candidate(arrays)
    del arrays
    path = bound_member(branch, seal, "candidate.npz", evidence)
    with np.load(path, allow_pickle=False) as archive:
        new = {key: archive[key] for key in archive.files}
    require(signature(new) == docs["summary"]["candidate_arrays"], "candidate proof differs")
    for candidate in (old, new):
        metrics.validate_candidate(candidate, FIRST, STOP)
    return dict(zip(ARMS, (old, new), strict=True)), metadata


def load_observations(original, seal, metadata, evidence):
    shared = original["inputs"]["seed"]["SL_and_calibration_sha256"]
    require(len(shared) == 7, "registered SL3/calibration4 required")
    for path, digest in shared.items():
        require(
            seal["inputs"].get(str(local(path))) == digest, "new observation input edge differs"
        )
        evidence.add(path, digest)
    paths = {
        view: [
            Path(p)
            for p in shared
            if Path(p).name == metrics.old.prior.EXPERIMENT + "__" + view + ".npz"
        ]
        for view in metrics.old.prior.VIEWS
    }
    require(all(len(p) == 1 for p in paths.values()), "unique SL archive per camera required")
    folders = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(folders) == 1, "one calibrated rig required")
    rig = metrics.old.prior.load_rig(folders[0])
    seed_info = original["inputs"]["seed"]
    registration = read(seed_info["registration_path"], evidence, seed_info["registration_sha256"])
    seed = np.asarray(registration["pose_xyz"], dtype=np.float64)
    require(
        metadata["context"]["initial_pose"] == support.COMMON.digest_array(seed)
        and metadata["context"]["cameras"]
        == support.COMMON.normalize([asdict(c) for c in rig.cameras]),
        "control seed geometry differs",
    )
    nodes = metrics.old.prior.RegisteredNodes(
        {view: found[0] for view, found in paths.items()}, np.arange(START, STOP, dtype=np.int64)
    )
    return nodes, rig, seed


def same_horizon_prefix(old, new):
    """Terminal path choice may differ although the producer retained history exactly."""
    result, arrays = metrics.horizon_changes(old, new, FIRST, START)
    result.pop("snapshot_horizon")
    result.pop("final_horizon")
    result.update(
        control_terminal_horizon=STOP,
        variant_terminal_horizon=STOP,
        interpretation="Same-horizon selected prefix paths, separate from immutable beam history.",
    )
    return result, arrays


def geometry_deltas(old, new):
    keep = old["frame_idx"] >= START
    xyz = np.stack([a["pose_xyz"][keep] for a in (old, new)])
    observed = np.stack([a["observed"][keep] for a in (old, new)])
    arrays = dict(
        frame_idx=old["frame_idx"][keep].copy(),
        pose_delta_mm=np.linalg.norm(xyz[1] - xyz[0], axis=-1),
        all_estimated_speed_mm_s=np.linalg.norm(np.diff(xyz, axis=1), axis=-1) * 166,
        all_estimated_acceleration_mm_s2=np.linalg.norm(np.diff(xyz, n=2, axis=1), axis=-1)
        * 166**2,
        speed_both_direct=observed[:, 1:] & observed[:, :-1],
        acceleration_all_three_direct=observed[:, 2:] & observed[:, 1:-1] & observed[:, :-2],
        chest_pair_distance_mm=np.stack(
            [
                np.linalg.norm(xyz[:, :, i, 1] - xyz[:, :, j, 1], axis=-1)
                for i, j in ((0, 1), (0, 2), (1, 2))
            ],
            axis=-1,
        ),
    )
    stats = metrics.geometry._stats
    return dict(
        pose_delta_mm=stats(arrays["pose_delta_mm"]),
        speed_by_arm=[stats(a) for a in arrays["all_estimated_speed_mm_s"]],
        acceleration_by_arm=[stats(a) for a in arrays["all_estimated_acceleration_mm_s2"]],
        chest_pair_distance_by_arm=[stats(a) for a in arrays["chest_pair_distance_mm"]],
        no_anatomical_or_identity_threshold_introduced=True,
        raw_slot_estimated_geometry_scope=GEOMETRY_SCOPE,
    ), arrays


def run(branch, expected_sha, output):
    require(EXECUTION_READY, "source review and explicit bounded evaluation Go required")
    require(proof.valid_sha(expected_sha), "explicit successful new tracking seal SHA required")
    branch, output = branch.resolve(strict=True), output.resolve()
    require(branch == WORK / PRODUCER_STAGE, "fixed variant branch required")
    require(output.is_relative_to(WORK) and not output.exists(), "fresh comparison output required")
    out, evidence = proof.Output(output), proof.Evidence()
    started = proof.utc()
    try:
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                process_interval=[START, STOP],
                terminal_horizon=STOP,
                arms=list(ARMS),
                legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
                raw_slot_estimated_geometry_scope=GEOMETRY_SCOPE,
                control_checkpoint_sha256=CONTROL_SHA,
                new_tracking_sha256=expected_sha,
                raw_support_ray_mm=2.0,
                raw_support_pixel_gate=12.0,
                evaluation_thresholds_retuned=False,
                new_instance_score_used_as_accuracy_metric=False,
                diagnosis_selected_frames=[77709, 77723, 77880],
                held_out_identity_truth=False,
                control_checkpoint_payload_decodes=1,
                new_checkpoint_payload_decodes=0,
                raw_video_reads=0,
                masks_decoded=0,
                tracking_or_fit_performed=False,
            ),
        )
        control, seal, docs = load_headers(branch, expected_sha, evidence)
        candidates, metadata = load_candidates(branch, control, seal, docs, evidence)
        before = {arm: signature(candidate) for arm, candidate in candidates.items()}
        nodes, rig, seed = load_observations(
            control["bindings"]["original_binding"], seal, metadata, evidence
        )
        node_before = {
            view: signature(dict(points=p, offsets=o)) for view, (p, o) in nodes.data.items()
        }
        methods = {}
        for arm, candidate in candidates.items():
            out.write(f"candidates/{arm}.npz", candidate)
            part = {
                k: candidate[k][START - FIRST :]
                for k in ("frame_idx", "pose_xyz", "observed", "sources")
            }
            strict = support.verify_support_partitioned(
                part,
                nodes,
                rig,
                block_frames=STOP - START,
                on_block=lambda meta, arrays, arm=arm: out.write(f"support/{arm}.npz", arrays),
            )
            values, motion, shape = metrics.interval_metrics(candidate, seed, START, STOP)
            out.write(f"motion/{arm}.json", motion)
            out.write(f"geometry/{arm}.npz", shape)
            methods[arm] = dict(interval=values, strict_support=strict)
        old, new = (candidates[arm] for arm in ARMS)
        changes, rows, transitions = metrics.differences(old, new, START, STOP)
        out.write("differences/direct_gain_loss.json", rows)
        out.write("differences/transitions.npz", transitions)
        prefix, prefix_arrays = same_horizon_prefix(old, new)
        out.write("differences/selected_prefix_paths.npz", prefix_arrays)
        geometry, geometry_arrays = geometry_deltas(old, new)
        out.write("differences/geometry.npz", geometry_arrays)
        require(all(signature(candidates[a]) == before[a] for a in ARMS), "candidate mutated")
        require(
            all(
                signature(dict(points=p, offsets=o)) == node_before[v]
                for v, (p, o) in nodes.data.items()
            ),
            "SL observations mutated",
        )
        require(not sys.modules["torch"].cuda.is_initialized(), "comparison initialized CUDA")
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                native_frames=STOP - START,
                terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
                legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
                raw_slot_estimated_geometry_scope=GEOMETRY_SCOPE,
                candidate_arrays=before,
                methods=methods,
                support_changes=changes,
                geometry_deltas=geometry,
                selected_prefix_path_changes=prefix,
                producer_prior_history_prefix_unchanged=docs["summary"][
                    "prior_history_prefix_unchanged"
                ],
                immutable_history_proof_reused_not_recomputed=True,
                current_candidate_backtrace_is_not_immutable_online_history=True,
                control_checkpoint_payload_decodes=1,
                new_checkpoint_payload_decodes=0,
                raw_video_reads=0,
                masks_decoded=0,
                GPU_used=False,
                tracking_or_fit_performed=False,
                human_truth_created=False,
                all_three_physical_ID_truth=False,
                adopted=False,
                limitations=[
                    "Slots are numeric labels, not independently aligned biological IDs.",
                    "Support and geometry diagnostics do not establish tracking accuracy.",
                    "The three prior diagnosis frames are not held-out evidence.",
                    "Neutral or lower instance score is not evaluated as independent truth here.",
                ],
            ),
        )
        evidence.final_guard()
        out.final_guard()
        result = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        result = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "comparison.provenance.json").write_bytes(proof.json_bytes(result))
        raise
    (output / "comparison.provenance.json").write_bytes(proof.json_bytes(result))
    print(json.dumps(dict(status="ok", output=str(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--expected-tracking-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.branch, args.expected_tracking_sha, args.output)
