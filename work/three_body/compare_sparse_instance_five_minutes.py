"""Saved-only, same-116200 comparison of FastTail and mask-only SparseInstance.

Consume final candidate21 and registered SL, never checkpoint arrays or images.
Both final tracking seals must succeed before candidate payloads are opened.
The unchanged -6 mm lower bound remains an experimental limitation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import compare_fast_tail_two_minutes as female_helpers
import compare_instance_rescore_10s as common
import numpy as np

proof, metrics, support = common.proof, common.metrics, common.support
require = proof.require
WORK, REPO = common.WORK, common.REPO
FIRST, START, MASK_START, SPARSE_START, STOP, CHUNK = 49800, 66400, 76360, 78020, 116200, 1660
STAGE = "sparse_instance_five_minute_comparison_v1"
ARMS = ("fast_tail_control", "sparse_instance_rescore")
SPECS = {
    ARMS[0]: dict(
        stage="fast_tail_five_minute_continuation_v1",
        runner="continue_fast_tail_five_minutes.py",
        source_sha="d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6",
        resume=MASK_START,
        restores=5,
    ),
    ARMS[1]: dict(
        stage="sparse_instance_five_minute_continuation_v1",
        runner="continue_sparse_instance_five_minutes.py",
        source_sha="3f5141a5aeec6fadc464ad242ac19fe13e227065af6f1188c638af4c58129096",
        resume=SPARSE_START,
        restores=6,
    ),
}
PARENT_STAGE = "instance_rescore_ten_second_continuation_v1"
PARENT_SHA = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
QUALIFICATION_SHA = "941ca436a5e59fca5a63794483d315da3950402aafdb4ce13c34630e98014125"
EXECUTION_CONTRACT_SHA = "097277d2a28c43edccafba39f96c779890bf553b8d83996051b2bf98198814b4"
PINS = {
    "compare_instance_rescore_10s.py": (
        "bb7653e5f7d71f2b4c078418dee93433fb4a1fc3fdefd7f38ef9c4694ae9040a"
    ),
    "compare_fast_tail_two_minutes.py": (
        "e07b51f364f5d6f85e8e26d34a0051145412df29e34c91d379d2141aec9b649d"
    ),
    "compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
    "fast_tail_two_minute_metrics.py": common.METRICS_SHA,
    "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md": EXECUTION_CONTRACT_SHA,
    **common.REPORTING_PINS,
    "sparse_instance_rescore_tracker.py": (
        "d041aff3a0d8cbd71341dbe67dd9b5a7a1e2e7aeb165f55e4b08f8fabdb8fb79"
    ),
}
INTERVALS = {
    "full": (START, STOP),
    **{f"minute_{i + 1}": (START + i * 9960, START + (i + 1) * 9960) for i in range(5)},
    "dense_mask_first_ten_seconds": (MASK_START, SPARSE_START),
    "sparse_continuation": (SPARSE_START, STOP),
}
LEGACY_DIFFERENCE_FIELD_ARMS = {
    "original_sources": ARMS[0],
    "original_review": ARMS[0],
    "tail_sources": ARMS[1],
    "tail_review": ARMS[1],
}
EXECUTION_READY = True


def source_code():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        # Some third-party generated modules expose relative placeholder filenames.
        # They do not name workspace source files and must not resolve against cwd.
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.is_relative_to(REPO) and ".venv" not in path.parts and path.suffix == ".py":
                paths.add(path)
    code = {str(path): proof.sha(path) for path in sorted(paths)}
    for name, digest in PINS.items():
        path = WORK / name
        require(proof.sha(path) == digest, "frozen reporting/scorer source differs: " + name)
        code[str(path)] = digest
    return code


def member_doc(root, seal, name, evidence):
    return json.loads(common.bound_member(root, seal, name, evidence).read_text("utf8"))


def is_producer_source(path, digest):
    path = Path(path)
    return path.suffix == ".py" or (
        path.resolve() == WORK / "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md"
        and digest == EXECUTION_CONTRACT_SHA
    )


def completion(summary, arm):
    spec = SPECS[arm]
    common.exact_fields(
        summary,
        dict(
            schema=spec["stage"],
            status="ok",
            scientific_interval=[START, STOP],
            process_interval=[spec["resume"], STOP],
            native_frames=49800,
            process_frames=STOP - spec["resume"],
            initialize_count=1,
            restore_count=spec["restores"],
            external_pose_resets=0,
            new_begin_calls=0,
            finish_calls=1,
            physical_ID_truth=False,
        ),
        "completed fixed five-minute tracking required: " + arm,
    )
    if arm == ARMS[1]:
        common.exact_fields(
            summary,
            dict(
                retained_interval=[FIRST, STOP],
                all39_roundtrip=True,
                prior_history_prefix_unchanged=True,
                finish_matches_checkpoint_backtrace=True,
                automatic_adoption=False,
            ),
            "sparse full-state/final-backtrace completion missing",
        )
    rows = summary["checkpoints"]
    stops = list(range(spec["resume"] + CHUNK, STOP + CHUNK, CHUNK))
    require(len(rows) == len(stops), "checkpoint count differs")
    for row, stop in zip(rows, stops, strict=True):
        common.exact_fields(
            row,
            dict(
                arrays=39,
                previous_frame=stop - 1,
                processed_frames=stop - FIRST,
                initialize_count=1,
                restore_count=spec["restores"],
            ),
            "checkpoint lifecycle/sequence differs",
        )


def checkpoint_header(root, seal, stop, evidence):
    """Bind small headers; the retained39 payload descriptor is not consumed."""
    prefix = f"checkpoints/through_{stop:06d}/"
    manifest = member_doc(root, seal, prefix + "manifest.json", evidence)
    metadata = member_doc(root, seal, prefix + "metadata.json", evidence)
    common.exact_fields(
        manifest,
        dict(schema="medaka_tracker_checkpoint_v1", status="ok"),
        "successful checkpoint header required",
    )
    for field, name in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
        row, declared = seal["outputs"][prefix + name], manifest[field]
        require(
            declared["file"] == name
            and declared["sha256"] == row["sha256"]
            and declared["bytes"] == row["size_bytes"]
            and Path(row["path"]).resolve() == root / prefix / name,
            "checkpoint sealed descriptor differs",
        )
    return manifest, metadata


def load_headers(arm, expected_sha, evidence):
    spec = SPECS[arm]
    require(proof.valid_sha(expected_sha), "explicit successful final tracking SHA required")
    root = WORK / spec["stage"]
    seal = common.read(root / "tracking.provenance.json", evidence, expected_sha)
    common.exact_fields(seal, dict(stage=spec["stage"], status="ok"), "final tracking seal failed")
    docs = {
        name: member_doc(root, seal, name + ".json", evidence)
        for name in ("summary", "plan", "input_index")
    }
    completion(docs["summary"], arm)
    plan, index = docs["plan"], docs["input_index"]
    common.exact_fields(
        plan,
        dict(
            schema=spec["stage"],
            scientific_interval=[START, STOP],
            process_interval=[spec["resume"], STOP],
            precision="float64",
            initialize_count=1,
            restore_count=spec["restores"],
            external_pose_resets=0,
            new_begin_calls=0,
        ),
        "producer plan differs",
    )
    binding = plan["checkpoint_binding"]
    require(
        binding["schema"] == spec["stage"] + "_checkpoint"
        and binding["original_binding"] == plan["original_binding"] == index["original_binding"]
        and binding["source_sha256"] == plan["source_sha256"]
        and binding["scientific_interval"] == [START, STOP]
        and binding["process_interval"] == [spec["resume"], STOP],
        "plan/source binding differs",
    )
    runner = str(WORK / spec["runner"])
    require(plan["source_sha256"].get(runner) == spec["source_sha"], "producer source differs")
    for path, digest in plan["source_sha256"].items():
        require(
            is_producer_source(path, digest) and seal["inputs"].get(path) == digest,
            "producer consumed source edge differs",
        )
        evidence.add(common.local(path), digest)
    chunks = index["prefix_chunks"]
    require(len(chunks) == 40, "all40 consumed chunks required")
    for i, row in enumerate(chunks):
        common.exact_fields(
            row,
            dict(index=i, start=FIRST + i * CHUNK, stop=FIRST + (i + 1) * CHUNK),
            "consumed native chunk order differs",
        )
    final, meta = checkpoint_header(root, seal, STOP, evidence)
    require(final["bindings"] == binding, "final checkpoint binding differs")
    common.exact_fields(
        meta,
        dict(
            initialize_count=1,
            restore_count=spec["restores"],
            processed_frames=STOP - FIRST,
            previous_frame=STOP - 1,
        ),
        "final checkpoint lifecycle differs",
    )
    row = docs["summary"]["checkpoints"][-1]
    require(
        row["path"] == str(root / "checkpoints/through_116200")
        and row["manifest_sha256"]
        == seal["outputs"]["checkpoints/through_116200/manifest.json"]["sha256"]
        and row["payload_bytes"] == final["payload"]["bytes"],
        "final checkpoint row differs",
    )
    milestone = member_doc(root, seal, "milestones/through_116200/summary.json", evidence)
    common.exact_fields(
        milestone,
        dict(
            schema=spec["stage"] + "_milestone",
            status="ok",
            horizon_stop=STOP,
            scientific_interval=[START, STOP],
            native_frames=49800,
            initialize_count=1,
            restore_count=spec["restores"],
            external_pose_resets=0,
            finish_calls_so_far=0,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
        ),
        "final milestone proof differs",
    )
    require(
        milestone["original_binding"] == plan["original_binding"]
        and milestone["source_sha256"] == plan["source_sha256"]
        and milestone["checkpoint"] == row
        and milestone["candidate"] == seal["outputs"]["milestones/through_116200/candidate.npz"],
        "milestone candidate/source binding differs",
    )
    docs.update(metadata=meta, manifest=final, milestone=milestone)
    if arm == ARMS[0]:
        restored = member_doc(root, seal, "restore.json", evidence)
        common.exact_fields(
            restored,
            dict(
                all39_BYTEEXACT=True,
                metadata_only_change="restore_count:4->5",
                initializes=1,
                new_begin_calls=0,
                external_pose_resets=0,
                restored_previous_native_frame=MASK_START - 1,
                previous_images_velocity_all_beams_and_history_preserved=True,
            ),
            "FastTail original full-state restoration proof differs",
        )
        docs["restore"] = restored
    return dict(arm=arm, root=root, seal=seal, docs=docs, seal_sha256=expected_sha)


def validate_sparse_lineage(control, variant, evidence):
    """Re-use sealed parent and all39 proofs; never decode a retained state."""
    root, seal, docs = variant["root"], variant["seal"], variant["docs"]
    old_context = control["docs"]["metadata"]["context"]
    plan, ctx = docs["plan"], docs["metadata"]["context"]
    require(
        old_context["implementation"] == "FastTailRecoveryContinuousTracker"
        and ctx["implementation"] == "SparseInstanceRescoreContinuousTracker",
        "registered raw candidate implementations required",
    )
    common.exact_fields(
        plan,
        dict(
            retained_interval=[FIRST, STOP],
            dense_serial_control_interval=[SPARSE_START, SPARSE_START + 2],
            sparse_rollout_continues_after_control_without_restore=True,
            mask_policy_effective_from_native=MASK_START,
            sparse_backend_effective_from_native=SPARSE_START,
            semantic_landmark_term_used=False,
        ),
        "declared mask/sparse policy differs",
    )
    imported = member_doc(root, seal, "state_import.json", evidence)
    common.exact_fields(
        imported,
        dict(
            schema="sparse_instance_dense_state_import_v1",
            status="ok",
            all_arrays_byte_preserved=True,
            original_metadata_unchanged=True,
            logical_initialize_count=1,
            new_begin_calls=0,
            added_restore_count=1,
            processed_frames=SPARSE_START - FIRST,
            previous_frame=SPARSE_START - 1,
            effective_from_native=SPARSE_START,
            backend_applied_retroactively_to_prior_history=False,
            review_cleared_by_migration=False,
            companion_diagnostics_required_each_future_frame=True,
        ),
        "sparse migration proof differs",
    )
    require(
        len(imported["arrays"]) == 39 and imported["new_context"] == ctx,
        "all39 import/final context differs",
    )
    dense_ctx = copy.deepcopy(ctx)
    backend = dense_ctx.pop("sparse_distance_backend")
    dense_ctx["implementation"] = imported["old_context"]["implementation"]
    require(
        dense_ctx == imported["old_context"]
        and dense_ctx["implementation"] == "InstanceRescoreContinuousTracker"
        and backend == plan["checkpoint_binding"]["sparse_distance_backend"],
        "undeclared sparse migration context change",
    )
    require(
        backend["effective_from_native"] == SPARSE_START
        and backend["saved_workload_qualification_sha256"] == QUALIFICATION_SHA,
        "qualified sparse backend differs",
    )
    policy = dense_ctx.pop("instance_rescore_policy")
    dense_ctx["implementation"] = old_context["implementation"]
    require(
        dense_ctx == old_context
        and policy == plan["checkpoint_binding"]["scoring_policy"]
        and policy["effective_from_native"] == MASK_START,
        "calibration/seed/bounds/fitter or scientific policy mixed",
    )
    require(
        ctx["lower"] == support.COMMON.digest_array(np.array([0.0, 0.0, -6.0]))
        and ctx["upper"] == support.COMMON.digest_array(np.array([192.0, 192.0, 186.0]))
        and ctx["recovery_config"]["lower_mm"] == [0.0, 0.0, -6.0],
        "this comparison requires unchanged -6 mm engineering lower bound",
    )
    before = imported["original_metadata"]
    common.exact_fields(
        before,
        dict(
            previous_frame=SPARSE_START - 1,
            processed_frames=SPARSE_START - FIRST,
            initialize_count=1,
            restore_count=5,
        ),
        "dense parent lifecycle differs",
    )
    require(before["context"] == imported["old_context"], "source decoder context differs")
    adapted = copy.deepcopy(before)
    adapted.update(context=ctx, restore_count=6)
    require(imported["imported_metadata"] == adapted, "import changed undeclared metadata")
    parent_root = WORK / PARENT_STAGE
    require(
        seal["inputs"].get(str(parent_root / "tracking.provenance.json")) == PARENT_SHA
        and plan["checkpoint_binding"]["parent_tracking_sha256"] == PARENT_SHA,
        "actual successful dense parent edge missing",
    )
    parent = common.read(parent_root / "tracking.provenance.json", evidence, PARENT_SHA)
    common.exact_fields(parent, dict(stage=PARENT_STAGE, status="ok"), "dense parent incomplete")
    pm = member_doc(parent_root, parent, "checkpoints/through_078020/metadata.json", evidence)
    ps = member_doc(parent_root, parent, "summary.json", evidence)
    require(
        pm == before
        and ps["checkpoint_manifest_sha256"]
        == plan["checkpoint_binding"]["parent_checkpoint_sha256"],
        "dense parent state binding differs",
    )
    require(
        ps["parent_checkpoint_sha256"] == common.PARENT_SHA
        and ps["parent_audit_sha256"] == common.PARENT_AUDIT_SHA
        and control["docs"]["plan"]["checkpoint_binding"]["parent_checkpoint_sha256"]
        == common.PARENT_SHA
        and control["docs"]["plan"]["checkpoint_binding"]["audited_parent_sha256"]
        == common.PARENT_AUDIT_SHA,
        "same audited76360 branch point required",
    )
    serial = member_doc(root, seal, "serial_dense_sparse_qualification.json", evidence)
    common.exact_fields(
        serial,
        dict(
            status="ok",
            native_interval=[SPARSE_START, SPARSE_START + 2],
            all39_BYTEEXACT=True,
            initialize_count=1,
            restore_count=6,
            metadata_difference="declared backend context and implementation only",
            full_five_minute_equivalence_proven=False,
        ),
        "actual two-frame GPU qualification missing",
    )
    require(
        serial["backend"] == backend and len(serial["compared_arrays"]) == 39,
        "serial backend/all39 proof differs",
    )
    qpath = WORK / "sparse_instance_saved_ten_second_qualification_v1/qualification.provenance.json"
    require(
        seal["inputs"].get(str(qpath)) == QUALIFICATION_SHA, "saved1660 qualification not consumed"
    )
    qseal = common.read(qpath, evidence, QUALIFICATION_SHA)
    require(qseal["status"] == "ok", "saved1660 qualification failed")
    return dict(
        sparse_import=imported,
        serial_two_native=serial,
        parent_tracking_sha256=PARENT_SHA,
        saved1660_qualification_sha256=QUALIFICATION_SHA,
    )


def load_candidates(bundles, evidence):
    candidates, references = {}, {}
    for arm, bundle in bundles.items():
        root, seal = bundle["root"], bundle["seal"]
        path = common.bound_member(root, seal, "candidate.npz", evidence)
        with np.load(path, allow_pickle=False) as saved:
            candidate = {key: saved[key] for key in saved.files}
        metrics.validate_candidate(candidate, FIRST, STOP)
        signatures = common.signature(candidate)
        milestone = common.bound_member(
            root, seal, "milestones/through_116200/candidate.npz", evidence
        )
        # Archive SHA/order can differ; all21 numerical bytes must still agree.
        with np.load(milestone, allow_pickle=False) as saved:
            require(set(saved.files) == set(candidate), "milestone field membership differs")
            require(
                all(proof.signature(saved[k]) == signatures[k] for k in candidate),
                "final candidate differs from same116200 retained backtrace",
            )
        if arm == ARMS[1]:
            require(
                signatures == bundle["docs"]["summary"]["candidate_arrays"],
                "all21 sealed sparse candidate signature differs",
            )
        for a in candidate.values():
            a.setflags(write=False)
        candidates[arm] = candidate
        references[arm] = dict(
            path=str(path),
            sha256=evidence.paths[str(path)],
            milestone_path=str(milestone),
            milestone_sha256=evidence.paths[str(milestone)],
            terminal_horizon=STOP,
            arrays=signatures,
        )
    return candidates, references


def common_observation_binding(bundles):
    original = bundles[ARMS[0]]["docs"]["plan"]["original_binding"]
    require(
        all(b["docs"]["plan"]["original_binding"] == original for b in bundles.values()),
        "observation/config bindings differ",
    )
    require(
        bundles[ARMS[0]]["docs"]["input_index"] == bundles[ARMS[1]]["docs"]["input_index"],
        "actual consumed40 mask/native chunk records differ",
    )
    return original


def load_observations(bundles, evidence):
    original = common_observation_binding(bundles)
    seed_info = original["inputs"]["seed"]
    shared = seed_info["SL_and_calibration_sha256"]
    require(len(shared) == 7, "same SL3/calibration4 required")
    for path, digest in shared.items():
        require(
            all(b["seal"]["inputs"].get(path) == digest for b in bundles.values()),
            "actual consumed SL/calibration edge differs",
        )
        evidence.add(common.local(path), digest)
    paths = {
        v: [
            Path(p)
            for p in shared
            if Path(p).name == metrics.old.prior.EXPERIMENT + "__" + v + ".npz"
        ]
        for v in metrics.old.prior.VIEWS
    }
    require(all(len(p) == 1 for p in paths.values()), "unique SL cache per view required")
    calibration = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(calibration) == 1, "one calibrated rig required")
    rig = metrics.old.prior.load_rig(calibration[0])
    seed = np.asarray(
        common.read(seed_info["registration_path"], evidence, seed_info["registration_sha256"])[
            "pose_xyz"
        ],
        np.float64,
    )
    for b in bundles.values():
        ctx = b["docs"]["metadata"]["context"]
        require(
            ctx["initial_pose"] == support.COMMON.digest_array(seed)
            and ctx["cameras"] == support.COMMON.normalize([asdict(c) for c in rig.cameras]),
            "registered calibration/seed context differs",
        )
    nodes = metrics.old.prior.RegisteredNodes(
        {v: p[0] for v, p in paths.items()}, np.arange(START, STOP)
    )
    return nodes, rig, seed, shared


def selected_prefix_changes(control, variant):
    value, arrays = metrics.horizon_changes(control, variant, FIRST, MASK_START)
    value.pop("snapshot_horizon")
    value.pop("final_horizon")
    value.update(
        terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
        interpretation=(
            "Same terminal horizon can select different old prefix paths; "
            "retained beam history was not edited."
        ),
    )
    return value, arrays


def geometry_deltas(control, variant, start=START, stop=STOP):
    require(np.array_equal(control["frame_idx"], variant["frame_idx"]), "geometry frame mismatch")
    keep = (control["frame_idx"] >= start) & (control["frame_idx"] < stop)
    require(
        int(keep.sum()) == stop - start and stop - start >= 3, "complete motion interval required"
    )
    pose = np.stack([a["pose_xyz"][keep] for a in (control, variant)])
    observed = np.stack([a["observed"][keep] for a in (control, variant)])
    arrays = dict(
        frame_idx=control["frame_idx"][keep].copy(),
        pose_delta_mm=np.linalg.norm(pose[1] - pose[0], axis=-1),
        all_estimated_speed_mm_s=np.linalg.norm(np.diff(pose, axis=1), axis=-1) * 166,
        all_estimated_acceleration_mm_s2=np.linalg.norm(np.diff(pose, n=2, axis=1), axis=-1)
        * 166**2,
        speed_both_direct=observed[:, 1:] & observed[:, :-1],
        acceleration_all_three_direct=observed[:, 2:] & observed[:, 1:-1] & observed[:, :-2],
        chest_pair_distance_mm=np.stack(
            [
                np.linalg.norm(pose[:, :, i, 1] - pose[:, :, j, 1], axis=-1)
                for i, j in ((0, 1), (0, 2), (1, 2))
            ],
            axis=-1,
        ),
    )
    stats = metrics.geometry._stats
    return dict(
        pose_delta_mm=stats(arrays["pose_delta_mm"]),
        speed_by_arm={
            a: stats(v) for a, v in zip(ARMS, arrays["all_estimated_speed_mm_s"], strict=True)
        },
        acceleration_by_arm={
            a: stats(v)
            for a, v in zip(ARMS, arrays["all_estimated_acceleration_mm_s2"], strict=True)
        },
        chest_pair_distance_by_arm={
            a: stats(v) for a, v in zip(ARMS, arrays["chest_pair_distance_mm"], strict=True)
        },
        no_anatomical_or_identity_threshold_introduced=True,
        raw_slot_estimated_geometry_scope=common.GEOMETRY_SCOPE,
    ), arrays


def run(control_sha, variant_sha, output):
    require(EXECUTION_READY, "source review and successful final tracking seals required")
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    require(
        all(proof.valid_sha(s) for s in (control_sha, variant_sha)),
        "explicit final seal SHAs required",
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        require(not sys.modules["torch"].cuda.is_initialized(), "CPU-only comparison required")
        code = source_code()
        for path, digest in code.items():
            evidence.add(path, digest)
        bundles = {
            arm: load_headers(arm, sha, evidence)
            for arm, sha in zip(ARMS, (control_sha, variant_sha), strict=True)
        }
        lineage = validate_sparse_lineage(bundles[ARMS[0]], bundles[ARMS[1]], evidence)
        common_observation_binding(bundles)
        # Only successful complete sources reach any candidate/SL payload read.
        candidates, references = load_candidates(bundles, evidence)
        nodes, rig, seed, shared = load_observations(bundles, evidence)
        node_before = {
            v: common.signature(dict(points=p, offsets=o)) for v, (p, o) in nodes.data.items()
        }
        anchors = female_helpers.female_anchors(shared, evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                arms=list(ARMS),
                native_interval=[START, STOP],
                native_frames=49800,
                intervals=INTERVALS,
                terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
                candidate_references=references,
                tracking_seals=dict(zip(ARMS, (control_sha, variant_sha), strict=True)),
                source_sha256=code,
                strict_support_ray_mm=2.0,
                strict_support_pixel_gate=12.0,
                legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
                raw_slot_estimated_geometry_scope=common.GEOMETRY_SCOPE,
                female_endpoint_frames=list(metrics.old.old.ANCHOR_FRAMES),
                female_evaluation_interval=[START, MASK_START],
                mask_scoring_starts=MASK_START,
                sparse_execution_starts=SPARSE_START,
                lower_z_mm=-6.0,
                new_boundary_used=False,
                thresholds_retuned=False,
                instance_cost_used_as_independent_accuracy=False,
                checkpoint_payload_reads=0,
                raw_video_reads=0,
                masks_decoded=0,
                GPU_used=False,
                all_three_physical_ID_truth=False,
                automatic_adoption=False,
            ),
        )
        out.write("producer_lineage.json", lineage)
        methods = {}
        for arm, candidate in candidates.items():
            out.write(f"candidates/{arm}.npz", candidate)
            part = {
                k: candidate[k][START - FIRST :]
                for k in ("frame_idx", "pose_xyz", "observed", "sources")
            }

            def on_block(meta, arrays, arm=arm):
                a, b = meta["frame_range"]
                out.write(f"support/{arm}_{a:06d}-{b:06d}.npz", arrays)
                print(json.dumps(dict(event="strict_support_block", arm=arm, **meta)), flush=True)

            strict = support.verify_support_partitioned(
                part, nodes, rig, block_frames=CHUNK, on_block=on_block
            )
            values = {}
            for name, (a, b) in INTERVALS.items():
                value, motion, geometry = metrics.interval_metrics(candidate, seed, a, b)
                values[name] = value
                out.write(f"motion/{arm}_{name}.json", motion)
                if name == "full":
                    out.write(f"geometry/{arm}.npz", geometry)
            methods[arm] = dict(
                intervals=values,
                all_native_strict_support=strict,
                female_first_minute=metrics.old.prior.female_scores(candidate, anchors),
            )
        control, variant = (candidates[a] for a in ARMS)
        differences = {}
        for name, (a, b) in INTERVALS.items():
            value, rows, arrays = metrics.differences(control, variant, a, b)
            differences[name] = value
            if name == "full":
                out.write("differences/direct_gain_loss.json", rows)
                out.write("differences/all_native_transitions.npz", arrays)
        prefix, arrays = selected_prefix_changes(control, variant)
        out.write("differences/selected_prefix_paths.npz", arrays)
        geometry_summary, geometry_arrays = geometry_deltas(control, variant)
        out.write("differences/raw_slot_geometry.npz", geometry_arrays)
        require(
            all(common.signature(candidates[a]) == references[a]["arrays"] for a in ARMS),
            "candidate mutated",
        )
        require(
            all(
                common.signature(dict(points=p, offsets=o)) == node_before[v]
                for v, (p, o) in nodes.data.items()
            ),
            "SL arrays mutated",
        )
        require(not sys.modules["torch"].cuda.is_initialized(), "comparison initialized CUDA")
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                native_frames=49800,
                terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
                candidate_references=references,
                methods=methods,
                support_delta=differences,
                geometry_deltas=geometry_summary,
                selected_prefix_path_changes=prefix,
                legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
                raw_slot_estimated_geometry_scope=common.GEOMETRY_SCOPE,
                female_evaluation_interval=[START, MASK_START],
                requested_female_anchors=11,
                requested_female_intervals=10,
                all_three_physical_ID_truth=False,
                review_modified=False,
                candidate_modified=False,
                human_truth_created=False,
                checkpoint_payload_reads=0,
                raw_video_reads=0,
                masks_decoded=0,
                GPU_used=False,
                tracking_or_fit_performed=False,
                adopted=False,
                limitations=[
                    "Numeric slots are not independently aligned physical IDs.",
                    "SL support and estimated geometry are diagnostics, not biological truth.",
                    "The eleven existing female endpoints cover only the first minute, "
                    "before the new mask policy.",
                    "Both terminal backtraces can change earlier path selection "
                    "without editing retained beam history.",
                    "The -6 mm lower bound remains development padding, not a measured tank floor.",
                    "No new boundary, semantic score, mask/image evaluation or automatic adoption.",
                ],
            ),
        )
        require(source_code() == code, "source closure changed")
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        with (output / "comparison.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (output / "comparison.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-seal-sha256", required=True)
    parser.add_argument("--variant-seal-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.control_seal_sha256, args.variant_seal_sha256, args.output)
