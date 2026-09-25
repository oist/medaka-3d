"""Disabled saved-only five-minute comparison of Sparse -6 and ExpandedDepth -12.

Both successful final116200 stages and their numerical21 milestones are required.
Only candidates and registered SLEAP are decoded; checkpoint39, video and masks
are not opened. Development frames and the eleven first-minute female endpoints
remain limited evidence, never all-three physical-ID or physical-depth truth.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import compare_expanded_depth_instance_10s as depth
import compare_sparse_instance_five_minutes as saved
import numpy as np

common, proof, metrics, support = saved.common, saved.proof, saved.metrics, saved.support
require, WORK, REPO = proof.require, saved.WORK, saved.REPO
FIRST, START, CHANGE, RESUME, STOP, CHUNK = 49800, 66400, 76360, 78020, 116200, 1660
STAGE = "expanded_depth_five_minute_comparison_v1"
ARMS = (saved.ARMS[1], "expanded_depth_instance_rescore")
VARIANT_STAGE = "expanded_depth_five_minute_continuation_v1"
VARIANT_RUNNER = WORK / "continue_expanded_depth_five_minutes.py"
PARENT_ROOTS = {
    ARMS[0]: WORK / "instance_rescore_ten_second_continuation_v1",
    ARMS[1]: WORK / "expanded_depth_instance_rescore_ten_second_v1",
}
PARENT_SHAS = {
    ARMS[0]: "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a",
    ARMS[1]: "0ff5f6a31f020a2f3773176e5af60990ffb6bbf5fc7c67cbf505dd208d6e1d8c",
}
PARENT_COMPARE = WORK / "expanded_depth_instance_ten_second_comparison_v1"
PARENT_COMPARE_SHA = "bb6002ab37441a86daac6a05ed19176b512601e8ad8c30a1df620fb064848f59"
QA = WORK / "expanded_depth_ten_second_independent_qa_v1/qa.json"
QA_SHA = "984f9c0efeb688930a42a3c2ea25d058b1754f4d7e3864e27340c542f7198d8c"
QA_SOURCE_SHA = "d06447f72ef972707761c8ef61bf8ba7e8ccad333fd87ad40606d4b2519e38d0"
QA_SELECTED_SHA = "96f6e9164b2f84384207bdf809a52772396813cc0b4b81cdbcdf827c6954472c"
PINS = {
    "compare_sparse_instance_five_minutes.py": (
        "1ec0893d7dd2bbc746004f71d9b81abc60096cc0173bf44871f56be948e1c8dc"
    ),
    "compare_expanded_depth_instance_10s.py": (
        "628178b5083308df12e1a290111e5c97c0bb1dd11f0000d704be4f9ff8f35c72"
    ),
    "expanded_depth_instance_rescore_tracker.py": (
        "522f4012fb45b2e1fc89850c5448991fa3979193f736d91fe81b5f6d0344fe57"
    ),
}
INTERVALS = {
    "full": (START, STOP),
    **{f"minute_{i + 1}": (START + i * 9960, START + (i + 1) * 9960) for i in range(5)},
    "before_depth_change": (START, CHANGE),
    "development_ten_seconds": (CHANGE, RESUME),
    "continuation_after_development": (RESUME, STOP),
}
LEGACY_DIFFERENCE_FIELD_ARMS = {
    "original_sources": ARMS[0],
    "original_review": ARMS[0],
    "tail_sources": ARMS[1],
    "tail_review": ARMS[1],
}
EXECUTION_READY = True


def source_code():
    code = saved.source_code()
    require(str(Path(__file__).resolve()) in code, "new comparison source absent")
    for name, digest in PINS.items():
        require(proof.sha(WORK / name) == digest, "frozen helper/source differs: " + name)
        code[str(WORK / name)] = digest
    return code


def variant_completion(summary):
    common.exact_fields(
        summary,
        dict(
            schema=VARIANT_STAGE,
            status="ok",
            scientific_interval=[START, STOP],
            process_interval=[RESUME, STOP],
            retained_interval=[FIRST, STOP],
            native_frames=STOP - START,
            process_frames=STOP - RESUME,
            initialize_count=1,
            restore_count=6,
            external_pose_resets=0,
            new_begin_calls=0,
            finish_calls=1,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            finish_matches_checkpoint_backtrace=True,
            diagnostic_frames=STOP - RESUME,
            policy_transition=False,
            physical_ID_truth=False,
            physical_depth_truth=False,
            automatic_adoption=False,
            parent_tracking_sha256=PARENT_SHAS[ARMS[1]],
        ),
        "completed expanded-depth five-minute stage required",
    )
    stops = list(range(RESUME + CHUNK, STOP + CHUNK, CHUNK))
    require(len(summary["checkpoints"]) == len(stops), "checkpoint count differs")
    for row, stop in zip(summary["checkpoints"], stops, strict=True):
        common.exact_fields(
            row,
            dict(
                arrays=39,
                previous_frame=stop - 1,
                processed_frames=stop - FIRST,
                initialize_count=1,
                restore_count=6,
            ),
            "checkpoint lifecycle differs",
        )


def bind_variant_sources(plan, seal, runner_sha, evidence):
    require(proof.valid_sha(runner_sha), "explicit reviewed source SHA required")
    require(
        plan["source_sha256"].get(str(VARIANT_RUNNER)) == runner_sha,
        "CLI reviewed runner source differs",
    )
    require(
        plan["source_sha256"].get(str(WORK / "expanded_depth_instance_rescore_tracker.py"))
        == PINS["expanded_depth_instance_rescore_tracker.py"],
        "reviewed depth tracker differs",
    )
    for path, digest in plan["source_sha256"].items():
        require(
            saved.is_producer_source(path, digest) and seal["inputs"].get(path) == digest,
            "producer consumed source edge differs",
        )
        evidence.add(common.local(path), digest)


def load_variant_headers(seal_sha, runner_sha, evidence):
    require(
        proof.valid_sha(seal_sha) and proof.valid_sha(runner_sha),
        "explicit final seal and reviewed source SHAs required",
    )
    root = WORK / VARIANT_STAGE
    seal = common.read(root / "tracking.provenance.json", evidence, seal_sha)
    common.exact_fields(seal, dict(stage=VARIANT_STAGE, status="ok"), "final depth tracking failed")
    docs = {
        name: saved.member_doc(root, seal, name + ".json", evidence)
        for name in ("summary", "plan", "input_index", "state_restore")
    }
    plan, summary, index = docs["plan"], docs["summary"], docs["input_index"]
    variant_completion(summary)
    common.exact_fields(
        plan,
        dict(
            schema=VARIANT_STAGE,
            scientific_interval=[START, STOP],
            process_interval=[RESUME, STOP],
            retained_interval=[FIRST, STOP],
            initialize_count=1,
            restore_count=6,
            new_begin_calls=0,
            external_pose_resets=0,
            precision="float64",
            policy_transition=False,
            semantic_landmark_term_used=False,
            physical_ID_truth=False,
            physical_depth_truth=False,
        ),
        "new depth producer plan differs",
    )
    binding = plan["checkpoint_binding"]
    require(
        binding["schema"] == VARIANT_STAGE + "_checkpoint"
        and binding["source_sha256"] == plan["source_sha256"]
        and binding["original_binding"] == plan["original_binding"] == index["original_binding"]
        and binding["scientific_interval"] == [START, STOP]
        and binding["process_interval"] == [RESUME, STOP]
        and binding["retained_interval"] == [FIRST, STOP]
        and binding["policy_transition"] is False,
        "depth checkpoint binding differs",
    )
    bind_variant_sources(plan, seal, runner_sha, evidence)
    chunks = index["prefix_chunks"]
    require(
        set(index) == {"original_binding", "prefix_chunks"} and len(chunks) == 40,
        "40chunk index schema differs",
    )
    for i, row in enumerate(chunks):
        common.exact_fields(
            row,
            dict(index=i, start=FIRST + i * CHUNK, stop=FIRST + (i + 1) * CHUNK),
            "source chunks reordered",
        )
    manifest, metadata = saved.checkpoint_header(root, seal, STOP, evidence)
    require(manifest["bindings"] == binding, "final depth context/source binding differs")
    common.exact_fields(
        metadata,
        dict(
            initialize_count=1,
            restore_count=6,
            previous_frame=STOP - 1,
            processed_frames=STOP - FIRST,
        ),
        "final depth lifecycle differs",
    )
    row = summary["checkpoints"][-1]
    require(
        row["path"] == str(root / "checkpoints/through_116200")
        and row["manifest_sha256"]
        == seal["outputs"]["checkpoints/through_116200/manifest.json"]["sha256"]
        and row["payload_bytes"] == manifest["payload"]["bytes"],
        "final checkpoint descriptor differs",
    )
    milestone = saved.member_doc(root, seal, "milestones/through_116200/summary.json", evidence)
    common.exact_fields(
        milestone,
        dict(
            schema=VARIANT_STAGE + "_milestone",
            status="ok",
            horizon_stop=STOP,
            scientific_interval=[START, STOP],
            retained_interval=[FIRST, STOP],
            process_interval=[RESUME, STOP],
            native_frames=STOP - START,
            initialize_count=1,
            restore_count=6,
            external_pose_resets=0,
            new_begin_calls=0,
            finish_calls_so_far=0,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            policy_transition=False,
            whole_process_completion_pending=True,
        ),
        "saved final milestone differs",
    )
    require(
        milestone["checkpoint"] == row
        and milestone["candidate"] == seal["outputs"]["milestones/through_116200/candidate.npz"]
        and milestone["source_sha256"] == plan["source_sha256"]
        and milestone["original_binding"] == plan["original_binding"],
        "milestone source/candidate binding differs",
    )
    docs.update(metadata=metadata, manifest=manifest, milestone=milestone)
    return dict(
        arm=ARMS[1], root=root, seal=seal, docs=docs, seal_sha256=seal_sha, runner_sha256=runner_sha
    )


def manifest_signatures(manifest):
    require(len(manifest["arrays"]) == 39, "all39 manifest signatures required")
    return {
        k: dict(dtype=v["dtype"]["str"], shape=v["shape"], sha256=v["bytes_sha256"])
        for k, v in manifest["arrays"].items()
    }


def parent_headers(arm, bundle, evidence):
    root, expected = PARENT_ROOTS[arm], PARENT_SHAS[arm]
    require(
        bundle["seal"]["inputs"].get(str(root / "tracking.provenance.json")) == expected,
        "successful78020 parent input edge missing",
    )
    seal = common.read(root / "tracking.provenance.json", evidence, expected)
    common.exact_fields(seal, dict(stage=root.name, status="ok"), "parent stage failed")
    summary = saved.member_doc(root, seal, "summary.json", evidence)
    manifest, metadata = saved.checkpoint_header(root, seal, RESUME, evidence)
    binding = bundle["docs"]["plan"]["checkpoint_binding"]
    require(
        binding["parent_tracking_sha256"] == expected
        and binding["parent_checkpoint_sha256"]
        == summary["checkpoint_manifest_sha256"]
        == seal["outputs"]["checkpoints/through_078020/manifest.json"]["sha256"],
        "parent checkpoint edge differs",
    )
    require(
        summary["parent_checkpoint_sha256"] == common.PARENT_SHA
        and summary["parent_audit_sha256"] == common.PARENT_AUDIT_SHA,
        "same audited76360 branch point required",
    )
    common.exact_fields(
        metadata,
        dict(
            initialize_count=1,
            restore_count=5,
            previous_frame=RESUME - 1,
            processed_frames=RESUME - FIRST,
        ),
        "parent restore lifecycle differs",
    )
    return dict(metadata=metadata, arrays=manifest_signatures(manifest), summary=summary)


def validate_restorations(control, variant, parents):
    """Compare small codec byte signatures and actual metadata; never decode39."""
    dc, dv = control["docs"], variant["docs"]
    old, new = dc["metadata"]["context"], dv["metadata"]["context"]
    require(
        old["implementation"] == "SparseInstanceRescoreContinuousTracker"
        and new["implementation"] == "ExpandedDepthInstanceRescoreContinuousTracker",
        "actual candidate classes differ",
    )
    imported = dc["state_import"]
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
            previous_frame=RESUME - 1,
            processed_frames=RESUME - FIRST,
            effective_from_native=RESUME,
            review_cleared_by_migration=False,
            backend_applied_retroactively_to_prior_history=False,
        ),
        "sparse parent migration proof differs",
    )
    pm = parents[ARMS[0]]["metadata"]
    require(
        imported["arrays"] == parents[ARMS[0]]["arrays"]
        and imported["original_metadata"] == pm
        and imported["old_context"] == pm["context"]
        and imported["new_context"] == old,
        "sparse source39/metadata differs",
    )
    adapted = copy.deepcopy(pm)
    adapted.update(context=old, restore_count=6)
    require(imported["imported_metadata"] == adapted, "sparse restore changed metadata")
    reconstructed = copy.deepcopy(old)
    backend = reconstructed.pop("sparse_distance_backend")
    reconstructed["implementation"] = pm["context"]["implementation"]
    require(
        reconstructed == pm["context"] and backend["effective_from_native"] == RESUME,
        "undeclared sparse parent context change",
    )
    restored = dv["state_restore"]
    common.exact_fields(
        restored,
        dict(
            schema=VARIANT_STAGE + "_same_policy_restore",
            status="ok",
            all39_byte_preserved=True,
            source_inputs_unchanged=True,
            context_unchanged=True,
            initialize_count=1,
            source_restore_count=5,
            restore_count=6,
            new_begin_calls=0,
            policy_transition=False,
            review_cleared=False,
            resume_native=RESUME,
        ),
        "same-context depth restoration missing",
    )
    pm = parents[ARMS[1]]["metadata"]
    require(
        restored["arrays"] == parents[ARMS[1]]["arrays"]
        and restored["original_metadata"] == pm
        and new == pm["context"],
        "depth parent39/context differs",
    )
    require(
        restored["restored_metadata"] == {**pm, "restore_count": 6},
        "depth restore rewrote metadata",
    )
    adapted = copy.deepcopy(new)
    policy = adapted.pop("expanded_depth_policy")
    require(
        policy == parents[ARMS[1]]["summary"]["expanded_depth_policy"]
        and policy["effective_from_native"] == CHANGE
        and policy["expanded_lower_z_mm"] == -12.0,
        "depth policy differs",
    )
    require(
        new["lower"] == support.COMMON.digest_array(np.array([0.0, 0.0, -12.0]))
        and old["lower"] == support.COMMON.digest_array(np.array([0.0, 0.0, -6.0]))
        and new["recovery_config"]["lower_mm"] == [0.0, 0.0, -12.0]
        and old["recovery_config"]["lower_mm"] == [0.0, 0.0, -6.0],
        "declared depth domains differ",
    )
    require(
        new["sparse_distance_backend"]["effective_from_native"] == CHANGE,
        "expanded sparse execution start differs",
    )
    adapted["implementation"] = old["implementation"]
    adapted["lower"] = old["lower"]
    adapted["recovery_config"]["lower_mm"][2] = -6.0
    adapted["sparse_distance_backend"]["effective_from_native"] = RESUME
    require(adapted == old, "undeclared mask/calibration/fitter/support/anatomy difference")
    require(
        old["instance_rescore_policy"]["effective_from_native"] == CHANGE,
        "mask policy start differs",
    )
    for docs, ctx in ((dc, old), (dv, new)):
        for key in ("scoring_policy", "sparse_distance_backend"):
            context_key = "instance_rescore_policy" if key == "scoring_policy" else key
            require(
                docs["plan"]["checkpoint_binding"][key] == ctx[context_key],
                "checkpoint scientific policy differs",
            )
    for doc in (dv["plan"], dv["summary"], dv["plan"]["checkpoint_binding"], dv["milestone"]):
        require(
            doc["scoring_policy"] == new["instance_rescore_policy"]
            and doc["sparse_distance_backend"] == new["sparse_distance_backend"]
            and doc["expanded_depth_policy"] == policy,
            "depth stage policies disagree",
        )
    return dict(
        compared_parent_signatures={a: p["arrays"] for a, p in parents.items()},
        parent_tracking_sha256=PARENT_SHAS,
        depth_policy=policy,
        restore_count_by_arm=dict.fromkeys(ARMS, 6),
        same_audited76360_parent=True,
        full39_payload_reads=0,
        immutable_history_proofs_reused_not_recomputed=True,
    )


def load_lineage(bundles, evidence):
    control, variant = (bundles[a] for a in ARMS)
    control["docs"]["state_import"] = saved.member_doc(
        control["root"], control["seal"], "state_import.json", evidence
    )
    parents = {a: parent_headers(a, b, evidence) for a, b in bundles.items()}
    lineage = validate_restorations(control, variant, parents)
    serial = saved.member_doc(
        control["root"], control["seal"], "serial_dense_sparse_qualification.json", evidence
    )
    common.exact_fields(
        serial,
        dict(
            status="ok",
            native_interval=[RESUME, RESUME + 2],
            all39_BYTEEXACT=True,
            initialize_count=1,
            restore_count=6,
            full_five_minute_equivalence_proven=False,
        ),
        "existing sparse serial proof missing",
    )
    require(
        serial["backend"] == control["docs"]["metadata"]["context"]["sparse_distance_backend"]
        and len(serial["compared_arrays"]) == 39,
        "serial proof backend differs",
    )
    qa_reference = variant["docs"]["plan"]["parent_qa"]
    for doc in (
        variant["docs"]["summary"],
        variant["docs"]["plan"]["checkpoint_binding"],
        variant["docs"]["milestone"],
    ):
        require(doc["parent_qa"] == qa_reference, "parent QA references differ")
    sources = (
        (QA, QA_SHA),
        (WORK / "qa_expanded_depth_saved_10s.py", QA_SOURCE_SHA),
        (QA.parent / "selected_candidates.npz", QA_SELECTED_SHA),
        (PARENT_COMPARE / "comparison.provenance.json", PARENT_COMPARE_SHA),
    )
    for path, digest in sources:
        require(
            variant["seal"]["inputs"].get(str(path)) == digest,
            "successful parent QA source edge differs",
        )
        evidence.add(path, digest)
    qa = proof.read_json(QA)
    require(
        qa["status"] == "ok"
        and qa["producer_seal_sha256"] == PARENT_SHAS[ARMS[1]]
        and qa["comparison_seal_sha256"] == PARENT_COMPARE_SHA
        and qa["qa_source_sha256"] == QA_SOURCE_SHA
        and qa["selected_candidates"]["sha256"] == QA_SELECTED_SHA
        and qa_reference["sha256"] == QA_SHA
        and qa_reference["source_sha256"] == QA_SOURCE_SHA
        and qa_reference["selected_candidates"] == qa["selected_candidates"],
        "parent QA binding differs",
    )
    prior_seal = proof.read_json(PARENT_COMPARE / "comparison.provenance.json")
    require(
        prior_seal["status"] == "ok" and prior_seal["stage"] == PARENT_COMPARE.name,
        "prior development comparison failed",
    )
    require(
        all(
            prior_seal["inputs"].get(str(PARENT_ROOTS[a] / "tracking.provenance.json"))
            == PARENT_SHAS[a]
            for a in ARMS
        ),
        "prior comparison used different parents",
    )
    lineage.update(parent_qa=qa_reference, prior_development_comparison_sha256=PARENT_COMPARE_SHA)
    return lineage


def observation_binding(bundles):
    a, b = (bundles[k]["docs"] for k in ARMS)
    require(
        a["input_index"] == b["input_index"]
        and set(a["input_index"]) == {"original_binding", "prefix_chunks"},
        "same actual40 chunk records required",
    )
    binding = a["input_index"]["original_binding"]
    require(
        a["plan"]["original_binding"] == b["plan"]["original_binding"] == binding,
        "same observation/calibration binding required",
    )
    return binding


def load_observations(bundles, evidence):
    binding = observation_binding(bundles)
    seed_info = binding["inputs"]["seed"]
    shared = seed_info["SL_and_calibration_sha256"]
    require(len(shared) == 7, "SL3/calibration4 required")
    for path, digest in shared.items():
        require(
            all(b["seal"]["inputs"].get(path) == digest for b in bundles.values()),
            "actual SL/calibration consumption differs",
        )
        evidence.add(common.local(path), digest)
    paths = {
        view: [
            Path(p)
            for p in shared
            if Path(p).name == metrics.old.prior.EXPERIMENT + "__" + view + ".npz"
        ]
        for view in metrics.old.prior.VIEWS
    }
    require(all(len(p) == 1 for p in paths.values()), "unique SL per view required")
    calibration = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    require(len(calibration) == 1, "one rig required")
    rig = metrics.old.prior.load_rig(calibration[0])
    seed = np.asarray(
        common.read(seed_info["registration_path"], evidence, seed_info["registration_sha256"])[
            "pose_xyz"
        ],
        np.float64,
    )
    for bundle in bundles.values():
        ctx = bundle["docs"]["metadata"]["context"]
        require(
            ctx["initial_pose"] == support.COMMON.digest_array(seed)
            and ctx["cameras"] == support.COMMON.normalize([asdict(c) for c in rig.cameras]),
            "registered calibration/seed differs",
        )
    nodes = metrics.old.prior.RegisteredNodes(
        {v: p[0] for v, p in paths.items()}, np.arange(START, STOP)
    )
    return nodes, rig, seed, shared


def load_candidates(bundles, evidence):
    candidates, references = {}, {}
    for arm, bundle in bundles.items():
        path = common.bound_member(bundle["root"], bundle["seal"], "candidate.npz", evidence)
        with np.load(path, allow_pickle=False) as archive:
            require(
                len(archive.files) == len(set(archive.files)) == 21, "candidate archive membership"
            )
            candidate = {k: archive[k] for k in archive.files}
        metrics.validate_candidate(candidate, FIRST, STOP)
        actual = common.signature(candidate)
        require(
            actual == bundles[arm]["docs"]["summary"]["candidate_arrays"],
            "all21 final candidate signature differs",
        )
        milestone = common.bound_member(
            bundle["root"], bundle["seal"], "milestones/through_116200/candidate.npz", evidence
        )
        with np.load(milestone, allow_pickle=False) as archive:
            require(
                len(archive.files) == len(set(archive.files)) == 21
                and set(archive.files) == set(candidate),
                "milestone archive membership",
            )
            require(
                all(proof.signature(archive[k]) == actual[k] for k in candidate),
                "final differs from116200 milestone",
            )
        if arm == ARMS[1]:
            require(
                actual == bundles[arm]["docs"]["milestone"]["candidate_arrays"],
                "all21 milestone signature differs",
            )
        for array in candidate.values():
            array.setflags(write=False)
        candidates[arm] = candidate
        references[arm] = dict(
            path=str(path),
            sha256=evidence.paths[str(path)],
            milestone_path=str(milestone),
            milestone_sha256=evidence.paths[str(milestone)],
            terminal_horizon=STOP,
            arrays=actual,
            tracking_seal_sha256=bundle["seal_sha256"],
        )
    return candidates, references


def interval_report(candidate, seed, start, stop):
    values, motion, geometry = metrics.interval_metrics(candidate, seed, start, stop)
    z, z_arrays = depth.z_diagnostics(candidate, start, stop)
    # Keep strict out-of-domain counts and the requested inclusive old boundary.
    at_or_below = z_arrays["z_mm"] <= -6.0
    z_arrays["at_or_below_old_lower"] = at_or_below
    for label, eligible in (
        ("all_estimated_including_latent", np.ones(at_or_below.shape, bool)),
        ("directly_observed", z_arrays["directly_observed"]),
    ):
        mask = at_or_below & eligible
        z["subsets"][label]["counts"]["at_or_below_old_lower"] = dict(
            total=int(mask.sum()), by_slot_node=mask.sum(0).tolist()
        )
    return dict(support_and_geometry=values, z_distribution=z), motion, geometry, z_arrays


def geometry_delta(control, variant, start, stop):
    value, arrays = saved.geometry_deltas(control, variant, start, stop)
    for field in ("speed_by_arm", "acceleration_by_arm", "chest_pair_distance_by_arm"):
        value[field] = {new: value[field][old] for old, new in zip(saved.ARMS, ARMS, strict=True)}
    value["array_axis0_arm_order"] = list(ARMS)
    return value, arrays


def run(control_sha, variant_sha, variant_runner_sha, output):
    require(EXECUTION_READY, "source review and successful final producers required")
    require(
        all(proof.valid_sha(value) for value in (control_sha, variant_sha, variant_runner_sha)),
        "explicit seal/source SHA required",
    )
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        require(not sys.modules["torch"].cuda.is_initialized(), "CPU only")
        code = source_code()
        for path, digest in code.items():
            evidence.add(path, digest)
        bundles = {
            ARMS[0]: saved.load_headers(saved.ARMS[1], control_sha, evidence),
            ARMS[1]: load_variant_headers(variant_sha, variant_runner_sha, evidence),
        }
        lineage = load_lineage(bundles, evidence)
        observation_binding(bundles)
        candidates, references = load_candidates(bundles, evidence)
        nodes, rig, seed, shared = load_observations(bundles, evidence)
        node_before = {
            v: common.signature(dict(points=p, offsets=o)) for v, (p, o) in nodes.data.items()
        }
        anchors = saved.female_helpers.female_anchors(shared, evidence)
        scope = dict(
            native_interval=[START, STOP],
            native_frames=STOP - START,
            intervals=INTERVALS,
            terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
            candidate_references=references,
            legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
            raw_slot_estimated_geometry_scope=common.GEOMETRY_SCOPE,
            female_evaluation_interval=[START, CHANGE],
            requested_female_anchors=11,
            depth_and_mask_change_native=CHANGE,
            sparse_backend_start_by_arm={ARMS[0]: RESUME, ARMS[1]: CHANGE},
            lower_z_mm_by_arm={ARMS[0]: -6.0, ARMS[1]: -12.0},
            boundary_band_inclusive_mm=[-12.0, -11.75],
            boundary_band_is_acceptance_gate=False,
            development_frames=list(depth.DEVELOPMENT_FRAMES),
            development_frames_are_holdout=False,
            strict_support_ray_mm=2.0,
            strict_support_pixel_gate=12.0,
            mask_payload_lineage_independently_rehashed=False,
            masks_decoded=0,
            checkpoint_payload_reads=0,
            raw_video_reads=0,
            GPU_used=False,
            physical_ID_truth=False,
            physical_depth_truth=False,
            thresholds_retuned=False,
            instance_cost_used_as_independent_accuracy=False,
            automatic_adoption=False,
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                arms=list(ARMS),
                source_sha256=code,
                variant_runner_sha256=variant_runner_sha,
                **scope,
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

            def block(meta, arrays, arm=arm):
                a, b = meta["frame_range"]
                out.write(f"support/{arm}_{a:06d}-{b:06d}.npz", arrays)
                print(json.dumps(dict(event="strict_support_block", arm=arm, **meta)), flush=True)

            strict = support.verify_support_partitioned(
                part, nodes, rig, block_frames=CHUNK, on_block=block
            )
            intervals = {}
            for name, (a, b) in INTERVALS.items():
                value, motion, geometry, z = interval_report(candidate, seed, a, b)
                intervals[name] = value
                out.write(f"motion/{arm}_{name}.json", motion)
                if name == "full":
                    out.write(f"geometry/{arm}.npz", geometry)
                    out.write(f"z/{arm}.npz", z)
            methods[arm] = dict(
                intervals=intervals,
                all_native_strict_support=strict,
                female_first_minute=metrics.old.prior.female_scores(candidate, anchors),
            )
            indices = np.array(depth.DEVELOPMENT_FRAMES) - FIRST
            out.write(
                f"development_frames/{arm}.npz", {k: v[indices] for k, v in candidate.items()}
            )
        control, variant = (candidates[a] for a in ARMS)
        differences, losses, geometry_summaries = {}, {}, {}
        for name, (a, b) in INTERVALS.items():
            value, rows, arrays = metrics.differences(control, variant, a, b)
            differences[name] = value
            losses[name] = depth.support_loss_runs(arrays["frame_idx"], arrays["support_loss"])
            geometry, geo_arrays = geometry_delta(control, variant, a, b)
            geometry_summaries[name] = geometry
            if name == "full":
                out.write("differences/direct_gain_loss.json", rows)
                out.write("differences/all_native_transitions.npz", arrays)
                out.write("differences/raw_slot_geometry.npz", geo_arrays)
        out.write("differences/contiguous_support_losses.json", losses)
        prefix, arrays = metrics.horizon_changes(control, variant, FIRST, CHANGE)
        prefix.pop("snapshot_horizon")
        prefix.pop("final_horizon")
        prefix.update(
            terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
            interpretation=(
                "Same116200 selected earlier prefix paths, "
                "separate from immutable retained beam history."
            ),
        )
        out.write("differences/selected_prefix_paths.npz", arrays)
        require(
            all(common.signature(candidates[a]) == references[a]["arrays"] for a in ARMS),
            "candidate mutated",
        )
        require(
            all(
                common.signature(dict(points=p, offsets=o)) == node_before[v]
                for v, (p, o) in nodes.data.items()
            ),
            "SL inputs mutated",
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                methods=methods,
                support_delta=differences,
                contiguous_support_losses=losses,
                geometry_deltas=geometry_summaries,
                selected_prefix_path_changes=prefix,
                candidate_modified=False,
                review_modified=False,
                all_three_physical_ID_truth=False,
                human_truth_created=False,
                adopted=False,
                limitations=[
                    "Numeric slots and SLEAP support are not independently aligned biological IDs.",
                    "Eleven female endpoints cover only the first minute, before the depth change.",
                    "The development ten seconds include observations used to choose -12 mm.",
                    "The later continuation is a robustness check, not labeled identity truth.",
                    "Both lower bounds are engineering domains, not a measured tank floor.",
                    "Both future backtraces can change earlier selected paths "
                    "without rewriting history.",
                    "Forty consumed chunk records are compared; "
                    "mask payloads are not rehashed here.",
                ],
                **scope,
            ),
        )
        require(
            source_code() == code and not sys.modules["torch"].cuda.is_initialized(),
            "source/CUDA state changed",
        )
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
    parser.add_argument("--variant-runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.control_seal_sha256, args.variant_seal_sha256, args.variant_runner_sha256, args.output)
