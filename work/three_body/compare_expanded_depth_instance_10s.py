"""Same-parent/same-78020 saved comparison of Dense -6 and ExpandedSparse -12.

Candidate21, registered SL and small sealed metadata only. No retained39 state
decode, mask/video read, fitting, GPU execution or biological-ID certification.
Development-selected frames are reported alongside the complete1660 interval.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import compare_instance_rescore_10s as common
import numpy as np

proof, metrics, support = common.proof, common.metrics, common.support
require = proof.require
WORK, REPO = common.WORK, common.REPO
FIRST, START, STOP = 49800, 76360, 78020
STAGE = "expanded_depth_instance_ten_second_comparison_v1"
ARMS = ("dense_instance_lower_minus6", "expanded_sparse_instance_lower_minus12")
CONTROL_STAGE = "instance_rescore_ten_second_continuation_v1"
VARIANT_STAGE = "expanded_depth_instance_rescore_ten_second_v1"
CONTROL_SHA = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
CONTROL_RUNNER_SHA = "91a66646f89a6abb59b6d60bc1b357b383167c21eb9d40bc47491802d05e9544"
VARIANT_RUNNER = WORK / "run_expanded_depth_instance_rescore_10s.py"
VARIANT_RUNNER_SHA = "b1a16cedb9a4c7cacf92d96c6fa496f1d0a143e27b023fdc5b022126923f49aa"
TRACKER_SHA = "522f4012fb45b2e1fc89850c5448991fa3979193f736d91fe81b5f6d0344fe57"
QUALIFICATION_SHA = "941ca436a5e59fca5a63794483d315da3950402aafdb4ce13c34630e98014125"
COMPARISON_PLAN = WORK / "EXPANDED_DEPTH_TEN_SECOND_COMPARISON_PLAN.md"
COMPARISON_PLAN_SHA = "8ff420a580f49b4b9a8652d3e79f29377dd9760f10742cfd738be22bb08cbb65"
DEVELOPMENT_FRAMES = (77709, 77723, 77880)
BOUNDARY_BAND_MM = 0.25
EXECUTION_CONTRACT_SHA = "097277d2a28c43edccafba39f96c779890bf553b8d83996051b2bf98198814b4"
PINS = {
    "compare_instance_rescore_10s.py": (
        "bb7653e5f7d71f2b4c078418dee93433fb4a1fc3fdefd7f38ef9c4694ae9040a"
    ),
    "expanded_depth_instance_rescore_tracker.py": TRACKER_SHA,
    "fast_tail_two_minute_metrics.py": common.METRICS_SHA,
    "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md": EXECUTION_CONTRACT_SHA,
    **common.REPORTING_PINS,
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
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.is_relative_to(REPO) and ".venv" not in path.parts and path.suffix == ".py":
                paths.add(path)
    code = {str(path): proof.sha(path) for path in sorted(paths)}
    for name, digest in PINS.items():
        path = WORK / name
        require(proof.sha(path) == digest, "frozen comparison source differs: " + name)
        code[str(path)] = digest
    return code


def document(root, seal, name, evidence):
    return json.loads(common.bound_member(root, seal, name, evidence).read_text("utf8"))


def is_producer_source(path, digest):
    path = Path(path)
    return path.suffix == ".py" or (
        path.resolve() == WORK / "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md"
        and digest == EXECUTION_CONTRACT_SHA
    )


def completion(summary, stage):
    common.exact_fields(
        summary,
        dict(
            schema=stage,
            status="ok",
            process_interval=[START, STOP],
            native_frames=1660,
            retained_interval=[FIRST, STOP],
            parent_audit_sha256=common.PARENT_AUDIT_SHA,
            parent_checkpoint_sha256=common.PARENT_SHA,
            initialize_count=1,
            restore_count=5,
            new_begin_calls=0,
            external_pose_resets=0,
            finish_calls=1,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            finish_matches_checkpoint_backtrace=True,
            diagnostic_frames=1660,
            physical_ID_truth=False,
            automatic_adoption=False,
        ),
        "complete same78020 tracking and unchanged history proof required",
    )


def load_headers(stage, seal_sha, runner_sha, evidence):
    require(stage in (CONTROL_STAGE, VARIANT_STAGE), "registered producer stage required")
    require(
        proof.valid_sha(seal_sha) and proof.valid_sha(runner_sha),
        "explicit sealed source SHAs required",
    )
    root = WORK / stage
    seal = common.read(root / "tracking.provenance.json", evidence, seal_sha)
    common.exact_fields(seal, dict(stage=stage, status="ok"), "final tracking seal failed")
    docs = {
        name: document(root, seal, name + ".json", evidence)
        for name in ("plan", "summary", "state_import", "input_index")
    }
    plan, summary = docs["plan"], docs["summary"]
    completion(summary, stage)
    common.exact_fields(
        plan,
        dict(
            schema=stage,
            process_interval=[START, STOP],
            retained_interval=[FIRST, STOP],
            parent_audit_sha256=common.PARENT_AUDIT_SHA,
            initialize_count=1,
            restore_count=5,
            new_begin_calls=0,
            external_pose_resets=0,
            precision="float64",
            physical_ID_truth=False,
            automatic_adoption=False,
        ),
        "producer plan differs",
    )
    binding = plan["checkpoint_binding"]
    require(
        binding["schema"] == stage + "_checkpoint"
        and binding["original_binding"]
        == plan["original_binding"]
        == docs["input_index"]["original_binding"]
        and binding["source_sha256"] == plan["source_sha256"]
        and binding["parent_checkpoint_sha256"] == common.PARENT_SHA
        and binding["parent_audit_sha256"] == common.PARENT_AUDIT_SHA
        and binding["retained_interval"] == [FIRST, STOP]
        and binding["process_interval"] == [START, STOP],
        "source/parent binding differs",
    )
    runner = VARIANT_RUNNER if stage == VARIANT_STAGE else WORK / "run_instance_rescore_10s.py"
    require(
        plan["source_sha256"].get(str(runner)) == runner_sha, "reviewed producer source differs"
    )
    for path, digest in plan["source_sha256"].items():
        require(
            is_producer_source(path, digest) and seal["inputs"].get(path) == digest,
            "producer consumed source edge differs",
        )
        evidence.add(common.local(path), digest)
    if stage == VARIANT_STAGE:
        require(
            runner_sha == VARIANT_RUNNER_SHA
            and plan["source_sha256"].get(str(WORK / "expanded_depth_instance_rescore_tracker.py"))
            == TRACKER_SHA,
            "reviewed depth tracker source differs",
        )
    chunks = docs["input_index"]["prefix_chunks"]
    require(len(chunks) == 17, "17 consumed source chunks required")
    for i, row in enumerate(chunks):
        common.exact_fields(
            row,
            dict(index=i, start=FIRST + i * 1660, stop=FIRST + (i + 1) * 1660),
            "consumed native chunks differ",
        )
    prefix = "checkpoints/through_078020/"
    manifest = document(root, seal, prefix + "manifest.json", evidence)
    meta = document(root, seal, prefix + "metadata.json", evidence)
    common.exact_fields(
        manifest,
        dict(schema="medaka_tracker_checkpoint_v1", status="ok"),
        "successful retained checkpoint header required",
    )
    require(
        manifest["bindings"] == binding
        and seal["outputs"][prefix + "manifest.json"]["sha256"]
        == summary["checkpoint_manifest_sha256"],
        "final checkpoint binding differs",
    )
    for field, name in (("metadata", "metadata.json"), ("payload", "arrays.npz")):
        row, declared = seal["outputs"][prefix + name], manifest[field]
        require(
            declared["file"] == name
            and declared["sha256"] == row["sha256"]
            and declared["bytes"] == row["size_bytes"]
            and Path(row["path"]).resolve() == root / prefix / name,
            "checkpoint descriptor differs; retained39 is not read",
        )
    common.exact_fields(
        meta,
        dict(
            previous_frame=STOP - 1,
            processed_frames=STOP - FIRST,
            initialize_count=1,
            restore_count=5,
        ),
        "checkpoint lifecycle differs",
    )
    require(
        summary["checkpoint"]["manifest_sha256"] == summary["checkpoint_manifest_sha256"],
        "summary checkpoint identity differs",
    )
    docs.update(manifest=manifest, metadata=meta)
    return dict(root=root, seal=seal, docs=docs, seal_sha256=seal_sha, runner_sha256=runner_sha)


def validate_domains(control, variant):
    """Identical audited FastTail state; only explicit mask/backend/depth contexts differ."""
    dc, dv = control["docs"], variant["docs"]
    a, b = dc["state_import"], dv["state_import"]
    old = a["old_context"]
    common.validate_import(a, old, dc["metadata"], dc["plan"]["new_scoring_policy"])
    common.exact_fields(
        b,
        dict(
            schema="expanded_depth_fast_tail_state_import_v1",
            status="ok",
            all_arrays_byte_preserved=True,
            original_metadata_unchanged=True,
            logical_initialize_count=1,
            new_begin_calls=0,
            added_restore_count=1,
            processed_frames=START - FIRST,
            previous_frame=START - 1,
            effective_from_native=START,
            depth_policy_applied_retroactively_to_prior_history=False,
            review_cleared_by_migration=False,
            companion_diagnostics_required_each_future_frame=True,
        ),
        "expanded depth single-boundary import proof differs",
    )
    require(
        len(b["arrays"]) == 39
        and b["arrays"] == a["arrays"]
        and b["old_context"] == old
        and b["original_metadata"] == a["original_metadata"]
        and b["new_context"] == dv["metadata"]["context"],
        "same actual all39 parent required",
    )
    ctx = b["new_context"]
    require(
        ctx["implementation"] == "ExpandedDepthInstanceRescoreContinuousTracker"
        and dc["metadata"]["context"]["implementation"] == "InstanceRescoreContinuousTracker",
        "actual depth/Dense implementations required",
    )
    require(
        old["lower"] == support.COMMON.digest_array(np.array([0.0, 0.0, -6.0]))
        and old["recovery_config"]["lower_mm"] == [0.0, 0.0, -6.0]
        and ctx["lower"] == support.COMMON.digest_array(np.array([0.0, 0.0, -12.0]))
        and ctx["recovery_config"]["lower_mm"] == [0.0, 0.0, -12.0],
        "registered lower bounds differ",
    )
    expected = copy.deepcopy(ctx)
    policies = {
        k: expected.pop(k)
        for k in ("instance_rescore_policy", "sparse_distance_backend", "expanded_depth_policy")
    }
    expected["implementation"] = old["implementation"]
    expected["lower"] = copy.deepcopy(old["lower"])
    expected["recovery_config"]["lower_mm"][2] = -6.0
    require(expected == old, "undeclared calibration/fitter/bounds/RMS/anatomy/support change")
    policy = policies["instance_rescore_policy"]
    require(
        policy
        == dc["plan"]["new_scoring_policy"]
        == dc["plan"]["checkpoint_binding"]["new_scoring_policy"],
        "mask scientific loss changed",
    )
    backend, depth = policies["sparse_distance_backend"], policies["expanded_depth_policy"]
    require(
        backend["effective_from_native"] == START
        and backend["saved_workload_qualification_sha256"] == QUALIFICATION_SHA,
        "saved1660 qualified sparse distance backend required",
    )
    common.exact_fields(
        depth,
        dict(
            schema="expanded_depth_instance_rescore_policy_v1",
            effective_from_native=START,
            original_lower_z_mm=-6.0,
            expanded_lower_z_mm=-12.0,
            scientific_search_domain_changed=True,
            old_history_rewritten=False,
            depth_policy_applied_retroactively_to_prior_history=False,
            measured_floor_or_physical_depth_truth_claimed=False,
            clears_identity_review=False,
        ),
        "fixed development depth policy differs",
    )
    scope = dict(
        post_fit_candidate_selection_only=False,
        lower_z_changed_mm=[-6.0, -12.0],
        proposal_and_fitting_domains_changed=True,
        mask_loss_definition_and_normalization_unchanged=True,
        new_differentiated_instance_loss=False,
        x_y_upper_bounds_calibration_RMS_anatomy_support_age_review_unchanged=True,
        development_reuse=True,
        physical_depth_or_ID_truth=False,
        automatic_adoption=False,
    )
    for document in (dv["plan"], dv["summary"], dv["plan"]["checkpoint_binding"]):
        require(
            document["new_scoring_policy"] == policy
            and document["distance_backend"] == backend
            and document["expanded_depth_policy"] == depth
            and document["scientific_scope"] == scope,
            "producer scientific scope/policies disagree",
        )
    adapted = copy.deepcopy(b["original_metadata"])
    adapted.update(context=ctx, restore_count=5)
    require(b["imported_metadata"] == adapted, "depth import changed undeclared metadata")
    shared_index_fields = {"original_binding", "prefix_chunks"}
    require(
        set(dc["input_index"]) == shared_index_fields
        and set(dv["input_index"]) == shared_index_fields | {"rationale_references"}
        and all(dc["input_index"][k] == dv["input_index"][k] for k in shared_index_fields)
        and dc["input_index"]["original_binding"] == dc["plan"]["original_binding"]
        and dv["input_index"]["original_binding"] == dv["plan"]["original_binding"]
        and dc["plan"]["original_binding"] == dv["plan"]["original_binding"],
        "same input/calibration/17chunk binding required",
    )
    rationale = dv["input_index"]["rationale_references"]
    require(
        set(rationale) == {"free", "bounded", "sparse", "dense_control"}
        and all(
            d["rationale_references"] == rationale
            for d in (dv["plan"], dv["summary"], dv["plan"]["checkpoint_binding"])
        ),
        "depth-specific rationale references disagree",
    )
    return dict(
        same_parent_arrays=a["arrays"],
        same_parent_metadata=a["original_metadata"],
        sparse_qualification_sha256=QUALIFICATION_SHA,
        expanded_depth_policy=depth,
        scientific_scope=scope,
        rationale_references=rationale,
        retained39_payload_reads=0,
    )


def bind_rationale_edges(variant, evidence):
    """Bind successful small reference seals/summaries, without their payloads."""
    references = variant["docs"]["input_index"]["rationale_references"]
    for role, row in references.items():
        path = common.local(row["path"])
        summary = row["summary"]
        require(
            variant["seal"]["inputs"].get(str(path)) == row["sha256"]
            and variant["seal"]["inputs"].get(summary["path"]) == summary["sha256"],
            "producer rationale input edge differs: " + role,
        )
        seal = common.read(path, evidence, row["sha256"])
        require(
            seal["status"] == "ok"
            and seal["stage"] == path.parent.name
            and seal["outputs"]["summary.json"] == summary,
            "successful rationale summary binding differs: " + role,
        )
        common.bound_member(path.parent, seal, "summary.json", evidence)
        if role == "dense_control":
            require(row["sha256"] == CONTROL_SHA, "rationale Dense control differs")
        elif role == "sparse":
            require(row["sha256"] == QUALIFICATION_SHA, "rationale sparse qualification differs")


def load_candidates(bundles, evidence):
    result, references = {}, {}
    for arm, bundle in bundles.items():
        path = common.bound_member(bundle["root"], bundle["seal"], "candidate.npz", evidence)
        with np.load(path, allow_pickle=False) as saved:
            candidate = {k: saved[k] for k in saved.files}
        metrics.validate_candidate(candidate, FIRST, STOP)
        signatures = common.signature(candidate)
        require(
            signatures == bundle["docs"]["summary"]["candidate_arrays"],
            "candidate all21 signature differs",
        )
        for value in candidate.values():
            value.setflags(write=False)
        result[arm] = candidate
        references[arm] = dict(
            path=str(path),
            sha256=evidence.paths[str(path)],
            arrays=signatures,
            terminal_horizon=STOP,
            tracking_seal_sha256=bundle["seal_sha256"],
        )
    return result, references


def z_diagnostics(candidate, start=START, stop=STOP):
    frames = candidate["frame_idx"]
    keep = (frames >= start) & (frames < stop)
    require(
        np.array_equal(frames[keep], np.arange(start, stop)),
        "complete z diagnostic interval required",
    )
    z = candidate["pose_xyz"][keep, :, :, 2].copy()
    observed = candidate["observed"][keep]
    require(
        np.isfinite(z).all() and observed.shape == z.shape, "finite z/direct support shape required"
    )
    predicates = dict(
        below_old_lower=z < -6.0,
        below_expanded_lower=z < -12.0,
        within_expanded_lower_band=(z >= -12.0) & (z <= -12.0 + BOUNDARY_BAND_MM),
    )
    scopes = {}
    for label, eligible in (
        ("all_estimated_including_latent", np.ones(z.shape, bool)),
        ("directly_observed", observed),
    ):
        scopes[label] = dict(
            denominator_by_slot_node=eligible.sum(0).tolist(),
            total=int(eligible.sum()),
            distribution=metrics.geometry._stats(z[eligible]),
            distribution_by_slot_node=[
                [
                    metrics.geometry._stats(z[:, body, node][eligible[:, body, node]])
                    for node in range(3)
                ]
                for body in range(3)
            ],
            counts={
                name: dict(
                    total=int((mask & eligible).sum()),
                    by_slot_node=(mask & eligible).sum(0).tolist(),
                )
                for name, mask in predicates.items()
            },
        )
    return dict(
        native_interval=[start, stop],
        native_frames=stop - start,
        subsets=scopes,
        old_lower_z_mm=-6.0,
        expanded_lower_z_mm=-12.0,
        boundary_band_mm=BOUNDARY_BAND_MM,
        boundary_band_inclusive_mm=[-12.0, -12.0 + BOUNDARY_BAND_MM],
        threshold_retuned=False,
        boundary_band_is_acceptance_gate=False,
        physical_depth_truth=False,
    ), dict(frame_idx=frames[keep].copy(), z_mm=z, directly_observed=observed.copy(), **predicates)


def support_loss_runs(frames, loss):
    require(loss.shape == (len(frames), 3, 3) and loss.dtype == bool, "loss tensor shape differs")
    require(
        len(frames) > 0 and np.all(np.diff(frames) == 1), "native loss interval must be continuous"
    )
    rows = []
    for body in range(3):
        for node in range(3):
            changes = np.diff(np.r_[False, loss[:, body, node], False].astype(np.int8))
            for a, b in zip(
                np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
            ):
                rows.append(
                    dict(
                        slot=body,
                        node=metrics.old.old.NODES[node],
                        start=int(frames[a]),
                        stop=int(frames[b - 1]) + 1,
                        frames=int(b - a),
                        seconds=float(b - a) / 166,
                    )
                )
    return sorted(rows, key=lambda row: (row["start"], row["slot"], row["node"]))


def run(variant_sha, variant_runner_sha, output):
    require(EXECUTION_READY, "source review and successful expanded-depth tracking required")
    require(
        proof.valid_sha(variant_sha) and proof.valid_sha(variant_runner_sha),
        "explicit final seal/source SHAs required",
    )
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh comparison output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        require(not sys.modules["torch"].cuda.is_initialized(), "CPU-only comparison required")
        code = source_code()
        for path, digest in code.items():
            evidence.add(path, digest)
        evidence.add(COMPARISON_PLAN, COMPARISON_PLAN_SHA)
        bundles = {
            ARMS[0]: load_headers(CONTROL_STAGE, CONTROL_SHA, CONTROL_RUNNER_SHA, evidence),
            ARMS[1]: load_headers(VARIANT_STAGE, variant_sha, variant_runner_sha, evidence),
        }
        lineage = validate_domains(bundles[ARMS[0]], bundles[ARMS[1]])
        bind_rationale_edges(bundles[ARMS[1]], evidence)
        qpath = (
            WORK / "sparse_instance_saved_ten_second_qualification_v1/qualification.provenance.json"
        )
        require(
            bundles[ARMS[1]]["seal"]["inputs"].get(str(qpath)) == QUALIFICATION_SHA,
            "actual sparse qualification input edge missing",
        )
        qseal = common.read(qpath, evidence, QUALIFICATION_SHA)
        require(qseal["status"] == "ok", "sparse qualification failed")
        candidates, references = load_candidates(bundles, evidence)
        binding = bundles[ARMS[0]]["docs"]["plan"]["original_binding"]
        shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
        require(
            all(
                b["seal"]["inputs"].get(path) == digest
                for b in bundles.values()
                for path, digest in shared.items()
            ),
            "both consumed same SL/calibration edges required",
        )
        nodes, rig, seed = common.load_observations(
            binding, bundles[ARMS[1]]["seal"], bundles[ARMS[0]]["docs"]["metadata"], evidence
        )
        node_before = {
            v: common.signature(dict(points=p, offsets=o)) for v, (p, o) in nodes.data.items()
        }
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                arms=list(ARMS),
                native_interval=[START, STOP],
                native_frames=1660,
                terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
                candidate_references=references,
                comparison_plan_sha256=COMPARISON_PLAN_SHA,
                source_sha256=code,
                variant_runner_sha256=variant_runner_sha,
                strict_support_ray_mm=2.0,
                strict_support_pixel_gate=12.0,
                legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
                raw_slot_estimated_geometry_scope=common.GEOMETRY_SCOPE,
                development_frames=list(DEVELOPMENT_FRAMES),
                development_frames_are_holdout=False,
                boundary_band_mm=BOUNDARY_BAND_MM,
                boundary_band_is_acceptance_gate=False,
                thresholds_retuned=False,
                physical_ID_truth=False,
                physical_depth_truth=False,
                checkpoint_payload_reads=0,
                raw_video_reads=0,
                masks_decoded=0,
                GPU_used=False,
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
            strict = support.verify_support_partitioned(
                part,
                nodes,
                rig,
                block_frames=STOP - START,
                on_block=lambda meta, arrays, arm=arm: out.write(f"support/{arm}.npz", arrays),
            )
            values, motion, geometry = metrics.interval_metrics(candidate, seed, START, STOP)
            out.write(f"motion/{arm}.json", motion)
            out.write(f"geometry/{arm}.npz", geometry)
            z, arrays = z_diagnostics(candidate)
            out.write(f"z/{arm}.npz", arrays)
            indices = np.array(DEVELOPMENT_FRAMES) - FIRST
            out.write(
                f"development_frames/{arm}.npz", {k: a[indices] for k, a in candidate.items()}
            )
            methods[arm] = dict(interval=values, strict_support=strict, z_distribution=z)
        control, variant = (candidates[a] for a in ARMS)
        difference, rows, arrays = metrics.differences(control, variant, START, STOP)
        out.write("differences/direct_gain_loss.json", rows)
        out.write("differences/all_native_transitions.npz", arrays)
        loss_runs = support_loss_runs(arrays["frame_idx"], arrays["support_loss"])
        out.write("differences/contiguous_support_losses.json", loss_runs)
        geometry, arrays = common.geometry_deltas(control, variant)
        # The frozen helper's positional arm names are explicitly bound here.
        out.write("differences/raw_slot_geometry.npz", arrays)
        geometry["array_axis0_arm_order"] = list(ARMS)
        prefix, arrays = common.same_horizon_prefix(control, variant)
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
            "SL arrays mutated",
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                native_frames=1660,
                terminal_horizon_by_arm=dict.fromkeys(ARMS, STOP),
                candidate_references=references,
                methods=methods,
                support_delta=difference,
                support_loss_runs=loss_runs,
                geometry_deltas=geometry,
                selected_prefix_path_changes=prefix,
                legacy_difference_field_arms=LEGACY_DIFFERENCE_FIELD_ARMS,
                raw_slot_estimated_geometry_scope=common.GEOMETRY_SCOPE,
                development_frames=list(DEVELOPMENT_FRAMES),
                development_frames_are_holdout=False,
                checkpoint_payload_reads=0,
                raw_video_reads=0,
                masks_decoded=0,
                GPU_used=False,
                candidate_modified=False,
                review_modified=False,
                human_truth_created=False,
                all_three_physical_ID_truth=False,
                physical_depth_truth=False,
                adopted=False,
                limitations=[
                    "Development comparison includes the three frames "
                    "used to choose the new domain.",
                    "The -12 mm domain is not an independently measured tank floor.",
                    "Support and raw-slot geometry are not independently aligned "
                    "biological-ID accuracy.",
                    "Sparse parity was qualified on saved candidate banks; "
                    "new trajectories are not promised equal.",
                    "Mask loss is unchanged; the proposal and fitting domain is changed.",
                    "A lower-bound concentration does not authorize "
                    "further threshold tuning or adoption.",
                    "This ten-second result does not establish "
                    "five-minute or thirty-minute performance.",
                ],
            ),
        )
        require(not sys.modules["torch"].cuda.is_initialized(), "comparison initialized CUDA")
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
    parser.add_argument("--variant-seal-sha256", required=True)
    parser.add_argument("--variant-runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.variant_seal_sha256, args.variant_runner_sha256, args.output)
