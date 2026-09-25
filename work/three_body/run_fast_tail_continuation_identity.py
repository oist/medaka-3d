"""One exact streaming chest+mouth identity solve on a compared two/five-minute body track.

The raw21 geometry and review flags remain immutable. An ID-to-raw-slot mapping,
all alternative gaps and unsettled intervals are separate research outputs.
Fixed capacity failures retain their artifacts and never trigger pruning,
reinitialization, a larger cap or an Original identity-solver rerun.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import compare_fast_tail_continuation as comparison
import numpy as np
import qualify_streaming_tail_minute as qualified

proof, streaming = comparison.proof, qualified.streaming
WORK, REPO, require = comparison.WORK, comparison.REPO, comparison.require
START, NODES = 66400, (1, 0)
STAGE = "fast_tail_continuation_streaming_identity_v1"
QUALIFICATION = WORK / qualified.STAGE
QUALIFICATION_SHA = "edcf44b18eb582e044b7f89460ba1aa5809475f971d44fee65e39bef9ce837ff"
QUALIFIER_SHA = "f9b33819abf260f11fd8276939ee47a6ea012c768ebbcf94b8d8978896dfb222"
COMPARISON_SHA = "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
CONFIG = dict(scale_mm=0.8, weight=0.2, initial_state=0, tie_atol=1e-10, tie_rtol=1e-12)


def source_code():
    code = comparison.source_code()
    pins = {
        Path(comparison.__file__).resolve(): COMPARISON_SHA,
        Path(qualified.__file__).resolve(): QUALIFIER_SHA,
        Path(streaming.__file__).resolve(): qualified.STREAM_SHA,
        Path(streaming.frozen.__file__).resolve(): streaming.FROZEN_SHA,
        streaming.LAB / "identity_dp.py": streaming.CORE_SHA,
        Path(qualified.proof.__file__).resolve(): qualified.PROOF_SHA,
    }
    for path, digest in pins.items():
        require(code.get(str(path)) == digest, "qualified identity source changed: " + str(path))
    return code


def qualification(evidence):
    seal = comparison.json_input(
        QUALIFICATION / "qualification.provenance.json", evidence, QUALIFICATION_SHA
    )
    require(
        seal["stage"] == qualified.STAGE and seal["status"] == "ok",
        "successful streaming qualification required",
    )
    for module, digest in ((qualified, QUALIFIER_SHA), (streaming, qualified.STREAM_SHA)):
        require(
            seal["inputs"].get(str(Path(module.__file__).resolve())) == digest,
            "qualification source input edge",
        )
    plan = proof.read_json(comparison.custom_member(seal, QUALIFICATION, "plan.json", evidence))
    summary = proof.read_json(
        comparison.custom_member(seal, QUALIFICATION, "summary.json", evidence)
    )
    fields = proof.read_json(
        comparison.custom_member(seal, QUALIFICATION, "field_comparison.json", evidence)
    )
    validate_qualification(plan, summary, fields)
    return summary


def validate_qualification(plan, summary, fields):
    require(
        plan["schema"] == summary["schema"] == qualified.STAGE
        and plan["native_interval"] == summary["native_interval"] == [START, 76360]
        and summary["status"] == "ok"
        and summary["native_frames"] == 9960
        and summary["all_29_arrays_exact"] is True
        and summary["all_19_scalars_exact"] is True
        and summary["raw21_unchanged"] is True
        and summary["finite_saved_medaka_minute_qualified"] is True
        and summary["five_minute_qualified"] is False
        and summary["all_three_ID_truth"] is False
        and summary["original_solver_rerun"] is False
        and summary["intermediate_ID_reset"] is False,
        "finite real-minute qualification scope differs",
    )
    require(
        plan["target_block_frames"] == 256
        and plan["node_indices"] == [1, 0]
        and plan["dp_config"] == qualified.proof.DP_CONFIG
        and plan["resources"]
        == summary["resources"]
        == dataclasses.asdict(streaming.DEFAULT_RESOURCES),
        "qualified objective or capacity differs",
    )
    require(
        fields["array_count"] == len(fields["arrays"]) == 29
        and fields["scalar_count"] == len(fields["scalars"]) == 19
        and fields["all_exact"] is True
        and all(v["exact"] is True and v["read_only"] is True for v in fields["arrays"].values())
        and all(v["exact"] is True for v in fields["scalars"].values()),
        "all29/19 qualification comparison required",
    )


def validate_comparison(plan, summary, stop, milestone_sha):
    comparison.intervals(stop)
    require(
        plan["schema"] == summary["schema"] == comparison.STAGE
        and plan["horizon_stop"] == summary["horizon_stop"] == stop
        and plan["milestone_sha256"] == milestone_sha
        and summary["status"] == "ok"
        and summary["native_interval"] == [START, stop]
        and summary["native_frames"] == stop - START
        and summary["candidate_all21_and_source_arrays_unchanged"] is True
        and summary["review_modified"] is False
        and summary["same_parent_and_horizon"] is True
        and summary["logical_initialization_count"] == 1
        and summary["restore_count_by_arm"]["fast_tail"] == 5
        and summary["external_pose_resets"] == 0
        and summary["all_three_ID_truth"] is False
        and plan["candidate_references"] == summary["candidate_references"],
        "successful matching body comparison required",
    )


def load_geometry(root, seal_sha, milestone_sha, stop, evidence):
    root = root.resolve(strict=True)
    require(root.is_relative_to(WORK), "body comparison must be in workspace")
    seal = comparison.json_input(root / "comparison.provenance.json", evidence, seal_sha)
    require(
        seal["stage"] == comparison.STAGE and seal["status"] == "ok",
        "successful comparison seal required",
    )
    require(
        seal["inputs"].get(str(Path(comparison.__file__).resolve())) == COMPARISON_SHA,
        "comparison source edge",
    )
    plan = proof.read_json(comparison.custom_member(seal, root, "plan.json", evidence))
    summary = proof.read_json(comparison.custom_member(seal, root, "summary.json", evidence))
    validate_comparison(plan, summary, stop, milestone_sha)
    milestone_path = comparison.BRANCH / f"milestones/through_{stop:06d}/summary.json"
    require(
        seal["inputs"].get(str(milestone_path)) == milestone_sha,
        "comparison milestone input differs",
    )
    milestone = comparison.json_input(milestone_path, evidence, milestone_sha)
    comparison.validate_milestone(milestone, stop)
    full, reference = comparison.load_milestone_candidate(dict(summary=milestone), stop, evidence)
    require(
        reference == summary["candidate_references"]["fast_tail"]
        and seal["inputs"].get(reference["path"]) == reference["sha256"],
        "compared raw21 candidate differs",
    )
    full_signatures = comparison.previous.signature(full)
    for value in full.values():
        value.flags.writeable = False
    geometry = {k: v[START - comparison.FIRST :] for k, v in full.items()}
    comparison.metrics.validate_candidate(geometry, START, stop)
    return geometry, full, full_signatures, reference, summary


def solve_once(geometry, spool):
    """Exactly one call; target blocks carry the same global identity frontier."""
    expected_config = {
        k: v for k, v in qualified.proof.DP_CONFIG.items() if not k.startswith("max_")
    }
    require(
        expected_config == CONFIG and NODES == (1, 0),
        "qualified chest-mouth objective and initial state must remain unchanged",
    )
    frames = np.ascontiguousarray(geometry["frame_idx"])
    pose = np.ascontiguousarray(geometry["pose_xyz"][:, :, NODES, :])
    direct = np.ascontiguousarray(geometry["observed"][:, :, NODES])
    unary = np.zeros((len(frames), 6), np.float64)
    require(direct[0].all(), "all initial chest and mouth nodes must be direct")
    values = dict(frame_idx=frames, unary=unary, node_xyz=pose, direct_nodes=direct)
    signatures = {}
    for key, value in values.items():
        value.flags.writeable = False
        signatures[key] = qualified.array_signature(value)
    result = streaming.solve_streaming_identity_node_memory_dp(
        frames,
        unary,
        pose,
        direct,
        spool_root=spool,
        resources=streaming.DEFAULT_RESOURCES,
        target_block_frames=256,
        node_indices=NODES,
        **CONFIG,
    )
    require(
        all(qualified.array_signature(v) == signatures[k] for k, v in values.items()),
        "solver input changed",
    )
    return result, signatures


def validate_result(spooled, geometry, input_signatures, spool):
    metadata, records = qualified.verify_spool(spool, spooled)
    result = spooled.result
    require(
        type(result) is streaming.frozen.IdentityNodeMemoryDPResult,
        "full memory DP result required",
    )
    require(metadata["inputs"] == input_signatures, "spool inputs differ from whole interval")
    require(
        result.status == "optimal"
        and result.initial_state == 0
        and result.scale_mm == CONFIG["scale_mm"]
        and result.weight == CONFIG["weight"]
        and result.node_indices == NODES
        and result.max_layer_states == 100000
        and result.max_total_states == 1000000
        and result.max_total_edges == 6000000
        and result.approximation_used is False
        and np.array_equal(result.frame_idx, geometry["frame_idx"])
        and np.array_equal(result.permutation_by_id[0], [0, 1, 2])
        and np.array_equal(
            np.sort(result.permutation_by_id, axis=1),
            np.broadcast_to([0, 1, 2], (len(result.frame_idx), 3)),
        ),
        "result interval, initial identity, permutation or fixed caps differ",
    )
    manifest = proof.read_json(spooled.manifest_path)
    arrays, scalars = {}, {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            require(not value.flags.writeable, "result arrays must remain read-only")
            arrays[field.name] = qualified.array_signature(value)
        else:
            scalars[field.name] = streaming._scalar_json(value)
    require(len(arrays) == 29 and len(scalars) == 19, "all29 arrays and19 scalars required")

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    require(
        canonical(scalars) == canonical(manifest["scalars"]), "saved all19 scalar values differ"
    )
    return metadata, records, arrays, manifest["scalars"]


def identity_report(result):
    frames, mapping = result.frame_idx, result.permutation_by_id
    changed = mapping != np.array([0, 1, 2])
    transition = np.zeros(mapping.shape, bool)
    transition[1:] = mapping[1:] != mapping[:-1]
    tied = result.identity_alternative_status == "tie_within_tolerance"
    multiple = result.optimal_state_mask.sum(axis=1) > 1
    arrays = dict(
        frame_idx=frames,
        identity_to_raw_slot=mapping,
        permutation_state=result.path_state,
        state_alternative_gap=result.state_alternative_gap,
        identity_alternative_gap=result.identity_alternative_gap,
        state_alternative_status=result.state_alternative_status,
        identity_alternative_status=result.identity_alternative_status,
        optimal_state_mask=result.optimal_state_mask,
        min_marginal_delta=result.min_marginal_delta,
        id_slot_min_marginal_delta=result.id_slot_min_marginal_delta,
        mapping_changed_from_initial_slots=changed,
        mapping_transition=transition,
        tied_identity_slots=tied,
        terminal_age_frames=result.terminal_age_frames,
        terminal_unsettled_gap=result.terminal_unsettled_gap,
    )
    report = dict(
        native_frames=len(frames),
        changed_mapping_native_frames=int(changed.any(1).sum()),
        mapping_transition_native_frames=int(transition.any(1).sum()),
        changed_from_initial_slot_intervals=qualified.proof.intervals(frames, changed),
        tied_identity_intervals=qualified.proof.intervals(frames, tied),
        multiple_optimal_permutation_frames=int(multiple.sum()),
        multiple_optimal_permutation_intervals=qualified.proof.intervals(
            frames, np.repeat(multiple[:, None], 3, axis=1)
        ),
        numerical_tie_tolerance=result.numerical_tie_tolerance,
        global_alternative_gap=streaming._scalar_json(result.global_alternative_gap),
        global_alternative_status=result.global_alternative_status,
        terminal_unsettled_gap=result.terminal_unsettled_gap.tolist(),
        terminal_age_frames=result.terminal_age_frames.tolist(),
        terminal_last_event_frame=result.terminal_last_event_frame.tolist(),
        terminal_last_event_slot=result.terminal_last_event_slot.tolist(),
        permutation_scope=(
            "One whole-body permutation shared by chest and mouth; applied to no raw array"
        ),
        initial_identity_scope=(
            "Initial numeric body slots, not independently verified sex or physical IDs"
        ),
        all_three_ID_truth=False,
        biological_swap_zero_verified=False,
        review_flags_cleared=False,
        female_endpoint_rescoring_performed=False,
        confidence_scope="Engineering min-marginal cost differences, not calibrated probabilities",
    )
    return arrays, report


def record_spool(output, records=None):
    """Preserve success records or inventory only this run's unfinished spool."""
    if records is None:
        records = {}
        for path in (output.root / "spool").rglob("*"):
            if path.is_file():
                records[path] = dict(
                    path=str(path), sha256=proof.sha(path), size_bytes=path.stat().st_size
                )
    for path, row in records.items():
        require(path.resolve().is_relative_to(output.root.resolve()), "spool outside output")
        output.records[path.relative_to(output.root).as_posix()] = row


def capacity_details(error, stop):
    if not isinstance(error, streaming.frozen.IdentityMemoryCapacityError):
        return None
    offset = error.frame_offset
    return dict(
        resource=error.resource,
        attempted=error.attempted,
        limit=error.limit,
        reported_frame_offset=offset,
        native_frame=START + offset if 0 <= offset < stop - START else None,
        fixed_caps_unchanged=True,
        approximate_pruning=False,
        retry_performed=False,
        partial_mapping_published=False,
    )


def run(args):
    comparison.intervals(args.horizon_stop)
    require(
        args.execute is True
        and proof.valid_sha(args.comparison_sha)
        and proof.valid_sha(args.milestone_sha),
        "explicit execution and comparison/milestone SHAs required",
    )
    destination = args.output.resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "fresh identity output required",
    )
    out, evidence = proof.Output(destination), comparison.producer.original.phase.Digests()
    code, began = source_code(), proof.utc()
    full = full_signatures = None
    try:
        require(
            not comparison.producer.torch.cuda.is_initialized(), "CPU-only identity run required"
        )
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        q = qualification(evidence)
        geometry, full, full_signatures, reference, body = load_geometry(
            args.comparison_root,
            args.comparison_sha,
            args.milestone_sha,
            args.horizon_stop,
            evidence,
        )
        raw_signatures = comparison.previous.signature(geometry)
        out.write("raw21_signatures.json", raw_signatures)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[START, args.horizon_stop],
                comparison_sha256=args.comparison_sha,
                milestone_sha256=args.milestone_sha,
                raw_candidate_reference=reference,
                node_indices=list(NODES),
                dp_config=qualified.proof.DP_CONFIG,
                resources=dataclasses.asdict(streaming.DEFAULT_RESOURCES),
                target_block_frames=256,
                real_minute_qualification_sha256=QUALIFICATION_SHA,
                real_minute_qualification_scope=q["native_interval"],
                body_process_complete=body["whole_five_minute_process_complete"],
                one_solver_call=True,
                initial_P=[0, 1, 2],
                unary="float64 exact zero T,6",
                intermediate_ID_reset=False,
                cap_increase=False,
                original_ID_solver_rerun=False,
                raw_geometry_sources_support_review_modified=False,
                physical_ID_truth=False,
            ),
        )
        tick = time.perf_counter()
        spooled, input_signatures = solve_once(geometry, out.root / "spool")
        solve_seconds = time.perf_counter() - tick
        metadata, records, arrays, scalars = validate_result(
            spooled, geometry, input_signatures, out.root / "spool"
        )
        record_spool(out, records)
        mappings, report = identity_report(spooled.result)
        out.write("identity_mapping_and_ambiguity.npz", mappings)
        out.write("identity_report.json", report)
        out.write("dp_array_signatures.json", arrays)
        out.write("dp_scalars.json", scalars)
        require(
            comparison.previous.signature(full) == full_signatures
            and comparison.previous.signature(geometry) == raw_signatures,
            "raw21 geometry/source/support/review changed",
        )
        require(
            not comparison.producer.torch.cuda.is_initialized(), "identity solve initialized CUDA"
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, args.horizon_stop],
                native_frames=args.horizon_stop - START,
                mathematical_status=spooled.result.status,
                solve_seconds=solve_seconds,
                solver_calls=1,
                node_indices=list(NODES),
                blocks=metadata["block_count"],
                global_states=metadata["global_states"],
                global_edges=metadata["global_edges"],
                maximum_layer_states=metadata["maximum_layer_states"],
                resources=metadata["resources"],
                all29_arrays_and19_scalars_saved=True,
                spool_all_files_SHA_layout_membership_verified=True,
                raw21_unchanged=True,
                intermediate_ID_reset=False,
                cap_increase=False,
                all_three_ID_truth=False,
                biological_swap_zero_verified=False,
                original_ID_solver_rerun=False,
                GPU_execution=False,
                raw_decoded_frames=0,
                raw_candidate_reference=reference,
                report=report,
            ),
        )
        require(source_code() == code, "identity source closure changed")
        evidence.final_guard()
        out.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=began,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        record_spool(out)
        raw_unchanged = full is not None and comparison.previous.signature(full) == full_signatures
        details = capacity_details(error, args.horizon_stop)
        out.write(
            "failure.json",
            dict(
                error=repr(error),
                capacity=details,
                raw21_unchanged=raw_unchanged,
                partial_spool_preserved=True,
                complete_solution_available=False,
            ),
        )
        guard_error = None
        try:
            require(source_code() == code, "identity source closure changed")
            evidence.final_guard()
            out.final_guard()
        except BaseException as guard:
            guard_error = repr(guard)
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=began,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
            capacity=details,
            final_guard_error=guard_error,
        )
        with (destination / "identity.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(seal))
        raise
    with (destination / "identity.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(seal))
    print(
        json.dumps(dict(status="ok", output=str(destination), solve_seconds=solve_seconds)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--milestone-sha", required=True)
    parser.add_argument(
        "--horizon-stop", type=int, choices=sorted(comparison.ORIGINALS), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    run(parser.parse_args())
