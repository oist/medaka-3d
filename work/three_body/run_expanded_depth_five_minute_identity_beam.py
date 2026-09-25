"""Disabled CPU-only width256 ID search on the final ExpandedDepth five minutes.

A successful Sparse -6 / ExpandedDepth -12 comparison must bind both successful
final116200 producers, their original final/milestone candidates and the saved
comparison copy. Only the variant copy is decoded; its numerical21 signatures
must match the comparison. Raw geometry, source assignments and review stay intact.
One continuous frozen beam call covers [66400,116200), with initial P012 only.
All gaps are conditional on retained hypotheses, never physical identity truth.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import compare_expanded_depth_five_minutes as comparison
import numpy as np
import run_sparse_instance_five_minute_identity_beam as frozen

proof, require = frozen.proof, frozen.require
WORK, REPO = frozen.WORK, frozen.REPO
FIRST, START, STOP = frozen.FIRST, frozen.START, frozen.STOP
ARM = "expanded_depth_instance_rescore"
STAGE = "expanded_depth_five_minute_identity_beam256_v1"
COMPARISON_ROOT = WORK / comparison.STAGE
COMPARISON_SHA = "db30f8fd32a58a60b77ef1daa50dbf42da7a0d181eb4b43da47cdf2b2ee2694e"
FROZEN_SHA = "d433ce526804f5d72594b942c6ab93e9d8e6a2b9b6c1ae46e3f276b6cc925631"
RUNNER_SHA = "4befd129b62f8b00e59dbd52ebe10dd4c8040011aea11f308417067b5e54ea98"
TRACKER_NAME = "expanded_depth_instance_rescore_tracker.py"
TRACKER_SHA = "522f4012fb45b2e1fc89850c5448991fa3979193f736d91fe81b5f6d0344fe57"
EXECUTION_READY = True
NODES, WIDTH, CONFIG = frozen.NODES, frozen.WIDTH, frozen.CONFIG
BEAM_SHA = frozen.BEAM_SHA
cpu_only = frozen.cpu_only
qualification = frozen.qualification
scientific_geometry = frozen.scientific_geometry
solve_once = frozen.solve_once
checked_ledger = frozen.checked_ledger
resource_report = frozen.resource_report
write_result = frozen.write_result
explicit_json = frozen.explicit_json
failure_record = frozen.failure_record
signature = comparison.common.signature


def source_code():
    code = frozen.source_code()
    for module, digest in ((frozen, FROZEN_SHA), (comparison, COMPARISON_SHA)):
        require(code.get(str(Path(module.__file__).resolve())) == digest, "frozen source differs")
    require(str(Path(__file__).resolve()) in code, "new caller absent from source closure")
    return code


def document(root, seal, name, evidence):
    return proof.read_json(comparison.common.bound_member(root, seal, name, evidence))


def edge(seal, path, digest):
    require(
        proof.valid_sha(digest) and seal["inputs"].get(str(path.resolve())) == digest,
        "missing consumed input edge",
    )


def comparison_headers(expected_sha, evidence):
    """Check small sealed metadata; never decode original candidates or checkpoint39."""
    require(proof.valid_sha(expected_sha), "explicit successful comparison SHA required")
    path = COMPARISON_ROOT / "comparison.provenance.json"
    evidence.add(path, expected_sha)
    seal = proof.read_json(path)
    require(
        seal["stage"] == comparison.STAGE and seal["status"] == "ok",
        "successful same116200 comparison required before candidates",
    )
    edge(seal, Path(comparison.__file__), COMPARISON_SHA)
    summary = document(COMPARISON_ROOT, seal, "summary.json", evidence)
    plan = document(COMPARISON_ROOT, seal, "plan.json", evidence)
    scope = dict(
        schema=comparison.STAGE,
        native_interval=[START, STOP],
        native_frames=STOP - START,
        intervals={key: list(value) for key, value in comparison.INTERVALS.items()},
        terminal_horizon_by_arm=dict.fromkeys(comparison.ARMS, STOP),
        lower_z_mm_by_arm={comparison.ARMS[0]: -6.0, ARM: -12.0},
        depth_and_mask_change_native=76360,
        sparse_backend_start_by_arm={comparison.ARMS[0]: 78020, ARM: 76360},
        strict_support_ray_mm=2.0,
        strict_support_pixel_gate=12.0,
        physical_ID_truth=False,
        physical_depth_truth=False,
        automatic_adoption=False,
    )
    for doc in (summary, plan):
        comparison.common.exact_fields(doc, scope, "comparison scope differs")
    comparison.common.exact_fields(
        summary,
        dict(
            status="ok",
            candidate_modified=False,
            review_modified=False,
            all_three_physical_ID_truth=False,
            human_truth_created=False,
            adopted=False,
        ),
        "comparison raw/truth contract differs",
    )
    require(
        plan["arms"] == list(comparison.ARMS)
        and plan["variant_runner_sha256"] == RUNNER_SHA
        and plan["candidate_references"] == summary["candidate_references"]
        and set(plan["candidate_references"]) == set(comparison.ARMS),
        "comparison candidate/runner binding differs",
    )
    producers = {}
    for arm in comparison.ARMS:
        reference = plan["candidate_references"][arm]
        if arm == ARM:
            stage, runner, digest = comparison.VARIANT_STAGE, comparison.VARIANT_RUNNER, RUNNER_SHA
        else:
            spec = comparison.saved.SPECS[arm]
            stage, runner, digest = spec["stage"], WORK / spec["runner"], spec["source_sha"]
        root = WORK / stage
        producer_path = root / "tracking.provenance.json"
        producer_sha = reference["tracking_seal_sha256"]
        edge(seal, producer_path, producer_sha)
        evidence.add(producer_path, producer_sha)
        producer = proof.read_json(producer_path)
        require(
            producer["stage"] == stage and producer["status"] == "ok",
            "both final producer seals must succeed",
        )
        edge(producer, runner, digest)
        edge(seal, runner, digest)
        evidence.add(runner, digest)
        if arm == ARM:
            tracker = WORK / TRACKER_NAME
            edge(producer, tracker, TRACKER_SHA)
            edge(seal, tracker, TRACKER_SHA)
            evidence.add(tracker, TRACKER_SHA)
        require(
            reference["terminal_horizon"] == STOP and len(reference["arrays"]) == 21,
            "same116200 all21 reference required",
        )
        archives = {}
        for member, path_key, sha_key in (
            ("candidate.npz", "path", "sha256"),
            (f"milestones/through_{STOP}/candidate.npz", "milestone_path", "milestone_sha256"),
        ):
            named = (root / member).resolve()
            row = producer["outputs"][member]
            require(
                Path(reference[path_key]).resolve() == named == Path(row["path"]).resolve()
                and reference[sha_key] == row["sha256"]
                and type(row["size_bytes"]) is int
                and row["size_bytes"] > 0,
                "original final/milestone archive descriptor differs",
            )
            edge(seal, named, reference[sha_key])
            archives[member] = copy.deepcopy(row)
        producers[arm] = dict(
            seal_sha256=producer_sha,
            runner_sha256=digest,
            original_archives=archives,
        )
    return seal, dict(
        comparison_seal_sha256=expected_sha,
        comparison_source_sha256=COMPARISON_SHA,
        raw_candidate_reference=copy.deepcopy(plan["candidate_references"][ARM]),
        producer_bindings=producers,
        original_final_milestone_equality_verified_by_successful_comparison=True,
        original_candidate_payloads_rehashed_here=False,
        comparison_candidate_copy_available=True,
    )


def load_candidate(seal, binding, evidence):
    name = f"candidates/{ARM}.npz"
    path = comparison.common.bound_member(COMPARISON_ROOT, seal, name, evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) == 21, "21 unique members required")
        raw = {key: archive[key] for key in archive.files}
    comparison.metrics.validate_candidate(raw, FIRST, STOP)
    signatures = signature(raw)
    require(
        signatures == binding["raw_candidate_reference"]["arrays"],
        "comparison copy differs from all21 source signatures",
    )
    for value in raw.values():
        value.setflags(write=False)
    return raw, signatures, copy.deepcopy(seal["outputs"][name])


def run(comparison_sha, output):
    require(EXECUTION_READY, "source-only preparation; ExpandedDepth ID execution is not enabled")
    cpu_only()
    output = output.resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "fresh independent output required"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    calls, raw, signatures, binding, code = 0, None, None, None, {}
    try:
        code = source_code()
        for path, digest in code.items():
            evidence.add(path, digest)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        qualified = qualification(evidence)
        seal, binding = comparison_headers(comparison_sha, evidence)
        raw, signatures, copy_reference = load_candidate(seal, binding, evidence)
        binding.update(raw21_signatures=signatures, consumed_comparison_copy=copy_reference)
        part = scientific_geometry(raw)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[START, STOP],
                native_frames=STOP - START,
                retained_raw_interval=[FIRST, STOP],
                terminal_horizon=STOP,
                **binding,
                source_sha256=code,
                prior_minute_qualification=qualified,
                beam_source_sha256=BEAM_SHA,
                beam_width=WIDTH,
                dp_config=CONFIG,
                node_indices=list(NODES),
                unary="float64 exact zero T,6",
                initial_P=[0, 1, 2],
                initial_P_scope="numeric slot convention only",
                initialization_frame=START,
                solve_calls_planned=1,
                intermediate_ID_resets=0,
                gaps_conditional_on_retained_graph=True,
                gaps_are_probabilities=False,
                global_optimality_certified=False,
                physical_ID_truth=False,
                review_flags_cleared=False,
                automatic_adoption=False,
                GPU_used=False,
                exact_DP_rerun=False,
            ),
        )
        tick = time.perf_counter()
        calls = 1
        result = solve_once(part)
        seconds = time.perf_counter() - tick
        ledger, ledger_report = checked_ledger(part, result)
        resources = resource_report(result)
        require(signature(raw) == signatures, "raw geometry/source/review changed")
        write_result(out, result, ledger)
        out.write("independent_path_ledger.json", explicit_json(ledger_report))
        out.write(
            "summary.json",
            explicit_json(
                dict(
                    schema=STAGE,
                    status="ok",
                    native_interval=[START, STOP],
                    native_frames=STOP - START,
                    terminal_horizon=STOP,
                    **binding,
                    solve_calls=calls,
                    solve_seconds=seconds,
                    beam_width=WIDTH,
                    resources=resources,
                    path_cost=result.path_cost,
                    conditional_global_alternative_gap=result.conditional_global_alternative_gap,
                    conditional_global_alternative_status=result.conditional_global_alternative_status,
                    unknown_ID_alternative_cells=int(
                        (
                            result.conditional_identity_alternative_status == "not_retained_unknown"
                        ).sum()
                    ),
                    conditional_tied_frames=int(
                        (result.conditional_optimal_state_mask.sum(axis=1) > 1).sum()
                    ),
                    terminal_unsettled_gap=result.terminal_unsettled_gap,
                    pruning_occurred=result.pruning_occurred,
                    finite_prefixes_discarded=result.finite_prefixes_discarded,
                    gaps_conditional_on_retained_graph=True,
                    gaps_are_probabilities=False,
                    global_optimality_certified=False,
                    physical_ID_truth=False,
                    original_raw21_unchanged=True,
                    raw_sources_review_unchanged=True,
                    review_flags_cleared=False,
                    intermediate_ID_resets=0,
                    candidate_npz_decodes=1,
                    checkpoint39_reads=0,
                    raw_video_reads=0,
                    masks_decoded=0,
                    GPU_used=False,
                    exact_DP_rerun=False,
                    automatic_retry=False,
                    cap_increase=False,
                    automatic_adoption=False,
                )
            ),
        )
        cpu_only()
        require(source_code() == code, "source closure changed")
        evidence.final_guard()
        out.final_guard()
        final = dict(stage=STAGE, status="ok")
    except BaseException as error:
        failure = failure_record(error, calls)
        failure.update(
            native_interval=[START, STOP],
            unresolved_interval=[START, STOP],
            binding=binding,
            exact_DP_rerun=False,
            automatic_adoption=False,
        )
        if raw is not None:
            failure["raw21_unchanged_at_failure"] = signature(raw) == signatures
        out.write("failure.json", failure)
        final = dict(
            stage=STAGE,
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            **failure,
        )
        with (output / "identity.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(final))
        raise
    final.update(
        started_at=started, finished_at=proof.utc(), inputs=evidence.paths, outputs=out.records
    )
    with (output / "identity.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison_sha, args.output)
