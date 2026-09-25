"""Disabled one-call width256 ID control on the successful final OldFast five minutes.

The comparison binds the milestone candidate directly; it did not save a copy.
For this fixed completed run, final and milestone archives have the same SHA.
Both archives are guarded, the milestone is decoded once, and all21 signatures
must equal the comparison reference. Raw geometry, sources and review stay read-only.
The frozen beam/qualification/ledger/codec are reused without changing their globals.
This known-body-mixing control is neither physical identity truth nor adoption.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import compare_fast_tail_continuation as comparison
import numpy as np
import run_sparse_instance_five_minute_identity_beam as frozen

proof, require = frozen.proof, frozen.require
WORK, REPO = frozen.WORK, frozen.REPO
FIRST, START, STOP = frozen.FIRST, frozen.START, frozen.STOP
ARM = "fast_tail"
STAGE = "fast_tail_five_minute_identity_beam256_v1"
COMPARISON_ROOT = WORK / "fast_tail_five_minute_continuation_comparison_v1"
TRACKING_ROOT = WORK / comparison.runner.STAGE
COMPARISON_CODE_SHA = "2e4a749d5ea13116f05a622afd47478af311525fbd856b32a116bd9ac765acba"
FROZEN_CODE_SHA = "d433ce526804f5d72594b942c6ab93e9d8e6a2b9b6c1ae46e3f276b6cc925631"
COMPARISON_SEAL_SHA = "91a2670ba206f9cadca44376c9d5b36ea7e1caf06d8cd7e4998b31f99316642e"
TRACKING_SEAL_SHA = "d049a2d9244d9c0c761de7d75035a12f10dee14289af1b58d741fad9ec710469"
MILESTONE_SHA = "ab169c81d337b0970217032ffddb2124713653c591399bc90cd65088aabe5c27"
CANDIDATE_SHA = "2976ccf4bdd97b1dc679ec022e0fb27c39429bc48a7c817585d14cd4785fd81d"
EXECUTION_READY = True
NODES, WIDTH, CONFIG = frozen.NODES, frozen.WIDTH, frozen.CONFIG
cpu_only = frozen.cpu_only
qualification = frozen.qualification
scientific_geometry = frozen.scientific_geometry
solve_once = frozen.solve_once
checked_ledger = frozen.checked_ledger
resource_report = frozen.resource_report
write_result = frozen.write_result
explicit_json = frozen.explicit_json
signature = comparison.previous.signature


def source_code():
    code = frozen.source_code()
    for module, digest in ((frozen, FROZEN_CODE_SHA), (comparison, COMPARISON_CODE_SHA)):
        require(code.get(str(Path(module.__file__).resolve())) == digest, "frozen caller changed")
    require(str(Path(__file__).resolve()) in code, "new caller absent from source closure")
    return code


def document(root, seal, name, evidence):
    return proof.read_json(comparison.custom_member(seal, root, name, evidence))


def edge(seal, path, digest):
    require(seal["inputs"].get(str(path.resolve())) == digest, "missing consumed input edge")


def comparison_headers(expected_sha, evidence):
    """Small successful seals/JSON only; no archive or checkpoint payload opens."""
    require(expected_sha == COMPARISON_SEAL_SHA, "fixed successful OldFast comparison required")
    path = COMPARISON_ROOT / "comparison.provenance.json"
    evidence.add(path, expected_sha)
    seal = proof.read_json(path)
    require(
        seal["stage"] == comparison.STAGE and seal["status"] == "ok",
        "successful same116200 comparison required",
    )
    edge(seal, Path(comparison.__file__), COMPARISON_CODE_SHA)
    summary = document(COMPARISON_ROOT, seal, "summary.json", evidence)
    plan = document(COMPARISON_ROOT, seal, "plan.json", evidence)
    expected = dict(
        schema=comparison.STAGE,
        status="ok",
        horizon_stop=STOP,
        native_interval=[START, STOP],
        native_frames=STOP - START,
        whole_five_minute_process_complete=True,
        final_tracking_seal_verified=True,
        same_parent_and_horizon=True,
        same_registered_input_binding=True,
        logical_initialization_count=1,
        external_pose_resets=0,
        candidate_all21_and_source_arrays_unchanged=True,
        strict_support_recomputed_arms=2,
        strict_support_native_frames=2 * (STOP - START),
        physical_ID_truth=False,
        all_three_ID_truth=False,
        swap_zero_verified=False,
        review_modified=False,
        identity_solver_included=False,
    )
    require(
        all(type(summary.get(k)) is type(v) and summary[k] == v for k, v in expected.items()),
        "comparison scope/lifecycle/raw/truth contract differs",
    )
    windows = {key: list(value) for key, value in comparison.intervals(STOP).items()}
    require(
        plan["schema"] == comparison.STAGE
        and plan["horizon_stop"] == STOP
        and plan["intervals"] == summary["intervals"] == windows
        and plan["candidate_references"] == summary["candidate_references"]
        and set(plan["candidate_references"]) == {"original", ARM}
        and plan["final_tracking_sha256"] == TRACKING_SEAL_SHA
        and plan["milestone_sha256"] == MILESTONE_SHA
        and plan["original_source_seal_sha256"] == comparison.ORIGINALS[STOP]["seal"]
        and plan["support_gates"] == dict(ray_mm=2.0, native_pixel=12.0, block_frames=1660),
        "comparison plan and summary bindings differ",
    )
    reference = copy.deepcopy(plan["candidate_references"][ARM])
    require(len(reference["arrays"]) == 21, "all21 comparison signatures required")
    tracking_path = TRACKING_ROOT / "tracking.provenance.json"
    edge(seal, tracking_path, TRACKING_SEAL_SHA)
    evidence.add(tracking_path, TRACKING_SEAL_SHA)
    tracking = proof.read_json(tracking_path)
    require(
        tracking["stage"] == comparison.runner.STAGE and tracking["status"] == "ok",
        "successful final tracking required",
    )
    edge(tracking, Path(comparison.runner.__file__), comparison.RUNNER_SHA)
    final = document(TRACKING_ROOT, tracking, "summary.json", evidence)
    expected_final = dict(
        schema=comparison.runner.STAGE,
        status="ok",
        scientific_interval=[START, STOP],
        process_interval=[comparison.HALF, STOP],
        native_frames=STOP - START,
        process_frames=STOP - comparison.HALF,
        initialize_count=1,
        restore_count=5,
        external_pose_resets=0,
        new_begin_calls=0,
        finish_calls=1,
        physical_ID_truth=False,
    )
    require(
        all(type(final.get(k)) is type(v) and final[k] == v for k, v in expected_final.items())
        and len(final["checkpoints"]) == 24
        and [row["previous_frame"] + 1 for row in final["checkpoints"]]
        == list(range(comparison.HALF + 1660, STOP + 1660, 1660)),
        "final tracking extent/lifecycle differs",
    )
    milestone_name = f"milestones/through_{STOP:06d}/summary.json"
    edge(seal, TRACKING_ROOT / milestone_name, MILESTONE_SHA)
    require(tracking["outputs"][milestone_name]["sha256"] == MILESTONE_SHA, "milestone seal edge")
    milestone = document(TRACKING_ROOT, tracking, milestone_name, evidence)
    comparison.validate_milestone(milestone, STOP)
    candidate_name = f"milestones/through_{STOP:06d}/candidate.npz"
    rows = {
        "final": copy.deepcopy(tracking["outputs"]["candidate.npz"]),
        "milestone": copy.deepcopy(tracking["outputs"][candidate_name]),
    }
    for role, name in (("final", "candidate.npz"), ("milestone", candidate_name)):
        row, path = rows[role], (TRACKING_ROOT / name).resolve()
        require(
            Path(row["path"]).resolve() == path
            and path.is_relative_to(TRACKING_ROOT.resolve())
            and row["sha256"] == CANDIDATE_SHA
            and type(row["size_bytes"]) is int
            and row["size_bytes"] > 0,
            "fixed final/milestone archive descriptor differs",
        )
        edge(seal, path, row["sha256"])
    require(
        milestone["candidate"] == rows["milestone"]
        and rows["final"]["size_bytes"] == rows["milestone"]["size_bytes"]
        and Path(reference["path"]).resolve() == Path(rows["milestone"]["path"]).resolve()
        and reference["sha256"] == CANDIDATE_SHA,
        "comparison must reference this exact milestone and final archive",
    )
    return dict(
        comparison_seal_sha256=expected_sha,
        comparison_source_sha256=COMPARISON_CODE_SHA,
        tracking_seal_sha256=TRACKING_SEAL_SHA,
        milestone_summary_sha256=MILESTONE_SHA,
        comparison_candidate_reference=reference,
        candidate_archive_references=rows,
        comparison_candidate_copy_available=False,
        consumed_role="comparison-referenced milestone; no fabricated comparison copy",
        final_milestone_archive_byte_identity_required_for_this_fixed_run=True,
    )


def load_candidate(binding, evidence):
    """Guard both same-byte archives; decode only the comparison-referenced milestone."""
    rows = binding["candidate_archive_references"]
    for row in rows.values():
        path = Path(row["path"])
        require(path.stat().st_size == row["size_bytes"], "candidate archive resized")
        evidence.add(path, row["sha256"])
    with np.load(rows["milestone"]["path"], allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)) == 21, "21 unique members required")
        raw = {key: archive[key] for key in archive.files}
    comparison.metrics.validate_candidate(raw, FIRST, STOP)
    signatures = signature(raw)
    require(
        signatures == binding["comparison_candidate_reference"]["arrays"],
        "raw all21 signature differs from successful comparison",
    )
    for value in raw.values():
        value.setflags(write=False)
    return raw, signatures


def run(comparison_sha, output):
    require(EXECUTION_READY, "source-only preparation; OldFast ID execution is not enabled")
    cpu_only()
    output = output.resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh output required")
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
        binding = comparison_headers(comparison_sha, evidence)
        raw, signatures = load_candidate(binding, evidence)
        binding["raw21_signatures"] = signatures
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[START, STOP],
                native_frames=STOP - START,
                retained_raw_interval=[FIRST, STOP],
                **binding,
                source_sha256=code,
                prior_minute_qualification=qualified,
                beam_width=WIDTH,
                dp_config=CONFIG,
                node_indices=list(NODES),
                initial_P=[0, 1, 2],
                initial_P_scope="numeric slot convention only",
                known_body_mixing_development_frames=[77709, 77723],
                gaps_conditional_on_retained_graph=True,
                gaps_are_probabilities=False,
                global_optimality_certified=False,
                physical_ID_truth=False,
                automatic_adoption=False,
                GPU_used=False,
                exact_DP_rerun=False,
            ),
        )
        part = scientific_geometry(raw)
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
        failure = frozen.failure_record(error, calls)
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
