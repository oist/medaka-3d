"""One frozen full-interval ID beam solve for the successful resumed Nodewise path.

The resumed seal has a different lifecycle/schema from the original caller.
Validate that schema explicitly, then reuse its qualified CPU solver unchanged.
The six-field remapped stream preserves stored support/source/review evidence.
No labels, images, fitting, tracking, endpoint constraints or ID resets are used.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import compare_resumed_nodewise_five_minutes as resumed
import numpy as np
import run_nodewise_body_row_five_minute_identity_beam as previous

proof, require, common = previous.proof, previous.require, previous.common
metric, frozen = resumed.metric, previous.frozen
WORK = previous.WORK
FIRST, START, STOP = previous.FIRST, previous.START, previous.STOP
STAGE = "resumed_nodewise_five_minute_identity_beam256_v1"
COMPARISON_ROOT = WORK / resumed.STAGE
PRODUCER_ROOT = resumed.NODEWISE_ROOT.resolve()
COMPARISON_SHA = "22f2c68aaa0fe4a2975eb1e95f383a7762b1ce078f6bcac82b4d2ba7e38ffffd"
PREVIOUS_SHA = "dbb289f4fc154309e0aa3a372a693363da40d7ed4ae23c54288ae1d197fbe0a0"
PRODUCER_SOURCE_SHA = "30c729dc668be636abf2df33a9b2657d3291ab01a899a3da952a3126b0dbad5f"
FIELDS = metric.FIELDS
FISH_FIELDS = ("pose_xyz", "observed", "sources", "identity_review")


def source_code(evidence):
    code = previous.source_code(evidence)
    for path, digest in (
        (Path(previous.__file__), PREVIOUS_SHA),
        (Path(resumed.__file__), COMPARISON_SHA),
        (Path(__file__), None),
    ):
        code[str(path.resolve())] = evidence.add(path, digest)
    return code


def comparison_headers(digest, evidence):
    require(proof.valid_sha(digest), "explicit successful resumed comparison SHA required")
    path = COMPARISON_ROOT / "comparison.provenance.json"
    evidence.add(path, digest)
    seal = proof.read_json(path)
    common.exact_fields(
        seal, dict(stage=resumed.STAGE, status="ok"), "successful resumed comparison required"
    )
    previous.edge(seal, Path(resumed.__file__), COMPARISON_SHA)
    comparison = previous.document(COMPARISON_ROOT, seal, "comparison.json", evidence)
    common.exact_fields(
        comparison,
        dict(
            schema=resumed.STAGE,
            status="ok",
            scientific_interval=[START, STOP],
            native_frames=STOP - START,
            arms=["sparse", "nodewise"],
            sparse_tracking_sha256=metric.SPARSE_SHA,
            candidate_modified=False,
            support_recomputed=False,
            physical_ID_truth=False,
            temporal_ID_mapping_computed=False,
        ),
        "same-horizon raw resumed comparison required",
    )
    tracking_sha = comparison["nodewise_tracking_sha256"]
    previous.edge(seal, PRODUCER_ROOT / "tracking.provenance.json", tracking_sha)
    item = resumed.load_nodewise(PRODUCER_ROOT, tracking_sha, evidence)
    runner = WORK / "resume_nodewise_body_row_five_minutes_099600.py"
    previous.edge(seal, runner, PRODUCER_SOURCE_SHA)
    previous.edge(item["seal"], runner, PRODUCER_SOURCE_SHA)
    require(item["root"] == PRODUCER_ROOT, "different resumed producer root")
    for member in ("summary.json", "plan.json", "input_index.json", "candidate.npz"):
        row = item["seal"]["outputs"][member]
        previous.edge(seal, row["path"], row["sha256"])
    reference = dict(
        root=str(PRODUCER_ROOT),
        tracking_seal_sha256=tracking_sha,
        candidate=copy.deepcopy(item["summary"]["candidate"]),
        all21_declared_signatures=copy.deepcopy(item["summary"]["candidate_arrays"]),
        decoded_field_signatures={k: item["summary"]["candidate_arrays"][k] for k in FIELDS},
    )
    return dict(item=item, reference=reference)


def load_candidate(headers, evidence):
    raw = metric.load_candidate(headers["item"], evidence)
    signatures = common.signature(raw)
    require(
        signatures == headers["reference"]["decoded_field_signatures"],
        "resumed decoded-field reference differs",
    )
    return raw, signatures


def remap_stream(raw, mapping):
    """Reindex whole fish on the scientific interval; source row values stay native."""
    selection = slice(START - FIRST, STOP - FIRST)
    count = STOP - START
    require(
        mapping.shape == (count, 3)
        and mapping.dtype.kind in "iu"
        and np.array_equal(np.sort(mapping, axis=1), np.broadcast_to(np.arange(3), (count, 3))),
        "complete bijective beam-ID to raw-slot mapping required",
    )
    frames = np.arange(count)[:, None]
    result = {}
    for key in FIELDS:
        value = raw[key][selection]
        result[key] = value[frames, mapping] if key in FISH_FIELDS else value.copy()
        result[key].flags.writeable = False
    require(
        np.array_equal(result["frame_idx"], np.arange(START, STOP)),
        "remapped stream scientific interval differs",
    )
    for key in FISH_FIELDS:
        restored = result[key][frames, np.argsort(mapping, axis=1)]
        require(np.array_equal(restored, raw[key][selection]), "remap changed evidence: " + key)
    return result


def run(comparison_sha, output):
    frozen.cpu_only()
    output = Path(output).resolve()
    require(output.is_relative_to(WORK) and not output.exists(), "fresh workspace output required")
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    calls, raw, before = 0, None, None
    try:
        code = source_code(evidence)
        qualified = frozen.qualification(evidence)
        headers = comparison_headers(comparison_sha, evidence)
        raw, before = load_candidate(headers, evidence)
        part = frozen.scientific_geometry(raw)
        for path in code:
            out.write(
                "code_snapshot/" + Path(path).relative_to(proof.REPO).as_posix(),
                Path(path).read_bytes(),
            )
        scope = dict(
            schema=STAGE,
            native_interval=[START, STOP],
            native_frames=STOP - START,
            retained_raw_interval=[FIRST, STOP],
            terminal_horizon=STOP,
            source_sha256=code,
            comparison_seal_sha256=comparison_sha,
            comparison_source_sha256=COMPARISON_SHA,
            tracking_source_sha256=PRODUCER_SOURCE_SHA,
            raw_candidate_reference=headers["reference"],
            decoded_field_signatures=before,
            decoded_candidate_fields=list(FIELDS),
            undecoded15_numerically_verified=False,
            beam_source_sha256=frozen.BEAM_SHA,
            beam_width=frozen.WIDTH,
            node_indices=list(frozen.NODES),
            dp_config=frozen.CONFIG,
            initial_P=[0, 1, 2],
            initialization_frame=START,
            unary="float64 exact zero T,6",
            intermediate_ID_resets=0,
            global_optimality_certified=False,
            gaps_conditional_on_retained_graph=True,
            gaps_are_probabilities=False,
            physical_ID_truth=False,
            biological_naming_applied=False,
            review_flags_cleared=False,
            remapped_review_rule="permutation of raw per-fish flags; no clearing or acceptance",
            raw_candidate_archive_modified=False,
            labels_or_endpoint_constraints_read=False,
            GPU_used=False,
            tracking_rerun=False,
        )
        plan = out.write(
            "plan.json",
            dict(**scope, solve_calls_planned=1, control_qualification=qualified),
        )
        evidence.final_guard()
        out.write(
            "freeze_before_solve.json",
            dict(
                frozen_at=proof.utc(),
                plan=plan,
                inputs=dict(evidence.paths),
                solve_calls_started=0,
                female_labels_read=False,
            ),
        )
        print(
            f"Frozen plan {plan['sha256']}; starting one full {START}:{STOP} CPU solve", flush=True
        )
        tick = time.perf_counter()
        calls += 1
        result = frozen.solve_once(part)
        elapsed = time.perf_counter() - tick
        ledger, ledger_check = frozen.checked_ledger(part, result)
        resources = frozen.resource_report(result)
        remapped = remap_stream(raw, result.permutation_by_id)
        require(common.signature(raw) == before, "search or remapping changed decoded inputs")
        frozen.write_result(out, result, ledger)
        out.write("independent_path_ledger_check.json", ledger_check)
        remapped_record = out.write("beam_remapped_stream.npz", remapped)
        out.write(
            "summary.json",
            frozen.explicit_json(
                dict(
                    **scope,
                    status="ok",
                    solve_calls=calls,
                    solve_seconds=elapsed,
                    path_cost=result.path_cost,
                    resources=resources,
                    numerical_tie_tolerance=result.numerical_tie_tolerance,
                    conditional_global_alternative_gap=result.conditional_global_alternative_gap,
                    conditional_global_alternative_status=result.conditional_global_alternative_status,
                    conditional_tied_frames=int(
                        (result.conditional_optimal_state_mask.sum(1) > 1).sum()
                    ),
                    conditional_unknown_ID_cells=int(
                        np.sum(
                            result.conditional_identity_alternative_status == "not_retained_unknown"
                        )
                    ),
                    pruning_occurred=result.pruning_occurred,
                    finite_prefixes_discarded=result.finite_prefixes_discarded,
                    candidate_npz_decodes=1,
                    decoded_fields_unchanged=True,
                    support_and_geometry_reestimated=False,
                    exact_DP_rerun=False,
                    automatic_adoption=False,
                    beam_remapped_stream=remapped_record,
                    beam_remapped_signatures=common.signature(remapped),
                    raw_beam_disagreement_frames=int(
                        np.any(result.permutation_by_id != np.arange(3), axis=1).sum()
                    ),
                )
            ),
        )
        frozen.cpu_only()
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
        failure = frozen.failure_record(error, calls)
        if raw is not None:
            failure["decoded_input_fields_unchanged"] = common.signature(raw) == before
        out.write("failure.json", failure)
        seal = dict(
            stage=STAGE,
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            **failure,
        )
        (output / "identity.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "identity.provenance.json").write_bytes(proof.json_bytes(seal))
    print(f"Complete: {output}; solve seconds={elapsed:.3f}", flush=True)
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison_sha, args.output)
