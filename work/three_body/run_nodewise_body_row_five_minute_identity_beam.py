"""CPU-only one-call frozen approximate ID search on final Nodewise five minutes.

Requires the successful final comparison and tracking seals. Decode only native
frame, latent pose and stored direct-support fields from the unchanged producer
candidate21. Remaining fields are bound by the archive and producer declarations,
not independently decoded. Fixed initial P012 is a numeric convention, not sex or
physical identity truth. No reset, start shift, retry or increased resource cap.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import compare_nodewise_body_row_five_minutes as comparison
import numpy as np
import run_sparse_instance_five_minute_identity_beam as frozen

proof, require, common = comparison.proof, comparison.require, comparison.common
WORK = comparison.WORK
FIRST, START, STOP = comparison.FIRST, comparison.START, comparison.STOP
STAGE = "nodewise_body_row_five_minute_identity_beam256_v1"
ARM = "nodewise"
COMPARISON_SHA = "512c46a8aee7b5ce7182fc86c282033ad34408509c2a32315447620c930f921e"
FROZEN_SHA = "d433ce526804f5d72594b942c6ab93e9d8e6a2b9b6c1ae46e3f276b6cc925631"
FIELDS = ("frame_idx", "pose_xyz", "observed")
PINS = {
    **comparison.PINS,
    "compare_nodewise_body_row_five_minutes.py": COMPARISON_SHA,
    "run_sparse_instance_five_minute_identity_beam.py": FROZEN_SHA,
    "identity_node_memory_beam.py": frozen.BEAM_SHA,
    "compare_sparse_instance_five_minutes.py": frozen.COMPARISON_SHA,
}


def source_code(evidence):
    """Only explicit used helpers; never invoke an old entrypoint's dynamic closure."""
    code = {}
    for name, digest in PINS.items():
        path = (WORK / name).resolve()
        code[str(path)] = evidence.add(path, digest)
    for path, digest in (
        (Path(frozen.beam.frozen.__file__), frozen.beam.FROZEN_SHA),
        (frozen.beam.LAB / "identity_dp.py", frozen.beam.CORE_SHA),
    ):
        code[str(path.resolve())] = evidence.add(path, digest)
    code[str(Path(__file__).resolve())] = evidence.add(Path(__file__))
    frozen.beam.verify_sources()
    return code


def document(root, seal, name, evidence):
    return proof.read_json(common.bound_member(root, seal, name, evidence))


def edge(seal, path, digest):
    require(
        proof.valid_sha(digest) and seal["inputs"].get(str(Path(path).resolve())) == digest,
        "missing sealed source/input edge",
    )


def comparison_headers(digest, evidence):
    require(proof.valid_sha(digest), "explicit successful comparison SHA required")
    root = WORK / comparison.STAGE
    evidence.add(root / "comparison.provenance.json", digest)
    seal = proof.read_json(root / "comparison.provenance.json")
    common.exact_fields(
        seal, dict(stage=comparison.STAGE, status="ok"), "successful complete comparison required"
    )
    edge(seal, WORK / "compare_nodewise_body_row_five_minutes.py", COMPARISON_SHA)
    summary = document(root, seal, "summary.json", evidence)
    plan = document(root, seal, "plan.json", evidence)
    for doc in (summary, plan):
        common.exact_fields(
            doc,
            dict(schema=comparison.STAGE, scientific_interval=[START, STOP], terminal_horizon=STOP),
            "same final science horizon required",
        )
    common.exact_fields(
        summary,
        dict(
            status="ok",
            arms=list(comparison.ARMS),
            native_frames=STOP - START,
            candidate_modified=False,
            support_recomputed=False,
            review_cleared=False,
            physical_ID_truth=False,
            temporal_ID_mapping_computed=False,
        ),
        "comparison did not preserve raw tracking results",
    )
    require(
        plan["retained_interval"] == [FIRST, STOP]
        and plan["decoded_fields"] == list(comparison.FIELDS)
        and plan["undecoded15_numerically_verified"] is False
        and plan["candidate_references"] == summary["candidate_references"]
        and set(plan["candidate_references"]) == set(comparison.ARMS),
        "new comparison reference/schema differs",
    )
    lineage = plan["lineage"]
    require(
        lineage["original_tracking_sha256"] == comparison.ORIGINAL_SHA
        and lineage["original_checkpoint_sha256"] == comparison.ORIGINAL_CHECKPOINT_SHA
        and lineage["new_import_all39_signatures_equal"] is True
        and lineage["same_complete40_source_prefix"] is True
        and lineage["nodewise_backend_and_body_start"] == START,
        "same original66400 comparison lineage required",
    )
    reference = copy.deepcopy(plan["candidate_references"][ARM])
    require(
        set(reference)
        == {
            "root",
            "tracking_seal_sha256",
            "candidate",
            "all21_declared_signatures",
            "decoded_field_signatures",
        },
        "explicit new comparison candidate schema required",
    )
    producer_root = (WORK / comparison.NODEWISE_STAGE).resolve()
    require(Path(reference["root"]).resolve() == producer_root, "different producer stage/root")
    producer_path = producer_root / "tracking.provenance.json"
    edge(seal, producer_path, reference["tracking_seal_sha256"])
    evidence.add(producer_path, reference["tracking_seal_sha256"])
    producer = proof.read_json(producer_path)
    common.exact_fields(
        producer,
        dict(stage=comparison.NODEWISE_STAGE, status="ok"),
        "Nodewise has not successfully completed",
    )
    runner = WORK / "run_nodewise_body_row_five_minutes.py"
    edge(producer, runner, comparison.NODEWISE_RUNNER_SHA)
    edge(seal, runner, comparison.NODEWISE_RUNNER_SHA)
    evidence.add(runner, comparison.NODEWISE_RUNNER_SHA)
    produced = document(producer_root, producer, "summary.json", evidence)
    common.exact_fields(
        produced,
        dict(
            schema=comparison.NODEWISE_STAGE,
            status="ok",
            process_interval=[START, STOP],
            scientific_interval=[START, STOP],
            retained_interval=[FIRST, STOP],
            native_frames=STOP - START,
            initialize_count=1,
            restore_count=4,
            new_begin_calls=0,
            external_pose_resets=0,
            finish_calls=1,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            finish_matches_checkpoint_backtrace=True,
            mask_split_priority_applied=False,
        ),
        "full continuous Nodewise producer required",
    )
    require(
        reference["candidate"] == producer["outputs"]["candidate.npz"] == produced["candidate"]
        and reference["all21_declared_signatures"] == produced["candidate_arrays"]
        and set(reference["all21_declared_signatures"]) == comparison.ALL21
        and reference["decoded_field_signatures"]
        == {k: reference["all21_declared_signatures"][k] for k in comparison.FIELDS},
        "producer final candidate/selected signatures differ",
    )
    edge(seal, Path(reference["candidate"]["path"]), reference["candidate"]["sha256"])
    return dict(
        comparison_seal=seal,
        comparison_sha256=digest,
        reference=reference,
        producer_root=producer_root,
        producer_seal=producer,
        lineage=lineage,
    )


def validate_geometry(raw):
    count = STOP - FIRST
    require(set(raw) == set(FIELDS), "only three geometry/support fields required")
    for key, shape, dtype in (
        ("frame_idx", (count,), np.int64),
        ("pose_xyz", (count, 3, 3, 3), np.float64),
        ("observed", (count, 3, 3), np.bool_),
    ):
        require(
            raw[key].shape == shape and raw[key].dtype == dtype, "candidate shape/dtype: " + key
        )
    require(
        np.array_equal(raw["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64)),
        "exact retained native frame range required",
    )
    require(np.isfinite(raw["pose_xyz"]).all(), "finite latent geometry required")


def load_candidate(headers, evidence):
    ref = headers["reference"]
    path = common.bound_member(
        headers["producer_root"], headers["producer_seal"], "candidate.npz", evidence
    )
    require(Path(ref["candidate"]["path"]).resolve() == path, "different final candidate path")
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == 21 and set(archive.files) == comparison.ALL21,
            "candidate21 membership differs",
        )
        raw = {k: archive[k] for k in FIELDS}
    validate_geometry(raw)
    signatures = common.signature(raw)
    require(
        signatures == {k: ref["decoded_field_signatures"][k] for k in FIELDS},
        "decoded geometry/support signature differs",
    )
    for value in raw.values():
        value.flags.writeable = False
    return raw, signatures


def run(comparison_sha, output):
    frozen.cpu_only()
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "fresh independent workspace output"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    calls, raw, before = 0, None, None
    try:
        code = source_code(evidence)
        qualified = frozen.qualification(evidence)
        headers = comparison_headers(comparison_sha, evidence)
        raw, before = load_candidate(headers, evidence)
        part = frozen.scientific_geometry(raw)
        binding = dict(
            comparison_seal_sha256=comparison_sha,
            comparison_source_sha256=COMPARISON_SHA,
            tracking_seal_sha256=headers["reference"]["tracking_seal_sha256"],
            tracking_source_sha256=comparison.NODEWISE_RUNNER_SHA,
            raw_candidate_reference=headers["reference"],
            decoded_field_signatures=before,
            decoded_candidate_fields=list(FIELDS),
            undecoded18_numerically_verified=False,
        )
        scope = dict(
            schema=STAGE,
            native_interval=[START, STOP],
            native_frames=STOP - START,
            retained_raw_interval=[FIRST, STOP],
            terminal_horizon=STOP,
            source_sha256=code,
            **binding,
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
            raw_candidate_archive_modified=False,
            raw_sources_or_review_decoded=False,
        )
        out.write(
            "plan.json",
            dict(
                **scope,
                solve_calls_planned=1,
                control_qualification=qualified,
                producer_lineage=headers["lineage"],
            ),
        )
        tick = time.perf_counter()
        calls += 1
        result = frozen.solve_once(part)
        elapsed = time.perf_counter() - tick
        ledger, ledger_check = frozen.checked_ledger(part, result)
        resources = frozen.resource_report(result)
        require(common.signature(raw) == before, "input geometry/support modified by search")
        frozen.write_result(out, result, ledger)
        out.write("independent_path_ledger_check.json", ledger_check)
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
                    checkpoint39_SL_video_mask_GPU_reads=False,
                    automatic_adoption=False,
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
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison_sha, args.output)
