"""Disabled CPU-only, one-call approximate ID search on the final Sparse five minutes.

A successful same-horizon comparison binds immutable candidate21 geometry.
The fixed width256 retained-memory beam is called once on all 49,800 scientific
frames. Initial P012 is a slot convention, not physical identity evidence.
No exact-DP retry, segmentation, fitting, boundary change or ID reset occurs.
Every gap is conditional on the retained graph, never a probability or global
certificate. Capacity failure returns no partial successful identity solution.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import time
from pathlib import Path

import compare_sparse_instance_five_minutes as comparison
import identity_node_memory_beam as beam
import numpy as np

proof, require = comparison.proof, comparison.require
WORK, REPO = comparison.WORK, comparison.REPO
FIRST, START, STOP = 49800, 66400, 116200
ARM = "sparse_instance_rescore"
STAGE = "sparse_instance_five_minute_identity_beam256_v1"
COMPARISON_ROOT = WORK / comparison.STAGE
COMPARISON_SHA = "1ec0893d7dd2bbc746004f71d9b81abc60096cc0173bf44871f56be948e1c8dc"
BEAM_SHA = "019791d34fda3292aa35371ba3aa8297c3f92ace1e19e8ffe514cd39fd4c92bd"
QUALIFICATION_ROOT = WORK / "identity_beam_exact_minute_qualification_v1"
QUALIFICATION_SHA = "0776b3a5ded87d9aa524d8ae6b84bc96ae5c4c14acd5083785ab6f856accb0b1"
EXECUTION_READY = False
NODES, WIDTH = (1, 0), 256
CONFIG = dict(
    scale_mm=0.8,
    weight=0.2,
    initial_state=0,
    max_layer_states=100_000,
    max_total_states=1_000_000,
    max_total_edges=6_000_000,
    tie_atol=1e-10,
    tie_rtol=1e-12,
)
TERMINAL = (
    "terminal_last_event_frame",
    "terminal_last_event_slot",
    "terminal_age_frames",
    "terminal_unsettled_gap",
)
MAPPING_FIELDS = (
    "frame_idx",
    "path_state",
    "permutation_by_id",
    "conditional_min_marginal_delta",
    "conditional_id_slot_min_marginal_delta",
    "conditional_optimal_state_mask",
    "conditional_state_alternative_gap",
    "conditional_identity_alternative_gap",
    "conditional_state_alternative_status",
    "conditional_identity_alternative_status",
    *TERMINAL,
)


def cpu_only():
    torch = sys.modules.get("torch")
    require(
        torch is None or not torch.cuda.is_initialized(), "identity caller must remain CPU-only"
    )


def source_code():
    code = comparison.source_code()
    for path, digest in (
        (Path(comparison.__file__), COMPARISON_SHA),
        (Path(beam.__file__), BEAM_SHA),
        (Path(beam.frozen.__file__), beam.FROZEN_SHA),
        (beam.LAB / "identity_dp.py", beam.CORE_SHA),
    ):
        require(code.get(str(path.resolve())) == digest, "frozen identity/reporting source differs")
    require(
        str(Path(__file__).resolve()) in code, "identity caller missing from actual source closure"
    )
    beam.verify_sources()
    return code


def document(root, seal, name, evidence):
    return proof.read_json(comparison.common.bound_member(root, seal, name, evidence))


def qualification(evidence):
    path = QUALIFICATION_ROOT / "qualification.provenance.json"
    evidence.add(path, QUALIFICATION_SHA)
    seal = proof.read_json(path)
    require(
        seal["stage"] == QUALIFICATION_ROOT.name and seal["status"] == "ok",
        "successful width256 minute qualification required",
    )
    require(
        seal["inputs"].get(str(Path(beam.__file__).resolve())) == BEAM_SHA,
        "qualified beam source differs",
    )
    summary = document(QUALIFICATION_ROOT, seal, "summary.json", evidence)
    plan = document(QUALIFICATION_ROOT, seal, "plan.json", evidence)
    require(
        summary["status"] == "ok"
        and summary["native_interval"] == [66400, 76360]
        and plan["dp_config"] == CONFIG
        and plan["node_indices"] == list(NODES)
        and plan["initial_P"] == [0, 1, 2]
        and plan["beam_source_sha256"] == BEAM_SHA,
        "qualified objective differs",
    )
    width = summary["widths"]["256"]
    require(
        width["beam_width"] == WIDTH
        and width["same_minute_path_and_terminal_preserved"] is True
        and width["global_optimality_certified"] is False
        and width["review_cleared"] is False,
        "qualified result or its uncertainty scope differs",
    )
    return dict(
        seal_sha256=QUALIFICATION_SHA,
        scope="only the saved exact-control minute [66400,76360)",
        whole_five_minute_qualification=False,
        physical_ID_truth=False,
        qualified_retained_states=width["retained_states"],
        qualified_generated_edges=width["expanded_edges"],
        naive_times_five_retained_states=5 * width["retained_states"],
        naive_times_five_generated_edges=5 * width["expanded_edges"],
        naive_scaling_is_not_a_capacity_guarantee=True,
    )


def comparison_headers(expected_sha, evidence):
    require(proof.valid_sha(expected_sha), "explicit successful comparison SHA required")
    path = COMPARISON_ROOT / "comparison.provenance.json"
    evidence.add(path, expected_sha)
    seal = proof.read_json(path)
    require(
        seal["stage"] == comparison.STAGE and seal["status"] == "ok",
        "successful five-minute comparison required before candidates",
    )
    require(
        seal["inputs"].get(str(Path(comparison.__file__).resolve())) == COMPARISON_SHA,
        "comparison was produced by a different source",
    )
    summary = document(COMPARISON_ROOT, seal, "summary.json", evidence)
    plan = document(COMPARISON_ROOT, seal, "plan.json", evidence)
    for doc in (summary, plan):
        require(
            doc["schema"] == comparison.STAGE
            and doc["native_interval"] == [START, STOP]
            and doc["native_frames"] == STOP - START
            and doc["terminal_horizon_by_arm"] == dict.fromkeys(comparison.ARMS, STOP),
            "comparison interval or terminal horizon differs",
        )
    require(
        summary["status"] == "ok"
        and summary["review_modified"] is False
        and summary["candidate_modified"] is False
        and summary["human_truth_created"] is False
        and summary["all_three_physical_ID_truth"] is False,
        "comparison raw/truth contract differs",
    )
    require(
        plan["candidate_references"] == summary["candidate_references"],
        "plan/summary candidate bindings differ",
    )
    for arm in comparison.ARMS:
        reference = plan["candidate_references"][arm]
        root = WORK / comparison.SPECS[arm]["stage"]
        producer_path = root / "tracking.provenance.json"
        digest = plan["tracking_seals"][arm]
        require(
            proof.valid_sha(digest) and seal["inputs"].get(str(producer_path.resolve())) == digest,
            "comparison lacks the final producer seal edge",
        )
        evidence.add(producer_path, digest)
        producer = proof.read_json(producer_path)
        require(
            producer["status"] == "ok" and producer["stage"] == root.name,
            "both final producer seals must succeed",
        )
        require(
            producer["inputs"].get(str((WORK / comparison.SPECS[arm]["runner"]).resolve()))
            == comparison.SPECS[arm]["source_sha"],
            "final tracking implementation differs",
        )
        require(
            reference["terminal_horizon"] == STOP and len(reference["arrays"]) == 21,
            "final all21 same-horizon reference required",
        )
        for member, path_key, sha_key in (
            ("candidate.npz", "path", "sha256"),
            (f"milestones/through_{STOP}/candidate.npz", "milestone_path", "milestone_sha256"),
        ):
            named = str((root / member).resolve())
            row = producer["outputs"][member]
            require(
                str(Path(reference[path_key]).resolve())
                == named
                == str(Path(row["path"]).resolve())
                and proof.valid_sha(reference[sha_key])
                and reference[sha_key] == row["sha256"]
                and seal["inputs"].get(named) == reference[sha_key],
                "final/milestone candidate edge differs",
            )
    return seal, copy.deepcopy(plan["candidate_references"][ARM])


def load_candidate(seal, reference, evidence):
    # Hash guards may reread bytes; the candidate archive is decoded only once.
    path = comparison.common.bound_member(COMPARISON_ROOT, seal, f"candidates/{ARM}.npz", evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == 21,
            "exactly 21 unique NPZ members required",
        )
        raw = {key: archive[key] for key in archive.files}
    comparison.metrics.validate_candidate(raw, FIRST, STOP)
    signatures = comparison.common.signature(raw)
    require(
        signatures == reference["arrays"],
        "comparison candidate does not match all21 reference bytes",
    )
    for value in raw.values():
        value.setflags(write=False)
    return raw, signatures, copy.deepcopy(seal["outputs"][f"candidates/{ARM}.npz"])


def scientific_geometry(raw):
    selection = slice(START - FIRST, STOP - FIRST)
    part = {key: raw[key][selection] for key in ("frame_idx", "pose_xyz", "observed")}
    require(
        np.array_equal(part["frame_idx"], np.arange(START, STOP, dtype=np.int64)),
        "exact scientific interval required; no shifted start",
    )
    require(
        part["observed"][0][:, NODES].all(),
        "first scientific frame lacks direct chest/mouth; no shifted initialization",
    )
    return part


def solve_once(part):
    """One call, entire interval. Capacity exceptions propagate without retry/reset."""
    return beam.solve_identity_node_memory_beam(
        part["frame_idx"],
        np.zeros((STOP - START, 6), np.float64),
        part["pose_xyz"][:, :, NODES, :],
        part["observed"][:, :, NODES],
        beam_width=WIDTH,
        node_indices=NODES,
        **CONFIG,
    )


def checked_ledger(part, result):
    require(
        type(result) is beam.IdentityNodeMemoryBeamResult
        and result.schema == beam.SCHEMA
        and result.status == "optimal_on_retained_graph"
        and result.global_optimality_certified is False
        and result.gaps_conditional_on_retained_graph is True
        and result.physical_ID_truth is False
        and result.review_flags_cleared is False
        and result.intermediate_ID_reset is False,
        "complete distinctly approximate result required",
    )
    require(
        np.array_equal(result.frame_idx, part["frame_idx"])
        and result.beam_width == WIDTH
        and result.node_indices == NODES
        and all(
            getattr(result, k) == v for k, v in CONFIG.items() if k not in {"tie_atol", "tie_rtol"}
        )
        and result.numerical_tie_tolerance
        == CONFIG["tie_atol"] + CONFIG["tie_rtol"] * result.numerical_cost_span,
        "solve extent or frozen configuration differs",
    )
    mapping = result.permutation_by_id
    require(
        mapping.shape == (STOP - START, 3)
        and mapping.dtype.kind in "iu"
        and np.array_equal(np.sort(mapping, axis=1), np.broadcast_to(np.arange(3), mapping.shape))
        and np.array_equal(mapping[0], [0, 1, 2]),
        "one bijective whole-fish mapping per frame with initial P012 required",
    )
    require(
        result.path_state.shape == (STOP - START,)
        and result.path_state.dtype.kind in "iu"
        and np.all((result.path_state >= 0) & (result.path_state < 6))
        and np.array_equal(beam.frozen.PERMUTATIONS[result.path_state], mapping),
        "path-state mapping differs",
    )
    ledger = beam.frozen.node_memory_path_links(
        part["frame_idx"],
        part["pose_xyz"][:, :, NODES, :],
        part["observed"][:, :, NODES],
        mapping,
        node_indices=NODES,
        scale_mm=0.8,
        weight=0.2,
    )
    require(
        np.isfinite(result.path_cost) and result.path_cost >= 0,
        "finite nonnegative motion objective required",
    )
    tolerance = max(
        1e-10,
        64
        * np.finfo(np.float64).eps
        * (STOP - START)
        * (1 + max(result.path_cost, ledger["motion_cost"])),
    )
    difference = float(ledger["motion_cost"] - result.path_cost)
    terminal = {key: np.array_equal(ledger[key], getattr(result, key)) for key in TERMINAL}
    require(
        abs(difference) <= tolerance and all(terminal.values()),
        "independent motion ledger or terminal memory differs",
    )
    return ledger, dict(
        motion_cost=ledger["motion_cost"],
        reported_path_cost=result.path_cost,
        signed_difference=difference,
        tolerance=tolerance,
        terminal_arrays_exact=terminal,
    )


def resource_report(result):
    retained, expanded_edges = result.retained_state_counts, result.expanded_edge_counts
    require(
        retained.shape == expanded_edges.shape == (STOP - START,)
        and retained[0] == 1
        and np.all((retained >= 1) & (retained <= WIDTH))
        and expanded_edges[0] == 0
        and np.array_equal(expanded_edges[1:], 6 * retained[:-1]),
        "per-frame resource accounting differs",
    )
    require(
        np.array_equal(np.diff(result.layer_offsets), retained)
        and int(retained.sum()) == len(result.state_permutation)
        and int(result.retained_edge_counts.sum()) == len(result.edge_source),
        "saved graph resource totals differ",
    )
    require(
        int(retained.sum()) <= CONFIG["max_total_states"]
        and int(expanded_edges.sum()) <= CONFIG["max_total_edges"]
        and int(result.expanded_state_counts.max()) <= CONFIG["max_layer_states"],
        "successful result exceeds registered cap",
    )
    return dict(
        retained_states=int(retained.sum()),
        generated_edges=int(expanded_edges.sum()),
        retained_edges=int(result.retained_edge_counts.sum()),
        expanded_states=int(result.expanded_state_counts.sum()),
        discarded_states=int(result.discarded_state_counts.sum()),
        discarded_finite_states=int(result.discarded_finite_state_counts.sum()),
        largest_expanded_layer=int(result.expanded_state_counts.max()),
        largest_retained_layer=int(retained.max()),
        cap_scopes=dict(
            states="stored retained nodes",
            edges="all generated including discarded",
            layer="merged expanded before pruning",
        ),
    )


def explicit_json(value):
    if isinstance(value, np.ndarray):
        return explicit_json(value.tolist())
    if isinstance(value, np.generic):
        return explicit_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: explicit_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [explicit_json(v) for v in value]
    return value


def write_result(out, result, ledger):
    arrays, scalars = {}, {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            require(not value.flags.writeable, "solver result arrays must remain immutable")
            arrays[field.name] = value
        else:
            scalars[field.name] = value
    out.write("retained_graph_and_diagnostics.npz", arrays)
    out.write("beam_scalars.json", explicit_json(scalars))
    out.write("identity_mapping_and_ambiguity.npz", {key: arrays[key] for key in MAPPING_FIELDS})
    out.write(
        "independent_path_ledger.npz",
        {key: v for key, v in ledger.items() if isinstance(v, np.ndarray)},
    )


def failure_record(error, calls):
    record = dict(
        status="failed",
        error=repr(error),
        solve_calls=calls,
        complete_identity_result=False,
        partial_identity_solution_published=False,
        automatic_retry=False,
        cap_increase=False,
        intermediate_ID_reset=False,
        physical_ID_truth=False,
    )
    if isinstance(error, beam.IdentityBeamCapacityError):
        record.update(
            reason="registered_capacity_exceeded",
            frame_offset=error.frame_offset,
            native_frame=START + error.frame_offset,
            resource=error.resource,
            attempted=error.attempted,
            limit=error.limit,
        )
    return record


def run(comparison_sha, output):
    require(EXECUTION_READY, "source-only preparation; five-minute ID execution is not enabled")
    cpu_only()
    output = output.resolve()
    require(
        output.is_relative_to(WORK) and not output.exists(), "fresh independent output required"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    calls, raw, signatures, code = 0, None, None, {}
    try:
        code = source_code()
        for path, sha in code.items():
            evidence.add(path, sha)
            out.write(
                "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), Path(path).read_bytes()
            )
        control_qualification = qualification(evidence)
        seal, reference = comparison_headers(comparison_sha, evidence)
        raw, signatures, copy_reference = load_candidate(seal, reference, evidence)
        part = scientific_geometry(raw)
        binding = dict(
            comparison_seal_sha256=comparison_sha,
            comparison_source_sha256=COMPARISON_SHA,
            raw_candidate_reference=reference,
            consumed_comparison_copy=copy_reference,
            raw21_signatures=signatures,
        )
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[START, STOP],
                native_frames=STOP - START,
                retained_raw_interval=[FIRST, STOP],
                terminal_horizon=STOP,
                source_sha256=code,
                **binding,
                beam_source_sha256=BEAM_SHA,
                beam_width=WIDTH,
                node_indices=list(NODES),
                dp_config=CONFIG,
                unary="float64 exact zero T,6",
                initial_P=[0, 1, 2],
                initialization_frame=START,
                solve_calls_planned=1,
                intermediate_ID_resets=0,
                qualification=control_qualification,
                tolerance_rule="max(1e-10,64*eps64*49800*(1+max(path_cost,chronological_cost)))",
                raw21_modified=False,
                review_flags_cleared=False,
                conditional_gaps_are_probabilities=False,
                global_optimality_certified=False,
                physical_ID_truth=False,
                automatic_adoption=False,
                GPU_used=False,
            ),
        )
        tick = time.perf_counter()
        calls = 1
        result = solve_once(part)
        solve_seconds = time.perf_counter() - tick
        ledger, ledger_report = checked_ledger(part, result)
        resources = resource_report(result)
        require(
            comparison.common.signature(raw) == signatures,
            "raw21 geometry/source/review bytes changed",
        )
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
                    retained_raw_interval=[FIRST, STOP],
                    terminal_horizon=STOP,
                    **binding,
                    solve_calls=calls,
                    beam_width=WIDTH,
                    solve_seconds=solve_seconds,
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
                    automatic_adoption=False,
                )
            ),
        )
        cpu_only()
        require(source_code() == code, "source closure changed")
        evidence.final_guard()
        out.final_guard()
        final = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        failure = failure_record(error, calls)
        if raw is not None:
            failure["raw21_unchanged_at_failure"] = comparison.common.signature(raw) == signatures
        out.write("failure.json", failure)
        final = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            **{k: v for k, v in failure.items() if k != "status"},
        )
        with (output / "identity.provenance.json").open("xb") as stream:
            stream.write(proof.json_bytes(final))
        raise
    with (output / "identity.provenance.json").open("xb") as stream:
        stream.write(proof.json_bytes(final))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.comparison_sha, args.output)
