"""One fixed CPU beam-ID call on the saved Nodewise first-minute milestone.

Only the successful comparison's three-field science export is decoded. The
source trajectory ends at 76360 and its five-minute producer may still be live.
Width256, chest/mouth, zero unary and initial66400/P012 are fixed. No fitting,
review clearing, initialization shift, retry, exact-DP rerun or capacity increase.
Every alternative gap is conditional on the retained graph, not a probability.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import identity_node_memory_beam as beam
import numpy as np
import qualify_identity_beam_minute as minute
import run_sparse_instance_five_minute_identity_beam as frozen

proof, common, require = frozen.proof, frozen.comparison.common, frozen.require
WORK, REPO = Path(__file__).resolve().parent, Path(__file__).resolve().parents[2]
FIRST, START, STOP = 49800, 66400, 76360
STAGE = "nodewise_body_row_first_minute_identity_beam256_v1"
COMPARISON_STAGE = "nodewise_body_row_first_minute_comparison_v1"
COMPARISON_SHA = "705ede6ebeee394b7dfa7fc43697c62d0dd9fa23f4208173737ac4a75f170d98"
MILESTONE_SHA = "bed9dd1f0d7192261f4e985a3021097f444a941e180bd1982bad4dbcb01525de"
CHECKPOINT_SHA = "d0cc931c72563597a8bdfcf3ea09ffe614306416b6cb1e6f6a0bf17fdce39f8c"
ORIGINAL_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
ORIGINAL_CHECKPOINT_SHA = "0f8db6356175de3223589576b4539a3a73977eb8151324b94340c5b1b174730f"
PRODUCER_SHA = "0ea3d27b939a804c081d6249c7462498ee0a16504e623e021037b0fc32f50b64"
PRODUCER_STAGE = "nodewise_body_row_five_minute_continuation_v1"
FIELDS = ("frame_idx", "pose_xyz", "observed")
NODES, WIDTH, CONFIG = frozen.NODES, frozen.WIDTH, frozen.CONFIG.copy()
PINS = {
    "run_sparse_instance_five_minute_identity_beam.py": (
        "d433ce526804f5d72594b942c6ab93e9d8e6a2b9b6c1ae46e3f276b6cc925631"
    ),
    "compare_sparse_instance_five_minutes.py": frozen.COMPARISON_SHA,
    "compare_instance_rescore_10s.py": (
        "bb7653e5f7d71f2b4c078418dee93433fb4a1fc3fdefd7f38ef9c4694ae9040a"
    ),
    "compare_temporal_point64_v2.py": (
        "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
    ),
    "identity_node_memory_beam.py": frozen.BEAM_SHA,
    "qualify_identity_beam_minute.py": (
        "16f428cf21a9a7e030b5f7d8ee58e4bc68981284e8814597ac9b7f56293a96c0"
    ),
    "qualify_streaming_tail_minute.py": minute.QUALIFIED_SHA,
    "run_tail_minute_identity.py": minute.qualified.PROOF_SHA,
}


def source_code(evidence):
    require(proof.valid_sha(COMPARISON_SHA), "reviewed comparison source pin required")
    paths = {WORK / name: digest for name, digest in PINS.items()}
    paths.update(
        {
            WORK / "compare_nodewise_body_row_first_minute.py": COMPARISON_SHA,
            Path(beam.frozen.__file__): beam.FROZEN_SHA,
            beam.LAB / "identity_dp.py": beam.CORE_SHA,
        }
    )
    code = {str(path.resolve()): evidence.add(path, digest) for path, digest in paths.items()}
    code[str(Path(__file__).resolve())] = evidence.add(Path(__file__))
    beam.verify_sources()
    return code


def edge(seal, path, digest):
    require(
        proof.valid_sha(digest) and seal["inputs"].get(str(Path(path).resolve())) == digest,
        "missing sealed source/input edge",
    )


def document(root, seal, name, evidence):
    return proof.read_json(common.bound_member(root, seal, name, evidence))


def qualification(evidence):
    # Read only the existing successful width256 qualification's small documents.
    qualified = frozen.qualification(evidence)
    return {
        k: qualified[k]
        for k in (
            "seal_sha256",
            "scope",
            "physical_ID_truth",
            "qualified_retained_states",
            "qualified_generated_edges",
        )
    } | {"old_control_recomputed": False, "new_geometry_exact_optimality_certified": False}


def comparison_headers(digest, evidence):
    require(proof.valid_sha(digest), "explicit successful first-minute comparison SHA required")
    root = WORK / COMPARISON_STAGE
    evidence.add(root / "comparison.provenance.json", digest)
    seal = proof.read_json(root / "comparison.provenance.json")
    common.exact_fields(seal, dict(stage=COMPARISON_STAGE, status="ok"), "successful comparison")
    edge(seal, WORK / "compare_nodewise_body_row_first_minute.py", COMPARISON_SHA)
    plan, summary = (document(root, seal, name + ".json", evidence) for name in ("plan", "summary"))
    expected = dict(
        native_interval=[START, STOP],
        retained_interval=[FIRST, STOP],
        terminal_horizon=STOP,
        original_tracking_sha256=ORIGINAL_SHA,
        original_checkpoint_sha256=ORIGINAL_CHECKPOINT_SHA,
        same_original_metadata_and39_signatures=True,
        common_source_prefix_chunks=16,
        nodewise_source_sha256=PRODUCER_SHA,
        nodewise_milestone_summary_sha256=MILESTONE_SHA,
        nodewise_checkpoint_manifest_sha256=CHECKPOINT_SHA,
        whole_five_minute_completion_claimed=False,
        support_recomputed=False,
        review_cleared=False,
        all21_membership_verified=True,
        undecoded15_per_arm_numerically_verified=False,
        source_candidate_npz_decodes_per_arm=1,
    )
    for doc in (plan, summary):
        common.exact_fields(
            doc, dict(schema=COMPARISON_STAGE, **expected), "comparison lineage/scope"
        )
    require(summary["status"] == "ok", "completed comparison required")
    signatures = document(root, seal, "science_signatures.json", evidence)
    require(
        signatures == summary["science_signatures"] == plan["science_signatures"],
        "science export signatures differ",
    )
    require(set(signatures) == {"tail", "nodewise"}, "same-horizon two-arm comparison required")
    require(set(signatures["nodewise"]) == set(FIELDS), "exact three-field science export")
    for key in ("nodewise_milestone_candidate", "nodewise_all21_declared_signatures"):
        require(plan[key] == summary[key], "milestone candidate reference differs")
    milestone_path = WORK / PRODUCER_STAGE / f"milestones/through_{STOP:06d}/summary.json"
    edge(seal, milestone_path, MILESTONE_SHA)
    evidence.add(milestone_path, MILESTONE_SHA)
    milestone = proof.read_json(milestone_path)
    common.exact_fields(
        milestone,
        dict(
            schema=PRODUCER_STAGE + "_milestone",
            status="ok",
            horizon_stop=STOP,
            scientific_interval=[START, STOP],
            native_frames=STOP - START,
            initialize_count=1,
            restore_count=4,
            finish_calls_so_far=0,
            new_begin_calls=0,
            external_pose_resets=0,
            all39_roundtrip=True,
            prior_history_prefix_unchanged=True,
            whole_process_completion_pending=True,
            physical_ID_truth=False,
        ),
        "committed first-minute milestone required",
    )
    require(
        milestone["candidate"] == plan["nodewise_milestone_candidate"]
        and milestone["candidate_arrays"] == plan["nodewise_all21_declared_signatures"]
        and milestone["checkpoint"]["manifest_sha256"] == CHECKPOINT_SHA,
        "milestone candidate/signature binding differs",
    )
    edge(seal, Path(milestone["candidate"]["path"]), milestone["candidate"]["sha256"])
    edge(seal, WORK / "run_nodewise_body_row_five_minutes.py", PRODUCER_SHA)
    return dict(
        root=root,
        seal=seal,
        comparison_sha256=digest,
        signatures=signatures["nodewise"],
        lineage=expected,
        milestone_summary=milestone,
        science=seal["outputs"]["science/nodewise.npz"],
    )


def load_geometry(headers, evidence):
    path = common.bound_member(headers["root"], headers["seal"], "science/nodewise.npz", evidence)
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == 3 and set(archive.files) == set(FIELDS), "three science members"
        )
        part = {key: archive[key] for key in FIELDS}
    n = STOP - START
    for key, shape, dtype in (
        ("frame_idx", (n,), np.int64),
        ("pose_xyz", (n, 3, 3, 3), np.float64),
        ("observed", (n, 3, 3), np.bool_),
    ):
        require(part[key].shape == shape and part[key].dtype == dtype, "science axes/dtype: " + key)
    require(np.array_equal(part["frame_idx"], np.arange(START, STOP)), "exact unshifted minute")
    require(np.isfinite(part["pose_xyz"]).all(), "finite saved latent geometry required")
    require(part["observed"][0][:, NODES].all(), "first66400 chest/mouth must be direct; no shift")
    require(common.signature(part) == headers["signatures"], "science signatures differ")
    for value in part.values():
        value.flags.writeable = False
    return part


def solve_once(part):
    return beam.solve_identity_node_memory_beam(
        part["frame_idx"],
        np.zeros((len(part["frame_idx"]), 6), np.float64),
        part["pose_xyz"][:, :, NODES, :],
        part["observed"][:, :, NODES],
        beam_width=WIDTH,
        node_indices=NODES,
        **CONFIG,
    )


def check_result(part, result):
    n = len(part["frame_idx"])
    require(
        type(result) is beam.IdentityNodeMemoryBeamResult
        and result.schema == beam.SCHEMA
        and result.status == "optimal_on_retained_graph"
        and not result.global_optimality_certified
        and result.gaps_conditional_on_retained_graph
        and not result.physical_ID_truth
        and not result.review_flags_cleared
        and not result.intermediate_ID_reset
        and not result.terminal_missing_endpoint_cost_applied,
        "complete conditional result required",
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
        "fixed solve interval/configuration differs",
    )
    p, state = result.permutation_by_id, result.path_state
    require(
        p.shape == (n, 3)
        and p.dtype.kind in "iu"
        and np.array_equal(p[0], [0, 1, 2])
        and np.array_equal(np.sort(p, axis=1), np.broadcast_to(np.arange(3), p.shape))
        and state.shape == (n,)
        and state.dtype.kind in "iu"
        and np.all((state >= 0) & (state < 6))
        and np.array_equal(beam.frozen.PERMUTATIONS[state], p),
        "whole-body mapping/state P012 contract differs",
    )
    require(np.isfinite(result.path_cost) and result.path_cost >= 0, "finite nonnegative cost")
    ledger, check = minute.ledger_for(
        part,
        p,
        result.path_cost,
        {k: getattr(result, k) for k in frozen.TERMINAL},
    )
    retained, edges = result.retained_state_counts, result.expanded_edge_counts
    require(
        retained.shape == edges.shape == (n,)
        and retained[0] == 1
        and edges[0] == 0
        and np.all((retained >= 1) & (retained <= WIDTH))
        and np.array_equal(edges[1:], 6 * retained[:-1])
        and np.array_equal(np.diff(result.layer_offsets), retained)
        and int(retained.sum()) == len(result.state_permutation)
        and int(result.retained_edge_counts.sum()) == len(result.edge_source)
        and int(retained.sum()) <= CONFIG["max_total_states"]
        and int(edges.sum()) <= CONFIG["max_total_edges"]
        and int(result.expanded_state_counts.max()) <= CONFIG["max_layer_states"],
        "registered resource accounting/cap differs",
    )
    resources = dict(
        retained_states=int(retained.sum()),
        generated_edges=int(edges.sum()),
        retained_edges=int(result.retained_edge_counts.sum()),
        expanded_states=int(result.expanded_state_counts.sum()),
        discarded_states=int(result.discarded_state_counts.sum()),
        discarded_finite_states=int(result.discarded_finite_state_counts.sum()),
        largest_expanded_layer=int(result.expanded_state_counts.max()),
        largest_retained_layer=int(retained.max()),
    )
    return ledger, check, resources


def run(comparison_sha, output):
    frozen.cpu_only()
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output",
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    calls, part, before = 0, None, None
    try:
        code = source_code(evidence)
        qualified = qualification(evidence)
        headers = comparison_headers(comparison_sha, evidence)
        part = load_geometry(headers, evidence)
        before = common.signature(part)
        scope = dict(
            schema=STAGE,
            native_interval=[START, STOP],
            native_frames=STOP - START,
            terminal_horizon=STOP,
            source_sha256=code,
            comparison_seal_sha256=comparison_sha,
            comparison_source_sha256=COMPARISON_SHA,
            source_science_export=headers["science"],
            decoded_signatures=before,
            milestone_lineage=headers["lineage"],
            beam_source_sha256=frozen.BEAM_SHA,
            beam_width=WIDTH,
            node_indices=list(NODES),
            dp_config=CONFIG,
            initial_P=[0, 1, 2],
            initialization_frame=START,
            intermediate_ID_resets=0,
            unary="float64 exact zero T,6",
            gaps_conditional_on_retained_graph=True,
            global_optimality_certified=False,
            gaps_are_probabilities=False,
            physical_ID_truth=False,
            biological_naming_applied=False,
            review_flags_cleared=False,
            geometry_or_support_modified=False,
            whole_five_minute_completion_claimed=False,
        )
        out.write(
            "plan.json",
            dict(**scope, existing_minute_qualification=qualified, planned_solve_calls=1),
        )
        tick = time.perf_counter()
        calls += 1
        result = solve_once(part)
        elapsed = time.perf_counter() - tick
        ledger, ledger_check, resources = check_result(part, result)
        require(common.signature(part) == before, "science export changed during solve")
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
                    conditional_global_alternative_status=result.conditional_global_alternative_status,
                    conditional_tied_frames=int(
                        (result.conditional_optimal_state_mask.sum(1) > 1).sum()
                    ),
                    conditional_unknown_ID_cells=int(
                        (
                            result.conditional_identity_alternative_status == "not_retained_unknown"
                        ).sum()
                    ),
                    pruning_occurred=result.pruning_occurred,
                    finite_prefixes_discarded=result.finite_prefixes_discarded,
                    science_archive_decodes=1,
                    decoded_fields=list(FIELDS),
                    original_candidate_checkpoint_SL_video_mask_reads=0,
                    GPU_calls=0,
                    exact_solver_calls=0,
                    automatic_retry=False,
                    cap_increase=False,
                    automatic_adoption=False,
                )
            ),
        )
        frozen.cpu_only()
        evidence.final_guard()
        out.final_guard()
        status = dict(status="ok")
    except BaseException as error:
        status = frozen.failure_record(error, calls)
        if part is not None:
            status["decoded_input_fields_unchanged"] = common.signature(part) == before
        out.write("failure.json", status)
        raise
    finally:
        with (output / "identity.provenance.json").open("xb") as stream:
            stream.write(
                proof.json_bytes(
                    dict(
                        stage=STAGE,
                        started_at=started,
                        finished_at=proof.utc(),
                        inputs=evidence.paths,
                        outputs=out.records,
                        **status,
                    )
                )
            )
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-sha", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    args = parser.parse_args()
    run(args.comparison_sha, args.output)
