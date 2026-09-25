"""Saved-only CT identity unaries on the fixed Original200 geometry.

The frozen DirectMemory solver compares motion-only against motion plus CT for
every causal initial anchor. Full state graphs and alternative paths are saved.
No native RGB, new ray rendering, fitting, model inference, or GPU execution.
The proposed identity order is a separate offline artifact; original candidate21,
sources, support, NaN and sticky review are never edited or cleared.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import time
from pathlib import Path

import body_point_transport as point_transport
import evaluate_body_point200 as evaluated
import identity_memory_dp as memory
import numpy as np
import render_mask_transport200_v2 as candidate_schema
from identity_memory_links import chosen_memory_links
from identity_output import build_identity_output

base, prepared, producer = evaluated.base, evaluated.prepared, evaluated.producer
previous = evaluated.previous
WORK, REPO, FIRST, STOP = evaluated.WORK, evaluated.REPO, evaluated.FIRST, evaluated.STOP
COUNT = 200
STAGE = "body_point_identity200_evaluation_v1"
EVALUATION = WORK / "body_point200_independent_evaluation_v1"
EVALUATION_SHA = "376414c274110ea523146cc2ebbd350edecd93528e1621fa71a8f4c453412ffe"
EVALUATOR_SHA = "faacc694e1d4cfb0c4a33304a8da3883bc16ce32768f52e2f67e2aa71b83c02e"
TRACKING = WORK / "body_point200_tracking_v1"
TRACKING_SHA = "f5cc08f200a6eb7d3029110f876ce449d9875edbe9ea8e42217b00efd08c63df"
PLAN = WORK / "BODY_POINT_ID_WEIGHT_FEASIBILITY.md"
PLAN_SHA = "0725fcc40f6363828cf70f161a81c79f54bc3eb10fa699eaf3502b850399e8ee"
HELPER_SHA = "5e138fdc68a44c49a339d159136bd223c0ab6106468648ab74eeaf85a6064dc4"
EXECUTION_READY = True
require = base.require
DP_CONFIG = dict(
    scale_mm=0.8,
    weight=0.2,
    initial_state=0,
    max_layer_states=100000,
    max_total_states=1000000,
    max_total_edges=6000000,
    tie_atol=1e-10,
    tie_rtol=1e-12,
)
GRAPH_FIELDS = (
    "frame_idx",
    "permutations",
    "layer_offsets",
    "state_permutation",
    "state_last_event",
    "edge_offsets",
    "edge_source",
    "edge_target",
    "edge_cost",
    "edge_delta_frames",
    "edge_squared_distance_mm2",
)
SOURCE_PINS = {
    **producer.SOURCE_PINS,
    **evaluated.SOURCE_PINS,
    "work/three_body/evaluate_body_point200.py": EVALUATOR_SHA,
    "work/three_body/run_body_point200.py": evaluated.PRODUCER_SHA,
    "work/three_body/render_mask_transport200_v2.py": (
        "5903b976f6f809d77d684eceeefb8064a76f0e3e92d52a73998d5cc1f221822d"
    ),
    "work/three_body/render_semantic_node200.py": (
        "8773ef01eae48fb8e4cdc9110c7e7f62c487ec1f8cde1b66700625a5618ae856"
    ),
    "labs/20260909_three_body/identity_dp.py": (
        "2e848b6c4b14eebcd3d4d3791a4f61aae3e8f4e9fd0413588bd6a1548121e354"
    ),
    "labs/20260909_three_body/identity_memory_dp.py": (
        "2ef045165dd9f0342d234064335f0f8dd015ada5a864696cfab8fb5449da1adf"
    ),
    "labs/20260909_three_body/identity_memory_links.py": (
        "672dcd2d3c0175863179fa9b21bffec4336ada76ffc59c0e67130566f964140b"
    ),
    "labs/20260909_three_body/identity_output.py": (
        "3e462fe7e3d7c9a3128f751c7e32c9530dbb3cad7ff9f9eb913a40731aaaa136"
    ),
}


def ready():
    require(
        EXECUTION_READY and base.valid_sha(HELPER_SHA),
        "reviewed final identity-cost helper required before execution",
    )


def signatures(values):
    return {name: base.signature(value) for name, value in values.items()}


def validate_evaluation(plan, summary, seal):
    require(
        plan["schema"] == summary["schema"] == evaluated.STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["arms"] == summary["arms"] == list(evaluated.ARMS)
        and plan["evidence"] == summary["evidence"] == list(evaluated.EVIDENCE)
        and plan["source_tracking_sha256"] == TRACKING_SHA
        and plan["packet_preparation_sha256"] == producer.PACKETS_SHA
        and plan["renderer"] == "original_dynamic_all7"
        and plan["point_mass_domain"] == "saved FIT1024 m/S; IN-SAMPLE"
        and summary["previous_six_common19_exact_frames"] == 200
        and summary["strict_candidate_support_rechecks"] == 1400
        and summary["tracking_CT_packet_exact_frames"] == 199
        and summary["point_evaluation_packets"] == 400
        and summary["point_metrics_are_IN_SAMPLE"] is True
        and summary["t0_packet_used_in_fit"] is False
        and summary["trajectory_or_review_modified"] is False
        and summary["GPU_initialized"] is False,
        "completed same-geometry seven-arm point evidence required",
    )
    for path, digest in (
        (WORK / "evaluate_body_point200.py", EVALUATOR_SHA),
        (TRACKING / "tracking.provenance.json", TRACKING_SHA),
        (producer.PACKETS / "preparation.provenance.json", producer.PACKETS_SHA),
        (producer.PREVIOUS / "tracking.provenance.json", producer.PREVIOUS_SHA),
    ):
        evaluated.input_member(seal, path, digest, "saved identity evidence")


def load_inputs(evidence):
    seal, members = base.verified(
        EVALUATION, "evaluation.provenance.json", evaluated.STAGE, EVALUATION_SHA, evidence
    )
    plan, summary = (
        base.bound_json(EVALUATION, members, name) for name in ("plan.json", "summary.json")
    )
    validate_evaluation(plan, summary, seal)
    for relative, digest in {
        **producer.SOURCE_PINS,
        **evaluated.SOURCE_PINS,
        "work/three_body/run_body_point200.py": evaluated.PRODUCER_SHA,
    }.items():
        evaluated.input_member(seal, REPO / relative, digest, "saved evaluator source")
    packet_seal, packet_members = base.verified(
        producer.PACKETS,
        "preparation.provenance.json",
        evaluated.PACKET_STAGE,
        producer.PACKETS_SHA,
        evidence,
    )
    packet_plan, packet_summary = (
        base.bound_json(producer.PACKETS, packet_members, name)
        for name in ("plan.json", "summary.json")
    )
    evaluated.validate_packet_preparation(packet_plan, packet_summary, packet_seal)
    require(
        packet_plan["all_initial_anchor_ids"]
        == plan["all_initial_anchor_ids"]
        == packet_summary["all_initial_anchor_ids"]
        == [0, 1],
        "both registered causal anchors required, never select one by its score",
    )
    original_seal, original_members = base.verified(
        producer.PREVIOUS,
        "tracking.provenance.json",
        previous.STAGE,
        producer.PREVIOUS_SHA,
        evidence,
    )
    original_plan = base.bound_json(producer.PREVIOUS, original_members, "plan.json")
    require(
        original_plan["original_context"] == packet_plan["original_context"],
        "packet initial labels and original geometry context differ",
    )
    path = prepared.bound_path(producer.PREVIOUS, original_members, "original/candidate.npz")
    candidate_digest = original_members[str(path)]["sha256"]
    references = plan["candidate_references"]
    require(
        [r["arm"] for r in references] == list(evaluated.ARMS)
        and references[0] == dict(arm="original", path=str(path), sha256=candidate_digest),
        "Original column must name the exact original candidate21",
    )
    evaluated.input_member(seal, path, candidate_digest, "Original candidate")
    original = previous.read_arrays(producer.PREVIOUS, original_members, "original/candidate.npz")
    candidate_schema.validate_candidate(original)
    require(original["observed"][0, :, 1].all(), "frozen DP requires three direct initial chests")
    return dict(
        evaluation_members=members,
        packet_members=packet_members,
        packet_plan=packet_plan,
        original=original,
        original_reference=dict(path=str(path), sha256=candidate_digest),
        original_input_signatures=signatures(original),
        original_producer_inputs=original_seal["inputs"],
    )


def selected_frame(values, packet, native, packet_plan):
    """Select CT/Original by the fixed saved axes; no new geometry or ray operation."""
    require(
        packet.native == native
        and packet.initial_native == FIRST
        and tuple(packet.query_sha256) == tuple(packet_plan["query_sha256"]),
        "complete prepared CT query/native binding differs",
    )
    previous.same(packet.anchor_ids, np.asarray([0, 1], np.int64), "all causal initial anchors")
    previous.same(
        packet.release_horizons,
        np.full(3, previous.horizon(native), np.int64),
        "saved first-release horizons",
    )
    previous.same(values["point_frame_idx"], np.asarray(native, np.int64), "point score native")
    previous.same(values["point_anchor_ids"], packet.anchor_ids, "evaluated all initial anchors")
    previous.same(values["point_query_counts"], packet.query_counts, "evaluated all fixed queries")
    q = packet.point_pixels.shape[1]
    owners, distances = values["point_rendered_ownership"], values["point_rendered_q"]
    valid = values["point_ray_valid"]
    require(
        owners.dtype == distances.dtype == np.float64
        and owners.shape == distances.shape == (2, 7, 3, q, 3)
        and valid.dtype == bool
        and valid.shape == (2, 3, q),
        "complete evidence/geometry/query/raw-body axes required",
    )
    previous.same(valid[1], packet.ray_valid, "CT saved calibrated query rays")
    # Stored per-identity masses are independently tied to the canonical packet.
    mass = (packet.point_weight * packet.ray_valid[None, :, :, None]).sum(axis=2)
    previous.same(values["point_mass_by_view_body"][1], mass, "CT fit1024 mass unchanged")
    require(
        values["point_combined_loss"].dtype == np.float64
        and values["point_combined_loss"].shape == (2, 7, 2),
        "all evaluated point losses required for the P012 diagonal control",
    )
    return dict(
        ownership=owners[1, 0].copy(),
        ray_distance_sq=distances[1, 0].copy(),
        point_weight=packet.point_weight.copy(),
        ray_valid=valid[1].copy(),
        release_horizons=packet.release_horizons.copy(),
        old_mass=packet.old_mass.copy(),
        single_count=packet.single_count.copy(),
        valid_single_count=packet.valid_single_count.copy(),
        saved_point_combined_loss=values["point_combined_loss"][1, 0].copy(),
    )


def collect_inputs(loaded, output):
    rows, bank = [], None
    for native in range(FIRST, STOP):
        raw = evaluated.old_evaluation.load_tree(
            producer.PACKETS, loaded["packet_members"], f"packets/ct_first/native_{native}"
        )
        # Full algebra/membership/lineage canonical check. Calibrated rays were
        # byte-checked in the sealed preparation AND evaluation; no rig is rebuilt.
        packet = point_transport.validate_packet(raw)
        if bank is None:
            bank = packet.whole_mask_packet.anchors
        else:
            point_transport._same(
                packet.whole_mask_packet.anchors, bank, "fixed whole initial bank"
            )
        values = previous.read_arrays(
            EVALUATION, loaded["evaluation_members"], f"point_frames/native_{native}.npz"
        )
        rows.append(selected_frame(values, packet, native, loaded["packet_plan"]))
    require(len(rows) == COUNT, "all200 point evidence frames required")
    result = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    output.write(
        "point_support.npz",
        {
            k: result[k]
            for k in ("release_horizons", "old_mass", "single_count", "valid_single_count")
        },
    )
    return result


def solve(geometry, unary):
    result = memory.solve_identity_memory_dp(
        geometry["frame_idx"], unary, geometry["xyz"], geometry["observed"][:, :, 1], **DP_CONFIG
    )
    require(
        result.status == "optimal" and not result.approximation_used, "full exact DP incomplete"
    )
    identity = build_identity_output(geometry, result.permutation_by_id)
    links = chosen_memory_links(
        geometry,
        result.permutation_by_id,
        scale_mm=DP_CONFIG["scale_mm"],
        weight=DP_CONFIG["weight"],
    )
    selected_unary = unary[np.arange(len(unary)), result.path_state]
    chosen_cost = float(selected_unary.sum() + links["brownian_motion_cost_by_id"].sum())
    require(
        np.isclose(result.path_cost, chosen_cost, rtol=1e-11, atol=1e-10),
        "DP objective differs from independently reconstructed direct-event links",
    )
    # Raw21 stays in its sealed original artifact; proposal arrays retain source
    # provenance and original review without duplicating all raw geometry arrays.
    identity = {k: value for k, value in identity.items() if not k.startswith("raw_")}
    return (
        result,
        identity,
        links,
        dict(
            motion_cost=float(links["brownian_motion_cost_by_id"].sum()),
            unary_cost=float(selected_unary.sum()),
            recomputed_path_cost=chosen_cost,
            solver_path_cost=result.path_cost,
            independently_recomputed=True,
        ),
    )


def same_motion_graph(reference, current):
    for key in GRAPH_FIELDS:
        previous.same(
            getattr(current, key), getattr(reference, key), "unchanged motion graph " + key
        )
    return dict(
        byte_exact=True,
        fields=list(GRAPH_FIELDS),
        states=len(reference.state_permutation),
        edges=len(reference.edge_source),
        maximum_layer_states=int(np.diff(reference.layer_offsets).max()),
    )


def diagonal_control(costs, saved):
    """P012 uses the old identity/raw diagonal, including measured t0 before its zero unary."""
    actual = costs["measured_unary"][:, :, 0] / 3.0
    require(actual.shape == saved.shape, "saved point/identity diagonal shape differs")
    require(
        np.isfinite(actual).all()
        and np.isfinite(saved).all()
        and np.allclose(actual, saved, atol=1e-12, rtol=1e-12),
        "P012 identity cost differs from the sealed CT/Original point loss",
    )
    return dict(
        measured_P012_divided_by_coefficient=actual.copy(),
        saved_point_combined_loss=saved.copy(),
        difference=actual - saved,
    )


def save_solution(output, prefix, bundle, geometry, baseline, helper):
    result, identity, links, objective = bundle
    previous.save_tree(
        output,
        prefix + "/dp",
        {f.name: getattr(result, f.name) for f in dataclasses.fields(result)},
    )
    output.write(prefix + "/identity_proposal.npz", identity)
    output.write(prefix + "/memory_links.npz", links)
    output.write(prefix + "/objective.json", objective)
    fixed = np.tile(np.arange(3, dtype=np.int64), (len(geometry["frame_idx"]), 1))
    comparisons = {}
    for label, mapping in (("original_fixed", fixed), ("motion_only", baseline.permutation_by_id)):
        values = helper.compare_mapping(
            geometry["frame_idx"], mapping, result.permutation_by_id, geometry["observed"][:, :, 1]
        )
        output.write(prefix + "/vs_" + label + ".npz", values)
        comparisons[label] = dict(
            support_definition=(
                "direct-involved: old or new raw chest directly observed; latent-only: "
                "both raw chests unsupported, even if another node is direct"
            ),
            changed_identity_times=int(values["changed_by_identity"].sum()),
            direct_involved_identity_times=int(values["direct_involved_change"].sum()),
            latent_only_identity_times=int(values["latent_only_change"].sum()),
            changed_native=values["changed_native"].tolist(),
            direct_involved_native=values["direct_involved_native"].tolist(),
            latent_only_native=values["latent_only_native"].tolist(),
        )
    return dict(
        objective=objective,
        comparisons=comparisons,
        permutation_frame_counts=np.bincount(result.path_state, minlength=6).tolist(),
        permutation_transition_count=int(np.count_nonzero(np.diff(result.path_state))),
        states=len(result.state_permutation),
        edges=len(result.edge_source),
        terminal_unsettled_gap=result.terminal_unsettled_gap.tolist(),
        terminal_age_frames=result.terminal_age_frames.tolist(),
        global_alternative_gap=result.global_alternative_gap,
        global_alternative_status=result.global_alternative_status,
        numerical_tie_tolerance=result.numerical_tie_tolerance,
    )


def run(destination):
    ready()
    destination = Path(destination).resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "new workspace output")
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, completed = base.utc(), time.perf_counter(), []
    try:
        require(not producer.torch.cuda.is_initialized(), "CPU saved-only stage requires no CUDA")
        evidence.add(Path(__file__).resolve())
        evidence.add(PLAN, PLAN_SHA)
        for path, digest in SOURCE_PINS.items():
            evidence.add(REPO / path, digest)
        evidence.add(WORK / "body_point_identity.py", HELPER_SHA)
        helper = importlib.import_module("body_point_identity")
        loaded = load_inputs(evidence)
        inputs = collect_inputs(loaded, output)
        before = signatures(inputs)
        costs = helper.identity_costs(
            inputs["ownership"],
            inputs["ray_distance_sq"],
            inputs["point_weight"],
            inputs["ray_valid"],
            coefficient=3.0,
        )
        require(costs["unary"].shape == (COUNT, 2, 6), "all200/two-anchor/six-state costs required")
        require(not costs["unary"][0].any(), "t0 must use only its initial permutation constraint")
        output.write("identity_costs.npz", costs)
        diagonal = diagonal_control(costs, inputs["saved_point_combined_loss"])
        output.write("original_diagonal_control.npz", diagonal)
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[FIRST, STOP],
                evidence="ct_first",
                geometry="original",
                evaluation_sha256=EVALUATION_SHA,
                packet_preparation_sha256=producer.PACKETS_SHA,
                original_tracking_sha256=producer.PREVIOUS_SHA,
                original_candidate=loaded["original_reference"],
                original_candidate21_signatures=loaded["original_input_signatures"],
                plan_sha256=PLAN_SHA,
                helper_sha256=HELPER_SHA,
                source_pins=SOURCE_PINS,
                anchors=[0, 1],
                coefficient=3.0,
                dp_config=DP_CONFIG,
                unary_shape=[COUNT, 2, 6],
                first_unary_zero=True,
                initial_identity_permutation=[0, 1, 2],
                runs=["motion_only", "ct_anchor_0", "ct_anchor_1"],
                point_mass_domain="saved FIT1024 m/S; IN-SAMPLE; no invalid renormalization",
                geometry_rendering=False,
                new_fit=False,
                model_inference=False,
                raw_RGB_decode=False,
                original_state_restore=False,
                online=False,
                independent_ID_truth=False,
                proposal_output_is_separate=True,
                review_clearing=False,
                release_horizons="point_support.npz; first release per native/view",
            ),
        )
        geometry = loaded["original"]
        baseline = solve(geometry, np.zeros((COUNT, 6), np.float64))
        results = dict(
            motion_only=save_solution(
                output, "motion_only", baseline, geometry, baseline[0], helper
            )
        )
        completed.append("motion_only")
        controls = {}
        for anchor in (0, 1):
            label = f"ct_anchor_{anchor}"
            current = solve(geometry, costs["unary"][:, anchor])
            controls[label] = same_motion_graph(baseline[0], current[0])
            output.write(label + "/motion_graph_control.json", controls[label])
            results[label] = save_solution(output, label, current, geometry, baseline[0], helper)
            completed.append(label)
        require(signatures(inputs) == before, "saved point input arrays changed")
        require(signatures(geometry) == loaded["original_input_signatures"], "Original21 changed")
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            frames=COUNT,
            runs=results,
            all_anchor_ids=[0, 1],
            independent_anchor_replications=False,
            CT_packets_consumed=COUNT,
            prepared_two_method_packet_count=400,
            saved_point_frame_arrays=COUNT,
            all_six_permutations=True,
            motion_graph_controls=controls,
            P012_diagonal_control=dict(
                compared_cells=int(diagonal["difference"].size),
                measured_t0_included=True,
                atol=1e-12,
                rtol=1e-12,
                max_absolute_difference=float(np.abs(diagonal["difference"]).max()),
                passed=True,
                byte_exact_required=False,
            ),
            original_candidate21_unchanged=True,
            point_arrays_unchanged=True,
            original_support_and_review_unchanged=True,
            evidence_temporal_independence=False,
            point_metrics_IN_SAMPLE=True,
            online=False,
            independent_ID_truth=False,
            mapped_support_count_is_not_accuracy=True,
            model_inference=False,
            fit=False,
            GPU_initialized=False,
            native_RGB_decodes=0,
            elapsed_before_guards_seconds=time.perf_counter() - start,
        )
        output.write("summary.json", result)
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        require(not producer.torch.cuda.is_initialized(), "saved-only CPU stage initialized CUDA")
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
        )
    except BaseException as error:
        status = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_runs=completed,
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "evaluation.provenance.json").open("xb") as f:
            f.write(base.json_bytes(status))
        raise
    with (destination / "evaluation.provenance.json").open("xb") as f:
        f.write(base.json_bytes(status))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output)), flush=True)


if __name__ == "__main__":
    main()
