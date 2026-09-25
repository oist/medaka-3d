"""Qualify widths 64/256 on one frozen, previously solved Tail chest-mouth minute.

CPU only, two new approximate calls on immutable saved geometry. The full exact
solver is never rerun and its edge/DAG arrays are not decoded. Conditional gaps
are compared as engineering diagnostics, never global certificates. This cannot
repair mixed-fish body geometry and authorizes no two-minute or five-minute use.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import identity_node_memory_beam as beam
import numpy as np
import qualify_streaming_tail_minute as qualified

proof, require = qualified.proof, qualified.require
WORK, REPO, START, STOP = qualified.WORK, qualified.REPO, qualified.START, qualified.STOP
STAGE = "identity_beam_exact_minute_qualification_v1"
BEAM_SHA = "019791d34fda3292aa35371ba3aa8297c3f92ace1e19e8ffe514cd39fd4c92bd"
QUALIFIED_SHA = "f9b33819abf260f11fd8276939ee47a6ea012c768ebbcf94b8d8978896dfb222"
WIDTHS, NODES = (64, 256), (1, 0)
TERMINAL = (
    "terminal_last_event_frame",
    "terminal_last_event_slot",
    "terminal_age_frames",
    "terminal_unsettled_gap",
)
SELECTED = (
    "frame_idx",
    "path_state",
    "permutation_by_id",
    "min_marginal_cost",
    "min_marginal_delta",
    "optimal_state_mask",
    "id_slot_min_marginal_delta",
    "state_alternative_gap",
    "identity_alternative_gap",
    "state_alternative_status",
    "identity_alternative_status",
    *TERMINAL,
)


def safe(value):
    """Keep nonfinite engineering gaps explicit in strict JSON, never silently null."""
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    return value


def objective_tolerance(cost):
    """Fixed-before-execution conservative float64 accumulation allowance.

    With zero unary, every motion event is nonnegative. Bound sequential
    accumulation/centering by 64 roundoff operations per native frame, scaled
    to the compared nonnegative path objective. This is not an accuracy target.
    """
    require(np.isfinite(cost) and cost >= 0, "finite nonnegative fixed motion cost required")
    return max(1e-10, 64 * np.finfo(np.float64).eps * (STOP - START) * (1 + abs(float(cost))))


def load_control(seal, evidence):
    geometry, _ = proof.read_candidate(
        proof.named_read(seal, qualified.CONTROL, "raw_candidate21.npz", evidence),
        support_only=False,
        first=START,
        start=START,
        stop=STOP,
    )
    signatures = json.loads(
        proof.named_read(seal, qualified.CONTROL, "raw_array_signatures.json", evidence)
    )
    require(
        {k: proof.signature(v) for k, v in geometry.items()} == signatures,
        "raw21 signature differs",
    )
    raw = proof.named_read(seal, qualified.CONTROL, "MP/dp_arrays.npz", evidence)
    array_names = {
        f.name
        for f in dataclasses.fields(beam.frozen.IdentityNodeMemoryDPResult)
        if f.type is np.ndarray
    }
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(set(archive.files)) == len(array_names) == 29
            and set(archive.files) == array_names,
            "full-control archive membership differs",
        )
        control = {k: archive[k].copy() for k in SELECTED}
    frames, mapping = control["frame_idx"], control["permutation_by_id"]
    require(
        np.array_equal(frames, geometry["frame_idx"])
        and mapping.dtype.kind in "iu"
        and mapping.shape == (STOP - START, 3)
        and np.array_equal(np.sort(mapping, axis=1), np.tile([0, 1, 2], (STOP - START, 1)))
        and np.array_equal(mapping[0], [0, 1, 2]),
        "control frame/mapping contract differs",
    )
    state = control["path_state"]
    require(
        state.shape == frames.shape
        and state.dtype.kind in "iu"
        and np.all((state >= 0) & (state < 6))
        and np.array_equal(beam.frozen.PERMUTATIONS[state], mapping),
        "control P-state differs",
    )
    for values in (geometry, control):
        for value in values.values():
            value.flags.writeable = False
    return geometry, signatures, control


def validate_scalars(scalars):
    require(
        scalars["status"] == "optimal"
        and scalars["approximation_used"] is False
        and scalars["initial_state"] == 0
        and scalars["node_indices"] == [1, 0]
        and scalars["scale_mm"] == 0.8
        and scalars["weight"] == 0.2
        and scalars["max_layer_states"] == 100000
        and scalars["max_total_states"] == 1000000
        and scalars["max_total_edges"] == 6000000
        and scalars["terminal_missing_endpoint_cost_applied"] is False,
        "same fixed exact chest-mouth objective required",
    )


def ledger_for(geometry, mapping, expected_cost, expected_terminal):
    ledger = beam.frozen.node_memory_path_links(
        geometry["frame_idx"],
        geometry["pose_xyz"][:, :, NODES, :],
        geometry["observed"][:, :, NODES],
        mapping,
        node_indices=NODES,
        scale_mm=0.8,
        weight=0.2,
    )
    tolerance = objective_tolerance(max(expected_cost, ledger["motion_cost"]))
    difference = float(ledger["motion_cost"] - expected_cost)
    require(abs(difference) <= tolerance, "independent chronological path cost disagrees")
    exact_terminal = {k: np.array_equal(ledger[k], expected_terminal[k]) for k in TERMINAL}
    require(all(exact_terminal.values()), "independent terminal memory differs")
    report = dict(
        motion_cost=ledger["motion_cost"],
        reported_path_cost=expected_cost,
        signed_difference=difference,
        tolerance=tolerance,
        terminal_arrays_exact=exact_terminal,
        cost_by_node=ledger["cost_by_event"].sum((0, 1)).tolist(),
    )
    return ledger, report


def compare_gap(before, after, tolerance):
    require(before.shape == after.shape, "same native conditional-gap axes required")
    both = np.isfinite(before) & np.isfinite(after)
    difference = np.full(before.shape, np.nan)
    difference[both] = after[both] - before[both]
    return difference, dict(
        finite_both=int(both.sum()),
        exact_finite_but_no_retained_alternative=int(
            (np.isfinite(before) & np.isposinf(after)).sum()
        ),
        exact_unavailable_but_retained_finite=int((np.isposinf(before) & np.isfinite(after)).sum()),
        neither_finite=int((~np.isfinite(before) & ~np.isfinite(after)).sum()),
        finite_difference_exceeds_objective_tolerance=int(
            (np.abs(difference[both]) > tolerance).sum()
        ),
        maximum_absolute_finite_difference=float(np.abs(difference[both]).max())
        if both.any()
        else None,
        interpretation=(
            "Conditional on retained graph; missing alternative is not globally impossible"
        ),
    )


def compare_result(result, geometry, control, scalars, control_ledger, tolerance):
    require(
        type(result) is beam.IdentityNodeMemoryBeamResult
        and result.schema == beam.SCHEMA
        and result.status == "optimal_on_retained_graph"
        and result.global_optimality_certified is False
        and result.gaps_conditional_on_retained_graph is True
        and result.physical_ID_truth is False
        and result.review_flags_cleared is False
        and result.intermediate_ID_reset is False
        and np.array_equal(result.frame_idx, geometry["frame_idx"]),
        "distinct conditional result required",
    )
    ledger, ledger_report = ledger_for(
        geometry,
        result.permutation_by_id,
        result.path_cost,
        {k: getattr(result, k) for k in TERMINAL},
    )
    delta_cost = float(result.path_cost - scalars["path_cost"])
    require(
        delta_cost >= -tolerance,
        "approximate search beats exact objective beyond numerical tolerance",
    )
    changed = result.permutation_by_id != control["permutation_by_id"]
    terminal = {k: np.array_equal(getattr(result, k), control[k]) for k in TERMINAL}
    arrays = dict(
        frame_idx=result.frame_idx,
        exact_mapping=control["permutation_by_id"],
        beam_mapping=result.permutation_by_id,
        mapping_changed_by_ID=changed,
        per_frame_ledger_cost_difference=(
            ledger["cost_by_event"] - control_ledger["cost_by_event"]
        ).sum((1, 2)),
        conditional_optimal_mask_changed=result.conditional_optimal_state_mask
        != control["optimal_state_mask"],
    )
    gaps = {}
    for name in (
        "min_marginal_delta",
        "id_slot_min_marginal_delta",
        "state_alternative_gap",
        "identity_alternative_gap",
    ):
        arrays[name + "_finite_difference"], gaps[name] = compare_gap(
            control[name], getattr(result, "conditional_" + name), tolerance
        )
    report = dict(
        beam_width=result.beam_width,
        changed_native_frames=int(changed.any(1).sum()),
        changed_ID_frames=int(changed.sum()),
        changed_intervals=proof.intervals(result.frame_idx, changed),
        every_native_mapping_equal=not bool(changed.any()),
        exact_path_cost=scalars["path_cost"],
        beam_path_cost=result.path_cost,
        signed_objective_difference=delta_cost,
        objective_tolerance=tolerance,
        objective_matches_control=abs(delta_cost) <= tolerance,
        independent_path_ledger=ledger_report,
        terminal_matches_control=terminal,
        pruning_occurred=result.pruning_occurred,
        finite_prefixes_discarded=result.finite_prefixes_discarded,
        expanded_states=int(result.expanded_state_counts.sum()),
        retained_states=int(result.retained_state_counts.sum()),
        discarded_states=int(result.discarded_state_counts.sum()),
        discarded_finite_states=int(result.discarded_finite_state_counts.sum()),
        expanded_edges=int(result.expanded_edge_counts.sum()),
        retained_edges=int(result.retained_edge_counts.sum()),
        discarded_edges=int(result.discarded_edge_counts.sum()),
        largest_expanded_layer=int(result.expanded_state_counts.max()),
        largest_retained_layer=int(result.retained_state_counts.max()),
        conditional_gap_comparison=gaps,
        exact_global_alternative_gap=scalars["global_alternative_gap"],
        conditional_global_alternative_gap=result.conditional_global_alternative_gap,
        conditional_global_alternative_status=result.conditional_global_alternative_status,
        exact_numerical_tie_tolerance=scalars["numerical_tie_tolerance"],
        conditional_numerical_tie_tolerance=result.numerical_tie_tolerance,
        exact_multiple_optimal_frames=int((control["optimal_state_mask"].sum(1) > 1).sum()),
        conditional_multiple_optimal_frames=int(
            (result.conditional_optimal_state_mask.sum(1) > 1).sum()
        ),
        unknown_state_alternative_frames=int(
            (result.conditional_state_alternative_status == "not_retained_unknown").sum()
        ),
        unknown_ID_alternative_cells=int(
            (result.conditional_identity_alternative_status == "not_retained_unknown").sum()
        ),
        same_minute_path_and_terminal_preserved=not bool(changed.any())
        and abs(delta_cost) <= tolerance
        and all(terminal.values()),
        global_optimality_certified=False,
        all_three_ID_truth=False,
        review_cleared=False,
        whole_graph_BYTEEXACT_required=False,
        two_minute_experiment_authorized=False,
    )
    return arrays, ledger, report


def save_result(output, prefix, result):
    arrays, scalars = {}, {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            require(not value.flags.writeable, "beam output arrays must remain read-only")
            arrays[field.name] = value
        else:
            scalars[field.name] = value
    output.save(prefix + "/retained_graph_and_diagnostics.npz", arrays, arrays=True)
    output.save(prefix + "/beam_scalars.json", safe(scalars))


def run(destination):
    require("torch" not in sys.modules, "standalone CPU-only qualification required")
    evidence, output = proof.Evidence(), proof.Output(destination)
    began, reports = datetime.now(UTC).isoformat(), {}
    geometry = signatures = None
    try:
        evidence.read(Path(qualified.__file__), QUALIFIED_SHA)
        evidence.read(Path(beam.__file__), BEAM_SHA)
        seal, old_plan = qualified.load_inputs(evidence)
        scalars = json.loads(
            proof.named_read(seal, qualified.CONTROL, "MP/dp_scalars.json", evidence)
        )
        validate_scalars(scalars)
        tolerance = objective_tolerance(scalars["path_cost"])
        output.save(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[START, STOP],
                widths=list(WIDTHS),
                node_indices=list(NODES),
                control_seal_sha256=qualified.CONTROL_SHA,
                raw_control_reference=proof.output_member(
                    seal, qualified.CONTROL, "raw_candidate21.npz"
                ),
                exact_mapping_reference=proof.output_member(
                    seal, qualified.CONTROL, "MP/dp_arrays.npz"
                ),
                selected_exact_array_members=list(SELECTED),
                beam_source_sha256=BEAM_SHA,
                dp_config=old_plan["dp_config"],
                initial_P=[0, 1, 2],
                unary="float64 exact zero T,6",
                calls_per_width=1,
                cap_numbers_unchanged=True,
                capacity_scopes=dict(
                    total_states="stored retained nodes",
                    total_edges="all generated including discarded",
                    layer_states="merged expanded before pruning",
                ),
                tolerance_rule=(
                    "max(1e-10,64*eps64*9960*(1+abs(cost))); "
                    "nonnegative zero-unary motion objective"
                ),
                reference_path_cost=scalars["path_cost"],
                reference_objective_tolerance=tolerance,
                ledger_tolerance_uses_same_rule_on_maximum_compared_cost=True,
                original_exact_solver_rerun=False,
                full_exact_DAG_decoded=False,
                GPU_used=False,
                source_geometry_or_review_modified=False,
                automatic_adoption=False,
                physical_ID_truth=False,
                two_minute_qualification=False,
                environment=dict(python=platform.python_version(), numpy=np.__version__),
            ),
        )
        for path, digest in tuple(evidence.inputs.items()):
            source = Path(path)
            if source.suffix == ".py":
                relative = "code_snapshot/" + source.relative_to(REPO).as_posix()
                target = output.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(evidence.read(source, digest))
                output.outputs[relative] = dict(
                    path=str(target), size_bytes=target.stat().st_size, sha256=digest
                )
        geometry, signatures, control = load_control(seal, evidence)
        output.save("raw21_signatures.json", signatures)
        control_signatures = {k: proof.signature(v) for k, v in control.items()}
        output.save("exact_selected_arrays.npz", control, arrays=True)
        control_ledger, control_report = ledger_for(
            geometry, control["permutation_by_id"], scalars["path_cost"], control
        )
        output.save("exact_path_ledger.json", safe(control_report))
        output.save(
            "exact_path_ledger_arrays.npz",
            {k: v for k, v in control_ledger.items() if isinstance(v, np.ndarray)},
            arrays=True,
        )
        for width in WIDTHS:
            print(json.dumps(dict(event="beam_minute_started", width=width)), flush=True)
            began_width = time.perf_counter()
            result = beam.solve_identity_node_memory_beam(
                geometry["frame_idx"],
                np.zeros((STOP - START, 6), np.float64),
                geometry["pose_xyz"][:, :, NODES, :],
                geometry["observed"][:, :, NODES],
                beam_width=width,
                node_indices=NODES,
                **proof.DP_CONFIG,
            )
            seconds = time.perf_counter() - began_width
            prefix = f"width_{width}"
            save_result(output, prefix, result)
            arrays, ledger, report = compare_result(
                result, geometry, control, scalars, control_ledger, tolerance
            )
            report["solve_seconds"] = seconds
            output.save(prefix + "/per_native_comparison.npz", arrays, arrays=True)
            output.save(
                prefix + "/path_ledger.npz",
                {k: v for k, v in ledger.items() if isinstance(v, np.ndarray)},
                arrays=True,
            )
            output.save(prefix + "/comparison.json", safe(report))
            reports[str(width)] = report
            require(
                {k: proof.signature(v) for k, v in geometry.items()} == signatures
                and {k: proof.signature(v) for k, v in control.items()} == control_signatures,
                "raw21 or saved exact control changed",
            )
            print(
                json.dumps(
                    dict(
                        event="beam_minute_compared",
                        width=width,
                        mapping_changes=report["changed_native_frames"],
                        objective_difference=report["signed_objective_difference"],
                        seconds=seconds,
                    )
                ),
                flush=True,
            )
        require("torch" not in sys.modules, "CPU qualification imported torch")
        output.save(
            "summary.json",
            safe(
                dict(
                    schema=STAGE,
                    status="ok",
                    native_interval=[START, STOP],
                    native_frames=STOP - START,
                    widths=reports,
                    calls=2,
                    raw21_unchanged=True,
                    raw_sources_review_unchanged=True,
                    all_widths_same_minute_path_preserved=all(
                        r["same_minute_path_and_terminal_preserved"] for r in reports.values()
                    ),
                    conditional_gaps_are_not_global_certificates=True,
                    physical_ID_truth=False,
                    body_instance_mix_repaired=False,
                    two_minute_qualified=False,
                    five_minute_qualified=False,
                    automatic_adoption=False,
                    raw_frames_decoded=0,
                    GPU_used=False,
                )
            ),
        )
        evidence.guard()
        output.guard()
        final = dict(
            stage=STAGE,
            status="ok",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
        )
    except BaseException as error:
        raw_unchanged = (
            geometry is not None
            and {k: proof.signature(v) for k, v in geometry.items()} == signatures
        )
        output.save(
            "failure.json",
            dict(
                error=repr(error),
                raw21_unchanged=raw_unchanged,
                completed_widths=list(reports),
                cap_raise_or_retry_performed=False,
                partial_results_preserved=True,
            ),
        )
        guard_error = None
        try:
            evidence.guard()
            output.guard()
        except BaseException as guard:
            guard_error = repr(guard)
        final = dict(
            stage=STAGE,
            status="failed",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
            error=repr(error),
            final_guard_error=guard_error,
        )
        with (output.root / "qualification.provenance.json").open("x", encoding="utf-8") as stream:
            json.dump(final, stream, indent=2, allow_nan=False)
        raise
    with (output.root / "qualification.provenance.json").open("x", encoding="utf-8") as stream:
        json.dump(final, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status="ok", output=str(output.root), widths=list(reports))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "explicit execution required after caller review")
    run(args.output)
