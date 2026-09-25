"""Tail-only chest+mouth DirectMemory against the same Tail chest-only solution.

Original MP exceeds the registered layer cap and is not solved. A single common
permutation applies to both nodes. Raw21, SL source rows, geometry and review are
immutable; the new identity mapping remains a separate offline research result.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import run_tail_minute_mouth_memory_preflight as preflight

previous = preflight.previous
WORK, REPO, START, STOP = preflight.WORK, preflight.REPO, preflight.START, preflight.STOP
STAGE = "tail_minute_mouth_memory_motion_v1"
PREFLIGHT = WORK / "tail_minute_mouth_memory_reachability_v1"
PREFLIGHT_SHA = "64570fc7d6d539e367dbd67fb6b01798e5bb0c7b0497ff17c7cbbefacf3f0566"
PREFLIGHT_SOURCE_SHA = "324219935cc741f63dfe01f7bff8ba787031deb7670790e26e44bec5c2319178"
HELPER_SHA = "e5331f706137cc361154cc065caaa58f02686a53019c1971795adb5702e94dd6"
TIE_NATIVES = (76237, 76238)
NODES = (1, 0)
require = previous.require


def read_arrays(seal, root, role, evidence):
    raw = previous.named_read(seal, root, role, evidence)
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), "duplicate array key")
        return {k: saved[k].copy() for k in saved.files}


def validate_preflight(plan, summary, expected):
    require(
        plan == expected
        and summary["schema"] == preflight.STAGE
        and summary["status"] == "ok"
        and summary["native_interval"] == [START, STOP]
        and summary["node_indices"] == [1, 0]
        and summary["native_frames"] == STOP - START,
        "same fixed MP preflight required",
    )
    require(
        summary["methods"]["tail"]["completed"] is True
        and summary["methods"]["tail"]["counted_native_frames"] == STOP - START
        and summary["methods"]["tail"]["stop"] is None,
        "Tail all-native MP capacity qualification required",
    )
    original = summary["methods"]["original"]
    require(
        original["completed"] is False
        and original["stop"]["reason"] == "maximum_layer_states"
        and original["stop"]["attempted_frame"] == 70691
        and original["stop"]["attempted_layer_states_lower_bound"] == 100001
        and summary["all_within_fixed_caps"] is False,
        "Original failed-cap evidence must remain explicit; no new Original DP",
    )
    require(
        summary["graph_scored"] is False
        and summary["positions_read"] is False
        and summary["cap_or_weight_changed"] is False,
        "support-only unchanged-cap qualification required",
    )


def load_inputs(evidence):
    """Source/small JSON gate only; no raw21 or previous DP numerical payload."""
    evidence.read(WORK / "run_tail_minute_mouth_memory_preflight.py", PREFLIGHT_SOURCE_SHA)
    evidence.read(WORK / "identity_node_memory_dp.py", HELPER_SHA)
    evidence.read(Path(__file__), previous.file_sha(__file__))
    prior, expected_plan, comparison = preflight.load_inputs(evidence)
    qualified = previous.seal_read(
        PREFLIGHT, "preflight.provenance.json", preflight.STAGE, PREFLIGHT_SHA, evidence
    )
    qplan = json.loads(previous.named_read(qualified, PREFLIGHT, "plan.json", evidence))
    qsummary = json.loads(previous.named_read(qualified, PREFLIGHT, "summary.json", evidence))
    validate_preflight(qplan, qsummary, expected_plan)
    previous.input_member(
        qualified, WORK / "run_tail_minute_mouth_memory_preflight.py", PREFLIGHT_SOURCE_SHA
    )
    previous.input_member(
        qualified, preflight.PRIOR / "identity.provenance.json", preflight.PRIOR_SHA
    )
    raw_member = previous.output_member(prior, preflight.PRIOR, "tail/raw_candidate21.npz")
    previous.input_member(qualified, raw_member["path"], raw_member["sha256"])
    summary = json.loads(previous.named_read(prior, preflight.PRIOR, "summary.json", evidence))
    female_seal = previous.seal_read(
        previous.FEMALE,
        "evaluation.provenance.json",
        "long_female_endpoint_evaluation_160s",
        previous.FEMALE_SHA,
        evidence,
    )
    rows = json.loads(
        previous.named_read(female_seal, previous.FEMALE, "anchor_matches.json", evidence)
    )
    anchors = [r for r in rows if START <= r["frame"] < STOP]
    require(tuple(r["frame"] for r in anchors) == previous.ANCHOR_FRAMES, "same11 female anchors")
    plan = dict(
        schema=STAGE,
        native_interval=[START, STOP],
        geometry="tail",
        node_indices=list(NODES),
        prior_chest_identity_sha256=preflight.PRIOR_SHA,
        MP_preflight_sha256=PREFLIGHT_SHA,
        prior_input_binding=expected_plan["prior_input_binding"],
        raw_candidate_reference=raw_member,
        original_MP_capacity_failure=qsummary["methods"]["original"],
        tail_MP_capacity=qsummary["methods"]["tail"],
        dp_config=previous.DP_CONFIG,
        source_pins={
            "run_tail_minute_mouth_memory_solve.py": previous.file_sha(__file__),
            "identity_node_memory_dp.py": HELPER_SHA,
            "run_tail_minute_mouth_memory_preflight.py": PREFLIGHT_SOURCE_SHA,
        },
        initial_P=0,
        unary="float64 exact zero T,6",
        cost="unchanged chest plus mouth; each .2/3 with .8mm scale and 3log(delta)",
        fixed_prior_tie_natives=list(TIE_NATIVES),
        comparison_scope=(
            "same Tail fixed geometry; chest-only vs chest+mouth, not biological truth"
        ),
        original_MP_solved=False,
    )
    return prior, qualified, plan, summary, comparison, anchors


def exact(a, b, what):
    require(previous.signature(a) == previous.signature(b), what + " logical bytes differ")


def validate_saved_control(geometry, signatures, control, old_dp, old_scalar, counts, support):
    require(
        {k: previous.signature(v) for k, v in geometry.items()} == signatures,
        "saved Tail raw21 signatures differ",
    )
    frames = geometry["frame_idx"]
    exact(support["frame_idx"], frames, "preflight frames")
    exact(support["direct_nodes"], geometry["observed"][:, :, NODES], "preflight MP support")
    exact(counts["frame_idx"], frames, "complete counter frame range")
    require(
        counts["state_counts"].shape == (len(frames),)
        and counts["edge_counts"].shape == (len(frames) - 1,),
        "counter axes differ",
    )
    exact(control["frame_idx"], frames, "chest identity frames")
    for field in ("pose_xyz", "sources", "observed", "identity_review"):
        exact(control["raw_" + field], geometry[field], "same Tail " + field)
    exact(old_dp["frame_idx"], frames, "chest DP frames")
    exact(old_dp["permutation_by_id"], control["identity_to_raw_slot"], "chest DP mapping")
    require(
        old_scalar["status"] == "optimal"
        and old_scalar["initial_state"] == 0
        and old_scalar["scale_mm"] == 0.8
        and old_scalar["weight"] == 0.2
        and old_scalar["approximation_used"] is False,
        "same exact chest control required",
    )
    require(
        old_dp["optimal_state_mask"].dtype == bool
        and old_dp["optimal_state_mask"].shape == (len(frames), 6),
        "old tie mask axes",
    )
    actual = tuple(map(int, frames[(old_dp["optimal_state_mask"].sum(axis=1) > 1)]))
    require(actual == TIE_NATIVES, "retain all original two tied native times")


def compare_mappings(frames, before, after, direct):
    chest_arrays, chest_report = previous.mapping_comparison(frames, before, after, direct[:, :, 0])
    old = np.take_along_axis(direct, before[:, :, None], axis=1)
    new = np.take_along_axis(direct, after[:, :, None], axis=1)
    changed = before != after
    direct_cells = changed[:, :, None] & (old | new)
    latent_cells = changed[:, :, None] & ~(old | new)
    mp_any = direct_cells.any(axis=2)
    arrays = dict(
        frame_idx=frames,
        before_mapping=before,
        after_mapping=after,
        changed_by_ID=changed,
        before_direct_MP=old,
        after_direct_MP=new,
        direct_involved_MP_cells=direct_cells,
        latent_only_MP_cells=latent_cells,
        any_MP_direct_change=mp_any,
        all_MP_latent_change=changed & ~mp_any,
    )
    report = dict(
        chest_only=chest_report,
        by_node={
            str(node): dict(
                direct_involved_cells=int(direct_cells[:, :, k].sum()),
                latent_only_cells=int(latent_cells[:, :, k].sum()),
                direct_involved_intervals=previous.intervals(frames, direct_cells[:, :, k]),
                latent_only_intervals=previous.intervals(frames, latent_cells[:, :, k]),
            )
            for k, node in enumerate(NODES)
        },
        any_MP_direct_intervals=previous.intervals(frames, mp_any),
        all_MP_latent_intervals=previous.intervals(frames, changed & ~mp_any),
        changed_ID_frames=int(changed.sum()),
        changed_native_frames=int(changed.any(axis=1).sum()),
        scope=(
            "Direct means either old/new assigned raw slot supports that node; "
            "latent means neither. Reassignment is not an accuracy judgment."
        ),
    )
    return arrays, chest_arrays, report


def tie_comparison(frames, observed, old, new):
    rows = []
    for native in TIE_NATIVES:
        matches = np.flatnonzero(frames == native)
        require(len(matches) == 1, "fixed prior tie native missing")
        t = int(matches[0])
        rows.append(
            dict(
                native=native,
                raw_observed_MPT=observed[t].tolist(),
                before_mapping=old["permutation_by_id"][t].tolist(),
                after_mapping=new.permutation_by_id[t].tolist(),
                before_optimal_P=np.flatnonzero(old["optimal_state_mask"][t]).tolist(),
                after_optimal_P=np.flatnonzero(new.optimal_state_mask[t]).tolist(),
                before_state_alternative_gap=previous.json_scalar(
                    float(old["state_alternative_gap"][t])
                ),
                after_state_alternative_gap=previous.json_scalar(
                    float(new.state_alternative_gap[t])
                ),
                before_min_marginal_delta=[
                    previous.json_scalar(float(x)) for x in old["min_marginal_delta"][t]
                ],
                after_min_marginal_delta=[
                    previous.json_scalar(float(x)) for x in new.min_marginal_delta[t]
                ],
            )
        )
    return rows


def solve(geometry, counts, control, old_dp, anchors, helper):
    signatures = {k: previous.signature(v) for k, v in geometry.items()}
    frames = geometry["frame_idx"]
    xyz = geometry["pose_xyz"][:, :, NODES, :]
    direct = geometry["observed"][:, :, NODES]
    require(xyz.shape == (len(frames), 3, 2, 3) and direct.shape == (len(frames), 3, 2), "MP axes")
    result = helper.solve_identity_node_memory_dp(
        frames,
        np.zeros((len(frames), 6), np.float64),
        xyz,
        direct,
        node_indices=NODES,
        **previous.DP_CONFIG,
    )
    require(
        result.status == "optimal"
        and result.node_indices == NODES
        and not result.approximation_used,
        "exact MP result required",
    )
    require(
        np.array_equal(np.diff(result.layer_offsets), counts["state_counts"])
        and np.array_equal(np.diff(result.edge_offsets), counts["edge_counts"]),
        "graph differs from MP support count",
    )
    links = helper.node_memory_path_links(
        frames, xyz, direct, result.permutation_by_id, node_indices=NODES, scale_mm=0.8, weight=0.2
    )
    cost = float(links["cost_by_event"].sum())
    require(
        links["cost_by_event"].shape == (len(frames), 3, 2)
        and np.isclose(cost, result.path_cost, rtol=1e-11, atol=1e-10),
        "independent MP link cost differs",
    )
    for key in (
        "terminal_last_event_frame",
        "terminal_last_event_slot",
        "terminal_age_frames",
        "terminal_unsettled_gap",
    ):
        exact(links[key], getattr(result, key), "terminal link " + key)
    identity = previous.build_identity_output(geometry, result.permutation_by_id)
    before = control["identity_to_raw_slot"]
    change_arrays, chest_changes, changes = compare_mappings(
        frames, before, result.permutation_by_id, direct
    )
    old_links = helper.node_memory_path_links(
        frames, xyz, direct, before, node_indices=NODES, scale_mm=0.8, weight=0.2
    )
    # Both paths are evaluated with the same new objective. Different K objective
    # totals are retained separately and are not directly called an improvement.
    ties = tie_comparison(frames, geometry["observed"], old_dp, result)
    report = dict(
        path_cost=result.path_cost,
        independent_link_sum=cost,
        path_cost_by_node=links["cost_by_event"].sum(axis=(0, 1)).tolist(),
        chest_only_path_under_MP_cost=float(old_links["cost_by_event"].sum()),
        chest_only_path_under_MP_cost_by_node=old_links["cost_by_event"].sum(axis=(0, 1)).tolist(),
        changes=changes,
        prior_ties=ties,
        terminal_age_frames=result.terminal_age_frames.tolist(),
        terminal_unsettled_gap=result.terminal_unsettled_gap.tolist(),
        female_raw_geometry=previous.female_scores(
            frames, geometry["sources"], geometry["observed"], geometry["identity_review"], anchors
        ),
        female_proposed=previous.female_scores(
            frames,
            identity["proposed_sources"],
            identity["proposed_observed"],
            identity["prior_review_for_proposed_identity"],
            anchors,
        ),
        all_result_fields_saved=True,
        all_three_ID_truth=False,
    )
    require(signatures == {k: previous.signature(v) for k, v in geometry.items()}, "raw21 mutated")
    return result, identity, links, old_links, change_arrays, chest_changes, report


def run(args):
    evidence, output = previous.Evidence(), previous.Output(args.output)
    began = datetime.now(UTC).isoformat()
    try:
        prior, qualified, plan, old_summary, comparison, anchors = load_inputs(evidence)
        import identity_node_memory_dp as helper

        require(
            Path(helper.__file__).resolve() == (WORK / "identity_node_memory_dp.py").resolve(),
            "unexpected node solver module",
        )
        output.save("plan.json", plan)
        for path, digest in evidence.inputs.items():
            p = Path(path)
            if p.suffix == ".py":
                dst = output.root / "code_snapshot" / p.relative_to(REPO)
                dst.parent.mkdir(exist_ok=True, parents=True)
                dst.write_bytes(evidence.read(p, digest))
                output.outputs[dst.relative_to(output.root).as_posix()] = dict(
                    path=str(dst), sha256=digest, size_bytes=dst.stat().st_size
                )
        raw = previous.named_read(prior, preflight.PRIOR, "tail/raw_candidate21.npz", evidence)
        geometry, headers = previous.read_candidate(
            raw, support_only=False, first=START, start=START, stop=STOP
        )
        signatures = json.loads(
            previous.named_read(prior, preflight.PRIOR, "tail/raw_array_signatures.json", evidence)
        )
        control = read_arrays(prior, preflight.PRIOR, "tail/identity_output.npz", evidence)
        old_dp = read_arrays(prior, preflight.PRIOR, "tail/dp_arrays.npz", evidence)
        old_scalar = json.loads(
            previous.named_read(prior, preflight.PRIOR, "tail/dp_scalars.json", evidence)
        )
        counts = read_arrays(qualified, PREFLIGHT, "tail/counts.npz", evidence)
        support = read_arrays(qualified, PREFLIGHT, "tail/support.npz", evidence)
        validate_saved_control(geometry, signatures, control, old_dp, old_scalar, counts, support)
        require(
            int(counts["state_counts"].sum()) == plan["tail_MP_capacity"]["total_states"]
            and int(counts["edge_counts"].sum()) == plan["tail_MP_capacity"]["total_edges"],
            "count totals differ",
        )
        output.save("raw_candidate21.npz", geometry, arrays=True)
        output.save("raw_array_signatures.json", signatures)
        output.save("candidate_headers.json", headers)
        result, identity, links, old_links, changes, chest_changes, report = solve(
            geometry, counts, control, old_dp, anchors, helper
        )
        require(
            report["female_raw_geometry"]
            == comparison["methods"]["tail"]["female"]
            == old_summary["methods"]["tail"]["female_raw_geometry"],
            "same female raw endpoints required",
        )
        report["prior_chest_only_objective"] = old_scalar["path_cost"]
        # The new path's chest event contribution is also checked through the old
        # frozen one-node link helper; no old DP is recomputed.
        old_chest_check = previous.chosen_memory_links(
            geometry, result.permutation_by_id, scale_mm=0.8, weight=0.2
        )
        require(
            np.allclose(
                old_chest_check["brownian_motion_cost_by_id"],
                links["cost_by_event"][:, :, 0],
                rtol=1e-11,
                atol=1e-10,
            ),
            "original chest event costs changed",
        )
        require(
            np.isclose(
                old_links["cost_by_event"][:, :, 0].sum(),
                old_scalar["path_cost"],
                rtol=1e-11,
                atol=1e-10,
            ),
            "prior chest objective changed",
        )
        previous.save_result(output, "MP", result)
        for name, value in (
            ("identity_output", identity),
            ("chosen_links", links),
            ("old_path_MP_links", old_links),
            ("mapping_MP_changes", changes),
            ("mapping_chest_changes", chest_changes),
        ):
            arrays = {k: v for k, v in value.items() if isinstance(v, np.ndarray)}
            scalars = {
                k: previous.json_scalar(v)
                for k, v in value.items()
                if not isinstance(v, np.ndarray)
            }
            output.save(name + ".npz", arrays, arrays=True)
            if scalars:
                output.save(name + ".json", scalars)
        output.save("prior_ties.json", report["prior_ties"])
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[START, STOP],
            native_frames=STOP - START,
            geometry="tail",
            node_indices=list(NODES),
            original_MP_solved=False,
            original_MP_capacity_failure=plan["original_MP_capacity_failure"],
            raw_geometry_sources_review_modified=False,
            RGB_fit_GPU_executed=False,
            report=report,
            limitations=(
                "Offline model-conditional mapping and gap. Female endpoints do not validate "
                "both males or zero swaps. Terminal missing endpoints remain unsettled."
            ),
        )
        output.save("summary.json", summary)
        evidence.guard()
        output.guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
            error=repr(error),
        )
        (output.root / "identity.provenance.json").write_text(
            json.dumps(seal, indent=2, allow_nan=False), encoding="utf-8"
        )
        raise
    p = output.root / "identity.provenance.json"
    p.write_text(json.dumps(seal, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            dict(
                status="ok",
                seal_sha256=previous.file_sha(p),
                outputs=len(output.outputs),
                report=report,
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.metadata_only:
        evidence = previous.Evidence()
        _, _, plan, _, _, _ = load_inputs(evidence)
        evidence.guard()
        print(
            json.dumps(
                dict(
                    metadata_only=True,
                    inputs=len(evidence.inputs),
                    node_indices=plan["node_indices"],
                )
            )
        )
    else:
        parser.error("--output required") if args.output is None else run(args)


if __name__ == "__main__":
    main()
