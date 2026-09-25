"""Post-inference review priorities from sealed memory-identity diagnostics.

Priorities are an explicit allocation of review effort, not calibrated risk or
identity truth. At most five one-second contexts are selected; this is neither
exhaustive nor evidence that five human corrections would be sufficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import evaluate_identity_memory as evidence
import numpy as np
from identity_memory_links import chosen_memory_links

from medaka3d.provenance import provenance

common, base = evidence.common, evidence.base
CONTEXT_FRAMES = 83
MAX_SCENES = 5


def runs(values):
    values = np.asarray(values)
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    return [(int(a), int(b), values[a].item())
            for a, b in zip(starts, np.r_[starts[1:], len(values)], strict=True)]


def missing_age(observed):
    result = np.zeros_like(observed, dtype=np.int64)
    for t in range(1, len(observed)):
        result[t] = np.where(observed[t], 0, result[t-1]+1)
    return result


def extract_events(dp, identity, scores, links, motion_mapping):
    """Fixed diagnostic ordering; no annotation, outcome metric or fitted threshold."""
    n, events = len(dp["frame_idx"]), []

    def event(kind, t, priority, order, ids=(), **details):
        events.append(dict(kind=kind, frame_offset=int(t), priority=int(priority),
                           order=list(order), identity_indices=list(map(int, ids)), **details))

    segments, seen = runs(dp["path_state"]), {}
    for k, (a, _b, p) in enumerate(segments):
        if k:
            changed = np.flatnonzero(identity["identity_to_raw_slot"][a]
                                    != identity["identity_to_raw_slot"][a-1])
            event("mapping_change", a, 1, [-len(changed)], changed,
                  previous_permutation=int(segments[k-1][2]), new_permutation=int(p))
        if p in seen:
            departure = segments[seen[p]+1][0]
            event("mapping_departure_then_return", departure, 0, [-(a-departure)],
                  return_frame_offset=a, return_permutation=int(p),
                  excursion_range_offsets=[departure, a])
        seen[p] = k
    disagreement = np.any(identity["identity_to_raw_slot"] != motion_mapping, axis=1)
    for a, b, active in runs(disagreement):
        if active:
            event("color_vs_motion_mapping_difference", a, 1, [-(b-a)],
                  episode_range_offsets=[a, b])
    observed = identity["proposed_observed"][:, :, 1]
    distance = np.linalg.norm(links["displacement_mm_when_reobserved"], axis=-1)
    for t, fish in np.argwhere(links["gap_frames_when_reobserved"] > 1):
        dt = int(links["gap_frames_when_reobserved"][t, fish])
        event("reobserved_after_gap", t, 2, [-dt, -float(distance[t, fish])], [fish],
              gap_native_frames=dt, displacement_mm=float(distance[t, fish]))
    for fish in range(3):
        for a, b, missing in runs(~observed[:, fish]):
            if missing:
                event("terminal_unsettled_gap" if b == n else "direct_missing_episode",
                      b-1, 2, [-(b-a), 0.], [fish], episode_range_offsets=[a, b],
                      missing_native_frames=b-a, terminal_unsettled=bool(b == n))
        gaps = dp["identity_alternative_gap"][:, fish]
        # One candidate per equal-cost plateau at a local minimum. Infinity and
        # fixed initial constraints do not become an artificial low-risk score.
        for a, b, value in runs(gaps):
            if not np.isfinite(value) or a == 0:
                continue
            if (a == 0 or value <= gaps[a-1]) and (b == n or value <= gaps[b]):
                event("finite_identity_alternative_gap", (a+b-1)//2, 3, [float(value)], [fish],
                      alternative_gap=float(value), plateau_range_offsets=[a, b],
                      status=str(dp["identity_alternative_status"][a, fish]))
        prior_review = identity["prior_review_for_proposed_identity"][:, fish]
        for t in np.flatnonzero(prior_review & ~np.r_[False, prior_review[:-1]]):
            event("prior_review_onset", t, 4, [0.], [fish])
    # Large observed displacement is a measured diagnostic, not a max-speed gate.
    for t, fish in np.argwhere(links["motion_cost_applied"]):
        event("observed_displacement", t, 5, [-float(distance[t, fish])], [fish],
              displacement_mm=float(distance[t, fish]),
              gap_native_frames=int(links["gap_frames_when_reobserved"][t, fish]))
    return events


def select_scenes(events, n, *, context_frames=CONTEXT_FRAMES, maximum=MAX_SCENES):
    ordered = sorted(
        events, key=lambda e: (e["priority"], e["order"], e["frame_offset"], e["kind"])
    )
    selected = []
    for candidate in ordered:
        t = candidate["frame_offset"]
        a, b = max(0, t-context_frames), min(n, t+context_frames+1)
        if any(a < item["stop_offset"] and item["start_offset"] < b for item in selected):
            continue
        reasons = [e for e in events if a <= e["frame_offset"] < b]
        selected.append(dict(priority_order=len(selected)+1, start_offset=a, stop_offset=b,
                             focus_offset=t, primary_reason=candidate,
                             reason_counts=dict(Counter(e["kind"] for e in reasons))))
        if len(selected) == maximum:
            break
    return selected


class GraphIndex:
    """Sparse incoming-edge index; outgoing edges already follow source-node order."""

    def __init__(self, dp, unary):
        self.incoming_order = np.argsort(dp["edge_target"], kind="stable")
        self.incoming_offsets = np.r_[0, np.cumsum(np.bincount(
            dp["edge_target"], minlength=len(dp["state_permutation"])
        ))]
        expected = np.repeat(np.arange(dp["layer_offsets"][-2]), 6)
        if not np.array_equal(dp["edge_source"], expected):
            raise ValueError("memory graph must retain all six ordered successors of every node")
        n, count = len(unary), len(dp["state_permutation"])
        minimum = np.min(unary, axis=1)
        self.unary = unary - np.where(np.isfinite(minimum), minimum, 0.)[:, None]
        self.edge = dp["edge_cost"].copy()
        self.edge_offset = np.zeros(n)
        self.forward, self.backward = np.full(count, np.inf), np.full(count, np.inf)
        self.forward[0] = self.unary[0, 0]
        self.backward[dp["layer_offsets"][-2]:] = 0.
        # Recompute only costs on the already sealed graph. Absolute saved F/B
        # can have lost small differences after large common offsets were restored.
        with np.errstate(over="raise", invalid="raise"):
            for t in range(1, n):
                edge = slice(dp["edge_offsets"][t-1], dp["edge_offsets"][t])
                self.edge_offset[t] = self.edge[edge].min()
                self.edge[edge] -= self.edge_offset[t]
                source, target = dp["edge_source"][edge], dp["edge_target"][edge]
                np.minimum.at(self.forward, target, self.forward[source]+self.edge[edge])
                nodes = slice(dp["layer_offsets"][t], dp["layer_offsets"][t+1])
                self.forward[nodes] += self.unary[t, dp["state_permutation"][nodes]]
            for t in range(n-1, 0, -1):
                edge = slice(dp["edge_offsets"][t-1], dp["edge_offsets"][t])
                source, target = dp["edge_source"][edge], dp["edge_target"][edge]
                cost = self.edge[edge]+self.unary[t, dp["state_permutation"][target]]
                np.minimum.at(self.backward, source, cost+self.backward[target])
        self.node_marginal = self.forward+self.backward
        self.best = self.forward[dp["layer_offsets"][-2]:].min()
        marginal = np.full((n, 6), np.inf)
        frame = np.repeat(np.arange(n), np.diff(dp["layer_offsets"]))
        np.minimum.at(marginal, (frame, dp["state_permutation"]), self.node_marginal)
        delta = np.maximum(marginal-self.best, 0.)
        if not np.allclose(delta, dp["min_marginal_delta"], atol=1e-8, rtol=1e-12):
            raise ValueError("centered graph replay differs from saved min-marginal gaps")

    def incoming(self, node):
        a, b = self.incoming_offsets[node:node+2]
        return self.incoming_order[a:b]


def reconstruct_alternative(dp, unary, geometry, t, permutation, config, index=None):
    """Trace a best complete path through a particular alternative P and memory node.

    F includes current unary; B excludes it. Shared offsets are removed before
    replay/trace so absolute saved costs cannot hide small differences. The path is costed from
    last-direct links, not accepted as a corrected biological identity sequence.
    """
    if (permutation == dp["path_state"][t]
            or not np.isfinite(dp["min_marginal_delta"][t, permutation])):
        raise ValueError("a different feasible permutation is required")
    index = GraphIndex(dp, unary) if index is None else index
    n = len(dp["frame_idx"])
    layer = np.arange(dp["layer_offsets"][t], dp["layer_offsets"][t+1])
    nodes = layer[dp["state_permutation"][layer] == permutation]
    anchor = nodes[np.argmin(index.node_marginal[nodes])]
    path_nodes, path_edges = np.full(n, -1, np.int64), np.full(n-1, -1, np.int64)
    path_nodes[t] = anchor
    for k in range(t, 0, -1):
        candidates = index.incoming(path_nodes[k])
        score = index.forward[dp["edge_source"][candidates]] + index.edge[candidates]
        edge = candidates[np.argmin(score)]
        path_nodes[k-1], path_edges[k-1] = dp["edge_source"][edge], edge
    for k in range(t, n-1):
        candidates = np.arange(6*path_nodes[k], 6*path_nodes[k]+6)
        target = dp["edge_target"][candidates]
        score = (index.edge[candidates] + index.unary[k+1, dp["state_permutation"][target]]
                 + index.backward[target])
        edge = candidates[np.argmin(score)]
        path_nodes[k+1], path_edges[k] = dp["edge_target"][edge], edge
    if (np.any(path_nodes < dp["layer_offsets"][:-1])
            or np.any(path_nodes >= dp["layer_offsets"][1:])):
        raise ValueError("conditional path traversed the wrong native layer")
    path = dp["state_permutation"][path_nodes]
    if path[0] != 0 or path[t] != permutation:
        raise ValueError("conditional path violated its initial or interior constraint")
    links = chosen_memory_links(
        geometry, dp["permutations"][path],
        scale_mm=config["motion_scale_mm"], weight=config["motion_weight"],
    )
    independent = float(unary[np.arange(n), path].sum()+links["brownian_motion_cost_by_id"].sum())
    centered_unary = index.unary[np.arange(n), path].sum()
    independent_centered = float(centered_unary + (
        links["brownian_motion_cost_by_id"].sum(axis=1)-index.edge_offset
    ).sum())
    edge_cost = float(centered_unary+index.edge[path_edges].sum())
    expected_centered = float(index.node_marginal[anchor])
    expected = float(dp["node_min_marginal_cost"][anchor])
    for cost in (independent_centered, edge_cost,
                 index.best+dp["min_marginal_delta"][t, permutation]):
        if not np.isclose(cost, expected_centered, atol=1e-8, rtol=1e-12):
            raise ValueError("conditional F/B trace does not reproduce its saved constrained cost")
    changed = path != dp["path_state"]
    info = dict(
        constrained_frame=int(dp["frame_idx"][t]), alternative_permutation=int(permutation),
        anchor_memory_node=int(anchor), anchor_last_event=dp["state_last_event"][anchor].tolist(),
        saved_min_marginal_cost=expected, independent_link_objective=independent,
        independent_cost_absolute_error=abs(independent-expected),
        independent_centered_objective=independent_centered,
        centered_min_marginal_cost=expected_centered,
        independent_centered_cost_absolute_error=abs(independent_centered-expected_centered),
        conditional_gap=float(dp["min_marginal_delta"][t, permutation]),
        changed_native_frames=int(changed.sum()),
        same_first_and_last_permutation=bool(path[0] == dp["path_state"][0]
                                             and path[-1] == dp["path_state"][-1]),
        differing_ranges=[[int(dp["frame_idx"][a]), int(dp["frame_idx"][b-1]+1)]
                          for a, b, active in runs(changed) if active],
        terminal_unsettled_gap=links["terminal_unsettled_gap"].tolist(),
    )
    return path, info


def verify_evaluation(evaluation, run, producer_hashes):
    root = common.local_path(evaluation)
    seal_path = root / "evaluation.provenance.json"
    seal = common.read_json(seal_path)
    if (seal.get("status") != "ok"
            or seal.get("stage") != "three_body_identity_memory_post_inference_evaluation"
            or common.local_path(seal["params"]["run"]) != run
            or seal["params"].get("final_inputs_sources_snapshots_outputs_unchanged") is not True):
        raise ValueError("the matching memory evaluation must be completed and sealed")
    hashes = {str(seal_path): base.sha256(seal_path)}
    for role, item in seal["outputs"].items():
        common.output_binding(seal, role, item["path"], hashes)
    for role, name in (("comparison", "comparison.json"),
                       ("motion_only_identity", "motion_only_identity.npz")):
        common.output_binding(seal, role, root / name, hashes)
    recorded = {str(common.local_path(item["path"])): item["sha256"]
                for item in seal["inputs"].values()}
    for path in (run / name for name in (
        "identity.npz", "dp.npz", "scores.npz", "memory_links.npz", "experiment.provenance.json"
    )):
        if recorded.get(str(path)) != producer_hashes[str(path)]:
            raise ValueError("evaluation does not bind the selected producer arrays")
    report = common.read_json(root / "comparison.json")
    if (common.local_path(report["identity_run"]) != run
            or report.get("variable_graph_dp_export_and_links_replayed_exactly") is not True
            or report.get("original_review_cleared") is not False):
        raise ValueError("the successful evaluation lacks exact memory replay evidence")
    return common.npz(root / "motion_only_identity.npz"), hashes


def write_frame_table(path, dp, identity, scores, links, motion_mapping):
    age = missing_age(identity["proposed_observed"][:, :, 1])
    distance = np.linalg.norm(links["displacement_mm_when_reobserved"], axis=-1)
    columns = ["frame", "raw_fixed_id_to_slot", "memory_id_to_slot", "motion_only_id_to_slot"]
    kinds = ("raw_slot", "direct_mouth", "direct_pectoral", "direct_tail", "missing_age_frames",
             "alternative_gap", "alternative_status", "reobservation_dt", "displacement_mm",
             "reobservation_brownian_cost",
             "top_visible_mass", "top_visible_mass_per_valid_pixel", "prior_review")
    columns += [f"id{fish}_{kind}" for fish in range(3) for kind in kinds]
    columns += ["top_valid_pixels", "top_color_enabled", "raw_review_slots_012"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for t, frame in enumerate(dp["frame_idx"]):
            mapping = identity["identity_to_raw_slot"][t]
            row = dict(frame=int(frame), raw_fixed_id_to_slot="0 1 2",
                       memory_id_to_slot=" ".join(map(str, mapping)),
                       motion_only_id_to_slot=" ".join(map(str, motion_mapping[t])),
                       top_valid_pixels=int(scores["valid_pixels"][t, 0]),
                       top_color_enabled=bool(scores["enabled_views"][t, 0]),
                       raw_review_slots_012=" ".join(map(str, identity["raw_identity_review"][t])))
            for fish, slot in enumerate(mapping):
                gap = dp["identity_alternative_gap"][t, fish]
                mass, valid = scores["visible_mass"][t, 0, slot], scores["valid_pixels"][t, 0]
                values = [int(slot), *identity["proposed_observed"][t, fish].tolist(),
                          int(age[t, fish]), "" if not np.isfinite(gap) else float(gap),
                          str(dp["identity_alternative_status"][t, fish]),
                          int(links["gap_frames_when_reobserved"][t, fish]),
                          "" if not np.isfinite(distance[t, fish]) else float(distance[t, fish]),
                          float(links["brownian_motion_cost_by_id"][t, fish]),
                          float(mass), "" if not valid else float(mass/valid),
                          bool(identity["prior_review_for_proposed_identity"][t, fish])]
                row.update({f"id{fish}_{kind}": value
                            for kind, value in zip(kinds, values, strict=True)})
            writer.writerow(row)


def scene_statistics(a, b, dp, identity, scores, links):
    def finite_minimum(values):
        return float(np.min(values[np.isfinite(values)])) if np.isfinite(values).any() else None

    mapping = identity["identity_to_raw_slot"][a:b]
    mass = np.take_along_axis(scores["visible_mass"][a:b, 0], mapping, axis=1)
    valid = scores["valid_pixels"][a:b, 0]
    visibility = np.divide(mass, valid[:, None], out=np.full_like(mass, np.nan),
                           where=valid[:, None] > 0)
    distance = np.linalg.norm(links["displacement_mm_when_reobserved"][a:b], axis=-1)
    return dict(
        direct_node_frames_by_id=identity["proposed_observed"][a:b].sum(axis=0).tolist(),
        prior_review_frames_by_id=identity["prior_review_for_proposed_identity"][a:b].sum(0).tolist(),
        maximum_direct_missing_age_by_id=missing_age(
            identity["proposed_observed"][:, :, 1]
        )[a:b].max(axis=0).tolist(),
        maximum_reobservation_dt_by_id=links["gap_frames_when_reobserved"][a:b].max(0).tolist(),
        maximum_reobservation_displacement_mm_by_id=[
            None if not np.isfinite(distance[:, i]).any() else float(np.nanmax(distance[:, i]))
            for i in range(3)
        ],
        maximum_reobservation_brownian_cost_by_id=links["brownian_motion_cost_by_id"][a:b].max(0).tolist(),
        minimum_finite_alternative_gap_by_id=[finite_minimum(dp["identity_alternative_gap"][a:b, i])
                                             for i in range(3)],
        minimum_top_visible_mass_per_valid_pixel_by_id=[finite_minimum(visibility[:, i])
                                                       for i in range(3)],
        top_color_enabled_frames=int(scores["enabled_views"][a:b, 0].sum()),
    )


def write_selection_report(path, result):
    lines = [
        "Review priorities from the frozen identity-memory graph\n",
        "These are candidate review locations, not verified ID errors or probabilities. "
        "Original geometry and review flags remain unchanged.\n",
        "| Priority | Native frames [start, stop) | Focus | Reason | "
        "Alternative gap | Different frames / same endpoints |",
        "|---|---|---|---|---|---|",
    ]
    for scene in result["scenes"]:
        alternate = scene["conditional_alternative"]
        gap = str(alternate.get("conditional_gap", "unavailable"))
        difference = (f"{alternate.get('changed_native_frames', 'unavailable')} / "
                      f"{alternate.get('same_first_and_last_permutation', 'unavailable')}")
        lines.append(
            f"| {scene['priority_order']} | {scene['source_range']} | {scene['focus_frame']} | "
            f"{scene['primary_reason']['kind']} | {gap} | {difference} |"
        )
    lines += ["", "Each context includes at most 83 native frames on either side. "
              "A departure/return can extend outside it; selected_scenes.json says when it does.",
              "frame_comparison.csv retains every native frame's raw-slot and proposed mappings, "
              "ID gaps, missing age, reobservation motion, top-view visible mass and prior review.",
              "The complete conditional alternative paths are hypothetical. They are validated "
              "against the saved graph and independent link costs, not external identity truth.",
              "The five-scene cap is a review budget. It does not establish complete coverage "
              "or that five corrections suffice. No new raw review video was generated."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "evaluation", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = common.local_path(args.run), args.output.resolve()
    if not output.is_relative_to(base.REPO) or output.exists() or output.is_symlink():
        raise ValueError("review selection requires a new workspace output")
    source_paths = set(evidence.code_paths()) | {Path(__file__).resolve()}
    source_hashes = {str(path): base.sha256(path) for path in source_paths}
    summary, context, hashes = evidence.verify_completed(run)
    motion, evaluation_hashes = verify_evaluation(args.evaluation, run, hashes)
    hashes.update(evaluation_hashes)
    identity, dp, scores, links = (common.npz(run / (name+".npz"))
                                  for name in ("identity", "dp", "scores", "memory_links"))
    common.exact_array(motion["frame_idx"], context.frame_idx, "motion-only native frames")
    events = extract_events(dp, identity, scores, links, motion["identity_to_raw_slot"])
    scenes = select_scenes(events, len(context.frame_idx))
    index, alternative_paths = GraphIndex(dp, scores["unary"]), []
    for scene in scenes:
        a, b = scene["start_offset"], scene["stop_offset"]
        alternatives = dp["min_marginal_delta"][a:b].copy()
        alternatives[np.arange(b-a), dp["path_state"][a:b]] = np.inf
        scene["source_range"] = [int(context.frame_idx[a]), int(context.frame_idx[b-1]+1)]
        scene["focus_frame"] = int(context.frame_idx[scene["focus_offset"]])
        scene["diagnostics"] = scene_statistics(a, b, dp, identity, scores, links)
        returning = scene["primary_reason"].get("return_frame_offset")
        if returning is not None:
            scene["return_frame"] = int(context.frame_idx[returning])
            scene["return_is_inside_selected_context"] = bool(a <= returning < b)
        if np.isfinite(alternatives).any():
            local_t, p = np.unravel_index(np.argmin(alternatives), alternatives.shape)
            path, diagnostic = reconstruct_alternative(
                dp, scores["unary"], context.geometry, a+local_t, p, summary["config"], index
            )
            scene["conditional_alternative"] = diagnostic
            scene["alternative_path_index"] = len(alternative_paths)
            alternative_paths.append(path)
        else:
            scene["conditional_alternative"] = {"status": "no_finite_alternative_not_confidence"}
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_identity_memory_review_selection", output, name="selection"
    ) as record:
        record.params_from(args)
        for path, digest in {**hashes, **source_hashes}.items():
            record.input(path, path, sha256=digest)
        for path, digest in source_hashes.items():
            target = output / "code_snapshot" / Path(path).relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output("code:"+str(Path(path).relative_to(base.REPO)), target, sha256=digest)
        table = output / "frame_comparison.csv"
        write_frame_table(table, dp, identity, scores, links, motion["identity_to_raw_slot"])
        record.output("frame_comparison", table, sha256=base.sha256(table))
        paths = output / "conditional_alternative_paths.npz"
        np.savez_compressed(
            paths, frame_idx=context.frame_idx,
            path_state=np.asarray(alternative_paths, np.int64).reshape(-1, len(dp["frame_idx"])),
        )
        record.output("conditional_alternative_paths", paths, sha256=base.sha256(paths))
        result = dict(
            status="diagnostic_priorities_only", source_range=summary["frames"],
            context_native_frames_each_side=CONTEXT_FRAMES,
            maximum_scenes=MAX_SCENES, scenes=scenes,
            candidate_event_counts=dict(Counter(e["kind"] for e in events)),
            excluded_nonfinite_id_gap_entries=int((~np.isfinite(dp["identity_alternative_gap"])).sum()),
            excluded_id_gap_reason_counts=dict(Counter(map(str,
                dp["identity_alternative_status"][~np.isfinite(dp["identity_alternative_gap"])]))),
            priority_order=["mapping departure and return", "mapping/color-motion disagreement",
                            "missing/reobservation duration", "finite local alternative gaps",
                            "prior-review onset", "observed displacement"],
            old_review_preserved=True, new_identity_acceptance=False, evaluation_metrics_used=False,
            images_or_gt_used_for_ranking=False,
            registered_prefix_images_may_be_decoded_for_input_equivalence=True,
            centered_forward_backward_recomputed_on_sealed_graph=True,
            limitations=[
                "Five contexts do not cover every risk or imply that five corrections suffice.",
                "Mapping changes may be corrections or errors; gaps are not probabilities.",
                "Fixed duplicated/missing bodies cannot be repaired by ID permutations.",
                "A blank gap means no finite alternative was available, not high confidence.",
                "Top visible mass describes foreground support, not identity alone.",
                "Alternative paths remain hypothetical with all original geometry/review limits.",
            ],
        )
        path = output / "selected_scenes.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("selected_scenes", path, sha256=base.sha256(path))
        report_path = output / "RESULT.md"
        write_selection_report(report_path, result)
        record.output("report", report_path, sha256=base.sha256(report_path))
        outputs = {item["path"]: item["sha256"] for item in record.outputs.values()}
        if any(base.sha256(path) != digest
               for path, digest in {**hashes, **source_hashes, **outputs}.items()):
            raise RuntimeError("review selection inputs, sources, snapshots or outputs changed")
        record.param("inputs_sources_snapshots_outputs_unchanged", True)
    print(json.dumps({"output": str(output), "selected_scenes": len(scenes)}), flush=True)
    return result


if __name__ == "__main__":
    main()
