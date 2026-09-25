"""Evaluate a sealed observation-memory ID graph on immutable fish geometry.

Both full color+motion and zero-color memory objectives are solved and their
exports checked before female annotations are opened. No GPU, renderer, body
fit, template update, review clearing or existing-file mutation occurs here.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import evaluate_identity_dp as common
import numpy as np
import run_experiment as base
from audit_appearance_enrollment import exact_array
from identity_inputs import load_identity_inputs
from identity_memory_dp import solve_identity_memory_dp
from identity_memory_links import chosen_memory_links
from identity_output import build_identity_output

from medaka3d.provenance import provenance

STAGE = "three_body_fixed_geometry_identity_memory_dp"
OUTPUTS = {**common.OUTPUTS, "memory_links": "memory_links.npz"}
FIXED_CONFIG = {
    **common.FIXED_CONFIG,
    "memory": {
        "model": "brownian_direct_event",
        "max_layer_states": 100000,
        "max_total_states": 1000000,
        "max_total_edges": 6000000,
    },
}
REQUIRED_MEMORY_SOURCE = {
    f"labs/20260909_three_body/{name}"
    for name in (
        "run_identity_memory.py",
        "identity_memory_dp.py",
        "identity_memory_links.py",
        "IDENTITY_MEMORY_PROTOCOL.md",
    )
}
IMAGE_SHAPES = {
    "frame_idx": (),
    "unary": (6,),
    "per_view_costs": (6, 3),
    "enabled_views": (3,),
    "sampled_pixels": (3,),
    "valid_pixels": (3,),
    "available_core_pixels": (3,),
    "visible_mass": (3, 3),
    "rendering_performed": (),
}


def verify_completed(run):
    """Bind the new stage/config/output contract; no annotation/reference arrays are read."""
    run = common.local_path(run)
    record_path = run / "experiment.provenance.json"
    record = common.read_json(record_path)
    if record.get("status") != "ok" or record.get("stage") != STAGE:
        raise ValueError("memory identity producer is not a completed registered stage")
    params = record["params"]
    if common.local_path(params["output"]) != run:
        raise ValueError("memory producer output path differs from its seal")
    for key, value in (
        ("evaluation_data_opened_during_inference", False),
        ("code_unchanged_during_run", True),
        ("inputs_unchanged_during_run", True),
    ):
        if params.get(key) is not value:
            raise ValueError(f"memory producer lacks completion evidence: {key}")
    hashes = {str(record_path): base.sha256(record_path)}
    for role, filename in OUTPUTS.items():
        common.output_binding(record, role, run / filename, hashes)
    summary = common.read_json(run / "summary.json")
    if not common.equal_json(params["config"], FIXED_CONFIG) or not common.equal_json(
        summary["config"], FIXED_CONFIG
    ):
        raise ValueError(
            "memory objective or resource caps differ from the registered configuration"
        )
    if params["package_versions"].get("numpy") != np.__version__:
        raise ValueError("exact CPU memory replay requires the producer NumPy version")
    for key, value in (
        ("protocol", STAGE),
        ("technical_completion", True),
        ("native_fps", 166),
        ("geometry_initialization_count", 1),
        ("external_reset_count", 0),
        ("identity_initial_state", 0),
        ("geometry_fits_in_this_stage", 0),
        ("temporal_subsampling", False),
        ("raw_geometry_and_review_preserved", True),
        ("evaluation_data_opened_during_inference", False),
        ("code_unchanged_during_run", True),
        ("inputs_unchanged_during_run", True),
    ):
        if not common.equal_json(summary.get(key), value):
            raise ValueError(f"memory completion metadata differs: {key}")
    declared_code = {base.safe_relative(k).as_posix() for k in params["code_hashes"]}
    if not REQUIRED_MEMORY_SOURCE.issubset(declared_code):
        raise ValueError("memory-specific inference source is missing")
    hashes.update(
        common.verify_sources(run, record, summary, common.read_json(run / "source_manifest.json"))
    )
    context = load_identity_inputs(
        common.local_path(params["geometry_run"]),
        common.local_path(params["enrollment_run"]),
        common.local_path(params["enrollment_audit"]),
    )
    frames = context.frame_idx
    bounds = [int(frames[0]), int(frames[-1] + 1)]
    if bounds not in ([333519, 333719], [76620, 77286], [76620, 86580]):
        raise ValueError("memory evaluation requires a full registered 200/666/9960 native window")
    if (
        summary["frames"] != bounds
        or summary["requested_frames"] != len(frames)
        or summary["returned_frames"] != len(frames)
    ):
        raise ValueError("memory output native interval differs from immutable geometry")
    recorded = {}
    for key, item in record["inputs"].items():
        path = common.local_path(item["path"])
        if not Path(key).is_absolute() or common.local_path(key) != path or str(path) in recorded:
            raise ValueError("memory inputs must be unique absolute-path records")
        recorded[str(path)] = item["sha256"]
    if recorded != context.verified_hashes:
        raise ValueError("memory producer inputs differ from independently verified evidence")
    for name in ("input_sha256_before", "input_sha256_after"):
        values = {str(common.local_path(k)): v for k, v in summary[name].items()}
        if values != recorded or len(values) != len(summary[name]):
            raise ValueError("memory inference inputs changed or lack before/after binding")
    if not common.equal_json(common.read_json(run / "prefix_audit.json"), context.prefix_audit):
        raise ValueError("memory prefix audit differs from the independently verified transfer")
    hashes.update(context.verified_hashes)
    return summary, context, hashes


def validate_image_scores(scores, frames, enrollment, config):
    """Validate the nine frozen image fields without inventing adjacent-only transitions."""
    if set(scores) != set(IMAGE_SHAPES):
        raise ValueError("memory scores must contain exactly the nine image arrays")
    for key, tail in IMAGE_SHAPES.items():
        if scores[key].shape != (len(frames), *tail) or scores[key].dtype.hasobject:
            raise ValueError(f"image field shape/type mismatch: {key}")
    exact_array(scores["frame_idx"], frames, "scores native frames")
    for key in ("enabled_views", "rendering_performed"):
        if scores[key].dtype != bool:
            raise ValueError(f"image flags must be boolean: {key}")
    for key in ("sampled_pixels", "valid_pixels", "available_core_pixels"):
        if scores[key].dtype != np.int64 or np.any(scores[key] < 0):
            raise ValueError(f"image counts must be nonnegative int64: {key}")
    for key in ("unary", "per_view_costs", "visible_mass"):
        if (
            scores[key].dtype != np.float64
            or not np.isfinite(scores[key]).all()
            or np.any(scores[key] < 0)
        ):
            raise ValueError(f"image costs/masses must be finite nonnegative float64: {key}")
    samples, valid, available = (
        scores[k] for k in ("sampled_pixels", "valid_pixels", "available_core_pixels")
    )
    if (
        np.any(valid > samples)
        or np.any(samples > available)
        or np.any(samples > config["emission"]["maximum_samples"])
    ):
        raise ValueError("image sample counts are inconsistent")
    permitted = np.array([view in config["emission"]["active_views"] for view in base.VIEWS])
    permitted &= enrollment["enabled_views"]
    if (
        np.any(samples[:, ~permitted])
        or np.any(scores["enabled_views"][:, ~permitted])
        or not np.array_equal(scores["enabled_views"], valid > 0)
        or not np.array_equal(scores["rendering_performed"], samples.any(axis=1))
    ):
        raise ValueError("image scores used an unavailable view or unrecorded rendering")
    for key in IMAGE_SHAPES.keys() - {"frame_idx"}:
        if np.any(scores[key][: config["training_frames"]] != 0):
            raise ValueError("causal enrollment frames cannot provide color costs")
    if np.any(scores["visible_mass"].sum(axis=2) > valid + 1e-9):
        raise ValueError("visible body mass exceeds observed valid pixel mass")
    if np.any(scores["per_view_costs"] * (~scores["enabled_views"])[:, None] != 0):
        raise ValueError("disabled views must have neutral color costs")
    expected = (
        config["appearance_weight"]
        * (scores["per_view_costs"] * valid[:, None]).sum(axis=2)
        / np.maximum(valid.sum(1), 1)[:, None]
    )
    if not np.allclose(scores["unary"], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("memory unary is not the recorded valid-pixel mean of view scores")


def solve_memory(geometry, unary, config):
    memory = config["memory"]
    if memory["model"] != "brownian_direct_event":
        raise ValueError("unknown direct-event motion model")
    return solve_identity_memory_dp(
        geometry["frame_idx"],
        unary,
        geometry["xyz"],
        geometry["observed"][:, :, 1],
        scale_mm=config["motion_scale_mm"],
        weight=config["motion_weight"],
        initial_state=config["initial_state"],
        tie_atol=config["tie_atol"],
        tie_rtol=config["tie_rtol"],
        **{key: memory[key] for key in ("max_layer_states", "max_total_states", "max_total_edges")},
    )


def validate_path_events(geometry, result):
    """A canonical permutation index is never a global variable-layer node index."""
    n = len(geometry["frame_idx"])
    nodes = result.path_node
    if (
        nodes.shape != (n,)
        or nodes.dtype != np.int64
        or np.any(nodes < result.layer_offsets[:-1])
        or np.any(nodes >= result.layer_offsets[1:])
    ):
        raise ValueError("memory path_node does not belong to its native frame layer")
    if not np.array_equal(result.state_permutation[nodes], result.path_state):
        raise ValueError("memory path node and canonical permutation indices disagree")
    if not np.array_equal(result.permutations[result.path_state], result.permutation_by_id):
        raise ValueError("memory path_state does not encode the exported raw-slot mapping")
    events = np.arange(3, dtype=np.int64)
    for t, slots in enumerate(result.permutation_by_id):
        direct = geometry["observed"][t, slots, 1]
        events = np.where(direct, 3 * t + slots, events)
        if not np.array_equal(result.state_last_event[nodes[t]], events):
            raise ValueError("chosen memory event is not the last direct observation for that ID")


def solution_totals(result, scores, links):
    return dict(
        scored_frames=int(scores["enabled_views"].any(axis=1).sum()),
        sampled_pixels=scores["sampled_pixels"].sum(axis=0).tolist(),
        mapping_changes=int(np.count_nonzero(np.diff(result.path_state))),
        nonidentity_mapping_frames=int(np.count_nonzero(result.path_state)),
        reobservation_links=int(links["motion_cost_applied"].sum()),
        terminal_unobserved_age_frames=links["terminal_unobserved_age_frames"].tolist(),
        terminal_unsettled_gap=links["terminal_unsettled_gap"].tolist(),
    )


def independent_links(geometry, result, unary, config):
    links = chosen_memory_links(
        geometry,
        result.permutation_by_id,
        scale_mm=config["motion_scale_mm"],
        weight=config["motion_weight"],
    )
    total = float(
        unary[np.arange(len(unary)), result.path_state].sum()
        + links["brownian_motion_cost_by_id"].sum()
    )
    if not np.isclose(result.path_cost, total, rtol=1e-11, atol=1e-10):
        raise ValueError("memory objective differs from independent chosen reobservation links")
    for field, key in (
        ("terminal_last_event_frame", "terminal_last_direct_frame"),
        ("terminal_last_event_slot", "terminal_last_direct_raw_slot"),
        ("terminal_age_frames", "terminal_unobserved_age_frames"),
        ("terminal_unsettled_gap", "terminal_unsettled_gap"),
    ):
        exact_array(getattr(result, field), links[key], "terminal/" + field)
    return links


def validate_saved_solution(geometry, enrollment, identity, dp, scores, memory_links, summary):
    config = summary["config"]
    validate_image_scores(scores, geometry["frame_idx"], enrollment, config)
    solved = solve_memory(geometry, scores["unary"], config)
    if solved.status != "optimal" or solved.approximation_used:
        raise ValueError("completed memory inference must have an exact feasible solution")
    validate_path_events(geometry, solved)
    expected_arrays, expected_scalars = common.split_solution(solved)
    if set(dp) != set(expected_arrays):
        raise ValueError("saved memory DP schema differs from its variable-state result")
    for key, value in expected_arrays.items():
        exact_array(dp[key], value, "memory DP/" + key)
    if not common.equal_json(summary["dp"], expected_scalars):
        raise ValueError("memory DP scalar metadata differs from complete CPU replay")
    links = independent_links(geometry, solved, scores["unary"], config)
    if set(memory_links) != set(links):
        raise ValueError("saved reobservation link fields differ from the independent calculation")
    for key, value in links.items():
        exact_array(memory_links[key], value, "memory links/" + key)
    for key, value in solution_totals(solved, scores, links).items():
        if not common.equal_json(summary.get(key), value):
            raise ValueError(f"memory producer total differs from saved evidence: {key}")
    expected_identity = build_identity_output(geometry, solved.permutation_by_id)
    if set(identity) != set(expected_identity):
        raise ValueError("memory identity export has missing or unknown arrays")
    for key, value in expected_identity.items():
        exact_array(identity[key], value, "memory identity/" + key)
    return solved, common.conservation_counts(geometry, identity)


def motion_only_solution(geometry, scores, config):
    """Use the same support-memory graph and Brownian costs with all color unaries zero."""
    unary = np.zeros_like(scores["unary"])
    solved = solve_memory(geometry, unary, config)
    if solved.status != "optimal" or solved.approximation_used:
        raise ValueError("motion-only memory solution must complete exactly")
    validate_path_events(geometry, solved)
    identity = build_identity_output(geometry, solved.permutation_by_id)
    common.conservation_counts(geometry, identity)
    return solved, identity, independent_links(geometry, solved, unary, config)


def code_paths():
    paths = {Path(__file__).resolve(), base.LAB / "IDENTITY_MEMORY_PROTOCOL.md"}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(base.LAB) or path.is_relative_to(base.REPO / "src")
            ):
                paths.add(path)
    return sorted(paths)


def write_report(path, report):
    lines = [
        "Frozen-geometry identity memory comparison\n",
        "Every method retains the original geometry, direct support and prior reviews.\n",
        "| Method | Female raw preserved / swap / unscored | Prior-review accepted |",
        "|---|---|---|",
    ]
    for name, method in report["methods"].items():
        cells = [
            " / ".join(
                str(method[key]["interval_counts"].get(k, 0))
                for k in ("preserved", "swap", "unscored")
            )
            for key in ("female_raw", "female_accepted")
        ]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "Relative female endpoint gate: "
        + report["raw_female_endpoint_regression_gate"]["status"]
        + ".",
        "The graph retains last direct observations through gaps; both objectives are solved "
        "before annotations are read. Motion-only uses zero color unaries with the same memory "
        "states, motion model and first-state constraint.",
        "Brownian motion costs are uncalibrated; terminal gaps lack a returning observation. "
        "Ties, large model gaps and female endpoints do not prove male or dense identity. "
        "Original reviews remain, and no swap-free identity is certified.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(base.REPO.resolve()) or output.exists() or output.is_symlink():
        raise FileExistsError("memory evaluation requires a new workspace output directory")
    output.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_identity_memory_post_inference_evaluation", output, name="evaluation"
    ) as record:
        record.params_from(args)
        code = {str(path): base.sha256(path) for path in code_paths()}
        summary, context, hashes = verify_completed(args.run)
        run = common.local_path(args.run)
        identity, dp, scores, links = (
            common.npz(run / f"{name}.npz") for name in ("identity", "dp", "scores", "memory_links")
        )
        solved, counts = validate_saved_solution(
            context.geometry, context.enrollment, identity, dp, scores, links, summary
        )
        motion_dp, motion_identity, motion_links = motion_only_solution(
            context.geometry, scores, summary["config"]
        )
        motion_arrays, motion_scalars = common.split_solution(motion_dp)
        for key in (
            "layer_offsets",
            "state_permutation",
            "state_last_event",
            "edge_offsets",
            "edge_source",
            "edge_target",
            "edge_cost",
            "edge_delta_frames",
            "edge_squared_distance_mm2",
        ):
            exact_array(getattr(motion_dp, key), getattr(solved, key), "same memory graph/" + key)
        for role, arrays in (
            ("motion_only_dp", motion_arrays),
            ("motion_only_identity", motion_identity),
            ("motion_only_links", motion_links),
        ):
            path = output / f"{role}.npz"
            np.savez_compressed(path, **arrays)
            record.output(role, path, sha256=base.sha256(path))
        # Both exact objectives are now fixed. Only evaluation opens annotations below.
        catalog, _ = base.staging_catalog(context.input_root)
        _, _, registered = base.registered_seed(int(context.frame_idx[0]))
        gt_path = base.verified_input(
            context.input_root, registered["female_annotations_relative"], catalog
        )
        if base.sha256(gt_path) != registered["female_annotations_sha256"]:
            raise ValueError("female annotations differ from the registered common evaluator")
        hashes[str(gt_path)] = base.sha256(gt_path)
        nodes = common.shared.CachedNodes(context.input_root, base.EXPERIMENT, context.frame_idx)
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = common.shared.match_mouth_anchors(
                csv.DictReader(stream),
                lambda view, frame: nodes.at(view, frame)[:, 0],
                int(context.frame_idx[0]),
                int(context.frame_idx[-1] + 1),
            )
        methods = {}
        for label, arrays in (
            ("frozen_geometry_ids", context.geometry),
            ("memory_color_ids", common.proposed_geometry(context.frame_idx, identity)),
            (
                "memory_motion_only_ids",
                common.proposed_geometry(context.frame_idx, motion_identity),
            ),
        ):
            source, review, observed, xyz, accepted = common.shared.normalized_points(arrays)
            common.shared.validate_sources(source, context.frame_idx, nodes)
            methods[label] = common.shared.common_metrics(
                context.frame_idx, source, review, observed, xyz, accepted, anchors
            )
        raw, changed = methods["frozen_geometry_ids"], methods["memory_color_ids"]
        gate = common.female_regression_gate(raw["female_raw"], changed["female_raw"])
        report = dict(
            status="technical_memory_post_inference_comparison",
            frames=summary["frames"],
            native_fps=166,
            native_frames=len(context.frame_idx),
            methods=methods,
            raw_female_endpoint_regression_gate=gate,
            conservative_female_endpoint_diagnostic=common.female_regression_gate(
                raw["female_accepted"], changed["female_accepted"]
            ),
            motion_only_raw_female_endpoint_regression_gate=common.female_regression_gate(
                raw["female_raw"], methods["memory_motion_only_ids"]["female_raw"]
            ),
            motion_only_reference=dict(
                dp=motion_scalars,
                **solution_totals(motion_dp, scores, motion_links),
                unary="all zero",
                same_support_memory_graph=True,
                same_initial_state=True,
                computed_before_reading_evaluation_labels=True,
                note="Sampling totals describe the color input; motion-only renders no images.",
            ),
            conservation=counts,
            geometry_run=str(context.geometry_root),
            identity_run=str(run),
            identity_npz_sha256=hashes[str(run / "identity.npz")],
            geometry_npz_sha256=hashes[str(context.geometry_root / "candidate.npz")],
            variable_graph_dp_export_and_links_replayed_exactly=True,
            color_emissions_rerendered=False,
            original_review_cleared=False,
            independent_all_three_truth_loaded=False,
            dp_global_alternative_gap=common.scalar_json(solved.global_alternative_gap),
            dp_global_alternative_status=solved.global_alternative_status,
            limitations=[
                "Geometry/support conservation is a construction invariant, not improved accuracy.",
                "Brownian direct-event motion is uncalibrated; no observation model is used.",
                "Terminal missing endpoints remain unsettled; zero missing cost is not certainty.",
                "Conditional min-marginal gaps do not certify biological identity.",
                "Female endpoints do not verify males, dense IDs or swaps followed by a return.",
                "Frozen duplicates or missing bodies cannot be repaired by assigning identity.",
            ],
        )
        for path, digest in {**hashes, **code}.items():
            record.input(path, path, sha256=digest)
        for path, digest in code.items():
            target = output / "code_snapshot" / Path(path).relative_to(base.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if base.sha256(target) != digest:
                raise RuntimeError("memory evaluator source changed during snapshot")
            record.output("code:" + str(Path(path).relative_to(base.REPO)), target, sha256=digest)
        for role, filename in (("comparison", "comparison.json"), ("report", "RESULT.md")):
            path = output / filename
            if role == "comparison":
                path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            else:
                write_report(path, report)
            record.output(role, path, sha256=base.sha256(path))
        outputs = {entry["path"]: entry["sha256"] for entry in record.outputs.values()}
        for path, digest in {**hashes, **code, **outputs}.items():
            if base.sha256(path) != digest:
                raise RuntimeError("memory evaluation input/source/snapshot/output changed")
        record.param("final_inputs_sources_snapshots_outputs_unchanged", True)
        record.param("both_memory_objectives_solved_before_gt", True)
    print(
        json.dumps(
            dict(
                output=str(output),
                female_endpoint_gate=gate["status"],
                identity_verdict="unverified",
            )
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
