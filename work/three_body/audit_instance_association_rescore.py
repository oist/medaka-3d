"""Independent CPU replay of fixed-pose observation assignments and exports.

The rescore runner and its assignment implementation are not imported. The
assignment oracle enumerates the complete three-body partial-injection domain.
Original body coordinates and spatial errors remain the sealed A values.
"""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import audit_body_volume_probe as evidence
import numpy as np

p = evidence.p
WEIGHTS = np.array([1.0, 1.0, 0.35])
DENOMINATOR = 9.0 * WEIGHTS.sum()
MODES = ("original", "nodewise", "coherent")


def complete_injection(cost, null):
    """Full {-1,0,...,D-1}^3 oracle; exact ties, null-equivalent rows canonicalized."""
    domain = range(-1, cost.shape[1])
    records = []
    for choice in product(domain, repeat=3):
        real = [row for row in choice if row >= 0]
        if len(set(real)) != len(real):
            continue
        if any(row >= 0 and not cost[fish, row] < null for fish, row in enumerate(choice)):
            continue
        value = sum(null if row < 0 else cost[fish, row] for fish, row in enumerate(choice))
        records.append((float(value), choice))
    records.sort()
    minimum = records[0][0]
    return np.array(records[0][1], np.int32), sum(row[0] == minimum for row in records)


def complete_assignment(costs, coherent):
    plan = np.full((3, 3, 3), -1, np.int32)
    tie_counts = []
    for view, costs_by_row in enumerate(costs):
        if coherent:
            row_cost = (np.minimum(costs_by_row, 9.0) * WEIGHTS).sum(-1)
            row_cost[~np.any(costs_by_row < 9.0, axis=-1)] = np.inf
            rows, ties = complete_injection(row_cost, 9.0 * WEIGHTS.sum())
            for fish, row in enumerate(rows):
                if row >= 0:
                    plan[fish, view] = np.where(costs_by_row[fish, row] < 9.0, row, -1)
        else:
            ties = 1
            for node in range(3):
                rows, count = complete_injection(
                    costs_by_row[..., node] * WEIGHTS[node], 9.0 * WEIGHTS[node]
                )
                plan[:, view, node] = rows
                ties *= count
        tie_counts.append(ties)
    return plan, np.array(tie_counts, np.int64)


def raw_costs(pose, rays):
    """Expand the frozen robust half-ray expression, keeping float32 cache rounding."""
    result = []
    for vi in range(3):
        origin = np.asarray(rays.origins[vi], np.float32).astype(np.float64)
        direction = np.asarray(rays.directions[vi], np.float32).astype(np.float64)
        valid = np.isfinite(origin).all(-1) & np.isfinite(direction).all(-1)
        valid &= np.linalg.norm(direction, axis=-1) > 0
        array = np.full((3, len(origin), 3), np.inf)
        for fish in range(3):
            for node in range(3):
                eligible = valid[:, node]
                d = pose[fish, node] - origin[eligible, node]
                axis = direction[eligible, node]
                travel = np.clip((d * axis).sum(-1), 0, None)
                error = ((d - travel[:, None] * axis) ** 2).sum(-1) / 0.7**2
                robust = error.copy()
                large = error > 9.0
                robust[large] = 6 * np.sqrt(error[large] + 1e-12) - 9
                array[fish, eligible, node] = robust
        result.append(array)
    return result


def validate_plan(plan, costs, coherent=False):
    if plan.shape != (3, 3, 3) or plan.dtype != np.int32 or np.any(plan < -1):
        raise ValueError("invalid fish/view/node assignment schema")
    contribution = np.full((3, 3, 3), 9.0)
    mixed = shared = 0
    for view in range(3):
        rows = plan[:, view]
        if np.any(rows >= costs[view].shape[1]):
            raise ValueError("assignment row out of the current frame range")
        for node in range(3):
            used = rows[:, node][rows[:, node] >= 0]
            if len(set(used.tolist())) != len(used):
                raise ValueError("same view/node row used by several bodies")
        groups = [set(rows[fish][rows[fish] >= 0].tolist()) for fish in range(3)]
        mixed += sum(len(group) > 1 for group in groups)
        shared += sum(bool(groups[a] & groups[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
        for fish, node in np.argwhere(rows >= 0):
            contribution[fish, view, node] = costs[view][fish, rows[fish, node], node]
    if coherent and (mixed or shared):
        raise ValueError("whole-row assignment mixes or shares a current 2D row")
    if not np.isfinite(contribution).all():
        raise ValueError("assigned observation has invalid ray geometry")
    return float((contribution * WEIGHTS).sum() / DENOMINATOR), mixed, shared


def strict_support(pose, plan, rays, projections):
    """Independent replay of the original infinite-line 2mm and pixel 12px gates."""
    accepted = np.full((3, 3, 3), -1, np.int32)
    for fish, view, node in np.argwhere(plan >= 0):
        row = plan[fish, view, node]
        origin = rays.origins[view][row, node].astype(np.float32).astype(float)
        direction = rays.directions[view][row, node].astype(np.float32).astype(float)
        delta = pose[fish, node] - origin
        distance = np.linalg.norm(delta - np.sum(delta * direction) * direction)
        pixel = np.linalg.norm(
            projections[view][fish, node] - rays.points[p.base.VIEWS[view]][row, node]
        )
        if distance <= 2.0 and np.isfinite(pixel) and pixel <= 12.0:
            accepted[fish, view, node] = row
    return accepted, (accepted >= 0).sum(axis=1) >= 2


def non_node_parts(poses, original_parts, predicted, config):
    relative = poses - poses[:, :, 1:2]
    previous = predicted - predicted[:, 1:2]
    orientation = ((relative - previous) ** 2).sum(-1).mean((1, 2)) / 25
    return np.column_stack(
        (
            original_parts[:, 1] * config["shape_weight"],
            original_parts[:, 2] * config["motion_weight"],
            orientation * 0.05 * config["motion_weight"],
            original_parts[:, 3] * config["repulsion_weight"],
            original_parts[:, 4] * config["mask_weight"],
        )
    )


def verify_frame(arrays, scored, observations, rig, config):
    expected = {"original_plan", "original_scores"} | {
        mode + suffix
        for mode in MODES[1:]
        for suffix in (
            "_plan",
            "_node_cost",
            "_sources",
            "_direct_nodes",
            "_optimal_view_counts",
            "_scores",
        )
    }
    if set(scored) != expected:
        raise ValueError("rescore array schema differs from the registered fourteen fields")
    evidence.equal(scored["original_plan"], arrays["fit_plans"])
    evidence.equal(scored["original_scores"], arrays["fit_scores"])
    poses = arrays["fit_pose_xyz"]
    points = {v: observations["sleap_" + v] for v in p.base.VIEWS}
    rays = p.RayObservations.from_points(rig, points)
    projections = [rig.camera(view).project(poses) for view in p.base.VIEWS]
    nonnode = non_node_parts(poses, arrays["fit_loss_parts"], arrays["predicted_pose_xyz"], config)
    np.testing.assert_allclose(
        nonnode.sum(1) + arrays["fit_loss_parts"][:, 0],
        arrays["fit_scores"],
        rtol=1e-12,
        atol=1e-12,
    )
    differences = []
    for index, pose in enumerate(poses):
        costs = raw_costs(pose, rays)
        old_cost, _, _ = validate_plan(scored["original_plan"][index], costs)
        np.testing.assert_allclose(
            old_cost, arrays["fit_loss_parts"][index, 0], rtol=1e-12, atol=1e-12
        )
        differences.append(abs(old_cost - arrays["fit_loss_parts"][index, 0]))
        for mode in MODES[1:]:
            optimal, counts = complete_assignment(costs, mode == "coherent")
            evidence.equal(optimal, scored[mode + "_plan"][index])
            evidence.equal(counts, scored[mode + "_optimal_view_counts"][index])
            cost, _, _ = validate_plan(optimal, costs, coherent=mode == "coherent")
            np.testing.assert_allclose(
                cost, scored[mode + "_node_cost"][index], rtol=1e-12, atol=1e-12
            )
            source, direct = strict_support(pose, optimal, rays, [x[index] for x in projections])
            evidence.equal(source, scored[mode + "_sources"][index])
            evidence.equal(direct, scored[mode + "_direct_nodes"][index])
            np.testing.assert_allclose(
                nonnode[index].sum() + cost, scored[mode + "_scores"][index], rtol=1e-12, atol=1e-12
            )
    if np.any(scored["coherent_node_cost"] < scored["nodewise_node_cost"] - 1e-12):
        raise ValueError("constrained minimum is below its relaxed superset")
    if np.any(scored["nodewise_node_cost"] > arrays["fit_loss_parts"][:, 0] + 1e-12):
        raise ValueError("relaxed minimum exceeds an available original assignment")
    return dict(poses=len(poses), maximum_node_replay_difference=max(differences)), nonnode


def verify_selections(frame_summary, arrays, scored, nonnode):
    rows = []
    for arm in ("refit_original", "mask", "random"):
        if arm == "refit_original":
            admitted = np.array([0])
        else:
            mapping = arrays[arm + "_fit_index"]
            admitted = mapping[mapping >= 0]
        for mode in MODES:
            row = frame_summary["arms"][arm][mode]
            index = int(admitted[np.argmin(scored[mode + "_scores"][admitted])])
            if row["fit_index"] != index:
                raise ValueError("selected pose is not its arm minimum of the registered score")
            expected = dict(
                score=scored[mode + "_scores"][index],
                spatial_error=arrays["evaluation_mean_view_loss"][index + 1],
            )
            node = (
                arrays["fit_loss_parts"][index, 0]
                if mode == "original"
                else scored[mode + "_node_cost"][index]
            )
            expected["node_cost"] = node
            for field, value in zip(
                (
                    "weighted_shape",
                    "weighted_pectoral_motion",
                    "weighted_orientation",
                    "weighted_repulsion",
                    "weighted_silhouette",
                ),
                nonnode[index],
                strict=True,
            ):
                expected[field] = value
            for field, value in expected.items():
                np.testing.assert_allclose(row[field], value, rtol=1e-12, atol=1e-12)
            plan = scored[mode + "_plan"][index]
            mixed = shared = 0
            for view in range(3):
                groups = [set(plan[f, view][plan[f, view] >= 0].tolist()) for f in range(3)]
                mixed += sum(len(g) > 1 for g in groups)
                shared += sum(bool(groups[a] & groups[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
            direct = (
                arrays["fit_observed"][index]
                if mode == "original"
                else scored[mode + "_direct_nodes"][index]
            )
            for field, value in dict(
                chimeric_fish_views=mixed,
                shared_row_fish_pairs=shared,
                direct_nodes=int(direct.sum()),
                direct_pectoral=int(direct[:, 1].sum()),
            ).items():
                if row[field] != value:
                    raise ValueError("summary support/grouping count disagrees with saved arrays")
            rows.append(dict(row))
    return rows


def region_comparison(root, seal, arrays, chosen):
    summary = json.loads(evidence.require_output(root / "summary.json", seal).read_text())
    data = p._npz(evidence.require_output(root / "all_candidates.npz", seal))
    indexes = []
    for fit in (0, chosen):
        matches = [
            i
            for i, row in enumerate(summary["candidate_rows"])
            if row["stage"] == "after_14_step_fit" and row["fit_index"] == fit
        ]
        if len(matches) != 1:
            raise ValueError("dense attribution does not uniquely identify the A fitted pose")
        index = matches[0]
        evidence.equal(data["pose_xyz"][index], arrays["fit_pose_xyz"][fit])
        indexes.append(index)
    poses = arrays["fit_pose_xyz"][[0, chosen]]
    return dict(
        fit_indices=[0, chosen],
        dense_indices=indexes,
        original_region_definitions=summary["region_definitions"],
        unexplained_fraction=data["unexplained_fraction"][indexes],
        shared_pair_fraction=data["shared_pair_fraction"][indexes],
        pair_order=summary["shared_pair_order"],
        pectoral_pair_distances_mm=np.stack(
            [
                np.linalg.norm(poses[:, a, 1] - poses[:, b, 1], axis=-1)
                for a, b in [(0, 1), (0, 2), (1, 2)]
            ],
            axis=-1,
        ),
        selected_vs_original_pectoral_displacement_mm=np.linalg.norm(
            poses[1, :, 1] - arrays["original_pose_xyz"][:, 1], axis=-1
        ),
    )


def write_report(path, rows, summary):
    lines = [
        "# Fixed-pose instance-association rescore: independent audit",
        "",
        "All 1,384 previously fitted poses were retained. This audit recomputed half-ray "
        "costs, enumerated the full partial-injection domain, replayed support, and checked "
        "all selected indices and exports. No geometry was fitted again.",
        "",
        "| Frame | Old/fresh/coherent mask indices | Spatial error old/fresh/coherent | "
        "Direct nodes old/fresh/coherent |",
        "|---|---|---|---|",
    ]
    for frame in p.FRAMES:
        group = [
            next(
                row
                for row in rows
                if row["frame"] == frame and row["arm"] == "mask" and row["assignment"] == mode
            )
            for mode in MODES
        ]
        indices = "/".join(str(row["fit_index"]) for row in group)
        error = "/".join(f"{row['spatial_error']:.5f}" for row in group)
        support = "/".join(str(row["direct_nodes"]) for row in group)
        lines.append(f"| {frame} | {indices} | {error} | {support} |")
    lines += [
        "",
        "The new coherent constraint prevents combining rows in one body and sharing "
        "a row across bodies. Its score at any fixed pose cannot beat the relaxed "
        "nodewise minimum. Different winners reflect a change in observation assignment "
        "and relative penalties within this finite pool; they are not new tracked poses.",
        "",
        "Fresh nodewise reassignment must be kept as a separate control: the original "
        "fit retained each start plan through fourteen optimizer steps. A change from "
        "that original plan is not attributable solely to instance coherence.",
        "",
        "At 79372 the coherent mask winner is fit index "
        f"{summary['frame79372_dense_attribution']['fit_indices'][1]}. The previously sealed "
        "dense attribution is joined by exact pose bytes, not by a nearest pose. The "
        "top row-0 and side1 row-2 regions remain almost entirely unexplained. Reducing "
        "S0/S1 overlap on top row 2 can coincide with moving S0 towards S2 instead. "
        "Eliminating source-row mixing therefore does not establish a complete image "
        "explanation for three fish.",
        "",
        "All spatial errors come from the original saved A evaluation pixels. Those "
        "pixels were not used to select this rescore, but share the same images and D2 "
        "detector. SLEAP rows may themselves be missing, split, merged or misgrouped. "
        "No biological ID, temporal continuity, old review or accepted trajectory changed.",
        "",
        "Across the thirteen mask-arm selections (lower image error is better):",
        "",
        "| Assignment | Mean spatial error | Direct nodes / 117 | Mixed body-view cells | "
        "Shared body pairs across views |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, values in summary["mask_arm_aggregate"].items():
        lines.append(
            f"| {mode} | {values['mean_spatial_error']:.8f} | {values['direct_nodes']} | "
            f"{values['mixed_body_view_cells']} | {values['shared_body_pairs_across_views']} |"
        )
    lines += [
        "",
        "Mixed cells and shared pairs here use assignment plans before strict support "
        "filtering. A shared pair is counted once per view if its two bodies use any "
        "common row, regardless of how many rows they share; this differs from counting "
        "each shared instance separately. Neither statistic is an ID-swap count.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(p.REPO) or output.exists():
        raise FileExistsError("new workspace audit output required")
    output.mkdir(parents=True)
    with p.provenance(
        "three_body_instance_association_independent_audit", output, name="audit"
    ) as rec:
        hashes = {
            str(path): p.base.sha256(path)
            for path in [
                *p.source_paths(),
                Path(evidence.__file__).resolve(),
                Path(__file__).resolve(),
            ]
        }
        sealed = []
        for root, stage, name in [
            (args.run, "three_body_fixed_pose_instance_association", "rescore"),
            (args.body, "three_body_local_volume_start_probe", "experiment"),
            (args.regions, "body_pool_region_diagnostic_79372", "diagnostic"),
        ]:
            seal, bound = evidence.completed(root.resolve(), stage, name)
            evidence.merge_hashes(hashes, bound)
            sealed.append(seal)
        rescore_seal, body_seal, region_seal = sealed
        for path, digest in hashes.items():
            rec.input(path, path, sha256=digest)
        summary = json.loads(
            evidence.require_output(args.run / "summary.json", rescore_seal).read_text()
        )
        old = json.loads(evidence.require_output(args.body / "summary.json", body_seal).read_text())
        cfg = old["tracker_config"]
        if cfg["ray_sigma_mm"] != 0.7 or cfg["missing_cost"] != 9:
            raise ValueError("changed fixed observation objective")
        calibration = Path(old["original_geometry_run"]) / "calibration_snapshot"
        for path in calibration.rglob("*"):
            if path.is_file() and str(path.resolve()) not in hashes:
                raise ValueError("calibration read path lacks a verified producer binding")
        rig = p.load_rig(calibration)
        if [row["frame"] for row in summary["frames"]] != list(p.FRAMES):
            raise ValueError("missing or reordered registered frame")
        rows, audits, dense = [], [], None
        for frame_summary in summary["frames"]:
            frame = frame_summary["frame"]
            root = args.body / f"frame_{frame}"
            arrays = p._npz(evidence.require_output(root / "candidates.npz", body_seal))
            observations = p._npz(evidence.require_output(root / "observations.npz", body_seal))
            scored = p._npz(evidence.require_output(args.run / f"frame_{frame}.npz", rescore_seal))
            audit, nonnode = verify_frame(arrays, scored, observations, rig, cfg)
            audits.append(dict(frame=frame, **audit))
            selected = verify_selections(frame_summary, arrays, scored, nonnode)
            rows.extend(selected)
            if frame == 79372:
                chosen = next(
                    row["fit_index"]
                    for row in selected
                    if row["arm"] == "mask" and row["assignment"] == "coherent"
                )
                dense = region_comparison(args.regions, region_seal, arrays, chosen)
            print(json.dumps(dict(frame=frame, verified_poses=audit["poses"])), flush=True)
        if sum(row["poses"] for row in audits) != 1384 or summary["total_poses"] != 1384:
            raise ValueError("registered body pool count differs")
        csv_path = evidence.require_output(args.run / "selected.csv", rescore_seal)
        with csv_path.open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != len(rows):
            raise ValueError("selected CSV row count differs")
        for actual, expected in zip(csv_rows, rows, strict=True):
            if set(actual) != set(expected):
                raise ValueError("selected CSV fields differ")
            for key, value in expected.items():
                if isinstance(value, str):
                    if actual[key] != value:
                        raise ValueError("selected CSV string/order differs")
                else:
                    np.testing.assert_allclose(float(actual[key]), value, rtol=1e-12, atol=1e-12)
        result = dict(
            status="completed_independent_assignment_output_audit",
            full_pool_poses=1384,
            full_injection_enumeration=True,
            audits=audits,
            selected=rows,
            frame79372_dense_attribution=dense,
            strict_source_and_tie_counts_exact=True,
            source_or_pose_mutation=False,
            ground_truth_used=False,
            refit=False,
            gpu_used=False,
            original_fixed_pool_pixels_and_geometry_preserved=True,
            mask_arm_aggregate={
                mode: dict(
                    mean_spatial_error=float(
                        np.mean(
                            [
                                row["spatial_error"]
                                for row in rows
                                if row["arm"] == "mask" and row["assignment"] == mode
                            ]
                        )
                    ),
                    direct_nodes=sum(
                        row["direct_nodes"]
                        for row in rows
                        if row["arm"] == "mask" and row["assignment"] == mode
                    ),
                    mixed_body_view_cells=sum(
                        row["chimeric_fish_views"]
                        for row in rows
                        if row["arm"] == "mask" and row["assignment"] == mode
                    ),
                    shared_body_pairs_across_views=sum(
                        row["shared_row_fish_pairs"]
                        for row in rows
                        if row["arm"] == "mask" and row["assignment"] == mode
                    ),
                )
                for mode in MODES
            },
        )
        p.save_json(rec, output, "summary", "summary.json", result)
        report = output / "RESULT.md"
        write_report(report, rows, result)
        rec.output("report", report, sha256=p.base.sha256(report))
        p.assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in rec.outputs.values()}})
        rec.param("all_input_source_output_sha_unchanged", True)
    return result


if __name__ == "__main__":
    main()
