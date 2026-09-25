"""Post-inference local A findings, with all thirteen frames and explicit limits."""

import argparse
import json
from pathlib import Path

import audit_body_volume_probe as audit
import numpy as np

p = audit.p


def cost_parts(poses, parts, predicted, cfg):
    relative = poses - poses[:, :, 1:2]
    prior_relative = predicted - predicted[:, 1:2]
    orientation = ((relative - prior_relative) ** 2).sum(-1).mean((1, 2)) / 25
    return np.column_stack(
        (
            parts[:, 0],
            cfg["shape_weight"] * parts[:, 1],
            cfg["motion_weight"] * parts[:, 2],
            cfg["motion_weight"] * 0.05 * orientation,
            cfg["repulsion_weight"] * parts[:, 3],
            cfg["mask_weight"] * parts[:, 4],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(p.REPO) or output.exists():
        raise FileExistsError("new workspace report output required")
    output.mkdir(parents=True)
    with p.provenance("three_body_local_volume_interpretation", output, name="report") as record:
        hashes = {
            str(path): p.base.sha256(path)
            for path in [
                *p.source_paths(),
                Path(audit.__file__).resolve(),
                Path(__file__).resolve(),
            ]
        }
        seal, consumed = audit.completed(args.run.resolve(), "three_body_local_volume_start_probe")
        audit.merge_hashes(hashes, consumed)
        _, consumed = audit.completed(
            args.audit.resolve(), "three_body_local_volume_output_audit_cpu", "audit"
        )
        audit.merge_hashes(hashes, consumed)
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        summary = json.loads(audit.require_output(args.run / "summary.json", seal).read_text())
        cfg = summary["tracker_config"]
        rows = []
        for saved in summary["frames"]:
            frame = saved["frame"]
            root = args.run / f"frame_{frame}"
            arrays = p._npz(audit.require_output(root / "candidates.npz", seal))
            choice = saved["selected"]["fit_indices"]["mask"]
            indices = np.array([0, choice])
            poses = arrays["fit_pose_xyz"][indices]
            parts = cost_parts(
                poses, arrays["fit_loss_parts"][indices], arrays["predicted_pose_xyz"], cfg
            )
            np.testing.assert_allclose(
                parts.sum(1), arrays["fit_scores"][indices], rtol=1e-12, atol=1e-12
            )
            nominal = int(np.flatnonzero(arrays["mask_fit_index"] == choice)[0])
            origin = arrays["mask_component_slot_sign_roll"][nominal]
            source_slot = int(origin[1])
            source_component = int(origin[0])
            if source_component >= 0:
                component_number = int(arrays["oriented_component_indices"][source_component])
                components = json.loads(
                    audit.require_output(root / "components.json", seal).read_text()
                )
                origin_component = components["components"][component_number]
            else:
                origin_component = None
            center_distances = np.stack(
                [
                    np.linalg.norm(poses[:, a, 1] - poses[:, b, 1], axis=-1)
                    for a, b in [(0, 1), (0, 2), (1, 2)]
                ],
                axis=-1,
            )
            error = saved["selected"]["evaluation_mean_view_loss"]
            rows.append(
                dict(
                    frame=frame,
                    arms=saved["arms"],
                    mask_chosen_fit_index=choice,
                    mask_fit_advantage=float(
                        saved["selected"]["fit_scores"][1] - saved["selected"]["fit_scores"][2]
                    ),
                    random_fit_advantage=float(
                        saved["selected"]["fit_scores"][1] - saved["selected"]["fit_scores"][3]
                    ),
                    disjoint_error_refit=float(error[1]),
                    disjoint_error_mask=float(error[2]),
                    disjoint_error_delta_mask_minus_refit=float(error[2] - error[1]),
                    all_selected_direct_node_counts=saved["selected"]["direct_node_counts"],
                    refit_and_mask_direct_nodes=arrays["fit_observed"][indices],
                    refit_and_mask_weighted_cost_parts=parts,
                    weighted_cost_delta_mask_minus_refit=parts[1] - parts[0],
                    refit_to_mask_node_displacement_mm=np.linalg.norm(poses[1] - poses[0], axis=-1),
                    refit_and_mask_pectoral_pair_distance_mm=center_distances,
                    moved_raw_slot=source_slot,
                    nominal_trial_index=nominal,
                    selected_mask_component_slot_sign_roll=origin,
                    selected_component=origin_component,
                    selected_instance_mass=saved["selected_instance_mass"],
                    preview_path=str((root / "preview.jpg").resolve()),
                    preview_sha256=p.base.sha256(root / "preview.jpg"),
                )
            )
        result = dict(
            frames=rows,
            same_native_probes=list(p.FRAMES),
            nominal_attempts_per_arm=sum(row["arms"]["mask"]["nominal"] for row in rows),
            fitted_mask_including_shared_bases=sum(row["arms"]["mask"]["fitted"] for row in rows),
            fitted_random_including_shared_bases=sum(
                row["arms"]["random"]["fitted"] for row in rows
            ),
            shared_actual_fit_count=sum(
                row["joint_admitted_fit_count"] for row in summary["frames"]
            ),
            shared_fit_seconds=sum(row["fit_seconds"] for row in summary["frames"]),
            producer_seal_started=seal["started_at"],
            producer_seal_finished=seal["finished_at"],
            mask_advantage_frames=[
                row["frame"] for row in rows if row["mask_fit_advantage"] > 1e-12
            ],
            random_advantage_frames=[
                row["frame"] for row in rows if row["random_fit_advantage"] > 1e-12
            ],
            weighted_cost_columns=[
                "node",
                "shape",
                "pectoral_motion",
                "orientation",
                "repulsion",
                "silhouette",
            ],
            biological_id_truth_used=False,
            continuous_tracking_improvement_tested=False,
            prior_reviews_cleared=False,
        )
        p.save_json(record, output, "summary", "summary.json", result)
        lines = [
            "# Local body candidates: evidence and limits",
            "",
            "All thirteen registered frames completed. Mask starts lowered the frozen fitting "
            "objective beyond the refitted original at 79322 and 82719; random starts did so at none.",
            "There were 920 admitted mask trials and 477 random trials, including thirteen shared "
            "bases in each arm: 1,384 actual fits. The nominal budgets match; admitted computation "
            "does not. Local fitting took 19.12 s in total; the sealed invocation took 91.06 s.",
            "",
            "| Frame | Mask fit-score advantage | Disjoint refit error | Disjoint mask error |",
            "|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['frame']} | {row['mask_fit_advantage']:.6f} | "
                f"{row['disjoint_error_refit']:.6f} | {row['disjoint_error_mask']:.6f} |"
            )
        lines += [
            "",
            "Visual inspection of the saved native-image overlays:",
            "",
            "- **79322:** the mask winner moves raw S0 towards the same top D2 row 0 already "
            "explained by S1. This improves the silhouette while increasing duplicate explanation. "
            "On 2,061 row-0 pixels, S0 alpha mass rises from 0.033 to 807.10 and S0/S1 shared mass "
            "from 0.0032 to 134.50. All S0 nodes remain directly unsupported. The top image includes "
            "another visible fish-shaped region without a D2 outline there; this top-instance "
            "proposal arm cannot target a detector-missed region. This is a geometric failure "
            "example, not evidence of correct three-fish reconstruction.",
            "- **82719:** the score-winning candidate moves raw S1. Top-camera disjoint error "
            "improves, but side1 foreground error rises from 0.28259 to 0.48794. Overall disjoint "
            "error worsens from 0.34355 to 0.37978. One extra node passes the geometric source "
            "gate (three to four direct nodes overall); that alone does not establish its "
            "anatomical or biological identity.",
            "- **79372:** both arms choose the common refitted original. Its overlay still fails "
            "to follow the U-shaped top row-0 region while raw S0 and S1 overlap near top row 2. "
            "Generating a candidate somewhere in the volume is not sufficient to make this "
            "fixed objective select a physically complete scene.",
            "",
            "The original A silhouette agrees with the independently executed zero-offset "
            "B renderer within 1e-12; all fitting and evaluation pixel arrays agree exactly. "
            "The separate output audit also verifies every arm map, direct-source exclusivity, "
            "boundary flag and saved evaluation error. These implementation checks do not "
            "supply animal-ID truth or clear any previous review flag.",
            "",
            "The 2-mm grid, top D2 misses, reflections, merged masks, duplicate body "
            "explanation, one-body-at-a-time proposals, fixed two-segment geometry and the "
            "motion/shape/observation score all remain possible limitations. This result cannot "
            "reject image-based 3D proposals generally or establish an improved continuous track.",
        ]
        for frame in [79322, 82719, 79372]:
            row = next(row for row in rows if row["frame"] == frame)
            lines += ["", f"[{frame} native overlay]({Path(row['preview_path']).as_posix()})"]
        report = output / "RESULT.md"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record.output("report", report, sha256=p.base.sha256(report))
        p.assert_hashes({**hashes, **{v["path"]: v["sha256"] for v in record.outputs.values()}})
        record.param("all_input_source_output_sha_unchanged", True)
    return result


if __name__ == "__main__":
    main()
