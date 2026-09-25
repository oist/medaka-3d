"""Post-fit arithmetic for every new chest candidate; no new objective evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import summarize_chest_candidate_replay as evidence_tools

REPO = Path(__file__).resolve().parents[2]
from medaka3d.provenance import provenance  # noqa: E402

PARTS = ("node", "shape", "chest_motion", "orientation_motion", "repulsion", "silhouette")
require, sha, read = evidence_tools.require, evidence_tools.sha, evidence_tools.read


def main(fit, recovery, output):
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    fit_seal = evidence_tools.verify(
        fit, "fit.provenance.json", "chest_recovery_same_objective_local_CPU_fit", hashes
    )
    evidence_tools.verify(
        recovery, "probe.provenance.json", "chest_recovery_counterfactual_proposals", hashes
    )
    require(
        fit_seal["inputs"][str(recovery / "probe.provenance.json")]["sha256"]
        == hashes[str(recovery / "probe.provenance.json")],
        "different input proposals",
    )
    summary = read(fit / "summary.json")
    frames = read(fit / "frames.json")
    require(
        summary["frames"] == len(frames) == 20 and summary["new_candidates"] == 31,
        "fixed local scope incomplete",
    )
    require(summary["original_cpu_gpu_support_all_exact"], "CPU changed original support decisions")
    rows = []
    maximum_parity = {key: 0.0 for key in ("fitted_pose", "local_cost", "loss_parts")}
    all_numerical_parity = all_mask_parity = True
    for frame in frames:
        native = frame["frame_idx"]
        a_path, r_path = fit / f"frame_{native:06d}.npz", recovery / f"frame_{native:06d}.npz"
        require(str(a_path) in hashes and str(r_path) in hashes, "unsealed frame data")
        with np.load(a_path, allow_pickle=False) as z:
            a = {key: z[key].copy() for key in z.files}
        with np.load(r_path, allow_pickle=False) as z:
            targets = z["target_slot"].copy()
        new = np.flatnonzero(a["new_candidate"])
        require(len(new) == len(targets) == frame["new_candidates"], "candidate indexing differs")
        old_best = int(a["original_selected_indices"][0])
        require(
            old_best == frame["original_cpu_best"] and not a["new_candidate"][old_best],
            "original winner changed",
        )
        for key in maximum_parity:
            values = frame["original_cpu_gpu"][key]
            maximum_parity[key] = max(maximum_parity[key], values["maximum_absolute_difference"])
            all_numerical_parity &= values["within_fixed_atol"]
        all_mask_parity &= frame["original_cpu_gpu"]["mask_supported_exact"]
        for idx, target in zip(new, targets, strict=True):
            target = int(target)
            start = a["initial_pose_xyz"][idx, target, 1]
            end = a["fitted_pose_xyz"][idx, target, 1]
            predicted = a["predicted_pose_xyz"][idx, target, 1]
            toward, movement = predicted - start, end - start
            denominator = np.linalg.norm(toward) * np.linalg.norm(movement)
            projection = None if denominator == 0 else float(np.dot(toward, movement) / denominator)
            delta = a["final_weighted_six_parts"][idx] - a["final_weighted_six_parts"][old_best]
            during = a["final_weighted_six_parts"][idx] - a["initial_weighted_six_parts"][idx]
            gap = float(a["local_cost"][idx] - a["local_cost"][old_best])
            require(np.isclose(delta.sum(), gap, rtol=0, atol=1e-10), "cost decomposition mismatch")
            row = dict(
                frame_idx=native,
                candidate=int(idx),
                target_slot=target,
                old_best_candidate=old_best,
                before_target_chest_views=int(
                    (a["before_strict_sources"][idx, target, :, 1] >= 0).sum()
                ),
                after_target_chest_views=int(
                    (a["after_strict_sources"][idx, target, :, 1] >= 0).sum()
                ),
                before_target_mouth=bool(a["before_strict_observed"][idx, target, 0]),
                after_target_mouth=bool(a["after_strict_observed"][idx, target, 0]),
                before_target_tail=bool(a["before_strict_observed"][idx, target, 2]),
                after_target_tail=bool(a["after_strict_observed"][idx, target, 2]),
                initial_distance_from_prediction_mm=float(np.linalg.norm(toward)),
                fitted_distance_from_prediction_mm=float(np.linalg.norm(end - predicted)),
                movement_toward_prediction_cosine=projection,
                local_cost_gap_to_original_best=gap,
                path_cost_gap_to_original_best=float(
                    a["path_cost"][idx] - a["path_cost"][old_best]
                ),
                selected=bool(np.isin(idx, a["selected_indices"])),
                before_target_mask_support=None,
                after_target_mask_support=bool(a["mask_supported"][idx, target]),
                after_target_ray_fail_count=int(
                    (
                        a["after_assigned_source"][idx, target, :, 1]
                        & ~a["after_ray_gate_pass"][idx, target, :, 1]
                    ).sum()
                ),
                after_target_pixel_fail_count=int(
                    (
                        a["after_assigned_source"][idx, target, :, 1]
                        & ~a["after_pixel_gate_pass"][idx, target, :, 1]
                    ).sum()
                ),
            )
            row.update(
                {"gap_" + key: float(value) for key, value in zip(PARTS, delta, strict=True)}
            )
            row.update(
                {
                    "during_fit_" + key: float(value)
                    for key, value in zip(PARTS, during, strict=True)
                }
            )
            rows.append(row)
    require(len(rows) == 31, "candidate coverage incomplete")

    def describe(key):
        values = np.asarray([r[key] for r in rows], float)
        return dict(
            minimum=float(values.min()),
            median=float(np.median(values)),
            maximum=float(values.max()),
            mean=float(values.mean()),
        )

    result = dict(
        schema="chest_recovery_local_fit_attribution_v1",
        status="ok",
        frames=20,
        new_candidates=31,
        original_cpu_gpu_all_within_registered_atol=bool(all_numerical_parity),
        original_cpu_gpu_maximum_absolute_differences=maximum_parity,
        original_cpu_gpu_all_mask_bridges_exact=bool(all_mask_parity),
        new_target_chest_supported_after=sum(r["after_target_chest_views"] >= 2 for r in rows),
        new_selected=sum(r["selected"] for r in rows),
        candidates_moved_closer_to_prediction=sum(
            r["fitted_distance_from_prediction_mm"] < r["initial_distance_from_prediction_mm"]
            for r in rows
        ),
        distance_before=describe("initial_distance_from_prediction_mm"),
        distance_after=describe("fitted_distance_from_prediction_mm"),
        cost_gap=describe("local_cost_gap_to_original_best"),
        mean_cost_gap_parts={key: describe("gap_" + key)["mean"] for key in PARTS},
        mean_during_fit_parts={key: describe("during_fit_" + key)["mean"] for key in PARTS},
        independent_ID_truth=False,
        image_generalization_evaluated=False,
        interpretation_scope=("Cost contributions describe this unchanged objective, "
                              "not true fish positions."),
    )
    output.mkdir(parents=True, exist_ok=False)
    with provenance("chest_recovery_local_fit_attribution", output, name="analysis") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (output / "candidates.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            "# 新しい胸部候補のfit結果",
            "",
            f"全20フレームの31候補を元の評価式でfitし、胸部支持が残った候補は"
            f"{result['new_target_chest_supported_after']}、選択された候補は{result['new_selected']}だった。",
            "",
            "CPUに移した元候補の支持判定は、保存済みGPU結果と全フレームで一致。",
            "数値の最大差と、候補ごとの口・胸部・尾、mask支持、距離と全評価項は添付に保持した。",
            "",
            "|評価項|元候補の最良fitに対する新候補の平均差|新候補のfit前後の平均変化|",
            "|---|---:|---:|",
        ]
        for key in PARTS:
            lines.append(
                f"|{key}|{result['mean_cost_gap_parts'][key]:+.6f}|"
                f"{result['mean_during_fit_parts'][key]:+.6f}|"
            )
        lines += [
            "",
            "各値は実際の重みを掛けた項で、orientationを含む6項の和が総costと一致する。",
            "31候補は相関する20フレームの候補であり、独立した標本やID正解数ではない。",
            "画像項はfit用画素での結果。全3個体のID維持や画像への汎化は評価していない。",
        ]
        (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=sha(path))
        for path, digest in hashes.items():
            require(sha(path) == digest, "evidence changed during attribution")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    require(target.is_relative_to(REPO / "work") and not target.exists(), "new output required")
    main(args.fit.resolve(), args.recovery.resolve(), target)
