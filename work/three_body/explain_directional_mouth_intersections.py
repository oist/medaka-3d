"""All 202 saved-input mouth intersections and an explicit old-point counterfactual.

No threshold change, fit, adopted pose, raw video, SL search or GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
import recompute_directional_mouth_generation as generated
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig, _injective

from medaka3d.calib import load_rig, triangulate
from medaka3d.provenance import provenance

prior, checks = generated.prior, generated.checks
ROOT = prior.REPO / "work/three_body/directional_mouth_generation_lost33_v1"
SHA = "9a6cbd189e90c162649cacf934442754c2723d9b6571e2c8d9ea8b9873dce638"
STAGE = "directional_mouth_202_intersection_diagnosis_v1"


def facts(xyz, error, origins, directions, chest, length, cfg):
    lower, upper = np.asarray(cfg.lower_mm), np.asarray(cfg.upper_mm)
    forward = ((xyz - origins) * directions).sum(-1)
    low, high = np.maximum(lower - xyz, 0), np.maximum(xyz - upper, 0)
    distance = float(np.linalg.norm(xyz - chest))
    ratio = distance / length
    invalid = not np.isfinite(xyz).all() or not np.isfinite(error) or np.any(forward < 0)
    stage = (
        "nonfinite_or_behind"
        if invalid
        else "ray_rms"
        if error > cfg.ray_rms_mm
        else "outside_box"
        if np.any(low > 0) or np.any(high > 0)
        else "anatomy"
        if not cfg.minimum_length_ratio <= ratio <= cfg.maximum_length_ratio
        else "reaches_strict_support"
    )
    return dict(
        xyz=xyz,
        ray_rms_mm=float(error),
        forward_distance_along_each_ray_mm=forward,
        box_lower_excess_xyz_mm=low,
        box_upper_excess_xyz_mm=high,
        box_inside=bool(np.all(low == 0) and np.all(high == 0)),
        mouth_chest_length_mm=distance,
        initial_mouth_chest_length_mm=float(length),
        allowed_mouth_chest_length_mm=[
            cfg.minimum_length_ratio * length,
            cfg.maximum_length_ratio * length,
        ],
        length_ratio=float(ratio),
        length_allowed=bool(cfg.minimum_length_ratio <= ratio <= cfg.maximum_length_ratio),
        first_gate=stage,
    )


def numeric(values):
    return dict(
        minimum=float(np.min(values)),
        median=float(np.median(values)),
        maximum=float(np.max(values)),
    )


def run(output):
    output = Path(output).resolve()
    checks.require(
        output.is_relative_to(prior.REPO / "work") and not output.exists(), "new output required"
    )
    codes = generated.comparison.source_registry.local_code()
    hashes = dict(codes)
    prior.bind(ROOT / "generation.provenance.json", SHA, hashes)
    seal = checks.verify(ROOT, "generation.provenance.json", generated.STAGE, hashes)
    reference = checks.read(ROOT / "summary.json")
    checks.require(
        reference["native_interval"] == [generated.START, generated.STOP]
        and reference["S0_returned_by_original_helper"] == 0,
        "wrong fixed generator result",
    )
    rows = [json.loads(line) for line in (ROOT / "frames.jsonl").read_text().splitlines()]
    checks.require(
        [row["frame_idx"] for row in rows] == list(range(generated.START, generated.STOP)),
        "all fixed33 required",
    )
    checkpoint = prior.BRANCH / "checkpoints/through_068060"
    for filename in ("manifest.json", "metadata.json"):
        prior.audit_input(seal, checkpoint / filename, hashes)
    context = checks.read(checkpoint / "metadata.json")["context"]
    cfg = NodeRecoveryConfig(**context["recovery_config"])
    binding = checks.read(checkpoint / "manifest.json")["bindings"]["original_binding"]
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    for path, digest in shared.items():
        if Path(path).suffix == ".json":
            prior.bind(path, digest, hashes)
    manifest = next(Path(path) for path in shared if Path(path).name == "calibration_manifest.json")
    rig = load_rig(manifest.parent)
    normalize = generated.comparison.support_audit.COMMON.normalize
    checks.require(
        normalize([dataclasses.asdict(c) for c in rig.cameras]) == context["cameras"],
        "rig context differs",
    )
    prior.bind(
        gates_root := generated.GATE_ROOT / "gate.provenance.json", generated.GATE_SHA, hashes
    )
    # The original local trace was verified by the gate and generator provenance chain.
    gate_seal = checks.read(gates_root)
    old_path = generated.gates.ROOT / "original_same_horizon.npz"
    prior.audit_input(gate_seal, old_path, hashes)
    with np.load(old_path, allow_pickle=False) as archive:
        take = (archive["frame_idx"] >= generated.START) & (archive["frame_idx"] < generated.STOP)
        old_pose = archive["pose_xyz"][take, 0].copy()
        old_source = archive["sources"][take, 0, :, 0].copy()
        checks.require(
            len(old_pose) == 33 and archive["observed"][take, 0, 0].all(),
            "old reference mouth support differs",
        )
    all_points, old_points, exact_old = [], [], []
    for ti, row in enumerate(rows):
        frame = row["frame_idx"]
        with np.load(ROOT / f"f{frame}_generation.npz", allow_pickle=False) as stored:
            d = {key: stored[key].copy() for key in stored.files}
        prediction, plan = d["predicted_pose"], d["predicted_plan"]
        rays = RayObservations(
            {v: d[v + "_points"] for v in prior.VIEWS},
            tuple(d[v + "_ray_origins"] for v in prior.VIEWS),
            tuple(d[v + "_ray_directions"] for v in prior.VIEWS),
        )
        free = []
        for vi, view in enumerate(prior.VIEWS):
            finite = np.isfinite(rays.points[view][:, 0]).all(-1)
            finite &= np.isfinite(rays.origins[vi][:, 0]).all(-1)
            finite &= np.isfinite(rays.directions[vi][:, 0]).all(-1)
            free.append(
                np.flatnonzero(finite & ~np.isin(np.arange(len(finite)), plan[1:, vi, 0])).tolist()
            )
        info = row["original_helper_diagnostics"]["fish"][0]
        checks.require(list(map(len, free)) == info["free_mouth_rows_per_view"], "free rows differ")
        frame_points = []
        for count in (3, 2):
            for views in combinations(range(3), count):
                choices = list(product(*(free[v] for v in views)))
                if not choices:
                    continue
                origins = np.array(
                    [
                        [rays.origins[v][r, 0] for v, r in zip(views, choice, strict=True)]
                        for choice in choices
                    ]
                )
                directions = np.array(
                    [
                        [rays.directions[v][r, 0] for v, r in zip(views, choice, strict=True)]
                        for choice in choices
                    ]
                )
                xyz, rms = triangulate(origins, directions)
                projector = np.eye(3) - directions[..., :, None] * directions[..., None, :]
                normal = projector.sum(axis=-3)
                condition = np.linalg.cond(normal)
                determinant = np.linalg.det(normal)
                for i, choice in enumerate(choices):
                    source = np.full(3, -1, np.int32)
                    source[list(views)] = choice
                    f = facts(
                        xyz[i],
                        rms[i],
                        origins[i],
                        directions[i],
                        prediction[0, 1],
                        d["initial_lengths"][0, 0],
                        cfg,
                    )
                    strict = None
                    if f["first_gate"] == "reaches_strict_support":
                        trial, trial_plan = prediction.copy(), plan.copy()
                        trial[0, 0], trial_plan[0, :, 0] = xyz[i], source
                        checks.require(_injective(trial_plan), "intersection stole protected row")
                        _, held, _ = supported_nodes(trial, trial_plan, rays, rig)
                        strict = bool(held[0, 0] and held[0, 1])
                        f["first_gate"] = "accepted" if strict else "strict_support"
                    item = dict(
                        frame_idx=frame,
                        index_in_frame=len(frame_points),
                        views=[prior.VIEWS[v] for v in views],
                        source_rows=source,
                        **f,
                        normal_matrix=normal[i],
                        normal_determinant=float(determinant[i]),
                        normal_condition_2norm=float(condition[i]),
                        strict_support_if_reached=strict,
                        same_full_old_mouth_assignment=bool(np.array_equal(source, old_source[ti])),
                        subset_of_old_mouth_assignment=bool(
                            np.all(source[list(views)] == old_source[ti, list(views)])
                        ),
                        distance_to_saved_old_mouth_mm=float(
                            np.linalg.norm(xyz[i] - old_pose[ti, 0])
                        ),
                        delta_from_saved_old_mouth_xyz_mm=xyz[i] - old_pose[ti, 0],
                        distance_to_current_predicted_mouth_mm=float(
                            np.linalg.norm(xyz[i] - prediction[0, 0])
                        ),
                    )
                    frame_points.append(prior.finite_json(item))
        counts = Counter(item["first_gate"] for item in frame_points)
        expected = {key: counts.get(key, 0) for key in info["rejection_counts"]}
        checks.require(
            len(frame_points) == info["combinations"]
            and expected == info["rejection_counts"]
            and counts.get("accepted", 0) == info["admitted"],
            "original rejection partition differs",
        )
        matches = [item for item in frame_points if item["same_full_old_mouth_assignment"]]
        checks.require(len(matches) == 1, "one exact old assignment intersection required")
        exact_old.extend(matches)
        all_points.extend(frame_points)
        views = np.flatnonzero(old_source[ti] >= 0)
        origin = np.array([rays.origins[v][old_source[ti, v], 0] for v in views])
        direction = np.array([rays.directions[v][old_source[ti, v], 0] for v in views])
        delta = old_pose[ti, 0] - origin
        perpendicular = delta - (delta * direction).sum(-1, keepdims=True) * direction
        old_rms = np.sqrt(np.sum(perpendicular * perpendicular) / len(views))
        f = facts(
            old_pose[ti, 0],
            old_rms,
            origin,
            direction,
            prediction[0, 1],
            d["initial_lengths"][0, 0],
            cfg,
        )
        trial, trial_plan = prediction.copy(), plan.copy()
        trial[0, 0] = old_pose[ti, 0]
        trial_plan[0, :, 0] = old_source[ti]
        checks.require(_injective(trial_plan), "old-point counterfactual has row collision")
        strict_source, held, distance = supported_nodes(trial, trial_plan, rays, rig)
        point_item = dict(
            frame_idx=frame,
            source_rows=old_source[ti],
            **f,
            all_pre_strict_gates_pass=f["first_gate"] == "reaches_strict_support",
            mouth_and_new_predicted_chest_strict=bool(held[0, 0] and held[0, 1]),
            strict_sources=strict_source[0, :, 0],
            strict_float32_ray_mm=distance[0, :, 0],
            current_predicted_chest_xyz=prediction[0, 1],
            saved_old_chest_xyz=old_pose[ti, 1],
            chest_displacement_mm=float(np.linalg.norm(prediction[0, 1] - old_pose[ti, 1])),
            comparison_scope=(
                "Saved old mouth coordinate at new predicted chest; "
                "not the unconstrained intersection, and not adopted."
            ),
        )
        old_points.append(prior.finite_json(point_item))
    checks.require(len(all_points) == 202 and len(old_points) == 33, "fixed202/33 scope differs")
    groups = Counter(item["first_gate"] for item in all_points)
    old_groups = Counter(item["first_gate"] for item in exact_old)
    outside = [x for x in all_points if x["first_gate"] == "outside_box"]
    anatomy = [x for x in all_points if x["first_gate"] == "anatomy"]
    summary = dict(
        schema=STAGE,
        status="ok",
        native_interval=[generated.START, generated.STOP],
        generation_seal_sha256=SHA,
        all202_rejection_counts_match_original_helper=True,
        total_combinations=202,
        first_rejection_counts=dict(groups),
        full_old_assignment_intersection_count=33,
        full_old_assignment_rejection_counts=dict(old_groups),
        original_assignment_intersection_displacement_mm=numeric(
            [x["distance_to_saved_old_mouth_mm"] for x in exact_old]
        ),
        original_assignment_normal_condition=numeric(
            [x["normal_condition_2norm"] for x in exact_old]
        ),
        all_intersection_normal_condition=numeric(
            [x["normal_condition_2norm"] for x in all_points]
        ),
        box_first_rejection_lower_xyz_counts=(
            np.array([x["box_lower_excess_xyz_mm"] for x in outside]) > 0
        )
        .sum(0)
        .tolist(),
        box_first_rejection_upper_xyz_counts=(
            np.array([x["box_upper_excess_xyz_mm"] for x in outside]) > 0
        )
        .sum(0)
        .tolist(),
        box_first_rejection_maximum_lower_xyz_mm=np.array(
            [x["box_lower_excess_xyz_mm"] for x in outside]
        )
        .max(0)
        .tolist(),
        box_first_rejection_maximum_upper_xyz_mm=np.array(
            [x["box_upper_excess_xyz_mm"] for x in outside]
        )
        .max(0)
        .tolist(),
        anatomy_first_rejection_frames=dict(Counter(str(x["frame_idx"]) for x in anatomy)),
        anatomy_first_rejection_length_mm=numeric([x["mouth_chest_length_mm"] for x in anatomy]),
        anatomy_first_rejection_distance_from_old_mouth_mm=numeric(
            [x["distance_to_saved_old_mouth_mm"] for x in anatomy]
        ),
        old_point_counterfactual_first_gates=dict(Counter(x["first_gate"] for x in old_points)),
        old_point_counterfactual_all_pre_strict_and_strict_pass=sum(
            x["all_pre_strict_gates_pass"] and x["mouth_and_new_predicted_chest_strict"]
            for x in old_points
        ),
        old_point_counterfactual_adopted=False,
        box_is_frozen_engineering_bound_not_surveyed_truth=True,
        condition_number_used_as_new_gate=False,
        no_fit=True,
        no_GPU=True,
        no_threshold_change=True,
        geometry_truth_or_ID_inferred=False,
    )
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="intersections") as record:
        for source, digest in codes.items():
            target = output / "code_snapshot" / Path(source).relative_to(prior.REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
            checks.require(checks.sha(target) == digest, "source snapshot changed")
            record.output(str(target.relative_to(output)), target, sha256=digest)
        for name, value in (
            ("summary.json", summary),
            ("full_old_assignment_intersections.json", exact_old),
            ("old_point_new_chest_counterfactuals.json", old_points),
            ("anatomy_rejected_intersections.json", anatomy),
        ):
            (output / name).write_text(
                json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
            )
        (output / "all202.jsonl").write_text(
            "".join(json.dumps(x, allow_nan=False) + "\n" for x in all_points), encoding="utf-8"
        )
        lines = [
            "# 202交差点の棄却と旧口位置の反実仮想",
            "",
            "33フレームの保存済みray/予測pose/planから、元と同じ2/3視点組合せ順でtriangulateを実行。"
            "全202点の先行棄却理由は元helper diagnosticsとフレームごとに完全一致した。",
            "",
            f"全棄却: {dict(groups)}。"
            f"旧方式の実mouth source全組合せに一致する33点: {dict(old_groups)}。",
            "旧source交差点のnormal condition(2-norm)は "
            f"{summary['original_assignment_normal_condition']}。"
            "条件数は診断であり新しい受理条件に使っていない。",
            "",
            "box外132件の違反数 (x,y,z)、下限: "
            f"{summary['box_first_rejection_lower_xyz_counts']}、"
            f"上限: {summary['box_first_rejection_upper_xyz_counts']}。"
            "最大超過mm、下限: "
            f"{summary['box_first_rejection_maximum_lower_xyz_mm']}、"
            f"上限: {summary['box_first_rejection_maximum_upper_xyz_mm']}。",
            "",
            f"解剖長棄却14件のframe内訳: {summary['anatomy_first_rejection_frames']}。"
            f"口胸長mm: {summary['anatomy_first_rejection_length_mm']}。"
            f"旧口からの距離mm: {summary['anatomy_first_rejection_distance_from_old_mouth_mm']}。",
            "",
            "|frame|旧source交差点のfirst gate|交差XYZ|box上限超過XYZ mm|ray RMS mm|口胸長mm|",
            "|---:|---|---|---|---:|---:|",
        ]
        for row in exact_old:
            lines.append(
                f"|{row['frame_idx']}|{row['first_gate']}|{np.round(row['xyz'], 4).tolist()}|"
                f"{np.round(row['box_upper_excess_xyz_mm'], 4).tolist()}|"
                f"{row['ray_rms_mm']:.4f}|{row['mouth_chest_length_mm']:.4f}|"
            )
        lines += [
            "",
            "別の反実仮想として、旧mouthの保存位置を新しい実予測胸と組み合わせた。"
            "交差前段相当の条件と既存2mm/12pxの口・胸支持を満たすのは "
            f"{summary['old_point_counterfactual_all_pre_strict_and_strict_pass']}/33。"
            f"前段相当first gate: {summary['old_point_counterfactual_first_gates']}。",
            "",
            "これは無制約の最小二乗交差点とは別の既存座標を調べる反実仮想であり、オンライン生成できる証明でも、"
            "旧poseを新候補として採用した結果でもない。boxは凍結済み工程上の範囲で、測量された真の水槽境界とは断定しない。"
            "全202点のXYZ・ray RMS・box両側超過・長さ許容区間・normal行列/条件数を保存した。"
            "原画、fit、GPU、閾値変更、ID判定は行っていない。",
            "",
        ]
        (output / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
        for path in output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=checks.sha(path))
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in hashes.items():
            checks.require(checks.sha(path) == digest, "intersection input changed")
        for item in record.outputs.values():
            checks.require(
                checks.sha(item["path"]) == item["sha256"], "intersection output changed"
            )
        record.param("all_opened_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
