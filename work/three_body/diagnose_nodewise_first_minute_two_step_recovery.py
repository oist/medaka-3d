"""Fixed-pose old-tail 75053->75054 recovery edge; NumPy only, no fit or tracking."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
INPUT = WORK / "nodewise_first_minute_candidate_reacquisition_v1"
OUTPUT = WORK / "nodewise_first_minute_two_step_recovery_cost_v1"
SEAL = "b518608186f3471e044c36b98c31289e124a909d8d8a658931d6d22cc1f9394a"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
LOWER = np.array([0.0, 0.0, -6.0])
UPPER = np.array([192.0, 192.0, 186.0])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signature(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=sha(a.tobytes()))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def update_state(pose, parent_pose, parent_velocity, parent_case, observed, body_supported):
    """continuous_tracker 232:237 and persistent_node_tracker.updated_ages."""
    velocity = 0.5 * parent_velocity + 0.5 * (pose[:, 1] - parent_pose[:, 1])
    pre_cap = velocity.copy()
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    velocity *= np.minimum(1.0, 2.0 / np.maximum(speed, 1e-12))
    velocity = np.where(body_supported[:, None], velocity, 0.8 * parent_velocity)
    missing = np.where(body_supported, 0, np.asarray(parent_case["state0_missing"]) + 1)
    node_missing = np.where(observed, 0, np.asarray(parent_case["state0_node_missing"]) + 1)
    direct_age = np.where(
        observed[:, 1], np.asarray(parent_case["state0_direct_pectoral_age"]) + 1, 0
    )
    unbounded = pose + velocity[:, None]
    return dict(
        velocity=velocity,
        velocity_before_cap=pre_cap,
        cap_applied=np.linalg.norm(pre_cap, axis=-1) > 2,
        missing=missing,
        node_missing=node_missing,
        direct_pectoral_age=direct_age,
        predicted=np.clip(unbounded, LOWER, UPPER),
        clipped_coordinates=(unbounded < LOWER) | (unbounded > UPPER),
    )


def motion(pose, predicted, missing, node_missing, chest_sources, directions):
    """Same directional parent rule and graph_fit center/orientation reductions."""
    sigma = 0.8 + 0.3 * np.minimum(missing, 20)
    chest_age = node_missing[:, 1]
    parallel_sigma = np.maximum(sigma, 0.8 + 0.3 * np.minimum(chest_age, 20))
    delta = pose[:, 1] - predicted[:, 1]
    isotropic = ((delta / sigma[:, None]) ** 2).sum(-1)
    centers = isotropic.copy()
    enabled = np.zeros(3, bool)
    axes = np.zeros((3, 3))
    for body in range(3):
        views = np.flatnonzero(chest_sources[body] >= 0)
        if len(views) != 1 or chest_age[body] < 4:
            continue
        view = int(views[0])
        axis = directions[view][chest_sources[body, view], 1]
        np.testing.assert_allclose(np.linalg.norm(axis), 1, rtol=0, atol=1e-10)
        axes[body] = axis
        enabled[body] = True
        if parallel_sigma[body] > sigma[body]:
            parallel = np.sum(delta[body] * axis)
            transverse = delta[body] - parallel * axis
            centers[body] = (
                np.sum((transverse / sigma[body]) ** 2) + (parallel / parallel_sigma[body]) ** 2
            )
    relative = pose - pose[:, 1:2]
    previous_relative = predicted - predicted[:, 1:2]
    orientation_by_body = ((relative - previous_relative) ** 2).sum(-1).mean(-1) / 25
    center, orientation = centers.mean(), orientation_by_body.mean()
    return dict(
        center=center,
        center_by_body=centers,
        isotropic_center_by_body=isotropic,
        orientation=orientation,
        orientation_by_body=orientation_by_body,
        weighted_center=0.2 * center,
        weighted_orientation=0.2 * 0.05 * orientation,
        weighted_motion=0.2 * (center + 0.05 * orientation),
        motion_sigma=sigma,
        parallel_sigma=parallel_sigma,
        directional_enabled=enabled,
        directional_axes=axes,
        chest_sources=chest_sources,
        chest_displacement_mm=np.linalg.norm(delta, axis=-1),
    )


def reused_fixed_cost(row):
    node, shape, _, repulsion, silhouette = row["parts"]
    assert row["sixth_term"] == 0
    return node + 0.3 * shape + 0.15 * repulsion + 3.0 * silhouette


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    inputs, decoded = {}, {}

    def read(path, reference):
        raw = path.read_bytes()
        assert sha(raw) == reference["sha256"] and len(raw) == reference["size_bytes"]
        inputs[str(path)] = reference
        return raw

    seal_path = INPUT / "diagnosis.provenance.json"
    raw = seal_path.read_bytes()
    assert sha(raw) == SEAL
    seal = json.loads(raw)
    assert seal["status"] == "ok"
    inputs[str(seal_path)] = dict(sha256=SEAL, size_bytes=len(raw))
    report = json.loads(read(INPUT / "diagnostics.json", seal["outputs"]["diagnostics.json"]))
    cfg = report["config"]
    assert {
        k: cfg[k]
        for k in (
            "motion_sigma_mm",
            "motion_weight",
            "shape_weight",
            "mask_weight",
            "repulsion_weight",
        )
    } == dict(
        motion_sigma_mm=0.8,
        motion_weight=0.2,
        shape_weight=0.3,
        mask_weight=3.0,
        repulsion_weight=0.15,
    )
    cases = {c["frame_idx"]: c for c in report["cases"] if c["arm"] == "tail"}
    c53, c54 = cases[75053], cases[75054]
    rows53, rows54 = c53["candidates"], c54["candidates"]
    assert c53["beam0_candidate_index"] == 7 and c54["beam0_candidate_index"] == 0
    for row in (rows53[7], rows53[10], rows54[0], rows54[11]):
        assert row["parent"] == 0 and row["permutation"] == [0, 1, 2]
    path = INPUT / "candidate_samples.npz"
    poses = {}
    with np.load(io.BytesIO(read(path, seal["outputs"][path.name])), allow_pickle=False) as z:
        for frame in (75053, 75054):
            key = f"tail_{frame}_fitted_pose"
            poses[frame] = z[key]
            sig = signature(poses[frame])
            assert sig == report["old_candidate_member_signatures"][str(frame)]["fitted_pose"]
            decoded.setdefault(str(path), {})[key] = sig

    def original(frame):
        p = WORK / f"tail_one_minute_66400-76360_v1/diagnostics/frame_{frame:06d}/arrays.npz"
        return p, np.load(io.BytesIO(read(p, seal["inputs"][str(p)])), allow_pickle=False)

    def member(z, path, key, frame):
        value = z[key]
        sig = signature(value)
        declared = report["old_candidate_member_signatures"][str(frame)].get(key)
        if declared is not None:
            assert sig == declared
        decoded.setdefault(str(path), {})[key] = sig
        return value

    p53, z53 = original(75053)
    with z53:
        parent_pose = member(z53, p53, "parent_pose", 75053)
        parent_velocity = member(z53, p53, "parent_velocity", 75053)
    assert parent_pose.shape == (6, 3, 3, 3) and parent_pose.dtype == np.float64
    assert parent_velocity.shape == (6, 3, 3) and parent_velocity.dtype == np.float64
    p54, z54 = original(75054)
    with z54:
        predicted = member(z54, p54, "proposal_predicted", 75054)
        generated_sources = member(z54, p54, "generated_geometric_sources", 75054)
        chest_sources = generated_sources[0, :, :, 1]
        directions = {}
        ages = np.asarray(c54["state0_node_missing"])[:, 1]
        for body in range(3):
            views = np.flatnonzero(chest_sources[body] >= 0)
            if len(views) == 1 and ages[body] >= 4:
                view = int(views[0])
                if view not in directions:
                    directions[view] = member(z54, p54, VIEWS[view] + "_ray_directions", 75054)

    # The actual next parent missing counters establish old body support without
    # reopening renderer masks; this counterfactual body's all9 observed needs none.
    old_body_supported = np.asarray(c54["state0_missing"]) == 0
    old_state = update_state(
        poses[75053][7],
        parent_pose[0],
        parent_velocity[0],
        c53,
        np.asarray(rows53[7]["fitted_geometric_observed"]),
        old_body_supported,
    )
    for key in ("missing", "node_missing", "direct_pectoral_age"):
        np.testing.assert_array_equal(old_state[key], c54["state0_" + key])
    np.testing.assert_array_equal(old_state["predicted"], predicted[0])
    np.testing.assert_array_equal(predicted[0], predicted[11])
    actual = {}
    for index in (0, 11):
        m = motion(
            poses[75054][index],
            predicted[index],
            old_state["missing"],
            old_state["node_missing"],
            chest_sources,
            directions,
        )
        local = reused_fixed_cost(rows54[index]) + m["weighted_motion"]
        error = local - rows54[index]["local_cost"]
        np.testing.assert_allclose(m["center"], rows54[index]["parts"][2], rtol=0, atol=1e-10)
        np.testing.assert_allclose(local, rows54[index]["local_cost"], rtol=0, atol=1e-10)
        actual[index] = dict(
            motion=m, local_recomputed=local, stored_local=rows54[index]["local_cost"], error=error
        )

    observed = np.asarray(rows53[10]["fitted_geometric_observed"])
    assert observed.shape == (3, 3) and observed.all()
    assert np.asarray(rows53[10]["fitted_strict_observed"]).all()
    recovery_state = update_state(
        poses[75053][10], parent_pose[0], parent_velocity[0], c53, observed, observed[:, 1]
    )
    assert not recovery_state["node_missing"].any() and not recovery_state["missing"].any()
    # Chest age 0 makes directional_enabled false independently of unsaved future rays.
    new_motion = motion(
        poses[75054][11],
        recovery_state["predicted"],
        recovery_state["missing"],
        recovery_state["node_missing"],
        np.full((3, 3), -1),
        {},
    )
    new_local = reused_fixed_cost(rows54[11]) + new_motion["weighted_motion"]
    reference_sum = rows53[7]["local_cost"] + rows54[0]["local_cost"]
    recovery_sum = rows53[10]["local_cost"] + new_local
    common_parent = rows53[7]["parent_path_cost"]
    assert common_parent == rows53[10]["parent_path_cost"]
    np.testing.assert_allclose(
        common_parent + reference_sum, rows54[0]["path_cost"], rtol=0, atol=1e-10
    )
    result = dict(
        schema=OUTPUT.name,
        status="ok",
        inputs=inputs,
        decoded_members=decoded,
        source_sha256=sha(Path(__file__).read_bytes()),
        config=cfg,
        old54_baseline_reproduction=actual,
        old53_selected7_state=old_state,
        hypothetical53_recovery10_state=recovery_state,
        hypothetical54_candidate11=dict(
            fixed_pose=poses[75054][11],
            reused_parts=rows54[11]["parts"],
            unchanged_nonmotion_cost=reused_fixed_cost(rows54[11]),
            motion=new_motion,
            local_cost=new_local,
            old_parent_local_cost=rows54[11]["local_cost"],
            new_minus_old_motion=new_motion["weighted_motion"]
            - actual[11]["motion"]["weighted_motion"],
        ),
        comparison=dict(
            common_parent_cost=common_parent,
            reference_53_7_local=rows53[7]["local_cost"],
            reference_54_0_local=rows54[0]["local_cost"],
            reference_two_local_sum=reference_sum,
            reference_cumulative=common_parent + reference_sum,
            recovery_53_10_local=rows53[10]["local_cost"],
            recovery_54_11_local=new_local,
            recovery_two_local_sum=recovery_sum,
            recovery_cumulative=common_parent + recovery_sum,
            recovery_minus_reference=recovery_sum - reference_sum,
            first_frame_recovery_penalty=rows53[10]["local_cost"] - rows53[7]["local_cost"],
        ),
        scope=dict(
            same_fixed54_pose_and_plan=True,
            fixed_node_shape_repulsion_D2_cost_reused=True,
            actual_generator_and_optimizer_not_run=True,
            future_retention_not_demonstrated=True,
            no_identity_review_cleared_or_recomputed=True,
            no_ID_or_accuracy_truth=True,
            new_directional_policy_disabled_by_zero_node_age_not_assumed_sources=True,
            hypothetical54_chest_sources_unknown=True,
            old_support_inherited_from_sealed_diagnostic=True,
            prior_source_and_nonmotion_cost_validation_inherited=True,
            original_video_SL_masks_checkpoint39_GPU_reads=0,
        ),
    )
    OUTPUT.mkdir()
    target = OUTPUT / "diagnostics.json"
    target.write_text(
        json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    c = result["comparison"]
    note = f"""# 75053→75054 固定姿勢2段階の費用

旧75053候補10を保持した仮想親から、旧75054候補11の保存済み姿勢へ進む費用は **{recovery_sum:.9f}**。
実際の75053候補7→75054候補0は **{reference_sum:.9f}**、
差は **{recovery_sum - reference_sum:+.9f}** です。


|経路|75053局所費用|75054局所費用|2段階和|
|---|---:|---:|---:|
|実際の7→0|{c["reference_53_7_local"]:.9f}|{c["reference_54_0_local"]:.9f}|{reference_sum:.9f}|
|固定姿勢の10→11|{c["recovery_53_10_local"]:.9f}|{new_local:.9f}|{recovery_sum:.9f}|

53候補10の全9支持によりbody/node欠測年齢は0、連続胸支持回数は{recovery_state["direct_pectoral_age"].tolist()}。
胸速度は旧式の0.5更新と2 mm上限を適用し、54予測を探索boxへclipしました。
54のmotion sigmaは全身体0.8 mmへ戻り、chest age=0のためdirectional緩和は無効です。


54候補11のweighted motionは元の{actual[11]["motion"]["weighted_motion"]:.9f}から
{new_motion["weighted_motion"]:.9f}へ変化。
node/shape/repulsion/D2画像項と候補姿勢・計画は保存値を使用しています。
旧54候補0/11の局所費用を元の予測・directional規則で再現した最大誤差は
{max(abs(v["error"]) for v in actual.values()):.3g}(許容1e-10)、
旧53選択の状態更新から旧54予測は全要素一致しました。


これは固定pose edgeの反実仮想で、変更後の候補生成・最適化・beam保持・未来終端選択を
実行した結果ではありません。
物理ID、正解姿勢、軌跡改善を認定しません。
reviewは消去せず、本診断ではambiguity/reviewや支持を再判定していません。
費用・支持の元データ妥当性は既存sealの範囲を継承します。


[詳細](diagnostics.json)。
元入力は候補診断seal `{SEAL}`。
実際に追加decodeした各memberと署名を詳細JSONへ記録しました。

"""
    (OUTPUT / "RESULT.md").write_text(note, encoding="utf-8")
    print(
        json.dumps(
            jsonable(
                dict(
                    comparison=c,
                    baseline_max_error=max(abs(v["error"]) for v in actual.values()),
                    old_directional_enabled=actual[11]["motion"]["directional_enabled"],
                    new_motion=new_motion,
                    decoded={str(Path(p).relative_to(WORK)): list(v) for p, v in decoded.items()},
                    output=str(target),
                    sha256=sha(target.read_bytes()),
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
