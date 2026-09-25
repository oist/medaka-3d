"""Independent saved-output arithmetic and support-change localization; CPU only."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK / "observation_reserve_ten_second_comparison_v1"
OUT = WORK / "observation_reserve_ten_second_output_qa_v1"
SEAL_SHA = "41edf2644ac37ba1cac0bb2672545e76d4e1f5a42cde5597d5ea3e2f4ce88347"
TRACKING_SHA = "6ad6ef2dd90d7d693c38331b01a471b0dd43637d9b880813e004e34bece7d2fe"
COMPARISON_SOURCE_SHA = "761c7ecf33f7134d19ae8d1406ae78e6021342039c11b3b0544688f9ea48b8e8"
FIRST, START, STOP = 49800, 74700, 76360
ARMS = ("nodewise", "observation_reserve")
FIELDS = ("frame_idx", "pose_xyz", "observed")
NODES = ("mouth", "chest", "tail")
TOL = 1e-12


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signature(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=sha(a.tobytes()))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def check(actual, expected):
    if isinstance(actual, dict):
        for k, value in actual.items():
            check(value, expected[k])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            check(a, b)
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=TOL)
    else:
        assert actual == expected, (actual, expected)


def runs(flags):
    edges = np.diff(np.r_[0, np.asarray(flags, dtype=np.int8), 0])
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]


def metric(data):
    observed, pose = data["observed"], data["pose_xyz"]
    delta = pose[1:] - pose[:-1]
    steps = np.sqrt(np.einsum("tbnx,tbnx->tbn", delta, delta))
    both = np.logical_and(observed[:-1], observed[1:])
    longest = [
        [max((b - a for a, b in runs(~observed[:, s, n])), default=0) for n in range(3)]
        for s in range(3)
    ]
    direct_max = [
        [
            float(steps[:, s, n][both[:, s, n]].max()) if both[:, s, n].any() else None
            for n in range(3)
        ]
        for s in range(3)
    ]
    return dict(
        interval=[START, STOP],
        frames=STOP - START,
        adjacent_pairs_per_slot_node=STOP - START - 1,
        directly_observed_by_slot_node=np.count_nonzero(observed, axis=0).tolist(),
        missing_by_slot_node=np.count_nonzero(~observed, axis=0).tolist(),
        longest_missing_run_by_slot_node=longest,
        all_three_chests_direct_frames=int(np.count_nonzero(observed[:, :, 1].all(1))),
        directly_supported_adjacent_pairs_by_slot_node=np.count_nonzero(both, axis=0).tolist(),
        direct_max_step_by_slot_node_mm=direct_max,
        all_estimated_max_step_by_slot_node_mm=steps.max(0).tolist(),
        direct_adjacent_steps_over_10mm=[
            dict(
                previous_frame=START + int(t),
                frame_idx=START + int(t) + 1,
                raw_slot=int(s),
                node=int(n),
                distance_mm=float(steps[t, s, n]),
            )
            for t, s, n in np.argwhere(both & (steps > 10))
        ],
        nonfinite_pose_count=int(np.count_nonzero(~np.isfinite(pose))),
    )


def geometries(a, b):
    permutations = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
    chest, whole, counts = [], [], []
    support_masks = []
    for p in permutations:
        d = a["pose_xyz"] - np.take(b["pose_xyz"], p, axis=1)
        squared = np.einsum("tbnx,tbnx->tbn", d, d)
        chest.append(np.sqrt(squared[:, :, 1].sum(1) / 3))
        whole.append(np.sqrt(squared.sum((1, 2)) / 9))
        mask = a["observed"] & np.take(b["observed"], p, axis=1)
        support_masks.append(mask)
        counts.append(mask.sum((1, 2)))
    arrays = dict(
        frame_idx=a["frame_idx"],
        permutations=permutations,
        chest_rms_mm=np.column_stack(chest),
        all_node_rms_mm=np.column_stack(whole),
        both_direct_node_counts_by_permutation=np.column_stack(counts),
    )
    summary = dict(
        permutation_maps_A_raw_slot_to_B_raw_slot=True,
        framewise_geometry_is_not_temporal_ID_alignment=True,
    )
    masks = np.stack(support_masks, axis=1)
    for name, value in (("chest", arrays["chest_rms_mm"]), ("all_node", arrays["all_node_rms_mm"])):
        minimum = value.min(1)
        tied = np.isclose(value, minimum[:, None], rtol=0, atol=TOL)
        unique, winner = tied.sum(1) == 1, value.argmin(1)
        arrays[name + "_optimal_permutation_mask"] = tied
        summary[name] = dict(
            raw_identity_rms_median_mm=float(np.median(value[:, 0])),
            raw_identity_rms_maximum_mm=float(value[:, 0].max()),
            minimum_rms_median_mm=float(np.median(minimum)),
            minimum_rms_maximum_mm=float(minimum.max()),
            unique_nonidentity_frames=int(np.count_nonzero(unique & ~tied[:, 0])),
            tied_frames=int(np.count_nonzero(~unique)),
            unique_winner_counts=[int(np.count_nonzero(unique & (winner == i))) for i in range(6)],
            both_direct_node_pairs_on_unique_frames=int(
                masks[np.arange(len(value)), winner][unique].sum()
            ),
        )
    return arrays, summary


def support_changes(a, b):
    lost, gained = a["observed"] & ~b["observed"], ~a["observed"] & b["observed"]
    distances = np.linalg.norm(b["pose_xyz"] - a["pose_xyz"], axis=-1)
    reports, rankings = {}, {}
    for label, flags in (("newly_lost", lost), ("newly_supported", gained)):
        per_node = []
        for slot, node in itertools.product(range(3), repeat=2):
            segments = [
                dict(start_frame=START + x, stop_frame_exclusive=START + y, length=y - x)
                for x, y in runs(flags[:, slot, node])
            ]
            per_node.append(
                dict(
                    raw_slot=slot,
                    node=node,
                    node_name=NODES[node],
                    count=int(flags[:, slot, node].sum()),
                    native_frames=(START + np.flatnonzero(flags[:, slot, node])).tolist(),
                    contiguous_runs=segments,
                )
            )
        episodes = []
        for left, right in runs(flags.any((1, 2))):
            masked = np.where(flags[left:right], distances[left:right], -np.inf)
            peak = np.unravel_index(np.argmax(masked), masked.shape)
            t, slot, node = int(peak[0] + left), int(peak[1]), int(peak[2])
            episodes.append(
                dict(
                    start_frame=START + left,
                    stop_frame_exclusive=START + right,
                    duration_frames=right - left,
                    changed_node_frames=int(flags[left:right].sum()),
                    counts_by_slot_node=flags[left:right].sum(0).tolist(),
                    representative_frame=START + t,
                    peak_raw_slot=slot,
                    peak_node=node,
                    maximum_affected_raw_node_difference_mm=float(masked[peak]),
                    nodewise_support=a["observed"][t].tolist(),
                    reserve_support=b["observed"][t].tolist(),
                    all_node_distances_mm=distances[t].tolist(),
                )
            )
        episodes.sort(
            key=lambda r: (
                -r["changed_node_frames"],
                -r["maximum_affected_raw_node_difference_mm"],
                r["start_frame"],
            )
        )
        for i, row in enumerate(episodes):
            row["rank"] = i + 1
        reports[label] = dict(node_frames=int(flags.sum()), per_slot_node=per_node)
        rankings[label] = episodes
    transitions = {}
    for arm, data in zip(ARMS, (a, b), strict=True):
        obs = data["observed"]
        transitions[arm] = [
            dict(
                raw_slot=s,
                node=n,
                lost_native_frames=(
                    START + 1 + np.flatnonzero(obs[:-1, s, n] & ~obs[1:, s, n])
                ).tolist(),
                regained_native_frames=(
                    START + 1 + np.flatnonzero(~obs[:-1, s, n] & obs[1:, s, n])
                ).tolist(),
            )
            for s, n in itertools.product(range(3), repeat=2)
        ]
    selected = []
    for category in ("newly_lost", "newly_supported"):
        if rankings[category]:
            row = rankings[category][0]
            selected.append(
                dict(
                    category=category,
                    **row,
                    rationale=(
                        "largest changed node-frame run across the full window; "
                        "ties by affected node position difference then native frame"
                    ),
                )
            )
    samples = set()
    for selected_row in selected:
        for frame in (
            selected_row["start_frame"],
            selected_row["stop_frame_exclusive"] - 1,
            selected_row["representative_frame"],
        ):
            samples.update(f for f in (frame - 1, frame, frame + 1) if START <= f < STOP)
    return dict(
        raw_slot_support_comparison=reports,
        temporal_support_transitions=transitions,
        ranked_runs=rankings,
        proposed_visual_scenes=selected,
        ranking_uses_support_disagreement_not_accuracy=True,
        framewise_permutations_do_not_supply_physical_ID_truth=True,
        selected_samples=[
            dict(
                frame_idx=f,
                arms={
                    arm: dict(
                        pose_xyz=data["pose_xyz"][f - START].tolist(),
                        observed=data["observed"][f - START].tolist(),
                    )
                    for arm, data in zip(ARMS, (a, b), strict=True)
                },
            )
            for f in sorted(samples)
        ],
    )


def main():
    assert not OUT.exists(), "fresh QA output required"
    raw_seal = (ROOT / "comparison.provenance.json").read_bytes()
    assert sha(raw_seal) == SEAL_SHA
    seal = json.loads(raw_seal)
    assert seal["status"] == "ok" and seal["stage"] == ROOT.name
    actual_members = {p.name for p in ROOT.iterdir() if p.is_file()}
    assert actual_members == set(seal["outputs"]) | {"comparison.provenance.json"}
    captured, output_checks = {}, {}
    for name, ref in seal["outputs"].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path == Path(ref["path"]).resolve()
        data = path.read_bytes()
        assert len(data) == ref["size_bytes"] and sha(data) == ref["sha256"]
        captured[name] = data
        output_checks[name] = ref
    plan, summary = (json.loads(captured[k]) for k in ("plan.json", "summary.json"))
    check(plan, summary)
    assert summary["process_interval"] == [START, STOP] and summary["native_frames"] == STOP - START
    assert summary["tracking_sha256"] == TRACKING_SHA
    source_path = WORK / "compare_observation_reserve_continuation.py"
    assert summary["source_sha256"][str(source_path)] == COMPARISON_SOURCE_SHA
    assert sha(source_path.read_bytes()) == COMPARISON_SOURCE_SHA
    data, inputs = {}, {}
    for arm in ARMS:
        ref = plan["candidate_references"][arm]
        descriptor = ref["candidate"]
        path = Path(descriptor["path"]).resolve()
        assert path.is_relative_to(WORK)
        assert seal["inputs"][str(path)] == descriptor["sha256"]
        capture = path.read_bytes()  # Exactly one disk capture per candidate archive.
        assert len(capture) == descriptor["size_bytes"] and sha(capture) == descriptor["sha256"]
        with np.load(io.BytesIO(capture), allow_pickle=False) as archive:
            assert len(archive.files) == 21 and set(archive.files) == set(ref["signatures"])
            full = {field: archive[field] for field in FIELDS}
        signatures = {k: signature(v) for k, v in full.items()}
        assert signatures == {k: ref["signatures"][k] for k in FIELDS}
        assert signatures == {k: plan["selected_field_signatures"][arm][k] for k in FIELDS}
        np.testing.assert_array_equal(full["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64))
        assert full["frame_idx"].dtype == np.int64
        assert full["pose_xyz"].dtype == np.float64 and full["pose_xyz"].shape == (
            STOP - FIRST,
            3,
            3,
            3,
        )
        assert full["observed"].dtype == bool and full["observed"].shape == (STOP - FIRST, 3, 3)
        assert np.isfinite(full["pose_xyz"]).all()
        data[arm] = {k: v[START - FIRST :].copy() for k, v in full.items()}
        inputs[arm] = dict(
            **descriptor,
            decoded_fields=list(FIELDS),
            signatures=signatures,
            undeclared_numeric_validation_of_other18=False,
        )
        del capture, full
    with np.load(io.BytesIO(captured["geometric_permutations.npz"]), allow_pickle=False) as archive:
        geo_declared = json.loads(captured["geometric_permutation_signatures.json"])
        assert len(archive.files) == len(geo_declared) and set(archive.files) == set(geo_declared)
        saved_geo = {k: archive[k] for k in archive.files}
    assert {k: signature(v) for k, v in saved_geo.items()} == geo_declared
    computed = {arm: metric(data[arm]) for arm in ARMS}
    for arm in ARMS:
        check(computed[arm], summary["methods"][arm])
    arrays, geometric = geometries(*(data[arm] for arm in ARMS))
    assert set(arrays) == set(saved_geo)
    max_errors = {}
    for key, value in arrays.items():
        if value.dtype.kind == "f":
            np.testing.assert_allclose(value, saved_geo[key], rtol=0, atol=TOL)
            max_errors[key] = float(np.max(np.abs(value - saved_geo[key])))
        else:
            np.testing.assert_array_equal(value, saved_geo[key])
    check(geometric, summary["geometric_comparison"])
    changes = support_changes(*(data[arm] for arm in ARMS))
    total_missing = [sum(map(sum, computed[arm]["missing_by_slot_node"])) for arm in ARMS]
    change_counts = changes["raw_slot_support_comparison"]
    assert total_missing[1] - total_missing[0] == (
        change_counts["newly_lost"]["node_frames"] - change_counts["newly_supported"]["node_frames"]
    )
    report = dict(
        status="ok",
        schema="observation_reserve_independent_output_qa_v1",
        comparison_seal_sha256=SEAL_SHA,
        tracking_seal_sha256=TRACKING_SHA,
        source_sha256=sha(Path(__file__).read_bytes()),
        output_files_verified=output_checks,
        inputs=inputs,
        process_interval=[START, STOP],
        frames=STOP - START,
        adjacent_pairs_per_node=STOP - START - 1,
        node_count=9,
        independent_methods=computed,
        total_missing=total_missing,
        independent_geometric_comparison=geometric,
        maximum_arithmetic_errors_mm=max_errors,
        arithmetic_tolerance_mm=TOL,
        **changes,
        scope=dict(
            candidate_archives_captured_once_per_arm=True,
            candidate_fields_decoded_once=list(FIELDS),
            comparison_geo_decoded_once=True,
            comparison_review_samples_hash_verified_not_decoded=True,
            support_thresholds_not_rescored=True,
            source_review_kind_selected_prefix_checks_inherited_not_independently_reproved=True,
            reserve_selection_counts_and_final_ancestor_attribution_not_recomputed=True,
            no_images_SL_masks_rig_original39_fit_solver_or_GPU=True,
            review_scenes_are_development_examples_not_ground_truth=True,
        ),
    )
    OUT.mkdir()
    result_path = OUT / "diagnostics.json"
    result_path.write_bytes(json_bytes(report))
    lines = [
        "# Observation reserve 10秒: 独立数値QAと支持変化\n",
        f"全{STOP - START} frame・9 node・各node {STOP - START - 1}隣接対と"
        "全6身体Pを独立NumPyで検算し一致した。",
        f"総欠測は{total_missing[0]}→{total_missing[1]}。"
        f"新規支持喪失{change_counts['newly_lost']['node_frames']} node-frame、"
        f"新規支持獲得{change_counts['newly_supported']['node_frames']} node-frame。",
        "支持の増減は保存directフラグの比較であり、身体・物理IDの正解率ではない。\n",
        "| 方式 | 全3胸direct | directの10mm超step |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        row = computed[arm]
        lines.append(
            f"| {arm} | {row['all_three_chests_direct_frames']} | "
            f"{len(row['direct_adjacent_steps_over_10mm'])} |"
        )
    lines.append(
        "\n追加原画候補は全窓の連続runを変更node-frame数、影響nodeの位置差、native順で順位付けした最上位各1場面。"
    )
    for scene in changes["proposed_visual_scenes"]:
        lines.append(
            f"- {scene['category']}: [{scene['start_frame']},{scene['stop_frame_exclusive']})、"
            f"{scene['changed_node_frames']} node-frame。代表{scene['representative_frame']}、"
            f"raw S{scene['peak_raw_slot']} {NODES[scene['peak_node']]}の位置差"
            f"{scene['maximum_affected_raw_node_difference_mm']:.6f} mm。"
        )
    lines.extend(
        [
            "\n各slot/nodeの全変化frame・連続run、各方式内の支持消失/再獲得時点、"
            "候補場面の前後姿勢・支持は diagnostics.json に保存。",
            "元candidateは各1capture、frame_idx/pose_xyz/observedの3配列だけ各1回復元した。"
            "比較6出力は署名・size・membership確認、geo配列は再計算した。"
            "source/review/kind・前74700選択prefix・reserve最終祖先は継承/対象外。"
            "画像・SL・mask・校正・39状態・fit・GPUを読んでいない。",
            f"\n比較seal `{SEAL_SHA}`。このJSON SHA `{sha(result_path.read_bytes())}`。",
        ]
    )
    (OUT / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {
        p.name: dict(path=str(p), sha256=sha(p.read_bytes()), size_bytes=p.stat().st_size)
        for p in OUT.iterdir()
        if p.is_file()
    }
    (OUT / "qa.provenance.json").write_bytes(
        json_bytes(
            dict(
                status="ok",
                source=str(Path(__file__).resolve()),
                source_sha256=report["source_sha256"],
                comparison_seal_sha256=SEAL_SHA,
                outputs=outputs,
            )
        )
    )
    print(
        json.dumps(
            dict(
                status="ok",
                total_missing=total_missing,
                lost=change_counts["newly_lost"]["node_frames"],
                gained=change_counts["newly_supported"]["node_frames"],
                scenes=changes["proposed_visual_scenes"],
                json_sha256=sha(result_path.read_bytes()),
                seal_sha256=sha((OUT / "qa.provenance.json").read_bytes()),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
