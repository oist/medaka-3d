"""Geometry/cost interpretation of three sealed saved candidate-bank ablations.

Reads only the ablation seal, three small JSON trees and their three result NPZs.
No raw images, masks, SLEAP, calibration, tracking checkpoints or fitter are read.
Differences compare raw-slot hypotheses in a single frame, never animal motion.
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
INPUT = WORK / "mask_only_saved_transition_cost_ablation_v2"
OUTPUT = WORK / "mask_only_saved_transition_geometry_analysis_v1"
SEAL_SHA = "6b8ca822e509d589958136741f70915d9378fba67e88d265e587a361ac37e5c7"
CASES = ((90141, "side1_cam"), (90258, "side2_cam"), (90259, "side2_cam"))
SAMPLES = ["M", "C", "T", "MC25", "MC50", "MC75", "CT25", "CT50", "CT75"]
PAIRS = ((0, 1), (0, 2), (1, 2))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signature(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=sha(array.tobytes()))


def restore(tree, arrays):
    used = set()

    def visit(item):
        if isinstance(item, dict) and "__array__" in item:
            assert set(item) == {"__array__", "signature"}
            key = item["__array__"]
            assert key not in used and key in arrays
            used.add(key)
            value = arrays[key]
            assert not value.dtype.hasobject and signature(value) == item["signature"]
            value.flags.writeable = False
            return value
        if isinstance(item, dict):
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, list):
            return [visit(v) for v in item]
        return item

    value = visit(tree)
    assert used == set(arrays)
    return value


def candidate_detail(bank, score, ablation, index):
    pose = bank["candidate_pose_xyz"][index]
    assignment = score["best_assignment"][index]
    assert np.all((assignment >= 0) & (assignment < 3))
    bodies = []
    for slot, instance in enumerate(assignment):
        distance = score["distances_by_body_sample_instance"][index, slot, :, instance]
        endpoint = (np.maximum(distance[:3] - 6.0, 0.0).max() / 6.0) ** 2
        interior = np.mean((np.maximum(distance[3:] - 2.0, 0.0) / 6.0) ** 2)
        np.testing.assert_allclose(
            endpoint, score["worst_endpoint_costs"][index, slot, instance], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            interior, score["segment_leakage_costs"][index, slot, instance], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            endpoint + interior,
            score["body_instance_costs"][index, slot, instance],
            rtol=0,
            atol=1e-12,
        )
        bodies.append(
            dict(
                raw_slot=slot,
                assigned_local_mask=int(instance),
                sample_to_assigned_mask_distance_px=distance.tolist(),
                endpoint_cost=float(endpoint),
                interior_cost=float(interior),
                body_cost=float(endpoint + interior),
                projected_MCT_uv=score["projected_samples"][index, slot, :3].tolist(),
                nearest_mask_per_endpoint=np.argmin(
                    score["distances_by_body_sample_instance"][index, slot, :3], axis=-1
                ).tolist(),
            )
        )
    np.testing.assert_allclose(
        sum(b["body_cost"] for b in bodies), score["best_cost"][index], rtol=0, atol=1e-12
    )
    return dict(
        candidate_index=int(index),
        kind=bank["candidate_kind"][index],
        parent=int(bank["candidate_parent"][index]),
        permutation=bank["candidate_permutation"][index].tolist(),
        pose_xyz_mm=pose.tolist(),
        pectoral_pair_separation_mm=[
            float(np.linalg.norm(pose[a, 1] - pose[b, 1])) for a, b in PAIRS
        ],
        segment_lengths_MC_CT_mm=np.linalg.norm(np.diff(pose, axis=1), axis=-1).tolist(),
        mouth_tail_chord_mm=np.linalg.norm(pose[:, 2] - pose[:, 0], axis=-1).tolist(),
        assigned_view_counts_MCT=(bank["candidate_plan"][index] >= 0).sum(axis=1).tolist(),
        assigned_view_counts_are_direct_support=False,
        base_local_cost=float(bank["base_local_cost"][index]),
        parent_path_cost=float(bank["parent_path_cost"][index]),
        old_local_cost=float(ablation["old_local_cost"][index]),
        new_local_cost=float(ablation["new_local_cost"][index]),
        old_fixed_parent_path_cost=float(ablation["old_fixed_parent_path_cost"][index]),
        new_fixed_parent_path_cost=float(ablation["new_fixed_parent_path_cost"][index]),
        new_view_raw_scene_cost=float(score["best_cost"][index]),
        old_sixth_term=float(bank["sixth_instance_term"][index]),
        new_sixth_term=float(ablation["new_sixth_instance_term"][index]),
        bodies=bodies,
    )


def analyze(case):
    bank, score, ablation = (
        case[k] for k in ("input_candidate_bank", "new_score", "candidate_cost_ablation")
    )
    assert bank["policy"]["weight"] == 3.0
    assert bank["policy"]["observation_config"] == dict(
        minimum_instance_gap_px=3.0,
        sleap_mask_tolerance_px=8.0,
        endpoint_tolerance_px=6.0,
        interior_tolerance_px=2.0,
        pixel_scale_px=6.0,
        worst_node_weight=1.0,
        segment_leakage_weight=1.0,
        interior_fractions=[0.25, 0.5, 0.75],
        tie_atol=1e-12,
    )
    np.testing.assert_array_equal(
        ablation["new_local_cost"], bank["base_local_cost"] + ablation["new_sixth_instance_term"]
    )
    np.testing.assert_array_equal(
        ablation["new_fixed_parent_path_cost"],
        bank["parent_path_cost"] + ablation["new_local_cost"],
    )
    orders = {}
    for key in (
        "old_local_cost",
        "new_local_cost",
        "old_fixed_parent_path_cost",
        "new_fixed_parent_path_cost",
    ):
        order = np.argsort(ablation[key], kind="stable")
        assert order.tolist() == case["rankings"][key]["stable_candidate_order"]
        orders[key] = order
    old, new = (
        int(orders[k][0]) for k in ("old_fixed_parent_path_cost", "new_fixed_parent_path_cost")
    )
    assert old == orders["old_local_cost"][0] and new == orders["new_local_cost"][0]
    first, second = [candidate_detail(bank, score, ablation, j) for j in (old, new)]
    displacement = bank["candidate_pose_xyz"][new] - bank["candidate_pose_xyz"][old]
    old_advantage = float(
        ablation["old_fixed_parent_path_cost"][new] - ablation["old_fixed_parent_path_cost"][old]
    )
    new_advantage = float(
        ablation["new_fixed_parent_path_cost"][old] - ablation["new_fixed_parent_path_cost"][new]
    )
    instance_advantage = (first["new_sixth_term"] - first["old_sixth_term"]) - (
        second["new_sixth_term"] - second["old_sixth_term"]
    )
    np.testing.assert_allclose(
        new_advantage, instance_advantage - old_advantage, rtol=0, atol=1e-10
    )
    return dict(
        frame=case["frame"],
        view=case["view"],
        candidate_count=len(bank["candidate_index"]),
        old_winner=first,
        new_winner=second,
        old_to_new_node_displacement_xyz_mm=displacement.tolist(),
        old_to_new_node_displacement_norm_mm=np.linalg.norm(displacement, axis=-1).tolist(),
        old_path_advantage_for_old_winner=old_advantage,
        changed_instance_term_advantage_for_new_winner=instance_advantage,
        resulting_new_path_advantage_for_new_winner=new_advantage,
        new_path_runner_up=int(orders["new_fixed_parent_path_cost"][1]),
        new_path_gap_to_runner_up=float(
            np.diff(
                ablation["new_fixed_parent_path_cost"][orders["new_fixed_parent_path_cost"][:2]]
            )[0]
        ),
        same_raw_slot_pair_order=list(PAIRS),
        best_assignment_body_cost_reduction=[
            a["body_cost"] - b["body_cost"]
            for a, b in zip(first["bodies"], second["bodies"], strict=True)
        ],
        all_saved_candidate_pectoral_min_pair_separation_mm=[
            min(np.linalg.norm(p[a, 1] - p[b, 1]) for a, b in PAIRS)
            for p in bank["candidate_pose_xyz"]
        ],
    )


def main():
    assert not OUTPUT.exists(), "fresh output required"
    inputs = {}

    def read(path, expected=None):
        data = path.read_bytes()
        digest = sha(data)
        assert expected is None or digest == expected
        inputs[str(path)] = digest
        return data

    seal = json.loads(read(INPUT / "scoring.provenance.json", SEAL_SHA))
    assert seal["status"] == "ok" and seal["stage"] == "mask_only_saved_transition_cost_ablation_v1"

    def member(name):
        path = (INPUT / name).resolve()
        matches = [v for v in seal["outputs"].values() if Path(v["path"]).resolve() == path]
        assert len(matches) == 1
        data = read(path, matches[0]["sha256"])
        assert len(data) == matches[0]["size_bytes"]
        return data

    results = []
    for frame, view in CASES:
        stem = f"cases/f{frame}_{view}"
        document = json.loads(member(stem + ".json"))
        data = member(stem + ".npz")
        assert sha(data) == document["archive"]["sha256"]
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            assert len(archive.files) == len(set(archive.files))
            arrays = {key: archive[key] for key in archive.files}
        restored = restore(document["tree"], arrays)
        assert restored["frame"] == frame and restored["view"] == view
        results.append(analyze(restored))
    source_sha = sha(Path(__file__).read_bytes())
    report = dict(
        schema="saved_mask_only_candidate_geometry_analysis_v1",
        status="ok",
        input_seal_sha256=SEAL_SHA,
        consumed_saved_result_sha256=inputs,
        source_sha256=source_sha,
        sample_order=SAMPLES,
        records=results,
        scope=(
            "Fixed saved single-frame candidate hypotheses with original parent costs; "
            "raw-slot differences are not actual fish movement."
        ),
        limits=[
            "Mask row assignments are camera/frame local, not anatomical or physical IDs.",
            "Candidate plan counts are assigned views, not strict 2mm/12px supported views.",
            "No trajectory recomputation, mask pixels, image/SL/calibration reads, "
            "fit, GPU or checkpoint decode.",
            "Geometric separation and lower mask cost do not establish "
            "correct anatomical or physical identities.",
        ],
    )
    OUTPUT.mkdir()
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    (OUTPUT / "analysis.json").write_bytes(payload)
    lines = [
        "Saved mask-only ablation: geometry and cost differences",
        "",
        report["scope"],
        "",
        "| Native/view | Old→new winner | Chest gaps 01/02/12 old→new (mm) | "
        "Max node difference (mm) | Old advantage / added-mask reversal / new advantage |",
        "|---|---:|---|---:|---|",
    ]
    for r in results:
        a, b = r["old_winner"], r["new_winner"]

        def gaps(x):
            return "/".join(f"{v:.3f}" for v in x["pectoral_pair_separation_mm"])

        lines.append(
            f"| {r['frame']}/{r['view']} | {a['candidate_index']}→{b['candidate_index']} | "
            f"{gaps(a)}→{gaps(b)} | {np.max(r['old_to_new_node_displacement_norm_mm']):.3f} | "
            f"{r['old_path_advantage_for_old_winner']:.6f} / "
            f"{r['changed_instance_term_advantage_for_new_winner']:.6f} / "
            f"{r['resulting_new_path_advantage_for_new_winner']:.6f} |"
        )
        print(
            json.dumps(
                dict(
                    frame=r["frame"],
                    winners=[a["candidate_index"], b["candidate_index"]],
                    parents=[a["parent"], b["parent"]],
                    node_displacements=r["old_to_new_node_displacement_norm_mm"],
                    chest_gaps=[a["pectoral_pair_separation_mm"], b["pectoral_pair_separation_mm"]],
                    segment_lengths=[a["segment_lengths_MC_CT_mm"], b["segment_lengths_MC_CT_mm"]],
                    cost_advantages=[
                        r["old_path_advantage_for_old_winner"],
                        r["changed_instance_term_advantage_for_new_winner"],
                        r["resulting_new_path_advantage_for_new_winner"],
                    ],
                    body_details=[a["bodies"], b["bodies"]],
                )
            )
        )
    lines += [
        "",
        *report["limits"],
        "",
        "Full per-body endpoint/interior costs, nine mask distances, projected endpoints, "
        "segment lengths and assigned-view counts are preserved in analysis.json.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(dict(output=str(OUTPUT), analysis_sha256=sha(payload), source_sha256=source_sha))
    )


if __name__ == "__main__":
    main()
