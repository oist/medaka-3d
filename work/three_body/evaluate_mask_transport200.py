"""Independently score the completed fixed200 three-arm tracking experiment.

All200 frames, original unused pixels, both evidence methods and all causal t0
anchors are retained. Current SLEAP support is rechecked for all600 arm-frames.
This analysis never changes a trajectory, target policy, review or biological ID.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mask_transport_diagnostics as diagnostics
import mask_transport_evaluation as metrics
import mask_transport_inputs as inputs
import numpy as np
import object_mask_transport as transport
import run_mask_transport200_v2 as producer
import torch
from node_geometry import RayObservations, supported_nodes

base, prepared = inputs.base, inputs.prepared
WORK, REPO = inputs.WORK, inputs.REPO
FIRST, STOP = inputs.FIRST, inputs.STOP
STAGE = "mask_transport200_independent_evaluation_v1"
METRICS_SHA = "4b5aa36ed3660586ca94b275742bb69edbc807da04ebfb8de4d7084c1a5676e7"
DIAGNOSTICS_SHA = "6567710637b4a3b69dd03106cc525a1f7fac1659a7ab54acadef3c98517f3e56"


def load_tree(root, members, name):
    header = base.bound_json(root, members, name + ".json")
    arrays = producer.read_arrays(root, members, name + ".npz")
    require(
        header["array_signatures"]
        == {k: producer.tracking.signature(v) for k, v in arrays.items()},
        "tree signatures differ",
    )
    used = set()
    value = producer.tracking._unpack(header["tree"], arrays, used)
    require(used == set(arrays), "tree arrays missing/unused")
    return value


require = base.require


def validate_run(plan, summary, control):
    require(
        plan["schema"] == summary["schema"] == producer.STAGE
        and summary["status"] == "ok"
        and plan["arms"] == list(metrics.ARMS)
        and summary["native_interval"] == plan["native_interval"] == [FIRST, STOP],
        "completed same200 three-arm experiment required",
    )
    binding = plan["input_binding"]
    require(
        binding["preparation_sha256"] == inputs.PREPARATION_SHA
        and binding["point_sha256"] == producer.POINTS_SHA
        and binding["samples_sha256"] == producer.SAMPLES_SHA
        and binding["implementation_plan_sha256"] == inputs.PLAN_SHA,
        "producer evidence differs",
    )
    require(
        plan["original_context"]["tracker_config"]["width_ratio"] == 0.06,
        "evaluation renderer width differs from original",
    )
    require(
        summary["original_control_passed"] is True
        and control["passed"] is True
        and all(control[k]["byte_exact"] is True for k in ("candidate", "history", "state")),
        "original numerical replay must pass",
    )
    require(
        summary["initialize_calls"] == 3
        and summary["restore_calls"] == 0
        and summary["advance_calls"] == 600
        and summary["maximum_pool"] == 24
        and summary["steps"] == 14
        and len(summary["arms"]) == 3,
        "fixed experiment budget/lifecycle differs",
    )
    require([a["arm"] for a in summary["arms"]] == list(metrics.ARMS), "arm ordering differs")
    for row in summary["arms"]:
        require(
            row["frames"] == 200 and row["initialize_count"] == 1 and row["restore_count"] == 0,
            "all arms must be complete without intermediate reset",
        )


def recheck_support(candidates, offset, frame, rig):
    rays = RayObservations.from_points(rig, frame.detections)
    rows = {}
    for arm in metrics.ARMS:
        candidate = candidates[arm]
        sources, observed, distance = supported_nodes(
            candidate["pose_xyz"][offset], candidate["sources"][offset], rays, rig
        )
        producer.same(sources, candidate["sources"][offset], "current strict source: " + arm)
        producer.same(observed, candidate["observed"][offset], "current strict support: " + arm)
        for vi in range(3):
            for node in range(3):
                selected = sources[:, vi, node]
                used = selected[selected >= 0]
                require(len(used) == len(np.unique(used)), "source row reused by two bodies")
        rows[arm] = dict(sources=sources, observed=observed, ray_distance_mm=distance)
    return rows


def validate_reuse(plan, summary, proof):
    require(
        plan["new_arms_this_process"] == ["lk", "ct_first"]
        and plan["reused_original_source_seal_sha256"] == producer.FAILED_SHA
        and summary["reused_original_source_seal_sha256"] == producer.FAILED_SHA
        and summary["this_process_initialize_calls"] == 2
        and summary["this_process_restore_calls"] == 0
        and summary["new_advance_calls"] == 400
        and summary["reused_original_frames"] == 200,
        "reused versus newly computed tracking scope differs",
    )
    require(
        summary["arms"][0]["computed_in_this_process"] is False
        and summary["arms"][0]["reused_from_seal_sha256"] == producer.FAILED_SHA
        and all(a["computed_in_this_process"] is True for a in summary["arms"][1:]),
        "individual arm origins differ",
    )
    require(
        proof["schema"] == "mask_transport_original_reuse_qualification_v1"
        and proof["status"] == "ok"
        and proof["passed"] is True
        and proof["original_source_seal_sha256"] == producer.FAILED_SHA
        and proof["original_source_sha256"] == producer.FAILED_SOURCE_SHA
        and proof["frames"] == 200
        and proof["actual_fit4_byte_exact_frames"] == 200
        and proof["saved_frame_fit_and_records_validated"] == 200
        and proof["original_files_copied_byte_exact"] == 408
        and proof["initial_preparation_SL_masks_byte_exact_views"] == 3,
        "complete saved Original reuse proof required",
    )
    for key in (
        "candidate_21_byte_exact",
        "history_19_plus_offsets_byte_exact",
        "core_39_byte_exact",
        "core_metadata_unchanged",
        "semantic_state_decode_passed",
        "sidecar_validation_passed",
        "original_context_and_sidecar_bytes_preserved",
    ):
        require(proof[key] is True, "Original reuse proof incomplete: " + key)
    require(
        all(
            proof["validation_" + k + "_calls"] == 0
            for k in ("initialize", "restore", "advance", "fit")
        )
        and proof["GPU_used"] is False,
        "Original reuse validation recomputed tracking",
    )


def summary_values(score_rows, support, anchors):
    union = np.stack([r["union_loss"] for r in score_rows])
    per_view = np.stack([r["union_per_view"] for r in score_rows])
    owners = np.stack([r["owner_loss"] for r in score_rows])
    masses = np.stack([r["target_mass"] for r in score_rows])
    comparison = {}
    for index, arm in enumerate(metrics.ARMS):
        delta = union[:, index] - union[:, 0]
        comparison[arm] = dict(
            mean=float(union[:, index].mean()),
            per_view_mean=per_view[:, index].mean(axis=0).tolist(),
            delta_mean=float(delta.mean()),
            delta_median=float(np.median(delta)),
            better_native_frames=int((delta < 0).sum()),
            equal_native_frames=int((delta == 0).sum()),
            worse_native_frames=int((delta > 0).sum()),
            per_view_delta_mean=(per_view[:, index] - per_view[:, 0]).mean(0).tolist(),
        )
    return dict(
        schema=STAGE,
        status="ok",
        native_interval=[FIRST, STOP],
        arms=list(metrics.ARMS),
        evidence=list(metrics.EVIDENCE),
        anchor_ids=[a.anchor_id for a in anchors],
        original_unused_union=comparison,
        owner_mean_evidence_arm_anchor=owners.mean(axis=0).tolist(),
        owner_target_positive_body_view_times_by_evidence_anchor=(masses > 0)
        .sum((0, 3, 4))
        .tolist(),
        support=support,
        strict_candidate_support_rechecks=600,
        unused_pixel_draws=153600,
        temporal_holdout=False,
        shared_D2_spatial_resampling=True,
        independent_ID_truth=False,
        anchor_selection_by_pose=False,
        masks_selected_by_pose=False,
        trajectory_or_review_modified=False,
        GPU_initialized=torch.cuda.is_initialized(),
    )


def report(summary):
    lines = [
        "# マスク対応を追加した200フレームの独立評価",
        "",
        "3方式を全200フレームで比較した。Originalの全21出力・全19履歴・全39状態は元の結果と一致した。",
        "追加項に使わなかった同じD2画素で評価し、"
        "全600 arm-frameの現在SL支持を屈折幾何で再確認した。",
        "これはID正解や別の動画による検証ではなく、スワップゼロを証明しない。",
        "",
        "|方式|未使用画素の平均union損失|Originalとの差|改善/同値/悪化フレーム|",
        "|---|---:|---:|---|",
    ]
    for arm in metrics.ARMS:
        r = summary["original_unused_union"][arm]
        lines.append(
            f"|{arm}|{r['mean']:.9f}|{r['delta_mean']:+.9f}|"
            f"{r['better_native_frames']}/{r['equal_native_frames']}/{r['worse_native_frames']}|"
        )
    lines += ["", "|方式|口の直接SL支持/600|胸ヒレ/600|尾/600|", "|---|---:|---:|---:|"]
    for arm in metrics.ARMS:
        counts = summary["support"][arm]["direct_count_by_node"]
        lines.append(f"|{arm}|{counts[0]}|{counts[1]}|{counts[2]}|")
    lines += [
        "",
        "身体所有の追加損失はLK/CTの両方の証拠と全初期anchorに対して計算し、"
        "候補に都合のよいanchorへ絞っていない。全行とマスクの競合・保留理由を保存した。",
        "支持の増減・欠測の全区間はsummary.json、画素ごとの描画・所有割合はframes/にある。",
        "maskとSLは誤ることがあり、この診断で全3個体の生物IDを確定していない。",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def run(tracking_root, tracking_sha, destination):
    root, destination = Path(tracking_root).resolve(), Path(destination).resolve()
    require(
        base.valid_sha(tracking_sha)
        and root.is_relative_to(WORK)
        and destination.is_relative_to(WORK)
        and not destination.exists(),
        "workspace input/output",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, rows = base.utc(), time.perf_counter(), []
    old_cv = cv2.getNumThreads()
    try:
        evidence.add(Path(__file__).resolve())
        evidence.add(Path(metrics.__file__).resolve(), METRICS_SHA)
        evidence.add(Path(diagnostics.__file__).resolve(), DIAGNOSTICS_SHA)
        seal, members = base.verified(
            root, "tracking.provenance.json", producer.STAGE, tracking_sha, evidence
        )
        source = str(Path(producer.__file__).resolve())
        require(source in seal["inputs"], "producer source not sealed")
        evidence.add(source, seal["inputs"][source])
        plan, summary, control = (
            base.bound_json(root, members, name)
            for name in ("plan.json", "summary.json", "original_control.json")
        )
        validate_run(plan, summary, control)
        reuse = base.bound_json(root, members, "original_reuse_qualification.json")
        validate_reuse(plan, summary, reuse)
        common, sample_members, points, _ = producer.load_inputs(evidence, producer.SAMPLES_SHA)
        require(common["original_context"] == plan["original_context"], "original context drift")
        cv2.setNumThreads(32)
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "evaluation must use CPU")
        prepared.snapshot_sources(evidence, output, captured)
        candidates = {
            arm: producer.read_arrays(root, members, f"{arm}/candidate.npz") for arm in metrics.ARMS
        }
        support = metrics.summarize_support(candidates)
        frame_trees = {
            arm: [load_tree(root, members, f"{arm}/frames/native_{n}") for n in range(FIRST, STOP)]
            for arm in metrics.ARMS
        }
        geometry = diagnostics.candidate_geometry(candidates, common["initial"])
        retained = diagnostics.retained_history(frame_trees, candidates, common["initial"])
        decisions = diagnostics.transport_diagnostics(
            frame_trees, frame_idx=candidates["original"]["frame_idx"]
        )
        producer.save_tree(output, "diagnostics/body_geometry", geometry)
        producer.save_tree(output, "diagnostics/retained_history", retained)
        output.write("diagnostics/transport_decisions.json", decisions)
        ancillary = dict(
            body_geometry=geometry["summary"],
            retained_history=retained["summary"],
            transport=decisions["summary"],
        )
        del frame_trees, geometry, retained, decisions
        bank = load_tree(root, members, "original/initial_anchors")
        anchors = []
        require(
            1 <= len(bank["facts"]) == len(bank["anchors"]) <= 6, "all initial anchors required"
        )
        for index, fact in enumerate(bank["facts"]):
            anchor = transport.build_initial_anchor(
                tuple(fact["references"]),
                points.group_counts,
                anchor_id=index,
                initial_native=FIRST,
            )
            for vi in range(3):
                producer.same(
                    anchor.links[vi],
                    bank["anchors"][index]["links"][vi],
                    "initial anchor reconstruction",
                )
            anchors.append(anchor)
        anchors = tuple(anchors)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_tracking_sha256=tracking_sha,
                native_interval=[FIRST, STOP],
                all_initial_anchor_ids=list(range(len(anchors))),
                evidence=list(metrics.EVIDENCE),
                arms=list(metrics.ARMS),
                rendering_device="cpu",
                rendering_dtype="float64",
                width_ratio=0.06,
                maximum_ray_mm=2.0,
                maximum_pixel_error=12.0,
                samples_sha256=producer.SAMPLES_SHA,
                first_release_primary=True,
                reused_original_source_seal_sha256=producer.FAILED_SHA,
                newly_computed_tracking_arms=["lk", "ct_first"],
            ),
        )
        for offset, frame in enumerate(common["reader"]):
            native = frame.frame_idx
            sample = producer.read_arrays(
                producer.SAMPLES, sample_members, f"samples/native_{native}.npz"
            )
            producer.validate_samples(sample, native)
            checked = recheck_support(candidates, offset, frame, common["rig"])
            points.prepare(frame)
            packets = tuple(
                transport.packet_for_pixels(
                    points.provider(method)(native),
                    anchors,
                    tuple(frame.masks[v] for v in metrics.VIEWS),
                    sample["unused_pixels"],
                )
                for method in metrics.EVIDENCE
            )
            poses = np.stack([candidates[a]["pose_xyz"][offset] for a in metrics.ARMS])
            scores = metrics.score_fixed_images(poses, sample, packets, common["rig"])
            output.write(f"frames/native_{native}.npz", scores)
            producer.save_tree(
                output, f"evidence/native_{native}", dict(packets=packets, strict_SLEAP=checked)
            )
            rows.append(
                {
                    k: scores[k]
                    for k in ("union_loss", "union_per_view", "owner_loss", "target_mass")
                }
            )
        require(len(rows) == 200, "all native200 evaluations required")
        result = summary_values(rows, support, anchors)
        result["ancillary"] = ancillary
        result["reused_original_source_seal_sha256"] = producer.FAILED_SHA
        result["elapsed_before_guards_seconds"] = time.perf_counter() - start
        output.write("summary.json", result)
        output.write("REPORT.md", report(result))
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(all_guards_passed=True),
        )
    except BaseException as error:
        status = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_frames=len(rows),
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "evaluation.provenance.json").open("xb") as f:
            f.write(base.json_bytes(status))
        raise
    finally:
        cv2.setNumThreads(old_cv)
    with (destination / "evaluation.provenance.json").open("xb") as f:
        f.write(base.json_bytes(status))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--tracking-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.tracking_root, args.tracking_sha256, args.output)), flush=True)


if __name__ == "__main__":
    main()
