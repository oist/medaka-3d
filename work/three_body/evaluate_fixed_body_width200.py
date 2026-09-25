"""Six saved trajectories under two common renderers on unchanged unused pixels.

Primary uses original dynamic widths for every arm; secondary uses registered
fixed widths for every arm. Neither scoring renderer is selected per trajectory.
No fitting, model, state initialization/restore or CUDA execution is permitted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import cv2
import evaluate_mask_transport200 as old_evaluation
import evaluate_mask_transport_gap200 as prior_evaluation
import mask_transport_diagnostics as geometry
import numpy as np
import object_mask_transport as transport
import run_fixed_body_width200_v2 as producer
import torch

previous = producer.previous
base, prepared, inputs = producer.base, producer.prepared, producer.inputs
WORK, FIRST, STOP = producer.WORK, producer.FIRST, producer.STOP
require = producer.require
STAGE = "fixed_body_width200_independent_evaluation_v1"
ARMS = (*prior_evaluation.ARMS, *producer.ARMS)
METRICS_SHA = "d959cd1aa9570ec797a764dd697bf26a13078290bf417d97d1c049815d9c7e5f"
TRACKER_SHA = "16294aa7264f3aa8634304e53c658623b621402a6ccb957b65d7c750d2a994fd"
PRODUCER_SHA = "d3083a061fc599c1e066f58f3144593dd48d979c915de3e66a390a354d811dcf"
AUDIT_SHA = "41fe61834aa9bc11254898260ee808c7dd7a63a1a5dac1d14cfd3792ac78eb73"
OLD_GAP = WORK / "mask_transport_gap200_tracking_v1"
OLD_GAP_SHA = "a3222e383ce5e08b63a2f4bba8a2404da95cf27f1e6513be44f2f90e4aaf5aa9"
OLD_EVALUATION = WORK / "mask_transport_gap200_independent_evaluation_v1"
OLD_EVALUATION_SHA = "4a68497916105c4d205101542c79223bddac072bfcf037781a39447c5bc73ef9"
PRIOR_EVALUATOR_SHA = "eb26fe9c67463ac004e88fae1921536668c435f1305b8bdd19973de679000e4e"


def validate_run(plan, summary, initial):
    require(
        plan["schema"] == summary["schema"] == producer.STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["arms"] == list(producer.ARMS)
        and set(summary["arms"]) == set(producer.ARMS),
        "complete same-native two-arm fixed-width run required",
    )
    binding = plan["input_binding"]
    for key, expected in dict(
        prior_tracking_sha256=producer.prior_gap.PREVIOUS_SHA,
        prior_evaluation_sha256=producer.prior_gap.EVALUATION_SHA,
        samples_sha256=previous.SAMPLES_SHA,
        point_sha256=previous.POINTS_SHA,
        preparation_sha256=inputs.PREPARATION_SHA,
        shape_plan_sha256=producer.PLAN_SHA,
        cuda_qualification_sha256=producer.QUALIFICATION_SHA,
    ).items():
        require(binding[key] == expected, "fixed-width input binding differs: " + key)
    require(
        binding["source_pins"] == producer.SOURCE_PINS
        and plan["point_evidence_arm"] == "ct_first"
        and plan["gap_relative_coefficient"] == 1.0
        and plan["mask_coefficients"]
        == dict(zip(producer.ARMS, producer.COEFFICIENTS, strict=True))
        and plan["fixed_width_applied_from_native"] == FIRST + 1
        and plan["extra_proposals"] == 0
        and plan["maximum_pool"] == summary["maximum_pool"] == 24
        and plan["opt_steps"] == summary["steps"] == 14
        and summary["initialize_calls"] == 2
        and summary["restore_calls"] == 0
        and summary["advance_calls"] == 400
        and summary["query_refreshes"] == summary["model_forwards"] == 0
        and summary["no_external_pose_reset"] is True,
        "fixed-width objective/budget/lifecycle differs",
    )
    for arm, coefficient in zip(producer.ARMS, producer.COEFFICIENTS, strict=True):
        row, first = summary["arms"][arm], initial[arm]
        require(
            row["arm"] == arm
            and row["coefficient"] == coefficient
            and row["initialize_count"] == 1
            and row["restore_count"] == 0
            and row["frames"] == 200
            and row["fixed_width_fits"] == 199
            and first["passed"] is True
            and first["native"] == FIRST
            and first["prior_trajectory_used"] is False
            and first["all19_retained_records_exact"] is True
            and first["all_initial_anchors_facts_exact"] is True,
            "fixed-width arm or causal initialization differs: " + arm,
        )


def previous_image_control(scores, original):
    for key in (
        "rendered_union",
        "rendered_ownership",
        "rendered_q",
        "union_per_view",
        "union_loss",
    ):
        previous.same(scores[key][:4], original[key], "old-four common renderer " + key)
    for key in (
        "owner_per_body_view",
        "owner_per_view",
        "owner_loss",
        "gap_per_body_view",
        "gap_per_view",
        "gap_loss",
        "combined_per_body_view",
        "combined_per_view",
        "combined_loss",
    ):
        previous.same(scores[key][:, :4], original[key], "old-four image evidence " + key)
    for key in ("target_mass", "positive_target_mass", "ray_valid", "anchor_ids", "frame_idx"):
        previous.same(scores[key], original[key], "common all-anchor evidence " + key)


def support_recheck(candidates, offset, frame, rig):
    rows = prior_evaluation.support_recheck(candidates, offset, frame, rig)
    rays = old_evaluation.RayObservations.from_points(rig, frame.detections)
    for arm in producer.ARMS:
        candidate = candidates[arm]
        sources, observed, distances = old_evaluation.supported_nodes(
            candidate["pose_xyz"][offset],
            candidate["sources"][offset],
            rays,
            rig,
        )
        previous.same(sources, candidate["sources"][offset], arm + " strict source")
        previous.same(observed, candidate["observed"][offset], arm + " strict support")
        for vi in range(3):
            for node in range(3):
                used = sources[:, vi, node]
                used = used[used >= 0]
                require(len(used) == len(np.unique(used)), "source reused across bodies")
        rows[arm] = dict(sources=sources, observed=observed, ray_distance_mm=distances)
    return rows


def validate_old_gap(evidence):
    _, members = base.verified(
        OLD_GAP,
        "tracking.provenance.json",
        producer.prior_gap.STAGE,
        OLD_GAP_SHA,
        evidence,
    )
    plan, summary, initial = (
        base.bound_json(OLD_GAP, members, n)
        for n in ("plan.json", "summary.json", "initial_control.json")
    )
    prior_evaluation.validate_run(plan, summary, initial)
    seal, eval_members = base.verified(
        OLD_EVALUATION,
        "evaluation.provenance.json",
        prior_evaluation.STAGE,
        OLD_EVALUATION_SHA,
        evidence,
    )
    result = base.bound_json(OLD_EVALUATION, eval_members, "summary.json")
    require(
        seal["inputs"].get(str(OLD_GAP / "tracking.provenance.json")) == OLD_GAP_SHA
        and seal["inputs"].get(str(Path(prior_evaluation.__file__).resolve()))
        == PRIOR_EVALUATOR_SHA
        and result["status"] == "ok"
        and result["native_interval"] == [FIRST, STOP]
        and result["arms"] == list(prior_evaluation.ARMS)
        and result["original_three_CPU_image_control_exact_frames"] == 200
        and result["strict_candidate_support_rechecks"] == 800,
        "saved four-arm independent comparison differs",
    )
    return members, eval_members


def summary_values(rows, support, shape):
    union = np.stack([r["union_loss"] for r in rows])
    views = np.stack([r["union_per_view"] for r in rows])
    images = {}
    for index, arm in enumerate(ARMS):
        delta = union[:, index] - union[:, 0]
        images[arm] = dict(
            mean=float(union[:, index].mean()),
            per_view_mean=views[:, index].mean(0).tolist(),
            delta_mean=float(delta.mean()),
            per_view_delta_mean=(views[:, index] - views[:, 0]).mean(0).tolist(),
            better_native_frames=int((delta < 0).sum()),
            equal_native_frames=int((delta == 0).sum()),
            worse_native_frames=int((delta > 0).sum()),
        )
    return dict(
        schema=STAGE,
        status="ok",
        native_interval=[FIRST, STOP],
        arms=list(ARMS),
        evidence=["lk", "ct_first"],
        union=images,
        support=support,
        body_geometry=shape,
        owner_mean_evidence_arm_anchor=np.stack([r["owner_loss"] for r in rows]).mean(0).tolist(),
        gap_mean_evidence_arm_anchor=np.stack([r["gap_loss"] for r in rows]).mean(0).tolist(),
        combined_mean_evidence_arm_anchor=np.stack([r["combined_loss"] for r in rows])
        .mean(0)
        .tolist(),
        combined_definition="owner + gap; coefficient3 not applied; not full tracking objective",
        original_four_CPU_image_control_exact_frames=200,
        strict_candidate_support_rechecks=1200,
        unused_pixel_draws=153600,
        shared_D2_spatial_resampling=True,
        temporal_holdout=False,
        independent_ID_truth=False,
        anchor_selection_by_pose=False,
        masks_selected_by_pose=False,
        trajectory_or_review_modified=False,
        GPU_initialized=False,
    )


def paired_scores(rows):
    union = np.stack([row["union_loss"] for row in rows])
    pairs = (("width_base", 4, 0), ("fixed_auxiliary", 5, 4), ("dynamic_auxiliary", 3, 0))
    result = {}
    for label, a, b in pairs:
        delta = union[:, a] - union[:, b]
        result[label] = dict(
            first=ARMS[a],
            reference=ARMS[b],
            mean_delta=float(delta.mean()),
            better_native_frames=int((delta < 0).sum()),
            equal_native_frames=int((delta == 0).sum()),
            worse_native_frames=int((delta > 0).sum()),
        )
    result["auxiliary_interaction_mean_delta"] = float(
        ((union[:, 5] - union[:, 4]) - (union[:, 3] - union[:, 0])).mean()
    )
    return result


def report(result):
    lines = [
        "# 固定体幅200フレーム: 共通する2種類の描画モデルによる6方式比較",
        "",
        "一次評価は旧描画モデルを全6方式へ、二次評価は固定幅モデルを全6方式へ適用した。",
        "共通の未使用画素153600点と全アンカーを使用。SL支持は全1200 arm-frameで再確認した。",
        "保存済み4方式の一次評価は完全一致。ID真値や独立動画による検証ではない。",
        "",
        "|方式|一次・旧描画誤差|二次・固定幅誤差|口 /600|胸 /600|尾 /600|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        n = result["support"][arm]["direct_count_by_node"]
        lines.append(
            f"|{arm}|{result['union'][arm]['mean']:.9f}|"
            f"{result['secondary_fixed_width']['union'][arm]['mean']:.9f}|{n[0]}|{n[1]}|{n[2]}|"
        )
    lines.extend(
        [
            "",
            "身体長・折れ角・支持の獲得と損失、全保持候補、全視点の内訳も保存した。",
            "描画誤差の低下だけで採用せず、元画像で節点と身体を確認する。",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def run(root, digest, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(
        root.is_relative_to(WORK) and destination.is_relative_to(WORK) and not destination.exists(),
        "workspace input and new output required",
    )
    require(
        all(base.valid_sha(x) for x in (digest, METRICS_SHA, TRACKER_SHA, PRODUCER_SHA, AUDIT_SHA)),
        "final source/producer SHA required",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, rows = base.utc(), time.perf_counter(), []
    old_cv = cv2.getNumThreads()
    try:
        evidence.add(Path(__file__).resolve())
        evidence.add(Path(producer.__file__).resolve(), PRODUCER_SHA)
        metrics = importlib.import_module("fixed_body_width_evaluation")
        width_module = importlib.import_module("fixed_body_width_tracker")
        audit_module = importlib.import_module("fixed_body_width_state_audit")
        evidence.add(Path(audit_module.__file__).resolve(), AUDIT_SHA)
        evidence.add(Path(prior_evaluation.__file__).resolve(), PRIOR_EVALUATOR_SHA)
        evidence.add(Path(metrics.__file__).resolve(), METRICS_SHA)
        evidence.add(Path(width_module.__file__).resolve(), TRACKER_SHA)
        evidence.add(Path(geometry.__file__).resolve(), old_evaluation.DIAGNOSTICS_SHA)
        seal, members = base.verified(
            root, "tracking.provenance.json", producer.STAGE, digest, evidence
        )
        require(
            seal["inputs"].get(str(Path(producer.__file__).resolve())) == PRODUCER_SHA,
            "new producer differs",
        )
        for relative, source_sha in {
            **producer.prior_gap.SOURCE_PINS,
            **producer.SOURCE_PINS,
        }.items():
            path = producer.REPO / relative
            require(
                seal["inputs"].get(str(path)) == source_sha,
                "tracking implementation differs from registered source: " + relative,
            )
            evidence.add(path, source_sha)
        plan, summary = (base.bound_json(root, members, n) for n in ("plan.json", "summary.json"))
        initial = {
            a: base.bound_json(root, members, f"{a}/initial_control.json") for a in producer.ARMS
        }
        validate_run(plan, summary, initial)
        gap_members, eval_members = validate_old_gap(evidence)
        prior_plan, _, _, bank = producer.load_controls(evidence)
        common, sample_members, points, _ = previous.load_inputs(evidence, previous.SAMPLES_SHA)
        require(
            common["original_context"]
            == plan["original_context"]
            == prior_plan["original_context"],
            "science context differs",
        )
        prior_seal = base.read_json(producer.prior_gap.PREVIOUS / "tracking.provenance.json")
        prior_members = previous.member_index(prior_seal, producer.prior_gap.PREVIOUS)
        cv2.setNumThreads(32)
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "evaluation must remain CPU")
        prepared.snapshot_sources(evidence, output, captured)
        candidates = {
            a: previous.read_arrays(
                producer.prior_gap.PREVIOUS, prior_members, f"{a}/candidate.npz"
            )
            for a in previous.ARMS
        }
        candidates["ct_gap"] = previous.read_arrays(OLD_GAP, gap_members, "ct_gap/candidate.npz")
        for arm in producer.ARMS:
            candidates[arm] = previous.read_arrays(root, members, f"{arm}/candidate.npz")
        support = metrics.summarize_support(candidates)
        cores, contents, width_audits = {}, {}, {}
        for arm, coefficient in zip(producer.ARMS, producer.COEFFICIENTS, strict=True):
            # Pure validation view adapter: exact original metadata, no restore or CUDA.
            prefix = f"{arm}/checkpoint/"
            core = previous.read_arrays(root, members, prefix + "arrays.npz")
            meta = base.bound_json(root, members, prefix + "metadata.json")
            side = previous.read_arrays(root, members, prefix + "side_arrays.npz")
            side_meta = base.bound_json(root, members, prefix + "side_metadata.json")
            core, content, audit = audit_module.validate_state(
                core,
                meta,
                side,
                side_meta,
                rig=common["rig"],
                reference_pose=common["initial"],
                original_context=common["original_context"],
                first_native=FIRST,
                group_counts=points.group_counts,
                query_sha256=points.query_sha256,
                input_binding=plan["input_binding"],
                coefficient=coefficient,
            )
            cores[arm], contents[arm], width_audits[arm] = core, content, audit
            chosen = prior_evaluation.candidate_backtrace_control(candidates[arm], core)
            output.write(
                f"diagnostics/{arm}_candidate_backtrace.npz", dict(flat_retained_rows=chosen)
            )
            output.write(f"diagnostics/{arm}_state_audit.json", audit)
            current_bank = old_evaluation.load_tree(root, members, f"{arm}/initial_anchors")
            previous.same_tree(current_bank, bank, arm + " original causal bank")
        shapes = {}
        for arm, candidate in candidates.items():
            values = geometry._geometry(
                candidate["pose_xyz"], candidate["observed"], common["initial"]
            )
            output.write(f"diagnostics/{arm}_body_geometry.npz", values)
            shapes[arm] = geometry._geometry_summary(values)
        for arm, core in cores.items():
            retained = geometry._geometry(
                core["history_pose_xyz"], core["history_observed"], common["initial"]
            )
            output.write(f"diagnostics/{arm}_all_retained_geometry.npz", retained)
            output.write(
                f"diagnostics/{arm}_all_retained_summary.json",
                dict(
                    rows=int(core["history_offsets"][-1]),
                    bodies=geometry._geometry_summary(retained),
                ),
            )
        anchors = []
        for index, fact in enumerate(bank["facts"]):
            anchor = transport.build_initial_anchor(
                tuple(fact["references"]),
                points.group_counts,
                anchor_id=index,
                initial_native=FIRST,
            )
            for vi in range(3):
                previous.same(
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
                source_tracking_sha256=digest,
                native_interval=[FIRST, STOP],
                arms=list(ARMS),
                evidence=list(metrics.EVIDENCE),
                all_initial_anchor_ids=list(range(len(anchors))),
                previous_tracking_sha256=producer.prior_gap.PREVIOUS_SHA,
                previous_evaluation_sha256=OLD_EVALUATION_SHA,
                primary_renderer="original_dynamic_all6",
                secondary_renderer="registered_fixed_radial_all6",
                samples_sha256=previous.SAMPLES_SHA,
                rendering_device="cpu",
                rendering_dtype="float64",
                width_ratio=0.06,
            ),
        )
        secondary_rows = []
        for offset, frame in enumerate(common["reader"]):
            native = frame.frame_idx
            sample = previous.read_arrays(
                previous.SAMPLES, sample_members, f"samples/native_{native}.npz"
            )
            previous.validate_samples(sample, native)
            for arm, core in cores.items():
                tree = old_evaluation.load_tree(root, members, f"{arm}/frames/native_{native}")
                previous.same_tree(
                    tree["fit"], contents[arm]["fits"][offset], "gap full frame/sidecar fit"
                )
                previous.fit_sample_match(tree["fit"], sample)
                lo, hi = map(int, core["history_offsets"][offset : offset + 2])
                require(len(tree["records"]) == hi - lo, "gap full retained rows differ")
                for record, index in zip(tree["records"], range(lo, hi), strict=True):
                    for key in previous.RECORD_FIELDS:
                        tracking = previous.tracking
                        tracking.same_record(
                            record[key], core["history_" + key][index], "gap frame/core " + key
                        )
            checked = support_recheck(candidates, offset, frame, common["rig"])
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
            poses = np.stack([candidates[a]["pose_xyz"][offset] for a in ARMS])
            scores = metrics.score_fixed_images(poses, sample, packets, common["rig"])
            saved = previous.read_arrays(
                OLD_EVALUATION, eval_members, f"frames/native_{native}.npz"
            )
            previous_image_control(scores, saved)
            output.write(f"frames/native_{native}.npz", scores)
            secondary = metrics.score_fixed_images(
                poses,
                sample,
                packets,
                common["rig"],
                reference_pose_xyz=common["initial"],
            )
            output.write(f"fixed_frames/native_{native}.npz", secondary)
            secondary_rows.append(
                {
                    k: secondary[k]
                    for k in (
                        "union_loss",
                        "union_per_view",
                        "owner_loss",
                        "gap_loss",
                        "combined_loss",
                    )
                }
            )
            previous.save_tree(
                output, f"evidence/native_{native}", dict(packets=packets, strict_SLEAP=checked)
            )
            rows.append(
                {
                    k: scores[k]
                    for k in (
                        "union_loss",
                        "union_per_view",
                        "owner_loss",
                        "gap_loss",
                        "combined_loss",
                    )
                }
            )
        require(len(rows) == 200, "all native200 required")
        result = summary_values(rows, support, shapes)
        secondary_result = summary_values(secondary_rows, support, shapes)
        result["secondary_fixed_width"] = {
            k: secondary_result[k]
            for k in (
                "union",
                "owner_mean_evidence_arm_anchor",
                "gap_mean_evidence_arm_anchor",
                "combined_mean_evidence_arm_anchor",
            )
        }
        result["paired_primary"] = paired_scores(rows)
        result["paired_secondary"] = paired_scores(secondary_rows)
        result["static_support_comparison"] = metrics.summarize_static_support(candidates)
        result["width_reference_audits"] = width_audits
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
        with (destination / "evaluation.provenance.json").open("xb") as stream:
            stream.write(base.json_bytes(status))
        raise
    finally:
        cv2.setNumThreads(old_cv)
    with (destination / "evaluation.provenance.json").open("xb") as stream:
        stream.write(base.json_bytes(status))
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
