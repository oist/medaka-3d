"""Independent four-arm fixed-pixel evaluation of one new mask-gap trajectory.

All three prior arms must reproduce their saved CPU image scores. Both point
evidence streams and every causal initial anchor are retained for the new arm.
This is spatial resampling of shared D2 evidence, not independent ID truth.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import time
from pathlib import Path

import cv2
import evaluate_mask_transport200 as old_evaluation
import mask_transport_diagnostics as geometry
import numpy as np
import object_mask_transport as transport
import run_mask_transport_gap200 as producer
import torch

previous = producer.previous
base, prepared, inputs = producer.base, producer.prepared, producer.inputs
WORK, FIRST, STOP = producer.WORK, producer.FIRST, producer.STOP
require = producer.require
STAGE = "mask_transport_gap200_independent_evaluation_v1"
ARMS = (*previous.ARMS, producer.ARM)
METRICS_SHA = "835ed6d11ba44d88361566d32c1860416eb232b7032c6cfa8803cbd684469991"
TRACKER_SHA = "42875f30ede026d2e315044b30c4625b75b2f45c98610f6581d607d93aaa9789"
PRODUCER_SHA = "af36a32adc4be70acda870c6037d7fb6a3d7f8a53796ba9fb0c1e4e7bc1991b4"


def validate_run(plan, summary, initial):
    require(
        plan["schema"] == summary["schema"] == producer.STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP],
        "complete same-native gap run required",
    )
    binding = plan["input_binding"]
    require(
        binding["prior_tracking_sha256"]
        == summary["prior_tracking_sha256"]
        == producer.PREVIOUS_SHA
        and binding["prior_evaluation_sha256"]
        == summary["prior_evaluation_sha256"]
        == producer.EVALUATION_SHA
        and binding["samples_sha256"] == previous.SAMPLES_SHA
        and binding["point_sha256"] == previous.POINTS_SHA
        and binding["preparation_sha256"] == inputs.PREPARATION_SHA
        and binding["gap_plan_sha256"] == producer.PLAN_SHA,
        "gap evidence lineage differs",
    )
    require(
        plan["arm"] == summary["arm"]["arm"] == producer.ARM
        and plan["point_evidence_arm"] == "ct_first"
        and plan["objective"] == "base + 3*(owner + ray_gap)"
        and plan["gap_relative_coefficient"] == 1.0
        and plan["mask_coefficient"] == 3.0
        and plan["extra_proposals"] == 0
        and plan["maximum_pool"] == summary["maximum_pool"] == 24
        and plan["opt_steps"] == summary["steps"] == 14,
        "gap objective or budget differs",
    )
    require(
        summary["initialize_calls"] == summary["arm"]["initialize_count"] == 1
        and summary["restore_calls"] == summary["arm"]["restore_count"] == 0
        and summary["advance_calls"] == summary["arm"]["frames"] == 200
        and summary["query_refreshes"] == summary["model_forwards"] == 0
        and summary["no_external_pose_reset"] is True
        and summary["causal_initial_control_passed"] is True
        and initial["passed"] is True
        and initial["native"] == FIRST
        and initial["prior_trajectory_used"] is False,
        "new gap arm must be complete without intermediate reinitialization",
    )


def previous_image_control(scores, original):
    """Four-arm rendering cannot silently alter the prior three reference arms."""
    for key in ("rendered_union", "rendered_ownership", "union_per_view", "union_loss"):
        previous.same(scores[key][:3], original[key], "prior CPU image control " + key)
    for key in ("owner_per_body_view", "owner_per_view", "owner_loss"):
        previous.same(scores[key][:, :3], original[key], "prior owner control " + key)
    for key in ("target_mass", "positive_target_mass", "ray_valid", "anchor_ids", "frame_idx"):
        previous.same(scores[key], original[key], "prior evidence control " + key)


def support_recheck(candidates, offset, frame, rig):
    rows = old_evaluation.recheck_support(candidates, offset, frame, rig)
    candidate = candidates[producer.ARM]
    rays = old_evaluation.RayObservations.from_points(rig, frame.detections)
    sources, observed, distances = old_evaluation.supported_nodes(
        candidate["pose_xyz"][offset], candidate["sources"][offset], rays, rig
    )
    previous.same(sources, candidate["sources"][offset], "gap current strict source")
    previous.same(observed, candidate["observed"][offset], "gap current strict support")
    for vi in range(3):
        for node in range(3):
            used = sources[:, vi, node]
            used = used[used >= 0]
            require(len(used) == len(np.unique(used)), "gap source row reused by two bodies")
    rows[producer.ARM] = dict(sources=sources, observed=observed, ray_distance_mm=distances)
    return rows


def validate_gap_state(root, members, common, points, gap_module, plan):
    """Decode the new logical state/sidecar in an uninitialized CPU validator."""
    prefix = f"{producer.ARM}/checkpoint/"
    core = previous.read_arrays(root, members, prefix + "arrays.npz")
    meta = base.bound_json(root, members, prefix + "metadata.json")
    require(
        meta["context"]["mask_transport"]["input_binding"] == plan["input_binding"],
        "gap checkpoint evidence binding differs from successful plan",
    )
    arrays = previous.read_arrays(root, members, prefix + "side_arrays.npz")
    side_meta = base.bound_json(root, members, prefix + "side_metadata.json")
    original = common["original_context"]
    validator = gap_module.MaskTransportGapTracker(
        common["rig"],
        common["initial"].copy(),
        config=previous.TrackerConfig(**{**original["tracker_config"], "device": "cpu"}),
        recovery_config=previous.NodeRecoveryConfig(**original["recovery_config"]),
        persistence_config=previous.PersistentNodeConfig(**original["persistence_config"]),
        first_native=FIRST,
        group_counts=points.group_counts,
        query_sha256=points.query_sha256,
        vote_provider=lambda _: (_ for _ in ()).throw(
            AssertionError("validation cannot request votes")
        ),
        input_binding=meta["context"]["mask_transport"]["input_binding"],
        coefficient=3,
    )
    expected = copy.deepcopy(meta["context"])
    expected["tracker_config"]["device"] = "cpu"
    require(validator._context() == expected, "CPU state validator context differs beyond device")
    cpu_meta = copy.deepcopy(meta)
    cpu_meta["context"] = expected
    validator._decode_state(core, cpu_meta)
    content = validator._validated_sidecar(core, meta, arrays, side_meta)
    require(
        validator.initialize_count == validator.restore_count == 0 and validator._states is None,
        "evaluation initialized or restored a tracker",
    )
    require(
        meta["processed_frames"] == 200
        and meta["initialize_count"] == 1
        and meta["restore_count"] == 0,
        "saved gap lifecycle differs",
    )
    return core, content


def candidate_backtrace_control(candidate, core):
    """All 21 outputs must follow terminal rank zero through saved parent rows.

    These are the frozen finish equations, evaluated without initializing or
    restoring a tracker. Flattening can widen Unicode storage; compare textual
    fields by value and every other field by exact dtype/shape/numeric bytes.
    """
    offsets = core["history_offsets"]
    require(
        offsets.dtype == np.int64
        and offsets.ndim == 1
        and len(offsets) >= 2
        and offsets[0] == 0
        and np.all(np.diff(offsets) > 0),
        "complete retained offsets required",
    )
    chosen, state_index = [], 0
    for t in range(len(offsets) - 2, -1, -1):
        lo, hi = int(offsets[t]), int(offsets[t + 1])
        require(0 <= state_index < hi - lo, "backtrace parent outside retained beam")
        index = lo + state_index
        chosen.append(index)
        state_index = int(core["history_parent"][index])
    chosen = np.asarray(chosen[::-1], np.int64)
    expected = {k: core["history_" + k][chosen] for k in previous.RECORD_FIELDS if k != "parent"}
    expected["xyz"] = np.where(
        expected["observed"][:, :, 1, None], expected["pose_xyz"][:, :, 1], np.nan
    )
    expected["accepted_xyz"] = np.where(
        expected["identity_review"][:, :, None], np.nan, expected["xyz"]
    )
    expected["body_xyz"] = np.where(
        expected["body_supported"][:, :, None], expected["pose_xyz"][:, :, 1], np.nan
    )
    require(
        set(candidate) == set(expected) and len(expected) == 21, "all21 backtrace fields required"
    )
    for key in expected:
        previous.tracking.same_record(candidate[key], expected[key], "gap backtrace " + key)
    return chosen


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
        original_three_CPU_image_control_exact_frames=200,
        strict_candidate_support_rechecks=800,
        unused_pixel_draws=153600,
        shared_D2_spatial_resampling=True,
        temporal_holdout=False,
        independent_ID_truth=False,
        anchor_selection_by_pose=False,
        masks_selected_by_pose=False,
        trajectory_or_review_modified=False,
        GPU_initialized=False,
    )


def report(result):
    lines = [
        "# 距離項を加えた200フレームの独立比較",
        "",
        "全4方式を同じ未使用画素で評価し、元の3方式の描画・点数は保存済み結果と完全一致した。",
        "全800 arm-frameのSL支持を現在の屈折幾何で再確認した。ID真値や独立動画の検証ではない。",
        "",
        "|方式|未使用投影損失|口 /600|胸ヒレ /600|尾 /600|",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        counts = result["support"][arm]["direct_count_by_node"]
        lines.append(
            f"|{arm}|{result['union'][arm]['mean']:.9f}|{counts[0]}|{counts[1]}|{counts[2]}|"
        )
    lines.extend(
        [
            "",
            "追加項の低下だけで採用せず、投影全体・SL支持・身体形状・原画像を比較する。",
            "全保持候補の形状、画素別q/所有率/距離内訳、全アンカーの点数と支持の増減を保存した。",
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
        all(base.valid_sha(x) for x in (digest, METRICS_SHA, TRACKER_SHA, PRODUCER_SHA)),
        "final source/producer SHA required",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, rows = base.utc(), time.perf_counter(), []
    old_cv = cv2.getNumThreads()
    try:
        evidence.add(Path(__file__).resolve())
        evidence.add(Path(producer.__file__).resolve(), PRODUCER_SHA)
        metrics = importlib.import_module("mask_transport_gap_evaluation")
        gap_module = importlib.import_module("mask_transport_gap_tracker")
        evidence.add(Path(metrics.__file__).resolve(), METRICS_SHA)
        evidence.add(Path(gap_module.__file__).resolve(), TRACKER_SHA)
        evidence.add(Path(geometry.__file__).resolve(), old_evaluation.DIAGNOSTICS_SHA)
        seal, members = base.verified(
            root, "tracking.provenance.json", producer.STAGE, digest, evidence
        )
        require(
            seal["inputs"].get(str(Path(producer.__file__).resolve())) == PRODUCER_SHA,
            "new producer differs",
        )
        plan, summary, initial = (
            base.bound_json(root, members, n)
            for n in ("plan.json", "summary.json", "initial_control.json")
        )
        validate_run(plan, summary, initial)
        prior_plan, _, _, bank = producer.load_controls(evidence)
        common, sample_members, points, _ = previous.load_inputs(evidence, previous.SAMPLES_SHA)
        require(
            common["original_context"]
            == plan["original_context"]
            == prior_plan["original_context"],
            "science context differs",
        )
        prior_seal = base.read_json(producer.PREVIOUS / "tracking.provenance.json")
        prior_members = previous.member_index(prior_seal, producer.PREVIOUS)
        eval_seal = base.read_json(producer.EVALUATION / "evaluation.provenance.json")
        eval_members = previous.member_index(eval_seal, producer.EVALUATION)
        cv2.setNumThreads(32)
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "evaluation must remain CPU")
        prepared.snapshot_sources(evidence, output, captured)
        candidates = {
            a: previous.read_arrays(producer.PREVIOUS, prior_members, f"{a}/candidate.npz")
            for a in previous.ARMS
        }
        candidates[producer.ARM] = previous.read_arrays(
            root, members, f"{producer.ARM}/candidate.npz"
        )
        support = metrics.summarize_support(candidates)
        core, content = validate_gap_state(root, members, common, points, gap_module, plan)
        chosen_rows = candidate_backtrace_control(candidates[producer.ARM], core)
        output.write(
            "diagnostics/gap_candidate_backtrace.npz", dict(flat_retained_rows=chosen_rows)
        )
        output.write(
            "diagnostics/gap_candidate_control.json",
            dict(
                passed=True,
                all21_fields=True,
                terminal_rank=0,
                numeric_fields_byte_exact=True,
                unicode_fields_value_exact=True,
                checkpoint_input_binding_equals_successful_plan=True,
                initialized_or_restored=False,
            ),
        )
        current_bank = old_evaluation.load_tree(root, members, f"{producer.ARM}/initial_anchors")
        previous.same_tree(current_bank, bank, "gap/prior entire causal bank")
        shapes = {}
        for arm, candidate in candidates.items():
            values = geometry._geometry(
                candidate["pose_xyz"], candidate["observed"], common["initial"]
            )
            output.write(f"diagnostics/{arm}_body_geometry.npz", values)
            shapes[arm] = geometry._geometry_summary(values)
        retained = geometry._geometry(
            core["history_pose_xyz"], core["history_observed"], common["initial"]
        )
        output.write("diagnostics/gap_all_retained_geometry.npz", retained)
        output.write(
            "diagnostics/gap_all_retained_summary.json",
            dict(
                rows=int(core["history_offsets"][-1]), bodies=geometry._geometry_summary(retained)
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
                previous_tracking_sha256=producer.PREVIOUS_SHA,
                previous_evaluation_sha256=producer.EVALUATION_SHA,
                samples_sha256=previous.SAMPLES_SHA,
                rendering_device="cpu",
                rendering_dtype="float64",
                width_ratio=0.06,
            ),
        )
        for offset, frame in enumerate(common["reader"]):
            native = frame.frame_idx
            sample = previous.read_arrays(
                previous.SAMPLES, sample_members, f"samples/native_{native}.npz"
            )
            previous.validate_samples(sample, native)
            tree = old_evaluation.load_tree(root, members, f"{producer.ARM}/frames/native_{native}")
            previous.same_tree(tree["fit"], content["fits"][offset], "gap full frame/sidecar fit")
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
                producer.EVALUATION, eval_members, f"frames/native_{native}.npz"
            )
            previous_image_control(scores, saved)
            output.write(f"frames/native_{native}.npz", scores)
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
