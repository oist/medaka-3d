"""Prepare all200 x two evidence methods x all causal initial body anchors.

Uses sealed observations/masks, saved first-release GFTT coordinates and fixed
fit pixels only. It does not render a pose, fit, restore a tracker or run a model.
The existing common loader decodes three registered background PNGs; this stage
never decodes the600 native RGB PNGs. Point diagnostics from these fit-mass
packets are in-sample; independent256-pixel image evaluation remains separate.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import body_point_transport as body
import cv2
import evaluate_mask_transport200 as old_evaluation
import numpy as np
import run_mask_transport_gap200 as controls
import torch

previous = controls.previous
base, prepared, inputs = previous.base, previous.prepared, previous.inputs
WORK, REPO, FIRST, STOP, COUNT = previous.WORK, previous.REPO, previous.FIRST, previous.STOP, 200
VIEWS, require = body.VIEWS, body.require
EVIDENCE = ("lk", "ct_first")
STAGE = "body_point200_packet_preparation_v1"
PLAN = WORK / "NEWBODY_POINT_EVIDENCE_FOLLOWUP.md"
PLAN_SHA = body.PLAN_SHA256
HELPER_SHA = "a532a395d12f8ee7019bfed27d1b2850bed86e5f6f38dad002c52c646f6c2406"
SOURCE_PINS = {
    **previous.SOURCE_PINS,
    **controls.SOURCE_PINS,
    "work/three_body/run_mask_transport_gap200.py": (
        "af36a32adc4be70acda870c6037d7fb6a3d7f8a53796ba9fb0c1e4e7bc1991b4"
    ),
    "work/three_body/body_point_transport.py": HELPER_SHA,
    "work/three_body/mask_transport_evaluation.py": (
        "4b5aa36ed3660586ca94b275742bb69edbc807da04ebfb8de4d7084c1a5676e7"
    ),
    "work/three_body/mask_transport_diagnostics.py": (
        "6567710637b4a3b69dd03106cc525a1f7fac1659a7ab54acadef3c98517f3e56"
    ),
    "work/three_body/temporal_point_core.py": (
        "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b"
    ),
}


def load_inputs(evidence):
    for relative, digest in SOURCE_PINS.items():
        evidence.add(REPO / relative, digest)
    evidence.add(Path(__file__).resolve())
    evidence.add(PLAN, PLAN_SHA)
    prior_plan, initial_records, saved_anchors, bank = controls.load_controls(evidence)
    common, members, saved_points, _ = previous.load_inputs(evidence, previous.SAMPLES_SHA)
    require(common["original_context"] == prior_plan["original_context"], "same science context")
    require(
        1 <= len(bank["facts"]) == len(saved_anchors) == len(initial_records) <= 6,
        "actual complete causal t0 bank required",
    )
    anchors = []
    for index, (fact, record) in enumerate(zip(bank["facts"], initial_records, strict=True)):
        require(set(fact["record"]) == set(record) == set(previous.RECORD_FIELDS), "t0 all19")
        for key in previous.RECORD_FIELDS:
            previous.tracking.same_record(fact["record"][key], record[key], "bank t0 " + key)
        anchor = body.old.build_initial_anchor(
            tuple(fact["references"]),
            saved_points.group_counts,
            anchor_id=index,
            initial_native=FIRST,
        )
        body._same(anchor, bank["anchors"][index], "same original causal anchor")
        anchors.append(anchor)
    return common, members, body.SavedBodyPointEvidence(saved_points), tuple(anchors), bank


def mask_frame(reader, native):
    """Read each saved observation once; never call reader.at or decode RGB."""
    require(type(native) is int and FIRST <= native < STOP, "fixed native200 required")
    masks = {}
    for view in VIEWS:
        row = reader.rows[native, view]
        arrays = prepared.arrays_from_bytes(
            reader.read(row["observation_path"], row["observation_sha256"])
        )
        inputs.validate_observation(arrays, row)
        masks[view] = arrays["masks"] if bool(arrays["mask_available"]) else None
    return inputs.FrameObservation(native, {}, {}, masks, {})


def packet_pair(frame, samples, point_reader, anchors, rig):
    """One frame, both fixed methods, old f32 target y (not its bool companion)."""
    previous.same(
        samples["frame_idx"], np.asarray(frame.frame_idx, np.int64), "packet sample native"
    )
    pixels, target, weights = (samples[k] for k in ("fit_pixels", "fit_target", "fit_weights"))
    require(
        pixels.dtype == np.float64
        and target.dtype == weights.dtype == np.float32
        and pixels.shape[:2] == target.shape == weights.shape
        and pixels.shape[0] == 3
        and pixels.shape[-1] == 2,
        "original f64 pixels/f32 target and weights required",
    )
    # The frozen sample NPZ stores UNUSED ray validity, not fit-ray validity.
    # Recompute fit validity from the same calibrated pixels; no pose render.
    _, _, fit_valid = body._rays(rig, pixels, np.full(3, pixels.shape[1], np.int64))
    point_reader.prepare(frame)
    result = {}
    for method in EVIDENCE:
        points = point_reader.provider(method)(frame.frame_idx)
        packet = body.build_packet(
            points,
            anchors,
            tuple(frame.masks[v] for v in VIEWS),
            pixels,
            fit_foreground=target,
            fit_weights=weights,
            fit_ray_valid=fit_valid,
            rig=rig,
            native_shape_hw=next(m.shape[1:] for m in frame.masks.values() if m is not None)
            if any(m is not None for m in frame.masks.values())
            else (1200, 1200),
        )
        # Independent reduction/order matching the old saved-image scorer.
        old_t = packet.whole_mask_packet.target
        expected_mass = (
            old_t * ((weights.astype(np.float64) * fit_valid) * target)[None, :, :, None]
        ).sum(2)
        previous.same(packet.old_mass, expected_mass, "unchanged whole-mask m")
        require(packet.native == frame.frame_idx, "body packet native differs")
        result[method] = packet
    left, right = result["lk"], result["ct_first"]
    for key in (
        "anchor_ids",
        "pixels",
        "release_horizons",
        "fit_foreground",
        "fit_weights",
        "fit_ray_valid",
    ):
        previous.same(getattr(left, key), getattr(right, key), "both evidence same " + key)
    require(left.query_sha256 == right.query_sha256, "evidence initial queries differ")
    return result


def save_checked_packet(output, name, packet, rig):
    previous.save_tree(output, name, packet)
    members = {row["path"]: row for row in output.records.values()}
    restored = old_evaluation.load_tree(output.root, members, name)
    checked = body.validate_packet(restored, rig=rig)
    body._same(packet, checked, "saved packet readback")
    return checked


def packet_statistics(method, packet):
    votes = [e.vote for e in packet.point_evidence]
    raw = []
    for v, (ev, vote) in enumerate(zip(packet.point_evidence, votes, strict=True)):
        count = len(ev.xy)
        finite = np.isfinite(ev.xy).all(axis=1)
        raw.append(
            dict(
                view=VIEWS[v],
                query_count=count,
                group_count=vote.group_count,
                reported_count=int(vote.reported.sum()),
                mask_available=vote.mask_available,
                mask_count=vote.membership.shape[1],
                reported_by_group=vote.reported_count.tolist(),
                single_by_group=vote.single_count.tolist(),
                multi_by_group=vote.multi_count.tolist(),
                unassigned_by_group=vote.unassigned_count.tolist(),
                nonfinite_points=int((~finite).sum()),
                native_outside_finite_points=int((finite & ~packet.native_inside[v, :count]).sum()),
                ray_invalid_points=int((~packet.ray_valid[v, :count]).sum()),
            )
        )
    return dict(
        evidence=method,
        native=packet.native,
        used_in_tracking=packet.native > FIRST,
        release_horizons=packet.release_horizons.tolist(),
        raw_views=raw,
        old_mass=packet.old_mass.tolist(),
        effective_mass=packet.effective_mass.tolist(),
        single_count=packet.single_count.tolist(),
        valid_single_count=packet.valid_single_count.tolist(),
        admitted_by_anchor_view_group=[
            [d.group_admitted.tolist() for d in row] for row in packet.whole_mask_packet.assignments
        ],
        reasons_by_anchor_view_group=[
            [[list(r) for r in d.group_reasons] for d in row]
            for row in packet.whole_mask_packet.assignments
        ],
    )


def summarize(rows, anchors):
    result = {}
    for method in EVIDENCE:
        selected = [row for row in rows if row["evidence"] == method]
        require([r["native"] for r in selected] == list(range(FIRST, STOP)), "all native200 rows")
        raw = [row["raw_views"] for row in selected]
        mass = np.asarray([row["old_mass"] for row in selected], np.float64)
        effective = np.asarray([row["effective_mass"] for row in selected], np.float64)
        single = np.asarray([row["single_count"] for row in selected], np.int64)
        valid = np.asarray([row["valid_single_count"] for row in selected], np.int64)
        reasons = [Counter() for _ in anchors]
        admitted = np.zeros(len(anchors), np.int64)
        for row in selected:
            for a, views in enumerate(row["reasons_by_anchor_view_group"]):
                for v, view in enumerate(views):
                    flags = row["admitted_by_anchor_view_group"][a][v]
                    require(len(flags) == len(view), "admitted/reason group counts differ")
                    for flag, group in zip(flags, view, strict=True):
                        require(
                            type(flag) is bool and flag == (not group), "admission/reason differs"
                        )
                        reasons[a].update(group)
                        admitted[a] += flag

        def sum_raw(key, frames=raw):
            return [sum(frame[v][key] for frame in frames) for v in range(3)]

        result[method] = dict(
            native_frames=len(selected),
            packets=len(selected),
            all_query_times=sum(sum_raw("query_count")),
            tracking_query_times=sum(sum_raw("query_count"))
            - sum(r["query_count"] for r in raw[0]),
            query_times_by_view=sum_raw("query_count"),
            reported_times_by_view=sum_raw("reported_count"),
            group_times_per_anchor=sum(
                sum(frame[v]["group_count"] for v in range(3)) for frame in raw
            ),
            body_view_times_per_anchor=len(selected) * 9,
            mask_missing_view_times=[
                sum(not f[v]["mask_available"] for f in raw) for v in range(3)
            ],
            mask_available_empty_view_times=[
                sum(f[v]["mask_available"] and f[v]["mask_count"] == 0 for f in raw)
                for v in range(3)
            ],
            ray_invalid_query_times_by_view=sum_raw("ray_invalid_points"),
            nonfinite_query_times_by_view=sum_raw("nonfinite_points"),
            outside_finite_query_times_by_view=sum_raw("native_outside_finite_points"),
            admitted_group_times_by_anchor=admitted.tolist(),
            reason_counts_by_anchor=[dict(c) for c in reasons],
            old_mass_sum_by_anchor_view_body=mass.sum(0).tolist(),
            effective_mass_sum_by_anchor_view_body=effective.sum(0).tolist(),
            zero_old_mass_body_view_times_by_anchor=(mass == 0).sum((0, 2, 3)).tolist(),
            positive_old_mass_body_view_times_by_anchor=(mass > 0).sum((0, 2, 3)).tolist(),
            selected_single_point_times_by_anchor=single.sum((0, 2, 3)).tolist(),
            valid_selected_point_times_by_anchor=valid.sum((0, 2, 3)).tolist(),
        )
    return result


def run(destination):
    destination = Path(destination).resolve()
    require(destination.is_relative_to(WORK) and not destination.exists(), "new workspace output")
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, rows = base.utc(), time.perf_counter(), []
    old_cv = cv2.getNumThreads()
    timings = dict(
        input_seconds=0.0,
        observation_seconds=0.0,
        build_seconds=0.0,
        save_readback_seconds=0.0,
        final_guard_seconds=0.0,
    )
    try:
        cv2.setNumThreads(32)
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), "packet preparation must remain CPU")
        at = time.perf_counter()
        common, sample_members, points, anchors, bank = load_inputs(evidence)
        timings["input_seconds"] = time.perf_counter() - at
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                native_interval=[FIRST, STOP],
                views=list(VIEWS),
                evidence=list(EVIDENCE),
                implementation_plan_sha256=PLAN_SHA,
                helper_sha256=HELPER_SHA,
                source_pins=SOURCE_PINS,
                preparation_sha256=inputs.PREPARATION_SHA,
                point_sha256=previous.POINTS_SHA,
                samples_sha256=previous.SAMPLES_SHA,
                causal_tracking_sha256=controls.PREVIOUS_SHA,
                causal_prior_evaluation_sha256=controls.EVALUATION_SHA,
                all_initial_anchor_ids=[a.anchor_id for a in anchors],
                query_sha256=list(points.query_sha256),
                group_counts=list(points.group_counts),
                fit_samples_per_view=1024,
                point_mass_domain="same fit1024 W*y*T",
                point_diagnostic="in_sample; no unused-mass reconstruction",
                t0_packet_saved=True,
                t0_packet_used_in_tracking=False,
                original_context=common["original_context"],
                native_RGB_decode=False,
                common_background_decodes=3,
                candidate_pose_input=False,
                no_model=True,
                no_fit=True,
                no_GPU=True,
            ),
        )
        output.write("query_rows.json", points.saved.query_rows)
        previous.save_tree(output, "initial_bank", bank)
        for native in range(FIRST, STOP):
            at = time.perf_counter()
            frame = mask_frame(common["reader"], native)
            sample = previous.read_arrays(
                previous.SAMPLES, sample_members, f"samples/native_{native}.npz"
            )
            previous.validate_samples(sample, native)
            timings["observation_seconds"] += time.perf_counter() - at
            at = time.perf_counter()
            packets = packet_pair(frame, sample, points, anchors, common["rig"])
            timings["build_seconds"] += time.perf_counter() - at
            at = time.perf_counter()
            for method, packet in packets.items():
                checked = save_checked_packet(
                    output, f"packets/{method}/native_{native}", packet, common["rig"]
                )
                rows.append(packet_statistics(method, checked))
            timings["save_readback_seconds"] += time.perf_counter() - at
        require(len(rows) == COUNT * 2, "all400 saved/readback packets required")
        summaries = summarize(rows, anchors)
        require(
            all(
                r["all_query_times"] == 66800 and r["group_times_per_anchor"] == 2000
                for r in summaries.values()
            ),
            "fixed334 queries/all10 groups required",
        )
        mask_counts = {
            method: sum(
                view["mask_count"]
                for row in rows
                if row["evidence"] == method
                for view in row["raw_views"]
            )
            for method in EVIDENCE
        }
        require(
            mask_counts["lk"] == mask_counts["ct_first"] == 1730,
            "all prepared1730 mask rows required",
        )
        output.write("packet_statistics.json", rows)
        prepared.snapshot_sources(evidence, output, captured)
        at = time.perf_counter()
        evidence.final_guard()
        output.final_guard()
        timings["final_guard_seconds"] = time.perf_counter() - at
        require(not torch.cuda.is_initialized(), "packet preparation initialized CUDA")
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            native_frames=200,
            evidence=list(EVIDENCE),
            packet_count=400,
            all_initial_anchor_ids=[a.anchor_id for a in anchors],
            observation_NPZ_reads=600,
            native_RGB_decodes=0,
            common_background_decodes=3,
            mask_count=mask_counts["lk"],
            mask_count_by_evidence=mask_counts,
            methods=summaries,
            all_old_T_exact=True,
            all_old_mass_exact=True,
            all400_packet_readbacks_exact=True,
            all400_query_ray_readbacks_exact=True,
            fit_mass_point_diagnostic_in_sample=True,
            unused_mass_reconstruction=False,
            new_model=False,
            new_fit=False,
            tracker_restore_calls=0,
            CUDA_initialized=False,
            timings=timings,
            save_timing_includes_SHA_and_readback=True,
            elapsed_through_guards_seconds=time.perf_counter() - start,
        )
        output.write("summary.json", summary)
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            params=dict(all_guards_passed=True),
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
            completed_packets=len(rows),
            error_type=type(error).__name__,
            error=str(error),
        )
        with (destination / "preparation.provenance.json").open("xb") as stream:
            stream.write(base.json_bytes(seal))
        raise
    finally:
        cv2.setNumThreads(old_cv)
    with (destination / "preparation.provenance.json").open("xb") as stream:
        stream.write(base.json_bytes(seal))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output)), flush=True)


if __name__ == "__main__":
    main()
