"""Seven-arm fixed200 evaluation after completed body-point tracking and packet preparation.

Old six trajectories/state audits are reused through their successful seals and
all200 exact common image scores. Only the new core/sidecar is semantically
decoded. Body-point scores use saved FIT1024-derived mass and are IN-SAMPLE;
common unused256 image scores and current SL support are reported separately.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import cv2
import evaluate_mask_transport200 as old_evaluation
import evaluate_mask_transport_gap200 as old_gap_evaluation
import mask_transport_diagnostics as geometry
import numpy as np
import object_mask_transport as transport
import run_body_point200 as producer
import torch

previous = producer.previous
base, prepared, inputs = producer.base, producer.prepared, producer.inputs
WORK, REPO, FIRST, STOP = producer.WORK, producer.REPO, producer.FIRST, producer.STOP
require = producer.require
STAGE = "body_point200_independent_evaluation_v1"
ARMS = ("original", "lk", "ct_first", "ct_gap", "static_base", "static_gap", "ct_body_points")
EVIDENCE = ("lk", "ct_first")
PACKET_STAGE = "body_point200_packet_preparation_v1"
METRICS_SHA = "d31aa9a4d89039b3d86c481e6e90fad3ce6ff7f9e3ef639fbca793d96aec38a4"
AUDIT_SHA = "df730569610cb81cd42342b930594e5e167653831e42313489d5d5c228ff7678"
TRACKER_SHA = "ca4ccc28bfb529c198e5079e78ff5897af41b878fe98fcfc65d7cf4333742877"
POINT_HELPER_SHA = "a532a395d12f8ee7019bfed27d1b2850bed86e5f6f38dad002c52c646f6c2406"
# Final producer/preparation are pinned; actual evaluation still requires its success seal.
PRODUCER_SHA = "9c8d9a24ebfb98e2346d4b028d1963a2f41a17afb7f41cd35cbab0b412e483e6"
PACKET_PRODUCER_SHA = "73bbaaf11fe4186e72e2fea310c7e7f4ff6cf604e27641c68c0e2024592089ca"
PACKET_SHA = "8e2f31d5ac9364b72ea899bbc06a48e6af9bccf15fc6de5760179677095d3862"
EXECUTION_READY = True
SIX_ROOT = WORK / "fixed_body_width200_independent_evaluation_v1"
SIX_SHA = "a5b95006cf407e70e03a4368eba9576ea2df1bfebb9fe0c64c6b3bfb01b670db"
SIX_SOURCE_SHA = "ab1659274444a077d9b12376717bf1c3563c7433854f1d16196bc46599fa59cf"
OLD_SOURCES = (
    (producer.PREVIOUS, producer.PREVIOUS_SHA, previous.STAGE, ("original", "lk", "ct_first")),
    (
        WORK / "mask_transport_gap200_tracking_v1",
        "a3222e383ce5e08b63a2f4bba8a2404da95cf27f1e6513be44f2f90e4aaf5aa9",
        "mask_transport_gap200_tracking_v1",
        ("ct_gap",),
    ),
    (
        WORK / "fixed_body_width200_tracking_v2",
        "f0381888f90e9f56ae41d76fb287148a44746b94f1222ca337bef0e25d0b9dda",
        "fixed_body_width200_tracking_v2",
        ("static_base", "static_gap"),
    ),
)
SOURCE_PINS = {
    "work/three_body/body_point_evaluation.py": METRICS_SHA,
    "work/three_body/body_point_state_audit.py": AUDIT_SHA,
    "work/three_body/body_point_tracker.py": TRACKER_SHA,
    "work/three_body/body_point_transport.py": POINT_HELPER_SHA,
    "work/three_body/evaluate_fixed_body_width200.py": SIX_SOURCE_SHA,
    "work/three_body/evaluate_mask_transport_gap200.py": (
        "eb26fe9c67463ac004e88fae1921536668c435f1305b8bdd19973de679000e4e"
    ),
    "work/three_body/evaluate_mask_transport200.py": (
        "61c94354a8f0c1618234cedfdb2a0640b08a62fc66297bc60070a4021bbfdf0e"
    ),
    "work/three_body/mask_transport_diagnostics.py": old_evaluation.DIAGNOSTICS_SHA,
}


def input_member(seal, path, digest, label):
    require(seal["inputs"].get(str(path)) == digest, label + " input SHA membership differs")


def ready(digest, packet_digest):
    require(
        EXECUTION_READY
        and all(base.valid_sha(s) for s in (digest, PRODUCER_SHA, PACKET_PRODUCER_SHA, PACKET_SHA))
        and packet_digest == PACKET_SHA
        and PACKET_SHA == producer.PACKETS_SHA
        and base.valid_sha(producer.QUALIFICATION_SHA)
        and all(base.valid_sha(s) for s in producer.SOURCE_PINS.values()),
        "final reviewed producer/preparation pins required; draft evaluation is not executable",
    )


def validate_run(plan, summary, initial):
    require(
        plan["schema"] == summary["schema"] == producer.STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP],
        "successful complete body-point native200 required",
    )
    binding = plan["input_binding"]
    expected = dict(
        schema=producer.STAGE,
        arm=producer.ARM,
        point_evidence_arm="ct_first",
        preparation_sha256=inputs.PREPARATION_SHA,
        point_sha256=previous.POINTS_SHA,
        samples_sha256=previous.SAMPLES_SHA,
        prior_tracking_sha256=producer.PREVIOUS_SHA,
        prior_evaluation_sha256=producer.EVALUATION_SHA,
        prior_six_evaluation_sha256=SIX_SHA,
        body_point_preparation_sha256=PACKET_SHA,
        cuda_qualification_sha256=producer.QUALIFICATION_SHA,
        body_point_plan_sha256=producer.PLAN_SHA,
        source_pins=producer.SOURCE_PINS,
    )
    require(
        all(binding.get(k) == v for k, v in expected.items()),
        "body-point input/source binding differs",
    )
    require(
        plan["arm"] == summary["arm"]["arm"] == producer.ARM
        and plan["point_evidence_arm"] == "ct_first"
        and plan["objective"] == "base + 3*(body_point_owner + body_point_ray_gap)"
        and plan["mask_coefficient"] == 3.0
        and plan["gap_relative_coefficient"] == 1.0
        and plan["extra_proposals"] == 0
        and plan["maximum_pool"] == summary["maximum_pool"] == 24
        and plan["opt_steps"] == summary["steps"] == 14
        and plan["point_to_node_correspondence"] is False
        and plan["D2_missing_fallback"] is False
        and plan["first_release_horizons"] == [previous.horizon(n) for n in range(FIRST, STOP)],
        "fixed body-point policy/budget differs",
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
        "body-point lifecycle/initial control differs",
    )
    for key in (
        "prior_tracking_sha256",
        "prior_evaluation_sha256",
        "prior_six_evaluation_sha256",
        "body_point_preparation_sha256",
        "cuda_qualification_sha256",
    ):
        require(summary[key] == binding[key], "body-point summary lineage differs")


def validate_packet_preparation(plan, summary, seal):
    require(
        plan["schema"] == summary["schema"] == PACKET_STAGE
        and summary["status"] == "ok"
        and plan["native_interval"] == summary["native_interval"] == [FIRST, STOP]
        and plan["evidence"] == summary["evidence"] == list(EVIDENCE)
        and summary["native_frames"] == 200
        and summary["packet_count"] == 400
        and summary["mask_count"] == 1730,
        "completed fixed400 point packets required",
    )
    for key, value in dict(
        implementation_plan_sha256=producer.PLAN_SHA,
        helper_sha256=POINT_HELPER_SHA,
        preparation_sha256=inputs.PREPARATION_SHA,
        point_sha256=previous.POINTS_SHA,
        samples_sha256=previous.SAMPLES_SHA,
        causal_tracking_sha256=producer.PREVIOUS_SHA,
    ).items():
        require(plan[key] == value, "point preparation lineage differs: " + key)
    require(
        plan["fit_samples_per_view"] == 1024
        and plan["point_mass_domain"] == "same fit1024 W*y*T"
        and plan["point_diagnostic"] == "in_sample; no unused-mass reconstruction"
        and plan["t0_packet_saved"] is True
        and plan["t0_packet_used_in_tracking"] is False
        and all(
            summary[k] is True
            for k in (
                "all_old_T_exact",
                "all_old_mass_exact",
                "all400_packet_readbacks_exact",
                "all400_query_ray_readbacks_exact",
                "fit_mass_point_diagnostic_in_sample",
            )
        )
        and all(
            summary[k] is False
            for k in ("unused_mass_reconstruction", "new_model", "new_fit", "CUDA_initialized")
        ),
        "point preparation domain/readback proof differs",
    )
    for path, sha in (
        (WORK / "prepare_body_point200_packets.py", PACKET_PRODUCER_SHA),
        (WORK / "body_point_transport.py", POINT_HELPER_SHA),
        (inputs.PREPARATION / "preparation.provenance.json", inputs.PREPARATION_SHA),
        (previous.POINTS / "tracking.provenance.json", previous.POINTS_SHA),
        (previous.SAMPLES / "samples.provenance.json", previous.SAMPLES_SHA),
        (producer.PREVIOUS / "tracking.provenance.json", producer.PREVIOUS_SHA),
    ):
        input_member(seal, path, sha, "prepared point evidence")


def load_previous(evidence):
    """Only old candidate21/t0 are decoded; full old outputs are hash-verified."""
    seal, members = base.verified(
        SIX_ROOT,
        "evaluation.provenance.json",
        "fixed_body_width200_independent_evaluation_v1",
        SIX_SHA,
        evidence,
    )
    plan, summary = (base.bound_json(SIX_ROOT, members, k) for k in ("plan.json", "summary.json"))
    require(
        summary["status"] == "ok"
        and summary["arms"] == list(ARMS[:6])
        and summary["native_interval"] == [FIRST, STOP]
        and summary["strict_candidate_support_rechecks"] == 1200
        and summary["original_four_CPU_image_control_exact_frames"] == 200
        and summary["unused_pixel_draws"] == 153600
        and summary["trajectory_or_review_modified"] is False
        and plan["primary_renderer"] == "original_dynamic_all6"
        and plan["samples_sha256"] == previous.SAMPLES_SHA,
        "same completed six-arm independent image/state evidence required",
    )
    input_member(seal, WORK / "evaluate_fixed_body_width200.py", SIX_SOURCE_SHA, "six-arm source")
    audits = summary["width_reference_audits"]
    require(set(audits) == {"static_base", "static_gap"}, "both sealed width state audits required")
    for audit in audits.values():
        require(
            audit["passed"] is True
            and audit["core_arrays"] == 39
            and audit["history_frames"] == 200
            and audit["semantic_core_and_sidecar_verified"] is True
            and audit["initial_geometry_links_verified"] is True
            and audit["metadata_rewritten"] is False,
            "prior six-arm state evidence differs",
        )
    references = {}
    contexts = []
    old_members = None
    for root, digest, stage, arms in OLD_SOURCES:
        input_member(seal, root / "tracking.provenance.json", digest, "six-arm producer")
        _, source_members = base.verified(root, "tracking.provenance.json", stage, digest, evidence)
        source_plan = base.bound_json(root, source_members, "plan.json")
        contexts.append(source_plan["original_context"])
        for arm in arms:
            path = prepared.bound_path(root, source_members, f"{arm}/candidate.npz")
            input_member(
                seal, path, source_members[str(path)]["sha256"], "six-arm candidate " + arm
            )
            references[arm] = (root, source_members)
        if root == producer.PREVIOUS:
            old_members = source_members
    require(all(c == contexts[0] for c in contexts), "six-arm original science context differs")
    bank = old_evaluation.load_tree(producer.PREVIOUS, old_members, "original/initial_anchors")
    initial = old_evaluation.load_tree(
        producer.PREVIOUS, old_members, f"original/frames/native_{FIRST}"
    )
    return members, references, contexts[0], bank, initial["records"]


def previous_image_control(scores, saved):
    """Exactly the nineteen shared arrays; three old width-policy fields stay separate."""
    extra = {"renderer_policy", "fixed_radial_sigma_mm", "reference_pose_xyz"}
    require(
        len(scores) == 19 and set(saved) - extra == set(scores),
        "nineteen old common image fields required",
    )
    shared = {"target_mass", "positive_target_mass", "ray_valid", "anchor_ids", "frame_idx"}
    outer = {"union_loss", "union_per_view", "rendered_union", "rendered_ownership", "rendered_q"}
    for key, expected in scores.items():
        value = expected if key in shared else expected[:6] if key in outer else expected[:, :6]
        previous.same(value, saved[key], "six-arm CPU image control " + key)


def observation_frame(reader, native):
    """Use sealed detections/masks only; native RGB is not needed for scoring."""
    masks, detections = {}, {}
    for view in transport.VIEWS:
        row = reader.rows[native, view]
        values = prepared.arrays_from_bytes(
            reader.read(row["observation_path"], row["observation_sha256"])
        )
        inputs.validate_observation(values, row)
        masks[view] = values["masks"] if bool(values["mask_available"]) else None
        detections[view] = values["detections"]
    return inputs.FrameObservation(native, detections, {}, masks, {})


def support_recheck(candidates, offset, frame, rig):
    rays = old_evaluation.RayObservations.from_points(rig, frame.detections)
    result = {}
    for arm in ARMS:
        candidate = candidates[arm]
        source, observed, distance = old_evaluation.supported_nodes(
            candidate["pose_xyz"][offset], candidate["sources"][offset], rays, rig
        )
        previous.same(source, candidate["sources"][offset], "strict source " + arm)
        previous.same(observed, candidate["observed"][offset], "strict support " + arm)
        for vi in range(3):
            for node in range(3):
                used = source[:, vi, node]
                used = used[used >= 0]
                require(len(used) == len(np.unique(used)), "source row reused across bodies")
        result[arm] = dict(sources=source, observed=observed, ray_distance_mm=distance)
    return result


def validate_new_state(root, members, common, points, plan, audit_module):
    prefix = f"{producer.ARM}/checkpoint/"
    core = previous.read_arrays(root, members, prefix + "arrays.npz")
    meta = base.bound_json(root, members, prefix + "metadata.json")
    require(
        meta["context"]["mask_transport"]["input_binding"] == plan["input_binding"],
        "new checkpoint binding differs from successful plan",
    )
    side = previous.read_arrays(root, members, prefix + "side_arrays.npz")
    side_meta = base.bound_json(root, members, prefix + "side_metadata.json")
    return audit_module.validate_state(
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
        coefficient=3,
        expected_frames=200,
    )


def frame_control(tree, content, core, offset, sample):
    previous.same_tree(tree["fit"], content["fits"][offset], "new full frame/sidecar fit")
    previous.fit_sample_match(tree["fit"], sample)
    lo, hi = map(int, core["history_offsets"][offset : offset + 2])
    require(len(tree["records"]) == hi - lo, "new full retained row count differs")
    for record, index in zip(tree["records"], range(lo, hi), strict=True):
        require(set(record) == set(previous.RECORD_FIELDS), "all19 native record fields required")
        for key in previous.RECORD_FIELDS:
            previous.tracking.same_record(
                record[key], core["history_" + key][index], "new frame/core " + key
            )


def checked_packets(
    root, members, native, sample, frame, anchors, rig, point_module, fit, *, query_sha256
):
    result = []
    for method in EVIDENCE:
        raw = old_evaluation.load_tree(root, members, f"packets/{method}/native_{native}")
        packet = point_module.validate_packet(raw, rig=rig)
        require(
            packet.native == native and packet.initial_native == FIRST,
            "prepared packet native differs",
        )
        require(
            tuple(packet.query_sha256) == tuple(query_sha256),
            "prepared packet fixed query SHA differs",
        )
        previous.same(
            packet.release_horizons,
            np.full(3, previous.horizon(native), np.int64),
            "prepared native release horizon",
        )
        previous.same(packet.pixels, sample["fit_pixels"], "point packet original fit pixels")
        previous.same(packet.fit_foreground, sample["fit_target"], "point packet actual f32 target")
        previous.same(packet.fit_weights, sample["fit_weights"], "point packet fit weights")
        _, _, valid = point_module._rays(
            rig, sample["fit_pixels"], np.full(3, sample["fit_pixels"].shape[1], np.int64)
        )
        previous.same(packet.fit_ray_valid, valid, "point packet original calibrated fit rays")
        point_module._same(
            packet.whole_mask_packet.anchors, anchors, "prepared complete initial bank"
        )
        for v, view in enumerate(transport.VIEWS):
            masks = frame.masks[view]
            require(
                packet.whole_mask_packet.mask_signatures[v]
                == (None if masks is None else transport.signature(masks)),
                "point packet native all-N masks differ",
            )
            actual = point_module.mask_memberships(
                packet.point_evidence[v].xy, masks, packet.whole_mask_packet.native_shape_hw
            )
            previous.same(
                actual["membership"],
                packet.point_evidence[v].vote.membership,
                "current point/mask membership",
            )
        if method == "ct_first":
            if native == FIRST:
                require(fit["packet"] is None, "t0 point packet cannot enter fitting")
            else:
                point_module._same(packet, fit["packet"], "tracking/prepared CT packet exact")
        result.append(packet)
    return tuple(result)


def paired_scores(rows):
    result = {}
    union = np.stack([r["union_loss"] for r in rows])
    views = np.stack([r["union_per_view"] for r in rows])
    for reference in ("original", "ct_gap"):
        i = ARMS.index(reference)
        delta = union[:, -1] - union[:, i]
        result[reference] = dict(
            baseline=reference,
            arm=producer.ARM,
            per_native_delta=delta,
            per_native_view_delta=views[:, -1] - views[:, i],
            mean_delta=float(delta.mean()),
            better=int((delta < 0).sum()),
            equal=int((delta == 0).sum()),
            worse=int((delta > 0).sum()),
        )
    return result


def support_pair(candidates, reference):
    old = candidates[reference]["observed"]
    new = candidates[producer.ARM]["observed"]
    gained = new & ~old
    lost = old & ~new
    return dict(
        baseline=reference,
        arm=producer.ARM,
        gains=int(gained.sum()),
        losses=int(lost.sum()),
        gain_by_body_node=gained.sum(0).tolist(),
        loss_by_body_node=lost.sum(0).tolist(),
        gain_rows=[
            dict(native=FIRST + int(t), body=int(b), node=int(n)) for t, b, n in np.argwhere(gained)
        ],
        loss_rows=[
            dict(native=FIRST + int(t), body=int(b), node=int(n)) for t, b, n in np.argwhere(lost)
        ],
    )


def report(summary):
    lines = [
        "# 身体追跡点の固定200診断",
        "",
        "共通未使用256画素の全7方式比較。点の損失は元fit1024から固定したmassを使うIN-SAMPLE診断であり、独立精度や生物IDの正解ではない。",
        "",
        "|方式|共通画像誤差|口支持|胸支持|尾支持|",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        c = summary["support"][arm]["direct_count_by_node"]
        lines.append(f"|{arm}|{summary['union'][arm]['mean']:.9f}|{c[0]}|{c[1]}|{c[2]}|")
    lines += [
        "",
        "全199 CT fit packetと準備証拠の一致、全7方式の現在SL支持、全保持形状、"
        "全evidence・全anchorの点損失を保存。t0 packetは採点のみで初期fitには使っていない。",
    ]
    return ("\n".join(lines) + "\n").encode()


def run(root, digest, packet_root, packet_digest, destination):
    ready(digest, packet_digest)
    root, packet_root, destination = map(
        lambda p: Path(p).resolve(), (root, packet_root, destination)
    )
    require(
        all(p.is_relative_to(WORK) for p in (root, packet_root, destination))
        and not destination.exists(),
        "workspace inputs and new output required",
    )
    output, evidence, captured = base.Output(destination), base.Evidence(), {}
    started, start, rows, point_rows = base.utc(), time.perf_counter(), [], []
    old_cv, old_threads = cv2.getNumThreads(), torch.get_num_threads()
    try:
        require(not torch.cuda.is_initialized(), "CPU evaluation cannot inherit initialized CUDA")
        cv2.setNumThreads(32)
        torch.set_num_threads(2)
        evidence.add(Path(__file__).resolve())
        for relative, sha in {
            **previous.SOURCE_PINS,
            **producer.SOURCE_PINS,
            **SOURCE_PINS,
            "work/three_body/run_body_point200.py": PRODUCER_SHA,
            "work/three_body/prepare_body_point200_packets.py": PACKET_PRODUCER_SHA,
        }.items():
            evidence.add(REPO / relative, sha)
        metrics = importlib.import_module("body_point_evaluation")
        audit = importlib.import_module("body_point_state_audit")
        point_module = importlib.import_module("body_point_transport")
        seal, members = base.verified(
            root, "tracking.provenance.json", producer.STAGE, digest, evidence
        )
        for relative, sha in {
            **producer.SOURCE_PINS,
            "work/three_body/run_body_point200.py": PRODUCER_SHA,
        }.items():
            input_member(seal, REPO / relative, sha, "new producer source")
        plan, summary, initial = (
            base.bound_json(root, members, k)
            for k in ("plan.json", "summary.json", "initial_control.json")
        )
        validate_run(plan, summary, initial)
        for path, sha in (
            (producer.PREVIOUS / "tracking.provenance.json", producer.PREVIOUS_SHA),
            (SIX_ROOT / "evaluation.provenance.json", SIX_SHA),
            (inputs.PREPARATION / "preparation.provenance.json", inputs.PREPARATION_SHA),
            (previous.POINTS / "tracking.provenance.json", previous.POINTS_SHA),
            (previous.SAMPLES / "samples.provenance.json", previous.SAMPLES_SHA),
            (packet_root / "preparation.provenance.json", packet_digest),
            (producer.QUALIFICATION / "qualification.provenance.json", producer.QUALIFICATION_SHA),
        ):
            input_member(seal, path, sha, "new trajectory evidence")
        require(packet_root == producer.PACKETS, "registered prepared packet root required")
        producer.load_qualification(evidence)
        packet_seal, packet_members = base.verified(
            packet_root, "preparation.provenance.json", PACKET_STAGE, packet_digest, evidence
        )
        packet_plan, packet_summary = (
            base.bound_json(packet_root, packet_members, k) for k in ("plan.json", "summary.json")
        )
        validate_packet_preparation(packet_plan, packet_summary, packet_seal)
        eval_members, references, old_context, bank, old_initial = load_previous(evidence)
        common, sample_members, points, _ = previous.load_inputs(evidence, previous.SAMPLES_SHA)
        require(
            common["original_context"]
            == plan["original_context"]
            == packet_plan["original_context"]
            == old_context,
            "same original science context required",
        )
        require(
            packet_plan["group_counts"] == list(points.group_counts)
            and packet_plan["query_sha256"] == list(points.query_sha256),
            "prepared fixed query table differs",
        )
        previous.same_tree(
            old_evaluation.load_tree(packet_root, packet_members, "initial_bank"),
            bank,
            "prepared causal initial bank",
        )
        previous.same_tree(
            old_evaluation.load_tree(root, members, f"{producer.ARM}/initial_anchors"),
            bank,
            "new causal initial bank",
        )
        references[producer.ARM] = (root, members)
        candidates = {a: previous.read_arrays(*references[a], f"{a}/candidate.npz") for a in ARMS}
        expected_fields = (set(previous.RECORD_FIELDS) - {"parent"}) | {
            "xyz",
            "accepted_xyz",
            "body_xyz",
        }
        require(
            all(set(c) == expected_fields for c in candidates.values()),
            "all seven candidate21 schemas required",
        )
        support = metrics.summarize_support(candidates)
        core, content, state_report = validate_new_state(root, members, common, points, plan, audit)
        selected = old_gap_evaluation.candidate_backtrace_control(candidates[producer.ARM], core)
        output.write("diagnostics/new_candidate_backtrace.npz", dict(flat_retained_rows=selected))
        output.write("diagnostics/new_state_audit.json", state_report)
        require(
            len(old_initial) == len(content["anchor_facts"]) == len(bank["facts"]),
            "complete original t0 retained bank required",
        )
        for a, record in enumerate(old_initial):
            for key in previous.RECORD_FIELDS:
                previous.tracking.same_record(
                    content["anchor_facts"][a]["record"][key], record[key], "new t0 original19"
                )
        anchors = tuple(
            transport.build_initial_anchor(
                tuple(f["references"]), points.group_counts, anchor_id=i, initial_native=FIRST
            )
            for i, f in enumerate(bank["facts"])
        )
        require(
            packet_plan["all_initial_anchor_ids"]
            == packet_summary["all_initial_anchor_ids"]
            == list(range(len(anchors))),
            "prepared all initial anchors differ",
        )
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
        output.write("diagnostics/new_all_retained_geometry.npz", retained)
        output.write(
            "diagnostics/new_all_retained_summary.json",
            dict(
                rows=int(core["history_offsets"][-1]), bodies=geometry._geometry_summary(retained)
            ),
        )
        prepared.snapshot_sources(evidence, output, captured)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                source_tracking_sha256=digest,
                packet_preparation_sha256=packet_digest,
                previous_six_evaluation_sha256=SIX_SHA,
                native_interval=[FIRST, STOP],
                arms=list(ARMS),
                evidence=list(EVIDENCE),
                all_initial_anchor_ids=list(range(len(anchors))),
                samples_sha256=previous.SAMPLES_SHA,
                rendering_device="cpu",
                rendering_dtype="float64",
                renderer="original_dynamic_all7",
                point_mass_domain="saved FIT1024 m/S; IN-SAMPLE",
                point_t0_saved_not_fitted=True,
                candidate_references=[
                    dict(
                        arm=a,
                        path=str(prepared.bound_path(*references[a], f"{a}/candidate.npz")),
                        sha256=references[a][1][
                            str(prepared.bound_path(*references[a], f"{a}/candidate.npz"))
                        ]["sha256"],
                    )
                    for a in ARMS
                ],
                old_state_arrays_deserialized=False,
                original_control_horizon=STOP,
                raw_native_RGB_decodes=0,
            ),
        )
        for offset, native in enumerate(range(FIRST, STOP)):
            frame = observation_frame(common["reader"], native)
            sample = previous.read_arrays(
                previous.SAMPLES, sample_members, f"samples/native_{native}.npz"
            )
            previous.validate_samples(sample, native)
            tree = old_evaluation.load_tree(root, members, f"{producer.ARM}/frames/native_{native}")
            frame_control(tree, content, core, offset, sample)
            checked = support_recheck(candidates, offset, frame, common["rig"])
            packets = checked_packets(
                packet_root,
                packet_members,
                native,
                sample,
                frame,
                anchors,
                common["rig"],
                point_module,
                tree["fit"],
                query_sha256=points.query_sha256,
            )
            image_packets = tuple(
                transport.packet_for_pixels(
                    tuple(e.vote for e in p.point_evidence),
                    anchors,
                    tuple(frame.masks[v] for v in transport.VIEWS),
                    sample["unused_pixels"],
                )
                for p in packets
            )
            poses = np.stack([candidates[a]["pose_xyz"][offset] for a in ARMS])
            scores = metrics.score_fixed_images(poses, sample, image_packets, common["rig"])
            saved = previous.read_arrays(SIX_ROOT, eval_members, f"frames/native_{native}.npz")
            previous_image_control(scores, saved)
            point_scores = metrics.score_body_points(poses, packets, common["rig"])
            output.write(f"frames/native_{native}.npz", scores)
            output.write(f"point_frames/native_{native}.npz", point_scores)
            previous.save_tree(
                output,
                f"evidence/native_{native}",
                dict(
                    image_packets=image_packets,
                    strict_SLEAP=checked,
                    prepared_point_prefixes=[
                        str(packet_root / f"packets/{method}/native_{native}")
                        for method in EVIDENCE
                    ],
                ),
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
            point_rows.append(
                {
                    k: point_scores[k]
                    for k in (
                        "point_owner_loss",
                        "point_gap_loss",
                        "point_combined_loss",
                        "point_mass_by_view_body",
                    )
                }
            )
        require(len(rows) == len(point_rows) == 200, "complete native200 metrics required")
        union = np.stack([r["union_loss"] for r in rows])
        views = np.stack([r["union_per_view"] for r in rows])
        paired = paired_scores(rows)
        previous.save_tree(output, "paired_image_scores", paired)
        result = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            arms=list(ARMS),
            evidence=list(EVIDENCE),
            union={
                a: dict(mean=float(union[:, i].mean()), per_view_mean=views[:, i].mean(0).tolist())
                for i, a in enumerate(ARMS)
            },
            support=support,
            paired_support={ref: support_pair(candidates, ref) for ref in ("original", "ct_gap")},
            body_geometry=shapes,
            paired_images={
                ref: {k: v for k, v in values.items() if not isinstance(v, np.ndarray)}
                for ref, values in paired.items()
            },
            owner_mean_evidence_arm_anchor=np.stack([r["owner_loss"] for r in rows])
            .mean(0)
            .tolist(),
            gap_mean_evidence_arm_anchor=np.stack([r["gap_loss"] for r in rows]).mean(0).tolist(),
            combined_mean_evidence_arm_anchor=np.stack([r["combined_loss"] for r in rows])
            .mean(0)
            .tolist(),
            point_means={
                k: np.stack([r[k] for r in point_rows]).mean(0).tolist() for k in point_rows[0]
            },
            point_metrics_are_IN_SAMPLE=True,
            point_evaluation_packets=400,
            tracking_CT_packet_exact_frames=199,
            t0_packet_used_in_fit=False,
            previous_six_common19_exact_frames=200,
            strict_candidate_support_rechecks=1400,
            unused_pixel_draws=153600,
            new_state_audit=state_report,
            old_states_reinitialized_or_restored=False,
            raw_native_RGB_decodes=0,
            GPU_initialized=False,
            trajectory_or_review_modified=False,
            independent_ID_truth=False,
            temporal_holdout=False,
            elapsed_before_guards_seconds=time.perf_counter() - start,
        )
        output.write("summary.json", result)
        output.write("REPORT.md", report(result))
        prepared.snapshot_sources(evidence, output, captured)
        evidence.final_guard()
        output.final_guard()
        require(not torch.cuda.is_initialized(), "CPU evaluation initialized CUDA")
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
        torch.set_num_threads(old_threads)
    with (destination / "evaluation.provenance.json").open("xb") as f:
        f.write(base.json_bytes(status))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--tracking-sha256", required=True)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    a = parser.parse_args()
    print(
        json.dumps(
            run(a.tracking_root, a.tracking_sha256, a.packet_root, a.packet_sha256, a.output)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
