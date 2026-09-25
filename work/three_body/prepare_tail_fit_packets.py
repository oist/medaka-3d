"""CPU proposal packets for 301 independent, source-bound pre-frame contexts.

The original BoundedMouth class and all original proposals are retained. Only
free tail seeds from its first prediction are appended within the old capacity.
No fit, advance, initialization, state restoration or CUDA execution occurs.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]

import bounded_mouth_tracker as variant
import cv2
import numpy as np
import prepare_bounded_mouth_fit_packets as packets
import prepare_bounded_tail_parent_states as parent_source
import scipy
import tail_recovery as tail
import torch
from continuous_tracker import RECORD_FIELDS, STATE_FIELDS, _observation_copy
from motion import FlowObservations
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig
from observations import sample_pixels
from persistent_node_tracker import PersistentNodeConfig, PersistentNodeState
from resumable_observations import iter_from
from tracker import Proposal, TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, comparison, interim = packets.checks, packets.comparison, parent_source.interim
WORK = REPO / "work/three_body"
PARENT_ROOT = WORK / "bounded_tail_parent_states_301_v1"
PARENT_SHA = "9bd268bb619542304bdf4918e55de09255ad57800b593cb00f9383ee9d714fee"
FRAMES = np.r_[np.array([66524], np.int64), np.arange(66620, 66920, dtype=np.int64)]
RAW_SPANS = ((66523, 66525), (66619, 66920))
CPU_FRAMES = (66524, 66628, 66703, 66907)
PLAN = WORK / "TAIL_RECOVERY_LOCAL_PLAN.md"
PLAN_SHA = "049d80aef8252def9df4541c4b85eb5856da6948c0d2e5cc880ce115c3617382"
STAGE = "tail_local_fit_packet_preparation_v1"
PINS = {
    Path(
        tail.__file__
    ).resolve(): "636509cf37bb485a5e8efd34bbd55320d928df35fce59e4617addfee4b75a201",
    Path(
        variant.__file__
    ).resolve(): "64a1760f669c998932c4885516722358643adbee248ef76284f7a58fc924c466",
    Path(
        packets.__file__
    ).resolve(): "5094ce9b8ba5b5a7bcbcb31abd838340cfc374709d3c14c8a15e976bdf336da3",
    Path(
        parent_source.__file__
    ).resolve(): "3f673fbd2fe507375d11a86163abbfc522064f6f4ad8b21d8b2e0b52838a18c0",
}


def validate_saved_parents(parents, retained, summary):
    """Require exact fixed frames, state/record schemas, bytes and native row order."""
    checks.require(
        set(parents) == {*STATE_FIELDS, "frame_idx", "previous_frame_idx"}, "parent schema"
    )
    checks.require(
        set(retained) == {*RECORD_FIELDS, "selected_native_frames", "history_row_indices"},
        "retained schema",
    )
    for actual, expected in (
        (parents["frame_idx"], FRAMES),
        (parents["previous_frame_idx"], FRAMES - 1),
        (retained["selected_native_frames"], FRAMES),
        (retained["frame_idx"], np.repeat(FRAMES[:, None], 6, axis=1)),
    ):
        checks.require(
            checks.equal_bytes(actual, expected), "fixed native parent/retained frames differ"
        )
    rows = retained["history_row_indices"]
    checks.require(
        rows.dtype == np.int64
        and rows.shape == (301, 6)
        and np.all(np.diff(rows, axis=1) == 1)
        and np.all(np.diff(rows[:, 0]) > 0),
        "retained row indices differ",
    )
    for group, keys, name in (
        (parents, STATE_FIELDS, "previous_states"),
        (retained, RECORD_FIELDS, "retained_records"),
    ):
        arrays = {k: group[k] for k in keys}
        checks.require(
            all(v.shape[:2] == (301, 6) and not v.dtype.hasobject for v in arrays.values()),
            "six ordered parents/records required",
        )
        checks.require(
            packets.signature(arrays) == summary["outputs"][name]["arrays"],
            "sealed state/record signatures differ",
        )
    return (
        {k: parents[k].copy() for k in STATE_FIELDS},
        [{k: retained[k][i].copy() for k in RECORD_FIELDS} for i in range(301)],
    )


def get_states(hashes):
    interim.bind(PARENT_ROOT / "preparation.provenance.json", PARENT_SHA, hashes)
    seal = checks.verify(PARENT_ROOT, "preparation.provenance.json", parent_source.STAGE, hashes)
    for path, digest in list(hashes.items()):
        if path in seal["inputs"]:
            checks.require(
                seal["inputs"][path]["sha256"] == digest, "historical source changed: " + path
            )
    for name in (
        "summary.json",
        "checkpoint_metadata.json",
        "checkpoint_binding.json",
        "previous_all6_exact_states.npz",
        "retained_all6_records.npz",
    ):
        checks.require(str(PARENT_ROOT / name) in hashes, "unsealed parent output")
    summary = checks.read(PARENT_ROOT / "summary.json")
    checks.require(
        summary["status"] == "ok"
        and summary["frames"] == FRAMES.tolist()
        and summary["reconstructed_suffix"] == [66400, 68060]
        and summary["checkpoint_manifest_sha256"] == parent_source.MANIFEST_SHA
        and all(
            summary[k] is True
            for k in (
                "same_original66400_parent_history_exact",
                "parent_and_terminal39_codec_bound",
                "terminal_state7_including_velocity_age_byte_exact",
                "input_arrays_unchanged",
            )
        ),
        "successful fixed parent reconstruction required",
    )
    metadata = checks.read(PARENT_ROOT / "checkpoint_metadata.json")
    binding = checks.read(PARENT_ROOT / "checkpoint_binding.json")
    checkpoint = parent_source.BRANCH / "checkpoints/through_068060"
    for name in ("manifest.json", "metadata.json"):
        interim.input_member(seal, checkpoint / name, hashes)
    checks.require(
        hashes[str(checkpoint / "manifest.json")] == parent_source.MANIFEST_SHA
        and checks.read(checkpoint / "manifest.json")["bindings"] == binding
        and checks.read(checkpoint / "metadata.json") == metadata,
        "copied checkpoint context differs",
    )
    comparison.validate_lifecycle(metadata, 68060, 4)
    checks.require(
        binding["schema"] == "bounded_mouth_one_minute_checkpoint_binding_v1",
        "bounded checkpoint binding required",
    )
    for path, digest in binding["experimental_code_sha256"].items():
        source = interim.bind(path, digest, hashes)
        snapshot = parent_source.BRANCH / "code_snapshot" / source.relative_to(REPO)
        interim.input_member(seal, snapshot, hashes)
        checks.require(hashes[str(snapshot)] == digest, "historical source snapshot differs")
    fixed = binding["original_binding"]
    for path, digest in fixed["code_sha256"].items():
        interim.bind(path, digest, hashes)
    with np.load(PARENT_ROOT / "previous_all6_exact_states.npz", allow_pickle=False) as z:
        parents = {k: z[k].copy() for k in z.files}
    with np.load(PARENT_ROOT / "retained_all6_records.npz", allow_pickle=False) as z:
        retained = {k: z[k].copy() for k in z.files}
    states, records = validate_saved_parents(parents, retained, summary)
    seed = fixed["inputs"]["seed"]
    path = interim.bind(seed["registration_path"], seed["registration_sha256"], hashes)
    initial = np.asarray(checks.read(path)["pose_xyz"], np.float64)
    shared = seed["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "same SL3/calibration4 required")
    for path, digest in shared.items():
        interim.bind(path, digest, hashes)
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    rig = load_rig(calibration)
    # The frozen reader binds the same original source chunk through its completed
    # Directional audit. No historical candidate coordinates are deserialized.
    interim.bind(packets.prior.AUDIT / "audit.provenance.json", packets.prior.AUDIT_SHA, hashes)
    audit = checks.verify(
        packets.prior.AUDIT,
        "audit.provenance.json",
        "directional_first_chunk_interim_audit",
        hashes,
    )
    return states, records, metadata, binding, initial, rig, audit


def append_tail(original, states, rays, rig, lengths, recovery_config, persistence_config, cap):
    """Preserve every original object; free-only tail seeds use state0 age gates."""
    checks.require(bool(original), "original pool required")
    main = original[0]
    checks.require(
        main.kind == "predict" and main.parent == 0 and main.permutation == (0, 1, 2),
        "first-parent identity predict required",
    )
    checks.require(type(cap) is int and len(original) <= cap, "original pool exceeds capacity")
    age = (
        (states[0].node_missing[:, 2] >= persistence_config.minimum_missing_frames)
        & (states[0].direct_pectoral_age >= persistence_config.minimum_pectoral_frames)
        & persistence_config.enabled
    )
    before = packets.signature(packets.packet_proposals(original))
    info = dict(
        allow_bounded=False,
        age_eligible_slots=np.flatnonzero(age).tolist(),
        helper_called=False,
        helper_returned=0,
        after_age=0,
        admitted=0,
        excluded_by_age=0,
        excluded_by_budget=0,
        capacity=cap,
        helper_diagnostics=None,
        target_node=2,
    )
    result = None
    if age.any():
        result = tail.tail_recovery_proposals(
            main.pose, main.plan, rays, rig, lengths, recovery_config, allow_bounded=False
        )
        checks.require(
            all(p.node == 2 and p.seed_method == "free_intersection" for p in result.proposals),
            "free-only native tail proposals required",
        )
        allowed = [p for p in result.proposals if age[p.fish]]
        admitted = allowed[: cap - len(original)]
        info.update(
            helper_called=True,
            helper_returned=len(result.proposals),
            after_age=len(allowed),
            admitted=len(admitted),
            excluded_by_age=len(result.proposals) - len(allowed),
            excluded_by_budget=len(allowed) - len(admitted),
            helper_diagnostics=result.diagnostics,
        )
    else:
        admitted = []
    proposals = list(original) + [
        Proposal(
            p.pose_xyz.copy(),
            main.predicted.copy(),
            p.plan.copy(),
            main.parent,
            "tail_recovery",
            main.permutation,
        )
        for p in admitted
    ]
    after = packets.packet_proposals(proposals)
    checks.require(
        before == packets.signature(packets.packet_proposals(original)),
        "tail helper mutated original pool",
    )
    for key, value in packets.packet_proposals(original).items():
        checks.require(
            checks.equal_bytes(value, after[key][: len(original)]), "original prefix changed"
        )
    return proposals, info, np.asarray([-1] * len(original) + [p.fish for p in admitted], np.int64)


def raw_pairs(reader, on_observation, spans=RAW_SPANS):
    """Yield exact t-1/t pairs and close each independent seek without overread."""
    for start, stop in spans:
        iterator = iter_from(reader, start)
        try:
            value = next(iterator, None)
            checks.require(value is not None, "previous native frame EOF")
            previous = _observation_copy(value)
            checks.require(previous.frame_idx == start, "previous native frame differs")
            on_observation(previous)
            for native in range(start + 1, stop):
                value = next(iterator, None)
                checks.require(value is not None, "current native frame EOF")
                frame = _observation_copy(value)
                checks.require(frame.frame_idx == native, "current native frame differs")
                on_observation(frame)
                yield previous, frame
                previous = frame
        finally:
            iterator.close()


def build_packet(
    native, original, proposals, fish, parents, retained, initial, tracker, rays, target, flow
):
    arrays = dict(
        frame_idx=np.asarray(native, np.int64),
        original_count=np.asarray(len(original), np.int64),
        proposal_is_added=np.arange(len(proposals)) >= len(original),
        tail_fish=fish.copy(),
        proposal_recovery_fish=fish.copy(),
        flow_present=np.asarray(True),
        **packets.packet_proposals(proposals),
        initial_pose_xyz=initial.copy(),
        initial_lengths=tracker.lengths.copy(),
        fit_pixels=np.stack([target.pixels[v] for v in VIEWS]),
        fit_target=target.target.copy(),
        fit_weights=target.weights.copy(),
        fit_foreground=target.foreground.copy(),
    )
    arrays.update({"parent_" + k: np.stack([getattr(p, k) for p in parents]) for k in STATE_FIELDS})
    arrays.update({"retained_" + k: v.copy() for k, v in retained.items()})
    arrays.update(
        {
            "flow_" + f.name: np.asarray(getattr(flow, f.name)).copy()
            for f in dataclasses.fields(flow)
        }
    )
    strict = [supported_nodes(p.pose, p.plan, rays, tracker.rig) for p in proposals]
    arrays.update(
        generated_strict_sources=np.stack([d[0] for d in strict]),
        generated_observed=np.stack([d[1] for d in strict]),
        generated_ray_distance_mm=np.stack([d[2] for d in strict]),
    )
    for vi, view in enumerate(VIEWS):
        arrays[view + "_points"] = rays.points[view].copy()
        arrays[view + "_ray_origins"] = rays.origins[vi].copy()
        arrays[view + "_ray_directions"] = rays.directions[vi].copy()
    checks.require(not any(a.dtype.hasobject for a in arrays.values()), "object array prohibited")
    return arrays


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(WORK) and not output.exists(), "new work output required")
    code = comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    for path, digest in PINS.items():
        interim.bind(path, digest, hashes)
    interim.bind(PLAN, PLAN_SHA, hashes)
    output.mkdir(parents=True)
    old_threads = torch.get_num_threads(), cv2.getNumThreads()
    iterator = None
    with provenance(STAGE, output, name="preparation") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        try:
            for path, digest in code.items():
                target = output / "code_snapshot" / Path(path).relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(target) == digest, "source snapshot changed")
                emit(str(target.relative_to(output)))
            emit(
                "plan.json",
                dict(
                    schema=STAGE,
                    frames=FRAMES.tolist(),
                    raw_spans=RAW_SPANS,
                    expected_raw_native_frames=303,
                    expected_raw_images=909,
                    cpu_fit_frames=CPU_FRAMES,
                    parent_evidence_sha256=PARENT_SHA,
                    allow_bounded=False,
                    tail_minimum_missing=4,
                    chest_minimum_direct=4,
                    parent="original state0 identity predict",
                    preserve_every_original=True,
                    capacity="max_hypotheses+6 remaining only",
                    serial_original_flow=True,
                    fit=False,
                    GPU=False,
                    advance=False,
                    restore=False,
                ),
            )
            print("Verifying sealed301 parent evidence and exact native source", flush=True)
            states, retained, meta, binding, initial, rig, audit = get_states(hashes)
            fixed, context = binding["original_binding"], meta["context"]
            torch.set_num_threads(fixed["torch_cpu_threads"])
            cv2.setNumThreads(fixed["opencv_threads"])
            runtime = dict(
                numpy=np.__version__,
                scipy=scipy.__version__,
                opencv=cv2.__version__,
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
            )
            checks.require(
                all(fixed["package_versions"][k] == v for k, v in runtime.items()),
                "CPU software differs",
            )
            cfg = copy.deepcopy(context["tracker_config"])
            checks.require(
                cfg["device"] == "cuda" and cfg["mode"] == "detectron2" and cfg["flow_proposals"],
                "original CUDA/D2/flow context required",
            )
            cfg["device"] = "cpu"
            tracker = variant.BoundedMouthContinuousTracker(
                rig,
                initial,
                TrackerConfig(**cfg),
                recovery_config=NodeRecoveryConfig(**context["recovery_config"]),
                persistence_config=PersistentNodeConfig(**context["persistence_config"]),
            )
            checks.require(
                tracker.persistence_config.minimum_missing_frames == 4
                and tracker.persistence_config.minimum_pectoral_frames == 4
                and tracker.persistence_config.enabled,
                "fixed original age4/enabled required",
            )
            cpu_context = tracker._context()
            adapted = copy.deepcopy(cpu_context)
            adapted["tracker_config"]["device"] = "cuda"
            checks.require(adapted == context, "CPU context changed beyond device")
            reader, chunk = packets.get_reader(binding, audit, hashes)
            emit(
                "metadata.json",
                dict(
                    schema=STAGE,
                    frames=FRAMES.tolist(),
                    native_interval=[66524, 66920],
                    selection="{66524} union [66620,66920)",
                    raw_spans=RAW_SPANS,
                    cpu_fit_frames=CPU_FRAMES,
                    calibration_directory=str(
                        next(
                            Path(p).parent
                            for p in fixed["inputs"]["seed"]["SL_and_calibration_sha256"]
                            if Path(p).name == "calibration_manifest.json"
                        )
                    ),
                    tracker_config=context["tracker_config"],
                    recovery_config=context["recovery_config"],
                    persistence_config=context["persistence_config"],
                    original_context=context,
                    generation_context=cpu_context,
                    checkpoint_binding=binding,
                    source_chunk=chunk,
                    runtime=runtime,
                    torch_cpu_threads=torch.get_num_threads(),
                    opencv_threads=cv2.getNumThreads(),
                    parent_fields=list(STATE_FIELDS),
                    retained_fields=list(RECORD_FIELDS),
                    allow_bounded=False,
                    tail_policy=(
                        "state0 tail_missing>=4 & direct_chest_age>=4 & enabled; "
                        "remaining max_hypotheses+6 only"
                    ),
                    proposal_kind="tail_recovery",
                    original_prefix_preserved=True,
                    serial_flow=True,
                    historical_unavailable_fields=[
                        "prefit_proposal_pool",
                        "prefit_assignment_plan",
                        "retained_permutation",
                        "fit_iteration_path",
                    ],
                    retained_pose_role="post-fit validation only; never seeds generation",
                    temporal_policy=(
                        "301 independent original six-parent contexts; "
                        "no newly fitted result propagation"
                    ),
                    device_scope=(
                        "Only constructor device differs; original NumPy/CPU "
                        "proposals and serial flow; no fit/CUDA"
                    ),
                    unarchived_state=(
                        "graph cache and cumulative diagnostic counters do not feed "
                        "proposals; previous observation reconstructed from native t-1"
                    ),
                    source_parent_seal_sha256=PARENT_SHA,
                    fitted=False,
                    tracked=False,
                    biological_ID_truth=False,
                ),
            )
            rows, raw_rows = [], []
            iterator = raw_pairs(
                reader, lambda f: raw_rows.extend(packets.observation_evidence(f, reader))
            )
            for i, (previous, frame) in enumerate(iterator):
                native = int(frame.frame_idx)
                checks.require(
                    i < len(FRAMES) and native == FRAMES[i] and previous.frame_idx == native - 1,
                    "fixed selected pair order differs",
                )
                observed_before = packets.observation_signature(previous, frame)
                parents = [
                    PersistentNodeState(**{k: states[k][i, j].copy() for k in STATE_FIELDS})
                    for j in range(6)
                ]
                parent_before = packets.signature({k: states[k][i] for k in STATE_FIELDS})
                rays = RayObservations.from_points(rig, frame.detections)
                ray_arrays = {
                    v + s: a
                    for vi, v in enumerate(VIEWS)
                    for s, a in (
                        ("points", rays.points[v]),
                        ("origins", rays.origins[vi]),
                        ("directions", rays.directions[vi]),
                    )
                }
                ray_before = packets.signature(ray_arrays)
                target = sample_pixels(
                    frame,
                    cfg["mode"],
                    cfg["positive_samples"],
                    cfg["near_samples"],
                    cfg["global_samples"],
                )
                target_before = packets.signature(
                    {
                        "target": target.target,
                        "weights": target.weights,
                        "foreground": target.foreground,
                        **target.pixels,
                    }
                )
                flow = FlowObservations.from_frames(previous, frame, rig, parents[0].pose)
                flow_arrays = {
                    f.name: np.asarray(getattr(flow, f.name)) for f in dataclasses.fields(flow)
                }
                flow_before = packets.signature(flow_arrays)
                started = time.perf_counter()
                original = tracker._proposals(parents, rays, flow)
                packets.validate_pool_history(original, flow, retained[i])
                proposals, detail, fish = append_tail(
                    original,
                    parents,
                    rays,
                    rig,
                    tracker.lengths,
                    tracker.recovery_config,
                    tracker.persistence_config,
                    cfg["max_hypotheses"] + 6,
                )
                checks.require(
                    observed_before == packets.observation_signature(previous, frame)
                    and ray_before == packets.signature(ray_arrays)
                    and flow_before == packets.signature(flow_arrays)
                    and target_before
                    == packets.signature(
                        {
                            "target": target.target,
                            "weights": target.weights,
                            "foreground": target.foreground,
                            **target.pixels,
                        }
                    ),
                    "generation mutated observation/ray/flow/target",
                )
                checks.require(
                    parent_before
                    == packets.signature(
                        {k: np.stack([getattr(p, k) for p in parents]) for k in STATE_FIELDS}
                    ),
                    "generation mutated parents",
                )
                arrays = build_packet(
                    native,
                    original,
                    proposals,
                    fish,
                    parents,
                    retained[i],
                    initial,
                    tracker,
                    rays,
                    target,
                    flow,
                )
                np.savez_compressed(output / f"f{native}_packet.npz", **arrays)
                emit(f"f{native}_packet.npz")
                emit(f"f{native}_arrays.json", packets.signature(arrays))
                row = dict(
                    frame_idx=native,
                    original_count=len(original),
                    total_count=len(proposals),
                    added=detail,
                    generation_seconds=time.perf_counter() - started,
                    original_kind_counts={
                        k: sum(p.kind == k for p in original)
                        for k in sorted({p.kind for p in original})
                    },
                    original_chest_diagnostics=copy.deepcopy(
                        tracker.last_chest_proposal_diagnostic
                    ),
                    original_bounded_mouth_diagnostics=copy.deepcopy(
                        tracker.last_bounded_mouth_diagnostic
                    ),
                )
                rows.append(row)
                if len(rows) % 25 == 0 or len(rows) == 1:
                    print(
                        json.dumps(
                            dict(
                                prepared=len(rows),
                                native=native,
                                original=len(original),
                                tail_added=detail["admitted"],
                            )
                        ),
                        flush=True,
                    )
            iterator.close()
            iterator = None
            checks.require(
                len(rows) == 301
                and len(raw_rows) == 909
                and len({r["frame_idx"] for r in raw_rows}) == 303,
                "fixed packet/raw counts differ",
            )
            checks.require(
                tracker.initialize_count == tracker.processed_frames == tracker.restore_count == 0
                and tracker._states is None
                and tracker._previous_observation is None,
                "tracker initialized/restored/advanced",
            )
            emit("frames.json", rows)
            emit("native_observation_evidence.json", raw_rows)
            summary = dict(
                schema=STAGE,
                status="ok",
                frames=FRAMES.tolist(),
                packets=len(rows),
                raw_spans=RAW_SPANS,
                raw_delivered_native_frames=303,
                raw_delivered_images=len(raw_rows),
                raw_decoder_internal_keyframe_work_not_counted=True,
                all_raw_and_decoded_mask_SHA_pass=True,
                all6_parent7_fields_reused_from_sealed_exact_recurrence=True,
                all_original_hypothesis_counts_and_flow_flags_match_history=True,
                original_pool_direct_byte_golden_available=False,
                original_refit_and_reselection_validation_pending=True,
                originals=sum(r["original_count"] for r in rows),
                added=sum(r["added"]["admitted"] for r in rows),
                helper_calls=sum(r["added"]["helper_called"] for r in rows),
                dropped_by_age=sum(r["added"]["excluded_by_age"] for r in rows),
                dropped_by_budget=sum(r["added"]["excluded_by_budget"] for r in rows),
                original_prefix_preserved=True,
                allow_bounded=False,
                serial_flow=True,
                initialize_calls=0,
                restore_calls=0,
                advance_calls=0,
                fit_calls=0,
                CUDA_calls=0,
                biological_ID_truth=False,
                no_new_temporal_trajectory=True,
            )
            emit("summary.json", summary)
            (output / "RESULT.md").write_text(
                "# 尾の局所fit入力準備\n\n"
                f"固定301時刻で全原候補 {summary['originals']} 本を保持し、"
                f"free-only尾候補 {summary['added']} 本を追加。"
                "元state0の尾欠測4・胸部直接支持4・有効条件とmax_hypotheses+6残枠を適用した。\n\n"
                "原画は[66523,66525)と[66619,66920)の303時刻 x 3視点909枚。"
                "native BGR/mask SHAを照合し、元serial Flowを同じstate0 poseから作成した。"
                "全301の元6親7項目と保持19項目は封印済み再帰証拠に結合した。\n\n"
                "保持poseは検証用で候補初期値に渡していない。元poolの初期bytes/plan/permutation全ては保存されていないため、"
                "現在のhypotheses/flow/parent-kind整合は後続の元pool再fit・19項目照合を代替しない。"
                "301個の独立contextであり、fit/advance/restore/GPU・生物ID確認は未実施。\n",
                encoding="utf-8",
            )
            emit("RESULT.md")
            checks.require(
                code == comparison.source_registry.local_code(), "loaded source set changed"
            )
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            packets.final_guard(hashes, outputs)
            record.param("all_input_source_snapshot_output_SHA_unchanged", True)
        finally:
            if iterator is not None:
                iterator.close()
            torch.set_num_threads(old_threads[0])
            cv2.setNumThreads(old_threads[1])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
