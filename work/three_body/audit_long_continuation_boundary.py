"""CPU audit of the completed first actual long-input checkpoint boundary.

No tracking, GPU, fitting, biological labels or later baseline coordinates.
The old prefix is allowed to have a different final winning backtrace.
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import audit_long_continuation_first1660 as first
import audit_long_continuation_support as support_helpers
import audit_persistent_minute as support
import cv2
import numpy as np
from motion import FlowConfig, FlowObservations
from observations import CachedNodes, FrameObservation
from prepare_sequence import decode_frame_masks
from source_sequence import SourceSequenceReader
from tracker_checkpoint import read_checkpoint

from medaka3d.provenance import provenance

COMMON, BASE, REPO, LAB, VIEWS = first.common, first.BASE, first.REPO, first.LAB, first.VIEWS
FIRST, SPLIT, STOP, FULL_STOP = 49800, 51460, 53120, 348600
PARENT = REPO / "work/three_body/long_continuation_first1660_v1"
RUN = REPO / "work/three_body/long_continuation_through53120_v1"
PLAN = REPO / "work/three_body/long_boundary_resume_plan_v1"
PLAN_SHA = "5db1175353c28ee0f4bd5095592aa31c4d4bbae034b338347b58f30e3593add8"
PARENT_FILES = (
    "continuation.provenance.json",
    "summary.json",
    "consumed_chunks.json",
    "checkpoint/manifest.json",
    "checkpoint/arrays.npz",
    "checkpoint/metadata.json",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def json_equal(a, b, label):
    require(COMMON.canonical(a) == COMMON.canonical(b), label)


def phase(root, seal, inputs, evidence, *, parent=None):
    total = (SPLIT if parent is None else STOP) - FIRST
    prior = 0 if parent is None else SPLIT - FIRST
    binding = seal["params"]["bindings"]
    current_hashes = dict(inputs.verified_hashes)
    if parent is None:
        current_hashes = {
            p: d
            for p, d in current_hashes.items()
            if not Path(p).is_relative_to(inputs.chunk_roots[1])
        }
    expected = {**current_hashes, **binding["code_sha256"]}
    if parent is not None:
        expected.update({str(parent / name): BASE.sha256(parent / name) for name in PARENT_FILES})
    require(set(seal["inputs"]) == set(expected), "phase input membership differs")
    for name, digest in expected.items():
        info = seal["inputs"][name]
        require(
            Path(info["path"]).resolve() == Path(name) and info["sha256"] == digest,
            "phase input role/path/hash differs",
        )
        evidence.add(name, digest)
    for info in seal["outputs"].values():
        path = Path(info["path"]).resolve(strict=True)
        require(
            path.is_relative_to(root) and path.stat().st_size == info["size"],
            "phase output path or size differs",
        )
        evidence.add(path, info["sha256"])
    named = {
        "candidate": "candidate.npz",
        "summary": "summary.json",
        "progress": "progress.jsonl",
        "consumed_chunks": "consumed_chunks.json",
        "checkpoint_manifest": "checkpoint/manifest.json",
    }
    for role, filename in named.items():
        require(
            Path(seal["outputs"][role]["path"]).resolve() == root / filename,
            "phase named output differs",
        )
    for filename in ("checkpoint/arrays.npz", "checkpoint/metadata.json"):
        require(str(root / filename) in seal["outputs"], "checkpoint payload not sealed")
    for filename, digest in binding["code_sha256"].items():
        path = root / "code_snapshot" / Path(filename).relative_to(REPO)
        require(str(path) in seal["outputs"], "producer source snapshot not sealed")
        evidence.add(path, digest)
    arrays, metadata, manifest = read_checkpoint(root / "checkpoint", expected_bindings=binding)
    summary = COMMON.read_json(root / "summary.json")
    values = dict(
        schema="long_continuation_phase_v1",
        technical_completion=True,
        full_sequence_complete=False,
        prefix_backtrace_provisional=True,
        registered_interval=[FIRST, FULL_STOP],
        processed_interval=[FIRST, FIRST + total],
        processed_total=total,
        phase_frames=[FIRST + prior, FIRST + total],
        phase_processed=total - prior,
        consumed_chunk_count=1 if parent is None else 2,
        logical_initialization_count=1,
        checkpoint_restore_count=int(parent is not None),
        external_pose_reset_count=0,
        temporal_subsampling=False,
        human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False,
    )
    for key, value in values.items():
        json_equal(summary.get(key), value, "phase summary differs: " + key)
    require(
        metadata["processed_frames"] == total
        and metadata["previous_frame"] == FIRST + total - 1
        and metadata["initialize_count"] == 1
        and metadata["restore_count"] == int(parent is not None)
        and metadata["node_recovery_diagnostics"] == summary["node_recovery_diagnostics"]
        and manifest["payload"]["bytes"] == summary["checkpoint_payload_bytes"],
        "checkpoint lifecycle/counters differ",
    )
    require(len(arrays) == 39, "expected full 39-array checkpoint")
    progress = [json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()]
    counts = [v["processed"] for v in progress]
    require(
        counts
        and counts == sorted(set(counts))
        and counts[-1] == total
        and all(
            v["source_frame"] == FIRST + v["processed"] - 1
            and v["phase_processed"] == v["processed"] - prior
            for v in progress
        ),
        "phase native progress differs",
    )
    consumed = COMMON.read_json(root / "consumed_chunks.json")
    json_equal(
        consumed,
        inputs.chunk_records[: values["consumed_chunk_count"]],
        "consumed chunk evidence differs from current validation",
    )
    with np.load(root / "candidate.npz", allow_pickle=False) as data:
        candidate = {key: data[key] for key in data.files}
    traced = COMMON.traced_candidate(arrays)
    require(set(candidate) == set(traced) and len(candidate) == 21, "candidate schema differs")
    comparisons = {key: COMMON.same_array(candidate[key], traced[key], key) for key in candidate}
    return dict(
        arrays=arrays,
        metadata=metadata,
        summary=summary,
        consumed=consumed,
        candidate=candidate,
        comparisons=comparisons,
    )


def suffix_history(parent, current):
    old, arrays = parent["arrays"], current["arrays"]
    cut = SPLIT - FIRST
    offsets, old_offsets = arrays["history_offsets"], old["history_offsets"]
    require(
        offsets.dtype == np.int64
        and offsets.shape == (STOP - FIRST + 1,)
        and np.all((np.diff(offsets) >= 1) & (np.diff(offsets) <= 6)),
        "extended history offsets differ",
    )
    COMMON.same_array(offsets[: cut + 1], old_offsets, "prefix history offsets")
    copied = {}
    for key in COMMON.HISTORY_FIELDS:
        require(len(arrays["history_" + key]) == offsets[-1], "unaccounted history rows")
        copied[key] = COMMON.same_array(
            arrays["history_" + key][: old_offsets[-1]], old["history_" + key], "prefix " + key
        )
    require(
        old_offsets[-1] == 9953 and len(old["state_pose"]) == 6,
        "registered parent full beam differs",
    )
    previous = [{key: old["state_" + key][i] for key in COMMON.STATE_FIELDS} for i in range(6)]
    eligible = parent["metadata"]["node_recovery_diagnostics"]["calls_with_age_eligible_fish"]
    first_records, nonzero_parents = [], 0
    for t in range(cut, STOP - FIRST):
        eligible += bool(
            np.any(
                (previous[0]["node_missing"][:, 0] >= 4) & (previous[0]["direct_pectoral_age"] >= 4)
            )
        )
        following = []
        for row in range(int(offsets[t]), int(offsets[t + 1])):
            r = {key: arrays["history_" + key][row] for key in COMMON.HISTORY_FIELDS}
            pi = int(r["parent"])
            require(
                0 <= pi < len(previous) and int(r["frame_idx"]) == FIRST + t,
                "new history parent/native frame differs",
            )
            nonzero_parents += pi != 0
            p = previous[pi]
            pose, sources, held = r["pose_xyz"], r["sources"], r["observed"]
            require(
                np.isfinite(pose).all()
                and np.all(pose >= [0, 0, -6])
                and np.all(pose <= [192, 192, 186])
                and np.all(sources >= -1),
                "new history engineering geometry/source contract differs",
            )
            COMMON.same_array(held, (sources >= 0).sum(1) >= 2, "new observed")
            for view in range(3):
                for node in range(3):
                    used = sources[:, view, node]
                    used = used[used >= 0]
                    require(len(used) == len(np.unique(used)), "new history source reused")
            body = held[:, 1] | (r["mask_supported"] & (sources[:, :, :2] >= 0).any((1, 2)))
            missing = np.where(body, 0, p["missing"] + 1)
            age = np.where(held, 0, p["node_missing"] + 1)
            pectoral_age = np.where(held[:, 1], p["direct_pectoral_age"] + 1, 0)
            for key, expected in (
                ("body_supported", body),
                ("missing", missing),
                ("node_missing", age),
                ("direct_pectoral_age", pectoral_age),
            ):
                COMMON.same_array(r[key], expected, "new parent recurrence " + key)
            require(
                not np.any(p["review"] & ~r["identity_review"])
                and not np.any((missing >= 4) & ~r["identity_review"]),
                "new state erased sticky identity review",
            )
            cost = p["cost"] + r["costs"]
            COMMON.same_array(np.asarray(cost), np.asarray(r["path_costs"]), "new path cost")
            velocity = 0.5 * p["velocity"] + 0.5 * (pose[:, 1] - p["pose"][:, 1])
            norm = np.linalg.norm(velocity, axis=-1, keepdims=True)
            velocity *= np.minimum(1.0, 2.0 / np.maximum(norm, 1e-12))
            velocity = np.where(body[:, None], velocity, 0.8 * p["velocity"])
            following.append(
                dict(
                    pose=pose,
                    velocity=velocity,
                    missing=missing,
                    review=r["identity_review"],
                    cost=float(cost),
                    node_missing=age,
                    direct_pectoral_age=pectoral_age,
                )
            )
            if t == cut:
                first_records.append(
                    dict(
                        parent=pi,
                        proposal_kind=str(r["proposal_kind"]),
                        missing=missing.tolist(),
                        node_missing=age.tolist(),
                        direct_pectoral_age=pectoral_age.tolist(),
                        cumulative_cost=float(cost),
                        flow_parent=int(r["proposal_flow_parent"]),
                    )
                )
        previous = following
    for key in COMMON.STATE_FIELDS:
        COMMON.same_array(arrays["state_" + key], np.asarray([p[key] for p in previous]), key)
    require(
        eligible
        == current["metadata"]["node_recovery_diagnostics"]["calls_with_age_eligible_fish"],
        "extended age-eligible counter differs",
    )
    require(
        current["summary"]["retained_history_rows"] == offsets[-1]
        and current["summary"]["active_beam_count"] == len(previous),
        "beam summary differs",
    )
    return dict(
        old_ordered_records_retained=9953,
        old_all_19_history_fields_byte_exact=copied,
        all_six_parent_states_independently_reconstructed=True,
        new_nonzero_parent_rows=nonzero_parents,
        retained_history_rows=int(offsets[-1]),
        first_new_native_frame=SPLIT,
        first_new_records=first_records,
        all_new_parent_state_recurrences_byte_exact=True,
        independent_age_eligible_calls=eligible,
    )


def previous_observation(checkpoint, reader):
    arrays, metadata = checkpoint["arrays"], checkpoint["metadata"]
    native = metadata["previous_frame"]
    time = native - int(reader.frames[0])
    fields = {name: {} for name in ("detections", "images_bgr", "masks", "backgrounds_bgr")}
    comparisons = {}
    for view in VIEWS:
        cap = cv2.VideoCapture(str(reader.videos[view]))
        try:
            require(cap.isOpened() and cap.set(cv2.CAP_PROP_POS_FRAMES, native), "seek failed")
            ok, image = cap.read()
            require(ok and cap.get(cv2.CAP_PROP_POS_FRAMES) == native + 1, "native seek differs")
        finally:
            cap.release()
        require(
            hashlib.sha256(image.tobytes()).hexdigest() == reader.pixel_hashes[view][time],
            "checkpoint previous raw image changed",
        )
        mask = (
            decode_frame_masks(reader.masks[view], time)
            if reader.masks[view]["view_available"][time]
            else None
        )
        values = dict(
            detections=reader.nodes.at(view, native),
            images_bgr=image,
            masks=mask,
            backgrounds_bgr=reader.backgrounds[view],
        )
        for name, value in values.items():
            key = "previous_" + name + "_" + view
            require(
                metadata["previous_observation"][key] == (value is not None),
                "previous observation availability differs",
            )
            if value is None:
                require(key not in arrays, "invented previous observation")
            else:
                comparisons[key] = COMMON.same_array(arrays[key], value, key)
            fields[name][view] = None if value is None else value.copy()
    return FrameObservation(native, **fields), comparisons


def native_audit(inputs, parent, current, protocol):
    first_reader = SourceSequenceReader(inputs.chunk_roots[0], inputs.input_root)
    old_frame, parent_comparison = previous_observation(parent, first_reader)
    reader = SourceSequenceReader(inputs.chunk_roots[1], inputs.input_root)
    require(np.array_equal(reader.frames, np.arange(SPLIT, STOP)), "new chunk frame range differs")
    n = STOP - SPLIT
    evidence = {
        "frame_idx": np.arange(SPLIT, STOP),
        "raw_bgr_sha256": np.empty((n, 3), "S64"),
        "mask_decoded_sha256": np.empty((n, 3), "S64"),
        "sleap_points_sha256": np.empty((n, 3), "S64"),
        "mask_count": np.empty((n, 3), np.int64),
    }
    flow_info, count = None, 0
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(protocol["opencv_threads"])
    try:
        for index, frame in enumerate(reader):
            require(index < n and frame.frame_idx == SPLIT + index, "extra/gapped native frame")
            count += 1
            for vi, view in enumerate(VIEWS):
                image_hash = hashlib.sha256(frame.images_bgr[view].tobytes()).hexdigest()
                require(image_hash == reader.pixel_hashes[view][index], "raw native SHA differs")
                evidence["raw_bgr_sha256"][index, vi] = image_hash
                mask = frame.masks[view]
                evidence["mask_count"][index, vi] = -1 if mask is None else len(mask)
                mask_hash = "" if mask is None else hashlib.sha256(mask.tobytes()).hexdigest()
                if mask is not None:
                    expected = bytes(reader.masks[view]["frame_mask_sha256"][index]).decode("ascii")
                    require(mask_hash == expected, "decoded native mask SHA differs")
                evidence["mask_decoded_sha256"][index, vi] = mask_hash
                evidence["sleap_points_sha256"][index, vi] = hashlib.sha256(
                    frame.detections[view].tobytes()
                ).hexdigest()
            if index == 0:
                config = FlowConfig()
                json_equal(
                    dataclasses.asdict(config),
                    COMMON.read_json(first.PROTOCOL)["flow_config"],
                    "flow default config differs",
                )
                flow = FlowObservations.from_frames(
                    old_frame, frame, inputs.rig, parent["arrays"]["state_pose"][0].copy(), config
                )
                arrays = current["arrays"]
                left, right = arrays["history_offsets"][SPLIT - FIRST : SPLIT - FIRST + 2]
                for row in range(int(left), int(right)):
                    COMMON.same_array(
                        arrays["history_proposal_flow_nodes"][row],
                        flow.node_valid,
                        "first new frame flow valid flags",
                    )
                    require(
                        arrays["history_proposal_flow_parent"][row] == 0,
                        "first new frame discarded the prior flow parent",
                    )
                flow_info = dict(
                    previous_native_frame=old_frame.frame_idx,
                    current_native_frame=frame.frame_idx,
                    node_valid=flow.node_valid.tolist(),
                    valid_nodes=int(flow.node_valid.sum()),
                    all_first_new_records_flow_flags_byte_exact=True,
                    scope="CPU boundary flow replay and stored node flags; full flow "
                    "coordinates were not saved by the tracker.",
                )
        require(count == n, "new source chunk ended early")
    finally:
        cv2.setNumThreads(old_threads)
    _, final_comparison = previous_observation(current, reader)
    return dict(
        new_native_frames=count,
        new_view_images=3 * count,
        all_new_native_BGR_and_decoded_mask_SHA_match=True,
        new_SLEAP_rows_from_bound_native_cache=True,
        parent_previous_payload_byte_exact=parent_comparison,
        final_previous_payload_byte_exact=final_comparison,
        boundary_flow=flow_info,
    ), evidence


def prefix_changes(previous, current):
    result = {}
    for key, a in previous.items():
        b = current[key][: SPLIT - FIRST]
        same_dtype = a.dtype == b.dtype
        changed = [i for i in range(len(a)) if not same_dtype or a[i].tobytes() != b[i].tobytes()]
        result[key] = dict(
            dtype_same=same_dtype,
            changed_frames=len(changed),
            first_changed_native_frame=None if not changed else FIRST + changed[0],
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.exists() and output.is_relative_to(REPO), "new workspace output required")
    code = sorted(
        {
            *COMMON.code_paths(),
            Path(__file__).resolve(),
            Path(first.__file__).resolve(),
            Path(support.__file__).resolve(),
            Path(support_helpers.__file__).resolve(),
        }
    )
    evidence = support_helpers.Digests()
    for path in code:
        evidence.add(path)
    evidence.add(
        Path(first.__file__), "9195695aa6317bf00dda86bfbb8dc200806191dc28580bf01bedc702d183a46c"
    )
    evidence.add(
        Path(support.__file__), "31d5d9796c5ba10e7f211381c8b37a5fe2dbb3367ff2e829b18b3ebd4f94d0bc"
    )
    evidence.add(
        Path(support_helpers.__file__),
        "cd79f2582919504ef1e625b5968997ef03a98f0e7f87d7f6e5957695d368b41c",
    )
    # Both phases must be successful before any candidate/checkpoint arrays are opened.
    seals = []
    for root in (PARENT, RUN):
        path = root / "continuation.provenance.json"
        evidence.add(path)
        seal = COMMON.read_json(path)
        require(
            seal.get("status") == "ok" and seal.get("stage") == "three_body_long_continuation",
            "both long phases must be successfully sealed",
        )
        require(
            seal["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
            "phase guard incomplete",
        )
        seals.append(seal)
    plan = COMMON.read_json(PLAN / "plan.json")
    evidence.add(PLAN / "plan.json", PLAN_SHA)
    registration = COMMON.read_json(PLAN / "registration.provenance.json")
    require(
        registration["status"] == "ok"
        and registration["stage"] == "register_long_boundary_execution",
        "boundary plan is not sealed",
    )
    evidence.add(PLAN / "registration.provenance.json")
    for info in registration["inputs"].values():
        evidence.add(info["path"], info["sha256"])
    require(
        Path(registration["outputs"]["plan"]["path"]).resolve() == PLAN / "plan.json"
        and registration["outputs"]["plan"]["sha256"] == PLAN_SHA,
        "registered plan output binding differs",
    )
    require(
        datetime.fromisoformat(registration["finished_at"])
        <= datetime.fromisoformat(seals[1]["started_at"]),
        "plan was not fixed before execution",
    )
    binding = seals[1]["params"]["bindings"]
    json_equal(binding, seals[0]["params"]["bindings"], "parent/current frozen binding differs")
    json_equal(binding, plan["frozen_input_and_code_binding"], "plan binding differs")
    require(
        plan["next_native_interval"] == [SPLIT, STOP]
        and plan["next_output_interval"] == [FIRST, STOP]
        and Path(plan["parent_phase"]).resolve() == PARENT
        and seals[0]["params"]["resume_phase"] is None
        and Path(seals[1]["params"]["resume_phase"]).resolve() == PARENT
        and seals[0]["params"]["through_frame"] == SPLIT
        and seals[1]["params"]["through_frame"] == STOP,
        "registered lifecycle differs",
    )
    inputs = first.verified.load_long_inputs(
        Path(binding["inputs"]["seed"]["registration_path"]).parent,
        Path(binding["inputs"]["input_stage_root"]),
        Path(binding["inputs"]["seed"]["input_root"]),
        through_frame=STOP,
    )
    json_equal(inputs.fixed_bindings, binding["inputs"], "verified long input binding differs")
    require(
        np.array_equal(inputs.frame_idx, np.arange(FIRST, STOP))
        and len(inputs.chunk_roots) == 2
        and inputs.full_frame_range == (FIRST, FULL_STOP),
        "verified two-chunk native interval differs",
    )
    parent = phase(PARENT, seals[0], inputs, evidence)
    current = phase(RUN, seals[1], inputs, evidence, parent=PARENT)
    first.verified.verify_prefix_records(parent["consumed"], current["consumed"])
    json_equal(parent["metadata"]["context"], current["metadata"]["context"], "context differs")
    require(
        current["metadata"]["context"]["initial_pose"]
        == COMMON.digest_array(inputs.initial_pose_xyz),
        "initial registered pose differs",
    )
    context = current["metadata"]["context"]
    protocol = COMMON.read_json(first.PROTOCOL)
    json_equal(
        context["cameras"],
        COMMON.normalize([dataclasses.asdict(camera) for camera in inputs.rig.cameras]),
        "saved calibration differs",
    )
    for context_key, protocol_key in (
        ("tracker_config", "tracker_config"),
        ("recovery_config", "node_recovery_config"),
        ("persistence_config", "persistence_config"),
    ):
        json_equal(context[context_key], protocol[protocol_key], "saved numerical config differs")
    output.mkdir(parents=True, exist_ok=False)
    with provenance("long_continuation_boundary_independent_audit", output, name="audit") as record:
        for path in code:
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            evidence.add(target, evidence.paths[str(path)])
        parent_history = first.history_check(
            parent["arrays"], parent["metadata"], inputs.initial_pose_xyz
        )
        history = suffix_history(parent, current)
        native, native_arrays = native_audit(inputs, parent, current, binding)
        nodes = CachedNodes(inputs.input_root, inputs.protocol["experiment"], inputs.frame_idx)
        geometry, geometry_arrays = support.verify_all_support(
            current["candidate"], nodes, inputs.rig
        )
        counts = support.direct_counts(current["candidate"])
        require(
            counts["direct_pectoral_fish_frames"] == current["summary"]["direct_pectoral_count"]
            and counts["accepted_pectoral_fish_frames"]
            == current["summary"]["accepted_pectoral_count"]
            and counts["review_fish_frames"] == current["summary"]["identity_review_count"],
            "summary direct/review counts differ",
        )
        require(
            counts["direct_pectoral_per_slot"] == current["summary"]["per_slot_direct_pectoral"]
            and counts["accepted_pectoral_per_slot"]
            == current["summary"]["per_slot_accepted_pectoral"],
            "per-slot counts differ",
        )
        changes = prefix_changes(parent["candidate"], current["candidate"])
        result = dict(
            schema="long_boundary_independent_audit_v1",
            status="ok",
            parent=str(PARENT),
            run=str(RUN),
            candidate_sha256=evidence.paths[str(RUN / "candidate.npz")],
            output_interval=[FIRST, STOP],
            new_interval=[SPLIT, STOP],
            full_sequence_complete=False,
            prefix_backtrace_provisional=True,
            logical_initialization_count=1,
            checkpoint_restore_count=1,
            external_pose_reset_count=0,
            parent_history=parent_history,
            history=history,
            all_current_21_arrays_byte_exact_backtrace=current["comparisons"],
            current_checkpoint_arrays=39,
            previous_consumed_chunk_proof_byte_exact=True,
            native_evidence=native,
            all_source_geometry=geometry,
            counts=counts,
            provisional_prefix_changes_by_array=changes,
            phase_tracking_seconds=current["summary"]["tracking_seconds"],
            preparation_seconds=current["summary"]["preparation_seconds"],
            checkpoint_seconds=current["summary"]["checkpoint_seconds"],
            checkpoint_file_bytes=current["summary"]["checkpoint_payload_bytes"],
            independent_ID_truth=False,
            uninterrupted_3320_reference_comparison=False,
            limits=[
                "The same ordered parent history survives, while the winning "
                "prefix backtrace is allowed to change with future observations.",
                "The first-frame correspondence is an optimistic registered seed. "
                "No GT, later baseline coordinates or independent ID truth is used.",
                "This 20-second prefix is not the completed thirty-minute goal.",
            ],
        )
        np.savez_compressed(output / "native_evidence.npz", **native_arrays)
        np.savez_compressed(output / "native_support.npz", **geometry_arrays)
        (output / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        text = (
            "# 実チャンク境界の独立監査\n\n"
            "親の全9,953候補履歴と全6 active statesを検証し、延長結果の同じprefixが"
            "dtype/shape/bytes完全一致することを確認しました。新frameの親cost・velocity・"
            "欠測age・reviewを再計算し、最終21配列は全履歴のbacktraceと完全一致しました。\n\n"
            f"[{FIRST},{STOP}) の3320 frames、追加は[{SPLIT},{STOP}) の1660 framesです。"
            "初期化1回、復元1回、外部姿勢リセット0回で、出力は暫定prefixです。\n\n"
            "新チャンクの全4,980原画像とdecoded masksのSHA、SL native行の連続性、"
            "親51459/終端53119の画像・mask・SL・背景とcheckpointを照合しました。"
            "51459→51460のCPU flow再計算は最初の全候補に保存されたvalid flagsと一致しました。\n\n"
            f"直接支持は全{geometry['source_rows_checked']}対応で2mm/12px条件を満たします。"
            f"胸部の直接観測は{counts['direct_pectoral_fish_frames']}/9960、条件付き受理は"
            f"{counts['accepted_pectoral_fish_frames']}/9960、reviewは{counts['review_fish_frames']}です。\n\n"
            "暫定prefixに変化のある配列: "
            + ", ".join(key for key, v in changes.items() if v["changed_frames"])
            + "。詳細はsummary.jsonに保存しました。暫定解の変更を保存状態の欠落とは扱いません。\n\n"
            "途切れない3320frame対照を新たに実行した比較ではなく、保存された境界の整合監査です。"
            "全3個体の人手真値やゼロswapの証明、30分間の完了結果ではありません。\n"
        )
        (output / "RESULT.md").write_text(text, encoding="utf-8")
        for path, digest in evidence.paths.items():
            if not Path(path).is_relative_to(output):
                record.input(path, path, sha256=digest)
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                outputs[str(path)] = BASE.sha256(path)
                record.output(path.relative_to(output).as_posix(), path, sha256=outputs[str(path)])
        evidence.final_guard()
        COMMON.unchanged(outputs)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("GPU_or_new_tracking", False)
        record.param("GT_or_later_baseline_inputs", False)
    print(json.dumps({"output": str(output), "status": "ok", "counts": counts}))


if __name__ == "__main__":
    main()
