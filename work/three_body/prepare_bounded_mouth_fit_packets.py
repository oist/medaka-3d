"""CPU-only exact-source proposal packets for 33 independent historical contexts.

No begin/restore/advance/fit or CUDA call is made. Original candidate starts were
not archived: this regenerates them and preserves the actual six retained records
for a later, separately authorized fit/reselection comparison.
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

_REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_REPOSITORY / "labs/20260909_three_body"), str(_REPOSITORY / "src")]

import bounded_mouth_recovery as bounded
import cv2
import numpy as np
import recompute_directional_mouth_generation as generated
import scipy
import torch
from continuous_tracker import RECORD_FIELDS, STATE_FIELDS, _observation_copy
from directional_chest_tracker import DirectionalChestContinuousTracker
from motion import FlowObservations
from node_geometry import RayObservations, supported_nodes
from node_recovery import NodeRecoveryConfig
from observations import sample_pixels
from persistent_node_tracker import PersistentNodeConfig, PersistentNodeState
from prepare_sequence import array_sha256
from resumable_observations import iter_from
from source_sequence import SourceSequenceReader
from tracker import Proposal, TrackerConfig

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

prior, checks, comparison = generated.prior, generated.checks, generated.comparison
REPO = prior.REPO
START, STOP = generated.START, generated.STOP
ROOT = REPO / "work/three_body/directional_mouth_generation_lost33_v1"
GENERATION_SHA = "9a6cbd189e90c162649cacf934442754c2723d9b6571e2c8d9ea8b9873dce638"
HELPER_SHA = "b128932013196d2f54cd99e69763f88ef492057222bd62800bf8e0f58e30f9df"
STAGE = "bounded_mouth_local_fit_packet_preparation_v1"


def signature(arrays):
    return {
        key: dict(dtype=value.dtype.str, shape=list(value.shape), sha256=array_sha256(value))
        for key, value in arrays.items()
    }


def observation_signature(*frames):
    arrays = {}
    for i, frame in enumerate(frames):
        for field in ("detections", "images_bgr", "masks", "backgrounds_bgr"):
            for view, value in getattr(frame, field).items():
                if value is not None:
                    arrays[f"{i}_{field}_{view}"] = value
    return signature(arrays)


def packet_proposals(proposals):
    checks.require(bool(proposals), "nonempty original proposal pool required")
    return dict(
        proposal_pose_xyz=np.stack([p.pose for p in proposals]),
        proposal_predicted=np.stack([p.predicted for p in proposals]),
        proposal_plan=np.stack([p.plan for p in proposals]),
        proposal_parent=np.asarray([p.parent for p in proposals], np.int64),
        proposal_kind=np.asarray([p.kind for p in proposals], dtype="U32"),
        proposal_permutation=np.asarray([p.permutation for p in proposals], np.int64),
    )


def append_bounded(original, states, result, cfg, cap):
    """Preserve the original prefix; age-filter only the new helper's proposals."""
    checks.require(bool(original), "original proposals required")
    main = original[0]
    checks.require(
        main.kind == "predict" and main.parent == 0 and main.permutation == (0, 1, 2),
        "first-parent original prediction required",
    )
    checks.require(type(cap) is int and cap >= len(original), "original pool exceeds capacity")
    parent = states[main.parent]
    age = (
        (parent.node_missing[:, 0] >= cfg.minimum_missing_frames)
        & (parent.direct_pectoral_age >= cfg.minimum_pectoral_frames)
        & cfg.enabled
    )
    eligible = [p for p in result.proposals if age[p.fish]]
    admitted = eligible[: cap - len(original)]
    proposals = list(original) + [
        Proposal(
            p.pose_xyz.copy(),
            main.predicted.copy(),
            p.plan.copy(),
            main.parent,
            "bounded_mouth_recovery",
            main.permutation,
        )
        for p in admitted
    ]
    before, after = packet_proposals(original), packet_proposals(proposals)
    for key in before:
        checks.require(
            checks.equal_bytes(before[key], after[key][: len(original)]), "prefix changed"
        )
    return (
        proposals,
        dict(
            age_eligible_slots=np.flatnonzero(age).tolist(),
            helper_returned=len(result.proposals),
            after_age=len(eligible),
            admitted=len(admitted),
            excluded_by_age=len(result.proposals) - len(eligible),
            excluded_by_budget=len(eligible) - len(admitted),
            capacity=cap,
        ),
        np.asarray([-1] * len(original) + [p.fish for p in admitted], np.int64),
    )


def retained_records(state, frame):
    t = frame - comparison.FIRST
    offsets = state["history_offsets"]
    rows = np.arange(offsets[t], offsets[t + 1])
    checks.require(
        len(rows) == 6 and np.all(state["history_frame_idx"][rows] == frame),
        "six matching native retained rows required",
    )
    return {key: state["history_" + key][rows].copy() for key in RECORD_FIELDS}


def validate_pool_history(original, flow, records):
    checks.require(np.all(records["hypotheses"] == len(original)), "historical pool count differs")
    checks.require(
        np.all(records["proposal_flow_parent"] == 0)
        and all(np.array_equal(row, flow.node_valid) for row in records["proposal_flow_nodes"]),
        "historical flow support differs",
    )
    pairs = {(p.parent, p.kind) for p in original}
    checks.require(
        all(
            (int(parent), str(kind)) in pairs
            for parent, kind in zip(records["parent"], records["proposal_kind"], strict=True)
        ),
        "historical selected parent/kind absent from regenerated original pool",
    )
    for parent in range(6):
        predictions = [p for p in original if p.parent == parent and p.kind == "predict"]
        checks.require(
            len(predictions) == 1 and predictions[0].permutation == (0, 1, 2),
            "exactly one original prediction per parent required",
        )


def get_states(hashes):
    prior.bind(ROOT / "generation.provenance.json", GENERATION_SHA, hashes)
    generation = checks.verify(ROOT, "generation.provenance.json", generated.STAGE, hashes)
    for source, digest in list(hashes.items()):
        if source in generation["inputs"]:
            checks.require(
                generation["inputs"][source]["sha256"] == digest,
                "historically loaded generation source/input differs: " + source,
            )
    checkpoint = prior.BRANCH / "checkpoints/through_068060"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        prior.audit_input(generation, checkpoint / name, hashes)
    binding = checks.read(checkpoint / "manifest.json")["bindings"]
    for path, digest in binding["experimental_code_sha256"].items():
        source = prior.bind(path, digest, hashes)
        prior.bind(prior.BRANCH / "code_snapshot" / source.relative_to(REPO), digest, hashes)
    for path, digest in binding["original_binding"]["code_sha256"].items():
        prior.bind(path, digest, hashes)
    state, meta, _ = comparison.load_bound_checkpoint(checkpoint, binding, hashes)
    comparison.validate_lifecycle(meta, generated.HORIZON, 4)
    prior.bind(prior.AUDIT / "audit.provenance.json", prior.AUDIT_SHA, hashes)
    audit = checks.verify(
        prior.AUDIT, "audit.provenance.json", "directional_first_chunk_interim_audit", hashes
    )
    parent_root = REPO / "work/three_body/long_continuation_through66400_v1/checkpoint"
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        prior.audit_input(audit, parent_root / name, hashes)
    parent, parent_meta, _ = comparison.load_bound_checkpoint(
        parent_root, binding["original_binding"], hashes
    )
    comparison.validate_lifecycle(parent_meta, prior.START, 3)
    migration = prior.bind(
        prior.BRANCH / "state_import.json", binding["state_import_sha256"], hashes
    )
    comparison.verify_import(checks.read(migration), parent, parent_meta)
    checks.require(checks.read(migration)["new_context"] == meta["context"], "context changed")
    states, _ = generated.reconstruct(parent, state, meta)
    saved_path = ROOT / "previous_all6_exact_states.npz"
    checks.require(str(saved_path) in hashes, "saved parents are not sealed")
    with np.load(saved_path, allow_pickle=False) as archive:
        checks.require(
            np.array_equal(archive["frame_idx"], np.arange(START, STOP)), "parent frames"
        )
        checks.require(set(archive.files) == {"frame_idx", *STATE_FIELDS}, "parent schema")
        for key in STATE_FIELDS:
            checks.require(checks.equal_bytes(states[key], archive[key]), "parent bytes: " + key)
    records = [retained_records(state, frame) for frame in range(START, STOP)]
    seed = binding["original_binding"]["inputs"]["seed"]
    path = prior.bind(seed["registration_path"], seed["registration_sha256"], hashes)
    initial = np.asarray(checks.read(path)["pose_xyz"], np.float64)
    shared = seed["SL_and_calibration_sha256"]
    checks.require(len(shared) == 7, "SL3/calibration4 required")
    for path, digest in shared.items():
        prior.bind(path, digest, hashes)
    manifest = next(Path(path) for path in shared if Path(path).name == "calibration_manifest.json")
    rig = load_rig(manifest.parent)
    return states, records, meta, binding, initial, rig, audit


def get_reader(binding, audit, hashes):
    """Bind consumed chunk10 to the completed original and actual directional inputs."""
    branch = checks.verify(
        prior.BRANCH, "tracking.provenance.json", "directional_one_minute_continuous", hashes
    )
    old_seal_path = prior.audit_input(
        audit, prior.ORIGINAL / "continuation.provenance.json", hashes
    )
    old = checks.read(old_seal_path)
    checks.require(
        old["status"] == "ok" and old["stage"] == "three_body_long_continuation",
        "successful original input owner required",
    )
    consumed_path = prior.ORIGINAL / "consumed_chunks.json"
    item = old["outputs"]["consumed_chunks"]
    checks.require(Path(item["path"]).resolve() == consumed_path, "consumed output role")
    prior.bind(consumed_path, item["sha256"], hashes)
    records = checks.read(consumed_path)
    selected = [r for r in records if r["index"] == 10]
    checks.require(len(selected) == 1, "one registered source chunk10 required")
    selected = selected[0]
    checks.require([selected["start"], selected["stop"]] == [66400, 68060], "source chunk range")
    stage = Path(binding["original_binding"]["inputs"]["input_stage_root"]).resolve()
    root = (stage / selected["attempt_path"]).resolve(strict=True)
    checks.require(root.is_relative_to(stage), "chunk escaped registered stage")
    prior.bind(root / "manifest.json", selected["manifest_sha256"], hashes)
    prior.bind(root / "staging.provenance.json", selected["provenance_sha256"], hashes)
    seal = checks.read(root / "staging.provenance.json")
    checks.require(
        seal["status"] == "ok" and seal["stage"] == "prepare_independent_mask_chunk",
        "successful independent mask chunk required",
    )
    checks.require(
        set(seal["outputs"]) == set(selected["output_hashes"]), "chunk output membership"
    )
    manifest = checks.read(root / "manifest.json")
    pending_raw = {}
    for role, entry in selected["output_hashes"].items():
        path = Path(entry["path"]).resolve(strict=True)
        output = seal["outputs"][role]
        checks.require(
            path.is_relative_to(root)
            and Path(output["path"]).resolve() == path
            and output["sha256"] == entry["sha256"]
            and path.stat().st_size == output["size"],
            "consumed output differs: " + role,
        )
        if role.endswith("_raw"):
            view = role.removesuffix("_raw")
            raw = selected["raw_hardlinks"][view]
            canonical = Path(raw["source_path"]).resolve(strict=True)
            stat = path.stat()
            checks.require(
                Path(raw["path"]).resolve() == path
                and path.samefile(canonical)
                and [stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns]
                == [raw["bytes"], raw["device"], raw["inode"], raw["mtime_ns"]]
                and entry["sha256"] == raw["sha256"] == manifest["raw"][view]["sha256"],
                "raw hardlink binding differs",
            )
            member = branch["inputs"].get(str(canonical))
            checks.require(
                member is not None
                and member["sha256"] == raw["sha256"]
                and Path(member["path"]).resolve() == canonical,
                "raw branch input",
            )
            # Actual file SHA is verified once here by the original reader below.
            pending_raw[str(path)] = entry["sha256"]
        else:
            prior.audit_input(branch, path, hashes)
            checks.require(hashes[str(path)] == entry["sha256"], "branch/chunk SHA differs")
    seed = binding["original_binding"]["inputs"]["seed"]
    reader = SourceSequenceReader(root, seed["input_root"])
    checks.require(
        reader.manifest == manifest and np.array_equal(reader.frames, np.arange(66400, 68060)),
        "reader native interval differs",
    )
    checks.require(
        set(map(str, reader.nodes.paths.values()))
        == {p for p in seed["SL_and_calibration_sha256"] if Path(p).suffix == ".npz"},
        "reader SL source differs",
    )
    hashes.update(pending_raw)
    return reader, selected


def observation_evidence(frame, reader):
    t = frame.frame_idx - int(reader.frames[0])
    rows = []
    for view in VIEWS:
        image, mask = frame.images_bgr[view], frame.masks[view]
        bgr = array_sha256(image)
        checks.require(bgr == reader.pixel_hashes[view][t], "native BGR mismatch")
        mask_sha = None if mask is None else array_sha256(mask)
        expected = reader.masks[view]["frame_mask_sha256"][t].decode()
        checks.require(mask is None or mask_sha == expected, "decoded mask SHA mismatch")
        checks.require(
            (mask is not None) == bool(reader.masks[view]["view_available"][t]),
            "mask availability differs",
        )
        rows.append(
            dict(
                frame_idx=frame.frame_idx,
                time_seconds=frame.frame_idx / 166,
                view=view,
                bgr_sha256=bgr,
                mask_sha256=mask_sha,
                mask_count=None if mask is None else len(mask),
                detection_signature=signature({"points": frame.detections[view]})["points"],
                background_sha256=array_sha256(frame.backgrounds_bgr[view]),
            )
        )
    return rows


def final_guard(hashes, outputs):
    for path, digest in {**hashes, **outputs}.items():
        checks.require(checks.sha(path) == digest, "final input/source/output changed: " + path)


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = comparison.source_registry.local_code()
    hashes, output_hashes = dict(code), {}
    checks.require(checks.sha(bounded.__file__) == HELPER_SHA, "frozen additional helper changed")
    output.mkdir(parents=True)
    before_threads = torch.get_num_threads(), cv2.getNumThreads()
    iterator = None
    with provenance(STAGE, output, name="preparation") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            digest = checks.sha(path)
            output_hashes[str(path)] = digest
            record.output(name, path, sha256=digest)

        try:
            for path, digest in code.items():
                target = output / "code_snapshot" / Path(path).relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(path).read_bytes())
                checks.require(checks.sha(target) == digest, "source snapshot differs")
                emit(str(target.relative_to(output)))
            print("Verifying completed all-six parent context and exact source chunk", flush=True)
            states, retained, meta, binding, initial, rig, audit = get_states(hashes)
            fixed = binding["original_binding"]
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
                "CPU software differs from original generation",
            )
            cfg = copy.deepcopy(meta["context"]["tracker_config"])
            checks.require(
                cfg["device"] == "cuda" and cfg["mode"] == "detectron2" and cfg["flow_proposals"],
                "registered CUDA/D2/flow context required",
            )
            cfg["device"] = "cpu"
            tracker = DirectionalChestContinuousTracker(
                rig,
                initial,
                TrackerConfig(**cfg),
                recovery_config=NodeRecoveryConfig(**meta["context"]["recovery_config"]),
                persistence_config=PersistentNodeConfig(**meta["context"]["persistence_config"]),
            )
            cpu_context = tracker._context()
            adapted = copy.deepcopy(cpu_context)
            adapted["tracker_config"]["device"] = "cuda"
            checks.require(adapted == meta["context"], "CPU context changed more than device")
            reader, chunk = get_reader(binding, audit, hashes)
            emit(
                "metadata.json",
                dict(
                    schema=STAGE,
                    native_interval=[START, STOP],
                    frames=list(range(START, STOP)),
                    raw_interval=[START - 1, STOP],
                    calibration_directory=str(
                        next(
                            Path(p).parent
                            for p in fixed["inputs"]["seed"]["SL_and_calibration_sha256"]
                            if Path(p).name == "calibration_manifest.json"
                        )
                    ),
                    tracker_config=meta["context"]["tracker_config"],
                    recovery_config=meta["context"]["recovery_config"],
                    persistence_config=meta["context"]["persistence_config"],
                    original_context=meta["context"],
                    generation_context=cpu_context,
                    checkpoint_binding=binding,
                    source_chunk=chunk,
                    runtime=runtime,
                    torch_cpu_threads=torch.get_num_threads(),
                    opencv_threads=cv2.getNumThreads(),
                    proposal_fields=list(
                        packet_proposals(
                            [Proposal(initial, initial, np.zeros((3, 3, 3), int), 0, "predict")]
                        ).keys()
                    ),
                    parent_fields=list(STATE_FIELDS),
                    retained_fields=list(RECORD_FIELDS),
                    historical_unavailable_fields=[
                        "prefit_proposal_pool",
                        "prefit_assignment_plan",
                        "retained_permutation",
                        "fit_iteration_path",
                    ],
                    retained_pose_role="post-fit validation only; never seeds generation",
                    temporal_policy=(
                        "33 independent original parent contexts; "
                        "no newly fitted result propagation"
                    ),
                    candidate_origin=(
                        "original deterministic pool regenerated from raw t-1/t "
                        "and exact full6 state"
                    ),
                    device_scope=(
                        "Only constructor device differs; NumPy/CPU proposals and flow; "
                        "no tensor fit/CUDA calls"
                    ),
                    unarchived_state=(
                        "graph cache and accumulated diagnostic counters do not "
                        "feed proposals; previous observation reconstructed from actual raw"
                    ),
                    biological_ID_truth=False,
                    fitted=False,
                    tracked=False,
                ),
            )
            iterator = iter_from(reader, START - 1)
            previous = _observation_copy(next(iterator))
            checks.require(previous.frame_idx == START - 1, "previous native frame differs")
            raw_rows = observation_evidence(previous, reader)
            rows = []
            for i, native in enumerate(range(START, STOP)):
                frame = _observation_copy(next(iterator))
                checks.require(frame.frame_idx == native, "current native frame differs")
                raw_rows.extend(observation_evidence(frame, reader))
                observations_before = observation_signature(previous, frame)
                parents = [
                    PersistentNodeState(**{k: states[k][i, j].copy() for k in STATE_FIELDS})
                    for j in range(6)
                ]
                parent_before = signature({k: states[k][i] for k in STATE_FIELDS})
                rays = RayObservations.from_points(rig, frame.detections)
                target = sample_pixels(
                    frame,
                    cfg["mode"],
                    cfg["positive_samples"],
                    cfg["near_samples"],
                    cfg["global_samples"],
                )
                flow = FlowObservations.from_frames(previous, frame, rig, parents[0].pose)
                flow_arrays = {
                    f.name: np.asarray(getattr(flow, f.name)) for f in dataclasses.fields(flow)
                }
                flow_before = signature(flow_arrays)
                started = time.perf_counter()
                original = tracker._proposals(parents, rays, flow)
                validate_pool_history(original, flow, retained[i])
                data_path = ROOT / f"f{native}_generation.npz"
                checks.require(str(data_path) in hashes, "unsealed prediction reference")
                with np.load(data_path, allow_pickle=False) as reference:
                    for key, value in (
                        ("predicted_pose", original[0].pose),
                        ("predicted_plan", original[0].plan),
                        ("initial_lengths", tracker.lengths),
                    ):
                        checks.require(
                            checks.equal_bytes(value, reference[key]),
                            "prediction bytes differ: " + key,
                        )
                    for vi, view in enumerate(VIEWS):
                        for suffix, value in (
                            ("_points", rays.points[view]),
                            ("_ray_origins", rays.origins[vi]),
                            ("_ray_directions", rays.directions[vi]),
                        ):
                            checks.require(
                                checks.equal_bytes(value, reference[view + suffix]),
                                "current ray differs",
                            )
                original_before = signature(packet_proposals(original))
                added = bounded.bounded_mouth_recovery_proposals(
                    original[0].pose,
                    original[0].plan,
                    rays,
                    rig,
                    tracker.lengths,
                    tracker.recovery_config,
                )
                proposals, detail, fish = append_bounded(
                    original, parents, added, tracker.persistence_config, cfg["max_hypotheses"] + 6
                )
                checks.require(
                    observations_before == observation_signature(previous, frame)
                    and flow_before == signature(flow_arrays),
                    "candidate preparation mutated observations or flow",
                )
                checks.require(
                    original_before == signature(packet_proposals(original)),
                    "helper mutated original",
                )
                checks.require(
                    parent_before
                    == signature(
                        {k: np.stack([getattr(p, k) for p in parents]) for k in STATE_FIELDS}
                    ),
                    "generation mutated parent states",
                )
                arrays = dict(
                    frame_idx=np.asarray(native, np.int64),
                    original_count=np.asarray(len(original), np.int64),
                    proposal_is_added=np.arange(len(proposals)) >= len(original),
                    bounded_fish=fish,
                    flow_present=np.asarray(True),
                    **packet_proposals(proposals),
                    initial_pose_xyz=initial.copy(),
                    initial_lengths=tracker.lengths.copy(),
                    fit_pixels=np.stack([target.pixels[view] for view in VIEWS]),
                    fit_target=target.target.copy(),
                    fit_weights=target.weights.copy(),
                    fit_foreground=target.foreground.copy(),
                )
                arrays.update({"parent_" + k: states[k][i].copy() for k in STATE_FIELDS})
                arrays.update({"retained_" + k: value.copy() for k, value in retained[i].items()})
                arrays.update(
                    {
                        "flow_" + f.name: np.asarray(getattr(flow, f.name)).copy()
                        for f in dataclasses.fields(flow)
                    }
                )
                direct = [supported_nodes(p.pose, p.plan, rays, rig) for p in proposals]
                arrays.update(
                    generated_strict_sources=np.stack([d[0] for d in direct]),
                    generated_observed=np.stack([d[1] for d in direct]),
                    generated_ray_distance_mm=np.stack([d[2] for d in direct]),
                )
                for vi, view in enumerate(VIEWS):
                    arrays[view + "_points"] = rays.points[view].copy()
                    arrays[view + "_ray_origins"] = rays.origins[vi].copy()
                    arrays[view + "_ray_directions"] = rays.directions[vi].copy()
                checks.require(
                    not any(a.dtype.hasobject for a in arrays.values()), "object array prohibited"
                )
                filename = f"f{native}_packet.npz"
                np.savez_compressed(output / filename, **arrays)
                emit(filename)
                emit(f"f{native}_arrays.json", signature(arrays))
                row = dict(
                    frame_idx=native,
                    original_count=len(original),
                    total_count=len(proposals),
                    added=detail,
                    generation_seconds=time.perf_counter() - started,
                    original_kind_counts={
                        kind: sum(p.kind == kind for p in original)
                        for kind in sorted({p.kind for p in original})
                    },
                    original_chest_diagnostics=tracker.last_chest_proposal_diagnostic,
                    bounded_diagnostics=added.diagnostics,
                )
                rows.append(row)
                previous = frame
                print(
                    f"Prepared {native}: original={len(original)}, added={detail['admitted']}",
                    flush=True,
                )
            iterator.close()
            iterator = None
            checks.require(
                tracker.initialize_count == 0
                and tracker.processed_frames == 0
                and tracker._previous_observation is None
                and tracker._states is None,
                "preparation advanced or initialized a tracker",
            )
            emit("frames.json", rows)
            emit("native_observation_evidence.json", raw_rows)
            summary = dict(
                schema=STAGE,
                status="ok",
                native_interval=[START, STOP],
                packets=len(rows),
                raw_delivered_native_interval=[START - 1, STOP],
                raw_delivered_images=len(raw_rows),
                raw_decoder_internal_keyframe_work_not_counted=True,
                source_chunk_interval=[66400, 68060],
                all_raw_and_decoded_mask_SHA_pass=True,
                all6_parent7_fields_reconstructed_byte_exact=True,
                all_original_hypothesis_counts_and_flow_flags_match_history=True,
                original_pool_direct_byte_golden_available=False,
                original_refit_and_reselection_validation_pending=True,
                originals=sum(r["original_count"] for r in rows),
                added=sum(r["added"]["admitted"] for r in rows),
                dropped_by_budget=sum(r["added"]["excluded_by_budget"] for r in rows),
                dropped_by_age=sum(r["added"]["excluded_by_age"] for r in rows),
                original_prefix_preserved=True,
                initialize_calls=0,
                advance_calls=0,
                fit_calls=0,
                CUDA_calls=0,
                biological_ID_truth=False,
                no_new_temporal_trajectory=True,
            )
            emit("summary.json", summary)
            (output / "RESULT.md").write_text(
                "# 局所 fit 入力の準備\n\n33 時刻の元6親・全原候補 "
                f"{summary['originals']} 件を保存し、既存 age gate と残り24枠の下で "
                f"bounded mouth {summary['added']} 件を追加した。\n\n"
                "原画は [67662,67696) の34時刻 x 3視点102枚。"
                "元manifestのnative BGRとdecoded mask SHAを照合。"
                "33個の独立した元親contextであり、新しい連続追跡ではない。\n\n"
                "原候補の初期bytesは過去に保存されていない。全親7状態・元予測/plan/current rays・"
                "元hypotheses数・flow支持は照合済みだが、全原候補のGPU再fit/6候補再選択照合は未実施。"
                "元retained19項は検証用に保存し、"
                "permutationとpre-fit planは未記録のため補完しない。\n\n"
                "CPU constructorのdeviceだけを変更し、元のNumPy候補生成を実行した。"
                "fit/advance/GPU/生物ID判定は行っていない。\n",
                encoding="utf-8",
            )
            emit("RESULT.md")
            for path, digest in hashes.items():
                record.input(path, path, sha256=digest)
            record.param("all_input_source_snapshot_output_SHA_unchanged", True)
            final_guard(hashes, output_hashes)
        finally:
            if iterator is not None:
                iterator.close()
            torch.set_num_threads(before_threads[0])
            cv2.setNumThreads(before_threads[1])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
