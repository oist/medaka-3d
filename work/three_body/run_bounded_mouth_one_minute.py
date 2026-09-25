"""Run a fixed 9960-frame bounded-mouth branch with one imported complete state.

The six native input chunks share one tracker, previous observation, graph cache
and complete beam history. Periodic checkpoints preserve that state; they do not
restore or reinitialize it. GPU allocation is limited to the qualified 25% cap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import fit_chest_recovery_candidates as base
import numpy as np
import replay_chest_deficit as original
import torch
from bounded_mouth_tracker import BoundedMouthContinuousTracker, import_original_bounded_state
from chest_candidate_diagnostics import CandidateDiagnostics
from continuous_tracker import RECORD_FIELDS, AcceleratedContinuousPersistentNodeTracker
from fit_bounded_mouth_packets import bind_input, member
from mask_validation_receipts import ReceiptStore
from node_recovery import NodeRecoveryConfig
from persistent_node_tracker import PersistentNodeConfig
from qualify_concurrent_directional_gpu import gpu_sample
from source_sequence import SourceSequenceReader
from tracker import TrackerConfig
from tracker_checkpoint import read_checkpoint, write_checkpoint

from medaka3d.provenance import provenance

REPO = Path(__file__).resolve().parents[2]
START, STOP, CHUNK = 66400, 76360, 1660
DIAGNOSTIC_RANGES = ((66656, 66706), (67630, 67785), (75047, 75163))
checks = base.evidence_tools
LOCAL_ROOT = REPO / "work/three_body/bounded_mouth_local_cuda33_v1"
LOCAL_SHA = "184bb27d17028ddb2c3df8edf7cb944096bfbf343692adafbf1b9d39f082333f"
PIXEL_ROOT = REPO / "work/three_body/bounded_mouth_unused_pixel_evaluation_33_v1"
PIXEL_SHA = "3bc1f256dbdebd963b6cbc91e0fa24609b5cc2799e8f47427106ff177e01b09b"


def preserve_checkpoint(tracker, parent, path, binding):
    arrays, metadata = tracker.export_state()
    prefix = START - original.FIRST
    for key in ("history_offsets", *("history_" + k for k in RECORD_FIELDS)):
        end = prefix + 1 if key == "history_offsets" else parent["arrays"]["history_offsets"][-1]
        prior, current = parent["arrays"][key], arrays[key][:end]
        exact = (
            np.array_equal(prior, current)
            if prior.dtype.kind == "U"
            else original.byte_equal(prior, current)
        )
        checks.require(exact, "original retained history prefix changed: " + key)
    write_checkpoint(path, arrays, metadata, bindings=binding)
    loaded, loaded_meta, _ = read_checkpoint(path, expected_bindings=binding)
    checks.require(
        loaded_meta == metadata and all(original.byte_equal(loaded[k], arrays[k]) for k in arrays),
        "periodic checkpoint roundtrip differs",
    )
    return dict(
        path=str(path),
        previous_frame=metadata["previous_frame"],
        processed_frames=metadata["processed_frames"],
        initialize_count=metadata["initialize_count"],
        restore_count=metadata["restore_count"],
        arrays=len(arrays),
        manifest_sha256=checks.sha(path / "manifest.json"),
    )


def main(args):
    output, pilot, short = (getattr(args, k).resolve() for k in ("output", "pilot", "short"))
    checks.require(
        output.is_relative_to(REPO / "work") and not output.exists(),
        "new workspace output required",
    )
    code = base.local_code()
    hashes = dict(code)
    pilot_seal = checks.verify(
        pilot, "pilot.provenance.json", "concurrent_directional_GPU_pilot", hashes
    )
    short_seal = checks.verify(
        short, "replay.provenance.json", "directional_chest_continuous_replay", hashes
    )
    pilot_result = checks.read(member(pilot, "summary.json", hashes))
    short_result = checks.read(member(short, "summary.json", hashes))
    checks.require(checks.sha(LOCAL_ROOT / "fit.provenance.json") == LOCAL_SHA, "local fit changed")
    local_seal = checks.verify(
        LOCAL_ROOT, "fit.provenance.json", "bounded_mouth_same_objective_local_fit", hashes
    )
    local_result = checks.read(member(LOCAL_ROOT, "summary.json", hashes))
    checks.require(
        local_result["status"] == "ok"
        and local_result["frames"] == list(range(67663, 67696))
        and local_result["control_all19_records_reproduced"]
        and local_result["CPU_CUDA_all_candidate_frames"] == [67663, 67683, 67684, 67695],
        "qualified all33 local comparison required",
    )
    for path in set(code) & set(local_seal["inputs"]):
        bind_input(local_seal, path, hashes)
    checks.require(
        checks.sha(PIXEL_ROOT / "evaluation.provenance.json") == PIXEL_SHA,
        "unused-pixel comparison changed",
    )
    pixel_seal = checks.verify(
        PIXEL_ROOT, "evaluation.provenance.json", "bounded_mouth_unused_pixel_evaluation", hashes
    )
    bind_input(pixel_seal, LOCAL_ROOT / "fit.provenance.json", hashes)
    generation = args.generation.resolve(strict=True)
    checks.require(
        checks.sha(generation / "qualification.provenance.json") == args.generation_sha,
        "generation qualification changed",
    )
    generation_seal = checks.verify(
        generation,
        "qualification.provenance.json",
        "bounded_mouth_tracker_generation_qualification_v1",
        hashes,
    )
    qualified = checks.read(member(generation, "summary.json", hashes))
    checks.require(
        qualified["passed"]
        and qualified["all_proposals_byte_exact"]
        and qualified["off_original_prefix_byte_exact"]
        and qualified["frames"] == list(range(67663, 67696))
        and qualified["originals"] == 462
        and qualified["added"] == 62
        and qualified["fit_calls"] == qualified["advance_calls"] == 0,
        "actual all33 on/off generation must reproduce qualified proposals",
    )
    for name in ("bounded_mouth_tracker.py", "bounded_mouth_recovery.py", "bounded_ray_seed.py"):
        path = str(Path(__file__).with_name(name))
        checks.require(path in code, "required loaded bounded source missing")
        bind_input(generation_seal, path, hashes)
    for path in set(code) & set(generation_seal["inputs"]):
        bind_input(generation_seal, path, hashes)
    packet_seal = (
        REPO / "work/three_body/bounded_mouth_local_fit_packets_33_v1/preparation.provenance.json"
    )
    for seal in (local_seal, generation_seal):
        bind_input(seal, packet_seal, hashes)
    checks.require(
        pilot_result["resource_pilot_passed"]
        and pilot_result["fits"] == 32
        and pilot_result["allocator_fraction_cap"] == 0.25,
        "required concurrent pilot incomplete",
    )
    checks.require(
        short_result["replay_complete"]
        and short_result["native_frames"] == 306
        and short_result["external_pose_resets"] == 0,
        "continuous prototype not qualified",
    )
    for name in (
        "directional_chest_fitter.py",
        "directional_device_fitter.py",
        "directional_chest_tracker.py",
    ):
        path = Path(__file__).with_name(name)
        checks.require(
            short_seal["inputs"][str(path)]["sha256"] == code[str(path)],
            "continuous algorithm changed",
        )
        if name != "directional_chest_tracker.py":
            checks.require(
                pilot_seal["inputs"][str(path)]["sha256"] == code[str(path)],
                "GPU pilot used another fitter",
            )
    output.mkdir(parents=True)
    evidence = original.phase.Digests()
    for path, digest in hashes.items():
        evidence.add(path, digest)
    store = ReceiptStore(args.receipt_root.resolve())
    tracker = iterator = diagnostic = None
    checkpoints, telemetry, proposal_rows = [], [], []
    result = dict(
        schema="bounded_mouth_one_minute_v1",
        status="incomplete",
        native_interval=[START, STOP],
        one_minute_completed=False,
        independent_ID_truth=False,
        adopted=False,
        external_pose_resets=0,
        periodic_checkpoints_are_not_reinitializations=True,
    )
    with provenance("bounded_mouth_one_minute_continuous", output, name="tracking") as record:
        try:
            original.snapshot(code, output, record)
            observed = output / "registered_reference"
            observed.mkdir()
            reference = original.ReplayGPUWindow(
                args.reference_supervisor_root, args.reference_child_pid, observed
            )
            try:
                reference.attach()
                checks.require(
                    not reference.child.signalled() and not reference.owner.signalled(),
                    "owned reference is not live at branch startup",
                )
                record.param("reference_ownership_verified_at_start", True)
                record.param("reference_natural_phase_transitions_allowed", True)
            finally:
                reference.close()
            record.param("native_interval", [START, STOP])
            record.param("diagnostic_ranges", DIAGNOSTIC_RANGES)
            record.param("allocator_fraction_cap", 0.25)
            record.param("concurrent_reference_and_D2", True)
            prep_start = time.perf_counter()
            context = original.prepare(evidence, store)
            binding, protocol, parent = context["binding"], context["protocol"], context["parent"]
            torch.set_num_threads(binding["torch_cpu_threads"])
            original.runtime(binding, gpu=False)
            sample = gpu_sample()
            checks.require(
                sample["total_MiB"] - sample["used_MiB"] >= 6144,
                "insufficient initial GPU headroom",
            )
            telemetry.append(sample)
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
            torch.cuda.reset_peak_memory_stats()
            original.runtime(binding, gpu=True)
            cfg = TrackerConfig(**protocol["tracker_config"])
            kwargs = dict(
                recovery_config=NodeRecoveryConfig(**protocol["node_recovery_config"]),
                persistence_config=PersistentNodeConfig(**protocol["persistence_config"]),
            )
            rig, seed = context["inputs"].rig, context["inputs"].initial_pose_xyz.copy()
            source = AcceleratedContinuousPersistentNodeTracker(rig, seed, cfg, **kwargs)
            tracker = BoundedMouthContinuousTracker(rig, seed, cfg, **kwargs)
            migration = import_original_bounded_state(
                tracker, source, parent["arrays"], parent["metadata"]
            )
            original.save_json(output / "state_import.json", migration)
            checks.require(
                len(tracker._states) == 6 and tracker.previous_frame == START - 1,
                "initial full beam boundary differs",
            )
            checkpoint_binding = dict(
                schema="bounded_mouth_one_minute_checkpoint_binding_v1",
                original_binding=binding,
                experimental_code_sha256=code,
                state_import_sha256=checks.sha(output / "state_import.json"),
                native_interval=[START, STOP],
            )
            preparation_seconds = time.perf_counter() - prep_start
            original.save_json(
                output / "ready.json",
                dict(
                    preparation_seconds=preparation_seconds,
                    logical_initialization_count=tracker.initialize_count,
                    restore_count=tracker.restore_count,
                    native_interval=[START, STOP],
                    previous_observation=context["previous_observation"],
                ),
            )
            evidence.final_guard()
            store.final_guard()
            print(
                json.dumps(dict(event="one_minute_prepared", native_interval=[START, STOP])),
                flush=True,
            )
            reader_seconds = GPU_seconds = checkpoint_seconds = 0.0
            with CandidateDiagnostics(
                tracker, frame_range=(DIAGNOSTIC_RANGES[0][0], DIAGNOSTIC_RANGES[-1][1])
            ) as diagnostic:
                for chunk_start in range(START, STOP, CHUNK):
                    chunk_stop = chunk_start + CHUNK
                    reader_start = time.perf_counter()
                    root = context["inputs"].chunk_roots[(chunk_start - original.FIRST) // CHUNK]
                    reader = (
                        context["reader"]
                        if chunk_start == START
                        else SourceSequenceReader(root, context["inputs"].input_root)
                    )
                    checks.require(
                        reader.frames[0] == chunk_start and reader.frames[-1] + 1 == chunk_stop,
                        "native chunk range differs",
                    )
                    reader_seconds += time.perf_counter() - reader_start
                    iterator = original.bounded_frames(reader, first=chunk_start, stop=chunk_stop)
                    for frame in iterator:
                        native = frame.frame_idx
                        fit_start = time.perf_counter()
                        if any(first <= native < stop for first, stop in DIAGNOSTIC_RANGES):
                            with diagnostic.frame(native):
                                tracker.advance(frame)
                        else:
                            tracker.advance(frame)
                        torch.cuda.synchronize()
                        GPU_seconds += time.perf_counter() - fit_start
                        proposal_rows.append(
                            dict(
                                frame_idx=native,
                                **tracker.last_chest_proposal_diagnostic,
                                bounded_mouth=tracker.last_bounded_mouth_diagnostic,
                                enabled_parent_fish=int(tracker.last_diagnostics["enabled"].sum()),
                            )
                        )
                        processed = native - START + 1
                        if processed % 100 == 0:
                            sample = gpu_sample()
                            sample["frame_idx"] = native
                            telemetry.append(sample)
                            checks.require(
                                sample["total_MiB"] - sample["used_MiB"] >= 2048,
                                "remaining GPU headroom below 2GiB",
                            )
                        if processed % 25 == 0 or native == STOP - 1:
                            print(
                                json.dumps(
                                    dict(
                                        event="one_minute_progress",
                                        frame=native,
                                        processed=processed,
                                        total=STOP - START,
                                        accumulated_advance_seconds=GPU_seconds,
                                    )
                                ),
                                flush=True,
                            )
                    iterator.close()
                    iterator = None
                    checkpoint_start = time.perf_counter()
                    checkpoint = output / "checkpoints" / f"through_{chunk_stop:06d}"
                    checkpoints.append(
                        preserve_checkpoint(tracker, parent, checkpoint, checkpoint_binding)
                    )
                    checkpoint_seconds += time.perf_counter() - checkpoint_start
                    original.save_json(output / "checkpoint_progress.json", checkpoints)
                    original.save_json(output / "GPU_telemetry.json", telemetry)
                    print(
                        json.dumps(
                            dict(
                                event="checkpoint_preserved",
                                through_frame=chunk_stop,
                                restore_count=tracker.restore_count,
                            )
                        ),
                        flush=True,
                    )
            checks.require(
                tracker.previous_frame == STOP - 1
                and tracker.initialize_count == 1
                and tracker.restore_count == parent["metadata"]["restore_count"] + 1,
                "completed lifecycle differs",
            )
            expected = [
                native for first, stop in DIAGNOSTIC_RANGES for native in range(first, stop)
            ]
            checks.require(
                [r["frame_idx"] for r in diagnostic.records] == expected,
                "diagnostic coverage differs",
            )
            peak_allocated, peak_reserved = (
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            )
            tracker.__dict__.get("_fit_graph_cache", {}).clear()
            torch.cuda.synchronize()
            candidate = tracker.finish()
            np.savez_compressed(output / "candidate.npz", **candidate)
            original.save_json(output / "final_checkpoint.json", checkpoints[-1])
            keep = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
            checks.require(int(keep.sum()) == 9960, "one minute frame count differs")
            result.update(
                status="ok",
                one_minute_completed=True,
                native_frames=9960,
                logical_initialization_count=tracker.initialize_count,
                restore_count=tracker.restore_count,
                preparation_seconds=preparation_seconds,
                advance_seconds=GPU_seconds,
                reader_seconds=reader_seconds,
                checkpoint_seconds=checkpoint_seconds,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                checkpoint_count=len(checkpoints),
                diagnostic_frames=len(expected),
                direct_node_support=candidate["observed"][keep].sum((0, 1)).tolist(),
                identity_review_fish_frames=int(candidate["identity_review"][keep].sum()),
            )
        finally:
            if iterator is not None:
                iterator.close()
            if diagnostic is not None:
                original.save_diagnostics(output, diagnostic.records)
            original.save_json(output / "summary.json", result)
            original.save_json(output / "chest_proposal_diagnostics.json", proposal_rows)
            original.save_json(output / "GPU_telemetry.json", telemetry)
            original.save_json(output / "mask_validation_receipts.json", store.uses)
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
            for path in output.rglob("*"):
                if path.is_file() and path != output / "tracking.provenance.json":
                    record.output(str(path.relative_to(output)), path, sha256=checks.sha(path))
            original.unchanged(code)
            evidence.final_guard()
            store.final_guard()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "pilot",
        "short",
        "receipt-root",
        "generation",
        "reference-supervisor-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--generation-sha", required=True)
    parser.add_argument("--reference-child-pid", type=int, required=True)
    main(parser.parse_args())
