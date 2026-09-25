"""Run a fixed 9960-frame directional branch with one imported complete state.

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
from chest_candidate_diagnostics import CandidateDiagnostics
from continuous_tracker import RECORD_FIELDS, AcceleratedContinuousPersistentNodeTracker
from directional_chest_tracker import DirectionalChestContinuousTracker, import_original_state
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
DIAGNOSTIC_RANGES = ((66656, 66706), (75047, 75163))
checks = base.evidence_tools


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
    pilot_result = checks.read(pilot / "summary.json")
    short_result = checks.read(short / "summary.json")
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
        schema="directional_one_minute_v1",
        status="incomplete",
        native_interval=[START, STOP],
        one_minute_completed=False,
        independent_ID_truth=False,
        adopted=False,
        external_pose_resets=0,
        periodic_checkpoints_are_not_reinitializations=True,
    )
    with provenance("directional_one_minute_continuous", output, name="tracking") as record:
        try:
            original.snapshot(code, output, record)
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
            tracker = DirectionalChestContinuousTracker(rig, seed, cfg, **kwargs)
            migration = import_original_state(tracker, source, parent["arrays"], parent["metadata"])
            original.save_json(output / "state_import.json", migration)
            checks.require(
                len(tracker._states) == 6 and tracker.previous_frame == START - 1,
                "initial full beam boundary differs",
            )
            checkpoint_binding = dict(
                schema="directional_one_minute_checkpoint_binding_v1",
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
    for name in ("output", "pilot", "short", "receipt-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
