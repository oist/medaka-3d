"""Fixed306 serial replay of FastTail against all retained frozen Tail beams.

The first through68060 checkpoint supplies per-native history, never its future
backtrace winner. One original66400 state is imported once; flow stays serial.
Only ROOT executes this driver after the separate code and CPU-test review.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import audit_tail_first_chunk as tail_audit
import benchmark_fast_directional_fitter_v2 as qualification
import fit_chest_recovery_candidates as base
import numpy as np
import replay_chest_deficit as original
import run_tail_one_minute as tail_run
import torch
from continuous_tracker import RECORD_FIELDS, AcceleratedContinuousPersistentNodeTracker
from fast_tail_recovery_tracker import (
    FastTailRecoveryContinuousTracker,
    import_original_fast_tail_state,
)
from fit_tail_packets import bind_input, member
from mask_validation_receipts import ReceiptStore
from node_recovery import NodeRecoveryConfig
from persistent_node_tracker import PersistentNodeConfig
from replay_gpu_window import ReplayGPUWindow
from tracker import TrackerConfig

from medaka3d.provenance import provenance

checks, REPO = base.evidence_tools, base.REPO
WORK = REPO / "work/three_body"
START, STOP = 66400, 66706
STAGE = "fast_tail_serial_replay_306_v1"
QUALIFIED = WORK / "fast_directional_fitter_paired_301_v2"
QUALIFIED_SHA = "21244298988acd69ac6bc59edc0a833298144af80f5582b6b06de888ac0bb58c"
REFERENCE = WORK / "tail_one_minute_66400-76360_v1"
REFERENCE_MANIFEST_SHA = "c9217aeaf86bd919b8de40090ff2255d5ba291e661e00f8ba84b5364d530befb"
REFERENCE_AUDIT = WORK / "tail_first_chunk_interim_audit_v1"
REFERENCE_AUDIT_SHA = "7f7fe6254e3881fa62ace428f39242cddd850386f24165a4de51c21fae7ff547"
MAX_RESERVED_BYTES = 4 * 1024**3
SOURCE_PINS = {
    "fast_tail_recovery_tracker.py": (
        "fbaa1a81f337e291f487886ace1fbceb7ac9adf95db86b82896904d42cafa3ad"
    ),
    "fast_directional_device_fitter.py": (
        "5804003d537fe5ad144927c9629bf12927fc0ce9dc460615a78125cfe16c4abd"
    ),
    "directional_device_fitter.py": (
        "3228e3815966a5695fc8647a9b8310b28dea9368e177fd4941d20b35024c45aa"
    ),
    "tail_recovery_tracker.py": "b5a98085c6b1b03bd17e2f0b627ffc3eebdd569cf5add545146d9322761868a2",
    "benchmark_fast_directional_fitter_v2.py": (
        "684bbe0960545ec5da39f93945f4e558150a9b60948df8f0b3e870e888df8a71"
    ),
    "run_tail_one_minute.py": "3c11909ab244cfdb1bae3589baa987a9102827111372db00c853d4666ca18ca8",
}


def validate_execution_args(args):
    checks.require(
        type(args.owned_session) is int
        and args.owned_session == 77700
        and type(args.owned_exit_code) is int
        and args.owned_exit_code == 0,
        "successful ROOT-owned benchmark session77700 exit0 required",
    )
    checks.require(
        type(args.reference_child_pid) is int and args.reference_child_pid > 0,
        "positive explicit owned reference child PID required",
    )


def validate_qualification(summary, parities):
    checks.require(
        summary.get("schema") == qualification.STAGE
        and summary.get("status") == "ok"
        and summary.get("passed") is True
        and summary.get("frames") == list(qualification.FRAMES)
        and summary.get("CPU_reference_frames") == list(qualification.CPU_FRAMES)
        and all(
            summary.get(key) == 602
            for key in (
                "fitted_calls",
                "fit_calls_attempted",
                "fit_calls_returned",
                "validated_fit_calls",
            )
        )
        and summary.get("all77_arrays_per_arm_pass") is True
        and summary.get("paired_all77_pass") is True
        and all(
            summary.get(key) == 0 for key in ("initialize_calls", "restore_calls", "advance_calls")
        )
        and summary.get("new_tracking") is False
        and set(parities) == set(qualification.FRAMES),
        "complete successful fixed301 CUDA qualification required",
    )
    for native in qualification.FRAMES:
        value = parities[native]
        checks.require(
            isinstance(value, dict)
            and set(value) == {*qualification.ARMS, "paired"}
            and all(
                isinstance(value[arm], dict) and isinstance(value[arm].get("CUDA_golden"), dict)
                for arm in qualification.ARMS
            )
            and isinstance(value["paired"], dict),
            "all77 original/fast/paired qualification groups required",
        )
        groups = [value[arm]["CUDA_golden"] for arm in qualification.ARMS] + [value["paired"]]
        keys = set(groups[0])
        checks.require(
            len(keys) == 77
            and all(set(group) == keys for group in groups)
            and all(
                item["byte_exact"] is True and item["maximum_absolute_difference"] == 0
                for group in groups
                for item in group.values()
            ),
            "all77 CUDA original/fast/golden arrays must be byte exact",
        )
        if native in qualification.CPU_FRAMES:
            for arm in qualification.ARMS:
                cpu = value[arm].get("CPU_reference")
                checks.require(
                    isinstance(cpu, dict)
                    and set(cpu) == keys
                    and all(
                        np.isfinite(item["maximum_absolute_difference"])
                        and 0 <= item["maximum_absolute_difference"] <= 1e-10
                        for item in cpu.values()
                    ),
                    "all77 CPU4 qualification differs",
                )


def load_qualification(hashes, code):
    seal_path = QUALIFIED / "benchmark.provenance.json"
    checks.require(checks.sha(seal_path) == QUALIFIED_SHA, "fixed301 qualification seal changed")
    seal = tail_audit.verify_seal(QUALIFIED, seal_path.name, qualification.STAGE, hashes)
    checks.require(hashes[str(seal_path)] == QUALIFIED_SHA, "qualification changed during read")
    for name in (
        "benchmark_fast_directional_fitter_v2.py",
        "fast_directional_device_fitter.py",
        "directional_device_fitter.py",
        "fit_tail_packets.py",
    ):
        bind_input(seal, WORK / name, hashes)
    for source in set(code) & set(seal["inputs"]):
        bind_input(seal, source, hashes)
    summary = checks.read(member(QUALIFIED, "summary.json", hashes))
    parities = {
        native: checks.read(member(QUALIFIED, f"f{native}_parity.json", hashes))
        for native in qualification.FRAMES
    }
    validate_qualification(summary, parities)
    return summary


def verify_prefix(arrays, parent_arrays):
    prefix = START - original.FIRST
    for key in ("history_offsets", *("history_" + field for field in RECORD_FIELDS)):
        end = prefix + 1 if key == "history_offsets" else int(parent_arrays["history_offsets"][-1])
        actual, expected = arrays[key][:end], parent_arrays[key]
        exact = (
            actual.shape == expected.shape and np.array_equal(actual, expected)
            if actual.dtype.kind == expected.dtype.kind == "U"
            else original.byte_equal(actual, expected)
        )
        checks.require(exact, "original retained history prefix changed: " + key)


def load_reference(hashes, context):
    seal_path = REFERENCE_AUDIT / "audit.provenance.json"
    checks.require(checks.sha(seal_path) == REFERENCE_AUDIT_SHA, "fixed Tail audit seal changed")
    seal = tail_audit.verify_seal(REFERENCE_AUDIT, seal_path.name, tail_audit.STAGE, hashes)
    checks.require(hashes[str(seal_path)] == REFERENCE_AUDIT_SHA, "Tail audit changed during read")
    summary = checks.read(member(REFERENCE_AUDIT, "summary.json", hashes))
    arrays, metadata, fixed, migration, row = tail_audit.committed(
        REFERENCE, REFERENCE_MANIFEST_SHA, hashes
    )
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        tail_audit.input_member(seal, REFERENCE / "checkpoints/through_068060" / name, hashes)
    checks.require(
        summary["schema"] == tail_audit.STAGE
        and summary["status"] == "ok"
        and summary["native_interval"] == [START, 68060]
        and summary["horizon_stop"] == 68060
        and summary["expected_manifest_sha256"] == REFERENCE_MANIFEST_SHA
        and summary["all39_arrays_semantically_decoded"] is True
        and summary["all6_ordered_history_saved"] is True
        and summary["initial_parent_history_prefix_preserved"] is True
        and summary["shared_input_binding"] == fixed["original_binding"] == context["binding"],
        "audited Tail checkpoint must share the exact original parent/input binding",
    )
    parent = context["parent"]
    checks.require(
        parent["metadata"]["restore_count"] == 3 and len(parent["arrays"]) == 39,
        "complete39 original parent with restore3 required",
    )
    tail_audit.verify_tail_import(
        migration,
        parent["arrays"],
        parent["metadata"],
        metadata,
        {"context": migration["bounded_context"]},
    )
    verify_prefix(arrays, parent["arrays"])
    return dict(arrays=arrays, metadata=metadata, fixed=fixed, row=row)


def check_headroom(sample, *, startup=False):
    minimum = 6144 if startup else 2048
    checks.require(
        np.isfinite(sample["free_MiB"]) and sample["free_MiB"] >= minimum,
        f"less than {minimum}MiB measured free GPU memory",
    )


def check_memory(sample):
    checks.require(
        0 <= sample["allocated_bytes"] <= sample["reserved_bytes"] <= MAX_RESERVED_BYTES
        and 0
        <= sample["peak_allocated_bytes"]
        <= sample["peak_reserved_bytes"]
        <= MAX_RESERVED_BYTES,
        "single fitter exceeded fixed4GiB process budget",
    )


def check_lifecycle(tracker, next_native):
    checks.require(
        tracker.initialize_count == 1
        and tracker.restore_count == 4
        and tracker.previous_frame == next_native - 1
        and tracker.processed_frames == next_native - original.FIRST
        and len(tracker._states) == 6,
        "single imported full-six lifecycle changed or reset/restored during replay",
    )


def run_frames(tracker, frames, expected, *, before_frame, after_frame, on_result, synchronize):
    for native, frame in zip(range(START, STOP), frames, strict=True):
        checks.require(frame.frame_idx == native, "native frame skipped or reordered")
        check_lifecycle(tracker, native)
        before_frame(native)
        started = time.perf_counter()
        tracker.advance(frame)
        synchronize()
        elapsed = time.perf_counter() - started
        check_lifecycle(tracker, native + 1)
        after_frame(native)
        comparison = original.compare_records(tracker._history[-1], expected, native)
        row = dict(**comparison, advance_seconds=elapsed)
        on_result(row)
        checks.require(
            comparison["all_beams"] == 6 and comparison["exact"],
            "all-six retained Tail history diverged at native " + str(native),
        )


def main(args):
    validate_execution_args(args)
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(WORK) and not output.exists(), "new workspace output required"
    )
    code = base.local_code()
    checks.require(str(Path(__file__).resolve()) in code, "driver source not registered")
    for name, digest in SOURCE_PINS.items():
        checks.require(code.get(str(WORK / name)) == digest, "frozen source differs: " + name)
    hashes, outputs = dict(code), {}
    output.mkdir(parents=True)
    evidence = original.phase.Digests()
    store = ReceiptStore(args.receipt_root.resolve())
    tracker = iterator = guard = None
    rows, telemetry, memories, diagnostics = [], [], [], []
    summary = dict(
        schema=STAGE,
        status="incomplete",
        replay_complete=False,
        native_interval=[START, STOP],
        GPU_used=False,
        GPU_initialization_attempted=False,
        external_pose_resets=0,
        independent_ID_truth=False,
        adopted=False,
        qualification_owned_session=args.owned_session,
        qualification_owned_exit_code=args.owned_exit_code,
        qualification_seal_sha256=QUALIFIED_SHA,
        reference_manifest_sha256=REFERENCE_MANIFEST_SHA,
        reference_audit_seal_sha256=REFERENCE_AUDIT_SHA,
    )
    with provenance(STAGE, output, name="replay") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            outputs[str(path)] = checks.sha(path)
            record.output(name, path, sha256=outputs[str(path)])

        try:
            load_qualification(hashes, code)
            for path, digest in hashes.items():
                evidence.add(path, digest)
            original.snapshot(code, output, record)
            for path, digest in code.items():
                snapshot = output / "code_snapshot" / Path(path).relative_to(REPO)
                checks.require(checks.sha(snapshot) == digest, "source snapshot changed")
                emit(snapshot.relative_to(output).as_posix())
            context = original.prepare(evidence, store)
            reference = load_reference(hashes, context)
            for path, digest in hashes.items():
                evidence.add(path, digest)
            binding, protocol, parent = context["binding"], context["protocol"], context["parent"]
            torch.set_num_threads(binding["torch_cpu_threads"])
            original.runtime(binding, gpu=False)
            observed = output / "registered_reference"
            observed.mkdir()
            guard = ReplayGPUWindow(
                args.reference_supervisor_root, args.reference_child_pid, observed
            )
            qualification.attach_startup_reference(guard)
            record.param("reference_ownership_verified_at_start", True)
            record.param("reference_natural_phase_transitions_allowed", True)
            sample = qualification.gpu_sample()
            telemetry.append(dict(stage="startup", **sample))
            check_headroom(sample, startup=True)
            summary["GPU_initialization_attempted"] = True
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
            summary["GPU_used"] = True
            torch.cuda.reset_peak_memory_stats()
            original.runtime(binding, gpu=True)
            cfg = TrackerConfig(**protocol["tracker_config"])
            kwargs = dict(
                recovery_config=NodeRecoveryConfig(**protocol["node_recovery_config"]),
                persistence_config=PersistentNodeConfig(**protocol["persistence_config"]),
            )
            rig, seed = context["inputs"].rig, context["inputs"].initial_pose_xyz.copy()
            source = AcceleratedContinuousPersistentNodeTracker(rig, seed, cfg, **kwargs)
            tracker = FastTailRecoveryContinuousTracker(rig, seed, cfg, **kwargs)
            migration = import_original_fast_tail_state(
                tracker, source, parent["arrays"], parent["metadata"]
            )
            checks.require(
                migration["tail_context"] == reference["metadata"]["context"],
                "FastTail changes more than execution context from saved Tail",
            )
            restored, _ = tracker.export_state()
            checks.require(
                all(original.byte_equal(restored[k], v) for k, v in parent["arrays"].items()),
                "import changed original39 arrays including previous images",
            )
            check_lifecycle(tracker, START)
            emit("state_import.json", migration)
            emit(
                "ready.json",
                dict(
                    native_interval=[START, STOP],
                    initialize_count=1,
                    restore_count=4,
                    previous_observation=context["previous_observation"],
                    parent_arrays_byte_preserved=39,
                ),
            )
            record.param("original_binding", binding)
            record.param("allocator_fraction_cap", 0.25)
            record.param("maximum_process_reserved_bytes", MAX_RESERVED_BYTES)
            record.param("headroom_measurement", "nvidia-smi memory.free; startup6GiB/frame2GiB")
            record.param("serial_flow_unchanged", True)
            evidence.final_guard()
            store.final_guard()
            original.unchanged(code)
            iterator = original.bounded_frames(context["reader"], first=START, stop=STOP)

            def before_frame(native):
                sample = qualification.gpu_sample()
                telemetry.append(dict(frame_idx=native, stage="before_frame", **sample))
                check_headroom(sample)

            def after_frame(native):
                memory = qualification.memory_sample()
                memories.append(dict(frame_idx=native, **memory))
                check_memory(memory)
                sample = qualification.gpu_sample()
                telemetry.append(dict(frame_idx=native, stage="after_frame", **sample))
                check_headroom(sample)

            def on_result(row):
                rows.append(row)
                diagnostics.append(
                    dict(
                        frame_idx=row["frame_idx"],
                        tail_recovery=copy.deepcopy(tracker.last_tail_recovery_diagnostic),
                        bounded_mouth=copy.deepcopy(tracker.last_bounded_mouth_diagnostic),
                    )
                )
                emit("history_comparison.json", rows)
                if not row["exact"]:
                    emit("first_difference.json", row)
                if len(rows) % 25 == 0 or row["frame_idx"] == STOP - 1:
                    print(json.dumps(dict(event="fast_tail_replay_progress", **row)), flush=True)

            run_frames(
                tracker,
                iterator,
                reference["arrays"],
                before_frame=before_frame,
                after_frame=after_frame,
                on_result=on_result,
                synchronize=torch.cuda.synchronize,
            )
            iterator.close()
            iterator = None
            check_lifecycle(tracker, STOP)
            arrays, _metadata = tracker.export_state()
            verify_prefix(arrays, parent["arrays"])
            checkpoint_binding = dict(
                schema="fast_tail_replay_checkpoint_binding_v1",
                original_binding=binding,
                experimental_code_sha256=code,
                state_import_sha256=checks.sha(output / "state_import.json"),
                native_interval=[START, STOP],
                qualification_seal_sha256=QUALIFIED_SHA,
                reference_manifest_sha256=REFERENCE_MANIFEST_SHA,
            )
            checkpoint = output / "checkpoint"
            checkpoint_row = tail_run.preserve_checkpoint(
                tracker, parent, checkpoint, checkpoint_binding
            )
            for name in ("arrays.npz", "metadata.json", "manifest.json"):
                emit("checkpoint/" + name)
            checks.require(
                len(rows) == 306 and all(r["exact"] and r["all_beams"] == 6 for r in rows),
                "incomplete fixed306 all-six replay",
            )
            summary.update(
                status="ok",
                replay_complete=True,
                exact_frames=len(rows),
                all6_ordered_history19_byte_exact=True,
                compared_fields=list(RECORD_FIELDS),
                final_previous_frame=tracker.previous_frame,
                logical_initialization_count=tracker.initialize_count,
                checkpoint_restore_count=tracker.restore_count,
                restores_during_interval=0,
                new_begin_calls=0,
                parent_history_prefix_preserved=True,
                checkpoint=checkpoint_row,
                final_arrays=len(arrays),
                previous_observation_byte_preserved_at_import=True,
                final_winner_prefix_used_as_replay_criterion=False,
                numeric_bytes_exact=True,
                unicode_labels_exact_ignoring_capacity=True,
                serial_flow_unchanged=True,
                memory=qualification.memory_sample(),
                summed_advance_seconds=sum(r["advance_seconds"] for r in rows),
                timing_scope=(
                    "advance including serial flow and CUDA completion; excludes I/O/telemetry"
                ),
                timing_is_concurrent_observation=True,
                qualification_scope="306 sequential frames only; no full-minute or identity claim",
            )
        except BaseException as error:
            summary.update(status="failed", error=repr(error), completed_comparisons=len(rows))
            raise
        finally:
            if iterator is not None:
                iterator.close()
            if guard is not None:
                guard.close()
                for path in (output / "registered_reference").iterdir():
                    if path.is_file():
                        emit(str(path.relative_to(output)))
            if tracker is not None:
                summary.update(
                    logical_initialization_count=tracker.initialize_count,
                    checkpoint_restore_count=tracker.restore_count,
                    final_previous_frame=tracker.previous_frame,
                    processed_frames=tracker.processed_frames,
                )
            for name, value in (
                ("summary.json", summary),
                ("history_comparison.json", rows),
                ("proposal_diagnostics.json", diagnostics),
                ("GPU_telemetry.json", telemetry),
                ("memory.json", memories),
                ("mask_validation_receipts.json", store.uses),
            ):
                emit(name, value)
            checks.require(code == base.local_code(), "loaded source set or bytes changed")
            for path, digest in hashes.items():
                evidence.add(path, digest)
            evidence.final_guard()
            store.final_guard()
            for path, digest in evidence.paths.items():
                record.input(path, path, sha256=digest)
            for path, digest in outputs.items():
                checks.require(checks.sha(path) == digest, "emitted output changed")
            record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "receipt-root", "reference-supervisor-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("reference-child-pid", "owned-session", "owned-exit-code"):
        parser.add_argument("--" + name, type=int, required=True)
    main(parser.parse_args())
