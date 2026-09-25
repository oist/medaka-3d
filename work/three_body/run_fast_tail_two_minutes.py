"""One original66400 import, 19920 serial Fast-FP64 frames, twelve checkpoints.

No detector, reset, intermediate finish, Full-fit graph or mixed precision.
Saved Tail states are controls only. ROOT supplies a current shared-load note;
this process never opens, polls or signals another process handle. Source review
and the focused CPU fixtures precede any --execute invocation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import replay_fast_tail as qualified
import torch
from fast_tail_recovery_tracker import (
    FastTailRecoveryContinuousTracker,
    import_original_fast_tail_state,
)
from tracker_checkpoint import read_checkpoint, write_checkpoint

from medaka3d.provenance import provenance

original, checks = qualified.original, qualified.checks
WORK, REPO = qualified.WORK, qualified.REPO
FIRST, START, CONTROL_STOP, STOP, CHUNK = 49800, 66400, 76360, 86320, 1660
STAGE = "fast_tail_two_minute_continuous_v1"
PLAN = WORK / "FAST_TAIL_TWO_MINUTE_EXECUTION_CONTRACT.md"
PLAN_SHA = "097277d2a28c43edccafba39f96c779890bf553b8d83996051b2bf98198814b4"
PARENT_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
TAIL = WORK / "tail_one_minute_66400-76360_v1"
TAIL_SHA = "50ccc8d2a4bbcc7e09aa2b512b70374fb9b37a4d4ece0d0939322c66bfa59416"
FAST = WORK / "fast_tail_serial_replay_306_v1"
FAST_SHA = "c86ded53f52fcf2853745a55377cd01bfdb0392587c318ada7b1772d2a7ced36"
CONTROL_MANIFESTS = (
    "c9217aeaf86bd919b8de40090ff2255d5ba291e661e00f8ba84b5364d530befb",
    "c1135e290820bb31199c2cab0eaea44601b6a4d32775e97e343eaf1c996cfbe1",
    "4790ae7886edb0686477ba9ac1bb2ecdac5b500fa41798f963ba04c32bd79cff",
    "da622c4d06ba192bbb8971c38f1a6f9a647dd54a96b02202ea30e3187f03003e",
    "d62878afffe24094a03ae3045e0d8767c6601d3fcb2c7c5506181ef724d00f35",
    "4ad8f5eb0b381bcaeee107aec3b4c62ad7b900373e9d8e7c0437898313593f8b",
)
SOURCE_PINS = {
    **qualified.SOURCE_PINS,
    "replay_fast_tail.py": "5c133617c16fd274d8d088839da1874d50b8774ce03caeafce6aaa739ba5d944",
    "replay_chest_deficit.py": "80227f45ff0217d436166ff9c04919192de3d0c442726006b87b4f134e114bff",
    "receipted_long_phase_audit.py": (
        "44438d6d36668e96cd311152df6b811a1c0a5fb8ec1277667d8fab9956414774"
    ),
}


def chunks():
    return tuple((n, n + CHUNK) for n in range(START, STOP, CHUNK))


def local_code():
    """Real loaded local files; dynamic Torch metadata cannot create fake paths."""
    paths = {Path(__file__).resolve(), PLAN}
    scopes = (WORK, REPO / "labs/20260909_three_body", REPO / "src")
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__dict__", {}).get("__file__")
        if not isinstance(name, str) or not Path(name).is_absolute():
            continue
        path = Path(name).resolve()
        if path.suffix == ".py" and any(path.is_relative_to(p) for p in scopes):
            paths.add(path)
    return {str(path): checks.sha(path) for path in paths}


def source_contract(code):
    checks.require(code.get(str(PLAN)) == PLAN_SHA, "approved two-minute contract changed")
    checks.require(str(Path(__file__).resolve()) in code, "caller source missing")
    for name, digest in SOURCE_PINS.items():
        checks.require(code.get(str(WORK / name)) == digest, "qualified source changed: " + name)


def shared_load_note(note):
    """A ROOT observation, not an ownership probe or an exclusive GPU reservation."""
    checks.require(
        note.get("schema") == "root_shared_GPU_load_observation_v1"
        and note.get("observer") == "ROOT"
        and isinstance(note.get("observed_at"), str)
        and bool(note["observed_at"])
        and note.get("natural_phase_transitions_allowed") is True,
        "explicit current ROOT shared-load observation required",
    )
    rows = note.get("workloads")
    roles = {"long_reference", "D2_producer", "held_exit_witness"}
    checks.require(
        isinstance(rows, list) and len(rows) == 3 and {r.get("role") for r in rows} == roles,
        "three declared ROOT workload roles required",
    )
    for row in rows:
        checks.require(
            type(row.get("owned_session")) is int
            and row["owned_session"] > 0
            and row.get("status") in {"LIVE", "CLOSED0", "CLOSED1"}
            and (row.get("pid") is None or (type(row["pid"]) is int and row["pid"] > 0)),
            "invalid ROOT workload observation",
        )
    return copy.deepcopy(note)


def read_seal(root, name, stage, digest, evidence):
    """Bind small seal and index aliases; hash only payloads actually consumed."""
    root = root.resolve(strict=True)
    path = root / name
    evidence.add(path, digest)
    seal = checks.read(path)
    checks.require(
        seal["status"] == "ok" and seal["stage"] == stage, "successful fixed seal required"
    )
    index = {}
    for item in seal["outputs"].values():
        member = Path(item["path"]).resolve()
        checks.require(member.is_relative_to(root), "sealed output escaped root")
        previous = index.get(str(member))
        checks.require(
            previous is None
            or (previous["sha256"], previous["size"]) == (item["sha256"], item["size"]),
            "conflicting sealed output aliases",
        )
        index[str(member)] = item
    return dict(root=root, seal=seal, index=index)


def consumed(bundle, relative, evidence):
    path = (bundle["root"] / relative).resolve(strict=True)
    item = bundle["index"].get(str(path))
    checks.require(
        item is not None and path.stat().st_size == item["size"], "unsealed or resized input"
    )
    evidence.add(path, item["sha256"])
    return path


def validate_qualification(summary):
    required = dict(
        schema=qualified.STAGE,
        status="ok",
        replay_complete=True,
        native_interval=[START, 66706],
        exact_frames=306,
        all6_ordered_history19_byte_exact=True,
        logical_initialization_count=1,
        checkpoint_restore_count=4,
        restores_during_interval=0,
        new_begin_calls=0,
        external_pose_resets=0,
        previous_observation_byte_preserved_at_import=True,
        numeric_bytes_exact=True,
        unicode_labels_exact_ignoring_capacity=True,
        serial_flow_unchanged=True,
        qualification_seal_sha256=qualified.QUALIFIED_SHA,
        compared_fields=list(qualified.RECORD_FIELDS),
    )
    checks.require(
        all(summary.get(k) == v for k, v in required.items()), "fixed306 qualification differs"
    )


def metadata_inputs(evidence, code):
    fast = read_seal(FAST, "replay.provenance.json", qualified.STAGE, FAST_SHA, evidence)
    validate_qualification(checks.read(consumed(fast, "summary.json", evidence)))
    checks.require(
        fast["seal"]["params"]["all_source_input_snapshot_output_SHA_unchanged"] is True,
        "qualified replay final guards absent",
    )
    for name, digest in SOURCE_PINS.items():
        key = str(WORK / name)
        checks.require(
            fast["seal"]["inputs"].get(key, {}).get("sha256") == digest == code[key],
            "source not bound by qualified replay: " + name,
        )
    tail = read_seal(
        TAIL, "tracking.provenance.json", "tail_one_minute_continuous", TAIL_SHA, evidence
    )
    summary = checks.read(consumed(tail, "summary.json", evidence))
    checks.require(
        summary["status"] == "ok"
        and summary["one_minute_completed"] is True
        and summary["native_frames"] == CONTROL_STOP - START
        and summary["logical_initialization_count"] == 1
        and summary["restore_count"] == 4
        and summary["external_pose_resets"] == 0,
        "complete unchanged Tail minute required",
    )
    progress = checks.read(consumed(tail, "checkpoint_progress.json", evidence))
    checks.require(len(progress) == 6, "six committed Tail controls required")
    headers = []
    for (start, stop), digest, row in zip(chunks()[:6], CONTROL_MANIFESTS, progress, strict=True):
        relative = f"checkpoints/through_{stop:06d}"
        manifest_path = consumed(tail, relative + "/manifest.json", evidence)
        checks.require(
            checks.sha(manifest_path) == digest == row["manifest_sha256"],
            "Tail endpoint pin changed",
        )
        manifest = checks.read(manifest_path)
        fixed = manifest["bindings"]
        meta = checks.read(consumed(tail, relative + "/metadata.json", evidence))
        checks.require(
            Path(row["path"]).resolve() == TAIL / relative
            and fixed["schema"] == "tail_one_minute_checkpoint_binding_v1"
            and fixed["native_interval"] == [START, CONTROL_STOP]
            and row["arrays"] == 39
            and meta["initialize_count"] == 1
            and meta["restore_count"] == 4
            and meta["previous_frame"] == stop - 1
            and meta["processed_frames"] == stop - FIRST,
            "Tail checkpoint scope/lifecycle changed",
        )
        for key in ("previous_frame", "processed_frames", "initialize_count", "restore_count"):
            checks.require(row[key] == meta[key], "Tail commit header differs")
        headers.append(
            dict(start=start, stop=stop, relative=relative, binding=fixed, metadata=meta)
        )
    initial_context = headers[0]["metadata"]["context"]
    fixed = headers[0]["binding"]
    checks.require(
        all(h["binding"] == fixed and h["metadata"]["context"] == initial_context for h in headers),
        "Tail context or binding changes within minute",
    )
    migration = checks.read(consumed(tail, "state_import.json", evidence))
    checks.require(
        fixed["state_import_sha256"] == evidence.paths[str(TAIL / "state_import.json")],
        "Tail migration binding changed",
    )
    for path, digest in fixed["experimental_code_sha256"].items():
        evidence.add(path, digest)
        snapshot = consumed(
            tail, "code_snapshot/" + Path(path).relative_to(REPO).as_posix(), evidence
        )
        checks.require(evidence.paths[str(snapshot)] == digest, "Tail source snapshot differs")
    return dict(fast=fast, tail=tail, headers=headers, migration=migration)


def prepare(evidence, store, controls):
    evidence.add(original.PARENT / "continuation.provenance.json", PARENT_SHA)
    parent_root, seal = original.phase.sealed(original.PARENT, evidence)
    binding = seal["params"]["bindings"]
    checks.require(
        binding
        == controls["headers"][0]["binding"]["original_binding"]
        == controls["fast"]["seal"]["params"]["original_binding"],
        "parent, Tail control and Fast qualification scientific bindings differ",
    )
    inputs, protocol = original.verified_inputs(binding, STOP, evidence, receipt_store=store)
    checks.require(
        len(inputs.chunk_roots) == len(inputs.chunk_records) == 22, "fixed22 input prefix required"
    )
    selected = inputs.chunk_records[(START - FIRST) // CHUNK :]
    checks.require(
        [(r["start"], r["stop"]) for r in selected] == list(chunks()),
        "twelve sealed input chunks differ",
    )
    parent = original.phase.load_phase(parent_root, seal, inputs, evidence, protocol)
    checks.require(
        parent["metadata"]["initialize_count"] == 1
        and parent["metadata"]["restore_count"] == 3
        and parent["metadata"]["previous_frame"] == START - 1
        and parent["metadata"]["processed_frames"] == START - FIRST,
        "fixed original66400 parent required",
    )
    qualified.tail_audit.verify_tail_import(
        controls["migration"],
        parent["arrays"],
        parent["metadata"],
        controls["headers"][0]["metadata"],
        {"context": controls["migration"]["bounded_context"]},
    )
    previous = original.verify_previous_observation(parent, inputs)
    store.final_guard()
    for path, digest in store.evidence_hashes.items():
        evidence.add(path, digest)
    return dict(inputs=inputs, protocol=protocol, binding=binding, parent=parent, previous=previous)


def load_control(stop, controls, evidence):
    if stop > CONTROL_STOP:
        return None
    header = controls["headers"][(stop - START) // CHUNK - 1]
    checks.require(header["stop"] == stop, "wrong control endpoint")
    root = TAIL / header["relative"]
    consumed(controls["tail"], header["relative"] + "/arrays.npz", evidence)
    arrays, meta, _ = read_checkpoint(root, expected_bindings=header["binding"])
    checks.require(len(arrays) == 39 and meta == header["metadata"], "Tail codec/header differs")
    return dict(arrays=arrays, metadata=meta)


def compare_checkpoint(arrays, metadata, expected, migration):
    fields = {}
    keys = set(arrays) | set(expected["arrays"])
    for key in sorted(keys):
        a, b = arrays.get(key), expected["arrays"].get(key)
        unicode = (
            a is not None
            and b is not None
            and a.dtype.kind == b.dtype.kind == "U"
            and key.startswith("history_")
        )
        exact = (
            a is not None
            and b is not None
            and (
                a.shape == b.shape and np.array_equal(a, b)
                if unicode
                else original.byte_equal(a, b)
            )
        )
        fields[key] = dict(
            exact=bool(exact),
            unicode_value_comparison=unicode,
            actual_dtype=None if a is None else a.dtype.str,
            expected_dtype=None if b is None else b.dtype.str,
        )
    adapted = copy.deepcopy(metadata)
    context_valid = adapted.get("context") == migration["new_context"]
    adapted["context"] = migration["tail_context"]
    metadata_exact = context_valid and adapted == expected["metadata"]
    return dict(
        exact=len(keys) == len(arrays) == len(expected["arrays"]) == 39
        and all(v["exact"] for v in fields.values())
        and metadata_exact,
        all39=fields,
        all19_history_and_offsets_included=True,
        metadata_exact_after_declared_fast_context=metadata_exact,
        allowed_metadata_difference=["context.implementation", "context.fast_execution_policy"],
    )


def compare_candidate(actual, expected, stop):
    keys = set(actual) | set(expected)
    fields = {
        k: k in actual and k in expected and original.byte_equal(actual[k], expected[k])
        for k in keys
    }
    return dict(
        exact=len(keys) == len(actual) == len(expected) == 21
        and all(fields.values())
        and np.array_equal(actual["frame_idx"], np.arange(FIRST, stop, dtype=np.int64)),
        fields=fields,
        horizon_stop=stop,
        all21_bytes_required=True,
    )


def preserve_checkpoint(tracker, parent, path, binding):
    arrays, metadata = tracker.export_state()
    qualified.verify_prefix(arrays, parent["arrays"])
    checks.require(len(arrays) == 39, "complete39 checkpoint required")
    write_checkpoint(path, arrays, metadata, bindings=binding)
    loaded, loaded_meta, manifest = read_checkpoint(path, expected_bindings=binding)
    checks.require(
        loaded_meta == metadata
        and set(loaded) == set(arrays)
        and all(original.byte_equal(loaded[k], v) for k, v in arrays.items()),
        "checkpoint roundtrip differs",
    )
    return (
        loaded,
        loaded_meta,
        dict(
            path=str(path),
            arrays=39,
            manifest_sha256=checks.sha(path / "manifest.json"),
            payload_bytes=manifest["payload"]["bytes"],
            **{
                k: metadata[k]
                for k in ("previous_frame", "processed_frames", "initialize_count", "restore_count")
            },
        ),
    )


def drive_chunks(
    tracker,
    *,
    reader_for,
    reference_for,
    save_checkpoint,
    before_frame,
    after_frame,
    on_frame,
    synchronize,
    timers,
    clock=time.perf_counter,
):
    """Single tracker, fixed19920 frames; no finish or restore anywhere here."""
    for first, stop in chunks():
        tick = clock()
        expected = reference_for(stop)
        timers["control_read_verify_seconds"] += clock() - tick
        tick = clock()
        reader = reader_for(first, stop)
        checks.require(
            reader.frames[0] == first and reader.frames[-1] + 1 == stop, "reader chunk differs"
        )
        timers["reader_ctor_seconds"] += clock() - tick
        iterator = original.bounded_frames(reader, first=first, stop=stop)
        try:
            for native in range(first, stop):
                qualified.check_lifecycle(tracker, native)
                tick = clock()
                frame = next(iterator)
                timers["native_next_seconds"] += clock() - tick
                checks.require(frame.frame_idx == native, "native read order differs")
                tick = clock()
                before_frame(native)
                timers["telemetry_seconds"] += clock() - tick
                tick = clock()
                tracker.advance(frame)
                synchronize()
                elapsed = clock() - tick
                timers["advance_seconds"] += elapsed
                qualified.check_lifecycle(tracker, native + 1)
                tick = clock()
                control = (
                    original.compare_records(tracker._history[-1], expected["arrays"], native)
                    if expected is not None
                    else None
                )
                timers["history_compare_seconds"] += clock() - tick
                on_frame(dict(frame_idx=native, advance_seconds=elapsed, control=control))
                checks.require(
                    control is None or (control["exact"] and control["all_beams"] == 6),
                    "retained all-six history diverged at " + str(native),
                )
                tick = clock()
                after_frame(native)
                timers["telemetry_seconds"] += clock() - tick
        finally:
            tick = clock()
            iterator.close()
            timers["native_next_seconds"] += clock() - tick
        save_checkpoint(stop, expected)
        qualified.check_lifecycle(tracker, stop)
    qualified.check_lifecycle(tracker, STOP)


def new_timers():
    return dict.fromkeys(
        (
            "source_and_metadata_seconds",
            "input_verification_seconds",
            "runtime_and_migration_seconds",
            "control_read_verify_seconds",
            "reader_ctor_seconds",
            "native_next_seconds",
            "advance_seconds",
            "history_compare_seconds",
            "checkpoint_save_readback_seconds",
            "checkpoint_control_seconds",
            "artifact_save_seconds",
            "telemetry_seconds",
            "pretracking_guard_seconds",
            "final_guard_seconds",
        ),
        0.0,
    )


def runtime(binding, *, gpu):
    result = original.runtime(binding, gpu=gpu)
    checks.require(torch.get_num_interop_threads() == 1, "qualified interop1 required")
    return {**result, "torch_interop_threads": 1}


def cuda_memory_record(free_bytes, total_bytes):
    """Pure conversion of the CUDA global-memory API; not allocator-only memory."""
    checks.require(
        type(free_bytes) is int
        and type(total_bytes) is int
        and 0 <= free_bytes <= total_bytes
        and total_bytes > 0,
        "invalid CUDA global memory values",
    )
    return dict(
        measurement_api="torch.cuda.mem_get_info(0)",
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        free_MiB=free_bytes / 1024**2,
        total_MiB=total_bytes / 1024**2,
    )


def cuda_headroom():
    return dict(
        **cuda_memory_record(*torch.cuda.mem_get_info(0)), sampled_monotonic=time.monotonic()
    )


def smi_sample():
    return dict(measurement_api="nvidia-smi", **qualified.qualification.gpu_sample())


def main(args):
    checks.require(args.execute is True, "explicit --execute required after source review")
    output = args.output.resolve()
    checks.require(
        output.is_relative_to(WORK) and not output.exists(), "new workspace output required"
    )
    note_path = args.shared_load_note.resolve(strict=True)
    checks.require(note_path.is_relative_to(WORK), "ROOT load note must be in workspace")
    checks.require(
        len(args.shared_load_sha) == 64
        and all(c in "0123456789abcdef" for c in args.shared_load_sha),
        "explicit ROOT load-note SHA required",
    )
    code = local_code()
    source_contract(code)
    output.mkdir(parents=True)
    wall_start = time.perf_counter()
    evidence = original.phase.Digests()
    store = qualified.ReceiptStore(args.receipt_root.resolve())
    tracker = context = None
    rows, checkpoints, controls_saved, telemetry, memories, proposals = [], [], [], [], [], []
    timers = new_timers()
    outputs = {}
    summary = dict(
        schema=STAGE,
        status="incomplete",
        two_minutes_completed=False,
        native_interval=[START, STOP],
        processed_full_history_interval=[FIRST, STOP],
        GPU_used=False,
        GPU_initialization_attempted=False,
        adopted=False,
        independent_ID_truth=False,
        external_pose_resets=0,
        restores_during_interval=0,
        new_begin_calls=0,
        source_execution_policy="qualified Fast FP64 only; Full and Mixed not used",
        periodic_checkpoints_are_not_reinitializations=True,
    )
    with provenance(STAGE, output, name="tracking") as record:

        def emit(name, value=None):
            tick = time.perf_counter()
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if value is not None:
                original.save_json(path, value)
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)
            timers["artifact_save_seconds"] += time.perf_counter() - tick

        try:
            tick = time.perf_counter()
            for path, digest in code.items():
                evidence.add(path, digest)
            evidence.add(note_path, args.shared_load_sha)
            note = shared_load_note(checks.read(note_path))
            controls = metadata_inputs(evidence, code)
            original.snapshot(code, output, record)
            for source, digest in code.items():
                snapshot = output / "code_snapshot" / Path(source).relative_to(REPO)
                checks.require(checks.sha(snapshot) == digest, "snapshot differs")
                emit(snapshot.relative_to(output).as_posix())
            timers["source_and_metadata_seconds"] = time.perf_counter() - tick
            emit(
                "plan.json",
                dict(
                    schema=STAGE,
                    contract_sha256=PLAN_SHA,
                    native_interval=[START, STOP],
                    processed_history_interval=[FIRST, STOP],
                    chunk_intervals=[list(c) for c in chunks()],
                    control_horizons=[c[1] for c in chunks()[:6]],
                    parent_seal_sha256=PARENT_SHA,
                    old_tail_seal_sha256=TAIL_SHA,
                    fast306_seal_sha256=FAST_SHA,
                    source_sha256=code,
                    shared_load_note=note,
                    shared_load_note_sha256=args.shared_load_sha,
                    shared_load_scope=(
                        "ROOT observation only; no other-process handle opened or polled"
                    ),
                    shared_load_is_exclusive_reservation=False,
                    natural_phase_transitions_allowed=True,
                    precision="float64",
                    adam_updates=14,
                    scores_per_fit=15,
                    fitter_cache_capacity=4,
                    allocator_fraction=0.25,
                    maximum_reserved_bytes=qualified.MAX_RESERVED_BYTES,
                    startup_free_MiB=6144,
                    frame_free_MiB=2048,
                    startup_memory_api="nvidia-smi",
                    frame_memory_api="torch.cuda.mem_get_info(0)",
                    utilization_power_sampling="startup/post-CUDA-init/twelve checkpoints",
                    post_init_API_values_need_not_match_under_shared_load=True,
                    future_reference_states_only_for_comparison=True,
                    independent_ID_truth=False,
                    intermediate_finish_forbidden=True,
                    no_automatic_retry=True,
                    final_86320_prefix_may_differ_from_76360_winner=True,
                    timing_scope={
                        "reader_ctor": (
                            "SL archive unpack/slice, masks, background, input SHA including raw"
                        ),
                        "native_next": (
                            "native video read/seek, BGR SHA, mask decode, SL lookup and close"
                        ),
                        "advance": (
                            "unchanged serial flow/proposals/Fast fit/update plus CUDA synchronize"
                        ),
                        "wall": (
                            "verification, telemetry, comparisons, serialization; shared load"
                        ),
                    },
                ),
            )
            tick = time.perf_counter()
            context = prepare(evidence, store, controls)
            timers["input_verification_seconds"] = time.perf_counter() - tick
            binding, protocol, parent = context["binding"], context["protocol"], context["parent"]
            emit(
                "input_index.json",
                dict(
                    prefix_chunks=context["inputs"].chunk_records,
                    tracking_chunks=context["inputs"].chunk_records[(START - FIRST) // CHUNK :],
                    previous_observation=context["previous"],
                    original_binding=binding,
                    original_parent_root=str(original.PARENT),
                    detector_started=False,
                ),
            )
            tick = time.perf_counter()
            original.cv2.setNumThreads(binding["opencv_threads"])
            torch.set_num_threads(binding["torch_cpu_threads"])
            torch.set_num_interop_threads(1)
            runtime(binding, gpu=False)
            sample = smi_sample()
            telemetry.append(dict(stage="startup", **sample))
            qualified.check_headroom(sample, startup=True)
            summary["GPU_initialization_attempted"] = True
            torch.cuda.set_per_process_memory_fraction(0.25, 0)
            summary["GPU_used"] = True
            torch.cuda.reset_peak_memory_stats()
            environment = runtime(binding, gpu=True)
            telemetry.append(dict(stage="post_CUDA_init_smi", **smi_sample()))
            initial_cuda_memory = cuda_headroom()
            telemetry.append(dict(stage="post_CUDA_init_global", **initial_cuda_memory))
            qualified.check_headroom(initial_cuda_memory)
            config = qualified.TrackerConfig(**protocol["tracker_config"])
            checks.require(
                config.opt_steps == 14 and config.device == "cuda",
                "qualified CUDA14 config required",
            )
            kwargs = dict(
                recovery_config=qualified.NodeRecoveryConfig(**protocol["node_recovery_config"]),
                persistence_config=qualified.PersistentNodeConfig(**protocol["persistence_config"]),
            )
            rig, seed = context["inputs"].rig, context["inputs"].initial_pose_xyz.copy()
            source = qualified.AcceleratedContinuousPersistentNodeTracker(
                rig, seed, config, **kwargs
            )
            tracker = FastTailRecoveryContinuousTracker(rig, seed, config, **kwargs)
            migration = import_original_fast_tail_state(
                tracker, source, parent["arrays"], parent["metadata"]
            )
            checks.require(
                migration["tail_context"] == controls["headers"][0]["metadata"]["context"],
                "Fast changes science relative to old Tail control",
            )
            imported, imported_meta = tracker.export_state()
            checks.require(
                len(imported) == 39
                and set(imported) == set(parent["arrays"])
                and all(original.byte_equal(imported[k], v) for k, v in parent["arrays"].items())
                and imported_meta == migration["imported_metadata"],
                "migration did not preserve all39 original arrays",
            )
            del source, imported, imported_meta
            qualified.check_lifecycle(tracker, START)
            emit("state_import.json", migration)
            checkpoint_binding = dict(
                schema="fast_tail_two_minute_checkpoint_binding_v1",
                original_binding=binding,
                experimental_code_sha256=code,
                native_interval=[START, STOP],
                state_import_sha256=outputs[str(output / "state_import.json")],
                parent_seal_sha256=PARENT_SHA,
                qualification_seal_sha256=FAST_SHA,
                control_seal_sha256=TAIL_SHA,
            )
            record.param("bindings", checkpoint_binding)
            timers["runtime_and_migration_seconds"] = time.perf_counter() - tick
            emit(
                "ready.json",
                dict(
                    environment=environment,
                    previous_observation=context["previous"],
                    original39_bytes_preserved=True,
                    initialize_count=1,
                    restore_count=4,
                    native_interval=[START, STOP],
                    timers_before_tracking=timers.copy(),
                ),
            )
            tick = time.perf_counter()
            evidence.final_guard()
            store.final_guard()
            checks.require(local_code() == code, "source closure changed before tracking")
            timers["pretracking_guard_seconds"] = time.perf_counter() - tick
            print(
                json.dumps(
                    dict(event="fast_tail_two_minute_prepared", native_interval=[START, STOP])
                ),
                flush=True,
            )

            def reader_for(first, stop):
                root = context["inputs"].chunk_roots[(first - FIRST) // CHUNK]
                return original.SourceSequenceReader(root, context["inputs"].input_root)

            def before_frame(native):
                sample = cuda_headroom()
                telemetry.append(dict(frame_idx=native, stage="before_frame", **sample))
                qualified.check_headroom(sample)

            def after_frame(native):
                memory = qualified.qualification.memory_sample()
                memories.append(dict(frame_idx=native, **memory))
                qualified.check_memory(memory)
                sample = cuda_headroom()
                telemetry.append(dict(frame_idx=native, stage="after_frame", **sample))
                qualified.check_headroom(sample)

            def on_frame(row):
                rows.append(row)
                proposals.append(
                    dict(
                        frame_idx=row["frame_idx"],
                        tail_recovery=copy.deepcopy(tracker.last_tail_recovery_diagnostic),
                        bounded_mouth=copy.deepcopy(tracker.last_bounded_mouth_diagnostic),
                        chest=copy.deepcopy(tracker.last_chest_proposal_diagnostic),
                    )
                )
                if row["control"] is not None and not row["control"]["exact"]:
                    emit("first_difference.json", row)
                if len(rows) % 100 == 0 or row["frame_idx"] == STOP - 1:
                    print(
                        json.dumps(
                            dict(
                                event="fast_tail_two_minute_progress",
                                frame=row["frame_idx"],
                                processed=len(rows),
                                total=STOP - START,
                                accumulated_advance_seconds=timers["advance_seconds"],
                            )
                        ),
                        flush=True,
                    )

            def save_endpoint(stop, expected):
                tick = time.perf_counter()
                path = output / "checkpoints" / f"through_{stop:06d}"
                arrays, metadata, row = preserve_checkpoint(
                    tracker, parent, path, checkpoint_binding
                )
                checkpoints.append(row)
                for name in ("arrays.npz", "metadata.json", "manifest.json"):
                    emit((path / name).relative_to(output).as_posix())
                timers["checkpoint_save_readback_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                if expected is not None:
                    control = dict(
                        through_frame=stop,
                        **compare_checkpoint(arrays, metadata, expected, migration),
                    )
                    controls_saved.append(control)
                    emit(f"controls/through_{stop:06d}.json", control)
                    checks.require(control["exact"], "saved all39/metadata Tail control mismatch")
                if stop == CONTROL_STOP:
                    traced = original.phase.COMMON.traced_candidate(arrays)
                    candidate_path = output / "candidate_through_076360.npz"
                    np.savez_compressed(candidate_path, **traced)
                    emit(candidate_path.name)
                    old_path = consumed(controls["tail"], "candidate.npz", evidence)
                    old = original.phase.old.read_npz(old_path)
                    control = compare_candidate(traced, old, CONTROL_STOP)
                    emit("controls/candidate_through_076360.json", control)
                    checks.require(control["exact"], "same76360 candidate21 mismatch")
                timers["checkpoint_control_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                sample = smi_sample()
                telemetry.append(dict(stage="checkpoint", through_frame=stop, **sample))
                qualified.check_headroom(sample)
                timers["telemetry_seconds"] += time.perf_counter() - tick
                emit("checkpoint_progress.json", checkpoints)
                emit("history_comparison.json", rows)
                emit("GPU_telemetry.json", telemetry)
                emit("memory.json", memories)
                print(
                    json.dumps(
                        dict(event="checkpoint_preserved", through_frame=stop, restore_count=4)
                    ),
                    flush=True,
                )

            drive_chunks(
                tracker,
                reader_for=reader_for,
                reference_for=lambda stop: load_control(stop, controls, evidence),
                save_checkpoint=save_endpoint,
                before_frame=before_frame,
                after_frame=after_frame,
                on_frame=on_frame,
                synchronize=torch.cuda.synchronize,
                timers=timers,
            )
            checks.require(
                len(rows) == STOP - START and len(checkpoints) == 12 and len(controls_saved) == 6,
                "incomplete two-minute controls/checkpoints",
            )
            tick = time.perf_counter()
            arrays, _ = tracker.export_state()
            final_trace = original.phase.COMMON.traced_candidate(arrays)
            candidate = tracker.finish()  # The ONLY finish: all twelve saves are already complete.
            np.savez_compressed(output / "candidate.npz", **candidate)
            emit("candidate.npz")
            final_control = compare_candidate(candidate, final_trace, STOP)
            emit("controls/final_backtrace.json", final_control)
            checks.require(final_control["exact"], "final21 differs from complete retained history")
            keep = (candidate["frame_idx"] >= START) & (candidate["frame_idx"] < STOP)
            checks.require(int(keep.sum()) == STOP - START, "final native interval incomplete")
            timers["checkpoint_control_seconds"] += time.perf_counter() - tick
            summary.update(
                status="ok",
                two_minutes_completed=True,
                native_frames=len(rows),
                native_BGR_view_frames=3 * len(rows),
                checkpoints=12,
                first_minute_exact_checkpoints=6,
                first_minute_all6_history19_exact_frames=sum(
                    r["control"] is not None for r in rows
                ),
                first_minute_same76360_candidate21_exact=True,
                final_candidate21_exact_backtrace=True,
                final_horizon_stop=STOP,
                first_minute_snapshot_horizon=CONTROL_STOP,
                final_prefix_not_required_equal_previous_horizon=True,
                finish_calls=1,
                numeric_bytes_exact=True,
                history_unicode_value_exact_ignoring_capacity=True,
                observed_node_counts=candidate["observed"][keep].sum((0, 1)).tolist(),
                review_fish_frames=int(candidate["identity_review"][keep].sum()),
                memory=qualified.qualification.memory_sample(),
                scientific_comparison_performed=False,
                independent_two_minute_evaluation_pending=True,
            )
        except BaseException as error:
            summary.update(status="failed", error=repr(error), completed_frames=len(rows))
            if (
                tracker is not None
                and not getattr(tracker, "_finished", False)
                and tracker._history
            ):
                try:
                    path = output / "partial_checkpoint"
                    arrays, meta = tracker.export_state()
                    write_checkpoint(path, arrays, meta, bindings=checkpoint_binding)
                    for name in ("arrays.npz", "metadata.json", "manifest.json"):
                        emit((path / name).relative_to(output).as_posix())
                    summary["partial_checkpoint_scope"] = (
                        "failure retention only; not a qualified continuation"
                    )
                except BaseException as partial_error:
                    summary["partial_checkpoint_error"] = repr(partial_error)
            raise
        finally:
            if tracker is not None:
                summary.update(
                    logical_initialization_count=tracker.initialize_count,
                    checkpoint_restore_count=tracker.restore_count,
                    previous_frame=tracker.previous_frame,
                    processed_frames=tracker.processed_frames,
                )
            for name, value in (
                ("checkpoint_progress.json", checkpoints),
                ("history_comparison.json", rows),
                ("proposal_diagnostics.json", proposals),
                ("GPU_telemetry.json", telemetry),
                ("memory.json", memories),
                ("mask_validation_receipts.json", store.uses),
            ):
                emit(name, value)
            try:
                tick = time.perf_counter()
                checks.require(code == local_code(), "source closure changed")
                store.final_guard()
                for path, digest in store.evidence_hashes.items():
                    evidence.add(path, digest)
                evidence.final_guard()
                for path, digest in outputs.items():
                    checks.require(checks.sha(path) == digest, "output changed after saving")
                timers["final_guard_seconds"] = time.perf_counter() - tick
                for path, digest in evidence.paths.items():
                    record.input(path, path, sha256=digest)
                record.param("all_source_input_snapshot_output_SHA_unchanged", True)
            except BaseException as guard_error:
                summary.update(
                    status="failed",
                    two_minutes_completed=False,
                    final_guard_error=repr(guard_error),
                )
                raise
            finally:
                summary.update(
                    timers=timers.copy(),
                    wall_seconds_before_summary_and_seal=time.perf_counter() - wall_start,
                    timing_categories_may_include_artifact_save_subtotals=True,
                    timing_is_shared_load_observation_not_exclusive_speed_benchmark=True,
                )
                emit("summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--shared-load-note", type=Path, required=True)
    parser.add_argument("--shared-load-sha", required=True)
    main(parser.parse_args())
