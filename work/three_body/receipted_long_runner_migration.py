"""Explicit I/O-only binding transition for the frozen long tracker.

No process launch, raw decode, fit, advance, GPU initialization or global patch.
The old runner remains imported for its immutable frame loop and validators.
CPU validation objects retain the registered context, but never execute a fit.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import run_long_continuation as original  # noqa: E402
from cached_long_tracking_inputs import load_cached_long_inputs  # noqa: E402
from continuous_tracker import RECORD_FIELDS  # noqa: E402
from mask_validation_receipts import ReceiptStore  # noqa: E402
from tracker_checkpoint import _canonical, _decode, read_checkpoint, write_checkpoint  # noqa: E402

FIRST, FULL_STOP, CHUNK = 49800, 348600, 1660
STAGE = "three_body_receipted_long_continuation_v1"
SUMMARY_SCHEMA = "receipted_long_continuation_phase_v1"
REGISTRATION_STAGE = "register_receipted_long_continuation_v1"
BINDING_SCHEMA = "receipted_long_continuation_fixed_binding_v1"
LEGACY_ANCHOR = REPO / "work/three_body/long_continuation_through66400_v1"
LEGACY_ANCHOR_SHA = "9915c44204c7acaee0abcada3a3fa2dfa82b7ac017c0b7d4fc9b50b628d48a6e"
ADDED_IO = (
    "work/three_body/cached_long_tracking_inputs.py",
    "work/three_body/mask_validation_receipts.py",
    "work/three_body/receipted_long_runner_migration.py",
    "work/three_body/run_receipted_long_continuation.py",
)
PINNED_IO = {
    ADDED_IO[0]: "c25ab121b2ffdf80309ebe18e03821af559e5e3cbb900bd17edb8a4f4e0a62c1",
    ADDED_IO[1]: "82a284d9874608003529f9cc5ccb158f2f69adcb6da5faeda072daab11938df0",
}
PARENT_ROLES = {
    "candidate": "candidate.npz",
    "summary": "summary.json",
    "progress": "progress.jsonl",
    "consumed_chunks": "consumed_chunks.json",
    "checkpoint_manifest": "checkpoint/manifest.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(_canonical(actual) == _canonical(expected), message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def read(path):
    return _decode(Path(path).read_bytes())


def local(path, *, exists=True):
    path = Path(path)
    require(not path.is_symlink(), "symbolic link is not an evidence file")
    path = path.resolve(strict=exists)
    require(path.is_relative_to(REPO), "evidence outside workspace")
    return path


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected=None):
        path = local(path)
        if str(path) in self.paths:
            require(
                expected is None or expected == self.paths[str(path)],
                f"conflicting evidence SHA: {path}",
            )
            return self.paths[str(path)]
        actual = sha(path)
        require(expected is None or expected == actual, f"evidence SHA differs: {path}")
        require(
            str(path) not in self.paths or self.paths[str(path)] == actual,
            f"evidence changed during operation: {path}",
        )
        self.paths[str(path)] = actual
        return actual

    def final_guard(self):
        for path, expected in self.paths.items():
            require(sha(path) == expected, f"evidence changed after use: {path}")


def verified_outputs(root, seal, evidence):
    require(
        seal.get("status") == "ok"
        and seal["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "successful guarded producer required",
    )
    for item in seal["outputs"].values():
        path = local(item["path"])
        require(
            path.is_relative_to(root) and path.stat().st_size == item["size"],
            "producer output path/size differs",
        )
        evidence.add(path, item["sha256"])


def legacy_binding(evidence):
    """A pinned historical seal establishes expected old code, never the new parent itself."""
    path = LEGACY_ANCHOR / "continuation.provenance.json"
    evidence.add(path, LEGACY_ANCHOR_SHA)
    seal = read(path)
    require(
        seal["stage"] == original.STAGE
        and seal["status"] == "ok"
        and seal["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
        "historical binding anchor is not successful",
    )
    binding = seal["params"]["bindings"]
    require(binding["schema"] == "long_continuation_fixed_binding_v1", "legacy schema differs")
    for source, expected in binding["code_sha256"].items():
        source = local(source)
        evidence.add(source, expected)
        snapshot = LEGACY_ANCHOR / "code_snapshot" / source.relative_to(REPO)
        item = seal["outputs"][str(snapshot)]
        require(
            local(item["path"]) == snapshot
            and item["sha256"] == expected
            and snapshot.stat().st_size == item["size"],
            "legacy code snapshot differs",
        )
        evidence.add(snapshot, expected)
    return copy.deepcopy(binding)


def execution_binding(legacy, code):
    """Pure finite allowlist: every old source retained byte-for-byte; exactly four additions."""
    require(legacy.get("schema") == "long_continuation_fixed_binding_v1", "wrong legacy binding")
    old_code = legacy["code_sha256"]
    added = {str((REPO / relative).resolve()) for relative in ADDED_IO}
    require(
        not added.intersection(old_code) and set(code) == set(old_code) | added,
        "execution source additions/removals exceed the four I/O files",
    )
    for path, expected in old_code.items():
        require(code[path] == expected, "scientific or legacy source changed: " + path)
    for relative, expected in PINNED_IO.items():
        require(code[str((REPO / relative).resolve())] == expected, "frozen receipt code changed")
    for path, value in code.items():
        require(
            type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "invalid source SHA: " + path,
        )
    result = copy.deepcopy(legacy)
    result.update(
        schema=BINDING_SCHEMA,
        code_sha256=copy.deepcopy(code),
        verifier_policy=dict(
            schema="long_input_receipt_execution_policy_v1",
            legacy_binding_sha256=digest(legacy),
            added_io_sources={key: code[key] for key in sorted(added)},
            loader="cached_long_tracking_inputs.load_cached_long_inputs",
            receipt_store="mask_validation_receipts.ReceiptStore",
            retained_tracker="AcceleratedContinuousPersistentNodeTracker",
            retained_frame_loop="run_long_continuation._advance_chunks",
            scientific_context_unchanged=True,
            serial_flow_unchanged=True,
            full_current_file_sha_retained=True,
            receipt_evidence_outside_tracker_state=True,
        ),
    )
    return result


def execution_sources(legacy, evidence):
    expected = set(legacy["code_sha256"]) | {str((REPO / p).resolve()) for p in ADDED_IO}
    # Restrict this executable to a fresh interpreter with precisely this source closure.
    loaded = set()
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB)
                or path.is_relative_to(REPO / "src/medaka3d")
                or path.is_relative_to(REPO / "work/three_body")
            ):
                loaded.add(str(path))
    require(
        loaded == {p for p in expected if p.endswith(".py")},
        "late, missing or foreign workspace imports: " + str(sorted(loaded ^ expected)),
    )
    code = {path: evidence.add(path) for path in sorted(expected)}
    execution_binding(legacy, code)
    return code


def transition(parent_binding, current_binding, legacy):
    equal(
        current_binding,
        execution_binding(legacy, current_binding["code_sha256"]),
        "new binding is not the registered finite I/O change",
    )
    if _canonical(parent_binding) == _canonical(legacy):
        kind = "legacy_to_receipted"
    else:
        equal(parent_binding, current_binding, "later receipted binding must match exactly")
        kind = "receipted_to_receipted"
    return dict(
        kind=kind,
        old_binding_sha256=digest(parent_binding),
        new_binding_sha256=digest(current_binding),
        context_changed=False,
        scientific_code_changed=False,
        logical_restore_increment=1,
    )


def native_phase(start, stop):
    require(
        type(start) is int
        and type(stop) is int
        and start >= 66400
        and (start - FIRST) % CHUNK == 0
        and start < FULL_STOP
        and stop == min(start + 6 * CHUNK, FULL_STOP),
        "six chunks or final remainder required",
    )


def load_inputs(binding, through, receipt_root, evidence, *, store_ready=None):
    values = binding["inputs"]
    store = ReceiptStore(local(receipt_root, exists=False))
    if store_ready is not None:
        store_ready(store)
    inputs = load_cached_long_inputs(
        Path(values["seed"]["registration_path"]).parent,
        values["input_stage_root"],
        values["seed"]["input_root"],
        through_frame=through,
        receipt_store=store,
    )
    equal(inputs.fixed_bindings, values, "receipt loader changed scientific inputs")
    for path, expected in inputs.verified_hashes.items():
        evidence.add(path, expected)
    bind_receipts(store, evidence)
    return inputs, store


def bind_receipts(store, evidence):
    store.final_guard()
    for path, expected in store.evidence_hashes.items():
        evidence.add(path, expected)


def bind_parent_audit(root, expected_sha, parent, parent_seal, evidence):
    """Require exact candidate/parent membership; never promote an unaudited parent."""
    path = local(root)
    evidence.add(path, expected_sha)
    audit = read(path)
    allowed = {
        "long_receipted_phase_operation_v1": ("audit", "audit.json"),
        "long_supervised_phase_operation": ("audit", "audit.json"),
        "long_supervised_bootstrap_audit": ("summary", "summary.json"),
        "long_receipted_runner_phase_audit_v1": ("summary", "summary.json"),
    }
    require(audit.get("stage") in allowed, "unsupported independent parent audit stage")
    verified_outputs(path.parent, audit, evidence)
    if audit["stage"] in {"long_receipted_phase_operation_v1", "long_supervised_phase_operation"}:
        child_path = path.parent / "child_execution.json"
        require(
            local(audit["outputs"]["child_execution"]["path"]) == child_path,
            "independent operation lacks child-exit evidence",
        )
        child = read(child_path)
        require(
            child["returncode"] == 0 and child["child_alive_after_return"] is False,
            "independent operation did not observe successful child exit",
        )
    role, filename = allowed[audit["stage"]]
    require(
        local(audit["outputs"][role]["path"]) == path.parent / filename,
        "parent audit summary role differs",
    )
    summary = read(path.parent / filename)
    candidate = parent / "candidate.npz"
    for input_path, expected in (
        (parent / "continuation.provenance.json", sha(parent / "continuation.provenance.json")),
        (candidate, parent_seal["outputs"]["candidate"]["sha256"]),
    ):
        item = audit["inputs"].get(str(input_path))
        require(
            item is not None and local(item["path"]) == input_path and item["sha256"] == expected,
            "parent audit input binding differs",
        )
    require(
        summary.get("status") == "ok"
        and local(summary["run"]) == parent
        and summary["candidate_sha256"] == parent_seal["outputs"]["candidate"]["sha256"]
        and summary["all_39_checkpoint_arrays_verified"] is True
        and len(summary["all_21_candidate_arrays_exact_backtrace"]) == 21
        and summary["logical_initialization_count"] == 1
        and summary["external_pose_reset_count"] == 0,
        "parent independent audit is incomplete or belongs to another candidate",
    )
    producer_summary = read(parent / "summary.json")
    require(
        summary["checkpoint_restore_count"] == producer_summary["checkpoint_restore_count"]
        and summary["frame_range"] == producer_summary["processed_interval"]
        and summary["history"]["active_beam_count"] == 6,
        "parent audit native lifecycle differs",
    )
    require(
        datetime.fromisoformat(parent_seal["finished_at"])
        <= datetime.fromisoformat(audit["finished_at"]),
        "parent audit predates producer",
    )
    return dict(
        path=str(path),
        sha256=expected_sha,
        stage=audit["stage"],
        summary_sha256=evidence.paths[str(path.parent / filename)],
    )


def load_parent(root, expected_seal, binding, inputs, evidence):
    root = local(root)
    seal_path = root / "continuation.provenance.json"
    evidence.add(seal_path, expected_seal)
    seal = read(seal_path)
    expected_stage = original.STAGE if binding["schema"] != BINDING_SCHEMA else STAGE
    require(seal.get("stage") == expected_stage, "parent stage/binding version differs")
    equal(seal["params"]["bindings"], binding, "parent fixed binding differs")
    stop = seal["params"]["through_frame"]
    require(
        type(stop) is int and FIRST < stop < FULL_STOP and (stop - FIRST) % CHUNK == 0,
        "parent is not an unfinished registered native prefix",
    )
    later = inputs.chunk_roots[(stop - FIRST) // CHUNK :]
    expected_inputs = {
        p: d
        for p, d in inputs.verified_hashes.items()
        if not any(Path(p).is_relative_to(r) for r in later)
    }
    expected_inputs.update(binding["code_sha256"])
    for path, expected in expected_inputs.items():
        item = seal["inputs"].get(path)
        require(
            item is not None and str(local(item["path"])) == path and item["sha256"] == expected,
            "parent scientific input binding differs",
        )
        evidence.add(path, expected)
    # Do not recursively re-open every ancestor payload from transitive operational
    # input lists. The pinned independent parent audit owns that lineage check.
    for role, item in seal["inputs"].items():
        require(str(Path(item["path"]).resolve()) == role, "parent operational input path differs")
    verified_outputs(root, seal, evidence)
    for role, filename in PARENT_ROLES.items():
        require(
            local(seal["outputs"][role]["path"]) == root / filename,
            "parent named output differs: " + role,
        )
    for filename in ("checkpoint/arrays.npz", "checkpoint/metadata.json"):
        require(str(root / filename) in seal["outputs"], "parent checkpoint payload is not sealed")
    for source, expected in binding["code_sha256"].items():
        snapshot = root / "code_snapshot" / Path(source).relative_to(REPO)
        require(str(snapshot) in seal["outputs"], "parent source snapshot missing")
        evidence.add(snapshot, expected)
    consumed = read(root / "consumed_chunks.json")
    original.verify_prefix_records(consumed, inputs.chunk_records)
    arrays, metadata, manifest = read_checkpoint(root / "checkpoint", expected_bindings=binding)
    summary = read(root / "summary.json")
    stop = metadata["previous_frame"] + 1
    equal(
        {
            key: summary.get(key)
            for key in (
                "technical_completion",
                "logical_initialization_count",
                "checkpoint_restore_count",
                "external_pose_reset_count",
                "processed_total",
                "processed_interval",
                "consumed_chunk_count",
            )
        },
        dict(
            technical_completion=True,
            logical_initialization_count=1,
            checkpoint_restore_count=metadata["restore_count"],
            external_pose_reset_count=0,
            processed_total=metadata["processed_frames"],
            processed_interval=[FIRST, stop],
            consumed_chunk_count=len(consumed),
        ),
        "parent summary field types or values differ",
    )
    expected_schema = (
        "long_continuation_phase_v1" if expected_stage == original.STAGE else SUMMARY_SCHEMA
    )
    require(
        summary["schema"] == expected_schema
        and summary["technical_completion"] is True
        and summary["logical_initialization_count"] == metadata["initialize_count"] == 1
        and summary["checkpoint_restore_count"] == metadata["restore_count"]
        and summary["external_pose_reset_count"] == 0
        and summary["processed_total"] == metadata["processed_frames"] == stop - FIRST
        and summary["processed_interval"] == [FIRST, stop]
        and seal["params"]["through_frame"] == stop
        and len(consumed) * CHUNK == stop - FIRST
        and summary["consumed_chunk_count"] == len(consumed)
        and len(arrays) == 39
        and arrays["state_pose"].shape == (6, 3, 3, 3)
        and summary["node_recovery_diagnostics"] == metadata["node_recovery_diagnostics"],
        "parent lifecycle/native/full beam contract differs",
    )
    with np.load(root / "candidate.npz", allow_pickle=False) as data:
        require(len(data.files) == 21 == len(set(data.files)), "parent candidate schema differs")
        candidate = {key: data[key] for key in data.files}
    compare_backtrace(arrays, candidate)
    return dict(
        root=root,
        seal=seal,
        arrays=arrays,
        metadata=metadata,
        summary=summary,
        consumed=consumed,
        manifest=manifest,
    )


def validation_tracker(inputs, protocol):
    """CPU-only decoder object, not an execution fallback or a modified checkpoint context."""
    config = original.TrackerConfig(**protocol["tracker_config"])
    tracker = original.AcceleratedContinuousPersistentNodeTracker(
        inputs.rig,
        inputs.initial_pose_xyz.copy(),
        dataclasses.replace(config, device="cpu"),
        recovery_config=original.NodeRecoveryConfig(**protocol["node_recovery_config"]),
        persistence_config=original.PersistentNodeConfig(**protocol["persistence_config"]),
    )
    # _decode_state uses config/context, never device tensors or CUDA. No metadata is rewritten.
    tracker.config = config
    tracker._initial_context = tracker._context()
    return tracker


def restore_parent(tracker, arrays, metadata):
    require(
        type(tracker) is original.AcceleratedContinuousPersistentNodeTracker,
        "only the unchanged original accelerated persistent class may restore",
    )
    require(
        tracker.initialize_count == tracker.restore_count == tracker.processed_frames == 0,
        "fresh never-started tracker required",
    )
    equal(tracker._context(), metadata["context"], "original scientific context differs")
    signatures = array_signatures(arrays)
    old_metadata = copy.deepcopy(metadata)
    tracker.restore_state(arrays, metadata)
    restored, restored_metadata = tracker.export_state()
    original._same_arrays(restored, arrays)
    expected = copy.deepcopy(metadata)
    expected["restore_count"] += 1
    equal(restored_metadata, expected, "restore changed metadata beyond one restore counter")
    equal(metadata, old_metadata, "caller metadata mutated")
    equal(array_signatures(arrays), signatures, "caller checkpoint arrays mutated")
    require(
        not any(np.shares_memory(restored[k], arrays[k]) for k in arrays),
        "restored export aliases supplied checkpoint",
    )
    return dict(
        all_checkpoint_arrays_exact=list(sorted(arrays)),
        array_signatures=signatures,
        context_unchanged=True,
        initialize_count=1,
        restore_count=tracker.restore_count,
        previous_frame=tracker.previous_frame,
        external_pose_reset_count=0,
    )


def array_signatures(arrays):
    return {
        key: dict(
            dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(a.tobytes()).hexdigest()
        )
        for key, a in arrays.items()
    }


def compare_prefix(parent, current):
    offsets = parent["history_offsets"]
    count, rows = len(offsets) - 1, int(offsets[-1])
    original._same_arrays(
        {"offsets": current["history_offsets"][: count + 1]}, {"offsets": offsets}
    )
    original._same_arrays(
        {
            k: current[k][:rows]
            for k in parent
            if k.startswith("history_") and k != "history_offsets"
        },
        {k: parent[k] for k in parent if k.startswith("history_") and k != "history_offsets"},
    )
    return dict(parent_native_frames=count, parent_rows=rows, all_history_prefix_exact=True)


def traced_candidate(arrays):
    offsets, selected, parent = arrays["history_offsets"], [], 0
    for t in range(len(offsets) - 2, -1, -1):
        require(0 <= parent < offsets[t + 1] - offsets[t], "invalid backtrace parent")
        row = int(offsets[t]) + parent
        selected.append(row)
        parent = int(arrays["history_parent"][row])
    selected.reverse()
    result = {
        key: np.asarray([arrays["history_" + key][i] for i in selected])
        for key in RECORD_FIELDS
        if key != "parent"
    }
    result["xyz"] = np.where(result["observed"][:, :, 1, None], result["pose_xyz"][:, :, 1], np.nan)
    result["accepted_xyz"] = np.where(result["identity_review"][:, :, None], np.nan, result["xyz"])
    result["body_xyz"] = np.where(
        result["body_supported"][:, :, None], result["pose_xyz"][:, :, 1], np.nan
    )
    return result


def compare_backtrace(arrays, candidate):
    original._same_arrays(candidate, traced_candidate(arrays))
    return list(sorted(candidate))


def checkpoint_roundtrip(output, arrays, metadata, binding):
    """No logical restore: exact codec transport, usable for a later authorized CPU probe."""
    signatures, meta = array_signatures(arrays), copy.deepcopy(metadata)
    write_checkpoint(output, arrays, metadata, bindings=binding)
    after, after_meta, manifest = read_checkpoint(output, expected_bindings=binding)
    original._same_arrays(after, arrays)
    equal(after_meta, metadata, "codec changed metadata")
    equal(metadata, meta, "codec mutated caller metadata")
    equal(array_signatures(arrays), signatures, "codec mutated caller arrays")
    return manifest
