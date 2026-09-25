"""Track a sealed prefix of the long recording with one persistent initial state.

Chunk boundaries change input storage only. Every retained beam, parent history
and previous image survives both a boundary and a successful process checkpoint.
Prefix backtraces are provisional until the complete registered interval ends.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch

LAB = Path(__file__).resolve().parent
REPO = LAB.parent.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import run_experiment as base  # noqa: E402
from continuous_tracker import AcceleratedContinuousPersistentNodeTracker  # noqa: E402
from long_tracking_inputs import load_long_inputs, verify_prefix_records  # noqa: E402
from motion import FlowConfig  # noqa: E402
from node_recovery import NodeRecoveryConfig  # noqa: E402
from observations import CachedNodes  # noqa: E402
from persistent_node_tracker import PersistentNodeConfig  # noqa: E402
from resumable_observations import iter_from  # noqa: E402
from run_persistent_node_recovery import validate_arm_arrays  # noqa: E402
from source_sequence import SourceSequenceReader  # noqa: E402
from tracker import TrackerConfig  # noqa: E402
from tracker_checkpoint import read_checkpoint, write_checkpoint  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

STAGE = "three_body_long_continuation"
PROTOCOL_PATH = LAB / "long_tracking_protocol.json"


def _sources():
    paths = {Path(__file__).resolve(), PROTOCOL_PATH}
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src/medaka3d")
            ):
                paths.add(path)
    return sorted(paths)


def _unchanged(hashes):
    for path, digest in hashes.items():
        if base.sha256(path) != digest:
            raise RuntimeError("long continuation evidence changed: " + str(path))


def _same_arrays(first, second):
    if set(first) != set(second):
        raise ValueError("checkpoint array names differ")
    for key, array in first.items():
        expected = second[key]
        if (array.dtype != expected.dtype or array.shape != expected.shape
                or array.tobytes() != expected.tobytes()):
            raise ValueError("checkpoint round trip differs: " + key)


def _parent_checkpoint(root, bindings, chunk_records):
    root = Path(root).resolve(strict=True)
    if not root.is_relative_to(REPO):
        raise ValueError("resume phase must be in the workspace")
    path = root / "continuation.provenance.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("stage") != STAGE or record.get("status") != "ok":
        raise ValueError("resume phase is not successfully sealed")
    if record.get("params", {}).get("bindings") != bindings:
        raise ValueError("resume phase uses different fixed inputs, code or runtime")
    hashes = {str(path): base.sha256(path)}
    for name, filename in (("checkpoint_manifest", "checkpoint/manifest.json"),
                           ("summary", "summary.json"),
                           ("consumed_chunks", "consumed_chunks.json")):
        actual = (root / filename).resolve(strict=True)
        item = record.get("outputs", {}).get(name)
        if (item is None or Path(item["path"]).resolve(strict=True) != actual
                or base.sha256(actual) != item["sha256"]):
            raise ValueError("resume output is not bound to its successful seal: " + name)
        hashes[str(actual)] = item["sha256"]
    previous_chunks = json.loads((root / "consumed_chunks.json").read_text(encoding="utf-8"))
    verify_prefix_records(previous_chunks, chunk_records)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    arrays, metadata, _ = read_checkpoint(root / "checkpoint", expected_bindings=bindings)
    for filename in ("arrays.npz", "metadata.json"):
        payload = root / "checkpoint" / filename
        hashes[str(payload)] = base.sha256(payload)
    if (summary.get("schema") != "long_continuation_phase_v1"
            or summary.get("technical_completion") is not True
            or summary.get("logical_initialization_count") != 1
            or summary.get("external_pose_reset_count") != 0
            or summary.get("processed_total") != metadata.get("processed_frames")
            or summary.get("checkpoint_restore_count") != metadata.get("restore_count")):
        raise ValueError("resume summary and checkpoint state disagree")
    return (arrays, metadata), hashes, summary, previous_chunks


def _advance_chunks(tracker, inputs, start_frame, progress_path):
    """Consume complete declared readers without a begin/reset between them."""
    expected = int(start_frame)
    phase_start = tracker.processed_frames
    input_seconds = tracking_seconds = 0.0
    with progress_path.open("x", encoding="utf-8", buffering=1) as progress:
        for root in inputs.chunk_roots:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            first, stop = manifest["frames"][0], manifest["frames"][-1] + 1
            if stop <= expected:
                continue
            if expected < first:
                raise ValueError("a source chunk creates a native frame gap")
            prepared = time.perf_counter()
            reader = SourceSequenceReader(root, inputs.input_root)
            input_seconds += time.perf_counter() - prepared
            iterator = iter_from(reader, expected)
            started = time.perf_counter()
            try:
                for frame in iterator:
                    if (expected >= stop or frame.frame_idx != expected or tracker.processed_frames
                            != expected - inputs.full_frame_range[0]):
                        raise ValueError("long reader yielded an incorrect native frame")
                    tracker.advance(frame)
                    expected += 1
                    if tracker.processed_frames % 25 == 0 or expected == stop:
                        event = {
                            "processed": tracker.processed_frames,
                            "phase_processed": tracker.processed_frames - phase_start,
                            "source_frame": frame.frame_idx,
                            "tracking_seconds": tracking_seconds + time.perf_counter() - started,
                        }
                        progress.write(json.dumps(event) + "\n")
                        print(json.dumps(event), flush=True)
                torch.cuda.synchronize()
            finally:
                iterator.close()
            tracking_seconds += time.perf_counter() - started
            if expected != stop:
                raise ValueError("source chunk ended before its declared native boundary")
    if expected != int(inputs.frame_idx[-1]) + 1:
        raise ValueError("tracking did not reach the requested registered prefix boundary")
    return input_seconds, tracking_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--input-stage-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=REPO / "work/generative_tracking_inputs")
    parser.add_argument("--through-frame", type=int, required=True,
                        help="exclusive native end at a registered complete input chunk")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    wall_started = time.perf_counter()
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists() or output.is_symlink():
        raise ValueError("output must be a new workspace directory")
    sources = _sources()
    code_hashes = {str(path): base.sha256(path) for path in sources}
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("schema") != "three_body_long_tracking_protocol_v1":
        raise ValueError("unknown registered long tracking protocol")
    for relative, digest in protocol["frozen_tracker_sources_sha256"].items():
        if base.sha256(REPO / relative) != digest:
            raise ValueError("a registered tracking dependency changed: " + relative)
    inputs = load_long_inputs(args.seed_root, args.input_stage_root, args.input_root,
                              through_frame=args.through_frame)
    input_protocol_path = LAB / "long_input_protocol.json"
    if (list(inputs.full_frame_range) != protocol["source_frame_range"]
            or base.sha256(LAB / "long_input_protocol.json") != protocol["input_protocol_sha256"]
            or inputs.protocol != json.loads(input_protocol_path.read_text(encoding="utf-8"))
            or base.sha256(args.seed_root / "registration.json")
            != protocol["seed_registration_sha256"]):
        raise ValueError("long inputs differ from the fixed tracking protocol")
    if set(_sources()) != set(sources):
        raise ValueError("late local source imports must be registered before input loading")
    if not torch.cuda.is_available():
        raise RuntimeError("the registered long continuation requires CUDA")
    torch.set_num_threads(protocol["torch_cpu_threads"])
    versions = {
        "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
    }
    if versions != protocol["package_versions"]:
        raise ValueError("long tracking runtime differs from the registered version")
    config = TrackerConfig(**protocol["tracker_config"])
    recovery = NodeRecoveryConfig(**protocol["node_recovery_config"])
    persistence = PersistentNodeConfig(**protocol["persistence_config"])
    if dataclasses.asdict(FlowConfig()) != protocol["flow_config"]:
        raise ValueError("default image flow configuration changed")
    bindings = {
        "schema": "long_continuation_fixed_binding_v1",
        "inputs": inputs.fixed_bindings, "code_sha256": code_hashes,
        "tracking_protocol_sha256": base.sha256(PROTOCOL_PATH),
        "package_versions": versions, "torch_cpu_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    parent_state, parent_summary, parent_chunks = None, None, None
    parent_hashes = {}
    if args.resume is not None:
        parent_state, parent_hashes, parent_summary, parent_chunks = _parent_checkpoint(
            args.resume, bindings, inputs.chunk_records)
    hashes = {**inputs.verified_hashes, **code_hashes, **parent_hashes}
    _unchanged(hashes)
    output.mkdir(parents=True, exist_ok=False)
    with provenance(STAGE, output, name="continuation") as record:
        record.param("bindings", bindings)
        record.param("through_frame", args.through_frame)
        record.param("resume_phase", None if args.resume is None else str(args.resume.resolve()))
        for path, digest in hashes.items():
            record.input(str(path), path, sha256=digest)
        output_hashes = {}
        for source in sources:
            target = output / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            output_hashes[str(target)] = code_hashes[str(source)]
        chunk_path = output / "consumed_chunks.json"
        chunk_path.write_text(json.dumps(inputs.chunk_records, indent=2, allow_nan=False),
                              encoding="utf-8")
        record.output("consumed_chunks", chunk_path, sha256=base.sha256(chunk_path))
        output_hashes[str(chunk_path)] = base.sha256(chunk_path)
        tracker = AcceleratedContinuousPersistentNodeTracker(
            inputs.rig, inputs.initial_pose_xyz.copy(), config,
            recovery_config=recovery, persistence_config=persistence)
        if parent_state is None:
            tracker.begin()
        else:
            tracker.restore_state(*parent_state)
        previous_count = tracker.processed_frames
        first, full_stop = inputs.full_frame_range
        through = args.through_frame
        if previous_count >= len(inputs.frame_idx):
            raise ValueError("resume must advance beyond the saved prefix")
        expected_previous = None if previous_count == 0 else first + previous_count - 1
        if tracker.previous_frame != expected_previous:
            raise ValueError("checkpoint last frame differs from the registered start")
        if parent_summary is not None and (
                parent_summary["processed_interval"] != [first, first + previous_count]
                or parent_summary["consumed_chunk_count"] != len(parent_chunks)
                or previous_count != len(parent_chunks) * inputs.protocol["chunk_frames"]):
            raise ValueError("checkpoint state is not at the end of its proven input prefix")
        preparation_seconds = time.perf_counter() - wall_started
        progress_path = output / "progress.jsonl"
        reader_seconds, tracking_seconds = _advance_chunks(
            tracker, inputs, first + previous_count, progress_path)
        if (tracker.processed_frames != len(inputs.frame_idx)
                or tracker.initialize_count != 1):
            raise ValueError("long phase did not consume the exact requested prefix")
        saved_at = time.perf_counter()
        arrays, metadata = tracker.export_state()
        write_checkpoint(output / "checkpoint", arrays, metadata, bindings=bindings)
        roundtrip, roundtrip_meta, _ = read_checkpoint(
            output / "checkpoint", expected_bindings=bindings)
        _same_arrays(arrays, roundtrip)
        if roundtrip_meta != metadata:
            raise ValueError("checkpoint metadata round trip differs")
        checkpoint_seconds = time.perf_counter() - saved_at
        checkpoint_manifest = output / "checkpoint/manifest.json"
        record.output("checkpoint_manifest", checkpoint_manifest,
                      sha256=base.sha256(checkpoint_manifest))
        for filename in ("manifest.json", "metadata.json", "arrays.npz"):
            path = output / "checkpoint" / filename
            output_hashes[str(path)] = base.sha256(path)
        # Snapshot all beams before this provisional prefix backtrace terminates
        # the in-memory tracker. A future process restores the full saved beam.
        result = tracker.finish()
        nodes = CachedNodes(inputs.input_root, inputs.protocol["experiment"], inputs.frame_idx)
        base.validate_result(result, inputs.frame_idx, nodes)
        validate_arm_arrays(result, inputs.frame_idx)
        candidate_path = output / "candidate.npz"
        np.savez_compressed(candidate_path, **result)
        record.output("candidate", candidate_path, sha256=base.sha256(candidate_path))
        output_hashes[str(candidate_path)] = base.sha256(candidate_path)
        complete = through == full_stop
        summary = {
            "schema": "long_continuation_phase_v1", "technical_completion": True,
            "full_sequence_complete": complete, "prefix_backtrace_provisional": not complete,
            "registered_interval": [first, full_stop], "processed_interval": [first, through],
            "processed_total": tracker.processed_frames,
            "phase_frames": [first + previous_count, through],
            "phase_processed": tracker.processed_frames - previous_count,
            "consumed_chunk_count": len(inputs.chunk_records),
            "logical_initialization_count": tracker.initialize_count,
            "checkpoint_restore_count": tracker.restore_count, "external_pose_reset_count": 0,
            "temporal_subsampling": False, "human_annotation_input": False,
            "post_initialization_baseline_coordinates_input": False,
            "preparation_seconds": preparation_seconds, "reader_setup_seconds": reader_seconds,
            "tracking_seconds": tracking_seconds, "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_payload_bytes": (output / "checkpoint/arrays.npz").stat().st_size,
            "retained_history_rows": int(arrays["history_offsets"][-1]),
            "active_beam_count": len(arrays["state_pose"]),
            "node_recovery_diagnostics": tracker.node_recovery_diagnostics,
            "direct_pectoral_count": int(result["observed"][:, :, 1].sum()),
            "accepted_pectoral_count": int(np.isfinite(result["accepted_xyz"]).all(axis=-1).sum()),
            "identity_review_count": int(result["identity_review"].sum()),
            "per_slot_direct_pectoral": result["observed"][:, :, 1].sum(axis=0).tolist(),
            "per_slot_accepted_pectoral": (
                np.isfinite(result["accepted_xyz"]).all(axis=-1).sum(axis=0).tolist()),
            "limitations": protocol["limitations"],
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", summary_path, sha256=base.sha256(summary_path))
        record.output("progress", progress_path, sha256=base.sha256(progress_path))
        output_hashes[str(summary_path)] = base.sha256(summary_path)
        output_hashes[str(progress_path)] = base.sha256(progress_path)
        _unchanged({**hashes, **output_hashes})
        for path, digest in output_hashes.items():
            record.output(path, path, sha256=digest)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(json.dumps({"output": str(output), "full_sequence_complete": complete,
                      "processed": tracker.processed_frames}), flush=True)
    return summary


if __name__ == "__main__":
    main()
