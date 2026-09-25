"""Resume verified persistent development tracking without a new geometric seed.

This driver tests operational continuity on completed registered windows. The
thirty-minute start and full input staging require a separate long-run binding.
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

import identity_inputs as verified  # noqa: E402
from continuous_tracker import AcceleratedContinuousPersistentNodeTracker  # noqa: E402
from motion import FlowConfig  # noqa: E402
from node_recovery import NodeRecoveryConfig  # noqa: E402
from observations import SequenceReader  # noqa: E402
from persistent_node_tracker import PersistentNodeConfig  # noqa: E402
from resumable_observations import iter_from  # noqa: E402
from tracker import TrackerConfig  # noqa: E402
from tracker_checkpoint import read_checkpoint, write_checkpoint  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

base = verified.base
STAGE = "three_body_persistent_continuation"


def _sources():
    paths = {Path(__file__).resolve(), LAB / "CONTINUATION_PROTOCOL.md"}
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
            raise RuntimeError("continuation evidence changed: " + path)


def _same_arrays(first, second):
    if set(first) != set(second):
        raise ValueError("continuation result array names differ")
    for key, array in first.items():
        expected = second[key]
        if (array.dtype != expected.dtype or array.shape != expected.shape
                or array.tobytes() != expected.tobytes()):
            raise ValueError("continuation result differs in dtype/shape/bytes: " + key)


def _parent_checkpoint(root, bindings):
    root = Path(root).resolve(strict=True)
    if not root.is_relative_to(REPO):
        raise ValueError("resume phase must be in the workspace")
    path = root / "continuation.provenance.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("stage") != STAGE or record.get("status") != "ok":
        raise ValueError("resume phase is not successfully sealed")
    if record.get("params", {}).get("bindings") != bindings:
        raise ValueError("resume phase uses different inputs, code or runtime")
    hashes = {str(path): base.sha256(path)}
    for name, filename in (("checkpoint_manifest", "checkpoint/manifest.json"),
                           ("summary", "summary.json")):
        actual = (root / filename).resolve(strict=True)
        item = record.get("outputs", {}).get(name)
        if (item is None or Path(item["path"]).resolve(strict=True) != actual
                or base.sha256(actual) != item["sha256"]):
            raise ValueError("resume phase checkpoint/summary is not bound to its seal")
        hashes[str(actual)] = item["sha256"]
    arrays, metadata, manifest = read_checkpoint(root / "checkpoint", expected_bindings=bindings)
    for filename in ("arrays.npz", "metadata.json"):
        payload = root / "checkpoint" / filename
        hashes[str(payload)] = base.sha256(payload)
    return arrays, metadata, manifest, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int,
                        help="total completed frames including a restored prefix; default full")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists() or output.is_symlink():
        raise ValueError("output must be a new workspace directory")
    # Hash the loaded transitive source before even verifying old producer bytes.
    sources = _sources()
    code_hashes = {str(path): base.sha256(path) for path in sources}
    run = verified._verify_run(args.reference_run, appearance=False)
    for relative, digest in run.code.items():
        if base.sha256(REPO / relative) != digest:
            raise ValueError("a frozen original tracking dependency changed: " + relative)
    if not torch.cuda.is_available():
        raise RuntimeError("the registered continuation experiment requires CUDA")
    torch.set_num_threads(run.params["torch_cpu_threads"])
    versions = {
        "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
    }
    if versions != run.params["package_versions"]:
        raise ValueError("continuation requires the original measured software/device versions")
    config = TrackerConfig(**run.params["tracker_config"])
    recovery = NodeRecoveryConfig(**run.params["node_recovery_config"])
    persistence = PersistentNodeConfig(**run.params["persistence_config"])
    if dataclasses.asdict(FlowConfig()) != run.params["flow_config"]:
        raise ValueError("default image flow configuration changed")
    count = len(run.frames)
    stop_after = count if args.stop_after is None else args.stop_after
    if not 1 <= stop_after <= count:
        raise ValueError("stop-after must be inside the full registered frame count")
    bindings = {
        "schema": "registered_continuation_binding_v1",
        "reference_run": str(run.root),
        "source_range": [int(run.frames[0]), int(run.frames[-1] + 1)],
        "input_sha256": run.hashes, "code_sha256": code_hashes,
        "package_versions": versions, "torch_cpu_threads": torch.get_num_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tracker_config": dataclasses.asdict(config),
    }
    parent_state = None
    parent_hashes = {}
    if args.resume is not None:
        arrays, metadata, _manifest, parent_hashes = _parent_checkpoint(args.resume, bindings)
        parent_state = arrays, metadata
    hashes = {**run.hashes, **code_hashes, **parent_hashes}
    _unchanged(hashes)
    output.mkdir(parents=True, exist_ok=False)
    with provenance(STAGE, output, name="continuation") as record:
        record.param("bindings", bindings)
        record.param("stop_after", stop_after)
        record.param("resume_phase", None if args.resume is None else str(args.resume.resolve()))
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        output_hashes = {}
        for source in sources:
            target = output / "code_snapshot" / source.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            output_hashes[str(target)] = code_hashes[str(source)]
        prepared = time.perf_counter()
        rig = load_rig(run.root / "calibration_snapshot")
        reader = SequenceReader(run.sequence, run.input_root)
        tracker = AcceleratedContinuousPersistentNodeTracker(
            rig, run.initial.copy(), config, recovery_config=recovery,
            persistence_config=persistence)
        if parent_state is None:
            tracker.begin()
        else:
            tracker.restore_state(*parent_state)
        previous_count = tracker.processed_frames
        if previous_count > stop_after or (previous_count == stop_after and stop_after != count):
            raise ValueError("resume must advance beyond the saved prefix or finalize full input")
        expected_previous = None if previous_count == 0 else int(run.frames[previous_count - 1])
        if tracker.previous_frame != expected_previous:
            raise ValueError("checkpoint last frame differs from registered sequence")
        preparation_seconds = time.perf_counter() - prepared
        start_frame = int(run.frames[0]) + previous_count
        iterator = iter_from(reader, start_frame)
        progress_path = output / "progress.jsonl"
        started = time.perf_counter()
        with progress_path.open("x", encoding="utf-8", buffering=1) as progress:
            try:
                for frame in iterator:
                    index = tracker.processed_frames
                    if index >= count or frame.frame_idx != run.frames[index]:
                        raise ValueError("continuation reader yielded an incorrect native frame")
                    tracker.advance(frame)
                    processed = tracker.processed_frames
                    if processed % 25 == 0 or processed == stop_after:
                        event = {"processed": processed,
                                 "phase_processed": processed - previous_count,
                                 "source_frame": frame.frame_idx,
                                 "elapsed_seconds": time.perf_counter() - started}
                        progress.write(json.dumps(event) + "\n")
                        print(json.dumps(event), flush=True)
                    if processed == stop_after and stop_after != count:
                        break
                torch.cuda.synchronize()
            finally:
                iterator.close()
        tracking_seconds = time.perf_counter() - started
        if tracker.processed_frames != stop_after or tracker.initialize_count != 1:
            raise ValueError("continuation did not consume the complete requested phase")
        saved_at = time.perf_counter()
        checkpoint_arrays, checkpoint_metadata = tracker.export_state()
        write_checkpoint(output / "checkpoint", checkpoint_arrays, checkpoint_metadata,
                         bindings=bindings)
        # Independent read at the serialization boundary must recover every byte.
        roundtrip, roundtrip_meta, _ = read_checkpoint(
            output / "checkpoint", expected_bindings=bindings)
        _same_arrays(checkpoint_arrays, roundtrip)
        if roundtrip_meta != checkpoint_metadata:
            raise ValueError("checkpoint metadata round trip differs")
        checkpoint_seconds = time.perf_counter() - saved_at
        checkpoint_manifest = output / "checkpoint/manifest.json"
        record.output("checkpoint_manifest", checkpoint_manifest,
                      sha256=base.sha256(checkpoint_manifest))
        for filename in ("manifest.json", "metadata.json", "arrays.npz"):
            path = output / "checkpoint" / filename
            output_hashes[str(path)] = base.sha256(path)
        complete = stop_after == count
        exact = False
        if complete:
            result = tracker.finish()
            base.validate_result(result, run.frames, reader.nodes)
            verified.persistent_runner.validate_arm_arrays(result, run.frames)
            candidate = output / "candidate.npz"
            np.savez_compressed(candidate, **result)
            record.output("candidate", candidate, sha256=base.sha256(candidate))
            output_hashes[str(candidate)] = base.sha256(candidate)
            # First loading of any prior output coordinates occurs after inference/save.
            with np.load(run.root / "candidate.npz", allow_pickle=False) as old:
                _same_arrays(result, {key: old[key] for key in old.files})
            old_summary = json.loads((run.root / "summary.json").read_text(encoding="utf-8"))
            if tracker.node_recovery_diagnostics != old_summary["node_recovery_diagnostics"]:
                raise ValueError("continuation recovery diagnostics differ from original")
            exact = True
        summary = {
            "schema": "persistent_continuation_phase_v1", "technical_completion": True,
            "full_sequence_complete": complete, "frames": bindings["source_range"],
            "processed_total": stop_after,
            "phase_frames": [start_frame, int(run.frames[0]) + stop_after],
            "phase_processed": stop_after - previous_count,
            "logical_initialization_count": tracker.initialize_count,
            "checkpoint_restore_count": tracker.restore_count, "external_pose_reset_count": 0,
            "temporal_subsampling": False, "original_candidate_21_arrays_byte_exact": exact,
            "original_candidate_loaded_only_after_full_inference": complete,
            "human_annotation_input": False, "preparation_seconds": preparation_seconds,
            "tracking_seconds": tracking_seconds, "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_payload_bytes": (output / "checkpoint/arrays.npz").stat().st_size,
            "retained_history_rows": int(checkpoint_arrays["history_offsets"][-1]),
            "active_beam_count": len(checkpoint_arrays["state_pose"]),
            "node_recovery_diagnostics": tracker.node_recovery_diagnostics,
            "limitations": [
                "Full-state process continuation is not biological identity validation.",
                "The full retained history is saved; storage/restore costs grow with time.",
                "Development-window equivalence does not measure a thirty-minute run.",
                "Wall time excludes input staging, hashes and final comparison.",
            ],
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
                      "processed": stop_after}), flush=True)
    return summary


if __name__ == "__main__":
    main()
