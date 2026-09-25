"""Small-file, CPU-only cost arithmetic for the frozen long continuation runner.

No videos, mask arrays, tracker, detector, GPU or unfinished outputs are opened.
Uses successful first-chunk metadata and the previous explicit allocation model.
"""

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"
OUT = ROOT / "work/three_body/long_continuation_execution_feasibility_v1"
K, C = 180, 1660


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sealed_file(root, seal_name, filename, hashes):
    seal_path = root / seal_name
    seal = read_json(seal_path)
    if seal.get("status") != "ok":
        raise ValueError("source result is incomplete")
    path = root / filename
    entries = [v for v in seal["outputs"].values() if Path(v["path"]).resolve() == path]
    if not entries or any(v["sha256"] != sha(path) for v in entries):
        raise ValueError("metadata is not bound to its successful source result")
    hashes[str(seal_path)] = sha(seal_path)
    hashes[str(path)] = sha(path)
    return path


def candidate_bytes(path):
    total = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError("unexpected NPY header version")
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                total += int(np.prod(shape)) * dtype.itemsize
    return total


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    code = [
        Path(__file__).resolve(),
        *(
            LAB / name
            for name in (
                "run_long_continuation.py",
                "long_tracking_inputs.py",
                "long_tracking_protocol.json",
                "continuous_tracker.py",
                "tracker_checkpoint.py",
                "observations.py",
                "source_sequence.py",
                "resumable_observations.py",
                "accelerated_tracker.py",
                "prepare_chunked_source_masks.py",
                "prepare_source_masks.py",
                "chunk_masks.py",
            )
        ),
        ROOT / "src/medaka3d/provenance.py",
    ]
    hashes = {str(p): sha(p) for p in code}
    run = ROOT / "work/three_body/long_continuation_first1660_v1"
    summary = read_json(sealed_file(run, "continuation.provenance.json", "summary.json", hashes))
    manifest = read_json(
        sealed_file(run, "continuation.provenance.json", "checkpoint/manifest.json", hashes)
    )
    archive = sealed_file(run, "continuation.provenance.json", "checkpoint/arrays.npz", hashes)
    candidate = sealed_file(run, "continuation.provenance.json", "candidate.npz", hashes)
    detector_root = ROOT / "work/three_body/long_first_chunk_independent_audit_v1"
    detector = read_json(
        sealed_file(detector_root, "audit.provenance.json", "summary.json", hashes)
    )
    old_root = ROOT / "work/three_body/long_run_feasibility_v3"
    old = read_json(sealed_file(old_root, "audit.provenance.json", "summary.json", hashes))
    if summary["processed_interval"] != [49800, 51460] or summary["full_sequence_complete"]:
        raise ValueError("wrong operational pilot")

    rows = summary["retained_history_rows"]
    history = sum(
        v["nbytes"]
        for key, v in manifest["arrays"].items()
        if key.startswith("history_") and key != "history_offsets"
    )
    row_bytes = history // rows
    assert history == rows * row_bytes
    fixed = sum(
        v["nbytes"] for key, v in manifest["arrays"].items() if not key.startswith("history_")
    )
    compressed = {"history": 0, "fixed": 0}
    with zipfile.ZipFile(archive) as data:
        for key, entry in manifest["arrays"].items():
            group = "history" if key.startswith("history_") else "fixed"
            compressed[group] += data.getinfo(entry["storage_key"] + ".npy").compress_size
    target_frames = K * C
    triangle = K * (K + 1) // 2
    history_at_end = target_frames * 6 * row_bytes + 8 * (target_frames + 1)
    # The older measured ownership model lacks the two per-beam age arrays.
    # Header sizes are current CPython/NumPy object sizes, not an RSS measurement.
    added_age_bytes = sys.getsizeof(np.zeros((3, 3), np.int64)) + sys.getsizeof(
        np.zeros(3, np.int64)
    )
    b18 = old["allocation_model"]["18"]["projected_298800_frames_bytes"]
    adjusted_b18 = b18 + target_frames * 6 * added_age_bytes
    adjusted_b24 = adjusted_b18 + target_frames * 6 * (27 * 8 + 5 * 8 + 3)
    raw_bytes = detector["sizes"]["raw_logical_bytes"]
    track_hours = summary["tracking_seconds"] * K / 3600
    detector_hours = detector["timing"]["chunk_seal_wall_seconds"] * K / 3600
    result = {
        "schema": "long_execution_feasibility_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "no_new_tracking_or_GPU": True,
        "no_unfinished_run_opened": True,
        "target_frames": target_frames,
        "chunk_frames": C,
        "chunks": K,
        "observed_first_chunk": {
            "tracking_seconds": summary["tracking_seconds"],
            "preparation_seconds": summary["preparation_seconds"],
            "reader_setup_seconds": summary["reader_setup_seconds"],
            "checkpoint_seconds": summary["checkpoint_seconds"],
            "checkpoint_file_bytes": summary["checkpoint_payload_bytes"],
            "detector_chunk_wall_seconds": detector["timing"]["chunk_seal_wall_seconds"],
            "retained_history_rows": rows,
            "history_bytes_per_record": row_bytes,
            "checkpoint_fixed_logical_bytes": fixed,
        },
        "descriptive_linear_arithmetic_not_prediction": {
            "tracking_hours": track_hours,
            "detector_chunk_wall_hours": detector_hours,
            "sum_hours_before_extra_overhead": track_hours + detector_hours,
            "linear_all_first_checkpoint_time_hours_for_180_prefixes": summary["checkpoint_seconds"]
            * triangle
            / 3600,
            "checkpoint_timing_warning": "This assumes the entire 3.862s grows with history; "
            "constant previous-image compression and unmeasured restore cost make it "
            "an illustration, not a measured total or bound.",
        },
        "static_180_prefix_work": {
            "prefix_chunk_visits": triangle,
            "semantic_mask_frame_stack_validations": 3 * C * triangle,
            "single_phase_semantic_mask_frame_stack_validations": 3 * C * K,
            "repetition_factor": (K + 1) / 2,
            "new_mask_frame_stack_decodes_during_tracking": 3 * C * K,
            "native_image_decodes_and_BGR_SHA": 3 * C * K,
            "full_raw_SHA_passes_per_video_180_phases": 6 * K,
            "full_raw_SHA_passes_per_video_one_phase": K + 5,
            "raw_SHA_target_bytes_180_phases": raw_bytes * 6 * K,
            "raw_SHA_target_bytes_one_phase": raw_bytes * (K + 5),
            "full_mask_NPZ_SHA_calls_180_phases": 3 * (4 * triangle + K),
            "semantic_validation_sensitivity_hours": {
                str(seconds): seconds * triangle / 3600 for seconds in (1, 5, 10)
            },
            "sensitivity_unit": "Unmeasured seconds to validate one 1660-frame chunk in all "
            "three views; examples are algebra, not observed throughput.",
            "raw_complexity": "For this fixed full MP4 size B, repeated full hashes are "
            "O(K*B). Old mask decode and full-prefix history operations are O(K^2). "
            "SHA target bytes are not physical disk I/O because of file caches.",
        },
        "memory_and_storage_models": {
            "full_6beam_history_array_bytes": history_at_end,
            "checkpoint_payload_logical_bytes_same_previous_observation": history_at_end + fixed,
            "all_180_snapshot_logical_bytes": C * 6 * row_bytes * triangle
            + 8 * (C * triangle + K)
            + fixed * K,
            "full_history_ownership_adjusted_B18_bytes": adjusted_b18,
            "full_history_ownership_adjusted_B24_bytes": adjusted_b24,
            "age_arrays_added_bytes_per_beam_record": added_age_bytes,
            "final_candidate_21array_logical_bytes_same_dtype": candidate_bytes(candidate) * K,
            "same_compression_full_checkpoint_bytes": compressed["fixed"]
            + compressed["history"] * K,
            "same_compression_all_180_checkpoint_bytes": compressed["fixed"] * K
            + compressed["history"] * triangle,
            "same_firstchunk_mask_NPZ_bytes_full_input": detector["sizes"]["mask_npz_bytes"] * K,
            "same_firstchunk_nonraw_staging_bytes_full_input": detector["sizes"][
                "chunk_non_raw_logical_bytes"
            ]
            * K,
            "workspace_disk_free_bytes_at_audit": shutil.disk_usage(ROOT).free,
            "scope": "Array sizes are header-derived; ownership extrapolates the earlier "
            "1000-frame CPU object model plus current age-array sizes. Neither is peak "
            "RSS or a 30-minute allocation. Compression and mask density can change.",
        },
        "fixed_contract": [
            "Initial 49800 registration; 166fps order; all raw BGR and all packed mask "
            "values/rows; SL/calibration; background pixels; detector outputs.",
            "Tracker/flow/recovery/persistence configs, float64 objective, optimizer order, "
            "steps, beam order/budget, review, all history parents, previous observation.",
            "Frozen source/runtime/backend bindings and codec validation; final prefix is "
            "provisional until the complete native interval ends.",
        ],
        "minimal_plan": [
            "Use existing frozen --through-frame to consume several already completed chunks "
            "in one phase. _advance_chunks preserves the same tracker and skips consumed "
            "chunks before constructing readers. No new code or numerical setting is needed.",
            "Keep failed attempts immutable and validate the actual second-chunk restore "
            "before increasing the phase length. Choose phase endpoints by acceptable redo "
            "cost and available completed inputs, not by identity metrics.",
            "If later necessary, a separate verified-reader wrapper can deduplicate full raw "
            "SHA per canonical file and cache semantic mask validation by payload SHA + "
            "schema + validator-code SHA. Keep compressed-payload hashes and native BGR "
            "checks. mtime-only trust would weaken the evidence contract.",
            "Only if frequent checkpoints remain necessary: immutable history blocks plus "
            "small state manifests, or release provably unused temporary copies. This "
            "requires a new explicit compatibility contract and all-array continuation "
            "parity; current exact code bindings reject silent runner replacement.",
        ],
        "limits": [
            "No full 30-minute runtime, peak RAM, disk growth or GPU reservation was measured.",
            "Frame-local GPU graphs are bounded to four cache entries at <=24 proposals; "
            "no growing CUDA tensor appears in CPU history. Long native-library leaks "
            "or graph-cache churn are not ruled out by source inspection.",
            "Existing decoded masks are chunk-resident, unlike the older all-mask reader "
            "estimate. Huge/abundant masks can still make an individual chunk expensive.",
            "No ID truth or quality equivalence to previous seed/background windows is claimed.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    with provenance("long_execution_feasibility_readonly", OUT, name="audit") as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path in code:
            target = OUT / "code_snapshot" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if sha(target) != hashes[str(path)]:
                raise ValueError("source snapshot changed")
        (OUT / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        note = ROOT / "work/three_body/long_execution_note.md"
        report = note.read_bytes()
        hashes[str(note)] = hashlib.sha256(report).hexdigest()
        (OUT / "FEASIBILITY.md").write_bytes(report)
        record.input(str(note), note, sha256=hashes[str(note)])
        outputs = {}
        for path in OUT.rglob("*"):
            if path.is_file():
                outputs[str(path)] = sha(path)
                record.output(path.relative_to(OUT).as_posix(), path, sha256=outputs[str(path)])
        if any(sha(path) != value for path, value in {**hashes, **outputs}.items()):
            raise RuntimeError("evidence changed during read-only arithmetic")
        record.param("no_new_tracking_or_GPU", True)
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
    print(
        json.dumps(
            {
                "output": str(OUT),
                "status": "ok",
                "tracking_hours": track_hours,
                "detector_chunk_hours": detector_hours,
                "full_history_arrays_GiB": history_at_end / 2**30,
            }
        )
    )


if __name__ == "__main__":
    main()
