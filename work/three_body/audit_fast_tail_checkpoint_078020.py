"""Read one committed first continuation checkpoint and its audited parent once.

This checks persistence, ordered history and file integrity, not physical IDs.
No tracker, renderer, CUDA, raw-image reader, detector or process handle is used.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import tracker_checkpoint as codec  # noqa: E402 - load the frozen lab codec after its path.

ROOT = WORK / "fast_tail_five_minute_continuation_v1"
PARENT = WORK / "fast_tail_two_minute_continuous_v1/checkpoints/through_076360"
CURRENT = ROOT / "checkpoints/through_078020"
PARENT_SHA = "cef133c3e6192e373df095a3ec8b3b1cd546363a3aed8a6bf8ead670ff529cf6"
CURRENT_SHA = "0e8d002761370a71c97c4a9d09dece1d65be462c007008b42e8e384309aa8354"
RUNNER_SHA = "d90e9eb49038dac4fd1e18068334ba89eb8df1a813a37c6aec071c20db1a61a6"
CODEC_SHA = "39272a664d3f2056d4d876b75e9241ae1a775d19f7352a0423a3e1c76125a32b"
FIELDS = (
    "frame_idx",
    "pose_xyz",
    "observed",
    "sources",
    "identity_review",
    "mask_supported",
    "body_supported",
    "missing",
    "costs",
    "path_costs",
    "loss_parts",
    "score_margin",
    "parent",
    "proposal_kind",
    "hypotheses",
    "proposal_flow_nodes",
    "proposal_flow_parent",
    "node_missing",
    "direct_pectoral_age",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, inputs, expected=None):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(expected is None or digest == expected, "input SHA differs: " + str(path))
    inputs[str(path)] = dict(sha256=digest, size_bytes=len(data))
    return json.loads(data)


def main():
    inputs = {}
    runner = WORK / "continue_fast_tail_five_minutes.py"
    codec_path = LAB / "tracker_checkpoint.py"
    require(sha(runner) == RUNNER_SHA and sha(codec_path) == CODEC_SHA, "frozen source differs")
    for path in (Path(__file__).resolve(), runner, codec_path):
        inputs[str(path)] = dict(sha256=sha(path), size_bytes=path.stat().st_size)
    old_manifest = read(PARENT / "manifest.json", inputs, PARENT_SHA)
    new_manifest = read(CURRENT / "manifest.json", inputs, CURRENT_SHA)
    binding = new_manifest["bindings"]
    require(
        binding["schema"] == "fast_tail_five_minute_continuation_v1_checkpoint"
        and binding["scientific_interval"] == [66400, 116200]
        and binding["process_interval"] == [76360, 116200]
        and binding["parent_checkpoint_sha256"] == PARENT_SHA
        and binding["source_sha256"][str(runner)] == RUNNER_SHA
        and binding["source_sha256"][str(codec_path)] == CODEC_SHA
        and binding["original_binding"] == old_manifest["bindings"]["original_binding"],
        "checkpoint source/interval/parent binding differs",
    )
    plan = read(ROOT / "plan.json", inputs)
    require(plan["checkpoint_binding"] == binding, "plan and committed checkpoint differ")
    snapshot = ROOT / "code_snapshot/work/three_body/continue_fast_tail_five_minutes.py"
    require(sha(snapshot) == RUNNER_SHA, "executed runner snapshot differs")
    inputs[str(snapshot)] = dict(sha256=RUNNER_SHA, size_bytes=snapshot.stat().st_size)
    audit_root = WORK / "fast_tail_interrupted_minute_checkpoint_audit_v1"
    audit = read(audit_root / "audit.provenance.json", inputs, binding["audited_parent_sha256"])
    require(audit["status"] == "ok", "parent checkpoint audit is not successful")
    row = audit["outputs"]["summary.json"]
    summary = read(audit_root / "summary.json", inputs, row["sha256"])
    require(
        summary["checkpoint_manifest_sha256"] == PARENT_SHA
        and summary["eligible_for_exact_state_restore"] is True,
        "wrong or ineligible audited parent",
    )
    progress = read(ROOT / "checkpoint_records/progress_078020.json", inputs)
    require(len(progress["checkpoints"]) == 1, "first checkpoint commit marker differs")
    runner_row = progress["checkpoints"][0]
    require(
        runner_row
        == dict(
            path=str(CURRENT),
            arrays=39,
            manifest_sha256=CURRENT_SHA,
            payload_bytes=new_manifest["payload"]["bytes"],
            previous_frame=78019,
            processed_frames=28220,
            initialize_count=1,
            restore_count=5,
        ),
        "committed runner checkpoint row differs",
    )
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        record = read(ROOT / f"checkpoint_records/078020_{name}.json", inputs)
        expected_sha = (
            CURRENT_SHA
            if name == "manifest.json"
            else new_manifest["payload" if name == "arrays.npz" else "metadata"]["sha256"]
        )
        require(
            record["path"] == str(CURRENT / name)
            and record["sha256"] == expected_sha
            and record["size_bytes"] == (CURRENT / name).stat().st_size,
            "checkpoint file record differs: " + name,
        )
    loaded = []
    for path, manifest in ((PARENT, old_manifest), (CURRENT, new_manifest)):
        arrays, metadata, verified = codec.read_checkpoint(
            path, expected_bindings=manifest["bindings"]
        )
        require(verified == manifest and len(arrays) == 39, "complete39 codec read differs")
        for key in ("metadata", "payload"):
            record = manifest[key]
            inputs[str(path / record["file"])] = dict(
                sha256=record["sha256"], size_bytes=record["bytes"]
            )
        loaded.append((arrays, metadata))
    (old, old_meta), (new, new_meta) = loaded
    require(set(old) == set(new), "full state field membership differs")
    require(
        [
            old_meta[k]
            for k in ("initialize_count", "restore_count", "previous_frame", "processed_frames")
        ]
        == [1, 4, 76359, 26560]
        and [
            new_meta[k]
            for k in ("initialize_count", "restore_count", "previous_frame", "processed_frames")
        ]
        == [1, 5, 78019, 28220],
        "saved lifecycle differs",
    )
    require(
        old_meta["context"] == new_meta["context"]
        and old_meta["record_fields"] == new_meta["record_fields"] == list(FIELDS),
        "tracking context/retained history fields changed",
    )
    old_offsets, offsets = old["history_offsets"], new["history_offsets"]
    require(
        old_offsets.dtype == offsets.dtype == np.int64
        and old_offsets.shape == (26561,)
        and offsets.shape == (28221,)
        and old_offsets[0] == offsets[0] == 0
        and (np.diff(old_offsets) > 0).all()
        and (np.diff(offsets) > 0).all(),
        "native history offset extent differs",
    )
    prefix_checks = {}
    for key in ("history_offsets", *("history_" + name for name in FIELDS)):
        expected = old[key]
        end = len(old_offsets) if key == "history_offsets" else int(old_offsets[-1])
        actual = new[key][:end]
        unicode = actual.dtype.kind == expected.dtype.kind == "U"
        exact = actual.shape == expected.shape and (
            np.array_equal(actual, expected)
            if unicode
            else actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
        )
        require(exact, "retained history prefix differs: " + key)
        prefix_checks[key] = (
            "unicode_values_exact_capacity_ignored" if unicode else "numeric_BYTES_exact"
        )
    for key in FIELDS:
        require(
            len(old["history_" + key]) == int(old_offsets[-1])
            and len(new["history_" + key]) == int(offsets[-1]),
            "history record offset cardinality differs: " + key,
        )
    counts = np.diff(offsets)[26560:]
    require(
        counts.shape == (1660,) and np.array_equal(counts, np.full(1660, 6)),
        "six ordered beams required",
    )
    suffix_start = int(old_offsets[-1])
    suffix_frames = new["history_frame_idx"][suffix_start:]
    require(
        np.array_equal(suffix_frames, np.repeat(np.arange(76360, 78020), counts)),
        "appended native frame order differs",
    )
    parents = new["history_parent"][suffix_start:]
    require(
        parents.shape == (9960,) and (parents >= 0).all() and (parents < 6).all(),
        "appended beam parent index differs",
    )
    # Recheck all small bound inputs. Payloads were each loaded and verified once by codec.
    for path, record in inputs.items():
        if Path(path).name == "arrays.npz":
            require(Path(path).stat().st_size == record["size_bytes"], "checkpoint payload resized")
        else:
            require(sha(path) == record["sha256"], "small bound input changed")
    report = dict(
        schema="fast_tail_continuation_first_checkpoint_saved_audit_v1",
        status="PASS",
        checked_at=datetime.now(UTC).isoformat(),
        checkpoint=str(CURRENT),
        parent_checkpoint=str(PARENT),
        checkpoint_manifest_sha256=CURRENT_SHA,
        parent_manifest_sha256=PARENT_SHA,
        runner_sha256=RUNNER_SHA,
        scientific_interval=[66400, 116200],
        process_interval=[76360, 116200],
        committed_through=78020,
        initialize_count=1,
        restore_count=5,
        previous_frame=78019,
        processed_frames=28220,
        all39_codec_integrity_both_snapshots=True,
        checkpoint_payloads_loaded_once_each=True,
        file_records_SHA_size_all3_match=True,
        runner_committed_row_exact=True,
        retained_native_frames=26560,
        retained_beam_records=suffix_start,
        retained_all19_and_offsets=prefix_checks,
        appended_native_interval=[76360, 78020],
        appended_native_frames=1660,
        appended_beam_records=9960,
        ordered_native_frames_all6_beams=True,
        appended_parent_indices_valid=True,
        changed_current_states_compared_to_old=False,
        new_ID_initialization_count=0,
        physical_ID_truth=False,
        five_minute_complete=False,
        process_exit_status_not_checked=True,
        GPU_initialized=False,
        raw_image_reads=0,
        tracker_calls=0,
        inputs=inputs,
    )
    output = WORK / "FAST_TAIL_CHECKPOINT_078020_SAVED_AUDIT.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"inputs", "retained_all19_and_offsets"}},
            indent=2,
        )
    )
    print("AUDIT_SHA256 " + sha(output))


if __name__ == "__main__":
    main()
