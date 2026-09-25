"""Warm CPU timing of one fixed three-view chunk; no raw video/GPU/algorithm changes."""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src")]
import chunk_masks as original  # noqa: E402
import prepare_sequence  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

CHUNK = (
    REPO
    / "work/three_body/inputs/long_source_masks_49800-348600_v1/chunks"
    / "f064740-066400/attempt_0001"
)
START, STOP = 64740, 66400
SOURCE_PINS = {
    LAB / "chunk_masks.py": "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81",
    LAB / "prepare_sequence.py": "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(data):
    return {
        key: dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            sha256=prepare_sequence.array_sha256(value),
        )
        for key, value in data.items()
    }


def timed(function):
    wall, cpu = time.perf_counter(), time.process_time()
    result = function()
    return result, dict(
        wall_seconds=time.perf_counter() - wall, process_cpu_seconds=time.process_time() - cpu
    )


def decompress(path):
    with np.load(path, allow_pickle=False) as archive:
        require(
            len(archive.files) == 13 and len(set(archive.files)) == 13,
            "exactly thirteen unique packed arrays required",
        )
        return {key: archive[key] for key in archive.files}


def trial(path, expected, view, repetition, warmup):
    stamp = datetime.now(UTC).isoformat()
    digest, hashing = timed(lambda: sha(path))
    require(digest == expected, "timed input NPZ SHA differs")
    data, decoding = timed(lambda: decompress(path))
    before = fingerprint(data)  # Additional invariance QA is outside all three timers.
    _, validation = timed(lambda: original.validate_packed_masks(data))
    require(fingerprint(data) == before, "original validator changed its input arrays")
    require(
        np.array_equal(data["frame_idx"], np.arange(START, STOP, dtype=np.int64)),
        "fixed native chunk differs",
    )
    row = dict(
        view=view,
        repetition=repetition,
        warmup=warmup,
        started_at=stamp,
        finished_at=datetime.now(UTC).isoformat(),
        file_sha256=digest,
        file_sha=hashing,
        decompress_all_13_arrays=decoding,
        original_validator=validation,
    )
    metrics = dict(
        file_bytes=path.stat().st_size,
        uncompressed_array_bytes=sum(value.nbytes for value in data.values()),
        mask_instances=len(data["mask_boxes"]),
        dense_frame_sha_payload_bytes=len(data["mask_boxes"]) * 1200 * 1200,
        native_frames=len(data["frame_idx"]),
        available_frames=int(data["view_available"].sum()),
        zero_instance_frames=int((np.diff(data["frame_offsets"]) == 0).sum()),
        thirteen_array_fingerprints=before,
    )
    return row, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    output.mkdir(parents=True, exist_ok=False)
    code = {
        Path(__file__).resolve(),
        *SOURCE_PINS,
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/assets.py",
        REPO / "src/medaka3d/__init__.py",
    }
    initial_code = {path: sha(path) for path in sorted(code)}
    require(
        all(initial_code[path] == value for path, value in SOURCE_PINS.items()),
        "frozen validator source changed",
    )
    with provenance("long_mask_validation_cpu_profile", output, name="profile") as record:
        record.params_from(args)
        seal_path, manifest_path = CHUNK / "staging.provenance.json", CHUNK / "manifest.json"
        seal = read(seal_path)
        require(
            seal["status"] == "ok" and seal["stage"] == "prepare_independent_mask_chunk",
            "successful fixed independent mask chunk required",
        )
        manifest_digest = sha(manifest_path)
        require(
            Path(seal["outputs"]["manifest"]["path"]).resolve() == manifest_path
            and manifest_digest == seal["outputs"]["manifest"]["sha256"],
            "chunk manifest not bound to success seal",
        )
        manifest = read(manifest_path)
        require(
            manifest["frames"] == list(range(START, STOP))
            and manifest["image_shape_hw"] == [1200, 1200]
            and manifest["fps"] == 166
            and manifest["experiment"] == "20251225_K_1_2_FCMC",
            "unexpected fixed profile chunk",
        )
        inputs = {**initial_code, seal_path: sha(seal_path), manifest_path: manifest_digest}
        masks = {}
        for view in VIEWS:
            info, registered = manifest["masks"][view], seal["outputs"][view + "_masks"]
            path = (CHUNK / info["path"]).resolve(strict=True)
            require(
                path.is_relative_to(CHUNK)
                and Path(registered["path"]).resolve() == path
                and info["sha256"] == registered["sha256"] == sha(path),
                "fixed mask path/SHA binding differs",
            )
            inputs[path] = info["sha256"]
            masks[view] = path
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
        for path, digest in initial_code.items():
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == digest, "source snapshot differs")
        rows, view_metrics = [], {}
        for repetition in range(3):
            for view in VIEWS:
                row, metrics = trial(
                    masks[view], inputs[masks[view]], view, repetition, warmup=repetition == 0
                )
                if view in view_metrics:
                    require(view_metrics[view] == metrics, "packed arrays changed across trials")
                view_metrics[view] = metrics
                rows.append(row)
                print(json.dumps(row), flush=True)
        measured = [row for row in rows if not row["warmup"]]
        per_pass = []
        for repetition in (1, 2):
            group = [row for row in measured if row["repetition"] == repetition]
            per_pass.append(
                dict(
                    repetition=repetition,
                    **{
                        key: sum(row[key]["wall_seconds"] for row in group)
                        for key in ("file_sha", "decompress_all_13_arrays", "original_validator")
                    },
                )
            )
        means = {
            key: float(np.mean([row[key] for row in per_pass]))
            for key in ("file_sha", "decompress_all_13_arrays", "original_validator")
        }
        report = dict(
            schema="long_mask_validation_cpu_profile_v1",
            status="ok",
            chunk=str(CHUNK),
            native_interval=[START, STOP],
            views=list(VIEWS),
            warmup_passes=1,
            measured_passes=2,
            measured_frame_stack_validations=2 * 3 * (STOP - START),
            rows=rows,
            per_view=view_metrics,
            measured_pass_totals_seconds=per_pass,
            mean_three_view_chunk_seconds=means,
            possible_receipt_semantic_time_avoided_seconds=means["original_validator"],
            possible_receipt_no_decompression_time_avoided_seconds=(
                means["original_validator"] + means["decompress_all_13_arrays"]
            ),
            verified_receipt_hit_speedup_measured=False,
            runtime=dict(
                python=sys.version,
                executable=sys.executable,
                platform=platform.platform(),
                numpy=np.__version__,
            ),
            concurrent_condition="Root's existing D2 stager and continuous tracker remain active. "
            "No suspension/priority change, no isolated-machine timing. "
            "Their device utilization is not measured by this script.",
            limitations=[
                "One fixed chunk and ordered top/side1/side2 passes; two warm measurements only.",
                "Input SHA preflight and one warmup precede timing; no cold-storage benchmark.",
                "file_sha measures compressed NPZ bytes. original_validator includes dense mask "
                "decode, decoded SHA, structural/box checks; internal costs are not isolated.",
                "Array-fingerprint invariance QA is outside the three reported timing columns.",
                "Receipt lookup, seal checks, required current file SHA, history/backtrace, "
                "source/runtime validation and final guards would remain additional costs.",
                "No raw video read, GPU query/work, detector, tracker, GT or ID evaluation.",
                "No validator/source monkeypatch or production edits; the original validator "
                "is called directly on every one of the thirteen arrays.",
            ],
        )
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        lines = [
            "# 固定1チャンクのmask検証費用\n",
            "[64740,66400)・1660frame・3視点。元validatorへ全13配列を直接渡し、"
            "1回warmup後に2回測定しました。D2と追跡は並行継続、独占CPUの測定ではありません。\n",
            "| 3視点1チャンクのwall time | 測定1 | 測定2 | 平均 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key in means:
            lines.append(
                f"| {key} | {per_pass[0][key]:.6f}s | {per_pass[1][key]:.6f}s | {means[key]:.6f}s |"
            )
        lines += [
            "",
            "SHA列は圧縮NPZの全byte hashです。validator列はdense mask復元・"
            "各frame SHA・構造/bbox検査を含み、内部hashだけの時間とは分離していません。",
            "",
            f"同じbyteに結んだ成功receiptで全semantic再検証を省ける場合、今回の尺度では"
            f"1チャンクあたり約{means['original_validator']:.3f}秒を避ける余地があります。"
            "解凍も不要ならその列も省けますが、現在NPZ SHA・receipt検証・親状態監査等は残り、"
            "実装済みreceiptの実測速度や全30分の短縮予測ではありません。",
            "",
            "入力・source・全13配列は3回とも不変。原動画再読/新GPU処理/追跡/GTなし。"
            "warmupを測定平均へ含めず、全9行のwall/process CPU timeとSHAをsummaryへ保存しました。",
        ]
        (output / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs = {}
        for path in output.rglob("*"):
            if path.is_file():
                outputs[path] = sha(path)
                record.output(path.relative_to(output).as_posix(), path, sha256=outputs[path])
        require(all(sha(path) == digest for path, digest in inputs.items()), "input/source changed")
        require(all(sha(path) == digest for path, digest in outputs.items()), "output changed")
        record.param("all_input_source_snapshot_output_sha_unchanged", True)
        record.param("raw_video_detector_GPU_tracking_GT_used", False)
    print(json.dumps(dict(status="ok", output=str(output), means=means)), flush=True)


if __name__ == "__main__":
    main()
