"""Create smaller report-only MP4 copies without changing reviewed originals."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def decode(path, expected_frames, expected_shape, preview=None):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    count = 0
    try:
        if not cap.isOpened() or not np.isclose(cap.get(cv2.CAP_PROP_FPS), 30):
            raise RuntimeError("Report variant FPS/open check failed")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if list(frame.shape) != expected_shape:
                raise RuntimeError("Report variant shape changed")
            if (
                preview is not None
                and count == expected_frames // 2
                and not cv2.imwrite(str(preview), frame)
            ):
                raise RuntimeError("Report preview could not be saved")
            count += 1
    finally:
        cap.release()
    if count != expected_frames:
        raise RuntimeError(f"Report variant lost frames: {count} != {expected_frames}")
    return {"verified": True, "decoded_frames": count, "shape": expected_shape}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("work/clean_tracklets_v2/review"))
    parser.add_argument("--output", type=Path, default=Path("work/clean_tracklets_v2/report_media"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_manifest = args.source / "manifest.json"
    original = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest = {key: copy.deepcopy(value) for key, value in original.items() if key != "assets"}
    manifest.update(
        assets=[], status="in_progress", original_review_directory=str(args.source.resolve())
    )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    total_before, total_after = 0, 0
    started = time.perf_counter()
    with provenance("clean_tracklets_report_media", args.output, name="report_media") as rec:
        rec.params_from(args)
        rec.input("code", Path(__file__), sha256=sha256(Path(__file__)))
        rec.input("original_manifest", source_manifest, sha256=sha256(source_manifest))
        rec.param("encoder", {"codec": "libx264", "crf": 28, "preset": "fast", "threads": 2})
        rec.param("spatial_resize", False)
        rec.param("temporal_resampling", False)
        shutil.copy2(__file__, args.output / "report_media.py")
        for original_asset in original["assets"]:
            asset = copy.deepcopy(original_asset)
            source = args.source / asset["video"]
            digest = sha256(source)
            if digest != asset["video_sha256"]:
                raise ValueError("Original reviewed video changed")
            rec.input(asset["name"] + "_source", source, sha256=digest)
            output = args.output / source.name
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "28",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
                "-movflags",
                "+faststart",
                str(output),
            ]
            print(f"Encoding report copy: {asset['name']}", flush=True)
            subprocess.run(command, check=True, capture_output=True)
            preview = (
                args.output / "label_readability_preview.png"
                if len(manifest["assets"]) == 0
                else None
            )
            verified = decode(
                output, asset["output_frames"], asset["full_decode"]["shape"], preview
            )
            before, after = source.stat().st_size, output.stat().st_size
            total_before += before
            total_after += after
            asset["video"] = output.name
            asset["video_sha256"] = sha256(output)
            asset["still"] = str((args.source / asset["still"]).resolve())
            if sha256(Path(asset["still"])) != asset["still_sha256"]:
                raise ValueError("Original reviewed PNG changed")
            asset["full_decode"] = verified
            asset["ffmpeg_command"] = command
            asset["report_variant"] = {
                "source_video": str(source.resolve()),
                "source_sha256": digest,
                "source_bytes": before,
                "output_bytes": after,
                "byte_ratio": after / before,
                "crf": 28,
                "preset": "fast",
                "threads": 2,
                "same_spatial_resolution": True,
                "all_native_sample_frames_preserved": True,
                "note": "Lossy report copy; original reviewed media remains unchanged",
            }
            manifest["assets"].append(asset)
            manifest["source_video_bytes"] = total_before
            manifest["output_video_bytes"] = total_after
            manifest["video_byte_ratio"] = total_after / total_before
            write_json(args.output / "manifest.json", manifest)
            rec.output(asset["name"], output, sha256=asset["video_sha256"])
            if preview is not None:
                rec.output("label_readability_preview", preview, sha256=sha256(preview))
            print(f"  {before:,} -> {after:,} bytes ({after / before:.3f})", flush=True)
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "manifest.json", manifest)
        rec.output(
            "manifest", args.output / "manifest.json", sha256=sha256(args.output / "manifest.json")
        )
        rec.output(
            "code_snapshot",
            args.output / "report_media.py",
            sha256=sha256(args.output / "report_media.py"),
        )


if __name__ == "__main__":
    main()
