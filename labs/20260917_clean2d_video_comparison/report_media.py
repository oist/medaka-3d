"""Create JPEG report stills while preserving reviewed PNG/video originals."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2

from medaka3d.aquarium import sha256
from medaka3d.provenance import provenance


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("work/clean2d_video_comparison_v1/review")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("work/clean2d_video_comparison_v1/report_media")
    )
    args = parser.parse_args()
    source_manifest = args.source / "manifest.json"
    original = json.loads(source_manifest.read_text(encoding="utf-8"))
    if original["status"] != "complete" or len(original["assets"]) != 42:
        raise ValueError("Expected the complete, reviewed 42-video comparison")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Report media output must be a new empty directory")
    result = copy.deepcopy(original)
    result["source_manifest"] = str(source_manifest.resolve())
    result["source_manifest_sha256"] = sha256(source_manifest)
    result["report_media_code_sha256"] = sha256(Path(__file__))
    result["still_encoding"] = dict(
        format="JPEG",
        quality=90,
        resize=False,
        lossy_pixel_encoding=True,
        coordinate_geometry_changed=False,
    )
    source_bytes = output_bytes = 0
    with provenance("clean2d_report_stills", args.output, name="prepare") as record:
        record.params_from(args)
        record.input("converter", Path(__file__), sha256=sha256(Path(__file__)))
        record.input("source_manifest", source_manifest, sha256=sha256(source_manifest))
        for asset in result["assets"]:
            name = asset["name"]
            png, video = (args.source / asset[k] for k in ("still", "video"))
            png_hash, video_hash = sha256(png), sha256(video)
            if png_hash != asset["still_sha256"] or video_hash != asset["video_sha256"]:
                raise ValueError(f"Reviewed source changed: {name}")
            record.input(name + "_png", png, sha256=png_hash)
            record.input(name + "_video", video, sha256=video_hash)
            image = cv2.imread(str(png), cv2.IMREAD_COLOR)
            if image is None or image.shape != (870, 2560, 3):
                raise ValueError(f"Unexpected PNG shape: {name}")
            target = args.output / (name + ".jpg")
            if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError("JPEG encoding failed")
            restored = cv2.imread(str(target), cv2.IMREAD_COLOR)
            if restored is None or restored.shape != image.shape:
                raise RuntimeError("JPEG readback shape differs")
            digest = sha256(target)
            asset["report_variant"] = dict(
                type="jpeg_still",
                jpeg_quality=90,
                source_still=str(png.resolve()),
                source_still_sha256=png_hash,
                source_still_bytes=png.stat().st_size,
                output_still_sha256=digest,
                output_still_bytes=target.stat().st_size,
                unchanged_shape=list(image.shape),
                readback_verified=True,
            )
            asset["still"] = target.name
            asset["still_sha256"] = digest
            asset["video"] = str(video.resolve())
            source_bytes += png.stat().st_size
            output_bytes += target.stat().st_size
            record.output(name + "_jpeg", target, sha256=digest)
        result["still_bytes"] = dict(
            original_png=source_bytes,
            report_jpeg=output_bytes,
            report_fraction=output_bytes / source_bytes,
        )
        manifest = args.output / "manifest.json"
        write_json(manifest, result)
        record.output("manifest", manifest, sha256=sha256(manifest))
    print(json.dumps(result["still_bytes"], indent=2))


if __name__ == "__main__":
    main()
