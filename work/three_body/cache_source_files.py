"""Create the authorized, immutable local subset of Medaka source assets.

Copies fixed paths only. Existing output directories are refused; partial failed
copies are retained with failure provenance. No inference or source edits occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from medaka3d.provenance import provenance  # noqa: E402

GIB = 2**30
EXPERIMENT = "20251225_K_1_2_FCMC"
ASSETS = {
    "detectron2.top": (
        "top_detectron2_model/model_final.pth",
        "665e0b2834e9fe7af1810d9e20c457592d7b0c8de23e454667d40ef8b094a7ee",
    ),
    "detectron2.top.config": (
        "top_detectron2_model/detectron2_config.yaml",
        "64574b9580f1570510557b195b37f2ad5bdfb6f9a95dbfc77604432fb2994542",
    ),
    "detectron2.side": (
        "side_detectron2_model_firstprompt/model_final.pth",
        "ea2154060e33a4231de031885808399279ad02ad263c8fa81650b9bf73140e13",
    ),
    "detectron2.side.config": (
        "side_detectron2_model_firstprompt/detectron2_config.yaml",
        "76a81336dbd7a28e8980291581c0c07bf128373fb09593e41d7dab9879e7d6d5",
    ),
}
VIDEOS = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_record(path):
    value = path.stat()
    if not path.is_file():
        raise ValueError(f"required source is not a file: {path}")
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def validate_capacity(total_bytes, free_bytes):
    if total_bytes > 100 * GIB:
        raise ValueError("source subset exceeds the authorized 100 GiB ceiling")
    if free_bytes - total_bytes < 300 * GIB:
        raise ValueError("copy would leave less than the required 300 GiB free")


def run(source_root, out):
    source_root = source_root.resolve(strict=True)
    out = out.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite source cache: {out}")
    if not out.parent.is_dir():
        raise FileNotFoundError("cache parent must already exist")
    source_manifest = source_root / "assets/MANIFEST.json"
    manifest_bytes = source_manifest.read_bytes()
    asset_manifest = json.loads(manifest_bytes)
    relative_paths = ["assets/MANIFEST.json"]
    expected_hashes = {"assets/MANIFEST.json": hashlib.sha256(manifest_bytes).hexdigest()}
    asset_keys = {"assets/MANIFEST.json": "asset_manifest"}
    for key, (suffix, expected) in ASSETS.items():
        relative = f"medaka_202512/detectron2_models/{suffix}"
        entry = asset_manifest["assets"][key]
        if entry["sha256"] != expected:
            raise ValueError(f"asset manifest differs from previously verified model: {key}")
        path = source_root / relative
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"model/config source size differs from asset manifest: {key}")
        relative_paths.append(relative)
        expected_hashes[relative] = expected
        asset_keys[relative] = key
    relative_paths += [
        f"medaka_202512/{EXPERIMENT}/{view}/{name}" for view, name in VIDEOS.items()
    ]
    for relative in relative_paths:
        if not (source_root / relative).resolve(strict=True).is_relative_to(source_root):
            raise ValueError("fixed source path escaped source root")
    initial_stats = {relative: stat_record(source_root / relative) for relative in relative_paths}
    total_bytes = sum(value["bytes"] for value in initial_stats.values())
    free_before = shutil.disk_usage(out.parent).free
    validate_capacity(total_bytes, free_before)
    source_files = [
        Path(__file__),
        REPO / "src/medaka3d/provenance.py",
        REPO / "labs/20260909_three_body/prepare_sequence.py",
    ]
    code_hashes = {path.relative_to(REPO).as_posix(): sha256(path) for path in source_files}
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    with provenance("three_body_local_source_cache", out, name="source_cache") as record:
        record.param("maximum_copy_bytes", 100 * GIB)
        record.param("minimum_remaining_free_bytes", 300 * GIB)
        record.param("copy_bytes", total_bytes)
        record.param("free_bytes_before", free_before)
        record.param("source_code_sha256", code_hashes)
        record.param("source_root", str(source_root))
        record.param("gpu_used", False)
        for source in source_files:
            destination = out / "code_snapshot" / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
        entries = {}
        remaining = total_bytes
        for relative in relative_paths:
            if shutil.disk_usage(out).free - remaining < 300 * GIB:
                raise RuntimeError("free disk space fell below the reserved capacity")
            source, destination = source_root / relative, out / relative
            if stat_record(source) != initial_stats[relative]:
                raise RuntimeError(f"source changed before copy: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            transferred = 0
            tic, last_report = time.perf_counter(), time.perf_counter()
            print(json.dumps({"phase": "copy", "path": relative}), flush=True)
            with source.open("rb") as reader, destination.open("xb") as writer:
                for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    writer.write(block)
                    digest.update(block)
                    transferred += len(block)
                    if time.perf_counter() - last_report >= 15:
                        print(
                            json.dumps({"phase": "copy", "path": relative, "bytes": transferred}),
                            flush=True,
                        )
                        last_report = time.perf_counter()
                writer.flush()
                os.fsync(writer.fileno())
            copied_sha = digest.hexdigest()
            source_after = stat_record(source)
            if source_after != initial_stats[relative] or transferred != source_after["bytes"]:
                raise RuntimeError(f"source changed during copy: {relative}")
            if relative in expected_hashes and copied_sha != expected_hashes[relative]:
                raise ValueError(f"copied source disagrees with trusted model/manifest: {relative}")
            print(json.dumps({"phase": "destination_sha256", "path": relative}), flush=True)
            destination_sha = sha256(destination)
            if destination_sha != copied_sha or destination.stat().st_size != transferred:
                raise ValueError(f"destination full checksum differs: {relative}")
            entries[relative] = {
                "source": str(source),
                "local_path": relative,
                "source_pre": initial_stats[relative],
                "source_post": source_after,
                "bytes": transferred,
                "source_stream_sha256": copied_sha,
                "local_full_sha256": destination_sha,
                "trusted_expected_sha256": expected_hashes.get(relative),
                "asset_key": asset_keys.get(relative),
                "copy_and_verify_seconds": time.perf_counter() - tic,
            }
            record.input(relative, source, sha256=copied_sha)
            record.output(relative, destination, sha256=destination_sha)
            remaining -= transferred
            progress = out / "copy_progress.json"
            progress.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            print(
                json.dumps({"phase": "verified", "path": relative, "bytes": transferred}),
                flush=True,
            )
        for relative in relative_paths:
            if stat_record(source_root / relative) != initial_stats[relative]:
                raise RuntimeError(f"source changed before cache completed: {relative}")
        for path, expected in code_hashes.items():
            if sha256(REPO / path) != expected:
                raise RuntimeError(f"cache construction source changed: {path}")
        free_after = shutil.disk_usage(out).free
        if free_after < 300 * GIB:
            raise RuntimeError("completed copy leaves less than reserved free capacity")
        result = {
            "status": "verified",
            "experiment": EXPERIMENT,
            "source_root": str(source_root),
            "local_root": str(out),
            "total_bytes": total_bytes,
            "free_bytes_before": free_before,
            "free_bytes_after": free_after,
            "elapsed_seconds": time.perf_counter() - started,
            "files": entries,
            "source_code_sha256": code_hashes,
            "scope": "exact copies of three source videos, two weights/configs and asset manifest",
            "limitations": [
                "This subset mirror omits other files mentioned in assets/MANIFEST.json.",
                "Video SHA256 is measured now; no historic trusted video hash is assumed.",
                "Size and mtime are checked before/after copying and again at completion.",
                "No detector inference, calibration changes or tracking is performed.",
            ],
        }
        path = out / "source_cache_manifest.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.output("cache_manifest", path, sha256=sha256(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root, args.out)


if __name__ == "__main__":
    main()
