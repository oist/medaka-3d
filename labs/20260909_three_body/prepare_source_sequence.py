"""Create and fully audit a local original-video descriptor without copying videos.

The immutable parent retains its lossless clips. This command creates hardlinks
to the verified local originals, copies small mask/background files, and checks
every emitted image, detection and mask against the parent SequenceReader.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from observations import SequenceReader, sha256  # noqa: E402
from source_sequence import SOURCE_TOTAL_FRAMES, SourceSequenceReader  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

VIDEO_NAMES = dict(zip(VIEWS, ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4"), strict=True))


def contained_file(root, relative):
    text = str(relative).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or ":" in text or not path.parts:
        raise ValueError("input path must be relative and remain inside its root")
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root.resolve(strict=True)) or not result.is_file():
        raise ValueError("input path escaped its root or is not a file")
    return result


def copy_exclusive(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, 1024 * 1024)


def stat_record(path):
    value = path.stat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns)


def preflight(parent, cache, output, input_root):
    if output.exists() or output.is_symlink():
        raise FileExistsError("compact descriptor output already exists; no overwrite")
    if not output.resolve().is_relative_to(REPO):
        raise ValueError("new descriptor and hardlinks must remain inside this workspace")
    for path in (parent, cache):
        if not path.resolve(strict=True).is_relative_to(REPO):
            raise ValueError("parent and verified source cache must be inside this workspace")
    parent_manifest = parent / "manifest.json"
    cache_manifest = cache / "source_cache_manifest.json"
    cache_provenance = cache / "source_cache.provenance.json"
    original = json.loads(parent_manifest.read_text(encoding="utf-8"))
    cached = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cached.get("status") != "verified":
        raise ValueError("source cache must have completed verified status")
    sealed = json.loads(cache_provenance.read_text(encoding="utf-8"))
    if sealed.get("status") != "ok":
        raise ValueError("source cache provenance did not complete")
    if sealed["outputs"]["cache_manifest"]["sha256"] != sha256(cache_manifest):
        raise ValueError("source cache manifest differs from its completed provenance")
    if original.get("raw_storage") == "original_source_video":
        raise ValueError("parent must be the original lossless-clip sequence")
    hashes = {str(p): sha256(p) for p in (parent_manifest, cache_manifest, cache_provenance)}
    small_files, source_files = {}, {}
    for view in VIEWS:
        for category in ("raw", "masks", "backgrounds"):
            info = original[category][view]
            path = contained_file(parent, info["path"])
            hashes[str(path)] = info["sha256"]
            # Base SequenceReader verifies large clip bytes during the full audit.
            if category != "raw":
                if sha256(path) != info["sha256"]:
                    raise ValueError(f"changed parent {category} for {view}")
                hashes[str(path)] = info["sha256"]
                small_files[(category, view)] = path
        relative = f"medaka_202512/{original['experiment']}/{view}/{VIDEO_NAMES[view]}"
        entry = cached["files"][relative]
        source = contained_file(cache, relative)
        if source.stat().st_size != entry["bytes"]:
            raise ValueError(f"local source cache size changed for {view}")
        if entry["local_full_sha256"] != entry["source_stream_sha256"]:
            raise ValueError("source cache copy was not byte-identical")
        source_files[view] = (source, relative, entry)
        hashes[str(source)] = entry["local_full_sha256"]
        node = (
            input_root
            / "work/c01_s16_predictions/bottomup"
            / f"{original['experiment']}__{view}.npz"
        )
        if not node.is_file():
            raise FileNotFoundError(node)
        hashes[str(node)] = sha256(node)
    return original, small_files, source_files, hashes


def audit_equivalence(compact, parent):
    counts = {
        view: dict(
            images=0, masks=0, detections=0, available_empty_frames=[], unavailable_frames=[]
        )
        for view in VIEWS
    }
    if not np.array_equal(compact.frames, parent.frames):
        raise ValueError("compact and parent frame indices differ")
    for view in VIEWS:
        np.testing.assert_array_equal(compact.backgrounds[view], parent.backgrounds[view])
    left, right = iter(compact), iter(parent)
    started = time.perf_counter()
    try:
        for t in range(len(parent.frames)):
            current, original = next(left), next(right)
            if current.frame_idx != original.frame_idx:
                raise ValueError("native frame mismatch during parallel reader audit")
            for view in VIEWS:
                np.testing.assert_array_equal(current.images_bgr[view], original.images_bgr[view])
                np.testing.assert_array_equal(current.detections[view], original.detections[view])
                cmask, omask = current.masks[view], original.masks[view]
                if cmask is None or omask is None:
                    if cmask is not None or omask is not None:
                        raise ValueError("mask availability changed")
                    counts[view]["unavailable_frames"].append(current.frame_idx)
                else:
                    np.testing.assert_array_equal(cmask, omask)
                    if len(cmask) == 0:
                        counts[view]["available_empty_frames"].append(current.frame_idx)
                for key in ("images", "masks", "detections"):
                    counts[view][key] += 1
            if (t + 1) % 50 == 0 or t + 1 == len(parent.frames):
                print(
                    json.dumps({"verified_frames": t + 1, "total": len(parent.frames)}), flush=True
                )
        sentinel = object()
        if next(left, sentinel) is not sentinel or next(right, sentinel) is not sentinel:
            raise ValueError("reader yielded an undeclared extra frame")
    finally:
        left.close()
        right.close()
    return dict(
        status="verified",
        frame_count=len(parent.frames),
        images=3 * len(parent.frames),
        per_view=counts,
        decode_and_compare_seconds=time.perf_counter() - started,
        pixel_comparison="all native uint8 BGR pixels exact; both readers also verify SHA256",
        mask_comparison="all decoded instance masks exact, including missing versus empty",
        detection_comparison="all untracked SLEAP rows exact, with NaNs equal",
        background_comparison="all three decoded images exact",
        source_clip_end="iterator stops at declared window, original source continues",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-sequence", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=REPO / "work/generative_tracking_inputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.parent_sequence = args.parent_sequence.resolve(strict=True)
    args.source_cache = args.source_cache.resolve(strict=True)
    args.input_root = args.input_root.resolve(strict=True)
    args.out = args.out.resolve()
    started = time.perf_counter()
    original, small_files, sources, input_hashes = preflight(
        args.parent_sequence, args.source_cache, args.out, args.input_root
    )
    code_paths = [
        Path(__file__),
        LAB / "source_sequence.py",
        LAB / "observations.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/types.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    code_hashes = {str(p): sha256(p) for p in code_paths}
    source_stats = {view: stat_record(values[0]) for view, values in sources.items()}
    args.out.mkdir(parents=True, exist_ok=False)
    with provenance(
        "three_body_compact_source_sequence", args.out, name="compact_sequence"
    ) as record:
        record.params_from(args)
        for path, digest in input_hashes.items():
            record.input(path, path, sha256=digest)
        for path, digest in code_hashes.items():
            copy_exclusive(Path(path), args.out / "code_snapshots" / Path(path).relative_to(REPO))
            record.input("code:" + path, path, sha256=digest)
        for path in (args.parent_sequence / "manifest.json",):
            copy_exclusive(path, args.out / "parent_manifest.json")
        copy_exclusive(
            args.source_cache / "source_cache_manifest.json",
            args.out / "source_cache_manifest.json",
        )
        compact = copy.deepcopy(original)
        compact["raw_storage"] = "original_source_video"
        compact["raw_codec"] = "original source MP4; no re-encoding"
        compact["storage_qa_status"] = "pending_full_decode_qa"
        links = {}
        small_bytes = 0
        for (category, view), source in small_files.items():
            target = args.out / original[category][view]["path"]
            copy_exclusive(source, target)
            if sha256(target) != original[category][view]["sha256"]:
                raise ValueError("small-file copy checksum failed")
            small_bytes += target.stat().st_size
        (args.out / "raw").mkdir()
        for view, (source, relative, entry) in sources.items():
            target = args.out / "raw" / VIDEO_NAMES[view]
            os.link(source, target)
            if not os.path.samefile(source, target):
                raise ValueError("new source reference is not a hardlink to the verified cache")
            links[view] = dict(
                source_cache_relative_path=relative,
                descriptor_relative_path=target.relative_to(args.out).as_posix(),
                same_file=True,
                inode=target.stat().st_ino,
                hardlink_count=target.stat().st_nlink,
                bytes=entry["bytes"],
                sha256=entry["local_full_sha256"],
            )
            compact["raw"][view] = dict(
                path=target.relative_to(args.out).as_posix(),
                sha256=entry["local_full_sha256"],
                bytes=entry["bytes"],
                source_offset_frames=original["frames"][0],
                source_total_frames=SOURCE_TOTAL_FRAMES,
                storage_link="hardlink",
                source_cache_relative_path=relative,
                frame_bgr_sha256=original["raw"][view]["frame_bgr_sha256"],
            )
        compact["storage_provenance"] = dict(
            parent_manifest_sha256=input_hashes[str(args.parent_sequence / "manifest.json")],
            source_cache_manifest_sha256=input_hashes[
                str(args.source_cache / "source_cache_manifest.json")
            ],
            parent_sequence_path=str(args.parent_sequence),
            source_cache_path=str(args.source_cache),
            parent_staging_metadata=(
                "inherited detector/background metadata describes unchanged parent preparation"
            ),
            raw_video_bytes_copied=0,
            hardlinks=links,
            mask_background_bytes_copied=small_bytes,
            code_sha256=code_hashes,
        )
        manifest_path = args.out / "manifest.json"
        manifest_path.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
        init_started = time.perf_counter()
        current = SourceSequenceReader(args.out, args.input_root)
        parent = SequenceReader(args.parent_sequence, args.input_root)
        init_seconds = time.perf_counter() - init_started
        audit = audit_equivalence(current, parent)
        audit["reader_initialization_full_hash_seconds"] = init_seconds
        audit["runtime"] = dict(
            python=platform.python_version(),
            numpy=np.__version__,
            opencv=cv2.__version__,
            gpu_used=False,
        )
        # Rehash sources after decoding as well; hardlinks do not provide immutability.
        for view, (source, _, entry) in sources.items():
            if (
                sha256(source) != entry["local_full_sha256"]
                or stat_record(source) != source_stats[view]
            ):
                raise ValueError("original source bytes or metadata changed during audit")
        source_paths = {str(values[0]) for values in sources.values()}
        if any(sha256(p) != digest for p, digest in input_hashes.items() if p not in source_paths):
            raise ValueError("parent, cache metadata or untracked nodes changed during audit")
        if {p: sha256(p) for p in code_hashes} != code_hashes:
            raise ValueError("reader or audit code changed during execution")
        audit["code_and_inputs_unchanged"] = True
        audit["source_video_full_sha256_before_after_verified"] = True
        audit["elapsed_seconds"] = time.perf_counter() - started
        audit["limitations"] = [
            "Storage equivalence only; no inference rerun or biological identity validation.",
            "Hardlinks share original bytes. Access read-only; never edit them in place.",
            "Logical video file sizes are repeated directory entries, not newly copied payload.",
            "Main OpenCV decoder verified every frame in these two development windows only.",
        ]
        compact["storage_qa_status"] = "verified"
        compact["storage_provenance"]["qa"] = "decode_audit.json"
        manifest_path.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding="utf-8")
        audit["compact_manifest_sha256"] = sha256(manifest_path)
        audit_path = args.out / "decode_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        record.output("manifest", manifest_path, sha256=sha256(manifest_path))
        record.output("full_decode_audit", audit_path, sha256=sha256(audit_path))
        print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
