"""Stage the fixed K_1_2 comparison inputs from a share or mounted source root.

Only the 18 explicitly named detection, calibration, baseline and annotation
files are copied. Raw recordings are read separately by benchmark.py. All
sources are checked before the output directory is created; an interrupted
copy remains in place with failure provenance and must not be reused as a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from medaka3d.provenance import provenance

EXPERIMENT = "20251225_K_1_2_FCMC"
BASELINE = f"work/c34_microflip/{EXPERIMENT}__chain-c28-c33-c30v4-c34"
CALIBRATION = (
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
RELATIVE_INPUTS = (
    "assets/MANIFEST.json",
    *(f"{BASELINE}/C34-label__{EXPERIMENT}__lam{value}.npz"
      for value in ("0", "0.1", "0.25", "0.5", "0.767", "2", "8")),
    *(f"work/c01_s16_predictions/bottomup/{EXPERIMENT}__{view}.npz"
      for view in ("top_cam", "side1_cam", "side2_cam")),
    f"{CALIBRATION}/calibration_manifest.json",
    *(f"{CALIBRATION}/cam{camera}_bundle_refractive.json" for camera in range(3)),
    f"assets/groundtruth/fish_sex_gt/{EXPERIMENT}/female_sex_gt.csv",
    f"assets/groundtruth/swap_event_gt/{EXPERIMENT}/swap_events.csv",
    f"assets/groundtruth/swap_event_gt/{EXPERIMENT}/swap_events.json",
)


def _relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or "\\" in value or ":" in value or str(relative) != value):
        raise ValueError(f"input path must be a normalized portable relative path: {value!r}")
    return relative


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> tuple[str, str, int]:
    """Hash source bytes as read, then independently verify the written copy."""
    before = source.stat()
    source_digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            source_digest.update(block)
            writer.write(block)
            copied += len(block)
    after = source.stat()
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or copied != before.st_size):
        raise RuntimeError(f"source changed during staging: {source}")
    local_digest = sha256(destination)
    if local_digest != source_digest.hexdigest() or destination.stat().st_size != copied:
        raise RuntimeError(f"staged copy differs from source bytes: {destination}")
    return source_digest.hexdigest(), local_digest, copied


def stage_inputs(source_root: Path, out: Path) -> Path:
    """Create a new, verified staging tree and return its manifest path.

    The file allowlist is fixed in this module. Source symlinks/junctions that
    escape the requested root are rejected. Destination writes use exclusive
    creation, including when another process races this operation.
    """
    source_root = Path(source_root).resolve(strict=True)
    if not source_root.is_dir():
        raise NotADirectoryError(f"source root is not a directory: {source_root}")
    requested_out = Path(out)
    if requested_out.exists() or requested_out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing staging path: {requested_out}")
    out = requested_out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {out}")
    sources = []
    missing = []
    for value in RELATIVE_INPUTS:
        relative = _relative_path(value)
        candidate = source_root.joinpath(*relative.parts)
        try:
            source = candidate.resolve(strict=True)
        except FileNotFoundError:
            missing.append(value)
            continue
        if not source.is_relative_to(source_root):
            raise ValueError(f"source resolves outside requested root: {candidate} -> {source}")
        if not source.is_file():
            missing.append(value)
            continue
        sources.append((relative, source))
    if missing:
        raise FileNotFoundError("required staging inputs are missing:\n" + "\n".join(missing))
    if len({relative for relative, _ in sources}) != len(sources):
        raise ValueError("fixed staging input list contains duplicate relative paths")

    # No output or parent directories are created until the entire preflight passes.
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "staging_manifest.json"
    with provenance("stage_generative_inputs", out, name="staging") as record:
        record.param("experiment", EXPERIMENT)
        record.param("source_root", str(source_root))
        record.param("out", str(out))
        record.param("relative_inputs", list(RELATIVE_INPUTS))
        record.param("copies_raw_videos", False)
        record.input("staging_code", Path(__file__), sha256=sha256(Path(__file__)))
        entries = []
        for relative, source in sources:
            destination = out.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A freshly created tree has no links, but check again before writing
            # so an unexpected pre-existing link cannot redirect this operation.
            if not destination.resolve().is_relative_to(out):
                raise ValueError(f"destination escapes staging root: {destination}")
            if source.resolve(strict=True) != source or not source.is_relative_to(source_root):
                raise ValueError(f"source path changed after preflight: {source}")
            source_hash, local_hash, size = _copy_verified(source, destination)
            entries.append({
                "relative": relative.as_posix(),
                "source": str(source), "local": str(destination),
                "source_sha256": source_hash, "sha256": local_hash,
                "source_bytes": size, "bytes": size,
            })
            record.input(relative.as_posix(), source, sha256=source_hash)
            record.output(relative.as_posix(), destination, sha256=local_hash)
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "experiment": EXPERIMENT,
            "source_root": str(source_root),
            "file_policy": "fixed 18-file K_1_2 comparison allowlist; raw videos excluded",
            "files": entries,
        }
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        record.output("staging_manifest", manifest_path, sha256=sha256(manifest_path))
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="source medaka root containing assets/, work/ and calibration/")
    parser.add_argument("--out", type=Path, required=True,
                        help="new local staging directory; existing paths are refused")
    args = parser.parse_args(argv)
    manifest = stage_inputs(args.source_root, args.out)
    print(json.dumps({"manifest": str(manifest), "files": len(RELATIVE_INPUTS)}, indent=2))


if __name__ == "__main__":
    main()
