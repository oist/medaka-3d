"""Build a model-blind raw-video review pack; never generate identity truth.

Input: the native FFV1 staging manifest from stage_native.py. Only ``frames``
and ``raw`` are read. Masks, poses, track slots and previous annotations are
deliberately absent from this interface. Human JSON exports are drafts until
independently adjudicated; this command neither scores nor uploads them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

VIEWS = ("top_cam", "side1_cam", "side2_cam")
ANIMALS = ("A", "B", "C")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("raw path must remain inside the staged input directory")
    return path


def make_template(frames: list[int], manifest_sha: str, probes: list[int]) -> dict:
    if len(frames) < 3 or frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("review input must contain at least three consecutive source frames")
    if len(probes) != 3 or probes != sorted(set(probes)) or any(f not in frames for f in probes):
        raise ValueError("three distinct chronological probe frames must occur in input")
    return {
        "schema": "raw_encounter_review_v1",
        "status": "unreviewed_template_not_ground_truth",
        "experiment": "20251225_K_1_2_FCMC",
        "source_manifest_sha256": manifest_sha,
        "source_frame_range": [frames[0], frames[-1] + 1],
        "source_fps": 166,
        "reviewer": None,
        "reviewed_utc": None,
        "elapsed_review_seconds": None,
        "model_outputs_shown": False,
        "coordinate_system": "native image pixels; x right, y down; mouth",
        "anonymous_identity_scope": "within this clip only; A/B/C are human entry choices",
        "evidence_rule": (
            "direct visual continuity by scrubbing is permitted; courtship role, "
            "model slot, geometry and automated suggestions are not identity evidence"
        ),
        "biological_naming": {
            animal: {"sex": "unknown", "physical_name": None, "evidence": None}
            for animal in ANIMALS
        },
        "anchors": [
            {
                "phase": phase,
                "frame": frame,
                "frame_choice": "proposed_not_human_confirmed",
                "animals": {
                    animal: {"points": {}, "visibility": "unreviewed"} for animal in ANIMALS
                },
            }
            for phase, frame in zip(("entry", "after_reported_event", "exit"), probes, strict=True)
        ],
        "continuity": {
            animal: {
                "entry_to_after": "unreviewed",
                "after_to_exit": "unreviewed",
                "basis": "unreviewed",
                "notes": "",
            }
            for animal in ANIMALS
        },
        "contact_occurrence": "not_assumed_from_legacy_swap_report",
        "notes": "",
    }


def validate_review(review: dict) -> list[str]:
    """Structural checks only, never certify that a human correspondence is true."""
    issues: list[str] = []
    if review.get("schema") != "raw_encounter_review_v1":
        issues.append("unexpected schema")
    if review.get("model_outputs_shown") is not False:
        issues.append("model-blind status is not confirmed")
    if not review.get("reviewer"):
        issues.append("reviewer is missing")
    anchors = review.get("anchors", [])
    if len(anchors) != 3:
        return [*issues, "exactly three anchors are required"]
    frames = [a["frame"] for a in anchors]
    start, stop = review["source_frame_range"]
    if frames != sorted(set(frames)) or not all(start <= f < stop for f in frames):
        issues.append("anchor frames must be distinct, chronological and in range")
    for animal in ANIMALS:
        state = review["continuity"][animal]
        for index, edge in enumerate(("entry_to_after", "after_to_exit")):
            if state[edge] == "confirmed":
                if state["basis"] not in ("direct_visual_continuity", "morphology_and_continuity"):
                    issues.append(f"{animal} {edge}: unsupported confirmation basis")
                for anchor in anchors[index : index + 2]:
                    mark = anchor["animals"][animal]
                    if not mark["points"] or mark["visibility"] != "visible":
                        issues.append(f"{animal} {edge}: missing visible mouth mark")
        for anchor in anchors:
            for view, xy in anchor["animals"][animal]["points"].items():
                if view not in VIEWS or len(xy) != 2 or not all(np.isfinite(xy)):
                    issues.append(f"{animal}: invalid native pixel mark")
                elif not all(0 <= value < 1200 for value in xy):
                    issues.append(f"{animal}: native pixel mark outside 1200-square image")
    return issues


def build_pack(manifest_path: Path, out: Path, probes: list[int]) -> dict:
    if out.exists():
        raise FileExistsError(out)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for a browser-playable H.264 review clip")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = [int(frame) for frame in manifest["frames"]]
    template = make_template(frames, sha256(manifest_path), probes)
    paths = {}
    for view in VIEWS:
        item = manifest["raw"][view]
        paths[view] = relative_file(manifest_path.parent, item["path"])
        if sha256(paths[view]) != item["sha256"]:
            raise ValueError(f"raw file checksum mismatch: {view}")
        if len(item["frame_bgr_sha256"]) != len(frames):
            raise ValueError(f"raw per-frame checksums missing: {view}")
    out.mkdir(parents=True)
    (out / "frames").mkdir()
    captures = {view: cv2.VideoCapture(str(path)) for view, path in paths.items()}
    intermediate_path = out / "raw_three_views_mpeg4_intermediate.mp4"
    video_path = out / "raw_three_views_slow.mp4"
    writer = cv2.VideoWriter(
        str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1800, 636)
    )
    if not writer.isOpened():
        raise RuntimeError("MP4 review writer is unavailable")
    checksums = {}
    try:
        for index, frame in enumerate(frames):
            montage = np.zeros((636, 1800, 3), dtype=np.uint8)
            for column, view in enumerate(VIEWS):
                ok, bgr = captures[view].read()
                if not ok or bgr.shape != (1200, 1200, 3):
                    raise ValueError(f"native frame missing or dimensions changed: {view} {frame}")
                expected = manifest["raw"][view]["frame_bgr_sha256"][index]
                if hashlib.sha256(bgr.tobytes()).hexdigest() != expected:
                    raise ValueError(f"decoded raw pixel checksum mismatch: {view} {frame}")
                path = out / "frames" / f"f{frame}_{view}.jpg"
                if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                    raise RuntimeError(f"cannot write {path}")
                checksums[str(path.relative_to(out))] = sha256(path)
                if frame in probes:
                    native = out / f"f{frame}_{view}.png"
                    if not cv2.imwrite(str(native), bgr):
                        raise RuntimeError(f"cannot write {native}")
                    checksums[native.name] = sha256(native)
                montage[36:, column * 600 : (column + 1) * 600] = cv2.resize(bgr, (600, 600))
                cv2.putText(
                    montage,
                    f"{view} source {frame}",
                    (column * 600 + 12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(montage)
        if any(capture.read()[0] for capture in captures.values()):
            raise ValueError("staged raw contains undeclared trailing frames")
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(intermediate_path),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    decoded = cv2.VideoCapture(str(video_path))
    decoded_count = 0
    try:
        while True:
            ok, _ = decoded.read()
            if not ok:
                break
            decoded_count += 1
    finally:
        decoded.release()
    if decoded_count != len(frames):
        raise RuntimeError("review video frame-count QA failed")
    html_source = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    html = (
        html_source.replace("__TRUTH__", json.dumps(template, ensure_ascii=False))
        .replace("__FRAMES__", json.dumps(frames))
        .replace("__VIEWS__", json.dumps(VIEWS))
        .replace("__LAST__", str(len(frames) - 1))
    )
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "truth_template.json").write_text(
        json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    provenance = {
        "status": "ok",
        "created_utc": datetime.now(UTC).isoformat(),
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "raw": {view: {"path": str(paths[view]), "sha256": sha256(paths[view])} for view in VIEWS},
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
            "html_sha256": sha256(Path(__file__).with_suffix(".html")),
        },
        "frame_range": [frames[0], frames[-1] + 1],
        "frame_count": len(frames),
        "probe_frames": probes,
        "native_pixel_hashes_verified": len(frames) * len(VIEWS),
        "review_video": {
            "path": video_path.name,
            "sha256": sha256(video_path),
            "decoded_frame_count": decoded_count,
            "playback_fps": 20,
            "source_fps": 166,
            "all_input_frames_included": True,
            "codec": "H.264 libx264 CRF17 yuv420p (from retained MPEG4 intermediate)",
        },
        "preview_encoding": "native-size JPEG quality 96; three default probes also lossless PNG",
        "output_hashes": checksums,
        "model_outputs_used": False,
        "human_truth_created": False,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probes", type=int, nargs=3, required=True)
    args = parser.parse_args()
    result = build_pack(args.manifest, args.out, args.probes)
    print(
        json.dumps(
            {"status": result["status"], "frames": result["frame_count"], "out": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
