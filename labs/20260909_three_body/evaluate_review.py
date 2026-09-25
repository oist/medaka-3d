"""Evaluate an existing candidate against a separately saved human review JSON.

This command never runs a tracker, modifies truth, or uploads human data. The
input's draft status is retained verbatim; confirmed visual continuity is
scored as supplied, with no additional approval or adjudication step. The
shared scoring API is ``medaka3d.eval.encounter_review.score_encounter_review``.

``--sequence`` accepts the native staging directory or its manifest.json.
``--candidate`` requires sibling summary.json and experiment.provenance.json
from the completed three-body runner. All validation precedes output creation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from medaka3d.calib import load_rig
from medaka3d.eval import encounter_review
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

LAB = Path(__file__).resolve().parent
REPO = LAB.parent.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
from review_truth import ANIMALS, relative_file, sha256, validate_review  # noqa: E402


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def validate_inputs(review: dict, sequence: dict, manifest_sha: str) -> tuple:
    """Bind a human export to native source frames without filling any labels."""
    try:
        issues = validate_review(review)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"malformed human review: {exc}") from exc
    if issues:
        raise ValueError("invalid human review: " + "; ".join(issues))
    if not review.get("status") or review["status"] == "unreviewed_template_not_ground_truth":
        raise ValueError("an unreviewed input template cannot be ground truth")
    if review.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("review source manifest checksum mismatch")
    if review.get("experiment") != sequence.get("experiment"):
        raise ValueError("review and sequence experiment mismatch")
    if sequence.get("fps") != 166 or review.get("source_fps") != 166:
        raise ValueError("review and sequence must declare native 166 fps")
    if sequence.get("image_shape_hw") != [1200, 1200]:
        raise ValueError("sequence must declare native 1200-square images")
    if review.get("coordinate_system") != "native image pixels; x right, y down; mouth":
        raise ValueError("review must declare native mouth pixel coordinates")
    frames = np.asarray(sequence["frames"])
    if (frames.ndim != 1 or len(frames) < 3 or frames.dtype.kind not in "iu"
            or np.any(np.diff(frames) != 1)):
        raise ValueError("sequence frames must be consecutive native integers")
    if review.get("source_frame_range") != [int(frames[0]), int(frames[-1]) + 1]:
        raise ValueError("review source frame range mismatch")
    anchors = review["anchors"]
    expected_phases = ("entry", "after_reported_event", "exit")
    if tuple(a.get("phase") for a in anchors) != expected_phases:
        raise ValueError("review anchor phases must be entry, after_reported_event, exit")
    anchor_frames = np.asarray([a["frame"] for a in anchors])
    if anchor_frames.dtype.kind not in "iu" or not np.isin(anchor_frames, frames).all():
        raise ValueError("human anchor frames must be native integer source frames")
    points = np.full((3, 3, len(VIEWS), 2), np.nan)
    continuity = np.empty((3, 2), dtype="U12")
    for animal_index, animal in enumerate(ANIMALS):
        state = review["continuity"][animal]
        for edge, key in enumerate(("entry_to_after", "after_to_exit")):
            if state[key] not in encounter_review.CONTINUITY_STATES:
                raise ValueError("unknown human continuity state")
            continuity[animal_index, edge] = state[key]
            if state[key] == "confirmed" and any(
                a.get("frame_choice") != "human_selected" for a in anchors[edge:edge + 2]
            ):
                raise ValueError("confirmed continuity requires human-selected probe frames")
        for probe, anchor in enumerate(anchors):
            mark = anchor["animals"][animal]
            if mark["visibility"] not in ("visible", "ambiguous", "occluded", "unreviewed"):
                raise ValueError("unknown human visibility state")
            if mark["visibility"] == "visible":
                for view, xy in mark["points"].items():
                    if any(isinstance(value, bool) for value in xy):
                        raise ValueError("native pixel coordinates cannot be boolean")
                    points[probe, animal_index, VIEWS.index(view)] = xy
    return frames, anchor_frames, points, continuity


def validate_candidate(arrays: dict, sequence_frames: np.ndarray) -> None:
    required = ("frame_idx", "pose_xyz", "observed", "sources", "identity_review")
    if any(key not in arrays for key in required):
        raise ValueError(f"candidate requires {required}")
    frames = arrays["frame_idx"]
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in "iu"
            or np.any(np.diff(frames) != 1) or not np.isin(frames, sequence_frames).all()):
        raise ValueError("candidate frames must be a consecutive subset of source frames")
    count = len(frames)
    for key, shape in (("pose_xyz", (count, 3, 3, 3)), ("observed", (count, 3, 3)),
                       ("sources", (count, 3, 3, 3)), ("identity_review", (count, 3))):
        if arrays[key].shape != shape:
            raise ValueError(f"candidate {key} has an inconsistent shape")
    if any(arrays[key].dtype.kind != "b" for key in ("observed", "identity_review")):
        raise ValueError("candidate observation and review masks must be boolean")
    if arrays["sources"].dtype.kind not in "iu" or np.any(arrays["sources"] < -1):
        raise ValueError("candidate sources must contain integer rows or -1")
    if arrays["pose_xyz"].dtype.kind != "f" or np.isinf(arrays["pose_xyz"]).any():
        raise ValueError("candidate pose must be floating point with NaN for missing, not infinity")
    if np.any(arrays["observed"] & ~np.isfinite(arrays["pose_xyz"]).all(axis=-1)):
        raise ValueError("observed candidate nodes must have finite coordinates")
    support = (arrays["sources"] >= 0).sum(axis=2)
    if np.any(arrays["observed"] & (support < 2)):
        raise ValueError("observed candidate nodes must have at least two source views")


def verified_raw(sequence_path: Path, sequence: dict, anchors: np.ndarray) -> dict:
    """Check native raw provenance and decode only the three requested probes."""
    checked = {}
    indices = {int(frame): i for i, frame in enumerate(sequence["frames"])}
    for view in VIEWS:
        item = sequence["raw"][view]
        path = relative_file(sequence_path.parent, item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"raw file checksum mismatch: {view}")
        hashes = item["frame_bgr_sha256"]
        if len(hashes) != len(indices):
            raise ValueError(f"raw per-frame checksum count mismatch: {view}")
        capture = cv2.VideoCapture(str(path))
        try:
            if (not capture.isOpened() or capture.get(cv2.CAP_PROP_FRAME_COUNT) != len(indices)
                    or not np.isclose(capture.get(cv2.CAP_PROP_FPS), 166, atol=1e-6, rtol=0)):
                raise ValueError(f"native raw frame count or fps mismatch: {view}")
            for frame in anchors:
                index = indices[int(frame)]
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError(f"raw seek failed: {view} {frame}")
                ok, bgr = capture.read()
                if (not ok or bgr.shape != (1200, 1200, 3) or bgr.dtype != np.uint8
                        or capture.get(cv2.CAP_PROP_POS_FRAMES) != index + 1):
                    raise ValueError(f"native raw probe decode mismatch: {view} {frame}")
                if hashlib.sha256(bgr.tobytes()).hexdigest() != hashes[index]:
                    raise ValueError(f"raw probe pixel checksum mismatch: {view} {frame}")
        finally:
            capture.release()
        checked[path] = item["sha256"]
    return checked


def verified_calibration(input_root: Path) -> tuple:
    staging = input_root / "staging_manifest.json"
    entries = read_json(staging)["files"]
    catalog = {entry["relative"].replace("\\", "/"): entry for entry in entries}
    manifests = [key for key in catalog if key.endswith("/calibration_manifest.json")]
    if len(manifests) != 1:
        raise ValueError("input root must contain exactly one registered calibration manifest")
    checked = {staging.resolve(): sha256(staging)}

    def verify(relative: str) -> Path:
        path = relative_file(input_root, relative)
        entry = catalog.get(relative)
        if (entry is None or path.stat().st_size != entry["bytes"]
                or sha256(path) != entry["sha256"]):
            raise ValueError(f"staged calibration checksum mismatch: {relative}")
        checked[path] = entry["sha256"]
        return path

    manifest = verify(manifests[0])
    prefix = manifests[0].rsplit("/", 1)[0]
    for entry in read_json(manifest)["model_files"]:
        path = verify(f"{prefix}/{entry['path']}")
        if checked[path] != entry["sha256"]:
            raise ValueError("calibration bundle differs from its own manifest")
    return load_rig(manifest.parent), checked


def verify_run_binding(candidate: Path, sequence: dict, arrays: dict, inputs: dict) -> dict:
    """Require a completed run sidecar binding candidate bytes to these inputs."""
    summary_path = candidate.parent / "summary.json"
    provenance_path = candidate.parent / "experiment.provenance.json"
    summary, record = read_json(summary_path), read_json(provenance_path)
    if record.get("status") != "ok" or summary.get("technical_completion") is not True:
        raise ValueError("candidate run must have completed successfully before human evaluation")
    for name, path in (("candidate", candidate), ("summary", summary_path)):
        if record.get("outputs", {}).get(name, {}).get("sha256") != sha256(path):
            raise ValueError(f"completed run {name} checksum mismatch")
    if (summary.get("experiment") != sequence["experiment"] or summary.get("native_fps") != 166
            or summary.get("temporal_subsampling") is not False):
        raise ValueError("candidate run experiment or native frame declaration mismatch")
    frames = arrays["frame_idx"]
    if summary.get("frames") != [int(frames[0]), int(frames[-1]) + 1]:
        raise ValueError("candidate frame range differs from completed summary")
    if summary.get("returned_frames") != len(frames):
        raise ValueError("candidate frame count differs from completed summary")
    before, after = summary["input_sha256_before"], summary["input_sha256_after"]
    if before != after or summary.get("inputs_unchanged_during_run") is not True:
        raise ValueError("candidate source inputs changed during inference")
    # Bind by recorded content, allowing a verified input root to be relocated.
    # Every expected source digest must be recorded, not merely its file name.
    registered = set(before.values())
    if any(digest not in registered for digest in inputs.values()):
        raise ValueError("candidate run was not bound to these sequence/raw/calibration inputs")
    return {summary_path: sha256(summary_path), provenance_path: sha256(provenance_path)}


def project_mouth(rig, arrays: dict, anchor_frames: np.ndarray) -> np.ndarray:
    """Project all finite latent mouths so missing competitors cannot be hidden."""
    count = len(arrays["frame_idx"])
    projected = np.full((count, 3, len(VIEWS), 2), np.nan)
    selected = (
        np.isin(arrays["frame_idx"], anchor_frames)[:, None]
        & np.isfinite(arrays["pose_xyz"][:, :, 0]).all(axis=-1)
    )
    if not selected.any():
        return projected
    for view_index, view in enumerate(VIEWS):
        projected[:, :, view_index][selected] = rig.camera(view).project(
            arrays["pose_xyz"][:, :, 0][selected]
        )
    return projected


def evaluate(review_path: Path, sequence_path: Path, candidate_path: Path,
             input_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    sequence_path = sequence_path / "manifest.json" if sequence_path.is_dir() else sequence_path
    review_path, sequence_path, candidate_path = (
        path.resolve(strict=True) for path in (review_path, sequence_path, candidate_path)
    )
    review_bytes = review_path.read_bytes()
    review, sequence = read_json(review_path), read_json(sequence_path)
    manifest_hash = sha256(sequence_path)
    frames, anchors, points, continuity = validate_inputs(review, sequence, manifest_hash)
    with np.load(candidate_path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in
                  ("frame_idx", "pose_xyz", "observed", "sources", "identity_review")}
    validate_candidate(arrays, frames)
    consumed = {sequence_path: manifest_hash, **verified_raw(sequence_path, sequence, anchors)}
    rig, calibration = verified_calibration(input_root.resolve(strict=True))
    consumed.update(calibration)
    consumed.update(verify_run_binding(candidate_path, sequence, arrays, consumed))
    consumed.update({review_path: hashlib.sha256(review_bytes).hexdigest(),
                     candidate_path: sha256(candidate_path)})
    sources = [Path(__file__), LAB / "review_truth.py", Path(encounter_review.__file__),
               REPO / "src/medaka3d/calib.py", REPO / "src/medaka3d/types.py",
               REPO / "src/medaka3d/provenance.py"]
    code_hashes = {str(path.resolve()): sha256(path) for path in sources}
    result = encounter_review.score_encounter_review(
        arrays["frame_idx"], project_mouth(rig, arrays, anchors), arrays["observed"][:, :, 0],
        arrays["identity_review"], anchors, points, continuity,
    )
    if any(sha256(path) != digest for path, digest in consumed.items()):
        raise ValueError("evaluation input changed while it was being read")
    if any(sha256(Path(path)) != digest for path, digest in code_hashes.items()):
        raise ValueError("evaluation source changed during scoring")
    report = {
        "schema": "three_body_human_review_evaluation_v1",
        "evaluation_only_after_inference": True, "human_input_modified": False,
        "human_input_metadata": {key: review.get(key) for key in
                                 ("status", "reviewer", "reviewed_utc", "elapsed_review_seconds",
                                  "anonymous_identity_scope", "biological_naming")},
        "human_status_interpretation": "Retained as supplied; no status transition or new approval",
        "experiment": sequence["experiment"], "n_groups": 1, "view_order": list(VIEWS),
        "source_manifest_sha256": manifest_hash, "native_raw_probe_pixels_verified": 9,
        "candidate_mouth_support": "observed mouth with at least two declared source rays",
        "scoring": result, "input_sha256": {str(path): value for path, value in consumed.items()},
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    with provenance("anonymous_encounter_review", output, name="evaluation") as record:
        for path, digest in consumed.items():
            record.input(str(path), path, sha256=digest)
        record.param("match_radius_px", encounter_review.MATCH_RADIUS_PX)
        record.param("evaluation_only_after_inference", True)
        copied, scored = output / "human_review.json", output / "score.json"
        copied.write_bytes(review_bytes)
        scored.write_text(encoded, encoding="utf-8")
        record.output("human_review_copy", copied, sha256=sha256(copied))
        record.output("score", scored, sha256=sha256(scored))
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("review", "sequence", "candidate", "input-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.review, args.sequence, args.candidate, args.input_root, args.output)
    print(json.dumps({"output": str(args.output), "counts": {
        key: result["scoring"][key]["interval_counts"] for key in ("raw", "accepted")
    }}))
    return result


if __name__ == "__main__":
    main()
