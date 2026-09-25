"""Stage an actual raw-review export without relabelling any trajectory.

The CLI validates original JSON/manifest bytes against the completed raw pack.
Only the explicitly supplied review JSON may be outside the workspace (for
example Downloads). It is read-only; its exact bytes/hash are copied into the
new workspace artifact. Pack, source, code and output paths remain workspace-only.
It never opens candidate arrays, original MP4s, SLEAP arrays, a browser or GPU.
The pure match_anchor_rows API is ready for a future trusted final-run loader;
it does not claim to verify final-run files or create correction-core bundles.
Human draft status is preserved, not treated as an extra approval requirement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src"), str(Path(__file__).resolve().parent)]

import evaluate_review  # noqa: E402
import node_geometry  # noqa: E402
from prepare_long_raw_review import (  # noqa: E402
    Evidence,
    local,
    output_binding,
    require,
    sealed,
    validate_source,
)
from run_long_remaining_supervisor import final_outputs, save_json, sha256, snapshot  # noqa: E402

from medaka3d.eval import encounter_review  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

ANIMALS = ("A", "B", "C")
PHASES = ("entry", "after_reported_event", "exit")
EDGES = ("entry_to_after", "after_to_exit")
STAGE = "long_raw_review_document_validation_v1"


def document(data, label):
    """Reject duplicate keys and non-JSON NaN/Infinity before frozen validation."""
    require(isinstance(data, bytes), label + " requires original bytes")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def number(value):
        result = float(value)
        require(np.isfinite(result), "nonfinite JSON number")
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs,
                           parse_constant=constant, parse_float=number)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid " + label) from exc
    require(isinstance(value, dict), label + " must be an object")
    return value


def range_pair(value, label):
    require(isinstance(value, list) and len(value) == 2
            and all(type(x) is int for x in value) and value[0] < value[1],
            label + " must be an increasing native integer range")
    return value


@dataclass(frozen=True)
class ParsedReview:
    review_bytes: bytes
    manifest_bytes: bytes
    review: dict
    manifest: dict
    anchor_frames: np.ndarray
    human_points: np.ndarray
    continuity: np.ndarray


def parse_review(review_bytes, manifest_bytes, *, clip_range):
    """Actual HTML export schema plus source/clip binding; no label filling."""
    review, manifest = document(review_bytes, "review"), document(manifest_bytes, "manifest")
    clip = range_pair(clip_range, "clip")
    require(type(review.get("source_fps")) is int and type(manifest.get("fps")) is int,
            "native FPS must be an integer")
    require(range_pair(review.get("review_clip_range"), "review clip") == clip,
            "review belongs to a different displayed clip")
    frames = manifest.get("frames")
    require(isinstance(frames, list) and len(frames) >= 3
            and all(type(x) is int for x in frames)
            and frames == list(range(frames[0], frames[-1] + 1)), "native manifest frames differ")
    require(frames[0] <= clip[0] < clip[1] <= frames[-1] + 1, "clip outside manifest")
    require(range_pair(review.get("source_frame_range"), "review source")
            == [frames[0], frames[-1] + 1], "review source range differs")
    require(review.get("status") == "human_draft_requires_adjudication",
            "actual HTML draft export required; an empty template is not an answer")
    require(isinstance(review.get("reviewer"), str) and bool(review["reviewer"].strip()),
            "nonempty reviewer text required")
    require(review.get("anonymous_identity_scope")
            == "within this clip only; A/B/C are human entry choices", "local ABC scope changed")
    stamp = review.get("reviewed_utc")
    require(isinstance(stamp, str), "review timestamp required")
    try:
        parsed_time = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid review timestamp") from exc
    require(parsed_time.tzinfo is not None, "review timestamp needs a timezone")
    seconds = review.get("elapsed_review_seconds")
    require(type(seconds) in (int, float) and np.isfinite(seconds) and seconds >= 0,
            "nonnegative finite review elapsed seconds required")
    anchors = review.get("anchors")
    require(isinstance(anchors, list) and len(anchors) == 3, "exactly three anchors required")
    require(isinstance(review.get("continuity"), dict)
            and set(review["continuity"]) == set(ANIMALS), "continuity must use local A/B/C only")
    for phase, anchor in zip(PHASES, anchors, strict=True):
        require(isinstance(anchor, dict) and anchor.get("phase") == phase
                and type(anchor.get("frame")) is int and clip[0] <= anchor["frame"] < clip[1],
                "anchor phase/frame outside displayed clip")
        require(anchor.get("frame_choice") in
                ("human_selected", "proposed_not_human_confirmed"), "unknown frame choice")
        require(isinstance(anchor.get("animals"), dict)
                and set(anchor["animals"]) == set(ANIMALS), "anchor must use local A/B/C only")
        for mark in anchor["animals"].values():
            require(isinstance(mark, dict) and isinstance(mark.get("points"), dict),
                    "mouth mark requires a points object")
            for view, xy in mark["points"].items():
                require(view in VIEWS and isinstance(xy, list) and len(xy) == 2
                        and all(type(x) in (int, float) and np.isfinite(x) and 0 <= x < 1200
                                for x in xy), "invalid native mouth coordinates")
    _, anchor_frames, points, continuity = evaluate_review.validate_inputs(
        review, manifest, hashlib.sha256(manifest_bytes).hexdigest())
    return ParsedReview(review_bytes, manifest_bytes, review, manifest,
                        anchor_frames, points, continuity)


def read_bound(path, evidence, expected=None):
    path = evidence.bind(path, expected)
    return document(path.read_bytes(), str(path))


def read_explicit_review(path, evidence):
    """The sole external-input exception: read one caller-selected JSON file."""
    path = Path(path).expanduser().resolve(strict=True)
    require(path.is_file(), "explicit review JSON must be a regular file")
    digest = sha256(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest, "review JSON changed while read")
    require(evidence.hashes.get(path, digest) == digest, "review JSON input binding changed")
    evidence.hashes[path] = digest
    return data


def load_document(review_pack, review_path):
    """Verify pack outputs/small source seals, without original-video or pose reads."""
    pack = local(review_pack)
    evidence = Evidence()
    record = sealed(pack, "review.provenance.json", "long_original_video_raw_review", evidence)
    require(record["params"].get("all_input_source_output_sha_unchanged") is True,
            "raw pack lacks its final guard")
    for name in ("summary.json", "source_manifest.json", "truth_template.json", "index.html"):
        output_binding(record, name, pack / name, evidence)
    summary = read_bound(pack / "summary.json", evidence)
    require(summary.get("schema") == "long_raw_review_pack_v1"
            and summary.get("technical_completion") is True
            and summary.get("model_outputs_shown") is False
            and summary.get("human_truth_created") is False
            and summary.get("original_raw_storage") == "original_source_video",
            "completed original-video raw-only pack required")
    clip = range_pair(summary["review_clip_range"], "displayed clip")
    require(summary["displayed_frames"] == list(range(*clip))
            and summary["frame_count"] == clip[1] - clip[0]
            and summary["native_pixel_hashes_verified"] == 3 * (clip[1] - clip[0]),
            "pack continuous native coverage differs")
    manifest_path = local(summary["source_manifest"])
    chunk = summary["consumed_record"]
    source_seal = manifest_path.parent / "staging.provenance.json"
    for path, expected in ((manifest_path, chunk["manifest_sha256"]),
                           (source_seal, chunk["provenance_sha256"])):
        item = record["inputs"].get(str(path))
        require(item is not None and local(item["path"]) == path
                and item["sha256"] == expected, "pack source input binding differs")
        evidence.bind(path, expected)
    require(sha256(pack / "source_manifest.json") == summary["source_manifest_sha256"]
            == chunk["manifest_sha256"], "copied/original manifest differs")
    source = sealed(manifest_path.parent, "staging.provenance.json",
                    "prepare_independent_mask_chunk", evidence, verify_outputs=False)
    output_binding(source, "manifest", manifest_path, evidence)
    manifest_bytes = manifest_path.read_bytes()
    manifest = document(manifest_bytes, "source manifest")
    validate_source(manifest, chunk, source["params"]["binding"])
    require(summary["source_frame_range"] == [chunk["start"], chunk["stop"]]
            and chunk["start"] <= clip[0] < clip[1] <= chunk["stop"],
            "pack chunk scope differs")
    first, last = clip[0] - chunk["start"], clip[1] - chunk["start"]
    for view in VIEWS:
        require(summary["native_bgr_sha256"][view]
                == manifest["raw"][view]["frame_bgr_sha256"][first:last],
                "recorded native BGR coverage differs from source")
        role = source["outputs"][view + "_raw"]
        raw_path = (manifest_path.parent / manifest["raw"][view]["path"]).resolve()
        require(raw_path.is_relative_to(manifest_path.parent)
                and Path(role["path"]).resolve() == raw_path
                and role["sha256"] == manifest["raw"][view]["sha256"],
                "raw manifest/output recording differs")
        raw_input = record["inputs"].get(str(raw_path))
        require(raw_input is not None and Path(raw_input["path"]).resolve() == raw_path
                and raw_input["sha256"] == role["sha256"],
                "pack lacks recorded original raw input binding")
    template = read_bound(pack / "truth_template.json", evidence)
    require(template["source_manifest_sha256"] == chunk["manifest_sha256"]
            and template["review_clip_range"] == clip, "pack template source/clip differs")
    review_bytes = read_explicit_review(review_path, evidence)
    parsed = parse_review(review_bytes, manifest_bytes, clip_range=clip)
    evidence.final_guard()
    return parsed, summary, evidence


def _json_array(value):
    array = np.asarray(value)
    return np.where(np.isfinite(array), array, None).tolist()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([value.dtype.str, list(value.shape)]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def match_anchor_rows(parsed, arrays, rig, points_by_frame, *, candidate_sha256,
                      calibration_sha256, projection_origin_sha256):
    """Pure future-loader API: all latent slots compete before measured support.

    arrays contains exactly the THREE selected final-candidate anchor rows, in
    native order, with pose/source/observed/review unchanged. Digests are scope
    declarations here, not evidence of file verification. Caller must bind the
    completed final candidate, original chunk/raw, SL and calibration first.
    No dense correction, core-compatible constraint bundle or ID output is made.
    """
    original = document(parsed.review_bytes, "review")
    checked = parse_review(parsed.review_bytes, parsed.manifest_bytes,
                           clip_range=original["review_clip_range"])
    require(parsed.review == checked.review and parsed.manifest == checked.manifest
            and all(array_sha256(getattr(parsed, key)) == array_sha256(getattr(checked, key))
                    for key in ("anchor_frames", "human_points", "continuity")),
            "parsed review fields differ from original bytes")
    parsed = checked
    require(isinstance(calibration_sha256, dict) and calibration_sha256,
            "calibration digest mapping required")
    for digest in (candidate_sha256, projection_origin_sha256, *calibration_sha256.values()):
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "invalid source digest")
    frames = np.asarray(arrays["frame_idx"])
    require(frames.shape == (3,) and frames.dtype.kind in "iu"
            and np.array_equal(frames, parsed.anchor_frames), "exact selected native rows required")
    for key, shape, kind in (("pose_xyz", (3, 3, 3, 3), "f"),
                             ("sources", (3, 3, 3, 3), "iu"),
                             ("observed", (3, 3, 3), "b"),
                             ("identity_review", (3, 3), "b")):
        require(isinstance(arrays[key], np.ndarray), "numpy anchor array required: " + key)
        value = arrays[key]
        require(value.shape == shape and value.dtype.kind in kind, "anchor state schema: " + key)
    pose, source = arrays["pose_xyz"], arrays["sources"]
    require(not np.isinf(pose).any() and np.all(source >= -1)
            and np.all(source <= np.iinfo(np.int32).max), "invalid latent pose/source rows")
    require(not np.any(arrays["observed"] & ~np.isfinite(pose).all(axis=-1)),
            "observed node is nonfinite")
    for fish in range(3):
        for other in range(fish):
            require(not np.any((source[:, fish] >= 0) & (source[:, fish] == source[:, other])),
                    "assigned source rows are not exclusive")
    projected = evaluate_review.project_mouth(rig, arrays, frames)
    # Spatial matching sees unsupported/reviewed competitors, never filtered candidates.
    spatial = encounter_review.score_encounter_review(
        frames, projected, np.ones((3, 3), bool), np.zeros((3, 3), bool),
        frames, parsed.human_points, parsed.continuity)["raw"]["anchors"]
    support = []
    for t, frame in enumerate(frames):
        points = points_by_frame[int(frame)]
        require(set(points) == set(VIEWS), "all native SL views required")
        for v, view in enumerate(VIEWS):
            uv = np.asarray(points[view])
            require(uv.ndim == 3 and uv.shape[1:] == (3, 2) and uv.dtype.kind == "f"
                    and not np.isinf(uv).any(), "SL frame/node schema differs")
            for fish, node in np.argwhere(source[t, :, v] >= 0):
                row = int(source[t, fish, v, node])
                require(row < len(uv) and np.isfinite(uv[row, node]).all(),
                        "assigned SL row is unavailable")
        rays = node_geometry.RayObservations.from_points(rig, points)
        supported, observed, distance = node_geometry.supported_nodes(
            pose[t], source[t], rays, rig, maximum_ray_mm=2., maximum_pixel_error=12.)
        require(np.array_equal(supported, source[t])
                and np.array_equal(observed, arrays["observed"][t]),
                "saved source/observed differs from native SL 2mm/12px support")
        error = np.full((3, 3, 3), np.nan)
        for v, view in enumerate(VIEWS):
            uv = rig.camera(view).project(pose[t])
            for fish, node in np.argwhere(source[t, :, v] >= 0):
                error[fish, v, node] = np.linalg.norm(
                    uv[fish, node] - points[view][source[t, fish, v, node], node])
        support.append(dict(frame=int(frame), source_rows=source[t].tolist(),
                            observed=observed.tolist(), ray_distance_mm=_json_array(
                                np.where(source[t] >= 0, distance, np.nan)),
                            reprojection_error_px=_json_array(error),
                            mouth_direct_view_count=(source[t, :, :, 0] >= 0).sum(axis=1).tolist(),
                            identity_review=arrays["identity_review"][t].tolist()))
    scored = encounter_review.score_encounter_review(
        frames, projected, arrays["observed"][:, :, 0], arrays["identity_review"],
        frames, parsed.human_points, parsed.continuity)
    endpoints = []
    for t, frame in enumerate(frames):
        distances = np.linalg.norm(parsed.human_points[t, :, None] - projected[t, None], axis=-1)
        for animal, label in enumerate(ANIMALS):
            raw = scored["raw"]["anchors"][t]["animals"][animal]
            endpoints.append(dict(frame=int(frame), phase=PHASES[t], anonymous_label=label,
                                  spatial_match=spatial[t]["animals"][animal], raw_match=raw,
                                  accepted_match=scored["accepted"]["anchors"][t]["animals"][animal],
                                  all_slot_error_px_by_view=_json_array(distances[animal])))
    links = []
    for animal, label in enumerate(ANIMALS):
        for edge, name in enumerate(EDGES):
            left, right = [endpoints[p * 3 + animal]["raw_match"]["slot"]
                           for p in (edge, edge + 1)]
            state = str(parsed.continuity[animal, edge])
            status = "human_unconfirmed" if state != "confirmed" else (
                "endpoint_unresolved" if left is None or right is None else "staged_endpoint_link")
            links.append(dict(anonymous_label=label, edge=name, human_status=state, status=status,
                              start_frame=int(frames[edge]), stop_frame=int(frames[edge + 1]),
                              left_raw_slot=left, right_raw_slot=right,
                              interior_identity_certified=False, raw_review_cleared=False))
    return dict(
        schema="long_review_anchor_mapping_prototype_v1", endpoints=endpoints, links=links,
        support=support, projected_all_latent_mouths=_json_array(projected),
        candidate_sha256=candidate_sha256, calibration_sha256=copy.deepcopy(calibration_sha256),
        projection_origin_sha256=projection_origin_sha256,
        anchor_array_sha256={k: array_sha256(arrays[k]) for k in
                             ("frame_idx", "pose_xyz", "sources", "observed", "identity_review")},
        projected_anchor_array_sha256=array_sha256(projected),
        original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
        original_manifest_sha256=hashlib.sha256(parsed.manifest_bytes).hexdigest(),
        requested_endpoints=9, requested_links=6, final_candidate_files_verified_here=False,
        production_correction_ready=False, original_review_flags_preserved=True,
        biological_naming_preserved_not_applied=copy.deepcopy(parsed.review.get("biological_naming")),
        dense_trajectory_modified=False, global_identity_assigned=False,
        scope="Pure staged local correspondence. A trusted final-run/raw/SL/calibration loader "
              "and explicit full-scope binding are still required before any correction.")


def code_hashes():
    paths = {Path(__file__).resolve(), LAB / "review_scene.html",
             LAB / "review_truth.html", Path(__file__).parent / "LONG_REVIEW_IMPORT_DESIGN.md"}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(REPO) and ".venv" not in path.parts:
                paths.add(path)
    return {str(path): sha256(path) for path in sorted(paths)}


def stage_document(review_pack, review_path, output):
    output = Path(output).resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    codes = code_hashes()
    parsed, pack_summary, evidence = load_document(review_pack, review_path)
    for path, digest in codes.items():
        evidence.bind(path, digest)
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="import") as record:
        for path, digest in evidence.hashes.items():
            record.input(str(path), path, sha256=digest)
        snapshot(codes, output, record)
        try:
            for name, data in (("original_review.json", parsed.review_bytes),
                               ("original_source_manifest.json", parsed.manifest_bytes)):
                path = output / name
                with path.open("xb") as stream:
                    stream.write(data)
                record.output(name, path, sha256=sha256(path))
            summary = dict(
                schema=STAGE, status="document_validated_final_candidate_pending",
                review_pack=str(local(review_pack)), review_sha256=hashlib.sha256(
                    parsed.review_bytes).hexdigest(), source_manifest_sha256=hashlib.sha256(
                    parsed.manifest_bytes).hexdigest(),
                original_review_path=str(Path(review_path).expanduser().resolve(strict=True)),
                external_review_written=False,
                source_frame_range=parsed.review["source_frame_range"],
                review_clip_range=parsed.review["review_clip_range"],
                actual_anchor_frames=parsed.anchor_frames.tolist(),
                original_proposed_anchor_frames=pack_summary["probe_frames"],
                human_status_preserved=parsed.review["status"],
                confirmed_requested_links=int(np.sum(parsed.continuity == "confirmed")),
                requested_links=6, local_labels=list(ANIMALS),
                candidate_mapping=None, final_candidate_verified=False,
                original_raw_files_rehashed=False, original_images_decoded=0,
                prior_pack_native_pixel_provenance_checked=True,
                tracking_or_correction_performed=False, production_correction_ready=False,
                note="A valid human draft is an input assertion, not independent identity truth. "
                     "Original answers remain unchanged; final-run matching is still pending.")
            save_json(output / "summary.json", summary, record, "summary")
        finally:
            evidence.final_guard()
            require(code_hashes() == codes, "import source set changed")
            final_outputs(record)
        record.param("all_opened_input_source_snapshot_output_sha_unchanged", True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-pack", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    stage_document(args.review_pack, args.review, args.output)


if __name__ == "__main__":
    main()
