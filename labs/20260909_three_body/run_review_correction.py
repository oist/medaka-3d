"""Export human correction separately from a completed, immutable geometry run.

Only explicitly supplied human review JSON files enter this stage. An empty
review list is a meaningful control, never an instruction to invent annotations.
The existing review validator rejects unreviewed templates and preserves human
status, visibility, naming and confirmation basis without filling any fields.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parent.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

import evaluate_review  # noqa: E402
import identity_inputs as verified  # noqa: E402
import review_constraints  # noqa: E402
from observations import CachedNodes  # noqa: E402

from medaka3d import calib  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

base = verified.base
STAGE = "three_body_human_review_dense_correction"


@dataclass(frozen=True)
class PreparedReviews:
    """Verified files and extracted endpoint constraints; no corrected identity yet."""

    run: object
    geometry: dict[str, np.ndarray]
    review_paths: tuple[Path, ...]
    review_bytes: tuple[bytes, ...]
    bundles: tuple[dict, ...]
    combined: dict | None
    verified_hashes: dict[str, str]
    metadata: dict


def _merge(hashes, path, digest):
    name = str(Path(path).resolve(strict=True))
    if name in hashes and hashes[name] != digest:
        raise ValueError("one input/source path has conflicting recorded digests")
    hashes[name] = digest


def assert_unchanged(hashes):
    for path, digest in hashes.items():
        if base.sha256(path) != digest:
            raise RuntimeError(f"review-correction evidence changed: {path}")


def code_paths():
    """Include loaded local transitive helpers before any input is verified."""
    paths = {
        Path(__file__).resolve(),
        LAB / "HUMAN_CORRECTION_PROTOCOL.md",
        LAB / "DENSE_REVIEW_CORRECTION_PROTOCOL.md",
    }
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            path = Path(origin).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src/medaka3d")
            ):
                paths.add(path)
    return sorted(paths)


def summarize_review_workload(documents, contents):
    """Count final saved marks/decisions once per exact document, not UI actions."""
    unique = {}
    for file_index, (review, content) in enumerate(zip(documents, contents, strict=True)):
        digest = hashlib.sha256(content).hexdigest()
        if digest in unique:
            unique[digest]["supplied_file_indices"].append(file_index)
            continue
        elapsed = review.get("elapsed_review_seconds")
        known_elapsed = type(elapsed) in (int, float) and np.isfinite(elapsed) and elapsed >= 0
        marks = [mark for anchor in review["anchors"] for mark in anchor["animals"].values()]
        edges = [
            dict(anonymous_label=animal, edge=edge)
            for animal, states in review["continuity"].items()
            for edge in ("entry_to_after", "after_to_exit")
            if states[edge] == "confirmed"
        ]
        unique[digest] = dict(
            review_sha256=digest,
            supplied_file_indices=[file_index],
            elapsed_review_seconds=elapsed,
            elapsed_value_is_known_nonnegative_seconds=bool(known_elapsed),
            declared_review_clip_range=review.get("review_clip_range"),
            source_frame_range=review.get("source_frame_range"),
            saved_mouth_coordinate_pairs=sum(len(mark["points"]) for mark in marks),
            visible_mouth_coordinate_pairs=sum(
                len(mark["points"]) for mark in marks if mark["visibility"] == "visible"
            ),
            confirmed_edge_count=len(edges),
            confirmed_edges=edges,
        )
    rows = list(unique.values())
    known = [
        row["elapsed_review_seconds"]
        for row in rows
        if row["elapsed_value_is_known_nonnegative_seconds"]
    ]
    return dict(
        unique_documents=rows,
        supplied_files=len(contents),
        distinct_document_sha256=len(rows),
        repeated_document_files=len(contents) - len(rows),
        saved_mouth_coordinate_pairs=sum(row["saved_mouth_coordinate_pairs"] for row in rows),
        visible_mouth_coordinate_pairs=sum(row["visible_mouth_coordinate_pairs"] for row in rows),
        confirmed_edge_count=sum(row["confirmed_edge_count"] for row in rows),
        known_elapsed_documents=len(known),
        unknown_or_invalid_elapsed_documents=len(rows) - len(known),
        known_elapsed_partial_seconds=sum(known) if known else None,
        elapsed_total_seconds=sum(known) if len(known) == len(rows) else None,
        click_count_interpretation=(
            "Saved final native mouth marks, not counted UI clicks, revisions or deleted marks."
        ),
        confirmed_edges_are_not_resolved_global_id_count=True,
        elapsed_is_user_declared_not_independently_measured=True,
        duplicate_policy="Exact review bytes/SHA counted once; no content-based fuzzy merge.",
    )


def load_review_inputs(geometry_run, review_paths):
    """Reuse no-GT completion checks and validate actual review probe raw pixels.

    Producer summary bytes remain checksum evidence only. Neither their evaluation
    metrics nor any post-inference GT/reference assets are opened by the loader.
    Project every latent competitor at every review anchor, including unobserved
    mouths, so an unsupported near-overlap cannot be hidden from raw matching.
    """
    run = verified._verify_run(geometry_run, appearance=False)
    hashes = dict(run.hashes)
    geometry = verified._npz(run.root / "candidate.npz")
    if set(geometry) != verified.GEOMETRY_ARRAYS:
        raise ValueError("completed persistent geometry requires exactly 21 arrays")
    verified.persistent_runner.validate_arm_arrays(geometry, run.frames)
    nodes = CachedNodes(run.input_root, base.EXPERIMENT, run.frames)
    base.validate_result(geometry, run.frames, nodes)
    manifest_path = run.sequence / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != hashes[str(manifest_path)]:
        raise ValueError("sequence manifest changed after completion verification")
    calibration_root = run.root / "calibration_snapshot"
    calibration_hashes = {}
    for name in (
        "calibration_manifest.json",
        *(f"cam{i}_bundle_refractive.json" for i in range(3)),
    ):
        path = calibration_root / name
        if str(path) not in hashes or base.sha256(path) != hashes[str(path)]:
            raise ValueError("actual calibration read lacks a verified snapshot binding")
        calibration_hashes[name] = hashes[str(path)]
    projector_path = Path(calib.__file__).resolve()
    projector_sha = base.sha256(projector_path)
    if run.code.get(projector_path.relative_to(REPO).as_posix()) != projector_sha:
        raise ValueError("current refractive projector differs from the frozen geometry source")
    _merge(hashes, projector_path, projector_sha)

    paths = tuple(verified._path(path) for path in review_paths)
    if len(set(paths)) != len(paths):
        raise ValueError("each explicitly supplied review file must appear only once")
    documents, contents, anchor_lists = [], [], []
    for path in paths:
        content = path.read_bytes()
        review, digest = review_constraints._document(content, "review")
        _, anchors, _, _ = evaluate_review.validate_inputs(
            review, run.manifest, hashes[str(manifest_path)]
        )
        _merge(hashes, path, digest)
        documents.append(review)
        contents.append(content)
        anchor_lists.append(anchors)
    anchors = (
        np.unique(np.concatenate(anchor_lists)) if anchor_lists else np.empty(0, dtype=np.int64)
    )
    if len(anchors):
        for path, digest in evaluate_review.verified_raw(
            manifest_path, run.manifest, anchors
        ).items():
            _merge(hashes, path, digest)
        rig = calib.load_rig(calibration_root)
        projected = evaluate_review.project_mouth(rig, geometry, anchors)
    else:
        projected = None
    bundles = tuple(
        review_constraints.extract_review_constraints(
            content,
            manifest_bytes,
            geometry,
            projected,
            candidate_sha256=hashes[str(run.root / "candidate.npz")],
            calibration_sha256=calibration_hashes,
            projection_origin_sha256=projector_sha,
        )
        for content in contents
    )
    combined = review_constraints.combine_constraints(bundles) if bundles else None
    metadata = dict(
        geometry_run=str(run.root),
        native_frame_range=[int(run.frames[0]), int(run.frames[-1] + 1)],
        native_frames=len(run.frames),
        candidate_sha256=hashes[str(run.root / "candidate.npz")],
        source_manifest_sha256=hashes[str(manifest_path)],
        calibration_sha256=calibration_hashes,
        projection_origin_sha256=projector_sha,
        supplied_review_files=len(paths),
        distinct_review_documents=len({hashlib.sha256(value).hexdigest() for value in contents}),
        reviewed_anchor_frames=anchors.tolist(),
        native_raw_probe_pixels_verified=3 * len(anchors),
        human_metadata_preserved=[
            {
                key: item.get(key)
                for key in (
                    "status",
                    "reviewer",
                    "reviewed_utc",
                    "elapsed_review_seconds",
                    "review_clip_range",
                    "source_frame_range",
                    "anonymous_identity_scope",
                    "biological_naming",
                )
            }
            for item in documents
        ],
        candidate_calibration_files_verified_by_cli=True,
        post_inference_assets_opened=False,
        producer_summary_deserialized=False,
        review_template_created=False,
        human_input_modified=False,
        biological_naming_applied=False,
        empty_review_list_is_no_human_evidence=not paths,
        review_workload=summarize_review_workload(documents, contents),
    )
    assert_unchanged(hashes)
    return PreparedReviews(
        run, geometry, paths, tuple(contents), bundles, combined, hashes, metadata
    )


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def snapshot_code(output, paths, digests, record):
    snapshots = {}
    for path in paths:
        target = output / "code_snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digest = digests[str(path)]
        if base.sha256(target) != digest:
            raise RuntimeError("source changed during review-correction snapshot")
        snapshots[str(target)] = digest
        record.output(str(target.relative_to(output)), target, sha256=digest)
    return snapshots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument(
        "--review",
        type=Path,
        action="append",
        default=[],
        help="Explicit saved human JSON; repeat for more documents, omit for zero-review control.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def validate_core_result(result, geometry, review_count):
    """Check export conservation independently of the core's review propagation."""
    if not isinstance(result, dict) or set(result) != {"arrays", "metadata"}:
        raise ValueError("dense core must return exactly arrays and metadata")
    arrays, metadata = result["arrays"], result["metadata"]
    if not isinstance(arrays, dict) or not arrays or not isinstance(metadata, dict):
        raise ValueError("dense core requires a nonempty array mapping and JSON metadata")
    if any(
        not isinstance(key, str) or not isinstance(value, np.ndarray) or value.dtype.hasobject
        for key, value in arrays.items()
    ):
        raise ValueError("dense output requires string keys and non-object NumPy arrays")
    json.dumps(metadata, allow_nan=False)
    for name, shape, kind in (
        ("frame_idx", geometry["frame_idx"].shape, "iu"),
        ("corrected_identity_to_raw_slot", (len(geometry["frame_idx"]), 3), "iu"),
        ("corrected_pose_xyz", geometry["pose_xyz"].shape, "f"),
        ("corrected_sources", geometry["sources"].shape, "iu"),
        ("corrected_observed", geometry["observed"].shape, "b"),
        ("corrected_node_xyz", geometry["pose_xyz"].shape, "f"),
        ("corrected_xyz", geometry["accepted_xyz"].shape, "f"),
        ("identity_origin", geometry["identity_review"].shape, "iu"),
        ("claim_index", geometry["identity_review"].shape, "iu"),
        ("tracklet_index", geometry["identity_review"].shape, "iu"),
        ("identity_conflict", geometry["identity_review"].shape, "b"),
        ("prior_raw_review_for_identity", geometry["identity_review"].shape, "b"),
    ):
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype.kind not in kind:
            raise ValueError(f"dense output has an invalid {name} schema")
    if not np.array_equal(arrays["frame_idx"], geometry["frame_idx"]):
        raise ValueError("corrected frame order differs from the original native frames")
    mapping = arrays["corrected_identity_to_raw_slot"]
    if np.any((mapping < -1) | (mapping >= 3)):
        raise ValueError("corrected mapping must be a raw slot or -1")
    held = mapping >= 0
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if np.any(held[:, a] & held[:, b] & (mapping[:, a] == mapping[:, b])):
            raise ValueError("corrected identities cannot use the same raw slot at one frame")
    times, identities = np.nonzero(held)
    slots = mapping[times, identities]
    for new, old in (
        ("corrected_pose_xyz", "pose_xyz"),
        ("corrected_sources", "sources"),
        ("corrected_observed", "observed"),
        ("prior_raw_review_for_identity", "identity_review"),
    ):
        if not np.array_equal(
            arrays[new][times, identities], geometry[old][times, slots], equal_nan=True
        ):
            raise ValueError("corrected export changed selected raw geometry/source/support")
    if (
        np.any(arrays["corrected_observed"][~held])
        or np.any(arrays["corrected_sources"][~held] != -1)
        or not np.isnan(arrays["corrected_pose_xyz"][~held]).all()
        or not arrays["prior_raw_review_for_identity"][~held].all()
    ):
        raise ValueError("unknown identity must retain missing geometry and source support")
    expected_node = np.where(
        arrays["corrected_observed"][..., None], arrays["corrected_pose_xyz"], np.nan
    )
    if not np.array_equal(
        arrays["corrected_node_xyz"], expected_node, equal_nan=True
    ) or not np.array_equal(arrays["corrected_xyz"], expected_node[:, :, 1], equal_nan=True):
        raise ValueError("corrected measured coordinates must retain every unobserved NaN")
    origin = arrays["identity_origin"]
    if np.any((origin < 0) | (origin > 3)) or np.any((origin > 0) != held):
        raise ValueError("corrected identity origin does not match known/unknown mapping")
    old_available = np.isfinite(geometry["accepted_xyz"]).all(-1)
    new_available = np.isfinite(arrays["corrected_xyz"]).all(-1)
    if not review_count:
        if np.any(held & (mapping != np.arange(3))):
            raise ValueError("zero human reviews cannot create a new global ID assignment")
        if np.any(new_available & ~old_available) or np.any(origin > 1):
            raise ValueError("zero human reviews cannot increase automatic acceptance")
    return dict(
        original_automatic_accepted_pectoral=int(old_available.sum()),
        corrected_available_pectoral=int(new_available.sum()),
        newly_available_pectoral=int((new_available & ~old_available).sum()),
        newly_unavailable_pectoral=int((old_available & ~new_available).sum()),
        unresolved_identity_frame_slots=int((~held).sum()),
        coverage_difference_is_identity_risk_policy_not_new_geometry=True,
    )


def save_core_result(result, geometry, output, record, *, review_count):
    """Persist the pure core contract without coercing values or filling NaNs."""
    coverage = validate_core_result(result, geometry, review_count)
    arrays, metadata = result["arrays"], result["metadata"]
    path = output / "correction.npz"
    np.savez_compressed(path, **arrays)
    restored = verified._npz(path)
    if set(restored) != set(arrays) or any(
        restored[key].shape != value.shape
        or restored[key].dtype != value.dtype
        or restored[key].tobytes() != value.tobytes()
        for key, value in arrays.items()
    ):
        raise ValueError("saved dense correction does not preserve exact array bytes")
    record.output("correction", path, sha256=base.sha256(path))
    path = output / "correction_metadata.json"
    write_json(path, metadata)
    record.output("correction_metadata", path, sha256=base.sha256(path))
    return coverage


def main(argv=None):
    args = parse_args(argv)
    args.geometry_run = args.geometry_run.resolve(strict=True)
    args.review = [path.resolve(strict=True) for path in args.review]
    args.output = args.output.resolve()
    if (
        not args.output.is_relative_to(REPO.resolve())
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise ValueError("a new output directory inside the workspace is required")
    # Import before the initial source capture. There is deliberately no fallback
    # that relabels automatically if the reviewed dense core is unavailable.
    core = importlib.import_module("review_correction")
    sources = code_paths()
    code = {str(path): base.sha256(path) for path in sources}
    versions = dict(numpy=np.__version__, opencv=evaluate_review.cv2.__version__)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with provenance(STAGE, args.output, name="correction") as record:
        record.param("geometry_run", str(args.geometry_run))
        record.param("review_files", [str(path) for path in args.review])
        record.param("output", str(args.output))
        record.param("code_hashes", code)
        record.param("package_versions", versions)
        snapshots = snapshot_code(args.output, sources, code, record)
        prepared = load_review_inputs(args.geometry_run, args.review)
        inputs = dict(prepared.verified_hashes)
        for path, digest in code.items():
            _merge(inputs, path, digest)
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        assert_unchanged({**inputs, **snapshots})
        original_arrays = {
            key: review_constraints._array_digest(value) for key, value in prepared.geometry.items()
        }
        original_path = args.output / "raw_geometry.npz"
        shutil.copyfile(prepared.run.root / "candidate.npz", original_path)
        original_sha = prepared.metadata["candidate_sha256"]
        if base.sha256(original_path) != original_sha:
            raise RuntimeError("original geometry copy differs from the completed producer")
        record.output("raw_geometry", original_path, sha256=original_sha)
        for index, content in enumerate(prepared.review_bytes):
            path = args.output / "human_reviews" / f"review_{index:04d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            if base.sha256(path) != digest:
                raise RuntimeError("human review copy differs from the supplied original bytes")
            record.output(f"human_review/{index}", path, sha256=digest)
        constraints_path = args.output / "constraints.json"
        write_json(
            constraints_path,
            dict(
                bundles=list(prepared.bundles),
                combined=prepared.combined,
                loader_metadata=prepared.metadata,
            ),
        )
        record.output("constraints", constraints_path, sha256=base.sha256(constraints_path))
        corrected = core.build_review_correction(
            prepared.geometry,
            list(prepared.bundles),
            tracker_config=prepared.run.params["tracker_config"],
            candidate_sha256=original_sha,
        )
        if original_arrays != {
            key: review_constraints._array_digest(value) for key, value in prepared.geometry.items()
        }:
            raise RuntimeError("dense core modified the original geometry arrays")
        coverage = save_core_result(
            corrected,
            prepared.geometry,
            args.output,
            record,
            review_count=len(prepared.review_bytes),
        )
        summary = dict(
            schema="three_body_review_correction_export_v1",
            technical_completion=True,
            stage=STAGE,
            **prepared.metadata,
            corrected_array_names=sorted(corrected["arrays"]),
            correction_metadata=corrected["metadata"],
            export_conservation=coverage,
            original_array_sha256=original_arrays,
            original_geometry_npz_copied_byte_exact=True,
            original_geometry_arrays_unchanged=True,
            geometry_refitted=False,
            raw_source_rows_reassigned=False,
            manual_review_used_as_input=bool(prepared.review_bytes),
            biological_identity_ground_truth_created=False,
            elapsed_seconds=time.perf_counter() - started,
            input_sha256=inputs,
            code_sha256=code,
            package_versions=versions,
        )
        path = args.output / "summary.json"
        write_json(path, summary)
        record.output("summary", path, sha256=base.sha256(path))
        assert_unchanged(
            {
                **inputs,
                **snapshots,
                **{entry["path"]: entry["sha256"] for entry in record.outputs.values()},
            }
        )
        if versions != dict(numpy=np.__version__, opencv=evaluate_review.cv2.__version__):
            raise RuntimeError("correction package runtime changed")
        record.param("all_input_code_snapshot_output_sha_unchanged", True)
        record.param("original_geometry_arrays_unchanged", True)
        record.param("post_inference_assets_opened", False)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                frames=len(prepared.run.frames),
                supplied_review_files=len(prepared.review_bytes),
            )
        )
    )
    return summary


if __name__ == "__main__":
    main()
