"""Bounded actual SL/camera helper smoke; not a human or completed-final mapping."""

from pathlib import Path
from time import perf_counter

import map_long_review_to_final as mapper
import numpy as np

REFERENCE = mapper.REPO / (
    "work/three_body/long_remaining_supervisor_v1/phases/f066400-076360/tracking"
)
OUTPUT = mapper.REPO / "work/three_body/final_review_anchor_geometry_smoke_v1"
FRAMES = np.array([65931, 66151, 66217], dtype=np.int64)
SOURCE_SHA = "b0ad0a2225efe86a9cd4ef51b05818f3cabe1de780623949dd513d7b1528ee78"


def main():
    mapper.require(not OUTPUT.exists(), "new smoke output required")
    evidence = mapper.final.phase.Digests()
    codes = mapper.code_hashes()
    for path, digest in codes.items():
        evidence.add(path, digest)
    evidence.add(Path(mapper.__file__), SOURCE_SHA)
    OUTPUT.mkdir(parents=True)
    with mapper.provenance(
        "final_review_anchor_geometry_helper_smoke", OUTPUT, name="audit"
    ) as out:
        try:
            mapper.snapshot(codes, OUTPUT, out)
            seal_path = REFERENCE / "continuation.provenance.json"
            evidence.add(seal_path)
            producer = mapper.review.document(seal_path.read_bytes(), "reference producer")
            mapper.require(
                producer.get("status") == "ok"
                and producer.get("stage") == "three_body_long_continuation"
                and producer["params"]["through_frame"] == 76360
                and producer["params"]["all_input_source_snapshot_output_sha_unchanged"] is True,
                "completed reference prefix required",
            )
            manifest_path = REFERENCE / "checkpoint/manifest.json"
            manifest_digest = producer["outputs"]["checkpoint_manifest"]["sha256"]
            mapper.final.legacy.output_binding(producer, manifest_path, manifest_digest)
            evidence.add(manifest_path, manifest_digest)
            manifest = mapper.review.document(manifest_path.read_bytes(), "checkpoint manifest")
            mapper.require(
                manifest["schema"] == "medaka_tracker_checkpoint_v1"
                and manifest["status"] == "ok"
                and manifest["metadata"]["file"] == "metadata.json",
                "checkpoint metadata declaration differs",
            )
            mapper.equal(manifest["bindings"], producer["params"]["bindings"], "binding mismatch")
            metadata_path = REFERENCE / "checkpoint/metadata.json"
            metadata_digest = manifest["metadata"]["sha256"]
            mapper.final.legacy.output_binding(producer, metadata_path, metadata_digest)
            evidence.add(metadata_path, metadata_digest)
            mapper.require(
                metadata_path.stat().st_size == manifest["metadata"]["bytes"],
                "metadata size differs",
            )
            metadata = mapper.review.document(metadata_path.read_bytes(), "checkpoint metadata")
            mapper.require(
                metadata["processed_frames"] == 76360 - 49800
                and metadata["previous_frame"] == 76359,
                "reference metadata native scope differs",
            )
            shared = manifest["bindings"]["inputs"]["seed"]["SL_and_calibration_sha256"]
            for path, digest in shared.items():
                mapper.input_item(producer, path, digest)
            # Invoke the production helper unchanged, with actual registered cameras/SL.
            started = perf_counter()
            rig, points, calibration = mapper.load_anchor_geometry(
                shared, metadata["context"], FRAMES, evidence
            )
            elapsed = perf_counter() - started
            arrays = {"frame_idx": FRAMES.copy()}
            rows = []
            for frame in FRAMES:
                for view in mapper.VIEWS:
                    array = points[int(frame)][view]
                    key = f"f{frame}_{view}"
                    arrays[key] = array.copy()
                    rows.append(
                        dict(
                            frame=int(frame),
                            view=view,
                            shape=list(array.shape),
                            dtype=array.dtype.str,
                            sha256=mapper.review.array_sha256(array),
                            fully_finite_rows=int(np.isfinite(array).all(axis=(1, 2)).sum()),
                        )
                    )
            archive = OUTPUT / "native_SL_anchor_rows.npz"
            np.savez_compressed(archive, **arrays)
            out.output("native_SL_anchor_rows", archive, sha256=mapper.review.sha256(archive))
            summary = dict(
                schema="final_review_anchor_geometry_helper_smoke_v1",
                status="ok",
                reference_run=str(REFERENCE),
                reference_seal_sha256=evidence.paths[str(seal_path)],
                checkpoint_manifest_sha256=manifest_digest,
                checkpoint_metadata_sha256=metadata_digest,
                mapper_sha256=SOURCE_SHA,
                helper="load_anchor_geometry",
                native_frames=FRAMES.tolist(),
                camera_views=list(rig.views),
                world_units=rig.world_units,
                normalized_camera_context_exact=True,
                calibration_sha256=calibration,
                SL_calibration_sha256=shared,
                arrays=rows,
                elapsed_helper_seconds=elapsed,
                actual_final_mapping=False,
                actual_human_review_received=False,
                synthetic_human_review_created=False,
                final_audit_complete=False,
                final_loader_called=False,
                run_mapping_called=False,
                candidate_coordinate_arrays_opened=False,
                checkpoint_arrays_opened=False,
                raw_images_decoded=0,
                GPU_used=False,
                all_reference_outputs_reverified=False,
                scope="Only successful reference seal, bound checkpoint metadata, actual camera "
                "context and three native SL anchors were verified. No final-chain, raw-review "
                "matching, support/identity result or full-run audit is certified.",
            )
            mapper.save_json(OUTPUT / "summary.json", summary, out, "summary")
            report = OUTPUT / "RESULT.md"
            report.write_text(
                "The actual load_anchor_geometry helper passed for 65931, 66151, 66217.\n\n"
                "The registered three-view refractive camera context equals the completed "
                "160-second reference checkpoint context. All nine native SLEAP arrays retain "
                "the original row order and are saved with shape, dtype and byte digests.\n\n"
                "This is a helper compatibility smoke. No human JSON, final mapping, candidate "
                "coordinate arrays, raw-video decode, GPU work or completed 30-minute audit "
                "was performed. final_audit_complete remains false.\n",
                encoding="utf-8",
            )
            out.output("RESULT", report, sha256=mapper.review.sha256(report))
        finally:
            for path, digest in evidence.paths.items():
                out.input(path, path, sha256=digest)
            mapper.require(mapper.code_hashes() == codes, "loaded source set changed")
            evidence.final_guard()
            mapper.final_outputs(out)
        out.param("all_opened_source_input_snapshot_output_sha_unchanged", True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
