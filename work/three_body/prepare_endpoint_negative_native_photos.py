"""Save nine lossless native photographs for a later three-method comparison.

The three fixed absolute frames are decoded from the original full-length video
hardlinks. Chunk-relative positions are used only to check consumed BGR hashes.
No SLEAP/mask decoding, candidate loading, tracking, fitting or GPU execution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import render_depth_identity_transition_review as frozen

proof, require, previous, prior = frozen.proof, frozen.require, frozen.previous, frozen.prior
WORK, REPO, VIEWS = frozen.WORK, frozen.REPO, frozen.VIEWS
STAGE = "endpoint_negative_native_photos_v1"
FRAMES = (90141, 90258, 90259)
START, STOP = 89640, 91300
PRODUCER = WORK / "nodewise_body_row_negative_ten_second_continuation_v1"
PRODUCER_SHA = "2774d83411c7b2a28a7d6bfc42a1ca424d1409c8f431cce0b3b6c64adf39ac44"
PRODUCER_SOURCE_SHA = "4f2d31a383a8d59320ef0c09f8dcddf645dfb20e0849bd2f1dc27398d719669e"
MANIFEST_SHA = "17a9cf8721d1dc4868b7bbc9881e0222e47b6b84a8955ccb0b4b8b29e840cd92"
STAGING_SHA = "c0456a25bf4f209f65bb2cf61afa42133bf0a708e5116a2673f69109223eaadb"
RAWPACK = WORK / "fast_tail_transition_raw_review_pack_v1"
RAWPACK_SHA = "8622bd5655ecc34058c5e3a7a2fe64e7ff357a3d9eb508cfca4e258405fb0c7f"
PINS = {
    **frozen.PINS,
    "work/three_body/render_depth_identity_transition_review.py": (
        "26dcca519ee7364a55e9c40af6579eec13d9bdb3c253dc5d3ef66cc552d5312a"
    ),
    "work/three_body/run_nodewise_body_row_negative_continuation.py": PRODUCER_SOURCE_SHA,
}


def source_code(evidence):
    for name, digest in PINS.items():
        evidence.add(REPO / name, digest)
    code = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if (
                path.suffix == ".py"
                and path.is_relative_to(REPO)
                and ".venv" not in path.parts
                and path.is_file()
            ):
                code[str(path)] = evidence.add(path)
    return code


def load_context(evidence):
    producer = previous.read(PRODUCER / "tracking.provenance.json", evidence, PRODUCER_SHA)
    require(
        producer["stage"] == PRODUCER.name and producer["status"] == "ok",
        "completed negative producer required",
    )
    frozen.edge(
        producer, WORK / "run_nodewise_body_row_negative_continuation.py", PRODUCER_SOURCE_SHA
    )
    summary, index = [
        frozen.document(PRODUCER, producer, name + ".json", evidence)
        for name in ("summary", "input_index")
    ]
    require(
        summary["schema"] == PRODUCER.name
        and summary["status"] == "ok"
        and summary["process_interval"] == [START, STOP]
        and summary["retained_interval"] == [49800, STOP]
        and summary["native_frames"] == 1660
        and summary["finish_calls"] == 1
        and summary["initialize_count"] == 1
        and summary["restore_count"] == 7,
        "completed native extent differs",
    )
    require(
        set(index) == {"original_binding", "prefix_chunks"} and len(index["prefix_chunks"]) == 25,
        "actual consumed25 prefix required",
    )
    chunk = index["prefix_chunks"][24]
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (24, START, STOP)
        and chunk["manifest_sha256"] == MANIFEST_SHA
        and chunk["provenance_sha256"] == STAGING_SHA,
        "exact consumed chunk24 required",
    )
    # This frozen helper reads only manifest/staging metadata and checks source
    # edges, including canonical full videos. It never opens video/mask/SL data.
    context = frozen.load_chunk(chunk, index["original_binding"]["inputs"], [producer], evidence)
    pack = previous.read(RAWPACK / "review.provenance.json", evidence, RAWPACK_SHA)
    require(
        pack["status"] == "ok" and pack["stage"] == "fast_tail_transition_raw_review_pack_v1",
        "saved rawpack required",
    )
    old = frozen.document(RAWPACK, pack, "summary.json", evidence)
    require(
        old["status"] == "ok"
        and old["review_clip_range"] == [90040, 90400]
        and old["consumed_record"] == chunk,
        "rawpack source chunk differs",
    )
    for frame in FRAMES:
        for view in VIEWS:
            desc = context["manifest"]["raw"][view]
            require(
                len(desc["frame_bgr_sha256"]) == STOP - START,
                "one native BGR digest per consumed frame required",
            )
            require(
                desc["frame_bgr_sha256"][frame - START]
                == old["native_bgr_sha256"][view][frame - 90040],
                "consumed native BGR digest and old rawpack source digest differ",
            )
    context["original_binding"] = index["original_binding"]
    context["rawpack_summary_reference"] = pack["outputs"]["summary.json"]
    return context


def native_frame(context, frame):
    require(type(frame) is int and frame in FRAMES, "only three registered absolute frames")
    # Videos have all348600 native frames. Never use frame-START as a seek index.
    previous.video_stat_guard(context["videos"], context["chunk"])
    iterator = prior.raw.iter_native(context["videos"], context["manifest"], frame, frame + 1)
    try:
        native, images = next(iterator)
        require(
            native == frame and next(iterator, None) is None, "exactly one requested native frame"
        )
        require(set(images) == set(VIEWS), "all three cameras required")
        return images
    finally:
        iterator.close()
        previous.video_stat_guard(context["videos"], context["chunk"])


def write_native_png(out, frame, view, image, expected_sha):
    require(frame in FRAMES and view in VIEWS, "unregistered native output")
    require(
        image.dtype == np.uint8
        and image.shape == (1200, 1200, 3)
        and proof.signature(image)["sha256"] == expected_sha,
        "native BGR pixels differ from consumed evidence",
    )
    okay, encoded = prior.cv2.imencode(".png", image)
    require(okay, "native PNG encoding failed")
    restored = prior.cv2.imdecode(encoded, prior.cv2.IMREAD_COLOR)
    require(
        restored is not None and proof.signature(restored) == proof.signature(image),
        "lossless PNG roundtrip differs",
    )
    record = out.write(f"native/frame_{frame}_{view}.png", encoded.tobytes())
    return dict(
        frame_idx=frame,
        view=view,
        image=record,
        bgr_signature=proof.signature(image),
        native_absolute_seek=frame,
        consumed_chunk_offset=frame - START,
        camera_sync_offset_applied=0,
        lossless_PNG_roundtrip_verified=True,
    )


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK and not output.exists(),
        "fresh workspace output required",
    )
    require(
        "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(), "CPU only"
    )
    out, evidence, started = proof.Output(output), proof.Evidence(), proof.utc()
    try:
        code = source_code(evidence)
        context = load_context(evidence)
        out.write(
            "plan.json",
            dict(
                schema=STAGE,
                frames=list(FRAMES),
                views=list(VIEWS),
                source_code=code,
                producer_sha256=PRODUCER_SHA,
                source_chunk=context["chunk"],
                rawpack_sha256=RAWPACK_SHA,
                rawpack_summary=context["rawpack_summary_reference"],
                requested_BGR_images=9,
                original_video_absolute_seek=True,
                original_video_full_rehashes=0,
                masks_SLEAP_candidates_checkpoint_reads=0,
                original_binding=context["original_binding"],
                raw_photos_only=True,
            ),
        )
        rows = []
        for frame in FRAMES:
            images = native_frame(context, frame)
            for view in VIEWS:
                expected = context["manifest"]["raw"][view]["frame_bgr_sha256"][frame - START]
                rows.append(write_native_png(out, frame, view, images[view], expected))
            del images
        require(
            [(r["frame_idx"], r["view"]) for r in rows] == [(f, v) for f in FRAMES for v in VIEWS],
            "nine native output coverage differs",
        )
        out.write(
            "frame_evidence.json",
            dict(
                schema=STAGE + "_frame_evidence",
                records=rows,
                pixel_digest_source=(
                    "consumed native manifest and saved rawpack ORIGINAL BGR digests"
                ),
                JPEG_pixels_used=False,
                physical_ID_truth=False,
            ),
        )
        out.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                views=list(VIEWS),
                native_PNG_count=9,
                requested_original_BGR_frames=9,
                source_chunk_index=24,
                source_interval=[START, STOP],
                codec_internal_keyframe_decodes_not_counted=True,
                original_video_full_rehashes=0,
                original_mask_or_SLEAP_reads=0,
                candidate_or_checkpoint_reads=0,
                tracking_fitting_or_GPU_calls=0,
                labels_or_human_answers_created=False,
                physical_ID_truth=False,
            ),
        )
        previous.video_stat_guard(context["videos"], context["chunk"])
        evidence.final_guard()
        out.final_guard()
        require(
            "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized(),
            "CUDA initialized",
        )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
        )
    except BaseException as error:
        seal = dict(
            stage=STAGE,
            status="failed",
            started_at=started,
            finished_at=proof.utc(),
            inputs=evidence.paths,
            outputs=out.records,
            error=repr(error),
        )
        (output / "preparation.provenance.json").write_bytes(proof.json_bytes(seal))
        raise
    (output / "preparation.provenance.json").write_bytes(proof.json_bytes(seal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
