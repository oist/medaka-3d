"""Extract nine registered top-camera frames and a separate median background.

The original MP4 is opened once; each requested native frame is sought directly.
Its whole-file SHA is an inherited registration, not a current payload rehash.
Current evidence is hardlink/stat identity and nine exact decoded BGR digests.
No existing background, tracking source, mask or SLEAP input is modified/read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
STAGE = "top_background_5to6_minutes_v1"
FRAMES = (49800, 51045, 52290, 53535, 54780, 56025, 57270, 58515, 59760)
FPS, TOTAL, FIRST, CHUNK = 166, 348600, 49800, 1660
IMAGE_SHAPE = (1200, 1200, 3)
INDEX = WORK / "nodewise_body_row_five_minute_continuation_v1/input_index.json"
INDEX_SHA = "f5840d96f166f04516c72326aee157e78fac01a290296653aabdec01dd8ae568"
STAGED = WORK / "inputs/long_source_masks_49800-348600_v1"
VIDEO = WORK / "source_cache/medaka_202512/20251225_K_1_2_FCMC/top_cam/cam0_top.mp4"
VIDEO_SHA = "d769ba8a550292f0869ccd6d75a2f425b4408c2f188e3c684db825a1e1870186"
VIDEO_BYTES = 5067172992


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(UTC).isoformat()


def stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def signature(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=sha(array.tobytes()))


class Evidence:
    """Only small metadata and this source are captured; no video payload hashing."""

    def __init__(self):
        self.records = {}

    def capture(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        before = stamp(path)
        data = path.read_bytes()
        require(before == stamp(path), "metadata changed while reading")
        digest = sha(data)
        require(expected is None or digest == expected, "metadata/source SHA differs: " + str(path))
        self.records[str(path)] = dict(sha256=digest, size_bytes=len(data), stat=before)
        return data

    def read(self, path, expected):
        return json.loads(self.capture(path, expected))

    def final_guard(self):
        for path, record in self.records.items():
            require(stamp(path) == record["stat"], "metadata/source changed after capture")


def load_context(evidence):
    index = evidence.read(INDEX, INDEX_SHA)
    require(len(index["prefix_chunks"]) == 40, "registered forty-chunk index required")
    detector = index["original_binding"]["inputs"]["detector_binding"]
    require(detector["input_sha256"]["video:top_cam"] == VIDEO_SHA, "registered video differs")
    rows, links, common_stamp, old_background = [], [], None, None
    for number in range(7):
        chunk = index["prefix_chunks"][number]
        start, stop = FIRST + CHUNK * number, FIRST + CHUNK * (number + 1)
        require(
            (chunk["index"], chunk["start"], chunk["stop"]) == (number, start, stop),
            "chunk interval differs",
        )
        root = (STAGED / chunk["attempt_path"]).resolve()
        require(root.is_relative_to(STAGED.resolve()), "staging path escapes registered root")
        manifest_path = root / "manifest.json"
        seal = evidence.read(root / "staging.provenance.json", chunk["provenance_sha256"])
        manifest = evidence.read(manifest_path, chunk["manifest_sha256"])
        require(
            seal["status"] == "ok" and seal["stage"] == "prepare_independent_mask_chunk",
            "successful staging required",
        )
        require(seal["params"]["binding"] == detector, "detector input lineage differs")
        require(
            manifest["source_interval_half_open"] == [start, stop]
            and manifest["frames"] == list(range(start, stop))
            and manifest["fps"] == FPS
            and manifest["image_shape_hw"] == list(IMAGE_SHAPE[:2])
            and manifest["raw_storage"] == "original_source_video"
            and manifest["protocol_sha256"] == detector["protocol_sha256"],
            "native manifest metadata differs",
        )
        mref = seal["outputs"]["manifest"]
        require(
            Path(mref["path"]).resolve() == manifest_path
            and mref["sha256"] == chunk["manifest_sha256"],
            "manifest output edge differs",
        )
        raw = manifest["raw"]["top_cam"]
        path = (root / raw["path"]).resolve()
        require(path.is_relative_to(root), "raw path escapes staging root")
        require(
            raw["sha256"] == VIDEO_SHA
            and raw["bytes"] == VIDEO_BYTES
            and raw["source_total_frames"] == TOTAL
            and raw["source_offset_frames"] == start
            and raw["storage_link"] == "hardlink"
            and len(raw["frame_bgr_sha256"]) == CHUNK,
            "native video descriptor differs",
        )
        source, output = seal["inputs"]["video:top_cam"], seal["outputs"]["top_cam_raw"]
        require(
            Path(source["path"]).resolve() == VIDEO.resolve()
            and source["sha256"] == VIDEO_SHA
            and source["size"] == VIDEO_BYTES,
            "canonical staging input differs",
        )
        require(
            Path(output["path"]).resolve() == path
            and output["sha256"] == VIDEO_SHA
            and output["size"] == VIDEO_BYTES,
            "staged output differs",
        )
        recorded = chunk["output_hashes"]["top_cam_raw"]
        require(
            Path(recorded["path"]).resolve() == path and recorded["sha256"] == VIDEO_SHA,
            "consumed raw edge differs",
        )
        link = chunk["raw_hardlinks"]["top_cam"]
        require(
            Path(link["path"]).resolve() == path
            and Path(link["source_path"]).resolve() == VIDEO.resolve()
            and link["sha256"] == VIDEO_SHA,
            "hardlink lineage differs",
        )
        current = [link["device"], link["inode"], link["bytes"], link["mtime_ns"]]
        require(
            current[2] == VIDEO_BYTES and (common_stamp is None or common_stamp == current),
            "shared video stamp differs",
        )
        common_stamp = current
        links.append(str(path))
        previous = manifest["backgrounds"]["top_cam"]
        require(
            old_background is None or old_background == previous,
            "registered old background differs",
        )
        old_background = previous
        for frame in FRAMES:
            if start <= frame < stop:
                rows.append(
                    dict(
                        frame_idx=frame,
                        recording_seconds=frame / FPS,
                        chunk_index=number,
                        manifest_path=str(manifest_path),
                        manifest_sha256=chunk["manifest_sha256"],
                        manifest_relative_row=frame - start,
                        native_absolute_seek=frame,
                        expected_bgr_sha256=raw["frame_bgr_sha256"][frame - start],
                    )
                )
    require(tuple(row["frame_idx"] for row in rows) == FRAMES, "exact nine ordered frames required")
    return dict(
        frames=rows,
        video_path=str(VIDEO.resolve()),
        hardlinks=links,
        expected_stat=common_stamp,
        inherited_whole_video_sha256=VIDEO_SHA,
        inherited_video_bytes=VIDEO_BYTES,
        prior_background_descriptor_only=old_background,
    )


def video_guard(context):
    source = Path(context["video_path"])
    require(
        stamp(source) == context["expected_stat"],
        "canonical video stat differs from registered input",
    )
    for name in context["hardlinks"]:
        path = Path(name)
        require(
            path.samefile(source) and stamp(path) == context["expected_stat"],
            "staged hardlink/stat differs",
        )


def extract_frames(context, *, capture_factory=None):
    require(
        tuple(row["frame_idx"] for row in context["frames"]) == FRAMES,
        "exact fixed native frames required",
    )
    factory = cv2.VideoCapture if capture_factory is None else capture_factory
    capture = factory(context["video_path"])
    images = []
    try:
        require(
            capture.isOpened()
            and capture.get(cv2.CAP_PROP_FRAME_COUNT) == TOTAL
            and np.isclose(capture.get(cv2.CAP_PROP_FPS), FPS, rtol=0, atol=1e-6)
            and (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            == IMAGE_SHAPE[:2],
            "original video header mismatch",
        )
        for row in context["frames"]:
            native = row["frame_idx"]
            require(
                capture.set(cv2.CAP_PROP_POS_FRAMES, native)
                and np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), native, rtol=0, atol=1e-6),
                "absolute native seek failed",
            )
            okay, bgr = capture.read()
            require(
                okay
                and bgr is not None
                and bgr.dtype == np.uint8
                and bgr.shape == IMAGE_SHAPE
                and np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), native + 1, rtol=0, atol=1e-6),
                "native decode shape/dtype/position mismatch",
            )
            require(
                signature(bgr)["sha256"] == row["expected_bgr_sha256"], "native BGR SHA mismatch"
            )
            images.append(bgr.copy())
        return images
    finally:
        capture.release()


def median_background(images):
    require(
        len(images) == len(FRAMES)
        and all(image.shape == IMAGE_SHAPE and image.dtype == np.uint8 for image in images),
        "nine native uint8 BGR images required",
    )
    # Nine is odd: every channel's median is an existing integer sample, no rounding.
    return np.median(np.stack(images, axis=0), axis=0).astype(np.uint8)


def png_bytes(image):
    okay, encoded = cv2.imencode(".png", image)
    require(okay, "PNG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    require(
        decoded is not None and signature(decoded) == signature(image),
        "PNG lossless roundtrip differs",
    )
    return encoded.tobytes()


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK.resolve()) and output != WORK.resolve() and not output.exists(),
        "fresh workspace output required",
    )
    output.mkdir(parents=True)
    started, evidence, outputs = utc(), Evidence(), {}

    def write(name, data):
        if not isinstance(data, bytes):
            data = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        outputs[name] = dict(path=str(path), sha256=sha(data), size_bytes=len(data))
        return outputs[name]

    try:
        source = evidence.capture(Path(__file__))
        write("source_snapshot.py", source)
        context = load_context(evidence)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                source_sha256=sha(source),
                context=context,
                native_frames=list(FRAMES),
                recording_seconds=[f / FPS for f in FRAMES],
                science_interval=[66400, 116200],
                all_background_samples_before_science=True,
                median="per-pixel per-BGR-channel median of ordered nine uint8 samples",
                whole_video_rehashes=0,
                current_verification=(
                    "registered hardlink/stat + nine decoded BGR SHA + absolute seek/header/shape"
                ),
                inherited_video_digest_is_not_current_whole_byte_verification=True,
                existing_backgrounds_or_tracking_changed=False,
                runtime=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__),
            ),
        )
        video_guard(context)
        try:
            images = extract_frames(context)
        finally:
            video_guard(context)
        records = []
        for row, image in zip(context["frames"], images, strict=True):
            record = write(f"native/frame_{row['frame_idx']}_top_cam.png", png_bytes(image))
            records.append(
                dict(
                    **row, image=record, bgr_signature=signature(image), png_roundtrip_verified=True
                )
            )
        background = median_background(images)
        bg = write("background_top_cam.png", png_bytes(background))
        write(
            "frame_evidence.json",
            dict(
                schema=STAGE + "_frames",
                frames=records,
                background=dict(
                    image=bg, bgr_signature=signature(background), png_roundtrip_verified=True
                ),
            ),
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            native_frames=list(FRAMES),
            requested_BGR_reads=9,
            video_capture_count=1,
            native_PNG_count=9,
            background_PNG_count=1,
            whole_video_rehashes=0,
            codec_internal_keyframe_decodes_not_counted=True,
            original_mask_SL_candidate_checkpoint_reads=0,
            GPU_calls=0,
            science_frames_used=False,
            median_bgr_signature=signature(background),
            background_adopted_for_tracking=False,
            image_term_scoring_performed=False,
        )
        write("summary.json", summary)
        evidence.final_guard()
        video_guard(context)
        for record in outputs.values():
            data = Path(record["path"]).read_bytes()
            require(
                len(data) == record["size_bytes"] and sha(data) == record["sha256"],
                "written output differs",
            )
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=utc(),
            inputs=evidence.records,
            outputs=dict(outputs),
            video_reference=dict(
                path=context["video_path"],
                inherited_sha256=VIDEO_SHA,
                registered_stat=context["expected_stat"],
                whole_bytes_rehashed=False,
                requested_decoded_BGR_verified=9,
            ),
        )
    except BaseException as error:
        write(
            "preparation.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=utc(),
                inputs=evidence.records,
                outputs=dict(outputs),
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        raise
    write("preparation.provenance.json", seal)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
