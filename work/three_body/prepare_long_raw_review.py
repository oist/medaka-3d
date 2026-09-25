"""Raw-only review of one scene inside one sealed original-video chunk.

The original source manifest is preserved verbatim: it is not an FFV1 clip.
Candidate bytes are hashed but no pose, mask array or SLEAP array is loaded.
Actual raw decoding requires --execute; cross-chunk scenes fail explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src")]
import review_scene as scene_ui  # noqa: E402
from review_truth import VIEWS, relative_file, sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FIRST, STOP, CHUNK = 49800, 348600, 1660
STAGE = "long_original_video_raw_review"
INVENTORY_STAGE = "long_saved_backtrace_review_inventory"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(REPO), "input path must remain in workspace")
    return path


class Evidence:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = local(path)
        digest = sha256(path)
        require(expected is None or digest == expected, "input SHA mismatch: " + str(path))
        require(self.hashes.get(path, digest) == digest, "input changed while verifying")
        self.hashes[path] = digest
        return path

    def final_guard(self):
        for path, digest in self.hashes.items():
            require(sha256(path) == digest, "input changed before seal: " + str(path))


def sealed(root, name, stage, evidence, *, verify_outputs=True):
    path = evidence.bind(root / name)
    record = read(path)
    require(
        record.get("status") == "ok" and record.get("stage") == stage,
        "successful expected producer stage required",
    )
    if verify_outputs:
        for item in record["outputs"].values():
            output = local(item["path"])
            require(
                output.is_relative_to(root) and output.stat().st_size == item["size"],
                "producer output path/size differs",
            )
            evidence.bind(output, item["sha256"])
    return record


def output_binding(record, role, path, evidence):
    item = record["outputs"][role]
    require(local(item["path"]) == path, "producer output role mismatch: " + role)
    return evidence.bind(path, item["sha256"])


def selected_scene(proposal, inventory_summary, scene_id):
    require(type(scene_id) is int and scene_id >= 0, "nonnegative integer scene ID required")
    require(
        proposal.get("schema") == "three_body_raw_review_proposal_v1"
        and proposal["risk_counts"] == inventory_summary["risk_counts"],
        "inventory proposal/count binding differs",
    )
    matches = [item for item in proposal["scenes"] if item["scene_id"] == scene_id]
    require(len(matches) == 1, "scene ID is missing or duplicated")
    item = matches[0]
    require(
        item["status"] == "proposed_for_human_review" and not item["selection_issues"],
        "scene still requires human probe selection",
    )
    start, stop = item["context_start_frame"], item["context_stop_frame_exclusive"]
    require(
        type(start) is int and type(stop) is int and item["context_frames"] == stop - start,
        "scene native count differs",
    )
    first, last = inventory_summary["processed_interval"]
    require(first <= start < stop <= last, "scene outside completed producer prefix")
    scene_ui.select_scene(list(range(first, last)), start, stop, item["proposed_anchor_frames"])
    return item


def select_chunk(consumed, first, stop, clip_start, clip_stop):
    require(
        first == FIRST and (stop - first) % CHUNK == 0 and len(consumed) == (stop - first) // CHUNK,
        "consumed prefix count differs",
    )
    for index, item in enumerate(consumed):
        require(
            type(item["index"]) is int
            and item["index"] == index
            and type(item["start"]) is int
            and item["start"] == first + CHUNK * index
            and type(item["stop"]) is int
            and item["stop"] == first + CHUNK * (index + 1),
            "consumed chunks are not the contiguous declared prefix",
        )
    matches = [item for item in consumed if item["start"] <= clip_start < clip_stop <= item["stop"]]
    require(
        len(matches) == 1, "this bounded adapter requires the whole scene inside one sealed chunk"
    )
    return matches[0]


def validate_source(manifest, consumed, detector_binding):
    require(
        manifest.get("experiment") == "20251225_K_1_2_FCMC"
        and manifest.get("raw_storage") == "original_source_video"
        and manifest.get("fps") == 166
        and manifest.get("image_shape_hw") == [1200, 1200]
        and manifest.get("tracking_performed") is False
        and manifest.get("human_truth_created") is False
        and set(manifest["raw"]) == set(VIEWS),
        "original source manifest contract differs",
    )
    require(
        manifest["frames"] == list(range(consumed["start"], consumed["stop"]))
        and manifest["binding"] == detector_binding,
        "source native range/detector binding differs",
    )
    for view in VIEWS:
        item = manifest["raw"][view]
        require(
            type(item["source_offset_frames"]) is int
            and item["source_offset_frames"] == consumed["start"]
            and type(item["source_total_frames"]) is int
            and item["source_total_frames"] == STOP,
            "original source offset/frame count differs",
        )
        values = item["frame_bgr_sha256"]
        require(
            len(values) == CHUNK
            and all(
                isinstance(value, str)
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value)
                for value in values
            ),
            "native BGR SHA sequence differs",
        )
        require(
            item["sha256"] == detector_binding["input_sha256"]["video:" + view],
            "original video differs from detector input",
        )


@dataclass
class ReviewInput:
    run: Path
    inventory: Path
    scene: dict
    producer_summary: dict
    consumed_record: dict
    chunk_root: Path
    manifest: dict
    manifest_path: Path
    videos: dict
    evidence: Evidence


def prepare_inputs(run, inventory, scene_id):
    """Hash completed evidence; do not deserialize a candidate or decode any image."""
    run, inventory = local(run), local(inventory)
    evidence = Evidence()
    producer = sealed(run, "continuation.provenance.json", "three_body_long_continuation", evidence)
    require(
        producer["params"].get("all_input_source_snapshot_output_sha_unchanged") is True,
        "producer final guard missing",
    )
    for role, name in (
        ("summary", "summary.json"),
        ("candidate", "candidate.npz"),
        ("consumed_chunks", "consumed_chunks.json"),
    ):
        output_binding(producer, role, run / name, evidence)
    summary, consumed = read(run / "summary.json"), read(run / "consumed_chunks.json")
    first, stop = summary["processed_interval"]
    require(
        summary["schema"] == "long_continuation_phase_v1"
        and summary["technical_completion"] is True
        and first == FIRST
        and type(stop) is int
        and first < stop <= STOP
        and producer["params"]["through_frame"] == stop
        and summary["full_sequence_complete"] is (stop == STOP)
        and summary["prefix_backtrace_provisional"] is (stop != STOP),
        "producer completion/native prefix flags differ",
    )
    audit = sealed(inventory, "inventory.provenance.json", INVENTORY_STAGE, evidence)
    require(
        audit["params"].get("all_bound_input_source_output_sha_unchanged") is True,
        "inventory final guard missing",
    )
    for role, name in (("summary", "summary.json"), ("review_proposal", "review_proposal.json")):
        output_binding(audit, role, inventory / name, evidence)
    for name in (
        "continuation.provenance.json",
        "candidate.npz",
        "summary.json",
        "consumed_chunks.json",
    ):
        path = run / name
        item = audit["inputs"].get(str(path))
        require(
            item is not None
            and local(item["path"]) == path
            and item["sha256"] == evidence.hashes[path],
            "inventory belongs to a different run",
        )
    audit_summary = read(inventory / "summary.json")
    require(
        audit_summary["schema"] == "long_backtrace_review_inventory_v1"
        and audit_summary["processed_interval"] == [first, stop]
        and audit_summary["native_frames"] == stop - first
        and audit_summary["candidate_sha256"] == evidence.hashes[run / "candidate.npz"]
        and audit_summary["producer_seal_sha256"]
        == evidence.hashes[run / "continuation.provenance.json"]
        and audit_summary["full_sequence_complete"] is (stop == STOP)
        and audit_summary["prefix_backtrace_provisional"] is (stop != STOP),
        "inventory completion/candidate binding differs",
    )
    item = selected_scene(read(inventory / "review_proposal.json"), audit_summary, scene_id)
    chunk = select_chunk(
        consumed, first, stop, item["context_start_frame"], item["context_stop_frame_exclusive"]
    )
    fixed = producer["params"]["bindings"]["inputs"]
    stage = local(fixed["input_stage_root"])
    chunk_root = relative_file(stage, chunk["attempt_path"])
    manifest_path, seal_path = chunk_root / "manifest.json", chunk_root / "staging.provenance.json"
    evidence.bind(seal_path, chunk["provenance_sha256"])
    # Only raw/manifest/code evidence is needed; no D2 array is deserialized.
    chunk_seal = sealed(
        chunk_root,
        "staging.provenance.json",
        "prepare_independent_mask_chunk",
        evidence,
        verify_outputs=False,
    )
    evidence.bind(manifest_path, chunk["manifest_sha256"])
    output_binding(chunk_seal, "manifest", manifest_path, evidence)
    manifest = read(manifest_path)
    require(
        chunk_seal["params"]["binding"] == fixed["detector_binding"]
        and manifest["protocol_sha256"] == fixed["protocol_sha256"],
        "consumed detector protocol binding differs",
    )
    validate_source(manifest, chunk, fixed["detector_binding"])
    videos = {}
    for view in VIEWS:
        raw = manifest["raw"][view]
        path = relative_file(chunk_root, raw["path"])
        output_binding(chunk_seal, view + "_raw", path, evidence)
        require(
            evidence.hashes[path] == raw["sha256"]
            and chunk["output_hashes"][view + "_raw"]["sha256"] == raw["sha256"]
            and local(chunk["output_hashes"][view + "_raw"]["path"]) == path,
            "consumed raw payload binding differs",
        )
        videos[view] = path
    return ReviewInput(
        run, inventory, item, summary, chunk, chunk_root, manifest, manifest_path, videos, evidence
    )


def iter_native(videos, manifest, start, stop, *, capture_factory=None):
    """Absolute MP4 seek, manifest-relative SHA index; close before reading stop."""
    factory = cv2.VideoCapture if capture_factory is None else capture_factory
    captures = {}
    first, last = manifest["frames"][0], manifest["frames"][-1] + 1
    require(first <= start < stop <= last, "requested raw frames outside sealed chunk")
    try:
        for view in VIEWS:
            capture = factory(str(videos[view]))
            captures[view] = capture
            require(
                capture.isOpened()
                and capture.get(cv2.CAP_PROP_FRAME_COUNT) == STOP
                and np.isclose(capture.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
                and (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                == (1200, 1200),
                "original video metadata mismatch: " + view,
            )
            require(
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                and np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), start, rtol=0, atol=1e-6),
                "absolute source seek failed: " + view,
            )
        for frame in range(start, stop):
            images = {}
            for view in VIEWS:
                capture = captures[view]
                ok, bgr = capture.read()
                require(
                    ok
                    and bgr is not None
                    and bgr.shape == (1200, 1200, 3)
                    and bgr.dtype == np.uint8
                    and np.isclose(
                        capture.get(cv2.CAP_PROP_POS_FRAMES), frame + 1, rtol=0, atol=1e-6
                    ),
                    "native raw decode/frame position mismatch: " + view,
                )
                digest = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
                require(
                    digest == manifest["raw"][view]["frame_bgr_sha256"][frame - first],
                    "native BGR SHA mismatch: " + view,
                )
                images[view] = bgr
            yield frame, images
    finally:
        for capture in captures.values():
            capture.release()


def write_images(context, output):
    """Same JPEG/PNG/montage encoding as frozen review_scene.write_frames."""
    item = context.scene
    start, stop = item["context_start_frame"], item["context_stop_frame_exclusive"]
    probes = item["proposed_anchor_frames"]
    stream = iter_native(context.videos, context.manifest, start, stop)
    hashes, native = {}, {view: [] for view in VIEWS}
    intermediate = output / "raw_three_views_mpeg4_intermediate.mp4"
    writer = None
    try:
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            scene_ui.PLAYBACK_FPS,
            scene_ui.VIDEO_SHAPE[::-1],
        )
        require(writer.isOpened(), "review video writer unavailable")
        for frame, images in stream:
            montage = np.zeros((*scene_ui.VIDEO_SHAPE, 3), np.uint8)
            for column, view in enumerate(VIEWS):
                bgr = images[view]
                native[view].append(hashlib.sha256(bgr.tobytes()).hexdigest())
                jpeg = output / "frames" / f"f{frame}_{view}.jpg"
                require(
                    cv2.imwrite(str(jpeg), bgr, [cv2.IMWRITE_JPEG_QUALITY, 96]), "JPEG write failed"
                )
                restored = cv2.imread(str(jpeg))
                require(
                    restored is not None
                    and restored.shape == bgr.shape
                    and restored.dtype == np.uint8,
                    "JPEG decode failed",
                )
                hashes[jpeg.relative_to(output).as_posix()] = sha256(jpeg)
                if frame in probes:
                    png = output / f"f{frame}_{view}.png"
                    require(cv2.imwrite(str(png), bgr), "probe PNG write failed")
                    require(np.array_equal(cv2.imread(str(png)), bgr), "probe PNG is not lossless")
                    hashes[png.name] = sha256(png)
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
    finally:
        stream.close()
        if writer is not None:
            writer.release()
    return intermediate, hashes, native


def validate_written_scene(manifest, frames, probes, hashes, native):
    require(set(native) == set(VIEWS), "written native view order/membership differs")
    offset = frames[0] - manifest["frames"][0]
    for view in VIEWS:
        require(
            native[view]
            == manifest["raw"][view]["frame_bgr_sha256"][offset : offset + len(frames)],
            "written native frame SHA count/content differs",
        )
        require(
            all(f"frames/f{frame}_{view}.jpg" in hashes for frame in frames)
            and all(f"f{frame}_{view}.png" in hashes for frame in probes),
            "static image link or lossless probe is missing",
        )


def build_pack(run, inventory, scene_id, output):
    output = Path(output).resolve()
    require(output.is_relative_to(REPO) and not output.exists(), "new workspace output required")
    code_paths = [
        Path(__file__).resolve(),
        LAB / "review_scene.py",
        LAB / "review_truth.py",
        LAB / "review_truth.html",
        LAB / "review_scene.html",
        REPO / "src/medaka3d/provenance.py",
    ]
    code = {path: sha256(path) for path in code_paths}
    require(
        code[LAB / "review_truth.html"] == scene_ui.HTML_ORIGIN["sha256"],
        "original UI source changed",
    )
    context = prepare_inputs(run, inventory, scene_id)
    item = context.scene
    start, stop, probes = (
        item["context_start_frame"],
        item["context_stop_frame_exclusive"],
        item["proposed_anchor_frames"],
    )
    manifest_sha = context.evidence.hashes[context.manifest_path]
    template, frames, _ = scene_ui.bind_template(
        context.manifest["frames"], manifest_sha, start, stop, probes
    )
    html = scene_ui.render_html(
        (LAB / "review_scene.html").read_text(encoding="utf-8"), template, frames
    )
    executable = shutil.which("ffmpeg")
    require(executable is not None, "ffmpeg is required for existing H.264 review encoding")
    ffmpeg = Path(executable).resolve()
    ffmpeg_sha = sha256(ffmpeg)
    version = subprocess.run(
        [str(ffmpeg), "-version"], capture_output=True, check=True, text=True
    ).stdout.splitlines()[0]
    output.mkdir(parents=True, exist_ok=False)
    with provenance(STAGE, output, name="review") as record:
        for path, digest in {**context.evidence.hashes, **code, ffmpeg: ffmpeg_sha}.items():
            record.input(str(path), path, sha256=digest)
        record.param("run", str(context.run))
        record.param("inventory", str(context.inventory))
        record.param("scene_id", scene_id)
        record.param("raw_only", True)
        record.param("candidate_arrays_deserialized", False)
        snapshots = {}
        for path, digest in code.items():
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha256(target) == digest, "source snapshot mismatch")
            snapshots[target.relative_to(output).as_posix()] = digest
        (output / "frames").mkdir()
        copied_manifest = output / "source_manifest.json"
        shutil.copyfile(context.manifest_path, copied_manifest)
        require(sha256(copied_manifest) == manifest_sha, "source manifest copy changed")
        intermediate, hashes, native = write_images(context, output)
        validate_written_scene(context.manifest, frames, probes, hashes, native)
        video = output / "raw_three_views_slow.mp4"
        scene_ui.transcode(ffmpeg, intermediate, video)
        video_qa = scene_ui.verify_video(video, len(frames))
        intermediate_qa = scene_ui.verify_video(intermediate, len(frames))
        (output / "index.html").write_text(html, encoding="utf-8")
        (output / "truth_template.json").write_text(
            json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        hashes.update(snapshots)
        for name in (
            "index.html",
            "truth_template.json",
            copied_manifest.name,
            intermediate.name,
            video.name,
        ):
            hashes[name] = sha256(output / name)
        result = dict(
            schema="long_raw_review_pack_v1",
            technical_completion=True,
            producer_run=str(context.run),
            inventory=str(context.inventory),
            scene_id=scene_id,
            source_manifest=str(context.manifest_path),
            source_manifest_sha256=manifest_sha,
            source_frame_range=template["source_frame_range"],
            review_clip_range=[start, stop],
            displayed_frames=frames,
            probe_frames=probes,
            frame_count=len(frames),
            original_source_video_frame_count=STOP,
            absolute_seek_frame=start,
            first_manifest_sha_offset=start - context.manifest["frames"][0],
            native_pixel_hashes_verified=3 * len(frames),
            native_bgr_sha256=native,
            review_video=video_qa,
            intermediate_video=intermediate_qa,
            producer_prefix_backtrace_provisional=context.producer_summary[
                "prefix_backtrace_provisional"
            ],
            consumed_record=context.consumed_record,
            inventory_scene=item,
            raw_video_payload_bytes_copied=0,
            original_raw_storage=context.manifest["raw_storage"],
            source_manifest_copy_is_audit_only_not_a_relocated_reader_descriptor=True,
            model_outputs_shown=False,
            candidate_arrays_deserialized=False,
            masks_or_sleap_arrays_deserialized=False,
            human_truth_created=False,
            ui_operations_verified=False,
            browser_used=False,
            gpu_used=False,
            output_sha256=hashes,
            code_sha256={str(p): h for p, h in code.items()},
            runtime=dict(
                python=sys.version, numpy=np.__version__, opencv=cv2.__version__, ffmpeg=version
            ),
            completion_rule="review.provenance.json status ok and all output hashes verified",
            limitations=[
                "Scene selection is model-derived; the page shows raw images "
                "and an empty human template only.",
                "A/B/C belong to this clip; "
                "they are not the initial global slots or biological names.",
                "The source is one sealed original-video chunk, not a fabricated FFV1 descriptor.",
                "Original answers must be preserved; future final candidates require "
                "fresh unique mouth matching and scope verification.",
                "The current prefix backtrace may change; "
                "its slots/array hashes are not future identity truth.",
                "No browser/localhost/Chrome/Playwright/CDP used; "
                "interactive UI remains unverified.",
                "JPEG/video are lossy; proposed probe PNGs preserve exact native pixels.",
                "Producer/inventory bytes and selected chunk raw evidence checked; "
                "independent history/geometry and D2 inference are not rerun.",
            ],
        )
        target = output / "summary.json"
        target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        all_outputs = {**hashes, "summary.json": sha256(target)}
        for name, digest in all_outputs.items():
            require(sha256(output / name) == digest, "output changed before sealing")
            record.output(name, output / name, sha256=digest)
        context.evidence.final_guard()
        for path, digest in {**code, ffmpeg: ffmpeg_sha}.items():
            require(sha256(path) == digest, "source/runtime changed while making review")
        for item in record.outputs.values():
            require(
                sha256(Path(item["path"])) == item["sha256"],
                "output changed during final input check",
            )
        record.param("all_input_source_output_sha_unchanged", True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "inventory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument(
        "--execute", action="store_true", help="Decode the already selected raw review scene"
    )
    args = parser.parse_args(argv)
    require(args.execute, "--execute required; no implicit raw decoding")
    result = build_pack(args.run, args.inventory, args.scene_id, args.output)
    print(json.dumps({"output": str(args.output), "frames": result["frame_count"]}))


if __name__ == "__main__":
    main()
