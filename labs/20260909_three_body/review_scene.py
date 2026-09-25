"""Model-blind review of an explicit subset of a sealed native FFV1 sequence.

The human template stays bound to the full source manifest/range. Only the
displayed clip, HTML frame array and proposed probes use the selected subset.
No browser automation is performed; static data and encoded artifacts are QA'd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
REPO = LAB.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from review_truth import VIEWS, make_template, relative_file, sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

VIDEO_SHAPE = (636, 1800)
PLAYBACK_FPS = 20.0
HTML_ORIGIN = {
    "path": "labs/20260909_three_body/review_truth.html",
    "sha256": "4b74d6036623cc987f2ecfb0e7849ca0f9d1a2798e7070c075e48247c8ecfdef",
    "changes": (
        "copied to review_scene.html; import additionally requires equal source/clip ranges, "
        "three chronological in-clip anchors, and handles malformed JSON"
    ),
}


def select_scene(frames, start, stop, probes):
    if (
        not isinstance(frames, list)
        or len(frames) < 3
        or any(type(frame) is not int for frame in frames)
        or frames != list(range(frames[0], frames[-1] + 1))
    ):
        raise ValueError("full source frames must be consecutive native integers")
    if (
        type(start) is not int
        or type(stop) is not int
        or not frames[0] <= start < stop <= frames[-1] + 1
        or stop - start < 3
    ):
        raise ValueError("review clip must contain at least three frames inside the source range")
    if (
        len(probes) != 3
        or any(type(frame) is not int for frame in probes)
        or probes != sorted(set(probes))
        or not all(start <= frame < stop for frame in probes)
    ):
        raise ValueError("three distinct chronological probes must lie inside the review clip")
    return list(range(start, stop)), start - frames[0]


def bind_template(frames, manifest_sha, start, stop, probes):
    subset, offset = select_scene(frames, start, stop, probes)
    template = make_template(frames, manifest_sha, probes)
    template["review_clip_range"] = [start, stop]
    return template, subset, offset


def preflight(sequence, output, start, stop, probes):
    if output.exists() or output.is_symlink():
        raise FileExistsError("review output exists; no overwrite")
    manifest_path = sequence / "manifest.json" if sequence.is_dir() else sequence
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256(manifest_path)
    template, subset, offset = bind_template(manifest["frames"], digest, start, stop, probes)
    if (
        manifest.get("experiment") != "20251225_K_1_2_FCMC"
        or manifest.get("fps") != 166
        or manifest.get("image_shape_hw") != [1200, 1200]
        or manifest.get("raw_storage") == "original_source_video"
        or not str(manifest.get("raw_codec", "")).startswith("FFV1")
        or set(manifest["raw"]) != set(VIEWS)
    ):
        raise ValueError(
            "review_scene requires a native 166 fps, 1200-square three-view FFV1 stage"
        )
    seal_path = root / "staging.provenance.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "ok" or seal["outputs"]["manifest"]["sha256"] != digest:
        raise ValueError("native staging manifest is not successfully sealed")
    hashes = {manifest_path: digest, seal_path: sha256(seal_path)}
    paths = {}
    for view in VIEWS:
        info = manifest["raw"][view]
        path = relative_file(root, info["path"])
        if not path.is_file() or sha256(path) != info["sha256"]:
            raise ValueError(f"sealed raw checksum mismatch: {view}")
        if seal["outputs"][f"{view}_raw"]["sha256"] != info["sha256"]:
            raise ValueError("raw input is not bound into the completed staging seal")
        values = info.get("frame_bgr_sha256")
        if (
            not isinstance(values, list)
            or len(values) != len(manifest["frames"])
            or any(
                not isinstance(v, str)
                or len(v) != 64
                or any(c not in "0123456789abcdef" for c in v)
                for v in values
            )
        ):
            raise ValueError("every native source frame must have a valid BGR SHA256")
        hashes[path] = info["sha256"]
        paths[view] = path
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for the existing H.264 review format")
    return manifest_path, manifest, template, subset, offset, paths, hashes, Path(ffmpeg)


def open_sources(paths, source_count, offset):
    captures = {}
    try:
        for view in VIEWS:
            capture = cv2.VideoCapture(str(paths[view]))
            captures[view] = capture
            if (
                not capture.isOpened()
                or capture.get(cv2.CAP_PROP_FRAME_COUNT) != source_count
                or not np.isclose(capture.get(cv2.CAP_PROP_FPS), 166, rtol=0, atol=1e-6)
                or (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                != (1200, 1200)
                # OpenCV 5 reports the registered codec as lowercase 'ffv1'.
                or capture.get(cv2.CAP_PROP_FOURCC)
                not in (cv2.VideoWriter_fourcc(*"FFV1"), cv2.VideoWriter_fourcc(*"ffv1"))
            ):
                raise ValueError(f"native FFV1 metadata mismatch: {view}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, offset):
                raise ValueError(f"native clip seek failed: {view}")
            if not np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), offset, rtol=0, atol=1e-6):
                raise ValueError(f"native clip seek position mismatch: {view}")
    except BaseException:
        for capture in captures.values():
            capture.release()
        raise
    return captures


def write_frames(manifest, subset, offset, probes, paths, output):
    captures = open_sources(paths, len(manifest["frames"]), offset)
    writer = None
    intermediate = output / "raw_three_views_mpeg4_intermediate.mp4"
    checksums = {}
    native_hashes = {view: [] for view in VIEWS}
    try:
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, VIDEO_SHAPE[::-1]
        )
        if not writer.isOpened():
            raise RuntimeError("review MP4 writer is unavailable")
        for index, frame in enumerate(subset):
            montage = np.zeros((*VIDEO_SHAPE, 3), np.uint8)
            for column, view in enumerate(VIEWS):
                capture = captures[view]
                ok, bgr = capture.read()
                if (
                    not ok
                    or bgr is None
                    or bgr.shape != (1200, 1200, 3)
                    or bgr.dtype != np.uint8
                    or not np.isclose(
                        capture.get(cv2.CAP_PROP_POS_FRAMES), offset + index + 1, rtol=0, atol=1e-6
                    )
                ):
                    raise ValueError(f"native BGR/frame position mismatch: {view}/{frame}")
                digest = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
                if digest != manifest["raw"][view]["frame_bgr_sha256"][offset + index]:
                    raise ValueError(f"native BGR SHA256 mismatch: {view}/{frame}")
                native_hashes[view].append(digest)
                jpeg = output / "frames" / f"f{frame}_{view}.jpg"
                if not cv2.imwrite(str(jpeg), bgr, [cv2.IMWRITE_JPEG_QUALITY, 96]):
                    raise RuntimeError("native-size review JPEG write failed")
                check = cv2.imread(str(jpeg))
                if check is None or check.shape != bgr.shape or check.dtype != np.uint8:
                    raise ValueError("native-size review JPEG decode failed")
                checksums[jpeg.relative_to(output).as_posix()] = sha256(jpeg)
                if frame in probes:
                    png = output / f"f{frame}_{view}.png"
                    if not cv2.imwrite(str(png), bgr):
                        raise RuntimeError("native probe PNG write failed")
                    if not np.array_equal(cv2.imread(str(png)), bgr):
                        raise ValueError("lossless probe PNG pixel mismatch")
                    checksums[png.name] = sha256(png)
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
        # A selected scene normally ends before the source FFV1 clip does.
        # No extra read or source EOF assertion belongs here.
    finally:
        if writer is not None:
            writer.release()
        for capture in captures.values():
            capture.release()
    return intermediate, checksums, native_hashes


def transcode(ffmpeg, source, target):
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(source),
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
            str(target),
        ],
        check=True,
        capture_output=True,
    )


def verify_video(path, expected_frames):
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        if (
            not capture.isOpened()
            or capture.get(cv2.CAP_PROP_FRAME_COUNT) != expected_frames
            or not np.isclose(capture.get(cv2.CAP_PROP_FPS), PLAYBACK_FPS, rtol=0, atol=1e-6)
            or (capture.get(cv2.CAP_PROP_FRAME_HEIGHT), capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            != VIDEO_SHAPE
        ):
            raise ValueError("review video metadata failed QA")
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded += 1
            if image.shape != (*VIDEO_SHAPE, 3) or image.dtype != np.uint8:
                raise ValueError("review video frame shape/dtype failed QA")
            if not np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), decoded, rtol=0, atol=1e-6):
                raise ValueError("review video frame position failed QA")
    finally:
        capture.release()
    if decoded != expected_frames:
        raise ValueError("review video complete decode frame count failed QA")
    return dict(
        decoded_frame_count=decoded,
        playback_fps=PLAYBACK_FPS,
        image_shape_hw=list(VIDEO_SHAPE),
        all_selected_native_frames_included=True,
    )


def render_html(source, template, frames):
    for token in ("__TRUTH__", "__FRAMES__", "__VIEWS__", "__LAST__"):
        if source.count(token) != 1:
            raise ValueError("frozen review HTML placeholders are inconsistent")
    html = (
        source.replace("__TRUTH__", json.dumps(template, ensure_ascii=False))
        .replace("__FRAMES__", json.dumps(frames))
        .replace("__VIEWS__", json.dumps(VIEWS))
        .replace("__LAST__", str(len(frames) - 1))
    )
    if any(token in html for token in ("__TRUTH__", "__FRAMES__", "__VIEWS__", "__LAST__")):
        raise ValueError("review HTML retains an unsubstituted placeholder")
    return html


def build_pack(sequence, start, stop, probes, output):
    output, sequence = Path(output).resolve(), Path(sequence)
    source_path, manifest, template, frames, offset, paths, inputs, ffmpeg = preflight(
        sequence, output, start, stop, probes
    )
    code_paths = (
        Path(__file__),
        LAB / "review_truth.py",
        LAB / "review_truth.html",
        LAB / "review_scene.html",
        LAB / "evaluate_review.py",
        REPO / "src/medaka3d/provenance.py",
    )
    code = {path: sha256(path) for path in code_paths}
    if code[LAB / "review_truth.html"] != HTML_ORIGIN["sha256"]:
        raise ValueError("the registered original review HTML changed")
    html_source = (LAB / "review_scene.html").read_text(encoding="utf-8")
    html = render_html(html_source, template, frames)
    inputs[ffmpeg] = sha256(ffmpeg)
    ffmpeg_version = subprocess.run(
        [str(ffmpeg), "-version"], capture_output=True, check=True, text=True
    ).stdout.splitlines()[0]
    output.mkdir(parents=True, exist_ok=False)
    with provenance("three_body_raw_review_scene", output, name="review_scene") as record:
        (output / "frames").mkdir()
        snapshot_hashes = {}
        for path, digest in inputs.items():
            record.input(str(path), path, sha256=digest)
        for path, digest in code.items():
            destination = output / "code_snapshots" / path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if sha256(destination) != digest:
                raise ValueError("review source snapshot mismatch")
            snapshot_hashes[destination.relative_to(output).as_posix()] = digest
            record.input("code:" + str(path), path, sha256=digest)
        record.param("review_clip_range", [start, stop])
        record.param("probe_frames", probes)
        record.param("model_outputs_shown", False)
        record.param("browser_used", False)
        intermediate, output_hashes, raw_hashes = write_frames(
            manifest, frames, offset, probes, paths, output
        )
        output_hashes.update(snapshot_hashes)
        video = output / "raw_three_views_slow.mp4"
        transcode(ffmpeg, intermediate, video)
        qa = verify_video(video, len(frames))
        intermediate_qa = verify_video(intermediate, len(frames))
        (output / "index.html").write_text(html, encoding="utf-8")
        (output / "truth_template.json").write_text(
            json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name in (intermediate.name, video.name, "index.html", "truth_template.json"):
            output_hashes[name] = sha256(output / name)
        if any(sha256(path) != digest for path, digest in inputs.items()):
            raise ValueError("review source input changed during rendering")
        if any(sha256(path) != digest for path, digest in code.items()):
            raise ValueError("frozen review/UI/evaluation source changed during rendering")
        for name, digest in output_hashes.items():
            if sha256(output / name) != digest:
                raise ValueError("review output changed before sealing")
            record.output(name, output / name, sha256=digest)
        result = dict(
            source_manifest=str(source_path),
            source_manifest_sha256=inputs[source_path],
            source_frame_range=template["source_frame_range"],
            review_clip_range=[start, stop],
            first_source_clip_offset=offset,
            displayed_frames=frames,
            probe_frames=probes,
            frame_count=len(frames),
            native_pixel_hashes_verified=3 * len(frames),
            native_bgr_sha256=raw_hashes,
            review_video=qa,
            intermediate_video=intermediate_qa,
            output_sha256=output_hashes,
            input_sha256_before={str(p): value for p, value in inputs.items()},
            input_sha256_after={str(p): value for p, value in inputs.items()},
            code_sha256_before={str(p): value for p, value in code.items()},
            code_sha256_after={str(p): value for p, value in code.items()},
            input_and_source_unchanged=True,
            model_outputs_used=False,
            human_truth_created=False,
            ui_operations_verified=False,
            browser_used=False,
            static_html_data_qa=("four data placeholders substituted; subset frames/probes bound"),
            html_origin=HTML_ORIGIN,
            html_template_sha256=code[LAB / "review_scene.html"],
            preview_encoding="native-size JPEG quality 96; three probes additionally lossless PNG",
            runtime=dict(
                python=sys.version,
                numpy=np.__version__,
                opencv=cv2.__version__,
                ffmpeg=ffmpeg_version,
                gpu_used=False,
            ),
            limitations=[
                "No browser, localhost server, Chrome, Playwright or CDP was used.",
                "Interactive UI is unverified; only static data and artifacts were checked.",
                "The template is unreviewed, without animal marks or biological naming.",
                "Full source manifest/range binding is preserved; clip range is display metadata.",
                "Import rejects another source/clip range or anchors outside the displayed clip.",
                "Display video and JPEGs are lossy; probe PNGs preserve native pixels.",
            ],
            completion_rule="requires review_scene.provenance.json status ok and summary SHA match",
        )
        summary = output / "summary.json"
        summary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", summary, sha256=sha256(summary))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--probes", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_pack(args.sequence, args.start, args.stop, args.probes, args.output)
    print(
        json.dumps(
            {
                "frames": result["frame_count"],
                "native_pixels_checked": result["native_pixel_hashes_verified"],
            }
        )
    )


if __name__ == "__main__":
    main()
