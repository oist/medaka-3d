"""Static and decoded-file QA of the requested model-blind scene0 review pack."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from prepare_sequence import VIEWS, sha256  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

ROOT = REPO / "work/three_body/minute_reviews/scene0_79236-79488"
OUT = ROOT / "artifact_qa"
FRAMES = list(range(79236, 79488))
PROBES = [79316, 79405, 79480]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    if OUT.exists():
        raise FileExistsError("QA directory already exists; no overwrite")
    summary_path, seal_path = ROOT / "summary.json", ROOT / "review_scene.provenance.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    require(seal["status"] == "ok", "raw pack is not complete")
    require(
        seal["outputs"]["summary"]["sha256"] == sha256(summary_path), "pack summary is not sealed"
    )
    inputs = {str(summary_path): sha256(summary_path), str(seal_path): sha256(seal_path)}
    for item in seal["outputs"].values():
        require(sha256(item["path"]) == item["sha256"], "pack output SHA mismatch")
        inputs[item["path"]] = item["sha256"]
    for path, digest in summary["code_sha256_before"].items():
        require(sha256(path) == digest, "frozen pack/UI source changed")
        inputs[path] = digest
    require(
        summary["input_sha256_before"] == summary["input_sha256_after"],
        "source changed during pack creation",
    )
    require(
        summary["code_sha256_before"] == summary["code_sha256_after"],
        "pack source changed during creation",
    )
    require(
        summary["displayed_frames"] == FRAMES and summary["probe_frames"] == PROBES,
        "wrong requested scene/probes",
    )
    require(summary["first_source_clip_offset"] == 2616, "wrong native source offset")
    require(
        summary["native_pixel_hashes_verified"] == 756, "not all selected native pixels verified"
    )
    require(
        not summary["model_outputs_used"] and not summary["human_truth_created"],
        "pack contains model or truth claims",
    )
    require(
        not summary["browser_used"] and not summary["ui_operations_verified"],
        "unexpected browser/UI claim",
    )
    source_manifest_path = Path(summary["source_manifest"])
    require(
        sha256(source_manifest_path) == summary["source_manifest_sha256"],
        "original input manifest changed",
    )
    inputs[str(source_manifest_path)] = sha256(source_manifest_path)
    template_path = ROOT / "truth_template.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    require(
        template["status"] == "unreviewed_template_not_ground_truth",
        "human confirmation was invented",
    )
    require(template["source_frame_range"] == [76620, 86580], "template lost full-source binding")
    require(template["review_clip_range"] == [79236, 79488], "template lost clip binding")
    require(
        template["source_manifest_sha256"] == summary["source_manifest_sha256"],
        "template source hash differs",
    )
    require(not template["model_outputs_shown"], "template shows model suggestions")
    require([a["frame"] for a in template["anchors"]] == PROBES, "template probes differ")
    require(
        all(a["frame_choice"] == "proposed_not_human_confirmed" for a in template["anchors"]),
        "template probes are falsely confirmed",
    )
    require(
        all(
            not a["points"] and a["visibility"] == "unreviewed"
            for anchor in template["anchors"]
            for a in anchor["animals"].values()
        ),
        "raw template already contains animal marks",
    )
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    embedded_truth, _ = decoder.raw_decode(html.split("let truth=", 1)[1])
    embedded_frames, _ = decoder.raw_decode(html.split("const frames=", 1)[1])
    embedded_views, _ = decoder.raw_decode(html.split("viewNames=", 1)[1])
    require(
        embedded_truth == template and embedded_frames == FRAMES and embedded_views == list(VIEWS),
        "HTML embedded data differs",
    )
    require(
        'max="251"' in html and 'src="raw_three_views_slow.mp4"' in html,
        "HTML slider/video link differs",
    )
    require("frames/f${frame}_${view}.jpg" in html, "HTML dynamic JPEG path changed")
    for frame in FRAMES:
        for view in VIEWS:
            path = ROOT / "frames" / f"f{frame}_{view}.jpg"
            image = cv2.imread(str(path))
            require(
                image is not None and image.shape == (1200, 1200, 3),
                "missing native-size linked JPEG",
            )
    probe_checks = []
    for frame in PROBES:
        for view in VIEWS:
            path = ROOT / f"f{frame}_{view}.png"
            image = cv2.imread(str(path))
            require(image is not None and image.shape == (1200, 1200, 3), "missing probe PNG")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            require(
                digest == summary["native_bgr_sha256"][view][frame - FRAMES[0]],
                "probe PNG changed native pixels",
            )
            probe_checks.append(dict(frame_idx=frame, view=view, native_bgr_sha256=digest))
    guard_path = REPO / "work/three_body/review_scene_import_guard_qa.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    require(
        guard["status"] == "passed"
        and guard["source_html_sha256"] == summary["html_template_sha256"],
        "prior pure-JS guard QA applies to different UI source",
    )
    inputs[str(guard_path)] = sha256(guard_path)
    inputs[str(Path(__file__))] = sha256(Path(__file__))
    OUT.mkdir(exist_ok=False)
    with provenance("qa_model_blind_minute_scene0", OUT, name="artifact") as record:
        for path, digest in inputs.items():
            record.input(path, path, sha256=digest)
        for path in (Path(__file__), guard_path):
            target = OUT / path.name
            shutil.copyfile(path, target)
            record.output(path.name, target, sha256=sha256(target))
        video = ROOT / "raw_three_views_slow.mp4"
        capture = cv2.VideoCapture(str(video))
        decoded, selected = 0, []
        try:
            require(capture.isOpened(), "H264 video cannot open")
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> (8 * index)) & 255) for index in range(4))
            require(codec.lower() in {"avc1", "h264"}, f"unexpected video codec {codec}")
            require(
                capture.get(cv2.CAP_PROP_FRAME_COUNT) == 252
                and capture.get(cv2.CAP_PROP_FPS) == 20,
                "wrong H264 count/fps",
            )
            wanted = {0, 80, 126, 169, 244, 251}
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                require(
                    image.shape == (636, 1800, 3) and image.dtype == np.uint8,
                    "H264 frame shape/dtype differs",
                )
                require(
                    capture.get(cv2.CAP_PROP_POS_FRAMES) == decoded + 1, "H264 frame index differs"
                )
                if decoded in wanted:
                    path = OUT / f"decoded_source_{FRAMES[decoded]}.png"
                    require(cv2.imwrite(str(path), image), "decoded QA PNG write failed")
                    record.output(path.name, path, sha256=sha256(path))
                    selected.append(
                        dict(video_index=decoded, source_frame=FRAMES[decoded], path=path.name)
                    )
                decoded += 1
        finally:
            capture.release()
        require(decoded == 252 and len(selected) == 6, "H264 full decode incomplete")
        require(
            all(sha256(path) == digest for path, digest in inputs.items()),
            "input changed during artifact QA",
        )
        result = dict(
            status="passed",
            source_frame_range=[76620, 86580],
            review_clip_range=[79236, 79488],
            native_frame_count=252,
            linked_jpegs_decoded=756,
            lossless_native_probe_pngs=probe_checks,
            h264_codec=codec,
            full_decoded_frames=decoded,
            playback_fps=20,
            playback_duration_s=252 / 20,
            source_duration_s=252 / 166,
            slowdown=166 / 20,
            decoded_image_qa=selected,
            static_html_data_and_local_links_verified=True,
            source_and_clip_hash_binding_verified=True,
            all_pack_output_sha_verified=True,
            template_unreviewed_without_marks=True,
            human_truth_created=False,
            model_outputs_shown=False,
            browser_used=False,
            ui_operations_verified=False,
            gpu_used=False,
            original_pack_outputs_unchanged=True,
            original_native_BGR_validation=(
                "all 756 images and original source SHA before/after "
                "verified by sealed pack builder"
            ),
            runtime=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__),
            limitations=[
                "Static HTML/data and decoded files were checked; no browser, localhost, "
                "Chrome, Playwright or CDP was used.",
                "Interactive UI remains unverified; no human identity annotation has been made.",
                "H264/JPEG are display encodings; native PNG probes retain exact original pixels.",
            ],
        )
        path = OUT / "summary.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "linked_jpegs_decoded",
                        "h264_codec",
                        "full_decoded_frames",
                        "playback_duration_s",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
