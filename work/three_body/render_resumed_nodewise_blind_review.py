"""Build raw three-camera review clips from a frozen five-minute scene ledger.

No tracker, detector, inference output, model ID or annotation is used to draw a
frame. Scene selection remains separate from this blinded reviewer interface.
All native video frames are retained at 20 fps; full-resolution stills every
four native frames, plus exact entry/middle/exit, support anonymous mouth clicks.
"""

# ruff: noqa: E501, RUF001 -- HTML/JavaScript literals and Japanese typography.

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from review_truth import make_template  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

VIEWS = ("top_cam", "side1_cam", "side2_cam")
PLAYBACK_FPS = 20
STILL_STRIDE = 4
STAGE = "resumed_nodewise_blind_three_camera_review_v1"
TRACKING_ROOT = WORK / "nodewise_body_row_five_minute_resume_099600_v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def read_checked(path, digest=None):
    data = Path(path).read_bytes()
    require(digest is None or sha_bytes(data) == digest, f"SHA differs: {path}")
    return json.loads(data)


def stat_signature(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


class Source:
    def __init__(self, selected_frames):
        seal_path = TRACKING_ROOT / "tracking.provenance.json"
        seal = read_checked(seal_path)
        require(seal["status"] == "ok", "successful resumed candidate required")
        index_path = TRACKING_ROOT / "input_index.json"
        ref = seal["outputs"]["input_index.json"]
        require(Path(ref["path"]).resolve() == index_path.resolve(), "sealed input index path")
        index = read_checked(index_path, ref["sha256"])
        self.records = {
            str(seal_path): sha_file(seal_path),
            str(index_path): ref["sha256"],
        }
        self.videos, self.expected, self.frame_origins = {}, {}, {}
        wanted = set(selected_frames)
        binding = index["original_binding"]["inputs"]
        for chunk in index["prefix_chunks"]:
            frames = sorted(wanted.intersection(range(chunk["start"], chunk["stop"])))
            if not frames:
                continue
            manifest_path = Path(chunk["output_hashes"]["manifest"]["path"])
            manifest = read_checked(manifest_path, chunk["manifest_sha256"])
            self.records[str(manifest_path)] = chunk["manifest_sha256"]
            require(
                manifest["frames"] == list(range(chunk["start"], chunk["stop"])),
                "manifest frame order",
            )
            require(
                manifest["fps"] == 166 and manifest["image_shape_hw"] == [1200, 1200],
                "source geometry/time",
            )
            staging_path = manifest_path.parent / "staging.provenance.json"
            staging = read_checked(staging_path, chunk["provenance_sha256"])
            require(staging["status"] == "ok", "successful staging required")
            self.records[str(staging_path)] = chunk["provenance_sha256"]
            for view in VIEWS:
                descriptor = manifest["raw"][view]
                link = chunk["raw_hardlinks"][view]
                path = Path(link["source_path"]).resolve()
                hardlink = Path(link["path"]).resolve()
                expected_stat = [link[key] for key in ("device", "inode", "bytes", "mtime_ns")]
                digest = binding["detector_binding"]["input_sha256"]["video:" + view]
                require(descriptor["sha256"] == link["sha256"] == digest, "raw source digest edge")
                require(
                    stat_signature(path) == stat_signature(hardlink) == expected_stat,
                    "raw hardlink/stat differs",
                )
                require(
                    descriptor["source_offset_frames"] == chunk["start"],
                    "source frame offset differs",
                )
                require(
                    len(descriptor["frame_bgr_sha256"]) == chunk["stop"] - chunk["start"],
                    "native hash coverage",
                )
                video = dict(path=str(path), inherited_sha256=digest, expected_stat=expected_stat)
                require(
                    view not in self.videos or self.videos[view] == video,
                    "canonical video changed between chunks",
                )
                self.videos[view] = video
                for frame in frames:
                    self.expected[(frame, view)] = descriptor["frame_bgr_sha256"][
                        frame - chunk["start"]
                    ]
                    self.frame_origins[frame] = dict(
                        manifest_path=str(manifest_path),
                        manifest_sha256=chunk["manifest_sha256"],
                        manifest_row=frame - chunk["start"],
                    )
        require(
            set(self.frame_origins) == wanted,
            "selected frames not entirely covered by consumed sources",
        )
        self.captures = {}
        for view, record in self.videos.items():
            cap = cv2.VideoCapture(record["path"])
            require(
                cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 348600,
                "native video header",
            )
            require(abs(cap.get(cv2.CAP_PROP_FPS) - 166) < 1e-6, "native fps differs")
            self.captures[view] = cap

    def seek(self, frame):
        for cap in self.captures.values():
            require(cap.set(cv2.CAP_PROP_POS_FRAMES, frame), "native source seek failed")
            require(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame, "native seek position differs")

    def read(self, frame):
        images = {}
        for view, cap in self.captures.items():
            okay, bgr = cap.read()
            require(okay and bgr.shape == (1200, 1200, 3), "native decode failed")
            require(
                int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1,
                "native decoded position differs",
            )
            require(
                sha_bytes(bgr.tobytes()) == self.expected[(frame, view)],
                f"native BGR digest differs {frame} {view}",
            )
            images[view] = bgr
        return images

    def close(self):
        for cap in self.captures.values():
            cap.release()
        for record in self.videos.values():
            require(
                stat_signature(record["path"]) == record["expected_stat"],
                "canonical source changed during render",
            )


def panel(images, frame):
    composite = np.zeros((640, 1800, 3), np.uint8)
    for i, view in enumerate(VIEWS):
        composite[40:, i * 600 : (i + 1) * 600] = cv2.resize(
            images[view], (600, 600), interpolation=cv2.INTER_AREA
        )
        label = f"{view} | frame {frame} | source {frame / 166:.3f}s"
        cv2.putText(
            composite,
            label,
            (i * 600 + 8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return composite


def review_html(template, still_frames):
    text = (LAB / "review_scene.html").read_text(encoding="utf8")
    text = (
        text.replace("異常報告の直後", "中間")
        .replace("入口→直後", "入口→中間")
        .replace("直後→出口", "中間→出口")
    )
    text = text.replace(
        "初期の 3 時点は既知の短時間の追跡異常を挟む確認候補です。物理的な接触が起きたという意味ではありません。",
        "初期の 3 時点は入口・中間・出口の確認候補です。見やすいフレームへ変更できます。",
    )
    text = text.replace(
        "正確な 1 フレーム送りとクリックには下の原画像を使ってください。",
        "下の静止画は原寸 1200×1200、4 フレーム間隔（約 24 ms）と確認候補の各フレームです。クリックには静止画を使ってください。",
    )
    text = text.replace("−1 frame", "前の静止画").replace("+1 frame", "次の静止画")
    text = text.replace(
        "<h1>", '<p><a href="../index.html" style="color:#9cf">確認一覧へ戻る</a></p><h1>', 1
    )
    text = (
        text.replace("__TRUTH__", json.dumps(template, ensure_ascii=False))
        .replace("__FRAMES__", json.dumps(still_frames))
        .replace("__VIEWS__", json.dumps(VIEWS))
        .replace("__LAST__", str(len(still_frames) - 1))
    )
    text = text.replace(
        "`source ${frame} loading`", "`source ${frame} | ${(frame/166).toFixed(3)} s loading`"
    ).replace("`source ${frame}`", "`source ${frame} | ${(frame/166).toFixed(3)} s`")
    text = text.replace(
        "img.onerror=no;", "img.onerror=()=>no(new Error('画像を読み込めません: '+view+' '+frame));"
    )
    # Reject out-of-order anchors at export, and preserve all ambiguous responses.
    text = text.replace(
        "q('save').onclick=()=>{",
        "q('save').onclick=()=>{const af=truth.anchors.map(a=>a.frame);if(!(af[0]<af[1]&&af[1]<af[2])){alert('入口・中間・出口を時間順に指定してください');return;}",
    )
    require("__TRUTH__" not in text and "__LAST__" not in text, "review UI placeholders remain")
    return text


def verify_video(path, expected_frames, expected_probes):
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), "encoded video cannot open")
    require(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames, "encoded frame count differs"
    )
    require(
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        == (1800, 640),
        "encoded video geometry differs",
    )
    require(abs(cap.get(cv2.CAP_PROP_FPS) - PLAYBACK_FPS) < 1e-6, "encoded fps differs")
    errors, n = {}, 0
    while True:
        okay = cap.grab()
        if not okay:
            break
        if n in expected_probes:
            okay, decoded = cap.retrieve()
            require(okay, "encoded probe decode failed")
            error = float(np.abs(decoded.astype(np.float32) - expected_probes[n]).mean())
            require(error < 8, f"encoded camera content differs: {error}")
            errors[str(n)] = error
        n += 1
    cap.release()
    require(
        n == expected_frames and len(errors) == len(expected_probes),
        "encoded full decode count/probes differ",
    )
    return dict(
        decoded_frame_count=n,
        fps=PLAYBACK_FPS,
        width=1800,
        height=640,
        probe_mean_absolute_bgr_error=errors,
        all_frames_decoded=True,
    )


def render_scene(scene, source, output, ledger_sha, ffmpeg):
    root = output / scene["scene_id"]
    seal_path = root / "summary.json"
    if seal_path.exists():
        old = read_checked(seal_path)
        require(
            old["status"] == "ok" and old["selection_ledger_sha256"] == ledger_sha,
            "existing scene has different selection",
        )
        for name, ref in read_checked(root / "asset_hashes.json").items():
            require(sha_file(root / name) == ref["sha256"], "existing scene asset changed: " + name)
        return old
    require(
        not root.exists(), "incomplete scene directory must be retained; use a fresh output root"
    )
    root.mkdir(parents=True)
    (root / "frames").mkdir()
    first, stop = scene["start_frame"], scene["stop_frame_exclusive"]
    probes = scene["anchor_frames"]
    frames = list(range(first, stop))
    still_frames = sorted(set(range(first, stop, STILL_STRIDE)) | set(probes) | {stop - 1})
    still_set = set(still_frames)
    public = dict(
        schema="blinded_native_source_map_v1",
        scene_id=scene["scene_id"],
        source_fps=166,
        review_clip_range=[first, stop],
        views=VIEWS,
        videos=source.videos,
        source_metadata=source.records,
        frames=[dict(frame=f, **source.frame_origins[f]) for f in frames],
    )
    write_json(root / "source_manifest.json", public)
    manifest_sha = sha_file(root / "source_manifest.json")
    template = make_template(frames, manifest_sha, probes)
    template["review_clip_range"] = [first, stop]
    template["scene_id"] = scene["scene_id"]
    template["anchors"][1]["phase"] = "middle"
    template["contact_occurrence"] = "unknown_pending_human_review"
    template["still_frames_available"] = still_frames
    template["unscored_rule"] = (
        "unreviewed, ambiguous and not_visible are unscored; no automatic continuity labels"
    )
    write_json(root / "truth_template.json", template)
    (root / "index.html").write_text(review_html(template, still_frames), encoding="utf8")
    assets = {}

    def save_image(name, bgr, quality=95):
        extension = Path(name).suffix
        options = (
            [cv2.IMWRITE_JPEG_QUALITY, quality]
            if extension == ".jpg"
            else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
        okay, encoded = cv2.imencode(extension, bgr, options)
        require(okay, "image encoding failed")
        data = encoded.tobytes()
        (root / name).write_bytes(data)
        assets[name] = dict(sha256=sha_bytes(data), size_bytes=len(data))

    video = root / "raw_three_views_slow.mp4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        "1800x640",
        "-framerate",
        str(PLAYBACK_FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-n",
        str(video),
    ]
    expected_probes, contact_rows = {}, []
    start_clock = time.monotonic()
    with (
        (root / "encode.stderr.txt").open("wb") as error_stream,
        (root / "frame_map.csv").open("w", newline="", encoding="utf8") as frame_stream,
    ):
        writer = csv.writer(frame_stream)
        writer.writerow(
            [
                "encoded_frame",
                "native_frame",
                "native_seconds",
                "playback_seconds",
                "still_available",
                *[v + "_native_bgr_sha256" for v in VIEWS],
            ]
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=error_stream,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            source.seek(first)
            for row, frame in enumerate(frames):
                images = source.read(frame)
                composite = panel(images, frame)
                process.stdin.write(composite.tobytes())
                writer.writerow(
                    [
                        row,
                        frame,
                        frame / 166,
                        row / PLAYBACK_FPS,
                        frame in still_set,
                        *[source.expected[(frame, view)] for view in VIEWS],
                    ]
                )
                if frame in still_set:
                    for view in VIEWS:
                        save_image(f"frames/f{frame}_{view}.jpg", images[view])
                if frame in probes:
                    phase = ("entry", "middle", "exit")[probes.index(frame)]
                    save_image(phase + ".jpg", composite)
                    for view in VIEWS:
                        save_image(f"{phase}_{view}.png", images[view])
                    expected_probes[row] = composite
                    contact_rows.append(composite)
            process.stdin.close()
            require(process.wait() == 0, "ffmpeg encoding failed; see encode.stderr.txt")
        except BaseException:
            process.kill()
            process.wait()
            raise
    save_image("entry_middle_exit.jpg", np.concatenate(contact_rows, axis=0))
    qa = verify_video(video, len(frames), expected_probes)
    for name in (
        "raw_three_views_slow.mp4",
        "frame_map.csv",
        "source_manifest.json",
        "truth_template.json",
        "index.html",
        "encode.stderr.txt",
    ):
        path = root / name
        assets[name] = dict(sha256=sha_file(path), size_bytes=path.stat().st_size)
    write_json(root / "asset_hashes.json", assets)
    result = dict(
        schema=STAGE,
        status="ok",
        scene_id=scene["scene_id"],
        selection_ledger_sha256=ledger_sha,
        native_interval=[first, stop],
        anchor_frames=probes,
        video_native_frames=len(frames),
        native_seconds=len(frames) / 166,
        playback_seconds=len(frames) / PLAYBACK_FPS,
        still_stride=STILL_STRIDE,
        full_resolution_stills_per_camera=len(still_frames),
        camera_views=list(VIEWS),
        native_bgr_hashes_verified=len(frames) * 3,
        video_qa=qa,
        elapsed_render_seconds=time.monotonic() - start_clock,
        model_outputs_shown=False,
        human_truth_created=False,
        human_review_status="unreviewed",
        ambiguous_responses="unscored",
        assets_manifest_sha256=sha_file(root / "asset_hashes.json"),
    )
    write_json(seal_path, result)
    print(
        json.dumps(
            dict(
                scene=scene["scene_id"],
                frames=len(frames),
                seconds=result["elapsed_render_seconds"],
                status="ok",
            )
        ),
        flush=True,
    )
    return result


def index_html(results):
    rows = []
    for row in results:
        sid = html.escape(row["scene_id"])
        first, stop = row["native_interval"]
        rows.append(
            f'<tr><td><a href="{sid}/index.html">{sid} を確認</a></td><td>{first}–{stop - 1}</td><td>{row["native_seconds"]:.2f} 秒</td><td><a href="{sid}/raw_three_views_slow.mp4">3 カメラ動画</a></td><td><a href="{sid}/entry_middle_exit.jpg">入口・中間・出口</a></td></tr>'
        )
    return (
        """<!doctype html><html lang="ja"><meta charset="utf-8"><title>匿名 3 個体の連続性確認</title>
<style>body{font:17px system-ui;margin:32px;max-width:1400px;background:#17191c;color:#eee;line-height:1.65}a{color:#9cf}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #556;text-align:left}p{max-width:1100px}</style>
<h1>匿名 3 個体の連続性確認</h1><p>各場面の入口で、3 匹に任意の A・B・C を付け、動画を前後に見て同じ魚を中間・出口まで追ってください。口を各時点でクリックして記録します。オス 2 匹も対象です。匿名名は場面ごとにリセットします。性別は判断できる場合だけ入力してください。</p>
<p>動画は全フレームを保持し、元映像 166 fps を 20 fps で再生します（8.3 倍スロー）。原寸静止画は 4 フレーム間隔と入口・中間・出口にあり、確認時点の変更も可能です。分からない対応は「曖昧」「見えない」で保存し、未採点に残してください。各場面で JSON をダウンロードすると、その場面の入力を保存できます。</p>
<p>口の位置は最低限、3 匹 × 3 時点を各 1 つの見やすい視点で記録します。確信がないときは無理に対応を決める必要はありません。</p>
<table><thead><tr><th>場面</th><th>元フレーム</th><th>元映像の長さ</th><th>動画</th><th>確認画像</th></tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table></html>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(WORK), "output must remain inside work/three_body")
    ledger = read_checked(args.scenes)
    ledger_sha = sha_file(args.scenes)
    require(
        ledger["fps"] == 166 and ledger["interval"] == [66400, 116200],
        "fixed five-minute scope required",
    )
    scenes = ledger["scenes"]
    require(
        scenes and len({s["scene_id"] for s in scenes}) == len(scenes),
        "unique nonempty scene list required",
    )
    for scene in scenes:
        require(re.fullmatch(r"scene_\d{3}", scene["scene_id"]), "anonymous scene naming required")
        first, stop = scene["start_frame"], scene["stop_frame_exclusive"]
        require(66400 <= first < stop <= 116200, "scene outside scientific interval")
        probes = scene["anchor_frames"]
        require(
            len(probes) == 3
            and probes == sorted(set(probes))
            and all(first <= f < stop for f in probes),
            "three chronological in-clip probes required",
        )
    ffmpeg = shutil.which("ffmpeg")
    require(ffmpeg is not None, "ffmpeg required")
    source = Source({f for s in scenes for f in range(s["start_frame"], s["stop_frame_exclusive"])})
    cv2.setNumThreads(2)
    try:
        with provenance(STAGE, output, name="review") as record:
            record.input("selection_ledger", args.scenes.resolve(), sha256=ledger_sha)
            record.input("renderer", Path(__file__).resolve(), sha256=sha_file(__file__))
            for name in ("review_scene.html", "review_truth.py"):
                record.input(name, LAB / name, sha256=sha_file(LAB / name))
            for path, digest in source.records.items():
                record.input(path, path, sha256=digest)
            for view, reference in source.videos.items():
                record.input(view, reference["path"], sha256=reference["inherited_sha256"])
            record.param("tracking_performed", False)
            record.param(
                "native_frame_sha_validation",
                "every rendered frame in all three cameras against consumed staged manifest",
            )
            record.param(
                "whole_video_sha_validation",
                "inherited registration; current file identity/stat and exact decoded BGR checked",
            )
            record.param("video_encoder", "libx264 ultrafast CRF18 yuv420p CPU threads2")
            record.param("selection_and_methods_blinded", True)
            results = [render_scene(s, source, output, ledger_sha, ffmpeg) for s in scenes]
            source.close()
            require(sha_file(args.scenes) == ledger_sha, "selection ledger changed while rendering")
            (output / "index.html").write_text(index_html(results), encoding="utf8")
            summary = dict(
                schema=STAGE,
                status="ok",
                scene_count=len(results),
                all_selected_scenes_rendered=True,
                scene_ids=[r["scene_id"] for r in results],
                total_native_video_frames=sum(r["video_native_frames"] for r in results),
                source_fps=166,
                playback_fps=PLAYBACK_FPS,
                human_review_status="unreviewed",
                human_truth_created=False,
                model_outputs_shown=False,
                results=results,
            )
            write_json(output / "summary.json", summary)
            with (output / "response_sheet.csv").open("w", newline="", encoding="utf8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "scene_id",
                        "anonymous_fish",
                        "sex",
                        "entry_to_middle",
                        "middle_to_exit",
                        "basis",
                        "notes",
                    ]
                )
                for result in results:
                    for animal in ("A", "B", "C"):
                        writer.writerow(
                            [
                                result["scene_id"],
                                animal,
                                "unknown",
                                "unreviewed",
                                "unreviewed",
                                "unreviewed",
                                "",
                            ]
                        )
            for name in ("index.html", "summary.json", "response_sheet.csv"):
                record.output(name, output / name, sha256=sha_file(output / name))
            for row in results:
                path = output / row["scene_id"] / "summary.json"
                record.output(row["scene_id"], path, sha256=sha_file(path))
            print(
                json.dumps(
                    dict(
                        status="ok",
                        scenes=len(results),
                        frames=summary["total_native_video_frames"],
                    )
                ),
                flush=True,
            )
    finally:
        source.close()


if __name__ == "__main__":
    main()
