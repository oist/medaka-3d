"""Export native-frame, three-view review clips without adjudicating identities."""

import html
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from run_pipeline import HERE, settings, sha256, write_json

from medaka3d.provenance import provenance

VIEWS = ("top_cam", "side1_cam", "side2_cam")
VIDEO_NAMES = ("cam0_top.mp4", "cam1_side.mp4", "cam2_side.mp4")


def contact_tile(resized, frame, when, center):
    """Place labels above the full image; never obscure source pixels."""
    panel = resized.shape[1]
    contact = np.zeros((resized.shape[0] + 60, panel, 3), np.uint8)
    contact[60:] = resized
    for text, y in (
        (f"Native {frame}; delta {when - center:+.3f} s", 25),
        (f"Top clock {when:.6f} s", 50),
    ):
        cv2.putText(contact, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return contact


def export_clip(example, videos, times, destination, policy, ffmpeg):
    """Keep every native frame and publish the acquisition/playback mapping."""
    boundary = example["boundary_frame"]
    native_times = times["top_cam"]
    center = native_times[boundary]
    first = int(np.searchsorted(native_times, center - policy["context_seconds"]))
    stop = int(np.searchsorted(native_times, center + policy["context_seconds"], side="right"))
    panel = policy["panel_pixels"]
    banner = 70
    width, height = panel * len(VIEWS), panel + banner
    fps = policy["playback_frames_per_second"]
    video_path = destination / f"{example['name']}.mp4"
    captures = [cv2.VideoCapture(str(path)) for path in videos]
    frames, contact_frames = [], {}
    contact_offsets = (-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0)
    contact_indices = {
        int(np.searchsorted(native_times, center + offset)) for offset in contact_offsets
    }
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    try:
        for capture in captures:
            if not capture.isOpened() or capture.get(cv2.CAP_PROP_FRAME_COUNT) != len(native_times):
                raise ValueError("Source video does not match the native frame grid")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise ValueError("Source seek failed")
        with (destination / f"{example['name']}_encoder.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                for frame in range(first, stop):
                    canvas = np.zeros((height, width, 3), np.uint8)
                    for column, (view, capture) in enumerate(zip(VIEWS, captures, strict=True)):
                        ok, image = capture.read()
                        if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                            raise ValueError(f"Source decode/cursor failure: {view} frame {frame}")
                        resized = cv2.resize(image, (panel, panel), interpolation=cv2.INTER_AREA)
                        if view == "top_cam" and frame in contact_indices:
                            contact_frames[frame] = contact_tile(
                                resized, frame, native_times[frame], center
                            )
                        canvas[banner:, column * panel : (column + 1) * panel] = resized
                        cv2.putText(
                            canvas,
                            f"{view}: {times[view][frame]:.6f} s",
                            (column * panel + 10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            1,
                        )
                    cv2.putText(
                        canvas,
                        f"Native frame {frame}; boundary {boundary}; "
                        f"delta {native_times[frame] - center:+.6f} s; SLOW PLAYBACK",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (90, 230, 255),
                        1,
                    )
                    process.stdin.write(canvas.tobytes())
                    frames.append(
                        {
                            "output_frame": frame - first,
                            "playback_time_s": (frame - first) / fps,
                            "native_frame": frame,
                            **{f"{view}_time_s": float(times[view][frame]) for view in VIEWS},
                        }
                    )
            finally:
                process.stdin.close()
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"Video encoder failed; inspect {log.name}")
    finally:
        for capture in captures:
            capture.release()
    output_capture = cv2.VideoCapture(str(video_path))
    try:
        if (
            not output_capture.isOpened()
            or int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT)) != stop - first
        ):
            raise ValueError("Encoded frame count differs from the declared interval")
        if output_capture.get(cv2.CAP_PROP_FPS) != fps:
            raise ValueError("Encoded playback frame rate differs")
    finally:
        output_capture.release()
    pd.DataFrame(frames).to_csv(destination / f"{example['name']}_frames.csv", index=False)
    ordered = [contact_frames[key] for key in sorted(contact_frames)]
    contact_sheet = np.vstack([np.hstack(ordered[index : index + 3]) for index in range(0, 9, 3)])
    if not cv2.imwrite(str(destination / f"{example['name']}_top_contact.png"), contact_sheet):
        raise OSError("Could not save contact sheet")
    return {
        **example,
        "first_native_frame": first,
        "stop_native_frame_exclusive": stop,
        "native_frame_count": stop - first,
        "first_acquisition_s": float(native_times[first]),
        "last_acquisition_s": float(native_times[stop - 1]),
        "playback_fps": fps,
        "physical_identity_review": "unadjudicated",
        "video": video_path.name,
    }


def main():
    os.environ["MEDAKA3D_P11_CONFIG"] = str(HERE / "config_row_mapping_v1.toml")
    cfg, paths, _, repair = settings()
    policy_path = HERE / "config_video_review_v1.toml"
    policy = tomllib.loads(policy_path.read_text())
    destination = repair / policy["version"]
    destination.mkdir(exist_ok=False)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("The review export requires an installed ffmpeg executable")
    ledger = repair / "join_ledger_v2" / policy["source_variant"] / f"lam{policy['lambda_mm']:g}"
    edges_path = ledger / "edges.csv"
    edges = pd.read_csv(edges_path).set_index("edge_id")
    inventory = repair / "source_frame_preview_v1" / "sources.json"
    sources = {row["view"]: row for row in json.loads(inventory.read_text())}
    videos = [
        paths.raw_data_dir / cfg["experiment"] / view / filename
        for view, filename in zip(VIEWS, VIDEO_NAMES, strict=True)
    ]
    records = []
    with provenance("p11_video_review_pack", destination) as rec:
        for name, source in (
            ("code", Path(__file__)),
            ("config", policy_path),
            ("ledger_edges", edges_path),
            ("source_inventory", inventory),
            ("native_timing", repair / "native_timing.npz"),
        ):
            rec.input(name, source, sha256=sha256(source))
        for source in (Path(__file__), policy_path):
            shutil.copy2(source, destination / source.name)
        rec.param("scope", policy["note"])
        rec.param(
            "camera_alignment",
            "Inherited equal native index; hardware sync not independently certified",
        )
        for view, video in zip(VIEWS, videos, strict=True):
            actual = sha256(video)
            if actual != sources[view]["video_sha256"]:
                raise ValueError(f"Pinned source video changed: {view}")
            rec.input(view, video, sha256=actual)
            print(f"Verified source video: {view}", flush=True)
        with np.load(repair / "native_timing.npz") as data:
            times = {view: data[view] for view in VIEWS}
        for example in policy["examples"]:
            selected = edges.loc[example["edge_ids"]]
            if not selected.after_frame.eq(example["boundary_frame"]).all():
                raise ValueError("Review example no longer matches its pinned ledger")
            record = export_clip(example, videos, times, destination, policy, ffmpeg)
            record["reference_endpoint_outcomes"] = selected.outcome.to_dict()
            selected.to_csv(destination / f"{example['name']}_edges.csv")
            records.append(record)
            print(f"Exported native-frame clip: {example['name']}", flush=True)
        write_json(destination / "examples.json", records)
        blocks = []
        for record in records:
            name = record["name"]
            blocks.append(
                f"<section><h2>{html.escape(name.replace('_', ' '))}</h2>"
                f"<p>{html.escape(record['selection'])}</p>"
                f'<video controls preload="metadata" src="{record["video"]}"></video>'
                f'<p><a href="{name}_frames.csv">Native/playback frame mapping</a> | '
                f'<a href="{name}_edges.csv">Reference endpoint evidence</a> | '
                f'<a href="{name}_top_contact.png">Top-view contact sheet</a></p></section>'
            )
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>P-1-1 connection review</title>"
        )
        page += (
            "<style>body{font:17px system-ui;max-width:1500px;margin:32px auto;"
            "padding:0 24px;color:#18232d}video{width:100%}section{margin:36px 0}"
            "a{color:#176f76}</style>"
        )
        page += "<h1>P-1-1 connection review: baseline lambda 0.767</h1>"
        page += "<p>Every native frame is retained. Playback is slowed to 25 frames per second. "
        page += "Read acquisition time from the overlay or frame mapping, not the player clock. "
        page += "Views retain the existing equal-native-index alignment; hardware synchronization "
        page += "has not been independently re-established.</p>"
        page += "<p>These purposively selected examples are unadjudicated. Endpoint agreement with "
        page += "the reviewed reference is not a new physical-identity annotation. No train/test "
        page += (
            "partition or event count is established. The complete source videos remain available "
        )
        page += "for context beyond these four-second excerpts.</p>"
        page += "".join(blocks) + "</html>"
        (destination / "index.html").write_text(page, encoding="utf-8")
        for artifact in sorted(destination.iterdir()):
            if artifact.suffix in (".csv", ".json", ".mp4", ".html", ".png"):
                rec.output(artifact.name, artifact, sha256=sha256(artifact))
    print(destination / "index.html", flush=True)


if __name__ == "__main__":
    main()
