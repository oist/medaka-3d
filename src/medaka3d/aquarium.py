"""Replay an existing C34 correspondence as a diagnostic aquarium CG.

This is visualization, not a tracking algorithm. Slot assignment and recorded
pectoral positions remain unchanged. Head and tail are triangulated from the
same stored detection rows; absent data is never interpolated. Native-rate
motion events are retained even when the movie samples at a lower frame rate.

Run ``python -m medaka3d.aquarium --help`` for the explicit input paths.
"""

# Typographic multiplication signs and en dashes in the player page are intentional.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from medaka3d.calib import load_rig, triangulate
from medaka3d.provenance import provenance
from medaka3d.types import FPS, VIEWS

NODES = ("mouth", "pectoral", "tail")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gather_detections(
    cache: Path, experiment: str, view: str, frames: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    """Use frame-local row numbers, strictly checking identity and row bounds."""
    with np.load(cache, allow_pickle=False) as data:
        if str(data["experiment"]) != experiment or str(data["view"]) != view:
            raise ValueError(f"Detection cache identity mismatch: {cache}")
        if tuple(data["nodes"]) != NODES:
            raise ValueError(f"Unexpected skeleton order: {cache}")
        frame_idx = data["frame_idx"]
        n_frames = int(data["n_frames"])
        if frame_idx.ndim != 1 or np.any(np.diff(frame_idx) < 0):
            raise ValueError(f"Unsorted cache frames: {cache}")
        if np.any(frame_idx < 0) or np.any(frame_idx >= n_frames):
            raise ValueError(f"Invalid cache frame index: {cache}")
        if np.any(frames < 0) or np.any(frames >= n_frames):
            raise ValueError(f"Requested frames outside cache: {cache}")
        if rows.shape != (len(frames), 3) or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("Source rows must have integer shape (frames, 3)")
        counts = np.bincount(frame_idx, minlength=n_frames)
        offsets = np.r_[0, np.cumsum(counts)]
        present = rows >= 0
        if np.any(present & (rows >= counts[frames, None])):
            raise ValueError(f"Stored source row exceeds frame detection count: {cache}")
        points = data["points"]
        if points.shape != (len(frame_idx), 3, 2):
            raise ValueError(f"Invalid points shape: {cache}")
        flat = offsets[frames, None] + rows
        result = np.full((len(frames), 3, 3, 2), np.nan)
        result[present] = points[flat[present]]
    return result


def reconstruct_pose(
    source: np.ndarray,
    anchors: np.ndarray,
    frames: np.ndarray,
    detections_dir: Path,
    experiment: str,
    calibration: Path,
):
    """Render source-row geometry while retaining the original pectoral anchor."""
    rig = load_rig(calibration)
    origins, directions, valid = [], [], []
    paths = []
    for vi, view in enumerate(VIEWS):
        path = detections_dir / f"{experiment}__{view}.npz"
        uv = gather_detections(path, experiment, view, frames, source[:, :, vi])
        origin, direction = rig.camera(view).rays(uv)
        origins.append(origin)
        directions.append(direction)
        valid.append(np.isfinite(origin).all(-1) & np.isfinite(direction).all(-1))
        paths.append(path)
    pose, residual = triangulate(
        np.stack(origins, axis=-2), np.stack(directions, axis=-2), np.stack(valid, axis=-1)
    )
    # This must not undo upstream ray rejection or silently fill a missing anchor.
    reconstructed_pectoral = pose[:, :, 1].copy()
    pose[:, :, 1] = anchors
    pose[~np.isfinite(anchors).all(-1)] = np.nan
    return pose, residual, reconstructed_pectoral, paths


def detect_events(
    anchors: np.ndarray, frames: np.ndarray, source_fps: float, jump_mm: float = 4.0
) -> list[dict]:
    """Mark adjacent-frame jumps and post-gap reappearances separately.

    A jump is a diagnostic candidate, never an identity-swap assertion. The
    threshold applies to native adjacent samples, not movie frame intervals.
    """
    anchors = np.asarray(anchors)
    frames = np.asarray(frames)
    if anchors.shape != (len(frames), 3, 3) or len(frames) == 0:
        raise ValueError("Expected nonempty anchors (frames, 3 slots, 3 coordinates)")
    if np.any(np.diff(frames) != 1):
        raise ValueError("Motion diagnostics require contiguous native frames")
    if not np.isfinite(source_fps) or source_fps <= 0 or not np.isfinite(jump_mm) or jump_mm <= 0:
        raise ValueError("FPS and jump threshold must be positive and finite")
    events = []
    for slot in range(3):
        last = None
        for i in range(len(frames)):
            if not np.isfinite(anchors[i, slot]).all():
                continue
            if last is not None:
                gap = i - last - 1
                distance = float(np.linalg.norm(anchors[i, slot] - anchors[last, slot]))
                if gap or distance > jump_mm:
                    events.append(
                        {
                            "type": "reappearance" if gap else "jump",
                            "slot": slot,
                            "frame": int(frames[i]),
                            "previous_frame": int(frames[last]),
                            "clip_seconds": float((frames[i] - frames[0]) / source_fps),
                            "source_seconds": float(frames[i] / source_fps),
                            "distance_mm": distance,
                            "missing_frames": gap,
                            "start_xyz": anchors[last, slot].tolist(),
                            "end_xyz": anchors[i, slot].tolist(),
                        }
                    )
            last = i
    return sorted(events, key=lambda event: (event["frame"], event["slot"]))


def movie_indices(n_native: int, source_fps: float, output_fps: float) -> np.ndarray:
    """Nearest native sample at each output timestamp; no coordinate interpolation."""
    if n_native < 1 or not 0 < output_fps <= source_fps:
        raise ValueError("Require samples and 0 < output FPS <= source FPS")
    count = round(n_native / source_fps * output_fps)
    if count < 1:
        raise ValueError("Window shorter than one movie frame")
    return np.minimum(np.rint(np.arange(count) * source_fps / output_fps).astype(int), n_native - 1)


def trail_at(
    anchors: np.ndarray, index: int, length: int, events: list[dict], first_frame: int
) -> list[np.ndarray]:
    """Keep native samples and explicit breaks before suspicious jumps."""
    lo = max(0, index - length + 1)
    trails = []
    for slot in range(3):
        points = anchors[lo : index + 1, slot].copy()
        breaks = {
            e["frame"] - first_frame - lo
            for e in events
            if e["slot"] == slot and lo < e["frame"] - first_frame <= index
        }
        output = []
        for j, point in enumerate(points):
            if j in breaks:
                output.append(np.full(3, np.nan))
            output.append(point)
        trails.append(np.asarray(output))
    return trails


def write_review(path: Path, metadata: dict) -> None:
    """A local, network-free player with event seeking and playback speed."""
    rows = []
    for event in metadata["events"]:
        label = "sudden-move candidate" if event["type"] == "jump" else "reappearance after dropout"
        rows.append(
            f'<button class="event" data-time="{max(0, event["clip_seconds"] - 0.6):.4f}">'
            f"<b>+{event['clip_seconds']:.3f} s</b> · ID {event['slot']} · {label}"
            f" · {event['distance_mm']:.1f} mm · <small>f {event['frame']}</small></button>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medaka tracking / Aquarium CG</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0c1422;color:#e9f1fa;
font:15px system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}h1{font-size:26px;margin:7px 0 12px}
.eyebrow{color:#75cddb;font-size:12px;letter-spacing:.18em}p{color:#adbdcf;line-height:1.7}
video{display:block;width:100%;background:#080e18;border:1px solid #294257;border-radius:12px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}
button,select,a{font:inherit}button,select{background:#17283b;color:#eef7ff;
border:1px solid #345069;
border-radius:7px;padding:9px 14px;cursor:pointer}button:hover{background:#243e56}
.events{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:7px}
.event{text-align:left;font-size:13px}.event b{color:#ffbd69}small{color:#9cafc2}
summary{cursor:pointer;margin:22px 0 12px}a{color:#8ed5df}code{overflow-wrap:anywhere}
</style><main><div class="eyebrow">TRACKED MOTION · DIAGNOSTIC REPLAY</div>
<h1>The medaka tank, replayed from tracking data</h1>
<p>__RANGE__ / __EXPERIMENT__ / __TITLE__<br>
Colours are tracking slot IDs 0, 1, 2. Oblique 3D, top and side views are shown together.</p>
<video id="video" controls preload="metadata" poster="preview.png" src="aquarium.mp4"></video>
<div class="controls"><button id="play">Play / pause</button>
<button id="back">-1 frame</button><button id="next">+1 frame</button>
<label>Playback speed <select id="speed"><option value=".1">0.1×</option>
<option value=".25">0.25×</option>
<option value=".5">0.5×</option><option value="1" selected>1×</option></select></label>
<button id="loop">Loop OFF</button><span id="time"></span></div>
<p>Sudden moves of more than __JUMP__ mm in one frame, and reappearances after dropout, are
detected over every frame at the original __SOURCE_FPS__ fps.
Red dotted lines are sudden-move candidates, not confirmed swaps. Display is __FPS__ fps,
with no coordinate interpolation or smoothing.
Even brief candidates are shown for __HOLD__ s. Dropouts are shown as disappearance or as a
partial skeleton.
Body thickness and fins are simplified CG shapes. Mouth, chest and tail orientation comes from
the detected points; there is no added tail animation.</p>
<details open><summary>Candidate list: click to play from just before</summary>
<div class="events">__EVENTS__</div></details>
<details><summary>Inputs and reproducibility</summary><p><code>__SOURCE__</code><br>
Frame range __FRAMES__ (end exclusive). Tank interior __TANK__ mm.
Colours are not a guarantee of biological sex or individual identity.<br>
<a href="metadata.json">Interval, candidates, settings</a> ·
<a href="native_pose.npz">3D pose at the source frame rate</a> ·
<a href="aquarium.provenance.json">Input and code hashes</a></p></details>
</main><script>
const v=document.getElementById('video'), fps=__FPS__;
document.getElementById('play').onclick=()=>v.paused?v.play():v.pause();
document.getElementById('back').onclick=()=>{v.pause();v.currentTime=Math.max(0,v.currentTime-1/fps)};
document.getElementById('next').onclick=()=>{v.pause();v.currentTime=Math.min(v.duration,v.currentTime+1/fps)};
document.getElementById('speed').onchange=e=>v.playbackRate=Number(e.target.value);
document.getElementById('loop').onclick=e=>{
v.loop=!v.loop;e.target.textContent='Loop '+(v.loop?'ON':'OFF')};
document.querySelectorAll('.event').forEach(b=>b.onclick=()=>{v.currentTime=Number(b.dataset.time);v.play()});
v.ontimeupdate=()=>document.getElementById('time').textContent=
v.currentTime.toFixed(2)+' / '+v.duration.toFixed(2)+' s';
</script></html>"""
    replacements = {
        "__RANGE__": f"{metadata['duration_seconds']:.0f} s / source video "
        f"{metadata['start_frame'] / metadata['source_fps']:.3f}"
        f"–{metadata['stop_frame_exclusive'] / metadata['source_fps']:.3f} s",
        "__EXPERIMENT__": html.escape(metadata["experiment"]),
        "__TITLE__": html.escape(metadata["title"]),
        "__SOURCE_FPS__": str(metadata["source_fps"]),
        "__TANK__": " × ".join(f"{d:g}" for d in np.diff(metadata["tank_bounds_mm"], axis=0)[0]),
        "__SOURCE__": html.escape(metadata["source"]),
        "__JUMP__": str(metadata["jump_threshold_mm"]),
        "__FPS__": str(metadata["output_fps"]),
        "__HOLD__": str(metadata["alert_hold_seconds"]),
        "__FRAMES__": f"{metadata['start_frame']}:{metadata['stop_frame_exclusive']}",
        "__EVENTS__": "\n".join(rows) or "No candidates exceed the threshold in this interval.",
    }
    for key, value in replacements.items():
        page = page.replace(key, value)
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--source-fps", type=float, default=FPS)
    parser.add_argument("--jump-mm", type=float, default=4.0)
    parser.add_argument("--trail-seconds", type=float, default=1.0)
    parser.add_argument("--alert-hold-seconds", type=float, default=0.7)
    parser.add_argument("--title", default="Medaka / C34 lambda 0.767")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("seconds", "source_fps", "fps", "jump_mm", "trail_seconds", "alert_hold_seconds"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg must be installed and on PATH")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; use a new directory to preserve old renders")
    args.output.mkdir(parents=True, exist_ok=True)
    with provenance("aquarium", args.output) as record:
        record.params_from(args)
        for name, path in (
            ("tracks", args.tracks),
            ("renderer_driver", Path(__file__)),
            ("renderer", Path(__file__).with_name("aquarium_render.py")),
        ):
            record.input(name, path, sha256=sha256(path))
        manifest_path = args.calibration / "calibration_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record.input("calibration_manifest", manifest_path, sha256=sha256(manifest_path))
        for entry in manifest["model_files"]:
            path = args.calibration / entry["path"]
            digest = sha256(path)
            if digest != entry["sha256"]:
                raise ValueError(f"Calibration hash mismatch: {path}")
            record.input(f"calibration_{entry['camera']}", path, sha256=digest)
        physical = manifest["physical_geometry"]
        bounds = np.array(
            [
                [0.0, 0.0, 0.0],
                [
                    physical["inside_x_span_mm"],
                    physical["inside_y_span_mm"],
                    physical["water_height_from_inside_floor_mm"],
                ],
            ]
        )
        n_native = round(args.seconds * args.source_fps)
        frames = np.arange(args.start_frame, args.start_frame + n_native)
        indices = movie_indices(n_native, args.source_fps, args.fps)
        with np.load(args.tracks, allow_pickle=False) as data:
            if args.start_frame < 0 or frames[-1] >= len(data["xyz"]):
                raise ValueError("Requested window outside tracks")
            anchors = data["xyz"][frames]
            source = data["source"][frames]
            slot_track = data["slot_track"][frames]
        if anchors.shape != (n_native, 3, 3) or source.shape != (n_native, 3, 3):
            raise ValueError("Expected C34 xyz/source shape (frames, slots, 3)")
        print(f"Reconstructing {n_native} native poses from stored source rows", flush=True)
        pose, residual, chest, paths = reconstruct_pose(
            source, anchors, frames, args.detections_dir, args.experiment, args.calibration
        )
        for view, path in zip(VIEWS, paths, strict=True):
            record.input(f"detections_{view}", path, sha256=sha256(path))
        events = detect_events(anchors, frames, args.source_fps, args.jump_mm)
        pose_path = args.output / "native_pose.npz"
        np.savez_compressed(
            pose_path,
            frames=frames,
            pose=pose,
            anchors=anchors,
            source=source,
            slot_track=slot_track,
            residual_mm=residual,
            reconstructed_pectoral=chest,
            source_fps=args.source_fps,
            nodes=np.array(NODES),
            movie_native_indices=indices,
        )
        delta = np.linalg.norm(chest - anchors, axis=-1)
        metadata = {
            "experiment": args.experiment,
            "source": str(args.tracks.resolve()),
            "title": args.title,
            "start_frame": int(frames[0]),
            "stop_frame_exclusive": int(frames[-1] + 1),
            "source_fps": args.source_fps,
            "output_fps": args.fps,
            "duration_seconds": len(indices) / args.fps,
            "native_frames": n_native,
            "output_frames": len(indices),
            "jump_threshold_mm": args.jump_mm,
            "alert_hold_seconds": args.alert_hold_seconds,
            "trail_seconds": args.trail_seconds,
            "tank_bounds_mm": bounds.tolist(),
            "full_pose_fraction_per_slot": np.isfinite(pose).all(axis=(2, 3)).mean(0).tolist(),
            "missing_anchor_frames_per_slot": (~np.isfinite(anchors).all(-1)).sum(0).tolist(),
            "anchor_retriangulation_max_difference_mm": float(np.nanmax(delta)),
            "anchor_outside_tank_frames_per_slot": (
                ((anchors < bounds[0]) | (anchors > bounds[1])).any(-1)
                & np.isfinite(anchors).all(-1)
            )
            .sum(0)
            .tolist(),
            "event_counts": dict(Counter(e["type"] for e in events)),
            "events": events,
            "semantics": {
                "identity": "C34 slot columns, not verified biological identities",
                "anchor": "Original stored pectoral xyz, including gaps and out-of-tank values",
                "heading": "Mouth/tail triangulated from stored source rows, without reassignment",
                "shape": "Illustrative body width/fins; no artificial tail oscillation",
                "sampling": "Nearest native sample, no spatial or temporal interpolation",
                "alerts": "Computed at native rate; held for visibility; not confirmed swaps",
                "limitations": "30 fps cannot show every 166 fps pose; native_pose.npz retains all",
            },
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Native diagnostic candidates: {metadata['event_counts']}", flush=True)
        from medaka3d.aquarium_render import render_frame

        movie_path = args.output / "aquarium.mp4"
        width, height = 1600, 900
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(args.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie_path),
        ]
        record.param("ffmpeg_command", command)
        with (args.output / "ffmpeg.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for output_index, i in enumerate(indices):
                    active = [
                        event
                        for event in events
                        if 0
                        <= frames[i] - event["frame"]
                        < args.alert_hold_seconds * args.source_fps
                    ]
                    trails = trail_at(
                        anchors,
                        int(i),
                        round(args.trail_seconds * args.source_fps),
                        events,
                        int(frames[0]),
                    )
                    image = render_frame(
                        pose[i],
                        trails,
                        frame=int(frames[i]),
                        source_fps=args.source_fps,
                        elapsed_seconds=output_index / args.fps,
                        duration_seconds=len(indices) / args.fps,
                        alerts=active,
                        title=args.title,
                        width=width,
                        height=height,
                        tank_bounds=bounds,
                    )
                    if image.shape != (height, width, 3) or image.dtype != np.uint8:
                        raise ValueError("Renderer did not return the expected BGR frame")
                    process.stdin.write(image.tobytes())
                    if output_index == 0:
                        cv2.imwrite(str(args.output / "preview.png"), image)
                    if output_index % 90 == 0:
                        print(f"Rendered {output_index}/{len(indices)} frames", flush=True)
                process.stdin.close()
                if process.wait(timeout=120) != 0:
                    raise RuntimeError("ffmpeg failed; inspect ffmpeg.log")
            except BaseException:
                process.kill()
                process.wait()
                raise
        write_review(args.output / "index.html", metadata)
        for path in args.output.iterdir():
            if path.is_file():
                record.output(path.name, path, sha256=sha256(path))
        record.param("diagnostic_only", True)
        print(f"Complete: {movie_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
