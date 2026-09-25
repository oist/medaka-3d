"""Reproducible short-window recovery experiment against seven frozen C34 arms.

Only the first baseline frame initializes the candidate. Later baseline poses
are read after tracking for shared evaluation and previews. This deliberately
optimistic initialization is not an end-to-end or full-recording benchmark.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from medaka3d.calib import load_rig, triangulate
from medaka3d.eval.generative import (
    match_mouth_anchors,
    mouth_reprojection_summary,
    observed_motion_summary,
    score_female_identity,
    skeleton_shape_summary,
)
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

# Local experiments are executable modules, not production package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from observations import EvidenceConfig, SceneEvidence

EXPERIMENT = "20251225_K_1_2_FCMC"
NODES = ("mouth", "pectoral", "tail")
BASELINE_DIR = Path("work/c34_microflip") / f"{EXPERIMENT}__chain-c28-c33-c30v4-c34"
CALIBRATION_DIR = Path(
    "calibration/fixed_refractive_192x192x186/"
    "phase1-calibration-preserve-20260802T180200JST-60e13a7746d1/calibration"
)
BACKGROUND_FRAMES = (20000, 40000, 60000, 80000, 100000, 160000, 220000, 280000, 340000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedDetections:
    """Strict reader for the existing flat untracked c01 schema."""

    def __init__(self, root: Path, experiment: str):
        self.data = {}
        self.paths = {}
        self.offsets = {}
        for view in VIEWS:
            path = root / "work/c01_s16_predictions/bottomup" / f"{experiment}__{view}.npz"
            with np.load(path, allow_pickle=False) as arrays:
                data = {key: arrays[key] for key in arrays.files}
            if str(data["experiment"]) != experiment or str(data["view"]) != view:
                raise ValueError(f"cache identity mismatch: {path}")
            if tuple(data["nodes"]) != NODES:
                raise ValueError(f"cache node order differs from {NODES}: {path}")
            if np.any(np.diff(data["frame_idx"]) < 0):
                raise ValueError(f"cache frames are not sorted: {path}")
            self.data[view], self.paths[view] = data, path
            count = np.bincount(data["frame_idx"], minlength=int(data["n_frames"]))
            self.offsets[view] = np.r_[0, np.cumsum(count)]
        counts = {int(data["n_frames"]) for data in self.data.values()}
        if len(counts) != 1:
            raise ValueError("cache frame counts disagree between views")
        self.n_frames = counts.pop()

    def points(self, view: str, frame: int) -> np.ndarray:
        data = self.data[view]
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame outside cache: {frame}")
        lo, hi = self.offsets[view][frame:frame + 2]
        return data["points"][lo:hi]

    def mouth(self, view: str, frame: int) -> np.ndarray:
        return self.points(view, frame)[:, 0]


def skeleton_from_sources(rig, detections: CachedDetections, frames, source):
    """Reconstruct each node under already-chosen source rows, without relinking."""
    source = np.asarray(source)
    uv = np.full((len(frames), 3, 3, 3, 2), np.nan)  # t, fish, view, node, xy
    for vi, view in enumerate(VIEWS):
        for t, frame in enumerate(frames):
            points = detections.points(view, int(frame))
            for fish in range(3):
                row = int(source[t, fish, vi])
                if row >= 0:
                    if row >= len(points):
                        raise ValueError(f"source row {row} out of range at {view}/{frame}")
                    uv[t, fish, vi] = points[row]
    origins, directions, valid = [], [], []
    for vi, view in enumerate(VIEWS):
        origin, direction = rig.camera(view).rays(uv[:, :, vi])
        origins.append(origin)
        directions.append(direction)
        valid.append(np.isfinite(origin).all(axis=-1) & np.isfinite(direction).all(axis=-1))
    held = np.stack(valid, axis=-1)  # t, fish, node, view
    xyz, residual = triangulate(np.stack(origins, axis=-2),
                                np.stack(directions, axis=-2), valid=held)
    return xyz, (held.sum(axis=-1) >= 2) & np.isfinite(xyz).all(axis=-1), residual


def seek_read(capture, frame: int, label: str) -> np.ndarray:
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
        raise RuntimeError(f"could not seek {label} to {frame}")
    ok, bgr = capture.read()
    position = round(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if not ok or position != frame + 1:
        raise RuntimeError(f"video frame alignment failed for {label}/{frame}: {position}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def open_videos(raw_root: Path, experiment: str, n_frames: int):
    paths, captures, metadata = {}, {}, {}
    try:
        for view in VIEWS:
            files = sorted((raw_root / experiment / view).glob("*.mp4"))
            if len(files) != 1:
                raise ValueError(f"expected one raw mp4 under {raw_root / experiment / view}")
            path = files[0]
            capture = cv2.VideoCapture(str(path))
            captures[view] = capture
            if not capture.isOpened():
                raise RuntimeError(f"cannot open video: {path}")
            count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < n_frames:
                raise ValueError(f"video {view} has only {count} frames, cache has {n_frames}")
            paths[view] = path
            metadata[view] = {"path": str(path), "bytes": path.stat().st_size,
                              "frames": count, "fps": capture.get(cv2.CAP_PROP_FPS),
                              "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                              "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                              "alignment": "OpenCV position checked; not timestamp sync audit"}
    except BaseException:
        for capture in captures.values():
            capture.release()
        raise
    return paths, captures, metadata


def make_backgrounds(captures, frames):
    return {view: np.median(np.stack([seek_read(cap, f, view) for f in frames]),
                            axis=0).astype(np.uint8) for view, cap in captures.items()}


def write_previews(out, snapshots, result, baseline_xyz, rig):
    files = []
    colors = ((255, 185, 45), (65, 205, 240), (240, 90, 170))  # RGB
    positions = {int(f): i for i, f in enumerate(result.frame_idx)}
    for frame, views in snapshots.items():
        t = positions[frame]
        crops = {}
        for view, rgb in views.items():
            projected = np.concatenate((rig.camera(view).project(
                result.pose_xyz[t].reshape(-1, 3)), rig.camera(view).project(baseline_xyz[t])))
            projected = projected[np.isfinite(projected).all(axis=1)]
            if len(projected):
                lo = np.maximum(np.floor(projected.min(axis=0) - 85).astype(int), 0)
                hi = np.minimum(np.ceil(projected.max(axis=0) + 85).astype(int),
                                [rgb.shape[1], rgb.shape[0]])
                if np.all(hi > lo):
                    crops[view] = (slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        panels = []
        for label, candidate in (("C34 lambda 0.767", False), ("3D-driven candidate", True)):
            row = []
            for view in VIEWS:
                panel = views[view].copy()
                for fish in range(3):
                    color = colors[fish]
                    if candidate:
                        uv = rig.camera(view).project(result.pose_xyz[t, fish])
                        known = result.observed[t, fish]
                        for node, pixel in enumerate(uv):
                            if np.isfinite(pixel).all():
                                p = tuple(np.rint(pixel).astype(int))
                                cv2.circle(panel, p, 5 if known[node] else 8, color,
                                           -1 if known[node] else 1)
                        for a, b in ((0, 1), (1, 2)):
                            if np.isfinite(uv[[a, b]]).all():
                                cv2.line(panel, tuple(np.rint(uv[a]).astype(int)),
                                         tuple(np.rint(uv[b]).astype(int)), color, 2)
                        pixel = uv[1]
                        suffix = " review" if result.review[t, fish] else ""
                    else:
                        pixel = rig.camera(view).project(baseline_xyz[t, fish])
                        suffix = ""
                    if np.isfinite(pixel).all():
                        p = tuple(np.rint(pixel).astype(int))
                        cv2.circle(panel, p, 10, color, 2)
                        cv2.putText(panel, f"S{fish}{suffix}", (p[0] + 12, p[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                if view in crops:
                    panel = panel[crops[view]]
                # Both methods use the identical crop within a view/frame.
                # Preserve its aspect ratio rather than stretching fish bodies.
                scale = min(480 / panel.shape[1], 480 / panel.shape[0])
                scaled = cv2.resize(panel, (round(panel.shape[1] * scale),
                                            round(panel.shape[0] * scale)))
                panel = np.full((480, 480, 3), 15, dtype=np.uint8)
                dy, dx = (480 - scaled.shape[0]) // 2, (480 - scaled.shape[1]) // 2
                panel[dy:dy + scaled.shape[0], dx:dx + scaled.shape[1]] = scaled
                cv2.rectangle(panel, (0, 0), (480, 54), (15, 15, 15), -1)
                cv2.putText(panel, f"{label} | {view}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1)
                cv2.putText(panel, f"frame {frame}; hollow nodes = predictions", (8, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1)
                row.append(panel)
            panels.append(np.concatenate(row, axis=1))
        path = out / f"preview_{frame:09d}.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels), cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"could not write {path}")
        files.append(path.name)
    html = ("<!doctype html><meta charset='utf-8'><title>Local identity recovery</title>"
            "<style>body{background:#171b21;color:#eee;font-family:sans-serif}"
            "img{display:block;width:min(100%,1440px);margin:20px 0}</style>"
            "<h1>Window-initialized recovery preview</h1><p>Top: frozen C34. Bottom: candidate. "
            "Colors denote arbitrary initial slots. Views crop around all projected fish; "
            "each method uses the same crop. Hollow nodes are latent predictions; "
            "review means identity is not accepted. These pictures are not ground truth.</p>")
    (out / "preview.html").write_text(html + "".join(
        f"<img src='{file}' alt='{file}'>" for file in files), encoding="utf-8")
    return files


def main():
    from tracker import FrameInput, GenerativeTracker, TrackerConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--background-frames", nargs="+", type=int,
                        default=list(BACKGROUND_FRAMES))
    parser.add_argument("--preview-count", type=int, default=6)
    args = parser.parse_args()
    if args.experiment != EXPERIMENT:
        raise ValueError("this locked baseline bundle covers K_1_2 only")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {args.out}")
    if not 0 <= args.start < args.stop:
        raise ValueError("require 0 <= start < stop")
    args.out.mkdir(parents=True)
    with provenance("window_initialized_recovery", args.out, name="benchmark") as record:
        record.params_from(args)
        staged = args.input_root / "staging_manifest.json"
        manifest = json.loads(staged.read_text(encoding="utf-8-sig"))
        # Verify every staged input against its recorded source hash before use.
        for entry in manifest["files"]:
            path = args.input_root / Path(entry["relative"])
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"staging checksum mismatch: {path}")
        record.input("staged_manifest", staged, sha256=sha256(staged))
        root = args.input_root
        rig = load_rig(root / CALIBRATION_DIR)
        detections = CachedDetections(root, args.experiment)
        if args.stop > detections.n_frames:
            raise ValueError("requested stop extends past the detection cache")
        frames = np.arange(args.start, args.stop, dtype=np.int64)
        baseline_paths = sorted((root / BASELINE_DIR).glob("C34-label*.npz"))
        if len(baseline_paths) != 7:
            raise ValueError("the comparison requires all seven frozen baseline arms")
        seed_path = root / BASELINE_DIR / f"C34-label__{args.experiment}__lam0.767.npz"
        with np.load(seed_path, allow_pickle=False) as seed_data:
            seed_source = seed_data["source"][args.start:args.start + 1]
        initial, initial_observed, _ = skeleton_from_sources(
            rig, detections, frames[:1], seed_source)
        if not initial_observed.all():
            raise ValueError("the initial frame lacks a complete three-animal skeleton")
        config = TrackerConfig()
        evidence_config = EvidenceConfig()
        record.param("tracker_config", dataclasses.asdict(config))
        record.param("evidence_config", dataclasses.asdict(evidence_config))
        record.param("seed", {"frame": args.start, "baseline": str(seed_path),
                              "sha256": sha256(seed_path), "source": seed_source.tolist(),
                              "pose_xyz": initial[0].tolist(), "uses_future_baseline": False,
                              "optimistic": "baseline seed used the whole recording"})
        code_paths = (
            Path(__file__), Path(__file__).with_name("tracker.py"),
            Path(__file__).with_name("observations.py"),
            Path(__file__).resolve().parents[2] / "src/medaka3d/calib.py",
            Path(__file__).resolve().parents[2] / "src/medaka3d/eval/generative.py",
        )
        record.param("code_hashes", {str(path): sha256(path) for path in code_paths})
        # A dirty working-tree hash alone is insufficient to recover runnable code.
        code_dir = args.out / "code_snapshot"
        code_dir.mkdir()
        for path in code_paths:
            shutil.copyfile(path, code_dir / path.name)
        background_frames = [f for f in args.background_frames
                             if 0 <= f < detections.n_frames and
                             (f < args.start - 332 or f >= args.stop + 332)]
        if len(background_frames) < 3:
            raise ValueError("need at least three predeclared distant background samples")
        record.param("background_frames_used", background_frames)
        record.param("background_rule", "fixed absolute samples excluding target +/-2s; no GT")
        _, captures, video_metadata = open_videos(args.raw_root, args.experiment,
                                                  detections.n_frames)
        record.param("videos", video_metadata)
        snapshots = {}
        preview_frames = set(map(int, np.linspace(args.start, args.stop - 1,
                                                  min(args.preview_count, len(frames)))))
        prep_started = time.perf_counter()
        try:
            backgrounds = make_backgrounds(captures, background_frames)
            previous = None
            if args.start:
                previous = {view: seek_read(cap, args.start - 1, view)
                            for view, cap in captures.items()}
            else:
                for capture in captures.values():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            preparation_seconds = time.perf_counter() - prep_started

            def inputs():
                nonlocal previous
                for frame in frames:
                    rgb = {}
                    for view, cap in captures.items():
                        ok, bgr = cap.read()
                        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                            raise RuntimeError(f"sequential frame alignment failed: {view}/{frame}")
                        rgb[view] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    evidence = SceneEvidence.from_frames(rgb, backgrounds=backgrounds,
                                                         previous_frames=previous,
                                                         config=evidence_config)
                    if frame in preview_frames:
                        snapshots[int(frame)] = {view: image.copy() for view, image in rgb.items()}
                    previous = rgb
                    if (frame - args.start) % 20 == 0:
                        print(f"tracking {int(frame)} / {args.stop}", flush=True)
                    yield FrameInput(frame_idx=int(frame),
                                     detections={view: detections.points(view, int(frame))
                                                 for view in VIEWS}, evidence=evidence)

            tracking_started = time.perf_counter()
            result = GenerativeTracker(rig, initial[0], config=config).track(inputs())
            tracking_seconds = time.perf_counter() - tracking_started
        finally:
            for capture in captures.values():
                capture.release()
        if not np.array_equal(result.frame_idx, frames):
            raise ValueError("tracker output frames differ from the requested window")
        candidate_path = args.out / "candidate.npz"
        np.savez_compressed(candidate_path, frame_idx=result.frame_idx,
                            pose_xyz=result.pose_xyz, xyz=result.xyz,
                            accepted_xyz=result.accepted_xyz, observed=result.observed,
                            node_source=result.node_source, review=result.review,
                            margin=result.margin, identity_margin=result.identity_margin)
        record.output("candidate", candidate_path)
        gt_path = root / "assets/groundtruth/fish_sex_gt" / args.experiment / "female_sex_gt.csv"
        with gt_path.open(encoding="utf-8-sig", newline="") as stream:
            anchors = match_mouth_anchors(csv.DictReader(stream), detections.mouth,
                                          args.start, args.stop)
        raw_score = score_female_identity(frames, result.node_source[:, :, :, 0],
                                          result.observed[:, :, 0], anchors)
        accepted_score = score_female_identity(frames, result.node_source[:, :, :, 0],
                                               result.observed[:, :, 0], anchors,
                                               review=result.review)
        def project(view, xyz):
            return rig.camera(view).project(xyz)
        candidate = {
            "motion": observed_motion_summary(result.xyz, frames),
            "accepted_motion": observed_motion_summary(result.accepted_xyz, frames),
            "skeleton": skeleton_shape_summary(result.pose_xyz, result.observed, initial[0]),
            "female_identity_raw": raw_score, "female_identity_accepted": accepted_score,
            "mouth_reprojection": mouth_reprojection_summary(
                result.pose_xyz, frames, anchors, raw_score, project),
            "review_animal_frames": int(result.review.sum()),
            "review_fraction": float(result.review.mean()),
            "accepted_observed_animal_frames": int(
                np.isfinite(result.accepted_xyz).all(axis=-1).sum()),
            "beam_margin": "retained hypotheses only; not a calibrated probability",
            "diagnostics": result.diagnostics,
        }
        baselines = {}
        preview_baseline = None
        for path in baseline_paths:
            with np.load(path, allow_pickle=False) as arrays:
                source = arrays["source"][frames]
                accepted = arrays["slot_track"][frames] >= 0
                xyz = np.where(accepted[..., None], arrays["xyz"][frames], np.nan)
            pose, observed, _ = skeleton_from_sources(rig, detections, frames, source)
            observed &= accepted[..., None]
            score = score_female_identity(frames, source, observed[:, :, 0], anchors)
            baselines[path.stem.split("__lam")[-1]] = {
                "path": str(path), "sha256": sha256(path),
                "motion": observed_motion_summary(xyz, frames), "female_identity": score,
                "skeleton": skeleton_shape_summary(pose, observed, pose[0]),
                "mouth_reprojection": mouth_reprojection_summary(pose, frames, anchors,
                                                                    score, project),
            }
            if path == seed_path:
                preview_baseline = xyz
        previews = write_previews(args.out, snapshots, result, preview_baseline, rig)
        summary = {
            "protocol": "window_initialized_recovery", "experiment": args.experiment,
            "n_groups": 1, "frames": [args.start, args.stop],
            "preparation_seconds": preparation_seconds, "tracking_seconds": tracking_seconds,
            "tracking_fps": len(frames) / tracking_seconds,
            "candidate": candidate, "baseline_arms": baselines, "previews": previews,
            "limitations": [
                "First frame initialized from the whole-recording baseline; optimistic local test.",
                "No baseline predictions after initialization enter the tracker.",
                "Only candidate runtime measured; frozen baseline runtime was not measured.",
                "Female endpoint intervals cannot count transient swaps or validate male IDs.",
                "Known failure windows are development diagnostics, not held-out generalization.",
                "No lower jump count or residual alone establishes correct identity.",
            ],
        }
        summary_path = args.out / "comparison.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("comparison", summary_path)
        record.output("preview", args.out / "preview.html")
        print(json.dumps({"out": str(args.out), "tracking_seconds": tracking_seconds,
                          "candidate": candidate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
