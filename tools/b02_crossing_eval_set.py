"""B0.2 — build a crossing-frame evaluation set from round-7 top_cam predictions.

Locates encounters (windows where identity is at risk, bounded by sustained
separation on both sides) in the retained round-7 `top_cam` predictions, and
reports the annotation budget they imply.

Three honesty constraints, each of which changes the number:

1. **Top view only.** The retained predictions are `top_cam`. A top-down
   projection collapses depth, so two fish at the same (x, y) and different
   depths read as coincident. This *over*-counts encounters relative to true 3D
   separation. The count here is an upper bound, and is labelled as such.
2. **Predictions are untracked** (`len(labels.tracks) == 0`), so slots must be
   assigned before pairwise distances mean anything. The linker below is
   deliberately crude — greedy nearest-centroid. Linking errors would inflate the
   count, so every experiment is also measured **link-free**, by the minimum
   pairwise distance between any two detections in the frame. Where the two
   disagree, the link-free number is the one to trust.
3. **The pixel scale is derived from the data, not assumed.** B0.1 found that
   1200 px = 200 mm is inconsistent with a 25-35 mm fish, so the scale is
   estimated per experiment from the spatial extent the animals actually cover,
   against the calibrated inner tank span (192 mm, PRIOR_FINDINGS 2.11).

Run:
    .venv\\Scripts\\python.exe tools/b02_crossing_eval_set.py [EXPERIMENT ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio

from medaka3d import load_paths
from medaka3d.eval.encounters import find_encounters, find_pair_encounters, summarise

#: Calibrated inner tank span, PRIOR_FINDINGS 2.11 (fixed_refractive_192x192x186).
TANK_MM = 192.0
#: Percentile band used for the extent estimate, to shrug off stray detections.
EXTENT_LO, EXTENT_HI = 0.5, 99.5
N_SLOTS = 3

CHUNK = re.compile(r"top_cam\.frames_(\d+)-(\d+)\.slp$")


def chunk_files(experiment_dir: Path) -> list[tuple[int, Path]]:
    """(start_frame, path) for every top_cam chunk, in frame order."""
    found: list[tuple[int, Path]] = []
    for path in experiment_dir.rglob("top_cam.frames_*.slp"):
        match = CHUNK.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def centroids_per_frame(experiment_dir: Path) -> tuple[list[np.ndarray], int]:
    """Per-frame (n_detections, 2) centroid arrays, indexed by absolute frame."""
    chunks = chunk_files(experiment_dir)
    if not chunks:
        return [], 0

    total = 0
    per_frame: dict[int, np.ndarray] = {}
    for start, path in chunks:
        labels = sio.load_slp(str(path))
        total = max(total, labels.videos[0].shape[0]) if labels.videos else total
        for frame in labels.labeled_frames:
            points = []
            for instance in frame.instances:
                array = np.asarray(instance.numpy(), dtype=float)[:, :2]
                if np.isnan(array).all():
                    continue
                points.append(np.nanmean(array, axis=0))
            index = int(frame.frame_idx)
            # chunk frame_idx is already absolute in these files; guard anyway
            if index < start:
                index += start
            per_frame[index] = np.array(points) if points else np.empty((0, 2))
    if not total:
        total = max(per_frame) + 1
    return [per_frame.get(i, np.empty((0, 2))) for i in range(total)], total


def pixel_scale_mm(frames: list[np.ndarray]) -> float:
    """mm per pixel, from the extent the animals actually cover."""
    stacked = np.vstack([f for f in frames if len(f)])
    lo = np.percentile(stacked, EXTENT_LO, axis=0)
    hi = np.percentile(stacked, EXTENT_HI, axis=0)
    span_px = float(np.mean(hi - lo))
    return TANK_MM / span_px


def link_greedy(frames: list[np.ndarray]) -> np.ndarray:
    """Crude nearest-centroid linker into fixed slots. (n_frames, N_SLOTS, 2).

    Not a tracker. It exists only so that pairwise distances have a consistent
    subject, and it is cross-checked against a link-free measurement.
    """
    out = np.full((len(frames), N_SLOTS, 2), np.nan)
    last = np.full((N_SLOTS, 2), np.nan)
    for index, points in enumerate(frames):
        if len(points) == 0:
            out[index] = np.nan
            continue
        free = list(range(len(points)))
        for slot in range(N_SLOTS):
            if np.isnan(last[slot]).any() or not free:
                continue
            distances = [float(np.hypot(*(points[p] - last[slot]))) for p in free]
            best = int(np.argmin(distances))
            out[index, slot] = points[free[best]]
            last[slot] = points[free[best]]
            free.pop(best)
        for slot in range(N_SLOTS):  # seed empty slots from leftovers
            if np.isnan(out[index, slot]).any() and free:
                out[index, slot] = points[free.pop(0)]
                last[slot] = out[index, slot]
    return out


def min_pairwise_mm(frames: list[np.ndarray], mm_per_px: float) -> np.ndarray:
    """Link-free: closest pair of detections in each frame, in mm.

    NaN where fewer than two detections exist — an unobserved frame is not
    evidence of separation, and ``find_pair_encounters`` treats NaN as such.
    """
    out = np.full(len(frames), np.nan)
    for index, points in enumerate(frames):
        if len(points) < 2:
            continue
        diff = points[:, None, :] - points[None, :, :]
        distance = np.hypot(diff[..., 0], diff[..., 1])
        np.fill_diagonal(distance, np.inf)
        out[index] = distance.min() * mm_per_px
    return out


def analyse(experiment: str, experiment_dir: Path) -> dict | None:
    frames, total = centroids_per_frame(experiment_dir)
    if not frames:
        print(f"{experiment}: no top_cam chunks")
        return None

    mm_per_px = pixel_scale_mm(frames)
    detections = np.array([len(f) for f in frames])

    linked = link_greedy(frames)
    tracks_xyz = np.concatenate([linked * mm_per_px, np.zeros((len(frames), N_SLOTS, 1))], axis=2)
    linked_encounters = find_encounters(tracks_xyz, experiment=experiment)

    free_trace = min_pairwise_mm(frames, mm_per_px)
    free_encounters = find_pair_encounters(free_trace, experiment=experiment)

    linked_summary = summarise(linked_encounters)
    free_summary = summarise(free_encounters)

    print(f"\n{'=' * 74}\n{experiment}\n{'=' * 74}")
    print(f"frames: {total}, scale: {mm_per_px:.4f} mm/px (tank span {TANK_MM} mm)")
    print(
        f"detections/frame: mean {detections.mean():.2f}; "
        f"0 in {(detections == 0).sum()}, <3 in {(detections < 3).sum()}, "
        f">3 in {(detections > 3).sum()} frames"
    )
    print(
        f"  linked (3 slots, all pairs): {linked_summary['count']:5d} encounters, "
        f"{linked_summary['annotation_minutes_estimate']:8.1f} annotation minutes"
    )
    print(
        f"  link-free (closest pair):    {free_summary['count']:5d} encounters, "
        f"{free_summary['annotation_minutes_estimate']:8.1f} annotation minutes"
    )
    if free_summary["count"]:
        print(
            f"  link-free duration: median {free_summary['median_seconds']:.2f}s, "
            f"max {free_summary['max_seconds']:.1f}s, "
            f">=1s: {free_summary['over_one_second']}"
        )
    return {"experiment": experiment, "linked": linked_summary, "free": free_summary}


def main() -> None:
    paths = load_paths()
    root = paths.sleap_output_dir
    if root is None:
        raise SystemExit("sleap_output_dir is not configured")

    wanted = sys.argv[1:]
    directories = sorted(p for p in root.iterdir() if p.is_dir())
    if wanted:
        directories = [p for p in directories if p.name in wanted]

    print(f"round-7 top_cam predictions under {root}")
    print(f"{len(directories)} experiment(s) selected\n")
    print("NOTE: top view only — depth is collapsed, so these counts are an UPPER BOUND")
    print("on true 3D encounters. Predictions are untracked; see module docstring.")

    results = [r for p in directories if (r := analyse(p.name, p))]
    if len(results) < 2:
        return

    print(f"\n{'=' * 74}\nTOTAL over {len(results)} experiments\n{'=' * 74}")
    for field in ("linked", "free"):
        count = sum(r[field]["count"] for r in results)
        minutes = sum(r[field]["annotation_minutes_estimate"] for r in results)
        print(f"  {field:10s} {count:6d} encounters, {minutes:9.1f} min ({minutes / 60:.1f} h)")


if __name__ == "__main__":
    main()
