"""B0.1 — has the production model ever seen a crossing?

The experimenter reports that round 8 degrades exactly at crossings: one tail
joined to two mouths, unstable detection in X-crossings. PRIOR_FINDINGS 2.10
predicts the training set contains approximately no such frames — a model cannot
learn to separate two fish it has never seen close together.

Counts labelled frames in ``labels.round8.combined`` holding two instances that
are close, under three separate definitions, because "close" is not one number:

* **centroid**   — midpoint of mouth and tail, the usual proxy
* **min-node**   — nearest node of A to nearest node of B; catches a head
                   touching a tail, which centroid distance hides
* **mouth-mouth**— the head-on case

Also reports node sharing (two instances at the same coordinate — the reported
failure mode, in the training data) and the pixel scale, so that "100 px" can be
read as a fraction of a fish.

**Which dataset.** Defaults to the latest and largest labels available per view,
which are not the same round:

* **top** — `labels.round8.top.train` (1,382 frames). There is no round-9 top;
  round 8 is top-down only.
* **side** — `retrain_20260728_round9b` (3,118 frames), which supersedes both
  round 8 side (2,764) and round9_side (3,086). Every one of its frames carries
  exactly 3 instances, where round9_side still had one 2-instance and two
  4-instance frames — it is the completed set.

Pass asset keys as arguments to analyse a different selection, e.g. the single
combined round-8 package:

    .venv\\Scripts\\python.exe tools/b01_crossing_frames_in_training.py labels.round8.combined

Run:
    .venv\\Scripts\\python.exe tools/b01_crossing_frames_in_training.py
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import sleap_io as sio

from medaka3d import load_catalog, load_paths

#: Thresholds in pixels. 100 px is the roadmap's figure; the others bracket it.
THRESHOLDS = (50, 100, 150, 200)

#: Tank is 20 cm across and images are 1200 px, so ~0.167 mm/px. A 30 mm medaka
#: is therefore ~180 px long — i.e. the roadmap's 100 px is roughly half a fish.
MM_PER_PX = 200.0 / 1200.0


def view_of(video: sio.Video) -> str:
    """top / side / unknown, from the embedded source-video path.

    Two path shapes occur: the label packages (``/top/``, ``/side/``) and raw
    recording folders (``/top_cam/``, ``/side1_cam/``). Match both — missing the
    second silently binned 279 frames as "unknown" on the first pass.

    Both the embedded ``source_video`` and the video's own ``filename`` are
    checked. Embedded ``.pkg.slp`` sets carry the view only on ``source_video``;
    non-embedded sets like round9b have no ``source_video`` at all and carry it on
    ``filename``. Checking only the first binned 1,332 round9b frames as
    "unknown".
    """
    source = getattr(video, "source_video", None)
    candidates = (
        str(getattr(source, "filename", "") or ""),
        str(getattr(video, "filename", "") or ""),
    )
    for text in candidates:
        match = re.search(r"/(top|side\d?)(?:_cam)?/", text.replace("\\", "/"))
        if match:
            return "top" if match.group(1) == "top" else "side"
    return "unknown"


def points_of(instance) -> np.ndarray:
    """(n_nodes, 2) array of visible points; NaN where not labelled."""
    array = np.asarray(instance.numpy(), dtype=float)
    return array[:, :2] if array.ndim == 2 and array.shape[1] >= 2 else array


def pair_distances(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(centroid, min-node, mouth-mouth) distance between two instances."""
    if np.isnan(a).all() or np.isnan(b).all():
        return float("nan"), float("nan"), float("nan")
    centroid_a, centroid_b = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    centroid = float(np.hypot(*(centroid_a - centroid_b)))

    gaps = [
        float(np.hypot(*(pa - pb)))
        for pa in a
        for pb in b
        if not (np.isnan(pa).any() or np.isnan(pb).any())
    ]
    min_node = min(gaps) if gaps else float("nan")

    mouth = float("nan")
    if not (np.isnan(a[0]).any() or np.isnan(b[0]).any()):
        mouth = float(np.hypot(*(a[0] - b[0])))
    return centroid, min_node, mouth


#: Latest and largest labels per view. Not the same round — see module docstring.
DEFAULT_KEYS = (
    "labels.round8.top.train",
    "training_labels:training/sleap/legacy/retrain_20260728_round9b/side/train_labels.pkg.slp",
)


def main() -> None:
    import sys

    catalog = load_catalog(load_paths().asset_root)
    keys = sys.argv[1:] or list(DEFAULT_KEYS)

    all_frames = []
    videos: list = []
    nodes: list[str] = []
    for key in keys:
        path = catalog.get(key)
        part = sio.load_slp(str(path))
        nodes = nodes or [n.name for n in part.skeletons[0].nodes]
        all_frames.extend(part.labeled_frames)
        videos.extend(part.videos)
        print(
            f"{key}\n    {path}\n    {len(part.labeled_frames)} frames, "
            f"{len(part.videos)} videos, "
            f"{sum(len(f.instances) for f in part.labeled_frames)} instances"
        )

    print(f"\n{len(all_frames)} labelled frames, {len(videos)} videos across {len(keys)} set(s)")
    print(f"skeleton nodes: {nodes}")
    print(f"scale: {MM_PER_PX:.3f} mm/px -> a 30 mm fish is ~{30 / MM_PER_PX:.0f} px long\n")

    views = Counter(view_of(v) for v in videos)
    print(f"videos by view: {dict(views)}")

    per_view_frames: Counter[str] = Counter()
    instances_per_frame: Counter[int] = Counter()
    body_lengths: list[float] = []

    # closest pair per frame, per metric
    closest: dict[str, list[tuple[float, str]]] = {"centroid": [], "min_node": [], "mouth": []}
    shared_node_frames = 0
    multi_instance_frames = 0

    for frame in all_frames:
        view = view_of(frame.video)
        per_view_frames[view] += 1
        instances = [points_of(i) for i in frame.instances]
        instances_per_frame[len(instances)] += 1

        for points in instances:
            if not (np.isnan(points[0]).any() or np.isnan(points[-1]).any()):
                body_lengths.append(float(np.hypot(*(points[0] - points[-1]))))

        if len(instances) < 2:
            continue
        multi_instance_frames += 1

        best = {"centroid": np.inf, "min_node": np.inf, "mouth": np.inf}
        shared = False
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                centroid, min_node, mouth = pair_distances(instances[i], instances[j])
                best["centroid"] = min(best["centroid"], centroid)
                if not np.isnan(min_node):
                    best["min_node"] = min(best["min_node"], min_node)
                    shared = shared or min_node < 1.0
                if not np.isnan(mouth):
                    best["mouth"] = min(best["mouth"], mouth)
        shared_node_frames += shared
        for metric, value in best.items():
            if np.isfinite(value):
                closest[metric].append((value, view))

    print(f"frames by view:  {dict(per_view_frames)}")
    print(f"instances/frame: {dict(sorted(instances_per_frame.items()))}")
    print(f"frames with >=2 instances: {multi_instance_frames}\n")

    if body_lengths:
        lengths = np.array(body_lengths)
        median_px = float(np.median(lengths))
        print(
            f"labelled mouth-tail span: median {median_px:.0f} px, "
            f"5-95% {np.percentile(lengths, 5):.0f}-{np.percentile(lengths, 95):.0f} px, "
            f"n={len(lengths)}"
        )
        print(
            f"  ! Under the assumed scale that is {median_px * MM_PER_PX:.1f} mm, but medaka are\n"
            f"    25-35 mm (CLAUDE.md). So either the frame does not span the full 200 mm tank,\n"
            f"    or the 'tail' node sits at the tail base rather than the fin tip. UNRESOLVED —\n"
            f"    do not convert these pixels to mm until calibration settles it (blocks B0.4).\n"
            f"    Everything below is reported in pixels and in body lengths, never in mm.\n"
        )

    print("=" * 74)
    print("CROSSING FRAMES IN THE TRAINING SET")
    print("=" * 74)
    header = f"{'metric':10s} {'threshold':>10s} {'frames':>8s} {'% of >=2':>9s}   top / side"
    print(header)
    for metric in ("centroid", "min_node", "mouth"):
        values = closest[metric]
        for threshold in THRESHOLDS:
            hits = [v for v in values if v[0] < threshold]
            by_view = Counter(view for _, view in hits)
            share = 100.0 * len(hits) / multi_instance_frames if multi_instance_frames else 0.0
            print(
                f"{metric:10s} {threshold:>7d} px {len(hits):>8d} {share:>8.2f}%   "
                f"{by_view.get('top', 0)} / {by_view.get('side', 0)}"
            )
        print()

    for metric in ("centroid", "min_node"):
        values = sorted(v for v, _ in closest[metric])
        if values:
            arr = np.array(values)
            print(
                f"{metric:10s} closest-pair distance: min {arr.min():.0f} px, "
                f"1% {np.percentile(arr, 1):.0f}, 5% {np.percentile(arr, 5):.0f}, "
                f"median {np.median(arr):.0f}"
            )

    print(f"\nframes where two instances share a node (<1 px apart): {shared_node_frames}")

    # ---- Body-length units: scale-free, so it survives a wrong mm/px --------
    if not body_lengths:
        return
    body = float(np.median(body_lengths))
    print()
    print("=" * 74)
    print(f"IN BODY LENGTHS (median labelled mouth-tail = {body:.0f} px)")
    print("=" * 74)
    print("This is the number that matters: it does not depend on the mm/px assumption.")
    print(f"\n{'metric':10s} {'separation':>22s} {'frames':>8s} {'% of >=2':>9s}")
    for metric in ("centroid", "min_node"):
        values = np.array([v for v, _ in closest[metric]])
        for fraction in (0.25, 0.5, 1.0, 2.0):
            hits = int((values < fraction * body).sum())
            share = 100.0 * hits / multi_instance_frames if multi_instance_frames else 0.0
            print(
                f"{metric:10s} {f'< {fraction:g} body ({fraction * body:.0f} px)':>22s} "
                f"{hits:>8d} {share:>8.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
