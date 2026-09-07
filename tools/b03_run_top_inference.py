"""B0.3 (top) — regenerate the missing round-7 top validation predictions.

All four round-7 `top` runs shipped an **empty** `labels_pr.val.0.slp` (0 frames
against 129 ground-truth frames), so the production view could not be scored.
This re-runs inference to produce them.

Two things had to be worked around, both worth knowing:

**The frame-to-video assignment in the saved file is a permutation.** Only **17 of
129** frames have a `frame_idx` present in their *own* video's `source_inds`; all
129 are present somewhere in the union. The video list maps to datasets in the
order video6, video1, video7, video3, video5, video4, video0, video2. Trusting
`frame.video` runs inference on the wrong image, or more usually raises
`IndexError: Frame index 309598 out of range`.

The fix is to re-attach each frame to the video whose `frame_map` contains it.
**The frame index itself is left alone** — `HDF5Video` keys `frame_map` on the
*source* index (69857 -> dataset row 0) and overrides `has_frame` to consult it,
so `frame_idx` was already correct. Rewriting it to a dataset row, which looks
like the obvious fix, breaks reading instead.

**Self-check.** If the remapping were wrong, inference would run on unrelated
images and predictions would not land near the ground truth. The scoring step
(`b03_score_checkpoints_at_crossings.py`) therefore validates the remapping as a
side effect: high recall means the images matched.

Outputs go to ``work_dir``. The inherited `/bucket` files are never modified.

Run:
    .venv\\Scripts\\python.exe tools/b03_run_top_inference.py [--limit N]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sleap_io as sio
import sleap_nn

ROUND7_TOP = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260704\top\runs")

#: (label, [model dirs]) — top-down needs centroid + centered_instance.
#:
#: RESOLUTION MATTERS. `round7_260704.bottomup_*` was trained at
#: `preprocessing.scale = 0.5`; `round7s1.bottomup_*` and both top-down models at
#: `scale = 1.0`. Comparing top-down against the 0.5 run confounds architecture
#: with input resolution — at half scale, two fish 25 px apart are 12.5 px apart
#: in the model's input, which is exactly the regime under test. `round7s1` is
#: therefore the primary bottom-up arm; the half-resolution run is kept only as a
#: named sensitivity arm.
ARCHITECTURES = {
    "topdown": [
        ROUND7_TOP / "round7_260704.centroid_n1295",
        ROUND7_TOP / "round7_260704.centered_instance_n1295",
    ],
    "bottomup_full": [ROUND7_TOP / "round7s1.bottomup_n1295"],
    "bottomup_half": [ROUND7_TOP / "round7_260704.bottomup_n1295"],
}

#: Ground truth is identical across runs; take it from the production run.
GT = ROUND7_TOP / "round7_260704.centered_instance_n1295" / "labels_gt.val.0.slp"


def build_index(labels: sio.Labels) -> dict[int, sio.Video]:
    """source frame index -> the video whose frame_map actually contains it.

    Note what is NOT done here: the index is *not* rewritten. `HDF5Video` keys its
    `frame_map` on the **source** index (69857 -> dataset row 0) and overrides
    `has_frame` to consult it, so `frame_idx` is already correct. Converting to a
    dataset row would break reading. Only the video attachment is wrong.
    """
    index: dict[int, sio.Video] = {}
    collisions = 0
    for video in labels.videos:
        for original in np.asarray(video.backend.source_inds):
            key = int(original)
            if key in index:
                collisions += 1
                continue  # first wins, deterministically
            index[key] = video
    if collisions:
        print(f"  note: {collisions} source index(es) appear in more than one video; first wins")
    return index


def remap(labels: sio.Labels) -> tuple[sio.Labels, int]:
    """Re-attach every labelled frame to the video that actually contains it."""
    index = build_index(labels)
    frames, unresolved = [], 0
    for frame in labels.labeled_frames:
        video = index.get(int(frame.frame_idx))
        if video is None:
            unresolved += 1
            continue
        frames.append(
            sio.LabeledFrame(
                video=video, frame_idx=frame.frame_idx, instances=list(frame.instances)
            )
        )
    remapped = sio.Labels(
        labeled_frames=frames, videos=labels.videos, skeletons=labels.skeletons
    )
    return remapped, unresolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    # "auto" picks CUDA when available. Plain torch inference is fine on this
    # machine's RTX 5070 Ti; only ONNX/TensorRT *export* is broken there
    # (PRIOR_FINDINGS dead-end D1).
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from medaka3d import load_paths

    out_dir = load_paths().work_dir / "b03_top_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = sio.load_slp(str(GT))
    total = len(ground_truth.labeled_frames)
    print(f"ground truth: {total} frames, {len(ground_truth.videos)} videos")

    own = sum(
        1
        for f in ground_truth.labeled_frames
        if int(f.frame_idx) in set(np.asarray(f.video.backend.source_inds).tolist())
    )
    print(f"frames whose frame_idx is in their OWN video's source_inds: {own}/{total}")

    remapped, unresolved = remap(ground_truth)
    print(f"remapped {len(remapped.labeled_frames)} frames; {unresolved} unresolved")
    if unresolved:
        raise SystemExit("some frames could not be resolved — refusing to run a partial comparison")

    if args.limit:
        remapped = sio.Labels(
            labeled_frames=remapped.labeled_frames[: args.limit],
            videos=remapped.videos,
            skeletons=remapped.skeletons,
        )
        print(f"limited to {len(remapped.labeled_frames)} frames")

    gt_out = out_dir / "labels_gt.val.remapped.slp"
    sio.save_slp(remapped, str(gt_out))
    print(f"wrote remapped GT -> {gt_out}")

    for name, models in ARCHITECTURES.items():
        missing = [m for m in models if not m.is_dir()]
        if missing:
            print(f"\n{name}: SKIPPED, missing {missing}")
            continue
        print(f"\n=== {name} ===")
        start = time.time()
        predictions = sleap_nn.predict(
            sio.Labels(
                labeled_frames=list(remapped.labeled_frames),
                videos=remapped.videos,
                skeletons=remapped.skeletons,
            ),
            model_paths=[str(m) for m in models],
            device=args.device,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - start
        instances = sum(len(f.instances) for f in predictions.labeled_frames)
        print(
            f"{name}: {len(predictions.labeled_frames)} frames, {instances} instances "
            f"in {elapsed:.1f}s"
        )
        path = out_dir / f"labels_pr.val.{name}.slp"
        sio.save_slp(predictions, str(path))
        print(f"wrote -> {path}")


if __name__ == "__main__":
    main()
