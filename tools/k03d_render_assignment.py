"""K03d — before/after crops for every frame where two mouths claimed one tail.

The counting in k03c says shared tails go 28 -> 0. This shows what that costs and
what it buys, one frame at a time, because a table cannot distinguish "the loser
recovered its own tail" from "the loser abstained" from "the loser took something
absurd", and those are three different outcomes with the same summary statistic.

Each row is one frame, drawn twice at the same crop:

  left    round-7 top-down as decoded — the shared tail is drawn in red
  right   after K03c assignment — mouth-only instances are drawn as an open
          circle with no axis, which is the honest abstention this method trades
          a silent wrong tail for

Green is ground truth throughout. Drawing is `k02_render_misdetections`'s, so a
node here means exactly what it means there.

Output: ``work_dir/k03d_assignment/`` — one PNG per frame plus a contact sheet.

Run:
    .venv\\Scripts\\python.exe tools/k03d_render_assignment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import sleap_io as sio
from b03_heldout_comparison import VIEWS, build_heldout, points, run_inference
from k01_misdetection_taxonomy import label_stats
from k02_render_misdetections import COLOURS, draw_instance
from k03_assignment_heldout import SPAN_LABELS

from medaka3d import load_catalog, load_paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment")
)
from assignment import (
    MOUTH,
    TAIL,
    OriginParams,
    assign_frame_origin,
    span_model_from_labels,
)

SHARE_PX = 5.0
MATCH_TOL_PX = 20.0


def shared_tail_pairs(instances: list[np.ndarray]) -> list[tuple[int, int]]:
    out = []
    for a in range(len(instances)):
        for b in range(a + 1, len(instances)):
            ta, tb = instances[a][TAIL], instances[b][TAIL]
            if np.isnan(ta).any() or np.isnan(tb).any():
                continue
            if float(np.hypot(*(ta - tb))) < SHARE_PX:
                out.append((a, b))
    return out


def outcome(gt: list[np.ndarray], after: list[np.ndarray]) -> str:
    """What became of the animals in this frame, in words rather than a count."""
    recovered = abstained = 0
    for instance in after:
        if np.isnan(instance[TAIL]).any():
            abstained += 1
            continue
        for truth in gt:
            if np.isnan(truth).any():
                continue
            if float(np.hypot(*(truth[TAIL] - instance[TAIL]))) <= MATCH_TOL_PX and float(
                np.hypot(*(truth[MOUTH] - instance[MOUTH]))
            ) <= MATCH_TOL_PX:
                recovered += 1
                break
    return f"{recovered}/{len(gt)} complete, {abstained} mouth-only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-body", type=float, default=2.6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "k03d_assignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, body = label_stats(catalog)
    span = span_model_from_labels(sio.load_slp(str(catalog.get(SPAN_LABELS))))
    frames, _ = build_heldout(catalog, "top")
    shape = tuple(frames[0].video.shape[1:3])
    print(__doc__.split("Run:")[0])
    preds = run_inference(frames, VIEWS["top"]["topdown"], args.device, args.batch_size)

    selected = []
    for frame in frames:
        pr = preds.get(frame.key, [])
        pairs = shared_tail_pairs(pr)
        if pairs:
            selected.append((frame, pr, pairs))
    print(f"frames with a shared tail: {len(selected)}")

    half = args.crop_body * body
    tiles = []
    for frame, pr, pairs in selected:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
        after = assign_frame_origin(pr, shape, span, OriginParams()).instances
        focus = np.nanmean([pr[pairs[0][0]][TAIL], pr[pairs[0][1]][TAIL]], axis=0)
        try:
            image = frame.video[frame.frame_idx]
        except Exception as error:
            print(f"  frame {frame.frame_idx}: image unavailable ({error})")
            image = None

        figure, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        for ax, instances, title in (
            (axes[0], pr, "before — round-7 top-down"),
            (axes[1], after, "after — K03c assignment"),
        ):
            if image is not None:
                ax.imshow(image, cmap="gray")
            for truth in gt:
                draw_instance(ax, truth, COLOURS["gt"])
            shared = {i for pair in pairs for i in pair} if instances is pr else set()
            for index, instance in enumerate(instances):
                colour = COLOURS["fp"] if index in shared else COLOURS["pred"]
                draw_instance(ax, instance, colour)
            ax.set_xlim(focus[0] - half, focus[0] + half)
            ax.set_ylim(focus[1] + half, focus[1] - half)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        figure.suptitle(
            f"frame {frame.frame_idx} — {len(pr)} predicted, {len(gt)} GT  ->  "
            f"{outcome(gt, after)}",
            fontsize=9,
        )
        figure.tight_layout()
        path = out_dir / f"f{frame.frame_idx}.png"
        figure.savefig(path, dpi=110)
        plt.close(figure)
        tiles.append(path)
        print(f"  frame {frame.frame_idx:>7d}  {len(pr)} pred -> {len(after)} out   "
              f"{outcome(gt, after)}")

    print(f"\nwrote {len(tiles)} tiles to {out_dir}")
    print("Red = the two instances sharing a tail. An open circle with no axis on the")
    print("right is a mouth-only instance: the anchor kept, the orientation withheld.")


if __name__ == "__main__":
    main()
