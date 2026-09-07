"""S08 — render the locked-benchmark comparison: rows are frames, columns are models.

s06 reduced 480 predictions to a table. A table cannot show *how* a model
fails, and the headline finding — top-down joins parts of two animals while
bottom-up does not — is a picture, not a number. This draws every one of the 40
benchmark frames under all four models so the failures can be looked at.

=============================================================================
WHAT IS DRAWN
=============================================================================

Per cell: the human ground truth in **white**, the model's prediction in a
colour per instance, and a **red ring** wherever two predicted instances put a
node within 5 px of each other — the shared-node event s06 counts.

The crop is the bounding box of the human instances plus a margin, so all three
animals are visible and large. Different frames therefore have different
scales; the panel title carries the crop size so nothing is read off the image
as if it were an absolute distance.

Ground truth is 2-node (the benchmark has no pectoral), so a 3-node model's
pectoral is drawn without a white counterpart. That is expected and not a
missing label.

Run::

    .venv\\Scripts\\python.exe tools/s08_benchmark_montage.py --view top
    .venv\\Scripts\\python.exe tools/s08_benchmark_montage.py --view side1
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import sleap_io as sio
from s06_locked_benchmark_3node import (
    LOCKED,
    MODELS_BY_VIEW,
    SHARED_NODE_PX,
    frame_key,
    run_model,
)

from medaka3d import load_paths

INSTANCE_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
CHAIN = ("mouth", "pectoral", "tail")


def all_nodes(instance) -> dict[str, np.ndarray | None]:
    """Every node the instance carries, by name. 2-node models simply lack one."""
    out: dict[str, np.ndarray | None] = {}
    for point in instance.points:
        xy = np.asarray(point["xy"], dtype=float)
        out[str(point["name"])] = xy if np.isfinite(xy).all() else None
    return out


def draw(ax, nodes_list, colours, *, lw=1.4, ms=7, alpha=1.0):
    for index, nodes in enumerate(nodes_list):
        colour = colours[index % len(colours)] if isinstance(colours, tuple) else colours
        chain = [nodes[n] for n in CHAIN if nodes.get(n) is not None]
        if len(chain) >= 2:
            ax.plot([p[0] for p in chain], [p[1] for p in chain],
                    color=colour, lw=lw, alpha=alpha, zorder=3)
        for name, marker in (("mouth", "o"), ("pectoral", "+"), ("tail", "s")):
            p = nodes.get(name)
            if p is None:
                continue
            if marker == "+":
                ax.plot(p[0], p[1], marker, ms=ms + 3, mew=1.8, color=colour,
                        alpha=alpha, zorder=4)
            else:
                ax.plot(p[0], p[1], marker, ms=ms, mfc="none", mec=colour, mew=1.6,
                        alpha=alpha, zorder=4)


def shared_points(nodes_list) -> list[np.ndarray]:
    """Where two predictions put the same node within the threshold."""
    hits = []
    for name in ("mouth", "tail"):
        pts = [n[name] for n in nodes_list if n.get(name) is not None]
        for a, b in itertools.combinations(pts, 2):
            if float(np.linalg.norm(a - b)) <= SHARED_NODE_PX:
                hits.append((a + b) / 2.0)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--view", choices=("top", "side1", "side2"), default="top")
    parser.add_argument("--per-page", type=int, default=10)
    parser.add_argument("--margin", type=int, default=90)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    want = {"top": "top_cam", "side1": "side1_cam", "side2": "side2_cam"}[args.view]
    labels = sio.load_slp(str(LOCKED))
    frames = [f for f in labels.labeled_frames if frame_key(f)[1] == want]
    models = MODELS_BY_VIEW["top" if args.view == "top" else "side"]

    paths = load_paths()
    out_dir = paths.work_dir / "s08_benchmark_montage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.view}: {len(frames)} frames x {len(models)} models")
    predictions = {}
    for name, model_paths in models.items():
        print(f"  running {name} ...")
        predictions[name] = run_model(frames, model_paths, args.device, args.batch_size)

    names = list(models)
    written = []
    for page_start in range(0, len(frames), args.per_page):
        page = frames[page_start:page_start + args.per_page]
        rows, cols = len(page), len(names)
        fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.1 * rows))
        axes = np.atleast_2d(axes)

        for r, frame in enumerate(page):
            key = frame_key(frame)
            gt = [all_nodes(i) for i in frame.instances]
            pts = np.array([p for n in gt for p in n.values() if p is not None])
            if len(pts):
                lo = pts.min(axis=0) - args.margin
                hi = pts.max(axis=0) + args.margin
                side = float(max(hi - lo))
                centre = (lo + hi) / 2.0
                x0, y0 = centre - side / 2.0
            else:
                x0, y0, side = 0.0, 0.0, 1200.0
            image = np.asarray(frame.video[int(frame.frame_idx)])

            for c, name in enumerate(names):
                ax = axes[r, c]
                ax.imshow(image)
                draw(ax, gt, "#ffffff", lw=1.0, ms=9, alpha=0.55)
                pr = [all_nodes(i) for i in predictions[name].get(key, [])]
                draw(ax, pr, INSTANCE_COLOURS)
                hits = shared_points(pr)
                for h in hits:
                    ax.plot(h[0], h[1], "o", ms=22, mfc="none", mec="#ff0000",
                            mew=2.6, zorder=6)
                ax.set_xlim(x0, x0 + side)
                ax.set_ylim(y0 + side, y0)
                ax.set_xticks([])
                ax.set_yticks([])
                flag = f"  SHARED x{len(hits)}" if hits else ""
                title = name if r == 0 else ""
                ax.set_title(f"{title}\nf{key[2]}  {len(pr)} pred  crop {side:.0f}px{flag}",
                             fontsize=7.5,
                             color=("#cc0000" if hits else "#000000"))

        fig.suptitle(
            f"locked benchmark - {want} - frames {page_start + 1}-{page_start + rows}"
            f"   (white = human GT, colour = model, red ring = shared node "
            f"<= {SHARED_NODE_PX:g} px)",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        path = out_dir / f"montage_{args.view}_p{page_start // args.per_page + 1}.png"
        fig.savefig(path, dpi=85)
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path.name}")

    print(f"\n{len(written)} pages in {out_dir}")


if __name__ == "__main__":
    main()
