"""Render every Group-K held-out false positive as an annotated image crop.

Companion to ``k01_misdetection_taxonomy.py``. That tool counts and classifies;
this one shows the pixels, so the classification can be checked by eye rather
than trusted. Categories A-G are the same, assigned by the same function.

Scope is the **113 held-out frames** — the only Group-K frames not in round-7's
training data, and therefore the only ones whose false-positive rate is honest.

Each tile is a crop centred on the false positive, with:

* **red**    the false-positive instance, labelled with its category
* **blue**   other predicted instances in the same frame (matched to ground truth)
* **green**  ground-truth instances

Mouth is drawn as a circle, tail as a square, and the two are joined, so a
shared node, a collapsed instance and a partial instance are visually distinct.

Outputs, under ``work_dir/k02_misdetections/``:

* ``tiles/<CATEGORY>_<experiment>_f<frame>.png`` — one crop per false positive
* ``montage_<CATEGORY>.png``                    — all crops of one category
* ``montage_ALL.png``                           — every crop, grouped by category

Run:
    .venv\\Scripts\\python.exe tools/k02_render_misdetections.py [--crop-body 3.0]
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
from b03_heldout_comparison import VIEWS, mean_visible_error, points, run_inference
from k01_misdetection_taxonomy import (
    build_k_frames,
    centroid,
    classify,
    label_stats,
    regime_of,
    span,
)
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

MATCH_TOL_PX = 20.0
COLOURS = {"fp": "#ff3b30", "pred": "#2b8cff", "gt": "#22c55e"}


def draw_instance(ax, inst: np.ndarray, colour: str, label: str = "", lw: float = 1.6) -> None:
    """Mouth = circle, tail = square, joined by a line. NaN nodes are skipped."""
    mouth, tail = inst[0], inst[1]
    if not (np.isnan(mouth).any() or np.isnan(tail).any()):
        ax.plot([mouth[0], tail[0]], [mouth[1], tail[1]], "-", color=colour, lw=lw, alpha=0.95)
    # Ground truth is drawn first and is usually sitting underneath a matched
    # prediction (~48% of GT instances are verbatim top-down output), so it is
    # drawn hollow and larger to stay visible through the overlay.
    hollow = colour == COLOURS["gt"]
    face = "none" if hollow else colour
    if not np.isnan(mouth).any():
        ax.plot(
            mouth[0], mouth[1], "o", mfc=face, mec=colour if hollow else "white",
            ms=10 if hollow else 6, mew=1.6 if hollow else 0.8,
        )
    if not np.isnan(tail).any():
        ax.plot(
            tail[0], tail[1], "s", mfc=face, mec=colour if hollow else "white",
            ms=9 if hollow else 5, mew=1.6 if hollow else 0.8,
        )
    if label:
        anchor = mouth if not np.isnan(mouth).any() else tail
        ax.annotate(
            label,
            (anchor[0], anchor[1]),
            textcoords="offset points",
            xytext=(6, 6),
            color=colour,
            fontsize=7,
            weight="bold",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crop-body", type=float, default=3.0, help="crop half-width, body lengths"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "k02_misdetections"
    (out_dir / "tiles").mkdir(parents=True, exist_ok=True)

    lo, hi, body = label_stats(catalog)
    frames = [f for f in build_k_frames(catalog) if f["heldout"]]
    print(f"held-out Group-K frames: {len(frames)}")
    print(f"span: median {body:.1f} px, bounds [{lo:.1f}, {hi:.1f}] px")

    all_pts = np.vstack([points(i) for f in frames for i in f["instances"]])
    finite = all_pts[~np.isnan(all_pts).any(axis=1)]
    arena = (
        float(finite[:, 0].min() - body),
        float(finite[:, 1].min() - body),
        float(finite[:, 0].max() + body),
        float(finite[:, 1].max() + body),
    )

    simple = [
        type("F", (), {"key": f["key"], "video": f["video"], "frame_idx": f["frame_idx"]})()
        for f in frames
    ]
    preds = run_inference(simple, VIEWS["top"]["topdown"], args.device, args.batch_size)

    half = args.crop_body * body
    found: list[dict] = []

    for frame in frames:
        gt = [p for p in (points(i) for i in frame["instances"]) if not np.isnan(p).all()]
        pr = preds.get(frame["key"], [])
        if not pr:
            continue
        matched: set[int] = set()
        if gt:
            cost = np.array([[mean_visible_error(g, p) for p in pr] for g in gt])
            rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
            for r, c in zip(rows, cols, strict=False):
                if np.isfinite(cost[r, c]) and cost[r, c] <= MATCH_TOL_PX:
                    matched.add(c)

        fp_indices = [i for i in range(len(pr)) if i not in matched]
        if not fp_indices:
            continue

        image = None
        try:
            image = np.asarray(frame["video"][frame["frame_idx"]])
        except Exception as exc:
            print(f"  ! image unreadable {frame['experiment']} f{frame['frame_idx']}: {exc}")

        for i in fp_indices:
            category = classify(pr[i], pr, lo, hi, body, arena)
            found.append(
                {
                    "category": category,
                    "experiment": frame["experiment"],
                    "frame_idx": frame["frame_idx"],
                    "regime": regime_of(gt, body),
                    "instance": pr[i],
                    "others": [pr[j] for j in range(len(pr)) if j != i],
                    "gt": gt,
                    "image": image,
                    "span": span(pr[i]),
                }
            )

    print(f"false positives to render: {len(found)}")
    by_cat: dict[str, list[dict]] = {}
    for item in found:
        by_cat.setdefault(item["category"], []).append(item)
    for cat in sorted(by_cat):
        print(f"  {cat:16s} {len(by_cat[cat]):3d}")

    def render(ax, item: dict, index: int) -> None:
        inst = item["instance"]
        c = centroid(inst)
        if np.isnan(c).any():
            c = np.nanmean(np.vstack([inst, *item["gt"]]), axis=0)
        # Clamp the crop inside the frame so tiles are filled rather than padded
        # with blank canvas, keeping the requested width where possible.
        w = h = 1200
        if item["image"] is not None:
            h, w = item["image"].squeeze().shape[:2]
        x0 = float(np.clip(c[0] - half, 0, max(0, w - 2 * half)))
        y0 = float(np.clip(c[1] - half, 0, max(0, h - 2 * half)))
        x1, y1 = x0 + 2 * half, y0 + 2 * half
        if item["image"] is not None:
            img = item["image"]
            ax.imshow(img.squeeze(), cmap="gray" if img.squeeze().ndim == 2 else None)
        else:
            ax.set_facecolor("#111")
            ax.text(c[0], c[1], "image\nunavailable", color="w", ha="center", fontsize=7)
        for g in item["gt"]:
            draw_instance(ax, g, COLOURS["gt"], lw=1.2)
        for o in item["others"]:
            draw_instance(ax, o, COLOURS["pred"], lw=1.2)
        draw_instance(ax, inst, COLOURS["fp"], label=item["category"].split("_", 1)[0], lw=2.2)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])
        span_txt = "NaN" if np.isnan(item["span"]) else f"{item['span']:.0f}px"
        ax.set_title(
            f"{index}. {item['category']}\n{item['experiment'][9:]} f{item['frame_idx']}"
            f"  {item['regime']}  span {span_txt}",
            fontsize=6.5,
        )
        for spine in ax.spines.values():
            spine.set_color(COLOURS["fp"])
            spine.set_linewidth(1.4)

    legend = [
        mpatches.Patch(color=COLOURS["fp"], label="false positive (classified)"),
        mpatches.Patch(color=COLOURS["pred"], label="other predictions (matched)"),
        mpatches.Patch(color=COLOURS["gt"], label="ground truth"),
    ]

    # per-category montages
    for cat, items in sorted(by_cat.items()):
        cols = min(6, len(items))
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.75 * rows), squeeze=False)
        for n, item in enumerate(items):
            render(axes[n // cols][n % cols], item, n + 1)
        for n in range(len(items), rows * cols):
            axes[n // cols][n % cols].axis("off")
        fig.suptitle(
            f"Group K held-out false positives — {cat}  (n={len(items)})",
            fontsize=11,
            weight="bold",
        )
        fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        path = out_dir / f"montage_{cat}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"wrote {path}")

        for n, item in enumerate(items):
            f2, a2 = plt.subplots(figsize=(3.0, 3.2))
            render(a2, item, n + 1)
            f2.tight_layout()
            f2.savefig(
                out_dir / "tiles" / f"{cat}_{item['experiment'][9:]}_f{item['frame_idx']}.png",
                dpi=140,
            )
            plt.close(f2)

    # combined montage, grouped by category
    ordered = [item for cat in sorted(by_cat) for item in by_cat[cat]]
    cols = 8
    rows = int(np.ceil(len(ordered) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.55 * rows), squeeze=False)
    for n, item in enumerate(ordered):
        render(axes[n // cols][n % cols], item, n + 1)
    for n in range(len(ordered), rows * cols):
        axes[n // cols][n % cols].axis("off")
    fig.suptitle(
        f"Group K — all {len(ordered)} held-out false positives, grouped by category",
        fontsize=13,
        weight="bold",
    )
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    path = out_dir / "montage_ALL.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
