"""S04 — do the side models say anything different from interpolation?

The side arms were judged on 28 held-out human view-frames, where neither beat
interpolation. This asks the question on the population that actually matters
for propagation: the **2,486 side view-frames that were never annotated**, which
are exactly the frames a round-2 seed would cover.

=============================================================================
WHAT THIS MEASURES, AND WHAT IT DOES NOT
=============================================================================

There is **no ground truth on these frames**. So this measures *disagreement*
between a model and the interpolation rule, not the accuracy of either. A large
disagreement means one of them is wrong, and the number alone cannot say which.

That is why the montage exists. The histogram locates the disagreements; the
montage renders them so a human can adjudicate. Read them in that order, and do
not treat "the model differs from interpolation" as "the model is better" —
that inference is exactly the one this project keeps having to retract.

**Mouth and tail are human labels** from round 8, held fixed. The interpolated
pectoral is `mouth + t * (tail - mouth)` with `t = 0.300`, the per-view median
measured from the round-1 annotation. The model's pectoral comes from its own
prediction, matched to the human instance by mouth+tail geometry.

Instances the model failed to detect, or whose mouth or tail is missing, are
counted and excluded rather than silently dropped.

Run::

    .venv\\Scripts\\python.exe tools/s04_pectoral_propagation_check.py --limit 100
    .venv\\Scripts\\python.exe tools/s04_pectoral_propagation_check.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths
from medaka3d.provenance import provenance

VIEWS = ("top_cam", "side1_cam", "side2_cam")
T_BY_VIEW = {"top_cam": 0.282, "side1_cam": 0.300, "side2_cam": 0.300}

ANNOTATION = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-seed-20260804T103622JST-3d5f46b7c830\edits\pectoral_edits_full.pkg.slp"
)
RUNS = Path(r"Z:\StephensU\medaka\work\s03_bottomup_3node\side")
MODELS = {
    "paf8": RUNS / "s03.bottomup_3node.round8_f24_ms64_annotated-only_n250",
    "paf5": RUNS / "s03.bottomup_3node.round8_f24_ms64_paf5_annotated-only_n250",
}


def leaf_video(video) -> str:
    current, seen = video, 0
    while seen < 8:
        source = getattr(current, "source_video", None)
        if source is None:
            break
        current, seen = source, seen + 1
    return str(getattr(current, "filename", "") or "").replace("\\", "/")


def experiment_of(path: str) -> str | None:
    match = re.search(r"/(\d{8}_[A-Z]+_\d+_\d+_[A-Za-z0-9]+)/", path)
    return match.group(1) if match else None


def view_of(path: str) -> str | None:
    return next((v for v in VIEWS if f"/{v}/" in path), None)


def frame_key(frame) -> tuple:
    path = leaf_video(frame.video)
    return experiment_of(path), view_of(path), int(frame.frame_idx)


def node_xy(instance, name: str) -> np.ndarray | None:
    for point in instance.points:
        if str(point["name"]) == name:
            xy = np.asarray(point["xy"], dtype=float)
            return xy if np.isfinite(xy).all() else None
    return None


def build_targets(catalog, limit: int | None) -> tuple[list, dict]:
    """Round-8 side view-frames that were never annotated."""
    annotated = {frame_key(f) for f in sio.load_slp(str(ANNOTATION)).labeled_frames}
    labels = sio.load_slp(str(catalog.get("labels.round8.side.train")))
    frames = [f for f in labels.labeled_frames if frame_key(f) not in annotated]
    if limit:
        frames = frames[:limit]
    meta = {
        "round8_side_train_frames": len(labels.labeled_frames),
        "annotated_excluded": len(labels.labeled_frames) - len(frames) if not limit else None,
        "target_frames": len(frames),
    }
    return frames, meta


def run_model(frames, model_dir: Path, device: str, batch: int):
    import sleap_nn

    videos = list({id(f.video): f.video for f in frames}.values())
    blank = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=f.video, frame_idx=int(f.frame_idx), instances=[])
            for f in frames
        ],
        videos=videos,
        skeletons=[sio.Skeleton(["mouth", "pectoral", "tail"])],
    )
    predicted = sleap_nn.predict(
        blank, model_paths=[str(model_dir)], device=device, batch_size=batch
    )
    # Keyed by the FULL frame key, never by frame_idx alone. side1 and side2
    # are the same instant from two cameras and carry identical frame indices:
    # every one of the 14 side val moments collides. Pooling them matched a
    # side1 human instance against a side2 prediction and manufactured
    # 600-900 px "errors" that were really cross-view mismatches.
    out: dict[tuple, list] = {}
    for frame in predicted.labeled_frames:
        out.setdefault(frame_key(frame), []).extend(frame.instances)
    return out


def compare(frames, predictions, name: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    counts = {"instances": 0, "no_mouth_or_tail": 0, "unmatched": 0, "no_pectoral": 0}

    for frame in frames:
        experiment, view, frame_idx = frame_key(frame)
        t = T_BY_VIEW.get(view, 0.297)
        human = []
        for instance in frame.instances:
            counts["instances"] += 1
            mouth, tail = node_xy(instance, "mouth"), node_xy(instance, "tail")
            if mouth is None or tail is None:
                counts["no_mouth_or_tail"] += 1
                continue
            human.append((mouth, tail, mouth + t * (tail - mouth)))
        if not human:
            continue

        pred = predictions.get((experiment, view, frame_idx), [])
        pred_mt = []
        for instance in pred:
            pred_mt.append((node_xy(instance, "mouth"), node_xy(instance, "tail"),
                            node_xy(instance, "pectoral")))

        # Match on the human-labelled body axis, not on the pectoral: matching
        # on the quantity under test would bias the comparison toward agreement.
        cost = np.full((len(human), len(pred_mt)), 1e6)
        for i, (hm, ht, _) in enumerate(human):
            for j, (pm, pt, _) in enumerate(pred_mt):
                parts = [np.linalg.norm(hm - pm) if pm is not None else None,
                         np.linalg.norm(ht - pt) if pt is not None else None]
                usable = [p for p in parts if p is not None]
                if usable:
                    cost[i, j] = float(np.mean(usable))
        matched: dict[int, int] = {}
        if len(pred_mt):
            rows_i, cols_i = linear_sum_assignment(cost)
            for i, j in zip(rows_i, cols_i, strict=False):
                if cost[i, j] < 1e6:
                    matched[i] = j

        for i, (mouth, tail, interp) in enumerate(human):
            j = matched.get(i)
            if j is None:
                counts["unmatched"] += 1
                continue
            predicted_pec = pred_mt[j][2]
            if predicted_pec is None:
                counts["no_pectoral"] += 1
                continue
            pm, pt = pred_mt[j][0], pred_mt[j][1]
            rows.append({
                # The model's OWN mouth/tail, so a montage can show whether the
                # matched instance sits on the animal at all or on a neighbour
                # / a surface reflection.
                "pred_mouth_x": pm[0] if pm is not None else "",
                "pred_mouth_y": pm[1] if pm is not None else "",
                "pred_tail_x": pt[0] if pt is not None else "",
                "pred_tail_y": pt[1] if pt is not None else "",
                "n_pred_in_frame": len(pred_mt),
                "model": name, "experiment": experiment, "view": view,
                "frame_idx": frame_idx, "instance": i,
                "mouth_x": mouth[0], "mouth_y": mouth[1],
                "tail_x": tail[0], "tail_y": tail[1],
                "interp_x": interp[0], "interp_y": interp[1],
                "pred_x": predicted_pec[0], "pred_y": predicted_pec[1],
                "disagreement_px": float(np.linalg.norm(predicted_pec - interp)),
                "span_px": float(np.linalg.norm(tail - mouth)),
                "axis_match_px": float(cost[i, j]),
            })
    counts["compared"] = len(rows)
    return rows, counts


def instance_counts(frames, predictions) -> dict:
    """How many instances the model emits per frame. Ground truth is 3."""
    hist: dict[int, int] = {}
    for frame in frames:
        n = len(predictions.get(frame_key(frame), []))
        hist[n] = hist.get(n, 0) + 1
    return {str(k): hist[k] for k in sorted(hist)}


def histogram(all_rows: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0, 40, 81)
    colors = {"paf8": "#1f77b4", "paf5": "#d62728"}

    for name, rows in all_rows.items():
        d = np.array([r["disagreement_px"] for r in rows])
        axes[0].hist(np.clip(d, 0, 40), bins=bins, alpha=0.55,
                     label=f"{name}  (n={len(d)}, median {np.median(d):.2f} px)",
                     color=colors.get(name))
        srt = np.sort(d)
        axes[1].plot(srt, np.arange(1, len(srt) + 1) / len(srt) * 100,
                     label=name, color=colors.get(name), lw=1.8)

    axes[0].set_xlabel("|model pectoral - interpolated pectoral|  (px)")
    axes[0].set_ylabel("instances")
    axes[0].set_title("Disagreement with interpolation\n(no ground truth: this is not accuracy)")
    axes[0].legend(fontsize=8)
    axes[0].axvline(5, color="k", ls=":", lw=1)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("disagreement (px, log)")
    axes[1].set_ylabel("cumulative %")
    axes[1].set_title("Cumulative distribution")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    for pct in (50, 90, 99):
        axes[1].axhline(pct, color="k", ls=":", lw=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def montage(rows: list[dict], frames, out_path: Path, n: int, crop: int, name: str,
            *, same_fish_px: float | None = None, title_note: str = "") -> None:
    """Render the largest disagreements as image crops.

    ``same_fish_px`` filters on ``axis_match_px`` -- how far the matched
    prediction's own mouth/tail sit from the human ones. Without it the top of
    the ranking is entirely *matching* failures (the model's instance is a
    different animal, 600-900 px away), which say nothing about where the
    pectoral belongs. Those are worth seeing, but as their own category.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {frame_key(f): f for f in frames}
    pool = rows if same_fish_px is None else [
        r for r in rows if r["axis_match_px"] <= same_fish_px
    ]
    worst = sorted(pool, key=lambda r: -r["disagreement_px"])[:n]
    if not worst:
        return
    cols = 4
    rows_n = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.0 * cols, 4.15 * rows_n))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, worst, strict=False):
        frame = by_key.get((r["experiment"], r["view"], int(r["frame_idx"])))
        image = None
        if frame is not None:
            try:
                image = np.asarray(frame.video[int(frame.frame_idx)])
            except Exception:
                image = None
        centre = np.array([r["interp_x"], r["interp_y"]])
        x0, y0 = int(centre[0] - crop // 2), int(centre[1] - crop // 2)
        if image is not None:
            h, w = image.shape[:2]
            x0, y0 = max(0, min(x0, w - crop)), max(0, min(y0, h - crop))
            ax.imshow(image[y0:y0 + crop, x0:x0 + crop])
        # Without explicit limits matplotlib autoscales to include a model
        # point hundreds of px outside the crop, shrinking the fish to a
        # thumbnail in a white field. Points outside the box are clipped.
        ax.set_xlim(0, crop)
        ax.set_ylim(crop, 0)
        ax.plot([r["mouth_x"] - x0, r["tail_x"] - x0],
                [r["mouth_y"] - y0, r["tail_y"] - y0],
                color="#00ff00", lw=1.0, alpha=0.8, zorder=2)
        ax.plot(r["mouth_x"] - x0, r["mouth_y"] - y0, "o", ms=7, mfc="none",
                mec="#00ff00", mew=1.6, label="mouth (human)", zorder=3)
        ax.plot(r["tail_x"] - x0, r["tail_y"] - y0, "s", ms=7, mfc="none",
                mec="#00ff00", mew=1.6, label="tail (human)", zorder=3)
        ax.plot(r["interp_x"] - x0, r["interp_y"] - y0, "x", ms=11, mew=2.4,
                color="#00e5ff", label="interpolated", zorder=4)
        pmx, pmy = r.get("pred_mouth_x", ""), r.get("pred_mouth_y", "")
        ptx, pty = r.get("pred_tail_x", ""), r.get("pred_tail_y", "")
        has_m = pmx not in ("", None)
        has_t = ptx not in ("", None)
        if has_m and has_t:
            ax.plot([float(pmx) - x0, r["pred_x"] - x0, float(ptx) - x0],
                    [float(pmy) - y0, r["pred_y"] - y0, float(pty) - y0],
                    color="#ff00ff", lw=1.2, alpha=0.85, zorder=3)
        if has_m:
            ax.plot(float(pmx) - x0, float(pmy) - y0, "o", ms=7, mfc="none",
                    mec="#ff00ff", mew=1.6, label="mouth (model)", zorder=4)
        if has_t:
            ax.plot(float(ptx) - x0, float(pty) - y0, "s", ms=7, mfc="none",
                    mec="#ff00ff", mew=1.6, label="tail (model)", zorder=4)
        ax.plot(r["pred_x"] - x0, r["pred_y"] - y0, "+", ms=13, mew=2.4,
                color="#ff00ff", label="pectoral (model)", zorder=5)
        ax.set_title(f"{r['view']} f{r['frame_idx']}\n"
                     f"disagreement {r['disagreement_px']:.1f} px "
                     f"({100*r['disagreement_px']/max(r['span_px'],1e-6):.0f}% of span)"
                     f"  |  {int(float(r.get('n_pred_in_frame', 0)))} detected",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(worst):]:
        ax.axis("off")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{name}: largest model-vs-interpolation disagreements{title_note}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--montage-n", type=int, default=16)
    parser.add_argument("--crop", type=int, default=180)
    parser.add_argument("--same-fish-px", type=float, default=15.0,
                        help="Max mouth/tail distance for the prediction to count as "
                             "the same animal; above this it is a matching failure.")
    parser.add_argument("--from-csv", type=Path, default=None,
                        help="Skip inference and re-render from a previous run's CSV.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "s04_pectoral_propagation"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("pectoral_propagation_check", out_dir, name="s04") as record:
        record.params_from(args)
        record.param("interpolation_t", T_BY_VIEW)
        record.param("caveat", "no ground truth on these frames; this is disagreement, "
                               "not accuracy")
        frames, meta = build_targets(catalog, args.limit)
        record.param("frames", meta)

        if args.from_csv:
            # Re-render figures from a completed run. Inference is the expensive
            # part and its output is already on disk; re-running it to change a
            # plot would be five minutes of GPU to redraw a picture.
            numeric = {"mouth_x", "mouth_y", "tail_x", "tail_y", "interp_x", "interp_y",
                       "pred_x", "pred_y", "disagreement_px", "span_px", "axis_match_px",
                       "frame_idx", "instance"}
            all_rows = {}
            with args.from_csv.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    typed = {k: (float(v) if k in numeric and v != "" else v)
                             for k, v in row.items()}
                    all_rows.setdefault(typed["model"], []).append(typed)
            print(f"loaded {sum(len(v) for v in all_rows.values())} rows from "
                  f"{args.from_csv}")
            render(all_rows, frames, out_dir, args, record)
            return
        print(f"target view-frames: {meta['target_frames']}  "
              f"(round-8 side train {meta['round8_side_train_frames']}, "
              f"annotated excluded {meta['annotated_excluded']})")

        all_rows: dict[str, list[dict]] = {}
        summary: dict[str, dict] = {}
        for name, model_dir in MODELS.items():
            if not (model_dir / "best.ckpt").is_file():
                raise SystemExit(f"missing model: {model_dir}")
            print(f"\n--- {name}: {model_dir.name}")
            predictions = run_model(frames, model_dir, args.device, args.batch_size)
            rows, counts = compare(frames, predictions, name)
            counts["instances_per_frame"] = instance_counts(frames, predictions)
            all_rows[name] = rows
            d = np.array([r["disagreement_px"] for r in rows]) if rows else np.array([0.0])
            summary[name] = {
                **counts,
                "median_px": float(np.median(d)),
                "p90_px": float(np.percentile(d, 90)),
                "p99_px": float(np.percentile(d, 99)),
                "max_px": float(d.max()),
                "pct_within_5px": float(np.mean(d <= 5) * 100),
                "pct_over_20px": float(np.mean(d > 20) * 100),
            }
            print(json.dumps(summary[name], indent=2))
            record.param(f"summary_{name}", summary[name])

        render(all_rows, frames, out_dir, args, record)

    print(f"\nall outputs in {out_dir}")


def render(all_rows, frames, out_dir, args, record) -> None:
    """Write the CSV and every figure."""
    if True:
        csv_path = out_dir / "disagreement.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fields = list(next(iter(r for rows in all_rows.values() for r in rows)))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rows in all_rows.values():
                writer.writerows(rows)
        record.output("disagreement_csv", csv_path)

        hist_path = out_dir / "disagreement_histogram.png"
        histogram(all_rows, hist_path)
        record.output("histogram", hist_path)
        print(f"\nwrote {hist_path}")

        for name, rows in all_rows.items():
            same = out_dir / f"montage_{name}_samefish.png"
            montage(rows, frames, same, args.montage_n, args.crop, name,
                    same_fish_px=args.same_fish_px,
                    title_note=(f"  |  same animal only (mouth/tail within "
                                f"{args.same_fish_px:g} px)"))
            record.output(f"montage_{name}_samefish", same)
            n_same = sum(1 for r in rows if r["axis_match_px"] <= args.same_fish_px)
            print(f"wrote {same}   ({n_same} same-animal matches of {len(rows)})")

            fails = [r for r in rows if r["axis_match_px"] > args.same_fish_px]
            bad = out_dir / f"montage_{name}_matchfail.png"
            montage(fails, frames, bad, args.montage_n, args.crop, name,
                    title_note=f"  |  MATCHING FAILURES ({len(fails)} of {len(rows)})")
            record.output(f"montage_{name}_matchfail", bad)
            print(f"wrote {bad}   ({len(fails)} matching failures)")

    print(f"\nall outputs in {out_dir}")


if __name__ == "__main__":
    main()
