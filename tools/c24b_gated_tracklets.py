"""C24b — cut a tracklet where the link is physically impossible.

=============================================================================
WHY THIS EXISTS
=============================================================================

`c24` removes the track cap, which removes *"there was no id left for the real
fish"*. It does not remove *"the id was put on the wrong instance"*, and that
second failure is the one `c25` cannot see: the correspondence trusts a label
to mean one animal, so a label that steps from a fish onto a reflection is
inherited as truth.

Two properties of `sleap_nn`'s tracker make it possible, both read off the
code rather than supposed:

* **No similarity threshold.** `scores_to_cost_matrix` is ``-scores`` with NaN
  sent to ``inf``, and `assign_tracks` takes the Hungarian answer whatever it
  costs. There is no "too far, start a new track" rule outside the Kalman path
  — and the Kalman path requires the very instance cap this pipeline removed.
* **No retirement.** `LocalQueueCandidates.current_tracks` is appended to and
  never emptied, so a track last seen 300,000 frames ago is still a candidate.

Measured on `side1_cam` 100,000-120,000, one-frame mouth displacement within a
label: p50 1.69 px, p99.9 12.5 px — and then **18 links over 30 px**, five of
them over 100 px, which is 3.5 body spans in 6 ms. Uncapping already cuts those
from `c16`'s 54 to 18. It does not reach zero, and it cannot: nothing in the
matcher refuses a link.

=============================================================================
WHAT THIS DOES
=============================================================================

One rule, in 2D only:

    a link between consecutive appearances of a label, ``dt`` frames apart, is
    cut when a tracked node moves more than ``gate_px * dt``.

Cutting means the later instances become a **new label**. Nothing is deleted,
no instance loses its label, and the count of labelled instances is unchanged —
the tool asserts all three. A cut turns a wrong link into a break, and a break
is what the next stage is built to absorb: `c25` charges one lambda for the
slot that changes label, exactly as it does for the handovers that already
occupy 12.6% of frames.

**It stays in 2D on purpose.** The 3D residual would find these links too, and
better. But `c16` exists to be *an independent series that can refuse a 3D
reassignment* — cutting 2D tracklets on 3D evidence would spend that
independence to buy a cleaner input, and the refusal is worth more.

``gate_px`` is not invented here. It is `s10`'s threshold, derived the same way
from the same human labels: **half the median mouth-tail span of the complete
instances in ``labels.round8.<view>.train``, rounded up to 10 px** — 60 px for
``top_cam``, 50 px for the sides. The uncapped p99.9 is 12.5 px, so the gate
sits four times above anything a fish does.

**n_groups = 1, n_experiments = 1.**

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c24b_gated_tracklets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from s10_shared_node_20k import median_span_px, tail_jump_threshold_px

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
RUN = "bottomup"
UNCAPPED_2D = Path("c24_uncapped_tracklets") / "mouth-pectoral_oks"

#: Which human-labelled corpus supplies each view's body length. `s10`'s map.
LABEL_KEY = {"top_cam": "labels.round8.top.train",
             "side1_cam": "labels.round8.side.train",
             "side2_cam": "labels.round8.side.train"}


def link_displacements(detections, labels: np.ndarray, nodes: tuple[str, ...]):
    """``(rows, dt, moved)`` for every consecutive pair within a label.

    ``moved`` is the largest distance any tracked node covered between the two
    appearances — the largest, not the mean, because one node teleporting is
    enough to say the two instances are not the same animal. `track2d` gates the
    same way.
    """
    points = np.stack([detections.node(node) for node in nodes], axis=1)
    frame = detections.frame_idx
    rows, gaps, moved = [], [], []
    for label in np.unique(labels[labels >= 0]):
        member = np.flatnonzero(labels == label)
        member = member[np.argsort(frame[member], kind="stable")]
        if member.size < 2:
            continue
        step = np.linalg.norm(points[member[1:]] - points[member[:-1]], axis=2)
        with np.errstate(invalid="ignore"):
            largest = np.nanmax(step, axis=1)
        rows.append(member[1:])
        gaps.append(np.diff(frame[member]).astype(np.int64))
        moved.append(largest)
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(rows), np.concatenate(gaps), np.concatenate(moved)


def cut_tracklets(detections, labels: np.ndarray, nodes: tuple[str, ...],
                  gate_px: float) -> tuple[np.ndarray, dict]:
    """Relabel so that no surviving link exceeds ``gate_px`` per frame elapsed."""
    frame = detections.frame_idx
    points = np.stack([detections.node(node) for node in nodes], axis=1)
    out = labels.copy()
    next_id = int(labels.max()) + 1 if (labels >= 0).any() else 0
    cuts, examined = 0, 0
    by_gap: dict[int, int] = {}

    for label in np.unique(labels[labels >= 0]):
        member = np.flatnonzero(labels == label)
        member = member[np.argsort(frame[member], kind="stable")]
        if member.size < 2:
            continue
        gap = np.diff(frame[member]).astype(np.int64)
        step = np.linalg.norm(points[member[1:]] - points[member[:-1]], axis=2)
        with np.errstate(invalid="ignore"):
            largest = np.nanmax(step, axis=1)
        # A pair with no node in common carries no evidence either way, and a
        # link is never cut on absent evidence.
        cut = np.where(np.isfinite(largest), largest > gate_px * gap, False)
        examined += cut.size
        if not cut.any():
            continue
        segment = np.r_[0, np.cumsum(cut)]
        for piece in range(1, int(segment.max()) + 1):
            out[member[segment == piece]] = next_id
            next_id += 1
        cuts += int(cut.sum())
        for size in gap[cut]:
            by_gap[int(size)] = by_gap.get(int(size), 0) + 1

    return out, {"links_examined": int(examined), "links_cut": int(cuts),
                 "cuts_by_gap": dict(sorted(by_gap.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--nodes", nargs="+", default=["mouth", "pectoral"])
    parser.add_argument("--views", nargs="+", default=list(VIEWS))
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    nodes = tuple(args.nodes)
    out_dir = paths.work_dir / "c24b_gated_tracklets" / "mouth-pectoral_oks"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("gated_tracklets", out_dir, name="c24b") as record:
        record.params_from(args)
        record.param("n_groups", 1)
        record.param("n_experiments", 1)
        print(f"{EXPERIMENT}   nodes {list(nodes)}   n_groups = 1, n_experiments = 1\n")

        spans, summary = {}, {}
        for view in args.views:
            label_path = catalog.get(LABEL_KEY[view])
            record.input(LABEL_KEY[view], label_path)
            span, n_span = median_span_px(label_path)
            gate = tail_jump_threshold_px(span)
            spans[view] = {"median_span_px": span, "instances": n_span, "gate_px": gate}
            print(f"  {view:<10} median mouth-tail span {span:6.1f} px over {n_span:,} "
                  f"human instances  ->  gate {gate:.0f} px per frame")
        record.param("gate", spans)

        print()
        for view in args.views:
            detections = c01.load(paths.work_dir, EXPERIMENT, view, RUN)
            labels = np.load(paths.work_dir / UNCAPPED_2D / f"track2d__{view}.npy")
            gate = spans[view]["gate_px"]

            _, gap, moved = link_displacements(detections, labels, nodes)
            finite = np.isfinite(moved)
            one = finite & (gap == 1)
            cut_labels, stats = cut_tracklets(detections, labels, nodes, gate)

            # Nothing may be lost: a cut renames, it never unlabels or deletes.
            if int((cut_labels >= 0).sum()) != int((labels >= 0).sum()):
                raise SystemExit("a cut changed how many instances carry a label")
            if not np.array_equal(labels >= 0, cut_labels >= 0):
                raise SystemExit("a cut moved which instances carry a label")
            expected = int(labels.max()) + 1 + stats["links_cut"]
            if int(cut_labels.max()) + 1 != expected:
                raise SystemExit(f"expected {expected} labels, got {cut_labels.max() + 1}")

            np.save(out_dir / f"track2d__{view}.npy", cut_labels)
            record.output(f"track2d__{view}", out_dir / f"track2d__{view}.npy")
            summary[view] = {
                **stats, **spans[view],
                "labels_before": int(labels.max()) + 1,
                "labels_after": int(cut_labels.max()) + 1,
                "one_frame_p99_9_px": float(np.percentile(moved[one], 99.9)),
                "one_frame_max_px": float(moved[one].max()),
            }
            cell = summary[view]
            share = cell["links_cut"] / max(cell["links_examined"], 1)
            print(f"  {view:<10} links {cell['links_examined']:>9,}   "
                  f"cut {cell['links_cut']:>5,} ({share:7.4%})   "
                  f"labels {cell['labels_before']:>4} -> {cell['labels_after']:>4}   "
                  f"one-frame move p99.9 {cell['one_frame_p99_9_px']:6.1f} px  "
                  f"max {cell['one_frame_max_px']:7.1f} px")
            if stats["cuts_by_gap"]:
                head = list(stats["cuts_by_gap"].items())[:6]
                print("             cuts by frames elapsed: "
                      + "  ".join(f"{k}f:{v}" for k, v in head))

        record.param("per_view", summary)
        (out_dir / "c24b_summary.json").write_text(
            json.dumps({"experiment": EXPERIMENT, "n_groups": 1, "n_experiments": 1,
                        "groups": [ExperimentId.parse(EXPERIMENT).group],
                        "nodes": list(nodes), "per_view": summary},
                       indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
