"""C06 — top-down or bottom-up 3-node for production? The detector choice, re-opened.

=============================================================================
SPECIFICATION AND DECISION RULE — fixed in code before the top-down run finished
=============================================================================

**What is being re-opened, and by whose decision.** Two decisions already chose
bottom-up:

* **2026-08-09** (`s10`, 20,000 frames x 3 views x 4 models). Top-down 3-node's
  shared-node frame rate on the side views was **5.6% and 7.4%** against
  bottom-up's **0.000-0.010%**, and its 1st-percentile tail-to-tail distance was
  **0.45 px** — two predicted instances placing a tail on the same animal. That
  decision also established that shared-node is *degenerate for bottom-up* (its
  distribution is floored far above the 5 px threshold) and must not be used to
  rank architectures; `tail_swap` was named the successor metric.
* **2026-08-13** (`s15`). Retrained on the same 543 reviewed moments these
  top-down checkpoints come from, top-down's mouth-linked swap rate **rose** in
  all three views while bottom-up's fell 73% on top. `s16` therefore ran
  bottom-up only.

The experimenter has asked for top-down to be scored as a live production
candidate anyway. This tool does that, on evidence neither earlier decision
had: a **3D reconstruction** and a **tracker**.

**The confound that would have made this meaningless, and its control.**
`sleap-nn/0.3.1`, the runtime `s16` used, was withdrawn from saion between
2026-08-11 and 2026-08-14; its install directory is now empty. `c05` runs under
**0.3.3**, so architecture and runtime would have varied together. Controlled by
measurement: `c05 --control` re-ran three bottom-up chunks (75,000 frames,
225,000 instances) under 0.3.3 against the 0.3.1 originals and the coordinates
are **bit-identical, 100.00% exactly zero delta, in all three views**. The
confound is not argued away; it is measured to be zero, and `--control-report`
reprints that measurement beside the results.

**What tracking evidence can and cannot do here.** It is new evidence about a
question the detector metrics cannot reach — whether part-mixing survives into
identity error once three cameras constrain each other. It is **not** ground
truth: no human M1/M2 track exists. So the rule below gives tracking metrics the
power to *confirm* a detector regression and to *fail* a challenger, and
deliberately **denies them the power to overturn an established detection
regression on their own**.

-----------------------------------------------------------------------------
THE RULE, applied as written
-----------------------------------------------------------------------------

Bottom-up is the **incumbent**; ties go to the incumbent. Top-down replaces it
only if it clears **every** one of these:

1. **Reconstruction coverage** — the resolved-frame rate is not lower.
2. **Track jumps** — the pooled rate over the three views is relatively >= 20%
   lower, at p < 0.05. This is the metric that decided the tracking method,
   because it is the only one that tests a tracker against physics rather than
   against the geometry it was built on.
3. **Fragmentation** — the number of tracks needed to cover 90% of assigned
   detections is not higher.
4. **Cross-view consistency** — the pectoral-derived agreement rate is not lower.
5. **Detection quality is not significantly worse** on the metric the 2026-08-09
   decision named as its successor: mouth-linked **`tail_swap`**, per view.

Outcomes other than a clean sweep:

* Wins on some and loses on others -> **trade-off, not a replacement.** The
  incumbent stays and the trade is reported.
* Clears 1-4 but **fails 5** -> **conflict, and the conflict is the finding.**
  Report that the tracking chain absorbs a detection regression, state that
  resolving it needs the encounter ground truth Gate A calls for, and keep the
  incumbent. A tracking metric with no human truth does not retire a measured
  detection defect.
* Indistinguishable -> say so, keep the incumbent, do not rank noise.

**n_groups = 1.** Group K. `K_1_1`, `K_1_2`, `K_1_3` are one trio of fish on
three consecutive days — repeated measures, not replicates. The corpus holds 14
groups; 13 are unmeasured. A production detector choice made on one trio is a
provisional choice, and that is true whichever way it falls.

Run::

    set PYTHONIOENCODING=utf-8
    .venv\\Scripts\\python.exe tools/c06_compare_architectures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c01_s16_predictions as c01
from c02_tracking_probe import CALIBRATION
from s10_shared_node_20k import (
    SHARED_NODE_PX,
    link_by_mouth,
    median_span_px,
    tail_jump_threshold_px,
)

from medaka3d import load_catalog, load_paths
from medaka3d.calib import load_rig
from medaka3d.cohort import ExperimentId
from medaka3d.eval.tracking import two_proportion_z
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

INCUMBENT = "bottomup"
CHALLENGER = "topdown"
#: The tracking method chosen on 2026-08-13. Both architectures are scored with
#: it; comparing two detectors through two different trackers would confound them.
METHOD = "gated"

MIN_RELATIVE_REDUCTION = 0.20
ALPHA = 0.05

#: The 0.3.1-vs-0.3.3 runtime control, measured by `c05 --control`.
CONTROL_CHUNKS = ("top_cam", "side1_cam", "side2_cam")
CONTROL_FRAMES = 25_000


def detection_metrics(detections: c01.Detections, gate_px: float) -> dict:
    """Cardinality, node completeness, shared nodes and `tail_swap`, per view.

    `tail_swap` is imported from `s10` — mouth-linked Hungarian with the same
    per-view gate — because the 2026-08-09 decision named it the successor to
    shared-node for exactly this comparison, and re-implementing it here would
    make the two incomparable.

    Shared-node is reported and is **not** in the rule. That decision established
    it is degenerate for bottom-up: its within-frame tail separation is floored
    an order of magnitude above the 5 px threshold, so the metric measures the
    architecture's peak-NMS floor rather than its part mixing.
    """
    counts = detections.counts()
    three = counts == 3
    start = np.searchsorted(detections.frame_idx, np.flatnonzero(three), side="left")
    rows = start[:, None] + np.arange(3)[None, :]

    nodes = {n: detections.node(n)[rows] for n in detections.nodes}
    missing = {n: float((~np.isfinite(v).all(axis=-1)).mean()) for n, v in nodes.items()}

    # Shared node: two instances placing the same node within 5 px, in a frame.
    shared = {}
    for name in ("mouth", "tail"):
        v = nodes[name]
        d = np.stack(
            [np.linalg.norm(v[:, a] - v[:, b], axis=-1) for a, b in ((0, 1), (0, 2), (1, 2))],
            axis=-1,
        )
        shared[name] = int(np.nansum(d <= SHARED_NODE_PX))

    # tail_swap: link adjacent frames by mouth, then ask how far that instance's
    # tail moved. A stationary mouth with a teleporting tail is a permutation.
    frames = np.flatnonzero(three)
    position = {int(f): i for i, f in enumerate(frames)}
    swap_n = swap_over = rejected = 0
    mouth, tail = nodes["mouth"], nodes["tail"]
    for i, f in enumerate(frames):
        j = position.get(int(f) + 1)
        if j is None:
            continue
        here = [{"mouth": mouth[i][k], "tail": tail[i][k]} for k in range(3)]
        there = [{"mouth": mouth[j][k], "tail": tail[j][k]} for k in range(3)]
        for a in here + there:
            for key in ("mouth", "tail"):
                if not np.isfinite(a[key]).all():
                    a[key] = None
        for index_here, index_there, moved_mouth in link_by_mouth(here, there):
            if moved_mouth > gate_px:
                rejected += 1
                continue
            moved = float(np.linalg.norm(here[index_here]["tail"] - there[index_there]["tail"]))
            swap_n += 1
            swap_over += moved > gate_px

    return {
        "frames": int(counts.size),
        "instances": int(detections.frame_idx.size),
        "mean_instances": float(counts.mean()),
        "frames_lt3": int((counts < 3).sum()),
        "frames_eq3": int(three.sum()),
        "frames_gt3": int((counts > 3).sum()),
        "missing_node_rate": missing,
        "shared_mouth_pairs": shared["mouth"],
        "shared_tail_pairs": shared["tail"],
        "tail_swap_n": swap_n,
        "tail_swap_over": swap_over,
        "tail_swap_rate": float(swap_over / swap_n) if swap_n else float("nan"),
        "link_rejected": rejected,
        "gate_px": gate_px,
    }


def tracking_metrics(work_dir: Path, run: str) -> dict:
    """Read what `c04` already computed for one run, rather than recomputing it."""
    path = work_dir / "c04_score_tracking" / run / "c04_scores.json"
    if not path.is_file():
        raise SystemExit(
            f"no scores at {path}. Run:\n"
            f"    .venv\\Scripts\\python.exe tools/c03_track_3d.py --run {run}\n"
            f"    .venv\\Scripts\\python.exe tools/c04_score_tracking.py --run {run}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def coverage(work_dir: Path, run: str, experiments) -> dict:
    """Resolved-frame rate, from the tracks `c03` wrote."""
    resolved = total = 0
    residual = []
    for experiment in experiments:
        data = np.load(
            work_dir / "c03_track_3d" / run / f"{METHOD}__{experiment}.npz", allow_pickle=False
        )
        used = data["n_views_used"]
        resolved += int((used >= 2).sum())
        total += int(used.size)
        residual.append(data["residual_mm"][used >= 2].ravel())
    finite = np.concatenate(residual)
    finite = finite[np.isfinite(finite)]
    return {
        "resolved": resolved,
        "frames": total,
        "resolved_rate": resolved / total,
        "residual_median_mm": float(np.median(finite)),
    }


def fragments_for_90(scores: dict) -> int:
    """Tracks needed to cover 90% of assigned detections — the honest fragmentation
    number. `tracks_per_animal` counts a 2-frame speck the same as a 100,000-frame
    track and badly overstates the problem."""
    return int(scores["results"][METHOD]["fragmentation"].get("tracks_to_90", 0))


def pooled_jumps(scores: dict) -> tuple[int, int]:
    jumps = sum(scores["results"][METHOD]["jumps"][v]["jumps"] for v in VIEWS)
    pairs = sum(scores["results"][METHOD]["jumps"][v]["adjacent_pairs"] for v in VIEWS)
    return jumps, pairs


def pooled_consistency(scores: dict) -> tuple[int, int]:
    agree = resolved = 0
    for pairs in scores["results"][METHOD]["consistency"].values():
        for name, stats in pairs.items():
            if name.startswith("pectoral:"):
                agree += stats["agree"]
                resolved += stats["resolved"]
    return agree, resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--experiments", nargs="*", default=list(c01.EXPERIMENTS))
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    args = parser.parse_args()

    paths = load_paths()
    out_dir = paths.work_dir / "c06_compare_architectures"
    catalog = load_catalog(paths.asset_root)
    load_rig(args.calibration)  # fails loudly if the promoted calibration moved

    with provenance("compare_architectures", out_dir, name="c06") as record:
        record.params_from(args)
        groups = sorted({ExperimentId.parse(e).group for e in args.experiments})
        record.param("n_groups", len(groups))
        print(f"incumbent  : {INCUMBENT}   challenger: {CHALLENGER}   tracker: {METHOD}")
        print(f"n_groups   : {len(groups)} {groups} — one trio, three consecutive days\n")

        # --- thresholds, derived and printed -------------------------------
        gate_px = {}
        print("gates — half a median body span, per view, from the labels")
        for view in VIEWS:
            key = f"labels.round8.{'top' if view == 'top_cam' else 'side'}.train"
            label_path = catalog.get(key)
            record.input(key, label_path)
            span, n_span = median_span_px(label_path)
            gate_px[view] = tail_jump_threshold_px(span)
            print(f"  {view:<12} {span:7.2f} px ({n_span:>5} instances) -> {gate_px[view]:.0f} px")

        # --- detection side -------------------------------------------------
        print("\n1. DETECTION — full length, both architectures, identical frames")
        detection = {}
        header = (f"{'run':<10} {'view':<11} {'inst/frame':>11} {'<3':>8} {'=3%':>7} {'>3':>7} "
                  f"{'miss mouth%':>12} {'miss tail%':>11} {'shared m/t':>12} {'swap%':>8}")
        print(header)
        for run in (INCUMBENT, CHALLENGER):
            detection[run] = {}
            for view in VIEWS:
                pooled = None
                for experiment in args.experiments:
                    m = detection_metrics(
                        c01.load(paths.work_dir, experiment, view, run), gate_px[view]
                    )
                    if pooled is None:
                        pooled = m
                        pooled["missing"] = {k: [v] for k, v in m["missing_node_rate"].items()}
                    else:
                        for k in ("frames", "instances", "frames_lt3", "frames_eq3",
                                  "frames_gt3", "shared_mouth_pairs", "shared_tail_pairs",
                                  "tail_swap_n", "tail_swap_over", "link_rejected"):
                            pooled[k] += m[k]
                        for k, v in m["missing_node_rate"].items():
                            pooled["missing"][k].append(v)
                pooled["mean_instances"] = pooled["instances"] / pooled["frames"]
                pooled["missing_node_rate"] = {
                    k: float(np.mean(v)) for k, v in pooled.pop("missing").items()
                }
                pooled["tail_swap_rate"] = (
                    pooled["tail_swap_over"] / pooled["tail_swap_n"]
                    if pooled["tail_swap_n"] else float("nan")
                )
                detection[run][view] = pooled
                print(
                    f"{run:<10} {view:<11} {pooled['mean_instances']:>11.3f} "
                    f"{pooled['frames_lt3']:>8,} "
                    f"{100 * pooled['frames_eq3'] / pooled['frames']:>7.2f} "
                    f"{pooled['frames_gt3']:>7,} "
                    f"{100 * pooled['missing_node_rate']['mouth']:>12.3f} "
                    f"{100 * pooled['missing_node_rate']['tail']:>11.3f} "
                    f"{pooled['shared_mouth_pairs']:>5,}/{pooled['shared_tail_pairs']:<6,} "
                    f"{100 * pooled['tail_swap_rate']:>8.4f}"
                )
        print("  shared m/t is reported and is NOT in the rule: s10 established it is\n"
              "  degenerate for bottom-up, whose tail separation is floored above 5 px.")

        # --- tracking side ---------------------------------------------------
        print("\n2. TRACKING — same reconstruction pipeline, same tracker, same thresholds")
        scores = {run: tracking_metrics(paths.work_dir, run) for run in (INCUMBENT, CHALLENGER)}
        cover = {run: coverage(paths.work_dir, run, args.experiments)
                 for run in (INCUMBENT, CHALLENGER)}
        track_header = (f"{'run':<10} {'resolved%':>10} {'residual':>10} {'jumps':>8} "
                        f"{'/pairs':>12} {'jump%':>9} {'frag@90%':>9} {'enc kept%':>10} "
                        f"{'x-view%':>9}")
        print(track_header)
        for run in (INCUMBENT, CHALLENGER):
            jumps, pairs = pooled_jumps(scores[run])
            agree, resolved = pooled_consistency(scores[run])
            enc = scores[run]["results"][METHOD]["encounters"]["preserved_rate"]
            print(
                f"{run:<10} {100 * cover[run]['resolved_rate']:>10.2f} "
                f"{cover[run]['residual_median_mm']:>10.3f} {jumps:>8,} {pairs:>12,} "
                f"{100 * jumps / pairs:>9.4f} {fragments_for_90(scores[run]):>9,} "
                f"{100 * enc:>10.2f} {100 * agree / resolved:>9.2f}"
            )

        # --- the pre-registered rule ------------------------------------------
        print("\n" + "=" * len(track_header))
        print("DECISION RULE — the challenger must clear every clause; ties keep the incumbent")
        print("=" * len(track_header))
        clauses = {}

        clauses["1 coverage not lower"] = {
            "challenger": cover[CHALLENGER]["resolved_rate"],
            "incumbent": cover[INCUMBENT]["resolved_rate"],
            "pass": cover[CHALLENGER]["resolved_rate"] >= cover[INCUMBENT]["resolved_rate"],
        }
        cj, cp = pooled_jumps(scores[CHALLENGER])
        ij, ip = pooled_jumps(scores[INCUMBENT])
        rate_c, rate_i = cj / cp, ij / ip
        relative = (rate_i - rate_c) / rate_i if rate_i else float("nan")
        z, p = two_proportion_z(cj, cp, ij, ip)
        clauses["2 track jumps >=20% lower, p<0.05"] = {
            "challenger": rate_c, "incumbent": rate_i, "relative_reduction": relative,
            "z": z, "p": p,
            "pass": bool(relative >= MIN_RELATIVE_REDUCTION and p < ALPHA),
        }
        clauses["3 fragmentation not higher"] = {
            "challenger": fragments_for_90(scores[CHALLENGER]),
            "incumbent": fragments_for_90(scores[INCUMBENT]),
            "pass": fragments_for_90(scores[CHALLENGER]) <= fragments_for_90(scores[INCUMBENT]),
        }
        ca, cr = pooled_consistency(scores[CHALLENGER])
        ia, ir = pooled_consistency(scores[INCUMBENT])
        clauses["4 cross-view not lower"] = {
            "challenger": ca / cr, "incumbent": ia / ir,
            "pass": (ca / cr) >= (ia / ir),
        }
        swap = {}
        worse_anywhere = False
        for view in VIEWS:
            c = detection[CHALLENGER][view]
            i = detection[INCUMBENT][view]
            _z_swap, p_s = two_proportion_z(
                c["tail_swap_over"], c["tail_swap_n"], i["tail_swap_over"], i["tail_swap_n"]
            )
            worse = bool(c["tail_swap_rate"] > i["tail_swap_rate"] and p_s < ALPHA)
            worse_anywhere |= worse
            swap[view] = {"challenger": c["tail_swap_rate"], "incumbent": i["tail_swap_rate"],
                          "p": p_s, "significantly_worse": worse}
        clauses["5 tail_swap not significantly worse"] = {"per_view": swap,
                                                          "pass": not worse_anywhere}

        for name, clause in clauses.items():
            mark = "PASS" if clause["pass"] else "FAIL"
            if "per_view" in clause:
                detail = "  ".join(
                    f"{v.replace('_cam',''):>5} {100 * s['challenger']:.4f}% vs "
                    f"{100 * s['incumbent']:.4f}% p={s['p']:.2g}"
                    for v, s in clause["per_view"].items()
                )
            else:
                detail = (f"{clause['challenger']:.6g} vs {clause['incumbent']:.6g}"
                          + (f"  rel {100 * clause['relative_reduction']:+.1f}%  "
                             f"p={clause['p']:.3g}" if "p" in clause else ""))
            print(f"  {mark}  {name:<38} {detail}")

        passed = [k for k, v in clauses.items() if v["pass"]]
        if len(passed) == len(clauses):
            verdict = f"ADOPT {CHALLENGER}"
        elif not clauses["5 tail_swap not significantly worse"]["pass"] and len(passed) == 4:
            verdict = (f"CONFLICT — the tracking chain absorbs a detection regression. "
                       f"Keep {INCUMBENT}; resolving this needs encounter ground truth.")
        elif any(clauses[k]["pass"] for k in clauses):
            verdict = f"TRADE-OFF, not a replacement. Keep {INCUMBENT}."
        else:
            verdict = f"No clause passed. Keep {INCUMBENT}."
        print(f"\n  -> {verdict}")
        print(f"     n_groups = {len(groups)}. One trio of the corpus's 14 groups, so this is\n"
              f"     a provisional choice whichever way it falls.")

        result = {
            "n_groups": len(groups), "groups": groups, "method": METHOD,
            "incumbent": INCUMBENT, "challenger": CHALLENGER,
            "gate_px": gate_px, "detection": detection, "coverage": cover,
            "clauses": clauses, "verdict": verdict,
        }
        record.param("result", result)
        out = out_dir / "c06_architectures.json"
        out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        record.output("comparison", out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
