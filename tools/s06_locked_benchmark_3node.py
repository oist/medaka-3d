"""S06 — does the 3-node skeleton reduce part-mixing? Measured on the locked benchmark.

This is the question the whole 3-node effort exists to answer, and it has not
been asked once. Everything so far measured *pectoral placement*, which is a
labelling convenience. The experimenter's stated obstacle to tracking is
different: the centered-instance head **links parts from different individuals**,
injecting silent ID switches that no downstream cleverness recovers.

=============================================================================
SPECIFICATION — fixed before the run
=============================================================================

**Frames.** The locked HITL-v2 benchmark, top view: 40 human-annotated frames,
120 instances, verified by k08 to have **zero** overlap with round-7 training,
round-8 train and round-8 val. The 3-node models train on a subset of round-8
train, so they are frame-disjoint from it too — checked here, not assumed.

**Primary metric: shared nodes.** Two predicted instances whose `mouth` (or
`tail`) fall within **5 px** of each other. Round 7's top-down emitted 5 on
these frames and bottom-up 0; the human labels never do it. This needs no
ground truth — it is a property of the prediction — so the benchmark's known
contamination cannot flatter it.

**Secondary, and they need ground truth, so read the contamination line first.**
Instance coverage (how many human animals got a prediction at all), both-node
recall @ 20 px after Hungarian matching, and the predicted instance-count
distribution against the exact-3 cardinality anchor.

**Contamination is measured before any score is read.** 30% of this benchmark's
instances are byte-identical to round-8 output, because it was annotated by
correcting round-8 predictions. A model cannot be ranked by ground truth that
is its own output, so the verbatim share is reported per model and the recall
column is bracketed by it.

**The 3-node model is scored on mouth and tail only.** The benchmark has no
pectoral. That is not a limitation for this question: part-mixing is about
whether mouth and tail belong to the same animal, and the pectoral's role is to
change *how the model groups*, not to be scored.

**n_groups = 1.** Group K. Forty frames of one trio is a locked test set, not a
population.

Run::

    .venv\\Scripts\\python.exe tools/s06_locked_benchmark_3node.py
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
from pathlib import Path

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths
from medaka3d.provenance import provenance

LOCKED = Path(
    r"Z:\StephensU\medaka\benchmarks\sleap_hitl_v2"
    r"\phase1-hitl-v2-preserve-20260802T174200JST-60e13a7746d1\final"
    r"\shared_benchmark_40_hitl_v2_locked.pkg.slp"
)
ROUND7 = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260704")
ROUND8 = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260722_round8")
S03 = Path(r"Z:\StephensU\medaka\work\s03_bottomup_3node")
S07 = Path(r"Z:\StephensU\medaka\work\s07_topdown_3node")

# The 2x2: {top-down, bottom-up} x {2-node, 3-node}, per view.
#
# One asymmetry is deliberate and has to be read with the results. The 2-node
# top-down arm is round 8 itself, trained on 1,382 (top) / 2,764 (side) frames,
# while the 3-node arms train on 409 / 248. A difference between those two is
# therefore skeleton *or* data volume, and this measurement cannot separate
# them. Controlling it would need a 2-node CI on the same frames; that run was
# offered and declined, so the confound is reported rather than removed.
MODELS_BY_VIEW = {
    "top": {
        "topdown 2-node (round8, PRODUCTION)": [
            ROUND8 / "top" / "runs" / "round8_260723_strictmouth.centroid_n1382",
            ROUND8 / "top" / "runs" / "round8_260723_strictmouth.centered_instance_n1382",
        ],
        "topdown 3-node (s07, n409)": [
            ROUND8 / "top" / "runs" / "round8_260723_strictmouth.centroid_n1382",
            S07 / "top" / "s07.topdown_3node.round8params_annotated-only_n409",
        ],
        "bottomup 2-node (round7)": [ROUND7 / "top" / "runs" / "round7s1.bottomup_n1295"],
        "bottomup 3-node (s03, n409)": [
            S03 / "top" / "s03.bottomup_3node.round8_f24_ms64_paf8_annotated-only_n409"
        ],
    },
    "side": {
        "topdown 2-node (round8, PRODUCTION)": [
            ROUND8 / "side" / "runs" / "round8_260723_strictmouth.centroid_n2764",
            ROUND8 / "side" / "runs" / "round8_260723_strictmouth.centered_instance_n2764",
        ],
        "topdown 3-node (s07, n250)": [
            ROUND8 / "side" / "runs" / "round8_260723_strictmouth.centroid_n2764",
            S07 / "side" / "s07.topdown_3node.round8params_annotated-only_n250",
        ],
        "bottomup 2-node (round7)": [ROUND7 / "side" / "runs" / "round7s1.bottomup_n2590"],
        "bottomup 3-node (s03, n250)": [
            S03 / "side" / "s03.bottomup_3node.round8_f24_ms64_annotated-only_n250"
        ],
    },
}

SHARED_NODE_PX = 5.0
RECALL_TOL_PX = 20.0
VERBATIM_PX = 0.01
VIEWS = ("top_cam", "side1_cam", "side2_cam")


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


def axis(instance) -> dict[str, np.ndarray | None]:
    """mouth/tail only. The benchmark has no pectoral, and part-mixing is a
    question about the body axis anyway."""
    out: dict[str, np.ndarray | None] = {"mouth": None, "tail": None}
    for point in instance.points:
        name = str(point["name"])
        if name in out:
            xy = np.asarray(point["xy"], dtype=float)
            out[name] = xy if np.isfinite(xy).all() else None
    return out


def run_model(frames, model_paths, device: str, batch: int) -> dict:
    import sleap_nn

    blank = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=f.video, frame_idx=int(f.frame_idx), instances=[])
            for f in frames
        ],
        videos=list({id(f.video): f.video for f in frames}.values()),
        skeletons=[sio.Skeleton(["mouth", "tail"])],
    )
    predicted = sleap_nn.predict(
        blank, model_paths=[str(m) for m in model_paths], device=device, batch_size=batch
    )
    out: dict[tuple, list] = {}
    for frame in predicted.labeled_frames:
        out.setdefault(frame_key(frame), []).extend(frame.instances)
    return out


def score(frames, predictions) -> dict:
    shared = collections.Counter()
    counts = collections.Counter()
    verbatim = matched = gt_total = recalled = 0
    fp = 0
    errors: list[float] = []

    for frame in frames:
        key = frame_key(frame)
        gt = [axis(i) for i in frame.instances]
        pr = [axis(i) for i in predictions.get(key, [])]
        gt_total += len(gt)
        counts[len(pr)] += 1
        fp += max(0, len(pr) - len(gt))

        # --- shared nodes: a property of the prediction alone ---
        for node in ("mouth", "tail"):
            pts = [p[node] for p in pr if p[node] is not None]
            for a, b in itertools.combinations(pts, 2):
                if float(np.linalg.norm(a - b)) <= SHARED_NODE_PX:
                    shared[node] += 1

        # --- contamination: is this GT instance verbatim this model's output? ---
        for g in gt:
            for p in pr:
                pairs = [(g[n], p[n]) for n in ("mouth", "tail")
                         if g[n] is not None and p[n] is not None]
                if pairs and max(float(np.linalg.norm(a - b)) for a, b in pairs) <= VERBATIM_PX:
                    verbatim += 1
                    break

        if not pr:
            continue
        cost = np.full((len(gt), len(pr)), 1e6)
        for i, g in enumerate(gt):
            for j, p in enumerate(pr):
                d = [float(np.linalg.norm(g[n] - p[n])) for n in ("mouth", "tail")
                     if g[n] is not None and p[n] is not None]
                if d:
                    cost[i, j] = float(np.mean(d))
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols, strict=False):
            if cost[i, j] >= 1e6:
                continue
            matched += 1
            g, p = gt[i], pr[j]
            both = [(g[n], p[n]) for n in ("mouth", "tail")
                    if g[n] is not None and p[n] is not None]
            if len(both) == 2 and all(
                float(np.linalg.norm(a - b)) <= RECALL_TOL_PX for a, b in both
            ):
                recalled += 1
            errors.extend(float(np.linalg.norm(a - b)) for a, b in both)

    err = np.array(errors) if errors else np.array([np.nan])
    return {
        "shared_mouth": shared["mouth"],
        "shared_tail": shared["tail"],
        "shared_total": shared["mouth"] + shared["tail"],
        "gt_instances": gt_total,
        "predicted_instances": sum(k * v for k, v in counts.items()),
        "coverage_pct": round(100.0 * matched / max(gt_total, 1), 2),
        "both_node_recall_pct": round(100.0 * recalled / max(gt_total, 1), 2),
        "extra_instances": fp,
        "verbatim_gt_instances": verbatim,
        "verbatim_pct": round(100.0 * verbatim / max(gt_total, 1), 2),
        "median_node_err_px": round(float(np.nanmedian(err)), 2),
        "instances_per_frame": {str(k): counts[k] for k in sorted(counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--view", choices=("top", "side1", "side2"), default="top")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Benchmark package to score against. Defaults to the "
                             "locked original; pass the corrected v2 to see whether "
                             "a ground-truth fix moves the conclusion.")
    parser.add_argument("--tag", default="",
                        help="Suffix for the results filename, so two benchmarks "
                             "do not overwrite each other.")
    args = parser.parse_args()

    benchmark = args.benchmark or LOCKED
    if not benchmark.is_file():
        raise SystemExit(f"benchmark not found: {benchmark}")
    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "s06_locked_benchmark_3node"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sio.load_slp(str(benchmark))
    want = {"top": "top_cam", "side1": "side1_cam", "side2": "side2_cam"}[args.view]
    frames = [f for f in labels.labeled_frames if frame_key(f)[1] == want]
    models = MODELS_BY_VIEW["top" if args.view == "top" else "side"]
    keys = {frame_key(f) for f in frames}

    with provenance("locked_benchmark_3node", out_dir, name="s06") as record:
        record.params_from(args)
        record.input("benchmark", benchmark)
        record.param("benchmark_path", str(benchmark))

        # Disjointness, checked rather than inherited from k08.
        train3 = sio.load_slp(
            str(Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260804_3node")
                / "top" / "annotated-only" / "train_labels.3node_annotated.pkg.slp")
        )
        overlap_3node = len(keys & {frame_key(f) for f in train3.labeled_frames})
        r8_key = "labels.round8.top.train" if args.view == "top" else "labels.round8.side.train"
        overlap_r8 = len(keys & {
            frame_key(f) for f in sio.load_slp(str(catalog.get(r8_key))).labeled_frames})
        print(f"benchmark: {benchmark.name}")
        print(f"frames: {len(frames)}  "
              f"GT instances: {sum(len(f.instances) for f in frames)}")
        print(f"overlap with the 3-node training split: {overlap_3node}")
        print(f"overlap with round-8 top train:         {overlap_r8}")
        record.param("overlap_3node_train", overlap_3node)
        record.param("overlap_round8_train", overlap_r8)
        if overlap_3node or overlap_r8:
            raise SystemExit("benchmark is not frame-disjoint; scores would be meaningless")

        results = {}
        for name, model_paths in models.items():
            missing = [m for m in model_paths if not (m / "best.ckpt").is_file()]
            if missing:
                print(f"  SKIP {name}: missing {missing}")
                continue
            print(f"\n--- {name}")
            predictions = run_model(frames, model_paths, args.device, args.batch_size)
            results[name] = score(frames, predictions)
            print(json.dumps(results[name], indent=2))
            record.param(f"result_{name}", results[name])

        print(f"\n{'=' * 96}\nSUMMARY - 40 frame-disjoint top frames, n_groups = 1\n{'=' * 96}")
        print(f"{'model':>30} {'shared':>7} {'m/t':>9} {'cover%':>7} {'recall%':>8} "
              f"{'extra':>6} {'verbatim%':>10} {'medErr':>7}")
        for name, r in results.items():
            split = f"{r['shared_mouth']}/{r['shared_tail']}"
            print(f"{name:>30} {r['shared_total']:>7} {split:>9} "
                  f"{r['coverage_pct']:>7} {r['both_node_recall_pct']:>8} "
                  f"{r['extra_instances']:>6} {r['verbatim_pct']:>10} "
                  f"{r['median_node_err_px']:>7}")
        print("\nRead the verbatim column first: a model cannot be ranked by ground "
              "truth that is\nits own output. `shared` needs no ground truth and is "
              "the primary metric.")

        out = out_dir / f"s06_results_{args.view}{args.tag}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        record.output("results", out)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
