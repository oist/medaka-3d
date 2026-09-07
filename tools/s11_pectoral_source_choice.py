"""S11 — which source should place the pectoral on the frames nobody annotated?

The 3-node retrain needs a pectoral on every round-8 frame, because sleap-nn
0.3.1 has no per-node loss mask: a NaN pectoral is an explicit "nothing here"
target and, since both edges end on the pectoral, it zeroes every PAF channel
for that frame. `s02_merge_pectoral` documents that with the measurement behind
it. So the ~2,000 unannotated moments need a filled pectoral, and this decides
what fills them.

=============================================================================
THE CANDIDATES
=============================================================================

* **interpolation** at the per-view median `t` measured from the human labels —
  the incumbent, used for the round-2 side seed.
* **s07 top-down 3-node** — the model `s05` chose for the round-2 *top* seed,
  where it beat interpolation 2.25 px vs 4.02 px on 15 held-out frames.
* **s03 bottom-up 3-node**.

=============================================================================
WHY THIS CAN BE MEASURED HONESTLY, AND WHERE IT CANNOT
=============================================================================

The comparison needs human pectorals the candidate models never trained on.

* **side: 600 view-frames, genuinely held out.** The side 3-node models trained
  on round 1 only (250 train + 28 val = 278 view-frames). Round 2's batch 1 and
  batch 2 each returned 150 side1 + 150 side2 frames that were **never merged
  into any side training set** — see `s02`'s `ANNOTATION_SOURCES`, which lists
  both round-2 packages as authoritative for `top_cam` alone. Those 600 frames
  are out-of-sample for every side model and are the evidence used here.
* **top: 30 view-frames only.** Every round-2 top frame went into the s07 top
  training split, so the only held-out human top frames are the 30 carved off
  as validation. That is a weak test and is reported as one.

**Only human-*moved* pectorals count as ground truth.** A round-2 point left
exactly where the seed put it is either "checked and correct" or "not looked
at", and those are not the same thing — the 3-node decision document already
flags this. Scoring against an unmoved point would be scoring the seed against
itself, which for the side arm *is* interpolation, and would hand interpolation
a guaranteed win. Round 1 is treated as human throughout, moved or not, because
the experimenter confirmed the unmoved ones were checked.

Run::

    .venv\\Scripts\\python.exe tools/s11_pectoral_source_choice.py --view side
    .venv\\Scripts\\python.exe tools/s11_pectoral_source_choice.py --view top
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s02_merge_pectoral import ROUND1, frame_key, node_map
from s06_locked_benchmark_3node import MODELS_BY_VIEW, run_model

from medaka3d import load_paths
from medaka3d.provenance import provenance

ROUND2 = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-round2-20260806T214904JST"
)
BATCH1_RECOVERED = ROUND2 / "round1_review_recovered" / "recovered_batch01_v2.npy"
BATCH2_EDITS = ROUND2 / "edits" / "batch2_edits_full.pkg.slp"
LEDGER = ROUND2 / "final" / "pectoral_round2_ledger.csv"

#: The splits the candidate 3-node models actually trained on. Any frame in
#: here is in-sample and cannot be used to score them. Checked, not assumed:
#: the first run of this tool reported top at "median 1.45 px, 96.4% within
#: 5 px" because all 439 round-2 top frames sit inside s07's own train+val
#: split. That is the same mistake as the 93-of-113 "held-out" frames in
#: docs/decisions/2026-08-03-k-detection-quality.md, one turn later.
TRAINED_ON = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260804_3node")

#: A returned point counts as human only if it moved this far from its seed.
MOVED_PX = 0.01
#: Match a predicted instance to a human one on mouth+tail; reject beyond this.
MATCH_GATE_PX = 20.0
VIEW_CAMS = {"top": ("top_cam",), "side": ("side1_cam", "side2_cam")}


def load_seed_ledger() -> dict[tuple, np.ndarray]:
    """(experiment, view, frame_idx, instance) -> seeded pectoral xy."""
    seed: dict[tuple, np.ndarray] = {}
    with LEDGER.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["pectoral_x"]:
                continue
            seed[(row["experiment"], row["view"], int(row["frame_idx"]), int(row["instance"]))] = (
                np.array([float(row["pectoral_x"]), float(row["pectoral_y"])])
            )
    return seed


def load_round1() -> dict[tuple, list[dict]]:
    labels = sio.load_slp(str(ROUND1))
    return {frame_key(f): [node_map(i) for i in f.instances] for f in labels.labeled_frames}


def load_batch2() -> dict[tuple, list[dict]]:
    labels = sio.load_slp(str(BATCH2_EDITS))
    return {frame_key(f): [node_map(i) for i in f.instances] for f in labels.labeled_frames}


def load_batch1() -> dict[tuple, list[dict]]:
    """Batch 1 from its ``.npy`` recovery.

    The annotator's full save of batch 1 died inside ``embed_videos`` — the
    ``IndexError`` this project has met before — so ``batch1_edits.pkg.slp`` is
    unreadable and ``save_error.log`` sits beside it. The recovery holds
    ``(experiment, view, frame_idx) -> [[(x, y, visible) x 3] x n_instances]``
    in node order, which is exactly what is needed and is keyed on all three
    fields, so it cannot cross side1 with side2.
    """
    raw = np.load(BATCH1_RECOVERED, allow_pickle=True)
    out: dict[tuple, list[dict]] = {}
    for key, instances in raw:
        experiment, view, frame_idx = key
        out[(experiment, view, int(frame_idx))] = [
            {
                name: (np.array([float(p[0]), float(p[1])]), bool(p[2]))
                for name, p in zip(("mouth", "pectoral", "tail"), points, strict=True)
            }
            for points in instances
        ]
    return out


def in_sample_keys(view: str) -> set[tuple]:
    """Frame keys the candidate 3-node models trained or validated on."""
    keys: set[tuple] = set()
    for split in ("train", "val"):
        path = (
            TRAINED_ON / view / "annotated-only" / f"{split}_labels.3node_annotated.pkg.slp"
        )
        if not path.is_file():
            raise SystemExit(f"cannot verify disjointness: {path} missing")
        keys |= {frame_key(f) for f in sio.load_slp(str(path)).labeled_frames}
    return keys


def human_pectorals(cams: tuple[str, ...]) -> tuple[dict, dict]:
    """Held-out human pectorals, and the in-sample round-1 set for calibration.

    Returns ``(heldout, calibration)``, each ``(key, instance_index) -> dict``
    with ``mouth``/``pectoral``/``tail``. ``heldout`` is round-2 moved points
    only; ``calibration`` is round 1, used to measure the interpolation ``t``.
    """
    seed = load_seed_ledger()
    calibration: dict[tuple, dict] = {}
    for key, instances in load_round1().items():
        if key[1] not in cams:
            continue
        for index, nodes in enumerate(instances):
            calibration[(key, index)] = nodes

    heldout: dict[tuple, dict] = {}
    for source in (load_batch1(), load_batch2()):
        for key, instances in source.items():
            if key[1] not in cams:
                continue
            for index, nodes in enumerate(instances):
                pectoral = nodes.get("pectoral", (None, False))[0]
                seeded = seed.get((key[0], key[1], key[2], index))
                if pectoral is None or seeded is None or not np.isfinite(pectoral).all():
                    continue
                if float(np.linalg.norm(pectoral - seeded)) <= MOVED_PX:
                    continue  # untouched seed: not evidence
                heldout[(key, index)] = nodes
    return heldout, calibration


def chord_t(nodes: dict) -> float | None:
    """Where the pectoral sits along mouth->tail, as a fraction."""
    mouth = nodes.get("mouth", (None, False))[0]
    tail = nodes.get("tail", (None, False))[0]
    pectoral = nodes.get("pectoral", (None, False))[0]
    if any(x is None or not np.isfinite(x).all() for x in (mouth, tail, pectoral)):
        return None
    chord = tail - mouth
    denominator = float(chord @ chord)
    if denominator <= 0:
        return None
    return float((pectoral - mouth) @ chord / denominator)


def predict_pectorals(keys, model_paths, device, batch) -> dict[tuple, list[dict]]:
    """Run a 3-node model over exactly the frames the human labels cover."""
    paths = load_paths()
    by_video: dict[tuple[str, str], list[int]] = {}
    for experiment, view, frame_idx in keys:
        by_video.setdefault((experiment, view), []).append(frame_idx)

    out: dict[tuple, list[dict]] = {}
    for (experiment, view), indices in sorted(by_video.items()):
        video = sio.Video.from_filename(str(paths.video(experiment, view)))
        frames = [
            sio.LabeledFrame(video=video, frame_idx=int(i), instances=[])
            for i in sorted(set(indices))
        ]
        predicted = run_model(frames, model_paths, device, batch)
        for key, instances in predicted.items():
            rows = []
            for instance in instances:
                nodes = {}
                for point in instance.points:
                    xy = np.asarray(point["xy"], dtype=float)
                    nodes[str(point["name"])] = xy if np.isfinite(xy).all() else None
                rows.append(nodes)
            out[key] = rows
    return out


def match(human: dict, predicted: list[dict]) -> dict | None:
    """Pair a human instance with a predicted one on mouth+tail, or nothing."""
    mouth = human.get("mouth", (None, False))[0]
    tail = human.get("tail", (None, False))[0]
    if mouth is None or tail is None:
        return None
    best, best_cost = None, MATCH_GATE_PX
    for candidate in predicted:
        pairs = [
            (mouth, candidate.get("mouth")),
            (tail, candidate.get("tail")),
        ]
        usable = [(a, b) for a, b in pairs if b is not None]
        if len(usable) < 2:
            continue
        cost = float(np.mean([np.linalg.norm(a - b) for a, b in usable]))
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best


def describe(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    a = np.array(errors)
    return {
        "n": len(a),
        "median_px": round(float(np.median(a)), 2),
        "p90_px": round(float(np.percentile(a, 90)), 2),
        "within_5px_pct": round(100.0 * float((a <= 5).mean()), 1),
        "over_20px_pct": round(100.0 * float((a > 20).mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--view", choices=sorted(VIEW_CAMS), default="side")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cams = VIEW_CAMS[args.view]
    paths = load_paths()
    out_dir = paths.work_dir / "s11_pectoral_source_choice"
    out_dir.mkdir(parents=True, exist_ok=True)

    with provenance("pectoral_source_choice", out_dir, name=f"s11_{args.view}") as record:
        record.params_from(args)
        heldout, calibration = human_pectorals(cams)
        trained = in_sample_keys(args.view)
        before = len(heldout)
        heldout = {k: v for k, v in heldout.items() if k[0] not in trained}
        keys = sorted({key for key, _ in heldout})
        print(f"view: {args.view} ({', '.join(cams)})")
        print(f"round-2 moved human instances:             {before}")
        print(f"  dropped as in-sample for the candidates: {before - len(heldout)}")
        print(f"held-out human instances:                  {len(heldout)} over {len(keys)} frames")
        print(f"calibration instances (round 1):           {len(calibration)}")
        record.param("instances_dropped_in_sample", before - len(heldout))
        if not heldout:
            raise SystemExit(
                f"no held-out human pectorals for {args.view}: every annotated frame is "
                f"inside the candidate models' own train/val split, so they cannot be "
                f"scored here. Do not substitute in-sample numbers."
            )
        record.param("heldout_instances", len(heldout))
        record.param("heldout_frames", len(keys))

        # Interpolation t, measured from round 1 — never from the held-out set.
        ts = [t for nodes in calibration.values() if (t := chord_t(nodes)) is not None]
        t_median = float(np.median(ts))
        print(f"interpolation t from round 1: {t_median:.4f}  (n={len(ts)})")
        record.param("interpolation_t", round(t_median, 4))

        results: dict[str, dict] = {}

        interp_errors = []
        for (_key, _index), nodes in heldout.items():
            mouth = nodes["mouth"][0]
            tail = nodes["tail"][0]
            pectoral = nodes["pectoral"][0]
            if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
                continue
            guess = mouth + t_median * (tail - mouth)
            interp_errors.append(float(np.linalg.norm(guess - pectoral)))
        results[f"interpolation t={t_median:.3f}"] = describe(interp_errors)

        for name, model_paths in MODELS_BY_VIEW[args.view].items():
            if "3-node" not in name:
                continue
            if any(not (m / "best.ckpt").is_file() for m in model_paths):
                print(f"  SKIP {name}")
                continue
            print(f"\n--- {name}")
            predicted = predict_pectorals(keys, model_paths, args.device, args.batch_size)
            errors, paired_interp, unmatched, no_pectoral = [], [], 0, 0
            for (key, _index), nodes in heldout.items():
                candidate = match(nodes, predicted.get(key, []))
                if candidate is None:
                    unmatched += 1
                    continue
                guess = candidate.get("pectoral")
                if guess is None:
                    no_pectoral += 1
                    continue
                truth = nodes["pectoral"][0]
                errors.append(float(np.linalg.norm(guess - truth)))
                # The same instance, filled by interpolation. Comparing a model
                # scored on the 83% it could match against interpolation scored
                # on 100% would credit the model for declining the hard ones.
                mouth, tail = nodes["mouth"][0], nodes["tail"][0]
                if np.isfinite(mouth).all() and np.isfinite(tail).all():
                    paired_interp.append(
                        float(np.linalg.norm(mouth + t_median * (tail - mouth) - truth))
                    )
            entry = describe(errors)
            entry["unmatched"] = unmatched
            entry["matched_without_pectoral"] = no_pectoral
            entry["usable_pct"] = round(100.0 * len(errors) / max(len(heldout), 1), 1)
            entry["interpolation_on_same_instances"] = describe(paired_interp)
            beat = sum(
                m < i for m, i in zip(errors, paired_interp, strict=False)
            )
            entry["beats_interpolation_pct"] = round(
                100.0 * beat / max(len(paired_interp), 1), 1
            )
            results[name] = entry

        print(f"\n{'=' * 104}")
        print(f"PECTORAL PLACEMENT vs HUMAN — {args.view}, held-out round-2 moved points")
        print("=" * 104)
        print(
            f"{'source':<42}{'n':>7}{'median':>9}{'p90':>8}{'<=5px%':>9}"
            f"{'>20px%':>9}{'usable%':>9}"
        )
        for name, r in results.items():
            if not r.get("n"):
                print(f"{name:<42}{'no data':>7}")
                continue
            print(
                f"{name:<42}{r['n']:>7}{r['median_px']:>9}{r['p90_px']:>8}"
                f"{r['within_5px_pct']:>9}{r['over_20px_pct']:>9}"
                f"{r.get('usable_pct', 100.0):>9}"
            )
            paired = r.get("interpolation_on_same_instances")
            if paired and paired.get("n"):
                print(
                    f"{'   ^ interpolation, same instances':<42}{paired['n']:>7}"
                    f"{paired['median_px']:>9}{paired['p90_px']:>8}"
                    f"{paired['within_5px_pct']:>9}{paired['over_20px_pct']:>9}"
                    f"{'':>9}   model closer on "
                    f"{r['beats_interpolation_pct']}% of them"
                )
        record.param("results", results)

        out = out_dir / f"s11_results_{args.view}.json"
        out.write_text(
            json.dumps(
                {
                    "view": args.view,
                    "cameras": list(cams),
                    "heldout_instances": len(heldout),
                    "interpolation_t": t_median,
                    "results": results,
                },
                indent=2,
                default=float,
            ),
            encoding="utf-8",
        )
        record.output("results", out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
