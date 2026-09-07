"""B0.3 (redone) — round-7 architectures on frames held out from their training.

The first attempt scored each checkpoint on its *own* validation split, drawn from
the same labelling pool as its training data. That leaks: top-down's top-view
crossing recall was 84.44% in-sample against 73.33% held out under an identical
criterion, while bottom-up was unchanged — enough to invert the ranking. This
version fixes the frame set.

=============================================================================
PRE-REGISTERED SPECIFICATION — fixed before any number in this run was read
=============================================================================

**Provenance of this specification, stated honestly.** It is adopted from a
pre-registered lab that ran earlier on this machine and whose headline result I
have already seen (top favoured bottom-up by +6.67 crossing points; side did
not). So this run is a *replication of a pre-registered analysis whose result is
known*, which is weaker than true pre-registration. It is still far stronger than
choosing a criterion after seeing one's own output, and the criterion is fixed
here in code before execution. Any reader should weigh it accordingly.

**Frames.** Per view, every labelled frame in the round-8 packages (train + val)
whose canonical ``(leaf source video, frame_idx)`` key does **not** occur in the
matching round-7 package. Both round-7 architectures were trained on the round-7
package, so these frames are held out from both, equally.

**Regimes.** Computed from ground truth only, never from predictions:
  * ``crossing``  — closest pair of *complete* GT instance centroids < 100 px
  * ``separated`` — the complement
Complete = both skeleton nodes labelled. Frames with fewer than two complete
instances are ineligible and are excluded from the regime split (reported).

**Matching.** Per frame, global one-to-one Hungarian assignment minimising mean
distance over nodes visible in both the GT and the predicted instance.

**Primary metric.** Both-node instance recall @ 20 px: a GT instance counts as
detected only if its assigned prediction has *both* nodes present and *each*
within 20 px. Denominator is GT instances in the regime.

**Secondary.** precision@20px, missing/extra instance counts, predicted
shared-node rate (two predictions within 1 px on a node), median and p90
localisation error over matched pairs.

**Decision rule.**
  1. Switch to bottom-up only if it exceeds top-down crossing recall by >= 5.0
     points in **both** view buckets, with no worse precision, miss rate, or
     shared-node rate.
  2. If the views disagree, retain the current architecture and report the
     view-specific result. Do not average the views.
  3. If both architectures lose > 10 points crossing-vs-separated, or both fall
     below 80% crossing recall, flag architecture-independent
     retraining/decoder work.
  4. Otherwise retain top-down.

**Unit.** Report the number of distinct groups. A result from one group is
descriptive for that trio, never a population claim.

Run:
    .venv\\Scripts\\python.exe tools/b03_heldout_comparison.py [--view top|side] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import CohortError, ExperimentId

CROSSING_PX = 100.0
RECALL_TOL_PX = 20.0
SHARED_NODE_PX = 1.0
SWITCH_MARGIN_PTS = 5.0
CROSSING_FLOOR = 80.0
DEGRADATION_FLAG_PTS = 10.0

ROUND7 = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260704")

VIEWS = {
    "top": {
        "round7_key": "labels.round7.top",
        "round8_keys": ("labels.round8.top.train", "labels.round8.top.val"),
        "topdown": [ROUND7 / "top" / "runs" / "round7_260704.centroid_n1295",
                    ROUND7 / "top" / "runs" / "round7_260704.centered_instance_n1295"],
        "bottomup": [ROUND7 / "top" / "runs" / "round7s1.bottomup_n1295"],
    },
    "side": {
        "round7_key": "labels.round7.side",
        "round8_keys": ("labels.round8.side.train", "labels.round8.side.val"),
        "topdown": [ROUND7 / "side" / "runs" / "round7_260704.centroid_n2590",
                    ROUND7 / "side" / "runs" / "round7_260704.centered_instance_n2590"],
        "bottomup": [ROUND7 / "side" / "runs" / "round7s1.bottomup_n2590"],
    },
}


def leaf_video(video) -> str:
    """Canonical leaf source path, following the embedded-package chain.

    Packages nest: round-8 embeds round-7, which embeds an annotator bundle, which
    embeds the raw mp4. Only the leaf identifies the recording.
    """
    current, seen = video, 0
    while seen < 8:
        source = getattr(current, "source_video", None)
        if source is None:
            break
        current, seen = source, seen + 1
    # Case is preserved: group letters are uppercase in folder names and
    # ExperimentId.parse requires them. Lowercasing here silently made every
    # experiment unparseable and reported n_groups=0.
    return str(getattr(current, "filename", "") or "").replace("\\", "/")


def experiment_of(path: str) -> str | None:
    match = re.search(r"/(\d{8}_[A-Z]+_\d+_\d+_[A-Za-z0-9]+)/", path, re.IGNORECASE)
    return match.group(1) if match else None


@dataclass(frozen=True)
class Frame:
    key: tuple[str, int]
    video: object
    frame_idx: int
    instances: list


def points(instance) -> np.ndarray:
    return np.asarray(instance.numpy(), dtype=float)[:, :2]


def complete(array: np.ndarray) -> bool:
    return not np.isnan(array).any()


def load_keys(catalog, key: str) -> set[tuple[str, int]]:
    labels = sio.load_slp(str(catalog.get(key)))
    return {(leaf_video(f.video).lower(), int(f.frame_idx)) for f in labels.labeled_frames}


def build_heldout(catalog, view: str) -> tuple[list[Frame], dict]:
    spec = VIEWS[view]
    exclude = load_keys(catalog, spec["round7_key"])

    frames: list[Frame] = []
    seen: set[tuple[str, int]] = set()
    per_source: dict[str, int] = {}
    source_paths: set[str] = set()
    for key in spec["round8_keys"]:
        labels = sio.load_slp(str(catalog.get(key)))
        # Re-attach each frame to the video whose frame_map actually contains it;
        # the saved video order is a permutation (see b03_run_top_inference).
        index: dict[int, object] = {}
        for video in labels.videos:
            source_inds = getattr(video.backend, "source_inds", None)
            if source_inds is None:
                continue
            for original in np.asarray(source_inds):
                index.setdefault(int(original), video)

        kept = 0
        for frame in labels.labeled_frames:
            video = index.get(int(frame.frame_idx), frame.video)
            path = leaf_video(video)
            canonical = (path.lower(), int(frame.frame_idx))
            if canonical in exclude or canonical in seen:
                continue
            seen.add(canonical)
            frames.append(
                Frame(canonical, video, int(frame.frame_idx), list(frame.instances))
            )
            source_paths.add(path)
            kept += 1
        per_source[key] = kept

    experiments = {e for p in source_paths if (e := experiment_of(p))}
    groups: set[str] = set()
    for name in experiments:
        try:
            groups.add(ExperimentId.parse(name).group)
        except CohortError:
            continue
    meta = {
        "round7_excluded_keys": len(exclude),
        "heldout_frames": len(frames),
        "per_source": per_source,
        "experiments": sorted(experiments),
        "groups": sorted(groups),
        "n_groups": len(groups),
    }
    return frames, meta


def regime_of(frame: Frame) -> str:
    full = [p for p in (points(i) for i in frame.instances) if complete(p)]
    if len(full) < 2:
        return "ineligible"
    closest = min(
        float(np.hypot(*(full[i].mean(axis=0) - full[j].mean(axis=0))))
        for i in range(len(full))
        for j in range(i + 1, len(full))
    )
    return "crossing" if closest < CROSSING_PX else "separated"


def mean_visible_error(gt: np.ndarray, pr: np.ndarray) -> float:
    usable = ~(np.isnan(gt).any(axis=1) | np.isnan(pr).any(axis=1))
    if not usable.any():
        return np.inf
    return float(np.hypot(*(gt[usable] - pr[usable]).T).mean())


def both_node_within(gt: np.ndarray, pr: np.ndarray, tol: float) -> bool:
    if not (complete(gt) and complete(pr)):
        return False
    return bool((np.hypot(*(gt - pr).T) <= tol).all())


VERBATIM_PX = 0.01


def accepted_mask(
    frames: list[Frame], topdown: dict[tuple[str, int], list[np.ndarray]]
) -> dict[tuple[str, int], list[bool]]:
    """Which GT instances are byte-identical to a top-down prediction.

    The round-8 labels were produced by human-correcting round-7 **top-down**
    output, and roughly half of the instances were accepted unchanged: ~48% of
    held-out GT instances match a top-down prediction to <= 0.01 px, against
    **0%** for bottom-up. Those instances are the model's own output fed back as
    ground truth. They are not evidence about top-down, and they penalise any
    competitor for disagreeing with top-down.
    """
    out: dict[tuple[str, int], list[bool]] = {}
    for frame in frames:
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
        pr = topdown.get(frame.key, [])
        flags = []
        for g in gt:
            hit = False
            for p in pr:
                both = ~(np.isnan(g).any(axis=1) | np.isnan(p).any(axis=1))
                if both.any() and float(np.hypot(*(g[both] - p[both]).T).max()) <= VERBATIM_PX:
                    hit = True
                    break
            flags.append(hit)
        out[frame.key] = flags
    return out


def score(
    frames: list[Frame],
    predictions: dict[tuple[str, int], list[np.ndarray]],
    accepted: dict[tuple[str, int], list[bool]] | None = None,
) -> dict:
    """Score. If ``accepted`` is given, GT instances flagged True are excluded."""
    out = {
        r: {"n_gt": 0, "n_pred": 0, "hits": 0, "errors": [], "frames": 0, "shared": 0}
        for r in ("crossing", "separated")
    }
    for frame in frames:
        regime = regime_of(frame)
        if regime == "ineligible":
            continue
        bucket = out[regime]
        bucket["frames"] += 1
        gt = [p for p in (points(i) for i in frame.instances) if not np.isnan(p).all()]
        keep = [True] * len(gt)
        if accepted is not None:
            flags = accepted.get(frame.key, [False] * len(gt))
            keep = [not flags[i] if i < len(flags) else True for i in range(len(gt))]
        pr = predictions.get(frame.key, [])
        bucket["n_gt"] += sum(keep)
        bucket["n_pred"] += len(pr)

        for a in range(len(pr)):
            for b in range(a + 1, len(pr)):
                gaps = np.hypot(*(pr[a] - pr[b]).T)
                if np.nanmin(gaps) < SHARED_NODE_PX:
                    bucket["shared"] += 1

        if not gt or not pr:
            continue
        cost = np.array([[mean_visible_error(g, p) for p in pr] for g in gt])
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
        for r, c in zip(rows, cols, strict=False):
            if not np.isfinite(cost[r, c]) or not keep[r]:
                continue
            bucket["errors"].append(cost[r, c])
            if both_node_within(gt[r], pr[c], RECALL_TOL_PX):
                bucket["hits"] += 1

    summary = {}
    for regime, bucket in out.items():
        errors = np.array(bucket["errors"]) if bucket["errors"] else np.array([np.nan])
        summary[regime] = {
            "frames": bucket["frames"],
            "n_gt": bucket["n_gt"],
            "n_pred": bucket["n_pred"],
            "recall": 100.0 * bucket["hits"] / bucket["n_gt"] if bucket["n_gt"] else 0.0,
            "precision": 100.0 * bucket["hits"] / bucket["n_pred"] if bucket["n_pred"] else 0.0,
            "missing": bucket["n_gt"] - bucket["hits"],
            "shared_node_pairs": bucket["shared"],
            "median_err": float(np.nanmedian(errors)),
            "p90_err": float(np.nanpercentile(errors, 90)),
        }
    return summary


def run_inference(frames: list[Frame], models: list[Path], device: str, batch: int) -> dict:
    import sleap_nn

    videos = list({id(f.video): f.video for f in frames}.values())
    labels = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=f.video, frame_idx=f.frame_idx, instances=[])
            for f in frames
        ],
        videos=videos,
        skeletons=[sio.Skeleton(["mouth", "tail"])],
    )
    predicted = sleap_nn.predict(
        labels, model_paths=[str(m) for m in models], device=device, batch_size=batch
    )
    out: dict[tuple[str, int], list[np.ndarray]] = {}
    by_index: dict[int, list[np.ndarray]] = {}
    for frame in predicted.labeled_frames:
        by_index.setdefault(int(frame.frame_idx), []).extend(points(i) for i in frame.instances)
    for frame in frames:
        out[frame.key] = by_index.get(frame.frame_idx, [])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["top", "side", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "b03_heldout"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(__doc__.split("Run:")[0])
    results: dict[str, dict] = {}
    views = ("top", "side") if args.view == "both" else (args.view,)

    for view in views:
        frames, meta = build_heldout(catalog, view)
        if args.limit:
            frames = frames[: args.limit]
        print(f"\n{'=' * 78}\n{view.upper()}\n{'=' * 78}")
        print(
            f"round-7 package keys excluded: {meta['round7_excluded_keys']}\n"
            f"held-out frames: {len(frames)}  (per source: {meta['per_source']})\n"
            f"experiments: {meta['experiments']}\n"
            f"n_groups: {meta['n_groups']} {meta['groups']}"
        )
        regimes = [regime_of(f) for f in frames]
        print(
            f"regimes: crossing={regimes.count('crossing')} "
            f"separated={regimes.count('separated')} "
            f"ineligible={regimes.count('ineligible')}"
        )

        view_result = {"meta": meta}
        predictions = {}
        for arch in ("topdown", "bottomup"):
            models = VIEWS[view][arch]
            missing = [m for m in models if not m.is_dir()]
            if missing:
                print(f"\n{arch}: SKIPPED, missing {[str(m) for m in missing]}")
                continue
            predictions[arch] = run_inference(frames, models, args.device, args.batch_size)

        accepted = accepted_mask(frames, predictions["topdown"]) if "topdown" in predictions else {}
        n_acc = sum(sum(v) for v in accepted.values())
        n_all = sum(len(v) for v in accepted.values())
        meta["accepted_instances"] = n_acc
        meta["total_instances"] = n_all
        print(
            f"\nLABEL CONTAMINATION: {n_acc}/{n_all} "
            f"({100 * n_acc / max(n_all, 1):.1f}%) GT instances are byte-identical to a\n"
            f"round-7 TOP-DOWN prediction — the round-8 labels were made by correcting\n"
            f"top-down output and accepting about half of it unchanged."
        )

        for arm, mask in (("ALL instances", None), ("CORRECTED only", accepted)):
            bias = (
                "favours top-down (its own output is ~half the ground truth)"
                if mask is None
                else "favours bottom-up (keeps only where the human overrode top-down)"
            )
            print(f"\n--- arm: {arm} — {bias}")
            for arch, preds in predictions.items():
                summary = score(frames, preds, mask)
                view_result.setdefault(arm, {})[arch] = summary
                print(f"  {arch}  [{', '.join(m.name for m in VIEWS[view][arch])}]")
                for regime in ("separated", "crossing"):
                    s = summary[regime]
                    print(
                        f"    {regime:10s} {s['frames']:4d} fr  {s['n_gt']:4d} GT /"
                        f" {s['n_pred']:4d} pred  recall@20px {s['recall']:6.2f}%"
                        f"  precision {s['precision']:6.2f}%  missing {s['missing']:3d}"
                        f"  shared {s['shared_node_pairs']:3d}"
                        f"  err {s['median_err']:.2f}/{s['p90_err']:.2f} px"
                    )
        results[view] = view_result

    decide(results)
    path = out_dir / "heldout_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def decide(results: dict) -> None:
    print(f"\n{'=' * 78}\nDECISION RULE (fixed before the run)\n{'=' * 78}")
    for arm in ("ALL instances", "CORRECTED only"):
        deltas, flags = {}, []
        print(f"\n[{arm}]")
        for view, data in results.items():
            block = data.get(arm, {})
            if "topdown" not in block or "bottomup" not in block:
                print(f"  {view}: incomplete, cannot evaluate")
                continue
            td, bu = block["topdown"], block["bottomup"]
            delta = bu["crossing"]["recall"] - td["crossing"]["recall"]
            deltas[view] = delta
            print(
                f"  {view}: bottom-up - top-down at crossings = {delta:+.2f} pts "
                f"(need >= +{SWITCH_MARGIN_PTS} to switch)"
            )
            for name, s in (("top-down", td), ("bottom-up", bu)):
                drop = s["separated"]["recall"] - s["crossing"]["recall"]
                if drop > DEGRADATION_FLAG_PTS:
                    flags.append(f"{view}/{name} loses {drop:.1f} pts at crossings")
                if s["crossing"]["recall"] < CROSSING_FLOOR:
                    flags.append(
                        f"{view}/{name} crossing recall "
                        f"{s['crossing']['recall']:.1f}% < {CROSSING_FLOOR}%"
                    )
        if deltas and all(d >= SWITCH_MARGIN_PTS for d in deltas.values()):
            print("  -> rule 1: SWITCH to bottom-up (satisfied in every view)")
        elif deltas and any(d >= SWITCH_MARGIN_PTS for d in deltas.values()):
            print("  -> rule 2: VIEWS DISAGREE, retain top-down, report per view")
        else:
            print("  -> rule 4: RETAIN top-down")
        for f in sorted(set(flags)):
            print(f"  -> rule 3 flag: {f}")

    print(
        "\nNEITHER ARM IS AN UNBIASED MEASUREMENT. The 'ALL' arm scores top-down\n"
        "partly against its own accepted output; the 'CORRECTED' arm keeps only the\n"
        "instances a human overrode, i.e. exactly where top-down was wrong. The true\n"
        "value is bracketed by the two. A clean answer needs labels produced\n"
        "independently of both checkpoints."
    )


if __name__ == "__main__":
    main()
