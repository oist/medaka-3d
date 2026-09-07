"""S10 — which of four detectors mixes parts between animals least, at 20,000 frames.

=============================================================================
SPECIFICATION — fixed in code before the first inference
=============================================================================

**The question.** A 2x2 — {top-down, bottom-up} x {2-node, 3-node} — already ran
on the locked HITL-v2 benchmark (`s06`). It produced a clean-looking answer that
**cannot be trusted at that size**: 40 frames per view, and the shared-node
counts were 0 / 1 / 6 / 7. Those are counts where a difference of one is a
difference of one frame. This repeats the same measurement, with the same models
and the same scorer, on **20,000 frames per view per model**, where a 1.0% ->
0.7% shift is detectable at 80% power.

**No ground truth is needed, and that is the point.** A shared node is a
property of the prediction alone: two predicted instances placing the same node
within 5 px. Nothing here reads a human label except the two threshold-deriving
medians. That is why this runs on unlabelled video, and why the locked
benchmark's known contamination (19-28% of its instances byte-identical to
round-8 output) cannot reach these numbers.

**Frames.** Four experiments, chosen to span the corpus rather than confirm a
hunch — K is the only group any of these models trained on; I and H are the
worst two groups by exact-3 count under round 7; G is the best. Five evenly
spaced blocks of 1,000 **contiguous** frames each. Not one 5,000-frame block:
5,000 frames at 166 fps is 30 s of a ~35-minute recording, and would sample a
single behavioural episode. Contiguity is required because ``tail_jump`` reads
adjacent frames. Block starts are deterministic — no randomness anywhere.

**Metrics.**

1. ``shared_node`` (primary). Two predicted instances whose ``mouth``, or whose
   ``tail``, lie within 5 px. Reported as *frames with >=1 shared pair / frames*
   and as *pairs per 1,000 frames*, with mouth and tail broken out: every event
   seen so far was a tail, and mouth is the top-down anchor, so a duplicated
   mouth is structurally unlikely and would mean something different.
2. ``tail_jump``. For each predicted tail, the distance to the nearest tail in
   the adjacent frame; the rate beyond half a body length. This is the metric
   the decision rule is pre-registered on, and it is computed exactly as
   specified.
3. ``tail_swap``, added here. **The nearest-tail form of metric 2 cannot detect
   the permutation chimera it was introduced to detect**, and that is a property
   of the arithmetic rather than of the data. A permutation — A takes B's tail,
   B takes C's — leaves the *set* of predicted tail positions unchanged; only
   the instance-to-tail assignment moves. Nearest-neighbour distance is
   assignment-blind, so every tail still finds a tail 0 px away in the next
   frame and ``tail_jump`` reads zero. Demonstrated in
   ``tests/test_shared_node_20k.py``. The variant that does see it links
   instances across adjacent frames by **mouth** (Hungarian, and only where the
   mouth itself moved less than the threshold, since a mouth cannot cross half a
   body length in 6 ms) and then asks how far *that instance's* tail moved. A
   stationary mouth with a teleporting tail is the permutation, and it is
   visible. Reported alongside, never substituted for the pre-registered
   criterion.
4. Instance-count distribution (``<3``, ``=3``, ``>3``). Context, not a decision
   criterion.

**Thresholds are view-specific and derived here, not hardcoded.** The shared-node
radius is 5 px in both views. The tail-jump threshold is half the median
mouth-tail span of the complete instances in ``labels.round8.<view>.train``,
rounded up to 10 px — 112.8 -> 60 px top, 94.0 -> 50 px side. The animal projects
~17% smaller from the side, so carrying the top number over would be too lenient.
Both medians are recomputed at run time and printed.

**Decision rule, fixed before the run and applied as written.** Pooled over all
20,000 frames per model per view, a challenger beats a baseline only if

* its shared-node **frame rate** is lower by a **relative >= 20%**, *and*
* a two-proportion z-test gives **p < 0.05**, *and*
* its **tail-jump rate is not higher**.

A model that wins on shared-node and loses on tail-jump has made the permutation
trade: that is reported as the finding and is **not** called a win. If every
model comes in under ~0.3% the test is underpowered and the honest answer is
"indistinguishable" rather than a ranking of noise.

``tail_swap`` is **not** in the rule — the rule was fixed before the metric was
found to be necessary, and quietly widening a pre-registered rule is how a rule
stops being one. Where ``tail_swap`` and ``tail_jump`` disagree, the
disagreement is reported as a finding.

**A confound to state, not to design around.** The 2-node top-down arm is round 8
itself, trained on 1,382 (top) / 2,764 (side) frames; the 3-node arms trained on
409 / 250. A difference between them is skeleton *or* data volume, and this
measurement cannot separate the two. The 2-node-on-the-same-409-frames control
was offered and declined, so the confound is reported rather than removed.

**n_groups = 4.** Four experiments is four experiments. The independent unit is
the group, so results are pooled by group and never averaged across folders as
if they were replicates.

Run (one view per invocation; the three views are independent)::

    .venv\\Scripts\\python.exe tools/s10_shared_node_20k.py --view top
    .venv\\Scripts\\python.exe tools/s10_shared_node_20k.py --view side1
    .venv\\Scripts\\python.exe tools/s10_shared_node_20k.py --view side2

Predictions are cached per (view, model, experiment) under
``work_dir/s10_shared_node_20k/cache``, so a crash costs one experiment and
``--score-only`` re-reports without re-inferring.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s06_locked_benchmark_3node import (
    MODELS_BY_VIEW,
    SHARED_NODE_PX,
    axis,
    frame_key,
    run_model,
)

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import ExperimentId
from medaka3d.provenance import provenance

#: Chosen to span the corpus: K is the only group any model trained on, I and H
#: are the worst two groups by exact-3 count under round 7, G is the best.
EXPERIMENTS: tuple[str, ...] = (
    "20251225_K_1_2_FCMC",
    "20251224_I_1_2_FCMC",
    "20260106_H_1_2_FCMC",
    "20251223_G_1_2_FCMC",
)

BLOCKS = 5
BLOCK_LEN = 1000

VIEW_CAM = {"top": "top_cam", "side1": "side1_cam", "side2": "side2_cam"}
#: Which model set serves a view. One side pair serves side1 and side2.
VIEW_MODELS = {"top": "top", "side1": "side", "side2": "side"}

#: Relative reduction in shared-node frame rate a challenger must clear.
MIN_RELATIVE_REDUCTION = 0.20
ALPHA = 0.05
#: Below this, 20,000 frames cannot separate the arms; say so rather than rank.
UNDERPOWERED_RATE = 0.003

#: Challenger vs baseline, matched by name prefix so the n409/n250 suffixes do
#: not have to be repeated. Fixed here, before any number exists.
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("topdown 3-node", "topdown 2-node"),
    ("bottomup 3-node", "bottomup 2-node"),
    ("bottomup 2-node", "topdown 2-node"),
    ("bottomup 3-node", "topdown 3-node"),
)


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------
def block_starts(n_frames: int, blocks: int = BLOCKS, block_len: int = BLOCK_LEN) -> list[int]:
    """Deterministic starts for evenly spaced, contiguous blocks.

    Each block sits at the centre of its fifth of the recording. Centring rather
    than spanning [0, n-block_len] keeps the sample off frame 0 — the first
    seconds of a recording are handling, not behaviour — and off the final frame,
    while still covering the whole session.
    """
    if blocks < 1 or block_len < 2:
        raise ValueError("need at least one block of at least two frames")
    segment = n_frames // blocks
    if block_len > segment:
        raise ValueError(
            f"block_len {block_len} exceeds the {segment}-frame segment available "
            f"for {blocks} blocks of {n_frames} frames"
        )
    offset = (segment - block_len) // 2
    return [i * segment + offset for i in range(blocks)]


def block_frames(starts: list[int], block_len: int = BLOCK_LEN) -> list[list[int]]:
    return [list(range(s, s + block_len)) for s in starts]


# --------------------------------------------------------------------------
# Thresholds, derived from the labelled data rather than asserted
# --------------------------------------------------------------------------
def median_span_px(labels_path: Path) -> tuple[float, int]:
    """Median mouth-tail distance over complete instances, and how many there were."""
    labels = sio.load_slp(str(labels_path))
    spans: list[float] = []
    for frame in labels.labeled_frames:
        for instance in frame.instances:
            points = axis(instance)
            if points["mouth"] is not None and points["tail"] is not None:
                spans.append(float(np.linalg.norm(points["mouth"] - points["tail"])))
    if not spans:
        raise SystemExit(f"no complete instances in {labels_path}")
    return float(np.median(spans)), len(spans)


def tail_jump_threshold_px(median_span: float) -> float:
    """Half a body length, rounded up to 10 px.

    Rounding keeps the printed threshold a round number without letting the
    choice drift: 112.8 -> 60, 94.0 -> 50, exactly the table in the brief.
    """
    return float(math.ceil(median_span / 2.0 / 10.0) * 10.0)


# --------------------------------------------------------------------------
# Inference, cached per (view, model, experiment)
# --------------------------------------------------------------------------
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def release_gpu() -> None:
    """Hand the card back between models.

    Measured, not assumed: the 3-node bottom-up model ran at 0.69 fps when it
    came fourth in a process and 1.61 fps alone, because the earlier models'
    cached allocations left it thrashing against a 16 GiB card. Same model, same
    frames, 2.3x apart on nothing but arrival order.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_experiment(
    experiment: str,
    view: str,
    model_paths: list[Path],
    frames: list[int],
    device: str,
    batch: int,
) -> dict[int, list[dict]]:
    """Run one model over one experiment's sampled frames of one view.

    Every returned key is checked to be exactly this (experiment, view). side1
    and side2 share every ``frame_idx``, so a key that is right by index and
    wrong by camera is the one mistake this project has already published once.
    """
    paths = load_paths()
    cam = VIEW_CAM[view]
    video = sio.Video.from_filename(str(paths.video(experiment, cam)))
    labeled = [sio.LabeledFrame(video=video, frame_idx=int(i), instances=[]) for i in frames]

    expected = {frame_key(f) for f in labeled}
    if {(experiment, cam)} != {(e, v) for e, v, _ in expected}:
        raise SystemExit(f"frame_key does not resolve {video.filename} to ({experiment}, {cam})")

    predicted = run_model(labeled, model_paths, device, batch)

    out: dict[int, list[dict]] = {}
    for key, instances in predicted.items():
        if key not in expected:
            raise SystemExit(
                f"prediction key {key} is not one of the frames requested for "
                f"({experiment}, {cam}); refusing to score across cameras"
            )
        out.setdefault(int(key[2]), []).extend(axis(i) for i in instances)
    return out


def cache_path(root: Path, view: str, model: str, experiment: str, plan_id: str) -> Path:
    """One file per (view, model, experiment, frame plan).

    ``plan_id`` is in the name so a smoke run over 100 frames cannot be picked up
    as the cache for the 20,000-frame run. The frame list is re-checked on load
    as well; this just stops the two colliding in the first place.
    """
    return root / f"{view}__{slug(model)}__{experiment}__{plan_id}.npz"


def save_cache(path: Path, frames: list[int], per_frame: dict[int, list[dict]]) -> None:
    idx: list[int] = []
    mouth: list[list[float]] = []
    tail: list[list[float]] = []
    for i in frames:
        for points in per_frame.get(i, []):
            idx.append(i)
            for node, sink in (("mouth", mouth), ("tail", tail)):
                xy = points[node]
                sink.append([float(xy[0]), float(xy[1])] if xy is not None else [np.nan, np.nan])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        requested=np.asarray(frames, dtype=np.int64),
        frame_idx=np.asarray(idx, dtype=np.int64),
        mouth=np.asarray(mouth, dtype=np.float64).reshape(-1, 2),
        tail=np.asarray(tail, dtype=np.float64).reshape(-1, 2),
    )


def load_cache(path: Path) -> tuple[list[int], dict[int, list[dict]]]:
    data = np.load(path)
    requested = [int(i) for i in data["requested"]]
    per_frame: dict[int, list[dict]] = {}
    for i, m, t in zip(data["frame_idx"], data["mouth"], data["tail"], strict=True):
        per_frame.setdefault(int(i), []).append(
            {
                "mouth": None if not np.isfinite(m).all() else m,
                "tail": None if not np.isfinite(t).all() else t,
            }
        )
    return requested, per_frame


# --------------------------------------------------------------------------
# Scoring — every field is an additive count, so pooling is a sum
# --------------------------------------------------------------------------
COUNT_FIELDS = (
    "frames",
    "frames_shared_mouth",
    "frames_shared_tail",
    "frames_shared_any",
    "pairs_mouth",
    "pairs_tail",
    "pairs_at_risk",
    "instances",
    "frames_lt3",
    "frames_eq3",
    "frames_gt3",
    "tail_jump_n",
    "tail_jump_over",
    "tail_swap_n",
    "tail_swap_over",
    "link_rejected",
)


def link_by_mouth(here: list[dict], there: list[dict]) -> list[tuple[int, int, float]]:
    """Match instances across adjacent frames by mouth position.

    Returns ``(index_here, index_there, mouth_distance)`` for every Hungarian
    pairing, gate included so the caller can count what it rejected. Only
    instances carrying *both* nodes take part: an instance with no tail has
    nothing to say about whether a tail moved.
    """
    def complete(instances: list[dict]) -> list[int]:
        return [
            i for i, p in enumerate(instances) if p["mouth"] is not None and p["tail"] is not None
        ]

    usable_here, usable_there = complete(here), complete(there)
    if not usable_here or not usable_there:
        return []
    cost = np.array(
        [
            [float(np.linalg.norm(here[i]["mouth"] - there[j]["mouth"])) for j in usable_there]
            for i in usable_here
        ]
    )
    rows, cols = linear_sum_assignment(cost)
    return [
        (usable_here[r], usable_there[c], float(cost[r, c]))
        for r, c in zip(rows, cols, strict=False)
    ]


def score_blocks(
    blocks: list[list[int]],
    per_frame: dict[int, list[dict]],
    tail_jump_px: float,
) -> Counter:
    """Count shared nodes, tail jumps and cardinalities over contiguous blocks."""
    c: Counter = Counter({field: 0 for field in COUNT_FIELDS})

    for block in blocks:
        for i in block:
            predictions = per_frame.get(i, [])
            c["frames"] += 1
            c["instances"] += len(predictions)
            c["frames_lt3"] += len(predictions) < 3
            c["frames_eq3"] += len(predictions) == 3
            c["frames_gt3"] += len(predictions) > 3

            # A model cannot share a node with an instance it never emitted, so
            # the count of shared pairs has to be read against the number of
            # pairs that existed to be confused. Round 7's bottom-up emits ~2.3
            # instances per frame where the others emit 3.0, which is 1 pair at
            # risk instead of 3 — a 3x head start on the raw rate, earned by
            # missing an animal rather than by keeping it separate.
            c["pairs_at_risk"] += len(predictions) * (len(predictions) - 1) // 2

            shared_here = {}
            for node in ("mouth", "tail"):
                pts = [p[node] for p in predictions if p[node] is not None]
                pairs = sum(
                    float(np.linalg.norm(a - b)) <= SHARED_NODE_PX
                    for a, b in itertools.combinations(pts, 2)
                )
                c[f"pairs_{node}"] += int(pairs)
                shared_here[node] = pairs > 0
                c[f"frames_shared_{node}"] += int(pairs > 0)
            c["frames_shared_any"] += int(any(shared_here.values()))

        # Both temporal metrics need the *adjacent* frame, which only exists
        # inside a block. Across a block boundary the frames are minutes apart.
        for i in block[:-1]:
            first, second = per_frame.get(i, []), per_frame.get(i + 1, [])

            # tail_jump, exactly as pre-registered: nearest tail, no linking.
            here = [p["tail"] for p in first if p["tail"] is not None]
            there = [p["tail"] for p in second if p["tail"] is not None]
            if here and there:
                for a in here:
                    nearest = min(float(np.linalg.norm(a - b)) for b in there)
                    c["tail_jump_n"] += 1
                    c["tail_jump_over"] += nearest > tail_jump_px

            # tail_swap: mouth-linked, and therefore able to see a permutation.
            for index_here, index_there, mouth_moved in link_by_mouth(first, second):
                if mouth_moved > tail_jump_px:
                    c["link_rejected"] += 1
                    continue
                moved = float(
                    np.linalg.norm(first[index_here]["tail"] - second[index_there]["tail"])
                )
                c["tail_swap_n"] += 1
                c["tail_swap_over"] += moved > tail_jump_px
    return c


def rates(c: Counter) -> dict:
    frames = max(int(c["frames"]), 1)
    jumps = max(int(c["tail_jump_n"]), 1)
    at_risk = max(int(c["pairs_at_risk"]), 1)
    return {
        "shared_per_1k_pairs_at_risk": 1000.0
        * (int(c["pairs_mouth"]) + int(c["pairs_tail"]))
        / at_risk,
        "pairs_at_risk_per_frame": int(c["pairs_at_risk"]) / frames,
        **{field: int(c[field]) for field in COUNT_FIELDS},
        "shared_frame_rate": int(c["frames_shared_any"]) / frames,
        "shared_mouth_frame_rate": int(c["frames_shared_mouth"]) / frames,
        "shared_tail_frame_rate": int(c["frames_shared_tail"]) / frames,
        "pairs_mouth_per_1k": 1000.0 * int(c["pairs_mouth"]) / frames,
        "pairs_tail_per_1k": 1000.0 * int(c["pairs_tail"]) / frames,
        "tail_jump_rate": int(c["tail_jump_over"]) / jumps,
        "tail_swap_rate": int(c["tail_swap_over"]) / max(int(c["tail_swap_n"]), 1),
        "mean_instances": int(c["instances"]) / frames,
    }


# --------------------------------------------------------------------------
# The decision rule
# --------------------------------------------------------------------------
def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, float(2.0 * norm.sf(abs(z)))


def judge(challenger: dict, baseline: dict) -> dict:
    """Apply the pre-registered rule. No field here is decided after the fact."""
    c_rate = challenger["shared_frame_rate"]
    b_rate = baseline["shared_frame_rate"]
    relative = (b_rate - c_rate) / b_rate if b_rate > 0 else float("nan")
    z, p = two_proportion_z(
        challenger["frames_shared_any"],
        challenger["frames"],
        baseline["frames_shared_any"],
        baseline["frames"],
    )
    tail_not_higher = challenger["tail_jump_rate"] <= baseline["tail_jump_rate"]
    shared_win = bool(relative >= MIN_RELATIVE_REDUCTION and p < ALPHA)
    underpowered = bool(max(c_rate, b_rate) < UNDERPOWERED_RATE)
    if underpowered:
        verdict = "indistinguishable (underpowered)"
    elif shared_win and tail_not_higher:
        verdict = "challenger wins"
    elif shared_win:
        verdict = "PERMUTATION TRADE - not a win"
    else:
        verdict = "no win"
    return {
        "challenger_shared_frame_rate": c_rate,
        "baseline_shared_frame_rate": b_rate,
        "relative_reduction": relative,
        "z": z,
        "p": p,
        "challenger_tail_jump_rate": challenger["tail_jump_rate"],
        "baseline_tail_jump_rate": baseline["tail_jump_rate"],
        "tail_jump_not_higher": tail_not_higher,
        "shared_node_criterion_met": shared_win,
        "underpowered": underpowered,
        "verdict": verdict,
        # Not in the rule. Reported so that a rule-passing result which the
        # permutation-sensitive metric contradicts cannot be read as clean.
        "challenger_tail_swap_rate": challenger.get("tail_swap_rate"),
        "baseline_tail_swap_rate": baseline.get("tail_swap_rate"),
        "tail_swap_not_higher": (
            None
            if challenger.get("tail_swap_rate") is None or baseline.get("tail_swap_rate") is None
            else bool(challenger["tail_swap_rate"] <= baseline["tail_swap_rate"])
        ),
    }


def match_model(models: dict, prefix: str) -> str | None:
    return next((name for name in models if name.startswith(prefix)), None)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--view", choices=tuple(VIEW_CAM), default="top")
    parser.add_argument("--device", default="auto")
    # 4, not 16. At 1200x1200 the 3-node bottom-up head emits stride-2 confmaps
    # and PAFs; batch 16 peaks at 14.29 GiB on a 16 GiB card and thrashes to
    # 2.8 fps, where batch 4 peaks at 3.80 GiB and runs 18.9 fps — 6.8x on the
    # same arithmetic. Checked rather than assumed to be a throughput knob:
    # against batch 16 over 200 frames, instance counts were identical in every
    # frame and the median per-frame coordinate delta was 0.0002 px (99th
    # percentile 0.0005 px, single worst 1.67 px) against a 5 px threshold.
    # One value is used for every model so no arm gets different treatment.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Name prefixes to run, for splitting one view across invocations. "
             "Throughput differs 55x between these arms, so the fast three "
             "finish in an hour and the slow one takes a night. Results are "
             "cached per model, so a later --score-only pass reports all four "
             "together. Selecting a subset never changes what any model does.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Re-report from cached predictions without running inference.",
    )
    parser.add_argument("--tag", default="", help="Suffix for the results filename.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "s10_shared_node_20k"
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cam = VIEW_CAM[args.view]
    models = MODELS_BY_VIEW[VIEW_MODELS[args.view]]
    if args.models:
        models = {
            name: paths_
            for name, paths_ in models.items()
            if any(name.startswith(prefix) for prefix in args.models)
        }
        if not models:
            raise SystemExit(f"--models {args.models} matched none of {list(MODELS_BY_VIEW)}")
    label_key = f"labels.round8.{VIEW_MODELS[args.view]}.train"

    with provenance("shared_node_20k", out_dir, name=f"s10_{args.view}{args.tag}") as record:
        record.params_from(args)

        # --- thresholds, derived and printed ---------------------------------
        label_path = catalog.get(label_key)
        record.input(label_key, label_path)
        span, n_span = median_span_px(label_path)
        tail_jump_px = tail_jump_threshold_px(span)
        print(f"view: {args.view} ({cam})")
        print(f"threshold source: {label_key}  ({n_span} complete instances)")
        print(f"  median mouth-tail span : {span:.2f} px")
        print(f"  tail_jump threshold    : {tail_jump_px:.0f} px  (half span, up to 10 px)")
        print(f"  shared-node threshold  : {SHARED_NODE_PX:.0f} px")
        record.param("median_span_px", round(span, 3))
        record.param("tail_jump_px", tail_jump_px)
        record.param("shared_node_px", SHARED_NODE_PX)
        record.param("n_span_instances", n_span)

        # --- frames -----------------------------------------------------------
        plan: dict[str, list[list[int]]] = {}
        for experiment in args.experiments:
            video_path = paths.video(experiment, cam)
            n_frames = int(sio.Video.from_filename(str(video_path)).shape[0])
            starts = block_starts(n_frames, args.blocks, args.block_len)
            plan[experiment] = block_frames(starts, args.block_len)
            record.input(f"video:{experiment}", video_path)
            record.param(f"blocks_{experiment}", {"n_frames": n_frames, "starts": starts})
            print(f"{experiment}: {n_frames} frames, blocks at {starts}")
        per_experiment_frames = args.blocks * args.block_len
        total_frames = per_experiment_frames * len(args.experiments)
        groups = sorted({ExperimentId.parse(e).group for e in args.experiments})
        print(
            f"\n{total_frames} frames per model "
            f"({per_experiment_frames} x {len(args.experiments)} experiments), "
            f"n_groups = {len(groups)} {groups}"
        )
        record.param("frames_per_model", total_frames)
        record.param("n_groups", len(groups))
        record.param("groups", groups)

        # --- inference + scoring ---------------------------------------------
        results: dict[str, dict] = {}
        for name, model_paths in models.items():
            missing = [m for m in model_paths if not (m / "best.ckpt").is_file()]
            if missing:
                print(f"  SKIP {name}: missing {missing}")
                continue
            print(f"\n--- {name}")
            record.param(f"model_{slug(name)}", [str(m) for m in model_paths])

            per_exp: dict[str, Counter] = {}
            for experiment in args.experiments:
                frames = [i for block in plan[experiment] for i in block]
                cache = cache_path(
                    cache_dir, args.view, name, experiment, f"b{args.blocks}x{args.block_len}"
                )
                if cache.is_file():
                    cached_frames, per_frame = load_cache(cache)
                    if cached_frames != frames:
                        raise SystemExit(
                            f"{cache} was built for a different frame set; delete it or "
                            f"pass the block arguments it was built with"
                        )
                    print(f"    {experiment}: cached")
                elif args.score_only:
                    raise SystemExit(f"--score-only but no cache at {cache}")
                else:
                    started = time.time()
                    per_frame = predict_experiment(
                        experiment, args.view, model_paths, frames,
                        args.device, args.batch_size,
                    )
                    elapsed = time.time() - started
                    save_cache(cache, frames, per_frame)
                    print(
                        f"    {experiment}: predicted in {elapsed / 60:.1f} min "
                        f"({len(frames) / elapsed:.2f} fps), cached to {cache.name}"
                    )
                per_exp[experiment] = score_blocks(plan[experiment], per_frame, tail_jump_px)

            release_gpu()

            by_group: dict[str, Counter] = {}
            for experiment, counts in per_exp.items():
                group = ExperimentId.parse(experiment).group
                by_group.setdefault(group, Counter()).update(counts)
            pooled: Counter = Counter()
            for counts in per_exp.values():
                pooled.update(counts)

            results[name] = {
                "per_experiment": {e: rates(c) for e, c in per_exp.items()},
                "per_group": {g: rates(c) for g, c in by_group.items()},
                "pooled": rates(pooled),
            }
            print(json.dumps(results[name]["pooled"], indent=2, default=float))

        # --- report ------------------------------------------------------------
        header = (
            f"{'model':<36} {'shared%':>8} {'m%':>6} {'t%':>6} "
            f"{'pairs/1k m':>11} {'pairs/1k t':>11} {'/1k risk':>9} "
            f"{'jump%':>7} {'swap%':>7} {'inst':>6} {'<3':>7} {'=3':>7} {'>3':>7}"
        )
        print(f"\n{'=' * len(header)}")
        print(
            f"POOLED — {args.view}, {total_frames} frames/model, "
            f"n_groups = {len(groups)}, tail_jump > {tail_jump_px:.0f} px"
        )
        print("=" * len(header))
        print(header)
        for name, result in results.items():
            r = result["pooled"]
            print(
                f"{name:<36} {100 * r['shared_frame_rate']:>8.3f} "
                f"{100 * r['shared_mouth_frame_rate']:>6.3f} "
                f"{100 * r['shared_tail_frame_rate']:>6.3f} "
                f"{r['pairs_mouth_per_1k']:>11.3f} {r['pairs_tail_per_1k']:>11.3f} "
                f"{r['shared_per_1k_pairs_at_risk']:>9.3f} "
                f"{100 * r['tail_jump_rate']:>7.3f} {100 * r['tail_swap_rate']:>7.3f} "
                f"{r['mean_instances']:>6.3f} "
                f"{r['frames_lt3']:>7} {r['frames_eq3']:>7} {r['frames_gt3']:>7}"
            )
        print(
            "jump% is the pre-registered nearest-tail metric; swap% is the mouth-linked one.\n"
            "A permutation leaves the tail point-set unchanged, so jump% cannot see it "
            "and swap% can.\n"
            "/1k risk normalises shared pairs by the pairs of instances that existed to be "
            "confused.\nRead it beside `inst`: a model that misses an animal cannot mix it "
            "with anything."
        )

        print("\nBy group (the independent unit) — shared-node frame rate %:")
        group_header = f"{'model':<36} " + " ".join(f"{g:>8}" for g in groups)
        print(group_header)
        for name, result in results.items():
            cells = " ".join(
                f"{100 * result['per_group'][g]['shared_frame_rate']:>8.3f}"
                if g in result["per_group"]
                else f"{'-':>8}"
                for g in groups
            )
            print(f"{name:<36} {cells}")

        print("\nPer experiment — shared-node frame rate % (folders, not replicates):")
        exp_header = f"{'model':<36} " + " ".join(f"{e[:13]:>14}" for e in args.experiments)
        print(exp_header)
        for name, result in results.items():
            cells = " ".join(
                f"{100 * result['per_experiment'][e]['shared_frame_rate']:>14.3f}"
                for e in args.experiments
            )
            print(f"{name:<36} {cells}")

        # --- the pre-registered decision ---------------------------------------
        print(f"\n{'=' * len(header)}")
        print("DECISION RULE — relative reduction >= 20%, p < 0.05, tail-jump not higher")
        print("=" * len(header))
        verdicts = {}
        for challenger_prefix, baseline_prefix in COMPARISONS:
            challenger = match_model(results, challenger_prefix)
            baseline = match_model(results, baseline_prefix)
            if challenger is None or baseline is None:
                continue
            outcome = judge(results[challenger]["pooled"], results[baseline]["pooled"])
            verdicts[f"{challenger_prefix} vs {baseline_prefix}"] = outcome
            print(
                f"{challenger_prefix:>17} vs {baseline_prefix:<17} "
                f"{100 * outcome['challenger_shared_frame_rate']:>7.3f}% vs "
                f"{100 * outcome['baseline_shared_frame_rate']:>7.3f}%  "
                f"rel {100 * outcome['relative_reduction']:>+7.1f}%  "
                f"p = {outcome['p']:.3g}  "
                f"jump {100 * outcome['challenger_tail_jump_rate']:.3f}% vs "
                f"{100 * outcome['baseline_tail_jump_rate']:.3f}%  "
                f"swap {100 * outcome['challenger_tail_swap_rate']:.3f}% vs "
                f"{100 * outcome['baseline_tail_swap_rate']:.3f}%  "
                f"-> {outcome['verdict']}"
            )
            if outcome["verdict"] == "challenger wins" and not outcome["tail_swap_not_higher"]:
                print(
                    "      NOTE: passes the rule but its mouth-linked swap rate is HIGHER. "
                    "The rule cannot see a permutation; this is the trade it was meant to "
                    "catch, arriving through the metric that was left out."
                )

        record.param("results", results)
        record.param("verdicts", verdicts)
        out = out_dir / f"s10_results_{args.view}{args.tag}.json"
        out.write_text(
            json.dumps(
                {
                    "view": args.view,
                    "camera": cam,
                    "median_span_px": span,
                    "tail_jump_px": tail_jump_px,
                    "shared_node_px": SHARED_NODE_PX,
                    "experiments": list(args.experiments),
                    "groups": groups,
                    "n_groups": len(groups),
                    "frames_per_model": total_frames,
                    "blocks": {e: [b[0] for b in plan[e]] for e in args.experiments},
                    "block_len": args.block_len,
                    "results": results,
                    "verdicts": verdicts,
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
