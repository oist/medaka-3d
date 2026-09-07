"""S02 — merge the returned pectoral annotation into the round-8 training labels.

Round 1 of annotation covered 139 synchronised moments (417 view-frames) of the
1,382 in `labels.round8.*.train` — 10%. This builds the 3-node training package
by taking round 8 whole and substituting the corrected frames.

=============================================================================
SPECIFICATION
=============================================================================

**Skeleton.** `mouth -> pectoral -> tail`, node order `(mouth, pectoral, tail)`,
matching the annotation package. Read by name everywhere; `tail` is at index 2.

**Coverage, and why this tool has modes at all.** Only 10% of round-8 train was
annotated. The obvious move — leave the rest `NaN` and let the loss ignore it —
**does not work**, and that was verified rather than assumed:

    pectoral LABELLED : confmap peaks [1.0, 1.0, 1.0]   PAFs populated
    pectoral MISSING  : confmap peaks [1.0, 0.0, 1.0]   ALL PAF channels 0.0

`sleap_nn.data.confidence_maps` calls `torch.nan_to_num(cm)` and
`sleap_nn.data.edge_maps` sets `paf[isnan] = 0.0`. There is **no per-node loss
mask** anywhere in sleap-nn 0.3.1. An unlabelled node is therefore not ignored;
it is an explicit *"nothing here"* target, and because both edges have the
pectoral as an endpoint, a frame without one has **every PAF channel zero**.

For bottom-up the PAFs *are* the grouping mechanism, so training 90% of frames
that way would teach the model that limbs generally do not exist. Hence:

* ``--pectoral-mode annotated-only`` — keep only frames carrying a real
  pectoral. Every label honest; 139 train frames instead of 1,382.
* ``--pectoral-mode interpolate`` — fill the unlabelled 90% at the per-view
  median `t` measured from the annotation itself (top 0.282, side 0.300).
  Keeps all 1,382 frames and every PAF populated. The filled points are
  **synthetic**, and measurably so: against the human labels this rule lands
  within 5 px 66% of the time, median error 3.5 px, against a confmap sigma of
  2.5. Every synthetic instance is counted in the provenance sidecar.

Neither is free. `annotated-only` cannot support a 125 M-parameter model on 139
frames; `interpolate` fabricates 90% of one node and biases it toward a
straight body. Which one is right depends on the question being asked, so this
tool refuses to choose — pass the mode explicitly.

**Mouth and tail come from the annotation where it exists.** 529 of 2,455
mouth/tail points were moved during the pectoral pass, i.e. the annotator
corrected inherited round-8 errors. Discarding those to "change one thing"
would mean knowingly training on labels a human had already marked wrong. The
corrected frames are listed in the sidecar so the effect is auditable.

**Images are round 8's.** The video objects are reused unchanged, so every
training image is byte-identical to what round 8 and round 10 trained on and
only the labels differ. Frames are re-embedded into the output package: writing
a *subset* by reference drops the embedded-video frame map, and sleap-nn then
reads the package as a plain 0..N-1 video and fails on the original frame
indices. Found by a 2-epoch smoke run, not by reasoning.

**The validation split is rebuilt, and it has to be.** Round 8's own val split
is 20 frames drawn before any pectoral existed, so under the behaviour above
every one of them would be an all-zero PAF target — validation would reward a
model for predicting *no limbs*, and `ckpt_save_top_k=1` selects on that loss.
Validation is therefore carved from the **annotated** moments (`--val-moments`,
default 15, chosen by the same stable hash as the seed), so every val frame
carries real pectoral and real PAF targets.

Round 8's original 20 val frames are excluded from both splits rather than
folded into training: they were held out by round 8 and stay that way, so they
remain available as clean extra evaluation. The cost is 1.4% of the frames.

This is a deliberate departure from round 8's split, so a `val/loss` comparison
against round 8 is not meaningful. That costs nothing here: B0.8 already showed
`val/loss` decides nothing in this project, and the real evaluation is the
locked HITL-v2 benchmark, which is frame-disjoint from every split.

Run::

    .venv\\Scripts\\python.exe tools/s02_merge_pectoral.py --view top
    .venv\\Scripts\\python.exe tools/s02_merge_pectoral.py --view side
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from medaka3d import load_catalog, load_paths
from medaka3d.provenance import provenance

VIEWS = ("top_cam", "side1_cam", "side2_cam")
NODE_ORDER = ("mouth", "pectoral", "tail")
EDGES = (("mouth", "pectoral"), ("pectoral", "tail"))
SKELETON_NAME = "medaka_3node_v1"
HASH_TAG = "pectoral-seed-v1"  # same tag the seed sampled with

SOURCES = {
    "top": {"train": "labels.round8.top.train", "val": "labels.round8.top.val"},
    "side": {"train": "labels.round8.side.train", "val": "labels.round8.side.val"},
}

# Annotation sources, applied in order; a later source wins where they overlap.
# Each entry names the views it is authoritative for, and that matters: round 2's
# packages carry *interpolated* side points alongside reviewed top ones, so
# taking a round-2 package wholesale would feed the interpolation rule back in
# dressed as human labelling — the round-8-from-round-7 mistake, one turn later.
ROUND1 = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-seed-20260804T103622JST-3d5f46b7c830\edits\pectoral_edits_full.pkg.slp"
)
ROUND2_BATCH1 = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-round2-20260806T214904JST\final\pectoral_round2_batch01_n150.pkg.slp"
)
ROUND2_BATCH2 = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-round2-20260806T214904JST\edits\batch2_edits_full.pkg.slp"
)
ANNOTATION_SOURCES = [
    (ROUND1, {"top_cam", "side1_cam", "side2_cam"}),
    (ROUND2_BATCH1, {"top_cam"}),
    (ROUND2_BATCH2, {"top_cam"}),
]
ANNOTATION = ROUND1

# --- full-coverage sources, used by --pectoral-mode model ------------------
#
# The list above takes round 2 as authoritative for `top_cam` only, and the
# reason was sound: round 2's *side* seed was interpolation, so importing those
# packages wholesale would feed the interpolation rule back in dressed as human
# labelling. The cost was that 600 view-frames of genuinely reviewed side
# annotation went unused — batch 1 moved 19.6% / 16.1% of side1 / side2
# pectorals and batch 2 moved 30.7% / 36.5%.
#
# `model` mode keeps the protection and spends the data, by classifying every
# round-2 point individually against the seed ledger it was drawn from:
# moved -> human, untouched -> a fill like any other, never a label. Round 1 is
# human throughout, moved or not, because the experimenter confirmed the
# untouched ones were checked.
ROUND2_ROOT = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-round2-20260806T214904JST"
)
#: Batch 1's own `.pkg.slp` is unreadable — its full save died inside
#: `embed_videos` (`save_error.log` sits beside it), the same IndexError this
#: project has met before. This is the recovered payload, keyed on all three of
#: (experiment, view, frame_idx) so it cannot cross side1 with side2.
ROUND2_BATCH1_RECOVERED = ROUND2_ROOT / "round1_review_recovered" / "recovered_batch01_v2.npy"
ROUND2_LEDGER = ROUND2_ROOT / "final" / "pectoral_round2_ledger.csv"
ALL_VIEWS = frozenset(VIEWS)

#: A returned point counts as human only if it moved this far from its seed.
MOVED_PX = 0.01
#: Pair a prediction with a human instance on mouth+tail; reject beyond this.
MATCH_GATE_PX = 20.0

#: What fills a pectoral nobody placed. Chosen by measurement, not preference:
#: on 409 human-moved side instances held out from every side model
#: (`s11_pectoral_source_choice`), the top-down 3-node model beat interpolation
#: on the *same* instances — median 3.26 vs 3.63 px, p90 7.03 vs 10.09, and
#: errors over 20 px in 1.18% against 3.54%. Bottom-up came in *behind*
#: interpolation (closer on 47.0% of instances) and is not used. For `top` the
#: comparison could not be rerun — every annotated top frame is inside s07's
#: own train+val split — so that view rests on s05's 15-frame held-out test,
#: 2.25 px against interpolation's 4.02 px, plus the side result above.
FILL_MODEL_KEY = "topdown 3-node"

# --- the 2026-08-10 review pass ------------------------------------------
REVIEW_ROOT = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-review-all-20260809T200125JST"
)
REVIEW_SEED = REVIEW_ROOT / "pectoral_3node_all_n1382.pkg.slp"
REVIEW_RETURN = REVIEW_ROOT / "edits" / "review_all_edits_full.pkg.slp"


DEFAULT_OUT = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260804_3node")


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


def frame_key(labeled_frame) -> tuple[str | None, str | None, int]:
    path = leaf_video(labeled_frame.video)
    return experiment_of(path), view_of(path), int(labeled_frame.frame_idx)


def node_map(instance) -> dict[str, tuple[np.ndarray, bool]]:
    """``{name: (xy, visible)}`` read by name from the structured points array."""
    return {
        str(p["name"]): (np.asarray(p["xy"], dtype=float), bool(p["visible"]))
        for p in instance.points
    }


def build_instance(nodes: dict[str, tuple[np.ndarray, bool]], skeleton) -> object:
    """Assemble an instance on the skeleton's own node list.

    Driven by the skeleton rather than NODE_ORDER so the 2-node control emits
    two points, not three with a NaN.
    """
    order = [n.name for n in skeleton.nodes]
    coordinates, visibility = [], []
    for name in order:
        xy, visible = nodes.get(name, (np.array([np.nan, np.nan]), False))
        if not np.isfinite(xy).all():
            xy, visible = np.array([np.nan, np.nan]), False
        coordinates.append(xy)
        visibility.append(visible)
    instance = sio.Instance.from_numpy(np.stack(coordinates), skeleton=skeleton)
    for index in range(len(order)):
        instance.points["xy"][index] = coordinates[index]
        instance.points["visible"][index] = visibility[index]
    return instance


def round1_keys() -> set[tuple]:
    """Frame keys from the round-1 pass, used to pin the validation split.

    Validation must not move when a new batch is folded in. Ranking *all*
    annotated keys re-draws it every time, which is how n124 and n259 ended up
    measured on different sets and reported as if they were comparable. Round 1
    is the one source that never grows, so the split is anchored there.
    """
    labels = sio.load_slp(str(ROUND1))
    return {frame_key(f) for f in labels.labeled_frames}


def load_annotation(sources=None) -> tuple[dict[tuple, list], dict]:
    """Corrected instances, keyed by (experiment, view, frame_idx).

    Only the views a source is authoritative for are taken from it.
    """
    out: dict[tuple, list] = {}
    counts: dict[str, int] = {}
    for path, views in (sources or ANNOTATION_SOURCES):
        if not Path(path).is_file():
            raise SystemExit(f"annotation source not found: {path}")
        labels = sio.load_slp(str(path))
        kept = 0
        for frame in labels.labeled_frames:
            key = frame_key(frame)
            if key[1] not in views:
                continue
            out[key] = [node_map(i) for i in frame.instances]
            kept += 1
        counts[Path(path).name] = kept
    return out, counts


def load_seed_ledger() -> dict[tuple, np.ndarray]:
    """(experiment, view, frame_idx, instance) -> the pectoral the seed placed."""
    seed: dict[tuple, np.ndarray] = {}
    with ROUND2_LEDGER.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["pectoral_x"]:
                continue
            seed[(row["experiment"], row["view"], int(row["frame_idx"]), int(row["instance"]))] = (
                np.array([float(row["pectoral_x"]), float(row["pectoral_y"])])
            )
    return seed


def load_batch1_recovered() -> dict[tuple, list[dict]]:
    """Batch 1's returned edits, from the `.npy` its failed save left behind."""
    raw = np.load(ROUND2_BATCH1_RECOVERED, allow_pickle=True)
    out: dict[tuple, list[dict]] = {}
    for key, instances in raw:
        experiment, view, frame_idx = key
        out[(experiment, view, int(frame_idx))] = [
            {
                name: (np.array([float(p[0]), float(p[1])]), bool(p[2]))
                for name, p in zip(NODE_ORDER, points, strict=True)
            }
            for points in instances
        ]
    return out


def reviewed_moments() -> set[tuple[str, int]]:
    """Moments the 2026-08-10 pass touched, against the package it was given.

    A moment counts as reviewed if any node of any instance moved. The
    reviewer sees three synchronised views at once, so a moved point is
    evidence the whole moment was inspected.
    """
    if not REVIEW_RETURN.is_file():
        raise SystemExit(f'review return not found: {REVIEW_RETURN}')
    seed = {
        frame_key(f): [node_map(i) for i in f.instances]
        for f in sio.load_slp(str(REVIEW_SEED), open_videos=False).labeled_frames
    }
    out: set[tuple[str, int]] = set()
    for frame in sio.load_slp(str(REVIEW_RETURN), open_videos=False).labeled_frames:
        key = frame_key(frame)
        before = seed.get(key)
        if before is None:
            continue
        for got, was in zip((node_map(i) for i in frame.instances), before, strict=False):
            for name in NODE_ORDER:
                a = got.get(name, (None, False))[0]
                b = was.get(name, (None, False))[0]
                if a is None or b is None:
                    continue
                if (np.isfinite(a).all() and np.isfinite(b).all()
                        and float(np.linalg.norm(a - b)) > MOVED_PX):
                    out.add((key[0], key[2]))
                    break
    return out


def load_review_return() -> dict[tuple, list[dict]]:
    """Every instance as the reviewer left it."""
    return {
        frame_key(f): [node_map(i) for i in f.instances]
        for f in sio.load_slp(str(REVIEW_RETURN), open_videos=False).labeled_frames
    }


def load_annotation_full() -> tuple[dict[tuple, dict], dict]:
    """Every returned annotation, with each pectoral marked human or not.

    Returns ``{key: {"nodes": [...], "human": [bool, ...]}}``. Mouth and tail
    are taken from the annotation wherever it exists — the annotator corrected
    inherited round-8 errors while placing pectorals, and discarding those to
    "change one thing" would mean knowingly training on labels a human has
    already marked wrong.
    """
    seed = load_seed_ledger()
    out: dict[tuple, dict] = {}
    counts: collections.Counter = collections.Counter()

    def absorb(key, instances, *, trust_all: bool, label: str) -> None:
        human = []
        for index, nodes in enumerate(instances):
            if trust_all:
                pectoral = nodes.get("pectoral", (None, False))[0]
                is_human = pectoral is not None and bool(np.isfinite(pectoral).all())
            else:
                pectoral = nodes.get("pectoral", (None, False))[0]
                seeded = seed.get((key[0], key[1], key[2], index))
                is_human = (
                    pectoral is not None
                    and bool(np.isfinite(pectoral).all())
                    and seeded is not None
                    and float(np.linalg.norm(pectoral - seeded)) > MOVED_PX
                )
            human.append(is_human)
            counts[f"{label}:{key[1]}:{'human' if is_human else 'untouched'}"] += 1
        out[key] = {"nodes": list(instances), "human": human}

    for key, instances in load_round1().items():
        absorb(key, instances, trust_all=True, label="round1")
    for key, instances in load_batch1_recovered().items():
        absorb(key, instances, trust_all=False, label="round2_batch1")
    for frame in sio.load_slp(str(ROUND2_BATCH2)).labeled_frames:
        absorb(
            frame_key(frame),
            [node_map(i) for i in frame.instances],
            trust_all=False,
            label="round2_batch2",
        )
    return out, dict(counts)


def load_round1() -> dict[tuple, list[dict]]:
    labels = sio.load_slp(str(ROUND1))
    return {frame_key(f): [node_map(i) for i in f.instances] for f in labels.labeled_frames}


def match_prediction(human: dict, predicted: list[dict]) -> np.ndarray | None:
    """The pectoral of the predicted instance that best matches this animal.

    Anchored on the human mouth and tail, so a prediction that grouped parts
    from two animals fails the gate instead of contributing its pectoral. This
    is why a model with mediocre grouping is still safe to seed from.
    """
    mouth = human.get("mouth", (None, False))[0]
    tail = human.get("tail", (None, False))[0]
    if mouth is None or tail is None:
        return None
    if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
        return None
    best, best_cost = None, MATCH_GATE_PX
    for candidate in predicted:
        got_mouth, got_tail = candidate.get("mouth"), candidate.get("tail")
        if got_mouth is None or got_tail is None:
            continue
        cost = 0.5 * (
            float(np.linalg.norm(mouth - got_mouth)) + float(np.linalg.norm(tail - got_tail))
        )
        if cost < best_cost:
            best, best_cost = candidate, cost
    if best is None:
        return None
    pectoral = best.get("pectoral")
    return pectoral if pectoral is not None and np.isfinite(pectoral).all() else None


def predict_pectorals(source, view: str, device: str, batch: int) -> dict[tuple, list[dict]]:
    """Run the fill model over round 8's own frames.

    Inference reads the **round-8 package**, not the mp4, so the image a
    pectoral is placed on is byte-identical to the image the model will later
    train on. Re-decoding from video would introduce a second image source for
    no benefit.
    """
    from s06_locked_benchmark_3node import MODELS_BY_VIEW, run_model

    name = next(k for k in MODELS_BY_VIEW[view] if k.startswith(FILL_MODEL_KEY))
    model_paths = MODELS_BY_VIEW[view][name]
    missing = [m for m in model_paths if not (m / "best.ckpt").is_file()]
    if missing:
        raise SystemExit(f"fill model not found: {missing}")
    print(f"  fill model: {name}")

    frames = [
        sio.LabeledFrame(video=f.video, frame_idx=int(f.frame_idx), instances=[])
        for f in source.labeled_frames
    ]
    predicted = run_model(frames, model_paths, device, batch)
    out: dict[tuple, list[dict]] = {}
    for key, instances in predicted.items():
        rows = []
        for instance in instances:
            nodes: dict[str, np.ndarray | None] = {}
            for point in instance.points:
                xy = np.asarray(point["xy"], dtype=float)
                nodes[str(point["name"])] = xy if np.isfinite(xy).all() else None
            rows.append(nodes)
        out[key] = rows
    return out, name


def interpolate_pectoral(nodes: dict, t: float) -> dict:
    """Fill a missing pectoral at ``t`` along mouth->tail. Synthetic, by design."""
    mouth = nodes.get("mouth", (None, False))[0]
    tail = nodes.get("tail", (None, False))[0]
    if mouth is None or tail is None:
        return nodes
    if not (np.isfinite(mouth).all() and np.isfinite(tail).all()):
        return nodes
    out = dict(nodes)
    out["pectoral"] = (mouth + t * (tail - mouth), True)
    return out


def merge(
    catalog, view: str, annotation: dict, out_dir: Path, *,
    mode: str, val_moments: int, tag: str, drop_pectoral: bool = False,
    full: dict | None = None, device: str = "auto", batch: int = 4,
    keep_moments: set | None = None,
) -> dict:
    nodes = list(NODE_ORDER) if not drop_pectoral else ["mouth", "tail"]
    edges = ([list(e) for e in EDGES] if not drop_pectoral else [["mouth", "tail"]])
    skeleton = sio.Skeleton(
        nodes=nodes, edges=edges,
        name=SKELETON_NAME if not drop_pectoral else "medaka_2node_control",
    )
    source = sio.load_slp(str(catalog.get(SOURCES[view]["train"])))

    # Validation is carved from the annotated moments so it carries real
    # pectoral and real PAF targets. Deterministic, by the seed's own hash.
    # Anchored on round 1 only, so adding batches never moves validation.
    anchor = round1_keys()
    annotated_keys = sorted(
        {frame_key(f) for f in source.labeled_frames
         if frame_key(f) in annotation and frame_key(f) in anchor}
    )
    ranked = sorted(annotated_keys, key=lambda k: hashlib.blake2b(
        f"{k[0]}|{k[2]}|{tag}".encode(), digest_size=10
    ).hexdigest())
    val_keys = set(ranked[:val_moments])

    per_view_t = {"top_cam": 0.282, "side1_cam": 0.300, "side2_cam": 0.300}

    # `model` mode covers every frame, so validation can no longer be "the
    # annotated ones" — everything is annotated. It is pinned to the moments
    # whose pectoral is *human*, so val still scores against real labels rather
    # than against the fill model's own output.
    predictions: dict[tuple, list[dict]] = {}
    fill_model_name = None
    if mode == "reviewed-only":
        if not keep_moments:
            raise SystemExit('reviewed-only needs the reviewed moment set')
        # Validation stays anchored on round 1 so it is the same 30 moments
        # every 3-node run so far has used, and remains comparable.
        ranked = sorted(
            {k for k in anchor if (k[0], k[2]) in keep_moments}
            & {frame_key(f) for f in source.labeled_frames},
            key=lambda k: hashlib.blake2b(
                f"{k[0]}|{k[2]}|{tag}".encode(), digest_size=10).hexdigest(),
        )
        val_keys = set(ranked[:val_moments])
    if mode == "model":
        if full is None:
            raise SystemExit("model mode needs the full annotation index")
        human_keys = {
            key for key, entry in full.items() if any(entry["human"])
        } & {frame_key(f) for f in source.labeled_frames}
        ranked = sorted(human_keys & anchor, key=lambda k: hashlib.blake2b(
            f"{k[0]}|{k[2]}|{tag}".encode(), digest_size=10
        ).hexdigest())
        val_keys = set(ranked[:val_moments])
        predictions, fill_model_name = predict_pectorals(source, view, device, batch)

    buckets: dict[str, list] = {"train": [], "val": []}
    counts: dict[str, collections.Counter] = {
        "train": collections.Counter(), "val": collections.Counter()
    }
    videos: dict[int, object] = {}
    moved_nodes = compared_nodes = 0

    ledger: list[dict] = []

    for frame in source.labeled_frames:
        fkey = frame_key(frame)
        corrected = annotation.get(fkey) if mode != "model" else None
        split = "val" if fkey in val_keys else "train"
        instance_sources: list[str] = []

        if mode == "reviewed-only":
            if (fkey[0], fkey[2]) not in keep_moments:
                continue
            node_sets = full[fkey]["nodes"]
            counts[split]["frames_reviewed"] += 1
            for _ in node_sets:
                counts[split]["pectoral_human"] += 1
        elif mode == "model":
            # Per instance, not per frame: round 2 reviewed some animals in a
            # frame and left others, so a frame is rarely all-human or all-fill.
            entry = full.get(fkey)
            base = entry["nodes"] if entry else [node_map(i) for i in frame.instances]
            human_flags = entry["human"] if entry else [False] * len(base)
            if entry is not None:
                for old, new in zip(
                    (node_map(i) for i in frame.instances), base, strict=False
                ):
                    for name in ("mouth", "tail"):
                        a = old.get(name, (None, False))[0]
                        b = new.get(name, (None, False))[0]
                        if a is None or b is None:
                            continue
                        if np.isfinite(a).all() and np.isfinite(b).all():
                            compared_nodes += 1
                            moved_nodes += float(np.linalg.norm(a - b)) > 0.51
                counts[split]["frames_touched_by_annotation"] += 1
            t = per_view_t.get(fkey[1], 0.297)
            node_sets = []
            for index, nodes in enumerate(base):
                if index < len(human_flags) and human_flags[index]:
                    origin = "human"
                    node_sets.append(nodes)
                else:
                    guess = match_prediction(nodes, predictions.get(fkey, []))
                    if guess is not None:
                        origin = "model"
                        node_sets.append({**nodes, "pectoral": (guess, True)})
                    else:
                        origin = "interpolated"
                        node_sets.append(interpolate_pectoral(nodes, t))
                counts[split][f"pectoral_{origin}"] += 1
                instance_sources.append(origin)
                ledger.append(
                    {
                        "experiment": fkey[0], "view": fkey[1], "frame_idx": fkey[2],
                        "instance": index, "split": split, "pectoral_source": origin,
                    }
                )
        elif corrected is not None:
            for old, new in zip(
                (node_map(i) for i in frame.instances), corrected, strict=False
            ):
                for name in ("mouth", "tail"):
                    a, b = old.get(name, (None, False))[0], new.get(name, (None, False))[0]
                    if a is None or b is None:
                        continue
                    if np.isfinite(a).all() and np.isfinite(b).all():
                        compared_nodes += 1
                        moved_nodes += float(np.linalg.norm(a - b)) > 0.51
            node_sets = corrected
            counts[split]["frames_annotated"] += 1
        else:
            if mode == "annotated-only":
                continue
            t = per_view_t.get(fkey[1], 0.297)
            node_sets = [
                interpolate_pectoral(node_map(i), t) for i in frame.instances
            ]
            counts[split]["frames_interpolated"] += 1

        instances = [build_instance(n, skeleton) for n in node_sets]
        if drop_pectoral:
            counts[split]["instances_2node"] += len(instances)
        else:
            pectoral_index = [n.name for n in skeleton.nodes].index("pectoral")
            for position, inst in enumerate(instances):
                xy = inst.points["xy"][pectoral_index]
                if not np.isfinite(xy).all():
                    counts[split]["pectoral_missing"] += 1
                elif mode == "reviewed-only":
                    counts[split]["pectoral_real"] += 1
                elif mode == "model":
                    if instance_sources[position] == "human":
                        counts[split]["pectoral_real"] += 1
                    else:
                        counts[split]["pectoral_synthetic"] += 1
                elif corrected is not None:
                    counts[split]["pectoral_real"] += 1
                else:
                    counts[split]["pectoral_synthetic"] += 1
        videos.setdefault(id(frame.video), frame.video)
        buckets[split].append(
            sio.LabeledFrame(
                video=frame.video, frame_idx=int(frame.frame_idx), instances=instances
            )
        )

    stats: dict[str, dict] = {}
    written: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"annotated-only": "annotated", "interpolate": "interp",
              "model": "full", "reviewed-only": "reviewed"}[mode]
    if drop_pectoral:
        suffix += "_2node"
    for split, frames in buckets.items():
        if not frames:
            raise SystemExit(f"{split} split is empty; check --val-moments")
        used = {id(f.video): f.video for f in frames}
        labels = sio.Labels(
            skeletons=[skeleton], videos=list(used.values()), labeled_frames=frames
        )
        # embed=True is not optional. Saving a *subset* of round-8's frames by
        # reference drops the embedded-video frame map, after which sleap-nn
        # reads the package as a plain 0..N-1 video and the original frame
        # indices go out of range ("Frame index 184626 out of range").
        path = out_dir / f"{split}_labels.3node_{suffix}.pkg.slp"
        sio.save_slp(labels, str(path), embed=True)
        written[split] = str(path)
        c = counts[split]
        total = (c["pectoral_real"] + c["pectoral_synthetic"] + c["pectoral_missing"]
                 + c["instances_2node"])
        stats[split] = {
            "frames": len(frames), "instances": total, "videos": len(used),
            **dict(c),
            "pectoral_real_pct": round(100.0 * c["pectoral_real"] / max(total, 1), 2),
        }
    stats["inherited_mouth_tail_corrected"] = moved_nodes
    stats["inherited_mouth_tail_compared"] = compared_nodes
    stats["mode"] = mode
    stats["val_moments_held_out"] = len(val_keys)
    stats["round8_val_frames_excluded"] = "not used in either split; left clean"
    if mode == "model":
        stats["fill_model"] = fill_model_name
        # Every point's origin, per instance, so "how much of this is human"
        # is answerable without re-deriving it.
        ledger_path = out_dir / f"pectoral_sources_{view}.csv"
        with ledger_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["experiment", "view", "frame_idx", "instance", "split",
                            "pectoral_source"],
            )
            writer.writeheader()
            writer.writerows(ledger)
        written["ledger"] = str(ledger_path)
    return {"stats": stats, "written": written}


def verify(path: Path) -> dict:
    labels = sio.load_slp(str(path))
    skeleton = labels.skeleton
    first = labels.labeled_frames[0]
    image = first.video[int(first.frame_idx)]
    return {
        "nodes": [n.name for n in skeleton.nodes],
        "edges": [(e.source.name, e.destination.name) for e in skeleton.edges],
        "frames": len(labels.labeled_frames),
        "instances": sum(len(f.instances) for f in labels.labeled_frames),
        "image_shape": None if image is None else list(np.shape(image)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--view", choices=sorted(SOURCES), default="top")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pectoral-mode",
                        choices=("annotated-only", "interpolate", "model", "reviewed-only"),
                        required=True,
                        help="How to treat the frames with no pectoral label. "
                             "There is no safe default; see the module docstring.")
    parser.add_argument("--val-moments", type=int, default=15,
                        help="Annotated moments held out for validation. In model "
                             "mode these are drawn from moments carrying a HUMAN "
                             "pectoral, so validation never scores the fill model "
                             "against its own output.")
    parser.add_argument("--device", default="auto", help="model mode only")
    parser.add_argument("--batch-size", type=int, default=4, help="model mode only")
    parser.add_argument("--drop-pectoral", action="store_true",
                        help="Emit the 2-node control: identical frames, identical "
                             "mouth/tail, pectoral removed. Without it a 3-node "
                             "model can only be compared against one trained on "
                             "different data, which confounds skeleton with data.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    for path, _ in ANNOTATION_SOURCES:
        if not Path(path).is_file():
            raise SystemExit(f"annotation source not found: {path}")

    out_dir = args.out_root / args.view / args.pectoral_mode
    with provenance("merge_pectoral", out_dir, name="merge_pectoral") as record:
        record.params_from(args)
        record.param("node_order", list(NODE_ORDER))
        record.param("edges", [list(e) for e in EDGES])
        record.param(
            "synthetic_pectoral_accuracy_vs_human",
            "median 3.5 px, p90 9.6 px, 66% within 5 px (confmap sigma 2.5)",
        )
        for index, (path, _) in enumerate(ANNOTATION_SOURCES):
            record.input(f"annotation{index}", path)
        for key in SOURCES[args.view].values():
            record.input(key, catalog.get(key))

        annotation, per_source = load_annotation()
        record.param("annotation_sources", {str(p): sorted(v)
                                            for p, v in ANNOTATION_SOURCES})
        record.param("annotation_view_frames_per_source", per_source)
        print(f"annotation: {len(annotation)} view-frames  {per_source}")

        full = None
        keep = None
        if args.pectoral_mode == "reviewed-only":
            keep = reviewed_moments()
            returned = load_review_return()
            full = {k: {'nodes': v, 'human': [True] * len(v)}
                    for k, v in returned.items()}
            record.input('review_return', REVIEW_RETURN)
            record.input('review_seed', REVIEW_SEED)
            record.param('reviewed_moments', len(keep))
            print(f'reviewed moments: {len(keep)}')
        if args.pectoral_mode == "model":
            full, classification = load_annotation_full()
            record.input("annotation_batch1_recovered", ROUND2_BATCH1_RECOVERED)
            record.input("annotation_round2_ledger", ROUND2_LEDGER)
            record.param("round2_point_classification", classification)
            record.param("moved_px", MOVED_PX)
            record.param("match_gate_px", MATCH_GATE_PX)
            human = sum(sum(e["human"]) for e in full.values())
            total = sum(len(e["human"]) for e in full.values())
            print(f"returned annotation: {len(full)} view-frames, "
                  f"{human}/{total} pectorals human-placed")

        result = merge(
            catalog, args.view, annotation, out_dir,
            mode=args.pectoral_mode, val_moments=args.val_moments, tag=HASH_TAG,
            drop_pectoral=args.drop_pectoral,
            full=full, device=args.device, batch=args.batch_size,
            keep_moments=keep,
        )
        record.param("merge", result["stats"])

        for split, path in result["written"].items():
            if split == "ledger":  # a CSV of point origins, not a labels package
                record.output("pectoral_source_ledger", path)
                print(f"\nledger: {path}")
                continue
            checks = verify(Path(path))
            record.param(f"verification_{split}", checks)
            record.output(f"{split}_labels", path)
            print(f"\n{split}: {json.dumps(result['stats'][split], indent=2)}")
            print(f"  verified: {checks['nodes']} {checks['edges']} "
                  f"image {checks['image_shape']}")

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
