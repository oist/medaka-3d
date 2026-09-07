"""S01 — build the 3-node annotation seed: mouth -> pectoral -> tail.

The production skeleton has two nodes, ``mouth -> tail``. This tool prepares the
human-in-the-loop bootstrap for a third node, ``pectoral`` (the centre point
between the left and right pectoral fins), by sampling a fraction of the round-8
training set and writing a ``.pkg.slp`` in which the new node is pre-placed by
linear interpolation along the existing mouth->tail segment.

=============================================================================
SPECIFICATION — fixed before the file was written
=============================================================================

**Source.** ``labels.round8.top.train`` + ``labels.round8.side.train``. These
two packages are *exactly* synchronised: 1,382 top frames and 2,764 side frames
form **1,382 moments with all three views present**. Verified, not assumed — see
``--audit``. Round-8 *val* is deliberately excluded so it stays clean for
evaluation.

Cardinality is very nearly exact but **not** exact: of the 4,146 view-frames,
4,143 carry three instances, 2 carry four and 1 carries two. Elsewhere in this
project "every labelled frame has exactly 3 instances" is used as a free label;
that holds for the *top* package and is off by three view-frames across all
three views. Whatever is there is carried through unchanged — this tool adds a
node, it does not adjudicate instance counts.

**Sampling.** ``--fraction`` of the moments per experiment, chosen by a stable
BLAKE2b hash of ``(experiment, frame_idx, tag)`` — deterministic, reproducible
without an RNG, and independent of file order. Selection is greedy in hash order
subject to a **minimum temporal separation** (``--min-gap``, default 166 frames
= 1 s at 166 fps) within an experiment.

That gap matters. Labelled frames in this corpus arrive in short runs spaced
**4 frames apart** — 24 ms — so a plain uniform sample would spend a large part
of its budget on near-duplicate images of the same animal in the same pose. This
is ROADMAP B.3's rule ("split by time window, never randomly") applied to
sampling rather than splitting.

**Interpolation.** For every instance whose mouth *and* tail both carry finite
coordinates::

    pectoral = mouth + t * (tail - mouth)          # t = --t, default 0.30

``t`` is a **seed, not a measurement.** There has never been a pectoral-fin
label in this project, so no fraction here can be derived from data; 0.30
encodes the anatomy — the ``tail`` node sits at the tail base, so mouth->tail
spans the whole trunk and the pectoral girdle sits just behind the operculum,
roughly a third along. The annotator overrides it. It is recorded in the
provenance sidecar so a later analysis can measure the true fraction and say by
how much the prior was wrong.

**Partial instances.** 5.9% of top instances and 5.9% of side instances are
missing a node. Where a parent node has coordinates the pectoral is interpolated
from them **whether or not that node is flagged visible**, and inherits
``visible = mouth.visible and tail.visible``. Where a parent has no coordinates
at all, the pectoral is left ``NaN`` / not-visible for the annotator to place.
The audit reports both populations separately; do not read "partial" as
"hidden".

**Node order.** ``(mouth, pectoral, tail)`` — anatomical. This changes the
positional index of ``tail`` from 1 to 2. Nothing in the repo reads this file
today, so no existing tool is affected; anything written against the *returned*
annotations must index by node name, not position.

**Videos.** Each labelled frame is re-attached to the **original mp4** for its
``(experiment, view)``, giving exactly ``3 x n_experiments`` videos and a
one-hop provenance chain.

That is not cosmetic. Reusing round-8's own video objects gave 26 videos —
round-8's internal partitioning of the same nine recordings, which carries no
meaning for annotation — and a package in that shape **breaks the annotator's
quick save**: the written file listed 25 videos while its frames referenced
index 25, and reloading it raised ``IndexError``. Since quick-save output is
also what the auto-resume path reads back, the failure would have cost an
annotator their session. Verified not to be a regression in the GUI edits: the
2-node original fails the same way on this package and both annotators reload a
9-video one. ``--verify-pixels`` re-decodes every frame and requires it to be
byte-identical to what round 8 embedded, so the change of source cannot quietly
change an image.

Run::

    .venv\\Scripts\\python.exe tools/s01_pectoral_seed.py --audit
    .venv\\Scripts\\python.exe tools/s01_pectoral_seed.py --out-root Z:\\...
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sleap_io as sio

from medaka3d import load_catalog, load_paths
from medaka3d.cohort import CohortError, ExperimentId
from medaka3d.provenance import provenance

SOURCE_KEYS = ("labels.round8.top.train", "labels.round8.side.train")
VIEWS = ("top_cam", "side1_cam", "side2_cam")
NEW_NODE = "pectoral"
NODE_ORDER = ("mouth", NEW_NODE, "tail")
EDGES = (("mouth", NEW_NODE), (NEW_NODE, "tail"))
SKELETON_NAME = "medaka_3node_v1"

DEFAULT_FRACTION = 0.10
DEFAULT_T = 0.30
DEFAULT_MIN_GAP = 166  # frames; 1 s at 166 fps
HASH_TAG = "pectoral-seed-v1"

DEFAULT_OUT_ROOT = Path(r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed")


# ---------------------------------------------------------------------------
# Source inventory
# ---------------------------------------------------------------------------

def leaf_video(video) -> str:
    """Canonical leaf source path, following the embedded-package chain.

    Packages nest: round-8 embeds round-7, which embeds the raw mp4. Only the
    leaf identifies the recording. Shared with ``b03_heldout_comparison``; kept
    here so this tool has no import-time dependency on that module's constants.
    """
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
    for view in VIEWS:
        if f"/{view}/" in path:
            return view
    return None


@dataclass(frozen=True)
class ViewFrame:
    """One labelled view of one moment, tied back to its source package."""

    experiment: str
    frame_idx: int
    view: str
    source_key: str
    leaf_path: str
    labeled_frame: object


def build_inventory(catalog) -> tuple[dict[tuple[str, int], dict[str, ViewFrame]], dict]:
    """Index every labelled view-frame by ``(experiment, frame_idx)`` then view.

    ``frame.video`` is trusted here, and that is checked rather than assumed:
    every labelled frame's index must appear in its own video's ``source_inds``.
    (``b03_heldout_comparison`` re-attaches frames to videos by scanning
    ``source_inds``; that mapping is *ambiguous* for the side packages, where
    side1 and side2 carry identical frame indices, and silently resolves every
    side frame to side1. The direct attribute is both correct and unambiguous.)
    """
    moments: dict[tuple[str, int], dict[str, ViewFrame]] = collections.defaultdict(dict)
    problems: list[str] = []
    counts: dict[str, int] = {}

    for key in SOURCE_KEYS:
        labels = sio.load_slp(str(catalog.get(key)))
        counts[key] = len(labels.labeled_frames)
        for frame in labels.labeled_frames:
            source_inds = getattr(frame.video.backend, "source_inds", None)
            if source_inds is None or int(frame.frame_idx) not in {
                int(x) for x in np.asarray(source_inds)
            }:
                problems.append(f"{key}: frame {frame.frame_idx} not in its video's source_inds")
                continue
            path = leaf_video(frame.video)
            experiment, view = experiment_of(path), view_of(path)
            if experiment is None or view is None:
                problems.append(f"{key}: unresolvable source path {path!r}")
                continue
            moment = moments[(experiment, int(frame.frame_idx))]
            if view in moment:
                problems.append(f"{key}: duplicate {view} for {experiment}@{frame.frame_idx}")
                continue
            moment[view] = ViewFrame(experiment, int(frame.frame_idx), view, key, path, frame)

    meta = {
        "source_frame_counts": counts,
        "moments": len(moments),
        "problems": problems[:20],
        "n_problems": len(problems),
    }
    return dict(moments), meta


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stable_rank(experiment: str, frame_idx: int, tag: str) -> str:
    """Deterministic, order-independent sort key for one moment."""
    payload = f"{experiment}|{frame_idx}|{tag}".encode()
    return hashlib.blake2b(payload, digest_size=10).hexdigest()


def select(
    moments: dict[tuple[str, int], dict[str, ViewFrame]],
    *,
    fraction: float,
    min_gap: int,
    tag: str,
) -> tuple[list[tuple[str, int]], dict]:
    """Choose ``fraction`` of complete moments per experiment, temporally spread.

    Greedy in stable-hash order, accepting a moment only if it is at least
    ``min_gap`` frames from every moment already accepted for that experiment.
    Deterministic: no RNG, no dependence on input ordering.
    """
    complete = {k: v for k, v in moments.items() if len(v) == len(VIEWS)}
    by_experiment: dict[str, list[int]] = collections.defaultdict(list)
    for experiment, frame_idx in complete:
        by_experiment[experiment].append(frame_idx)

    chosen: list[tuple[str, int]] = []
    report: dict[str, dict] = {}
    for experiment in sorted(by_experiment):
        available = sorted(by_experiment[experiment])
        target = int(len(available) * fraction + 0.5)
        ordered = sorted(available, key=lambda i: stable_rank(experiment, i, tag))
        accepted: list[int] = []
        for frame_idx in ordered:
            if len(accepted) >= target:
                break
            if all(abs(frame_idx - other) >= min_gap for other in accepted):
                accepted.append(frame_idx)
        chosen.extend((experiment, i) for i in sorted(accepted))
        report[experiment] = {
            "available_moments": len(available),
            "target": target,
            "selected": len(accepted),
            "short_by": target - len(accepted),
        }

    meta = {
        "complete_moments": len(complete),
        "incomplete_moments": len(moments) - len(complete),
        "selected": len(chosen),
        "per_experiment": report,
    }
    return sorted(chosen), meta


# ---------------------------------------------------------------------------
# Skeleton conversion
# ---------------------------------------------------------------------------

def node_state(instance, name: str) -> tuple[np.ndarray, bool]:
    """Raw ``(xy, visible)`` for one node.

    Read from the structured ``points`` array, never from ``numpy()``: that
    method masks a not-visible point to NaN, which would erase exactly the
    "hidden but positioned" case this tool is required to interpolate through.
    """
    points = instance.points
    index = [p["name"] for p in points].index(name)
    return np.asarray(points["xy"][index], dtype=float), bool(points["visible"][index])


def convert(instance, skeleton, t: float) -> tuple[object, str]:
    """Rebuild one instance on the 3-node skeleton, seeding ``pectoral``.

    Returns the instance and a classification of how the new node was produced.
    """
    mouth_xy, mouth_vis = node_state(instance, "mouth")
    tail_xy, tail_vis = node_state(instance, "tail")
    mouth_ok = bool(np.isfinite(mouth_xy).all())
    tail_ok = bool(np.isfinite(tail_xy).all())

    if mouth_ok and tail_ok:
        pectoral_xy = mouth_xy + t * (tail_xy - mouth_xy)
        pectoral_vis = mouth_vis and tail_vis
        kind = "interpolated" if pectoral_vis else "interpolated_from_hidden"
    else:
        pectoral_xy = np.array([np.nan, np.nan])
        pectoral_vis = False
        kind = (
            "unseeded_no_mouth" if not mouth_ok and tail_ok
            else "unseeded_no_tail" if mouth_ok
            else "unseeded_empty"
        )

    lookup = {"mouth": (mouth_xy, mouth_vis), NEW_NODE: (pectoral_xy, pectoral_vis),
              "tail": (tail_xy, tail_vis)}
    coordinates = np.stack([lookup[name][0] for name in NODE_ORDER])
    new = sio.Instance.from_numpy(coordinates, skeleton=skeleton)
    for index, name in enumerate(NODE_ORDER):
        new.points["xy"][index] = lookup[name][0]
        new.points["visible"][index] = lookup[name][1]
    return new, kind


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_labels(
    moments: dict[tuple[str, int], dict[str, ViewFrame]],
    chosen: list[tuple[str, int]],
    t: float,
    paths,
    *,
    verify_pixels: bool = True,
) -> tuple[object, list[dict], dict]:
    skeleton = sio.Skeleton(nodes=list(NODE_ORDER), edges=[list(e) for e in EDGES],
                            name=SKELETON_NAME)
    labeled_frames: list[object] = []
    videos: dict[tuple[str, str], object] = {}
    ledger: list[dict] = []
    kinds: collections.Counter[str] = collections.Counter()
    mismatches: list[str] = []
    checked = 0

    for experiment, frame_idx in chosen:
        for view in VIEWS:
            view_frame = moments[(experiment, frame_idx)][view]
            source = view_frame.labeled_frame
            instances = []
            for fish_index, instance in enumerate(source.instances):
                new, kind = convert(instance, skeleton, t)
                kinds[kind] += 1
                instances.append(new)
                row = {
                    "experiment": experiment, "frame_idx": frame_idx, "view": view,
                    "fish_index": fish_index, "pectoral_seed": kind,
                }
                for name in NODE_ORDER:
                    xy, visible = node_state(new, name)
                    row[f"{name}_x"] = "" if not np.isfinite(xy[0]) else f"{xy[0]:.6f}"
                    row[f"{name}_y"] = "" if not np.isfinite(xy[1]) else f"{xy[1]:.6f}"
                    row[f"{name}_visible"] = visible
                ledger.append(row)

            key = (experiment, view)
            if key not in videos:
                videos[key] = sio.load_video(str(paths.video(experiment, view)))
            if verify_pixels:
                # The frame_idx is round-8's index into the original recording,
                # so decoding it from the mp4 must reproduce exactly what round 8
                # embedded. Checked on every frame rather than sampled: a seek
                # that is off by one would move the annotator's image away from
                # the labels it is being shown with, and no later step would
                # notice.
                embedded = np.asarray(source.video[frame_idx])
                decoded = np.asarray(videos[key][frame_idx])
                checked += 1
                if embedded.shape != decoded.shape or not np.array_equal(embedded, decoded):
                    mismatches.append(f"{experiment} {view} frame {frame_idx}")

            labeled_frames.append(
                sio.LabeledFrame(video=videos[key], frame_idx=frame_idx, instances=instances)
            )

    if mismatches:
        raise SystemExit(
            f"{len(mismatches)} frame(s) decoded from the mp4 differ from what round 8 "
            f"embedded. The seed would show images that do not match its labels. "
            f"First: {mismatches[:3]}"
        )

    labels = sio.Labels(
        skeletons=[skeleton],
        videos=list(videos.values()),
        labeled_frames=labeled_frames,
    )
    stats = {
        "labeled_frames": len(labeled_frames),
        "videos": len(videos),
        "video_keys": [f"{e}/{v}" for e, v in videos],
        "instances": sum(len(f.instances) for f in labeled_frames),
        "pectoral_seed": dict(kinds),
        "pixel_identical_to_round8": checked if verify_pixels else "not checked",
    }
    return labels, ledger, stats


def verify_output(path: Path) -> dict:
    """Re-open the written package and check the annotator's assumptions hold.

    The multi-view annotator groups panes by walking ``source_video`` to an
    on-disk ``.mp4`` and reading the experiment and view from that path. If
    embedding broke that chain the GUI would open with three blank panes, so it
    is checked here rather than discovered by the annotator.
    """
    labels = sio.load_slp(str(path))
    skeleton = labels.skeleton
    resolved: collections.Counter[str] = collections.Counter()
    unresolved: list[str] = []
    for frame in labels.labeled_frames:
        path_str = leaf_video(frame.video)
        experiment, view = experiment_of(path_str), view_of(path_str)
        if experiment is None or view is None or not path_str.lower().endswith(".mp4"):
            unresolved.append(path_str)
        else:
            resolved[view] += 1
    first = labels.labeled_frames[0]
    image = first.video[int(first.frame_idx)]

    # Re-save without embedding and reload. This is exactly what the
    # annotator's Ctrl+S does, and it is the operation a 26-video package
    # silently broke: the file wrote fewer videos than its frames referenced
    # and reloading raised IndexError. Quick-save output is also what the
    # auto-resume path reads back, so a package that fails here would cost an
    # annotator their session.
    resave = path.parent / f".{path.name}.quicksave_check.slp"
    try:
        sio.save_slp(labels, str(resave), embed=False)
        reloaded = sio.load_slp(str(resave))
        quicksave = {
            "ok": True,
            "labeled_frames": len(reloaded.labeled_frames),
            "videos": len(reloaded.videos),
        }
    except Exception as exc:
        quicksave = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        resave.unlink(missing_ok=True)

    return {
        "nodes": [n.name for n in skeleton.nodes],
        "edges": [(e.source.name, e.destination.name) for e in skeleton.edges],
        "labeled_frames": len(labels.labeled_frames),
        "videos": len(labels.videos),
        "instances": sum(len(f.instances) for f in labels.labeled_frames),
        "frames_resolved_by_view": dict(resolved),
        "unresolved_source_paths": sorted(set(unresolved))[:5],
        "n_unresolved": len(unresolved),
        "first_frame_image_shape": None if image is None else list(np.shape(image)),
        "quicksave_roundtrip": quicksave,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def audit(moments: dict, meta: dict) -> None:
    per_view: collections.Counter[str] = collections.Counter()
    per_experiment: collections.Counter[str] = collections.Counter()
    instance_counts: collections.Counter[int] = collections.Counter()
    node_state_counts: collections.Counter[tuple] = collections.Counter()
    for (experiment, _), views in moments.items():
        per_experiment[experiment] += 1
        for view, view_frame in views.items():
            per_view[view] += 1
            instance_counts[len(view_frame.labeled_frame.instances)] += 1
            for instance in view_frame.labeled_frame.instances:
                for name in ("mouth", "tail"):
                    xy, visible = node_state(instance, name)
                    node_state_counts[(name, bool(np.isfinite(xy).all()), visible)] += 1

    print(f"source frames        {meta['source_frame_counts']}")
    print(f"moments              {meta['moments']}  (problems: {meta['n_problems']})")
    for problem in meta["problems"]:
        print(f"    ! {problem}")
    print(f"view-frames per view {dict(per_view)}")
    print(f"moments per experiment {dict(sorted(per_experiment.items()))}")
    print(f"instances per view-frame {dict(instance_counts)}")
    print("node coordinate/visibility cross-tab (the rule for partial instances "
          "depends on this):")
    for key in sorted(node_state_counts):
        name, finite, visible = key
        print(f"    {name:6s} finite={finite!s:5s} visible={visible!s:5s}  "
              f"n={node_state_counts[key]}")
    hidden = sum(n for (_, finite, visible), n in node_state_counts.items()
                 if finite and not visible)
    print(f"  -> hidden-but-positioned points: {hidden}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--t", type=float, default=DEFAULT_T,
                        help="Interpolation fraction from mouth toward tail.")
    parser.add_argument("--min-gap", type=int, default=DEFAULT_MIN_GAP,
                        help="Minimum frame separation within one experiment.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--audit", action="store_true",
                        help="Report the source inventory and exit without writing.")
    parser.add_argument("--no-verify-pixels", action="store_true",
                        help="Skip the per-frame check that the mp4 decode matches "
                             "what round 8 embedded. Faster; not recommended.")
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit(f"--fraction must be in (0, 1]; got {args.fraction}")
    if not 0.0 <= args.t <= 1.0:
        raise SystemExit(f"--t must be in [0, 1]; got {args.t}")

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)

    print("reading round-8 training labels ...")
    moments, inventory_meta = build_inventory(catalog)
    if inventory_meta["n_problems"]:
        raise SystemExit(
            f"{inventory_meta['n_problems']} source problem(s); refusing to build a seed "
            f"from an inventory that does not resolve. First: {inventory_meta['problems'][:3]}"
        )

    if args.audit:
        audit(moments, inventory_meta)
        return

    chosen, sample_meta = select(
        moments, fraction=args.fraction, min_gap=args.min_gap, tag=HASH_TAG
    )
    short = {e: r for e, r in sample_meta["per_experiment"].items() if r["short_by"]}
    if short:
        print(f"  WARNING: min-gap {args.min_gap} left these experiments short: {short}")

    groups: set[str] = set()
    for experiment, _ in chosen:
        with contextlib.suppress(CohortError):
            groups.add(ExperimentId.parse(experiment).group)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.blake2b(
        f"{HASH_TAG}|{args.fraction}|{args.t}|{args.min_gap}|{stamp}".encode(), digest_size=6
    ).hexdigest()
    run_id = args.run_id or f"pectoral-seed-{stamp}JST-{digest}"
    out_dir = args.out_root / run_id
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pectoral_seed_{len(chosen)}_r8train_v1"
    slp_path = final_dir / f"{stem}.pkg.slp"
    ledger_path = final_dir / f"{stem}_instance_ledger.csv"
    manifest_path = final_dir / f"{stem}_manifest.csv"

    with provenance("pectoral_seed", out_dir, name="pectoral_seed") as record:
        record.params_from(args)
        record.param("new_node", NEW_NODE)
        record.param("node_order", list(NODE_ORDER))
        record.param("edges", [list(e) for e in EDGES])
        record.param("hash_tag", HASH_TAG)
        record.param("interpolation", f"pectoral = mouth + {args.t} * (tail - mouth)")
        record.param("interpolation_is_a_prior_not_a_measurement", True)
        record.param("run_id", run_id)
        record.param("inventory", inventory_meta)
        record.param("sampling", sample_meta)
        record.param("n_groups", len(groups))
        record.param("groups", sorted(groups))
        for key in SOURCE_KEYS:
            record.input(key, catalog.get(key))

        print(f"selected {len(chosen)} moments "
              f"({100.0 * len(chosen) / sample_meta['complete_moments']:.2f}% of "
              f"{sample_meta['complete_moments']})")
        labels, ledger, build_stats = build_labels(
            moments, chosen, args.t, paths, verify_pixels=not args.no_verify_pixels
        )
        record.param("build", build_stats)
        print(f"built {build_stats['labeled_frames']} view-frames, "
              f"{build_stats['instances']} instances")
        print(f"pectoral seeding: {build_stats['pectoral_seed']}")

        print(f"embedding images -> {slp_path}  (this reads ~{build_stats['labeled_frames']} "
              f"frames from /bucket; expect several minutes)")
        sio.save_slp(labels, str(slp_path), embed=True)

        with ledger_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
            writer.writeheader()
            writer.writerows(ledger)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["experiment", "frame_idx", "n_views", "source_keys"])
            for experiment, frame_idx in chosen:
                views = moments[(experiment, frame_idx)]
                writer.writerow([
                    experiment, frame_idx, len(views),
                    ";".join(sorted({v.source_key for v in views.values()})),
                ])

        print("verifying the written package ...")
        checks = verify_output(slp_path)
        record.param("verification", checks)
        print(json.dumps(checks, indent=2))
        if checks["n_unresolved"]:
            raise SystemExit(
                f"{checks['n_unresolved']} labelled frame(s) do not trace back to an .mp4. "
                f"The multi-view annotator resolves its three panes from that path, so it "
                f"would open blank. Examples: {checks['unresolved_source_paths']}"
            )
        if list(checks["nodes"]) != list(NODE_ORDER):
            raise SystemExit(f"skeleton did not round-trip: {checks['nodes']}")
        if not checks["quicksave_roundtrip"]["ok"]:
            raise SystemExit(
                "the package cannot survive a no-embed re-save and reload, which is "
                "what the annotator's Ctrl+S and auto-resume do: "
                f"{checks['quicksave_roundtrip']['error']}"
            )

        record.output("seed_slp", slp_path)
        record.output("instance_ledger", ledger_path)
        record.output("manifest", manifest_path)

    print(f"\nwrote {out_dir}")
    print(f"  {slp_path.name}   {slp_path.stat().st_size / 1e6:.1f} MB")
    print(f"n_groups = {len(groups)} {sorted(groups)} - this is one trio, not a population.")


if __name__ == "__main__":
    sys.exit(main())
