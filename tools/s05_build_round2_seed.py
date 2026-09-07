"""S05 — build the round-2 annotation seed for the frames round 1 did not cover.

Round 1 annotated 139 of the 1,382 round-8 moments. This seeds the remaining
**1,243 moments x 3 views** for review in the same multi-view GUI, so the pass
is a *correction* pass rather than a placement pass.

=============================================================================
WHERE EACH PECTORAL COMES FROM, AND WHY IT DIFFERS BY VIEW
=============================================================================

**top: the trained model.** On 15 held-out human frames it beat interpolation
by a clear margin — pectoral median **2.25 px vs 4.02 px**, closer on **77.3%**
of instances, p90 6.42 vs 12.41, and no error above 20 px. It earns its place.

**side1 / side2: interpolation at t = 0.300.** The side model's advantage over
interpolation is small (median 3.05 vs 3.31 px, closer on 57.3%), and it brings
failure modes interpolation cannot have: 2.5% of its predictions disagree by
more than 20 px, and of those **37% have no detected mouth** and **28% put the
pectoral nearer another animal's body axis** — neighbours and surface
reflections. A retrain on the interpolated side corpus is running; when it is
measured this choice can be revisited.

That asymmetry is not arbitrary. Medaka bend laterally and barely
dorsoventrally, so the top camera sees the curvature a straight chord cannot
represent, while from the side mouth and tail project nearly collinear and
interpolation is already close to optimal. The measured perpendicular spread
agrees: top 0.122 of span p10-p90, side 0.094 / 0.100.

**Mouth and tail are always the human round-8 labels**, never the model's. The
model is used for one node only.

**Every seeded point is provisional.** Round 1 moved 90.6% of its seeds, and
the experimenter reported the rest were checked and judged correct. The same
standard applies here: a point left untouched should mean "looked at and
right", not "not looked at".

**One video per (experiment, view), backed by the original mp4.** Not
negotiable, and learned the hard way. The first build reused round-8's own
embedded ``Video`` objects, which put **15 videos** in a single batch because
round 8 partitions the same nine recordings that way. The annotator's save
deduplicates videos by mp4 source path, collapsed those 15 into 10, and left
frames pointing at an index that no longer existed - so the reviewer's quick
save could not be reopened and the full save died with
``IndexError: Frame index 49850 out of range`` after a full review session.
Round 1 never hit this because it attached frames to mp4-backed videos, one per
(experiment, view). This does the same. ``--verify-save`` reproduces the
annotator's own save path end to end before anything is handed over.

**Batching.** Written in batches of ``--batch-size`` moments (default 150,
matching round 1's 139) rather than one 7 GB package. One batch is one review
session, and a failure part-way through costs one batch rather than everything.

Run::

    .venv\\Scripts\\python.exe tools/s05_build_round2_seed.py --dry-run
    .venv\\Scripts\\python.exe tools/s05_build_round2_seed.py
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import sleap_io as sio
from scipy.optimize import linear_sum_assignment

from medaka3d import load_catalog, load_paths
from medaka3d.provenance import provenance

VIEWS = ("top_cam", "side1_cam", "side2_cam")
NODE_ORDER = ("mouth", "pectoral", "tail")
EDGES = (("mouth", "pectoral"), ("pectoral", "tail"))
SKELETON_NAME = "medaka_3node_v1"
T_BY_VIEW = {"top_cam": 0.282, "side1_cam": 0.300, "side2_cam": 0.300}

ANNOTATION = Path(
    r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed"
    r"\pectoral-seed-20260804T103622JST-3d5f46b7c830\edits\pectoral_edits_full.pkg.slp"
)
TOP_MODEL = Path(
    r"Z:\StephensU\medaka\work\s03_bottomup_3node\top"
    r"\s03.bottomup_3node.round8_f24_ms64_annotated-only_n124"
)
SOURCES = ("labels.round8.top.train", "labels.round8.side.train")
DEFAULT_OUT = Path(r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed")

# Accepting the model's pectoral takes three tests, not a distance threshold.
#
# The first version used a flat 15 px cap on how far the model's own mouth/tail
# could sit from the human's. It threw away good predictions for the sake of
# 0.1 px: on one frame the model found all three fish with all three nodes, and
# matches at 15.1 and 15.8 px were rejected and replaced by interpolation. That
# is what made two of three pectorals sit exactly on the mouth-tail line.
#
# Measured on batch 1 (394 top matches), the flat rule rejected 29, and 41% of
# those had the runner-up more than 30 px away - unambiguous matches discarded
# for being slightly imprecise. Meanwhile the 15-20 px band had a median margin
# of 1.1 px: two fish swimming abreast, genuinely indistinguishable, and right
# to reject - but for ambiguity, not for distance.
#
# So: bound the axis error by a fraction of the animal's own length rather than
# an absolute; require the pairing to be unambiguous; and require the pectoral
# to belong to the fish it was matched to.
# RELAXED 2026-08-06, at the experimenter's direction, after they found a frame
# where two fish swimming abreast both fell back to interpolation. The tight
# rule was optimising for the wrong thing: every seeded point is reviewed by
# hand, and a pectoral sitting on the *wrong fish* is obvious at a glance, while
# an interpolated point that is subtly wrong is not. Preferring interpolation
# whenever the pairing was less than certain therefore hid errors instead of
# preventing them.
#
# What remains is the one test whose failure is genuinely harmful: the pectoral
# must belong to the animal it was attached to. On batch 1 that fires 3 times in
# 450; the ambiguity and axis-error tests fired 28 more and are now gone.
MAX_AXIS_ERROR_FRACTION = 0.5    # of the human mouth-tail span; a sanity cap only
MIN_AXIS_ERROR_PX = 30.0         # floor, for short spans


def segment_distance(point, a, b) -> float:
    """Distance from ``point`` to the segment ab."""
    ab = b - a
    length2 = float(ab @ ab)
    if length2 < 1e-9:
        return float(np.linalg.norm(point - a))
    t = float(np.clip((point - a) @ ab / length2, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


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
    """(experiment, view, frame_idx). Never frame_idx alone: side1 and side2 are
    the same instant from two cameras and share every frame index."""
    path = leaf_video(frame.video)
    return experiment_of(path), view_of(path), int(frame.frame_idx)


def node_xy(instance, name: str):
    for point in instance.points:
        if str(point["name"]) == name:
            xy = np.asarray(point["xy"], dtype=float)
            return xy if np.isfinite(xy).all() else None
    return None


def build_instance(nodes: dict, skeleton):
    coordinates, visibility = [], []
    for name in NODE_ORDER:
        xy, visible = nodes.get(name, (None, False))
        if xy is None or not np.isfinite(xy).all():
            xy, visible = np.array([np.nan, np.nan]), False
        coordinates.append(np.asarray(xy, dtype=float))
        visibility.append(bool(visible))
    instance = sio.Instance.from_numpy(np.stack(coordinates), skeleton=skeleton)
    for index in range(len(NODE_ORDER)):
        instance.points["xy"][index] = coordinates[index]
        instance.points["visible"][index] = visibility[index]
    return instance


def predict_top(frames, device: str, batch: int) -> dict:
    import sleap_nn

    videos = list({id(f.video): f.video for f in frames}.values())
    blank = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=f.video, frame_idx=int(f.frame_idx), instances=[])
            for f in frames
        ],
        videos=videos,
        skeletons=[sio.Skeleton(list(NODE_ORDER))],
    )
    predicted = sleap_nn.predict(
        blank, model_paths=[str(TOP_MODEL)], device=device, batch_size=batch
    )
    out: dict[tuple, list] = {}
    for frame in predicted.labeled_frames:
        out.setdefault(frame_key(frame), []).extend(frame.instances)
    return out


def seed_frame(frame, predictions, skeleton, stats: collections.Counter):
    """Human mouth/tail + a pectoral from the model (top) or interpolation."""
    experiment, view, frame_idx = frame_key(frame)
    t = T_BY_VIEW.get(view, 0.297)
    human = []
    for instance in frame.instances:
        human.append((node_xy(instance, "mouth"), node_xy(instance, "tail")))

    matched: dict[int, object] = {}
    if predictions is not None:
        pred = predictions.get((experiment, view, frame_idx), [])
        pred_nodes = [
            (node_xy(p, "mouth"), node_xy(p, "tail"), node_xy(p, "pectoral")) for p in pred
        ]
        cost = np.full((len(human), len(pred_nodes)), 1e6)
        for i, (hm, ht) in enumerate(human):
            for j, (pm, pt, _) in enumerate(pred_nodes):
                parts = [np.linalg.norm(hm - pm) if (hm is not None and pm is not None) else None,
                         np.linalg.norm(ht - pt) if (ht is not None and pt is not None) else None]
                usable = [p for p in parts if p is not None]
                if usable:
                    cost[i, j] = float(np.mean(usable))
        if len(pred_nodes):
            rows_i, cols_i = linear_sum_assignment(cost)
            for i, j in zip(rows_i, cols_i, strict=False):
                if cost[i, j] >= 1e6:
                    continue
                mouth, tail = human[i]
                span = (float(np.linalg.norm(tail - mouth))
                        if (mouth is not None and tail is not None) else 0.0)
                limit = max(MIN_AXIS_ERROR_PX, MAX_AXIS_ERROR_FRACTION * span)
                if cost[i, j] > limit:
                    stats[f"{view}:reject_axis_error"] += 1
                    continue
                pectoral = pred_nodes[j][2]
                if pectoral is not None and mouth is not None and tail is not None:
                    own = segment_distance(pectoral, mouth, tail)
                    rival = [segment_distance(pectoral, m2, t2)
                             for k, (m2, t2) in enumerate(human)
                             if k != i and m2 is not None and t2 is not None]
                    if rival and min(rival) < own:
                        stats[f"{view}:reject_other_animal"] += 1
                        continue
                matched[i] = pectoral

    instances, ledger = [], []
    for i, (mouth, tail) in enumerate(human):
        pectoral, source = None, "none"
        model_pec = matched.get(i)
        if model_pec is not None:
            pectoral, source = model_pec, "model"
        elif mouth is not None and tail is not None:
            pectoral, source = mouth + t * (tail - mouth), "interpolated"
        stats[f"{view}:{source}"] += 1
        nodes = {
            "mouth": (mouth, mouth is not None),
            "tail": (tail, tail is not None),
            "pectoral": (pectoral, pectoral is not None),
        }
        instances.append(build_instance(nodes, skeleton))
        ledger.append({
            "experiment": experiment, "view": view, "frame_idx": frame_idx,
            "instance": i, "pectoral_source": source,
            "pectoral_x": "" if pectoral is None else f"{pectoral[0]:.4f}",
            "pectoral_y": "" if pectoral is None else f"{pectoral[1]:.4f}",
        })
    return instances, ledger


def verify_annotator_save(path: Path) -> dict:
    """Reproduce the annotator's own save, then reload the result.

    This is not a generic round-trip. `_do_save` in the annotator rebuilds the
    Labels object and resolves every frame's video through a dict keyed by the
    **mp4 source path** -- so two embedded videos sharing one mp4 collapse into
    one and the frames of the loser point at nothing. That is the exact failure
    that cost a review session, and a plain `save_slp`/`load_slp` does not
    exercise it.
    """
    labels = sio.load_slp(str(path))

    # The annotator's dedupe, verbatim in behaviour.
    video_by_src: dict[str, object] = {}
    for video in labels.videos:
        src = leaf_video(video) or str(video.filename)
        video_by_src.setdefault(src, video)
    resolved = [video_by_src[leaf_video(f.video) or str(f.video.filename)]
                for f in labels.labeled_frames]
    collapsed = len(labels.videos) - len(video_by_src)

    rebuilt = sio.Labels(
        skeletons=labels.skeletons,
        videos=list({id(v): v for v in resolved}.values()),
        labeled_frames=[
            sio.LabeledFrame(video=v, frame_idx=int(f.frame_idx), instances=list(f.instances))
            for f, v in zip(labels.labeled_frames, resolved, strict=False)
        ],
    )
    out = path.parent / f".{path.name}.savecheck.pkg.slp"
    result = {"videos_in_seed": len(labels.videos),
              "videos_after_dedupe": len(video_by_src),
              "collapsed": collapsed}
    try:
        sio.save_slp(rebuilt, str(out), embed="all")
        back = sio.load_slp(str(out))
        first = back.labeled_frames[0]
        image = first.video[int(first.frame_idx)]
        result.update({"ok": True, "frames": len(back.labeled_frames),
                       "videos": len(back.videos),
                       "image_shape": list(np.shape(image))})
        # Release the HDF5 handles, or Windows refuses to delete the file.
        for video in back.videos:
            with contextlib.suppress(Exception):
                video.close()
    except Exception as exc:
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        with contextlib.suppress(OSError):
            out.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--batch-size", type=int, default=150,
                        help="Moments per package; 0 writes a single package.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--infer-batch", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply-edits", type=Path, default=None,
                        help="Recovered review (.npy) to fold back into the seed.")
    parser.add_argument("--apply-views", default="top_cam",
                        help="Comma-separated views the review is applied to.")
    parser.add_argument("--verify-save", action="store_true",
                        help="Reproduce the annotator's own save on batch 1 and "
                             "reload the result. Slow, and worth it.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    if not (TOP_MODEL / "best.ckpt").is_file():
        raise SystemExit(f"top model not found: {TOP_MODEL}")

    annotated = {frame_key(f) for f in sio.load_slp(str(ANNOTATION)).labeled_frames}
    by_moment: dict[tuple, dict] = collections.defaultdict(dict)
    for key in SOURCES:
        labels = sio.load_slp(str(catalog.get(key)))
        for frame in labels.labeled_frames:
            experiment, view, frame_idx = frame_key(frame)
            if (experiment, view, frame_idx) in annotated:
                continue
            by_moment[(experiment, frame_idx)][view] = frame

    moments = sorted(k for k, v in by_moment.items() if len(v) == len(VIEWS))
    partial = len(by_moment) - len(moments)
    if args.limit:
        moments = moments[:args.limit]
    print(f"round-1 annotated : {len(annotated)} view-frames")
    print(f"remaining moments : {len(moments)} with all 3 views"
          f"{f' ({partial} incomplete, skipped)' if partial else ''}")
    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    edits = None
    apply_views = {v.strip() for v in args.apply_views.split(",") if v.strip()}
    if args.apply_edits:
        edits = dict(np.load(args.apply_edits, allow_pickle=True).tolist())
        print(f"review to fold in: {len(edits)} view-frames, applied to "
              f"{sorted(apply_views)}")

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"pectoral-round2-{stamp}JST"
    out_dir = args.out_root / run_id
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    with provenance("build_round2_seed", out_dir, name="round2_seed") as record:
        record.params_from(args)
        record.param("top_pectoral_source", str(TOP_MODEL))
        record.param("side_pectoral_source", f"interpolation t={T_BY_VIEW['side1_cam']}")
        record.param("accept_rule", {
            "max_axis_error_fraction_of_span": MAX_AXIS_ERROR_FRACTION,
            "min_axis_error_px": MIN_AXIS_ERROR_PX,
            "pectoral_must_be_nearest_own_axis": True,
        })
        record.param("moments", len(moments))
        record.input("round1_annotation", ANNOTATION)
        for key in SOURCES:
            record.input(key, catalog.get(key))

        top_frames = [by_moment[m]["top_cam"] for m in moments]
        print(f"\nrunning the top model over {len(top_frames)} frames ...")
        predictions = predict_top(top_frames, args.device, args.infer_batch)

        skeleton = sio.Skeleton(nodes=list(NODE_ORDER), edges=[list(e) for e in EDGES],
                                name=SKELETON_NAME)
        stats: collections.Counter[str] = collections.Counter()
        ledger: list[dict] = []
        batch = args.batch_size or len(moments)
        written = []

        mp4_cache: dict[tuple, object] = {}

        def mp4_video(experiment: str, view: str):
            """One Video per (experiment, view), read from the original mp4.

            Round-8's embedded videos cannot be used: several of them share one
            mp4, and the annotator's save collapses videos by that path.
            """
            key = (experiment, view)
            if key not in mp4_cache:
                mp4_cache[key] = sio.load_video(str(paths.video(experiment, view)))
            return mp4_cache[key]

        for start in range(0, len(moments), batch):
            chunk = moments[start:start + batch]
            frames, videos = [], {}
            for moment in chunk:
                for view in VIEWS:
                    source = by_moment[moment][view]
                    instances, rows = seed_frame(
                        source, predictions if view == "top_cam" else None, skeleton, stats
                    )
                    ledger.extend(rows)
                    if edits is not None:
                        key = (moment[0], view, int(source.frame_idx))
                        replacement = edits.get(key)
                        if replacement is not None and view in apply_views:
                            instances = [
                                build_instance({
                                    NODE_ORDER[k]: (np.array(pt[:2]), bool(pt[2]))
                                    for k, pt in enumerate(inst[:3])
                                }, skeleton) for inst in replacement
                            ]
                            stats[f"{view}:from_review"] += len(instances)
                    video = mp4_video(moment[0], view)
                    videos.setdefault(id(video), video)
                    frames.append(sio.LabeledFrame(
                        video=video, frame_idx=int(source.frame_idx),
                        instances=instances))
            labels = sio.Labels(skeletons=[skeleton], videos=list(videos.values()),
                                labeled_frames=frames)
            index = start // batch + 1
            path = final_dir / f"pectoral_round2_batch{index:02d}_n{len(chunk)}.pkg.slp"
            print(f"  batch {index}: {len(chunk)} moments, {len(frames)} view-frames "
                  f"-> {path.name}")
            sio.save_slp(labels, str(path), embed=True)
            written.append({"batch": index, "moments": len(chunk),
                            "view_frames": len(frames), "path": str(path),
                            "size_mb": round(path.stat().st_size / 1e6, 1)})
            record.output(f"batch{index:02d}", path)

        if args.verify_save and written:
            first = Path(written[0]["path"])
            print(f"\nverifying the annotator's save path on {first.name} ...")
            check = verify_annotator_save(first)
            record.param("annotator_save_check", check)
            print(json.dumps(check, indent=2))
            if not check.get("ok"):
                raise SystemExit(
                    "the seed cannot survive the annotator's own full save: "
                    f"{check.get('error')}\nThis is the failure that cost a "
                    "review session; refusing to hand over another one."
                )

        ledger_path = final_dir / "pectoral_round2_ledger.csv"
        with ledger_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
            writer.writeheader()
            writer.writerows(ledger)
        record.output("ledger", ledger_path)
        record.param("pectoral_sources", dict(stats))
        record.param("batches", written)

        print(f"\npectoral sources: {json.dumps(dict(stats), indent=2)}")

    print(f"\nwrote {out_dir}")
    for entry in written:
        print(f"  {entry['path']}   {entry['size_mb']} MB")


if __name__ == "__main__":
    main()
