"""S12 — one three-view package covering every round-8 moment, for review.

The 3-node training set built by `s02 --pectoral-mode model` is split by view
(a top package and a side package) because that is how the models train. It
cannot be opened in the three-pane annotator: a moment's three cameras live in
two different files. This assembles the same labels into the shape the GUI
needs.

=============================================================================
WHAT THIS IS
=============================================================================

**1,382 moments x 3 views = 4,146 view-frames, in a single package.** Requested
explicitly: no batching. The cost is stated rather than designed around — round
2 was split into nine ~150-moment batches because one batch is one review
session and a failure part-way through costs one batch instead of everything.
At ~7 GB this is one file, and a failed full save loses the session. `Ctrl+S`
quick-saves and the annotator resumes from it, so save often.

**9 videos, one mp4 per (experiment, view).** Not negotiable, and the reason is
scarred in: round 1's first build reused round-8's own `Video` objects and got
26 of them for the same recordings; the annotator's save resolves frames
through a dict keyed by mp4 source path, collapsed them, and left frames
pointing at an index that no longer existed — a quick save that could not be
reloaded, which is also the auto-resume path. Round 8 partitions these three
recordings into 9 embedded videos for top and 18 for side; both counts are
artefacts of its own storage and carry no meaning for annotation. Here every
frame hangs off the original mp4 for its `(experiment, view)`, so the dedupe
has nothing to collapse. `verify_annotator_save` reproduces that exact save
path before this declares success.

**The labels are read back from the built training packages, not recomputed.**
What gets reviewed is byte-for-byte what was written, so a correction here
lands on the same point the model will train on. Re-deriving them would risk
reviewing one thing and training another.

**Every frame is re-decoded from the mp4 and required to be byte-identical to
what round 8 embedded.** Changing the image source must not quietly change an
image.

**n_groups = 1.** Round-8 train is `20251225_K_1_{1,2,3}` — one trio of fish on
three consecutive days. Three experiments, not three replicates.

Run::

    .venv\\Scripts\\python.exe tools/s12_build_3node_review.py --dry-run
    .venv\\Scripts\\python.exe tools/s12_build_3node_review.py
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s02_merge_pectoral import DEFAULT_OUT as MERGED_ROOT
from s02_merge_pectoral import (
    EDGES,
    NODE_ORDER,
    SKELETON_NAME,
    build_instance,
    frame_key,
    node_map,
)
from s05_build_round2_seed import verify_annotator_save

from medaka3d import load_catalog, load_paths
from medaka3d.provenance import provenance

DEFAULT_OUT = Path(r"Z:\StephensU\medaka\annotations\sleap_pectoral_seed")
SOURCES = ("labels.round8.top.train", "labels.round8.side.train")
#: The three panes the annotator shows, in pane order.
NODE_VIEWS = ("top_cam", "side1_cam", "side2_cam")


def load_built_labels() -> tuple[dict[tuple, list[dict]], dict]:
    """Every 3-node instance from the built training packages, by frame key.

    Reads both views and both splits, so the review package covers the same
    1,382 moments the dataset does — including the 30 held out for validation,
    which are human-annotated and are exactly the ones worth spot-checking.
    """
    out: dict[tuple, list[dict]] = {}
    counts: collections.Counter = collections.Counter()
    for view in ("top", "side"):
        for split in ("train", "val"):
            path = MERGED_ROOT / view / "model" / f"{split}_labels.3node_full.pkg.slp"
            if not path.is_file():
                raise SystemExit(
                    f"missing {path}\nBuild it first:\n"
                    f"  .venv\\Scripts\\python.exe tools/s02_merge_pectoral.py "
                    f"--view {view} --pectoral-mode model --val-moments 30"
                )
            labels = sio.load_slp(str(path))
            for frame in labels.labeled_frames:
                key = frame_key(frame)
                if key in out:
                    raise SystemExit(f"{key} appears in more than one built package")
                out[key] = [node_map(i) for i in frame.instances]
                counts[f"{view}_{split}"] += 1
            for video in labels.videos:
                with contextlib.suppress(Exception):
                    video.close()
    return out, dict(counts)


def source_frames(catalog) -> dict[tuple, object]:
    """Round-8's own labelled frames, keyed the same way. The image reference."""
    out: dict[tuple, object] = {}
    for key_name in SOURCES:
        labels = sio.load_slp(str(catalog.get(key_name)))
        for frame in labels.labeled_frames:
            out[frame_key(frame)] = frame
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Build only the first N moments. For a smoke run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be built, write nothing.")
    parser.add_argument("--skip-image-check", action="store_true",
                        help="Skip the byte-identical re-decode. Only for a smoke run; "
                             "the check is the guard that the image source did not "
                             "change what the reviewer sees.")
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)

    built, per_package = load_built_labels()
    source = source_frames(catalog)
    print(f"built labels: {len(built)} view-frames  {per_package}")
    print(f"round-8 source: {len(source)} view-frames")

    missing = sorted(set(built) - set(source))
    if missing:
        raise SystemExit(f"{len(missing)} built frames are not in round 8, e.g. {missing[:3]}")

    moments = sorted({(key[0], key[2]) for key in built})
    if args.limit:
        moments = moments[: args.limit]
    experiments = sorted({m[0] for m in moments})
    print(f"moments: {len(moments)} over {len(experiments)} experiments {experiments}")

    # One video per (experiment, view): 3 x 3 = 9. Round 8's 9 top and 18 side
    # embedded videos are its own partitioning of these same three recordings.
    videos: dict[tuple[str, str], object] = {}
    for experiment in experiments:
        for view in NODE_VIEWS:
            videos[(experiment, view)] = sio.Video.from_filename(
                str(paths.video(experiment, view))
            )
    print(f"videos: {len(videos)} (one mp4 per experiment-view)")

    complete = [
        m for m in moments if all((m[0], v, m[1]) in built for v in NODE_VIEWS)
    ]
    if len(complete) != len(moments):
        print(f"  {len(moments) - len(complete)} moments lack all three views and are dropped")
    moments = complete

    if args.dry_run:
        print(f"\nDRY RUN — would write {len(moments)} moments x 3 = "
              f"{len(moments) * 3} view-frames, {len(videos)} videos")
        return

    stamp = datetime.now().strftime("%Y%m%dT%H%M%SJST")
    out_dir = args.out_root / f"pectoral-review-all-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    package = out_dir / f"pectoral_3node_all_n{len(moments)}.pkg.slp"

    with provenance("build_3node_review", out_dir, name="review_all") as record:
        record.params_from(args)
        record.param("moments", len(moments))
        record.param("experiments", experiments)
        record.param("n_groups", len({e.split("_")[1] for e in experiments}))
        record.param("videos", len(videos))
        record.param("built_packages", per_package)
        for key_name in SOURCES:
            record.input(key_name, catalog.get(key_name))

        skeleton = sio.Skeleton(
            nodes=list(NODE_ORDER), edges=[list(e) for e in EDGES], name=SKELETON_NAME
        )
        frames: list[object] = []
        checked = mismatched = 0
        ledger: list[dict] = []

        for position, (experiment, frame_idx) in enumerate(moments):
            for view in NODE_VIEWS:
                key = (experiment, view, frame_idx)
                nodes = built[key]

                if not args.skip_image_check:
                    want = source[key].video[int(frame_idx)]
                    got = videos[(experiment, view)][int(frame_idx)]
                    checked += 1
                    if want is None or got is None or not np.array_equal(want, got):
                        mismatched += 1
                        raise SystemExit(
                            f"image mismatch at {key}: the mp4 does not reproduce the "
                            f"frame round 8 embedded. Do not review against a different "
                            f"image than the model trains on."
                        )

                frames.append(
                    sio.LabeledFrame(
                        video=videos[(experiment, view)],
                        frame_idx=int(frame_idx),
                        instances=[build_instance(n, skeleton) for n in nodes],
                    )
                )
                for index, node_set in enumerate(nodes):
                    pectoral = node_set.get("pectoral", (None, False))[0]
                    ledger.append({
                        "order": position, "experiment": experiment, "view": view,
                        "frame_idx": frame_idx, "instance": index,
                        "pectoral_x": "" if pectoral is None or not np.isfinite(pectoral).all()
                                      else round(float(pectoral[0]), 4),
                        "pectoral_y": "" if pectoral is None or not np.isfinite(pectoral).all()
                                      else round(float(pectoral[1]), 4),
                    })
            if position and position % 200 == 0:
                print(f"  {position}/{len(moments)} moments")

        print(f"image check: {checked} frames re-decoded, {mismatched} mismatched")
        record.param("frames_image_checked", checked)

        labels = sio.Labels(
            skeletons=[skeleton],
            videos=list(videos.values()),
            labeled_frames=frames,
        )
        print(f"writing {len(frames)} view-frames to {package.name} ...")
        sio.save_slp(labels, str(package), embed=True)
        size_gb = package.stat().st_size / 2**30
        print(f"  {size_gb:.2f} GB")
        record.output("package", package)
        record.param("size_gb", round(size_gb, 2))

        ledger_path = out_dir / "pectoral_review_ledger.csv"
        with ledger_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
            writer.writeheader()
            writer.writerows(ledger)
        record.output("ledger", ledger_path)

        # The operation that actually failed once: the annotator's own save.
        check = verify_annotator_save(package)
        record.param("annotator_save_check", check)
        print(f"annotator save check: {json.dumps(check)}")
        if not check.get("ok"):
            raise SystemExit("the annotator's save path fails on this package; not shipping it")
        if check["collapsed"]:
            raise SystemExit(
                f"{check['collapsed']} videos collapse on dedupe — the round-1 failure. "
                f"Every frame must hang off its own mp4."
            )

    print(f"\nwrote {package}")





if __name__ == "__main__":
    main()
