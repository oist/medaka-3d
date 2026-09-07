"""S09 — build a corrected locked benchmark (v2) and record exactly what changed.

The locked HITL-v2 benchmark carries at least one ground-truth error. At frame
19323 the side2 view joined a point in open water to another animal's tail —
a 422.7 px "body" against a ~94 px median. The top view had two labels
overlapping on one animal, one of them spanning 43.5 px against a ~112 px
median.

A correction that was claimed and is NOT true, recorded so it is not repeated:
those two side2 tails were also called a shared node on sight. They are 36.3 px
apart, and the shared-node threshold is 5 px. s06's premise — that human labels
do not produce shared nodes — survives this frame intact. The lesson is the
ordinary one: a threshold metric is not eyeballed, it is computed.

=============================================================================
HOW THE CORRECTION IS HANDLED
=============================================================================

**The locked original is never modified.** Every number already reported was
measured against it, and silently editing it would retroactively invalidate all
of them with no way to tell which run saw which bytes. The original's sha256 is
recorded here and checked; v2 is a new file beside it.

**v2 = original, with the corrected view-frames substituted.** Only frames the
correction actually touched are replaced; the other 117 view-frames are carried
through unchanged and that is verified, not assumed.

**Instance order is not meaningful and is not preserved.** The correction was
made with the annotator's calibration re-match (`F`), which re-derives the
cross-view correspondence and therefore re-orders instances within a view. s06
matches by Hungarian assignment, so order never enters any score.

**Both files stay usable.** s06 takes `--benchmark`, so the 2x2 can be run
against the original and against v2 and the two answers compared. If the
conclusion moves, the movement is the finding.

Run::

    .venv\\Scripts\\python.exe tools/s09_benchmark_v2.py \\
        --edits C:/Users/machi/medaka_benchmark_fix/benchmark_f19323_edits.pkg.slp
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sleap_io as sio
from s06_locked_benchmark_3node import LOCKED, frame_key

from medaka3d.provenance import provenance

OUT_ROOT = Path(
    r"Z:\StephensU\medaka\benchmarks\sleap_hitl_v2\correction-20260809-f19323"
)
ORIGINAL_SHA256_PREFIX = "58b2788c9fa7b1dd"


def nodes(instance) -> dict:
    out = {}
    for point in instance.points:
        xy = np.asarray(point["xy"], dtype=float)
        out[str(point["name"])] = (
            xy if np.isfinite(xy).all() else None, bool(point["visible"])
        )
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(frame) -> list[dict]:
    rows = []
    for index, instance in enumerate(frame.instances):
        n = nodes(instance)
        mouth, tail = n.get("mouth", (None, False)), n.get("tail", (None, False))
        span = (float(np.linalg.norm(tail[0] - mouth[0]))
                if mouth[0] is not None and tail[0] is not None else None)
        rows.append({
            "instance": index,
            "mouth": None if mouth[0] is None else [round(float(v), 2) for v in mouth[0]],
            "mouth_visible": mouth[1],
            "tail": None if tail[0] is None else [round(float(v), 2) for v in tail[0]],
            "tail_visible": tail[1],
            "span_px": None if span is None else round(span, 1),
        })
    return rows


def shared_nodes(frame, threshold: float = 5.0) -> list[str]:
    n = [nodes(i) for i in frame.instances]
    hits = []
    for name in ("mouth", "tail"):
        pts = [(i, d[name][0]) for i, d in enumerate(n) if d.get(name, (None,))[0] is not None]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                d = float(np.linalg.norm(pts[a][1] - pts[b][1]))
                if d <= threshold:
                    hits.append(f"{name}: inst{pts[a][0]}/inst{pts[b][0]} at {d:.2f} px")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    if not args.edits.is_file():
        raise SystemExit(f"edits not found: {args.edits}")

    original_sha = sha256(LOCKED)
    if not original_sha.startswith(ORIGINAL_SHA256_PREFIX):
        raise SystemExit(
            f"the locked benchmark is not the file this tool was written against.\n"
            f"  expected sha256 starting {ORIGINAL_SHA256_PREFIX}\n"
            f"  found              {original_sha[:16]}\n"
            f"Someone has modified the locked original. Stop and find out who."
        )

    final_dir = args.out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_dir / "shared_benchmark_40_hitl_v2_locked_v2_f19323fix.pkg.slp"

    with provenance("benchmark_v2", args.out_root, name="benchmark_v2") as record:
        record.params_from(args)
        record.input("locked_original", LOCKED, sha256=original_sha)
        record.input("correction", args.edits)
        record.param("original_sha256", original_sha)
        record.param("original_unmodified", True)

        original = sio.load_slp(str(LOCKED))
        edits = {frame_key(f): f for f in sio.load_slp(str(args.edits)).labeled_frames}

        changed, carried = [], 0
        frames = []
        for frame in original.labeled_frames:
            key = frame_key(frame)
            replacement = edits.get(key)
            if replacement is None:
                frames.append(frame)
                carried += 1
                continue
            before, after = describe(frame), describe(replacement)
            if before == after:
                frames.append(frame)
                carried += 1
                continue
            changed.append({
                "experiment": key[0], "view": key[1], "frame_idx": key[2],
                "before": before, "after": after,
                "shared_nodes_before": shared_nodes(frame),
                "shared_nodes_after": shared_nodes(replacement),
            })
            frames.append(sio.LabeledFrame(
                video=frame.video, frame_idx=int(frame.frame_idx),
                instances=list(replacement.instances)))

        print(f"view-frames carried through unchanged : {carried}")
        print(f"view-frames replaced                  : {len(changed)}")
        touched = sorted({c["frame_idx"] for c in changed})
        print(f"source frames touched                 : {touched}")
        record.param("carried_unchanged", carried)
        record.param("replaced", len(changed))
        record.param("frames_touched", touched)
        record.param("changes", changed)

        before_shared = sum(len(c["shared_nodes_before"]) for c in changed)
        after_shared = sum(len(c["shared_nodes_after"]) for c in changed)
        print(f"\nGT shared-node pairs in the touched frames: "
              f"{before_shared} -> {after_shared}")
        for c in changed:
            for line in c["shared_nodes_before"]:
                print(f"  BEFORE  {c['view']} f{c['frame_idx']}  {line}")
            for line in c["shared_nodes_after"]:
                print(f"  AFTER   {c['view']} f{c['frame_idx']}  {line}")
        record.param("gt_shared_before", before_shared)
        record.param("gt_shared_after", after_shared)

        corrected = sio.Labels(
            skeletons=original.skeletons,
            videos=list({id(f.video): f.video for f in frames}.values()),
            labeled_frames=frames,
        )
        sio.save_slp(corrected, str(out_path), embed="all")

        back = sio.load_slp(str(out_path))
        checks = {
            "view_frames": len(back.labeled_frames),
            "videos": len(back.videos),
            "instances": sum(len(f.instances) for f in back.labeled_frames),
            "nodes": [n.name for n in back.skeleton.nodes],
        }
        print(f"\nv2 written and reloaded: {json.dumps(checks)}")
        record.param("verification", checks)
        record.output("benchmark_v2", out_path, sha256=sha256(out_path))

        if sha256(LOCKED) != original_sha:
            raise SystemExit("the locked original changed during this run")
        print(f"locked original still {original_sha[:16]} - unmodified")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
