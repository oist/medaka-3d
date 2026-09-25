"""Render sealed semantic-node tracks on the same raw frames for visual review.

Colors mean conditional initial body slots, not verified biological identities.
No edits to predictions, exclusions or manual labels are produced.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import track_semantic_node200 as tracking

base = tracking.base
FIRST, STOP, VIEWS = tracking.FIRST, tracking.STOP, tracking.VIEWS
COLORS = ((0, 170, 255), (255, 220, 40), (210, 80, 245))
NODES = ("M", "P", "T")
STAGE = "semantic_node200_visual_review_v1"


def text(image, value, xy, color=(235, 235, 235), size=0.52):
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def panel(image, title, points=None, flags=None, query=None):
    canvas = np.zeros((676, 600, 3), np.uint8)
    canvas[46:646] = cv2.resize(image, (600, 600), interpolation=cv2.INTER_AREA)
    text(canvas, title, (12, 26), size=0.62)
    if points is None:
        text(canvas, "Unmarked raw image", (12, 666))
        return canvas
    valid = flags & np.isfinite(points).all(axis=1) & ((points >= 0) & (points < 1200)).all(axis=1)
    for body in range(3):
        indices = np.flatnonzero(query["body_slot"] == body)
        for left, right in pairwise(indices):
            if valid[left] and valid[right]:
                ends = np.rint(points[[left, right]] * 0.5).astype(np.int32)
                ends[:, 1] += 46
                cv2.line(canvas, tuple(ends[0]), tuple(ends[1]), COLORS[body], 1, cv2.LINE_AA)
    pixel = np.rint(points[valid] * 0.5).astype(np.int32)
    selected = np.flatnonzero(valid)
    for index, p in zip(selected, pixel, strict=True):
        body, node = int(query["body_slot"][index]), int(query["node_index"][index])
        x, y = int(p[0]), int(p[1]) + 46
        cv2.circle(canvas, (x, y), 3, COLORS[body], 1, cv2.LINE_AA)
        text(canvas, f"{body}{NODES[node]}", (min(x + 4, 567), max(y - 4, 57)), COLORS[body], 0.4)
    text(canvas, f"Reported {int(flags.sum())}/9; finite inside {int(valid.sum())}/9", (12, 666))
    return canvas


def run(args):
    root, destination = args.tracking_root.resolve(), args.output.resolve()
    base.require(root.is_relative_to(base.WORK), "workspace producer required")
    base.require(
        destination.is_relative_to(base.WORK) and not destination.exists(), "new output required"
    )
    evidence, output = base.Evidence(), base.Output(destination)
    started, completed = base.utc(), []
    try:
        record, members = base.verified(
            root,
            "tracking.provenance.json",
            tracking.STAGE,
            args.tracking_sha256,
            evidence,
        )
        plan = base.bound_json(root, members, "plan.json")
        _, pmembers = base.verified(
            tracking.PREPARATION,
            "preparation.provenance.json",
            tracking.PREPARATION_STAGE,
            plan["preparation_sha256"],
            evidence,
        )
        base.require(
            record["inputs"][str(tracking.PREPARATION / "preparation.provenance.json")]
            == plan["preparation_sha256"],
            "tracking/preparation binding differs",
        )
        native = base.bound_json(tracking.PREPARATION, pmembers, "native_evidence.json")
        tracking.validate_native_rows(native)
        indexed = {(r["view"], r["frame_idx"]): r for r in native}
        for module in (sys.modules[__name__], tracking, base):
            path = Path(module.__file__).resolve()
            evidence.add(path)
            output.write(
                "code_snapshot/" + path.relative_to(base.REPO).as_posix(), path.read_bytes()
            )
        for view in VIEWS:
            arrays = {}
            for kind in ("queries", "lk", "emissions"):
                path = tracking.bound_path(root, members, f"{view}/{kind}.npz")
                arrays[kind] = tracking.arrays_from_bytes(tracking.checked_bytes(path, members))
            query, lk, ct = arrays["queries"], arrays["lk"], arrays["emissions"]
            tracking.validate_queries(query, dict(view=view), view)
            base.require(
                ct["frame_idx"].tolist() == list(range(FIRST, STOP)), "native tracks differ"
            )
            name = f"{view}.mp4"
            path = destination / name
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1800, 676))
            base.require(writer.isOpened(), "video writer unavailable")
            try:
                for t, frame in enumerate(range(FIRST, STOP)):
                    row = indexed[view, frame]
                    p = tracking.bound_path(tracking.PREPARATION, pmembers, row["png_path"])
                    bgr = cv2.imdecode(
                        np.frombuffer(tracking.checked_bytes(p, pmembers), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    base.require(
                        bgr is not None
                        and hashlib.sha256(bgr.tobytes()).hexdigest() == row["bgr_sha256"],
                        "native overlay pixels differ",
                    )
                    horizon = int(ct["first_release_native"][t])
                    canvas = np.hstack(
                        (
                            panel(bgr, f"{view} | native {frame}"),
                            panel(bgr, "Initial-node LK chain", lk["xy"][t], lk["valid"][t], query),
                            panel(
                                bgr,
                                f"CT first | release {horizon}",
                                ct["first_xy"][t],
                                ct["first_visibility"][t],
                                query,
                            ),
                        )
                    )
                    writer.write(canvas)
                    if t in (0, 40, 80, 120, 160, 199):
                        ok, encoded = cv2.imencode(".png", canvas)
                        base.require(ok, "PNG encoding failed")
                        output.write(f"{view}_{frame}.png", encoded.tobytes())
            finally:
                writer.release()
            cap, count = cv2.VideoCapture(str(path)), 0
            try:
                base.require(
                    cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS) - 20) < 1e-6,
                    "video FPS differs",
                )
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    base.require(image.shape == (676, 1800, 3), "video dimensions differ")
                    count += 1
            finally:
                cap.release()
            base.require(count == 200, "full200 video decode required")
            output.records[name] = dict(
                path=str(path), sha256=base.sha(path), size_bytes=path.stat().st_size
            )
            completed.append(dict(view=view, frames=count, fps=20, dimensions=[1800, 676]))
        output.write(
            "summary.json",
            dict(
                stage=STAGE,
                videos=completed,
                PNGs=18,
                original_rate_fps=166,
                native_interval=[FIRST, STOP],
                slowed_display_fps=20,
                colors="conditional initial body slots0/1/2, not verified biological IDs",
                nodes="M mouth, P pectoral, T tail",
                geometry_or_ID_changed=False,
                manual_review=False,
                truth_labels=False,
            ),
        )
        evidence.final_guard()
        output.final_guard()
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records,
        )
        with (destination / "review.provenance.json").open("xb") as f:
            f.write(base.json_bytes(status))
    except BaseException as error:
        with (destination / "review.provenance.json").open("xb") as f:
            f.write(
                base.json_bytes(
                    dict(
                        stage=STAGE,
                        status="failed",
                        started_at=started,
                        finished_at=base.utc(),
                        inputs=evidence.paths,
                        outputs=output.records,
                        completed_views=completed,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                )
            )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--tracking-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
