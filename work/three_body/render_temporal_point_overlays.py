"""Render the two frozen 64-frame comparisons without running a tracker.

Colors identify initial anonymous mask rows, never fish IDs. Coordinates are
rendered only when the saved method reports them and they are finite/in-image.
The videos run at 20 fps for inspection; source timing remains 166 fps.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
VIEWS = ("top_cam", "side1_cam", "side2_cam")
SPECS = (
    (
        "initial",
        66483,
        "temporal_point64_preparation_v1",
        "9771ee33cf36ba320cb2cf5b055c3c44b59f906d835ffac346fccb65241769f3",
        "temporal_point64_comparison_v2",
        "90b35bf8183367f432be4333f16816e58b34c526b387ea1799cce57683913e95",
    ),
    (
        "known_gap",
        333519,
        "temporal_known_gap64_preparation_v1",
        "447a713a715ccb900883bcebff42ab629fbd55f422b36a0e633ad6ed7f544c8e",
        "temporal_known_gap64_comparison_v1",
        "bd0bae4df3da0d8f2f4a31f0e754a93e89460c6b72803d2054327655662e732e",
    ),
)
# BGR colors, fixed before viewing any point positions.
COLORS = ((0, 220, 255), (255, 160, 0), (230, 80, 240), (80, 240, 80), (90, 100, 255))
CONTACT_FRAMES = (0, 16, 32, 48, 63)


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run():
    out = WORK / "temporal_point_overlay_review_v1"
    out.mkdir(exist_ok=False)
    inputs = {str(Path(__file__).resolve()): sha(Path(__file__))}
    outputs, summaries = {}, []
    started = datetime.now(UTC).isoformat()

    def record(p):
        outputs[p.name] = dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)

    def bind_seal(folder, name, expected):
        p = WORK / folder / name
        assert sha(p) == expected
        inputs[str(p)] = expected
        seal = json.loads(p.read_text(encoding="utf-8"))
        assert seal["status"] == "ok"
        return {str(Path(r["path"]).resolve()): r for r in seal["outputs"].values()}

    def bound(p, members):
        p = p.resolve()
        r = members[str(p)]
        assert p.stat().st_size == r.get("size_bytes", r.get("size"))
        assert sha(p) == r["sha256"]
        inputs[str(p)] = r["sha256"]
        return p

    def arrays(p, members):
        with np.load(bound(p, members), allow_pickle=False) as a:
            return {k: a[k] for k in a.files}

    try:
        for window, first, prep, ps, comp, cs in SPECS:
            pm = bind_seal(prep, "preparation.provenance.json", ps)
            cm = bind_seal(comp, "comparison.provenance.json", cs)
            for view in VIEWS:
                root = WORK / comp / view
                q = arrays(root / "queries.npz", cm)
                methods = {a: arrays(root / a / "metrics.npz", cm) for a in ("lk", "ct_first")}
                n = len(q["xy"])
                assert int(q["frame_idx"]) == first
                assert q["mask_index"].shape == (n,) and q["mask_index"].max() < len(COLORS)
                for a in methods.values():
                    assert a["xy"].shape == (64, n, 2) and a["reported"].shape == (64, n)
                video = out / f"{window}_{view}.mp4"
                writer = cv2.VideoWriter(
                    str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1800, 696)
                )
                assert writer.isOpened(), "MP4 writer unavailable"
                count_rows = []
                try:
                    for t in range(64):
                        native = first + t
                        p = bound(WORK / prep / "raw" / f"raw_{native}_{view}.png", pm)
                        raw = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        assert raw is not None and raw.shape == (1200, 1200, 3)
                        small = cv2.resize(raw, (600, 600), interpolation=cv2.INTER_AREA)
                        canvas = np.zeros((696, 1800, 3), np.uint8)
                        counts = dict(native_frame=native)
                        for col, label in enumerate(("raw", "lk", "ct_first")):
                            panel = small.copy()
                            title = "Raw image"
                            if label != "raw":
                                a = methods[label]
                                xy = a["xy"][t]
                                reported = a["reported"][t]
                                draw = reported & np.isfinite(xy).all(axis=1)
                                draw &= ((xy >= 0) & (xy < 1200)).all(axis=1)
                                only = (
                                    reported
                                    & ~methods["lk" if label == "ct_first" else "ct_first"][
                                        "reported"
                                    ][t]
                                )
                                counts[label] = dict(
                                    reported=int(reported.sum()),
                                    drawn=int(draw.sum()),
                                    only=int(only.sum()),
                                )
                                for i in np.flatnonzero(draw):
                                    px = tuple(np.rint(xy[i] / 2).astype(int))
                                    color = COLORS[int(q["mask_index"][i])]
                                    if only[i]:
                                        cv2.drawMarker(
                                            panel, px, color, cv2.MARKER_CROSS, 11, 2, cv2.LINE_AA
                                        )
                                    else:
                                        cv2.circle(panel, px, 3, color, -1, cv2.LINE_AA)
                                title = (
                                    "LK" if label == "lk" else "CT first emission"
                                ) + f"  reported {int(reported.sum())}/{n}"
                            canvas[48:648, col * 600 : (col + 1) * 600] = panel
                            cv2.putText(
                                canvas,
                                title,
                                (col * 600 + 10, 29),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.66,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                        for g in np.unique(q["mask_index"]):
                            cv2.putText(
                                canvas,
                                str(int(g)),
                                (330 + 48 * int(g), 29),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                COLORS[int(g)],
                                2,
                                cv2.LINE_AA,
                            )
                        foot = f"{window} | {view} | native {native} | +{t}/166 s | 20 fps playback | colors: initial mask rows (not fish IDs)"
                        cv2.putText(
                            canvas,
                            foot,
                            (8, 670),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            canvas,
                            "Cross: reported only by that method; dot: reported by both. CT may use later frames; no new inference.",
                            (8, 690),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (220, 220, 220),
                            1,
                            cv2.LINE_AA,
                        )
                        writer.write(canvas)
                        if t in CONTACT_FRAMES:
                            png = out / f"{window}_{view}_{native}.png"
                            assert cv2.imwrite(str(png), canvas)
                            assert np.array_equal(cv2.imread(str(png)), canvas)
                            record(png)
                        count_rows.append(counts)
                finally:
                    writer.release()
                capture = cv2.VideoCapture(str(video))
                decoded = 0
                try:
                    assert capture.isOpened()
                    assert capture.get(cv2.CAP_PROP_FPS) == 20
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        assert frame.shape == (696, 1800, 3)
                        decoded += 1
                finally:
                    capture.release()
                assert decoded == 64
                record(video)
                summaries.append(
                    dict(
                        window=window,
                        view=view,
                        source_interval=[first, first + 64],
                        query_count=n,
                        decoded_video_frames=decoded,
                        counts=count_rows,
                    )
                )
                print(json.dumps(dict(window=window, view=view, frames=decoded)), flush=True)
        summary = out / "summary.json"
        write_json(
            summary,
            dict(
                rows=summaries,
                source_fps=166,
                playback_fps=20,
                tracker_calls=0,
                ID_truth=False,
                contact_relative_frames=CONTACT_FRAMES,
            ),
        )
        record(summary)
        for p, digest in inputs.items():
            assert sha(Path(p)) == digest
        for r in outputs.values():
            assert sha(Path(r["path"])) == r["sha256"]
        assert {p.name for p in out.iterdir() if p.is_file()} == set(outputs)
        write_json(
            out / "render.provenance.json",
            dict(
                status="ok",
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
                inputs=inputs,
                outputs=outputs,
                interpretation="Qualitative saved-point overlays, not point or biological ID ground truth",
            ),
        )
    except BaseException as exc:
        write_json(
            out / "render.provenance.json",
            dict(
                status="failed",
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
                inputs=inputs,
                outputs=outputs,
                error=repr(exc),
            ),
        )
        raise


if __name__ == "__main__":
    run()
