"""Show every saved raw SLEAP row and D2 contour at three fixed native frames.

No video decoding, detector inference, fitting, or tracking. Images remain full
1200-square; source row numbers are local to one camera and are not fish IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
from chunk_masks import SCHEMA_DTYPES  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402

STAGE = "nodewise_first_minute_raw_detection_review_v1"
PHOTOS = WORK / "nodewise_first_minute_peak_raw_review_v1"
PHOTO_SHA = "e00b8b99f67245d3fe5867df712978b946c863dcccd53d87b474f95c5f6f4840"
HISTORY = WORK / "SEGMENTATION_BODY_PROPOSAL_75053_SOURCE_HISTORY_V1.json"
HISTORY_SHA = "a4564d763d84785e808bd17d0dbf6156cba662fd5a31f61b2a6c9c02a92632fd"
ASSIGNED = WORK / "NODEWISE_FIRST_MINUTE_REACQUISITION_SOURCE_ROWS_V1.json"
ASSIGNED_SHA = "fbae74148ba4eb2a69c3831ec0d348d2aa92714d442ba36f5fac365ea343f42d"
FRAMES, VIEWS = (75053, 75054, 75055), ("top_cam", "side1_cam", "side2_cam")
NODES = ("mouth", "pectoral", "tail")
COLORS = ("#ffe34d", "#42dcff", "#ff85df", "#87ef76", "#ff9855", "#bba0ff")
SOURCE_PINS = {
    "labs/20260909_three_body/prepare_sequence.py": (
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8"
    ),
    "labs/20260909_three_body/chunk_masks.py": (
        "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81"
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sig(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=sha(array.tobytes()))


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Capture:
    def __init__(self):
        self.refs, self.stats = {}, {}

    def read(self, path, expected=None, size=None):
        path = Path(path).resolve()
        assert path.is_relative_to(REPO)
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        state = (before.st_size, before.st_mtime_ns)
        assert state == (after.st_size, after.st_mtime_ns)
        digest = sha(payload)
        assert expected is None or digest == expected, path
        assert size is None or len(payload) == size, path
        self.refs[str(path)] = dict(path=str(path), sha256=digest, size_bytes=len(payload))
        self.stats[path] = state
        return payload

    def document(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def guard(self):
        for path, state in self.stats.items():
            s = path.stat()
            assert (s.st_size, s.st_mtime_ns) == state, path


def member(capture, seal, name):
    ref = seal["outputs"][name]
    assert Path(ref["path"]).resolve() == (PHOTOS / name).resolve()
    return capture.read(PHOTOS / name, ref["sha256"], ref["size_bytes"])


def sl_rows(capture, history):
    result, report = {}, {}
    for ref in history["raw_SL_read_contract"]["registered_refs"]:
        view = Path(ref["path"]).stem.split("__")[-1]
        assert view in VIEWS
        payload = capture.read(
            ref["path"], ref["registered_sha256_not_rehashed"], ref["stat_size_bytes"]
        )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            assert len(archive.files) == len(set(archive.files))
            names = list(archive.files)
            # The full native-index and point members are expanded once, then sliced.
            needed = ("view", "experiment", "nodes", "frame_idx", "points")
            assert set(needed).issubset(names)
            data = {key: archive[key] for key in needed}
        assert str(data["view"]) == view
        assert str(data["experiment"]) == history["raw_SL_read_contract"]["experiment"]
        assert tuple(data["nodes"]) == NODES
        index, points = data["frame_idx"], data["points"]
        assert index.dtype.kind in "iu" and index.ndim == 1 and np.all(index[1:] >= index[:-1])
        assert points.dtype.kind == "f" and points.shape == (len(index), 3, 2)
        assert not np.isinf(points).any(), "infinite detection coordinates"
        report[view] = dict(
            archive_members=names,
            decoded_members=list(needed),
            arrays={k: sig(v) for k, v in data.items()},
            full_detection_rows=len(index),
            full_points_nan_coordinate_count=int(np.isnan(points).sum()),
            compressed_bytes=len(payload),
            archive_capture_count=1,
            archive_decode_count=1,
            full_members_expanded_before_three_frame_selection=True,
        )
        for frame in FRAMES:
            lo, hi = (
                np.searchsorted(index, frame, side="left"),
                np.searchsorted(index, frame, side="right"),
            )
            result[frame, view] = points[lo:hi].copy()
        del payload, data, index, points
    assert set(report) == set(VIEWS)
    return result, report


def selected_masks(capture, history, manifest, manifest_path):
    result, report = {}, {}
    for ref in history["mask_archive_refs"]:
        view = ref["view"]
        mref = manifest["masks"][view]
        assert Path(ref["path"]).resolve() == (manifest_path.parent / mref["path"]).resolve()
        assert ref["registered_sha256_not_rehashed"] == mref["sha256"]
        payload = capture.read(ref["path"], mref["sha256"], mref["bytes"])
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            assert len(archive.files) == 13 and set(archive.files) == set(SCHEMA_DTYPES)
            data = {key: archive[key] for key in archive.files}
        assert all(data[k].dtype == dt for k, dt in SCHEMA_DTYPES.items())
        np.testing.assert_array_equal(data["frame_idx"], np.arange(74700, 76360))
        np.testing.assert_array_equal(data["image_shape_hw"], [1200, 1200])
        report[view] = dict(
            arrays={k: sig(v) for k, v in data.items()},
            archive_capture_count=1,
            archive_decode_count=1,
            full_mask_frame_decode_count=0,
            selected_frame_decodes=3,
            frames={},
        )
        for frame in FRAMES:
            i = frame - 74700
            available = bool(data["view_available"][i])
            masks = decode_frame_masks(data, i)
            assert masks.dtype == bool and masks.shape[1:] == (1200, 1200)
            assert sha(masks.tobytes()) == data["frame_mask_sha256"][i].decode("ascii")
            assert (
                len(masks)
                == ref["native_instance_counts"][str(frame)]
                == mref["instances_per_frame"][i]
            )
            lo, hi = data["frame_offsets"][i : i + 2]
            np.testing.assert_array_equal(data["instance_id"][lo:hi], np.arange(len(masks)))
            assert available or len(masks) == 0
            result[frame, view] = masks if available else None
            report[view]["frames"][str(frame)] = dict(
                available=available,
                signature=sig(masks),
                instances=len(masks),
                mask_pixels=[int(m.sum()) for m in masks],
                original_mask_indices=list(range(len(masks))),
                scores=data["scores"][lo:hi].tolist(),
            )
        del payload, data
    return result, report


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)


def text(draw, xy, label, fill="white", size=22):
    draw.text(xy, label, font=font(size), fill=fill, stroke_width=2, stroke_fill="black")


def mark(draw, xy, node, color, label):
    x, y = (float(v) for v in xy)
    radius = 7
    if node == 0:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    elif node == 1:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    else:
        draw.polygon(
            [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)],
            outline=color,
            width=3,
        )
    text(draw, (min(max(x + 10, 2), 1070), min(max(y - 25, 2), 1160)), label, color, 21)


def panels(bgr, masks, points):
    raw = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d2, sl = raw.copy(), raw.copy()
    d2draw, sldraw = ImageDraw.Draw(d2), ImageDraw.Draw(sl)
    if masks is None:
        text(d2draw, (15, 15), "D2 unavailable: not background evidence", size=26)
    else:
        for number, mask in enumerate(masks):
            color = COLORS[(number + 3) % len(COLORS)]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                coords = [tuple(map(int, xy)) for xy in contour[:, 0]]
                if len(coords) > 1:
                    d2draw.line(coords + coords[:1], fill=color, width=3)
            yy, xx = np.nonzero(mask)
            if len(xx):
                # Label in a reproducible corner of its bounding box, preserving all contours.
                text(
                    d2draw,
                    (min(int(xx.min()), 1030), max(0, int(yy.min()) - 28)),
                    f"D2#{number}",
                    color,
                    24,
                )
    for row, body in enumerate(points):
        color = COLORS[row % len(COLORS)]
        finite = np.isfinite(body).all(-1)
        for a, b in ((0, 1), (1, 2)):
            if finite[a] and finite[b]:
                sldraw.line([tuple(body[a]), tuple(body[b])], fill=color, width=2)
        for node in np.flatnonzero(finite):
            mark(sldraw, body[node], int(node), color, f"r{row}{'MCT'[node]}")
    return raw, d2, sl


def self_check():
    """Only synthetic arrays; no input reader is entered."""
    raw = np.zeros((1200, 1200, 3), np.uint8)
    masks = np.zeros((2, 1200, 1200), bool)
    masks[0, 100:200, 100:200] = True
    points = np.array(
        [
            [[120.0, 110.0], [np.nan, np.nan], [170.0, 180.0]],
            [[500.0, 400.0], [520.0, 450.0], [540.0, 500.0]],
        ]
    )
    raw_before, points_before = raw.copy(), points.copy()
    rendered = panels(raw, masks, points)
    assert all(im.size == (1200, 1200) for im in rendered)
    assert np.array_equal(raw, raw_before) and np.array_equal(points, points_before, equal_nan=True)
    assert np.any(np.asarray(rendered[1])) and np.any(np.asarray(rendered[2]))
    print("synthetic drawing check PASS; no real inputs read")


def run(output):
    cv2.setNumThreads(1)
    output = Path(output).resolve()
    assert output.is_relative_to(WORK) and not output.exists(), "fresh output required"
    capture, outputs = Capture(), {}
    started = datetime.now(UTC).isoformat()
    for path, digest in SOURCE_PINS.items():
        capture.read(REPO / path, digest)
    source_bytes = capture.read(Path(__file__))
    history = capture.document(HISTORY, HISTORY_SHA)
    assert tuple(history["native_frames"]) == FRAMES
    photo_seal = capture.document(PHOTOS / "review.provenance.json", PHOTO_SHA)
    assert photo_seal["status"] == "ok" and photo_seal["stage"] == PHOTOS.name
    projection = json.loads(member(capture, photo_seal, "projection_evidence.json"))
    assigned = capture.document(ASSIGNED, ASSIGNED_SHA)
    assert assigned["selected_frames"] == list(FRAMES)
    manifest_ref = next(r for r in history["references"] if Path(r["path"]).name == "manifest.json")
    manifest_path = Path(manifest_ref["path"])
    manifest = capture.document(manifest_path, manifest_ref["sha256"])
    assert manifest["frames"] == list(range(74700, 76360))
    assert manifest["source_interval_half_open"] == [74700, 76360]
    assert manifest["image_shape_hw"] == [1200, 1200]
    points, sl_report = sl_rows(capture, history)
    masks, mask_report = selected_masks(capture, history, manifest, manifest_path)
    photo_rows = {(row["frame_idx"], row["view"]): row for row in projection["native_photographs"]}
    selected_sources = {
        arm: {row["frame_idx"]: row for row in rows} for arm, rows in assigned["arms"].items()
    }
    output.mkdir()
    (output / "views").mkdir()

    def write(name, payload):
        path = output / name
        with path.open("xb") as stream:
            stream.write(payload)
        outputs[name] = dict(path=str(path), sha256=sha(payload), size_bytes=len(payload))

    def write_image(name, im, fmt):
        buffer = io.BytesIO()
        im.save(buffer, format=fmt, **({"quality": 95, "subsampling": 0} if fmt == "JPEG" else {}))
        payload = buffer.getvalue()
        with Image.open(io.BytesIO(payload)) as check:
            check.load()
            assert check.size == im.size
        write(name, payload)

    records, thin_points = [], {}
    for frame in FRAMES:
        sheet = Image.new("RGB", (3600, 4180), "#14202b")
        title = ImageDraw.Draw(sheet)
        text(
            title, (25, 15), f"Native {frame} | ALL raw detections | full images, no crop", size=40
        )
        text(
            title,
            (25, 72),
            "Rows: original / D2 contours / SLEAP M(circle), C(square), T(triangle). "
            "Numbers are local, NOT fish IDs.",
            size=28,
        )
        for camera, view in enumerate(VIEWS):
            image_ref = photo_rows[frame, view]
            png = member(capture, photo_seal, f"native/frame_{frame}_{view}.png")
            assert sha(png) == image_ref["image"]["sha256"]
            bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            assert bgr is not None and sig(bgr) == image_ref["bgr_signature"]
            assert sha(bgr.tobytes()) == manifest["raw"][view]["frame_bgr_sha256"][frame - 74700]
            local_points, local_masks = points[frame, view], masks[frame, view]
            tiles = panels(bgr, local_masks, local_points)
            finite = np.isfinite(local_points).all(-1)
            inventories = [
                f"SL r{i}: "
                + "/".join(n if present else "-" for n, present in zip("MCT", row, strict=True))
                for i, row in enumerate(finite)
            ]
            x = camera * 1200
            for row, (tile, label) in enumerate(
                zip(tiles, ("ORIGINAL", "D2 - all instances", "SLEAP - all raw rows"), strict=True)
            ):
                y = 140 + row * 1260
                text(title, (x + 15, y), f"{view} | {label}", size=27)
                sheet.paste(tile, (x, y + 45))
            footer_y = 3930
            for i, label in enumerate(inventories):
                text(
                    title,
                    (x + 15 + (i // 6) * 560, footer_y + (i % 6) * 30),
                    label,
                    COLORS[i % len(COLORS)],
                    23,
                )
            selections = {}
            for arm, by_frame in selected_sources.items():
                source = np.asarray(by_frame[frame]["sources_by_body_view_node"])
                assert source.shape == (3, 3, 3)
                chosen = source[:, camera]
                for body, node in np.argwhere(chosen >= 0):
                    assert chosen[body, node] < len(local_points)
                    assert finite[chosen[body, node], node]
                selections[arm] = chosen.tolist()
            record = dict(
                frame_idx=frame,
                view=view,
                photo=image_ref,
                raw_SL_rows=len(local_points),
                raw_SL_points=clean(local_points),
                raw_SL_signature=sig(local_points),
                raw_SL_finite_nodes_by_row=finite.tolist(),
                raw_SL_nan_coordinate_count=int(np.isnan(local_points).sum()),
                selected_sources_by_body_node=selections,
                D2=mask_report[view]["frames"][str(frame)],
            )
            records.append(record)
            thin_points[f"f{frame}_{view}"] = local_points
            # Native-size two-panel detail preview retains screen edges and reflection regions.
            detail = Image.new("RGB", (2400, 1460), "#14202b")
            dd = ImageDraw.Draw(detail)
            text(
                dd,
                (20, 10),
                f"{frame} {view} | D2 contours (left) / ALL SLEAP rows (right) "
                "| local numbers, no IDs",
                size=28,
            )
            detail.paste(tiles[1], (0, 60))
            detail.paste(tiles[2], (1200, 60))
            for i, label in enumerate(inventories):
                text(
                    dd,
                    (1220 + (i // 5) * 570, 1270 + (i % 5) * 30),
                    label,
                    COLORS[i % len(COLORS)],
                    23,
                )
            for j, arm in enumerate(("tail", "nodewise")):
                s2 = selections[arm][2]
                text(
                    dd,
                    (20, 1290 + j * 42),
                    f"Saved {arm} S2 selected row M/C/T: {s2} (-1 = not assigned)",
                    size=24,
                )
            write_image(f"views/frame_{frame}_{view}.jpg", detail, "JPEG")
        write_image(f"frame_{frame}_all_detections.png", sheet, "PNG")
        preview = sheet.resize(
            (2400, round(sheet.height * 2400 / sheet.width)), Image.Resampling.LANCZOS
        )
        write_image(f"frame_{frame}_preview.jpg", preview, "JPEG")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **thin_points)
    write("raw_SL_selected.npz", buffer.getvalue())
    diagnostics = dict(
        schema=STAGE,
        status="ok",
        frames=FRAMES,
        views=VIEWS,
        source_sha256=sha(source_bytes),
        registered_contract_sha256=HISTORY_SHA,
        photo_seal_sha256=PHOTO_SHA,
        raw_SL_caches=sl_report,
        masks=mask_report,
        observations=records,
        raw_SL_selected_signatures={k: sig(v) for k, v in thin_points.items()},
        masks_and_SL_rows_are_independent_local_detection_indices=True,
        same_number_across_views_does_not_establish_correspondence=True,
        no_detection_filtered_by_tracking_assignment=True,
        all_pixels_and_image_edges_retained=True,
        overlay_is_visual_diagnostic_not_accuracy_or_physical_truth=True,
        projection_evidence_inherited_not_recomputed=True,
        stored_support_not_recomputed=True,
        full_SL_cache_members_decoded_only=list(
            ("view", "experiment", "nodes", "frame_idx", "points")
        ),
        original_video_decode_calls=0,
        detector_inference_calls=0,
        checkpoint39_reads=0,
        candidate_reads=0,
        grid_fit_tracking_GPU_calls=0,
    )
    write(
        "diagnostics.json",
        (json.dumps(clean(diagnostics), indent=2, allow_nan=False) + "\n").encode(),
    )
    write("source_snapshot.py", source_bytes)
    capture.guard()
    for name, ref in outputs.items():
        assert (output / name).stat().st_size == ref["size_bytes"]
    seal = dict(
        stage=STAGE,
        status="ok",
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        inputs=capture.refs,
        outputs=outputs,
    )
    with (output / "review.provenance.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(seal, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status="ok",
                output=str(output),
                output_count=len(outputs),
                seal_sha256=sha((output / "review.provenance.json").read_bytes()),
                SL_rows={f"{r['frame_idx']}_{r['view']}": r["raw_SL_rows"] for r in records},
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WORK / STAGE)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        run(args.output)
