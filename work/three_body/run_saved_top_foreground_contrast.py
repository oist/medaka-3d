"""Fixed three-anchor foreground disagreement diagnostic; no tracking or video decode.

Uses existing native PNGs, the frozen median background and one captured D2
archive. Threshold 12 / component area 6 are inherited without tuning. D2 is a
comparison observation, never ground truth. Original images and sources remain
unchanged. No SLEAP/candidate/checkpoint/identity arrays are inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from chunk_masks import SCHEMA_DTYPES  # noqa: E402
from observations import foreground_contrast  # noqa: E402
from prepare_sequence import decode_frame_masks  # noqa: E402

STAGE = "saved_top_foreground_contrast_v1"
FRAMES = (90120, 90220, 90380)
PACK = WORK / "fast_tail_transition_raw_review_pack_v1"
CHUNK = WORK / "inputs/long_source_masks_49800-348600_v1/chunks/f089640-091300/attempt_0001"
PACK_SHA = "8622bd5655ecc34058c5e3a7a2fe64e7ff357a3d9eb508cfca4e258405fb0c7f"
MANIFEST_SHA = "17a9cf8721d1dc4868b7bbc9881e0222e47b6b84a8955ccb0b4b8b29e840cd92"
STAGING_SHA = "c0456a25bf4f209f65bb2cf61afa42133bf0a708e5116a2673f69109223eaadb"
PINS = {
    "labs/20260909_three_body/observations.py": (
        "f1efcd27b31e63ce693229c9e765152acfc7984b3b8839f1792bcc2efd9cb5df"
    ),
    "labs/20260909_three_body/prepare_sequence.py": (
        "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8"
    ),
    "labs/20260909_three_body/chunk_masks.py": (
        "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81"
    ),
    "src/medaka3d/types.py": "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
    "src/medaka3d/provenance.py": (
        "1e2c260c959be0f240f244d064ec0f2b4c7f82cd7ecf0d9f29a82bdb1b2150c1"
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signature(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=sha(array.tobytes()))


class Evidence:
    """Hash the exact consumed bytes; final stat guard never rereads the mask archive."""

    def __init__(self):
        self.files = {}

    def capture(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        before = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        data = path.read_bytes()
        after = path.stat()
        require(
            before == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "input changed",
        )
        digest = sha(data)
        require(expected is None or digest == expected, "input SHA mismatch: " + str(path))
        self.files[str(path)] = dict(sha256=digest, size_bytes=len(data), stat=list(before))
        return data

    def read(self, path, expected=None):
        return json.loads(self.capture(path, expected))

    def member(self, root, seal, filename):
        path = (root / filename).resolve()
        matches = [v for v in seal["outputs"].values() if Path(v["path"]).resolve() == path]
        require(len(matches) == 1, "unique sealed member required")
        return self.capture(path, matches[0]["sha256"])

    def final_guard(self):
        for path, record in self.files.items():
            stat = Path(path).stat()
            require(
                record["stat"] == [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
                "input changed after capture",
            )


def decode_png(data, expected_pixel_sha):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    require(
        image is not None and image.shape == (1200, 1200, 3) and image.dtype == np.uint8,
        "native BGR PNG required",
    )
    require(
        expected_pixel_sha is None or sha(image.tobytes()) == expected_pixel_sha,
        "native BGR SHA mismatch",
    )
    return image


def load_headers(evidence):
    pack = evidence.read(PACK / "review.provenance.json", PACK_SHA)
    require(
        pack["status"] == "ok" and pack["stage"] == "fast_tail_transition_raw_review_pack_v1",
        "successful raw review required",
    )
    summary = json.loads(evidence.member(PACK, pack, "summary.json"))
    manifest = evidence.read(CHUNK / "manifest.json", MANIFEST_SHA)
    require(
        json.loads(evidence.member(PACK, pack, "source_manifest.json")) == manifest,
        "raw source manifest differs",
    )
    staging = evidence.read(CHUNK / "staging.provenance.json", STAGING_SHA)
    chunk = summary["consumed_record"]
    require(
        staging["status"] == "ok" and staging["stage"] == "prepare_independent_mask_chunk",
        "successful D2 staging required",
    )
    require(
        (chunk["index"], chunk["start"], chunk["stop"]) == (24, 89640, 91300), "wrong native chunk"
    )
    require(
        chunk["manifest_sha256"] == MANIFEST_SHA and chunk["provenance_sha256"] == STAGING_SHA,
        "wrong consumed chunk SHA",
    )
    require(
        {k: dict(path=v["path"], sha256=v["sha256"]) for k, v in staging["outputs"].items()}
        == chunk["output_hashes"],
        "D2 consumed outputs differ",
    )
    require(
        manifest["experiment"] == "20251225_K_1_2_FCMC"
        and manifest["fps"] == 166
        and manifest["image_shape_hw"] == [1200, 1200],
        "native experiment differs",
    )
    require(
        manifest["frames"] == list(range(89640, 91300))
        and manifest["raw_storage"] == "original_source_video",
        "native source interval differs",
    )
    require(
        summary["probe_frames"] == list(FRAMES)
        and summary["review_clip_range"] == [90040, 90400]
        and summary["model_outputs_shown"] is False,
        "saved raw anchors differ",
    )
    require(
        manifest["background_frames"] == list(range(5000, 45001, 5000)),
        "background sampling differs",
    )
    for category, role in (("backgrounds", "top_cam_background"), ("masks", "top_cam_masks")):
        desc = manifest[category]["top_cam"]
        require(
            Path(chunk["output_hashes"][role]["path"]).resolve() == (CHUNK / desc["path"]).resolve()
            and chunk["output_hashes"][role]["sha256"] == desc["sha256"],
            "top observation binding differs",
        )
    return pack, summary, manifest


def load_masks(data, manifest):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        require(
            len(archive.files) == len(SCHEMA_DTYPES) and set(archive.files) == set(SCHEMA_DTYPES),
            "packed schema additions or duplicates",
        )
        arrays = {k: archive[k] for k in archive.files}
    for name, dtype in SCHEMA_DTYPES.items():
        require(arrays[name].dtype == dtype, "packed dtype differs: " + name)
        arrays[name].flags.writeable = False
    require(
        np.array_equal(arrays["frame_idx"], manifest["frames"])
        and np.array_equal(arrays["image_shape_hw"], [1200, 1200]),
        "packed native frame order differs",
    )
    return arrays


def disagreement(foreground, masks):
    require(foreground.dtype == bool and foreground.ndim == 2, "boolean foreground required")
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    components = [
        dict(component=i, area=int(stats[i, cv2.CC_STAT_AREA]), bbox_xywh=stats[i, :4].tolist())
        for i in range(1, count)
    ]
    result = dict(
        foreground_pixels=int(foreground.sum()),
        component_count=len(components),
        component_areas_descending=sorted((c["area"] for c in components), reverse=True),
        components=components,
        D2_available=masks is not None,
    )
    if masks is None:
        result["disagreement"] = None
        return result, None
    require(
        masks.dtype == bool and masks.ndim == 3 and masks.shape[1:] == foreground.shape,
        "aligned boolean D2 masks required",
    )
    union = masks.any(axis=0)
    both, outside, missed = (
        int((foreground & union).sum()),
        int((foreground & ~union).sum()),
        int((~foreground & union).sum()),
    )
    result.update(
        D2_instance_count=len(masks),
        D2_instance_areas=masks.sum(axis=(1, 2)).tolist(),
        D2_union_pixels=int(union.sum()),
        disagreement=dict(
            foreground_inside_D2_union=both,
            foreground_outside_D2_union=outside,
            D2_union_outside_foreground=missed,
            union_IoU=None if both + outside + missed == 0 else both / (both + outside + missed),
            per_D2_instance_foreground_intersection=(masks & foreground[None])
            .sum(axis=(1, 2))
            .tolist(),
        ),
    )
    return result, union


def panels(image, foreground, masks, frame):
    """Every source panel preserves the native lattice; labels are in added header rows."""
    fg = image.copy()
    fg[foreground] = ((fg[foreground].astype(np.uint16) + np.array([0, 255, 0])) // 2).astype(
        np.uint8
    )
    d2 = image.copy()
    palette = ((0, 150, 255), (255, 180, 0), (180, 0, 255), (0, 255, 150), (255, 0, 100))
    if masks is not None:
        for i, mask in enumerate(masks):
            d2[mask] = (
                (d2[mask].astype(np.uint16) + np.array(palette[i % len(palette)])) // 2
            ).astype(np.uint8)
    union = np.zeros(foreground.shape, bool) if masks is None else masks.any(axis=0)
    diff = np.zeros_like(image)
    diff[foreground & union] = (255, 255, 255)
    diff[foreground & ~union] = (255, 0, 255)
    diff[~foreground & union] = (255, 255, 0)
    canvas = np.zeros((2528, 2400, 3), np.uint8)
    titles = (
        f"Native top_cam {frame} / {frame / 166:.3f}s",
        "Frozen foreground: threshold12, min6; green overlay",
        "Independent D2 masks; colors are frame-local",
        "White=both; magenta=FG only; cyan=D2 only (not truth)",
    )
    for i, tile in enumerate((image, fg, d2, diff)):
        y, x = (i // 2) * 1264, (i % 2) * 1200
        canvas[y + 64 : y + 1264, x : x + 1200] = tile
        cv2.putText(
            canvas,
            titles[i],
            (x + 12, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def execute(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK.resolve()) and not output.exists(),
        "fresh output inside work/three_body required",
    )
    output.mkdir()
    evidence, started = Evidence(), perf_counter()
    try:
        for name, digest in PINS.items():
            evidence.capture(REPO / name, digest)
        evidence.capture(Path(__file__))
        pack, summary, manifest = load_headers(evidence)
        desc = manifest["backgrounds"]["top_cam"]
        background = decode_png(evidence.capture(CHUNK / desc["path"], desc["sha256"]), None)
        desc = manifest["masks"]["top_cam"]
        arrays = load_masks(evidence.capture(CHUNK / desc["path"], desc["sha256"]), manifest)
        records = []
        for frame in FRAMES:
            i = frame - 89640
            pixel_sha = manifest["raw"]["top_cam"]["frame_bgr_sha256"][i]
            require(
                pixel_sha == summary["native_bgr_sha256"]["top_cam"][frame - 90040],
                "saved/source native pixel binding differs",
            )
            image = decode_png(evidence.member(PACK, pack, f"f{frame}_top_cam.png"), pixel_sha)
            decoded = decode_frame_masks(arrays, i)
            require(
                sha(decoded.tobytes()) == bytes(arrays["frame_mask_sha256"][i]).decode("ascii"),
                "decoded mask frame SHA differs",
            )
            masks = decoded if arrays["view_available"][i] else None
            tic = perf_counter()
            foreground = foreground_contrast(image, background)
            elapsed = perf_counter() - tic
            metrics, _ = disagreement(foreground, masks)
            metrics.update(
                frame_idx=frame,
                native_BGR_sha256=pixel_sha,
                mask_signature=signature(decoded),
                foreground_signature=signature(foreground),
                foreground_seconds=elapsed,
            )
            records.append(metrics)
            for filename, pixels in (
                (f"f{frame}_panels.png", panels(image, foreground, masks, frame)),
                (f"f{frame}_foreground.png", foreground.astype(np.uint8) * 255),
            ):
                require(cv2.imwrite(str(output / filename), pixels), "PNG output failed")
        write_json(
            output / "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                frames=list(FRAMES),
                records=records,
                packed_arrays={k: signature(v) for k, v in arrays.items()},
                background_BGR_signature=signature(background),
                runtime=dict(numpy=np.__version__, opencv=cv2.__version__),
                elapsed_seconds=perf_counter() - started,
                mask_archive_reads=1,
                mask_frame_decodes=3,
                raw_video_decodes=0,
                SLEAP_candidate_checkpoint_reads=0,
                GPU_used=False,
                foreground_default_unchanged=True,
                comparison_is_truth=False,
                tracking_or_review_changed=False,
                interpretation=(
                    "Overlap, component counts and IoU are disagreement statistics between two "
                    "observations, not accuracy, physical identities or fish count. Background "
                    "samples are from the user-described introduction period; no replacement "
                    "or threshold selection is performed."
                ),
            ),
        )
        evidence.final_guard()
        outputs = {
            p.name: dict(path=str(p), size_bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
            for p in output.iterdir()
            if p.is_file()
        }
        write_json(
            output / "diagnostic.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                inputs=evidence.files,
                outputs=outputs,
                input_capture_policy=(
                    "Exact consumed bytes SHA; final unchanged stat guard. "
                    "Mask archive captured once and never rehashed."
                ),
            ),
        )
        return dict(
            status="ok",
            output=str(output),
            frames=list(FRAMES),
            elapsed_seconds=perf_counter() - started,
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_seconds=perf_counter() - started,
            ),
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(execute(args.output)), flush=True)


if __name__ == "__main__":
    main()
