"""Extract exact raw frames for the independent-mask feasibility diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from medaka3d.types import VIEWS

EXPERIMENT = "20251225_K_1_2_FCMC"
VIDEO_NAMES = {
    "top_cam": "cam0_top.mp4",
    "side1_cam": "cam1_side.mp4",
    "side2_cam": "cam2_side.mp4",
}
BACKGROUND_FRAMES = (20000, 40000, 60000, 80000, 100000, 160000, 220000, 280000, 340000)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_exact(cap, frame: int, view: str):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame):
        raise RuntimeError(f"cannot seek {view} to {frame}")
    ok, image = cap.read()
    if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
        raise RuntimeError(f"frame position mismatch {view}/{frame}")
    if image.shape != (1200, 1200, 3):
        raise RuntimeError(f"unexpected shape {view}/{frame}: {image.shape}")
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = [int(f) for f in manifest["frames"]]
    if len(frames) != len(set(frames)) or not frames or frames != sorted(frames):
        raise ValueError("frame indices must be nonempty, sorted and unique")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "experiment": EXPERIMENT,
        "frames": frames,
        "videos": {},
        "outputs": {},
        "input_manifest_sha256": digest(args.manifest),
        "alignment": "OpenCV frame positions verified; no new timestamp sync audit",
        "background_frames": list(BACKGROUND_FRAMES),
        "background_rule": "fixed offline temporal median; no labels or track inputs",
        "source_sha256": digest(Path(__file__)),
    }
    for view in VIEWS:
        path = args.raw_root / EXPERIMENT / view / VIDEO_NAMES[view]
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {path}")
            metadata = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "frames": round(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
            }
            if metadata["frames"] != 348600 or not np.isclose(metadata["fps"], 166):
                raise ValueError(f"unexpected raw video metadata: {metadata}")
            record["videos"][view] = metadata
            for frame in frames:
                image = read_exact(cap, frame, view)
                output = args.out / f"f{frame:06d}_{view}.png"
                if not cv2.imwrite(str(output), image):
                    raise RuntimeError(f"cannot write {output}")
                record["outputs"][output.name] = {"sha256": digest(output)}
            background = np.median(
                np.stack([read_exact(cap, f, view) for f in BACKGROUND_FRAMES]), axis=0
            ).astype(np.uint8)
            output = args.out / f"background_{view}.png"
            if not cv2.imwrite(str(output), background):
                raise RuntimeError(f"cannot write {output}")
            record["outputs"][output.name] = {"sha256": digest(output)}
            print(
                json.dumps(
                    {
                        "view": view,
                        "raw_frames": len(frames),
                        "elapsed_s": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
        finally:
            cap.release()
    record["elapsed_s"] = time.perf_counter() - started
    (args.out / "manifest.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
