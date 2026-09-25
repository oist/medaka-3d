"""Exact native observations from a window inside the original local source video.

This storage alternative reuses SequenceReader's raw-file SHA, mask, background
and untracked-node validation. Only source-video offset/length handling changes.
The decoder opens source files read-only; hardlinks must never be edited in place.
"""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from observations import FrameObservation, SequenceReader
from prepare_sequence import decode_frame_masks

from medaka3d.types import VIEWS

SOURCE_TOTAL_FRAMES = 348600


def validate_source_manifest(root, manifest):
    if manifest.get("raw_storage") != "original_source_video":
        raise ValueError("source reader requires raw_storage='original_source_video'")
    values = manifest.get("frames")
    if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
        raise ValueError("source frames must be a nonempty integer list")
    if any(b != a + 1 for a, b in pairwise(values)):
        raise ValueError("source frames must be consecutive at native rate")
    if values[0] < 0 or values[-1] >= SOURCE_TOTAL_FRAMES:
        raise ValueError("source frame window is outside the registered recording")
    offsets = {}
    for category in ("raw", "masks", "backgrounds"):
        if set(manifest[category]) != set(VIEWS):
            raise ValueError("every source-view mapping must explicitly name the same three views")
        for view in VIEWS:
            text = str(manifest[category][view]["path"]).replace("\\", "/")
            relative = PurePosixPath(text)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or ":" in text
                or not relative.parts
            ):
                raise ValueError(
                    "source descriptor paths must remain relative and inside the descriptor"
                )
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root.resolve(strict=True)) or not path.is_file():
                raise ValueError("source descriptor path escaped its own directory")
            if category == "raw":
                info = manifest[category][view]
                offset, total = info.get("source_offset_frames"), info.get("source_total_frames")
                if type(offset) is not int or offset != values[0]:
                    raise ValueError(
                        "source offset must equal the first declared native source frame"
                    )
                if type(total) is not int or total != SOURCE_TOTAL_FRAMES:
                    raise ValueError("source_total_frames must equal the registered 348600 frames")
                offsets[view] = offset
    return offsets


class SourceSequenceReader(SequenceReader):
    """Yield exactly the declared window; the original video may continue after it."""

    def __init__(self, root, sleap_root):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.source_offsets = validate_source_manifest(root, manifest)
        super().__init__(root, sleap_root)
        if self.manifest != manifest:
            raise ValueError("source manifest changed while the reader was initialized")

    def __iter__(self):
        captures = {}
        try:
            # Creation is inside try: an open error after camera one still closes it.
            for view in VIEWS:
                capture = cv2.VideoCapture(str(self.videos[view]))
                captures[view] = capture
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open original source video for {view}")
                if not np.isclose(capture.get(cv2.CAP_PROP_FPS), self.fps, rtol=0, atol=1e-6):
                    raise ValueError(f"source fps mismatch for {view}")
                if capture.get(cv2.CAP_PROP_FRAME_COUNT) != SOURCE_TOTAL_FRAMES:
                    raise ValueError(f"original source frame count mismatch for {view}")
                shape = (
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                )
                if shape != self.image_shape:
                    raise ValueError(f"original source dimensions mismatch for {view}")
                offset = self.source_offsets[view]
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, offset):
                    raise RuntimeError(f"original source seek failed for {view}")
                if not np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), offset, rtol=0, atol=1e-6):
                    raise ValueError(f"original source seek position mismatch for {view}")
            for t, frame in enumerate(self.frames):
                images, masks = {}, {}
                for view in VIEWS:
                    capture = captures[view]
                    ok, image = capture.read()
                    expected = self.source_offsets[view] + t + 1
                    if not ok or not np.isclose(
                        capture.get(cv2.CAP_PROP_POS_FRAMES), expected, rtol=0, atol=1e-6
                    ):
                        raise RuntimeError(f"original source frame alignment failed {view}/{frame}")
                    if (
                        image is None
                        or image.shape != (*self.image_shape, 3)
                        or image.dtype != np.uint8
                    ):
                        raise ValueError(
                            f"decoded original source shape/dtype mismatch {view}/{frame}"
                        )
                    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                    if digest != self.pixel_hashes[view][t]:
                        raise ValueError(
                            f"decoded original source BGR SHA256 mismatch {view}/{frame}"
                        )
                    images[view] = image
                    masks[view] = (
                        decode_frame_masks(self.masks[view], t)
                        if self.masks[view]["view_available"][t]
                        else None
                    )
                yield FrameObservation(
                    int(frame),
                    {v: self.nodes.at(v, frame) for v in VIEWS},
                    images,
                    masks,
                    self.backgrounds,
                )
            # No extra read: stop is the requested window boundary, not source-video EOF.
        finally:
            for capture in captures.values():
                capture.release()
