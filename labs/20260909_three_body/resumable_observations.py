"""Seek a verified full sequence to its next native frame without changing inputs.

The caller must verify the producer's completion seal and construct the ordinary
reader first. This adapter retains its full manifest, SLEAP cache, mask arrays and
backgrounds. It neither initializes a tracker nor reconstructs its previous image;
that observation belongs in the tracker's checkpoint. Close the returned generator
when stopping early. Reading the declared end yields nothing and opens no videos.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import cv2
import numpy as np
from observations import FrameObservation, SequenceReader
from prepare_sequence import decode_frame_masks
from source_sequence import SOURCE_TOTAL_FRAMES, SourceSequenceReader

from medaka3d.types import VIEWS


def _start_index(reader, start_frame):
    if not isinstance(reader, SequenceReader):
        raise TypeError("resume requires an initialized SequenceReader or SourceSequenceReader")
    if isinstance(start_frame, (bool, np.bool_)) or not isinstance(start_frame, (int, np.integer)):
        raise TypeError("start_frame must be an integer native frame number")
    frames = reader.frames
    if (
        not isinstance(frames, np.ndarray)
        or frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or len(frames) == 0
        or np.any(np.diff(frames) != 1)
        or frames[0] < 0
        or not np.array_equal(frames, reader.manifest["frames"])
        or not np.array_equal(frames, reader.nodes.frames)
    ):
        raise ValueError("resume requires the unchanged full consecutive native frame interval")
    if reader.fps != 166.0 or reader.image_shape != (1200, 1200):
        raise ValueError("resume requires native 166 fps and 1200-square BGR")
    source = isinstance(reader, SourceSequenceReader)
    if (reader.manifest.get("raw_storage") == "original_source_video") != source:
        raise ValueError("source-video storage must use SourceSequenceReader")
    for view in VIEWS:
        if (
            not np.array_equal(reader.masks[view]["frame_idx"], frames)
            or len(reader.pixel_hashes[view]) != len(frames)
            or len(reader.masks[view]["view_available"]) != len(frames)
        ):
            raise ValueError(f"resume mask/pixel indices differ from the full interval for {view}")
        if source and (
            reader.source_offsets[view] != int(frames[0])
            or reader.manifest["raw"][view]["source_offset_frames"] != int(frames[0])
            or reader.manifest["raw"][view]["source_total_frames"] != SOURCE_TOTAL_FRAMES
            or frames[-1] >= SOURCE_TOTAL_FRAMES
        ):
            raise ValueError(
                f"resume source offset/count differs from the full interval for {view}"
            )
    index = int(start_frame) - int(frames[0])
    if not 0 <= index <= len(frames):
        raise ValueError("resume frame is outside the declared interval (or its exclusive end)")
    return index, source


def iter_from(reader: SequenceReader, start_frame: int) -> Iterator[FrameObservation]:
    """Yield ``start_frame .. reader.frames[-1]`` at the original native rate.

    FFV1 clips seek to ``start_frame - full_first_frame``; original MP4s seek to
    ``start_frame``. Pixel hashes and packed masks always use the full-sequence
    index. The original-video window ends without probing source EOF; a lossless
    clip must have no extra frame. Opening, alignment or decode errors release
    every capture opened so far, as does explicit generator.close().
    """
    index, source = _start_index(reader, start_frame)
    if index == len(reader.frames):
        return
    offset = int(start_frame) if source else index
    expected_count = SOURCE_TOTAL_FRAMES if source else len(reader.frames)
    captures = {}
    try:
        for view in VIEWS:
            capture = cv2.VideoCapture(str(reader.videos[view]))
            captures[view] = capture
            if not capture.isOpened():
                raise RuntimeError(f"cannot open resume video for {view}")
            if not np.isclose(capture.get(cv2.CAP_PROP_FPS), reader.fps, rtol=0, atol=1e-6):
                raise ValueError(f"resume video fps mismatch for {view}")
            if capture.get(cv2.CAP_PROP_FRAME_COUNT) != expected_count:
                raise ValueError(f"resume video frame count mismatch for {view}")
            shape = (
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            )
            if shape != reader.image_shape:
                raise ValueError(f"resume video dimensions mismatch for {view}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, offset):
                raise RuntimeError(f"resume video seek failed for {view}")
            if not np.isclose(capture.get(cv2.CAP_PROP_POS_FRAMES), offset, rtol=0, atol=1e-6):
                raise ValueError(f"resume video seek position mismatch for {view}")
        for t in range(index, len(reader.frames)):
            frame = int(reader.frames[t])
            images, masks = {}, {}
            for view in VIEWS:
                capture = captures[view]
                ok, image = capture.read()
                expected = frame + 1 if source else t + 1
                if not ok or not np.isclose(
                    capture.get(cv2.CAP_PROP_POS_FRAMES), expected, rtol=0, atol=1e-6
                ):
                    raise RuntimeError(f"resume native frame alignment failed {view}/{frame}")
                if (
                    image is None
                    or image.shape != (*reader.image_shape, 3)
                    or image.dtype != np.uint8
                ):
                    raise ValueError(f"resume decoded shape/dtype mismatch {view}/{frame}")
                digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                if digest != reader.pixel_hashes[view][t]:
                    raise ValueError(f"resume decoded BGR SHA256 mismatch {view}/{frame}")
                images[view] = image
                masks[view] = (
                    decode_frame_masks(reader.masks[view], t)
                    if reader.masks[view]["view_available"][t]
                    else None
                )
            yield FrameObservation(
                frame,
                {view: reader.nodes.at(view, frame) for view in VIEWS},
                images,
                masks,
                reader.backgrounds,
            )
        if not source:
            for view, capture in captures.items():
                if capture.read()[0]:
                    raise ValueError(f"resume raw clip contains undeclared extra frames for {view}")
    finally:
        for capture in captures.values():
            capture.release()
