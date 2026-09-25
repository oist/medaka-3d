"""Independent raw/mask/SLEAP observations and fixed pixel sampling for body fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from prepare_sequence import decode_frame_masks

from medaka3d.types import VIEWS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrameObservation:
    frame_idx: int
    detections: dict[str, np.ndarray]
    images_bgr: dict[str, np.ndarray]
    masks: dict[str, np.ndarray | None]
    backgrounds_bgr: dict[str, np.ndarray]


@dataclass(frozen=True)
class PixelTargets:
    pixels: dict[str, np.ndarray]
    target: np.ndarray  # V,N; foreground indicator, not calibrated probability
    weights: np.ndarray  # balanced positive/negative contribution; unavailable is zero
    foreground: np.ndarray  # V,N, independent of the candidate pose


class CachedNodes:
    """Read only the requested interval from the flat untracked SLEAP schema."""

    def __init__(self, input_root, experiment, frames):
        self.frames = np.asarray(frames)
        self.paths, self.data = {}, {}
        if len(self.frames) == 0 or np.any(np.diff(self.frames) != 1):
            raise ValueError("SLEAP interval must be nonempty and contiguous")
        for view in VIEWS:
            path = (
                Path(input_root) / "work/c01_s16_predictions/bottomup" / f"{experiment}__{view}.npz"
            )
            with np.load(path, allow_pickle=False) as arrays:
                if str(arrays["view"]) != view or str(arrays["experiment"]) != experiment:
                    raise ValueError(f"wrong detection-cache identity: {path}")
                if tuple(arrays["nodes"]) != ("mouth", "pectoral", "tail"):
                    raise ValueError("unexpected node order")
                index = arrays["frame_idx"]
                if np.any(np.diff(index) < 0):
                    raise ValueError("unsorted detection frames")
                lo = np.searchsorted(index, self.frames[0], side="left")
                hi = np.searchsorted(index, self.frames[-1], side="right")
                points = arrays["points"][lo:hi]
                selected = index[lo:hi] - self.frames[0]
            count = np.bincount(selected, minlength=len(frames))
            self.data[view] = (points, np.r_[0, np.cumsum(count)])
            self.paths[view] = path

    def at(self, view, frame):
        t = int(frame - self.frames[0])
        if not 0 <= t < len(self.frames):
            raise IndexError("frame outside staged node interval")
        points, offsets = self.data[view]
        return points[offsets[t] : offsets[t + 1]]


class SequenceReader:
    def __init__(self, root, sleap_root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("fps") != 166.0 or self.manifest.get("image_shape_hw") != [1200, 1200]:
            raise ValueError(
                "staged fps/image shape must declare native 166 fps and 1200-square BGR"
            )
        self.fps = float(self.manifest["fps"])
        self.image_shape = tuple(self.manifest["image_shape_hw"])
        self.frames = np.asarray(self.manifest["frames"], np.int64)
        self.nodes = CachedNodes(sleap_root, self.manifest["experiment"], self.frames)
        self.masks, self.backgrounds, self.videos, self.pixel_hashes = {}, {}, {}, {}
        for view in VIEWS:
            for category in ("raw", "masks", "backgrounds"):
                info = self.manifest[category][view]
                path = self.root / info["path"]
                if sha256(path) != info["sha256"]:
                    raise ValueError(f"changed {category} input for {view}")
                if category == "raw":
                    hashes = info.get("frame_bgr_sha256")
                    if (
                        not isinstance(hashes, list)
                        or len(hashes) != len(self.frames)
                        or any(
                            not isinstance(value, str)
                            or len(value) != 64
                            or any(char not in "0123456789abcdef" for char in value)
                            for value in hashes
                        )
                    ):
                        raise ValueError(f"missing or invalid per-frame BGR SHA256 for {view}")
                    self.videos[view] = path
                    self.pixel_hashes[view] = tuple(hashes)
                elif category == "masks":
                    with np.load(path, allow_pickle=False) as arrays:
                        self.masks[view] = {k: arrays[k] for k in arrays.files}
                    if not np.array_equal(self.masks[view]["frame_idx"], self.frames):
                        raise ValueError("mask frame mismatch")
                else:
                    image = cv2.imread(str(path))
                    if image is None or image.shape != (1200, 1200, 3):
                        raise ValueError("invalid background image")
                    self.backgrounds[view] = image

    def __iter__(self):
        captures = {v: cv2.VideoCapture(str(p)) for v, p in self.videos.items()}
        try:
            if any(not c.isOpened() for c in captures.values()):
                raise RuntimeError("cannot open one of the lossless raw clips")
            for view, capture in captures.items():
                if not np.isclose(capture.get(cv2.CAP_PROP_FPS), self.fps, rtol=0, atol=1e-6):
                    raise ValueError(f"raw fps mismatch for {view}")
                if capture.get(cv2.CAP_PROP_FRAME_COUNT) != len(self.frames):
                    raise ValueError(f"raw frame count mismatch for {view}")
                shape = (
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                )
                if shape != self.image_shape:
                    raise ValueError(f"raw image shape metadata mismatch for {view}")
            for t, frame in enumerate(self.frames):
                images, masks = {}, {}
                for view, capture in captures.items():
                    ok, image = capture.read()
                    if not ok or not np.isclose(
                        capture.get(cv2.CAP_PROP_POS_FRAMES), t + 1, rtol=0, atol=1e-6
                    ):
                        raise RuntimeError(f"raw frame alignment failed {view}/{frame}")
                    if (
                        image is None
                        or image.shape != (*self.image_shape, 3)
                        or image.dtype != np.uint8
                    ):
                        raise ValueError(f"decoded raw image shape/dtype mismatch {view}/{frame}")
                    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                    if digest != self.pixel_hashes[view][t]:
                        raise ValueError(f"decoded raw BGR SHA256 mismatch {view}/{frame}")
                    images[view] = image
                    if not self.masks[view]["view_available"][t]:
                        masks[view] = None
                    else:
                        masks[view] = decode_frame_masks(self.masks[view], t)
                yield FrameObservation(
                    int(frame),
                    {v: self.nodes.at(v, frame) for v in VIEWS},
                    images,
                    masks,
                    self.backgrounds,
                )
            for view, capture in captures.items():
                if capture.read()[0]:
                    raise ValueError(f"raw clip contains undeclared extra frames for {view}")
        finally:
            for capture in captures.values():
                capture.release()


def foreground_contrast(image, background):
    difference = image.astype(np.float32) - background.astype(np.float32)
    difference -= np.median(difference, axis=(0, 1))
    foreground = (np.max(np.abs(difference), axis=-1) >= 12).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 6
    return keep[labels]


def sample_pixels(frame, mode="detectron2", n_positive=384, n_near=384, n_global=256):
    """Sample evidence independently of every pose/ID candidate; no empty-mask veto.

    Segmentation returning zero foreground has zero fitting weight: model silence
    is not confident absence of all fish. Runtime availability is still preserved
    in the input. Samples use a fixed per-frame/view seed, shared by all hypotheses.
    """
    if mode not in ("detectron2", "top_contrast"):
        raise ValueError("unknown observation mode")
    for name, count in (("n_positive", n_positive), ("n_near", n_near), ("n_global", n_global)):
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count <= 0
        ):
            raise ValueError(f"{name} must be a strictly positive integer")
    n_samples = n_positive + n_near + n_global
    target = np.zeros((3, n_samples), np.float32)
    weights = np.zeros_like(target)
    fg = np.zeros((3, n_samples), bool)
    pixels = {}
    for v, view in enumerate(VIEWS):
        image = frame.images_bgr[view]
        height, width = image.shape[:2]
        rng = np.random.default_rng(frame.frame_idx * 7 + v + 3029)
        instance_masks = frame.masks[view]
        if mode == "top_contrast" and v == 0:
            union = foreground_contrast(image, frame.backgrounds_bgr[view])
        elif instance_masks is None:
            union = np.zeros((height, width), bool)
        else:
            union = np.any(instance_masks, axis=0)
        positive = np.flatnonzero(union)
        dilation = cv2.dilate(union.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        nearby = np.flatnonzero(dilation & ~union)
        background = np.flatnonzero(~union)

        def choose(pool, count, generator=rng):
            if len(pool):
                return generator.choice(pool, count, replace=len(pool) < count)
            return np.zeros(count, int)

        flat = np.concatenate(
            (choose(positive, n_positive), choose(nearby, n_near), choose(background, n_global))
        )
        y, x = np.unravel_index(flat, (height, width))
        pixels[view] = np.column_stack((x, y)).astype(float)
        if not len(positive):
            continue
        target[v, :n_positive] = 1
        fg[v, :n_positive] = True
        weights[v, :n_positive] = 1 / n_positive
        if len(nearby):
            weights[v, n_positive : n_positive + n_near] = 0.75 / n_near
        if len(background):
            weights[v, -n_global:] = 0.25 / n_global
    return PixelTargets(pixels, target, weights, fg)
