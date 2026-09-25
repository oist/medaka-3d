"""Separate, measurable acceleration paths for frozen v3 image observations.

CPU parallelism retains v3 processing. CUDA moves only integer-valued RGB
differences, exact illumination medians and channel maxima to PyTorch; connected
components, distance maps and Farneback remain OpenCV CPU operations. Probability
division stays in NumPy to retain the original float32 rounding. GPU availability
never silently selects a different algorithm or falls back to CPU.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

import cv2
import numpy as np
from observations import (
    EvidenceConfig,
    SceneEvidence,
    ViewEvidence,
    _flow,
    _foreground,
    _readonly,
    _rgb,
)


class ParallelEvidenceBuilder:
    """Reusable three-view CPU worker pool. Call close(), or use a with block."""

    name = "v3-cpu-parallel"

    def __init__(self, config: EvidenceConfig | None = None, workers: int = 3):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.config = config or EvidenceConfig()
        self.workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fish-images")
        self._closed = False
        self.timings = {"calls": 0, "total_seconds": 0.0, "gpu_foreground_seconds": 0.0}
        self.last_timings = {}

    @property
    def metadata(self):
        return {"backend": self.name, "workers": self.workers, "opencv": cv2.__version__,
                "flow_backend": "OpenCV CPU Farneback", "flow_max_side": self.config.flow_max_side,
                "changes_flow_resolution": self.config.flow_max_side != 640}

    def close(self):
        self._pool.shutdown(wait=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _validate(self, frames, backgrounds, previous_frames):
        if self._closed:
            raise RuntimeError("evidence builder is closed")
        if set(backgrounds) - set(frames) or set(previous_frames) - set(frames):
            raise ValueError("background/previous camera names must be present in frames")

    def _record(self, started, gpu_seconds=0.0):
        duration = time.perf_counter() - started
        self.last_timings = {"total_seconds": duration, "gpu_foreground_seconds": gpu_seconds}
        self.timings["calls"] += 1
        self.timings["total_seconds"] += duration
        self.timings["gpu_foreground_seconds"] += gpu_seconds

    def from_frames(self, frames, backgrounds=None, previous_frames=None, config=None):
        started = time.perf_counter()
        backgrounds = {} if backgrounds is None else backgrounds
        previous_frames = {} if previous_frames is None else previous_frames
        self._validate(frames, backgrounds, previous_frames)
        config = config or self.config

        def make(name):
            scene = SceneEvidence.from_frames(
                {name: frames[name]},
                {name: backgrounds[name]} if name in backgrounds else {},
                {name: previous_frames[name]} if name in previous_frames else {}, config)
            return name, scene.views[name]

        views = dict(self._pool.map(make, frames))
        self._record(started)
        return SceneEvidence(MappingProxyType(views), config)


def _finish_foreground(probability, config):
    """Frozen v3 postprocessing of a probability map, without changing its mask."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (probability >= 0.25).astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= config.minimum_component_area
    keep[0] = False
    probability *= keep[labels]
    mask = probability >= 0.25
    if not mask.any():
        shape = probability.shape
        return (np.zeros(shape, np.float32), np.zeros(shape, np.float32),
                np.zeros(shape, np.int32), np.zeros(1, np.int32), 0.0)
    fraction = float(mask.mean())
    quality = min(1.0, config.maximum_foreground_fraction / fraction)
    quality *= max(0.0, (1.0 - fraction) / (1.0 - config.maximum_foreground_fraction))
    quality = min(quality, 1.0)
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return probability, distance, labels, areas, quality


class CudaEvidenceBuilder(ParallelEvidenceBuilder):
    """Exact-value CUDA foreground arithmetic plus parallel CPU flow and geometry."""

    name = "v3-cuda-foreground-cpu-flow"

    def __init__(self, config: EvidenceConfig | None = None, workers: int = 3, device="cuda:0"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CudaEvidenceBuilder requires an available CUDA device")
        # Force an actual kernel, not only a driver/device availability check.
        torch.ones(1, device=self.device).square().sum().item()
        super().__init__(config, workers)

    @property
    def metadata(self):
        return {**super().metadata, "torch": self.torch.__version__,
                "cuda_build": self.torch.version.cuda,
                "device": self.torch.cuda.get_device_name(self.device),
                "gpu_operations": "RGB difference, exact histogram median, channel max",
                "cpu_operations": "probability, connected components, distance transform, flow"}

    def _magnitudes(self, images, backgrounds):
        torch = self.torch
        groups = defaultdict(list)
        for view, image in images.items():
            if view in backgrounds:
                if image.shape != backgrounds[view].shape:
                    raise ValueError("a background must match its camera frame shape")
                groups[image.shape].append(view)
        magnitudes = {}
        with torch.inference_mode():
            for shape, names in groups.items():
                pixels = shape[0] * shape[1]
                if pixels >= 2**24:
                    raise ValueError("exact CUDA histogram counts require fewer than 2**24 pixels")
                current = torch.from_numpy(np.stack([images[name] for name in names])).to(
                    self.device, dtype=torch.float32)
                reference = torch.from_numpy(np.stack([backgrounds[name] for name in names])).to(
                    self.device, dtype=torch.float32)
                difference = current - reference
                channels = difference.permute(0, 3, 1, 2).reshape(-1, pixels)
                offsets = torch.arange(len(names) * 3, device=self.device)[:, None] * 512
                bins = len(names) * 3 * 512
                histogram = torch.histc((channels + 255 + offsets).flatten(), bins=bins,
                                       min=-0.5, max=bins - 0.5).reshape(-1, 512)
                cumulative = histogram.cumsum(dim=1)
                first = (cumulative <= (pixels - 1) // 2).sum(dim=1) - 255
                second = (cumulative <= pixels // 2).sum(dim=1) - 255
                median = ((first + second).to(torch.float32) * 0.5).reshape(len(names), 1, 1, 3)
                magnitude = (difference - median).abs().amax(dim=-1).cpu().numpy()
                magnitudes.update(zip(names, magnitude, strict=True))
        return magnitudes

    def from_frames(self, frames, backgrounds=None, previous_frames=None, config=None):
        started = time.perf_counter()
        backgrounds = {} if backgrounds is None else backgrounds
        previous_frames = {} if previous_frames is None else previous_frames
        self._validate(frames, backgrounds, previous_frames)
        config = config or self.config
        images = {name: _rgb(frame) for name, frame in frames.items()}
        backgrounds = {name: _rgb(frame) for name, frame in backgrounds.items()}
        previous = {name: _rgb(frame) for name, frame in previous_frames.items()}
        self.torch.cuda.synchronize(self.device)
        gpu_started = time.perf_counter()
        magnitudes = self._magnitudes(images, backgrounds)
        self.torch.cuda.synchronize(self.device)
        gpu_seconds = time.perf_counter() - gpu_started

        def make(name):
            rgb = images[name]
            if name in magnitudes:
                probability = np.clip(
                    (magnitudes[name] - config.difference_threshold)
                    / (2 * config.difference_threshold), 0, 1).astype(np.float32)
                foreground, distance, labels, areas, quality = _finish_foreground(
                    probability, config)
            else:
                foreground, distance, labels, areas, quality = _foreground(rgb, None, config)
            flow, confidence, scale = _flow(rgb, previous.get(name), config)
            return name, ViewEvidence(
                rgb, _readonly(foreground), _readonly(distance), _readonly(labels),
                _readonly(areas), quality, None if flow is None else _readonly(flow),
                None if confidence is None else _readonly(confidence), scale)

        views = dict(self._pool.map(make, images))
        self._record(started, gpu_seconds)
        return SceneEvidence(MappingProxyType(views), config)
