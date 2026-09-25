"""Acceleration must preserve observation values; flow resizing is explicit."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"))

from accelerated_observations import CudaEvidenceBuilder, ParallelEvidenceBuilder
from observations import EvidenceConfig, SceneEvidence


def assert_same(reference, actual):
    assert reference.config == actual.config
    assert reference.views.keys() == actual.views.keys()
    for view in reference.views:
        first, second = reference.views[view], actual.views[view]
        for field in ("rgb", "foreground", "distance", "labels", "component_areas",
                      "flow", "flow_confidence"):
            assert np.array_equal(getattr(first, field), getattr(second, field)), (view, field)
        assert first.quality == second.quality
        assert first.flow_scale == second.flow_scale


@pytest.mark.parametrize("shape", [(63, 81), (64, 80)])
def test_parallel_cpu_is_exact_and_keeps_explicit_flow_configuration(shape):
    rng = np.random.default_rng(764)
    backgrounds = {view: rng.integers(30, 220, (*shape, 3), dtype=np.uint8)
                   for view in ("top_cam", "side1_cam", "side2_cam")}
    frames = {view: image.copy() for view, image in backgrounds.items()}
    for image in frames.values():
        image[20:27, 20:48] = (10, 240, 30)
    config = EvidenceConfig(flow_max_side=32)
    reference = SceneEvidence.from_frames(frames, backgrounds, backgrounds, config)
    with ParallelEvidenceBuilder(config) as builder:
        actual = builder.from_frames(frames, backgrounds, backgrounds)
        assert_same(reference, actual)
        assert builder.metadata["changes_flow_resolution"]
        assert builder.metadata["flow_max_side"] == 32
        assert builder.timings["calls"] == 1
        assert builder.timings["gpu_foreground_seconds"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        builder.from_frames(frames)


@pytest.fixture(scope="module")
def cuda_builder():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for actual kernel equivalence checks")
    with CudaEvidenceBuilder() as builder:
        yield builder


@pytest.mark.parametrize("mode", ["random", "illumination", "extreme", "stationary", "missing"])
def test_cuda_foreground_matches_v3_all_maps_and_flow(cuda_builder, mode):
    rng = np.random.default_rng(284)
    backgrounds, frames = {}, {}
    # Different image sizes exercise both odd/even medians and grouped uploads.
    for view, shape in (("top_cam", (65, 71)), ("side1_cam", (64, 72)),
                        ("side2_cam", (64, 72))):
        background = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
        if mode == "random":
            current = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
        elif mode == "illumination":
            current = np.minimum(background.astype(np.int16) + 17, 255).astype(np.uint8)
        elif mode == "extreme":
            current = np.zeros_like(background)
            current[:, ::2] = 255
        else:
            current = background.copy()
            current[20:28, 15:40] = (0, 100, 255)
        backgrounds[view], frames[view] = background, current
    previous = frames if mode == "stationary" else backgrounds
    chosen_background = {} if mode == "missing" else backgrounds
    reference = SceneEvidence.from_frames(frames, chosen_background, previous)
    actual = cuda_builder.from_frames(frames, chosen_background, previous)
    assert_same(reference, actual)
    assert "cuda" in cuda_builder.metadata["backend"]
    assert cuda_builder.last_timings["gpu_foreground_seconds"] >= 0


def test_invalid_cameras_fail_before_gpu_processing(cuda_builder):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="camera names"):
        cuda_builder.from_frames({"top": image}, {"other": image})
    with pytest.raises(ValueError, match="match"):
        cuda_builder.from_frames({"top": image}, {"top": image[:4]})
