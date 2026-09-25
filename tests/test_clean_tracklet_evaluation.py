"""The new comparison must not reward invented points or detector row reuse."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "labs/20260917_clean_tracklets/evaluate.py"
SPEC = importlib.util.spec_from_file_location("clean_tracklet_evaluation", PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def fixture():
    frames = np.arange(2)
    views = {
        v: {
            "frame_idx": np.repeat(frames, 3),
            "points": np.arange(36).reshape(6, 3, 2).astype(float),
        }
        for v in ("top_cam", "side1_cam", "side2_cam")
    }
    source = np.broadcast_to(np.arange(3)[None, :, None], (2, 3, 3)).copy()
    uv = np.stack([views[v]["points"].reshape(2, 3, 3, 2) for v in views], axis=2)
    stage = {"xyz": np.zeros((2, 3, 3)), "source": source, "uv": uv}
    return frames, views, stage


def test_absence_cannot_disguise_as_lower_error():
    stage = {"xyz": np.full((4, 3, 3), np.nan), "source": np.full((4, 3, 3), -1)}
    stats = evaluation.geometry_metrics(stage, np.ones(4, bool))
    assert stats["missing_fish_frames"] == 12
    assert stats["observed_native_steps"] == 0
    assert stats["step_mm"]["maximum"] is None


def test_reused_observation_fails_source_audit():
    frames, views, stage = fixture()
    stage["source"][1, 2, 0] = 1
    with pytest.raises(ValueError, match="multiple fish"):
        evaluation.validate_sources(stage, frames, views)


def test_projected_missing_observation_is_not_measured():
    frames, views, stage = fixture()
    stage["source"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="Predicted pixels"):
        evaluation.validate_sources(stage, frames, views)
    stage["uv"][0, 0, 0] = np.nan
    assert evaluation.validate_sources(stage, frames, views)["status"] == "pass"


def test_display_alignment_does_not_change_measurements():
    frames = np.arange(25)
    xyz = np.broadcast_to(np.array([[0, 0, 0], [20, 0, 0], [40, 0, 0]])[None], (25, 3, 3)).copy()
    output = {"xyz": xyz[:, [1, 2, 0]].copy()}
    before = output["xyz"].copy()
    aligned = evaluation.prefix_permutation(output, {"xyz": xyz}, frames, 20)
    assert aligned["permutation"] == [2, 0, 1]
    np.testing.assert_array_equal(output["xyz"], before)
