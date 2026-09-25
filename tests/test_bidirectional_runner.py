"""Review-time accounting and native-context boundary regressions for the lab."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs" / "20260916_bidirectional_reconnect"


@pytest.fixture(scope="module")
def runner():
    if not (LAB / "run.py").exists():
        pytest.skip("bidirectional reconnect experiment has been retired")
    originals = {name: sys.modules.get(name) for name in ("directions", "reconnect")}
    for name in ("directions", "reconnect", "run"):
        module_name = "_bidirectional_runner" if name == "run" else name
        spec = importlib.util.spec_from_file_location(module_name, LAB / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    yield module
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def straight(frames):
    initial = np.array([[0.0, 0, 0], [0, 20.0, 0], [0, 0, 30.0]])
    return initial + np.asarray(frames)[:, None, None] * np.array([0.2, 0, 0])


def test_review_duration_counts_overlapping_episode_frames_once(runner):
    frames = np.arange(1000, 1160)
    xyz = straight(frames)
    xyz[61:, 0, 2] += 8
    xyz[65:, 1:, 2] += 8
    diagnostic = runner.diagnose(xyz, frames)
    result = runner.method_summary(diagnostic, {"start": 1060, "stop": 1070}, "distance")
    # Padded intervals are [1059, 1064) and [1063, 1068): 9 distinct frames.
    # Two fish at the second edge increase slot-step count, not review time.
    assert result["episodes"] == 2
    assert result["slot_steps"] == 3
    assert result["native_frames"] == 2
    assert result["flagged_seconds"] == pytest.approx(2 / 166)
    assert result["inspection_seconds_with_padding"] == pytest.approx(9 / 166)


def test_unflagged_window_has_zero_review_duration(runner):
    frames = np.arange(1000, 1160)
    diagnostic = runner.diagnose(straight(frames), frames)
    result = runner.method_summary(diagnostic, {"start": 1060, "stop": 1070}, "bidirectional")
    assert result["episodes"] == result["native_frames"] == result["slot_steps"] == 0
    assert result["inspection_seconds_with_padding"] == 0


def test_fifty_step_reference_needs_53_prior_poses_at_first_core_edge(runner):
    first_core, stop_core = 1000, 1010
    config = runner.DirectionConfig(window=50)
    measurements = {}
    for context in (52, 53):
        frames = np.arange(first_core - context, stop_core + context)
        diagnostic = runner.diagnose(straight(frames), frames, config)
        first = int(np.flatnonzero(diagnostic["frames"] == first_core)[0])
        last = int(np.flatnonzero(diagnostic["frames"] == stop_core - 1)[0])
        measurements[context] = diagnostic["past_angle_deg"][first]
        # Future references already fit at both boundaries, exposing the old
        # asymmetry rather than assuming both sides were equally incomplete.
        assert np.isfinite(diagnostic["future_angle_deg"][[first, last]]).all()
        assert np.isfinite(diagnostic["past_angle_deg"][first + 1]).all()
    assert np.isnan(measurements[52]).all()
    np.testing.assert_allclose(measurements[53], 0, atol=1e-6)
