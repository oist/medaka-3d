"""Behavioral checks for native-frame bidirectional jump diagnostics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def directions():
    path = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "20260916_bidirectional_reconnect"
        / "directions.py"
    )
    spec = importlib.util.spec_from_file_location("_bidirectional_directions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def straight(n=180):
    initial = np.array([[0.0, 0, 0], [0, 20, 0], [0, 0, 30]])
    return initial + np.arange(n)[:, None, None] * np.array([0.1, 0, 0])


def test_native_one_frame_excursion_survives_display_decimation(directions):
    xyz = straight()
    xyz[61, 2, 1] += 8
    original = xyz.copy()
    result = directions.diagnose(xyz, np.arange(1000, 1000 + len(xyz)))
    hit_step, hit_slot = np.nonzero(result["distance"])
    assert list(zip(result["frames"][hit_step], hit_slot, strict=True)) == [(1061, 2), (1062, 2)]
    assert np.all(result["bidirectional"][hit_step, hit_slot])
    # A 30 fps display can skip the entire out-and-back at native 166 fps.
    display_rows = np.floor(np.arange(0, len(xyz), 166 / 30)).astype(int)
    assert 61 not in display_rows
    assert np.max(np.linalg.norm(np.diff(xyz[display_rows], axis=0), axis=-1)) < 4
    np.testing.assert_array_equal(xyz, original)


def test_future_only_candidate_and_time_reversal_sign_convention(directions):
    xyz = straight(140)
    displacement = np.zeros((139, 3))
    displacement[:61] = [0.2, 0, 0]
    displacement[60] = [1.5, 0, 0]
    displacement[61:] = [0, 0.2, 0]
    xyz[1:, 0] = np.cumsum(displacement, axis=0)
    result = directions.diagnose(xyz, np.arange(4000, 4140))
    edge = 60
    assert result["frames"][edge] == 4061
    assert not result["distance"][edge, 0]
    assert not result["forward"][edge, 0]
    assert result["bidirectional"][edge, 0]
    assert result["past_angle_deg"][edge, 0] == pytest.approx(0)
    assert result["future_angle_deg"][edge, 0] == pytest.approx(90)
    np.testing.assert_allclose(result["future_velocity"][edge, 0], [0, 0.2, 0])

    reverse = directions.diagnose(xyz[::-1], np.arange(4000, 4140))
    mirrored_edge = len(xyz) - 2 - edge
    assert reverse["forward"][mirrored_edge, 0]
    assert reverse["past_angle_deg"][mirrored_edge, 0] == pytest.approx(90)
    assert reverse["future_angle_deg"][mirrored_edge, 0] == pytest.approx(0)
    np.testing.assert_allclose(
        result["past_velocity"], -reverse["future_velocity"][::-1], equal_nan=True
    )
    np.testing.assert_allclose(
        result["future_angle_deg"], reverse["past_angle_deg"][::-1], equal_nan=True
    )
    np.testing.assert_array_equal(result["bidirectional"], reverse["bidirectional"][::-1])


@pytest.mark.parametrize("excursion", [1.5, 5.0])
def test_stationary_flanks_have_no_heading_but_large_distance_still_flags(directions, excursion):
    xyz = np.zeros((140, 3, 3))
    xyz[70, 0, 1] = excursion
    result = directions.diagnose(xyz, np.arange(len(xyz)))
    assert np.isnan(result["past_angle_deg"]).all()
    assert np.isnan(result["future_angle_deg"]).all()
    expected = np.zeros((139, 3), dtype=bool)
    if excursion > 4:
        expected[[69, 70], 0] = True
    for method in ("distance", "forward", "bidirectional"):
        np.testing.assert_array_equal(result[method], expected)


def test_missing_positions_never_create_flags_or_bridge_missing_steps(directions):
    xyz = straight(140)
    xyz[40:61, 1] = np.nan
    xyz[61:, 1, 1] += 30
    xyz[:, 2] = np.nan
    result = directions.diagnose(xyz, np.arange(len(xyz)))
    assert np.isnan(result["step_mm"][39:61, 1]).all()
    for method in ("distance", "forward", "bidirectional"):
        assert not result[method].any()
    assert np.isnan(result["past_angle_deg"][:, 2]).all()
    assert np.isnan(result["future_angle_deg"][:, 2]).all()


def test_episode_grouping_keeps_destination_frames_slots_and_half_open_bounds(directions):
    xyz = straight(150)
    xyz[61, 0, 1] += 8
    xyz[64, 2, 1] += 8
    xyz[70, 1, 2] += 8
    result = directions.diagnose(xyz, np.arange(1000, 1150))
    groups = directions.episodes(result, 1061, 1071, method="distance", merge_frames=3, pad=2)
    assert len(groups) == 2
    assert groups[0]["first_edge_frame"] == 1061
    assert groups[0]["last_edge_frame"] == 1065
    assert groups[0]["start_frame"] == 1059
    assert groups[0]["stop_frame_exclusive"] == 1068
    assert [(e["frame"], e["slot"]) for e in groups[0]["edges"]] == [
        (1061, 0),
        (1062, 0),
        (1064, 2),
        (1065, 2),
    ]
    assert [(e["frame"], e["slot"]) for e in groups[1]["edges"]] == [(1070, 1)]
    assert groups[1]["index"] == 1
    # The exit at 1071 is excluded from this core even though padding extends past it.
    assert groups[1]["stop_frame_exclusive"] == 1073
    json.dumps(groups, allow_nan=False)


def test_native_edge_has_distance_without_fabricated_past_reference(directions):
    xyz = straight(60)
    xyz[0, 1, 2] += 8
    result = directions.diagnose(xyz, np.arange(500, 560))
    assert result["frames"][0] == 501
    assert result["distance"][0, 1]
    assert np.isnan(result["past_angle_deg"][0, 1])
    assert result["future_angle_deg"][0, 1] > 80
    group = directions.episodes(result, 500, 502, method="distance")[0]
    assert group["edges"][0]["past_angle_deg"] is None
    assert group["edges"][0]["frame"] == 501
    json.dumps(group, allow_nan=False)


def test_frame_gaps_are_rejected_instead_of_treated_as_native_steps(directions):
    xyz = straight(60)
    frames = np.arange(60)
    frames[30:] += 1
    with pytest.raises(ValueError, match="contiguous native frames"):
        directions.diagnose(xyz, frames)
