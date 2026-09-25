"""Check that the comparison cannot disguise cuts as missing detections."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def renderer():
    path = Path(__file__).resolve().parents[1] / "labs/20260917_clean2d_video_comparison/render.py"
    spec = importlib.util.spec_from_file_location("_clean2d_video_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs():
    frames = np.array([100, 100, 101, 101, 103, 103], np.int32)
    points = np.tile(np.array([[110.0, 100], [100, 100], [80, 100]]), (6, 1, 1))
    points[1::2] += [200, 200]
    base = dict(
        frame_idx=frames,
        points=points,
        global_rows=np.arange(6),
        clean_edge=np.array([False, False, True, True, False, False]),
    )
    sleap = dict(
        **base,
        labels=np.array([1, 2, 1, 2, 1, -1]),
        previous_row=np.array([-1, -1, 0, 1, 2, -1]),
        link_gap=np.array([0, 0, 1, 1, 2, 0]),
    )
    clean = dict(
        **base,
        labels=np.array([0, 1, 0, 1, 2, 3]),
        previous_row=np.array([-1, -1, 0, 1, -1, -1]),
        link_gap=np.array([0, 0, 1, 1, 0, 0]),
    )
    return {"SLEAP": sleap, **{arm: copy.deepcopy(clean) for arm in ("P", "N", "F")}}


def test_same_points_including_missing_nodes_are_required(renderer):
    arms = inputs()
    for data in arms.values():
        data["points"][0, 0] = np.nan
    assert renderer.validate_shared(arms)["all_input_rows_preserved"]
    arms["F"]["points"][2, 1, 0] += 1
    with pytest.raises(ValueError, match="shared detection"):
        renderer.validate_shared(arms)


def test_previous_links_must_use_same_identity_and_native_gap(renderer):
    arms = inputs()
    arms["SLEAP"]["link_gap"][4] = 1
    with pytest.raises(ValueError, match="native frame"):
        renderer.validate_shared(arms)
    arms = inputs()
    arms["F"]["previous_row"][4] = 2
    with pytest.raises(ValueError, match="different IDs"):
        renderer.validate_shared(arms)


def test_trails_preserve_sleap_gaps_but_never_cross_clean_fragment_boundaries(renderer):
    arms = inputs()
    assert renderer.edge_segments(arms["SLEAP"], 4, 103) == [(2, 4, 2), (0, 2, 1)]
    assert renderer.edge_segments(arms["F"], 4, 103) == []
    assert renderer.edge_segments(arms["SLEAP"], 5, 103) == []


def test_every_detection_reaches_every_panel_even_if_unassigned_or_all_nodes_missing(
    renderer, monkeypatch
):
    arms = inputs()
    for data in arms.values():
        data["points"][5] = np.nan
    calls = []
    original = copy.deepcopy(arms)

    def observe(image, arrays, row, frame, arm, view, scale, offset, occupied):
        calls.append((arm, row))

    monkeypatch.setattr(renderer, "overlay_detection", observe)
    raw = np.zeros(renderer.SOURCE_SHAPE, np.uint8)
    frame = renderer.render_frame(raw, {"arms": arms}, 103, "synthetic", "top_cam", "test")
    assert frame.shape == renderer.CANVAS_SHAPE
    assert calls == [(arm, row) for arm in renderer.ARMS for row in (4, 5)]
    for arm in renderer.ARMS:
        for key in arms[arm]:
            np.testing.assert_array_equal(arms[arm][key], original[arm][key])
    # A native frame with no input rows must not invent observations.
    calls.clear()
    renderer.render_frame(raw, {"arms": arms}, 102, "synthetic", "top_cam", "test")
    assert not calls


def test_four_panels_receive_identical_background_pixels(renderer, monkeypatch):
    arms = inputs()
    monkeypatch.setattr(renderer, "overlay_detection", lambda *args: None)
    raw = np.random.default_rng(11).integers(0, 256, renderer.SOURCE_SHAPE, dtype=np.uint8)
    frame = renderer.render_frame(raw, {"arms": arms}, 101, "synthetic", "top_cam", "test")
    patches = [
        frame[
            renderer.HEADER : renderer.HEADER + renderer.WIDTH - 8,
            col * renderer.WIDTH + 4 : (col + 1) * renderer.WIDTH - 4,
        ]
        for col in range(4)
    ]
    for patch in patches[1:]:
        np.testing.assert_array_equal(patches[0], patch)


def test_nearby_identity_tags_are_kept_separate_and_inside_panel(renderer):
    image = np.zeros((632, 632, 3), np.uint8)
    occupied = []
    for name in ("ID2", "ID127 NEW", "UNASSIGNED"):
        renderer.tag(image, name, (610, 3), (255, 255, 255), occupied)
    for i, box in enumerate(occupied):
        assert 0 <= box[0] < box[2] < image.shape[1]
        assert 0 <= box[1] < box[3] < image.shape[0]
        for old in occupied[:i]:
            assert box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]


def test_render_jobs_exclude_controls_and_preserve_all_primary_anchor_frames(renderer):
    protocol = {
        "cases": [
            {
                "name": "cg20s",
                "start": 75270,
                "stop": 78590,
                "centers": [76235, 76930],
                "render": True,
            },
            {
                "name": "known_30735",
                "start": 30569,
                "stop": 30913,
                "centers": [30735],
                "render": True,
            },
            {"name": "control", "start": 1, "stop": 500, "centers": [100], "render": False},
        ]
    }
    jobs = renderer.make_jobs(protocol)
    assert len(jobs) == 12
    assert not any(job["case"] == "control" for job in jobs)
    for job in jobs:
        if job["center"] is not None:
            assert job["stop"] - job["start"] == 166
            assert job["start"] <= job["center"] < job["stop"]
