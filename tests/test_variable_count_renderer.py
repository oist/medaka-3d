"""Rendering cannot hide surplus detections or turn predictions into observations."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def renderer():
    path = Path(__file__).resolve().parents[1] / "labs/20260917_variable_count_tracklets/render.py"
    spec = importlib.util.spec_from_file_location("_variable_count_render_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample(renderer):
    frames = np.array([100] * 5 + [101] * 5 + [103], np.int64)
    points = np.tile(np.array([[110.0, 100], [100, 100], [80, 100]]), (11, 1, 1))
    for row in range(10):
        points[row] += [150 * (row % 5), 80 * (row % 5)]
    points[-1] = np.nan
    base = dict(
        frame_idx=frames,
        points=points,
        global_rows=np.arange(70, 81),
        labels=np.array([0, 1, 2, 3, 4] * 2 + [5]),
        clean_edge=np.array([False] * 5 + [True] * 5 + [False]),
        previous_row=np.array([-1] * 5 + [0, 1, 2, 3, 4] + [-1]),
    )
    return {method: copy.deepcopy(base) for method in renderer.METHODS}


def test_every_surplus_and_missing_node_row_reaches_each_panel(renderer, monkeypatch):
    arms = sample(renderer)
    calls = []
    monkeypatch.setattr(
        renderer,
        "overlay_detection",
        lambda image, arrays, row, frame, method, *rest: calls.append((method, row)),
    )
    raw = np.zeros(renderer.SOURCE_SHAPE, np.uint8)
    renderer.render_frame(raw, {"arms": arms}, 100, "test", "top_cam")
    assert calls == [(method, row) for method in renderer.ARMS for row in range(5)]
    calls.clear()
    renderer.render_frame(raw, {"arms": arms}, 103, "test", "top_cam")
    assert calls == [(method, 10) for method in renderer.ARMS]
    calls.clear()
    renderer.render_frame(raw, {"arms": arms}, 102, "test", "top_cam")
    assert not calls


def test_shared_points_and_unique_assignment_are_enforced(renderer):
    arms = sample(renderer)
    assert renderer.validate_shared(arms)["all_input_rows_preserved"]
    arms["P1"]["points"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="shared detection"):
        renderer.validate_shared(arms)
    arms = sample(renderer)
    arms["P1"]["labels"][1] = 0
    with pytest.raises(ValueError, match="two rows"):
        renderer.validate_shared(arms)


def test_three_panels_use_identical_raw_pixels(renderer, monkeypatch):
    monkeypatch.setattr(renderer, "overlay_detection", lambda *args: None)
    raw = np.random.default_rng(2).integers(0, 256, renderer.SOURCE_SHAPE, dtype=np.uint8)
    image = renderer.render_frame(raw, {"arms": sample(renderer)}, 100, "test", "top_cam")
    patches = [
        image[
            renderer.HEADER : renderer.HEADER + renderer.WIDTH,
            i * renderer.WIDTH : (i + 1) * renderer.WIDTH,
        ]
        for i in range(3)
    ]
    np.testing.assert_array_equal(patches[0], patches[1])
    np.testing.assert_array_equal(patches[0], patches[2])


def test_crop_and_job_scopes_keep_context_and_all_anchors(renderer):
    arrays = sample(renderer)["C27"]
    x, y, width, height = renderer.choose_crop(arrays, 100)
    assert 0 <= x <= 1200 - width and 0 <= y <= 1200 - height
    assert width == height < 1200
    protocol = {
        "cases": [
            dict(
                name="cg20s",
                start=75270,
                stop=78590,
                centers=[76235, 76791, 76930, 76988, 77415],
                render=True,
            ),
            *[
                dict(name=f"known_{n}", centers=[n], render=True)
                for n in (21232, 30735, 31479, 105541, 260434, 302762, 323819, 333619)
            ],
            dict(name="control", centers=[], render=False),
        ]
    }
    jobs = renderer.make_jobs(protocol)
    assert len(jobs) == 42
    assert sum(j["center"] is None for j in jobs) == 3
    for job in jobs:
        if job["center"] is not None:
            assert job["stop"] - job["start"] == 166
            assert job["start"] <= job["center"] < job["stop"]


def test_compact_diagnostics_keep_disagreement_and_adjacent_candidates(renderer, tmp_path):
    path = Path(renderer.__file__).with_name("build_report.py")
    spec = importlib.util.spec_from_file_location("_variable_count_report_test", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("render")
    sys.modules["render"] = renderer
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["render"]
        else:
            sys.modules["render"] = previous
    case = tmp_path / "example"
    case.mkdir()
    (case / "edges_P1_top_cam.json").write_text("[]", encoding="utf-8")
    candidates = [
        dict(age_frames=age, prediction_error_px=age, from_global_row=50 + age)
        for age in (8, 5, 2, 1)
    ]
    events = [
        dict(
            kind="directional_link_refused",
            global_row=91,
            frame=100,
            candidates=candidates,
            reason="open_endpoint_competition",
        ),
        dict(
            kind="directional_disagreement",
            from_global_row=90,
            to_global_row=91,
            frame=100,
            reason="forward_backward_disagreement",
            forward=dict(prediction_error_px=1.0),
            backward=None,
        ),
    ]
    (case / "events_P1_top_cam.json").write_text(json.dumps(events), encoding="utf-8")
    by_row, _, projected = module.selected_diagnostics(tmp_path, "example", "top_cam", 100, 101)
    assert set(by_row) == {"91"}
    first, second = by_row["91"]["events_first_4"]
    assert first["candidate_count"] == 4
    assert first["adjacent_candidates_first_2"][0]["age_frames"] == 1
    assert second["kind"] == "directional_disagreement"
    assert second["forward"]["prediction_error_px"] == 1.0
    assert projected[0]["old_endpoint_count"] == 3
