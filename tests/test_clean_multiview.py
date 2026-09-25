"""Clean anchors must be independent, conservative, and observation preserving."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS


@pytest.fixture(scope="module")
def mv():
    path = Path(__file__).resolve().parents[1] / "labs/20260917_clean_tracklets/multiview.py"
    if not path.exists():
        pytest.skip("clean tracklet lab retired")
    spec = importlib.util.spec_from_file_location("_clean_multiview", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Camera:
    def __init__(self, center):
        self.center = np.asarray(center, dtype=float)

    def rays(self, uv):
        assert np.isfinite(uv).all()
        xy = (np.asarray(uv) - [400, 300]) / 600
        direction = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        return np.broadcast_to(self.center, direction.shape).copy(), direction

    def project(self, xyz):
        assert np.isfinite(xyz).all()
        relative = xyz - self.center
        return 600 * relative[..., :2] / relative[..., 2, None] + [400, 300]


class Rig:
    def __init__(self):
        self.cameras = dict(
            zip(VIEWS, [Camera(c) for c in ([-40, 0, 0], [40, 0, 0], [0, 40, 0])], strict=True)
        )

    def camera(self, view):
        return self.cameras[view]


def scene(n=9):
    frames = np.arange(100, 100 + n)
    xyz = np.array([[48, 72, 90], [68, 60, 100], [78, 84, 110]], dtype=float)
    xyz = xyz[None] + np.arange(n)[:, None, None] * np.array([0.12, 0.05, 0.02])
    pose = xyz[:, :, None] + np.array([[-5, 0, 0], [0, 0, 0], [10, 0, 0]])
    rig = Rig()
    views = {}
    for view in VIEWS:
        views[view] = dict(
            frame_idx=np.repeat(frames, 3),
            points=rig.camera(view).project(pose).reshape(-1, 3, 2),
            labels=np.tile(np.arange(3), n),
            clean_edge=np.repeat(np.arange(n) > 0, 3),
        )
    return frames, pose, rig, views


def split(views, view, frame, label=1, new_label=17):
    data = views[view]
    mask = (data["frame_idx"] >= frame) & (data["labels"] == label)
    data["labels"][mask] = new_label
    data["clean_edge"][mask & (data["frame_idx"] == frame)] = False


def test_two_clean_views_bridge_third_fragment_without_rewriting_initial_edges(mv):
    frames, pose, rig, views = scene()
    split(views, VIEWS[2], 104)
    original = {v: {k: a.copy() for k, a in data.items()} for v, data in views.items()}
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-9)
    np.testing.assert_array_equal(
        result["source"], np.tile(np.arange(3)[None, :, None], (len(frames), 1, 3))
    )
    bridges = [e for e in result["bridges"] if e["accepted"]]
    assert len(bridges) == 1
    bridge = bridges[0]
    assert (bridge["from_frame"], bridge["to_frame"]) == (103, 104)
    assert bridge["supporting_labels"] == {VIEWS[0]: 1, VIEWS[1]: 1}
    assert max(bridge["endpoint_projection_error_px"]) < 1e-8
    for view in VIEWS:
        for key in views[view]:
            np.testing.assert_array_equal(views[view][key], original[view][key])
        np.testing.assert_array_equal(result["labels_by_view"][view], original[view]["labels"])
        np.testing.assert_array_equal(
            result["clean_edge_by_view"][view], original[view]["clean_edge"]
        )
    assert np.unique(result["joined_labels_by_view"][VIEWS[2]]).size == 3
    assert not result["identity_review"].any()


def test_all_views_cut_cannot_certify_bridge_from_smooth_3d_alone(mv):
    frames, pose, rig, views = scene()
    for view in VIEWS:
        split(views, view, 104)
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-9)
    assert not any(e["accepted"] for e in result["bridges"])
    assert {e["reason"] for e in result["bridges"]} == {"two_clean_tracklets_not_contiguous"}


def test_geometry_ambiguous_all_views_abstain_without_fabricated_detections(mv):
    frames, _, rig, views = scene(4)
    for data in views.values():
        p = data["points"].reshape(len(frames), 3, 3, 2)
        p[:, 1:] = p[:, :1]
    result = mv.reconstruct(frames, views, rig)
    assert np.isnan(result["xyz"]).all()
    assert (result["source"] == -1).all()
    assert result["identity_review"].all()
    assert not any(e["accepted"] for e in result["bridges"])
    assert any(e["reason"] == "correspondence_margin" for e in result["events"])
    assert all(len(result["labels_by_view"][v]) == len(views[v]["labels"]) for v in VIEWS)


def test_detection_row_shuffle_and_arbitrary_clean_label_names_do_not_change_geometry(mv):
    frames, pose, rig, views = scene()
    split(views, VIEWS[2], 104)
    baseline = mv.reconstruct(frames, views, rig)
    rng = np.random.default_rng(129)
    for data in views.values():
        data["labels"] = data["labels"] * 101 + 71
        order = rng.permutation(len(data["frame_idx"]))
        for key in data:
            data[key] = data[key][order]
    changed = mv.reconstruct(frames, views, rig.cameras)
    np.testing.assert_allclose(changed["pose"], pose, atol=1e-9)
    np.testing.assert_allclose(changed["uv"], baseline["uv"], atol=1e-9)
    assert sum(e["accepted"] for e in changed["bridges"]) == 1
    for ti, frame in enumerate(frames):
        for vi, view in enumerate(VIEWS):
            raw = views[view]["points"][views[view]["frame_idx"] == frame]
            np.testing.assert_array_equal(
                raw[changed["source"][ti, :, vi]], changed["uv"][ti, :, vi]
            )


def test_missing_third_view_keeps_source_missing_but_projection_separate(mv):
    frames, pose, rig, views = scene()
    data = views[VIEWS[2]]
    split(views, VIEWS[2], 105)
    remove = np.isin(data["frame_idx"], [103, 104]) & (data["labels"] == 1)
    for key in data:
        data[key] = data[key][~remove]
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-9)
    assert (result["source"][3:5, 1, 2] == -1).all()
    assert np.isnan(result["uv"][3:5, 1, 2]).all()
    assert np.isfinite(result["projected_uv"][3:5, 1, 2]).all()
    assert sum(e["accepted"] for e in result["bridges"]) == 1


def test_duplicate_third_view_competitor_blocks_join_and_is_retained(mv):
    frames, _, rig, views = scene()
    split(views, VIEWS[2], 104)
    data = views[VIEWS[2]]
    row = np.flatnonzero((data["frame_idx"] == 104) & (data["labels"] == 17))[0]
    for key in ["frame_idx", "points"]:
        data[key] = np.concatenate([data[key], data[key][row : row + 1]])
    data["labels"] = np.r_[data["labels"], 42]
    data["clean_edge"] = np.r_[data["clean_edge"], False]
    result = mv.reconstruct(frames, views, rig)
    assert not any(e["accepted"] for e in result["bridges"])
    assert len(result["labels_by_view"][VIEWS[2]]) == 28
    assert 42 in result["joined_labels_by_view"][VIEWS[2]]


def test_long_unanchored_gap_sets_persistent_identity_review(mv):
    frames, _, rig, views = scene(12)
    for view, data in views.items():
        for label in range(3):
            split(views, view, 108, label=label, new_label=label + 10)
        keep = (data["frame_idx"] < 101) | (data["frame_idx"] >= 108)
        for key in data:
            data[key] = data[key][keep]
    result = mv.reconstruct(frames, views, rig)
    assert np.isnan(result["xyz"][1:8]).all()
    assert result["identity_review"][8:].all()
    assert any(
        e["reason"] == "long_gap_unanchored_identity_reacquisition" for e in result["events"]
    )


def test_noncontiguous_clean_component_fails_loudly(mv):
    frames, _, rig, views = scene()
    views[VIEWS[0]]["clean_edge"][9] = False
    with pytest.raises(ValueError, match="contiguous component"):
        mv.reconstruct(frames, views, rig)


def test_twenty_frame_third_view_gap_uses_continuously_observed_pair(mv):
    frames, pose, rig, views = scene(27)
    data = views[VIEWS[2]]
    split(views, VIEWS[2], 123)
    keep = ~((data["frame_idx"] >= 103) & (data["frame_idx"] < 123) & (data["labels"] == 1))
    for key in data:
        data[key] = data[key][keep]
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-9)
    bridges = [e for e in result["bridges"] if e["accepted"]]
    assert len(bridges) == 1
    assert (bridges[0]["from_frame"], bridges[0]["to_frame"]) == (102, 123)
    assert (result["source"][3:23, 1, 2] == -1).all()
    assert not result["identity_review"].any()
    for view in VIEWS:
        for frame in frames:
            labels = result["joined_labels_by_view"][view][views[view]["frame_idx"] == frame]
            assert len(np.unique(labels)) == len(labels)


def test_two_view_bootstrap_assigns_three_distinct_animals(mv):
    frames, pose, rig, views = scene(4)
    for key, value in views[VIEWS[2]].items():
        views[VIEWS[2]][key] = value[:0]
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-9)
    assert (result["source"][:, :, 2] == -1).all()
    assert not result["identity_review"].any()


def test_one_view_only_never_produces_a_three_dimensional_observation(mv):
    frames, _, rig, views = scene(4)
    for view in VIEWS[1:]:
        for key, value in views[view].items():
            views[view][key] = value[:0]
    result = mv.reconstruct(frames, views, rig)
    assert np.isnan(result["pose"]).all()
    assert (result["source"] == -1).all()
    assert len(result["labels_by_view"][VIEWS[0]]) == 12


def test_perfect_intersections_outside_water_are_not_observations(mv):
    frames, pose, rig, views = scene(4)
    pose[..., 2] += 200
    for view in VIEWS:
        views[view]["points"] = rig.camera(view).project(pose).reshape(-1, 3, 2)
    result = mv.reconstruct(frames, views, rig)
    assert np.isnan(result["pose"]).all()
    assert (result["source"] == -1).all()


def test_all_nan_third_view_row_is_retained_but_is_not_observation_support(mv):
    frames, pose, rig, views = scene(4)
    views[VIEWS[2]]["points"][views[VIEWS[2]]["labels"] == 1] = np.nan
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"], pose, atol=1e-9)
    assert (result["source"][:, 1, 2] == -1).all()
    assert len(result["labels_by_view"][VIEWS[2]]) == 12
    assert np.isnan(result["uv"][:, 1, 2]).all()


def test_two_clean_views_preserve_slot_when_third_clean_anchor_conflicts(mv):
    frames, pose, rig, views = scene(5)
    points = views[VIEWS[2]]["points"].reshape(len(frames), 3, 3, 2)
    points[2:, [0, 1]] = points[2:, [1, 0]]
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["xyz"], pose[:, :, 1], atol=1e-9)
    assert (result["source"][2:, :2, 2] == -1).all()
    assert any(e["reason"] == "contradicting_clean_anchors" for e in result["events"])
    assert not result["identity_review"].any()


def test_bad_tail_geometry_is_masked_without_changing_identity_or_raw_pixels(mv):
    frames, pose, rig, views = scene(4)
    points = views[VIEWS[2]]["points"].reshape(len(frames), 3, 3, 2)
    points[:, 1, 2, 1] += 120
    result = mv.reconstruct(frames, views, rig)
    np.testing.assert_allclose(result["pose"][:, :, :2], pose[:, :, :2], atol=1e-9)
    assert np.isnan(result["pose"][:, 1, 2]).all()
    assert (result["raw_residual_mm"][:, 1, 2] > mv.MultiViewConfig().ray_gate_mm).all()
    assert np.isnan(result["residual_mm"][:, 1, 2]).all()
    np.testing.assert_array_equal(result["uv"][:, 1, 2, 2], points[:, 1, 2])
    assert result["metadata"]["tail_high_ray_residual_fish_frames"] == len(frames)
