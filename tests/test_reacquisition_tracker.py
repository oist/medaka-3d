"""Recovery must use new observations, preserve exclusivity and admit ambiguity."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"))

from reacquisition import (
    ReacquisitionLinker,
    ReacquisitionTracker,
    RecoveryConfig,
    node_candidates,
)
from tracker import FrameInput, TrackerConfig


@pytest.fixture
def recovery_rig(rig):
    return replace(rig, cameras=tuple(
        replace(c, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if c.view == "top_cam" else 2)
        for c in rig.cameras))


@pytest.fixture
def bodies():
    centers = np.array([[45., 50., 40.], [130., 90., 90.], [75., 150., 150.]])
    nodes = np.array([[5., 0., 0.], [0., 0., 0.], [-12., 0., 0.]])
    return centers[:, None] + nodes[None]


def frame_for(rig, poses, frame, hidden=(), permute=False):
    uv = {c.view: c.project(poses) for c in rig.cameras}
    for v, points in enumerate(uv.values()):
        points[list(hidden)] = np.nan
        if permute:
            uv[rig.views[v]] = np.roll(points, (frame + v) % 3, axis=0)
    return FrameInput(frame, uv)


def config(**kwargs):
    return RecoveryConfig(local=TrackerConfig(beam_width=2, proposals_per_parent=2), **kwargs)


def test_ray_candidates_recover_two_views_when_third_is_false(recovery_rig, bodies):
    frame = frame_for(recovery_rig, bodies, 0)
    frame.detections["top_cam"][0] += [300., -200.]
    candidates = node_candidates(recovery_rig, frame, config())
    for node, pool in enumerate(candidates):
        nearest = min(pool, key=lambda c: np.linalg.norm(c.xyz - bodies[0, node]))
        np.testing.assert_allclose(nearest.xyz, bodies[0, node], atol=1e-7)
        assert nearest.source[0] == -1
        assert (nearest.source[1:] >= 0).all()


def test_remote_reappearance_recovers_geometry_but_does_not_certify_identity(
    recovery_rig, bodies,
):
    moved = bodies.copy()
    moved[0] += [35., 0., 0.]
    frames = [frame_for(recovery_rig, bodies, 0)]
    frames += [frame_for(recovery_rig, bodies, f, hidden=[0], permute=True)
               for f in range(1, 7)]
    frames += [frame_for(recovery_rig, moved, f, permute=True) for f in range(7, 19)]
    result = ReacquisitionTracker(recovery_rig, bodies, config()).track(frames)
    assert result.diagnostics["initializations"] == 1
    assert result.diagnostics["selected_recovery_animal_frames"] >= 1
    assert result.observed[-1].all()
    assert result.identity_review[-1, 0]
    assert np.isnan(result.accepted_xyz[-1, 0]).all()
    assert np.isfinite(result.accepted_xyz[-1, 1:]).all()
    assert np.linalg.norm(result.xyz[-1, 0] - moved[0, 1]) < 1.0
    assert np.isnan(result.xyz[1:7, 0]).all()
    # Per-frame detection rows are not persistent IDs and must never be
    # reserved using the previous frame's ordering.
    for source in result.node_source:
        for view in range(3):
            for node in range(3):
                rows = source[:, view, node]
                rows = rows[rows >= 0]
                assert len(rows) == len(set(rows))
    assert result.observed[:, 1:].all()


def test_two_lost_fish_are_not_certified_by_three_frames_of_reobservation(
    recovery_rig, bodies,
):
    moved = bodies.copy()
    moved[0] += [25., 0., 0.]
    moved[1] += [-25., 0., 0.]
    frames = [frame_for(recovery_rig, bodies, 0)]
    frames += [frame_for(recovery_rig, bodies, f, hidden=[0, 1]) for f in range(1, 7)]
    frames += [frame_for(recovery_rig, moved, f, permute=True) for f in range(7, 20)]
    result = ReacquisitionTracker(recovery_rig, bodies, config()).track(frames)
    assert result.observed[-1, :2, 1].all()
    assert result.identity_review[-1, :2].all()
    assert np.isnan(result.accepted_xyz[-1, :2]).all()
    assert not result.review[-1, 2]


def test_empty_cameras_keep_missing_positions_missing(recovery_rig, bodies):
    frames = [frame_for(recovery_rig, bodies, 0)]
    frames += [FrameInput(f, {v: np.empty((0, 3, 2)) for v in recovery_rig.views})
               for f in range(1, 8)]
    result = ReacquisitionTracker(recovery_rig, bodies, config()).track(frames)
    assert not result.observed[1:].any()
    assert np.isnan(result.accepted_xyz[1:]).all()
    assert np.isfinite(result.pose_xyz).all()
    assert result.diagnostics["selected_recovery_animal_frames"] == 0


def test_nominal_floor_and_explicit_gauge_tolerance_have_distinct_candidate_domains(
    recovery_rig, bodies,
):
    low = bodies.copy()
    low[0, :, 2] = -3.
    frame = frame_for(recovery_rig, low, 0)
    nominal = node_candidates(recovery_rig, frame, config())
    cfg = config()
    tolerance = replace(cfg, local=replace(cfg.local, lower_mm=(0., 0., -6.)))
    expanded = node_candidates(recovery_rig, frame, tolerance)
    assert all(np.linalg.norm(c.xyz - low[0, 1]) > 1 for c in nominal[1])
    assert any(np.linalg.norm(c.xyz - low[0, 1]) < 1e-6 for c in expanded[1])


def test_recovery_linker_ignores_poisoned_future_baseline_items(recovery_rig, bodies):
    from medaka3d.types import Instance3D

    frames = [frame_for(recovery_rig, bodies, f) for f in range(2)]
    seeds = [Instance3D(0, pose, i) for i, pose in enumerate(bodies)]
    bad_future = [Instance3D(1, np.full((3, 3), np.nan), 0)]
    clean = ReacquisitionLinker(recovery_rig, frames, config()).link(seeds)
    dirty = ReacquisitionLinker(recovery_rig, frames, config()).link([*seeds, *bad_future])
    np.testing.assert_array_equal(clean, dirty)
