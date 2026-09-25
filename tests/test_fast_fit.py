"""A compressed Jacobian must preserve the frozen residual and fit decisions."""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

_LAB = Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"
sys.path.insert(0, str(_LAB))

import tracker as frozen  # noqa: E402
from fast_fit import FastFitTracker  # noqa: E402
from observations import SceneEvidence  # noqa: E402


@pytest.mark.parametrize("case", ["observed", "missing", "bounds", "image", "missing_image"])
def test_compressed_jacobian_and_fit_match_frozen_objective(rig, monkeypatch, case):
    rig = replace(rig, cameras=tuple(
        replace(cam, index_air=1.0003, index_glass=1.52, index_water=1.3325,
                layer_count=1 if cam.view == "top_cam" else 2,
                distortion=(-0.222, 0.204, 0.0044, 0.0019, -0.112))
        for cam in rig.cameras))
    centers = np.array([[45., 50., 40.], [130., 90., 90.], [75., 150., 150.]])
    body = np.array([[5., 0., 0.], [0., 0., 0.], [-12., 0., 0.]])
    prediction = centers[:, None] + body[None]
    if case == "bounds":
        prediction[1, :, 2] = 185 - 2e-5
        prediction[2, :, 2] = 2e-5
    target = np.clip(prediction + np.array([0.2, -0.3, 0.1]), 1e-5, 185 - 1e-5)
    detections = {cam.view: cam.project(target) for cam in rig.cameras}
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    if case == "missing":
        plan[0, 0, 2] = -1
        plan[1, :, 1] = -1
        detections["side1_cam"][2, 0] = np.nan
    evidence = None
    if "image" in case:
        images, backgrounds = {}, {}
        for cam in rig.cameras:
            image = np.zeros((1200, 1200, 3), np.uint8)
            for fish in cam.project(target):
                cv2.polylines(image, [np.rint(fish).astype(np.int32)], False, (230,) * 3, 3)
            images[cam.view] = image
            backgrounds[cam.view] = np.zeros_like(image)
        evidence = SceneEvidence.from_frames(images, backgrounds)
    if case == "missing_image":
        plan[:] = -1
    frame = frozen.FrameInput(8, detections, evidence)
    config = frozen.TrackerConfig(upper_mm=(185., 185., 185.), max_nfev=12)
    slow = frozen.GenerativeTracker(rig, prediction, config)
    fast = FastFitTracker(rig, prediction, config)
    problem = fast.fit_problem(prediction, plan, frame, 1)
    solver = frozen.least_squares
    captured = []

    def capture(residual, initial, jac, **kwargs):
        for flat in (initial, np.clip(initial + 1e-4, problem.lower, problem.upper)):
            np.testing.assert_array_equal(problem.residual(flat), residual(flat))
            expected, actual = jac(flat), problem.jacobian(flat)
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
            # Explicitly check the structural premise against the original Jacobian.
            other_fish = np.arange(27)[None] // 9 != problem.row_fish[:, None]
            np.testing.assert_allclose(expected[other_fish], 0, atol=1e-9, rtol=0)
        captured.append(True)
        return solver(residual, initial, jac=jac, **kwargs)

    monkeypatch.setattr(frozen, "least_squares", capture)
    expected = slow._fit(prediction, plan, frame, 1)
    actual = fast._fit(prediction, plan, frame, 1)
    assert captured == [True]
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_allclose(actual[3], expected[3], atol=1e-8, rtol=1e-9)
    assert actual[4:] == expected[4:]


def test_only_ten_batches_are_projected_for_compressed_jacobian():
    from fast_fit import FitProblem

    batches = []
    rows = np.repeat(np.arange(3), 9)

    def residual_batch(flat):
        batches.append(len(flat))
        return np.square(flat)

    initial = np.linspace(-1., 1., 27)
    problem = FitProblem(residual_batch, rows, None, None, np.full(27, -2.),
                         np.full(27, 2.), initial)
    actual = problem.jacobian(initial)
    np.testing.assert_allclose(actual, np.diag(2 * initial + 1e-3), atol=1e-12)
    assert batches == [10]


def test_grouped_image_projection_preserves_previous_pose_and_invalid_points(rig):
    from batched_image_score import image_point_groups, project_image_groups

    pose = np.array([
        [[45., 50., 40.], [40., 50., 40.], [28., 50., 40.]],
        [[80., 80., 80.], [75., 80., 80.], [63., 80., 80.]],
        [[95., 100., 100.], [90., 100., 100.], [78., 100., 100.]],
    ])
    previous = pose + np.array([0.5, -1., 2.])
    pose[1, 2] = np.nan
    pose[2, :, 2] = 190.  # Before the top water interface: invalid only in that camera.
    points, body_shape, sizes = image_point_groups(pose, previous)
    for camera in rig.cameras:
        actual = project_image_groups(camera, points, body_shape, sizes)
        expected = (camera.project(pose), camera.project(frozen._body_samples(pose)),
                    camera.project(previous))
        for grouped, separate in zip(actual, expected, strict=True):
            np.testing.assert_allclose(grouped, separate, atol=1e-9, rtol=0, equal_nan=True)
