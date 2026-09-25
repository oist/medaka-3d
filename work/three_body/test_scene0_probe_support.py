import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import audit_scene0_probe_support as a


class Camera:
    def project(self, pose):
        return np.asarray(pose)[..., :2]

    def rays(self, points):
        origin = np.concatenate((points, np.zeros((*points.shape[:-1], 1))), axis=-1)
        direction = np.zeros_like(origin)
        direction[..., 2] = 1
        return origin, direction


class Rig:
    def camera(self, view):
        return Camera()


def fixture():
    pose = np.ones((3, 3, 3))
    pose[:, :, 0] = np.arange(3)[:, None] * 20 + 30
    sources = np.full((3, 3, 3), -1, np.int32)
    sources[:, :2, 0] = np.arange(3)[:, None]
    observed = np.zeros((3, 3), bool)
    observed[:, 0] = True
    points = {v: pose[..., :2].copy() for v in a.m.VIEWS}
    return pose, sources, observed, np.ones(3, bool), points, Rig()


def test_finite_projection_does_not_create_third_view_support_or_clear_review():
    rows, projection, gate, error = a.inspect_state(*fixture())
    assert np.isfinite(projection).all()
    assert gate.sum(1).tolist() == [2, 2, 2]
    assert np.isnan(error[:, 2]).all()
    assert all(row["original_identity_review"] for row in rows)


def test_single_view_and_false_saved_observed_are_rejected():
    inputs = list(fixture())
    inputs[1][0, 1, 0] = -1
    with pytest.raises(ValueError, match="observed flags"):
        a.inspect_state(*inputs)
    inputs[2][0, 0] = False
    rows, _, gate, _ = a.inspect_state(*inputs)
    assert gate[0].sum() == 1 and not rows[0]["mouth_observed"]


def test_shared_saved_sleap_row_is_not_valid_independent_support():
    inputs = list(fixture())
    inputs[1][1, 0, 0] = 0
    with pytest.raises(ValueError, match="shared SLEAP"):
        a.inspect_state(*inputs)


def test_native_eligible_intervals_are_half_open_and_do_not_fill_missing_frames():
    assert a.intervals([]) == []
    assert a.intervals([79237, 79238, 79316, 79403, 79404, 79405]) == [
        [79237, 79239],
        [79316, 79317],
        [79403, 79406],
    ]
