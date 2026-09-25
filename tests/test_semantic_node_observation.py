"""Synthetic gate and native-contract tests; no real inputs or model execution."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import semantic_node_observation as m


class Camera:
    def __init__(self, axis=0, error=0.0):
        self.direction = np.eye(3)[axis]
        self.error = error
        self.calls = 0

    def rays(self, xy):
        self.calls += 1
        direction = np.broadcast_to(self.direction, (len(xy), 3)).copy()
        return -direction, direction

    def project(self, xyz):
        return np.asarray([self.error, 0.0])


class Rig:
    views = m.VIEWS
    world_units = "mm"

    def __init__(self):
        self.cameras = [Camera(i) for i in range(3)]

    def camera(self, view):
        return self.cameras[self.views.index(view)]


def inputs():
    return (
        np.zeros((3, 3, 3, 2), np.float32),
        np.ones((3, 3, 3), bool),
        np.ones((3, 3, 3), bool),
        Rig(),
        np.zeros((3, 3, 3)),
    )


def build(values=None, **clock):
    args = dict(initial_frame_idx=100, previous_frame_idx=100, frame_idx=101, release_horizon=115)
    args.update(clock)
    return m.build_observation(*(inputs() if values is None else values), **args)


def measured(monkeypatch, *, xyz=None, rms=0.0, depth=1.0, error=0.0):
    """Isolate exact acceptance boundaries, independently of solve rounding.

    Physical camera/triangulate behavior is checked in the separate geometry tests.
    This stub supplies a finite solver result so an exact threshold can be tested.
    """
    xyz = np.zeros(3) if xyz is None else np.asarray(xyz, float)
    direction = np.eye(3)[:2]
    origins = xyz - depth * direction
    monkeypatch.setattr(m, "triangulate", lambda origins, directions: (xyz.copy(), np.asarray(rms)))
    arrays = m.allocate()
    cameras = [Camera(error=error) for _ in range(3)]
    m.evaluate_subset(
        arrays,
        (0, 0, 1),
        np.asarray([0, 1]),
        origins,
        direction,
        np.zeros((3, 2)),
        np.zeros(3),
        cameras,
    )
    return arrays, (0, 0, 1)


@pytest.mark.parametrize(
    "field,at,after,flag",
    [
        ("rms", 1.0, np.nextafter(1.0, np.inf), m.SubsetRejection.RAY_RMS),
        ("error", 3.0, np.nextafter(3.0, np.inf), m.SubsetRejection.REPROJECTION),
        ("xyz", [6.0, 0, 0], [np.nextafter(6.0, np.inf), 0, 0], m.SubsetRejection.NODE_STEP),
        ("depth", 0.0, -np.finfo(float).eps, m.SubsetRejection.BEHIND_RAY),
    ],
)
def test_exact_geometry_limits_accept_equality_reject_next_value(
    monkeypatch, field, at, after, flag
):
    a, k = measured(monkeypatch, **{field: at})
    assert a["admitted"][k] and a["rejection_bits"][k] == 0
    a, k = measured(monkeypatch, **{field: after})
    assert not a["admitted"][k]
    assert int(a["rejection_bits"][k]) & int(flag)


def test_all_measurable_failures_retained_after_rms_failure(monkeypatch):
    a, k = measured(monkeypatch, xyz=[7, 0, 0], rms=2.0, depth=-1.0, error=4.0)
    expected = (
        m.SubsetRejection.RAY_RMS
        | m.SubsetRejection.BEHIND_RAY
        | m.SubsetRejection.NODE_STEP
        | m.SubsetRejection.REPROJECTION
    )
    assert int(a["rejection_bits"][k]) == int(expected)
    assert a["ray_rms_mm"][k] == 2.0 and a["step_mm"][k] == 7.0
    assert a["forward_depth_mm"][k][:2].tolist() == [-1.0, -1.0]
    assert a["reprojection_error_px"][k][:2].tolist() == [4.0, 4.0]
    assert np.isnan(a["reprojection_error_px"][k][2])


def test_nonfinite_projection_is_not_a_zero_reprojection_error(monkeypatch):
    a, k = measured(monkeypatch, error=np.nan)
    assert int(a["rejection_bits"][k]) == int(m.SubsetRejection.PROJECTION_NONFINITE)
    assert np.isnan(a["reprojection_error_px"][k]).all()
    assert np.isfinite(a["step_mm"][k]) and not a["admitted"][k]


@pytest.mark.parametrize(
    "value,usable",
    [
        (0.0, True),
        (np.nextafter(np.float32(1200), np.float32(0)), True),
        (1200.0, False),
        (-0.01, False),
        (np.nan, False),
    ],
)
def test_native_continuous_pixel_bounds_no_rounding(value, usable):
    values = inputs()
    values[0][:] = value
    result = build(values)
    assert result["arrays"]["ray_available"].all() == usable
    if not usable:
        assert all(c.calls == 0 for c in values[3].cameras)
        assert not result["arrays"]["attempted"].any()


def test_fixed_initial_query_can_be_unreported_then_reappear():
    values = inputs()
    values[1][:] = False
    assert not build(values)["arrays"]["node_valid"].any()
    values[1][:] = True
    a = build(values, previous_frame_idx=101, frame_idx=102)["arrays"]
    assert a["node_valid"].all()
    assert a["chosen_subset"].tolist() == [[0, 0, 0]] * 3


def test_no_new_query_and_no_frame_zero_predecessor_synthesis():
    values = inputs()
    values[2][0, 0, 0] = False
    with pytest.raises(ValueError, match="create an initial query"):
        build(values)
    with pytest.raises(ValueError, match="after anchor"):
        build(previous_frame_idx=99, frame_idx=100)


@pytest.mark.parametrize(
    "clock",
    [
        dict(frame_idx=102),
        dict(frame_idx=True),
        dict(release_horizon=100),
        dict(release_horizon=300),
        dict(previous_frame_idx=299, frame_idx=300, release_horizon=300),
    ],
)
def test_native_clock_and_200_bounds(clock):
    with pytest.raises(ValueError):
        build(**clock)


def test_final_200_edge_and_zero_lookahead_allowed():
    result = build(previous_frame_idx=298, frame_idx=299, release_horizon=299)
    assert result["metadata"]["lookahead_frames"] == 0


def test_all36_subsets_and27_slots_with_missing_observations():
    values = inputs()
    values[1][:] = False
    values[2][:] = False
    values[0][:] = np.nan
    result = build(values)
    a, meta = result["arrays"], result["metadata"]
    assert len(meta["slot_records"]) == 27 and len(meta["subset_records"]) == 36
    assert not a["attempted"].any()
    assert (a["rejection_bits"] == int(m.SubsetRejection.MISSING_VIEWS)).all()
    assert np.isnan(a["subset_xyz"]).all() and (a["normal_rank"] == -1).all()
    assert all(not x.flags.writeable for x in a.values())
    assert not any("quality" in key or "fb" in key for key in a)


def test_rays_nonfinite_reason_and_subset_fallback():
    values = inputs()

    def bad(xy):
        return np.full((len(xy), 3), np.nan), np.ones((len(xy), 3))

    values[3].cameras[2].rays = bad
    a = build(values)["arrays"]
    assert (a["slot_rejection_bits"][:, 2] == int(m.SlotRejection.RAY_NONFINITE)).all()
    assert (a["chosen_subset"] == 1).all()


def test_equal_two_view_candidates_choose_fixed_subset_order(monkeypatch):
    # Isolate the new tie rule: no CT quality factor or fitted score breaks a tie.
    original = m.triangulate

    def measured_rms(origins, directions):
        xyz, _residual = original(origins, directions)
        return xyz, np.asarray(2.0 if len(origins) == 3 else 0.0)

    monkeypatch.setattr(m, "triangulate", measured_rms)
    a = build()["arrays"]
    assert not a["admitted"][:, :, 0].any()
    assert a["admitted"][:, :, 1:].all()
    assert (a["chosen_subset"] == 1).all()


@pytest.mark.parametrize("field", ["dtype", "shape", "previous", "world_units", "views"])
def test_explicit_input_schema(field):
    values = list(inputs())
    if field == "dtype":
        values[0] = values[0].astype(np.float64)
    elif field == "shape":
        values[1] = np.ones((3, 3), bool)
    elif field == "previous":
        values[4][0, 0] = np.nan
    elif field == "world_units":
        values[3].world_units = "m"
    else:
        values[3].views = tuple(reversed(m.VIEWS))
    with pytest.raises(ValueError):
        build(values)
