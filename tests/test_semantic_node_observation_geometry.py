"""Small actual refractive geometry, entirely synthetic and independent of assets."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from medaka3d.calib import RefractiveCamera, Rig, triangulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import semantic_node_observation as m

VIEWS = ("top_cam", "side1_cam", "side2_cam")


def synthetic_rig():
    # The top interface is air/water; both side interfaces include glass.
    poses = (
        ([96, 96, 500], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0, -1], [0, 0, 186]),
        ([96, -400, 93], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [0, 1, 0], [0, -4, 0]),
        ([500, 96, 93], [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], [-1, 0, 0], [196, 0, 0]),
    )
    cameras = []
    for index, (view, (center, rotation, normal, point)) in enumerate(
        zip(VIEWS, poses, strict=True)
    ):
        cameras.append(
            RefractiveCamera(
                name=view,
                view=view,
                fx=2000.0,
                fy=2100.0,
                cx=600.0,
                cy=600.0,
                distortion=(-0.02, 0.004, 0.0004, -0.0002, 0.0),
                rotation=np.asarray(rotation, np.float64),
                center=np.asarray(center, np.float64),
                interface_normal=np.asarray(normal, np.float64),
                interface_point=np.asarray(point, np.float64),
                glass_thickness_mm=4.0,
                index_air=1.0003,
                index_glass=1.52,
                index_water=1.3325,
                layer_count=1 if index == 0 else 2,
            )
        )
    # A descriptive nonexistent path; neither this fixture nor the API opens it.
    return Rig(tuple(cameras), Path("synthetic-unwritten-calibration.json"))


def synthetic_pose():
    centers = np.array([[80.0, 84.0, 60.0], [96.0, 98.0, 88.0], [112.0, 111.0, 105.0]])
    nodes = np.array([[-4.0, -2.0, 1.0], [0.0, 0.0, 0.0], [5.0, 1.0, -1.0]])
    return centers[:, None] + nodes[None]


def observation_inputs(rig=None):
    rig = synthetic_rig() if rig is None else rig
    target = synthetic_pose()
    uv = np.stack([camera.project(target) for camera in rig.cameras], axis=1).astype(np.float32)
    assert uv.shape == (3, 3, 3, 2)
    assert np.isfinite(uv).all() and ((uv >= 0) & (uv < 1200)).all()
    return target, dict(
        uv=uv,
        reported=np.ones((3, 3, 3), bool),
        initial_selected=np.ones((3, 3, 3), bool),
        rig=rig,
        previous_pose_xyz=target - np.array([0.75, -0.5, 0.25]),
        initial_frame_idx=1000,
        previous_frame_idx=1006,
        frame_idx=1007,
        release_horizon=1015,
    )


def input_arrays(inputs):
    arrays = [inputs[key] for key in ("uv", "reported", "initial_selected", "previous_pose_xyz")]
    for camera in inputs["rig"].cameras:
        arrays.extend(
            (camera.rotation, camera.center, camera.interface_normal, camera.interface_point)
        )
    return arrays


def snapshots(arrays):
    return [(array.dtype.str, array.shape, array.tobytes()) for array in arrays]


def test_actual_refractive_rays_reconstruct_all_nodes_with_three_views():
    target, inputs = observation_inputs()
    arrays_before = input_arrays(inputs)
    before = snapshots(arrays_before)
    result = m.build_observation(**inputs)
    arrays = result["arrays"]
    assert arrays["node_valid"].all()
    assert arrays["used_views"].all()
    assert (arrays["chosen_subset"] == 0).all()
    assert arrays["attempted"].all() and arrays["admitted"].all()
    assert not arrays["rejection_bits"].any()
    assert len(result["metadata"]["slot_records"]) == 27
    assert len(result["metadata"]["subset_records"]) == 36
    np.testing.assert_allclose(arrays["pose_xyz"], target, atol=2e-4, rtol=0)
    # Independent all-view assembly through the calibration API, including its
    # published residual; this does not copy the observation's subset loop.
    origins, directions = inputs["rig"].rays(
        {view: inputs["uv"][:, index] for index, view in enumerate(VIEWS)}
    )
    expected, rms = triangulate(origins, directions)
    np.testing.assert_allclose(arrays["pose_xyz"], expected, atol=1e-10, rtol=0)
    assert np.max(rms) < 1e-4
    assert snapshots(arrays_before) == before
    for value in arrays.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            assert all(not np.shares_memory(value, source) for source in arrays_before)


@pytest.mark.parametrize("missing_view, subset", [(0, 3), (1, 2), (2, 1)])
def test_one_missing_view_falls_back_to_the_correct_real_two_ray_pair(missing_view, subset):
    target, inputs = observation_inputs()
    inputs["reported"][:, missing_view, :] = False
    # Missing image evidence stays absent even when the latent point is known.
    inputs["uv"][:, missing_view, :, :] = np.nan
    before = snapshots(input_arrays(inputs))
    arrays = m.build_observation(**inputs)["arrays"]
    assert arrays["node_valid"].all()
    assert (arrays["chosen_subset"] == subset).all()
    rejected = [index for index in range(4) if index != subset]
    assert not arrays["attempted"][:, :, rejected].any()
    assert (arrays["rejection_bits"][:, :, rejected] == int(m.SubsetRejection.MISSING_VIEWS)).all()
    np.testing.assert_allclose(arrays["pose_xyz"], target, atol=2e-4, rtol=0)
    expected_used = np.ones((3, 3, 3), bool)
    expected_used[:, missing_view, :] = False
    np.testing.assert_array_equal(arrays["used_views"], expected_used)
    assert snapshots(input_arrays(inputs)) == before


def test_inconsistent_third_real_ray_is_rejected_while_good_pair_survives():
    target, inputs = observation_inputs()
    inputs["uv"][:, 2, :, :] += np.array([120.0, -80.0], np.float32)
    before = snapshots(input_arrays(inputs))
    arrays = m.build_observation(**inputs)["arrays"]
    assert arrays["node_valid"].all()
    assert (arrays["chosen_subset"] == 1).all()
    assert arrays["attempted"].all()
    assert not arrays["admitted"][:, :, 0].any()
    assert (arrays["ray_rms_mm"][:, :, 0] > 1.0).all()
    assert (arrays["rejection_bits"][:, :, 0] & int(m.SubsetRejection.RAY_RMS)).all()
    assert not arrays["rejection_bits"][:, :, 1].any()
    np.testing.assert_allclose(arrays["pose_xyz"], target, atol=2e-4, rtol=0)
    assert snapshots(input_arrays(inputs)) == before


def test_coincident_real_cameras_cannot_turn_parallel_rays_into_observations():
    one = synthetic_rig().cameras[0]
    rig = Rig(
        tuple(replace(one, name=view, view=view) for view in VIEWS), Path("synthetic-coincident")
    )
    _, inputs = observation_inputs(rig)
    before = snapshots(input_arrays(inputs))
    arrays = m.build_observation(**inputs)["arrays"]
    assert not arrays["node_valid"].any()
    assert not arrays["used_views"].any()
    assert (arrays["chosen_subset"] == -1).all()
    assert np.isnan(arrays["pose_xyz"]).all()
    assert arrays["attempted"].all() and not arrays["admitted"].any()
    assert (np.abs(arrays["normal_determinant"]) <= 1e-6).all()
    assert (arrays["rejection_bits"] == int(m.SubsetRejection.TRIANGULATION_SINGULAR)).all()
    assert np.isnan(arrays["subset_xyz"]).all()
    assert np.isnan(arrays["ray_rms_mm"]).all()
    assert snapshots(input_arrays(inputs)) == before
