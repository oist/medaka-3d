"""The common downstream preserves native rows and honestly measures replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS


@pytest.fixture(scope="module")
def downstream():
    path = (
        Path(__file__).resolve().parents[1]
        / "labs"
        / "20260917_custom2d_comparison"
        / "downstream.py"
    )
    if not path.exists():
        pytest.skip("The custom 2D comparison lab was retired")
    spec = importlib.util.spec_from_file_location("_custom2d_downstream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Camera:
    def __init__(self, center):
        self.center = np.asarray(center, dtype=float)
        self.ray_input_shapes = []

    def rays(self, uv):
        uv = np.asarray(uv)
        self.ray_input_shapes.append(uv.shape)
        xy = (uv - [400, 300]) / 600
        direction = np.concatenate([xy, np.ones((*xy.shape[:-1], 1))], axis=-1)
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        return np.broadcast_to(self.center, direction.shape).copy(), direction

    def project(self, xyz):
        rel = xyz - self.center
        return 600 * rel[..., :2] / rel[..., 2, None] + [400, 300]


class Rig:
    views = VIEWS

    def __init__(self):
        self.cameras = [Camera(c) for c in ([-40, 0, 0], [40, 0, 0], [0, 40, 0])]

    def camera(self, view):
        return self.cameras[VIEWS.index(view)]


def data(module, frames=100):
    rig = Rig()
    centers = np.array([[-12, 2, 90], [8, -10, 100], [18, 14, 110]], dtype=float)
    pose = centers[None, :, None] + np.array([[-5, 0, 0], [0, 0, 0], [10, 0, 0]])
    pose = pose + np.arange(frames)[:, None, None, None] * np.array([0.015, 0.02, 0.01])
    detections, labels = {}, {}
    for view in VIEWS:
        points = rig.camera(view).project(pose).reshape(-1, 3, 2).astype(np.float32)
        detections[view] = module.c01.Detections(
            experiment="synthetic",
            view=view,
            frame_idx=np.repeat(np.arange(frames), 3),
            points=points,
            score=np.full(frames * 3, 2.0),
            n_frames=frames,
        )
        labels[view] = np.tile([20, 900, 30000], frames)
    return rig, detections, labels, pose


def test_real_common_chain_keeps_native_rows_pose_and_inputs(downstream):
    rig, det, labels, truth = data(downstream)
    before = {view: det[view].points.copy() for view in VIEWS}
    result = downstream.run_downstream(det, labels, rig, 40, 55)
    np.testing.assert_array_equal(result["frames"], np.arange(40, 55))
    expected = np.broadcast_to(np.arange(3)[None, :, None], (15, 3, 3))
    for stage in ("c28", "final"):
        arm = result[stage]
        # Naming is arbitrary: map the first source row once across this clean interval.
        order = np.argsort(arm["source"][0, :, 0])
        np.testing.assert_array_equal(arm["source"][:, order], expected)
        np.testing.assert_allclose(arm["pose"][:, order], truth[40:55], atol=2e-5)
        np.testing.assert_array_equal(arm["pose"][:, :, 1], arm["xyz"])
        assert arm["xyz"].dtype == np.float32
    for view in VIEWS:
        np.testing.assert_array_equal(det[view].points, before[view])
        np.testing.assert_array_equal(labels[view], np.tile([20, 900, 30000], 100))
        assert result["diagnostics"]["input_row_slices"][view] == (120, 165)
    assert result["diagnostics"]["reference_reproduction"]["status"] == "reference_not_supplied"
    assert not result["diagnostics"]["c34_windows"]


def test_preparation_projects_only_window_and_keeps_global_detection_row_offsets(downstream):
    rig, det, labels, _ = data(downstream, frames=200)
    prepared = downstream.prepare_window(det, labels, rig, 110, 120)
    assert prepared.base == 5
    for view in VIEWS:
        assert prepared.detections[view].n_frames == 10
        np.testing.assert_array_equal(
            prepared.detections[view].frame_idx, np.repeat(np.arange(10), 3)
        )
        assert all(shape == (10, 12, 2) for shape in rig.camera(view).ray_input_shapes)
        assert prepared.input_row_slices[view] == (330, 360)


def test_duplicate_frame_identity_is_rejected(downstream):
    rig, det, labels, _ = data(downstream)
    labels[VIEWS[0]][120:122] = 20
    with pytest.raises(ValueError, match="duplicate 2D identity"):
        downstream.prepare_window(det, labels, rig, 40, 50)


def test_one_view_support_stays_unresolved_without_invented_points(downstream):
    rig, det, labels, _ = data(downstream)
    labels[VIEWS[0]][:] = -1
    labels[VIEWS[1]][:] = -1
    result = downstream.run_downstream(det, labels, rig, 40, 50)
    for name in ("c28", "final"):
        arm = result[name]
        assert np.isnan(arm["xyz"]).all()
        assert np.isnan(arm["pose"]).all()
        assert (arm["slot_track"] == -1).all()
        assert (arm["source"][:, :, :2] == -1).all()
        assert (arm["node_view_counts"] == 1).all()


def test_legacy_degraded_dilation_has_effective_radius_three_even_at_window_edge(downstream):
    rig, det, labels, _ = data(downstream)
    for view in VIEWS:
        det[view].score[37 * 3 : 38 * 3] = 1.0
    prepared = downstream.prepare_window(det, labels, rig, 40, 50)
    np.testing.assert_array_equal(prepared.degraded, [True] + [False] * 9)


def test_zero_lambda_geometry_objective_is_unchanged_by_label_connection_cuts(downstream):
    rig, det, labels, _ = data(downstream)
    a = downstream.run_downstream(det, labels, rig, 40, 50, lambda_mm=0)
    cuts = {view: values.copy() for view, values in labels.items()}
    for values in cuts.values():
        values[45 * 3 :] += 100_000
    b = downstream.run_downstream(det, cuts, rig, 40, 50, lambda_mm=0)
    assert b["diagnostics"]["viterbi_objective"] == pytest.approx(
        a["diagnostics"]["viterbi_objective"], abs=1e-10
    )
    for i in range(10):
        # Compare the three physical correspondences, independent of slot names.
        sa, sb = a["c28"]["source"][i], b["c28"]["source"][i]
        np.testing.assert_array_equal(sa[np.argsort(sa[:, 0])], sb[np.argsort(sb[:, 0])])


def test_reference_alignment_uses_one_global_permutation_and_does_not_change_output(downstream):
    source = np.broadcast_to(np.arange(3)[None, :, None], (10, 3, 3)).copy()
    xyz = np.broadcast_to(np.arange(9).reshape(1, 3, 3), (10, 3, 3)).astype(np.float32).copy()
    arm = {"source": source[:, [2, 0, 1]], "xyz": xyz[:, [2, 0, 1]]}
    reference = {"source": source, "xyz": xyz}
    original = arm["source"].copy()
    report = downstream.compare_reference(arm, reference, np.arange(100, 110), (102, 108))
    assert report["status"] == "exact_core_reproduction"
    assert report["single_permutation_output_reference_slot_selects_window_slot"] == [1, 2, 0]
    assert not report["permutation_applied_to_output"]
    np.testing.assert_array_equal(arm["source"], original)
    arm["source"][5] = arm["source"][5, [1, 0, 2]]
    arm["xyz"][5] = arm["xyz"][5, [1, 0, 2]]
    report = downstream.compare_reference(arm, reference, np.arange(100, 110), (102, 108))
    assert report["status"] == "windowed_core_differs"
    assert report["source_mismatch_native_frames"] == [105]
    assert report["source_exact_frame_fraction"] == pytest.approx(5 / 6)


def test_full_recording_reference_is_sliced_by_native_frame(downstream):
    reference = {
        "source": np.arange(100 * 9).reshape(100, 3, 3),
        "xyz": np.arange(100 * 9, dtype=float).reshape(100, 3, 3),
    }
    arm = {key: value[60:70].copy() for key, value in reference.items()}
    report = downstream.compare_reference(arm, reference, np.arange(60, 70), (62, 68))
    assert report["status"] == "exact_core_reproduction"
    assert report["xyz_max_paired_distance_mm"] == 0


def test_existing_microflip_repairs_only_its_short_window_and_reports_native_frames(downstream):
    rig, det, labels, _ = data(downstream)
    prepared = downstream.prepare_window(det, labels, rig, 20, 80)
    result = downstream.solve_window(prepared)
    arm = {
        key: result["c28"][key].copy()
        for key in ("xyz", "source", "residual_mm", "slot_rays", "margin_mm", "n_views_used")
    }
    for key in ("xyz", "source", "residual_mm", "slot_rays"):
        arm[key][30] = arm[key][30, [1, 0, 2]]
    final, diagnostic = downstream.postprocess(arm, prepared)
    np.testing.assert_array_equal(final["source"], result["c28"]["source"])
    np.testing.assert_array_equal(final["xyz"], result["c28"]["xyz"])
    assert diagnostic["c34_windows"][0]["first_frame"] == 50
    assert diagnostic["c34_windows"][0]["last_frame"] == 50
    assert diagnostic["c30_velocity_needs_median_mm"] == 2.5
    assert diagnostic["c30_bridge_margin_mm"] == 30.0
