"""Synthetic intersection tests use the real CPU triangulator and support gates."""

from dataclasses import replace

import diagnose_grouped_mouth_intersections as diag
import numpy as np
import pytest


class Camera:
    def __init__(self, view):
        self.axes = ((0, 1), (2, 1), (0, 2))[view]
        self.depth = (2, 0, 1)[view]

    def project(self, xyz):
        return np.asarray(xyz)[..., self.axes]

    def rays(self, uv):
        origin = np.zeros((*uv.shape[:-1], 3), dtype=np.float64)
        origin[..., self.axes] = uv
        direction = np.zeros_like(origin)
        direction[..., self.depth] = 1.0
        return origin, direction


class Rig:
    def camera(self, view):
        return Camera(diag.VIEWS.index(view))


def scene():
    truth = np.array(
        [[[x, 20, 30], [x + 5, 20, 30], [x + 20, 20, 30]] for x in (20, 80, 130)], dtype=np.float64
    )
    rig = Rig()
    points = {v: rig.camera(v).project(truth).astype(np.float32) for v in diag.VIEWS}
    rays = diag.RayObservations.from_points(rig, points)
    pose = truth.copy()
    pose[2, 0] = [135, 25, 30]
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    plan[2, :, 0] = -1
    direct, observed, _ = diag.supported_nodes(pose, plan, rays, rig)
    assert np.array_equal(direct, plan)
    return dict(
        pose=pose,
        base_plan=plan,
        expected_observed=observed,
        rays=rays,
        rig=rig,
        initial_lengths=np.full((3, 2), [5, 15], dtype=np.float64),
        config=diag.NodeRecoveryConfig(),
        reference_available=np.ones(3, bool),
        reference_rows=np.broadcast_to(np.arange(3), (3, 3)).copy(),
    )


def test_all_four_subsets_recover_true_mouth_without_changing_other_nodes_or_inputs():
    inputs = scene()
    before = {k: diag.proof.signature(v) for k, v in inputs.items() if isinstance(v, np.ndarray)}
    result, arrays = diag.assess_scene(**inputs)
    assert len(result["trials"]) == result["feasible_subsets"] == 4
    assert [len(t["views"]) for t in result["trials"]] == [3, 2, 2, 2]
    for i, row in enumerate(result["trials"]):
        np.testing.assert_array_equal(row["mouth_xyz"], [130, 20, 30])
        assert row["ray_rms_mm"] == 0 and row["seed_length_ratio"] == 1
        assert row["mouth_supported_views"] == len(row["views"])
        assert row["other_direct_support_losses"] == []
        np.testing.assert_array_equal(arrays[f"trial_{i:02d}__pose_xyz"][:2], inputs["pose"][:2])
        np.testing.assert_array_equal(
            arrays[f"trial_{i:02d}__pose_xyz"][:, 1:], inputs["pose"][:, 1:]
        )
    assert before == {
        k: diag.proof.signature(v) for k, v in inputs.items() if isinstance(v, np.ndarray)
    }


def test_neutral_mask_is_unknown_but_eligible_group_mismatch_excludes_a_view():
    inputs = scene()
    inputs["reference_available"][0] = False
    inputs["reference_rows"][0, 2] = -1
    inputs["reference_rows"][1, 2] = 1
    result, _ = diag.assess_scene(**inputs)
    assert result["chest_group_rows"][0]["chest_row_matches_mask_group"] is None
    assert result["chest_group_rows"][0]["allowed_for_intersection"]
    assert not result["chest_group_rows"][1]["allowed_for_intersection"]
    assert result["feasible_subsets"] == 1
    assert result["trials"][0]["views"] == [diag.VIEWS[0], diag.VIEWS[2]]


def test_other_fish_direct_mouth_is_protected_even_if_chest_group_is_eligible():
    inputs = scene()
    inputs["base_plan"][0, 0, 0] = 2
    rows = diag.chest_rows(
        inputs["base_plan"],
        inputs["rays"],
        inputs["reference_available"],
        inputs["reference_rows"],
        2,
    )
    assert rows[0]["competing_other_fish_direct_mouth"] == [0]
    assert not rows[0]["allowed_for_intersection"]


def test_missing_chest_and_missing_same_row_mouth_leave_no_two_view_subset():
    inputs = scene()
    inputs["base_plan"][2, 0, 1] = -1
    inputs["rays"].points[diag.VIEWS[1]][2, 0] = np.nan
    inputs["rays"] = diag.RayObservations.from_points(inputs["rig"], inputs["rays"].points)
    result, _ = diag.assess_scene(**inputs)
    assert result["feasible_subsets"] == 0 and result["trials"] == []
    assert not result["chest_group_rows"][0]["allowed_for_intersection"]
    assert not result["chest_group_rows"][1]["mouth_finite"]


@pytest.mark.parametrize(
    "gate,config",
    [
        ("box", replace(diag.NodeRecoveryConfig(), upper_mm=(129.0, 192.0, 186.0))),
        ("anatomy", replace(diag.NodeRecoveryConfig(), minimum_length_ratio=1.1)),
    ],
)
def test_existing_anatomy_and_box_limits_remain_hard_rejections(gate, config):
    inputs = scene()
    inputs["config"] = config
    result, _ = diag.assess_scene(**inputs)
    assert result["feasible_subsets"] == 0
    assert all(not trial["gates"][gate] for trial in result["trials"])


def test_degenerate_mouth_rays_are_preserved_as_invalid_numeric_evidence():
    inputs = scene()
    for origin, direction in zip(inputs["rays"].origins, inputs["rays"].directions, strict=True):
        origin[2, 0] = [130, 20, 0]
        direction[2, 0] = [0, 0, 1]
    result, arrays = diag.assess_scene(**inputs)
    assert len(result["trials"]) == 4 and result["feasible_subsets"] == 0
    assert all(not trial["gates"]["finite"] for trial in result["trials"])
    assert np.isnan(arrays["trial_00__mouth_xyz"]).all()
    assert not arrays["trial_00__direct_observed"][2, 0]


def test_base_observed_disagreement_is_rejected_before_any_intersection():
    inputs = scene()
    inputs["expected_observed"][2, 0] = True
    with pytest.raises(ValueError, match="base direct source/support differs"):
        diag.assess_scene(**inputs)
