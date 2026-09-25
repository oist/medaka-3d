"""Independent synthetic geometry tests; no recordings, payloads or GPU.

Reuse only the old test module's orthographic camera and point factory. Expected
camera subsets below follow their coordinate intersections, not the new helper.
"""

import copy

import body_row_proposals as old
import nodewise_body_row_proposals as new
import numpy as np
import pytest
from test_body_row_proposals import LOWER, UPPER, fixture, snapshot

from medaka3d.types import VIEWS


@pytest.fixture(autouse=True)
def no_payload_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic geometry tests may not read an array archive")

    monkeypatch.setattr(np, "load", forbidden)


def equal_tree(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert (left.dtype, left.shape, left.tobytes()) == (
            right.dtype,
            right.shape,
            right.tobytes(),
        )
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def crossed_bad_views(points):
    # Cameras project (x,y), (y,z), (x,z), respectively.
    # Mouth: only views01 agree. Tail: only views12 agree.
    points[VIEWS[2]][0, 0] += 40.0
    points[VIEWS[0]][0, 2] += 40.0


def evaluate(rig, rays, lower=LOWER, upper=UPPER, config=None):
    config = old.BodyRowConfig() if config is None else config
    previous, _ = old.body_row_pool(rig, rays, lower, upper, config)
    before = snapshot(rays)
    saved_previous = copy.deepcopy(previous)
    result, stats = new.nodewise_body_row_pool(rig, rays, lower, upper, config)
    assert len(result) >= len(previous)
    equal_tree(result[: len(previous)], saved_previous)
    assert snapshot(rays) == before
    assert [p["pool_index"] for p in result] == list(range(len(result)))
    return previous, result[len(previous) :], stats


def check_proposal_only(candidate):
    assert candidate["triangulated_nodes"].tolist() == [True, True, True]
    assert np.isfinite(candidate["pose_xyz"]).all()
    assert candidate["same_row_within_each_view"]
    assert candidate["direct_support_not_recomputed"]
    assert "observed" not in candidate and not candidate.get("physical_IDs_assigned", False)
    plan, rows = candidate["proposed_sources"], candidate["view_rows"]
    assert plan.shape == (3, 3) and plan.dtype.kind in "iu"
    assert (plan >= 0).sum(axis=0).min() >= 2
    assert np.all((plan < 0) | (plan == rows[:, None]))


def test_different_pairs_reconstruct_mouth_and_tail_in_one_coherent_row_body():
    bodies, rig, rays = fixture(point_changes=crossed_bad_views)
    previous, added, _ = evaluate(rig, rays)
    assert previous == [] and len(added) == 1
    p = added[0]
    check_proposal_only(p)
    np.testing.assert_allclose(p["pose_xyz"], bodies[0], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(p["view_rows"], [0, 0, 0])
    np.testing.assert_array_equal(p["proposed_sources"], [[0, 0, -1], [0, 0, 0], [-1, 0, 0]])
    assert p["used_ray_count"] == 7
    errors = p["reprojection_error_px"]
    assert errors.shape == (3, 3) and np.isfinite(errors).all()
    assert errors[2, 0] > 40 and errors[0, 2] > 40
    np.testing.assert_allclose(errors[p["proposed_sources"] >= 0], 0, atol=1e-12)


def test_two_valid_chest_pairs_are_both_retained_without_choosing_one_arbitrarily():
    def changes(points):
        crossed_bad_views(points)
        points[VIEWS[0]][0, 1, 0] += 10.0
        # 01 agrees in Y, 12 in Z, but 02 disagrees in X by10.

    bodies, rig, rays = fixture(point_changes=changes)
    previous, added, _ = evaluate(rig, rays)
    assert previous == [] and len(added) == 2
    by_plan = {tuple(p["proposed_sources"][:, 1]): p for p in added}
    assert set(by_plan) == {(0, 0, -1), (-1, 0, 0)}
    for key, p in by_plan.items():
        check_proposal_only(p)
        np.testing.assert_array_equal(p["proposed_sources"][:, 0], [0, 0, -1])
        np.testing.assert_array_equal(p["proposed_sources"][:, 2], [-1, 0, 0])
        expected = bodies[0].copy()
        if key == (0, 0, -1):
            expected[1, 0] += 10.0
        np.testing.assert_allclose(p["pose_xyz"], expected, atol=1e-12, rtol=0)
        excluded_view = 2 if key == (0, 0, -1) else 0
        assert p["reprojection_error_px"][excluded_view, 1] == pytest.approx(10.0)
    assert len({p["proposed_sources"].tobytes() for p in added}) == 2


def test_good_three_view_solution_does_not_add_redundant_pair_variants():
    _, rig, rays = fixture()
    previous, added, _ = evaluate(rig, rays)
    # The four OLD full/paired plans remain; maximal-view new alternatives add none.
    assert len(previous) == 4 and added == []
    assert sum(np.all(p["proposed_sources"] >= 0) for p in previous) == 1


def test_duplicate_plan_is_not_added_under_a_new_three_view_tuple():
    def bad_third(points):
        points[VIEWS[2]] += 40.0

    _, rig, rays = fixture(point_changes=bad_third)
    previous, added, _ = evaluate(rig, rays)
    assert len(previous) == 1 and added == []
    np.testing.assert_array_equal(previous[0]["view_rows"], [0, 0, -1])
    np.testing.assert_array_equal(
        previous[0]["proposed_sources"], [[0, 0, 0], [0, 0, 0], [-1, -1, -1]]
    )


def test_original_nonzero_row_numbers_and_missing_padding_rows_are_preserved():
    def padded(points):
        crossed_bad_views(points)
        for vi, view in enumerate(VIEWS):
            data = np.full((vi + 2, 3, 2), np.nan)
            data[vi + 1] = points[view][0]
            points[view] = data

    bodies, rig, rays = fixture(point_changes=padded)
    previous, added, _ = evaluate(rig, rays)
    assert previous == [] and len(added) == 1
    p = added[0]
    check_proposal_only(p)
    np.testing.assert_array_equal(p["view_rows"], [1, 2, 3])
    np.testing.assert_array_equal(p["proposed_sources"], [[1, 1, -1], [2, 2, 2], [-1, 3, 3]])
    np.testing.assert_allclose(p["pose_xyz"], bodies[0], atol=1e-12, rtol=0)


def test_partial_old_body_is_retained_but_no_new_partial_body_is_manufactured():
    def partial(points):
        points[VIEWS[2]][0, 0] += 40.0
        points[VIEWS[1]][0, 2] = np.nan
        points[VIEWS[2]][0, 2] = np.nan

    _, rig, rays = fixture(point_changes=partial)
    previous, added, _ = evaluate(rig, rays)
    assert len(previous) == 1 and added == []
    assert previous[0]["triangulated_nodes"].tolist() == [True, True, False]
    assert np.isnan(previous[0]["pose_xyz"][2]).all()
    assert previous[0]["proposed_sources"][:, 2].tolist() == [0, -1, -1]


def test_node_with_only_two_finite_views_uses_both_without_inventing_the_third():
    def missing_and_bad(points):
        points[VIEWS[2]][0, 0] = np.nan
        points[VIEWS[0]][0, 2] += 40.0

    bodies, rig, rays = fixture(point_changes=missing_and_bad)
    previous, added, stats = evaluate(rig, rays)
    assert len(previous) == 1 and len(added) == 1
    assert previous[0]["triangulated_nodes"].tolist() == [False, True, True]
    candidate = added[0]
    check_proposal_only(candidate)
    np.testing.assert_allclose(candidate["pose_xyz"], bodies[0], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(candidate["proposed_sources"][:, 0], [0, 0, -1])
    np.testing.assert_array_equal(candidate["available_node_views"][0], [True, True, False])
    np.testing.assert_array_equal(candidate["node_view_mask"][0], [True, True, False])
    assert not candidate["excluded_available_node_views"][0].any()
    assert np.isnan(candidate["reprojection_error_px"][2, 0])
    assert stats["legacy_candidate_count"] == stats["additional_candidate_count"] == 1


def test_one_impossible_node_cannot_be_dropped_to_accept_the_other_two():
    def inconsistent_chest(points):
        crossed_bad_views(points)
        points[VIEWS[0]][0, 1, 0] += 10.0  # X, disagrees with view2
        points[VIEWS[1]][0, 1, 0] += 10.0  # Y, disagrees with view0
        points[VIEWS[2]][0, 1, 1] += 10.0  # Z, disagrees with view1

    _, rig, rays = fixture(point_changes=inconsistent_chest)
    previous, added, _ = evaluate(rig, rays)
    assert previous == added == []


@pytest.mark.parametrize("error,accepted", [(3.0, True), (3.1, False)])
def test_pair_rms_gate_keeps_inclusive_existing_threshold(error, accepted):
    def changes(points):
        crossed_bad_views(points)
        points[VIEWS[1]][0, 0, 0] += error

    _, rig, rays = fixture(point_changes=changes)
    _, added, _ = evaluate(rig, rays)
    assert bool(added) == accepted
    if accepted:
        assert len(added) == 1 and added[0]["ray_rms_mm"][0] == pytest.approx(1.5)


@pytest.mark.parametrize("bias,accepted", [(12.0, True), (12.01, False)])
def test_assigned_view_pixel_gate_is_checked_even_when_rays_intersect_exactly(bias, accepted):
    _, rig, rays = fixture(point_changes=crossed_bad_views)
    # After constructing exact rays, change only projector output in one coordinate.
    rig.camera(VIEWS[1]).projection_bias = np.array([bias, 0.0])
    _, added, _ = evaluate(rig, rays)
    assert bool(added) == accepted
    if accepted:
        np.testing.assert_allclose(added[0]["reprojection_error_px"][1], bias, atol=1e-12)


@pytest.mark.parametrize("failure", ["parallel", "behind", "bounds"])
def test_invalid_intersections_do_not_become_nodewise_candidates(failure):
    _, rig, rays = fixture(
        point_changes=crossed_bad_views,
        camera_axes=(2, 2, 2) if failure == "parallel" else (2, 0, 1),
        plane=100.0 if failure == "behind" else -100.0,
    )
    lower = [1.0, -50.0, -50.0] if failure == "bounds" else LOWER
    previous, added, _ = evaluate(rig, rays, lower=lower)
    assert previous == added == []


def test_empty_and_missing_camera_cannot_create_a_new_three_row_tuple():
    def absent(points):
        points[VIEWS[2]] = np.empty((0, 3, 2))

    _, rig, rays = fixture(point_changes=absent)
    previous, added, _ = evaluate(rig, rays)
    assert len(previous) == 1 and added == []

    def all_missing(points):
        for value in points.values():
            value[:] = np.nan

    _, rig, rays = fixture(point_changes=all_missing)
    previous, added, _ = evaluate(rig, rays)
    assert previous == added == []
