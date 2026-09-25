"""Chosen-event audit uses true direct observations, even when raw slots change."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module():
    names = ("identity_output", "identity_memory_links")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"_memory_links_test_{name}",
                                                         LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


LINK = load_module()


def geometry(direct, mapping=None):
    """Make physical-ID positions, then attach them to the requested raw slots."""
    direct = np.asarray(direct, dtype=bool)
    n = len(direct)
    mapping = np.tile([0, 1, 2], (n, 1)) if mapping is None else np.asarray(mapping)
    centers = np.array([[10, 15, 20], [50, 40, 35], [90, 80, 65]], float)
    velocity = np.array([[1, 2, -1], [-2, 1, 1], [1, -1, 2]], float)
    physical = centers[None] + np.arange(n)[:, None, None] * velocity
    skeleton = physical[:, :, None] + [[2, 0, 0], [0, 0, 0], [-4, 1, 0]]
    pose = np.empty_like(skeleton)
    observed = np.ones((n, 3, 3), bool)
    review = np.zeros((n, 3), bool)
    for t in range(n):
        pose[t, mapping[t]] = skeleton[t]
        observed[t, mapping[t], 1] = direct[t]
        review[t, mapping[t, 1]] = True  # Prior identity review does not erase raw measurements.
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    for t, raw in np.argwhere(~observed[:, :, 1]):
        sources[t, raw, 1:, 1] = -1
    xyz = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    data = dict(frame_idx=np.arange(333519, 333519+n), pose_xyz=pose, sources=sources,
                observed=observed, identity_review=review, xyz=xyz,
                accepted_xyz=np.where(~review[..., None], xyz, np.nan))
    return data, mapping, velocity


def test_all_direct_every_link_has_dt_one_and_matches_adjacent_displacement_cost():
    original, mapping, velocity = geometry(np.ones((5, 3), bool))
    result = LINK.chosen_memory_links(original, mapping)
    assert not result["motion_cost_applied"][0].any()
    assert np.isnan(result["displacement_mm_when_reobserved"][0]).all()
    np.testing.assert_array_equal(result["previous_direct_frame"][0], [-1, -1, -1])
    np.testing.assert_array_equal(result["gap_frames_when_reobserved"][1:], np.ones((4, 3), int))
    np.testing.assert_array_equal(result["displacement_mm_when_reobserved"][1:],
                                  np.broadcast_to(velocity, (4, 3, 3)))
    expected = .2 / 3 * (velocity**2).sum(-1) / .8**2
    np.testing.assert_allclose(result["brownian_motion_cost_by_id"][1:],
                               np.broadcast_to(expected, (4, 3)), rtol=1e-14)
    assert not result["terminal_unsettled_gap"].any()
    assert not result["terminal_unobserved_age_frames"].any()
    assert result["motion_cost_applied"][1:].all()


def test_missing_memory_survives_zero_or_one_visible_fish_then_charges_whole_return_gap():
    support = [[1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]]
    original, mapping, velocity = geometry(support)
    result = LINK.chosen_memory_links(original, mapping, scale_mm=2., weight=.3)
    start = original["frame_idx"][0]
    np.testing.assert_array_equal(result["previous_direct_frame"], [
        [-1, -1, -1], [start, start, start], [start, start+1, start],
        [start, start+1, start], [start+3, start+1, start],
    ])
    np.testing.assert_array_equal(result["gap_frames_when_reobserved"], [
        [0, 0, 0], [0, 1, 0], [0, 0, 0], [3, 0, 0], [1, 0, 4],
    ])
    for t, fish, dt in ((1, 1, 1), (3, 0, 3), (4, 0, 1), (4, 2, 4)):
        dx = velocity[fish] * dt
        expected = .3 / 3 * ((dx @ dx) / (4 * dt) + 3 * np.log(dt))
        np.testing.assert_array_equal(result["displacement_mm_when_reobserved"][t, fish], dx)
        assert result["brownian_motion_cost_by_id"][t, fish] == pytest.approx(expected)
    unapplied = ~result["motion_cost_applied"]
    assert not result["brownian_motion_cost_by_id"][unapplied].any()
    assert np.isnan(result["displacement_mm_when_reobserved"][unapplied]).all()
    np.testing.assert_array_equal(result["terminal_last_direct_frame"], [start+4, start+1, start+4])
    np.testing.assert_array_equal(result["terminal_unobserved_age_frames"], [0, 3, 0])
    np.testing.assert_array_equal(result["terminal_unsettled_gap"], [False, True, False])


def test_last_direct_event_follows_identity_across_non_self_inverse_raw_cycles():
    mappings = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 0, 1], [1, 0, 2]])
    support = [[1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]]
    original, mapping, velocity = geometry(support, mappings)
    result = LINK.chosen_memory_links(original, mapping)
    # ID0's intermediate missing raw slot changes twice. Its remembered event
    # remains the directly observed frame0/raw0, not the preceding raw slot.
    np.testing.assert_array_equal(result["previous_direct_raw_slot"][1:4, 0], [0, 0, 0])
    assert result["previous_direct_raw_slot"][4, 0] == 2
    assert result["gap_frames_when_reobserved"][3, 0] == 3
    np.testing.assert_array_equal(result["displacement_mm_when_reobserved"][3, 0], velocity[0]*3)
    np.testing.assert_array_equal(result["terminal_last_direct_raw_slot"], mappings[-1])
    # Removing raw reordering leaves every ID's physical motion objective equal.
    plain, identity, _ = geometry(support)
    reference = LINK.chosen_memory_links(plain, identity)
    for key in ("direct_observed", "gap_frames_when_reobserved",
                "displacement_mm_when_reobserved", "brownian_motion_cost_by_id"):
        np.testing.assert_array_equal(result[key], reference[key])


def test_unreturned_terminal_ids_are_unsettled_without_synthetic_displacement_or_charge():
    original, mapping, _ = geometry([[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = LINK.chosen_memory_links(original, mapping)
    np.testing.assert_array_equal(result["terminal_unobserved_age_frames"], [2, 3, 3])
    assert result["terminal_unsettled_gap"].all()
    assert np.isnan(result["displacement_mm_when_reobserved"][2:]).all()
    assert not result["brownian_motion_cost_by_id"][2:].any()
    assert result["brownian_motion_cost_by_id"].sum() > 0


def test_missing_latent_coordinates_and_prior_review_do_not_change_direct_event_cost():
    original, mapping, _ = geometry([[1, 1, 1], [0, 1, 0], [1, 1, 1]])
    result = LINK.chosen_memory_links(original, mapping)
    changed = {key: value.copy() for key, value in original.items()}
    changed["pose_xyz"][1, [0, 2], 1] = 1e8
    changed["identity_review"][:] = True
    changed["accepted_xyz"][:] = np.nan
    replay = LINK.chosen_memory_links(changed, mapping)
    for key in result:
        np.testing.assert_array_equal(replay[key], result[key])


def test_arrays_are_copied_and_caller_mutations_cannot_change_recorded_memory():
    original, mapping, _ = geometry([[1, 1, 1], [0, 1, 0], [1, 1, 1]])
    before = {key: value.copy() for key, value in original.items()}
    result = LINK.chosen_memory_links(original, mapping)
    for value in result.values():
        assert not np.shares_memory(value, mapping)
        assert all(not np.shares_memory(value, v) for v in original.values())
    for key, value in original.items():
        np.testing.assert_array_equal(value, before[key])
    expected_frame = result["terminal_last_direct_frame"].copy()
    original["frame_idx"][:] = 0
    mapping[:] = 2
    np.testing.assert_array_equal(result["terminal_last_direct_frame"], expected_frame)
    result["identity_to_raw_slot"][:] = 1
    np.testing.assert_array_equal(original["pose_xyz"], before["pose_xyz"])


def test_one_initial_frame_has_no_motion_and_three_settled_events():
    original, mapping, _ = geometry([[1, 1, 1]])
    result = LINK.chosen_memory_links(original, mapping)
    assert not result["motion_cost_applied"].any()
    assert not result["brownian_motion_cost_by_id"].any()
    np.testing.assert_array_equal(result["terminal_last_direct_raw_slot"], [0, 1, 2])
    assert not result["terminal_unsettled_gap"].any()


def test_initial_incomplete_direct_support_is_rejected_even_with_finite_latent_pose():
    original, mapping, _ = geometry([[1, 0, 1], [1, 1, 1]])
    assert np.isfinite(original["pose_xyz"]).all()
    with pytest.raises(ValueError, match="first frame must directly observe all three"):
        LINK.chosen_memory_links(original, mapping)


@pytest.mark.parametrize("keyword,value", [
    ("scale_mm", 0), ("scale_mm", -1), ("scale_mm", True), ("scale_mm", np.nan),
    ("weight", -1), ("weight", False), ("weight", np.inf), ("weight", 1j),
])
def test_invalid_numeric_parameters_fail_loudly(keyword, value):
    original, mapping, _ = geometry([[1, 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError, match="invalid"):
        LINK.chosen_memory_links(original, mapping, **{keyword: value})


def test_zero_weight_keeps_event_displacements_but_no_motion_charge():
    original, mapping, _ = geometry([[1, 1, 1], [0, 1, 0], [1, 1, 1]])
    result = LINK.chosen_memory_links(original, mapping, weight=0)
    assert not result["brownian_motion_cost_by_id"].any()
    assert result["motion_cost_applied"][2].all()
    np.testing.assert_array_equal(result["gap_frames_when_reobserved"][2], [2, 1, 2])


@pytest.mark.parametrize("terminal_returns", [False, True])
def test_chosen_links_independently_match_memory_graph_objective_and_terminal(terminal_returns):
    saved = sys.modules.get("identity_dp")
    try:
        for name in ("identity_dp", "identity_memory_dp"):
            spec = importlib.util.spec_from_file_location(f"_memory_links_cross_test_{name}",
                                                         LAB / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            if name == "identity_dp":
                sys.modules[name] = module
            spec.loader.exec_module(module)
        memory = module
    finally:
        if saved is None:
            sys.modules.pop("identity_dp", None)
        else:
            sys.modules["identity_dp"] = saved
    support = [[1, 1, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]]
    if terminal_returns:
        support[-1] = [1, 1, 1]
    original, _, _ = geometry(support)
    unary = np.random.default_rng(107).random((5, 6))
    solved = memory.solve_identity_memory_dp(
        original["frame_idx"], unary, original["xyz"], original["observed"][:, :, 1],
    )
    links = LINK.chosen_memory_links(original, solved.permutation_by_id)
    objective = links["brownian_motion_cost_by_id"].sum() + unary[
        np.arange(5), solved.path_state
    ].sum()
    assert solved.path_cost == pytest.approx(objective, rel=1e-12, abs=1e-12)
    for own, graph in (
        ("terminal_last_direct_frame", "terminal_last_event_frame"),
        ("terminal_last_direct_raw_slot", "terminal_last_event_slot"),
        ("terminal_unobserved_age_frames", "terminal_age_frames"),
        ("terminal_unsettled_gap", "terminal_unsettled_gap"),
    ):
        np.testing.assert_array_equal(links[own], getattr(solved, graph))
    for t in range(1, 5):
        selected = np.flatnonzero((solved.edge_source == solved.path_node[t-1])
                                  & (solved.edge_target == solved.path_node[t]))
        assert len(selected) == 1
        edge = selected[0]
        np.testing.assert_array_equal(links["gap_frames_when_reobserved"][t],
                                      solved.edge_delta_frames[edge])
        assert solved.edge_cost[edge] == pytest.approx(
            links["brownian_motion_cost_by_id"][t].sum(), rel=1e-12, abs=1e-12
        )
