"""Conditional review paths retain complete histories and numerical offset invariance."""

import copy
import csv
import importlib
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module():
    names = {path.stem for path in LAB.glob("*.py")}
    saved, old_path = {name: sys.modules.get(name) for name in names}, sys.path.copy()
    try:
        sys.path.insert(0, str(LAB))
        for name in names:
            sys.modules.pop(name, None)
        return importlib.import_module("select_identity_memory_reviews")
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


m = load_module()


def fixture(n=5, seed=46):
    rng = np.random.default_rng(seed)
    pose = rng.normal(0, 5, (n, 3, 3, 3))
    observed = np.ones((n, 3, 3), bool)
    observed[1, :, 1] = [1, 0, 0]
    observed[2, :, 1] = [0, 1, 0]
    observed[3, :, 1] = [1, 1, 0]
    xyz = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    review = np.zeros((n, 3), bool)
    review[2:, 2] = True
    geometry = dict(frame_idx=np.arange(100, 100+n, dtype=np.int64), pose_xyz=pose,
                    observed=observed, sources=np.zeros((n, 3, 3, 3), np.int64),
                    identity_review=review, xyz=xyz,
                    accepted_xyz=np.where(~review[..., None], xyz, np.nan))
    unary = rng.uniform(0, .2, (n, 6))
    config = copy.deepcopy(m.evidence.FIXED_CONFIG)
    result = m.evidence.solve_memory(geometry, unary, config)
    dp, _ = m.common.split_solution(result)
    identity = m.common.build_identity_output(geometry, result.permutation_by_id)
    links = m.chosen_memory_links(geometry, result.permutation_by_id)
    scores = dict(valid_pixels=np.full((n, 3), 10, np.int64),
                  visible_mass=np.tile([[1., 2., 3.], [0., 0., 0.], [0., 0., 0.]], (n, 1, 1)),
                  enabled_views=np.tile([True, False, False], (n, 1)))
    return geometry, unary, config, dp, identity, links, scores


def path_cost(path, geometry, unary, permutations):
    links = m.chosen_memory_links(geometry, permutations[path])
    return float(unary[np.arange(len(path)), path].sum()+links["brownian_motion_cost_by_id"].sum())


def test_every_feasible_alternative_matches_complete_path_enumeration_with_multiple_memories():
    geometry, unary, config, dp, *_ = fixture()
    choices = np.array([(0, *tail) for tail in product(range(6), repeat=len(unary)-1)])
    costs = np.array([path_cost(path, geometry, unary, dp["permutations"]) for path in choices])
    index = m.GraphIndex(dp, unary)
    saw_multimemory = False
    for t, p in np.ndindex(len(unary), 6):
        if t == 0 or p == dp["path_state"][t]:
            continue
        result, info = m.reconstruct_alternative(dp, unary, geometry, t, p, config, index)
        expected = costs[choices[:, t] == p].min()
        assert path_cost(result, geometry, unary, dp["permutations"]) == pytest.approx(expected)
        assert result[t] == p and result[0] == 0
        assert info["conditional_gap"] == pytest.approx(expected-costs.min())
        nodes = np.arange(dp["layer_offsets"][t], dp["layer_offsets"][t+1])
        same_p = nodes[dp["state_permutation"][nodes] == p]
        saw_multimemory |= len(same_p) > 1
        expected_node = same_p[np.argmin(index.node_marginal[same_p])]
        assert info["anchor_memory_node"] == expected_node
    assert saw_multimemory


def test_large_common_offset_does_not_change_conditional_path_or_small_gap():
    geometry, _, config, _, *_ = fixture()
    geometry["pose_xyz"][:] = 0
    geometry["xyz"][:] = np.where(geometry["observed"][:, :, 1, None], 0., np.nan)
    geometry["accepted_xyz"][:] = np.where(~geometry["identity_review"][..., None],
                                           geometry["xyz"], np.nan)
    unary = np.tile(np.arange(6)/16, (len(geometry["frame_idx"]), 1))
    outcomes = []
    for shift in (0., float(2**40)):
        result = m.evidence.solve_memory(geometry, unary+shift, config)
        dp, _ = m.common.split_solution(result)
        p = next(p for p in range(6) if p != dp["path_state"][2])
        outcomes.append(m.reconstruct_alternative(dp, unary+shift, geometry, 2, p, config))
    np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
    assert outcomes[0][1]["conditional_gap"] == outcomes[1][1]["conditional_gap"]
    assert outcomes[1][1]["independent_centered_cost_absolute_error"] < 1e-12


def test_initial_constraint_and_chosen_permutation_cannot_be_false_alternatives():
    geometry, unary, config, dp, *_ = fixture()
    for t, p in ((0, 1), (2, dp["path_state"][2])):
        with pytest.raises(ValueError, match="different feasible"):
            m.reconstruct_alternative(dp, unary, geometry, t, p, config)


def test_internal_alternative_that_returns_is_preserved_as_full_path():
    geometry, _, config, _, *_ = fixture()
    geometry["observed"][:] = True
    geometry["pose_xyz"][:] = 0
    geometry["xyz"][:] = 0
    geometry["accepted_xyz"][:] = np.where(~geometry["identity_review"][..., None], 0., np.nan)
    unary = np.full((5, 6), 100.)
    unary[:, 0] = 0
    unary[2, 1] = .01
    result = m.evidence.solve_memory(geometry, unary, config)
    dp, _ = m.common.split_solution(result)
    path, info = m.reconstruct_alternative(dp, unary, geometry, 2, 1, config)
    assert path.tolist() == [0, 0, 1, 0, 0]
    assert info["same_first_and_last_permutation"]
    assert info["differing_ranges"] == [[102, 103]]
    assert info["conditional_gap"] == pytest.approx(.01)


def test_changed_graph_gap_is_rejected_before_selecting_a_conditional_path():
    _, unary, _, dp, *_ = fixture()
    dp["min_marginal_delta"][2, 1] += .1
    with pytest.raises(ValueError, match="graph replay"):
        m.GraphIndex(dp, unary)


def test_scene_priority_is_explicit_nonoverlapping_and_limited_not_exhaustive():
    events = [dict(kind="finite_identity_alternative_gap", frame_offset=t, priority=3,
                   order=[t/1000], identity_indices=[0]) for t in range(100, 2000, 100)]
    events += [dict(kind="mapping_change", frame_offset=1000, priority=1,
                    order=[-2], identity_indices=[0, 2])]
    scenes = m.select_scenes(events, 2000)
    assert len(scenes) == 5 and scenes[0]["focus_offset"] == 1000
    for i, scene in enumerate(scenes):
        assert scene["stop_offset"]-scene["start_offset"] <= 167
        assert scene["primary_reason"] in events
        for other in scenes[i+1:]:
            assert not (scene["start_offset"] < other["stop_offset"]
                        and other["start_offset"] < scene["stop_offset"])


def test_return_events_and_nonfinite_alternative_gaps_keep_distinct_reasons():
    _, _, _, dp, identity, links, scores = fixture()
    dp["path_state"][:] = [0, 1, 2, 0, 0]
    identity["identity_to_raw_slot"][:] = dp["permutations"][dp["path_state"]]
    dp["identity_alternative_gap"][:] = np.inf
    events = m.extract_events(dp, identity, scores, links, identity["identity_to_raw_slot"])
    returning = [event for event in events if event["kind"] == "mapping_departure_then_return"]
    assert len(returning) == 1 and returning[0]["excursion_range_offsets"] == [1, 3]
    assert not any(event["kind"] == "finite_identity_alternative_gap" for event in events)


def test_csv_labels_raw_geometry_slots_and_reorders_visibility_with_the_proposed_mapping(tmp_path):
    _, _, _, dp, identity, links, scores = fixture()
    identity["identity_to_raw_slot"][:] = [2, 0, 1]
    scores["valid_pixels"][0, 0] = 0
    scores["enabled_views"][0, 0] = False
    path = tmp_path / "table.csv"
    m.write_frame_table(path, dp, identity, scores, links,
                       np.tile([0, 1, 2], (len(dp["frame_idx"]), 1)))
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["id0_top_visible_mass"] == "3.0"
    assert rows[1]["id0_top_visible_mass_per_valid_pixel"] == "0.3"
    assert rows[0]["id0_top_visible_mass_per_valid_pixel"] == ""
    assert rows[1]["raw_fixed_id_to_slot"] == "0 1 2"
    assert rows[1]["memory_id_to_slot"] == "2 0 1"
    assert "id0_reobservation_brownian_cost" in rows[0]


def test_unsealed_evaluation_is_rejected_before_loading_any_result_arrays(tmp_path):
    run, evaluation = tmp_path / "run", tmp_path / "evaluation"
    run.mkdir()
    evaluation.mkdir()
    (evaluation / "evaluation.provenance.json").write_text('{"status":"failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be completed"):
        m.verify_evaluation(evaluation, run, {})


def test_scene_statistics_keep_missing_review_and_unavailable_visibility_distinct():
    _, _, _, dp, identity, links, scores = fixture()
    scores["valid_pixels"][:, 0] = 0
    scores["enabled_views"][:, 0] = False
    stats = m.scene_statistics(0, 5, dp, identity, scores, links)
    assert stats["minimum_top_visible_mass_per_valid_pixel_by_id"] == [None, None, None]
    assert stats["top_color_enabled_frames"] == 0
    assert sum(stats["prior_review_frames_by_id"]) == 3
    assert max(stats["maximum_direct_missing_age_by_id"]) > 0
    assert max(stats["maximum_reobservation_dt_by_id"]) > 1
    assert len(stats["maximum_reobservation_brownian_cost_by_id"]) == 3
