"""Synthetic CPU integration and evidence arithmetic; no actual dataset access."""

import copy
from dataclasses import replace

import evaluate_tail_recovery_feasibility as e
import pytest
from test_tail_recovery import pose, rig, setup  # noqa: F401


def frame_result(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    scene, plan, rays, lengths = data[:4]
    observed = e.supported_nodes(scene, plan, rays, rig)[1]
    config = e.NodeRecoveryConfig()
    pools, diagnostics, times = e.run_current_frame(
        scene, plan, observed, rays, rig, lengths, config
    )
    return data, config, pools, diagnostics, times


def test_real_pure_helper_both_arms_keeps_inputs_and_returns_native_arrays(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    scene, plan, rays, lengths = data[:4]
    before = e.signature(scene, plan, lengths, rays)
    observed = e.supported_nodes(scene, plan, rays, rig)[1]
    pools, diagnostics, times = e.run_current_frame(
        scene, plan, observed, rays, rig, lengths, e.NodeRecoveryConfig()
    )
    assert e.signature(scene, plan, lengths, rays) == before
    for arm in e.ARMS:
        proposals, checked = pools[arm]
        assert proposals and all(p.fish == 1 and p.node == 2 for p in proposals)
        saved = e.pool_arrays(proposals, checked)
        assert saved["pose_xyz"].shape == (len(proposals), 3, 3, 3)
        assert saved["observed"][:, 1, 1:].all()
        assert saved["seed_method"].tolist() == ["free_intersection"] * len(proposals)
        assert diagnostics[arm]["age_gate_applied"] is False
        assert times[arm] >= 0


def test_saved_base_flags_mismatch_is_rejected_before_generation(rig, pose):  # noqa: F811
    data = setup(rig, pose)
    expected = e.supported_nodes(data[0], data[1], data[2], rig)[1]
    expected[1, 2] = True
    with pytest.raises(ValueError, match="filtered base"):
        e.run_current_frame(
            data[0], data[1], expected, data[2], rig, data[3], e.NodeRecoveryConfig()
        )


@pytest.mark.parametrize("field", ["pose_xyz", "plan"])
def test_candidate_cannot_change_non_tail_entries(rig, pose, field):  # noqa: F811
    data, config, pools, _, _ = frame_result(rig, pose)
    candidate = pools["free_only"][0][0]
    changed = getattr(candidate, field).copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="mouth/chest or another fish"):
        e.validate_candidate(
            replace(candidate, **{field: changed}), data[0], data[1], data[2], rig, config
        )


def test_old_coordinates_only_change_posthoc_annotations(rig, pose):  # noqa: F811
    data, _, pools, _, _ = frame_result(rig, pose)
    proposals, checked = pools["free_only"]
    before = copy.deepcopy(e.pool_arrays(proposals, checked))
    a = e.posthoc_rows(100, proposals, checked, [1], data[4], data[5])
    b = e.posthoc_rows(100, proposals, checked, [], data[4] + 50, data[5])
    assert a[0]["old_tail_distance_mm"] != b[0]["old_tail_distance_mm"]
    assert a[0]["target_lost_fish"] and not b[0]["target_lost_fish"]
    assert not a[0]["old_pose_used_by_generator"]
    after = e.pool_arrays(proposals, checked)
    for key in before:
        assert e.checks.equal_bytes(before[key], after[key])


def test_empty_pools_preserve_schema_and_zero_denominator_shapes():
    arrays = e.pool_arrays([], [])
    assert arrays["pose_xyz"].shape == (0, 3, 3, 3)
    assert arrays["sources"].shape == (0, 3, 3, 3)
    assert arrays["observed"].shape == (0, 3, 3)
    assert arrays["angle_deg"].shape == (0, 3)


def test_case_denominator_and_other_fish_not_counted_as_target_success():
    cases = [
        dict(frame_idx=t, slot=0, free_only_count=int(t < 100), free_and_bounded_count=int(t < 150))
        for t in range(215)
    ]
    rows = [
        dict(arm=arm, target_lost_fish=target, seed_method="free_intersection")
        for arm in e.ARMS
        for target in (True, False, False)
    ]
    summary = e.summarize_cases(cases, rows)
    assert summary["free_only"]["cases_with_candidates"] == 100
    assert summary["free_and_bounded"]["cases_with_candidates"] == 150
    assert summary["free_only"]["target_lost_proposals"] == 1
    assert summary["free_only"]["other_fish_proposals"] == 2
    assert summary["incremental_case_availability"] == dict(
        only_with_bounded=50, both=100, neither=65
    )
    with pytest.raises(ValueError, match="fixed215"):
        e.summarize_cases(cases[:-1], rows)
    cases[-1] = cases[0].copy()
    with pytest.raises(ValueError, match="fixed215"):
        e.summarize_cases(cases, rows)
