"""Synthetic byte/context/input-life-cycle checks; no fits or raw observations."""

import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import qualify_tail_tracker as q
from tracker import Proposal


def fixture():
    pose = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    proposals = [Proposal(pose.copy(), pose.copy(), np.zeros((3, 3, 3), np.int32), 0, "predict")]
    return proposals, q.packets.packet_proposals(proposals)


def test_exact_off_prefix_and_full_count_are_not_interchangeable():
    props, expected = fixture()
    props.append(
        Proposal(
            props[0].pose.copy(),
            props[0].predicted.copy(),
            props[0].plan.copy(),
            0,
            "tail_recovery",
        )
    )
    all_fields = q.packets.packet_proposals(props)
    q.validate_proposals(props, all_fields, 2)
    q.validate_proposals(props[:1], all_fields, 1)
    with pytest.raises(ValueError, match="count"):
        q.validate_proposals(props, expected, 1)


@pytest.mark.parametrize("field", q.PROPOSAL_KEYS)
def test_each_full_proposal_field_is_checked_byte_exact(field):
    props, expected = fixture()
    expected[field].flat[0] = "changed" if expected[field].dtype.kind == "U" else 999
    with pytest.raises(ValueError, match="bytes differ"):
        q.validate_proposals(props, expected, 1)


def test_numeric_dtype_is_not_tolerance_qualified():
    props, expected = fixture()
    expected["proposal_pose_xyz"] = expected["proposal_pose_xyz"].astype(np.float32)
    with pytest.raises(ValueError, match="bytes differ"):
        q.validate_proposals(props, expected, 1)


def context():
    expected = dict(
        implementation="Bounded",
        tracker_config={"device": "cpu", "ray_sigma_mm": 2.0},
        bounded_mouth_policy={"enabled": True},
        persistence_config={"minimum_missing_frames": 4},
    )
    new = copy.deepcopy(expected)
    new["implementation"] = "Tail"
    new["tail_recovery_policy"] = dict(
        schema="tail_recovery_continuous_policy_v1",
        enabled=True,
        target_node=2,
        allow_bounded=False,
        clears_identity_review=False,
    )
    return new, expected


def test_only_explicit_tail_policy_and_implementation_change_without_mutation():
    new, expected = context()
    before = copy.deepcopy((new, expected))
    q.validate_context(new, expected, True)
    assert (new, expected) == before
    new["tail_recovery_policy"]["enabled"] = False
    q.validate_context(new, expected, False)


@pytest.mark.parametrize(
    "fault",
    ["tail_bound", "tail_node", "tail_enabled", "review", "existing_mouth", "threshold", "age"],
)
def test_changed_scientific_or_unclaimed_policy_rejected(fault):
    new, expected = context()
    if fault == "tail_bound":
        new["tail_recovery_policy"]["allow_bounded"] = True
    if fault == "tail_node":
        new["tail_recovery_policy"]["target_node"] = 0
    if fault == "tail_enabled":
        new["tail_recovery_policy"]["enabled"] = False
    if fault == "review":
        new["tail_recovery_policy"]["clears_identity_review"] = True
    if fault == "existing_mouth":
        new["bounded_mouth_policy"]["enabled"] = False
    if fault == "threshold":
        new["tracker_config"]["ray_sigma_mm"] = 3
    if fault == "age":
        new["persistence_config"]["minimum_missing_frames"] = 3
    with pytest.raises(ValueError):
        q.validate_context(new, expected, True)


def test_every_parent_ray_flow_field_participates_in_signature():
    parents = [
        SimpleNamespace(**{k: np.zeros(3) for k in q.packets.STATE_FIELDS}) for _ in range(6)
    ]
    rays = SimpleNamespace(
        points={v: np.zeros((2, 3, 2)) for v in q.VIEWS},
        origins=tuple(np.zeros((2, 3, 3)) for _ in q.VIEWS),
        directions=tuple(np.ones((2, 3, 3)) for _ in q.VIEWS),
    )
    flow = q.FlowObservations(
        **{f.name: np.zeros((3, 3)) for f in dataclasses.fields(q.FlowObservations)}
    )
    signature = q.context_input_signature(parents, rays, flow)
    assert len(signature) == 7 + 9 + len(dataclasses.fields(flow))
    changed = copy.deepcopy(flow)
    changed.pose_xyz.flat[0] = 1
    assert q.context_input_signature(parents, rays, changed) != signature
    changed_rays = copy.deepcopy(rays)
    changed_rays.origins[0].flat[0] = 1
    assert q.context_input_signature(parents, changed_rays, flow) != signature
    changed_parents = copy.deepcopy(parents)
    changed_parents[5].review.flat[0] = 1
    assert q.context_input_signature(changed_parents, rays, flow) != signature


def rows():
    result = [dict(frame_idx=f, original_count=14, total_count=16) for f in q.FRAMES]
    result[0]["original_count"] += 4273 - 14 * 301
    result[0]["total_count"] += 4949 - 16 * 301
    return result


def test_fixed301_order_and_global4273_plus676_totals():
    result = rows()
    assert q.validate_totals(result) == (4273, 676, 4949)
    result[0]["frame_idx"] += 1
    with pytest.raises(ValueError, match="sequence"):
        q.validate_totals(result)
    result = rows()
    result[-1]["total_count"] += 1
    with pytest.raises(ValueError, match="totals"):
        q.validate_totals(result)
