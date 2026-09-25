"""Strict comparison of generated proposals, with no fit or GPU calls."""

import numpy as np
import pytest
import qualify_bounded_mouth_tracker as q
from tracker import Proposal


def fixture():
    pose = np.arange(27, dtype=float).reshape(3, 3, 3)
    props = [Proposal(pose.copy(), pose.copy(), np.zeros((3, 3, 3), int), 0, "predict")]
    return props, q.packets.packet_proposals(props)


def test_exact_pass_and_short_count_rejected():
    props, expected = fixture()
    q.validate_proposals(props, expected, 1)
    with pytest.raises(ValueError, match="count"):
        q.validate_proposals(props, expected, 0)


@pytest.mark.parametrize("field", q.PROPOSAL_KEYS)
def test_any_proposal_field_change_is_not_qualified(field):
    props, expected = fixture()
    expected[field].flat[0] = "changed" if expected[field].dtype.kind == "U" else 999
    with pytest.raises(ValueError, match="bytes differ"):
        q.validate_proposals(props, expected, 1)


def test_numeric_dtype_change_not_tolerance_accepted():
    props, expected = fixture()
    expected["proposal_pose_xyz"] = expected["proposal_pose_xyz"].astype(np.float32)
    with pytest.raises(ValueError, match="bytes differ"):
        q.validate_proposals(props, expected, 1)
