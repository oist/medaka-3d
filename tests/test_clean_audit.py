"""Audit distinguishes observed uncertainty and only permits recorded joins."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def audit():
    lab = Path(__file__).resolve().parents[1] / "labs/20260917_clean_tracklets"
    if not (lab / "audit.py").exists():
        pytest.skip("clean tracklet lab retired")
    sys.path.insert(0, str(lab))
    try:
        spec = importlib.util.spec_from_file_location("_clean_audit", lab / "audit.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(lab))
    return module


def test_only_accepted_joins_in_this_view_can_change_components(audit):
    labels = np.array([10, 90, 10, 90, 20, 90, 30, 90, 40, 90])
    events = [
        dict(accepted=True, view="top_cam", left_label=10, right_label=20),
        dict(accepted=True, view="top_cam", left_label=20, right_label=30),
        dict(accepted=False, view="top_cam", left_label=30, right_label=40),
        dict(accepted=True, view="side1_cam", left_label=30, right_label=40),
    ]
    joined = audit.expected_joined_labels(labels, "top_cam", events)
    np.testing.assert_array_equal(joined, [10, 90, 10, 90, 10, 90, 10, 90, 40, 90])
    np.testing.assert_array_equal(labels, [10, 90, 10, 90, 20, 90, 30, 90, 40, 90])


def test_two_accepted_joins_cannot_claim_the_same_endpoint(audit):
    events = [
        dict(accepted=True, view="top_cam", left_label=10, right_label=right) for right in [20, 30]
    ]
    with pytest.raises(AssertionError):
        audit.expected_joined_labels(np.array([10, 20, 30]), "top_cam", events)


def test_observed_uncertain_is_separated_from_review_flags_on_missing_rows(audit):
    xyz = np.zeros((3, 3, 3))
    xyz[1, 2] = np.nan
    xyz[2, 0] = np.nan
    review = np.array([[True, True, True], [False, True, True], [True, False, False]])
    counts = audit.observation_identity_counts(
        dict(xyz=xyz, identity_review=review), np.array([False, True, True])
    )
    assert counts["possible_fish_frames"] == 6
    assert counts["missing_fish_frames"] == 2
    assert counts["observed_but_identity_review_fish_frames"] == 1
    assert counts["observed_and_confident_fish_frames"] == 3
