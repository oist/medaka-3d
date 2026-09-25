"""View ambiguity must not erase independent shared two-view measurements."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from medaka3d.types import VIEWS


@pytest.fixture(scope="module")
def modules():
    root = Path(__file__).resolve().parents[1]
    lab = root / "labs/20260917_clean_tracklets"
    if not (lab / "multiview_consensus.py").exists():
        pytest.skip("clean tracklet experiment retired")
    sys.path.insert(0, str(lab))
    try:
        spec = importlib.util.spec_from_file_location(
            "_clean_consensus", lab / "multiview_consensus.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(lab))
    helpers = runpy.run_path(str(root / "tests/test_clean_multiview.py"))
    return module, module.base, helpers


def duplicate(data, frame, label, extra_labels):
    row = np.flatnonzero((data["frame_idx"] == frame) & (data["labels"] == label))[0]
    for key in ("frame_idx", "points"):
        data[key] = np.concatenate(
            [data[key], np.repeat(data[key][row : row + 1], len(extra_labels), 0)]
        )
    data["labels"] = np.r_[data["labels"], extra_labels]
    data["clean_edge"] = np.r_[data["clean_edge"], np.zeros(len(extra_labels), dtype=bool)]


def test_third_view_competitors_reproduce_v1_overabstention_and_preserve_common_pair(modules):
    consensus, old, helpers = modules
    frames, pose, rig, views = helpers["scene"](6)
    duplicate(views[VIEWS[2]], 102, 1, [90, 91])
    baseline = old.reconstruct(frames, views, rig)
    changed = consensus.reconstruct(frames, views, rig)
    assert np.isnan(baseline["xyz"][2, 1]).all()
    np.testing.assert_allclose(changed["xyz"], pose[:, :, 1], atol=1e-9)
    np.testing.assert_array_equal(changed["source"][2, 1], [1, 1, -1])
    assert np.isnan(changed["uv"][2, 1, 2]).all()
    assert np.isfinite(changed["projected_uv"][2, 1, 2]).all()
    assert changed["consensus_preserved"][2, 1]
    assert changed["withheld_view"][2, 1, 2]
    assert changed["competitive_joint_state_count"][2] >= 3
    assert changed["assignment_margin_mm"][2, 1] < old.MultiViewConfig().correspondence_margin_mm
    assert len(changed["labels_by_view"][VIEWS[2]]) == 20
    assert not changed["identity_review"].any()


def test_clear_geometry_and_missing_third_view_preserve_v1_outputs(modules):
    consensus, old, helpers = modules
    frames, _, rig, views = helpers["scene"](7)
    data = views[VIEWS[2]]
    helpers["split"](views, VIEWS[2], 104)
    keep = ~((data["frame_idx"] == 103) & (data["labels"] == 1))
    for key in data:
        data[key] = data[key][keep]
    baseline = old.reconstruct(frames, views, rig)
    changed = consensus.reconstruct(frames, views, rig)
    for key in ("xyz", "pose", "source", "uv", "identity_review", "assignment_margin_mm"):
        np.testing.assert_array_equal(changed[key], baseline[key])
    assert not changed["consensus_preserved"].any()
    assert consensus.MultiViewConfig is old.MultiViewConfig


def test_ambiguity_in_two_views_does_not_leave_an_invented_single_view_point(modules):
    consensus, _, helpers = modules
    frames, _, rig, views = helpers["scene"](5)
    # Reset the fish's clean fragments at the crossing, so history does not
    # prejudge either duplicate. One unambiguous view cannot measure 3D.
    for view in VIEWS:
        helpers["split"](views, view, 102)
    for view in VIEWS[1:]:
        duplicate(views[view], 102, 17, [99])
    changed = consensus.reconstruct(frames, views, rig)
    assert np.isnan(changed["xyz"][2, 1]).all()
    assert (changed["source"][2, 1] == -1).all()
    assert changed["identity_review"][2, 1]


def test_exact_objective_winners_and_margins_match_frozen_v1(modules):
    consensus, old, helpers = modules
    frames, pose, rig, views = helpers["scene"](6)
    duplicate(views[VIEWS[2]], 103, 1, [99])
    config = old.MultiViewConfig()
    prepared = old._prepare(frames, views, old._cameras(rig))
    for ti in (0, 1, 3, 5):
        h = old._hypotheses(frames[ti], prepared, config)
        histories = [
            ([] if ti == 0 else [(int(frames[ti - 1]), pose[ti - 1, s, 1])]) for s in range(3)
        ]
        owners = {v: {0: 0, 1: 1, 2: 2} for v in VIEWS}
        votes = old._vote_counts(h, prepared, owners)
        best, margin, _, conflicts, count = old._assignment(h, histories, votes, frames[ti], config)
        decision = consensus._consensus(h, histories, votes, frames[ti], config)
        np.testing.assert_array_equal(decision["best"], best)
        np.testing.assert_array_equal(decision["margins"], margin)
        assert decision["conflicts"] == conflicts and decision["state_count"] == count


def test_competing_fish_identities_have_no_shared_pair_and_remain_unresolved(modules):
    consensus, old, _ = modules
    h = dict(
        source=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        pose=np.zeros((3, 3, 3)),
        cost=np.zeros(3),
    )
    h["pose"][:, 1] = [[49, 50, 50], [51, 50, 50], [80, 80, 80]]
    histories = [
        [(99, np.array(p, dtype=float))] for p in ([50, 50, 50], [50, 50, 50], [80, 80, 80])
    ]
    decision = consensus._consensus(h, histories, np.zeros((3, 3), int), 100, old.MultiViewConfig())
    np.testing.assert_array_equal(decision["selected"], [3, 3, 2])
    assert not decision["preserved"].any()


def test_third_view_presence_alternative_keeps_the_same_measured_pair(modules):
    consensus, old, _ = modules
    h = dict(
        source=np.array([[0, 0, 0], [0, 0, -1], [1, 1, 1], [2, 2, 2]]),
        pose=np.zeros((4, 3, 3)),
        cost=np.array([1.4, 1.5, 0.0, 0.0]),
    )
    h["pose"][:, 1] = [[40, 40, 40], [40, 40, 40], [70, 70, 70], [90, 90, 90]]
    histories = [
        [(99, np.array(p, dtype=float))] for p in ([40, 40, 40], [70, 70, 70], [90, 90, 90])
    ]
    decision = consensus._consensus(h, histories, np.zeros((4, 3), int), 100, old.MultiViewConfig())
    assert decision["selected"][0] == 1
    assert decision["preserved"][0]
    np.testing.assert_array_equal(decision["common_source"][0], [0, 0, -1])


def test_common_pair_must_pass_its_own_physical_gate(modules):
    consensus, old, _ = modules
    h = dict(
        source=np.array([[0, 0, 0], [0, 0, 1], [0, 0, -1], [1, 1, 2], [2, 2, 3]]),
        pose=np.zeros((5, 3, 3)),
        cost=np.array([0.0, 0.0, 1.5, 0.0, 0.0]),
    )
    h["pose"][:, 1] = [[40, 40, 40], [40, 40, 40], [55, 40, 40], [70, 70, 70], [90, 90, 90]]
    histories = [
        [(99, np.array(p, dtype=float))] for p in ([40, 40, 40], [70, 70, 70], [90, 90, 90])
    ]
    decision = consensus._consensus(h, histories, np.zeros((5, 3), int), 100, old.MultiViewConfig())
    assert decision["selected"][0] == 5


def test_row_order_invariance_in_ambiguous_third_view(modules):
    consensus, _, helpers = modules
    frames, _, rig, views = helpers["scene"](5)
    duplicate(views[VIEWS[2]], 102, 1, [99])
    original = consensus.reconstruct(frames, views, rig)
    rng = np.random.default_rng(60)
    for data in views.values():
        perm = rng.permutation(len(data["labels"]))
        for key in data:
            data[key] = data[key][perm]
    shuffled = consensus.reconstruct(frames, views, rig)
    np.testing.assert_allclose(shuffled["xyz"], original["xyz"], atol=1e-9)
    np.testing.assert_array_equal(shuffled["uv"], original["uv"])
    np.testing.assert_array_equal(shuffled["consensus_preserved"], original["consensus_preserved"])
