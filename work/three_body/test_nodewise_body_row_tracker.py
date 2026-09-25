"""Synthetic nodewise tracker contracts; no scientific readers or GPU fitting.

Reuse the eighteen frozen whole-row contracts in a separate module namespace.
One extra integration case runs the real nodewise
pool, scene construction, strict support and temporal update with a fixed CPU
optimizer result. This does not qualify an actual fitted trajectory.
"""

# ruff: noqa: E402

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import nodewise_body_row_tracker as target
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_nodewise_body_row_contracts", Path(__file__).with_name("test_body_row_tracker.py")
)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)
contract.m = target

# These factories allocate only synthetic in-memory camera/state arrays. The
# original autouse guard blocks np.load, public rollouts and both GPU fitters.
inputs = contract.inputs
checkpoint = contract.checkpoint
forbid_real_execution = contract.forbid_real_execution


# Preserve the original parametrizations and assertions without editing their
# source or mutating the normally imported old test module's reference.
for _name, _function in vars(contract).items():
    if _name.startswith("test_") and callable(_function):
        globals()[_name] = _function


def test_nodewise_full_body_keeps_distinct_view_subsets_through_actual_support(
    monkeypatch, inputs, checkpoint
):
    config = target.BodyRowConfig(per_slot_limit=1, scene_limit=1)
    tracker = contract.make(inputs, body_row_config=config)
    target.import_sparse_state(
        tracker,
        contract.sparse.SparseInstanceRescoreContinuousTracker(*inputs),
        *copy.deepcopy(checkpoint),
    )
    policy = tracker._context()["body_row_policy"]
    assert policy["nodewise_body_row_source_sha256"] == (
        "ce8c3f7716ef36d87f1dc32e10faa4686575d5ed78b2f102064a78a5d9577f06"
    )
    assert policy["legacy_pool_prefix_preserved"]
    assert policy["supplemental_nodes_all_triangulated_without_template"]
    assert policy["excluded_views_are_not_verified"]

    rig, pose, _ = inputs
    points = {v: rig.camera(v).project(pose).copy() for v in contract.VIEWS}
    # The three synthetic cameras project XY, XZ and YZ. Mouth needs views01;
    # tail needs views12, while the chest is consistent in all three cameras.
    points[contract.VIEWS[2]][0, 0] += 40.0
    points[contract.VIEWS[0]][0, 2] += 40.0
    before = {v: p.tobytes() for v, p in points.items()}
    images = {v: np.zeros((192, 192, 3), np.uint8) for v in contract.VIEWS}
    frame = contract.FrameObservation(
        contract.NATIVE,
        points,
        images,
        {v: np.empty((0, 192, 192), bool) for v in contract.VIEWS},
        images,
    )
    original = [
        contract.Proposal(pose.copy(), pose.copy(), contract.plans(), 0, "predict", (0, 1, 2))
    ]

    def inherited_proposals(self, states, rays, flow):
        self._recovery_calls += 1
        return original

    monkeypatch.setattr(
        contract.sparse.SparseInstanceRescoreContinuousTracker,
        "_proposals",
        inherited_proposals,
    )
    monkeypatch.setattr(contract.continuous, "sample_pixels", lambda *args: object())
    monkeypatch.setattr(
        contract.continuous.FlowObservations,
        "from_frames",
        lambda *args: SimpleNamespace(node_valid=np.zeros((3, 3), bool)),
    )
    expected = contract.plans()
    expected[0, 2, 0] = -1
    expected[0, 0, 2] = -1
    fitted_plans = []

    def fixed_optimizer_result(self, proposals, states, rays, targets):
        assert self is tracker and proposals[0] is original[0] and len(proposals) == 2
        assert proposals[1].kind == "body_row" and proposals[1].parent == 0
        np.testing.assert_allclose(proposals[1].pose, pose, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(proposals[1].plan, expected)
        fitted_plans.append(proposals[1].plan.copy())
        return (
            np.stack([p.pose for p in proposals]),
            np.array([100.0, 0.0]),
            np.zeros((2, 5)),
            np.zeros((2, 3), bool),
        )

    monkeypatch.setattr(
        contract.sparse.FastTailRecoveryContinuousTracker, "_fit", fixed_optimizer_result
    )
    monkeypatch.setattr(
        contract.sparse.FastTailRecoveryContinuousTracker,
        "_advance_owned",
        contract.FROZEN_ADVANCE_OWNED,
    )
    tracker._advance_owned(frame)

    record = tracker._history[-1][0]
    assert len(fitted_plans) == 1 and record["proposal_kind"] == "body_row"
    np.testing.assert_array_equal(record["sources"], expected)
    np.testing.assert_array_equal(record["observed"], np.ones((3, 3), bool))
    np.testing.assert_array_equal(record["node_missing"], np.zeros((3, 3), int))
    np.testing.assert_allclose(record["pose_xyz"], pose, rtol=0, atol=1e-12)
    assert record["identity_review"][2]  # no clearing of inherited review
    assert {v: p.tobytes() for v, p in points.items()} == before

    diagnostic = tracker.export_body_row_diagnostics()
    assert diagnostic["frame_completed"] and diagnostic["added_proposal_count"] == 1
    assert not diagnostic["filled_nodes_by_proposal"].any()
    assert diagnostic["pool_stats"]["additional_candidate_count"] > 0
    selected = diagnostic["scenes"][0]["pool_indices"][0]
    body = diagnostic["pool"][selected]
    assert body["supplemental_nodewise_body"] and body["triangulated_nodes"].all()
    np.testing.assert_array_equal(body["proposed_sources"], expected[0])
    assert body["reprojection_error_px"][2, 0] > 40
    assert body["reprojection_error_px"][0, 2] > 40
    assert body["excluded_available_node_views"][0, 2]
    assert body["excluded_available_node_views"][2, 0]
    assert body["direct_support_not_recomputed"]  # pool remains proposal evidence
    assert tracker._body_row_frame is tracker._instance_frame is None
