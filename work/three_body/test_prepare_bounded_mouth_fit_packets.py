"""Synthetic packet arithmetic and explicit source-mismatch rejection only."""

from types import SimpleNamespace

import numpy as np
import prepare_bounded_mouth_fit_packets as m
import pytest


def fixture_pool():
    pose = np.arange(27, dtype=float).reshape(3, 3, 3)
    plan = np.arange(27).reshape(3, 3, 3)
    original = [m.Proposal(pose.copy(), pose.copy(), plan.copy(), p, "predict") for p in range(6)]
    states = [
        SimpleNamespace(
            node_missing=np.array([[4, 0, 0], [3, 0, 0], [9, 0, 0]]),
            direct_pectoral_age=np.array([4, 9, 3]),
        )
        for _ in range(6)
    ]
    result = SimpleNamespace(
        proposals=[
            SimpleNamespace(pose_xyz=pose + i + 1, plan=plan.copy(), fish=fish)
            for i, fish in enumerate([0, 1, 2, 0])
        ]
    )
    return original, states, result


def test_original_order_bytes_identity_and_owned_append():
    original, states, result = fixture_pool()
    before = m.signature(m.packet_proposals(original))
    proposals, info, fish = m.append_bounded(original, states, result, m.PersistentNodeConfig(), 8)
    assert len(proposals) == 8 and info["after_age"] == 2
    assert all(proposals[i] is original[i] for i in range(6))
    assert before == m.signature(m.packet_proposals(original))
    assert fish.tolist() == [-1] * 6 + [0, 0]
    assert all(p.parent == 0 and p.permutation == (0, 1, 2) for p in proposals[6:])
    assert all(p.kind == "bounded_mouth_recovery" for p in proposals[6:])
    proposals[6].pose.fill(-999)
    assert np.all(result.proposals[0].pose_xyz >= 1)
    assert before == m.signature(m.packet_proposals(original))


@pytest.mark.parametrize("cap,admitted,dropped", [(6, 0, 2), (7, 1, 1), (8, 2, 0)])
def test_budget_only_removes_added(cap, admitted, dropped):
    original, states, result = fixture_pool()
    proposals, info, _ = m.append_bounded(original, states, result, m.PersistentNodeConfig(), cap)
    assert len(proposals) == 6 + admitted
    assert info["excluded_by_budget"] == dropped and info["excluded_by_age"] == 2


def test_disabled_preserves_every_original():
    original, states, result = fixture_pool()
    proposals, info, _ = m.append_bounded(
        original, states, result, m.PersistentNodeConfig(enabled=False), 24
    )
    assert len(proposals) == 6 and info["age_eligible_slots"] == []


@pytest.mark.parametrize("bad", ["parent", "kind", "permutation", "cap"])
def test_reject_wrong_main_or_capacity(bad):
    original, states, result = fixture_pool()
    if bad == "parent":
        original[0].parent = 1
    if bad == "kind":
        original[0].kind = "bounded_mouth_recovery"
    if bad == "permutation":
        original[0].permutation = (1, 0, 2)
    with pytest.raises(ValueError):
        m.append_bounded(
            original, states, result, m.PersistentNodeConfig(), 5 if bad == "cap" else 8
        )


def test_retained_19_fields_native_offset_and_owned_copy():
    state = {"history_offsets": np.array([0, 6, 12])}
    for k in m.RECORD_FIELDS:
        state["history_" + k] = np.arange(12)
    state["history_frame_idx"] = np.repeat([m.comparison.FIRST, m.comparison.FIRST + 1], 6)
    result = m.retained_records(state, m.comparison.FIRST + 1)
    assert set(result) == set(m.RECORD_FIELDS)
    assert "permutation" not in result and "plan" not in result
    assert result["parent"].tolist() == list(range(6, 12))
    result["parent"].fill(-1)
    assert np.all(state["history_parent"] >= 0)
    state["history_frame_idx"][-1] += 1
    with pytest.raises(ValueError, match="native"):
        m.retained_records(state, m.comparison.FIRST + 1)


def test_pool_history_counts_flow_and_parent_kind():
    original, _, _ = fixture_pool()
    flow = SimpleNamespace(node_valid=np.zeros((3, 3), bool))
    records = dict(
        hypotheses=np.full(6, 6),
        proposal_flow_parent=np.zeros(6, int),
        proposal_flow_nodes=np.zeros((6, 3, 3), bool),
        parent=np.arange(6),
        proposal_kind=np.full(6, "predict"),
    )
    m.validate_pool_history(original, flow, records)
    for key in ("hypotheses", "proposal_flow_nodes", "proposal_flow_parent", "parent"):
        bad = {k: v.copy() for k, v in records.items()}
        bad[key].flat[-1] = 10
        with pytest.raises(ValueError):
            m.validate_pool_history(original, flow, bad)


def test_native_mask_and_image_hashes_not_only_metadata():
    image = np.zeros((2, 2, 3), np.uint8)
    masks = np.zeros((0, 2, 2), bool)
    frame = SimpleNamespace(
        frame_idx=10,
        images_bgr={v: image.copy() for v in m.VIEWS},
        masks={v: masks.copy() for v in m.VIEWS},
        backgrounds_bgr={v: image.copy() for v in m.VIEWS},
        detections={v: np.empty((0, 3, 2)) for v in m.VIEWS},
    )
    reader = SimpleNamespace(
        frames=np.array([10]),
        pixel_hashes={v: [m.array_sha256(image)] for v in m.VIEWS},
        masks={
            v: dict(
                view_available=np.array([True]),
                frame_mask_sha256=np.array([m.array_sha256(masks)], "S64"),
            )
            for v in m.VIEWS
        },
    )
    assert len(m.observation_evidence(frame, reader)) == 3
    frame.images_bgr[m.VIEWS[0]][0, 0, 0] = 1
    with pytest.raises(ValueError, match="BGR"):
        m.observation_evidence(frame, reader)
    frame.images_bgr[m.VIEWS[0]].fill(0)
    frame.masks[m.VIEWS[0]] = np.ones((1, 2, 2), bool)
    with pytest.raises(ValueError, match="mask SHA"):
        m.observation_evidence(frame, reader)
