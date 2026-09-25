"""Synthetic native-span, state-proof and original-prefix packet contracts."""

from types import SimpleNamespace

import numpy as np
import prepare_tail_fit_packets as m
import pytest
from observations import FrameObservation
from test_tail_recovery import pose, rig, setup  # noqa: F401


def pool():
    p = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    plan = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
    original = [m.Proposal(p.copy(), p.copy(), plan.copy(), i, "predict") for i in range(6)]
    states = [
        SimpleNamespace(
            node_missing=np.array([[0, 0, 4], [9, 0, 3], [0, 0, 8]]),
            direct_pectoral_age=np.array([4, 9, 3]),
        )
        for _ in range(6)
    ]
    return original, states


def fake_result(original):
    return SimpleNamespace(
        proposals=[
            SimpleNamespace(
                pose_xyz=original[0].pose + j,
                plan=original[0].plan.copy(),
                fish=f,
                node=2,
                seed_method="free_intersection",
            )
            for j, f in enumerate([0, 1, 2, 0])
        ],
        diagnostics={"synthetic": True},
    )


@pytest.mark.parametrize("capacity,count,dropped", [(6, 0, 2), (7, 1, 1), (8, 2, 0)])
def test_original_prefix_and_age_budget_only_remove_extras(monkeypatch, capacity, count, dropped):
    original, states = pool()
    result = fake_result(original)
    calls = []

    def helper(*args, **kw):
        calls.append((args, kw))
        return result

    monkeypatch.setattr(m.tail, "tail_recovery_proposals", helper)
    before = m.packets.signature(m.packets.packet_proposals(original))
    proposals, detail, fish = m.append_tail(
        original,
        states,
        "rays",
        "rig",
        "lengths",
        m.NodeRecoveryConfig(),
        m.PersistentNodeConfig(),
        capacity,
    )
    assert len(proposals) == 6 + count and detail["after_age"] == 2
    assert detail["excluded_by_age"] == 2 and detail["excluded_by_budget"] == dropped
    assert all(a is b for a, b in zip(original, proposals[:6], strict=True))
    assert before == m.packets.signature(m.packets.packet_proposals(original))
    assert len(calls) == 1 and calls[0][1] == {"allow_bounded": False}
    assert calls[0][0][0] is original[0].pose and calls[0][0][1] is original[0].plan
    assert fish.tolist() == [-1] * 6 + [0] * count
    for p in proposals[6:]:
        assert p.parent == 0 and p.permutation == (0, 1, 2) and p.kind == "tail_recovery"
        assert not np.shares_memory(p.predicted, original[0].predicted)
        assert not np.shares_memory(p.pose, result.proposals[0].pose_xyz)


@pytest.mark.parametrize("reason", ["disabled", "tail_age", "chest_age", "only_other_parent"])
def test_closed_state0_gate_never_calls_helper(monkeypatch, reason):
    original, states = pool()
    cfg = m.PersistentNodeConfig(enabled=reason != "disabled")
    if reason in ("tail_age", "only_other_parent"):
        states[0].node_missing[:, 2] = 0
    if reason == "chest_age":
        states[0].direct_pectoral_age[:] = 0
    monkeypatch.setattr(
        m.tail, "tail_recovery_proposals", lambda *a, **k: pytest.fail("closed gate")
    )
    proposals, info, fish = m.append_tail(
        original, states, None, None, None, m.NodeRecoveryConfig(), cfg, 24
    )
    assert len(proposals) == 6 and not info["helper_called"] and not info["admitted"]
    assert fish.tolist() == [-1] * 6


@pytest.mark.parametrize("fault", ["parent", "kind", "permutation", "capacity"])
def test_wrong_original_main_rejected(fault):
    original, states = pool()
    if fault == "parent":
        original[0].parent = 1
    if fault == "kind":
        original[0].kind = "tail_recovery"
    if fault == "permutation":
        original[0].permutation = (1, 0, 2)
    with pytest.raises(ValueError):
        m.append_tail(
            original,
            states,
            None,
            None,
            None,
            m.NodeRecoveryConfig(),
            m.PersistentNodeConfig(),
            5 if fault == "capacity" else 24,
        )


def test_physical_free_tail_append_preserves_pose_plan_and_strict_support(rig, pose):  # noqa: F811
    scene, plan, rays, lengths, *_ = setup(rig, pose)
    original = [m.Proposal(scene.copy(), scene.copy(), plan.copy(), i, "predict") for i in range(6)]
    states = [
        SimpleNamespace(node_missing=np.full((3, 3), 4), direct_pectoral_age=np.full(3, 4))
        for _ in range(6)
    ]
    before = m.packets.signature(m.packets.packet_proposals(original))
    proposals, info, fish = m.append_tail(
        original, states, rays, rig, lengths, m.NodeRecoveryConfig(), m.PersistentNodeConfig(), 24
    )
    assert info["admitted"] > 0 and not info["allow_bounded"]
    assert before == m.packets.signature(m.packets.packet_proposals(original))
    for p, f in zip(proposals[6:], fish[6:], strict=True):
        assert np.array_equal(p.pose[:, :2], scene[:, :2])
        assert np.array_equal(p.plan[:, :, :2], plan[:, :, :2])
        assert m.supported_nodes(p.pose, p.plan, rays, rig)[1][f, 1:].all()


def state_fixture():
    parents = {k: np.zeros((301, 6), np.float64) for k in m.STATE_FIELDS}
    records = {k: np.zeros((301, 6), np.float64) for k in m.RECORD_FIELDS}
    parents.update(frame_idx=m.FRAMES.copy(), previous_frame_idx=m.FRAMES - 1)
    records.update(
        selected_native_frames=m.FRAMES.copy(),
        frame_idx=np.repeat(m.FRAMES[:, None], 6, axis=1),
        history_row_indices=np.arange(301 * 6, dtype=np.int64).reshape(301, 6),
    )
    summary = {
        "outputs": {
            name: {"arrays": m.packets.signature({k: data[k] for k in keys})}
            for data, keys, name in [
                (parents, m.STATE_FIELDS, "previous_states"),
                (records, m.RECORD_FIELDS, "retained_records"),
            ]
        }
    }
    return parents, records, summary


def test_sealed_parent_and_retained_native_axes_are_distinct_owned():
    parents, records, summary = state_fixture()
    p, r = m.validate_saved_parents(parents, records, summary)
    assert set(p) == set(m.STATE_FIELDS) and len(r) == 301 and set(r[0]) == set(m.RECORD_FIELDS)
    p["pose"].fill(123)
    r[0]["pose_xyz"].fill(321)
    assert not parents["pose"].any() and not records["pose_xyz"].any()


@pytest.mark.parametrize("fault", ["previous", "dtype", "state", "record", "rows", "extra"])
def test_parent_evidence_rejects_native_byte_or_schema_tamper(fault):
    p, r, s = state_fixture()
    if fault == "previous":
        p["previous_frame_idx"][0] += 1
    if fault == "dtype":
        p["frame_idx"] = p["frame_idx"].astype(np.uint64)
    if fault == "state":
        p["velocity"][0, 0] += 1
    if fault == "record":
        r["pose_xyz"][0, 0] += 1
    if fault == "rows":
        r["history_row_indices"][0, -1] += 1
    if fault == "extra":
        p["invented"] = np.zeros(1)
    with pytest.raises(ValueError):
        m.validate_saved_parents(p, r, s)


def frame(native):
    return FrameObservation(
        native,
        {v: np.empty((0, 3, 2)) for v in m.VIEWS},
        {v: np.zeros((2, 2, 3), np.uint8) for v in m.VIEWS},
        {v: np.empty((0, 2, 2), bool) for v in m.VIEWS},
        {v: np.zeros((2, 2, 3), np.uint8) for v in m.VIEWS},
    )


def test_two_native_spans_have_no_gap_decode_or_extra_frame_and_close(monkeypatch):
    starts, closed, delivered = [], [], []

    def fake(reader, start):
        starts.append(start)
        try:
            for n in range(start, start + 1000):
                delivered.append(n)
                yield frame(n)
        finally:
            closed.append(start)

    monkeypatch.setattr(m, "iter_from", fake)
    raw = []
    pairs = list(m.raw_pairs(None, lambda x: raw.append(x.frame_idx)))
    assert starts == closed == [66523, 66619]
    assert raw == delivered == list(range(66523, 66525)) + list(range(66619, 66920))
    assert len(raw) == 303 and [b.frame_idx for a, b in pairs] == m.FRAMES.tolist()
    assert all(a.frame_idx + 1 == b.frame_idx for a, b in pairs)


@pytest.mark.parametrize("fault", ["previous", "current", "EOF", "consumer_abort"])
def test_native_failures_and_consumer_close_release_iterator(monkeypatch, fault):
    closed = []

    def fake(reader, start):
        try:
            yield frame(start + int(fault == "previous"))
            if fault == "EOF":
                return
            yield frame(start + 1 + int(fault == "current"))
            yield frame(start + 2)
        finally:
            closed.append(start)

    monkeypatch.setattr(m, "iter_from", fake)
    iterator = m.raw_pairs(None, lambda x: None, spans=((10, 13),))
    if fault == "consumer_abort":
        next(iterator)
        iterator.close()
    else:
        with pytest.raises(ValueError):
            list(iterator)
    assert closed == [10]
