"""Synthetic CPU tail policy and 39-array migration; no fit, advance or real data."""

# ruff: noqa: E402

import copy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
    str(Path(__file__).resolve().parents[2] / "src"),
]

import numpy as np
import prepare_tail_fit_packets as reference
import pytest
import tail_recovery_tracker as m
import test_directional_chest_tracker as toy
from continuous_tracker import RECORD_FIELDS, SCHEMA, STATE_FIELDS, ContinuousPersistentNodeTracker
from node_geometry import RayObservations, supported_nodes
from persistent_node_tracker import PersistentNodeConfig, PersistentNodeState
from tracker import Proposal

from medaka3d.types import VIEWS


@pytest.fixture(scope="module")
def inputs():
    return toy.inputs.__wrapped__()


@pytest.fixture(autouse=True)
def forbid_temporal_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic policy/migration tests must never begin, advance or fit")

    for name in ("begin", "advance", "_advance_owned"):
        monkeypatch.setattr(ContinuousPersistentNodeTracker, name, forbidden)
    monkeypatch.setattr(m.BoundedMouthContinuousTracker, "_fit", forbidden)
    monkeypatch.setattr(m.AcceleratedContinuousPersistentNodeTracker, "_fit", forbidden)


@pytest.fixture(scope="module")
def checkpoint(inputs):
    """Construct two retained branches directly; exercise the real frozen decoder."""
    rig, pose, cfg = inputs
    original = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    observed = np.ones((3, 3), bool)
    observed[0, 2] = observed[2, 1] = False
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    sources = np.where(observed[:, None], sources, -1)
    history = []
    for frame in range(5):
        for branch in range(2):
            history.append(
                dict(
                    frame_idx=1000 + frame,
                    pose_xyz=pose.copy(),
                    observed=observed.copy(),
                    sources=sources.copy(),
                    identity_review=np.array([False, False, True]),
                    mask_supported=np.zeros(3, bool),
                    body_supported=np.array([True, True, False]),
                    missing=np.array([0, 0, frame + 1]),
                    costs=float(branch + 1),
                    path_costs=float((frame + 1) * (branch + 1)),
                    loss_parts=np.array([1.0, 0, 0, 0, 0]),
                    score_margin=np.inf,
                    parent=branch if frame else 0,
                    proposal_kind="predict",
                    hypotheses=cfg.max_hypotheses,
                    proposal_flow_nodes=np.zeros((3, 3), bool),
                    proposal_flow_parent=0 if frame else -1,
                    node_missing=np.where(observed, 0, frame + 1),
                    direct_pectoral_age=np.array([frame + 1, frame + 1, 0]),
                )
            )
    arrays = {"history_offsets": np.arange(0, 11, 2, dtype=np.int64)}
    arrays.update(
        {"history_" + key: np.asarray([row[key] for row in history]) for key in RECORD_FIELDS}
    )
    arrays.update(
        {
            "state_pose": np.stack([pose, pose]),
            "state_velocity": np.zeros((2, 3, 3)),
            "state_missing": np.array([[0, 0, 5], [0, 0, 5]]),
            "state_review": np.array([[False, False, True], [False, False, True]]),
            "state_cost": np.array([5.0, 10.0]),
            "state_node_missing": np.stack(
                [history[-2]["node_missing"], history[-1]["node_missing"]]
            ),
            "state_direct_pectoral_age": np.array([[5, 5, 0], [5, 5, 0]]),
        }
    )
    presence = {}
    for vi, view in enumerate(VIEWS):
        detections = rig.camera(view).project(pose)
        detections[~observed] = np.nan
        for field, value in (
            ("detections", detections),
            ("images_bgr", np.full((4, 5, 3), vi + 10, np.uint8)),
            ("masks", np.empty((0, 4, 5), bool)),
            ("backgrounds_bgr", np.full((4, 5, 3), vi + 20, np.uint8)),
        ):
            key = "previous_" + field + "_" + view
            arrays[key], presence[key] = value, True
    diagnostics = copy.deepcopy(original.node_recovery_diagnostics)
    diagnostics["calls"] = 5
    metadata = dict(
        schema=SCHEMA,
        context=original._context(),
        initialize_count=1,
        restore_count=0,
        processed_frames=5,
        previous_frame=1004,
        record_fields=list(RECORD_FIELDS),
        state_fields=list(STATE_FIELDS),
        previous_observation=presence,
        node_recovery_diagnostics=diagnostics,
    )
    assert len(arrays) == 39
    original._decode_state(arrays, metadata)
    assert original.initialize_count == original.processed_frames == original.restore_count == 0
    return arrays, metadata


def policy_setup(monkeypatch, inputs, *, enabled=True, remaining=4, persistence=None):
    tracker = m.TailRecoveryContinuousTracker(
        *inputs, tail_recovery_enabled=enabled, persistence_config=persistence
    )
    pose = inputs[1]
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    count = tracker.config.max_hypotheses + 6 - remaining
    kinds = ("predict", "flow", "chest_recovery", "bounded_mouth_recovery")
    original = [
        Proposal(pose.copy(), pose.copy() + 0.5, plan.copy(), i % 2, kinds[i % 4])
        for i in range(count)
    ]
    cfg = tracker.persistence_config
    first = PersistentNodeState(
        pose.copy(),
        np.zeros((3, 3)),
        np.zeros(3, int),
        np.array([False, True, True]),
        7.0,
        np.array(
            [
                [0, 0, cfg.minimum_missing_frames],
                [99, 0, cfg.minimum_missing_frames - 1],
                [0, 0, 99],
            ]
        ),
        np.array([cfg.minimum_pectoral_frames, 99, cfg.minimum_pectoral_frames - 1]),
    )
    other = copy.deepcopy(first)
    other.node_missing[:, 2] = other.direct_pectoral_age[:] = 99
    states = [first, other]
    result = SimpleNamespace(proposals=[], diagnostics={"synthetic": True})
    for k, fish in enumerate((1, 0, 2, 0)):
        changed_pose, changed_plan = pose.copy(), plan.copy()
        changed_pose[fish, 2] += k + 1
        changed_plan[fish, :, 2] = k + 10
        result.proposals.append(
            SimpleNamespace(
                fish=fish,
                node=2,
                seed_method="free_intersection",
                pose_xyz=changed_pose,
                plan=changed_plan,
            )
        )
    original_calls, helper_calls = [], []

    def base(self, *args):
        original_calls.append(args)
        return original

    def helper(*args, **kwargs):
        helper_calls.append((args, kwargs))
        return result

    monkeypatch.setattr(m.BoundedMouthContinuousTracker, "_proposals", base)
    monkeypatch.setattr(m.tail_recovery, "tail_recovery_proposals", helper)
    return tracker, states, original, result, original_calls, helper_calls


@pytest.mark.parametrize(
    "case", ["off", "persistence", "tail_age", "chest_age", "only_other_parent", "only_mouth_age"]
)
def test_closed_state0_gate_never_calls_tail_helper(monkeypatch, inputs, case):
    persistence = PersistentNodeConfig(enabled=case != "persistence")
    tracker, states, original, _, base_calls, helper_calls = policy_setup(
        monkeypatch, inputs, enabled=case != "off", persistence=persistence
    )
    if case in ("tail_age", "only_other_parent", "only_mouth_age"):
        states[0].node_missing[:, 2] = 3
    if case == "only_mouth_age":
        states[0].node_missing[:, 0] = 99
    if case == "chest_age":
        states[0].direct_pectoral_age[:] = 3
    rays, flow = object(), object()
    actual = tracker._proposals(states, rays, flow)
    assert actual is original
    assert len(base_calls) == 1 and base_calls[0][0] is states
    assert base_calls[0][1] is rays and base_calls[0][2] is flow
    assert helper_calls == []
    detail = tracker.last_tail_recovery_diagnostic
    assert not detail["enabled"] and not detail["helper_called"] and detail["admitted"] == 0
    assert tracker.initialize_count == tracker.processed_frames == tracker.restore_count == 0


@pytest.mark.parametrize("remaining", [0, 1, 4])
@pytest.mark.parametrize("thresholds", [(4, 4), (7, 6)])
def test_boundary_ages_order_and_capacity_match_frozen_append(
    monkeypatch, inputs, remaining, thresholds
):
    persistence = PersistentNodeConfig(
        minimum_missing_frames=thresholds[0], minimum_pectoral_frames=thresholds[1]
    )
    tracker, states, original, result, _, calls = policy_setup(
        monkeypatch, inputs, remaining=remaining, persistence=persistence
    )
    state_before = copy.deepcopy(states)
    before = reference.packets.signature(reference.packets.packet_proposals(original))
    rays = object()
    actual = tracker._proposals(states, rays)
    assert len(calls) == 1 and calls[0][1] == {"allow_bounded": False}
    for actual_arg, expected_arg in zip(
        calls[0][0],
        (
            original[0].pose,
            original[0].plan,
            rays,
            tracker.rig,
            tracker.lengths,
            tracker.recovery_config,
        ),
        strict=True,
    ):
        assert actual_arg is expected_arg
    expected, detail, fish = reference.append_tail(
        original,
        states,
        rays,
        tracker.rig,
        tracker.lengths,
        tracker.recovery_config,
        tracker.persistence_config,
        tracker.config.max_hypotheses + 6,
    )
    assert reference.packets.signature(
        reference.packets.packet_proposals(actual)
    ) == reference.packets.signature(reference.packets.packet_proposals(expected))
    assert all(a is b for a, b in zip(actual[: len(original)], original, strict=True))
    assert before == reference.packets.signature(reference.packets.packet_proposals(original))
    assert {k: tracker.last_tail_recovery_diagnostic[k] for k in detail} == detail
    assert detail["age_eligible_slots"] == [0]
    assert detail["after_age"] == 2 and detail["excluded_by_age"] == 2
    assert detail["admitted"] == min(remaining, 2)
    assert fish.tolist() == [-1] * len(original) + [0] * min(remaining, 2)
    for actual_state, before_state in zip(states, state_before, strict=True):
        for key in STATE_FIELDS:
            np.testing.assert_array_equal(getattr(actual_state, key), getattr(before_state, key))
    for proposal, source in zip(
        actual[len(original) :], (result.proposals[1], result.proposals[3]), strict=False
    ):
        for field, helper_value in (
            ("pose", source.pose_xyz),
            ("plan", source.plan),
            ("predicted", original[0].predicted),
        ):
            np.testing.assert_array_equal(getattr(proposal, field), helper_value)
            assert not np.shares_memory(getattr(proposal, field), helper_value)
        assert (
            proposal.kind == "tail_recovery"
            and proposal.parent == 0
            and proposal.permutation == (0, 1, 2)
        )


def test_physical_free_tail_matches_packet_reference_without_changing_other_nodes(
    monkeypatch, inputs
):
    rig, truth, cfg = inputs
    points = {v: rig.camera(v).project(truth) for v in VIEWS}
    rays = RayObservations.from_points(rig, points)
    current, plan = truth.copy(), np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    current[1, 2] = 2 * truth[1, 1] - truth[1, 2]
    plan[1, :, 2] = -1
    original = [
        Proposal(current.copy(), current.copy(), plan.copy(), i, "predict") for i in range(2)
    ]
    state = SimpleNamespace(node_missing=np.full((3, 3), 4), direct_pectoral_age=np.full(3, 4))
    tracker = m.TailRecoveryContinuousTracker(rig, truth, cfg)
    monkeypatch.setattr(m.BoundedMouthContinuousTracker, "_proposals", lambda *a: original)
    before = reference.packets.signature(reference.packets.packet_proposals(original))
    actual = tracker._proposals([state], rays)
    expected, detail, fish = reference.append_tail(
        original,
        [state],
        rays,
        rig,
        tracker.lengths,
        tracker.recovery_config,
        tracker.persistence_config,
        cfg.max_hypotheses + 6,
    )
    assert detail["admitted"] > 0
    assert reference.packets.signature(
        reference.packets.packet_proposals(actual)
    ) == reference.packets.signature(reference.packets.packet_proposals(expected))
    assert before == reference.packets.signature(reference.packets.packet_proposals(original))
    for proposal, slot in zip(actual[2:], fish[2:], strict=True):
        assert slot == 1
        np.testing.assert_allclose(proposal.pose[slot, 2], truth[slot, 2], atol=1e-7)
        np.testing.assert_array_equal(proposal.pose[:, :2], current[:, :2])
        np.testing.assert_array_equal(proposal.plan[:, :, :2], plan[:, :, :2])
        np.testing.assert_array_equal(proposal.pose[[0, 2]], current[[0, 2]])
        np.testing.assert_array_equal(proposal.plan[[0, 2]], plan[[0, 2]])
        assert supported_nodes(proposal.pose, proposal.plan, rays, rig)[1][slot, 1:].all()


@pytest.mark.parametrize("fault", ["empty", "parent", "kind", "permutation", "capacity"])
def test_invalid_original_pool_is_rejected_before_helper(monkeypatch, inputs, fault):
    tracker, states, original, _, _, calls = policy_setup(monkeypatch, inputs)
    if fault == "empty":
        original.clear()
    elif fault == "capacity":
        original.extend([original[-1]] * (tracker.config.max_hypotheses + 6))
    else:
        setattr(
            original[0],
            fault,
            {"parent": 1, "kind": "tail_recovery", "permutation": (1, 0, 2)}[fault],
        )
    with pytest.raises(ValueError):
        tracker._proposals(states, None)
    assert calls == []


@pytest.mark.parametrize("fault", ["node", "seed_method", "pose", "plan", "predicted", "kind"])
def test_wrong_helper_policy_or_original_mutation_is_rejected(monkeypatch, inputs, fault):
    tracker, states, original, result, _, _ = policy_setup(monkeypatch, inputs)

    def helper(*args, **kwargs):
        if fault in ("node", "seed_method"):
            setattr(result.proposals[0], fault, 0 if fault == "node" else "bounded_intersection")
        elif fault == "kind":
            original[-1].kind = "changed"
        else:
            getattr(original[-1], fault).flat[0] += 1
        return result

    monkeypatch.setattr(m.tail_recovery, "tail_recovery_proposals", helper)
    with pytest.raises(ValueError, match=r"free-only|mutated"):
        tracker._proposals(states, None)


@pytest.mark.parametrize("value", [0, 1, None, "true", np.bool_(True)])
def test_tail_switch_requires_python_boolean(inputs, value):
    with pytest.raises(ValueError, match="boolean"):
        m.TailRecoveryContinuousTracker(*inputs, tail_recovery_enabled=value)


def test_context_declares_only_tail_policy_and_implementation_and_inherits_updates(inputs):
    baseline = m.BoundedMouthContinuousTracker(*inputs)._context()
    tracker = m.TailRecoveryContinuousTracker(*inputs)
    context = tracker._context()
    assert context["implementation"] == "TailRecoveryContinuousTracker"
    policy = context.pop("tail_recovery_policy")
    context["implementation"] = baseline["implementation"]
    assert context == baseline
    assert policy["enabled"] and policy["target_node"] == 2
    assert not policy["allow_bounded"] and not policy["clears_identity_review"]
    assert tracker.last_tail_recovery_diagnostic is None
    for method in ("_fit", "advance", "_advance_owned", "_age_eligible", "restore_state", "finish"):
        assert getattr(m.TailRecoveryContinuousTracker, method) is getattr(
            m.BoundedMouthContinuousTracker, method
        )


@pytest.mark.parametrize("restore_count,enabled", [(0, True), (3, False)])
def test_migration_preserves_39_arrays_all_history_and_single_restore(
    inputs, checkpoint, restore_count, enabled
):
    arrays, metadata = copy.deepcopy(checkpoint)
    metadata["restore_count"] = restore_count
    before, old_meta = m.array_signature(arrays), copy.deepcopy(metadata)
    original = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = m.TailRecoveryContinuousTracker(*inputs, tail_recovery_enabled=enabled)
    report = m.import_original_tail_state(target, original, arrays, metadata)
    restored, new_meta = target.export_state()
    assert len(restored) == 39 and m.array_signature(restored) == before
    assert m.array_signature(arrays) == before and metadata == old_meta
    assert target.restore_count == restore_count + 1
    assert target.initialize_count == 1 and original.initialize_count == 0
    assert target.processed_frames == 5 and target.previous_frame == 1004
    assert target._states[0].review[2] and target._states[0].node_missing[0, 2] == 5
    assert report["bounded_validation"]["status"] == "ok"
    assert report["new_begin_calls"] == 0 and report["added_restore_count"] == 1
    assert not report["review_cleared_by_migration"]
    assert new_meta["context"]["tail_recovery_policy"]["enabled"] is enabled
    expected_context = copy.deepcopy(new_meta["context"])
    expected_context.pop("tail_recovery_policy")
    expected_context["implementation"] = report["bounded_context"]["implementation"]
    assert expected_context == report["bounded_context"]
    resumed = m.TailRecoveryContinuousTracker(*inputs, tail_recovery_enabled=enabled)
    resumed.restore_state(restored, new_meta)
    assert resumed.restore_count == restore_count + 2
    assert m.array_signature(resumed.export_state()[0]) == before
    for wrong in (
        m.AcceleratedContinuousPersistentNodeTracker(*inputs),
        m.BoundedMouthContinuousTracker(*inputs),
        m.TailRecoveryContinuousTracker(*inputs, tail_recovery_enabled=not enabled),
    ):
        with pytest.raises(ValueError, match="context"):
            wrong.restore_state(restored, new_meta)
        assert wrong.initialize_count == 0


@pytest.mark.parametrize(
    "fault",
    [
        "config",
        "persistence",
        "seed",
        "history",
        "age",
        "metadata_context",
        "target_started",
        "original_started",
    ],
)
def test_migration_rejects_drift_and_invalid_original_state_before_target_restore(
    inputs, checkpoint, fault
):
    arrays, metadata = copy.deepcopy(checkpoint)
    rig, pose, config = inputs
    target = m.TailRecoveryContinuousTracker(
        rig,
        pose + 0.1 if fault == "seed" else pose,
        replace(config, motion_weight=0.3) if fault == "config" else config,
        persistence_config=PersistentNodeConfig(minimum_missing_frames=7)
        if fault == "persistence"
        else None,
    )
    original = m.AcceleratedContinuousPersistentNodeTracker(*inputs)
    if fault == "history":
        arrays["history_parent"][-1] = 999
    elif fault == "age":
        arrays["state_node_missing"][0, 0, 2] = 0
    elif fault == "metadata_context":
        metadata["context"]["unexpected"] = True
    elif fault == "target_started":
        target.initialize_count = 1
    elif fault == "original_started":
        original.initialize_count = 1
    with pytest.raises(ValueError):
        m.import_original_tail_state(target, original, arrays, metadata)
    assert target.initialize_count == int(fault == "target_started")


def test_migration_requires_exact_fresh_tracker_types(inputs, checkpoint):
    class DerivedTail(m.TailRecoveryContinuousTracker):
        pass

    class DerivedOriginal(m.AcceleratedContinuousPersistentNodeTracker):
        pass

    for target, original in (
        (DerivedTail(*inputs), m.AcceleratedContinuousPersistentNodeTracker(*inputs)),
        (m.TailRecoveryContinuousTracker(*inputs), DerivedOriginal(*inputs)),
        (
            m.BoundedMouthContinuousTracker(*inputs),
            m.AcceleratedContinuousPersistentNodeTracker(*inputs),
        ),
    ):
        with pytest.raises(ValueError, match="explicit original and tail"):
            m.import_original_tail_state(target, original, *checkpoint)
        assert target.initialize_count == original.initialize_count == 0
