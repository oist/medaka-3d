"""Synthetic recurrence extraction, without original video, tracker or fit."""

import ast
import copy
import inspect

import numpy as np
import prepare_bounded_tail_parent_states as m
import pytest


def recurrence_ast(function):
    tree = ast.parse(inspect.getsource(function))
    loop = next(n for n in tree.body[0].body if isinstance(n, ast.For))
    start = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "pi"
    )
    return [ast.dump(n, include_attributes=False) for n in loop.body[start:]]


def test_all_recurrence_operations_and_order_equal_frozen_original():
    assert recurrence_ast(m.reconstruct_selected) == recurrence_ast(m.old.reconstruct)
    assert len(m.FRAMES) == 301
    assert m.FRAMES.tolist() == [66524, *range(66620, 66920)]


def fixture(monkeypatch):
    monkeypatch.setattr(m, "FIRST", 0)
    monkeypatch.setattr(m, "START", 1)
    monkeypatch.setattr(m, "HORIZON", 5)
    previous = dict(
        pose=np.zeros((6, 3, 3, 3)),
        velocity=np.zeros((6, 3, 3)),
        missing=np.zeros((6, 3), np.int64),
        review=np.zeros((6, 3), bool),
        cost=np.arange(6, dtype=float),
        node_missing=np.zeros((6, 3, 3), np.int64),
        direct_pectoral_age=np.zeros((6, 3), np.int64),
    )
    previous["pose"][..., 0] = np.arange(6)[:, None, None] * 20.0
    previous["velocity"][..., 0] = 1.5
    previous["review"][:, 2] = True
    parent = {"state_" + k: v.copy() for k, v in previous.items()}
    history = {k: [] for k in m.RECORD_FIELDS}
    previous_frames = []
    for native in range(5):
        if native:
            previous_frames.append(copy.deepcopy(previous))
        records, following = [], []
        for rank in range(6):
            pi = (rank + native) % 6
            pose = previous["pose"][pi].copy()
            pose[..., 0] += 10
            observed = np.zeros((3, 3), bool)
            observed[0, 1] = native % 2 == 1
            sources = np.full((3, 3, 3), -1, np.int32)
            if observed[0, 1]:
                sources[0, :2, 1] = 0
            # Fish1 has mask plus one head ray: body support with no direct chest.
            sources[1, 0, 0] = 1
            mask = np.array([False, True, False])
            body = np.array([native % 2 == 1, True, False])
            missing = np.where(body, 0, previous["missing"][pi] + 1)
            age = np.where(observed, 0, previous["node_missing"][pi] + 1)
            chest = np.where(observed[:, 1], previous["direct_pectoral_age"][pi] + 1, 0)
            review = previous["review"][pi].copy()
            cost = previous["cost"][pi] + 0.125
            # All vectors are x-only; a direct/bridge fish clips to2,
            # missing fish uses0.8 of its actual inherited nonzero velocity.
            velocity = previous["velocity"][pi].copy()
            velocity[:, 0] = np.where(body, 2.0, 0.8 * velocity[:, 0])
            row = dict(
                frame_idx=np.int64(native),
                pose_xyz=pose,
                observed=observed,
                sources=sources,
                identity_review=review,
                mask_supported=mask,
                body_supported=body,
                missing=missing,
                costs=np.float64(0.125),
                path_costs=np.float64(cost),
                loss_parts=np.zeros(5),
                score_margin=np.inf,
                parent=np.int64(pi),
                proposal_kind="bounded_mouth_recovery",
                hypotheses=np.int64(24),
                proposal_flow_nodes=np.zeros((3, 3), bool),
                proposal_flow_parent=np.int64(0),
                node_missing=age,
                direct_pectoral_age=chest,
            )
            records.append(row)
            following.append(
                dict(
                    pose=pose,
                    velocity=velocity,
                    missing=missing,
                    review=review,
                    cost=cost,
                    node_missing=age,
                    direct_pectoral_age=chest,
                )
            )
        for key in history:
            history[key].extend(r[key] for r in records)
        if native == 0:
            # Recordingprefix only; the authenticated checkpointstate is the seed
            # of suffix recurrence and must not be replaced by a replayedzero seed.
            continue
        previous = {k: np.asarray([r[k] for r in following]) for k in m.STATE_FIELDS}
    state = {"history_" + k: np.asarray(v) for k, v in history.items()}
    state["history_offsets"] = np.arange(0, 31, 6, dtype=np.int64)
    state.update({"state_" + k: v for k, v in previous.items()})
    parent["history_offsets"] = np.array([0, 6], np.int64)
    parent.update({"history_" + k: state["history_" + k][:6].copy() for k in m.RECORD_FIELDS})
    return parent, state, dict(previous_frame=4), previous_frames


def test_disjoint_native_selection_all6_fields_velocity_and_input_immutability(monkeypatch):
    parent, state, meta, previous = fixture(monkeypatch)
    before = m.array_signature(parent), m.array_signature(state)
    frames = np.array([2, 4], np.int64)
    actual, records, rows = m.reconstruct_selected(parent, state, meta, frames)
    assert set(actual) == set(m.STATE_FIELDS) and set(records) == set(m.RECORD_FIELDS)
    for key in m.STATE_FIELDS:
        assert m.checks.equal_bytes(actual[key], np.stack([previous[1][key], previous[3][key]]))
    assert np.array_equal(rows, [[12, 13, 14, 15, 16, 17], [24, 25, 26, 27, 28, 29]])
    for key in m.RECORD_FIELDS:
        assert m.checks.equal_bytes(records[key], state["history_" + key][rows])
    assert actual["velocity"][0, 0, 2, 0] != 0
    assert actual["velocity"][0, 0, 1, 0] == 2
    assert (m.array_signature(parent), m.array_signature(state)) == before
    actual["pose"].fill(-1)
    assert m.array_signature(state) == before[1]


@pytest.mark.parametrize(
    "corruption", ["terminal_velocity", "terminal_age", "unchosen_parent", "prefix", "sticky"]
)
def test_corruption_in_unselected_or_terminal_state_cannot_pass(monkeypatch, corruption):
    parent, state, meta, _ = fixture(monkeypatch)
    if corruption == "terminal_velocity":
        state["state_velocity"][5, 2, 0] += 0.1
    elif corruption == "terminal_age":
        state["state_node_missing"][5, 2, 2] += 1
    elif corruption == "unchosen_parent":
        state["history_parent"][23] = 8
    elif corruption == "prefix":
        state["history_pose_xyz"][4, 2, 1, 0] += 1
    else:
        state["history_identity_review"][23, 2] = False
    with pytest.raises(ValueError):
        m.reconstruct_selected(parent, state, meta, np.array([2, 4], np.int64))


@pytest.mark.parametrize(
    "frames",
    [
        np.array([2, 2]),
        np.array([4, 2]),
        np.array([0, 2]),
        np.array([2, 5]),
        np.array([2.0, 4.0]),
        np.array([], np.int64),
    ],
)
def test_selection_rejects_wrong_or_coerced_native_input(monkeypatch, frames):
    parent, state, meta, _ = fixture(monkeypatch)
    with pytest.raises(ValueError):
        m.reconstruct_selected(parent, state, meta, frames)
