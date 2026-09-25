"""New objective/state edge on tiny synthetic CPU data, plus injected-fit capture.

No saved fish data, pretrained model or GPU. Frozen full suites are not repeated.
"""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import mask_transport_gap_tracker as m  # noqa: E402
from node_geometry import RayObservations  # noqa: E402
from observations import PixelTargets  # noqa: E402
from tracker import Proposal  # noqa: E402

import test_mask_transport_tracker as toy  # noqa: E402


def tracker(data, coefficient=0, provider=None):
    def forbidden(native):
        raise AssertionError("off fit must not request votes")

    return m.MaskTransportGapTracker(
        *data,
        first_native=100,
        group_counts=(0, 0, 0),
        query_sha256=("a" * 64,) * 3,
        vote_provider=provider or forbidden,
        input_binding={"fixture": "synthetic CPU only"},
        coefficient=coefficient,
    )


@pytest.fixture(scope="module")
def qualified():
    data = toy.inputs.__wrapped__()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old, new = toy.tracker(data), tracker(data)
        old.begin()
        new.begin()
        for native in (100, 101):
            frame = toy.frame(data, native)
            old.advance(frame)
            new.advance(frame)
        old_bundle, new_bundle = old.export_transport_state(), new.export_transport_state()
        old_result, new_result = old.finish(), new.finish()
        return data, old_bundle, new_bundle, old_result, new_result
    finally:
        torch.set_num_threads(previous)


def test_off_new_objective_all39_and_all21_match_old_but_context_sidecar_are_distinct(qualified):
    _, old, new, old_result, new_result = qualified
    assert len(old[0]) == len(new[0]) == 39
    assert toy.signatures(old[0]) == toy.signatures(new[0])
    assert len(old_result) == len(new_result) == 21
    assert toy.signatures(old_result) == toy.signatures(new_result)
    policy = new[1]["context"]["mask_transport"]
    assert policy["schema"] == new[3]["schema"] == m.SCHEMA != old[3]["schema"]
    assert policy["objective"] == new[3]["objective"] == m.OBJECTIVE
    decoded = m.decode_sidecar(new[2], new[3])
    assert len(decoded["fits"]) == 2
    for fit in decoded["fits"]:
        assert fit["owner"]["gap_loss"].shape == fit["cost"].shape
        assert not fit["owner"]["target_diagnostics_computed"]
        assert fit["owner"]["target_mass"].shape == (0,)
        assert fit["owner"]["reason"] == "coefficient_zero"


def test_new_restore_keeps_all_core_side_arrays_and_original_parent_anchor_history(qualified):
    data, _, bundle, _, _ = qualified
    before = toy.signatures(bundle[0]), toy.signatures(bundle[2]), copy.deepcopy(bundle[3])
    restored = tracker(data)
    restored.restore_transport_state(*bundle)
    after = restored.export_transport_state()
    assert restored.initialize_count == restored.restore_count == 1
    assert toy.signatures(after[0]) == before[0]
    assert toy.signatures(after[2]) == before[1]
    assert (toy.signatures(bundle[0]), toy.signatures(bundle[2]), bundle[3]) == before
    assert after[1]["context"] == bundle[1]["context"]
    assert m.MaskTransportGapTracker._advance_owned is m.frozen.MaskTransportTracker._advance_owned
    assert (
        m.MaskTransportGapTracker.restore_transport_state
        is m.frozen.MaskTransportTracker.restore_transport_state
    )
    assert m.MaskTransportGapTracker.__init__ is m.frozen.MaskTransportTracker.__init__


def test_owner_only_context_and_sidecar_cannot_be_accepted_as_gap(qualified):
    data, old, new, _, _ = qualified
    with pytest.raises(ValueError):
        tracker(data).restore_transport_state(*old)
    with pytest.raises(ValueError):
        m.decode_sidecar(old[2], old[3])
    with pytest.raises(ValueError):
        m.frozen.decode_sidecar(new[2], new[3])


@pytest.mark.parametrize("fault", ["objective", "gap", "active", "unselected_parent", "anchor"])
def test_rehashed_new_objective_or_state_drift_is_rejected_before_restore(qualified, fault):
    data, _, bundle, _, _ = qualified
    core, metadata, arrays, header = bundle
    content = m.decode_sidecar(arrays, header)
    changed = copy.deepcopy(header)
    if fault == "objective":
        changed["objective"]["gap_relative_weight"] = 2.0
    elif fault == "gap":
        content["fits"][1]["owner"]["gap_loss"][0] = 0.5
    elif fault == "active":
        content["fits"][1]["owner"]["target_diagnostics_computed"] = True
    elif fault == "unselected_parent":
        fit = content["fits"][1]
        index = next(i for i in range(len(fit["parent"])) if i not in fit["selected"])
        fit["parent_cost"][index] += 1
    else:
        content["active_anchor_ids"][0] += 1
    new_arrays = {}
    changed["tree"] = m._pack(content, new_arrays)
    changed["array_signatures"] = toy.signatures(new_arrays)
    changed.pop("content_sha256")
    changed["content_sha256"] = m.json_hash(changed)
    target = tracker(data)
    with pytest.raises(ValueError):
        target.restore_transport_state(core, metadata, new_arrays, changed)
    assert target.initialize_count == target.restore_count == 0


def test_candidate_target_gather_is_parent_lineage_not_proposal_permutation(monkeypatch):
    data = toy.inputs.__wrapped__()
    votes = tuple(
        SimpleNamespace(native=101, release_horizon=115, group_count=0, query_sha256="a" * 64)
        for _ in m.VIEWS
    )
    seen = {}
    target = tracker(data, coefficient=3, provider=lambda native: votes)
    target.begin()
    target._history = [[]]
    target._anchor_ids = np.array([1, 0], np.int64)
    target._anchors = ("first anchor", "second anchor")
    target._transport_frame = SimpleNamespace(frame_idx=101, masks={v: None for v in m.VIEWS})
    p = data[1]
    plan = np.zeros((3, 3, 3), np.int64)
    proposals = [
        Proposal(p.copy(), p.copy(), plan.copy(), parent, "predict", perm)
        for parent, perm in ((0, (2, 1, 0)), (1, (1, 0, 2)), (0, (0, 1, 2)))
    ]
    states = [SimpleNamespace(cost=2.0), SimpleNamespace(cost=3.0)]
    pixels = {v: np.array([[10.0, 20.0], [30.0, 40.0]]) for v in m.VIEWS}
    samples = SimpleNamespace(
        pixels=pixels,
        target=np.ones((3, 2)),
        weights=np.ones((3, 2)),
        foreground=np.ones((3, 2), bool),
    )
    packet = SimpleNamespace(
        anchor_ids=np.arange(2), target=np.arange(36, dtype=np.float64).reshape(2, 3, 2, 3)
    )

    def packet_for_pixels(received_votes, anchors, masks, received_pixels):
        assert received_votes == votes and anchors is target._anchors
        seen["pixels"] = received_pixels.copy()
        return packet

    def fit(tracker, received, parents, rays, targets, values, *, coefficient):
        assert (
            received is proposals and parents is states and targets is samples and coefficient == 3
        )
        seen["target"] = values.copy()
        assert not np.shares_memory(values, packet.target)
        result = (
            np.stack([item.pose for item in received]),
            np.array([1.0, 1.5, 2.0]),
            np.zeros((3, 5)),
            np.ones((3, 3), bool),
        )
        # Injected fit result only, not a numerical qualification of ray validity.
        m.fitter._zero_gap_diagnostics(tracker, result, "zero_valid_ray_target", coefficient)
        return result

    monkeypatch.setattr(m, "transport", SimpleNamespace(packet_for_pixels=packet_for_pixels))
    monkeypatch.setattr(m.fitter, "fit_with_gap", fit)
    target._fit(proposals, states, None, samples)
    expected = np.array([1, 0, 1], np.int64)
    np.testing.assert_array_equal(seen["target"], packet.target[expected])
    np.testing.assert_array_equal(target._fit_capture["candidate_anchor"], expected)
    np.testing.assert_array_equal(target._fit_capture["path_cost"], [3.0, 4.5, 4.0])
    np.testing.assert_array_equal(target._fit_capture["fit_pixels"], seen["pixels"])
    assert "gap_loss" in target._fit_capture["owner"]
    assert not hasattr(target, "last_owner_fit")


def test_tiny_real_cpu_gap_fitter_outputs_satisfy_new_diagnostic_contract():
    # One synthetic candidate, two score evaluations: catches real fitter/schema
    # or reduction-order differences that an injected diagnostic cannot expose.
    data = toy.inputs.__wrapped__()
    rig, pose, cfg = data
    target = tracker(data, coefficient=3)
    target.begin()
    points = {v: rig.camera(v).project(pose) for v in m.VIEWS}
    rays = RayObservations.from_points(rig, points)
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    proposals = [Proposal(pose.copy(), pose.copy(), plan.copy(), 0, "predict")]
    pixels = {v: np.rint(rig.camera(v).project(pose[0])).astype(np.float64) for v in m.VIEWS}
    sample = PixelTargets(pixels, np.ones((3, 3)), np.full((3, 3), 1 / 3), np.ones((3, 3), bool))
    desired = np.zeros((1, 3, 3, 3), np.float64)
    desired[..., 1] = 0.5
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fitted = m.fitter.fit_with_gap(target, proposals, target._states, rays, sample, desired)
    finally:
        torch.set_num_threads(previous)
    fit = dict(
        owner=target.last_gap_fit,
        parent=np.zeros(1, np.int64),
        cost=fitted[1],
        parts=fitted[2],
        fit_pixels=np.stack([pixels[v] for v in m.VIEWS]),
    )
    m.validate_gap_diagnostics(fit, coefficient=3.0, opt_steps=cfg.opt_steps)
    assert fit["owner"]["active"]
    assert np.any(fit["owner"]["gap_loss"] > 0)
    assert np.any(fit["owner"]["iteration_gap_gradient_norm"] > 0)
