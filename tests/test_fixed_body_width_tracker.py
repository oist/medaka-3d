"""Focused fixed-width tracker edges on small synthetic CPU observations only."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import fixed_body_width_tracker as m  # noqa: E402
from tracker import Proposal  # noqa: E402

import test_mask_transport_gap_diagnostics as diagnostic_toy  # noqa: E402
import test_mask_transport_tracker as toy  # noqa: E402


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def empty_votes(native):
    return tuple(
        m.transport.transport_votes(
            np.empty(0, np.int64),
            np.empty(0, bool),
            np.empty((0, 0), bool),
            group_count=0,
            mask_available=True,
            native=native,
            release_horizon=115,
            query_sha256="a" * 64,
        )
        for _ in m.VIEWS
    )


def tracker(data, coefficient=0, provider=None):
    def forbidden(native):
        raise AssertionError("coefficient zero must not read transport evidence")

    return m.FixedBodyWidthTracker(
        *data,
        first_native=100,
        group_counts=(0, 0, 0),
        query_sha256=("a" * 64,) * 3,
        vote_provider=provider or (empty_votes if coefficient else forbidden),
        input_binding={"fixture": "synthetic CPU only"},
        coefficient=coefficient,
    )


@pytest.fixture(scope="module")
def qualified():
    data = toy.inputs.__wrapped__()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old, fixed = toy.tracker(data), tracker(data)
        old.begin()
        fixed.begin()
        initial = toy.frame(data, 100)
        old.advance(initial)
        fixed.advance(initial)
        t0_old, t0_fixed = old.export_transport_state(), fixed.export_transport_state()
        old_result, fixed_result = old.finish(), fixed.finish()
        running = tracker(data)
        running.begin()
        for native in (100, 101):
            running.advance(toy.frame(data, native))
        bundle = running.export_transport_state()
        return data, t0_old, t0_fixed, old_result, fixed_result, bundle
    finally:
        torch.set_num_threads(previous)


def test_t0_all39_all21_all19_and_anchor_facts_exact(qualified):
    _, old, fixed, old_result, fixed_result, _ = qualified
    assert len(old[0]) == len(fixed[0]) == 39
    assert toy.signatures(old[0]) == toy.signatures(fixed[0])
    assert len(old_result) == len(fixed_result) == 21
    assert toy.signatures(old_result) == toy.signatures(fixed_result)
    old_content = m.frozen.decode_sidecar(old[2], old[3])
    content = m.decode_sidecar(fixed[2], fixed[3])
    assert len(m.RECORD_FIELDS) == 19
    for a, b in zip(old_content["anchor_facts"], content["anchor_facts"], strict=True):
        for key in m.RECORD_FIELDS:
            m.same_record(a["record"][key], b["record"][key], key)
        for ar, br in zip(a["references"], b["references"], strict=True):
            assert diagnostic_toy.fingerprint(ar) == diagnostic_toy.fingerprint(br)
    old_anchor_arrays, fixed_anchor_arrays = {}, {}
    assert m._pack(old_content["anchors"], old_anchor_arrays) == m._pack(
        content["anchors"], fixed_anchor_arrays
    )
    assert toy.signatures(old_anchor_arrays) == toy.signatures(fixed_anchor_arrays)
    d = content["fits"][0]["owner"]
    assert d["reason"] == "coefficient_zero" and not d["active"]
    assert d["iteration_base_score"].shape[0] == 0
    assert np.all(d["best_iteration"] == -1)
    assert content["fits"][0]["packet"] is None


def test_after_t0_lambda_zero_has_real_fixed_base_iterations_and_distinct_policy(qualified):
    _, old, _, _, _, bundle = qualified
    content = m.decode_sidecar(bundle[2], bundle[3])
    fit = content["fits"][1]
    assert not fit["owner"]["active"]
    assert fit["owner"]["reason"] == "fixed_radial_coefficient_zero"
    steps = bundle[1]["context"]["tracker_config"]["opt_steps"]
    assert fit["owner"]["iteration_base_score"].shape == (steps + 1, len(fit["cost"]))
    assert fit["owner"]["iteration_gap_gradient_norm"].shape == (steps, len(fit["cost"]))
    assert np.all(fit["owner"]["best_iteration"] >= 0)
    assert fit["owner"]["target_mass"].shape == (0,)
    assert fit["packet"] is None
    policy = bundle[1]["context"]["mask_transport"]
    assert policy["schema"] == bundle[3]["schema"] == m.SCHEMA != old[3]["schema"]
    assert policy["objective"] == bundle[3]["objective"] == m.OBJECTIVE
    assert policy["fixed_width"] == bundle[3]["fixed_width"]
    assert policy["fixed_width"]["apply_from_native"] == 101
    assert policy["fixed_width"]["radial_sigma_mm"] == m.signature(
        content["fixed_width_reference"]["radial_sigma_mm"]
    )
    assert m.FixedBodyWidthTracker._advance_owned is m.frozen.MaskTransportTracker._advance_owned
    assert (
        m.FixedBodyWidthTracker.restore_transport_state
        is m.frozen.MaskTransportTracker.restore_transport_state
    )


def test_t0_never_invokes_fixed_fitter_but_lambda_zero_next_native_always_does(monkeypatch):
    data = toy.inputs.__wrapped__()
    target = tracker(data)
    fixed_fit = m.fitter.fit_with_fixed_width
    calls = []

    def checked(*args, **kwargs):
        calls.append(args[0]._transport_frame.frame_idx)
        return fixed_fit(*args, **kwargs)

    monkeypatch.setattr(m.fitter, "fit_with_fixed_width", checked)
    target.begin()
    target.advance(toy.frame(data, 100))
    assert calls == []
    target.advance(toy.frame(data, 101))
    assert calls == [101]


def test_empty_target_with_nonzero_coefficient_keeps_packet_but_runs_fixed_base():
    data = toy.inputs.__wrapped__()
    target = tracker(data, coefficient=3)
    target.begin()
    for native in (100, 101):
        target.advance(toy.frame(data, native))
    bundle = target.export_transport_state()
    fit = m.decode_sidecar(bundle[2], bundle[3])["fits"][1]
    assert fit["packet"] is not None
    assert fit["owner"]["reason"] == "fixed_radial_zero_effective_target"
    assert not fit["owner"]["active"] and not fit["owner"]["target_diagnostics_computed"]
    assert fit["owner"]["iteration_base_score"].shape[0] == data[2].opt_steps + 1
    assert np.all(fit["owner"]["best_iteration"] >= 0)
    assert fit["owner"]["target_mass_by_view_body"].shape == (0, 3, 3)


def test_same_device_restore_all39_side_arrays_then_continuation_exact(qualified):
    data, _, _, _, _, bundle = qualified
    before = toy.signatures(bundle[0]), toy.signatures(bundle[2]), copy.deepcopy(bundle[3])
    resumed, direct = tracker(data), tracker(data)
    resumed.restore_transport_state(*bundle)
    roundtrip = resumed.export_transport_state()
    assert resumed.initialize_count == resumed.restore_count == 1
    assert toy.signatures(roundtrip[0]) == before[0]
    assert toy.signatures(roundtrip[2]) == before[1]
    assert roundtrip[1]["context"] == bundle[1]["context"]
    direct.begin()
    for native in (100, 101, 102):
        frame = toy.frame(data, native)
        direct.advance(frame)
        if native == 102:
            resumed.advance(frame)
    assert toy.signatures(direct.export_transport_state()[0]) == toy.signatures(
        resumed.export_transport_state()[0]
    )
    assert toy.signatures(direct.finish()) == toy.signatures(resumed.finish())
    assert (toy.signatures(bundle[0]), toy.signatures(bundle[2]), bundle[3]) == before


def test_registered_width_reference_is_owned_immutable_and_current_initial_mutation_rejected():
    rig, seed, cfg = toy.inputs.__wrapped__()
    caller = seed.copy()
    target = tracker((rig, caller, cfg))
    expected = target._fixed_width_reference_pose.copy()
    caller[0, 0, 0] += 1
    np.testing.assert_array_equal(target._fixed_width_reference_pose, expected)
    with pytest.raises(ValueError):
        target._fixed_width_reference_pose.setflags(write=True)
    with pytest.raises(ValueError):
        target._fixed_width_radial.setflags(write=True)
    target.initial[0, 0, 0] += 1
    with pytest.raises(ValueError, match="initial width reference"):
        target.begin()
        target.advance(toy.frame((rig, seed, cfg), 100))


def test_actual_renderer_reference_must_equal_registered_policy():
    data = toy.inputs.__wrapped__()
    target = tracker(data)
    pixels = {v: np.zeros((1, 2), np.float64) for v in m.VIEWS}
    renderer = m.FixedWidthRenderer(
        data[0],
        pixels,
        m.RendererConfig(width_ratio=data[2].width_ratio),
        dtype=torch.float64,
        reference_pose_xyz=data[1],
    )
    assert target.validate_fixed_width_renderer(renderer) == target._fixed_width_signature
    changed = data[1].copy()
    changed[0, 0, 0] += 1
    other = m.FixedWidthRenderer(
        data[0],
        pixels,
        m.RendererConfig(width_ratio=data[2].width_ratio),
        dtype=torch.float64,
        reference_pose_xyz=changed,
    )
    with pytest.raises(ValueError, match="fitted renderer reference"):
        target.validate_fixed_width_renderer(other)


@pytest.mark.parametrize(
    "fault", ["width", "reference", "apply_native", "unselected_parent", "anchor"]
)
def test_rehashed_sidecar_mutations_rejected_before_restore(qualified, fault):
    data, _, _, _, _, bundle = qualified
    core, metadata, arrays, header = bundle
    content = m.decode_sidecar(arrays, header)
    changed = copy.deepcopy(header)
    if fault == "width":
        content["fixed_width_reference"]["radial_sigma_mm"][0, 0] += 0.1
    elif fault == "reference":
        content["fixed_width_reference"]["pose"][0, 0, 0] += 1
    elif fault == "apply_native":
        changed["fixed_width"]["apply_from_native"] += 1
        content["fixed_width_reference"]["policy"] = copy.deepcopy(changed["fixed_width"])
    elif fault == "unselected_parent":
        fit = content["fits"][1]
        j = next(i for i in range(len(fit["cost"])) if i not in fit["selected"])
        fit["parent_cost"][j] += 1
    else:
        content["active_anchor_ids"][0] += 1
    new_arrays = {}
    changed["tree"] = m._pack(content, new_arrays)
    changed["array_signatures"] = toy.signatures(new_arrays)
    changed.pop("content_sha256")
    changed["content_sha256"] = m.json_hash(changed)
    restored = tracker(data)
    with pytest.raises(ValueError):
        restored.restore_transport_state(core, metadata, new_arrays, changed)
    assert restored.initialize_count == restored.restore_count == 0


def test_old_dynamic_context_and_bare_state_restore_refused(qualified):
    data, old, _, _, _, _ = qualified
    target = tracker(data)
    with pytest.raises(ValueError):
        target.restore_transport_state(*old)
    with pytest.raises(ValueError):
        m.decode_sidecar(old[2], old[3])
    with pytest.raises(RuntimeError, match="sidecar"):
        target.restore_state(*old[:2])


def measured_inactive(coefficient=0.0, reason="fixed_radial_coefficient_zero"):
    fit = diagnostic_toy.active_fit()
    d = fit["owner"]
    d.update(
        active=False, reason=reason, coefficient=coefficient, target_diagnostics_computed=False
    )
    d["base_score"] = d["total_score"].copy()
    d["iteration_base_score"] = d["iteration_total_score"].copy()
    for key in (
        "owner_loss",
        "gap_loss",
        "iteration_owner_loss",
        "iteration_gap_loss",
        "iteration_owner_gradient_norm",
        "iteration_gap_gradient_norm",
    ):
        d[key] = np.zeros_like(d[key])
    d["target_mass"] = np.empty(0, np.float64)
    for key in ("final_gap_by_view_body", "target_mass_by_view_body", "positive_gap_pixel_count"):
        d[key] = np.empty((0, 3, 3), np.int64 if key == "positive_gap_pixel_count" else np.float64)
    return fit


@pytest.mark.parametrize(
    "coefficient,reason",
    [
        (0.0, "fixed_radial_coefficient_zero"),
        (3.0, "fixed_radial_zero_effective_target"),
        (3.0, "fixed_radial_zero_valid_ray_target"),
    ],
)
def test_inactive_fixed_base_has_measured_first_tie_best_with_uncomputed_target(
    coefficient, reason
):
    fit = measured_inactive(coefficient, reason)
    before = diagnostic_toy.fingerprint(fit)
    m.validate_fixed_width_diagnostics(fit, coefficient=coefficient, opt_steps=14, initial=False)
    assert diagnostic_toy.fingerprint(fit) == before
    np.testing.assert_array_equal(fit["owner"]["best_iteration"], [4, 14])


def test_active_numerical_contract_preserved_with_new_reason():
    fit = diagnostic_toy.active_fit()
    fit["owner"]["reason"] = "fixed_radial_owner_plus_gap"
    m.validate_fixed_width_diagnostics(fit, coefficient=3.0, opt_steps=14, initial=False)
    fit["owner"]["gap_loss"][0] += 0.1
    with pytest.raises(ValueError, match="decomposition"):
        m.validate_fixed_width_diagnostics(fit, coefficient=3.0, opt_steps=14, initial=False)


@pytest.mark.parametrize(
    "fault",
    [
        "old_empty",
        "old_reason",
        "minus_one",
        "later_tie",
        "off_nonzero_gradient",
        "missing_iterate",
        "hidden_base_change",
    ],
)
def test_new_inactive_cannot_disguise_old_dispatch_or_missing_measurements(fault):
    fit = measured_inactive()
    d = fit["owner"]
    if fault == "old_empty":
        fit = diagnostic_toy.inactive_fit(0.0, "coefficient_zero")
    elif fault == "old_reason":
        d["reason"] = "coefficient_zero"
    elif fault == "minus_one":
        d["best_iteration"][0] = -1
    elif fault == "later_tie":
        d["best_iteration"][0] = 5
    elif fault == "off_nonzero_gradient":
        d["iteration_gap_gradient_norm"][1, 0] = 1
    elif fault == "missing_iterate":
        d["iteration_base_score"] = d["iteration_base_score"][:-1]
    else:
        d["iteration_base_score"][10, 0] += 1
    with pytest.raises(ValueError):
        m.validate_fixed_width_diagnostics(fit, coefficient=0.0, opt_steps=14, initial=False)


def test_initial_scope_cannot_accept_fixed_base_diagnostics_and_requires_explicit_bool():
    with pytest.raises(ValueError):
        m.validate_fixed_width_diagnostics(
            measured_inactive(), coefficient=0.0, opt_steps=14, initial=True
        )
    with pytest.raises(ValueError):
        m.validate_fixed_width_diagnostics(
            measured_inactive(), coefficient=0.0, opt_steps=14, initial=1
        )


def test_fixed_fit_target_gather_remains_parent_lineage_not_permutation(monkeypatch):
    data = toy.inputs.__wrapped__()
    votes = tuple(
        SimpleNamespace(native=101, release_horizon=115, group_count=0, query_sha256="a" * 64)
        for _ in m.VIEWS
    )
    target = tracker(data, coefficient=3, provider=lambda native: votes)
    target.begin()
    target._history = [[]]
    target._anchor_ids = np.array([1, 0], np.int64)
    target._anchors = ("anchor0", "anchor1")
    target._transport_frame = SimpleNamespace(frame_idx=101, masks={v: None for v in m.VIEWS})
    pose, plan = data[1], np.zeros((3, 3, 3), np.int64)
    proposals = [
        Proposal(pose.copy(), pose.copy(), plan.copy(), parent, "predict", permutation)
        for parent, permutation in ((0, (2, 1, 0)), (1, (1, 0, 2)))
    ]
    parents = [SimpleNamespace(cost=2.0), SimpleNamespace(cost=3.0)]
    packet = SimpleNamespace(
        anchor_ids=np.arange(2),
        target=np.arange(24, dtype=np.float64).reshape(2, 3, 4, 1).repeat(3, -1),
    )
    samples = SimpleNamespace(
        pixels={v: np.zeros((4, 2)) for v in m.VIEWS},
        target=np.ones((3, 4)),
        weights=np.ones((3, 4)),
        foreground=np.ones((3, 4), bool),
    )
    monkeypatch.setattr(m, "transport", SimpleNamespace(packet_for_pixels=lambda *args: packet))
    seen = []

    def fitted(
        tracker, actual_proposals, actual_parents, rays, actual_samples, values, *, coefficient
    ):
        assert (
            actual_proposals is proposals
            and actual_parents is parents
            and actual_samples is samples
        )
        seen.append(values.copy())
        assert not np.shares_memory(values, packet.target)
        diag = measured_inactive(3.0, "fixed_radial_zero_valid_ray_target")
        # The real tracker fixture uses one Adam step, not this pure 14-step diagnostic.
        d = diag["owner"]
        for key in ("base_score", "owner_loss", "gap_loss", "total_score"):
            d["iteration_" + key] = np.stack([d[key], d[key]])
        for key in ("iteration_owner_gradient_norm", "iteration_gap_gradient_norm"):
            d[key] = np.zeros((1, 2))
        d["best_iteration"] = np.zeros(2, np.int64)
        tracker.last_gap_fit = d
        return (
            np.stack([p.pose for p in proposals]),
            diag["cost"],
            diag["parts"],
            np.ones((2, 3), bool),
        )

    monkeypatch.setattr(m.fitter, "fit_with_fixed_width", fitted)
    target._fit(proposals, parents, None, samples)
    np.testing.assert_array_equal(seen[0], packet.target[[1, 0]])
    np.testing.assert_array_equal(target._fit_capture["candidate_anchor"], [1, 0])
    np.testing.assert_array_equal(target._fit_capture["parent_cost"], [2.0, 3.0])
