"""Small real CPU fits on synthetic orthographic geometry; no actual fish data."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import mask_transport_tracker as m  # noqa: E402
import test_directional_chest_tracker as toy  # noqa: E402
from continuous_tracker import ContinuousPersistentNodeTracker  # noqa: E402
from observations import FrameObservation, PixelTargets  # noqa: E402
from tracker import Proposal  # noqa: E402


@pytest.fixture(scope="module")
def inputs():
    rig, pose, cfg = toy.inputs.__wrapped__()
    return rig, pose, replace(cfg, review_gap_frames=2)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def frame(inputs, native):
    rig, pose, _ = inputs
    detections = {v: rig.camera(v).project(pose) for v in m.VIEWS}
    # Exercise sticky review and full state ages, with all native frame buffers.
    for points in detections.values():
        points[2, 1] = np.nan
    images = {v: np.zeros((1200, 1200, 3), np.uint8) for v in m.VIEWS}
    masks = {v: np.empty((0, 1200, 1200), bool) for v in m.VIEWS}
    return FrameObservation(native, detections, images, masks, images)


def tracker(inputs, coefficient=0, provider=None):
    def forbidden(native):
        raise AssertionError("lambda0 must never ask for point evidence")

    return m.MaskTransportTracker(
        *inputs,
        first_native=100,
        group_counts=(0, 0, 0),
        query_sha256=("a" * 64,) * 3,
        vote_provider=provider or forbidden,
        input_binding={"fixture": "synthetic CPU only"},
        coefficient=coefficient,
    )


def signatures(arrays):
    return {k: m.signature(v) for k, v in arrays.items()}


@pytest.fixture(scope="module")
def off_bundle(inputs):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = ContinuousPersistentNodeTracker(*inputs)
        new = tracker(inputs)
        old.begin()
        new.begin()
        for native in (100, 101):
            observed = frame(inputs, native)
            before = signatures(observed.detections)
            old.advance(observed)
            new.advance(observed)
            assert signatures(observed.detections) == before
        old_arrays, _ = old.export_state()
        bundle = new.export_transport_state()
        assert len(old_arrays) == len(bundle[0]) == 39
        assert signatures(old_arrays) == signatures(bundle[0])
        return bundle, old.finish(), new.finish()
    finally:
        torch.set_num_threads(old_threads)


def test_lambda0_all39_all21_exact_and_causal_sidecar(off_bundle):
    bundle, old_result, new_result = off_bundle
    assert len(old_result) == len(new_result) == 21
    assert signatures(old_result) == signatures(new_result)
    decoded = m.decode_sidecar(bundle[2], bundle[3])
    b0 = int(bundle[0]["history_offsets"][1])
    assert len(decoded["anchors"]) == len(decoded["anchor_facts"]) == b0
    assert np.array_equal(decoded["anchor_history"][0], np.arange(b0))
    assert decoded["fits"][0]["packet"] is None
    assert decoded["fits"][1]["packet"] is None
    assert all(f["owner"]["reason"] == "coefficient_zero" for f in decoded["fits"])
    assert all(f["fit_pixels"].shape == (3, 12, 2) for f in decoded["fits"])
    assert new_result["identity_review"][-1, 2]


def test_restored_sidecar_continues_exact_and_only_restore_counter_changes(inputs, off_bundle):
    bundle = off_bundle[0]
    initial = signatures(bundle[0]), signatures(bundle[2]), copy.deepcopy(bundle[3])
    resumed = tracker(inputs)
    resumed.restore_transport_state(*bundle)
    roundtrip = resumed.export_transport_state()
    assert signatures(roundtrip[0]) == signatures(bundle[0])
    assert signatures(roundtrip[2]) == signatures(bundle[2])
    assert resumed.restore_count == 1 and resumed.initialize_count == 1
    fresh = tracker(inputs)
    fresh.begin()
    for native in (100, 101, 102):
        observed = frame(inputs, native)
        fresh.advance(observed)
        if native == 102:
            resumed.advance(observed)
    assert signatures(fresh.export_transport_state()[0]) == signatures(
        resumed.export_transport_state()[0]
    )
    assert signatures(fresh.finish()) == signatures(resumed.finish())
    assert (signatures(bundle[0]), signatures(bundle[2]), bundle[3]) == initial


@pytest.mark.parametrize("change", ["coefficient", "binding", "array", "core", "old_api"])
def test_wrong_context_or_unsealed_sidecar_refused(inputs, off_bundle, change):
    bundle = list(off_bundle[0])
    target = tracker(inputs, coefficient=3 if change == "coefficient" else 0)
    if change == "binding":
        target._binding["fixture"] = "changed"
    if change == "array":
        bundle[2] = dict(bundle[2])
        key = "root/active_anchor_ids"
        bundle[2][key] = bundle[2][key] + 1
    if change == "core":
        bundle[0] = dict(bundle[0])
        bundle[0]["state_node_missing"] = bundle[0]["state_node_missing"] + 1
    with pytest.raises((ValueError, RuntimeError)):
        if change == "old_api":
            target.restore_state(*bundle[:2])
        else:
            target.restore_transport_state(*bundle)
    assert target.initialize_count == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ("unselected_anchor", "all saved candidate anchors"),
        ("unselected_path", "all candidate path costs"),
        ("coefficient", "coefficient policy"),
        ("t0_record", "t0 retained anchor"),
        ("off_packet", "off arm"),
        ("selection", "saved candidate selection"),
    ],
)
def test_rehashed_semantic_sidecar_tamper_is_not_accepted(inputs, off_bundle, change, match):
    core, metadata, side_arrays, side_metadata = off_bundle[0]
    content = m.decode_sidecar(side_arrays, side_metadata)
    second = content["fits"][1]
    unselected = next(i for i in range(len(second["cost"])) if i not in second["selected"])
    if change == "unselected_anchor":
        second["candidate_anchor"][unselected] += 1
    elif change == "unselected_path":
        second["path_cost"][unselected] += 1
    elif change == "coefficient":
        second["owner"]["coefficient"] = 3.0
    elif change == "t0_record":
        content["anchor_facts"][0]["record"]["pose_xyz"][0, 0, 0] += 1
    elif change == "off_packet":
        second["packet"] = {"unexpected": True}
    else:
        second["selected"] = second["selected"][::-1].copy()
    changed_arrays = {}
    changed_header = copy.deepcopy(side_metadata)
    changed_header["tree"] = m._pack(content, changed_arrays)
    changed_header["array_signatures"] = signatures(changed_arrays)
    del changed_header["content_sha256"]
    changed_header["content_sha256"] = m.json_hash(changed_header)
    target = tracker(inputs)
    with pytest.raises(ValueError, match=match):
        target.restore_transport_state(core, metadata, changed_arrays, changed_header)
    assert target.initialize_count == 0


def test_active_empty_evidence_uses_original_dispatch_and_keeps_packet(inputs):
    def votes(native):
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

    active, off = tracker(inputs, 3, votes), tracker(inputs)
    active.begin()
    off.begin()
    for native in (100, 101):
        observed = frame(inputs, native)
        active.advance(observed)
        off.advance(observed)
    a, b = active.export_transport_state(), off.export_transport_state()
    assert signatures(a[0]) == signatures(b[0])
    decoded = m.decode_sidecar(a[2], a[3])
    assert decoded["fits"][1]["owner"]["reason"] == "zero_effective_target"
    assert decoded["fits"][1]["packet"]["target"].shape[-3:] == (3, 12, 3)


def test_real_cpu_nonzero_fit_matches_direct_final_owner_equation(inputs):
    rig, pose, cfg = inputs
    config = replace(cfg, opt_steps=2)
    base = ContinuousPersistentNodeTracker(rig, pose, config)
    base.begin()
    points = {v: rig.camera(v).project(pose) for v in m.VIEWS}
    rays = m.RayObservations.from_points(rig, points)
    plan = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    proposals = [Proposal(pose.copy(), pose.copy(), plan.copy(), 0, "predict")]
    # Deliberately request a different body's ownership at body0's pixels.
    pixels = {v: np.rint(rig.camera(v).project(pose[0])).astype(np.float64) for v in m.VIEWS}
    pixel_targets = PixelTargets(
        pixels, np.ones((3, 3)), np.full((3, 3), 1 / 3), np.ones((3, 3), bool)
    )
    target = np.zeros((1, 3, 3, 3), np.float64)
    target[..., 1] = 0.5
    before = signatures({"pose": proposals[0].pose, "plan": plan, "target": target})
    actual = m.fitter.fit_with_ownership(base, proposals, base._states, rays, pixel_targets, target)
    diagnostic = base.last_owner_fit
    assert diagnostic["active"] and diagnostic["iteration_total_score"].shape == (3, 1)
    assert diagnostic["iteration_owner_gradient_norm"].shape == (2, 1)
    assert np.all(diagnostic["iteration_owner_gradient_norm"] > 0)
    assert np.array_equal(
        diagnostic["best_iteration"], diagnostic["iteration_total_score"].argmin(0)
    )
    renderer = m.fitter.RefractiveFishRenderer(
        rig,
        pixels,
        m.fitter.RendererConfig(width_ratio=cfg.width_ratio),
        device="cpu",
        dtype=torch.float64,
    )
    with torch.no_grad():
        rendering = renderer.render(torch.tensor(actual[0]))
        manual = (
            (1 - rendering.ownership.numpy()) ** 2
            * target
            * pixel_targets.weights[None, :, :, None]
            * renderer.ray_valid.numpy()[None, :, :, None]
        ).sum((1, 2, 3)) / 3
    np.testing.assert_allclose(diagnostic["owner_loss"], manual, rtol=0, atol=1e-15)
    np.testing.assert_allclose(actual[1], diagnostic["base_score"] + 3 * manual, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        actual[1], diagnostic["iteration_total_score"].min(0), rtol=0, atol=1e-10
    )
    assert actual[2].shape == (1, 5) and actual[3].dtype == bool
    assert signatures({"pose": proposals[0].pose, "plan": plan, "target": target}) == before


def test_causal_initial_anchor_uses_global_strict_support_and_each_retained_body_axis(
    inputs, off_bundle
):
    rig, pose, _ = inputs
    detections = {v: rig.camera(v).project(pose) for v in m.VIEWS}
    masks = {}
    for view in m.VIEWS:
        mask = np.zeros((3, 1200, 1200), bool)
        for body, uv in enumerate(detections[view]):
            low = np.floor(uv.min(0)).astype(int) - 5
            high = np.ceil(uv.max(0)).astype(int) + 6
            mask[body, low[1] : high[1], low[0] : high[0]] = True
        masks[view] = mask
    source = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    # S2 chest has a current top-view row, but no global two-view support.
    source[2, 1:, 1] = -1
    rays = m.RayObservations.from_points(rig, detections)
    strict, observed, _ = m.supported_nodes(pose, source, rays, rig)
    template = {k: off_bundle[0][0]["history_" + k][0].copy() for k in m.RECORD_FIELDS}
    first = dict(template, pose_xyz=pose.copy(), sources=strict, observed=observed)
    permutation = [1, 0, 2]
    second = dict(
        template,
        pose_xyz=pose[permutation].copy(),
        sources=strict[permutation].copy(),
        observed=observed[permutation].copy(),
    )
    observed_frame = FrameObservation(100, detections, {}, masks, {})
    before = signatures(detections), signatures(masks)
    anchors, facts = m.initial_anchors([first, second], observed_frame, rig, (3, 3, 3))
    assert len(anchors) == len(facts) == 2
    for view in range(3):
        np.testing.assert_array_equal(
            anchors[0].links[view],
            [[True, False, False], [False, True, False], [False, False, False]],
        )
        np.testing.assert_array_equal(
            anchors[1].links[view],
            [[False, True, False], [True, False, False], [False, False, False]],
        )
    assert (signatures(detections), signatures(masks)) == before
