"""New body-point/state boundary on tiny synthetic CPU fixtures only."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import body_point_tracker as m  # noqa: E402
from renderer import RefractiveFishRenderer  # noqa: E402
from tracker import Proposal  # noqa: E402

import test_mask_transport_tracker as toy  # noqa: E402


def tracker(data, coefficient=0, provider=None, cls=m.BodyPointTracker):
    def forbidden(native):
        raise AssertionError("off/t0 must not request point evidence")

    return cls(
        *data,
        first_native=100,
        group_counts=(0, 0, 0),
        query_sha256=("a" * 64,) * 3,
        vote_provider=provider or forbidden,
        input_binding={"fixture": "new body-point synthetic CPU only"},
        coefficient=coefficient,
    )


@pytest.fixture(scope="module")
def qualified():
    data = toy.inputs.__wrapped__()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = tracker(data, cls=m.previous.MaskTransportGapTracker)
        new = tracker(data)
        old.begin()
        new.begin()
        for native in (100, 101):
            frame = toy.frame(data, native)
            old.advance(frame)
            new.advance(frame)
        old_bundle, new_bundle = old.export_transport_state(), new.export_transport_state()
        return data, old_bundle, new_bundle, old.finish(), new.finish()
    finally:
        torch.set_num_threads(old_threads)


def test_off_t0_and_later_all39_all21_exact_new_domain_is_explicit(qualified):
    _, old, new, old_candidate, new_candidate = qualified
    assert len(new[0]) == 39 and len(new_candidate) == 21
    assert toy.signatures(old[0]) == toy.signatures(new[0])
    assert toy.signatures(old_candidate) == toy.signatures(new_candidate)
    policy = new[1]["context"]["mask_transport"]
    assert policy["schema"] == new[3]["schema"] == m.SCHEMA != old[3]["schema"]
    assert policy["objective"] == new[3]["objective"] == m.OBJECTIVE
    before, after = (m.previous.decode_sidecar(old[2], old[3]), m.decode_sidecar(new[2], new[3]))
    a, b = {}, {}
    assert m._pack(before, a) == m._pack(after, b)
    assert toy.signatures(a) == toy.signatures(b)
    assert new_candidate["identity_review"][-1, 2]
    for name in ("__init__", "_advance_owned", "_proposals", "restore_transport_state", "finish"):
        assert getattr(m.BodyPointTracker, name) is getattr(
            m.previous.MaskTransportGapTracker, name
        )


def test_joint_restore_then_continue_keeps_all_core_and_anchor_fields(qualified):
    data, _, bundle, _, _ = qualified
    before = toy.signatures(bundle[0]), toy.signatures(bundle[2]), copy.deepcopy(bundle[3])
    restored = tracker(data)
    restored.restore_transport_state(*bundle)
    copy_bundle = restored.export_transport_state()
    assert toy.signatures(copy_bundle[0]) == before[0]
    assert toy.signatures(copy_bundle[2]) == before[1]
    assert restored.initialize_count == restored.restore_count == 1
    fresh = tracker(data)
    fresh.begin()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for native in (100, 101, 102):
            frame = toy.frame(data, native)
            fresh.advance(frame)
            if native == 102:
                restored.advance(frame)
        assert toy.signatures(fresh.export_transport_state()[0]) == toy.signatures(
            restored.export_transport_state()[0]
        )
        assert toy.signatures(fresh.finish()) == toy.signatures(restored.finish())
    finally:
        torch.set_num_threads(old_threads)
    assert (toy.signatures(bundle[0]), toy.signatures(bundle[2]), bundle[3]) == before


def test_old_gap_schema_not_accepted_as_new_point_domain(qualified):
    data, old, new, _, _ = qualified
    with pytest.raises(ValueError):
        tracker(data).restore_transport_state(*old)
    with pytest.raises(ValueError):
        m.decode_sidecar(old[2], old[3])
    with pytest.raises(ValueError):
        m.previous.decode_sidecar(new[2], new[3])


@pytest.mark.parametrize(
    "fault", ["domain", "gap", "unselected_parent", "anchor", "off_packet", "binding"]
)
def test_rehashed_new_sidecar_tamper_refused_before_state_initialization(qualified, fault):
    data, _, bundle, _, _ = qualified
    core, meta, arrays, header = bundle
    content = m.decode_sidecar(arrays, header)
    changed = copy.deepcopy(header)
    if fault == "domain":
        changed["objective"]["domain"] = "whole mask"
    elif fault == "gap":
        content["fits"][1]["owner"]["gap_loss"][0] = 0.5
    elif fault == "unselected_parent":
        fit = content["fits"][1]
        i = next(i for i in range(len(fit["parent"])) if i not in fit["selected"])
        fit["parent_cost"][i] += 1
    elif fault == "anchor":
        content["active_anchor_ids"][0] += 1
    elif fault == "off_packet":
        content["fits"][1]["packet"] = {"schema": "unbound"}
    new_arrays = {}
    changed["tree"] = m._pack(content, new_arrays)
    changed["array_signatures"] = toy.signatures(new_arrays)
    changed.pop("content_sha256")
    changed["content_sha256"] = m.json_hash(changed)
    target = tracker(data)
    if fault == "binding":
        target._binding["fixture"] = "other input"
    with pytest.raises(ValueError):
        target.restore_transport_state(core, meta, new_arrays, changed)
    assert target.initialize_count == target.restore_count == 0


def active_diagnostics(batch, q, steps=1):
    weight = np.zeros((2, 3, q, 3), np.float64)
    weight[0, ..., 0], weight[1, ..., 2] = 3.0, 6.0
    anchors = np.array([1, 0, 1], np.int64)[:batch]
    mass = weight[anchors].sum(2) / 3
    gap_detail = mass.copy()
    gap = gap_detail.sum((1, 2))
    owner = gap / 2
    base = np.arange(batch, dtype=np.float64) + 1
    total = base + 3 * (owner + gap)
    d = dict(
        active=True,
        reason="fixed_body_point_owner_plus_gap",
        coefficient=3.0,
        base_score=base,
        owner_loss=owner,
        gap_loss=gap,
        total_score=total,
        original_loss_parts=np.zeros((batch, 5)),
        best_iteration=np.zeros(batch, np.int64),
        target_mass=mass.sum((1, 2)),
        target_diagnostics_computed=True,
        final_gap_by_view_body=gap_detail,
        target_mass_by_view_body=mass,
        positive_gap_pixel_count=np.count_nonzero(weight[anchors], axis=2),
        iteration_owner_gradient_norm=np.zeros((steps, batch)),
        iteration_gap_gradient_norm=np.ones((steps, batch)),
    )
    for key in ("base_score", "owner_loss", "gap_loss", "total_score"):
        d["iteration_" + key] = np.broadcast_to(d[key], (steps + 1, batch)).copy()
    packet = dict(
        point_pixels=np.zeros((3, q, 2)), point_weight=weight, ray_valid=np.ones((3, q), bool)
    )
    fit = dict(
        owner=d,
        packet=packet,
        parent=np.zeros(batch, np.int64),
        cost=total.copy(),
        parts=d["original_loss_parts"].copy(),
        candidate_anchor=anchors,
        fit_pixels=np.zeros((3, 1, 2)),
    )
    return fit


@pytest.mark.parametrize(
    "fault", [None, "old_reason", "mass", "count", "combined", "extra", "inactive_positive"]
)
def test_query_domain_diagnostic_uses_actual_query_count_and_never_mutates_saved_fit(fault):
    fit = active_diagnostics(3, 4)
    if fault == "old_reason":
        fit["owner"]["reason"] = "fixed_transport_owner_plus_gap"
    elif fault == "mass":
        fit["owner"]["target_mass_by_view_body"][0, 0, 2] += 1
    elif fault == "count":
        fit["owner"]["positive_gap_pixel_count"][0, 0, 2] = 5
    elif fault == "combined":
        fit["owner"]["gap_loss"][0] += 1
    elif fault == "extra":
        fit["owner"]["invented"] = 1
    elif fault == "inactive_positive":
        fit["owner"]["active"] = False
    a, b = {}, {}
    tree_before = m._pack(fit, a)
    if fault:
        with pytest.raises(ValueError):
            m.validate_body_point_diagnostics(fit, coefficient=3.0, opt_steps=1)
    else:
        m.validate_body_point_diagnostics(fit, coefficient=3.0, opt_steps=1)
        assert (
            fit["fit_pixels"].shape[1] == 1
        )  # Four query points exceed this unchanged fit sample count.
        assert fit["owner"]["reason"] == "fixed_body_point_owner_plus_gap"
    assert m._pack(fit, b) == tree_before and toy.signatures(a) == toy.signatures(b)


@pytest.mark.parametrize("fault", [None, "native", "release", "query", "group"])
def test_fit_packet_and_gather_follow_parent_anchor_never_permutation(monkeypatch, fault):
    data = toy.inputs.__wrapped__()
    evidence = tuple(
        SimpleNamespace(
            vote=SimpleNamespace(
                native=101, release_horizon=115, group_count=0, query_sha256="a" * 64
            )
        )
        for _ in range(3)
    )
    if fault:
        setattr(
            evidence[0].vote,
            {
                "native": "native",
                "release": "release_horizon",
                "query": "query_sha256",
                "group": "group_count",
            }[fault],
            "b" * 64 if fault == "query" else 999,
        )
    target = tracker(data, coefficient=3, provider=lambda native: evidence)
    target.begin()
    target._history = [[]]
    target._anchor_ids = np.array([1, 0], np.int64)
    target._anchors = ("anchor0", "anchor1")
    target._transport_frame = SimpleNamespace(frame_idx=101, masks={v: None for v in m.VIEWS})
    pose = data[1]
    plan = np.zeros((3, 3, 3), np.int64)
    proposals = [
        Proposal(pose.copy(), pose.copy(), plan.copy(), parent, "predict", perm)
        for parent, perm in ((0, (2, 1, 0)), (1, (1, 0, 2)), (0, (0, 1, 2)))
    ]
    states = [SimpleNamespace(cost=2.0), SimpleNamespace(cost=3.0)]
    samples = SimpleNamespace(
        pixels={v: np.array([[10.0, 20.0]]) for v in m.VIEWS},
        target=np.ones((3, 1), np.float32),
        weights=np.ones((3, 1)),
        foreground=np.zeros((3, 1), bool),
    )
    active = active_diagnostics(3, 4, steps=data[2].opt_steps)
    packet = SimpleNamespace(anchor_ids=np.arange(2), **active["packet"])
    seen = []

    def build(ev, anchors, masks, px, **kwargs):
        assert ev == evidence and anchors is target._anchors
        assert kwargs["fit_foreground"] is samples.target
        assert kwargs["fit_foreground"].dtype == np.float32 and np.all(
            kwargs["fit_foreground"] == 1
        )
        assert np.array_equal(kwargs["fit_ray_valid"], m.fit_ray_valid(data[0], px))
        seen.append("build")
        return packet

    def fit(
        instance,
        supplied,
        parents,
        rays,
        targets,
        supplied_packet,
        *,
        candidate_anchor,
        coefficient,
    ):
        assert (
            instance is target
            and supplied is proposals
            and parents is states
            and targets is samples
        )
        assert supplied_packet is packet and coefficient == 3
        np.testing.assert_array_equal(candidate_anchor, [1, 0, 1])
        seen.append("fit")
        instance.last_gap_fit = active["owner"]
        return (
            np.stack([p.pose for p in supplied]),
            active["cost"],
            active["parts"],
            np.ones((3, 3), bool),
        )

    monkeypatch.setattr(m.point_transport, "build_packet", build)
    monkeypatch.setattr(m.fitter, "fit_with_body_points", fit)
    monkeypatch.setattr(
        m, "packet_dict", lambda p: vars(p) if isinstance(p, SimpleNamespace) else p
    )
    if fault:
        with pytest.raises(ValueError, match="binding"):
            target._fit(proposals, states, None, samples)
        assert seen == []
    else:
        result = target._fit(proposals, states, None, samples)
        assert seen == ["build", "fit"] and result[1] is active["cost"]
        np.testing.assert_array_equal(target._fit_capture["candidate_anchor"], [1, 0, 1])
        np.testing.assert_array_equal(
            target._fit_capture["path_cost"], active["cost"] + [2.0, 3.0, 2.0]
        )
        assert target._fit_capture["packet"] is packet


def test_frozen_ray_boundary_nan_and_nonfinite_rules_are_identical():
    class Camera:
        def rays(self, pixels):
            n = len(pixels)
            o = np.zeros((n, 3))
            d = np.zeros((n, 3))
            d[:, 2] = 1
            for i, p in enumerate(pixels):
                if p[0] == 1:
                    d[i, 2] = 1e-12
                elif p[0] == 2:
                    d[i, 2] = np.nextafter(1e-12, np.inf)
                elif p[0] == 3:
                    o[i, 0] = np.nan
                elif p[0] == 4:
                    d[i, 0] = np.inf
            return o, d

    rig = SimpleNamespace(views=m.VIEWS, world_units="mm", camera=lambda view: Camera())
    pixels = np.broadcast_to(
        np.array([[0.0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [np.nan, 0]]), (3, 6, 2)
    ).copy()
    actual = m.fit_ray_valid(rig, pixels)
    renderer = RefractiveFishRenderer(
        rig, {v: pixels[i] for i, v in enumerate(m.VIEWS)}, device="cpu", dtype=torch.float64
    )
    assert actual.tobytes() == renderer.ray_valid.numpy().tobytes()
    np.testing.assert_array_equal(actual[0], [True, False, True, False, False, False])


@pytest.fixture(scope="module")
def real_active_packet_bundle():
    data = toy.inputs.__wrapped__()
    rig, pose, _ = data
    current = {}

    def frame(native):
        result = toy.frame(data, native)
        for v in m.VIEWS:
            masks = np.zeros((3, 1200, 1200), bool)
            for body, points in enumerate(rig.camera(v).project(pose)):
                low = np.floor(points.min(0)).astype(int) - 5
                high = np.ceil(points.max(0)).astype(int) + 6
                masks[body, low[1] : high[1], low[0] : high[0]] = True
            result.masks[v] = masks
        current["frame"] = result
        return result

    def provider(native):
        assert native == current["frame"].frame_idx and native > 100
        values = []
        for view in m.VIEWS:
            xy = rig.camera(view).project(pose[:, :2]).reshape(-1, 2).astype(np.float32)
            masks = current["frame"].masks[view]
            member = m.point_transport.mask_memberships(xy, masks, (1200, 1200))["membership"]
            vote = m.transport.transport_votes(
                np.repeat(np.arange(3, dtype=np.int64), 2),
                np.ones(6, bool),
                member,
                group_count=3,
                mask_available=True,
                native=native,
                release_horizon=115,
                query_sha256="a" * 64,
            )
            values.append(m.point_transport.PointEvidence(vote, xy))
        return tuple(values)

    def make(coefficient=3, cls=m.BodyPointTracker):
        return cls(
            *data,
            first_native=100,
            group_counts=(3, 3, 3),
            query_sha256=("a" * 64,) * 3,
            vote_provider=provider,
            input_binding={"fixture": "real synthetic packet/full 3-mask geometry"},
            coefficient=coefficient,
        )

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = make(0, m.previous.MaskTransportGapTracker)
        new = make()
        old.begin()
        new.begin()
        initial = frame(100)
        old.advance(initial)
        new.advance(initial)
        t0old, t0new = old.export_transport_state(), new.export_transport_state()
        assert toy.signatures(t0old[0]) == toy.signatures(t0new[0])
        new.advance(frame(101))
        bundle = new.export_transport_state()
        return data, bundle, make, frame, new
    finally:
        torch.set_num_threads(before)


def test_actual_packet_fit_sidecar_and_restore_preserve_all_query_evidence(
    real_active_packet_bundle,
):
    _, bundle, make, frame, fresh = real_active_packet_bundle
    content = m.decode_sidecar(bundle[2], bundle[3])
    fit = content["fits"][1]
    assert fit["owner"]["active"] and fit["owner"]["reason"] == "fixed_body_point_owner_plus_gap"
    assert fit["packet"]["query_counts"].tolist() == [6, 6, 6]
    assert np.any(fit["packet"]["old_mass"] > 0)
    assert len(fit["packet"]["whole_mask_packet"]["anchors"]) == len(content["anchors"])
    before = toy.signatures(bundle[0]), toy.signatures(bundle[2]), copy.deepcopy(bundle[3])
    resumed = make()
    resumed.restore_transport_state(*bundle)
    after = resumed.export_transport_state()
    assert toy.signatures(after[0]) == before[0] and toy.signatures(after[2]) == before[1]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        obs = frame(102)
        fresh.advance(obs)
        resumed.advance(obs)
        assert toy.signatures(fresh.export_transport_state()[0]) == toy.signatures(
            resumed.export_transport_state()[0]
        )
    finally:
        torch.set_num_threads(threads)
    assert (toy.signatures(bundle[0]), toy.signatures(bundle[2]), bundle[3]) == before


@pytest.mark.parametrize("fault", ["query_xy", "weight", "fit_foreground", "release", "anchor"])
def test_rehashed_active_packet_tamper_rejected_before_restore(real_active_packet_bundle, fault):
    _, bundle, make, _, _ = real_active_packet_bundle
    core, metadata, arrays, header = bundle
    content = m.decode_sidecar(arrays, header)
    p = content["fits"][1]["packet"]
    if fault == "query_xy":
        p["point_evidence"][0]["xy"][0, 0] += 1
    elif fault == "weight":
        p["point_weight"][0, 0, 0, 0] += 1
    elif fault == "fit_foreground":
        content["fits"][1]["fit_target"][0, 0] = 1 - content["fits"][1]["fit_target"][0, 0]
    elif fault == "release":
        p["release_horizons"][0] += 1
    else:
        p["whole_mask_packet"]["anchors"][0]["links"][0][0, 0] = ~p["whole_mask_packet"]["anchors"][
            0
        ]["links"][0][0, 0]
    changed = copy.deepcopy(header)
    new_arrays = {}
    changed["tree"] = m._pack(content, new_arrays)
    changed["array_signatures"] = toy.signatures(new_arrays)
    changed.pop("content_sha256")
    changed["content_sha256"] = m.json_hash(changed)
    target = make()
    with pytest.raises(ValueError):
        target.restore_transport_state(core, metadata, new_arrays, changed)
    assert target.initialize_count == target.restore_count == 0
