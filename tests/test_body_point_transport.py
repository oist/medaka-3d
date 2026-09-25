"""Body-point ancestry, frozen admission, mass distribution and CPU ray fixtures."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import body_point_transport as m


class Camera:
    def __init__(self, view, invalid_x=()):
        self.view, self.invalid_x = view, tuple(invalid_x)

    def rays(self, xy):
        origins = np.zeros((len(xy), 3), np.float64)
        origins[:, 0] = self.view * 10
        directions = np.c_[xy[:, 0] / 13, xy[:, 1] / 17, np.ones(len(xy))]
        directions[np.isin(xy[:, 0], self.invalid_x)] = np.nan
        return origins, directions


class Rig:
    views, world_units = m.VIEWS, "mm"

    def __init__(self, invalid_x=()):
        self.cameras = {v: Camera(i, invalid_x) for i, v in enumerate(self.views)}

    def camera(self, view):
        return self.cameras[view]


def anchor(ident=0, rows=(0, 1, -1)):
    reference = dict(
        slots=[
            dict(slot=b, assigned_mask_row=r, geometry_eligible=r >= 0, reasons=[])
            for b, r in enumerate(rows)
        ]
    )
    return m.old.build_initial_anchor([reference] * 3, [3] * 3, anchor_id=ident, initial_native=100)


def point(xy, groups, masks, *, reported=None, native=101, release=107):
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    groups = np.asarray(groups, np.int64)
    membership = m.mask_memberships(xy, masks, (4, 8))
    vote = m.old.transport_votes(
        groups,
        np.ones(len(xy), bool) if reported is None else np.asarray(reported, bool),
        membership["membership"],
        group_count=3,
        mask_available=masks is not None,
        native=native,
        release_horizon=release,
        query_sha256="a" * 64,
    )
    return m.PointEvidence(vote, xy)


def fixture(*, invalid_x=(), second_anchor=False):
    masks = np.zeros((2, 4, 8), bool)
    masks[0, 0, :3] = True
    masks[1, 0, 5:7] = True
    ev = point([[0, 0], [1, 0], [2, 0], [5, 0], [6, 0], [3, 3]], [0, 0, 0, 1, 1, 2], masks)
    return dict(
        point_evidence=[ev] * 3,
        anchors=[anchor(), anchor(1, (1, 0, -1))] if second_anchor else [anchor()],
        masks_by_view=[masks.copy() for _ in range(3)],
        fit_pixels=np.tile(np.array([[0, 0], [2, 0], [5, 0], [6, 0]], np.float64), (3, 1, 1)),
        fit_foreground=np.ones((3, 4), np.float32),
        fit_weights=np.tile(np.array([2, 3, 5, 7], np.float32), (3, 1)),
        fit_ray_valid=np.ones((3, 4), bool),
        rig=Rig(invalid_x),
        native_shape_hw=(4, 8),
    )


def exact(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def test_old_packet_is_exact_and_all_anchors_receive_same_original_mass():
    args = fixture(second_anchor=True)
    packet = m.build_packet(**args)
    old = m.old.packet_for_pixels(
        [e.vote for e in args["point_evidence"]],
        args["anchors"],
        args["masks_by_view"],
        args["fit_pixels"],
        native_shape_hw=(4, 8),
    )
    m._same(old, packet.whole_mask_packet, "old packet untouched")
    expected = (old.target * args["fit_weights"].astype(np.float64)[None, :, :, None]).sum(2)
    exact(packet.old_mass, expected)
    np.testing.assert_array_equal(packet.single_count, [[[3, 2, 0]] * 3, [[2, 3, 0]] * 3])
    np.testing.assert_allclose(packet.effective_mass, packet.old_mass, atol=1e-15, rtol=0)
    assert packet.point_weight.max() == 6  # Deliberately exceeds old target's upper-one bound.
    assert packet.point_selected[0, 0, 0, 0] and packet.point_selected[1, 0, 0, 1]


def test_invalid_query_ray_keeps_single_count_and_loses_mass_without_renormalizing():
    args = fixture(invalid_x=(1,))
    packet = m.build_packet(**args)
    assert packet.single_count[0, 0, 0] == 3
    assert packet.valid_single_count[0, 0, 0] == 2
    assert packet.old_mass[0, 0, 0] == 5
    assert packet.effective_mass[0, 0, 0] == pytest.approx(10 / 3)
    assert not packet.point_weight[:, :, 1].any()
    exact(packet.ray_origins[0, 1], np.zeros(3, np.float64))
    exact(packet.ray_directions[0, 1], np.array([0, 0, 1], np.float64))


def test_native_outside_but_roundable_point_is_not_confused_with_geometric_ray_validity():
    args = fixture()
    xy = args["point_evidence"][0].xy.copy()
    xy[0, 0] = -0.1
    ev = point(xy, [0, 0, 0, 1, 1, 2], args["masks_by_view"][0])
    args["point_evidence"] = [ev] * 3
    packet = m.build_packet(**args)
    assert packet.ray_valid[0, 0] and not packet.native_inside[0, 0]
    assert packet.point_selected[0, 0, 0, 0]
    assert packet.single_count[0, 0, 0] == 3
    assert not packet.point_weight[:, :, 0].any()


@pytest.mark.parametrize("unreported", [False, True])
def test_unknown_group_cannot_enter_known_body_even_inside_its_admitted_mask(unreported):
    args = fixture()
    xy = args["point_evidence"][0].xy.copy()
    xy[-1] = [0, 0]
    ev = point(
        xy,
        [0, 0, 0, 1, 1, 2],
        args["masks_by_view"][0],
        reported=[1, 1, 1, 1, 1, not unreported],
    )
    args["point_evidence"] = [ev] * 3
    packet = m.build_packet(**args)
    assert not packet.point_selected[:, :, -1].any()
    assert not packet.point_weight[:, :, -1].any()
    if unreported:
        assert packet.old_mass[0, 0, 0] == 5
    else:
        assert packet.old_mass[0, 0, 0] == 0
        assert (
            "mask_has_unknown_group_votes"
            in packet.whole_mask_packet.assignments[0][0].group_reasons[0]
        )


def test_split_group_is_not_rescued_by_local_point_support():
    args = fixture()
    xy = args["point_evidence"][0].xy.copy()
    xy[2] = [5, 0]
    ev = point(xy, [0, 0, 0, 1, 1, 2], args["masks_by_view"][0])
    args["point_evidence"] = [ev] * 3
    packet = m.build_packet(**args)
    assert not packet.point_weight.any()
    assert "split_positive_masks" in packet.whole_mask_packet.assignments[0][0].group_reasons[0]


def test_multi_member_does_not_enter_J_and_all_reported_denominator_is_preserved():
    args = fixture()
    masks = args["masks_by_view"][0].copy()
    masks[1, 0, 2] = True
    # Other initial group no longer reports, so only its own multi point remains.
    ev = point(
        [[0, 0], [1, 0], [2, 0], [5, 0], [6, 0], [3, 3]],
        [0, 0, 0, 1, 1, 2],
        masks,
        reported=[1, 1, 1, 0, 0, 1],
    )
    args["point_evidence"] = [ev] * 3
    args["masks_by_view"] = [masks] * 3
    packet = m.build_packet(**args)
    assert packet.single_count[0, 0, 0] == 2
    assert not packet.point_selected[0, 0, 2].any()
    # x2 overlap is removed from old fit support; m=(2/3)*weight(x0).
    assert packet.old_mass[0, 0, 0] == pytest.approx(4 / 3)
    assert packet.whole_mask_packet.assignments[0][0].evidence.reported_count[0] == 3


@pytest.mark.parametrize("masks", [None, np.empty((0, 4, 8), bool)])
def test_no_mask_does_not_gain_fallback_and_keeps_absent_vs_empty(masks):
    args = fixture()
    args["masks_by_view"] = [masks] * 3
    args["point_evidence"] = [point([[0, 0], [1, 0]], [0, 0], masks)] * 3
    packet = m.build_packet(**args)
    assert not packet.point_weight.any() and not packet.old_mass.any()
    reason = "masks_unavailable" if masks is None else "no_current_masks"
    assert reason in packet.whole_mask_packet.assignments[0][0].group_reasons[0]


def test_zero_old_fit_mass_is_not_rescued_even_with_valid_reported_points():
    args = fixture()
    args["fit_ray_valid"][:] = False
    packet = m.build_packet(**args)
    assert packet.point_selected.any() and packet.ray_valid.all()
    assert not packet.old_mass.any() and not packet.point_weight.any()


def test_ragged_empty_view_and_nonfinite_queries_keep_padding_and_original_bytes():
    args = fixture()
    masks = args["masks_by_view"][0]
    xy = np.array([[0, 0], [1, 0], [np.nan, np.inf]], np.float32)
    args["point_evidence"] = [
        point(xy, [0, 0, 0], masks),
        point([], [], masks),
        args["point_evidence"][2],
    ]
    packet = m.build_packet(**args)
    assert packet.query_counts.tolist() == [3, 0, 6]
    assert np.isnan(packet.point_pixels[1]).all() and not packet.ray_valid[1].any()
    exact(packet.point_evidence[0].xy, xy)
    assert packet.whole_mask_packet.assignments[0][0].evidence.reported_count[0] == 3
    assert packet.single_count[0, 0, 0] == 2


def test_all_zero_query_views_are_finite_zero_weight_packets():
    args = fixture()
    args["point_evidence"] = [point([], [], masks) for masks in args["masks_by_view"]]
    packet = m.build_packet(**args)
    assert packet.point_pixels.shape == (3, 0, 2)
    assert packet.point_weight.shape == (1, 3, 0, 3)
    assert not packet.old_mass.any()


def test_roundtrip_rig_recheck_and_deep_immutable_ownership():
    args = fixture(second_anchor=True)
    packet = m.build_packet(**args)
    restored = m.validate_packet(dataclasses.asdict(packet), rig=args["rig"])
    m._same(packet, restored, "roundtrip")
    before = dataclasses.asdict(packet)
    args["masks_by_view"][0][:] = False
    args["fit_weights"][:] = 0
    m._same(packet, before, "caller cannot mutate packet")
    for arr in (
        packet.point_pixels,
        packet.point_weight,
        packet.old_mass,
        packet.point_evidence[0].xy,
        packet.point_evidence[0].vote.reported,
        packet.whole_mask_packet.target,
        restored.ray_origins,
    ):
        with pytest.raises(ValueError):
            arr.setflags(write=True)


@pytest.mark.parametrize(
    "field", ["point_weight", "single_count", "old_mass", "point_selected", "release_horizons"]
)
def test_saved_algebra_or_binding_mutation_is_rejected(field):
    args = fixture()
    record = dataclasses.asdict(m.build_packet(**args))
    array = record[field]
    array.flat[0] = ~array.flat[0] if array.dtype == bool else array.flat[0] + 1
    with pytest.raises(ValueError):
        m.validate_packet(record, rig=args["rig"])


def test_query_movement_without_updated_membership_is_rejected_before_ray_calculation():
    args = fixture()
    ev = args["point_evidence"][0]
    xy = ev.xy.copy()
    xy[0] = [7, 3]
    args["point_evidence"][0] = m.PointEvidence(ev.vote, xy)
    with pytest.raises(ValueError, match="membership"):
        m.build_packet(**args)


def test_rig_change_is_rejected_even_if_saved_packet_algebra_remains_consistent():
    args = fixture()
    packet = m.build_packet(**args)
    changed = Rig(invalid_x=(1,))
    with pytest.raises(ValueError, match="calibrated query rays"):
        m.validate_packet(dataclasses.asdict(packet), rig=changed)


def test_query_rays_match_frozen_renderer_cpu_constructor_exactly():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs/20260909_three_body"))
    import torch
    from renderer import RefractiveFishRenderer

    args = fixture(invalid_x=(1,))
    ev = args["point_evidence"][0]
    args["point_evidence"][1] = point([[np.nan, 0]], [2], args["masks_by_view"][1])
    packet = m.build_packet(**args)
    renderer = RefractiveFishRenderer(
        args["rig"],
        {v: packet.point_pixels[i, : packet.query_counts[i]] for i, v in enumerate(m.VIEWS)},
        device="cpu",
        dtype=torch.float64,
    )
    exact(packet.ray_origins, renderer.ray_origins.numpy())
    exact(packet.ray_directions, renderer.ray_directions.numpy())
    exact(packet.ray_valid, renderer.ray_valid.numpy())
    exact(packet.point_evidence[0].xy, ev.xy)


def test_saved_adapter_uses_the_same_native_flags_group_and_release_only():
    args = fixture()
    ev = args["point_evidence"][0]
    saved = SimpleNamespace(
        group_counts=(3, 3, 3),
        query_sha256=("a" * 64,) * 3,
        queries={
            v: {"frame_idx": np.array(100), "mask_index": ev.vote.query_group} for v in m.VIEWS
        },
        coords={
            (method, v): np.stack([ev.xy + 4, ev.xy])
            for method in ("lk", "ct_first")
            for v in m.VIEWS
        },
        flags={
            (method, v): np.stack([~ev.vote.reported, ev.vote.reported])
            for method in ("lk", "ct_first")
            for v in m.VIEWS
        },
        prepare=lambda frame: None,
        provider=lambda method: lambda native: tuple(e.vote for e in args["point_evidence"]),
    )
    adapter = m.SavedBodyPointEvidence(saved)
    provider = adapter.provider("ct_first")
    with pytest.raises(ValueError, match="cursor"):
        provider(101)
    adapter.prepare(SimpleNamespace(frame_idx=101))
    result = provider(101)
    exact(result[0].xy, ev.xy)
    assert result[0].vote.release_horizon == 107
    saved.coords["ct_first", m.VIEWS[0]][1] = 999
    exact(result[0].xy, ev.xy)
    saved.flags["ct_first", m.VIEWS[0]][1, 0] = False
    with pytest.raises(ValueError, match="reported"):
        provider(101)
