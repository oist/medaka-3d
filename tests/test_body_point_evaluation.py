"""Independent body-point metrics and seven-arm image control on synthetic CPU inputs."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import body_point_evaluation as m  # noqa: E402
import fixed_body_width_evaluation as old  # noqa: E402

from test_body_point_transport import fixture as point_fixture  # noqa: E402
from test_body_point_transport import point  # noqa: E402
from test_mask_transport_gap_evaluation import image_fixture  # noqa: E402


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def first_six(result, expected):
    shared = {"target_mass", "positive_target_mass", "ray_valid", "anchor_ids", "frame_idx"}
    outer = {"union_per_view", "union_loss", "rendered_union", "rendered_ownership", "rendered_q"}
    common = set(expected) - {"renderer_policy", "fixed_radial_sigma_mm", "reference_pose_xyz"}
    assert set(result) == common and len(common) == 19
    for key in common:
        actual = (
            result[key]
            if key in shared
            else result[key][:6]
            if key in outer
            else result[key][:, :6]
        )
        exact(actual, expected[key])


def test_real_cpu_seven_arm_images_preserve_all19_six_arm_common_fields_exact():
    poses, sample, packets, rig = image_fixture()
    poses = np.concatenate([poses, poses[:3] + np.array([0.2, 0.1, 0.0])])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    before = poses.tobytes(), {k: v.tobytes() for k, v in sample.items()}
    try:
        expected = old.score_fixed_images(poses[:6], sample, packets, rig)
        actual = m.score_fixed_images(poses, sample, packets, rig)
    finally:
        torch.set_num_threads(threads)
    first_six(actual, expected)
    assert actual["owner_loss"].shape == (2, 7, 2)
    assert before == (poses.tobytes(), {k: v.tobytes() for k, v in sample.items()})


def point_numeric(q=4):
    ownership = np.zeros((2, 7, 3, q, 3), np.float64)
    distance = np.ones_like(ownership)
    weights = np.zeros((2, 2, 3, q, 3), np.float64)
    valid = np.ones((2, 3, q), bool)
    return ownership, distance, weights, valid


def test_manual_all_evidence_arm_anchor_view_body_axes_and_fixed_divisor():
    ownership, q, weights, valid = point_numeric()
    for e in range(2):
        for a in range(2):
            for v in range(3):
                for j in range(4):
                    weights[e, a, v, j, (e + a + v + j) % 3] = (a + 1) * (v + 1) * 3 / 4
        for arm in range(7):
            ownership[e, arm] = ((arm + e) % 4) / 4
            q[e, arm] = ((arm % 3) + 1) ** 2
    valid[0, 0, 1] = False
    valid[1, 2, 2:] = False
    invalid = np.broadcast_to(~valid[:, None, :, :, None], q.shape)
    q[invalid] = np.inf
    ownership[invalid] = np.nan
    originals = [a.tobytes() for a in (ownership, q, weights, valid)]
    with np.errstate(all="raise"):
        actual = m.point_render_scores(ownership, q, weights, valid)
    owner = np.zeros((2, 7, 2, 3, 3))
    gap = np.zeros_like(owner)
    mass = np.zeros((2, 2, 3, 3))
    count = np.zeros(owner.shape, np.int64)
    for e in range(2):
        for a in range(2):
            for v in range(3):
                for j in range(4):
                    if not valid[e, v, j]:
                        continue
                    for b in range(3):
                        w = weights[e, a, v, j, b]
                        mass[e, a, v, b] += w
                        for arm in range(7):
                            owner[e, arm, a, v, b] += w * (1 - ownership[e, arm, v, j, b]) ** 2
                            gap[e, arm, a, v, b] += w * (max(q[e, arm, v, j, b], 1) ** 0.5 - 1)
                            count[e, arm, a, v, b] += int(w > 0 and q[e, arm, v, j, b] > 1)
    exact(actual["point_owner_per_body_view"], owner)
    exact(actual["point_gap_per_body_view"], gap)
    exact(actual["point_combined_per_body_view"], owner + gap)
    exact(actual["point_owner_loss"], owner.sum((-1, -2)) / 3)
    exact(actual["point_gap_loss"], gap.sum((-1, -2)) / 3)
    exact(actual["point_combined_loss"], (owner + gap).sum((-1, -2)) / 3)
    exact(actual["point_mass_by_view_body"], mass)
    exact(actual["positive_gap_point_count"], count)
    assert originals == [a.tobytes() for a in (ownership, q, weights, valid)]


def test_invalid_loses_mass_without_renormalization_and_weight_above_one_is_valid():
    owner, q, weight, valid = point_numeric(q=2)
    owner[:] = 0
    q[:] = 9
    weight[:, :, :, 0, 0] = 6
    weight[:, :, :, 1, 0] = 6
    valid[:, :, 1] = False
    q[:, :, :, 1, :] = np.inf
    result = m.point_render_scores(owner, q, weight, valid)
    assert np.all(result["point_owner_loss"] == 6)
    assert np.all(result["point_gap_loss"] == 12)
    assert np.all(result["point_combined_loss"] == 18)
    assert np.all(result["point_mass_by_view_body"][..., 0] == 6)
    assert np.all(result["positive_gap_point_count"][..., 0] == 1)


@pytest.mark.parametrize(
    "field,bad", [("q", np.inf), ("q", np.nan), ("q", -1.0), ("owner", np.inf), ("owner", np.nan)]
)
def test_valid_nonfinite_checked_even_when_all_observation_weights_are_zero(field, bad):
    owner, q, weight, valid = point_numeric()
    (q if field == "q" else owner)[1, 6, 2, 0, 2] = bad
    with pytest.raises(ValueError, match="finite"):
        m.point_render_scores(owner, q, weight, valid)


@pytest.mark.parametrize(
    "fault", ["overlap", "negative", "nonfinite", "wrong_arms", "wrong_anchor", "wrong_valid"]
)
def test_point_shape_and_mass_contract(fault):
    owner, q, weight, valid = point_numeric()
    if fault == "overlap":
        weight[0, 0, 0, 0, :2] = 1
    elif fault == "negative":
        weight[0, 0, 0, 0, 0] = -1
    elif fault == "nonfinite":
        weight[0, 0, 0, 0, 0] = np.nan
    elif fault == "wrong_arms":
        owner = owner[:, :6]
    elif fault == "wrong_anchor":
        weight = weight[:, :0]
    else:
        valid = valid.astype(np.int64)
    with pytest.raises(ValueError):
        m.point_render_scores(owner, q, weight, valid)


def test_zero_queries_and_all_invalid_keep_every_method_anchor_as_zero_not_missing():
    for qcount in (0, 3):
        owner, q, weight, valid = point_numeric(q=qcount)
        valid[:] = False
        owner[:] = np.nan
        q[:] = np.inf
        with np.errstate(all="raise"):
            result = m.point_render_scores(owner, q, weight, valid)
        assert result["point_owner_loss"].shape == (2, 7, 2)
        assert result["point_mass_by_view_body"].shape == (2, 2, 3, 3)
        assert all(not a.any() and np.isfinite(a).all() for a in result.values())


def point_packets(*, empty=False):
    args = point_fixture(second_anchor=True)
    first = m.point_transport.build_packet(**args)
    masks = args["masks_by_view"][0]
    if empty:
        args["point_evidence"] = [point([], [], masks)] * 3
        first = m.point_transport.build_packet(**args)
    else:
        xy = args["point_evidence"][0].xy.copy()
        xy[:, 0] += 0.125
        args["point_evidence"] = [point(xy, [0, 0, 0, 1, 1, 2], masks)] * 3
    second = m.point_transport.build_packet(**args)
    source_pose = image_fixture()[0][0]
    poses = np.stack([source_pose + np.array([0.1 * i, -0.2 * i, 0]) for i in range(7)])
    return poses, [first, second], args["rig"]


def test_actual_cpu_two_different_point_domains_and_both_anchor_banks_match_direct_render():
    poses, packets, rig = point_packets()
    assert not np.array_equal(packets[0].point_pixels, packets[1].point_pixels)
    originals = [poses.tobytes()] + [p.point_weight.tobytes() for p in packets]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = m.score_body_points(poses, packets, rig)
        for e, packet in enumerate(packets):
            renderer = m.RefractiveFishRenderer(
                rig,
                {
                    v: packet.point_pixels[i, : packet.query_counts[i]]
                    for i, v in enumerate(m.VIEWS)
                },
                m.RendererConfig(width_ratio=0.06),
                device="cpu",
                dtype=torch.float64,
            )
            with torch.no_grad():
                rendered = renderer.render(torch.as_tensor(poses, dtype=torch.float64))
            exact(result["point_rendered_ownership"][e], rendered.ownership.numpy())
            exact(result["point_rendered_q"][e], rendered.per_fish_mahalanobis_sq.numpy())
    finally:
        torch.set_num_threads(threads)
    manual = m.point_render_scores(
        result["point_rendered_ownership"],
        result["point_rendered_q"],
        np.stack([p.point_weight for p in packets]),
        np.stack([p.ray_valid for p in packets]),
    )
    for key, value in manual.items():
        exact(result[key], value)
    assert result["point_owner_loss"].shape == (2, 7, 2)
    assert result["point_owner_per_body_view"].shape == (2, 7, 2, 3, 3)
    assert not np.array_equal(result["point_rendered_q"][0], result["point_rendered_q"][1])
    assert originals == [poses.tobytes()] + [p.point_weight.tobytes() for p in packets]


@pytest.mark.parametrize(
    "fault",
    ["fit_pixels", "fit_weight", "fit_target", "fit_valid", "query", "release", "anchor", "ray"],
)
def test_body_point_shared_input_drift_is_rejected(fault, monkeypatch):
    poses, packets, rig = point_packets()
    name = {
        "fit_pixels": "pixels",
        "fit_weight": "fit_weights",
        "fit_target": "fit_foreground",
        "fit_valid": "fit_ray_valid",
        "release": "release_horizons",
        "ray": "ray_origins",
    }.get(fault)
    if name:
        changed = getattr(packets[1], name).copy()
        changed.flat[0] = not changed.flat[0] if changed.dtype == bool else changed.flat[0] + 1
        packets[1] = dataclasses.replace(packets[1], **{name: changed})
    elif fault == "query":
        packets[1] = dataclasses.replace(packets[1], query_sha256=("b" * 64,) * 3)
    else:
        whole = packets[1].whole_mask_packet
        a = dataclasses.replace(whole.anchors[0], reference_sha256=("b" * 64,) * 3)
        packets[1] = dataclasses.replace(
            packets[1],
            whole_mask_packet=dataclasses.replace(whole, anchors=(a, *whole.anchors[1:])),
        )
    if fault != "ray":

        def forbidden(*args, **kwargs):
            raise AssertionError("different common evidence reached renderer")

        monkeypatch.setattr(m, "RefractiveFishRenderer", forbidden)
    with pytest.raises(ValueError):
        m.score_body_points(poses, packets, rig)


def test_zero_query_packets_never_initialize_renderer(monkeypatch):
    poses, packets, rig = point_packets(empty=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty point packet reached renderer")

    monkeypatch.setattr(m, "RefractiveFishRenderer", forbidden)
    result = m.score_body_points(poses, packets, rig)
    assert result["point_rendered_q"].shape == (2, 7, 3, 0, 3)
    assert result["point_owner_loss"].shape == (2, 7, 2) and not result["point_owner_loss"].any()
