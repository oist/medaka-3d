"""Four-arm evaluation on synthetic CPU evidence; no saved experiment inputs."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import mask_transport_evaluation as old  # noqa: E402
import mask_transport_gap_evaluation as m  # noqa: E402
import object_mask_transport as transport  # noqa: E402
from test_object_mask_transport import anchor, packet_fixture  # noqa: E402

from medaka3d.calib import Rig  # noqa: E402


def exact(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def first_three(result, reference):
    for key, expected in reference.items():
        if key.startswith("owner_"):
            actual = result[key][:, :3]
        elif key in ("union_loss", "union_per_view", "rendered_union", "rendered_ownership"):
            actual = result[key][:3]
        else:
            actual = result[key]
        exact(actual, expected)


def scene():
    rng = np.random.default_rng(119)
    union = rng.random((4, 3, 7))
    ownership = rng.random((4, 3, 7, 3))
    targets = np.zeros((2, 2, 3, 7, 3), np.float64)
    for e in range(2):
        for a in range(2):
            for v in range(3):
                for p in range(7):
                    targets[e, a, v, p, (e + a + v + p) % 3] = rng.random()
    foreground = rng.integers(0, 2, (3, 7)).astype(np.float64)
    weights = rng.random((3, 7)).astype(np.float32)
    valid = rng.random((3, 7)) > 0.2
    q = rng.random((4, 3, 7, 3)) * 20
    q[:, ~valid] = np.inf
    return [union, ownership, targets, foreground, weights, valid], q


@pytest.mark.parametrize("all_invalid", [False, True])
def test_first_three_legacy_metrics_are_byte_exact_with_all_evidence_and_anchors(all_invalid):
    arrays, q = scene()
    if all_invalid:
        arrays[-1][:] = False
        q[:] = np.inf
    before = [a.tobytes() for a in [*arrays, q]]
    reference = old.render_scores(arrays[0][:3], arrays[1][:3], *arrays[2:])
    result = m.render_scores(*arrays, rendered_q=q)
    first_three(result, reference)
    assert result["gap_per_body_view"].shape == (2, 4, 2, 3, 3)
    assert result["gap_loss"].shape == (2, 4, 2)
    for suffix in ("per_body_view", "per_view", "loss"):
        exact(result[f"combined_{suffix}"], result[f"owner_{suffix}"] + result[f"gap_{suffix}"])
    assert [a.tobytes() for a in [*arrays, q]] == before


def test_manual_fourth_gap_is_amodal_and_uses_fixed_three_view_denominator():
    union = np.zeros((4, 3, 2), np.float64)
    ownership = np.zeros((4, 3, 2, 3), np.float64)
    targets = np.zeros((2, 2, 3, 2, 3), np.float64)
    targets[0, 0, 0, 0, 2] = 0.5
    targets[0, 1, 2, 0, 0] = 1
    targets[1, 0, 0, 0, 2] = 0.25
    targets[1, 1, 1, 1, 0] = 1  # An invalid ray cannot create mass or a gap.
    fg = np.ones((3, 2), np.float64)
    weight = np.full((3, 2), 0.5, np.float64)
    valid = np.ones((3, 2), bool)
    valid[1, 1] = False
    q = np.ones_like(ownership)
    q[3, 0, 0, 2] = 9  # sqrt(9)-1 = 2, without ownership or visibility gating.
    q[3, 2, 0, 0] = 4
    q[:, 1, 1] = np.inf
    with np.errstate(all="raise"):
        result = m.render_scores(union, ownership, targets, fg, weight, valid, rendered_q=q)
    expected = np.zeros((2, 4, 2), np.float64)
    expected[0, 3] = [1 / 6, 1 / 6]
    expected[1, 3] = [1 / 12, 0]
    exact(result["gap_loss"], expected)
    assert result["gap_per_body_view"][0, 3, 0, 0, 2] == 0.5
    assert result["owner_loss"][0, 3, 0] == 1 / 12
    assert result["combined_loss"][0, 3, 0] == 0.25
    assert not result["target_mass"][:, :, 1, 0].any()


@pytest.mark.parametrize("bad_q", [np.inf, np.nan, -1.0])
def test_every_valid_ray_q_is_checked_even_without_any_target_mass(bad_q):
    arrays, q = scene()
    arrays[2][:] = 0
    arrays[-1][0, 0] = True
    q[:, 0, 0] = 1
    q[3, 0, 0, 0] = bad_q
    with pytest.raises(ValueError, match="finite and nonnegative"):
        m.render_scores(*arrays, rendered_q=q)


@pytest.mark.parametrize("qvalue", [0.0, 1.0, np.nextafter(1.0, 2.0), 4.0])
def test_clamp_boundary_and_numpy_formula(qvalue):
    arrays, q = scene()
    q[:] = qvalue
    result = m.render_scores(*arrays, rendered_q=q)
    expected = np.sqrt(max(qvalue, 1.0)) - 1.0
    np.testing.assert_allclose(
        result["gap_per_body_view"],
        np.broadcast_to(result["target_mass"][:, None] * expected, (2, 4, 2, 3, 3)),
        rtol=1e-15,
        atol=0,
    )


class ParallelCamera:
    def __init__(self, view):
        self.view = view

    def rays(self, xy):
        origins = np.column_stack([xy, np.zeros(len(xy))])
        return origins, np.broadcast_to([0, 0, 1], origins.shape).copy()


def image_fixture():
    kwargs = packet_fixture()
    kwargs["anchors"].append(anchor(rows=(1, 0, -1), ident=3))
    lk_packet = transport.packet_for_pixels(**kwargs)
    # A different report count changes evidence weights without changing masks,
    # pixel coordinates, query bytes, release time, or the complete anchor bank.
    votes = transport.transport_votes(
        np.array([0, 0, 0, 1, 1], np.int64),
        np.ones(5, bool),
        np.array([[1, 0], [1, 0], [0, 0], [0, 1], [0, 1]], bool),
        group_count=2,
        mask_available=True,
        native=101,
        release_horizon=108,
        query_sha256="a" * 64,
    )
    kwargs["votes_by_view"] = [votes] * 3
    ct_packet = transport.packet_for_pixels(**kwargs)
    pixels = kwargs["pixels"]
    sample = dict(
        frame_idx=np.array(101, np.int64),
        unused_pixels=pixels.copy(),
        unused_target=np.ones(pixels.shape[:2], np.float64),
        unused_weights=np.full(pixels.shape[:2], 0.25, np.float64),
        unused_fit_overlap=np.zeros(pixels.shape[:2], bool),
        unused_ray_valid=np.ones(pixels.shape[:2], bool),
    )
    pose = np.array(
        [
            [[5.0, 0.0, 25.0], [0.0, 0.0, 25.0], [-12.0, 2.0, 26.0]],
            [[7.0, 3.0, 35.0], [2.0, 3.0, 35.0], [-10.0, 5.0, 36.0]],
            [[105.0, 30.0, 45.0], [100.0, 30.0, 45.0], [88.0, 32.0, 46.0]],
        ]
    )
    poses = np.stack([pose + np.array([i * 0.17, i * -0.09, 0.0]) for i in range(4)])
    rig = Rig(tuple(ParallelCamera(v) for v in m.VIEWS), Path("synthetic-only"))
    return poses, sample, [lk_packet, ct_packet], rig


def test_real_cpu_renderer_four_arm_batch_first_three_are_byte_exact_to_frozen_scorer():
    args = image_fixture()
    before_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        reference = old.score_fixed_images(args[0][:3], *args[1:])
        result = m.score_fixed_images(*args)
    finally:
        torch.set_num_threads(before_threads)
    first_three(result, reference)
    assert result["rendered_q"].shape == (4, 3, 4, 3)
    assert result["rendered_q"].dtype == np.float64
    assert result["gap_loss"].max() > 0


@pytest.mark.parametrize("fault", ["pixels", "mask", "query", "release", "anchor", "overlap"])
def test_common_evidence_drift_rejected_before_render(monkeypatch, fault):
    poses, sample, packets, rig = image_fixture()
    if fault == "pixels":
        sample["unused_pixels"][0, 0, 0] += 1
    elif fault == "mask":
        packets[1] = dataclasses.replace(packets[1], mask_signatures=(None,) * 3)
    elif fault == "query":
        packets[1] = dataclasses.replace(packets[1], query_sha256=("b" * 64,) * 3)
    elif fault == "release":
        packets[1] = dataclasses.replace(packets[1], release_horizons=np.array([109] * 3))
    elif fault == "anchor":
        changed = dataclasses.replace(packets[1].anchors[0], reference_sha256=("b" * 64,) * 3)
        packets[1] = dataclasses.replace(packets[1], anchors=(changed, packets[1].anchors[1]))
    else:
        sample["unused_fit_overlap"][0, 0] = True

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid evidence reached renderer")

    monkeypatch.setattr(m, "RefractiveFishRenderer", forbidden)
    with pytest.raises(ValueError):
        m.score_fixed_images(poses, sample, packets, rig)


def test_four_arm_support_summary_preserves_legacy_values_and_full_denominators():
    frames = np.arange(333519, 333719, dtype=np.int64)
    observed = np.ones((200, 3, 3), bool)
    observed[15:20, 2, 2] = False
    candidates = {
        a: dict(
            frame_idx=frames.copy(),
            observed=observed.copy(),
            pose_xyz=np.zeros((200, 3, 3, 3), np.float64),
        )
        for a in m.ARMS
    }
    candidates["ct_first"]["observed"][16, 2, 2] = True
    candidates["ct_gap"]["observed"][17, 2, 2] = True
    candidates["ct_gap"]["observed"][199, 0, 0] = False
    candidates["ct_gap"]["pose_xyz"][199, 0, 0, 0] = 11
    reference = old.summarize_support({a: candidates[a] for a in old.ARMS})
    result = m.summarize_support(candidates)
    assert {a: result[a] for a in old.ARMS} == reference
    row = result["ct_gap"]
    assert row["all_node_time_denominator"] == 1800 and row["per_node_denominator"] == 600
    assert row["gain_rows"] == [dict(native=333536, body=2, node=2)]
    assert row["loss_rows"] == [dict(native=333718, body=0, node=0)]
    assert row["adjacent_latent_jump_over10mm"] == 1
    assert row["longest_missing_by_body_node"][2][2] == 2
    assert not row["independent_ID_truth"]
