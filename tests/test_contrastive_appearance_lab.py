"""Crop geometry, decisions and label-free sampling of the contrastive appearance lab."""

import importlib.util
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from medaka3d.stages import Identifier
from medaka3d.types import Identity

spec = importlib.util.spec_from_file_location(
    "test_contrastive_appearance",
    Path(__file__).parents[1] / "labs/20261106_contrastive_appearance_link/appearance.py",
)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def synthetic_fish(heading_deg, centre=(300.0, 200.0), length=100.0):
    """Image with a grey body, a red mouth end and a blue tail end."""
    import cv2

    image = np.zeros((480, 640, 3), np.uint8)
    h = np.deg2rad(heading_deg)
    axis = np.array([np.cos(h), np.sin(h)])
    mouth = np.asarray(centre) + axis * length / 2
    tail = np.asarray(centre) - axis * length / 2
    cv2.line(image, tuple(int(v) for v in tail), tuple(int(v) for v in mouth), (128, 128, 128), 9)
    cv2.circle(image, tuple(int(v) for v in mouth), 6, (0, 0, 255), -1)
    cv2.circle(image, tuple(int(v) for v in tail), 6, (255, 0, 0), -1)
    points = np.array([mouth, np.asarray(centre), tail])
    return image, points


@pytest.mark.parametrize("heading", [0, 37, 90, 180, -120])
def test_crop_points_the_mouth_right_at_fixed_scale(heading):
    image, points = synthetic_fish(heading)
    centre, angle, length = lab.crop_pose(points)
    assert np.allclose(centre, points[1], atol=1e-6)
    assert length == pytest.approx(100.0, abs=1e-3)
    crop = lab.extract_crop(image, centre, angle)
    assert crop.shape == (lab.OUT_PX, lab.OUT_PX, 3)
    red = np.argwhere((crop[..., 2] > 100) & (crop[..., 0] < 60))
    blue = np.argwhere((crop[..., 0] > 100) & (crop[..., 2] < 60))
    half = 50 * lab.OUT_PX / lab.NATIVE_CROP_PX
    mid = lab.OUT_PX / 2
    assert abs(red[:, 1].mean() - (mid + half)) < 2.5 and abs(red[:, 0].mean() - mid) < 2.5
    assert abs(blue[:, 1].mean() - (mid - half)) < 2.5 and abs(blue[:, 0].mean() - mid) < 2.5


def test_crop_pose_falls_back_and_refuses_single_nodes():
    pts = np.array([[np.nan, np.nan], [10.0, 0.0], [0.0, 0.0]])
    centre, angle, length = lab.crop_pose(pts)
    assert np.allclose(centre, [10, 0]) and angle == pytest.approx(0.0) and np.isnan(length)
    assert lab.crop_pose(np.array([[np.nan, np.nan], [1.0, 1.0], [np.nan, np.nan]])) is None


def test_overlap_ignores_own_row_and_missing_points():
    frame = np.full((3, 3, 2), np.nan)
    frame[0] = [[100, 100], [110, 100], [120, 100]]
    frame[1] = [[500, 500], [510, 500], [520, 500]]
    assert not lab.overlaps(frame, 0, np.array([110.0, 100.0]))
    frame[2, 1] = [150, 100]
    assert lab.overlaps(frame, 0, np.array([110.0, 100.0]))


def test_decide_returns_permutation_margin_and_refuses_missing():
    d = np.full((3, 3), 5.0)
    d[[0, 1, 2], [1, 0, 2]] = 1.0
    perm, margin = lab.decide(d)
    assert list(perm) == [1, 0, 2] and margin > 0
    d[0, 0] = np.nan
    assert lab.decide(d)[0] is None


def test_constrained_labels_recover_permuted_identities():
    rng = np.random.default_rng(1)
    centres = np.eye(3) * 5
    truth = [np.array([0, 1, 2])]
    descs = [[{"cam": (centres[k] + rng.normal(0, 0.2, 3), 10)} for k in truth[0]]]
    for _ in range(20):
        perm = rng.permutation(3)
        truth.append(perm)
        descs.append([{"cam": (centres[k] + rng.normal(0, 0.2, 3), 10)} for k in perm])
    labels, _ = lab.constrained_labels(descs, np.array([0, 1, 2]))
    assert np.array_equal(labels, np.array(truth))
    descs[3][1] = None
    labels, _ = lab.constrained_labels(descs, np.array([0, 1, 2]))
    assert (labels[3] == -1).all()


def test_pair_sampler_uses_geometry_structure_only():
    interval = np.repeat(np.arange(4), 30)
    slot = np.tile(np.repeat(np.arange(3), 10), 4)
    fragment = interval * 3 + slot
    usable = np.ones(interval.size, bool)
    usable[:10] = False  # interval 0, slot 0 unusable
    draw, _ = lab.pair_sampler(fragment, interval, usable, seed=0)
    a, p, n = draw(500)
    assert np.all(fragment[a] == fragment[p]) and np.all(a != p)
    assert np.all(interval[a] == interval[n]) and np.all(fragment[a] != fragment[n])
    assert usable[np.concatenate([a, p, n])].all()


def test_training_smoke_and_embedding_shape():
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, size=(60, lab.OUT_PX, lab.OUT_PX, 3), dtype=np.uint8)
    interval = np.repeat(np.arange(2), 30)
    slot = np.tile(np.repeat(np.arange(3), 10), 2)
    model, stats, log = lab.train_contrastive(
        crops, interval * 3 + slot, interval, np.ones(60, bool), steps=2, pairs=4, device="cpu"
    )
    assert len(log) == 2 and np.isfinite(log[-1]["loss"])
    assert lab.embed(model, crops, stats).shape == (60, 16)


def test_color_features_are_bounded():
    image, points = synthetic_fish(15)
    centre, angle, length = lab.crop_pose(points)
    feats = lab.color_features(lab.extract_crop(image, centre, angle), length)
    assert feats.shape == (6,) and np.all((feats >= 0) & (feats <= 1))


def test_to_crop_matches_the_crop_builder():
    _, points = synthetic_fish(37)
    centre, angle, _ = lab.crop_pose(points)
    q = lab.to_crop(points, centre, angle)
    half = 50 * lab.OUT_PX / lab.NATIVE_CROP_PX
    mid = lab.OUT_PX / 2
    assert np.allclose(q[lab.MOUTH], [mid + half, mid], atol=1e-6)
    assert np.allclose(q[lab.TAIL], [mid - half, mid], atol=1e-6)


def test_band_mask_follows_the_body_and_extends_past_the_tail():
    pts = np.array([[56.0, 32.0], [48.0, 32.0], [16.0, 32.0]])  # length 40 crop px = 100 native
    mask = lab.band_mask(pts, 100.0)
    assert mask[32, 30] and mask[32, 16] and not mask[10, 30]
    half = lab.MASK_HALF_WIDTH * 40
    end = 16 - lab.MASK_TAIL_EXTENSION * 40 - half  # round cap of the same half-width
    assert mask[32, int(end + 1.5)] and not mask[32, int(end - 2.5)]
    assert mask[int(32 + half - 1.5), 30] and not mask[int(32 + half + 2.5), 30]
    assert not lab.band_mask(
        np.array([[1.0, 1.0], [np.nan, np.nan], [np.nan, np.nan]]), 100.0
    ).any()


def test_mask_crop_fills_other_rows_and_measures_own_occlusion():
    crop = np.full((lab.OUT_PX, lab.OUT_PX, 3), 200, np.uint8)
    centre, heading = np.array([300.0, 200.0]), 0.0
    frame = np.full((2, 3, 2), np.nan)
    frame[0] = [[350.0, 200.0], [330.0, 200.0], [250.0, 200.0]]
    frame[1] = [[300.0, 150.0], [300.0, 140.0], [300.0, 60.0]]  # above, not touching
    out, occluded = lab.mask_crop(crop, frame, 0, centre, heading, np.array([0, 0, 0]))
    assert occluded == 0.0 and (out == 0).any() and (out[32, 32] == 200).all()
    frame[1] = [[300.0, 260.0], [300.0, 240.0], [300.0, 140.0]]  # crosses the own body
    _, occluded = lab.mask_crop(crop, frame, 0, centre, heading, np.array([0, 0, 0]))
    assert 0.0 < occluded < 1.0


def test_assign_partial_eliminates_and_abstains():
    centroids = {"cam": [np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([0.0, 5.0])]}
    near = [{"cam": (np.array([0.1, 0.0]), 9)}, {"cam": (np.array([0.0, 4.9]), 9)}]
    names = lab.assign_partial([near[0], None, near[1]], centroids)
    assert list(names) == [0, 1, 2]
    names = lab.assign_partial([None, near[1], None], centroids)
    assert list(names) == [-1, 2, -1]
    assert lab.assign_partial([None, None, None], centroids) is None
    assert lab.assign_partial([near[0], near[0], near[1]], centroids) is None


def test_frame_names_carry_warnings_only_when_flanks_agree():
    lower = [(0, 0, 10), (1, 12, 20), (2, 25, 30)]
    names = {0: np.array([0, 1, 2]), 1: np.array([0, 2, 1]), 2: np.array([0, 2, -1])}
    out = lab.frame_names(32, lower, names)
    assert (out[5] == [0, 1, 2]).all() and (out[15] == [0, 2, 1]).all()
    assert (out[10] == [0, -1, -1]).all()
    assert (out[22] == [0, 2, -1]).all()
    assert (out[31] == -1).all()
    strict = lab.frame_names(32, lower, names, carry_warnings=False)
    assert (strict[10] == -1).all() and (strict[22] == -1).all()


def test_bridging_skips_unnamed_intervals():
    lower = [(0, 0, 10), (1, 12, 13), (2, 15, 20), (3, 22, 30)]
    names = {
        0: np.array([0, 1, 2]),
        1: np.array([-1, -1, -1]),
        2: np.array([0, 1, 2]),
        3: np.array([0, 2, 1]),
    }
    out = lab.frame_names_bridged(32, lower, names)
    assert (out[11] == [0, 1, 2]).all() and (out[12] == [0, 1, 2]).all()
    assert (out[14] == [0, 1, 2]).all()
    assert (out[21] == [0, -1, -1]).all()
    anchors, gaps = lab.anchors_and_gaps(lower, names)
    assert [a[0] for a in anchors] == [0, 2, 3] and gaps == [(0, 2, 10, 15), (2, 3, 20, 22)]


def test_fit_temperature_recovers_the_generating_scale():
    rng = np.random.default_rng(0)
    dist2 = rng.uniform(0, 1, size=(4000, 3))
    tau = 0.2
    logp = lab.log_identity_posterior(dist2, tau)
    labels = np.array([rng.choice(3, p=np.exp(row)) for row in logp])
    assert abs(np.log10(lab.fit_temperature(dist2, labels)) - np.log10(tau)) < 0.1


def test_gap_posterior_follows_flanks_without_evidence():
    flat = [np.zeros((3, 3))] * 4
    post = lab.gap_posterior([0, 1, 2], [0, 1, 2], [0.05] * 5, flat, rate=0.1)
    assert post.shape == (4, 6) and np.allclose(post.sum(axis=1), 1)
    assert (post[:, lab.perm_index([0, 1, 2])] > 0.97).all()
    slow = lab.gap_posterior([0, 1, 2], [0, 1, 2], [5.0] * 5, flat, rate=0.1)
    assert slow[2, lab.perm_index([0, 1, 2])] < post[2, lab.perm_index([0, 1, 2])]


def test_gap_posterior_places_the_switch_where_appearance_changes():
    def evidence(names):
        ll = np.full((3, 3), -5.0)
        ll[np.arange(3), names] = 0.0
        return ll

    obs = [evidence([0, 1, 2])] * 3 + [evidence([0, 2, 1])] * 3
    post = lab.gap_posterior([0, 1, 2], [0, 2, 1], [0.05] * 7, obs, rate=0.5, alpha=1.0)
    names = lab.node_names(lab.slot_marginals(post))
    assert (names[:3] == [0, 1, 2]).all() and (names[3:] == [0, 2, 1]).all()


def test_fill_between_nodes_keeps_only_shared_names():
    out = np.full((10, 3), -1, np.int8)
    nodes = [1, 4, 8]
    names = [np.array([0, 1, 2]), np.array([0, -1, 2]), np.array([0, 2, 1])]
    lab.fill_between_nodes(out, nodes, names)
    assert (out[2] == [0, -1, 2]).all() and (out[4] == [0, -1, 2]).all()
    assert (out[6] == [0, -1, -1]).all() and (out[8] == [0, 2, 1]).all()
    assert (out[9] == -1).all()


def test_identifier_protocol_adapter():
    slots = np.array([[0, 1, 2], [2, -1, 0]])
    ident = lab.PrecomputedIdentifier(slots)
    assert isinstance(ident, Identifier)
    out = ident.identify(None, [0, 1])
    assert out[1] == {0: Identity.MALE_2, 1: Identity.UNKNOWN, 2: Identity.FEMALE}


# --------------------------------------------------------------------------- Registration 4


def test_dorsal_up_flips_only_leftward_crops():
    crops = np.zeros((3, lab.OUT_PX, lab.OUT_PX, 3), np.uint8)
    crops[:, 0, :, :] = 255  # top row bright
    out = lab.dorsal_up(crops, np.array([0.3, np.pi - 0.2, -2.0]))
    assert np.array_equal(out[0], crops[0])
    assert out[1, -1].all() and not out[1, 0].any()
    assert out[2, -1].all() and not out[2, 0].any()
    assert np.array_equal(crops[1, 0], np.full((lab.OUT_PX, 3), 255))  # input untouched


def test_clean_mask_requires_all_three_conditions():
    occluded = np.array([0.0, 0.0, 0.1, 0.0])
    overlap = np.array([False, True, False, False])
    rows = np.array([3, 3, 3, 2])
    assert lab.clean_mask(occluded, overlap, rows).tolist() == [True, False, False, False]


def test_kmeans_and_silhouette_separate_blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = np.concatenate([c + rng.normal(scale=0.3, size=(50, 2)) for c in centres])
    truth = np.repeat(np.arange(3), 50)
    labels, inertia = lab.kmeans(x, k=3, seed=1)
    assert len({tuple(np.unique(labels[truth == k])) for k in range(3)}) == 3
    assert lab.silhouette(x, labels) > 0.9
    assert lab.silhouette(x, rng.integers(0, 3, size=len(x))) < 0.2
    assert lab.silhouette(x, np.zeros(len(x), int)) == -1.0
    assert np.isfinite(inertia)


def test_pca2_returns_two_components_with_explained_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 16)) * np.r_[5.0, 2.0, np.ones(14) * 0.1]
    xy, explained = lab.pca2(x)
    assert xy.shape == (40, 2) and explained.shape == (2,)
    assert explained[0] > explained[1] > 0.0 and explained.sum() > 0.9


def test_silhouette_monitor_keeps_the_best_and_stops_on_patience():
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, size=(60, lab.OUT_PX, lab.OUT_PX, 3), dtype=np.uint8)
    interval = np.repeat(np.arange(2), 30)
    slot = np.tile(np.repeat(np.arange(3), 10), 2)
    monitor = lab.SilhouetteMonitor(
        np.arange(0, 60, 2),
        np.array([[0, 1], [2, 3]]),
        patience=2,
        target=2.0,
        target_patience=1,
        seed=0,
    )
    model, stats, _log = lab.train_contrastive(
        crops,
        interval * 3 + slot,
        interval,
        np.ones(60, bool),
        steps=40,
        pairs=4,
        log_every=5,
        monitor=monitor,
        device="cpu",
    )
    steps = [e["step"] for e in monitor.evaluations]
    assert steps[0] == 0 and all(b - a == 5 for a, b in pairwise(steps))
    assert monitor.stopped in ("patience", None)
    assert len(monitor.snapshots) == len(steps) and monitor.snapshots[0].shape == (30, 16)
    assert np.isfinite(monitor.evaluations[0]["cross_camera_median"])
    best_score, _best_step, state = monitor.best
    assert best_score == max(e["silhouette"] for e in monitor.evaluations)
    for key, value in model.state_dict().items():
        assert np.array_equal(value.cpu().numpy(), state[key].cpu().numpy())
    assert lab.embed(model, crops, stats).shape == (60, 16)


@pytest.mark.parametrize("out", [64, 128])
def test_crop_size_scales_geometry_and_masks(out):
    image, points = synthetic_fish(30)
    centre, angle, length = lab.crop_pose(points)
    crop = lab.extract_crop(image, centre, angle, out)
    assert crop.shape == (out, out, 3)
    assert lab.train_px(out) == out * 7 // 8
    # The mouth lands on the right at the same fraction of the crop at any size.
    mouth = lab.to_crop(points[[0]], centre, angle, out)[0]
    assert np.allclose(mouth, [out / 2 + 50 * out / 160, out / 2], atol=1e-6)
    band = lab.band_mask(lab.to_crop(points, centre, angle, out), length, out)
    assert band.shape == (out, out) and band.any()
    masked, occluded = lab.mask_crop(crop, points[None], 0, centre, angle, np.zeros(3, np.uint8))
    assert masked.shape == crop.shape and occluded == 0.0
    assert lab.color_features(crop, length).shape == (6,)


def test_resnet_encoder_embeds_eight_dimensions():
    rng = np.random.default_rng(1)
    crops = rng.integers(0, 255, size=(12, lab.OUT_PX, lab.OUT_PX, 3), dtype=np.uint8)
    model = lab.build_encoder(kind="resnet18_8d")
    model.eval()
    assert model.out_dim == 8 and model.fc.bias is None
    z = lab.embed(model, crops, lab.channel_stats(crops))
    assert z.shape == (12, 8) and np.isfinite(z).all()
    with pytest.raises(ValueError):
        lab.build_encoder(kind="vgg")


def test_channel_stats_chunked_matches_direct():
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, size=(37, 8, 8, 3), dtype=np.uint8)
    x = crops.reshape(-1, 3).astype(np.float64) / 255
    mean, std = lab.channel_stats(crops, chunk=5)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    np.testing.assert_allclose(mean, x.mean(0), rtol=0, atol=1e-6)
    np.testing.assert_allclose(std, x.std(0), rtol=0, atol=1e-6)


def test_embed_is_independent_of_batch_size():
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    crops = rng.integers(0, 256, size=(23, 64, 64, 3), dtype=np.uint8)
    model = lab.build_encoder().eval()
    stats = lab.channel_stats(crops)
    full = lab.embed(model, crops, stats)
    small = lab.embed(model, crops, stats, batch=5)
    assert full.shape == (23, 16)
    np.testing.assert_allclose(small, full, rtol=0, atol=1e-5)
