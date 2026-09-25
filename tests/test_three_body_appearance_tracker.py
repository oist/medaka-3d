"""Color reranking preserves causal enrollment, neutral cases and sticky review."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from medaka3d.types import VIEWS


def modules():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    names = (
        "prepare_sequence", "observations", "node_geometry", "renderer", "fast_union",
        "tracker", "graph_union", "graph_fit", "accelerated_tracker", "node_recovery",
        "persistent_node_tracker", "appearance_diagnostic", "online_appearance",
        "pixel_template_assignment", "appearance_ranking", "pixel_appearance_loss",
        "appearance_tracker",
    )
    previous, loaded = {name: sys.modules.get(name) for name in names}, {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"appearance_tracker_test_{name}",
                                                         lab / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return loaded


M = modules()
A, P, T = (M[name] for name in ("appearance_tracker", "persistent_node_tracker", "tracker"))
R, OBS = M["appearance_ranking"], M["observations"]


class OrthographicCamera:
    def __init__(self, index):
        self.index = index

    def project(self, xyz):
        return np.asarray(xyz)[..., ((0, 1), (0, 2), (1, 2))[self.index]].copy()

    def rays(self, uv):
        uv = np.asarray(uv)
        origins = np.zeros((*uv.shape[:-1], 3))
        directions = np.zeros_like(origins)
        if self.index == 0:
            origins[..., :2], origins[..., 2] = uv, 200
            directions[..., 2] = -1
        elif self.index == 1:
            origins[..., 0], origins[..., 1], origins[..., 2] = uv[..., 0], -100, uv[..., 1]
            directions[..., 1] = 1
        else:
            origins[..., 0], origins[..., 1:] = 200, uv
            directions[..., 0] = -1
        return origins, directions


class SmallRig:
    views = VIEWS
    world_units = "mm"

    def camera(self, view):
        return OrthographicCamera(VIEWS.index(view))


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def scene(frame_idx=80):
    rig = SmallRig()
    centers = np.array([[25, 25, 25], [50, 50, 45], [75, 25, 70]], float)
    pose = centers[:, None] + np.array([[3, 0, 0], [0, 0, 0], [-6, 1, -1]])
    images, masks, points = {}, {}, {}
    colors = [[30, 30, 180], [30, 180, 30], [180, 30, 30]]
    for view in VIEWS:
        points[view] = rig.camera(view).project(pose)
        mask = np.zeros((3, 96, 96), np.uint8)
        image = np.full((96, 96, 3), 20, np.uint8)
        for fish in range(3):
            uv = points[view][fish].astype(np.int32)
            cv2.polylines(mask[fish], [uv], False, 1, 9)
            image[mask[fish].astype(bool)] = colors[fish]
        masks[view], images[view] = mask.astype(bool), image
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).astype(np.int32).copy()
    frame = OBS.FrameObservation(frame_idx, points, images, masks, {})
    return rig, pose, frame, sources


def config(**kwargs):
    return replace(T.TrackerConfig(device="cpu", mode="nodes", beam_size=2,
                                   max_hypotheses=4, opt_steps=1), **kwargs)


def appearance(rig, pose, weight=1, prefix=2, minimum=1, **kwargs):
    return A.AppearanceThreeBodyTracker(
        rig, pose, config(), appearance_config=A.AppearanceRankingConfig(weight=weight),
        enrollment_config=M["online_appearance"].OnlineAppearanceConfig(prefix, minimum), **kwargs
    )


def all21_equal(actual, expected):
    assert len(expected) == 21
    assert set(actual) == set(expected) | {
        "geometric_cost", "appearance_cost", "appearance_enabled", "appearance_pixels"
    }
    for key, value in expected.items():
        assert value.shape == actual[key].shape and value.dtype == actual[key].dtype, key
        assert value.tobytes() == actual[key].tobytes(), key


def test_weight_zero_is_byte_identical_after_templates_become_available():
    rig, pose, frame, _ = scene()
    frames = [replace(frame, frame_idx=80+i) for i in range(5)]
    expected = P.PersistentNodeThreeBodyTracker(rig, pose, config()).track(frames)
    tracker = appearance(rig, pose, weight=0)
    actual = tracker.track(frames)
    all21_equal(actual, expected)
    assert tracker.enrollment.frozen and tracker.enrollment.enabled_views.all()
    assert not actual["appearance_cost"].any() and not actual["appearance_enabled"].any()
    assert tracker.appearance_scored_frames == 0


def test_positive_weight_is_byte_identical_through_complete_enrollment_prefix():
    rig, pose, frame, _ = scene()
    frames = [replace(frame, frame_idx=80+i) for i in range(166)]
    expected = P.PersistentNodeThreeBodyTracker(rig, pose, config()).track(frames)
    # Actual default 166/5 registration, real optimizer and support tests; only
    # the synthetic image size and existing CPU fit budget keep this bounded.
    tracker = A.AppearanceThreeBodyTracker(rig, pose, config())
    actual = tracker.track(frames)
    all21_equal(actual, expected)
    assert tracker.enrollment.frozen and tracker.enrollment.enabled_views.all()
    assert tracker.enrollment.training_frames_seen == 166
    np.testing.assert_array_equal(tracker.enrollment.counts, np.full((3, 3), 166))
    assert not actual["appearance_cost"].any()


def test_first_166_frames_register_once_current_state_then_freeze_before_frame_167(monkeypatch):
    rig, pose, frame, sources = scene(900)
    tracker = A.AppearanceThreeBodyTracker(rig, pose, config(beam_size=1))
    observed = []
    original = tracker.enrollment.observe

    def enrollment(frame, best_pose, sources, supported, review):
        frozen_before = tracker.enrollment.frozen
        original(frame, best_pose, sources, supported, review)
        observed.append((frame.frame_idx, frozen_before, tracker.enrollment.training_frames_seen))

    monkeypatch.setattr(tracker.enrollment, "observe", enrollment)
    # Isolate loop timing from optimizer convergence; actual node support, image
    # enrollment, pixel sampler, renderer and mixture scorer remain active.
    def proposal(states, rays):
        return [T.Proposal(pose.copy(), pose.copy(), sources.copy(), 0, "predict")]

    def fit(proposals, states, rays, targets):
        return (np.stack([p.pose for p in proposals]), np.zeros(1), np.zeros((1, 5)),
                np.zeros((1, 3), bool))

    monkeypatch.setattr(tracker, "_proposals", proposal)
    monkeypatch.setattr(tracker, "_fit", fit)
    frames = [replace(frame, frame_idx=900+i) for i in range(168)]
    result = tracker.track(frames)
    assert [row[0] for row in observed] == list(range(900, 1068))
    assert not any(row[1] for row in observed[:166])
    assert all(row[1] for row in observed[166:])
    assert observed[-1][2] == 166
    np.testing.assert_array_equal(tracker.enrollment.snapshot()["frame_idx"], np.arange(900, 1066))
    np.testing.assert_array_equal(tracker.enrollment.counts, np.full((3, 3), 166))
    assert not result["appearance_enabled"][:166].any()
    assert result["appearance_enabled"][166:, 0].all()
    assert not result["appearance_enabled"][:, 1:].any()
    assert tracker.appearance_scored_frames == 2
    before = tracker.enrollment.templates
    tracker.enrollment.observe(SimpleNamespace(frame_idx=1068), None, None, None, None)
    np.testing.assert_array_equal(tracker.enrollment.templates, before)


def enrolled_tracker():
    rig, pose, frame, sources = scene()
    tracker = appearance(rig, pose, prefix=1)
    tracker.enrollment.observe(frame, pose, sources, np.ones((3, 3), bool), np.zeros(3, bool))
    assert tracker.enrollment.enabled_views.all()
    return tracker, pose, replace(frame, frame_idx=81)


def test_real_renderer_color_score_ranks_correct_template_permutation_lower():
    tracker, pose, frame = enrolled_tracker()
    hypotheses = np.stack([pose, pose[[1, 0, 2]], pose[[2, 1, 0]]])
    before = hypotheses.copy()
    cost, enabled, counts = tracker._appearance_scores(frame, hypotheses)
    assert np.isfinite(cost).all() and cost[0] < cost[1] and cost[0] < cost[2]
    np.testing.assert_array_equal(enabled, [True, False, False])
    assert 0 < counts[0] <= 512 and not counts[1:].any()
    np.testing.assert_array_equal(hypotheses, before)


def test_enrollment_uses_current_best_even_when_future_backtrace_selects_other_parent(monkeypatch):
    rig, pose, frame, sources = scene()
    tracker = appearance(rig, pose, prefix=1)
    tracker.config = replace(tracker.config, ambiguity_margin=0.1)
    permutation = [1, 0, 2]
    step = [0]

    def proposals(states, rays):
        if step[0] == 0:
            return [
                T.Proposal(pose.copy(), pose.copy(), sources.copy(), 0, "initial_best"),
                T.Proposal(pose[permutation].copy(), pose.copy(), sources[permutation].copy(),
                           0, "initial_second", tuple(permutation)),
            ]
        return [
            T.Proposal(states[0].pose.copy(), pose.copy(), sources.copy(), 0, "future_worse"),
            T.Proposal(states[1].pose.copy(), pose.copy(), sources[permutation].copy(),
                       1, "future_better", tuple(permutation)),
        ]

    def fit(proposed, states, rays, targets):
        costs = np.array([0.0, 1.0]) if step[0] == 0 else np.array([100.0, 0.0])
        step[0] += 1
        return (np.stack([item.pose for item in proposed]), costs, np.zeros((2, 5)),
                np.zeros((2, 3), bool))

    monkeypatch.setattr(tracker, "_proposals", proposals)
    monkeypatch.setattr(tracker, "_fit", fit)
    result = tracker.track([frame, replace(frame, frame_idx=81)])
    assert result["proposal_kind"].tolist() == ["initial_second", "future_better"]
    np.testing.assert_array_equal(result["pose_xyz"][0], pose[permutation])
    expected = np.array([
        M["appearance_diagnostic"].describe_core(frame.images_bgr[VIEWS[0]], mask)[0]
        for mask in frame.masks[VIEWS[0]]
    ])
    np.testing.assert_array_equal(tracker.enrollment.templates[0], expected)
    assert not np.array_equal(tracker.enrollment.templates[0], expected[permutation])
    np.testing.assert_array_equal(tracker.enrollment.counts, np.ones((3, 3), int))
    np.testing.assert_array_equal(tracker.enrollment.snapshot()["frame_idx"], [80])


@pytest.mark.parametrize(
    "reason", ["empty_mask", "missing_mask", "disabled_template", "invalid_ray"]
)
def test_empty_or_disabled_or_invalid_color_view_returns_neutral_cost(reason):
    tracker, pose, frame = enrolled_tracker()
    if reason == "empty_mask":
        frame.masks[VIEWS[0]] = np.zeros((0, 96, 96), bool)
    elif reason == "missing_mask":
        frame.masks[VIEWS[0]] = None
    elif reason == "disabled_template":
        tracker.enrollment = SimpleNamespace(frozen=True, enabled_views=np.zeros(3, bool))
    else:
        class BadCamera(OrthographicCamera):
            def rays(self, uv):
                origin, direction = super().rays(uv)
                return np.full_like(origin, np.nan), np.full_like(direction, np.nan)

        tracker.rig = SimpleNamespace(views=VIEWS, world_units="mm",
                                     camera=lambda v: BadCamera(VIEWS.index(v)))
    costs, enabled, counts = tracker._appearance_scores(frame, np.stack([pose, pose[[1, 0, 2]]]))
    np.testing.assert_array_equal(costs, [0, 0])
    assert not enabled.any() and not counts.any()


def test_reranking_preserves_fit_coordinates_and_never_clears_review(monkeypatch):
    tracker, pose, frame = enrolled_tracker()
    # Starting track with an externally primed accumulator is not a deployment
    # path. Use a local read-only stub only to isolate beam color-rank behavior.
    template = tracker.enrollment.templates
    tracker.enrollment = SimpleNamespace(frozen=True, enabled_views=np.ones(3, bool),
                                         templates=template, observe=lambda *args: None)
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).astype(np.int32).copy()
    permutation = [1, 0, 2]
    frozen_fits = []
    step = [0]

    def proposals(states, rays):
        if step[0] == 0:
            plan = np.full((3, 3, 3), -1, np.int32)
            return [T.Proposal(pose.copy(), pose.copy(), plan, 0, "predict")]
        return [
            T.Proposal(pose[permutation].copy(), pose.copy(), sources[permutation].copy(),
                       0, "swapped", tuple(permutation)),
            T.Proposal(pose.copy(), pose.copy(), sources.copy(), 0, "matching"),
        ]

    def fit(proposed, states, rays, targets):
        values = np.stack([item.pose for item in proposed])
        frozen_fits.append(values.copy())
        step[0] += 1
        cost = np.zeros(len(proposed))
        if len(cost) == 2:
            cost[1] = 0.001  # Geometry alone prefers the swapped candidate.
        return values, cost, np.zeros((len(proposed), 5)), np.zeros((len(proposed), 3), bool)

    monkeypatch.setattr(tracker, "_proposals", proposals)
    monkeypatch.setattr(tracker, "_fit", fit)
    tracker.config = replace(tracker.config, review_gap_frames=1, ambiguity_margin=0.001)
    frames = [replace(frame, frame_idx=81+i) for i in range(2)]
    result = tracker.track(frames)
    assert result["proposal_kind"].tolist() == ["predict", "matching"]
    np.testing.assert_array_equal(result["pose_xyz"][1], frozen_fits[1][1])
    assert result["geometric_cost"][1] == 0.001
    assert result["identity_review"].all()
    assert result["observed"][1].all() and np.isnan(result["accepted_xyz"]).all()
    np.testing.assert_allclose(result["costs"], result["geometric_cost"]+result["appearance_cost"])


def complex_sampler_frame(frame_idx=90):
    image = np.empty((80, 110, 3), np.uint8)
    y, x = np.indices(image.shape[:2])
    image[..., 0], image[..., 1], image[..., 2] = x, y, (x+y) % 255
    masks = np.zeros((5, 80, 110), bool)
    for i in range(5):
        masks[i, 8:38, 3+20*i:18+20*i] = True
    masks[0, 20:27, 10:37] = True  # One merged mask joins two neighboring regions.
    return SimpleNamespace(frame_idx=frame_idx, images_bgr={VIEWS[0]: image},
                           masks={VIEWS[0]: masks}, detections=object(), pose=object())


def test_sampler_uses_union_of_all_five_masks_including_merged_and_retains_native_features():
    frame = complex_sampler_frame()
    evidence = R.sample_appearance_pixels(frame, np.array([True, False, False]), 10000)
    _, core = M["pixel_template_assignment"].union_core(frame.masks[VIEWS[0]])
    _, first_three = M["pixel_template_assignment"].union_core(frame.masks[VIEWS[0]][:3])
    assert core.sum() > first_three.sum()
    expected = np.argwhere(core)[:, ::-1]
    np.testing.assert_array_equal(evidence.pixels[VIEWS[0]], expected)
    assert evidence.counts[0] == evidence.available_core_counts[0] == len(expected)
    expected_features = M["pixel_template_assignment"].pixel_features(
        frame.images_bgr[VIEWS[0]][core]
    )
    np.testing.assert_array_equal(evidence.features[0], expected_features)
    assert evidence.weights[0].sum() == len(expected)
    assert not evidence.weights[1:].any()


def test_sampler_is_order_duplicate_and_sleap_pose_independent_with_fixed_frame_seed():
    frame = complex_sampler_frame()
    enabled = np.array([True, False, False])
    first = R.sample_appearance_pixels(frame, enabled, 31)
    masks = frame.masks[VIEWS[0]]
    frame.masks[VIEWS[0]] = np.concatenate([masks[::-1], masks[:1]])
    frame.detections, frame.pose = {"fake": np.nan}, np.full((3, 3, 3), np.nan)
    second = R.sample_appearance_pixels(frame, enabled, 31)
    np.testing.assert_array_equal(first.pixels[VIEWS[0]], second.pixels[VIEWS[0]])
    np.testing.assert_array_equal(first.features, second.features)
    assert first.counts[0] == 31
    xy = first.pixels[VIEWS[0]].astype(int)
    actual = M["pixel_template_assignment"].pixel_features(
        frame.images_bgr[VIEWS[0]][xy[:, 1], xy[:, 0]]
    )
    np.testing.assert_array_equal(first.features[0], actual)
    frame.frame_idx += 1
    third = R.sample_appearance_pixels(frame, enabled, 31)
    assert not np.array_equal(first.pixels[VIEWS[0]], third.pixels[VIEWS[0]])


def test_accelerated_mro_overrides_only_geometric_fit():
    cls = A.AcceleratedAppearanceTracker
    assert cls.track is A.AppearanceThreeBodyTracker.track
    assert cls._fit is M["accelerated_tracker"].AcceleratedThreeBodyTracker._fit
    assert cls._proposals is P.PersistentNodeThreeBodyTracker._proposals
