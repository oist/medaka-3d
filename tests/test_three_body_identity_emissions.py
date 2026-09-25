"""Fixed geometry color emissions preserve permutation meaning and absent evidence."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from medaka3d.types import VIEWS


def modules():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    names = ("pixel_template_assignment", "appearance_ranking", "pixel_appearance_loss",
             "renderer", "identity_emissions")
    previous, loaded = {name: sys.modules.get(name) for name in names}, {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(f"identity_emission_test_{name}",
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
E, R, F = (M[name] for name in ("identity_emissions", "renderer", "pixel_template_assignment"))


class OrthographicCamera:
    def __init__(self, index, invalid=False):
        self.index, self.invalid = index, invalid

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
        if self.invalid:
            directions[:] = 0
        return origins, directions


class SmallRig:
    views = VIEWS
    world_units = "mm"

    def __init__(self, invalid_views=()):
        self.invalid_views = invalid_views

    def camera(self, view):
        return OrthographicCamera(VIEWS.index(view), view in self.invalid_views)


@pytest.fixture(autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def scene():
    rig = SmallRig()
    centers = np.array([[22, 22, 25], [49, 49, 45], [75, 22, 70]], float)
    # Unequal lengths make a mistaken exchange of identity-attached widths visible.
    scale = np.array([0.7, 1, 1.4])[:, None, None]
    pose = centers[:, None] + scale * np.array([[3, 0, 0], [0, 0, 0], [-6, 1, -1]])
    colors = np.array([[30, 30, 180], [30, 180, 30], [180, 30, 30]], np.uint8)
    images, masks, points = {}, {}, {}
    for view in VIEWS:
        points[view] = rig.camera(view).project(pose)
        mask = np.zeros((3, 96, 96), np.uint8)
        image = np.full((96, 96, 3), 20, np.uint8)
        for fish in range(3):
            cv2.polylines(mask[fish], [points[view][fish].astype(np.int32)], False, 1, 9)
            image[mask[fish].astype(bool)] = colors[fish]
        masks[view], images[view] = mask.astype(bool), image
    frame = SimpleNamespace(frame_idx=78799, detections=points, images_bgr=images, masks=masks)
    templates = np.broadcast_to(F.pixel_features(colors), (3, 3, 6)).copy()
    return rig, pose, frame, templates


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def actual_six_render_oracle(rig, frame, pose, templates, config):
    pixels = M["appearance_ranking"].sample_appearance_pixels(
        frame, np.array([v in config.active_views for v in VIEWS]), config.maximum_samples
    )
    renderer = R.RefractiveFishRenderer(
        rig, pixels.pixels, R.RendererConfig(width_ratio=config.width_ratio),
        device="cpu", dtype=torch.float64,
    )
    # Render six genuinely reordered 3D skeletons. Each body's length and derived
    # width move together; there is no identity-specific width replacement.
    with torch.no_grad():
        rendered = renderer.render(tensor(pose[E.PERMUTATIONS]))
        scored = M["pixel_appearance_loss"].pixel_appearance_loss(
            rendered.visible_alpha, tensor(pixels.features), tensor(templates),
            tensor(pixels.weights) * rendered.ray_valid, torch.as_tensor(pixels.enabled),
        )
    return scored, rendered, pixels


def test_one_render_channel_permutations_equal_six_real_renders_with_unequal_bodies(monkeypatch):
    rig, pose, frame, templates = scene()
    config = E.IdentityEmissionConfig(maximum_samples=128, active_views=VIEWS)
    calls = []

    class CountingRenderer(R.RefractiveFishRenderer):
        def render(self, pose_xyz, *args, **kwargs):
            calls.append(tuple(pose_xyz.shape))
            return super().render(pose_xyz, *args, **kwargs)

    monkeypatch.setattr(E, "RefractiveFishRenderer", CountingRenderer)
    actual = E.score_identity_permutations(rig, frame, pose, templates, np.ones(3, bool),
                                           config=config)
    assert calls == [(3, 3, 3)]
    expected, rendered, pixels = actual_six_render_oracle(rig, frame, pose, templates, config)
    np.testing.assert_allclose(actual.costs, expected.loss.numpy(), rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual.per_view_costs, expected.per_view_loss.numpy(),
                               rtol=0, atol=2e-14)
    expected_mass = (rendered.visible_alpha[0] * tensor(pixels.weights)[..., None]).sum(1)
    np.testing.assert_allclose(actual.visible_mass, expected_mass.numpy(), rtol=0, atol=2e-14)
    assert actual.rendering_performed and actual.enabled_views.all()
    assert np.argmin(actual.costs) == 0


def test_non_self_inverse_cycle_means_identity_to_raw_slot_and_mass_stays_raw():
    rig, pose, frame, templates = scene()
    config = E.IdentityEmissionConfig(maximum_samples=128)
    original = E.score_identity_permutations(rig, frame, pose, templates, np.ones(3, bool),
                                             config=config)
    order = np.array([1, 2, 0])
    result = E.score_identity_permutations(rig, frame, pose[order], templates, np.ones(3, bool),
                                           config=config)
    np.testing.assert_array_equal(E.PERMUTATIONS[np.argmin(result.costs)], [2, 0, 1])
    np.testing.assert_allclose(result.visible_mass, original.visible_mass[:, order],
                               rtol=0, atol=2e-14)
    for state, permutation in enumerate(E.PERMUTATIONS):
        original_state = np.flatnonzero((order[permutation] == E.PERMUTATIONS).all(axis=1))[0]
        assert result.costs[state] == pytest.approx(original.costs[original_state], abs=2e-14)


def test_coincident_depth_ties_do_not_create_a_preferred_identity():
    rig, pose, frame, templates = scene()
    pose[:] = pose[1]
    config = E.IdentityEmissionConfig(maximum_samples=64, active_views=VIEWS)
    actual = E.score_identity_permutations(rig, frame, pose, templates, np.ones(3, bool),
                                           config=config)
    expected, _, _ = actual_six_render_oracle(rig, frame, pose, templates, config)
    np.testing.assert_allclose(actual.costs, actual.costs[0], rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual.costs, expected.loss.numpy(), rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual.visible_mass,
                               np.broadcast_to(actual.visible_mass[:, :1], (3, 3)),
                               rtol=0, atol=2e-14)


def test_identical_color_templates_make_every_assignment_tied():
    rig, pose, frame, templates = scene()
    templates[:] = templates[:, :1]
    result = E.score_identity_permutations(rig, frame, pose, templates, np.ones(3, bool))
    np.testing.assert_allclose(result.costs, result.costs[0], rtol=0, atol=2e-14)
    assert (result.visible_mass[0] > 0).all()


def test_all_shapes_off_foreground_have_zero_visible_mass_and_equal_background_cost():
    rig, pose, frame, templates = scene()
    pose[:, :, :2] += 5000
    result = E.score_identity_permutations(rig, frame, pose, templates, np.ones(3, bool))
    np.testing.assert_array_equal(result.visible_mass, np.zeros((3, 3)))
    np.testing.assert_allclose(result.costs, 2, rtol=0, atol=1e-14)
    assert result.enabled_views[0] and result.valid_pixels[0] > 0
    # Pixels can be available while the geometry explains none of them.
    assert result.rendering_performed


@pytest.mark.parametrize("missing", ["disabled", "missing_mask", "empty_mask", "nan_template"])
def test_unavailable_top_view_is_neutral_without_rendering(missing):
    rig, pose, frame, templates = scene()
    enabled = np.ones(3, bool)
    if missing == "disabled":
        enabled[0] = False
    elif missing == "missing_mask":
        frame.masks["top_cam"] = None
    elif missing == "empty_mask":
        frame.masks["top_cam"] = np.zeros((0, 96, 96), bool)
    else:
        # One missing label disables the view; remaining labels cannot infer it.
        templates[0, 1] = np.nan
    result = E.score_identity_permutations(rig, frame, pose, templates, enabled)
    np.testing.assert_array_equal(result.costs, np.zeros(6))
    assert not result.enabled_views.any() and not result.rendering_performed
    assert not result.visible_mass.any()


def test_missing_one_view_leaves_other_enrolled_views_and_raw_mass_axes_intact():
    rig, pose, frame, templates = scene()
    frame.masks["side1_cam"] = None
    result = E.score_identity_permutations(
        rig, frame, pose, templates, np.ones(3, bool),
        config=E.IdentityEmissionConfig(maximum_samples=64, active_views=VIEWS),
    )
    np.testing.assert_array_equal(result.enabled_views, [True, False, True])
    assert not result.per_view_costs[:, 1].any() and not result.visible_mass[1].any()
    assert result.valid_pixels[[0, 2]].min() > 0


def test_invalid_rays_are_excluded_and_do_not_become_color_evidence():
    _, pose, frame, templates = scene()
    result = E.score_identity_permutations(
        SmallRig(invalid_views=VIEWS), frame, pose, templates, np.ones(3, bool),
        config=E.IdentityEmissionConfig(maximum_samples=64, active_views=VIEWS),
    )
    assert result.rendering_performed and result.sampled_pixels.min() > 0
    assert not result.valid_pixels.any() and not result.enabled_views.any()
    assert not result.visible_mass.any() and not result.costs.any()


def test_original_inputs_are_unchanged_and_output_arrays_do_not_alias_them():
    rig, pose, frame, templates = scene()
    enabled = np.ones(3, bool)
    inputs = [pose, templates, enabled, *frame.images_bgr.values(), *frame.masks.values(),
              *frame.detections.values(), E.PERMUTATIONS]
    before = [(value.dtype, value.shape, value.tobytes()) for value in inputs]
    result = E.score_identity_permutations(rig, frame, pose, templates, enabled)
    result.costs[:] = -10
    result.visible_mass[:] = -10
    result.enabled_views[:] = False
    assert before == [(value.dtype, value.shape, value.tobytes()) for value in inputs]


def test_sampler_uses_all_mask_union_and_native_pixels_without_pose_or_sleap(monkeypatch):
    rig, pose, frame, templates = scene()
    extra = np.zeros((2, 96, 96), bool)
    extra[0, 65:86, 7:26] = True
    extra[1] = frame.masks["top_cam"][:2].any(0)  # Merged fourth/fifth candidate retained.
    frame.masks["top_cam"] = np.concatenate((frame.masks["top_cam"], extra))
    union = np.any(frame.masks["top_cam"], axis=0)
    core = cv2.erode(union.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    expected_yx = np.argwhere(core)
    captured = []
    original_sampler = E.sample_appearance_pixels

    def capture(*args):
        value = original_sampler(*args)
        captured.append(value)
        return value

    monkeypatch.setattr(E, "sample_appearance_pixels", capture)
    config = E.IdentityEmissionConfig(maximum_samples=10000)
    E.score_identity_permutations(rig, frame, pose, templates, np.ones(3, bool), config=config)
    frame.detections = {view: np.full((9, 3, 2), np.nan) for view in VIEWS}
    frame.masks["top_cam"] = frame.masks["top_cam"][[4, 2, 1, 3, 0]]
    E.score_identity_permutations(rig, frame, pose + 1000, templates, np.ones(3, bool),
                                  config=config)
    for value in captured:
        np.testing.assert_array_equal(value.pixels["top_cam"], expected_yx[:, ::-1])
        expected = F.pixel_features(frame.images_bgr["top_cam"][core])
        np.testing.assert_array_equal(value.features[0], expected)
        assert value.counts[0] == value.available_core_counts[0] == len(expected_yx)
        assert ((value.pixels["top_cam"][:, 1] >= 67)
                & (value.pixels["top_cam"][:, 0] < 25)).any()
    # Also exercise the native-frame-only random cap with those same inputs.
    a = original_sampler(frame, np.array([True, False, False]), 32)
    frame.detections.clear()
    b = original_sampler(frame, np.array([True, False, False]), 32)
    np.testing.assert_array_equal(a.pixels["top_cam"], b.pixels["top_cam"])
    np.testing.assert_array_equal(a.features, b.features)
    assert a.counts[0] == 32 < a.available_core_counts[0]


@pytest.mark.parametrize("problem", ["nan_pose", "reordered_views", "bad_enabled"])
def test_invalid_geometry_and_axis_contracts_fail_loudly(problem):
    rig, pose, frame, templates = scene()
    enabled = np.ones(3, bool)
    if problem == "nan_pose":
        pose[0, 0, 0] = np.nan
    elif problem == "reordered_views":
        rig.views = VIEWS[::-1]
    else:
        enabled = np.ones(3, int)
    with pytest.raises(ValueError):
        E.score_identity_permutations(rig, frame, pose, templates, enabled)
