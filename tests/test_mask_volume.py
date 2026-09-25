"""Geometry and support semantics, without real-video or identity assumptions."""

import importlib.util
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from medaka3d.calib import Rig
from medaka3d.stages import Associator
from medaka3d.types import VIEWS, Detection2D

_PATH = Path(__file__).resolve().parents[1] / "labs/20260908_mask_volume/volume.py"
_SPEC = importlib.util.spec_from_file_location("mask_volume_test_module", _PATH)
_VOLUME = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VOLUME
_SPEC.loader.exec_module(_VOLUME)
MaskVolume = _VOLUME.MaskVolume
MaskVolumeAssociator = _VOLUME.MaskVolumeAssociator
VolumeConfig = _VOLUME.VolumeConfig


@dataclass
class PlaneCamera:
    """Analytic test projection, independent of the refractive solver."""

    view: str
    axes: tuple[int, int]
    offset: tuple[float, float] = (0.0, 0.0)

    def project(self, xyz):
        return xyz[..., self.axes] + self.offset


@pytest.fixture
def plane_rig():
    return Rig(cameras=tuple(PlaneCamera(view, axes) for view, axes in
                             zip(VIEWS, [(0, 1), (0, 2), (1, 2)], strict=True)),
               manifest=Path("synthetic-not-a-real-file"))


def small_config():
    return VolumeConfig(bounds_mm=((0, 4), (0, 4), (0, 4)), spacing_mm=1)


def ones(shape=(8, 8)):
    return {view: np.ones(shape, np.float32) for view in VIEWS}


def test_grid_is_cell_centered_and_volume_does_not_exceed_bounds(plane_rig):
    volume = MaskVolume(plane_rig, VolumeConfig(bounds_mm=((0, 8), (2, 6), (-6, 0))))
    assert volume.grid_shape == (4, 2, 3)
    np.testing.assert_array_equal(volume.xyz.min(axis=0), [1, 3, -5])
    np.testing.assert_array_equal(volume.xyz.max(axis=0), [7, 5, -1])
    np.testing.assert_array_equal(volume.xyz[:3], [[1, 3, -5], [1, 3, -3], [1, 3, -1]])


@pytest.mark.parametrize("kwargs", [
    {"spacing_mm": 0}, {"spacing_mm": float("nan")},
    {"bounds_mm": ((0, 3), (0, 4), (0, 4))},
    {"bounds_mm": ((0, 0), (0, 4), (0, 4))},
    {"bounds_mm": ((0, float("inf")), (0, 4), (0, 4))},
    {"support_threshold": 0}, {"support_threshold": 1.1},
    {"sampling_dtype": "float16"},
])
def test_malformed_grid_and_support_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        VolumeConfig(**kwargs)


def test_bilinear_sample_matches_analytic_linear_image(plane_rig):
    cameras = tuple(replace(cam, offset=(0.17, 0.23)) for cam in plane_rig.cameras)
    volume = MaskVolume(replace(plane_rig, cameras=cameras), small_config())
    yy, xx = np.mgrid[:8, :8]
    masks = {view: (0.1 + 0.03 * xx + 0.05 * yy).astype(np.float32) for view in VIEWS}
    result = volume.evaluate(masks)
    for index, cam in enumerate(cameras):
        uv = cam.project(volume.xyz)
        expected = 0.1 + 0.03 * uv[:, 0] + 0.05 * uv[:, 1]
        np.testing.assert_allclose(result.sampled[index], expected, atol=1e-7)
    assert (result.known_count == 3).all()


def test_missing_view_is_unknown_but_present_empty_mask_is_background(plane_rig):
    volume = MaskVolume(plane_rig, small_config())
    masks = ones()
    masks[VIEWS[2]] = None
    missing = volume.evaluate(masks)
    assert (missing.known_count == 2).all()
    assert (missing.support_count == 2).all()
    assert np.isnan(missing.sampled[2]).all()
    assert missing.two_of_three.all() and not missing.strict3.any()
    np.testing.assert_array_equal(missing.soft_score, 1)
    masks[VIEWS[2]] = np.zeros((8, 8), np.float32)
    background = volume.evaluate(masks)
    assert (background.known_count == 3).all()
    assert (background.support_count == 2).all()
    np.testing.assert_allclose(background.soft_score, 2 / 3)
    assert background.two_of_three.all() and not background.strict3.any()
    entirely_missing = volume.evaluate({})
    assert not entirely_missing.known.any()
    assert np.isnan(entirely_missing.soft_score).all()
    assert not entirely_missing.components["two_of_three"]


def test_confidence_weights_mean_and_zero_is_unknown(plane_rig):
    volume = MaskVolume(plane_rig, small_config())
    masks = ones()
    masks[VIEWS[1]][:] = 0
    result = volume.evaluate(masks, {VIEWS[0]: 1, VIEWS[1]: 0.25, VIEWS[2]: 0})
    np.testing.assert_allclose(result.soft_score, 0.8)
    assert (result.known_count == 2).all()
    assert (result.support_count == 1).all()
    assert np.isnan(result.sampled[2]).all()
    explicit_none = volume.evaluate(masks, {VIEWS[0]: None})
    assert not explicit_none.known[0].any()


def test_nan_pixels_are_interpolated_as_unknown_not_background(plane_rig):
    volume = MaskVolume(plane_rig, VolumeConfig(bounds_mm=((0, 1),) * 3, spacing_mm=1))
    mask = np.array([[1, np.nan], [1, np.nan]], dtype=np.float32)
    result = volume.evaluate({view: mask for view in VIEWS})
    np.testing.assert_array_equal(result.sampled, 1)
    np.testing.assert_array_equal(result.confidence, 0.5)
    assert result.strict3.all()


def test_outside_or_invalid_projection_is_unknown_instead_of_padding_background(plane_rig):
    cameras = list(plane_rig.cameras)
    cameras[1] = replace(cameras[1], offset=(100, 0))
    cameras[2] = replace(cameras[2], offset=(np.nan, 0))
    volume = MaskVolume(replace(plane_rig, cameras=tuple(cameras)), small_config())
    result = volume.evaluate(ones())
    assert (result.known_count == 1).all()
    assert (result.support_count == 1).all()
    assert np.isnan(result.sampled[1:]).all()
    assert not result.two_of_three.any()


def test_four_disconnected_candidates_are_retained_with_no_three_fish_constraint(plane_rig):
    config = VolumeConfig(bounds_mm=((-0.5, 8.5),) * 3, spacing_mm=1)
    volume = MaskVolume(plane_rig, config)
    mask = np.zeros((9, 9), np.float32)
    mask[[1, 3, 5, 7], [1, 3, 5, 7]] = 1
    result = volume.evaluate({view: mask for view in VIEWS})
    components = result.components["strict3"]
    assert len(components) == 4
    np.testing.assert_array_equal([item["centroid_xyz"] for item in components],
                                  [[1, 1, 1], [3, 3, 3], [5, 5, 5], [7, 7, 7]])
    assert all(item["volume_mm3"] == 1 for item in components)


def test_merged_masks_can_form_one_component_instead_of_forced_separation(plane_rig):
    config = VolumeConfig(bounds_mm=((-0.5, 4.5),) * 3, spacing_mm=1)
    volume = MaskVolume(plane_rig, config)
    # This common merged region contains two nominal bodies; masks alone have
    # no evidence for an internal split, and this stage must not invent one.
    mask = np.zeros((5, 5), np.float32)
    mask[1:4, 1:4] = 1
    result = volume.evaluate({view: mask for view in VIEWS})
    assert len(result.components["strict3"]) == 1
    assert result.components["strict3"][0]["voxel_count"] == 27


def test_three_true_points_can_generate_a_fourth_cross_view_ghost(plane_rig):
    config = VolumeConfig(bounds_mm=((-0.5, 4.5),) * 3, spacing_mm=1)
    volume = MaskVolume(plane_rig, config)
    true_points = np.array([[1, 1, 3], [1, 3, 1], [3, 1, 1]])
    masks = {}
    for cam in plane_rig.cameras:
        pixels = cam.project(true_points).astype(int)
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[pixels[:, 1], pixels[:, 0]] = 1
        masks[cam.view] = mask
    result = volume.evaluate(masks)
    centroids = {tuple(item["centroid_xyz"]) for item in result.components["strict3"]}
    assert centroids == {(1, 1, 3), (1, 3, 1), (3, 1, 1), (1, 1, 1)}
    # Even perfect silhouettes in all three views do not prove an object exists.


def test_component_centroid_is_support_weighted_and_volume_is_cell_count(plane_rig):
    config = VolumeConfig(bounds_mm=((-0.5, 1.5), (-0.5, 0.5), (-0.5, 0.5)), spacing_mm=1)
    volume = MaskVolume(plane_rig, config)
    masks = ones((2, 2))
    masks[VIEWS[0]][0, 0] = 0.5
    result = volume.evaluate(masks)
    component = result.components["strict3"][0]
    assert component["voxel_count"] == 2
    assert component["volume_mm3"] == 2
    np.testing.assert_allclose(component["centroid_xyz"], [6 / 11, 0, 0], atol=1e-7)
    np.testing.assert_array_equal(component["bounds_mm"], config.bounds_mm)


def sphere_masks(rig, center, radius, shape=(320, 320)):
    """Render a sphere silhouette from ray/sphere intersection, not projection."""
    yy, xx = np.indices(shape)
    pixels = np.stack([xx, yy], axis=-1)
    masks = {}
    for cam in rig.cameras:
        origins, directions = cam.rays(pixels)
        offset = center - origins
        along = np.sum(offset * directions, axis=-1)
        distance_squared = np.sum(offset * offset, axis=-1) - along ** 2
        masks[cam.view] = ((along >= 0) & (distance_squared <= radius ** 2)).astype(np.float32)
    return masks


@pytest.fixture
def refractive_case(rig):
    cameras = tuple(replace(cam, fx=600, fy=600, cx=160, cy=160, index_air=1.0003,
                            index_glass=1.52, index_water=1.3325,
                            layer_count=1 if cam.view == VIEWS[0] else 2,
                            distortion=(-0.1, 0.04, 0.003, -0.002, -0.01))
                    for cam in rig.cameras)
    rig = replace(rig, cameras=cameras)
    center = np.array([90.0, 100.0, 80.0])
    config = VolumeConfig(bounds_mm=((68, 112), (78, 122), (58, 102)), spacing_mm=2)
    return rig, center, config, sphere_masks(rig, center, 10)


def test_refractive_sphere_support_contains_interior_and_rejects_distant_points(refractive_case):
    rig, center, config, masks = refractive_case
    volume = MaskVolume(rig, config)
    result = volume.evaluate(masks)
    distance = np.linalg.norm(volume.xyz - center, axis=1)
    assert result.strict3[distance < 8].all()
    assert not result.strict3[distance > 18].any()
    assert len(result.components["strict3"]) == 1
    np.testing.assert_allclose(result.components["strict3"][0]["centroid_xyz"], center, atol=0.8)
    # Three silhouettes contain a visual hull, not the exact spherical volume.
    assert result.components["strict3"][0]["volume_mm3"] > 4 / 3 * np.pi * 10 ** 3


@pytest.mark.parametrize("sampling_dtype", ["float32", "float64"])
def test_cuda_cpu_parity_for_refractive_masks_and_missing_view(refractive_case, sampling_dtype):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; CPU semantics are tested separately")
    rig, _, config, masks = refractive_case
    config = replace(config, sampling_dtype=sampling_dtype)
    cpu = MaskVolume(rig, config, device="cpu")
    gpu = MaskVolume(rig, config, device="cuda")
    for missing in (False, True):
        inputs = masks | ({VIEWS[2]: None} if missing else {})
        expected = cpu.evaluate(inputs)
        actual = gpu.evaluate(inputs)
        tolerance = 4e-5 if sampling_dtype == "float32" else 1e-11
        np.testing.assert_allclose(actual.sampled, expected.sampled, atol=tolerance, rtol=0,
                                   equal_nan=True)
        assert actual.sampled.dtype == expected.sampled.dtype == np.dtype(sampling_dtype)
        assert actual.soft_score.dtype == expected.soft_score.dtype == np.dtype(sampling_dtype)
        np.testing.assert_array_equal(actual.known_count, expected.known_count)
        np.testing.assert_array_equal(actual.support_count, expected.support_count)
        np.testing.assert_array_equal(actual.strict3, expected.strict3)
        np.testing.assert_array_equal(actual.two_of_three, expected.two_of_three)


def test_double_precision_preserves_near_threshold_edges_at_native_resolution(plane_rig):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; CPU semantics are tested separately")
    cameras = tuple(replace(cam, axes=(0, 1), offset=(599.99999998, 700.25))
                    for cam in plane_rig.cameras)
    rig = replace(plane_rig, cameras=cameras)
    config = VolumeConfig(bounds_mm=((0, 2), (0, 1), (0, 1)), spacing_mm=1)
    assert asdict(config)["sampling_dtype"] == "float64"
    mask = np.zeros((1200, 1200))
    mask[:, 601] = 1
    # Both grid centres project about 2e-8 pixels from an exactly half-covered
    # edge, but on opposite sides. Neither should be rounded onto the threshold.
    masks = {view: mask for view in VIEWS}
    expected = MaskVolume(rig, config, device="cpu").evaluate(masks)
    actual = MaskVolume(rig, config, device="cuda").evaluate(masks)
    np.testing.assert_array_equal(expected.support_count, [0, 3])
    np.testing.assert_array_equal(actual.support_count, expected.support_count)
    np.testing.assert_allclose(actual.sampled, expected.sampled, atol=3e-13, rtol=0)


def test_associator_protocol_unions_instances_and_returns_frame_local_centroids(plane_rig):
    config = VolumeConfig(bounds_mm=((-0.5, 4.5),) * 3, spacing_mm=1)
    adapter = MaskVolumeAssociator(MaskVolume(plane_rig, config))
    assert isinstance(adapter, Associator)
    detections = {}
    for view in VIEWS:
        masks = []
        for pixel in (1, 3):
            mask = np.zeros((5, 5), dtype=bool)
            mask[pixel, pixel] = True
            masks.append(Detection2D(view, frame=17, mask=mask, score=0.1))
        detections[view] = masks
    instances = adapter.associate(detections, 17)
    assert len(instances) == 2
    np.testing.assert_array_equal([instance.xyz for instance in instances], [[1, 1, 1], [3, 3, 3]])
    assert [instance.slot for instance in instances] == [0, 1]
    assert all(instance.frame == 17 and not instance.sources and np.isnan(instance.residual)
               for instance in instances)
    assert adapter.associate({}, 18) == []
    with pytest.raises(ValueError, match="view and frame"):
        adapter.associate(detections, 18)


def test_associator_preserves_unknown_mask_pixels(plane_rig):
    adapter = MaskVolumeAssociator(MaskVolume(plane_rig, small_config()))
    detections = {view: [Detection2D(view, frame=4, mask=np.ones((8, 8)))] for view in VIEWS}
    detections[VIEWS[2]] = [Detection2D(VIEWS[2], frame=4, mask=np.full((8, 8), np.nan))]
    assert adapter.associate(detections, 4) == []


def test_invalid_mask_view_and_confidence_fail_loudly(plane_rig):
    volume = MaskVolume(plane_rig, small_config())
    with pytest.raises(ValueError, match="unrecognised"):
        volume.evaluate({"wrong_camera": np.ones((8, 8))})
    with pytest.raises(ValueError, match="2D"):
        volume.evaluate({VIEWS[0]: np.ones((8, 8, 3))})
    with pytest.raises(ValueError, match="mask values"):
        volume.evaluate({VIEWS[0]: np.full((8, 8), 255)})
    with pytest.raises(ValueError, match="same shape"):
        volume.evaluate(ones(), {VIEWS[0]: np.ones((9, 9))})
    with pytest.raises(ValueError, match="confidence values"):
        volume.evaluate(ones(), {VIEWS[0]: -1})
    with pytest.raises(ValueError, match="device must"):
        MaskVolume(plane_rig, small_config(), device="typo")
