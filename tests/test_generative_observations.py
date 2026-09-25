"""Known-pixel checks for the image terms, separate from fish-ID benchmarks."""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.optimize import least_squares

MODULE = Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking/observations.py"
SPEC = importlib.util.spec_from_file_location("_generative_observations_test", MODULE)
observations = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observations
SPEC.loader.exec_module(observations)
SceneEvidence = observations.SceneEvidence
EvidenceConfig = observations.EvidenceConfig


def fish_image(color=(240, 130, 30), center=(50, 50), shape=(100, 120)):
    background = np.full((*shape, 3), 180, np.uint8)
    current = background.copy()
    cv2.ellipse(current, center, (18, 5), 0, 0, 360, color, -1)
    skeleton = np.array([[-16, 0], [-5, 0], [16, 0]], float) + center
    return current, background, skeleton


def test_actual_foreground_support_refines_projected_body_position():
    image, background, uv = fish_image()
    scene = SceneEvidence.from_frames({"top": image}, {"top": background})
    correct = scene.score("top", uv)
    displaced = scene.score("top", uv + np.array([0, 14]))
    assert correct.silhouette_quality == pytest.approx(1)
    assert correct.silhouette_cost < 0.05
    assert displaced.silhouette_cost > 1

    def residual(offset):
        distances, quality = scene.silhouette_residuals("top", uv + np.array([0, offset[0]]))
        return distances * np.sqrt(quality)

    fitted = least_squares(residual, [14.0], bounds=([-20], [20]))
    assert fitted.cost < 1e-6
    assert abs(fitted.x[0]) < 5


def test_soft_silhouette_caps_far_outlier_and_missing_observations_are_neutral():
    image, background, uv = fish_image(shape=(240, 240))
    scene = SceneEvidence.from_frames({"top": image}, {"top": background})
    far = scene.score("top", uv + np.array([120, 120]))
    assert far.silhouette_cost == pytest.approx(25)
    missing = SceneEvidence.from_frames({"top": image})
    empty = SceneEvidence.from_frames({"top": background}, {"top": background})
    for unavailable in (missing, empty):
        score = unavailable.score("top", uv)
        assert score.cost == 0
        assert score.silhouette_quality == 0
        assert score.appearance is None
    outside = scene.score("top", uv + np.array([1000, 1000]))
    assert outside.silhouette_quality == 0
    assert np.isnan(scene.views["top"].distance_at(uv + np.array([1000, 1000]))).all()


def test_stationary_foreground_survives_and_uniform_illumination_does_not_become_fish():
    image, background, uv = fish_image()
    scene = SceneEvidence.from_frames({"top": image}, {"top": background}, {"top": image})
    assert scene.score("top", uv).silhouette_quality > 0.9
    brightened = np.minimum(background.astype(int) + 20, 255).astype(np.uint8)
    changed = SceneEvidence.from_frames({"top": brightened}, {"top": background})
    assert changed.views["top"].quality == 0


def test_foreground_color_distinguishes_candidates_with_translation_invariance():
    first, background, uv = fish_image()
    moved, _, moved_uv = fish_image(center=(80, 70))
    other, _, _ = fish_image(color=(30, 100, 240))
    reference = SceneEvidence.from_frames({"top": first}, {"top": background}).score("top", uv)
    same = SceneEvidence.from_frames({"top": moved}, {"top": background}).score(
        "top", moved_uv, reference.appearance
    )
    different = SceneEvidence.from_frames({"top": other}, {"top": background}).score(
        "top", uv, reference.appearance
    )
    assert reference.appearance_quality > 0.8
    assert same.appearance_cost < 1e-8
    assert different.appearance_cost > 0.5
    assert reference.appearance.sum() == pytest.approx(1)
    assert not reference.appearance.flags.writeable


def test_identical_reflection_is_ambiguous_and_occlusion_disables_identity_terms():
    image, background, uv = fish_image()
    cv2.ellipse(image, (85, 75), (18, 5), 0, 0, 360, (240, 130, 30), -1)
    scene = SceneEvidence.from_frames({"top": image}, {"top": background}, {"top": image})
    original = scene.score("top", uv)
    reflected = scene.score("top", uv + np.array([35, 25]), original.appearance)
    assert reflected.silhouette_cost == pytest.approx(original.silhouette_cost)
    assert reflected.appearance_cost == pytest.approx(0)
    # Raw image evidence cannot establish that an identical copy is a reflection.
    overlap = scene.score("top", uv, original.appearance, previous_uv=uv, occluded=True)
    assert overlap.appearance is None
    assert overlap.appearance_quality == 0
    assert overlap.motion_quality == 0
    assert overlap.silhouette_quality < original.silhouette_quality


def test_large_component_and_flood_reduce_local_appearance_reliability():
    image, background, uv = fish_image()
    good = SceneEvidence.from_frames({"top": image}, {"top": background}).score("top", uv)
    flooded = image.copy()
    flooded[20:80, 20:100] = (240, 130, 30)
    uncertain = SceneEvidence.from_frames({"top": flooded}, {"top": background}).score("top", uv)
    assert uncertain.appearance_quality < good.appearance_quality * 0.5
    assert uncertain.silhouette_quality < good.silhouette_quality


def test_flow_recovers_translation_and_discount_flat_image():
    rng = np.random.default_rng(416)
    previous = np.full((96, 128), 130, np.uint8)
    previous[30:65, 30:80] = rng.integers(30, 230, size=(35, 50), dtype=np.uint8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 4], [0, 1, 2]]), (128, 96),
                             borderMode=cv2.BORDER_REPLICATE)
    before = np.array([[40, 40], [52, 45], [65, 50]], float)
    scene = SceneEvidence.from_frames({"side": current}, previous_frames={"side": previous})
    correct = scene.score("side", before + np.array([4, 2]), previous_uv=before)
    wrong = scene.score("side", before + np.array([-4, -2]), previous_uv=before)
    assert correct.motion_quality > 0.8
    assert correct.motion_cost < 0.05
    assert wrong.motion_cost > 5
    flat = np.full_like(previous, 130)
    unknown = SceneEvidence.from_frames({"side": flat}, previous_frames={"side": flat})
    assert unknown.score("side", before + np.array([4, 2]), previous_uv=before).motion_quality == 0
    disappeared = SceneEvidence.from_frames({"side": flat}, previous_frames={"side": previous})
    hidden = disappeared.score("side", before, previous_uv=before)
    assert hidden.motion_quality < 0.2


def test_reduced_flow_preserves_original_pixel_displacement():
    rng = np.random.default_rng(911)
    previous = rng.integers(30, 220, (192, 256), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0.8)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 6], [0, 1, 4]]), (256, 192),
                             borderMode=cv2.BORDER_REPLICATE)
    before = np.array([[80, 70], [120, 85], [150, 105]], float)
    scene = SceneEvidence.from_frames(
        {"side": current}, previous_frames={"side": previous},
        config=EvidenceConfig(flow_max_side=128),
    )
    result = scene.score("side", before + np.array([6, 4]), previous_uv=before)
    assert scene.views["side"].flow.shape[:2] == (96, 128)
    assert result.motion_quality > 0.5
    assert result.motion_cost < 0.1


def test_maps_are_readonly_and_camera_mismatch_fails_loudly():
    image, background, uv = fish_image()
    scene = SceneEvidence.from_frames({"top": image}, {"top": background})
    with pytest.raises(TypeError):
        scene.views["other"] = scene.views["top"]
    with pytest.raises(ValueError):
        scene.views["top"].foreground[0, 0] = 1
    with pytest.raises(ValueError, match="camera names"):
        SceneEvidence.from_frames({"top": image}, {"side": background})
    with pytest.raises(ValueError, match="match"):
        SceneEvidence.from_frames({"top": image}, {"top": background[:10]})
    with pytest.raises(ValueError, match="shape"):
        scene.score("top", uv[:2])
    with pytest.raises(ValueError, match="histogram"):
        scene.score("top", uv, np.ones(2))
    with pytest.raises(ValueError, match="floating"):
        SceneEvidence.from_frames({"top": image.astype(float)})


def test_rgb_float_and_uint8_produce_same_evidence_and_nan_nodes_are_neutral():
    image, background, _ = fish_image()
    byte = SceneEvidence.from_frames({"top": image}, {"top": background})
    floating = SceneEvidence.from_frames({"top": image / 255}, {"top": background / 255})
    assert np.array_equal(byte.views["top"].foreground, floating.views["top"].foreground)
    score = byte.score("top", np.full((3, 2), np.nan))
    assert score.cost == 0
    assert score.appearance is None


@pytest.mark.parametrize("shape", [(41, 53), (40, 54)])
def test_histogram_median_matches_partition_for_random_extreme_and_illumination_scenes(
    monkeypatch, shape
):
    rng = np.random.default_rng(925)
    background = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    extreme = np.empty_like(image)
    extreme.flat[::2] = 0
    extreme.flat[1::2] = 255
    brighter = np.minimum(background.astype(np.int16) + 18, 255).astype(np.uint8)
    scenes = [image, extreme, brighter]
    fast = [SceneEvidence.from_frames({"top": item}, {"top": background}) for item in scenes]
    original_median = observations._integer_channel_median
    for item in scenes:
        delta = np.subtract(item, background, dtype=np.int16)
        assert np.array_equal(original_median(delta), np.median(delta.astype(np.float32),
                                                                axis=(0, 1)))
    monkeypatch.setattr(observations, "_integer_channel_median",
                        lambda delta: np.median(delta.astype(np.float32), axis=(0, 1)))
    for item, expected in zip(scenes, fast, strict=True):
        reference = SceneEvidence.from_frames({"top": item}, {"top": background})
        for attribute in ("foreground", "distance", "labels", "component_areas"):
            assert np.array_equal(getattr(expected.views["top"], attribute),
                                  getattr(reference.views["top"], attribute))
        assert expected.views["top"].quality == reference.views["top"].quality


def test_uint8_fastpath_copies_noncontiguous_input_without_freezing_caller():
    source = np.arange(120, dtype=np.uint8).reshape(4, 10, 3)[:, ::2]
    expected = source.copy()
    scene = SceneEvidence.from_frames({"top": source})
    assert source.flags.writeable
    source[:] = 0
    assert np.array_equal(scene.views["top"].rgb, expected)
    assert scene.views["top"].rgb.flags.c_contiguous
    assert not scene.views["top"].rgb.flags.writeable
