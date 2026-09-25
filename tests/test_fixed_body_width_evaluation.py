"""Six-arm common-policy evaluation; synthetic CPU observations only."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import fixed_body_width_evaluation as m  # noqa: E402
import mask_transport_gap_evaluation as old  # noqa: E402
from renderer import RendererConfig  # noqa: E402

from test_mask_transport_gap_evaluation import exact, image_fixture, scene  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def first_four(result, reference):
    for key, expected in reference.items():
        if key.startswith(("owner_", "gap_", "combined_")):
            actual = result[key][:, :4]
        elif key.startswith(("union_", "rendered_")):
            actual = result[key][:4]
        else:
            actual = result[key]
        exact(actual, expected)


def six_images():
    poses, sample, packets, rig = image_fixture()
    static_base = (poses[0] - poses[0, :, 1:2]) * 1.25 + poses[0, :, 1:2]
    static_gap = (poses[3] - poses[3, :, 1:2]) * 1.5 + poses[3, :, 1:2]
    return np.concatenate((poses, np.stack((static_base, static_gap)))), sample, packets, rig


@pytest.mark.parametrize("all_invalid", [False, True])
def test_six_score_axes_preserve_every_old_four_value_and_common_zero_mass(all_invalid):
    arrays, q = scene()
    reference = old.render_scores(*arrays, rendered_q=q)
    six = [a.copy() for a in arrays]
    six[0] = np.concatenate((arrays[0], arrays[0][-2:]))
    six[1] = np.concatenate((arrays[1], arrays[1][-2:]))
    six_q = np.concatenate((q, q[-2:]))
    if all_invalid:
        six[-1][:] = False
        six_q[:] = np.inf
        reference = old.render_scores(six[0][:4], six[1][:4], *six[2:], rendered_q=six_q[:4])
    before = [a.tobytes() for a in (*six, six_q)]
    result = m.render_scores(*six, rendered_q=six_q)
    first_four(result, reference)
    assert result["combined_per_body_view"].shape == (2, 6, 2, 3, 3)
    assert result["combined_loss"].shape == (2, 6, 2)
    assert result["target_mass"].shape == (2, 2, 3, 3)
    assert [a.tobytes() for a in (*six, six_q)] == before


def test_actual_synthetic_cpu_primary_renderer_first_four_arrays_exact():
    poses, sample, packets, rig = six_images()
    expected = old.score_fixed_images(poses[:4], sample, packets, rig)
    result = m.score_fixed_images(poses, sample, packets, rig)
    first_four(result, expected)
    assert str(result["renderer_policy"]) == "dynamic_pose_length"
    assert result["fixed_radial_sigma_mm"].shape == (0, 0)
    assert result["reference_pose_xyz"].shape == (0, 3, 3)


def test_secondary_uses_one_registered_renderer_for_all_six_including_old_arms(monkeypatch):
    poses, sample, packets, rig = six_images()
    reference = poses[0].copy()
    expected_renderer = m.FixedWidthRenderer(
        rig,
        dict(zip(m.VIEWS, sample["unused_pixels"], strict=True)),
        RendererConfig(width_ratio=0.06),
        dtype=torch.float64,
        reference_pose_xyz=reference,
    )
    with torch.no_grad():
        rendered = expected_renderer.render(poses)
    expected = old.render_scores(
        rendered.union_alpha.numpy()[:4],
        rendered.ownership.numpy()[:4],
        np.stack([p.target for p in packets]),
        sample["unused_target"],
        sample["unused_weights"],
        rendered.ray_valid.numpy(),
        rendered_q=rendered.per_fish_mahalanobis_sq.numpy()[:4],
    )
    constructors = []
    implementation = m.FixedWidthRenderer

    def construct(*args, **kwargs):
        assert kwargs["reference_pose_xyz"] is reference
        constructors.append(True)
        return implementation(*args, **kwargs)

    monkeypatch.setattr(m, "FixedWidthRenderer", construct)
    result = m.score_fixed_images(poses, sample, packets, rig, reference_pose_xyz=reference)
    assert constructors == [True]
    first_four(result, expected)
    exact(result["rendered_union"], rendered.union_alpha.numpy())
    exact(result["rendered_ownership"], rendered.ownership.numpy())
    exact(result["rendered_q"], rendered.per_fish_mahalanobis_sq.numpy())
    exact(result["fixed_radial_sigma_mm"], expected_renderer.radial_sigma_mm.numpy())
    exact(result["reference_pose_xyz"], reference)
    assert str(result["renderer_policy"]) == "registered_fixed_radial"
    primary = m.score_fixed_images(poses, sample, packets, rig)
    assert not np.array_equal(primary["rendered_q"][4:], result["rendered_q"][4:])


@pytest.mark.parametrize("reference", ["per_arm", "nonfinite", "f32"])
def test_invalid_reference_or_per_arm_policies_are_rejected(reference):
    poses, sample, packets, rig = six_images()
    value = poses[0].copy()
    if reference == "per_arm":
        value = poses.copy()
    elif reference == "nonfinite":
        value[0, 0, 0] = np.nan
    else:
        value = value.astype(np.float32)
    with pytest.raises(ValueError, match="registered reference"):
        m.score_fixed_images(poses, sample, packets, rig, reference_pose_xyz=value)


@pytest.mark.parametrize("policy", ["primary", "secondary"])
def test_shared_target_pixels_and_anchor_guards_precede_both_renderer_policies(monkeypatch, policy):
    poses, sample, packets, rig = six_images()
    packets[1] = dataclasses.replace(packets[1], query_sha256=("0" * 64,) * 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("renderer reached with noncommon evidence")

    monkeypatch.setattr(m, "RefractiveFishRenderer", forbidden)
    monkeypatch.setattr(m, "FixedWidthRenderer", forbidden)
    reference = None if policy == "primary" else poses[0].copy()
    with pytest.raises(ValueError, match="common masks/query bytes/release"):
        m.score_fixed_images(poses, sample, packets, rig, reference_pose_xyz=reference)


def test_static_base_contrast_is_distinct_from_original_without_mutating_candidates():
    frames = np.arange(333519, 333719, dtype=np.int64)
    baseline = np.ones((200, 3, 3), bool)
    baseline[10:15, 2, 2] = False
    candidates = {
        a: dict(
            frame_idx=frames.copy(),
            observed=baseline.copy(),
            pose_xyz=np.zeros((200, 3, 3, 3), np.float64),
        )
        for a in m.ARMS
    }
    candidates["static_base"]["observed"][20:25, 0, 1] = False
    candidates["static_gap"]["observed"][20:23, 0, 1] = False
    candidates["static_gap"]["observed"][199, 1, 0] = False
    before = {a: {k: v.tobytes() for k, v in row.items()} for a, row in candidates.items()}
    expected = old.summarize_support({a: candidates[a] for a in old.ARMS})
    primary = m.summarize_support(candidates)
    assert {a: primary[a] for a in old.ARMS} == expected
    secondary = m.summarize_static_support(candidates)
    assert secondary["baseline"] == "static_base" and secondary["arm"] == "static_gap"
    row = secondary["support"]
    assert row["all_node_time_denominator"] == 1800 and row["per_node_denominator"] == 600
    assert row["gain_rows"] == [
        dict(native=333542, body=0, node=1),
        dict(native=333543, body=0, node=1),
    ]
    assert row["loss_rows"] == [dict(native=333718, body=1, node=0)]
    assert primary["static_gap"]["direct_gain_by_body_node"] == np.zeros((3, 3), int).tolist()
    assert before == {a: {k: v.tobytes() for k, v in row.items()} for a, row in candidates.items()}
