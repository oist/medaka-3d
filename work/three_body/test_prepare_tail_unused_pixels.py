"""Synthetic fixed sampling, strict exclusion and calibrated ray contracts."""

from types import SimpleNamespace

import numpy as np
import prepare_tail_unused_pixels as m
import pytest
import torch
from renderer import RefractiveFishRenderer
from test_prepare_bounded_mouth_unused_pixels import fixture
from test_tail_recovery import rig  # noqa: F401


def test_exact_frozen_sampler_call_and_fixed301_scope():
    assert m.prepare_frame is m.prior.prepare_frame
    assert m.FRAMES.tolist() == [66524, *range(66620, 66920)]
    assert m.sampler.COUNTS == (96, 96, 64) and m.sampler.SEED_OFFSET == 8803010
    frame, cfg, saved = fixture()
    before = m.packets.signature(saved)
    actual, member = m.prepare_frame(frame, cfg, saved)
    repeated, _ = m.prior.prepare_frame(frame, cfg, saved)
    assert m.packets.signature(actual) == m.packets.signature(repeated)
    assert before == m.packets.signature(saved)
    audited = m.coordinate_audit(actual, member, shape=(48, 48))
    assert len(audited) == 3
    for row in audited:
        assert row["selected_draw_count"] == 256
        assert row["selected_unique_all"] + row["duplicate_draws_including_cross_class"] == 256
        assert row["strict_fit_evaluation_overlap"] == 0
        assert row["selected_draws_by_class"] == [96, 96, 64]


@pytest.mark.parametrize("field", m.FIT_KEYS)
def test_actual_fit_field_change_is_rejected_before_unused_sampling(field):
    frame, cfg, saved = fixture()
    saved[field] = saved[field].copy()
    saved[field].flat[0] = not saved[field].flat[0] if saved[field].dtype == bool else 999
    with pytest.raises(ValueError, match="fit bytes"):
        m.prepare_frame(frame, cfg, saved)


def test_duplicate_draws_and_cross_stratum_reuse_are_not_unique_evidence():
    frame, cfg, saved = fixture()
    value, member = m.prepare_frame(frame, cfg, saved)
    for vi in range(3):
        value["pixels"][vi] = np.tile(value["pixels"][vi, 0], (256, 1))
        member[vi]["selected_unique_by_class"] = [1, 1, 1]
    rows = m.coordinate_audit(value, member, shape=(48, 48))
    assert all(
        r["selected_unique_all"] == 1 and r["duplicate_draws_including_cross_class"] == 255
        for r in rows
    )
    assert all(r["duplicate_draws_within_class"] == [95, 95, 63] for r in rows)


@pytest.mark.parametrize("fault", ["overlap", "fraction", "bounds", "order"])
def test_disjoint_native_coordinate_validation(fault):
    frame, cfg, saved = fixture()
    value, member = m.prepare_frame(frame, cfg, saved)
    if fault == "overlap":
        value["pixels"][0, 0] = saved["fit_pixels"][0, 0]
    if fault == "fraction":
        value["pixels"][0, 0, 0] += 0.5
    if fault == "bounds":
        value["pixels"][0, 0, 0] = 48
    if fault == "order":
        member[0]["view"] = "wrong"
    with pytest.raises(ValueError):
        m.coordinate_audit(value, member, shape=(48, 48))


def test_calibrated_rays_equal_frozen_renderer_cache_without_render(rig):  # noqa: F811
    pixels = np.broadcast_to(np.array([[600.0, 600.0], [500.0, 550.0]]), (3, 2, 2))
    pixels = np.tile(pixels, (1, 128, 1))
    before = pixels.copy()
    ray = m.sampled_rays(rig, pixels)
    renderer = RefractiveFishRenderer(
        rig, {v: pixels[i] for i, v in enumerate(m.VIEWS)}, device="cpu", dtype=torch.float64
    )
    assert np.array_equal(pixels, before)
    for name in ("ray_origins", "ray_directions", "ray_valid"):
        assert m.checks.equal_bytes(ray[name], getattr(renderer, name).numpy())


def test_invalid_calibrated_rays_are_flagged_and_not_fabricated_valid():
    def rays(pixels):
        o = np.zeros((len(pixels), 3))
        d = np.tile([0.0, 0.0, 2.0], (len(pixels), 1))
        o[1] = np.nan
        d[2] = 0
        d[3] = np.inf
        return o, d

    fake_rig = SimpleNamespace(camera=lambda view: SimpleNamespace(rays=rays))
    result = m.sampled_rays(fake_rig, np.zeros((3, 256, 2)))
    assert result["ray_valid"].sum() == 3 * 253
    assert not result["ray_valid"][:, 1:4].any()
    assert np.isfinite(result["ray_origins"]).all() and np.isfinite(result["ray_directions"]).all()
    assert np.array_equal(result["ray_directions"], np.broadcast_to([0.0, 0.0, 1.0], (3, 256, 3)))


@pytest.mark.parametrize("missing", [False, True])
def test_empty_or_unavailable_foreground_is_neutral_without_rawpixels(missing):
    frame, cfg, _ = fixture()
    frame.masks = {v: None if missing else np.empty((0, 48, 48), bool) for v in m.VIEWS}
    target = m.prior.sample_pixels(frame, "detectron2", 32, 32, 16)
    saved = dict(
        fit_pixels=np.stack([target.pixels[v] for v in m.VIEWS]),
        fit_target=target.target,
        fit_weights=target.weights,
        fit_foreground=target.foreground,
    )
    values, audit = m.prepare_frame(frame, cfg, saved)
    assert not values["weights"].any()
    assert all(r["source_mask_available"] is (not missing) for r in audit)
    assert not values["reused_fitted_coordinate"].any()
