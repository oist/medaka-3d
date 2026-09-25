"""Mask-only synthetic tests; no fitted result or actual image is opened."""

from types import SimpleNamespace

import numpy as np
import prepare_bounded_mouth_unused_pixels as m
import pytest


def fixture():
    masks = np.zeros((1, 48, 48), bool)
    masks[:, 18:30, 18:30] = True
    frame = SimpleNamespace(
        frame_idx=67663,
        masks={v: masks.copy() for v in m.VIEWS},
        images_bgr={v: SimpleNamespace(shape=(48, 48, 3)) for v in m.VIEWS},
    )
    cfg = dict(mode="detectron2", positive_samples=32, near_samples=32, global_samples=16)
    target = m.sample_pixels(frame, "detectron2", 32, 32, 16)
    saved = dict(
        fit_pixels=np.stack([target.pixels[v] for v in m.VIEWS]),
        fit_target=target.target,
        fit_weights=target.weights,
        fit_foreground=target.foreground,
    )
    return frame, cfg, saved


def test_fixed_counts_seed_exact_reproducible_and_no_overlap():
    frame, cfg, saved = fixture()
    before = m.packets.signature(saved)
    arrays, membership = m.prepare_frame(frame, cfg, saved)
    repeated, _ = m.prepare_frame(frame, cfg, saved)
    assert m.packets.signature(arrays) == m.packets.signature(repeated)
    assert before == m.packets.signature(saved)
    assert arrays["pixels"].shape == (3, 256, 2)
    assert not arrays["reused_fitted_coordinate"].any()
    for vi in range(3):
        assert membership[vi]["sampler_seed"] == 67663 * 7 + vi + 8803010
        assert not set(map(tuple, arrays["pixels"][vi])) & set(map(tuple, saved["fit_pixels"][vi]))


@pytest.mark.parametrize("field", m.FIT_KEYS)
def test_fit_field_tamper_rejected_before_sampling(field):
    frame, cfg, saved = fixture()
    saved[field] = saved[field].copy()
    saved[field].flat[0] = not saved[field].flat[0] if saved[field].dtype == bool else 999
    with pytest.raises(ValueError, match="fit bytes"):
        m.prepare_frame(frame, cfg, saved)


def test_contrast_requires_raw_and_is_explicitly_rejected():
    frame, cfg, saved = fixture()
    cfg["mode"] = "top_contrast"
    with pytest.raises(ValueError, match="mask-only"):
        m.prepare_frame(frame, cfg, saved)


def test_frozen_schedule_and_two_sampling_sources():
    assert m.FRAMES.tolist() == list(range(67663, 67696))
    assert m.sampler.COUNTS == (96, 96, 64)
    assert m.sampler.SEED_OFFSET == 8803010
    assert set(m.FIT_KEYS) == {"fit_pixels", "fit_target", "fit_weights", "fit_foreground"}
