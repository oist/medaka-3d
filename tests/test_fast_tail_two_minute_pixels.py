"""Focused preparation boundaries: no tracking or real mask payloads."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import prepare_fast_tail_two_minute_pixels as m


def samples():
    fit = np.tile(np.column_stack((np.arange(1024), np.zeros(1024))), (3, 1, 1)).astype(float)
    unused = np.tile(np.column_stack((np.arange(256), np.ones(256))), (3, 1, 1)).astype(float)
    return dict(
        fit_pixels=fit,
        fit_target=np.zeros((3, 1024), np.float32),
        fit_weights=np.zeros((3, 1024), np.float32),
        fit_foreground=np.zeros((3, 1024), bool),
        pixels=unused,
        target=np.zeros((3, 256), np.float32),
        weights=np.zeros((3, 256), np.float32),
        reused_fitted_coordinate=np.zeros((3, 256), bool),
    )


def test_fixed_schedule_extends_old_uniform_minute():
    assert m.FRAMES[:60] == tuple(m.old.SCORED.tolist())
    assert len(m.FRAMES) == 120 and m.FRAMES[-1] == 86237
    assert all(b - a == 166 for a, b in zip(m.FRAMES[:-1], m.FRAMES[1:], strict=True))
    assert sum(m.HALF <= f < m.STOP for f in m.FRAMES) == 60


def test_neutral_empty_sample_validates():
    m.validate_samples(samples())


@pytest.mark.parametrize(
    "corruption", ["overlap", "fractional", "outside", "nan", "dtype", "weight", "target"]
)
def test_sample_corruption_rejected_even_if_overlap_flag_false(corruption):
    a = samples()
    if corruption == "overlap":
        a["pixels"][2, 11] = a["fit_pixels"][2, 999]
    elif corruption == "fractional":
        a["pixels"][0, 0, 0] = 0.5
    elif corruption == "outside":
        a["pixels"][0, 0, 1] = 1200
    elif corruption == "nan":
        a["weights"][1, 2] = np.nan
    elif corruption == "dtype":
        a["pixels"] = a["pixels"].astype(np.float32)
    elif corruption == "weight":
        a["weights"][1, 2] = -1
    else:
        a["target"][1, 2] = 0.5
    with pytest.raises(ValueError):
        m.validate_samples(a)


def packed_fixture():
    masks = np.zeros((0, 1200, 1200), bool)
    digest = hashlib.sha256(masks.tobytes()).hexdigest().encode()
    return masks, {
        v: dict(
            frame_idx=np.arange(m.HALF, m.HALF + m.CHUNK),
            image_shape_hw=np.array([1200, 1200]),
            view_available=np.full(m.CHUNK, i != 1, bool),
            frame_mask_sha256=np.full(m.CHUNK, digest, "S64"),
        )
        for i, v in enumerate(m.old.VIEWS)
    }


def test_absent_view_distinct_from_available_empty(monkeypatch):
    decoded, packed = packed_fixture()
    monkeypatch.setattr(m.old, "decode_frame_masks", lambda data, index: decoded)
    frame = m.frame_from_masks(packed, m.HALF, m.HALF + 83)
    assert frame.masks[m.old.VIEWS[1]] is None
    assert frame.masks[m.old.VIEWS[0]].shape == (0, 1200, 1200)
    assert frame.images_bgr[m.old.VIEWS[0]].shape == (1200, 1200, 3)


def test_decoded_mask_corruption_rejected(monkeypatch):
    decoded, packed = packed_fixture()
    packed[m.old.VIEWS[0]]["frame_mask_sha256"][83] = b"0" * 64
    monkeypatch.setattr(m.old, "decode_frame_masks", lambda data, index: decoded)
    with pytest.raises(ValueError, match="decoded mask SHA"):
        m.frame_from_masks(packed, m.HALF, m.HALF + 83)


@pytest.mark.requires_artifacts(
    m.ORIGINAL / "continuation.provenance.json",
    m.PREPARED / "preparation.provenance.json",
)
def test_metadata_only_real_seals_never_open_arrays_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected numerical payload or CUDA initialization")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(m.torch.cuda, "init", forbidden)
    e = m.proof.Evidence()
    original, previous, summary, config, sources = m.metadata(e)
    assert len(sources) == 36
    assert summary["scored_frames"] == list(m.FRAMES[:60])
    assert config["mode"] == "detectron2"
    assert not any(Path(p).suffix == ".npz" for p in e.paths)
    assert original["seal"]["status"] == previous["seal"]["status"] == "ok"
    assert not m.torch.cuda.is_initialized()
