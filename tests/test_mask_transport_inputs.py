"""Input integrity, native frame alignment and unused-pixel separation."""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import mask_transport_inputs as m  # noqa: E402


def observation(native, available=True):
    arrays = dict(
        frame_idx=np.asarray(native, np.int64),
        mask_available=np.asarray(available, bool),
        masks=np.zeros((0, 1200, 1200), bool),
        detections=np.full((2, 3, 2), np.nan, np.float32),
        example_packed_field=np.zeros(1, np.uint8),
    )
    row = dict(
        frame_idx=native,
        mask_available=available,
        mask_count=0,
        mask_sha256=m.base.signature(arrays["masks"])["sha256"],
        detections_sha256=m.base.signature(arrays["detections"])["sha256"],
        observation_array_signatures=m.prepared.delta.signatures(arrays),
    )
    return arrays, row


@pytest.mark.parametrize("change", ["points", "packed", "native", "fields"])
def test_observation_binds_all_fields_and_raw_SL_bytes(change):
    arrays, row = observation(m.FIRST)
    m.validate_observation(arrays, row)
    if change == "points":
        arrays["detections"][0, 0, 0] = 14
    elif change == "packed":
        arrays["example_packed_field"][0] = 1
    elif change == "native":
        arrays["frame_idx"] += 1
    else:
        arrays.pop("example_packed_field")
    with pytest.raises(ValueError):
        m.validate_observation(arrays, row)


def test_reader_last_native_three_views_and_mask_none_distinction(tmp_path, monkeypatch):
    image = np.zeros((1200, 1200, 3), np.uint8)
    ok, png = cv2.imencode(".png", image)
    assert ok
    image_sha = hashlib.sha256(image.tobytes()).hexdigest()
    store = m.base.Output(tmp_path / "input")
    png_record = store.write("raw.png", png.tobytes())
    all_rows = []
    for vi, view in enumerate(m.VIEWS):
        arrays, base_row = observation(m.STOP - 1, available=vi != 1)
        record = store.write(view + ".npz", arrays)
        for native in range(m.FIRST, m.STOP):
            all_rows.append(
                dict(
                    base_row,
                    view=view,
                    frame_idx=native,
                    native_offset=native - m.FIRST,
                    shape_hw=[1200, 1200],
                    bgr_sha256=image_sha,
                    png_path="raw.png",
                    png_sha256=png_record["sha256"],
                    observation_path=view + ".npz",
                    observation_sha256=record["sha256"],
                )
            )
    members = {r["path"]: r for r in store.records.values()}
    backgrounds = [
        dict(
            view=v,
            path="raw.png",
            encoded_sha256=png_record["sha256"],
            decoded_bgr_signature=m.base.signature(image),
        )
        for v in m.VIEWS
    ]
    reader = m.Prepared200(dict(natives=all_rows, members=members), backgrounds, root=store.root)
    frame = reader.at(m.STOP - 1)
    assert frame.frame_idx == m.STOP - 1
    assert frame.masks["side1_cam"] is None
    assert frame.masks["top_cam"].shape == (0, 1200, 1200)
    assert all(v.shape == (2, 3, 2) and v.dtype == np.float32 for v in frame.detections.values())
    assert all(not v.flags.writeable for v in frame.backgrounds_bgr.values())
    with pytest.raises(ValueError, match="outside"):
        reader.at(m.STOP)
    with pytest.raises(ValueError, match="native mismatch"):
        reader.at(m.FIRST)


class Camera:
    def rays(self, pixels):
        origins = np.zeros((len(pixels), 3), np.float64)
        directions = np.zeros_like(origins)
        directions[:, 2] = 1
        directions[0] = np.nan
        return origins, directions


class Rig:
    views = m.VIEWS
    world_units = "mm"

    def camera(self, _view):
        return Camera()


def test_fixed_samples_exclude_every_fit_coordinate_and_preserve_invalid_rays():
    image = np.zeros((64, 64, 3), np.uint8)
    masks = np.zeros((2, 64, 64), bool)
    masks[0, 16:48, 16:32] = True
    masks[1, 16:48, 32:48] = True
    frame = m.FrameObservation(
        m.FIRST,
        {v: np.zeros((0, 3, 2), np.float32) for v in m.VIEWS},
        {v: image for v in m.VIEWS},
        {v: masks for v in m.VIEWS},
        {v: image for v in m.VIEWS},
    )
    arrays, audit = m.fixed_samples(frame, Rig())
    assert arrays["fit_pixels"].shape == (3, 1024, 2)
    assert arrays["unused_pixels"].shape == (3, 256, 2)
    assert not arrays["unused_fit_overlap"].any()
    assert (~arrays["unused_ray_valid"]).sum() == 3
    for vi in range(3):
        old = {tuple(x) for x in arrays["fit_pixels"][vi]}
        assert not any(tuple(x) in old for x in arrays["unused_pixels"][vi])
        xy = arrays["unused_pixels"][vi].astype(int)
        truth = masks.any(axis=0)[xy[:, 1], xy[:, 0]]
        assert np.array_equal(truth, arrays["unused_target"][vi].astype(bool))
        assert audit[vi]["invalid_ray_draws"] == 1
        assert audit[vi]["sampler_seed"] == m.FIRST * 7 + vi + 8803010


def test_bgr_transport_rejects_same_shape_wrong_pixels():
    image = np.zeros((1200, 1200, 3), np.uint8)
    ok, png = cv2.imencode(".png", image)
    assert ok
    image[1, 2, 0] = 1
    with pytest.raises(ValueError, match="BGR bytes"):
        m.decode_bgr(png.tobytes(), hashlib.sha256(image.tobytes()).hexdigest())
