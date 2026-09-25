"""Thin sample selection, shared crop and isolated synthetic rendering only."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import render_observation_reserve_continuation as m


def fail(*args, **kwargs):
    pytest.fail("original data or calibration pipeline invoked")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "PHOTO_ROOT", tmp_path / "photos")
    monkeypatch.setattr(m.drawing, "load_context", fail)
    monkeypatch.setattr(m.drawing, "extract_native", fail)
    monkeypatch.setattr(m.drawing, "load_science", fail)
    monkeypatch.setattr(m.calib, "load_rig", fail)
    monkeypatch.setattr(m.cv2, "VideoCapture", fail)
    original_open, original_load = Path.open, np.load

    def guarded(path, *args, **kwargs):
        assert path.resolve().is_relative_to(tmp_path)
        return original_open(path, *args, **kwargs)

    def load(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(m.np, "load", load)


def sample_fixture(tmp_path):
    data = {}
    for arm_index, arm in enumerate(m.ARMS):
        poses = np.zeros((5, 3, 3, 3), np.float64)
        for t in range(5):
            poses[t, :, :, 0] = 400 + t * 40 + arm_index * 70
            poses[t, :, :, 1] = 400 + np.arange(3)[:, None] * 80
            poses[t, :, :, 2] = np.arange(3)[None, :] * 15
        fields = dict(
            frame_idx=np.array(m.PROBES, np.int32),
            pose_xyz=poses,
            observed=np.ones((5, 3, 3), bool),
        )
        fields["observed"][2, 1, 2] = False
        for field in m.ALL_FIELDS:
            # Loading any undeclared field would require pickle and immediately fail.
            data[arm + "_" + field] = fields.get(field, np.array([object()], dtype=object))
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    payload = stream.getvalue()
    path = tmp_path / "review_samples.npz"
    path.write_bytes(payload)
    context = dict(
        sample_reference=dict(
            path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
        ),
        sample_signatures={key: m.prep.signature(value) for key, value in data.items()},
    )
    return context, data


def test_only_six_fields_and_middle_three_native_frames(tmp_path):
    context, data = sample_fixture(tmp_path)
    evidence = m.prep.Evidence()
    selected, verified = m.load_samples(context, evidence)
    assert len(verified) == 6 and len(evidence.records) == 1
    for arm in m.ARMS:
        assert selected[arm]["frame_idx"].tolist() == list(m.FRAMES)
        for field in m.FIELDS:
            npt.assert_array_equal(selected[arm][field], data[arm + "_" + field][1:4])
        assert not np.shares_memory(selected[arm]["pose_xyz"], data[arm + "_pose_xyz"])


class Camera:
    def __init__(self, view):
        self.shift = m.VIEWS.index(view) * 20

    def project(self, pose):
        return pose[..., :2] + self.shift


class Rig:
    def camera(self, view):
        return Camera(view)


def test_common_crop_accounts_for_all_three_frames_and_both_arms(tmp_path):
    context, _ = sample_fixture(tmp_path)
    samples, _ = m.load_samples(context, m.prep.Evidence())
    projected, crops = m.project_and_crop(samples, Rig())
    assert len(crops) == 3
    for view in m.VIEWS:
        expected = m.drawing.crop_for(np.stack([projected[arm][view] for arm in m.ARMS]))
        assert crops[view] == expected
        x, y, width, height = crops[view]
        points = np.stack([projected[arm][view] for arm in m.ARMS]).reshape(-1, 2)
        assert ((points >= [x, y]) & (points < [x + width, y + height])).all()


def test_wrong_native_samples_rejected_even_with_own_valid_signature(tmp_path):
    context, data = sample_fixture(tmp_path)
    data["nodewise_frame_idx"][2] = 123
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    payload = stream.getvalue()
    Path(context["sample_reference"]["path"]).write_bytes(payload)
    context["sample_reference"].update(sha256=m.prep.sha(payload), size_bytes=len(payload))
    context["sample_signatures"]["nodewise_frame_idx"] = m.prep.signature(
        data["nodewise_frame_idx"]
    )
    with pytest.raises(ValueError, match="ordered native"):
        m.load_samples(context, m.prep.Evidence())


def test_isolated_render_real_sample_decode_nine_pngs_and_json(monkeypatch, tmp_path):
    context, _ = sample_fixture(tmp_path)
    context.update(comparison_binding=dict(synthetic=True), rig_references={}, photographs={})
    for frame in m.FRAMES:
        for view in m.VIEWS:
            image = np.zeros(m.SHAPE, np.uint8)
            image[..., 0] = frame % 255
            image[..., 1] = np.arange(1200, dtype=np.uint16)[:, None] % 256
            image[..., 2] = m.VIEWS.index(view) * 80
            data = m.prep.png_bytes(image)
            path = tmp_path / f"native_{frame}_{view}.png"
            path.write_bytes(data)
            context["photographs"][(frame, view)] = dict(
                frame_idx=frame,
                view=view,
                bgr_signature=m.prep.signature(image),
                image=dict(path=str(path), sha256=m.prep.sha(data), size_bytes=len(data)),
            )
    monkeypatch.setattr(m, "source_code", lambda evidence: b"synthetic source")
    monkeypatch.setattr(m, "load_headers", lambda *args: context)
    monkeypatch.setattr(m, "load_rig", lambda *args: Rig())
    original_draw = m.drawing.draw_panel
    calls = []

    def draw(image, crop, title, projected=None, observed=None):
        calls.append(
            (crop.copy(), title, projected is None, None if observed is None else observed.copy())
        )
        return original_draw(image, crop, title, projected, observed)

    monkeypatch.setattr(m.drawing, "draw_panel", draw)
    output = tmp_path / "result"
    assert m.run(tmp_path / "comparison", "c" * 64, output) == dict(status="ok")
    seal = json.loads((output / "render.provenance.json").read_bytes())
    assert seal["status"] == "ok" and len(seal["outputs"]) == 10
    assert len(seal["inputs"]) == 10  # one NPZ plus nine PNG, no old input loaders
    assert len(calls) == 27 and sum(row[2] for row in calls) == 9
    crops = json.loads((output / "plan.json").read_bytes())["crops_native_xywh"]
    for crop, title, _, _ in calls:
        view = next(view for view in m.VIEWS if view in title)
        assert crop == crops[view]
    result = m.cv2.imdecode(
        np.frombuffer((output / "frame_75054.png").read_bytes(), np.uint8), m.cv2.IMREAD_COLOR
    )
    assert result.shape == (2006, 1800, 3)
    projected = json.loads((output / "projection_evidence.json").read_bytes())
    assert [r["frame_idx"] for r in projected["selected"]] == list(m.FRAMES)
    assert projected["selected"][1]["arms"]["nodewise"]["observed"][1][2] is False
    assert projected["raw_slots_are_physical_IDs"] is False
    assert not projected["support_recomputed"]
