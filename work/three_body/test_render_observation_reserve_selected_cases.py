"""Two native selections and isolated six-frame renderer; no real files decoded."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import render_observation_reserve_selected_cases as m


def fail(*args, **kwargs):
    pytest.fail("actual input pipeline invoked")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "QA", tmp_path / "qa")
    monkeypatch.setattr(m.np, "load", fail)
    monkeypatch.setattr(m.cv2, "VideoCapture", fail)
    monkeypatch.setattr(m.prep, "video_guard", fail)
    monkeypatch.setattr(m.prior, "load_rig", fail)
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path.resolve().is_relative_to(tmp_path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def diagnostic():
    poses = np.zeros((3, 3, 3), np.float64)
    poses[..., 0] = 500 + np.arange(3)[:, None] * 80
    poses[..., 1] = 500 + np.arange(3)[None, :] * 25
    rows = []
    for frame in m.FRAMES:
        rows.append(
            dict(
                frame_idx=frame,
                arms={
                    arm: dict(
                        pose_xyz=(poses + index).tolist(), observed=np.ones((3, 3), bool).tolist()
                    )
                    for index, arm in enumerate(m.ARMS)
                },
            )
        )
    rows[0]["arms"]["observation_reserve"]["observed"][2][1] = False
    return dict(
        status="ok",
        schema="observation_reserve_independent_output_qa_v1",
        comparison_seal_sha256=m.COMPARISON_SHA,
        tracking_seal_sha256=m.TRACKING_SHA,
        source_sha256=m.QA_SOURCE_SHA,
        process_interval=[74700, 76360],
        frames=1660,
        proposed_visual_scenes=[
            dict(category="newly_lost", representative_frame=75016),
            dict(category="newly_supported", representative_frame=75052),
        ],
        selected_samples=rows,
    )


def video_fixture():
    images = {frame: np.full(m.SHAPE, frame % 255, np.uint8) for frame in m.FRAMES}
    videos = {
        view: dict(
            video_path=view,
            frames=[
                dict(
                    frame_idx=f,
                    manifest_relative_row=f - 74700,
                    expected_bgr_sha256=m.prep.signature(images[f])["sha256"],
                )
                for f in m.FRAMES
            ],
        )
        for view in m.VIEWS
    }
    captures = []

    class Capture:
        def __init__(self, path):
            assert path in m.VIEWS
            self.path, self.position, self.seeks, self.released = path, 0, [], False
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                m.cv2.CAP_PROP_FRAME_COUNT: 348600,
                m.cv2.CAP_PROP_FPS: 166,
                m.cv2.CAP_PROP_FRAME_HEIGHT: m.SHAPE[0],
                m.cv2.CAP_PROP_FRAME_WIDTH: m.SHAPE[1],
                m.cv2.CAP_PROP_POS_FRAMES: self.position,
            }
            return values[prop]

        def set(self, prop, value):
            assert prop == m.cv2.CAP_PROP_POS_FRAMES and value in (75016, 75052)
            assert value not in (316, 352)
            self.position = value
            self.seeks.append(value)
            return True

        def read(self):
            image = images[self.position]
            self.position += 1
            return True, image

        def release(self):
            self.released = True

    return videos, Capture, captures


def test_json_pose_and_bool_recovery_rejects_null_and_wrong_selection():
    d = diagnostic()
    samples, scenes = m.select_samples(json.loads(json.dumps(d, allow_nan=False)))
    assert len(scenes) == 2
    for value in samples.values():
        assert value["pose_xyz"].dtype == np.float64
        assert value["pose_xyz"].shape == (2, 3, 3, 3)
        assert value["frame_idx"].tolist() == [75016, 75052]
    assert not samples["observation_reserve"]["observed"][0, 2, 1]
    bad = copy.deepcopy(d)
    bad["selected_samples"][0]["arms"]["nodewise"]["pose_xyz"][0][0][0] = None
    with pytest.raises(ValueError, match="finite whole-body"):
        m.select_samples(bad)
    bad = copy.deepcopy(d)
    bad["proposed_visual_scenes"][0]["representative_frame"] = 75053
    with pytest.raises(ValueError, match="representative"):
        m.select_samples(bad)


def test_six_absolute_frames_and_three_captures(monkeypatch):
    videos, factory, captures = video_fixture()
    guards = []
    monkeypatch.setattr(m.prep, "video_guard", lambda context: guards.append(context["video_path"]))
    images = m.extract_native(videos, capture_factory=factory)
    assert len(captures) == 3 and len(guards) == 6
    assert all(c.seeks == [75016, 75052] and c.released for c in captures)
    assert tuple(images) == (75016, 75052)
    assert all(tuple(views) == m.VIEWS for views in images.values())


def test_wrong_bgr_hash_still_releases_capture(monkeypatch):
    videos, factory, captures = video_fixture()
    videos[m.VIEWS[0]]["frames"][0]["expected_bgr_sha256"] = "a" * 64
    guards = []
    monkeypatch.setattr(m.prep, "video_guard", lambda context: guards.append(context["video_path"]))
    with pytest.raises(ValueError, match="BGR SHA"):
        m.extract_native(videos, capture_factory=factory)
    assert len(captures) == 1 and captures[0].released
    assert captures[0].seeks == [75016] and len(guards) == 2


def test_synthetic_run_two_scenes_six_native_pngs_and_previews(monkeypatch, tmp_path):
    samples, scenes = m.select_samples(diagnostic())
    videos, factory, captures = video_fixture()
    context = dict(samples=samples, scenes=scenes, videos=videos, rig_references={})
    monkeypatch.setattr(m, "source_code", lambda evidence: b"synthetic source only")
    monkeypatch.setattr(m, "load_headers", lambda evidence: context)
    monkeypatch.setattr(m.cv2, "VideoCapture", factory)
    monkeypatch.setattr(m.prep, "video_guard", lambda context: None)

    class Camera:
        def project(self, pose):
            return pose[..., :2]

    class Rig:
        def camera(self, view):
            return Camera()

    monkeypatch.setattr(m.prior, "load_rig", lambda *args: Rig())
    output = tmp_path / "output"
    assert m.run(output) == dict(status="ok")
    assert len(captures) == 3 and all(c.seeks == [75016, 75052] for c in captures)
    seal = json.loads((output / "render.provenance.json").read_bytes())
    assert seal["status"] == "ok" and len(seal["outputs"]) == 14
    summary = json.loads((output / "summary.json").read_bytes())
    assert summary["native_PNG_count"] == summary["actual_decoded_BGR_hashes_verified"] == 6
    assert summary["original_video_full_rehashes"] == summary["NPZ_decodes"] == 0
    assert summary["comparison_PNG_count"] == summary["preview_JPEG_count"] == 2
    metadata = json.loads((output / "projection_evidence.json").read_bytes())
    assert len(metadata["native_photographs"]) == 6
    assert {r["manifest_relative_row"] for r in metadata["native_photographs"]} == {316, 352}
    for row in metadata["native_photographs"]:
        raw = Path(row["image"]["path"]).read_bytes()
        image = m.cv2.imdecode(np.frombuffer(raw, np.uint8), m.cv2.IMREAD_COLOR)
        assert m.prep.signature(image) == row["bgr_signature"]
