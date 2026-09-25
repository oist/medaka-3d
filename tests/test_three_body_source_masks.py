"""Fresh mask-only staging never publishes partial camera coverage as complete."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
SPEC = importlib.util.spec_from_file_location(
    "_source_mask_staging_test", LAB / "prepare_source_masks.py"
)
stage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage
SPEC.loader.exec_module(stage)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "WORKSPACE", tmp_path)
    cache, output = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    asset_entries = {}
    for kind, folder in stage.original.MODEL_DIRS.items():
        for role, filename in (
            ("weights", "model_final.pth"),
            ("config", "detectron2_config.yaml"),
        ):
            path = cache / folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fake {kind} {role}".encode())
            key = f"detectron2.{kind}" + (".config" if role == "config" else "")
            asset_entries[key] = dict(size=path.stat().st_size, sha256=stage.original.sha256(path))
    (cache / "assets").mkdir()
    (cache / "assets/MANIFEST.json").write_text(json.dumps(dict(assets=asset_entries)))
    files = {}
    for view in stage.original.VIEWS:
        relative = (
            f"medaka_202512/{stage.original.EXPERIMENT}/{view}/{stage.original.VIDEO_NAMES[view]}"
        )
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake native video {view}".encode())
        digest = stage.original.sha256(path)
        files[relative] = dict(
            bytes=path.stat().st_size, local_full_sha256=digest, source_stream_sha256=digest
        )
    path = cache / "source_cache_manifest.json"
    path.write_text(json.dumps(dict(status="verified", files=files)))
    (cache / "source_cache.provenance.json").write_text(
        json.dumps(
            dict(status="ok", outputs=dict(cache_manifest=dict(sha256=stage.original.sha256(path))))
        )
    )
    captures, inference_calls, runtime_calls, options = [], [], [], {}
    names = {value: key for key, value in stage.original.VIDEO_NAMES.items()}

    class Capture:
        def __init__(self, path):
            self.view, self.position, self.released = names[Path(path).name], 0, False
            captures.append(self)

        def isOpened(self):
            return self.view != options.get("open_failure")

        def get(self, key):
            return {
                cv2.CAP_PROP_FRAME_COUNT: options.get("source_count", 348600),
                cv2.CAP_PROP_FPS: options.get("fps", 166),
                cv2.CAP_PROP_FRAME_HEIGHT: 1200,
                cv2.CAP_PROP_FRAME_WIDTH: 1200,
                cv2.CAP_PROP_POS_FRAMES: self.position,
            }[key]

        def set(self, key, frame):
            assert key == cv2.CAP_PROP_POS_FRAMES
            self.position = frame
            return True

        def read(self):
            frame = self.position
            if options.get("read_failure") == (self.view, frame):
                return False, None
            if frame >= 348600:
                return False, None
            self.position += 1
            return True, np.full((1200, 1200, 3), frame % 251, np.uint8)

        def release(self):
            self.released = True

    class Tensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

    class Instances:
        def __init__(self, value):
            count = 4 if value == 100 else 0
            mask = np.zeros((count, 1200, 1200), bool)
            for k in range(count):
                mask[k, 20 + k, 50 + k] = True
            self.pred_masks = Tensor(mask)
            self.scores = Tensor(np.full(count, options.get("score", 0.8), np.float32))
            self.pred_boxes = SimpleNamespace(tensor=Tensor(np.tile([50, 20, 55, 25], (count, 1))))
            self.pred_classes = Tensor(np.zeros(count, np.int64))

        def to(self, device):
            assert device == "cpu"
            return self

    def runtime_factory():
        runtime_calls.append(True)

        def make_predictor(kind, assets, out):
            path = out / f"{kind}_effective_config.yaml"
            path.write_text("test fake predictor only", encoding="utf-8")

            def predictor(image):
                assert image.shape == (1200, 1200, 3) and image.dtype == np.uint8
                inference_calls.append((kind, int(image[0, 0, 0])))
                return {"instances": Instances(int(image[0, 0, 0]))}

            model = dict(
                effective_config=path.name, effective_config_sha256=stage.original.sha256(path)
            )
            return predictor, model

        return stage.InferenceRuntime(
            dict(test_only=True, batch_size=1), make_predictor, lambda: None
        )

    def no_writer(*args, **kwargs):
        raise AssertionError("mask-only staging must never open a VideoWriter")

    monkeypatch.setattr(cv2, "VideoWriter", no_writer)
    return SimpleNamespace(
        cache=cache,
        output=output,
        captures=captures,
        Capture=Capture,
        runtime_factory=runtime_factory,
        runtime_calls=runtime_calls,
        inference_calls=inference_calls,
        options=options,
    )


def test_all_three_views_seal_compatible_raw_masks_and_empty_observations(fixture, monkeypatch):
    f = fixture
    result = stage.run(
        f.cache, f.output, 100, 102, capture_factory=f.Capture, runtime_factory=f.runtime_factory
    )
    assert stage.verify_completion(f.output) == result
    assert result["frames"] == [100, 101] and result["video_payload_bytes_copied"] == 0
    assert result["raw_storage"] == "original_source_video"
    assert len(f.inference_calls) == 6 and len(f.runtime_calls) == 1
    assert all(cap.released for cap in f.captures)
    for view in stage.original.VIEWS:
        info = result["raw"][view]
        assert info["source_offset_frames"] == 100 and info["source_total_frames"] == 348600
        assert os.path.samefile(
            f.output / info["path"], f.cache / info["source_cache_relative_path"]
        )
        with np.load(f.output / result["masks"][view]["path"], allow_pickle=False) as data:
            assert data["view_available"].tolist() == [True, True]
            assert data["source_track_id"].tolist() == [-1] * 4
            assert stage.original.decode_frame_masks(data, 0).shape == (4, 1200, 1200)
            assert stage.original.decode_frame_masks(data, 1).shape == (0, 1200, 1200)
        assert result["backgrounds"][view]["frames"] == list(stage.original.BACKGROUND_CANDIDATES)
    # Staging succeeded without any SLEAP files. Supply tiny independent nodes only
    # now to exercise the existing reader's manifest and full-file validation.
    node_root = f.output.parent / "nodes"
    folder = node_root / "work/c01_s16_predictions/bottomup"
    folder.mkdir(parents=True)
    for view in stage.original.VIEWS:
        np.savez_compressed(
            folder / f"{stage.original.EXPERIMENT}__{view}.npz",
            view=view,
            experiment=stage.original.EXPERIMENT,
            nodes=["mouth", "pectoral", "tail"],
            frame_idx=np.array([100]),
            points=np.zeros((1, 3, 2)),
        )
    monkeypatch.setattr(cv2, "VideoCapture", f.Capture)
    from source_sequence import SourceSequenceReader

    frames = list(SourceSequenceReader(f.output, node_root))
    assert [frame.frame_idx for frame in frames] == [100, 101]
    assert frames[1].masks["top_cam"].shape[0] == 0
    assert frames[1].detections["top_cam"].shape[0] == 0


def test_camera_failure_after_two_views_never_publishes_complete_manifest(fixture):
    f = fixture
    f.options["read_failure"] = ("side2_cam", 101)
    with pytest.raises(RuntimeError, match="position mismatch"):
        stage.run(
            f.cache,
            f.output,
            100,
            102,
            capture_factory=f.Capture,
            runtime_factory=f.runtime_factory,
        )
    assert (f.output / "masks/side1_cam.npz").exists()
    assert not (f.output / "manifest.json").exists()
    seal = json.loads((f.output / "staging.provenance.json").read_text())
    assert seal["status"] == "failed"
    with pytest.raises(ValueError, match="did not complete"):
        stage.verify_completion(f.output)
    assert all(cap.released for cap in f.captures)


def test_long_window_background_shortage_fails_before_output_or_runtime(fixture):
    f = fixture
    with pytest.raises(ValueError, match="separately registered background protocol"):
        stage.run(
            f.cache,
            f.output,
            49800,
            348600,
            capture_factory=f.Capture,
            runtime_factory=f.runtime_factory,
        )
    assert not f.output.exists() and not f.runtime_calls and not f.captures


@pytest.mark.parametrize("start,stop", [(0, 0), (-1, 10), (348599, 348601), (True, 10)])
def test_invalid_native_interval_is_rejected_without_inference(fixture, start, stop):
    f = fixture
    with pytest.raises(ValueError, match="native window"):
        stage.run(
            f.cache,
            f.output,
            start,
            stop,
            capture_factory=f.Capture,
            runtime_factory=f.runtime_factory,
        )
    assert not f.output.exists() and not f.runtime_calls


@pytest.mark.parametrize("field,value", [("fps", 166.0001), ("source_count", 348599)])
def test_wrong_native_video_metadata_fails_before_output_or_runtime(fixture, field, value):
    f = fixture
    f.options[field] = value
    with pytest.raises(ValueError):
        stage.run(
            f.cache,
            f.output,
            100,
            102,
            capture_factory=f.Capture,
            runtime_factory=f.runtime_factory,
        )
    assert not f.output.exists() and not f.runtime_calls
    assert all(cap.released for cap in f.captures)


def test_existing_output_is_not_overwritten(fixture):
    f = fixture
    f.output.mkdir()
    (f.output / "sentinel").write_bytes(b"untouched")
    with pytest.raises(FileExistsError, match="no overwrite"):
        stage.run(
            f.cache,
            f.output,
            100,
            102,
            capture_factory=f.Capture,
            runtime_factory=f.runtime_factory,
        )
    assert (f.output / "sentinel").read_bytes() == b"untouched" and not f.runtime_calls


def test_changed_cached_video_fails_before_output_or_predictor(fixture):
    f = fixture
    path = (
        f.cache
        / "medaka_202512"
        / stage.original.EXPERIMENT
        / "top_cam"
        / stage.original.VIDEO_NAMES["top_cam"]
    )
    path.write_bytes(b"changed source bytes")
    with pytest.raises(ValueError, match="differs from the verified cache"):
        stage.run(
            f.cache,
            f.output,
            100,
            102,
            capture_factory=f.Capture,
            runtime_factory=f.runtime_factory,
        )
    assert not f.output.exists() and not f.runtime_calls


def test_runtime_initialization_failure_is_recorded_as_failed(fixture):
    f = fixture

    def failed_runtime():
        raise RuntimeError("test runtime unavailable")

    with pytest.raises(RuntimeError, match="runtime unavailable"):
        stage.run(
            f.cache, f.output, 100, 102, capture_factory=f.Capture, runtime_factory=failed_runtime
        )
    assert not (f.output / "manifest.json").exists()
    assert json.loads((f.output / "staging.provenance.json").read_text())["status"] == "failed"
    assert all(cap.released for cap in f.captures)
