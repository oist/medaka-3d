"""A resumed native stream must not confuse source indices with clip positions."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
import resumable_observations as module  # noqa: E402
from prepare_sequence import PackedMasks  # noqa: E402


@pytest.fixture
def fixture(monkeypatch):
    readers, options, opened = {}, {}, []
    frames = np.arange(100, 104, dtype=np.int64)
    images = [np.full((1200, 1200, 3), t, np.uint8) for t in range(4)]
    hashes = tuple(hashlib.sha256(image.tobytes()).hexdigest() for image in images)
    packed = PackedMasks()
    for t, frame in enumerate(frames):
        masks = np.zeros((1 if t in (0, 3) else 0, 1200, 1200), bool)
        if len(masks):
            masks[0, 10 + t, 20] = True
        packed.append(
            int(frame), masks, np.ones(len(masks)), np.zeros((len(masks), 4)),
            np.zeros(len(masks)),
        )
    packed = packed.arrays()
    packed["view_available"][2] = False
    for source, reader_type in (
        (False, module.SequenceReader), (True, module.SourceSequenceReader)
    ):
        reader = reader_type.__new__(reader_type)
        reader.frames = frames.copy()
        reader.fps, reader.image_shape = 166.0, (1200, 1200)
        reader.manifest = dict(frames=frames.tolist(), raw={})
        if source:
            reader.manifest["raw_storage"] = "original_source_video"
            reader.source_offsets = dict.fromkeys(module.VIEWS, 100)
            reader.manifest["raw"] = {
                view: dict(source_offset_frames=100, source_total_frames=348600)
                for view in module.VIEWS
            }
        reader.nodes = SimpleNamespace(
            frames=frames.copy(),
            at=lambda view, frame: np.full((1, 3, 2), frame * 10 + module.VIEWS.index(view)),
        )
        reader.videos = {v: Path(f"{int(source)}/{v}") for v in module.VIEWS}
        reader.masks = {
            v: {key: value.copy() for key, value in packed.items()} for v in module.VIEWS
        }
        reader.pixel_hashes = dict.fromkeys(module.VIEWS, hashes)
        reader.backgrounds = dict.fromkeys(module.VIEWS, images[0])
        readers[source] = reader

    class Capture:
        def __init__(self, path):
            path = Path(path)
            self.source = bool(int(path.parent.name))
            self.view = path.name
            if self.view == options.get("constructor_error"):
                raise RuntimeError("fake constructor failed")
            self.position, self.reads, self.released = 0, [], False
            self.seeks = []
            opened.append(self)

        def isOpened(self):
            return self.view != options.get("unopened")

        def get(self, key):
            return {
                cv2.CAP_PROP_FPS: options.get("fps", 166),
                cv2.CAP_PROP_FRAME_COUNT: options.get("count", 348600 if self.source else 4),
                cv2.CAP_PROP_FRAME_HEIGHT: 1200,
                cv2.CAP_PROP_FRAME_WIDTH: options.get("width", 1200),
                cv2.CAP_PROP_POS_FRAMES: self.position,
            }[key]

        def set(self, key, value):
            assert key == cv2.CAP_PROP_POS_FRAMES
            self.seeks.append(value)
            self.position = value + options.get("seek_shift", 0)
            return not options.get("seek_failure", False)

        def read(self):
            self.reads.append(self.position)
            t = int(self.position) - (100 if self.source else 0)
            self.position += 1 + options.get("read_shift", 0)
            if options.get("read_failure"):
                return False, None
            if t >= 4:
                return self.source or options.get("extra", False), images[-1]
            if not 0 <= t < 4:
                raise AssertionError("adapter used the wrong clip/source coordinate system")
            image = images[t]
            if options.get("corrupt_pixels"):
                image = images[(t + 1) % 4]
            if options.get("bad_dtype"):
                image = image.astype(np.float32)
            return True, image

        def release(self):
            self.released = True

    monkeypatch.setattr(module.cv2, "VideoCapture", Capture)
    return readers, options, opened, images


@pytest.mark.parametrize("source", [False, True])
def test_resume_seeks_once_and_uses_full_native_mask_pixel_node_indices(fixture, source):
    readers, _, opened, images = fixture
    reader = readers[source]
    retained = (reader.manifest, reader.frames, reader.nodes, reader.masks, reader.backgrounds)
    observations = list(module.iter_from(reader, np.int64(101)))
    assert [frame.frame_idx for frame in observations] == [101, 102, 103]
    assert all(a is b for a, b in zip(
        retained, (reader.manifest, reader.frames, reader.nodes, reader.masks, reader.backgrounds),
        strict=True,
    ))
    for v, view in enumerate(module.VIEWS):
        for t, observation in enumerate(observations, 1):
            assert observation.images_bgr[view] is images[t]
            assert np.all(observation.detections[view] == (100 + t) * 10 + v)
            assert observation.backgrounds_bgr is reader.backgrounds
        assert observations[0].masks[view].shape == (0, 1200, 1200)
        assert observations[1].masks[view] is None
        assert observations[2].masks[view][0, 13, 20]
    assert len(opened) == 3
    for capture in opened:
        assert capture.seeks == [101 if source else 1]
        assert capture.reads == ([101, 102, 103] if source else [1, 2, 3, 4])
        assert capture.released


@pytest.mark.parametrize("source", [False, True])
def test_close_after_one_yield_reads_no_future_or_previous_frame(fixture, source):
    readers, _, opened, _ = fixture
    iterator = module.iter_from(readers[source], 102)
    assert next(iterator).frame_idx == 102
    iterator.close()
    assert len(opened) == 3
    assert all(c.released and c.reads == [102 if source else 2] for c in opened)


@pytest.mark.parametrize("source", [False, True])
def test_exclusive_end_is_empty_without_opening_any_video(fixture, source):
    readers, _, opened, _ = fixture
    assert list(module.iter_from(readers[source], 104)) == []
    assert not opened


@pytest.mark.parametrize("frame,error", [(99, ValueError), (105, ValueError), (True, TypeError),
                                         (101.0, TypeError)])
def test_invalid_resume_boundaries_are_rejected_before_open(fixture, frame, error):
    readers, _, opened, _ = fixture
    with pytest.raises(error):
        list(module.iter_from(readers[False], frame))
    assert not opened


@pytest.mark.parametrize("kind", ["gap", "manifest", "nodes", "mask", "source_offset", "storage"])
def test_changed_full_reader_binding_is_rejected_before_open(fixture, kind):
    readers, _, opened, _ = fixture
    reader = readers[True]
    if kind == "gap":
        reader.frames[2] += 1
    elif kind == "manifest":
        reader.manifest["frames"][0] -= 1
    elif kind == "nodes":
        reader.nodes.frames[0] -= 1
    elif kind == "mask":
        reader.masks[module.VIEWS[0]]["frame_idx"][1] -= 1
    elif kind == "source_offset":
        reader.source_offsets[module.VIEWS[0]] += 1
    else:
        reader.manifest.pop("raw_storage")
    with pytest.raises(ValueError):
        list(module.iter_from(reader, 101))
    assert not opened


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("option,value,error", [
    ("fps", 165, "fps mismatch"), ("count", 200, "frame count mismatch"),
    ("width", 1199, "dimensions mismatch"), ("seek_failure", True, "seek failed"),
    ("seek_shift", 1, "seek position mismatch"), ("read_shift", 1, "alignment failed"),
    ("read_failure", True, "alignment failed"), ("corrupt_pixels", True, "BGR SHA256"),
    ("bad_dtype", True, "shape/dtype"),
])
def test_metadata_seek_and_frame_failure_releases_every_open_capture(fixture, source,
                                                                   option, value, error):
    readers, options, opened, _ = fixture
    options[option] = value
    with pytest.raises((ValueError, RuntimeError), match=error):
        list(module.iter_from(readers[source], 101))
    assert opened and all(c.released for c in opened)


@pytest.mark.parametrize("option", ["unopened", "constructor_error"])
def test_later_camera_open_failure_releases_prior_captures(fixture, option):
    readers, options, opened, _ = fixture
    options[option] = module.VIEWS[2]
    with pytest.raises(RuntimeError):
        list(module.iter_from(readers[False], 101))
    assert len(opened) == (3 if option == "unopened" else 2)
    assert all(c.released for c in opened)


def test_clip_must_end_but_original_source_may_continue(fixture):
    readers, options, opened, _ = fixture
    options["extra"] = True
    with pytest.raises(ValueError, match="undeclared extra frames"):
        list(module.iter_from(readers[False], 103))
    assert all(c.released for c in opened)
    opened.clear()
    assert len(list(module.iter_from(readers[True], 103))) == 1
    assert all(c.reads == [103] and c.released for c in opened)
