"""Native source offsets, clip ends, original-video metadata and resource ownership."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
SPEC = importlib.util.spec_from_file_location("_source_sequence_test", LAB / "source_sequence.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
from prepare_sequence import PackedMasks  # noqa: E402
from prepare_source_sequence import contained_file, copy_exclusive, preflight  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compact_staging_refuses_existing_directory_without_changing_it(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="no overwrite"):
        preflight(tmp_path / "absent", tmp_path / "absent_cache", output, tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_copy_exclusive_never_replaces_a_prior_file(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        copy_exclusive(source, target)
    assert target.read_bytes() == b"original"


@pytest.mark.parametrize("relative", ["../outside.bin", "C:/elsewhere.bin", "/elsewhere.bin"])
def test_compact_staging_rejects_paths_outside_descriptor(tmp_path, relative):
    with pytest.raises(ValueError, match="relative"):
        contained_file(tmp_path, relative)


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    root, nodes = tmp_path / "source", tmp_path / "nodes"
    root.mkdir()
    node_folder = nodes / "work/c01_s16_predictions/bottomup"
    node_folder.mkdir(parents=True)
    manifest = dict(
        experiment="fixture",
        frames=[100, 101, 102],
        fps=166,
        image_shape_hw=[1200, 1200],
        raw_storage="original_source_video",
        raw={},
        masks={},
        backgrounds={},
    )
    images, options, opened = {}, {}, []
    for vi, view in enumerate(module.VIEWS):
        images[view] = [np.full((1200, 1200, 3), vi * 30 + t, np.uint8) for t in range(3)]
        raw_path = root / f"{view}.mp4"
        raw_path.write_bytes(f"mock native source {view}".encode())
        manifest["raw"][view] = dict(
            path=raw_path.name,
            sha256=digest(raw_path),
            source_offset_frames=100,
            source_total_frames=348600,
            frame_bgr_sha256=[hashlib.sha256(i.tobytes()).hexdigest() for i in images[view]],
        )
        packed = PackedMasks()
        for frame in manifest["frames"]:
            masks = np.zeros((0 if frame == 101 else 1, 1200, 1200), bool)
            if len(masks):
                masks[0, 10, 20] = True
            packed.append(
                frame, masks, np.ones(len(masks)), np.zeros((len(masks), 4)), np.zeros(len(masks))
            )
        arrays = packed.arrays()
        arrays["view_available"][-1] = False
        path = root / f"{view}.npz"
        np.savez_compressed(path, **arrays)
        manifest["masks"][view] = dict(path=path.name, sha256=digest(path))
        path = root / f"{view}.png"
        assert cv2.imwrite(str(path), images[view][0])
        manifest["backgrounds"][view] = dict(path=path.name, sha256=digest(path))
        np.savez_compressed(
            node_folder / f"fixture__{view}.npz",
            view=view,
            experiment="fixture",
            nodes=["mouth", "pectoral", "tail"],
            frame_idx=np.array([100, 100, 102]),
            points=np.arange(18.0).reshape(3, 3, 2),
        )

    class Capture:
        def __init__(self, path):
            self.view = Path(path).stem
            self.position, self.reads, self.released = 0, 0, False
            opened.append(self)

        def isOpened(self):
            return self.view != options.get("unopened")

        def get(self, key):
            return {
                cv2.CAP_PROP_FPS: options.get("fps", 166),
                cv2.CAP_PROP_FRAME_COUNT: options.get("frame_count", 348600),
                cv2.CAP_PROP_FRAME_HEIGHT: 1200,
                cv2.CAP_PROP_FRAME_WIDTH: 1200,
                cv2.CAP_PROP_POS_FRAMES: self.position,
            }[key]

        def set(self, key, value):
            assert key == cv2.CAP_PROP_POS_FRAMES
            self.position = value + options.get("seek_shift", 0)
            return True

        def read(self):
            self.reads += 1
            index = int(self.position) - 100
            self.position += 1
            # The full original video still has valid frames after this window.
            return True, images[self.view][min(max(index, 0), 2)]

        def release(self):
            self.released = True

    monkeypatch.setattr(module.cv2, "VideoCapture", Capture)

    def save():
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    save()
    return root, nodes, manifest, images, options, opened, save


def test_window_stops_without_false_source_eof_and_preserves_nodes_masks(source_fixture):
    root, nodes, _, images, _, opened, _ = source_fixture
    frames = list(module.SourceSequenceReader(root, nodes))
    assert [f.frame_idx for f in frames] == [100, 101, 102]
    for view in module.VIEWS:
        for t, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.images_bgr[view], images[view][t])
        assert frames[0].detections[view].shape == (2, 3, 2)
        assert frames[1].detections[view].shape == (0, 3, 2)
        assert frames[0].masks[view][0, 10, 20]
        assert frames[1].masks[view].shape == (0, 1200, 1200)
        assert frames[2].masks[view] is None
    assert len(opened) == 3
    assert all(c.reads == 3 and c.released for c in opened)


def test_early_close_releases_all_three_source_captures(source_fixture):
    root, nodes, _, _, _, opened, _ = source_fixture
    iterator = iter(module.SourceSequenceReader(root, nodes))
    assert next(iterator).frame_idx == 100
    iterator.close()
    assert all(c.reads == 1 and c.released for c in opened)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("source_offset_frames", 101, "source offset"),
        ("source_offset_frames", True, "source offset"),
        ("source_total_frames", 200, "source_total_frames"),
    ],
)
def test_wrong_declared_source_metadata_is_rejected(source_fixture, field, value, error):
    root, nodes, manifest, _, _, opened, save = source_fixture
    manifest["raw"][module.VIEWS[0]][field] = value
    save()
    with pytest.raises(ValueError, match=error):
        module.SourceSequenceReader(root, nodes)
    assert not opened


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("fps", 165, "fps mismatch"),
        ("frame_count", 200, "frame count mismatch"),
        ("seek_shift", 1, "seek position mismatch"),
    ],
)
def test_wrong_actual_source_metadata_and_seek_are_rejected(source_fixture, field, value, error):
    root, nodes, _, _, options, opened, _ = source_fixture
    options[field] = value
    with pytest.raises(ValueError, match=error):
        list(module.SourceSequenceReader(root, nodes))
    assert opened and all(c.released for c in opened)


def test_bad_native_pixel_hash_is_rejected_and_captures_released(source_fixture):
    root, nodes, manifest, _, _, opened, save = source_fixture
    manifest["raw"][module.VIEWS[1]]["frame_bgr_sha256"][0] = "0" * 64
    save()
    with pytest.raises(ValueError, match="BGR SHA256 mismatch"):
        list(module.SourceSequenceReader(root, nodes))
    assert all(c.released for c in opened)


def test_changed_whole_source_hash_fails_in_reused_base_validation(source_fixture):
    root, nodes, manifest, _, _, opened, save = source_fixture
    manifest["raw"][module.VIEWS[1]]["sha256"] = "0" * 64
    save()
    with pytest.raises(ValueError, match="changed raw"):
        module.SourceSequenceReader(root, nodes)
    assert not opened


def test_later_camera_open_failure_releases_earlier_handles(source_fixture):
    root, nodes, _, _, options, opened, _ = source_fixture
    options["unopened"] = module.VIEWS[1]
    with pytest.raises(RuntimeError, match="cannot open"):
        list(module.SourceSequenceReader(root, nodes))
    assert len(opened) == 2 and all(c.released for c in opened)
