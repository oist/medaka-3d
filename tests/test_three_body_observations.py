"""Independent input checks: native frame association and honest missing evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

LAB = Path(__file__).parents[1] / "labs/20260909_three_body"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stage = _load("_three_body_stage_for_observation_tests", LAB / "prepare_sequence.py")
_previous_stage = sys.modules.get("prepare_sequence")
sys.modules["prepare_sequence"] = stage
try:
    obs = _load("_three_body_observations_tests", LAB / "observations.py")
finally:
    if _previous_stage is None:
        del sys.modules["prepare_sequence"]
    else:
        sys.modules["prepare_sequence"] = _previous_stage


def _write_nodes(root, frames=(99, 100, 100, 102, 103)):
    folder = root / "work/c01_s16_predictions/bottomup"
    folder.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(obs.VIEWS):
        points = np.arange(len(frames) * 6, dtype=float).reshape(-1, 3, 2) + index * 100
        np.savez_compressed(
            folder / f"experiment__{view}.npz",
            view=view,
            experiment="experiment",
            nodes=["mouth", "pectoral", "tail"],
            frame_idx=np.asarray(frames, dtype=np.int64),
            points=points,
            source_track_id=np.array([987] * len(frames)),
        )


def _replace_npz(path, **fields):
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    arrays.update(fields)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def staged(tmp_path):
    root, nodes = tmp_path / "raw", tmp_path / "nodes"
    root.mkdir()
    _write_nodes(nodes)
    manifest = {
        "experiment": "experiment",
        "frames": [100, 101, 102],
        "fps": 166.0,
        "image_shape_hw": [1200, 1200],
        "raw": {},
        "masks": {},
        "backgrounds": {},
    }
    expected = {}
    for index, view in enumerate(obs.VIEWS):
        path = root / f"{view}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 166.0, (1200, 1200))
        assert writer.isOpened()
        images = []
        hashes = []
        packed = stage.PackedMasks()
        for t, frame in enumerate(manifest["frames"]):
            image = np.full((1200, 1200, 3), index * 50 + t * 3 + 25, np.uint8)
            image[10 + t : 20 + t, 30:40] = (3, 200, 17)
            images.append(image)
            hashes.append(stage.array_sha256(image))
            writer.write(image)
            count = (4, 0, 1)[t]
            masks = np.zeros((count, 1200, 1200), bool)
            for n in range(count):
                masks[n, 20 + n : 23 + n, 100 + n * 5 : 105 + n * 5] = True
            packed.append(frame, masks, np.full(count, 0.9), np.zeros((count, 4)), np.zeros(count))
        writer.release()
        expected[view] = images
        manifest["raw"][view] = {
            "path": path.name,
            "sha256": obs.sha256(path),
            "frame_bgr_sha256": hashes,
        }
        arrays = packed.arrays()
        arrays["view_available"][-1] = False
        path = root / f"{view}.npz"
        np.savez_compressed(path, **arrays)
        manifest["masks"][view] = {"path": path.name, "sha256": obs.sha256(path)}
        path = root / f"{view}.png"
        assert cv2.imwrite(str(path), np.zeros((1200, 1200, 3), np.uint8))
        manifest["backgrounds"][view] = {"path": path.name, "sha256": obs.sha256(path)}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, nodes, manifest, expected


def test_nodes_use_source_frame_not_row_id_and_preserve_empty_frames(tmp_path):
    _write_nodes(tmp_path)
    cache = obs.CachedNodes(tmp_path, "experiment", [100, 101, 102])
    for index, view in enumerate(obs.VIEWS):
        assert cache.at(view, 100).shape == (2, 3, 2)
        np.testing.assert_array_equal(
            cache.at(view, 100)[0], np.arange(6).reshape(3, 2) + 6 + 100 * index
        )
        assert cache.at(view, 101).shape == (0, 3, 2)
        assert cache.at(view, 102).shape == (1, 3, 2)
    with pytest.raises(IndexError, match="outside"):
        cache.at(obs.VIEWS[0], 103)


def test_nodes_reject_wrong_identity_order_unsorted_and_noncontiguous_frames(tmp_path):
    _write_nodes(tmp_path)
    path = tmp_path / "work/c01_s16_predictions/bottomup/experiment__top_cam.npz"
    with pytest.raises(ValueError, match="contiguous"):
        obs.CachedNodes(tmp_path, "experiment", [100, 102])
    _replace_npz(path, view="side2_cam")
    with pytest.raises(ValueError, match="identity"):
        obs.CachedNodes(tmp_path, "experiment", [100, 101, 102])
    _write_nodes(tmp_path)
    _replace_npz(path, nodes=["tail", "pectoral", "mouth"])
    with pytest.raises(ValueError, match="node order"):
        obs.CachedNodes(tmp_path, "experiment", [100, 101, 102])
    _write_nodes(tmp_path)
    _replace_npz(path, frame_idx=np.array([99, 100, 103, 102, 104]))
    with pytest.raises(ValueError, match="unsorted"):
        obs.CachedNodes(tmp_path, "experiment", [100, 101, 102])


def test_sequence_native_bgr_source_frames_all_instances_and_availability(staged):
    root, nodes, _, expected = staged
    reader = obs.SequenceReader(root, nodes)
    frames = list(reader)
    assert [frame.frame_idx for frame in frames] == [100, 101, 102]
    for view in obs.VIEWS:
        for t, frame in enumerate(frames):
            np.testing.assert_array_equal(frame.images_bgr[view], expected[view][t])
        assert frames[0].masks[view].shape == (4, 1200, 1200)
        assert frames[0].masks[view][3, 23, 115]
        assert frames[1].masks[view].shape == (0, 1200, 1200)
        assert frames[2].masks[view] is None
    # Repeat iteration starts at clip zero, preserving source-frame offset again.
    again = list(reader)
    np.testing.assert_array_equal(again[0].images_bgr[obs.VIEWS[0]], expected[obs.VIEWS[0]][0])


def test_sequence_rejects_changed_raw_hash_and_changed_mask_frame_list(staged):
    root, nodes, manifest, _ = staged
    manifest["raw"]["top_cam"]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="changed raw"):
        obs.SequenceReader(root, nodes)
    manifest["raw"]["top_cam"]["sha256"] = obs.sha256(root / "top_cam.avi")
    path = root / "top_cam.npz"
    _replace_npz(path, frame_idx=np.array([101, 102, 103]))
    manifest["masks"]["top_cam"]["sha256"] = obs.sha256(path)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mask frame mismatch"):
        obs.SequenceReader(root, nodes)


def _frame(masks, images=None, frame_idx=333619):
    shape = (48, 64, 3)
    zeros = {view: np.zeros(shape, np.uint8) for view in obs.VIEWS}
    return obs.FrameObservation(
        frame_idx,
        {view: np.zeros((0, 3, 2)) for view in obs.VIEWS},
        images if images is not None else zeros,
        dict(zip(obs.VIEWS, masks, strict=True)),
        {view: value.copy() for view, value in zeros.items()},
    )


def test_missing_empty_and_all_false_masks_are_neutral_without_losing_availability():
    frame = _frame([None, np.zeros((0, 48, 64), bool), np.zeros((2, 48, 64), bool)])
    result = obs.sample_pixels(frame)
    assert not result.weights.any()
    assert not result.target.any()
    assert frame.masks["top_cam"] is None
    assert frame.masks["side1_cam"].shape[0] == 0
    assert frame.masks["side2_cam"].shape[0] == 2
    assert all(np.isfinite(xy).all() for xy in result.pixels.values())


def test_fixed_sampling_uses_native_xy_balances_and_is_independent_of_instance_order():
    masks = np.zeros((4, 48, 64), bool)
    for n in range(4):
        masks[n, 10 + n : 14 + n, 20 + n * 4 : 23 + n * 4] = True
    frame = _frame([masks, masks, masks])
    first = obs.sample_pixels(frame, n_positive=23, n_near=13, n_global=17)
    shuffled = obs.sample_pixels(
        _frame([masks[::-1], masks[::-1], masks[::-1]]), n_positive=23, n_near=13, n_global=17
    )
    union = masks.any(axis=0)
    for v, view in enumerate(obs.VIEWS):
        xy = first.pixels[view].astype(int)
        assert xy.shape == (53, 2)
        np.testing.assert_array_equal(first.target[v], union[xy[:, 1], xy[:, 0]])
        np.testing.assert_array_equal(first.pixels[view], shuffled.pixels[view])
        np.testing.assert_array_equal(first.foreground[v], first.target[v].astype(bool))
        assert first.weights[v, :23].sum() == pytest.approx(1)
        assert first.weights[v, 23:36].sum() == pytest.approx(0.75)
        assert first.weights[v, 36:].sum() == pytest.approx(0.25)
    np.testing.assert_array_equal(first.target, shuffled.target)
    np.testing.assert_array_equal(first.weights, shuffled.weights)


def test_top_contrast_is_raw_evidence_even_when_top_detector_is_unavailable():
    images = {view: np.full((48, 64, 3), 7, np.uint8) for view in obs.VIEWS}
    images["top_cam"][14:20, 28:35] = 70
    frame = _frame([None, None, None], images)
    assert not obs.sample_pixels(frame).weights.any()
    contrast = obs.sample_pixels(frame, mode="top_contrast")
    assert contrast.weights[0].sum() == pytest.approx(2)
    assert not contrast.weights[1:].any()
    assert frame.masks["top_cam"] is None
    with pytest.raises(ValueError, match="mode"):
        obs.sample_pixels(frame, mode="unknown")


def test_contrast_rejects_global_illumination_and_isolated_pixels():
    background = np.full((48, 64, 3), 45, np.uint8)
    image = background + 20
    assert not obs.foreground_contrast(image, background).any()
    image[6, 8] = 140
    assert not obs.foreground_contrast(image, background).any()
    image[10:13, 25:29] = 140
    result = obs.foreground_contrast(image, background)
    assert result.sum() == 12
    assert result[10:13, 25:29].all()


def test_all_foreground_has_positive_support_and_no_fake_negative_weight():
    full = np.ones((1, 48, 64), bool)
    result = obs.sample_pixels(_frame([full, full, full]), n_positive=4, n_near=3, n_global=2)
    assert np.all(result.target[:, :4] == 1)
    assert np.all(result.weights[:, 4:] == 0)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1)


@pytest.mark.parametrize("name", ["n_positive", "n_near", "n_global"])
@pytest.mark.parametrize("value", [0, -1, 1.0, True, np.bool_(True), np.nan])
def test_sample_counts_must_be_strictly_positive_integers(name, value):
    frame = _frame([None, None, None])
    with pytest.raises(ValueError, match=f"{name} must be a strictly positive integer"):
        obs.sample_pixels(frame, **{name: value})


def test_numpy_integer_sample_counts_are_valid():
    result = obs.sample_pixels(
        _frame([None, None, None]), n_positive=np.int64(3), n_near=np.int32(2), n_global=1
    )
    assert result.target.shape == (3, 6)
    assert not result.weights.any()


@pytest.mark.parametrize("change", ["fps", "image_shape_hw", "missing_hashes", "short_hashes"])
def test_reader_requires_native_declaration_and_every_decoded_frame_hash(staged, change):
    root, nodes, manifest, _ = staged
    if change in ("fps", "image_shape_hw"):
        del manifest[change]
        message = "staged fps/image shape"
    else:
        hashes = manifest["raw"]["top_cam"]["frame_bgr_sha256"]
        if change == "missing_hashes":
            del manifest["raw"]["top_cam"]["frame_bgr_sha256"]
        else:
            hashes.pop()
        message = "per-frame BGR SHA256"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        obs.SequenceReader(root, nodes)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("fps", "raw fps mismatch"),
        ("frame_count", "raw frame count mismatch"),
        ("metadata_shape", "raw image shape metadata mismatch"),
        ("shape", "decoded raw image shape/dtype mismatch"),
        ("dtype", "decoded raw image shape/dtype mismatch"),
        ("pixel", "decoded raw BGR SHA256 mismatch"),
        ("position", "raw frame alignment failed"),
        ("trailing_frame", "undeclared extra frames"),
    ],
)
def test_reader_rejects_decoder_changes_and_releases_all_captures(
    staged, monkeypatch, change, message
):
    root, nodes, _, _ = staged
    reader = obs.SequenceReader(root, nodes)
    original = cv2.VideoCapture
    opened = []

    class ChangedDecoder:
        def __init__(self, path):
            self.capture = original(path)
            self.closed = False
            opened.append(self)

        def isOpened(self):
            return self.capture.isOpened()

        def get(self, prop):
            if change == "fps" and prop == cv2.CAP_PROP_FPS:
                return 30.0
            if change == "frame_count" and prop == cv2.CAP_PROP_FRAME_COUNT:
                return 4
            if change == "metadata_shape" and prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 600
            if change == "position" and prop == cv2.CAP_PROP_POS_FRAMES:
                return float("nan")
            return self.capture.get(prop)

        def read(self):
            ok, image = self.capture.read()
            if not ok:
                if change == "trailing_frame":
                    return True, np.zeros((1200, 1200, 3), np.uint8)
                return ok, image
            if change == "shape":
                image = image[:600]
            elif change == "dtype":
                image = image.astype(np.uint16)
            elif change == "pixel":
                image[0, 0, 0] ^= 1
            return ok, image

        def release(self):
            self.capture.release()
            self.closed = True

    monkeypatch.setattr(obs.cv2, "VideoCapture", ChangedDecoder)
    error = RuntimeError if change == "position" else ValueError
    with pytest.raises(error, match=message):
        list(reader)
    assert len(opened) == 3
    assert all(capture.closed for capture in opened)
