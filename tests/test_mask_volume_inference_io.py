"""Fresh D2 experiment I/O without requiring Detectron2 in the project env."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "labs/20260908_mask_volume/infer_masks.py"
SPEC = importlib.util.spec_from_file_location("_mask_inference_io_test", MODULE)
infer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(infer)


def test_packing_preserves_zero_and_four_proposals_without_ids():
    shape = (6, 7)
    empty = {
        "masks": np.zeros((0, *shape), bool),
        "scores": np.zeros(0),
        "pred_boxes": np.zeros((0, 4)),
        "classes": np.zeros(0, int),
    }
    masks = np.zeros((4, *shape), bool)
    masks[:, 1:3, 2:5] = True
    four = {
        "masks": masks,
        "scores": np.array([0.99, 0.8, 0.7, 0.51], np.float32),
        "pred_boxes": np.tile([1.9, 0.9, 5.1, 3.1], (4, 1)),
        "classes": np.zeros(4, int),
    }
    data = infer.pack_predictions([10, 20], [empty, four], shape)
    np.testing.assert_array_equal(data["frame_offsets"], [0, 0, 4])
    assert data["view_available"].all()
    assert np.all(data["source_track_id"] == -1)
    np.testing.assert_array_equal(data["boxes"], np.tile([2, 1, 5, 3], (4, 1)))
    np.testing.assert_allclose(data["pred_boxes"], four["pred_boxes"])
    np.testing.assert_array_equal(data["masks"], masks)


def test_bad_prediction_shapes_fail_loudly():
    malformed = {
        "masks": np.zeros((1, 6, 7), bool),
        "scores": np.array([np.nan]),
        "pred_boxes": np.zeros((1, 4)),
        "classes": np.zeros(1, int),
    }
    with pytest.raises(ValueError, match="invalid score"):
        infer.pack_predictions([0], [malformed], (6, 7))
    with pytest.raises(ValueError, match="frame count"):
        infer.pack_predictions([0, 1], [malformed], (6, 7))


def test_raw_png_checksum_and_native_bgr_values(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = np.full((1200, 1200, 3), [10, 20, 30], np.uint8)
    manifest = {"frames": [42], "outputs": {}}
    for view in infer.VIEWS:
        path = raw / f"f000042_{view}.png"
        assert cv2.imwrite(str(path), image)
        manifest["outputs"][path.name] = {"sha256": infer.sha256(path)}
    (raw / "manifest.json").write_text(json.dumps(manifest))
    _, images, hashes = infer.read_raw(raw)
    assert len(hashes) == 3
    np.testing.assert_array_equal(images["top_cam"][0], image)
    (raw / "f000042_top_cam.png").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum mismatch"):
        infer.read_raw(raw)


def test_asset_hash_is_verified_before_loading_and_output_is_not_overwritten(tmp_path):
    source = tmp_path / "assets_root"
    (source / "assets").mkdir(parents=True)
    manifest = {"assets": {}}
    for kind, relative in infer.MODEL_DIRS.items():
        directory = source / relative
        directory.mkdir(parents=True)
        for suffix, filename in (("", "model_final.pth"), (".config", "detectron2_config.yaml")):
            path = directory / filename
            path.write_bytes(b"fixture")
            manifest["assets"][f"detectron2.{kind}{suffix}"] = {
                "size": path.stat().st_size,
                "sha256": infer.sha256(path),
                "path": str(path),
            }
    (source / "assets/MANIFEST.json").write_text(json.dumps(manifest))
    verified, _ = infer.verify_assets(source)
    assert len(verified) == 4
    (source / infer.MODEL_DIRS["top"] / "model_final.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        infer.verify_assets(source)
    out = tmp_path / "keep"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        infer.run(tmp_path / "missing_raw", source, out)
    assert sentinel.read_bytes() == b"keep"
