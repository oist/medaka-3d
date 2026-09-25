"""Identity truth must stay human-authored and independent of staged predictions."""

import copy
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "labs/20260909_three_body/review_truth.py"
_SPEC = importlib.util.spec_from_file_location("three_body_review_truth", _PATH)
assert _SPEC and _SPEC.loader
review = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review)


def test_template_never_promotes_predictions_to_truth():
    truth = review.make_template(list(range(10, 20)), "abc", [10, 15, 19])
    assert truth["status"] == "unreviewed_template_not_ground_truth"
    assert truth["model_outputs_shown"] is False
    assert all(v["physical_name"] is None for v in truth["biological_naming"].values())
    assert all(not a["points"] for row in truth["anchors"] for a in row["animals"].values())
    assert review.validate_review(truth) == ["reviewer is missing"]


def test_confirmation_requires_visible_marks_and_visual_evidence():
    truth = review.make_template(list(range(10, 20)), "abc", [10, 15, 19])
    truth["reviewer"] = "independent_human"
    truth["continuity"]["A"]["entry_to_after"] = "confirmed"
    assert len(review.validate_review(truth)) == 3
    truth["continuity"]["A"]["basis"] = "direct_visual_continuity"
    for anchor in truth["anchors"][:2]:
        anchor["animals"]["A"] = {"points": {"top_cam": [110.5, 600]}, "visibility": "visible"}
    assert review.validate_review(truth) == []
    truth["anchors"][0]["animals"]["A"]["points"]["top_cam"] = [float("nan"), 600]
    assert any("invalid native pixel mark" in x for x in review.validate_review(truth))


def test_rejects_changed_anchor_order_and_escaping_raw_paths(tmp_path):
    with pytest.raises(ValueError, match="chronological"):
        review.make_template(list(range(10, 20)), "abc", [10, 19, 15])
    with pytest.raises(ValueError, match="consecutive"):
        review.make_template([10, 12, 15], "abc", [10, 12, 15])
    with pytest.raises(ValueError, match="inside"):
        review.relative_file(tmp_path, "../outside.avi")


def test_pack_roundtrip_blindness_pixels_and_no_overwrite(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = {"frames": [10, 11, 12], "raw": {}, "masks": "DO_NOT_IMPORT_MODEL_IDENTITIES"}
    for index, view in enumerate(review.VIEWS):
        path = source / f"{view}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 166, (1200, 1200))
        assert writer.isOpened()
        hashes = []
        for frame in range(3):
            bgr = np.full((1200, 1200, 3), 30 + index * 40 + frame, dtype=np.uint8)
            bgr[12:24, 15:28] = (23, 142, 242)
            hashes.append(review.hashlib.sha256(bgr.tobytes()).hexdigest())
            writer.write(bgr)
        writer.release()
        manifest["raw"][view] = {
            "path": path.name,
            "sha256": review.sha256(path),
            "frame_bgr_sha256": hashes,
        }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "review"
    result = review.build_pack(manifest_path, out, [10, 11, 12])
    assert result["native_pixel_hashes_verified"] == 9
    assert result["review_video"]["decoded_frame_count"] == 3
    assert not result["human_truth_created"]
    assert "DO_NOT_IMPORT_MODEL_IDENTITIES" not in (out / "index.html").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        review.build_pack(manifest_path, out, [10, 11, 12])
    poisoned = copy.deepcopy(manifest)
    poisoned["raw"]["top_cam"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(poisoned), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        review.build_pack(manifest_path, tmp_path / "invalid", [10, 11, 12])
    assert not (tmp_path / "invalid").exists()
