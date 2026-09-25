"""A bounded raw review must retain the full manifest/range binding for evaluation."""

import copy
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
SPEC = importlib.util.spec_from_file_location("_review_scene_test", LAB / "review_scene.py")
scene = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scene
SPEC.loader.exec_module(scene)
from evaluate_review import validate_inputs  # noqa: E402


@pytest.mark.parametrize(
    "start,stop,probes",
    [
        (99, 104, [100, 101, 103]),
        (103, 111, [103, 104, 105]),
        (103, 105, [103, 104, 105]),
        (103, 106, [103, 104, 106]),
        (103, 106, [103, 103, 105]),
        (103, 106, [105, 104, 103]),
    ],
)
def test_ranges_and_probes_are_rejected(start, stop, probes):
    with pytest.raises(ValueError):
        scene.select_scene(list(range(100, 110)), start, stop, probes)


def test_template_is_bound_to_full_source_while_html_uses_only_subset():
    frames = list(range(100, 110))
    template, subset, offset = scene.bind_template(frames, "abc", 103, 106, [103, 104, 105])
    assert template["source_frame_range"] == [100, 110]
    assert template["review_clip_range"] == [103, 106]
    assert subset == [103, 104, 105] and offset == 3
    html_source = (LAB / "review_scene.html").read_text(encoding="utf-8")
    html = scene.render_html(html_source, template, subset)
    assert "const frames=[103, 104, 105]" in html
    assert 'max="2"' in html and "__FRAMES__" not in html
    assert template["status"] == "unreviewed_template_not_ground_truth"
    assert "reviewImportIssue(incoming,truth,frames)" in html
    assert "incoming.review_clip_range,expected.review_clip_range" in html
    assert scene.sha256(LAB / "review_truth.html") == scene.HTML_ORIGIN["sha256"]
    # A synthetic draft only exercises the existing structural binding contract;
    # no marks/continuity are invented or exported as actual human truth.
    draft = copy.deepcopy(template)
    draft["status"], draft["reviewer"] = "human_draft_requires_adjudication", "test fixture"
    sequence = dict(
        frames=frames, fps=166, image_shape_hw=[1200, 1200], experiment=template["experiment"]
    )
    full, anchors, _, _ = validate_inputs(draft, sequence, "abc")
    assert full.tolist() == frames and anchors.tolist() == subset
    draft["source_frame_range"] = [103, 106]
    with pytest.raises(ValueError, match="source frame range mismatch"):
        validate_inputs(draft, sequence, "abc")


@pytest.fixture
def raw_sequence(tmp_path):
    root = tmp_path / "sequence"
    root.mkdir()
    frames = list(range(100, 108))
    manifest = dict(
        frames=frames,
        raw={},
        fps=166,
        image_shape_hw=[1200, 1200],
        experiment="20251225_K_1_2_FCMC",
        raw_codec="FFV1 AVI",
        masks="MUST_NOT_OPEN_OR_DISPLAY_MODEL_MASKS",
    )
    for vi, view in enumerate(scene.VIEWS):
        path = root / f"{view}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 166, (1200, 1200))
        assert writer.isOpened()
        hashes = []
        for frame in frames:
            image = np.full((1200, 1200, 3), frame + vi * 10, np.uint8)
            image[7:15, 19:27] = [15, 42, 91]
            hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
            writer.write(image)
        writer.release()
        manifest["raw"][view] = dict(
            path=path.name, sha256=scene.sha256(path), frame_bgr_sha256=hashes
        )

    def seal():
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        outputs = {
            f"{view}_raw": dict(sha256=info["sha256"]) for view, info in manifest["raw"].items()
        }
        outputs["manifest"] = dict(sha256=scene.sha256(path))
        (root / "staging.provenance.json").write_text(
            json.dumps(dict(status="ok", outputs=outputs))
        )

    seal()
    return root, manifest, seal


def test_real_subset_first_offset_video_count_and_output_hashes(raw_sequence, tmp_path):
    root, _, _ = raw_sequence
    out = tmp_path / "review"
    result = scene.build_pack(root, 103, 106, [103, 104, 105], out)
    assert result["first_source_clip_offset"] == 3
    assert result["source_frame_range"] == [100, 108]
    assert result["review_clip_range"] == [103, 106]
    assert result["review_video"]["decoded_frame_count"] == 3
    assert result["native_pixel_hashes_verified"] == 9
    assert result["browser_used"] is False and result["ui_operations_verified"] is False
    for vi, view in enumerate(scene.VIEWS):
        image = cv2.imread(str(out / f"f103_{view}.png"))
        assert image[100, 100, 0] == 103 + vi * 10
    assert "MUST_NOT_OPEN_OR_DISPLAY_MODEL_MASKS" not in (out / "index.html").read_text(
        encoding="utf-8"
    )
    assert len(list((out / "frames").glob("*.jpg"))) == 9
    for path, digest in result["output_sha256"].items():
        assert scene.sha256(out / path) == digest
    assert json.loads((out / "review_scene.provenance.json").read_text())["status"] == "ok"
    with pytest.raises(FileExistsError, match="no overwrite"):
        scene.build_pack(root, 103, 106, [103, 104, 105], out)


def test_bad_subset_pixel_hash_fails_without_successful_pack(raw_sequence, tmp_path):
    root, manifest, seal = raw_sequence
    manifest["raw"]["side1_cam"]["frame_bgr_sha256"][3] = "0" * 64
    seal()
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="native BGR SHA256 mismatch"):
        scene.build_pack(root, 103, 106, [103, 104, 105], out)
    assert not (out / "summary.json").exists()
    assert not (out / "truth_template.json").exists()
    assert json.loads((out / "review_scene.provenance.json").read_text())["status"] == "failed"


def test_unsealed_input_fails_before_output_creation(raw_sequence, tmp_path):
    root, _, _ = raw_sequence
    path = root / "staging.provenance.json"
    seal = json.loads(path.read_text())
    seal["status"] = "running"
    path.write_text(json.dumps(seal))
    out = tmp_path / "unstarted"
    with pytest.raises(ValueError, match="not successfully sealed"):
        scene.build_pack(root, 103, 106, [103, 104, 105], out)
    assert not out.exists()
