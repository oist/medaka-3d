"""Small synthetic behavior and binding checks; no scientific input loading."""

import io

import cv2
import numpy as np
import pytest
import run_saved_top_foreground_contrast as diagnostic


def test_frozen_threshold_area_and_global_illumination():
    background = np.full((30, 30, 3), 80, np.uint8)
    image = background + 15  # Global shift is removed before thresholding.
    image[2:4, 2:5] += 12  # Six pixels retained at exact threshold.
    image[10, 2:7] += 30  # Five pixels rejected.
    image[18:20, 2:6] += 11  # Eight below-threshold pixels rejected.
    expected = np.zeros((30, 30), bool)
    expected[2:4, 2:5] = True
    before = image.copy()
    assert np.array_equal(diagnostic.foreground_contrast(image, background), expected)
    assert np.array_equal(image, before)


def test_disagreement_overlapping_instances_not_double_counted():
    fg = np.zeros((8, 8), bool)
    fg[1:3, 1:4] = True
    masks = np.zeros((2, 8, 8), bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 1:3, 2:3] = True
    masks[1, 5, 5] = True
    result, union = diagnostic.disagreement(fg, masks)
    assert result["component_areas_descending"] == [6]
    assert result["D2_union_pixels"] == 5
    assert result["disagreement"] == dict(
        foreground_inside_D2_union=4,
        foreground_outside_D2_union=2,
        D2_union_outside_foreground=1,
        union_IoU=4 / 7,
        per_D2_instance_foreground_intersection=[4, 2],
    )
    assert union.sum() == 5


def test_empty_and_unavailable_are_distinct():
    fg = np.zeros((5, 5), bool)
    unavailable, union = diagnostic.disagreement(fg, None)
    empty, _ = diagnostic.disagreement(fg, np.empty((0, 5, 5), bool))
    assert unavailable["disagreement"] is None and union is None
    assert empty["D2_available"] and empty["D2_instance_count"] == 0
    assert empty["disagreement"]["union_IoU"] is None


def test_evidence_refuses_tampered_bytes_and_final_stat_change(tmp_path):
    path = tmp_path / "test"
    path.write_bytes(b"first")
    evidence = diagnostic.Evidence()
    with pytest.raises(ValueError, match="SHA"):
        evidence.capture(path, "0" * 64)
    assert evidence.capture(path, diagnostic.sha(b"first")) == b"first"
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed"):
        evidence.final_guard()


def test_native_png_shape_and_pixel_sha():
    image = np.zeros((1200, 1200, 3), np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    decoded = diagnostic.decode_png(encoded.tobytes(), diagnostic.sha(image.tobytes()))
    assert np.array_equal(decoded, image)
    with pytest.raises(ValueError, match="BGR SHA"):
        diagnostic.decode_png(encoded.tobytes(), "a" * 64)


def test_native_source_panel_exact_and_no_input_mutation():
    image = np.full((1200, 1200, 3), 47, np.uint8)
    fg = np.zeros((1200, 1200), bool)
    fg[10:12, 10:13] = True
    masks = fg[None].copy()
    copies = [v.copy() for v in (image, fg, masks)]
    rendered = diagnostic.panels(image, fg, masks, 90120)
    assert rendered.shape == (2528, 2400, 3)
    assert np.array_equal(rendered[64:1264, :1200], image)
    assert all(np.array_equal(a, b) for a, b in zip((image, fg, masks), copies, strict=True))


def test_unknown_packed_sidecar_rejected():
    stream = io.BytesIO()
    np.savez_compressed(stream, invented=np.array([1]))
    with pytest.raises(ValueError, match="schema"):
        diagnostic.load_masks(stream.getvalue(), {})


def test_output_outside_workspace_never_created(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "WORK", tmp_path / "allowed")
    monkeypatch.setattr(
        diagnostic, "load_headers", lambda *_: pytest.fail("real inputs must never load")
    )
    path = tmp_path / "outside"
    with pytest.raises(ValueError, match="inside"):
        diagnostic.execute(path)
    assert not path.exists()
