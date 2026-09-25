"""Small native-byte and fixed-crop checks, without actual images or tracking."""

import numpy as np
import pytest
import render_tail_shape_raw_review as m


def test_native_evidence_byte_change_rejected():
    image = np.zeros((1200, 1200, 3), np.uint8)
    points = np.zeros((3, 3, 2), np.float32)
    mask = np.zeros((1, 1200, 1200), bool)
    obs = dict(masks=mask, detections=points, mask_available=np.asarray(True))
    evidence = dict(
        bgr_sha256=m.array_sha256(image),
        mask_count=1,
        mask_sha256=m.array_sha256(mask),
        detection_signature=m.audit.array_signature({"points": points})["points"],
    )
    m.check_native(image, obs, evidence)
    image[0, 0, 0] = 1
    with pytest.raises(ValueError, match="BGR"):
        m.check_native(image, obs, evidence)
    image[0, 0, 0] = 0
    mask[0, 0, 0] = True
    with pytest.raises(ValueError, match="mask"):
        m.check_native(image, obs, evidence)
    mask[0, 0, 0] = False
    points[0, 0, 0] = 1
    with pytest.raises(ValueError, match="SLEAP"):
        m.check_native(image, obs, evidence)


def test_empty_known_mask_is_not_unavailable():
    image = np.zeros((1200, 1200, 3), np.uint8)
    points = np.empty((0, 3, 2), np.float32)
    masks = np.empty((0, 1200, 1200), bool)
    obs = dict(masks=masks, detections=points, mask_available=np.asarray(False))
    evidence = dict(
        bgr_sha256=m.array_sha256(image),
        mask_count=None,
        mask_sha256=None,
        detection_signature=m.audit.array_signature({"points": points})["points"],
    )
    m.check_native(image, obs, evidence)
    obs["mask_available"] = np.asarray(True)
    with pytest.raises(ValueError, match="mask"):
        m.check_native(image, obs, evidence)


def test_fixed_crop_never_expands_and_nonfinite_separate():
    crop = np.array([100, 200, 300, 400], np.int64)
    xy = np.full((3, 3, 2), 250.0)
    xy[0, 0] = [99, 250]
    xy[1, 1] = [350, 250]
    xy[2, 2] = [np.nan, np.nan]
    before = crop.copy()
    finite, inside = m.crop_status(xy, crop)
    assert finite.sum() == 8 and inside.sum() == 6
    assert not inside[0, 0] and not inside[1, 1] and not finite[2, 2]
    assert np.array_equal(before, crop)


def test_pose_drawing_marks_latent_node_and_both_incident_segments():
    fig, ax = m.plt.subplots()
    xyz = np.tile(np.array([[100.0, 100.0], [110.0, 100.0], [120.0, 100.0]]), (3, 1, 1))
    observed = np.ones((3, 3), bool)
    observed[0, 1] = False
    m.draw_pose(ax, xyz, observed)
    assert len(ax.lines) == 15
    assert ax.lines[0].get_linestyle() == ax.lines[1].get_linestyle() == "--"
    assert ax.lines[3].get_markerfacecolor() == "none"
    assert ax.lines[5].get_linestyle() == "-"
    m.plt.close(fig)
