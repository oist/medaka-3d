import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "body_pool_diagnostic", Path(__file__).with_name("diagnose_body_pool_79372.py")
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def inputs():
    original = np.arange(27, dtype=float).reshape(3, 3, 3)
    mask = np.stack((original, original + 1, original + 100))
    random = np.stack((original, original + 2, original + 3))
    before = np.stack((original, mask[1], random[1], random[2]))
    parts = np.arange(20, dtype=float).reshape(4, 5) / 7
    cfg = dict(shape_weight=0.3, motion_weight=0.2, repulsion_weight=0.15, mask_weight=3.0)

    def scores(pose):
        relative = pose - pose[:, :, 1:2]
        p_relative = original - original[:, 1:2]
        orientation = ((relative - p_relative) ** 2).sum(-1).mean((1, 2)) / 25
        return (
            parts[:, 0]
            + 0.3 * parts[:, 1]
            + 0.2 * parts[:, 2]
            + 0.01 * orientation
            + 0.15 * parts[:, 3]
            + 3 * parts[:, 4]
        )

    after = before.copy()
    after[:, 0, 0] += 4
    return dict(
        original_pose_xyz=original,
        predicted_pose_xyz=original,
        mask_pose_xyz=mask,
        random_pose_xyz=random,
        mask_fit_index=np.array([0, 1, -1]),
        random_fit_index=np.array([0, 2, 3]),
        mask_admissible=np.array([True, True, False]),
        random_admissible=np.ones(3, bool),
        mask_component_slot_sign_roll=np.array([[-1, -1, 0, 0], [0, 1, 1, 0], [0, 2, 1, 1]]),
        random_raw_slot=np.array([-1, 1, 2]),
        fit_input_pose_xyz=before,
        fit_pose_xyz=after,
        fit_initial_loss_parts=parts,
        fit_loss_parts=parts,
        fit_initial_scores=scores(before),
        fit_scores=scores(after),
        fit_initial_observed=np.ones((4, 3, 3), bool),
        fit_observed=np.ones((4, 3, 3), bool),
        fit_plans=np.zeros((4, 3, 3, 3), np.int32),
    ), cfg


def test_all_nominal_starts_and_unique_fits_keep_rejected_denominator_and_mapping():
    arrays, _ = inputs()
    poses, rows, links, offset = m.collect_poses(arrays)
    assert len(poses) == 9 and offset == 5
    assert rows[2]["admissible"] is False and rows[2]["fit_index"] == -1
    np.testing.assert_array_equal(links, [[0, 5], [1, 6], [3, 7], [4, 8]])
    assert rows[7]["arm"] == "random" and rows[7]["trial"] == 1
    poses[0] = -1
    assert arrays["original_pose_xyz"][0, 0, 0] == 0


def test_weighted_terms_include_unsaved_orientation_and_reject_mismatch():
    arrays, config = inputs()
    poses, rows, _, _ = m.collect_poses(arrays)
    scores = m.score_components(arrays, poses, rows, config)
    assert not scores["score_available"][2]
    assert np.isnan(scores["score"][2])
    assert scores["weighted_loss_parts"][5, 3] > 0
    np.testing.assert_allclose(scores["weighted_loss_parts"][5:].sum(1), arrays["fit_scores"])
    arrays["fit_scores"][0] += 1
    with pytest.raises(AssertionError):
        m.score_components(arrays, poses, rows, config)


def test_invalid_admitted_index_cannot_hide_rejected_start():
    arrays, _ = inputs()
    arrays["mask_fit_index"][2] = 2
    with pytest.raises(ValueError, match="admission"):
        m.collect_poses(arrays)


def test_partial_run_is_rejected_before_any_candidate_read(tmp_path):
    (tmp_path / "experiment.provenance.json").write_text(json.dumps(dict(status="running")))
    with pytest.raises(ValueError, match="success-sealed"):
        m.verify_complete(tmp_path)


def test_region_selection_preserves_full_masks_and_empty_eroded_core():
    arrays = {}
    for view in ("top_cam", "side1_cam", "side2_cam"):
        arrays["available_" + view] = np.asarray(True)
        arrays["masks_" + view] = np.zeros((3, 1200, 1200), bool)
    arrays["masks_top_cam"][0, 5, 7] = True
    arrays["masks_top_cam"][2, 15:30, 25:40] = True
    arrays["masks_side1_cam"][2, 35:50, 65:80] = True
    pixels, masks = m.region_pixels(arrays, ("top_cam", "side1_cam", "side2_cam"))
    assert len(pixels["top_cam"]) == 226 and len(pixels["side2_cam"]) == 0
    assert masks[("top_cam", 0)][0].sum() == 1
    assert not masks[("top_cam", 0)][1].any()
    np.testing.assert_array_equal(pixels["top_cam"][0], [7, 5])
