"""Core colors exclude borders; ambiguous instance ownership stays unavailable."""

import importlib.util
from pathlib import Path

import numpy as np


def load_module():
    path = Path(__file__).resolve().parents[1] / "labs/20260909_three_body/appearance_diagnostic.py"
    spec = importlib.util.spec_from_file_location("_appearance_diagnostic_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load_module()


def test_descriptor_uses_eroded_instance_core_not_bright_background():
    image = np.full((30, 60, 3), 255, np.uint8)
    mask = np.zeros((30, 60), bool)
    mask[5:25, 5:25] = True
    image[mask] = [20, 40, 100]
    descriptor, core = m.describe_core(image, mask)
    assert 0 < core.sum() < mask.sum()
    altered = image.copy()
    altered[~core] = [255, 0, 0]
    other, _ = m.describe_core(altered, mask)
    np.testing.assert_array_equal(descriptor, other)
    np.testing.assert_allclose(descriptor[:2], [100 / 160, 40 / 160], atol=1e-7)


def test_grey_core_has_no_spurious_hue_and_tiny_mask_abstains():
    image = np.full((20, 20, 3), 120, np.uint8)
    mask = np.ones((20, 20), bool)
    descriptor, _ = m.describe_core(image, mask)
    np.testing.assert_allclose(descriptor[2:5], [0.5, 0.5, 0])
    mask[:] = False
    mask[4:6, 4:6] = True
    assert m.describe_core(image, mask)[0] is None


def scene():
    uv = np.array([[[10, 10], [13, 10]], [[30, 10], [33, 10]], [[50, 10], [53, 10]]], float)
    masks = np.zeros((3, 30, 65), bool)
    for fish in range(3):
        masks[fish, 5:16, 5 + 20 * fish : 19 + 20 * fish] = True
    return uv, masks


def test_unique_instances_are_assigned_without_persistent_mask_row_identity():
    uv, masks = scene()
    rows, reasons = m.assign_instances(masks, uv, np.ones(3, bool))
    np.testing.assert_array_equal(rows, [0, 1, 2])
    assert list(reasons) == ["usable"] * 3
    rows, _ = m.assign_instances(masks[[2, 0, 1]], uv, np.ones(3, bool))
    np.testing.assert_array_equal(rows, [1, 2, 0])


def test_merged_mask_includes_unobserved_competitor_and_is_not_reassigned():
    uv, masks = scene()
    masks[0] |= masks[1]
    rows, reasons = m.assign_instances(masks, uv, np.array([True, False, True]))
    assert rows[0] == -1 and reasons[0] == "mask_contains_other_projected_node"
    assert rows[1] == -1 and reasons[1] == "no_direct_mouth_pect"
    assert rows[2] == 2


def test_duplicate_masks_and_invalid_competitor_projection_are_unknown():
    uv, masks = scene()
    rows, reasons = m.assign_instances(np.concatenate([masks, masks[:1]]), uv, np.ones(3, bool))
    assert rows[0] == -1 and reasons[0] == "multiple_joint_instances"
    uv[2] = np.nan
    rows, reasons = m.assign_instances(masks, uv, np.ones(3, bool))
    assert np.all(rows == -1)
    assert reasons[0] == "unknown_latent_competitor_projection"


def test_missing_view_and_zero_detections_stay_distinct():
    uv, masks = scene()
    assert np.all(m.assign_instances(None, uv, np.ones(3, bool))[1] == "view_unavailable")
    assert np.all(m.assign_instances(masks[:0], uv, np.ones(3, bool))[1] == "no_masks")


def test_validation_keeps_unavailable_and_color_ties_in_denominator():
    frames = np.arange(7)
    features = np.ones((7, 3, 3, 6)) * 0.5
    reasons = np.full((7, 3, 3), "usable", dtype="U64")
    features[6, 0, 0] = np.nan
    reasons[6, 0, 0] = "no_masks"
    report, _, _, predictions = m.summarize(features, reasons, frames, 5)
    assert report["0"]["validation_requested_fish_probes"] == 6
    assert report["0"]["validation_classified_fish_probes"] == 0
    assert report["0"]["validation_reason_counts"] == {"usable": 5, "no_masks": 1}
    assert np.all(predictions == -1)
    assert (
        np.asarray(
            report["0"][
                "conditional_confusion_rows_reference_slot_columns_nearest_template_plus_unscored"
            ]
        ).sum()
        == 6
    )
