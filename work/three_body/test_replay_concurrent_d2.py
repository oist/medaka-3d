"""Bounded CPU tests for fixed native targets and exact packed comparisons."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "concurrent_d2_replay_tested", Path(__file__).with_name("replay_concurrent_d2.py")
)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def packed():
    encoder = mod.staging.single.original.PackedMasks((9, 12))
    for frame, count in ((53120, 2), (53121, 0), (53122, 3), (53123, 1)):
        masks = np.zeros((count, 9, 12), bool)
        for row in range(count):
            if row != 1:  # The second row deliberately has an empty cropped mask.
                masks[row, row + 1 : row + 3, row + 2 : row + 5] = True
        encoder.append(
            frame,
            masks,
            np.full(count, 0.75, np.float32),
            np.zeros((count, 4), np.float32),
            np.zeros(count, np.int64),
        )
    return encoder.arrays()


def test_targets_cover_all_fixed_1810_images_with_no_top_or_duplicates():
    result = mod.targets()
    assert len(result) == len({tuple(value) for value in result}) == 1810
    assert result == (
        [["side1_cam", f] for f in range(53120, 54780)]
        + [["side2_cam", f] for f in range(53120, 53270)]
    )


def test_slice_reencodes_identical_bytes_across_empty_frame_and_empty_crop():
    source = packed()
    clipped = mod.slice_packed(source, 1, 4)
    encoder = mod.staging.single.original.PackedMasks((9, 12))
    for i in range(1, 4):
        lo, hi = source["frame_offsets"][i : i + 2]
        encoder.append(
            int(source["frame_idx"][i]),
            mod.staging.single.original.decode_frame_masks(source, i),
            source["scores"][lo:hi],
            source["pred_boxes"][lo:hi],
            source["classes"][lo:hi],
        )
    expected = encoder.arrays()
    assert len(clipped) == 13
    for key in expected:
        assert mod.byte_equal(clipped[key], expected[key]), key
    assert mod.compare_packed(clipped, expected)["all_thirteen_fields_byte_exact"]
    assert not np.shares_memory(clipped["mask_bits"], source["mask_bits"])


def test_zero_detections_and_unavailable_remain_distinct():
    a = mod.slice_packed(packed(), 1, 2)
    b = {key: value.copy() for key, value in a.items()}
    b["view_available"][0] = False
    result = mod.compare_packed(a, b)
    assert not result["all_thirteen_fields_byte_exact"]
    assert result["frames"][0]["fields_byte_exact"]["frame_mask_sha256"]
    assert not result["frames"][0]["fields_byte_exact"]["view_available"]


def test_one_ulp_score_change_is_not_hidden_by_tolerance():
    a = packed()
    b = {key: value.copy() for key, value in a.items()}
    b["scores"][0] = np.nextafter(b["scores"][0], np.float32(1))
    result = mod.compare_packed(a, b)
    assert not result["all_thirteen_fields_byte_exact"]
    assert result["arrays"]["scores"]["max_absolute_float_difference"] > 0
    assert not result["frames"][0]["all_thirteen_fields_byte_exact"]
    assert all(item["all_thirteen_fields_byte_exact"] for item in result["frames"][1:])


def test_signed_zero_box_bits_are_compared():
    a = packed()
    b = {key: value.copy() for key, value in a.items()}
    b["pred_boxes"][0, 0] = np.float32(-0.0)
    result = mod.compare_packed(a, b)
    assert not result["arrays"]["pred_boxes"]["byte_exact"]
    assert result["arrays"]["pred_boxes"]["max_absolute_float_difference"] == 0


def test_permuting_row_contents_is_not_relabelled_away():
    source = packed()
    encoder = mod.staging.single.original.PackedMasks((9, 12))
    masks = mod.staging.single.original.decode_frame_masks(source, 0)
    encoder.append(
        53120, masks[::-1], source["scores"][:2], source["pred_boxes"][:2], source["classes"][:2]
    )
    result = mod.compare_packed(encoder.arrays(), mod.slice_packed(source, 0, 1))
    assert not result["all_thirteen_fields_byte_exact"]
    assert result["frames"][0]["fields_byte_exact"]["instance_id"]
    assert not result["frames"][0]["fields_byte_exact"]["frame_mask_sha256"]


@pytest.mark.parametrize("bad", ["frame", "missing_field", "payload"])
def test_bad_comparison_is_rejected(bad):
    a = packed()
    b = {key: value.copy() for key, value in a.items()}
    if bad == "frame":
        b["frame_idx"] += 1
    elif bad == "missing_field":
        b.pop("scores")
    else:
        b["mask_bits"][0] ^= 1
    with pytest.raises(ValueError):
        mod.compare_packed(a, b)


def test_unregistered_or_changed_plan_rejected_before_predictor(tmp_path):
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="plan SHA"):
        mod.verify_plan(tmp_path, "0" * 64)
