"""Scientific scoring invariants on small, explicitly synthetic trajectories."""

import numpy as np
import pytest

from medaka3d.eval.review_identity import (
    contiguous_runs,
    fixed_anchor_mapping,
    identity_status,
    interval_weights,
    match_positions,
    slot_change_events,
)


def world(n=8):
    return np.tile(np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0]]), (n, 1, 1))


def match(ref, pred, ref_ok=None):
    valid = np.ones(ref.shape[:2], bool)
    return match_positions(
        ref,
        pred,
        valid if ref_ok is None else ref_ok,
        valid,
        max_distance_mm=5.0,
        min_margin_mm=2.0,
    )


def test_lasting_swap_is_not_removed_by_per_frame_spatial_matching():
    ref = world()
    pred = ref.copy()
    pred[4:] = pred[4:, [1, 0, 2]]
    result = match(ref, pred)
    anchor = fixed_anchor_mapping(result.slots, np.arange(8) < 4, minimum_matches=3)
    status = identity_status(result, anchor["reference_to_slot"])
    assert np.count_nonzero(status == "wrong_id") == 8
    assert np.count_nonzero(status == "correct_id") == 16


def test_identical_crossing_is_ambiguous_and_cannot_duplicate_matches():
    ref = world(1)
    ref[0, 1] = ref[0, 0]
    result = match(ref, ref)
    assert result.slots.tolist() == [[-1, -1, 2]]
    assert result.reason[0, 0] == "spatial_ambiguity"


def test_missing_reference_does_not_remove_unaffected_fish():
    ref = world(2)
    pred = ref.copy()
    ref[:, 1] = np.nan
    result = match(ref, pred)
    assert result.slots.tolist() == [[0, -1, 2], [0, -1, 2]]
    assert result.reason[0, 1] == "reference_unavailable"


def test_invalid_but_finite_neighbor_still_prevents_ambiguous_match():
    ref = world(1)
    pred = ref.copy()
    ref[:, 1] = ref[:, 0] + 0.1
    valid = np.array([[True, False, True]])
    result = match(ref, pred, valid)
    assert result.slots[0, 0] == -1


def test_missing_predictions_are_not_zero_positions():
    ref = world(1)
    pred = np.full_like(ref, np.nan)
    result = match(ref, pred)
    assert (result.reason == "prediction_missing").all()
    assert (result.slots == -1).all()


@pytest.mark.parametrize("wrong_reference", [True, False])
def test_quality_masks_cannot_silently_broadcast_between_animals(wrong_reference):
    ref = world(2)
    valid = np.ones((2, 3), bool)
    wrong = np.ones((2, 1), bool)
    with pytest.raises(ValueError, match="Validity mask shape mismatch"):
        match_positions(
            ref,
            ref,
            wrong if wrong_reference else valid,
            valid if wrong_reference else wrong,
            max_distance_mm=5.0,
            min_margin_mm=2.0,
        )


def test_insufficient_anchor_remains_unresolved():
    result = match(world(2), world(2))
    mapping = fixed_anchor_mapping(result.slots, np.ones(2, bool), minimum_matches=3)
    assert mapping["reference_to_slot"] == [-1, -1, -1]
    assert (identity_status(result, mapping["reference_to_slot"]) == "anchor_unresolved").all()


def test_anchor_uses_only_declared_frames():
    slots = np.tile([1, 2, 0], (100, 1))
    slots[:5] = [0, 1, 2]
    result = fixed_anchor_mapping(slots, np.arange(100) < 5, minimum_matches=3)
    assert result["reference_to_slot"] == [0, 1, 2]


def test_native_time_weights_do_not_extrapolate_or_fill_leading_gap():
    weights, denominator = interval_weights(np.array([0.0, 0.8, 1.7, 3.0]), 1.0)
    assert weights.tolist() == [0.0, 0.0, 1.3, 0.0]
    assert denominator == 2.0
    with pytest.raises(ValueError):
        interval_weights(np.array([0.0, 0.0, 1.0]), 0.0)


def test_runs_break_at_unknown_frame():
    assert contiguous_runs([True, True, False, True]).tolist() == [[0, 2], [3, 4]]


def test_gap_change_not_counted_as_adjacent_switch():
    slots = np.array([[0, 1, 2], [-1, 1, 2], [1, 0, 2]])
    events = slot_change_events(slots, np.ones(3, bool))
    assert {e["kind"] for e in events} == {"adjacent_change", "gap_endpoint_change"}
