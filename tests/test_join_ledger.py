"""Constructed identity errors, gaps, mixed tracks, and event-group boundaries."""

import numpy as np
import pytest

from medaka3d.eval.join_ledger import JoinLedgerConfig, build_join_ledger


def ledger(tracks, matches, *, times=None, valid=None, config=None):
    tracks, matches = np.array(tracks), np.array(matches)
    times = np.arange(len(tracks), dtype=float) + 300 if times is None else np.array(times)
    return build_join_ledger(
        tracks,
        matches,
        times,
        np.r_[np.diff(times), 0.0],
        np.ones(len(tracks), bool),
        np.ones(tracks.shape, bool) if valid is None else valid,
        reference_labels=("M1", "M2"),
        config=config or JoinLedgerConfig(minimum_matched_frames=1),
    )


def test_two_edges_one_candidate_group_for_a_constructed_exchange():
    tracks = [[0, 1]] * 3 + [[2, 3]] * 3
    result = ledger(tracks, [[0, 1]] * 3 + [[1, 0]] * 3)
    assert result["edges"].outcome.tolist() == ["wrong", "wrong"]
    assert result["summary"]["edge_count"] == 2
    assert result["summary"]["candidate_group_count"] == 1
    assert result["summary"]["adjudicated_event_count"] is None
    assert result["event_candidates"].iloc[0].review_status == "unreviewed_temporal_candidate"


def test_correct_controls_remain_in_the_edge_denominator():
    result = ledger([[0, 1]] * 3 + [[2, 3]] * 3, [[0, 1]] * 6)
    assert result["summary"]["edge_outcomes"] == {"correct": 2, "wrong": 0, "undecidable": 0}
    assert result["summary"]["assessed_fraction_of_all_edges"] == 1


def test_mixed_track_is_not_certified_by_a_pure_subsegment():
    # Track 0 returns after a gap with the other identity. Each piece is pure.
    tracks = [[0, 1]] * 2 + [[-1, -1]] + [[0, 2]] * 3
    result = ledger(tracks, [[0, 1]] * 3 + [[1, 0]] * 3)
    track = result["tracks"].set_index("local_track_id").loc[0]
    assert track.label_status == "mixed"
    edge = result["edges"].iloc[0]
    assert edge.outcome == "undecidable"
    assert "track_mixed" in edge.undecidable_reasons


def test_gap_keeps_endpoint_agreement_but_no_assertion_about_its_interior():
    tracks = [[0, 1]] * 2 + [[-1, -1]] + [[2, 3]] * 3
    result = ledger(tracks, [[0, 1]] * 6)
    assert result["edges"].endpoint_outcome.tolist() == ["same_reference_identity"] * 2
    assert result["edges"].outcome.tolist() == ["correct"] * 2
    assert result["edges"].gap_interior_identity_status.tolist() == ["unassessed"] * 2
    assert result["edges"].gap_frames.tolist() == [1, 1]
    assert result["summary"]["expected_grid_fish_seconds"] == 10
    assert result["summary"]["local_position_fish_seconds"] == 8


def test_reference_unavailable_is_not_a_correct_edge():
    result = ledger([[0, 1]] * 3 + [[2, 3]] * 3, [[0, 1]] * 3 + [[-1, -1]] * 3)
    assert result["edges"].outcome.tolist() == ["undecidable"] * 2
    assert result["summary"]["assessed_fraction_of_all_edges"] == 0
    assert result["summary"]["correct_fraction_of_assessed_edges"] is None


def test_low_support_and_missing_reference_have_separate_reasons():
    tracks = [[0, 1]] * 4 + [[2, 3]] * 2
    result = ledger(
        tracks,
        [[0, 1], [-1, -1], [0, 1], [0, 1], [0, 1], [0, 1]],
        config=JoinLedgerConfig(minimum_matched_frames=3),
    )
    statuses = result["tracks"].set_index("local_track_id").label_status
    assert statuses[0] == "reference_incomplete"
    assert statuses[2] == "insufficient_support"


def test_duration_majority_uses_native_intervals_not_frame_counts():
    result = ledger(
        [[0, 1]] * 4,
        [[0, 1], [0, 1], [1, 0], [1, 0]],
        times=[300, 300.1, 300.2, 301.2],
    )
    track = result["tracks"].set_index("local_track_id").loc[0]
    assert track.majority_label == "M2"
    assert track.label_status == "mixed"
    assert track.reference_M2_seconds == pytest.approx(1.0)


def test_time_groups_have_a_bounded_span_not_transitive_single_linkage():
    tracks = [[0, 1], [0, 1], [2, 1], [2, 3], [4, 3], [4, 3], [4, 3]]
    result = ledger(
        tracks,
        [[0, 1]] * len(tracks),
        times=[300, 301, 302, 302.2, 302.4, 303, 304],
    )
    assert result["event_candidates"].edge_count.tolist() == [2, 1]


@pytest.mark.parametrize("duplicate", ["track", "match"])
def test_nonunique_correspondence_is_rejected(duplicate):
    tracks = [[0, 1]] * 3
    matches = [[0, 1]] * 3
    if duplicate == "track":
        tracks[1] = [0, 0]
    else:
        matches[1] = [0, 0]
    with pytest.raises(ValueError, match=r"multiple slots|one-to-one"):
        ledger(tracks, matches)


def test_no_local_tracks_preserves_the_expected_denominator():
    result = ledger([[-1, -1]] * 3, [[-1, -1]] * 3)
    assert result["edges"].empty
    assert result["summary"]["edge_count"] == 0
    assert result["summary"]["expected_grid_fish_seconds"] == 4
    assert result["summary"]["local_position_fish_seconds"] == 0
