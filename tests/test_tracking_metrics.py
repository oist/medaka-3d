"""The no-ground-truth tracking metrics, pinned on cases whose answer is known.

Each test builds a tracker output whose failure is stated in the test name, so a
metric that stops detecting it fails here rather than in a decision document.
"""

from __future__ import annotations

import numpy as np

from medaka3d.eval.tracking import (
    UNTRACKED,
    ViewTracks,
    fragmentation,
    identity_preservation,
    modal_agreement,
    switch_candidates,
    track_jumps,
    two_proportion_z,
)
from medaka3d.types import Encounter


def build(paths: np.ndarray, tracks: np.ndarray, n_frames: int | None = None) -> ViewTracks:
    """``paths`` (n_frames, 3, 2) px, ``tracks`` (n_frames, 3) IDs -> flat ViewTracks."""
    n = paths.shape[0]
    frame_idx = np.repeat(np.arange(n), 3)
    return ViewTracks(
        experiment="E",
        view="top_cam",
        frame_idx=frame_idx,
        mouth=paths.reshape(-1, 2),
        track=tracks.reshape(-1).astype(np.int32),
        n_frames=n_frames or n,
    )


def clean(n=50, drift=(2.0, 1.0)):
    """Three animals 300 px apart, each drifting 2.2 px per frame."""
    base = np.array([[100.0, 100.0], [400.0, 120.0], [250.0, 420.0]])
    return base[None] + np.asarray(drift, dtype=float)[None, None] * np.arange(n)[:, None, None]


class TestSwitchCandidates:
    def test_a_clean_tracker_has_none(self):
        paths = clean()
        tracks = np.tile([0, 1, 2], (paths.shape[0], 1))
        out = switch_candidates(build(paths, tracks), gate=60.0)
        assert out["scored"] == paths.shape[0] - 1
        assert out["switch_candidates"] == 0
        assert out["switch_candidate_rate"] == 0.0

    def test_a_swapped_pair_is_caught(self):
        paths = clean()
        tracks = np.tile([0, 1, 2], (paths.shape[0], 1))
        tracks[25:, [0, 1]] = tracks[25:, [1, 0]]  # the tracker exchanges two IDs
        out = switch_candidates(build(paths, tracks), gate=60.0)
        assert out["switch_candidates"] == 1
        # Exactly one transition is wrong; the metric must not smear it across
        # every later frame, or one swap would read as twenty-five.
        assert out["scored"] == paths.shape[0] - 1

    def test_untracked_instances_are_excluded_not_counted_as_correct(self):
        paths = clean(n=10)
        tracks = np.tile([0, 1, 2], (10, 1))
        tracks[5, 0] = UNTRACKED
        out = switch_candidates(build(paths, tracks), gate=60.0)
        assert out["excluded_untracked_or_degenerate"] == 2  # into and out of frame 5
        assert out["scored"] == 7

    def test_a_transition_geometry_cannot_adjudicate_is_excluded(self):
        # Everything moves further than the gate, so the geometric winner is not
        # evidence about anything and the tracker is not the thing at fault.
        paths = clean(n=6, drift=np.array([400.0, 400.0]))
        tracks = np.tile([0, 1, 2], (6, 1))
        out = switch_candidates(build(paths, tracks), gate=10.0)
        assert out["excluded_over_gate"] == 5
        assert out["scored"] == 0
        assert np.isnan(out["switch_candidate_rate"])


class TestTrackJumps:
    def test_smooth_motion_has_none(self):
        paths = clean()
        tracks = np.tile([0, 1, 2], (paths.shape[0], 1))
        out = track_jumps(build(paths, tracks), gate=60.0)
        assert out["jumps"] == 0
        assert out["adjacent_pairs"] == 3 * (paths.shape[0] - 1)

    def test_a_teleport_is_caught(self):
        paths = clean(n=10)
        paths[5:, 0] += np.array([500.0, 0.0])
        tracks = np.tile([0, 1, 2], (10, 1))
        out = track_jumps(build(paths, tracks), gate=60.0)
        assert out["jumps"] == 1

    def test_a_gap_is_counted_as_a_gap_not_a_jump(self):
        # Across four missing frames the animal had time to move, so a large
        # displacement is not evidence of a swap.
        paths = clean(n=10)
        tracks = np.tile([0, 1, 2], (10, 1))
        view = build(paths, tracks)
        keep = ~((view.frame_idx >= 4) & (view.frame_idx <= 7) & (view.track == 0))
        gapped = ViewTracks(
            experiment="E",
            view="top_cam",
            frame_idx=view.frame_idx[keep],
            mouth=view.mouth[keep],
            track=view.track[keep],
            n_frames=10,
        )
        out = track_jumps(gapped, gate=60.0)
        assert out["bridged_gaps"] == 1
        assert out["jumps"] == 0


class TestFragmentation:
    def test_three_unbroken_tracks_are_ideal(self):
        paths = clean()
        tracks = np.tile([0, 1, 2], (paths.shape[0], 1))
        out = fragmentation(build(paths, tracks))
        assert out["tracks"] == 3
        assert out["tracks_per_animal"] == 1.0
        assert out["coverage"] == 1.0
        assert out["frames_covered_by_top3"] == 1.0

    def test_a_tracker_that_never_links_is_penalised_here_and_only_here(self):
        # This is the failure the switch rate cannot see: a fresh ID every frame
        # has a perfect switch rate by construction.
        paths = clean(n=20)
        tracks = np.arange(60).reshape(20, 3)
        switches = switch_candidates(build(paths, tracks), gate=60.0)
        assert switches["scored"] == 0  # no ID survives to be compared
        out = fragmentation(build(paths, tracks))
        assert out["tracks"] == 60
        assert out["tracks_per_animal"] == 20.0

    def test_untracked_instances_lower_coverage(self):
        paths = clean(n=10)
        tracks = np.tile([0, 1, 2], (10, 1))
        tracks[0:5, 2] = UNTRACKED
        out = fragmentation(build(paths, tracks))
        assert out["coverage"] == 25 / 30


class TestIdentityPreservation:
    def encounter(self, entry=10, exit_=20):
        return Encounter(
            experiment="E",
            start_frame=0,
            end_frame=30,
            entry_frame=entry,
            exit_frame=exit_,
            participants=(0, 1),
            min_distance_mm=5.0,
        )

    def test_unbroken_identities_are_preserved(self):
        slot_track = np.tile([0, 1, 2], (30, 1))
        out = identity_preservation([self.encounter()], slot_track)
        assert out["preserved"] == 1 and out["broken"] == 0
        assert out["preserved_rate"] == 1.0

    def test_a_swap_inside_the_window_is_broken(self):
        slot_track = np.tile([0, 1, 2], (30, 1))
        slot_track[15:, [0, 1]] = slot_track[15:, [1, 0]]
        out = identity_preservation([self.encounter()], slot_track)
        assert out["broken"] == 1

    def test_the_uninvolved_third_animal_does_not_dilute_the_rate(self):
        slot_track = np.tile([0, 1, 2], (30, 1))
        slot_track[15:, 2] = 9  # the bystander's track breaks; the pair's does not
        assert identity_preservation([self.encounter()], slot_track)["preserved"] == 1
        assert (
            identity_preservation([self.encounter()], slot_track, participants_only=False)[
                "broken"
            ]
            == 1
        )

    def test_an_untracked_boundary_is_skipped_not_scored(self):
        slot_track = np.tile([0, 1, 2], (30, 1))
        slot_track[9, 0] = UNTRACKED
        out = identity_preservation([self.encounter()], slot_track)
        assert out["skipped_untracked_boundary"] == 1
        assert out["scored"] == 0
        assert np.isnan(out["preserved_rate"])


class TestModalAgreement:
    def test_a_constant_pairing_agrees_everywhere(self):
        out = modal_agreement(np.tile([0, 1, 2], (100, 1)))
        assert out["agreement_rate"] == 1.0
        assert out["modal_pairing"] == [0, 1, 2]

    def test_a_minority_of_disagreeing_frames_is_measured(self):
        pairings = np.tile([0, 1, 2], (100, 1))
        pairings[:10] = [1, 0, 2]
        assert modal_agreement(pairings)["agreement_rate"] == 0.9

    def test_unresolved_frames_are_excluded_from_the_denominator(self):
        pairings = np.tile([0, 1, 2], (100, 1))
        pairings[:50] = -1
        out = modal_agreement(pairings)
        assert out["resolved"] == 50 and out["agreement_rate"] == 1.0


class TestTwoProportionZ:
    def test_identical_proportions_give_p_one(self):
        _, p = two_proportion_z(50, 1000, 50, 1000)
        assert p == 1.0

    def test_a_large_difference_is_significant(self):
        _, p = two_proportion_z(10, 1000, 60, 1000)
        assert p < 0.001

    def test_empty_denominators_do_not_raise(self):
        z, p = two_proportion_z(0, 0, 1, 10)
        assert np.isnan(z) and np.isnan(p)
