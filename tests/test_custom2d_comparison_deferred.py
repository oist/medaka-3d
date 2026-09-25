"""Scientific contract checks for bounded re-evaluation of fixed C tracklets."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def modules():
    directory = Path(__file__).resolve().parents[1] / "labs/20260917_custom2d_comparison"
    previous = sys.modules.get("tracking")
    loaded = []
    for name in ("tracking", "deferred"):
        key = name if name == "tracking" else "_comparison_deferred_test"
        spec = importlib.util.spec_from_file_location(key, directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    yield tuple(loaded)
    sys.modules.pop("_comparison_deferred_test", None)
    if previous is None:
        sys.modules.pop("tracking", None)
    else:
        sys.modules["tracking"] = previous


def partition(labels):
    return labels[:, None] == labels[None, :]


def example(tracking, *, gap=2, third=None, tie=False, interior=True):
    frames, points, labels, truth = [], [], [], []
    last = 2 + gap + 2
    for frame in range(last + 1):
        for fish in range(2 + (third is not None)):
            if fish == 2:
                x, label = third, 99
            elif frame <= 2:
                x = 0.0 if tie else (frame - 3) * 0.8 * (1 - 2 * fish)
                label = fish
            elif frame >= 2 + gap:
                x = (1.0 if tie else (frame - 3) * 0.8) * (1 - 2 * fish)
                label = fish + 2
            elif interior:
                x, label = 0.0, 10 + 2 * frame + fish
            else:
                continue
            frames.append(75000 + frame)
            points.append(np.array([[4.0, 0], [0, 0], [-12, 0]]) + np.array([x, 0]))
            labels.append(label)
            truth.append(fish)
    frames = np.asarray(frames, np.int64)
    labels = np.asarray(labels, np.int64)
    events = [
        dict(
            kind="ambiguous_directional_assignment",
            from_frame=75002,
            to_frame=75002 + gap,
            from_row=int(np.flatnonzero((frames == 75002) & (labels == 0))[0]),
            to_row=int(np.flatnonzero((frames == 75002 + gap) & (labels == 2))[0]),
        )
    ]
    base = tracking.TrackingResult(
        labels, events, {"algorithm_revision": "v2_retire_unmatched_competing_histories"}
    )
    return frames, np.asarray(points), np.asarray(truth), base


def assert_preserved(base, result, frames):
    assert result.labels.shape == base.labels.shape
    assert result.labels.dtype == np.int64
    assert np.all(result.labels >= 0)
    assert np.all(partition(result.labels)[partition(base.labels)])
    assert result.metadata["retained_detections"] == len(frames)
    assert result.metadata["identity_accuracy_claim"] is False
    for frame in np.unique(frames):
        values = result.labels[frames == frame]
        assert len(set(values)) == len(values)


def decisions(result):
    return [e for e in result.events if e["kind"] == "deferred_candidate"]


def test_future_sequence_resolves_boundary_ambiguity_and_keeps_uncertain_interior(modules):
    tracking, deferred = modules
    frames, points, truth, base = example(tracking)
    saved = copy.deepcopy(base)
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    assert_preserved(base, result, frames)
    assert result.metadata["deferred_accepted_episodes"] == 1
    assert result.metadata["deferred_links_added"] == 2
    decision = decisions(result)[0]
    # At the boundary alone, the sum of the wrong alternatives is 3.2 px < 4 px.
    # Two later observations provide 6.4 px separation without threshold changes.
    assert decision["score_margin_px"] == pytest.approx(6.4)
    assert decision["required_margin_px"] == 4.0
    flanks = frames != 75003
    np.testing.assert_array_equal(partition(result.labels[flanks]), partition(truth[flanks]))
    for row in np.flatnonzero(~flanks):
        assert np.count_nonzero(result.labels == result.labels[row]) == 1
    np.testing.assert_array_equal(base.labels, saved.labels)
    assert base.events == saved.events and base.metadata == saved.metadata


def test_true_tie_is_refused_even_though_both_bridges_are_short(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking, tie=True)
    result = deferred.refine_deferred(frames, points, base, view="side2_cam")
    np.testing.assert_array_equal(partition(result.labels), partition(base.labels))
    assert decisions(result)[0]["reason"] == "insufficient_two_hypothesis_margin"
    assert decisions(result)[0]["score_margin_px"] == pytest.approx(0)


@pytest.mark.parametrize("third,accepted", [(150.0, 1), (0.0, 0)])
def test_distant_third_is_retained_but_participating_third_prevents_guess(modules, third, accepted):
    tracking, deferred = modules
    frames, points, truth, base = example(tracking, third=third)
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    assert_preserved(base, result, frames)
    assert result.metadata["deferred_accepted_episodes"] == accepted
    assert len(set(result.labels[truth == 2])) == 1
    assert not set(result.labels[truth == 2]) & set(result.labels[truth != 2])
    if not accepted:
        assert decisions(result)[0]["reason"] == "third_detection_in_candidate_neighbourhood"


def test_three_tracklet_components_are_not_reduced_to_favourite_pair(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking, third=0.0)
    base.labels[(base.labels == 99) & (frames >= 75004)] = 100
    base.labels[(base.labels == 99) & (frames == 75003)] = 101
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    np.testing.assert_array_equal(partition(result.labels), partition(base.labels))
    assert decisions(result)[0]["reason"] == "not_exactly_two_competing_tracks_on_both_sides"


@pytest.mark.parametrize("gap,accepted", [(3, 1), (4, 0), (9, 0)])
def test_deadline_uses_native_gap_including_missing_frames(modules, gap, accepted):
    tracking, deferred = modules
    frames, points, _, base = example(tracking, gap=gap, interior=False)
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    assert_preserved(base, result, frames)
    assert result.metadata["deferred_accepted_episodes"] == accepted
    if gap == 4:
        assert decisions(result)[0]["reason"] == "native_deadline_exceeded"
    if not accepted:
        np.testing.assert_array_equal(partition(result.labels), partition(base.labels))


def test_sparse_support_does_not_count_as_two_consecutive_future_observations(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking)
    frames[frames >= 75005] += 1
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    np.testing.assert_array_equal(partition(result.labels), partition(base.labels))


def test_missing_node_support_is_not_zero_error_and_tail_cannot_resolve_ids(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking)
    points[base.labels == 0, 0] = np.nan
    points[base.labels == 1, 1] = np.nan
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    np.testing.assert_array_equal(partition(result.labels), partition(base.labels))
    assert not result.metadata["deferred_accepted_episodes"]


def test_shared_single_primary_node_suffices_and_tail_changes_no_result(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking)
    points[:, 0] = np.nan
    first = deferred.refine_deferred(frames, points, base, view="side1_cam")
    points[:, 2] = np.random.default_rng(41).normal(0, 1e5, (len(points), 2))
    second = deferred.refine_deferred(frames, points, base, view="side1_cam")
    assert first.metadata["deferred_accepted_episodes"] == 1
    assert decisions(first)[0]["support_nodes"] == ["pectoral"]
    np.testing.assert_array_equal(first.labels, second.labels)


def transformed(base, tracking, order, transform_frame):
    inverse = np.argsort(order)
    events = copy.deepcopy(base.events)
    for event in events:
        for key in ("from_frame", "to_frame"):
            event[key] = transform_frame(event[key])
        for key in ("from_row", "to_row"):
            event[key] = int(inverse[event[key]])
    return tracking.TrackingResult(base.labels[order], events, base.metadata.copy())


def test_time_reverse_and_within_frame_row_permutation_preserve_partition(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking, third=150.0)
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    for reverse_time in (False, True):
        transform = (lambda f: 150006 - f) if reverse_time else (lambda f: f)
        transformed_frames = transform(frames)
        order = np.argsort(transformed_frames, kind="stable")
        order = np.concatenate(
            [order[transformed_frames[order] == frame][::-1] for frame in np.unique(frames)]
        )
        other_base = transformed(base, tracking, order, transform)
        other_base.labels = other_base.labels * 7 + 13
        other = deferred.refine_deferred(
            transformed_frames[order], points[order], other_base, view="side1_cam"
        )
        np.testing.assert_array_equal(partition(result.labels[order]), partition(other.labels))


def test_unlogged_cut_is_not_reconnected_and_empty_is_supported(modules):
    tracking, deferred = modules
    frames, points, _, base = example(tracking)
    base.events.clear()
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    np.testing.assert_array_equal(partition(result.labels), partition(base.labels))
    empty = tracking.TrackingResult(np.empty(0, np.int64), [], {})
    result = deferred.refine_deferred(
        np.empty(0, np.int64), np.empty((0, 3, 2)), empty, view="top_cam"
    )
    assert_preserved(empty, result, np.empty(0))


def test_actual_v2_ambiguity_can_be_revisited_from_saved_style_result(modules):
    tracking, deferred = modules
    frames = np.repeat(np.arange(25), 2)
    x = np.c_[np.arange(25) - 10, 10 - np.arange(25)].ravel()
    points = np.array([[[4.0, 0], [0, 0], [-12, 0]]]) + np.c_[x, x * 0][:, None]
    base = tracking.track_bidirectional(frames, points, view="side1_cam")
    result = deferred.refine_deferred(frames, points, base, view="side1_cam")
    assert_preserved(base, result, frames)
    assert result.metadata["deferred_accepted_episodes"] == 1
    assert result.metadata["output_tracklets"] < base.metadata["output_tracklets"]
