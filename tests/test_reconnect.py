"""Global permutation evidence must preserve geometry and expose ambiguity."""

import importlib.util
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_LAB = Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"
sys.path.insert(0, str(_LAB))
_SPEC = importlib.util.spec_from_file_location("generative_reconnect_test", _LAB / "reconnect.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)
sys.path.remove(str(_LAB))


def scene(n=10, identical=False):
    centers = np.array([[-5, 0, 20], [5, 0, 20], [0, 60, 20]], dtype=float)
    initial = centers[:, None] + np.array([[0, 3, 0], [0, 0, 0], [0, -10, 0]])
    pose = np.broadcast_to(initial, (n, 3, 3, 3)).copy()
    observed = np.ones((n, 3, 3), bool)
    if n > 2:
        observed[2, :2] = False
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    source[~observed[:, :, None].repeat(3, axis=2)] = -1
    hist = np.zeros((n, 3, 3, 24))
    quality = np.zeros((n, 3, 3))
    for fish in range(3):
        hist[0, fish, :, 0 if identical and fish < 2 else fish] = 1
    quality[0] = 1
    if n > 6:
        for fish, identity in enumerate((1, 0, 2)):
            hist[6:, fish, :, 0 if identical and fish < 2 else identity] = 1
        quality[6:] = 1
    return initial, SimpleNamespace(
        frame_idx=np.arange(n), pose_xyz=pose,
        xyz=np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan),
        observed=observed, node_source=source, review=~observed[:, :, :2].all(axis=2),
        geometry_review=~observed[:, :, :2].all(axis=2), identity_review=np.zeros((n, 3), bool),
        appearance=hist, appearance_quality=quality, diagnostics={"algorithm": "synthetic"})


def test_block_solver_matches_exhaustive_global_paths_and_min_margins():
    rng = np.random.default_rng(145)
    n = 5
    emission = rng.uniform(0, 3, (n, 6))
    transition = rng.uniform(0, 2, (n, 6, 6))
    anchors = np.array([0, 3, 4])
    gaps = [rng.uniform(0, 4, (6, 6)), np.zeros((6, 6))]
    path, margins, optimum = _MOD._solve(emission, transition, anchors, gaps)
    enumerated = []
    for suffix in product(range(6), repeat=n - 1):
        sequence = (0, *suffix)
        cost = sum(emission[t, state] for t, state in enumerate(sequence))
        cost += sum(transition[t, sequence[t - 1], sequence[t]] for t in range(1, n))
        cost += gaps[0][sequence[0], sequence[3]]
        enumerated.append((cost, sequence))
    expected_cost, expected_path = min(enumerated)
    assert optimum == pytest.approx(expected_cost)
    np.testing.assert_array_equal(path, expected_path)
    expected = np.full((n, 6), np.inf)
    for cost, sequence in enumerated:
        for t, state in enumerate(sequence):
            expected[t, state] = min(expected[t, state], cost - expected_cost)
    np.testing.assert_allclose(margins, expected, atol=1e-12)


def test_future_appearance_changes_previously_selected_correspondence():
    initial, raw = scene()
    config = _MOD.ReconnectConfig(template_frames=1)
    prefix = SimpleNamespace(**{key: value[:6].copy() if isinstance(value, np.ndarray) else value
                               for key, value in vars(raw).items()})
    past_only = _MOD.reconnect(prefix, initial, config)
    full = _MOD.reconnect(raw, initial, config)
    np.testing.assert_array_equal(past_only.permutation[3], [0, 1, 2])
    np.testing.assert_array_equal(full.permutation[3], [1, 0, 2])
    np.testing.assert_array_equal(full.permutation[0], [0, 1, 2])
    assert full.identity_margin[3, :2].min() > config.review_margin
    assert not full.identity_review[3:, :2].any()
    assert full.geometry_review[2, :2].all()
    assert np.isnan(full.xyz[2, :2]).all()


def test_identical_appearance_after_missing_support_remains_unresolved():
    initial, raw = scene(14, identical=True)
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=1))
    # Raw falsely claimed zero identity review. Re-observing for more than
    # three frames cannot distinguish the two identical appearances.
    assert not raw.identity_review.any()
    assert result.identity_review[3:, :2].all()
    np.testing.assert_allclose(result.appearance_margin_gain[3:, :2], 0, atol=1e-10)
    assert np.isnan(result.accepted_xyz[3:, :2]).all()
    assert np.isfinite(result.xyz[3:, :2]).all()


def test_relabeling_preserves_sources_and_has_no_cross_frame_row_semantics():
    initial, raw = scene()
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=1))
    shuffled = SimpleNamespace(**vars(raw))
    shuffled.node_source = raw.node_source.copy()
    for t in range(len(raw.frame_idx)):
        valid = shuffled.node_source[t] >= 0
        shuffled.node_source[t][valid] = (shuffled.node_source[t][valid] + t) % 3 + 10 * t
    changed = _MOD.reconnect(shuffled, initial, _MOD.ReconnectConfig(template_frames=1))
    np.testing.assert_array_equal(result.permutation, changed.permutation)
    np.testing.assert_allclose(result.pose_xyz, changed.pose_xyz)
    for t in range(len(raw.frame_idx)):
        np.testing.assert_array_equal(changed.node_source[t],
                                      shuffled.node_source[t, result.permutation[t]])
        np.testing.assert_array_equal(changed.observed[t], raw.observed[t, result.permutation[t]])
    assert result.diagnostics["coordinate_refitting_or_interpolation"] is False


def test_geometry_review_survives_distinct_appearance_and_thresholds_do_not_fit():
    initial, raw = scene()
    raw.geometry_review[-1, 0] = True
    config = _MOD.ReconnectConfig(template_frames=1)
    a = _MOD.reconnect(raw, initial, config)
    b = _MOD.reconnect(raw, initial, replace(config, review_margin=1e6))
    np.testing.assert_array_equal(a.permutation, b.permutation)
    np.testing.assert_allclose(a.pose_xyz, b.pose_xyz)
    np.testing.assert_array_equal(a.node_source, b.node_source)
    persistent_id = np.flatnonzero(a.permutation[-1] == 0)[0]
    assert a.geometry_review[-1, persistent_id]
    assert a.review[-1, persistent_id]
    assert np.isnan(a.accepted_xyz[-1, persistent_id]).all()


def test_templates_stop_at_first_uncertain_frame_and_no_missing_interpolation():
    initial, raw = scene()
    raw.identity_review[1, :2] = True
    # These later feature vectors would contaminate a template averaged over
    # the full video. The initial trusted prefix is the only training input.
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=20))
    assert result.diagnostics["template_counts"][:2] == [[1, 1, 1], [1, 1, 1]]
    template = np.asarray(result.diagnostics["template_histograms"])
    np.testing.assert_array_equal(template[0, :, 0], 1)
    np.testing.assert_array_equal(template[1, :, 1], 1)
    assert np.isnan(result.xyz[2, :2]).all()


def test_fully_missing_interior_has_zero_permutation_margin():
    initial, raw = scene(5, identical=True)
    raw.observed[1:4] = False
    raw.node_source[1:4] = -1
    raw.xyz[1:4] = np.nan
    raw.geometry_review[1:4] = True
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=1))
    np.testing.assert_allclose(result.identity_margin[2], 0, atol=1e-12)
    assert result.review[1:4].all()
    assert np.isnan(result.xyz[1:4]).all()


def test_single_frame_is_anchored_and_input_source_duplicates_fail():
    initial, raw = scene(1)
    result = _MOD.reconnect(raw, initial)
    np.testing.assert_array_equal(result.permutation, [[0, 1, 2]])
    assert np.isinf(result.identity_margin).all()
    raw.node_source[:, 1] = raw.node_source[:, 0]
    with pytest.raises(ValueError, match="not exclusive"):
        _MOD.reconnect(raw, initial)


def test_accumulated_appearance_cannot_buy_a_large_one_frame_teleport():
    initial, raw = scene(100)
    raw.pose_xyz[:, 1] += np.array([100, 0, 0])
    initial[1] += np.array([100, 0, 0])
    raw.observed[:] = True
    raw.geometry_review[:] = False
    raw.node_source[:] = np.arange(3)[None, :, None, None]
    raw.xyz = raw.pose_xyz[:, :, 1].copy()
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=1))
    np.testing.assert_array_equal(result.permutation, np.tile([0, 1, 2], (100, 1)))
    assert result.diagnostics["physically_feasible_global_path"]


def test_repeating_weak_color_differences_does_not_certify_an_occluded_identity():
    initial, raw = scene(500)
    raw.appearance = raw.appearance * 0.01
    raw.appearance[..., 10] += 0.99
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=1))
    assert result.appearance_margin_gain[-1, :2].min() > 2
    assert result.identity_review[3:, :2].all()
    assert not result.appearance_verified[3:, :2].any()


def test_one_discriminative_camera_cannot_alone_release_gap_identity_review():
    initial, raw = scene(30)
    raw.appearance[:, :, 1:] = 0
    raw.appearance[:, :, 1:, 10] = 1
    result = _MOD.reconnect(raw, initial, _MOD.ReconnectConfig(template_frames=1))
    assert result.appearance_margin_gain[-1, :2].min() > 2
    assert not result.appearance_verified[3:, :2].any()
    assert result.identity_review[3:, :2].all()


def test_infeasible_observed_geometry_is_preserved_but_all_ids_are_reviewed():
    initial, raw = scene(2)
    raw.pose_xyz[1] += np.array([500, 0, 0])
    raw.xyz = raw.pose_xyz[:, :, 1].copy()
    result = _MOD.reconnect(raw, initial)
    assert result.review.all()
    assert not result.diagnostics["physically_feasible_global_path"]
    np.testing.assert_allclose(result.pose_xyz, raw.pose_xyz)
    assert np.isnan(result.accepted_xyz).all()


def test_partial_gap_cannot_hide_a_teleport_from_the_final_connection_audit():
    _initial, raw = scene(15)
    raw.observed[:] = True
    raw.observed[1:13, 0] = False  # keeps the common three-fish block open
    raw.observed[9, 1] = False
    raw.pose_xyz[10:, 1] += np.array([70, 0, 0])
    raw.geometry_review = ~raw.observed[:, :, :2].all(axis=2)
    # The short gap of ID1 is hidden inside a much longer gap of ID0.
    # Common-block endpoint speed allows it; ID1's own t8->t10 does not.
    review, events = _MOD._connection_review(
        raw.frame_idx, raw.pose_xyz, raw.observed, _MOD.ReconnectConfig())
    assert review[10:, 1].all()
    assert not review[:10].any()
    assert not review[:, [0, 2]].any()
    assert events == [{"id": 1, "previous_frame": 8, "frame": 10,
                       "displacement_mm": 70.0, "engineering_bound_mm": 14.0}]
