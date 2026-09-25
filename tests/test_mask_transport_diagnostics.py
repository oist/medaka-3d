"""Small synthetic reporting fixtures; no artifact, fit, renderer or GPU."""

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import mask_transport_diagnostics as m
import object_mask_transport as transport


def seed_pose():
    return np.array(
        [[[0, b * 10, 0], [2, b * 10, 0], [5, b * 10, 0]] for b in range(3)], np.float64
    )


def candidates(pose, observed=None):
    t = len(pose)
    row = dict(
        frame_idx=np.arange(100, 100 + t, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((t, 3, 3), bool) if observed is None else observed,
        identity_review=np.zeros((t, 3), bool),
    )
    return {arm: copy.deepcopy(row) for arm in m.ARMS}


def test_fold_and_seed_ratios_use_body_segment_axes_without_anatomical_cutoffs():
    pose = np.repeat(seed_pose()[None], 3, axis=0)
    pose[1, :, 2] = pose[1, :, 1] + [0, 3, 0]
    pose[2, :, 2] = pose[2, :, 1] + [-3, 0, 0]
    result = m.candidate_geometry(candidates(pose), seed_pose())
    for arm in m.ARMS:
        values = result["arrays"][arm]
        np.testing.assert_array_equal(values["fold_angle_deg"], [[0] * 3, [90] * 3, [180] * 3])
        np.testing.assert_array_equal(values["segment_length_mm"], np.tile([2, 3], (3, 3, 1)))
        np.testing.assert_array_equal(values["seed_length_ratio"], np.ones((3, 3, 2)))
        assert result["summary"][arm]["body_times"] == 9
        assert result["summary"][arm]["bodies"][0]["all_estimated_fold_angle_deg"]["maximum"] == 180
    assert result["independent_ID_truth"] is False


def test_zero_segments_and_zero_seed_keep_unavailable_angle_and_ratio():
    seed = seed_pose()
    seed[1, 1] = seed[1, 0]
    pose = np.repeat(seed_pose()[None], 2, axis=0)
    pose[0, 0, 1] = pose[0, 0, 0]
    pose[1, 0] = np.nan
    result = m.candidate_geometry(candidates(pose), seed)
    values, summary = result["arrays"]["original"], result["summary"]["original"]
    assert np.isnan(values["fold_angle_deg"][:, 0]).all()
    assert np.isnan(values["seed_length_ratio"][:, 1, 0]).all()
    assert values["zero_segment"][0, 0, 0]
    assert values["nonfinite_segment"][1, 0].all()
    angle = summary["bodies"][0]["all_estimated_fold_angle_deg"]
    assert angle["requested_count"] == angle["selected_nonfinite_count"] == 2
    assert angle["finite_count"] == 0 and angle["maximum"] is None
    json.dumps(summary, allow_nan=False)


def test_latent_and_direct_support_denominators_are_separate_and_inputs_unchanged():
    pose = np.repeat(seed_pose()[None], 2, axis=0)
    observed = np.ones((2, 3, 3), bool)
    observed[1, 0, 1] = False
    data = candidates(pose, observed)
    data["lk"]["identity_review"][:, 2] = True
    before = copy.deepcopy(data)
    result = m.candidate_geometry(data, seed_pose())
    body = result["summary"]["lk"]["bodies"][0]
    for segment in body["segments"]:
        assert segment["all_estimated_length_mm"]["finite_count"] == 2
        assert segment["direct_endpoint_length_mm"]["selected_count"] == 1
        assert segment["direct_endpoint_length_mm"]["requested_count"] == 2
    assert result["summary"]["lk"]["review_by_body"] == [0, 0, 2]
    result["arrays"]["lk"]["identity_review"][:] = False
    result["arrays"]["original"]["segment_length_mm"][:] = 999
    for arm in m.ARMS:
        for key in data[arm]:
            assert data[arm][key].tobytes() == before[arm][key].tobytes()


def history_fixture():
    frames = []
    parents = [[0, 0], [1, 0], [1, 1]]
    anchor_ids = [[0, 1], [1, 0], [0, 0]]
    reviews = [[[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 1]]]
    for t in range(3):
        records = [
            dict(
                frame_idx=np.asarray(100 + t, np.int64),
                pose_xyz=seed_pose() + np.array([t, rank, 0]),
                observed=np.ones((3, 3), bool),
                identity_review=np.array(reviews[t][rank], bool),
                parent=np.asarray(parents[t][rank], np.int64),
                costs=np.asarray(1 + rank, np.float64),
                path_costs=np.asarray(10 * t + rank, np.float64),
            )
            for rank in range(2)
        ]
        frames.append(
            dict(
                records=records,
                anchor_ids=np.array(anchor_ids[t], np.int64),
                fit=dict(
                    native=100 + t,
                    selected=np.array([2, 0], np.int64),
                    parent=np.array([parents[t][1], 0, parents[t][0]], np.int64),
                ),
            )
        )
    chosen = [frames[t]["records"][rank] for t, rank in enumerate([0, 1, 0])]
    candidate = {
        key: np.stack([row[key] for row in chosen])
        for key in ("frame_idx", "pose_xyz", "observed", "identity_review")
    }
    return (
        {arm: copy.deepcopy(frames) for arm in m.ARMS},
        {arm: copy.deepcopy(candidate) for arm in m.ARMS},
    )


def test_all_retained_rows_follow_parent_review_and_keep_causal_vs_final_anchor_counts():
    trees, candidate = history_fixture()
    result = m.retained_history(trees, candidate, seed_pose())
    for arm in m.ARMS:
        values, row = result["arrays"][arm], result["summary"][arm]
        np.testing.assert_array_equal(values["history_offsets"], [0, 2, 4, 6])
        np.testing.assert_array_equal(values["parent_flat_index"], [-1, -1, 1, 0, 3, 3])
        np.testing.assert_array_equal(values["selected_candidate_index"], [2, 0, 2, 0, 2, 0])
        np.testing.assert_array_equal(values["final_backtrace_flat_index"], [0, 3, 4])
        assert row["all_retained_anchor_rows"] == [4, 2]
        assert row["causal_best_anchor_native_counts"] == [2, 1]
        assert row["final_backtrace_anchor_native_counts"] == [3, 0]
        assert row["review_by_body"] == [4, 1, 1]
        assert row["parent_review_clear_by_body"] == [0, 0, 0]
        assert row["review_parent_edges"] == 4 and row["retained_body_times"] == 18
        assert values["segment_length_mm"].shape == (6, 3, 2)
    before = trees["original"][0]["records"][0]["pose_xyz"].copy()
    result["arrays"]["original"]["pose_xyz"][:] = -99
    np.testing.assert_array_equal(trees["original"][0]["records"][0]["pose_xyz"], before)


def test_review_clear_is_reported_on_actual_parent_edge_without_changing_flags():
    trees, candidate = history_fixture()
    trees["lk"][2]["records"][1]["identity_review"][0] = False
    result = m.retained_history(trees, candidate, seed_pose())
    assert result["summary"]["lk"]["parent_review_clear_by_body"] == [1, 0, 0]
    assert not trees["lk"][2]["records"][1]["identity_review"][0]


def test_six_retained_rows_are_all_counted_without_filtering_to_final_lineage():
    trees, candidate = history_fixture()
    for arm in m.ARMS:
        records = trees[arm][0]["records"]
        rows = [copy.deepcopy(records[i % 2]) for i in range(6)]
        for rank, row in enumerate(rows):
            row["pose_xyz"] += rank * 20
            row["identity_review"][:] = rank == 5
        trees[arm] = [
            dict(
                records=rows,
                anchor_ids=np.arange(6, dtype=np.int64),
                fit=dict(
                    native=100, selected=np.arange(6, dtype=np.int64), parent=np.zeros(6, np.int64)
                ),
            )
        ]
        candidate[arm] = {
            k: np.stack([rows[0][k]])
            for k in ("frame_idx", "pose_xyz", "observed", "identity_review")
        }
    result = m.retained_history(trees, candidate, seed_pose())
    assert result["summary"]["lk"]["retained_rows"] == 6
    assert result["summary"]["lk"]["all_retained_anchor_rows"] == [1] * 6
    assert result["summary"]["lk"]["review_by_body"] == [1] * 3
    assert result["summary"]["lk"]["final_backtrace_anchor_native_counts"] == [1, 0, 0, 0, 0, 0]
    assert result["arrays"]["lk"]["pose_xyz"].shape == (6, 3, 3, 3)


@pytest.mark.parametrize("bad", ["anchor", "parent", "selected", "backtrace", "native"])
def test_history_link_or_candidate_drift_fails_instead_of_selecting_other_rows(bad):
    trees, candidate = history_fixture()
    if bad == "anchor":
        trees["lk"][1]["anchor_ids"][0] = 0
    elif bad == "parent":
        trees["lk"][1]["records"][0]["parent"] = np.asarray(4, np.int64)
    elif bad == "selected":
        trees["lk"][1]["fit"]["selected"][:] = 2
    elif bad == "backtrace":
        candidate["lk"]["pose_xyz"][0, 0, 0, 0] = -0.0
    else:
        trees["lk"][1]["fit"]["native"] += 1
    with pytest.raises(ValueError):
        m.retained_history(trees, candidate, seed_pose())


def packet_fixture(native, *, empty_available=False, zero_reported=False):
    groups = np.array([0, 0, 1, 1, 2], np.int64)
    member = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0]], bool)
    competing = member.copy()
    competing[4, 0] = True
    masks = np.zeros((2, 2, 3), bool)
    masks[0, 0, 0] = masks[1, 0, 1] = True
    votes = tuple(
        transport.transport_votes(
            groups,
            np.zeros(5, bool) if zero_reported else np.ones(5, bool),
            membership,
            group_count=3,
            mask_available=(v < 2 or empty_available),
            native=native,
            release_horizon=115,
            query_sha256="a" * 64,
        )
        for v, membership in enumerate((member, competing, np.zeros((5, 0), bool)))
    )
    anchors = tuple(
        transport.InitialAnchor(
            a,
            100,
            tuple(
                np.eye(3, dtype=bool)[[a, 1 - a, 2]] & np.array([[1], [1], [0]], bool)
                for _ in range(3)
            ),
            ("{}",) * 3,
            ("b" * 64,) * 3,
        )
        for a in range(2)
    )
    px = np.tile(np.array([[0, 0], [1, 0], [1, 0], [2, 1]], np.int64)[None], (3, 1, 1))
    packet = transport.packet_for_pixels(
        votes,
        anchors,
        (masks, masks, np.zeros((0, 2, 3), bool) if empty_available else None),
        px,
        native_shape_hw=(2, 3),
    )
    return asdict(packet)


def transport_fixture(*, zero_reported=False):
    trees = {}
    for arm in m.ARMS:
        trees[arm] = []
        for t in range(3):
            packet = (
                None
                if not t or arm == "original"
                else packet_fixture(100 + t, empty_available=(t == 2), zero_reported=zero_reported)
            )
            trees[arm].append(
                dict(
                    anchor_ids=np.arange(2, dtype=np.int64),
                    fit=dict(
                        native=100 + t,
                        packet=packet,
                        fit_target=np.tile([1.0, 1.0, 1.0, 0.0], (3, 1)),
                        fit_weights=np.tile([0.0, 0.5, 0.5, 0.0], (3, 1)),
                    ),
                )
            )
    return trees


def test_transport_denominators_keep_unknown_and_empty_all_anchors_and_all_mask_cells():
    result = m.transport_diagnostics(
        transport_fixture(), frame_idx=np.arange(100, 103, dtype=np.int64)
    )
    original = result["summary"]["original"]
    assert original["packet_frames"] == 0 and original["disabled_packet_frames"] == 3
    assert original["ray_valid_used_views"] is None
    row = result["summary"]["lk"]
    assert row["evidence_group_view_times"] == 18
    assert row["anchor_group_view_times"] == 36
    assert row["anchor_group_mask_cells"] == 48
    assert row["anchor_body_view_times"] == 36
    assert row["positive_target_anchor_body_view_times"] == 12
    assert row["positive_pre_ray_anchor_body_view_times"] == 8
    ev = [r for r in result["evidence_rows"] if r["arm"] == "lk" and r["view"] == "side2_cam"]
    assert len(ev) == 6
    assert ev[0]["reported_count"] == 2 and ev[0]["single_count"] is None
    assert ev[0]["unassigned_count"] is None and ev[0]["available"] is False
    assert ev[3]["single_count"] == 0 and ev[3]["unassigned_count"] == 2
    assert ev[3]["available"] is True and ev[3]["current_mask_count"] == 0
    assert row["abstention_reason_counts"]["mask_has_unknown_group_votes"] == 4
    assert row["reasons_are_nonexclusive"] is True
    json.dumps(result, allow_nan=False)


def test_transport_raw_mass_and_pre_ray_views_do_not_drop_zeros_or_duplicate_draws():
    trees = transport_fixture()
    before = trees["lk"][1]["fit"]["packet"]["target"].tobytes()
    result = m.transport_diagnostics(trees, frame_idx=np.arange(100, 103, dtype=np.int64))
    rows = [
        r
        for r in result["body_view_rows"]
        if r["arm"] == "lk"
        and r["native"] == 101
        and r["anchor_id"] == 0
        and r["view"] == "top_cam"
    ]
    assert [r["target_sum"] for r in rows] == [1, 2, 0]
    assert [r["target_nonzero_draws"] for r in rows] == [1, 2, 0]
    assert [r["pre_ray_weighted_foreground_mass"] for r in rows] == [0, 1, 0]
    assert rows[0]["assigned_view"] and rows[0]["positive_target_view"]
    assert not rows[0]["positive_pre_ray_weighted_view"]
    assert all(r["ray_valid_used_view"] is None for r in rows)
    assert trees["lk"][1]["fit"]["packet"]["target"].tobytes() == before


def test_zero_reported_retains_every_zero_tie_with_null_fraction_and_no_targets():
    result = m.transport_diagnostics(
        transport_fixture(zero_reported=True), frame_idx=np.arange(100, 103, dtype=np.int64)
    )
    rows = result["group_mask_rows"]
    assert len(rows) == 96
    assert all(r["single_fraction_of_all_reported"] is None for r in rows)
    assert all(r["vote_maximum_tie"] and r["single_maximum_tie"] for r in rows)
    assert all(r["votes"] == r["single_votes"] == 0 for r in rows)
    assert result["summary"]["lk"]["positive_target_anchor_body_view_times"] == 0


@pytest.mark.parametrize(
    "bad", ["packet_missing", "original_packet", "mass", "shared_evidence", "bank"]
)
def test_transport_missing_or_contradictory_saved_rows_are_rejected(bad):
    trees = transport_fixture()
    packet = trees["lk"][1]["fit"]["packet"]
    if bad == "packet_missing":
        trees["lk"][1]["fit"]["packet"] = None
    elif bad == "original_packet":
        trees["original"][1]["fit"]["packet"] = packet
    elif bad == "mass":
        packet["target_mass"][0, 0, 0] += 1
    elif bad == "shared_evidence":
        packet["assignments"][1][0]["evidence"]["reported_count"][0] += 1
    else:
        packet["anchor_ids"][1] = 2
    with pytest.raises(ValueError):
        m.transport_diagnostics(trees, frame_idx=np.arange(100, 103, dtype=np.int64))


@pytest.mark.parametrize("bad", ["missing_arm", "native_dtype", "native_gap", "native_other_arm"])
def test_candidate_scope_is_not_silently_repaired(bad):
    data = candidates(np.repeat(seed_pose()[None], 2, axis=0))
    if bad == "missing_arm":
        del data["ct_first"]
    elif bad == "native_dtype":
        data["original"]["frame_idx"] = data["original"]["frame_idx"].astype(np.int32)
    elif bad == "native_gap":
        data["original"]["frame_idx"][1] += 1
    else:
        data["lk"]["frame_idx"] += 1
    with pytest.raises(ValueError):
        m.candidate_geometry(data, seed_pose())
