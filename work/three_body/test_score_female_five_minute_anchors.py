"""Synthetic-only qualification; never reads existing annotations or trajectories."""

import csv
import io

import numpy as np
import pytest
import score_female_five_minute_anchors as module


def record(frame=66889, **changes):
    row = dict(
        frame=str(frame),
        gt_status="confirmed",
        sex_label="F",
        side1_selection_status="sleap_linked",
        side2_selection_status="not_selected",
        side1_mouth_x="0",
        side1_mouth_y="0",
        side2_mouth_x="",
        side2_mouth_y="",
    )
    return row | changes


def csv_bytes(rows):
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def rows_and_anchors(n=3):
    frames = np.arange(66889, 66889 + n, dtype=np.int64)
    rows = dict(
        frame_idx=frames,
        pose_xyz=np.arange(n * 27, dtype=float).reshape(n, 3, 3, 3),
        sources=np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy(),
        observed=np.ones((n, 3, 3), bool),
        identity_review=np.zeros((n, 3), bool),
    )
    anchors = [module.generative.MouthAnchor(int(f), {"side1_cam": 1}, {}, {}) for f in frames]
    return rows, anchors


def mapping(n=4):
    status = np.full((n, 3), "finite_gap", dtype="U32")
    gap = np.ones((n, 3), float)
    status[0], gap[0] = "fixed_initial_constraint", np.inf
    state = np.array([0, 5, 0, 5])[:n]
    return dict(
        frame_idx=np.arange(module.START, module.START + n),
        path_state=state,
        permutation_by_id=module.PERMS[state].copy(),
        conditional_identity_alternative_status=status,
        conditional_identity_alternative_gap=gap,
    )


def test_fixed_57_and_missing_annotation_points_remain_in_denominator():
    assert len(module.FRAMES) == len(set(module.FRAMES)) == 57
    assert sum(f < 76360 for f in module.FRAMES) == 11
    result = module.selected_records(csv_bytes([record(f) for f in module.FRAMES]))
    assert len(result) == 57
    assert result[0]["side2_mouth_x"] == ""


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "not_female", "unconfirmed"])
def test_annotation_denominator_and_label_are_fixed(mutation):
    records = [record(f) for f in module.FRAMES]
    if mutation == "duplicate":
        records.append(records[0].copy())
    elif mutation == "missing":
        records.pop()
    elif mutation == "not_female":
        records[0]["sex_label"] = "M"
    else:
        records[0]["gt_status"] = "unreviewed"
    with pytest.raises(ValueError):
        module.selected_records(csv_bytes(records))


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (np.array([[5.0, 0.0]]), "matched"),
        (np.array([[np.nextafter(5.0, np.inf), 0.0]]), "outside_5px"),
        (np.array([[1.0, 0.0], [-1.0, 0.0]]), "ambiguous_nearest_tie"),
        (np.array([[1.0, 0.0], [1.000001, 0.0]]), "ambiguous_nearest_tie"),
        (np.empty((0, 2)), "no_finite_SL_mouth"),
        (np.full((2, 2), np.nan), "no_finite_SL_mouth"),
    ],
)
def test_frozen_nearest_row_threshold_ties_and_empty(points, expected):
    anchors, diagnostics = module.match_records([record()], lambda v, f: points)
    assert len(anchors) == 1
    assert diagnostics[0]["views"]["side1_cam"]["status"] == expected
    assert diagnostics[0]["views"]["side2_cam"]["status"] == "not_selected"
    assert bool(anchors[0].detection_rows) == (expected == "matched")


def test_unavailable_human_coordinate_is_not_filled_or_dropped():
    def forbidden(view, frame):
        raise AssertionError("no SLEAP read when annotation is unavailable")

    anchors, diagnostics = module.match_records([record(side1_mouth_x="")], forbidden)
    assert len(anchors) == 1 and anchors[0].detection_rows == {}
    assert diagnostics[0]["views"]["side1_cam"]["status"] == "annotation_mouth_unavailable"


def test_source_correspondence_not_nearest_3D_and_review_preservation():
    rows, anchors = rows_and_anchors()
    rows["identity_review"][1, 1] = True
    rows["pose_xyz"] *= 1e9
    before = {k: v.tobytes() for k, v in rows.items()}
    score = module.score_rows(rows, anchors)
    assert [a["identity"] for a in score["raw"]["anchors"]] == [1, 1, 1]
    assert score["with_original_review"]["available_anchors"] == 2
    assert score["with_original_review"]["interval_counts"]["unscored"] == 2
    assert all(v.tobytes() == before[k] for k, v in rows.items())


def test_latent_source_competitor_is_not_removed_to_make_unique_match():
    rows, anchors = rows_and_anchors()
    rows["sources"][0, 0, 1, 0] = 1
    rows["observed"][0, 0, 0] = False
    rows["observed"][1, 1, 0] = False
    rows["observed"][2, 1, 0] = False
    rows["pose_xyz"][2, 1, 0] = np.nan
    score = module.score_rows(rows, anchors)
    assert score["raw"]["available_anchors"] == 0
    assert score["raw"]["anchors"][0]["reason"] == "conflicting source identities"
    assert score["endpoint_details"][0]["duplicated_source_row"]
    assert score["endpoint_details"][1]["mouth_state_by_slot"][1] == "latent_finite"
    assert score["endpoint_details"][2]["mouth_state_by_slot"][1] == "nonfinite"
    assert score["raw"]["requested_anchors"] == 3


def test_cross_view_source_conflict_and_unmatched_keep_denominator():
    rows, anchors = rows_and_anchors()
    anchors[0].detection_rows["side2_cam"] = 2
    anchors[1].detection_rows.clear()
    rows["sources"][2, :, 1, 0] = -1
    score = module.score_rows(rows, anchors)
    assert score["raw"]["requested_anchors"] == 3
    assert score["raw"]["available_anchors"] == 0
    assert score["raw"]["requested_intervals"] == 2
    assert score["raw"]["interval_counts"] == dict(preserved=0, swap=0, unscored=2)


def test_swap_return_counts_endpoint_intervals_and_whole_fish_reorder():
    rows, anchors = rows_and_anchors()
    rows["identity_review"][:, 0] = True
    p = module.PERMS[[0, 2, 0]]
    reordered = module.reorder_rows(rows, p)
    score = module.score_rows(reordered, anchors)
    assert [a["identity"] for a in score["raw"]["anchors"]] == [1, 0, 1]
    assert score["raw"]["interval_counts"] == dict(preserved=0, swap=2, unscored=0)
    assert score["with_original_review"]["available_anchors"] == 3
    for key in module.FIELDS[1:]:
        assert np.array_equal(reordered[key][1], rows[key][1, p[1]])
        assert not np.shares_memory(reordered[key], rows[key])


def test_valid_initial_constraint_and_pruned_unknown_remain_distinct():
    arrays = mapping()
    arrays["conditional_identity_alternative_gap"][1, 1] = np.inf
    arrays["conditional_identity_alternative_status"][1, 1] = "not_retained_unknown"
    arrays["conditional_identity_alternative_gap"][2, 0] = 1e-6
    arrays["conditional_identity_alternative_status"][2, 0] = "tie_within_tolerance"
    module.validate_mapping(arrays, n=4, tolerance=1e-5)


@pytest.mark.parametrize("mutation", ["frame_gap", "initial", "not_bijective", "wrong_status"])
def test_mapping_tampering_rejected(mutation):
    arrays = mapping()
    if mutation == "frame_gap":
        arrays["frame_idx"][2] += 1
    elif mutation == "initial":
        arrays["path_state"][0] = 5
        arrays["permutation_by_id"][0] = module.PERMS[5]
    elif mutation == "not_bijective":
        arrays["permutation_by_id"][2] = [0, 0, 2]
    else:
        arrays["conditional_identity_alternative_status"][2, 1] = "no_feasible_alternative"
    with pytest.raises(ValueError):
        module.validate_mapping(arrays, n=4)


def test_execution_stays_disabled_without_any_output(tmp_path):
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="source-only"):
        module.run(output)
    assert not output.exists()
