"""Synthetic, in-memory human links: no review files or real labels are created."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from test_three_body_review_evaluation import human_input

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
from review_constraints import extract_review_constraints  # noqa: E402

CONFIG = {"review_gap_frames": 4, "ambiguity_margin": 1.0}
CANDIDATE_SHA = "c" * 64


@pytest.fixture(scope="module")
def core():
    spec = importlib.util.spec_from_file_location(
        "three_body_review_correction_tests", LAB / "review_correction.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def geometry(count=40):
    frames = np.arange(100, 100 + count, dtype=np.int64)
    pose = np.zeros((count, 3, 3, 3), float)
    pose[..., 0] = 10 + 30 * np.arange(3)[None, :, None] + 0.01 * np.arange(count)[:, None, None]
    pose[..., 1] = 10 + np.arange(3)[None, None, :]
    pose[..., 2] = 20
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], (count, 3, 3, 3)).copy()
    return dict(
        frame_idx=frames,
        pose_xyz=pose,
        observed=np.ones((count, 3, 3), bool),
        sources=sources.astype(np.int32),
        identity_review=np.zeros((count, 3), bool),
        body_supported=np.ones((count, 3), bool),
        mask_supported=np.zeros((count, 3), bool),
        missing=np.zeros((count, 3), np.int64),
        score_margin=np.full(count, np.inf),
        xyz=pose[:, :, 1].copy(),
        accepted_xyz=pose[:, :, 1].copy(),
    )


def loss(arrays, lo, hi, slot=0):
    """Native-array indices, actual body loss; keep latent coordinates finite."""
    arrays["body_supported"][lo:hi, slot] = False
    arrays["observed"][lo:hi, slot] = False
    arrays["sources"][lo:hi, slot] = -1
    arrays["missing"][lo:hi, slot] = np.arange(1, hi - lo + 1)
    if hi - lo >= 4:
        arrays["identity_review"][lo + 3 :, slot] = True
    refresh_export(arrays)


def refresh_export(arrays):
    arrays["xyz"] = np.where(arrays["observed"][:, :, 1, None], arrays["pose_xyz"][:, :, 1], np.nan)
    arrays["accepted_xyz"] = np.where(arrays["identity_review"][:, :, None], np.nan, arrays["xyz"])


def projections(arrays):
    mouth = arrays["pose_xyz"][:, :, 0, :2] * 10
    return np.stack([mouth + view for view in range(3)], axis=2)


def bundle(
    arrays,
    anchors=(105, 112, 125),
    *,
    mappings=None,
    states=None,
    reviewer="synthetic_dense_unit_fixture",
    projected=None,
):
    """Use the real validated extractor; synthetic raw marks never go to disk."""
    review, sequence = human_input()
    sequence["frames"] = arrays["frame_idx"].tolist()
    manifest = json.dumps(sequence).encode()
    review["source_manifest_sha256"] = hashlib.sha256(manifest).hexdigest()
    review["source_frame_range"] = [sequence["frames"][0], sequence["frames"][-1] + 1]
    review["reviewer"] = reviewer
    projected = projections(arrays) if projected is None else projected
    mappings = ((0, 1, 2),) * 3 if mappings is None else mappings
    states = {"A": ("confirmed", "unreviewed")} if states is None else states
    for probe, anchor in enumerate(review["anchors"]):
        anchor["frame"] = anchors[probe]
        anchor["frame_choice"] = "human_selected"
        t = int(np.flatnonzero(arrays["frame_idx"] == anchors[probe])[0])
        for animal, label in enumerate(("A", "B", "C")):
            slot = mappings[probe][animal]
            anchor["animals"][label] = dict(
                visibility="visible",
                points={
                    view: projected[t, slot, v].tolist()
                    for v, view in enumerate(("top_cam", "side1_cam", "side2_cam"))
                },
            )
    for label, entry in review["continuity"].items():
        first, second = states.get(label, ("ambiguous", "unreviewed"))
        entry.update(entry_to_after=first, after_to_exit=second, basis="direct_visual_continuity")
    return extract_review_constraints(
        json.dumps(review).encode(),
        manifest,
        arrays,
        projected,
        candidate_sha256=CANDIDATE_SHA,
        calibration_sha256={"manifest": "d" * 64},
        projection_origin_sha256="e" * 64,
    )


def apply(core, data, bundles=()):
    return core.build_review_correction(
        data, list(bundles), tracker_config=CONFIG, candidate_sha256=CANDIDATE_SHA
    )


def mapping(result):
    return result["arrays"]["corrected_identity_to_raw_slot"]


def test_no_reviews_keep_only_conditional_initial_tracklets_until_first_risk(core):
    data = geometry()
    loss(data, 6, 11)
    result = apply(core, data)
    np.testing.assert_array_equal(mapping(result)[:6, 0], 0)
    np.testing.assert_array_equal(mapping(result)[6:, 0], -1)
    np.testing.assert_array_equal(mapping(result)[:, 1:], np.tile([1, 2], (40, 1)))
    assert np.all(result["arrays"]["identity_origin"][:6] == 1)


def test_same_slot_confirmed_endpoints_still_leave_open_interior_nan(core):
    data = geometry()
    loss(data, 6, 10)
    result = apply(core, data, [bundle(data)])
    assert mapping(result)[5, 0] == 0 and mapping(result)[12, 0] == 0
    np.testing.assert_array_equal(mapping(result)[6:12, 0], -1)
    assert np.isnan(result["arrays"]["corrected_node_xyz"][6:12, 0]).all()
    assert result["arrays"]["identity_origin"][12, 0] == 2
    assert result["arrays"]["identity_origin"][13, 0] == 3
    np.testing.assert_array_equal(mapping(result)[12:, 0], 0)
    assert data["identity_review"][12, 0]  # The original sticky flag remains true.


def test_partial_confirmed_swap_withholds_displaced_automatic_id_from_p_plus_one(core):
    data = geometry()
    claim = bundle(data, mappings=((0, 1, 2), (1, 0, 2), (1, 0, 2)))
    result = apply(core, data, [claim])
    np.testing.assert_array_equal(mapping(result)[:6], np.tile([0, 1, 2], (6, 1)))
    np.testing.assert_array_equal(mapping(result)[6:12, :2], -1)
    np.testing.assert_array_equal(mapping(result)[12:, 0], 1)
    np.testing.assert_array_equal(mapping(result)[6:, 1], -1)
    np.testing.assert_array_equal(mapping(result)[:, 2], 2)
    for row in mapping(result):
        selected = row[row >= 0]
        assert len(np.unique(selected)) == len(selected)


def test_two_returning_changes_preserve_middle_endpoint_without_filling_either_gap(core):
    data = geometry()
    claim = bundle(
        data,
        anchors=(105, 112, 120),
        mappings=((0, 1, 2), (1, 0, 2), (0, 1, 2)),
        states={"A": ("confirmed", "confirmed")},
    )
    result = apply(core, data, [claim])
    assert mapping(result)[5, 0] == 0
    assert mapping(result)[12, 0] == 1
    assert mapping(result)[20, 0] == 0
    assert np.all(mapping(result)[6:12, 0] == -1)
    assert np.all(mapping(result)[13:20, 0] == -1)
    assert np.all(mapping(result)[20:, 0] == 0)
    assert np.all(mapping(result)[6:, 1] == -1)


def test_later_recurring_risk_stops_propagation_despite_old_sticky_review(core):
    data = geometry()
    loss(data, 6, 10)
    loss(data, 18, 23)
    result = apply(core, data, [bundle(data)])
    np.testing.assert_array_equal(mapping(result)[12:18, 0], 0)
    np.testing.assert_array_equal(mapping(result)[18:, 0], -1)


def test_q_inside_overlapping_unknown_pair_risk_cannot_clear_then_auto_resume(core):
    data = geometry()
    loss(data, 6, 10)
    data["score_margin"][9:15] = 0.5
    result = apply(core, data, [bundle(data)])
    np.testing.assert_array_equal(mapping(result)[6:, 0], -1)
    np.testing.assert_array_equal(mapping(result)[9:, 1:], -1)


def test_q_at_half_open_risk_end_is_eligible_but_other_ids_stay_unresolved(core):
    data = geometry()
    data["score_margin"][6:12] = 0.5
    data["identity_review"][6:] = True
    refresh_export(data)
    result = apply(core, data, [bundle(data)])
    np.testing.assert_array_equal(mapping(result)[12:, 0], 0)
    np.testing.assert_array_equal(mapping(result)[6:, 1:], -1)


def test_initial_review_prevents_global_anchor_even_with_local_confirmed_links(core):
    data = geometry()
    data["identity_review"][:, 0] = True
    refresh_export(data)
    result = apply(core, data, [bundle(data)])
    np.testing.assert_array_equal(mapping(result)[:, 0], -1)
    assert np.all(result["arrays"]["identity_origin"][:, 0] == 0)
    np.testing.assert_array_equal(mapping(result)[:, 1:], np.tile([1, 2], (40, 1)))


def test_after_to_exit_alone_does_not_invent_chain_across_earlier_unresolved_risk(core):
    data = geometry()
    loss(data, 6, 10)
    result = apply(core, data, [bundle(data, states={"A": ("ambiguous", "confirmed")})])
    np.testing.assert_array_equal(mapping(result)[6:, 0], -1)


def test_confirmed_link_can_bridge_unknown_review_boundary_without_claiming_onset(core):
    data = geometry()
    data["identity_review"][8:, 0] = True
    refresh_export(data)
    result = apply(core, data, [bundle(data)])
    np.testing.assert_array_equal(mapping(result)[6:12, 0], -1)
    np.testing.assert_array_equal(mapping(result)[12:, 0], 0)


def test_target_slots_old_history_does_not_prevent_new_human_anchored_tracklet(core):
    data = geometry()
    data["identity_review"][:, 1] = True  # No global G1 anchor exists.
    refresh_export(data)
    claim = bundle(data, mappings=((0, 1, 2), (1, 0, 2), (1, 0, 2)))
    result = apply(core, data, [claim])
    np.testing.assert_array_equal(mapping(result)[12:, 0], 1)
    np.testing.assert_array_equal(mapping(result)[:, 1], -1)


def test_source_frame_gap_has_no_automatic_identity_bridge(core):
    data = geometry()
    data["frame_idx"][20:] += 100
    result = apply(core, data)
    np.testing.assert_array_equal(mapping(result)[:20], np.tile([0, 1, 2], (20, 1)))
    np.testing.assert_array_equal(mapping(result)[20:], -1)


def test_mask_only_support_and_short_gaps_keep_latent_but_not_measured_positions(core):
    data = geometry()
    loss(data, 6, 10)
    data["observed"][14:17, 0, :2] = False
    data["sources"][14:17, 0, 1:, :2] = -1
    data["mask_supported"][14:17, 0] = True
    loss(data, 20, 22)
    result = apply(core, data, [bundle(data)])
    out = result["arrays"]
    np.testing.assert_array_equal(mapping(result)[12:, 0], 0)
    assert np.isfinite(out["corrected_pose_xyz"][14:17, 0]).all()
    assert np.isnan(out["corrected_node_xyz"][14:17, 0, :2]).all()
    assert np.isnan(out["corrected_xyz"][14:17, 0]).all()
    assert np.isfinite(out["corrected_node_xyz"][14:17, 0, 2]).all()
    assert np.isnan(out["corrected_node_xyz"][20:22, 0]).all()
    np.testing.assert_array_equal(out["corrected_sources"][14:17, 0], data["sources"][14:17, 0])


def test_same_anonymous_A_across_documents_cannot_rename_another_global_identity(core):
    data = geometry()
    first = bundle(data)
    other = bundle(
        data, anchors=(115, 122, 130), mappings=((2, 1, 0),) * 3, reviewer="second_synthetic_scope"
    )
    result = apply(core, data, [first, other])
    assert first["clip_scope_id"] != other["clip_scope_id"]
    assert mapping(result)[12, 0] == 0
    assert mapping(result)[22, 2] == 2
    assert mapping(result)[22, 0] == 0
    assert np.all(mapping(result)[16:22, 2] == -1)


def test_human_conflict_blocks_both_branches_and_dependent_later_link(core):
    data = geometry()
    first = bundle(
        data, mappings=((0, 1, 2), (1, 0, 2), (1, 0, 2)), states={"A": ("confirmed", "confirmed")}
    )
    second = bundle(
        data,
        mappings=((0, 1, 2), (2, 1, 0), (2, 1, 0)),
        reviewer="contradictory_synthetic_reviewer",
    )
    result = apply(core, data, [first, second])
    assert mapping(result)[5, 0] == 0
    assert np.all(mapping(result)[6:, 0] == -1)
    assert result["arrays"]["identity_conflict"][12, 0]
    assert mapping(result)[25, 0] == -1


def test_document_order_and_duplicates_are_idempotent_and_inputs_do_not_alias(core):
    data = geometry()
    loss(data, 6, 10)
    first = bundle(data)
    second = bundle(
        data, anchors=(115, 122, 130), mappings=((2, 1, 0),) * 3, reviewer="second_synthetic_scope"
    )
    before_data, before_bundles = copy.deepcopy(data), copy.deepcopy([first, second])
    expected = apply(core, data, [first, second])
    for changed in ([second, first], [first, second, first]):
        actual = apply(core, data, changed)
        for key in expected["arrays"]:
            np.testing.assert_array_equal(actual["arrays"][key], expected["arrays"][key])
    for key, value in data.items():
        np.testing.assert_array_equal(value, before_data[key])
    assert [first, second] == before_bundles
    for output in expected["arrays"].values():
        for value in data.values():
            assert not np.shares_memory(output, value)
    expected["arrays"]["corrected_pose_xyz"][:] = 999
    np.testing.assert_array_equal(data["pose_xyz"], before_data["pose_xyz"])


@pytest.mark.parametrize("failure", ["unsupported", "latent_competitor"])
def test_failed_raw_unique_mouth_match_cannot_clear_review(core, failure):
    data = geometry()
    loss(data, 6, 10)
    projected = projections(data)
    if failure == "unsupported":
        data["observed"][12, 0, 0] = False
        data["sources"][12, 0, :, 0] = -1
    else:
        projected[12, 1] = projected[12, 0]
        data["observed"][12, 1, 0] = False
        data["sources"][12, 1, :, 0] = -1
    claim = bundle(data, projected=projected)
    result = apply(core, data, [claim])
    np.testing.assert_array_equal(mapping(result)[6:, 0], -1)


@pytest.mark.parametrize("failure", ["candidate_sha", "changed_arrays", "changed_bundle"])
def test_stale_or_different_candidate_evidence_cannot_be_applied(core, failure):
    data = geometry()
    claim = bundle(data)
    digest = CANDIDATE_SHA
    if failure == "candidate_sha":
        digest = "f" * 64
    elif failure == "changed_arrays":
        data["pose_xyz"][12, 0, 0, 0] += 0.1
    else:
        claim["links"][0]["right_raw_slot"] = 2
    with pytest.raises(ValueError):
        core.build_review_correction(data, [claim], tracker_config=CONFIG, candidate_sha256=digest)


def test_complete_three_cycle_reorders_all_original_fields_without_creating_observation(core):
    data = geometry()
    data["observed"][13:, 2, 2] = False
    data["sources"][13:, 2, :, 2] = -1
    data["identity_review"][8:] = True
    refresh_export(data)
    claim = bundle(
        data,
        mappings=((0, 1, 2), (1, 2, 0), (1, 2, 0)),
        states={label: ("confirmed", "unreviewed") for label in ("A", "B", "C")},
    )
    out = apply(core, data, [claim])["arrays"]
    np.testing.assert_array_equal(
        out["corrected_identity_to_raw_slot"][12:], np.tile([1, 2, 0], (28, 1))
    )
    for output, original in (
        ("corrected_pose_xyz", "pose_xyz"),
        ("corrected_sources", "sources"),
        ("corrected_observed", "observed"),
        ("prior_raw_review_for_identity", "identity_review"),
    ):
        np.testing.assert_array_equal(out[output][12:], data[original][12:, [1, 2, 0]])
    expected_nodes = np.where(
        data["observed"][12:, [1, 2, 0], :, None], data["pose_xyz"][12:, [1, 2, 0]], np.nan
    )
    np.testing.assert_array_equal(out["corrected_node_xyz"][12:], expected_nodes)
    np.testing.assert_array_equal(out["corrected_xyz"][12:], expected_nodes[:, :, 1])
    assert np.isnan(out["corrected_node_xyz"][13:, 1, 2]).all()
    assert np.isfinite(out["corrected_pose_xyz"][13:, 1, 2]).all()
    assert np.all(out["prior_raw_review_for_identity"][12:])
    assert np.all(out["claim_index"][12:] >= 0)
    assert np.all(out["tracklet_index"][12:] >= 0)
    unknown = out["corrected_identity_to_raw_slot"] < 0
    assert np.isnan(out["corrected_pose_xyz"][unknown]).all()
    assert np.all(out["corrected_sources"][unknown] == -1)
    assert not out["corrected_observed"][unknown].any()
    assert np.all(out["identity_origin"][unknown] == 0)


def test_endpoint_inside_active_risk_can_support_next_confirmed_endpoint_only(core):
    data = geometry()
    data["score_margin"][6:18] = 0.5
    data["identity_review"][6:] = True
    refresh_export(data)
    claim = bundle(data, states={"A": ("confirmed", "confirmed")})
    out = apply(core, data, [claim])["arrays"]
    np.testing.assert_array_equal(out["corrected_identity_to_raw_slot"][6:25, 0], -1)
    np.testing.assert_array_equal(out["corrected_identity_to_raw_slot"][25:, 0], 0)
    assert out["identity_origin"][25, 0] == 2
    assert out["identity_origin"][26, 0] == 3
    np.testing.assert_array_equal(out["corrected_identity_to_raw_slot"][6:, 1:], -1)


def test_retrospective_displacement_invalidates_dependent_human_lineage(core):
    data = geometry()
    earlier_b = bundle(
        data,
        anchors=(106, 114, 130),
        states={"B": ("confirmed", "confirmed")},
        reviewer="earlier_B_synthetic",
    )
    later_a = bundle(
        data,
        anchors=(105, 120, 135),
        mappings=((0, 1, 2), (1, 0, 2), (1, 0, 2)),
        reviewer="later_A_synthetic",
    )
    result = apply(core, data, [earlier_b, later_a])
    np.testing.assert_array_equal(mapping(result)[6:, 1], -1)
    np.testing.assert_array_equal(mapping(result)[20:, 0], 1)
    assert mapping(result)[14, 1] == -1
    assert mapping(result)[30, 1] == -1
    assert any(
        link["correction_status"] == "dependency_invalidated"
        for link in result["metadata"]["link_applications"]
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("review_gap_frames", 5),
        ("review_gap_frames", True),
        ("ambiguity_margin", 0.5),
        ("ambiguity_margin", np.nan),
    ],
)
def test_risk_contract_must_match_frozen_extractor(core, key, value):
    config = dict(CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        core.build_review_correction(
            geometry(), [], tracker_config=config, candidate_sha256=CANDIDATE_SHA
        )


def test_large_signed_source_row_is_preserved_without_int32_truncation(core):
    data = geometry()
    data["sources"] = data["sources"].astype(np.int64)
    data["sources"][:, 0] = 2**32
    result = apply(core, data)["arrays"]
    np.testing.assert_array_equal(result["corrected_sources"], data["sources"])


def test_unsigned_source_row_above_int64_range_is_rejected(core):
    data = geometry()
    data["sources"] = data["sources"].astype(np.uint64)
    data["sources"][:, 0] = 2**63
    with pytest.raises(ValueError):
        apply(core, data)
