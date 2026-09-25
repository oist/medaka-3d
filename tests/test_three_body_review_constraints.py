"""Endpoint-only human links preserve ambiguity, separate scopes and whole conflicts.

All human data below are explicitly synthetic in-memory fixtures. These tests
never create a human review file or modify a trajectory.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_three_body_review_evaluation import candidate, evidence, human_input

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
SPEC = importlib.util.spec_from_file_location(
    "review_constraints_tests", LAB / "review_constraints.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def inputs():
    review, sequence = human_input()
    sequence_bytes = json.dumps(sequence).encode()
    review["source_manifest_sha256"] = hashlib.sha256(sequence_bytes).hexdigest()
    return review, sequence_bytes, candidate(), evidence()["projected_mouth"]


def extract(review=None, sequence=None, arrays=None, projected=None, **overrides):
    original = inputs()
    review = original[0] if review is None else review
    sequence = original[1] if sequence is None else sequence
    arrays = original[2] if arrays is None else arrays
    projected = original[3] if projected is None else projected
    origins = {"candidate_sha256": "c"*64, "calibration_sha256": {"manifest": "d"*64},
               "projection_origin_sha256": "e"*64}
    origins.update(overrides)
    return MOD.extract_review_constraints(json.dumps(review).encode(), sequence, arrays,
                                          projected, **origins)


def only_a(review, first="confirmed", second="unreviewed"):
    for animal, state in review["continuity"].items():
        state["entry_to_after"] = first if animal == "A" else "ambiguous"
        state["after_to_exit"] = second if animal == "A" else "unreviewed"


def test_confirmed_raw_links_can_reference_reviewed_endpoints_without_clearing_flags():
    review, sequence, arrays, projected = inputs()
    arrays["identity_review"][:] = True
    before = copy.deepcopy(arrays)
    bundle = extract(review, sequence, arrays, projected)
    assert len(bundle["links"]) == 6 and len(bundle["endpoints"]) == 9
    assert all(link["status"] == "confirmed_endpoint_link" for link in bundle["links"])
    assert all(item["raw_identity_review"] for item in bundle["endpoints"])
    assert all(item["accepted_match"]["slot"] is None for item in bundle["endpoints"])
    assert bundle["human_review_metadata"]["status"] == review["status"]
    assert not bundle["review_flags_cleared"] and not bundle["dense_trajectory_relabelled"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    json.dumps(bundle, allow_nan=False)


def test_middle_returning_change_keeps_two_endpoint_links_without_dense_or_switch_time_claim():
    review, sequence, arrays, projected = inputs()
    projected[1] = projected[1, [1, 0, 2]]
    bundle = extract(review, sequence, arrays, projected)
    a = [link for link in bundle["links"] if link["anonymous_label"] == "A"]
    assert [(link["left_raw_slot"], link["right_raw_slot"]) for link in a] == [(0, 1), (1, 0)]
    assert all(
        not link["switch_time_known"] and not link["interior_identity_certified"] for link in a
    )
    assert all(link["edge"] != "entry_to_exit" for link in bundle["links"])
    combined = MOD.combine_constraints([bundle])
    assert len(combined["components"]) == 3 and not combined["counts"]["conflict"]


def test_partial_confirmation_never_assigns_third_fish_by_elimination():
    review, sequence, arrays, projected = inputs()
    only_a(review)
    bundle = extract(review, sequence, arrays, projected)
    combined = MOD.combine_constraints([bundle])
    assert combined["counts"]["requested_links"] == 6
    assert combined["counts"]["confirmed_endpoint_link"] == 1
    assert combined["counts"]["human_unconfirmed"] == 5
    assert len(combined["components"]) == 1
    assert len(combined["components"][0]["endpoint_ids"]) == 2
    assert not combined["identity_by_elimination"]


def test_confirmed_later_link_is_local_even_if_entry_chain_is_unconfirmed():
    review, sequence, arrays, projected = inputs()
    only_a(review, first="ambiguous", second="confirmed")
    bundle = extract(review, sequence, arrays, projected)
    link = next(item for item in bundle["links"] if item["status"] == "confirmed_endpoint_link")
    assert link["edge"] == "after_to_exit" and not link["chain_to_entry_confirmed"]
    assert bundle["shared_raw_interval_counts_unchanged"]["unscored"] == 6


@pytest.mark.parametrize("failure", ["unsupported", "latent_competitor", "cross_view", "far"])
def test_unsupported_nonunique_or_crossview_inconsistent_endpoint_remains_unresolved(failure):
    review, sequence, arrays, projected = inputs()
    only_a(review, second="confirmed")
    if failure == "unsupported":
        arrays["observed"][1, 0, 0] = False
        arrays["sources"][1, 0, :, 0] = -1
    elif failure == "latent_competitor":
        projected[1, 1] = projected[1, 0]
        arrays["observed"][1, 1, 0] = False
        arrays["identity_review"][1, 1] = True
    elif failure == "cross_view":
        projected[1, :, 1] = projected[1, [1, 0, 2], 1]
    else:
        projected[1, 0, :, 0] += 5.00001
    bundle = extract(review, sequence, arrays, projected)
    a = [item for item in bundle["links"] if item["anonymous_label"] == "A"]
    assert all(item["status"] == "endpoint_unresolved" for item in a)
    assert len(bundle["links"]) == 6
    combined = MOD.combine_constraints([bundle])
    assert combined["counts"]["endpoint_unresolved"] == 2
    assert not combined["components"]


def test_missing_candidate_endpoint_frame_is_preserved_instead_of_filled():
    review, sequence, arrays, projected = inputs()
    arrays = {key: value[:1] for key, value in arrays.items()}
    bundle = extract(review, sequence, arrays, projected[:1])
    assert all(item["status"] == "endpoint_unresolved" for item in bundle["links"])
    assert sum(item["endpoint_id"] is None for item in bundle["endpoints"]) == 6
    assert any(item["raw_match"]["reason"] == "candidate frame unavailable"
               for item in bundle["endpoints"])


@pytest.mark.parametrize("failure", ["template", "proposed_probe", "manifest_hash", "native_size"])
def test_unreviewed_template_or_wrong_native_provenance_is_rejected(failure):
    review, sequence, arrays, projected = inputs()
    if failure == "template":
        review["status"] = "unreviewed_template_not_ground_truth"
    elif failure == "proposed_probe":
        review["anchors"][1]["frame_choice"] = "proposed_not_human_confirmed"
    elif failure == "manifest_hash":
        review["source_manifest_sha256"] = "0"*64
    else:
        changed = json.loads(sequence)
        changed["image_shape_hw"] = [600, 600]
        sequence = json.dumps(changed).encode()
        review["source_manifest_sha256"] = hashlib.sha256(sequence).hexdigest()
    with pytest.raises(ValueError):
        extract(review, sequence, arrays, projected)


def test_source_collision_and_missing_second_direct_view_fail_loudly():
    review, sequence, arrays, projected = inputs()
    arrays["sources"][:, 1, 0, 0] = arrays["sources"][:, 0, 0, 0]
    with pytest.raises(ValueError, match="exclusive"):
        extract(review, sequence, arrays, projected)
    arrays = candidate()
    arrays["sources"][:, :, 1:, 0] = -1
    with pytest.raises(ValueError, match="at least two source"):
        extract(review, sequence, arrays, projected)


@pytest.mark.parametrize("change", [
    {"candidate_sha256": "1"*64}, {"calibration_sha256": {"manifest": "2"*64}},
    {"projection_origin_sha256": "3"*64},
])
def test_different_candidate_calibration_or_projector_origins_cannot_be_merged(change):
    with pytest.raises(ValueError, match="origin mismatch"):
        MOD.combine_constraints([extract(), extract(**change)])


def test_same_A_in_another_document_does_not_connect_disjoint_endpoint_pairs():
    review, sequence, arrays, projected = inputs()
    only_a(review)
    first = extract(review, sequence, arrays, projected)
    other = copy.deepcopy(review)
    other["reviewer"] = "different_synthetic_reviewer"
    for probe in range(3):
        other["anchors"][probe]["animals"]["A"]["points"] = copy.deepcopy(
            review["anchors"][probe]["animals"]["C"]["points"]
        )
        other["anchors"][probe]["animals"]["C"] = {"visibility": "unreviewed", "points": {}}
    second = extract(other, sequence, arrays, projected)
    assert first["clip_scope_id"] != second["clip_scope_id"]
    combined = MOD.combine_constraints([first, second])
    assert len(combined["components"]) == 2
    assert not combined["anonymous_names_merged_across_documents"]


def test_conflicting_connections_withhold_entire_component_and_dependent_later_links():
    review, sequence, arrays, projected = inputs()
    only_a(review)
    # A: slot0(entry) -> slot1(after).
    review["anchors"][1]["animals"]["A"]["points"] = copy.deepcopy(
        review["anchors"][1]["animals"]["B"]["points"]
    )
    review["anchors"][1]["animals"]["B"] = {"visibility": "unreviewed", "points": {}}
    first = extract(review, sequence, arrays, projected)
    # Another review conflicts: slot0(entry) -> slot2(after) -> slot2(exit).
    other, _, _, _ = inputs()
    only_a(other, second="confirmed")
    other["reviewer"] = "second_synthetic_reviewer"
    for probe in (1, 2):
        other["anchors"][probe]["animals"]["A"]["points"] = copy.deepcopy(
            other["anchors"][probe]["animals"]["C"]["points"]
        )
        other["anchors"][probe]["animals"]["C"] = {"visibility": "unreviewed", "points": {}}
    second = extract(other, sequence, arrays, projected)
    combined = MOD.combine_constraints([first, second])
    assert combined["counts"]["conflicting_components"] == 1
    assert combined["counts"]["conflict"] == 3
    assert combined["counts"]["confirmed_endpoint_link"] == 0
    assert combined["components"][0]["conflicts"][0]["slots"] == [1, 2]
    assert combined["counts"]["requested_links"] == 12
    assert first["links"][0]["status"] == "confirmed_endpoint_link"  # Input unmodified.


def test_same_claimed_endpoint_with_changed_candidate_content_conflicts_not_last_wins():
    first = extract()
    review, sequence, arrays, projected = inputs()
    review["reviewer"] = "another_synthetic_reviewer"
    arrays["pose_xyz"][1, 0, 0, 0] += 0.1
    second = extract(review, sequence, arrays, projected)
    combined = MOD.combine_constraints([first, second])
    assert combined["counts"]["conflict"] == 12
    assert combined["counts"]["conflicting_components"] == 3
    assert all(any(c["kind"] == "endpoint_source_or_projection_content_disagrees"
                   for c in component["conflicts"]) for component in combined["components"])


def test_bundle_mutation_is_rejected_and_identical_document_not_double_counted():
    original = extract()
    altered = copy.deepcopy(original)
    altered["links"][0]["right_raw_slot"] = 2
    with pytest.raises(ValueError, match="content/schema changed"):
        MOD.combine_constraints([original, altered])
    combined = MOD.combine_constraints([original, original])
    assert combined["counts"]["distinct_documents"] == 1
    assert combined["counts"]["requested_links"] == 6
