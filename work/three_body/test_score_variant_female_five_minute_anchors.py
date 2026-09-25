"""Synthetic-only tests for variant female57 metadata and unchanged endpoint rules."""

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import score_variant_female_five_minute_anchors as module


def reference(with_tracking=True):
    value = dict(
        path="final.npz",
        sha256="a" * 64,
        milestone_path="milestone.npz",
        milestone_sha256="b" * 64,
        terminal_horizon=116200,
        arrays={str(k): dict(sha256="c" * 64, dtype="<i8", shape=[3]) for k in range(21)},
    )
    if with_tracking:
        value["tracking_seal_sha256"] = "d" * 64
    return value


def saved_labels():
    frames = list(module.prior.FRAMES)
    fixed = dict(
        frames=frames,
        CSV_sha256=module.prior.GT_SHA,
        source_interval=[module.START, module.STOP],
        labels_created=False,
    )
    matches = [
        dict(
            frame=f,
            detection_rows={"side1_cam": 1},
            pixels={"side1_cam": [10.0, 20.0]},
            match_errors_px={"side1_cam": 0.5},
        )
        for f in frames
    ]
    return fixed, dict(matches=matches, diagnostics=[dict(frame=f) for f in frames])


def three_rows():
    frames = np.array([66889, 68186, 69363], dtype=np.int64)
    rows = dict(
        frame_idx=frames,
        pose_xyz=np.arange(81, dtype=float).reshape(3, 3, 3, 3),
        sources=np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy(),
        observed=np.ones((3, 3, 3), bool),
        identity_review=np.zeros((3, 3), bool),
    )
    anchors = [
        module.prior.generative.MouthAnchor(int(f), {"side1_cam": 1}, {}, {}) for f in frames
    ]
    mapping = dict(
        permutation_by_id=module.prior.PERMS[[0, 2, 0]],
        conditional_identity_alternative_status=np.full((3, 3), "finite_gap", dtype="U32"),
        conditional_identity_alternative_gap=np.ones((3, 3), float),
    )
    return rows, anchors, mapping


def test_legacy_sparse6_fields_bind_new7_via_explicit_tracking_seal():
    module.same_candidate_reference(
        reference(False), reference(), legacy_sparse=True, tracking_sha="d" * 64
    )
    module.same_candidate_reference(reference(), reference())


@pytest.mark.parametrize("field", ["path", "sha256", "milestone_sha256", "arrays", "tracking"])
def test_candidate_binding_rejects_changes_to_original_even_if_copy_is_different(field):
    actual, expected = reference(False), reference()
    tracking = "d" * 64
    if field == "tracking":
        tracking = "e" * 64
    elif field == "arrays":
        actual[field]["2"]["sha256"] = "e" * 64
    else:
        actual[field] = "changed"
    with pytest.raises(ValueError):
        module.same_candidate_reference(actual, expected, legacy_sparse=True, tracking_sha=tracking)


def test_missing_tracking_metadata_cannot_pass_by_erasing_seventh_field():
    with pytest.raises(ValueError):
        module.same_candidate_reference(reference(False), reference(), legacy_sparse=True)
    with pytest.raises(ValueError):
        module.same_candidate_reference(
            reference(False), reference(False), legacy_sparse=True, tracking_sha="d" * 64
        )


def test_saved57_retains_no_match_anchor_without_synthetic_filling():
    fixed, observations = saved_labels()
    observations["matches"][16].update(detection_rows={}, pixels={}, match_errors_px={})
    anchors = module.validate_saved_anchors(fixed, observations)
    assert len(anchors) == 57 and anchors[16].detection_rows == {}


@pytest.mark.parametrize(
    "mutation",
    ["frame_order", "missing_frame", "CSV", "outside5", "negative_row", "view", "created_label"],
)
def test_saved_annotation_denominator_binding_and_local_row_guards(mutation):
    fixed, observations = saved_labels()
    if mutation == "frame_order":
        observations["matches"][:2] = observations["matches"][1::-1]
    elif mutation == "missing_frame":
        observations["matches"].pop()
    elif mutation == "CSV":
        fixed["CSV_sha256"] = "e" * 64
    elif mutation == "outside5":
        observations["matches"][0]["match_errors_px"]["side1_cam"] = np.nextafter(5.0, np.inf)
    elif mutation == "negative_row":
        observations["matches"][0]["detection_rows"]["side1_cam"] = -1
    elif mutation == "view":
        observations["matches"][0]["pixels"]["other_cam"] = [1, 2]
    else:
        fixed["labels_created"] = True
    with pytest.raises(ValueError):
        module.validate_saved_anchors(fixed, observations)


def test_bracket_is_secondary_and_does_not_bridge_primary_unknown():
    rows, anchors, mapping = three_rows()
    anchors[1].detection_rows.clear()
    anchors[2].detection_rows["side1_cam"] = 0
    before = {k: v.tobytes() for k, v in rows.items()}
    result = module.evaluate(rows, anchors, mapping)
    assert result["raw_slots"]["raw"]["interval_counts"] == dict(preserved=0, swap=0, unscored=2)
    brackets = result["secondary_label_change_brackets"]["raw_slots"]["raw"]
    assert len(brackets) == 1 and brackets[0]["unavailable_native_frames"] == [68186]
    assert brackets[0]["left_available"]["identity"] == 1
    assert brackets[0]["right_available"]["identity"] == 0
    assert brackets[0]["event_time_localized"] is False
    assert brackets[0]["physical_swap_count_inferred"] is False
    assert all(v.tobytes() == before[k] for k, v in rows.items())


def test_raw_to_ID_reorder_preserves_node_and_review_and_conditional_unknown():
    rows, anchors, mapping = three_rows()
    rows["observed"][1, 1, 2] = False
    rows["identity_review"][1, 1] = True
    mapping["conditional_identity_alternative_status"][1, 0] = "not_retained_unknown"
    mapping["conditional_identity_alternative_gap"][1, 0] = np.inf
    result = module.evaluate(rows, anchors, mapping)
    automatic = result["automatic_IDs"]
    assert [a["identity"] for a in automatic["raw"]["anchors"]] == [1, 0, 1]
    assert automatic["with_original_review"]["available_anchors"] == 2
    detail = automatic["endpoint_details"][1]
    assert detail["original_review_by_slot"] == [True, False, False]
    assert detail["conditional_ID_alternative_status"][0] == "not_retained_unknown"
    assert detail["conditional_ID_alternative_gap"][0] == "+Infinity"
    assert detail["initial_ID1_sex_certified"] is False


def test_brackets_do_not_invent_change_for_same_label_or_missing_boundaries():
    rows = [
        dict(frame=1, identity=None),
        dict(frame=2, identity=1),
        dict(frame=3, identity=None),
        dict(frame=4, identity=1),
        dict(frame=5, identity=None),
    ]
    assert module.label_change_brackets(dict(anchors=rows)) == []


def test_permitted_execution_flag_only_is_reversible():
    old = b"HEADER\nEXECUTION_READY = False\nTAIL\n"
    new = old.replace(b"False", b"True")
    assert module.reviewed_flag_bytes(old) == module.reviewed_flag_bytes(new) == old
    assert (
        hashlib.sha256(module.reviewed_flag_bytes(new.replace(b"TAIL", b"CHANGED"))).digest()
        != hashlib.sha256(old).digest()
    )
    with pytest.raises(ValueError):
        module.reviewed_flag_bytes(old + old)


def test_entrypoint_disabled_before_sources_or_payloads(tmp_path):
    args = SimpleNamespace(output=tmp_path / "not_created")
    with pytest.raises(ValueError, match="source-only"):
        module.run(args)
    assert not args.output.exists()


def test_saved_labels_validator_does_not_modify_inputs():
    fixed, observations = saved_labels()
    saved = copy.deepcopy((fixed, observations))
    module.validate_saved_anchors(fixed, observations)
    assert (fixed, observations) == saved
