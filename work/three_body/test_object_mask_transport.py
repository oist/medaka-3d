"""Finite synthetic transport qualification; no images, model, GPU or fits."""

import copy
import dataclasses
import inspect

import numpy as np
import object_mask_transport as m
import pytest

SHA = "a" * 64


def evidence(groups, membership, reported=None, *, count=None, available=True, native=101):
    group = np.asarray(groups, np.int64)
    member = np.asarray(membership, bool)
    return m.transport_votes(
        group,
        np.ones(len(group), bool) if reported is None else np.asarray(reported, bool),
        member,
        group_count=(int(group.max()) + 1 if len(group) else 0) if count is None else count,
        mask_available=available,
        native=native,
        release_horizon=native + 7,
        query_sha256=SHA,
    )


def refs(rows=(0, 1, -1), eligible=(True, True, False)):
    return dict(
        slots=[
            dict(
                slot=b,
                assigned_mask_row=r,
                geometry_eligible=e,
                reasons=[] if e else ["ambiguous_or_missing"],
            )
            for b, (r, e) in enumerate(zip(rows, eligible, strict=True))
        ]
    )


def anchor(rows=(0, 1, -1), eligible=(True, True, False), *, count=2, ident=0):
    return m.build_initial_anchor(
        [refs(rows, eligible) for _ in range(3)],
        [count] * 3,
        anchor_id=ident,
        initial_native=100,
    )


def packet_fixture():
    ev = evidence([0, 0, 1, 1], [[1, 0], [1, 0], [0, 1], [0, 1]])
    masks = np.array([[[1, 1, 0]], [[0, 1, 1]]], bool)
    pixels = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]), (3, 1, 1))
    return dict(
        votes_by_view=[ev] * 3,
        anchors=[anchor()],
        masks_by_view=[masks.copy() for _ in range(3)],
        pixels=pixels,
        native_shape_hw=(1, 3),
    )


def test_full_reported_denominator_keeps_multi_unassigned_and_nonreported_separate():
    ev = evidence([0] * 6, [[1, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]], [1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(ev.reported_count, [5])
    np.testing.assert_array_equal(ev.votes, [[3, 1]])
    np.testing.assert_array_equal(ev.single_votes, [[2, 0]])
    np.testing.assert_array_equal(ev.weights, [[0.4, 0.0]])
    np.testing.assert_array_equal(ev.single_count, [2])
    np.testing.assert_array_equal(ev.multi_count, [1])
    np.testing.assert_array_equal(ev.unassigned_count, [2])
    decision = m.assign_transport(ev, np.array([[1, 0, 0]], bool))
    assert decision.group_admitted.tolist() == [True]
    assert decision.body_weight.tolist() == [0.4, 0.0, 0.0]


def test_unknown_group_single_vote_vetoes_without_being_dropped():
    ev = evidence([0, 0, 1], [[1], [1], [1]])
    result = m.assign_transport(ev, np.array([[1, 0, 0], [0, 0, 0]], bool))
    assert not result.assignment.any()
    assert result.candidate_mask_row.tolist() == [0, 0]
    assert result.unknown_competing_group_mask.tolist() == [[True], [False]]
    assert "mask_has_unknown_group_votes" in result.group_reasons[0]
    assert "no_strict_initial_link" in result.group_reasons[1]
    np.testing.assert_array_equal(result.evidence.single_votes, [[2], [1]])


def test_unknown_multi_membership_also_vetoes_a_known_single_mask():
    ev = evidence([0, 0, 1], [[1, 0], [1, 0], [1, 1]])
    result = m.assign_transport(ev, np.array([[1, 0, 0], [0, 0, 0]], bool))
    assert not result.group_admitted.any()
    assert ev.single_count.tolist() == [2, 0]
    assert ev.votes.tolist() == [[2, 0], [1, 1]]
    assert result.competing_group_mask[0, 0]


def test_unreported_unknown_membership_cannot_veto():
    ev = evidence([0, 0, 1], [[1], [1], [1]], [1, 1, 0])
    result = m.assign_transport(ev, np.array([[1, 0, 0], [0, 0, 0]], bool))
    assert result.assignment.tolist() == [[True, False, False]]
    assert not result.competing_group_mask[0, 0]


def test_merged_known_groups_both_abstain_instead_of_awarding_largest():
    ev = evidence([0, 0, 0, 1, 1], [[1]] * 5)
    result = m.assign_transport(ev, np.array([[1, 0, 0], [0, 1, 0]], bool))
    assert not result.assignment.any()
    for why in result.group_reasons:
        assert "mask_has_other_linked_group_votes" in why
    assert result.candidate_mask_row.tolist() == [0, 0]


@pytest.mark.parametrize("second_count", [1, 2, 3])
def test_fragmentation_abstains_even_when_one_mask_is_unique_largest(second_count):
    ev = evidence([0] * (3 + second_count), [[1, 0]] * 3 + [[0, 1]] * second_count)
    result = m.assign_transport(ev, np.array([[1, 0, 0]], bool))
    assert result.group_reasons == (("split_positive_masks",),)
    assert not result.assignment.any()
    assert ev.single_maximum_ties.tolist() == [[True, second_count == 3]]
    assert result.candidate_mask_row.tolist() == [-1]


def test_zero_and_one_votes_are_not_promoted_and_zero_ties_are_preserved():
    ev = evidence([0, 1], [[1, 0], [0, 0]], [True, False], count=3)
    result = m.assign_transport(ev, np.eye(3, dtype=bool))
    assert not result.assignment.any()
    assert ev.reported_count.tolist() == [1, 0, 0]
    assert np.isnan(ev.weights[1:]).all()
    assert ev.vote_maximum_ties.tolist() == [[True, False], [True, True], [True, True]]
    assert "fewer_than_two_single_votes_in_mask" in result.group_reasons[0]
    assert all("fewer_than_two_reported_points" in x for x in result.group_reasons)


def test_unavailable_and_available_empty_are_distinct_not_silent_zero_measurements():
    absent = evidence([0, 0], np.empty((2, 0), bool), available=False)
    empty = evidence([0, 0], np.empty((2, 0), bool))
    for ev in (absent, empty):
        assert ev.reported_count.tolist() == [2]
        assert ev.weights.shape == (1, 0)
    assert (
        absent.single_count.tolist()
        == absent.multi_count.tolist()
        == absent.unassigned_count.tolist()
        == [-1]
    )
    assert absent.membership_degree.tolist() == [-1, -1]
    assert empty.single_count.tolist() == empty.multi_count.tolist() == [0]
    assert empty.unassigned_count.tolist() == [2]
    assert (
        "masks_unavailable"
        in m.assign_transport(absent, np.array([[1, 0, 0]], bool)).group_reasons[0]
    )
    assert (
        "no_current_masks"
        in m.assign_transport(empty, np.array([[1, 0, 0]], bool)).group_reasons[0]
    )


def test_zero_groups_and_zero_queries_keep_all_current_mask_columns():
    ev = evidence([], np.empty((0, 5), bool), count=0)
    result = m.assign_transport(ev, np.empty((0, 3), bool))
    assert ev.votes.shape == ev.weights.shape == (0, 5)
    assert result.assignment.shape == (5, 3)
    assert result.body_reasons == (("no_strict_initial_link",),) * 3


def test_votes_match_independent_point_enumeration_with_all_group_rows():
    # Deterministic sparse synthetic data, includes nonreported/multi and empty groups.
    rng = np.random.default_rng(21)
    groups = rng.integers(0, 4, 37, dtype=np.int32)
    reported = rng.integers(0, 2, 37).astype(bool)
    member = rng.integers(0, 5, (37, 7)) == 0
    actual = evidence(groups, member, reported, count=6)
    expected_votes = np.zeros((6, 7), np.int64)
    expected_single = np.zeros_like(expected_votes)
    denominator = np.zeros(6, np.int64)
    for q in range(37):
        if not reported[q]:
            continue
        denominator[groups[q]] += 1
        rows = [r for r in range(7) if member[q, r]]
        for r in rows:
            expected_votes[groups[q], r] += 1
            if len(rows) == 1:
                expected_single[groups[q], r] += 1
    np.testing.assert_array_equal(actual.reported_count, denominator)
    np.testing.assert_array_equal(actual.votes, expected_votes)
    np.testing.assert_array_equal(actual.single_votes, expected_single)


@pytest.mark.parametrize(
    "bad",
    [
        "group_dtype",
        "group_range",
        "negative_group",
        "report_dtype",
        "report_shape",
        "membership_dtype",
        "membership_shape",
        "availability",
        "unavailable_rows",
        "horizon",
        "native_bool",
        "sha",
        "group_count_bool",
    ],
)
def test_invalid_evidence_is_rejected(bad):
    kw = dict(
        query_group=np.array([0, 0]),
        reported=np.ones(2, bool),
        membership=np.ones((2, 1), bool),
        group_count=1,
        mask_available=True,
        native=100,
        release_horizon=107,
        query_sha256=SHA,
    )
    if bad == "group_dtype":
        kw["query_group"] = np.zeros(2, float)
    elif bad == "group_range":
        kw["query_group"][0] = 1
    elif bad == "negative_group":
        kw["query_group"][0] = -1
    elif bad == "report_dtype":
        kw["reported"] = np.ones(2, np.uint8)
    elif bad == "report_shape":
        kw["reported"] = np.ones((2, 1), bool)
    elif bad == "membership_dtype":
        kw["membership"] = np.ones((2, 1), int)
    elif bad == "membership_shape":
        kw["membership"] = np.ones((1, 2), bool)
    elif bad == "availability":
        kw["mask_available"] = np.bool_(True)
    elif bad == "unavailable_rows":
        kw["mask_available"] = False
    elif bad == "horizon":
        kw["release_horizon"] = 99
    elif bad == "native_bool":
        kw["native"] = True
    elif bad == "sha":
        kw["query_sha256"] = "A" * 64
    else:
        kw["group_count"] = True
    with pytest.raises(ValueError):
        m.transport_votes(**kw)


@pytest.mark.parametrize(
    "links",
    [
        np.array([[1, 1, 0]], bool),
        np.array([[1, 0, 0], [1, 0, 0]], bool),
        np.ones((1, 3), int),
        np.empty((0, 3), bool),
    ],
)
def test_inconsistent_initial_row_or_body_mapping_fails(links):
    ev = evidence([0, 0], [[1], [1]])
    with pytest.raises(ValueError):
        m.assign_transport(ev, links)


def test_two_distinct_initial_groups_cannot_both_claim_one_body():
    ev = evidence([0, 0, 1, 1], [[1, 0], [1, 0], [0, 1], [0, 1]])
    with pytest.raises(ValueError, match="body linked to multiple groups"):
        m.assign_transport(ev, np.array([[1, 0, 0], [1, 0, 0]], bool))


def test_initial_anchor_delegates_frozen_eligibility_and_preserves_ambiguity(monkeypatch):
    original = m.initial_link_matrix
    calls = []

    def observed(ref, count):
        calls.append((ref, count))
        return original(ref, count)

    monkeypatch.setattr(m, "initial_link_matrix", observed)
    source = [refs((2, 1, -1), (True, False, False)) for _ in range(3)]
    before = copy.deepcopy(source)
    a = m.build_initial_anchor(source, [4, 4, 4], anchor_id=5, initial_native=100)
    assert len(calls) == 3 and source == before
    assert all(x.shape == (4, 3) and x.sum() == 1 and x[2, 0] for x in a.links)
    saved = a.references_json
    source[0]["slots"][0]["assigned_mask_row"] = 0
    assert a.links[0][2, 0] and a.references_json == saved


def test_current_single_view_reference_cannot_be_relabelled_strict_initial():
    data = refs()
    data["scope"] = "single_current_view_geometry"
    with pytest.raises(ValueError, match="current-view"):
        m.build_initial_anchor([data] * 3, [2] * 3, anchor_id=0, initial_native=100)


def test_target_excludes_pixel_overlap_preserves_duplicate_draws_and_body_axis():
    kw = packet_fixture()
    kw["anchors"].append(anchor(rows=(1, 0, -1), ident=3))
    packet = m.packet_for_pixels(**kw)
    assert packet.target.shape == (2, 3, 4, 3)
    assert packet.anchor_ids.tolist() == [0, 3]
    expected = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    for v in range(3):
        np.testing.assert_array_equal(packet.target[0, v], expected)
        np.testing.assert_array_equal(packet.target[1, v], expected[:, [1, 0, 2]])
    assert packet.sampled_mask_degree.tolist() == [[1, 2, 1, 1]] * 3
    assert packet.target_mass[0].tolist() == [[2.0, 1.0, 0.0]] * 3
    assert packet.target_nonzero_count[0].tolist() == [[2, 1, 0]] * 3
    assert np.all((packet.target > 0).sum(axis=-1) <= 1)


def test_overlap_with_unknown_mask_is_excluded_even_if_unknown_has_no_query_votes():
    kw = packet_fixture()
    ev = evidence([0, 0], [[1, 0], [1, 0]], count=2)
    kw["votes_by_view"] = [ev] * 3
    kw["anchors"] = [anchor(eligible=(True, False, False))]
    packet = m.packet_for_pixels(**kw)
    assert packet.assignments[0][0].group_admitted.tolist() == [True, False]
    assert packet.target[0, 0, :, 0].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_packet_preserves_none_empty_and_unknown_sample_degree_with_zero_target():
    kw = packet_fixture()
    absent = evidence([0, 0], np.empty((2, 0), bool), count=2, available=False)
    empty = evidence([0, 0], np.empty((2, 0), bool), count=2)
    kw["votes_by_view"] = [absent, empty, empty]
    kw["masks_by_view"] = [None, np.empty((0, 1, 3), bool), np.empty((0, 1, 3), bool)]
    packet = m.packet_for_pixels(**kw)
    assert not packet.target.any()
    assert packet.mask_signatures[0] is None and packet.mask_signatures[1].shape == (0, 1, 3)
    assert packet.sampled_mask_degree.tolist() == [[-1] * 4, [0] * 4, [0] * 4]


def test_packet_and_all_owned_arrays_remain_immutable_when_inputs_change():
    kw = packet_fixture()
    packet = m.packet_for_pixels(**kw)
    before = packet.target.tobytes(), packet.pixels.tobytes(), packet.mask_signatures
    kw["pixels"][:] = 2
    for mask in kw["masks_by_view"]:
        mask[:] = False
    assert before == (packet.target.tobytes(), packet.pixels.tobytes(), packet.mask_signatures)
    arrays = [
        packet.target,
        packet.pixels,
        packet.target_mass,
        packet.anchor_ids,
        packet.anchors[0].links[0],
        packet.assignments[0][0].assignment,
        packet.assignments[0][0].evidence.membership,
        packet.sampled_mask_membership[0],
    ]
    for a in arrays:
        with pytest.raises(ValueError):
            a.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.native = 999


def test_votes_copy_inputs_with_dtype_signatures_and_signed_zero_pixels_preserved():
    groups = np.array([0, 0], np.int32)
    report = np.ones(2, bool)
    member = np.ones((2, 1), bool)
    ev = m.transport_votes(
        groups,
        report,
        member,
        group_count=1,
        mask_available=True,
        native=101,
        release_horizon=108,
        query_sha256=SHA,
    )
    groups[:] = 4
    report[:] = False
    member[:] = False
    assert ev.query_group.dtype == np.int32 and ev.query_group.tolist() == [0, 0]
    assert ev.reported.all() and ev.membership.all()
    kw = packet_fixture()
    kw["pixels"][0, 0, 0] = -0.0
    packet = m.packet_for_pixels(**kw)
    assert packet.pixels_signature == m.signature(kw["pixels"])
    assert np.signbit(packet.pixels[0, 0, 0])
    opposite = kw["pixels"].copy()
    opposite[0, 0, 0] = 0.0
    assert m.signature(opposite).sha256 != packet.pixels_signature.sha256


@pytest.mark.parametrize(
    "bad",
    [
        "native",
        "duplicate_anchor",
        "future_anchor",
        "nonfinite_pixel",
        "fractional_pixel",
        "outside_pixel",
        "pixel_shape",
        "mask_count",
        "mask_dtype",
        "mask_shape",
        "available_none",
        "no_anchors",
    ],
)
def test_packet_rejects_time_coordinate_mask_or_anchor_drift(bad):
    kw = packet_fixture()
    if bad == "native":
        kw["votes_by_view"][1] = evidence([0, 0, 1, 1], [[1, 0]] * 4, native=102)
    elif bad == "duplicate_anchor":
        kw["anchors"] *= 2
    elif bad == "future_anchor":
        kw["anchors"] = [
            m.build_initial_anchor([refs()] * 3, [2] * 3, anchor_id=0, initial_native=102)
        ]
    elif bad == "nonfinite_pixel":
        kw["pixels"][0, 0, 0] = np.nan
    elif bad == "fractional_pixel":
        kw["pixels"][0, 0, 0] = 0.25
    elif bad == "outside_pixel":
        kw["pixels"][0, 0, 0] = 3
    elif bad == "pixel_shape":
        kw["pixels"] = np.zeros((3, 4, 3))
    elif bad == "mask_count":
        kw["masks_by_view"][0] = np.ones((1, 1, 3), bool)
    elif bad == "mask_dtype":
        kw["masks_by_view"][0] = kw["masks_by_view"][0].astype(np.uint8)
    elif bad == "mask_shape":
        kw["masks_by_view"][0] = np.ones((2, 3, 1), bool)
    elif bad == "available_none":
        kw["masks_by_view"][0] = None
    else:
        kw["anchors"] = []
    with pytest.raises(ValueError):
        m.packet_for_pixels(**kw)


def test_target_api_has_no_candidate_or_geometry_selection_inputs():
    for fn in (m.transport_votes, m.assign_transport, m.packet_for_pixels):
        names = set(inspect.signature(fn).parameters)
        assert not names.intersection(
            {"pose", "predicted", "candidate", "permutation", "coverage", "score"}
        )
