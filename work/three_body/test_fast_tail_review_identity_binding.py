"""Synthetic five-minute binding failures and whole-fish review-row preservation."""

import hashlib
import json
from pathlib import Path

import fast_tail_review_identity_binding as helper
import numpy as np
import pytest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected):
        assert digest(path) == expected
        self.paths[str(Path(path).resolve())] = expected


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return digest(path)


def record(path):
    return dict(path=str(path), sha256=digest(path), size_bytes=path.stat().st_size)


def arrays():
    n = helper.STOP - helper.START
    state = np.zeros(n, np.int64)
    state[120:220] = 3
    mapping = helper.PERMUTATIONS[state].copy()
    transition = np.zeros((n, 3), bool)
    transition[1:] = mapping[1:] != mapping[:-1]
    optimal = np.zeros((n, 6), bool)
    optimal[np.arange(n), state] = True
    marginal = np.full((n, 6), 2.0, np.float64)
    marginal[np.arange(n), state] = 0.0
    result = dict(
        frame_idx=np.arange(helper.START, helper.STOP),
        identity_to_raw_slot=mapping,
        permutation_state=state,
        state_alternative_gap=np.full(n, 2.0),
        identity_alternative_gap=np.full((n, 3), 2.0),
        state_alternative_status=np.full(n, "finite_gap", dtype="U32"),
        identity_alternative_status=np.full((n, 3), "finite_gap", dtype="U32"),
        optimal_state_mask=optimal,
        min_marginal_delta=marginal,
        id_slot_min_marginal_delta=np.zeros((n, 3, 3)),
        mapping_changed_from_initial_slots=mapping != [0, 1, 2],
        mapping_transition=transition,
        tied_identity_slots=np.zeros((n, 3), bool),
        terminal_age_frames=np.zeros((3, 2), np.int64),
        terminal_unsettled_gap=np.zeros((3, 2), bool),
    )
    for prefix in ("state", "identity"):
        result[prefix + "_alternative_gap"][0] = np.inf
        result[prefix + "_alternative_status"][0] = "fixed_initial_constraint"
    result["min_marginal_delta"][0, 1:] = np.inf
    result["id_slot_min_marginal_delta"][0] = np.where(np.eye(3, dtype=bool), 0, np.inf)
    return result


@pytest.mark.parametrize("prefix", ["state", "identity"])
def test_initial_constraint_status_and_gap_are_not_generic_alternatives(prefix):
    values = arrays()
    helper.validate_mapping(values, report(values))
    values[prefix + "_alternative_status"][0] = "no_feasible_alternative"
    with pytest.raises(ValueError, match="status disagrees"):
        helper.validate_mapping(values, report(values))
    values[prefix + "_alternative_status"][0] = "fixed_initial_constraint"
    values[prefix + "_alternative_gap"][0] = 2.0
    with pytest.raises(ValueError, match="fixed initial identity"):
        helper.validate_mapping(values, report(values))


def test_saved_mapping_contract_matches_real_tiny_frozen_solver(monkeypatch):
    # Exercise the actual producer's initial-status override, not a hand-coded
    # approximation of that schema. The native five-minute import gate is not run.
    monkeypatch.syspath_prepend(str(helper.WORK.parents[1] / "labs/20260909_three_body"))
    from identity_node_memory_dp import solve_identity_node_memory_dp

    n = 4
    monkeypatch.setattr(helper, "START", 0)
    monkeypatch.setattr(helper, "STOP", n)
    xyz = np.zeros((n, 3, 2, 3))
    xyz[..., 0] = np.arange(3)[None, :, None] * 10.0
    direct = np.ones((n, 3, 2), bool)
    direct[1:3, 2, 1] = False
    solved = solve_identity_node_memory_dp(np.arange(n), np.zeros((n, 6)), xyz, direct)
    mapped = {
        key: getattr(solved, key).copy() for key in helper.MAPPING_FIELDS if hasattr(solved, key)
    }
    mapped["identity_to_raw_slot"] = solved.permutation_by_id.copy()
    mapped["permutation_state"] = solved.path_state.copy()
    changed = solved.permutation_by_id != [0, 1, 2]
    transition = np.zeros((n, 3), bool)
    transition[1:] = solved.permutation_by_id[1:] != solved.permutation_by_id[:-1]
    mapped.update(
        mapping_changed_from_initial_slots=changed,
        mapping_transition=transition,
        tied_identity_slots=solved.identity_alternative_status == "tie_within_tolerance",
    )
    summary = report(mapped)
    summary.update(
        changed_mapping_native_frames=int(changed.any(1).sum()),
        mapping_transition_native_frames=int(transition.any(1).sum()),
        multiple_optimal_permutation_frames=int((solved.optimal_state_mask.sum(1) > 1).sum()),
        numerical_tie_tolerance=solved.numerical_tie_tolerance,
    )
    helper.validate_mapping(mapped, summary)


def report(values):
    return dict(
        native_frames=helper.STOP - helper.START,
        changed_mapping_native_frames=100,
        mapping_transition_native_frames=2,
        multiple_optimal_permutation_frames=0,
        numerical_tie_tolerance=1e-8,
        terminal_age_frames=values["terminal_age_frames"].tolist(),
        terminal_unsettled_gap=values["terminal_unsettled_gap"].tolist(),
        all_three_ID_truth=False,
        biological_swap_zero_verified=False,
        review_flags_cleared=False,
    )


@pytest.fixture
def bound(tmp_path, monkeypatch):
    identity, body = tmp_path / "identity", tmp_path / "comparison"
    identity.mkdir()
    body.mkdir()
    raw = tmp_path / "synthetic_raw_candidate.npz"
    raw.write_bytes(b"synthetic candidate binding; no medaka data")
    reference = dict(path=str(raw), sha256=digest(raw), arrays={"synthetic": "not actual raw21"})
    milestone = tmp_path / "milestone_summary.json"
    monkeypatch.setattr(helper, "MILESTONE_PATH", milestone)
    milestone_sha = dump(
        milestone,
        dict(
            status="ok",
            horizon_stop=helper.STOP,
            scientific_interval=[helper.START, helper.STOP],
            five_minute_candidate=True,
            candidate=record(raw),
        ),
    )
    body_plan = dict(
        schema=helper.COMPARISON_STAGE,
        horizon_stop=helper.STOP,
        candidate_references={"fast_tail": reference},
        milestone_sha256=milestone_sha,
    )
    body_summary = dict(
        **body_plan,
        status="ok",
        native_interval=[helper.START, helper.STOP],
        native_frames=helper.STOP - helper.START,
        candidate_all21_and_source_arrays_unchanged=True,
        review_modified=False,
        all_three_ID_truth=False,
        external_pose_resets=0,
        logical_initialization_count=1,
        same_parent_and_horizon=True,
    )
    dump(body / "plan.json", body_plan)
    dump(body / "summary.json", body_summary)
    body_seal = dict(
        stage=helper.COMPARISON_STAGE,
        status="ok",
        inputs={
            str(helper.COMPARISON_SOURCE): helper.COMPARISON_SOURCE_SHA,
            str(raw): reference["sha256"],
            str(milestone): milestone_sha,
        },
        outputs={name: record(body / name) for name in ("plan.json", "summary.json")},
    )
    body_sha = dump(body / "comparison.provenance.json", body_seal)
    values = arrays()
    identity_report = report(values)
    identity_plan = dict(
        schema=helper.IDENTITY_STAGE,
        native_interval=[helper.START, helper.STOP],
        intermediate_ID_reset=False,
        cap_increase=False,
        raw_candidate_reference=reference,
        comparison_sha256=body_sha,
        milestone_sha256=milestone_sha,
        one_solver_call=True,
        initial_P=[0, 1, 2],
        node_indices=[1, 0],
        raw_geometry_sources_support_review_modified=False,
        physical_ID_truth=False,
    )
    summary = dict(
        schema=helper.IDENTITY_STAGE,
        native_interval=[helper.START, helper.STOP],
        intermediate_ID_reset=False,
        cap_increase=False,
        raw_candidate_reference=reference,
        status="ok",
        mathematical_status="optimal",
        native_frames=helper.STOP - helper.START,
        solver_calls=1,
        node_indices=[1, 0],
        raw21_unchanged=True,
        all29_arrays_and19_scalars_saved=True,
        spool_all_files_SHA_layout_membership_verified=True,
        all_three_ID_truth=False,
        biological_swap_zero_verified=False,
        report=identity_report,
    )
    for name, value in (
        ("plan.json", identity_plan),
        ("summary.json", summary),
        ("identity_report.json", identity_report),
    ):
        dump(identity / name, value)
    np.savez_compressed(identity / "identity_mapping_and_ambiguity.npz", **values)
    identity_seal = dict(
        stage=helper.IDENTITY_STAGE,
        status="ok",
        inputs={
            str(helper.IDENTITY_SOURCE): helper.IDENTITY_SOURCE_SHA,
            str(helper.COMPARISON_SOURCE): helper.COMPARISON_SOURCE_SHA,
            str(body / "comparison.provenance.json"): body_sha,
            str(raw): reference["sha256"],
            str(milestone): milestone_sha,
        },
        outputs={
            name: record(identity / name)
            for name in (
                "plan.json",
                "summary.json",
                "identity_report.json",
                "identity_mapping_and_ambiguity.npz",
            )
        },
    )
    identity_sha = dump(identity / "identity.provenance.json", identity_seal)
    evidence = Evidence()
    evidence.add(raw, reference["sha256"])
    return dict(
        identity_root=identity,
        identity_sha=identity_sha,
        comparison_root=body,
        comparison_sha=body_sha,
        milestone_sha=milestone_sha,
        raw_candidate_reference=reference,
        anchor_frames=np.array([66420, 66530, 66640]),
        evidence=evidence,
    )


def reseal(bound, name, mutate):
    root = bound["identity_root"]
    path = root / name
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    dump(path, value)
    seal_path = root / "identity.provenance.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["outputs"][name] = record(path)
    bound["identity_sha"] = dump(seal_path, seal)


def test_success_is_bound_readonly_and_initial_slot_scoped(bound):
    value = helper.load_identity_anchors(**bound)
    assert value["available"] and value["reason"] is None
    np.testing.assert_array_equal(value["permutation_by_id"], helper.PERMUTATIONS[[0, 3, 0]])
    assert not value["permutation_by_id"].flags.writeable
    assert not value["binding"]["all_three_ID_truth"]
    assert not value["binding"]["review_flags_cleared"]
    assert not value["ambiguity"]["identity_alternative_gap"].flags.writeable
    assert not any("spool" in p for p in bound["evidence"].paths)


@pytest.mark.parametrize(
    "anchors,outside",
    [([65931, 66151, 66217], [65931, 66151, 66217]), ([66420, 66530, 116200], [116200])],
)
def test_outside_anchors_never_extrapolate_or_return_partial_mapping(bound, anchors, outside):
    bound["anchor_frames"] = np.array(anchors)
    value = helper.load_identity_anchors(**bound)
    assert value["available"] is False and value["permutation_by_id"] is None
    assert value["ambiguity"] is None and value["outside_frames"] == outside
    assert value["reason"] == "anchors_outside_identity_interval"


@pytest.mark.parametrize("key", ["identity_sha", "comparison_sha", "milestone_sha"])
def test_explicit_seal_and_milestone_digests_are_required(bound, key):
    bound[key] = "0" * 64
    with pytest.raises(ValueError):
        helper.load_identity_anchors(**bound)


def test_mapping_archive_byte_tamper_rejected(bound):
    path = bound["identity_root"] / "identity_mapping_and_ambiguity.npz"
    data = bytearray(path.read_bytes())
    data[30] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA differs"):
        helper.load_identity_anchors(**bound)


def test_resealed_mapping_still_requires_all_non_anchor_rows_to_be_valid(bound):
    path = bound["identity_root"] / "identity_mapping_and_ambiguity.npz"
    values = arrays()
    values["identity_to_raw_slot"][240] = [0, 0, 2]
    np.savez_compressed(path, **values)
    seal_path = bound["identity_root"] / "identity.provenance.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["outputs"][path.name] = record(path)
    bound["identity_sha"] = dump(seal_path, seal)
    with pytest.raises(ValueError, match="bijective"):
        helper.load_identity_anchors(**bound)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(status="failed"),
        lambda s: s["inputs"].update({str(helper.IDENTITY_SOURCE): "0" * 64}),
        lambda s: s["inputs"].pop(str(helper.COMPARISON_SOURCE)),
        lambda s: s["outputs"]["identity_mapping_and_ambiguity.npz"].update(size_bytes=1),
        lambda s: s["outputs"]["identity_mapping_and_ambiguity.npz"].update(path="wrong.npz"),
    ],
)
def test_identity_seal_status_source_output_edges_rejected(bound, mutation):
    path = bound["identity_root"] / "identity.provenance.json"
    seal = json.loads(path.read_text(encoding="utf-8"))
    mutation(seal)
    bound["identity_sha"] = dump(path, seal)
    with pytest.raises(ValueError):
        helper.load_identity_anchors(**bound)


@pytest.mark.parametrize(
    "name,mutation",
    [
        ("plan.json", lambda d: d.update(milestone_sha256="0" * 64)),
        ("plan.json", lambda d: d.update(comparison_sha256="0" * 64)),
        ("plan.json", lambda d: d.update(initial_P=[2, 1, 0])),
        ("plan.json", lambda d: d["raw_candidate_reference"].update(sha256="0" * 64)),
        ("summary.json", lambda d: d.update(raw21_unchanged=False)),
        ("summary.json", lambda d: d.update(all_three_ID_truth=True)),
        ("summary.json", lambda d: d.update(native_interval=[66400, 86320])),
        ("identity_report.json", lambda d: d.update(review_flags_cleared=True)),
    ],
)
def test_bound_plan_summary_and_report_must_describe_same_result(bound, name, mutation):
    reseal(bound, name, mutation)
    with pytest.raises(ValueError):
        helper.load_identity_anchors(**bound)


def test_unverified_candidate_reference_rejected(bound):
    bound["evidence"].paths.clear()
    with pytest.raises(ValueError, match="first bind"):
        helper.load_identity_anchors(**bound)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["frame_idx"].__setitem__(240, 66641),
        lambda a: a["frame_idx"].__setitem__(slice(10, 12), a["frame_idx"][[11, 10]]),
        lambda a: a["identity_to_raw_slot"].__setitem__(240, [0, 0, 2]),
        lambda a: a["identity_to_raw_slot"].__setitem__(0, [2, 1, 0]),
        lambda a: a["permutation_state"].__setitem__(240, 5),
        lambda a: a["mapping_transition"].__setitem__((240, 0), True),
        lambda a: a["identity_alternative_status"].__setitem__((240, 0), "tie_within_tolerance"),
        lambda a: a["identity_alternative_gap"].__setitem__((240, 0), np.nan),
        lambda a: a["optimal_state_mask"].__setitem__((240, 0), False),
    ],
)
def test_whole_interval_mapping_validation_includes_non_anchor_rows(mutation):
    values = arrays()
    saved_report = report(values)
    mutation(values)
    with pytest.raises(ValueError):
        helper.validate_mapping(values, saved_report)


def test_unsettled_terminal_memory_cannot_be_claimed_settled():
    values = arrays()
    values["terminal_age_frames"][2, 1] = 9
    saved_report = report(values)
    with pytest.raises(ValueError, match="missing-memory"):
        helper.validate_mapping(values, saved_report)


def rows():
    result = dict(
        frame_idx=np.array([66420, 66530, 66640]),
        pose_xyz=np.arange(81, dtype=np.float64).reshape(3, 3, 3, 3),
        sources=np.arange(81, dtype=np.int32).reshape(3, 3, 3, 3),
        observed=np.arange(27).reshape(3, 3, 3) % 3 == 0,
        identity_review=np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], bool),
    )
    result["pose_xyz"][0, 1, 2] = np.nan
    result["sources"][1, 2, 0, 2] = -1
    return result


def test_whole_fish_reordering_preserves_every_node_view_source_and_review():
    values = rows()
    before = {k: v.tobytes() for k, v in values.items()}
    permutation = helper.PERMUTATIONS[[3, 5, 1]].copy()
    result = helper.reorder_anchor_rows(values, permutation)
    assert set(result) == set(values)
    for key, value in values.items():
        assert value.tobytes() == before[key]
        assert result[key].dtype == value.dtype and not result[key].flags.writeable
        assert not np.shares_memory(result[key], value)
        if key == "frame_idx":
            assert result[key].tobytes() == value.tobytes()
        else:
            for t in range(3):
                for identity in range(3):
                    assert (
                        result[key][t, identity].tobytes()
                        == value[t, permutation[t, identity]].tobytes()
                    )
    assert result["identity_review"].sum() == values["identity_review"].sum()


@pytest.mark.parametrize("frames", [[66530, 66420, 66640], [66420, 66420, 66640], [66420, 66530]])
def test_anchor_row_order_and_membership_required(frames):
    values = rows()
    values["frame_idx"] = np.array(frames)
    with pytest.raises(ValueError):
        helper.reorder_anchor_rows(values, helper.PERMUTATIONS[[0, 1, 2]])


def test_extra_raw_fields_or_node_specific_permutation_rejected():
    values = rows()
    values["support"] = np.zeros(3)
    with pytest.raises(ValueError, match="five anchor"):
        helper.reorder_anchor_rows(values, helper.PERMUTATIONS[[0, 1, 2]])
    with pytest.raises(ValueError, match="whole-fish"):
        helper.reorder_anchor_rows(rows(), np.zeros((3, 3, 3), np.int64))
