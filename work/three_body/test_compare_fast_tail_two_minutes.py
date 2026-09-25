"""New comparison boundaries only: synthetic candidates and tiny metadata."""

from __future__ import annotations

import copy

import compare_fast_tail_two_minutes as caller
import fast_tail_two_minute_metrics as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def prohibit_gpu_raw(monkeypatch):
    assert not caller.producer.torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("new output/raw/GPU not permitted")

    monkeypatch.setattr(caller.producer.torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(caller.producer.original.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(caller.producer.original.cv2, "imread", forbidden)
    yield
    assert not caller.producer.torch.cuda.is_initialized()


def candidate(first=100, stop=108):
    t = stop - first
    seed = np.zeros((3, 3, 3), np.float64)
    seed[:, :, 0] = np.array([0.0, 2.0, 5.0])
    seed[:, :, 1] = np.arange(3)[:, None] * 20
    pose = np.broadcast_to(seed, (t, 3, 3, 3)).copy()
    data = dict(
        frame_idx=np.arange(first, stop, dtype=np.int64),
        pose_xyz=pose,
        observed=np.ones((t, 3, 3), bool),
        sources=np.broadcast_to(
            np.arange(3, dtype=np.int32)[None, :, None, None], (t, 3, 3, 3)
        ).copy(),
        identity_review=np.zeros((t, 3), bool),
        mask_supported=np.ones((t, 3), bool),
        body_supported=np.ones((t, 3), bool),
        missing=np.zeros((t, 3), np.int64),
        costs=np.zeros(t),
        path_costs=np.zeros(t),
        loss_parts=np.zeros((t, 5)),
        score_margin=np.zeros(t),
        proposal_kind=np.array(["prediction"] * t),
        hypotheses=np.zeros(t, np.int64),
        proposal_flow_nodes=np.zeros((t, 3, 3), bool),
        proposal_flow_parent=np.full(t, -1, np.int64),
        node_missing=np.zeros((t, 3, 3), np.int64),
        direct_pectoral_age=np.zeros((t, 3), np.int64),
    )
    refresh_derived(data)
    return data, seed


def refresh_derived(data):
    data["xyz"] = np.where(data["observed"][:, :, 1, None], data["pose_xyz"][:, :, 1], np.nan)
    data["accepted_xyz"] = np.where(data["identity_review"][:, :, None], np.nan, data["xyz"])
    data["body_xyz"] = np.where(
        data["body_supported"][:, :, None], data["pose_xyz"][:, :, 1], np.nan
    )


def test_all21_extent_and_derived_review_mask():
    data, _ = candidate()
    data["identity_review"][2, 1] = True
    refresh_derived(data)
    m.validate_candidate(data, 100, 108)
    data["accepted_xyz"][2, 1] = data["xyz"][2, 1]
    with pytest.raises(ValueError, match="derived support/review"):
        m.validate_candidate(data, 100, 108)


@pytest.mark.parametrize("change", ["field", "native", "pose32", "source_axis", "bool"])
def test_candidate_rejects_schema_or_native_difference(change):
    data, _ = candidate()
    if change == "field":
        data.pop("costs")
    elif change == "native":
        data["frame_idx"][3] += 1
    elif change == "pose32":
        data["pose_xyz"] = data["pose_xyz"].astype(np.float32)
    elif change == "source_axis":
        data["sources"] = data["sources"][:, :, :, :2]
    else:
        data["observed"] = data["observed"].astype(np.int64)
    with pytest.raises(ValueError):
        m.validate_candidate(data, 100, 108)


def test_direct_and_accepted_differ_without_changing_observed_or_review():
    data, seed = candidate()
    data["identity_review"][:, 1] = True
    data["observed"][2:4, 0, 2] = False
    refresh_derived(data)
    before = caller.signature(data)
    summary, _, arrays = m.interval_metrics(data, seed, 100, 108)
    assert summary["native_frames"] == 8
    assert summary["direct_by_node"] == [24, 24, 22]
    assert summary["accepted_by_node"] == [16, 16, 14]
    assert summary["all_three_direct_chest_frames"] == 8
    assert summary["all_three_accepted_chest_frames"] == 0
    assert summary["review_by_slot"] == [0, 8, 0]
    assert arrays["direct_segment"].shape == (8, 3, 2)
    assert caller.signature(data) == before


def test_missing_run_clips_halves_but_preserves_reacquisition_context():
    data, seed = candidate()
    data["observed"][2:6, 0, 2] = False
    data["observed"][6:, 2, 0] = False
    data["pose_xyz"][6:, 0, 2, 0] += 20
    refresh_derived(data)
    full, gaps, _ = m.interval_metrics(data, seed, 100, 108)
    left, _, _ = m.interval_metrics(data, seed, 100, 104)
    right, _, _ = m.interval_metrics(data, seed, 104, 108)

    def row(x, b, n):
        return next(r for r in x["node_metrics"] if r["slot"] == b and r["node"] == n)

    assert row(full, 0, "tail")["longest_missing_frames_in_interval"] == 4
    assert row(left, 0, "tail")["longest_missing_frames_in_interval"] == 2
    assert row(right, 0, "tail")["longest_missing_frames_in_interval"] == 2
    assert row(full, 0, "tail")["maximum_directly_supported_adjacent_step_mm"] == 0
    assert row(full, 0, "tail")["maximum_latent_adjacent_step_mm"] == 20
    assert any(r["terminal_open"] and r["slot"] == 2 for r in gaps["missing_runs"])
    reacquired = next(r for r in gaps["reacquisitions"] if r["slot"] == 0)
    assert reacquired["frame_idx"] == 106 and reacquired["missing_frames"] == 4
    assert reacquired["biological_identity_confirmed"] is False


def test_shape_zero_length_unavailable_and_direct_denominator():
    data, seed = candidate()
    data["pose_xyz"][0, 0, 1] = data["pose_xyz"][0, 0, 0]
    data["observed"][1, 1, 0] = False
    refresh_derived(data)
    summary, _, shape = m.interval_metrics(data, seed, 100, 108)
    assert np.isnan(shape["fold_angle_deg"][0, 0])
    assert shape["direct_fold"][:, 1].sum() == 7
    assert summary["body_geometry"][0]["all_estimated_fold_angle_deg"]["finite_count"] == 7


def test_all_source_transition_classes_and_gain_loss_preserve_axes():
    a, _ = candidate()
    b = copy.deepcopy(a)
    a["sources"][0, 0, 0, 0] = b["sources"][0, 0, 0, 0] = -1
    a["sources"][1, 0, 2, 1] = -1  # added: body0, side2, chest
    b["sources"][2, 1, 0, 2] = -1  # removed: body1, top, tail
    b["sources"][3, 2, 1, 0] = 8  # different row: body2, side1, mouth
    a["observed"][4, 0, 2] = False
    b["observed"][5, 1, 0] = False
    b["identity_review"][4, 0] = True
    before_a, before_b = caller.signature(a), caller.signature(b)
    summary, rows, arrays = m.differences(a, b, 100, 108)
    assert arrays["source_transition_class"][1, 0, 2, 1] == 2
    assert arrays["source_transition_class"][2, 1, 0, 2] == 3
    assert arrays["source_transition_class"][3, 2, 1, 0] == 4
    assert summary["source_transition_totals"] == dict(
        both_absent=1, same_present_row=212, added=1, removed=1, different_present_row=1
    )
    assert summary["gained_by_slot_node"][0][2] == 1
    assert summary["lost_by_slot_node"][1][0] == 1
    assert summary["accepted_gain_by_slot_node"][0][2] == 0
    assert [r["frame_idx"] for r in rows] == [104, 105]
    assert rows[0]["tail_review"] is True
    assert caller.signature(a) == before_a and caller.signature(b) == before_b


def test_horizon_changes_detect_bytes_and_ignore_unicode_capacity_only():
    old, _ = candidate(100, 104)
    final, _ = candidate(100, 108)
    final["proposal_kind"] = final["proposal_kind"].astype("U40")
    final["pose_xyz"][2, 1, 0, 2] = -0.0
    final["observed"][1, 1, 1] = False
    result, arrays = m.horizon_changes(old, final, 100, 104)
    assert result["fields"]["proposal_kind"]["changed_rows"] == 0
    assert result["fields"]["pose_xyz"]["changed_rows"] == 1
    assert result["fields"]["pose_xyz"]["first_changed_frame"] == 102
    assert arrays["observed_row_changed"].tolist() == [False, True, False, False]
    assert result["snapshot_horizon"] == 104 and result["final_horizon"] == 108


def completed_summary():
    return dict(
        schema=caller.producer.STAGE,
        status="ok",
        two_minutes_completed=True,
        native_interval=[66400, 86320],
        processed_full_history_interval=[49800, 86320],
        native_frames=19920,
        native_BGR_view_frames=59760,
        checkpoints=12,
        first_minute_exact_checkpoints=6,
        first_minute_all6_history19_exact_frames=9960,
        first_minute_same76360_candidate21_exact=True,
        final_candidate21_exact_backtrace=True,
        final_horizon_stop=86320,
        first_minute_snapshot_horizon=76360,
        logical_initialization_count=1,
        checkpoint_restore_count=4,
        processed_frames=36520,
        previous_frame=86319,
        restores_during_interval=0,
        new_begin_calls=0,
        external_pose_resets=0,
        finish_calls=1,
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("two_minutes_completed", False),
        ("native_frames", 9960),
        ("checkpoint_restore_count", 5),
        ("first_minute_exact_checkpoints", 5),
        ("finish_calls", 12),
    ],
)
def test_incomplete_or_wrong_lifecycle_rejected_before_array_read(key, value):
    summary = completed_summary()
    caller.completion(summary)
    summary[key] = value
    with pytest.raises(ValueError, match="completed fixed"):
        caller.completion(summary)


def test_metadata_gate_failure_never_opens_candidate(monkeypatch):
    monkeypatch.setattr(
        caller.producer,
        "read_seal",
        lambda *args: dict(
            seal=dict(params=dict(all_source_input_snapshot_output_SHA_unchanged=True))
        ),
    )
    monkeypatch.setattr(caller.producer, "consumed", lambda *args: "summary-only")
    monkeypatch.setattr(
        caller.checks, "read", lambda *args: {**completed_summary(), "status": "failed"}
    )
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("candidate opened before completion gate")
    )
    with pytest.raises(ValueError, match="completed fixed"):
        caller.tracking_metadata(caller.WORK / "never-open", "0" * 64, None)


def test_input_edge_requires_same_canonical_file_and_digest(tmp_path):
    p = tmp_path / "one.json"
    seal = dict(inputs={str(p): dict(path=str(p), sha256="a" * 64)})
    caller.input_edge(seal, p, "a" * 64)
    with pytest.raises(ValueError, match="edge"):
        caller.input_edge(seal, p, "b" * 64)


def test_all21_proof_cannot_drop_one_field_or_false_flag():
    proof = dict(exact=True, all21_bytes_required=True, fields={str(i): True for i in range(21)})
    caller.proof21(proof)
    proof["fields"]["0"] = False
    with pytest.raises(ValueError, match="all21"):
        caller.proof21(proof)
