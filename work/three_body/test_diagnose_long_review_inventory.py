"""Long result adapter refuses mismatched lifecycle and retains censored risks."""

import json

import diagnose_long_review_inventory as mod
import numpy as np
import pytest


def fixture(stop=51460):
    count = stop-mod.FIRST
    config = {"review_gap_frames": 4, "ambiguity_margin": 1.0}
    seal = {"stage": "three_body_long_continuation", "status": "ok", "params": {
        "all_input_source_snapshot_output_sha_unchanged": True, "through_frame": stop}}
    summary = {
        "schema": "long_continuation_phase_v1", "technical_completion": True,
        "full_sequence_complete": stop == mod.STOP,
        "prefix_backtrace_provisional": stop != mod.STOP,
        "registered_interval": [mod.FIRST, mod.STOP],
        "processed_interval": [mod.FIRST, stop], "processed_total": count,
        "consumed_chunk_count": count//1660, "logical_initialization_count": 1,
        "external_pose_reset_count": 0, "checkpoint_restore_count": 0,
        "temporal_subsampling": False, "human_annotation_input": False,
        "post_initialization_baseline_coordinates_input": False, "active_beam_count": 6,
    }
    metadata = {"schema": "persistent_tracker_full_state_v1",
                "context": {"tracker_config": config}, "initialize_count": 1,
                "processed_frames": count, "previous_frame": stop-1, "restore_count": 0}
    consumed = [{"index": i, "start": mod.FIRST+i*1660, "stop": mod.FIRST+(i+1)*1660}
                for i in range(count//1660)]
    protocol = {"source_frame_range": [mod.FIRST, mod.STOP], "fps": 166,
                "tracker_config": config}
    frames = np.arange(mod.FIRST, stop, dtype=np.int64)
    return [seal, summary, metadata, consumed, protocol, frames]


@pytest.mark.parametrize("stop", [51460, 59760, 66400, 348600])
def test_native_prefix_and_final_flags(stop):
    args = fixture(stop)
    assert mod.validate_metadata(*args) == args[4]["tracker_config"]


@pytest.mark.parametrize("field,value", [
    ("full_sequence_complete", True), ("prefix_backtrace_provisional", False),
    ("logical_initialization_count", 2), ("external_pose_reset_count", 1),
    ("temporal_subsampling", True), ("active_beam_count", 1),
    ("technical_completion", 1), ("checkpoint_restore_count", 4),
    ("checkpoint_restore_count", False),
])
def test_rejects_false_completion_or_lifecycle(field, value):
    args = fixture()
    args[1][field] = value
    with pytest.raises(ValueError):
        mod.validate_metadata(*args)


@pytest.mark.parametrize("kind", ["missing_frame", "chunk_order", "config", "failed", "schema"])
def test_rejects_native_input_or_policy_mismatch(kind):
    args = fixture(53120)
    if kind == "missing_frame":
        args[5][10] += 1
    elif kind == "chunk_order":
        args[3].reverse()
    elif kind == "config":
        args[2]["context"]["tracker_config"] = {"review_gap_frames": 3, "ambiguity_margin": 1.0}
    elif kind == "schema":
        args[2]["schema"] = "unsupported"
    else:
        args[0]["status"] = "failed"
    with pytest.raises(ValueError):
        mod.validate_metadata(*args)


@pytest.mark.parametrize("loss", [False, True])
def test_risks_and_missing_probes_remain_without_identity_claim(loss):
    count = 10
    observed = np.ones((count, 3, 3), dtype=bool)
    body = np.ones((count, 3), dtype=bool)
    missing = np.zeros((count, 3), dtype=np.int64)
    review = np.zeros((count, 3), dtype=bool)
    if loss:
        observed[-4:, 0] = False
        body[-4:, 0] = False
        missing[-4:, 0] = np.arange(1, 5)
        review[-1, 0] = True
    data = {"frame_idx": np.arange(49800, 49800+count),
            "pose_xyz": np.zeros((count, 3, 3, 3)), "observed": observed,
            "body_supported": body, "mask_supported": np.zeros((count, 3), dtype=bool),
            "missing": missing, "identity_review": review,
            "score_margin": np.full(count, np.inf)}
    direct, accepted = observed[:, :, 1], observed[:, :, 1] & ~review
    producer = {"processed_interval": [49800, 49800+count], "full_sequence_complete": False,
                "prefix_backtrace_provisional": True, "direct_pectoral_count": int(direct.sum()),
                "accepted_pectoral_count": int(accepted.sum()),
                "identity_review_count": int(review.sum()),
                "per_slot_direct_pectoral": direct.sum(0).tolist(),
                "per_slot_accepted_pectoral": accepted.sum(0).tolist()}
    report = mod.make_reports(data, producer, {"review_gap_frames": 4, "ambiguity_margin": 1.0})
    summary = report["summary"]
    json.dumps(report, allow_nan=False)
    assert summary["margin"]["positive_infinity_no_compared_alternative_frames"] == count
    assert summary["identities_repaired"] is False
    assert summary["all_three_identity_accuracy_measured"] is False
    assert summary["risk_counts"]["grouped_scenes"] == int(loss)
    assert summary["incomplete_scene_count"] == int(loss)
    assert summary["missing_probes"] == 2*int(loss)
    assert summary["ledger_counts"]["raw_review_true_frames_per_slot"] == [int(loss), 0, 0]
