"""Ledger boundaries preserve uncertainty without repairing IDs or coordinates."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def load_ledger():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    old = {name: sys.modules.get(name) for name in ("review_events", "tracklet_ledger")}
    try:
        for name in old:
            spec = importlib.util.spec_from_file_location(
                f"_ledger_test_{name}", lab / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in old.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


module = load_ledger()


def case(count=40):
    return {
        "frame_idx": np.arange(100, 100 + count),
        "pose_xyz": np.ones((count, 3, 3, 3)),
        "observed": np.ones((count, 3, 3), bool),
        "body_supported": np.ones((count, 3), bool),
        "mask_supported": np.zeros((count, 3), bool),
        "identity_review": np.zeros((count, 3), bool),
        "missing": np.zeros((count, 3), int),
        "score_margin": np.full(count, np.inf),
    }


def lose(data, lo, hi, fish=0):
    data["observed"][lo:hi, fish] = False
    data["body_supported"][lo:hi, fish] = False
    data["missing"][lo:hi, fish] = np.arange(1, hi - lo + 1)
    if hi - lo >= 4:
        data["identity_review"][lo + 3 :, fish] = True


def ledger(data):
    return module.build_tracklet_ledger(
        data, tracker_config={"review_gap_frames": 4, "ambiguity_margin": 1.0}
    )


def ranges(result, slot):
    return [
        (t["start_frame"], t["stop_frame_exclusive"])
        for t in result["tracklets"]
        if t["slot"] == slot
    ]


def test_recurring_losses_after_sticky_review_are_separate_and_inputs_unchanged():
    data = case()
    lose(data, 5, 10)
    lose(data, 20, 25)
    before = copy.deepcopy(data)
    result = ledger(data)
    assert result["counts"]["interval_risks"] == 2
    assert ranges(result, 0) == [(100, 105), (110, 120), (125, 140)]
    links = [t for t in result["tracklets"] if t["slot"] == 0]
    assert [len(t["unresolved_risk_ids"]) for t in links] == [0, 1, 2]
    assert result["counts"]["excluded_risk_slot_frames_per_slot"] == [10, 0, 0]
    assert result["coordinates_emitted"] is False and result["identities_repaired"] is False
    np.testing.assert_array_equal(result["raw_identity_review"], data["identity_review"])
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])


def test_overlapping_body_and_unknown_pair_margin_use_union_not_double_counting():
    data = case()
    lose(data, 5, 12)
    data["score_margin"][10:15] = 0.5
    result = ledger(data)
    assert result["counts"]["interval_risks"] == 2
    assert result["counts"]["excluded_risk_slot_frames_per_slot"] == [10, 5, 5]
    assert ranges(result, 0) == [(100, 105), (115, 140)]
    assert ranges(result, 1) == [(100, 110), (115, 140)]
    event = next(e for e in result["risk_events"] if e["kind"] == "candidate_permutation_risk")
    assert event["original_affected_slots"] is None and event["affected_slots"] == [0, 1, 2]
    # Raw false on another fish cannot turn a post-risk tracklet into verified identity.
    later = [t for t in result["tracklets"] if t["slot"] == 1][-1]
    assert later["raw_review_true_frames"] == 0 and later["unresolved_risk_ids"]


def test_unknown_review_onset_becomes_boundary_without_inventing_one_frame_extent():
    data = case()
    data["identity_review"][10:, 2] = True
    result = ledger(data)
    assert result["counts"]["excluded_risk_slot_frames_per_slot"] == [0, 0, 0]
    assert ranges(result, 2) == [(100, 110), (110, 140)]
    event = result["risk_events"][0]
    assert event["kind"] == "review_onset_without_recorded_risk"
    assert event["boundary_frame"] == 110
    assert event["stop_frame_exclusive"] is None and event["cause_onset_frame"] is None
    assert [t for t in result["tracklets"] if t["slot"] == 2][-1]["unresolved_risk_ids"]


def test_initial_review_and_later_raw_flag_clearing_do_not_restore_identity():
    data = case()
    data["identity_review"][:10, 1] = True
    result = ledger(data)
    assert ranges(result, 1) == [(100, 140)]
    row = next(t for t in result["tracklets"] if t["slot"] == 1)
    assert row["unresolved_risk_ids"] and row["raw_review_true_frames"] == 10
    assert result["risk_events"][0]["kind"] == "initial_review_history_unknown"
    assert result["raw_review_clearing_observations"][0]["frame"] == 110


def test_frame_gap_splits_all_slots_and_right_side_has_unknown_identity_link():
    data = case()
    data["frame_idx"][20:] += 100
    result = ledger(data)
    for fish in range(3):
        assert ranges(result, fish) == [(100, 120), (220, 240)]
        row = [t for t in result["tracklets"] if t["slot"] == fish][-1]
        assert row["unresolved_risk_ids"] and row["raw_review_true_frames"] == 0
    assert result["risk_events"][0]["unavailable_source_interval"] == [120, 220]


def test_mask_only_support_and_short_position_gaps_do_not_create_direct_nodes():
    data = case()
    data["observed"][5:10, 0, :2] = False
    data["mask_supported"][5:10, 0] = True
    lose(data, 20, 22)
    data["pose_xyz"][20:22, 0] = np.nan
    result = ledger(data)
    assert result["counts"]["interval_risks"] == 0
    assert ranges(result, 0) == [(100, 140)]
    row = next(t for t in result["tracklets"] if t["slot"] == 0)
    assert row["mask_only_body_frames"] == 5
    assert row["missing_direct_pectoral_frames"] == 7
    assert row["direct_node_frames"]["pectoral"] == 33
    assert result["coordinates_emitted"] is False


def test_censored_loss_preserves_unknown_onset_and_absent_return():
    data = case()
    lose(data, 0, 40)
    data["missing"][:, 0] += 20
    data["identity_review"][:, 0] = True
    result = ledger(data)
    event = next(e for e in result["risk_events"] if e["kind"] == "body_unsupported")
    assert event["left_censored"] and event["right_censored"]
    assert event["known_risk_start_frame"] is None
    assert event["known_risk_stop_frame_exclusive"] is None
    assert event["source_event"]["first_return_frame"] is None
    assert ranges(result, 0) == []


@pytest.mark.parametrize(
    "config",
    [{}, {"review_gap_frames": 3}, {"review_gap_frames": 4.0}, {"review_gap_frames": True}, None],
)
def test_fixed_configuration_binding_is_required(config):
    with pytest.raises(ValueError):
        module.build_tracklet_ledger(case(), tracker_config=config)


@pytest.mark.parametrize("margin", [None, True, "1.0", float("nan"), float("inf"), 0.5, 2.0])
def test_fixed_margin_configuration_is_bound_to_frozen_extractor(margin):
    with pytest.raises(ValueError, match=r"ambiguity_margin=1\.0"):
        module.build_tracklet_ledger(
            case(), tracker_config={"review_gap_frames": 4, "ambiguity_margin": margin}
        )


def test_cli_does_not_seal_if_input_changes_during_snapshot_write(tmp_path, monkeypatch):
    root = tmp_path / "completed"
    root.mkdir()
    candidate = root / "candidate.npz"
    np.savez(candidate, **case())
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "technical_completion": True,
                "native_fps": 166,
                "temporal_subsampling": False,
                "returned_frames": 40,
                "frames": [100, 140],
                "tracker_config": {"review_gap_frames": 4, "ambiguity_margin": 1.0},
            }
        )
    )
    (root / "experiment.provenance.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "outputs": {
                    "candidate": {"sha256": module.sha(candidate)},
                    "summary": {"sha256": module.sha(summary)},
                },
            }
        )
    )
    original = module.shutil.copyfile

    def copy_then_change_input(*args, **kwargs):
        value = original(*args, **kwargs)
        summary.write_text(summary.read_text() + " ")
        return value

    monkeypatch.setattr(module.shutil, "copyfile", copy_then_change_input)
    output = tmp_path / "must_not_seal"
    with pytest.raises(ValueError, match="changed while saving"):
        module.main(["--run", str(root), "--output", str(output)])
    saved = json.loads((output / "ledger.provenance.json").read_text())
    assert saved["status"] != "ok"


def test_unrelated_review_onset_is_not_explained_by_another_fish_loss():
    data = case()
    lose(data, 5, 12, fish=0)
    data["identity_review"][8:, 1] = True
    result = ledger(data)
    unknown = [e for e in result["risk_events"] if e["representation"] == "unknown_extent_boundary"]
    assert len(unknown) == 1 and unknown[0]["affected_slots"] == [1]
    assert unknown[0]["boundary_frame"] == 108


def test_no_risks_means_three_automatic_tracklets_but_no_verified_identity():
    result = ledger(case())
    assert result["counts"]["tracklets_per_slot"] == [1, 1, 1]
    assert result["risk_events"] == []
    assert all(not t["global_identity_link_verified"] for t in result["tracklets"])
