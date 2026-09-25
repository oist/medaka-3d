"""Review workload is repeated uncertainty, not the number of sticky flag entries."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "labs/20260909_three_body/review_events.py"
SPEC = importlib.util.spec_from_file_location("three_body_review_events", PATH)
assert SPEC and SPEC.loader
events = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(events)


def result(count=30):
    pose = np.array([[[10, 20, 30], [15, 20, 30], [25, 20, 30]],
                     [[50, 20, 30], [55, 20, 30], [65, 20, 30]],
                     [[90, 20, 30], [95, 20, 30], [105, 20, 30]]], dtype=float)
    return {
        "frame_idx": np.arange(100, 100 + count),
        "pose_xyz": np.repeat(pose[None], count, axis=0),
        "observed": np.ones((count, 3, 3), bool),
        "body_supported": np.ones((count, 3), bool),
        "mask_supported": np.zeros((count, 3), bool),
        "identity_review": np.zeros((count, 3), bool),
        "missing": np.zeros((count, 3), int), "score_margin": np.full(count, np.inf),
    }


def lose(data, lo, hi, fish=0, counter_start=1):
    data["body_supported"][lo:hi, fish] = False
    data["observed"][lo:hi, fish, 1] = False
    data["missing"][lo:hi, fish] = np.arange(counter_start, counter_start + hi - lo)


def test_repeated_losses_after_sticky_review_produce_multiple_episodes():
    data = result()
    lose(data, 2, 6)
    lose(data, 12, 17)
    data["identity_review"][5:, 0] = True
    data["pose_xyz"][17:, 0, :, 0] += 20
    report = events.extract_review_events(data)
    assert report["counts"]["body_unsupported_episodes"] == 2
    assert report["counts"]["review_true_runs_per_fish_not_risk_event_count"] == [1, 0, 0]
    assert report["counts"]["body_episodes_with_review_already_true"] == 1
    assert report["counts"]["grouped_scenes"] == 1
    first, second = report["raw_events"]
    assert first["first_threshold_frame_in_available_input"] == 105
    assert second["last_body_support_frame"] == 111
    assert second["last_direct_pectoral_frame"] == 111
    assert second["first_return_frame"] == 117
    assert second["first_return_support"] == "direct_pectoral"
    assert second["saved_latent_pose_step_including_return"]["pectoral"] == {
        "maximum_mm": 20.0, "to_frame": 117, "finite_steps": 6,
    }
    assert report["scenes"][0]["raw_episode_count"] == 2
    assert report["counts"]["risk_union_frames"] == 9


def test_grouping_and_context_do_not_cross_missing_source_frames():
    data = result(20)
    data["frame_idx"][10:] += 10
    lose(data, 2, 6)
    lose(data, 12, 16, fish=1)
    report = events.extract_review_events(data)
    assert report["available_blocks"] == [[100, 110], [120, 130]]
    assert report["counts"]["grouped_scenes"] == 2
    assert [(s["context_start_frame"], s["context_stop_frame_exclusive"])
            for s in report["scenes"]] == [(100, 110), (120, 130)]
    assert report["counts"]["scene_context_frames"] == 20


def test_censored_missing_boundary_uses_saved_counter_without_inventing_support():
    data = result(3)
    lose(data, 0, 3, counter_start=5)
    data["identity_review"][:, 0] = True
    report = events.extract_review_events(data)
    episode = report["raw_events"][0]
    assert report["counts"]["body_unsupported_episodes"] == 1
    assert episode["available_frames"] == 3 and episode["maximum_saved_missing_counter"] == 7
    assert episode["left_censored"] and episode["right_censored"]
    assert episode["onset_frame"] is None and episode["actual_threshold_crossing_frame"] is None
    assert episode["last_body_support_frame"] is None and episode["first_return_frame"] is None
    assert episode["review_already_true_before_episode"] is None
    assert report["counts"]["scene_context_frames"] == 3


def test_finite_low_margin_is_unknown_pair_and_infinity_is_not_confidence():
    data = result(6)
    data["score_margin"] = np.array([np.inf, -np.inf, np.nan, 1.0, -0.5, 2.0])
    report = events.extract_review_events(data)
    episode = report["raw_events"][0]
    assert report["counts"]["candidate_permutation_risk_episodes"] == 1
    assert episode["start_frame"] == 103 and episode["stop_frame_exclusive"] == 105
    assert episode["affected_fish"] is None
    assert episode["minimum_finite_margin"] == -0.5
    assert report["scenes"][0]["contains_unknown_affected_fish"]
    assert report["scenes"][0]["known_affected_fish"] == []
    assert report["margin"] == {
        "finite_risk_frames": 2, "positive_infinity_no_compared_alternative_frames": 1,
        "nan_frames": 1, "negative_infinity_frames": 1,
    }
    json.dumps(report, allow_nan=False)


def test_mask_return_is_distinct_from_direct_observation_and_mask_alone_can_be_unsupported():
    data = result()
    lose(data, 2, 6)
    data["mask_supported"][2:10, 0] = True
    data["observed"][6:10, 0, 1] = False
    report = events.extract_review_events(data)
    episode = report["raw_events"][0]
    assert episode["first_return_frame"] == 106
    assert episode["first_return_support"] == "mask_bridge"
    assert episode["raw_mask_support_while_body_unsupported_frames"] == 4
    assert episode["direct_observed_node_frames"] == {"mouth": 4, "pectoral": 0, "tail": 4}
    assert report["support"]["mask_bridge_frames_per_fish"] == [4, 0, 0]
    direct = report["direct_pectoral_missing_episodes"][0]
    assert direct["frames"] == 8 and direct["mask_bridge_frames"] == 4


def test_subthreshold_loss_stays_visible_but_is_not_counted_as_new_risk():
    data = result()
    lose(data, 2, 5)
    report = events.extract_review_events(data)
    assert report["counts"]["raw_risk_episodes"] == 0
    assert report["counts"]["subthreshold_body_loss_episodes"] == 1
    assert report["counts"]["direct_pectoral_missing_episodes"] == 1
    assert report["margin"]["positive_infinity_no_compared_alternative_frames"] == 30
    assert report["counts"]["grouped_scenes"] == 0


def test_scene_gap_boundary_is_fixed_and_overlapping_risk_frames_are_not_double_counted():
    data = result(500)
    lose(data, 10, 14)
    lose(data, 180, 184, fish=1)  # 166 frames after previous half-open stop
    lose(data, 351, 355, fish=2)  # 167 frames after previous stop
    data["score_margin"][11:13] = 0.5
    report = events.extract_review_events(data)
    assert report["counts"]["raw_risk_episodes"] == 4
    assert report["counts"]["grouped_scenes"] == 2
    assert report["counts"]["risk_union_frames"] == 12
    assert report["counts"]["raw_episode_frame_sum_includes_overlap"] == 14
    assert report["scenes"][0]["known_affected_fish"] == [0, 1]
    assert report["scenes"][0]["contains_unknown_affected_fish"]
    assert report["scenes"][1]["context_start_frame"] == 368
    assert report["scenes"][1]["context_stop_frame_exclusive"] == 538


def test_no_cross_gap_jump_or_false_return_when_body_loss_reaches_gap():
    data = result(10)
    data["frame_idx"][5:] += 10
    lose(data, 1, 5)
    data["pose_xyz"][5:, 0] += 1000
    report = events.extract_review_events(data)
    episode = report["raw_events"][0]
    assert episode["first_return_frame"] is None and episode["right_censored"]
    assert episode["saved_latent_pose_step_including_return"]["pectoral"]["maximum_mm"] == 0


def test_counter_inconsistency_fails_loudly_and_input_is_not_mutated():
    data = result()
    lose(data, 2, 6)
    before = copy.deepcopy(data)
    events.extract_review_events(data)
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    data["missing"][3, 0] = 8
    with pytest.raises(ValueError, match="do not increment"):
        events.extract_review_events(data)


@pytest.mark.parametrize("threshold", [True, 0, -1, 4.0])
def test_invalid_missing_threshold_does_not_silently_change_inventory(threshold):
    with pytest.raises(ValueError, match="positive integer"):
        events.extract_review_events(result(), review_gap_frames=threshold)


def completed_run(tmp_path):
    run = tmp_path / "synthetic_run"
    run.mkdir()
    data = result()
    lose(data, 2, 6)
    lose(data, 12, 17)
    data["identity_review"][5:, 0] = True
    candidate_path = run / "candidate.npz"
    np.savez(candidate_path, **data)
    summary_path = run / "summary.json"
    summary_path.write_text(json.dumps({
        "technical_completion": True, "native_fps": 166, "temporal_subsampling": False,
        "tracker_config": {"review_gap_frames": 4},
    }), encoding="utf-8")
    (run / "experiment.provenance.json").write_text(json.dumps({
        "status": "ok", "outputs": {
            "candidate": {"sha256": events._sha256(candidate_path)},
            "summary": {"sha256": events._sha256(summary_path)},
        },
    }), encoding="utf-8")
    return run


def test_cli_preserves_source_and_writes_strict_json_with_no_overwrite(tmp_path):
    run = completed_run(tmp_path)
    candidate_before = (run / "candidate.npz").read_bytes()
    out = tmp_path / "inventory"
    comparison = events.main(["--run", str(run), "--output", str(out)])
    assert comparison[run.name]["raw_risk_episodes"] == 2
    report = json.loads((out / "synthetic_run.json").read_text(encoding="utf-8"))
    assert report["counts"]["body_episodes_with_review_already_true"] == 1
    assert report["source"]["human_truth_loaded"] is False
    assert (run / "candidate.npz").read_bytes() == candidate_before
    assert json.loads((out / "review_events.provenance.json").read_text())["status"] == "ok"
    with pytest.raises(FileExistsError):
        events.main(["--run", str(run), "--output", str(out)])


def test_cli_detects_changed_candidate_before_any_output_is_created(tmp_path):
    run = completed_run(tmp_path)
    with (run / "candidate.npz").open("ab") as stream:
        stream.write(b"changed candidate")
    out = tmp_path / "invalid_inventory"
    with pytest.raises(ValueError, match="candidate checksum mismatch"):
        events.main(["--run", str(run), "--output", str(out)])
    assert not out.exists()
