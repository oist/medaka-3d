"""Reporting guards: no partial runs, no erased unknown outcomes or old reviews."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "work/three_body/compare_identity_memory_minute.py"
SPEC = importlib.util.spec_from_file_location("memory_comparison_join", SCRIPT)
join = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(join)


def geometry():
    pose = np.arange(108, dtype=np.float64).reshape(4, 3, 3, 3)
    observed = np.ones((4, 3, 3), bool)
    observed[1, 0] = False
    review = np.zeros((4, 3), bool)
    review[2:, 1] = True
    xyz = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    return dict(
        frame_idx=np.arange(10, 14, dtype=np.int64), pose_xyz=pose,
        observed=observed, identity_review=review,
        sources=np.full((4, 3, 3, 3), -1, np.int64), xyz=xyz,
        accepted_xyz=np.where(review[..., None], np.nan, xyz),
    )


def test_path_intervals_preserve_mapping_changes_and_identity_not_swap_event():
    frames = np.arange(50, 57)
    motion = np.tile([0, 1, 2], (7, 1))
    color = motion.copy()
    color[1:3] = [2, 1, 0]
    color[3:5] = [1, 0, 2]
    report, different = join.path_differences(frames, color, motion)
    assert report["different_frames"] == 4
    assert report["different_frames_by_identity"] == [4, 2, 2]
    assert [(x["start"], x["stop"]) for x in report["intervals"]] == [(51, 53), (53, 55)]
    assert report["intervals"][0]["changed_ids"] == [0, 2]
    assert report["intervals"][1]["changed_ids"] == [0, 1]
    assert [x["frame"] for x in report["transitions"]["color"]] == [51, 53, 55]
    assert report["biological_swap_event_count"] is None
    assert different.sum() == 8


def test_identical_nonidentity_paths_are_equal_but_transitions_remain():
    path = np.array([[0, 1, 2], [2, 1, 0], [2, 1, 0]])
    result, diff = join.path_differences(np.arange(3), path, path)
    assert result["different_frames"] == 0 and not diff.any()
    assert len(result["transitions"]["color"]) == 1


def test_invalid_permutation_is_rejected():
    with pytest.raises(ValueError, match="permutations"):
        join.path_differences(np.arange(2), np.array([[0, 0, 2]] * 2), np.array([[0, 1, 2]] * 2))


def test_relabel_moves_prior_flags_without_clearing():
    raw = geometry()
    mapping = np.tile([2, 0, 1], (4, 1))
    exported = join.build_identity_output(raw, mapping)
    report = join.check_export(raw, exported)
    assert report["raw_review_by_slot"] == [0, 2, 0]
    assert report["reordered_review_by_identity"] == [0, 0, 2]
    assert report["old_flags_cleared"] == 0
    assert np.isnan(exported["proposed_xyz"][1, 1]).all()


@pytest.mark.parametrize("field", [
    "prior_review_for_proposed_identity", "raw_identity_review", "proposed_observed",
    "proposed_sources", "conservative_accepted_xyz", "proposed_pose_xyz",
])
def test_export_tampering_is_rejected(field):
    raw = geometry()
    exported = join.build_identity_output(raw, np.tile([2, 0, 1], (4, 1)))
    value = exported[field]
    value.flat[0] = not value.flat[0] if value.dtype == bool else value.flat[0] + 1
    with pytest.raises(ValueError, match="byte-exact"):
        join.check_export(raw, exported)


def metric():
    count = dict(preserved=3, swap=0, unscored=10)
    female = dict(requested_intervals=13, interval_counts=count, available_anchors=4)
    motion = dict(total_animal_frames=29880, observed_animal_frames=6000,
                  observed_fraction=6000 / 29880, all_three_observed_frames=500)
    return dict(direct_motion=motion, accepted_motion=copy.deepcopy(motion),
                female_raw=female, female_accepted=copy.deepcopy(female), review_fish_frames=100)


def test_compact_report_keeps_unscored_and_declares_scope():
    result = join.compact_method(metric(), "no equivalent review flag")
    assert result["female_accepted"]["interval_counts"]["unscored"] == 10
    assert result["female_accepted"]["swap_events_inferred"] is False
    assert result["review_scope"] == "no equivalent review flag"
    assert result["accepted"]["fraction"] == 6000 / 29880


def test_missing_unknown_intervals_is_rejected():
    values = metric()
    values["female_raw"]["interval_counts"]["unscored"] = 0
    with pytest.raises(ValueError, match="omit requested"):
        join.compact_method(values, "scope")


def test_metric_bridge_rejects_changed_common_female_score():
    values = {key: {"value": 1} for key in join.METRIC_KEYS}
    old = dict(frames=[76620, 86580], fps=166, methods={"current_variant": values})
    new = dict(frames=[76620, 86580], native_fps=166, native_frames=9960,
               methods={"frozen_geometry_ids": copy.deepcopy(values)})
    assert join.check_metric_bridge(old, new)
    new["methods"]["frozen_geometry_ids"]["female_raw"]["value"] = 2
    with pytest.raises(ValueError, match="female_raw"):
        join.check_metric_bridge(old, new)


def write_seal(tmp_path, monkeypatch, *, status="ok"):
    monkeypatch.setattr(join, "REPO", tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    source = tmp_path / "source.py"
    source.write_text("unchanged", encoding="utf-8")
    result = root / "comparison.json"
    result.write_text("{}", encoding="utf-8")
    record = dict(
        status=status, stage="test", params=dict(output=str(root)),
        inputs={str(source): dict(path=str(source), sha256=join.base.sha256(source))},
        outputs={"comparison": dict(path=str(result), sha256=join.base.sha256(result))},
    )
    seal = root / "evaluation.provenance.json"
    seal.write_text(json.dumps(record), encoding="utf-8")
    return root, record, source, result


def test_success_seal_binds_inputs_and_outputs(tmp_path, monkeypatch):
    root, _record, source, output = write_seal(tmp_path, monkeypatch)
    hashes = {}
    join.verify_record(root, "evaluation.provenance.json", "test",
                       {"comparison": "comparison.json"}, hashes)
    assert set(hashes) == {str(root / "evaluation.provenance.json"), str(source), str(output)}


@pytest.mark.parametrize("status", ["running", "failed", None])
def test_partial_or_failed_seal_never_reads_result_values(tmp_path, monkeypatch, status):
    root, _record, _source, output = write_seal(tmp_path, monkeypatch, status=status)
    output.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        join.verify_record(root, "evaluation.provenance.json", "test",
                           {"comparison": "comparison.json"}, {})


@pytest.mark.parametrize("target", ["source", "output"])
def test_changed_input_or_output_rejected(tmp_path, monkeypatch, target):
    root, _record, source, output = write_seal(tmp_path, monkeypatch)
    (source if target == "source" else output).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        join.verify_record(root, "evaluation.provenance.json", "test",
                           {"comparison": "comparison.json"}, {})


def test_missing_required_output_role_rejected(tmp_path, monkeypatch):
    root, _record, _source, _output = write_seal(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="missing required"):
        join.verify_record(root, "evaluation.provenance.json", "test",
                           {"motion": "motion.npz"}, {})
