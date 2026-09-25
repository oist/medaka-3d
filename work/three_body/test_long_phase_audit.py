"""General phase endpoints and recurrence against the frozen real CPU loop."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "tests")]
import long_phase_audit as module  # noqa: E402

import test_three_body_continuous_tracker as fixture  # noqa: E402


@pytest.fixture(scope="module")
def actual_branching_history():
    rig = fixture.SmallRig(tuple(fixture.SmallCamera(axes, np.zeros(2))
                                 for axes in ((0, 1), (0, 2), (1, 2))))
    centers = np.array([[45., 48., 50.], [95., 100., 92.], [150., 145., 148.]])
    pose = centers[:, None]+np.array([[8., 6., 4.], [0., 0., 0.], [-14., -10., -6.]])[None]
    tracker = fixture.C.ContinuousPersistentNodeTracker(
        rig, pose, fixture.config(beam_size=6, max_hypotheses=18, opt_steps=1))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tracker.begin()
        for t in range(9):
            tracker.advance(fixture.frame(rig, pose, module.FIRST+t, empty=3 <= t < 7))
        a, m = tracker.export_state()
        before = dict(arrays=a, metadata=m)
        for t in range(9, 15):
            tracker.advance(fixture.frame(rig, pose, module.FIRST+t, empty=9 <= t < 13))
        a, m = tracker.export_state()
        after = dict(arrays=a, metadata=m, summary=dict(
            retained_history_rows=int(a["history_offsets"][-1]),
            active_beam_count=len(a["state_pose"])))
        return before, after
    finally:
        torch.set_num_threads(old_threads)


def test_real_frozen_cpu_history_suffix_exact_with_missing_review_and_multiple_parents(
        actual_branching_history):
    before, after = actual_branching_history
    originals = {key: value.tobytes() for key, value in after["arrays"].items()}
    result = module.suffix_history(before, after)
    assert result["active_beam_count"] == 6
    assert result["new_history_rows_verified"] == 36
    assert result["new_nonzero_parent_rows"] > 0
    assert after["arrays"]["history_identity_review"].any()
    assert after["arrays"]["history_missing"].max() >= 4
    assert all(value.tobytes() == originals[key] for key, value in after["arrays"].items())


@pytest.mark.parametrize("field", ["prefix", "parent", "age", "velocity", "cost", "review",
                                  "exclusivity", "frame", "final_beam"])
def test_corrupt_saved_history_is_rejected(actual_branching_history, field):
    before, after = copy.deepcopy(actual_branching_history)
    a = after["arrays"]
    first = int(a["history_offsets"][before["metadata"]["processed_frames"]])
    if field == "prefix":
        a["history_pose_xyz"][0, 0, 0, 0] += 1
    elif field == "parent":
        a["history_parent"][first] = 6
    elif field == "age":
        a["history_node_missing"][first, 0, 0] += 1
    elif field == "velocity":
        a["state_velocity"][0, 0, 0] += .1
    elif field == "cost":
        a["history_path_costs"][first] += .2
    elif field == "review":
        a["history_identity_review"][-6:] = False
    elif field == "exclusivity":
        a["history_sources"][-1, 1] = a["history_sources"][-1, 0]
    elif field == "frame":
        a["history_frame_idx"][first] += 1
    else:
        a["state_pose"] = a["state_pose"][:1]
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        module.suffix_history(before, after)


def valid_summary(stop):
    start = stop-2*module.CHUNK
    return dict(
        schema="long_continuation_phase_v1", technical_completion=True,
        full_sequence_complete=stop == module.FULL_STOP,
        prefix_backtrace_provisional=stop != module.FULL_STOP,
        registered_interval=[module.FIRST, module.FULL_STOP],
        processed_interval=[module.FIRST, stop], processed_total=stop-module.FIRST,
        phase_frames=[start, stop], phase_processed=stop-start,
        consumed_chunk_count=(stop-module.FIRST)//module.CHUNK,
        logical_initialization_count=1, checkpoint_restore_count=30,
        external_pose_reset_count=0, temporal_subsampling=False, human_annotation_input=False,
        post_initialization_baseline_coordinates_input=False)


@pytest.mark.parametrize("stop", [66400, module.FULL_STOP])
def test_final_vs_prefix_completion_flags_are_not_fixed_false(stop):
    summary = valid_summary(stop)
    kwargs = dict(stop=stop, phase_start=stop-2*module.CHUNK, restores=30,
                  chunks=(stop-module.FIRST)//module.CHUNK)
    module.validate_summary(summary, **kwargs)
    summary["full_sequence_complete"] = not summary["full_sequence_complete"]
    with pytest.raises((ValueError, AssertionError)):
        module.validate_summary(summary, **kwargs)


@pytest.mark.parametrize("value", [True, -1, 66401, 348601, 66400., 2**64])
def test_invalid_native_endpoints_rejected(value):
    with pytest.raises(ValueError, match="integer chunk endpoint"):
        module.native_endpoint(value)


@pytest.mark.parametrize("start,stop", [(66400, 76360), (345280, 348600)])
def test_prepare_inherits_parent_and_registers_exact_six_or_last_two(
        tmp_path, monkeypatch, start, stop):
    monkeypatch.setattr(module, "REPO", tmp_path)
    parent, output = tmp_path / "parent", tmp_path / "new_phase"
    binding = {"registered": "unchanged"}
    seal = {"params": {"through_frame": start, "bindings": binding}}
    monkeypatch.setattr(module, "sealed", lambda *_: (parent, seal))
    chunks = [dict(start=f, stop=f+module.CHUNK)
              for f in range(module.FIRST, stop, module.CHUNK)]
    inputs = SimpleNamespace(chunk_records=chunks)
    monkeypatch.setattr(module, "verified_inputs", lambda *_: (inputs, {}))
    previous = dict(arrays=dict(state_pose=np.zeros((6, 3, 3, 3)),
                                history_offsets=np.array([0, 357])),
                    metadata=dict(restore_count=8, context={"fixed": "context"}))
    monkeypatch.setattr(module, "load_phase", lambda *_: previous)
    evidence = SimpleNamespace(paths={str(parent / "continuation.provenance.json"): "pinned"})
    plan, context = module.prepare_phase(parent, stop, output, evidence)
    assert plan["expected_restores"] == 9
    assert plan["parent_history_rows"] == 357
    assert len(plan["next_selected_chunk_records"]) == (stop-start)//module.CHUNK
    assert plan["full_sequence_complete"] is (stop == module.FULL_STOP)
    assert context["parent"] is previous
    assert not output.exists()


@pytest.mark.parametrize("start,stop", [(66400, 69720), (66400, 78020),
                                       (348600, 348600), (64740, 74700)])
def test_arbitrary_partial_or_skipped_group_rejected(start, stop):
    with pytest.raises(ValueError):
        module.phase_range(start, stop)


@pytest.fixture
def bootstrap_plan():
    return dict(
        schema="three_body_long_next_phase_execution_plan_v1",
        full_target_interval=[module.FIRST, module.FULL_STOP],
        parent_processed_interval=[module.FIRST, 59760], next_native_interval=[59760, 66400],
        output_interval=[module.FIRST, 66400], logical_initializations_expected=1,
        checkpoint_restores_expected=3, external_pose_resets_expected=0,
        parent_active_beam_expected=6, parent_audit_seal_sha256=module.MINUTE_AUDIT_SHA,
        parent_phase="parent", expected_output="next", parent_history_rows_expected=59753,
        frozen_input_and_code_binding={"unchanged": True}, parent_checkpoint_context={"beam": 6},
        full_selected_chunk_records=[{}]*10, next_selected_chunk_records=[{}]*4)


def test_bootstrap_uses_existing_four_chunk_registration_without_future_inputs(bootstrap_plan):
    before = copy.deepcopy(bootstrap_plan)
    result = module.adapt_bootstrap_plan(bootstrap_plan, "sealed_parent")
    assert result["parent_history_rows"] == 59753
    assert result["expected_restores"] == 3
    assert result["next_native_interval"] == [59760, 66400]
    assert len(result["full_selected_chunk_records"]) == 10
    assert len(result["next_selected_chunk_records"]) == 4
    assert bootstrap_plan == before


@pytest.mark.parametrize("key,value", [("parent_audit_seal_sha256", "foreign"),
                                       ("checkpoint_restores_expected", 1),
                                       ("next_native_interval", [59760, 76360])])
def test_bootstrap_adapter_cannot_admit_another_audit_or_interval(bootstrap_plan, key, value):
    bootstrap_plan[key] = value
    with pytest.raises((ValueError, AssertionError)):
        module.adapt_bootstrap_plan(bootstrap_plan, "sealed_parent")


@pytest.mark.parametrize("status", ["running", "failed"])
def test_unfinished_parent_rejected_before_any_checkpoint_read(tmp_path, monkeypatch, status):
    monkeypatch.setattr(module, "REPO", tmp_path)
    (tmp_path / "continuation.provenance.json").write_text(json.dumps(dict(
        status=status, stage="three_body_long_continuation", params={})))
    class Evidence:
        def add(self, path):
            assert Path(path).name == "continuation.provenance.json"
    with pytest.raises(ValueError, match="completed successful"):
        module.sealed(tmp_path, Evidence())
