"""Operational refusal and successful/failed seals; no real artifacts or GPU."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import register_long_post_minute_execution as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def scientific():
    registered = dict(preflight_only=False, qualification={"fixed": "qualified"},
                      config={"steps": 14})
    summary = dict(
        schema="reflection_temporal_comparison_v1", frame_range=[79236, 79436],
        native_fps=166, temporal_subsampling=False, arms=["direct", "fresnel"],
        technical_completion=True, initialized_once_each=True, external_resets_each=0,
        inference_old_candidate_deserialized=False, gt_read=False,
        evaluation_after_both_saved=True, code_unchanged=True, inputs_unchanged=True,
        qualification=registered["qualification"], config=registered["config"],
        timings={arm: dict(initialization_count=1, reset_count=0, track_invocation_count=1,
                           native_frames_read=200) for arm in ["direct", "fresnel"]})
    return summary, registered


def test_negative_scientific_performance_does_not_block_reference(scientific):
    summary, registered = scientific
    for outcome in [False, True, "not independently scored"]:
        summary["comparison"] = {"all_gates_passed": outcome, "coverage_improved": False}
        module.validate_science(summary, registered)


@pytest.mark.parametrize("defect", ["prefix", "reset", "preflight", "qualification", "no_save"])
def test_incomplete_or_foreign_scientific_evidence_rejected(scientific, defect):
    summary, registered = copy.deepcopy(scientific)
    if defect == "prefix":
        summary["timings"]["fresnel"]["native_frames_read"] = 199
    elif defect == "reset":
        summary["timings"]["direct"]["initialization_count"] = 2
    elif defect == "preflight":
        registered["preflight_only"] = True
    elif defect == "qualification":
        summary["qualification"] = {"other": "unqualified"}
    else:
        summary["evaluation_after_both_saved"] = False
    with pytest.raises((ValueError, AssertionError)):
        module.validate_science(summary, registered)


@pytest.mark.parametrize("status,finished", [("running", "x"), ("failed", "x"), ("ok", "")])
def test_seal_must_have_successful_terminal_stage(tmp_path, status, finished):
    write(tmp_path / "seal.json", dict(status=status, stage="stage", finished_at=finished))
    with pytest.raises(ValueError, match="completed successful"):
        module.completion(tmp_path, "seal.json", "stage")


def test_incomplete_science_is_rejected_before_evidence_or_candidate_read(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SCIENCE", tmp_path)
    write(tmp_path / "experiment.provenance.json",
          dict(status="failed", stage="reflection_temporal_comparison", finished_at="done"))

    class NeverHash:
        def add(self, *_):
            pytest.fail("must refuse unfinished evidence before hashing any payload")

    with pytest.raises(ValueError, match="completed successful"):
        module.prepare(NeverHash(), 87103, 0)


@pytest.fixture
def next_plan():
    return dict(
        schema="three_body_long_next_phase_execution_plan_v1",
        full_target_interval=[49800, 348600], next_native_interval=[59760, 66400],
        output_interval=[49800, 66400], consumed_prefix_length=6,
        logical_initializations_expected=1, checkpoint_restores_expected=3,
        external_pose_resets_expected=0, full_selected_chunk_records=[{} for _ in range(10)],
        parent_phase="parent", expected_output="future_tracking",
        frozen_input_and_code_binding={"inputs": dict(
            detector_inputs={"source_cache_manifest": {"path": "cache/manifest.json"}},
            input_stage_root="stage", protocol_path="fixed_protocol.json",
            detector_binding={"runtime": {"executable": "fixed_detectron2_python"}},
            seed={"registration_path": "seed/registration.json", "input_root": "SL_inputs"})})


def test_schedule_reuses_frozen_programs_and_has_exactly_one_unbounded_detector(next_plan):
    before = copy.deepcopy(next_plan)
    result = module.schedule(next_plan, 87103, 0)
    assert result["detector"]["remaining_chunks"] == 170
    assert result["detector"]["concurrent_writer_count"] == 1
    assert result["detector"]["max_new_chunks"] is None
    assert "--max-new-chunks" not in result["detector"]["argv"]
    assert result["tracking"]["expected_restores"] == 3
    assert result["later_tracking"]["approximate_chunks_per_phase"] == 6
    assert not result["processes_started"] and not result["GPU_queried"]
    assert not result["scientific_performance_is_restart_condition"]
    assert not result["caller_execution_observation"]["independently_observed_by_this_script"]
    assert next_plan == before


@pytest.mark.parametrize("session,exit_code", [(False, 0), (0, 0), (1, 1), (1, False)])
def test_caller_exit_observation_is_explicit_and_successful(next_plan, session, exit_code):
    with pytest.raises(ValueError, match="caller must record"):
        module.schedule(next_plan, session, exit_code)


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected=None):
        value = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if expected is not None and value != expected:
            raise ValueError("SHA differs")
        self.paths[str(Path(path))] = value
        return value

    def final_guard(self):
        for path, digest in self.paths.copy().items():
            self.add(path, digest)


@pytest.mark.parametrize("change", ["output", "source", "snapshot"])
def test_main_seals_failure_for_mutation_at_write_boundary(tmp_path, monkeypatch, change):
    exercise_main(tmp_path, monkeypatch, change)


def test_main_seals_new_plan_without_launching_anything(tmp_path, monkeypatch):
    exercise_main(tmp_path, monkeypatch, None)


def exercise_main(tmp_path, monkeypatch, change):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "registrar.py"
    source.write_text("# sealed source")
    monkeypatch.setattr(module, "__file__", str(source))
    monkeypatch.setattr(module, "PINS", {})
    monkeypatch.setattr(module.next_phase, "local_source", lambda _: False)
    monkeypatch.setattr(module.next_phase.minute.prior.support_helpers, "Digests", Evidence)
    monkeypatch.setattr(module, "prepare", lambda *_: {"processes_started": False})
    original = module.COMMON.unchanged

    def final_guard(outputs):
        if change:
            selected = (source if change == "source" else
                        tmp_path / "out/code_snapshot/registrar.py" if change == "snapshot" else
                        tmp_path / "out/plan.json")
            # source/snapshot changes are tested during the preceding copy.
            if change == "output":
                selected.write_text("changed")
        original(outputs)

    monkeypatch.setattr(module.COMMON, "unchanged", final_guard)
    copyfile = module.shutil.copyfile

    def copy_then_mutate(src, dst):
        result = copyfile(src, dst)
        if change == "source":
            Path(src).write_text("changed")
        elif change == "snapshot":
            Path(dst).write_text("changed")
        return result

    monkeypatch.setattr(module.shutil, "copyfile", copy_then_mutate)
    args = ["--output", str(tmp_path / "out"),
            "--scientific-session", "87103", "--scientific-exit-code", "0"]
    if change:
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            module.main(args)
    else:
        module.main(args)
    record = json.loads((tmp_path / "out/registration.provenance.json").read_text())
    assert record["status"] == ("failed" if change else "ok")
    if not change:
        assert record["params"]["processes_started"] is False
        assert record["params"]["all_input_source_snapshot_output_sha_unchanged"] is True
        with pytest.raises(ValueError, match="new workspace"):
            module.main(args)
