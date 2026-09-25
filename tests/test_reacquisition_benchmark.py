"""Checks for honest missing episodes and the fixed no-reset evaluation scope."""

import importlib.util
import json
from pathlib import Path
from threading import Event, current_thread

import numpy as np
import pytest

from medaka3d.eval.reacquisition import (
    assess_availability,
    availability_episodes,
    nominal_tank_summary,
)


def load_benchmark():
    path = (Path(__file__).resolve().parents[1] / "labs/20260908_generative_tracking"
            / "benchmark_reacquisition.py")
    spec = importlib.util.spec_from_file_location("recovery_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_episodes_separate_leading_internal_and_trailing():
    xyz = np.ones((9, 3, 3))
    xyz[[0, 1, 4, 7, 8], 0] = np.nan
    summary = availability_episodes(xyz, np.arange(100, 109))
    slot = summary["slots"][0]
    assert [(r["start"], r["stop"]) for r in slot["missing_episodes"]] == [
        (100, 102), (104, 105), (107, 109)]
    assert [r["internally_bounded"] for r in slot["missing_episodes"]] == [False, True, False]
    assert slot["maximum_missing_frames"] == 2
    assert slot["missing_at_end_frames"] == 2
    assert slot["available_frames"] == 4


def test_missing_video_frames_do_not_create_a_recovered_episode():
    xyz = np.ones((4, 3, 3))
    xyz[[1, 2], 0] = np.nan
    summary = availability_episodes(xyz, np.array([10, 11, 20, 21]))
    gaps = summary["slots"][0]["missing_episodes"]
    assert len(gaps) == 2
    assert not any(gap["internally_bounded"] for gap in gaps)
    assert summary["slots"][0]["maximum_missing_frames"] == 1


def test_high_overall_coverage_cannot_hide_a_lost_individual():
    xyz = np.ones((100, 3, 3))
    xyz[79:, 2] = np.nan
    assessed = assess_availability(xyz, np.arange(100))
    assert assessed["overall_accepted_fraction"] == .93
    assert assessed["per_slot_accepted_fraction"] == [1, 1, .79]
    assert not assessed["availability_gate_passed"]


def test_all_coordinates_available_still_does_not_certify_identity():
    assessed = assess_availability(np.ones((10, 3, 3)), np.arange(10))
    assert assessed["availability_gate_passed"]
    assert assessed["all_three_identity_verdict"].startswith("unverified")


def test_empty_output_cannot_pass_and_unsorted_frames_are_rejected():
    assessed = assess_availability(np.empty((0, 3, 3)), np.array([], dtype=int))
    assert not assessed["availability_gate_passed"]
    assert assessed["overall_accepted_fraction"] is None
    with pytest.raises(ValueError, match="strictly increasing"):
        availability_episodes(np.ones((2, 3, 3)), np.array([2, 1]))


def test_nominal_floor_diagnostic_separates_observed_and_latent_outliers():
    pose = np.ones((2, 3, 3, 3))
    observed = np.ones((2, 3, 3), dtype=bool)
    pose[0, 1, 0, 2] = -4
    pose[1, 2, 1, 2] = -5
    observed[1, 2, 1] = False
    summary = nominal_tank_summary(pose, observed)
    assert summary["observed_nodes_outside_nominal"] == 1
    assert summary["latent_nodes_outside_nominal"] == 2
    assert summary["observed_outside_by_slot"] == [0, 1, 0]
    assert summary["minimum_observed_z_mm"] == -4
    assert summary["minimum_latent_z_mm"] == -5


def test_registered_validation_windows_are_disjoint_and_minute_is_exact():
    benchmark = load_benchmark()
    path = Path(benchmark.__file__).with_name("reacquisition_windows.json")
    registration, minute = benchmark.load_registered_window(path, "continuous_one_minute")
    assert (minute["start"], minute["stop"], minute["n_frames"]) == (76620, 86580, 9960)
    validation = [w for w in registration["windows"] if w["name"].startswith("validation_")]
    assert len(validation) == 4
    for window in validation:
        assert len(window["female_anchors"]) == 2
        assert window["seed_source"] and window["seed_pose_xyz"]
        for other in registration["windows"]:
            if other["name"] != window["name"]:
                assert window["stop"] <= other["start"] or other["stop"] <= window["start"]


def test_minute_cannot_silently_be_shortened(tmp_path):
    benchmark = load_benchmark()
    path = Path(benchmark.__file__).with_name("reacquisition_windows.json")
    payload = json.loads(path.read_text())
    minute = next(w for w in payload["windows"] if w["name"] == "continuous_one_minute")
    minute["stop"] -= 1
    minute["n_frames"] -= 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="9960"):
        benchmark.load_registered_window(changed, "continuous_one_minute")


def test_prefetch_preserves_order_and_closes_source_on_early_consumer_exit():
    load_benchmark()
    from ordered_prefetch import ordered_prefetch
    closed = Event()
    threads = []

    def source():
        try:
            for value in range(20):
                threads.append(current_thread())
                yield value
        finally:
            closed.set()

    with ordered_prefetch(source(), capacity=2) as prepared:
        assert [next(prepared) for _ in range(4)] == [0, 1, 2, 3]
    assert closed.is_set()
    assert len({thread.ident for thread in threads}) == 1
    assert all(not thread.is_alive() for thread in threads)


def test_prefetch_propagates_producer_failure_after_prior_items():
    load_benchmark()
    from ordered_prefetch import ordered_prefetch

    def source():
        yield 1
        raise ValueError("decoder failed")

    with ordered_prefetch(source()) as prepared:
        assert next(prepared) == 1
        with pytest.raises(ValueError, match="decoder failed"):
            next(prepared)
