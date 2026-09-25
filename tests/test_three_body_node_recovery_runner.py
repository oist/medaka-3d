"""Boundary checks for the new full-window mouth-recovery algorithm runner."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_runner():
    old = sys.path.copy()
    try:
        sys.path.insert(0, str(LAB))
        spec = importlib.util.spec_from_file_location(
            "_node_recovery_runner_test", LAB / "run_node_recovery.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old


runner = load_runner()


@pytest.mark.parametrize("bounds", [(333519, 333719), (76620, 77286), (76620, 86580)])
def test_only_full_registered_native_windows_are_admitted(bounds):
    manifest = {"experiment": runner.base.EXPERIMENT, "fps": 166, "frames": list(range(*bounds))}
    frames, reference = runner.full_registered_frames(manifest)
    np.testing.assert_array_equal(frames, np.arange(*bounds))
    assert reference == runner.FULL_WINDOWS[bounds]


@pytest.mark.parametrize(
    "frames",
    [
        list(range(76620, 76786)),
        list(range(333519, 333550)),
        list(range(76621, 77286)),
        [76620, *range(76622, 77286)],
    ],
)
def test_prefix_new_seed_and_dropped_native_frame_are_rejected(frames):
    with pytest.raises(ValueError):
        runner.full_registered_frames(
            {"experiment": runner.base.EXPERIMENT, "fps": 166, "frames": frames}
        )


def test_prefix_cli_option_is_unavailable_before_any_input_or_gpu_access():
    with pytest.raises(SystemExit) as caught:
        runner.main(["--sequence", "unused", "--output", "unused_out", "--max-frames", "3"])
    assert caught.value.code == 2


def arrays(count=3):
    result = {name: np.zeros(count) for name in runner.ORIGINAL_ARRAYS}
    result["hypotheses"] = np.array([18, 22, 24])
    return result


def test_extra_candidate_budget_preserves_all_original_arrays_without_truncation():
    data = arrays()
    saved = runner.validate_arm_arrays(data, np.arange(3))
    assert set(saved) == runner.ORIGINAL_ARRAYS
    assert len(saved) == 19
    for key in data:
        assert saved[key] is data[key]


@pytest.mark.parametrize("mutation", ["missing_original", "over_budget", "object", "short"])
def test_missing_or_unsafe_result_arrays_do_not_become_a_complete_candidate(mutation):
    data = arrays()
    if mutation == "missing_original":
        data.pop("proposal_flow_nodes")
    elif mutation == "over_budget":
        data["hypotheses"][1] = 25
    elif mutation == "object":
        data["proposal_kind"] = np.array([{}, {}, {}], dtype=object)
    else:
        data["body_xyz"] = np.zeros(2)
    with pytest.raises(ValueError):
        runner.validate_arm_arrays(data, np.arange(3))


def test_input_binding_cannot_hide_conflicting_relative_and_absolute_paths(tmp_path, monkeypatch):
    path = tmp_path / "input.txt"
    path.write_text("fixed")
    monkeypatch.setattr(runner.base, "REPO", tmp_path)
    with pytest.raises(ValueError, match="conflicting"):
        runner.normalized_hashes({"input.txt": "a", str(path): "b"})
