"""The output distinguishes geometric fitting, color ranking and enrollment."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_three_body_persistent_node_runner import RUNNER as _persistent_runner  # noqa: F401

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
SPEC = importlib.util.spec_from_file_location("appearance_runner_test",
                                             LAB / "run_appearance_tracking.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
RANKING = SimpleNamespace(weight=1.0, maximum_samples=512, active_views=("top_cam",))
ENROLLMENT = SimpleNamespace(training_frames=3)


def example():
    color = np.array([0, 0, 0, 0.25, 0, 0.5])
    enabled = np.zeros((6, 3), bool)
    enabled[[3, 5], 0] = True
    pixels = np.where(enabled, 32, 0)
    geometry = np.arange(6, dtype=float) + 1
    return dict(frame_idx=np.arange(10, 16), appearance_cost=color, geometric_cost=geometry,
                appearance_enabled=enabled, appearance_pixels=pixels, costs=geometry + color)


def test_prefix_and_missing_view_stay_neutral_and_cost_components_reconcile():
    RUNNER.validate_appearance_arrays(example(), RANKING, ENROLLMENT)


@pytest.mark.parametrize("error", ["prefix", "side", "counts", "cap", "missing_cost",
                                   "total", "nonfinite", "inactive_cost"])
def test_saved_appearance_misrepresentations_are_rejected(error):
    values = example()
    if error == "prefix":
        values["appearance_enabled"][1, 0] = True
        values["appearance_pixels"][1, 0] = 32
    elif error == "side":
        values["appearance_enabled"][3, 1] = True
        values["appearance_pixels"][3, 1] = 32
    elif error == "counts":
        values["appearance_pixels"][3, 0] = 0
    elif error == "cap":
        values["appearance_pixels"][3, 0] = 513
    elif error == "missing_cost":
        del values["geometric_cost"]
    elif error == "total":
        values["costs"][3] += 1
    elif error == "nonfinite":
        values["appearance_cost"][3] = np.nan
    else:
        values["appearance_cost"][4] = 0.2
    with pytest.raises((ValueError, AssertionError)):
        RUNNER.validate_appearance_arrays(values, RANKING, ENROLLMENT)


def test_appearance_totals_are_saved_evidence_counts():
    arrays = example()
    RUNNER.validate_appearance_totals(arrays, np.array([64, 0, 0]), 2)
    with pytest.raises(AssertionError):
        RUNNER.validate_appearance_totals(arrays, np.array([65, 0, 0]), 2)
    with pytest.raises(ValueError, match="scored-frame"):
        RUNNER.validate_appearance_totals(arrays, np.array([64, 0, 0]), 3)


def audit_example():
    return dict(frame_idx=np.arange(10, 13), frozen=True,
                online_pose_xyz=np.zeros((3, 3, 3, 3), np.float64),
                online_sources=np.full((3, 3, 3, 3), -1, np.int64),
                online_observed=np.zeros((3, 3, 3), bool),
                online_identity_review=np.zeros((3, 3), bool))


def test_online_audit_covers_exact_enrollment_prefix_separately_from_final_track():
    RUNNER.validate_enrollment_audit(audit_example(), np.arange(10, 16), 3)


@pytest.mark.parametrize("error", ["frame", "frozen", "missing", "dtype", "nan", "source"])
def test_invalid_online_enrollment_audit_is_rejected(error):
    audit = audit_example()
    if error == "frame":
        audit["frame_idx"][2] = 13
    elif error == "frozen":
        audit["frozen"] = False
    elif error == "missing":
        del audit["online_identity_review"]
    elif error == "dtype":
        audit["online_observed"] = audit["online_observed"].astype(np.int64)
    elif error == "nan":
        audit["online_pose_xyz"][0, 0, 0, 0] = np.nan
    else:
        audit["online_sources"][0, 0, 0, 0] = -2
    with pytest.raises((ValueError, AssertionError)):
        RUNNER.validate_enrollment_audit(audit, np.arange(10, 16), 3)
