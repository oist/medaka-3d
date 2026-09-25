"""Saved age counters must follow selected native-frame evidence, not proposals."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from test_three_body_node_recovery_runner import runner as base_runner

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_runner():
    old = sys.modules.get("run_experiment")
    sys.modules["run_experiment"] = base_runner.base
    try:
        spec = importlib.util.spec_from_file_location("persistent_node_runner_test",
                                                     LAB / "run_persistent_node_recovery.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old is None:
            sys.modules.pop("run_experiment", None)
        else:
            sys.modules["run_experiment"] = old


RUNNER = load_runner()


def example():
    observed = np.ones((6, 3, 3), bool)
    observed[:4, 1, 0] = False
    observed[2:5, 2, 1] = False
    node = np.zeros((6, 3, 3), np.int64)
    pect = np.zeros((6, 3), np.int64)
    last_node, last_pect = np.zeros((3, 3), int), np.zeros(3, int)
    for t, value in enumerate(observed):
        last_node = np.where(value, 0, last_node + 1)
        last_pect = np.where(value[:, 1], last_pect + 1, 0)
        node[t], pect[t] = last_node, last_pect
    return dict(observed=observed, node_missing=node, direct_pectoral_age=pect)


def test_support_restoration_resets_only_the_corresponding_node_counter():
    values = example()
    RUNNER.validate_node_ages(values)
    np.testing.assert_array_equal(values["node_missing"][:, 1, 0], [1, 2, 3, 4, 0, 0])
    np.testing.assert_array_equal(values["direct_pectoral_age"][:, 2], [1, 2, 0, 0, 0, 1])


@pytest.mark.parametrize("kind", ["missing", "shape", "float", "wrong_parent", "mask_bridge"])
def test_incomplete_or_inconsistent_saved_ages_are_rejected(kind):
    values = example()
    if kind == "missing":
        del values["node_missing"]
    elif kind == "shape":
        values["node_missing"] = values["node_missing"][:, :, 0]
    elif kind == "float":
        values["node_missing"] = values["node_missing"].astype(float)
    elif kind == "wrong_parent":
        values["node_missing"][3, 1, 0] = 2
    else:
        values["direct_pectoral_age"][3, 2] = 4
    with pytest.raises(ValueError):
        RUNNER.validate_node_ages(values)
